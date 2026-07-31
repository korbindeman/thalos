//! Whole-card NVIDIA telemetry for Windows crash forensics.
//!
//! Wgpu's allocator counters describe allocations owned by one process. They
//! cannot distinguish "the app used its budget" from temperature, board-power,
//! driver, or whole-card VRAM pressure, and the adapter can disappear before
//! Bevy gets a `DeviceLost` callback. NVML is the driver's own view of those
//! quantities. Load it dynamically so AMD-only systems keep working and so
//! diagnostics never make the NVIDIA driver a runtime dependency.
//!
//! Two readers share one NVML handle:
//!
//! - the **health sampler** ([`start`]) — the full one-second record, opt-in
//!   behind [`crate::GPU_HEALTH_ENV`] because it is investigation-grade volume;
//! - the **VRAM poller** ([`memory_snapshot`]) — memory used/total only, started
//!   on first read and cheap enough to be always available, so a live readout
//!   (the loading screen) can show whole-card pressure without turning the
//!   investigation lane on.

use std::{
    ffi::{CStr, c_char, c_void},
    sync::{
        Arc, Mutex, Once, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use libloading::Library;
use serde_json::{Map, Value};

use crate::DiagnosticSink;

const TARGET: &str = "thalos::diagnostic::gpu_health";
const SAMPLE_PERIOD: Duration = Duration::from_secs(1);
const NVML_SUCCESS: NvmlReturn = 0;
const NVML_TEMPERATURE_GPU: u32 = 0;
const NVML_CLOCK_GRAPHICS: u32 = 0;

type NvmlReturn = u32;
type NvmlDevice = *mut c_void;
type InitFn = unsafe extern "C" fn() -> NvmlReturn;
type ShutdownFn = unsafe extern "C" fn() -> NvmlReturn;
type ErrorStringFn = unsafe extern "C" fn(NvmlReturn) -> *const c_char;
type DriverVersionFn = unsafe extern "C" fn(*mut c_char, u32) -> NvmlReturn;
type DeviceByIndexFn = unsafe extern "C" fn(u32, *mut NvmlDevice) -> NvmlReturn;
type MemoryInfoFn = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> NvmlReturn;
type TemperatureFn = unsafe extern "C" fn(NvmlDevice, u32, *mut u32) -> NvmlReturn;
type PowerFn = unsafe extern "C" fn(NvmlDevice, *mut u32) -> NvmlReturn;
type UtilizationFn = unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> NvmlReturn;
type ClockFn = unsafe extern "C" fn(NvmlDevice, u32, *mut u32) -> NvmlReturn;
type PerformanceStateFn = unsafe extern "C" fn(NvmlDevice, *mut u32) -> NvmlReturn;
type ThrottleReasonsFn = unsafe extern "C" fn(NvmlDevice, *mut u64) -> NvmlReturn;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

#[derive(Clone, Copy, Debug)]
struct GpuSample {
    memory: NvmlMemory,
    temperature_c: Option<u32>,
    power_mw: Option<u32>,
    power_limit_mw: Option<u32>,
    utilization: Option<NvmlUtilization>,
    graphics_clock_mhz: Option<u32>,
    performance_state: Option<u32>,
    throttle_reasons: Option<u64>,
}

struct Nvml {
    // Function pointers remain valid only while the library is loaded.
    _library: Library,
    shutdown: ShutdownFn,
    error_string: ErrorStringFn,
    driver_version: DriverVersionFn,
    device: NvmlDevice,
    memory_info: MemoryInfoFn,
    temperature: TemperatureFn,
    power_usage: PowerFn,
    power_limit: PowerFn,
    utilization: UtilizationFn,
    clock_info: ClockFn,
    performance_state: PerformanceStateFn,
    throttle_reasons: ThrottleReasonsFn,
}

// NVML owns the opaque device handle and documents its query API as
// thread-safe. This value moves once into the dedicated sampler thread.
unsafe impl Send for Nvml {}

impl Nvml {
    unsafe fn load() -> Result<Self, String> {
        // Use the driver-installed system DLL explicitly. Searching the
        // process working directory for a diagnostic DLL would create an
        // avoidable DLL-preloading boundary in every game launch.
        let library_path = std::env::var_os("SystemRoot")
            .map(std::path::PathBuf::from)
            .map(|root| root.join("System32").join("nvml.dll"))
            .unwrap_or_else(|| std::path::PathBuf::from(r"C:\Windows\System32\nvml.dll"));
        let library = unsafe { Library::new(&library_path) }
            .map_err(|error| format!("nvml.dll unavailable: {error}"))?;

        macro_rules! symbol {
            ($name:literal, $ty:ty) => {{
                let value = unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .map_err(|error| format!("NVML symbol {} unavailable: {error}", $name))?;
                *value
            }};
        }

        let init = symbol!("nvmlInit_v2", InitFn);
        let shutdown = symbol!("nvmlShutdown", ShutdownFn);
        let error_string = symbol!("nvmlErrorString", ErrorStringFn);
        let driver_version = symbol!("nvmlSystemGetDriverVersion", DriverVersionFn);
        let device_by_index = symbol!("nvmlDeviceGetHandleByIndex_v2", DeviceByIndexFn);
        let memory_info = symbol!("nvmlDeviceGetMemoryInfo", MemoryInfoFn);
        let temperature = symbol!("nvmlDeviceGetTemperature", TemperatureFn);
        let power_usage = symbol!("nvmlDeviceGetPowerUsage", PowerFn);
        let power_limit = symbol!("nvmlDeviceGetEnforcedPowerLimit", PowerFn);
        let utilization = symbol!("nvmlDeviceGetUtilizationRates", UtilizationFn);
        let clock_info = symbol!("nvmlDeviceGetClockInfo", ClockFn);
        let performance_state = symbol!("nvmlDeviceGetPerformanceState", PerformanceStateFn);
        let throttle_reasons = symbol!(
            "nvmlDeviceGetCurrentClocksThrottleReasons",
            ThrottleReasonsFn
        );

        let init_result = unsafe { init() };
        if init_result != NVML_SUCCESS {
            return Err(error_message(error_string, init_result));
        }

        let mut device = std::ptr::null_mut();
        let device_result = unsafe { device_by_index(0, &mut device) };
        if device_result != NVML_SUCCESS {
            unsafe {
                shutdown();
            }
            return Err(error_message(error_string, device_result));
        }

        Ok(Self {
            _library: library,
            shutdown,
            error_string,
            driver_version,
            device,
            memory_info,
            temperature,
            power_usage,
            power_limit,
            utilization,
            clock_info,
            performance_state,
            throttle_reasons,
        })
    }

    fn driver_version(&self) -> Option<String> {
        let mut buffer = [0_i8; 96];
        let result = unsafe { (self.driver_version)(buffer.as_mut_ptr(), buffer.len() as u32) };
        if result != NVML_SUCCESS {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(buffer.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    fn memory(&self) -> Result<NvmlMemory, NvmlReturn> {
        let mut memory = NvmlMemory::default();
        let result = unsafe { (self.memory_info)(self.device, &mut memory) };
        if result != NVML_SUCCESS {
            return Err(result);
        }
        Ok(memory)
    }

    fn sample(&self) -> Result<GpuSample, NvmlReturn> {
        Ok(GpuSample {
            memory: self.memory()?,
            temperature_c: query_u32(|out| unsafe {
                (self.temperature)(self.device, NVML_TEMPERATURE_GPU, out)
            }),
            power_mw: query_u32(|out| unsafe { (self.power_usage)(self.device, out) }),
            power_limit_mw: query_u32(|out| unsafe { (self.power_limit)(self.device, out) }),
            utilization: {
                let mut value = NvmlUtilization::default();
                (unsafe { (self.utilization)(self.device, &mut value) } == NVML_SUCCESS)
                    .then_some(value)
            },
            graphics_clock_mhz: query_u32(|out| unsafe {
                (self.clock_info)(self.device, NVML_CLOCK_GRAPHICS, out)
            }),
            performance_state: query_u32(|out| unsafe {
                (self.performance_state)(self.device, out)
            }),
            throttle_reasons: {
                let mut value = 0_u64;
                (unsafe { (self.throttle_reasons)(self.device, &mut value) } == NVML_SUCCESS)
                    .then_some(value)
            },
        })
    }

    fn error(&self, code: NvmlReturn) -> String {
        error_message(self.error_string, code)
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        unsafe {
            (self.shutdown)();
        }
    }
}

fn query_u32(query: impl FnOnce(*mut u32) -> NvmlReturn) -> Option<u32> {
    let mut value = 0_u32;
    (query(&mut value) == NVML_SUCCESS).then_some(value)
}

fn error_message(error_string: ErrorStringFn, code: NvmlReturn) -> String {
    let pointer = unsafe { error_string(code) };
    if pointer.is_null() {
        return format!("NVML error {code}");
    }
    format!(
        "NVML error {code}: {}",
        unsafe { CStr::from_ptr(pointer) }.to_string_lossy()
    )
}

/// Process-wide NVML handle, loaded on first use by whichever reader asks
/// first. One DLL load and one `nvmlInit` however many readers there are.
static NVML: OnceLock<Result<Mutex<Nvml>, String>> = OnceLock::new();

fn nvml() -> Result<&'static Mutex<Nvml>, &'static str> {
    match NVML.get_or_init(|| unsafe { Nvml::load() }.map(Mutex::new)) {
        Ok(handle) => Ok(handle),
        Err(error) => Err(error.as_str()),
    }
}

pub(crate) fn start(sink: Arc<DiagnosticSink>) {
    let _ = thread::Builder::new()
        .name("thalos-gpu-health".into())
        .spawn(move || sampler_main(&sink));
}

fn sampler_main(sink: &DiagnosticSink) {
    let nvml = match nvml() {
        Ok(nvml) => nvml,
        Err(error) => {
            let mut fields = fields("availability");
            fields.insert("available".into(), Value::from(false));
            fields.insert("error".into(), Value::from(error));
            sink.write_event(TARGET, "INFO", fields);
            return;
        }
    };

    let mut availability = fields("availability");
    availability.insert("available".into(), Value::from(true));
    if let Some(version) = nvml.lock().ok().and_then(|nvml| nvml.driver_version()) {
        availability.insert("driver_version".into(), Value::from(version));
    }
    sink.write_event(TARGET, "INFO", availability);

    loop {
        let Ok(nvml) = nvml.lock() else {
            return; // a reader panicked mid-query; the handle is no longer trustworthy
        };
        let sampled = nvml.sample().map_err(|code| (code, nvml.error(code)));
        drop(nvml);
        match sampled {
            Ok(sample) => sink.write_event(TARGET, "INFO", sample_fields(sample)),
            Err((code, message)) => {
                // Stop after the first whole-card query failure. Repeated NVML
                // calls cannot recover a lost adapter and would obscure the
                // first-failure timestamp or add pressure to a wedged driver.
                let mut failure = fields("sample_error");
                failure.insert("nvml_error_code".into(), Value::from(code));
                failure.insert("error".into(), Value::from(message));
                sink.write_event(TARGET, "ERROR", failure);
                return;
            }
        }
        thread::sleep(SAMPLE_PERIOD);
    }
}

/// How often the VRAM poller refreshes. Whole-card usage moves on the scale of
/// a texture upload, so twice a second is live enough for a readout and far
/// below the rate at which the driver query would cost anything.
const MEMORY_POLL_PERIOD: Duration = Duration::from_millis(500);

/// Latest whole-card VRAM, published by [`memory_poller_main`]. `TOTAL == 0`
/// means "no reading" — not yet sampled, no NVIDIA driver, or the card stopped
/// answering — so readers never have to consult a second flag.
static VRAM_USED_BYTES: AtomicU64 = AtomicU64::new(0);
static VRAM_TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);

/// Whole-card VRAM as the driver reports it, or `None` when it cannot be read.
///
/// Never blocks and never queries the driver on the caller's thread: the first
/// call starts a background poller and returns `None`, and every later call is
/// two relaxed atomic loads. A UI may therefore call this every frame.
pub(crate) fn memory_snapshot() -> Option<crate::GpuMemory> {
    static POLLER: Once = Once::new();
    POLLER.call_once(|| {
        let _ = thread::Builder::new()
            .name("thalos-vram".into())
            .spawn(memory_poller_main);
    });

    let total_bytes = VRAM_TOTAL_BYTES.load(Ordering::Relaxed);
    (total_bytes > 0).then(|| crate::GpuMemory {
        used_bytes: VRAM_USED_BYTES.load(Ordering::Relaxed),
        total_bytes,
    })
}

fn memory_poller_main() {
    let Ok(nvml) = nvml() else {
        return; // no NVIDIA driver here; readers keep seeing `None`
    };
    loop {
        let Ok(guard) = nvml.lock() else {
            return;
        };
        let sampled = guard.memory();
        drop(guard);
        match sampled {
            Ok(memory) => {
                // Used before total: a reader that sees a non-zero total is
                // then guaranteed a numerator from this same sample or newer.
                VRAM_USED_BYTES.store(memory.used, Ordering::Relaxed);
                VRAM_TOTAL_BYTES.store(memory.total, Ordering::Relaxed);
            }
            Err(_) => {
                // Same first-failure rule as the health sampler: a lost adapter
                // does not come back, and a wedged driver should not be poked.
                // Zeroing the total is what turns the readout back to "no data"
                // rather than freezing the last good number on screen.
                VRAM_TOTAL_BYTES.store(0, Ordering::Relaxed);
                return;
            }
        }
        thread::sleep(MEMORY_POLL_PERIOD);
    }
}

fn fields(event: &'static str) -> Map<String, Value> {
    let mut fields = Map::new();
    fields.insert("event".into(), Value::from(event));
    fields.insert("message".into(), Value::from("NVIDIA whole-card health"));
    fields
}

fn sample_fields(sample: GpuSample) -> Map<String, Value> {
    let mut result = fields("sample");
    let mib = 1024.0 * 1024.0;
    let used_mib = sample.memory.used as f64 / mib;
    let total_mib = sample.memory.total as f64 / mib;
    result.insert("memory_used_mib".into(), Value::from(used_mib));
    result.insert("memory_total_mib".into(), Value::from(total_mib));
    result.insert(
        "memory_used_frac".into(),
        Value::from(if total_mib > 0.0 {
            used_mib / total_mib
        } else {
            0.0
        }),
    );
    if let Some(value) = sample.temperature_c {
        result.insert("temperature_c".into(), Value::from(value));
    }
    if let Some(value) = sample.power_mw {
        result.insert("power_w".into(), Value::from(value as f64 / 1000.0));
    }
    if let Some(value) = sample.power_limit_mw {
        let limit_w = value as f64 / 1000.0;
        result.insert("power_limit_w".into(), Value::from(limit_w));
        if let Some(power_mw) = sample.power_mw {
            result.insert(
                "power_frac".into(),
                Value::from(if limit_w > 0.0 {
                    power_mw as f64 / 1000.0 / limit_w
                } else {
                    0.0
                }),
            );
        }
    }
    if let Some(value) = sample.utilization {
        result.insert(
            "gpu_util_frac".into(),
            Value::from(value.gpu as f64 / 100.0),
        );
        result.insert(
            "memory_util_frac".into(),
            Value::from(value.memory as f64 / 100.0),
        );
    }
    if let Some(value) = sample.graphics_clock_mhz {
        result.insert("graphics_clock_mhz".into(), Value::from(value));
    }
    if let Some(value) = sample.performance_state {
        result.insert("performance_state".into(), Value::from(value));
    }
    if let Some(value) = sample.throttle_reasons {
        result.insert("clock_throttle_reasons".into(), Value::from(value));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_carries_each_denominator_and_uses_declared_units() {
        let fields = sample_fields(GpuSample {
            memory: NvmlMemory {
                total: 12 * 1024 * 1024,
                free: 9 * 1024 * 1024,
                used: 3 * 1024 * 1024,
            },
            temperature_c: Some(72),
            power_mw: Some(142_500),
            power_limit_mw: Some(285_000),
            utilization: Some(NvmlUtilization {
                gpu: 80,
                memory: 40,
            }),
            graphics_clock_mhz: Some(2_700),
            performance_state: Some(0),
            throttle_reasons: Some(4),
        });

        assert_eq!(fields["event"], "sample");
        assert_eq!(fields["memory_used_mib"], 3.0);
        assert_eq!(fields["memory_total_mib"], 12.0);
        assert_eq!(fields["memory_used_frac"], 0.25);
        assert_eq!(fields["temperature_c"], 72);
        assert_eq!(fields["power_w"], 142.5);
        assert_eq!(fields["power_limit_w"], 285.0);
        assert_eq!(fields["power_frac"], 0.5);
        assert_eq!(fields["gpu_util_frac"], 0.8);
        assert_eq!(fields["memory_util_frac"], 0.4);
    }
}
