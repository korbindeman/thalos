use std::{
    collections::BTreeMap,
    env, fs,
    fs::OpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use thalos_capture_protocol::{
    CAPTURE_PRESETS, CAPTURE_PROTOCOL_SCHEMA, CaptureAction, CaptureCameraOverride,
    CaptureGraphicsOverrides, CaptureGraphicsSettings, CaptureRequest, CaptureResponse,
    CaptureServerState, CaptureSourceSnapshot, CapturedCameraState, ViewpointCatalog,
};
use uuid::Uuid;

mod compare;
mod gallery;
mod runlog;

const REQUEST_FILE: &str = "visual_capture_request.json";
const RESPONSE_FILE: &str = "visual_capture_response.json";
const STATE_FILE: &str = "visual_capture_server.json";
const LAUNCHER_FILE: &str = "visual_capture_launcher.json";
const LOG_FILE: &str = "visual_capture_server.log";
const CLIENT_LOCK_FILE: &str = "visual_capture_client.lock";
const RESOURCE_FAULT_FILE: &str = "visual_capture_resource_fault.json";
const RESOURCE_EVENT_FILE: &str = "visual_capture_resource_events.jsonl";
#[cfg(windows)]
const CAPTURE_MUTEX_NAME: &str = "Local\\ThalosMachineCaptureV1";

/// Until typed fidelity profiles land, catalog viewpoints render their sensor
/// aspect at this safe standard extent. Output pixels are not viewpoint state.
const SAFE_VIEWPOINT_WIDTH: u32 = 1920;
const SAFE_VIEWPOINT_HEIGHT: u32 = 1080;
/// The game renderer may use the whole machine-wide 4 GiB tile allowance.
/// Headless evidence should leave room for the desktop and other agents.
const DEFAULT_CAPTURE_TILE_BUDGET_MB: &str = "2048";
/// Last-resort workstation protection. A healthy capture host stays well below
/// this; the observed device-loss retry grew to 12.6 GiB in seconds.
const DEFAULT_CAPTURE_RSS_LIMIT_MB: u64 = 8 * 1024;
/// Prevent waiting agents from stampeding a renderer/device that just failed.
const RESOURCE_FAULT_COOLDOWN_SECS: u64 = 5 * 60;

const OVERRIDE_KEYS: &[&str] = &[
    "THALOS_SCREENSHOT_OUT",
    "THALOS_SCREENSHOT_REPORT",
    "THALOS_SCREENSHOT_SIZE",
    "THALOS_SCREENSHOT_AZIMUTH",
    "THALOS_SCREENSHOT_ELEVATION",
    "THALOS_SCREENSHOT_DISTANCE",
    "THALOS_SCREENSHOT_TIME",
    "THALOS_SCREENSHOT_WARMUP",
    "THALOS_SCREENSHOT_HUD",
    "THALOS_SCREENSHOT_CAMERA_ALTITUDE",
    "THALOS_SCREENSHOT_LOOK_ELEVATION",
    "THALOS_SCREENSHOT_SUN_ELEVATION",
    "THALOS_SCREENSHOT_CLOUD_QUALITY",
    "THALOS_SCREENSHOT_CLOUD_TEMPORAL",
    "THALOS_SCREENSHOT_CLOUD_COVERAGE",
    "THALOS_SCREENSHOT_CLOUD_RECONSTRUCTION",
    "THALOS_SCREENSHOT_CLOUD_DENSITY_COUPLING",
    "THALOS_SCREENSHOT_CLOUD_TIER",
    "THALOS_SCREENSHOT_CLOUD_FAR_FILTER",
    "THALOS_SCREENSHOT_CLOUD_FAR_AGGREGATION",
    "THALOS_SCREENSHOT_OCEAN_TIME",
    "THALOS_PLUME_PRESSURE",
    "THALOS_CONTACT_SHADOW",
    "THALOS_CLOUD_SHADOW",
    "THALOS_CLOUD_GODRAY",
    "THALOS_SSAO",
    "THALOS_TERRAIN_INSPECTION",
    "THALOS_TERRAIN_CULL",
    "THALOS_TERRAIN",
    "THALOS_TILE_RENDERER",
    "THALOS_TILE_CACHE",
    "THALOS_TILE_BUDGET_MB",
    "THALOS_CAPTURE_RSS_LIMIT_MB",
    "THALOS_RUNWAY_SITE",
    "THALOS_WGPU_BACKEND",
];

/// Request inputs that shape boot-time world or renderer construction rather
/// than a live capture resource. A change in this subset requires a restart.
const STARTUP_OVERRIDE_KEYS: &[&str] = &[
    "THALOS_SCREENSHOT_SIZE",
    "THALOS_TERRAIN_CULL",
    "THALOS_TERRAIN",
    "THALOS_TILE_RENDERER",
    "THALOS_TILE_CACHE",
    "THALOS_TILE_BUDGET_MB",
    "THALOS_CAPTURE_RSS_LIMIT_MB",
    "THALOS_RUNWAY_SITE",
    "THALOS_WGPU_BACKEND",
];

#[derive(Debug, Deserialize, Serialize)]
struct LauncherState {
    schema_version: u32,
    pid: u32,
    preset: String,
    launched_unix_ms: u128,
    #[serde(default)]
    log_start_bytes: u64,
    #[serde(default)]
    startup_overrides: BTreeMap<String, String>,
    #[serde(default)]
    source: CaptureSourceSnapshot,
    command: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CaptureReceipt<'a> {
    schema: &'static str,
    request_id: &'a str,
    preset: &'a str,
    output: String,
    completed_unix_ms: u128,
    renderer_pid: u32,
    source: &'a CaptureSourceSnapshot,
    renderer_launch_source: &'a CaptureSourceSnapshot,
    workspace_after: &'a CaptureSourceSnapshot,
    source_floor_guaranteed: bool,
    workspace_relation: &'static str,
    camera: Option<CapturedCameraState>,
    graphics: CaptureGraphicsSettings,
    /// Compatibility field for existing receipt readers. `false` now means
    /// the workspace advanced after the source floor; it is not a capture
    /// failure and must not trigger a rebuild loop.
    workspace_matches: bool,
}

#[derive(Debug, Deserialize, Serialize)]
struct CaptureClientLockState {
    pid: u32,
    token: String,
    started_unix_ms: u128,
    command: String,
}

#[derive(Debug)]
struct CaptureClientLock {
    path: PathBuf,
    token: String,
    #[cfg(windows)]
    mutex: Option<windows_sys::Win32::Foundation::HANDLE>,
}

#[derive(Debug, Deserialize, Serialize)]
struct CaptureResourceFault {
    observed_unix_ms: u128,
    #[serde(default)]
    observed_uptime_ms: Option<u64>,
    kind: String,
    detail: String,
}

#[derive(Debug, Serialize)]
struct CaptureResourceEvent<'a> {
    schema: &'static str,
    event: &'a str,
    recorded_unix_ms: u128,
    recorded_uptime_ms: Option<u64>,
    client_pid: u32,
    command: String,
    workspace: String,
    fault_observed_unix_ms: u128,
    fault_observed_uptime_ms: Option<u64>,
    fault_age_ms: u128,
    kind: &'a str,
    policy: &'static str,
    cooldown_secs: Option<u64>,
    note: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    fault_detail: Option<&'a str>,
}

impl CaptureClientLock {
    /// Acquire the machine-wide capture lock, recording what the wait cost.
    ///
    /// Contention is process-level cost paid before any scene work starts, so
    /// it is its own record rather than a phase of whichever shot happened to
    /// be first — otherwise one agent's queueing reads as another agent's slow
    /// capture.
    fn acquire(command: String) -> Result<Self, String> {
        let wait = env::var("THALOS_CAPTURE_CLIENT_WAIT_SECS")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(1800));
        let started = Instant::now();
        let mut queued_behind = None;
        #[cfg(windows)]
        let acquired = Self::acquire_machine_mutex(command, wait, &mut queued_behind);
        #[cfg(not(windows))]
        let acquired = Self::acquire_at(
            machine_capture_path(CLIENT_LOCK_FILE),
            command,
            wait,
            &mut queued_behind,
        );
        runlog::record_lock_wait(
            started.elapsed(),
            queued_behind,
            if acquired.is_ok() {
                "acquired"
            } else {
                "failed"
            },
        );
        acquired
    }

    #[cfg(windows)]
    fn acquire_machine_mutex(
        command: String,
        wait: Duration,
        queued_behind: &mut Option<u32>,
    ) -> Result<Self, String> {
        Self::acquire_named_mutex(
            machine_capture_path(CLIENT_LOCK_FILE),
            CAPTURE_MUTEX_NAME,
            command,
            wait,
            queued_behind,
        )
    }

    #[cfg(windows)]
    fn acquire_named_mutex(
        path: PathBuf,
        mutex_name: &str,
        command: String,
        wait: Duration,
        queued_behind: &mut Option<u32>,
    ) -> Result<Self, String> {
        use windows_sys::Win32::{
            Foundation::{CloseHandle, WAIT_ABANDONED, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT},
            System::Threading::{CreateMutexW, ReleaseMutex, WaitForSingleObject},
        };

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        let name = mutex_name
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let mutex = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
        if mutex.is_null() {
            return Err(format!(
                "create machine capture mutex: {}",
                std::io::Error::last_os_error()
            ));
        }

        let started = Instant::now();
        let token = Uuid::new_v4().simple().to_string();
        let mut announced = false;
        loop {
            let remaining = wait.saturating_sub(started.elapsed());
            let poll_ms = remaining.min(Duration::from_millis(250)).as_millis() as u32;
            let result = unsafe { WaitForSingleObject(mutex, poll_ms) };
            match result {
                WAIT_OBJECT_0 | WAIT_ABANDONED => {
                    let state = CaptureClientLockState {
                        pid: std::process::id(),
                        token: token.clone(),
                        started_unix_ms: timestamp_millis(),
                        command,
                    };
                    if let Err(error) = write_json_atomic(&path, &state) {
                        unsafe {
                            ReleaseMutex(mutex);
                            CloseHandle(mutex);
                        }
                        return Err(format!(
                            "publish machine capture owner {}: {error}",
                            path.display()
                        ));
                    }
                    return Ok(Self {
                        path,
                        token,
                        mutex: Some(mutex),
                    });
                }
                WAIT_TIMEOUT => {
                    if started.elapsed() >= wait {
                        let owner = read_json::<CaptureClientLockState>(&path);
                        unsafe {
                            CloseHandle(mutex);
                        }
                        return Err(match owner {
                            Some(owner) => format!(
                                "machine capture lock timed out after {:.1}s; pid {} is running {:?}",
                                started.elapsed().as_secs_f64(),
                                owner.pid,
                                owner.command
                            ),
                            None => format!(
                                "machine capture lock timed out after {:.1}s; owner metadata is unavailable",
                                started.elapsed().as_secs_f64()
                            ),
                        });
                    }
                    if !announced {
                        match read_json::<CaptureClientLockState>(&path) {
                            Some(owner) => {
                                *queued_behind = Some(owner.pid);
                                println!(
                                    "capture queued behind pid {} ({}); one machine-wide capture operation is allowed",
                                    owner.pid, owner.command
                                )
                            }
                            None => {
                                println!("another machine-wide capture client is starting; waiting")
                            }
                        }
                        announced = true;
                    }
                }
                WAIT_FAILED => {
                    let error = std::io::Error::last_os_error();
                    unsafe {
                        CloseHandle(mutex);
                    }
                    return Err(format!("wait for machine capture mutex: {error}"));
                }
                other => {
                    unsafe {
                        CloseHandle(mutex);
                    }
                    return Err(format!(
                        "wait for machine capture mutex returned unexpected status {other:#x}"
                    ));
                }
            }
        }
    }

    #[cfg(any(not(windows), test))]
    fn acquire_at(
        path: PathBuf,
        command: String,
        wait: Duration,
        queued_behind: &mut Option<u32>,
    ) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        let started = Instant::now();
        let token = Uuid::new_v4().simple().to_string();
        let mut announced = false;
        loop {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let state = CaptureClientLockState {
                        pid: std::process::id(),
                        token: token.clone(),
                        started_unix_ms: timestamp_millis(),
                        command,
                    };
                    let bytes = serde_json::to_vec_pretty(&state)
                        .map_err(|error| format!("serialize capture client lock: {error}"))?;
                    if let Err(error) = file.write_all(&bytes).and_then(|_| file.sync_all()) {
                        let _ = fs::remove_file(&path);
                        return Err(format!(
                            "write capture client lock {}: {error}",
                            path.display()
                        ));
                    }
                    return Ok(Self {
                        path,
                        token,
                        #[cfg(windows)]
                        mutex: None,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let owner = read_json::<CaptureClientLockState>(&path);
                    let corrupt_stale = owner.is_none()
                        && fs::metadata(&path)
                            .ok()
                            .and_then(|metadata| metadata.modified().ok())
                            .and_then(|modified| modified.elapsed().ok())
                            .is_some_and(|age| age >= Duration::from_secs(5));
                    let dead_owner = owner
                        .as_ref()
                        .is_some_and(|owner| !process_alive(owner.pid));
                    if corrupt_stale || dead_owner {
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if started.elapsed() >= wait {
                        return Err(match owner {
                            Some(owner) => format!(
                                "capture client lock timed out after {:.1}s; pid {} is running {:?}",
                                started.elapsed().as_secs_f64(),
                                owner.pid,
                                owner.command
                            ),
                            None => format!(
                                "capture client lock {} is incomplete; retry shortly",
                                path.display()
                            ),
                        });
                    }
                    if !announced {
                        match owner {
                            Some(owner) => {
                                *queued_behind = Some(owner.pid);
                                println!(
                                    "capture queued behind pid {} ({}); compatible cameras reuse one renderer sequentially",
                                    owner.pid, owner.command
                                )
                            }
                            None => println!("another capture client is starting; waiting"),
                        }
                        announced = true;
                    }
                    thread::sleep(Duration::from_millis(250));
                }
                Err(error) => {
                    return Err(format!(
                        "acquire capture client lock {}: {error}",
                        path.display()
                    ));
                }
            }
        }
    }
}

impl Drop for CaptureClientLock {
    fn drop(&mut self) {
        if read_json::<CaptureClientLockState>(&self.path)
            .is_some_and(|owner| owner.token == self.token)
        {
            let _ = fs::remove_file(&self.path);
        }
        #[cfg(windows)]
        if let Some(mutex) = self.mutex.take() {
            use windows_sys::Win32::{Foundation::CloseHandle, System::Threading::ReleaseMutex};
            unsafe {
                ReleaseMutex(mutex);
                CloseHandle(mutex);
            }
        }
    }
}

fn main() -> ExitCode {
    runlog::install();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("visual capture failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let command_line = env::args().skip(1).collect::<Vec<_>>();
    let mut args = command_line.clone().into_iter();
    let command = args.next().unwrap_or_else(|| "shot".to_owned());
    let _client_lock = matches!(command.as_str(), "shot" | "capture" | "compare" | "reset")
        .then(|| CaptureClientLock::acquire(command_line.join(" ")))
        .transpose()?;
    match command.as_str() {
        "shot" | "capture" => {
            let ShotArgs {
                presets,
                output,
                report,
                assignments,
                camera,
                graphics,
            } = parse_capture_options(args)?;
            if presets.len() > 1 && (output.is_some() || report.is_some()) {
                return Err("--out and --report require exactly one preset".into());
            }
            if presets.len() == 1 {
                capture(&presets[0], output, report, assignments, camera, graphics)?;
            } else {
                capture_batch(presets, assignments, camera, graphics)?;
            }
        }
        "compare" => compare::run_cli(args)?,
        "list" => gallery::run_cli(args)?,
        "status" => status(),
        "stop" => stop_server(false)?,
        "reset" => reset_build_state()?,
        "-h" | "--help" | "help" => print_help(),
        other => return Err(format!("unknown command {other:?}; use --help")),
    }
    Ok(())
}

/// Capture every requested scene while minimizing full host boots. The host is
/// the authority on compatibility: after the first scene in a boot context,
/// pull every remaining compatible scene forward before crossing into the next
/// context.
fn capture_batch(
    mut pending: Vec<String>,
    assignments: Vec<String>,
    camera: CaptureCameraOverride,
    graphics: CaptureGraphicsOverrides,
) -> Result<(), String> {
    let total = pending.len();
    let mut completed = 0;
    while !pending.is_empty() {
        let first = pending.remove(0);
        completed += 1;
        println!("[{completed}/{total}] scene={first}");
        capture(&first, None, None, assignments.clone(), camera, graphics)?;

        let compatible = read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE))
            .map(|state| state.compatible_presets)
            .unwrap_or_default();
        let (same_context, later) = pending
            .into_iter()
            .partition(|preset| compatible.contains(preset));
        pending = later;
        for preset in same_context {
            completed += 1;
            println!("[{completed}/{total}] scene={preset} (reusing world)");
            capture(&preset, None, None, assignments.clone(), camera, graphics)?;
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq)]
struct ShotArgs {
    presets: Vec<String>,
    output: Option<PathBuf>,
    report: Option<PathBuf>,
    assignments: Vec<String>,
    camera: CaptureCameraOverride,
    graphics: CaptureGraphicsOverrides,
}

fn parse_capture_options(mut args: impl Iterator<Item = String>) -> Result<ShotArgs, String> {
    let mut presets = Vec::new();
    let mut output = None;
    let mut report = None;
    let mut assignments = Vec::new();
    let mut camera = CaptureCameraOverride::default();
    let mut graphics = env::var("THALOS_SCREENSHOT_GRAPHICS")
        .ok()
        .map(|raw| CaptureGraphicsOverrides::parse(&raw))
        .transpose()?
        .unwrap_or_default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => output = Some(PathBuf::from(args.next().ok_or("--out requires a path")?)),
            "--report" => {
                report = Some(PathBuf::from(
                    args.next().ok_or("--report requires a path")?,
                ))
            }
            "--time" => {
                let value = args.next().ok_or("--time requires canonical seconds")?;
                let parsed = value
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| format!("--time expects canonical seconds, got {value:?}"))?;
                if !parsed.is_finite() {
                    return Err(format!("--time expects a finite value, got {value:?}"));
                }
                assignments.push(format!("THALOS_SCREENSHOT_TIME={value}"));
            }
            "--size" => {
                let value = args.next().ok_or("--size requires WIDTHxHEIGHT")?;
                parse_capture_size(&value)?;
                assignments.push(format!("THALOS_SCREENSHOT_SIZE={value}"));
            }
            "--focal-length" | "--lens" => {
                let value = args.next().ok_or("--focal-length requires millimetres")?;
                let focal_length_mm = value
                    .trim()
                    .parse::<f32>()
                    .map_err(|_| format!("--focal-length expects millimetres, got {value:?}"))?;
                camera.focal_length_mm = Some(focal_length_mm);
                camera.validate()?;
            }
            "--graphics" => {
                graphics = CaptureGraphicsOverrides::parse(
                    &args
                        .next()
                        .ok_or("--graphics requires NAME=VALUE[,NAME=VALUE...]")?,
                )?;
            }
            "--set" => assignments.push(args.next().ok_or("--set requires KEY=VALUE")?),
            option if option.starts_with('-') => {
                return Err(format!("unknown shot option {option:?}"));
            }
            preset => presets.push(canonical_preset(preset)?),
        }
    }
    if presets.is_empty() {
        presets.push("spaceport-aerial".to_owned());
    }
    Ok(ShotArgs {
        presets,
        output,
        report,
        assignments,
        camera,
        graphics,
    })
}

fn parse_capture_size(raw: &str) -> Result<(u32, u32), String> {
    let Some((width, height)) = raw.split_once(['x', 'X', '*']) else {
        return Err(format!("--size expects WIDTHxHEIGHT, got {raw:?}"));
    };
    let width = width
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("--size expects WIDTHxHEIGHT, got {raw:?}"))?;
    let height = height
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("--size expects WIDTHxHEIGHT, got {raw:?}"))?;
    if width == 0 || height == 0 {
        return Err(format!("--size dimensions must be non-zero, got {raw:?}"));
    }
    Ok((width, height))
}

pub(crate) fn capture(
    preset: &str,
    output: Option<PathBuf>,
    report: Option<PathBuf>,
    assignments: Vec<String>,
    camera: CaptureCameraOverride,
    graphics: CaptureGraphicsOverrides,
) -> Result<PathBuf, String> {
    let preset = canonical_preset(preset)?;
    guard_resource_fault()?;
    let mut overrides = OVERRIDE_KEYS
        .iter()
        .filter_map(|key| env::var(key).ok().map(|value| ((*key).to_owned(), value)))
        .collect::<BTreeMap<_, _>>();
    for assignment in assignments {
        let (key, value) = assignment
            .split_once('=')
            .ok_or_else(|| format!("--set expects KEY=VALUE, got {assignment:?}"))?;
        if !OVERRIDE_KEYS.contains(&key) {
            return Err(format!("unsupported capture override {key:?}"));
        }
        overrides.insert(key.to_owned(), value.to_owned());
    }
    apply_safe_viewpoint_size(&preset, &mut overrides);
    validate_viewpoint_output_aspect(&preset, &overrides)?;
    overrides
        .entry("THALOS_TILE_BUDGET_MB".into())
        .or_insert_with(|| DEFAULT_CAPTURE_TILE_BUDGET_MB.into());
    if let Some(path) = output {
        overrides.insert(
            "THALOS_SCREENSHOT_OUT".into(),
            absolute(path).display().to_string(),
        );
    }
    if let Some(path) = report {
        overrides.insert(
            "THALOS_SCREENSHOT_REPORT".into(),
            absolute(path).display().to_string(),
        );
    }

    runlog::begin(format!("shot {preset}"));
    runlog::field("preset", preset.clone());
    let result = capture_instrumented(&preset, &overrides, camera, graphics);
    runlog::finish(&result);
    result
}

/// The shot itself, wrapped so every exit path closes the run record.
fn capture_instrumented(
    preset: &str,
    overrides: &BTreeMap<String, String>,
    camera: CaptureCameraOverride,
    graphics: CaptureGraphicsOverrides,
) -> Result<PathBuf, String> {
    let (mut state, mut source) = prepare_server(preset, overrides)?;
    runlog::field("renderer_pid", state.pid);
    runlog::field("source", short_fingerprint(&source.fingerprint).to_owned());
    match capture_once(preset, overrides, &state, &source, camera, graphics) {
        Ok(path) => {
            clear_resource_fault();
            runlog::field("output", path.display().to_string());
            Ok(path)
        }
        Err(error) if error.recoverable => {
            eprintln!(
                "capture host became unhealthy ({}); restarting once and retrying scene={preset}",
                error.message.lines().next().unwrap_or("unknown error")
            );
            // A recovered shot still costs an agent a full host boot, so the
            // retry is counted rather than hidden by the eventual success.
            runlog::count("retry");
            runlog::field(
                "retry_reason",
                error.message.lines().next().unwrap_or("unknown").to_owned(),
            );
            stop_server(true)?;
            (state, source) = prepare_server(preset, overrides)?;
            runlog::field("renderer_pid", state.pid);
            capture_once(preset, overrides, &state, &source, camera, graphics)
                .inspect(|path| {
                    clear_resource_fault();
                    runlog::field("output", path.display().to_string());
                })
                .map_err(|error| error.message)
        }
        Err(error) => {
            let _ = stop_server(true);
            Err(error.message)
        }
    }
}

fn prepare_server(
    preset: &str,
    overrides: &BTreeMap<String, String>,
) -> Result<(CaptureServerState, CaptureSourceSnapshot), String> {
    // This snapshot is a causal floor, not a demand that a shared checkout
    // remain byte-identical for the whole build. The renderer may consume this
    // state or edits made after it; chasing every newer aggregate fingerprint
    // can starve forever while parallel agents work.
    let source = capture_source_snapshot()?;
    let state = ensure_server(preset, overrides, &source)?;
    let current = capture_source_snapshot()?;
    if current.fingerprint != source.fingerprint {
        eprintln!(
            "workspace advanced while preparing the renderer (source floor {}, current {}); continuing without rebuilding again",
            short_fingerprint(&source.fingerprint),
            short_fingerprint(&current.fingerprint),
        );
    }
    Ok((state, source))
}

#[derive(Debug)]
struct CaptureFailure {
    message: String,
    recoverable: bool,
}

fn capture_once(
    preset: &str,
    overrides: &BTreeMap<String, String>,
    state: &CaptureServerState,
    source: &CaptureSourceSnapshot,
    camera: CaptureCameraOverride,
    graphics: CaptureGraphicsOverrides,
) -> Result<PathBuf, CaptureFailure> {
    let log_start = if state.completed_captures == 0 {
        read_json::<LauncherState>(&diagnostic_path(LAUNCHER_FILE))
            .map_or(0, |launcher| launcher.log_start_bytes)
    } else {
        file_len(&diagnostic_path(LOG_FILE))
    };
    let request = CaptureRequest {
        schema_version: CAPTURE_PROTOCOL_SCHEMA,
        id: Uuid::new_v4().simple().to_string(),
        action: CaptureAction::Capture,
        preset: preset.to_owned(),
        overrides: overrides.clone(),
        source: source.clone(),
        camera,
        graphics,
    };
    write_json_atomic(&diagnostic_path(REQUEST_FILE), &request).map_err(|message| {
        CaptureFailure {
            message,
            recoverable: false,
        }
    })?;
    // Software-Vulkan (llvmpipe) fallback boxes render warmup at seconds per
    // frame, so the response wait must cover a full preset warmup there; a
    // hardware run answers long before either limit.
    let requested = Instant::now();
    let deadline = requested + Duration::from_secs(capture_timeout_secs());
    while Instant::now() < deadline {
        if let Some(response) = read_json::<CaptureResponse>(&diagnostic_path(RESPONSE_FILE))
            && response.id == request.id
        {
            runlog::phase("render", requested.elapsed());
            let validating = Instant::now();
            if response.schema_version != CAPTURE_PROTOCOL_SCHEMA {
                return Err(CaptureFailure {
                    message: format!(
                        "capture response schema {} does not match client schema {}",
                        response.schema_version, CAPTURE_PROTOCOL_SCHEMA
                    ),
                    recoverable: true,
                });
            }
            if !response.ok {
                return Err(CaptureFailure {
                    recoverable: response.message.contains("different boot world")
                        || response.message.contains("restart required"),
                    message: response.message,
                });
            }
            if response.source.fingerprint != source.fingerprint {
                return Err(CaptureFailure {
                    message: format!(
                        "capture response used source {}, requested {}; restarting to avoid unattributed evidence",
                        short_fingerprint(&response.source.fingerprint),
                        short_fingerprint(&source.fingerprint),
                    ),
                    recoverable: true,
                });
            }
            let path =
                response
                    .output
                    .as_ref()
                    .map(PathBuf::from)
                    .ok_or_else(|| CaptureFailure {
                        message: "capture succeeded without an output path".into(),
                        recoverable: true,
                    })?;
            validate_capture_output(&path)?;
            validate_render_log(log_start)?;
            let workspace_after = capture_source_snapshot().map_err(|message| CaptureFailure {
                message,
                recoverable: false,
            })?;
            let workspace_matches = workspace_after.fingerprint == source.fingerprint;
            let workspace_relation = workspace_relation(source, &workspace_after);
            let effective_graphics = response.graphics.ok_or_else(|| CaptureFailure {
                message: "capture succeeded without effective graphics settings".into(),
                recoverable: true,
            })?;
            write_capture_receipt(
                &path,
                CaptureReceipt {
                    schema: "thalos.capture-receipt.v2",
                    request_id: &request.id,
                    preset,
                    output: path.display().to_string(),
                    completed_unix_ms: response.completed_unix_ms,
                    renderer_pid: state.pid,
                    source,
                    renderer_launch_source: &state.source,
                    workspace_after: &workspace_after,
                    source_floor_guaranteed: true,
                    workspace_relation,
                    workspace_matches,
                    camera: response.camera,
                    graphics: effective_graphics,
                },
            )
            .map_err(|message| CaptureFailure {
                message,
                recoverable: false,
            })?;
            runlog::phase("validate", validating.elapsed());
            runlog::field("workspace_relation", workspace_relation);
            println!(
                "captured {} [source {} · {}{}]",
                path.display(),
                short_fingerprint(&source.fingerprint),
                short_revision(&source.git_revision),
                if source.working_tree_dirty {
                    "+dirty"
                } else {
                    ""
                },
            );
            if !workspace_matches {
                eprintln!(
                    "workspace advanced after the capture source floor: floor={}, current={}; the image includes the floor, while later edits may or may not be present (see {})",
                    short_fingerprint(&source.fingerprint),
                    short_fingerprint(&workspace_after.fingerprint),
                    receipt_path(&path).display(),
                );
            }
            return Ok(path);
        }
        if !process_alive(state.pid) {
            let log = log_from(log_start);
            if let Some(kind) = resource_fault_kind(&log) {
                record_resource_fault(kind, &log);
                return Err(CaptureFailure {
                    message: resource_fault_message(kind, &log),
                    recoverable: false,
                });
            }
            return Err(CaptureFailure {
                message: format!("capture server exited\n{log}"),
                recoverable: true,
            });
        }
        if let Some(message) = capture_rss_failure(state.pid, overrides) {
            record_resource_fault("capture host memory runaway", &message);
            return Err(CaptureFailure {
                message: resource_fault_message("capture host memory runaway", &message),
                recoverable: false,
            });
        }
        thread::sleep(Duration::from_millis(100));
    }
    let log = log_from(log_start);
    if let Some(kind) = resource_fault_kind(&log) {
        record_resource_fault(kind, &log);
        return Err(CaptureFailure {
            message: resource_fault_message(kind, &log),
            recoverable: false,
        });
    }
    Err(CaptureFailure {
        message: format!("capture timed out\n{log}"),
        recoverable: true,
    })
}

fn ensure_server(
    preset: &str,
    overrides: &BTreeMap<String, String>,
    source: &CaptureSourceSnapshot,
) -> Result<CaptureServerState, String> {
    let state = read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE));
    // Why a shot did or did not pay for a host boot is the first question asked
    // of a slow capture, so the decision is recorded rather than inferred from
    // timing — and the same classifier makes it, so the record cannot drift
    // from the branch actually taken.
    let action = host_action(state.as_ref(), preset, overrides, source);
    runlog::field("host_action", action);
    let state = if action == HOST_REUSE {
        state.expect("reuse implies a compatible state exists")
    } else {
        // Stale Rust/Cargo sources restart the host through a rebuild: there
        // is no in-process code reload (Rust hot-patching was retired —
        // ADR-20260724T153619Z; an applied subsecond patch crashed the app,
        // INC-20260724T044418Z). `cargo run` below recompiles whatever
        // changed and relaunches, so a Rust edit can never leave a silently
        // stale or crashed server behind.
        start_server(preset, overrides, source)?
    };
    let started = Instant::now();
    let state = wait_for_shader_reload(state);
    runlog::phase("shader_reload", started.elapsed());
    state
}

/// The one `host_action` value that means "no boot was paid for".
const HOST_REUSE: &str = "reuse";

/// Decide — and name — whether this shot reuses the resident host or replaces
/// it, and why. This is the *only* place that decision is made: `ensure_server`
/// branches on the returned value, so the recorded reason and the branch taken
/// cannot disagree.
fn host_action(
    state: Option<&CaptureServerState>,
    preset: &str,
    overrides: &BTreeMap<String, String>,
    source: &CaptureSourceSnapshot,
) -> &'static str {
    if state.is_none() {
        "start"
    } else if host_sources_stale(state, source) {
        "restart_stale_source"
    } else if !compatible_state(state, preset, requested_size(overrides)) {
        "restart_incompatible_scene"
    } else if !launcher_matches(overrides) {
        "restart_startup_override"
    } else {
        HOST_REUSE
    }
}

/// Content equality, not timestamps, decides whether the resident binary
/// contains the caller's Rust/Cargo inputs. This remains correct across clock
/// skew, timestamp-preserving tools, and a shared checkout with several agents.
fn host_sources_stale(state: Option<&CaptureServerState>, source: &CaptureSourceSnapshot) -> bool {
    state.is_none_or(|state| {
        state.source.build_fingerprint.is_empty()
            || state.source.build_fingerprint != source.build_fingerprint
    })
}

/// Boot the host, recovering once from a self-inconsistent build tree.
///
/// A dev-lane build can fail for two very different reasons: the code is wrong
/// (the agent must read the error), or the *artifacts* disagree with each other
/// (nobody's error — the tree just needs its incremental cache dropped). The
/// second class used to end the run with a raw log tail, and the obvious manual
/// remedy — `cargo clean -p bevy_dylib` — is what produces it in the first
/// place. Recover automatically instead, so a stale-artifact link failure costs
/// a rebuild rather than a diagnosis session. See
/// INC-20260724T182642Z-stale-dylib-incremental-link-corruption.
fn start_server(
    preset: &str,
    overrides: &BTreeMap<String, String>,
    source: &CaptureSourceSnapshot,
) -> Result<CaptureServerState, String> {
    let offset = file_len(&diagnostic_path(LOG_FILE));
    runlog::count("host_start");
    match start_server_once(preset, overrides, source) {
        Ok(state) => verify_host_launch_source(state, source),
        Err(_) if toolchain_corruption(&log_from(offset)) => {
            runlog::count("rebuild_recovery");
            eprintln!(
                "build failed on stale-artifact corruption (objects reference internal symbols \
                 from a previous bevy_dylib link, not a code error); dropping \
                 target/debug/incremental and rebuilding once"
            );
            purge_incremental_cache()?;
            let retry_offset = file_len(&diagnostic_path(LOG_FILE));
            start_server_once(preset, overrides, source)
                .and_then(|state| verify_host_launch_source(state, source))
                .map_err(|retry| {
                    if toolchain_corruption(&log_from(retry_offset)) {
                        format!(
                            "{retry}\n\nstill corrupt after dropping the incremental cache — \
                             run `just build-reset` for a full lane reset"
                        )
                    } else {
                        retry
                    }
                })
        }
        Err(error) => Err(error),
    }
}

fn verify_host_launch_source(
    state: CaptureServerState,
    requested: &CaptureSourceSnapshot,
) -> Result<CaptureServerState, String> {
    if state.source.build_fingerprint == requested.build_fingerprint {
        return Ok(state);
    }
    let _ = stop_server(true);
    Err(format!(
        "capture host advertised build source {}, but launch requested {}; refusing unattributed evidence",
        short_fingerprint(&state.source.build_fingerprint),
        short_fingerprint(&requested.build_fingerprint),
    ))
}

fn workspace_relation(
    source_floor: &CaptureSourceSnapshot,
    workspace_after: &CaptureSourceSnapshot,
) -> &'static str {
    if source_floor.fingerprint == workspace_after.fingerprint {
        "exact"
    } else {
        "advanced-since-source-floor"
    }
}

/// Link/metadata failures that mean "the build tree disagrees with itself"
/// rather than "this code does not compile".
///
/// The observed signature is `undefined symbol: anon.<cgu>.N.llvm.<hash>`
/// referenced from a workspace rlib: LLVM-internalized symbols shared with
/// `bevy_dylib` that an incremental CGU still names after the dylib was rebuilt
/// or partially cleaned underneath it.
pub(crate) fn toolchain_corruption(log: &str) -> bool {
    let lower = log.to_ascii_lowercase();
    lower.contains("undefined symbol: anon.")
        || (lower.contains("undefined symbol") && lower.contains(".llvm."))
        || lower.contains("found invalid metadata files")
        || (lower.contains("incremental compilation") && lower.contains("corrupt"))
}

fn purge_incremental_cache() -> Result<(), String> {
    let cache = workspace_root().join("target/debug/incremental");
    match fs::remove_dir_all(&cache) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("purge {}: {error}", cache.display())),
    }
}

/// Full dev-lane reset: stop the host, drop the incremental cache, and clean
/// the dynamic-linking set **as one unit**.
///
/// The set matters. Cleaning `bevy_dylib` on its own (or any subset of the
/// crates that link against it) is exactly the partial state that produces
/// dangling internal-symbol references — the reset exists so nobody has to
/// hand-roll `cargo clean -p` again.
fn reset_build_state() -> Result<(), String> {
    let _ = stop_server(true);
    purge_incremental_cache()?;
    println!("dropped target/debug/incremental");
    let status = Command::new("cargo")
        .args([
            "clean",
            "-p",
            "bevy_dylib",
            "-p",
            "thalos_runtime",
            "-p",
            "thalos_capture_runtime",
            "-p",
            "thalos_capture_host",
            "-p",
            "thalos_game",
            "-p",
            "thalos_body_render",
        ])
        .current_dir(workspace_root())
        .status()
        .map_err(|error| format!("cargo clean: {error}"))?;
    if !status.success() {
        return Err(format!("cargo clean exited with {status}"));
    }
    println!("cleaned the dynamic-linking crate set; the next capture rebuilds it");
    Ok(())
}

fn start_server_once(
    preset: &str,
    overrides: &BTreeMap<String, String>,
    source: &CaptureSourceSnapshot,
) -> Result<CaptureServerState, String> {
    let started = Instant::now();
    let state = start_server_once_inner(preset, overrides, source);
    // Recorded on both outcomes: a boot that fails after two minutes of
    // rebuilding is exactly the cost worth seeing in the record.
    runlog::phase("host_start", started.elapsed());
    state
}

fn start_server_once_inner(
    preset: &str,
    overrides: &BTreeMap<String, String>,
    source: &CaptureSourceSnapshot,
) -> Result<CaptureServerState, String> {
    stop_server(true)?;
    fs::create_dir_all(diagnostics_dir()).map_err(|error| error.to_string())?;
    for filename in [REQUEST_FILE, RESPONSE_FILE, STATE_FILE] {
        let _ = fs::remove_file(diagnostic_path(filename));
    }

    // Plain cargo run on the shared dynamic dev fingerprint: cargo rebuilds
    // whatever changed, sets up the bevy_dylib search path for its child
    // (the INC-0008 contract), and stays resident as the launcher process
    // whose tree `stop` terminates.
    let command = vec![
        "run",
        "-p",
        "thalos_capture_host",
        "--features",
        "dev-renderer",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();

    let log_path = diagnostic_path(LOG_FILE);
    let log_start_bytes = file_len(&log_path);
    let startup_overrides = startup_overrides(overrides);
    let mut environment = startup_overrides.clone();
    environment.insert("THALOS_SCREENSHOT".into(), preset.to_owned());
    environment.insert("THALOS_CAPTURE_SERVER".into(), "1".into());
    environment.insert(
        "THALOS_CAPTURE_SOURCE_FINGERPRINT".into(),
        source.fingerprint.clone(),
    );
    environment.insert(
        "THALOS_CAPTURE_BUILD_FINGERPRINT".into(),
        source.build_fingerprint.clone(),
    );
    environment.insert(
        "THALOS_CAPTURE_GIT_REVISION".into(),
        source.git_revision.clone(),
    );
    environment.insert(
        "THALOS_CAPTURE_GIT_DIRTY".into(),
        source.working_tree_dirty.to_string(),
    );
    environment.insert(
        "BEVY_ASSET_ROOT".into(),
        workspace_root().display().to_string(),
    );
    let mut launcher_process = spawn_detached_launcher(&command, &environment, &log_path)?;
    let launcher = LauncherState {
        schema_version: CAPTURE_PROTOCOL_SCHEMA,
        pid: launcher_process.pid,
        preset: preset.to_owned(),
        launched_unix_ms: timestamp_millis(),
        log_start_bytes,
        startup_overrides,
        source: source.clone(),
        command: std::iter::once("cargo".to_owned()).chain(command).collect(),
    };
    write_json_atomic(&diagnostic_path(LAUNCHER_FILE), &launcher)?;
    println!(
        "starting capture renderer for {preset} at source {} (rebuilding changed sources; a cold build may take a while)",
        short_fingerprint(&source.fingerprint),
    );
    let deadline = Instant::now() + Duration::from_secs(1800);
    while Instant::now() < deadline {
        if !launcher_process.is_running() {
            let log = log_from(log_start_bytes);
            if let Some(kind) = resource_fault_kind(&log) {
                record_resource_fault(kind, &log);
                return Err(resource_fault_message(kind, &log));
            }
            return Err(format!("capture launcher exited\n{log}"));
        }
        if let Some(state) = read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE)) {
            if let Some(message) = capture_rss_failure(state.pid, overrides) {
                record_resource_fault("capture host memory runaway", &message);
                terminate_process_tree(launcher_process.pid);
                return Err(resource_fault_message(
                    "capture host memory runaway",
                    &message,
                ));
            }
            if compatible_state(Some(&state), preset, requested_size(overrides)) && state.ready {
                return Ok(state);
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
    Err(format!(
        "capture server did not become ready\n{}",
        log_tail(50)
    ))
}

/// Block until the running host has picked up any WGSL edit newer than its
/// launch (Bevy's `embedded_watcher` reloads both `assets/shaders/*` and
/// crate-embedded shaders in place). Rust staleness never reaches here — it
/// restarts the host in `ensure_server` instead.
fn wait_for_shader_reload(mut state: CaptureServerState) -> Result<CaptureServerState, String> {
    let launched = read_json::<LauncherState>(&diagnostic_path(LAUNCHER_FILE))
        .map_or(0, |launcher| launcher.launched_unix_ms);
    let shader_source = newest_mtime_ms(
        &[
            workspace_root().join("assets/shaders"),
            workspace_root().join("crates"),
        ],
        "wgsl",
    );
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut announced = false;
    loop {
        state = read_json(&diagnostic_path(STATE_FILE)).unwrap_or(state);
        let need_shader = shader_source > launched && state.shader_reload_unix_ms < shader_source;
        if !need_shader {
            thread::sleep(Duration::from_millis(750));
            return Ok(state);
        }
        if !announced {
            println!("waiting for shader hot reload");
            announced = true;
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "timed out waiting for shader hot reload\n{}",
                log_tail(50)
            ));
        }
        if !process_alive(state.pid) {
            return Err(format!(
                "capture server exited during reload\n{}",
                log_tail(50)
            ));
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn stop_server(quiet: bool) -> Result<(), String> {
    if let Some(state) = read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE))
        && process_alive(state.pid)
    {
        let request = CaptureRequest {
            schema_version: CAPTURE_PROTOCOL_SCHEMA,
            id: Uuid::new_v4().simple().to_string(),
            action: CaptureAction::Shutdown,
            preset: String::new(),
            overrides: BTreeMap::new(),
            source: CaptureSourceSnapshot::default(),
            camera: CaptureCameraOverride::default(),
            graphics: CaptureGraphicsOverrides::default(),
        };
        write_json_atomic(&diagnostic_path(REQUEST_FILE), &request)?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && process_alive(state.pid) {
            thread::sleep(Duration::from_millis(100));
        }
    }
    if let Some(launcher) = read_json::<LauncherState>(&diagnostic_path(LAUNCHER_FILE))
        && process_alive(launcher.pid)
    {
        terminate_process_tree(launcher.pid);
    }
    let _ = fs::remove_file(diagnostic_path(STATE_FILE));
    let _ = fs::remove_file(diagnostic_path(LAUNCHER_FILE));
    if !quiet {
        println!("visual capture server stopped");
    }
    Ok(())
}

fn status() {
    match read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE)) {
        Some(state) if process_alive(state.pid) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&state).unwrap_or_default()
            )
        }
        _ => println!("visual capture server is not running"),
    }
    let fault_path = machine_capture_path(RESOURCE_FAULT_FILE);
    if let Some(fault) = read_json::<CaptureResourceFault>(&fault_path) {
        let age_s = timestamp_millis().saturating_sub(fault.observed_unix_ms) / 1000;
        if fault.kind == "GPU device loss" {
            let rebooted = fault
                .observed_uptime_ms
                .zip(system_uptime_ms())
                .is_some_and(|(observed, current)| current < observed);
            if !rebooted {
                println!(
                    "resource quarantine active: GPU device loss ({}s ago; until OS reboot)",
                    age_s
                );
            }
        } else if age_s < RESOURCE_FAULT_COOLDOWN_SECS as u128 {
            println!(
                "resource quarantine active: {} ({}s ago; {}s remaining)",
                fault.kind,
                age_s,
                RESOURCE_FAULT_COOLDOWN_SECS as u128 - age_s
            );
        }
        println!("resource quarantine record: {}", fault_path.display());
    }
    let events = diagnostic_path(RESOURCE_EVENT_FILE);
    if events.exists() {
        println!("resource quarantine history: {}", events.display());
    }
}

fn compatible_state(
    state: Option<&CaptureServerState>,
    preset: &str,
    expected_size: Option<(u32, u32)>,
) -> bool {
    state.is_some_and(|state| {
        state.schema_version == CAPTURE_PROTOCOL_SCHEMA
            && state
                .compatible_presets
                .iter()
                .any(|compatible| compatible == preset)
            && process_alive(state.pid)
            && expected_size.is_none_or(|size| size == (state.width, state.height))
            && timestamp_millis().saturating_sub(state.heartbeat_unix_ms) < 300_000
            && state.ready
    })
}

fn requested_size(overrides: &BTreeMap<String, String>) -> Option<(u32, u32)> {
    let raw = overrides.get("THALOS_SCREENSHOT_SIZE")?;
    let (width, height) = raw.split_once(['x', 'X', '*'])?;
    Some((width.trim().parse().ok()?, height.trim().parse().ok()?))
}

fn startup_overrides(overrides: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    STARTUP_OVERRIDE_KEYS
        .iter()
        .filter_map(|key| {
            overrides
                .get(*key)
                .map(|value| ((*key).to_owned(), value.clone()))
        })
        .collect()
}

fn launcher_matches(overrides: &BTreeMap<String, String>) -> bool {
    read_json::<LauncherState>(&diagnostic_path(LAUNCHER_FILE))
        .is_some_and(|launcher| launcher.startup_overrides == startup_overrides(overrides))
}

fn canonical_preset(raw: &str) -> Result<String, String> {
    let slug = raw.trim().to_ascii_lowercase().replace('_', "-");
    let explicit_viewpoint = slug.strip_prefix("viewpoint:");
    let canonical = match slug.as_str() {
        "latest" | "perspective" => "latest-perspective",
        "spaceport" | "aerial" | "base" => "spaceport-aerial",
        "runway-sky" | "surface-atmosphere" => "runway-atmosphere",
        "space-center" | "spacecenter" | "play" => "hub",
        "dry" | "drybelt" | "desert" | "biome" => "dry-belt",
        "earth-ref" | "atmosphere" | "atmo" => "earth-reference",
        "open-ocean" | "sea" | "water" => "ocean",
        "sea-slopes" | "slope-field" => "ocean-slopes",
        "mira" => "mira-orbit",
        "regolith" => "mira-surface",
        "regolith-eva" => "mira-eva",
        "mira-full" | "mira-globe" => "mira-disc",
        "mira-oblique" | "mira-limb" => "mira-approach",
        "mira-crater" | "crater-rim" => "mira-rim",
        "clouds-runway" => "cloud-runway",
        "clouds-motion" => "cloud-motion",
        "clouds-cruise" | "cloud-deck" => "cloud-cruise",
        "inside-cloud" | "inside-clouds" => "cloud-interior",
        "cloud-orbit" | "clouds-orbit" => "cloud-limb",
        "cloud-globe" | "cloud-disc" | "full-planet" | "planet-disc" => "cloud-planet",
        "clouds-sunset" => "cloud-sunset",
        "engine" | "exhaust" | "rocket" => "plume",
        "massif" | "mountains" => "massif-aerial",
        "ridge" => "massif-ridge",
        "valley" => "massif-valley",
        value => value,
    };
    if let Ok(catalog) = read_viewpoint_catalog() {
        let viewpoint_id = explicit_viewpoint.unwrap_or(canonical);
        if let Some(viewpoint) = catalog.find_scripted(viewpoint_id) {
            Ok(viewpoint.driver.clone())
        } else if canonical == "latest-perspective" {
            Ok("latest-perspective".to_owned())
        } else if catalog.find(viewpoint_id).is_some() {
            Ok(format!("viewpoint:{viewpoint_id}"))
        } else {
            let ids = catalog
                .scripted_viewpoints
                .iter()
                .map(|viewpoint| viewpoint.id.as_str())
                .collect::<Vec<_>>();
            let saved_ids = catalog
                .viewpoints
                .iter()
                .map(|viewpoint| viewpoint.id.as_str())
                .collect::<Vec<_>>();
            Err(format!(
                "unknown capture scene {raw:?}; agent views: {}; saved views: {}",
                if ids.is_empty() {
                    "(none)".to_owned()
                } else {
                    ids.join(", ")
                },
                if saved_ids.is_empty() {
                    "(none)".to_owned()
                } else {
                    saved_ids.join(", ")
                }
            ))
        }
    } else if CAPTURE_PRESETS.contains(&canonical) {
        Ok(canonical.to_owned())
    } else {
        Err(format!(
            "unknown capture scene {raw:?}; expected one of {} or a viewpoint in {}",
            CAPTURE_PRESETS.join(", "),
            viewpoint_catalog_path().display()
        ))
    }
}

fn viewpoint_catalog_path() -> PathBuf {
    env::var_os("THALOS_VIEWPOINTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("assets/viewpoints.json"))
}

fn read_viewpoint_catalog() -> Result<ViewpointCatalog, String> {
    let path = viewpoint_catalog_path();
    let bytes =
        fs::read(&path).map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let catalog: ViewpointCatalog = serde_json::from_slice(&bytes)
        .map_err(|error| format!("could not parse {}: {error}", path.display()))?;
    catalog.validate()?;
    Ok(catalog)
}

fn apply_safe_viewpoint_size(preset: &str, overrides: &mut BTreeMap<String, String>) {
    if overrides.contains_key("THALOS_SCREENSHOT_SIZE") {
        return;
    }
    let Ok(catalog) = read_viewpoint_catalog() else {
        return;
    };
    let viewpoint = match preset {
        "latest-perspective" => catalog.latest(),
        scene => scene
            .strip_prefix("viewpoint:")
            .and_then(|id| catalog.find(id)),
    };
    let Some(viewpoint) = viewpoint else {
        return;
    };
    let [width, height] = viewpoint.optics.sensor.aspect;
    let fitted = fit_inside(width, height, SAFE_VIEWPOINT_WIDTH, SAFE_VIEWPOINT_HEIGHT);
    println!(
        "viewpoint sensor is {width}:{height}; rendering {}x{} at the temporary standard fidelity (use --size WIDTHxHEIGHT for an explicit output extent)",
        fitted.0, fitted.1,
    );
    overrides.insert(
        "THALOS_SCREENSHOT_SIZE".into(),
        format!("{}x{}", fitted.0, fitted.1),
    );
}

fn validate_viewpoint_output_aspect(
    preset: &str,
    overrides: &BTreeMap<String, String>,
) -> Result<(), String> {
    let Some((output_width, output_height)) = requested_size(overrides) else {
        return Ok(());
    };
    let catalog = read_viewpoint_catalog()?;
    let viewpoint = match preset {
        "latest-perspective" => catalog.latest(),
        scene => scene
            .strip_prefix("viewpoint:")
            .and_then(|id| catalog.find(id)),
    };
    let Some(viewpoint) = viewpoint else {
        return Ok(());
    };
    let [sensor_width, sensor_height] = viewpoint.optics.sensor.aspect;
    if u64::from(sensor_width) * u64::from(output_height)
        != u64::from(output_width) * u64::from(sensor_height)
    {
        return Err(format!(
            "viewpoint {} uses a {sensor_width}:{sensor_height} sensor window, but --size is {output_width}x{output_height}; choose a matching output aspect until an explicit crop/fit policy exists",
            viewpoint.id
        ));
    }
    Ok(())
}

fn fit_inside(width: u32, height: u32, max_width: u32, max_height: u32) -> (u32, u32) {
    let width_scale = max_width as f64 / width as f64;
    let height_scale = max_height as f64 / height as f64;
    let scale = width_scale.min(height_scale);
    (
        ((width as f64 * scale).round() as u32).max(1),
        ((height as f64 * scale).round() as u32).max(1),
    )
}

fn capture_timeout_secs() -> u64 {
    env::var("THALOS_CAPTURE_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(1800)
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("capture tool lives under <workspace>/tools/capture")
        .to_path_buf()
}

fn diagnostics_dir() -> PathBuf {
    workspace_root().join("artifacts/diagnostics")
}

fn machine_capture_dir() -> PathBuf {
    env::temp_dir().join("thalos-machine-capture")
}

fn machine_capture_path(filename: &str) -> PathBuf {
    machine_capture_dir().join(filename)
}

fn diagnostic_path(filename: &str) -> PathBuf {
    diagnostics_dir().join(filename)
}

fn guard_resource_fault() -> Result<(), String> {
    let path = machine_capture_path(RESOURCE_FAULT_FILE);
    let fault = read_json::<CaptureResourceFault>(&path);
    if env::var("THALOS_CAPTURE_IGNORE_RESOURCE_FAULT")
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
    {
        if let Some(fault) = fault.as_ref() {
            append_resource_event(
                "override-bypassed",
                fault,
                "THALOS_CAPTURE_IGNORE_RESOURCE_FAULT allowed a diagnostic attempt",
            );
        }
        return Ok(());
    }
    let Some(fault) = fault else {
        return Ok(());
    };
    if fault.kind == "GPU device loss" {
        let rebooted = fault
            .observed_uptime_ms
            .zip(system_uptime_ms())
            .is_some_and(|(observed, current)| current < observed);
        if rebooted {
            append_resource_event(
                "cleared-after-reboot",
                &fault,
                "system uptime moved behind the recorded fault uptime",
            );
            let _ = fs::remove_file(path);
            return Ok(());
        }
        append_resource_event(
            "blocked",
            &fault,
            "device-loss policy remains active for this OS boot",
        );
        return Err(
            "capture renderer is quarantined after GPU device loss for the rest of this OS boot. No renderer was started; reboot first. Override only after independently confirming recovery with THALOS_CAPTURE_IGNORE_RESOURCE_FAULT=1"
                .into(),
        );
    }
    let age_ms = timestamp_millis().saturating_sub(fault.observed_unix_ms);
    let cooldown_ms = RESOURCE_FAULT_COOLDOWN_SECS as u128 * 1000;
    if age_ms >= cooldown_ms {
        append_resource_event(
            "expired",
            &fault,
            "bounded resource-pressure cooldown elapsed",
        );
        let _ = fs::remove_file(path);
        return Ok(());
    }
    let remaining = (cooldown_ms - age_ms).div_ceil(1000);
    append_resource_event(
        "blocked",
        &fault,
        "bounded resource-pressure cooldown remains active",
    );
    Err(format!(
        "capture renderer is quarantined for another {remaining}s after {}. No renderer was started; this prevents queued agents from retrying a failed GPU. If Windows reports that the GPU is lost, reboot first. Override only after recovery with THALOS_CAPTURE_IGNORE_RESOURCE_FAULT=1",
        fault.kind
    ))
}

fn record_resource_fault(kind: &str, log: &str) {
    let detail = log
        .lines()
        .rev()
        .take(50)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    let fault = CaptureResourceFault {
        observed_unix_ms: timestamp_millis(),
        observed_uptime_ms: system_uptime_ms(),
        kind: kind.to_owned(),
        detail,
    };
    if write_json_atomic(&machine_capture_path(RESOURCE_FAULT_FILE), &fault).is_ok() {
        append_resource_event("recorded", &fault, "fatal capture resource signature");
    }
}

fn clear_resource_fault() {
    let path = machine_capture_path(RESOURCE_FAULT_FILE);
    if let Some(fault) = read_json::<CaptureResourceFault>(&path) {
        append_resource_event(
            "cleared-after-success",
            &fault,
            "a diagnostic override completed a valid capture",
        );
    }
    let _ = fs::remove_file(path);
}

fn append_resource_event(event: &str, fault: &CaptureResourceFault, note: &str) {
    let record = CaptureResourceEvent {
        schema: "thalos.capture-resource-event.v1",
        event,
        recorded_unix_ms: timestamp_millis(),
        recorded_uptime_ms: system_uptime_ms(),
        client_pid: std::process::id(),
        command: env::args().skip(1).collect::<Vec<_>>().join(" "),
        workspace: workspace_root().display().to_string(),
        fault_observed_unix_ms: fault.observed_unix_ms,
        fault_observed_uptime_ms: fault.observed_uptime_ms,
        fault_age_ms: timestamp_millis().saturating_sub(fault.observed_unix_ms),
        kind: &fault.kind,
        policy: if fault.kind == "GPU device loss" {
            "until-os-reboot"
        } else {
            "bounded-cooldown"
        },
        cooldown_secs: (fault.kind != "GPU device loss").then_some(RESOURCE_FAULT_COOLDOWN_SECS),
        note,
        fault_detail: (event == "recorded").then_some(fault.detail.as_str()),
    };
    let path = diagnostic_path(RESOURCE_EVENT_FILE);
    let result = (|| -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| error.to_string())?;
        serde_json::to_writer(&mut file, &record).map_err(|error| error.to_string())?;
        writeln!(file).map_err(|error| error.to_string())
    })();
    if let Err(error) = result {
        eprintln!(
            "warning: could not append resource quarantine event {}: {error}",
            path.display()
        );
    }
}

fn resource_fault_kind(log: &str) -> Option<&'static str> {
    let lower = log.to_ascii_lowercase();
    // WGPU tears a device down after allocation failure and reports that as
    // `Caught DeviceLost error: Unknown Out of memory`. Generic DeviceLost
    // text is therefore not proof that the adapter/driver remains lost. Only
    // an explicit lost-device diagnosis outranks the causal OOM signature.
    if [
        "gpu is lost",
        "unknown device is lost",
        "device has been lost",
    ]
    .into_iter()
    .any(|marker| lower.contains(marker))
    {
        return Some("GPU device loss");
    }
    if ["out of memory", "failed to allocate device memory"]
        .into_iter()
        .any(|marker| lower.contains(marker))
    {
        return Some("GPU out-of-memory");
    }
    if lower.contains("timed out while waiting on the last successful submission") {
        return Some("GPU submission timeout");
    }
    if lower.contains("capture host rss limit exceeded") {
        return Some("capture host memory runaway");
    }
    if lower.contains("device lost") || lower.contains("devicelost") {
        return Some("GPU device loss");
    }
    None
}

fn resource_fault_message(kind: &str, log: &str) -> String {
    let tail = log
        .lines()
        .rev()
        .take(50)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    if kind == "GPU device loss" {
        return format!(
            "capture stopped after GPU device loss; it will not auto-retry or start another renderer. The shared quarantine blocks queued agents until the OS reboots.\n{tail}"
        );
    }
    format!(
        "capture stopped after {kind}; it will not auto-retry or start another renderer. The workstation safety quarantine now blocks queued agents for {RESOURCE_FAULT_COOLDOWN_SECS}s. If the GPU is lost, reboot before overriding the quarantine.\n{tail}"
    )
}

fn absolute(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        workspace_root().join(path)
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Option<T> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn validate_capture_output(path: &Path) -> Result<(), CaptureFailure> {
    let metadata = fs::metadata(path).map_err(|error| CaptureFailure {
        message: format!(
            "capture reported success but {} is unavailable: {error}",
            path.display()
        ),
        recoverable: true,
    })?;
    if metadata.len() == 0 {
        return Err(CaptureFailure {
            message: format!("capture wrote an empty file at {}", path.display()),
            recoverable: true,
        });
    }
    image::ImageReader::open(path)
        .and_then(|reader| reader.with_guessed_format())
        .map_err(|error| CaptureFailure {
            message: format!("could not inspect {}: {error}", path.display()),
            recoverable: true,
        })?
        .decode()
        .map_err(|error| CaptureFailure {
            message: format!(
                "capture wrote an invalid image at {}: {error}",
                path.display()
            ),
            recoverable: true,
        })?;
    Ok(())
}

fn validate_render_log(offset: u64) -> Result<(), CaptureFailure> {
    let log = log_from(offset);
    if let Some(kind) = resource_fault_kind(&log) {
        record_resource_fault(kind, &log);
        return Err(CaptureFailure {
            message: resource_fault_message(kind, &log),
            recoverable: false,
        });
    }
    let Some(message) = render_log_failure(&log) else {
        return Ok(());
    };
    Err(CaptureFailure {
        message,
        recoverable: false,
    })
}

pub(crate) fn render_log_failure(log: &str) -> Option<String> {
    let lower = log.to_ascii_lowercase();
    let fatal_markers = [
        "shaderprocessorerror",
        "shader validation error",
        "pipeline validation error",
        "error occurred when trying to process shader",
        "wgpu error: validation error",
        "device::create_shader_module",
        "gpu is lost",
        "device lost",
        "panicked at",
    ];
    let broad_render_error = lower.lines().find(|line| {
        line.contains("error")
            && (line.contains("shader")
                || line.contains("pipeline")
                || line.contains("wgpu_core")
                || line.contains("wgpu_hal"))
    });
    let marker = fatal_markers
        .iter()
        .copied()
        .find(|marker| lower.contains(*marker))
        .or_else(|| broad_render_error.map(|_| "render error log"));
    if let Some(marker) = marker {
        return Some(format!(
            "capture produced an image after a fatal render error ({marker}); refusing invalid evidence\n{}",
            log.lines()
                .rev()
                .take(50)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    None
}

fn log_from(offset: u64) -> String {
    let Ok(mut file) = OpenOptions::new()
        .read(true)
        .open(diagnostic_path(LOG_FILE))
    else {
        return String::new();
    };
    if file.seek(SeekFrom::Start(offset)).is_err() {
        return String::new();
    }
    let mut text = String::new();
    let _ = file.read_to_string(&mut text);
    text
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("capture"),
        std::process::id()
    ));
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let _ = fs::remove_file(path);
    fs::rename(&temporary, path).map_err(|error| error.to_string())
}

/// Content-hash the source and asset trees. Called several times per shot
/// (floor, post-prepare, post-capture), so its cost is accumulated into one
/// `source_snapshot` phase rather than hiding inside whichever caller paid it.
fn capture_source_snapshot() -> Result<CaptureSourceSnapshot, String> {
    let started = Instant::now();
    let snapshot = capture_source_snapshot_inner();
    runlog::phase("source_snapshot", started.elapsed());
    snapshot
}

fn capture_source_snapshot_inner() -> Result<CaptureSourceSnapshot, String> {
    let root = workspace_root();
    let mut build_files = vec![
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
        root.join("rust-toolchain.toml"),
    ];
    for source_root in [
        root.join("apps"),
        root.join("crates"),
        root.join("tools/capture_host"),
    ] {
        build_files.extend(recursive_files(&source_root).into_iter().filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| matches!(extension, "rs" | "toml"))
        }));
    }
    build_files.sort();
    build_files.dedup();

    let mut capture_files = build_files.clone();
    // Runtime assets are evidence inputs too: an agent changing a terrain
    // package, texture, font, or authored catalog must be able to prove that
    // exact payload reached the image. The checked-in tree is small enough to
    // hash by content without a lossy mtime cache.
    capture_files.extend(recursive_files(&root.join("assets")));
    capture_files.extend(
        recursive_files(&root.join("crates"))
            .into_iter()
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "wgsl")
            }),
    );
    capture_files.sort();
    capture_files.dedup();

    let git_revision = git_output(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let working_tree_dirty =
        git_output(&["status", "--porcelain"]).is_none_or(|output| !output.is_empty());
    Ok(CaptureSourceSnapshot {
        fingerprint: fingerprint_files(&root, &capture_files)?,
        build_fingerprint: fingerprint_files(&root, &build_files)?,
        git_revision,
        working_tree_dirty,
    })
}

fn fingerprint_files(root: &Path, files: &[PathBuf]) -> Result<String, String> {
    let mut hasher = Sha256::new();
    for path in files {
        let relative = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let bytes =
            fs::read(path).map_err(|error| format!("fingerprint {}: {error}", path.display()))?;
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(workspace_root())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn receipt_path(output: &Path) -> PathBuf {
    output.with_extension("capture.json")
}

fn write_capture_receipt(output: &Path, receipt: CaptureReceipt<'_>) -> Result<PathBuf, String> {
    let path = receipt_path(output);
    write_json_atomic(&path, &receipt)?;
    Ok(path)
}

fn short_fingerprint(fingerprint: &str) -> &str {
    fingerprint.get(..12).unwrap_or(fingerprint)
}

fn short_revision(revision: &str) -> &str {
    revision.get(..10).unwrap_or(revision)
}

fn newest_mtime_ms(roots: &[PathBuf], extension: &str) -> u128 {
    roots
        .iter()
        .flat_map(|root| recursive_files(root))
        .filter(|path| path.extension().is_some_and(|value| value == extension))
        .filter_map(|path| fs::metadata(path).ok()?.modified().ok())
        .filter_map(|time| {
            time.duration_since(UNIX_EPOCH)
                .ok()
                .map(|value| value.as_millis())
        })
        .max()
        .unwrap_or(0)
}

fn file_len(path: &Path) -> u64 {
    fs::metadata(path).map_or(0, |metadata| metadata.len())
}

fn recursive_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return files;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files.extend(recursive_files(&path));
        } else {
            files.push(path);
        }
    }
    files
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn system_uptime_ms() -> Option<u64> {
    #[cfg(windows)]
    {
        Some(unsafe { windows_sys::Win32::System::SystemInformation::GetTickCount64() })
    }
    #[cfg(not(windows))]
    {
        let seconds = fs::read_to_string("/proc/uptime")
            .ok()?
            .split_whitespace()
            .next()?
            .parse::<f64>()
            .ok()?;
        Some((seconds * 1000.0) as u64)
    }
}

fn log_tail(lines: usize) -> String {
    fs::read_to_string(diagnostic_path(LOG_FILE))
        .map(|text| {
            text.lines()
                .rev()
                .take(lines)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_else(|_| "(capture-server log is unavailable)".to_owned())
}

fn capture_rss_failure(pid: u32, overrides: &BTreeMap<String, String>) -> Option<String> {
    let limit_mb = overrides
        .get("THALOS_CAPTURE_RSS_LIMIT_MB")
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .or_else(|| {
            env::var("THALOS_CAPTURE_RSS_LIMIT_MB")
                .ok()
                .and_then(|raw| raw.trim().parse::<u64>().ok())
        })
        .unwrap_or(DEFAULT_CAPTURE_RSS_LIMIT_MB);
    if limit_mb == 0 {
        return None;
    }
    let resident = process_resident_bytes(pid)?;
    let limit = limit_mb * 1024 * 1024;
    (resident > limit).then(|| {
        format!(
            "capture host RSS limit exceeded: pid {pid} uses {:.1} GiB, limit is {:.1} GiB",
            resident as f64 / (1024.0 * 1024.0 * 1024.0),
            limit as f64 / (1024.0 * 1024.0 * 1024.0),
        )
    })
}

fn process_resident_bytes(pid: u32) -> Option<u64> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::{
            Foundation::CloseHandle,
            System::{
                ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
                Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION},
            },
        };
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                return None;
            }
            let mut counters = PROCESS_MEMORY_COUNTERS {
                cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                ..Default::default()
            };
            let cb = counters.cb;
            let queried = K32GetProcessMemoryInfo(handle, &mut counters, cb) != 0;
            CloseHandle(handle);
            queried.then_some(counters.WorkingSetSize as u64)
        }
    }
    #[cfg(not(windows))]
    {
        let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        let kib = status
            .lines()
            .find_map(|line| line.strip_prefix("VmRSS:"))?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?;
        Some(kib * 1024)
    }
}

fn process_alive(pid: u32) -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::{
            Foundation::{CloseHandle, ERROR_INVALID_PARAMETER, GetLastError},
            System::Threading::{
                GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
            },
        };

        // `tasklist` is not a liveness API: it can return "Access denied" in a
        // restricted shell even for our own healthy process. The old code read
        // that as "dead" and repeatedly restarted the renderer. Query the
        // kernel object directly; if policy denies even limited access, err on
        // the side of "alive" so we never kill/replace a process we cannot
        // inspect.
        const STILL_ACTIVE: u32 = 259;
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                return GetLastError() != ERROR_INVALID_PARAMETER;
            }
            let mut exit_code = 0;
            let queried = GetExitCodeProcess(handle, &mut exit_code) != 0;
            CloseHandle(handle);
            queried && exit_code == STILL_ACTIVE
        }
    }
    #[cfg(not(windows))]
    {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .is_ok_and(|status| status.success())
    }
}

/// A launched host process, however the platform detaches it.
struct LauncherProcess {
    pid: u32,
    /// Present only when we own a real child handle (Unix). On Windows the
    /// host is deliberately *not* our child, so liveness is queried by pid.
    child: Option<std::process::Child>,
}

impl LauncherProcess {
    fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => process_alive(self.pid),
        }
    }
}

/// Start the persistent host so that it shares **no handle** with the shell
/// that invoked us.
///
/// The host outlives its client by design, so anything it inherits it holds
/// forever. On Windows that broke every piped or background-captured run: a
/// `CreateProcess(bInheritHandles = TRUE)` spawn duplicates *every* inheritable
/// handle into the child — including inheritable copies of the caller's stdout
/// pipe that never appear as our own std handles — so `just capture … | tail`
/// (and any agent harness that captures output) waited for an EOF that could
/// not arrive until the host died, minutes after the captures had succeeded.
/// Clearing `HANDLE_FLAG_INHERIT` on our std handles was measured to be
/// insufficient for exactly that reason.
///
/// The fix is to launch through `Start-Process`, i.e. `ShellExecuteEx`, which
/// creates the process with `bInheritHandles = FALSE` — a guaranteed-empty
/// inherited handle table, rather than an enumeration of leaks to plug. The
/// log redirection moves into a small `.cmd` shim so no handle needs to be
/// passed at all, and `cargo run` still fronts the host, preserving the
/// bevy_dylib search-path contract (INC-0008). See
/// INC-20260724T185500Z-persistent-host-inherits-caller-pipe.
#[cfg(windows)]
fn spawn_detached_launcher(
    command: &[String],
    environment: &BTreeMap<String, String>,
    log_path: &Path,
) -> Result<LauncherProcess, String> {
    let script_path = diagnostic_path("visual_capture_launch.cmd");
    let script = format!(
        "@echo off\r\ncd /d \"{root}\"\r\ncargo {args} 1>>\"{log}\" 2>&1\r\n",
        root = workspace_root().display(),
        args = command.join(" "),
        log = log_path.display(),
    );
    fs::write(&script_path, script)
        .map_err(|error| format!("write {}: {error}", script_path.display()))?;

    // `Start-Process` keeps `UseShellExecute = true` as long as no stream
    // redirection is requested — which is why the redirection lives in the
    // shim above. `-PassThru` hands back the shim's pid so `stop` can still
    // kill the whole tree.
    let quoted = script_path.display().to_string().replace('\'', "''");
    let ps =
        format!("$p = Start-Process -FilePath '{quoted}' -WindowStyle Hidden -PassThru; $p.Id");
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &ps])
        .current_dir(workspace_root())
        .envs(environment)
        .output()
        .map_err(|error| format!("launch capture host: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "launch capture host: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let pid = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u32>()
        .map_err(|error| format!("capture host launcher returned no pid: {error}"))?;
    Ok(LauncherProcess { pid, child: None })
}

/// Unix needs no such surgery: `std` sets `CLOEXEC` on every descriptor it does
/// not explicitly pass, so redirecting the host's stdio to the log is already
/// enough to keep the caller's pipe out of its hands.
#[cfg(unix)]
fn spawn_detached_launcher(
    command: &[String],
    environment: &BTreeMap<String, String>,
    log_path: &Path,
) -> Result<LauncherProcess, String> {
    use std::os::unix::process::CommandExt;

    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|error| format!("open capture log: {error}"))?;
    let stderr = log.try_clone().map_err(|error| error.to_string())?;
    let child = Command::new("cargo")
        .args(command)
        .current_dir(workspace_root())
        .envs(environment)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(stderr))
        .process_group(0)
        .spawn()
        .map_err(|error| format!("launch cargo run (capture host): {error}"))?;
    Ok(LauncherProcess {
        pid: child.id(),
        child: Some(child),
    })
}

fn terminate_process_tree(pid: u32) {
    #[cfg(windows)]
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .output();
    #[cfg(not(windows))]
    let _ = Command::new("kill")
        .args(["-TERM", "--", &format!("-{pid}")])
        .output();
}

fn print_help() {
    println!(
        "thalos_capture shot [PRESET ...] [--time SECONDS] [--focal-length MM] [--size WIDTHxHEIGHT] [--graphics NAME=VALUE,...] [--out PATH] [--report PATH] [--set KEY=VALUE]\n\
         thalos_capture compare [PRESET] [AXIS] [--out DIR] [--cold]\n\
         thalos_capture list viewpoints [--gallery] [--json] [--out DIR]\n\
         thalos_capture status\n\
         thalos_capture stop\n\
         thalos_capture reset   (stop + drop incremental + clean the dylib set)\n\n\
         Compatible scenes are rendered sequentially through one real camera and\n\
         one booted world. Viewpoint sensor aspect and lens define framing;\n\
         output pixels are selected independently and default safely near 1080p.\n\
         Viewpoint galleries reuse cached canonical captures and never render.\n\
         Graphics settings currently include clouds=on|off and grass=on|off.\n\
         --out/--report are single-preset options."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shot_parser_accepts_multiple_scenes_and_aliases() {
        let parsed = parse_capture_options(
            [
                "spaceport",
                "runway-atmosphere",
                "--time",
                "72000",
                "--size",
                "1600x900",
                "--focal-length",
                "85",
                "--graphics",
                "clouds=off,grass=on",
                "--set",
                "THALOS_SSAO=off",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .expect("parse");
        assert_eq!(
            parsed,
            ShotArgs {
                presets: vec!["spaceport-aerial".into(), "runway-atmosphere".into()],
                output: None,
                report: None,
                assignments: vec![
                    "THALOS_SCREENSHOT_TIME=72000".into(),
                    "THALOS_SCREENSHOT_SIZE=1600x900".into(),
                    "THALOS_SSAO=off".into(),
                ],
                camera: CaptureCameraOverride {
                    focal_length_mm: Some(85.0),
                },
                graphics: CaptureGraphicsOverrides {
                    clouds: Some(false),
                    grass: Some(true),
                },
            }
        );
    }

    #[test]
    fn shot_parser_rejects_non_finite_time() {
        assert!(
            parse_capture_options(
                ["spaceport", "--time", "NaN"]
                    .into_iter()
                    .map(str::to_owned)
            )
            .is_err()
        );
    }

    #[test]
    fn shot_parser_rejects_zero_or_malformed_size() {
        for size in ["0x1080", "1920x0", "native", "1920"] {
            assert!(
                parse_capture_options(["spaceport", "--size", size].into_iter().map(str::to_owned))
                    .is_err(),
                "{size}"
            );
        }
    }

    #[test]
    fn sensor_aspect_fills_safety_extent_without_changing_aspect() {
        assert_eq!(fit_inside(3840, 2160, 1920, 1080), (1920, 1080));
        assert_eq!(fit_inside(1920, 1200, 1920, 1080), (1728, 1080));
        assert_eq!(fit_inside(16, 9, 1920, 1080), (1920, 1080));
        assert_eq!(fit_inside(4, 3, 1920, 1080), (1440, 1080));
    }

    #[test]
    fn viewpoint_output_must_match_the_saved_sensor_aspect() {
        let mut overrides =
            BTreeMap::from([("THALOS_SCREENSHOT_SIZE".to_owned(), "1600x1200".to_owned())]);
        assert!(validate_viewpoint_output_aspect("viewpoint:dark-thalos", &overrides).is_err());
        overrides.insert("THALOS_SCREENSHOT_SIZE".to_owned(), "2560x1440".to_owned());
        assert!(validate_viewpoint_output_aspect("viewpoint:dark-thalos", &overrides).is_ok());
    }

    #[test]
    fn shot_parser_rejects_out_of_range_focal_length() {
        for focal_length in ["0", "11.9", "401", "NaN"] {
            assert!(
                parse_capture_options(
                    ["spaceport", "--focal-length", focal_length]
                        .into_iter()
                        .map(str::to_owned)
                )
                .is_err(),
                "{focal_length}"
            );
        }
    }

    /// The `host_action` a shot records is the branch `ensure_server` takes,
    /// so a wrong classification would mis-explain every slow capture. Stale
    /// build sources outrank scene compatibility: a host built from other code
    /// must restart even when it could otherwise serve the scene.
    #[test]
    fn host_action_names_why_a_shot_pays_for_a_boot() {
        let source = CaptureSourceSnapshot {
            fingerprint: "aaa".into(),
            build_fingerprint: "bbb".into(),
            git_revision: "rev".into(),
            working_tree_dirty: false,
        };
        let overrides = BTreeMap::new();

        assert_eq!(
            host_action(None, "spaceport-aerial", &overrides, &source),
            "start",
            "no resident host at all"
        );

        let stale = CaptureServerState {
            schema_version: CAPTURE_PROTOCOL_SCHEMA,
            pid: std::process::id(),
            preset: "spaceport-aerial".into(),
            compatible_presets: vec!["spaceport-aerial".into()],
            width: 1920,
            height: 1080,
            ready: true,
            busy: false,
            completed_captures: 0,
            shader_reload_unix_ms: 0,
            heartbeat_unix_ms: timestamp_millis(),
            source: CaptureSourceSnapshot {
                build_fingerprint: "older".into(),
                ..source.clone()
            },
        };
        assert_eq!(
            host_action(Some(&stale), "spaceport-aerial", &overrides, &source),
            "restart_stale_source",
            "a host built from other sources restarts even for a scene it serves"
        );

        let wrong_scene = CaptureServerState {
            compatible_presets: vec!["ocean".into()],
            source: source.clone(),
            ..stale
        };
        assert_eq!(
            host_action(Some(&wrong_scene), "spaceport-aerial", &overrides, &source),
            "restart_incompatible_scene"
        );
    }

    #[test]
    fn capture_client_lock_serializes_and_releases() {
        let path = workspace_root().join(format!(
            "target/capture-client-lock-test-{}.json",
            Uuid::new_v4().simple()
        ));
        let first =
            CaptureClientLock::acquire_at(path.clone(), "first".into(), Duration::ZERO, &mut None)
                .unwrap();
        let blocked =
            CaptureClientLock::acquire_at(path.clone(), "second".into(), Duration::ZERO, &mut None)
                .expect_err("second live owner must not acquire the lock");
        assert!(blocked.contains("pid"));
        drop(first);
        let second =
            CaptureClientLock::acquire_at(path.clone(), "second".into(), Duration::ZERO, &mut None)
                .unwrap();
        drop(second);
        assert!(!path.exists());
    }

    #[cfg(windows)]
    #[test]
    fn machine_mutex_cannot_be_stolen_by_a_second_waiter() {
        use std::sync::mpsc;

        let id = Uuid::new_v4().simple().to_string();
        let path = workspace_root().join(format!("target/capture-machine-lock-test-{id}.json"));
        let mutex_name = format!("Local\\ThalosMachineCaptureTest{id}");
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let owner_path = path.clone();
        let owner_name = mutex_name.clone();
        let owner = thread::spawn(move || {
            let lock = CaptureClientLock::acquire_named_mutex(
                owner_path,
                &owner_name,
                "owner".into(),
                Duration::ZERO,
                &mut None,
            )
            .unwrap();
            acquired_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(lock);
        });
        acquired_rx.recv().unwrap();

        let blocked = CaptureClientLock::acquire_named_mutex(
            path.clone(),
            &mutex_name,
            "waiter".into(),
            Duration::ZERO,
            &mut None,
        )
        .expect_err("a second waiter must not steal a live kernel mutex");
        assert!(blocked.contains("timed out"));

        release_tx.send(()).unwrap();
        owner.join().unwrap();
        let successor = CaptureClientLock::acquire_named_mutex(
            path.clone(),
            &mutex_name,
            "successor".into(),
            Duration::ZERO,
            &mut None,
        )
        .unwrap();
        drop(successor);
        assert!(!path.exists());
    }

    #[test]
    fn source_fingerprint_tracks_content_not_only_timestamps() {
        let root = workspace_root();
        let path = root.join(format!(
            "target/capture-source-fingerprint-test-{}.txt",
            Uuid::new_v4().simple()
        ));
        fs::write(&path, b"first").unwrap();
        let first = fingerprint_files(&root, std::slice::from_ref(&path)).unwrap();
        fs::write(&path, b"second").unwrap();
        let second = fingerprint_files(&root, std::slice::from_ref(&path)).unwrap();
        let _ = fs::remove_file(&path);
        assert_ne!(first, second);
    }

    #[test]
    fn workspace_source_snapshot_covers_build_and_capture_inputs() {
        let source = capture_source_snapshot().unwrap();
        assert_eq!(source.fingerprint.len(), 64);
        assert_eq!(source.build_fingerprint.len(), 64);
        assert_ne!(source.fingerprint, source.build_fingerprint);
        assert!(!source.git_revision.is_empty());
    }

    #[test]
    fn capture_receipt_sits_next_to_its_image() {
        assert_eq!(
            receipt_path(Path::new("artifacts/visual/latest/view.png")),
            PathBuf::from("artifacts/visual/latest/view.capture.json")
        );
    }

    #[test]
    fn source_mismatch_means_workspace_advanced_not_capture_rejected() {
        let floor = CaptureSourceSnapshot {
            fingerprint: "floor".into(),
            ..Default::default()
        };
        let exact = floor.clone();
        let advanced = CaptureSourceSnapshot {
            fingerprint: "newer".into(),
            ..Default::default()
        };
        assert_eq!(workspace_relation(&floor, &exact), "exact");
        assert_eq!(
            workspace_relation(&floor, &advanced),
            "advanced-since-source-floor"
        );
    }

    #[test]
    fn unknown_scene_is_rejected_instead_of_falling_back() {
        assert!(canonical_preset("typo-scene").is_err());
    }

    #[test]
    fn fatal_render_errors_are_promoted() {
        assert!(render_log_failure("wgpu error: Validation Error").is_some());
        assert!(render_log_failure("capture complete").is_none());
    }

    #[test]
    fn resource_faults_are_terminal_and_specific() {
        assert_eq!(
            resource_fault_kind("Caught rendering error: Out of Memory"),
            Some("GPU out-of-memory")
        );
        assert_eq!(
            resource_fault_kind(
                "We timed out while waiting on the last successful submission to complete!"
            ),
            Some("GPU submission timeout")
        );
        assert_eq!(
            resource_fault_kind(
                "Caught rendering error: Out of Memory; adapter reports GPU is lost"
            ),
            Some("GPU device loss")
        );
        assert_eq!(
            resource_fault_kind(
                "Caught DeviceLost error: Unknown Out of memory\n\
                 Quitting the application due to DeviceLost RenderError"
            ),
            Some("GPU out-of-memory"),
            "OOM-induced device teardown must use the bounded cooldown"
        );
        assert_eq!(
            resource_fault_kind(
                "Caught DeviceLost error: Unknown Device is lost\n\
                 Quitting the application due to DeviceLost RenderError"
            ),
            Some("GPU device loss")
        );
        assert_eq!(resource_fault_kind("ordinary compile error"), None);
    }

    #[test]
    fn stale_artifact_link_failures_are_recognized() {
        // The observed INC-20260724T182642Z signature.
        assert!(toolchain_corruption(
            "lld-link: error: undefined symbol: anon.7d2b28def3d478c7dd7908d0ff8f73af.72.llvm.11501338756853688745"
        ));
        assert!(toolchain_corruption(
            "error: undefined symbol: <bevy_window::monitor::PrimaryMonitor as Component>\
             ::register_required_components (.llvm.17986031716569828249)"
        ));
    }

    #[test]
    fn ordinary_build_errors_are_not_treated_as_corruption() {
        // A real code error must reach the agent verbatim, not trigger a purge
        // and a silent rebuild.
        assert!(!toolchain_corruption(
            "error[E0425]: cannot find value `foo` in this scope"
        ));
        assert!(!toolchain_corruption(
            "lld-link: error: undefined symbol: my_missing_extern_fn"
        ));
    }
}
