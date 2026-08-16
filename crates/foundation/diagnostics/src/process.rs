//! In-process memory introspection for the runtime gauges.
//!
//! The capture client already polls the host's working set *from outside*
//! (`tools/capture`, the `THALOS_CAPTURE_RSS_LIMIT_MB` watchdog) — but when
//! that watchdog kills a host, the post-mortem needs the *inside* view: what
//! the process itself thought it was holding when it crossed the limit. A host
//! died at 8.1 GiB RSS while every GPU-side gauge (tile residency, mesh slabs)
//! summed to ~2 GiB, and nothing in the lane measured the CPU side — this
//! helper exists so `frame_gauge` carries `rss_mib` and the OOM run localizes
//! itself.

/// Resident set (Windows: working set) of the current process, in bytes.
///
/// Cheap enough for a periodic gauge: one kernel query, no allocation on
/// Windows; one small `/proc` read elsewhere.
pub fn self_resident_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        unsafe {
            let mut counters = PROCESS_MEMORY_COUNTERS {
                cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                ..std::mem::zeroed()
            };
            let cb = counters.cb;
            (K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, cb) != 0)
                .then_some(counters.WorkingSetSize as u64)
        }
    }
    #[cfg(target_os = "macos")]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v2>::uninit();
        let result = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V2,
                usage.as_mut_ptr().cast(),
            )
        };
        (result == 0).then(|| unsafe { usage.assume_init().ri_resident_size })
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let kib = status
            .lines()
            .find_map(|line| line.strip_prefix("VmRSS:"))?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?;
        Some(kib * 1024)
    }
    #[cfg(not(any(
        windows,
        target_os = "macos",
        target_os = "linux",
        target_os = "android"
    )))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(
        windows,
        target_os = "macos",
        target_os = "linux",
        target_os = "android"
    ))]
    #[test]
    fn reports_a_plausible_resident_set() {
        let bytes = super::self_resident_bytes().expect("own-process RSS is always queryable");
        // A running test binary holds at least a megabyte and far less than a
        // terabyte; the bound only has to catch unit mistakes (KiB vs bytes).
        assert!(bytes > 1024 * 1024, "{bytes} bytes is implausibly small");
        assert!(bytes < 1 << 40, "{bytes} bytes is implausibly large");
    }
}
