//! Cross-process share of the tile residency budget.
//!
//! [`super::residency_budget_bytes`] is a **whole-machine** figure: the total
//! tile-mesh VRAM every Thalos renderer on this box may hold between them. This
//! module answers the only question needed to turn that into a per-process
//! budget — *how many of us are there right now?*
//!
//! # Why this exists
//!
//! INC-20260725T012104Z gave the tile renderer a byte budget, and it worked as
//! designed. It still did not prevent the second `DeviceLost` (2026-07-25
//! 20:08 UTC), because the budget was **per process** and sized as if the
//! process owned the card. Two `just game` instances were live for the eleven
//! minutes up to the crash (pids 10560 + 8376, dying three seconds apart);
//! neither ever braked — `split_scale` stayed at 1.00 in both — because each was
//! individually well inside its own 4 GiB while jointly they were entitled to
//! 8 GiB of tile meshes alone on a 12 GB card.
//!
//! A budget that each participant reads as if it were alone is not a budget.
//!
//! # Why not read the card's VRAM instead
//!
//! Because it does not answer the question. Sizing from real VRAM still hands
//! *each* instance the same fraction, so two instances still overcommit by 2×;
//! and the number is not portably available anyway — `wgpu` 29 exposes no total-
//! memory field, and Windows' WMI `AdapterRAM` is a 32-bit field that reports
//! 4 GB for the 12 GB card this incident happened on. Counting peers is both
//! cheaper and the thing that was actually wrong.
//!
//! # Mechanism
//!
//! Each renderer process touches `<dir>/<pid>.live` every [`HEARTBEAT_S`]; a
//! peer counts as live while its file's mtime is younger than [`STALE_AFTER_S`].
//! Timestamps rather than locks so this stays dependency-free and portable, and
//! so a process that *aborts* — which is exactly how the OOM ends, with no
//! unwinding and no destructor — releases its share on a timer instead of
//! wedging the budget for every later run. Files are best-effort deleted on
//! drop; staleness is the real correctness argument.
//!
//! The directory is the system temp dir by default (`THALOS_INSTANCE_DIR`
//! overrides). Any filesystem error means we count ourselves alone: the failure
//! mode of this module must be "budget behaves exactly as it did before", never
//! "the ground stops refining".

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

/// How often a live process refreshes its own marker.
const HEARTBEAT_S: u64 = 2;
/// A marker older than this belongs to a process that died without cleaning up.
/// Five heartbeats: long enough that a stalled frame (shader compile, a cold
/// tile burst) never makes a live instance look dead to its peers, short enough
/// that the survivor of a crash reclaims the full budget within seconds.
const STALE_AFTER_S: u64 = 10;
/// How long a peer count is reused before rescanning. The budget is read every
/// frame; the directory is not.
const RESCAN_S: u64 = 2;

fn instance_dir() -> PathBuf {
    match std::env::var_os("THALOS_INSTANCE_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => std::env::temp_dir().join("thalos-instances"),
    }
}

/// Our own marker file, created on first use and refreshed by [`live_instances`].
struct Marker(PathBuf);

impl Drop for Marker {
    fn drop(&mut self) {
        // Best-effort: an aborting process never runs this, which is why
        // staleness — not this — is what makes the count correct.
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Number of Thalos renderer processes currently live on this machine,
/// including this one. Never zero.
///
/// Cheap to call every frame: the directory is rescanned at most every
/// [`RESCAN_S`] seconds and the result cached.
pub fn live_instances() -> usize {
    use std::sync::{Mutex, OnceLock};

    struct Cached {
        marker: Option<Marker>,
        count: usize,
        checked: SystemTime,
        beat: SystemTime,
    }

    static STATE: OnceLock<Mutex<Cached>> = OnceLock::new();
    let epoch = SystemTime::UNIX_EPOCH;
    let state = STATE.get_or_init(|| {
        Mutex::new(Cached {
            marker: None,
            count: 1,
            checked: epoch,
            beat: epoch,
        })
    });
    let Ok(mut cached) = state.lock() else {
        return 1;
    };

    let now = SystemTime::now();
    let elapsed = |since: SystemTime| now.duration_since(since).unwrap_or(Duration::ZERO);

    if cached.marker.is_some() && elapsed(cached.checked) < Duration::from_secs(RESCAN_S) {
        return cached.count;
    }
    cached.checked = now;

    let dir = instance_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return 1;
    }

    // Claim (once) and refresh (periodically) our own marker. Rewriting the file
    // is the portable way to bump mtime — `filetime`-style APIs are not in std.
    let own = dir.join(format!("{}.live", std::process::id()));
    if cached.marker.is_none() {
        if std::fs::write(&own, b"thalos").is_err() {
            return 1;
        }
        cached.marker = Some(Marker(own.clone()));
        cached.beat = now;
    } else if elapsed(cached.beat) >= Duration::from_secs(HEARTBEAT_S) {
        let _ = std::fs::write(&own, b"thalos");
        cached.beat = now;
    }

    cached.count = scan(&dir, now).max(1);
    cached.count
}

/// Count live markers in `dir`, deleting the stale ones on the way past.
fn scan(dir: &std::path::Path, now: SystemTime) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 1;
    };
    let mut live = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("live") {
            continue;
        }
        let fresh = entry
            .metadata()
            .and_then(|m| m.modified())
            .map(|m| {
                now.duration_since(m).unwrap_or(Duration::ZERO) < Duration::from_secs(STALE_AFTER_S)
            })
            .unwrap_or(false);
        if fresh {
            live += 1;
        } else {
            // A crashed instance's marker. Removing it here is what keeps the
            // survivor from paying for a process that no longer exists.
            let _ = std::fs::remove_file(&path);
        }
    }
    live
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("thalos-vram-share-test-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create test dir");
        dir
    }

    #[test]
    fn a_lone_marker_counts_as_one_instance() {
        let dir = temp_dir("lone");
        std::fs::write(dir.join("1.live"), b"x").unwrap();
        assert_eq!(scan(&dir, SystemTime::now()), 1);
    }

    #[test]
    fn concurrent_markers_are_all_counted() {
        let dir = temp_dir("pair");
        std::fs::write(dir.join("1.live"), b"x").unwrap();
        std::fs::write(dir.join("2.live"), b"x").unwrap();
        // The exact case this module exists for: two live renderers must see
        // each other, so each takes half the machine budget.
        assert_eq!(scan(&dir, SystemTime::now()), 2);
    }

    #[test]
    fn a_stale_marker_is_ignored_and_reaped() {
        let dir = temp_dir("stale");
        let stale = dir.join("999.live");
        std::fs::write(&stale, b"x").unwrap();
        std::fs::write(dir.join("2.live"), b"x").unwrap();
        // Pretend "now" is well past the staleness window rather than sleeping.
        let later = SystemTime::now() + Duration::from_secs(STALE_AFTER_S + 5);
        // Both look stale at `later`, so the survivor floor (max(1)) is what the
        // caller sees; the point here is that the files are reaped.
        assert_eq!(scan(&dir, later), 0);
        assert!(!stale.exists(), "a crashed instance's marker is removed");
    }

    #[test]
    fn non_marker_files_are_not_counted() {
        let dir = temp_dir("junk");
        std::fs::write(dir.join("1.live"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        assert_eq!(scan(&dir, SystemTime::now()), 1);
    }
}
