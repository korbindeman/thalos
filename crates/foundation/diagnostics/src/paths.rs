//! Canonical paths for disposable runtime diagnostics.
//!
//! Curated scene images live in `artifacts/visual/latest/`. Machine-readable logs
//! belong under `artifacts/diagnostics/`, while agents put ad-hoc image iterations
//! in `artifacts/visual/runs/`. Keeping the path policy here prevents each
//! diagnostic from drifting back to the repository root or into image folders.
//!
//! This module is process-agnostic on purpose: the game, the capture host, the
//! capture client, and the offline bakers all resolve diagnostic paths the same
//! way, so one `artifacts/diagnostics/` layout serves every producer.

use std::{
    env,
    ffi::OsStr,
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
};

const DIAGNOSTICS_DIR: &str = "artifacts/diagnostics";

/// Resolve an opt-in JSONL sink named by `env_key`.
///
/// A bare filename is placed in [`DIAGNOSTICS_DIR`]. Absolute paths and
/// relative paths with an explicit parent are honored as written, so callers
/// can still send a reproduction artifact somewhere specific.
pub fn jsonl_path_from_env(env_key: &str) -> Option<PathBuf> {
    env::var_os(env_key)
        .filter(|value| !value.is_empty())
        .map(resolve_jsonl_path)
}
/// Resolve an optional JSONL override, falling back to a canonical diagnostic
/// filename when the environment variable is unset.
pub fn jsonl_path_from_env_or(env_key: &str, default_filename: &str) -> PathBuf {
    jsonl_path_from_env(env_key).unwrap_or_else(|| default_jsonl_path(default_filename))
}

/// Return the canonical path for a diagnostic emitted without a path override.
pub fn default_jsonl_path(filename: &str) -> PathBuf {
    default_diagnostic_path(filename)
}

/// Return the canonical path for any machine-readable runtime artifact that is
/// not necessarily JSONL (for example the latest saved camera perspective).
pub fn default_diagnostic_path(filename: &str) -> PathBuf {
    Path::new(DIAGNOSTICS_DIR).join(filename)
}

/// The diagnostics directory itself, for callers that enumerate it.
pub fn diagnostics_dir() -> &'static Path {
    Path::new(DIAGNOSTICS_DIR)
}

/// Open a JSONL sink for append, creating its parent directory when needed.
pub fn open_jsonl_append(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    OpenOptions::new().create(true).append(true).open(path)
}

/// A rotated JSONL sink exceeds this size at process start.
const ROTATE_OVER_BYTES: u64 = 64 * 1024 * 1024;
/// Total budget for `artifacts/diagnostics/` top-level JSONL files (active +
/// rotated). Rotated files are pruned oldest-first back under this at boot.
const DIAGNOSTICS_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
/// Rotation marker in rotated filenames: `runtime.rot<unix_ms>.jsonl`.
const ROTATION_TAG: &str = ".rot";

/// Boot-time storage hygiene for the diagnostics directory.
///
/// Two passes over top-level `*.jsonl` files (subdirectories such as
/// `reports/` are left alone):
///
/// 1. **Rotate**: any file over [`ROTATE_OVER_BYTES`] is renamed to
///    `<stem>.rot<unix_ms>.jsonl`, so the active sink restarts empty while
///    the history stays readable by `just perf-report`.
/// 2. **Prune**: while the directory total exceeds
///    [`DIAGNOSTICS_BUDGET_BYTES`], delete the oldest **rotated** files.
///    Active sinks are never deleted — a stale-but-active file rotates on a
///    later boot and becomes prunable then.
///
/// Errors are ignored per file: several processes may boot concurrently (game,
/// capture host, and capture client share this directory) and lose a rename
/// race harmlessly.
pub fn rotate_and_prune_diagnostics() {
    let dir = Path::new(DIAGNOSTICS_DIR);
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    let mut rotated: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension() != Some(OsStr::new("jsonl")) {
            continue;
        }
        let Ok(md) = entry.metadata() else { continue };
        let mut len = md.len();
        let stem = path
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or_default()
            .to_string();
        let is_rotated = stem.contains(ROTATION_TAG);
        if !is_rotated && len > ROTATE_OVER_BYTES {
            let target = dir.join(format!("{stem}{ROTATION_TAG}{now_ms}.jsonl"));
            if fs::rename(&path, &target).is_ok() {
                rotated.push((target, len, md.modified().unwrap_or(std::time::UNIX_EPOCH)));
                total += len;
                continue;
            }
            // Rename lost a race or failed; count it where it is.
            len = md.len();
        }
        if is_rotated {
            rotated.push((path, len, md.modified().unwrap_or(std::time::UNIX_EPOCH)));
        }
        total += len;
    }

    if total <= DIAGNOSTICS_BUDGET_BYTES {
        return;
    }
    rotated.sort_by_key(|(_, _, modified)| *modified);
    for (path, len, _) in rotated {
        if total <= DIAGNOSTICS_BUDGET_BYTES {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(len);
        }
    }
}

fn resolve_jsonl_path(raw: impl AsRef<OsStr>) -> PathBuf {
    let path = PathBuf::from(raw.as_ref());
    if path.is_absolute() || path.components().count() > 1 {
        path
    } else {
        Path::new(DIAGNOSTICS_DIR).join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_filenames_resolve_under_the_diagnostics_directory() {
        assert_eq!(
            resolve_jsonl_path("grass.jsonl"),
            Path::new(DIAGNOSTICS_DIR).join("grass.jsonl")
        );
    }

    #[test]
    fn explicit_parents_are_honored_as_written() {
        assert_eq!(
            resolve_jsonl_path("target/grass.jsonl"),
            PathBuf::from("target/grass.jsonl")
        );
    }
}
