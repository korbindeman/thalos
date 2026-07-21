//! Canonical paths for disposable runtime diagnostics.
//!
//! Curated scene images live in `artifacts/visual/latest/`. Machine-readable logs
//! belong under `artifacts/diagnostics/`, while agents put ad-hoc image iterations
//! in `artifacts/visual/runs/`. Keeping the path policy here prevents each
//! diagnostic from drifting back to the repository root or into image folders.

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

/// Open a JSONL sink for append, creating its parent directory when needed.
pub fn open_jsonl_append(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    OpenOptions::new().create(true).append(true).open(path)
}

fn resolve_jsonl_path(raw: impl AsRef<OsStr>) -> PathBuf {
    let path = PathBuf::from(raw.as_ref());
    if path.is_absolute() || path.components().count() > 1 {
        path
    } else {
        Path::new(DIAGNOSTICS_DIR).join(path)
    }
}
