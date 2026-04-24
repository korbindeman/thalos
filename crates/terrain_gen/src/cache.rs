//! On-disk cache for `BodyData`.
//!
//! Terrain generation for a single body can take tens of seconds. This
//! module persists a finished `BodyData` blob and loads it on subsequent
//! runs when the inputs are unchanged.
//!
//! Cache validity is decided by key only: the key hashes every input the
//! pipeline reads (`GeneratorParams` plus the builder-level fields
//! `radius_m`, `tidal_axis`, `axial_tilt_rad`). Changes to stage *code*
//! are NOT detected — wipe the cache directory when stage semantics
//! change (`just clear-terrain-cache`).

use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::body_data::BodyData;
use crate::generator::GeneratorParams;

const FORMAT_MAGIC: &[u8; 8] = b"THLSBD01";
const ZSTD_LEVEL: i32 = 3;

/// Deterministic hash of the inputs that produce a given `BodyData`.
///
/// Uses `Debug` formatting of `GeneratorParams` to cover all stage
/// params without requiring `Hash`/`Serialize` derives across every
/// stage struct. `Debug` output from `derive(Debug)` is stable in
/// practice; a mismatch just misses the cache.
pub fn cache_key(
    params: &GeneratorParams,
    radius_m: f32,
    tidal_axis: Option<Vec3>,
    axial_tilt_rad: f32,
) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{params:?}").hash(&mut h);
    radius_m.to_bits().hash(&mut h);
    axial_tilt_rad.to_bits().hash(&mut h);
    match tidal_axis {
        None => 0u8.hash(&mut h),
        Some(v) => {
            1u8.hash(&mut h);
            v.x.to_bits().hash(&mut h);
            v.y.to_bits().hash(&mut h);
            v.z.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

/// Cache file path: `<dir>/<sanitized body>-<hex key>.bin`.
pub fn cache_path(dir: &Path, body_name: &str, key: u64) -> PathBuf {
    let safe: String = body_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    dir.join(format!("{safe}-{key:016x}.bin"))
}

// File layout after magic: zstd-compressed bincode of (key, BodyData).
// Two separate wrappers so `store` can borrow and `load` can own.
#[derive(Serialize)]
struct PayloadRef<'a> {
    key: u64,
    data: &'a BodyData,
}

#[derive(Deserialize)]
struct PayloadOwned {
    key: u64,
    data: BodyData,
}

/// Try to load a cached `BodyData` from `path`. Returns `None` for any
/// failure (missing file, wrong magic, key mismatch, decode error).
pub fn load(path: &Path, key: u64) -> Option<BodyData> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < FORMAT_MAGIC.len() || &bytes[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return None;
    }
    let decompressed = zstd::decode_all(&bytes[FORMAT_MAGIC.len()..]).ok()?;
    let (payload, _): (PayloadOwned, usize) =
        bincode::serde::decode_from_slice(&decompressed, bincode_config()).ok()?;
    if payload.key != key {
        return None;
    }
    Some(payload.data)
}

/// Write `data` to `path`. Creates parent directories; writes atomically
/// via a `.tmp` rename.
pub fn store(path: &Path, key: u64, data: &BodyData) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = PayloadRef { key, data };
    let encoded = bincode::serde::encode_to_vec(&payload, bincode_config())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let compressed = zstd::encode_all(&encoded[..], ZSTD_LEVEL)?;
    let tmp = path.with_extension("bin.tmp");
    let mut out = Vec::with_capacity(FORMAT_MAGIC.len() + compressed.len());
    out.extend_from_slice(FORMAT_MAGIC);
    out.extend_from_slice(&compressed);
    fs::write(&tmp, out)?;
    fs::rename(tmp, path)
}

fn bincode_config()
-> bincode::config::Configuration<bincode::config::LittleEndian, bincode::config::Fixint> {
    bincode::config::standard().with_fixed_int_encoding()
}
