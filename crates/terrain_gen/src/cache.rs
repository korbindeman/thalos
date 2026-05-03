//! On-disk cache for `BodyData`.
//!
//! Terrain generation for a single body can take tens of seconds. This
//! module persists a finished `BodyData` blob and loads it on subsequent
//! runs when the inputs are unchanged.
//!
//! Cache validity is decided by key only: the key hashes generation inputs plus
//! a build-time signature of the terrain_gen source tree. In development, code
//! edits automatically move bakes to a new cache key while unchanged inputs can
//! still reuse cached `BodyData`.

use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::body_data::BodyData;
use crate::generator::GeneratorParams;
use crate::terrain_config::{TerrainCompileContext, TerrainCompileOptions, TerrainConfig};

const FORMAT_MAGIC: &[u8; 8] = b"THLSBD01";
const CACHE_KEY_VERSION: u32 = 2;
const ZSTD_LEVEL: i32 = 3;
const SOURCE_HASH: &str = env!("THALOS_TERRAIN_GEN_SOURCE_HASH");

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
    cache_key_from_source(&format!("{params:?}"), radius_m, tidal_axis, axial_tilt_rad)
}

pub fn terrain_cache_key(
    terrain: &TerrainConfig,
    context: &TerrainCompileContext,
    options: TerrainCompileOptions,
) -> u64 {
    let mut h = cache_hasher();
    format!("{terrain:?}").hash(&mut h);
    context.body_name.hash(&mut h);
    context.radius_m.to_bits().hash(&mut h);
    context.gravity_m_s2.to_bits().hash(&mut h);
    hash_optional_f32(&mut h, context.rotation_hours);
    hash_optional_f32(&mut h, context.obliquity_deg);
    context.axial_tilt_rad.to_bits().hash(&mut h);
    hash_optional_vec3(&mut h, context.tidal_axis);
    options.crater_count_scale.to_bits().hash(&mut h);
    h.finish()
}

pub fn cache_key_from_source(
    source: &str,
    radius_m: f32,
    tidal_axis: Option<Vec3>,
    axial_tilt_rad: f32,
) -> u64 {
    let mut h = cache_hasher();
    source.hash(&mut h);
    radius_m.to_bits().hash(&mut h);
    axial_tilt_rad.to_bits().hash(&mut h);
    hash_optional_vec3(&mut h, tidal_axis);
    h.finish()
}

fn cache_hasher() -> std::collections::hash_map::DefaultHasher {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    CACHE_KEY_VERSION.hash(&mut h);
    SOURCE_HASH.hash(&mut h);
    h
}

fn hash_optional_f32(h: &mut impl Hasher, value: Option<f32>) {
    match value {
        None => 0u8.hash(h),
        Some(value) => {
            1u8.hash(h);
            value.to_bits().hash(h);
        }
    }
}

fn hash_optional_vec3(h: &mut impl Hasher, value: Option<Vec3>) {
    match value {
        None => 0u8.hash(h),
        Some(v) => {
            1u8.hash(h);
            v.x.to_bits().hash(h);
            v.y.to_bits().hash(h);
            v.z.to_bits().hash(h);
        }
    }
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
