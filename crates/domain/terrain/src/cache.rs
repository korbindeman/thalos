//! On-disk store for `StaticSurfaceData`.
//!
//! Terrain generation for a single body can take tens of seconds. Local
//! bakes live at `target/bakes/<body>.bin`; the editor and the headless
//! `bake_dump` tool produce them. They are ignored by Git and are not the
//! release distribution path. The game never compiles terrain — it loads only.
//!
//! Each blob stores `(key, StaticSurfaceData)`. The key hashes generation
//! inputs plus a build-time signature of the `thalos_terrain` source
//! tree (see `crates/domain/terrain/build.rs`). On `load`, the stored key is
//! checked against the expected key and a mismatch is surfaced as
//! [`LoadError::HashMismatch`] so callers can choose between hard-error
//! (the game) and recompile-on-miss (tools).

use std::fmt;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::static_surface::StaticSurfaceData;
use crate::tectonics::TectonicConfig;
use crate::terrain_config::{TerrainCompileContext, TerrainCompileOptions, TerrainConfig};

// THLSBD02: uncompressed bincode. Local bakes are developer-only build
// artifacts in target/bakes/; zstd was buying ~10× on disk in exchange for
// ~500 ms of single-threaded decode at load. The previous magic
// (`THLSBD01`) was zstd-compressed; existing bakes will fail magic check,
// be flagged as corrupt by bake_check, and auto-rebaked on next launch.
const FORMAT_MAGIC: &[u8; 8] = b"THLSBD02";
// Bumped to 4 with the addition of tectonic-config hashing — bodies whose
// static surface depends on tectonics (currently AgingOceanicHomeworld) need
// the cache key to invalidate when their tectonic config changes.
const CACHE_KEY_VERSION: u32 = 4;
const SOURCE_HASH: &str = env!("THALOS_TERRAIN_SOURCE_HASH");

pub fn terrain_cache_key(
    terrain: &TerrainConfig,
    tectonics: Option<&TectonicConfig>,
    context: &TerrainCompileContext,
    options: TerrainCompileOptions,
) -> u64 {
    let mut h = cache_hasher();
    static_terrain_key_debug(terrain).hash(&mut h);
    static_tectonics_key_debug(tectonics).hash(&mut h);
    context.body_name.hash(&mut h);
    context.radius_m.to_bits().hash(&mut h);
    context.gravity_m_s2.to_bits().hash(&mut h);
    hash_optional_f32(&mut h, context.rotation_hours);
    hash_optional_f32(&mut h, context.obliquity_deg);
    context.axial_tilt_rad.to_bits().hash(&mut h);
    hash_optional_vec3(&mut h, context.tidal_axis);
    options.crater_count_scale.to_bits().hash(&mut h);
    match options.cubemap_resolution_override {
        None => 0u8.hash(&mut h),
        Some(resolution) => {
            1u8.hash(&mut h);
            resolution.hash(&mut h);
        }
    }
    h.finish()
}

fn static_terrain_key_debug(terrain: &TerrainConfig) -> String {
    let mut terrain = terrain.clone();
    if let TerrainConfig::Feature(feature) = &mut terrain {
        feature.ice_caps.clear();
        if let Some(style) = &mut feature.cold_desert_style {
            style.dune_regions.clear();
        }
    }
    format!("{terrain:?}")
}

/// Hashable debug form of the tectonic config. The tectonic *system* is
/// deterministic from `(config, body_radius, body_name)` — radius and body
/// name already enter the key via the context — so the config alone is
/// enough to capture changes that should invalidate cached terrain.
fn static_tectonics_key_debug(tectonics: Option<&TectonicConfig>) -> String {
    match tectonics {
        None => "None".to_string(),
        Some(cfg) => format!("{cfg:?}"),
    }
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

/// Bake file path: `<dir>/<sanitized body>.bin`.
///
/// Stable filename. The key lives inside the blob and is checked on
/// [`load`] — a mismatch is surfaced as [`LoadError::HashMismatch`] rather
/// than producing a separate file per input variant. Tools may overwrite
/// the file freely (bake regenerates it); the game errors on mismatch.
pub fn cache_path(dir: &Path, body_name: &str) -> PathBuf {
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
    dir.join(format!("{safe}.bin"))
}

// File layout after magic: bincode of (key, StaticSurfaceData), fixed-int LE.
// Two separate wrappers so `store` can borrow and `load` can own.
#[derive(Serialize)]
struct PayloadRef<'a> {
    key: u64,
    data: &'a StaticSurfaceData,
}

#[derive(Deserialize)]
struct PayloadOwned {
    key: u64,
    data: StaticSurfaceData,
}

/// Reason a [`load`] call did not return a usable [`StaticSurfaceData`].
///
/// The game treats every variant as fatal; tools may treat any variant as
/// "recompile and overwrite". Errors carry enough detail for an actionable
/// log message — the path of the bake file and, for [`HashMismatch`], both
/// the expected and stored keys.
#[derive(Debug)]
pub enum LoadError {
    /// The file does not exist (or could not be opened for reading).
    Missing { path: PathBuf },
    /// The file exists and decodes, but its stored key disagrees with the
    /// expected key. Indicates a stale bake: the body's terrain config or
    /// the `thalos_terrain` source tree has moved since the bake was
    /// produced.
    HashMismatch {
        path: PathBuf,
        expected: u64,
        found: u64,
    },
    /// The file exists but is corrupt, was written by a different format
    /// version, or otherwise failed to decode.
    Decode { path: PathBuf, message: String },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { path } => write!(f, "bake file not found: {}", path.display()),
            Self::HashMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "stale bake at {}: stored key {:016x}, expected {:016x}",
                path.display(),
                found,
                expected,
            ),
            Self::Decode { path, message } => {
                write!(f, "could not decode bake at {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Read only the stored cache key from a bake file, without decoding the
/// full `StaticSurfaceData`. Returns the same error variants as [`load`]
/// (`Missing` / `Decode`), but never `HashMismatch` — the caller compares
/// the returned key against its expected key.
///
/// Implementation note: `PayloadRef { key: u64, data: … }` serializes with
/// bincode's fixed-int LE encoding, so the key is the first 8 bytes of the
/// stream right after the magic. We open the file and read only those 16
/// bytes — no allocation of the full blob.
pub fn peek_key(path: &Path) -> Result<u64, LoadError> {
    use std::io::Read as _;
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(LoadError::Missing {
                path: path.to_path_buf(),
            });
        }
        Err(e) => {
            return Err(LoadError::Decode {
                path: path.to_path_buf(),
                message: format!("open failed: {e}"),
            });
        }
    };
    let mut header = [0u8; FORMAT_MAGIC.len() + 8];
    file.read_exact(&mut header)
        .map_err(|e| LoadError::Decode {
            path: path.to_path_buf(),
            message: format!("read failed: {e}"),
        })?;
    if &header[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return Err(LoadError::Decode {
            path: path.to_path_buf(),
            message: "magic mismatch (not a Thalos bake file)".into(),
        });
    }
    let key_bytes: [u8; 8] = header[FORMAT_MAGIC.len()..].try_into().unwrap();
    Ok(u64::from_le_bytes(key_bytes))
}

/// Load a `StaticSurfaceData` blob from `path` and validate its stored key
/// against `expected_key`. See [`LoadError`] for failure modes.
pub fn load(path: &Path, expected_key: u64) -> Result<StaticSurfaceData, LoadError> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(LoadError::Missing {
                path: path.to_path_buf(),
            });
        }
        Err(e) => {
            return Err(LoadError::Decode {
                path: path.to_path_buf(),
                message: format!("read failed: {e}"),
            });
        }
    };
    if bytes.len() < FORMAT_MAGIC.len() || &bytes[..FORMAT_MAGIC.len()] != FORMAT_MAGIC {
        return Err(LoadError::Decode {
            path: path.to_path_buf(),
            message: "magic mismatch (not a Thalos bake file)".into(),
        });
    }
    let (payload, _): (PayloadOwned, usize) =
        bincode::serde::decode_from_slice(&bytes[FORMAT_MAGIC.len()..], bincode_config()).map_err(
            |e| LoadError::Decode {
                path: path.to_path_buf(),
                message: format!("bincode decode failed: {e}"),
            },
        )?;
    if payload.key != expected_key {
        return Err(LoadError::HashMismatch {
            path: path.to_path_buf(),
            expected: expected_key,
            found: payload.key,
        });
    }
    Ok(payload.data)
}

/// Write `data` to `path`. Creates parent directories; writes atomically
/// via a `.tmp` rename.
pub fn store(path: &Path, key: u64, data: &StaticSurfaceData) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = PayloadRef { key, data };
    let encoded = bincode::serde::encode_to_vec(&payload, bincode_config())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let tmp = path.with_extension("bin.tmp");
    let mut out = Vec::with_capacity(FORMAT_MAGIC.len() + encoded.len());
    out.extend_from_slice(FORMAT_MAGIC);
    out.extend_from_slice(&encoded);
    fs::write(&tmp, out)?;
    fs::rename(tmp, path)
}

fn bincode_config()
-> bincode::config::Configuration<bincode::config::LittleEndian, bincode::config::Fixint> {
    bincode::config::standard().with_fixed_int_encoding()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cold_desert_field::ColdDesertStyle;
    use crate::feature_compiler::{
        AtmosphereSpec, BodyArchetype, CompositionClass, HydrosphereSpec, IceInventory,
        TerrainIntent,
    };
    use crate::terrain_config::{
        FeatureEnvironmentConfig, FeatureTerrainConfig, TerrainCompileContext,
        TerrainCompileOptions, TerrainConfig,
    };
    use crate::types::IceCapSpec;

    fn context() -> TerrainCompileContext {
        TerrainCompileContext {
            body_name: "Vaelen".to_string(),
            radius_m: 1_130_000.0,
            gravity_m_s2: 3.2,
            rotation_hours: Some(32.0),
            obliquity_deg: Some(9.0),
            tidal_axis: None,
            axial_tilt_rad: 9.0_f32.to_radians(),
        }
    }

    fn feature_config() -> FeatureTerrainConfig {
        FeatureTerrainConfig {
            seed: 42,
            cubemap_resolution: Some(64),
            body_age_gyr: 4.1,
            archetype: BodyArchetype::ColdDesertFormerlyWet,
            composition: CompositionClass::BasalticSilicate,
            environment: FeatureEnvironmentConfig {
                stellar_flux_earth: 0.7,
                atmosphere: AtmosphereSpec::ThinCo2 { pressure_bar: 0.08 },
                hydrosphere: HydrosphereSpec::AncientLost,
                ice_inventory: IceInventory::Trace,
            },
            intent: vec![TerrainIntent::VisibleAncientWaterStory],
            projection: Default::default(),
            ice_caps: Vec::new(),
            cold_desert_style: Some(ColdDesertStyle::default()),
            authored_features: Vec::new(),
        }
    }

    #[test]
    fn static_cache_key_ignores_dynamic_ice_definitions() {
        let context = context();
        let mut with_ice = feature_config();
        with_ice.ice_caps.push(IceCapSpec::default());

        let base = TerrainConfig::Feature(feature_config());
        let changed = TerrainConfig::Feature(with_ice);

        assert_eq!(
            terrain_cache_key(&base, None, &context, TerrainCompileOptions::default()),
            terrain_cache_key(&changed, None, &context, TerrainCompileOptions::default())
        );
    }

    /// `peek_key` assumes bincode (with fixed-int LE encoding) writes
    /// `PayloadRef`'s `key: u64` as 8 LE bytes at the start of the
    /// serialized payload. Pin that assumption — if a future bincode bump
    /// breaks it, the staleness check would read garbage and silently
    /// re-bake everything (or worse, accept a stale bake).
    #[test]
    fn bincode_u64_encodes_as_eight_le_bytes_at_offset_zero() {
        let key: u64 = 0x0123_4567_89AB_CDEF;
        let encoded = bincode::serde::encode_to_vec(key, bincode_config()).unwrap();
        assert_eq!(&encoded[..8], &key.to_le_bytes());
    }

    #[test]
    fn static_cache_key_ignores_active_dune_definitions() {
        let context = context();
        let mut changed = feature_config();
        changed
            .cold_desert_style
            .as_mut()
            .expect("test style")
            .dune_regions
            .clear();

        let base = TerrainConfig::Feature(feature_config());
        let changed = TerrainConfig::Feature(changed);

        assert_eq!(
            terrain_cache_key(&base, None, &context, TerrainCompileOptions::default()),
            terrain_cache_key(&changed, None, &context, TerrainCompileOptions::default())
        );
    }
}
