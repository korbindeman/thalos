//! On-disk cache for `StaticSurfaceData`.
//!
//! Terrain generation for a single body can take tens of seconds. This
//! module persists a finished `StaticSurfaceData` blob and loads it on subsequent
//! runs when the inputs are unchanged.
//!
//! Cache validity is decided by key only: the key hashes generation inputs plus
//! a build-time signature of the terrain_gen source tree. In development, code
//! edits automatically move bakes to a new cache key while unchanged inputs can
//! still reuse cached `StaticSurfaceData`.

use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::static_surface::StaticSurfaceData;
use crate::tectonics::TectonicConfig;
use crate::terrain_config::{TerrainCompileContext, TerrainCompileOptions, TerrainConfig};

const FORMAT_MAGIC: &[u8; 8] = b"THLSBD01";
// Bumped to 4 with the addition of tectonic-config hashing — bodies whose
// static surface depends on tectonics (currently AgingOceanicHomeworld) need
// the cache key to invalidate when their tectonic config changes.
const CACHE_KEY_VERSION: u32 = 4;
const ZSTD_LEVEL: i32 = 3;
const SOURCE_HASH: &str = env!("THALOS_TERRAIN_GEN_SOURCE_HASH");

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

// File layout after magic: zstd-compressed bincode of (key, StaticSurfaceData).
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

/// Try to load a cached `StaticSurfaceData` from `path`. Returns `None` for any
/// failure (missing file, wrong magic, key mismatch, decode error).
pub fn load(path: &Path, key: u64) -> Option<StaticSurfaceData> {
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
pub fn store(path: &Path, key: u64, data: &StaticSurfaceData) -> io::Result<()> {
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
