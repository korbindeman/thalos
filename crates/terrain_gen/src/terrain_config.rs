//! Asset-facing terrain schema and shared compile entry point.

use glam::Vec3;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::body_data::BodyData;
use crate::cubemap::CubemapFace;
use crate::feature_compiler::{
    AtmosphereSpec, AuthoredFeatureSpec, BodyArchetype, CompositionClass, FeatureCompileError,
    FeatureCompileOptions, FeatureFootprint, FeatureId, FeatureKind, FeatureLock, FeatureParam,
    FeatureProjectionConfig, FeatureSeed, HydrosphereSpec, IceInventory, PlanetPhysicalSpec,
    PlanetTerrainSpec, ScaleRangeM, TerrainIntent, compile_initial_body_data,
};
use crate::surface_field::quantize_unit_to_u8;
use crate::types::Composition;

#[derive(Clone, Debug, Default, Deserialize)]
pub enum TerrainConfig {
    #[default]
    None,
    Feature(FeatureTerrainConfig),
    Ocean(OceanTerrainConfig),
}

impl TerrainConfig {
    pub fn is_some(&self) -> bool {
        !matches!(self, Self::None)
    }

    pub fn route_label(&self) -> String {
        match self {
            Self::None => "None".to_string(),
            Self::Feature(config) => format!("Feature({:?})", config.archetype),
            Self::Ocean(_) => "Ocean".to_string(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct FeatureTerrainConfig {
    pub seed: u64,
    pub cubemap_resolution: u32,
    pub body_age_gyr: f32,
    pub archetype: BodyArchetype,
    pub composition: CompositionClass,
    pub environment: FeatureEnvironmentConfig,
    pub intent: Vec<TerrainIntent>,
    #[serde(default)]
    pub projection: FeatureProjectionConfig,
    #[serde(default)]
    pub authored_features: Vec<AuthoredFeatureConfig>,
}

/// Flat-water placeholder. The compiled `BodyData` has zero height
/// everywhere and `sea_level_m` set to a small positive value, so the
/// impostor's water BRDF fires for the entire surface.
#[derive(Clone, Debug, Deserialize)]
pub struct OceanTerrainConfig {
    pub seed: u64,
    pub cubemap_resolution: u32,
    /// sRGB linear seabed albedo. Only visible through shallow water; deep
    /// water is dominated by the shader's absorption tint.
    pub seabed_albedo: [f32; 3],
    /// Water surface roughness for the impostor PBR term. 0.04 ≈ flat
    /// open ocean; raise to introduce wave-scale microsurface.
    pub water_roughness: f32,
    /// Sea level above the (flat) heightfield. Any positive value works;
    /// 1.0 m is the convention.
    pub sea_level_m: f32,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct FeatureEnvironmentConfig {
    pub stellar_flux_earth: f32,
    pub atmosphere: AtmosphereSpec,
    pub hydrosphere: HydrosphereSpec,
    pub ice_inventory: IceInventory,
}

#[derive(Clone, Debug, Deserialize)]
pub enum AuthoredFeatureConfig {
    Megabasin(MegabasinFeatureConfig),
}

#[derive(Clone, Debug, Deserialize)]
pub struct MegabasinFeatureConfig {
    pub id: FeatureId,
    #[serde(default)]
    pub parent: Option<FeatureId>,
    pub center_dir: Vec3,
    pub radius_km: f32,
    pub depth_km: f32,
    #[serde(default)]
    pub ring_count: Option<u32>,
    #[serde(default)]
    pub seed: Option<FeatureSeed>,
    pub lock: FeatureLock,
}

#[derive(Clone, Debug)]
pub struct TerrainCompileContext {
    pub body_name: String,
    pub radius_m: f32,
    pub gravity_m_s2: f32,
    pub rotation_hours: Option<f32>,
    pub obliquity_deg: Option<f32>,
    pub tidal_axis: Option<Vec3>,
    pub axial_tilt_rad: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct TerrainCompileOptions {
    pub crater_count_scale: f32,
}

impl Default for TerrainCompileOptions {
    fn default() -> Self {
        Self {
            crater_count_scale: 1.0,
        }
    }
}

#[derive(Debug)]
pub enum TerrainCompileError {
    UnsupportedNone,
    Feature(FeatureCompileError),
}

impl std::fmt::Display for TerrainCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedNone => write!(f, "body has no terrain config"),
            Self::Feature(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for TerrainCompileError {}

impl From<FeatureCompileError> for TerrainCompileError {
    fn from(value: FeatureCompileError) -> Self {
        Self::Feature(value)
    }
}

pub fn compile_terrain_config(
    terrain: &TerrainConfig,
    context: &TerrainCompileContext,
    options: TerrainCompileOptions,
) -> Result<BodyData, TerrainCompileError> {
    match terrain {
        TerrainConfig::None => Err(TerrainCompileError::UnsupportedNone),
        TerrainConfig::Feature(feature) => {
            let spec = feature.to_planet_spec(context);
            compile_initial_body_data(
                &spec,
                FeatureCompileOptions {
                    cubemap_resolution: feature.cubemap_resolution,
                    crater_count_scale: options.crater_count_scale,
                    projection: feature.projection.clone(),
                },
            )
            .map_err(Into::into)
        }
        TerrainConfig::Ocean(config) => Ok(compile_ocean(config, context)),
    }
}

fn compile_ocean(config: &OceanTerrainConfig, context: &TerrainCompileContext) -> BodyData {
    let mut builder = BodyBuilder::new(
        context.radius_m,
        config.seed,
        // Composition is irrelevant for a flat-ocean placeholder — no
        // stage reads it. Pick a neutral value.
        Composition::new(1.0, 0.0, 0.0, 0.0, 0.0),
        config.cubemap_resolution,
        4.5,
        context.tidal_axis,
        context.axial_tilt_rad,
    );

    // Seabed albedo: linear RGB written into every accumulator texel with
    // alpha = 1 so `finalize_albedo` divides through cleanly and converts
    // to sRGB. Only visible through shallow water; deep water is dominated
    // by the impostor's absorption tint.
    let [r, g, b] = config.seabed_albedo;
    for face in CubemapFace::ALL {
        for v in builder.albedo_contributions.albedo.face_data_mut(face) {
            *v = [r, g, b, 1.0];
        }
    }

    let roughness_texel = quantize_unit_to_u8(config.water_roughness.clamp(0.0, 1.0));
    for face in CubemapFace::ALL {
        for v in builder.roughness_cubemap.face_data_mut(face) {
            *v = roughness_texel;
        }
    }

    builder.sea_level_m = Some(config.sea_level_m);
    builder.build()
}

impl FeatureTerrainConfig {
    pub fn to_planet_spec(&self, context: &TerrainCompileContext) -> PlanetTerrainSpec {
        PlanetTerrainSpec {
            body_id: context.body_name.to_ascii_lowercase(),
            root_seed: self.seed,
            physical: PlanetPhysicalSpec {
                radius_m: context.radius_m,
                gravity_m_s2: context.gravity_m_s2,
                age_gyr: self.body_age_gyr,
                stellar_flux_earth: self.environment.stellar_flux_earth,
                rotation_hours: context.rotation_hours,
                obliquity_deg: context.obliquity_deg,
                atmosphere: self.environment.atmosphere,
                hydrosphere: self.environment.hydrosphere,
                ice_inventory: self.environment.ice_inventory,
                composition: self.composition,
            },
            archetype: self.archetype,
            intent: self.intent.clone(),
            authored_features: self
                .authored_features
                .iter()
                .map(AuthoredFeatureConfig::to_spec)
                .collect(),
        }
    }
}

impl AuthoredFeatureConfig {
    fn to_spec(&self) -> AuthoredFeatureSpec {
        match self {
            Self::Megabasin(config) => {
                let mut params = vec![
                    FeatureParam::number("radius_km", config.radius_km),
                    FeatureParam::number("depth_km", config.depth_km),
                    FeatureParam::boolean("authored_landmark", true),
                ];
                if let Some(ring_count) = config.ring_count {
                    params.push(FeatureParam::number("ring_count", ring_count as f32));
                }

                AuthoredFeatureSpec {
                    id: config.id.clone(),
                    kind: FeatureKind::Megabasin,
                    parent: config.parent.clone(),
                    seed_override: config.seed,
                    footprint: Some(FeatureFootprint::Circle {
                        center: config.center_dir.normalize(),
                        angular_radius_rad: 0.0,
                    }),
                    scale_range_m: Some(ScaleRangeM::new(
                        config.radius_km * 1_000.0 * 0.08,
                        config.radius_km * 1_000.0 * 2.6,
                    )),
                    params,
                    lock: config.lock,
                }
            }
        }
    }
}
