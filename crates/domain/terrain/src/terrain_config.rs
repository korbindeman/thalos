//! Asset-facing terrain schema and shared compile entry point.

use glam::Vec3;
use serde::Deserialize;

use crate::body_builder::BodyBuilder;
use crate::cold_desert_field::ColdDesertStyle;
use crate::cubemap::CubemapFace;
use crate::feature_compiler::{
    AtmosphereSpec, AuthoredFeatureSpec, BodyArchetype, CompositionClass, FeatureCompileError,
    FeatureCompileOptions, FeatureFootprint, FeatureId, FeatureKind, FeatureLock, FeatureParam,
    FeatureProjectionConfig, FeatureSeed, HydrosphereSpec, IceInventory, PlanetPhysicalSpec,
    PlanetTerrainSpec, ScaleRangeM, TerrainIntent, compile_initial_static_surface,
    dynamic_surface_layers_for,
};
use crate::seeding::sub_seed;
use crate::static_surface::{PlanetSurface, StaticSurfaceData};
use crate::surface_color::{SurfaceColorSpec, WaterAppearance, paint_surface_albedo};
use crate::surface_field::quantize_unit_to_u8;
use crate::tectonics::{TectonicConfig, TectonicSystem};
use crate::types::{Composition, DynamicSurfaceLayers, IceCapSpec};

#[derive(Clone, Debug, Default, Deserialize)]
#[allow(clippy::large_enum_variant)]
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

    /// Sea-level datum — metres above the body's reference radius — at which a
    /// liquid ocean surface should be rendered, or `None` for a body with no
    /// current hydrosphere (airless, ancient-dry).
    ///
    /// The runtime [`crate::ProceduralSurface`] generator pins the shoreline at
    /// the reference radius (height 0), so an ocean-bearing feature body floods
    /// at the constant **0 m** datum: the authored ocean fraction is a
    /// generation *intent* that shapes how much land sits above that line, not a
    /// separate sea level. The flat-water [`OceanTerrainConfig`] placeholder
    /// keeps its own authored `sea_level_m`.
    ///
    /// This is the renderer's single source of truth for "does this body have an
    /// ocean, and where is its surface" — both the in-game ground-LOD water
    /// shell and (later) the orbital impostor read it.
    pub fn ocean_sea_level_m(&self) -> Option<f32> {
        match self {
            Self::None => None,
            Self::Feature(config) => match config.environment.hydrosphere {
                HydrosphereSpec::OceanFraction(fraction) if fraction > 0.0 => Some(0.0),
                _ => None,
            },
            Self::Ocean(config) => Some(config.sea_level_m),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct FeatureTerrainConfig {
    pub seed: u64,
    /// Authored override for the cubemap face resolution. When omitted, the
    /// pipeline derives one from body radius via
    /// [`crate::cubemap::default_resolution`] (constant ~m/equator-texel
    /// budget, clamped to 4096). Set this only for art reasons that justify
    /// deviating from the radius-proportional default.
    #[serde(default)]
    pub cubemap_resolution: Option<u32>,
    pub body_age_gyr: f32,
    pub archetype: BodyArchetype,
    pub composition: CompositionClass,
    pub environment: FeatureEnvironmentConfig,
    pub intent: Vec<TerrainIntent>,
    #[serde(default)]
    pub projection: FeatureProjectionConfig,
    /// Seasonal polar surface overlays. These compile into
    /// `DynamicSurfaceLayers` and are not baked into static terrain cubemaps.
    #[serde(default)]
    pub ice_caps: Vec<IceCapSpec>,
    /// Optional style override for cold-desert archetypes. Omitted means the
    /// default cold-desert preset.
    #[serde(default)]
    pub cold_desert_style: Option<ColdDesertStyle>,
    #[serde(default)]
    pub authored_features: Vec<AuthoredFeatureConfig>,
}

/// Flat-water placeholder. The compiled `StaticSurfaceData` has zero height
/// everywhere and `sea_level_m` set to a small positive value, so the
/// impostor's water BRDF fires for the entire surface.
#[derive(Clone, Debug, Deserialize)]
pub struct OceanTerrainConfig {
    pub seed: u64,
    /// Authored override for the cubemap face resolution. Same semantics as
    /// [`FeatureTerrainConfig::cubemap_resolution`] — omit to use the
    /// radius-derived default.
    #[serde(default)]
    pub cubemap_resolution: Option<u32>,
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
    pub cubemap_resolution_override: Option<u32>,
}

impl Default for TerrainCompileOptions {
    fn default() -> Self {
        Self {
            crater_count_scale: 1.0,
            cubemap_resolution_override: None,
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
    tectonics: Option<&TectonicConfig>,
    context: &TerrainCompileContext,
    options: TerrainCompileOptions,
    mid_freq: Option<crate::stages::MidFreqRunner>,
) -> Result<PlanetSurface, TerrainCompileError> {
    // Build the tectonic system once; downstream consumers (the static-
    // surface compile and the editor's `PreviewSurfaceOverlays` component) read
    // from the same instance. Cheap (~ms for 2k cells) but needs body
    // radius + body-name-derived seed, both of which live in `context`.
    let tectonics = compile_tectonics_from_config(tectonics, context);
    let static_surface =
        compile_static_terrain_config(terrain, tectonics.as_ref(), context, options, mid_freq)?;
    let dynamic_layers = compile_dynamic_surface_layers(terrain, context)?;
    Ok(PlanetSurface {
        static_surface,
        dynamic_layers,
        tectonics,
    })
}

/// Build the tectonic system for a body when one is configured. The system
/// is regenerated on every load (it lives outside the static-surface cache)
/// because computing it is cheap (~ms for 2k cells) and doing so keeps the
/// cache key surface small. Body identity contributes to the seed so two
/// bodies that share a `TectonicConfig::seed` value still produce distinct
/// plate graphs.
pub fn compile_tectonics_from_config(
    tectonics: Option<&TectonicConfig>,
    context: &TerrainCompileContext,
) -> Option<TectonicSystem> {
    tectonics.map(|cfg| {
        let body_seed = sub_seed(0, &context.body_name.to_lowercase());
        TectonicSystem::build(cfg, context.radius_m, body_seed)
    })
}

pub fn compile_static_terrain_config(
    terrain: &TerrainConfig,
    tectonics: Option<&TectonicSystem>,
    context: &TerrainCompileContext,
    options: TerrainCompileOptions,
    mid_freq: Option<crate::stages::MidFreqRunner>,
) -> Result<StaticSurfaceData, TerrainCompileError> {
    match terrain {
        TerrainConfig::None => Err(TerrainCompileError::UnsupportedNone),
        TerrainConfig::Feature(feature) => {
            let spec = feature.to_planet_spec(context);
            compile_initial_static_surface(
                &spec,
                tectonics,
                FeatureCompileOptions {
                    cubemap_resolution: options
                        .cubemap_resolution_override
                        .or(feature.cubemap_resolution),
                    crater_count_scale: options.crater_count_scale,
                    projection: feature.projection.clone(),
                    cold_desert_style: feature.cold_desert_style.clone().unwrap_or_default(),
                },
                mid_freq,
            )
            .map_err(Into::into)
        }
        TerrainConfig::Ocean(config) => {
            // Ocean bodies don't run the mid-frequency cascade — they're
            // a flat-water placeholder and there's no continental detail
            // to perturb. Drop the runner if one was provided.
            drop(mid_freq);
            Ok(compile_ocean(
                config,
                context,
                options.cubemap_resolution_override,
            ))
        }
    }
}

pub fn compile_dynamic_surface_layers(
    terrain: &TerrainConfig,
    context: &TerrainCompileContext,
) -> Result<DynamicSurfaceLayers, TerrainCompileError> {
    match terrain {
        TerrainConfig::None => Err(TerrainCompileError::UnsupportedNone),
        TerrainConfig::Feature(feature) => {
            let spec = feature.to_planet_spec(context);
            let style = feature.cold_desert_style.clone().unwrap_or_default();
            Ok(dynamic_surface_layers_for(&spec, &style))
        }
        TerrainConfig::Ocean(_) => Ok(DynamicSurfaceLayers::default()),
    }
}

fn compile_ocean(
    config: &OceanTerrainConfig,
    context: &TerrainCompileContext,
    cubemap_resolution_override: Option<u32>,
) -> StaticSurfaceData {
    let mut builder = BodyBuilder::new(
        context.radius_m,
        config.seed,
        // Composition is irrelevant for a flat-ocean placeholder — no
        // stage reads it. Pick a neutral value.
        Composition::new(1.0, 0.0, 0.0, 0.0, 0.0),
        cubemap_resolution_override.or(config.cubemap_resolution),
        4.5,
        context.tidal_axis,
        context.axial_tilt_rad,
    );

    let roughness_texel = quantize_unit_to_u8(config.water_roughness.clamp(0.0, 1.0));
    for face in CubemapFace::ALL {
        for v in builder.roughness_cubemap.face_data_mut(face) {
            *v = roughness_texel;
        }
    }

    builder.sea_level_m = Some(config.sea_level_m);
    let water = WaterAppearance::new([0.012, 0.045, 0.105], 130.0);
    builder.water_appearance = Some(water);
    paint_surface_albedo(
        &mut builder,
        &SurfaceColorSpec::ocean(config.seed, config.sea_level_m, config.seabed_albedo, water),
    );
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
            ice_caps: self.ice_caps.clone(),
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
