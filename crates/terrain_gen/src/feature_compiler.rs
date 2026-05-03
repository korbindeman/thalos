//! Feature-first terrain compiler model.
//!
//! This module is the replacement architecture's data/model layer. It does
//! not mutate `BodyBuilder` and has no Bevy dependency. The current stage
//! pipeline can keep producing `BodyData` while this module grows into the new
//! source of truth for body terrain.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::body_builder::BodyBuilder;
use crate::body_data::BodyData;
use crate::height_generator::{
    ColdDesertBiomeHeightGenerators, HeightGenerator, HeightGeneratorStack, IqDerivativeFbmHeight,
};
use crate::seeding::sub_seed;
use crate::stage::Stage;
use crate::stages::{
    BasinDef, BiomeRule, Biomes, Cratering, Differentiate, MareFlood as MareFloodStage, Megabasin,
    Regolith, Scarps, SpaceWeather, VaelenImpactColor,
};
use crate::surface_field::bake_surface_field_into_builder;
use crate::types::{BiomeParams, Composition};
use crate::vaelen_field::VaelenColdDesertField;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FeatureId(String);

impl FeatureId {
    pub fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        assert!(!value.is_empty(), "feature id must not be empty");
        assert!(
            !value.contains(char::is_whitespace),
            "feature id must not contain whitespace: {value}"
        );
        Self(value)
    }

    pub fn child(&self, local_name: &str) -> Self {
        assert!(!local_name.is_empty(), "child feature id must not be empty");
        assert!(
            !local_name.contains('.'),
            "child feature id segment must not contain '.': {local_name}"
        );
        Self::new(format!("{}.{}", self.0, local_name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for FeatureId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for FeatureId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for FeatureId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

/// Independent deterministic seed streams for one feature.
///
/// `root_seed` is intentionally not enough for authoring. These streams let an
/// editor keep placement while rerolling shape/detail, or keep a parent while
/// regenerating only its children.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSeed {
    pub identity: u64,
    pub placement: u64,
    pub shape: u64,
    pub detail: u64,
    pub children: u64,
}

impl FeatureSeed {
    pub fn derive(parent_seed: u64, id: &FeatureId) -> Self {
        let key = id.as_str();
        Self {
            identity: sub_seed(parent_seed, &format!("{key}:identity")),
            placement: sub_seed(parent_seed, &format!("{key}:placement")),
            shape: sub_seed(parent_seed, &format!("{key}:shape")),
            detail: sub_seed(parent_seed, &format!("{key}:detail")),
            children: sub_seed(parent_seed, &format!("{key}:children")),
        }
    }

    pub fn rerolled(mut self, stream: FeatureSeedStream, salt: &str) -> Self {
        match stream {
            FeatureSeedStream::Identity => {
                self.identity = sub_seed(self.identity, salt);
            }
            FeatureSeedStream::Placement => {
                self.placement = sub_seed(self.placement, salt);
            }
            FeatureSeedStream::Shape => {
                self.shape = sub_seed(self.shape, salt);
            }
            FeatureSeedStream::Detail => {
                self.detail = sub_seed(self.detail, salt);
            }
            FeatureSeedStream::Children => {
                self.children = sub_seed(self.children, salt);
            }
        }
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureSeedStream {
    Identity,
    Placement,
    Shape,
    Detail,
    Children,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureLock {
    Unlocked,
    Placement,
    Shape,
    Detail,
    ShapeAndPlacement,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyArchetype {
    AirlessImpactMoon,
    ColdDesertFormerlyWet,
    AgingOceanicHomeworld,
    GenericTerrestrial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionClass {
    SilicateDominated,
    BasalticSilicate,
    IronRichSilicate,
    IcySilicate,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AtmosphereSpec {
    None,
    ThinCo2 { pressure_bar: f32 },
    Breathable { pressure_bar: f32 },
    Other { pressure_bar: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum HydrosphereSpec {
    None,
    Trace,
    AncientLost,
    OceanFraction(f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IceInventory {
    None,
    Trace,
    Moderate,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerrainIntent {
    ReadAsMoon,
    DistinctNearSideFace,
    DifferentFarSide,
    FirstLandingWorld,
    ReadAsFirstInterplanetarySurfaceWorld,
    ForgivingLandingTerrain,
    VisibleAncientWaterStory,
    RustDustAndEvaporites,
    HomeworldIdentity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanetPhysicalSpec {
    pub radius_m: f32,
    pub gravity_m_s2: f32,
    pub age_gyr: f32,
    pub stellar_flux_earth: f32,
    pub rotation_hours: Option<f32>,
    pub obliquity_deg: Option<f32>,
    pub atmosphere: AtmosphereSpec,
    pub hydrosphere: HydrosphereSpec,
    pub ice_inventory: IceInventory,
    pub composition: CompositionClass,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanetTerrainSpec {
    pub body_id: String,
    pub root_seed: u64,
    pub physical: PlanetPhysicalSpec,
    pub archetype: BodyArchetype,
    pub intent: Vec<TerrainIntent>,
    #[serde(default)]
    pub authored_features: Vec<AuthoredFeatureSpec>,
}

impl PlanetTerrainSpec {
    pub fn mira_default(root_seed: u64) -> Self {
        Self {
            body_id: "mira".to_string(),
            root_seed,
            physical: PlanetPhysicalSpec {
                radius_m: 869_000.0,
                gravity_m_s2: 1.18,
                age_gyr: 4.4,
                stellar_flux_earth: 1.0,
                rotation_hours: None,
                obliquity_deg: None,
                atmosphere: AtmosphereSpec::None,
                hydrosphere: HydrosphereSpec::None,
                ice_inventory: IceInventory::None,
                composition: CompositionClass::SilicateDominated,
            },
            archetype: BodyArchetype::AirlessImpactMoon,
            intent: vec![
                TerrainIntent::ReadAsMoon,
                TerrainIntent::DistinctNearSideFace,
                TerrainIntent::DifferentFarSide,
                TerrainIntent::FirstLandingWorld,
            ],
            authored_features: Vec::new(),
        }
    }

    pub fn vaelen_default(root_seed: u64) -> Self {
        Self {
            body_id: "vaelen".to_string(),
            root_seed,
            physical: PlanetPhysicalSpec {
                radius_m: 1_130_000.0,
                gravity_m_s2: 2.06,
                age_gyr: 4.3,
                stellar_flux_earth: 0.33,
                rotation_hours: Some(28.0),
                obliquity_deg: Some(24.0),
                atmosphere: AtmosphereSpec::ThinCo2 {
                    pressure_bar: 0.015,
                },
                hydrosphere: HydrosphereSpec::AncientLost,
                ice_inventory: IceInventory::Moderate,
                composition: CompositionClass::BasalticSilicate,
            },
            archetype: BodyArchetype::ColdDesertFormerlyWet,
            intent: vec![
                TerrainIntent::ReadAsFirstInterplanetarySurfaceWorld,
                TerrainIntent::ForgivingLandingTerrain,
                TerrainIntent::VisibleAncientWaterStory,
                TerrainIntent::RustDustAndEvaporites,
            ],
            authored_features: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthoredFeatureSpec {
    pub id: FeatureId,
    pub kind: FeatureKind,
    pub parent: Option<FeatureId>,
    pub seed_override: Option<FeatureSeed>,
    pub footprint: Option<FeatureFootprint>,
    pub scale_range_m: Option<ScaleRangeM>,
    #[serde(default)]
    pub params: Vec<FeatureParam>,
    pub lock: FeatureLock,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerrainBiomeRole {
    HighlandRegolith,
    MarePlain,
    FreshEjecta,
    RustDustPlain,
    DuneSea,
    EvaporiteBasin,
    VolcanicPlain,
    RuggedBadland,
    SedimentaryLowland,
    BuriedIce,
    ContinentalCrust,
    OceanicBasin,
    IceCap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerrainBiomeSpec {
    pub id: FeatureId,
    pub role: TerrainBiomeRole,
    pub material: SurfaceMaterialClass,
    pub albedo_linear: [f32; 3],
    pub roughness: f32,
    #[serde(default)]
    pub height: HeightGeneratorStack,
    #[serde(default)]
    pub feature_budgets: Vec<FeatureBudget>,
}

impl TerrainBiomeSpec {
    pub fn new(
        id: impl Into<FeatureId>,
        role: TerrainBiomeRole,
        material: SurfaceMaterialClass,
        albedo_linear: [f32; 3],
        roughness: f32,
    ) -> Self {
        Self {
            id: id.into(),
            role,
            material,
            albedo_linear,
            roughness,
            height: HeightGeneratorStack::default(),
            feature_budgets: Vec::new(),
        }
    }

    pub fn with_height(mut self, height: HeightGeneratorStack) -> Self {
        self.height = height;
        self
    }

    pub fn with_feature_budget(mut self, budget: FeatureBudget) -> Self {
        self.feature_budgets.push(budget);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerrainPrior {
    pub archetype: BodyArchetype,
    pub relief_scale_m: f32,
    pub crater_retention: f32,
    pub crater_density: f32,
    pub large_basin_budget: f32,
    pub volcanic_budget: f32,
    pub tectonic_budget: f32,
    pub erosion_strength: f32,
    pub sediment_mobility: f32,
    pub ice_stability: f32,
    pub regolith_depth_m: f32,
    pub aeolian_activity: f32,
    pub ocean_fraction: f32,
    pub dominant_materials: Vec<SurfaceMaterialClass>,
    pub biomes: Vec<TerrainBiomeSpec>,
    pub feature_budgets: Vec<FeatureBudget>,
}

impl TerrainPrior {
    pub fn infer(spec: &PlanetTerrainSpec) -> Self {
        match spec.archetype {
            BodyArchetype::AirlessImpactMoon => Self::airless_impact_moon(spec),
            BodyArchetype::ColdDesertFormerlyWet => Self::cold_desert_formerly_wet(spec),
            BodyArchetype::AgingOceanicHomeworld => Self::aging_oceanic_homeworld(spec),
            BodyArchetype::GenericTerrestrial => Self::generic_terrestrial(spec),
        }
    }

    fn airless_impact_moon(spec: &PlanetTerrainSpec) -> Self {
        let radius_scale = (spec.physical.radius_m / 869_000.0).sqrt();
        Self {
            archetype: spec.archetype,
            relief_scale_m: 7_000.0 * radius_scale,
            crater_retention: 0.98,
            crater_density: 0.92,
            large_basin_budget: 0.85,
            volcanic_budget: 0.22,
            tectonic_budget: 0.02,
            erosion_strength: 0.0,
            sediment_mobility: 0.0,
            ice_stability: 0.0,
            regolith_depth_m: 8.0,
            aeolian_activity: 0.0,
            ocean_fraction: 0.0,
            dominant_materials: vec![
                SurfaceMaterialClass::AnorthositeHighland,
                SurfaceMaterialClass::MareBasalt,
                SurfaceMaterialClass::FreshEjecta,
                SurfaceMaterialClass::MatureRegolith,
            ],
            biomes: airless_impact_biomes(radius_scale),
            feature_budgets: vec![
                FeatureBudget::new(FeatureKind::Megabasin, 2, 1.0),
                FeatureBudget::new(FeatureKind::CraterPopulation, 1, 1.0),
                FeatureBudget::new(FeatureKind::MareFlood, 2, 0.8),
                FeatureBudget::new(FeatureKind::RegolithGarden, 1, 0.7),
            ],
        }
    }

    fn cold_desert_formerly_wet(spec: &PlanetTerrainSpec) -> Self {
        let ice_stability = match spec.physical.ice_inventory {
            IceInventory::None => 0.0,
            IceInventory::Trace => 0.25,
            IceInventory::Moderate => 0.68,
            IceInventory::High => 0.9,
        };
        Self {
            archetype: spec.archetype,
            relief_scale_m: 5_000.0,
            crater_retention: 0.54,
            crater_density: 0.46,
            large_basin_budget: 0.55,
            volcanic_budget: 0.28,
            tectonic_budget: 0.12,
            erosion_strength: 0.5,
            sediment_mobility: 0.72,
            ice_stability,
            regolith_depth_m: 4.0,
            aeolian_activity: 0.82,
            ocean_fraction: 0.0,
            dominant_materials: vec![
                SurfaceMaterialClass::BasalticHighland,
                SurfaceMaterialClass::RustDust,
                SurfaceMaterialClass::Evaporite,
                SurfaceMaterialClass::SedimentaryRock,
                SurfaceMaterialClass::BuriedIce,
            ],
            biomes: cold_desert_biomes(),
            feature_budgets: vec![
                FeatureBudget::new(FeatureKind::SedimentaryLowlands, 1, 1.0),
                FeatureBudget::new(FeatureKind::ChannelNetwork, 6, 0.95),
                FeatureBudget::new(FeatureKind::EvaporiteBasin, 8, 0.7),
                FeatureBudget::new(FeatureKind::BuriedIceZone, 3, 0.65),
                FeatureBudget::new(FeatureKind::AeolianMantle, 1, 0.8),
            ],
        }
    }

    fn aging_oceanic_homeworld(spec: &PlanetTerrainSpec) -> Self {
        let ocean_fraction = match spec.physical.hydrosphere {
            HydrosphereSpec::OceanFraction(f) => f,
            _ => 0.65,
        };
        Self {
            archetype: spec.archetype,
            relief_scale_m: 4_500.0,
            crater_retention: 0.02,
            crater_density: 0.03,
            large_basin_budget: 0.04,
            volcanic_budget: 0.35,
            tectonic_budget: 0.62,
            erosion_strength: 0.88,
            sediment_mobility: 0.86,
            ice_stability: 0.4,
            regolith_depth_m: 2.0,
            aeolian_activity: 0.25,
            ocean_fraction,
            dominant_materials: vec![
                SurfaceMaterialClass::ContinentalCrust,
                SurfaceMaterialClass::OceanicBasalt,
                SurfaceMaterialClass::RustSediment,
                SurfaceMaterialClass::Ice,
            ],
            biomes: oceanic_homeworld_biomes(ocean_fraction),
            feature_budgets: vec![
                FeatureBudget::new(FeatureKind::CrustalProvince, 40, 1.0),
                FeatureBudget::new(FeatureKind::OrogenBelt, 8, 0.9),
                FeatureBudget::new(FeatureKind::ChannelNetwork, 20, 0.95),
                FeatureBudget::new(FeatureKind::AeolianMantle, 4, 0.25),
            ],
        }
    }

    fn generic_terrestrial(spec: &PlanetTerrainSpec) -> Self {
        Self {
            archetype: spec.archetype,
            relief_scale_m: spec.physical.gravity_m_s2.recip().clamp(0.05, 0.5) * 25_000.0,
            crater_retention: 0.35,
            crater_density: 0.35,
            large_basin_budget: 0.35,
            volcanic_budget: 0.35,
            tectonic_budget: 0.25,
            erosion_strength: 0.35,
            sediment_mobility: 0.35,
            ice_stability: 0.25,
            regolith_depth_m: 3.0,
            aeolian_activity: 0.25,
            ocean_fraction: 0.0,
            dominant_materials: vec![SurfaceMaterialClass::BasalticHighland],
            biomes: generic_terrestrial_biomes(spec),
            feature_budgets: vec![
                FeatureBudget::new(FeatureKind::CrustalProvince, 8, 0.7),
                FeatureBudget::new(FeatureKind::ImpactBasinArchive, 4, 0.5),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceMaterialClass {
    AnorthositeHighland,
    MareBasalt,
    FreshEjecta,
    MatureRegolith,
    BasalticHighland,
    RustDust,
    Evaporite,
    SedimentaryRock,
    BuriedIce,
    ContinentalCrust,
    OceanicBasalt,
    RustSediment,
    Ice,
}

fn fbm_height(amplitude_m: f32, frequency: f32, octaves: u32) -> HeightGeneratorStack {
    HeightGeneratorStack::single(HeightGenerator::IqDerivativeFbm(
        IqDerivativeFbmHeight::new(amplitude_m, frequency, octaves),
    ))
}

fn airless_impact_biomes(radius_scale: f32) -> Vec<TerrainBiomeSpec> {
    vec![
        TerrainBiomeSpec::new(
            "biome.highland_regolith",
            TerrainBiomeRole::HighlandRegolith,
            SurfaceMaterialClass::AnorthositeHighland,
            [0.12, 0.12, 0.12],
            0.88,
        )
        .with_height(fbm_height(120.0 * radius_scale, 5.0, 7))
        .with_feature_budget(FeatureBudget::new(FeatureKind::CraterPopulation, 1, 1.0))
        .with_feature_budget(FeatureBudget::new(FeatureKind::RegolithGarden, 1, 0.8)),
        TerrainBiomeSpec::new(
            "biome.mare_basalt",
            TerrainBiomeRole::MarePlain,
            SurfaceMaterialClass::MareBasalt,
            [0.05, 0.05, 0.05],
            0.72,
        )
        .with_height(fbm_height(45.0 * radius_scale, 2.2, 5))
        .with_feature_budget(FeatureBudget::new(FeatureKind::MareFlood, 2, 0.8)),
        TerrainBiomeSpec::new(
            "biome.fresh_ejecta",
            TerrainBiomeRole::FreshEjecta,
            SurfaceMaterialClass::FreshEjecta,
            [0.28, 0.28, 0.28],
            0.82,
        )
        .with_height(fbm_height(75.0 * radius_scale, 8.0, 5))
        .with_feature_budget(FeatureBudget::new(
            FeatureKind::SecondaryCraterField,
            1,
            0.45,
        )),
    ]
}

fn cold_desert_biomes() -> Vec<TerrainBiomeSpec> {
    let heights = ColdDesertBiomeHeightGenerators::default();
    vec![
        TerrainBiomeSpec::new(
            "biome.rust_dust_plain",
            TerrainBiomeRole::RustDustPlain,
            SurfaceMaterialClass::RustDust,
            [0.44, 0.22, 0.12],
            0.88,
        )
        .with_height(heights.rust_dust_plain.clone())
        .with_feature_budget(FeatureBudget::new(FeatureKind::AeolianMantle, 1, 0.85)),
        TerrainBiomeSpec::new(
            "biome.dune_basin",
            TerrainBiomeRole::DuneSea,
            SurfaceMaterialClass::RustDust,
            [0.70, 0.39, 0.13],
            0.92,
        )
        .with_height(heights.dune_basin.clone())
        .with_feature_budget(FeatureBudget::new(FeatureKind::DuneField, 4, 1.0)),
        TerrainBiomeSpec::new(
            "biome.pale_evaporite_basin",
            TerrainBiomeRole::EvaporiteBasin,
            SurfaceMaterialClass::Evaporite,
            [0.78, 0.70, 0.54],
            0.68,
        )
        .with_height(heights.pale_evaporite_basin.clone())
        .with_feature_budget(FeatureBudget::new(FeatureKind::EvaporiteBasin, 8, 0.9))
        .with_feature_budget(FeatureBudget::new(FeatureKind::DeltaFan, 4, 0.6)),
        TerrainBiomeSpec::new(
            "biome.dark_volcanic_province",
            TerrainBiomeRole::VolcanicPlain,
            SurfaceMaterialClass::BasalticHighland,
            [0.12, 0.08, 0.07],
            0.72,
        )
        .with_height(heights.dark_volcanic_province.clone())
        .with_feature_budget(FeatureBudget::new(FeatureKind::VolcanicPlain, 2, 0.65)),
        TerrainBiomeSpec::new(
            "biome.rugged_badlands",
            TerrainBiomeRole::RuggedBadland,
            SurfaceMaterialClass::SedimentaryRock,
            [0.42, 0.27, 0.18],
            0.86,
        )
        .with_height(heights.rugged_badlands.clone())
        .with_feature_budget(FeatureBudget::new(FeatureKind::ChannelNetwork, 6, 0.8)),
        TerrainBiomeSpec::new(
            "biome.buried_ice",
            TerrainBiomeRole::BuriedIce,
            SurfaceMaterialClass::BuriedIce,
            [0.58, 0.56, 0.52],
            0.76,
        )
        .with_feature_budget(FeatureBudget::new(FeatureKind::BuriedIceZone, 3, 0.65)),
    ]
}

fn oceanic_homeworld_biomes(ocean_fraction: f32) -> Vec<TerrainBiomeSpec> {
    vec![
        TerrainBiomeSpec::new(
            "biome.continental_crust",
            TerrainBiomeRole::ContinentalCrust,
            SurfaceMaterialClass::ContinentalCrust,
            [0.32, 0.31, 0.27],
            0.76,
        )
        .with_height(fbm_height(520.0, 3.0, 7))
        .with_feature_budget(FeatureBudget::new(FeatureKind::OrogenBelt, 8, 0.9))
        .with_feature_budget(FeatureBudget::new(FeatureKind::ChannelNetwork, 20, 1.0)),
        TerrainBiomeSpec::new(
            "biome.oceanic_basin",
            TerrainBiomeRole::OceanicBasin,
            SurfaceMaterialClass::OceanicBasalt,
            [0.05, 0.07, 0.11],
            0.58,
        )
        .with_height(fbm_height(260.0, 2.0, 6))
        .with_feature_budget(FeatureBudget::new(
            FeatureKind::CrustalProvince,
            (40.0 * ocean_fraction).round() as u32,
            0.7,
        )),
        TerrainBiomeSpec::new(
            "biome.sedimentary_lowland",
            TerrainBiomeRole::SedimentaryLowland,
            SurfaceMaterialClass::RustSediment,
            [0.42, 0.30, 0.22],
            0.82,
        )
        .with_height(fbm_height(180.0, 1.8, 6))
        .with_feature_budget(FeatureBudget::new(FeatureKind::DeltaFan, 10, 0.7)),
        TerrainBiomeSpec::new(
            "biome.ice_cap",
            TerrainBiomeRole::IceCap,
            SurfaceMaterialClass::Ice,
            [0.86, 0.88, 0.90],
            0.42,
        )
        .with_height(fbm_height(80.0, 2.4, 5)),
    ]
}

fn generic_terrestrial_biomes(spec: &PlanetTerrainSpec) -> Vec<TerrainBiomeSpec> {
    let relief = spec.physical.gravity_m_s2.recip().clamp(0.05, 0.5) * 1_800.0;
    vec![
        TerrainBiomeSpec::new(
            "biome.basaltic_highland",
            TerrainBiomeRole::HighlandRegolith,
            SurfaceMaterialClass::BasalticHighland,
            [0.30, 0.25, 0.22],
            0.82,
        )
        .with_height(fbm_height(relief, 3.0, 8))
        .with_feature_budget(FeatureBudget::new(FeatureKind::CrustalProvince, 8, 0.7))
        .with_feature_budget(FeatureBudget::new(FeatureKind::ImpactBasinArchive, 4, 0.5)),
    ]
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureBudget {
    pub kind: FeatureKind,
    pub target_count: u32,
    pub importance: f32,
}

impl FeatureBudget {
    pub fn new(kind: FeatureKind, target_count: u32, importance: f32) -> Self {
        Self {
            kind,
            target_count,
            importance,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerrainEra {
    CrustFormation,
    HeavyBombardment,
    AncientResurfacing,
    FluvialEpoch,
    DryingEpoch,
    RecentSurface,
    PresentDetail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureKind {
    BodyRoot,
    GlobalCrust,
    CrustalProvince,
    HighlandProvince,
    SedimentaryLowlands,
    Megabasin,
    ImpactBasin,
    ImpactBasinArchive,
    BasinFloor,
    RimUplift,
    FracturedRing,
    EjectaApron,
    CraterPopulation,
    CraterCohort,
    SecondaryCraterField,
    MareFlood,
    RegolithGarden,
    SpaceWeathering,
    ChannelNetwork,
    DeltaFan,
    EvaporiteBasin,
    BuriedIceZone,
    VolcanicPlain,
    AeolianMantle,
    DuneField,
    YardangField,
    OrogenBelt,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FeatureFootprint {
    Global,
    Hemisphere {
        center: Vec3,
        angular_radius_rad: f32,
    },
    Circle {
        center: Vec3,
        angular_radius_rad: f32,
    },
    Band {
        min_lat_deg: f32,
        max_lat_deg: f32,
    },
    Polyline {
        points: Vec<Vec3>,
        width_rad: f32,
    },
    Unresolved,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScaleRangeM {
    pub min_m: f32,
    pub max_m: f32,
}

impl ScaleRangeM {
    pub fn new(min_m: f32, max_m: f32) -> Self {
        assert!(min_m >= 0.0, "scale minimum must be non-negative");
        assert!(max_m >= min_m, "scale maximum must be >= minimum");
        Self { min_m, max_m }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureParam {
    pub key: String,
    pub value: FeatureParamValue,
}

impl FeatureParam {
    pub fn number(key: &str, value: f32) -> Self {
        Self {
            key: key.to_string(),
            value: FeatureParamValue::Number(value),
        }
    }

    pub fn text(key: &str, value: &str) -> Self {
        Self {
            key: key.to_string(),
            value: FeatureParamValue::Text(value.to_string()),
        }
    }

    pub fn boolean(key: &str, value: bool) -> Self {
        Self {
            key: key.to_string(),
            value: FeatureParamValue::Bool(value),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FeatureParamValue {
    Number(f32),
    Text(String),
    Bool(bool),
    Direction(Vec3),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureInstance {
    pub id: FeatureId,
    pub kind: FeatureKind,
    pub parent: Option<FeatureId>,
    pub seed: FeatureSeed,
    pub era: TerrainEra,
    pub footprint: FeatureFootprint,
    pub scale_range_m: ScaleRangeM,
    #[serde(default)]
    pub params: Vec<FeatureParam>,
    pub lock: FeatureLock,
    #[serde(default)]
    pub children: Vec<FeatureId>,
}

impl FeatureInstance {
    pub fn new(
        id: FeatureId,
        kind: FeatureKind,
        parent: Option<FeatureId>,
        seed: FeatureSeed,
        era: TerrainEra,
        footprint: FeatureFootprint,
        scale_range_m: ScaleRangeM,
    ) -> Self {
        Self {
            id,
            kind,
            parent,
            seed,
            era,
            footprint,
            scale_range_m,
            params: Vec::new(),
            lock: FeatureLock::Unlocked,
            children: Vec::new(),
        }
    }

    pub fn with_params(mut self, params: Vec<FeatureParam>) -> Self {
        self.params = params;
        self
    }

    pub fn with_lock(mut self, lock: FeatureLock) -> Self {
        self.lock = lock;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureManifest {
    pub body_id: String,
    pub root_seed: u64,
    pub root: FeatureId,
    pub features: Vec<FeatureInstance>,
}

impl FeatureManifest {
    pub fn new(body_id: impl Into<String>, root_seed: u64) -> Self {
        let body_id = body_id.into();
        let root = FeatureId::new(body_id.clone());
        let root_feature = FeatureInstance::new(
            root.clone(),
            FeatureKind::BodyRoot,
            None,
            FeatureSeed::derive(root_seed, &root),
            TerrainEra::CrustFormation,
            FeatureFootprint::Global,
            ScaleRangeM::new(0.0, f32::INFINITY),
        )
        .with_lock(FeatureLock::Placement);

        Self {
            body_id,
            root_seed,
            root,
            features: vec![root_feature],
        }
    }

    pub fn get(&self, id: &FeatureId) -> Option<&FeatureInstance> {
        self.features.iter().find(|feature| &feature.id == id)
    }

    pub fn get_mut(&mut self, id: &FeatureId) -> Option<&mut FeatureInstance> {
        self.features.iter_mut().find(|feature| &feature.id == id)
    }

    pub fn add_feature(&mut self, feature: FeatureInstance) {
        assert!(
            self.get(&feature.id).is_none(),
            "duplicate feature id: {}",
            feature.id
        );
        if let Some(parent_id) = feature.parent.clone() {
            let parent = self
                .get_mut(&parent_id)
                .unwrap_or_else(|| panic!("unknown parent feature: {parent_id}"));
            parent.children.push(feature.id.clone());
        }
        self.features.push(feature);
    }

    pub fn derive_seed(&self, parent: Option<&FeatureId>, id: &FeatureId) -> FeatureSeed {
        let parent_seed = parent
            .and_then(|parent_id| self.get(parent_id))
            .map(|parent| parent.seed.children)
            .unwrap_or(self.root_seed);
        FeatureSeed::derive(parent_seed, id)
    }

    pub fn feature_count_by_kind(&self, kind: FeatureKind) -> usize {
        self.features
            .iter()
            .filter(|feature| feature.kind == kind)
            .count()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerrainCompilationPlan {
    pub prior: TerrainPrior,
    pub manifest: FeatureManifest,
    pub impostor: ImpostorProjectionPlan,
}

pub fn plan_initial_compilation(spec: &PlanetTerrainSpec) -> TerrainCompilationPlan {
    let prior = TerrainPrior::infer(spec);
    let manifest = generate_initial_manifest(spec);
    let impostor = ImpostorProjectionPlan::for_archetype(spec.archetype);
    TerrainCompilationPlan {
        prior,
        manifest,
        impostor,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureCompileOptions {
    /// `0` means use `BodyBuilder`'s radius-based default resolution.
    pub cubemap_resolution: u32,
    /// Multiplies expensive generated populations for iteration/test bakes.
    pub crater_count_scale: f32,
    pub projection: FeatureProjectionConfig,
}

impl Default for FeatureCompileOptions {
    fn default() -> Self {
        Self {
            cubemap_resolution: 0,
            crater_count_scale: 1.0,
            projection: FeatureProjectionConfig::Auto,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum FeatureProjectionConfig {
    #[default]
    Auto,
    AirlessImpact(AirlessImpactProjectionConfig),
    ColdDesert(ColdDesertProjectionConfig),
}

impl FeatureProjectionConfig {
    fn airless_impact(self) -> AirlessImpactProjectionConfig {
        match self {
            Self::AirlessImpact(config) => config,
            Self::Auto | Self::ColdDesert(_) => AirlessImpactProjectionConfig::default(),
        }
    }

    fn cold_desert(self) -> ColdDesertProjectionConfig {
        match self {
            Self::ColdDesert(config) => config,
            Self::Auto | Self::AirlessImpact(_) => ColdDesertProjectionConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AirlessImpactProjectionConfig {
    #[serde(default)]
    pub highland_biomes: Vec<BiomeParams>,
    #[serde(default)]
    pub highland_biome_rules: Vec<BiomeRule>,
    #[serde(default = "default_airless_base_crater_count")]
    pub base_crater_count: u32,
    #[serde(default = "default_airless_sfd_slope")]
    pub sfd_slope: f64,
    #[serde(default = "default_airless_min_crater_radius_m")]
    pub min_crater_radius_m: f32,
    #[serde(default = "default_airless_max_crater_radius_m")]
    pub max_crater_radius_m: f32,
    #[serde(default = "default_airless_crater_age_bias")]
    pub crater_age_bias: f64,
    #[serde(default = "default_airless_cubemap_bake_threshold_m")]
    pub cubemap_bake_threshold_m: f32,
    #[serde(default = "default_airless_secondary_parent_radius_m")]
    pub secondary_parent_radius_m: f32,
    #[serde(default = "default_airless_secondaries_per_parent")]
    pub secondaries_per_parent: u32,
    #[serde(default = "default_airless_crater_saturation_fraction")]
    pub crater_saturation_fraction: f32,
    #[serde(default = "default_airless_chain_count")]
    pub chain_count: u32,
    #[serde(default = "default_airless_chain_segment_count")]
    pub chain_segment_count: u32,
    #[serde(default = "default_airless_forced_young_count")]
    pub forced_young_count: u32,
    #[serde(default = "default_airless_mare_target_count")]
    pub mare_target_count: u32,
    #[serde(default = "default_airless_mare_additional_crater_count")]
    pub mare_additional_crater_count: u32,
    #[serde(default = "default_airless_mare_fill_fraction")]
    pub mare_fill_fraction: f32,
    #[serde(default = "default_airless_mare_near_side_bias")]
    pub mare_near_side_bias: f32,
    #[serde(default = "default_airless_mare_boundary_noise_amplitude_m")]
    pub mare_boundary_noise_amplitude_m: f32,
    #[serde(default = "default_airless_mare_boundary_noise_freq")]
    pub mare_boundary_noise_freq: f64,
    #[serde(default = "default_airless_mare_episode_count")]
    pub mare_episode_count: u32,
    #[serde(default = "default_airless_mare_wrinkle_ridges")]
    pub mare_wrinkle_ridges: bool,
    #[serde(default = "default_airless_regolith_amplitude_m")]
    pub regolith_amplitude_m: f32,
    #[serde(default = "default_airless_regolith_characteristic_wavelength_m")]
    pub regolith_characteristic_wavelength_m: f32,
    #[serde(default = "default_airless_regolith_crater_density_multiplier")]
    pub regolith_crater_density_multiplier: f32,
    #[serde(default = "default_airless_regolith_bake_d_min_m")]
    pub regolith_bake_d_min_m: f32,
    #[serde(default = "default_airless_regolith_bake_scale")]
    pub regolith_bake_scale: f32,
    #[serde(default = "default_airless_regolith_density_modulation")]
    pub regolith_density_modulation: f32,
    #[serde(default = "default_airless_regolith_density_wavelength_m")]
    pub regolith_density_wavelength_m: f32,
    #[serde(default = "default_airless_highland_mature_albedo")]
    pub highland_mature_albedo: f32,
    #[serde(default = "default_airless_highland_fresh_albedo")]
    pub highland_fresh_albedo: f32,
    #[serde(default = "default_airless_mare_mature_albedo")]
    pub mare_mature_albedo: f32,
    #[serde(default = "default_airless_mare_fresh_albedo")]
    pub mare_fresh_albedo: f32,
    #[serde(default = "default_airless_mare_tint")]
    pub mare_tint: [f32; 3],
    #[serde(default = "default_airless_young_crater_age_threshold")]
    pub young_crater_age_threshold: f32,
    #[serde(default = "default_airless_ray_age_threshold")]
    pub ray_age_threshold: f32,
    #[serde(default = "default_airless_ray_extent_radii")]
    pub ray_extent_radii: f32,
    #[serde(default = "default_airless_ray_count_per_crater")]
    pub ray_count_per_crater: u32,
    #[serde(default = "default_airless_ray_half_width")]
    pub ray_half_width: f32,
    #[serde(default)]
    pub scarps: Option<AirlessScarpProjectionConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AirlessScarpProjectionConfig {
    pub count: u32,
    pub min_length_m: f32,
    pub max_length_m: f32,
    pub width_m: f32,
    pub height_m: f32,
    #[serde(default = "default_airless_scarp_curvature")]
    pub curvature: f32,
}

impl Default for AirlessImpactProjectionConfig {
    fn default() -> Self {
        Self {
            highland_biomes: Vec::new(),
            highland_biome_rules: Vec::new(),
            base_crater_count: default_airless_base_crater_count(),
            sfd_slope: default_airless_sfd_slope(),
            min_crater_radius_m: default_airless_min_crater_radius_m(),
            max_crater_radius_m: default_airless_max_crater_radius_m(),
            crater_age_bias: default_airless_crater_age_bias(),
            cubemap_bake_threshold_m: default_airless_cubemap_bake_threshold_m(),
            secondary_parent_radius_m: default_airless_secondary_parent_radius_m(),
            secondaries_per_parent: default_airless_secondaries_per_parent(),
            crater_saturation_fraction: default_airless_crater_saturation_fraction(),
            chain_count: default_airless_chain_count(),
            chain_segment_count: default_airless_chain_segment_count(),
            forced_young_count: default_airless_forced_young_count(),
            mare_target_count: default_airless_mare_target_count(),
            mare_additional_crater_count: default_airless_mare_additional_crater_count(),
            mare_fill_fraction: default_airless_mare_fill_fraction(),
            mare_near_side_bias: default_airless_mare_near_side_bias(),
            mare_boundary_noise_amplitude_m: default_airless_mare_boundary_noise_amplitude_m(),
            mare_boundary_noise_freq: default_airless_mare_boundary_noise_freq(),
            mare_episode_count: default_airless_mare_episode_count(),
            mare_wrinkle_ridges: default_airless_mare_wrinkle_ridges(),
            regolith_amplitude_m: default_airless_regolith_amplitude_m(),
            regolith_characteristic_wavelength_m:
                default_airless_regolith_characteristic_wavelength_m(),
            regolith_crater_density_multiplier: default_airless_regolith_crater_density_multiplier(
            ),
            regolith_bake_d_min_m: default_airless_regolith_bake_d_min_m(),
            regolith_bake_scale: default_airless_regolith_bake_scale(),
            regolith_density_modulation: default_airless_regolith_density_modulation(),
            regolith_density_wavelength_m: default_airless_regolith_density_wavelength_m(),
            highland_mature_albedo: default_airless_highland_mature_albedo(),
            highland_fresh_albedo: default_airless_highland_fresh_albedo(),
            mare_mature_albedo: default_airless_mare_mature_albedo(),
            mare_fresh_albedo: default_airless_mare_fresh_albedo(),
            mare_tint: default_airless_mare_tint(),
            young_crater_age_threshold: default_airless_young_crater_age_threshold(),
            ray_age_threshold: default_airless_ray_age_threshold(),
            ray_extent_radii: default_airless_ray_extent_radii(),
            ray_count_per_crater: default_airless_ray_count_per_crater(),
            ray_half_width: default_airless_ray_half_width(),
            scarps: None,
        }
    }
}

fn default_airless_base_crater_count() -> u32 {
    280_000
}

fn default_airless_sfd_slope() -> f64 {
    2.0
}

fn default_airless_min_crater_radius_m() -> f32 {
    1_500.0
}

fn default_airless_max_crater_radius_m() -> f32 {
    80_000.0
}

fn default_airless_crater_age_bias() -> f64 {
    1.0
}

fn default_airless_cubemap_bake_threshold_m() -> f32 {
    1_500.0
}

fn default_airless_secondary_parent_radius_m() -> f32 {
    40_000.0
}

fn default_airless_secondaries_per_parent() -> u32 {
    18
}

fn default_airless_crater_saturation_fraction() -> f32 {
    0.05
}

fn default_airless_chain_count() -> u32 {
    3
}

fn default_airless_chain_segment_count() -> u32 {
    10
}

fn default_airless_forced_young_count() -> u32 {
    16
}

fn default_airless_mare_target_count() -> u32 {
    2
}

fn default_airless_mare_additional_crater_count() -> u32 {
    2
}

fn default_airless_mare_fill_fraction() -> f32 {
    0.72
}

fn default_airless_mare_near_side_bias() -> f32 {
    1.0
}

fn default_airless_mare_boundary_noise_amplitude_m() -> f32 {
    700.0
}

fn default_airless_mare_boundary_noise_freq() -> f64 {
    4.0
}

fn default_airless_mare_episode_count() -> u32 {
    3
}

fn default_airless_mare_wrinkle_ridges() -> bool {
    true
}

fn default_airless_regolith_amplitude_m() -> f32 {
    8.0
}

fn default_airless_regolith_characteristic_wavelength_m() -> f32 {
    120.0
}

fn default_airless_regolith_crater_density_multiplier() -> f32 {
    1.0
}

fn default_airless_regolith_bake_d_min_m() -> f32 {
    600.0
}

fn default_airless_regolith_bake_scale() -> f32 {
    1.0
}

fn default_airless_regolith_density_modulation() -> f32 {
    0.25
}

fn default_airless_regolith_density_wavelength_m() -> f32 {
    250_000.0
}

fn default_airless_highland_mature_albedo() -> f32 {
    0.12
}

fn default_airless_highland_fresh_albedo() -> f32 {
    0.28
}

fn default_airless_mare_mature_albedo() -> f32 {
    0.05
}

fn default_airless_mare_fresh_albedo() -> f32 {
    0.08
}

fn default_airless_mare_tint() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn default_airless_young_crater_age_threshold() -> f32 {
    0.9
}

fn default_airless_ray_age_threshold() -> f32 {
    0.40
}

fn default_airless_ray_extent_radii() -> f32 {
    9.0
}

fn default_airless_ray_count_per_crater() -> u32 {
    14
}

fn default_airless_ray_half_width() -> f32 {
    0.055
}

fn default_airless_scarp_curvature() -> f32 {
    0.1
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColdDesertProjectionConfig {
    #[serde(default = "default_cold_desert_relief_scale_m")]
    pub relief_scale_m: f32,
    #[serde(default, alias = "height_generators")]
    pub biome_height_generators: ColdDesertBiomeHeightGenerators,
    #[serde(default = "default_cold_desert_volcanic_dark_strength")]
    pub volcanic_dark_strength: f32,
    #[serde(default = "default_cold_desert_pale_basin_strength")]
    pub pale_basin_strength: f32,
    #[serde(default = "default_cold_desert_channel_strength")]
    pub channel_strength: f32,
    #[serde(default = "default_cold_desert_dune_strength")]
    pub dune_strength: f32,
    #[serde(default = "default_cold_desert_base_crater_count")]
    pub base_crater_count: u32,
    #[serde(default = "default_cold_desert_min_crater_radius_m")]
    pub min_crater_radius_m: f32,
    #[serde(default = "default_cold_desert_max_crater_radius_m")]
    pub max_crater_radius_m: f32,
    #[serde(default = "default_cold_desert_cubemap_bake_threshold_m")]
    pub cubemap_bake_threshold_m: f32,
}

impl Default for ColdDesertProjectionConfig {
    fn default() -> Self {
        Self {
            relief_scale_m: default_cold_desert_relief_scale_m(),
            biome_height_generators: ColdDesertBiomeHeightGenerators::default(),
            volcanic_dark_strength: default_cold_desert_volcanic_dark_strength(),
            pale_basin_strength: default_cold_desert_pale_basin_strength(),
            channel_strength: default_cold_desert_channel_strength(),
            dune_strength: default_cold_desert_dune_strength(),
            base_crater_count: default_cold_desert_base_crater_count(),
            min_crater_radius_m: default_cold_desert_min_crater_radius_m(),
            max_crater_radius_m: default_cold_desert_max_crater_radius_m(),
            cubemap_bake_threshold_m: default_cold_desert_cubemap_bake_threshold_m(),
        }
    }
}

fn default_cold_desert_relief_scale_m() -> f32 {
    1.0
}

fn default_cold_desert_volcanic_dark_strength() -> f32 {
    1.0
}

fn default_cold_desert_pale_basin_strength() -> f32 {
    1.0
}

fn default_cold_desert_channel_strength() -> f32 {
    1.0
}

fn default_cold_desert_dune_strength() -> f32 {
    1.0
}

fn default_cold_desert_base_crater_count() -> u32 {
    90_000
}

fn default_cold_desert_min_crater_radius_m() -> f32 {
    800.0
}

fn default_cold_desert_max_crater_radius_m() -> f32 {
    90_000.0
}

fn default_cold_desert_cubemap_bake_threshold_m() -> f32 {
    1_200.0
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeatureCompileError {
    UnsupportedArchetype(BodyArchetype),
    MissingFeature(FeatureId),
    InvalidFeatureFootprint(FeatureId),
}

impl std::fmt::Display for FeatureCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedArchetype(archetype) => {
                write!(f, "unsupported terrain compiler archetype: {archetype:?}")
            }
            Self::MissingFeature(id) => write!(f, "missing required feature: {id}"),
            Self::InvalidFeatureFootprint(id) => {
                write!(
                    f,
                    "feature has an invalid footprint for this projection: {id}"
                )
            }
        }
    }
}

impl std::error::Error for FeatureCompileError {}

/// Compile an initial feature terrain spec into the current render handoff.
///
/// This is the compatibility projection that lets the new feature compiler
/// drive the existing flat impostor renderer. The implementation currently
/// supports Mira-style airless impact moons and uses the existing bake stages
/// as projection primitives, but seeds and feature placement come from the
/// feature manifest.
pub fn compile_initial_body_data(
    spec: &PlanetTerrainSpec,
    options: FeatureCompileOptions,
) -> Result<BodyData, FeatureCompileError> {
    let plan = plan_initial_compilation(spec);
    compile_manifest_to_body_data(spec, &plan.manifest, &plan.prior, options)
}

pub fn compile_manifest_to_body_data(
    spec: &PlanetTerrainSpec,
    manifest: &FeatureManifest,
    prior: &TerrainPrior,
    options: FeatureCompileOptions,
) -> Result<BodyData, FeatureCompileError> {
    match spec.archetype {
        BodyArchetype::AirlessImpactMoon => {
            compile_airless_impact_moon(spec, manifest, prior, options)
        }
        BodyArchetype::ColdDesertFormerlyWet => {
            compile_cold_desert_formerly_wet(spec, manifest, prior, options)
        }
        archetype => Err(FeatureCompileError::UnsupportedArchetype(archetype)),
    }
}

pub fn generate_initial_manifest(spec: &PlanetTerrainSpec) -> FeatureManifest {
    let mut manifest = FeatureManifest::new(spec.body_id.clone(), spec.root_seed);
    match spec.archetype {
        BodyArchetype::AirlessImpactMoon => add_mira_style_features(&mut manifest, spec),
        BodyArchetype::ColdDesertFormerlyWet => add_vaelen_style_features(&mut manifest, spec),
        BodyArchetype::AgingOceanicHomeworld => add_homeworld_style_features(&mut manifest, spec),
        BodyArchetype::GenericTerrestrial => add_generic_terrestrial_features(&mut manifest, spec),
    }

    for authored in &spec.authored_features {
        add_authored_feature(&mut manifest, authored);
    }

    manifest
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImpostorProjectionPlan {
    pub baked_layers: Vec<BakedTerrainLayer>,
    pub analytic_buffers: Vec<AnalyticFeatureBuffer>,
    pub analytic_min_scale_m: f32,
    pub excluded_below_scale_m: f32,
}

impl ImpostorProjectionPlan {
    pub fn for_archetype(archetype: BodyArchetype) -> Self {
        match archetype {
            BodyArchetype::AirlessImpactMoon => Self {
                baked_layers: vec![
                    BakedTerrainLayer::Height,
                    BakedTerrainLayer::Albedo,
                    BakedTerrainLayer::MaterialId,
                    BakedTerrainLayer::ProvenanceId,
                ],
                analytic_buffers: vec![
                    AnalyticFeatureBuffer::Craters,
                    AnalyticFeatureBuffer::RaySystems,
                ],
                analytic_min_scale_m: 500.0,
                excluded_below_scale_m: 250.0,
            },
            BodyArchetype::ColdDesertFormerlyWet => Self {
                baked_layers: vec![
                    BakedTerrainLayer::Height,
                    BakedTerrainLayer::Albedo,
                    BakedTerrainLayer::MaterialId,
                    BakedTerrainLayer::MoistureHistory,
                    BakedTerrainLayer::IceStability,
                    BakedTerrainLayer::ProvenanceId,
                ],
                analytic_buffers: vec![
                    AnalyticFeatureBuffer::Craters,
                    AnalyticFeatureBuffer::Channels,
                    AnalyticFeatureBuffer::Scarps,
                    AnalyticFeatureBuffer::DuneFields,
                ],
                analytic_min_scale_m: 1_000.0,
                excluded_below_scale_m: 500.0,
            },
            BodyArchetype::AgingOceanicHomeworld => Self {
                baked_layers: vec![
                    BakedTerrainLayer::Height,
                    BakedTerrainLayer::Albedo,
                    BakedTerrainLayer::MaterialId,
                    BakedTerrainLayer::WaterMask,
                    BakedTerrainLayer::Climate,
                    BakedTerrainLayer::ProvenanceId,
                ],
                analytic_buffers: vec![
                    AnalyticFeatureBuffer::Channels,
                    AnalyticFeatureBuffer::Scarps,
                    AnalyticFeatureBuffer::DuneFields,
                ],
                analytic_min_scale_m: 2_000.0,
                excluded_below_scale_m: 1_000.0,
            },
            BodyArchetype::GenericTerrestrial => Self {
                baked_layers: vec![
                    BakedTerrainLayer::Height,
                    BakedTerrainLayer::Albedo,
                    BakedTerrainLayer::MaterialId,
                    BakedTerrainLayer::ProvenanceId,
                ],
                analytic_buffers: vec![AnalyticFeatureBuffer::Craters],
                analytic_min_scale_m: 1_000.0,
                excluded_below_scale_m: 500.0,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BakedTerrainLayer {
    Height,
    Albedo,
    MaterialId,
    WaterMask,
    Climate,
    MoistureHistory,
    IceStability,
    ProvenanceId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalyticFeatureBuffer {
    Craters,
    Channels,
    Scarps,
    DuneFields,
    RaySystems,
}

fn compile_airless_impact_moon(
    spec: &PlanetTerrainSpec,
    manifest: &FeatureManifest,
    prior: &TerrainPrior,
    options: FeatureCompileOptions,
) -> Result<BodyData, FeatureCompileError> {
    let FeatureCompileOptions {
        cubemap_resolution,
        crater_count_scale,
        projection,
    } = options;
    let projection = projection.airless_impact();
    let body_id = &manifest.root;
    let global_crust = manifest
        .get(&body_id.child("global_crust"))
        .ok_or_else(|| FeatureCompileError::MissingFeature(body_id.child("global_crust")))?;
    let crater_population = manifest
        .get(&body_id.child("crater_population"))
        .ok_or_else(|| FeatureCompileError::MissingFeature(body_id.child("crater_population")))?;
    let regolith = manifest
        .get(&body_id.child("regolith_garden"))
        .ok_or_else(|| FeatureCompileError::MissingFeature(body_id.child("regolith_garden")))?;
    let space_weathering = manifest
        .get(&body_id.child("space_weathering"))
        .ok_or_else(|| FeatureCompileError::MissingFeature(body_id.child("space_weathering")))?;

    let mut builder = BodyBuilder::new(
        spec.physical.radius_m,
        spec.root_seed,
        composition_for(spec.physical.composition),
        cubemap_resolution,
        spec.physical.age_gyr,
        Some(Vec3::X),
        spec.physical.obliquity_deg.unwrap_or(0.0).to_radians(),
    );

    apply_projection_stage(&mut builder, global_crust.seed.identity, Differentiate);

    let basins = megabasin_defs_from_manifest(manifest)?;
    if !basins.is_empty() {
        apply_projection_stage(
            &mut builder,
            sub_seed(spec.root_seed, "feature_projection:megabasin"),
            Megabasin {
                basins,
                hemispheric_lowering_m: 0.0,
            },
        );
    }

    if !projection.highland_biomes.is_empty() {
        apply_projection_stage(
            &mut builder,
            sub_seed(spec.root_seed, "feature_projection:biomes"),
            Biomes {
                biomes: projection.highland_biomes.clone(),
                rules: projection.highland_biome_rules.clone(),
            },
        );
    }

    let total_count = scaled_count(
        projection.base_crater_count,
        prior.crater_density,
        crater_count_scale,
    );
    apply_projection_stage(
        &mut builder,
        crater_population.seed.children,
        Cratering {
            total_count,
            sfd_slope: param_number(crater_population, "sfd_alpha")
                .map(f64::from)
                .unwrap_or(projection.sfd_slope),
            sfd_slope_small: None,
            sfd_break_radius_m: None,
            min_radius_m: crater_population
                .scale_range_m
                .min_m
                .max(projection.min_crater_radius_m) as f64,
            max_radius_m: crater_population
                .scale_range_m
                .max_m
                .min(projection.max_crater_radius_m) as f64,
            age_bias: projection.crater_age_bias,
            cubemap_bake_threshold_m: projection.cubemap_bake_threshold_m,
            secondary_parent_radius_m: projection.secondary_parent_radius_m,
            secondaries_per_parent: projection.secondaries_per_parent,
            saturation_fraction: projection.crater_saturation_fraction,
            chain_count: projection.chain_count,
            chain_segment_count: projection.chain_segment_count,
            forced_young_count: projection.forced_young_count,
        },
    );

    let mare_targets = projection
        .mare_target_count
        .min(manifest.feature_count_by_kind(FeatureKind::MareFlood) as u32);
    if mare_targets > 0 || projection.mare_additional_crater_count > 0 {
        let mare_seed = manifest
            .features
            .iter()
            .find(|feature| feature.kind == FeatureKind::MareFlood)
            .map(|feature| feature.seed.detail)
            .unwrap_or_else(|| sub_seed(spec.root_seed, "feature_projection:mare_flood"));
        apply_projection_stage(
            &mut builder,
            mare_seed,
            MareFloodStage {
                target_count: mare_targets,
                additional_crater_count: projection.mare_additional_crater_count,
                fill_fraction: projection.mare_fill_fraction,
                near_side_bias: projection.mare_near_side_bias,
                boundary_noise_amplitude_m: projection.mare_boundary_noise_amplitude_m,
                boundary_noise_freq: projection.mare_boundary_noise_freq,
                episode_count: projection.mare_episode_count,
                wrinkle_ridges: projection.mare_wrinkle_ridges,
                procellarum: None,
            },
        );
    }

    if let Some(scarps) = projection.scarps {
        apply_projection_stage(
            &mut builder,
            sub_seed(spec.root_seed, "feature_projection:scarps"),
            Scarps {
                count: scarps.count,
                min_length_m: scarps.min_length_m,
                max_length_m: scarps.max_length_m,
                width_m: scarps.width_m,
                height_m: scarps.height_m,
                curvature: scarps.curvature,
            },
        );
    }

    apply_projection_stage(
        &mut builder,
        regolith.seed.detail,
        Regolith {
            amplitude_m: projection.regolith_amplitude_m,
            characteristic_wavelength_m: projection.regolith_characteristic_wavelength_m,
            crater_density_multiplier: projection.regolith_crater_density_multiplier.max(0.0),
            bake_d_min_m: projection.regolith_bake_d_min_m,
            bake_scale: projection.regolith_bake_scale,
            density_modulation: projection.regolith_density_modulation,
            density_wavelength_m: projection.regolith_density_wavelength_m,
        },
    );

    apply_projection_stage(
        &mut builder,
        space_weathering.seed.detail,
        SpaceWeather {
            highland_mature_albedo: projection
                .highland_biomes
                .is_empty()
                .then_some(projection.highland_mature_albedo),
            highland_fresh_albedo: projection
                .highland_biomes
                .is_empty()
                .then_some(projection.highland_fresh_albedo),
            mare_mature_albedo: projection.mare_mature_albedo,
            mare_fresh_albedo: projection.mare_fresh_albedo,
            mare_tint: projection.mare_tint,
            young_crater_age_threshold: projection.young_crater_age_threshold,
            ray_age_threshold: projection.ray_age_threshold,
            ray_extent_radii: projection.ray_extent_radii,
            ray_count_per_crater: projection.ray_count_per_crater,
            ray_half_width: projection.ray_half_width,
        },
    );

    Ok(builder.build())
}

fn compile_cold_desert_formerly_wet(
    spec: &PlanetTerrainSpec,
    manifest: &FeatureManifest,
    prior: &TerrainPrior,
    options: FeatureCompileOptions,
) -> Result<BodyData, FeatureCompileError> {
    let FeatureCompileOptions {
        cubemap_resolution,
        crater_count_scale,
        projection,
    } = options;
    let body_id = &manifest.root;
    let global_crust = manifest
        .get(&body_id.child("crustal_provinces"))
        .ok_or_else(|| FeatureCompileError::MissingFeature(body_id.child("crustal_provinces")))?;
    let craters = manifest
        .get(&body_id.child("recent_craters"))
        .ok_or_else(|| FeatureCompileError::MissingFeature(body_id.child("recent_craters")))?;

    let projection = projection.cold_desert();
    let mut builder = BodyBuilder::new(
        spec.physical.radius_m,
        spec.root_seed,
        composition_for(spec.physical.composition),
        cubemap_resolution,
        spec.physical.age_gyr,
        None,
        spec.physical.obliquity_deg.unwrap_or(0.0).to_radians(),
    );

    builder.materials = VaelenColdDesertField::material_palette();
    let field = VaelenColdDesertField::new(global_crust.seed, projection.clone());
    bake_surface_field_into_builder(&mut builder, &field);

    let basins = megabasin_defs_from_manifest(manifest)?;
    if !basins.is_empty() {
        // Vaelen's ancient basins should read as buried terrain memory, not
        // clean radial megabasin stamps. Keep the authored definitions for
        // subtle downstream albedo overprint, but leave relief to the crater
        // archive until the cold-desert basin model grows a non-radial pass.
        builder.megabasins = basins;
    }

    if let Some(large_archive) = manifest.get(
        &body_id
            .child("impact_basin_archive")
            .child("large_degraded_crater_archive"),
    ) {
        let total_count = param_number(large_archive, "target_count")
            .unwrap_or(24.0)
            .round()
            .max(0.0) as u32;
        if total_count > 0 {
            apply_projection_stage(
                &mut builder,
                large_archive.seed.children,
                Cratering {
                    total_count,
                    sfd_slope: 1.35,
                    sfd_slope_small: None,
                    sfd_break_radius_m: None,
                    min_radius_m: large_archive.scale_range_m.min_m as f64,
                    max_radius_m: large_archive.scale_range_m.max_m as f64,
                    age_bias: 2.6,
                    cubemap_bake_threshold_m: projection.cubemap_bake_threshold_m,
                    secondary_parent_radius_m: large_archive.scale_range_m.max_m * 0.45,
                    secondaries_per_parent: 3,
                    saturation_fraction: 0.0,
                    chain_count: 0,
                    chain_segment_count: 0,
                    forced_young_count: 0,
                },
            );
        }
    }

    let total_count = scaled_count(
        projection.base_crater_count,
        prior.crater_density,
        crater_count_scale,
    );
    apply_projection_stage(
        &mut builder,
        craters.seed.children,
        Cratering {
            total_count,
            sfd_slope: 1.9,
            sfd_slope_small: None,
            sfd_break_radius_m: None,
            min_radius_m: craters
                .scale_range_m
                .min_m
                .max(projection.min_crater_radius_m) as f64,
            max_radius_m: craters
                .scale_range_m
                .max_m
                .min(projection.max_crater_radius_m) as f64,
            age_bias: 1.95,
            cubemap_bake_threshold_m: projection.cubemap_bake_threshold_m,
            secondary_parent_radius_m: projection.max_crater_radius_m * 0.55,
            secondaries_per_parent: 6,
            saturation_fraction: 0.045,
            chain_count: 1,
            chain_segment_count: 7,
            forced_young_count: 8,
        },
    );

    apply_projection_stage(
        &mut builder,
        sub_seed(spec.root_seed, "feature_projection:vaelen_impact_color"),
        VaelenImpactColor {
            crater_min_radius_m: projection.cubemap_bake_threshold_m.max(3_000.0),
        },
    );

    Ok(builder.build())
}

fn apply_projection_stage<S: Stage>(builder: &mut BodyBuilder, seed: u64, stage: S) {
    builder.stage_seed = seed;
    stage.apply(builder);
}

fn megabasin_defs_from_manifest(
    manifest: &FeatureManifest,
) -> Result<Vec<BasinDef>, FeatureCompileError> {
    manifest
        .features
        .iter()
        .filter(|feature| feature.kind == FeatureKind::Megabasin)
        .map(|feature| {
            let FeatureFootprint::Circle { center, .. } = &feature.footprint else {
                return Err(FeatureCompileError::InvalidFeatureFootprint(
                    feature.id.clone(),
                ));
            };
            let radius_m = param_number(feature, "radius_km")
                .map(|km| km * 1_000.0)
                .unwrap_or(feature.scale_range_m.max_m * 0.5);
            let depth_m = param_number(feature, "depth_km")
                .map(|km| km * 1_000.0)
                .unwrap_or(radius_m * 0.018);
            let ring_count = param_number(feature, "ring_count")
                .map(|value| value.round().max(0.0) as u32)
                .unwrap_or_else(|| basin_ring_count(radius_m));
            Ok(BasinDef {
                center_dir: center.normalize(),
                radius_m,
                depth_m,
                ring_count,
                seed: Some(feature.seed.shape),
            })
        })
        .collect()
}

fn basin_ring_count(radius_m: f32) -> u32 {
    if radius_m >= 300_000.0 {
        4
    } else if radius_m >= 200_000.0 {
        3
    } else if radius_m >= 100_000.0 {
        2
    } else {
        1
    }
}

fn scaled_count(base: u32, density: f32, scale: f32) -> u32 {
    ((base as f32 * density.max(0.01) * scale.max(0.0)).round() as u32).max(1)
}

fn composition_for(class: CompositionClass) -> Composition {
    match class {
        CompositionClass::SilicateDominated => Composition::new(0.93, 0.05, 0.0, 0.02, 0.0),
        CompositionClass::BasalticSilicate => Composition::new(0.90, 0.08, 0.0, 0.02, 0.0),
        CompositionClass::IronRichSilicate => Composition::new(0.68, 0.30, 0.0, 0.02, 0.0),
        CompositionClass::IcySilicate => Composition::new(0.38, 0.0, 0.60, 0.02, 0.0),
    }
}

fn param_number(feature: &FeatureInstance, key: &str) -> Option<f32> {
    feature
        .params
        .iter()
        .find(|param| param.key == key)
        .and_then(|param| match &param.value {
            FeatureParamValue::Number(value) => Some(*value),
            _ => None,
        })
}

fn add_mira_style_features(manifest: &mut FeatureManifest, spec: &PlanetTerrainSpec) {
    let root = manifest.root.clone();
    let radius = spec.physical.radius_m;
    let near = Vec3::X;
    let far = -Vec3::X;

    add_child(
        manifest,
        &root,
        "global_crust",
        FeatureKind::GlobalCrust,
        TerrainEra::CrustFormation,
        FeatureFootprint::Global,
        ScaleRangeM::new(1_000.0, radius * 2.0),
        vec![FeatureParam::text("material", "anorthosite_highland")],
        FeatureLock::Placement,
    );

    let basin_a = add_child(
        manifest,
        &root,
        "near_side_megabasin_a",
        FeatureKind::Megabasin,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Circle {
            center: offset_dir(near, 0.12, 0.24),
            angular_radius_rad: km_to_angle(250.0, radius),
        },
        ScaleRangeM::new(25_000.0, 650_000.0),
        vec![
            FeatureParam::number("radius_km", 250.0),
            FeatureParam::number("depth_km", 6.0),
            FeatureParam::boolean("authored_landmark", true),
        ],
        FeatureLock::Placement,
    );
    add_mira_basin_children(manifest, &basin_a, radius, 250.0);

    let basin_b = add_child(
        manifest,
        &root,
        "near_side_megabasin_b",
        FeatureKind::Megabasin,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Circle {
            center: offset_dir(near, -0.28, -0.18),
            angular_radius_rad: km_to_angle(180.0, radius),
        },
        ScaleRangeM::new(20_000.0, 500_000.0),
        vec![
            FeatureParam::number("radius_km", 180.0),
            FeatureParam::number("depth_km", 4.0),
            FeatureParam::boolean("authored_landmark", true),
        ],
        FeatureLock::Placement,
    );
    add_mira_basin_children(manifest, &basin_b, radius, 180.0);

    add_child(
        manifest,
        &root,
        "far_side_highlands",
        FeatureKind::HighlandProvince,
        TerrainEra::CrustFormation,
        FeatureFootprint::Hemisphere {
            center: far,
            angular_radius_rad: std::f32::consts::FRAC_PI_2,
        },
        ScaleRangeM::new(5_000.0, radius),
        vec![FeatureParam::text("role", "quiet_cratered_far_side")],
        FeatureLock::Unlocked,
    );

    let craters = add_child(
        manifest,
        &root,
        "crater_population",
        FeatureKind::CraterPopulation,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Global,
        ScaleRangeM::new(250.0, 250_000.0),
        vec![
            FeatureParam::number("sfd_alpha", 2.0),
            FeatureParam::number("retention", 0.98),
        ],
        FeatureLock::Unlocked,
    );
    for (name, min_m, max_m, era) in [
        (
            "ancient_large_craters",
            25_000.0,
            250_000.0,
            TerrainEra::HeavyBombardment,
        ),
        (
            "degraded_medium_craters",
            5_000.0,
            50_000.0,
            TerrainEra::HeavyBombardment,
        ),
        (
            "fresh_ray_craters",
            10_000.0,
            80_000.0,
            TerrainEra::RecentSurface,
        ),
        (
            "statistical_microcraters",
            10.0,
            1_500.0,
            TerrainEra::PresentDetail,
        ),
    ] {
        add_child(
            manifest,
            &craters,
            name,
            FeatureKind::CraterCohort,
            era,
            FeatureFootprint::Global,
            ScaleRangeM::new(min_m, max_m),
            Vec::new(),
            FeatureLock::Unlocked,
        );
    }

    add_child(
        manifest,
        &root,
        "regolith_garden",
        FeatureKind::RegolithGarden,
        TerrainEra::PresentDetail,
        FeatureFootprint::Global,
        ScaleRangeM::new(1.0, 5_000.0),
        vec![FeatureParam::number("nominal_depth_m", 8.0)],
        FeatureLock::Unlocked,
    );

    add_child(
        manifest,
        &root,
        "space_weathering",
        FeatureKind::SpaceWeathering,
        TerrainEra::PresentDetail,
        FeatureFootprint::Global,
        ScaleRangeM::new(100.0, radius),
        vec![FeatureParam::number("maturity_strength", 0.85)],
        FeatureLock::Unlocked,
    );
}

fn add_mira_basin_children(
    manifest: &mut FeatureManifest,
    basin: &FeatureId,
    radius_m: f32,
    basin_radius_km: f32,
) {
    let scale = basin_radius_km * 1_000.0;
    add_child(
        manifest,
        basin,
        "rim_uplift",
        FeatureKind::RimUplift,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Unresolved,
        ScaleRangeM::new(scale * 0.02, scale),
        Vec::new(),
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        basin,
        "basin_floor",
        FeatureKind::BasinFloor,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Unresolved,
        ScaleRangeM::new(scale * 0.05, scale),
        Vec::new(),
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        basin,
        "mare_fill",
        FeatureKind::MareFlood,
        TerrainEra::AncientResurfacing,
        FeatureFootprint::Unresolved,
        ScaleRangeM::new(10_000.0, scale),
        vec![
            FeatureParam::text("material", "mare_basalt"),
            FeatureParam::number("near_side_bias", 1.0),
            FeatureParam::number(
                "max_angular_radius_rad",
                km_to_angle(basin_radius_km, radius_m),
            ),
        ],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        basin,
        "secondary_crater_field",
        FeatureKind::SecondaryCraterField,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Unresolved,
        ScaleRangeM::new(500.0, scale * 0.25),
        Vec::new(),
        FeatureLock::Unlocked,
    );
}

fn add_vaelen_style_features(manifest: &mut FeatureManifest, spec: &PlanetTerrainSpec) {
    let root = manifest.root.clone();
    let radius = spec.physical.radius_m;

    add_child(
        manifest,
        &root,
        "crustal_provinces",
        FeatureKind::CrustalProvince,
        TerrainEra::CrustFormation,
        FeatureFootprint::Global,
        ScaleRangeM::new(50_000.0, radius * 2.0),
        vec![FeatureParam::number("target_count", 9.0)],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &root,
        "ancient_highlands",
        FeatureKind::HighlandProvince,
        TerrainEra::CrustFormation,
        FeatureFootprint::Global,
        ScaleRangeM::new(20_000.0, radius),
        vec![FeatureParam::text("material", "basaltic_highland")],
        FeatureLock::Unlocked,
    );

    let lowlands = add_child(
        manifest,
        &root,
        "northern_sedimentary_lowlands",
        FeatureKind::SedimentaryLowlands,
        TerrainEra::FluvialEpoch,
        FeatureFootprint::Hemisphere {
            center: Vec3::Y,
            angular_radius_rad: 1.25,
        },
        ScaleRangeM::new(10_000.0, radius),
        vec![
            FeatureParam::text("material", "sedimentary_rock"),
            FeatureParam::number("fill_strength", 0.72),
        ],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &lowlands,
        "evaporite_floors",
        FeatureKind::EvaporiteBasin,
        TerrainEra::DryingEpoch,
        FeatureFootprint::Unresolved,
        ScaleRangeM::new(2_000.0, 150_000.0),
        vec![FeatureParam::text("material", "evaporite")],
        FeatureLock::Unlocked,
    );

    let basins = add_child(
        manifest,
        &root,
        "impact_basin_archive",
        FeatureKind::ImpactBasinArchive,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Global,
        ScaleRangeM::new(10_000.0, 400_000.0),
        vec![FeatureParam::number("retention", 0.54)],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &basins,
        "southern_pale_megabasin",
        FeatureKind::Megabasin,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Circle {
            center: Vec3::new(-0.28, -0.74, 0.61).normalize(),
            angular_radius_rad: km_to_angle(360.0, radius),
        },
        ScaleRangeM::new(60_000.0, 720_000.0),
        vec![
            FeatureParam::number("radius_km", 360.0),
            FeatureParam::number("depth_km", 3.8),
            FeatureParam::boolean("authored_landmark", true),
        ],
        FeatureLock::Placement,
    );
    add_child(
        manifest,
        &basins,
        "western_dark_basin",
        FeatureKind::Megabasin,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Circle {
            center: Vec3::new(-0.92, -0.10, -0.38).normalize(),
            angular_radius_rad: km_to_angle(245.0, radius),
        },
        ScaleRangeM::new(50_000.0, 490_000.0),
        vec![
            FeatureParam::number("radius_km", 245.0),
            FeatureParam::number("depth_km", 3.2),
            FeatureParam::boolean("authored_landmark", true),
        ],
        FeatureLock::Placement,
    );
    add_child(
        manifest,
        &basins,
        "northeast_buried_basin",
        FeatureKind::Megabasin,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Circle {
            center: Vec3::new(0.66, 0.31, -0.68).normalize(),
            angular_radius_rad: km_to_angle(285.0, radius),
        },
        ScaleRangeM::new(55_000.0, 570_000.0),
        vec![
            FeatureParam::number("radius_km", 285.0),
            FeatureParam::number("depth_km", 3.4),
            FeatureParam::boolean("authored_landmark", true),
        ],
        FeatureLock::Placement,
    );
    add_child(
        manifest,
        &basins,
        "large_degraded_crater_archive",
        FeatureKind::CraterCohort,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Global,
        ScaleRangeM::new(30_000.0, 125_000.0),
        vec![
            FeatureParam::number("target_count", 36.0),
            FeatureParam::number("retention", 0.72),
        ],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &basins,
        "degraded_rims",
        FeatureKind::RimUplift,
        TerrainEra::FluvialEpoch,
        FeatureFootprint::Unresolved,
        ScaleRangeM::new(5_000.0, 250_000.0),
        vec![FeatureParam::number("degradation", 0.65)],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &basins,
        "lakebed_fill",
        FeatureKind::BasinFloor,
        TerrainEra::FluvialEpoch,
        FeatureFootprint::Unresolved,
        ScaleRangeM::new(5_000.0, 250_000.0),
        vec![FeatureParam::text("material", "sedimentary_rock")],
        FeatureLock::Unlocked,
    );

    let channels = add_child(
        manifest,
        &root,
        "ancient_channel_networks",
        FeatureKind::ChannelNetwork,
        TerrainEra::FluvialEpoch,
        FeatureFootprint::Global,
        ScaleRangeM::new(500.0, 700_000.0),
        vec![FeatureParam::number("target_trunks", 6.0)],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &channels,
        "delta_fans",
        FeatureKind::DeltaFan,
        TerrainEra::DryingEpoch,
        FeatureFootprint::Unresolved,
        ScaleRangeM::new(1_000.0, 80_000.0),
        vec![FeatureParam::text("material", "sedimentary_rock")],
        FeatureLock::Unlocked,
    );

    add_child(
        manifest,
        &root,
        "buried_ice_zones",
        FeatureKind::BuriedIceZone,
        TerrainEra::DryingEpoch,
        FeatureFootprint::Band {
            min_lat_deg: 45.0,
            max_lat_deg: 90.0,
        },
        ScaleRangeM::new(1_000.0, radius),
        vec![FeatureParam::number("stability", 0.68)],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &root,
        "volcanic_plains",
        FeatureKind::VolcanicPlain,
        TerrainEra::AncientResurfacing,
        FeatureFootprint::Global,
        ScaleRangeM::new(10_000.0, 300_000.0),
        vec![FeatureParam::text("material", "dark_basalt")],
        FeatureLock::Unlocked,
    );

    let aeolian = add_child(
        manifest,
        &root,
        "aeolian_mantle",
        FeatureKind::AeolianMantle,
        TerrainEra::RecentSurface,
        FeatureFootprint::Global,
        ScaleRangeM::new(10.0, 100_000.0),
        vec![
            FeatureParam::text("material", "rust_dust"),
            FeatureParam::number("activity", 0.82),
        ],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &aeolian,
        "dune_fields",
        FeatureKind::DuneField,
        TerrainEra::PresentDetail,
        FeatureFootprint::Unresolved,
        ScaleRangeM::new(10.0, 10_000.0),
        Vec::new(),
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &aeolian,
        "yardang_fields",
        FeatureKind::YardangField,
        TerrainEra::PresentDetail,
        FeatureFootprint::Unresolved,
        ScaleRangeM::new(100.0, 30_000.0),
        Vec::new(),
        FeatureLock::Unlocked,
    );

    add_child(
        manifest,
        &root,
        "recent_craters",
        FeatureKind::CraterPopulation,
        TerrainEra::RecentSurface,
        FeatureFootprint::Global,
        ScaleRangeM::new(250.0, 120_000.0),
        vec![FeatureParam::number("retention", 0.35)],
        FeatureLock::Unlocked,
    );
}

fn add_homeworld_style_features(manifest: &mut FeatureManifest, spec: &PlanetTerrainSpec) {
    let root = manifest.root.clone();
    let radius = spec.physical.radius_m;
    add_child(
        manifest,
        &root,
        "crustal_provinces",
        FeatureKind::CrustalProvince,
        TerrainEra::CrustFormation,
        FeatureFootprint::Global,
        ScaleRangeM::new(50_000.0, radius * 2.0),
        vec![FeatureParam::number("continental_area_fraction", 0.35)],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &root,
        "aging_orogen_belts",
        FeatureKind::OrogenBelt,
        TerrainEra::AncientResurfacing,
        FeatureFootprint::Global,
        ScaleRangeM::new(10_000.0, 1_000_000.0),
        vec![FeatureParam::number("active_fraction", 0.2)],
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &root,
        "drainage_networks",
        FeatureKind::ChannelNetwork,
        TerrainEra::RecentSurface,
        FeatureFootprint::Global,
        ScaleRangeM::new(100.0, 1_500_000.0),
        vec![FeatureParam::number("erosion_strength", 0.88)],
        FeatureLock::Unlocked,
    );
}

fn add_generic_terrestrial_features(manifest: &mut FeatureManifest, spec: &PlanetTerrainSpec) {
    let root = manifest.root.clone();
    let radius = spec.physical.radius_m;
    add_child(
        manifest,
        &root,
        "crustal_provinces",
        FeatureKind::CrustalProvince,
        TerrainEra::CrustFormation,
        FeatureFootprint::Global,
        ScaleRangeM::new(50_000.0, radius * 2.0),
        Vec::new(),
        FeatureLock::Unlocked,
    );
    add_child(
        manifest,
        &root,
        "impact_basin_archive",
        FeatureKind::ImpactBasinArchive,
        TerrainEra::HeavyBombardment,
        FeatureFootprint::Global,
        ScaleRangeM::new(10_000.0, radius),
        Vec::new(),
        FeatureLock::Unlocked,
    );
}

fn add_authored_feature(manifest: &mut FeatureManifest, authored: &AuthoredFeatureSpec) {
    if let Some(existing) = manifest.get_mut(&authored.id) {
        existing.kind = authored.kind;
        if let Some(seed) = authored.seed_override {
            existing.seed = seed;
        }
        if let Some(footprint) = &authored.footprint {
            existing.footprint = footprint.clone();
        }
        if let Some(scale_range_m) = authored.scale_range_m {
            existing.scale_range_m = scale_range_m;
        }
        if !authored.params.is_empty() {
            existing.params = authored.params.clone();
        }
        existing.lock = authored.lock;
        return;
    }

    let parent = authored
        .parent
        .clone()
        .unwrap_or_else(|| manifest.root.clone());
    let seed = authored
        .seed_override
        .unwrap_or_else(|| manifest.derive_seed(Some(&parent), &authored.id));
    let feature = FeatureInstance::new(
        authored.id.clone(),
        authored.kind,
        Some(parent),
        seed,
        TerrainEra::RecentSurface,
        authored
            .footprint
            .clone()
            .unwrap_or(FeatureFootprint::Unresolved),
        authored
            .scale_range_m
            .unwrap_or_else(|| ScaleRangeM::new(0.0, f32::INFINITY)),
    )
    .with_params(authored.params.clone())
    .with_lock(authored.lock);
    manifest.add_feature(feature);
}

fn add_child(
    manifest: &mut FeatureManifest,
    parent: &FeatureId,
    local_name: &str,
    kind: FeatureKind,
    era: TerrainEra,
    footprint: FeatureFootprint,
    scale_range_m: ScaleRangeM,
    params: Vec<FeatureParam>,
    lock: FeatureLock,
) -> FeatureId {
    let id = parent.child(local_name);
    let seed = manifest.derive_seed(Some(parent), &id);
    let feature = FeatureInstance::new(
        id.clone(),
        kind,
        Some(parent.clone()),
        seed,
        era,
        footprint,
        scale_range_m,
    )
    .with_params(params)
    .with_lock(lock);
    manifest.add_feature(feature);
    id
}

fn km_to_angle(radius_km: f32, body_radius_m: f32) -> f32 {
    radius_km * 1_000.0 / body_radius_m
}

fn offset_dir(base: Vec3, yaw: f32, pitch: f32) -> Vec3 {
    let tangent_a = if base.x.abs() < 0.8 { Vec3::X } else { Vec3::Y };
    let tangent_b = base.cross(tangent_a).normalize();
    (base + tangent_a * yaw + tangent_b * pitch).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_seed_streams_are_independent() {
        let id = FeatureId::from("mira.near_side_megabasin_a");
        let seed = FeatureSeed::derive(42, &id);
        let rerolled = seed.rerolled(FeatureSeedStream::Detail, "try_2");

        assert_eq!(seed.identity, rerolled.identity);
        assert_eq!(seed.placement, rerolled.placement);
        assert_eq!(seed.shape, rerolled.shape);
        assert_ne!(seed.detail, rerolled.detail);
        assert_eq!(seed.children, rerolled.children);
    }

    #[test]
    fn child_seed_is_derived_from_parent_children_stream() {
        let mut manifest = FeatureManifest::new("mira", 99);
        let root = manifest.root.clone();
        let child = add_child(
            &mut manifest,
            &root,
            "global_crust",
            FeatureKind::GlobalCrust,
            TerrainEra::CrustFormation,
            FeatureFootprint::Global,
            ScaleRangeM::new(1.0, 2.0),
            Vec::new(),
            FeatureLock::Unlocked,
        );

        let root_feature = manifest.get(&root).unwrap();
        let expected = FeatureSeed::derive(root_feature.seed.children, &child);
        let child_feature = manifest.get(&child).unwrap();
        assert_eq!(child_feature.seed, expected);
        assert_eq!(root_feature.children, vec![child]);
    }

    #[test]
    fn mira_manifest_contains_seed_addressable_basin_children() {
        let spec = PlanetTerrainSpec::mira_default(1234);
        let manifest = generate_initial_manifest(&spec);

        assert_eq!(manifest.feature_count_by_kind(FeatureKind::Megabasin), 2);
        assert_eq!(manifest.feature_count_by_kind(FeatureKind::MareFlood), 2);
        assert!(
            manifest
                .get(&FeatureId::from("mira.near_side_megabasin_a.mare_fill"))
                .is_some()
        );
        assert!(
            manifest
                .get(&FeatureId::from(
                    "mira.crater_population.statistical_microcraters"
                ))
                .is_some()
        );
    }

    #[test]
    fn vaelen_prior_and_manifest_capture_wet_desert_history() {
        let spec = PlanetTerrainSpec::vaelen_default(5678);
        let plan = plan_initial_compilation(&spec);

        assert!(plan.prior.sediment_mobility > plan.prior.crater_retention);
        assert!(plan.prior.aeolian_activity > 0.75);
        assert!(plan.prior.biomes.iter().any(|biome| {
            biome.role == TerrainBiomeRole::DuneSea
                && !biome.height.generators.is_empty()
                && biome
                    .feature_budgets
                    .iter()
                    .any(|budget| budget.kind == FeatureKind::DuneField)
        }));
        assert!(plan.prior.biomes.iter().any(|biome| {
            biome.role == TerrainBiomeRole::EvaporiteBasin
                && biome
                    .feature_budgets
                    .iter()
                    .any(|budget| budget.kind == FeatureKind::EvaporiteBasin)
        }));
        assert!(
            plan.manifest
                .get(&FeatureId::from(
                    "vaelen.ancient_channel_networks.delta_fans"
                ))
                .is_some()
        );
        assert!(
            plan.manifest
                .get(&FeatureId::from("vaelen.aeolian_mantle.yardang_fields"))
                .is_some()
        );
        assert!(plan.manifest.feature_count_by_kind(FeatureKind::Megabasin) >= 3);
        assert!(
            plan.manifest
                .get(&FeatureId::from(
                    "vaelen.impact_basin_archive.large_degraded_crater_archive"
                ))
                .is_some()
        );
        assert!(
            plan.impostor
                .analytic_buffers
                .contains(&AnalyticFeatureBuffer::Channels)
        );
    }

    #[test]
    fn mira_feature_spec_compiles_to_renderable_body_data() {
        let spec = PlanetTerrainSpec::mira_default(1234);
        let body = compile_initial_body_data(
            &spec,
            FeatureCompileOptions {
                cubemap_resolution: 64,
                crater_count_scale: 0.0005,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(body.radius_m, 869_000.0);
        assert_eq!(body.materials.len(), 4);
        assert_eq!(body.height_cubemap.resolution(), 64);
        assert!(body.height_range.is_finite());
        assert!(body.height_range > 0.0);
        assert!(body.craters.len() >= 10);

        let has_mare = crate::cubemap::CubemapFace::ALL.iter().any(|face| {
            body.material_cubemap
                .face_data(*face)
                .iter()
                .any(|material| *material == crate::stages::MAT_MARE as u8)
        });
        assert!(has_mare);
    }

    #[test]
    fn vaelen_feature_spec_compiles_to_renderable_body_data() {
        let spec = PlanetTerrainSpec::vaelen_default(5678);
        let body = compile_initial_body_data(
            &spec,
            FeatureCompileOptions {
                cubemap_resolution: 64,
                crater_count_scale: 0.01,
                projection: FeatureProjectionConfig::ColdDesert(ColdDesertProjectionConfig {
                    base_crater_count: 2_000,
                    ..Default::default()
                }),
            },
        )
        .unwrap();

        assert_eq!(body.radius_m, 1_130_000.0);
        assert_eq!(body.materials.len(), 5);
        assert_eq!(body.height_cubemap.resolution(), 64);
        assert!(body.height_range.is_finite());
        assert!(body.height_range > 0.0);
        assert!(!body.craters.is_empty());
        assert!(
            body.craters
                .iter()
                .filter(|crater| crater.radius_m >= 30_000.0)
                .count()
                >= 24
        );
    }

    #[test]
    fn authored_seed_override_updates_existing_generated_feature() {
        let mut spec = PlanetTerrainSpec::mira_default(42);
        let crater_id = FeatureId::from("mira.crater_population");
        spec.authored_features.push(AuthoredFeatureSpec {
            id: crater_id.clone(),
            kind: FeatureKind::CraterPopulation,
            parent: Some(FeatureId::from("mira")),
            seed_override: Some(FeatureSeed::derive(99_999, &crater_id)),
            footprint: None,
            scale_range_m: None,
            params: Vec::new(),
            lock: FeatureLock::Full,
        });

        let manifest = generate_initial_manifest(&spec);
        let crater_feature = manifest.get(&crater_id).unwrap();
        assert_eq!(crater_feature.lock, FeatureLock::Full);
        assert_eq!(crater_feature.seed, FeatureSeed::derive(99_999, &crater_id));
        assert_eq!(
            manifest.feature_count_by_kind(FeatureKind::CraterPopulation),
            1
        );
    }
}
