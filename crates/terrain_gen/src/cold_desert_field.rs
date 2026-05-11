//! Cold-desert continuous surface field.

use bevy_erosion_filter::cpu::{ErosionFilterParams, erosion_filter};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::biome_mask::{
    BiomeMaskContext, BiomeMaskExpr, BiomeMaskPlan, BiomeMaskRule, BiomeMaskSeedStream,
    BiomeMaskSeeds, BiomeMaskWeights,
};
use crate::feature_compiler::{ColdDesertProjectionConfig, FeatureSeed};
use crate::noise::fbm3;
use crate::seeding::sub_seed;
use crate::surface_field::{
    BiomeMix, ReliefPalette, SurfaceField, SurfaceFieldSample, SurfaceMaterialMix, mix3,
    scale_visibility, smoothstep,
};
use crate::types::{DuneSea, ImpactColorPalette, Material};

pub const COLD_DESERT_MAT_RUST_DUST: u8 = 0;
pub const COLD_DESERT_MAT_DARK_BASALT: u8 = 1;
pub const COLD_DESERT_MAT_PALE_SEDIMENT: u8 = 2;
pub const COLD_DESERT_MAT_DUNE_SAND: u8 = 3;
pub const COLD_DESERT_MAT_EVAPORITE: u8 = 4;
pub const COLD_DESERT_BIOME_COUNT: usize = 8;
pub const COLD_DESERT_SHIELD_VOLCANO_RADIUS_M: f32 = 390_000.0;
pub const COLD_DESERT_SHIELD_VOLCANO_HEIGHT_M: f32 = 7_200.0;

const COLD_DESERT_SHIELD_VOLCANO_RADIUS_RAD: f32 = 0.345;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColdDesertStyle {
    #[serde(default = "default_cold_desert_material_palette")]
    pub material_palette: Vec<Material>,
    #[serde(default = "default_cold_desert_biome_debug_colors")]
    pub biome_debug_colors_srgb: [[u8; 3]; COLD_DESERT_BIOME_COUNT],
    #[serde(default = "default_cold_desert_relief_palettes")]
    pub relief_palettes: Vec<ReliefPalette>,
    #[serde(default)]
    pub shield_volcano: Option<ColdDesertShieldVolcano>,
    #[serde(default = "default_cold_desert_pale_basin_anchors")]
    pub pale_basin_anchors: Vec<ColdDesertPaleBasinAnchor>,
    #[serde(default = "default_cold_desert_dark_province_anchors")]
    pub dark_province_anchors: Vec<ColdDesertBiomeAnchor>,
    #[serde(default = "default_cold_desert_dune_texture_center")]
    pub dune_texture_center: Vec3,
    #[serde(default = "default_cold_desert_dune_regions")]
    pub dune_regions: Vec<ColdDesertDuneRegion>,
    #[serde(default)]
    pub impact_palette: ImpactColorPalette,
}

impl ColdDesertStyle {
    pub fn vaelen() -> Self {
        Self::default()
    }

    pub fn material_palette(&self) -> Vec<Material> {
        self.material_palette.clone()
    }

    pub fn relief_palettes(&self) -> Vec<ReliefPalette> {
        self.relief_palettes.clone()
    }

    pub fn dune_regions(&self, radius_m: f32, root_seed: u64) -> Vec<DuneSea> {
        self.dune_regions
            .iter()
            .map(|region| region.to_dune_sea(radius_m, root_seed))
            .collect()
    }
}

impl Default for ColdDesertStyle {
    fn default() -> Self {
        Self {
            material_palette: default_cold_desert_material_palette(),
            biome_debug_colors_srgb: default_cold_desert_biome_debug_colors(),
            relief_palettes: default_cold_desert_relief_palettes(),
            shield_volcano: Some(ColdDesertShieldVolcano {
                center_dir: Vec3::new(0.74, 0.55, -0.38).normalize(),
                radius_m: COLD_DESERT_SHIELD_VOLCANO_RADIUS_M,
                height_m: COLD_DESERT_SHIELD_VOLCANO_HEIGHT_M,
                radius_rad: COLD_DESERT_SHIELD_VOLCANO_RADIUS_RAD,
                material_id: COLD_DESERT_MAT_DARK_BASALT as u32,
            }),
            pale_basin_anchors: default_cold_desert_pale_basin_anchors(),
            dark_province_anchors: default_cold_desert_dark_province_anchors(),
            dune_texture_center: default_cold_desert_dune_texture_center(),
            dune_regions: default_cold_desert_dune_regions(),
            impact_palette: ImpactColorPalette::vaelen_desert(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColdDesertShieldVolcano {
    pub center_dir: Vec3,
    pub radius_m: f32,
    pub height_m: f32,
    pub radius_rad: f32,
    pub material_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColdDesertPaleBasinAnchor {
    pub center_dir: Vec3,
    pub broad_radius_rad: f32,
    pub broad_feather_rad: f32,
    pub broad_weight: f32,
    pub biome_radius_rad: f32,
    pub biome_feather_rad: f32,
    pub biome_weight: f32,
    pub evaporite_radius_rad: f32,
    pub evaporite_feather_rad: f32,
    pub evaporite_weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColdDesertBiomeAnchor {
    pub center_dir: Vec3,
    pub radius_rad: f32,
    pub feather_rad: f32,
    pub weight: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColdDesertDuneRegion {
    pub center: Vec3,
    pub radius_rad: f32,
    pub feather_rad: f32,
    pub axis: Vec3,
    pub lambda_draa_m: f32,
    pub amplitude_draa_m: f32,
    pub lambda_dune_m: f32,
    pub amplitude_dune_m: f32,
    pub alpha_skew: f32,
    pub albedo_crest_lin: [f32; 3],
    pub crest_strength: f32,
    pub warp_scale: f32,
    pub warp_freq: f32,
    pub seed_salt: String,
}

impl ColdDesertDuneRegion {
    fn to_dune_sea(&self, radius_m: f32, root_seed: u64) -> DuneSea {
        let center = self.center.normalize();
        DuneSea {
            center,
            radius_rad: self.radius_rad,
            feather_rad: self.feather_rad,
            axis_tangent: tangent(center, self.axis),
            lambda_draa_m: self.lambda_draa_m,
            amplitude_draa_m: self.amplitude_draa_m,
            lambda_dune_m: self.lambda_dune_m,
            amplitude_dune_m: self.amplitude_dune_m,
            alpha_skew: self.alpha_skew,
            albedo_crest_lin: self.albedo_crest_lin,
            crest_strength: self.crest_strength,
            warp_amp_unit: self.warp_scale * self.lambda_draa_m / radius_m.max(1.0),
            warp_freq: self.warp_freq,
            seed: sub_seed(root_seed, &self.seed_salt),
        }
    }
}

fn tangent(center: Vec3, axis: Vec3) -> Vec3 {
    let projected = axis - center * axis.dot(center);
    projected.try_normalize().unwrap_or_else(|| {
        let up = if center.x.abs() < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        up.cross(center).normalize()
    })
}

fn default_cold_desert_material_palette() -> Vec<Material> {
    vec![
        Material {
            albedo: [0.50, 0.17, 0.090],
            roughness: 0.88,
        },
        Material {
            albedo: [0.12, 0.065, 0.050],
            roughness: 0.72,
        },
        Material {
            albedo: [0.64, 0.44, 0.28],
            roughness: 0.82,
        },
        Material {
            albedo: [0.66, 0.29, 0.125],
            roughness: 0.92,
        },
        Material {
            albedo: [0.76, 0.64, 0.44],
            roughness: 0.68,
        },
    ]
}

fn default_cold_desert_biome_debug_colors() -> [[u8; 3]; COLD_DESERT_BIOME_COUNT] {
    [
        [184, 86, 45],
        [202, 112, 52],
        [221, 198, 143],
        [62, 58, 55],
        [132, 78, 62],
        [224, 94, 34],
        [206, 192, 174],
        [120, 109, 98],
    ]
}

fn default_cold_desert_pale_basin_anchors() -> Vec<ColdDesertPaleBasinAnchor> {
    vec![
        ColdDesertPaleBasinAnchor {
            center_dir: Vec3::new(-0.34, -0.22, 0.91).normalize(),
            broad_radius_rad: 0.26,
            broad_feather_rad: 0.70,
            broad_weight: 0.42,
            biome_radius_rad: 0.34,
            biome_feather_rad: 0.92,
            biome_weight: 0.52,
            evaporite_radius_rad: 0.10,
            evaporite_feather_rad: 0.36,
            evaporite_weight: 0.18,
        },
        ColdDesertPaleBasinAnchor {
            center_dir: Vec3::new(0.72, 0.18, -0.67).normalize(),
            broad_radius_rad: 0.18,
            broad_feather_rad: 0.56,
            broad_weight: 0.34,
            biome_radius_rad: 0.22,
            biome_feather_rad: 0.64,
            biome_weight: 0.34,
            evaporite_radius_rad: 0.08,
            evaporite_feather_rad: 0.28,
            evaporite_weight: 0.14,
        },
        ColdDesertPaleBasinAnchor {
            center_dir: Vec3::new(-0.88, 0.20, 0.38).normalize(),
            broad_radius_rad: 0.14,
            broad_feather_rad: 0.44,
            broad_weight: 0.20,
            biome_radius_rad: 0.20,
            biome_feather_rad: 0.54,
            biome_weight: 0.22,
            evaporite_radius_rad: 0.0,
            evaporite_feather_rad: 0.0,
            evaporite_weight: 0.0,
        },
    ]
}

fn default_cold_desert_dark_province_anchors() -> Vec<ColdDesertBiomeAnchor> {
    vec![
        ColdDesertBiomeAnchor {
            center_dir: Vec3::new(-0.62, -0.08, -0.78).normalize(),
            radius_rad: 0.42,
            feather_rad: 1.02,
            weight: 0.92,
        },
        ColdDesertBiomeAnchor {
            center_dir: Vec3::new(0.50, 0.12, 0.86).normalize(),
            radius_rad: 0.28,
            feather_rad: 0.76,
            weight: 0.62,
        },
    ]
}

fn default_cold_desert_dune_texture_center() -> Vec3 {
    Vec3::new(0.18, 0.36, 0.92).normalize()
}

fn default_cold_desert_dune_regions() -> Vec<ColdDesertDuneRegion> {
    vec![
        ColdDesertDuneRegion {
            center: Vec3::new(0.00, -0.08, 1.0).normalize(),
            radius_rad: 0.86,
            feather_rad: 0.20,
            axis: Vec3::new(0.88, -0.42, 0.0),
            lambda_draa_m: 68_000.0,
            amplitude_draa_m: 230.0,
            lambda_dune_m: 220.0,
            amplitude_dune_m: 9.0,
            alpha_skew: 0.84,
            albedo_crest_lin: [0.78, 0.41, 0.18],
            crest_strength: 0.32,
            warp_scale: 0.82,
            warp_freq: 2.1,
            seed_salt: "cold_desert.dune_sea.basin_sheet".to_string(),
        },
        ColdDesertDuneRegion {
            center: Vec3::new(-0.36, -0.04, 0.93).normalize(),
            radius_rad: 0.40,
            feather_rad: 0.12,
            axis: Vec3::new(0.70, -0.64, 0.0),
            lambda_draa_m: 52_000.0,
            amplitude_draa_m: 195.0,
            lambda_dune_m: 190.0,
            amplitude_dune_m: 8.2,
            alpha_skew: 0.83,
            albedo_crest_lin: [0.76, 0.36, 0.16],
            crest_strength: 0.30,
            warp_scale: 0.82,
            warp_freq: 2.7,
            seed_salt: "cold_desert.dune_sea.west_gulf".to_string(),
        },
        ColdDesertDuneRegion {
            center: Vec3::new(0.40, 0.10, 0.91).normalize(),
            radius_rad: 0.36,
            feather_rad: 0.11,
            axis: Vec3::new(0.86, -0.36, 0.0),
            lambda_draa_m: 44_000.0,
            amplitude_draa_m: 180.0,
            lambda_dune_m: 170.0,
            amplitude_dune_m: 7.8,
            alpha_skew: 0.84,
            albedo_crest_lin: [0.75, 0.35, 0.16],
            crest_strength: 0.30,
            warp_scale: 0.82,
            warp_freq: 3.0,
            seed_salt: "cold_desert.dune_sea.east_embayment".to_string(),
        },
        ColdDesertDuneRegion {
            center: Vec3::new(0.08, -0.22, 0.97).normalize(),
            radius_rad: 0.22,
            feather_rad: 0.09,
            axis: Vec3::new(0.92, -0.30, 0.0),
            lambda_draa_m: 34_000.0,
            amplitude_draa_m: 145.0,
            lambda_dune_m: 150.0,
            amplitude_dune_m: 7.1,
            alpha_skew: 0.82,
            albedo_crest_lin: [0.78, 0.38, 0.17],
            crest_strength: 0.25,
            warp_scale: 0.78,
            warp_freq: 3.6,
            seed_salt: "cold_desert.dune_sea.active_core".to_string(),
        },
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColdDesertBiome {
    RustDustPlain,
    DuneBasin,
    PaleEvaporiteBasin,
    DarkVolcanicProvince,
    RuggedBadlands,
    /// Bright oxide-stained uplands. Lives on highland_ridges; distinct from
    /// the muted RustDustPlain in saturation and altitude bias.
    OxideHighland,
    /// Pale frost+dust mantle on high-latitude regions. Lobed by fbm so the
    /// boundary doesn't read as a circular cap from orbit.
    PolarVeneer,
    /// Gray ash mantle around the dark volcanic provinces — dust-fall
    /// transition between rust plain and basalt.
    AshMantle,
}

impl ColdDesertBiome {
    pub const ALL: [Self; COLD_DESERT_BIOME_COUNT] = [
        Self::RustDustPlain,
        Self::DuneBasin,
        Self::PaleEvaporiteBasin,
        Self::DarkVolcanicProvince,
        Self::RuggedBadlands,
        Self::OxideHighland,
        Self::PolarVeneer,
        Self::AshMantle,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::RustDustPlain => "rust_dust_plain",
            Self::DuneBasin => "dune_basin",
            Self::PaleEvaporiteBasin => "pale_evaporite_basin",
            Self::DarkVolcanicProvince => "dark_volcanic_province",
            Self::RuggedBadlands => "rugged_badlands",
            Self::OxideHighland => "oxide_highland",
            Self::PolarVeneer => "polar_veneer",
            Self::AshMantle => "ash_mantle",
        }
    }

    pub fn color_srgb(self) -> [u8; 3] {
        match self {
            Self::RustDustPlain => [184, 86, 45],
            Self::DuneBasin => [202, 112, 52],
            Self::PaleEvaporiteBasin => [221, 198, 143],
            Self::DarkVolcanicProvince => [62, 58, 55],
            Self::RuggedBadlands => [132, 78, 62],
            Self::OxideHighland => [224, 94, 34],
            Self::PolarVeneer => [206, 192, 174],
            Self::AshMantle => [120, 109, 98],
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::RustDustPlain => 0,
            Self::DuneBasin => 1,
            Self::PaleEvaporiteBasin => 2,
            Self::DarkVolcanicProvince => 3,
            Self::RuggedBadlands => 4,
            Self::OxideHighland => 5,
            Self::PolarVeneer => 6,
            Self::AshMantle => 7,
        }
    }

    fn from_index(index: usize) -> Self {
        match index {
            1 => Self::DuneBasin,
            2 => Self::PaleEvaporiteBasin,
            3 => Self::DarkVolcanicProvince,
            4 => Self::RuggedBadlands,
            5 => Self::OxideHighland,
            6 => Self::PolarVeneer,
            7 => Self::AshMantle,
            _ => Self::RustDustPlain,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColdDesertBiomeWeights {
    pub rust_dust_plain: f32,
    pub dune_basin: f32,
    pub pale_evaporite_basin: f32,
    pub dark_volcanic_province: f32,
    pub rugged_badlands: f32,
    pub oxide_highland: f32,
    pub polar_veneer: f32,
    pub ash_mantle: f32,
}

impl ColdDesertBiomeWeights {
    fn from_mask(weights: BiomeMaskWeights<COLD_DESERT_BIOME_COUNT>) -> Self {
        Self {
            rust_dust_plain: weights.weights[ColdDesertBiome::RustDustPlain.index()],
            dune_basin: weights.weights[ColdDesertBiome::DuneBasin.index()],
            pale_evaporite_basin: weights.weights[ColdDesertBiome::PaleEvaporiteBasin.index()],
            dark_volcanic_province: weights.weights[ColdDesertBiome::DarkVolcanicProvince.index()],
            rugged_badlands: weights.weights[ColdDesertBiome::RuggedBadlands.index()],
            oxide_highland: weights.weights[ColdDesertBiome::OxideHighland.index()],
            polar_veneer: weights.weights[ColdDesertBiome::PolarVeneer.index()],
            ash_mantle: weights.weights[ColdDesertBiome::AshMantle.index()],
        }
    }

    pub fn dominant(self) -> ColdDesertBiome {
        let weights = [
            self.rust_dust_plain,
            self.dune_basin,
            self.pale_evaporite_basin,
            self.dark_volcanic_province,
            self.rugged_badlands,
            self.oxide_highland,
            self.polar_veneer,
            self.ash_mantle,
        ];
        let index = BiomeMaskWeights { weights }.dominant_index();
        ColdDesertBiome::from_index(index)
    }

    pub fn weight_for(self, biome: ColdDesertBiome) -> f32 {
        match biome {
            ColdDesertBiome::RustDustPlain => self.rust_dust_plain,
            ColdDesertBiome::DuneBasin => self.dune_basin,
            ColdDesertBiome::PaleEvaporiteBasin => self.pale_evaporite_basin,
            ColdDesertBiome::DarkVolcanicProvince => self.dark_volcanic_province,
            ColdDesertBiome::RuggedBadlands => self.rugged_badlands,
            ColdDesertBiome::OxideHighland => self.oxide_highland,
            ColdDesertBiome::PolarVeneer => self.polar_veneer,
            ColdDesertBiome::AshMantle => self.ash_mantle,
        }
    }

    pub fn debug_color_srgb(self) -> [u8; 3] {
        self.debug_color_srgb_with_style(&ColdDesertStyle::default())
    }

    pub fn debug_color_srgb_with_style(self, style: &ColdDesertStyle) -> [u8; 3] {
        let mut rgb = [0.0; 3];
        for biome in ColdDesertBiome::ALL {
            let color = style
                .biome_debug_colors_srgb
                .get(biome.index())
                .copied()
                .unwrap_or_else(|| biome.color_srgb());
            let w = self.weight_for(biome);
            rgb[0] += color[0] as f32 * w;
            rgb[1] += color[1] as f32 * w;
            rgb[2] += color[2] as f32 * w;
        }
        [
            rgb[0].clamp(0.0, 255.0) as u8,
            rgb[1].clamp(0.0, 255.0) as u8,
            rgb[2].clamp(0.0, 255.0) as u8,
        ]
    }
}

#[derive(Clone, Debug)]
pub struct ColdDesertField {
    root_seed: FeatureSeed,
    projection: ColdDesertProjectionConfig,
    style: ColdDesertStyle,
    biome_plan: BiomeMaskPlan<COLD_DESERT_BIOME_COUNT>,
}

impl ColdDesertField {
    pub fn new(root_seed: FeatureSeed, projection: ColdDesertProjectionConfig) -> Self {
        Self::with_style(root_seed, projection, ColdDesertStyle::default())
    }

    pub fn with_style(
        root_seed: FeatureSeed,
        projection: ColdDesertProjectionConfig,
        style: ColdDesertStyle,
    ) -> Self {
        let biome_plan = cold_desert_biome_mask_plan(&projection, &style);
        Self {
            root_seed,
            projection,
            style,
            biome_plan,
        }
    }

    pub fn style(&self) -> &ColdDesertStyle {
        &self.style
    }

    pub fn sample_biomes(&self, dir: Vec3) -> ColdDesertBiomeWeights {
        sample_cold_desert_biomes(dir, self.root_seed, &self.biome_plan)
    }

    pub fn debug_biome_color_srgb(&self, dir: Vec3) -> [u8; 3] {
        self.sample_biomes(dir)
            .debug_color_srgb_with_style(&self.style)
    }

    pub fn sample_suture_debug(&self, dir: Vec3) -> ColdDesertSutureDebug {
        let macro_n = fbm_dir(dir, self.root_seed.shape, "macro", 1.15, 5, 0.55);
        let regional_n = fbm_dir(dir, self.root_seed.shape, "regional", 2.7, 4, 0.55);
        let highland_ridges = ridge(fbm_dir(
            dir,
            self.root_seed.shape,
            "highland_ridge",
            5.4,
            4,
            0.52,
        ));
        let lowland_bias = smoothstep(0.72, -0.20, macro_n + regional_n * 0.35);
        let contact = dune_basin_contact(dir, self.root_seed, lowland_bias, highland_ridges);

        ColdDesertSutureDebug {
            paleo_lowland: contact.paleo_lowland,
            dune_plate: contact.dune_plate,
            highland_plate: contact.highland_plate,
            suture_crest: contact.suture_crest,
            mountain_web: contact.mountain_web,
            dune_toe: contact.dune_toe,
        }
    }

    pub fn sample_dune_basin_mask(&self, dir: Vec3) -> ColdDesertDuneBasinMask {
        let macro_n = fbm_dir(dir, self.root_seed.shape, "macro", 1.15, 5, 0.55);
        let regional_n = fbm_dir(dir, self.root_seed.shape, "regional", 2.7, 4, 0.55);
        let highland_ridges = ridge(fbm_dir(
            dir,
            self.root_seed.shape,
            "highland_ridge",
            5.4,
            4,
            0.52,
        ));
        let lowland_bias = smoothstep(0.72, -0.20, macro_n + regional_n * 0.35);
        let contact = dune_basin_contact(dir, self.root_seed, lowland_bias, highland_ridges);

        ColdDesertDuneBasinMask {
            signed: contact.signed,
            fill: smoothstep(-0.035, 0.010, contact.signed),
        }
    }

    pub fn default_material_palette() -> Vec<Material> {
        ColdDesertStyle::default().material_palette()
    }

    pub fn shield_volcano_center_dir() -> Vec3 {
        ColdDesertStyle::default()
            .shield_volcano
            .map(|shield| shield.center_dir)
            .unwrap_or(Vec3::Y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColdDesertSutureDebug {
    pub paleo_lowland: f32,
    pub dune_plate: f32,
    pub highland_plate: f32,
    pub suture_crest: f32,
    pub mountain_web: f32,
    pub dune_toe: f32,
}

impl ColdDesertSutureDebug {
    pub fn debug_color_srgb(self) -> [u8; 3] {
        let r = self.paleo_lowland * 72.0
            + self.dune_plate * 168.0
            + self.dune_toe * 205.0
            + self.suture_crest * 255.0
            + self.mountain_web * 80.0;
        let g = self.paleo_lowland * 54.0
            + self.dune_plate * 125.0
            + self.dune_toe * 165.0
            + self.suture_crest * 235.0
            + self.mountain_web * 155.0;
        let b = self.highland_plate * 36.0 + self.suture_crest * 215.0 + self.mountain_web * 255.0;

        [
            r.clamp(0.0, 255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.clamp(0.0, 255.0) as u8,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColdDesertDuneBasinMask {
    pub signed: f32,
    pub fill: f32,
}

impl SurfaceField for ColdDesertField {
    fn sample(&self, dir: Vec3, sample_scale_m: f32) -> SurfaceFieldSample {
        sample_cold_desert(
            dir,
            sample_scale_m,
            self.root_seed,
            &self.projection,
            &self.style,
            &self.biome_plan,
        )
    }
}

fn sample_cold_desert(
    dir: Vec3,
    sample_scale_m: f32,
    root_seed: FeatureSeed,
    projection: &ColdDesertProjectionConfig,
    style: &ColdDesertStyle,
    biome_plan: &BiomeMaskPlan<COLD_DESERT_BIOME_COUNT>,
) -> SurfaceFieldSample {
    let relief = projection.relief_scale_m.max(0.05);
    let macro_n = fbm_dir(dir, root_seed.shape, "macro", 1.15, 5, 0.55);
    let regional_n = fbm_dir(dir, root_seed.shape, "regional", 2.7, 4, 0.55);
    let texture_visibility = scale_visibility(sample_scale_m, 35_000.0);
    let fine_visibility = scale_visibility(sample_scale_m, 6_000.0);
    let texture_n = fbm_dir(dir, root_seed.detail, "texture", 11.0, 4, 0.52) * texture_visibility;
    let fine_n = fbm_dir(dir, root_seed.detail, "fine", 38.0, 3, 0.50) * fine_visibility;

    let highland_ridges = ridge(fbm_dir(
        dir,
        root_seed.shape,
        "highland_ridge",
        5.4,
        4,
        0.52,
    ));

    let lowland_bias = smoothstep(0.72, -0.20, macro_n + regional_n * 0.35);
    let highland_bias = smoothstep(-0.18, 0.70, macro_n + highland_ridges * 0.25);
    let biomes = cold_desert_biome_weights(
        dir,
        root_seed,
        macro_n,
        regional_n,
        highland_ridges,
        lowland_bias,
        biome_plan,
    );
    let shield = shield_volcano_sample(dir, root_seed, sample_scale_m, style);
    let dune_contact = dune_basin_contact(dir, root_seed, lowland_bias, highland_ridges);
    let basin_dune_fill = (dune_contact.paleo_lowland * projection.dune_strength).clamp(0.0, 1.0);
    let basin_sediment = lowland_sediment_coherence(
        dir,
        root_seed,
        dune_contact.paleo_lowland,
        dune_contact.dune_plate,
    );
    let basin_smoothing = (dune_contact.paleo_lowland * 0.58
        + dune_contact.dune_plate * 0.24
        + biomes.pale_evaporite_basin * 0.36)
        .clamp(0.0, 0.86);
    let biome_height =
        sample_biome_height_generators(dir, root_seed, biomes, &projection.biome_height_generators);

    let macro_height = (macro_n * 1_700.0 + regional_n * 720.0) * relief;
    let highland_height = highland_ridges.powf(2.6) * 520.0 * relief;
    let texture_height = texture_n * 150.0 * relief;
    let fine_height = fine_n * 34.0 * relief;
    let mut height_m = macro_height
        + highland_height * (1.0 - basin_smoothing)
        + texture_height * (1.0 - basin_smoothing * 0.55)
        + fine_height * (1.0 - basin_smoothing * 0.45)
        + biome_height * relief;
    height_m -= dune_contact.paleo_lowland * 520.0 * relief;
    height_m -= dune_contact.dune_plate * 160.0 * relief;
    height_m += dune_contact.highland_plate * 560.0 * relief;
    height_m -= biomes.pale_evaporite_basin * 210.0 * relief;
    height_m += biomes.rugged_badlands * 230.0 * relief;
    height_m += dune_contact.suture_crest * 1_280.0 * relief;
    height_m += dune_contact.mountain_web * 680.0 * relief;
    height_m -= dune_contact.dune_toe * 210.0 * relief;
    height_m += dune_contact.shoreline_scarp * 520.0 * relief;
    height_m += dune_contact.shelf_break * 330.0 * relief;
    height_m += dune_contact.strandline * 115.0 * relief;
    height_m += dune_contact.interior_highs * 700.0 * relief;
    height_m += dune_contact.floor_undulation * 180.0 * relief;
    height_m += shield.height_m * relief;

    let dust_tone = fbm_dir(dir, root_seed.detail, "dust_tone", 0.85, 4, 0.54) * 0.030
        + fbm_dir(dir, root_seed.detail, "oxide_swells", 1.8, 3, 0.52) * 0.016;
    let mut albedo = [
        (0.45 + dust_tone + highland_bias * 0.025 - lowland_bias * 0.010).clamp(0.03, 0.92),
        (0.18 + dust_tone * 0.48 + highland_bias * 0.008 - lowland_bias * 0.006).clamp(0.03, 0.92),
        (0.095 + dust_tone * 0.26 - lowland_bias * 0.003).clamp(0.03, 0.92),
    ];

    let ocean_floor_color = mix3(
        [0.47, 0.25, 0.11],
        [0.58, 0.37, 0.20],
        smoothstep(0.18, 0.72, lowland_bias + basin_sediment * 0.24),
    );
    albedo = mix3(
        albedo,
        ocean_floor_color,
        (basin_sediment * 0.10).clamp(0.0, 0.18),
    );
    albedo = mix3(
        albedo,
        [0.68, 0.50, 0.31],
        (dune_contact.strandline * 0.12 + dune_contact.shelf_break * 0.06).clamp(0.0, 0.16),
    );
    albedo = mix3(
        albedo,
        [0.25, 0.16, 0.11],
        (dune_contact.shoreline_scarp * 0.14 + dune_contact.interior_highs * 0.10).clamp(0.0, 0.20),
    );

    let rugged_color = mix3(
        [0.30, 0.17, 0.12],
        [0.43, 0.22, 0.13],
        biomes.rust_dust_plain,
    );
    albedo = mix3(
        albedo,
        rugged_color,
        (biomes.rugged_badlands * 0.24).clamp(0.0, 0.34),
    );

    let pale_biome_color = [0.64, 0.44, 0.28];
    albedo = mix3(
        albedo,
        pale_biome_color,
        (biomes.pale_evaporite_basin * 0.26).clamp(0.0, 0.38),
    );
    albedo = mix3(
        albedo,
        [0.15, 0.12, 0.095],
        (dune_contact.suture_crest * 0.96).clamp(0.0, 0.98),
    );
    albedo = mix3(
        albedo,
        [0.24, 0.18, 0.13],
        (dune_contact.mountain_web * 0.82).clamp(0.0, 0.86),
    );

    let mut dark_score = 0.0;
    let mut sediment_score = 0.0;
    let mut evaporite_score = 0.0;
    let dune_score = 0.0;

    let dark_belt = biomes.dark_volcanic_province * projection.volcanic_dark_strength;
    let dark_belt_albedo = dark_belt * (1.0 - shield.apron * 0.62);
    if dark_belt_albedo > 0.05 {
        dark_score = dark_belt.clamp(0.0, 1.0);
        let dark_color = mix3(
            [0.19, 0.095, 0.065],
            [0.12, 0.060, 0.048],
            smoothstep(0.50, 0.95, dark_belt),
        );
        albedo = mix3(albedo, dark_color, dark_belt_albedo.clamp(0.0, 0.54));
        height_m -= dark_belt_albedo * 260.0 * relief;
    }

    if shield.apron > 0.01 {
        let flank_color = shield_volcano_albedo(shield);
        albedo = mix3(albedo, flank_color, (shield.apron * 0.70).clamp(0.0, 0.76));
        dark_score = dark_score
            .max(shield.apron * 0.40 + shield.basal_scarp * 0.38 + shield.caldera_rim * 0.36);
    }

    let pale_cap = style
        .pale_basin_anchors
        .iter()
        .map(|anchor| {
            cap_mask(
                dir,
                anchor.center_dir,
                anchor.broad_radius_rad,
                anchor.broad_feather_rad,
            ) * anchor.broad_weight
        })
        .sum::<f32>();
    let pale_coherence = smoothstep(
        0.18,
        0.92,
        fbm_dir(dir, root_seed.placement, "pale_basin", 1.25, 4, 0.55)
            + pale_cap
            + lowland_bias * 0.24
            + basin_sediment * 0.14
            + biomes.pale_evaporite_basin * 0.58
            - basin_dune_fill * 0.54
            - dark_belt * 0.18,
    );
    let sediment_w = (pale_coherence * 0.56 + biomes.pale_evaporite_basin * 0.62).clamp(0.0, 1.0)
        * projection.pale_basin_strength
        * 0.78
        * (1.0 - basin_dune_fill * 0.92).clamp(0.0, 1.0);
    if sediment_w > 0.04 {
        let evaporite_anchor_w = style
            .pale_basin_anchors
            .iter()
            .map(|anchor| {
                if anchor.evaporite_weight <= 0.0 {
                    return 0.0;
                }
                cap_mask(
                    dir,
                    anchor.center_dir,
                    anchor.evaporite_radius_rad,
                    anchor.evaporite_feather_rad,
                ) * anchor.evaporite_weight
            })
            .sum::<f32>();
        let evaporite = (smoothstep(0.62, 0.96, sediment_w + ridge(texture_n) * 0.05) * 0.35
            + evaporite_anchor_w)
            * (0.72 + biomes.pale_evaporite_basin * 0.36)
            * sediment_w;
        let sediment = mix3([0.50, 0.30, 0.18], [0.60, 0.40, 0.24], evaporite);
        albedo = mix3(albedo, sediment, sediment_w.clamp(0.0, 0.30));
        if evaporite > 0.015 {
            albedo = mix3(albedo, [0.72, 0.60, 0.40], evaporite.clamp(0.0, 0.16));
        }
        height_m -= sediment_w * 180.0 * relief;
        sediment_score = sediment_w * (0.52 + (1.0 - evaporite) * 0.16);
        evaporite_score = sediment_w * evaporite * 0.35;
    }

    let basin_mottle = dune_contact.paleo_lowland
        * (1.0 - dune_contact.dune_plate * 0.84)
        * smoothstep(
            0.16,
            0.62,
            ridge(fbm_dir(
                dir,
                root_seed.placement,
                "paleo_lowland_basin_mottle",
                3.2,
                4,
                0.54,
            )) * 0.78
                + fbm_dir(
                    dir,
                    root_seed.detail,
                    "paleo_lowland_basin_mottle_detail",
                    13.0,
                    3,
                    0.52,
                ) * 0.28
                + (1.0 - basin_sediment) * 0.26,
        );
    albedo = mix3(
        albedo,
        [0.33, 0.18, 0.11],
        (basin_mottle * 0.75).clamp(0.0, 0.72),
    );
    if basin_dune_fill > 0.01 {
        let dune_floor_variation = fbm_dir(
            dir,
            root_seed.detail,
            "paleo_lowland_dune_fill_tone",
            5.8,
            4,
            0.54,
        ) * 0.58
            + fbm_dir(
                dir,
                root_seed.detail,
                "paleo_lowland_dune_fill_grain",
                18.0,
                3,
                0.52,
            ) * 0.24;
        let dune_floor_color = mix3(
            [0.47, 0.19, 0.082],
            [0.66, 0.32, 0.145],
            smoothstep(-0.30, 0.58, dune_floor_variation),
        );
        albedo = mix3(
            albedo,
            dune_floor_color,
            (basin_dune_fill * 0.82).clamp(0.0, 0.92),
        );
    }

    let basin_margin = (sediment_w * (1.0 - sediment_w) * 3.2).clamp(0.0, 1.0);
    let channel_signal = ridge(fbm_dir(dir, root_seed.shape, "channels", 4.7, 4, 0.50)).powf(7.0)
        * (basin_margin * 0.82 + lowland_bias * 0.18)
        * smoothstep(-0.20, 0.68, macro_n + highland_ridges * 0.25)
        * projection.channel_strength;
    let channels = channel_signal.clamp(0.0, 1.0);
    if channels > 0.015 {
        height_m -= channels * 300.0 * relief;
        albedo = mix3(albedo, [0.28, 0.13, 0.085], channels.clamp(0.0, 0.32));
        dark_score = dark_score.max(channels * 0.28);
    }

    // Active, unconsolidated dune bodies are dynamic surface layers now. The
    // static cold-desert field keeps the basin substrate, margins, scarps, and
    // broad sand-sheet material tendency, but no longer bakes oriented dune
    // crests or migratory bodies into immutable terrain.

    let relief_slope_signal = (dune_contact.suture_crest * 0.86
        + dune_contact.mountain_web * 0.54
        + dune_contact.shoreline_scarp * 0.42
        + dune_contact.shelf_break * 0.18
        + channels * 0.34
        + shield.slope * 18.0 * shield.apron
        + shield.basal_scarp * 0.62
        + shield.caldera_rim * 0.45)
        .clamp(0.0, 1.0);
    albedo = cold_desert_relief_albedo_grade(albedo, height_m, relief_slope_signal);

    let dust_mottle =
        fbm_dir(dir, root_seed.detail, "dust_mottle", 12.0, 2, 0.55) * 0.024 * texture_visibility;
    let wind_polish =
        fbm_dir(dir, root_seed.detail, "wind_polish", 5.2, 2, 0.50) * 0.014 * texture_visibility;
    let dust_mottle = dust_mottle + wind_polish;
    albedo = [
        (albedo[0] + dust_mottle).clamp(0.03, 0.92),
        (albedo[1] + dust_mottle * 0.62).clamp(0.03, 0.92),
        (albedo[2] + dust_mottle * 0.38).clamp(0.03, 0.92),
    ];

    let relief_ridge_signal = (highland_ridges.powf(1.8) * 0.34
        + dune_contact.suture_crest * 0.52
        + dune_contact.mountain_web * 0.40
        + dune_contact.strandline * 0.16
        + shield.ridge * 0.46
        + shield.caldera_rim * 0.38)
        .clamp(0.0, 1.0);
    let relief_hollow_signal = (dune_contact.paleo_lowland * 0.24
        + basin_mottle * 0.28
        + channels * 0.62
        + shield.caldera_floor * 0.44
        + (1.0 - basin_sediment) * dune_contact.dune_plate * 0.12)
        .clamp(0.0, 1.0);
    let palette_variation = fbm_dir(dir, root_seed.detail, "relief_palette_broad", 1.75, 4, 0.55)
        * 0.62
        + fbm_dir(
            dir,
            root_seed.detail,
            "relief_palette_mottle",
            10.5,
            3,
            0.52,
        ) * 0.30
        + texture_n * 0.18
        + fine_n * 0.10;
    let palette_albedo = cold_desert_biome_relief_albedo(
        biomes,
        height_m,
        relief_slope_signal,
        relief_ridge_signal,
        relief_hollow_signal,
        palette_variation,
        style,
    );
    let palette_strength = (0.34
        + relief_slope_signal * 0.14
        + relief_ridge_signal * 0.08
        + relief_hollow_signal * 0.06
        + biomes.dark_volcanic_province * 0.10
        + biomes.pale_evaporite_basin * 0.08
        + biomes.dune_basin * 0.08
        + basin_dune_fill * 0.10)
        .clamp(0.30, 0.58);
    albedo = mix3(albedo, palette_albedo, palette_strength);
    let _surface_color_hint = cold_desert_rust_saturation_grade(albedo);

    // The renderer currently treats the dominant material id as the primary
    // palette lookup. Keep the dominant material conservative so broad
    // process masks do not become hard categorical paint regions; the
    // filterable albedo cube carries most orbital color variation.
    let rust_score = (0.26 + biomes.rust_dust_plain * 0.78
        - evaporite_score * 0.08
        - dune_score * 0.08
        - basin_dune_fill * 0.22)
        .max(0.08);
    let material_mix = SurfaceMaterialMix::from_weighted([
        (COLD_DESERT_MAT_RUST_DUST, rust_score),
        (
            COLD_DESERT_MAT_DARK_BASALT,
            biomes.dark_volcanic_province * 0.88
                + dark_score * 0.20
                + dune_contact.suture_crest * 1.12
                + dune_contact.mountain_web * 0.62
                + shield.apron * 1.12
                + shield.basal_scarp * 1.34
                + shield.dark_flows * 0.58,
        ),
        (
            COLD_DESERT_MAT_PALE_SEDIMENT,
            (biomes.pale_evaporite_basin * 0.74
                + sediment_score * 0.20
                + basin_sediment * (1.0 - basin_dune_fill) * 0.12)
                * (1.0 - basin_dune_fill * 0.95).clamp(0.0, 1.0),
        ),
        (
            COLD_DESERT_MAT_DUNE_SAND,
            basin_dune_fill * 1.74
                + dune_contact.dune_plate * 0.58
                + dune_score * 0.34
                + dune_contact.dune_toe * 0.10,
        ),
        (COLD_DESERT_MAT_EVAPORITE, evaporite_score * 0.55),
    ]);
    let roughness = 0.86 + dune_score * 0.04
        - evaporite_score * 0.12
        - dark_score * 0.08
        - shield.eroded_flanks * 0.025
        + shield.caldera_floor * 0.025;

    // Persist the full biome weights so the unified surface-color painter can
    // blend per-biome palettes after structural stages have finished. Several biomes get boosted on physical signals
    // (dune fill, shield flanks, sediment coherence) so the post-pass picks
    // up the right palette where features have actually landed.
    let biome_mix = BiomeMix::from_weighted([
        (
            ColdDesertBiome::RustDustPlain.index() as u8,
            biomes.rust_dust_plain,
        ),
        (
            ColdDesertBiome::DuneBasin.index() as u8,
            biomes.dune_basin + basin_dune_fill * 1.20 + dune_score * 0.40,
        ),
        (
            ColdDesertBiome::PaleEvaporiteBasin.index() as u8,
            biomes.pale_evaporite_basin + sediment_score * 0.30,
        ),
        (
            ColdDesertBiome::DarkVolcanicProvince.index() as u8,
            biomes.dark_volcanic_province + shield.apron * 0.60 + shield.basal_scarp * 0.80,
        ),
        (
            ColdDesertBiome::RuggedBadlands.index() as u8,
            biomes.rugged_badlands,
        ),
        (
            ColdDesertBiome::OxideHighland.index() as u8,
            biomes.oxide_highland,
        ),
        (
            ColdDesertBiome::PolarVeneer.index() as u8,
            biomes.polar_veneer,
        ),
        (ColdDesertBiome::AshMantle.index() as u8, biomes.ash_mantle),
    ]);

    SurfaceFieldSample::new(
        height_m,
        material_mix,
        biome_mix,
        roughness.clamp(0.55, 0.96),
        dir,
    )
}

fn sample_biome_height_generators(
    dir: Vec3,
    root_seed: FeatureSeed,
    biomes: ColdDesertBiomeWeights,
    generators: &crate::height_generator::ColdDesertBiomeHeightGenerators,
) -> f32 {
    let seed = root_seed.shape;
    // OxideHighland/PolarVeneer/AshMantle reuse existing height shapes —
    // they are color-driven biomes, so the per-biome contribution is
    // sampled from existing generators with a unique salt to keep the field
    // deterministic without inventing new amplitude/frequency profiles.
    biomes.rust_dust_plain
        * generators
            .rust_dust_plain
            .sample_height_m(dir, seed, "height:rust_dust_plain")
        + biomes.dune_basin
            * generators
                .dune_basin
                .sample_height_m(dir, seed, "height:dune_basin")
        + biomes.pale_evaporite_basin
            * generators.pale_evaporite_basin.sample_height_m(
                dir,
                seed,
                "height:pale_evaporite_basin",
            )
        + biomes.dark_volcanic_province
            * generators.dark_volcanic_province.sample_height_m(
                dir,
                seed,
                "height:dark_volcanic_province",
            )
        + biomes.rugged_badlands
            * generators
                .rugged_badlands
                .sample_height_m(dir, seed, "height:rugged_badlands")
        + biomes.oxide_highland
            * generators
                .rust_dust_plain
                .sample_height_m(dir, seed, "height:oxide_highland")
        + biomes.polar_veneer
            * generators
                .pale_evaporite_basin
                .sample_height_m(dir, seed, "height:polar_veneer")
        + biomes.ash_mantle
            * generators
                .dark_volcanic_province
                .sample_height_m(dir, seed, "height:ash_mantle")
}

fn cold_desert_rust_saturation_grade(albedo: [f32; 3]) -> [f32; 3] {
    [
        (albedo[0] * 1.045 + 0.008).clamp(0.025, 0.94),
        (albedo[1] * 0.915 + albedo[0] * 0.016).clamp(0.025, 0.94),
        (albedo[2] * 0.84 + albedo[1] * 0.018).clamp(0.020, 0.94),
    ]
}

fn cold_desert_relief_albedo_grade(albedo: [f32; 3], height_m: f32, slope_signal: f32) -> [f32; 3] {
    let low_floor = smoothstep(-350.0, -1_800.0, height_m);
    let high_dust = smoothstep(900.0, 5_600.0, height_m);
    let steep = smoothstep(0.22, 0.72, slope_signal);

    let mut color = albedo;
    color = mix3(color, [0.58, 0.34, 0.17], high_dust * 0.13);
    color = mix3(color, [0.66, 0.47, 0.27], low_floor * 0.07);
    color = mix3(color, [0.18, 0.11, 0.075], steep * 0.24);
    color = mix3(color, [0.48, 0.22, 0.11], high_dust * (1.0 - steep) * 0.08);
    color
}

/// Per-biome relief palettes for the default cold-desert preset, indexed by
/// `ColdDesertBiome::index()`.
/// The compile path copies this into `BodyBuilder::biome_palettes` so the
/// unified surface-color painter can blend palettes from the baked
/// biome-weights cubemap.
pub fn cold_desert_relief_palettes() -> Vec<ReliefPalette> {
    default_cold_desert_relief_palettes()
}

fn default_cold_desert_relief_palettes() -> Vec<ReliefPalette> {
    ColdDesertBiome::ALL
        .iter()
        .map(|biome| cold_desert_relief_palette_for_biome(*biome))
        .collect()
}

fn cold_desert_biome_relief_albedo(
    biomes: ColdDesertBiomeWeights,
    height_m: f32,
    slope_signal: f32,
    ridge_signal: f32,
    hollow_signal: f32,
    variation: f32,
    style: &ColdDesertStyle,
) -> [f32; 3] {
    let mut color = [0.0; 3];
    let mut total = 0.0;
    for biome in ColdDesertBiome::ALL {
        let weight = biomes.weight_for(biome).max(0.0);
        if weight <= 0.0 {
            continue;
        }

        let sample = style
            .relief_palettes
            .get(biome.index())
            .copied()
            .unwrap_or_else(|| cold_desert_relief_palette_for_biome(biome))
            .evaluate(
                height_m,
                slope_signal,
                ridge_signal,
                hollow_signal,
                variation,
            );
        color[0] += sample[0] * weight;
        color[1] += sample[1] * weight;
        color[2] += sample[2] * weight;
        total += weight;
    }

    if total <= 1.0e-5 {
        return style
            .relief_palettes
            .get(ColdDesertBiome::RustDustPlain.index())
            .copied()
            .unwrap_or_else(|| cold_desert_relief_palette_for_biome(ColdDesertBiome::RustDustPlain))
            .evaluate(
                height_m,
                slope_signal,
                ridge_signal,
                hollow_signal,
                variation,
            );
    }

    [color[0] / total, color[1] / total, color[2] / total]
}

fn cold_desert_relief_palette_for_biome(biome: ColdDesertBiome) -> ReliefPalette {
    match biome {
        ColdDesertBiome::RustDustPlain => ReliefPalette {
            low: [0.34, 0.135, 0.068],
            mid: [0.50, 0.210, 0.100],
            high: [0.67, 0.375, 0.195],
            steep: [0.18, 0.105, 0.074],
            ridge: [0.74, 0.50, 0.315],
            hollow: [0.245, 0.128, 0.082],
        },
        ColdDesertBiome::DuneBasin => ReliefPalette {
            low: [0.40, 0.160, 0.072],
            mid: [0.58, 0.265, 0.115],
            high: [0.74, 0.430, 0.210],
            steep: [0.255, 0.120, 0.072],
            ridge: [0.80, 0.535, 0.320],
            hollow: [0.33, 0.145, 0.074],
        },
        ColdDesertBiome::PaleEvaporiteBasin => ReliefPalette {
            low: [0.53, 0.37, 0.225],
            mid: [0.68, 0.53, 0.335],
            high: [0.84, 0.75, 0.55],
            steep: [0.32, 0.23, 0.16],
            ridge: [0.92, 0.84, 0.64],
            hollow: [0.45, 0.30, 0.20],
        },
        ColdDesertBiome::DarkVolcanicProvince => ReliefPalette {
            low: [0.052, 0.047, 0.043],
            mid: [0.105, 0.078, 0.060],
            high: [0.22, 0.155, 0.105],
            steep: [0.033, 0.031, 0.029],
            ridge: [0.36, 0.27, 0.18],
            hollow: [0.044, 0.036, 0.032],
        },
        ColdDesertBiome::RuggedBadlands => ReliefPalette {
            low: [0.235, 0.145, 0.105],
            mid: [0.40, 0.24, 0.15],
            high: [0.60, 0.42, 0.27],
            steep: [0.13, 0.088, 0.068],
            ridge: [0.74, 0.58, 0.39],
            hollow: [0.20, 0.12, 0.088],
        },
        // Strongly oxidized highlands — brighter, more saturated than the
        // rust dust plain. Reads as orange-red continents from orbit.
        ColdDesertBiome::OxideHighland => ReliefPalette {
            low: [0.45, 0.185, 0.080],
            mid: [0.65, 0.275, 0.100],
            high: [0.84, 0.420, 0.160],
            steep: [0.235, 0.105, 0.072],
            ridge: [0.92, 0.560, 0.250],
            hollow: [0.34, 0.145, 0.080],
        },
        // Pale frost+dust mantle. Warmer than evaporite (more dust, less
        // salt), so trends tan/cream rather than white.
        ColdDesertBiome::PolarVeneer => ReliefPalette {
            low: [0.62, 0.55, 0.46],
            mid: [0.78, 0.72, 0.62],
            high: [0.90, 0.86, 0.78],
            steep: [0.34, 0.28, 0.22],
            ridge: [0.94, 0.90, 0.82],
            hollow: [0.50, 0.43, 0.35],
        },
        // Gray ash, transition zone around the dark volcanic provinces.
        // Slightly warm so it reads as ash-on-rust, not lunar-grey.
        ColdDesertBiome::AshMantle => ReliefPalette {
            low: [0.18, 0.155, 0.135],
            mid: [0.30, 0.265, 0.235],
            high: [0.46, 0.40, 0.345],
            steep: [0.105, 0.092, 0.082],
            ridge: [0.56, 0.48, 0.40],
            hollow: [0.16, 0.135, 0.115],
        },
    }
}

fn shield_volcano_albedo(shield: ShieldVolcanoSample) -> [f32; 3] {
    let occlusion = (shield.erosion_delta + 0.5).clamp(0.0, 1.0);
    let cliff = smoothstep(0.018, 0.052, shield.slope) * shield.apron;
    let high = shield.dome.clamp(0.0, 1.0);

    let mut color = mix3([0.39, 0.145, 0.065], [0.56, 0.24, 0.095], high);
    color = mix3(color, [0.65, 0.36, 0.16], high * (1.0 - cliff) * 0.24);
    color = mix3(color, [0.17, 0.095, 0.062], cliff.clamp(0.0, 0.72));
    color = mix3(
        color,
        [0.24, 0.115, 0.070],
        (1.0 - occlusion) * shield.eroded_flanks * 0.72,
    );
    color = mix3(color, [0.67, 0.31, 0.105], shield.ridge * 0.24);
    color = mix3(color, [0.16, 0.090, 0.060], shield.crease * 0.34);
    color = mix3(
        color,
        [0.105, 0.070, 0.055],
        (shield.basal_scarp * 0.62 + shield.caldera_rim * 0.72).clamp(0.0, 0.86),
    );
    mix3(
        color,
        [0.20, 0.125, 0.085],
        (shield.caldera_floor * 0.58).clamp(0.0, 0.64),
    )
}

fn lowland_sediment_coherence(
    dir: Vec3,
    root_seed: FeatureSeed,
    paleo_lowland: f32,
    dune_plate: f32,
) -> f32 {
    let broad = fbm_dir(
        dir,
        root_seed.placement,
        "paleo_lowland_sediment_coherence",
        2.0,
        4,
        0.55,
    );
    let mottled = fbm_dir(
        dir,
        root_seed.detail,
        "paleo_lowland_sediment_mottle",
        7.4,
        3,
        0.52,
    );
    let lace = ridge(fbm_dir(
        dir,
        root_seed.detail,
        "paleo_lowland_sediment_lace",
        17.0,
        3,
        0.51,
    ));
    smoothstep(
        0.26,
        0.82,
        paleo_lowland * 0.68 + broad * 0.30 + mottled * 0.10 + lace * 0.08 - dune_plate * 0.16,
    )
}

fn cold_desert_biome_mask_plan(
    projection: &ColdDesertProjectionConfig,
    style: &ColdDesertStyle,
) -> BiomeMaskPlan<COLD_DESERT_BIOME_COUNT> {
    use BiomeMaskExpr as E;
    use BiomeMaskSeedStream::{Detail, Placement, Shape};

    let dune_score = E::product(vec![
        E::clamp(
            0.0,
            1.0,
            E::sum(vec![
                (1.0, E::signal("dune_plate")),
                (0.95, E::signal("paleo_lowland")),
            ]),
        ),
        E::constant(projection.dune_strength),
    ]);
    let dark_lat_warp = E::sum(vec![
        (1.0, E::signal("dir_y")),
        (0.06, E::fbm(Placement, "biome_dark_lat_warp", 1.6, 3, 0.52)),
    ]);
    let equatorial_memory = E::sum(vec![
        (0.16, E::constant(1.0)),
        (-0.16, E::smoothstep(0.10, 0.42, E::abs(dark_lat_warp))),
    ]);
    let mut dark_terms: Vec<(f32, E)> = style
        .dark_province_anchors
        .iter()
        .map(|anchor| {
            (
                anchor.weight,
                E::cap(anchor.center_dir, anchor.radius_rad, anchor.feather_rad),
            )
        })
        .collect();
    dark_terms.push((1.0, equatorial_memory));
    let dark_base = E::clamp(0.0, 1.0, E::sum(dark_terms));
    let dark_continuity = E::smoothstep(
        -0.28,
        0.54,
        E::sum(vec![
            (
                1.0,
                E::fbm(Placement, "biome_dark_plain_continuity", 1.25, 4, 0.55),
            ),
            (
                0.16,
                E::fbm(Detail, "biome_dark_plain_texture", 4.2, 3, 0.50),
            ),
        ]),
    );
    let dark_score = E::clamp(
        0.0,
        1.0,
        E::product(vec![
            dark_base,
            dark_continuity,
            E::constant(projection.volcanic_dark_strength),
        ]),
    );

    let pale_cap = E::sum(
        style
            .pale_basin_anchors
            .iter()
            .map(|anchor| {
                (
                    anchor.biome_weight,
                    E::cap(
                        anchor.center_dir,
                        anchor.biome_radius_rad,
                        anchor.biome_feather_rad,
                    ),
                )
            })
            .collect(),
    );
    let pale_score = E::product(vec![
        E::smoothstep(
            0.16,
            0.78,
            E::sum(vec![
                (1.0, pale_cap),
                (0.34, E::signal("lowland_bias")),
                (0.22, E::signal("paleo_lowland")),
                (0.10, E::fbm(Placement, "biome_pale_lowlands", 1.4, 4, 0.56)),
                (-0.16, E::signal("dark_score")),
                (-0.82, E::signal("dune_score")),
            ]),
        ),
        E::constant(projection.pale_basin_strength),
    ]);

    let rugged_seed = E::smoothstep(
        0.42,
        0.94,
        E::sum(vec![
            (1.0, E::signal("highland_ridges")),
            (0.18, E::signal("regional")),
            (0.12, E::signal("macro")),
            (0.25, E::signal("dark_score")),
            (-0.72, E::signal("dune_plate")),
            (-0.24, E::signal("paleo_lowland")),
            (-0.22, E::signal("pale_score")),
        ]),
    );
    let rugged_texture = E::smoothstep(
        -0.10,
        0.62,
        E::fbm(Shape, "biome_rugged_breaks", 6.2, 4, 0.52),
    );
    let rugged_score = E::product(vec![
        rugged_seed,
        E::sum(vec![(0.58, E::constant(1.0)), (0.42, rugged_texture)]),
    ]);

    // Polar veneer: high latitude band, lobed by fbm so the boundary is
    // fractal rather than a circular cap. `dir_y` is the body-frame
    // y-component, i.e. sin(latitude) in the bake frame. Threshold is loose
    // enough that polar zones extend visibly inward from the poles, with
    // the fbm warp breaking the edge so it never reads as a clean band.
    let polar_score = E::product(vec![
        E::smoothstep(
            0.20,
            0.74,
            E::sum(vec![
                (1.0, E::abs(E::signal("dir_y"))),
                (0.18, E::fbm(Placement, "biome_polar_warp", 1.4, 3, 0.55)),
                (0.10, E::fbm(Detail, "biome_polar_lace", 4.5, 3, 0.50)),
                (-0.30, E::signal("dune_score")),
                (-0.20, E::signal("dark_score")),
            ]),
        ),
        E::sum(vec![
            (0.86, E::constant(1.0)),
            (0.34, E::fbm(Shape, "biome_polar_continuity", 1.8, 3, 0.55)),
        ]),
    ]);

    // Oxide highland: bright orange highlands picking up where the macro
    // signal is strongly positive AND highland_ridges ridge-bias is firing.
    // Distinct enough from rust dust plain that the planet shows red/orange
    // vs. reddish-brown bands at orbital scale. Threshold lowered so oxide
    // territory is meaningful at orbital read.
    let oxide_seed = E::smoothstep(
        -0.22,
        0.62,
        E::sum(vec![
            (1.0, E::signal("highland_ridges")),
            (0.40, E::signal("macro")),
            (0.24, E::signal("regional")),
            (
                0.26,
                E::fbm(Placement, "biome_oxide_provinces", 1.10, 4, 0.55),
            ),
            (-0.62, E::signal("paleo_lowland")),
            (-0.34, E::signal("dune_score")),
            (-0.22, E::signal("pale_score")),
            (-0.18, E::signal("dark_score")),
            (-0.40, E::signal("polar_score")),
        ]),
    );
    let oxide_texture = E::smoothstep(
        -0.10,
        0.55,
        E::fbm(Detail, "biome_oxide_mottle", 5.6, 3, 0.52),
    );
    let oxide_score = E::product(vec![
        oxide_seed,
        E::sum(vec![(0.92, E::constant(1.0)), (0.38, oxide_texture)]),
    ]);

    // Ash mantle: dust-fall transition zone around the dark volcanic
    // provinces. Picks up where dark_score is non-zero but not dominant.
    // Reads as gray fringe between rust plain and basalt. Lower threshold
    // so the ash zone extends well beyond the basalt cores.
    let ash_score = E::product(vec![
        E::smoothstep(
            0.02,
            0.38,
            E::sum(vec![
                (1.0, E::signal("dark_score")),
                (
                    0.42,
                    E::fbm(Placement, "biome_ash_continuity", 1.45, 4, 0.54),
                ),
                (-0.22, E::signal("dune_score")),
                (-0.10, E::signal("pale_score")),
            ]),
        ),
        E::sum(vec![
            (0.62, E::constant(1.0)),
            (0.42, E::fbm(Detail, "biome_ash_mottle", 4.2, 3, 0.50)),
        ]),
    ]);

    // Rust dust plain becomes a thinner default — base weight lowered and
    // new biome scores subtract from it. The fallback (last position) only
    // fires when every other score is near zero.
    let rust_score = E::clamp(
        0.06,
        f32::INFINITY,
        E::sum(vec![
            (0.20, E::constant(1.0)),
            (
                0.24,
                E::smoothstep(
                    -0.22,
                    0.68,
                    E::sum(vec![
                        (1.0, E::signal("macro")),
                        (0.22, E::signal("highland_ridges")),
                    ]),
                ),
            ),
            (-0.28, E::signal("dune_score")),
            (-0.10, E::signal("paleo_lowland")),
            (-0.16, E::signal("pale_score")),
            (-0.08, E::signal("dark_score")),
            (-0.34, E::signal("oxide_score")),
            (-0.30, E::signal("polar_score")),
            (-0.26, E::signal("ash_score")),
        ]),
    );

    BiomeMaskPlan::new(
        vec![
            BiomeMaskRule::new(
                ColdDesertBiome::DuneBasin.index(),
                Some("dune_score"),
                dune_score,
            ),
            BiomeMaskRule::new(
                ColdDesertBiome::DuneBasin.index(),
                None,
                E::product(vec![E::constant(0.45), E::signal("dune_score")]),
            ),
            BiomeMaskRule::new(
                ColdDesertBiome::DarkVolcanicProvince.index(),
                Some("dark_score"),
                dark_score,
            ),
            BiomeMaskRule::new(
                ColdDesertBiome::DarkVolcanicProvince.index(),
                None,
                E::product(vec![E::constant(0.08), E::signal("dark_score")]),
            ),
            BiomeMaskRule::new(
                ColdDesertBiome::PaleEvaporiteBasin.index(),
                Some("pale_score"),
                pale_score,
            ),
            BiomeMaskRule::new(
                ColdDesertBiome::PolarVeneer.index(),
                Some("polar_score"),
                polar_score,
            ),
            BiomeMaskRule::new(
                ColdDesertBiome::OxideHighland.index(),
                Some("oxide_score"),
                oxide_score,
            ),
            BiomeMaskRule::new(
                ColdDesertBiome::AshMantle.index(),
                Some("ash_score"),
                ash_score,
            ),
            BiomeMaskRule::new(ColdDesertBiome::RuggedBadlands.index(), None, rugged_score),
            BiomeMaskRule::new(ColdDesertBiome::RustDustPlain.index(), None, rust_score),
        ],
        ColdDesertBiome::RustDustPlain.index(),
    )
}

pub fn sample_cold_desert_biomes(
    dir: Vec3,
    root_seed: FeatureSeed,
    biome_plan: &BiomeMaskPlan<COLD_DESERT_BIOME_COUNT>,
) -> ColdDesertBiomeWeights {
    let macro_n = fbm_dir(dir, root_seed.shape, "macro", 1.15, 5, 0.55);
    let regional_n = fbm_dir(dir, root_seed.shape, "regional", 2.7, 4, 0.55);
    let highland_ridges = ridge(fbm_dir(
        dir,
        root_seed.shape,
        "highland_ridge",
        5.4,
        4,
        0.52,
    ));
    let lowland_bias = smoothstep(0.72, -0.20, macro_n + regional_n * 0.35);

    cold_desert_biome_weights(
        dir,
        root_seed,
        macro_n,
        regional_n,
        highland_ridges,
        lowland_bias,
        biome_plan,
    )
}

fn cold_desert_biome_weights(
    dir: Vec3,
    root_seed: FeatureSeed,
    macro_n: f32,
    regional_n: f32,
    highland_ridges: f32,
    lowland_bias: f32,
    biome_plan: &BiomeMaskPlan<COLD_DESERT_BIOME_COUNT>,
) -> ColdDesertBiomeWeights {
    let dune_contact = dune_basin_contact(dir, root_seed, lowland_bias, highland_ridges);
    let seeds = BiomeMaskSeeds {
        identity: root_seed.identity,
        placement: root_seed.placement,
        shape: root_seed.shape,
        detail: root_seed.detail,
        children: root_seed.children,
    };
    let mut context = BiomeMaskContext::new(dir, seeds)
        .with_signal("macro", macro_n)
        .with_signal("regional", regional_n)
        .with_signal("highland_ridges", highland_ridges)
        .with_signal("lowland_bias", lowland_bias)
        .with_signal("paleo_lowland", dune_contact.paleo_lowland)
        .with_signal("dune_plate", dune_contact.dune_plate);

    ColdDesertBiomeWeights::from_mask(biome_plan.sample(&mut context))
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ShieldVolcanoSample {
    apron: f32,
    dome: f32,
    basal_scarp: f32,
    caldera_floor: f32,
    caldera_rim: f32,
    eroded_flanks: f32,
    crease: f32,
    ridge: f32,
    slope: f32,
    erosion_delta: f32,
    dark_flows: f32,
    height_m: f32,
}

fn shield_volcano_frame(shield: ColdDesertShieldVolcano) -> (Vec3, Vec3, Vec3) {
    let center = shield.center_dir.normalize();
    let east = Vec3::Y.cross(center).normalize();
    let north = center.cross(east).normalize();
    (center, east, north)
}

fn shield_volcano_sample(
    dir: Vec3,
    root_seed: FeatureSeed,
    sample_scale_m: f32,
    style: &ColdDesertStyle,
) -> ShieldVolcanoSample {
    let Some(shield) = style.shield_volcano else {
        return ShieldVolcanoSample::default();
    };
    let (center, east, north) = shield_volcano_frame(shield);
    let z = dir.dot(center).clamp(-0.999_999, 0.999_999);
    let local_x = dir.dot(east).atan2(z);
    let local_y = dir.dot(north).atan2(z);
    let body_radius_m = shield.radius_m / shield.radius_rad.max(1.0e-5);
    let x_m = local_x * body_radius_m;
    let y_m = local_y * body_radius_m;
    let rx = shield.radius_m * 1.06;
    let ry = shield.radius_m * 0.94;
    let r = shield_volcano_profile_radius(x_m, y_m, rx, ry, root_seed);

    if r > 1.22 {
        return ShieldVolcanoSample::default();
    }

    let apron = smoothstep(1.18, 0.90, r);
    let r_clamped = r.clamp(0.0, 1.0);
    let dome_base = (1.0 - r_clamped.powf(1.72)).max(0.0).powf(1.18);
    let dome = dome_base * smoothstep(1.03, 0.12, r);
    let basal_scarp = band_mask(r, 1.0, 0.048).powf(0.70) * smoothstep(1.16, 0.84, r);
    let outer_moat = band_mask(r, 1.105, 0.095) * smoothstep(1.23, 0.94, r);

    let caldera_floor = smoothstep(0.140, 0.070, r);
    let caldera_rim = band_mask(r, 0.142, 0.026).powf(0.62);

    let (dome_height, dh_dx, dh_dy) =
        shield_dome_height_and_slope(x_m, y_m, rx, ry, shield.height_m, root_seed);
    let flank_band = smoothstep(0.18, 0.36, r) * smoothstep(1.12, 0.54, r);
    let erosion_visibility = scale_visibility(sample_scale_m, 42_000.0);
    let erosion = if flank_band > 0.0 && erosion_visibility > 0.0 {
        let p = Vec2::new(x_m, y_m) + shield_erosion_offset(root_seed);
        let base = Vec3::new(dome_height, dh_dx, dh_dy);
        let params = shield_erosion_params();
        erosion_filter(p, base, (dome_height / 3_800.0).clamp(-1.0, 1.0), &params)
    } else {
        bevy_erosion_filter::cpu::ErosionFilterResult {
            delta: Vec3::ZERO,
            magnitude: 0.0,
            ridge_map: 0.0,
            debug: 0.0,
        }
    };
    let erosion_height_m = erosion.delta.x * flank_band * erosion_visibility * 0.62;
    let erosion_delta = if erosion.magnitude > 1.0e-5 {
        (erosion.delta.x / erosion.magnitude).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let ridge_unit = (erosion.ridge_map * 0.5 + 0.5).clamp(0.0, 1.0);
    let crease = ((1.0 - (ridge_unit / 0.30).clamp(0.0, 1.0)) * 1.5).clamp(0.0, 1.0)
        * flank_band
        * erosion_visibility;
    let ridge = smoothstep(0.58, 0.95, ridge_unit) * flank_band * erosion_visibility;
    let eroded_flanks = (erosion.ridge_map.abs() * 0.42 + erosion.delta.x.abs() * 0.0012)
        .clamp(0.0, 1.0)
        * flank_band
        * erosion_visibility;
    let dark_flows =
        (eroded_flanks * 0.48 + basal_scarp * 0.34 + caldera_rim * 0.56 + caldera_floor * 0.22)
            .clamp(0.0, 1.0);

    let height_m = dome_height + basal_scarp * 760.0 - outer_moat * 210.0 - caldera_floor * 1_380.0
        + caldera_rim * 760.0
        + erosion_height_m;
    let slope = Vec2::new(dh_dx, dh_dy).length();

    ShieldVolcanoSample {
        apron,
        dome,
        basal_scarp,
        caldera_floor,
        caldera_rim,
        eroded_flanks,
        crease,
        ridge,
        slope,
        erosion_delta,
        dark_flows,
        height_m,
    }
}

fn shield_dome_height_and_slope(
    x_m: f32,
    y_m: f32,
    rx: f32,
    ry: f32,
    height_m: f32,
    root_seed: FeatureSeed,
) -> (f32, f32, f32) {
    let raw_r = ((x_m / rx).powi(2) + (y_m / ry).powi(2)).sqrt();
    let r = shield_volcano_profile_radius(x_m, y_m, rx, ry, root_seed);
    if !(0.0..1.0).contains(&r) {
        return (0.0, 0.0, 0.0);
    }

    let p = 1.72;
    let q = 1.18;
    let inner = (1.0 - r.powf(p)).max(0.0);
    let profile = inner.powf(q);
    let height = profile * height_m;
    if r < 1.0e-5 || inner <= 1.0e-5 {
        return (height, 0.0, 0.0);
    }

    let dprofile_dr = -q * p * r.powf(p - 1.0) * inner.powf(q - 1.0);
    let dh_dr = dprofile_dr * height_m;
    let edge_scale = shield_volcano_boundary_scale(x_m, y_m, rx, ry, root_seed);
    let edge_weight = smoothstep(0.32, 0.92, raw_r);
    let profile_scale = 1.0 + (edge_scale - 1.0) * edge_weight;
    let raw_r = raw_r.max(1.0e-5);
    let dr_dx = x_m / (rx * rx * raw_r * profile_scale);
    let dr_dy = y_m / (ry * ry * raw_r * profile_scale);
    (height, dh_dr * dr_dx, dh_dr * dr_dy)
}

fn shield_volcano_profile_radius(
    x_m: f32,
    y_m: f32,
    rx: f32,
    ry: f32,
    root_seed: FeatureSeed,
) -> f32 {
    let raw_r = ((x_m / rx).powi(2) + (y_m / ry).powi(2)).sqrt();
    let edge_scale = shield_volcano_boundary_scale(x_m, y_m, rx, ry, root_seed);
    let edge_weight = smoothstep(0.32, 0.92, raw_r);
    raw_r / (1.0 + (edge_scale - 1.0) * edge_weight).max(0.55)
}

fn shield_volcano_boundary_scale(
    x_m: f32,
    y_m: f32,
    rx: f32,
    ry: f32,
    root_seed: FeatureSeed,
) -> f32 {
    let ux = x_m / rx;
    let uy = y_m / ry;
    let theta = uy.atan2(ux);
    let (sin_t, cos_t) = theta.sin_cos();
    let lobe_seed = seed32(root_seed.placement, "shield_volcano_boundary_lobes");
    let phase_a = seed_phase(lobe_seed, 0);
    let phase_b = seed_phase(lobe_seed, 8);
    let phase_c = seed_phase(lobe_seed, 16);
    let phase_d = seed_phase(lobe_seed, 24);
    let low_lobes = (theta * 2.0 + phase_a).sin() * 0.145
        + (theta * 3.0 + phase_b).sin() * 0.120
        + (theta * 5.0 + phase_c).sin() * 0.085
        + (theta * 8.0 + phase_d).sin() * 0.052;
    let directed_lobes = angular_lobe(theta, phase_a + 0.70, 0.52) * 0.20
        + angular_lobe(theta, phase_b + 1.25, 0.38) * 0.14
        - angular_lobe(theta, phase_c + 0.45, 0.46) * 0.17
        - angular_lobe(theta, phase_d + 2.10, 0.34) * 0.12;
    let broad = fbm3(
        cos_t * 1.45,
        sin_t * 1.45,
        0.37,
        seed32(root_seed.placement, "shield_volcano_boundary_broad"),
        4,
        0.56,
        2.03,
    ) * 0.160;
    let scallop = fbm3(
        ux * 5.8 + cos_t * 0.85,
        uy * 5.8 + sin_t * 0.85,
        1.19,
        seed32(root_seed.detail, "shield_volcano_boundary_scallop"),
        3,
        0.52,
        2.08,
    ) * 0.105;

    (1.0 + low_lobes + directed_lobes + broad + scallop).clamp(0.58, 1.58)
}

fn angular_lobe(theta: f32, center: f32, half_width: f32) -> f32 {
    let delta = (theta - center + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;
    smoothstep(half_width, 0.0, delta.abs())
}

fn seed_phase(seed: u32, shift: u32) -> f32 {
    let bits = (seed >> shift) & 0xff;
    bits as f32 / 255.0 * std::f32::consts::TAU
}

fn shield_erosion_params() -> ErosionFilterParams {
    let defaults = ErosionFilterParams::default();
    ErosionFilterParams {
        scale: 42_000.0,
        strength: 0.014,
        gully_weight: 0.34,
        octaves: 4,
        onset: defaults.onset * 0.22,
        assumed_slope: Vec2::new(0.45, 0.95),
        ..defaults
    }
}

fn shield_erosion_offset(root_seed: FeatureSeed) -> Vec2 {
    let seed = seed32(root_seed.detail, "shield_volcano_erosion_offset");
    let sx = (seed & 0xffff) as f32;
    let sy = (seed >> 16) as f32;
    Vec2::new(sx * 17.0, sy * 17.0)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DuneBasinContact {
    signed: f32,
    paleo_lowland: f32,
    dune_plate: f32,
    highland_plate: f32,
    suture_crest: f32,
    mountain_web: f32,
    dune_toe: f32,
    shoreline_scarp: f32,
    shelf_break: f32,
    strandline: f32,
    interior_highs: f32,
    floor_undulation: f32,
}

fn dune_basin_frame() -> (Vec3, Vec3, Vec3) {
    let center = Vec3::new(0.0, -0.08, 1.0).normalize();
    let east = Vec3::Y.cross(center).normalize();
    let north = center.cross(east).normalize();
    (center, east, north)
}

fn basin_offset_dir(x: f32, y: f32) -> Vec3 {
    let (center, east, north) = dune_basin_frame();
    let z = (1.0 - x * x - y * y).max(0.0).sqrt();
    (center * z + east * x + north * y).normalize()
}

fn project_axis_onto_tangent(axis: Vec3, normal: Vec3, fallback: Vec3) -> Vec3 {
    let projected = axis - normal * axis.dot(normal);
    if projected.length_squared() > 1.0e-8 {
        projected.normalize()
    } else {
        fallback.normalize()
    }
}

fn basin_geodesic_ellipse_signed(dir: Vec3, cx: f32, cy: f32, rx: f32, ry: f32) -> f32 {
    let (_, basin_east, basin_north) = dune_basin_frame();
    let center = basin_offset_dir(cx, cy);
    let east = project_axis_onto_tangent(basin_east, center, basin_north.cross(center));
    let north = center.cross(east).normalize();
    let z = dir.dot(center).clamp(-0.999_999, 0.999_999);
    let x = dir.dot(east).atan2(z);
    let y = dir.dot(north).atan2(z);
    1.0 - ((x / rx).powi(2) + (y / ry).powi(2)).sqrt()
}

fn dune_basin_shape_dir(
    dir: Vec3,
    root_seed: FeatureSeed,
    side_bias: f32,
    polar_bias: f32,
) -> Vec3 {
    let (_, basin_east, basin_north) = dune_basin_frame();
    let broad_x = fbm_dir(
        dir,
        root_seed.placement,
        "paleo_ocean_geodesic_warp_x",
        1.35,
        4,
        0.55,
    );
    let broad_y = fbm_dir(
        dir,
        root_seed.placement,
        "paleo_ocean_geodesic_warp_y",
        1.55,
        4,
        0.54,
    );
    let margin_x = fbm_dir(
        dir,
        root_seed.detail,
        "paleo_ocean_margin_vector_x",
        5.0,
        3,
        0.52,
    );
    let margin_y = fbm_dir(
        dir,
        root_seed.detail,
        "paleo_ocean_margin_vector_y",
        5.8,
        3,
        0.52,
    );
    let amount = 0.040 + side_bias * 0.090 + polar_bias * 0.070;
    let tangent = basin_east * (broad_x * 0.72 + margin_x * 0.28)
        + basin_north * (broad_y * 0.72 + margin_y * 0.28);
    (dir + tangent * amount).normalize()
}

fn central_dune_plate_field(dir: Vec3, root_seed: FeatureSeed) -> f32 {
    let (basin_center, east, north) = dune_basin_frame();
    let raw_x = dir.dot(east);
    let raw_y = dir.dot(north);
    let side_bias = smoothstep(0.44, 0.86, raw_x.abs());
    let polar_bias = smoothstep(0.34, 0.64, raw_y.abs());
    let shape_dir = dune_basin_shape_dir(dir, root_seed, side_bias, polar_bias);

    let local_x = shape_dir.dot(east);
    let local_y = shape_dir.dot(north);
    let local_z = shape_dir.dot(basin_center);
    let core = basin_geodesic_ellipse_signed(shape_dir, 0.10, -0.18, 0.34, 0.22);
    let west_spill = basin_geodesic_ellipse_signed(shape_dir, -0.18, -0.22, 0.28, 0.15);
    let east_spill = basin_geodesic_ellipse_signed(shape_dir, 0.33, -0.16, 0.34, 0.18);
    let south_spill = basin_geodesic_ellipse_signed(shape_dir, 0.07, -0.36, 0.34, 0.13);
    let northwest_spill = basin_geodesic_ellipse_signed(shape_dir, -0.09, 0.02, 0.24, 0.12);
    let basin_sheet = local_z - 0.24
        + fbm_dir(
            dir,
            root_seed.placement,
            "active_dune_plate_hemisphere_sheet",
            0.95,
            4,
            0.56,
        ) * 0.220
        + fbm_dir(
            dir,
            root_seed.detail,
            "active_dune_plate_hemisphere_lace",
            4.8,
            3,
            0.52,
        ) * 0.060;
    let north_bite = basin_geodesic_ellipse_signed(shape_dir, 0.02, 0.17, 0.36, 0.14);
    let west_bite = basin_geodesic_ellipse_signed(shape_dir, -0.36, -0.12, 0.16, 0.18);
    let east_bite = basin_geodesic_ellipse_signed(shape_dir, 0.52, -0.25, 0.14, 0.17);
    let south_bite = basin_geodesic_ellipse_signed(shape_dir, -0.14, -0.47, 0.20, 0.14);
    let southeast_bite = basin_geodesic_ellipse_signed(shape_dir, 0.30, -0.43, 0.16, 0.13);

    let mut field = soft_max(core, basin_sheet, 0.16);
    field = soft_max(field, west_spill, 0.08);
    field = soft_max(field, east_spill, 0.07);
    field = soft_max(field, south_spill, 0.06);
    field = soft_max(field, northwest_spill, 0.05);
    field -= smoothstep(-0.10, 0.24, north_bite) * 0.21;
    field -= smoothstep(-0.08, 0.22, west_bite) * 0.13;
    field -= smoothstep(-0.08, 0.22, east_bite) * 0.13;
    field -= smoothstep(-0.08, 0.22, south_bite) * 0.15;
    field -= smoothstep(-0.08, 0.20, southeast_bite) * 0.13;
    field
        + fbm_dir(
            dir,
            root_seed.placement,
            "active_dune_plate_macro_break",
            2.3,
            4,
            0.54,
        ) * 0.240
        + fbm_dir(
            dir,
            root_seed.detail,
            "active_dune_plate_edge_chop",
            14.0,
            3,
            0.52,
        ) * 0.070
        + (local_x * 16.0
            + local_y * 5.5
            + fbm_dir(
                dir,
                root_seed.placement,
                "active_dune_plate_wind_scallop",
                1.6,
                3,
                0.52,
            ) * 3.0)
            .sin()
            * 0.040
}

fn paleo_lowland_plate_coherence(dir: Vec3, root_seed: FeatureSeed, paleo_lowland: f32) -> f32 {
    let broad = fbm_dir(
        dir,
        root_seed.placement,
        "paleo_lowland_plate_coherence",
        2.4,
        4,
        0.55,
    );
    let mottled = fbm_dir(
        dir,
        root_seed.detail,
        "paleo_lowland_plate_mottle",
        8.0,
        3,
        0.52,
    );
    let lace = ridge(fbm_dir(
        dir,
        root_seed.detail,
        "paleo_lowland_plate_lace",
        20.0,
        3,
        0.51,
    ));
    smoothstep(
        0.22,
        0.82,
        paleo_lowland * 0.72 + broad * 0.30 + mottled * 0.10 + lace * 0.08,
    )
}

fn dune_basin_field(
    dir: Vec3,
    root_seed: FeatureSeed,
    lowland_bias: f32,
    highland_ridges: f32,
) -> f32 {
    let (basin_center, east, north) = dune_basin_frame();
    let raw_x = dir.dot(east);
    let raw_y = dir.dot(north);
    let side_bias = smoothstep(0.44, 0.86, raw_x.abs());
    let polar_bias = smoothstep(0.34, 0.64, raw_y.abs());
    let shape_dir = dune_basin_shape_dir(dir, root_seed, side_bias, polar_bias);
    let x = shape_dir.dot(east);
    let y = shape_dir.dot(north);
    let z = shape_dir.dot(basin_center);

    let x_warp = fbm_dir(
        dir,
        root_seed.placement,
        "paleo_ocean_margin_x_warp",
        2.2,
        4,
        0.55,
    ) * 0.070
        + fbm_dir(
            dir,
            root_seed.detail,
            "paleo_ocean_margin_x_chop",
            9.0,
            3,
            0.52,
        ) * 0.026
        + (y * 9.0
            + fbm_dir(
                dir,
                root_seed.placement,
                "paleo_ocean_margin_phase",
                1.4,
                2,
                0.50,
            ) * 2.5)
            .sin()
            * 0.035;
    let y_warp = fbm_dir(
        dir,
        root_seed.placement,
        "paleo_ocean_margin_y_warp",
        1.8,
        3,
        0.54,
    ) * 0.045
        + (x * 7.5).sin() * 0.018;
    let x_shape = x + x_warp * (0.55 + side_bias * 0.95);
    let y_shape = y + y_warp * (0.65 + side_bias * 0.45);

    let main_floor = basin_geodesic_ellipse_signed(shape_dir, 0.02, -0.05, 0.78, 0.58);
    let west_gulf = basin_geodesic_ellipse_signed(shape_dir, -0.40, 0.02, 0.37, 0.27);
    let northwest_bay = basin_geodesic_ellipse_signed(shape_dir, -0.58, 0.28, 0.29, 0.19);
    let southwest_bay = basin_geodesic_ellipse_signed(shape_dir, -0.55, -0.31, 0.35, 0.23);
    let south_bight = basin_geodesic_ellipse_signed(shape_dir, -0.02, -0.48, 0.62, 0.31);
    let northeast_sea = basin_geodesic_ellipse_signed(shape_dir, 0.35, 0.25, 0.40, 0.27);
    let east_embayment = basin_geodesic_ellipse_signed(shape_dir, 0.52, -0.09, 0.30, 0.22);
    let southeast_bay = basin_geodesic_ellipse_signed(shape_dir, 0.44, -0.40, 0.32, 0.20);
    let hemisphere_breakup = fbm_dir(
        dir,
        root_seed.placement,
        "paleo_ocean_hemisphere_floor",
        0.92,
        4,
        0.56,
    ) * 0.430
        + fbm_dir(
            dir,
            root_seed.detail,
            "paleo_ocean_hemisphere_floor_lace",
            3.8,
            3,
            0.52,
        ) * 0.120;
    let hemisphere_floor = z - 0.22 + hemisphere_breakup;
    let mut paleo_ocean = soft_max(main_floor, west_gulf, 0.13);
    paleo_ocean = soft_max(paleo_ocean, northwest_bay, 0.08);
    paleo_ocean = soft_max(paleo_ocean, southwest_bay, 0.09);
    paleo_ocean = soft_max(paleo_ocean, south_bight, 0.11);
    paleo_ocean = soft_max(paleo_ocean, northeast_sea, 0.10);
    paleo_ocean = soft_max(paleo_ocean, east_embayment, 0.08);
    paleo_ocean = soft_max(paleo_ocean, southeast_bay, 0.07);
    paleo_ocean = soft_max(paleo_ocean, hemisphere_floor, 0.18);

    let axial_suture = (1.0 - ((y_shape + x_shape * 0.34 + 0.03).abs() / 0.18)).clamp(-1.0, 1.0);
    let axial_extent = smoothstep(0.82, 0.35, x_shape.abs());
    paleo_ocean = soft_max(paleo_ocean, axial_suture * axial_extent * 0.34 - 0.12, 0.07);

    let northern_promontory = ellipse_signed(x_shape, y_shape, -0.06, 0.40, 0.34, 0.19);
    let eastern_horst = ellipse_signed(x_shape, y_shape, 0.31, 0.02, 0.18, 0.28);
    let inner_island = ellipse_signed(x_shape, y_shape, -0.18, -0.10, 0.16, 0.13);
    let shelf_bite = ellipse_signed(x_shape, y_shape, 0.10, -0.30, 0.22, 0.11);
    let west_wall_bite = basin_geodesic_ellipse_signed(shape_dir, -0.70, -0.02, 0.15, 0.34);
    let east_wall_bite = basin_geodesic_ellipse_signed(shape_dir, 0.64, 0.04, 0.16, 0.31);
    let west_north_wall_bite = basin_geodesic_ellipse_signed(shape_dir, -0.72, 0.31, 0.18, 0.17);
    let west_south_wall_bite = basin_geodesic_ellipse_signed(shape_dir, -0.70, -0.36, 0.18, 0.18);
    let east_north_wall_bite = basin_geodesic_ellipse_signed(shape_dir, 0.70, 0.31, 0.17, 0.17);
    let east_south_wall_bite = basin_geodesic_ellipse_signed(shape_dir, 0.68, -0.31, 0.17, 0.19);
    let north_cleft = basin_geodesic_ellipse_signed(shape_dir, -0.34, 0.58, 0.22, 0.13);
    let north_terrace = basin_geodesic_ellipse_signed(shape_dir, 0.18, 0.54, 0.26, 0.12);
    let northeast_notch = basin_geodesic_ellipse_signed(shape_dir, 0.56, 0.36, 0.18, 0.16);
    let north_fjord = basin_geodesic_ellipse_signed(shape_dir, -0.12, 0.49, 0.18, 0.20);
    let north_reentrant = basin_geodesic_ellipse_signed(shape_dir, 0.36, 0.45, 0.18, 0.19);
    let south_scour = basin_geodesic_ellipse_signed(shape_dir, -0.30, -0.67, 0.28, 0.15);
    let south_gate = basin_geodesic_ellipse_signed(shape_dir, 0.16, -0.70, 0.24, 0.13);
    let southeast_notch = basin_geodesic_ellipse_signed(shape_dir, 0.50, -0.56, 0.20, 0.16);
    let southwest_fjord = basin_geodesic_ellipse_signed(shape_dir, -0.50, -0.52, 0.20, 0.18);
    let south_peninsula = basin_geodesic_ellipse_signed(shape_dir, -0.10, -0.55, 0.24, 0.21);
    let south_island_chain = basin_geodesic_ellipse_signed(shape_dir, 0.30, -0.55, 0.20, 0.18);
    let south_arc_west = basin_geodesic_ellipse_signed(shape_dir, -0.36, -0.46, 0.17, 0.22);
    let south_arc_mid = basin_geodesic_ellipse_signed(shape_dir, 0.04, -0.52, 0.20, 0.20);
    let south_arc_east = basin_geodesic_ellipse_signed(shape_dir, 0.42, -0.46, 0.16, 0.21);
    paleo_ocean -= smoothstep(-0.16, 0.34, northern_promontory) * 0.22;
    paleo_ocean -= smoothstep(-0.13, 0.28, eastern_horst) * 0.15;
    paleo_ocean -= smoothstep(-0.08, 0.22, inner_island) * 0.16;
    paleo_ocean -= smoothstep(-0.12, 0.24, shelf_bite) * 0.08;
    paleo_ocean -= smoothstep(-0.10, 0.24, west_wall_bite) * 0.20;
    paleo_ocean -= smoothstep(-0.10, 0.24, east_wall_bite) * 0.18;
    paleo_ocean -= smoothstep(-0.09, 0.25, west_north_wall_bite) * 0.18;
    paleo_ocean -= smoothstep(-0.09, 0.25, west_south_wall_bite) * 0.20;
    paleo_ocean -= smoothstep(-0.09, 0.25, east_north_wall_bite) * 0.17;
    paleo_ocean -= smoothstep(-0.09, 0.25, east_south_wall_bite) * 0.19;
    paleo_ocean -= smoothstep(-0.10, 0.26, north_cleft) * 0.16;
    paleo_ocean -= smoothstep(-0.10, 0.25, north_terrace) * 0.12;
    paleo_ocean -= smoothstep(-0.08, 0.24, northeast_notch) * 0.12;
    paleo_ocean -= smoothstep(-0.10, 0.25, north_fjord) * 0.15;
    paleo_ocean -= smoothstep(-0.10, 0.25, north_reentrant) * 0.13;
    paleo_ocean -= smoothstep(-0.10, 0.28, south_scour) * 0.22;
    paleo_ocean -= smoothstep(-0.09, 0.26, south_gate) * 0.20;
    paleo_ocean -= smoothstep(-0.08, 0.25, southeast_notch) * 0.17;
    paleo_ocean -= smoothstep(-0.09, 0.26, southwest_fjord) * 0.17;
    paleo_ocean -= smoothstep(-0.10, 0.28, south_peninsula) * 0.20;
    paleo_ocean -= smoothstep(-0.08, 0.24, south_island_chain) * 0.16;
    paleo_ocean -= smoothstep(-0.09, 0.26, south_arc_west) * 0.17;
    paleo_ocean -= smoothstep(-0.09, 0.28, south_arc_mid) * 0.22;
    paleo_ocean -= smoothstep(-0.08, 0.25, south_arc_east) * 0.16;

    // Continent mask. The basin's near/far split was previously a smoothstep
    // on raw `z` with low-amplitude warp — that produced a sharp meridian
    // contour because every iso-z is a circle of latitude in the basin frame.
    // Here we let multi-octave 3D fbm dominate the smoothstep argument so the
    // boundary is a topology-free fractal coastline. `z` still biases the
    // basin toward the near hemisphere; the noise breaks the iso-contour into
    // bays, peninsulas, and outlier highland patches.
    let continent_warp = fbm_dir(
        dir,
        root_seed.placement,
        "paleo_ocean_continent_mask",
        1.15,
        5,
        0.55,
    ) * 0.85
        + fbm_dir(
            dir,
            root_seed.detail,
            "paleo_ocean_continent_lace",
            4.5,
            3,
            0.52,
        ) * 0.20;
    let far_side_penalty = smoothstep(-0.05, -0.95, z + continent_warp)
        * (1.52
            + ridge(fbm_dir(
                dir,
                root_seed.detail,
                "paleo_ocean_far_side_penalty_teeth",
                18.0,
                3,
                0.52,
            )) * 0.22);
    let basin_skew = x_shape * -0.035 + y_shape * 0.020 - far_side_penalty;
    let geologic_asymmetry = fbm_dir(
        dir,
        root_seed.placement,
        "biome_dune_basin_warp",
        1.7,
        3,
        0.54,
    ) * 0.060;
    let broken_edge = fbm_dir(
        dir,
        root_seed.placement,
        "biome_dune_basin_edge",
        7.5,
        4,
        0.54,
    ) * 0.095;
    let chipped_edge = fbm_dir(
        dir,
        root_seed.placement,
        "biome_dune_basin_chipped_edge",
        22.0,
        3,
        0.52,
    ) * 0.034;
    let sawtooth_edge = ridge(fbm_dir(
        dir,
        root_seed.detail,
        "biome_dune_basin_sawtooth_edge",
        48.0,
        2,
        0.50,
    )) * 0.018;
    let side_macro_break = side_bias
        * fbm_dir(
            dir,
            root_seed.placement,
            "paleo_ocean_sidewall_macro_break",
            2.8,
            4,
            0.55,
        )
        * 0.310;
    let side_wall_bend = side_bias
        * ((y_shape * 9.5
            + fbm_dir(
                dir,
                root_seed.placement,
                "paleo_ocean_sidewall_bend_phase",
                1.25,
                3,
                0.52,
            ) * 4.0)
            .sin()
            * 0.115
            + fbm_dir(
                dir,
                root_seed.placement,
                "paleo_ocean_sidewall_bend_macro",
                1.6,
                4,
                0.55,
            ) * 0.160);
    let polar_fray = smoothstep(0.34, 0.66, y_shape.abs())
        * (fbm_dir(
            dir,
            root_seed.detail,
            "paleo_ocean_polar_margin_fray",
            6.4,
            4,
            0.53,
        ) * 0.115
            + ridge(fbm_dir(
                dir,
                root_seed.detail,
                "paleo_ocean_polar_margin_teeth",
                26.0,
                3,
                0.51,
            )) * 0.036);
    let side_fray = side_bias
        * (fbm_dir(
            dir,
            root_seed.detail,
            "paleo_ocean_side_margin_fray",
            5.7,
            4,
            0.53,
        ) * 0.055
            + ridge(fbm_dir(
                dir,
                root_seed.detail,
                "paleo_ocean_side_margin_teeth",
                30.0,
                3,
                0.51,
            )) * 0.020);

    paleo_ocean
        + basin_skew
        + geologic_asymmetry
        + broken_edge
        + chipped_edge
        + sawtooth_edge
        + side_macro_break
        + side_wall_bend
        + polar_fray
        + side_fray
        + lowland_bias * 0.040
        - highland_ridges * 0.050
        + 0.050
}

fn dune_basin_contact(
    dir: Vec3,
    root_seed: FeatureSeed,
    lowland_bias: f32,
    highland_ridges: f32,
) -> DuneBasinContact {
    let signed = dune_basin_field(dir, root_seed, lowland_bias, highland_ridges);
    let (_, east, north) = dune_basin_frame();
    let x = dir.dot(east);
    let y = dir.dot(north);

    let paleo_lowland_base = smoothstep(0.000, 0.022, signed);
    let paleo_lowland = paleo_lowland_base
        * (0.12 + paleo_lowland_plate_coherence(dir, root_seed, paleo_lowland_base) * 0.88);
    let highland_plate = 1.0 - smoothstep(-0.018, -0.004, signed);
    let suture_band = band_mask(signed, -0.012, 0.030);
    let highland_fringe = highland_plate * band_mask(signed, -0.070, 0.145);

    let dune_plate =
        paleo_lowland * smoothstep(-0.18, 0.08, central_dune_plate_field(dir, root_seed));
    let dune_toe_band = paleo_lowland * band_mask(signed, 0.075, 0.065);

    let ridge_web = ridge(fbm_dir(
        dir,
        root_seed.shape,
        "dune_margin_ridge_web",
        18.0,
        5,
        0.57,
    ))
    .powf(2.0);
    let branch_web = ridge(fbm_dir(
        dir,
        root_seed.shape,
        "dune_margin_branch_web",
        46.0,
        4,
        0.55,
    ))
    .powf(2.55);
    let edge_breaks = smoothstep(
        -0.10,
        0.70,
        fbm_dir(
            dir,
            root_seed.detail,
            "dune_margin_erosion_breaks",
            7.8,
            4,
            0.53,
        ),
    );
    let scarp_chips = smoothstep(
        0.12,
        0.86,
        ridge(fbm_dir(
            dir,
            root_seed.detail,
            "dune_margin_scarp_chips",
            74.0,
            3,
            0.52,
        )),
    );
    let spur_web = ridge(fbm_dir(
        dir,
        root_seed.detail,
        "dune_margin_peak_spurs",
        92.0,
        2,
        0.50,
    ))
    .powf(3.15);

    let suture_crest = (suture_band.powf(0.55)
        * (0.72 + ridge_web * 0.26 + scarp_chips * 0.22 + spur_web * 0.18))
        .clamp(0.0, 1.0);
    let mountain_web = (highland_fringe
        * (ridge_web * 0.84 + branch_web * 0.64 + spur_web * 0.38)
        * (0.42 + edge_breaks * 0.58)
        * (1.0 - dune_plate * 0.85))
        .clamp(0.0, 1.0);
    let dune_toe =
        (dune_toe_band.powf(1.20) * (0.34 + ridge_web * 0.42 + scarp_chips * 0.18)).clamp(0.0, 1.0);

    let north_east_margin =
        (smoothstep(0.04, 0.42, y) * 0.74 + smoothstep(0.20, 0.68, x) * 0.56).clamp(0.0, 1.0);
    let shoreline_scarp = (suture_band.powf(0.72)
        * (0.24 + north_east_margin * 0.82 + scarp_chips * 0.22 + edge_breaks * 0.16)
        * (1.0 - paleo_lowland * 0.12))
        .clamp(0.0, 1.0);

    let shelf_noise = fbm_dir(dir, root_seed.shape, "paleo_ocean_shelf_warp", 5.1, 3, 0.54) * 0.025;
    let shelf_break = (paleo_lowland
        * (band_mask(signed + shelf_noise, 0.095, 0.030) * 0.76
            + band_mask(signed + shelf_noise * 0.60, 0.205, 0.045) * 0.48))
        .clamp(0.0, 1.0);

    let strand_noise = fbm_dir(
        dir,
        root_seed.detail,
        "paleo_ocean_strandline_warp",
        9.5,
        3,
        0.52,
    ) * 0.018;
    let strandline = (paleo_lowland
        * (band_mask(signed + strand_noise, 0.065, 0.010) * 0.62
            + band_mask(signed + strand_noise * 0.80, 0.135, 0.013) * 0.50
            + band_mask(signed + strand_noise * 0.55, 0.245, 0.016) * 0.36))
        .clamp(0.0, 1.0);

    let horst_a = smoothstep(-0.09, 0.26, ellipse_signed(x, y, -0.28, 0.02, 0.16, 0.09));
    let horst_b = smoothstep(-0.10, 0.24, ellipse_signed(x, y, 0.38, -0.03, 0.18, 0.12));
    let horst_c = smoothstep(-0.10, 0.22, ellipse_signed(x, y, -0.05, 0.30, 0.24, 0.08));
    let interior_highs =
        (paleo_lowland * (horst_a * 0.86 + horst_b * 0.68 + horst_c * 0.46)).clamp(0.0, 1.0);

    let floor_undulation = paleo_lowland
        * (fbm_dir(
            dir,
            root_seed.shape,
            "paleo_ocean_floor_swells",
            2.1,
            4,
            0.55,
        ) * 0.66
            + fbm_dir(
                dir,
                root_seed.detail,
                "paleo_ocean_floor_wrinkles",
                7.2,
                3,
                0.52,
            ) * 0.34)
            .clamp(-1.0, 1.0);

    DuneBasinContact {
        signed,
        paleo_lowland,
        dune_plate,
        highland_plate,
        suture_crest,
        mountain_web,
        dune_toe,
        shoreline_scarp,
        shelf_break,
        strandline,
        interior_highs,
        floor_undulation,
    }
}

fn fbm_dir(
    dir: Vec3,
    seed: u64,
    stream: &str,
    frequency: f32,
    octaves: u32,
    persistence: f32,
) -> f32 {
    let seed = seed32(seed, stream);
    fbm3(
        dir.x * frequency,
        dir.y * frequency,
        dir.z * frequency,
        seed,
        octaves,
        persistence,
        2.02,
    )
}

fn seed32(seed: u64, stream: &str) -> u32 {
    sub_seed(seed, stream) as u32
}

fn ridge(v: f32) -> f32 {
    1.0 - v.abs().clamp(0.0, 1.0)
}

fn band_mask(v: f32, center: f32, half_width: f32) -> f32 {
    (1.0 - ((v - center).abs() / half_width.max(1.0e-5))).clamp(0.0, 1.0)
}

fn cap_mask(dir: Vec3, center: Vec3, inner_rad: f32, outer_rad: f32) -> f32 {
    smoothstep(outer_rad.cos(), inner_rad.cos(), dir.dot(center))
}

fn ellipse_signed(x: f32, y: f32, cx: f32, cy: f32, rx: f32, ry: f32) -> f32 {
    1.0 - (((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2)).sqrt()
}

fn soft_max(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k.max(1.0e-5)).clamp(0.0, 1.0);
    a * (1.0 - h) + b * h + k * h * (1.0 - h)
}
