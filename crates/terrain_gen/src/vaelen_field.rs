//! Vaelen cold-desert continuous surface field.

use glam::Vec3;

use crate::biome_mask::{
    BiomeMaskContext, BiomeMaskExpr, BiomeMaskPlan, BiomeMaskRule, BiomeMaskSeedStream,
    BiomeMaskSeeds, BiomeMaskWeights,
};
use crate::feature_compiler::{ColdDesertProjectionConfig, FeatureSeed};
use crate::noise::fbm3;
use crate::seeding::sub_seed;
use crate::surface_field::{
    mix3, scale_visibility, smoothstep, SurfaceField, SurfaceFieldSample, SurfaceMaterialMix,
};
use crate::types::Material;

pub const VAELEN_MAT_RUST_DUST: u8 = 0;
pub const VAELEN_MAT_DARK_BASALT: u8 = 1;
pub const VAELEN_MAT_PALE_SEDIMENT: u8 = 2;
pub const VAELEN_MAT_DUNE_SAND: u8 = 3;
pub const VAELEN_MAT_EVAPORITE: u8 = 4;
pub const VAELEN_BIOME_COUNT: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaelenBiome {
    RustDustPlain,
    DuneBasin,
    PaleEvaporiteBasin,
    DarkVolcanicProvince,
    RuggedBadlands,
}

impl VaelenBiome {
    pub const ALL: [Self; 5] = [
        Self::RustDustPlain,
        Self::DuneBasin,
        Self::PaleEvaporiteBasin,
        Self::DarkVolcanicProvince,
        Self::RuggedBadlands,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::RustDustPlain => "rust_dust_plain",
            Self::DuneBasin => "dune_basin",
            Self::PaleEvaporiteBasin => "pale_evaporite_basin",
            Self::DarkVolcanicProvince => "dark_volcanic_province",
            Self::RuggedBadlands => "rugged_badlands",
        }
    }

    pub fn color_srgb(self) -> [u8; 3] {
        match self {
            Self::RustDustPlain => [176, 91, 48],
            Self::DuneBasin => [216, 142, 52],
            Self::PaleEvaporiteBasin => [221, 198, 143],
            Self::DarkVolcanicProvince => [62, 58, 55],
            Self::RuggedBadlands => [124, 83, 66],
        }
    }

    fn index(self) -> usize {
        match self {
            Self::RustDustPlain => 0,
            Self::DuneBasin => 1,
            Self::PaleEvaporiteBasin => 2,
            Self::DarkVolcanicProvince => 3,
            Self::RuggedBadlands => 4,
        }
    }

    fn from_index(index: usize) -> Self {
        match index {
            1 => Self::DuneBasin,
            2 => Self::PaleEvaporiteBasin,
            3 => Self::DarkVolcanicProvince,
            4 => Self::RuggedBadlands,
            _ => Self::RustDustPlain,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VaelenBiomeWeights {
    pub rust_dust_plain: f32,
    pub dune_basin: f32,
    pub pale_evaporite_basin: f32,
    pub dark_volcanic_province: f32,
    pub rugged_badlands: f32,
}

impl VaelenBiomeWeights {
    fn from_mask(weights: BiomeMaskWeights<VAELEN_BIOME_COUNT>) -> Self {
        Self {
            rust_dust_plain: weights.weights[VaelenBiome::RustDustPlain.index()],
            dune_basin: weights.weights[VaelenBiome::DuneBasin.index()],
            pale_evaporite_basin: weights.weights[VaelenBiome::PaleEvaporiteBasin.index()],
            dark_volcanic_province: weights.weights[VaelenBiome::DarkVolcanicProvince.index()],
            rugged_badlands: weights.weights[VaelenBiome::RuggedBadlands.index()],
        }
    }

    pub fn dominant(self) -> VaelenBiome {
        let weights = [
            self.rust_dust_plain,
            self.dune_basin,
            self.pale_evaporite_basin,
            self.dark_volcanic_province,
            self.rugged_badlands,
        ];
        let index = BiomeMaskWeights { weights }.dominant_index();
        VaelenBiome::from_index(index)
    }

    pub fn weight_for(self, biome: VaelenBiome) -> f32 {
        match biome {
            VaelenBiome::RustDustPlain => self.rust_dust_plain,
            VaelenBiome::DuneBasin => self.dune_basin,
            VaelenBiome::PaleEvaporiteBasin => self.pale_evaporite_basin,
            VaelenBiome::DarkVolcanicProvince => self.dark_volcanic_province,
            VaelenBiome::RuggedBadlands => self.rugged_badlands,
        }
    }

    pub fn debug_color_srgb(self) -> [u8; 3] {
        let mut rgb = [0.0; 3];
        for biome in VaelenBiome::ALL {
            let color = biome.color_srgb();
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
pub struct VaelenColdDesertField {
    root_seed: FeatureSeed,
    projection: ColdDesertProjectionConfig,
    biome_plan: BiomeMaskPlan<VAELEN_BIOME_COUNT>,
}

impl VaelenColdDesertField {
    pub fn new(root_seed: FeatureSeed, projection: ColdDesertProjectionConfig) -> Self {
        let biome_plan = cold_desert_biome_mask_plan(&projection);
        Self {
            root_seed,
            projection,
            biome_plan,
        }
    }

    pub fn sample_biomes(&self, dir: Vec3) -> VaelenBiomeWeights {
        sample_vaelen_biomes(dir, self.root_seed, &self.biome_plan)
    }

    pub fn sample_suture_debug(&self, dir: Vec3) -> VaelenSutureDebug {
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

        VaelenSutureDebug {
            paleo_lowland: contact.paleo_lowland,
            dune_plate: contact.dune_plate,
            highland_plate: contact.highland_plate,
            suture_crest: contact.suture_crest,
            mountain_web: contact.mountain_web,
            dune_toe: contact.dune_toe,
        }
    }

    pub fn material_palette() -> Vec<Material> {
        vec![
            Material {
                albedo: [0.47, 0.18, 0.095],
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
                albedo: [0.70, 0.33, 0.105],
                roughness: 0.92,
            },
            Material {
                albedo: [0.76, 0.64, 0.44],
                roughness: 0.68,
            },
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VaelenSutureDebug {
    pub paleo_lowland: f32,
    pub dune_plate: f32,
    pub highland_plate: f32,
    pub suture_crest: f32,
    pub mountain_web: f32,
    pub dune_toe: f32,
}

impl VaelenSutureDebug {
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

impl SurfaceField for VaelenColdDesertField {
    fn sample(&self, dir: Vec3, sample_scale_m: f32) -> SurfaceFieldSample {
        sample_vaelen_cold_desert(
            dir,
            sample_scale_m,
            self.root_seed,
            &self.projection,
            &self.biome_plan,
        )
    }
}

fn sample_vaelen_cold_desert(
    dir: Vec3,
    sample_scale_m: f32,
    root_seed: FeatureSeed,
    projection: &ColdDesertProjectionConfig,
    biome_plan: &BiomeMaskPlan<VAELEN_BIOME_COUNT>,
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
    let biomes = vaelen_biome_weights(
        dir,
        root_seed,
        macro_n,
        regional_n,
        highland_ridges,
        lowland_bias,
        biome_plan,
    );
    let dune_contact = dune_basin_contact(dir, root_seed, lowland_bias, highland_ridges);
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
    let mut dune_score = 0.0;

    let dark_belt = biomes.dark_volcanic_province * projection.volcanic_dark_strength;
    if dark_belt > 0.05 {
        dark_score = dark_belt.clamp(0.0, 1.0);
        let dark_color = mix3(
            [0.19, 0.095, 0.065],
            [0.12, 0.060, 0.048],
            smoothstep(0.50, 0.95, dark_belt),
        );
        albedo = mix3(albedo, dark_color, dark_belt.clamp(0.0, 0.54));
        height_m -= dark_belt * 260.0 * relief;
    }

    let pale_center_a = Vec3::new(-0.34, -0.22, 0.91).normalize();
    let pale_center_b = Vec3::new(0.72, 0.18, -0.67).normalize();
    let pale_center_c = Vec3::new(-0.88, 0.20, 0.38).normalize();
    let pale_cap = cap_mask(dir, pale_center_a, 0.26, 0.70) * 0.42
        + cap_mask(dir, pale_center_b, 0.18, 0.56) * 0.34
        + cap_mask(dir, pale_center_c, 0.14, 0.44) * 0.20;
    let pale_coherence = smoothstep(
        0.18,
        0.92,
        fbm_dir(dir, root_seed.placement, "pale_basin", 1.25, 4, 0.55)
            + pale_cap
            + lowland_bias * 0.24
            + basin_sediment * 0.14
            + biomes.pale_evaporite_basin * 0.58
            - dark_belt * 0.18,
    );
    let sediment_w = (pale_coherence * 0.56 + biomes.pale_evaporite_basin * 0.62).clamp(0.0, 1.0)
        * projection.pale_basin_strength
        * 0.78;
    if sediment_w > 0.04 {
        let evap_a = cap_mask(dir, pale_center_a, 0.10, 0.36);
        let evap_b = cap_mask(dir, pale_center_b, 0.08, 0.28);
        let evaporite = (smoothstep(0.62, 0.96, sediment_w + ridge(texture_n) * 0.05) * 0.35
            + evap_a * 0.18
            + evap_b * 0.14)
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

    let dune_center = Vec3::new(0.18, 0.36, 0.92).normalize();
    let active_dune_noise = smoothstep(
        0.12,
        0.62,
        fbm_dir(dir, root_seed.children, "dune_mask", 2.8, 4, 0.54) + 0.30,
    );
    let dune_coherence = smoothstep(
        0.20,
        0.78,
        dune_contact.dune_plate * 0.48
            + active_dune_noise * 0.46
            + fbm_dir(dir, root_seed.detail, "dune_mask_edge_lace", 9.0, 3, 0.52) * 0.16,
    );
    let dune_mask = dune_contact.dune_plate
        * dune_coherence
        * smoothstep(
            0.02,
            0.42,
            1.0 - biomes.rugged_badlands * 0.82 - biomes.dark_volcanic_province * 0.24,
        )
        * projection.dune_strength;
    if dune_mask > 0.015 {
        let east = Vec3::Y.cross(dune_center).normalize();
        let north = dune_center.cross(east).normalize();
        let wind_x = dir.dot(east);
        let wind_y = dir.dot(north);
        let dune_visibility = scale_visibility(sample_scale_m, 18_000.0);
        let phase = wind_x * 92.0
            + wind_y * 21.0
            + fbm_dir(dir, root_seed.detail, "dune_warp", 8.0, 3, 0.5) * 5.0;
        let wave = ((phase.sin() * 0.5 + 0.5).powf(2.8)) * dune_visibility;
        height_m += dune_mask * (wave * 70.0 + texture_n * 24.0) * relief;
        albedo = mix3(
            albedo,
            mix3([0.54, 0.24, 0.085], [0.70, 0.34, 0.12], wave),
            dune_mask.clamp(0.0, 0.38),
        );
        dune_score = dune_mask * 0.65;
    }

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
    albedo = vaelen_rust_saturation_grade(albedo);

    // The renderer currently treats the dominant material id as the primary
    // palette lookup. Keep Vaelen's dominant material conservative so broad
    // process masks do not become hard categorical paint regions; the
    // filterable albedo cube carries most orbital color variation.
    let rust_score =
        (0.26 + biomes.rust_dust_plain * 0.78 - evaporite_score * 0.08 - dune_score * 0.08)
            .max(0.08);
    let material_mix = SurfaceMaterialMix::from_weighted([
        (VAELEN_MAT_RUST_DUST, rust_score),
        (
            VAELEN_MAT_DARK_BASALT,
            biomes.dark_volcanic_province * 0.88
                + dark_score * 0.20
                + dune_contact.suture_crest * 1.12
                + dune_contact.mountain_web * 0.62,
        ),
        (
            VAELEN_MAT_PALE_SEDIMENT,
            biomes.pale_evaporite_basin * 0.74
                + sediment_score * 0.20
                + basin_sediment * (1.0 - dune_contact.dune_plate) * 0.12,
        ),
        (
            VAELEN_MAT_DUNE_SAND,
            dune_contact.dune_plate * 1.62 + dune_score * 0.34 + dune_contact.dune_toe * 0.10,
        ),
        (VAELEN_MAT_EVAPORITE, evaporite_score * 0.55),
    ]);
    let roughness = 0.86 + dune_score * 0.04 - evaporite_score * 0.12 - dark_score * 0.08;

    SurfaceFieldSample::new(
        height_m,
        albedo,
        material_mix,
        roughness.clamp(0.55, 0.96),
        dir,
    )
}

fn sample_biome_height_generators(
    dir: Vec3,
    root_seed: FeatureSeed,
    biomes: VaelenBiomeWeights,
    generators: &crate::height_generator::ColdDesertBiomeHeightGenerators,
) -> f32 {
    let seed = root_seed.shape;
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
}

fn vaelen_rust_saturation_grade(albedo: [f32; 3]) -> [f32; 3] {
    [
        (albedo[0] * 1.16 + 0.018).clamp(0.03, 0.92),
        (albedo[1] * 0.70 + albedo[0] * 0.020).clamp(0.03, 0.92),
        (albedo[2] * 0.48 + albedo[1] * 0.012).clamp(0.03, 0.92),
    ]
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
) -> BiomeMaskPlan<VAELEN_BIOME_COUNT> {
    use BiomeMaskExpr as E;
    use BiomeMaskSeedStream::{Detail, Placement, Shape};

    let dune_score = E::product(vec![
        E::signal("dune_plate"),
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
    let dark_base = E::clamp(
        0.0,
        1.0,
        E::sum(vec![
            (0.92, E::cap(Vec3::new(-0.62, -0.08, -0.78), 0.42, 1.02)),
            (0.62, E::cap(Vec3::new(0.50, 0.12, 0.86), 0.28, 0.76)),
            (1.0, equatorial_memory),
        ]),
    );
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

    let pale_cap = E::sum(vec![
        (0.52, E::cap(Vec3::new(-0.34, -0.22, 0.91), 0.34, 0.92)),
        (0.34, E::cap(Vec3::new(0.72, 0.18, -0.67), 0.22, 0.64)),
        (0.22, E::cap(Vec3::new(-0.88, 0.20, 0.38), 0.20, 0.54)),
    ]);
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
                (-0.10, E::signal("dune_score")),
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

    let rust_score = E::clamp(
        0.10,
        f32::INFINITY,
        E::sum(vec![
            (0.36, E::constant(1.0)),
            (
                0.34,
                E::smoothstep(
                    -0.22,
                    0.68,
                    E::sum(vec![
                        (1.0, E::signal("macro")),
                        (0.22, E::signal("highland_ridges")),
                    ]),
                ),
            ),
            (-0.18, E::signal("dune_score")),
            (-0.10, E::signal("paleo_lowland")),
            (-0.16, E::signal("pale_score")),
            (-0.08, E::signal("dark_score")),
        ]),
    );

    BiomeMaskPlan::new(
        vec![
            BiomeMaskRule::new(
                VaelenBiome::DuneBasin.index(),
                Some("dune_score"),
                dune_score,
            ),
            BiomeMaskRule::new(
                VaelenBiome::DuneBasin.index(),
                None,
                E::product(vec![E::constant(0.45), E::signal("dune_score")]),
            ),
            BiomeMaskRule::new(
                VaelenBiome::DarkVolcanicProvince.index(),
                Some("dark_score"),
                dark_score,
            ),
            BiomeMaskRule::new(
                VaelenBiome::DarkVolcanicProvince.index(),
                None,
                E::product(vec![E::constant(0.08), E::signal("dark_score")]),
            ),
            BiomeMaskRule::new(
                VaelenBiome::PaleEvaporiteBasin.index(),
                Some("pale_score"),
                pale_score,
            ),
            BiomeMaskRule::new(VaelenBiome::RuggedBadlands.index(), None, rugged_score),
            BiomeMaskRule::new(VaelenBiome::RustDustPlain.index(), None, rust_score),
        ],
        VaelenBiome::RustDustPlain.index(),
    )
}

pub fn sample_vaelen_biomes(
    dir: Vec3,
    root_seed: FeatureSeed,
    biome_plan: &BiomeMaskPlan<VAELEN_BIOME_COUNT>,
) -> VaelenBiomeWeights {
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

    vaelen_biome_weights(
        dir,
        root_seed,
        macro_n,
        regional_n,
        highland_ridges,
        lowland_bias,
        biome_plan,
    )
}

fn vaelen_biome_weights(
    dir: Vec3,
    root_seed: FeatureSeed,
    macro_n: f32,
    regional_n: f32,
    highland_ridges: f32,
    lowland_bias: f32,
    biome_plan: &BiomeMaskPlan<VAELEN_BIOME_COUNT>,
) -> VaelenBiomeWeights {
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

    VaelenBiomeWeights::from_mask(biome_plan.sample(&mut context))
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
    let (_, east, north) = dune_basin_frame();
    let raw_x = dir.dot(east);
    let raw_y = dir.dot(north);
    let side_bias = smoothstep(0.44, 0.86, raw_x.abs());
    let polar_bias = smoothstep(0.34, 0.64, raw_y.abs());
    let shape_dir = dune_basin_shape_dir(dir, root_seed, side_bias, polar_bias);

    let local_x = shape_dir.dot(east);
    let local_y = shape_dir.dot(north);
    let core = basin_geodesic_ellipse_signed(shape_dir, 0.10, -0.18, 0.34, 0.22);
    let west_spill = basin_geodesic_ellipse_signed(shape_dir, -0.18, -0.22, 0.28, 0.15);
    let east_spill = basin_geodesic_ellipse_signed(shape_dir, 0.33, -0.16, 0.34, 0.18);
    let south_spill = basin_geodesic_ellipse_signed(shape_dir, 0.07, -0.36, 0.34, 0.13);
    let northwest_spill = basin_geodesic_ellipse_signed(shape_dir, -0.09, 0.02, 0.24, 0.12);
    let north_bite = basin_geodesic_ellipse_signed(shape_dir, 0.02, 0.17, 0.36, 0.14);
    let west_bite = basin_geodesic_ellipse_signed(shape_dir, -0.36, -0.12, 0.16, 0.18);
    let east_bite = basin_geodesic_ellipse_signed(shape_dir, 0.52, -0.25, 0.14, 0.17);
    let south_bite = basin_geodesic_ellipse_signed(shape_dir, -0.14, -0.47, 0.20, 0.14);
    let southeast_bite = basin_geodesic_ellipse_signed(shape_dir, 0.30, -0.43, 0.16, 0.13);

    let mut field = soft_max(core, west_spill, 0.08);
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
    let mut paleo_ocean = soft_max(main_floor, west_gulf, 0.13);
    paleo_ocean = soft_max(paleo_ocean, northwest_bay, 0.08);
    paleo_ocean = soft_max(paleo_ocean, southwest_bay, 0.09);
    paleo_ocean = soft_max(paleo_ocean, south_bight, 0.11);
    paleo_ocean = soft_max(paleo_ocean, northeast_sea, 0.10);
    paleo_ocean = soft_max(paleo_ocean, east_embayment, 0.08);
    paleo_ocean = soft_max(paleo_ocean, southeast_bay, 0.07);

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

    let far_side_warp = fbm_dir(
        dir,
        root_seed.placement,
        "paleo_ocean_far_side_warp",
        2.4,
        4,
        0.54,
    ) * 0.16
        + fbm_dir(
            dir,
            root_seed.detail,
            "paleo_ocean_far_side_chop",
            9.5,
            3,
            0.52,
        ) * 0.045;
    let far_side_penalty = smoothstep(-0.08, -0.56, z + far_side_warp)
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
        * 0.230;
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
            * 0.070
            + fbm_dir(
                dir,
                root_seed.placement,
                "paleo_ocean_sidewall_bend_macro",
                1.6,
                4,
                0.55,
            ) * 0.105);
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
        + 0.105
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_field() -> VaelenColdDesertField {
        let id = crate::FeatureId::from("vaelen.crustal_provinces");
        VaelenColdDesertField::new(
            FeatureSeed::derive(1007, &id),
            ColdDesertProjectionConfig::default(),
        )
    }

    fn dir_from_basin_xy(x: f32, y: f32) -> Vec3 {
        let (center, east, north) = dune_basin_frame();
        let z = (1.0 - x * x - y * y).max(0.0).sqrt();
        (center * z + east * x + north * y).normalize()
    }

    fn equirect_dir(x: u32, y: u32, width: u32, height: u32) -> Vec3 {
        let lat = (0.5 - (y as f32 + 0.5) / height as f32) * std::f32::consts::PI;
        let (sl, cl) = lat.sin_cos();
        let lon = ((x as f32 + 0.5) / width as f32 - 0.5) * std::f32::consts::TAU;
        let (sln, cln) = lon.sin_cos();
        Vec3::new(cl * sln, sl, cl * cln)
    }

    fn contact_at_xy(field: &VaelenColdDesertField, x: f32, y: f32) -> DuneBasinContact {
        let dir = dir_from_basin_xy(x, y);
        let root_seed = field.root_seed;
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
        dune_basin_contact(dir, root_seed, lowland_bias, highland_ridges)
    }

    #[test]
    fn vaelen_field_samples_sane_orbital_values() {
        let field = test_field();
        let sample = field.sample(Vec3::new(0.3, 0.4, 0.86).normalize(), 1_000.0);

        assert!(sample.height_m.is_finite());
        assert!(sample.height_m.abs() < 10_000.0);
        assert!(sample.albedo_linear.iter().all(|channel| *channel > 0.0));
        assert!(sample.roughness >= 0.55);
    }

    #[test]
    fn paleo_ocean_coverage_stays_planet_scale_and_dunes_are_localized() {
        use crate::cubemap::{face_uv_to_dir, CubemapFace};

        let field = test_field();
        let root_seed = field.root_seed;
        let res = 48;
        let mut lowland_texels = 0usize;
        let mut active_dune_texels = 0usize;
        let mut dune_texels = 0usize;
        for face in CubemapFace::ALL {
            for y in 0..res {
                for x in 0..res {
                    let u = (x as f32 + 0.5) / res as f32;
                    let v = (y as f32 + 0.5) / res as f32;
                    let dir = face_uv_to_dir(face, u, v);
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
                    let contact = dune_basin_contact(dir, root_seed, lowland_bias, highland_ridges);
                    if contact.paleo_lowland > 0.5 {
                        lowland_texels += 1;
                    }
                    if contact.dune_plate > 0.5 {
                        active_dune_texels += 1;
                    }

                    let sample = field.sample(dir, 1_000.0);
                    if sample.material_mix.dominant_material_id() == VAELEN_MAT_DUNE_SAND {
                        dune_texels += 1;
                    }
                }
            }
        }

        let total = (CubemapFace::ALL.len() * res * res) as f32;
        let lowland_coverage = lowland_texels as f32 / total;
        let active_dune_coverage = active_dune_texels as f32 / total;
        let dune_coverage = dune_texels as f32 / total;
        assert!(
            (0.16..=0.30).contains(&lowland_coverage),
            "paleo-ocean lowland should be planet-scale, got {lowland_coverage:.3}"
        );
        assert!(
            (0.02..=0.18).contains(&active_dune_coverage),
            "active dunes should be a localized basin subset, got {active_dune_coverage:.3}"
        );
        assert!(
            active_dune_coverage < lowland_coverage * 0.80,
            "active dunes should not cover every paleo-lowland: lowland={lowland_coverage:.3}, active={active_dune_coverage:.3}"
        );
        assert!(
            dune_coverage <= lowland_coverage * 0.95,
            "dominant dune material should stay inside the localized low basin: lowland={lowland_coverage:.3}, dunes={dune_coverage:.3}"
        );
    }

    #[test]
    fn paleo_ocean_outline_is_not_a_round_stamp() {
        let field = test_field();
        let root_seed = field.root_seed;
        let mut radii = Vec::new();

        for i in 0..40 {
            let theta = i as f32 / 40.0 * std::f32::consts::TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            let mut lo = 0.02;
            let mut hi = 1.05;
            let center = dir_from_basin_xy(0.0, 0.0);
            let macro_n = fbm_dir(center, root_seed.shape, "macro", 1.15, 5, 0.55);
            let regional_n = fbm_dir(center, root_seed.shape, "regional", 2.7, 4, 0.55);
            let highland_ridges = ridge(fbm_dir(
                center,
                root_seed.shape,
                "highland_ridge",
                5.4,
                4,
                0.52,
            ));
            let lowland_bias = smoothstep(0.72, -0.20, macro_n + regional_n * 0.35);
            if dune_basin_contact(center, root_seed, lowland_bias, highland_ridges).signed <= 0.0 {
                continue;
            }

            let edge = dir_from_basin_xy(hi * cos_t, hi * sin_t);
            let macro_n = fbm_dir(edge, root_seed.shape, "macro", 1.15, 5, 0.55);
            let regional_n = fbm_dir(edge, root_seed.shape, "regional", 2.7, 4, 0.55);
            let highland_ridges = ridge(fbm_dir(
                edge,
                root_seed.shape,
                "highland_ridge",
                5.4,
                4,
                0.52,
            ));
            let lowland_bias = smoothstep(0.72, -0.20, macro_n + regional_n * 0.35);
            if dune_basin_contact(edge, root_seed, lowland_bias, highland_ridges).signed > 0.0 {
                continue;
            }

            for _ in 0..18 {
                let mid = (lo + hi) * 0.5;
                let dir = dir_from_basin_xy(mid * cos_t, mid * sin_t);
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
                if dune_basin_contact(dir, root_seed, lowland_bias, highland_ridges).signed > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            radii.push((lo + hi) * 0.5);
        }

        assert!(radii.len() >= 28, "too few closed shoreline rays");
        let min_r = radii.iter().copied().fold(f32::INFINITY, f32::min);
        let max_r = radii.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_r / min_r > 1.45,
            "paleo-ocean outline should have tectonic-scale lobes, min={min_r:.3}, max={max_r:.3}"
        );
    }

    #[test]
    fn paleo_ocean_side_margins_are_not_straight_cuts() {
        let field = test_field();
        let mut west_edges = Vec::new();
        let mut east_edges = Vec::new();

        for i in 0..34 {
            let y = -0.56 + i as f32 * 0.034;

            let mut last_inside = false;
            let mut west_edge = None;
            for xi in 0..80 {
                let x = -1.02 + xi as f32 * 0.018;
                if x * x + y * y > 0.995 {
                    continue;
                }
                let inside = contact_at_xy(&field, x, y).signed > 0.0;
                if inside && !last_inside {
                    west_edge = Some(x);
                    break;
                }
                last_inside = inside;
            }
            if let Some(x) = west_edge {
                west_edges.push(x);
            }

            let mut last_inside = false;
            let mut east_edge = None;
            for xi in 0..80 {
                let x = 1.02 - xi as f32 * 0.018;
                if x * x + y * y > 0.995 {
                    continue;
                }
                let inside = contact_at_xy(&field, x, y).signed > 0.0;
                if inside && !last_inside {
                    east_edge = Some(x);
                    break;
                }
                last_inside = inside;
            }
            if let Some(x) = east_edge {
                east_edges.push(x);
            }
        }

        assert!(west_edges.len() > 18, "not enough west margin samples");
        assert!(east_edges.len() > 18, "not enough east margin samples");
        let west_range = west_edges.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - west_edges.iter().copied().fold(f32::INFINITY, f32::min);
        let east_range = east_edges.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - east_edges.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            west_range > 0.16,
            "west paleo-ocean margin still reads too straight, x-range={west_range:.3}"
        );
        assert!(
            east_range > 0.14,
            "east paleo-ocean margin still reads too straight, x-range={east_range:.3}"
        );
    }

    #[test]
    fn paleo_ocean_floor_has_shelves_and_interior_highs() {
        let field = test_field();
        let mut quiet_floor_sum = 0.0f32;
        let mut quiet_floor_count = 0usize;
        let mut horst = (0.0f32, Vec3::Z, f32::NEG_INFINITY);
        let mut shelf = (0.0f32, Vec3::Z, f32::NEG_INFINITY);
        let mut strandline = 0.0f32;

        for yi in 0..56 {
            let y = -0.66 + yi as f32 * 0.024;
            for xi in 0..68 {
                let x = -0.82 + xi as f32 * 0.024;
                if x * x + y * y > 0.98 {
                    continue;
                }

                let dir = dir_from_basin_xy(x, y);
                let contact = contact_at_xy(&field, x, y);
                if contact.paleo_lowland < 0.75 {
                    continue;
                }

                let height = field.sample(dir, 1_000.0).height_m;
                if contact.interior_highs > horst.0 {
                    horst = (contact.interior_highs, dir, height);
                }
                if contact.shelf_break > shelf.0 {
                    shelf = (contact.shelf_break, dir, height);
                }
                strandline = strandline.max(contact.strandline);

                if contact.interior_highs < 0.05
                    && contact.shelf_break < 0.05
                    && contact.strandline < 0.05
                    && contact.shoreline_scarp < 0.05
                {
                    quiet_floor_sum += height;
                    quiet_floor_count += 1;
                }
            }
        }

        assert!(
            quiet_floor_count > 40,
            "not enough quiet paleo-ocean floor samples"
        );
        let quiet_floor_h = quiet_floor_sum / quiet_floor_count as f32;
        assert!(horst.0 > 0.55, "weak interior high mask: {:.3}", horst.0);
        assert!(shelf.0 > 0.50, "weak shelf break mask: {:.3}", shelf.0);
        assert!(
            strandline > 0.35,
            "fossil strandline mask should create visible benches, got {strandline:.3}"
        );
        assert!(
            horst.2 > quiet_floor_h + 180.0,
            "interior highs should rise above quiet floor: horst={:.1}, quiet={:.1}",
            horst.2,
            quiet_floor_h
        );
        assert!(
            shelf.0 * 330.0 > 160.0,
            "shelf breaks should add a strong local relief contribution, mask={:.3}",
            shelf.0
        );
    }

    #[test]
    fn suture_crest_is_a_real_height_peak() {
        let field = test_field();
        let root_seed = field.root_seed;

        let mut crest = (0.0f32, Vec3::Z, f32::NEG_INFINITY);
        let mut dune = (f32::NEG_INFINITY, Vec3::Z);
        let mut highland = (f32::INFINITY, Vec3::Z);
        for i in 0..180 {
            let x = 0.20 + i as f32 * 0.006;
            let dir = dir_from_basin_xy(x, 0.02);
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
            let contact = dune_basin_contact(dir, root_seed, lowland_bias, highland_ridges);

            if contact.suture_crest > crest.0 {
                crest = (
                    contact.suture_crest,
                    dir,
                    field.sample(dir, 1_000.0).height_m,
                );
            }
            if contact.signed > dune.0 {
                dune = (contact.signed, dir);
            }
            if contact.signed < highland.0 {
                highland = (contact.signed, dir);
            }
        }

        let dune_h = field.sample(dune.1, 1_000.0).height_m;
        let highland_h = field.sample(highland.1, 1_000.0).height_m;
        assert!(crest.0 > 0.65, "weak suture crest mask: {:.3}", crest.0);
        assert!(
            crest.2 > dune_h + 850.0,
            "suture crest should tower over the dune plate: crest={:.1}, dune={:.1}",
            crest.2,
            dune_h
        );
        assert!(
            crest.2 > highland_h + 350.0,
            "suture crest should peak above the highland plate: crest={:.1}, highland={:.1}",
            crest.2,
            highland_h
        );
    }

    #[test]
    fn dune_material_is_clipped_to_low_plate() {
        let field = test_field();
        let dune = field.sample(dir_from_basin_xy(0.0, 0.0), 1_000.0);
        let highland = field.sample(dir_from_basin_xy(0.99, 0.0), 1_000.0);

        assert!(
            dune.material_mix.weight_for(VAELEN_MAT_DUNE_SAND) > 0.55,
            "basin interior should be mostly dune sand"
        );
        assert!(
            highland.material_mix.weight_for(VAELEN_MAT_DUNE_SAND) < 0.04,
            "highland side should not carry dune paint"
        );
    }

    #[test]
    fn mountain_web_stays_on_highland_side_near_suture() {
        let field = test_field();
        let root_seed = field.root_seed;
        let width = 96;
        let height = 48;
        let mut highland_web = 0.0f32;
        let mut dune_web = 0.0f32;
        let mut far_crest = 0.0f32;
        let mut far_web = 0.0f32;

        for y in 0..height {
            for x in 0..width {
                let dir = equirect_dir(x, y, width, height);
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
                let contact = dune_basin_contact(dir, root_seed, lowland_bias, highland_ridges);

                if contact.signed < 0.0 {
                    highland_web += contact.mountain_web;
                } else {
                    dune_web += contact.mountain_web;
                }

                if contact.signed.abs() > 0.28 {
                    far_crest = far_crest.max(contact.suture_crest);
                    far_web = far_web.max(contact.mountain_web);
                }
            }
        }

        assert!(
            highland_web > dune_web * 8.0,
            "mountain web should mostly live on the highland side"
        );
        assert!(
            far_crest < 0.02,
            "suture crest should be confined to the boundary, got far mask {far_crest:.3}"
        );
        assert!(
            far_web < 0.08,
            "mountain web should decay away from the boundary, got far mask {far_web:.3}"
        );
    }
}
