use std::sync::Arc;

use bevy::prelude::*;
use thalos_body_render::CLOUD_BAND_COUNT;
use thalos_physics_canonical::{
    body_trajectory_provider::BodyTrajectoryProvider, canonical::Epoch, simulation::Simulation,
    types::BodyStates,
};
use thalos_terrain::DynamicSurfaceState;
use thalos_terrain::cubemap::{CubemapFace, face_uv_to_dir};
use thalos_world::{BodyId, CloudClimate, SolarSystemDefinition};

use crate::SimStage;

/// Central simulation state: the long-lived authority that advances time,
/// craft state, flight plans, and the active body trajectory provider.
#[derive(Resource)]
pub struct SimulationState {
    pub simulation: Simulation,
    pub system: SolarSystemDefinition,
    pub ephemeris: Arc<dyn BodyTrajectoryProvider>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudBandEnvironmentState {
    pub phases: [f64; CLOUD_BAND_COUNT],
    pub scroll_rate_rad_s: f64,
    pub differential_rotation: f64,
}

impl CloudBandEnvironmentState {
    // Constructed once banded-cloud bodies are wired at spawn (see
    // `install_cloud_band_state`); kept as the clamping constructor.
    #[allow(dead_code)]
    pub fn new(scroll_rate_rad_s: f64, differential_rotation: f64) -> Self {
        Self {
            phases: [0.0; CLOUD_BAND_COUNT],
            scroll_rate_rad_s,
            differential_rotation: differential_rotation.clamp(0.0, 1.0),
        }
    }

    pub fn advance(&mut self, dt: f64) {
        if dt == 0.0 || self.scroll_rate_rad_s.abs() < 1.0e-12 {
            return;
        }

        for i in 0..CLOUD_BAND_COUNT {
            let sin2 = i as f64 / (CLOUD_BAND_COUNT - 1) as f64;
            let lat_factor = 1.0 - self.differential_rotation * sin2;
            let omega = self.scroll_rate_rad_s * lat_factor;
            self.phases[i] = (self.phases[i] + omega * dt).rem_euclid(std::f64::consts::TAU);
        }
    }
}

/// Per-face resolution of the canonical cubemap weather field. A 90° face at
/// Thalos's 3,186 km radius spans about 5,005 km, so 1024 texels resolve
/// ~4.9 km at the face centre (the former 256 field was ~19.5 km/texel, not
/// the stale ~60 km estimate). CLOUD-3 still supplies finer 3-D shape detail.
pub const CLOUD_WEATHER_FACE_SIZE: u32 = 1024;

/// Mutable, per-body weather and broad-density authority. Both cubemap payloads
/// are stored face-major in [`CubemapFace::ALL`] order. `texels` carries RGBA8
/// coverage, cloud type, normalized base, and normalized top;
/// `surface_density_texels` carries the broad body-space shape signal at four
/// normalized-height strata. Every render projection consumes these together;
/// no projection owns an independent pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct CloudWeatherField {
    pub seed: u64,
    pub face_size: u32,
    pub texels: Vec<[u8; 4]>,
    pub surface_density_texels: Vec<[u8; 4]>,
    pub coverage_mean: f32,
    pub base_altitude_m: f32,
    pub top_altitude_m: f32,
    pub albedo: [f32; 3],
    pub wind_m_s: [f32; 2],
    /// Consumers re-upload or reject temporal history when this changes.
    pub version: u32,
}

impl CloudWeatherField {
    pub fn from_climate(climate: &CloudClimate) -> Self {
        let face_size = CLOUD_WEATHER_FACE_SIZE;
        let mut texels = Vec::with_capacity((face_size * face_size * 6) as usize);
        let mut surface_density_texels = Vec::with_capacity((face_size * face_size * 6) as usize);
        let mix_sum = climate.type_mix.iter().copied().sum::<f32>();
        let type_mix = if mix_sum > 1.0e-6 {
            climate.type_mix.map(|value| value.max(0.0) / mix_sum)
        } else {
            [0.25, 0.55, 0.20]
        };

        for face in CubemapFace::ALL {
            for y in 0..face_size {
                let v = (y as f32 + 0.5) / face_size as f32;
                for x in 0..face_size {
                    let u = (x as f32 + 0.5) / face_size as f32;
                    let dir = face_uv_to_dir(face, u, v).normalize();
                    // Meteorological banding is a *bias*, not a paint ring:
                    // warp the latitude the profile reads (so band edges
                    // meander like jet streams) and gate its strength with a
                    // continental-scale field (so bands appear as broken
                    // segments, not complete rings). The straight gaussian
                    // rings previously survived the far projection's coverage
                    // threshold as conspicuous synthetic stripes.
                    let band_warp = 0.55
                        * (fbm3(
                            dir * 3.1 + Vec3::new(7.0, -3.0, 11.0),
                            climate.seed ^ 0xBA9D,
                            3,
                        ) - 0.5);
                    let band_gate = (0.35
                        + 1.30
                            * fbm3(
                                dir * 2.1 + Vec3::new(-2.0, 9.0, 5.0),
                                climate.seed ^ 0x6A7E,
                                3,
                            ))
                    .clamp(0.0, 1.0);
                    let band = latitude_band_profile(dir.y.asin() + band_warp) * band_gate;
                    // Zonally-elongated ridge field: fronts. Compressing the
                    // noise domain in latitude while stretching it in
                    // longitude makes features read as elongated frontal
                    // bands; the ridge transform sharpens them into lines of
                    // enhanced coverage. The ridge domain must be warped by an
                    // independent field: un-warped ridged value noise draws
                    // closed contours around every lattice extremum, which the
                    // far projection rendered as bullseye rings across the
                    // whole disc. Squaring keeps only the strong crests.
                    let front_warp = fbm3(
                        dir * 1.9 + Vec3::new(-6.0, 2.0, 14.0),
                        climate.seed ^ 0x11AB,
                        2,
                    ) - 0.5;
                    let frontal_raw = fbm3(
                        Vec3::new(dir.x * 2.2, dir.y * 7.5, dir.z * 2.2)
                            + Vec3::new(3.0, -8.0, 1.0)
                            + Vec3::splat(1.6 * front_warp),
                        climate.seed ^ 0xF407,
                        3,
                    );
                    let ridge = 1.0 - (2.0 * frontal_raw - 1.0).abs();
                    let frontal = ridge * ridge;
                    let regional = fbm3(dir * 2.5, climate.seed, 4);
                    // Coverage needs two meteorological scales. The synoptic
                    // component establishes planetary bands and fronts while
                    // the mesoscale component breaks those systems into the
                    // distinct cells that the same field must expose both to
                    // the near volume and to an orbital projection. Cloud type
                    // remains categorical and must not be abused as a shape
                    // mask: doing so produces conspicuous cubemap-sized blocks.
                    let mesoscale_warp = fbm3(
                        dir * 17.0 + Vec3::new(13.0, -4.0, 29.0),
                        climate.seed ^ 0x5CA1_0A11,
                        3,
                    ) - 0.5;
                    let mesoscale = fbm3(
                        dir * 52.0
                            + Vec3::new(-11.0, 23.0, 7.0)
                            + Vec3::splat(2.4 * mesoscale_warp),
                        climate.seed ^ 0x5CA1_E5CA,
                        4,
                    );
                    let cellular_mass = fbm3(
                        dir * 128.0
                            + Vec3::new(19.0, -5.0, 37.0)
                            + Vec3::splat(-3.1 * mesoscale_warp),
                        climate.seed ^ 0xCE11_C10D,
                        3,
                    );
                    let cellular_cut_raw = fbm3(
                        dir * 211.0 + Vec3::new(-31.0, 47.0, 5.0),
                        climate.seed ^ 0xCE11_5EED,
                        2,
                    );
                    let cellular_cut = 1.0 - (2.0 * cellular_cut_raw - 1.0).abs();
                    let cellular = 0.68 * cellular_mass + 0.32 * cellular_cut;

                    // ── Regime-structured occupancy (BL-20260723T165923Z) ──
                    // Real skies are organized, not statistically uniform: a
                    // synoptic OCCUPANCY field thresholded into weather systems
                    // with genuinely clear air between them, and a coherent
                    // REGIME per region (scattered-cumulus field / stratus
                    // sheet / storm cluster, plus frontal ridges) that sets the
                    // local coverage texture, cloud type, and vertical extent.
                    // The previous producer summed fixed-scale noises around
                    // one mean, which rendered the whole planet as the same
                    // mid-cumulus speckle (2026-07-23 user verdict).
                    //
                    // Occupancy: threshold the synoptic field at the quantile
                    // matching authored mean coverage; soft edges so systems
                    // thin out rather than shear off.
                    let system_field = regional
                        + 0.35 * climate.band_strength * band
                        + 0.08 * (mesoscale - 0.5);
                    let occ_threshold = 0.70 - 0.36 * climate.coverage.clamp(0.0, 1.0);
                    let occupancy =
                        smoothstep(occ_threshold - 0.07, occ_threshold + 0.09, system_field);
                    // 0 at a system's fringe, 1 deep inside: deep systems are
                    // more developed (storm potential, higher fill).
                    let intensity =
                        smoothstep(occ_threshold + 0.02, occ_threshold + 0.17, system_field);

                    // Regime selector: an independent low-frequency partition,
                    // uniformized so the authored type_mix reads as area
                    // fractions of the cloudy world. Storms additionally need a
                    // developed system core.
                    let regime = fbm3(
                        dir * 3.6 + Vec3::new(23.0, 5.0, -12.0),
                        climate.seed ^ 0xC0DE,
                        3,
                    );
                    let regime_x = smoothstep(0.36, 0.64, regime);
                    let m_stratus = type_mix[0].clamp(0.0, 0.9);
                    let m_storm = type_mix[2].clamp(0.0, 0.9);
                    let stratus_region = smoothstep(
                        1.0 - m_stratus - 0.06,
                        (1.0 - m_stratus + 0.06).min(1.0),
                        regime_x,
                    );
                    let storm_region = (1.0
                        - smoothstep((m_storm - 0.06).max(0.0), m_storm + 0.06, regime_x))
                        * (0.30 + 0.70 * intensity);
                    let cumulus_region = (1.0 - stratus_region - storm_region).max(0.0);

                    // Per-regime coverage texture. `variation` scales how deep
                    // the mesoscale/cellular breakup cuts.
                    let breakup = (0.55 + climate.variation).clamp(0.5, 1.5);
                    let cell_broken =
                        smoothstep(0.42, 0.78, 0.62 * cellular + 0.38 * mesoscale);
                    let sheet_holes = smoothstep(0.66, 0.84, cellular);
                    let storm_core =
                        smoothstep(0.52, 0.78, 0.66 * mesoscale + 0.34 * cellular);
                    let frontal_boost = frontal * occupancy;
                    let cumulus_cov =
                        (0.66 - 0.52 * breakup * (1.0 - cell_broken) + 0.10 * intensity)
                            .clamp(0.0, 1.0);
                    let stratus_cov = 0.94 - 0.30 * breakup * sheet_holes;
                    let storm_cov =
                        (0.40 + 0.46 * storm_core + 0.10 * intensity).clamp(0.0, 1.0);
                    let cov_regime = stratus_region * stratus_cov
                        + cumulus_region * cumulus_cov
                        + storm_region * storm_cov;
                    let coverage =
                        (occupancy * cov_regime + 0.28 * frontal_boost).clamp(0.0, 1.0);

                    // Cloud type follows the regime: sheets read stratus, storm
                    // clusters read cumulonimbus at their cores, and building
                    // cells inside deep cumulus fields turn congestus. Fronts
                    // push toward storm so ridge lines carry tall cloud.
                    // Building cells appear in ordinary cumulus fields too, not
                    // only deep systems — within-horizon vertical hierarchy is
                    // the main local monotony breaker (2026-07-23 round 2).
                    let congestus = storm_core * (0.30 + 0.70 * intensity);
                    let cloud_type = (stratus_region * 0.08
                        + cumulus_region * (0.42 + 0.30 * congestus)
                        + storm_region * (0.78 + 0.18 * storm_core)
                        + 0.10 * frontal_boost)
                        .clamp(0.02, 0.97);
                    let vertical_noise = fbm3(
                        dir * 34.0 + Vec3::new(31.0, -7.0, 13.0),
                        climate.seed ^ 0xA11E,
                        3,
                    );
                    // Local base/top are fractions of the authored shell, per
                    // regime: thin low stratus decks, cumulus growing with its
                    // cells, storm towers claiming most of the thickness so
                    // limb silhouettes keep height.
                    let base_stratus = 0.09 + 0.05 * vertical_noise;
                    let base_cumulus = 0.10 + 0.07 * vertical_noise;
                    let base_storm = 0.05 + 0.04 * vertical_noise;
                    let top_stratus = base_stratus + 0.10 + 0.08 * vertical_noise;
                    // Ordinary cumulus fields develop real depth (round 7):
                    // the former 0.09 baseline gave plain fair-weather
                    // columns <1 km of a 10.5 km shell — everything below
                    // congestus rendered as a squat sheet. Building cells now
                    // carry more of the growth so broken fields read as
                    // mixed-height puffs rather than one flat deck.
                    let top_cumulus = base_cumulus
                        + 0.14
                        + 0.58 * (0.42 * cell_broken + 0.58 * congestus)
                        + 0.09 * vertical_noise;
                    let top_storm = 0.60 + 0.38 * storm_core;
                    let base = stratus_region * base_stratus
                        + cumulus_region * base_cumulus
                        + storm_region * base_storm;
                    let top = (stratus_region * top_stratus
                        + cumulus_region * top_cumulus
                        + storm_region * top_storm)
                        .max(base + 0.04);
                    // Canonical surface-space broad shape. These fields live on
                    // the unit direction sphere, so they are seamless across
                    // cubemap faces and never inherit the near volume's small
                    // Cartesian repeat. Four correlated strata let towers lean
                    // and split with height without storing a full 3-D shell.
                    // Coverage/type/base/top remain separate climate controls;
                    // shaders apply their shared threshold/profile contract.
                    let shape_warp = fbm3(
                        dir * 41.0 + Vec3::new(43.0, -17.0, 9.0),
                        climate.seed ^ 0x5A11_FACE,
                        2,
                    ) - 0.5;
                    let shape_mass = fbm3(
                        dir * 128.0 + Vec3::new(-37.0, 61.0, 23.0) + Vec3::splat(7.5 * shape_warp),
                        climate.seed ^ 0xD315_17A1,
                        3,
                    );
                    let shape_cut_raw = fbm3(
                        dir * 211.0 + Vec3::new(71.0, -29.0, 53.0) + Vec3::splat(-5.0 * shape_warp),
                        climate.seed ^ 0xCE11_B0D1,
                        2,
                    );
                    let shape_cut = 1.0 - (2.0 * shape_cut_raw - 1.0).abs();
                    let surface_shape = [
                        0.56 * shape_mass + 0.24 * shape_cut + 0.20 * cellular,
                        0.48 * shape_mass + 0.25 * shape_cut + 0.17 * regime_x + 0.10 * cellular,
                        0.39 * shape_mass
                            + 0.23 * shape_cut
                            + 0.21 * vertical_noise
                            + 0.17 * regime_x,
                        0.31 * shape_mass
                            + 0.20 * shape_cut
                            + 0.28 * vertical_noise
                            + 0.21 * regime_x,
                    ];
                    // Strata are LAYER-RELATIVE: four samples across the local
                    // [base, top] interval, not the whole shell. Fixed shell
                    // heights had a dead zone — a ~2 km deck sitting between
                    // two sampling heights read zero from every stratum, so
                    // the far tier showed clear sky over a solid near-volume
                    // deck (2026-07-23). Consumers map their shell height
                    // through the same weather base/top channels.
                    let top_c = top.max(base + 0.02);
                    let surface_density = [0.125, 0.375, 0.625, 0.875].map(|q| {
                        cloud_surface_density_cpu(
                            surface_shape,
                            base + q * (top_c - base),
                            coverage,
                            cloud_type,
                            base,
                            top_c,
                        )
                    });
                    let encode = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    texels.push([
                        encode(coverage),
                        encode(cloud_type),
                        encode(base),
                        encode(top_c),
                    ]);
                    surface_density_texels.push(surface_density.map(encode));
                }
            }
        }

        Self {
            seed: climate.seed,
            face_size,
            texels,
            surface_density_texels,
            coverage_mean: climate.coverage.clamp(0.0, 1.0),
            base_altitude_m: climate.base_altitude_m.max(0.0),
            top_altitude_m: (climate.base_altitude_m + climate.thickness_m).max(0.0),
            albedo: climate.albedo,
            wind_m_s: climate.wind_m_s,
            version: 0,
        }
    }

    /// Number of mip levels the weather cube carries (1024 → 8 px faces).
    /// Far projections select a level from their projected footprint; without
    /// this chain the mesoscale/cellular coverage aliases into ring/speckle
    /// moiré at disc scale.
    pub const MIP_LEVELS: u32 = 8;

    /// Full RGBA8 cube payload with a box-filtered mip chain, laid out
    /// layer-major (face0[mip0..], face1[mip0..], …) to match wgpu's
    /// `TextureDataOrder::LayerMajor` default used by Bevy's image uploads.
    pub fn rgba8_mip_chain(&self) -> Vec<u8> {
        rgba8_cube_mip_chain(&self.texels, self.face_size, Self::MIP_LEVELS)
    }

    /// Full four-stratum surface-density cube payload with the same layout and
    /// mip contract as [`Self::rgba8_mip_chain`].
    pub fn surface_density_rgba8_mip_chain(&self) -> Vec<u8> {
        rgba8_cube_mip_chain(
            &self.surface_density_texels,
            self.face_size,
            Self::MIP_LEVELS,
        )
    }
}

fn rgba8_cube_mip_chain(texels: &[[u8; 4]], face_size: u32, mip_levels: u32) -> Vec<u8> {
    let size = face_size as usize;
    let face_texels = size * size;
    assert_eq!(texels.len(), 6 * face_texels, "cloud cubemap texel count");
    let mut out = Vec::new();
    for face in 0..6 {
        let base = &texels[face * face_texels..(face + 1) * face_texels];
        let mut level: Vec<[u8; 4]> = base.to_vec();
        let mut level_size = size;
        out.extend(level.iter().flatten());
        for _ in 1..mip_levels {
            let next_size = (level_size / 2).max(1);
            let mut next = Vec::with_capacity(next_size * next_size);
            for y in 0..next_size {
                for x in 0..next_size {
                    let mut acc = [0u32; 4];
                    for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let sy = (y * 2 + dy).min(level_size - 1);
                        let sx = (x * 2 + dx).min(level_size - 1);
                        let t = level[sy * level_size + sx];
                        for c in 0..4 {
                            acc[c] += u32::from(t[c]);
                        }
                    }
                    next.push([
                        (acc[0] / 4) as u8,
                        (acc[1] / 4) as u8,
                        (acc[2] / 4) as u8,
                        (acc[3] / 4) as u8,
                    ]);
                }
            }
            out.extend(next.iter().flatten());
            level = next;
            level_size = next_size;
        }
    }
    out
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(f32::EPSILON)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// CPU producer for the canonical four-stratum density payload. Nonlinear
/// formation/profile response happens before the mip chain is built, so far
/// footprints average occupied area instead of thresholding an averaged raw
/// signal into salt-and-pepper cloud pixels.
fn cloud_surface_density_cpu(
    surface_shape: [f32; 4],
    normalized_height: f32,
    coverage: f32,
    cloud_type: f32,
    local_base: f32,
    local_top: f32,
) -> f32 {
    let h = (normalized_height - local_base) / (local_top - local_base).max(0.02);
    let cov = (coverage * 1.25).clamp(0.0, 1.0);
    if h <= 0.0 || h >= 1.0 || cov <= 1.0e-3 {
        return 0.0;
    }

    let stratus_w = 1.0 - smoothstep(0.18, 0.38, cloud_type);
    let storm_w = smoothstep(0.72, 0.88, cloud_type);
    let cumulus_w = (1.0 - stratus_w - storm_w).max(0.0);
    let threshold = 0.58 + (0.30 - 0.58) * cov;
    // Round-7 dome sculpting: convective tops are carved by a quadratically
    // height-rising threshold (a per-lobe noise isosurface — strong lobes
    // tower, weak lobes stay squat) instead of the former linear thinning;
    // tall congestus/storm columns keep more mass with height so towers stay
    // coherent. Mirrored in `get_cloud_map_density` (clouds_compute.wgsl) and
    // `march_column` (fill_lut.rs) — keep the three in lockstep.
    let column_tall = smoothstep(0.30, 0.65, local_top - local_base);
    let vertical_narrow = h * 0.04 * stratus_w
        + (h * h) * (0.42 * cumulus_w + 0.30 * storm_w) * (1.0 - 0.45 * column_tall);

    let z = normalized_height.clamp(0.0, 1.0) * 4.0 - 0.5;
    let shape = if z <= 0.0 {
        surface_shape[0]
    } else if z < 1.0 {
        surface_shape[0] + (surface_shape[1] - surface_shape[0]) * z
    } else if z < 2.0 {
        surface_shape[1] + (surface_shape[2] - surface_shape[1]) * (z - 1.0)
    } else if z < 3.0 {
        surface_shape[2] + (surface_shape[3] - surface_shape[2]) * (z - 2.0)
    } else {
        surface_shape[3]
    };
    let mut mass = shape - threshold - vertical_narrow;
    let anvil_profile = smoothstep(0.62, 0.76, h) * (1.0 - smoothstep(0.90, 1.0, h));
    mass = mass.max((shape - (threshold - 0.06)) * anvil_profile * storm_w);

    let bottom_softness = 0.16;
    // Thin condensation top skins (the dome term above owns top shape);
    // stratus stays a genuine sheet. Lockstep with the marcher + fill_lut.
    let stratus_profile =
        smoothstep(0.0, bottom_softness * 0.45, h) * (1.0 - smoothstep(0.72, 1.0, h));
    let cumulus_profile =
        smoothstep(0.0, bottom_softness * 0.75, h) * (1.0 - smoothstep(0.93, 1.0, h));
    let storm_profile =
        smoothstep(0.0, bottom_softness * 0.35, h) * (1.0 - smoothstep(0.94, 1.0, h));
    let vertical_profile =
        stratus_profile * stratus_w + cumulus_profile * cumulus_w + storm_profile * storm_w;
    smoothstep(0.0, 0.055, mass) * vertical_profile
}

fn latitude_band_profile(lat: f32) -> f32 {
    let gauss =
        |x: f32, center: f32, width: f32| (-((x - center) / width) * ((x - center) / width)).exp();
    let a = lat.abs();
    gauss(a, 0.0, 0.10) + 0.7 * gauss(a, 0.96, 0.24)
        - 0.8 * gauss(a, 0.44, 0.15)
        - 0.4 * gauss(a, std::f32::consts::FRAC_PI_2, 0.25)
}

fn hash3(p: IVec3, seed: u64) -> f32 {
    let mut h = (p.x as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (p.y as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (p.z as i64 as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ seed;
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h & 0x00FF_FFFF) as f32 / 16_777_216.0
}

fn value_noise3(p: Vec3, seed: u64) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let i = i.as_ivec3();
    let corner = |dx: i32, dy: i32, dz: i32| hash3(i + IVec3::new(dx, dy, dz), seed);
    let x00 = corner(0, 0, 0) + (corner(1, 0, 0) - corner(0, 0, 0)) * u.x;
    let x10 = corner(0, 1, 0) + (corner(1, 1, 0) - corner(0, 1, 0)) * u.x;
    let x01 = corner(0, 0, 1) + (corner(1, 0, 1) - corner(0, 0, 1)) * u.x;
    let x11 = corner(0, 1, 1) + (corner(1, 1, 1) - corner(0, 1, 1)) * u.x;
    let y0 = x00 + (x10 - x00) * u.y;
    let y1 = x01 + (x11 - x01) * u.y;
    y0 + (y1 - y0) * u.z
}

fn fbm3(p: Vec3, seed: u64, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 0.5;
    let mut norm = 0.0;
    let mut q = p;
    for _ in 0..octaves {
        sum += amplitude * value_noise3(q, seed);
        norm += amplitude;
        amplitude *= 0.5;
        q = q * 2.03 + Vec3::new(13.1, 7.7, 19.3);
    }
    sum / norm.max(f32::EPSILON)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BodyEnvironmentState {
    /// Mutable runtime state for terrain-owned dynamic layers: seasonal ice,
    /// active dunes, and later weather/tide-driven surface overlays.
    pub dynamic_surface: Option<DynamicSurfaceState>,
    /// Atmospheric cloud-band motion and phases. Kept here, not on render
    /// components, so map impostors, ship impostors, terrain skies, and future
    /// weather systems all see the same cloud state.
    pub cloud_bands: Option<CloudBandEnvironmentState>,
    /// Canonical large-scale volumetric-cloud weather. `None` mirrors an
    /// authored `CloudClimate::None`; renderers must not install defaults.
    pub cloud_weather: Option<CloudWeatherField>,
}

/// Canonical evaluated solar-system state for the current game frame.
///
/// This is the source that projections consume. Bevy entities, impostor
/// materials, terrain tile providers, map snapshots, and atmosphere passes may
/// cache derived data, but they should not independently evaluate or own body
/// state. Future wind, storms, tides, and dune migration belong in
/// [`BodyEnvironmentState`] so every projection reads the same runtime
/// environment for a body.
///
/// **Sole writer:** [`sync_solar_system_state`] (in [`SimStage::Sync`]). All
/// other systems read it; environment mutators go through `environment_mut`.
#[derive(Resource, Debug, Default)]
pub struct SolarSystemState {
    pub states: Option<BodyStates>,
    pub time: f64,
    pub environment: Vec<BodyEnvironmentState>,
}

impl SolarSystemState {
    pub fn environment_mut(&mut self, body_id: BodyId) -> Option<&mut BodyEnvironmentState> {
        self.environment.get_mut(body_id)
    }

    fn ensure_body_capacity(&mut self, body_count: usize) {
        if self.environment.len() < body_count {
            self.environment
                .resize_with(body_count, BodyEnvironmentState::default);
        }
    }

    // Forward environment-install API, ready for spawn-time wiring:
    // `install_cloud_band_state` lights up the `update_cloud_bands` drift
    // loop the moment a body is given cloud bands. Kept symmetric with the live
    // `install_cloud_weather`.
    #[allow(dead_code)]
    pub fn install_cloud_band_state(&mut self, body_id: BodyId, state: CloudBandEnvironmentState) {
        self.ensure_body_capacity(body_id + 1);
        self.environment[body_id].cloud_bands = Some(state);
    }

    pub fn install_cloud_weather(&mut self, body_id: BodyId, state: CloudWeatherField) {
        self.ensure_body_capacity(body_id + 1);
        self.environment[body_id].cloud_weather = Some(state);
    }
}

pub fn sync_solar_system_state(
    sim: Res<SimulationState>,
    mut solar_system: ResMut<SolarSystemState>,
) {
    let epoch = Epoch(sim.simulation.sim_time());
    if solar_system.states.is_some() && (solar_system.time - epoch.0).abs() < f64::EPSILON {
        return;
    }

    if let Some(states) = solar_system.states.as_mut() {
        sim.ephemeris.states_into(epoch, states);
    } else {
        let mut states = Vec::with_capacity(sim.ephemeris.body_count());
        sim.ephemeris.states_into(epoch, &mut states);
        solar_system.states = Some(states);
    }
    solar_system.time = epoch.0;
    solar_system.ensure_body_capacity(sim.ephemeris.body_count());
}

pub struct SolarSystemStatePlugin;

impl Plugin for SolarSystemStatePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SolarSystemState>()
            .add_systems(Update, sync_solar_system_state.in_set(SimStage::Sync));
    }
}

#[cfg(test)]
mod cloud_site_probe {
    use super::*;

    /// Dev probe for the BL-20260723T214730Z thickness-parity protocol: scan
    /// the authored Thalos weather field for *cloudy* sites near the runway's
    /// daylight longitude, so tier A/B captures can frame real cloud (the
    /// default spaceport column is authored nearly clear). Prints
    /// `THALOS_RUNWAY_SITE` candidates.
    ///
    /// Run: `cargo test -p thalos_runtime --lib cloud_site_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "dev probe: prints cloudy THALOS_RUNWAY_SITE candidates"]
    fn print_cloudy_sites() {
        let assets =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets");
        let system = thalos_world::parsing::load_solar_system_from_dir(&assets)
            .expect("load authored solar system");
        let thalos_id = system.name_to_id["Thalos"];
        let climate = system.bodies[thalos_id]
            .terrestrial_atmosphere
            .as_ref()
            .and_then(|atmosphere| atmosphere.clouds.clone())
            .expect("Thalos authored cloud climate");
        let field = CloudWeatherField::from_climate(&climate);

        // 2°x2° bins over the runway's daylight window (lon near 178°).
        const LAT_MIN: f32 = -45.0;
        const LAT_MAX: f32 = 45.0;
        const LON_MIN: f32 = 150.0;
        const LON_MAX: f32 = 206.0;
        const BIN_DEG: f32 = 2.0;
        let lat_bins = ((LAT_MAX - LAT_MIN) / BIN_DEG) as usize;
        let lon_bins = ((LON_MAX - LON_MIN) / BIN_DEG) as usize;
        #[derive(Clone, Copy, Default)]
        struct Bin {
            n: u32,
            cov: f64,
            sd_col: f64,
            cloudy: u32,
        }
        let mut bins = vec![Bin::default(); lat_bins * lon_bins];
        let size = field.face_size as usize;
        for (face_index, face) in CubemapFace::ALL.into_iter().enumerate() {
            for y in (0..size).step_by(2) {
                let v = (y as f32 + 0.5) / size as f32;
                for x in (0..size).step_by(2) {
                    let u = (x as f32 + 0.5) / size as f32;
                    let dir = face_uv_to_dir(face, u, v).normalize();
                    let lat = dir.y.asin().to_degrees();
                    let lon = dir.z.atan2(dir.x).to_degrees().rem_euclid(360.0);
                    if !(LAT_MIN..LAT_MAX).contains(&lat) || !(LON_MIN..LON_MAX).contains(&lon)
                    {
                        continue;
                    }
                    let bin = &mut bins[((lat - LAT_MIN) / BIN_DEG) as usize * lon_bins
                        + ((lon - LON_MIN) / BIN_DEG) as usize];
                    let index = face_index * size * size + y * size + x;
                    let weather = field.texels[index];
                    let strata = field.surface_density_texels[index];
                    let cov = f64::from(weather[0]) / 255.0;
                    let col = strata
                        .iter()
                        .map(|&s| f64::from(s) / 255.0)
                        .fold(0.0f64, f64::max);
                    bin.n += 1;
                    bin.cov += cov;
                    bin.sd_col += col;
                    bin.cloudy += u32::from(cov > 0.25);
                }
            }
        }

        // Rank by "broken moderate field" suitability: mean column strata near
        // 0.42 with substantial (but not total) cloudy-texel fraction.
        let mut ranked: Vec<(f32, f32, f64, f64, f64)> = Vec::new();
        for (i, bin) in bins.iter().enumerate() {
            if bin.n < 32 {
                continue;
            }
            let lat = LAT_MIN + (i / lon_bins) as f32 * BIN_DEG + BIN_DEG * 0.5;
            let lon = LON_MIN + (i % lon_bins) as f32 * BIN_DEG + BIN_DEG * 0.5;
            let n = f64::from(bin.n);
            ranked.push((lat, lon, bin.cov / n, bin.sd_col / n, f64::from(bin.cloudy) / n));
        }
        ranked.sort_by(|a, b| {
            // Broken moderate field wanted: real cloudy texels, mid strata.
            let score = |r: &(f32, f32, f64, f64, f64)| {
                (r.3 - 0.42).abs() - 0.6 * r.4.min(0.6)
            };
            score(a).total_cmp(&score(b))
        });

        // Local sun elevation at the runway morning boot epoch, so candidates
        // are known-daylit before spending a cold capture on them.
        use thalos_physics_canonical::body_trajectory_provider::BodyTrajectoryProvider;
        let provider =
            thalos_physics_canonical::patched_conics::PatchedConics::new(&system, 3.156e11);
        let states = provider.states(Epoch(59_100.0));
        let star = states.first().map(|s| s.position).unwrap_or_default();
        let thalos_state = &states[thalos_id];
        let sun_elevation_deg = |lat_deg: f32, lon_deg: f32| -> f64 {
            let lat = f64::from(lat_deg).to_radians();
            let lon = f64::from(lon_deg).to_radians();
            let dir_body = bevy::math::DVec3::new(
                lat.cos() * lon.cos(),
                lat.sin(),
                lat.cos() * lon.sin(),
            );
            let up_world = thalos_state.orientation * dir_body;
            let to_sun = (star - (thalos_state.position + up_world * thalos_state.radius_m))
                .normalize();
            90.0 - up_world.angle_between(to_sun).to_degrees()
        };

        println!("lat, lon, mean_cov, mean_col_strata, cloudy_frac, sun_elev_deg");
        for (lat, lon, cov, sd, cloudy) in ranked.iter().take(30) {
            println!(
                "{lat:7.1} {lon:7.1}   {cov:5.3}   {sd:5.3}   {cloudy:5.3}   {:6.1}",
                sun_elevation_deg(*lat, *lon)
            );
        }
        // Reference: the default runway site's bin.
        let default_bin = &bins[((7.6 - LAT_MIN) / BIN_DEG) as usize * lon_bins
            + ((178.0 - LON_MIN) / BIN_DEG) as usize];
        if default_bin.n > 0 {
            let n = f64::from(default_bin.n);
            println!(
                "default site (7.6, 178.0): cov {:5.3} col_strata {:5.3} cloudy_frac {:5.3}",
                default_bin.cov / n,
                default_bin.sd_col / n,
                f64::from(default_bin.cloudy) / n,
            );
        }
    }

    /// Dev probe: run the shared fill derivation on the real Thalos field and
    /// print the fitted curve + far response, without booting a capture.
    /// The per-bin convergence table lands in the log output (init a
    /// subscriber below so it prints).
    ///
    /// Run: `cargo test -p thalos_runtime --lib derive_fill -- --ignored --nocapture`
    #[test]
    #[ignore = "dev probe: prints the derived cloud fill calibration"]
    fn derive_fill_calibration_probe() {
        let subscriber = tracing_subscriber_fmt();
        let _guard = tracing::subscriber::set_default(subscriber);
        let assets =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets");
        let system = thalos_world::parsing::load_solar_system_from_dir(&assets)
            .expect("load authored solar system");
        let thalos_id = system.name_to_id["Thalos"];
        let body = &system.bodies[thalos_id];
        let climate = body
            .terrestrial_atmosphere
            .as_ref()
            .and_then(|atmosphere| atmosphere.clouds.clone())
            .expect("Thalos authored cloud climate");
        let field = CloudWeatherField::from_climate(&climate);
        let start = std::time::Instant::now();
        let calibration = crate::rendering::derive_body_fill_calibration_for_probe(
            &field,
            &climate,
            body.radius_m as f32,
        );
        println!(
            "derived in {:?}: threshold nodes {:?}\nfar_response {:?}",
            start.elapsed(),
            calibration.threshold_nodes,
            calibration.far_response,
        );

        // Cross-check the CPU mirror against the pixel-measured tier A/B at
        // the measurement site (22.0 N, 153.0 E, ~15 km crop).
        let lat = 22.0f32.to_radians();
        let lon = 153.0f32.to_radians();
        let site = Vec3::new(
            lat.cos() * lon.cos(),
            lat.sin(),
            lat.cos() * lon.sin(),
        );
        let climate_bottom = climate.base_altitude_m.max(0.0);
        let input = thalos_body_render::FillCalibrationInput {
            weather_texels: &field.texels,
            strata_texels: &field.surface_density_texels,
            face_size: field.face_size,
            coverage_scale: 1.25,
            density: 0.0026 * climate.density.max(0.0),
            detail_strength: 0.16,
            base_edge_softness: 0.055,
            bottom_softness: 0.16,
            base_shape_scale_m: climate.base_shape_scale_m.max(500.0),
            detail_scale_m: climate.detail_scale_m.max(50.0),
            bottom_height_m: climate_bottom,
            top_height_m: (climate.base_altitude_m + climate.thickness_m)
                .max(climate_bottom + 1.0),
            planet_radius_m: body.radius_m as f32,
            seed: field.seed,
        };
        for radius_km in [8.0f32, 20.0, 60.0] {
            let cos_radius = (radius_km * 1000.0 / body.radius_m as f32).cos();
            let stats = thalos_body_render::fill_lut::predict_region_fill(
                &input,
                &calibration,
                site,
                cos_radius,
                4000,
            );
            println!("site prediction r={radius_km} km: {stats:?}");
        }
    }

    fn tracing_subscriber_fmt() -> impl tracing::Subscriber + Send + Sync {
        use bevy::log::tracing_subscriber::{self, layer::SubscriberExt};
        tracing_subscriber::registry().with(tracing_subscriber::fmt::layer())
    }
}
