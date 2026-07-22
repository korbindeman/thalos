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
                    // Cellular weight is slightly higher than the first CLOUD-3
                    // checkpoint so open-sky pockets survive into interior and
                    // runway views, but stays below the level that shattered the
                    // limb into cubemap-scale confetti.
                    let coverage = (climate.coverage
                        + climate.band_strength * band
                        + climate.variation
                            * (0.62 * (regional - 0.5)
                                + 0.43 * (mesoscale - 0.5)
                                + 0.32 * (cellular - 0.5)
                                + 0.22 * (frontal - 0.35)))
                        .clamp(0.0, 1.0);

                    // Kind changes at synoptic/mesoscale rather than one type
                    // covering an entire horizon.
                    let selector = fbm3(dir * 42.0 + Vec3::splat(17.0), climate.seed ^ 0xC10D, 3);
                    let cloud_type = if selector < type_mix[0] {
                        0.08
                    } else if selector < type_mix[0] + type_mix[1] {
                        0.50
                    } else {
                        0.94
                    };
                    let vertical_noise = fbm3(
                        dir * 34.0 + Vec3::new(31.0, -7.0, 13.0),
                        climate.seed ^ 0xA11E,
                        3,
                    );
                    // Local base/top are fractions of the authored shell. Storm
                    // towers claim most of the thickness so limb silhouettes
                    // keep height; stratus stays a thin lower deck.
                    let base = match cloud_type {
                        value if value < 0.25 => 0.05 + 0.05 * vertical_noise,
                        value if value < 0.75 => 0.02 + 0.07 * vertical_noise,
                        _ => 0.01 + 0.04 * vertical_noise,
                    };
                    let top = match cloud_type {
                        value if value < 0.25 => 0.16 + 0.12 * vertical_noise,
                        value if value < 0.75 => 0.34 + 0.38 * vertical_noise,
                        _ => 0.78 + 0.20 * vertical_noise,
                    };
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
                        0.48 * shape_mass + 0.25 * shape_cut + 0.17 * selector + 0.10 * cellular,
                        0.39 * shape_mass
                            + 0.23 * shape_cut
                            + 0.21 * vertical_noise
                            + 0.17 * selector,
                        0.31 * shape_mass
                            + 0.20 * shape_cut
                            + 0.28 * vertical_noise
                            + 0.21 * selector,
                    ];
                    let surface_density = [0.125, 0.375, 0.625, 0.875].map(|height| {
                        cloud_surface_density_cpu(
                            surface_shape,
                            height,
                            coverage,
                            cloud_type,
                            base,
                            top.max(base + 0.02),
                        )
                    });
                    let encode = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    texels.push([
                        encode(coverage),
                        encode(cloud_type),
                        encode(base),
                        encode(top.max(base + 0.02)),
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
    let vertical_narrow = h * (0.04 * stratus_w + 0.19 * cumulus_w + 0.09 * storm_w);

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
    let stratus_profile =
        smoothstep(0.0, bottom_softness * 0.45, h) * (1.0 - smoothstep(0.72, 1.0, h));
    let cumulus_profile =
        smoothstep(0.0, bottom_softness * 0.75, h) * (1.0 - smoothstep(0.70, 1.0, h));
    let storm_profile =
        smoothstep(0.0, bottom_softness * 0.35, h) * (1.0 - smoothstep(0.88, 1.0, h));
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
