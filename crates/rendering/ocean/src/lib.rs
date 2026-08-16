//! Shared ocean-wave projection, shader mechanisms, and deterministic payloads.
//!
//! This module is independent of ocean geometry. Thalos's spherical analytic
//! ocean and Kòrsou's planar clipmap both consume the same authored sea state
//! and packed low/high-frequency directional packets. Geometry adapters remain
//! free to choose how they displace or shade their surface.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor};
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
pub use thalos_render_model::RenderFrameTime;
use thalos_world::OceanState;

const SLOPE_TEXTURE_SIZE: usize = 256;
const SPECTRUM_MODE_COUNT: usize = 128;
const LOW_PACKET_MIN_CYCLES: f32 = 6.0;
const LOW_PACKET_MAX_CYCLES: f32 = 22.0;
const HIGH_PACKET_MIN_CYCLES: f32 = 22.0;
const HIGH_PACKET_MAX_CYCLES: f32 = 92.0;
const TARGET_RMS_SLOPE: f32 = 0.58;

/// Physical domains sampled by `body_sky.wgsl`, longest to shortest.
/// Keep the shader's `OCEAN_CASCADE_DOMAINS_M` mirror in sync.
pub const OCEAN_CASCADE_DOMAINS_M: [f64; 4] = [8_192.0, 1_024.0, 128.0, 16.0];

const REFERENCE_WIND_SPEED_M_S: f32 = 11.0;
const REFERENCE_WAVE_HEIGHT_M: f32 = 2.4;
const REFERENCE_WAVELENGTH_M: f32 = 150.0;
const REFERENCE_SWELL_ENERGY: f32 = 0.34;
const BASE_SLOPE_AMPLITUDES: [f32; 4] = [0.018, 0.050, 0.105, 0.100];
const REFERENCE_SURFACE_WAVELENGTHS_M: [f32; 3] = [54.0, 18.0, 8.5];
const REFERENCE_SURFACE_AMPLITUDES_M: [f32; 3] = [0.42, 0.16, 0.055];
const SURFACE_PHASE_OFFSETS_RAD: [f64; 3] = [0.3, 2.1, 4.7];

/// Per-frame GPU projection of one authored [`OceanState`].
#[derive(Debug, Clone, Copy)]
pub struct OceanSpectrumProjection {
    /// Low-frequency packet phase in texture cycles for each physical cascade.
    pub low_phase: Vec4,
    /// High-frequency packet phase in texture cycles for each physical cascade.
    pub high_phase: Vec4,
    /// Calibrated resolved-slope amplitudes, longest to shortest cascade.
    pub slope_amplitudes: Vec4,
}

/// Precision-safe projection of the three resolved surface-wave harmonics.
///
/// The same height/slope/crest phases feed Kòrsou's displaced planar clipmap
/// and Thalos's analytic planetary water shading. Geometry adapters decide
/// whether height changes vertices; they do not invent another wave clock.
#[derive(Debug, Clone, Copy)]
pub struct OceanSurfaceProjection {
    /// Wavelengths in metres; xyz are active and w is reserved.
    pub wavelengths_m: Vec4,
    /// Wave amplitudes in metres; xyz are active and w is reserved.
    pub amplitudes_m: Vec4,
    /// Current angular phases in radians, reduced in f64 before upload.
    pub phases_rad: Vec4,
}

/// One canonical ocean-mechanism projection for a validated render frame.
///
/// The current spectrum drives shading in both adapters. Current and previous
/// surface projections also give displaced adapters motion-vector history
/// without making the application repeat ocean phase reduction.
#[derive(Debug, Clone, Copy)]
pub struct OceanFrameProjection {
    pub spectrum: OceanSpectrumProjection,
    pub current_surface: OceanSurfaceProjection,
    pub previous_surface: OceanSurfaceProjection,
}

/// Registers the shared `thalos::ocean_waves` WGSL library.
pub struct OceanMechanismPlugin;

impl Plugin for OceanMechanismPlugin {
    fn build(&self, app: &mut App) {
        bevy::shader::load_shader_library!(app, "shaders/waves.wgsl");
    }
}

/// Camera-local, body-fixed tangent frame for the periodic slope field.
#[derive(Debug, Clone, Copy)]
pub struct OceanWaveFrame {
    pub camera_phase_m: Vec4,
    pub wind_basis: Vec4,
    pub crosswind_basis: Vec4,
    /// Signed tangent-plane angle from wind to independent swell.
    pub swell_angle_rad: f32,
}

#[derive(Clone, Copy)]
struct SpectrumMode {
    wave_vector: Vec2,
    phase: f32,
    weight: f32,
}

/// Bake the shared periodic slope field used by every atmospheric ocean.
///
/// The two packed packets let each cascade evolve at two dispersion rates
/// instead of translating a broadband field as one rigid sheet. Integer wave
/// vectors make the base texture exactly tileable; every mip is authored on
/// the CPU because Bevy uploads `Image` payloads without generating mipmaps.
pub fn bake_ocean_slope_texture() -> Image {
    let low_packet = synthesize_slope_field(
        0x8ad4_16e9,
        LOW_PACKET_MIN_CYCLES,
        LOW_PACKET_MAX_CYCLES,
        0.42,
        0.82,
    );
    let high_packet = synthesize_slope_field(
        0x51f2_c73b,
        HIGH_PACKET_MIN_CYCLES,
        HIGH_PACKET_MAX_CYCLES,
        0.82,
        1.32,
    );

    let mut level = Vec::with_capacity(SLOPE_TEXTURE_SIZE * SLOPE_TEXTURE_SIZE * 4);
    for (low, high) in low_packet.into_iter().zip(high_packet) {
        level.extend_from_slice(&[
            encode_slope(low.x),
            encode_slope(low.y),
            encode_slope(high.x),
            encode_slope(high.y),
        ]);
    }

    let mip_count = SLOPE_TEXTURE_SIZE.ilog2() + 1;
    let mut data = Vec::with_capacity(level.len() * 4 / 3 + 4);
    data.extend_from_slice(&level);
    let mut size = SLOPE_TEXTURE_SIZE;
    while size > 1 {
        let next_size = size / 2;
        let mut downsampled = Vec::with_capacity(next_size * next_size * 4);
        for y in 0..next_size {
            for x in 0..next_size {
                for channel in 0..4 {
                    let sample = |sx: usize, sy: usize| {
                        level[((2 * y + sy) * size + 2 * x + sx) * 4 + channel] as u32
                    };
                    let average =
                        (sample(0, 0) + sample(1, 0) + sample(0, 1) + sample(1, 1) + 2) / 4;
                    downsampled.push(average as u8);
                }
            }
        }
        data.extend_from_slice(&downsampled);
        level = downsampled;
        size = next_size;
    }

    // `Image::new` validates against a base-level-only byte count. A manually
    // supplied mip chain therefore has to use `new_uninit`, as the coast atlas
    // does elsewhere in this crate.
    let mut image = Image::new_uninit(
        Extent3d {
            width: SLOPE_TEXTURE_SIZE as u32,
            height: SLOPE_TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = mip_count;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        anisotropy_clamp: 16,
        ..ImageSamplerDescriptor::linear()
    });
    image
}

/// Project one validated application frame into the complete shared ocean
/// payload. All wave clocks remain f64 until mechanism-specific phase reduction.
pub fn project_ocean_frame(
    state: &OceanState,
    gravity_m_s2: f64,
    time: RenderFrameTime,
) -> OceanFrameProjection {
    OceanFrameProjection {
        spectrum: project_ocean_spectrum(state, gravity_m_s2, time.current_epoch_s()),
        current_surface: project_ocean_surface(state, gravity_m_s2, time.current_epoch_s()),
        previous_surface: project_ocean_surface(state, gravity_m_s2, time.previous_epoch_s()),
    }
}

/// Project stable body state and canonical simulation time into precision-safe
/// dispersive packet phases. The CPU performs the f64 modulo before upload, so
/// very large epochs never collapse metre-scale motion in an f32 shader.
fn project_ocean_spectrum(
    state: &OceanState,
    gravity_m_s2: f64,
    simulation_time_s: f64,
) -> OceanSpectrumProjection {
    let low_cycles = f64::from((LOW_PACKET_MIN_CYCLES * LOW_PACKET_MAX_CYCLES).sqrt());
    let high_cycles = f64::from((HIGH_PACKET_MIN_CYCLES * HIGH_PACKET_MAX_CYCLES).sqrt());
    let gravity = gravity_m_s2.max(0.1);

    let low_phase = Vec4::from_array(
        OCEAN_CASCADE_DOMAINS_M
            .map(|domain_m| packet_phase_cycles(simulation_time_s, gravity, domain_m, low_cycles)),
    );
    let high_phase =
        Vec4::from_array(OCEAN_CASCADE_DOMAINS_M.map(|domain_m| {
            packet_phase_cycles(simulation_time_s, gravity, domain_m, high_cycles)
        }));

    // BL-12's accepted calibration is the exact reference point. Physical
    // authoring controls move energy smoothly around it without claiming this
    // two-packet tracer is already a full JONSWAP integration.
    let height_scale = (state.significant_wave_height_m.max(0.05) / REFERENCE_WAVE_HEIGHT_M).sqrt();
    let wind_scale = (state.wind_speed_10m_m_s.max(0.1) / REFERENCE_WIND_SPEED_M_S)
        .sqrt()
        .clamp(0.35, 1.8);
    let wavelength_scale =
        (state.dominant_wavelength_m.max(1.0) / REFERENCE_WAVELENGTH_M).clamp(0.4, 2.5);
    let swell_delta = state.swell_energy.clamp(0.0, 1.0) - REFERENCE_SWELL_ENERGY;
    let slope_amplitudes = Vec4::new(
        BASE_SLOPE_AMPLITUDES[0]
            * height_scale
            * wavelength_scale.powf(0.35)
            * (1.0 + 0.80 * swell_delta),
        BASE_SLOPE_AMPLITUDES[1]
            * height_scale
            * wavelength_scale.powf(0.15)
            * (1.0 + 0.25 * swell_delta),
        BASE_SLOPE_AMPLITUDES[2] * height_scale * wind_scale * wavelength_scale.powf(-0.12),
        BASE_SLOPE_AMPLITUDES[3] * height_scale.sqrt() * wind_scale,
    );

    OceanSpectrumProjection {
        low_phase,
        high_phase,
        slope_amplitudes,
    }
}

/// Project authored sea state into resolved wave geometry/shading parameters.
///
/// Wavelength and amplitude calibration is exact at [`OceanState::MODERATE`].
/// Phase is derived from the canonical clock with deep-water dispersion and
/// reduced before f32 upload, so large simulation epochs remain stable.
fn project_ocean_surface(
    state: &OceanState,
    gravity_m_s2: f64,
    simulation_time_s: f64,
) -> OceanSurfaceProjection {
    let wavelength_scale =
        (state.dominant_wavelength_m.max(1.0) / REFERENCE_WAVELENGTH_M).clamp(0.4, 2.5);
    let amplitude_scale =
        (state.significant_wave_height_m.max(0.05) / REFERENCE_WAVE_HEIGHT_M).clamp(0.1, 4.0);
    let wavelengths = REFERENCE_SURFACE_WAVELENGTHS_M.map(|value| value * wavelength_scale);
    let amplitudes = REFERENCE_SURFACE_AMPLITUDES_M.map(|value| value * amplitude_scale);
    let gravity = gravity_m_s2.max(0.1);
    let phases = std::array::from_fn::<_, 3, _>(|index| {
        let wave_number = std::f64::consts::TAU / f64::from(wavelengths[index]);
        let angular_speed = (gravity * wave_number).sqrt();
        (angular_speed * simulation_time_s + SURFACE_PHASE_OFFSETS_RAD[index])
            .rem_euclid(std::f64::consts::TAU) as f32
    });

    OceanSurfaceProjection {
        wavelengths_m: Vec4::new(wavelengths[0], wavelengths[1], wavelengths[2], 0.0),
        amplitudes_m: Vec4::new(amplitudes[0], amplitudes[1], amplitudes[2], 0.0),
        phases_rad: Vec4::new(phases[0], phases[1], phases[2], 0.0),
    }
}

/// Build the stable tangent frame and high-precision camera phase for an
/// authored ocean. Every physical cascade divides the common 8192 m period.
pub fn ocean_wave_frame(camera_body_m: DVec3, state: &OceanState) -> OceanWaveFrame {
    let up = camera_body_m.normalize_or_zero();
    let wind = tangent_axis(up, tuple_dvec3(state.wind_axis_body));
    let crosswind = up.cross(wind).normalize_or_zero();
    let swell = tangent_axis(up, tuple_dvec3(state.swell_axis_body));
    let swell_angle_rad = swell.dot(crosswind).atan2(swell.dot(wind)) as f32;
    let common_period_m = OCEAN_CASCADE_DOMAINS_M[0];

    OceanWaveFrame {
        camera_phase_m: Vec4::new(
            camera_body_m.dot(wind).rem_euclid(common_period_m) as f32,
            camera_body_m.dot(crosswind).rem_euclid(common_period_m) as f32,
            0.0,
            0.0,
        ),
        wind_basis: wind.as_vec3().extend(0.0),
        crosswind_basis: crosswind.as_vec3().extend(0.0),
        swell_angle_rad,
    }
}

/// Representative deep-water phase speeds for the two packets in each
/// cascade. Used by runtime diagnostics and GPU-budget/fidelity inspection.
pub fn ocean_packet_phase_speeds(gravity_m_s2: f64) -> ([f32; 4], [f32; 4]) {
    let gravity = gravity_m_s2.max(0.1);
    let low_cycles = f64::from((LOW_PACKET_MIN_CYCLES * LOW_PACKET_MAX_CYCLES).sqrt());
    let high_cycles = f64::from((HIGH_PACKET_MIN_CYCLES * HIGH_PACKET_MAX_CYCLES).sqrt());
    (
        OCEAN_CASCADE_DOMAINS_M
            .map(|domain| deep_water_phase_speed(gravity, domain / low_cycles) as f32),
        OCEAN_CASCADE_DOMAINS_M
            .map(|domain| deep_water_phase_speed(gravity, domain / high_cycles) as f32),
    )
}

fn packet_phase_cycles(time_s: f64, gravity_m_s2: f64, domain_m: f64, cycles: f64) -> f32 {
    let wavelength_m = domain_m / cycles;
    let speed_m_s = deep_water_phase_speed(gravity_m_s2, wavelength_m);
    (time_s * speed_m_s / domain_m).rem_euclid(1.0) as f32
}

fn deep_water_phase_speed(gravity_m_s2: f64, wavelength_m: f64) -> f64 {
    (gravity_m_s2 * wavelength_m / std::f64::consts::TAU).sqrt()
}

fn tangent_axis(up: DVec3, authored_axis: DVec3) -> DVec3 {
    let axis = authored_axis.normalize_or(DVec3::X);
    let mut tangent = axis - up * axis.dot(up);
    if tangent.length_squared() < 1.0e-8 {
        tangent = DVec3::X - up * up.x;
    }
    tangent.normalize_or(DVec3::Z)
}

fn tuple_dvec3(value: (f32, f32, f32)) -> DVec3 {
    DVec3::new(f64::from(value.0), f64::from(value.1), f64::from(value.2))
}

fn synthesize_slope_field(
    seed: u32,
    min_cycles: f32,
    max_cycles: f32,
    min_spread: f32,
    max_spread: f32,
) -> Vec<Vec2> {
    let modes = spectrum_modes(seed, min_cycles, max_cycles, min_spread, max_spread);
    let mut field = Vec::with_capacity(SLOPE_TEXTURE_SIZE * SLOPE_TEXTURE_SIZE);
    let tau = std::f32::consts::TAU;
    for y in 0..SLOPE_TEXTURE_SIZE {
        for x in 0..SLOPE_TEXTURE_SIZE {
            let uv = Vec2::new(
                (x as f32 + 0.5) / SLOPE_TEXTURE_SIZE as f32,
                (y as f32 + 0.5) / SLOPE_TEXTURE_SIZE as f32,
            );
            let mut slope = Vec2::ZERO;
            for mode in &modes {
                let carrier = (tau * mode.wave_vector.dot(uv) + mode.phase).cos();
                slope += mode.wave_vector.normalize() * (mode.weight * carrier);
            }
            field.push(slope);
        }
    }

    let rms = (field
        .iter()
        .map(|slope| slope.length_squared())
        .sum::<f32>()
        / field.len() as f32)
        .sqrt();
    let scale = TARGET_RMS_SLOPE / rms.max(1.0e-6);
    for slope in &mut field {
        *slope = (*slope * scale).clamp(Vec2::splat(-1.0), Vec2::splat(1.0));
    }
    field
}

fn spectrum_modes(
    seed: u32,
    min_cycles: f32,
    max_cycles: f32,
    min_spread: f32,
    max_spread: f32,
) -> Vec<SpectrumMode> {
    #[derive(Clone, Copy)]
    struct Candidate {
        wave_vector: Vec2,
        rank: f32,
        key: u32,
    }

    // Enumerate the finite lattice once, then choose the unused carrier nearest
    // each log-frequency stratum. The old rejection loop could become
    // impossible after rounding several nearby low-frequency strata onto the
    // same small set of integer vectors; this construction is bounded and
    // guarantees uniqueness whenever the authored annulus has enough carriers.
    let limit = max_cycles.ceil() as i32;
    let log_range = (max_cycles / min_cycles).ln();
    let mut candidates = Vec::new();
    for kx in 1..=limit {
        for ky in -limit..=limit {
            let wave_vector = Vec2::new(kx as f32, ky as f32);
            let cycles = wave_vector.length();
            if cycles < min_cycles || cycles > max_cycles {
                continue;
            }
            let rank = (cycles / min_cycles).ln() / log_range;
            let spread = min_spread + (max_spread - min_spread) * rank.sqrt();
            if wave_vector.y.atan2(wave_vector.x).abs() > spread {
                continue;
            }
            let lattice_key =
                (kx as u32).wrapping_mul(0x9e37_79b9) ^ (ky as u32).wrapping_mul(0x85eb_ca6b);
            candidates.push(Candidate {
                wave_vector,
                rank,
                key: hash_u32(seed ^ lattice_key),
            });
        }
    }
    assert!(
        candidates.len() >= SPECTRUM_MODE_COUNT,
        "ocean packet annulus has {} unique carriers, needs {}",
        candidates.len(),
        SPECTRUM_MODE_COUNT,
    );

    let mut modes = Vec::with_capacity(SPECTRUM_MODE_COUNT);
    for index in 0..SPECTRUM_MODE_COUNT {
        let target_rank = (index as f32 + 0.5) / SPECTRUM_MODE_COUNT as f32;
        let (candidate_index, candidate) = candidates
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let score_a = (a.rank - target_rank).abs() + hash01(a.key) * 0.002;
                let score_b = (b.rank - target_rank).abs() + hash01(b.key) * 0.002;
                score_a.total_cmp(&score_b).then_with(|| a.key.cmp(&b.key))
            })
            .expect("carrier count checked above");
        let candidate = *candidate;
        candidates.swap_remove(candidate_index);
        modes.push(SpectrumMode {
            wave_vector: candidate.wave_vector,
            phase: std::f32::consts::TAU
                * hash01(seed.rotate_right(7) ^ candidate.key.wrapping_mul(0xc2b2_ae35)),
            // A gentle high-frequency rolloff avoids a white-noise normal
            // while retaining enough capillary energy for close glitter.
            weight: 0.82 + 0.38 * candidate.rank,
        });
    }
    modes
}

fn encode_slope(value: f32) -> u8 {
    ((value.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0 + 0.5) as u8
}

fn hash01(value: u32) -> f32 {
    hash_u32(value) as f32 / u32::MAX as f32
}

fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn packet_carriers_are_unique_and_fully_populated() {
        for (seed, min, max, spread_min, spread_max) in [
            (0x8ad4_16e9, 6.0, 22.0, 0.42, 0.82),
            (0x51f2_c73b, 22.0, 92.0, 0.82, 1.32),
        ] {
            let modes = spectrum_modes(seed, min, max, spread_min, spread_max);
            let unique: HashSet<_> = modes
                .iter()
                .map(|mode| (mode.wave_vector.x as i32, mode.wave_vector.y as i32))
                .collect();
            assert_eq!(modes.len(), SPECTRUM_MODE_COUNT);
            assert_eq!(unique.len(), SPECTRUM_MODE_COUNT);
        }
    }

    #[test]
    fn moderate_state_preserves_the_accepted_slope_calibration() {
        let projection = project_ocean_spectrum(&OceanState::MODERATE, 9.06, 0.0);
        let expected = Vec4::from_array(BASE_SLOPE_AMPLITUDES);
        assert!(projection.slope_amplitudes.abs_diff_eq(expected, 1.0e-6));
    }

    #[test]
    fn moderate_state_preserves_resolved_surface_calibration() {
        let projection = project_ocean_surface(&OceanState::MODERATE, 9.81, 0.0);
        assert_eq!(projection.wavelengths_m, Vec4::new(54.0, 18.0, 8.5, 0.0));
        assert_eq!(projection.amplitudes_m, Vec4::new(0.42, 0.16, 0.055, 0.0));
        for phase in projection.phases_rad.to_array().into_iter().take(3) {
            assert!(phase.is_finite() && (0.0..std::f32::consts::TAU).contains(&phase));
        }
    }

    #[test]
    fn frame_projection_preserves_previous_current_order() {
        let time = RenderFrameTime::new(120.0, 120.25).unwrap();
        let projection = project_ocean_frame(&OceanState::MODERATE, 9.81, time);
        let expected_previous = project_ocean_surface(&OceanState::MODERATE, 9.81, 120.0);
        let expected_current = project_ocean_surface(&OceanState::MODERATE, 9.81, 120.25);

        assert_eq!(
            projection.previous_surface.phases_rad,
            expected_previous.phases_rad
        );
        assert_eq!(
            projection.current_surface.phases_rad,
            expected_current.phases_rad
        );
        assert_ne!(
            projection.previous_surface.phases_rad,
            projection.current_surface.phases_rad
        );
    }

    #[test]
    fn resolved_surface_phases_stay_finite_at_large_epochs() {
        let projection = project_ocean_surface(&OceanState::MODERATE, 9.81, 1.0e12);
        for phase in projection.phases_rad.to_array().into_iter().take(3) {
            assert!(phase.is_finite() && (0.0..std::f32::consts::TAU).contains(&phase));
        }
    }

    #[test]
    fn large_epochs_reduce_to_finite_packet_phases() {
        let projection = project_ocean_spectrum(&OceanState::MODERATE, 9.06, 1.0e12);
        for phase in projection
            .low_phase
            .to_array()
            .into_iter()
            .chain(projection.high_phase.to_array())
        {
            assert!(phase.is_finite());
            assert!((0.0..1.0).contains(&phase));
        }
    }
}
