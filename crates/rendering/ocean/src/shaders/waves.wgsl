// Shared resolved-wave and filtered-spectrum mechanisms.
//
// Geometry, coast data, and lighting remain adapter-owned. Both the planar
// clipmap and analytic planetary projection call these functions with their
// own local coordinates, footprint gradients, and coastal attenuation.

#define_import_path thalos::ocean_waves

const OCEAN_TAU: f32 = 6.28318530717958;
const OCEAN_CASCADE_DOMAINS_M: array<f32, 4> = array<f32, 4>(
    8192.0, 1024.0, 128.0, 16.0,
);

// Stable forward hit against an analytic ocean sphere. `camera_height_m` is
// computed in f64 on the CPU, so form |camera|² - radius² as h(2r + h) and
// recover the near root through Vieta instead of subtracting two planet-scale
// values. Both the ocean projection and later transparent composites use this
// exact hit; otherwise a pass can draw over water it does not know exists.
fn ocean_sphere_hit_distance_m(
    view_cosine: f32,
    ocean_radius_m: f32,
    camera_height_m: f32,
) -> f32 {
    let camera_radius_m = ocean_radius_m + camera_height_m;
    let c = camera_height_m * (2.0 * ocean_radius_m + camera_height_m);
    let b = camera_radius_m * view_cosine;
    let discriminant = b * b - c;
    if discriminant <= 0.0 {
        return 1.0e30;
    }
    let far_t = -b + sqrt(discriminant);
    if far_t <= 0.0 {
        return 1.0e30;
    }
    let near_t = c / far_t;
    return select(far_t, near_t, near_t > 0.0);
}

struct OceanSurfaceWave {
    height: f32,
    slope: vec2<f32>,
    crest: f32,
    omitted_variance: f32,
}

struct OceanSlopeSample {
    slope: vec2<f32>,
    alpha_ggx: f32,
    breakup: f32,
}

fn ocean_rotate_2d(value: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(c * value.x - s * value.y, s * value.x + c * value.y);
}

fn ocean_wave_visibility(wavelength_m: f32, sample_spacing_m: f32) -> f32 {
    return 1.0 - smoothstep(wavelength_m / 12.0, wavelength_m / 3.0, sample_spacing_m);
}

fn ocean_coastal_wave_scale(
    water_distance_m: f32,
    coastal_range_m: f32,
    exposure: f32,
) -> f32 {
    let range_m = max(coastal_range_m, 116.0);
    let distance_m = max(water_distance_m, 0.0);
    let near_coast = 1.0 - smoothstep(72.0, range_m, distance_m);
    let protection = mix(0.34, 1.0, clamp(exposure, 0.0, 1.0));
    let breaker_band = smoothstep(16.0, 38.0, distance_m)
        * (1.0 - smoothstep(78.0, 116.0, distance_m));
    let shoaling = 1.0 + breaker_band * clamp(exposure, 0.0, 1.0) * 0.32;
    let contact = smoothstep(2.0, 16.0, distance_m);
    return contact * mix(1.0, protection * shoaling, near_coast);
}

fn ocean_wave_component(
    position_m: vec2<f32>,
    direction: vec2<f32>,
    wavelength_m: f32,
    amplitude_m: f32,
    phase_rad: f32,
) -> vec4<f32> {
    let wave_number = OCEAN_TAU / max(wavelength_m, 0.1);
    let phase = wave_number * dot(position_m, direction) - phase_rad;
    let primary = sin(phase);
    let harmonic = sin(phase * 2.0);
    let height = amplitude_m * (primary + 0.16 * harmonic);
    let slope = direction * amplitude_m * wave_number
        * (cos(phase) + 0.32 * cos(phase * 2.0));
    return vec4<f32>(height, slope, primary * 0.5 + 0.5);
}

fn ocean_sample_surface_wave(
    position_m: vec2<f32>,
    sample_spacing_m: f32,
    wind_direction: vec2<f32>,
    wavelengths_m: vec4<f32>,
    amplitudes_m: vec4<f32>,
    phases_rad: vec4<f32>,
    coastal_scale: f32,
) -> OceanSurfaceWave {
    let wind = normalize(wind_direction);
    let swell = ocean_wave_component(
        position_m, wind, wavelengths_m.x, amplitudes_m.x, phases_rad.x,
    );
    let wind_wave = ocean_wave_component(
        position_m, ocean_rotate_2d(wind, 0.29),
        wavelengths_m.y, amplitudes_m.y, phases_rad.y,
    );
    let cross_wave = ocean_wave_component(
        position_m, ocean_rotate_2d(wind, -0.51),
        wavelengths_m.z, amplitudes_m.z, phases_rad.z,
    );
    let swell_visibility = ocean_wave_visibility(wavelengths_m.x, sample_spacing_m);
    let wind_visibility = ocean_wave_visibility(wavelengths_m.y, sample_spacing_m);
    let cross_visibility = ocean_wave_visibility(wavelengths_m.z, sample_spacing_m);

    var result: OceanSurfaceWave;
    result.height = (
        swell.x * swell_visibility
        + wind_wave.x * wind_visibility
        + cross_wave.x * cross_visibility
    ) * coastal_scale;
    result.slope = (
        swell.yz * swell_visibility
        + wind_wave.yz * wind_visibility
        + cross_wave.yz * cross_visibility
    ) * coastal_scale;
    let crest_weight = swell_visibility * 0.58
        + wind_visibility * 0.29
        + cross_visibility * 0.13;
    result.crest = select(
        0.5,
        clamp(
            (swell.w * swell_visibility * 0.58
                + wind_wave.w * wind_visibility * 0.29
                + cross_wave.w * cross_visibility * 0.13)
                / max(crest_weight, 1.0e-4),
            0.0,
            1.0,
        ),
        crest_weight > 1.0e-4,
    );
    let harmonic_variance = 1.0 + 0.32 * 0.32;
    result.omitted_variance = 0.5 * harmonic_variance * coastal_scale * coastal_scale * (
        pow(amplitudes_m.x * OCEAN_TAU / wavelengths_m.x, 2.0)
            * (1.0 - swell_visibility * swell_visibility)
        + pow(amplitudes_m.y * OCEAN_TAU / wavelengths_m.y, 2.0)
            * (1.0 - wind_visibility * wind_visibility)
        + pow(amplitudes_m.z * OCEAN_TAU / wavelengths_m.z, 2.0)
            * (1.0 - cross_visibility * cross_visibility)
    );
    return result;
}

fn ocean_sample_slope_packet(
    slope_texture: texture_2d<f32>,
    slope_sampler: sampler,
    local_m: vec2<f32>,
    local_dx_m: vec2<f32>,
    local_dy_m: vec2<f32>,
    domain_m: f32,
    phase_cycles: f32,
    angle: f32,
    offset: vec2<f32>,
    use_high_packet: bool,
) -> vec3<f32> {
    let position = ocean_rotate_2d(local_m, angle);
    let uv = position / domain_m
        - vec2<f32>(phase_cycles, phase_cycles * 0.11)
        + offset;
    let uv_dx = ocean_rotate_2d(local_dx_m, angle) / domain_m * 0.78;
    let uv_dy = ocean_rotate_2d(local_dy_m, angle) / domain_m * 0.78;
    let texel = textureSampleGrad(
        slope_texture,
        slope_sampler,
        uv,
        uv_dx,
        uv_dy,
    );
    let encoded = select(texel.xy, texel.zw, use_high_packet);
    let rotated_slope = encoded * 2.0 - 1.0;
    let slope = ocean_rotate_2d(rotated_slope, -angle);
    return vec3<f32>(slope, encoded.x * 0.63 + encoded.y * 0.37);
}

fn ocean_sample_slope_cascade(
    slope_texture: texture_2d<f32>,
    slope_sampler: sampler,
    local_m: vec2<f32>,
    local_dx_m: vec2<f32>,
    local_dy_m: vec2<f32>,
    domain_m: f32,
    low_phase: f32,
    high_phase: f32,
    low_angle: f32,
    high_angle: f32,
    offset: vec2<f32>,
    footprint_m: f32,
) -> vec3<f32> {
    let low = ocean_sample_slope_packet(
        slope_texture, slope_sampler, local_m, local_dx_m, local_dy_m,
        domain_m, low_phase, low_angle, offset, false,
    );
    let high = ocean_sample_slope_packet(
        slope_texture, slope_sampler, local_m, local_dx_m, local_dy_m,
        domain_m, high_phase, high_angle, offset + vec2<f32>(0.37, 0.19), true,
    );
    let low_visibility = ocean_wave_visibility(domain_m / 11.489125, footprint_m);
    let high_visibility = ocean_wave_visibility(domain_m / 44.988888, footprint_m);
    return vec3<f32>(
        low.xy * (0.78 * low_visibility) + high.xy * (0.62 * high_visibility),
        low.z * (0.58 * low_visibility) + high.z * (0.42 * high_visibility),
    );
}

fn ocean_cascade_omitted_variance(domain_m: f32, footprint_m: f32) -> f32 {
    let low_visibility = ocean_wave_visibility(domain_m / 11.489125, footprint_m);
    let high_visibility = ocean_wave_visibility(domain_m / 44.988888, footprint_m);
    return 0.58 * (1.0 - low_visibility * low_visibility)
        + 0.42 * (1.0 - high_visibility * high_visibility);
}

fn ocean_sample_slope_field(
    slope_texture: texture_2d<f32>,
    slope_sampler: sampler,
    local_m: vec2<f32>,
    local_dx_m: vec2<f32>,
    local_dy_m: vec2<f32>,
    low_phase: vec4<f32>,
    high_phase: vec4<f32>,
    amplitudes: vec4<f32>,
    swell_angle: f32,
    swell_energy: f32,
) -> OceanSlopeSample {
    let footprint_m = sqrt(max(length(local_dx_m) * length(local_dy_m), 1.0e-6));
    let swell = clamp(swell_energy, 0.0, 1.0);
    let long_wave = ocean_sample_slope_cascade(
        slope_texture, slope_sampler, local_m, local_dx_m, local_dy_m,
        OCEAN_CASCADE_DOMAINS_M[0], low_phase.x, high_phase.x,
        swell_angle * swell, 0.0, vec2<f32>(0.07, 0.31), footprint_m,
    );
    let medium_wave = ocean_sample_slope_cascade(
        slope_texture, slope_sampler, local_m, local_dx_m, local_dy_m,
        OCEAN_CASCADE_DOMAINS_M[1], low_phase.y, high_phase.y,
        0.23 + swell_angle * swell * 0.30, 0.23,
        vec2<f32>(0.43, 0.13), footprint_m,
    );
    let short_wave = ocean_sample_slope_cascade(
        slope_texture, slope_sampler, local_m, local_dx_m, local_dy_m,
        OCEAN_CASCADE_DOMAINS_M[2], low_phase.z, high_phase.z,
        -0.38, -0.38, vec2<f32>(0.19, 0.71), footprint_m,
    );
    let capillary = ocean_sample_slope_cascade(
        slope_texture, slope_sampler, local_m, local_dx_m, local_dy_m,
        OCEAN_CASCADE_DOMAINS_M[3], low_phase.w, high_phase.w,
        0.61, 0.61, vec2<f32>(0.79, 0.47), footprint_m,
    );

    var result: OceanSlopeSample;
    result.slope = long_wave.xy * amplitudes.x
        + medium_wave.xy * amplitudes.y
        + short_wave.xy * amplitudes.z
        + capillary.xy * amplitudes.w;
    let omitted_variance = 0.34 * (
        amplitudes.x * amplitudes.x
            * ocean_cascade_omitted_variance(OCEAN_CASCADE_DOMAINS_M[0], footprint_m)
        + amplitudes.y * amplitudes.y
            * ocean_cascade_omitted_variance(OCEAN_CASCADE_DOMAINS_M[1], footprint_m)
        + amplitudes.z * amplitudes.z
            * ocean_cascade_omitted_variance(OCEAN_CASCADE_DOMAINS_M[2], footprint_m)
        + amplitudes.w * amplitudes.w
            * ocean_cascade_omitted_variance(OCEAN_CASCADE_DOMAINS_M[3], footprint_m)
    );
    result.alpha_ggx = clamp(sqrt(0.0036 + 2.0 * omitted_variance), 0.06, 0.20);
    result.breakup = smoothstep(
        0.30,
        0.72,
        long_wave.z * 0.12
            + medium_wave.z * 0.24
            + short_wave.z * 0.38
            + capillary.z * 0.26,
    );
    return result;
}
