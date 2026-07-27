// Reusable atmosphere primitives for any body impostor.
//
// This library exports the rim-halo, limb-shading, and limb-darkening
// helpers that were previously inlined into `gas_giant.wgsl`. Terrestrial
// impostors (`planet_impostor.wgsl`) and the gas-giant disk both pull
// from here so the two render paths share one source of truth.
//
// The Rust mirror of `AtmosphereBlock` lives in
// `crates/planet_rendering/src/material.rs`. Field order, widths, and
// padding MUST match across both sides.

#define_import_path thalos::atmosphere

const PI: f32 = 3.14159265358979323846;

/// Packed uniform carrying the terrestrial atmosphere parameters.
///
/// The dominant block is the single-scattering Rayleigh + Mie set
/// consumed by `integrate_atmosphere`; cloud and limb-darkening fields
/// are kept independent because they are not produced by the
/// scattering integral. Every gate scalar is zero by default — bodies
/// without an atmosphere (Mira, Ignis, …) pay only the early-out
/// cost.
struct AtmosphereBlock {
    /// Rayleigh sea-level coefficient β_R, per render unit.
    ///   xyz = R/G/B (Earth ≈ vec3(0.046,0.108,0.264) / H_R, scaled
    ///         from per-meter to per-render-unit on the CPU),
    ///   w   = Rayleigh scale height H_R in render units.
    rayleigh_beta_h: vec4<f32>,
    /// Mie scattering parameters.
    ///   xyz = Mie sea-level coefficient β_M (per render unit; usually
    ///         spectrally white so x=y=z),
    ///   w   = Henyey-Greenstein asymmetry g ∈ [-1, 1] (Earth aerosols
    ///         ≈ 0.76 forward-peaked).
    mie_beta_g: vec4<f32>,
    /// Atmosphere geometry + global gates.
    ///   x = atmosphere top altitude above the surface (render units;
    ///       view raymarch terminates here),
    ///   y = Mie scale height H_M (render units),
    ///   z = strength multiplier (artistic; 0 disables the entire
    ///       scattering raymarch),
    ///   w = multiple-scattering gain (artistic horizon blue-fill; scales
    ///       the multi-scatter LUT term on top of `strength`, ground
    ///       multiscatter pass only; 1.0 = bare approximation).
    atmos_geom: vec4<f32>,
    /// Per-channel Minnaert limb darkening.
    ///   xyz = R/G/B exponents (typical 0.2–0.45),
    ///   w   = overall strength in [0, 1]. `0` disables darkening.
    limb_exponents: vec4<f32>,
    /// Cloud layer colour and coverage.
    ///   xyz = sunlit-cloud linear-RGB albedo,
    ///   w   = coverage fraction in [0, 1]. `0` disables the cloud
    ///         layer entirely.
    cloud_albedo_coverage: vec4<f32>,
    /// Cloud layer shape parameters.
    ///   x = layer base altitude above the surface (render units),
    ///   y = layer thickness (render units) — volumetric slab depth,
    ///   z = optical-density multiplier (scales raymarch extinction),
    ///   w = differential rotation coefficient in [0, 1].
    cloud_shape: vec4<f32>,
    /// Cloud layer dynamics.
    ///   x = equatorial scroll rate (radians / second of sim time),
    ///   y = current sim time (seconds, wrapped to a day-scale
    ///       modulus so f32 precision stays tight),
    ///   zw = reserved for future cloud dynamics controls.
    cloud_dynamics: vec4<f32>,
    /// Cloud main-deck band phases 0..=3. 16 phases total spread across
    /// four vec4s — `cloud_band_phase()` unpacks by index. Each band
    /// carries a rigidly-wrapped rotation angle (mod TAU) so
    /// `sample_cloud_banded()` can sample the cloud cubemap at two
    /// adjacent bands and blend, yielding seamless differential
    /// rotation at every latitude. See material.rs `CLOUD_BAND_COUNT`.
    cloud_bands_a: vec4<f32>,
    /// Cloud main-deck band phases 4..=7.
    cloud_bands_b: vec4<f32>,
    /// Cloud main-deck band phases 8..=11.
    cloud_bands_c: vec4<f32>,
    /// Cloud main-deck band phases 12..=15.
    cloud_bands_d: vec4<f32>,
}

const CLOUD_BAND_COUNT: u32 = 16u;

/// Fetch band `i`'s rotation phase (radians, wrapped to `[0, TAU)`).
/// 16 phases packed into four vec4s; this helper is the unpack.
fn cloud_band_phase(i: u32, layers: AtmosphereBlock) -> f32 {
    let clamped = min(i, CLOUD_BAND_COUNT - 1u);
    let vec_idx = clamped / 4u;
    let comp_idx = clamped % 4u;
    var v: vec4<f32>;
    if vec_idx == 0u {
        v = layers.cloud_bands_a;
    } else if vec_idx == 1u {
        v = layers.cloud_bands_b;
    } else if vec_idx == 2u {
        v = layers.cloud_bands_c;
    } else {
        v = layers.cloud_bands_d;
    }
    if comp_idx == 0u { return v.x; }
    if comp_idx == 1u { return v.y; }
    if comp_idx == 2u { return v.z; }
    return v.w;
}

/// Rotate `dir` around the body-local +Y axis by `phase` radians. Used
/// by the banded cloud sampler to build per-band sample directions.
fn rotate_around_y(dir: vec3<f32>, phase: f32) -> vec3<f32> {
    let cp = cos(phase);
    let sp = sin(phase);
    return vec3<f32>(
        dir.x * cp - dir.z * sp,
        dir.y,
        dir.x * sp + dir.z * cp,
    );
}

/// Apply per-channel Minnaert limb darkening. Strength 0 returns
/// `base` unchanged. Exponents are per-channel; typical terrestrial
/// values are near 1.0 (barely any effect); gas giants sit around 0.2–0.45.
fn apply_limb_darkening(
    base: vec3<f32>,
    n_dot_v: f32,
    exponents: vec3<f32>,
    strength: f32,
) -> vec3<f32> {
    if strength <= 0.0 {
        return base;
    }
    let nv = max(n_dot_v, 0.0);
    let mr = pow(nv, max(exponents.x, 1e-3));
    let mg = pow(nv, max(exponents.y, 1e-3));
    let mb = pow(nv, max(exponents.z, 1e-3));
    let darkened = base * vec3<f32>(mr, mg, mb);
    return mix(base, darkened, clamp(strength, 0.0, 1.0));
}

// ── Single-scattering Rayleigh + Mie atmosphere ─────────────────────────────
//
// One numeric raymarch produces both the in-scattered radiance along a
// view ray and the per-channel transmittance from the camera entry
// point to the ray exit (surface hit on body-pass, atmosphere shell
// exit on miss-pass). The same path delivers:
//
//   - the rim halo outside the silhouette (miss rays integrate the
//     full atmosphere chord, accumulating Rayleigh in-scatter along
//     it);
//   - the daylight haze across the lit disk (small but non-zero
//     in-scatter on the sub-solar surface);
//   - the orange/red sunset band at the terminator (long sun column
//     eats blue from `T_sun`, leaving the red residue scattering);
//   - the surface aerial perspective (transmittance dims and tints
//     the lit ground);
//   - the soft warm "wrap" around the terminator (Mie's forward-peaked
//     phase function brightens haze where the view ray is aligned
//     with the sun direction).
//
// All the legacy stand-in helpers (`apply_rayleigh_ground_transmission`,
// `apply_rayleigh_inscatter`, `apply_terminator_warmth`,
// `apply_fresnel_rim`, `rim_halo_contribution`) are gone — their
// approximations are now produced as natural consequences of this
// physical integral.
//
// Reference: Sébastien Hillaire's "A Scalable and Production-Ready
// Sky and Atmosphere Rendering Technique" (2020) for the integration
// scheme; Bucholtz 1995 for Rayleigh β; Henyey-Greenstein 1941 for
// the Mie phase function. Single-scattering only — multi-scatter
// matters in-atmosphere but is below visual threshold for orbital
// impostors. Ozone absorption (the source of Earth's blue twilight)
// is a queued M4 follow-up: cheap per-sample multiplier on transmittance,
// adds two parameters; defer until the visual budget calls for it.

const ATMOS_VIEW_STEPS: u32 = 16u;
const ATMOS_SUN_STEPS: u32 = 8u;
// Minimum view samples for very short paths. The view-step count is scaled by
// path length (see `adaptive_view_steps`): a full atmosphere-shell crossing
// uses the full `ATMOS_VIEW_STEPS`, but a ground pixel — camera near the
// surface looking at nearby terrain, the bulk of the screen on the surface —
// has a path of a few metres and only needs a couple of samples. Without this
// scaling those pixels paid 8 view steps × a 6-step sun column each for a
// near-zero in-scatter contribution.
const ATMOS_VIEW_STEPS_MIN: u32 = 4u;

/// View-sample count for a raymarch segment of length `path`, scaled so the
/// per-metre sample density matches a full-shell crossing at `ATMOS_VIEW_STEPS`
/// and clamped to `[ATMOS_VIEW_STEPS_MIN, ATMOS_VIEW_STEPS]`. `shell_alt` is the
/// atmosphere thickness (`atmos_geom.x`). Long sky paths keep full quality;
/// short ground paths collapse to the minimum.
fn adaptive_view_steps(path: f32, shell_alt: f32) -> u32 {
    let step_len = max(shell_alt, 1.0) / f32(ATMOS_VIEW_STEPS);
    let want = u32(ceil(path / max(step_len, 1e-3)));
    return clamp(want, ATMOS_VIEW_STEPS_MIN, ATMOS_VIEW_STEPS);
}

struct ScatterResult {
    /// In-scattered radiance accumulated along `[t_enter, t_exit]`.
    /// Already pre-multiplied by sun flux, β coefficients, phase
    /// functions, and the artistic `strength` knob — caller adds it
    /// to the surface colour without further scaling.
    in_scatter: vec3<f32>,
    /// Per-channel transmittance through `[t_enter, t_exit]`. Multiply
    /// the surface colour by this to get aerial perspective. `vec3(1)`
    /// = vacuum (no extinction).
    transmittance: vec3<f32>,
}

fn no_scatter() -> ScatterResult {
    return ScatterResult(vec3<f32>(0.0), vec3<f32>(1.0));
}

/// True when the body has an active scattering atmosphere. Caller
/// uses this to skip the raymarch entirely on airless / vacuum bodies.
fn atmosphere_scattering_active(layers: AtmosphereBlock) -> bool {
    let beta_total = layers.rayleigh_beta_h.x + layers.rayleigh_beta_h.y
        + layers.rayleigh_beta_h.z + layers.mie_beta_g.x;
    return layers.atmos_geom.z > 0.0
        && layers.atmos_geom.x > 0.0
        && beta_total > 0.0;
}

/// Rayleigh phase function: P_R(θ) = 3/(16π) · (1 + cos²θ).
///
/// Symmetric in cos θ → equal forward and backward scatter, with a
/// dip at 90°. This is what produces the "blue everywhere on the lit
/// disk, slightly brighter near and away from the sun" Rayleigh look.
fn phase_rayleigh(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

/// Henyey-Greenstein Mie phase: 1/(4π) · (1−g²)/(1+g²−2g·cosθ)^(3/2).
///
/// `g` controls anisotropy: positive = forward-peaked (Earth aerosols
/// ≈ 0.76 — sun-side haze whitens noticeably); 0 = isotropic; negative
/// = back-peaked (rare). At g ≈ 0.76 the forward peak is ~30× the
/// isotropic floor — this is what gives the lit limb its desaturated
/// glow and the haze-near-sun brightening.
fn phase_mie_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * cos_theta, 1e-6);
    return (1.0 / (4.0 * PI)) * (1.0 - g2) / pow(denom, 1.5);
}

/// Optical depth from `p` toward the sun, integrating exponential
/// densities of Rayleigh + Mie species. Returns the per-channel
/// `τ = β_R · ∫ρ_R + β_M · ∫ρ_M`, ready for `T = exp(-τ)`.
///
/// Includes a planet-occlusion test: if the sun ray from `p` hits the
/// solid sphere, the sun is below the local horizon and the function
/// returns a saturating optical depth so `T_sun → 0`. This is what
/// prevents the night-side surface from picking up a phantom haze.
fn sun_optical_depth(
    p: vec3<f32>,
    sun_dir: vec3<f32>,
    center: vec3<f32>,
    planet_r: f32,
    atmos_top_r: f32,
    beta_r: vec3<f32>,
    beta_m: f32,
    h_r: f32,
    h_m: f32,
) -> vec3<f32> {
    let oc = p - center;
    let half_b = dot(oc, sun_dir);

    // Planet occlusion. Ray (p, sun_dir) hits the solid sphere ahead?
    // Then the sun is below `p`'s local horizon — the sun column is
    // effectively infinite, T_sun = 0.
    let c_p = dot(oc, oc) - planet_r * planet_r;
    let disc_p = half_b * half_b - c_p;
    if disc_p > 0.0 {
        let t_p = -half_b - sqrt(disc_p);
        if t_p > 1e-3 {
            // exp(-40) ≈ 4e-18: indistinguishable from zero in any
            // tonemapped output; avoids `inf` propagating from a true
            // infinity sentinel.
            return vec3<f32>(40.0);
        }
    }

    // Atmosphere shell exit along the sun ray.
    let c_a = dot(oc, oc) - atmos_top_r * atmos_top_r;
    let disc_a = half_b * half_b - c_a;
    if disc_a < 0.0 {
        // `p` outside the shell — no extinction along this leg.
        return vec3<f32>(0.0);
    }
    let t_a = -half_b + sqrt(disc_a);
    if t_a <= 0.0 {
        return vec3<f32>(0.0);
    }

    let n = ATMOS_SUN_STEPS;
    let ds = t_a / f32(n);
    var od_r: f32 = 0.0;
    var od_m: f32 = 0.0;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let t = (f32(i) + 0.5) * ds;
        let q = p + t * sun_dir;
        let h = max(length(q - center) - planet_r, 0.0);
        od_r = od_r + exp(-h / h_r) * ds;
        od_m = od_m + exp(-h / h_m) * ds;
    }
    return beta_r * od_r + vec3<f32>(beta_m) * od_m;
}

/// Single-scattering raymarch along view ray segment `[t_enter, t_exit]`.
///
/// `cam_pos`, `ray_dir`, `center`, `sun_dir` are world-space; `t_enter`
/// is the atmosphere shell entry distance (or 0 if the camera is
/// inside the shell), `t_exit` is the surface hit distance (body
/// pass) or the atmosphere shell exit distance (miss / halo pass).
/// `planet_r` is the body's solid radius in render units.
///
/// `pixel_jitter ∈ [0, 1)` shifts the sample positions sub-step to
/// break the regular sampling pattern — without it, the terminator
/// shows banded artifacts at orbital views. `0.5` recovers the
/// centred-sample scheme (no jitter); per-pixel hash is the production
/// path. See `atmosphere_jitter`.
///
/// Returns `(in_scatter, transmittance)`. The caller does:
///   surface_lit = surface_lit * result.transmittance;
///   surface_lit = surface_lit + result.in_scatter;
fn integrate_atmosphere(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    sun_dir: vec3<f32>,
    sun_flux: f32,
    t_enter: f32,
    t_exit: f32,
    planet_r: f32,
    layers: AtmosphereBlock,
    pixel_jitter: f32,
) -> ScatterResult {
    if !atmosphere_scattering_active(layers) {
        return no_scatter();
    }
    let path = max(t_exit - t_enter, 0.0);
    if path <= 0.0 {
        return no_scatter();
    }

    let beta_r = layers.rayleigh_beta_h.xyz;
    let beta_m_scalar = layers.mie_beta_g.x;
    let h_r = max(layers.rayleigh_beta_h.w, 1e-3);
    let h_m = max(layers.atmos_geom.y, 1e-3);
    let atmos_top_r = planet_r + layers.atmos_geom.x;
    let strength = layers.atmos_geom.z;
    let g = layers.mie_beta_g.w;

    // Phase angle θ = angle between view direction (camera → scatter
    // point) and sun direction (scatter point → sun). Forward Mie
    // scatter peaks at θ = 0, which is when the camera looks toward
    // the sun through the atmosphere. cos θ = dot(ray_dir, sun_dir):
    // both vectors point in the "physical light/view" sense, so a
    // ray pointed at the sun has cos θ = +1 and the HG kernel peaks
    // there, brightening the haze on the lit limb / sub-solar side.
    let cos_theta = dot(ray_dir, sun_dir);
    let p_r = phase_rayleigh(cos_theta);
    let p_m = phase_mie_hg(cos_theta, g);

    let n = adaptive_view_steps(path, layers.atmos_geom.x);
    let ds = path / f32(n);
    let jitter = clamp(pixel_jitter, 0.0, 0.999);

    var sum_r = vec3<f32>(0.0);
    var sum_m = vec3<f32>(0.0);
    var od_r: f32 = 0.0;
    var od_m: f32 = 0.0;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let t = t_enter + (f32(i) + jitter) * ds;
        let p_pt = cam_pos + t * ray_dir;
        let h = max(length(p_pt - center) - planet_r, 0.0);
        let rho_r = exp(-h / h_r);
        let rho_m = exp(-h / h_m);
        od_r = od_r + rho_r * ds;
        od_m = od_m + rho_m * ds;

        // Transmittance from camera entry to this sample point.
        let tau_view = beta_r * od_r + vec3<f32>(beta_m_scalar) * od_m;
        let trans_view = exp(-tau_view);

        // Transmittance from sample point to sun.
        let tau_sun = sun_optical_depth(
            p_pt, sun_dir, center,
            planet_r, atmos_top_r,
            beta_r, beta_m_scalar, h_r, h_m,
        );
        let trans_sun = exp(-tau_sun);

        let weight = trans_view * trans_sun * ds;
        sum_r = sum_r + rho_r * weight;
        sum_m = sum_m + rho_m * weight;
    }

    let in_scatter = sun_flux * strength
        * (beta_r * (sum_r * p_r) + vec3<f32>(beta_m_scalar) * (sum_m * p_m));
    let total_tau = beta_r * od_r + vec3<f32>(beta_m_scalar) * od_m;
    let transmittance = exp(-total_tau);
    return ScatterResult(in_scatter, transmittance);
}

/// Single-scattering raymarch with a precomputed multi-scatter LUT.
///
/// Identical to [`integrate_atmosphere`] except that at each view sample the
/// average isotropic illumination at the sample point is read from `ms_lut`
/// and added to the in-scatter integral as `σ_s · ρ · L_ms · T_view · ds`.
/// `ms_lut` is the LUT produced by
/// `thalos_planet_lighting::bake_multi_scatter_lut`, indexed by
/// `(u = (μ_s + 1) / 2, v = h / atmos_top)`.
///
/// The multi-scatter term is what gives Earth's midday sky its blue luminance
/// — single-scattering alone leaves the horizon's long-path view dominated by
/// residual red because blue has scattered out along the view ray. The LUT
/// adds back the light that has bounced one or more times before reaching
/// the sample point, restoring blue across the dome and lifting in-scatter
/// brightness into the range where it can balance against star brightness in
/// the same shader-unit space.
///
/// Kept as a separate function from the single-scatter `integrate_atmosphere`
/// so call sites that don't need the LUT (impostor halo from space) don't
/// have to bind a dummy texture.
fn integrate_atmosphere_multiscatter(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    sun_dir: vec3<f32>,
    sun_flux: f32,
    t_enter: f32,
    t_exit: f32,
    planet_r: f32,
    layers: AtmosphereBlock,
    pixel_jitter: f32,
    ms_lut: texture_2d<f32>,
    ms_sampler: sampler,
) -> ScatterResult {
    if !atmosphere_scattering_active(layers) {
        return no_scatter();
    }
    let path = max(t_exit - t_enter, 0.0);
    if path <= 0.0 {
        return no_scatter();
    }

    let beta_r = layers.rayleigh_beta_h.xyz;
    let beta_m_scalar = layers.mie_beta_g.x;
    let h_r = max(layers.rayleigh_beta_h.w, 1e-3);
    let h_m = max(layers.atmos_geom.y, 1e-3);
    let atmos_top_alt = max(layers.atmos_geom.x, 1e-3);
    let atmos_top_r = planet_r + atmos_top_alt;
    let strength = layers.atmos_geom.z;
    let g = layers.mie_beta_g.w;

    let cos_theta = dot(ray_dir, sun_dir);
    let p_r = phase_rayleigh(cos_theta);
    let p_m = phase_mie_hg(cos_theta, g);

    let n = adaptive_view_steps(path, atmos_top_alt);
    let ds = path / f32(n);
    let jitter = clamp(pixel_jitter, 0.0, 0.999);

    var sum_r = vec3<f32>(0.0);
    var sum_m = vec3<f32>(0.0);
    var sum_ms = vec3<f32>(0.0);
    var od_r: f32 = 0.0;
    var od_m: f32 = 0.0;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let t = t_enter + (f32(i) + jitter) * ds;
        let p_pt = cam_pos + t * ray_dir;
        let to_center = p_pt - center;
        let r_pt = length(to_center);
        let h = max(r_pt - planet_r, 0.0);
        let rho_r = exp(-h / h_r);
        let rho_m = exp(-h / h_m);
        od_r = od_r + rho_r * ds;
        od_m = od_m + rho_m * ds;

        let tau_view = beta_r * od_r + vec3<f32>(beta_m_scalar) * od_m;
        let trans_view = exp(-tau_view);

        let tau_sun = sun_optical_depth(
            p_pt, sun_dir, center,
            planet_r, atmos_top_r,
            beta_r, beta_m_scalar, h_r, h_m,
        );
        let trans_sun = exp(-tau_sun);

        let weight = trans_view * trans_sun * ds;
        sum_r = sum_r + rho_r * weight;
        sum_m = sum_m + rho_m * weight;

        // Multi-scatter LUT lookup. Local-zenith at the sample point gives
        // the sun-zenith cosine; clamped to the LUT's symmetric range. The
        // sampler is `linear` so neighbouring cells blend smoothly; the
        // bilinear cost is one tex sample per view step.
        let zenith = to_center / max(r_pt, 1e-3);
        let mu_s = clamp(dot(sun_dir, zenith), -1.0, 1.0);
        let h_norm = clamp(h / atmos_top_alt, 0.0, 1.0);
        let lut_uv = vec2<f32>(mu_s * 0.5 + 0.5, h_norm);
        let l_ms = textureSampleLevel(ms_lut, ms_sampler, lut_uv, 0.0).rgb;

        // `σ_s · ρ · L_ms · T_view` — see Hillaire 2020 §5.2. The bake
        // already includes the isotropic-phase 1/(4π) integral, so this is
        // just per-channel β · density · stored radiance.
        let beta_rho = beta_r * rho_r + vec3<f32>(beta_m_scalar) * rho_m;
        sum_ms = sum_ms + trans_view * beta_rho * l_ms * ds;
    }

    // Multiple-scattering gain (atmos_geom.w): lifts only the blue-dominant
    // multi-scatter fill so the long-path horizon reads pale-blue instead of
    // the warm single-scatter residual, without dimming/re-warming the dome.
    let multi_gain = layers.atmos_geom.w;
    let in_scatter_single = sun_flux * strength
        * (beta_r * (sum_r * p_r) + vec3<f32>(beta_m_scalar) * (sum_m * p_m));
    let in_scatter_multi = sun_flux * strength * multi_gain * sum_ms;
    let total_tau = beta_r * od_r + vec3<f32>(beta_m_scalar) * od_m;
    let transmittance = exp(-total_tau);
    return ScatterResult(in_scatter_single + in_scatter_multi, transmittance);
}

/// Sample offset for atmosphere raymarches.
///
/// This used to return per-pixel interleaved-gradient noise to hide low-sample
/// terminator banding, but the screen-space pattern was visible in smooth sky
/// gradients. Keep the public helper so impostor and ground-sky paths stay in
/// lockstep, but use centered samples and rely on the higher step counts above
/// for smoothness instead of dithering.
fn atmosphere_jitter(coord: vec2<f32>) -> f32 {
    _ = coord;
    return 0.5;
}

// ── Cloud layer ─────────────────────────────────────────────────────────────
//
// Main cloud density is supplied by `planet_impostor.wgsl` from a reference
// cloud-cover cubemap. The procedural generator and shader-side procedural
// haze path are intentionally removed for now; the compositor only receives
// pre-sampled main and shadow densities.

/// Canonical weather-coverage remap for surface-following orbital cloud LODs.
/// The runtime weather field is authored around a mean coverage near 0.46.
/// Turn that continuous meteorological control into resolved clear/cloudy
/// regions for the far projection. A broad low-opacity shoulder reads as a
/// grey planetary veil; the narrow transition preserves open water between
/// weather cells. Shape comes from the continuous coverage field rather than
/// the categorical type channel, which is reserved for vertical structure.
fn weather_cloud_opacity(raw_coverage: f32) -> f32 {
    // AREAL-FRACTION semantics. The regime producer (2026-07-23) authors the
    // coverage channel as a true areal fraction with genuine zeros between
    // weather systems, so this is now close to identity: a soft formation toe
    // suppresses mip-filtering residue, nothing more. The previous
    // smoothstep(0.45, 0.80, cov·1.22) remap — calibrated to the old
    // everything-near-the-mean statistics — deleted moderate-coverage cumulus
    // fields from the far tier entirely while the near volume still rendered
    // them: the "impostor missing where volumetrics exist" failure.
    return smoothstep(0.04, 0.20, raw_coverage) * clamp(raw_coverage * 1.05, 0.0, 1.0);
}

/// Linearly reconstruct the canonical broad shape signal from four
/// LAYER-RELATIVE strata (channel centres at 1/8, 3/8, 5/8, 7/8 of the local
/// [base, top] interval — see the CPU producer). Callers map their shell
/// height through the same weather base/top channels:
/// `h_layer = (h_shell − base) / (top − base)`. Layer-relative sampling keeps
/// full vertical resolution for a thin deck wherever it sits in the shell;
/// the previous fixed shell-height strata had a dead zone where a ~2 km layer
/// between two sampling heights read zero from every stratum (2026-07-23).
/// Outside the layer the density is a hard zero — the strata edge values are
/// in-cloud samples, so clamping to them painted a halo above tops.
/// The payload is authored in body-direction space and therefore has no
/// Cartesian planet-scale repeat.
/// Catmull-Rom segment. End tangents are handled by duplicating the end knots
/// at the call site, which flattens the curve gently into the layer edges
/// instead of breaking its slope there.
fn cloud_strata_spline(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    return 0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

fn cloud_surface_shape(strata: vec4<f32>, layer_height: f32) -> f32 {
    if layer_height <= -0.04 || layer_height >= 1.04 { return 0.0; }
    // Knots at 1/8, 3/8, 5/8, 7/8 of the layer → z = 0..3.
    let z = clamp(clamp(layer_height, 0.0, 1.0) * 4.0 - 0.5, 0.0, 3.0);
    let k = clamp(floor(z), 0.0, 2.0);
    let t = z - k;
    // C1 reconstruction. This was piecewise LINEAR, and its slope breaks at the
    // four knots were the "straight artifacts cutting through the clouds"
    // (2026-07-26): `env` feeds `formation_threshold`, so a slope break in
    // altitude is a break in the density isosurface, and a deck viewed edge-on
    // maps altitude straight to screen-y — four horizontal shelves per cloud,
    // repeated across every cloud in the deck, with the sky showing through
    // between them. Reference quality reproduced it identically, which is what
    // ruled out the step budget, sparse scheduling and temporal reconstruction
    // and pointed here. Nothing masked it before this round because the
    // periodic Cartesian volume used to vary down the column; the cell field is
    // constant down a column by construction, so the strata are now the ONLY
    // vertical structure and their reconstruction is visible directly.
    var v = 0.0;
    if k < 1.0 {
        v = cloud_strata_spline(strata.x, strata.x, strata.y, strata.z, t);
    } else if k < 2.0 {
        v = cloud_strata_spline(strata.x, strata.y, strata.z, strata.w, t);
    } else {
        v = cloud_strata_spline(strata.y, strata.z, strata.w, strata.w, t);
    }
    // Catmull-Rom can overshoot between unequal knots; the payload is an areal
    // fraction, so clamp rather than let a negative density appear.
    return clamp(v, 0.0, 1.0);
}

/// Broad surface-space density contract shared by near and far projections.
/// Coverage, typed height response, and formation threshold are applied by the
/// CPU producer before its mip chain is built; this function reconstructs that
/// already-filterable density rather than thresholding a filtered raw signal.
/// `layer_height` is layer-relative (see [`cloud_surface_shape`]).
fn cloud_surface_density(
    strata: vec4<f32>,
    layer_height: f32,
) -> f32 {
    return cloud_surface_shape(strata, layer_height);
}

/// Footprint-filtered vertical column occupancy reconstructed from the same
/// four strata. `best + damped remainder` preserves discrete towers without
/// treating the levels as four independent opaque slabs.
fn cloud_surface_column_density(
    strata: vec4<f32>,
) -> f32 {
    let d0 = strata.x;
    let d1 = strata.y;
    let d2 = strata.z;
    let d3 = strata.w;
    let best = max(max(d0, d1), max(d2, d3));
    let sum = d0 + d1 + d2 + d3;
    return clamp(best + 0.22 * (sum - best) * (1.0 - best), 0.0, 1.0);
}

/// Column moments derived from the canonical weather cubemap (RGBA =
/// coverage, type, base, top). CLOUD-6's first orbital projection uses these
/// instead of coverage-only surface paint so the far LOD shares height and
/// optical-depth structure with the near volume. A later offline optical-depth
/// atlas can replace the producer without changing this consumer contract.
struct WeatherColumn {
    opacity: f32,
    optical_depth: f32,
    /// Local deck base as a fraction of the authored shell (0 = base altitude).
    base_frac: f32,
    /// Local deck mid as a fraction of the authored shell.
    mid_frac: f32,
    /// Local top fraction — limb silhouette / parallax shell.
    top_frac: f32,
    /// 0 = stratus, 1 = storm.
    stormness: f32,
}

fn weather_column_from_texel(weather: vec4<f32>) -> WeatherColumn {
    let cov = clamp(weather.r, 0.0, 1.0);
    let ty = clamp(weather.g, 0.0, 1.0);
    let local_base = clamp(weather.b, 0.0, 0.92);
    let local_top = max(clamp(weather.a, 0.02, 1.0), local_base + 0.02);
    let thickness = local_top - local_base;
    // Coverage is areal occupancy (see weather_cloud_opacity); it must NOT
    // also scale the column's optical depth — a cell inside a 30%-coverage
    // field is just as optically thick as one inside overcast. Multiplying
    // them made every moderate-coverage region simultaneously sparse AND
    // translucent, which is the grey-veil signature. Optical depth is a
    // per-cell property of type and thickness, with only a soft coverage
    // response so filtered fringes thin out.
    let opacity = weather_cloud_opacity(cov);
    let stratus_w = 1.0 - smoothstep(0.18, 0.38, ty);
    let storm_w = smoothstep(0.72, 0.88, ty);
    let cumulus_w = max(0.0, 1.0 - stratus_w - storm_w);
    let type_density = 0.55 * stratus_w + 0.95 * cumulus_w + 1.35 * storm_w;
    let optical_depth = (0.30 + 0.90 * smoothstep(0.06, 0.50, cov))
        * type_density
        * (0.40 + 2.6 * thickness);
    return WeatherColumn(
        opacity,
        optical_depth,
        local_base,
        mix(local_base, local_top, 0.55),
        local_top,
        storm_w,
    );
}

/// Altitude of the orbital cloud shell above the solid radius, in the same
/// render units as `AtmosphereBlock::cloud_shape` (x = base, y = thickness).
fn orbital_cloud_altitude(column: WeatherColumn, layers: AtmosphereBlock) -> f32 {
    let base_alt = max(layers.cloud_shape.x, 0.0);
    let thickness = max(layers.cloud_shape.y, 0.0);
    // Density-weighted representative height: most of a column's optical mass
    // sits near its mid, so parallax anchors there, with only storm towers
    // pulled toward their tops. The old 0.35–0.80 top bias floated the whole
    // far layer near the shell ceiling — a thin detached skin far above the
    // volume the near march actually renders.
    let frac = mix(column.mid_frac, column.top_frac, 0.10 + 0.25 * column.stormness);
    return base_alt + frac * thickness;
}

/// Orbital cloud shade from column moments. Returns premultiplied radiance
/// scale (multiply by albedo × sun_flux × SCENE_FLUX_SCALE) and opacity.
/// `n_dot_l` is the lit factor on the *cloud* normal (height-perturbed when
/// available); `view_mu` is cloud-normal · view for a cheap bright rim.
fn orbital_cloud_shade(
    column: WeatherColumn,
    n_dot_l: f32,
    view_mu: f32,
) -> vec2<f32> {
    if column.opacity <= 1.0e-4 {
        return vec2<f32>(0.0, 0.0);
    }
    // Wrap lighting: orbital clouds still read past the terminator slightly.
    let wrap = 0.18;
    let lit = clamp((n_dot_l + wrap) / (1.0 + wrap), 0.0, 1.0);
    // Dense / tall columns self-shadow and thicken without going charcoal or
    // filling the whole disc. One optical-depth core term drives both.
    let core = clamp(1.0 - exp(-column.optical_depth * 0.9), 0.0, 1.0);
    let self_shadow = mix(1.0, mix(0.62, 0.42, column.stormness), core * (1.0 - 0.55 * lit));
    // Thin glancing edges pick up a small energy-bounded rim so silhouettes
    // don't flatten into grey stickers.
    let rim = pow(1.0 - clamp(view_mu, 0.0, 1.0), 2.2) * (0.08 + 0.10 * column.stormness);
    // CLOUD-4: cooler radiance scale so the full disc no longer clips white;
    // sqrt lit keeps soft terminator while the 0.72 prefactor leaves room for
    // atmosphere-tinted sun on the caller.
    let radiance_scale = 0.72 * ((0.18 + 0.82 * sqrt(lit)) * self_shadow + rim * lit);
    // Occupancy stays with the formation gate (`column.opacity`); optical depth
    // only thickens already-resolved cells so storm cores read denser.
    let opacity = clamp(column.opacity * mix(0.72, 0.96, core), 0.0, 0.90);
    return vec2<f32>(radiance_scale, opacity);
}

/// Footprint-filtered weather fetch. The cube carries a box mip chain
/// (WEATHER_MIP_LEVELS); callers pass the mip level matching their projected
/// footprint (0 = full 256² resolution). The floor of 0.75 keeps a little
/// trilinear softening even close-up, which also removes the 256² face-square
/// faceting the old 5-tap cross existed to hide.
fn sample_weather_soft(
    weather_tex: texture_cube<f32>,
    weather_sampler: sampler,
    n: vec3<f32>,
    lod: f32,
) -> vec4<f32> {
    return textureSampleLevel(weather_tex, weather_sampler, n, max(lod, 0.75));
}

/// Angular size of one level-0 weather/surface texel (π/2 face over 1024
/// texels). This must track `WEATHER_FACE_SIZE`; retaining the old 256-face
/// value forced orbital consumers onto mip 0 and aliased surface cells.
const WEATHER_TEXEL_ANGLE: f32 = 0.001533981;

// ── Single-representation cloud march contract ──────────────────────────────
// (BL-20260724T003705Z; re-keyed by ADR-20260726T000929Z.) The near volumetric
// march carries ONE cloud representation across the whole flight envelope, with
// LOD keyed to the PROJECTED PIXEL FOOTPRINT and the ray's in-shell chord —
// never to raw camera distance.
//
// Distance was the wrong driver and it produced every ascent artifact at once.
// From orbit the deck is far away but its footprint is TINY: Bevy's default 45°
// vertical fov on a 1280×720 cloud target is 0.00115 rad/pixel, so at 300 km a
// pixel covers 345 m and a 5.4 km cell is ~12 pixels across. The old ladder read
// "300 km" and switched to 4800 m steps (≈10 samples through the whole deck —
// the horizontal comb on ascent cells) and then handed the frame to the
// statistical far estimator entirely (clouds simply vanished above 300 km).
// Footprint keying inverts that: the same orbital ray now marches at the radial
// floor, because that is what its pixel actually resolves.
//
// The alias-safety invariant is unchanged and now has the right driver: the
// density field band-limits to `filter_m` (this step, or the footprint,
// whichever is coarser) BEFORE the step grows — `cloud_cell_field` drops
// octaves, the Cartesian sub-cell sculptor retires, erosion retires. That is
// what the conservative-bounds rule (ADR-20260721T033055Z) requires; BL-33's
// moiré and INC-0011's isosurfaces came from stretching steps over the
// UNFILTERED field.
//
// Both `clouds_compute.wgsl` (the marcher) and `cloud_composite.wgsl` (the far
// tier) read the SAME functions below — the lockstep is structural.

/// Broad probes per projected pixel footprint. Refine runs at 1/5 of the broad
/// step throughout, so this is ~0.4 fine samples per pixel.
const CLOUD_MARCH_FOOTPRINT_STEPS: f32 = 2.0;
/// Near-field floor: a pixel 1 km away covers ~1 m, and marching at 2 m would
/// be absurd, so the footprint term needs a floor. But this floor is a pure
/// COST device, and it overrides the physically-correct footprint step at every
/// range that matters: at 30–60 km a pixel covers 35–70 m, so the honest step
/// is ~100 m and the floor was forcing 600 m.
///
/// At 600 m a grazing ray through a thin deck rendered as a field of horizontal
/// bricks with the sky showing through between them (user verdict 2026-07-26,
/// "straight artifacts cutting through" + "we can easily see through them" —
/// the gaps between the bars ARE the see-through). Measured on the level probe
/// (`artifacts/visual/runs/cloud-ghost-level/`, camera 700 m, look +3°):
/// 600 m banded, 300 m clean, 150 m clean. Cruise GPU cost on the development
/// 4070 Ti at 1280×720: 600 m → 1.98 ms mean / 3.33 p95; 300 m → 2.36 / 4.38;
/// 150 m → 2.74 / 5.12.
///
/// 300 m is the chosen point: the artifact is a correctness defect, and the
/// 3.5 ms p95 target is explicitly provisional ("adjusted after measurement
/// rather than treated as dogma", CLOUD-0). This is the knob to move if the
/// budget is re-tightened — and it must move with a matching capture, because
/// the failure it guards against is not subtle.
const CLOUD_MARCH_MIN_STEP_M: f32 = 300.0;
/// Vertical resolution cap: the step may never exceed this along the RADIAL
/// direction, or a thin deck is crossed in a handful of samples and renders as
/// horizontal slabs. Steep rays pay for it in short in-shell segments — this is
/// what lets an orbital nadir ray march the deck at 350 m.
const CLOUD_MARCH_MAX_RADIAL_STEP_M: f32 = 350.0;
/// Horizontal resolution cap, ~1/3 of the cell field's coarsest period: a
/// grazing ray moves almost nothing vertically per step, so what it must
/// resolve is the CELL, not the profile.
const CLOUD_MARCH_MAX_CELL_STEP_M: f32 = 1800.0;
/// Hard geometric cap on the marched shell segment.
const CLOUD_MARCH_REACH_M: f32 = 300000.0;
/// The marcher dissolves its density over the LAST fraction of the reach its
/// probe budget actually buys; the far tier fades in complementarily.
const CLOUD_MARCH_FADE_FRACTION: f32 = 0.85;

/// The far/orbital projection owns a WHOLE ray only once cell-scale morphology
/// is genuinely SUB-PIXEL — i.e. whole-disc and map framings, where a smooth
/// projection is correct by construction and no transition can be visible.
/// Keyed to footprint against the cell field's periods (5.4 km coarsest,
/// 0.96 km finest). On Thalos that puts the handoff near 1900–4700 km altitude,
/// so the entire flight envelope is volumetric — the point of the change: the
/// old 240–300 km entry-DISTANCE window put it in the middle of every ascent,
/// where a pixel still covers only ~345 m and cells are 12 px across.
const CLOUD_FAR_OWNERSHIP_START_M: f32 = 2200.0;
const CLOUD_FAR_OWNERSHIP_END_M: f32 = 5400.0;

fn cloud_far_ownership(footprint_m: f32) -> f32 {
    return smoothstep(CLOUD_FAR_OWNERSHIP_START_M, CLOUD_FAR_OWNERSHIP_END_M, footprint_m);
}

/// Broad-probe step at distance `t`. `radial_rate` is |dot(up, ray_dir)| — the
/// fraction of a step spent crossing the deck vertically.
///
/// The footprint term is a FLOOR (never sample finer than a pixel) and the two
/// resolution terms are CAPS (always resolve the deck vertically and the cells
/// horizontally), so the caps win where they disagree. A chord-length budget
/// floor was tried here instead and is the wrong shape: it makes the step a
/// function of how long the ray happens to be, which on a ground-level horizon
/// ray (chord ~600 km) produced ~9.7 km steps through a 1.15 km deck and
/// rendered the whole distant band as horizontal slabs. Budget belongs in the
/// REACH (below), not in the step.
fn cloud_march_step_m(t: f32, pixel_angle: f32, radial_rate: f32) -> f32 {
    let by_footprint = max(
        t * pixel_angle * CLOUD_MARCH_FOOTPRINT_STEPS,
        CLOUD_MARCH_MIN_STEP_M,
    );
    let radial_cap = CLOUD_MARCH_MAX_RADIAL_STEP_M / max(radial_rate, 1.0e-3);
    return min(min(by_footprint, radial_cap), CLOUD_MARCH_MAX_CELL_STEP_M);
}

/// Where a march entering the shell at `t_entry` with `steps` broad probes runs
/// out — the frontier the far tier fades in over. Integrates the SAME footprint
/// law in closed form: uniform at the floor while `a·t < MIN`, geometric while
/// the footprint governs, uniform again at the cell cap.
///
/// The resolution caps are deliberately not mirrored here. They only ever make
/// the real step SMALLER, and they bind exactly where the frontier is
/// irrelevant — steep rays, whose in-shell segments are geometrically short and
/// finish long before the budget does. Where the frontier does matter (grazing
/// rays at the floor) this is exact. Runs once per ray, so the transcendentals
/// are outside the per-sample density path.
fn cloud_march_stop_m(steps: f32, t_entry: f32, pixel_angle: f32) -> f32 {
    let a = max(pixel_angle * CLOUD_MARCH_FOOTPRINT_STEPS, 1.0e-9);
    let t_floor = CLOUD_MARCH_MIN_STEP_M / a;
    let t_cap = CLOUD_MARCH_MAX_CELL_STEP_M / a;
    var remaining = steps;
    var t = max(t_entry, 0.0);
    if t < t_floor {
        let n = (t_floor - t) / CLOUD_MARCH_MIN_STEP_M;
        if remaining <= n {
            return t + remaining * CLOUD_MARCH_MIN_STEP_M;
        }
        remaining -= n;
        t = t_floor;
    }
    if t < t_cap {
        let n = log(t_cap / t) / a;
        if remaining <= n {
            return t * exp(a * remaining);
        }
        remaining -= n;
        t = t_cap;
    }
    return min(t + remaining * CLOUD_MARCH_MAX_CELL_STEP_M, CLOUD_MARCH_REACH_M);
}

// ── Shared strata domain warp ────────────────────────────────────────────────
// Wherever the strata cube's ~5 km texels resolve on screen (the homogenized
// march bands and the far/orbital projection), the raw bilinear payload reads
// as rounded SQUARES — the texel lattice itself (user ascent verdict,
// 2026-07-24). Both consumers warp the strata lookup direction through THIS
// function with a matched amount, so cells turn organic while the two tiers
// stay registered. The warp is a measure-preserving remap of the direction
// domain, so the derived fill/response LUT statistics are unchanged — the CPU
// calibration needs no mirror.

fn strata_warp_hash(p: vec3<f32>) -> f32 {
    var q = vec3<u32>(bitcast<vec3<u32>>(vec3<i32>(floor(p))))
        * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    let n = (q.x ^ q.y ^ q.z) * 1597334673u;
    return f32(n) * (1.0 / 4294967295.0);
}

fn strata_warp_noise(x: vec3<f32>) -> f32 {
    let p = floor(x);
    var f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    let c000 = strata_warp_hash(p);
    let c100 = strata_warp_hash(p + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = strata_warp_hash(p + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = strata_warp_hash(p + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = strata_warp_hash(p + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = strata_warp_hash(p + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = strata_warp_hash(p + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = strata_warp_hash(p + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, f.x);
    let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x);
    let x11 = mix(c011, c111, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// ── Cell-scale cloud morphology: the shared aperiodic column field ──────────
//
// The planetary weather cube stores 4.9 km texels and its authored content
// bottoms out around 15–25 km, so nothing it carries can draw the 1–5 km cells
// that define a real broken deck — at any range. Those cells come from THIS
// field, and because it is the same function for the near march, the far
// projection and the CPU calibration mirror, cell identity does not change
// with distance. That is the answer to the LOD-seam family: the tiers now
// differ in how they INTEGRATE one field, never in which field they render.
//
// Two properties are load-bearing:
//
//   * It is parameterized by the body-fixed DIRECTION and evaluated from a
//     hash lattice, so it is genuinely aperiodic. Every previous attempt to
//     get cells at this scale drove a stored/periodic Cartesian tile, and the
//     spherical shell cut that repeat into planet-visible rows or combs —
//     ADR-20260722T141000Z, ADR-20260722T135123Z, and round 9's rejected
//     narrow shape period are all that one failure family.
//   * It is a COLUMN field: one horizontal identity from base to top, which is
//     what a convective column is. Vertical shape stays where it already
//     works — the marcher's typed profiles and dome threshold.
//
// `filter_m` is the sampling scale: the march step or the projected pixel
// footprint, whichever is coarser. An octave whose period falls under the
// sampler fades to its OWN MEAN, so the field band-limits ahead of the sampler
// and degrades cells → coarser cells → strata. It never collapses to a global
// mean: mean-preserving is not appearance-preserving, because opacity and
// lighting are nonlinear in density (E[shade(σ)] ≠ shade(E[σ])), and rendering
// the mean is exactly what made the old homogenized band a flat sheet with a
// contrast step at its edge.
/// Cell period. **Globally constant, and it must stay that way** — a spatially
/// varying period runs the field finer than the period its own band-limit
/// assumes, which aliases. Apparent cell size is varied by the octave WEIGHTS
/// instead; see the rule above `CLOUD_CELL_ROLL_ASPECT`.
const CLOUD_CELL_PERIOD_M: f32 = 5400.0;
const CLOUD_CELL_LACUNARITY: f32 = 2.37;
/// Spread correction about the mean. Smoothstep-interpolated lattice noise is
/// narrow (std 0.115 for this octave mix, measured over 1.5M directions), and
/// summing octaves narrows it further — so the raw field is nearly constant at
/// 0.5 and the formation threshold acts as a CLIFF: 0.40 → 0.60 swings areal
/// coverage 80 % → 20 %. Everything crossing such a threshold crosses it by
/// about the same margin, which renders as a carpet of identically-sized puffs
/// (first capture of this change). The gain restores std ≈ 0.20, so a threshold
/// selects peaks of genuinely different heights — cells of different sizes with
/// solid cores — which is the round-9 rule ("apparent cell size comes from the
/// threshold picking peaks out of a wide period") finally given a distribution
/// it can work on.
const CLOUD_CELL_GAIN: f32 = 3.2;
/// Soft-saturation knee. The gain MUST NOT be applied with a hard clamp: at
/// gain 2.4 that saturated 9.6 % of the sky (measured) to exactly 1.0, and a
/// constant region has no isosurface — the height-rising dome threshold then
/// cut every such core off at one exact altitude and the deck rendered as
/// flat-topped mesas with vertical sides (second capture of this change).
/// `x / (k + |x|)` keeps a gradient everywhere, so every core still carves a
/// rounded top, and is odd-symmetric about 0.5 so the mean is preserved
/// exactly. No transcendentals — the per-sample density path forbids them.
const CLOUD_CELL_KNEE: f32 = 0.45;
/// `E[|2v − 1|]` for `strata_warp_noise`, measured over 4M samples. The billow
/// octave is re-centred on it so its neutral value is exactly 0.5 and the
/// octave fade cannot shift the field's mean — a shift there would silently
/// de-calibrate the derived fill LUT at range.
const CLOUD_CELL_BILLOW_MEAN: f32 = 0.302816;
/// Per-octave billow ladder and octave weights. Named because the style's
/// spread normalization below has to reproduce the same mix analytically.
/// Weighted toward the low frequency so cells cluster into masses instead of
/// spreading as one uniform grain size.
const CLOUD_CELL_WEIGHTS: vec3<f32> = vec3<f32>(0.62, 0.26, 0.12);
const CLOUD_CELL_BILLOW: vec3<f32> = vec3<f32>(0.0, 0.75, 0.35);

// ── Morphology varies with PLACE (2026-07-26) ───────────────────────────────
//
// One planet must not render one cloud. Orbital photography of a terrestrial
// planet shows, within a single frame: wind-aligned roll streets, round
// open-cell honeycomb, sparse fair-weather cells over wide clear water, and
// solid decks split by narrow clear lanes. Until this, the field below had no
// spatial parameters at all — one period, isotropic, one lobe character
// everywhere — so the weather cube could vary how MUCH cloud a place had and
// how TALL it was, but never what KIND it was. Every region rendered the same
// popcorn carpet (user verdict 2026-07-26, with reference imagery).
//
// The style is analytic and direction-parameterized for the same two reasons
// the cell field is: nothing stored at 4.9 km texels can carry cell-scale
// morphology, and a stored tile repeats into planet-visible combs
// (ADR-20260725T222409Z). It varies over ~600 km — two orders above the cell
// period — so inside any one cell the domain map below is affine.
//
// CALIBRATION SAFETY shapes the whole design. The near tier's formation
// threshold is ONE Monte-Carlo fit over the whole planet (`fill_lut`), so a
// style that changed the field's DISTRIBUTION would silently make authored
// coverage mean different things in different places. Every knob here is
// therefore distribution-preserving by construction:
//
//   * anisotropy and tilt are LINEAR domain maps — exactly preserving;
//   * cell period is a domain rescale — exactly preserving;
//   * lobe character is NOT (σ swings 0.185 → 0.140 → 0.212 across the billow
//     range), so the octave mix is renormalized by its own analytic σ before
//     the gain.
//
// Measured over 4M directions per style (scratch harness, 2026-07-26): every
// style below lands at mean 0.4993–0.4995, σ 0.2031–0.2036, 0.00 % saturated —
// against the un-styled field's mean 0.4993 / σ 0.2036. The threshold curve
// therefore keeps its meaning everywhere, which is what lets this ship without
// a per-region calibration.

// ── The one rule this design exists to obey ─────────────────────────────────
//
// **A per-place style may never scale the sampling domain.**
//
// The field is sampled at `dir * (radius / period)` — about 590 lattice units on
// a Thalos-sized body. Let the period vary across the planet and the chain rule
// adds a second term to the sampling gradient: the field then runs at a LOCAL
// frequency well above the one its period nominally sets, while the octave fade
// still band-limits against that nominal period. The octave is under-filtered by
// exactly that factor, and it renders as fine feathered hatching, worst where
// the style gradient is steepest.
//
// Measured, not argued (`live_style_field_stays_coherent_across_style_boundaries`):
// the shipped varying period ran the field **1.70×** finer than its nominal
// period; this design runs at 1.13×, and that residual is the roll variant's
// genuinely finer across-street spacing, which `cloud_cell_roll_limit` accounts
// for. A varying `zonal_aspect` has the identical flaw, since it scales the
// domain by √a.
//
// This shipped twice before being isolated (user verdict 2026-07-26: "too
// distorted", then "strange artificial stripy patterns", then "not much
// better"). The capture that pinned it: styling the period ALONE, with billow
// and anisotropy neutral, reproduces the hatching; a constant period is clean.
//
// Everything below is therefore built from operations that cannot add a
// sampling-gradient term:
//
//   * **octave weights** — multiply already-sampled values, so apparent cell
//     size varies with no domain change at all;
//   * **billow** — blends two values taken at the SAME point;
//   * **roll blend** — cross-fades two globally CONSTANT domain transforms, so
//     every sample keeps the frequency its own band-limit assumes.
//
// A varying period/aspect is not recoverable by tuning; do not reintroduce one.

/// The fixed anisotropic variant of the arrangement octave. Constant, so the
/// transform contributes no phase gradient; regional variation comes from
/// blending toward it, never from moving it.
///
/// Anisotropy is applied to the arrangement octave only — a cloud street is a
/// row of round cumulus, not a ribbon, and stretching the octaves that carry an
/// individual cloud's silhouette turned every cloud into a tapering comet.
const CLOUD_CELL_ROLL_ASPECT: f32 = 3.0;
/// Shear off the exact east–west line. Also constant, and non-zero on purpose:
/// mixing longitude back into the lattice's y coordinate is what stops a high
/// aspect degenerating the field into pure latitude bands (circular contours on
/// a sphere — the fingerprint whorls of round 2).
const CLOUD_CELL_ROLL_TILT: f32 = 0.35;

/// σ of ONE octave as a function of its billow blend: a quadratic fit to 11
/// measured points, worst error 0.0023. Non-monotonic — billow first narrows
/// the distribution (0.4) and then widens it past the plain-noise value.
const CLOUD_CELL_SIGMA_FIT: vec3<f32> = vec3<f32>(0.184571, -0.200902, 0.230569);
/// σ of the default octave mix — the spread the derived threshold curve was fit
/// against, and therefore the target every style is renormalized back onto.
const CLOUD_CELL_SIGMA_REF: f32 = 0.123275;

/// How a place's clouds are ORGANIZED, as opposed to how much of them there is.
struct CloudCellStyle {
    /// Octave weights — the cell-size control. Must sum to 1 so the field's
    /// mean stays 0.5.
    weights: vec3<f32>,
    /// Blend toward the fixed anisotropic variant of the arrangement octave:
    /// 0 = round cells, 1 = wind-aligned rolls and lanes.
    roll: f32,
    /// Scale on the per-octave billow ladder: < 1 smooth sheets, > 1 puffy
    /// lobes with real gaps between them.
    billow: f32,
    /// `CLOUD_CELL_SIGMA_REF / σ(style)` — see the calibration note above.
    spread_norm: f32,
}

/// Analytic σ of the weighted octave mix. The octaves sit at decorrelated
/// offsets and frequencies, so the independent-sum model holds: measured
/// against direct sampling it is accurate to 0.25 % over the whole billow
/// range. Both the weights and the billow move it, so both are folded in here
/// — that is what keeps authored coverage meaning the same thing in every
/// region under one planet-wide threshold fit.
fn cloud_cell_spread_norm(weights: vec3<f32>, billow_scale: f32) -> f32 {
    let b = CLOUD_CELL_BILLOW * billow_scale;
    let s = CLOUD_CELL_SIGMA_FIT.x + CLOUD_CELL_SIGMA_FIT.y * b
        + CLOUD_CELL_SIGMA_FIT.z * b * b;
    return CLOUD_CELL_SIGMA_REF / max(sqrt(dot(weights * weights, s * s)), 1.0e-5);
}

/// Resolve the local cloud organization from the body-fixed vertical and the
/// weather cube's type channel.
///
/// Two inputs, deliberately: `cloud_type` is the producer's own regime
/// projection, so morphology agrees with the weather field BY CONSTRUCTION
/// rather than by a second noise field that would put fair-weather streets
/// inside a storm. `org` adds the independent degree of freedom type cannot
/// carry — whether a place's convection is organized into rolls or left as
/// round cells — and it is ONE low-frequency fetch, which is the entire added
/// per-sample cost of this feature.
///
/// The two axes span the reference imagery: low org + sheet = solid deck; high
/// org + sheet = deck with clear lanes; low org + cumulus = sparse round
/// fair-weather cells; high org + cumulus = roll streets; storm = coarse and
/// round at any org.
fn cloud_cell_style(dir: vec3<f32>, cloud_type: f32) -> CloudCellStyle {
    // ~640 km features on a Thalos-sized body: far above cell scale, far below
    // planetary, so a region holds one organization and neighbouring regions
    // differ.
    let org_raw = strata_warp_noise(dir * 5.0 + vec3<f32>(61.0, -23.0, 14.0));

    // Rolls need a sustained shear flow, which is a property of the trade and
    // mid-latitude belts, not of the deep tropics or the poles. Without this
    // gate the streets read as a planet-wide corduroy.
    let abs_lat = abs(clamp(dir.y, -1.0, 1.0));
    let roll_belt = smoothstep(0.08, 0.30, abs_lat) * (1.0 - smoothstep(0.60, 0.88, abs_lat));
    // Deep convection destroys roll organization — a storm is a round cluster.
    let not_storm = 1.0 - smoothstep(0.70, 0.90, cloud_type);
    let roll = smoothstep(0.44, 0.80, org_raw) * roll_belt * not_storm;

    let storm_w = smoothstep(0.72, 0.88, cloud_type);
    let sheet_w = 1.0 - smoothstep(0.14, 0.42, cloud_type);

    // ── Cell size, WITHOUT touching the domain ──────────────────────────────
    // Apparent cell size comes from shifting weight between the octaves, never
    // from scaling their periods. See `CLOUD_CELL_PERIOD_M`: a varying period is
    // a phase gradient of ~590 lattice units per unit relative change, which
    // decorrelates the field wherever the style varies. Weights multiply
    // already-sampled values, so they can vary as freely as we like.
    //
    // Both endpoints sum to 1, so the mix mean stays exactly 0.5 whatever the
    // blend. Deep convection and sheets organize at a coarser spacing, so they
    // pull toward the low-frequency end.
    let size_t = clamp(
        smoothstep(0.20, 0.80, org_raw) - 0.35 * storm_w - 0.15 * sheet_w,
        0.0,
        1.0,
    );
    let weights = mix(
        vec3<f32>(0.78, 0.16, 0.06),
        vec3<f32>(0.44, 0.34, 0.22),
        size_t,
    );

    // Sheets are smooth and continuous; convective fields are lobed with gaps
    // between the lobes. This is the knob that separates a stratus deck from a
    // cumulus field at the same coverage. Safe to vary freely: the billow blend
    // mixes two values taken at the SAME sample point, so it never moves the
    // domain.
    let billow = mix(1.15, 0.30, sheet_w);

    // ── Roll blend ──────────────────────────────────────────────────────────
    // How much of the arrangement octave comes from the FIXED anisotropic
    // variant rather than the isotropic one. A blend weight, not a transform
    // parameter — for the same reason the period is gone: a spatially varying
    // aspect scales the domain and therefore scrambles phase. Two globally
    // constant transforms, cross-faded, keep every sample coherent.
    let polar_fade = 1.0 - smoothstep(0.62, 0.90, abs_lat);
    let roll_blend = clamp(
        (roll + 0.55 * sheet_w * smoothstep(0.50, 0.86, org_raw)) * polar_fade,
        0.0,
        1.0,
    );

    return CloudCellStyle(weights, roll_blend, billow, cloud_cell_spread_norm(weights, billow));
}

/// The sampling domain for one octave, at a CONSTANT anisotropy.
///
/// Zonal elongation is the diagonal map `diag(1/a, 1, 1/a)` in the body frame:
/// a step east moves only `1/a` as far through the lattice, so features stretch
/// by exactly `a` along the wind at every latitude, while a step north is
/// untouched at the equator. It is linear (distribution preserved exactly),
/// transcendental-free, and seamless. The three alternatives all fail: scaling
/// a longitude ANGLE tears at the ±π meridian, a tangent-frame scale degenerates
/// (every tangent is perpendicular to `dir`, so the map is the identity), and a
/// stored anisotropic tile repeats — the failure family ADR-20260722T141000Z
/// already catalogues.
///
/// `aspect` and `tilt` are compile-time constants at every call site. Making
/// them vary per place is the phase-gradient error documented above.
fn cloud_cell_domain(
    dir: vec3<f32>,
    radius: f32,
    period_m: f32,
    aspect: f32,
    tilt: f32,
) -> vec3<f32> {
    let a = max(aspect, 1.0);
    // Constant cell area: across ÷ √a, along × √a, so a higher aspect turns
    // cells into rolls instead of enlarging everything.
    let k = radius / (period_m / sqrt(a));
    let inv = 1.0 / a;
    let p = vec3<f32>(dir.x * k * inv, dir.y * k, dir.z * k * inv);
    return vec3<f32>(p.x, p.y + tilt * (p.x + p.z), p.z);
}

/// The smallest feature dimension the anisotropic variant produces — what the
/// octave fade must band-limit against, since it is finer than the period.
fn cloud_cell_roll_limit(period_m: f32) -> f32 {
    let across = period_m / sqrt(CLOUD_CELL_ROLL_ASPECT);
    // The shear raises the peak gradient by sqrt(1 + tilt²).
    return across / sqrt(1.0 + CLOUD_CELL_ROLL_TILT * CLOUD_CELL_ROLL_TILT);
}

/// Shape one raw lattice sample into an octave value: billow blend, then the
/// band-limit fade toward the octave's own mean.
fn cloud_cell_shape(v: f32, billow: f32, fade: f32) -> f32 {
    let b = 0.5 + (abs(2.0 * v - 1.0) - CLOUD_CELL_BILLOW_MEAN);
    return mix(0.5, mix(v, b, billow), fade);
}

/// One isotropic octave, faded to its own mean once its period drops under the
/// sampler.
fn cloud_cell_octave(
    dir: vec3<f32>,
    radius: f32,
    period_m: f32,
    offset: vec3<f32>,
    filter_m: f32,
    billow: f32,
) -> f32 {
    let fade = 1.0 - smoothstep(0.45 * period_m, period_m, filter_m);
    if fade <= 1.0e-3 {
        return 0.5;
    }
    let v = strata_warp_noise(cloud_cell_domain(dir, radius, period_m, 1.0, 0.0) + offset);
    return cloud_cell_shape(v, billow, fade);
}

/// The arrangement octave: the isotropic sample cross-faded toward the fixed
/// anisotropic variant by `roll`.
///
/// Cross-fading two decorrelated fields costs variance — σ falls by
/// `sqrt(w² + (1−w)²)`, i.e. to 0.71σ at an even mix — so the blend is
/// renormalized back onto the isotropic spread. Without that the transition
/// belts would render as washed-out bands between the round-cell and rolled
/// regions, which is the same "render the mean" failure the LOD contract
/// forbids, arriving through a different door.
fn cloud_cell_arrangement(
    dir: vec3<f32>,
    radius: f32,
    period_m: f32,
    offset: vec3<f32>,
    filter_m: f32,
    billow: f32,
    roll: f32,
) -> f32 {
    let iso = cloud_cell_octave(dir, radius, period_m, offset, filter_m, billow);
    if roll <= 1.0e-3 {
        return iso;
    }
    let limit = cloud_cell_roll_limit(period_m);
    let fade = 1.0 - smoothstep(0.45 * limit, limit, filter_m);
    var rolled = 0.5;
    if fade > 1.0e-3 {
        let v = strata_warp_noise(
            cloud_cell_domain(
                dir,
                radius,
                period_m,
                CLOUD_CELL_ROLL_ASPECT,
                CLOUD_CELL_ROLL_TILT,
            ) + offset,
        );
        rolled = cloud_cell_shape(v, billow, fade);
    }
    let blended = mix(iso, rolled, roll);
    let shrink = sqrt(roll * roll + (1.0 - roll) * (1.0 - roll));
    return 0.5 + (blended - 0.5) / max(shrink, 1.0e-3);
}

/// Cell-scale occupancy in [0, 1] with mean ≈ 0.5, band-limited to `filter_m`.
/// `dir` is the body-fixed unit vertical at the sample; `radius` is the body
/// radius (the field's periods are metres of arc on the surface).
///
/// Octave PERIODS are global constants — only the weights vary per place. See
/// the phase-gradient note above; this is the invariant that makes the field
/// coherent across a style boundary instead of hatched.
///
/// `spread_norm` is applied to the octave mix and NOT to the fade: the fade's
/// variance loss is the intended band-limiting, and re-inflating it would undo
/// the aliasing protection the whole LOD contract rests on.
fn cloud_cell_field(
    dir: vec3<f32>,
    radius: f32,
    filter_m: f32,
    style: CloudCellStyle,
) -> f32 {
    let p0 = CLOUD_CELL_PERIOD_M;
    let p1 = p0 / CLOUD_CELL_LACUNARITY;
    let p2 = p1 / CLOUD_CELL_LACUNARITY;
    let b = CLOUD_CELL_BILLOW * style.billow;
    // Only the coarse octave carries the arrangement, so only it is elongated.
    let o0 = cloud_cell_arrangement(
        dir, radius, p0, vec3<f32>(11.3, -4.1, 27.9), filter_m, b.x, style.roll,
    );
    let o1 = cloud_cell_octave(dir, radius, p1, vec3<f32>(-23.7, 8.4, 3.2), filter_m, b.y);
    let o2 = cloud_cell_octave(dir, radius, p2, vec3<f32>(5.9, 31.2, -17.6), filter_m, b.z);
    let w = style.weights;
    let raw = w.x * o0 + w.y * o1 + w.z * o2;
    let x = (raw - 0.5) * style.spread_norm * CLOUD_CELL_GAIN;
    return 0.5 + 0.5 * x / (CLOUD_CELL_KNEE + abs(x));
}

/// Warp a body-fixed strata lookup direction by up to ~0.9 texels of organic
/// tangential offset. `amount` in [0, 1]; keyed only on the direction, so any
/// two consumers using the same amount sample the same warped field.
fn cloud_strata_warp(n: vec3<f32>, amount: f32) -> vec3<f32> {
    if amount <= 1.0e-3 {
        return n;
    }
    // ~2-texel noise period on the unit sphere (1024-face cube).
    let domain = n * 340.0;
    let wu = strata_warp_noise(domain) - 0.5;
    let wv = strata_warp_noise(domain + vec3<f32>(19.7, -7.3, 41.1)) - 0.5;
    // Any stable tangent frame works — the warp only needs to be tangential
    // and continuous away from the poles of the helper axis.
    var t = cross(n, vec3<f32>(0.0, 1.0, 0.0));
    if dot(t, t) < 1.0e-6 {
        t = cross(n, vec3<f32>(1.0, 0.0, 0.0));
    }
    t = normalize(t);
    let b = cross(n, t);
    let amp = amount * (0.9 * 2.0 * WEATHER_TEXEL_ANGLE);
    return normalize(n + (t * wu + b * wv) * amp);
}

/// Height-moment normal for orbital cloud lighting: finite difference of the
/// local top/coverage field in the body-fixed tangent plane. Gives soft relief
/// on storm cells without a pre-baked normal atlas.
fn orbital_cloud_normal_body(
    weather_tex: texture_cube<f32>,
    weather_sampler: sampler,
    n: vec3<f32>,
    lod: f32,
) -> vec3<f32> {
    var t = cross(n, vec3<f32>(0.0, 1.0, 0.0));
    if (dot(t, t) < 1.0e-8) {
        t = cross(n, vec3<f32>(1.0, 0.0, 0.0));
    }
    t = normalize(t);
    let b = cross(n, t);
    // Stencil widens with the sampled mip so the derived relief stays at the
    // resolved feature scale instead of amplifying sub-footprint texel noise.
    let e = 0.0045 * (1.0 + 0.8 * lod);
    let h = weather_column_from_texel(
        textureSampleLevel(weather_tex, weather_sampler, n, lod),
    );
    let h_t = weather_column_from_texel(
        textureSampleLevel(weather_tex, weather_sampler, normalize(n + e * t), lod),
    );
    let h_b = weather_column_from_texel(
        textureSampleLevel(weather_tex, weather_sampler, normalize(n + e * b), lod),
    );
    // Relief from both optical mass and local top so tall cells cast self-
    // shadow even when coverage is uniform.
    let c0 = h.optical_depth * 0.55 + h.top_frac * 0.45;
    let c_t = h_t.optical_depth * 0.55 + h_t.top_frac * 0.45;
    let c_b = h_b.optical_depth * 0.55 + h_b.top_frac * 0.45;
    let dh_t = (c_t - c0) * 1.8;
    let dh_b = (c_b - c0) * 1.8;
    return normalize(n - t * dh_t - b * dh_b);
}

/// Composite the cloud layer on top of an already-lit surface colour.
///
/// Pure Lambertian shading, no phase-function highlights. At orbital
/// scale clouds read as *matte* — bright white when lit, with only the
/// densest storm cores picking up any interior shading.
///
/// Main-deck density is **supplied by the caller** as two pre-sampled
/// scalars (`main_cloud_density`, `shadow_cloud_density`). The fetch lives
/// caller-side because the cubemap binding is on the impostor material, not
/// in this library module. Parallax — the
/// visual cue that clouds float above the surface — is preserved by
/// the caller's choice of cloud-shell (vs surface) intersection when
/// resolving the main sample direction.
///
/// Coverage scaling: raw cubemap value × 2 × coverage makes the authored
/// `coverage` parameter an approximate fraction of the disk that ends up
/// overcast.
///
/// `sun_flux_scaled` is the caller's pre-normalised sunlight
/// contribution (e.g., `sun_flux * hapke_scale`) so cloud brightness
/// stays in photometric lockstep with the surface beneath.
fn composite_clouds(
    surface_lit: vec3<f32>,
    normal_ws: vec3<f32>,
    sun_dir_ws: vec3<f32>,
    sun_flux_scaled: f32,
    ambient: f32,
    layers: AtmosphereBlock,
    main_cloud_density: f32,
    shadow_cloud_density: f32,
) -> vec3<f32> {
    let coverage = layers.cloud_albedo_coverage.w;
    if coverage <= 0.0 {
        return surface_lit;
    }

    let raw_ndl = dot(normal_ws, sun_dir_ws);

    // Coverage-scaled densities. The ×2×coverage map lets the author's
    // `coverage` value approximate the fraction of the disk that ends up
    // overcast.
    let cov_scale = 2.0 * coverage;
    let density = clamp(main_cloud_density * cov_scale, 0.0, 1.0);
    let shadow_density = clamp(shadow_cloud_density * cov_scale, 0.0, 1.0);

    // ── Cast shadow on the surface ──────────────────────────────────
    //
    // `shadow_cloud_density` was sampled at a direction offset *toward
    // the sun* from the SURFACE point — the "what cloud sits between
    // this terrain pixel and the sun" probe. Fade near the terminator
    // where the offset direction loses meaning.
    var shadowed_surface = surface_lit;
    if raw_ndl > -0.10 {
        let shadow_tau = shadow_density * shadow_density * 3.0;
        let shadow_opacity = 1.0 - exp(-shadow_tau);
        let shadow_factor = 1.0 - 0.65 * shadow_opacity;
        let shadow_fade = smoothstep(-0.10, 0.30, raw_ndl);
        shadowed_surface = surface_lit * mix(1.0, shadow_factor, shadow_fade);
    }

    let albedo = layers.cloud_albedo_coverage.xyz;
    let night_suppress = smoothstep(-0.15, 0.10, raw_ndl);

    // ── Main cumulus layer ──────────────────────────────────────────
    //
    // Cloud visibility uses the cloud-shell intersection direction —
    // this is where the view ray actually hits the cloud deck, and
    // at grazing angles it is DIFFERENT from the surface sample
    // direction. That parallax is the dominant perceptual cue that
    // clouds live above the surface, not painted on it.
    var result = shadowed_surface;
    if density >= 1e-3 {
        // Wrap-lit Lambert. Modest wrap (0.15) — clouds DO scatter past
        // the terminator but less than thick gas-giant decks.
        let wrap = 0.15;
        let n_dot_l = clamp((raw_ndl + wrap) / (1.0 + wrap), 0.0, 1.0);

        // Density-graded self-shadow. Only engages in truly dense
        // cores (density > 0.75) and dims at most 20%. Most of the
        // cloud body stays at full brightness because the squared-
        // density opacity curve below already makes thin clouds
        // translucent — no need to darken them further.
        let core = smoothstep(0.75, 1.00, density);
        let self_shadow = mix(1.0, 0.80, core);

        let cloud_sun = albedo * n_dot_l * self_shadow * sun_flux_scaled;
        let cloud_amb = albedo * ambient * 0.15;
        let cloud_lit = cloud_sun + cloud_amb;

        // Squared-density Beer-Lambert. τ ∝ density² means thin and
        // medium clouds stay translucent (density 0.3 → 24% opacity;
        // density 0.5 → 53%) and only the dense cores go opaque
        // (density 0.9 → 91%; density 1.0 → 95%). This captures the
        // reference: "most of Earth's weather isn't fully opaque
        // except maybe deep in the core." A linear τ made all
        // non-trivial densities read as solid cloud sheets.
        let tau = density * density * 3.0;
        let opacity = clamp(1.0 - exp(-tau), 0.0, 1.0);

        result = mix(shadowed_surface, cloud_lit * night_suppress,
                     opacity * night_suppress);
    }

    return result;
}
