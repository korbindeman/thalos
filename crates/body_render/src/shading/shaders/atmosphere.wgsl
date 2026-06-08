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
