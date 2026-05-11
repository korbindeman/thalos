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
    ///   w = reserved (ozone band — M4 follow-up).
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
    ///   x = fBm base frequency (cycles over the unit sphere),
    ///   y = softness of the cloud/no-cloud boundary,
    ///   z = fBm octaves (cast to u32 in the shader),
    ///   w = differential rotation coefficient in [0, 1].
    cloud_shape: vec4<f32>,
    /// Cloud layer dynamics.
    ///   x = equatorial scroll rate (radians / second of sim time),
    ///   y = current sim time (seconds, wrapped to a day-scale
    ///       modulus so f32 precision stays tight),
    ///   z = seed low 32 bits (as f32-bit reinterpret of u32),
    ///   w = seed high 32 bits.
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

const ATMOS_VIEW_STEPS: u32 = 8u;
const ATMOS_SUN_STEPS: u32 = 6u;

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

    let n = ATMOS_VIEW_STEPS;
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

/// Per-pixel jitter ∈ [0, 1) for raymarch sample offsets.
///
/// Interleaved Gradient Noise (Jorge Jimenez, "Next Generation Post
/// Processing in Call of Duty Advanced Warfare", 2014). Gives a
/// gradient-like high-frequency pattern that visually resolves to
/// uniform mid-grey at the eye's perceptual scale, instead of the
/// salt-and-pepper speckle a white-noise hash produces. Same
/// statistical decorrelation across pixels, much smoother under
/// the smooth-output integral; the jittered-sample variance lands
/// below the perceptual threshold where white-noise jitter is
/// visible as static.
///
/// Output is multiplied into `(f32(i) + jitter) * ds` in
/// `integrate_atmosphere`, shifting each pixel's sample positions
/// sub-step to break the regular sampling pattern that otherwise
/// produces banded artifacts at the terminator (where the sun-column
/// changes rapidly along the view ray).
fn atmosphere_jitter(coord: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(coord, magic.xy)));
}

// ── Cloud layer ─────────────────────────────────────────────────────────────
//
// Main cloud density is a baked cubemap (see `thalos_cloud_gen`) produced
// at planet load via Wedekind's curl-noise warp advection. The caller
// (`planet_impostor.wgsl`) samples the cubemap at the cloud-shell
// intersection direction and at an offset-toward-sun shadow probe
// direction, and hands the two scalars to `composite_clouds`. Drift
// over sim time is re-introduced by rotating the sample direction in
// `rotate_cloud_dir_local` before the cubemap lookup — equator
// fastest via the `diff` coefficient in `cloud_shape.w`.
//
// The procedural noise helpers below (`cloud_pcg`, `cloud_gradient_3d`,
// `cloud_value_noise_3d`, `cloud_fbm`) are retained because the
// cirrostratus/haze layer (`sample_haze_density`) is still live-shaded:
// haze is thin, broad, and uncorrelated with the main-deck weather
// structure, so the extra cost of keeping it procedural is small and
// authoring-time parameters (frequency, octaves, scroll rate) control
// it without a bake.

fn cloud_pcg(x: u32) -> u32 {
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Pick one of 12 gradient vectors (edges of a unit cube) from an
/// integer lattice coordinate. Classic Perlin gradient set — gives
/// visually much better results than a purely random hash-based
/// gradient, and avoids the axis-aligned "plus-shaped" artefacts you
/// get from naive random vectors.
fn cloud_gradient_3d(ix: i32, iy: i32, iz: i32, seed_lo: u32, seed_hi: u32) -> vec3<f32> {
    var h: u32 = (u32(ix) * 73856093u) ^ (u32(iy) * 19349663u) ^ (u32(iz) * 83492791u);
    h = cloud_pcg(h ^ seed_lo);
    h = cloud_pcg(h ^ (seed_hi * 1540483477u));
    let idx = h % 12u;
    // 12 edge vectors: permutations of (±1, ±1, 0).
    switch idx {
        case 0u:  { return vec3<f32>( 1.0,  1.0,  0.0); }
        case 1u:  { return vec3<f32>(-1.0,  1.0,  0.0); }
        case 2u:  { return vec3<f32>( 1.0, -1.0,  0.0); }
        case 3u:  { return vec3<f32>(-1.0, -1.0,  0.0); }
        case 4u:  { return vec3<f32>( 1.0,  0.0,  1.0); }
        case 5u:  { return vec3<f32>(-1.0,  0.0,  1.0); }
        case 6u:  { return vec3<f32>( 1.0,  0.0, -1.0); }
        case 7u:  { return vec3<f32>(-1.0,  0.0, -1.0); }
        case 8u:  { return vec3<f32>( 0.0,  1.0,  1.0); }
        case 9u:  { return vec3<f32>( 0.0, -1.0,  1.0); }
        case 10u: { return vec3<f32>( 0.0,  1.0, -1.0); }
        default:  { return vec3<f32>( 0.0, -1.0, -1.0); }
    }
}

/// 3D Perlin gradient noise. Output is approximately in [-1, 1].
///
/// Gradient noise has much better high-frequency content than value
/// noise: the first derivative is zero at lattice points (so there's
/// no "step up" at each grid cell), and the quintic interpolation
/// means the second derivative is continuous (no visible Mach bands).
/// The visible result is sharper detail per octave, which is what
/// carries the cumulus-cell look.
fn cloud_value_noise_3d(p: vec3<f32>, seed_lo: u32, seed_hi: u32) -> f32 {
    let ip_f = floor(p);
    let pf = p - ip_f;
    // Quintic fade: f(x) = 6x⁵ − 15x⁴ + 10x³. Smoother than the
    // smoothstep cubic and gives zero first + second derivatives
    // at integer points — no grid-aligned artefacts.
    let u = pf * pf * pf * (pf * (pf * 6.0 - 15.0) + 10.0);
    let ix = i32(ip_f.x);
    let iy = i32(ip_f.y);
    let iz = i32(ip_f.z);

    let g000 = cloud_gradient_3d(ix,     iy,     iz,     seed_lo, seed_hi);
    let g100 = cloud_gradient_3d(ix + 1, iy,     iz,     seed_lo, seed_hi);
    let g010 = cloud_gradient_3d(ix,     iy + 1, iz,     seed_lo, seed_hi);
    let g110 = cloud_gradient_3d(ix + 1, iy + 1, iz,     seed_lo, seed_hi);
    let g001 = cloud_gradient_3d(ix,     iy,     iz + 1, seed_lo, seed_hi);
    let g101 = cloud_gradient_3d(ix + 1, iy,     iz + 1, seed_lo, seed_hi);
    let g011 = cloud_gradient_3d(ix,     iy + 1, iz + 1, seed_lo, seed_hi);
    let g111 = cloud_gradient_3d(ix + 1, iy + 1, iz + 1, seed_lo, seed_hi);

    // Dot each gradient with the offset from its corner to `p`.
    let n000 = dot(g000, pf);
    let n100 = dot(g100, pf - vec3<f32>(1.0, 0.0, 0.0));
    let n010 = dot(g010, pf - vec3<f32>(0.0, 1.0, 0.0));
    let n110 = dot(g110, pf - vec3<f32>(1.0, 1.0, 0.0));
    let n001 = dot(g001, pf - vec3<f32>(0.0, 0.0, 1.0));
    let n101 = dot(g101, pf - vec3<f32>(1.0, 0.0, 1.0));
    let n011 = dot(g011, pf - vec3<f32>(0.0, 1.0, 1.0));
    let n111 = dot(g111, pf - vec3<f32>(1.0, 1.0, 1.0));

    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    let y0  = mix(x00, x10, u.y);
    let y1  = mix(x01, x11, u.y);
    // Output max magnitude for this gradient set is ~1.25; scale so
    // results land roughly in [-1, 1] for consistency with the
    // calibrated fBm gain.
    return mix(y0, y1, u.z) * 0.85;
}

fn cloud_fbm(p: vec3<f32>, octaves: u32, seed_lo: u32, seed_hi: u32) -> f32 {
    var sum: f32 = 0.0;
    var amp: f32 = 0.5;
    var freq: f32 = 1.0;
    var norm: f32 = 0.0;
    // Per-octave seed salt keeps octaves visually independent without
    // needing a different hash function for each layer.
    // Gain of 0.62 is higher than the textbook 0.5 — weather is closer
    // to a k^(-5/3) Kolmogorov spectrum than pure fBm, so the high
    // octaves get more weight than in a "smooth landscape" fBm. The
    // visible effect is sharper cumulus-scale texture instead of a
    // washed-out low-freq field.
    for (var i: u32 = 0u; i < octaves; i = i + 1u) {
        let s_lo = seed_lo ^ (i * 0x9E3779B9u);
        sum = sum + amp * cloud_value_noise_3d(p * freq, s_lo, seed_hi);
        norm = norm + amp;
        amp = amp * 0.62;
        freq = freq * 2.0;
    }
    return sum / max(norm, 1e-6);
}

/// Apply the differential-rotation drift to a body-local sample
/// direction before a cloud-cover cubemap lookup.
///
/// Rotates around the body's Y axis (north pole) by
///   `phase = t · scroll · (1 − diff · sin²(latitude))`
/// which is fastest at the equator and zero at the poles when
/// `diff = 1`. The latitude factor is computed from `dir.y` directly
/// (= sin(latitude) on the unit sphere). The caller must pass a unit
/// `dir`; the function returns a unit direction.
///
/// `diff` comes from `cloud_shape.w`, `scroll` from `cloud_dynamics.x`,
/// `t` from `cloud_dynamics.y`. When `scroll = 0` this is the identity
/// rotation (rigid bake, no drift).
fn rotate_cloud_dir_local(dir: vec3<f32>, layers: AtmosphereBlock) -> vec3<f32> {
    let scroll = layers.cloud_dynamics.x;
    let t = layers.cloud_dynamics.y;
    if scroll == 0.0 {
        return dir;
    }
    let lat = clamp(dir.y, -1.0, 1.0);
    let diff = clamp(layers.cloud_shape.w, 0.0, 1.0);
    let lat_factor = 1.0 - diff * lat * lat;
    let phase = t * scroll * lat_factor;
    let cp = cos(phase);
    let sp = sin(phase);
    return vec3<f32>(
        dir.x * cp - dir.z * sp,
        dir.y,
        dir.x * sp + dir.z * cp,
    );
}

/// Sample the thin cirrostratus/haze layer density.
///
/// Procedural: 8-octave fBm of Perlin gradient noise + domain warp +
/// high-frequency edge noise. The main-deck cumulus field is baked
/// (see `composite_clouds` for the cubemap fetch contract) but haze
/// is thin, broad, and uncorrelated with the main deck — the extra
/// per-fragment cost of keeping it procedural is small, and author-
/// time params (frequency, scroll, coverage cap) tune it without a
/// re-bake.
///
/// Physical identity:
///   - Higher base coverage (0.40) so haze is widespread
///   - Density capped at 0.35 so even peaks stay translucent
///     (paired with linear Beer-Lambert k=1.5 → peak opacity ~41%)
///   - No latitude / continentality / orographic bias — high
///     altitude decouples from surface geography
///   - Faster uniform drift (1.8× scroll) — upper-atmosphere feel
///   - Independent seed salt so haze pattern doesn't correlate
///     with cumulus underneath
fn sample_haze_density(
    sample_dir_local: vec3<f32>,
    layers: AtmosphereBlock,
) -> f32 {
    // TEMP: procedural haze disabled while the baked cube (currently a
    // storm-clouds reference photo) is the sole cloud source. Drop this
    // early-return when `thalos_cloud_gen` is back in charge of cloud
    // geometry — the fBm body below is the intended production path.
    return 0.0;

    let base_cov = layers.cloud_albedo_coverage.w;
    if base_cov <= 0.0 {
        return 0.0;
    }

    let main_freq = max(layers.cloud_shape.x, 1e-3);
    let freq = main_freq * 1.15;
    let softness = 0.15;
    let octaves_hint = max(u32(layers.cloud_shape.z), 1u);

    let scroll = layers.cloud_dynamics.x * 1.8;
    let t = layers.cloud_dynamics.y;
    let seed_lo = bitcast<u32>(layers.cloud_dynamics.z) ^ 0xC1CC1501u;
    let seed_hi = bitcast<u32>(layers.cloud_dynamics.w);

    // Uniform rotation — upper atmosphere has no differential.
    let phase = t * scroll;
    let cp = cos(phase);
    let sp = sin(phase);
    let rotated = vec3<f32>(
        sample_dir_local.x * cp - sample_dir_local.z * sp,
        sample_dir_local.y,
        sample_dir_local.x * sp + sample_dir_local.z * cp,
    );
    let p = rotated * freq;

    let slow_t = t * scroll * 0.2;
    let t_off = vec3<f32>(sin(slow_t * 1.1), cos(slow_t * 0.7), sin(slow_t * 1.3 + 0.4));

    // Same domain warp as cumulus (strength 0.6, 3 octaves per axis).
    let q = vec3<f32>(
        cloud_fbm(p + vec3<f32>(0.0, 0.0, 0.0) + t_off,
                  3u, seed_lo ^ 0xA1C37F19u, seed_hi),
        cloud_fbm(p + vec3<f32>(5.2, 1.3, 4.1) - t_off,
                  3u, seed_lo ^ 0x4B9D2C51u, seed_hi),
        cloud_fbm(p + vec3<f32>(2.8, 3.4, 8.2) + t_off.yzx,
                  3u, seed_lo ^ 0xD37AB602u, seed_hi),
    );
    let pwarp = p + 0.6 * q;

    // Same high-octave main fBm — this is what gives haze its
    // fractal detail parity with the cumulus layer.
    let main_octaves = max(octaves_hint, 8u);
    let mass = cloud_fbm(pwarp, main_octaves, seed_lo ^ 0xC0DE1234u, seed_hi);

    // Same symmetric edge noise — filamentary boundaries identical
    // in character to the cumulus layer.
    let edge_noise = cloud_fbm(pwarp * 2.2 + t_off * 0.4, 3u,
                               seed_lo ^ 0x7A3B1C5Du, seed_hi);
    let edge_bias = edge_noise * 1.2;

    let n_combined = clamp(mass * 3.0 + 0.5 + edge_bias, 0.0, 1.0);

    // Higher coverage than cumulus (0.40 vs ~0.25) — haze's
    // physical identity is "widespread translucent layer".
    let cov = 0.40;
    let threshold = 1.0 - cov;
    let raw_density = smoothstep(threshold, threshold + softness, n_combined);

    // Density cap 0.35 — the other half of the haze identity. Even
    // at peak this layer can't saturate, so linear Beer-Lambert with
    // k=1.5 in the compositor holds it below ~41% opacity permanently.
    return raw_density * 0.35;
}

/// Composite the cloud layer on top of an already-lit surface colour.
///
/// Pure Lambertian shading, no phase-function highlights. At orbital
/// scale clouds read as *matte* — bright white when lit, with only the
/// densest storm cores picking up any interior shading.
///
/// Main-deck density is **supplied by the caller** as two pre-sampled
/// scalars (`main_cloud_density`, `shadow_cloud_density`). The caller
/// is expected to:
///   1. Apply `rotate_cloud_dir_local` to both the cloud-shell
///      intersection direction (for `main_cloud_density`) and a sun-
///      offset shadow probe direction (for `shadow_cloud_density`).
///   2. Fetch the cloud-cover cubemap at both rotated directions.
/// The fetch lives caller-side because the cubemap binding is on the
/// impostor material, not in this library module. Parallax — the
/// visual cue that clouds float above the surface — is preserved by
/// the caller's choice of cloud-shell (vs surface) intersection when
/// resolving the main sample direction.
///
/// Coverage scaling: raw cubemap value × 2 × coverage makes the
/// authored `coverage` parameter an approximate fraction of the disk
/// that ends up overcast. The × 2 is because the raw Worley-fBm bake
/// peaks around 0.8 (most texels sit between 0.2 and 0.6), so a
/// `coverage = 0.5` authoring value scales most texels up into the
/// `[0.2, 1.0]` visible range.
///
/// `sun_flux_scaled` is the caller's pre-normalised sunlight
/// contribution (e.g., `sun_flux * hapke_scale`) so cloud brightness
/// stays in photometric lockstep with the surface beneath.
fn composite_clouds(
    surface_lit: vec3<f32>,
    cloud_sample_dir_local: vec3<f32>,
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

    // Coverage-scaled densities. Raw Worley fBm peaks around 0.8; the
    // ×2×coverage map lets the author's `coverage` value approximate
    // the fraction of the disk that ends up overcast.
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

    // ── Cirrostratus/haze layer ─────────────────────────────────────
    //
    // Higher-altitude thin layer composited ON TOP of the cumulus
    // (view ray from camera hits haze first). Produces the broad
    // translucent coverage missing from the cumulus-only render —
    // wispy torn-sheet cloud that lets surface + cumulus show through.
    // Always applied regardless of whether the main cumulus is
    // present, so it fills the "empty" areas the single layer leaves.
    let haze_density = sample_haze_density(cloud_sample_dir_local, layers);
    if haze_density >= 1e-3 {
        // Slightly wider wrap than cumulus — high altitude stays lit
        // past the geometric terminator a bit longer. No self-shadow:
        // the layer is thin enough that interior darkening would read
        // as noise rather than structure.
        let haze_wrap = 0.25;
        let haze_ndl = clamp((raw_ndl + haze_wrap) / (1.0 + haze_wrap), 0.0, 1.0);
        let haze_sun = albedo * haze_ndl * sun_flux_scaled;
        let haze_amb = albedo * ambient * 0.25;
        let haze_lit = haze_sun + haze_amb;

        // Low k — at density cap 0.35 opacity peaks at ~41%, typical
        // haze (density 0.20) sits at ~26%. Always translucent, never
        // paints over the surface like the previous k=2.0 did.
        let haze_k = 1.5;
        let haze_opacity = clamp(1.0 - exp(-haze_density * haze_k), 0.0, 1.0);
        result = mix(result, haze_lit * night_suppress,
                     haze_opacity * night_suppress);
    }

    return result;
}
