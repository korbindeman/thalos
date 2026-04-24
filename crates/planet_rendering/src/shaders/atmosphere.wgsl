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

/// Packed uniform carrying every optional atmosphere layer.
///
/// Every layer is gated by its own intensity/strength scalar: the shader
/// is expected to early-out when the scalar is zero so a body without an
/// atmosphere (Mira, Ignis, …) pays no extra cost beyond a couple of
/// scalar comparisons.
struct AtmosphereBlock {
    /// Rim halo colour and intensity.
    ///   xyz = linear-RGB colour at peak,
    ///   w   = intensity scalar. `0` disables the rim halo entirely.
    rim_color_intensity: vec4<f32>,
    /// Rim halo shape, in render units (pre-scaled by the CPU).
    ///   x = scale height,
    ///   y = outer-shell altitude (planet_radius + y = shell outer
    ///       radius),
    ///   z = reserved for a future ground-altitude offset,
    ///   w = reserved.
    rim_shape: vec4<f32>,
    /// Terminator warmth (sunset band).
    ///   xyz = linear-RGB tint applied near `n_dot_l ≈ 0` on the lit
    ///         side,
    ///   w   = strength scalar. `0` disables the warmth band.
    terminator_warmth: vec4<f32>,
    /// Fresnel rim on the lit limb (cold Rayleigh stand-in).
    ///   xyz = linear-RGB tint,
    ///   w   = strength scalar. `0` disables the Fresnel rim.
    fresnel_rim: vec4<f32>,
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
    /// Per-wavelength Rayleigh scattering.
    ///   xyz = vertical optical depth at zenith (R, G, B),
    ///   w   = overall strength multiplier. `0` disables Rayleigh
    ///         entirely and the surface renders with unattenuated
    ///         white sunlight.
    rayleigh: vec4<f32>,
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

/// Result of a rim-halo column integration.
struct RimHit {
    contribution: vec3<f32>,
    opacity: f32,
}

fn no_rim() -> RimHit {
    return RimHit(vec3<f32>(0.0), 0.0);
}

/// Rim-halo contribution for a ray that may or may not hit the body.
///
/// Integrates an exponential-density atmospheric column between the
/// body's inner sphere (radius `inner_r`) and the outer shell (radius
/// `inner_r + outer_alt`). Returns a coloured contribution and an
/// opacity for the compositor. Mirrors `rim_halo_contribution` in
/// `gas_giant.wgsl`, parameterised instead of reading globals.
///
/// The returned `contribution` is *not* scaled by the caller's light
/// flux; the caller multiplies by `sun_flux * (1 / 4π)` at the
/// compositing site so the normalisation matches the cloud-deck /
/// surface lighting budget already in that shader.
fn rim_halo_contribution(
    cam_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    light_dir_ws: vec3<f32>,
    inner_r: f32,
    outer_alt: f32,
    scale_h: f32,
    color: vec3<f32>,
    intensity: f32,
) -> RimHit {
    if intensity <= 0.0 || outer_alt <= 0.0 {
        return no_rim();
    }

    let r_inner = inner_r;
    let r_outer = inner_r + outer_alt;

    // Ray-sphere intersection against the outer shell.
    let oc = cam_pos - center;
    let half_b = dot(oc, ray_dir);
    let c_outer = dot(oc, oc) - r_outer * r_outer;
    let disc_o = half_b * half_b - c_outer;
    if disc_o < 0.0 {
        return no_rim();
    }
    let sq_o = sqrt(max(disc_o, 0.0));
    let t0 = -half_b - sq_o;
    let t1 = -half_b + sq_o;
    let t_entry = max(t0, 0.0);
    if t1 <= t_entry {
        return no_rim();
    }

    // Closest approach altitude.
    let closest = oc + ray_dir * (-half_b);
    let closest_d = length(closest);
    let shell_span = max(r_outer - r_inner, 1e-6);
    let closest_alt = clamp((closest_d - r_inner) / shell_span, -1.0, 1.0);

    // Exponential falloff with altitude. `scale_h` is in render units.
    let sh = max(scale_h, 1e-4);
    let rel_h = (closest_alt * shell_span) / sh;
    let density = exp(-max(rel_h, 0.0));

    // Column length along the ray within the shell; clamp at the
    // inner sphere if the ray hits it.
    let c_inner = dot(oc, oc) - r_inner * r_inner;
    let disc_i = half_b * half_b - c_inner;
    var t_exit = t1;
    if disc_i >= 0.0 {
        let ti = -half_b - sqrt(max(disc_i, 0.0));
        if ti > t_entry {
            t_exit = min(t_exit, ti);
        }
    }
    let column = max(t_exit - t_entry, 0.0);

    // Sun factor: the halo only appears on the lit side of the body.
    let closest_dir = normalize(oc + ray_dir * (-half_b));
    let sun_factor = clamp(dot(closest_dir, light_dir_ws) * 0.5 + 0.5, 0.0, 1.0);

    // Normalise the column by the shell thickness so intensity stays
    // consistent as `outer_altitude_m` is tuned, and clamp the path
    // factor so grazing rays don't peg the tonemapper.
    let column_norm = column / shell_span;
    let path_factor = column_norm / (1.0 + column_norm);
    let strength = clamp(intensity * density * sun_factor * path_factor, 0.0, intensity);
    return RimHit(color * strength, min(strength, 1.0));
}

/// Apply a terminator-warmth tint near the day/night line on the lit
/// hemisphere. Strength 0 returns `base` unchanged.
fn apply_terminator_warmth(
    base: vec3<f32>,
    n_dot_l: f32,
    tint: vec3<f32>,
    strength: f32,
) -> vec3<f32> {
    if strength <= 0.0 {
        return base;
    }
    let lit_side = smoothstep(-0.05, 0.10, n_dot_l);
    let near_zero = 1.0 - smoothstep(0.05, 0.30, n_dot_l);
    let w = clamp(lit_side * near_zero * strength, 0.0, 1.0);
    return mix(base, base * tint, w);
}

/// Apply a Fresnel-style tint to the lit limb (cool Rayleigh-like
/// rim). Gated by `n_dot_l²` so the unlit limb stays dark. Strength 0
/// returns `base` unchanged.
fn apply_fresnel_rim(
    base: vec3<f32>,
    n_dot_l: f32,
    n_dot_v: f32,
    tint: vec3<f32>,
    strength: f32,
) -> vec3<f32> {
    if strength <= 0.0 {
        return base;
    }
    let fresnel = pow(1.0 - clamp(n_dot_v, 0.0, 1.0), 4.0);
    let lit_gate = clamp(n_dot_l, 0.0, 1.0);
    let w = clamp(fresnel * lit_gate * lit_gate * strength, 0.0, 1.0);
    return mix(base, base * tint, w);
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

// ── Rayleigh scattering ─────────────────────────────────────────────────────
//
// Two complementary effects from one physical parameter set (per-channel
// vertical optical depth `τ_v`):
//
//   1. Ground transmission `T(λ) = exp(-τ_v(λ) · airmass(μ_sun))`. Multiplies
//      the direct-sun component of the lit surface. At high sun the
//      transmission is near-white; as the sun lowers, blue dies first and
//      the surface is progressively lit by redder light — which is what
//      actually produces sunset colours (not an ad-hoc tint).
//
//   2. View-path in-scatter `β(λ) · T_sun(λ) · airmass_view · sun_flux`.
//      β is proportional to `τ_v`. At the sub-solar point this reads blue
//      (β_blue large, T_blue still decent); at the terminator the long
//      sun column eats blue entirely and the orange/red β × T product
//      dominates — this is the terminator ring seen from orbit.
//
// The airmass approximation below is a Chapman-function soft clamp rather
// than pure `1/μ`: `1 / (μ + 0.15 · (1.65 - acos(μ)/π·180·k)^-1.253)` is
// the Kasten-Young formula; here we use the simpler `1/max(μ, 0.02)` clamp
// (asymptote 50× at μ=0.02 ≈ 88.8° zenith), which is fast and
// well-behaved through the terminator.

fn rayleigh_airmass(cos_zenith: f32) -> f32 {
    return 1.0 / max(cos_zenith, 0.02);
}

/// Multiply the lit surface by the per-channel Rayleigh illumination
/// the ground actually receives: direct sun (attenuated by airmass)
/// PLUS the fraction of scattered-out light that returns via the sky.
///
/// The naive formula `base * exp(-τ · airmass)` over-reddens the lit
/// disk because it assumes any blue photon scattered out of the sun's
/// beam is lost to the viewer. In reality most of it reaches the
/// ground indirectly via multiple-scatter through the sky, and the
/// ground under a moderate sun reads mostly neutral — only the
/// narrow band near the terminator genuinely reddens. Model:
///
///   effective(λ) = T(λ) + α · (1 − T(λ))
///
/// where α is the "sky-diffuse recovery fraction" — the share of the
/// scattered-out light at wavelength λ that returns to this ground
/// pixel via the sky hemisphere. α ≈ 0.45 is a first-order fit to
/// full radiative-transfer solutions under an isotropic-scattering
/// assumption, and matches the broad visual feel of Apollo-era
/// Earth-from-orbit imagery: near-natural ground under a high sun,
/// a concentrated red band within ~15–20° of the terminator.
///
/// α = 0 recovers the naive (over-reddening) formula; α = 1 is
/// indistinguishable from no atmosphere at all.
fn apply_rayleigh_ground_transmission(
    base: vec3<f32>,
    n_dot_l: f32,
    tau_v_rgb: vec3<f32>,
    strength: f32,
) -> vec3<f32> {
    if strength <= 0.0 {
        return base;
    }
    let airmass = rayleigh_airmass(n_dot_l);
    let t = exp(-tau_v_rgb * airmass);
    let alpha = 0.45;
    let effective = t + alpha * (vec3<f32>(1.0) - t);
    let gate = smoothstep(-0.05, 0.20, n_dot_l);
    return mix(base, base * effective, gate * strength);
}

/// Additive Rayleigh in-scatter along the view path. Per-wavelength tint
/// is naturally blue at the sub-solar point and orange/red at the
/// terminator — see the module header for the derivation. `β` is
/// proportional to `τ_v`, so we pass `τ_v` alone and scale it here.
///
/// Uses `(1 − exp(−τ · airmass_view))` as the view-path opacity per
/// channel. This is the fraction of the column occupied by atmosphere
/// from the viewer's perspective, saturating smoothly at the limb
/// (where airmass_view → large and the opacity approaches 1) instead
/// of blowing up like naive `airmass_view` would. At the sub-solar
/// point it's the vertical opacity `1 − exp(−τ)` — small, but nonzero,
/// so the daylight blue haze IS visible across the whole disk. That
/// "visible everywhere, stronger at the limb" distribution is what
/// reads as atmosphere in orbital photos.
///
/// `sun_flux_scaled` is the caller's pre-normalised sunlight contribution
/// (e.g. `sun_flux * hapke_scale`) so haze brightness tracks surface
/// brightness as the sun distance or luminosity changes.
fn apply_rayleigh_inscatter(
    base: vec3<f32>,
    n_dot_l: f32,
    n_dot_v: f32,
    tau_v_rgb: vec3<f32>,
    strength: f32,
    sun_flux_scaled: f32,
) -> vec3<f32> {
    if strength <= 0.0 {
        return base;
    }
    let airmass_sun = rayleigh_airmass(n_dot_l);
    let t_sun = exp(-tau_v_rgb * airmass_sun);
    let airmass_view = rayleigh_airmass(n_dot_v);
    // View-path opacity per channel. Saturates at the limb instead of
    // exploding; equals `1 − exp(−τ_v)` at zenith, which is what
    // gives the sub-solar point a faint blue haze.
    let view_opacity = vec3<f32>(1.0) - exp(-tau_v_rgb * airmass_view);
    // Soft lit-gate: the in-scatter ring reaches slightly past the
    // geometric terminator (atmosphere above the surface is lit even
    // when the surface is not) — hence the -0.15 lower edge.
    let lit_gate = smoothstep(-0.15, 0.30, n_dot_l);
    // Phase-function / isotropic-scatter normalisation. Of the flux
    // Rayleigh-scattered out of a view-path column, only `1/(4π)` of
    // it per steradian ends up travelling toward the viewer. Without
    // this factor a physical `τ_v = 0.264` on blue produces a haze
    // radiance brighter than the lit ground — the "whole planet
    // bathed in cyan" look. The proper Rayleigh phase function is
    // `3/(16π) · (1 + cos²θ)`, ranging 0.06 → 0.12; the isotropic
    // constant picks the middle-of-the-range value, which is close
    // enough for typical orbital viewing geometries and avoids
    // plumbing `cos_phase` through the call site.
    let phase_norm = 1.0 / (4.0 * PI);
    let w = lit_gate * strength * sun_flux_scaled * phase_norm;
    return base + view_opacity * t_sun * w;
}

/// Returns true if any atmosphere layer carries a non-zero strength.
/// The shader can skip the entire atmosphere path when this is false.
fn atmosphere_is_active(layers: AtmosphereBlock) -> bool {
    return layers.rim_color_intensity.w > 0.0
        || layers.terminator_warmth.w > 0.0
        || layers.fresnel_rim.w > 0.0
        || layers.limb_exponents.w > 0.0
        || layers.cloud_albedo_coverage.w > 0.0;
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
