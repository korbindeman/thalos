// Scattering model shared by every participating medium in Thalos.
//
// Clouds, vapour cones, fog, dust and (later) contrails are all the same physics
// with different densities and albedos: light scatters through a strongly
// forward-peaked phase function, scatters again many times, and leaves an
// optically thick sunlit volume close to isotropically. Anything that renders
// one of those media with a single scattering lobe reproduces a specific,
// already-diagnosed defect — see *Why media are white* below.
//
// **This library owns the radiance terms, not the march.** Every medium's loop
// stays its own: the cloud system marches a planetary shell with tiering, a
// coverage-calibrated fill LUT and temporal amortization; a vapour cone marches a
// twenty-metre analytic envelope. Those are not variants of one loop, and a
// generic `march(field, bounds)` in WGSL — which has no dynamic dispatch — buys a
// mega-shader with dead branches and puts the heaviest pass in the frame at risk.
// Share the terms; keep the loop local. (ADR-20260730T034500Z.)
//
// Extracted from `clouds_compute.wgsl` 2026-07-30 with no change to the cloud
// look: same functions, same constants, same call order.

#define_import_path thalos::volumetrics

// EXPORTED VALUES TRAVEL AS FUNCTIONS, NOT `const`.
//
// naga_oil accepts `#import thalos::volumetrics::{SOME_CONST}` and rewrites every
// use to a mangled name, but it only composes **functions and types** into the
// final module — the `const` itself is never carried across, so the reference
// dangles and the pipeline dies at creation with "no definition in scope for
// identifier". The consts below are therefore module-local, and anything a
// consumer needs is exposed by an accessor. See the `wgsl-bevy` skill.

const INV_PI: f32 = 0.31830989;
const INV_FOUR_PI: f32 = 0.07957747;

// ── Why media are white ──────────────────────────────────────────────────────
// A water cloud scatters conservatively (ϖ ≈ 0.9999) through a strongly
// forward phase, so an optically thick sunlit cell reaches the diffusion
// limit: it returns ~0.75–0.85 of the incident flux, and that light leaves
// close to isotropically. Its radiance is therefore ≈ A·E/π, roughly SIX
// times the single-scattering side lobe (p(90°)·E ≈ 0.043·E here).
//
// Single scattering alone renders a medium no brighter than the sky ambient
// filling it — grey-blue mud that takes its chroma from the ambient rather
// than the sun. That was the defect: the octave sum was divided by Σw, which
// makes it a weighted AVERAGE of phase values, so the "multiple-scattering
// octaves" only reshaped the lobe and added none of the energy they name.
// Measured on `cloud_cruise` (2026-07-24 capture): brightest near-tier cloud
// pixel 0.30 display luminance against 0.49–0.73 for the sky behind it and
// 0.73 for the far tier's rendering of the same field.
//
// The replacement splits the source term the way the physics does:
//   single  — exact normalized phase against the unattenuated beam; owns the
//             forward glare and the silver lining;
//   multi   — an isotropic-equivalent reservoir at the medium's diffusion
//             albedo, whose DEPTH response is the surviving job of the wider
//             octaves (they attenuate far more slowly, so light leaks around
//             occluders instead of multiplying cores to charcoal).
//
// The albedo is a **property of the medium, not a brightness knob**: it is what
// makes a lit cell as bright as a white Lambertian surface facing the same sun.
// It is therefore a PARAMETER here — water, ice and dust differ — while the
// canonical values live below so no consumer invents its own.

const WATER_CLOUD_ALBEDO: f32 = 0.80;
const DUST_ALBEDO: f32 = 0.35;

/// Diffusion-limit reflectance of a water cloud.
///
/// One number, one definition. It used to be written twice — `CLOUD_MS_ALBEDO`
/// in the near march and `FAR_CLOUD_ALBEDO` in the composite, each with a comment
/// saying it MUST equal the other. That is the drift hazard the shared-library
/// rule exists to remove: both tiers are anchored to this value, and a silent
/// divergence makes the near and far renderings of the same field disagree.
fn water_cloud_albedo() -> f32 {
    return WATER_CLOUD_ALBEDO;
}

/// Diffusion-limit reflectance of a lofted mineral dust volume. Dust absorbs —
/// it is emphatically not white — which is the whole reason albedo is a
/// parameter of [`volumetric_scattering`] rather than a constant baked into it.
fn dust_albedo() -> f32 {
    return DUST_ALBEDO;
}

/// Residual anisotropy of the multiply-scattered reservoir: it is not perfectly
/// isotropic — the sun side of a cell stays brighter. 0 = flat, 1 = the widest
/// octave's full lobe.
const MS_ANISO: f32 = 0.7;

// Octave energy weights and shadow-attenuation exponents. Energy drops per
// octave; attenuation drops faster so multiple scattering "leaks around"
// occluders (Wrenninge/Nubis approximation). Keep the higher octaves modest:
// at (1.0, 0.52, 0.26) deep shade retained ~40% of lit energy and lobes went
// flat cotton with no readable sun side.
const MS_OCTAVE_WEIGHTS: vec3f = vec3f(1.0, 0.34, 0.13);
const MS_OCTAVE_EXTINCTION: vec3f = vec3f(1.0, 0.25, 0.06);

fn henyey_greenstein(ray_dot_sun: f32, g: f32) -> f32 {
    let g_squared = g * g;
    return (1.0 - g_squared) / pow(1.0 + g_squared - 2.0 * g * ray_dot_sun, 1.5);
}

/// Per-octave dual-lobe phase values (Nubis/Frostbite multi-scatter octaves).
/// Evaluated ONCE PER RAY; per sample each octave is attenuated by its own
/// `exp(-τ_sun · c_i)`, so deep shade retains soft wide-lobe fill from the
/// later octaves instead of multiplying the whole direct term toward black
/// (the old `0.04 + 0.96 · shadow` fill collapsed shaded cores to charcoal).
fn multi_scatter_lobes(cos_theta: f32, g_fwd: f32, g_bwd: f32, lerp_g: f32) -> vec3f {
    var lobes = vec3f(0.0);
    var gf = g_fwd;
    var gb = g_bwd;
    for (var i = 0; i < 3; i++) {
        let lobe = mix(
            henyey_greenstein(cos_theta, gf),
            henyey_greenstein(cos_theta, gb),
            lerp_g,
        );
        // HG omits 1/(4π); normalize into scene units and bound the forward peak.
        lobes[i] = min(lobe * INV_FOUR_PI, 2.2);
        gf *= 0.5;
        gb *= 0.5;
    }
    return lobes;
}

/// Per-octave sun transmittance from one filtered sun optical depth.
fn octave_shadow(tau_sun: f32) -> vec3f {
    return vec3f(
        exp(-tau_sun * MS_OCTAVE_EXTINCTION.x),
        exp(-tau_sun * MS_OCTAVE_EXTINCTION.y),
        exp(-tau_sun * MS_OCTAVE_EXTINCTION.z),
    );
}

/// Scattering coefficient at one sample: single + multiple, for a medium of the
/// given diffusion albedo.
///
/// `lobes` comes from [`multi_scatter_lobes`] (once per ray); `tau_sun` is the
/// filtered optical depth toward the sun at this sample. Multiply the result by
/// the incident sun colour/transmittance and the powder term.
fn volumetric_scattering(lobes: vec3f, tau_sun: f32, albedo: f32) -> f32 {
    let shadow = octave_shadow(tau_sun);
    // Single scattering: exact normalized phase against the beam that actually
    // survives to this sample.
    let single = lobes.x * shadow.x;
    // Multiple scattering: the diffusion reservoir. The wider octaves no longer
    // carry phase energy — they supply the reservoir's depth response, which is
    // the part of them that was ever physical.
    let ms_depth =
        dot(MS_OCTAVE_WEIGHTS.yz, shadow.yz) / (MS_OCTAVE_WEIGHTS.y + MS_OCTAVE_WEIGHTS.z);
    let ms_aniso = mix(1.0, lobes.z / INV_FOUR_PI, MS_ANISO);
    let multi = albedo * INV_PI * ms_depth * ms_aniso;
    return single + multi;
}

/// Silver-lining / powder: thin edges facing the light brighten; the same thin
/// path looking *away* from the light darkens (HZD powder). Restrained: the
/// former 0.85 away-darkening painted lobes near-black and read as dirt rather
/// than shading. Caveat: the 0.35/0.35 constants (and MS_ANISO above) were
/// tuned while the phase argument was negated — that "dirt" was landing on the
/// SUNWARD side — so they are retune candidates against sunset captures now
/// that the geometry is correct.
fn powder_term(density_fraction: f32, cos_theta: f32) -> f32 {
    let d = clamp(density_fraction, 0.0, 1.0);
    let powder = 1.0 - exp(-d * 2.0);
    // cos_theta = ray·sun: +1 looking toward the sun (silver lining).
    let toward_sun = clamp(cos_theta, 0.0, 1.0);
    let away = clamp(-cos_theta, 0.0, 1.0);
    return mix(1.0, powder, away * 0.35) * (1.0 + toward_sun * d * 0.35);
}

/// Ambient self-occlusion: a deep interior sample sees far less sky than a
/// fringe. Without it, a physical-magnitude sky ambient flattens every lobe into
/// one pale sheet. Driven by the same filtered sun depth the octaves use — a
/// correlated stand-in for sky visibility that costs no extra probe.
fn ambient_occlusion(tau_sun: f32) -> f32 {
    return 0.30 + 0.70 * exp(-tau_sun * 0.45);
}
