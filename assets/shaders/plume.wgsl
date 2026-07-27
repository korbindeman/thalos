// Rocket-engine exhaust plume — an axisymmetric emitting gas column whose shape
// is set by the nozzle/ambient pressure ratio and whose brightness *follows from
// that shape* rather than from an authored fade curve.
//
// The model, in one place:
//
//   R(s)  = R0·lip + tan(theta)·s          free-expansion cone off the nozzle lip
//           × barrel(s)                    shock-cell pinching (needs back-pressure)
//   rho   ∝ (R0/R)²                        mass conservation along the column
//   T     ∝ T_exit · (R0/R)^(2(gamma-1))   adiabatic (expansion) cooling
//   T    ×= exp(-e·s/R0)                   entrainment cooling (atmosphere only)
//   S     = exp(-W·(1/T − 1))              visible-band emission, Wien side
//   tau   ∝ rho · chord                    optical depth across the line of sight
//   L     = S · (1 − exp(−tau))            emission through an absorbing column
//
// Three consequences make this cover pad-to-orbit with no regime switch:
//
//  * Near the nozzle tau >> 1, so the core saturates to a flat, blindingly bright
//    column — the sea-level look (dense, shock-celled).
//  * In vacuum the column cools by expanding, and the exponential Wien term
//    collapses, so the cone dissolves on its own with no tip taper.
//  * At sea level it barely expands, so *entrainment* of ambient air is what
//    cools it along its length. Without that term a sea-level plume stays
//    uniformly incandescent end to end.
//
// ONE LENGTH AUTHORITY. The billboard is cut exactly where `L` above — summed
// over *both* layers and scaled by the gain — falls below an absolute visibility
// floor; the CPU solves this same chain for that station (`visible_length_m` in
// `plume.rs`). Never cap the mesh length by anything the fragment stage cannot
// see: a limit the emission model doesn't know about ends the geometry while the
// column is still incandescent, which is a flat lit rim hanging in mid-air. See
// INC-20260724T235437Z-plume-ended-on-a-lit-rim.
//
// A turbulent shear layer (the sheath) wraps the core: cool and violet in vacuum,
// hot and orange in atmosphere where the fuel-rich exhaust afterburns with
// entrained air. Everything about the response lives CPU-side in
// `crate::rendering::plume`; this shader renders the resolved profile.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}

// Resolved per-engine plume profile. Mirrors `PlumeParams` (Rust, ShaderType);
// packed as vec4s so the std140 layout is unambiguous.
struct PlumeParams {
    // rgb = hot core colour (near-nozzle / shock nodes), a = HDR radiance scale.
    core_color: vec4<f32>,
    // rgb = expanded-plume colour, a = nozzle exit radius R0 (m).
    mid_color: vec4<f32>,
    // rgb = shear-layer / sheath colour, a = visible axial length L (m).
    edge_color: vec4<f32>,
    // x = lip radius scale, y = tan(core half-angle),
    // z = shear-layer spread rate, w = adiabatic exponent 2(gamma-1).
    shape: vec4<f32>,
    // x = core opacity kappa, y = sheath opacity kappa,
    // z = shock-cell wavenumber (rad/m), w = shock strength 0..1.
    shock: vec4<f32>,
    // x = shock decay length (m), y = afterburn 0..1,
    // z = turbulence amplitude, w = RESERVED (throttle; no longer read here —
    // the CPU folds throttle into kappa, entrainment and the flicker rate).
    mixing: vec4<f32>,
    // x = time (s), y = per-engine seed, z = ignition 0..1,
    // w = entrainment cooling rate (per nozzle radius; 0 in vacuum).
    anim: vec4<f32>,
    // Turbulent motion. x = eddy growth per axial metre,
    // y = convection rate (eddies/s), z = azimuthal swirl (rad/s),
    // w = radial wobble amplitude.
    flow: vec4<f32>,
    // x = tail dispersal growth, y = potential-core length (m),
    // z = flicker amplitude, w = flicker rate (Hz).
    tail: vec4<f32>,
    // x = exit temperature (normalized; ignition transient),
    // y = core turbulence weight, z = tail turbulence boost,
    // w = shock-cell lengthening per axial metre.
    therm: vec4<f32>,
}

@group(3) @binding(0) var<uniform> plume: PlumeParams;

// -- cheap hash / value noise for the turbulent shear layer ------------------

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// -- emission ---------------------------------------------------------------

// Dimensionless Wien parameter hc/(lambda·k·T_exit) for the visible band at
// combustion-chamber temperature. Larger = the plume darkens faster as it cools.
// MIRRORED in `plume.rs` (`WIEN`) — the CPU solves this same curve for the mesh
// length, so the two constants must stay equal.
const WIEN: f32 = 3.0;

const HALF_PI: f32 = 1.5707964;

// Visible-band emission from a cooling gas, normalised to 1 at chamber
// temperature.
//
// This is deliberately *not* a grey-body T^4. A rocket plume radiates in the
// visible from the Wien side of the Planck curve, where output falls off
// exponentially in 1/T rather than polynomially — which is why a real plume goes
// dark over a few nozzle diameters instead of trailing off forever. It is also
// what lets the geometry end without a lit disc at the tip: by the time the
// column has cooled enough, this term has already collapsed.
fn band_emission(temp_norm: f32) -> f32 {
    return exp(-WIEN * (1.0 / max(temp_norm, 1e-3) - 1.0));
}

// -- turbulent structure ----------------------------------------------------

// Dimensionless "eddy count" from the exit plane: turbulent structures in a jet
// grow linearly with distance, so integrating ds/eddy_size(s) gives a coordinate
// in which they are all the *same* size. Sampling noise on this coordinate
// instead of on raw distance is what makes structures coarsen as they travel
// downstream — the strongest single cue that this is a turbulent column and not
// a scrolling texture.
fn eddy_coord(s: f32) -> f32 {
    let g = max(plume.flow.x, 1e-3);
    let e0 = max(plume.mid_color.a * plume.shape.x, 1e-3);
    return log(1.0 + g * s / e0) / g;
}

// Low-frequency combustion roughness. Only ever *dims* and *shortens* the
// column, never lengthens it, so the visible plume always stays inside the mesh
// the CPU sized for the unflickered state.
fn flicker() -> f32 {
    let n = value_noise(vec2<f32>(plume.anim.x * plume.tail.w, plume.anim.y * 53.0));
    return 1.0 - plume.tail.z * n;
}

// Progress from the end of the potential core to the visible tip. 0 inside the
// coherent near field, 1 where the jet has fully broken down.
fn breakup(s: f32) -> f32 {
    let core_len = max(plume.tail.y, 1e-3);
    let len = max(plume.edge_color.a, core_len * 1.001);
    return smoothstep(core_len, len, s);
}

// Three independently-advected noise layers. Real exhaust does not slide as one
// rigid pattern: the shear layer convects slower than the core and the large
// downstream structures slower still, so the composite never reads as a single
// repeating scroll. Each layer also drifts azimuthally at its own rate, so the
// column rotates as well as flows.
fn turbulence(phi: f32, s: f32, b: f32) -> f32 {
    let xi = eddy_coord(s);
    let now = plume.anim.x;
    let adv = plume.flow.y;
    let swirl = plume.flow.z;
    let seed = plume.anim.y * 31.0;

    // Fine striations riding the core, the dominant texture of a dense jet.
    let fine = value_noise(vec2<f32>(
        phi * 7.0 + seed + swirl * now,
        xi * 1.00 - adv * now,
    ));
    // Shear-layer billows: finer around the circumference, convecting slower
    // because the mixing layer is subsonic relative to the core.
    let mid = value_noise(vec2<f32>(
        phi * 15.0 + seed * 1.7 - swirl * 1.9 * now,
        xi * 2.30 - adv * 0.62 * now,
    ));
    // Large, slow puffs — what a broken-down jet disperses into.
    let puff = value_noise(vec2<f32>(
        phi * 3.5 + seed * 0.6 + swirl * 0.5 * now,
        xi * 0.45 - adv * 0.33 * now,
    ));

    let near = 0.62 * fine + 0.38 * mid;
    return mix(near, puff, clamp(0.18 + 0.34 * b, 0.0, 1.0));
}

// -- plume geometry (shared by both stages so the mesh silhouette *is* the
//    analytic envelope — no wasted transparent strip, no edge mismatch) ------

// Phase along the shock train. Cells *lengthen* downstream as the train weakens
// — a constant wavenumber gives an evenly-spaced ladder of identical rungs,
// which is the one thing a real Mach-diamond train never looks like.
fn shock_phase(s: f32) -> f32 {
    let g = max(plume.therm.w, 1e-6);
    return plume.shock.z * log(1.0 + g * s) / g;
}

// Shock-cell phase modulation, decaying downstream. 1 = unmodulated.
fn barrel(s: f32) -> f32 {
    let strength = plume.shock.w;
    let decay = exp(-s / max(plume.mixing.x, 1e-3));
    return 1.0 + 0.10 * strength * decay * cos(shock_phase(s));
}

// Luminous core radius at axial station s (metres from the exit plane).
fn core_radius(s: f32) -> f32 {
    let r0 = plume.mid_color.a;
    let cone = r0 * plume.shape.x + plume.shape.y * s;
    return max(cone * barrel(s), 1e-4);
}

// Travelling constriction/bulge: eddies passing down the shear layer pinch and
// swell the column, so the silhouette boils instead of being a perfect analytic
// curve. A function of `s` alone, so the vertex and fragment stages agree
// exactly and the mesh edge stays *on* the envelope.
fn radius_wobble(s: f32, b: f32) -> f32 {
    let amp = plume.flow.w * (0.25 + 0.75 * b);
    let xi = eddy_coord(s);
    let n = value_noise(vec2<f32>(
        plume.anim.y * 19.0,
        xi * 0.8 - plume.anim.x * plume.flow.y * 0.8,
    ));
    return 1.0 + amp * (n - 0.5) * 2.0;
}

// Outer edge of the turbulent shear layer. Grows linearly with entrainment while
// the potential core survives; once the shear layer has eaten the core the jet
// *disperses* — growth accelerates and the column flares open, so it dissolves
// into widening structure rather than ending on a rim. Plus a thin lip so the
// sheath is never zero-width at the nozzle.
fn sheath_radius(s: f32) -> f32 {
    let r0 = plume.mid_color.a;
    let b = breakup(s);
    let spread = plume.shape.z * s * (1.0 + plume.tail.x * b * b);
    return (core_radius(s) + spread + r0 * 0.05) * radius_wobble(s, b);
}

// Compression-node brightness: peaks exactly where `barrel` pinches the column.
fn shock_node(s: f32) -> f32 {
    let strength = plume.shock.w;
    let decay = exp(-s / max(plume.mixing.x, 1e-3));
    return strength * decay * pow(max(0.0, -cos(shock_phase(s))), 8.0);
}

// -- vertex: cylindrical billboard whose width tracks the envelope ----------

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // x = axial fraction t (0..1), y = lateral fraction of the local sheath radius.
    @location(0) plume_uv: vec2<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    // Nozzle exit (plume-local origin) and exhaust axis (local -Y) in world space.
    let center = world_from_local[3].xyz;
    let axis = normalize((world_from_local * vec4<f32>(0.0, -1.0, 0.0, 0.0)).xyz);

    // Rotate the flat strip about the axis to face the camera.
    let to_cam = view.world_position - center;
    var right = cross(axis, to_cam);
    let right_len = length(right);
    if (right_len < 1e-4) {
        // Camera nearly on-axis: pick any stable perpendicular so the quad stays
        // finite (it degenerates to a sliver, which is correct end-on).
        let alt = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(axis.x) > 0.9);
        right = normalize(cross(axis, alt));
    } else {
        right = right / right_len;
    }

    let t = in.position.y;               // 0 exit -> 1 tip
    let s = t * plume.edge_color.a;      // metres from the exit plane
    let world_pos = center + axis * s + right * (in.position.x * sheath_radius(s));

    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world_pos);
    out.plume_uv = vec2<f32>(t, in.position.x);
    return out;
}

// -- fragment: emission through an absorbing, expanding column --------------

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = clamp(in.plume_uv.x, 0.0, 1.0);
    let x = clamp(in.plume_uv.y, -1.0, 1.0);

    let r0 = plume.mid_color.a;
    let s = t * plume.edge_color.a;
    let b = breakup(s);

    // The mesh edge *is* the sheath boundary, so `lateral` is exact. The core is
    // clamped inside it because the wobble perturbs only the sheath.
    let rs = sheath_radius(s);
    let rc = min(core_radius(s), rs * 0.94);
    let lateral = abs(x) * rs;

    // Line-of-sight integrals through the two layers.
    //
    // Neither is a top-hat cylinder. A saturated top-hat — where `1 - exp(-tau)`
    // is already 1 across the whole width — has a *razor* silhouette, because
    // the only thing that can end it is the chord going to zero, and sqrt() does
    // that with infinite slope. That is what made the core read as a fluorescent
    // tube and the shear layer as a hard-edged cone. Both layers therefore use
    // radial density profiles with compact support: smooth, and exactly zero at
    // the mesh boundary, so the silhouette feathers no matter how optically
    // thick the column gets.
    //
    // Core: rho ∝ (1 - (r/R)²)^(1/2), a near-flat plug with a soft shoulder.
    //   ∫ along the chord at impact parameter p = (pi/2)·R·(1 - (p/R)²).
    let fc = max(1.0 - (lateral * lateral) / (rc * rc), 0.0);
    let chord_core = HALF_PI * rc * fc;
    // Shear layer: rho ∝ (1 - (r/R)²)², the compact stand-in for the
    // self-similar Gaussian of a turbulent jet.
    //   ∫ along the chord = (16/15)·R·(1 - (p/R)²)^(5/2).
    let fs = max(1.0 - (lateral * lateral) / (rs * rs), 0.0);
    let path_sheath = (16.0 / 15.0) * rs * pow(fs, 2.5);

    // Thermodynamics along the column. `er` > 1 inside a shock waist (compressed,
    // hotter), < 1 downstream (expanded, cooler).
    let er = r0 / rc;
    // Weight the compression node hard toward the axis. The Mach disk is a
    // normal shock across the *centre* of the core, so its glow is a lens on the
    // axis that tapers well before the barrel edge. A gentler weight spreads it
    // to the full width, and in a column whose optical depth already saturates
    // that renders as a flat rung — the stacked-blocks look.
    let node = shock_node(s) * pow(fc, 3.0);
    // Two independent cooling mechanisms, which is what lets one model cover the
    // whole envelope. In vacuum the column cools by *expansion* (`er` shrinks).
    // At sea level it barely expands at all — it cools by entraining ambient air,
    // which expansion alone cannot express, and without which a sea-level plume
    // stays uniformly incandescent for its entire length.
    let entrained = exp(-plume.anim.w * s / r0);
    let flick = flicker();
    // Chamber temperature: below 1 only during the ignition transient, and dipped
    // by combustion roughness. Both therefore shorten the *visible* column
    // through the same law that sized the mesh — a start-up flare, not a pop.
    let temp_exit = plume.therm.x * flick;
    let temp = min(temp_exit * pow(er, plume.shape.w) * entrained * (1.0 + 0.8 * node), 1.6);
    let density = er * er;

    // Turbulent shear layer. `phi` is the true azimuth on the near surface, so
    // structures compress toward the silhouette like a real rotating column.
    let phi = asin(clamp(x, -1.0, 1.0));
    let turb = turbulence(phi, s, b);
    // Laminar through the potential core, ragged once the shear layer has eaten
    // it — real jets break down after a few diameters, and in atmosphere
    // entrainment tears the tail apart.
    let amp = plume.mixing.z * mix(0.12, 1.0, b) * smoothstep(0.0, 0.06, t);
    let amp_sheath = clamp(amp * (1.0 + plume.therm.z * b), 0.0, 1.0);
    let amp_core = clamp(amp * plume.therm.y, 0.0, 1.0);
    let mod_sheath = mix(1.0, 0.28 + 1.5 * turb, amp_sheath);
    let mod_core = mix(1.0, 0.45 + 1.1 * turb, amp_core);

    // Optical depth: rho · path, normalised so an on-axis ray at the exit plane
    // sees exactly `kappa`.
    let tau_core = plume.shock.x * density * (chord_core / (2.0 * r0)) * mod_core;
    let tau_sheath = plume.shock.y * density * (path_sheath / (2.0 * r0)) * mod_sheath;

    // Grey-body source functions, normalised to 1 at chamber temperature. The
    // sheath runs cooler than the core, except in atmosphere where entrained air
    // afterburns the fuel-rich exhaust and *adds* luminosity.
    let afterburn = plume.mixing.y;
    let temp_sheath = temp * mix(0.62, 0.88, afterburn);
    // Afterburning is a genuine energy release, so it brightens the whole column,
    // not just its edge — most strongly in the shear layer where the fuel-rich
    // exhaust actually meets the air.
    let src_core = band_emission(temp) * mix(1.0, 1.8, afterburn);
    let src_sheath = band_emission(temp_sheath) * mix(1.0, 2.4, afterburn);

    // Colour follows temperature: white-hot at the nozzle and in the shock nodes,
    // relaxing to the propellant tint as the column cools.
    let col_core = mix(plume.mid_color.rgb, plume.core_color.rgb, smoothstep(0.80, 1.0, temp));
    let col_sheath = mix(plume.edge_color.rgb, plume.mid_color.rgb, smoothstep(0.22, 0.78, temp_sheath));

    // NB: `anim.w` is the entrainment rate (consumed above), *not* a radiance
    // trim — it must not appear here. It multiplied this product until
    // 2026-07-22, which zeroed the whole plume in vacuum (rate 0) and dimmed it
    // ~60x at sea level. See docs/incidents/0020.
    let gain = plume.core_color.a * plume.anim.z * mix(1.0, flick, 0.6);

    var radiance = col_core * src_core * (1.0 - exp(-tau_core))
        + col_sheath * src_sheath * (1.0 - exp(-tau_sheath));
    radiance *= gain;

    // Bevy's AlphaMode::Add uses PREMULTIPLIED_ALPHA_BLENDING
    // (out = src.rgb + dst.rgb * (1 - src.a)); a pure-additive glow emits the
    // premultiplied radiance with alpha 0 so the background is preserved.
    return vec4<f32>(radiance, 0.0);
}
