// Rocket-engine exhaust plume — an axisymmetric emitting gas column whose shape
// is set by the nozzle/ambient pressure ratio and whose brightness *follows from
// that shape* rather than from an authored fade curve.
//
// The model, in one place:
//
//   R(s)  = R0·lip + tan(theta)·s          free-expansion cone off the nozzle lip
//           × barrel(s)                    shock-cell pinching (needs back-pressure)
//   rho   ∝ (R0/R)²                        mass conservation along the column
//   T     ∝ (R0/R)^(2(gamma-1))            adiabatic (expansion) cooling
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
    // z = turbulence amplitude, w = throttle.
    mixing: vec4<f32>,
    // x = time (s), y = per-engine seed, z = ignition 0..1,
    // w = entrainment cooling rate (per nozzle radius; 0 in vacuum).
    anim: vec4<f32>,
}

@group(3) @binding(0) var<uniform> plume: PlumeParams;

// -- cheap hash / value-noise / fBm for the turbulent shear layer -----------

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

fn fbm2(p: vec2<f32>) -> f32 {
    return 0.55 * value_noise(p) + 0.30 * value_noise(p * 2.3) + 0.15 * value_noise(p * 5.1);
}

// -- emission ---------------------------------------------------------------

// Dimensionless Wien parameter hc/(lambda·k·T_exit) for the visible band at
// combustion-chamber temperature. Larger = the plume darkens faster as it cools.
const WIEN: f32 = 3.0;

// Visible-band emission from a cooling gas, normalised to 1 at the nozzle exit.
//
// This is deliberately *not* a grey-body T^4. A rocket plume radiates in the
// visible from the Wien side of the Planck curve, where output falls off
// exponentially in 1/T rather than polynomially — which is why a real plume goes
// dark over a few nozzle diameters instead of trailing off forever. It is also
// what lets the geometry end without a lit disc at the tip: by the time the
// column has expanded a few-fold, this term has already collapsed.
fn band_emission(temp_norm: f32) -> f32 {
    return exp(-WIEN * (1.0 / max(temp_norm, 1e-3) - 1.0));
}

// -- plume geometry (shared by both stages so the mesh silhouette *is* the
//    analytic envelope — no wasted transparent strip, no edge mismatch) ------

// Shock-cell phase modulation, decaying downstream. 1 = unmodulated.
fn barrel(s: f32) -> f32 {
    let k = plume.shock.z;
    let strength = plume.shock.w;
    let decay = exp(-s / max(plume.mixing.x, 1e-3));
    return 1.0 + 0.10 * strength * decay * cos(k * s);
}

// Luminous core radius at axial station s (metres from the exit plane).
fn core_radius(s: f32) -> f32 {
    let r0 = plume.mid_color.a;
    let cone = r0 * plume.shape.x + plume.shape.y * s;
    return max(cone * barrel(s), 1e-4);
}

// Outer edge of the turbulent shear layer. Grows linearly (entrainment), plus a
// thin lip so the sheath is never zero-width at the nozzle.
fn sheath_radius(s: f32) -> f32 {
    let r0 = plume.mid_color.a;
    return core_radius(s) + plume.shape.z * s + r0 * 0.05;
}

// Compression-node brightness: peaks exactly where `barrel` pinches the column.
fn shock_node(s: f32) -> f32 {
    let k = plume.shock.z;
    let strength = plume.shock.w;
    let decay = exp(-s / max(plume.mixing.x, 1e-3));
    return strength * decay * pow(max(0.0, -cos(k * s)), 8.0);
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

    let rc = core_radius(s);
    let rs = sheath_radius(s);
    // The mesh edge *is* the sheath boundary, so this is exact.
    let lateral = abs(x) * rs;

    // Chords through the two nested cylinders; the sheath contributes only the
    // part of the line of sight outside the core.
    let chord_sheath = 2.0 * sqrt(max(rs * rs - lateral * lateral, 0.0));
    let chord_core = 2.0 * sqrt(max(rc * rc - lateral * lateral, 0.0));
    let path_sheath = max(chord_sheath - chord_core, 0.0);

    // Thermodynamics along the column. `er` > 1 inside a shock waist (compressed,
    // hotter), < 1 downstream (expanded, cooler).
    let er = r0 / rc;
    // Weight the compression node toward the axis. The Mach disk is a normal
    // shock across the core, so its glow is strongest on-axis and tapers to the
    // barrel edge — which reads as a lens/diamond rather than a flat rung across
    // the column.
    let axis_frac = chord_core / (2.0 * max(rc, 1e-4));
    let node = shock_node(s) * axis_frac;
    // Two independent cooling mechanisms, which is what lets one model cover the
    // whole envelope. In vacuum the column cools by *expansion* (`er` shrinks).
    // At sea level it barely expands at all — it cools by entraining ambient air,
    // which expansion alone cannot express, and without which a sea-level plume
    // stays uniformly incandescent for its entire length.
    let entrained = exp(-plume.anim.w * s / r0);
    let temp = min(pow(er, plume.shape.w) * entrained * (1.0 + 0.8 * node), 1.6);
    let density = er * er;

    // Turbulent shear layer: streaks stretched along the flow and advected
    // downstream. `phi` is the true azimuth on the near surface, so the streaks
    // compress toward the silhouette like a real rotating column would.
    let phi = asin(clamp(x, -1.0, 1.0));
    let seed = plume.anim.y;
    let flow = plume.anim.x * 2.6;
    // Strongly anisotropic: high frequency around the circumference, low along
    // the flow, so the noise reads as long filaments stretched downstream rather
    // than isotropic blobs. This is the striation that dominates a real
    // sea-level exhaust column.
    let turb = fbm2(vec2<f32>(phi * 9.0 + seed * 31.0, t * 3.0 - flow));
    // Laminar at the lip, ragged downstream — real jets break down after a few
    // diameters, and in atmosphere entrainment tears the tip apart.
    let amp = plume.mixing.z * smoothstep(0.04, 0.55, t);
    let turb_mod = mix(1.0, 0.30 + 1.4 * turb, amp);

    // Optical depth: rho · path, normalised so an on-axis ray at the exit plane
    // sees exactly `kappa`.
    let tau_core = plume.shock.x * density * (chord_core / (2.0 * r0));
    let tau_sheath = plume.shock.y * density * (path_sheath / (2.0 * r0)) * turb_mod;

    // Grey-body source functions, normalised to 1 at the nozzle exit. The sheath
    // runs cooler than the core, except in atmosphere where entrained air
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
    let gain = plume.core_color.a * plume.anim.z * mix(0.40, 1.0, plume.mixing.w);

    var radiance = col_core * src_core * (1.0 - exp(-tau_core))
        + col_sheath * src_sheath * (1.0 - exp(-tau_sheath));
    radiance *= gain;

    // Bevy's AlphaMode::Add uses PREMULTIPLIED_ALPHA_BLENDING
    // (out = src.rgb + dst.rgb * (1 - src.a)); a pure-additive glow emits the
    // premultiplied radiance with alpha 0 so the background is preserved.
    return vec4<f32>(radiance, 0.0);
}
