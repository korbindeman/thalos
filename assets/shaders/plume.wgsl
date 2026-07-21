// Rocket-engine exhaust plume — the immediate nozzle plume rendered as a
// camera-facing, axis-locked billboard with an analytic volumetric cross-section.
//
// The mesh is a unit quad in plume-local space (x = lateral in [-1, 1],
// y = axial in [0, 1], 0 at the nozzle exit). The vertex stage rebuilds it every
// frame as a *cylindrical billboard*: locked to the engine's exhaust axis
// (the part's local -Y, i.e. opposite thrust) but rotated about that axis to face
// the camera, so a flat strip reads as a round plume from any side view.
//
// The fragment stage integrates a radially-symmetric density through that round
// cross-section (chord through a cylinder of radius R(t)), so the plume is bright
// and thick on-axis and feathers to nothing at the silhouette — no hard mesh
// edge. On top of that envelope it layers the emission temperature field: a hot
// near-nozzle core, shock-diamond (Mach-disk) compression nodes that fade
// downstream, a cooler mixing-layer sheath, and turbulent breakup. Colour comes
// from a three-stop propellant palette (edge -> mid -> core) indexed by that
// temperature. Additive HDR, so the post-stack bloom haloes the core.
//
// Everything about the *shape and pressure response* (length, radial expansion,
// shock-cell count/contrast) is resolved CPU-side from the engine's PlumeSignals
// (throttle, pressure ratio, ignition) into PlumeParams; this shader only renders
// the resolved profile. See `crate::rendering::plume`.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}

// Resolved per-engine plume profile. Mirrors `PlumeParams` (Rust, ShaderType);
// packed as vec4s so the std140 layout is unambiguous.
struct PlumeParams {
    // rgb = hot core colour (near-nozzle / Mach disks), a = HDR emission scale.
    core_color: vec4<f32>,
    // rgb = mid plume colour, a = nozzle exit radius (m).
    mid_color: vec4<f32>,
    // rgb = cool sheath / tip colour, a = billboard half-width R_max (m).
    edge_color: vec4<f32>,
    // x = visible axial length (m), y = radial expansion factor,
    // z = shock-cell count over the length, w = shock-cell contrast (0 in vacuum).
    shape: vec4<f32>,
    // x = axial core decay rate, y = shock fade rate downstream,
    // z = edge softness (mixing-layer width), w = commanded throttle 0..1.
    response: vec4<f32>,
    // x = time (s), y = per-engine seed, z = ignition 0..1, w = density scale.
    anim: vec4<f32>,
}

// MaterialPlugin bind group: group 3, binding 0 (matches the other game materials).
@group(3) @binding(0) var<uniform> plume: PlumeParams;

const PI: f32 = 3.14159265;
const TAU: f32 = 6.28318531;

// -- cheap hash / value-noise / fBm for turbulent breakup ------------------

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
    return 0.6 * value_noise(p) + 0.3 * value_noise(p * 2.2) + 0.1 * value_noise(p * 4.7);
}

// Radial envelope of the plume at axial fraction t (0 exit -> 1 tip), in units of
// the nozzle exit radius. Contracts slightly, expands downstream toward the
// authored expansion factor, then closes to a point at the tip. A faint barrel
// modulation follows the shock structure so the silhouette breathes.
fn radius_profile(t: f32) -> f32 {
    let expansion = plume.shape.y;
    let cells = plume.shape.z;
    let contrast = plume.shape.w;
    // Widen from the throat toward the authored expansion over the first ~60%.
    let grow = mix(0.85, expansion, smoothstep(0.0, 0.6, t));
    // Close to a soft point over the last quarter.
    let taper = 1.0 - smoothstep(0.7, 1.0, t) * 0.92;
    // Barrel bulges tied to the shock cells (only meaningful with back-pressure).
    let barrel = 1.0 + contrast * 0.12 * sin(t * cells * PI) * (1.0 - t);
    return max(grow * taper * barrel, 0.0);
}

// Shock-diamond (Mach-disk) term: sharp bright compression nodes, strongest near
// the nozzle and fading downstream. Zero when there is no back-pressure.
fn shock_term(t: f32) -> f32 {
    let cells = plume.shape.z;
    let contrast = plume.shape.w;
    let fade = plume.response.y;
    // Rectified, sharpened cosine gives crisp rings at the compression points.
    let node = pow(max(0.0, -cos(t * cells * TAU)), 6.0);
    return contrast * node * exp(-t * fade);
}

// -- vertex: cylindrical billboard around the exhaust axis ------------------

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // x = axial fraction t (0..1), y = lateral fraction (-1..1 across R_max).
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
    let lateral = in.position.x;         // -1 .. 1
    let length_m = plume.shape.x;
    let half_width = plume.edge_color.a; // R_max

    let world_pos = center + axis * (t * length_m) + right * (lateral * half_width);

    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world_pos);
    out.plume_uv = vec2<f32>(t, lateral);
    return out;
}

// -- fragment: analytic volumetric cross-section + emission field -----------

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = clamp(in.plume_uv.x, 0.0, 1.0);
    let lateral = in.plume_uv.y;

    let nozzle_r = plume.mid_color.a;
    let half_width = plume.edge_color.a;

    // Radius of the round plume at this axial station, and the lateral distance
    // of this fragment from the axis — both in metres.
    let radius_m = nozzle_r * radius_profile(t);
    let lateral_m = abs(lateral) * half_width;
    if (radius_m <= 1e-4 || lateral_m >= radius_m) {
        return vec4<f32>(0.0);
    }

    // Normalised half-chord through the cylinder: 1 on-axis, softened to 0 at the
    // silhouette by the mixing-layer width so the edge feathers instead of
    // showing the ellipse boundary.
    let edge = plume.response.z;
    let chord = sqrt(max(radius_m * radius_m - lateral_m * lateral_m, 0.0)) / radius_m;
    let radial = smoothstep(0.0, edge, chord);

    // Emission temperature field (0..~1.2). Hot at the nozzle, cooler downstream,
    // hotter on-axis, with bright shock-diamond nodes.
    let core_decay = plume.response.x;
    let axial_temp = exp(-t * core_decay);
    let shock = shock_term(t);
    let temp = clamp(axial_temp * mix(0.35, 1.0, radial) + shock * radial, 0.0, 1.25);

    // Turbulent breakup, animated and per-engine varied; stronger downstream and
    // toward the sheath where the plume mixes with air.
    let time = plume.anim.x;
    let seed = plume.anim.y;
    let turb = fbm2(vec2<f32>(t * 9.0 - time * 6.0, lateral * 3.5 + seed * 17.0));
    let turb_mod = mix(1.0, 0.55 + 0.9 * turb, (1.0 - radial) * (0.35 + 0.65 * t));

    // Three-stop propellant palette indexed by temperature.
    var col = mix(plume.edge_color.rgb, plume.mid_color.rgb, smoothstep(0.12, 0.55, temp));
    col = mix(col, plume.core_color.rgb, smoothstep(0.55, 1.02, temp));

    // Coverage (additive opacity): gas column on-axis, denser near the nozzle,
    // thinner sheath downstream, broken up by turbulence and faded by ignition.
    let ignition = plume.anim.z;
    let density_scale = plume.anim.w;
    let throttle = plume.response.w;
    var coverage = radial * mix(0.30, 1.0, axial_temp) * turb_mod * density_scale * ignition;
    // Fade the extreme tip so the plume dissolves rather than ending on a disc.
    coverage *= 1.0 - smoothstep(0.85, 1.0, t);
    coverage = clamp(coverage, 0.0, 1.0);

    let intensity = plume.core_color.a;
    let emission = col * intensity * temp * (0.5 + 0.5 * throttle);
    // Bevy's AlphaMode::Add uses PREMULTIPLIED_ALPHA_BLENDING
    // (out = src.rgb + dst.rgb * (1 - src.a)); pure-additive glow means folding
    // the coverage into the (premultiplied) colour and emitting alpha 0 so the
    // background is preserved and the emission is simply added.
    return vec4<f32>(emission * coverage, 0.0);
}
