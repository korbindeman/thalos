// Batched grass-blade shader.
//
// Vertex: standard mesh transform plus a world-space wind sway displacement,
// weighted by UV.x (0 at the root, 1 at the tip) and phase-shifted per blade
// by UV.y so the field doesn't move as one sheet.
//
// Fragment: wrap-diffuse Lambert against the primary star, with constants
// mirroring `body_terrain.wgsl`'s vegetated path (`DIRECT_SUN_STRENGTH`,
// day/night sky fill) so blades land in the same brightness range as the
// ground they grow from. The vegetated terrain path deliberately ignores the
// scene flux (see `body_terrain.wgsl`), so the only lighting input needed
// here is the sun direction. Distance fade is a screen-space-dithered
// `discard` in the opaque pass — no sorting, no blend state.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}

struct GrassParams {
    // xyz = unit direction toward the star (world render space), w unused.
    sun_dir: vec4<f32>,
    // xyz = wind direction (world render space), w = tip sway amplitude (m).
    wind: vec4<f32>,
    // x = time (s), y = fade start (m), z = fade end (m), w unused.
    time_fade: vec4<f32>,
}

// Standard MaterialPlugin bind group in Bevy 0.18: group 3.
@group(3) @binding(0) var<uniform> grass: GrassParams;

// Mirrors of `body_terrain.wgsl`'s vegetated lighting constants.
const DIRECT_SUN_STRENGTH: f32 = 0.62;
const NIGHT_FILL: f32 = 0.012;
const DAY_FILL: f32 = 0.15;

const TAU: f32 = 6.28318530717958647;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    // rgb = blade tint (linear), a = per-blade dither jitter.
    @location(2) color: vec4<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    var world_pos =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0))
            .xyz;

    // Wind sway: two incommensurate sines per blade, displacing toward the
    // wind direction in world space. The displacement is sub-half-metre, so
    // applying it post-transform is exact enough and avoids inverse-rotating
    // the wind into each tile's body-fixed frame.
    let t = grass.time_fade.x;
    let phase = in.uv.y * TAU;
    let gust = 0.7 * sin(1.9 * t + phase) + 0.3 * sin(3.7 * t + 9.0 * in.uv.y);
    world_pos += grass.wind.xyz * (in.uv.x * grass.wind.w * (0.6 + 0.4 * gust));

    var out: VertexOutput;
    out.world_position = world_pos;
    out.clip_position = position_world_to_clip(world_pos);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
    out.color = in.color;
    return out;
}

// Small screen-space hash for the fade dither.
fn screen_hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Distance fade: dithered discard so the grass ring has no hard edge.
    let dist = distance(view.world_position, in.world_position);
    let fade = 1.0 - smoothstep(grass.time_fade.y, grass.time_fade.z, dist);
    if fade < screen_hash(in.clip_position.xy + vec2<f32>(in.color.a * 64.0)) {
        discard;
    }

    // Blades carry the *terrain* normal (not the card normal), so they light
    // like the ground they grow from and the card geometry doesn't read in
    // the shading. Wrap diffuse stands in for transmission through the blade.
    let n = normalize(in.world_normal);
    let sun_dir = grass.sun_dir.xyz;
    let n_dot_l = dot(n, sun_dir);
    let wrap = clamp((n_dot_l + 0.4) / 1.4, 0.0, 1.0);

    // Sky fill mirrors the terrain's day/night ambient gate.
    let daylight = smoothstep(-0.06, 0.12, n_dot_l);
    let fill = mix(NIGHT_FILL, DAY_FILL, daylight);
    let sky_tint = mix(vec3<f32>(1.0), vec3<f32>(0.62, 0.74, 1.0), 0.25 * daylight);

    let lit = in.color.rgb * (wrap * DIRECT_SUN_STRENGTH) + in.color.rgb * sky_tint * fill;
    return vec4<f32>(lit, 1.0);
}
