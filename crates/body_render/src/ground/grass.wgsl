// Batched grass-blade shader.
//
// Vertex: standard mesh transform plus a world-space wind sway displacement,
// weighted by UV.x (0 at the root, 1 at the tip) and phase-shifted per blade
// by UV.y so the field doesn't move as one sheet.
//
// Fragment: wrap-diffuse direct sun plus the SAME hemisphere sky model the
// ground uses, both pulled from `thalos::lighting` (`compute_surface_sky` /
// `sky_ambient_irradiance`) so blades and the ground they grow from can't drift.
// The driver hands the blades the sun flux, radial up, and Rayleigh τ_v the
// model needs (see `GrassParams`). Distance fade is a screen-space-dithered
// `discard` in the opaque pass — no sorting, no blend state.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import thalos::lighting::{SurfaceSky, compute_surface_sky, sky_ambient_irradiance}

struct GrassParams {
    // xyz = unit direction toward the star (world render space), w = sun flux
    // (lux × exposure gain — the same value the terrain `SceneLighting` carries).
    sun_dir: vec4<f32>,
    // xyz = wind direction (world render space), w = tip sway amplitude (m).
    wind: vec4<f32>,
    // x = time (s), y = fade start (m), z = fade end (m), w unused.
    time_fade: vec4<f32>,
    // xyz = local radial up (world render space) for the sky hemisphere, w unused.
    sky_up: vec4<f32>,
    // xyz = Rayleigh vertical optical depth τ_v, w = atmosphere strength.
    sky_tau: vec4<f32>,
}

// Standard MaterialPlugin bind group in Bevy 0.18: group 3.
@group(3) @binding(0) var<uniform> grass: GrassParams;

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
    // Clipmap cross-fade: each ring fades IN around its near edge and OUT around
    // its far edge, so adjacent rings dither-blend through their shared boundary
    // (no hard LOD seam). The innermost ring passes a large-negative near edge
    // so it never fades in. Dithered discard keeps it in the opaque pass.
    let dist = distance(view.world_position, in.world_position);
    let near_edge = grass.time_fade.y;
    let far_edge = grass.time_fade.z;
    let band = max(grass.time_fade.w, 1.0);
    let fade_in = smoothstep(near_edge - band, near_edge + band, dist);
    let fade_out = 1.0 - smoothstep(far_edge - band, far_edge + band, dist);
    let fade = fade_in * fade_out;
    if fade < screen_hash(in.clip_position.xy + vec2<f32>(in.color.a * 64.0)) {
        discard;
    }

    // Blades carry the *terrain* normal (not the card normal), so they light
    // like the ground they grow from and the card geometry doesn't read in
    // the shading. Wrap diffuse stands in for transmission through the blade.
    let n = normalize(in.world_normal);
    let sun_dir = grass.sun_dir.xyz;
    let up = grass.sky_up.xyz;

    // Same atmosphere-derived sky/sun environment the ground builds, so the
    // grass tracks the ground through the day and gets the same blue-sky fill.
    let sky = compute_surface_sky(grass.sky_tau.xyz, grass.sky_tau.w, up, sun_dir, grass.sun_dir.w);

    // Direct: wrap-diffuse (blades are translucent), reddened + exposure-scaled
    // by the shared sun term.
    let n_dot_l = dot(n, sun_dir);
    let wrap = clamp((n_dot_l + 0.4) / 1.4, 0.0, 1.0);
    let direct = in.color.rgb * (wrap * sky.sun_scale) * sky.sun_color;

    // Ambient: the hemisphere sky model (blue sky-dome + warm ground bounce).
    let ambient = in.color.rgb * sky_ambient_irradiance(sky, n, up);

    let lit = direct + ambient;
    return vec4<f32>(lit, 1.0);
}
