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
    // xyz = vegetation focus (player craft) in render space; w = 1 valid / 0 use
    // camera. The clipmap fade measures distance from THIS, not the camera, so
    // zooming / orbiting the camera doesn't change what's drawn.
    anchor: vec4<f32>,
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
    let world_normal = normalize(mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index));

    // Clipmap scale-fade: each blade's HEIGHT scales 0 → full → 0 across the
    // ring's near/far edges, so adjacent rings cross-fade by growing/shrinking
    // (seamless — no dither, no pop-in; a 0-height blade is a flat invisible
    // sliver). uv.x = height fraction (0 root → 1 tip); color.a = blade height H.
    // Distance is from the focus anchor (craft), not the camera, so zoom/orbit
    // doesn't change it. The innermost ring passes a large-negative near edge so
    // it never fades in.
    let ref_pos = select(view.world_position, grass.anchor.xyz, grass.anchor.w > 0.5);
    let dist = distance(ref_pos, world_pos);
    let near_edge = grass.time_fade.y;
    let far_edge = grass.time_fade.z;
    let band = max(grass.time_fade.w, 1.0);
    let fade_in = smoothstep(near_edge - band, near_edge + band, dist);
    let fade_out = 1.0 - smoothstep(far_edge - band, far_edge + band, dist);
    let grow = fade_in * fade_out;
    // Collapse this vertex toward its root along the terrain up by its un-grown
    // height.
    let above = in.uv.x * in.color.a;
    world_pos -= world_normal * (above * (1.0 - grow));

    // Wind sway: two incommensurate sines per blade, displacing toward the wind
    // direction in world space, scaled by `grow` so collapsed blades stay calm.
    let t = grass.time_fade.x;
    let phase = in.uv.y * TAU;
    let gust = 0.7 * sin(1.9 * t + phase) + 0.3 * sin(3.7 * t + 9.0 * in.uv.y);
    world_pos += grass.wind.xyz * (in.uv.x * grass.wind.w * (0.6 + 0.4 * gust) * grow);

    var out: VertexOutput;
    out.world_position = world_pos;
    out.clip_position = position_world_to_clip(world_pos);
    out.world_normal = world_normal;
    out.color = in.color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // No edge discard: the seamless ring cross-fade is the vertex scale-fade
    // (blades grow/shrink in height), so there's nothing to cut here.

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
