#define_import_path thalos::grass_displace

// Shared grass-blade vertex displacement (`ground/grass.wgsl`). Kept as a
// standalone library so a future depth-prepass shader can reproduce the main
// pass's clip depth EXACTLY by importing the same function (a mismatch lets a
// pre-populated prepass depth wrongly early-Z-reject visible blades — the
// removed grass prepass hit this). Pure — all inputs are parameters, no
// bindings — so it composes into any pipeline. The blade height-fade,
// altitude-collapse, and wind sway all live here.

const GRASS_DISPLACE_TAU: f32 = 6.28318530717958647;

// Final displaced WORLD position of a blade vertex.
//   world_pos_in   : un-displaced world position (mesh_position_local_to_world)
//   root_pos       : blade-root world position (the fade shrinks toward it)
//   uv             : x = height fraction (0 root → 1 tip), y = per-blade phase
//   wind           : xyz = wind dir, w = tip sway amplitude (m)
//   time_fade      : x = time, y = near edge, z = far edge, w = band half-width
//   sky_up_w       : altitude collapse 0 (full) → 1 (collapsed)
//   anchor         : xyz = craft offset from camera (render space), w = 1 valid
//   view_world_pos : view.world_position
fn grass_blade_world_pos(
    world_pos_in: vec3<f32>,
    root_pos: vec3<f32>,
    uv: vec2<f32>,
    wind: vec4<f32>,
    time_fade: vec4<f32>,
    sky_up_w: f32,
    anchor: vec4<f32>,
    view_world_pos: vec3<f32>,
) -> vec3<f32> {
    // Clipmap scale-fade reference = the craft, expressed in the current render
    // origin as camera + offset (origin-invariant across big_space recentres).
    let ref_pos = view_world_pos + anchor.xyz;
    let dist = distance(ref_pos, world_pos_in);
    let near_edge = time_fade.y;
    let far_edge = time_fade.z;
    let band = max(time_fade.w, 1.0);
    let fade_in = smoothstep(near_edge - band, near_edge + band, dist);
    let fade_out = 1.0 - smoothstep(far_edge - band, far_edge + band, dist);
    let altitude_grow = clamp(1.0 - sky_up_w, 0.0, 1.0);
    let grow = fade_in * fade_out * altitude_grow;

    // Shrink the blade UNIFORMLY toward its root (a fading blade is a smaller
    // upright blade, never a flattened card).
    var world_pos = root_pos + (world_pos_in - root_pos) * grow;

    // Wind sway: two incommensurate sines per blade, toward the wind direction,
    // weighted by height (uv.x) and scaled by `grow` so collapsed blades are calm.
    let t = time_fade.x;
    let phase = uv.y * GRASS_DISPLACE_TAU;
    let gust = 0.7 * sin(1.9 * t + phase) + 0.3 * sin(3.7 * t + 9.0 * uv.y);
    world_pos += wind.xyz * (uv.x * wind.w * (0.6 + 0.4 * gust) * grow);
    return world_pos;
}
