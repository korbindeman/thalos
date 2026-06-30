#define_import_path thalos::grass_displace

// Shared grass-blade vertex displacement, called by BOTH the main grass shader
// (`ground/grass.wgsl`) AND the depth-prepass (`ground/grass_prepass.wgsl`) so the
// two produce IDENTICAL clip depth. A mismatch lets the pre-populated prepass
// depth wrongly early-Z-reject visible blades (they flicker / vanish). Pure — all
// inputs are parameters, no bindings — so it composes into either pipeline. The
// blade height-fade, altitude-collapse, and wind sway all live here: edit the
// displacement in ONE place and both passes stay in lockstep.

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

// Procedural grass-tuft coverage for a clump CARD (far/mid clipmap rings): a dense
// fan of varied-height blades under a rounded dome envelope, so a single quad reads
// as a tuft of grass (not a row of fence-posts). `across` (0..1) runs left→right,
// `height` (0..1) bottom→top. The main grass shader and the depth-prepass call this
// identically, so their alpha-discard (hence depth) matches. Returns coverage in
// [0,1]; the caller discards below ~0.5.
fn grass_tuft_alpha(across: f32, height: f32) -> f32 {
    // A dense patch of grass: many vertical blades of varied height with a RAGGED
    // top (no dome envelope — that concentrated coverage into a few tall spikes).
    // ~18 blades across, each a soft streak slightly narrowing toward its tip.
    let slots = 18.0;
    let s = across * slots;
    let slot = floor(s);
    let f = s - slot;
    let bw = 0.40 - 0.14 * height; // wider at the base, taper to the tip
    let blade = smoothstep(0.5 - bw, 0.5 - bw + 0.12, f)
        * (1.0 - smoothstep(0.5 + bw - 0.12, 0.5 + bw, f));

    // Per-blade tip height, jittered, mostly tall-ish so the patch fills in (the
    // card is short, so this is a ragged lawn edge, not spikes).
    let jit = fract(sin(slot * 12.9898 + 4.1) * 43758.5453);
    let tip = 0.55 + 0.45 * jit;
    let along = 1.0 - smoothstep(tip - 0.12, tip, height);

    // Fade the extreme left/right card edges so neighbouring cards blend instead of
    // showing hard quad borders.
    let edge = smoothstep(0.0, 0.06, across) * (1.0 - smoothstep(0.94, 1.0, across));
    return blade * along * edge;
}
