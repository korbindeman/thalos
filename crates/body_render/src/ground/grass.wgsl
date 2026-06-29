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
#import thalos::lighting::{compute_surface_sky, FoliageSurface, shade_foliage}

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
    // xyz = vegetation focus offset = (player craft − camera) in render space;
    // w = 1 valid / 0 = fade around the camera. The fade reference is rebuilt as
    // `view.world_position + offset`, i.e. the craft expressed in the CURRENT
    // frame's render origin. Passing an *offset* (not an absolute anchor) makes
    // it robust to big_space floating-origin recentres — see `vertex`.
    anchor: vec4<f32>,
}

// Standard MaterialPlugin bind group in Bevy 0.18: group 3.
@group(3) @binding(0) var<uniform> grass: GrassParams;

// Cascaded sun-shadow maps — the SAME depth maps the terrain + trees sample,
// so a tree's shadow falls on the grass beneath it. Sampled PER-VERTEX (the
// blade is small; a per-vertex factor reads the same as per-fragment on this
// overdraw-heavy material) so the depth bindings carry `vertex` visibility.
// Mirrors `tree.wgsl` / `body_terrain.wgsl`.
struct ShadowCascadeBlock {
    view_proj: array<mat4x4<f32>, 3>,
    // per cascade: x = depth bias (clip), yzw reserved.
    params: array<vec4<f32>, 3>,
    // x = strength (0 ⇒ skip), y = active cascade count, zw reserved.
    gate: vec4<f32>,
}
@group(3) @binding(1) var<uniform> grass_shadow: ShadowCascadeBlock;
@group(3) @binding(2) var sun_shadow_map_0: texture_depth_2d;
@group(3) @binding(3) var sun_shadow_map_1: texture_depth_2d;
@group(3) @binding(4) var sun_shadow_map_2: texture_depth_2d;

const TAU: f32 = 6.28318530717958647;

// One cascade's shadow factor, or a negative sentinel if outside its box (the
// caller falls through to the next cascade). A single tap — grass interpolates
// the factor up the blade, so PCF is wasted here. Mirrors `tree.wgsl`'s box test.
fn cascade_factor(
    world_pos: vec3<f32>,
    vp: mat4x4<f32>,
    bias: f32,
    strength: f32,
    tex: texture_depth_2d,
    inset: f32,
) -> f32 {
    let clip = vp * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return -1.0;
    }
    let ndc = clip.xyz / clip.w;
    if (any(ndc.xy < vec2<f32>(-inset)) || any(ndc.xy > vec2<f32>(inset)) ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return -1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let dims = vec2<f32>(textureDimensions(tex));
    let texel = vec2<i32>(uv * dims);
    let stored = textureLoad(tex, texel, 0);
    let lit = select(1.0, 0.0, stored > ndc.z + bias);
    return 1.0 - strength * (1.0 - lit);
}

// Walk cascades near→far, use the tightest hit. `gate.x == 0` ⇒ fully lit. Grass
// only exists within the near cascade's reach, so this returns from cascade 0 in
// practice (1+2 are the fallthrough for blades near a cascade seam).
fn sun_shadow_factor(world_pos: vec3<f32>) -> f32 {
    let s = grass_shadow.gate.x;
    if (s <= 0.0) {
        return 1.0;
    }
    var f = cascade_factor(world_pos, grass_shadow.view_proj[0], grass_shadow.params[0].x, s, sun_shadow_map_0, 0.98);
    if (f < 0.0) {
        f = cascade_factor(world_pos, grass_shadow.view_proj[1], grass_shadow.params[1].x, s, sun_shadow_map_1, 0.98);
    }
    if (f < 0.0) {
        f = cascade_factor(world_pos, grass_shadow.view_proj[2], grass_shadow.params[2].x, s, sun_shadow_map_2, 1.0);
    }
    if (f < 0.0) {
        return 1.0;
    }
    return f;
}

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
    // Per-vertex sun-shadow factor (sampled at this vertex's world position).
    @location(3) shadow: f32,
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
    //
    // The anchor is passed as an OFFSET from the camera, not an absolute world
    // point: `view.world_position` is current-frame, but a CPU-supplied absolute
    // anchor is one frame stale, so on a big_space floating-origin recentre (the
    // origin jumps a whole cell while the parked craft co-rotates through space)
    // the stale anchor sits in the previous origin and `dist` jumps by a cell —
    // collapsing fade-band tiles for that frame ("tiles pop in/out while
    // moving"). `(ship − camera)` is origin-invariant, so rebuilding the
    // reference as camera + offset is recentre-safe. offset 0 → camera.
    let ref_pos = view.world_position + grass.anchor.xyz;
    let dist = distance(ref_pos, world_pos);
    let near_edge = grass.time_fade.y;
    let far_edge = grass.time_fade.z;
    let band = max(grass.time_fade.w, 1.0);
    let fade_in = smoothstep(near_edge - band, near_edge + band, dist);
    let fade_out = 1.0 - smoothstep(far_edge - band, far_edge + band, dist);
    // Altitude collapse: from a plane the blades subtend ~no pixels and the
    // terrain albedo already carries the grass colour, so the whole blade layer
    // sinks into the ground as the craft climbs (driver writes `sky_up.w` =
    // collapse, 0 near the ground → 1 high up). 0 by default, so ground-level and
    // the preview are unaffected.
    let altitude_grow = clamp(1.0 - grass.sky_up.w, 0.0, 1.0);
    let grow = fade_in * fade_out * altitude_grow;
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
    // Per-vertex sun-shadow: tree (and self) shadows on the grass, sampled at
    // the blade's final (swayed) world position and interpolated up the blade.
    out.shadow = sun_shadow_factor(world_pos);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // No edge discard: the seamless ring cross-fade is the vertex scale-fade
    // (blades grow/shrink in height), so there's nothing to cut here.

    // Blades carry the *terrain* normal (not the card normal), so they light
    // like the ground they grow from and the card geometry doesn't read in the
    // shading. Shaded through the shared `shade_foliage` with a ground-matching
    // hemisphere fill (ambient_scale 1.0) and no leaf-transmit term.
    let n = normalize(in.world_normal);
    let sun_dir = grass.sun_dir.xyz;
    let up = grass.sky_up.xyz;

    // Same atmosphere-derived sky/sun environment the ground builds, so the
    // grass tracks the ground through the day and gets the same blue-sky fill.
    let sky = compute_surface_sky(grass.sky_tau.xyz, grass.sky_tau.w, up, sun_dir, grass.sun_dir.w);

    var s: FoliageSurface;
    s.albedo = in.color.rgb;
    s.normal_ws = n;
    s.translucency = 0.0;
    s.ambient_scale = 1.0;
    s.ambient_bleed = 0.0;
    let view_dir = normalize(view.world_position - in.world_position);
    // Per-vertex sun-shadow factor (tree/self shadows), gating the direct term.
    let lit = shade_foliage(s, view_dir, up, sun_dir, sky, clamp(in.shadow, 0.0, 1.0));
    return vec4<f32>(lit, 1.0);
}
