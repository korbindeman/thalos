// Tree / shrub instanced shader.
//
// Vertex: standard mesh transform plus a world-space wind sway, weighted by the
// per-vertex wind weight in the vertex-colour alpha (0 trunk → 1 canopy top) and
// phase-shifted per instance (hashed from the instance world position) so a
// stand of trees doesn't sway in unison.
//
// Fragment: the SAME hemisphere sky model the grass and ground use, pulled from
// `thalos::lighting` (`compute_surface_sky` / `sky_ambient_irradiance`), so
// plants light identically to their surroundings. A small per-instance hue jitter
// (same hash) keeps a species from reading as copy-pasted. Opaque pass.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import thalos::lighting::{SurfaceSky, compute_surface_sky, sky_ambient_irradiance}

struct TreeParams {
    // xyz = unit direction toward the star (world render space), w = sun flux
    // (lux × exposure gain — the terrain `SceneLighting` star value).
    sun_dir: vec4<f32>,
    // xyz = wind direction (world render space), w = canopy sway amplitude (m).
    wind: vec4<f32>,
    // x = time (s); y/z/w unused for trees.
    time_fade: vec4<f32>,
    // xyz = local radial up (world render space) for the sky hemisphere split.
    sky_up: vec4<f32>,
    // xyz = Rayleigh vertical optical depth τ_v, w = atmosphere strength.
    sky_tau: vec4<f32>,
    // xyz = vegetation focus (player craft) in render space; w = 1 valid / 0 use
    // camera. The radial fade measures distance from THIS, not the camera, so
    // zooming / orbiting the camera doesn't change what's drawn.
    anchor: vec4<f32>,
}

// Standard MaterialPlugin bind group in Bevy 0.18: group 3.
@group(3) @binding(0) var<uniform> tree: TreeParams;

const TAU: f32 = 6.28318530717958647;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Tree base (this tree's root, tile-centre-relative), baked by the per-tile
    // combiner: uv0 = base.xy, uv1.x = base.z. Drives the per-tree scale-fade and
    // a stable wind/tint seed.
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    // rgb = trunk/canopy tint (linear), a = wind weight (0 trunk → 1 canopy top).
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) seed: f32,
}

fn hash1(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);

    // The mesh is ONE batched mesh for the whole tile; each tree's base (its
    // root, tile-centre-relative) is baked into the UVs. Vertices are relative to
    // the tile centre, so the per-tree seed (hashed from the base) is stable and
    // rebase-invariant.
    let base = vec3<f32>(in.uv0.x, in.uv0.y, in.uv1.x);
    let seed = hash1(base);

    // Per-tree scale-fade: each tree GROWS from zero at the far edge, full inside
    // the near edge, scaled about ITS OWN base — seamless (no dither, no pop-in);
    // a fully-collapsed tree is a degenerate (invisible) mesh, so no discard.
    // Distance is from the focus anchor (craft), so camera zoom/orbit doesn't
    // change it.
    let base_world = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(base, 1.0)).xyz;
    let ref_pos = select(view.world_position, tree.anchor.xyz, tree.anchor.w > 0.5);
    let inst_dist = distance(ref_pos, base_world);
    let fs = tree.time_fade.y;
    let fe = tree.time_fade.z;
    var grow = 1.0;
    if fe > fs {
        grow = clamp((fe - inst_dist) / (fe - fs), 0.0, 1.0);
    }
    let local = base + (in.position - base) * grow;
    var world_pos =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0)).xyz;

    // Wind sway: two incommensurate sines, scaled by the per-vertex weight so
    // the trunk is rigid and the canopy top moves most (and by `grow` so a
    // shrinking tree doesn't sway oddly).
    let t = tree.time_fade.x;
    let phase = seed * TAU;
    let weight = in.color.a;
    let gust = 0.6 * sin(1.1 * t + phase) + 0.4 * sin(2.3 * t + phase * 1.7);
    world_pos += tree.wind.xyz * (weight * tree.wind.w * (0.5 + 0.5 * gust) * grow);

    var out: VertexOutput;
    out.world_position = world_pos;
    out.clip_position = position_world_to_clip(world_pos);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
    out.color = in.color;
    out.seed = seed;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // No edge discard: the seamless far fade is the vertex scale-fade (trees grow
    // from zero), so there's nothing to cut here.
    let n = normalize(in.world_normal);
    let sun_dir = tree.sun_dir.xyz;
    let up = tree.sky_up.xyz;

    // Same atmosphere-derived sky/sun environment the grass + ground build.
    let sky = compute_surface_sky(tree.sky_tau.xyz, tree.sky_tau.w, up, sun_dir, tree.sun_dir.w);

    // Per-instance hue jitter (warm/cool) so a species isn't visibly stamped.
    let hue = (in.seed - 0.5) * 0.22;
    let tint = in.color.rgb * vec3<f32>(1.0 + hue, 1.0, 1.0 - hue);

    // Direct: wrap diffuse (foliage is translucent), reddened + exposure-scaled
    // by the shared sun term.
    let n_dot_l = dot(n, sun_dir);
    let wrap = clamp((n_dot_l + 0.3) / 1.3, 0.0, 1.0);
    let direct = tint * (wrap * sky.sun_scale) * sky.sun_color;

    // Ambient: the hemisphere sky model (blue sky-dome + warm ground bounce).
    let ambient = tint * sky_ambient_irradiance(sky, n, up);

    return vec4<f32>(direct + ambient, 1.0);
}
