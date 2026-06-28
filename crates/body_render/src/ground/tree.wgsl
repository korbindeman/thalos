// Tree / shrub instanced shader.
//
// Vertex: standard mesh transform plus a world-space wind sway, weighted by the
// per-vertex wind weight in the vertex-colour alpha (0 trunk → 1 canopy top) and
// phase-shifted per instance (hashed from the instance world position) so a
// stand of trees doesn't sway in unison.
//
// Fragment: samples the procedural foliage atlas (leaf-cluster / shell / bark)
// for shape + alpha (alpha-tested leaf cards), tinted by the per-vertex hue × AO,
// then lights it with the SAME hemisphere sky model the grass and ground use,
// pulled from `thalos::lighting` (`compute_surface_sky` /
// `sky_ambient_irradiance`), so plants light identically to their surroundings.
// Adds a two-sided **translucency** term so backlit leaves transmit a warm glow.
// Opaque pass with discard.

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
    // xyz = vegetation focus OFFSET = (player craft − camera) in render space;
    // w = 1 valid / 0 = camera. The fade reference is rebuilt as
    // `view.world_position + offset` (the craft in the current render origin),
    // so the fade tracks the craft (zoom/orbit-independent) AND survives
    // big_space recentres — see the `vertex` fn.
    anchor: vec4<f32>,
}

// Standard MaterialPlugin bind group in Bevy 0.18: group 3.
@group(3) @binding(0) var<uniform> tree: TreeParams;
@group(3) @binding(1) var atlas_tex: texture_2d<f32>;
@group(3) @binding(2) var atlas_samp: sampler;

// Cascaded sun-shadow maps (the SAME depth maps the terrain samples, published
// by the game's `rendering::sun_shadow`). Separate `texture_depth_2d` per
// cascade (no depth array). Mirrors `body_terrain.wgsl`.
struct ShadowCascadeBlock {
    view_proj: array<mat4x4<f32>, 3>,
    // per cascade: x = depth bias (clip), yzw reserved.
    params: array<vec4<f32>, 3>,
    // x = strength (0 ⇒ skip), y = active cascade count, zw reserved. Named
    // `gate` to match `body_terrain.wgsl` (where `config` collides with a udlod
    // import); layout is by field order, so the name is free to choose.
    gate: vec4<f32>,
}
@group(3) @binding(3) var<uniform> tree_shadow: ShadowCascadeBlock;
@group(3) @binding(4) var sun_shadow_map_0: texture_depth_2d;
@group(3) @binding(5) var sun_shadow_map_1: texture_depth_2d;
@group(3) @binding(6) var sun_shadow_map_2: texture_depth_2d;

// One cascade's shadow factor, or a negative sentinel if outside its box (caller
// falls through to the next cascade). See `body_terrain.wgsl::cascade_factor`.
fn cascade_factor(
    world_pos: vec3<f32>,
    vp: mat4x4<f32>,
    bias: f32,
    strength: f32,
    tex: texture_depth_2d,
    inset: f32,
    fade: bool,
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
    var lit = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let texel = vec2<i32>(uv * dims) + vec2<i32>(dx, dy);
            let stored = textureLoad(tex, texel, 0);
            lit = lit + select(1.0, 0.0, stored > ndc.z + bias);
        }
    }
    lit = lit / 9.0;
    var edge_fade = 1.0;
    if (fade) {
        let edge = max(abs(ndc.x), abs(ndc.y));
        edge_fade = 1.0 - smoothstep(0.85, 1.0, edge);
    }
    return 1.0 - strength * (1.0 - lit) * edge_fade;
}

// Canopy/ground shadow factor: walk cascades near→far, use the tightest hit.
// `config.x == 0` (inactive / preview) early-outs to fully lit.
fn sun_shadow_factor(world_pos: vec3<f32>) -> f32 {
    let s = tree_shadow.gate.x;
    if (s <= 0.0) {
        return 1.0;
    }
    var f = cascade_factor(
        world_pos, tree_shadow.view_proj[0], tree_shadow.params[0].x,
        s, sun_shadow_map_0, 0.98, false,
    );
    if (f < 0.0) {
        f = cascade_factor(
            world_pos, tree_shadow.view_proj[1], tree_shadow.params[1].x,
            s, sun_shadow_map_1, 0.98, false,
        );
    }
    if (f < 0.0) {
        f = cascade_factor(
            world_pos, tree_shadow.view_proj[2], tree_shadow.params[2].x,
            s, sun_shadow_map_2, 1.0, true,
        );
    }
    if (f < 0.0) {
        return 1.0;
    }
    return f;
}

// Canopy shadow bleed into the ambient term — a tree in shade loses skylight as
// well as direct sun. Keeps shaded trees from staying ambient-bright (the same
// idea as `AMBIENT_SHADOW_BLEED` on the ground).
const TREE_AMBIENT_SHADOW_BLEED: f32 = 0.5;

const TAU: f32 = 6.28318530717958647;
// Foliage atlas: 4×4 cells, 128 px each (512²). Cells 0..=11 are translucent
// foliage (leaves / needles); 12 = opaque shell; 13..=15 = opaque bark.
const ATLAS_N: f32 = 4.0;
const ATLAS_TEXEL: f32 = 1.0 / 1024.0;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Tree base (this tree's root, tile-centre-relative), baked by the per-tile
    // combiner: uv0 = base.xy, uv1.x = base.z. Drives the per-tree scale-fade and
    // a stable wind/tint seed. uv1.y = atlas leaf code (cell·4 + corner).
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    // rgb = trunk/canopy tint × AO (linear), a = wind weight (0 trunk → 1 top).
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) seed: f32,
    @location(4) atlas_uv: vec2<f32>,
    // 1 = translucent foliage (leaf/needle), 0 = opaque (shell/bark).
    @location(5) leaf: f32,
}

fn hash1(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

// Decode `cell·4 + corner` into the atlas UV for this vertex's corner, inset by
// half a texel so bilinear filtering never bleeds across cell borders.
fn atlas_uv_of(code: f32) -> vec2<f32> {
    let cell = floor(code / 4.0);
    let corner = code - cell * 4.0; // 0=BL,1=BR,2=TR,3=TL
    let col = cell - floor(cell / ATLAS_N) * ATLAS_N;
    let row = floor(cell / ATLAS_N);
    let cu = select(0.0, 1.0, corner == 1.0 || corner == 2.0);
    let cv = select(0.0, 1.0, corner == 2.0 || corner == 3.0);
    let cell_size = 1.0 / ATLAS_N;
    let iu = mix(ATLAS_TEXEL, cell_size - ATLAS_TEXEL, cu);
    let iv = mix(ATLAS_TEXEL, cell_size - ATLAS_TEXEL, cv);
    return vec2<f32>(col * cell_size + iu, row * cell_size + iv);
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

    // Per-tree clipmap scale-fade: each tree GROWS from zero across its ring's
    // near/far edges, scaled about ITS OWN base — so adjacent rings cross-fade by
    // growing/shrinking (seamless — no dither, no pop-in; a fully-collapsed tree
    // is a degenerate, invisible mesh, so no discard). Distance is from the focus
    // anchor (craft), so camera zoom/orbit doesn't change it. The innermost
    // (mesh) ring passes a large-negative near edge so it never fades in.
    let base_world = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(base, 1.0)).xyz;
    // Anchor is a camera-relative OFFSET (ship − camera), rebuilt here in the
    // current frame's render origin. An absolute anchor is one frame stale and
    // jumps a cell across a big_space floating-origin recentre, popping trees
    // in/out while moving (see `rendering::grass`). offset 0 → camera.
    let ref_pos = view.world_position + tree.anchor.xyz;
    let inst_dist = distance(ref_pos, base_world);
    let near_edge = tree.time_fade.y;
    let far_edge = tree.time_fade.z;
    let band = max(tree.time_fade.w, 1.0);
    let fade_in = smoothstep(near_edge - band, near_edge + band, inst_dist);
    let fade_out = 1.0 - smoothstep(far_edge - band, far_edge + band, inst_dist);
    let grow = fade_in * fade_out;
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

    let cell = floor(in.uv1.y / 4.0);

    var out: VertexOutput;
    out.world_position = world_pos;
    out.clip_position = position_world_to_clip(world_pos);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
    out.color = in.color;
    out.seed = seed;
    out.atlas_uv = atlas_uv_of(in.uv1.y);
    out.leaf = select(0.0, 1.0, cell < 11.5);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Leaf shape + coverage from the procedural atlas; alpha-test the leaf cards.
    let tex = textureSample(atlas_tex, atlas_samp, in.atlas_uv);
    if tex.a < 0.5 {
        discard;
    }

    let n = normalize(in.world_normal);
    let sun_dir = tree.sun_dir.xyz;
    let up = tree.sky_up.xyz;

    // Same atmosphere-derived sky/sun environment the grass + ground build.
    let sky = compute_surface_sky(tree.sky_tau.xyz, tree.sky_tau.w, up, sun_dir, tree.sun_dir.w);

    // Per-instance hue jitter (warm/cool) so a species isn't visibly stamped; the
    // atlas RGB supplies leaf luminance detail, the vertex colour the hue × AO.
    let hue = (in.seed - 0.5) * 0.22;
    let tint = in.color.rgb * vec3<f32>(1.0 + hue, 1.0, 1.0 - hue);
    let albedo = tex.rgb * tint;

    // Sun-shadow: tree-on-tree, canopy self-shadow, and the ground's shadows
    // cast onto trunks/low foliage. Gates the direct + transmitted sun terms,
    // and bleeds partially into ambient (a shaded canopy sees less sky too).
    let shadow = sun_shadow_factor(in.world_position);

    // Direct: wrap diffuse — foliage scatters a lot, so wrap past the terminator
    // so the shaded side stays leafy rather than black. Eased from 0.55 → 0.40
    // so canopies keep a readable shaded side (less flat-blob, more form).
    let n_dot_l = dot(n, sun_dir);
    let wrap = clamp((n_dot_l + 0.40) / 1.40, 0.0, 1.0);
    let direct = albedo * (wrap * sky.sun_scale * shadow) * sky.sun_color;

    // Ambient: the hemisphere sky model (blue sky-dome + warm ground bounce).
    // Kept modest — too much and the blue sky-dome washes the upward-facing top
    // leaves to grey-blue; the wrap term above already lifts the shadow side.
    let ambient = albedo * sky_ambient_irradiance(sky, n, up)
        * (0.8 * mix(1.0, shadow, TREE_AMBIENT_SHADOW_BLEED));

    // Two-sided translucency: backlit leaves transmit a warm forward-scattered
    // glow. A view-dependent lobe (looking toward the sun through the leaf) plus
    // a softer isotropic through-scatter, so the whole sunlit-from-behind canopy
    // glows, not just the rim.
    let v = normalize(view.world_position - in.world_position);
    let lt_dir = normalize(sun_dir + n * 0.30);
    let back = pow(clamp(dot(v, -lt_dir), 0.0, 1.0), 2.5);
    let warm = vec3<f32>(1.30, 1.05, 0.50); // green → yellow/orange shift
    let thru = (back + 0.16 * clamp(-n_dot_l, 0.0, 1.0)) * in.leaf;
    let transmit = albedo * warm * (thru * sky.sun_scale * shadow) * sky.sun_color;

    return vec4<f32>(direct + ambient + transmit, 1.0);
}
