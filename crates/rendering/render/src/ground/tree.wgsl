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
#import thalos::lighting::{compute_surface_sky, FoliageSurface, shade_foliage, SurfaceSky, object_aerial_recession, sun_daylight}
#import thalos::foliage::{foliage_base_albedo, foliage_hue_tint}
#import thalos::shadow::{ShadowCascadeBlock, sun_shadow_factor, is_ortho_projection}

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
// Companion material atlas: bark tangent-space normal (rgb) + roughness (a).
// Linear data; shares `atlas_samp`. Only bark fragments read it.
@group(3) @binding(7) var material_tex: texture_2d<f32>;

// Cascaded sun-shadow maps (the SAME depth maps the terrain samples, published
// by the game's `rendering::sun_shadow`). `ShadowCascadeBlock` + `sun_shadow_factor`
// are shared from `thalos::shadow`. One `texture_depth_2d` per cascade (no array).
@group(3) @binding(3) var<uniform> tree_shadow: ShadowCascadeBlock;
@group(3) @binding(4) var sun_shadow_map_0: texture_depth_2d;
@group(3) @binding(5) var sun_shadow_map_1: texture_depth_2d;
@group(3) @binding(6) var sun_shadow_map_2: texture_depth_2d;

// Canopy shadow bleed into the ambient term — a tree in shade loses skylight as
// well as direct sun. Keeps shaded trees from staying ambient-bright (the same
// idea as `AMBIENT_SHADOW_BLEED` on the ground).
const TREE_AMBIENT_SHADOW_BLEED: f32 = 0.5;
// How strongly the baked leaf normal map perturbs the smooth crown normal (0 =
// flat cards, 1 = full per-leaf tilt). Gentle — just enough that neighbouring
// leaves catch light differently and read as separate, not embossed/noisy.
const LEAF_NORMAL_MIX: f32 = 0.55;

const TAU: f32 = 6.28318530717958647;
const PI: f32 = 3.14159265358979324;
// Foliage atlas: 4×4 cells, 256 px each (1024²). Cells 0..=11 are translucent
// foliage (leaves / needles); 12 = opaque shell; 13..=15 = opaque bark. The
// companion `material_tex` mirrors this layout (bark normal + roughness).
const ATLAS_N: f32 = 4.0;
const ATLAS_TEXEL: f32 = 1.0 / 1024.0;

// Soft sun sheen on bark ridges — bark is near-matte, so a weak, broad lobe.
const BARK_SPEC_F0: f32 = 0.04;
const BARK_SPEC_STRENGTH: f32 = 0.6;

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
    // In the sun-shadow CASTER pass (orthographic cascade camera) the fade
    // reference above reconstructs the craft from the WRONG camera — the
    // cascade eye, not the player view — so `grow` collapsed every caster to
    // zero once the two cameras diverged (shadows vanished as the camera
    // boomed away). Casters render at full scale instead: the depth map is a
    // silhouette union, so mesh + impostor rings overlapping is harmless.
    let grow = select(fade_in * fade_out, 1.0, is_ortho_projection(view.clip_from_view));
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

// Perturb the geometric normal `n` by the tangent-space `tn`, building the TBN
// from screen-space derivatives of world position + atlas UV (Schüler's
// cotangent frame). No mesh tangents needed — and it tracks the wrap-tiled bark
// UV per-fragment (the tangent follows the local UV gradient automatically).
fn perturb_normal(n: vec3<f32>, world_pos: vec3<f32>, uv: vec2<f32>, tn: vec3<f32>) -> vec3<f32> {
    let dp1 = dpdx(world_pos);
    let dp2 = dpdy(world_pos);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let invmax = inverseSqrt(max(dot(t, t), dot(b, b)));
    let tbn = mat3x3<f32>(t * invmax, b * invmax, n);
    return normalize(tbn * tn);
}

// Weak, broad GGX sun sheen for bark ridges (bark is near-matte). Gated by the
// sun shadow; scaled by the same `sun_color * sun_scale` as the diffuse so it
// stays photometrically consistent with the sky model.
fn bark_specular(
    n: vec3<f32>, view_dir: vec3<f32>, sun_dir: vec3<f32>,
    roughness: f32, sky: SurfaceSky, shadow: f32,
) -> vec3<f32> {
    let n_dot_l = dot(n, sun_dir);
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = normalize(sun_dir + view_dir);
    let n_dot_h = max(dot(n, h), 0.0);
    let a = max(roughness * roughness, 0.02);
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * denom * denom);
    let spec = d * BARK_SPEC_F0 * n_dot_l * shadow * BARK_SPEC_STRENGTH;
    return sky.sun_color * sky.sun_scale * spec;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Leaf shape + coverage from the procedural atlas; alpha-test the leaf cards.
    let tex = textureSample(atlas_tex, atlas_samp, in.atlas_uv);
    // Framebuffer alpha written below. Stays 1.0 (fully opaque) on the MSAA-off
    // path, so the result is byte-identical to a plain opaque draw; under MSAA the
    // alpha-to-coverage branch rewrites it to a sharpened fractional coverage.
    var coverage = 1.0;
#ifdef TREE_ALPHA_TO_COVERAGE
    // Anti-aliased alpha test (Castano): the leaf cards are a 1-bit cutout that
    // crawls on every edge. Rescale the sampled alpha to a ~1px ramp around the
    // 0.5 cutoff with screen-space derivatives, then let hardware
    // alpha-to-coverage convert it to an MSAA sample mask so leaf edges resolve
    // smooth. Enabled (with the pipeline's A2C flag) only when the camera runs
    // MSAA — see `TreeMaterial::specialize`. Bark/shell are alpha 1, so they stay
    // fully covered (coverage 1) and are unaffected.
    coverage = clamp((tex.a - 0.5) / max(fwidth(tex.a), 1.0e-4) + 0.5, 0.0, 1.0);
    if coverage <= 0.0 {
        discard;
    }
#else
    if tex.a < 0.5 {
        discard;
    }
#endif
    // The foliage atlas is composited over transparent BLACK, so partial-coverage
    // texels store premultiplied colour (rgb = colour × alpha). Un-premultiply, or
    // the kept alpha-test fringe (alpha 0.5→1) reads as colour fading to black — a
    // dark rim around every leaf. Bark/shell are opaque (alpha 1), so this is a
    // no-op there.
    let atlas_rgb = tex.rgb / max(tex.a, 1.0e-3);

    let sun_dir = tree.sun_dir.xyz;
    let up = tree.sky_up.xyz;
    let view_dir = normalize(view.world_position - in.world_position);

    // Same atmosphere-derived sky/sun environment the grass + ground build.
    let sky = compute_surface_sky(tree.sky_tau.xyz, tree.sky_tau.w, up, sun_dir, tree.sun_dir.w);

    // Sun-shadow: tree-on-tree, canopy self-shadow, and the ground's shadows
    // cast onto trunks/low foliage. Gates the direct + transmitted sun terms,
    // and bleeds partially into ambient (a shaded canopy sees less sky too).
    let shadow = sun_shadow_factor(
        in.world_position, tree_shadow, sun_shadow_map_0, sun_shadow_map_1, sun_shadow_map_2,
    );

    let is_bark = in.leaf < 0.5;
    var n = normalize(in.world_normal);
    var albedo: vec3<f32>;
    var roughness = 0.9;

    if (is_bark) {
        // Bark / shell: painterly atlas colour (decoupled from the dark
        // `trunk_color` tint) from the SHARED foliage material model, normal-mapped
        // + roughness from the material atlas.
        let mat = textureSample(material_tex, atlas_samp, in.atlas_uv);
        n = perturb_normal(n, in.world_position, in.atlas_uv, mat.xyz * 2.0 - 1.0);
        roughness = mat.w;
        albedo = foliage_base_albedo(atlas_rgb, in.color.g, in.leaf, in.seed);
    } else {
        // Foliage: intrinsic leaf colour from the SHARED `thalos::foliage` material
        // model (exposure grade + olive naturalize) — the SAME function the impostor
        // bake (`tree_bake.wgsl`) calls, so the near canopy and the far impostor band
        // cannot drift. Per-instance hue is applied here (and identically on the
        // impostor), never baked into the atlas. Directional sunlit-leaf pop is the
        // lighting model's job below, so the albedo stays view-independent.
        albedo = foliage_base_albedo(atlas_rgb, in.color.g, in.leaf, in.seed)
            * foliage_hue_tint(in.seed);

        // Per-leaf normal: perturb the smooth crown-outward normal with the baked
        // leaf normal map so individual leaves catch light differently — depth,
        // instead of the whole card shading flat. Eased by LEAF_NORMAL_MIX so it
        // textures the lighting without reading as noisy.
        let lmat = textureSample(material_tex, atlas_samp, in.atlas_uv);
        let lnrm = normalize(mix(vec3<f32>(0.0, 0.0, 1.0), lmat.xyz * 2.0 - 1.0, LEAF_NORMAL_MIX));
        n = perturb_normal(n, in.world_position, in.atlas_uv, lnrm);
    }

    // Shade through the shared `shade_foliage`: wrap-diffuse direct + hemisphere
    // sky IBL + the two-sided leaf transmit. Canopies dim their ambient (0.8) and
    // bleed the sun-shadow into it; the per-vertex `leaf` flag drives the transmit
    // (1 = foliage, 0 = bark/shell).
    var s: FoliageSurface;
    s.albedo = albedo;
    s.normal_ws = n;
    s.translucency = in.leaf;
    s.ambient_scale = 0.8;
    s.ambient_bleed = TREE_AMBIENT_SHADOW_BLEED;
    var lit = shade_foliage(s, view_dir, up, sun_dir, sky, shadow);

    // Bark catches a soft sun sheen on its ridges (foliage stays matte). Gated by
    // the sun's own horizon (`sun_daylight`): a trunk facet whose horizontal normal
    // points at the buried sun still passes `bark_specular`'s n·l>0 test at night,
    // so without this the trunk keeps a faint sun glint after dark.
    if (is_bark) {
        lit += bark_specular(n, view_dir, sun_dir, roughness, sky, shadow)
            * sun_daylight(dot(up, sun_dir));
    }

    // Recede toward the air with distance, earlier than the terrain's BodySky
    // veil, so the canopy doesn't stay crisp-green against hazed terrain.
    lit = object_aerial_recession(lit, sky, in.world_position, view.world_position);

    // `coverage` is 1.0 on the MSAA-off path (identical opaque output) and the
    // sharpened A2C coverage under MSAA. Colour is written un-premultiplied — the
    // hardware sample mask handles coverage, not the blend.
    return vec4<f32>(lit, coverage);
}
