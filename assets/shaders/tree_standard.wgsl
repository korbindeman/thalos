// Tree / shrub instanced material — STANDARD-PATH port (keystone: one lighting
// universe). The batched tile mesh, wind sway, clipmap grow-fade, foliage-atlas
// alpha-tested cards, and the shared `thalos::foliage` albedo model are all
// unchanged from the spine-era `tree.wgsl`; what changed is WHO lights the
// result. The fragment now builds a stock `PbrInput` and shades through
// `apply_pbr_lighting`, so trees are lit by the SAME Bevy sun (airmass-reddened
// by `rendering::lighting::update_sun_light`), the same `GlobalAmbientLight`
// fill, and the same exposure/tonemap as the tile ground and the hull — the
// tree/ground colour split was two lighting universes disagreeing, not a
// palette problem (backlog: vegetation-on-standard-path; NTR-X5 residual).
//
// Ported over from the spine version:
//   * the shared `thalos::shadow` cascade receive (direct-only gate, exactly
//     like `shadowed_standard.wgsl` / `tile_terrain.wgsl`), and
//   * the cloud sun-transmittance gate (`thalos::cloud_shadow`) — NEW for
//     trees: the spine tree never sampled it, so a forest under a cloud deck
//     stayed in full sun while the ground beside it dimmed (one of the two
//     mechanisms behind "trees look pasted on").
//
// Deliberately dropped, with reasons:
//   * `compute_surface_sky` / `shade_foliage` — replaced by the standard path;
//     the wrap-diffuse's job (shaded side stays leafy) is carried by the
//     StandardMaterial `diffuse_transmission` the leaves now declare.
//   * `object_aerial_recession` — the BodySky composite applies aerial
//     perspective by scene depth to everything on the standard path; a
//     material-side veil on top would double-fog the canopy.
//   * the custom bark GGX sheen — stock PBR's specular with the bark atlas
//     roughness is the same lobe, driven by the same sun.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
#import thalos::foliage::foliage_base_albedo
#import thalos::shadow::{ShadowCascadeBlock, sun_shadow_factor, is_ortho_projection}
#import thalos::cloud_shadow::{CloudShadowBlock, cloud_sun_transmittance}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

// Mirror of the Rust `GrassParams` (field order load-bearing) — same uniform
// the spine tree/grass materials carry, minus any lighting duty: only wind,
// time/fade band, and the anchor are read here now.
struct TreeParams {
    // xyz = unit direction toward the star (kept for layout parity; lighting
    // comes from the Bevy sun now), w = sun flux (unused here).
    sun_dir: vec4<f32>,
    // xyz = wind direction (world render space), w = canopy sway amplitude (m).
    wind: vec4<f32>,
    // x = time (s); y/z = ring near/far fade edges (m); w = fade band (m).
    time_fade: vec4<f32>,
    // xyz = local radial up (layout parity; unused on the standard path).
    sky_up: vec4<f32>,
    // xyz = Rayleigh τ_v, w = strength (layout parity; unused).
    sky_tau: vec4<f32>,
    // xyz = fade-reference OFFSET = (craft − camera) in render space; w unused.
    // The reference is rebuilt as `view.world_position + offset` per frame so it
    // survives big_space recentres (see the spine tree.wgsl note).
    anchor: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> tree: TreeParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var atlas_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var atlas_samp: sampler;
// Companion material atlas: bark tangent-space normal (rgb) + roughness (a).
@group(#{MATERIAL_BIND_GROUP}) @binding(103)
var material_tex: texture_2d<f32>;
// Shared cascaded sun-shadow receive (same maps the terrain samples).
@group(#{MATERIAL_BIND_GROUP}) @binding(104)
var<uniform> tree_shadow: ShadowCascadeBlock;
@group(#{MATERIAL_BIND_GROUP}) @binding(111)
var sun_shadow_map_0: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(105)
var sun_shadow_map_1: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(106)
var sun_shadow_map_2: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(107)
var sun_shadow_map_3: texture_depth_2d;
// Cloud sun-transmittance cascade — same block/texture the tile ground samples,
// fanned in by `apply_tree_cloud_shadow`.
@group(#{MATERIAL_BIND_GROUP}) @binding(108)
var cloud_shadow_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(109)
var cloud_shadow_samp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(110)
var<uniform> cloud_shadow: CloudShadowBlock;

// Canopy shadow bleed into the ambient term — a tree in shade loses skylight as
// well as direct sun (same idea as the ground's ambient bleed).
const TREE_AMBIENT_SHADOW_BLEED: f32 = 0.5;
// How strongly the baked leaf normal map perturbs the smooth crown normal.
const LEAF_NORMAL_MIX: f32 = 0.55;
// Foliage is not a generic dielectric (Bevy reflectance convention,
// F0 = 0.16·r²): leaves scatter far more than they mirror — same constant
// family as the tile ground's `VEG_REFLECTANCE`. Bark keeps the stock 0.5.
const LEAF_REFLECTANCE: f32 = 0.32;
// Two-sided diffuse transmission for leaf cards. Mirrors the base material's
// `diffuse_transmission` (which switches the pipeline branch on); the fragment
// zeroes it on bark. This is what keeps a backlit canopy luminous now that the
// spine's warm transmit lobe is gone.
const LEAF_DIFFUSE_TRANSMISSION: f32 = 0.35;

const TAU: f32 = 6.28318530717958647;
// Foliage atlas: 4×4 cells, 256 px each (1024²). Cells 0..=11 translucent
// foliage; 12 = opaque shell; 13..=15 = opaque bark.
const ATLAS_N: f32 = 4.0;
const ATLAS_TEXEL: f32 = 1.0 / 1024.0;

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

// The standard `VertexOutput` has no spare varyings, so two of its fields are
// repurposed (both stages are this file, so the contract is local):
//   uv   = interpolated foliage-atlas UV (the per-corner decode must happen
//          per vertex — the leaf code itself does not interpolate meaningfully)
//   uv_b = (per-tree seed, leaf flag 1|0)
// COLOR passes through as authored: rgb = landcover tint ratios × AO in g,
// a = wind weight. The base material binds no textures, so nothing else reads
// `uv` with its standard meaning.
@vertex
fn vertex(in: Vertex) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);

    // One batched mesh per tile; each tree's base (tile-centre-relative) is
    // baked into the UVs: uv = base.xy, uv_b.x = base.z, uv_b.y = leaf code.
    let base = vec3<f32>(in.uv.x, in.uv.y, in.uv_b.x);
    let seed = hash1(base);

    // Per-tree clipmap scale-fade about its own base (grow from zero across the
    // ring's near/far edges) — see the spine tree.wgsl for the full rationale.
    let base_world = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(base, 1.0)).xyz;
    let ref_pos = view.world_position + tree.anchor.xyz;
    let inst_dist = distance(ref_pos, base_world);
    let near_edge = tree.time_fade.y;
    let far_edge = tree.time_fade.z;
    let band = max(tree.time_fade.w, 1.0);
    let fade_in = smoothstep(near_edge - band, near_edge + band, inst_dist);
    let fade_out = 1.0 - smoothstep(far_edge - band, far_edge + band, inst_dist);
    // Sun-shadow caster pass (ortho cascade camera): the fade reference is the
    // wrong camera there — cast at full scale; the depth map is a silhouette
    // union.
    let grow = select(fade_in * fade_out, 1.0, is_ortho_projection(view.clip_from_view));
    let local = base + (in.position - base) * grow;
    var world_pos =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0)).xyz;

    // Wind sway: two incommensurate sines, trunk rigid → canopy top moves most.
    let t = tree.time_fade.x;
    let phase = seed * TAU;
    let weight = in.color.a;
    let gust = 0.6 * sin(1.1 * t + phase) + 0.4 * sin(2.3 * t + phase * 1.7);
    world_pos += tree.wind.xyz * (weight * tree.wind.w * (0.5 + 0.5 * gust) * grow);

    let cell = floor(in.uv_b.y / 4.0);

    var out: VertexOutput;
    out.world_position = vec4<f32>(world_pos, 1.0);
    out.position = position_world_to_clip(world_pos);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index);
#ifdef VERTEX_UVS_A
    out.uv = atlas_uv_of(in.uv_b.y);
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vec2<f32>(seed, select(0.0, 1.0, cell < 11.5));
#endif
#ifdef VERTEX_COLORS
    out.color = in.color;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = in.instance_index;
#endif
    return out;
}

// Perturb the geometric normal `n` by the tangent-space `tn`, building the TBN
// from screen-space derivatives (Schüler's cotangent frame) — no mesh tangents.
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

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Build the standard PbrInput FIRST (uniform control flow for its own
    // derivative use), then overwrite the material fields the atlas drives.
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Leaf shape + coverage from the procedural atlas; alpha-test the cards.
    let tex = textureSample(atlas_tex, atlas_samp, in.uv);
    var coverage = 1.0;
#ifdef TREE_ALPHA_TO_COVERAGE
    // Anti-aliased alpha test (Castano): rescale to a ~1px ramp around the 0.5
    // cutoff; hardware alpha-to-coverage turns it into an MSAA sample mask.
    coverage = clamp((tex.a - 0.5) / max(fwidth(tex.a), 1.0e-4) + 0.5, 0.0, 1.0);
    if coverage <= 0.0 {
        discard;
    }
#else
    if tex.a < 0.5 {
        discard;
    }
#endif
    // Atlas is composited over transparent black → partial-coverage texels are
    // premultiplied; un-premultiply or the kept fringe darkens to a rim.
    let atlas_rgb = tex.rgb / max(tex.a, 1.0e-3);

    let seed = in.uv_b.x;
    let leaf = in.uv_b.y;
    let is_bark = leaf < 0.5;

    // Base the shading normal on `pbr_input.world_normal`, NOT the raw
    // `in.world_normal` varying: `prepare_world_normal` has already negated it
    // on back-facing fragments (double_sided, no tangents/normal map), and the
    // card's whole two-sided model — dim diffuse + warm transmission on the
    // anti-sun side — rides on that flip. Rebuilding from the varying discards
    // it, and every back-facing leaf shades front-lit with a dead transmit
    // lobe (reviews/20260730T011353Z §1).
    let n_geo = normalize(pbr_input.world_normal);
    var n = n_geo;
    var albedo: vec3<f32>;
    var roughness = 0.95;
    var reflectance = LEAF_REFLECTANCE;

    if (is_bark) {
        // Bark / shell: shared foliage material model + material-atlas normal
        // and roughness; stock dielectric reflectance.
        let mat = textureSample(material_tex, atlas_samp, in.uv);
        n = perturb_normal(n, in.world_position.xyz, in.uv, mat.xyz * 2.0 - 1.0);
        roughness = mat.w;
        reflectance = 0.5;
        albedo = foliage_base_albedo(atlas_rgb, in.color.g, leaf, seed);
    } else {
        // Foliage: the SHARED `thalos::foliage` albedo (same function the
        // impostor bake calls), per-instance hue jitter, and the landcover
        // tint the tile combiner bakes into COLOR r/b (g ≡ 1 for trees — the
        // baked AO rides the atlas-graded exposure instead).
        // Per-instance hue jitter now rides COLOR r/b (folded into the
        // landcover tint CPU-side from the ring-invariant Poisson cell —
        // hashing the tile-relative root here gave the same tree two hues
        // across the ring cross-fade).
        albedo = foliage_base_albedo(atlas_rgb, in.color.g, leaf, seed)
            * vec3<f32>(in.color.r, 1.0, in.color.b);
        // Per-leaf normal from the baked leaf normal map, eased so neighbouring
        // leaves catch light differently without reading as embossed noise.
        let lmat = textureSample(material_tex, atlas_samp, in.uv);
        let lnrm = normalize(mix(vec3<f32>(0.0, 0.0, 1.0), lmat.xyz * 2.0 - 1.0, LEAF_NORMAL_MIX));
        n = perturb_normal(n, in.world_position.xyz, in.uv, lnrm);
    }

    pbr_input.material.base_color = vec4<f32>(albedo, coverage);
    pbr_input.material.perceptual_roughness = roughness;
    pbr_input.material.reflectance = vec3<f32>(reflectance);
    // Leaves transmit (two-sided diffuse); bark does not. The base material
    // declares a non-zero value so the pipeline branch is compiled in.
    pbr_input.material.diffuse_transmission = LEAF_DIFFUSE_TRANSMISSION * leaf;
    pbr_input.N = n;
    // `pbr_input.world_normal` keeps the (flipped) geometric normal: it drives
    // the stable-CSM receiver offset, and a per-leaf N would wobble it.

    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    // Shared sun-shadow cascade (tree-on-tree, canopy self-shadow, ground and
    // structure shadows onto trunks) × the cloud deck's sun transmittance —
    // one gate for the whole direct beam, exactly as the tile ground composes
    // it. `axis_v.w == 0` (no cloud body / clouds off) reads fully lit.
    let hard_shadow = sun_shadow_factor(
        in.world_position.xyz,
        tree_shadow,
        sun_shadow_map_0,
        sun_shadow_map_1,
        sun_shadow_map_2,
        sun_shadow_map_3,
    );
    let cloud_t = cloud_sun_transmittance(
        cloud_shadow,
        cloud_shadow_tex,
        cloud_shadow_samp,
        in.world_position.xyz,
    );
    let shadow_f = hard_shadow * cloud_t;

    // A shaded canopy sees less sky too: bleed the shadow into the indirect
    // occlusions before the split, so the ambient share dims by
    // `TREE_AMBIENT_SHADOW_BLEED` where the sun is blocked.
    let ambient_bleed = mix(1.0, shadow_f, TREE_AMBIENT_SHADOW_BLEED);
    pbr_input.diffuse_occlusion *= vec3<f32>(ambient_bleed);
    pbr_input.specular_occlusion *= ambient_bleed;

    // Direct/indirect split (exact, by linearity — `shadowed_standard.wgsl`):
    // occlusions zeroed ⇒ pure exposure·direct; the shadow subtracts only the
    // sun's share and the canopy keeps its whole sky fill.
    var pbr_direct = pbr_input;
    pbr_direct.diffuse_occlusion = vec3<f32>(0.0);
    pbr_direct.specular_occlusion = 0.0;
    pbr_direct.material.emissive = vec4<f32>(0.0);
    let direct = apply_pbr_lighting(pbr_direct);
    out.color = apply_pbr_lighting(pbr_input);
    out.color = vec4<f32>(
        max(out.color.rgb - (1.0 - shadow_f) * direct.rgb, vec3<f32>(0.0)),
        out.color.a,
    );
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
