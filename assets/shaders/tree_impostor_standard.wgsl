// Octahedral tree impostor — STANDARD-PATH port (far band of the tree LOD
// cascade). Billboard/atlas-blend logic unchanged from the spine version; the
// lighting moved to Bevy's standard path in the same change as the mesh trees
// (`tree_standard.wgsl`), so the mesh→impostor handoff keeps reading as one
// continuous forest under the ONE lighting universe: same airmass-reddened
// sun, same ambient fill, same exposure — and now the same **cloud
// sun-transmittance gate**, which the spine impostor never sampled (a distant
// forest under the deck stayed lit while the ground dimmed).
//
// The standard `VertexOutput` has no spare varyings, so fields are repurposed
// (both stages live in this file):
//   world_normal = terrain up (the tree frame's vertical)
//   uv           = card corner UV
//   uv_b         = (per-tree seed, species index)
//   color        = (tint.rgb, yaw / TAU)
// The card view direction `d_local` is reconstructed per fragment from the
// fragment→camera ray and the rebuilt tree frame; against the per-quad value
// the spine carried this differs by the card's parallax (< card size /
// impostor distance — sub-atlas-cell at the ranges impostors draw).
//
// Dropped deliberately: `object_aerial_recession` (the BodySky composite fogs
// by scene depth on the standard path; impostors write depth).

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
}
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

// Mirror of the Rust `GrassParams` (field order load-bearing); only the fade
// band + anchor are read here now.
struct TreeParams {
    sun_dir: vec4<f32>,
    wind: vec4<f32>,
    time_fade: vec4<f32>, // x = time, y/z = ring near/far fade edges, w = band
    sky_up: vec4<f32>,
    sky_tau: vec4<f32>,
    anchor: vec4<f32>,    // xyz = (craft − camera) offset; ref = view pos + offset
}

struct ImpostorParams {
    grid: vec4<f32>,  // x = cells N, y = species count, z = alpha cutoff, w = v-flip
    atlas: vec4<f32>, // x = cell fill fraction
    species_geo: array<vec4<f32>, 4>, // per species: x = radius, y = centre height
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> tree: TreeParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var<uniform> imp: ImpostorParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var albedo_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103)
var albedo_smp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(104)
var normal_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105)
var normal_smp: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(106)
var cloud_shadow_tex: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(107)
var cloud_shadow_samp: sampler;
// Cascaded sun-shadow receive — see `TreeImpostorExtension::shadow`.
@group(#{MATERIAL_BIND_GROUP}) @binding(109)
var<uniform> imp_shadow: ShadowCascadeBlock;
@group(#{MATERIAL_BIND_GROUP}) @binding(113)
var imp_shadow_map_0: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(110)
var imp_shadow_map_1: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(111)
var imp_shadow_map_2: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(112)
var imp_shadow_map_3: texture_depth_2d;
@group(#{MATERIAL_BIND_GROUP}) @binding(108)
var<uniform> cloud_shadow: CloudShadowBlock;

// Same foliage optical constants as the mesh trees (`tree_standard.wgsl`) so
// the handoff stays photometrically continuous.
const LEAF_REFLECTANCE: f32 = 0.32;
const LEAF_DIFFUSE_TRANSMISSION: f32 = 0.35;

const TAU: f32 = 6.28318530717958647;

fn hash1(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

// Card basis for a view direction (object→camera), matching `impostor_bake_rotation`.
fn view_basis_right_up(fwd: vec3<f32>) -> mat2x3<f32> {
    let f = normalize(fwd);
    var up_ref = vec3<f32>(0.0, 1.0, 0.0);
    if abs(f.y) > 0.999 {
        up_ref = vec3<f32>(0.0, 0.0, 1.0);
    }
    let r = normalize(cross(up_ref, f));
    let u = cross(f, r);
    return mat2x3<f32>(r, u);
}

// Tree terrain frame (tangent, bitangent) from world up + per-tree yaw. One
// definition used by BOTH stages so the fragment rebuilds exactly the frame the
// vertex billboarded with.
fn tree_frame(up_w: vec3<f32>, yaw: f32) -> mat2x3<f32> {
    var ref_axis = vec3<f32>(0.0, 0.0, 1.0);
    if abs(up_w.y) > 0.99 {
        ref_axis = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t0 = normalize(cross(ref_axis, up_w));
    let b0 = cross(up_w, t0);
    let cw = cos(yaw);
    let sw = sin(yaw);
    return mat2x3<f32>(t0 * cw + b0 * sw, -t0 * sw + b0 * cw);
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    // Mesh attributes: POSITION = tree base (all four corners), NORMAL =
    // terrain up, uv = card corner, uv_b = (instance scale, species index),
    // COLOR = (tint.rgb, yaw / TAU).
    let base = in.position;
    let base_w =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(base, 1.0)).xyz;
    let up_w = normalize(mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index));

    let yaw = in.color.a * TAU;
    let frame = tree_frame(up_w, yaw);
    let tangent_w = frame[0];
    let bitangent_w = frame[1];

    // View direction (object→camera) in the tree frame. Caster pass (ortho
    // cascade camera): every "view ray" is parallel to the light — face the
    // SUN so the card casts the right canopy silhouette.
    let ortho_caster = is_ortho_projection(view.clip_from_view);
    var view_w = normalize(view.world_position - base_w);
    if (ortho_caster) {
        view_w = normalize(view.world_from_view[2].xyz);
    }
    let d_local = vec3<f32>(
        dot(view_w, tangent_w),
        dot(view_w, up_w),
        dot(view_w, bitangent_w),
    );

    // Card axes from the view direction (object-local) → world.
    let basis = view_basis_right_up(d_local);
    let right_l = basis[0];
    let up_l = basis[1];
    let right_w = right_l.x * tangent_w + right_l.y * up_w + right_l.z * bitangent_w;
    let cup_w = up_l.x * tangent_w + up_l.y * up_w + up_l.z * bitangent_w;

    // Per-species geometry → card size + centre.
    let sp = u32(in.uv_b.y + 0.5);
    let geo = imp.species_geo[sp];
    let scale = in.uv_b.x;
    var half = geo.x * scale;
    let center_w = base_w + up_w * (geo.y * scale);

    // Clipmap scale-fade (grow from zero across the ring's near/far edges) —
    // see the mesh tree shader for the anchor rationale. Caster: full scale.
    let ref_pos = view.world_position + tree.anchor.xyz;
    let inst_dist = distance(ref_pos, base_w);
    let near_edge = tree.time_fade.y;
    let far_edge = tree.time_fade.z;
    let band = max(tree.time_fade.w, 1.0);
    let fade_in = smoothstep(near_edge - band, near_edge + band, inst_dist);
    let fade_out = 1.0 - smoothstep(far_edge - band, far_edge + band, inst_dist);
    let grow = select(fade_in * fade_out, 1.0, ortho_caster);
    half = half * grow;

    let c = in.uv * 2.0 - 1.0;
    let world_pos = center_w + c.x * half * right_w + c.y * half * cup_w;

    var out: VertexOutput;
    out.world_position = vec4<f32>(world_pos, 1.0);
    out.position = position_world_to_clip(world_pos);
    out.world_normal = up_w;
#ifdef VERTEX_UVS_A
    out.uv = in.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vec2<f32>(hash1(base), in.uv_b.y);
#endif
#ifdef VERTEX_COLORS
    out.color = in.color;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = in.instance_index;
#endif
    return out;
}

// Hemisphere-octahedral encode (y up), inverse of the Rust `hemioct_decode`.
fn hemioct_encode(dir: vec3<f32>) -> vec2<f32> {
    var d = dir;
    d.y = max(d.y, 0.0);
    let l1 = abs(d.x) + abs(d.y) + abs(d.z);
    let p = d / max(l1, 1.0e-5);
    return vec2<f32>(p.x + p.z, p.x - p.z) * 0.5 + 0.5;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    let n_cells = imp.grid.x;
    let species_count = max(imp.grid.y, 1.0);
    let cutoff = imp.grid.z;
    let v_flip = imp.grid.w;
    let fill = imp.atlas.x;
    let sp = floor(in.uv_b.y + 0.5);
    let seed = in.uv_b.x;

    // Rebuild the tree frame and the card view direction (see header note on
    // the parallax approximation).
    let up_w = normalize(in.world_normal.xyz);
    let yaw = in.color.a * TAU;
    let frame = tree_frame(up_w, yaw);
    let tangent_w = frame[0];
    let bitangent_w = frame[1];
    var view_w = normalize(view.world_position - in.world_position.xyz);
    if (is_ortho_projection(view.clip_from_view)) {
        view_w = normalize(view.world_from_view[2].xyz);
    }
    let d_local = vec3<f32>(
        dot(view_w, tangent_w),
        dot(view_w, up_w),
        dot(view_w, bitangent_w),
    );

    let enc = hemioct_encode(d_local);
    let cf = enc * n_cells - 0.5;
    let i0 = floor(cf);
    let fr = cf - i0;

    // Card UV (billboard ±radius) → cell UV: the bounding sphere occupies
    // `fill` of the cell, centred; the rest is the anti-bleed gutter.
    var cell_uv = vec2<f32>(0.5) + (in.uv - vec2<f32>(0.5)) * fill;
    if v_flip > 0.5 {
        cell_uv.y = 1.0 - cell_uv.y;
    }

    let v_total = n_cells * species_count;
    var acc_alb = vec3<f32>(0.0);
    var acc_n = vec3<f32>(0.0);
    var acc_cov = 0.0;

    for (var dj = 0; dj < 2; dj = dj + 1) {
        for (var di = 0; di < 2; di = di + 1) {
            let ci = clamp(i0.x + f32(di), 0.0, n_cells - 1.0);
            let cj = clamp(i0.y + f32(dj), 0.0, n_cells - 1.0);
            let wx = select(1.0 - fr.x, fr.x, di == 1);
            let wy = select(1.0 - fr.y, fr.y, dj == 1);
            let w = wx * wy;

            let u = (ci + cell_uv.x) / n_cells;
            let vv = 1.0 - (sp * n_cells + cj + cell_uv.y) / v_total;
            let auv = vec2<f32>(u, vv);

            let a = textureSampleLevel(albedo_tex, albedo_smp, auv, 0.0);
            let nd = textureSampleLevel(normal_tex, normal_smp, auv, 0.0);
            // `a.rgb` is ALREADY premultiplied by coverage (atlas cleared to
            // transparent black) — accumulate straight; ÷ acc_cov below
            // un-premultiplies. Normals keep the coverage weight to down-weight
            // garbage edge normals.
            acc_alb += a.rgb * w;
            acc_n += (nd.rgb * 2.0 - 1.0) * a.a * w;
            acc_cov += a.a * w;
        }
    }

    if acc_cov < cutoff {
        discard;
    }

    let albedo = acc_alb / max(acc_cov, 1.0e-4);
    let n_local = normalize(acc_n / max(acc_cov, 1.0e-4));
    let n_world =
        normalize(n_local.x * tangent_w + n_local.y * up_w + n_local.z * bitangent_w);

    // Per-instance hue rides COLOR r/b (folded into the landcover tint
    // CPU-side from the ring-invariant Poisson cell), so a tree keeps ONE hue
    // across the ring cross-fade and the mesh↔impostor handoff. The baked
    // albedo is already the near-tree colour (the bake calls the same
    // `foliage_base_albedo`).
    let tint = albedo * in.color.rgb;

    pbr_input.material.base_color = vec4<f32>(tint, 1.0);
    pbr_input.material.perceptual_roughness = 0.95;
    pbr_input.material.reflectance = vec3<f32>(LEAF_REFLECTANCE);
    // The card is all canopy foliage — it transmits like the mesh leaves.
    pbr_input.material.diffuse_transmission = LEAF_DIFFUSE_TRANSMISSION;
    pbr_input.N = n_world;

    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    // Shared sun-shadow cascade × the cloud deck's sun transmittance — one
    // gate for the whole direct beam, exactly as the mesh trees compose it.
    // Impostor rings 0–1 CAST into cascades 1–2 (pinned to 3/6.5 km to cover
    // exactly this band), so the card must sample them too or it casts a
    // shadow it cannot receive — bright trees on dark ground, and a
    // shadowed→lit pop at the 1.2 km mesh↔impostor swap. Past cascade 2 the
    // sampler fades to lit and the W12 horizon term owns the far field.
    let hard_shadow = sun_shadow_factor(
        in.world_position.xyz,
        imp_shadow,
        imp_shadow_map_0,
        imp_shadow_map_1,
        imp_shadow_map_2,
        imp_shadow_map_3,
    );
    let cloud_t = cloud_sun_transmittance(
        cloud_shadow,
        cloud_shadow_tex,
        cloud_shadow_samp,
        in.world_position.xyz,
    );
    let shadow_f = hard_shadow * cloud_t;
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
