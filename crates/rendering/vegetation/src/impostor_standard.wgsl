// Standard-path hemisphere-octahedral vegetation impostor.
//
// Four degenerate mesh vertices identify one root. The vertex stage expands
// them into a view-facing card, while the fragment stage blends four nearby
// atlas views and re-lights the captured object-local normal.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    mesh_functions,
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
    forward_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}

struct ImpostorViewParams {
    // x/y = near/far distance, z = half fade band, w = fade enabled.
    fade: vec4<f32>,
    // xyz offsets the render camera to an adapter-defined stable anchor.
    anchor: vec4<f32>,
}

struct ImpostorParams {
    // x = cells, y = species, z = alpha cutoff, w = v flip.
    grid: vec4<f32>,
    // x = cell fill fraction.
    atlas: vec4<f32>,
    // x = bounding radius, y = centre height.
    species_geo: array<vec4<f32>, 4>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> foliage_view: ImpostorViewParams;
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

const LEAF_REFLECTANCE: f32 = 0.32;
const LEAF_DIFFUSE_TRANSMISSION: f32 = 0.35;
const TAU: f32 = 6.28318530717958647;

fn is_ortho_projection(clip_from_view: mat4x4<f32>) -> bool {
    return clip_from_view[3][3] != 0.0;
}

fn hash1(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

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
    let base_w = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(in.position, 1.0),
    ).xyz;
    let up_w = normalize(mesh_functions::mesh_normal_local_to_world(in.normal, in.instance_index));

    let yaw = in.color.a * TAU;
    let frame = tree_frame(up_w, yaw);
    let tangent_w = frame[0];
    let bitangent_w = frame[1];

    let ortho = is_ortho_projection(view.clip_from_view);
    var view_w = normalize(view.world_position - base_w);
    if ortho {
        view_w = normalize(view.world_from_view[2].xyz);
    }
    let d_local = vec3<f32>(
        dot(view_w, tangent_w),
        dot(view_w, up_w),
        dot(view_w, bitangent_w),
    );

    let basis = view_basis_right_up(d_local);
    let right_l = basis[0];
    let up_l = basis[1];
    let right_w = right_l.x * tangent_w + right_l.y * up_w + right_l.z * bitangent_w;
    let card_up_w = up_l.x * tangent_w + up_l.y * up_w + up_l.z * bitangent_w;

    let species = u32(in.uv_b.y + 0.5);
    let geo = imp.species_geo[species];
    let scale = in.uv_b.x;
    var half = geo.x * scale;
    let center_w = base_w + up_w * (geo.y * scale);

    if foliage_view.fade.w > 0.5 && !ortho {
        let reference = view.world_position + foliage_view.anchor.xyz;
        let distance_m = distance(reference, base_w);
        let band = max(foliage_view.fade.z, 1.0);
        let fade_in = smoothstep(
            foliage_view.fade.x - band,
            foliage_view.fade.x + band,
            distance_m,
        );
        let fade_out = 1.0 - smoothstep(
            foliage_view.fade.y - band,
            foliage_view.fade.y + band,
            distance_m,
        );
        half = half * fade_in * fade_out;
    }

    let corner = in.uv * 2.0 - 1.0;
    let world_pos = center_w + corner.x * half * right_w + corner.y * half * card_up_w;

    var out: VertexOutput;
    out.world_position = vec4<f32>(world_pos, 1.0);
    out.position = position_world_to_clip(world_pos);
    out.world_normal = up_w;
#ifdef VERTEX_UVS_A
    out.uv = in.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vec2<f32>(hash1(base_w), in.uv_b.y);
#endif
#ifdef VERTEX_COLORS
    out.color = in.color;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = in.instance_index;
#endif
    return out;
}

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
    let species = floor(in.uv_b.y + 0.5);

    let up_w = normalize(in.world_normal.xyz);
    let yaw = in.color.a * TAU;
    let frame = tree_frame(up_w, yaw);
    let tangent_w = frame[0];
    let bitangent_w = frame[1];
    var view_w = normalize(view.world_position - in.world_position.xyz);
    if is_ortho_projection(view.clip_from_view) {
        view_w = normalize(view.world_from_view[2].xyz);
    }
    let d_local = vec3<f32>(
        dot(view_w, tangent_w),
        dot(view_w, up_w),
        dot(view_w, bitangent_w),
    );

    let encoded = hemioct_encode(d_local);
    let cell_float = encoded * n_cells - 0.5;
    let cell_base = floor(cell_float);
    let blend = cell_float - cell_base;

    var cell_uv = vec2<f32>(0.5) + (in.uv - vec2<f32>(0.5)) * fill;
    if v_flip > 0.5 {
        cell_uv.y = 1.0 - cell_uv.y;
    }

    let vertical_cells = n_cells * species_count;
    var accumulated_albedo = vec3<f32>(0.0);
    var accumulated_normal = vec3<f32>(0.0);
    var accumulated_coverage = 0.0;

    for (var row = 0; row < 2; row = row + 1) {
        for (var column = 0; column < 2; column = column + 1) {
            let cell_x = clamp(cell_base.x + f32(column), 0.0, n_cells - 1.0);
            let cell_y = clamp(cell_base.y + f32(row), 0.0, n_cells - 1.0);
            let weight_x = select(1.0 - blend.x, blend.x, column == 1);
            let weight_y = select(1.0 - blend.y, blend.y, row == 1);
            let weight = weight_x * weight_y;

            let atlas_uv = vec2<f32>(
                (cell_x + cell_uv.x) / n_cells,
                1.0 - (species * n_cells + cell_y + cell_uv.y) / vertical_cells,
            );
            let albedo = textureSampleLevel(albedo_tex, albedo_smp, atlas_uv, 0.0);
            let normal = textureSampleLevel(normal_tex, normal_smp, atlas_uv, 0.0);
            accumulated_albedo += albedo.rgb * weight;
            accumulated_normal += (normal.rgb * 2.0 - 1.0) * albedo.a * weight;
            accumulated_coverage += albedo.a * weight;
        }
    }

    if accumulated_coverage < cutoff {
        discard;
    }

    let albedo = accumulated_albedo / max(accumulated_coverage, 1.0e-4);
    let normal_local = normalize(accumulated_normal / max(accumulated_coverage, 1.0e-4));
    let normal_world = normalize(
        normal_local.x * tangent_w + normal_local.y * up_w + normal_local.z * bitangent_w,
    );

    pbr_input.material.base_color = vec4<f32>(albedo * in.color.rgb, 1.0);
    pbr_input.material.perceptual_roughness = 0.95;
    pbr_input.material.reflectance = vec3<f32>(LEAF_REFLECTANCE);
    pbr_input.material.diffuse_transmission = LEAF_DIFFUSE_TRANSMISSION;
    pbr_input.N = normal_world;

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
