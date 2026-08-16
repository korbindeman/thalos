// Capture one woody species into a hemisphere-octahedral impostor atlas.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
}
#import thalos::foliage::foliage_base_albedo

struct BakeParams {
    // x = output mode (0 albedo, 1 object-local normal), y = depth scale.
    mode: vec4<f32>,
}

@group(3) @binding(0) var<uniform> bake: BakeParams;
@group(3) @binding(1) var atlas_tex: texture_2d<f32>;
@group(3) @binding(2) var atlas_samp: sampler;

const ATLAS_N: f32 = 4.0;
const ATLAS_TEXEL: f32 = 1.0 / 1024.0;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) uv1: vec2<f32>,
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_z: f32,
    @location(3) atlas_uv: vec2<f32>,
    @location(4) leaf: f32,
}

fn atlas_uv_of(code: f32) -> vec2<f32> {
    let cell = floor(code / 4.0);
    let corner = code - cell * 4.0;
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
    let world_pos =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0))
            .xyz;

    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world_pos);
    // Object-local on purpose: the runtime reconstructs each terrain frame.
    out.local_normal = in.normal;
    out.color = in.color;
    out.view_z = world_pos.z;
    out.atlas_uv = atlas_uv_of(in.uv1.y);
    out.leaf = select(0.0, 1.0, floor(in.uv1.y / 4.0) < 11.5);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex = textureSample(atlas_tex, atlas_samp, in.atlas_uv);
    if tex.a < 0.5 {
        discard;
    }
    let atlas_rgb = tex.rgb / max(tex.a, 1.0e-3);

    if bake.mode.x < 0.5 {
        return vec4<f32>(
            foliage_base_albedo(atlas_rgb, in.color.g, in.leaf, 0.5),
            1.0,
        );
    }
    let n = normalize(in.local_normal) * 0.5 + vec3<f32>(0.5);
    let depth = clamp(in.view_z * bake.mode.y + 0.5, 0.0, 1.0);
    return vec4<f32>(n, depth);
}
