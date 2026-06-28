// Tree impostor BAKE shader.
//
// Renders a species' LOD0 mesh from one hemisphere view direction (set by the
// per-instance rotation so the bake camera's −Z view sees that direction) into
// one atlas cell. Two modes (BakeParams.mode.x):
//   0 → albedo + coverage  (leaf colour × tint; alpha = 1 on covered leaf texels)
//   1 → object-local normal + depth  (rgb = n*0.5+0.5, a = cell-space depth)
//
// Leaf SHAPE and COLOUR come from the SAME procedural foliage atlas the mesh
// trees sample (`tree.wgsl`): the mesh stores a near-white vertex tint and the
// real green lives in the atlas, so the bake MUST sample it — otherwise the
// captured impostor is a solid pale quad. Leaf cards are alpha-tested
// (`tex.a < 0.5 → discard`) so the silhouette is leaf-shaped, not blocky, and
// both modes discard identically so the normal/depth footprint matches the
// albedo coverage.
//
// The normal is passed through in OBJECT-LOCAL space (not transformed to world)
// so the runtime impostor (`tree_impostor.wgsl`) can re-light each tree in its
// own terrain frame. The cleared (transparent) background gives coverage 0.

#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
}

struct BakeParams {
    // x = mode (0 albedo, 1 normal), y = depth scale, z/w unused.
    mode: vec4<f32>,
}

@group(3) @binding(0) var<uniform> bake: BakeParams;
@group(3) @binding(1) var atlas_tex: texture_2d<f32>;
@group(3) @binding(2) var atlas_samp: sampler;

// Foliage atlas: 4×4 cells, 128 px each (512²). Mirror of `tree.wgsl`.
const ATLAS_N: f32 = 4.0;
const ATLAS_TEXEL: f32 = 1.0 / 1024.0;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) uv1: vec2<f32>, // y = atlas leaf code (cell·4 + corner)
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_z: f32,
    @location(3) atlas_uv: vec2<f32>,
}

// Decode `cell·4 + corner` into the atlas UV, half-texel inset (matches
// `tree.wgsl::atlas_uv_of` exactly).
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
    let world_pos =
        mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(in.position, 1.0))
            .xyz;

    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world_pos);
    // Object-local normal, intentionally NOT world-transformed.
    out.local_normal = in.normal;
    out.color = in.color;
    // Bake camera looks down −Z from +Z, so world +z is "toward camera"; the
    // recentred + scaled tree spans roughly ±cell-fit in z about 0.
    out.view_z = world_pos.z;
    out.atlas_uv = atlas_uv_of(in.uv1.y);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Leaf shape + coverage from the procedural atlas; alpha-test the leaf cards
    // so the captured silhouette is leaf-shaped (same gate as the mesh trees).
    let tex = textureSample(atlas_tex, atlas_samp, in.atlas_uv);
    if tex.a < 0.5 {
        discard;
    }

    if bake.mode.x < 0.5 {
        // Albedo + coverage. Leaf colour × per-vertex tint (× AO); both linear.
        return vec4<f32>(tex.rgb * in.color.rgb, 1.0);
    }
    let n = normalize(in.local_normal) * 0.5 + vec3<f32>(0.5);
    let depth = clamp(in.view_z * bake.mode.y + 0.5, 0.0, 1.0);
    return vec4<f32>(n, depth);
}
