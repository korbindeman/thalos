// Tree impostor BAKE shader.
//
// Renders a species' LOD0 mesh from one hemisphere view direction (set by the
// per-instance rotation so the bake camera's −Z view sees that direction) into
// one atlas cell. Two modes (BakeParams.mode.x):
//   0 → albedo + coverage  (vertex colour; alpha = 1 wherever geometry is)
//   1 → object-local normal + depth  (rgb = n*0.5+0.5, a = cell-space depth)
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

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(5) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_z: f32,
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
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if bake.mode.x < 0.5 {
        // Albedo + coverage. Vertex colour is already linear.
        return vec4<f32>(in.color.rgb, 1.0);
    }
    let n = normalize(in.local_normal) * 0.5 + vec3<f32>(0.5);
    let depth = clamp(in.view_z * bake.mode.y + 0.5, 0.0, 1.0);
    return vec4<f32>(n, depth);
}
