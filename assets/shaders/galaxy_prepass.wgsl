//! Galaxy material prepass stub. Mirrors `stars_prepass.wgsl` — the
//! galaxy material is additive/transparent and has no meaningful
//! prepass contribution, but Bevy still specialises a prepass
//! pipeline for it. These entry points satisfy the IO contract with
//! our custom vertex layout and emit nothing.

#import bevy_pbr::mesh_view_bindings::view

struct VertexInput {
    @location(0) direction: vec3<f32>,
    @location(1) corner:    vec2<f32>,
    @location(2) shape:     vec3<f32>,
    @location(3) orient:    vec4<f32>,
    @location(4) color:     vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

const INFINITY_DISTANCE: f32 = 1.0e10;

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_pos = view.world_position + in.direction * INFINITY_DISTANCE;
    let center_clip = view.clip_from_world * vec4<f32>(world_pos, 1.0);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(center_clip.xy, 0.0, center_clip.w);
    return out;
}

@fragment
fn fragment(_in: VertexOutput) {
}
