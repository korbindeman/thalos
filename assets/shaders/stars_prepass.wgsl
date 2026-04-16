//! Stars material prepass stub.
//!
//! The stars material is additive and transparent — it has no
//! depth, normal, or motion-vector contribution worth prepassing.
//! But Bevy still specialises a prepass pipeline for every material
//! in the pipeline cache, and the default prepass vertex shader
//! reads attributes our mesh doesn't provide. This file supplies
//! minimal entry points that match our vertex layout (POSITION +
//! UV_0 + COLOR) and emit nothing meaningful. Depth write is
//! already disabled by `StarsMaterial::specialize`, so this pipeline
//! ends up a pure no-op.

#import bevy_pbr::mesh_view_bindings::view

struct VertexInput {
    @location(0) direction: vec3<f32>,
    @location(1) corner:    vec2<f32>,
    @location(2) color:     vec4<f32>,
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
    // Force to far plane under reverse-Z.
    out.clip_position = vec4<f32>(center_clip.xy, 1.0e-7 * center_clip.w, center_clip.w);
    return out;
}

@fragment
fn fragment(_in: VertexOutput) {
    // No outputs.
}
