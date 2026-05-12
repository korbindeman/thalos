// Navball sphere material.
//
// Samples the baked equirectangular navball texture on a unit sphere and
// applies limb darkening so the surface reads as a 3D ball. No physical
// lighting — the navball is an instrument display, not a lit object.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}
#import bevy_pbr::mesh_view_bindings::view

struct NavballParams {
    // Higher gamma → darkening pinches harder toward the limb.
    limb_darkening_gamma: f32,
    // Brightness floor at the silhouette edge (cos θ = 0).
    limb_floor: f32,
    // Strength of the lens-style UV bend near the edges. 0 = none.
    edge_distortion: f32,
    _pad: f32,
}

@group(3) @binding(0) var<uniform> params: NavballParams;
@group(3) @binding(1) var nav_texture: texture_2d<f32>;
@group(3) @binding(2) var nav_sampler: sampler;

struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv:           vec2<f32>,
}

@vertex
fn vertex(in: VertexInput) -> VertexOutput {
    let world_from_local = get_world_from_local(in.instance_index);
    var out: VertexOutput;
    out.clip_position = mesh_position_local_to_clip(world_from_local, vec4(in.position, 1.0));
    out.world_normal = normalize((world_from_local * vec4(in.normal, 0.0)).xyz);
    out.uv = in.uv;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // View-space normal: +Z faces the camera in Bevy's view-space.
    let n_view = (view.view_from_world * vec4<f32>(in.world_normal, 0.0)).xyz;
    let cos_theta = max(0.0, n_view.z);
    let limb = mix(params.limb_floor, 1.0, pow(cos_theta, params.limb_darkening_gamma));

    var uv = in.uv;
    if (params.edge_distortion > 0.0) {
        let bend = (1.0 - cos_theta) * params.edge_distortion * 0.04;
        uv = uv + vec2<f32>(n_view.x, -n_view.y) * bend;
    }

    let base = textureSample(nav_texture, nav_sampler, uv).rgb;
    return vec4<f32>(base * limb, 1.0);
}
