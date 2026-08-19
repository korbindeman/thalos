// Near-volume cloud composite for Kòrsou. The compute marcher is atmosphere-
// agnostic; this pass only overlays that layer against copied scene depth.
#import bevy_pbr::mesh_view_bindings::view

struct Vertex {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@group(3) @binding(0) var scene_depth_texture: texture_depth_2d;
@group(3) @binding(1) var cloud_layer_texture: texture_2d<f32>;
@group(3) @binding(2) var cloud_distance_texture: texture_2d<f32>;

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4(in.position.xy, 1.0, 1.0);
    return out;
}

fn scene_distance(pixel: vec2<f32>) -> f32 {
    let depth = textureLoad(scene_depth_texture, vec2<i32>(pixel), 0);
    if depth <= 0.0 {
        return 1.0e30;
    }
    let ndc_x = (pixel.x / view.viewport.z) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y / view.viewport.w) * 2.0;
    let view_h = view.view_from_clip * vec4<f32>(ndc_x, ndc_y, depth, 1.0);
    return length(view_h.xyz / view_h.w);
}

fn sample_near_cloud(pixel: vec2<f32>) -> vec4<f32> {
    let dims = textureDimensions(cloud_layer_texture);
    let cloud_res = vec2<f32>(dims);
    let uv = pixel / view.viewport.zw;
    let p = uv * cloud_res - 0.5;
    let base = floor(p);
    let f = p - base;
    let bilinear_max = vec2<i32>(i32(dims.x) - 2, i32(dims.y) - 4);
    let cb = clamp(vec2<i32>(base), vec2<i32>(0), bilinear_max);
    let cs00 = textureLoad(cloud_layer_texture, cb, 0);
    let cs10 = textureLoad(cloud_layer_texture, cb + vec2<i32>(1, 0), 0);
    let cs01 = textureLoad(cloud_layer_texture, cb + vec2<i32>(0, 1), 0);
    let cs11 = textureLoad(cloud_layer_texture, cb + vec2<i32>(1, 1), 0);
    let cd00 = textureLoad(cloud_distance_texture, cb, 0).r;
    let cd10 = textureLoad(cloud_distance_texture, cb + vec2<i32>(1, 0), 0).r;
    let cd01 = textureLoad(cloud_distance_texture, cb + vec2<i32>(0, 1), 0).r;
    let cd11 = textureLoad(cloud_distance_texture, cb + vec2<i32>(1, 1), 0).r;
    let ref_coord = cb + vec2<i32>(select(0, 1, f.x >= 0.5), select(0, 1, f.y >= 0.5));
    let cloud_near = textureLoad(cloud_distance_texture, ref_coord, 0).r;
    let depth_scale = max(cloud_near, 2000.0);
    let hit_ref = cloud_near < 1.0e8;
    let dw00 = select(0.0, select(1.0, exp(-abs(cd00 - cloud_near) / depth_scale), hit_ref && cd00 < 1.0e8), hit_ref == (cd00 < 1.0e8));
    let dw10 = select(0.0, select(1.0, exp(-abs(cd10 - cloud_near) / depth_scale), hit_ref && cd10 < 1.0e8), hit_ref == (cd10 < 1.0e8));
    let dw01 = select(0.0, select(1.0, exp(-abs(cd01 - cloud_near) / depth_scale), hit_ref && cd01 < 1.0e8), hit_ref == (cd01 < 1.0e8));
    let dw11 = select(0.0, select(1.0, exp(-abs(cd11 - cloud_near) / depth_scale), hit_ref && cd11 < 1.0e8), hit_ref == (cd11 < 1.0e8));
    let w00 = (1.0 - f.x) * (1.0 - f.y) * dw00;
    let w10 = f.x * (1.0 - f.y) * dw10;
    let w01 = (1.0 - f.x) * f.y * dw01;
    let w11 = f.x * f.y * dw11;
    let weight = max(w00 + w10 + w01 + w11, 1.0e-5);
    return (cs00 * w00 + cs10 * w10 + cs01 * w01 + cs11 * w11) / weight;
}

fn near_visibility(cloud_near: f32, slab_far: f32, transmittance: f32, scene_t: f32) -> f32 {
    if scene_t >= 1.0e29 {
        return 1.0;
    }
    if cloud_near >= scene_t {
        return 0.0;
    }
    let extent = max(slab_far - cloud_near, 0.0);
    let frac = select(
        clamp((scene_t - cloud_near) / extent, 0.0, 1.0),
        select(0.0, 1.0, scene_t >= cloud_near),
        extent <= 1.0,
    );
    let tau = -log(clamp(transmittance, 1.0e-4, 1.0));
    let opacity_full = 1.0 - exp(-tau);
    if opacity_full <= 1.0e-5 {
        return frac;
    }
    return clamp((1.0 - exp(-tau * frac)) / opacity_full, 0.0, 1.0);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene_t = scene_distance(in.clip_position.xy);
    let near_sample = sample_near_cloud(in.clip_position.xy);
    let dims = textureDimensions(cloud_layer_texture);
    let uv = in.clip_position.xy / view.viewport.zw;
    let ref_coord = clamp(
        vec2<i32>(uv * vec2<f32>(dims)),
        vec2<i32>(0),
        vec2<i32>(dims) - vec2<i32>(1),
    );
    let cloud_span = textureLoad(cloud_distance_texture, ref_coord, 0).rg;
    let vis = near_visibility(cloud_span.x, cloud_span.y, near_sample.a, scene_t);
    let opacity = (1.0 - near_sample.a) * vis;
    if opacity <= 1.0e-5 {
        discard;
    }
    return vec4(near_sample.rgb * vis, clamp(opacity, 0.0, 1.0));
}
