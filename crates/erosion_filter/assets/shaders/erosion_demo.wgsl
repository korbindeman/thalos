// Demo fragment shader for bevy_erosion_filter::examples::plane_demo.
// Builds an fBm heightmap, applies the erosion filter, and colorizes the
// result with a simple altitude/water/snow ramp + cheap diffuse from the
// analytical gradient.
//
// Not part of the public library — production users should write their own
// fragment shader and `#import bevy_erosion_filter::erosion`.

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import bevy_erosion_filter::erosion::{
    apply_erosion,
    fbm,
    ErosionParams,
}

struct DemoParams {
    erosion: ErosionParams,
    // 1.0 = show eroded heightmap, 0.0 = show base (un-eroded) heightmap.
    show_erosion: f32,
    // Sea level in normalized [0, 1] heightmap range.
    water_level: f32,
    // fBm settings for the base heightmap.
    base_freq: f32,
    base_octaves: i32,
    base_lacunarity: f32,
    base_gain: f32,
    base_amplitude: f32,
    // Pad to 16-byte alignment for std140.
    _pad: vec3<f32>,
};

@group(2) @binding(0) var<uniform> params: DemoParams;

fn heightmap(uv: vec2<f32>) -> vec3<f32> {
    var n = fbm(uv, params.base_freq, params.base_octaves, params.base_lacunarity, params.base_gain)
        * params.base_amplitude;
    // Map fBm from [-1, 1] to roughly [0, 1] (only height; gradient passes through).
    n = n * 0.5 + vec3<f32>(0.5, 0.0, 0.0);

    // Suppress erosion below the waterline (matches the Shadertoy reference).
    var p = params.erosion;
    let mask = smoothstep(params.water_level - 0.1, params.water_level + 0.1, n.x);
    p.strength = p.strength * mask * params.show_erosion;

    return apply_erosion(uv, n, p);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let h_and_s = heightmap(in.uv);
    let h = h_and_s.x;
    let grad = h_and_s.yz;

    // Cheap shading: pretend the surface is z = h(x, y), light from upper-left.
    let normal = normalize(vec3<f32>(-grad.x, -grad.y, 1.0));
    let light = normalize(vec3<f32>(-0.6, -0.4, 0.7));
    let diffuse = max(0.1, dot(normal, light));

    // Altitude colormap: water → sand → grass → rock → snow.
    var col: vec3<f32>;
    if (h < params.water_level) {
        let depth = clamp((params.water_level - h) * 4.0, 0.0, 1.0);
        col = mix(vec3<f32>(0.10, 0.35, 0.45), vec3<f32>(0.02, 0.07, 0.18), depth);
    } else {
        let alt = (h - params.water_level) / (1.0 - params.water_level);
        let sand   = vec3<f32>(0.78, 0.70, 0.55);
        let grass  = vec3<f32>(0.35, 0.42, 0.20);
        let rock   = vec3<f32>(0.42, 0.38, 0.34);
        let snow   = vec3<f32>(0.95, 0.96, 0.98);
        if (alt < 0.05) {
            col = mix(sand, grass, smoothstep(0.0, 0.05, alt));
        } else if (alt < 0.45) {
            col = mix(grass, rock, smoothstep(0.05, 0.45, alt));
        } else {
            col = mix(rock, snow, smoothstep(0.45, 0.75, alt));
        }
    }

    return vec4<f32>(col * diffuse, 1.0);
}
