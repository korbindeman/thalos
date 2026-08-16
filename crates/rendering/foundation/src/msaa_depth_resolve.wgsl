// Resolve sample 0 of the selected view's multisampled depth attachment into a
// single-sample depth target. The fullscreen triangle covers every pixel.

@group(0) @binding(0) var scene_depth_ms: texture_depth_multisampled_2d;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @builtin(frag_depth) f32 {
    return textureLoad(scene_depth_ms, vec2<i32>(in.position.xy), 0);
}
