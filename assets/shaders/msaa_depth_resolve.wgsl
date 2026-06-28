// MSAA scene-depth resolve.
//
// Reads the multisampled main-pass depth attachment and writes sample 0 to a
// single-sample depth target, so the unified atmosphere pass can keep sampling
// scene depth as `texture_depth_2d` (see `rendering::scene_depth`) exactly as it
// does when MSAA is off. Without this, enabling MSAA makes the depth copy a
// no-op (mismatched sample counts) and the sky raymarch loses its terrain/hull
// clip.
//
// Depth-only pipeline: no color targets; the fragment writes `frag_depth` with
// `depth_compare: Always`, and the fullscreen triangle covers every pixel, so
// it is a 1:1 passthrough of sample 0.

@group(0) @binding(0) var scene_depth_ms: texture_depth_multisampled_2d;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle from 3 vertices: (-1,-1), (3,-1), (-1,3).
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @builtin(frag_depth) f32 {
    // `in.position.xy` is the framebuffer pixel coordinate; the multisampled
    // depth texture and the single-sample target share dimensions, so this is a
    // 1:1 read of sample 0 at the same texel.
    let coord = vec2<i32>(in.position.xy);
    return textureLoad(scene_depth_ms, coord, 0);
}
