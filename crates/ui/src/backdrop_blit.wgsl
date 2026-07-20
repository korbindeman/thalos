// Fullscreen blit of the post-processed scene colour into the half-res
// UI-backdrop image (see `glass.rs`). A 4-tap box gives the downsample a
// touch of prefiltering before the glass shader's spiral blur.

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex(@builtin(vertex_index) vi: u32) -> FullscreenVertexOutput {
    var out: FullscreenVertexOutput;
    let uv = vec2<f32>(f32(vi >> 1u), f32(vi & 1u)) * 2.0;
    out.uv = uv;
    out.position = vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
    return out;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texel = vec2(1.0) / vec2<f32>(textureDimensions(src_texture));
    var c = textureSampleLevel(src_texture, src_sampler, in.uv + vec2(-0.5, -0.5) * texel, 0.0).rgb;
    c += textureSampleLevel(src_texture, src_sampler, in.uv + vec2(0.5, -0.5) * texel, 0.0).rgb;
    c += textureSampleLevel(src_texture, src_sampler, in.uv + vec2(-0.5, 0.5) * texel, 0.0).rgb;
    c += textureSampleLevel(src_texture, src_sampler, in.uv + vec2(0.5, 0.5) * texel, 0.0).rgb;
    return vec4(c * 0.25, 1.0);
}
