// Animated film grain applied late in the post stack.
//
// - Luma-weighted so grain stays strongest in shadows/void (where banding
//   lives) and fades out in highlights.
// - Hash-based blue-noise approximation, reseeded each frame by `time` so
//   the pattern doesn't look like dirt stuck on the lens.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct FilmGrainSettings {
    intensity: f32,
    time: f32,
    shadow_bias: f32,
    _pad: f32,
}

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> settings: FilmGrainSettings;

// IQ-style hash (https://www.shadertoy.com/view/4djSRW). Cheap, tileable,
// good enough for grain at this intensity.
fn hash13(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

// Two hashes combined per channel, biased to zero. Each channel is seeded
// independently so the grain carries per-pixel chroma jitter — reads like
// digital camera shadow noise where grain is strong, stays monochrome-ish
// where it's weak.
fn grain_rgb(uv: vec2<f32>, t: f32) -> vec3<f32> {
    let p = uv * 1024.0;
    let r = hash13(vec3<f32>(p, t)) + hash13(vec3<f32>(p + 17.0, t + 3.7)) - 1.0;
    let g = hash13(vec3<f32>(p + 41.0, t + 7.3)) + hash13(vec3<f32>(p + 53.0, t + 11.1)) - 1.0;
    let b = hash13(vec3<f32>(p + 71.0, t + 13.9)) + hash13(vec3<f32>(p + 89.0, t + 17.5)) - 1.0;
    return vec3<f32>(r, g, b);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(source_texture, source_sampler, in.uv);

    // Rec. 709 luma. We only need a rough shadow weight, not perceptual
    // accuracy.
    let luma = clamp(dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);

    // Weight: max at luma=0, drops toward highlights. shadow_bias pins a
    // floor of grain in mid/high tones so the effect still reads there.
    let shadow_weight = mix(settings.shadow_bias, 1.0, 1.0 - luma);

    // Gate grain against true-black. Additive grain on a zero pixel can
    // only push upward (can't go negative and be clamped), which lifts
    // pure-black voids into faint gray. Fading the grain in as soon as
    // there's any signal keeps the space background inky while the
    // terminator and dim surfaces still receive full shadow grain.
    let ink_gate = smoothstep(0.0, 0.004, luma);

    // Quantize time so the pattern reseeds at a fixed perceptual rate
    // regardless of framerate.
    let t_quantized = floor(settings.time * 24.0);

    let g = grain_rgb(in.uv, t_quantized) * settings.intensity * shadow_weight * ink_gate;

    return vec4<f32>(color.rgb + g, color.a);
}
