// Frosted-glass UI panel material (see `glass.rs`).
//
// Rounded-rect SDF shape from the node's border radius, a 16-tap jittered
// spiral blur over the half-res scene copy (true frost), a dark tint, a top
// sheen, fine frost grain, and a hairline edge stroke that catches light
// along the top edge. Falls back to a plain translucent tint when no
// backdrop is bound (`params.w = 0`).

#import bevy_render::view::View
#import bevy_ui::ui_vertex_output::UiVertexOutput

@group(0) @binding(0) var<uniform> view: View;

struct GlassUniform {
    tint: vec4<f32>,
    stroke: vec4<f32>,
    // x: blur radius px, y: grain, z: sheen, w: backdrop enable
    params: vec4<f32>,
};

@group(1) @binding(0) var<uniform> material: GlassUniform;
@group(1) @binding(1) var backdrop_texture: texture_2d<f32>;
@group(1) @binding(2) var backdrop_sampler: sampler;

// Rounded-box SDF (iq). `r` is per-corner radius ordered
// (top-left, top-right, bottom-right, bottom-left) — matching
// `UiVertexOutput::border_radius`. `p` is centred; +y is down (UV space).
fn sd_rounded_box(p: vec2<f32>, half_size: vec2<f32>, r: vec4<f32>) -> f32 {
    var rad = r.x; // top-left: x <= 0, y <= 0
    if p.x > 0.0 && p.y <= 0.0 { rad = r.y; }
    if p.x > 0.0 && p.y > 0.0 { rad = r.z; }
    if p.x <= 0.0 && p.y > 0.0 { rad = r.w; }
    let q = abs(p) - half_size + vec2(rad);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - rad;
}

// Interleaved gradient noise — cheap per-pixel jitter.
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

const TAU: f32 = 6.28318530718;
const GOLDEN_ANGLE: f32 = 2.39996323;

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    let half_size = in.size * 0.5;
    let p = in.uv * in.size - half_size;
    let d = sd_rounded_box(p, half_size, in.border_radius);
    // 1px anti-aliased shape edge.
    let shape = 1.0 - smoothstep(-1.0, 0.0, d);
    if shape <= 0.0 {
        discard;
    }

    var base: vec3<f32>;
    var alpha: f32;
    if material.params.w > 0.5 {
        // Frost: jittered golden-angle spiral over the half-res scene copy.
        let screen_uv = in.position.xy / view.viewport.zw;
        let radius_uv = material.params.x / view.viewport.zw;
        let rot = ign(in.position.xy) * TAU;
        var acc = vec3(0.0);
        for (var i = 0; i < 24; i = i + 1) {
            let t = sqrt((f32(i) + 0.5) / 24.0);
            let ang = f32(i) * GOLDEN_ANGLE + rot;
            let off = vec2(cos(ang), sin(ang)) * t * radius_uv;
            acc += textureSampleLevel(backdrop_texture, backdrop_sampler, screen_uv + off, 0.0).rgb;
        }
        var blurred = acc / 24.0;
        // Vibrancy: pull the blur toward its own luminance so the sheet reads
        // milky and calm rather than colour-cast, then tint for text contrast.
        let lum = dot(blurred, vec3(0.2126, 0.7152, 0.0722));
        blurred = mix(blurred, vec3(lum), 0.30);
        base = mix(blurred, material.tint.rgb, material.tint.a);
        // The glass replaces the scene with its own blurred copy, so it is
        // effectively opaque.
        alpha = 1.0;
    } else {
        base = material.tint.rgb;
        alpha = min(1.0, material.tint.a + 0.30);
    }

    // Top sheen: glass catches light along its upper region.
    base += vec3(material.params.z * (1.0 - in.uv.y) * 0.014);

    // Frost grain.
    base += vec3((ign(in.position.xy + vec2(17.0, 59.0)) - 0.5) * material.params.y);

    // Whisper of a rim ~1px inside the edge, slightly livelier along the top.
    let edge = 1.0 - smoothstep(0.4, 1.8, abs(d + 1.0));
    let top_boost = 1.0 + 0.7 * (1.0 - smoothstep(0.0, 0.25, in.uv.y));
    let stroke_a = clamp(material.stroke.a * edge * top_boost, 0.0, 1.0);
    base = mix(base, material.stroke.rgb, stroke_a);
    alpha = max(alpha, stroke_a);

    return vec4(base, alpha * shape);
}
