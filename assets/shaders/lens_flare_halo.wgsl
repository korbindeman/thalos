#import bevy_ui::ui_vertex_output::UiVertexOutput

@group(1) @binding(0) var<uniform> tint: vec4<f32>;
// params: xy = axis direction (unit, UV space, sun→center),
//         z  = unused,
//         w  = max ring thickness
@group(1) @binding(1) var<uniform> params: vec4<f32>;

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - vec3<f32>(3.0));
    return v * mix(vec3<f32>(1.0), clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    let p = in.uv * 2.0 - 1.0;
    let r = length(p);
    if (r > 1.0) {
        return vec4<f32>(0.0);
    }

    let axis = params.xy;
    let thickness = max(params.w, 0.005);

    let safe_r = max(r, 1e-4);
    let ndir = p / safe_r;
    let alignment = dot(ndir, axis);

    // Thickness tapers from 0 on the sun-facing tip to full at the apex.
    let t_weight = smoothstep(-0.35, 0.7, alignment);
    let local_thickness = thickness * t_weight;
    if (local_thickness <= 0.001) {
        return vec4<f32>(0.0);
    }

    let outer = 0.84;
    let inner = outer - local_thickness;
    let mid = (inner + outer) * 0.5;
    let half_width = local_thickness * 0.5;

    // Gaussian falloff across the ring width — soft blurred band, no
    // hard edges. `sigma` controls how tight the intensity peak is.
    let d = (r - mid) / half_width;
    let sigma = 0.55;
    let intensity = exp(-d * d / (2.0 * sigma * sigma));

    // Rainbow hue ramp across the ring width: violet on the inner edge,
    // red on the outer edge — the full visible spectrum, prism-style.
    let band_t = clamp((r - inner) / local_thickness, 0.0, 1.0);
    let hue = mix(0.75, 0.0, band_t);
    let rainbow = hsv_to_rgb(hue, 0.85, 1.0);

    let rgb = rainbow * intensity;
    return vec4<f32>(rgb * tint.rgb * tint.a, 0.0);
}
