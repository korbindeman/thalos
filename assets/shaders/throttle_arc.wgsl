#import bevy_ui::ui_vertex_output::UiVertexOutput

@group(1) @binding(0) var<uniform> geometry: vec4<f32>;
@group(1) @binding(1) var<uniform> levels: vec4<f32>;
@group(1) @binding(2) var<uniform> track_color: vec4<f32>;
@group(1) @binding(3) var<uniform> fill_color: vec4<f32>;
@group(1) @binding(4) var<uniform> warn_color: vec4<f32>;
@group(1) @binding(5) var<uniform> tick_color: vec4<f32>;
@group(1) @binding(6) var<uniform> tick_major_color: vec4<f32>;
@group(1) @binding(7) var<uniform> border_color: vec4<f32>;

fn band_mask(value: f32, min_v: f32, max_v: f32, aa: f32) -> f32 {
    let lower = smoothstep(min_v - aa, min_v + aa, value);
    let upper = 1.0 - smoothstep(max_v - aa, max_v + aa, value);
    return lower * upper;
}

fn overlay(base: vec4<f32>, top: vec4<f32>, amount: f32) -> vec4<f32> {
    let a = clamp(amount * top.a, 0.0, 1.0);
    return vec4<f32>(mix(base.rgb, top.rgb, a), max(base.a, a));
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    let p = in.uv * in.size;
    let center = geometry.xy;
    let inner_radius = geometry.z;
    let outer_radius = geometry.w;
    let commanded = clamp(levels.x, 0.0, 1.0);
    let effective = clamp(levels.y, 0.0, 1.0);
    let half_angle = levels.z;
    let border_width = levels.w;

    let d = p - center;
    let r = length(d);
    if (r <= 0.0) {
        return vec4<f32>(0.0);
    }

    // Angle around the navball centre: 0 points straight left. The end
    // caps are constant polar angles, so they are straight radial cuts.
    let phi = atan2(d.y, -d.x);
    let arc_distance = min(
        min(outer_radius - r, r - inner_radius),
        (half_angle - abs(phi)) * r,
    );
    let aa = max(fwidth(arc_distance), 0.75);
    let shape_alpha = smoothstep(0.0, aa, arc_distance);
    if (shape_alpha <= 0.0) {
        return vec4<f32>(0.0);
    }

    let level = 1.0 - (phi + half_angle) / (half_angle * 2.0);
    var color = track_color;
    if (level <= effective) {
        color = fill_color;
    } else if (level <= commanded) {
        color = warn_color;
    }

    let across_band = clamp((r - inner_radius) / (outer_radius - inner_radius), 0.0, 1.0);
    var tick_alpha = 0.0;
    var major_tick_alpha = 0.0;

    for (var i: u32 = 0u; i <= 10u; i = i + 1u) {
        let tick_level = f32(i) / 10.0;
        let tick_phi = half_angle - tick_level * half_angle * 2.0;
        let major = (i % 5u) == 0u;
        let tick_half_width = select(0.75, 1.25, major);
        let band_start = select(0.36, 0.18, major);
        let band_end = select(0.78, 0.88, major);
        let tick_dist = abs(phi - tick_phi) * r;
        let line = 1.0 - smoothstep(tick_half_width - aa, tick_half_width + aa, tick_dist);
        let span = band_mask(across_band, band_start, band_end, aa / (outer_radius - inner_radius));
        let amount = line * span;
        if (major) {
            major_tick_alpha = max(major_tick_alpha, amount);
        } else {
            tick_alpha = max(tick_alpha, amount);
        }
    }

    color = overlay(color, tick_color, tick_alpha * shape_alpha);
    color = overlay(color, tick_major_color, major_tick_alpha * shape_alpha);

    let border_alpha =
        (1.0 - smoothstep(border_width - aa, border_width + aa, arc_distance)) * shape_alpha;
    color = overlay(color, border_color, border_alpha);

    return vec4<f32>(color.rgb, color.a * shape_alpha);
}
