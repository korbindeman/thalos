// F3 debug-view performance graph (see game/src/perf/overlay.rs).
//
// One quad renders a whole sample ring from uniform arrays:
//   mode 0 — CPU frame-time bars (budget-coloured) + GPU-time line, with
//            horizontal budget marks (16.7 / 33.3 ms);
//   mode 1 — two memory curves (tile-resident / mesh-slab MiB), autoscaled.
//
// params = (sample count, full-scale value, mode, unused)
// marks  = (mark value 1, mark value 2, 0, 0); 0 disables a mark.
// Samples are packed 4 per vec4, chronological, newest at the right edge.

#import bevy_ui::ui_vertex_output::UiVertexOutput

const SERIES_VEC4S: u32 = 128u;

@group(1) @binding(0) var<uniform> params: vec4<f32>;
@group(1) @binding(1) var<uniform> marks: vec4<f32>;
@group(1) @binding(2) var<uniform> series_a: array<vec4<f32>, SERIES_VEC4S>;
@group(1) @binding(3) var<uniform> series_b: array<vec4<f32>, SERIES_VEC4S>;

fn sample_a(i: u32) -> f32 {
    return series_a[i / 4u][i % 4u];
}

fn sample_b(i: u32) -> f32 {
    return series_b[i / 4u][i % 4u];
}

// Colour a frame-time bar by budget: calm below 60 fps cost, amber past
// 16.7 ms, red past 33.3 ms.
fn bar_color(ms: f32) -> vec3<f32> {
    let calm = vec3<f32>(0.35, 0.78, 0.42);
    let warn = vec3<f32>(0.92, 0.74, 0.25);
    let over = vec3<f32>(0.94, 0.33, 0.28);
    if (ms > 33.34) {
        return over;
    }
    if (ms > 16.68) {
        return warn;
    }
    return calm;
}

// Distance-to-curve mask: is `y` within `half_th` of the segment between the
// current and next sample heights (both normalized 0..1)?
fn line_mask(y: f32, h0: f32, h1: f32, half_th: f32) -> f32 {
    let lo = min(h0, h1) - half_th;
    let hi = max(h0, h1) + half_th;
    if (y >= lo && y <= hi) {
        return 1.0;
    }
    return 0.0;
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    let count = max(params.x, 2.0);
    let full_scale = max(params.y, 1e-3);
    let mode = params.z;

    // y measured up from the bottom edge, 0..1.
    let y = 1.0 - in.uv.y;
    let px_y = 1.0 / max(in.size.y, 1.0);

    let idx_f = in.uv.x * (count - 1.0);
    let idx = u32(clamp(idx_f, 0.0, count - 1.0));
    let idx_next = min(idx + 1u, u32(count - 1.0));

    var color = vec3<f32>(0.0, 0.0, 0.0);
    var alpha = 0.42; // graph backdrop

    if (mode < 0.5) {
        // ── mode 0: frame-time bars + GPU line ──────────────────────────
        let cpu_ms = sample_a(idx);
        let cpu_h = clamp(cpu_ms / full_scale, 0.0, 1.0);
        if (y <= cpu_h) {
            color = bar_color(cpu_ms);
            alpha = 0.92;
        }

        // Budget marks: thin dashed horizontals.
        let dash = step(0.5, fract(in.uv.x * 24.0));
        for (var m = 0u; m < 2u; m = m + 1u) {
            let mark = marks[m];
            if (mark > 0.0) {
                let mh = mark / full_scale;
                if (abs(y - mh) < px_y && dash > 0.5) {
                    color = mix(color, vec3<f32>(0.85, 0.85, 0.85), 0.65);
                    alpha = max(alpha, 0.65);
                }
            }
        }

        // GPU line on top.
        let g0 = clamp(sample_b(idx) / full_scale, 0.0, 1.0);
        let g1 = clamp(sample_b(idx_next) / full_scale, 0.0, 1.0);
        if (line_mask(y, g0, g1, px_y) > 0.5) {
            color = vec3<f32>(0.36, 0.78, 0.94);
            alpha = 0.95;
        }
    } else {
        // ── mode 1: two memory curves ───────────────────────────────────
        let a0 = clamp(sample_a(idx) / full_scale, 0.0, 1.0);
        let a1 = clamp(sample_a(idx_next) / full_scale, 0.0, 1.0);
        let b0 = clamp(sample_b(idx) / full_scale, 0.0, 1.0);
        let b1 = clamp(sample_b(idx_next) / full_scale, 0.0, 1.0);

        // Soft fill under each curve, then the curve itself.
        if (y <= a0) {
            color = mix(color, vec3<f32>(0.35, 0.78, 0.42), 0.22);
            alpha = max(alpha, 0.5);
        }
        if (y <= b0) {
            color = mix(color, vec3<f32>(0.90, 0.58, 0.25), 0.22);
            alpha = max(alpha, 0.5);
        }
        if (line_mask(y, a0, a1, px_y) > 0.5) {
            color = vec3<f32>(0.42, 0.88, 0.50);
            alpha = 0.95;
        }
        if (line_mask(y, b0, b1, px_y) > 0.5) {
            color = vec3<f32>(0.96, 0.65, 0.30);
            alpha = 0.95;
        }
    }

    return vec4<f32>(color, alpha);
}
