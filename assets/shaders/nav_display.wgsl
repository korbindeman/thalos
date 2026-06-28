// Airliner-style navigation display: a heading-up top-down map. The craft is a
// fixed triangle at the centre pointing up; a compass rose rotates by the
// current heading; runways are oriented symbols with a dashed extended-
// centerline approach path. Fed by `crates/game/src/hud/mfd/widgets/nav_display.rs`;
// the `NavDisplayData` layout here must mirror that file's struct field-for-field.
//
// Everything is drawn procedurally in a centred [-1, 1] space (y points down).
// The CPU has already projected runways into this heading-up space.
#import bevy_ui::ui_vertex_output::UiVertexOutput

const PI: f32 = 3.14159265;
const MAX_RUNWAYS: u32 = 8u;

struct NavDisplayData {
    // x = runway_count
    params: vec4<f32>,
    // x = ring radius, y = craft radius, z = runway half-length, w = line half-width
    geom: vec4<f32>,
    // x = tick half-width, y = dash period, z = dash duty, w = runway half-width
    style: vec4<f32>,
    // x = heading (rad), y = approach length, z = tick length, w = north-tick length
    nav: vec4<f32>,
    col_ring: vec4<f32>,
    col_tick: vec4<f32>,
    col_north: vec4<f32>,
    col_craft: vec4<f32>,
    col_runway: vec4<f32>,
    col_approach: vec4<f32>,
    // per runway: xy = centre (plot), zw = heading-up unit direction
    runways: array<vec4<f32>, 8>,
}

@group(1) @binding(0) var<uniform> data: NavDisplayData;

// Alpha-composite `src` (straight alpha) over `dst`.
fn over(dst: vec4<f32>, src_rgb: vec3<f32>, src_a: f32) -> vec4<f32> {
    let a = clamp(src_a, 0.0, 1.0);
    return vec4<f32>(mix(dst.rgb, src_rgb, a), dst.a + a * (1.0 - dst.a));
}

// Distance from point `p` to segment a→b, plus the clamped parameter t.
fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-8), 0.0, 1.0);
    let proj = a + ab * t;
    return vec2<f32>(length(p - proj), t);
}

// Signed perpendicular distance from `p` to the directed line a→b.
fn edge_dist(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    let e = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    return e / max(length(b - a), 1e-6);
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // Centred coords: [-1, 1] across the square node, y points down.
    let c = (in.uv - vec2<f32>(0.5)) * 2.0;

    var col = vec4<f32>(0.0);

    let runway_count = u32(data.params.x + 0.5);
    let ring_r = data.geom.x;
    let craft_r = data.geom.y;
    let rwy_half_len = data.geom.z;
    let line_hw = data.geom.w;
    let tick_hw = data.style.x;
    let dash_period = max(data.style.y, 1e-4);
    let dash_duty = data.style.z;
    let rwy_half_width = data.style.w;
    let heading = data.nav.x;
    let approach_len = data.nav.y;
    let tick_len = data.nav.z;
    let north_tick_len = data.nav.w;

    // Compass ring outline.
    let radial = length(c);
    let raa = max(fwidth(radial), 0.001);
    let ring = 1.0 - smoothstep(tick_hw - raa, tick_hw + raa, abs(radial - ring_r));
    col = over(col, data.col_ring.rgb, ring * data.col_ring.a);

    // Heading ticks every 30°, rotating so they read heading-up. The north
    // tick (bearing 0) is longer + coloured to anchor the rose.
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        let bearing = f32(i) * (PI / 6.0);
        let theta = bearing - heading;
        let outward = vec2<f32>(sin(theta), -cos(theta));
        let is_north = i == 0u;
        var len = tick_len;
        if (is_north) {
            len = north_tick_len;
        }
        let a = (ring_r - len) * outward;
        let b = ring_r * outward;
        let r = seg_dist(c, a, b);
        let taa = max(fwidth(r.x), 0.001);
        let ta = 1.0 - smoothstep(tick_hw - taa, tick_hw + taa, r.x);
        var tick_col = data.col_tick;
        if (is_north) {
            tick_col = data.col_north;
        }
        col = over(col, tick_col.rgb, ta * tick_col.a);
    }

    // Runways + their extended-centerline approach paths.
    for (var i: u32 = 0u; i < runway_count && i < MAX_RUNWAYS; i = i + 1u) {
        let centre = data.runways[i].xy;
        let along = data.runways[i].zw;
        let across = vec2<f32>(along.y, -along.x);

        // Filled, oriented rectangle.
        let rel = c - centre;
        let dl = rwy_half_len - abs(dot(rel, along));
        let dw = rwy_half_width - abs(dot(rel, across));
        let edge = min(dl, dw);
        let eaa = max(fwidth(edge), 0.001);
        let ra = smoothstep(-eaa, eaa, edge);
        col = over(col, data.col_runway.rgb, ra * data.col_runway.a);

        // Dashed approach centerline from the threshold backward along -along.
        let threshold = centre - along * rwy_half_len;
        let far = threshold - along * approach_len;
        let seg = seg_dist(c, threshold, far);
        let caa = max(fwidth(seg.x), 0.001);
        let line = 1.0 - smoothstep(line_hw - caa, line_hw + caa, seg.x);
        let arc = seg.y * approach_len;
        let ph = fract(arc / dash_period);
        let dash = 1.0 - smoothstep(dash_duty - 0.08, dash_duty + 0.08, ph);
        col = over(col, data.col_approach.rgb, line * dash * data.col_approach.a);
    }

    // Craft glyph: a filled triangle at the centre pointing up.
    let p0 = vec2<f32>(0.0, -craft_r);
    let p1 = vec2<f32>(craft_r * 0.62, craft_r * 0.55);
    let p2 = vec2<f32>(-craft_r * 0.62, craft_r * 0.55);
    let d0 = edge_dist(p0, p1, c);
    let d1 = edge_dist(p1, p2, c);
    let d2 = edge_dist(p2, p0, c);
    let inside = min(min(d0, d1), d2);
    let iaa = max(fwidth(inside), 0.001);
    let tri = smoothstep(-iaa, iaa, inside);
    col = over(col, data.col_craft.rgb, tri * data.col_craft.a);

    return col;
}
