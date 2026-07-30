// Airliner-style navigation display: a heading-up top-down map. The craft is a
// fixed triangle at the centre pointing up; a compass rose rotates by the
// current heading; runways are drawn at their TRUE size and the armed approach
// route is drawn as a real polyline.
//
// Fed by `crates/runtime/game/src/hud/mfd/widgets/nav_display.rs`; the
// `NavDisplayData` layout here must mirror that file's struct field-for-field.
//
// Everything is drawn procedurally in a centred [-1, 1] space (y points down).
// The CPU has already projected and rotated every element into this heading-up
// space and scaled it by the plot range — so this shader never sees metres, and
// a runway's half-length here really is "how much of the plot it covers".
#import bevy_ui::ui_vertex_output::UiVertexOutput

const PI: f32 = 3.14159265;
const MAX_RUNWAYS: u32 = 8u;
// Route polyline points, packed two per vec4 (xy, zw).
const MAX_ROUTE_POINTS: u32 = 48u;
const MAX_WAYPOINTS: u32 = 4u;
// Sentinel for "this angular marker is not present".
const NO_ANGLE: f32 = 1.0e8;

struct NavDisplayData {
    // x = runway_count, y = route_point_count, z = waypoint_count,
    // w = index of the first route point on the final approach segment
    params: vec4<f32>,
    // x = ring radius, y = craft radius, z = line half-width, w = tick half-width
    geom: vec4<f32>,
    // x = dash period, y = dash duty, z = route half-width, w = min runway half-width
    style: vec4<f32>,
    // x = heading (rad), y = tick length, z = north-tick length,
    // w = bearing-to-destination marker (heading-up rad, or NO_ANGLE)
    nav: vec4<f32>,
    // x = range-ring radius (0 = none), y = ground-track marker (heading-up rad,
    // or NO_ANGLE), z = unused, w = unused
    extra: vec4<f32>,
    col_ring: vec4<f32>,
    col_tick: vec4<f32>,
    col_north: vec4<f32>,
    col_craft: vec4<f32>,
    col_runway: vec4<f32>,
    col_runway_armed: vec4<f32>,
    col_route: vec4<f32>,
    col_route_final: vec4<f32>,
    col_waypoint: vec4<f32>,
    // per runway: xy = centre (plot), zw = along-strip unit direction (heading-up)
    runways: array<vec4<f32>, 8>,
    // per runway: x = half-length, y = half-width (both plot units),
    // z = 1 if armed, w = +1/-1 for which end along `zw` is the landing threshold
    runway_ext: array<vec4<f32>, 8>,
    // route polyline, two points per element
    route: array<vec4<f32>, 24>,
    // per waypoint: xy = position (plot), z = kind (0 fix, 1 FAP, 2 threshold, 3 aim)
    waypoints: array<vec4<f32>, 4>,
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

// Antialiased "inside" coverage for a signed distance (positive inside).
fn fill(sd: f32) -> f32 {
    let aa = max(fwidth(sd), 0.001);
    return smoothstep(-aa, aa, sd);
}

// Antialiased line coverage for a distance-to-centreline and half-width.
fn stroke(dist: f32, half_width: f32) -> f32 {
    let aa = max(fwidth(dist), 0.001);
    return 1.0 - smoothstep(half_width - aa, half_width + aa, dist);
}

// Unpack route point `i` from the two-per-vec4 array.
fn route_point(i: u32) -> vec2<f32> {
    let v = data.route[i / 2u];
    if (i % 2u == 0u) {
        return v.xy;
    }
    return v.zw;
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // Centred coords: [-1, 1] across the square node, y points down.
    let c = (in.uv - vec2<f32>(0.5)) * 2.0;

    var col = vec4<f32>(0.0);

    let runway_count = u32(data.params.x + 0.5);
    let route_count = u32(data.params.y + 0.5);
    let waypoint_count = u32(data.params.z + 0.5);
    let route_final_index = u32(max(data.params.w, 0.0) + 0.5);
    let ring_r = data.geom.x;
    let craft_r = data.geom.y;
    let line_hw = data.geom.z;
    let tick_hw = data.geom.w;
    let dash_period = max(data.style.x, 1e-4);
    let dash_duty = data.style.y;
    let route_hw = data.style.z;
    let rwy_min_hw = data.style.w;
    let heading = data.nav.x;
    let tick_len = data.nav.y;
    let north_tick_len = data.nav.z;
    let bearing_marker = data.nav.w;
    let range_ring_r = data.extra.x;
    let track_marker = data.extra.y;

    let radial = length(c);

    // --- Compass ring, plus a dashed half-range ring for distance sense.
    let ring = stroke(abs(radial - ring_r), tick_hw);
    col = over(col, data.col_ring.rgb, ring * data.col_ring.a);
    if (range_ring_r > 0.0) {
        let angle = atan2(c.x, -c.y);
        let dashes = 1.0 - smoothstep(0.35, 0.5, fract(angle / (PI / 24.0)));
        let inner = stroke(abs(radial - range_ring_r), tick_hw * 0.7);
        col = over(col, data.col_ring.rgb, inner * dashes * data.col_ring.a * 0.75);
    }

    // --- Heading ticks every 30°, rotating so they read heading-up. The north
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
        var tick_col = data.col_tick;
        if (is_north) {
            tick_col = data.col_north;
        }
        col = over(col, tick_col.rgb, stroke(r.x, tick_hw) * tick_col.a);
    }

    // --- Runways, at true scale. A strip narrower than `rwy_min_hw` in plot
    // units still draws at that minimum, so a 90 m-wide strip stays visible at
    // long range instead of vanishing between pixels — the length stays true,
    // which is the dimension a pilot is judging.
    for (var i: u32 = 0u; i < runway_count && i < MAX_RUNWAYS; i = i + 1u) {
        let centre = data.runways[i].xy;
        let along = data.runways[i].zw;
        let across = vec2<f32>(along.y, -along.x);
        let half_len = data.runway_ext[i].x;
        let half_wid = max(data.runway_ext[i].y, rwy_min_hw);
        let armed = data.runway_ext[i].z > 0.5;
        let threshold_sign = data.runway_ext[i].w;

        var rwy_col = data.col_runway;
        if (armed) {
            rwy_col = data.col_runway_armed;
        }

        // Filled, oriented rectangle.
        let rel = c - centre;
        let dl = half_len - abs(dot(rel, along));
        let dw = half_wid - abs(dot(rel, across));
        col = over(col, rwy_col.rgb, fill(min(dl, dw)) * rwy_col.a);

        // Threshold bar: a stripe ACROSS the end you land on, so the approach
        // direction reads without any text. It must be clearly wider than the
        // strip and clearly thinner along it — sized symmetrically it just reads
        // as a box stuck on the end.
        let thr = centre + along * (half_len * threshold_sign);
        let bar_half_across = half_wid * 3.0;
        let bar_half_along = min(half_wid * 0.9, half_len * 0.25);
        let brel = c - thr;
        let bl = bar_half_along - abs(dot(brel, along));
        let bw = bar_half_across - abs(dot(brel, across));
        col = over(col, rwy_col.rgb, fill(min(bl, bw)) * rwy_col.a);
    }

    // --- The armed route: a real polyline through the planned legs. Segments on
    // the final approach draw in their own brighter colour, so "where the turn
    // ends and the stabilised approach begins" is visible at a glance.
    if (route_count >= 2u) {
        for (var i: u32 = 0u; i + 1u < route_count && i + 1u < MAX_ROUTE_POINTS; i = i + 1u) {
            let a = route_point(i);
            let b = route_point(i + 1u);
            let r = seg_dist(c, a, b);
            var route_col = data.col_route;
            var hw = route_hw;
            if (i >= route_final_index) {
                route_col = data.col_route_final;
                hw = route_hw * 1.35;
            }
            col = over(col, route_col.rgb, stroke(r.x, hw) * route_col.a);
        }
    }

    // --- Waypoint symbols: a diamond for a plain fix / final approach point, a
    // small square for the threshold and aim points.
    for (var i: u32 = 0u; i < waypoint_count && i < MAX_WAYPOINTS; i = i + 1u) {
        let p = data.waypoints[i].xy;
        let kind = data.waypoints[i].z;
        let rel = abs(c - p);
        let size = craft_r * 0.55;
        var sd: f32;
        if (kind >= 1.5) {
            // Square outline (threshold / aim).
            sd = size - max(rel.x, rel.y);
        } else {
            // Diamond outline (fix / final approach point).
            sd = size - (rel.x + rel.y);
        }
        // Outline only: fill minus an inset fill.
        let outline = fill(sd) - fill(sd - line_hw * 2.0);
        col = over(col, data.col_waypoint.rgb, clamp(outline, 0.0, 1.0) * data.col_waypoint.a);
    }

    // --- Ground-track line: a short, faint dashed line from the craft along its
    // actual track, which is what shows drift against the heading-up plot. It is
    // deliberately dim and short: at full brightness and full length it lies
    // right on top of the route on a straight-in and hides it.
    if (track_marker < NO_ANGLE) {
        let dir = vec2<f32>(sin(track_marker), -cos(track_marker));
        let tip = ring_r * 0.32;
        let r = seg_dist(c, dir * (craft_r * 1.4), dir * tip);
        let ph = fract(r.y * tip / dash_period);
        let dash = 1.0 - smoothstep(dash_duty - 0.08, dash_duty + 0.08, ph);
        col = over(col, data.col_craft.rgb, stroke(r.x, line_hw * 0.6) * dash * 0.45);
    }

    // --- Bearing pointer: a caret riding the compass ring at the bearing to the
    // armed threshold.
    if (bearing_marker < NO_ANGLE) {
        let dir = vec2<f32>(sin(bearing_marker), -cos(bearing_marker));
        let side = vec2<f32>(dir.y, -dir.x);
        let tip = dir * (ring_r + tick_hw * 2.0);
        let base = dir * (ring_r + tick_hw * 8.0);
        let p0 = tip;
        let p1 = base + side * tick_hw * 4.0;
        let p2 = base - side * tick_hw * 4.0;
        let inside = min(min(edge_dist(p0, p1, c), edge_dist(p1, p2, c)), edge_dist(p2, p0, c));
        col = over(
            col,
            data.col_route_final.rgb,
            fill(inside) * data.col_route_final.a,
        );
    }

    // --- Craft glyph: a filled triangle at the centre pointing up.
    let p0 = vec2<f32>(0.0, -craft_r);
    let p1 = vec2<f32>(craft_r * 0.62, craft_r * 0.55);
    let p2 = vec2<f32>(-craft_r * 0.62, craft_r * 0.55);
    let inside = min(
        min(edge_dist(p0, p1, c), edge_dist(p1, p2, c)),
        edge_dist(p2, p0, c),
    );
    col = over(col, data.col_craft.rgb, fill(inside) * data.col_craft.a);

    return col;
}
