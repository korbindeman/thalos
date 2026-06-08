// Burn-time system schematic: a top-down (XZ ecliptic) plot of the ship's
// local system centred on the dominant body, with the current ballistic
// trajectory drawn solid and the maneuver-node projected trajectory drawn
// dotted. Fed by `crates/game/src/hud/system_map_panel.rs`; the
// `SystemMapData` layout here must mirror that file's struct field-for-field.
//
// Everything is drawn procedurally as signed-distance shapes in a centred
// [-1, 1] coordinate space (origin = dominant body). The CPU side has already
// projected and normalised every point into this space.
#import bevy_ui::ui_vertex_output::UiVertexOutput

struct SystemMapData {
    // x = ring_count, y = solid_count, z = dotted_count, w = node_flag
    params: vec4<f32>,
    // x = central radius, y = ship radius, z = node radius, w = line half-width
    geom: vec4<f32>,
    // x = ring half-width, y = dash period, z = dash duty, w = body-dot radius
    style: vec4<f32>,
    // xy = ship marker pos, zw = maneuver-node marker pos (all in [-1, 1])
    markers: vec4<f32>,
    col_central: vec4<f32>,
    col_ring: vec4<f32>,
    col_body: vec4<f32>,
    col_solid: vec4<f32>,
    col_dotted: vec4<f32>,
    col_ship: vec4<f32>,
    col_node: vec4<f32>,
    // per ring: xy = body pos, z = ring radius, w = unused (dot radius is style.w)
    rings: array<vec4<f32>, 8>,
    // per point: xy = position, z = cumulative arc-length, w = valid flag
    solid: array<vec4<f32>, 96>,
    dotted: array<vec4<f32>, 96>,
}

@group(1) @binding(0) var<uniform> data: SystemMapData;

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

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // Centred coords: [-1, 1] across the square node, y points down.
    let c = (in.uv - vec2<f32>(0.5)) * 2.0;

    var col = vec4<f32>(0.0);

    let ring_count = u32(data.params.x + 0.5);
    let solid_count = u32(data.params.y + 0.5);
    let dotted_count = u32(data.params.z + 0.5);
    let node_flag = data.params.w;

    let central_r = data.geom.x;
    let ship_r = data.geom.y;
    let node_r = data.geom.z;
    let line_hw = data.geom.w;
    let ring_hw = data.style.x;
    let dash_period = max(data.style.y, 1e-4);
    let dash_duty = data.style.z;
    let body_r = data.style.w;

    let radial = length(c);
    let radial_aa = max(fwidth(radial), 0.001);

    // Orbit rings, concentric about the dominant body at the origin.
    for (var i: u32 = 0u; i < ring_count; i = i + 1u) {
        let d = abs(radial - data.rings[i].z);
        let a = 1.0 - smoothstep(ring_hw - radial_aa, ring_hw + radial_aa, d);
        col = over(col, data.col_ring.rgb, a * data.col_ring.a);
    }

    // Dominant body (filled disc).
    let cd = radial - central_r;
    let ca = 1.0 - smoothstep(-radial_aa, radial_aa, cd);
    col = over(col, data.col_central.rgb, ca * data.col_central.a);

    // Child-body dots sitting on their rings.
    for (var i: u32 = 0u; i < ring_count; i = i + 1u) {
        let dist = length(c - data.rings[i].xy);
        let baa = max(fwidth(dist), 0.001);
        let a = 1.0 - smoothstep(body_r - baa, body_r + baa, dist);
        col = over(col, data.col_body.rgb, a * data.col_body.a);
    }

    // Current ballistic trajectory (solid).
    if (solid_count >= 2u) {
        var best = 1e9;
        for (var i: u32 = 0u; i + 1u < solid_count; i = i + 1u) {
            let r = seg_dist(c, data.solid[i].xy, data.solid[i + 1u].xy);
            best = min(best, r.x);
        }
        let aa = max(fwidth(best), 0.001);
        let a = 1.0 - smoothstep(line_hw - aa, line_hw + aa, best);
        col = over(col, data.col_solid.rgb, a * data.col_solid.a);
    }

    // Maneuver-node projected trajectory (dotted via arc-length).
    if (dotted_count >= 2u) {
        var best = 1e9;
        var best_arc = 0.0;
        for (var i: u32 = 0u; i + 1u < dotted_count; i = i + 1u) {
            let r = seg_dist(c, data.dotted[i].xy, data.dotted[i + 1u].xy);
            if (r.x < best) {
                best = r.x;
                best_arc = mix(data.dotted[i].z, data.dotted[i + 1u].z, r.y);
            }
        }
        let aa = max(fwidth(best), 0.001);
        let line = 1.0 - smoothstep(line_hw - aa, line_hw + aa, best);
        let ph = fract(best_arc / dash_period);
        let dash = 1.0 - smoothstep(dash_duty - 0.08, dash_duty + 0.08, ph);
        col = over(col, data.col_dotted.rgb, line * dash * data.col_dotted.a);
    }

    // Maneuver-node marker (hollow ring).
    if (node_flag > 0.5) {
        let dist = length(c - data.markers.zw);
        let naa = max(fwidth(dist), 0.001);
        let d = abs(dist - node_r);
        let a = 1.0 - smoothstep(ring_hw - naa, ring_hw + naa, d);
        col = over(col, data.col_node.rgb, a * data.col_node.a);
    }

    // Ship marker (filled, drawn last so it stays on top).
    let sdist = length(c - data.markers.xy);
    let saa = max(fwidth(sdist), 0.001);
    let sa = 1.0 - smoothstep(ship_r - saa, ship_r + saa, sdist);
    col = over(col, data.col_ship.rgb, sa * data.col_ship.a);

    return col;
}
