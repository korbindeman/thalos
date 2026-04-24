// Procedural stainless-steel panel + rivet surface, layered on top of
// Bevy's StandardMaterial via ExtendedMaterial.
//
// Authoring model
// ---------------
// We carry a procedural height field h(u, v) defined in meters over the
// side surface, where:
//   u ∈ [0, 1)  — angular position (mesh UV.x), wrapping around the circumference
//   v ∈ [0, 1]  — axial position   (mesh UV.y), from the -Y end (v=0) to +Y end (v=1)
// The field has two contributions: horizontal panel seams (negative
// grooves) and circular rivets sitting on those seams (positive domes).
// Seams and rivets are quantized to integer counts across the surface so
// the pattern closes seamlessly — no UV-seam artefacts.
//
// Cylinders vs. conical frustums
// ------------------------------
// The surface is a cylinder when `radius_top == radius_bottom`, otherwise
// a conical frustum. `length` is the surface (slant) distance from v=0
// to v=1, which equals the vertical height for cylinders and
// sqrt(height² + Δr²) for cones. The radius at a given v interpolates
// linearly between the two ends. Rivet *count* per ring is a single
// integer derived from the midpoint circumference so all rings align
// vertically; physical rivet spacing then drifts slightly at the wide
// and narrow ends on steep cones, accepted for this proc-detail layer.
//
// The normal perturbation comes from the analytic-ish gradient of h
// (finite difference in metric units). The gradient is in tangent space
// and applied via a TBN reconstructed from the interpolated geometric
// normal plus an assumed axial direction (local +Y). The bitangent is
// `+Y` projected onto the tangent plane, which follows the cone slope
// correctly without additional work. The editor never rotates parts, so
// using local +Y directly is exact; when parts start rotating in-game
// we'll need real vertex tangents.
//
// End caps are detected by the geometric normal being near-parallel to
// the axis — there the detail mask fades out and the base PBR shows
// through.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_view_bindings::view,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

struct ShipPartParams {
    length: f32,
    radius_top: f32,
    panel_pitch: f32,
    rivet_spacing: f32,
    tint: vec3<f32>,
    rivet_height: f32,
    rivet_radius: f32,
    seam_depth: f32,
    seam_half_width: f32,
    seed: u32,
    rivet_seam_offset: f32,
    rivet_mid_rows: u32,
    radius_bottom: f32,
    _pad0: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> part: ShipPartParams;

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717958;

// Hash a 2D integer coordinate to [0, 1). Cheap, stable, good enough for
// subtle per-panel / per-rivet color and roughness jitter.
fn hash21(p: vec2<i32>) -> f32 {
    let n = u32(p.x) * 374761393u + u32(p.y) * 668265263u + part.seed * 1274126177u;
    let m = (n ^ (n >> 13u)) * 1274126177u;
    return f32(m & 0xffffffu) / f32(0x1000000u);
}

// Radius at parameter v. Linear between the two ends of the mesh;
// collapses to a constant for cylinders.
fn radius_at_v(v: f32) -> f32 {
    return mix(part.radius_bottom, part.radius_top, v);
}

// Arc-length around the body at a given v, in meters. Used whenever a
// physical distance in the circumferential direction depends on the
// *local* radius (e.g. finite-difference step size, rivet u-distance).
fn circumference_at_v(v: f32) -> f32 {
    return TAU * radius_at_v(v);
}

// Circumference at the mid-v point — the value we quantize the rivet
// count against. Picking a single count (rather than one per ring)
// keeps rivets aligned vertically across all rings; on cones the
// *physical* rivet spacing drifts at the wide/narrow ends as a result,
// tolerated for this detail layer.
fn mid_circumference() -> f32 {
    return PI * (part.radius_top + part.radius_bottom);
}

// Count of panel rings stacked along the axis. Rounded up from
// length / panel_pitch so every part gets at least one panel and seam
// spacing adapts smoothly to length changes. `length` is the *slant*
// distance on cones, so seams stay evenly spaced along the surface.
fn panel_count() -> f32 {
    return max(1.0, round(part.length / part.panel_pitch));
}

// Count of rivets around a single ring. Integer so the angular pattern
// closes at the UV seam without a gap. Derived from the midpoint
// circumference so all rings share the same count (see notes above).
fn rivet_ring_count() -> f32 {
    return max(4.0, round(mid_circumference() / part.rivet_spacing));
}

// Locate the nearest rivet ring to `v` and return its v coordinate.
//
// Rivet rings belong to two families:
//
//   1. PAIRED rings — every seam has two, one on each side, offset by
//      `rivet_seam_offset` meters. This matches how plate skin is
//      fastened to ring stringers just above and just below each weld.
//
//   2. MID rings — `rivet_mid_rows` additional rings interior to each
//      panel, evenly distributed between the paired rings.
//
// We don't need to inspect every ring on the tank: at most one paired
// ring and one mid ring from each of the two panels adjacent to the
// nearest seam can beat every other ring on distance. Iterating that
// small candidate set keeps the shader shallow.
fn nearest_rivet_ring_v(v: f32) -> f32 {
    let panels = panel_count();
    let offset_v = part.rivet_seam_offset / max(part.length, 1.0e-4);
    let k_near = round(v * panels);
    let mid = part.rivet_mid_rows;
    let mid_div = f32(mid) + 1.0;

    // Paired rings at the nearest seam, ±offset.
    var best_row_v = (k_near / panels) - offset_v;
    var best_dv = abs(v - best_row_v);

    let paired_below = (k_near / panels) + offset_v;
    let d_below = abs(v - paired_below);
    if (d_below < best_dv) {
        best_dv = d_below;
        best_row_v = paired_below;
    }

    // Mid rings in the two panels flanking the nearest seam. Panel p
    // spans seams p → p+1; its mid ring i (0-based) sits at
    // (p + (i+1)/(M+1)) / P in v.
    for (var p_off: i32 = -1; p_off <= 0; p_off = p_off + 1) {
        let p = k_near + f32(p_off);
        for (var i: u32 = 0u; i < mid; i = i + 1u) {
            let frac = (f32(i) + 1.0) / mid_div;
            let row_v = (p + frac) / panels;
            let dv = abs(v - row_v);
            if (dv < best_dv) {
                best_dv = dv;
                best_row_v = row_v;
            }
        }
    }

    return best_row_v;
}

// Height field in meters at (u, v), on the cylinder side.
//
// Layers:
//   1. Horizontal groove at each panel boundary (v = k / panel_count).
//      Depth is `seam_depth` at the seam, falling to 0 at `seam_half_width`.
//   2. Positive dome at each rivet — rivets sit on the nearest rivet
//      ring (paired above/below each seam, plus optional mid-panel
//      rings), at `rivet_ring_count` angular positions per ring.
fn sample_height(u: f32, v: f32) -> f32 {
    var h = 0.0;

    let panels = panel_count();
    let v_scaled = v * panels;
    let v_nearest_seam = round(v_scaled);
    let v_dist_panels = abs(v_scaled - v_nearest_seam);
    let v_dist_m = v_dist_panels * (part.length / panels);

    // Seam groove
    let seam_t = 1.0 - smoothstep(0.0, part.seam_half_width, v_dist_m);
    h -= seam_t * part.seam_depth;

    // Rivet ring — drop any ring that would land outside the tank
    // (e.g. paired ring above the top seam on the top panel).
    let row_v = nearest_rivet_ring_v(v);
    if (row_v < 0.0 || row_v > 1.0) {
        return h;
    }

    let rivets = rivet_ring_count();
    let u_scaled = u * rivets;
    let u_nearest_rivet = round(u_scaled);
    // Use the *local* circumference so the rivet footprint stays
    // isotropic in real meters at whatever height on the cone we're at.
    let u_dist_m = (u_scaled - u_nearest_rivet) * (circumference_at_v(v) / rivets);
    let v_from_row_m = (v - row_v) * part.length;
    let rivet_r_m = sqrt(u_dist_m * u_dist_m + v_from_row_m * v_from_row_m);
    let rivet_t = 1.0 - smoothstep(0.0, part.rivet_radius, rivet_r_m);
    h += rivet_t * part.rivet_height;

    return h;
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Side-vs-cap mask. The cylinder mesh has three regions: +Y cap,
    // side, -Y cap. End caps have a normal that's nearly parallel to
    // local +Y; sides have a normal perpendicular to it. We detect this
    // from the geometric (pre-normal-map) world normal. Because parts
    // aren't rotated in the editor, local Y ≈ world Y; the assumption
    // is revisited in the file header.
    let axis_ws = vec3<f32>(0.0, 1.0, 0.0);
    let axis_dot = abs(dot(normalize(pbr_input.world_normal), axis_ws));
    let side_mask = 1.0 - smoothstep(0.80, 0.95, axis_dot);

    if (side_mask > 0.001) {
        let u = in.uv.x;
        let v = in.uv.y;

        let h_center = sample_height(u, v);

        // Finite-difference gradient of h in metric space. Step size is
        // chosen a little smaller than the smallest feature (rivet
        // radius) so we don't alias across a rivet boundary. On a cone,
        // the circumferential step in UV has to account for the local
        // (v-dependent) radius.
        let step_m = max(part.rivet_radius * 0.25, 1.0e-4);
        let du = step_m / max(circumference_at_v(v), 1.0e-4);
        let dv = step_m / max(part.length, 1.0e-4);
        let h_u = sample_height(u + du, v);
        let h_v = sample_height(u, v + dv);
        let dh_du_m = (h_u - h_center) / step_m; // unitless slope
        let dh_dv_m = (h_v - h_center) / step_m;

        // Tangent-space normal. X aligns with +u (circumferential), Y
        // with +v (axial, pointing toward the +Y cap), Z is outward.
        // Scale the gradient by `side_mask` so the perturbation fades
        // cleanly into the caps.
        let strength = side_mask;
        var n_ts = normalize(vec3<f32>(
            -dh_du_m * strength,
            -dh_dv_m * strength,
             1.0,
        ));

        // Rebuild a TBN from the geometric normal + local axis. The
        // bitangent points along +v (toward +Y cap); the tangent points
        // along +u (the direction the angular UV increases, i.e. around
        // the cylinder).
        let n_ws = normalize(pbr_input.world_normal);
        var b_ws = axis_ws - n_ws * dot(axis_ws, n_ws);
        b_ws = normalize(b_ws);
        let t_ws = normalize(cross(b_ws, n_ws));

        let n_perturbed = normalize(
              t_ws * n_ts.x
            + b_ws * n_ts.y
            + n_ws * n_ts.z,
        );
        pbr_input.N = n_perturbed;

        // Seam darkening: grooves read a touch darker + slightly rougher
        // because the weld oxidizes differently than the polished plate.
        let seam_t = clamp(-h_center / max(part.seam_depth, 1.0e-5), 0.0, 1.0);
        let groove_mask = seam_t * side_mask;
        pbr_input.material.base_color = vec4<f32>(
            pbr_input.material.base_color.rgb * mix(vec3<f32>(1.0), vec3<f32>(0.82, 0.84, 0.87), groove_mask),
            pbr_input.material.base_color.a,
        );
        pbr_input.material.perceptual_roughness = clamp(
            pbr_input.material.perceptual_roughness + 0.18 * groove_mask,
            0.04,
            1.0,
        );

        // Per-panel color jitter: different mill lots, heat tints, etc.
        // Very subtle — enough to break up the repetition without
        // looking patchwork.
        let panels = panel_count();
        let panel_index = i32(floor(v * panels));
        let rivets = rivet_ring_count();
        let ring_index = i32(floor(u * rivets));
        let jitter = hash21(vec2<i32>(ring_index / 4, panel_index));
        let jitter_gain = 0.04 * side_mask;
        pbr_input.material.base_color = vec4<f32>(
            pbr_input.material.base_color.rgb * (1.0 - jitter_gain + 2.0 * jitter_gain * jitter),
            pbr_input.material.base_color.a,
        );
    }

    // Global tint — used by the editor for selection / hover highlight.
    pbr_input.material.base_color = vec4<f32>(
        pbr_input.material.base_color.rgb * part.tint,
        pbr_input.material.base_color.a,
    );

    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
