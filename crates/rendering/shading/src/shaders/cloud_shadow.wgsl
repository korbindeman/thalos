// Shared cloud sun-transmittance sampler — the receiving half of CLOUD-5 / W2
// (`docs/rendering/clouds.md` §3.5).
//
// The cloud compute pass marches a view-anchored cascade of the SAME density
// field the visible volume renders, storing, per texel of a reference plane
// under the view anchor, the fraction of the sun beam that survives the deck to
// reach that point. This module is the one place that reads it back, so terrain,
// foliage, rock, and hull cannot end up under different weather.
//
// The lookup is a parallax projection, not a vertical one: a receiver
// intersects ITS OWN sun ray with the reference plane and samples there. That
// is what puts a shadow kilometres downwind of the cloud casting it when the
// sun is low, and what keeps a mountainside and the valley floor below it under
// different parts of the same deck. A straight overhead-coverage lookup is the
// projection the spec rejects as a finished implementation — it can only ever
// stamp clouds onto the ground directly beneath them.
//
// Only the DIRECT sun term is gated by this. Ambient sky light is not: ground
// under solid overcast is dim and flat, not black, and the sky fill is what
// carries that. (Modulating the ambient by local overhead optical depth is the
// remaining half of §3.5 — it needs the overcast tail this cascade does not yet
// have, so overcast ambient stays the sky LUT's business for now.)

#define_import_path thalos::cloud_shadow

// Mirror of `CloudShadowBlock` (clouds/shadow_frame.rs, encase std140); field
// order is load-bearing. Embedded in each receiving material's existing params
// uniform — not its own binding, because the terrain pipeline is already at the
// Metal 16-vertex-buffer ceiling.
//
// `axis_v.w == 0` ⇒ no live cascade, and every entry point below early-outs
// fully lit, so an unwritten block cannot darken anything.
struct CloudShadowBlock {
    // World render space → body-fixed rotation (xyzw quaternion).
    world_to_body: vec4<f32>,
    // xyz = body centre in world render space, w = artistic strength (0 = off).
    body_center_ws: vec4<f32>,
    // xyz = map centre, body-fixed, w = half extent in metres.
    center: vec4<f32>,
    // xyz = +u tangent, w = metres per texel.
    axis_u: vec4<f32>,
    // xyz = +v tangent, w = MODE: 0 = no cascade, 1 = apply, 2 = paint the raw
    // transmittance (`THALOS_CLOUD_SHADOW=show`). Same lane convention as
    // `ShadowCascadeBlock::gate.z` uses for the contact tier, so a receiver
    // reads one kind of flag for both.
    axis_v: vec4<f32>,
    // xyz = reference-plane normal, w = sun elevation cosine at the centre
    // (= dot(up, sun), the denominator of the plane intersection below).
    up_sun: vec4<f32>,
    // xyz = body-fixed unit direction toward the sun — the axis the cascade was
    // marched along, so receivers project along the same beam. w reserved.
    sun_body: vec4<f32>,
}

// Fraction of the map's half-extent over which the term feathers back to fully
// lit. Without it the cascade border draws itself as a hard disc edge across
// the ground; with it the transition lands in the far field where a missing
// shadow reads as haze rather than as a line. (The planet-scale tail that would
// remove the border entirely is the cube-integral half of §3.5, not yet built.)
const CLOUD_SHADOW_EDGE_FADE: f32 = 0.12;

// True when the capture axis asked receivers to paint the cascade instead of
// shading it in.
fn cloud_shadow_debug(block: CloudShadowBlock) -> bool {
    return block.axis_v.w > 1.5;
}

fn cloud_shadow_quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Where this fragment lands in the cascade, as a texture UV. Returns the
// out-of-range flag separately so callers can feather rather than clamp.
struct CloudShadowLookup {
    uv: vec2<f32>,
    coverage: f32,
    // Map coordinates in [-1,1] before the border feather — kept so the
    // diagnostic paint can show HOW far outside a fragment landed, not just
    // that it did.
    local: vec2<f32>,
}

fn cloud_shadow_lookup(block: CloudShadowBlock, world_pos: vec3<f32>) -> CloudShadowLookup {
    var out: CloudShadowLookup;
    out.uv = vec2<f32>(0.5);
    out.coverage = 0.0;
    out.local = vec2<f32>(0.0);
    if (block.axis_v.w < 0.5 || block.body_center_ws.w <= 0.0) {
        return out;
    }
    // Into the marcher's frame, through the marcher's own rotation.
    let p = cloud_shadow_quat_rotate(
        block.world_to_body,
        world_pos - block.body_center_ws.xyz,
    );
    // Walk this fragment's own sun ray to the reference plane. `sun_elev` is
    // guaranteed above the producer's horizon cut-off (the frame goes inactive
    // below it), so this cannot blow up. `t` is negative for a receiver ABOVE
    // the plane — an aircraft, a summit — and the intersection is still the
    // right texel for it, because the map integrates the deck only: the segment
    // between plane and receiver is clear air either way.
    let up = block.up_sun.xyz;
    let sun_elev = max(block.up_sun.w, 1.0e-3);
    let t = dot(block.center.xyz - p, up) / sun_elev;
    let hit = p + block.sun_body.xyz * t;

    let offset = hit - block.center.xyz;
    let half_extent = max(block.center.w, 1.0);
    let local = vec2<f32>(dot(offset, block.axis_u.xyz), dot(offset, block.axis_v.xyz))
        / half_extent;

    // Feather to lit at the border (see CLOUD_SHADOW_EDGE_FADE).
    let edge = max(abs(local.x), abs(local.y));
    out.coverage = 1.0 - smoothstep(1.0 - CLOUD_SHADOW_EDGE_FADE, 1.0, edge);
    out.uv = local * 0.5 + vec2<f32>(0.5);
    out.local = local;
    return out;
}

// The cascade's raw payload at this fragment: r = beam transmittance,
// g = the local coverage the march saw, b = the optical depth it accumulated.
// `THALOS_CLOUD_SHADOW=show` paints this, so one capture distinguishes "clear
// sky here" (g ≈ 0) from "the march never entered the deck" (g > 0, b = 0) from
// "the receiver is projecting into the wrong texel" (a plausible field that
// does not line up with the clouds overhead).
fn cloud_shadow_payload(
    block: CloudShadowBlock,
    tex: texture_2d<f32>,
    samp: sampler,
    world_pos: vec3<f32>,
) -> vec3<f32> {
    let lookup = cloud_shadow_lookup(block, world_pos);
    if (lookup.coverage <= 0.0) {
        // Outside the cascade: paint the map coordinate itself (wrapped), with
        // a blue flag. A smooth ramp means the projection is sound and the map
        // is merely too small; stripes mean it is off by orders of magnitude.
        return vec3<f32>(fract(lookup.local * 0.5 + 0.5), 1.0);
    }
    return textureSampleLevel(tex, samp, lookup.uv, 0.0).rgb;
}

// Sun-beam transmittance reaching `world_pos` (world render space) through the
// cloud deck. 1 = unshadowed. Cheap enough for every opaque fragment: one
// rotation, one ray-plane intersection, one bilinear fetch.
fn cloud_sun_transmittance(
    block: CloudShadowBlock,
    tex: texture_2d<f32>,
    samp: sampler,
    world_pos: vec3<f32>,
) -> f32 {
    let lookup = cloud_shadow_lookup(block, world_pos);
    if (lookup.coverage <= 0.0) {
        return 1.0;
    }
    let transmittance = clamp(
        textureSampleLevel(tex, samp, lookup.uv, 0.0).r,
        0.0,
        1.0,
    );
    let strength = clamp(block.body_center_ws.w, 0.0, 1.0) * lookup.coverage;
    return mix(1.0, transmittance, strength);
}
