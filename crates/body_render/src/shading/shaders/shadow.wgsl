// Shared cascaded sun-shadow sampler.
//
// One copy of the cascade walk + depth compare, used by every surface material
// in the one shadow world: the UDLOD terrain (`body_terrain.wgsl`), scattered
// trees (`tree.wgsl`), grass blades (`grass.wgsl`), rocks (`rock.wgsl`), the
// preview ground patch (`ground_patch.wgsl`), the craft hull
// (`ship_part.wgsl`), and the generic shadowed StandardMaterial extension
// (`shadowed_standard.wgsl` — structures, runway, plain craft parts). All bind
// the same per-cascade depth maps + the same `ShadowCascadeBlock` (published by
// the game's `rendering::sun_shadow` rig) and call one of the two entry points
// here — replacing the near-verbatim copies that had started to drift.
//
// Two entry points (stable-CSM, graphics-fidelity W6):
//
// - `sun_shadow_factor_nrm(pos, normal, ...)` — preferred. Uses the receiver's
//   surface normal for a **normal-offset** (slides the sample point off the
//   surface by ~a shadow-map texel before projecting) and a **slope-scaled
//   depth bias** (grows on geometry grazing to the sun, where one texel spans
//   a long stretch of surface). Together these kill low-sun acne while keeping
//   the flat-ground base bias small enough that contact shadows stay attached
//   (no peter-panning).
// - `sun_shadow_factor(pos, ...)` — for receivers with no meaningful surface
//   normal (tree foliage, grass blades). Same walk, but a larger constant
//   depth bias (`NO_NORMAL_BIAS_SCALE`) stands in for the missing offset.
//
// The three depth maps are passed as FUNCTION PARAMETERS: WGSL permits
// handle-typed (`texture_*`) arguments as long as the call site binds them to
// module-scope globals, so each material keeps its own bind-group indices and
// just hands its three maps in. No texture arrays (a depth array broke terrain).

#define_import_path thalos::shadow

// Mirror of `ShadowCascadeBlock` in `body_material.rs` (encase std140); field
// order is load-bearing. Array sizes == CASCADE_COUNT (3).
//
// NOTE the field is named `gate`, not `config`: `body_terrain.wgsl` `#import`s a
// udlod global named `config`, and a struct field of the same name collides in
// naga_oil (see the wgsl-bevy skill). Keep it `gate` everywhere.
struct ShadowCascadeBlock {
    view_proj: array<mat4x4<f32>, 3>,
    // per cascade: x = clip units per METRE of light-space depth
    // (= 1 / (far − near); orthographic z is linear), y = shadow-map texel
    // size in world metres, zw reserved. The shader derives its bias/offset
    // from the texel size and converts metres → clip via x.
    params: array<vec4<f32>, 3>,
    // x = strength (0 ⇒ skip), y = active cascade count, zw reserved.
    gate: vec4<f32>,
    // xyz = normalized render-space direction TOWARD the sun, w reserved.
    // Drives the slope-scaled bias + normal offset in `sun_shadow_factor_nrm`.
    sun_dir: vec4<f32>,
}

// ── Bias / offset model ───────────────────────────────────────────────────────
//
// Acne needs a margin proportional to the shadow-map TEXEL (the depth a tilted
// texel spans); but a margin larger than a caster's height along the light
// ERASES that caster's shadow outright ("too large of an offset causes the
// depth test to erroneously pass" — the classic over-bias failure). The far
// cascades' texels are metres wide — and scale further with the altitude
// footprint — so a purely texel-proportional bias there would exceed a whole
// tree (~10 m) or even a building, which is exactly how far-cascade shadows
// used to vanish while near ones looked fine. Hence every bias/offset here is
// texel-proportional WITH A HARD ABSOLUTE CAP well below the smallest caster
// class (trees). Above the cap, residual acne belongs to caster-receivers that
// are sub-pixel on screen at those cascade scales anyway; the dominant
// receivers (terrain, grass) never render into the cascade maps at all, so
// they cannot self-shadow-acne regardless.

// Texel multiplier for the base depth bias, and its growth per unit of the
// slope term (tanθ of the surface's grazing angle to the sun).
const BIAS_TEXELS: f32 = 0.75;
// Absolute clamp on the depth bias, metres. Must stay well below tree height.
const BIAS_MIN_M: f32 = 0.05;
const BIAS_MAX_M: f32 = 2.5;
// Constant-bias multiplier for the normal-less path: without the receiver
// offset, foliage receivers (which DO self-cast — leaf-on-leaf) need a larger
// margin. Tuned so the near/mid cascades reproduce the pre-W6 hand values
// (0.6 m / 2.5 m) while the far cascade tightens from 10 m → ~7 m.
const NO_NORMAL_BIAS_SCALE: f32 = 2.5;
// Receiver normal-offset, in texels of the sampled cascade's map, and its
// absolute cap in metres (same erase-the-caster argument as the bias cap).
const NORMAL_OFFSET_TEXELS: f32 = 1.2;
const NORMAL_OFFSET_MAX_M: f32 = 1.5;
// Cap for the slope term (tanθ of surface-to-sun grazing angle). Beyond this
// the surface is nearly parallel to the light and the shadow test is
// unreliable anyway; capping bounds peter-panning on cliff faces.
const MAX_SLOPE_SCALE: f32 = 3.0;

// Depth bias in metres for a cascade with the given texel size, at the given
// slope term. Texel-proportional, hard-capped (see the model note above).
fn cascade_bias_m(texel_m: f32, slope: f32) -> f32 {
    return clamp(texel_m * BIAS_TEXELS * (1.0 + slope), BIAS_MIN_M, BIAS_MAX_M);
}

// True when the current pass renders through an ORTHOGRAPHIC camera — in this
// project that means a sun-shadow cascade camera (or an offline bake rig), never
// a player view. Vegetation/rock casters use this to (a) bypass their clipmap
// scale-grow fades, whose reference `view.world_position + anchor` reconstructs
// the craft from the MAIN camera's position and is therefore wrong by
// (cascade-cam − main-cam) in the caster pass — at a boomed-out camera that
// error zeroed every caster's scale, which is why tree/rock shadows vanished as
// the camera moved away — and (b) pick the parallel light axis as the impostor
// facing direction instead of eye-relative.
fn is_ortho_projection(clip_from_view: mat4x4<f32>) -> bool {
    return clip_from_view[3][3] != 0.0;
}

// One cascade's shadow factor at a render-space point. Returns the factor in
// `[1 - strength, 1]`, or a NEGATIVE sentinel if the point is outside this
// cascade's box (the caller falls through to the next, coarser cascade). `inset`
// shrinks inner cascades so an edge fragment hands off cleanly; `fade` edge-
// softens the outermost cascade (nothing covers beyond it). 3×3 PCF.
//
// Reverse-z: a caster closer to the sun has a LARGER stored depth, so the
// receiver is shadowed when `stored > frag_depth + bias`. Uses `textureLoad`
// (no derivatives), so it is valid in both the fragment and vertex stages.
fn cascade_factor(
    world_pos: vec3<f32>,
    vp: mat4x4<f32>,
    bias: f32,
    strength: f32,
    tex: texture_depth_2d,
    inset: f32,
    fade: bool,
) -> f32 {
    let clip = vp * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return -1.0;
    }
    let ndc = clip.xyz / clip.w;
    if (any(ndc.xy < vec2<f32>(-inset)) || any(ndc.xy > vec2<f32>(inset)) ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return -1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let dims = vec2<f32>(textureDimensions(tex));
    var lit = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let texel = vec2<i32>(uv * dims) + vec2<i32>(dx, dy);
            let stored = textureLoad(tex, texel, 0);
            lit = lit + select(1.0, 0.0, stored > ndc.z + bias);
        }
    }
    lit = lit / 9.0;
    var edge_fade = 1.0;
    if (fade) {
        let edge = max(abs(ndc.x), abs(ndc.y));
        edge_fade = 1.0 - smoothstep(0.85, 1.0, edge);
    }
    return 1.0 - strength * (1.0 - lit) * edge_fade;
}

// Directional sun-shadow factor for receivers WITHOUT a usable surface normal
// (foliage, grass blades): walk the cascades near→far and use the tightest one
// that contains the point. `gate.x == 0` (inactive pass) early-outs to fully
// lit. Unrolled because WGSL can't index a list of texture bindings; the three
// maps are passed in near→far.
fn sun_shadow_factor(
    world_pos: vec3<f32>,
    block: ShadowCascadeBlock,
    tex0: texture_depth_2d,
    tex1: texture_depth_2d,
    tex2: texture_depth_2d,
) -> f32 {
    let s = block.gate.x;
    if (s <= 0.0) {
        return 1.0;
    }
    // No normal ⇒ assume a mid slope (1.0) and scale up for the missing offset.
    var f = cascade_factor(
        world_pos, block.view_proj[0],
        cascade_bias_m(block.params[0].y, 1.0) * NO_NORMAL_BIAS_SCALE * block.params[0].x,
        s, tex0, 0.98, false,
    );
    if (f < 0.0) {
        f = cascade_factor(
            world_pos, block.view_proj[1],
            cascade_bias_m(block.params[1].y, 1.0) * NO_NORMAL_BIAS_SCALE * block.params[1].x,
            s, tex1, 0.98, false,
        );
    }
    if (f < 0.0) {
        f = cascade_factor(
            world_pos, block.view_proj[2],
            cascade_bias_m(block.params[2].y, 1.0) * NO_NORMAL_BIAS_SCALE * block.params[2].x,
            s, tex2, 1.0, true,
        );
    }
    if (f < 0.0) {
        return 1.0;
    }
    return f;
}

// Directional sun-shadow factor with a receiver surface normal (stable-CSM
// path, W6). Per cascade the sample point is pushed off the surface along the
// normal by ~`NORMAL_OFFSET_TEXELS` of that cascade's texel size, and the depth
// bias is slope-scaled — both grow as the sun grazes the surface, which is
// exactly when a fixed bias either acnes or peter-pans. Pass the geometric /
// coarse surface normal, not a detail-mapped one (detail normals wobble the
// offset and re-introduce shimmer).
fn sun_shadow_factor_nrm(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    block: ShadowCascadeBlock,
    tex0: texture_depth_2d,
    tex1: texture_depth_2d,
    tex2: texture_depth_2d,
) -> f32 {
    let s = block.gate.x;
    if (s <= 0.0) {
        return 1.0;
    }
    let sun = block.sun_dir.xyz;
    let ndotl = clamp(dot(normal, sun), 0.0, 1.0);
    // tanθ of the surface's grazing angle to the sun, clamped.
    let slope = clamp(
        sqrt(max(1.0 - ndotl * ndotl, 0.0)) / max(ndotl, 0.2),
        0.0,
        MAX_SLOPE_SCALE,
    );
    let offset0 = min(block.params[0].y * NORMAL_OFFSET_TEXELS * (1.0 + slope), NORMAL_OFFSET_MAX_M);
    let offset1 = min(block.params[1].y * NORMAL_OFFSET_TEXELS * (1.0 + slope), NORMAL_OFFSET_MAX_M);
    let offset2 = min(block.params[2].y * NORMAL_OFFSET_TEXELS * (1.0 + slope), NORMAL_OFFSET_MAX_M);

    var f = cascade_factor(
        world_pos + normal * offset0,
        block.view_proj[0],
        cascade_bias_m(block.params[0].y, slope) * block.params[0].x,
        s, tex0, 0.98, false,
    );
    if (f < 0.0) {
        f = cascade_factor(
            world_pos + normal * offset1,
            block.view_proj[1],
            cascade_bias_m(block.params[1].y, slope) * block.params[1].x,
            s, tex1, 0.98, false,
        );
    }
    if (f < 0.0) {
        f = cascade_factor(
            world_pos + normal * offset2,
            block.view_proj[2],
            cascade_bias_m(block.params[2].y, slope) * block.params[2].x,
            s, tex2, 1.0, true,
        );
    }
    if (f < 0.0) {
        return 1.0;
    }
    return f;
}
