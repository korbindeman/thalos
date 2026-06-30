// Shared cascaded sun-shadow sampler.
//
// One copy of the cascade walk + depth compare, used by every ground-level
// surface material: the UDLOD terrain (`body_terrain.wgsl`), scattered trees
// (`tree.wgsl`), grass blades (`grass.wgsl`), and the preview ground patch
// (`ground_patch.wgsl`). All four bind the same per-cascade depth maps + the
// same `ShadowCascadeBlock` (published by the game's `rendering::sun_shadow`
// rig) and call `sun_shadow_factor` here — replacing the four near-verbatim
// copies that had started to drift (e.g. grass was a 1-tap variant).
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
    // per cascade: x = depth bias (clip), yzw reserved.
    params: array<vec4<f32>, 3>,
    // x = strength (0 ⇒ skip), y = active cascade count, zw reserved.
    gate: vec4<f32>,
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

// Directional sun-shadow factor: walk the cascades near→far and use the tightest
// one that contains the point (highest resolution). `gate.x == 0` (inactive
// pass) early-outs to fully lit. Unrolled because WGSL can't index a list of
// texture bindings; the three maps are passed in near→far.
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
    var f = cascade_factor(
        world_pos, block.view_proj[0], block.params[0].x, s, tex0, 0.98, false,
    );
    if (f < 0.0) {
        f = cascade_factor(
            world_pos, block.view_proj[1], block.params[1].x, s, tex1, 0.98, false,
        );
    }
    if (f < 0.0) {
        f = cascade_factor(
            world_pos, block.view_proj[2], block.params[2].x, s, tex2, 1.0, true,
        );
    }
    if (f < 0.0) {
        return 1.0;
    }
    return f;
}
