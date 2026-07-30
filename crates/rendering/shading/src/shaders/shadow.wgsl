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
// - `sun_shadow_factor(pos, ...)` — for FRAGMENT-stage receivers with no
//   meaningful surface normal (tree foliage, rocks, ground patch). Same walk,
//   but a larger constant depth bias (`NO_NORMAL_BIAS_SCALE`) stands in for
//   the missing offset.
// - `sun_shadow_factor_vert(pos, ...)` — the PER-VERTEX variant of the above
//   (grass / GPU grass compute their shadow at the blade vertex and
//   interpolate). Identical contract, but keeps the cheap point-sampled 3×3
//   kernel instead of the filtered tent — interpolation across the blade
//   already smooths it, and it runs at grass vertex counts.
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

// ── PCSS (contact-hardening penumbra) ────────────────────────────────────────
//
// The tent PCF above has ONE fixed ~3-texel penumbra, which is exactly what
// makes shadows read as engine output: a real sun is an extended source, so a
// shadow is razor-sharp where the caster touches the ground and widens
// linearly with caster→receiver distance (a building tip's shadow is metres
// soft while its base is crisp). Classic PCSS restores that in three steps per
// fragment: (1) a sparse BLOCKER SEARCH estimates the average light-space
// depth of whatever occludes this point; (2) the penumbra width follows from
// the sun's angular size × the occluder distance; (3) a Vogel-disk PCF with
// that radius filters the visibility. Fragments the search finds unoccluded
// early-out fully lit, and radii inside the tent's own footprint fall back to
// the tent — so fully-lit and contact regions cost roughly what they did.
//
// Only the `_nrm` fragment path (terrain, structures, hull) runs PCSS; foliage
// keeps the fixed tent (leaf shadows are chaotic at texel scale — a blocker
// search there buys noise, not realism) and the per-vertex grass path keeps
// its cheap point kernel.

// tan of the sun's angular RADIUS (~0.266°): metres of penumbra radius per
// metre of caster→receiver light-space distance. The old value used the full
// diameter as a radius and made every separated edge about 2× too soft.
const SUN_TAN_ANGULAR_RADIUS: f32 = 0.00465;
// Blocker-search radius, in texels of the sampled cascade. Bounds the largest
// occluder distance whose penumbra the search can notice; beyond it the
// filter simply stays at its widest.
const PCSS_SEARCH_TEXELS: f32 = 6.0;
// Cap on the variable PCF radius (texels). Keeps the widest penumbra's taps
// dense enough that 16 samples do not band.
const PCSS_MAX_FILTER_TEXELS: f32 = 6.0;
const PCSS_TAPS: i32 = 16;

// PCSS variant of [`cascade_factor`]. `clip_per_m` converts stored-depth
// deltas back to light-space metres (`params.x`); `texel_m` is the cascade's
// texel size in metres (`params.y`).
fn cascade_factor_pcss(
    world_pos: vec3<f32>,
    vp: mat4x4<f32>,
    bias: f32,
    strength: f32,
    tex: texture_depth_2d,
    inset: f32,
    fade: bool,
    clip_per_m: f32,
    texel_m: f32,
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
    let dims_i = vec2<i32>(textureDimensions(tex));
    let dims = vec2<f32>(dims_i);
    let map_pos = uv * dims - vec2<f32>(0.5);

    // (1) Blocker search: 16 sparse taps over ±PCSS_SEARCH_TEXELS, plus one
    // guaranteed CENTER tap — the sparse ring's 4-texel stride can hop clean
    // over a thin caster (a pole's 2-texel shadow), and missing the blocker
    // here would punch a lit hole through that caster's own shadow. Average
    // the light-space depth delta of the taps that occlude this fragment.
    var occ_sum_m = 0.0;
    var occ_n = 0.0;
    let center = clamp(vec2<i32>(map_pos + vec2<f32>(0.5)), vec2<i32>(0), dims_i - 1);
    let center_stored = textureLoad(tex, center, 0);
    if (center_stored > ndc.z + bias) {
        occ_sum_m = (center_stored - ndc.z) / max(clip_per_m, 1.0e-9);
        occ_n = 1.0;
    }
    for (var j = 0; j < 4; j = j + 1) {
        for (var i = 0; i < 4; i = i + 1) {
            let off = (vec2<f32>(f32(i), f32(j)) - vec2<f32>(1.5)) * (PCSS_SEARCH_TEXELS / 1.5);
            let texel = clamp(vec2<i32>(map_pos + off), vec2<i32>(0), dims_i - 1);
            let stored = textureLoad(tex, texel, 0);
            if (stored > ndc.z + bias) {
                occ_sum_m = occ_sum_m + (stored - ndc.z) / max(clip_per_m, 1.0e-9);
                occ_n = occ_n + 1.0;
            }
        }
    }
    if (occ_n == 0.0) {
        // Nothing occludes anywhere in reach — fully lit, skip the filter.
        return 1.0;
    }

    // (2) Penumbra radius from the average occluder distance, in texels.
    let avg_occluder_m = occ_sum_m / occ_n;
    let radius_tx = clamp(
        SUN_TAN_ANGULAR_RADIUS * avg_occluder_m / max(texel_m, 1.0e-6),
        0.0,
        PCSS_MAX_FILTER_TEXELS,
    );

    var lit = 0.0;
    if (radius_tx <= 1.2) {
        // Contact range: a bilinear 2×2 comparison footprint. The previous
        // 4×4 tent spread a nominally sharp contact edge across ~3 texels.
        let base = vec2<i32>(floor(map_pos));
        let f = map_pos - floor(map_pos);
        var weights = vec4<f32>(
            (1.0 - f.x) * (1.0 - f.y),
            f.x * (1.0 - f.y),
            (1.0 - f.x) * f.y,
            f.x * f.y,
        );
        for (var j = 0; j < 2; j = j + 1) {
            for (var i = 0; i < 2; i = i + 1) {
                let texel = clamp(base + vec2<i32>(i, j), vec2<i32>(0), dims_i - 1);
                let stored = textureLoad(tex, texel, 0);
                lit = lit + weights[j * 2 + i] * select(1.0, 0.0, stored > ndc.z + bias);
            }
        }
    } else {
        // (3) Fixed-orientation Vogel disk. A per-fragment hash made the
        // sparse pattern crawl as a moving edge crossed shadow-map texels; in
        // this non-TAA renderer a deterministic kernel is the stable choice.
        for (var t = 0; t < PCSS_TAPS; t = t + 1) {
            let r = radius_tx * sqrt((f32(t) + 0.5) / f32(PCSS_TAPS));
            let a = f32(t) * 2.39996322972865332;
            let off = vec2<f32>(cos(a), sin(a)) * r;
            let texel = clamp(vec2<i32>(map_pos + off + vec2<f32>(0.5)), vec2<i32>(0), dims_i - 1);
            let stored = textureLoad(tex, texel, 0);
            lit = lit + select(1.0, 0.0, stored > ndc.z + bias);
        }
        lit = lit / f32(PCSS_TAPS);
    }

    var edge_fade = 1.0;
    if (fade) {
        let edge = max(abs(ndc.x), abs(ndc.y));
        edge_fade = 1.0 - smoothstep(0.85, 1.0, edge);
    }
    return 1.0 - strength * (1.0 - lit) * edge_fade;
}

// Normalized distance to the square cascade edge, or a negative sentinel when
// the point is outside. Inner cascades remain valid all the way to 1.0 so the
// receiver can cross-fade to the next map instead of switching maps in one
// fragment.
fn cascade_edge(world_pos: vec3<f32>, vp: mat4x4<f32>) -> f32 {
    let clip = vp * vec4<f32>(world_pos, 1.0);
    if clip.w <= 0.0 {
        return -1.0;
    }
    let ndc = clip.xyz / clip.w;
    if ndc.z < 0.0 || ndc.z > 1.0 {
        return -1.0;
    }
    let edge = max(abs(ndc.x), abs(ndc.y));
    return select(edge, -1.0, edge > 1.0);
}

const CASCADE_BLEND_START: f32 = 0.90;
const CASCADE_BLEND_END: f32 = 0.99;

fn cascade_blend(inner: f32, inner_edge: f32, outer: f32) -> f32 {
    if inner < 0.0 {
        return outer;
    }
    if outer < 0.0 {
        return inner;
    }
    return mix(
        inner,
        outer,
        smoothstep(CASCADE_BLEND_START, CASCADE_BLEND_END, inner_edge),
    );
}

// ── Contact tier (W18a) ───────────────────────────────────────────────────────
//
// The cascades above are the MID-FIELD tier of the three-tier shadow split
// (ADR-20260722T111848Z-shadows-three-tier-not-virtual-shadow-maps). They cannot
// serve the 0–50 m contact band — cascade 0 is ~0.2 m/texel, coarser than a
// landing-gear strut — so `rendering::contact_shadow` marches the copied scene
// depth toward the sun and publishes a full-res factor that receivers multiply
// into their DIRECT sun gate (never the ambient: ambient contact occlusion is
// F5's SSAO, and applying both to ambient double-darkens).
//
// The gate rides in `block.gate.z` (0 = skip, 1 = apply, 2 = the caller paints
// the raw value as a diagnostic), so a material that already binds the cascade
// block needs no extra uniform — only the texture pair.
fn contact_shadow_factor(
    block: ShadowCascadeBlock,
    tex: texture_2d<f32>,
    samp: sampler,
    frag_coord: vec2<f32>,
    viewport: vec2<f32>,
) -> f32 {
    if (block.gate.z < 0.5) {
        return 1.0;
    }
    // `frag_coord` is the framebuffer pixel coordinate; the full-res target spans
    // the viewport exactly, so this is a direct [0,1] screen UV. (Deriving the
    // viewport from `textureDimensions` instead mis-registers by a texel on odd
    // sizes — the banding source `screen_space_ao` documents.)
    let uv = frag_coord / max(viewport, vec2<f32>(1.0));
    return clamp(textureSampleLevel(tex, samp, uv, 0.0).r, 0.0, 1.0);
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
// softens the outermost cascade (nothing covers beyond it).
//
// FILTERED PCF: a separable-tent kernel over a 4×4 texel neighbourhood, exactly
// equivalent to averaging 3×3 *hardware-bilinear* comparison taps at texel-
// spaced offsets (per-axis texel weights `[1−f, 1, 1, f] / 3` around the
// fractional sample position `f`). The old point-sampled 3×3 flipped a whole
// texel per compare — every shadow edge was a hard staircase at shadow-map
// texel size, and any sub-texel edge motion (sun stepping, warp) popped a full
// texel at once. The tent turns edges into smooth ~3-texel gradients with the
// same `textureLoad` machinery (no comparison-sampler binding needed in any
// material). Fragment-stage receivers use this; per-vertex receivers (grass)
// keep the cheap point kernel below — interpolation across the blade already
// smooths them, and they run per-vertex at grass counts.
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
    // Texel-space position with the half-texel centre offset; `base` is the
    // upper-left of the centre 2×2, `f` the fractional position inside it.
    let pos = uv * dims - vec2<f32>(0.5);
    let base = vec2<i32>(floor(pos));
    let f = pos - floor(pos);
    // Separable tent weights over the 4 texel columns/rows [base-1 .. base+2].
    // `var` (not `let`) so the loop can index them dynamically (naga restricts
    // dynamic indexing of let-bound composites).
    var wx = vec4<f32>(1.0 - f.x, 1.0, 1.0, f.x) / 3.0;
    var wy = vec4<f32>(1.0 - f.y, 1.0, 1.0, f.y) / 3.0;
    var lit = 0.0;
    for (var j = 0; j < 4; j = j + 1) {
        for (var i = 0; i < 4; i = i + 1) {
            let texel = base + vec2<i32>(i - 1, j - 1);
            let stored = textureLoad(tex, texel, 0);
            lit = lit + wx[i] * wy[j] * select(1.0, 0.0, stored > ndc.z + bias);
        }
    }
    var edge_fade = 1.0;
    if (fade) {
        let edge = max(abs(ndc.x), abs(ndc.y));
        edge_fade = 1.0 - smoothstep(0.85, 1.0, edge);
    }
    return 1.0 - strength * (1.0 - lit) * edge_fade;
}

// Point-sampled 3×3 variant of [`cascade_factor`] — the pre-filtering kernel,
// kept for PER-VERTEX receivers (grass / GPU grass), where the result is
// interpolated across the blade anyway and the 16-load tent would nearly
// double the heaviest vertex workload in the game. Same walk, same bias
// contract.
fn cascade_factor_point(
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

// Directional sun-shadow factor for FRAGMENT receivers WITHOUT a usable
// surface normal (tree foliage, rocks): walk the cascades near→far and use the
// tightest one that contains the point. `gate.x == 0` (inactive pass)
// early-outs to fully lit. Unrolled because WGSL can't index a list of texture
// bindings; the three maps are passed in near→far.
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
    let f0 = cascade_factor(
        world_pos, block.view_proj[0],
        cascade_bias_m(block.params[0].y, 1.0) * NO_NORMAL_BIAS_SCALE * block.params[0].x,
        s, tex0, 1.0, false,
    );
    let e0 = cascade_edge(world_pos, block.view_proj[0]);
    if f0 >= 0.0 && e0 < CASCADE_BLEND_START {
        return f0;
    }
    let f1 = cascade_factor(
            world_pos, block.view_proj[1],
            cascade_bias_m(block.params[1].y, 1.0) * NO_NORMAL_BIAS_SCALE * block.params[1].x,
            s, tex1, 1.0, false,
    );
    if f0 >= 0.0 {
        return cascade_blend(f0, e0, f1);
    }
    let e1 = cascade_edge(world_pos, block.view_proj[1]);
    if f1 >= 0.0 && e1 < CASCADE_BLEND_START {
        return f1;
    }
    let f2 = cascade_factor(
            world_pos, block.view_proj[2],
            cascade_bias_m(block.params[2].y, 1.0) * NO_NORMAL_BIAS_SCALE * block.params[2].x,
            s, tex2, 1.0, true,
    );
    let f = cascade_blend(f1, e1, f2);
    if f < 0.0 {
        return 1.0;
    }
    return f;
}

// PER-VERTEX variant of [`sun_shadow_factor`] (see the entry-point notes at the
// top): same cascade walk and bias contract, point-sampled 3×3 kernel. Used by
// the grass paths, which evaluate at blade vertices and interpolate.
fn sun_shadow_factor_vert(
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
    let f0 = cascade_factor_point(
        world_pos, block.view_proj[0],
        cascade_bias_m(block.params[0].y, 1.0) * NO_NORMAL_BIAS_SCALE * block.params[0].x,
        s, tex0, 1.0, false,
    );
    let e0 = cascade_edge(world_pos, block.view_proj[0]);
    if f0 >= 0.0 && e0 < CASCADE_BLEND_START {
        return f0;
    }
    let f1 = cascade_factor_point(
            world_pos, block.view_proj[1],
            cascade_bias_m(block.params[1].y, 1.0) * NO_NORMAL_BIAS_SCALE * block.params[1].x,
            s, tex1, 1.0, false,
    );
    if f0 >= 0.0 {
        return cascade_blend(f0, e0, f1);
    }
    let e1 = cascade_edge(world_pos, block.view_proj[1]);
    if f1 >= 0.0 && e1 < CASCADE_BLEND_START {
        return f1;
    }
    let f2 = cascade_factor_point(
            world_pos, block.view_proj[2],
            cascade_bias_m(block.params[2].y, 1.0) * NO_NORMAL_BIAS_SCALE * block.params[2].x,
            s, tex2, 1.0, true,
    );
    let f = cascade_blend(f1, e1, f2);
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

    let pos0 = world_pos + normal * offset0;
    let f0 = cascade_factor_pcss(
        pos0,
        block.view_proj[0],
        cascade_bias_m(block.params[0].y, slope) * block.params[0].x,
        s, tex0, 1.0, false,
        block.params[0].x, block.params[0].y,
    );
    let e0 = cascade_edge(pos0, block.view_proj[0]);
    if f0 >= 0.0 && e0 < CASCADE_BLEND_START {
        return f0;
    }
    let pos1 = world_pos + normal * offset1;
    let f1 = cascade_factor_pcss(
            pos1,
            block.view_proj[1],
            cascade_bias_m(block.params[1].y, slope) * block.params[1].x,
            s, tex1, 1.0, false,
            block.params[1].x, block.params[1].y,
    );
    if f0 >= 0.0 {
        return cascade_blend(f0, e0, f1);
    }
    let e1 = cascade_edge(pos1, block.view_proj[1]);
    if f1 >= 0.0 && e1 < CASCADE_BLEND_START {
        return f1;
    }
    let pos2 = world_pos + normal * offset2;
    let f2 = cascade_factor_pcss(
            pos2,
            block.view_proj[2],
            cascade_bias_m(block.params[2].y, slope) * block.params[2].x,
            s, tex2, 1.0, true,
            block.params[2].x, block.params[2].y,
    );
    let f = cascade_blend(f1, e1, f2);
    if (f < 0.0) {
        return 1.0;
    }
    return f;
}
