//! Procedural texture generation for Thalos — pure Rust, no Bevy.
//!
//! Produces [`TextureData`] (raw sRGBA8) on the CPU, deterministically. The same
//! generators feed three consumers:
//! - the **runtime** (`thalos_body_render` wraps the output in a GPU `Image`),
//! - the **object preview** (`just preview`), and
//! - an offline **bake** (`cargo run -p thalos_texgen --example bake`) that writes
//!   PNGs — to inspect, or to prebake for the game.
//!
//! Today it generates the **foliage atlas** (leaf clusters + conifer needles +
//! bark) the tree meshes sample, plus a companion **foliage material atlas**
//! (bark normal + roughness). Rocks and other procedural textures will live here
//! too.
//!
//! ## Foliage atlas layout
//! An `ATLAS_N × ATLAS_N` grid of `CELL_PX` cells. A tree vertex carries a packed
//! `cell·4 + corner` code ([`leaf_code`]) the shader decodes to the cell's UVs:
//! leaf cards sample a leaf-cluster cell, conifer boughs the needle cell, trunks a
//! bark cell. **Leaf cells carry real colour** (a green palette, lighter toward
//! the top / outer rim) so foliage gets natural variation; the mesh's per-species
//! tint is just a light nudge on top.

/// Raw texture: tightly-packed sRGBA8, row-major, `width * height * 4` bytes.
#[derive(Clone)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    /// sRGB colour + straight (linear-coverage) alpha, 4 bytes per texel.
    pub rgba: Vec<u8>,
}

// ── Atlas layout (the single source of truth; the shader + mesh mirror it) ──────
/// Cells per atlas axis.
pub const ATLAS_N: u32 = 4;
/// Pixels per cell. `ATLAS_N * CELL_PX` is the atlas side.
pub const CELL_PX: u32 = 256;
const SIZE: usize = (ATLAS_N * CELL_PX) as usize;

/// First leaf-cluster cell and how many variants (broadleaf clumps).
pub const LEAF_CELL_FIRST: u32 = 0;
pub const LEAF_CELL_COUNT: u32 = 11;
/// Conifer needle-spray cell.
pub const NEEDLE_CELL: u32 = 11;
/// Opaque flat-green cell (unused by the current mesh; kept for layout stability).
pub const SHELL_CELL: u32 = 12;
/// First bark cell and count (trunks).
pub const BARK_CELL_FIRST: u32 = 13;
pub const BARK_CELL_COUNT: u32 = 3;

const TAU: f32 = std::f32::consts::TAU;
const GOLD: u64 = 0x9E37_79B9_7F4A_7C15;

/// Pack a `(cell, corner)` pair into the float a tree vertex stores in `UV_1.y`.
/// `corner ∈ 0..4` selects the cell's UV corner (0=BL,1=BR,2=TR,3=TL).
#[inline]
pub fn leaf_code(cell: u32, corner: u32) -> f32 {
    (cell * 4 + (corner & 3)) as f32
}

/// Generate the foliage atlas (sRGBA8). Deterministic.
pub fn foliage_atlas() -> TextureData {
    let mut px = vec![[0.0f32; 4]; SIZE * SIZE]; // straight-alpha linear scratch
    let cell = CELL_PX as usize;

    for c in 0..(ATLAS_N * ATLAS_N) {
        let ox = (c % ATLAS_N) as usize * cell;
        let oy = (c / ATLAS_N) as usize * cell;
        match c {
            c if (LEAF_CELL_FIRST..LEAF_CELL_FIRST + LEAF_CELL_COUNT).contains(&c) => {
                draw_leaf_cluster(&mut px, ox, oy, c as u64 * 1009 + 7);
            }
            c if c == NEEDLE_CELL => draw_needle_spray(&mut px, ox, oy, 4242),
            c if c == SHELL_CELL => fill_shell(&mut px, ox, oy),
            _ => draw_bark(&mut px, ox, oy, c as u64 * 733 + 3),
        }
    }

    pack(&px)
}

/// Generate the **foliage material atlas** (normal + roughness), mirroring the
/// [`foliage_atlas`] layout cell-for-cell. **Linear** data (NOT sRGB): RGB is a
/// tangent-space normal (`xyz·0.5+0.5`), A is perceptual roughness. Only the bark
/// cells carry real detail — derived from the *same* [`bark_height`] field the
/// albedo uses, so cracks/ridges line up across albedo, normal, and roughness.
/// Every other cell is left flat (normal `(0,0,1)`, mid roughness): leaves and
/// needles are matte and never sample this map. Bind alongside [`foliage_atlas`]
/// on the tree material as a `TextureFormat::Rgba8Unorm` image.
pub fn foliage_material_atlas() -> TextureData {
    // Flat default: normal (0,0,1) encoded, roughness 0.9.
    let mut px = vec![[0.5f32, 0.5, 1.0, 0.9]; SIZE * SIZE];
    let cell = CELL_PX as usize;

    for c in 0..(ATLAS_N * ATLAS_N) {
        let ox = (c % ATLAS_N) as usize * cell;
        let oy = (c / ATLAS_N) as usize * cell;
        if (BARK_CELL_FIRST..BARK_CELL_FIRST + BARK_CELL_COUNT).contains(&c) {
            // Same per-cell seed as the albedo bark (see `foliage_atlas`).
            draw_bark_material(&mut px, ox, oy, c as u64 * 733 + 3);
        } else if (LEAF_CELL_FIRST..LEAF_CELL_FIRST + LEAF_CELL_COUNT).contains(&c) {
            // Leaf cells carry a real normal/roughness now (not flat): a per-leaf
            // height→normal so individual leaves catch light differently → depth.
            // SAME per-cell seed as the albedo leaf cluster, so they line up.
            draw_leaf_cluster_material(&mut px, ox, oy, c as u64 * 1009 + 7);
        }
    }

    pack(&px)
}

/// Pack a straight-alpha linear scratch buffer into a [`TextureData`].
fn pack(px: &[[f32; 4]]) -> TextureData {
    let mut rgba = vec![0u8; SIZE * SIZE * 4];
    for (i, p) in px.iter().enumerate() {
        rgba[i * 4] = to_u8(p[0]);
        rgba[i * 4 + 1] = to_u8(p[1]);
        rgba[i * 4 + 2] = to_u8(p[2]);
        rgba[i * 4 + 3] = to_u8(p[3]);
    }
    TextureData {
        width: SIZE as u32,
        height: SIZE as u32,
        rgba,
    }
}

// ── Cell rasterizers ───────────────────────────────────────────────────────────

/// Leaf normal map: each leaf is a near-flat blade tilted a little in a random
/// direction so neighbours catch light differently and read as separate leaves —
/// NOT embossed midrib relief. `LEAF_TILT_*` is the tangent-space tilt (xy) amount
/// (kept small/gentle); roughness is fixed (waxy, a touch glossier than bark).
const LEAF_TILT_MIN: f32 = 0.10;
const LEAF_TILT_VAR: f32 = 0.16;
const LEAF_ROUGHNESS: f32 = 0.6;

/// One leaf to stamp: centre, orientation, size, palette `tone`, and stack
/// `layer` (0 = bottom of the clump .. 1 = top). Shared by the albedo cell and
/// the height→normal cell so they line up exactly.
struct LeafStamp {
    s: u64,
    cx: f32,
    cy: f32,
    ang: f32,
    len: f32,
    wid: f32,
    tone: f32,
    layer: f32,
}

/// Emit every leaf of one fluffy cluster, deterministic from `seed` — the SINGLE
/// source of leaf placement, consumed by both [`draw_leaf_cluster`] (albedo) and
/// [`draw_leaf_cluster_material`] (normal/roughness). A dense body, a sparse
/// outward fringe + stragglers, and a finer highlight pass, over an irregular
/// lobed outline with per-cell aspect / lean / lobe-count (so the 11 cells are
/// all different shapes, not the same disc).
fn each_leaf(seed: u64, mut emit: impl FnMut(LeafStamp)) {
    let w = CELL_PX as f32;
    let half = w * 0.5;
    // The clump FILLS the cell (base ≈0.80 of the cell radius). A thin
    // transparent margin is reserved (the `place` clamp) so the ragged rim never
    // hard-cuts at the cell border — the "blocky billboard" tell. Airiness comes
    // from the rim DENSITY tapering off, not from shrinking the body.
    let base = half * 0.80;
    let margin = half - 8.0;
    let asx = 0.82 + 0.40 * hash01(seed, 20);
    let asy = 0.82 + 0.40 * hash01(seed, 21);
    let lean = (hash01(seed, 22) - 0.5) * 0.55;
    let lobes = 3.0 + (3.0 * hash01(seed, 23)).floor(); // 3..5 integer lobes (wraps)
    let lobe_phase = hash01(seed, 24) * TAU;
    let edge_at = |a: f32| -> f32 {
        let nz = value_noise(a.cos() * 1.8 + 4.0, a.sin() * 1.8 + 4.0, seed ^ 0x77);
        let lobe = 0.5 + 0.5 * (a * lobes + lobe_phase).sin();
        (0.56 + 0.26 * nz + 0.20 * lobe).clamp(0.40, 1.12)
    };
    let place = |a: f32, rr: f32| -> (f32, f32) {
        let rad = (rr * edge_at(a) * base).min(margin);
        let (ca, sa) = (a.cos(), a.sin());
        (half + (ca * asx + sa * lean) * rad, half + sa * asy * rad)
    };

    // Body: a full clump that FILLS the cell, SOLID through the middle and
    // thinning toward the rim (radius ∝ u^0.70 → centre-dense, edge-sparse), so
    // the interior covers opaquely (branches never show through a stack of these
    // cards) yet the boundary still fades from solid mass into scattered leaves
    // instead of ending at a hard disc. Small leaves so each reads individually.
    let n_body = 290u32;
    for li in 0..n_body {
        let s = seed.wrapping_add((li as u64).wrapping_mul(GOLD));
        let a = hash01(s, 2) * TAU;
        let (cx, cy) = place(a, hash01(s, 1).powf(0.70));
        let len = w * (0.052 + 0.040 * hash01(s, 4));
        let wid = len * (0.40 + 0.18 * hash01(s, 5));
        let topness = (1.0 - cy / w).clamp(0.0, 1.0);
        emit(LeafStamp {
            s,
            cx,
            cy,
            ang: hash01(s, 3) * TAU,
            len,
            wid,
            tone: (0.16 + 0.40 * hash01(s, 6) + 0.22 * topness).clamp(0.0, 1.0),
            layer: 0.42 * (li as f32 / n_body as f32),
        });
    }

    // Detached rim: small leaves out at / just past the body's ragged outline
    // (rr 0.96..1.18, following the lobed `edge_at`), sparse and pointing
    // outward so they sit APART as individual leaves — the broken, irregular
    // silhouette of a real canopy rather than a clean clump edge.
    let n_rim = 64u32;
    for li in 0..n_rim {
        let s = seed.wrapping_add((li as u64 + 700).wrapping_mul(GOLD));
        let a = hash01(s, 2) * TAU;
        let (cx, cy) = place(a, 0.96 + 0.22 * hash01(s, 1));
        let outward = (cy - half).atan2(cx - half);
        let len = w * (0.040 + 0.032 * hash01(s, 4));
        let wid = len * (0.36 + 0.16 * hash01(s, 5));
        let topness = (1.0 - cy / w).clamp(0.0, 1.0);
        emit(LeafStamp {
            s,
            cx,
            cy,
            ang: outward + (hash01(s, 3) - 0.5) * 1.2,
            len,
            wid,
            tone: (0.30 + 0.40 * hash01(s, 6) + 0.20 * topness).clamp(0.0, 1.0),
            layer: 0.45 + 0.25 * (li as f32 / n_rim as f32),
        });
    }

    // Highlight: a few finer, brighter leaves over the body top for dapple.
    let n_hi = 34u32;
    for li in 0..n_hi {
        let s = seed.wrapping_add((li as u64 + 1500).wrapping_mul(GOLD));
        let a = hash01(s, 2) * TAU;
        let rr = hash01(s, 1).powf(0.8);
        let (cx, cy) = place(a, rr);
        let len = w * (0.036 + 0.028 * hash01(s, 4));
        let wid = len * (0.42 + 0.20 * hash01(s, 5));
        let topness = (1.0 - cy / w).clamp(0.0, 1.0);
        emit(LeafStamp {
            s,
            cx,
            cy,
            ang: hash01(s, 3) * TAU,
            len,
            wid,
            tone: (0.46 + 0.34 * hash01(s, 6) + 0.18 * topness + 0.12 * rr).clamp(0.0, 1.0),
            layer: 0.62 + 0.38 * (li as f32 / n_hi as f32),
        });
    }
}

/// Albedo leaf-cluster cell: stamp every leaf with its palette colour. Shape +
/// placement come from [`each_leaf`]; see it for the fluffy-silhouette design.
fn draw_leaf_cluster(px: &mut [[f32; 4]], ox: usize, oy: usize, seed: u64) {
    each_leaf(seed, |l| {
        stamp_leaf(px, ox, oy, l.cx, l.cy, l.ang, l.len, l.wid, leaf_palette(l.tone, l.s));
    });
}

/// Material leaf-cluster cell: a per-leaf **normal + roughness**, so neighbouring
/// leaves catch light slightly differently and read as separate leaves (gentle —
/// no embossed relief). Each leaf is a near-flat blade tilted a little; see
/// `stamp_leaf_normal`.
fn draw_leaf_cluster_material(mat: &mut [[f32; 4]], ox: usize, oy: usize, seed: u64) {
    each_leaf(seed, |l| stamp_leaf_normal(mat, ox, oy, l));
}

/// Leaf colour from a green palette (deep → mid → yellow-green highlight) by
/// `t ∈ [0,1]`, with a small per-leaf hue jitter.
fn leaf_palette(t: f32, seed: u64) -> [f32; 3] {
    const DEEP: [f32; 3] = [0.13, 0.28, 0.13];
    const MID: [f32; 3] = [0.29, 0.49, 0.19];
    const HI: [f32; 3] = [0.57, 0.69, 0.31];
    let c = if t < 0.5 {
        lerp3(DEEP, MID, t * 2.0)
    } else {
        lerp3(MID, HI, (t - 0.5) * 2.0)
    };
    let j = (hash01(seed, 8) - 0.5) * 0.07;
    [
        (c[0] + j).clamp(0.0, 1.0),
        c[1],
        (c[2] - j * 0.5).clamp(0.0, 1.0),
    ]
}

/// A flat conifer bough: a faint stem with fine blue-green needles fanning down
/// and out, tapering to a tip — an elongated, mostly-transparent spray.
fn draw_needle_spray(px: &mut [[f32; 4]], ox: usize, oy: usize, seed: u64) {
    let w = CELL_PX as f32;
    let stem_x = w * 0.5;
    let top = w * 0.10;
    let bot = w * 0.94;

    for i in 0..96u32 {
        let t = i as f32 / 96.0;
        let y = top + (bot - top) * t;
        stamp_leaf(px, ox, oy, stem_x, y, 0.0, w * 0.026, w * 0.008, [0.16, 0.24, 0.15]);
    }
    for li in 0..360u32 {
        let s = seed.wrapping_add((li as u64).wrapping_mul(GOLD));
        let t = hash01(s, 0);
        let y = top + (bot - top) * t;
        let taper = (1.0 - t).max(0.05);
        let side = if li & 1 == 0 { -1.0 } else { 1.0 };
        let len = w * (0.08 + 0.13 * taper) * (0.7 + 0.5 * hash01(s, 2));
        let wid = w * 0.012;
        let out = side * (0.5 + 0.4 * hash01(s, 3));
        let down = 0.7 + 0.3 * hash01(s, 4);
        let inv = 1.0 / (out * out + down * down).sqrt();
        let (dx, dy) = (out * inv, down * inv);
        let cx = stem_x + dx * len * 0.5;
        let cy = y + dy * len * 0.5;
        let ang = dy.atan2(dx);
        let nt = (0.25 + 0.5 * hash01(s, 5)).clamp(0.0, 1.0);
        let col = lerp3([0.10, 0.24, 0.17], [0.26, 0.44, 0.24], nt);
        stamp_leaf(px, ox, oy, cx, cy, ang, len, wid, col);
    }
}

/// Iterate the covered texels of one rotated leaf (a tapered ellipse), yielding
/// `(lx, ly, alpha, u, v)` — `u` along the length, `v` across the width — with a
/// crisp anti-aliased edge. Shared by the albedo stamp and the height bake so
/// both see identical leaf coverage.
fn leaf_footprint(
    cx: f32,
    cy: f32,
    ang: f32,
    len: f32,
    wid: f32,
    mut f: impl FnMut(i32, i32, f32, f32, f32),
) {
    let cell = CELL_PX as i32;
    let (sa, ca) = ang.sin_cos();
    let reach = len.max(wid).ceil() as i32 + 1;
    let (x0, y0) = (cx as i32, cy as i32);
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let (lx, ly) = (x0 + dx, y0 + dy);
            if lx < 0 || ly < 0 || lx >= cell || ly >= cell {
                continue;
            }
            let fx = lx as f32 + 0.5 - cx;
            let fy = ly as f32 + 0.5 - cy;
            let u = (fx * ca + fy * sa) / len;
            let v = (-fx * sa + fy * ca) / wid;
            let taper = 1.0 - 0.55 * u.abs();
            let d = (u * u + (v / taper.max(0.2)).powi(2)).sqrt();
            if d >= 1.0 {
                continue;
            }
            let alpha = smooth01((1.0 - d) / 0.24).clamp(0.0, 1.0);
            if alpha > 0.003 {
                f(lx, ly, alpha, u, v);
            }
        }
    }
}

/// Composite one leaf into the albedo as a painterly blade with a gentle
/// lengthwise gradient — no centre-line midrib stripe. The tone shades softly
/// along the leaf's length (`u`: darker toward one end, lighter toward the
/// other), giving a hand-painted falloff instead of a flat sheet. Each leaf also
/// carries its own palette tone (see [`leaf_palette`]), so a cluster reads as
/// varied foliage from the leaf-to-leaf colour spread on top of this gradient.
#[allow(clippy::too_many_arguments)]
fn stamp_leaf(
    px: &mut [[f32; 4]],
    ox: usize,
    oy: usize,
    cx: f32,
    cy: f32,
    ang: f32,
    len: f32,
    wid: f32,
    color: [f32; 3],
) {
    leaf_footprint(cx, cy, ang, len, wid, |lx, ly, alpha, u, _v| {
        // Soft ±12% lengthwise shade — `u` runs −1..1 along the blade.
        let grad = 1.0 + 0.12 * u;
        let src = [color[0] * grad, color[1] * grad, color[2] * grad, alpha];
        let idx = (oy + ly as usize) * SIZE + (ox + lx as usize);
        over(&mut px[idx], src);
    });
}

/// Bake one leaf's tangent-space normal: a near-flat blade tilted a little in a
/// per-leaf random direction, blended in by coverage so the top leaf wins and the
/// edges stay soft. Neighbouring leaves face slightly differently, so they read as
/// SEPARATE leaves catching light — without the embossed midrib relief the old
/// height ridge produced.
fn stamp_leaf_normal(mat: &mut [[f32; 4]], ox: usize, oy: usize, l: LeafStamp) {
    let ta = hash01(l.s, 30) * TAU;
    // Small per-leaf tilt; top leaves (higher `layer`) tilt a touch more.
    let tmag = (LEAF_TILT_MIN + LEAF_TILT_VAR * hash01(l.s, 31)) * (0.7 + 0.3 * l.layer);
    let nx = tmag * ta.cos();
    let ny = tmag * ta.sin();
    let nz = (1.0 - (nx * nx + ny * ny)).max(0.25).sqrt();
    let (ex, ey, ez) = (nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5);
    leaf_footprint(l.cx, l.cy, l.ang, l.len, l.wid, |lx, ly, alpha, _u, _v| {
        // Blend this leaf's flat normal in by coverage (top leaf wins; soft edges).
        let idx = (oy + ly as usize) * SIZE + (ox + lx as usize);
        let cur = mat[idx];
        mat[idx] = [
            cur[0] * (1.0 - alpha) + ex * alpha,
            cur[1] * (1.0 - alpha) + ey * alpha,
            cur[2] * (1.0 - alpha) + ez * alpha,
            LEAF_ROUGHNESS,
        ];
    });
}

/// Opaque flat-green cell (kept for layout stability; the current mesh is core-less).
fn fill_shell(px: &mut [[f32; 4]], ox: usize, oy: usize) {
    let cell = CELL_PX as usize;
    for y in 0..cell {
        for x in 0..cell {
            let n = 0.30 + 0.10 * value_noise(x as f32 * 0.05, y as f32 * 0.05, 11);
            let idx = (oy + y) * SIZE + (ox + x);
            px[idx] = [n * 0.5, n, n * 0.4, 1.0];
        }
    }
}

// ── Bark ─────────────────────────────────────────────────────────────────────
// Smooth-stem tree bark, authored as a coherent little material: a single height
// field (`bark_height` — fine continuous vertical grain lines, no furrows) drives
// BOTH the albedo (one consistent brown modulated in value) and the companion
// normal/roughness map (`draw_bark_material`), so the grain lines up across every
// channel. The field is **toroidally periodic** (over CELL_PX in both axes), so
// the mesh wrap-tiles ONE cell around the stem and up it seamlessly (see
// `push_trunk`) — no mirror, no seam, no dark-blob landmark to track the repeat.
// Colours are authored in display sRGB and stored straight — bark colour comes
// from the texture, not the mesh's dark `trunk_color` tint.

// ONE warm-brown stem tone. Bark variation is **value-only** (fine vertical grain
// + a gentle broad undulation), so the stem reads as a single consistent brown
// with lines running up it — not a patchwork of light/dark zones, and never dark
// spots. Authored in display sRGB, stored straight.
/// The stem brown; everything else is a value modulation of it. A deep, warm
/// walnut brown — dark and rich rather than pale tan.
const BARK_BROWN: [f32; 3] = [0.44, 0.32, 0.20];
/// Height→normal slope gain. Low — the relief is shallow smooth grain, not furrows.
const BARK_NORMAL_STRENGTH: f32 = 4.5;

/// The bark's structural fields at a texel — the shared source for both the albedo
/// and the normal/roughness map. `height` is the shallow smooth relief (grain a
/// touch recessed); `line` is the signed fine vertical grain (−dark .. +light);
/// `broad` is a gentle large-scale value undulation.
struct BarkH {
    height: f32,
    line: f32,
    broad: f32,
}

/// Evaluate the bark height field at texel `(x, y)`. The look is a **smooth young
/// stem**: fine continuous vertical grain lines running up it, no furrows or
/// blobs. Higher frequency across the stem (`x`), very low along it (`y`) → long
/// vertical lines. Every field is **periodic over `CELL_PX` in BOTH axes**, so one
/// cell wrap-tiles seamlessly around the stem (x) and up it (y) — no seam, no
/// mirror, no repeating landmark.
fn bark_height(x: f32, y: f32, seed: u64) -> BarkH {
    // Gentle wander so the grain bows organically up the stem (not ruler-straight).
    // Periodic so it doesn't break the wrap. Kept small so the grain stays mostly
    // VERTICAL — a larger warp smeared the lines into a diagonal band up close.
    let xw = x + (grad_noise_per(x, 0.005, y, 0.013, seed ^ 0xA11) - 0.5) * 5.0;

    // Vertical GRAIN: x-freq sets the line width (chunky, not hairline), very low
    // y-freq → long striations. Two layers — a dominant chunky band plus a finer
    // accent — combined as a CONTINUOUS signed value (never thresholded), so it
    // reads as lines, never dark spots.
    let g0 = grad_noise_per(xw, 0.030, y, 0.009, seed ^ 0x1234) - 0.5;
    let g1 = grad_noise_per(xw, 0.075, y, 0.017, seed ^ 0x5151) - 0.5;
    let line = g0 * 0.76 + g1 * 0.24;

    // A gentle broad undulation so the stem isn't perfectly uniform.
    let broad = grad_noise_per(xw, 0.013, y, 0.007, seed ^ 0x9D) - 0.5;

    // Shallow smooth relief: grain a touch recessed; no deep furrows.
    let height = (0.6 + 0.55 * line + 0.30 * broad).clamp(0.0, 1.0);

    BarkH { height, line, broad }
}

/// Opaque bark albedo: ONE warm-brown stem tone modulated in value only by the
/// fine vertical grain + the broad undulation — consistent brown, lines running up
/// the stem, no dark spots.
fn draw_bark(px: &mut [[f32; 4]], ox: usize, oy: usize, seed: u64) {
    let cell = CELL_PX as usize;
    for y in 0..cell {
        for x in 0..cell {
            let (fx, fy) = (x as f32, y as f32);
            let h = bark_height(fx, fy, seed);

            // Value modulation only: the fine vertical GRAIN carries the read
            // (stronger, so crisp lines run up the stem), while the broad
            // undulation is kept subtle so it doesn't pool into a diagonal blotch
            // up close. Stays one consistent brown.
            let v = (1.0 + 0.30 * h.line + 0.05 * h.broad).clamp(0.5, 1.3);
            let col = [BARK_BROWN[0] * v, BARK_BROWN[1] * v, BARK_BROWN[2] * v];

            let idx = (oy + y) * SIZE + (ox + x);
            px[idx] = [
                col[0].clamp(0.0, 1.0),
                col[1].clamp(0.0, 1.0),
                col[2].clamp(0.0, 1.0),
                1.0,
            ];
        }
    }
}

/// Companion bark normal + roughness, in the [`foliage_material_atlas`]. RGB is a
/// tangent-space normal from the central difference of [`bark_height`]; A is
/// roughness (furrows rougher, ridges a touch smoother). Linear data.
fn draw_bark_material(px: &mut [[f32; 4]], ox: usize, oy: usize, seed: u64) {
    let cell = CELL_PX as usize;
    let h = |xx: f32, yy: f32| bark_height(xx, yy, seed).height;
    for y in 0..cell {
        for x in 0..cell {
            let (fx, fy) = (x as f32, y as f32);
            let c = bark_height(fx, fy, seed);

            // Tangent-space normal from the height slope (OpenGL +Y convention;
            // the shader builds its TBN from screen-space derivatives to match).
            let dx = (h(fx + 1.0, fy) - h(fx - 1.0, fy)) * BARK_NORMAL_STRENGTH;
            let dy = (h(fx, fy + 1.0) - h(fx, fy - 1.0)) * BARK_NORMAL_STRENGTH;
            let inv = 1.0 / (dx * dx + dy * dy + 1.0).sqrt();
            let (nx, ny, nz) = (-dx * inv, -dy * inv, inv);

            // Near-uniform, a touch rougher in the recessed grain.
            let rough = (0.84 - 0.08 * c.line).clamp(0.7, 0.94);

            let idx = (oy + y) * SIZE + (ox + x);
            px[idx] = [nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5, rough];
        }
    }
}

// ── Grass clump-card atlas ───────────────────────────────────────────────────
//
// The far-band grass representation: instead of a fountain of blade strips, the
// game draws two crossed quads per clump and samples this atlas — the classic
// baked grass-card technique (paint a whole cluster of blades once, splat it on
// a card). Two layers per cell give the card depth: a dim, packed background
// understory behind sharp, bright foreground blades — so it reads as a slice of
// meadow, not a fence of stripes.
//
// The RGB channels store a **modulation**, not a colour: the grass shader
// multiplies the decoded texel (× [`GRASS_CARD_RGB_SCALE`]) over the per-clump
// landcover tint, so cards track the terrain palette exactly like the near
// blade meshes (whose vertex colours carry the same root-dark → tip-light ramp
// + per-blade hue/value jitter this texture bakes). Upload as **linear**
// (`Rgba8Unorm`), not sRGB. A is straight coverage alpha (shader discards
// below 0.5).

/// Card variants in the atlas, side by side left → right. Each card picks one
/// by clump hash so neighbouring cards differ.
pub const GRASS_CARD_VARIANTS: u32 = 4;
/// Pixels per card cell — ~2:1 wide, matching the card quad's world aspect.
pub const GRASS_CARD_CELL_W: u32 = 512;
pub const GRASS_CARD_CELL_H: u32 = 256;
/// The atlas RGB stores `modulation / GRASS_CARD_RGB_SCALE` so the per-blade
/// hue/value jitter that peaks above 1.0 survives u8 encoding; the shader
/// multiplies the decoded texel back up. Mirrored as a literal in `grass.wgsl`.
pub const GRASS_CARD_RGB_SCALE: f32 = 1.35;

/// Generate the grass clump-card atlas (**linear** RGBA8 — modulation + straight
/// coverage alpha, see the module comment above). Deterministic.
pub fn grass_card_atlas() -> TextureData {
    let (cw, ch) = (GRASS_CARD_CELL_W as usize, GRASS_CARD_CELL_H as usize);
    let aw = cw * GRASS_CARD_VARIANTS as usize;
    let mut rgba = vec![0u8; aw * ch * 4];
    for variant in 0..GRASS_CARD_VARIANTS {
        let cell = draw_grass_card_cell(variant as u64 * 6151 + 17);
        for y in 0..ch {
            for x in 0..cw {
                let p = cell[y * cw + x];
                let idx = (y * aw + variant as usize * cw + x) * 4;
                rgba[idx] = to_u8(p[0]);
                rgba[idx + 1] = to_u8(p[1]);
                rgba[idx + 2] = to_u8(p[2]);
                rgba[idx + 3] = to_u8(p[3]);
            }
        }
    }
    TextureData {
        width: aw as u32,
        height: ch as u32,
        rgba,
    }
}

/// Paint one card cell (straight-alpha scratch, `GRASS_CARD_CELL_W × _H`):
/// background understory first, foreground blades over it, then a left/right
/// edge fade so the two crossed quads of a card blend where they meet.
fn draw_grass_card_cell(seed: u64) -> Vec<[f32; 4]> {
    let (w, h) = (GRASS_CARD_CELL_W as usize, GRASS_CARD_CELL_H as usize);
    let mut px = vec![[0.0f32; 4]; w * h];
    let (fw, fh) = (w as f32, h as f32);
    // Roots just below the bottom edge so the base row is dense to the edge
    // (the card sits on the ground; a gap under the roots would float it).
    let root_y = fh * 1.02;
    // Per-variant prevailing lean, so different cards read as differently
    // wind-combed patches rather than four copies of one tuft.
    let common_lean = (hash01(seed, 90) - 0.5) * fw * 0.16;

    // Background understory: dim, slightly wider blades drawn first — the dark
    // packed mass the bright blades read against (the reference-card depth cue).
    for i in 0..52u32 {
        let s = seed ^ 0x00B1_ADE5 ^ (i as u64).wrapping_mul(GOLD);
        let frac = (i as f32 + 0.5) / 52.0;
        let rx = fw * frac + (hash01(s, 1) - 0.5) * fw * 0.05;
        let height = fh * (0.38 + 0.42 * hash01(s, 2));
        let lean = common_lean * (0.4 + hash01(s, 3)) + (hash01(s, 4) - 0.5) * fw * 0.20;
        let bow = (hash01(s, 5) - 0.5) * fw * 0.07;
        let base_w = fw * (0.009 + 0.008 * hash01(s, 6));
        // Slightly dimmed ramp: reads as shadowed depth behind the bright
        // blades, but stays close to the near-blade root value (0.62) so the
        // minified average doesn't sink below the terrain albedo (a too-dark
        // understory shows as a dark far-grass band in-game).
        let v = 0.85 + 0.25 * hash01(s, 7);
        let hue = (hash01(s, 8) - 0.5) * 0.10;
        stamp_card_blade(&mut px, w, h, rx, root_y, height, lean, bow, base_w, |t| {
            let val = (0.44 + 0.32 * t) * v / GRASS_CARD_RGB_SCALE;
            [val * (1.0 + hue), val, val * (1.0 - hue)]
        });
    }

    // Foreground blades: sharp, full modulation ramp — root-dark → bright tip,
    // matching the near blade meshes' vertex ramp (0.62 → 1.0) with the same
    // per-blade hue drift + value jitter `push_grass_blade` applies.
    for i in 0..34u32 {
        let s = seed ^ 0x0F0E_60A2 ^ (i as u64).wrapping_mul(GOLD);
        let frac = (i as f32 + 0.5) / 34.0;
        let rx = fw * frac + (hash01(s, 1) - 0.5) * fw * 0.06;
        let height = fh * (0.55 + 0.42 * hash01(s, 2));
        let lean = common_lean * (0.5 + hash01(s, 3)) + (hash01(s, 4) - 0.5) * fw * 0.16;
        let bow = (hash01(s, 5) - 0.5) * fw * 0.08;
        let base_w = fw * (0.008 + 0.007 * hash01(s, 6));
        let v = 0.86 + 0.24 * hash01(s, 7);
        let hue = (hash01(s, 8) - 0.5) * 0.12;
        stamp_card_blade(&mut px, w, h, rx, root_y, height, lean, bow, base_w, |t| {
            // The near-blade vertex ramp: 0.62 at the root → 1.0 at the tip.
            let val = (0.62 + 0.38 * t.powf(0.85)) * v / GRASS_CARD_RGB_SCALE;
            [val * (1.0 + hue), val, val * (1.0 - hue)]
        });
    }

    // The `over` compositing above blended blades onto a *black* transparent
    // background, so partially-covered texels carry black-darkened RGB — and
    // bilinear filtering near edges mixes fully-transparent black texels in.
    // Both show up as dark fringes / a dark cast once the card is minified at
    // distance. Un-premultiply the RGB back out of the coverage, then bleed
    // colour a few texels into the transparent surround.
    for p in px.iter_mut() {
        if p[3] > 1.0e-3 {
            let inv = 1.0 / p[3];
            p[0] *= inv;
            p[1] *= inv;
            p[2] *= inv;
        }
    }
    for _ in 0..6 {
        let snap = px.clone();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if snap[i][3] > 2.0e-2 {
                    continue;
                }
                let (mut acc, mut n) = ([0.0f32; 3], 0.0f32);
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let q = snap[ny as usize * w + nx as usize];
                    // Any texel that already carries colour (covered, or filled
                    // by an earlier dilation pass) contributes.
                    if q[0] + q[1] + q[2] > 1.0e-4 {
                        acc[0] += q[0];
                        acc[1] += q[1];
                        acc[2] += q[2];
                        n += 1.0;
                    }
                }
                if n > 0.0 && px[i][0] + px[i][1] + px[i][2] <= 1.0e-4 {
                    px[i][0] = acc[0] / n;
                    px[i][1] = acc[1] / n;
                    px[i][2] = acc[2] / n;
                }
            }
        }
    }

    // Left/right edge fade (alpha only, after the un-premultiply so it doesn't
    // re-darken RGB): neighbouring crossed quads blend instead of showing hard
    // quad borders.
    for y in 0..h {
        for x in 0..w {
            let fx = (x as f32 + 0.5) / fw;
            let edge = smooth01((fx / 0.05).min(1.0)) * smooth01(((1.0 - fx) / 0.05).min(1.0));
            px[y * w + x][3] *= edge;
        }
    }
    px
}

/// Rasterize one curved tapered grass blade: a quadratic Bézier from the root
/// `(rx, ry)` to the tip `(rx + lean, ry - height)` with a perpendicular `bow`,
/// stamped as overlapping soft discs whose radius tapers root → tip. `color`
/// maps the along-blade fraction `t` (0 root → 1 tip) to the disc colour.
#[allow(clippy::too_many_arguments)]
fn stamp_card_blade(
    px: &mut [[f32; 4]],
    w: usize,
    h: usize,
    rx: f32,
    ry: f32,
    height: f32,
    lean: f32,
    bow: f32,
    base_w: f32,
    color: impl Fn(f32) -> [f32; 3],
) {
    let tip_x = rx + lean;
    let tip_y = ry - height;
    // Control point: segment midpoint pushed along the perpendicular by `bow`.
    let (mx, my) = ((rx + tip_x) * 0.5, (ry + tip_y) * 0.5);
    let (dx, dy) = (tip_x - rx, tip_y - ry);
    let len = (dx * dx + dy * dy).sqrt().max(1.0);
    let (perp_x, perp_y) = (-dy / len, dx / len);
    let (cx, cy) = (mx + perp_x * bow, my + perp_y * bow);

    let steps = (height * 1.5) as u32 + 8;
    for k in 0..=steps {
        let t = k as f32 / steps as f32;
        let omt = 1.0 - t;
        let bx = omt * omt * rx + 2.0 * omt * t * cx + t * t * tip_x;
        let by = omt * omt * ry + 2.0 * omt * t * cy + t * t * tip_y;
        let bw = (base_w * omt.powf(0.6)).max(0.5);
        stamp_disk(px, w, h, bx, by, bw, color(t));
    }
}

/// Composite a soft-edged filled disc (1 px anti-aliased) with straight alpha.
fn stamp_disk(px: &mut [[f32; 4]], w: usize, h: usize, cx: f32, cy: f32, r: f32, color: [f32; 3]) {
    let reach = (r + 1.0).ceil() as i32;
    let (x0, y0) = (cx as i32, cy as i32);
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let (lx, ly) = (x0 + dx, y0 + dy);
            if lx < 0 || ly < 0 || lx >= w as i32 || ly >= h as i32 {
                continue;
            }
            let fx = lx as f32 + 0.5 - cx;
            let fy = ly as f32 + 0.5 - cy;
            let d = (fx * fx + fy * fy).sqrt();
            let alpha = smooth01(r - d).clamp(0.0, 1.0);
            if alpha <= 0.003 {
                continue;
            }
            let idx = ly as usize * w + lx as usize;
            over(&mut px[idx], [color[0], color[1], color[2], alpha]);
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn over(dst: &mut [f32; 4], src: [f32; 4]) {
    let a = src[3];
    let ia = 1.0 - a;
    dst[0] = src[0] * a + dst[0] * ia;
    dst[1] = src[1] * a + dst[1] * ia;
    dst[2] = src[2] * a + dst[2] * ia;
    dst[3] = a + dst[3] * ia;
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[inline]
fn smooth01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// 2D **gradient (Perlin) noise** in `[0, 1]`, **periodic in BOTH axes** with
/// period `CELL_PX` px. Gradient noise (not value noise) because its derivative is
/// near-isotropic, so differencing it for a normal map shows no axis-aligned
/// lattice "weave" (see the `wgsl-bevy` skill note). The toroidal periodicity lets
/// one cell wrap-tile seamlessly **around** a trunk (x) and **up** it (y): each
/// lattice wraps, and `fx`/`fy` are snapped to a whole number of lattice cells per
/// `CELL_PX`.
fn grad_noise_per(x: f32, fx: f32, y: f32, fy: f32, seed: u64) -> f32 {
    let px = (CELL_PX as f32 * fx).round().max(1.0);
    let py = (CELL_PX as f32 * fy).round().max(1.0);
    let (xs, ys) = (x * (px / CELL_PX as f32), y * (py / CELL_PX as f32));
    let (xi, yi) = (xs.floor(), ys.floor());
    let (fxf, fyf) = (xs - xi, ys - yi);
    // Hash a lattice corner to a unit gradient (both axes wrapped mod their
    // periods), dotted with the corner→point offset.
    let grad = |gx: f32, gy: f32, dx: f32, dy: f32| {
        let gxp = gx.rem_euclid(px);
        let gyp = gy.rem_euclid(py);
        let h = (gxp as i64 as u64)
            .wrapping_mul(GOLD)
            .wrapping_add((gyp as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
            .wrapping_add(seed.wrapping_mul(0x1656_67B1_9E37_79F9));
        let a = hash_to_unit(h) * TAU;
        dx * a.cos() + dy * a.sin()
    };
    let (u, v) = (smooth01(fxf), smooth01(fyf));
    let nx0 = {
        let a = grad(xi, yi, fxf, fyf);
        let b = grad(xi + 1.0, yi, fxf - 1.0, fyf);
        a + u * (b - a)
    };
    let nx1 = {
        let a = grad(xi, yi + 1.0, fxf, fyf - 1.0);
        let b = grad(xi + 1.0, yi + 1.0, fxf - 1.0, fyf - 1.0);
        a + u * (b - a)
    };
    let n = nx0 + v * (nx1 - nx0);
    (n * 0.9 + 0.5).clamp(0.0, 1.0)
}

/// Value noise in `[0, 1]` over a 2D coordinate (cheap, for texture mottling).
fn value_noise(x: f32, y: f32, seed: u64) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (fx, fy) = (x - xi, y - yi);
    let c = |gx: f32, gy: f32| {
        let h = (gx as i64 as u64)
            .wrapping_mul(GOLD)
            .wrapping_add((gy as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
            .wrapping_add(seed.wrapping_mul(0x1656_67B1_9E37_79F9));
        hash_to_unit(h)
    };
    let (sx, sy) = (smooth01(fx), smooth01(fy));
    let a = c(xi, yi) + sx * (c(xi + 1.0, yi) - c(xi, yi));
    let b = c(xi, yi + 1.0) + sx * (c(xi + 1.0, yi + 1.0) - c(xi, yi + 1.0));
    a + sy * (b - a)
}

/// Integer-mix hash → `[0, 1)` per `(seed, salt)`.
fn hash01(seed: u64, salt: u64) -> f32 {
    hash_to_unit(seed ^ salt.wrapping_mul(GOLD) ^ 0x2545_F491_4F6C_DD1D)
}

fn hash_to_unit(mut h: u64) -> f32 {
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h & 0x000F_FFFF_FFFF_FFFF) as f32 / (1u64 << 52) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_has_expected_size_and_alpha_extremes() {
        let tex = foliage_atlas();
        assert_eq!(tex.width, SIZE as u32);
        assert_eq!(tex.height, SIZE as u32);
        assert_eq!(tex.rgba.len(), SIZE * SIZE * 4);
        let min_a = tex.rgba.iter().skip(3).step_by(4).copied().min().unwrap();
        let max_a = tex.rgba.iter().skip(3).step_by(4).copied().max().unwrap();
        assert_eq!(max_a, 255, "opaque bark expected");
        assert!(min_a < 16, "transparent leaf gaps expected, got {min_a}");
    }

    #[test]
    fn material_atlas_matches_albedo_layout() {
        let tex = foliage_material_atlas();
        assert_eq!(tex.width, SIZE as u32);
        assert_eq!(tex.height, SIZE as u32);
        assert_eq!(tex.rgba.len(), SIZE * SIZE * 4);
        // A bark cell's centre normal should point mostly +Z (blue ≳ red/green).
        let bx = (BARK_CELL_FIRST % ATLAS_N) as usize * CELL_PX as usize + CELL_PX as usize / 2;
        let by = (BARK_CELL_FIRST / ATLAS_N) as usize * CELL_PX as usize + CELL_PX as usize / 2;
        let i = (by * SIZE + bx) * 4;
        assert!(
            tex.rgba[i + 2] > tex.rgba[i] && tex.rgba[i + 2] > tex.rgba[i + 1],
            "bark normal should be +Z dominant (blue)"
        );
    }

    #[test]
    fn bark_height_is_toroidally_periodic() {
        // The bark height field must repeat over CELL_PX in BOTH axes so one cell
        // wrap-tiles seamlessly around a trunk (x) and up it (y) — no seam, no
        // mirror. Check periodicity along each axis at a few offsets.
        let p = CELL_PX as f32;
        for &t in &[10.0, 73.0, 128.0, 201.0] {
            let yx = (bark_height(t, 0.0, 99).height - bark_height(t, p, 99).height).abs();
            assert!(yx < 1e-4, "bark height not y-periodic at x={t}");
            let xx = (bark_height(0.0, t, 99).height - bark_height(p, t, 99).height).abs();
            assert!(xx < 1e-4, "bark height not x-periodic at y={t}");
        }
    }

    #[test]
    fn leaf_code_round_trips() {
        assert_eq!(leaf_code(SHELL_CELL, 2), (SHELL_CELL * 4 + 2) as f32);
        assert_eq!(leaf_code(BARK_CELL_FIRST, 4), (BARK_CELL_FIRST * 4) as f32);
    }
}
