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
//! bark) the tree meshes sample. Rocks and other procedural textures will live
//! here too.
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

/// A dense clump of **many small, multi-toned leaves**: a faint deep-green base
/// for card coverage, then hundreds of tiny coloured leaves (a green palette,
/// lighter toward the top/outer rim) that build an opaque centre by overlap and
/// break into a leafy silhouette at the edges.
fn draw_leaf_cluster(px: &mut [[f32; 4]], ox: usize, oy: usize, seed: u64) {
    let cell = CELL_PX as usize;
    let w = CELL_PX as f32;
    let half = w * 0.5;

    // Faint deep-green base — guarantees the card's centre is opaque (so heavily
    // overlapping cards never see through), mostly hidden by the leaves on top.
    for y in 0..cell {
        for x in 0..cell {
            let dx = (x as f32 + 0.5 - half) / half;
            let dy = (y as f32 + 0.5 - half) / half;
            let d = (dx * dx + dy * dy).sqrt();
            let bump = 0.16 * (value_noise(x as f32 * 0.045, y as f32 * 0.045, seed ^ 0x51) - 0.5);
            let cover = smooth01((0.74 + bump - d) / 0.22) * 0.55;
            if cover <= 0.01 {
                continue;
            }
            let idx = (oy + y) * SIZE + (ox + x);
            over(&mut px[idx], [0.13, 0.26, 0.12, cover]);
        }
    }

    // Many small leaves.
    for li in 0..340u32 {
        let s = seed.wrapping_add((li as u64).wrapping_mul(GOLD));
        let rr = hash01(s, 1).powf(0.55); // centre-biased
        let r = rr * half * 1.04;
        let a = hash01(s, 2) * TAU;
        let cx = half + r * a.cos();
        let cy = half + r * a.sin();
        let ang = hash01(s, 3) * TAU;
        let len = w * (0.035 + 0.028 * hash01(s, 4)); // ~9–16 px in a 256 cell
        let wid = len * (0.42 + 0.22 * hash01(s, 5));
        let topness = (1.0 - cy / w).clamp(0.0, 1.0);
        let t = (0.18 + 0.50 * hash01(s, 6) + 0.28 * topness).clamp(0.0, 1.0);
        stamp_leaf(px, ox, oy, cx, cy, ang, len, wid, leaf_palette(t, s));
    }
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

/// Composite one rotated soft-edged leaf (a tapered ellipse with a faint midrib).
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
            let alpha = smooth01(1.0 - (d - 0.65) / 0.35).clamp(0.0, 1.0);
            if alpha <= 0.003 {
                continue;
            }
            let rib = 1.0 - 0.16 * smooth01(1.0 - (v.abs() / 0.18).min(1.0));
            let src = [color[0] * rib, color[1] * rib, color[2] * rib, alpha];
            let idx = (oy + ly as usize) * SIZE + (ox + lx as usize);
            over(&mut px[idx], src);
        }
    }
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

/// Opaque bark for the trunk: a subtle soft vertical grain with sparse furrows
/// (luminance detail; the mesh's `trunk_color` supplies the brown).
fn draw_bark(px: &mut [[f32; 4]], ox: usize, oy: usize, seed: u64) {
    let cell = CELL_PX as usize;
    for y in 0..cell {
        for x in 0..cell {
            let grain = value_noise(x as f32 * 0.09, y as f32 * 0.018, seed);
            let fine = value_noise(x as f32 * 0.3, y as f32 * 0.3, seed ^ 0x9E37);
            let crack = value_noise(x as f32 * 0.045, y as f32 * 0.01, seed ^ 0x1234);
            let furrow = smooth01((crack - 0.66) / 0.06);
            let lum =
                (0.72 + 0.08 * (grain - 0.5) + 0.05 * (fine - 0.5) - 0.16 * furrow).clamp(0.45, 0.92);
            let idx = (oy + y) * SIZE + (ox + x);
            px[idx] = [lum, lum * 0.95, lum * 0.86, 1.0];
        }
    }
}

// ── Grass billboard texture ──────────────────────────────────────────────────

/// Width of the grass billboard texture.
pub const GRASS_TEX_W: u32 = 256;
/// Height of the grass billboard texture.
pub const GRASS_TEX_H: u32 = 256;

/// Generate a **multi-blade grass billboard** texture (sRGBA8, straight alpha):
/// a row of thin tapered blades rooted at the bottom and fanning up — mostly
/// upright with a shared slight lean and some per-blade variation, so a tuft
/// reads as "facing the same direction with variation" rather than a chaotic
/// bush. Mapped onto the vertical cross-quad grass billboards (reads from any
/// horizontal angle); the transparent background alpha-tests away between blades.
pub fn grass_blades() -> TextureData {
    let (w, h) = (GRASS_TEX_W as usize, GRASS_TEX_H as usize);
    let mut px = vec![[0.0f32; 4]; w * h];
    let (fw, fh) = (w as f32, h as f32);
    let root_y = fh * 0.985;
    let seed = 0x9B_1A_DEu64;

    const NB: u32 = 15;
    // Shared prevailing lean (px the tip drifts sideways) — blades mostly face
    // one way; per-blade variance is added on top.
    let common_lean = fw * 0.07;
    for i in 0..NB {
        let s = seed ^ (i as u64).wrapping_mul(GOLD);
        // Roots spread across the width (evenly seeded + jittered) so blades
        // start at distinct spots, not all stacked.
        let frac = (i as f32 + 0.5) / NB as f32;
        let rx = fw * (0.10 + 0.80 * frac) + (hash01(s, 1) - 0.5) * fw * 0.045;
        let height = fh * (0.52 + 0.42 * hash01(s, 2));
        // Mostly the shared lean, ± per-blade variation (a few drift the other
        // way for naturalness).
        let lean = common_lean * (0.4 + hash01(s, 3)) + (hash01(s, 4) - 0.5) * fw * 0.12;
        let bow = (hash01(s, 5) - 0.5) * fw * 0.06;
        let base_w = fw * (0.011 + 0.010 * hash01(s, 6));
        stamp_blade(&mut px, w, h, rx, root_y, height, lean, bow, base_w, s);
    }

    let mut rgba = vec![0u8; w * h * 4];
    for (i, p) in px.iter().enumerate() {
        rgba[i * 4] = to_u8(p[0]);
        rgba[i * 4 + 1] = to_u8(p[1]);
        rgba[i * 4 + 2] = to_u8(p[2]);
        rgba[i * 4 + 3] = to_u8(p[3]);
    }
    TextureData {
        width: w as u32,
        height: h as u32,
        rgba,
    }
}

/// Rasterize one curved tapered grass blade: a quadratic Bézier from the root
/// `(rx, ry)` to the tip `(rx + lean, ry - height)` with a perpendicular `bow`,
/// stamped as overlapping soft discs whose radius tapers root → tip, coloured
/// deep-green at the root fading to a lighter tip.
#[allow(clippy::too_many_arguments)]
fn stamp_blade(
    px: &mut [[f32; 4]],
    w: usize,
    h: usize,
    rx: f32,
    ry: f32,
    height: f32,
    lean: f32,
    bow: f32,
    base_w: f32,
    seed: u64,
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
        // Deep green at the root → lighter toward the tip.
        let ct = (0.12 + 0.72 * t).clamp(0.0, 1.0);
        stamp_disk(px, w, h, bx, by, bw, leaf_palette(ct, seed));
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
    fn leaf_code_round_trips() {
        assert_eq!(leaf_code(SHELL_CELL, 2), (SHELL_CELL * 4 + 2) as f32);
        assert_eq!(leaf_code(BARK_CELL_FIRST, 4), (BARK_CELL_FIRST * 4) as f32);
    }
}
