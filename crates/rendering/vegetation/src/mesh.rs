//! Procedural tree mesh generation for the vegetation scatter layer.
//!
//! Builds one combined mesh (tapered trunk + a few canopy blobs) with
//! per-vertex colours, authored **+Y up with the trunk base at the origin**, so
//! the scatter driver orients it to the local terrain normal and scales it per
//! instance. A small library of these is generated once at startup and scattered
//! with per-instance variation — never per-instance geometry synthesis — so all
//! instances of one species share a `(Mesh, Material)` and Bevy auto-batches
//! them into instanced draws.
//!
//! `lod` reduces the radial/ring resolution and the blob count for the far mesh
//! LODs in the tree cascade (see `docs/world/vegetation.md`).

use bevy::asset::RenderAssetUsages;
use bevy::math::Vec3;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};

use crate::atlas::{
    BARK_CELL_COUNT, BARK_CELL_FIRST, LEAF_CELL_COUNT, LEAF_CELL_FIRST, NEEDLE_CELL, leaf_code,
};

const TAU: f32 = std::f32::consts::TAU;

/// Canopy silhouette — gives the forest more than one tree shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CanopyStyle {
    /// Rounded clump of leaf cards over an ellipsoid (bushes / shrubs).
    #[default]
    Round,
    /// Broadleaf tree: a dense volumetric crown packed with small foliage puffs
    /// over a few irregular lobes (branches hidden inside; only the trunk shows).
    Broadleaf,
    /// Conifer: a conical scatter of drooping needle-spray cards (a pine).
    Conifer,
}

/// Parameters for one procedurally generated tree.
#[derive(Debug, Clone, Copy)]
pub struct TreeMeshParams {
    pub trunk_height_m: f32,
    pub trunk_radius_m: f32,
    /// Canopy lateral radius (crown half-width).
    pub canopy_radius_m: f32,
    /// Canopy vertical radius (crown half-height).
    pub canopy_height_m: f32,
    pub trunk_color: Vec3,
    pub canopy_color: Vec3,
    pub style: CanopyStyle,
    /// Deterministic shape seed (offsets the canopy blobs).
    pub seed: u64,
    /// Mesh level of detail: 0 = full, 1 = mid, 2+ = far. Reduces tessellation
    /// and blob count.
    pub lod: u32,
}

impl Default for TreeMeshParams {
    fn default() -> Self {
        Self {
            trunk_height_m: 4.5,
            trunk_radius_m: 0.28,
            canopy_radius_m: 2.6,
            canopy_height_m: 2.4,
            trunk_color: Vec3::new(0.16, 0.090, 0.045),
            canopy_color: Vec3::new(0.055, 0.115, 0.040),
            style: CanopyStyle::Round,
            seed: 0,
            lod: 0,
        }
    }
}

/// Raw CPU mesh arrays for one tree species at one LOD. Kept on the CPU (not
/// just as a GPU `Handle<Mesh>`) so the scatter driver can *combine* many trees
/// into one batched per-tile mesh — the same one-mesh-per-tile batching the
/// grass uses, which removes the per-tree ECS entity overhead and lets forests
/// scale to dense/far. `colors[i].w` is the per-vertex wind weight (0 trunk → 1
/// canopy top).
#[derive(Clone, Default)]
pub struct TreeMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    /// Per-vertex `cell·4 + corner` code into the foliage atlas (see
    /// [`crate::atlas`]): leaf cards point at a leaf-cluster cell,
    /// the inner shell at the opaque green cell, the trunk at a bark cell. The
    /// per-tile combiner stores this in `UV_1.y` for the shader to decode.
    pub leaf_code: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Sky exposure a leaf card sees, from the direction it faces out of the crown:
/// `1` straight up (full sky), [`CARD_SKY_FLOOR`] straight down (fully shaded by
/// the canopy above it). The **one** definition — every leaf-card builder below
/// multiplies its height-in-crown term by this, so all species darken their
/// flanks and undersides identically and the impostor bake inherits it.
///
/// **Why orientation and not just height.** The baked AO used to be dominated by
/// height in the crown, with at most a `0.80..1.00` orientation nudge. Height and
/// sky exposure agree on the crown's top and diverge everywhere else: an outer
/// card halfway up read as nearly fully lit whether it faced the sun or the
/// ground. So a crown's whole visible shell came out at high exposure — it
/// rendered as a bright hollow shell with no interior depth ("the canopies look
/// thin") whose mean value landed at open-meadow brightness, when closed canopy
/// should read clearly darker than the meadow around it.
///
/// The tell that identified this: forcing the shaded end of
/// `thalos::foliage`'s `canopy_grade` to saturated red barely tinted the crowns,
/// proving almost no *visible* fragment was reaching the shaded end at all — the
/// grade wasn't the lever, the AO distribution feeding it was.
fn card_sky_exposure(n_out: Vec3) -> f32 {
    let up = n_out.normalize_or(Vec3::Y).y * 0.5 + 0.5;
    CARD_SKY_FLOOR + (1.0 - CARD_SKY_FLOOR) * up
}

/// Exposure of a leaf card facing straight down — the deep shade under a crown.
/// A ~3× top-to-underside range; the old `0.80` floor was a ~1.25× range, which
/// is why crowns had no volume.
const CARD_SKY_FLOOR: f32 = 0.30;

impl TreeMeshData {
    fn new() -> Self {
        Self::default()
    }

    /// `color` is the per-vertex tint *with ambient occlusion already
    /// premultiplied* (the atlas supplies leaf shape + luminance detail, this
    /// supplies hue × AO). `wind_weight` (colour alpha) drives the vertex sway
    /// (0 = rigid trunk → 1 = canopy top); `code` selects the atlas cell/corner.
    fn push_vert(&mut self, pos: Vec3, normal: Vec3, color: Vec3, wind_weight: f32, code: f32) {
        self.positions.push(pos.to_array());
        self.normals.push(normal.normalize_or_zero().to_array());
        self.colors
            .push([color.x, color.y, color.z, wind_weight.clamp(0.0, 1.0)]);
        self.leaf_code.push(code);
    }
}

/// Build the raw CPU mesh arrays for one tree species at `params.lod`.
pub fn build_tree_mesh_data(params: &TreeMeshParams) -> TreeMeshData {
    let mut b = TreeMeshData::new();

    let trunk_segs = match params.lod {
        0 => 8u32,
        1 => 6,
        2 => 5,
        _ => 4,
    };

    push_trunk(&mut b, params, trunk_segs);
    match params.style {
        CanopyStyle::Round => push_canopy(&mut b, params),
        CanopyStyle::Broadleaf => push_broadleaf_canopy(&mut b, params),
        CanopyStyle::Conifer => push_conifer(&mut b, params),
    }
    b
}

/// Conifer canopy: a dense **conical scatter of drooping needle-spray cards** (a
/// pine) — radius narrows with height so it fills from a wide base to the apex
/// with no tier gaps. Larger cards / fewer of them give a clean silhouette (so we
/// avoid individual needle shafts, which read as spider-fronds — SpeedTree's
/// conifer guidance). Cards face outward and hang from their stem.
fn push_conifer(b: &mut TreeMeshData, params: &TreeMeshParams) {
    let crown_base = params.trunk_height_m * 0.22;
    let crown_top = params.trunk_height_m + params.canopy_height_m * 1.1;
    let span = (crown_top - crown_base).max(0.5);
    let base_radius = params.canopy_radius_m;
    let (cards, sz) = match params.lod {
        0 => (96u32, 1.5f32),
        1 => (46, 1.65),
        2 => (18, 1.95),
        _ => (7, 2.3),
    };
    for i in 0..cards {
        let s = params.seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        // Height fraction biased toward the base (the cone is wider/denser low).
        let ht = hash01(s, 1).powf(0.7);
        let y = crown_base + span * ht;
        let cone_r = (base_radius * (1.0 - 0.92 * ht)).max(base_radius * 0.06);
        let az = hash01(s, 2) * TAU;
        let (sa, ca) = az.sin_cos();
        let outdir = Vec3::new(ca, 0.0, sa);
        // Spread from near the axis (covers the trunk) out to the cone edge.
        let rr = 0.1 + 0.95 * hash01(s, 3);
        let pos = Vec3::new(outdir.x * cone_r * rr, y, outdir.z * cone_r * rr);
        // Bough faces outward and droops; small roll keeps the spray hanging
        // (not spun sideways). Light normal is up-biased so the layering reads.
        let face = (outdir * 0.82 - Vec3::Y * 0.22).normalize_or(outdir);
        let n_light = (outdir * 0.4 + Vec3::Y * 0.7).normalize_or(Vec3::Y);
        let roll = (hash01(s, 4) - 0.5) * 0.5;
        let size = (cone_r * 1.4).max(base_radius * sz * 0.5) * (0.8 + 0.4 * hash01(s, 5));
        let ao = (0.45 + 0.55 * ht) * card_sky_exposure(outdir);
        let wind = (0.3 + 0.7 * ht).clamp(0.0, 1.0);
        push_leaf_card(
            b,
            pos,
            face,
            n_light,
            size,
            roll,
            NEEDLE_CELL,
            params.canopy_color * ao,
            wind,
            false, // needle sprays are already elongated; single quad
        );
    }
}

/// Build a single standalone tree mesh from `params` (used by tests / previews;
/// the runtime scatter path combines `TreeMeshData` per tile instead).
pub fn build_tree_mesh(params: &TreeMeshParams) -> Mesh {
    let b = build_tree_mesh_data(params);
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    let count = b.positions.len();
    // Standalone preview/test mesh: no per-tile base, so UV_0 = 0 and UV_1.x = 0;
    // UV_1.y carries the atlas leaf code (the runtime combiner does the same).
    let uv0 = vec![[0.0f32; 2]; count];
    let uv1: Vec<[f32; 2]> = b.leaf_code.iter().map(|&c| [0.0, c]).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, b.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, b.normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, b.colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_indices(Indices::U32(b.indices));
    mesh
}

/// Tapered cylinder trunk (no caps), base ring at `y = 0`, top at
/// `y = trunk_height`, narrowing toward the crown.
///
/// The bark cell **wrap-tiles in both axes** (NOT mirror): one cell per segment
/// around, and stacked bands up the trunk, each tile sampling the full cell
/// `0→1`. Because the bark cell is **toroidally periodic** (see `bark_height`),
/// every tile-to-tile join — around and up — is seamless, with no reflection
/// (the mirror seam) and no axis-aligned symmetry. Bands are sized so a tile is
/// ~square (no vertical stretch). Tiles are independent quads (the wrap needs
/// distinct cell-edge vertices on each side of every join).
fn push_trunk(b: &mut TreeMeshData, params: &TreeMeshParams, segments: u32) {
    let base_r = params.trunk_radius_m;
    let top_r = params.trunk_radius_m * 0.62;
    let h = params.trunk_height_m;
    let seg = segments.max(3);
    let bark = BARK_CELL_FIRST + (params.seed % BARK_CELL_COUNT as u64) as u32;

    // One cell per segment around → cell width ≈ circumference/seg. Make a tile a
    // few× taller than wide: the content is vertical grain LINES, which read
    // naturally when elongated and run longer up the stem with fewer band joins.
    let r_avg = 0.5 * (base_r + top_r);
    let cell_w = std::f32::consts::TAU * r_avg.max(0.05) / seg as f32;
    let bands = (h / (cell_w * 2.6)).round().clamp(1.0, 8.0) as u32;

    let ring_y = |j: u32| h * j as f32 / bands as f32;
    // Root flare: the base swells into a buttress that decays over the lowest
    // ~18 % of the stem, so the trunk meets the ground like a tree instead of a
    // pole planted in the grass.
    let flare = |j: u32| -> f32 {
        let t = j as f32 / bands as f32; // 0 base .. 1 top
        let k = (1.0 - t / 0.18).max(0.0);
        1.0 + 0.55 * k * k
    };
    let ring_r = |j: u32| (base_r + (top_r - base_r) * (j as f32 / bands as f32)) * flare(j);
    // Trunk darkens slightly toward the base (an AO nudge; the bark fragment takes
    // colour from the texture, but the tint is kept for any vertex-colour consumer).
    let ring_col = |j: u32| params.trunk_color * (0.85 + 0.15 * j as f32 / bands as f32);
    let weight = |j: u32| 0.05 * j as f32 / bands as f32;

    for band in 0..bands {
        let (yb, rb, cb, wb) = (ring_y(band), ring_r(band), ring_col(band), weight(band));
        let (yt, rt, ct, wt) = (
            ring_y(band + 1),
            ring_r(band + 1),
            ring_col(band + 1),
            weight(band + 1),
        );
        for i in 0..seg {
            let a0 = i as f32 / seg as f32 * TAU;
            let a1 = (i + 1) as f32 / seg as f32 * TAU;
            let (s0, c0) = a0.sin_cos();
            let (s1, c1) = a1.sin_cos();
            // Outward horizontal normals (taper is gentle; horizontal is fine).
            let (n0, n1) = (Vec3::new(c0, 0.0, s0), Vec3::new(c1, 0.0, s1));
            // One independent quad = the full cell: BL/BR/TL/TR → corners 0/1/3/2.
            let start = b.positions.len() as u32;
            b.push_vert(
                Vec3::new(c0 * rb, yb, s0 * rb),
                n0,
                cb,
                wb,
                leaf_code(bark, 0),
            );
            b.push_vert(
                Vec3::new(c1 * rb, yb, s1 * rb),
                n1,
                cb,
                wb,
                leaf_code(bark, 1),
            );
            b.push_vert(
                Vec3::new(c0 * rt, yt, s0 * rt),
                n0,
                ct,
                wt,
                leaf_code(bark, 3),
            );
            b.push_vert(
                Vec3::new(c1 * rt, yt, s1 * rt),
                n1,
                ct,
                wt,
                leaf_code(bark, 2),
            );
            b.indices.extend_from_slice(&[
                start,
                start + 1,
                start + 2,
                start + 1,
                start + 3,
                start + 2,
            ]);
        }
    }
}

/// Broadleaf canopy: layers of alpha-tested **leaf-cluster cards** over the crown
/// volume — *no solid core*. Each card's texture is an opaque-centred leaf clump,
/// so heavily overlapping cards self-cover (a full puffy canopy with a leafy
/// silhouette and no see-through), and each carries the crown-outward (spherical)
/// normal so the cloud of flat cards lights like a soft volume (SpeedTree
/// "puffiness"). AO (darker low/inner) is premultiplied into the tint.
fn push_canopy(b: &mut TreeMeshData, params: &TreeMeshParams) {
    let rx = params.canopy_radius_m;
    let ry = params.canopy_height_m;
    let rz = rx;
    // Sit the crown lower on the trunk so it isn't a lollipop, and let the top
    // bulge a little above the nominal ellipsoid.
    let crown_base = params.trunk_height_m * 0.74;
    let center = Vec3::new(0.0, crown_base + ry * 0.85, 0.0);
    let crown_top = center.y + ry * 1.1;

    let (skin, lobes, per_lobe, fill, sz) = match params.lod {
        0 => (58u32, 5u32, 8u32, 22u32, 0.6f32),
        1 => (30, 3, 5, 8, 0.7),
        2 => (12, 1, 4, 3, 0.88),
        _ => (6, 0, 0, 0, 1.15),
    };

    // 1. Skin: cards over the whole crown ellipsoid on an **even
    //    (Fibonacci-sphere) distribution** — random placement statistically
    //    leaves gaps you can see the sky through, even coverage doesn't. With the
    //    opaque-centred leaf texture this is the hole-free base the lobes sit on.
    for i in 0..skin {
        let s = params.seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5C1;
        let dir = fib_dir(i, skin, s);
        let size = rx * sz * (0.85 + 0.4 * hash01(s, 8));
        push_skin_card(
            b, params, center, rx, ry, rz, crown_base, crown_top, s, dir, size,
        );
    }

    // 2. Lobes: a few puffs protruding past the skin, to break the perfect-ball
    //    silhouette into a natural lumpy crown (each lit from its own centre).
    for lobe in 0..lobes {
        let s = params.seed ^ (lobe as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ 0x10BE;
        let off = Vec3::new(
            (hash01(s, 1) - 0.5) * 1.0 * rx,
            (hash01(s, 2) * 0.85 - 0.2) * ry,
            (hash01(s, 3) - 0.5) * 1.0 * rz,
        );
        let lobe_center = center + off;
        let lobe_r = rx * (0.5 + 0.24 * hash01(s, 4));
        for c in 0..per_lobe {
            let cs = s ^ (c as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
            let size = rx * sz * (0.8 + 0.5 * hash01(cs, 8));
            push_lobe_card(
                b,
                params,
                center,
                crown_base,
                crown_top,
                lobe_center,
                lobe_r,
                cs,
                size,
                0.12,
                1.0,
                true,
            );
        }
    }

    // 3. Inner fill across the core (incl. undersides) — backs the skin so even
    //    grazing lines of sight through the crown hit a card.
    for c in 0..fill {
        let cs = params.seed ^ (c as u64).wrapping_mul(0xA24B_AED4_963E_E407) ^ 0xF111;
        let size = rx * sz * 1.1;
        push_lobe_card(
            b,
            params,
            center,
            crown_base,
            crown_top,
            center,
            rx * 0.6,
            cs,
            size,
            0.05,
            0.85,
            false,
        );
    }
}

/// Even direction `i` of `n` over the upper ~73% of the sphere (Fibonacci /
/// golden-angle spiral), with a little hashed jitter so the regular lattice
/// doesn't read. Excludes the bottom cap (trees are sparse underneath).
fn fib_dir(i: u32, n: u32, seed: u64) -> Vec3 {
    let t = (i as f32 + 0.5) / n.max(1) as f32; // 0 (top) .. 1 (lower)
    let y = (1.0 - 1.45 * t + (hash01(seed, 3) - 0.5) * 0.10).clamp(-0.55, 1.0);
    let r = (1.0 - y * y).max(0.0).sqrt();
    let theta = i as f32 * 2.399_963_2 + hash01(seed, 4) * 0.6; // golden angle + jitter
    Vec3::new(r * theta.cos(), y, r * theta.sin())
}

/// One leaf card on the crown ellipsoid surface (the coverage skin), at the
/// given (even) direction.
#[allow(clippy::too_many_arguments)]
fn push_skin_card(
    b: &mut TreeMeshData,
    params: &TreeMeshParams,
    center: Vec3,
    rx: f32,
    ry: f32,
    rz: f32,
    crown_base: f32,
    crown_top: f32,
    seed: u64,
    dir: Vec3,
    size: f32,
) {
    let rr = 0.93 + 0.12 * hash01(seed, 5);
    let pos = center + Vec3::new(dir.x * rx * rr, dir.y * ry * rr, dir.z * rz * rr);
    let n_out = Vec3::new(dir.x / rx, dir.y / ry, dir.z / rz).normalize_or(Vec3::Y);
    let face = (n_out * 0.88 + rand_unit(seed, 30) * 0.12).normalize_or(n_out);
    // Dapple the *lighting* normal per card so the canopy gets tonal variation
    // instead of a flat, blown-out top where every card faces straight up.
    let n_light = (n_out + rand_unit(seed, 31) * 0.32).normalize_or(n_out);
    let roll = hash01(seed, 9) * TAU;
    let h01 = ((pos.y - crown_base) / (crown_top - crown_base).max(0.01)).clamp(0.0, 1.0);
    let ao = (0.52 + 0.4 * h01) * card_sky_exposure(n_out);
    let wind = (0.3 + 0.7 * h01).clamp(0.0, 1.0);
    let cell = LEAF_CELL_FIRST + (hash_u(seed, 12) % LEAF_CELL_COUNT.max(1));
    push_leaf_card(
        b,
        pos,
        face,
        n_light,
        size,
        roll,
        cell,
        params.canopy_color * ao,
        wind,
        params.lod <= 1, // cross-quad cards at near LODs
    );
}

/// Place one leaf-cluster card on a crown **lobe** (a puff centred at
/// `lobe_center`). `outer_bias` pushes the card to the lobe's outward, upper
/// face (the visible side); the lighting normal is the lobe-local outward
/// direction, so each puff reads as its own soft dome. AO from crown height.
#[allow(clippy::too_many_arguments)]
fn push_lobe_card(
    b: &mut TreeMeshData,
    params: &TreeMeshParams,
    crown_center: Vec3,
    crown_base: f32,
    crown_top: f32,
    lobe_center: Vec3,
    lobe_r: f32,
    seed: u64,
    size: f32,
    facing_scatter: f32,
    ao_scale: f32,
    outer_bias: bool,
) {
    let mut dir = rand_unit(seed, 20);
    if outer_bias {
        let outward = (lobe_center - crown_center).normalize_or(Vec3::Y);
        if dir.dot(outward) < 0.0 {
            dir = (dir + outward * 1.2).normalize_or(outward);
        }
        dir.y = dir.y.max(-0.2); // sparse undersides
        dir = dir.normalize_or(Vec3::Y);
    }
    let pos = lobe_center + dir * lobe_r * (0.85 + 0.22 * hash01(seed, 7));
    let n_out = dir;
    let face =
        (n_out * (1.0 - facing_scatter) + rand_unit(seed, 30) * facing_scatter).normalize_or(n_out);
    let n_light = (n_out + rand_unit(seed, 31) * 0.32).normalize_or(n_out);
    let roll = hash01(seed, 9) * TAU;
    let h01 = ((pos.y - crown_base) / (crown_top - crown_base).max(0.01)).clamp(0.0, 1.0);
    let ao = (0.4 + 0.6 * h01) * ao_scale * card_sky_exposure(n_out);
    let wind = (0.3 + 0.7 * h01).clamp(0.0, 1.0);
    let cell = LEAF_CELL_FIRST + (hash_u(seed, 12) % LEAF_CELL_COUNT.max(1));
    push_leaf_card(
        b,
        pos,
        face,
        n_light,
        size,
        roll,
        cell,
        params.canopy_color * ao,
        wind,
        params.lod <= 1, // cross-quad cards at near LODs
    );
}

// ---------------------------------------------------------------------------
// Broadleaf: a recursive branch skeleton with foliage clusters at the tips.
// ---------------------------------------------------------------------------

/// One node in the broadleaf branch-growth recursion: a tapered limb segment
/// growing from `base` along `dir` for `length`, starting at `radius`.
struct Limb {
    base: Vec3,
    dir: Vec3,
    length: f32,
    radius: f32,
    depth: u32,
}

/// Shared constants for one broadleaf grow pass (keeps the recursion arg list
/// short).
struct BroadleafCtx {
    depth_max: u32,
    seg: u32,
    bark: u32,
    crown_base: f32,
    crown_top: f32,
}

/// Broadleaf crown: grow a small **recursive branch skeleton** (trunk-top fork →
/// a few main limbs → each splits a couple of times, spreading outward and
/// reaching up via a phototropic bias, with Pipe-Model tapering) and hang a dense
/// rounded **foliage cluster** off every branch tip. The crown shape therefore
/// *emerges from the branching* — irregular, lobed, reaching — rather than being a
/// packed ellipsoid of puffs, which is what makes it read as a real tree. Only the
/// trunk and the thick first limbs are drawn; the twigs stay hidden inside the
/// overlapping clusters (the prior "barren skeleton" failure was sparse clusters
/// on visible branches, so here the clusters are big and overlap to cover them).
fn push_broadleaf_canopy(b: &mut TreeMeshData, params: &TreeMeshParams) {
    let rx = params.canopy_radius_m;
    let ry = params.canopy_height_m;
    let th = params.trunk_height_m;

    // (main limbs, recursion depth, limb radial segments, cards per cluster,
    // central fill clusters).
    let (n_main, depth_max, seg, cluster_cards, fill, shell) = match params.lod {
        0 => (4u32, 2u32, 6u32, 14u32, 3u32, 14u32),
        1 => (3, 1, 5, 10, 2, 8),
        2 => (3, 0, 4, 7, 0, 6),
        // Last mesh LOD before the impostor: a small but complete crown (a shell
        // of a few big cards) so it isn't a bald skeleton — the big `card_scale`
        // makes these cover.
        _ => (2, 0, 3, 6, 0, 5),
    };

    let ctx = BroadleafCtx {
        depth_max,
        seg,
        bark: BARK_CELL_FIRST + (params.seed % BARK_CELL_COUNT as u64) as u32,
        crown_base: th * 0.45,
        crown_top: th + ry * 1.1,
    };
    let top_r = params.trunk_radius_m * 0.62;
    let span = (ctx.crown_top - ctx.crown_base).max(0.01);

    // Grow the skeleton, collecting (centre, radius, height01) per cluster.
    let mut clusters: Vec<(Vec3, f32, f32)> = Vec::new();
    for m in 0..n_main.max(1) {
        let s = params.seed ^ (m as u64 + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ 0x10BE;
        // Main limbs fan out around the trunk, leaning upward by a varying amount
        // (some near-vertical, some spreading); their bases stagger down the upper
        // trunk so it isn't one candelabra fork.
        let az = (m as f32 / n_main.max(1) as f32) * TAU + hash01(s, 1) * 0.7;
        let (sa, ca) = az.sin_cos();
        let lean = 0.46 + 0.22 * hash01(s, 2);
        let up_bias = 0.80 + 0.35 * hash01(s, 6);
        let dir = Vec3::new(ca * lean, up_bias, sa * lean).normalize_or(Vec3::Y);
        let base_y = th - hash01(s, 3) * th * 0.18;
        let limb = Limb {
            base: Vec3::new(0.0, base_y, 0.0),
            dir,
            length: rx * (0.42 + 0.14 * hash01(s, 4)),
            radius: top_r * (0.5 + 0.2 * hash01(s, 5)),
            depth: 0,
        };
        grow_branch(b, params, &ctx, limb, s, &mut clusters);
    }

    // Base crown shell: clusters spread evenly over an (egg-bound) ellipsoid so
    // the crown is ALWAYS a full rounded mass — the branch tips above add the
    // irregular outer bumps and the lower limbs poke out the base, but this
    // guarantees no holes or pinched waist however the skeleton happened to grow.
    let cy = ctx.crown_base + ry * 0.92;
    for i in 0..shell {
        let s = params.seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5E11;
        let n = fib_dir(i, shell, s).normalize_or(Vec3::Y);
        let pos = Vec3::new(n.x * rx * 0.85, cy + n.y * ry * 0.95, n.z * rx * 0.85);
        let h01 = ((pos.y - ctx.crown_base) / span).clamp(0.0, 1.0);
        clusters.push((pos, rx * 0.46, h01));
    }

    // A few central fill clusters so grazing lines of sight through the crown
    // centre still hit foliage, not sky.
    let core = Vec3::new(0.0, ctx.crown_base + ry * 0.85, 0.0);
    for i in 0..fill {
        let s = params.seed ^ (i as u64 + 1).wrapping_mul(0xA24B_AED4_963E_E407) ^ 0xF111;
        let dir = rand_unit(s, 1);
        let pos = core + Vec3::new(dir.x * rx * 0.5, dir.y * ry * 0.5, dir.z * rx * 0.5);
        let h01 = ((pos.y - ctx.crown_base) / span).clamp(0.0, 1.0).max(0.45);
        clusters.push((pos, rx * 0.44, h01));
    }

    // Shape the collected clusters toward a rounded egg envelope (narrow base,
    // widest in the upper-middle, rounding off at the top) by scaling each
    // cluster's horizontal offset from the trunk axis, so the branch-driven crown
    // stays a coherent tree silhouette instead of an occasional pinched peanut.
    for (center, _, h01) in clusters.iter_mut() {
        let profile = crown_profile(*h01);
        center.x *= profile;
        center.z *= profile;
    }

    for &(center, radius, h01) in &clusters {
        let cs = params.seed ^ center.x.to_bits() as u64 ^ ((center.z.to_bits() as u64) << 1);
        push_foliage_cluster(b, params, center, radius, cluster_cards, h01, cs);
    }
}

/// Horizontal radius scale for a crown cluster at height fraction `t ∈ [0,1]`:
/// an egg/teardrop profile — pinched at the base, widest through the upper
/// middle, tapering again at the very crown.
fn crown_profile(t: f32) -> f32 {
    let smooth = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };
    let rise = smooth(t / 0.5);
    let top = smooth((t - 0.7) / 0.3);
    (0.45 + 0.55 * rise) * (1.0 - 0.40 * top)
}

/// Recursively grow one limb: draw it (only the thick near-trunk limbs), then
/// either drop a foliage cluster at its tip (terminal) or split into a couple of
/// child limbs that spread outward and reach up (phototropism), each thinned by
/// the Pipe Model (`r_parent² = Σ r_child²`).
fn grow_branch(
    b: &mut TreeMeshData,
    params: &TreeMeshParams,
    ctx: &BroadleafCtx,
    limb: Limb,
    seed: u64,
    clusters: &mut Vec<(Vec3, f32, f32)>,
) {
    let tip = limb.base + limb.dir * limb.length;
    let span = (ctx.crown_top - ctx.crown_base).max(0.01);
    let h1 = ((tip.y - ctx.crown_base) / span).clamp(0.0, 1.0);

    // Only the trunk-adjacent limbs are drawn; deeper twigs hide inside foliage.
    if limb.depth <= 1 {
        let h0 = ((limb.base.y - ctx.crown_base) / span).clamp(0.0, 1.0);
        push_limb(
            b,
            limb.base,
            tip,
            limb.radius,
            limb.radius * 0.66,
            ctx.bark,
            ctx.seg,
            params.trunk_color,
            h0 * 0.25,
            h1 * 0.25,
        );
    }

    if limb.depth >= ctx.depth_max {
        let cr = params.canopy_radius_m * (0.48 - 0.05 * limb.depth as f32).max(0.26);
        clusters.push((tip, cr, h1));
        return;
    }

    let nch = 2 + (hash_u(seed, 10) % 2);
    let (t, bi) = ortho_basis(limb.dir);
    for c in 0..nch {
        let cs = seed ^ (c as u64 + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let caz = (c as f32 / nch as f32) * TAU + hash01(cs, 1) * 1.2;
        let spread = 0.36 + 0.26 * hash01(cs, 2);
        let (ss, cc) = spread.sin_cos();
        let (sa, ca) = caz.sin_cos();
        let outdir = t * ca + bi * sa;
        let mut cdir = limb.dir * cc + outdir * ss + Vec3::Y * 0.38;
        cdir.y = cdir.y.max(-0.05); // keep child limbs from drooping below the crown
        let cdir = cdir.normalize_or(limb.dir);
        let child = Limb {
            base: tip,
            dir: cdir,
            length: limb.length * (0.62 + 0.12 * hash01(cs, 3)),
            radius: limb.radius / (nch as f32).sqrt() * (0.82 + 0.3 * hash01(cs, 4)),
            depth: limb.depth + 1,
        };
        grow_branch(b, params, ctx, child, cs, clusters);
    }
}

/// One foliage **cluster**: a rounded lobe of leaf-cluster cards spread evenly
/// over a small sphere (`fib_dir`), each facing outward so the lobe shades as a
/// soft dome. `h01` (the cluster's height in the crown) darkens lower lobes and
/// brightens the sunlit top; the per-card light normal is lifted toward the sky
/// so flank cards still catch the blue ambient and never read as a dark hole.
fn push_foliage_cluster(
    b: &mut TreeMeshData,
    params: &TreeMeshParams,
    center: Vec3,
    radius: f32,
    cards: u32,
    h01: f32,
    seed: u64,
) {
    let bright = 0.88 + 0.12 * hash01(seed, 3);
    // Constant-coverage on the mesh LOD chain: coarser LODs place FEWER cards
    // (the `cluster_cards` / `shell` counts drop), so each card GROWS to keep the
    // canopy covered instead of thinning to a sparse, small-leaved blob on the
    // first LOD step. (This is exactly what the `Round` canopy's `sz` already
    // does; the broadleaf previously kept a constant card size and just dropped
    // counts — the "leaves shrink and it goes sparse right after the close LOD"
    // report.) Cross-quads run one LOD further too (through LOD2) so the volume
    // doesn't collapse to edge-on slivers before the impostor takes over.
    let card_scale = match params.lod {
        0 => 1.0,
        1 => 1.3,
        2 => 1.6,
        _ => 2.0,
    };
    for i in 0..cards {
        let s = seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let dir = fib_dir(i, cards, s);
        let n_out = dir.normalize_or(Vec3::Y);
        let pos = center + n_out * radius * (0.85 + 0.16 * hash01(s, 9));
        let ao = bright * card_sky_exposure(n_out) * (0.62 + 0.38 * h01);
        let wind = (0.4 + 0.6 * h01).clamp(0.0, 1.0);
        let face = (n_out + rand_unit(s, 3) * 0.32).normalize_or(n_out);
        let n_light = (n_out + Vec3::Y * 0.24 + rand_unit(s, 4) * 0.20).normalize_or(n_out);
        let roll = hash01(s, 5) * TAU;
        let size = radius * card_scale * (0.9 + 0.4 * hash01(s, 6));
        let cell = LEAF_CELL_FIRST + (hash_u(s, 7) % LEAF_CELL_COUNT.max(1));
        push_leaf_card(
            b,
            pos,
            face,
            n_light,
            size,
            roll,
            cell,
            params.canopy_color * ao,
            wind,
            params.lod <= 2, // cross-quad cards through the mid LODs
        );
    }
}

/// A tapered branch cylinder from `base` (radius `r0`) to `tip` (radius `r1`)
/// along an arbitrary axis, bark-textured. Wind weight ramps base→tip.
#[allow(clippy::too_many_arguments)]
fn push_limb(
    b: &mut TreeMeshData,
    base: Vec3,
    tip: Vec3,
    r0: f32,
    r1: f32,
    bark: u32,
    segments: u32,
    color: Vec3,
    wind0: f32,
    wind1: f32,
) {
    let axis = (tip - base).normalize_or(Vec3::Y);
    let (t, bi) = ortho_basis(axis);
    let seg = segments.max(3);
    let start = b.positions.len() as u32;
    for i in 0..=seg {
        let a = i as f32 / seg as f32 * TAU;
        let (s, c) = a.sin_cos();
        let radial = (t * c + bi * s).normalize_or(t);
        let (cb, ct) = if i & 1 == 0 { (0, 3) } else { (1, 2) };
        b.push_vert(
            base + radial * r0,
            radial,
            color * 0.85,
            wind0,
            leaf_code(bark, cb),
        );
        b.push_vert(tip + radial * r1, radial, color, wind1, leaf_code(bark, ct));
    }
    for i in 0..seg {
        let r0i = start + i * 2;
        let r1i = start + (i + 1) * 2;
        b.indices
            .extend_from_slice(&[r0i, r1i, r0i + 1, r1i, r1i + 1, r0i + 1]);
    }
}

/// An orthonormal basis `(tangent, bitangent)` spanning the plane perpendicular
/// to `axis`.
fn ortho_basis(axis: Vec3) -> (Vec3, Vec3) {
    let up_ref = if axis.y.abs() > 0.95 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let t = axis.cross(up_ref).normalize_or(Vec3::X);
    let bi = axis.cross(t).normalize_or(Vec3::Z);
    (t, bi)
}

/// One leaf-cluster card centred at `pos`, carrying `light_normal` (the
/// crown-outward normal) so it shades as part of a soft sphere. Two-sided (the
/// material culls nothing). Corners map to the atlas cell's four UV corners.
///
/// When `cross` is set the card is a **2-quad cross** (a billboard cloud): a
/// second quad perpendicular to the first, sharing the card's `tr` spine. A
/// single flat quad reads as a paper-thin *sliver* whenever its face turns
/// edge-on to the camera — the "streaky wisps poking out of the canopy" tell up
/// close; the cross always presents a leaf face from any horizontal angle, so the
/// canopy reads as leafy volume. Near LODs cross; far LODs (where the impostor
/// takes over and edge-on never shows) stay single-quad to save triangles.
#[allow(clippy::too_many_arguments)]
fn push_leaf_card(
    b: &mut TreeMeshData,
    pos: Vec3,
    face: Vec3,
    light_normal: Vec3,
    size: f32,
    roll: f32,
    cell: u32,
    tint: Vec3,
    wind: f32,
    cross: bool,
) {
    let up_ref = if face.y.abs() > 0.95 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let t = face.cross(up_ref).normalize_or(Vec3::X);
    let bi = face.cross(t).normalize_or(Vec3::Z);
    let (sr, cr) = roll.sin_cos();
    let tr = t * cr + bi * sr;
    let br = -t * sr + bi * cr;
    let h = size * 0.5;

    // One quad spanned by in-plane axes `(au, av)`, corners → the cell's UV corners.
    let mut quad = |au: Vec3, av: Vec3| {
        let corners = [
            pos - au * h - av * h,
            pos + au * h - av * h,
            pos + au * h + av * h,
            pos - au * h + av * h,
        ];
        let start = b.positions.len() as u32;
        for (k, p) in corners.iter().enumerate() {
            b.push_vert(*p, light_normal, tint, wind, leaf_code(cell, k as u32));
        }
        b.indices
            .extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    };
    if cross {
        // Near-vertical cross: two perpendicular quads sharing a mostly-upright
        // spine, so from any ground-level / oblique angle at least one quad
        // presents a face — killing the "flat card edge-on → thin sliver poking
        // out of the canopy" tell. The spine LEANS a little (per-card, via `roll`)
        // off true vertical so the remaining edge-on slivers don't all line up
        // into vertical streaks across the canopy. The outward `light_normal`
        // still drives the soft-dome shading, independent of the quad geometry.
        let u1 = Vec3::new(cr, 0.0, sr); // horizontal lean direction (per card)
        let u2 = Vec3::new(-sr, 0.0, cr); // perpendicular horizontal
        let spine = (Vec3::Y + u1 * 0.25).normalize_or(Vec3::Y);
        let w1 = spine.cross(u2).normalize_or(u1); // ⟂ spine, in the lean plane
        quad(u2, spine);
        quad(w1, spine);
    } else {
        // Far LOD: a single outward-facing card (the impostor takes over before
        // edge-on slivers would read).
        quad(tr, br);
    }
}

/// A roughly uniform unit vector, for per-card facing scatter.
fn rand_unit(seed: u64, salt: u64) -> Vec3 {
    let y = 2.0 * hash01(seed, salt) - 1.0;
    let r = (1.0 - y * y).max(0.0).sqrt();
    let az = hash01(seed, salt + 1) * TAU;
    Vec3::new(r * az.cos(), y, r * az.sin())
}

/// Integer hash → `u32` in `[0, n)`-friendly range, deterministic per (seed, salt).
fn hash_u(seed: u64, salt: u64) -> u32 {
    let mut h = seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2545_F491_4F6C_DD1D;
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h & 0xFFFF_FFFF) as u32
}

/// Integer hash → `[0, 1)`, deterministic per (seed, salt).
fn hash01(seed: u64, salt: u64) -> f32 {
    let mut h = seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2545_F491_4F6C_DD1D;
    h ^= h >> 31;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 32;
    (h & 0x000F_FFFF_FFFF_FFFF) as f32 / (1u64 << 52) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_mesh_is_nonempty_and_finite() {
        let mesh = build_tree_mesh(&TreeMeshParams::default());
        let count = mesh.count_vertices();
        assert!(count > 0);
        if let Some(pos) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            // Sanity: vertices exist and there are triangle indices.
            assert!(!pos.is_empty());
        }
        assert!(mesh.indices().map(|i| i.len()).unwrap_or(0) >= 3);
    }

    #[test]
    fn lod_reduces_vertex_count() {
        let full = build_tree_mesh(&TreeMeshParams {
            lod: 0,
            ..Default::default()
        });
        let far = build_tree_mesh(&TreeMeshParams {
            lod: 2,
            ..Default::default()
        });
        assert!(far.count_vertices() < full.count_vertices());
    }
}
