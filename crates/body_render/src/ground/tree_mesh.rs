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
//! LODs in the tree cascade (see `docs/vegetation.md`).

use bevy::asset::RenderAssetUsages;
use bevy::math::Vec3;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};

use crate::ground::tree_atlas::{
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
    /// [`crate::ground::tree_atlas`]): leaf cards point at a leaf-cluster cell,
    /// the inner shell at the opaque green cell, the trunk at a bark cell. The
    /// per-tile combiner stores this in `UV_1.y` for the shader to decode.
    pub leaf_code: Vec<f32>,
    pub indices: Vec<u32>,
}

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
        let ao = 0.45 + 0.55 * ht;
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
fn push_trunk(b: &mut TreeMeshData, params: &TreeMeshParams, segments: u32) {
    let base_r = params.trunk_radius_m;
    let top_r = params.trunk_radius_m * 0.62;
    let h = params.trunk_height_m;
    let seg = segments.max(3);
    let start = b.positions.len() as u32;
    let bark = BARK_CELL_FIRST + (params.seed % BARK_CELL_COUNT as u64) as u32;

    for i in 0..=seg {
        let a = i as f32 / seg as f32 * TAU;
        let (s, c) = a.sin_cos();
        // Outward horizontal normal (taper is gentle; horizontal is close enough).
        let n = Vec3::new(c, 0.0, s);
        // Trunk darkens slightly toward the base. Bark cell, base/top corners
        // alternate per segment so the swatch maps across each quad.
        let base_col = params.trunk_color * 0.85;
        let (cb, ct) = if i & 1 == 0 { (0, 3) } else { (1, 2) };
        b.push_vert(
            Vec3::new(c * base_r, 0.0, s * base_r),
            n,
            base_col,
            0.0,
            leaf_code(bark, cb),
        );
        b.push_vert(
            Vec3::new(c * top_r, h, s * top_r),
            n,
            params.trunk_color,
            0.05,
            leaf_code(bark, ct),
        );
    }

    for i in 0..seg {
        let r0 = start + i * 2;
        let r1 = start + (i + 1) * 2;
        // (base_i, top_i, base_i+1) and (base_i+1, top_i, top_i+1)
        b.indices.extend_from_slice(&[r0, r1, r0 + 1, r1, r1 + 1, r0 + 1]);
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
        push_skin_card(b, params, center, rx, ry, rz, crown_base, crown_top, s, dir, size);
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
                b, params, center, crown_base, crown_top, lobe_center, lobe_r, cs, size, 0.12, 1.0,
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
            b, params, center, crown_base, crown_top, center, rx * 0.6, cs, size, 0.05, 0.85, false,
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
    let ao = 0.52 + 0.4 * h01;
    let wind = (0.3 + 0.7 * h01).clamp(0.0, 1.0);
    let cell = LEAF_CELL_FIRST + (hash_u(seed, 12) % LEAF_CELL_COUNT.max(1));
    push_leaf_card(b, pos, face, n_light, size, roll, cell, params.canopy_color * ao, wind);
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
    let ao = (0.4 + 0.6 * h01) * ao_scale;
    let wind = (0.3 + 0.7 * h01).clamp(0.0, 1.0);
    let cell = LEAF_CELL_FIRST + (hash_u(seed, 12) % LEAF_CELL_COUNT.max(1));
    push_leaf_card(b, pos, face, n_light, size, roll, cell, params.canopy_color * ao, wind);
}

// ---------------------------------------------------------------------------
// Broadleaf: dense volumetric crown of small foliage puffs (no visible branches)
// ---------------------------------------------------------------------------

/// Broadleaf crown: a **cluster of distinct rounded florets** (sub-crowns), not a
/// single ball or a branched skeleton. Each floret is its own dense little
/// puff-ball placed to *bulge separately*, so the crown reads as broccoli-like
/// lumps with structure instead of one uniform speckled mass. Each floret shades
/// as its own soft dome (its crown catches light, its flanks a touch less — but
/// floored well above black, so there's **no dark cavity inside**), and a
/// per-floret brightness jitter gives tonal variety between neighbouring lumps.
/// The branch armature is intentionally *not* drawn — only the trunk shows at the
/// base; the florets pack densely enough to hide the limbs and the sky.
fn push_broadleaf_canopy(b: &mut TreeMeshData, params: &TreeMeshParams) {
    let rx = params.canopy_radius_m;
    let ry = params.canopy_height_m;
    let rz = rx;
    // Crown sits low on the trunk (so the trunk is clearly visible at the base).
    let crown_base = params.trunk_height_m * 0.55;
    let center = Vec3::new(0.0, crown_base + ry * 0.92, 0.0);

    // (florets, surface puffs per floret, gap-fill puffs, cards per puff, card m).
    // The surface is deliberately DENSE so you can't see into the (darker)
    // interior through gaps — the up-close "dark inside" came from too few/thin
    // surface cards letting the shaded core show through.
    let (n_florets, surf_per, fill, puff_cards, card_sz) = match params.lod {
        0 => (8u32, 20u32, 6u32, 4u32, 1.15f32),
        1 => (5, 12, 2, 3, 1.35),
        2 => (3, 7, 0, 2, 1.7),
        _ => (1, 5, 0, 1, 2.2),
    };

    for f in 0..n_florets {
        let fs = params.seed ^ (f as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ 0xB07;
        // Floret 0 fills the core; the rest sit in an upper DOME around it (lifted
        // out of the lower hemisphere so they don't droop into awkward side-lobes
        // and don't expose the underside/interior).
        let (fc, fr) = if f == 0 {
            (center, rx * 0.55)
        } else {
            let mut dir = fib_dir(f, n_florets, fs);
            dir.y = dir.y * 0.55 + 0.28; // bias florets into the upper dome
            // Pulled in / larger so neighbouring florets overlap — closes the dark
            // gaps you could otherwise see into between them.
            let spread = 0.42 + 0.18 * hash01(fs, 1);
            let fcenter =
                center + Vec3::new(dir.x * rx * spread, dir.y * ry, dir.z * rz * spread);
            (fcenter, rx * (0.42 + 0.12 * hash01(fs, 2)))
        };
        // Per-floret tone: some lumps read brighter, some duller (kept high — never
        // dark) so the crown has variation without any black recesses.
        let bright = 0.86 + 0.14 * hash01(fs, 3);

        for i in 0..surf_per {
            let s = fs ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let dir = fib_dir(i, surf_per, s);
            // Hug the floret surface (small protrusion) so its outline is a clean
            // rounded lump, not stray tufts.
            let bump = 0.93 + 0.09 * hash01(s, 9);
            let pos = fc + dir * fr * bump;
            let n_out = dir.normalize_or(Vec3::Y);
            // Floret-local vertical form gives each lump roundness; floor ~0.8 so
            // even shaded flanks stay a bright leafy green, never dark.
            let form = dir.y * 0.5 + 0.5;
            let ao = bright * (0.86 + 0.14 * form);
            let wind = (0.35 + 0.55 * ((pos.y - crown_base) / (ry * 2.0)).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            let puff_r = fr * (0.20 + 0.08 * hash01(s, 10));
            push_puff(b, params, pos, n_out, puff_r, card_sz, ao, wind, puff_cards, s);
        }
    }

    // A few interior puffs ONLY to plug stray sky lines between florets — kept
    // bright, so even where one peeks through it reads as foliage, not a dark hole.
    for i in 0..fill {
        let s = params.seed ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407) ^ 0xF111;
        let dir = rand_unit(s, 1);
        let rr = 0.35 + 0.45 * hash01(s, 2);
        let pos = center
            + Vec3::new(dir.x * rx * 0.6 * rr, dir.y * ry * 0.5 * rr, dir.z * rz * 0.6 * rr);
        let up = (dir + Vec3::Y * 0.5).normalize_or(Vec3::Y); // face up → catches sky
        let cards = puff_cards.saturating_sub(1).max(1);
        push_puff(b, params, pos, up, rx * 0.18, card_sz, 0.72, 0.5, cards, s);
    }
}

/// One foliage **puff**: a tight little cluster of a few small leaf cards jittered
/// around `center`, all carrying the outward `normal` so the puff reads as a
/// single rounded dome — the broccoli/cauliflower bump the broadleaf crown is
/// built from. `puff_r` is the cluster's local spread, `card` the card size.
#[allow(clippy::too_many_arguments)]
fn push_puff(
    b: &mut TreeMeshData,
    params: &TreeMeshParams,
    center: Vec3,
    normal: Vec3,
    puff_r: f32,
    card: f32,
    ao: f32,
    wind: f32,
    cards: u32,
    seed: u64,
) {
    let n_out = normal.normalize_or(Vec3::Y);
    for i in 0..cards {
        let s = seed ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ 0x9F1;
        let jitter = rand_unit(s, 1) * puff_r * (0.3 + 0.6 * hash01(s, 2));
        let pos = center + jitter;
        // Cards face out from the puff centre (scattered), but light from the puff
        // outward normal dappled per card so the bump shades like a soft dome.
        let face = (n_out + rand_unit(s, 3) * 0.5).normalize_or(n_out);
        // Lift the lighting normal slightly toward the sky so even inward/flank
        // cards pick up the blue-sky ambient and don't read as dark interior.
        let n_light = (n_out + Vec3::Y * 0.22 + rand_unit(s, 4) * 0.26).normalize_or(n_out);
        let roll = hash01(s, 5) * TAU;
        let size = card * (0.78 + 0.5 * hash01(s, 6));
        let cell = LEAF_CELL_FIRST + (hash_u(s, 7) % LEAF_CELL_COUNT.max(1));
        push_leaf_card(b, pos, face, n_light, size, roll, cell, params.canopy_color * ao, wind);
    }
}

/// One leaf-cluster card: a quad centred at `pos`, in the plane spanned by a
/// `face`-derived (rolled) basis, but carrying `light_normal` (the crown-outward
/// normal) so it shades as part of a soft sphere. Two-sided (the material culls
/// nothing). Corners map to the atlas cell's four UV corners.
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
) {
    let up_ref = if face.y.abs() > 0.95 { Vec3::X } else { Vec3::Y };
    let t = face.cross(up_ref).normalize_or(Vec3::X);
    let bi = face.cross(t).normalize_or(Vec3::Z);
    let (sr, cr) = roll.sin_cos();
    let tr = t * cr + bi * sr;
    let br = -t * sr + bi * cr;
    let h = size * 0.5;
    let corners = [
        pos - tr * h - br * h,
        pos + tr * h - br * h,
        pos + tr * h + br * h,
        pos - tr * h + br * h,
    ];
    let start = b.positions.len() as u32;
    for (k, p) in corners.iter().enumerate() {
        b.push_vert(*p, light_normal, tint, wind, leaf_code(cell, k as u32));
    }
    b.indices
        .extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
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
    let mut h = seed
        ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ 0x2545_F491_4F6C_DD1D;
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
