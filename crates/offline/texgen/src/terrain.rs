//! Terrain material set (NTR-X7 P2) — the tiling PBR materials the tile
//! renderer's splat stack samples at contact scale.
//!
//! This is the standard shipped-terrain shape (Decima / RAGE / Frostbite /
//! Dunia, and MSFS below imagery resolution): a small set of *tiling* material
//! layers, each carrying albedo + normal + roughness + AO + **height**, blended
//! by weights the shader derives from slope / altitude / landcover. The height
//! channel is what makes the blend interlock instead of cross-fade (gravel
//! pokes through grass along a broken edge rather than ghosting into it).
//!
//! ## Division of labour with the macro palette
//!
//! These textures carry **material identity, not colour**. Each layer's albedo
//! is normalised at bake time so its **per-channel mean is exactly 0.5**, which
//! strips the layer's average hue and leaves only its *spatial* variation. The
//! shader then applies `macro_albedo * (tex_rgb / 0.5)`, so:
//!
//!   - the canonical palette (`procedural.rs::albedo_from_bands` + the climate
//!     model) keeps sole authority over what colour a region is — the
//!     shared-library rule, which an absolute albedo baked in here would fork;
//!   - the texture supplies the value structure, the grain, and the local
//!     chroma break-up that no palette can carry.
//!
//! Because the mean is exact, a layer swap can never shift regional colour —
//! the normalisation is the contract, not a tuning knob.
//!
//! ## Layout
//!
//! Two PNGs, each `TERRAIN_TILE_PX` wide and `TERRAIN_TILE_PX * TERRAIN_LAYERS`
//! tall — layers stacked top to bottom, ready for Bevy's
//! `reinterpret_stacked_2d_as_array`:
//!
//! | asset | ch | meaning |
//! |---|---|---|
//! | `terrain_albedo_array` (sRGB) | RGB | mean-normalised albedo |
//! | | A | **height**, for height-based blending |
//! | `terrain_material_array` (linear) | RG | tangent normal XY, 0.5-centred |
//! | | B | perceptual roughness |
//! | | A | ambient occlusion (cavity) |
//!
//! Every layer is **toroidally periodic** at `TERRAIN_TILE_PX`, so it tiles
//! seamlessly; the shader's job is to hide the *repetition*, not the seam.

use crate::{GOLD, TAU, TextureData, hash_to_unit, smooth01};

/// Ground covered by one tile of a material layer, in metres. The shader
/// mirrors this constant to turn body-fixed position into layer UVs; changing
/// it changes the apparent grain size of every material at once.
pub const TERRAIN_TILE_M: f32 = 8.0;

/// Texels per axis in one material layer.
pub const TERRAIN_TILE_PX: u32 = 512;

/// Layers in the array, in index order. The shader's class indices mirror this.
pub const TERRAIN_LAYERS: u32 = 6;

pub const LAYER_MEADOW: u32 = 0;
pub const LAYER_SOIL: u32 = 1;
pub const LAYER_ROCK: u32 = 2;
pub const LAYER_SCREE: u32 = 3;
pub const LAYER_SNOW: u32 = 4;
pub const LAYER_LITTER: u32 = 5;

/// Per-channel albedo mean every layer is normalised to. The shader divides by
/// it; the pair is the contract described in the module docs.
pub const ALBEDO_MEAN: f32 = 0.5;

/// How much of a layer's *chroma* variation survives normalisation.
///
/// Normalising each channel to a common mean is mean-neutral but it is **not**
/// chroma-neutral: it divides each channel by its own mean, so a saturated base
/// (grass at linear `(0.048, 0.082, 0.030)`) gets per-channel gains that differ
/// by ~3×, and every small authored hue difference is stretched by that ratio.
/// The first bake showed exactly that — meadow came out as pink-and-green
/// confetti, because a modest lush/straw split became a 2× swing on blue once
/// the blue channel was scaled up 17×.
///
/// So after normalisation each texel is pulled back toward its own luminance,
/// keeping only this fraction of the hue excursion. That is the module's stated
/// division of labour made literal: value structure and grain survive intact,
/// hue stays with the macro palette.
const CHROMA_KEEP: f32 = 0.22;

const N: usize = TERRAIN_TILE_PX as usize;
const TEXEL_M: f32 = TERRAIN_TILE_M / TERRAIN_TILE_PX as f32;

/// The baked pair. Generated together because both derive from the same
/// per-layer height field — splitting them would synthesise it twice and risk
/// the normal disagreeing with the albedo's cavities.
pub struct TerrainMaterialSet {
    /// sRGB: RGB = mean-normalised albedo, A = height.
    pub albedo: TextureData,
    /// Linear: RG = normal XY, B = roughness, A = AO.
    pub material: TextureData,
}

/// Generate the whole material set. Deterministic.
pub fn terrain_material_set() -> TerrainMaterialSet {
    let stride = N * N;
    let mut albedo = vec![0u8; stride * TERRAIN_LAYERS as usize * 4];
    let mut material = vec![0u8; stride * TERRAIN_LAYERS as usize * 4];

    for layer in 0..TERRAIN_LAYERS {
        let mut fields = synth_layer(layer);
        normalise_height(&mut fields.h);
        normalise_albedo(&mut fields.albedo);
        let (normal, ao) = derive_normal_ao(&fields);

        let base = layer as usize * stride * 4;
        for i in 0..stride {
            let c = fields.albedo[i];
            albedo[base + i * 4] = to_u8(linear_to_srgb(c[0]));
            albedo[base + i * 4 + 1] = to_u8(linear_to_srgb(c[1]));
            albedo[base + i * 4 + 2] = to_u8(linear_to_srgb(c[2]));
            // Height is data, not colour — no transfer function.
            albedo[base + i * 4 + 3] = to_u8(fields.h[i]);

            material[base + i * 4] = to_u8(normal[i][0] * 0.5 + 0.5);
            material[base + i * 4 + 1] = to_u8(normal[i][1] * 0.5 + 0.5);
            material[base + i * 4 + 2] = to_u8(fields.rough[i]);
            material[base + i * 4 + 3] = to_u8(ao[i]);
        }
    }

    let height = TERRAIN_TILE_PX * TERRAIN_LAYERS;
    TerrainMaterialSet {
        albedo: TextureData {
            width: TERRAIN_TILE_PX,
            height,
            rgba: albedo,
        },
        material: TextureData {
            width: TERRAIN_TILE_PX,
            height,
            rgba: material,
        },
    }
}

// ── Per-layer synthesis ────────────────────────────────────────────────────

struct LayerFields {
    /// Raw height, normalised to 0..1 before use.
    h: Vec<f32>,
    /// Linear albedo, mean-normalised before use.
    albedo: Vec<[f32; 3]>,
    rough: Vec<f32>,
    /// Peak-to-peak relief in metres — sets normal-map strength against
    /// [`TEXEL_M`], so a material's bumpiness is authored in world units
    /// rather than as an arbitrary "normal strength" slider.
    relief_m: f32,
    /// Cavity darkening at full occlusion (0 = no AO, 1 = black cavities).
    ao_strength: f32,
    /// Radius of the cavity comparison, in texels.
    ao_radius: usize,
}

fn synth_layer(layer: u32) -> LayerFields {
    match layer {
        LAYER_MEADOW => meadow(),
        LAYER_SOIL => soil(),
        LAYER_ROCK => rock(),
        LAYER_SCREE => scree(),
        LAYER_SNOW => snow(),
        _ => litter(),
    }
}

/// Domain warp — the difference between "a Voronoi diagram" and "geology".
///
/// Cellular features queried on a clean lattice read as a tiled mosaic no
/// matter how they are shaded; the first bake made that vivid (soil and rock
/// both came out as ceramic tiling). Displacing the lookup by a couple of
/// octaves of noise *before* the cell query bends the cell walls into
/// something that looks grown rather than tessellated. The warp field is
/// itself periodic, so the layer stays seamless.
fn warp(u: f32, v: f32, cells: u32, amount: f32, seed: u64) -> (f32, f32) {
    let du = fbm2(u, v, cells, cells, 2, seed) - 0.5;
    let dv = fbm2(u, v, cells, cells, 2, seed ^ 0xB0_5E) - 0.5;
    (u + amount * du, v + amount * dv)
}

/// Grass sward: tussock clumps over fine blade clutter. Read from 1–20 m, so
/// what matters is the clump rhythm (~20 cm) and the tone break-up, not
/// individual blades.
fn meadow() -> LayerFields {
    let mut f = fields(0.030, 0.50, 10);
    for (i, (u, v)) in texels().enumerate() {
        let (wu, wv) = warp(u, v, 10, 0.055, 0x51_E0);
        let tuft = worley2(wu, wv, 38, 38, 0x51EE_D);
        let fine = worley2(wu, wv, 92, 92, 0x51EE_E);
        let mound = (1.0 - smooth01(tuft.f1 / 0.70)) * (0.55 + 0.45 * tuft.id);
        let fine_mound = 1.0 - smooth01(fine.f1 / 0.74);
        let clutter = fbm2(u, v, 110, 110, 3, 0x9B1_A5);
        // Anisotropic lattice, not scaled coordinates: blades lie over rather
        // than standing in a grid, and both axes stay periodic.
        let lie = fbm2(u, v, 44, 260, 2, 0x77E_31);
        f.h[i] = 0.44 * mound + 0.20 * fine_mound + 0.22 * clutter + 0.14 * lie;

        // Lush / straw break-up. Authored gently — `CHROMA_KEEP` damps hue
        // afterwards, so what survives here is mostly the value difference,
        // which is the part that actually reads as patchy sward.
        let patch = fbm2(u, v, 5, 5, 2, 0x2C0_17);
        let dry = smooth01((patch * 0.7 + tuft.id * 0.3 - 0.30) / 0.46);
        let mut c = lerp3([0.052, 0.076, 0.036], [0.104, 0.098, 0.058], dry);
        // Gaps between tussocks show shaded thatch, not more grass.
        c = lerp3(c, [0.032, 0.033, 0.023], (1.0 - mound) * 0.5);
        let tone = 0.84 + 0.30 * tuft.id + 0.14 * (lie - 0.5);
        f.albedo[i] = [c[0] * tone, c[1] * tone, c[2] * tone];
        f.rough[i] = 0.90 + 0.07 * clutter;
    }
    f
}

/// Bare earth: clods and shrinkage cracks under scattered grit. The layer the
/// plain needs most — without exposed soil, vegetated ground has nothing to
/// break against.
fn soil() -> LayerFields {
    let mut f = fields(0.040, 0.65, 12);
    for (i, (u, v)) in texels().enumerate() {
        let (wu, wv) = warp(u, v, 8, 0.13, 0x50_10);
        let clod = worley2(wu, wv, 30, 30, 0x50_1D);
        let lump = 1.0 - smooth01(clod.f1 / 0.82);
        // Shrinkage cracks ride the cell boundaries (`f2 - f1` small there),
        // broken by grit so they are fissures rather than a drawn net.
        let grit = fbm2(u, v, 170, 170, 3, 0xA11_CE);
        let edge = (clod.f2 - clod.f1) + 0.10 * (grit - 0.5);
        let crack = 1.0 - smooth01(edge / 0.13);
        f.h[i] = (0.58 * lump * (0.55 + 0.45 * clod.id) + 0.24 * grit) - 0.30 * crack;

        let wet = fbm2(u, v, 7, 7, 2, 0x3D_A7);
        let mut c = lerp3(
            [0.096, 0.070, 0.046],
            [0.052, 0.038, 0.026],
            smooth01((wet - 0.42) / 0.36),
        );
        let tone = 0.82 + 0.34 * clod.id + 0.12 * (grit - 0.5);
        c = [c[0] * tone, c[1] * tone, c[2] * tone];
        c = lerp3(c, [0.024, 0.018, 0.013], crack * 0.7);
        // Occasional pale grit — neutral, and only on clod crowns.
        if clod.id > 0.93 {
            c = lerp3(c, [0.118, 0.112, 0.102], lump * 0.7);
        }
        f.albedo[i] = c;
        f.rough[i] = 0.93 + 0.05 * grit;
    }
    f
}

/// Bedrock: jointed rock with bedding and a crystalline grain. Joints are the
/// readable feature at every distance a cliff is seen from, so they carry the
/// relief — the *plates between them* deliberately do not, because a per-cell
/// height offset is what turns jointed rock into floor tiles.
fn rock() -> LayerFields {
    let mut f = fields(0.14, 0.80, 16);
    for (i, (u, v)) in texels().enumerate() {
        // Heavy warp, and two fracture generations at different scales — real
        // rock is jointed more than once, and a single set reads as a lattice.
        let (wu, wv) = warp(u, v, 5, 0.22, 0x4A_10);
        let major = worley2(wu, wv, 7, 7, 0x20C_C1);
        let (mu, mv) = warp(u, v, 13, 0.10, 0x4A_20);
        let minor = worley2(mu, mv, 19, 19, 0x20C_C2);
        let grain = fbm2(u, v, 140, 140, 3, 0x1C2_9E);
        let major_joint = 1.0 - smooth01(((major.f2 - major.f1) + 0.08 * (grain - 0.5)) / 0.24);
        let minor_joint = 1.0 - smooth01(((minor.f2 - minor.f1) + 0.06 * (grain - 0.5)) / 0.18);
        // Bedding as an anisotropic lattice (bands across one axis), not as a
        // scaled coordinate — scaling would break the tile's periodicity.
        let bedding = fbm2(u, v, 4, 34, 2, 0x8ED_00);
        f.h[i] = 0.34 * bedding + 0.30 * grain + 0.14 * major.id
            - 0.58 * major_joint
            - 0.26 * minor_joint;

        let joint = major_joint.max(minor_joint * 0.7);
        let mut c = lerp3([0.088, 0.089, 0.092], [0.100, 0.092, 0.079], major.id);
        let tone = 0.80 + 0.40 * bedding + 0.10 * (grain - 0.5);
        c = [c[0] * tone, c[1] * tone, c[2] * tone];
        // Iron staining seeps from the joints, not uniformly along them.
        let stain = joint * smooth01((fbm2(u, v, 11, 11, 2, 0x5EE_D1) - 0.44) / 0.36);
        c = lerp3(c, [0.086, 0.060, 0.036], stain * 0.5);
        c = lerp3(c, [0.032, 0.031, 0.032], joint * 0.5);
        f.albedo[i] = c;
        // Fresh joint faces are rougher than weathered rock.
        f.rough[i] = 0.70 + 0.22 * joint + 0.06 * grain;
    }
    f
}

/// Talus: packed angular fragments. Nearly all relief, almost no colour — the
/// look comes from the size distribution and the shadowed gaps between blocks.
fn scree() -> LayerFields {
    let mut f = fields(0.10, 0.88, 12);
    for (i, (u, v)) in texels().enumerate() {
        let (wu, wv) = warp(u, v, 9, 0.07, 0x5C_E0);
        // Two fragment sizes so the distribution is not monodisperse.
        let big = worley2(wu, wv, 20, 20, 0x5C_E1);
        let small = worley2(wu, wv, 54, 54, 0x5C_E2);
        let big_lump = (1.0 - smooth01(big.f1 / 0.84)) * (0.45 + 0.55 * big.id);
        let small_lump = (1.0 - smooth01(small.f1 / 0.86)) * (0.4 + 0.6 * small.id);
        let coarse = big.id > 0.45;
        let lump = if coarse {
            big_lump.max(0.45 * small_lump)
        } else {
            small_lump
        };
        f.h[i] = 0.72 * lump + 0.16 * fbm2(u, v, 190, 190, 2, 0xB10_0D);

        let frag_id = if coarse { big.id } else { small.id };
        let base = [0.118f32, 0.115, 0.108];
        let tone = 0.72 + 0.56 * frag_id;
        f.albedo[i] = [base[0] * tone, base[1] * tone, base[2] * tone];
        f.rough[i] = 0.94 + 0.05 * frag_id;
    }
    f
}

/// Wind-worked snow: sastrugi ridges over a granular surface. Low roughness is
/// the point — it is the one material out here that is allowed to be shiny.
fn snow() -> LayerFields {
    let mut f = fields(0.075, 0.30, 18);
    for (i, (u, v)) in texels().enumerate() {
        // Strongly anisotropic: few lattice cells along the wind, many across,
        // warped along-wind so ridges taper instead of running as stripes.
        let (wu, wv) = warp(u, v, 6, 0.09, 0x5_0F0);
        let ripple = fbm2(wu, wv, 3, 30, 3, 0x5_0F1);
        let sastrugi = 1.0 - (2.0 * ripple - 1.0).abs(); // ridged: sharp crests
        let drift = fbm2(u, v, 3, 3, 2, 0x5_0F2);
        let grain = fbm2(u, v, 230, 230, 2, 0x5_0F3);
        f.h[i] = 0.52 * sastrugi + 0.34 * drift + 0.14 * grain;

        // Snow's colour variation is almost entirely scattered skylight in the
        // hollows; the flats are neutral.
        let hollow = 1.0 - smooth01((f.h[i] - 0.22) / 0.5);
        f.albedo[i] = lerp3([0.660, 0.670, 0.684], [0.556, 0.582, 0.638], hollow * 0.75);
        // Wind-packed crests glaze; hollows stay powdery.
        f.rough[i] = 0.26 + 0.32 * hollow + 0.08 * grain;
    }
    f
}

/// Forest floor: leaf litter over humus, with moss taking the damp patches.
/// Leaves are elongated cells crossed at two angles — cheap, and it reads as
/// overlapping litter rather than as a bump field.
fn litter() -> LayerFields {
    let mut f = fields(0.045, 0.68, 12);
    for (i, (u, v)) in texels().enumerate() {
        let (wu, wv) = warp(u, v, 7, 0.06, 0x1EA_F0);
        // Fewer, larger leaves than the first bake, which was fine enough to
        // alias into static rather than reading as litter.
        let a = worley2(wu, wv, 14, 44, 0x1EA_F1);
        let b = worley2(wv, wu, 15, 47, 0x1EA_F2);
        let leaf_a = 1.0 - smooth01(a.f1 / 0.80);
        let leaf_b = 1.0 - smooth01(b.f1 / 0.80);
        let (leaf, top) = if leaf_a >= leaf_b {
            (leaf_a, a.id)
        } else {
            (leaf_b, b.id)
        };
        let humus = fbm2(u, v, 46, 46, 3, 0x40_C8);
        f.h[i] = 0.60 * leaf + 0.28 * humus + 0.12 * fbm2(u, v, 150, 150, 2, 0x40_C9);

        let litter_c = lerp3([0.082, 0.056, 0.030], [0.126, 0.088, 0.044], top);
        let mut c = lerp3([0.028, 0.023, 0.017], litter_c, smooth01(leaf / 0.6));
        // Moss claims the damp low ground between the leaves.
        let damp = fbm2(u, v, 8, 8, 2, 0x40_CA);
        let moss = smooth01((damp - 0.56) / 0.26) * (1.0 - leaf * 0.7);
        c = lerp3(c, [0.036, 0.058, 0.028], moss * 0.75);
        f.albedo[i] = c;
        f.rough[i] = 0.88 + 0.09 * humus;
    }
    f
}

// ── Shared post-processing ─────────────────────────────────────────────────

fn fields(relief_m: f32, ao_strength: f32, ao_radius: usize) -> LayerFields {
    LayerFields {
        h: vec![0.0; N * N],
        albedo: vec![[0.0; 3]; N * N],
        rough: vec![0.0; N * N],
        relief_m,
        ao_strength,
        ao_radius,
    }
}

/// Every texel of one layer, as `(u, v)` in `[0, 1)` tile space, row-major.
fn texels() -> impl Iterator<Item = (f32, f32)> {
    (0..N).flat_map(|y| {
        (0..N).map(move |x| (x as f32 / N as f32, y as f32 / N as f32))
    })
}

fn normalise_height(h: &mut [f32]) {
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for &v in h.iter() {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let span = (hi - lo).max(1.0e-5);
    for v in h.iter_mut() {
        *v = (*v - lo) / span;
    }
}

/// Force each channel's mean to [`ALBEDO_MEAN`]. This is the contract that
/// keeps hue authority with the macro palette (see the module docs), so it is
/// exact rather than approximate — and it is why the authored colours above
/// only ever matter through their *variation*.
///
/// A plain `gain = target / mean` does **not** hold that contract: the bright
/// layers (snow, scree) push texels past 1, and clipping them pulls the
/// realised mean back below target — silently, and by more the brighter the
/// layer, which is exactly the case where a colour shift would be visible. So
/// the gain is solved against the *compressed* result rather than assumed:
/// [`soft_clip`] rolls the top off smoothly instead of clipping it (keeping the
/// highlight detail a clamp would flatten), and the gain is then iterated until
/// the realised mean lands on target.
fn normalise_albedo(albedo: &mut [[f32; 3]]) {
    let n = albedo.len() as f32;

    // Chroma damping first, in ratio space, so the mean solve below sees the
    // values that will actually be written. See `CHROMA_KEEP`.
    let mean = [0, 1, 2].map(|ch| {
        (albedo.iter().map(|c| c[ch]).sum::<f32>() / n).max(1.0e-5)
    });
    for c in albedo.iter_mut() {
        let ratio = [c[0] / mean[0], c[1] / mean[1], c[2] / mean[2]];
        let lum = 0.2126 * ratio[0] + 0.7152 * ratio[1] + 0.0722 * ratio[2];
        for ch in 0..3 {
            c[ch] = (lum + (ratio[ch] - lum) * CHROMA_KEEP) * mean[ch];
        }
    }

    for ch in 0..3 {
        let mean: f32 = albedo.iter().map(|c| c[ch]).sum::<f32>() / n;
        let mut gain = ALBEDO_MEAN / mean.max(1.0e-5);
        // Converges in a handful of steps; the bound is a backstop for a
        // pathological layer (one where over half the texels saturate), which
        // would be an authoring bug worth seeing rather than silently fitting.
        for _ in 0..32 {
            let realised: f32 =
                albedo.iter().map(|c| soft_clip(c[ch] * gain)).sum::<f32>() / n;
            if (realised - ALBEDO_MEAN).abs() < 1.0e-4 {
                break;
            }
            gain *= ALBEDO_MEAN / realised.max(1.0e-5);
        }
        for c in albedo.iter_mut() {
            c[ch] = soft_clip(c[ch] * gain);
        }
    }
}

/// Monotone soft knee: identity below [`SOFT_KNEE`], asymptotic to 1 above it.
/// C¹ at the knee, so it introduces no visible break in a gradient.
fn soft_clip(v: f32) -> f32 {
    const SOFT_KNEE: f32 = 0.75;
    let v = v.max(0.0);
    if v <= SOFT_KNEE {
        v
    } else {
        let head = 1.0 - SOFT_KNEE;
        SOFT_KNEE + head * (1.0 - (-(v - SOFT_KNEE) / head).exp())
    }
}

/// Tangent-space normal from the height gradient, and a cavity AO from the
/// height's departure from its local mean. Both wrap, so the layer stays
/// seamless.
fn derive_normal_ao(f: &LayerFields) -> (Vec<[f32; 2]>, Vec<f32>) {
    let mut normal = vec![[0.0f32; 2]; N * N];
    // Height slope in world units: normalised height spans `relief_m` metres
    // over `TEXEL_M` per texel, so the gradient is dimensionless as it should
    // be — bumpiness is authored in metres, not in arbitrary strength units.
    let scale = f.relief_m / TEXEL_M;
    for y in 0..N {
        for x in 0..N {
            let l = f.h[y * N + (x + N - 1) % N];
            let r = f.h[y * N + (x + 1) % N];
            let d = f.h[((y + N - 1) % N) * N + x];
            let u = f.h[((y + 1) % N) * N + x];
            let dx = (r - l) * 0.5 * scale;
            let dy = (u - d) * 0.5 * scale;
            // normalize(vec3(-dx, -dy, 1)), storing xy only.
            let inv = 1.0 / (dx * dx + dy * dy + 1.0).sqrt();
            normal[y * N + x] = [-dx * inv, -dy * inv];
        }
    }

    let blurred = box_blur_wrapped(&f.h, f.ao_radius);
    let ao = f
        .h
        .iter()
        .zip(blurred.iter())
        .map(|(h, b)| {
            // Below the local mean = cavity. The 0.5 scale keeps a flat
            // material at AO 1 rather than at a grey wash.
            let cavity = ((b - h) / 0.5).clamp(0.0, 1.0);
            1.0 - f.ao_strength * cavity
        })
        .collect();
    (normal, ao)
}

/// Separable box blur with toroidal wrap.
fn box_blur_wrapped(src: &[f32], radius: usize) -> Vec<f32> {
    let width = (radius * 2 + 1) as f32;
    let mut tmp = vec![0.0f32; N * N];
    for y in 0..N {
        for x in 0..N {
            let mut sum = 0.0;
            for k in 0..=(radius * 2) {
                sum += src[y * N + (x + N + k - radius) % N];
            }
            tmp[y * N + x] = sum / width;
        }
    }
    let mut out = vec![0.0f32; N * N];
    for y in 0..N {
        for x in 0..N {
            let mut sum = 0.0;
            for k in 0..=(radius * 2) {
                sum += tmp[((y + N + k - radius) % N) * N + x];
            }
            out[y * N + x] = sum / width;
        }
    }
    out
}

// ── Periodic noise primitives ──────────────────────────────────────────────

/// Periodic 2D gradient (Perlin) noise on a `cx × cy` lattice over the unit
/// tile. Gradient rather than value noise for the same reason the WGSL side
/// uses it: value-noise derivatives are axis-aligned and show the lattice as a
/// weave in any normal built from them.
///
/// Separate `cx`/`cy` give anisotropy (wind ripples, bedding) for free while
/// keeping both axes periodic.
fn grad2(x: f32, y: f32, cx: u32, cy: u32, seed: u64) -> f32 {
    let (px, py) = (cx.max(1) as f32, cy.max(1) as f32);
    let (xs, ys) = (x * px, y * py);
    let (xi, yi) = (xs.floor(), ys.floor());
    let (fx, fy) = (xs - xi, ys - yi);
    let g = |gx: f32, gy: f32, dx: f32, dy: f32| {
        let wx = (gx as i64).rem_euclid(px as i64) as u64;
        let wy = (gy as i64).rem_euclid(py as i64) as u64;
        let h = wx
            .wrapping_mul(GOLD)
            .wrapping_add(wy.wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
            .wrapping_add(seed.wrapping_mul(0x1656_67B1_9E37_79F9));
        let a = hash_to_unit(h) * TAU;
        dx * a.cos() + dy * a.sin()
    };
    let (u, v) = (smooth01(fx), smooth01(fy));
    let n0 = {
        let a = g(xi, yi, fx, fy);
        let b = g(xi + 1.0, yi, fx - 1.0, fy);
        a + u * (b - a)
    };
    let n1 = {
        let a = g(xi, yi + 1.0, fx, fy - 1.0);
        let b = g(xi + 1.0, yi + 1.0, fx - 1.0, fy - 1.0);
        a + u * (b - a)
    };
    (n0 + v * (n1 - n0)) * 1.4
}

/// fBm over [`grad2`], returned in `[0, 1]`. Each octave doubles both lattice
/// counts, so periodicity survives every octave.
fn fbm2(x: f32, y: f32, cx: u32, cy: u32, octaves: u32, seed: u64) -> f32 {
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut amp = 0.5;
    let (mut ax, mut ay) = (cx, cy);
    for o in 0..octaves {
        sum += amp * grad2(x, y, ax, ay, seed.wrapping_add(o as u64 * 7919));
        norm += amp;
        ax = ax.saturating_mul(2);
        ay = ay.saturating_mul(2);
        amp *= 0.5;
    }
    (sum / norm.max(1.0e-5) * 0.5 + 0.5).clamp(0.0, 1.0)
}

struct Cell {
    /// Distance to the nearest feature point, in lattice-cell units.
    f1: f32,
    /// Distance to the second nearest — `f2 - f1` is the cell-boundary ridge
    /// (joints, shrinkage cracks).
    f2: f32,
    /// Stable per-cell hash in `[0, 1)`.
    id: f32,
}

/// Periodic 2D Worley/cellular noise on a `cx × cy` lattice. Distances are
/// measured in lattice-index space, so an anisotropic lattice yields elongated
/// cells (used for leaf litter).
fn worley2(x: f32, y: f32, cx: u32, cy: u32, seed: u64) -> Cell {
    let (px, py) = (cx.max(1) as i32, cy.max(1) as i32);
    let (xs, ys) = (x * px as f32, y * py as f32);
    let (xi, yi) = (xs.floor() as i32, ys.floor() as i32);
    let (mut f1, mut f2) = (f32::MAX, f32::MAX);
    let mut id = 0.0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (gx, gy) = (xi + dx, yi + dy);
            let wx = gx.rem_euclid(px) as u64;
            let wy = gy.rem_euclid(py) as u64;
            let h = wx
                .wrapping_mul(GOLD)
                .wrapping_add(wy.wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
                .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let fx = gx as f32 + hash_to_unit(h);
            let fy = gy as f32 + hash_to_unit(h ^ 0xA5A5_5A5A_1234_9876);
            let d = ((fx - xs).powi(2) + (fy - ys).powi(2)).sqrt();
            if d < f1 {
                f2 = f1;
                f1 = d;
                id = hash_to_unit(h ^ 0x1357_9BDF_2468_ACE0);
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    Cell { f1, f2, id }
}

// ── Small helpers ──────────────────────────────────────────────────────────

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_are_periodic() {
        // A seam would tile visibly; assert the primitives wrap exactly.
        for (cx, cy) in [(9u32, 9u32), (4, 26), (20, 74)] {
            let a = grad2(0.0, 0.37, cx, cy, 1234);
            let b = grad2(1.0, 0.37, cx, cy, 1234);
            assert!((a - b).abs() < 1.0e-5, "grad2 seam at {cx}x{cy}");
            let c = worley2(0.61, 0.0, cx, cy, 99);
            let d = worley2(0.61, 1.0, cx, cy, 99);
            assert!((c.f1 - d.f1).abs() < 1.0e-5, "worley2 seam at {cx}x{cy}");
        }
    }

    /// The mean is the contract the shader divides by, so it has to hold even
    /// for a layer bright enough that the naive gain would push texels past 1
    /// — the case a plain clamp silently got wrong.
    #[test]
    fn albedo_mean_is_the_contract() {
        for bright in [[0.9f32, 0.1, 0.5], [3.0, 2.4, 2.9]] {
            let mut albedo = vec![[0.2f32, 0.4, 0.05]; 64];
            albedo[0] = bright;
            normalise_albedo(&mut albedo);
            for ch in 0..3 {
                let mean: f32 = albedo.iter().map(|c| c[ch]).sum::<f32>() / albedo.len() as f32;
                assert!(
                    (mean - ALBEDO_MEAN).abs() < 1.0e-3,
                    "channel {ch} mean {mean} for {bright:?}"
                );
            }
            assert!(
                albedo.iter().all(|c| c.iter().all(|v| *v <= 1.0)),
                "soft clip must keep every channel in range"
            );
        }
    }

    /// Every layer must survive the full pipeline with a usable dynamic range
    /// — a flat height field would mean a flat normal map and no material at
    /// all, which is easy to introduce by mis-tuning one noise amplitude.
    #[test]
    fn every_layer_has_relief_and_range() {
        for layer in 0..TERRAIN_LAYERS {
            let mut f = synth_layer(layer);
            normalise_height(&mut f.h);
            let (normal, ao) = derive_normal_ao(&f);
            let tilted = normal
                .iter()
                .filter(|n| n[0].abs() > 0.05 || n[1].abs() > 0.05)
                .count();
            assert!(
                tilted > normal.len() / 20,
                "layer {layer}: normal map is nearly flat ({tilted} tilted texels)"
            );
            let occluded = ao.iter().filter(|v| **v < 0.9).count();
            assert!(occluded > 0, "layer {layer}: cavity AO is uniform");
        }
    }
}
