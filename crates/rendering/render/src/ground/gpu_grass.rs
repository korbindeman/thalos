//! GPU-generated grass field (vegetation cascade bands 0–1, slice 1).
//!
//! The Ghost-of-Tsushima model applied to Thalos: **no persistent blade
//! geometry**. One entity per active body carries a fixed *template* mesh whose
//! vertices encode only `(cell offset, band, blade, corner)`; the vertex shader
//! derives every blade per frame from deterministic cell hashes + a small
//! CPU-filled **window** of height/mask control data (the resident control
//! data — the GoT tile textures). Memory is O(1): the template (built once at
//! startup) + two window textures, regardless of reach, density, or movement —
//! this replaces the CPU megamesh blade rings whose per-tile meshes and
//! rebuild churn were the confirmed grass OOM (docs/world/vegetation.md §5.0/§13).
//!
//! Structure:
//! - **Bands** ([`GPU_GRASS_BANDS`]): concentric density/LOD rings baked into
//!   the template — cell size doubles outward, blades-per-clump falls, blade
//!   width grows (constant-coverage rule). Band fades are complementary
//!   cross-fades through the shared `thalos::grass_displace` scale-fade, so
//!   ring boundaries are invisible and the far edge melts into the band-2
//!   terrain-shading grass in `body_terrain.wgsl`.
//! - **Cells**: each band lays clumps on its own body-global cube-sphere cell
//!   lattice ([`TileLattice`] at `cell_m`), so a clump's existence, position,
//!   and look are a pure hash of its global `(face, x, y)` cell — stable while
//!   the anchor re-snaps, view-independent, nothing stored.
//! - **Window** ([`build_gpu_grass_window`]): a `WINDOW_SIZE_PX`² grid of
//!   terrain height (R32Float) + aux masks (Rgba8: grass weight, terrain
//!   normal xz, scatter treatment) sampled from the body's [`HeightSource`]
//!   around the anchor, rebuilt off-thread when the player moves. The shader
//!   gates and roots blades on it — the same data the CPU builder sampled per
//!   blade, amortized to per-move instead of per-tile-rebuild.
//!
//! The game-side driver (`thalos_runtime::rendering::gpu_grass`) owns anchoring
//! (runway-pattern f64 re-pose), window rebuild scheduling, and per-frame
//! shading params; this module is pure geometry/material/window math.

use std::sync::Arc;

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::math::{DVec3, UVec4, Vec4};
use bevy::mesh::{Indices, Mesh, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::ground::body_material::ShadowCascadeBlock;
use crate::ground::height_source::HeightSource;
use crate::ground::pipeline::material_masks_from_heights;
use crate::ground::scatter::{ScatterClass, ScatterRegion, classify_scatter};
use crate::ground::tile_lattice::{TileLattice, cube_face_uv, tiles_per_side};
use crate::ground::vegetation::GrassProfile;

// ---------------------------------------------------------------------------
// Bands
// ---------------------------------------------------------------------------

/// One template band: a concentric annulus of clump cells at one density/LOD.
///
/// **Mirrored in `gpu_grass.wgsl` (`BAND_*` consts)** — the template encodes a
/// vertex's band index, and the shader looks its parameters up from its own
/// copy of this table. Change them together.
#[derive(Debug, Clone, Copy)]
pub struct GpuGrassBand {
    /// Clump cell size (m) — one clump per cell, so density = 1/cell².
    pub cell_m: f64,
    /// Blades per clump (fixed in the template).
    pub blades: u32,
    /// Fade-in edge (m ground distance from the craft anchor); 0 = never.
    pub inner_m: f32,
    /// Fade-out edge (m).
    pub outer_m: f32,
    /// Cross-fade band half-width (m).
    pub fade_m: f32,
}

/// Number of cascade bands — mirrored in the WGSL band tables and sized into
/// [`GpuGrassParams`]' per-band arrays. Change everything together.
pub const GPU_GRASS_BAND_COUNT: usize = 5;

/// The band cascade. Constant-coverage: cells double outward while blades per
/// clump fall and the shader widens blades per band (see `GG_BAND_WIDTH_MUL`
/// in the WGSL). Reach ends at 340 m — the band-4 card-scale blades replace
/// the former CPU card ring, and the band-2 terrain-shading grass in
/// `body_terrain.wgsl` carries the field beyond.
pub const GPU_GRASS_BANDS: [GpuGrassBand; GPU_GRASS_BAND_COUNT] = [
    GpuGrassBand {
        cell_m: 0.30,
        blades: 26,
        inner_m: 0.0,
        outer_m: 10.0,
        fade_m: 3.0,
    },
    GpuGrassBand {
        cell_m: 0.60,
        blades: 18,
        inner_m: 10.0,
        outer_m: 30.0,
        fade_m: 4.0,
    },
    GpuGrassBand {
        cell_m: 1.30,
        blades: 10,
        inner_m: 30.0,
        outer_m: 80.0,
        fade_m: 8.0,
    },
    GpuGrassBand {
        cell_m: 2.60,
        blades: 6,
        inner_m: 80.0,
        outer_m: 170.0,
        fade_m: 14.0,
    },
    GpuGrassBand {
        cell_m: 5.20,
        blades: 4,
        inner_m: 170.0,
        outer_m: 340.0,
        fade_m: 28.0,
    },
];

/// Outermost blade reach (m) — the far fade edge of the last band.
pub const GPU_GRASS_REACH_M: f64 = 340.0;

/// How far the player's ground point may drift from the anchor before the
/// driver re-snaps it (m). The template's annuli carry this much slack beyond
/// each band edge so a between-snaps blade set still covers its annulus.
pub const GPU_GRASS_SNAP_SLACK_M: f64 = 4.0;

/// Window texture side (texels). The window spans `2 × GPU_GRASS_WINDOW_HALF_M`.
pub const GPU_GRASS_WINDOW_SIZE_PX: u32 = 768;
/// Window half-extent (m): blade reach (340) + fade (28) + snap slack (4) +
/// margin for the driver's ~25 m refill drift.
pub const GPU_GRASS_WINDOW_HALF_M: f64 = 420.0;

/// Height-sample LOD hint — matches the CPU blade builder so blades root on
/// the same surface the player stands on.
const GPU_GRASS_SAMPLE_LOD_M: f32 = 0.5;

// ---------------------------------------------------------------------------
// Template mesh
// ---------------------------------------------------------------------------

/// Blade strip: 7 vertices, 5 triangles (matches the CPU `Full` blade budget).
const VERTS_PER_BLADE: u32 = 7;
const BLADE_INDICES: [u32; 15] = [0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4, 4, 5, 6];

/// Build the one static template mesh the GPU grass field draws every frame.
///
/// `POSITION` carries no geometry — it packs `(cell dx, cell dy,
/// band·2048 + blade·8 + corner)`; the vertex shader synthesizes the blade.
/// Cells are pruned to each band's annulus (+ fade + snap slack), so template
/// size tracks the drawn field, not the bounding square.
pub fn build_gpu_grass_template() -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for (band_idx, band) in GPU_GRASS_BANDS.iter().enumerate() {
        let keep_hi =
            band.outer_m as f64 + band.fade_m as f64 + GPU_GRASS_SNAP_SLACK_M + band.cell_m;
        let keep_lo =
            (band.inner_m as f64 - band.fade_m as f64 - GPU_GRASS_SNAP_SLACK_M - band.cell_m)
                .max(0.0);
        let window = (keep_hi / band.cell_m).ceil() as i64;
        for dy in -window..=window {
            for dx in -window..=window {
                let cx = (dx as f64 + 0.5) * band.cell_m;
                let cy = (dy as f64 + 0.5) * band.cell_m;
                let r = (cx * cx + cy * cy).sqrt();
                if r < keep_lo || r > keep_hi {
                    continue;
                }
                for blade in 0..band.blades {
                    let base = positions.len() as u32;
                    let packed = (band_idx as u32) * 2048 + blade * 8;
                    for corner in 0..VERTS_PER_BLADE {
                        positions.push([dx as f32, dy as f32, (packed + corner) as f32]);
                    }
                    indices.extend(BLADE_INDICES.iter().map(|i| base + i));
                }
            }
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ---------------------------------------------------------------------------
// Anchor frame
// ---------------------------------------------------------------------------

/// The anchor's tangent frame + per-band lattice registration, computed once
/// per window rebuild and shared by the window fill (CPU) and the shader
/// (via [`GpuGrassParams`]) so both agree on the same axes.
#[derive(Debug, Clone, Copy)]
pub struct GpuGrassAnchor {
    /// Body-fixed unit direction of the anchor point.
    pub dir: DVec3,
    /// Body-fixed tangent axes (u along `east`, v along `north`) and radial up.
    pub east: DVec3,
    pub north: DVec3,
    /// Per band: the anchor's global cell `(x, y)` on that band's lattice and
    /// the anchor's fractional position within that cell (`[0, 1)` cells).
    pub band_cell: [(i64, i64); GPU_GRASS_BAND_COUNT],
    pub band_frac: [(f64, f64); GPU_GRASS_BAND_COUNT],
    /// Metric size of one cell along u/v at the anchor, per band. Uses the
    /// band's nominal `cell_m` scaled by the local cube distortion.
    pub band_cell_m: [(f64, f64); GPU_GRASS_BAND_COUNT],
    /// Cube face under the anchor (shared by every band's key).
    pub face: u8,
}

/// Deterministic ENU-ish tangent frame at a body-fixed direction. Both the
/// window fill and the shader params derive from this one function, so the
/// window's texel grid and the shader's cell math can never disagree.
pub fn gpu_grass_anchor(dir: DVec3, radius_m: f64) -> GpuGrassAnchor {
    let up = dir.normalize();
    let seed = if up.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
    let east = seed.cross(up).normalize();
    let north = up.cross(east);

    let (face, u, v) = cube_face_uv(up);
    let mut band_cell = [(0i64, 0i64); GPU_GRASS_BAND_COUNT];
    let mut band_frac = [(0.0f64, 0.0f64); GPU_GRASS_BAND_COUNT];
    let mut band_cell_m = [(0.0f64, 0.0f64); GPU_GRASS_BAND_COUNT];
    for (i, band) in GPU_GRASS_BANDS.iter().enumerate() {
        let cps = tiles_per_side(radius_m, band.cell_m);
        let n = cps as f64;
        // Same uv→index mapping as `TileLattice::key_of`, kept unclamped so the
        // fractional registration is exact.
        let xf = (u + 1.0) * 0.5 * n;
        let yf = (v + 1.0) * 0.5 * n;
        let cx = xf.floor();
        let cy = yf.floor();
        band_cell[i] = (cx as i64, cy as i64);
        band_frac[i] = (xf - cx, yf - cy);
        // Metric cell size at the anchor from the lattice's own extents.
        let lattice = TileLattice {
            tiles_per_side: cps,
        };
        let key = lattice.key_of(up);
        let (ext_u, ext_v) = lattice.tile_extents_m(key, radius_m);
        band_cell_m[i] = (ext_u, ext_v);
    }

    GpuGrassAnchor {
        dir: up,
        east,
        north,
        band_cell,
        band_frac,
        band_cell_m,
        face,
    }
}

// ---------------------------------------------------------------------------
// Window fill (async, CPU)
// ---------------------------------------------------------------------------

/// Everything the off-thread window fill needs, snapshotted by the driver.
pub struct GpuGrassWindowInput {
    pub height_source: Arc<dyn HeightSource>,
    pub radius_m: f64,
    pub anchor: GpuGrassAnchor,
    /// Building-terrain footprints (lawn/clear), as for the CPU blade builder.
    pub scatter_regions: Arc<Vec<ScatterRegion>>,
    /// Window texture side (texels) — [`GPU_GRASS_WINDOW_SIZE_PX`] in-game;
    /// the object preview uses a small confined window.
    pub size_px: u32,
    /// Window half-extent (m) — [`GPU_GRASS_WINDOW_HALF_M`] in-game.
    pub half_m: f64,
}

/// A finished control-data window: `heights` (row-major, `size²` f32 metres
/// above the reference radius) + `aux` (`size² × 4` bytes: grass weight,
/// terrain normal x/z in the anchor tangent frame biased to `[0,255]`,
/// scatter treatment 0 natural / 128 lawn / 255 clear).
pub struct GpuGrassWindow {
    pub heights: Vec<f32>,
    pub aux: Vec<u8>,
    /// Terrain height at the anchor (m above reference radius).
    pub anchor_height_m: f32,
    /// Macro landcover moisture at the anchor (`[-1, 1]`), sampled from the
    /// height source's wrapped surface. Per-window constant — the finest macro
    /// tier (~9 km) is far wider than the ±420 m window — carried to the
    /// shader in `GpuGrassParams.phase.w`, where the wrapped fine detail tier
    /// is added per blade (docs/world/terrain_macro.md).
    pub anchor_moisture: f32,
    /// `HeightSource::revision()` at fill time.
    pub built_revision: u64,
}

/// Fill one control window around the anchor. Pure; intended for
/// `AsyncComputeTaskPool`. Texels whose height probe misses (off the resident
/// atlas) get `grass weight = 0` (no blades) rather than failing the window.
pub fn build_gpu_grass_window(input: &GpuGrassWindowInput) -> GpuGrassWindow {
    let size = input.size_px as usize;
    let half_m = input.half_m;
    let texel_m = (2.0 * half_m) / size as f64;
    let source = input.height_source.as_ref();
    let built_revision = source.revision();
    let anchor_point = input.anchor.dir * input.radius_m;

    // Pass 1 — heights.
    let mut heights = vec![f32::MIN; size * size];
    for ty in 0..size {
        let dy = (ty as f64 + 0.5) * texel_m - half_m;
        for tx in 0..size {
            let dx = (tx as f64 + 0.5) * texel_m - half_m;
            let p = anchor_point + input.anchor.east * dx + input.anchor.north * dy;
            let dir = p.normalize();
            if let Some(h) = source.sample_height_m(dir.as_vec3(), GPU_GRASS_SAMPLE_LOD_M) {
                heights[ty * size + tx] = h;
            }
        }
    }

    // Pass 2 — masks + normal from the height grid (central differences), and
    // the scatter treatment from the structure footprints.
    let step = texel_m as f32;
    let sample = |x: isize, y: isize| -> f32 {
        let x = x.clamp(0, size as isize - 1) as usize;
        let y = y.clamp(0, size as isize - 1) as usize;
        heights[y * size + x]
    };
    let mut aux = vec![0u8; size * size * 4];
    for ty in 0..size {
        let dy = (ty as f64 + 0.5) * texel_m - half_m;
        for tx in 0..size {
            let h = heights[ty * size + tx];
            let o = (ty * size + tx) * 4;
            if h == f32::MIN {
                // Missing probe: bare (normal up, natural treatment).
                aux[o] = 0;
                aux[o + 1] = 127;
                aux[o + 2] = 127;
                aux[o + 3] = 0;
                continue;
            }
            let (xi, yi) = (tx as isize, ty as isize);
            let h_l = sample(xi - 1, yi);
            let h_r = sample(xi + 1, yi);
            let h_d = sample(xi, yi - 1);
            let h_u = sample(xi, yi + 1);
            // Missing neighbours fall back to the centre so gradients stay finite.
            let fix = |v: f32| if v == f32::MIN { h } else { v };
            let (h_l, h_r, h_d, h_u) = (fix(h_l), fix(h_r), fix(h_d), fix(h_u));

            let masks = material_masks_from_heights(h, h_l, h_r, h_d, h_u, step);
            let grad_x = (h_r - h_l) / (2.0 * step);
            let grad_y = (h_u - h_d) / (2.0 * step);
            // Normal in the anchor tangent frame (x = east, z = north), unit,
            // xz components biased into bytes; the shader reconstructs y.
            let inv_len = 1.0 / (1.0 + grad_x * grad_x + grad_y * grad_y).sqrt();
            let nx = -grad_x * inv_len;
            let nz = -grad_y * inv_len;

            let dx = (tx as f64 + 0.5) * texel_m - half_m;
            let p = anchor_point + input.anchor.east * dx + input.anchor.north * dy;
            let treatment = match classify_scatter(&input.scatter_regions, p.normalize()) {
                ScatterClass::Natural => 0u8,
                ScatterClass::Lawn => 128u8,
                ScatterClass::Clear => 255u8,
            };

            aux[o] = masks[0];
            aux[o + 1] = ((nx * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            aux[o + 2] = ((nz * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
            aux[o + 3] = treatment;
        }
    }

    let center = size / 2;
    let anchor_height_m = {
        let h = heights[center * size + center];
        if h == f32::MIN { 0.0 } else { h }
    };

    GpuGrassWindow {
        heights,
        aux,
        anchor_height_m,
        anchor_moisture: source.landcover_moisture(input.anchor.dir),
        built_revision,
    }
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// Per-frame + per-window shading/placement parameters.
///
/// Field order is load-bearing — the WGSL `GpuGrassParams` mirror in
/// `gpu_grass.wgsl` must match. The first six fields carry the exact
/// [`GrassParams`](super::vegetation::GrassParams) semantics (sun / wind /
/// time / sky / anchor) so blades light identically to the CPU rings and the
/// ground; the rest register the window + cell lattices.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct GpuGrassParams {
    /// xyz = unit direction toward the star (world render space), w = sun flux.
    pub sun_dir: Vec4,
    /// xyz = wind direction (world render space), w = tip sway amplitude (m).
    pub wind: Vec4,
    /// x = time (s), y = altitude collapse 0→1, z = sea level (m), w = enable
    /// (0 kills every blade — the driver's park switch).
    pub time_fade: Vec4,
    /// xyz = local radial up (world render space), w unused.
    pub sky_up: Vec4,
    /// xyz = Rayleigh τ_v, w = atmosphere strength.
    pub sky_tau: Vec4,
    /// xyz = craft − camera offset (render space), w = 1 valid (see
    /// `GrassParams::anchor`).
    pub anchor: Vec4,
    /// Body-fixed (= entity-local) tangent u axis; w = anchor height (m above
    /// the reference radius — window heights are made anchor-relative with it).
    pub frame_east: Vec4,
    /// Body-fixed tangent v axis; w = anchor offset from the window centre
    /// along `east` (the anchor re-registers every few metres, the window
    /// every ~25 m — window lookups add this offset).
    pub frame_north: Vec4,
    /// Body-fixed radial up at the anchor; w = anchor offset along `north`.
    pub frame_up: Vec4,
    /// x = window texel size (m), y = window size (px), z = window half (m),
    /// w = climate cold lift at the anchor (m — see
    /// `thalos::landcover::climate_cold_lift`; shifts the blade treeline fade
    /// and veg palette with latitude).
    pub window_meta: Vec4,
    /// xyz = anchor surface point, body-fixed metres, folded mod the landcover
    /// coordinate period (4 km) — the `thalos::landcover` sampling phase.
    /// w = macro landcover moisture at the anchor (`[-1, 1]`; the shader adds
    /// the wrapped fine detail tier per blade).
    pub phase: Vec4,
    /// Per band: anchor global cell x, y, cube face, unused.
    pub band_cell: [UVec4; GPU_GRASS_BAND_COUNT],
    /// Per band: metric cell size along u/v (m), anchor frac x/y (cells).
    pub band_geom: [Vec4; GPU_GRASS_BAND_COUNT],
    /// Grass style table — [`gpu_grass_style_table`]. Two vec4 rows per style
    /// (dry / lush / lawn): `(height_m, width_m, radius_m, droop)` then
    /// `(dome, dry_mix, sheen, stiffness)`. Authored on the Rust side so grass
    /// types are configured without shader edits.
    pub style: [Vec4; 6],
}

/// One authorable grass type: the geometric [`GrassProfile`] plus the
/// GPU-field shading/behaviour parameters layered on top.
#[derive(Debug, Clone, Copy)]
pub struct GrassStyle {
    pub profile: GrassProfile,
    /// Fraction of blades that read as dry straw when the landcover is at its
    /// driest (`[0, 1]`; the shader scales it down as moisture rises).
    pub dry_mix: f32,
    /// Specular sheen strength (`[0, 1]`) — the glossy highlight that rolls
    /// across a field as the view or wind moves.
    pub sheen: f32,
    /// Wind stiffness (`[0, 1]`): 1 = rigid (lawns), 0 = supple (tall prairie
    /// bends deeply with each gust).
    pub stiffness: f32,
}

impl GrassStyle {
    pub const fn dry() -> Self {
        Self {
            profile: GrassProfile::fluffy_short(),
            dry_mix: 0.32,
            sheen: 0.30,
            stiffness: 0.65,
        }
    }

    pub const fn lush() -> Self {
        Self {
            profile: GrassProfile::wispy_tall(),
            dry_mix: 0.08,
            sheen: 0.60,
            stiffness: 0.30,
        }
    }

    pub const fn lawn() -> Self {
        Self {
            profile: GrassProfile::lawn(),
            dry_mix: 0.04,
            sheen: 0.45,
            stiffness: 0.85,
        }
    }

    fn rows(&self) -> [Vec4; 2] {
        [
            Vec4::new(
                self.profile.height_m,
                self.profile.width_m,
                self.profile.radius_m,
                self.profile.droop,
            ),
            Vec4::new(self.profile.dome, self.dry_mix, self.sheen, self.stiffness),
        ]
    }
}

/// The default style table for [`GpuGrassParams::style`]: dry, lush, lawn.
/// Every param-fill site (game driver, object preview) must install a style
/// table — an all-zero table renders zero-size blades.
pub fn gpu_grass_style_table() -> [Vec4; 6] {
    let styles = [GrassStyle::dry(), GrassStyle::lush(), GrassStyle::lawn()];
    let mut rows = [Vec4::ZERO; 6];
    for (i, s) in styles.iter().enumerate() {
        let r = s.rows();
        rows[i * 2] = r[0];
        rows[i * 2 + 1] = r[1];
    }
    rows
}

/// GPU-generated grass field material: the template mesh in, blades out.
/// Shading matches [`GrassMaterial`](super::vegetation::GrassMaterial) — the
/// same `thalos::lighting` foliage model, per-vertex sun-shadow receive, and
/// `thalos::grass_displace` fade/wind — so the GPU field and the remaining CPU
/// card ring read as one meadow.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct GpuGrassMaterial {
    #[uniform(0)]
    pub params: GpuGrassParams,
    /// Cascaded sun-shadow transforms + strength (see [`ShadowCascadeBlock`]).
    #[uniform(1)]
    pub shadow: ShadowCascadeBlock,
    /// Per-cascade sun-shadow depth maps — the same handles the ground + tree
    /// materials bind; `vertex` visibility for the per-vertex blade sample.
    /// Cascade 0 — the ±64 m near box added 2026-07-31. Bound at the END of
    /// this material's range rather than renumbered into the near→far slot:
    /// only the ARGUMENT order at the `thalos::shadow` call site is
    /// ordering-significant, so shifting live binding indices would be risk
    /// with no payoff. Field name still equals the cascade index.
    #[texture(8, sample_type = "depth", visibility(vertex, fragment))]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(2, sample_type = "depth", visibility(vertex, fragment))]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(3, sample_type = "depth", visibility(vertex, fragment))]
    pub sun_shadow_map_2: Handle<Image>,
    #[texture(4, sample_type = "depth", visibility(vertex, fragment))]
    pub sun_shadow_map_3: Handle<Image>,
    /// Control window: terrain height, R32Float (non-filterable —
    /// `textureLoad` + manual bilinear in the shader).
    #[texture(5, sample_type = "float", filterable = false, visibility(vertex))]
    pub height_window: Handle<Image>,
    /// Control window: aux masks (grass weight, normal xz, treatment), Rgba8.
    #[texture(6, visibility(vertex))]
    #[sampler(7, visibility(vertex))]
    pub aux_window: Handle<Image>,
}

impl Material for GpuGrassMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/gpu_grass.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/gpu_grass.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Blades are single strips seen from both sides.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

pub(crate) fn embed_gpu_grass_shader(app: &mut App) {
    embedded_asset!(app, "gpu_grass.wgsl");
}

/// Standalone plugin that registers [`GpuGrassMaterial`] and embeds its
/// shader, for consumers that render the field without the full
/// [`ThalosTerrainPlugin`](crate::ground::ThalosTerrainPlugin) (the headless
/// object preview). The terrain plugin registers the same material directly —
/// don't add both.
pub struct GpuGrassMaterialPlugin;

impl Plugin for GpuGrassMaterialPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(MaterialPlugin::<GpuGrassMaterial>::default());
        embed_gpu_grass_shader(app);
    }
}
