//! Tree / shrub scatter driver.
//!
//! Maintains a camera-local set of body-fixed **scatter tiles** around the
//! player on the nearest vegetated body, using the placement + scatter system in
//! `thalos_body_render::ground::scatter`. Each tile's trees/shrubs are baked into
//! **one batched mesh per tile** (the same one-mesh-per-tile batching the grass
//! uses) — so there is *no per-tree ECS entity*, and forests scale to dense/far.
//! Per-tree variation (scale, rotation, tint, wind phase) and the seamless
//! per-tree scale-fade survive because each tree's base is baked into the mesh
//! UVs and read by `TreeMaterial`.
//!
//! Anchoring is the grass / runway pattern: each tile is a **root-grid big_space
//! child** re-posed in f64 every frame, so the f32 transform only acts on the
//! tile's small vertex offsets and trees stay rock-steady under high warp.
//!
//! Builds run on `AsyncComputeTaskPool` (scatter placement + mesh combine),
//! gated on terrain residency so plants seat on the streamed mesh. Tiles are
//! built a tile beyond the fade edge (invisible build → no pop-in), re-LOD'd by
//! rebuilding the tile mesh at the new LOD (old kept until the new is ready → no
//! vanish), and rebuilt when the height source revision advances.
//!
//! Dispatch policy: **coverage outranks refinement** (missing tiles beat re-LOD
//! upgrades, `TREE_UPGRADE_PENALTY`), a cold fill builds cheap impostors first
//! and upgrades nearest-first afterwards (`TREE_COLD_FILL_MISSING`), and a
//! fast-moving view keys its LOD off where it is about to be
//! (`TREE_MOTION_LEAD_S`, fed by the smoothed `ViewAnchor` speed) — so the
//! forest *exists* around the view quickly at speed and settles to full mesh
//! fidelity where the camera lingers.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::camera::Hdr;
use bevy::camera::primitives::MeshAabb;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ClearColorConfig, ImageRenderTarget, RenderTarget, ScalingMode};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::NotShadowCaster;
use bevy::math::{DVec3, Vec2, Vec3A};
use bevy::prelude::*;
use bevy::tasks::{Task, block_on, poll_once};
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::{
    AU_M, BakeParams, CanopyStyle, GrassParams, IMPOSTOR_MAX_SPECIES, ImpostorAtlasLayout,
    ImpostorParams, LIGHT_AT_1AU, TerrainShadingStyle, TileKey, TileLattice, TreeBakeMaterial,
    TreeImpostorExtension, TreeImpostorMaterial, TreeMaterial, TreeMeshData, TreeMeshParams,
    TreeShadingExtension, VegLayer, VegScatterInput, VegScatterTile, VegSpeciesPlacement,
    build_foliage_atlas, build_foliage_material_atlas, build_scatter_tile, build_tree_mesh_data,
    combine_impostor_tile_mesh, combine_tree_tile_mesh, fallback_shadow_map, hemioct_decode,
    impostor_bake_rotation, make_impostor_atlas, recenter_tree_mesh, tree_bounding_sphere,
    tree_impostor_material, tree_material, veg_scatter_pool,
};
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::SimStage;
use crate::coords::SHIP_LAYER;
use crate::rendering::ground_terrain::{TerrainFlattenRegistry, terrain_shading_style_for};
use crate::rendering::real_space::{RealSpaceRoot, real_space_grid};
use crate::rendering::types::CameraExposure;
use crate::rendering::view_anchor::ViewAnchor;
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};

// ── Clipmap rings ──────────────────────────────────────────────────────────────
/// One clipmap ring of the tree scatter: a cube-sphere lattice at `tile_size_m`,
/// covering ground distances `[inner_m, outer_m]` from the player.
///
/// **Ring 0** is the fine near/mid band — full mesh-LOD trees (`lod_for_dist`)
/// plus the natural-size octahedral impostor far band. **Rings ≥ 1** are
/// **impostor-only** rings carrying the forest out to ~22 km, handing off
/// (eventually) to the terrain albedo. Every ring draws trees at *natural size*
/// (`grove_scale = 1`; see the size-consistency note below).
///
/// **Placement: shared grid near, coarse grid far.** A ring covers `[inner_m,
/// outer_m]` and controls its tree set two ways:
/// - `spacing_scale` — the Poisson-grid coarsening (`1` = the authored
///   `TREE_SPACING_M` grid). Two rings with the same `spacing_scale` share the
///   *same* grid → the same tree sits at the *same* world position in both.
/// - `keep_fraction` — a nested-subset thinning (`1` = keep all) applied after
///   elimination. A ring with a smaller fraction renders a strict subset of the
///   trees a finer, same-grid ring keeps.
///
/// **Why:** a discrete, *individually resolvable* tree can't cross-fade between
/// two *independent* grids without the whole forest visibly dissolving (one grid
/// of trees shrinking away while a different grid grows in — the "trees appear
/// from nothing in between" report). So **ring 1 shares ring 0's grid**
/// (`spacing_scale = 1`) and only *thins* it (`keep_fraction < 1`): the trees it
/// draws are exactly ring 0's, at the same spots, so the 2.4 km handoff keeps
/// each shared tree in place and only the density-delta *infill* fades in on
/// approach. Rings 2–3 (≥ 6 km, where an 8 m tree is ~1 px and below the eye-line
/// horizon) keep cheap **coarse independent** grids — a dissolve out there is
/// invisible, and a full-density grid on a 2 km tile is ~16× the placement cost.
///
/// **Size consistency (`grove_scale = 1` everywhere).** The constant-coverage
/// rule (grow a far element to stand in for the clump it replaces — see
/// `docs/world/vegetation.md` §5.1) is right for *grass* (a blade is never resolvable)
/// but wrong for trees (enlarging a resolvable tree just makes a giant tree and
/// snaps its size at each ring boundary). `grove_scale` is kept as a knob but
/// stays 1.
struct TreeRing {
    tile_size_m: f64,
    inner_m: f64,
    outer_m: f64,
    /// Poisson-grid coarsening (`1` = shared with the authored fine grid).
    spacing_scale: f32,
    /// Nested-subset thinning (`1` = keep every survivor). Only meaningful
    /// against a same-`spacing_scale` ring, whose trees it strictly subsets.
    keep_fraction: f32,
    grove_scale: f32,
}

/// The tree clipmap. Ring 0 reproduces the near/mid mesh cascade + the near
/// natural-size impostor band; ring 1 shares ring 0's grid (thinned) so the
/// 2.4 km handoff is positionally stable; rings 2–3 carry cheap coarse impostors
/// out to ~22 km. Tile sizes grow ~2× per ring so each is a thin annulus.
///
/// **Tuning knobs:** `keep_fraction` on ring 1 is the near-far handoff dial —
/// toward `1.0` fewer trees "appear from nothing" on approach (all shared with
/// ring 0) but the mid-field is denser (more impostor quads); lower thins it.
/// `spacing_scale` on rings 2–3 is the far-density/perf knob. Tune from
/// screenshots + frame timings.
const TREE_RINGS: [TreeRing; 4] = [
    TreeRing {
        tile_size_m: 200.0,
        inner_m: 0.0,
        outer_m: 2400.0,
        spacing_scale: 1.0,
        keep_fraction: 1.0,
        grove_scale: 1.0,
    },
    TreeRing {
        // Shares ring 0's grid (spacing 1) and thins to a nested subset, so every
        // tree it draws is one of ring 0's at the identical spot — the 2.4 km
        // handoff keeps them in place instead of dissolving grids.
        tile_size_m: 500.0,
        inner_m: 2400.0,
        outer_m: 6000.0,
        spacing_scale: 1.0,
        keep_fraction: 0.5,
        grove_scale: 1.0,
    },
    TreeRing {
        // ≥ 6 km: coarse independent grid is fine (trees ~sub-pixel / below the
        // eye-line horizon, so a grid dissolve here is invisible and cheap).
        tile_size_m: 1000.0,
        inner_m: 6000.0,
        outer_m: 12000.0,
        spacing_scale: 2.5,
        keep_fraction: 1.0,
        grove_scale: 1.0,
    },
    TreeRing {
        tile_size_m: 2000.0,
        inner_m: 12000.0,
        outer_m: 22000.0,
        spacing_scale: 4.0,
        keep_fraction: 1.0,
        grove_scale: 1.0,
    },
];

/// Reach of ring 0 before the impostor atlas is baked: the far band falls back to
/// the minimal LOD3 mesh, so we keep it short — and the coarse impostor rings are
/// skipped entirely until the atlas is ready (a second or two into the session).
const TREE_MESH_ONLY_REACH_M: f64 = 2200.0;

/// Outer reach (m) of the last ring whose tiles still CAST into the sun-shadow
/// cascades. Rings 0–1 (mesh + natural-size near impostors, to 6 km) cast; the
/// coarse far rings (6–22 km, `spacing_scale` 2.5–4×) do not — their trees are
/// sub-pixel from any near-surface vantage, and covering them forced the far
/// shadow cascade out to a 23.5 km half-extent (~11.5 m/texel: every shadow
/// beyond the near cascade was a coarse blob). Trimming the caster band lets
/// `sun_shadow::CASCADE_MIN_HALF_M` shrink ~3× (the MSFS-style split: shadow
/// maps carry the near field only; far-terrain shading belongs to the
/// heightfield horizon term, W12). Keep in lockstep with those minimums.
const TREE_SHADOW_CASTER_MAX_M: f64 = 6_000.0;

/// Half-width of the ring cross-fade (m), fixed so it is identical on both sides
/// of every shared boundary (that is what makes adjacent rings *complementary* —
/// see [`tree_ring_fade`]). Also used as the build look-ahead margin.
const TREE_FADE_BAND_M: f32 = 300.0;

/// Fade band for a ring — a fixed handoff width. (Kept as a function of the ring
/// for the build-margin call sites; the value no longer depends on span.)
fn tree_ring_band_m(_r: &TreeRing) -> f32 {
    TREE_FADE_BAND_M
}

/// Near/far/band fade edges for a ring (ground distance, m), packed into
/// `GrassParams.time_fade`. The shader scale-fades each instance across
/// `[near, far]`: `fade_in = smoothstep(near-band, near+band, d)`,
/// `fade_out = 1 - smoothstep(far-band, far+band, d)`, `scale ∝ fade_in·fade_out`.
///
/// **Complementary cross-fade.** `near = inner_m`, `far = outer_m`, with the
/// *same* band on both sides of a shared boundary `B = outer_N = inner_{N+1}`.
/// At `B`, ring `N`'s `fade_out` and ring `N+1`'s `fade_in` are the same
/// smoothstep mirrored, so their scales sum to ~1: a tree shared across the
/// boundary (same grid — see [`TreeRing`]) hands off from one ring's card to the
/// next without doubling (the old *overlap-full* handoff would draw a shared tree
/// at full scale in *both* rings) and without a coverage dip (the shared tree is
/// ~full the whole way; only the density-delta infill grows in). Ring 0 is full
/// to the camera (`near = -∞`); the outermost ring's `far` fades the cascade to
/// nothing at its edge.
fn tree_ring_fade(idx: usize) -> (f32, f32, f32) {
    let r = &TREE_RINGS[idx];
    let near = if idx == 0 { -1.0e9 } else { r.inner_m as f32 };
    (near, r.outer_m as f32, TREE_FADE_BAND_M)
}

/// Pre-bake fade edges for the mesh material (the LOD3 far band fades out near
/// the mesh-only reach so its edge isn't a hard ring).
const TREE_MESH_ONLY_FADE: (f32, f32, f32) = (-1.0e9, 1975.0, 150.0);

// ── Tuning ───────────────────────────────────────────────────────────────────
/// Octahedral atlas: captured views per axis (`N`) and pixels per cell.
const IMPOSTOR_CELLS: u32 = 8;
const IMPOSTOR_CELL_PX: u32 = 128;
/// Coverage below which an impostor pixel is discarded.
const IMPOSTOR_ALPHA_CUTOFF: f32 = 0.35;
/// Fraction of each atlas cell the captured tree fills (the rest is an
/// anti-bleed gutter between cells).
const IMPOSTOR_CELL_FILL: f32 = 0.84;
/// Frames the off-screen bake rig renders before teardown — long enough to cover
/// async render-pipeline compilation. Until ready the far band uses LOD3 mesh.
const IMPOSTOR_BAKE_FRAMES: u32 = 90;
/// Dedicated render layers for the two off-screen bake passes (albedo, normal).
const IMPOSTOR_BAKE_ALBEDO_LAYER: usize = 6;
const IMPOSTOR_BAKE_NORMAL_LAYER: usize = 7;
/// Minimum trunk spacing (m) for the broadleaf — the widest canopy, so it sets
/// the shared tree grid's spacing. Below a canopy diameter so crowns touch into
/// connected groves, above a trunk width so trunks never interpenetrate.
const TREE_SPACING_M: f32 = 5.5;
/// Minimum spacing (m) for shrubs — their own grid, so they may sit under trees.
const SHRUB_SPACING_M: f32 = 2.3;
/// Above this altitude over the local terrain no new tiles are built (existing
/// ones persist). Generous so climbing aircraft keep their forest — the coarse
/// impostor rings are cheap, and a forested surface should read from altitude
/// (the far rings fold into the terrain albedo above this).
const TREE_MAX_AGL_M: f64 = 6000.0;
/// Above this altitude all tiles are despawned (e.g. after climb-out).
const TREE_DESPAWN_AGL_M: f64 = 16000.0;
/// Maximum concurrent tile builds. Higher than the original near-only driver
/// because the clipmap fills many more (coarse) tiles on a cold view — slow fill
/// is what shows the transient ring gaps. Sized as queue depth for the
/// `AsyncComputeTaskPool`, whose thread count is the real parallelism cap: a
/// deep queue keeps the pool fed when individual builds are short (impostor
/// combines), which is most of a cold fill under the coarse-first path below.
const TREE_MAX_IN_FLIGHT: usize = 24;
/// Motion look-ahead (s) folded into the slant distance the mesh-LOD pick is
/// keyed on: at eye speed `v` a tile is treated as `v ×` this farther away, so
/// a fast view builds cheap impostor tiles it will actually pass instead of
/// mesh tiles it out-runs before they land — and a settling view (speed → 0)
/// re-LODs back to full mesh fidelity in place, nearest first. Uses the same
/// smoothed `ViewAnchor` speed as the terrain motion brake.
const TREE_MOTION_LEAD_S: f64 = 3.0;
/// Missing-tile count above which the fill is "cold" (fresh spawn, teleport,
/// body handoff): missing ring-0 tiles outside the innermost mesh band build
/// impostor-FIRST — the combine is a few quads per tree instead of a batched
/// LOD mesh, so coverage lands in a fraction of the time — and the normal
/// re-LOD pass upgrades them to meshes nearest-first afterwards. The innermost
/// band (LOD0, < `TREE_LOD_BANDS_M[0]`) is exempt: a billboard 30 m from the
/// eye reads worse than a short wait, and that band is only a handful of tiles.
const TREE_COLD_FILL_MISSING: usize = 24;
/// Distance multiplier penalizing re-LOD upgrades against missing tiles in the
/// dispatch order: coverage outranks refinement (trees *existing* around the
/// view beats sharpening ones that already exist), except very near the eye,
/// where an upgrade at `d` still beats a missing tile beyond `penalty × d`.
const TREE_UPGRADE_PENALTY: f64 = 3.0;
/// Don't build until the terrain under a tile is resident at this texel size or
/// finer (mirrors the grass residency gate); scaled up for far tiles below.
const TREE_MAX_TERRAIN_TEXEL_M: f32 = 16.0;
/// World seed for placement hashes (distinct from grass).
const TREE_SEED: u64 = 0x7472_6565_7331;
/// Rebuild-staleness scan interval, seconds.
const TREE_REBUILD_CHECK_S: f32 = 0.75;
/// Rebuild a stale tile only when its centre height moved more than this.
const TREE_REBUILD_DELTA_M: f32 = 0.10;
/// Stale-tile rebuilds forgotten per scan tick (drive re-dispatches them).
const TREE_MAX_REBUILDS_PER_TICK: usize = 2;
/// LOD sample hint for the AGL ground probe.
const TREE_GROUND_LOD_M: f32 = 2.0;
/// Mesh-LOD band edges (ground distance, m): LOD0 < [0], LOD1 < [1], LOD2 < [2],
/// else the minimal far LOD3.
const TREE_LOD_BANDS_M: [f64; 3] = [260.0, 620.0, 1200.0];
/// Canopy wind sway amplitude at full weight, metres.
const TREE_WIND_SWAY_M: f32 = 0.35;
/// Number of mesh LODs per species.
const TREE_LOD_COUNT: usize = 4;
/// Largest per-instance scale any tree is placed at (the `scale_range` upper
/// bound). The impostor frustum-cull AABB pad uses it to bound the biggest card.
const TREE_SCALE_MAX: f32 = 1.6;

/// Mesh LOD index for a tile at ground distance `d`.
fn lod_for_dist(d: f64) -> usize {
    if d < TREE_LOD_BANDS_M[0] {
        0
    } else if d < TREE_LOD_BANDS_M[1] {
        1
    } else if d < TREE_LOD_BANDS_M[2] {
        2
    } else {
        3
    }
}

/// The procedural species library, built once at startup. `placement` is also
/// held as an `Arc<[…]>` for the async build; `lod_data[species][lod]` is the raw
/// CPU mesh combined per tile. All mesh-LOD species share one [`TreeMaterial`];
/// the far band shares one [`TreeImpostorMaterial`] (the octahedral atlas).
#[derive(Resource)]
struct SpeciesLibrary {
    placement: Arc<[VegSpeciesPlacement]>,
    lod_data: Vec<Vec<Arc<TreeMeshData>>>,
    material: Handle<TreeMaterial>,
    /// One impostor material per clipmap ring (same atlases; each carries its
    /// ring's cross-fade band in `params.time_fade`, written per frame).
    impostor_materials: Vec<Handle<TreeImpostorMaterial>>,
    /// Per placement-species index → octahedral atlas layer, or `None` for
    /// species with no impostor (shrubs). Snapshotted into the impostor build.
    atlas_species: Vec<Option<u32>>,
    /// Maximum tree canopy-top extent above the trunk base (authored units,
    /// `center.y + radius` of the LOD0 bounding sphere) over all tree species — the
    /// half-extent an impostor billboard can reach from a tile's tree bases. Used
    /// to pad an impostor tile's (degenerate-mesh) AABB so frustum culling keeps
    /// tall cards near the frustum edge instead of clipping them.
    max_tree_extent_m: f32,
}

/// State of the one-shot startup impostor-atlas bake. Until `ready`, the far
/// band falls back to the LOD3 mesh at the mesh-only reach.
#[derive(Resource, Default)]
struct ImpostorBake {
    ready: bool,
    frames: u32,
}

/// Marker on every off-screen bake-rig entity (the two cameras + the per-cell
/// instances), so the whole rig can be torn down once the atlas is captured.
#[derive(Component)]
struct ImpostorBakeRig;

/// What one async tile build produces: the combined mesh + anchor/staleness meta.
struct VegTileBuild {
    mesh: Mesh,
    /// True if this is a far-band octahedral-impostor mesh (billboard quads),
    /// false for a mesh-LOD batch.
    impostor: bool,
    center_surface_body_m: DVec3,
    built_revision: u64,
    center_height_m: f32,
    /// The placement this mesh was combined from, kept for re-LOD reuse (ring 0
    /// only — the coarse rings never change LOD), tagged with the
    /// [`VegLayer::mask`] bits that were actually placed.
    scatter: Option<(u32, Arc<VegScatterTile>)>,
    /// Wall time of the whole build (placement + combine), for the batched
    /// `build_batch` telemetry.
    build_micros: u64,
    instance_count: u32,
    /// True when placement came from the cache and only the combine ran.
    reused_placement: bool,
}

/// Rolling per-build telemetry, batched like the tile renderer's `GenStats`:
/// every 100 landings, one `build_batch` event with mean/p95 build wall time,
/// mean instance count, and the share of builds that reused a cached placement
/// — the numbers that say whether placement or combine is the current
/// bottleneck, and whether the re-LOD cache is actually being hit.
#[derive(Default)]
struct VegBuildStats {
    micros: Vec<u64>,
    instances_sum: u64,
    reused: u32,
    total_landed: u64,
}

impl VegBuildStats {
    fn record(&mut self, micros: u64, instances: u32, reused: bool) {
        self.micros.push(micros);
        self.instances_sum += instances as u64;
        self.reused += reused as u32;
        self.total_landed += 1;
        if self.micros.len() >= 100 {
            self.micros.sort_unstable();
            let mean = self.micros.iter().sum::<u64>() / self.micros.len() as u64;
            let p95 = self.micros[self.micros.len() * 95 / 100];
            info!(
                target: "thalos::diagnostic::vegetation",
                event = "build_batch",
                total_landed = self.total_landed,
                sample_count = self.micros.len(),
                mean_ms = mean as f64 / 1000.0,
                p95_ms = p95 as f64 / 1000.0,
                mean_instances = self.instances_sum as f64 / self.micros.len() as f64,
                reused_placement_frac = self.reused as f64 / self.micros.len() as f64,
                "vegetation build batch"
            );
            self.micros.clear();
            self.instances_sum = 0;
            self.reused = 0;
        }
    }
}

/// A tile key tagged with its clipmap ring (the same `(face,x,y)` indexes
/// different physical tiles at different ring lattices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RingTileKey {
    ring: u8,
    key: TileKey,
}

/// One finished tile. `entity: None` means the tile built empty (clearing,
/// water, rock, alpine) — recorded so it isn't rebuilt every frame.
struct BuiltTile {
    entity: Option<Entity>,
    built_revision: u64,
    center_height_m: f32,
    /// Mesh LOD this tile was baked at, so the driver can re-LOD on approach.
    lod: usize,
    /// Whether the tile is realized as impostors (vs mesh) — drives re-LOD when
    /// the impostor band flips on/off (atlas readiness).
    impostor: bool,
    /// Cached placement (ring 0 only) + the layer-mask it covers, so a re-LOD
    /// at the same height revision skips placement and only re-combines.
    /// Placement is the expensive half of a build and fully deterministic, so
    /// the cold-fill impostor-first pass and the settle-refinement burst both
    /// pay it once per tile, not once per LOD step.
    scatter: Option<(u32, Arc<VegScatterTile>)>,
}

/// Driver state. **Sole writer:** the systems in this module (run sequentially
/// via their `ResMut` access).
#[derive(Resource, Default)]
struct VegTiles {
    body: Option<BodyId>,
    /// One lattice per clipmap ring, aligned to [`TREE_RINGS`].
    lattices: Vec<TileLattice>,
    tiles: HashMap<RingTileKey, BuiltTile>,
    /// In-flight builds: (task, source revision, target LOD, want-impostor).
    in_flight: HashMap<RingTileKey, (Task<Option<VegTileBuild>>, u64, usize, bool)>,
    rebuild_timer: f32,
}

/// Marker on a spawned scatter-tile entity (one batched mesh).
#[derive(Component)]
struct VegTileVisual {
    body_id: BodyId,
    /// Body-fixed position of the tile centre on the surface.
    center_surface_body: DVec3,
}

pub struct VegetationRenderPlugin;

impl Plugin for VegetationRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VegTiles>()
            .init_resource::<ImpostorBake>()
            .add_systems(Startup, setup_species_library)
            .add_systems(
                Update,
                (
                    tick_impostor_bake,
                    check_veg_rebuilds,
                    drive_veg_tiles.after(check_veg_rebuilds),
                    finalize_veg_tiles.after(drive_veg_tiles),
                    update_veg_transforms.after(finalize_veg_tiles),
                    update_tree_material,
                )
                    .in_set(SimStage::Sync)
                    .after(sync_solar_system_state),
            )
            // `Last`, not `Update`/`PostUpdate`: same-frame with the cloud
            // driver's re-march — see `apply_tree_cloud_shadow`.
            .add_systems(bevy::app::Last, apply_tree_cloud_shadow);
    }
}

/// Build the procedural species library once at startup: a broadleaf tree and a
/// low shrub, each with a mesh-LOD chain of raw `TreeMeshData`, plus one shared
/// `TreeMaterial`; then bake the hemisphere octahedral impostor atlas for the
/// tree species and spawn the one-shot off-screen bake rig.
#[allow(clippy::too_many_arguments)]
fn setup_species_library(
    mut commands: Commands,
    mut materials: ResMut<Assets<TreeMaterial>>,
    mut impostor_materials: ResMut<Assets<TreeImpostorMaterial>>,
    mut bake_materials: ResMut<Assets<TreeBakeMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut lod_data: Vec<Vec<Arc<TreeMeshData>>> = Vec::new();
    let mut placement: Vec<VegSpeciesPlacement> = Vec::new();

    // --- Tree (broadleaf) ---
    let tree = TreeMeshParams {
        trunk_height_m: 5.2,
        trunk_radius_m: 0.32,
        canopy_radius_m: 3.0,
        canopy_height_m: 2.8,
        trunk_color: Vec3::new(0.16, 0.090, 0.045),
        // Light tint: the foliage atlas now carries the real leaf colour, so this
        // only nudges hue (warm) and AO modulates brightness.
        canopy_color: Vec3::new(0.92, 1.0, 0.82),
        style: CanopyStyle::Broadleaf,
        seed: 0xB1_05_50,
        lod: 0,
    };
    lod_data.push(build_lod_chain(&tree));
    placement.push(VegSpeciesPlacement {
        layer: VegLayer::Tree,
        min_spacing_m: TREE_SPACING_M,
        mix_weight: 1.0,
        scale_range: (0.8, 1.6),
        slope_limit: 0.40,
        altitude_band: (1800.0, 2900.0, 2400.0, 3100.0),
        // Full patch clumping → genuine treeless plains between distinct forest
        // patches (with a light-woodland apron ahead of each grove). See
        // scatter::clump_field / forest_coverage.
        clump_affinity: 1.0,
        min_grass_w: 0.22,
    });

    // (Pine/conifer species removed for now — to be rebuilt from scratch once the
    // broadleaf is dialled in. The `CanopyStyle::Conifer` mesh path still exists
    // for when it returns.)

    // --- Shrub (low bush) ---
    let shrub = TreeMeshParams {
        trunk_height_m: 0.35,
        trunk_radius_m: 0.06,
        canopy_radius_m: 0.78,
        canopy_height_m: 0.62,
        trunk_color: Vec3::new(0.13, 0.085, 0.050),
        canopy_color: Vec3::new(0.95, 1.0, 0.86),
        style: CanopyStyle::Round,
        seed: 0x5_417,
        lod: 0,
    };
    lod_data.push(build_lod_chain(&shrub));
    placement.push(VegSpeciesPlacement {
        layer: VegLayer::Shrub,
        min_spacing_m: SHRUB_SPACING_M,
        mix_weight: 1.0,
        scale_range: (0.6, 1.3),
        slope_limit: 0.46,
        altitude_band: (1600.0, 2700.0, 2300.0, 3000.0),
        clump_affinity: 0.60,
        min_grass_w: 0.28,
    });

    // Procedural foliage atlas (leaf clusters + crown shell + bark), built once
    // and shared by every plant on the body.
    let atlas = images.add(build_foliage_atlas());
    let material = materials.add(tree_material(TreeShadingExtension {
        atlas: atlas.clone(),
        material_atlas: images.add(build_foliage_material_atlas()),
        // Valid depth textures from the start (the `texture_depth_2d` bindings
        // have no usable fallback); `update_tree_material` swaps in the real
        // per-cascade maps each frame, and `shadow.config.x` stays 0 until then.
        sun_shadow_map_0: images.add(fallback_shadow_map()),
        sun_shadow_map_1: images.add(fallback_shadow_map()),
        sun_shadow_map_2: images.add(fallback_shadow_map()),
        sun_shadow_map_3: images.add(fallback_shadow_map()),
        ..default()
    }));

    // --- Octahedral impostor atlas (tree species only; shrubs are mesh-only) ---
    // Assign each tree species an atlas layer; build the albedo+coverage and
    // normal+depth atlases, the shared impostor material, and the off-screen
    // bake rig that fills them.
    let tree_species: Vec<usize> = placement
        .iter()
        .enumerate()
        .filter(|(_, p)| p.layer == VegLayer::Tree)
        .map(|(idx, _)| idx)
        .collect();
    let mut atlas_species: Vec<Option<u32>> = vec![None; placement.len()];
    for (layer, &sp) in tree_species.iter().enumerate() {
        atlas_species[sp] = Some(layer as u32);
    }
    let species_count = tree_species.len().min(IMPOSTOR_MAX_SPECIES) as u32;

    let layout = ImpostorAtlasLayout {
        cells: IMPOSTOR_CELLS,
        cell_px: IMPOSTOR_CELL_PX,
        species: species_count,
    };
    let albedo_atlas = images.add(make_impostor_atlas(layout));
    let normal_atlas = images.add(make_impostor_atlas(layout));

    // Per-species bounding geometry (authored units) the runtime billboard sizes
    // from; index by atlas layer.
    let mut species_geo = [Vec4::ZERO; IMPOSTOR_MAX_SPECIES];
    let mut max_tree_extent_m = 0.0f32;
    for (layer, &sp) in tree_species.iter().enumerate().take(IMPOSTOR_MAX_SPECIES) {
        let (center, radius) = tree_bounding_sphere(&lod_data[sp][0]);
        species_geo[layer] = Vec4::new(radius, center.y, 0.0, 0.0);
        max_tree_extent_m = max_tree_extent_m.max(center.y + radius);
    }

    let impostor_block = ImpostorParams {
        grid: Vec4::new(
            IMPOSTOR_CELLS as f32,
            species_count.max(1) as f32,
            IMPOSTOR_ALPHA_CUTOFF,
            0.0,
        ),
        atlas: Vec4::new(IMPOSTOR_CELL_FILL, 0.0, 0.0, 0.0),
        species_geo,
    };
    // One impostor material per clipmap ring (shared atlases; each gets its
    // ring's cross-fade band, written per frame by `update_tree_material`).
    let ring_impostor_materials: Vec<Handle<TreeImpostorMaterial>> = (0..TREE_RINGS.len())
        .map(|_| {
            impostor_materials.add(tree_impostor_material(TreeImpostorExtension {
                params: GrassParams::default(),
                impostor: impostor_block,
                albedo: albedo_atlas.clone(),
                normal: normal_atlas.clone(),
                ..default()
            }))
        })
        .collect();

    spawn_impostor_bake_rig(
        &mut commands,
        &mut bake_materials,
        &mut meshes,
        &lod_data,
        &tree_species,
        atlas.clone(),
        albedo_atlas,
        normal_atlas,
        species_count,
    );

    commands.insert_resource(SpeciesLibrary {
        placement: Arc::from(placement),
        lod_data,
        material,
        impostor_materials: ring_impostor_materials,
        atlas_species,
        max_tree_extent_m,
    });
}

/// Spawn the one-shot off-screen rig that bakes the octahedral atlas: for each
/// tree species and each of the `N×N` hemisphere view directions, one instance
/// of the recentred LOD0 mesh rotated so the bake camera sees that direction,
/// laid out in a grid (cells across, species stacked down). Two cameras render
/// the grid — one to the albedo+coverage atlas, one to the normal+depth atlas —
/// using the same instances on two layers. The rig renders for a handful of
/// frames (covering pipeline compilation) then `tick_impostor_bake` tears it
/// down; the atlases retain the captured content.
#[allow(clippy::too_many_arguments)]
fn spawn_impostor_bake_rig(
    commands: &mut Commands,
    bake_materials: &mut Assets<TreeBakeMaterial>,
    meshes: &mut Assets<Mesh>,
    lod_data: &[Vec<Arc<TreeMeshData>>],
    tree_species: &[usize],
    foliage_atlas: Handle<Image>,
    albedo_atlas: Handle<Image>,
    normal_atlas: Handle<Image>,
    species_count: u32,
) {
    if tree_species.is_empty() {
        return;
    }
    let n = IMPOSTOR_CELLS;
    // World radius the bounding sphere fills inside a 1×1 world-unit cell, and the
    // depth scale that maps cell-space view depth to [0,1].
    let cell_fit = IMPOSTOR_CELL_FILL * 0.5;
    let depth_scale = 0.5 / cell_fit;

    let albedo_mat = bake_materials.add(TreeBakeMaterial {
        params: BakeParams {
            mode: Vec4::new(0.0, depth_scale, 0.0, 0.0),
        },
        atlas: foliage_atlas.clone(),
    });
    let normal_mat = bake_materials.add(TreeBakeMaterial {
        params: BakeParams {
            mode: Vec4::new(1.0, depth_scale, 0.0, 0.0),
        },
        atlas: foliage_atlas,
    });

    for (layer, &sp) in tree_species.iter().enumerate().take(IMPOSTOR_MAX_SPECIES) {
        let (center, radius) = tree_bounding_sphere(&lod_data[sp][0]);
        let mesh = meshes.add(recenter_tree_mesh(&lod_data[sp][0], center));
        let scale = Vec3::splat(cell_fit / radius);
        for j in 0..n {
            for i in 0..n {
                let uv = Vec2::new((i as f32 + 0.5) / n as f32, (j as f32 + 0.5) / n as f32);
                let rot = impostor_bake_rotation(hemioct_decode(uv));
                let cell_xy = Vec3::new(i as f32 + 0.5, (layer as u32 * n + j) as f32 + 0.5, 0.0);
                let transform = Transform {
                    translation: cell_xy,
                    rotation: rot,
                    scale,
                };
                commands.spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(albedo_mat.clone()),
                    transform,
                    Visibility::Visible,
                    RenderLayers::layer(IMPOSTOR_BAKE_ALBEDO_LAYER),
                    ImpostorBakeRig,
                    Name::new("Impostor Bake (albedo)"),
                ));
                commands.spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(normal_mat.clone()),
                    transform,
                    Visibility::Visible,
                    RenderLayers::layer(IMPOSTOR_BAKE_NORMAL_LAYER),
                    ImpostorBakeRig,
                    Name::new("Impostor Bake (normal)"),
                ));
            }
        }
    }

    // The cameras frame the whole grid: `n` cells across, `n × species` down,
    // each a 1×1 world-unit cell, viewed orthographically from +Z.
    let grid_w = n as f32;
    let grid_h = (n * species_count.max(1)) as f32;
    let cam_center = Vec3::new(grid_w * 0.5, grid_h * 0.5, 0.0);
    let bake_camera = |order: isize, layer: usize, target: Handle<Image>, name: &'static str| {
        (
            Camera3d::default(),
            Camera {
                order,
                clear_color: ClearColorConfig::Custom(Color::NONE),
                ..default()
            },
            Hdr,
            Tonemapping::None,
            RenderTarget::Image(ImageRenderTarget::from(target)),
            Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::Fixed {
                    width: grid_w,
                    height: grid_h,
                },
                near: 0.1,
                far: 100.0,
                ..OrthographicProjection::default_3d()
            }),
            Transform::from_translation(cam_center + Vec3::Z * 10.0)
                .looking_at(cam_center, Vec3::Y),
            RenderLayers::layer(layer),
            ImpostorBakeRig,
            Name::new(name),
        )
    };
    commands.spawn(bake_camera(
        -20,
        IMPOSTOR_BAKE_ALBEDO_LAYER,
        albedo_atlas,
        "Impostor Bake Camera (albedo)",
    ));
    commands.spawn(bake_camera(
        -19,
        IMPOSTOR_BAKE_NORMAL_LAYER,
        normal_atlas,
        "Impostor Bake Camera (normal)",
    ));
}

/// Render the bake rig for a fixed number of frames (enough for the off-screen
/// pipelines to compile and the atlases to fill), then tear it down and flag the
/// impostor band ready. Until then `drive_veg_tiles` keeps the far band on the
/// LOD3 mesh at the mesh-only reach.
fn tick_impostor_bake(
    mut bake: ResMut<ImpostorBake>,
    rig: Query<Entity, With<ImpostorBakeRig>>,
    mut commands: Commands,
) {
    if bake.ready {
        return;
    }
    bake.frames += 1;
    if bake.frames >= IMPOSTOR_BAKE_FRAMES {
        for entity in &rig {
            commands.entity(entity).despawn();
        }
        bake.ready = true;
    }
}

/// Build the LOD0..LOD2 raw-mesh chain for a species template.
fn build_lod_chain(base: &TreeMeshParams) -> Vec<Arc<TreeMeshData>> {
    (0..TREE_LOD_COUNT as u32)
        .map(|lod| Arc::new(build_tree_mesh_data(&TreeMeshParams { lod, ..*base })))
        .collect()
}

/// Per-species mesh data for a given tile LOD, with shrubs skipped outside the
/// nearest band (too small to read far; bounds their geometry).
fn species_lod_for(library: &SpeciesLibrary, lod: usize) -> Vec<Option<Arc<TreeMeshData>>> {
    library
        .placement
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            if p.layer == VegLayer::Shrub && lod > 0 {
                None
            } else {
                let chain = &library.lod_data[idx];
                chain.get(lod.min(chain.len().saturating_sub(1))).cloned()
            }
        })
        .collect()
}

/// Keep the scatter-tile set around the **view anchor** (the render camera,
/// resolved body-fixed — see [`crate::rendering::view_anchor`]): trees exist
/// around whatever the camera is, flight or god view, with no craft anchoring.
#[allow(clippy::too_many_arguments)]
fn drive_veg_tiles(
    mut veg: ResMut<VegTiles>,
    library: Option<Res<SpeciesLibrary>>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    rendered_ground: Res<crate::terrain_registry::RenderedGroundRegistry>,
    mut flatten_registry: ResMut<TerrainFlattenRegistry>,
    bake: Res<ImpostorBake>,
    anchor: Res<ViewAnchor>,
    mut commands: Commands,
    mut diag: Local<u32>,
) {
    let Some(library) = library else {
        return;
    };
    if solar.states.is_none() {
        return;
    }

    let despawn_all = |veg: &mut VegTiles, commands: &mut Commands| {
        for (_, tile) in veg.tiles.drain() {
            if let Some(entity) = tile.entity {
                commands.entity(entity).despawn();
            }
        }
        veg.in_flight.clear();
    };

    // Leak-bisection kill switch (`THALOS_NO_SCATTER=1`): park the tree/shrub
    // scatter entirely (see `mem_diag::scatter_killed`).
    if crate::mem_diag::scatter_killed() {
        if veg.body.is_some() {
            despawn_all(&mut veg, &mut commands);
            veg.body = None;
        }
        return;
    }

    // Active body: the anchor's (nearest terrain-backed) body, when vegetated.
    let anchored = anchor.resolved.filter(|a| {
        sim.system
            .bodies
            .get(a.body)
            .is_some_and(|b| terrain_shading_style_for(b) == TerrainShadingStyle::Vegetated)
    });
    let Some(view) = anchored else {
        if veg.body.is_some() {
            despawn_all(&mut veg, &mut commands);
            veg.body = None;
        }
        return;
    };
    let body_id = view.body;
    if veg.body != Some(body_id) {
        despawn_all(&mut veg, &mut commands);
        veg.body = Some(body_id);
        let radius_m = sim.system.bodies[body_id].radius_m;
        veg.lattices = TREE_RINGS
            .iter()
            .map(|r| TileLattice::for_body(radius_m, r.tile_size_m))
            .collect();
    }

    let radius_m = view.radius_m;
    let Some(height_source) = height_sources.get(body_id) else {
        return;
    };
    let mirror = rendered_ground.get(body_id);

    let cam_dir = view.cam_dir;
    let ground_h = view.ground_h_m;
    let agl = view.agl_m;
    // Investigation trace, opt-in: this is an instrument for "why don't trees
    // build here", not a health gauge — nothing in `just diag` reads it, and at
    // ~1.5 s cadence it was 60 % of every record in the whole diagnostics lane.
    // Set THALOS_VEG_DIAG=1 when investigating scatter residency.
    *diag = diag.wrapping_add(1);
    if veg_diag_enabled() && *diag % 90 == 1 {
        // Split empty (cleared/wrong-biome) vs tree-bearing tiles, and the
        // distance to the nearest tree-bearing one — the decisive signal for
        // "trees don't build" vs "trees build but are far / cleared".
        let nonempty = veg.tiles.values().filter(|t| t.entity.is_some()).count();
        let nearest_tree_km = veg
            .tiles
            .iter()
            .filter(|(_, t)| t.entity.is_some())
            .filter_map(|(rk, _)| veg.lattices.get(rk.ring as usize)?.frame(rk.key))
            .map(|(center, _)| center.angle_between(cam_dir) * radius_m)
            .fold(f64::INFINITY, f64::min)
            / 1000.0;
        info!(
            target: "thalos::diagnostic::vegetation",
            event = "drive_gauge",
            body_id,
            ground_height_m = ground_h,
            agl_m = agl,
            tiles = veg.tiles.len(),
            nonempty,
            nearest_tree_km,
            in_flight = veg.in_flight.len(),
            bake_ready = bake.ready,
            view_speed_m_s = view.speed_m_s,
            "vegetation drive gauge"
        );
    }
    if agl > TREE_DESPAWN_AGL_M {
        if !veg.tiles.is_empty() || !veg.in_flight.is_empty() {
            despawn_all(&mut veg, &mut commands);
        }
        return;
    }

    let lattices = veg.lattices.clone();
    if lattices.len() != TREE_RINGS.len() {
        return;
    }
    let arc_dist = |center_dir: DVec3| -> f64 { center_dir.angle_between(cam_dir) * radius_m };

    // Only ring 0 is active until the impostor atlas is baked — the coarse rings
    // are impostor-only, so they wait for it (a second or two into the session).
    let rings_active = if bake.ready { TREE_RINGS.len() } else { 1 };
    let far_lod = TREE_LOD_COUNT - 1;

    // Build out to `build_hi`: the outer fade-out is fully covered (`+ 2·band`,
    // matching `tree_ring_fade`) plus a tile of look-ahead, so tiles finish
    // building while still scaled ~0 (invisible build, no pop-in). Despawn lags a
    // *further* tile, so a tile is never removed right at the edge where it might
    // immediately be needed again — combined with the overlap-full fade and the
    // keep-old-until-new re-LOD, the handoff never shows a gap. Ring 0 before the
    // atlas bakes uses the short mesh-only reach (LOD3 far band).
    let ring_build_hi = |ring_idx: usize| -> f64 {
        let r = &TREE_RINGS[ring_idx];
        if ring_idx == 0 && !bake.ready {
            TREE_MESH_ONLY_REACH_M + r.tile_size_m
        } else {
            r.outer_m + 2.0 * tree_ring_band_m(r) as f64 + r.tile_size_m
        }
    };
    let ring_despawn_reach =
        |ring_idx: usize| -> f64 { ring_build_hi(ring_idx) + TREE_RINGS[ring_idx].tile_size_m };

    // Despawn tiles past their ring's despawn reach (or whose ring is no longer
    // active). A re-LOD/rebuild of an in-range tile is NOT a despawn — it keeps
    // the old mesh until the new is ready (see `finalize_veg_tiles`).
    let stale: Vec<RingTileKey> = veg
        .tiles
        .keys()
        .filter(|rk| {
            let ring = rk.ring as usize;
            ring >= rings_active
                || lattices[ring]
                    .frame(rk.key)
                    .is_none_or(|(center, _)| arc_dist(center) > ring_despawn_reach(ring))
        })
        .copied()
        .collect();
    for rk in stale {
        if let Some(tile) = veg.tiles.remove(&rk)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
        }
    }

    if agl > TREE_MAX_AGL_M {
        return;
    }

    let slots = TREE_MAX_IN_FLIGHT.saturating_sub(veg.in_flight.len());
    if slots == 0 {
        return;
    }

    // Gather candidate tiles across every active ring, nearest first. A tile is a
    // candidate when it's missing, OR (ring 0) its baked LOD / impostor-ness no
    // longer matches its distance — the rebuild keeps the old mesh until the new
    // is ready (no vanish). Rings overlap by a fade band (cross-fade) and extend a
    // tile beyond their outer edge (invisible build).
    let speed = view.speed_m_s.max(0.0);
    let mut candidates: Vec<(f64, RingTileKey, usize, bool, bool)> = Vec::new();
    for ring_idx in 0..rings_active {
        let ring = &TREE_RINGS[ring_idx];
        let lat = lattices[ring_idx];
        let band = tree_ring_band_m(ring) as f64;
        // Cover the full fade-in region (`inner - 2·band`, matching the
        // overlap-full `tree_ring_fade`) so a ring's near tiles exist, already
        // scaled ~0, before the finer ring fades out over them.
        let lo = (ring.inner_m - 2.0 * band).max(0.0);
        let hi = ring_build_hi(ring_idx);
        let center_key = lat.key_of(cam_dir);
        let window = (hi / (ring.tile_size_m * 0.5)).ceil() as i64;
        for dy in -window..=window {
            for dx in -window..=window {
                let key = TileKey {
                    face: center_key.face,
                    x: center_key.x + dx,
                    y: center_key.y + dy,
                };
                let rk = RingTileKey {
                    ring: ring_idx as u8,
                    key,
                };
                if veg.in_flight.contains_key(&rk) {
                    continue;
                }
                let Some((center, _)) = lat.frame(key) else {
                    continue;
                };
                let d = arc_dist(center);
                if d < lo || d > hi {
                    continue;
                }
                // Ring 0 runs the mesh-LOD cascade + near impostor band; coarse
                // rings are always impostor groves.
                //
                // LOD is keyed by the **slant** distance (ground arc + altitude),
                // not the ground arc alone: from the air a tree directly below is
                // ~0 m of ground distance but kilometres away, so a ground-only
                // metric picks the close high-detail mesh and you see LOD0 meshes
                // (which read worse than the impostor) straight down. Folding AGL
                // in makes everything below the climbing craft fall back to the
                // impostor band, while a low pass (agl ≈ 0) is unchanged.
                // Plus the motion look-ahead: a fast view keys its LOD off
                // where it is about to be, not where it is (TREE_MOTION_LEAD_S).
                let view_d =
                    (d * d + agl.max(0.0) * agl.max(0.0)).sqrt() + speed * TREE_MOTION_LEAD_S;
                let (desired, want_impostor) = if ring_idx == 0 {
                    let l = lod_for_dist(view_d);
                    (l, l == far_lod && bake.ready)
                } else {
                    (far_lod, true)
                };
                let missing = !veg.tiles.contains_key(&rk);
                match veg.tiles.get(&rk) {
                    Some(tile) if tile.lod == desired && tile.impostor == want_impostor => continue,
                    _ => candidates.push((d, rk, desired, want_impostor, missing)),
                }
            }
        }
    }
    // Cold fill: coverage first, at the cheapest representation that reads as a
    // forest. Missing mesh-band tiles (outside the innermost LOD0 band) are
    // demoted to an impostor build now; once the fill drains below the
    // threshold, the normal LOD-mismatch pass upgrades them nearest-first.
    let missing_count = candidates.iter().filter(|c| c.4).count();
    if bake.ready && missing_count > TREE_COLD_FILL_MISSING {
        for (_, rk, desired, want_impostor, missing) in candidates.iter_mut() {
            if *missing && rk.ring == 0 && !*want_impostor && *desired > 0 {
                *desired = far_lod;
                *want_impostor = true;
            }
        }
    }
    // Nearest first, with re-LOD upgrades penalized against missing tiles
    // (coverage outranks refinement — see TREE_UPGRADE_PENALTY).
    candidates.sort_by(|a, b| {
        let pa = a.0 * if a.4 { 1.0 } else { TREE_UPGRADE_PENALTY };
        let pb = b.0 * if b.4 { 1.0 } else { TREE_UPGRADE_PENALTY };
        pa.total_cmp(&pb)
    });

    // Sea level is the project datum: the constant 0 m (= reference radius), the
    // shoreline the bimodal continent/ocean generator (Slice 1) puts at height 0.
    // Trees require `height > sea_level + 1 m`, so the seabed stays bare.
    let sea_level_m = 0.0;
    let flatten_exclusion = flatten_registry
        .handle(body_id)
        .read()
        .ok()
        .and_then(|guard| thalos_terrain::nearest_flatten(&guard, cam_dir));

    let ground = mirror.as_ref();
    // Dedicated bounded pool (see `veg_scatter_pool`): the shared
    // AsyncComputeTaskPool's width was the cold-fill throughput ceiling, and
    // scatter contended there with Avian's collider optimisation.
    let pool = veg_scatter_pool();
    let revision = height_source.revision();
    let mut dispatched = 0usize;
    for (_, rk, desired, want_impostor, _) in candidates {
        if dispatched >= slots {
            break;
        }
        let ring = &TREE_RINGS[rk.ring as usize];
        let lat = lattices[rk.ring as usize];
        // Layers this build can actually draw: shrubs render only in the
        // innermost mesh band (`species_lod_for`), so every other build is
        // trees-only — and skips the 2.3 m shrub grid, the densest (most
        // expensive) part of placement.
        let needed_mask = if !want_impostor && desired == 0 {
            VegLayer::Tree.mask() | VegLayer::Shrub.mask()
        } else {
            VegLayer::Tree.mask()
        };
        // Re-LOD fast path: placement is deterministic, so a tile re-baked at
        // the same height revision with a cached placement covering the needed
        // layers only re-runs the combine — no height sampling at all (which
        // is also why the residency gate below is skipped for it).
        let cached: Option<(u32, Arc<VegScatterTile>)> = veg.tiles.get(&rk).and_then(|t| {
            let (mask, scatter) = t.scatter.as_ref()?;
            (t.built_revision == revision && mask & needed_mask == needed_mask)
                .then(|| (*mask, Arc::clone(scatter)))
        });
        if cached.is_none() {
            // Far rings tolerate coarser terrain (their tiles are large), so the
            // residency threshold scales with tile size (mirrors the grass gate).
            let texel_limit = ((ring.tile_size_m * 0.5) as f32).max(TREE_MAX_TERRAIN_TEXEL_M);
            if let Some(ground) = ground {
                let Some((center, _)) = lat.frame(rk.key) else {
                    continue;
                };
                match ground.best_resident_texel_m(center.as_vec3()) {
                    Some(texel) if texel <= texel_limit => {}
                    _ => continue,
                }
            }
        }
        let input = VegScatterInput {
            key: rk.key,
            lattice: lat,
            radius_m,
            height_source: Arc::clone(&height_source),
            species: Arc::clone(&library.placement),
            seed: TREE_SEED,
            sea_level_m,
            flatten_exclusion,
            spacing_scale: ring.spacing_scale,
            keep_fraction: ring.keep_fraction,
            layer_mask: needed_mask,
        };
        // Only ring 0 re-LODs on approach, so only its placement is worth
        // keeping (the coarse rings would be pure memory).
        let keep_scatter = rk.ring == 0;
        let task = if want_impostor {
            // Impostor band: one natural-size (`grove_scale = 1`) billboard quad
            // per tree; count is bounded by the ring's spacing/keep decimation.
            let atlas_species = library.atlas_species.clone();
            let grove = ring.grove_scale;
            pool.spawn(async move {
                let started = std::time::Instant::now();
                let (mask, tile, reused) = match cached {
                    Some((mask, tile)) => (mask, tile, true),
                    None => (needed_mask, Arc::new(build_scatter_tile(&input)?), false),
                };
                let mesh = combine_impostor_tile_mesh(&tile.instances, &atlas_species, grove)?;
                Some(VegTileBuild {
                    mesh,
                    impostor: true,
                    center_surface_body_m: tile.center_surface_body_m,
                    built_revision: tile.built_revision,
                    center_height_m: tile.center_height_m,
                    build_micros: started.elapsed().as_micros() as u64,
                    instance_count: tile.instances.len() as u32,
                    reused_placement: reused,
                    scatter: keep_scatter.then_some((mask, tile)),
                })
            })
        } else {
            // Near/mid band: a batched mesh-LOD tile.
            let species_lod = species_lod_for(&library, desired);
            pool.spawn(async move {
                let started = std::time::Instant::now();
                let (mask, tile, reused) = match cached {
                    Some((mask, tile)) => (mask, tile, true),
                    None => (needed_mask, Arc::new(build_scatter_tile(&input)?), false),
                };
                let mesh = combine_tree_tile_mesh(&tile.instances, &species_lod)?;
                Some(VegTileBuild {
                    mesh,
                    impostor: false,
                    center_surface_body_m: tile.center_surface_body_m,
                    built_revision: tile.built_revision,
                    center_height_m: tile.center_height_m,
                    build_micros: started.elapsed().as_micros() as u64,
                    instance_count: tile.instances.len() as u32,
                    reused_placement: reused,
                    scatter: keep_scatter.then_some((mask, tile)),
                })
            })
        };
        veg.in_flight
            .insert(rk, (task, revision, desired, want_impostor));
        dispatched += 1;
    }
}

/// Poll in-flight builds; spawn each finished tile's batched mesh as a root-grid
/// big_space child. A rebuild/re-LOD spawns the new entity, then despawns the old
/// one (it stays visible until the new mesh is ready — no vanish).
fn finalize_veg_tiles(
    mut veg: ResMut<VegTiles>,
    solar: Res<SolarSystemState>,
    root: Option<Res<RealSpaceRoot>>,
    library: Option<Res<SpeciesLibrary>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut stats: Local<VegBuildStats>,
    mut commands: Commands,
) {
    if veg.in_flight.is_empty() {
        return;
    }
    let (Some(library), Some(states), Some(root), Some(body_id)) =
        (library, solar.states.as_deref(), root, veg.body)
    else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    let mut finished: Vec<(RingTileKey, u64, usize, bool, Option<VegTileBuild>)> = Vec::new();
    veg.in_flight.retain(
        |rk, (task, revision, lod, want)| match block_on(poll_once(task)) {
            Some(result) => {
                finished.push((*rk, *revision, *lod, *want, result));
                false
            }
            None => true,
        },
    );

    let orientation = body_state.orientation.normalize();
    for (rk, revision, lod, want_impostor, result) in finished {
        let old_entity = veg.tiles.get(&rk).and_then(|t| t.entity);

        let Some(build) = result else {
            if let Some(old) = old_entity {
                commands.entity(old).despawn();
            }
            veg.tiles.insert(
                rk,
                BuiltTile {
                    entity: None,
                    built_revision: revision,
                    center_height_m: 0.0,
                    lod,
                    impostor: want_impostor,
                    scatter: None,
                },
            );
            continue;
        };
        stats.record(
            build.build_micros,
            build.instance_count,
            build.reused_placement,
        );

        let center = build.center_surface_body_m;
        let center_world = body_state.position + orientation * center;
        let (cell, local) = real_space_grid().translation_to_grid(center_world);
        let transform = Transform {
            translation: local,
            rotation: orientation.as_quat(),
            scale: Vec3::ONE,
        };
        let visual = VegTileVisual {
            body_id,
            center_surface_body: center,
        };
        // Explicit frustum-cull AABB: these meshes are `RENDER_WORLD`-only, so Bevy
        // never auto-computes one (see `docs/world/vegetation.md`), and without it the
        // full 360° ring of tree tiles around the camera is processed every frame
        // (in every view). Per-view culling still feeds the sun-shadow pass from its
        // own frustum, so off-screen casters whose shadows fall into view stay.
        let local_aabb = build.mesh.compute_aabb();
        let mesh = Mesh3d(meshes.add(build.mesh));
        // Impostor tiles carry a different material and don't cast shadows (past
        // the shadow cutoff); mesh tiles keep the standard `TreeMaterial`.
        let entity = if build.impostor {
            let impostor_material = library
                .impostor_materials
                .get(rk.ring as usize)
                .cloned()
                .unwrap_or_else(|| library.impostor_materials[0].clone());
            // Impostor quads are degenerate in the mesh (all 4 corners share the
            // tree base; the vertex shader billboards them into camera-facing cards),
            // so the mesh AABB bounds only the bases. Pad it by the tallest canopy a
            // card can reach (`max_tree_extent × max scale × grove_scale`) so a tile
            // whose bases are just off-screen but whose cards are visible isn't
            // wrongly culled.
            let aabb = local_aabb.map(|mut a| {
                let grove = TREE_RINGS[rk.ring as usize].grove_scale;
                a.half_extents += Vec3A::splat(library.max_tree_extent_m * TREE_SCALE_MAX * grove);
                a
            });
            // Near impostor rings also cast into the custom sun-shadow
            // cascades so mid-distance trees ground with a shadow — not just
            // the mesh trees. The billboard orients to `view.world_position`,
            // which in the cascade caster pass is the cascade camera (up-sun),
            // so it faces the SUN and renders the canopy silhouette from the
            // sun angle (octahedral atlas sampled there) → casts the right
            // shape. The coarse FAR rings (past `TREE_SHADOW_CASTER_MAX_M`) do
            // NOT cast — see that constant. `NotShadowCaster` still excludes
            // every tile from Bevy's stock CSM.
            let ring_casts = TREE_RINGS
                .get(rk.ring as usize)
                .is_some_and(|r| r.outer_m <= TREE_SHADOW_CASTER_MAX_M);
            let layers = if ring_casts {
                RenderLayers::from_layers(&[
                    SHIP_LAYER,
                    crate::rendering::sun_shadow::SHADOW_CASTER_LAYER,
                ])
            } else {
                RenderLayers::layer(SHIP_LAYER)
            };
            let mut tile_cmd = commands.spawn((
                mesh,
                MeshMaterial3d(impostor_material),
                transform,
                cell,
                Visibility::Inherited,
                layers,
                NotShadowCaster,
                ChildOf(root.entity),
                visual,
                Name::new("Vegetation Impostor Tile"),
            ));
            if let Some(aabb) = aabb {
                tile_cmd.insert(aabb);
            }
            tile_cmd.id()
        } else {
            // Mesh tiles have real geometry, so the computed AABB is exact.
            let mut tile_cmd = commands.spawn((
                mesh,
                MeshMaterial3d(library.material.clone()),
                transform,
                cell,
                Visibility::Inherited,
                // Also visible to the sun-shadow camera so mesh trees cast
                // into the directional shadow map (the leaf alpha-discard
                // gives leaf-shaped shadows). The impostor tiles now cast too
                // (oriented to the sun in the caster pass), so distant trees
                // ground as well — see that branch above.
                RenderLayers::from_layers(&[
                    SHIP_LAYER,
                    crate::rendering::sun_shadow::SHADOW_CASTER_LAYER,
                ]),
                ChildOf(root.entity),
                visual,
                Name::new("Vegetation Tile"),
            ));
            if let Some(aabb) = local_aabb {
                tile_cmd.insert(aabb);
            }
            tile_cmd.id()
        };
        // Replace the old (previous-LOD) entity only now that the new one exists.
        if let Some(old) = old_entity {
            commands.entity(old).despawn();
        }
        veg.tiles.insert(
            rk,
            BuiltTile {
                entity: Some(entity),
                built_revision: build.built_revision,
                center_height_m: build.center_height_m,
                lod,
                impostor: build.impostor,
                scatter: build.scatter,
            },
        );
    }
}

/// Re-anchor every scatter tile in f64 each frame (the grass / runway pattern).
fn update_veg_transforms(
    solar: Res<SolarSystemState>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut tiles: Query<(&VegTileVisual, &mut CellCoord, &mut Transform)>,
) {
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let Ok(grid) = root_grid.single() else {
        return;
    };
    for (tile, mut cell, mut transform) in &mut tiles {
        let Some(state) = states.get(tile.body_id) else {
            continue;
        };
        let orientation = state.orientation.normalize();
        let center_world = state.position + orientation * tile.center_surface_body;
        let (next_cell, local) = grid.translation_to_grid(center_world);
        *cell = next_cell;
        transform.translation = local;
        transform.rotation = orientation.as_quat();
    }
}

/// Per-frame tree/shrub shading on the single shared [`TreeMaterial`]: sun
/// direction + flux, a slowly veering wind (drives the canopy sway), the shared
/// `thalos::lighting` sky inputs, and the scale-fade band + craft anchor (so the
/// fade is zoom-independent). Mirrors `rendering::grass::update_grass_material`.
#[allow(clippy::too_many_arguments)]
fn update_tree_material(
    library: Option<Res<SpeciesLibrary>>,
    veg: Res<VegTiles>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    time: Res<Time>,
    exposure: Res<CameraExposure>,
    bake: Res<ImpostorBake>,
    anchor: Res<ViewAnchor>,
    mut materials: ResMut<Assets<TreeMaterial>>,
    mut impostor_materials: ResMut<Assets<TreeImpostorMaterial>>,
) {
    let Some(library) = library else {
        return;
    };
    let (Some(body_id), Some(states)) = (veg.body, solar.states.as_deref()) else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    let star_pos = states.first().map(|s| s.position).unwrap_or(DVec3::ZERO);
    let offset = star_pos - body_state.position;
    let sun_dir = offset.normalize_or_zero().as_vec3();
    let au_over_d = (AU_M / offset.length().max(1.0)) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;

    let t = time.elapsed_secs();
    // Local vertical at the VIEW (the tiles exist around the view anchor, so
    // the sky/wind vertical must match — the craft may be on the far side of
    // the planet).
    let up = anchor
        .resolved
        .filter(|a| a.body == body_id)
        .map(|a| (body_state.orientation * a.cam_dir).as_vec3())
        .unwrap_or_else(|| {
            (sim.simulation.ship_state().position - body_state.position)
                .normalize_or_zero()
                .as_vec3()
        });
    let seed = if up.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let east = seed.cross(up).normalize_or_zero();
    let north = up.cross(east);
    let veer = t * 0.025;
    let wind_dir = (east * veer.cos() + north * veer.sin()).normalize_or_zero();

    let (tau, strength) = sim
        .system
        .bodies
        .get(body_id)
        .and_then(|b| b.terrestrial_atmosphere.as_ref())
        .and_then(|a| a.scattering.as_ref())
        .map(|s| (Vec3::from_array(s.vertical_optical_depth), s.strength))
        .unwrap_or((Vec3::ZERO, 0.0));

    let sun = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, flux);
    let wind = Vec4::new(wind_dir.x, wind_dir.y, wind_dir.z, TREE_WIND_SWAY_M);
    let sky_up = Vec4::new(up.x, up.y, up.z, 0.0);
    let sky_tau = Vec4::new(tau.x, tau.y, tau.z, strength);

    // Fade reference = the VIEW (`view.world_position` in the shader, offset 0):
    // the scale-fade is a per-instance LOD keyed by slant distance from the eye,
    // matching the build driver's slant-keyed LOD pick. Offset 0 is inherently
    // origin-invariant and this-frame-exact — the former craft-anchored offset
    // was a workaround for the main-world camera transform lagging a frame,
    // which the shader's own view position doesn't.
    let anchor = Vec4::ZERO;

    // Each material carries the shared lighting + ITS clipmap ring's cross-fade
    // band: `time_fade = (time, near_edge, far_edge, band)`. The shader grows each
    // instance from zero across [near, far], so adjacent rings cross-fade through
    // their shared boundary (no hard size step between groves) and the outermost
    // edge melts away seamlessly.
    let make_params = |near: f32, far: f32, band: f32| GrassParams {
        sun_dir: sun,
        wind,
        time_fade: Vec4::new(t, near, far, band),
        sky_up,
        sky_tau,
        anchor,
    };

    // Mesh material (ring 0 only): its trees are all in the near band, so once the
    // atlas is baked it never actually fades; pre-bake, the LOD3 far band fades
    // out near the mesh-only reach so its edge isn't a hard ring.
    if let Some(mut material) = materials.get_mut(&library.material) {
        let (near, far, band) = if bake.ready {
            tree_ring_fade(0)
        } else {
            TREE_MESH_ONLY_FADE
        };
        material.extension.params = make_params(near, far, band);
    }

    // One impostor material per clipmap ring, each with its own cross-fade band.
    for (idx, handle) in library.impostor_materials.iter().enumerate() {
        let Some(mut material) = impostor_materials.get_mut(handle) else {
            continue;
        };
        let (near, far, band) = tree_ring_fade(idx);
        material.extension.params = make_params(near, far, band);
    }
}

/// Fan the live cloud sun-transmittance cascade onto the tree + impostor
/// materials — in `Last`, beside the tile fan (`tiles::apply_cloud_shadow`) and
/// the hull/structure fan (`craft::apply_craft_shadow`), and for the same
/// reason: the game's cloud driver resolves the cascade frame in `PostUpdate`
/// and the compute pass marches THAT frame the same frame. This fan used to
/// ride `update_tree_material` (Update, i.e. strictly BEFORE the driver), so
/// trees sampled the freshly-marched map through a deterministically
/// one-frame-stale block — and the block's render-space anchors move at
/// orbital speed (~500 m per frame at warp 1), so canopy cloud shadows
/// jittered 2–16 texels against the exact shadow on the ground beside them.
/// Paused captures hid it, which is how it passed screenshot verification
/// (reviews/20260730T011353Z §5).
fn apply_tree_cloud_shadow(
    library: Option<Res<SpeciesLibrary>>,
    cloud_shadow: Option<Res<thalos_body_render::clouds::CloudShadowMap>>,
    mut materials: ResMut<Assets<TreeMaterial>>,
    mut impostor_materials: ResMut<Assets<TreeImpostorMaterial>>,
) {
    let (Some(library), Some(cloud)) = (library, cloud_shadow) else {
        return;
    };
    let block = cloud.block();
    if let Some(mut material) = materials.get_mut(&library.material) {
        material.extension.cloud_shadow = block;
        material.extension.cloud_shadow_map = cloud.handle.clone();
    }
    for handle in &library.impostor_materials {
        let Some(mut material) = impostor_materials.get_mut(handle) else {
            continue;
        };
        material.extension.cloud_shadow = block;
        material.extension.cloud_shadow_map = cloud.handle.clone();
    }
}

/// Periodically rebuild tiles whose underlying height shifted (finer atlas tile
/// streamed in, or a flatten pad installed): forget the stale tile so
/// `drive_veg_tiles` re-dispatches it (the old mesh stays until the new is ready).
fn check_veg_rebuilds(
    mut veg: ResMut<VegTiles>,
    height_sources: Res<HeightSourceRegistry>,
    time: Res<Time>,
) {
    veg.rebuild_timer += time.delta_secs();
    if veg.rebuild_timer < TREE_REBUILD_CHECK_S {
        return;
    }
    veg.rebuild_timer = 0.0;

    let Some(body_id) = veg.body else {
        return;
    };
    let Some(source) = height_sources.get(body_id) else {
        return;
    };
    let revision = source.revision();
    let lattices = veg.lattices.clone();
    if lattices.len() != TREE_RINGS.len() {
        return;
    }

    let mut rebuilt = 0usize;
    for (rk, tile) in veg.tiles.iter_mut() {
        if tile.built_revision == revision {
            continue;
        }
        let Some((center_dir, _)) = lattices[rk.ring as usize].frame(rk.key) else {
            continue;
        };
        let Some(h) = source.sample_height_m(center_dir.as_vec3(), TREE_GROUND_LOD_M) else {
            continue;
        };
        if tile.entity.is_some()
            && (h - tile.center_height_m).abs() > TREE_REBUILD_DELTA_M
            && rebuilt < TREE_MAX_REBUILDS_PER_TICK
        {
            // Force a re-dispatch by invalidating the recorded LOD; drive sees a
            // mismatch and rebuilds (old mesh kept until the new is ready).
            tile.lod = usize::MAX;
            rebuilt += 1;
        } else {
            tile.built_revision = revision;
            if tile.entity.is_some() {
                tile.center_height_m = h;
            }
        }
    }
}

/// Opt-in gate for the vegetation drive trace (`THALOS_VEG_DIAG=1`).
fn veg_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("THALOS_VEG_DIAG").is_ok_and(|value| matches!(value.as_str(), "1" | "true"))
    })
}
