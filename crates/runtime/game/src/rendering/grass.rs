//! Grass-blade decoration driver — clipmap rings.
//!
//! Maintains a camera-local set of body-fixed grass tiles around the player on
//! the nearest vegetated body, using the tile builder + material in
//! `thalos_body_render::ground::vegetation`. The driver owns the tile
//! lifecycle only — which tiles exist, where they are anchored, what the wind
//! and sun are doing; all placement/geometry logic lives in the engine crate.
//!
//! **Clipmap.** Grass reaches the horizon through concentric LOD rings: each
//! ring is a coarser cube-sphere lattice (tile size doubles outward), so each
//! is a thin annulus of a bounded number of tiles. Near rings use the full
//! curved blade at high density; far rings use a cheap wide "clump" blade at low
//! density but widened so ground *coverage* stays roughly constant (no bald
//! ground). Each ring has its own material carrying a near/far/band fade, so
//! adjacent rings cross-fade through their shared boundary (`grass.wgsl`).
//!
//! Anchoring follows the runway pattern exactly (`runway::update_runway_transform`):
//! each tile is a **root-grid big_space child** whose position is recomputed
//! in f64 every frame from the body's state, so the f32 `Transform.rotation`
//! only ever acts on the tile's small vertex offsets and the grass stays
//! rock-steady under high warp.
//!
//! Tiles are built on `AsyncComputeTaskPool` against the body's
//! [`HeightSource`] (GPU-atlas mirror with CPU fallback — the same source the
//! terrain collider uses), so blades sit on the rendered ground. A periodic
//! revision check rebuilds tiles whose underlying height data shifted (a finer
//! LOD tile streamed in, or a terrain-flatten pad — e.g. the runway — was
//! installed after the grass was built).

use std::collections::HashMap;
use std::sync::Arc;

use bevy::camera::primitives::MeshAabb;
use bevy::camera::visibility::RenderLayers;
use bevy::light::NotShadowCaster;
use bevy::math::{DVec3, Vec3, Vec4};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::{
    AU_M, GrassBladeLod, GrassMaterial, GrassProfile, GrassTileBuildInput, GrassTileKey,
    GrassTileMesh, LIGHT_AT_1AU, ScatterRegion, ScatterTreatment, TerrainShadingStyle,
    build_grass_card_atlas, build_grass_tile_mesh, fallback_shadow_map, grass_tile_frame,
    grass_tile_key, grass_tiles_per_side,
};
use thalos_physics_local::HeightSourceRegistry;
use thalos_terrain::TerrainFlatten;
use thalos_world::BodyId;

use crate::SimStage;
use crate::coords::SHIP_LAYER;
use crate::graphics_settings::GraphicsSettings;
use crate::rendering::ground_terrain::terrain_shading_style_for;
use crate::rendering::real_space::{RealSpaceRoot, real_space_grid};
use crate::rendering::sun_shadow::SunShadowState;
use crate::rendering::types::CameraExposure;
use crate::rendering::view_anchor::ViewAnchor;
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};
use crate::structures::{StructureKind, StructurePlacement, StructureRegistry, StructureSite};

// ── Clipmap rings ─────────────────────────────────────────────────────────────
/// One LOD ring: a coarser lattice with its own density / blade size / fade.
struct GrassRing {
    /// Metric tile side at a cube-face centre.
    tile_size_m: f64,
    /// Inner ground distance the ring covers (= previous ring's outer).
    inner_m: f64,
    /// Outer ground distance the ring covers.
    outer_m: f64,
    /// Candidate (placement-point) density per m² before gates.
    density_per_m2: f32,
    /// Blades-per-clump LOD multiplier in `(0, 1]` applied to the grass profile's
    /// blade count. Coverage = density × (profile blades × clump_scale), but only
    /// `density` pays the placement gate — so near rings get a thick fluffy carpet
    /// cheaply, far rings thin each tuft and lean on wider blades.
    clump_scale: f32,
    /// Blade width multiplier (constant-coverage rule: density ↓ ⇒ width ↑).
    width_scale: f32,
    /// Blade height multiplier.
    height_scale: f32,
    blade_lod: GrassBladeLod,
    /// Forest-cull strength `[0, 1]`: how aggressively this ring thins grass
    /// under tree canopy (occluded → pure overdraw). Near rings keep all grass
    /// (`0`); far rings ramp up so distant grass survives only on open plains.
    forest_cull: f32,
}

/// The clipmap: near full-detail blades → one far wide-clump ring → **terrain
/// albedo carries the rest** (the shared `landcover` field paints the ground the
/// same grass colour, so beyond the blades the field still reads as grass — see
/// `docs/vegetation.md`). Cut from 5 rings/1.5 km to 3 rings/~340 m: the far
/// rings were heavy churn + visible LOD bands for blades that, from any altitude,
/// the terrain albedo already covers. Gentle width progression + two Full rings
/// keep the ring boundaries subtle.
const GRASS_RINGS: [GrassRing; 3] = [
    GrassRing {
        tile_size_m: 25.0,
        inner_m: 0.0,
        // Full curved blades only out to 25 m — the immediate foreground where blade
        // detail actually reads. This was 0–60 m, but the cost is VERTEX-bound and the
        // Full blade is the most expensive (7 verts), so shrinking it to 25 m is an
        // ~80 % area cut on the priciest geometry; the cheap Wide ring covers 25 m on.
        outer_m: 25.0,
        // Each accepted point grows a full fluffy fountain tuft (~14–18 blades), so
        // the carpet reads thick from fewer, denser clumps rather than many spikes.
        density_per_m2: 14.0,
        clump_scale: 1.0,
        width_scale: 1.0,
        height_scale: 1.0,
        blade_lod: GrassBladeLod::Full,
        // Near band: keep all grass, including the forest floor between trunks.
        forest_cull: 0.0,
    },
    GrassRing {
        tile_size_m: 50.0,
        inner_m: 25.0,
        // Wide-blade band shortened 160 → 110 m: it's the largest remaining vertex
        // sink, and the clump CARDS are validated to read as a green grass surface at
        // distance (preview `grass_field_card_far`), so hand off to them sooner. Cheap
        // 3-vert Wide blade, lower density + wider blades (constant-coverage).
        outer_m: 110.0,
        density_per_m2: 4.0,
        clump_scale: 0.5,
        width_scale: 2.0,
        height_scale: 1.1,
        blade_lod: GrassBladeLod::Wide,
        forest_cull: 0.35,
    },
    GrassRing {
        // Reach restored to 340 m / 3.5 per m² now that the depth prepass gives the
        // foliage early-Z: the grazing-horizon overdraw this band caused is rejected
        // before shading, so reaching further is affordable again (it was trimmed to
        // 270 m as a pre-prepass stopgap). The durable path to the true horizon is
        // still efficient drawing (instancing/indirect + clump-card billboards).
        // Far band → clump CARDS (one crossed-quad billboard tuft per clump, ~8
        // verts vs ~100): the cheapest representation, for the band where blades are
        // tiny anyway. The procedural tuft alpha reads coarse up close but is
        // sub-pixel at 160 m+; the terrain albedo carries beyond. (If it reads poorly
        // at distance in-game, switch `blade_lod` back to `Wide` — one line.)
        tile_size_m: 100.0,
        inner_m: 110.0,
        outer_m: 340.0,
        density_per_m2: 2.5,
        clump_scale: 0.4,
        width_scale: 2.8,
        height_scale: 1.2,
        blade_lod: GrassBladeLod::Card,
        forest_cull: 0.7,
    },
];

/// The two grass *types* the meadow distributes per clump (blended by the
/// landcover moisture field): thick short fluffy grass on the drier ground,
/// longer wispier grass on the wetter. This is the single place to reshape the
/// in-game grass look — thickness, length, and fluffiness all live in these two
/// [`GrassProfile`]s; later bodies can carry their own pair.
const GRASS_PROFILE_DRY: GrassProfile = GrassProfile::fluffy_short();
const GRASS_PROFILE_LUSH: GrassProfile = GrassProfile::wispy_tall();

/// The grass *type* grown on a base's grassy ground (its [`StructureKind::BaseSite`]
/// lawn) — short, thick, manicured. Overrides the moisture-blended meadow type
/// inside the lawn footprint; retune the spaceport-lawn look here.
const GRASS_PROFILE_LAWN: GrassProfile = GrassProfile::lawn();

/// Placement-density multiplier for a lawn over its ring's wild density. A lawn
/// force-accepts every candidate (no gate thinning), so a denser grid is what
/// closes the patchy gaps; scoped to lawn tiles, so wild grass is unaffected.
const GRASS_LAWN_DENSITY_MULT: f32 = 1.6;

/// Outward margin added to a paved/built structure's footprint before clearing
/// grass (metres), so blades stop a touch short of a hard edge instead of
/// fringing right against the paving.
const STRUCTURE_CLEAR_MARGIN_M: f64 = 2.0;
/// Blend ramp on a structure's clearing footprint (metres) — short, so the lawn
/// runs almost to the structure edge.
const STRUCTURE_CLEAR_RAMP_M: f64 = 3.0;

/// Map one terrain-anchored structure to its building-terrain scatter footprint:
/// a [`StructureKind::BaseSite`] becomes a grass [`ScatterTreatment::Lawn`] over
/// its levelled area; a runway, building, launchpad, or tank becomes a
/// [`ScatterTreatment::Clear`] over its paved/built footprint (plus a small
/// margin). Returns `None` for a kind with no surface footprint.
fn site_scatter_region(site: &StructureSite, body_radius_m: f64) -> Option<ScatterRegion> {
    let across = site.anchor_dir.cross(site.heading_tangent).normalize();
    // `elevation_m` is irrelevant to `weight()` — only the rectangle matters.
    let rect = |half_along: f64, half_across: f64, ramp: f64| {
        TerrainFlatten::new(
            site.anchor_dir,
            site.heading_tangent,
            across,
            half_along,
            half_across,
            ramp,
            0.0,
            body_radius_m,
        )
    };
    let clear = |half_along: f64, half_across: f64| ScatterRegion {
        footprint: rect(
            half_along + STRUCTURE_CLEAR_MARGIN_M,
            half_across + STRUCTURE_CLEAR_MARGIN_M,
            STRUCTURE_CLEAR_RAMP_M,
        ),
        treatment: ScatterTreatment::Clear,
    };
    Some(match site.kind {
        // The base's flattened ground is the lawn (grass grows on it); its
        // footprint is the flatten rectangle the basin levelled.
        StructureKind::BaseSite => {
            let StructurePlacement::FlattenTo {
                half_along_m,
                half_across_m,
                ramp_m,
                rect_offset_along_m,
                rect_offset_across_m,
                ..
            } = site.placement
            else {
                return None;
            };
            ScatterRegion {
                // The lawn covers the levelled rectangle, offsets included
                // (the basin rect is pushed off its anchor toward the
                // secondary runway).
                footprint: rect(half_along_m, half_across_m, ramp_m)
                    .with_rect_offset(rect_offset_along_m, rect_offset_across_m),
                treatment: ScatterTreatment::Lawn,
            }
        }
        StructureKind::Runway {
            half_length_m,
            half_width_m,
        } => clear(half_length_m as f64, half_width_m as f64),
        StructureKind::Building {
            half_x_m, half_z_m, ..
        } => clear(half_x_m as f64, half_z_m as f64),
        // A round pad/tank clears its bounding square — a slightly generous
        // clearing under a disc, which reads fine against grass.
        StructureKind::Launchpad { radius_m } => clear(radius_m as f64, radius_m as f64),
        StructureKind::Tank { radius_m, .. } => clear(radius_m as f64, radius_m as f64),
    })
}

/// The building-terrain scatter regions for one body — a base's ground reads
/// as a lawn and its paved/built footprints clear the blades. Shared by the
/// CPU tile builds and the GPU grass field's control window
/// (`rendering::gpu_grass`), so both layers honour the same footprints.
pub(crate) fn grass_scatter_regions(
    structures: &StructureRegistry,
    body_id: BodyId,
    radius_m: f64,
) -> Vec<ScatterRegion> {
    structures
        .sites_on(body_id)
        .iter()
        .filter_map(|site| site_scatter_region(site, radius_m))
        .collect()
}

/// Fade band half-width for a ring's near/far cross-fade.
fn ring_band_m(ring: &GrassRing) -> f32 {
    (((ring.outer_m - ring.inner_m) as f32) * 0.12).clamp(3.0, 70.0)
}

/// Near/far/band fade parameters for a ring (packed into `GrassParams.time_fade`
/// yzw). The innermost ring uses a large-negative near edge so it never fades
/// in. The build/existence range extends one band beyond each edge so adjacent
/// rings overlap and cross-fade.
fn ring_fade(idx: usize) -> (f32, f32, f32) {
    let r = &GRASS_RINGS[idx];
    let band = ring_band_m(r);
    let near = if idx == 0 { -1.0e6 } else { r.inner_m as f32 };
    (near, r.outer_m as f32, band)
}

// ── Tuning ───────────────────────────────────────────────────────────────────
/// Grass blades are fully present below this AGL. Above it they collapse toward
/// the ground (a smooth height fade in `grass.wgsl` driven by `sky_up.w`): from
/// a plane the blades subtend ~no pixels and the terrain albedo already carries
/// the grass colour, so the whole blade layer is pure cost up there. This is the
/// "don't render full LOD at altitude, prioritize visible masses" fix.
const GRASS_FADE_LO_AGL_M: f64 = 150.0;
/// Grass blades fully collapsed (invisible) at/above this AGL.
const GRASS_FADE_HI_AGL_M: f64 = 300.0;
/// Above this altitude no new tiles are built — a little into the fade, so we
/// don't pay to build near-collapsed blades. (Below it, building resumes on
/// descent in time to populate before the blades are full at `FADE_LO`.)
const GRASS_MAX_AGL_M: f64 = 185.0;
/// Above this altitude all live tiles are despawned. Past the collapse, so the
/// despawn is invisible — blades are already 0-height by `GRASS_FADE_HI_AGL_M`.
const GRASS_DESPAWN_AGL_M: f64 = 340.0;
/// Maximum concurrent tile builds across all rings. Builds only dispatch for
/// tiles whose terrain is resident at a fine LOD (cheap GPU-mirror samples).
const GRASS_MAX_IN_FLIGHT: usize = 8;
/// Don't build grass until the terrain under a tile is resident at this texel
/// size or finer (the floating-carpet gate).
const GRASS_MAX_TERRAIN_TEXEL_M: f32 = 8.0;
/// World seed for blade placement hashes.
const GRASS_SEED: u64 = 0x6772_6173_7321;
/// Wind sway amplitude at the blade tip, metres.
const GRASS_WIND_SWAY_M: f32 = 0.06;
/// Rebuild-staleness scan interval, seconds.
const GRASS_REBUILD_CHECK_S: f32 = 0.5;
/// Rebuild a stale tile only when its centre height moved more than this.
///
/// Must stay **above the height-sample noise floor**, or rotation-driven atlas
/// re-streaming triggers a constant despawn→rebuild churn that pops tiles in and
/// out (the despawn→async-rebuild gap is far more visible than the height change
/// it chases). The GPU height mirror stores R16-quantized heights, so re-serving
/// the same point from a different LOD/atlas slot shifts it by ~1 quantization
/// step (~0.05–0.10 m on Thalos, measured). Terrain is LOD-invariant by design
/// (see `docs/terrain.md`), so the only *real* change grass must chase is a
/// flatten-pad install (the runway — metres); 0.5 m clears the noise with margin
/// and still catches pads.
const GRASS_REBUILD_DELTA_M: f32 = 0.5;
/// Stale-tile rebuilds dispatched per scan tick.
const GRASS_MAX_REBUILDS_PER_TICK: usize = 2;

/// A tile key tagged with its clipmap ring (the same `(face,x,y)` means
/// different tiles at different ring resolutions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RingTileKey {
    ring: u8,
    key: GrassTileKey,
}

/// One finished tile. `entity: None` means the tile built empty (water, rock,
/// alpine, flattened pad) — recorded so it isn't rebuilt every frame.
struct BuiltTile {
    entity: Option<Entity>,
    built_revision: u64,
    center_height_m: f32,
}

/// Driver state. **Sole writer:** the systems in this module (drive → finalize
/// → rebuild-check run sequentially via their `ResMut` access).
#[derive(Resource, Default)]
struct GrassTiles {
    body: Option<BodyId>,
    /// `tiles_per_side` for each clipmap ring on the current body.
    ring_tiles_per_side: Vec<i64>,
    tiles: HashMap<RingTileKey, BuiltTile>,
    /// In-flight builds, with the source revision snapshotted at dispatch.
    in_flight: HashMap<RingTileKey, (Task<Option<GrassTileMesh>>, u64)>,
    /// One material per ring (carries that ring's fade parameters).
    materials: Vec<Handle<GrassMaterial>>,
    /// Player AGL over the local terrain (m), written by `drive_grass_tiles` and
    /// read by `update_grass_material` for the altitude-collapse fade.
    agl_m: f64,
    rebuild_timer: f32,
    /// Per-second churn counters (grass-flicker investigation; logged + reset
    /// by `log_grass_diagnostics`). Remove once the flicker cause is pinned.
    dbg: GrassDiag,
}

/// Diagnostic event counters accumulated over one second.
#[derive(Default)]
struct GrassDiag {
    reach_despawns: u32,
    rebuild_despawns: u32,
    dispatched: u32,
    empty: u32,
    /// Largest |Δheight| that triggered a rebuild this second — tests whether
    /// the sampled terrain height under a fixed tile actually wobbles.
    max_rebuild_dh_m: f32,
    log_timer: f32,
    last_revision: u64,
}

/// Marker on a spawned grass-tile entity.
#[derive(Component)]
struct GrassTileVisual {
    body_id: BodyId,
    /// Body-fixed position of the tile centre on the surface.
    center_surface_body: DVec3,
}

pub struct GrassRenderPlugin;

impl Plugin for GrassRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GrassTiles>().add_systems(
            Update,
            (
                check_grass_rebuilds,
                drive_grass_tiles.after(check_grass_rebuilds),
                finalize_grass_tiles.after(drive_grass_tiles),
                update_grass_transforms.after(finalize_grass_tiles),
                update_grass_material,
                log_grass_diagnostics.after(update_grass_transforms),
            )
                .in_set(SimStage::Sync)
                .after(sync_solar_system_state),
        );
    }
}

/// Pick the active grass body and keep the clipmap tile set around the **view
/// anchor** (the render camera, resolved body-fixed at a coherent epoch — see
/// [`crate::rendering::view_anchor`]): dispatch builds for missing tiles
/// (nearest first across all rings), despawn tiles beyond each ring's reach.
#[allow(clippy::too_many_arguments)]
fn drive_grass_tiles(
    mut grass: ResMut<GrassTiles>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    gpu_height_mirrors: Res<crate::terrain_registry::GpuHeightMirrorRegistry>,
    structures: Res<StructureRegistry>,
    graphics: Res<GraphicsSettings>,
    anchor: Res<ViewAnchor>,
    mut commands: Commands,
) {
    if solar.states.is_none() {
        return;
    }

    let despawn_all = |grass: &mut GrassTiles, commands: &mut Commands| {
        for (_, tile) in grass.tiles.drain() {
            if let Some(entity) = tile.entity {
                commands.entity(entity).despawn();
            }
        }
        grass.in_flight.clear();
    };

    // Grass disabled in graphics settings: park the clipmap — despawn any live
    // tiles and build nothing (the terrain albedo still carries the grass colour,
    // so the ground reads green). Mirrors the clouds toggle's parked path.
    if !graphics.grass {
        if grass.body.is_some() {
            despawn_all(&mut grass, &mut commands);
            grass.body = None;
        }
        return;
    }

    // Active body: the view anchor's (nearest terrain-backed) body, when it can
    // grow grass. Grass exists around the VIEW — see `rendering::view_anchor`.
    let anchored = anchor.resolved.filter(|a| {
        sim.system
            .bodies
            .get(a.body)
            .is_some_and(|b| terrain_shading_style_for(b) == TerrainShadingStyle::Vegetated)
    });
    let Some(view) = anchored else {
        if grass.body.is_some() {
            despawn_all(&mut grass, &mut commands);
            grass.body = None;
        }
        return;
    };
    let body_id = view.body;
    if grass.body != Some(body_id) {
        despawn_all(&mut grass, &mut commands);
        grass.body = Some(body_id);
        let radius_m = sim.system.bodies[body_id].radius_m;
        grass.ring_tiles_per_side = GRASS_RINGS
            .iter()
            .map(|r| grass_tiles_per_side(radius_m, r.tile_size_m))
            .collect();
    }

    let radius_m = view.radius_m;
    let Some(height_source) = height_sources.get(body_id) else {
        return;
    };
    let mirror = gpu_height_mirrors.get(body_id);

    let cam_dir = view.cam_dir;
    let agl = view.agl_m;
    grass.agl_m = agl;
    if agl > GRASS_DESPAWN_AGL_M {
        if !grass.tiles.is_empty() || !grass.in_flight.is_empty() {
            despawn_all(&mut grass, &mut commands);
        }
        return;
    }

    let ring_tps = grass.ring_tiles_per_side.clone();
    if ring_tps.len() != GRASS_RINGS.len() {
        return;
    }
    let arc_dist = |center_dir: DVec3| -> f64 { center_dir.angle_between(cam_dir) * radius_m };

    // With the GPU grass field active (`rendering::gpu_grass`), the WHOLE CPU
    // clipmap parks — the field's five bands now cover the full ~340 m reach
    // (including the former card ring), with zero persistent geometry. Toggling
    // the setting off restores the full CPU clipmap as a fallback.
    let min_ring = if graphics.gpu_grass {
        GRASS_RINGS.len() as u8
    } else {
        0u8
    };

    // Despawn tiles past their ring's reach (outer edge + a fade band of
    // slack), and any tile in a ring the GPU field has taken over.
    let stale: Vec<RingTileKey> = grass
        .tiles
        .keys()
        .filter(|rk| {
            if rk.ring < min_ring {
                return true;
            }
            let ring = &GRASS_RINGS[rk.ring as usize];
            let reach = ring.outer_m + ring_band_m(ring) as f64 + ring.tile_size_m;
            grass_tile_frame(rk.key, ring_tps[rk.ring as usize])
                .is_none_or(|(center, _)| arc_dist(center) > reach)
        })
        .copied()
        .collect();
    for rk in stale {
        if let Some(tile) = grass.tiles.remove(&rk)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
            grass.dbg.reach_despawns += 1;
        }
    }

    // No new builds while high above the grass shell (existing tiles persist).
    if agl > GRASS_MAX_AGL_M {
        return;
    }

    let slots = GRASS_MAX_IN_FLIGHT.saturating_sub(grass.in_flight.len());
    if slots == 0 {
        return;
    }

    // Gather missing-tile candidates across every ring, nearest first. Each ring
    // overlaps its neighbour at the near edge (one fade band) so they cross-fade,
    // and extends a full tile *beyond* its outer (fade) edge so the outermost
    // tiles build while their blades are scaled to ~0 — the build is invisible
    // (no pop-in), and they grow in as the craft approaches.
    let mut candidates: Vec<(f64, RingTileKey)> = Vec::new();
    for (ring_idx, ring) in GRASS_RINGS.iter().enumerate().skip(min_ring as usize) {
        let tps = ring_tps[ring_idx];
        let band = ring_band_m(ring) as f64;
        let lo = (ring.inner_m - band).max(0.0);
        let hi = ring.outer_m + ring.tile_size_m;
        let center_key = grass_tile_key(cam_dir, tps);
        let window = (hi / (ring.tile_size_m * 0.5)).ceil() as i64;
        for dy in -window..=window {
            for dx in -window..=window {
                let key = GrassTileKey {
                    face: center_key.face,
                    x: center_key.x + dx,
                    y: center_key.y + dy,
                };
                let rk = RingTileKey {
                    ring: ring_idx as u8,
                    key,
                };
                if grass.tiles.contains_key(&rk) || grass.in_flight.contains_key(&rk) {
                    continue;
                }
                let Some((center, _)) = grass_tile_frame(key, tps) else {
                    continue;
                };
                let d = arc_dist(center);
                if d >= lo && d <= hi {
                    candidates.push((d, rk));
                }
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Sea level is the project datum: the constant 0 m (= reference radius), the
    // shoreline the bimodal continent/ocean generator (Slice 1) puts at height 0.
    // Grass requires `height > sea_level + 1 m`, so the seabed stays bare.
    let sea_level_m = 0.0;

    // Building-terrain scatter regions: the base's grassy ground reads as a
    // managed lawn (thick short grass) and its paved/built footprints (runway,
    // launchpads, buildings, tanks) clear the blades. Derived each frame from
    // the structure registry, so authored and player-placed bases both apply;
    // empty off-base, so wild terrain is untouched. Shared (`Arc`) into every
    // tile build dispatched this frame.
    let scatter_regions: Arc<Vec<ScatterRegion>> =
        Arc::new(grass_scatter_regions(&structures, body_id, radius_m));

    let mirror_guard = mirror.as_ref().and_then(|m| m.read().ok());
    let pool = AsyncComputeTaskPool::get();
    let mut dispatched = 0usize;
    for (_, rk) in candidates {
        if dispatched >= slots {
            break;
        }
        let ring = &GRASS_RINGS[rk.ring as usize];
        let tps = ring_tps[rk.ring as usize];
        if let Some(guard) = &mirror_guard {
            let Some((center, _)) = grass_tile_frame(rk.key, tps) else {
                continue;
            };
            // Far rings tolerate coarser terrain — their clump blades are huge,
            // so the residency threshold scales with tile size. Without this the
            // distant rings (over coarse far terrain) never pass the gate and
            // grass never reaches the horizon.
            let texel_limit = ((ring.tile_size_m * 0.5) as f32).max(GRASS_MAX_TERRAIN_TEXEL_M);
            match guard.best_resident_texel_m(center.as_vec3()) {
                Some(texel) if texel <= texel_limit => {}
                _ => continue, // terrain not detailed here yet — retry next frame
            }
        }
        let input = GrassTileBuildInput {
            key: rk.key,
            tiles_per_side: tps,
            height_source: Arc::clone(&height_source),
            radius_m,
            sea_level_m,
            blades_per_m2: ring.density_per_m2,
            profile_dry: GRASS_PROFILE_DRY,
            profile_lush: GRASS_PROFILE_LUSH,
            blade_lod: ring.blade_lod,
            width_scale: ring.width_scale,
            height_scale: ring.height_scale,
            clump_scale: ring.clump_scale,
            seed: GRASS_SEED,
            scatter_regions: Arc::clone(&scatter_regions),
            lawn_profile: GRASS_PROFILE_LAWN,
            lawn_density_per_m2: ring.density_per_m2 * GRASS_LAWN_DENSITY_MULT,
            forest_cull: ring.forest_cull,
        };
        let revision = height_source.revision();
        let task = pool.spawn(async move { build_grass_tile_mesh(&input) });
        grass.in_flight.insert(rk, (task, revision));
        dispatched += 1;
        grass.dbg.dispatched += 1;
    }
}

/// Poll in-flight builds; spawn finished tiles as root-grid big_space children
/// (the runway visual pattern), each on its ring's material.
#[allow(clippy::too_many_arguments)]
fn finalize_grass_tiles(
    mut grass: ResMut<GrassTiles>,
    solar: Res<SolarSystemState>,
    root: Option<Res<RealSpaceRoot>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GrassMaterial>>,
    mut images: ResMut<Assets<Image>>,
    sun_shadow: Option<Res<SunShadowState>>,
    mut commands: Commands,
) {
    if grass.in_flight.is_empty() {
        return;
    }
    let (Some(states), Some(root), Some(body_id)) = (solar.states.as_deref(), root, grass.body)
    else {
        return;
    };
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    ensure_ring_materials(
        &mut grass,
        &mut materials,
        &mut images,
        sun_shadow.as_deref(),
    );
    let ring_materials = grass.materials.clone();

    let mut finished: Vec<(RingTileKey, u64, Option<GrassTileMesh>)> = Vec::new();
    grass
        .in_flight
        .retain(|rk, (task, revision)| match block_on(poll_once(task)) {
            Some(result) => {
                finished.push((*rk, *revision, result));
                false
            }
            None => true,
        });

    let orientation = body_state.orientation.normalize();
    for (rk, revision, result) in finished {
        let Some(tile) = result else {
            grass.tiles.insert(
                rk,
                BuiltTile {
                    entity: None,
                    built_revision: revision,
                    center_height_m: 0.0,
                },
            );
            grass.dbg.empty += 1;
            continue;
        };

        let center_world = body_state.position + orientation * tile.center_surface_body_m;
        let (cell, local) = real_space_grid().translation_to_grid(center_world);
        let material = ring_materials[rk.ring as usize].clone();
        // Explicit local-space AABB so the tile is **frustum-culled**. The mesh is
        // `RENDER_WORLD`-only, so Bevy never auto-computes an `Aabb` for it (see
        // `docs/vegetation.md`) — without this, every tile in the full 360° ring
        // around the camera runs its (now per-vertex-lit, so heavier) vertex
        // shader every frame, including the ~⅔ behind/beside the view. Rest-pose
        // bound; the shader's ≤6 cm wind sway is far inside the frustum margin.
        let aabb = tile.mesh.compute_aabb();
        let mut tile_cmd = commands.spawn((
            Mesh3d(meshes.add(tile.mesh)),
            MeshMaterial3d(material),
            Transform {
                translation: local,
                rotation: orientation.as_quat(),
                scale: Vec3::ONE,
            },
            cell,
            Visibility::Inherited,
            RenderLayers::layer(SHIP_LAYER),
            NotShadowCaster,
            ChildOf(root.entity),
            GrassTileVisual {
                body_id,
                center_surface_body: tile.center_surface_body_m,
            },
            Name::new("Grass Tile"),
        ));
        if let Some(aabb) = aabb {
            tile_cmd.insert(aabb);
        }
        let entity = tile_cmd.id();
        grass.tiles.insert(
            rk,
            BuiltTile {
                entity: Some(entity),
                built_revision: tile.built_revision,
                center_height_m: tile.center_height_m,
            },
        );
    }
}

/// Lazily create one material per clipmap ring, each seeded with its fade band.
/// Each ring receives the cascaded sun-shadows trees cast — the same maps the
/// ground + tree materials bind. The depth bindings need a valid texture at
/// creation, so seed them with the live cascade maps (or a 1×1 fallback before
/// the rig exists); `update_grass_material` rebinds the live block each frame.
fn ensure_ring_materials(
    grass: &mut GrassTiles,
    materials: &mut Assets<GrassMaterial>,
    images: &mut Assets<Image>,
    sun_shadow: Option<&SunShadowState>,
) {
    if grass.materials.len() == GRASS_RINGS.len() {
        return;
    }
    let maps: [Handle<Image>; 3] = match sun_shadow {
        Some(s) => s.images.clone(),
        None => {
            let fb = images.add(fallback_shadow_map());
            [fb.clone(), fb.clone(), fb]
        }
    };
    // The baked clump-card atlas the far ring's CARD quads sample; one image
    // shared across the ring materials.
    let card_atlas = images.add(build_grass_card_atlas());
    grass.materials = (0..GRASS_RINGS.len())
        .map(|idx| {
            let (near, far, band) = ring_fade(idx);
            materials.add(GrassMaterial {
                params: thalos_body_render::GrassParams {
                    time_fade: Vec4::new(0.0, near, far, band),
                    ..default()
                },
                sun_shadow_map_0: maps[0].clone(),
                sun_shadow_map_1: maps[1].clone(),
                sun_shadow_map_2: maps[2].clone(),
                card_atlas: card_atlas.clone(),
                ..default()
            })
        })
        .collect();
}

/// Re-anchor every grass tile in f64 each frame — verbatim runway math: the
/// multi-Mm body-fixed offset is rotated in f64 here, and the f32
/// `Transform.rotation` only acts on the small blade vertex offsets.
fn update_grass_transforms(
    solar: Res<SolarSystemState>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut tiles: Query<(&GrassTileVisual, &mut CellCoord, &mut Transform)>,
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

/// Per-frame shading parameters: sun direction + flux toward the star, the
/// radial up and Rayleigh τ_v feeding the shared `thalos::lighting` sky model
/// (so blades light exactly like the ground), wall-clock sway time, and a
/// slowly veering wind direction tangent to the surface under the camera. The
/// shared fields are written to every ring material; each keeps its own fade.
#[allow(clippy::too_many_arguments)]
fn update_grass_material(
    grass: Res<GrassTiles>,
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    time: Res<Time>,
    exposure: Res<CameraExposure>,
    anchor: Res<ViewAnchor>,
    sun_shadow: Option<Res<SunShadowState>>,
    mut materials: ResMut<Assets<GrassMaterial>>,
) {
    let (Some(body_id), Some(states)) = (grass.body, solar.states.as_deref()) else {
        return;
    };
    if grass.materials.is_empty() {
        return;
    }
    let Some(body_state) = states.get(body_id) else {
        return;
    };

    let star_pos = states.first().map(|s| s.position).unwrap_or(DVec3::ZERO);
    let offset = star_pos - body_state.position;
    let sun_dir = offset.normalize_or_zero().as_vec3();
    // Sun flux in the same units the terrain `SceneLighting` carries (lux ×
    // exposure gain), so the shared sky model exposes grass identically.
    let au_over_d = (AU_M / offset.length().max(1.0)) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;
    let sun = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, flux);

    // Wind: tangent to the surface under the camera, veering slowly. Render
    // space is inertial-axis-aligned, so the world-space tangent basis comes
    // straight from the view anchor's local vertical (the blades exist around
    // the view; the craft may be elsewhere entirely).
    let t = time.elapsed_secs();
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
    let veer = t * 0.03;
    let wind_dir = (east * veer.cos() + north * veer.sin()).normalize_or_zero();
    let wind = Vec4::new(wind_dir.x, wind_dir.y, wind_dir.z, GRASS_WIND_SWAY_M);
    // Altitude collapse (sky_up.w): 0 near the ground (full blades), ramping to 1
    // above `GRASS_FADE_HI_AGL_M` so the shader sinks the blade layer into the
    // ground from a plane (terrain albedo carries the grass colour up there).
    let ramp = (grass.agl_m - GRASS_FADE_LO_AGL_M) / (GRASS_FADE_HI_AGL_M - GRASS_FADE_LO_AGL_M);
    let ramp = ramp.clamp(0.0, 1.0);
    let altitude_collapse = (ramp * ramp * (3.0 - 2.0 * ramp)) as f32;
    let sky_up = Vec4::new(up.x, up.y, up.z, altitude_collapse);

    // Sky hemisphere inputs: the body's authored Rayleigh τ_v + strength (the
    // same value the terrain shader recovers from its `AtmosphereBlock`).
    let (tau, strength) = sim
        .system
        .bodies
        .get(body_id)
        .and_then(|b| b.terrestrial_atmosphere.as_ref())
        .and_then(|a| a.scattering.as_ref())
        .map(|s| (Vec3::from_array(s.vertical_optical_depth), s.strength))
        .unwrap_or((Vec3::ZERO, 0.0));
    let sky_tau = Vec4::new(tau.x, tau.y, tau.z, strength);

    // Fade reference = the VIEW (`view.world_position` in the shader, offset 0):
    // blade density is a per-instance LOD keyed by distance from the eye. Offset
    // 0 is inherently origin-invariant and this-frame-exact — the former craft-
    // anchored offset worked around the main-world camera transform lagging a
    // frame, which the shader's own view position doesn't (see `grass.wgsl`).
    let anchor = Vec4::ZERO;

    for (idx, handle) in grass.materials.iter().enumerate() {
        let Some(mut material) = materials.get_mut(handle) else {
            continue;
        };
        let (near, far, band) = ring_fade(idx);
        material.params.sun_dir = sun;
        material.params.wind = wind;
        material.params.time_fade = Vec4::new(t, near, far, band);
        material.params.sky_up = sky_up;
        material.params.sky_tau = sky_tau;
        material.params.anchor = anchor;
        // Bind the live sun-shadow cascade so blades take the trees' (and the
        // ground's) shadows. `gate.x` is 0 off-surface, so the shader skips the
        // per-vertex sample there. Grass lives within the near cascade's reach,
        // so only cascade 0 is sampled in practice.
        if let Some(sun_shadow) = sun_shadow.as_deref() {
            material.shadow = sun_shadow.block;
            material.sun_shadow_map_0 = sun_shadow.images[0].clone();
            material.sun_shadow_map_1 = sun_shadow.images[1].clone();
            material.sun_shadow_map_2 = sun_shadow.images[2].clone();
        }
    }
}

/// Periodically reconcile tiles with the height source: when the source
/// revision advances (a finer atlas tile streamed in, a flatten pad was
/// installed), re-sample each stale tile's centre height and rebuild it if the
/// ground actually moved. Rebuild = despawn + forget; `drive_grass_tiles`
/// re-dispatches it on a later pass.
fn check_grass_rebuilds(
    mut grass: ResMut<GrassTiles>,
    height_sources: Res<HeightSourceRegistry>,
    time: Res<Time>,
    mut commands: Commands,
) {
    grass.rebuild_timer += time.delta_secs();
    if grass.rebuild_timer < GRASS_REBUILD_CHECK_S {
        return;
    }
    grass.rebuild_timer = 0.0;

    let Some(body_id) = grass.body else {
        return;
    };
    let Some(source) = height_sources.get(body_id) else {
        return;
    };
    let revision = source.revision();
    if grass.ring_tiles_per_side.len() != GRASS_RINGS.len() {
        return;
    }
    let ring_tps = grass.ring_tiles_per_side.clone();

    let mut rebuilt = 0usize;
    let mut to_remove: Vec<RingTileKey> = Vec::new();
    let mut max_dh = 0.0f32;
    for (rk, tile) in grass.tiles.iter_mut() {
        if tile.built_revision == revision {
            continue;
        }
        let Some((center_dir, _)) = grass_tile_frame(rk.key, ring_tps[rk.ring as usize]) else {
            continue;
        };
        let Some(h) = source.sample_height_m(center_dir.as_vec3(), 0.5) else {
            continue;
        };
        if tile.entity.is_some()
            && (h - tile.center_height_m).abs() > GRASS_REBUILD_DELTA_M
            && rebuilt < GRASS_MAX_REBUILDS_PER_TICK
        {
            max_dh = max_dh.max((h - tile.center_height_m).abs());
            to_remove.push(*rk);
            rebuilt += 1;
        } else {
            tile.built_revision = revision;
            if tile.entity.is_some() {
                tile.center_height_m = h;
            }
        }
    }
    let removed = to_remove.len() as u32;
    for rk in to_remove {
        if let Some(tile) = grass.tiles.remove(&rk)
            && let Some(entity) = tile.entity
        {
            commands.entity(entity).despawn();
        }
    }
    grass.dbg.rebuild_despawns += removed;
    grass.dbg.max_rebuild_dh_m = grass.dbg.max_rebuild_dh_m.max(max_dh);
}

/// **Diagnostic only** (grass-flicker investigation): once per second, append a
/// JSON line of the tile-churn counters + the height-source revision delta to
/// the file named by `THALOS_GRASS_LOG` (falls back to an `info!` line if the
/// env var is unset). A non-zero `rev_delta` while parked means the terrain
/// atlas is re-streaming under rotation; `rebuild_despawns` / `reach_despawns`
/// say which path is popping tiles; `max_rebuild_dh_m` says whether the sampled
/// height actually wobbled. Remove once the cause is pinned.
fn log_grass_diagnostics(
    mut grass: ResMut<GrassTiles>,
    height_sources: Res<HeightSourceRegistry>,
    time: Res<Time>,
) {
    grass.dbg.log_timer += time.delta_secs();
    if grass.dbg.log_timer < 1.0 {
        return;
    }
    grass.dbg.log_timer = 0.0;

    let Some(body_id) = grass.body else {
        return;
    };
    let revision = height_sources
        .get(body_id)
        .map(|s| s.revision())
        .unwrap_or(0);
    let rev_delta = revision.wrapping_sub(grass.dbg.last_revision);
    grass.dbg.last_revision = revision;

    let line = format!(
        "{{\"t_s\":{:.1},\"tiles\":{},\"in_flight\":{},\"revision\":{},\"rev_delta\":{},\
\"reach_despawns\":{},\"rebuild_despawns\":{},\"dispatched\":{},\"empty\":{},\
\"max_rebuild_dh_m\":{:.3}}}",
        time.elapsed_secs(),
        grass.tiles.len(),
        grass.in_flight.len(),
        revision,
        rev_delta,
        grass.dbg.reach_despawns,
        grass.dbg.rebuild_despawns,
        grass.dbg.dispatched,
        grass.dbg.empty,
        grass.dbg.max_rebuild_dh_m,
    );
    // Always write to the diagnostics tree so machine-readable traces never
    // mix with curated screenshots. A bare override filename is rooted there;
    // explicit relative/absolute paths are honored.
    let path =
        crate::artifact_paths::jsonl_path_from_env_or("THALOS_GRASS_LOG", "grass_churn.jsonl");
    use std::io::Write;
    if let Ok(mut f) = crate::artifact_paths::open_jsonl_append(&path) {
        let _ = writeln!(f, "{line}");
    }

    grass.dbg.reach_despawns = 0;
    grass.dbg.rebuild_despawns = 0;
    grass.dbg.dispatched = 0;
    grass.dbg.empty = 0;
    grass.dbg.max_rebuild_dh_m = 0.0;
}
