//! Hold the loading screen until terrain near a surface spawn has settled.
//!
//! Surface scenarios (runway, descents, EVA) place the craft within
//! metres-to-kilometres of the ground. The UDLOD terrain there streams from
//! coarse to fine over ~1–2 s, and the runway pad's [`thalos_terrain::
//! TerrainFlatten`] only reaches the collider and the visible mesh once those
//! tiles bake. Without a gate the player's first frame shows the ground heave
//! up to the runway and the aircraft bounce as the colliders catch up — the
//! exact symptoms this module exists to hide.
//!
//! The fix is to do the streaming *behind* the loading screen:
//!
//! 1. The deferred runway placement ([`crate::runway::finish_runway_spawn`])
//!    runs during `AppState::Loading`, not after it, so the aircraft is parked,
//!    the flatten pad is installed, and the camera reaches the surface while the
//!    screen is still up. The sim runs as usual during loading, so the parked
//!    aircraft settles onto its gear and the tile streamer refines the ground
//!    there — all hidden. The placement calls [`SurfaceSettle::mark_placed`].
//! 2. [`update_surface_settle`] watches the resident tile resolution *directly
//!    under the view* ([`renderer_tile_lod_m_at`] at the tile tree's own view
//!    position); once it has refined and plateaued (stopped getting finer for a
//!    handful of frames), or a safety timeout elapses, it flips
//!    [`SurfaceSettle::done`] and completes the loading tracker's
//!    [`step::SETTLE`] step, which the loading screen's reveal waits on — so
//!    the first visible frame is already flush and stable.
//!
//! Only the **parked** `Runway` start is gated. Every other start (orbit,
//! descents, EVA, and the airborne `RunwayApproach`) is a no-op here. The
//! airborne approach is deliberately excluded: the sim runs during loading, so a
//! multi-second settle would fly it down its approach behind the screen — and a
//! moving view never plateaus anyway. It streams its terrain in flight like any
//! normal descent.

use bevy::prelude::*;
use thalos_body_render::renderer_tile_lod_m_at;
use thalos_body_render::udlod::prelude::{TerrainViewComponents, TileAtlas, TileTree};

use crate::camera::ShipCamera;
use crate::loading::{AppState, LoadingTracker, step};
use crate::rendering::ground_terrain::BodyTerrain;
use crate::solar_system_state::SimulationState;
use crate::spawn::SpawnSituation;

/// Consecutive frames the resident tile resolution at the spawn point must hold
/// at its finest (stop getting finer) before the site counts as settled. A
/// little over a second at the slow loading-frame rate — long enough that the
/// last, deepest tile has finished loading and the height mirror/atlas upload
/// have caught up, not so long the reveal drags.
const SETTLE_STABLE_FRAMES: u32 = 12;

/// Safety ceiling on the settle wait, measured from the moment the surface
/// state is placed. Cold streaming of Thalos's expensive terrain to a usable
/// fineness takes ~15–20 s; this is the backstop if it stalls, so the screen
/// reveals rather than hangs.
const MAX_SETTLE_S: f64 = 30.0;
/// Resolution (metres per texel) the ground under the view must reach before a
/// plateau counts as settled, so a brief stall on the coarse pinned root doesn't
/// read as "done". The runway pad is flattened, so this need not be ultra-fine —
/// it just has to resolve the pad as flat ground rather than a single coarse
/// texel.
const SETTLE_TARGET_LOD_M: f32 = 50.0;

/// Hard ceiling measured from the first loading frame, regardless of whether
/// the deferred placement ever reported in. Guards against a stalled placement
/// hanging the loading screen forever; the bake-complete / residency gates in
/// [`crate::loading`] still apply, so this only ever *drops* the extra
/// tile-settle wait, never reveals before the world exists.
const HARD_TIMEOUT_S: f64 = 45.0;

/// State for the near-surface tile-settle gate. See the module docs.
///
/// Inserted at startup by [`init_surface_settle`] from the active
/// [`SpawnSituation`]; re-armed at runtime via [`SurfaceSettle::arm`] when the
/// start screen launches a runway scenario. The reveal is gated through the
/// loading tracker's [`step::SETTLE`] step, which this module completes.
#[derive(Resource, Debug)]
pub struct SurfaceSettle {
    /// This scenario spawns parked at the surface and needs the gate at all.
    needs_settle: bool,
    /// The surface state has been installed. Trivially `true` for scenarios
    /// whose placement is seeded directly in `main.rs` (orbit, EVA); set by the
    /// deferred placements for the runway/descent scenarios.
    placed: bool,
    /// Tile streaming at the site has settled (or the timeout fired).
    done: bool,
    /// Finest resident tile resolution seen at the spawn point so far (metres
    /// per texel; `INFINITY` until the first resident sample). Monotonically
    /// decreases as the streamer refines the ground there.
    best_lod_m: f32,
    /// Consecutive frames `best_lod_m` has held without getting finer.
    stable_frames: u32,
    /// Time since placement (the post-placement settle budget).
    elapsed_s: f64,
    /// Time since the first loading frame (the hard-timeout backstop).
    total_elapsed_s: f64,
}

impl SurfaceSettle {
    /// (Re-)arm the gate for a fresh loading pass. `placed = false` when a
    /// deferred placement will report in via [`mark_placed`](Self::mark_placed);
    /// `true` for scenarios whose surface state is already installed. Used at
    /// startup ([`init_surface_settle`]) and by the start screen's runway
    /// scenario starter ([`crate::main_menu`]).
    pub fn arm(&mut self, needs_settle: bool, placed: bool) {
        *self = SurfaceSettle {
            needs_settle,
            placed,
            done: false,
            best_lod_m: f32::INFINITY,
            stable_frames: 0,
            elapsed_s: 0.0,
            total_elapsed_s: 0.0,
        };
    }

    /// Called by the deferred runway placement once the aircraft is parked, the
    /// flatten pad is installed, and the site is known.
    pub fn mark_placed(&mut self) {
        self.placed = true;
    }
}

pub struct SurfaceSettlePlugin;

impl Plugin for SurfaceSettlePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_surface_settle).add_systems(
            Update,
            update_surface_settle.run_if(in_state(AppState::Loading)),
        );
    }
}

fn init_surface_settle(situation: Res<SpawnSituation>, mut commands: Commands) {
    let mut settle = SurfaceSettle {
        needs_settle: false,
        placed: false,
        done: false,
        best_lod_m: f32::INFINITY,
        stable_frames: 0,
        elapsed_s: 0.0,
        total_elapsed_s: 0.0,
    };
    settle.arm(
        // Only the parked `Runway` start is gated (see module docs): it installs
        // a flatten pad and sits still at the surface, so the ground there can
        // settle to a fixed flush state behind the screen.
        matches!(*situation, SpawnSituation::Runway),
        // Scenarios without a deferred placement install their surface state in
        // `main.rs`, so they're "placed" from frame 0; the settle then just
        // waits on the tiles.
        !situation.has_deferred_placement(),
    );
    commands.insert_resource(settle);
}

/// Drive the settle state machine each loading frame. Waits for the deferred
/// placement, then for the resident tile resolution *at the spawn point* to
/// refine and plateau (the local ground is settled), or for a safety timeout.
///
/// Gating on the whole atlas draining its load queue is too strict at planet
/// scale: the tile streamer keeps a steady backlog of distant tiles for minutes
/// and never goes fully quiet. The relevant question is narrower — has the
/// ground *under the craft* reached its final LOD? — so this watches
/// [`renderer_tile_lod_m_at`] at the craft's body-fixed surface point and waits
/// for it to stop getting finer.
fn update_surface_settle(
    time: Res<Time<Real>>,
    sim: Res<SimulationState>,
    mut settle: ResMut<SurfaceSettle>,
    mut tracker: ResMut<LoadingTracker>,
    tile_trees: Res<TerrainViewComponents<TileTree>>,
    terrains: Query<(Entity, &BodyTerrain, &TileAtlas)>,
    camera_q: Query<Entity, With<ShipCamera>>,
) {
    if settle.done || !settle.needs_settle {
        return;
    }
    let dt = time.delta_secs_f64();
    settle.total_elapsed_s += dt;
    // Backstop: never hang the loading screen if the deferred placement stalls.
    if settle.total_elapsed_s >= HARD_TIMEOUT_S {
        settle.done = true;
        tracker.complete(step::SETTLE);
        warn!(
            "surface settle hard-timeout at {:.1} s (placed={}) — revealing anyway",
            settle.total_elapsed_s, settle.placed
        );
        return;
    }
    // Wait for the deferred surface placement to install the site + move the
    // camera there before the streaming clock starts.
    if !settle.placed {
        return;
    }
    settle.elapsed_s += dt;

    let lod_m = resident_lod_under_view(&sim, &tile_trees, &terrains, &camera_q);

    // Settle diagnostics: a line every few seconds while the gate is active
    // (bounded — at most ~6 per load). The view radius tells whether the tile
    // streamer's view actually reached the surface site; a stall at a coarse
    // LOD with an orbital view radius means the camera never got there.
    let log_bucket = (settle.elapsed_s / 5.0) as u32;
    let prev_bucket = ((settle.elapsed_s - dt).max(0.0) / 5.0) as u32;
    if log_bucket != prev_bucket {
        let view_r_km = view_radius_km(&sim, &tile_trees, &terrains, &camera_q);
        info!(
            "surface settle: t={:.0}s lod={:?} m/texel (best {:.1}), view radius {:.1} km",
            settle.elapsed_s, lod_m, settle.best_lod_m, view_r_km
        );
    }

    match lod_m {
        // Got meaningfully finer this frame: streaming is still refining the
        // ground here. Record the new best and reset the plateau counter.
        Some(m) if m < settle.best_lod_m * 0.999 => {
            settle.best_lod_m = m;
            settle.stable_frames = 0;
        }
        // Held at (or near) the finest resolution seen: count toward settled.
        Some(_) => {
            settle.stable_frames += 1;
        }
        // No resident tile at the point yet (terrain entity not up, or the
        // camera hasn't reached the site): keep waiting.
        None => {
            settle.stable_frames = 0;
        }
    }

    // Publish a progress estimate to the loading screen: how close the
    // resident resolution is to the target (log-ish via the ratio), plus
    // the plateau countdown once it's there.
    if settle.best_lod_m.is_finite() {
        let refine = (SETTLE_TARGET_LOD_M / settle.best_lod_m).clamp(0.0, 1.0);
        let hold = settle.stable_frames as f32 / SETTLE_STABLE_FRAMES as f32;
        tracker.set_fraction(step::SETTLE, 0.7 * refine + 0.3 * hold.min(1.0));
        tracker.set_detail(step::SETTLE, format!("{:.0} m/texel", settle.best_lod_m));
    }

    let plateaued =
        settle.best_lod_m <= SETTLE_TARGET_LOD_M && settle.stable_frames >= SETTLE_STABLE_FRAMES;
    if plateaued || settle.elapsed_s >= MAX_SETTLE_S {
        settle.done = true;
        tracker.complete(step::SETTLE);
        info!(
            "surface terrain settled (lod {:.1} m/texel, {} stable frames, {:.1} s) — revealing",
            settle.best_lod_m, settle.stable_frames, settle.elapsed_s
        );
    }
}

/// Resident tile resolution (metres per texel) on the dominant body directly
/// under the camera, or `None` if no tile is resident there yet. Queries at the
/// tile tree's own [`TileTree::view_position`] — the exact body-fixed point the
/// streamer is refining around — so this is precisely "how settled is the
/// ground under the view". (Reading the canonical craft state instead is wrong:
/// the runway/descent placements install the craft under `OnRails`, so its
/// canonical translation still reads the placeholder orbit, not the surface.)
fn resident_lod_under_view(
    sim: &SimulationState,
    tile_trees: &TerrainViewComponents<TileTree>,
    terrains: &Query<(Entity, &BodyTerrain, &TileAtlas)>,
    camera_q: &Query<Entity, With<ShipCamera>>,
) -> Option<f32> {
    let body = sim.simulation.dominant_body();
    let (terrain_entity, _, atlas) = terrains.iter().find(|(_, t, _)| t.body_id == body)?;
    let camera = camera_q.iter().next()?;
    let tree = tile_trees.get(&(terrain_entity, camera))?;
    renderer_tile_lod_m_at(atlas, tree, tree.view_position())
}

/// Distance (km) from the body centre to the tile tree's view position — the
/// settle diagnostics' "did the streamer's view reach the surface" signal.
/// `NAN` while the tree isn't up.
fn view_radius_km(
    sim: &SimulationState,
    tile_trees: &TerrainViewComponents<TileTree>,
    terrains: &Query<(Entity, &BodyTerrain, &TileAtlas)>,
    camera_q: &Query<Entity, With<ShipCamera>>,
) -> f64 {
    let body = sim.simulation.dominant_body();
    let Some((terrain_entity, _, _)) = terrains.iter().find(|(_, t, _)| t.body_id == body) else {
        return f64::NAN;
    };
    let Some(camera) = camera_q.iter().next() else {
        return f64::NAN;
    };
    let Some(tree) = tile_trees.get(&(terrain_entity, camera)) else {
        return f64::NAN;
    };
    tree.view_position().length() / 1000.0
}
