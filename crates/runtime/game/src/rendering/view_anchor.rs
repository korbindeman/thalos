//! The **view anchor** — the one per-frame answer to "where is the view?" for
//! every view-dependent world-detail system.
//!
//! Thalos is one physical world that must render consistently from any camera
//! (see `docs/roadmap/graphics_fidelity.md` §2.3, the one-world principle). The UDLOD
//! terrain already streams around its camera view; this module extends the same
//! rule to everything else that builds or centres detail around a point:
//! surface scatter (trees / grass / rocks), the sun-shadow cascade centre, and
//! any future clipmap-like layer. They all read [`ViewAnchor`] — the **render
//! camera's** position resolved against the nearest terrain-backed body — and
//! none of them may anchor to the player craft, a mode flag, or a per-mode
//! focus override. The craft is just an object *in* the world; the view is what
//! decides where detail exists.
//!
//! Replacing the previous per-driver fallback chain (`scatter_view_center`:
//! god-view focus override → god-view / freecam flag → canonical craft state)
//! this makes new camera modes correct *by construction*: reposition the
//! [`ShipCamera`] however you like — flight orbit, freecam, god-view hub, base
//! editor, headless screenshot rig — and every detail system follows, with no
//! mode-specific plumbing. (The space-center hub is the cautionary tale: its
//! placeholder craft parks in orbit, so every craft-anchored system silently
//! built its detail 200 km above the base the camera was looking at.)
//!
//! ## Frame coherence (why the anchor is stored body-fixed)
//!
//! The camera's big_space pose (`CellCoord` + `Transform`) is written by the
//! camera drivers in `SimStage::Camera` (or plain `Update` for the god views),
//! *after* the detail drivers in `SimStage::Sync` have run — so a driver
//! reading the camera directly sees a **one-frame-stale heliocentric pose**.
//! At 1× a surface point co-rotates at hundreds of m/s, and under warp far
//! faster: a stale heliocentric point re-interpreted at this frame's epoch is
//! kilometres off in body-fixed terms, which made craft-anchored systems (the
//! shadow cascade, the grass fade) crawl or pop whenever they tried to read
//! the camera instead of the canonical craft. The fix is to resolve the pose
//! into the **body-fixed frame once, at a matching epoch**: the sole writer
//! runs at the top of `SimStage::Sync`, *before* [`sync_solar_system_state`]
//! refreshes [`SolarSystemState`] — so the (previous-frame) camera pose is
//! paired with the (previous-frame) body states it was posed against. In the
//! body frame the one-frame lag shrinks from "co-rotation × warp" to the
//! camera's own body-relative motion — metres. Consumers re-project the
//! body-fixed anchor with *their* frame's body state ([`AnchorBody::cam_world`]
//! / [`AnchorBody::ground_world`]) and get a this-epoch point that co-rotates
//! correctly.
//!
//! The anchor point is the camera itself (detail is built around the eye; the
//! nadir ground point below it is exposed for systems that centre on the
//! ground, like the shadow cascade). A future refinement could resolve the
//! camera's *look-at* ground intersection instead — nadir is the simple,
//! robust standard clipmap choice.

use bevy::math::DVec3;
use bevy::prelude::*;
use big_space::prelude::CellCoord;
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::SimStage;
use crate::camera::ShipCamera;
use crate::rendering::real_space::REAL_SPACE_CELL_SIZE_M;
use crate::rendering::transforms::{authored_lock_parent, surface_body_to_world_orientation_f64};
use crate::rendering::types::TidallyLocked;
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};

/// LOD hint for the anchor's single nadir terrain-height probe. Fine enough
/// that the published AGL is honest on rough terrain; the consumers' altitude
/// gates are all hundreds of metres or more.
const ANCHOR_GROUND_LOD_M: f32 = 2.0;

/// Smoothing time constant (s) for the published view speed. Long enough that
/// a one-frame hitch does not read as motion, short enough that the streaming
/// brakes keyed on the speed engage within a few frames of the camera actually
/// moving — and release about as quickly once it settles.
const ANCHOR_SPEED_TAU_S: f64 = 0.4;

/// Raw per-frame speed above which the anchor movement is a discontinuity
/// (teleport, viewpoint replay, scenario spawn), not motion: the tracker
/// reseeds at zero instead of letting one 100 km jump read as minutes of
/// hypersonic flight to the EMA. Comfortably above any real near-surface
/// speed (orbital velocity at Thalos is ~7.9 km/s).
const ANCHOR_TELEPORT_SPEED_M_S: f64 = 20_000.0;

// `ViewAnchor` / `AnchorBody` moved to `thalos_game_state::view_anchor`
// (Phase 5a); the sole writer (`update_view_anchor`) and its constants
// stay in this module.
pub use thalos_game_state::view_anchor::{AnchorBody, ViewAnchor};

pub struct ViewAnchorPlugin;

impl Plugin for ViewAnchorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewAnchor>().add_systems(
            Update,
            update_view_anchor
                .in_set(SimStage::Sync)
                .before(sync_solar_system_state),
        );
    }
}

/// Resolve the ship camera's big_space pose against the nearest terrain-backed
/// body. Runs **before** [`sync_solar_system_state`] so the previous-frame
/// camera pose is paired with the previous-frame body states it was posed
/// against (see the module note).
fn update_view_anchor(
    solar: Res<SolarSystemState>,
    sim: Res<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    ship_cam: Query<(&CellCoord, &Transform), With<ShipCamera>>,
    time: Res<Time<Real>>,
    mut speed_tracker: Local<Option<(BodyId, DVec3, f64)>>,
    mut anchor: ResMut<ViewAnchor>,
) {
    anchor.resolved = None;
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let Ok((cell, transform)) = ship_cam.single() else {
        return;
    };
    // f64-exact reconstruction: cell + local translation (invariant under
    // big_space recentring, which only moves magnitude between the two).
    let cam_pos = DVec3::new(cell.x as f64, cell.y as f64, cell.z as f64)
        * REAL_SPACE_CELL_SIZE_M as f64
        + transform.translation.as_dvec3();

    // Nearest terrain-backed body by camera altitude.
    let mut best: Option<(BodyId, f64)> = None;
    for (id, body) in sim.system.bodies.iter().enumerate() {
        if !height_sources.contains(id) {
            continue;
        }
        let Some(state) = states.get(id) else {
            continue;
        };
        let alt = (cam_pos - state.position).length() - body.radius_m;
        if best.is_none_or(|(_, best_alt)| alt < best_alt) {
            best = Some((id, alt));
        }
    }
    let Some((body, _)) = best else {
        return;
    };

    let radius_m = sim.system.bodies[body].radius_m;
    let state = &states[body];
    // Resolve into the SURFACE body-fixed frame — the frame the body grid's
    // rotation, the terrain renderers, and the height sources all share. For a
    // tidally-locked moon this is the lock composition, not the raw ephemeris
    // orientation (which is a different frame entirely).
    let lock_parent = authored_lock_parent(&sim.system.bodies[body]);
    let lock = lock_parent.map(|parent_id| TidallyLocked { parent_id });
    let orientation = surface_body_to_world_orientation_f64(body, lock.as_ref(), states)
        .unwrap_or_else(|| state.orientation.normalize());
    let cam_body = orientation.inverse() * (cam_pos - state.position);
    let cam_r = cam_body.length();
    if cam_r <= 0.0 {
        return;
    }
    let cam_dir = cam_body / cam_r;
    let ground_h_m = height_sources
        .get(body)
        .and_then(|hs| hs.sample_height_m(cam_dir.as_vec3(), ANCHOR_GROUND_LOD_M))
        .unwrap_or(0.0) as f64;

    // View speed: body-fixed displacement per REAL second (streaming happens in
    // wall time, so warp folds in exactly as it should — an orbiting view under
    // warp is genuinely outrunning the streamer). Reset on body change and on
    // teleport-sized jumps; otherwise EMA toward the raw frame speed.
    let dt = time.delta_secs_f64();
    let speed_m_s = match *speed_tracker {
        Some((prev_body, prev_cam, prev_speed)) if prev_body == body && dt > 1.0e-4 => {
            let raw = (cam_body - prev_cam).length() / dt;
            if raw > ANCHOR_TELEPORT_SPEED_M_S {
                0.0
            } else {
                let alpha = (dt / ANCHOR_SPEED_TAU_S).clamp(0.0, 1.0);
                prev_speed + (raw - prev_speed) * alpha
            }
        }
        _ => 0.0,
    };
    *speed_tracker = Some((body, cam_body, speed_m_s));

    anchor.resolved = Some(AnchorBody {
        body,
        cam_body,
        cam_dir,
        radius_m,
        ground_h_m,
        agl_m: cam_r - (radius_m + ground_h_m),
        speed_m_s,
        lock_parent,
    });
}
