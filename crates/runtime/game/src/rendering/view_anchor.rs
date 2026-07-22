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
use thalos_physics_canonical::types::BodyState;
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::SimStage;
use crate::camera::ShipCamera;
use crate::rendering::real_space::REAL_SPACE_CELL_SIZE_M;
use crate::solar_system_state::{SimulationState, SolarSystemState, sync_solar_system_state};

/// LOD hint for the anchor's single nadir terrain-height probe. Fine enough
/// that the published AGL is honest on rough terrain; the consumers' altitude
/// gates are all hundreds of metres or more.
const ANCHOR_GROUND_LOD_M: f32 = 2.0;

/// Where the view is, resolved against the nearest terrain-backed body.
/// `None` until a camera and body states exist (first frames of a boot, or a
/// deferred-world menu).
///
/// **Sole writer:** [`update_view_anchor`].
#[derive(Resource, Default, Clone, Copy)]
pub struct ViewAnchor {
    pub resolved: Option<AnchorBody>,
}

/// The view anchor resolved in the body-fixed frame of [`Self::body`]. All
/// positions/directions are body-fixed metres; re-project to heliocentric with
/// the *current* frame's [`BodyState`] via [`Self::cam_world`] /
/// [`Self::ground_world`] (see the module note on frame coherence).
#[derive(Clone, Copy)]
pub struct AnchorBody {
    /// The nearest terrain-backed body (has a registered height source).
    pub body: BodyId,
    /// Camera position, body-fixed.
    pub cam_body: DVec3,
    /// Unit nadir direction (`cam_body / |cam_body|`), body-fixed.
    pub cam_dir: DVec3,
    /// Body reference radius, metres.
    pub radius_m: f64,
    /// Terrain height at the nadir, metres above the reference radius
    /// (datum fallback `0.0` while tiles are cold).
    pub ground_h_m: f64,
    /// Camera altitude above the sampled terrain, metres.
    pub agl_m: f64,
}

impl AnchorBody {
    /// Heliocentric camera position at `state`'s epoch. (The ground point
    /// below it is derivable as `state.position + state.orientation *
    /// (cam_dir * (radius_m + ground_h_m))` should a consumer need it.)
    pub fn cam_world(&self, state: &BodyState) -> DVec3 {
        state.position + state.orientation * self.cam_body
    }
}

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
    let cam_body = state.orientation.inverse() * (cam_pos - state.position);
    let cam_r = cam_body.length();
    if cam_r <= 0.0 {
        return;
    }
    let cam_dir = cam_body / cam_r;
    let ground_h_m = height_sources
        .get(body)
        .and_then(|hs| hs.sample_height_m(cam_dir.as_vec3(), ANCHOR_GROUND_LOD_M))
        .unwrap_or(0.0) as f64;

    anchor.resolved = Some(AnchorBody {
        body,
        cam_body,
        cam_dir,
        radius_m,
        ground_h_m,
        agl_m: cam_r - (radius_m + ground_h_m),
    });
}
