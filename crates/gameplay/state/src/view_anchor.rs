//! The **view anchor** — the one per-frame answer to "where is the view?" for
//! every view-dependent world-detail system.
//!
//! Thalos is one physical world that must render consistently from any camera
//! (see `docs/roadmap/graphics_fidelity.md` §2.3, the one-world principle).
//! Surface scatter (trees / grass / rocks), the sun-shadow cascade centre, and
//! any future clipmap-like layer all read [`ViewAnchor`] — the **render
//! camera's** position resolved against the nearest terrain-backed body — and
//! none of them may anchor to the player craft, a mode flag, or a per-mode
//! focus override. The craft is just an object *in* the world; the view is
//! what decides where detail exists. This makes new camera modes correct *by
//! construction*: reposition the camera however you like — flight orbit,
//! freecam, god-view hub, base editor, headless screenshot rig — and every
//! detail system follows, with no mode-specific plumbing.
//!
//! ## Frame coherence (why the anchor is stored body-fixed)
//!
//! The camera's big_space pose is written by the camera drivers *after* the
//! detail drivers have run, so a driver reading the camera directly sees a
//! one-frame-stale heliocentric pose. At 1× a surface point co-rotates at
//! hundreds of m/s, and under warp far faster: a stale heliocentric point
//! re-interpreted at this frame's epoch is kilometres off in body-fixed
//! terms. The fix is to resolve the pose into the **body-fixed frame once, at
//! a matching epoch** (the sole writer runs at the top of `SimStage::Sync`,
//! before `sync_solar_system_state` refreshes the states, pairing the
//! previous-frame camera pose with the previous-frame body states it was
//! posed against). In the body frame the one-frame lag shrinks to the
//! camera's own body-relative motion — metres. Consumers re-project the
//! body-fixed anchor with *their* frame's body state ([`AnchorBody::cam_world`]
//! / the ground substitution documented there) and get a this-epoch point
//! that co-rotates correctly.
//!
//! The anchor point is the camera itself (detail is built around the eye; the
//! nadir ground point below it is exposed for systems that centre on the
//! ground, like the shadow cascade).

use bevy::math::{DQuat, DVec3};
use bevy::prelude::*;
use thalos_physics_canonical::types::BodyState;
use thalos_world::BodyId;

use crate::scene::TidallyLocked;
use crate::surface_frame::surface_body_to_world_orientation_f64;

/// Where the view is, resolved against the nearest terrain-backed body.
/// `None` until a camera and body states exist (first frames of a boot, or a
/// deferred-world menu).
///
/// **Sole writer:** the runtime's `update_view_anchor`.
#[derive(Resource, Default, Clone, Copy)]
pub struct ViewAnchor {
    pub resolved: Option<AnchorBody>,
}

/// The view anchor resolved in the body-fixed frame of [`Self::body`]. All
/// positions/directions are body-fixed metres, in the **surface frame** — the
/// same [`surface_body_to_world_orientation_f64`] frame the real-space body
/// grid, the udlod terrain, the tile terrain, and the height sources use.
/// (For a tidally-locked moon that frame differs from the raw ephemeris
/// `BodyState::orientation` by the full lock rotation — resolving with the
/// ephemeris frame put the anchor ~130° of longitude away from the camera on
/// Mira, see INC-20260723T232652Z's successor investigation.) Re-project to
/// heliocentric with the *current* frame's states via [`Self::cam_world`]
/// (see the module note on frame coherence).
#[derive(Clone, Copy)]
pub struct AnchorBody {
    /// The nearest terrain-backed body (has a registered height source).
    pub body: BodyId,
    /// Camera position, body-fixed (surface frame).
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
    /// Smoothed body-fixed camera speed (m/s) — how fast the view moves
    /// through the surface frame, EMA-filtered by the writer. Co-rotation
    /// counts (a world-hovering camera over a spinning body IS moving
    /// relative to the ground streaming under it). Detail systems use
    /// this to trade fidelity for coverage while the view is in motion and
    /// settle back to full fidelity where it lingers; zero after a teleport.
    pub speed_m_s: f64,
    /// Tidal-lock parent (from the body entity's `TidallyLocked` tag), carried
    /// so re-projection can rebuild the surface orientation at any epoch.
    pub lock_parent: Option<BodyId>,
}

impl AnchorBody {
    /// The body's surface (body-fixed → world) orientation at the epoch of
    /// `states` — the one orientation authority every surface consumer shares.
    pub fn surface_orientation(&self, states: &[BodyState]) -> DQuat {
        let lock = self
            .lock_parent
            .map(|parent_id| TidallyLocked { parent_id });
        surface_body_to_world_orientation_f64(self.body, lock.as_ref(), states)
            .or_else(|| states.get(self.body).map(|s| s.orientation.normalize()))
            .unwrap_or(DQuat::IDENTITY)
    }

    /// Heliocentric camera position at the epoch of `states`. (The ground
    /// point below it is derivable by substituting
    /// `cam_dir * (radius_m + ground_h_m)` for `cam_body`.)
    pub fn cam_world(&self, states: &[BodyState]) -> DVec3 {
        let position = states
            .get(self.body)
            .map(|s| s.position)
            .unwrap_or(DVec3::ZERO);
        position + self.surface_orientation(states) * self.cam_body
    }
}
