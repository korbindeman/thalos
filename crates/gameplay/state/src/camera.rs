//! Camera vocabulary shared across features: the active-camera markers, the
//! focus selection, and the input-block flag. The camera rigs and drivers
//! stay with the runtime's `camera` module.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics_canonical::types::BodyState;
use thalos_world::{BodyDefinition, BodyId};

use crate::coords::RenderGhostFocus;

/// Marker component placed on every orbit camera entity (one per view).
/// Both cameras carry it; consumers that need *the active* camera should
/// query [`ActiveCamera`] instead.
#[derive(Component)]
pub struct OrbitCamera;

/// Marker placed on whichever orbit camera is currently driving the
/// rendered view. Flipped between the two cameras when [`ViewMode`]
/// changes (see [`apply_active_camera`] in `view.rs`). Use this filter
/// in queries that need the camera the user is actually looking through
/// (billboard alignment, picking, screen-space sizing).
#[derive(Component)]
pub struct ActiveCamera;

/// Set to true by the maneuver plugin when the pointer is over a maneuver
/// element (arrow, slide sphere) or an active drag/placement is in progress.
/// Camera rotation is suppressed while this is set.
#[derive(Resource, Default)]
pub struct BlockCameraInput(pub bool);

/// Semantic camera focus shared across map and ship views.
///
/// This deliberately does not use body or ship ECS entities as the shared
/// identity. Map-view proxies and ship-view real entities are different
/// worlds; systems resolve this target into their own local entity/transform.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CameraFocusTarget {
    #[default]
    None,
    Body(BodyId),
    Ship,
    PlayerController,
    /// Map-only transient focus for future encounter projections.
    Ghost(RenderGhostFocus),
}

/// The camera orbits around `target` using spherical coordinates.
///
/// Distance is in metres, stored as f64 to cover the full range from
/// 100 km (low orbit) to ~67 AU without precision loss.
/// Azimuth and elevation are in radians.
///
/// Zoom is smoothed: scroll input sets `target_distance` and each frame
/// `distance` interpolates toward it in log-space for scale-independent feel.
#[derive(Resource)]
pub struct CameraFocus {
    /// Semantic target to orbit around.
    pub target: CameraFocusTarget,
    /// Current radial distance from the target, in metres (interpolated each frame).
    pub distance: f64,
    /// Desired radial distance — scroll input drives this, `distance` chases it.
    pub target_distance: f64,
    /// Horizontal angle around the target, in radians.
    pub azimuth: f32,
    /// Vertical angle from the equatorial plane, clamped to ±89°, in radians.
    pub elevation: f32,
    /// Minimum distance in metres — set to the focused body's surface radius.
    pub min_distance: f64,
    /// Physics-space (heliocentric, metres, f64) position the
    /// [`RenderOrigin`](crate::coords::RenderOrigin) sat at when the
    /// current focus transition began. While `Some`, the origin
    /// interpolates in f64 from this point to the new focus target's
    /// physics position over [`FOCUS_TRANSITION_DURATION_S`]. `None`
    /// when no transition is active — origin tracks the focus target
    /// directly.
    ///
    /// Stored in physics space (rather than as a render-space `Vec3`
    /// offset) so the camera never sits at large render-unit
    /// coordinates mid-switch — at MAP_SCALE the old visual position
    /// of a distant body can be 1e6+ RU, which collapses
    /// `looking_at`'s `(target − camera).normalize()` to f32 ulp
    /// noise. With the origin interpolating in f64, both the camera
    /// and its target stay near render-space (0,0,0) throughout.
    pub transition_origin_start: Option<DVec3>,
    /// Azimuth at the moment the current transition began. The renderer
    /// reads `azimuth` interpolated from this value toward the field's
    /// current value across the transition, so a focus pick that also
    /// retargets the camera (e.g. body-tree pick → Sun-side aim) pans
    /// smoothly instead of snapping. Only valid while `transition_origin_start`
    /// is `Some`.
    pub transition_azimuth_start: f32,
    /// Elevation at the moment the current transition began. See
    /// [`Self::transition_azimuth_start`] for the rationale.
    pub transition_elevation_start: f32,
    /// Seconds elapsed since the current transition began. Reset on each
    /// focus switch.
    pub transition_elapsed_s: f64,
}

/// Find the body whose sphere of influence contains `ship_pos` and is
/// smallest among such bodies — the same rule the patched-conics propagator
/// uses to pick an anchor. The star (infinite SOI) is the fallback.
pub fn find_reference_body(
    ship_pos: bevy::math::DVec3,
    bodies: &[BodyDefinition],
    states: &[BodyState],
) -> usize {
    let mut best: Option<(usize, f64)> = None;
    for body in bodies {
        let dist_sq = (ship_pos - states[body.id].position).length_squared();
        if dist_sq < body.soi_radius_m * body.soi_radius_m {
            match best {
                None => best = Some((body.id, body.soi_radius_m)),
                Some((_, soi)) if body.soi_radius_m < soi => {
                    best = Some((body.id, body.soi_radius_m));
                }
                _ => {}
            }
        }
    }
    // Fallback: the star (infinite SOI) is always a match, but be defensive
    // in case the body list is empty for any reason.
    best.map(|(id, _)| id).unwrap_or(0)
}

impl Default for CameraFocus {
    fn default() -> Self {
        Self {
            target: CameraFocusTarget::None,
            distance: 5e11, // ~3.3 AU, sees inner system
            target_distance: 5e11,
            // `azimuth = 0` sits in front of the nose looking back at the
            // craft; start behind it instead (a chase view) since that's
            // the far more useful default for flying.
            azimuth: std::f32::consts::PI,
            elevation: 0.3, // slight downward tilt so the horizon is visible
            min_distance: DISTANCE_MIN_DEFAULT,
            transition_origin_start: None,
            transition_azimuth_start: 0.0,
            transition_elevation_start: 0.0,
            transition_elapsed_s: 0.0,
        }
    }
}

impl CameraFocus {
    /// Begin a smooth transition to `target`. `current_origin` is the
    /// physics-space position the [`RenderOrigin`](crate::coords::RenderOrigin)
    /// sits at right now (typically the previous focus body's heliocentric
    /// position, possibly already mid-interpolation if the user retargets
    /// during a transition). The origin will interpolate in f64 from this
    /// point to the new target's physics position over
    /// [`FOCUS_TRANSITION_DURATION_S`] seconds regardless of distance, so
    /// the camera never sits at large render-unit coordinates during the
    /// switch.
    ///
    /// Preserves the current zoom (`target_distance`). Callers that want
    /// to also frame the new body to a comparable on-screen size should
    /// follow up with [`Self::frame_for_radius`].
    pub fn focus_on(&mut self, target: CameraFocusTarget, current_origin: DVec3) {
        // Capture *effective* (mid-transition) az/el so a retarget while
        // a transition is already in flight continues smoothly from where
        // the camera currently appears, not from the previous target's
        // stored values.
        let start_az = self.effective_azimuth();
        let start_el = self.effective_elevation();
        self.transition_origin_start = Some(current_origin);
        self.transition_azimuth_start = start_az;
        self.transition_elevation_start = start_el;
        self.transition_elapsed_s = 0.0;
        self.target = target;
    }

    pub fn focus_on_body(&mut self, body_id: BodyId, current_origin: DVec3) {
        self.focus_on(CameraFocusTarget::Body(body_id), current_origin);
    }

    pub fn focus_on_ship(&mut self, current_origin: DVec3) {
        self.focus_on(CameraFocusTarget::Ship, current_origin);
    }

    /// Set `target_distance` to a body-sized framing distance — bodies
    /// sharing a radius land at the same zoom, so on-screen size stays
    /// comparable across the system. Body-tree picks call this; passive
    /// refocus events (double-click, ghost retirement) do not, so they
    /// preserve whatever zoom the user had.
    pub fn frame_for_radius(&mut self, radius_m: f64) {
        self.target_distance = (radius_m * FOCUS_FRAMING_RADII).max(DISTANCE_MIN_DEFAULT);
    }

    /// Set [`azimuth`](Self::azimuth) and [`elevation`](Self::elevation)
    /// so the camera-to-target offset points along `world_dir` — i.e. the
    /// camera ends up sitting at `target + world_dir * distance`. Used to
    /// place the camera on the lit side of a body (Sun-direction) when the
    /// user picks it from the body tree.
    ///
    /// Only meaningful in map view, where the camera basis is the world
    /// axes. Ship view uses a gravity-aligned basis that this helper
    /// doesn't translate to.
    pub fn aim_from(&mut self, world_dir: Vec3) {
        let dir = world_dir.normalize_or_zero();
        if dir == Vec3::ZERO {
            return;
        }
        self.elevation = dir.y.asin();
        self.azimuth = dir.x.atan2(dir.z);
    }

    /// Azimuth as it appears this frame. While a focus transition is
    /// active, lerps from [`Self::transition_azimuth_start`] toward
    /// [`Self::azimuth`] using the same eased curve as the origin lerp;
    /// otherwise returns [`Self::azimuth`] directly. Shortest-arc wrapped
    /// so a 350°→10° transition pans 20° forward, not 340° back.
    pub fn effective_azimuth(&self) -> f32 {
        if self.transition_origin_start.is_none() {
            return self.azimuth;
        }
        let t = focus_transition_progress(self) as f32;
        let delta = wrap_pi(self.azimuth - self.transition_azimuth_start);
        self.transition_azimuth_start + delta * t
    }

    /// Elevation as it appears this frame — see [`Self::effective_azimuth`].
    /// No wrap needed: elevation is clamped to ±89°.
    pub fn effective_elevation(&self) -> f32 {
        if self.transition_origin_start.is_none() {
            return self.elevation;
        }
        let t = focus_transition_progress(self) as f32;
        self.transition_elevation_start + (self.elevation - self.transition_elevation_start) * t
    }
}

pub const DISTANCE_MIN_DEFAULT: f64 = 1e5;

/// Multiple of body radius used as the framing distance when switching
/// focus. ~10× gives an establishing-shot view — body clearly visible in
/// frame without dominating it. Must stay above [`SURFACE_MARGIN`] so
/// `camera_min_distance_system` doesn't clamp the framing back up.
pub const FOCUS_FRAMING_RADII: f64 = 10.0;

/// Eased progress of the active focus transition in `[0.0, 1.0]`.
/// Returns `1.0` when no transition is active so `update_render_origin`
/// lerps directly to the focus target. Ease-out cubic — most of the
/// visual movement lands in the first ~30 % of the duration, the last
/// fraction settles gently.
pub fn focus_transition_progress(focus: &CameraFocus) -> f64 {
    if focus.transition_origin_start.is_none() {
        return 1.0;
    }
    let t = (focus.transition_elapsed_s / FOCUS_TRANSITION_DURATION_S).clamp(0.0, 1.0);
    1.0 - (1.0 - t).powi(3)
}

/// Wrap `angle` to `(-π, π]` for shortest-arc azimuth interpolation.
pub fn wrap_pi(angle: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let mut a = angle % TAU;
    if a > PI {
        a -= TAU;
    } else if a < -PI {
        a += TAU;
    }
    a
}

/// Duration of a focus-switch transition, regardless of distance between
/// bodies. Tuned for snappy-but-not-jarring camera handoff.
pub const FOCUS_TRANSITION_DURATION_S: f64 = 0.8;
