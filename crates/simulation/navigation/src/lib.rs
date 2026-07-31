//! `thalos_navigation` — route navigation and guidance.
//!
//! One authority for "where am I going, what path gets me there, and how far
//! off it am I". Every navigation display is a **projection** of this crate's
//! output; no display, autopilot, or HUD element re-derives route geometry.
//!
//! Pure Rust, no Bevy: the game crate owns the resources and systems that pick
//! a destination and publish the result (`thalos_runtime::route`), this crate
//! owns the geometry and the control-independent guidance policy as testable
//! functions. Same split as [`thalos_control`](../thalos_control/index.html),
//! which consumes the guidance commands at the fly-by-wire seam.
//!
//! # Frames
//!
//! Route geometry is planned in a **route frame** ([`RouteFrame`]): a local
//! east/north tangent plane anchored at a body-fixed origin (for an approach,
//! the landing threshold), with altitude carried separately as height above the
//! body's reference radius. Lateral work is therefore plane geometry — cheap,
//! exact enough over the tens of km an approach spans — while the vertical axis
//! stays radial and never inherits the plane's curvature error. See
//! [`RouteFrame`] for the error budget.
//!
//! # Shape
//!
//! - [`waypoint`] — the waypoint/route model. A waypoint is horizontal by
//!   definition and vertical *optionally*: planes and submersibles constrain
//!   altitude/depth, rovers and boats do not.
//! - [`path`] — lateral path primitives (straight legs and constant-radius
//!   arcs), arclength parameterisation, and closest-point queries. This is what
//!   "cross-track error" is measured against and what the ND draws.
//! - [`dubins`] — the shortest bank-limited path between two poses. Turning a
//!   "fly to this runway" request into a path that is actually flyable from
//!   wherever the craft happens to be (including behind or offset from the
//!   runway) is exactly a Dubins problem.
//! - [`approach`] — runway descriptors and the approach planner: a straight
//!   final aligned with the landing heading, plus the Dubins transition that
//!   joins the craft's current pose to it.
//! - [`rejoin`] — the flyable way *back* onto a route after drifting off it: a
//!   bank-limited path to the earliest point on the route the craft can meet
//!   tangentially. Guidance steers along this, so being off course produces a
//!   real intercept rather than a heuristic nudge — and it is route-relative, so
//!   it works identically on any leg and on any future waypoint route.
//! - [`vnav`] — the vertical profile over distance-to-go: level at the capture
//!   altitude, then the glideslope to the threshold, with speed gates.
//! - [`guidance`] — the per-frame deviations and commands the displays and the
//!   autopilot read ([`Guidance`]).
//!
//! # Not here (deliberately)
//!
//! Craft-specific control law (that is `thalos_control`), terrain clearance
//! checks along the route, airspace/procedure data, and anything Bevy. The
//! deferred scope — arbitrary player-entered waypoints, ground-vehicle routes,
//! submersible depth legs — is designed for by the waypoint model but not
//! implemented; see `docs/gameplay/navigation.md`.

pub mod approach;
pub mod destination;
pub mod dubins;
pub mod guidance;
pub mod path;
pub mod rejoin;
pub mod vnav;
pub mod waypoint;

pub use approach::{
    ApproachParams, ApproachPhase, ApproachPlan, RunwayEnd, RunwayStrip, plan_approach,
};
pub use destination::{
    DestinationGuidance, DestinationInput, DestinationParams, angular_distance_rad,
    compute_destination_guidance, great_circle_tangent,
};
pub use dubins::{DubinsPath, DubinsWord, Pose2, plan_dubins};
pub use guidance::{
    GS_FULL_SCALE_RAD, Guidance, GuidanceInput, LOC_FULL_SCALE_RAD, compute_guidance,
};
pub use path::{Arc2, LateralPath, Leg, PathPoint};
pub use rejoin::{Rejoin, RejoinParams, plan_rejoin};
pub use vnav::{SpeedGate, VerticalProfile, VnavParams};
pub use waypoint::{
    RouteFrame, VerticalConstraint, Waypoint, WaypointKind, dir_from_theta, heading_to_theta,
    theta_of, theta_to_heading,
};

/// Standard-gravity constant used to size bank-limited turn radii (m/s²).
///
/// Turn radius is `v² / (g · tan φ)`. The `g` that belongs there is the *local*
/// gravitational acceleration, which the caller supplies
/// ([`ApproachParams::gravity_m_s2`]) because Thalos is not Earth and the same
/// craft turns wider on a heavier world. This constant is only the fallback /
/// documentation anchor for Earth-like sizing.
pub const EARTH_G_M_S2: f64 = 9.80665;

/// Wrap an angle into `(−π, π]` so heading errors take the short way around.
///
/// Shared by every module here: a guidance loop that wraps inconsistently
/// commands a 350° turn to fix a 10° error.
pub fn wrap_angle(angle: f64) -> f64 {
    let wrapped = (angle + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;
    if wrapped == -std::f64::consts::PI {
        std::f64::consts::PI
    } else {
        wrapped
    }
}

/// Wrap an angle into `[0, 2π)` — the form arc sweeps and compass bearings use.
pub fn wrap_positive(angle: f64) -> f64 {
    angle.rem_euclid(2.0 * std::f64::consts::PI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn wrap_angle_takes_the_short_way() {
        assert_abs_diff_eq!(wrap_angle(0.1), 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(
            wrap_angle(350.0_f64.to_radians()),
            -10.0_f64.to_radians(),
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(wrap_angle(std::f64::consts::PI), std::f64::consts::PI);
        assert_abs_diff_eq!(wrap_angle(-std::f64::consts::PI), std::f64::consts::PI);
    }

    #[test]
    fn wrap_positive_stays_in_turn() {
        assert_abs_diff_eq!(wrap_positive(-0.5), 2.0 * std::f64::consts::PI - 0.5);
        assert_abs_diff_eq!(wrap_positive(0.5), 0.5);
    }
}
