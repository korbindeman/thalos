//! The waypoint / route model and the local plane every route is planned in.
//!
//! # Angle conventions (read this before touching any geometry)
//!
//! Two conventions meet here, and mixing them is the classic way to command a
//! 350° turn to fix a 10° error:
//!
//! - **Compass heading** `h` — what a pilot, the HUD, and
//!   [`crate::guidance::Guidance`] speak: `0` = north, increasing **clockwise**
//!   (90° = east). Direction vector in local coordinates is `(sin h, cos h)`.
//! - **Math angle** `θ` — what all the internal geometry ([`crate::path`],
//!   [`crate::dubins`], [`crate::approach`]) uses: measured **counter-clockwise
//!   from +east** in the local `(x = east, y = north)` plane, so direction is
//!   `(cos θ, sin θ)` and rotations follow the usual sign rules.
//!
//! `θ = π/2 − h` converts either way ([`heading_to_theta`],
//! [`theta_to_heading`]). Because +y is north and θ is CCW, a **pilot's left
//! turn is a mathematically positive (CCW) rotation** — that is why
//! [`crate::dubins::DubinsWord`]'s `L` really is "turn left".

use glam::{DVec2, DVec3};

/// Compass heading (rad, 0 = north, CW) → internal math angle (rad, CCW from
/// east). Involutive: applying it twice returns the original angle.
pub fn heading_to_theta(heading_rad: f64) -> f64 {
    std::f64::consts::FRAC_PI_2 - heading_rad
}

/// Internal math angle (rad, CCW from east) → compass heading (rad, 0 = north,
/// CW), wrapped to `[0, 2π)`.
pub fn theta_to_heading(theta_rad: f64) -> f64 {
    crate::wrap_positive(std::f64::consts::FRAC_PI_2 - theta_rad)
}

/// Unit direction in the local plane for a math angle.
pub fn dir_from_theta(theta_rad: f64) -> DVec2 {
    DVec2::new(theta_rad.cos(), theta_rad.sin())
}

/// Math angle of a local-plane direction (zero vector → 0).
pub fn theta_of(dir: DVec2) -> f64 {
    dir.y.atan2(dir.x)
}

/// A local east/north tangent plane anchored at a body-fixed origin, plus the
/// body radius that turns radial distance into altitude.
///
/// # Why a plane, and what it costs
///
/// The lateral projection is **gnomonic**: a position is projected radially
/// outward onto the plane tangent at the origin. Three properties earn it the
/// job:
///
/// - **It is altitude-independent.** A craft at 3 km reads the same
///   `(east, north)` as the ground point beneath it. A naive chord projection
///   does not: it drifts by `d · Δalt / R`, which is ~9 m over a 25 km final —
///   enough to put the computed route length, the distance-to-go, and the
///   glideslope deviation all slightly wrong, in a way that looks like noise
///   rather than a bug.
/// - **It is an exact inverse of [`Self::to_body_fixed`]**, because that method
///   builds a point *on* the tangent plane before projecting it to the sphere.
///   Round-tripping is exact to floating-point, not to a tolerance.
/// - **Straight lines in the plane are great circles on the sphere** — which is
///   what a "straight leg" should mean for navigation.
///
/// The cost is a `tan`-shaped range distortion, `≈ d³/3R²` — about 0.25 m at
/// 30 km on a 6,000 km body, and it diverges at the horizon (guarded, see
/// [`Self::to_local`]).
///
/// **Altitude is never a plane coordinate.** The tangent plane runs `d²/2R`
/// ≈ 75 m above the sphere at 30 km, which would put a glideslope badly wrong,
/// so altitude is always height above the body's reference radius, measured
/// radially ([`Self::altitude_of`]).
///
/// The origin is body-**fixed** (it rotates with the planet), so a route stays
/// nailed to the ground while the body spins under an inertial craft. Callers
/// convert their inertial craft position into the body-fixed frame before
/// entering this frame; that is one rotation by the body orientation and it is
/// the game side's job (`thalos_runtime::route`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteFrame {
    /// Unit body-fixed direction to the plane's origin.
    pub origin_dir: DVec3,
    /// Body reference radius (m).
    pub body_radius_m: f64,
    /// Height of the origin above the reference radius (m) — for an approach,
    /// the threshold elevation.
    pub origin_altitude_m: f64,
    /// Local east basis vector (unit, body-fixed).
    pub east: DVec3,
    /// Local north basis vector (unit, body-fixed).
    pub north: DVec3,
}

impl RouteFrame {
    /// Cosine floor for the gnomonic projection ([`Self::to_local`]): positions
    /// within ~89.94° of the origin project faithfully, beyond that they clamp.
    pub const MIN_HORIZON_COS: f64 = 1.0e-3;

    /// Build the frame at a body-fixed origin direction. `north` is the body's
    /// spin axis (+Y) projected onto the tangent plane, with an X-axis fallback
    /// at the poles — the same construction as the HUD's shared
    /// `local_enu_basis`, so route bearings and HUD headings agree by
    /// definition rather than by coincidence.
    ///
    /// Returns `None` only for a degenerate origin (zero-length direction).
    pub fn new(origin_dir: DVec3, body_radius_m: f64, origin_altitude_m: f64) -> Option<Self> {
        let up = origin_dir.try_normalize()?;
        let mut north = DVec3::Y - DVec3::Y.dot(up) * up;
        if north.length_squared() < 1e-12 {
            north = DVec3::X - DVec3::X.dot(up) * up;
        }
        let north = north.try_normalize()?;
        let east = north.cross(up);
        Some(Self {
            origin_dir: up,
            body_radius_m,
            origin_altitude_m,
            east,
            north,
        })
    }

    /// The origin as a body-fixed point (m from the body centre).
    pub fn origin_point(&self) -> DVec3 {
        self.origin_dir * (self.body_radius_m + self.origin_altitude_m)
    }

    /// Project a body-fixed position into local `(east, north)` metres
    /// (gnomonic — see the type docs for why, and what it costs).
    ///
    /// Positions at or beyond the origin's horizon have no gnomonic image; the
    /// cosine is clamped to [`Self::MIN_HORIZON_COS`] so such a position lands
    /// far out in the correct direction instead of diverging or flipping sign.
    /// Nothing in an approach ever gets near this — it exists so a display that
    /// naively projects a runway on the far side of the planet degrades instead
    /// of drawing garbage.
    pub fn to_local(&self, body_fixed_pos: DVec3) -> DVec2 {
        let Some(dir) = body_fixed_pos.try_normalize() else {
            return DVec2::ZERO;
        };
        let cos = dir.dot(self.origin_dir).max(Self::MIN_HORIZON_COS);
        let on_plane = dir / cos * (self.body_radius_m + self.origin_altitude_m);
        let delta = on_plane - self.origin_point();
        DVec2::new(delta.dot(self.east), delta.dot(self.north))
    }

    /// Whether a body-fixed position is on the origin's visible hemisphere, i.e.
    /// whether [`Self::to_local`] is a faithful projection rather than a clamped
    /// one.
    pub fn is_projectable(&self, body_fixed_pos: DVec3) -> bool {
        body_fixed_pos
            .try_normalize()
            .is_some_and(|dir| dir.dot(self.origin_dir) > Self::MIN_HORIZON_COS)
    }

    /// Altitude (m above the body reference radius) of a body-fixed position —
    /// radial, never a plane coordinate. See the type docs.
    pub fn altitude_of(&self, body_fixed_pos: DVec3) -> f64 {
        body_fixed_pos.length() - self.body_radius_m
    }

    /// Local `(east, north)` + altitude back to a body-fixed position. Inverse
    /// of [`Self::to_local`] / [`Self::altitude_of`] to within the frame's
    /// documented error budget.
    pub fn to_body_fixed(&self, local: DVec2, altitude_m: f64) -> DVec3 {
        let flat = self.origin_point() + self.east * local.x + self.north * local.y;
        flat.normalize_or(self.origin_dir) * (self.body_radius_m + altitude_m)
    }

    /// Rotate a body-fixed direction (e.g. a runway heading tangent, or the
    /// craft's nose) into the local plane, as an unnormalised `(east, north)`
    /// vector. The vertical component is dropped, which is what "ground track"
    /// means.
    pub fn direction_to_local(&self, body_fixed_dir: DVec3) -> DVec2 {
        DVec2::new(
            body_fixed_dir.dot(self.east),
            body_fixed_dir.dot(self.north),
        )
    }
}

/// What kind of vertical constraint a waypoint carries.
///
/// A waypoint is horizontal by definition and vertical only when the vehicle
/// has a vertical axis to command: **planes and submersibles constrain it,
/// rovers and surface ships do not**. `None` (the [`Waypoint::vertical`] field
/// being absent) is therefore not "missing data" — it is the correct state for
/// a ground route, and guidance must degrade to lateral-only rather than
/// inventing an altitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerticalConstraint {
    /// Cross at exactly this altitude (m above the body reference radius).
    At(f64),
    /// Cross at or above.
    AtOrAbove(f64),
    /// Cross at or below.
    AtOrBelow(f64),
    /// Cross inside a window (`lo`, `hi`), both m above the reference radius.
    Window { lo_m: f64, hi_m: f64 },
}

impl VerticalConstraint {
    /// The altitude a vertical profile should aim for to satisfy this
    /// constraint: the exact/bounding value, or the middle of a window.
    pub fn target_m(self) -> f64 {
        match self {
            Self::At(a) | Self::AtOrAbove(a) | Self::AtOrBelow(a) => a,
            Self::Window { lo_m, hi_m } => 0.5 * (lo_m + hi_m),
        }
    }

    /// Whether `altitude_m` satisfies the constraint within `tolerance_m`.
    pub fn satisfied(self, altitude_m: f64, tolerance_m: f64) -> bool {
        match self {
            Self::At(a) => (altitude_m - a).abs() <= tolerance_m,
            Self::AtOrAbove(a) => altitude_m >= a - tolerance_m,
            Self::AtOrBelow(a) => altitude_m <= a + tolerance_m,
            Self::Window { lo_m, hi_m } => {
                altitude_m >= lo_m - tolerance_m && altitude_m <= hi_m + tolerance_m
            }
        }
    }
}

/// What a waypoint *is*, which decides how it is drawn and whether guidance may
/// sequence past it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaypointKind {
    /// A plain fly-over/fly-by point.
    Fix,
    /// The point where the route joins the final approach segment (FAP).
    FinalApproach,
    /// A runway threshold — the end of an approach route.
    Threshold,
    /// A touchdown/aim point beyond a threshold.
    Aim,
}

/// A single navigable point in a route.
///
/// Position is stored **body-fixed** (a unit direction) rather than in a route
/// frame, so a waypoint outlives the frame it was planned in and survives the
/// planet rotating. The lateral geometry that consumes it works in a
/// [`RouteFrame`]; conversion is [`RouteFrame::to_local`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waypoint {
    /// Unit body-fixed direction to the point.
    pub dir: DVec3,
    /// Vertical constraint, if the vehicle has a vertical axis. See
    /// [`VerticalConstraint`].
    pub vertical: Option<VerticalConstraint>,
    /// Target speed over the point (m/s), if constrained.
    pub speed_m_s: Option<f64>,
    pub kind: WaypointKind,
}

impl Waypoint {
    /// A lateral-only fix — the correct shape for a rover or surface-ship route.
    pub fn fix(dir: DVec3) -> Self {
        Self {
            dir,
            vertical: None,
            speed_m_s: None,
            kind: WaypointKind::Fix,
        }
    }

    /// Attach a vertical constraint (planes, submersibles).
    pub fn with_vertical(mut self, vertical: VerticalConstraint) -> Self {
        self.vertical = Some(vertical);
        self
    }

    /// Attach a speed target.
    pub fn with_speed(mut self, speed_m_s: f64) -> Self {
        self.speed_m_s = Some(speed_m_s);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn frame() -> RouteFrame {
        RouteFrame::new(DVec3::X, 6.0e6, 100.0).expect("non-degenerate origin")
    }

    #[test]
    fn compass_and_math_angles_round_trip() {
        // North.
        assert_abs_diff_eq!(heading_to_theta(0.0), std::f64::consts::FRAC_PI_2);
        // East compass → 0 rad math.
        assert_abs_diff_eq!(heading_to_theta(std::f64::consts::FRAC_PI_2), 0.0);
        for h_deg in [0.0_f64, 30.0, 143.0, 270.0, 359.0] {
            let h = h_deg.to_radians();
            assert_abs_diff_eq!(theta_to_heading(heading_to_theta(h)), h, epsilon = 1e-12);
        }
    }

    #[test]
    fn heading_direction_matches_compass_sense() {
        // Heading 90° (east) must point along +east = +x in the local plane.
        let d = dir_from_theta(heading_to_theta(std::f64::consts::FRAC_PI_2));
        assert_abs_diff_eq!(d.x, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.y, 0.0, epsilon = 1e-12);
        // Heading 0 (north) → +y.
        let d = dir_from_theta(heading_to_theta(0.0));
        assert_abs_diff_eq!(d.x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.y, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn a_pilots_left_turn_is_a_ccw_rotation() {
        // Facing north, turning left ends up facing west.
        let theta_north = heading_to_theta(0.0);
        let theta_west = heading_to_theta(270.0_f64.to_radians());
        // CCW (increasing θ) by 90° gets there.
        assert_abs_diff_eq!(
            crate::wrap_angle(theta_west - theta_north),
            std::f64::consts::FRAC_PI_2,
            epsilon = 1e-12
        );
    }

    #[test]
    fn local_round_trip_is_exact() {
        let f = frame();
        for (e, n, alt) in [
            (0.0, 0.0, 100.0),
            (1_000.0, -2_500.0, 800.0),
            (-30_000.0, 12_000.0, 3_000.0),
        ] {
            let bf = f.to_body_fixed(DVec2::new(e, n), alt);
            let back = f.to_local(bf);
            // Gnomonic projection inverts `to_body_fixed` exactly, not to a
            // tolerance — see the type docs.
            assert_abs_diff_eq!(back.x, e, epsilon = 1e-6);
            assert_abs_diff_eq!(back.y, n, epsilon = 1e-6);
            assert_abs_diff_eq!(f.altitude_of(bf), alt, epsilon = 1e-6);
        }
    }

    #[test]
    fn local_coordinates_do_not_drift_with_altitude() {
        // Regression: a chord projection made the same ground track read ~9 m
        // further out at 3 km than at 100 m over a 25 km final, which then
        // corrupted route length, distance-to-go, and glideslope deviation.
        let f = frame();
        let ground = f.to_body_fixed(DVec2::new(-25_000.0, 4_000.0), f.origin_altitude_m);
        let ground_local = f.to_local(ground);
        let dir = ground.normalize();
        for alt in [200.0, 1_000.0, 3_000.0, 11_000.0] {
            let aloft = dir * (f.body_radius_m + alt);
            let local = f.to_local(aloft);
            assert_abs_diff_eq!(local.x, ground_local.x, epsilon = 1e-6);
            assert_abs_diff_eq!(local.y, ground_local.y, epsilon = 1e-6);
        }
    }

    #[test]
    fn beyond_the_horizon_clamps_instead_of_diverging() {
        let f = frame();
        // Antipodal-ish: no gnomonic image exists.
        let far = -f.origin_dir * (f.body_radius_m + 1_000.0);
        assert!(!f.is_projectable(far));
        let local = f.to_local(far);
        assert!(
            local.x.is_finite() && local.y.is_finite(),
            "clamped projection must stay finite"
        );
        // A normal approach position is projectable.
        let near = f.to_body_fixed(DVec2::new(-20_000.0, 0.0), 2_000.0);
        assert!(f.is_projectable(near));
    }

    #[test]
    fn altitude_is_radial_not_planar() {
        // A point 30 km out *in the plane* at the origin's own altitude must
        // read back BELOW the origin altitude (the sphere curves away), which
        // is precisely why altitude is not a plane coordinate.
        let f = frame();
        let flat = f.origin_point() + f.east * 30_000.0;
        let alt = f.altitude_of(flat);
        let sagitta = 30_000.0_f64.powi(2) / (2.0 * (f.body_radius_m + f.origin_altitude_m));
        assert!(alt > f.origin_altitude_m + 0.5 * sagitta);
    }

    #[test]
    fn vertical_constraints_evaluate() {
        assert!(VerticalConstraint::At(100.0).satisfied(101.0, 5.0));
        assert!(!VerticalConstraint::At(100.0).satisfied(120.0, 5.0));
        assert!(VerticalConstraint::AtOrAbove(100.0).satisfied(5_000.0, 0.0));
        assert!(!VerticalConstraint::AtOrAbove(100.0).satisfied(50.0, 0.0));
        assert!(VerticalConstraint::AtOrBelow(100.0).satisfied(50.0, 0.0));
        let w = VerticalConstraint::Window {
            lo_m: 100.0,
            hi_m: 200.0,
        };
        assert!(w.satisfied(150.0, 0.0));
        assert!(!w.satisfied(250.0, 0.0));
        assert_abs_diff_eq!(w.target_m(), 150.0);
    }

    #[test]
    fn a_ground_waypoint_has_no_vertical_constraint() {
        let w = Waypoint::fix(DVec3::X);
        assert!(w.vertical.is_none(), "rover/ship routes are lateral-only");
        assert!(
            w.with_vertical(VerticalConstraint::At(10.0))
                .vertical
                .is_some()
        );
    }
}
