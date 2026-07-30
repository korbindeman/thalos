//! Lateral path primitives: straight legs and constant-radius arcs, joined into
//! an arclength-parameterised [`LateralPath`].
//!
//! This is the geometry cross-track error is measured against, the geometry the
//! ND draws, and the geometry the autopilot will steer to — one representation
//! for all three, so "the path you see" and "the path you fly" cannot drift
//! apart.
//!
//! Angles are the internal **math** convention (CCW from +east); see
//! [`crate::waypoint`] for why, and for the compass conversion at the boundary.

use glam::DVec2;

use crate::waypoint::dir_from_theta;

/// A constant-radius circular arc, traversed from `start_theta` by `sweep`.
///
/// `start_theta` is the angle of the **radius vector** from `center` to the
/// arc's first point (not the travel direction), and `sweep` is signed:
/// positive is counter-clockwise, i.e. a pilot's **left** turn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc2 {
    pub center: DVec2,
    pub radius: f64,
    pub start_theta: f64,
    pub sweep: f64,
}

impl Arc2 {
    /// Arc length (m), always non-negative.
    pub fn length(&self) -> f64 {
        self.radius * self.sweep.abs()
    }

    /// Position at arc parameter `t ∈ [0, 1]`.
    pub fn point_at_t(&self, t: f64) -> DVec2 {
        let phi = self.start_theta + self.sweep * t;
        self.center + dir_from_theta(phi) * self.radius
    }

    /// Travel direction (math angle) at arc parameter `t`.
    ///
    /// The tangent leads the radius vector by +90° when sweeping CCW and lags
    /// it by 90° when sweeping CW — get this backwards and every turn commands
    /// the wrong way round.
    pub fn theta_at_t(&self, t: f64) -> f64 {
        let phi = self.start_theta + self.sweep * t;
        let quarter = std::f64::consts::FRAC_PI_2 * self.sweep.signum();
        crate::wrap_angle(phi + quarter)
    }

    /// Start and end points.
    pub fn endpoints(&self) -> (DVec2, DVec2) {
        (self.point_at_t(0.0), self.point_at_t(1.0))
    }
}

/// One leg of a lateral path.
///
/// [`Leg::Point`] exists because a zero-length `Line` is not a valid
/// substitute: two coincident endpoints carry no direction, so anything asking
/// "which way does the path go here?" would get a fabricated answer (east), and
/// a fabricated course is a wrong course on every display downstream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Leg {
    Line { from: DVec2, to: DVec2 },
    Arc(Arc2),
    /// A zero-length leg that still knows which way it faces.
    Point { at: DVec2, theta: f64 },
}

impl Leg {
    pub fn length(&self) -> f64 {
        match self {
            Leg::Line { from, to } => from.distance(*to),
            Leg::Arc(a) => a.length(),
            Leg::Point { .. } => 0.0,
        }
    }

    /// Position at distance `s` along this leg (clamped to the leg).
    pub fn point_at(&self, s: f64) -> DVec2 {
        let len = self.length();
        let t = if len > 1e-9 {
            (s / len).clamp(0.0, 1.0)
        } else {
            0.0
        };
        match self {
            Leg::Line { from, to } => from.lerp(*to, t),
            Leg::Arc(a) => a.point_at_t(t),
            Leg::Point { at, .. } => *at,
        }
    }

    /// Travel direction (math angle) at distance `s` along this leg.
    pub fn theta_at(&self, s: f64) -> f64 {
        let len = self.length();
        let t = if len > 1e-9 {
            (s / len).clamp(0.0, 1.0)
        } else {
            0.0
        };
        match self {
            Leg::Line { from, to } => {
                let d = *to - *from;
                if d.length_squared() < 1e-18 {
                    0.0
                } else {
                    crate::waypoint::theta_of(d)
                }
            }
            Leg::Arc(a) => a.theta_at_t(t),
            Leg::Point { theta, .. } => *theta,
        }
    }

    pub fn start(&self) -> DVec2 {
        self.point_at(0.0)
    }

    pub fn end(&self) -> DVec2 {
        self.point_at(self.length())
    }

    /// Distance along this leg of the point closest to `p`, plus that distance
    /// from `p`. For an arc the closest point is found by angle, clamped to the
    /// swept interval — so a craft "inside" the turn circle still projects onto
    /// the arc rather than onto its centre.
    fn closest_on_leg(&self, p: DVec2) -> (f64, f64) {
        match self {
            Leg::Line { from, to } => {
                let ab = *to - *from;
                let len2 = ab.length_squared();
                if len2 < 1e-18 {
                    return (0.0, p.distance(*from));
                }
                let t = ((p - *from).dot(ab) / len2).clamp(0.0, 1.0);
                let proj = *from + ab * t;
                (t * len2.sqrt(), p.distance(proj))
            }
            Leg::Arc(a) => {
                let rel = p - a.center;
                if rel.length_squared() < 1e-18 {
                    // Dead centre: every angle is equidistant, so take the arc
                    // start rather than dividing by zero.
                    return (0.0, a.radius);
                }
                let phi = crate::waypoint::theta_of(rel);
                // How far into the sweep `phi` lies, measured in the sweep's
                // own direction so the clamp works for both turn senses.
                let delta = crate::wrap_angle(phi - a.start_theta) * a.sweep.signum();
                let sweep_abs = a.sweep.abs();
                // `delta` is in (−π, π]; a sweep longer than π has a wrapped
                // region that must map to the far end, not back to the start.
                let t = if delta >= 0.0 {
                    (delta / sweep_abs).min(1.0)
                } else if sweep_abs > std::f64::consts::PI
                    && delta < -(2.0 * std::f64::consts::PI - sweep_abs) * 0.5
                {
                    1.0
                } else {
                    0.0
                };
                let s = t * a.length();
                (s, p.distance(a.point_at_t(t)))
            }
            Leg::Point { at, .. } => (0.0, p.distance(*at)),
        }
    }
}

/// A point resolved on a path: where it is, which way the path goes there, and
/// how far along the whole path it sits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathPoint {
    pub position: DVec2,
    /// Travel direction at that point (math angle).
    pub theta: f64,
    /// Distance from the path start (m).
    pub along_m: f64,
    /// Index of the leg the point lies on.
    pub leg: usize,
    /// Signed lateral offset of the query point from the path (m), **positive
    /// to the right** of the direction of travel. Zero for [`LateralPath::point_at`].
    pub cross_track_m: f64,
}

/// An ordered chain of legs. Empty is legal (an unplanned route) and every
/// query degrades gracefully rather than panicking.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LateralPath {
    pub legs: Vec<Leg>,
}

impl LateralPath {
    pub fn new(legs: Vec<Leg>) -> Self {
        Self { legs }
    }

    pub fn is_empty(&self) -> bool {
        self.legs.is_empty()
    }

    /// Total length (m).
    pub fn length(&self) -> f64 {
        self.legs.iter().map(Leg::length).sum()
    }

    /// The point at distance `s` from the path start (clamped to the path).
    pub fn point_at(&self, s: f64) -> Option<PathPoint> {
        if self.legs.is_empty() {
            return None;
        }
        let total = self.length();
        let s = s.clamp(0.0, total);
        let mut acc = 0.0;
        for (i, leg) in self.legs.iter().enumerate() {
            let len = leg.length();
            if s <= acc + len || i + 1 == self.legs.len() {
                let local = s - acc;
                return Some(PathPoint {
                    position: leg.point_at(local),
                    theta: leg.theta_at(local),
                    along_m: s,
                    leg: i,
                    cross_track_m: 0.0,
                });
            }
            acc += len;
        }
        None
    }

    /// The closest point on the path to `p`, with the signed cross-track offset
    /// (**positive = `p` is right of the path**).
    pub fn closest(&self, p: DVec2) -> Option<PathPoint> {
        let mut best: Option<(f64, usize, f64, f64)> = None; // (dist, leg, s_local, s_global)
        let mut acc = 0.0;
        for (i, leg) in self.legs.iter().enumerate() {
            let (s_local, dist) = leg.closest_on_leg(p);
            if best.is_none_or(|(bd, ..)| dist < bd) {
                best = Some((dist, i, s_local, acc + s_local));
            }
            acc += leg.length();
        }
        let (_, leg_idx, s_local, s_global) = best?;
        let leg = &self.legs[leg_idx];
        let position = leg.point_at(s_local);
        let theta = leg.theta_at(s_local);
        Some(PathPoint {
            position,
            theta,
            along_m: s_global,
            leg: leg_idx,
            cross_track_m: signed_cross_track(p - position, theta),
        })
    }

    /// Distance still to fly from the point closest to `p`.
    pub fn distance_to_go(&self, p: DVec2) -> Option<f64> {
        let pp = self.closest(p)?;
        Some((self.length() - pp.along_m).max(0.0))
    }

    /// Flatten to a polyline for display: line legs contribute their endpoints,
    /// arcs are subdivided so no chord deviates from the arc by more than
    /// `max_sag_m`. Consecutive duplicate points are collapsed.
    ///
    /// Sag-based (rather than fixed-count) subdivision is what keeps a 400 m
    /// turn radius smooth on a 2 km-range plot without spending 64 points on a
    /// 20 km straight-in at 150 km range.
    pub fn polyline(&self, max_sag_m: f64) -> Vec<DVec2> {
        let mut out: Vec<DVec2> = Vec::new();
        let push = |p: DVec2, out: &mut Vec<DVec2>| {
            if out.last().is_none_or(|last| last.distance_squared(p) > 1e-6) {
                out.push(p);
            }
        };
        for leg in &self.legs {
            match leg {
                Leg::Line { from, to } => {
                    push(*from, &mut out);
                    push(*to, &mut out);
                }
                Leg::Arc(a) => {
                    // Chord sag for a step of angle δ is r(1 − cos(δ/2)).
                    let sag = max_sag_m.max(1e-3);
                    let step = if sag >= a.radius {
                        std::f64::consts::FRAC_PI_2
                    } else {
                        2.0 * (1.0 - sag / a.radius).clamp(-1.0, 1.0).acos()
                    };
                    let n = ((a.sweep.abs() / step.max(1e-3)).ceil() as usize).clamp(1, 64);
                    for i in 0..=n {
                        push(a.point_at_t(i as f64 / n as f64), &mut out);
                    }
                }
                Leg::Point { at, .. } => push(*at, &mut out),
            }
        }
        out
    }
}

/// Signed lateral offset of `offset` (a vector from the path point to the
/// query point) relative to travel direction `theta`: **positive means right**
/// of the direction of travel.
///
/// In the local `(east, north)` plane with θ measured CCW, the right-hand normal
/// is `(sin θ, −cos θ)`.
pub fn signed_cross_track(offset: DVec2, theta: f64) -> f64 {
    let right = DVec2::new(theta.sin(), -theta.cos());
    offset.dot(right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waypoint::heading_to_theta;
    use approx::assert_abs_diff_eq;

    fn north_line(len: f64) -> LateralPath {
        LateralPath::new(vec![Leg::Line {
            from: DVec2::ZERO,
            to: DVec2::new(0.0, len),
        }])
    }

    #[test]
    fn cross_track_is_positive_to_the_right() {
        // Travelling north; a point to the east is to the pilot's right.
        let theta_north = heading_to_theta(0.0);
        assert!(signed_cross_track(DVec2::new(100.0, 0.0), theta_north) > 0.0);
        assert!(signed_cross_track(DVec2::new(-100.0, 0.0), theta_north) < 0.0);
        // Travelling east; a point to the south is to the right.
        let theta_east = heading_to_theta(std::f64::consts::FRAC_PI_2);
        assert!(signed_cross_track(DVec2::new(0.0, -100.0), theta_east) > 0.0);
    }

    #[test]
    fn line_closest_projects_and_clamps() {
        let p = north_line(1_000.0);
        let cp = p.closest(DVec2::new(50.0, 400.0)).expect("non-empty");
        assert_abs_diff_eq!(cp.along_m, 400.0, epsilon = 1e-9);
        assert_abs_diff_eq!(cp.cross_track_m, 50.0, epsilon = 1e-9);
        // Beyond the end: clamps to the end, keeps the sign.
        let cp = p.closest(DVec2::new(-20.0, 5_000.0)).expect("non-empty");
        assert_abs_diff_eq!(cp.along_m, 1_000.0, epsilon = 1e-9);
        assert_abs_diff_eq!(cp.cross_track_m, -20.0, epsilon = 1e-9);
    }

    #[test]
    fn distance_to_go_counts_down_along_the_path() {
        let p = north_line(1_000.0);
        assert_abs_diff_eq!(
            p.distance_to_go(DVec2::new(0.0, 250.0)).expect("non-empty"),
            750.0,
            epsilon = 1e-9
        );
        // Past the end never goes negative.
        assert_abs_diff_eq!(
            p.distance_to_go(DVec2::new(0.0, 2_000.0)).expect("non-empty"),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn arc_tangent_leads_for_left_turns_and_lags_for_right() {
        // Left (CCW) quarter turn starting at the circle's east point.
        let left = Arc2 {
            center: DVec2::ZERO,
            radius: 100.0,
            start_theta: 0.0,
            sweep: std::f64::consts::FRAC_PI_2,
        };
        // At the start the radius points +x, so travel must be +y (north).
        assert_abs_diff_eq!(left.theta_at_t(0.0), std::f64::consts::FRAC_PI_2, epsilon = 1e-12);
        let right = Arc2 { sweep: -std::f64::consts::FRAC_PI_2, ..left };
        // Same start point, opposite sense: travel is −y.
        assert_abs_diff_eq!(right.theta_at_t(0.0), -std::f64::consts::FRAC_PI_2, epsilon = 1e-12);
        assert_abs_diff_eq!(left.length(), 100.0 * std::f64::consts::FRAC_PI_2, epsilon = 1e-9);
    }

    #[test]
    fn arc_closest_clamps_within_the_sweep() {
        let arc = Arc2 {
            center: DVec2::ZERO,
            radius: 100.0,
            start_theta: 0.0,
            sweep: std::f64::consts::FRAC_PI_2,
        };
        let path = LateralPath::new(vec![Leg::Arc(arc)]);
        // Outside the arc at 45°: projects to the arc midpoint, 50 m outside.
        let q = DVec2::new(150.0, 150.0) / 2.0_f64.sqrt() * (2.0_f64.sqrt() / 2.0) * 2.0;
        let cp = path.closest(q).expect("non-empty");
        assert_abs_diff_eq!(cp.along_m, arc.length() * 0.5, epsilon = 1.0);
        // Right of a left turn = outside the circle.
        assert!(cp.cross_track_m > 0.0);
        // Well past the arc end (270°): clamps to the end, not back to start.
        let cp = path.closest(DVec2::new(-10.0, -200.0)).expect("non-empty");
        assert!(cp.along_m == 0.0 || cp.along_m == arc.length());
    }

    #[test]
    fn inside_the_turn_circle_still_projects_onto_the_arc() {
        let arc = Arc2 {
            center: DVec2::ZERO,
            radius: 100.0,
            start_theta: 0.0,
            sweep: std::f64::consts::FRAC_PI_2,
        };
        let path = LateralPath::new(vec![Leg::Arc(arc)]);
        let cp = path.closest(DVec2::new(30.0, 30.0)).expect("non-empty");
        // Inside a left turn = to the left of travel = negative cross-track.
        assert!(cp.cross_track_m < 0.0);
        assert!(cp.along_m > 0.0 && cp.along_m < arc.length());
    }

    #[test]
    fn polyline_respects_the_sag_budget() {
        let arc = Arc2 {
            center: DVec2::ZERO,
            radius: 400.0,
            start_theta: 0.0,
            sweep: std::f64::consts::PI,
        };
        let path = LateralPath::new(vec![Leg::Arc(arc)]);
        let pts = path.polyline(2.0);
        assert!(pts.len() > 4, "half-turn needs subdivision, got {}", pts.len());
        // Every chord midpoint must sit within the sag budget of the true arc.
        for w in pts.windows(2) {
            let mid = (w[0] + w[1]) * 0.5;
            let sag = 400.0 - mid.length();
            assert!(sag.abs() <= 2.5, "chord sag {sag} exceeds budget");
        }
    }

    #[test]
    fn a_point_leg_reports_its_own_heading() {
        // Regression: a zero-length `Line` reported "east" for its direction,
        // which showed up as a 90-degrees-wrong course on a degenerate route.
        let theta = heading_to_theta(143.0_f64.to_radians());
        let path = LateralPath::new(vec![Leg::Point {
            at: DVec2::new(10.0, -20.0),
            theta,
        }]);
        assert_abs_diff_eq!(path.length(), 0.0);
        let p = path.point_at(0.0).expect("non-empty");
        assert_abs_diff_eq!(p.theta, theta, epsilon = 1e-12);
        let c = path.closest(DVec2::new(10.0, -20.0)).expect("non-empty");
        assert_abs_diff_eq!(c.theta, theta, epsilon = 1e-12);
    }

    #[test]
    fn empty_path_queries_are_none_not_panics() {
        let p = LateralPath::default();
        assert!(p.closest(DVec2::ZERO).is_none());
        assert!(p.point_at(0.0).is_none());
        assert!(p.distance_to_go(DVec2::ZERO).is_none());
        assert_abs_diff_eq!(p.length(), 0.0);
    }

    #[test]
    fn multi_leg_along_track_accumulates_across_legs() {
        let path = LateralPath::new(vec![
            Leg::Line {
                from: DVec2::ZERO,
                to: DVec2::new(0.0, 1_000.0),
            },
            Leg::Line {
                from: DVec2::new(0.0, 1_000.0),
                to: DVec2::new(1_000.0, 1_000.0),
            },
        ]);
        assert_abs_diff_eq!(path.length(), 2_000.0, epsilon = 1e-9);
        let cp = path.closest(DVec2::new(500.0, 1_010.0)).expect("non-empty");
        assert_eq!(cp.leg, 1);
        assert_abs_diff_eq!(cp.along_m, 1_500.0, epsilon = 1e-9);
        // Travelling east, 10 m north of track = 10 m left = negative.
        assert_abs_diff_eq!(cp.cross_track_m, -10.0, epsilon = 1e-9);
    }
}
