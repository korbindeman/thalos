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

/// How far *back* along the route [`LateralPath::closest_from`] will look (m).
///
/// Small, because a craft flying a route does not un-fly it. The allowance is
/// only there to absorb the projection sliding backwards a little as the craft
/// swings around the outside of a turn.
const TRACK_WINDOW_BACK_M: f64 = 250.0;
/// How far *ahead* along the route [`LateralPath::closest_from`] will look (m).
///
/// A frame advances the projection by tens of metres at approach speed, so this
/// is generous by two orders of magnitude and never binds in normal flight. It
/// is still an order of magnitude below the leg separation that produced the
/// recorded 13 km snap, which is the gap it exists to close.
const TRACK_WINDOW_AHEAD_M: f64 = 1_000.0;

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
    Line {
        from: DVec2,
        to: DVec2,
    },
    Arc(Arc2),
    /// A zero-length leg that still knows which way it faces.
    Point {
        at: DVec2,
        theta: f64,
    },
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

    /// Signed curvature (1/m): **positive is a CCW turn**, i.e. a pilot's left,
    /// matching the θ convention everywhere else in this crate.
    ///
    /// This is what a path *follower* needs and a path *aimer* does not. Holding
    /// an arc costs a standing bank angle before any error correction; a law
    /// built only on heading error has to grow an error first in order to
    /// produce that bank, which is precisely a standing cross-track offset
    /// (INC-20260801T035551Z).
    pub fn curvature(&self) -> f64 {
        match self {
            Leg::Line { .. } | Leg::Point { .. } => 0.0,
            Leg::Arc(a) => {
                if a.radius > 1e-9 {
                    a.sweep.signum() / a.radius
                } else {
                    0.0
                }
            }
        }
    }

    /// The portion of this leg after `s` metres, or `None` when `s` is at or
    /// past its end. Used to build the remainder of a route behind a spliced-in
    /// rejoin.
    fn after(&self, s: f64) -> Option<Leg> {
        let len = self.length();
        if s <= 0.0 {
            return Some(*self);
        }
        if s >= len {
            return None;
        }
        match self {
            Leg::Line { to, .. } => Some(Leg::Line {
                from: self.point_at(s),
                to: *to,
            }),
            Leg::Arc(a) => {
                let t = s / len;
                Some(Leg::Arc(Arc2 {
                    center: a.center,
                    radius: a.radius,
                    start_theta: a.start_theta + a.sweep * t,
                    sweep: a.sweep * (1.0 - t),
                }))
            }
            Leg::Point { .. } => None,
        }
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
    /// Signed curvature of the path there (1/m, + = CCW/left). The feedforward
    /// term a follower needs — see [`Leg::curvature`].
    pub curvature: f64,
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
                    curvature: leg.curvature(),
                    cross_track_m: 0.0,
                });
            }
            acc += len;
        }
        None
    }

    /// The closest point on the path to `p`, with the signed cross-track offset
    /// (**positive = `p` is right of the path**).
    ///
    /// This searches the **whole** path, so on a route that passes near itself —
    /// a procedure turn, a teardrop, any bank-limited join that doubles back —
    /// the answer can jump legs between one call and the next. Use it to *seed* a
    /// track (or when there is genuinely no prior), and [`Self::closest_from`]
    /// to follow one. See that method for why the distinction is load-bearing.
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
            curvature: leg.curvature(),
            cross_track_m: signed_cross_track(p - position, theta),
        })
    }

    /// The closest point on the path to `p`, resolved **near where the craft
    /// already was** rather than globally.
    ///
    /// # Why this exists
    ///
    /// Along-track position is not just a number on a plot: `dtg` is derived
    /// from it, and `dtg` is the argument to the entire vertical profile, the
    /// speed gates, and the approach phase. A global nearest-point search has no
    /// obligation to be continuous, so on a path that doubles back the projection
    /// can hop from the inbound leg to the final leg the instant the final gets
    /// marginally nearer — and everything downstream teleports with it.
    ///
    /// That is not hypothetical. A recorded autoland went from 20.24 km to
    /// 7.17 km to go, and from 26 m to 148 m of altitude error, **in one frame**,
    /// with the craft unmoved and the plan untouched: the projection had snapped
    /// legs. The autopilot read that as "you are suddenly 148 m high on final"
    /// and dumped the nose.
    ///
    /// So the projection is windowed around the previous along-track position:
    /// generous forward (the craft is flying), tight backward (it is not), and
    /// both are orders of magnitude larger than a frame's travel while being far
    /// smaller than the leg separation that produces a snap. `hint_along_m` of
    /// `None` falls back to the global search — correct for seeding a fresh plan,
    /// and the only place a discontinuity is legitimate.
    ///
    /// The caller owns the hint, so this stays a pure function of its arguments
    /// (ADR-20260730T005746Z), exactly like the rejoin's capture hint.
    pub fn closest_from(&self, p: DVec2, hint_along_m: Option<f64>) -> Option<PathPoint> {
        let Some(hint) = hint_along_m else {
            return self.closest(p);
        };
        self.closest_within(p, hint - TRACK_WINDOW_BACK_M, hint + TRACK_WINDOW_AHEAD_M)
            // A hint that no longer lands on this path (a plan swapped underneath a
            // stale hint) leaves the window empty; seeding globally beats returning
            // nothing.
            .or_else(|| self.closest(p))
    }

    /// The closest point on the path to `p` whose along-track distance lies in
    /// `[lo_m, hi_m]`. `None` when that interval misses the path entirely.
    ///
    /// Exact, not approximate: distance from a point to a line segment or to a
    /// circular arc is unimodal in arc length, so clamping each leg's
    /// unconstrained minimiser into the window yields the constrained minimiser
    /// rather than merely a nearby one.
    pub fn closest_within(&self, p: DVec2, lo_m: f64, hi_m: f64) -> Option<PathPoint> {
        let total = self.length();
        let lo = lo_m.max(0.0);
        let hi = hi_m.min(total);
        if hi < lo {
            return None;
        }
        let mut best: Option<(f64, usize, f64, f64)> = None; // (dist, leg, s_local, s_global)
        let mut acc = 0.0;
        for (i, leg) in self.legs.iter().enumerate() {
            let len = leg.length();
            let leg_lo = (lo - acc).max(0.0);
            let leg_hi = (hi - acc).min(len);
            if leg_hi >= leg_lo {
                let (unconstrained, _) = leg.closest_on_leg(p);
                let s_local = unconstrained.clamp(leg_lo, leg_hi);
                let dist = p.distance(leg.point_at(s_local));
                if best.is_none_or(|(bd, ..)| dist < bd) {
                    best = Some((dist, i, s_local, acc + s_local));
                }
            }
            acc += len;
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
            curvature: leg.curvature(),
            cross_track_m: signed_cross_track(p - position, theta),
        })
    }

    /// The remainder of this path from `s_m` onward, as its own path.
    ///
    /// The straddling leg is truncated rather than dropped, so the result starts
    /// exactly at `s_m` and no length is invented or lost.
    pub fn tail_from(&self, s_m: f64) -> LateralPath {
        let mut legs = Vec::new();
        let mut acc = 0.0;
        for leg in &self.legs {
            let len = leg.length();
            if acc + len > s_m
                && let Some(part) = leg.after(s_m - acc)
            {
                legs.push(part);
            }
            acc += len;
        }
        LateralPath::new(legs)
    }

    /// The first `s_m` metres of this path, as its own path. The mirror of
    /// [`Self::tail_from`]; used to draw the committed-rejoin prefix of an
    /// active route in its own colour.
    pub fn head_to(&self, s_m: f64) -> LateralPath {
        let mut legs = Vec::new();
        let mut acc = 0.0;
        for leg in &self.legs {
            if acc >= s_m {
                break;
            }
            let len = leg.length();
            if acc + len <= s_m {
                legs.push(*leg);
            } else {
                let keep = s_m - acc;
                legs.push(match leg {
                    Leg::Line { from, .. } => Leg::Line {
                        from: *from,
                        to: leg.point_at(keep),
                    },
                    Leg::Arc(a) => Leg::Arc(Arc2 {
                        center: a.center,
                        radius: a.radius,
                        start_theta: a.start_theta,
                        sweep: a.sweep * (keep / len),
                    }),
                    Leg::Point { .. } => *leg,
                });
            }
            acc += len;
        }
        LateralPath::new(legs)
    }

    /// This path followed by `next`.
    ///
    /// Geometric continuity is the caller's business — [`Self::splice_rejoin`]
    /// is the one that guarantees it, because the rejoin was planned to meet the
    /// route tangentially at exactly the point it cuts.
    pub fn then(mut self, next: LateralPath) -> LateralPath {
        self.legs.extend(next.legs);
        self
    }

    /// The path the craft should actually fly: `rejoin`, then the remainder of
    /// this route from `capture_along_m`.
    ///
    /// # Why this is a splice and not a second path
    ///
    /// A rejoin used as a *steering cue* leaves the system with two answers to
    /// "where should I go" — the route that is drawn, and the rejoin that is
    /// flown — and they visibly disagree, because a rejoin leaves the route on
    /// purpose. Worse, recomputing it every frame makes its aim point slide
    /// forward as the craft advances, so the craft chases a receding target and
    /// never captures: a recorded approach flew 385 s at a steady 12° of bank
    /// while its distance from the route grew to 2.7 km (INC-20260801T035551Z).
    ///
    /// Splicing makes the rejoin *part of the route*. What is drawn is what is
    /// flown, cross-track and distance-to-go measure the same object the pilot
    /// is looking at, and the follower has a single continuous path with a
    /// defined curvature at every point.
    ///
    /// Distance-to-go is preserved for every point on the route portion: the
    /// splice only adds length ahead of the capture point, so the vertical
    /// profile and the speed gates keep their meaning.
    pub fn splice_rejoin(&self, rejoin: LateralPath, capture_along_m: f64) -> LateralPath {
        rejoin.then(self.tail_from(capture_along_m))
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
            if out
                .last()
                .is_none_or(|last| last.distance_squared(p) > 1e-6)
            {
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

    /// A teardrop: out along +east, a 180° left turn, back along −east 600 m
    /// north of the outbound leg. The two straights pass within 600 m of each
    /// other, which is what makes a global projection ambiguous.
    fn teardrop() -> LateralPath {
        LateralPath::new(vec![
            Leg::Line {
                from: DVec2::ZERO,
                to: DVec2::new(10_000.0, 0.0),
            },
            Leg::Arc(Arc2 {
                center: DVec2::new(10_000.0, 300.0),
                radius: 300.0,
                start_theta: -std::f64::consts::FRAC_PI_2,
                sweep: std::f64::consts::PI,
            }),
            Leg::Line {
                from: DVec2::new(10_000.0, 600.0),
                to: DVec2::new(0.0, 600.0),
            },
        ])
    }

    #[test]
    fn a_global_projection_snaps_legs_where_the_route_doubles_back() {
        // The defect this exists to pin: 350 m north of the outbound leg is
        // nearer the *return* leg, so the whole-path search answers with an
        // along-track distance ~15 km further on. Everything derived from
        // along-track — dtg, the vertical profile, the speed gates, the phase —
        // inherits that jump. Recorded live as 20.24 km -> 7.17 km in one frame.
        let path = teardrop();
        let outbound = path.closest(DVec2::new(5_000.0, 100.0)).expect("non-empty");
        let snapped = path.closest(DVec2::new(5_000.0, 350.0)).expect("non-empty");
        assert_eq!(outbound.leg, 0);
        assert_eq!(snapped.leg, 2, "the return leg is genuinely nearer");
        assert!(
            snapped.along_m - outbound.along_m > 10_000.0,
            "expected a large along-track jump, got {} -> {}",
            outbound.along_m,
            snapped.along_m
        );
    }

    #[test]
    fn a_hinted_projection_stays_on_the_leg_the_craft_is_flying() {
        let path = teardrop();
        // Same two query points, but following a track that is already at 5 km
        // along the outbound leg. The projection must not jump the gap.
        let outbound = path
            .closest_from(DVec2::new(5_000.0, 100.0), Some(5_000.0))
            .expect("non-empty");
        let next = path
            .closest_from(DVec2::new(5_000.0, 350.0), Some(outbound.along_m))
            .expect("non-empty");
        assert_eq!(next.leg, 0, "must stay on the outbound leg");
        assert_abs_diff_eq!(next.along_m, 5_000.0, epsilon = 1.0);
        // And it still reports the honest cross-track on that leg, so a craft
        // this far off course is visible rather than silently "on" the return.
        assert_abs_diff_eq!(next.cross_track_m, -350.0, epsilon = 1e-6);
    }

    #[test]
    fn a_hinted_projection_still_advances_normally() {
        let path = teardrop();
        // Frame-to-frame travel is tens of metres; the window must never bind.
        let mut hint = 0.0;
        for step in 1..=100 {
            let east = step as f64 * 80.0;
            let pp = path
                .closest_from(DVec2::new(east, 0.0), Some(hint))
                .expect("non-empty");
            assert_abs_diff_eq!(pp.along_m, east, epsilon = 1e-6);
            assert!(pp.along_m >= hint, "along-track went backwards");
            hint = pp.along_m;
        }
        // Continuing around the turn and onto the return leg, progress stays
        // continuous — the window never binds on a craft actually flying the
        // route, only on a projection trying to jump across it.
        for step in 0..400 {
            let s = 8_000.0 + step as f64 * 10.0;
            let target = path.point_at(s).expect("on path");
            let pp = path
                .closest_from(target.position, Some(hint))
                .expect("non-empty");
            assert!(
                (pp.along_m - s).abs() < 5.0,
                "at s={s} the hinted projection answered {}",
                pp.along_m
            );
            hint = pp.along_m;
        }
    }

    #[test]
    fn no_hint_falls_back_to_the_global_search() {
        let path = teardrop();
        let p = DVec2::new(5_000.0, 350.0);
        assert_eq!(
            path.closest_from(p, None).expect("non-empty"),
            path.closest(p).expect("non-empty"),
        );
        // A hint stranded past the end of a shorter path also degrades to the
        // global answer rather than to nothing.
        let short = north_line(1_000.0);
        assert!(
            short
                .closest_from(DVec2::new(0.0, 500.0), Some(9e9))
                .is_some()
        );
    }

    #[test]
    fn closest_within_is_exact_at_the_window_edge() {
        // The window cuts a line leg mid-span: the constrained answer must be
        // the window edge, not the leg's unconstrained minimiser and not a
        // rejection of the leg.
        let p = north_line(1_000.0);
        let cp = p
            .closest_within(DVec2::new(10.0, 800.0), 0.0, 400.0)
            .expect("window overlaps the path");
        assert_abs_diff_eq!(cp.along_m, 400.0, epsilon = 1e-9);
        assert!(p.closest_within(DVec2::ZERO, 2_000.0, 3_000.0).is_none());
    }

    #[test]
    fn head_and_tail_partition_the_path_exactly() {
        let path = teardrop();
        let total = path.length();
        for cut in [0.0, 1.0, 5_000.0, 9_999.0, 10_000.0, total * 0.75, total] {
            let head = path.head_to(cut);
            let tail = path.tail_from(cut);
            assert_abs_diff_eq!(head.length() + tail.length(), total, epsilon = 1e-6);
            assert_abs_diff_eq!(head.length(), cut.min(total), epsilon = 1e-6);
            // The two meet where they were cut, with no gap and no jump.
            if cut > 0.0 && cut < total {
                let meet_head = head.point_at(head.length()).expect("non-empty").position;
                let meet_tail = tail.point_at(0.0).expect("non-empty").position;
                assert!(
                    meet_head.distance(meet_tail) < 1e-6,
                    "cut at {cut} left a {} m gap",
                    meet_head.distance(meet_tail)
                );
            }
        }
    }

    #[test]
    fn a_spliced_rejoin_adds_its_length_and_nothing_else() {
        // The invariant the vertical profile depends on: splicing changes
        // along-track distances but leaves every distance-to-go on the route
        // portion untouched, because the added length is all *ahead* of them.
        let route = teardrop();
        let capture = 12_000.0;
        let rejoin = LateralPath::new(vec![Leg::Line {
            from: DVec2::new(-2_000.0, -2_000.0),
            to: route.point_at(capture).expect("on path").position,
        }]);
        let rejoin_len = rejoin.length();
        let spliced = route.splice_rejoin(rejoin, capture);

        assert_abs_diff_eq!(
            spliced.length(),
            rejoin_len + route.length() - capture,
            epsilon = 1e-6
        );
        // A point 3 km from the end has the same distance-to-go on both paths.
        let dtg = 3_000.0;
        let on_route = route.point_at(route.length() - dtg).expect("on path");
        let on_spliced = spliced.point_at(spliced.length() - dtg).expect("on path");
        assert!(
            on_route.position.distance(on_spliced.position) < 1e-6,
            "the same distance-to-go must be the same place"
        );
    }

    #[test]
    fn curvature_is_zero_on_straights_and_signed_on_arcs() {
        let path = teardrop();
        assert_abs_diff_eq!(
            path.point_at(5_000.0).expect("on path").curvature,
            0.0,
            epsilon = 1e-12
        );
        // Mid-turn: a CCW (left) sweep is positive curvature of 1/radius.
        let mid = path.point_at(10_000.0 + 150.0 * std::f64::consts::PI / 2.0);
        assert_abs_diff_eq!(mid.expect("on path").curvature, 1.0 / 300.0, epsilon = 1e-9);
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
            p.distance_to_go(DVec2::new(0.0, 2_000.0))
                .expect("non-empty"),
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
        assert_abs_diff_eq!(
            left.theta_at_t(0.0),
            std::f64::consts::FRAC_PI_2,
            epsilon = 1e-12
        );
        let right = Arc2 {
            sweep: -std::f64::consts::FRAC_PI_2,
            ..left
        };
        // Same start point, opposite sense: travel is −y.
        assert_abs_diff_eq!(
            right.theta_at_t(0.0),
            -std::f64::consts::FRAC_PI_2,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            left.length(),
            100.0 * std::f64::consts::FRAC_PI_2,
            epsilon = 1e-9
        );
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
        assert!(
            pts.len() > 4,
            "half-turn needs subdivision, got {}",
            pts.len()
        );
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
