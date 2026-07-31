//! The shortest **bank-limited** path between two poses — the geometry that
//! turns "fly me to that runway" into something an aircraft can actually track.
//!
//! A craft has a minimum turn radius (`v² / (g·tan φ_max)`), so joining the
//! craft's current position-and-heading to the start of a final approach is
//! exactly the classical Dubins problem: two circular arcs of that radius joined
//! by a straight tangent line, `C S C`.
//!
//! # Only the four CSC words, on purpose
//!
//! Dubins' full solution set also contains the three-arc `CCC` words (`RLR`,
//! `LRL`), which can be *shorter* when the two poses are closer together than
//! ~4 turn radii. They are deliberately not implemented, for two reasons:
//!
//! - A `CCC` solution is a tight S of back-to-back full-bank spirals. For a
//!   landing approach that is the wrong shape even when it is the shortest — you
//!   do not want the autopilot (or a pilot following the drawn line) hauling the
//!   craft through it a few hundred metres up.
//! - **A `CSC` solution always exists**, so dropping `CCC` costs coverage
//!   nowhere. Expanding LSL's discriminant gives
//!   `2 + d² − 2cos(α−β) + 2d(sin α − sin β) = (d + sin α − sin β)² + (cos β − cos α)²`
//!   — a sum of squares, hence never negative (and RSR's is the mirror image).
//!   So [`plan_dubins`] can only fail on a non-finite or non-positive radius,
//!   never on geometry. `only_sum_of_squares_words_are_ever_needed` pins that
//!   property.
//!
//! Equations follow Shkel & Lumelsky's normalised formulation; every word is
//! verified in tests by reconstructing the path and checking it lands on the
//! goal pose.

use glam::DVec2;

use crate::path::{Arc2, LateralPath, Leg};
use crate::waypoint::dir_from_theta;

/// A planar pose: position in the local `(east, north)` plane plus a travel
/// direction as a **math** angle (CCW from east — see [`crate::waypoint`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose2 {
    pub position: DVec2,
    pub theta: f64,
}

impl Pose2 {
    pub fn new(position: DVec2, theta: f64) -> Self {
        Self { position, theta }
    }

    /// Unit travel direction.
    pub fn direction(&self) -> DVec2 {
        dir_from_theta(self.theta)
    }
}

/// Which turn-straight-turn word a solution used. `L` = left (CCW), `R` =
/// right (CW), `S` = straight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DubinsWord {
    Lsl,
    Rsr,
    Lsr,
    Rsl,
}

impl DubinsWord {
    /// Turn senses of the two arcs: `+1` = left/CCW, `−1` = right/CW.
    fn senses(self) -> (f64, f64) {
        match self {
            Self::Lsl => (1.0, 1.0),
            Self::Rsr => (-1.0, -1.0),
            Self::Lsr => (1.0, -1.0),
            Self::Rsl => (-1.0, 1.0),
        }
    }
}

/// A planned bank-limited path.
#[derive(Debug, Clone, PartialEq)]
pub struct DubinsPath {
    pub word: DubinsWord,
    /// Turn radius used (m).
    pub radius_m: f64,
    /// The legs, ready to fly or draw. A zero-length arc or straight is dropped,
    /// so a straight-in solution really is one line leg.
    pub path: LateralPath,
    /// Total length (m).
    pub length_m: f64,
}

/// Plan the shortest `CSC` path from `start` to `goal` with turn radius
/// `radius_m`.
///
/// Returns `None` only for a non-finite or non-positive radius — never for
/// geometry, see the module docs.
pub fn plan_dubins(start: Pose2, goal: Pose2, radius_m: f64) -> Option<DubinsPath> {
    if !radius_m.is_finite() || radius_m <= 0.0 {
        return None;
    }
    let (word, t, p, q) = best_word(start, goal, radius_m)?;
    let mut legs = Vec::with_capacity(3);
    let (s1, s2) = word.senses();
    let mut cursor = start;
    if t > 1e-9 {
        let arc = arc_from(cursor, radius_m, s1, t);
        cursor = Pose2::new(
            arc.point_at_t(1.0),
            crate::wrap_angle(cursor.theta + s1 * t),
        );
        legs.push(Leg::Arc(arc));
    }
    let straight = p * radius_m;
    if straight > 1e-9 {
        let to = cursor.position + cursor.direction() * straight;
        legs.push(Leg::Line {
            from: cursor.position,
            to,
        });
        cursor = Pose2::new(to, cursor.theta);
    }
    if q > 1e-9 {
        legs.push(Leg::Arc(arc_from(cursor, radius_m, s2, q)));
    }
    if legs.is_empty() {
        // Start and goal coincide with the same heading. This must still be a
        // well-formed path so downstream `point_at` / `closest` resolve — and it
        // must be a `Leg::Point`, not a zero-length `Leg::Line`: a line with
        // coincident endpoints has no direction to report, and reporting a
        // fabricated one puts a wrong course on the display.
        legs.push(Leg::Point {
            at: start.position,
            theta: start.theta,
        });
    }
    let path = LateralPath::new(legs);
    Some(DubinsPath {
        word,
        radius_m,
        length_m: path.length(),
        path,
    })
}

/// The arc leaving `pose` with turn sense `sense` (+1 left) through `sweep`
/// radians of turn.
fn arc_from(pose: Pose2, radius_m: f64, sense: f64, sweep: f64) -> Arc2 {
    // Turn centre is one radius to the left (sense +1) or right (−1) of travel.
    let to_center = DVec2::new(-pose.theta.sin(), pose.theta.cos()) * sense;
    let center = pose.position + to_center * radius_m;
    Arc2 {
        center,
        radius: radius_m,
        // Radius vector at the start lags travel by 90° for a left turn and
        // leads it by 90° for a right turn (the inverse of `Arc2::theta_at_t`).
        start_theta: pose.theta - sense * std::f64::consts::FRAC_PI_2,
        sweep: sense * sweep,
    }
}

/// Shortest valid word and its normalised `(t, p, q)` segment parameters
/// (`t`/`q` in radians of turn, `p` in units of the radius).
fn best_word(start: Pose2, goal: Pose2, radius_m: f64) -> Option<(DubinsWord, f64, f64, f64)> {
    let delta = goal.position - start.position;
    let dist = delta.length();
    let d = dist / radius_m;
    // Direction of the connecting line; for coincident positions any reference
    // direction works and the words degenerate to pure turns.
    let theta = if dist > 1e-12 {
        delta.y.atan2(delta.x)
    } else {
        start.theta
    };
    let alpha = crate::wrap_positive(start.theta - theta);
    let beta = crate::wrap_positive(goal.theta - theta);

    let (sin_a, cos_a) = alpha.sin_cos();
    let (sin_b, cos_b) = beta.sin_cos();
    let cos_ab = (alpha - beta).cos();
    let m2 = crate::wrap_positive;

    let mut best: Option<(DubinsWord, f64, f64, f64)> = None;
    let mut consider = |word: DubinsWord, t: f64, p: f64, q: f64| {
        if !(t.is_finite() && p.is_finite() && q.is_finite()) {
            return;
        }
        let cost = t + p + q;
        if best.is_none_or(|(_, bt, bp, bq)| cost < bt + bp + bq) {
            best = Some((word, t, p, q));
        }
    };

    // LSL
    let p_sq = 2.0 + d * d - 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b);
    if p_sq >= 0.0 {
        let tmp = (cos_b - cos_a).atan2(d + sin_a - sin_b);
        consider(
            DubinsWord::Lsl,
            m2(-alpha + tmp),
            p_sq.sqrt(),
            m2(beta - tmp),
        );
    }
    // RSR
    let p_sq = 2.0 + d * d - 2.0 * cos_ab + 2.0 * d * (sin_b - sin_a);
    if p_sq >= 0.0 {
        let tmp = (cos_a - cos_b).atan2(d - sin_a + sin_b);
        consider(
            DubinsWord::Rsr,
            m2(alpha - tmp),
            p_sq.sqrt(),
            m2(-beta + tmp),
        );
    }
    // LSR
    let p_sq = -2.0 + d * d + 2.0 * cos_ab + 2.0 * d * (sin_a + sin_b);
    if p_sq >= 0.0 {
        let p = p_sq.sqrt();
        let tmp = (-cos_a - cos_b).atan2(d + sin_a + sin_b) - (-2.0_f64).atan2(p);
        consider(DubinsWord::Lsr, m2(-alpha + tmp), p, m2(-beta + tmp));
    }
    // RSL
    let p_sq = d * d - 2.0 + 2.0 * cos_ab - 2.0 * d * (sin_a + sin_b);
    if p_sq >= 0.0 {
        let p = p_sq.sqrt();
        let tmp = (cos_a + cos_b).atan2(d - sin_a - sin_b) - 2.0_f64.atan2(p);
        consider(DubinsWord::Rsl, m2(alpha - tmp), p, m2(beta - tmp));
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Fly the planned path to its end and report the pose reached.
    fn end_pose(path: &DubinsPath) -> Pose2 {
        let p = path
            .path
            .point_at(path.path.length())
            .expect("planned path is non-empty");
        Pose2::new(p.position, p.theta)
    }

    fn assert_reaches(start: Pose2, goal: Pose2, radius: f64) -> DubinsPath {
        let plan = plan_dubins(start, goal, radius).expect("planner never fails on geometry");
        let end = end_pose(&plan);
        assert_abs_diff_eq!(end.position.x, goal.position.x, epsilon = 1e-6);
        assert_abs_diff_eq!(end.position.y, goal.position.y, epsilon = 1e-6);
        assert_abs_diff_eq!(
            crate::wrap_angle(end.theta - goal.theta),
            0.0,
            epsilon = 1e-6
        );
        // The reported length must be the path's real length, or every
        // distance-to-go readout downstream is wrong.
        assert_abs_diff_eq!(plan.length_m, plan.path.length(), epsilon = 1e-9);
        plan
    }

    #[test]
    fn straight_ahead_is_a_single_line() {
        let plan = assert_reaches(
            Pose2::new(DVec2::ZERO, 0.0),
            Pose2::new(DVec2::new(10_000.0, 0.0), 0.0),
            1_000.0,
        );
        assert_abs_diff_eq!(plan.length_m, 10_000.0, epsilon = 1e-6);
        assert_eq!(plan.path.legs.len(), 1, "no spurious zero-length arcs");
    }

    #[test]
    fn a_right_offset_goal_turns_right_first() {
        // Goal ahead and to the right, same heading: an S-turn starting right.
        let plan = assert_reaches(
            Pose2::new(DVec2::ZERO, std::f64::consts::FRAC_PI_2), // heading north
            Pose2::new(DVec2::new(3_000.0, 12_000.0), std::f64::consts::FRAC_PI_2),
            1_200.0,
        );
        assert!(
            matches!(plan.word, DubinsWord::Rsl),
            "expected right-then-left S, got {:?}",
            plan.word
        );
    }

    #[test]
    fn a_left_offset_goal_turns_left_first() {
        let plan = assert_reaches(
            Pose2::new(DVec2::ZERO, std::f64::consts::FRAC_PI_2),
            Pose2::new(DVec2::new(-3_000.0, 12_000.0), std::f64::consts::FRAC_PI_2),
            1_200.0,
        );
        assert!(
            matches!(plan.word, DubinsWord::Lsr),
            "expected left-then-right S, got {:?}",
            plan.word
        );
    }

    #[test]
    fn reaches_a_goal_behind_the_start() {
        // The hard case an approach planner actually hits: overflying the field
        // and having to come back around.
        assert_reaches(
            Pose2::new(DVec2::ZERO, 0.0),
            Pose2::new(DVec2::new(-8_000.0, 0.0), 0.0),
            1_500.0,
        );
    }

    #[test]
    fn reaches_goals_across_a_deterministic_pose_sweep() {
        let radius = 900.0;
        // A deterministic lattice of relative positions and heading pairs,
        // including inside-the-turn-circle cases (the CSC-impossible region).
        for &gx in &[-9_000.0, -2_000.0, -400.0, 0.0, 700.0, 5_000.0, 20_000.0] {
            for &gy in &[-6_000.0, -1_100.0, 0.0, 300.0, 3_000.0] {
                for gh_deg in [0, 37, 90, 143, 200, 270, 315] {
                    for sh_deg in [0, 90, 210] {
                        let start = Pose2::new(DVec2::ZERO, (sh_deg as f64).to_radians());
                        let goal = Pose2::new(DVec2::new(gx, gy), (gh_deg as f64).to_radians());
                        assert_reaches(start, goal, radius);
                    }
                }
            }
        }
    }

    #[test]
    fn coincident_poses_with_opposite_headings_fly_a_u_turn() {
        // d = 0 with a 180° heading change is the tightest case there is: the
        // answer is turn 90°, cross over by two radii, turn 90° back.
        let r = 800.0;
        let plan = assert_reaches(
            Pose2::new(DVec2::ZERO, 0.0),
            Pose2::new(DVec2::ZERO, std::f64::consts::PI),
            r,
        );
        // Returning to the same point on the reciprocal heading cannot be done
        // in less than a full circle's worth of turning at a bounded radius, and
        // the verified reconstruction above already proves it arrives.
        assert!(
            plan.length_m > 2.0 * std::f64::consts::PI * r,
            "a reversal in place needs more than one circle of turning, got {}",
            plan.length_m
        );
        assert!(plan.length_m < 4.0 * std::f64::consts::PI * r);
    }

    #[test]
    fn identical_poses_give_a_well_formed_zero_length_path() {
        let pose = Pose2::new(DVec2::new(120.0, -40.0), 1.1);
        let plan = plan_dubins(pose, pose, 500.0).expect("solvable");
        assert_abs_diff_eq!(plan.length_m, 0.0, epsilon = 1e-9);
        assert!(!plan.path.is_empty(), "queries must still resolve");
        assert!(plan.path.closest(DVec2::ZERO).is_some());
    }

    #[test]
    fn only_sum_of_squares_words_are_ever_needed() {
        // Pins the module's existence argument: LSL and RSR have sum-of-squares
        // discriminants, so at least one word is always available and the
        // planner never fails on geometry. If someone "optimises" `best_word`
        // and breaks that, this fires.
        let radius = 750.0;
        for &gx in &[-5_000.0, -750.0, 0.0, 10.0, 900.0, 40_000.0] {
            for &gy in &[-3_000.0, -100.0, 0.0, 250.0, 8_000.0] {
                for gh in 0..12 {
                    for sh in 0..7 {
                        let start = Pose2::new(DVec2::ZERO, sh as f64 * 0.9);
                        let goal = Pose2::new(DVec2::new(gx, gy), gh as f64 * 0.52);
                        assert!(
                            plan_dubins(start, goal, radius).is_some(),
                            "no solution at goal ({gx}, {gy}) heading {gh}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn arc_senses_match_the_word() {
        // LSL: both arcs must sweep CCW (positive).
        let plan = plan_dubins(
            Pose2::new(DVec2::ZERO, 0.0),
            Pose2::new(DVec2::new(2_000.0, 6_000.0), std::f64::consts::PI),
            700.0,
        )
        .expect("solvable");
        for leg in &plan.path.legs {
            if let Leg::Arc(a) = leg {
                let (s1, s2) = plan.word.senses();
                assert!(
                    a.sweep.signum() == s1 || a.sweep.signum() == s2,
                    "arc sweep {} matches neither sense of {:?}",
                    a.sweep,
                    plan.word
                );
            }
        }
    }

    #[test]
    fn rejects_a_degenerate_radius() {
        let s = Pose2::new(DVec2::ZERO, 0.0);
        let g = Pose2::new(DVec2::new(1_000.0, 0.0), 0.0);
        assert!(plan_dubins(s, g, 0.0).is_none());
        assert!(plan_dubins(s, g, -100.0).is_none());
        assert!(plan_dubins(s, g, f64::NAN).is_none());
    }

    #[test]
    fn a_bigger_radius_never_yields_a_shorter_path() {
        let s = Pose2::new(DVec2::ZERO, std::f64::consts::FRAC_PI_2);
        let g = Pose2::new(DVec2::new(4_000.0, 9_000.0), 0.0);
        let tight = plan_dubins(s, g, 600.0).expect("solvable").length_m;
        let wide = plan_dubins(s, g, 2_400.0).expect("solvable").length_m;
        assert!(
            wide >= tight - 1e-6,
            "wide radius {wide} beat tight {tight}"
        );
    }
}
