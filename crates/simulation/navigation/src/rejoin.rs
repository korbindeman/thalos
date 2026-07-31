//! Getting back onto the route: a **flyable** path from wherever the craft is
//! to a point on the route it can meet *tangentially*.
//!
//! # Why this exists
//!
//! The naive answer to "you are off course" is to steer at the nearest point on
//! the route. That arrives perpendicular — a corner, not a join — and an
//! aircraft cannot fly a corner. The classic softened version (aim at a point a
//! fixed lookahead ahead) converges smoothly but is a *heuristic*: nothing in it
//! knows the craft's turn radius, so at low speed it dawdles and at high speed it
//! demands turns the craft cannot make.
//!
//! So the rejoin is planned, with the same bank-limited machinery the route
//! itself was planned with: search forward along the route for the earliest
//! point the craft **can** meet without an absurd detour, and join it with a
//! Dubins path ([`crate::dubins`]). "The earliest point we can meet" is the
//! whole idea — it is what makes a small deviation produce a gentle convergence
//! and a large one produce a proper intercept, with no gain to tune between the
//! two cases.
//!
//! # This is guidance, not a re-plan
//!
//! A rejoin **never** modifies the route. The route is the commitment; the
//! rejoin is advice about how to get back to it, recomputed as the craft moves.
//! Keeping them separate is what lets the plan stay frozen on final (see
//! `docs/gameplay/navigation.md` § Re-plan policy) while the steering cue still
//! responds to being blown off the centreline.
//!
//! # Route-relative, not destination-relative
//!
//! Everything here works against the [`LateralPath`], so it is identical for an
//! approach's base leg, its final, or a future waypoint route. The runway is
//! only where today's path happens to end.

use crate::dubins::{Pose2, plan_dubins};
use crate::path::LateralPath;

/// Tuning for [`plan_rejoin`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RejoinParams {
    /// Minimum turn radius (m) — the same bank-limited radius the route was
    /// planned with, so a rejoin is never tighter than the route itself.
    pub turn_radius_m: f64,
    /// Never capture closer than this far along the route from the craft's
    /// current position (m). Stops the cue from aiming at a point the craft is
    /// about to pass.
    pub min_capture_m: f64,
    /// How far ahead to search for a capture point (m).
    pub search_span_m: f64,
    /// Search granularity (m).
    pub step_m: f64,
    /// The comfort test: how much turning the join may demand **beyond what the
    /// craft's current heading error already forces** (rad).
    ///
    /// Total swept angle alone is the wrong measure — a craft pointing 180° from
    /// the route must turn 180° no matter which point it aims at, and that is
    /// not the capture point's fault. Path length is also the wrong measure: a
    /// hard 90°-out-and-back S can have a perfectly respectable length ratio
    /// while being exactly the "corner" this module exists to avoid. What
    /// separates a gentle join from a wrenching one is the turning it adds on
    /// top of the unavoidable heading change, so that is what is bounded.
    pub max_excess_turn_rad: f64,
}

impl RejoinParams {
    /// Defaults derived from a turn radius: capture no nearer than two radii,
    /// search twenty radii ahead, step a quarter radius.
    pub fn for_radius(turn_radius_m: f64) -> Self {
        let r = turn_radius_m.max(1.0);
        Self {
            turn_radius_m: r,
            min_capture_m: 2.0 * r,
            search_span_m: 30.0 * r,
            step_m: 0.25 * r,
            // ~70°: enough to fly a normal intercept, tight enough that a
            // capture needing an S-turn gets pushed further along the route.
            max_excess_turn_rad: 1.2,
        }
    }
}

/// A planned rejoin.
#[derive(Debug, Clone, PartialEq)]
pub struct Rejoin {
    /// The flyable path from the craft's pose to the capture point.
    pub path: LateralPath,
    /// Along-route distance of the capture point (m).
    pub capture_along_m: f64,
    /// Length of the rejoin path (m).
    pub length_m: f64,
    /// Straight-line distance from the craft to the capture point (m).
    pub direct_m: f64,
    /// Total swept angle of the rejoin's arcs (rad, always ≥ 0).
    pub total_turn_rad: f64,
    /// Turning demanded beyond the unavoidable heading change (rad) — the
    /// comfort measure, see [`RejoinParams::max_excess_turn_rad`].
    pub excess_turn_rad: f64,
}

impl Rejoin {
    /// How much longer the flyable rejoin is than flying straight at the capture
    /// point — 1.0 means the craft is already pointing at it. Reported for
    /// diagnostics; the comfort decision is [`Self::excess_turn_rad`].
    pub fn detour_ratio(&self) -> f64 {
        if self.direct_m > 1.0 {
            self.length_m / self.direct_m
        } else {
            1.0
        }
    }

    /// The point to aim at `lookahead_m` along the rejoin, and the direction of
    /// travel there. This is what a steering cue points at: near the end of the
    /// rejoin it *is* the route's own course, so the cue converges to "fly the
    /// route" without a mode change.
    pub fn aim(&self, lookahead_m: f64) -> Option<(glam::DVec2, f64)> {
        let p = self
            .path
            .point_at(lookahead_m.clamp(0.0, self.path.length()))?;
        Some((p.position, p.theta))
    }
}

/// Plan a rejoin from `pose` onto `route`.
///
/// `from_along_m` is where the craft currently projects onto the route (the
/// caller already has this from the closest-point query). `hint_along_m` is the
/// capture point chosen last frame: if it is still feasible it is kept, which is
/// what stops the cue from twitching as the search flips between neighbouring
/// candidates. Passing `None` simply searches from scratch.
///
/// Returns `None` only when the route is empty or the radius is degenerate.
pub fn plan_rejoin(
    route: &LateralPath,
    pose: Pose2,
    from_along_m: f64,
    params: &RejoinParams,
    hint_along_m: Option<f64>,
) -> Option<Rejoin> {
    if route.is_empty() || !params.turn_radius_m.is_finite() || params.turn_radius_m <= 0.0 {
        return None;
    }
    let total = route.length();
    let first = (from_along_m + params.min_capture_m).min(total);

    // A hint that is still ahead of the craft and still comfortable wins, so the
    // capture point stays put while the craft flies toward it.
    // A held capture point is kept until the craft is nearly on top of it —
    // *not* until it re-enters the `min_capture` window, which would drop the
    // hint on almost every frame as the craft closes and put the twitch back.
    let hold_floor = from_along_m + params.min_capture_m * 0.5;
    if let Some(hint) = hint_along_m
        && hint >= hold_floor
        && hint <= total
        && let Some(candidate) = try_capture(route, pose, hint, params)
        && candidate.excess_turn_rad <= params.max_excess_turn_rad
    {
        return Some(candidate);
    }

    let mut best: Option<Rejoin> = None;
    let mut along = first;
    let last = (from_along_m + params.min_capture_m + params.search_span_m).min(total);
    let step = params.step_m.max(1.0);
    while along <= last {
        if let Some(candidate) = try_capture(route, pose, along, params) {
            if candidate.excess_turn_rad <= params.max_excess_turn_rad {
                // The *earliest* comfortable capture — "the next point we can
                // meet" — not the shortest path, which would happily fly past
                // the route and turn back onto it.
                return Some(candidate);
            }
            // Keep the least-bad option in case nothing is comfortable.
            if best
                .as_ref()
                .is_none_or(|b| candidate.excess_turn_rad < b.excess_turn_rad)
            {
                best = Some(candidate);
            }
        }
        along += step;
    }
    // Nothing comfortable: the end of the route is the last thing worth aiming
    // at, and the least-bad candidate is still a flyable path.
    best.or_else(|| try_capture(route, pose, total, params))
}

/// The flyable path onto the route at `along_m`, if the route has a point there.
fn try_capture(
    route: &LateralPath,
    pose: Pose2,
    along_m: f64,
    params: &RejoinParams,
) -> Option<Rejoin> {
    let target = route.point_at(along_m)?;
    let dubins = plan_dubins(
        pose,
        Pose2::new(target.position, target.theta),
        params.turn_radius_m,
    )?;
    let direct_m = pose.position.distance(target.position);
    let total_turn_rad: f64 = dubins
        .path
        .legs
        .iter()
        .filter_map(|leg| match leg {
            crate::path::Leg::Arc(arc) => Some(arc.sweep.abs()),
            _ => None,
        })
        .sum();
    let forced_turn_rad = crate::wrap_angle(target.theta - pose.theta).abs();
    Some(Rejoin {
        capture_along_m: along_m,
        length_m: dubins.length_m,
        path: dubins.path,
        direct_m,
        total_turn_rad,
        excess_turn_rad: (total_turn_rad - forced_turn_rad).max(0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path::Leg;
    use approx::assert_abs_diff_eq;
    use glam::DVec2;

    /// A 40 km route running due north (math θ = π/2) from the origin.
    fn north_route() -> LateralPath {
        LateralPath::new(vec![Leg::Line {
            from: DVec2::ZERO,
            to: DVec2::new(0.0, 40_000.0),
        }])
    }

    fn params() -> RejoinParams {
        RejoinParams::for_radius(1_000.0)
    }

    #[test]
    fn a_hard_s_turn_is_rejected_in_favour_of_reaching_further_out() {
        // 6 km off course with only 2 km of route ahead of the minimum capture
        // is a 90-degrees-out-and-back S. Its *length ratio* is respectable,
        // which is exactly why length is not the test — the excess-turn bound
        // is what pushes the capture point out to where the join is gentle.
        let route = north_route();
        let p = params();
        let pose = Pose2::new(DVec2::new(6_000.0, 5_000.0), std::f64::consts::FRAC_PI_2);
        let tight = try_capture(&route, pose, 5_000.0 + p.min_capture_m, &p).expect("capture");
        assert!(
            tight.excess_turn_rad > p.max_excess_turn_rad,
            "a 6 km S-turn should not read as comfortable: {:.0} deg excess",
            tight.excess_turn_rad.to_degrees()
        );
        let chosen = plan_rejoin(&route, pose, 5_000.0, &p, None).expect("plannable");
        assert!(chosen.capture_along_m > 5_000.0 + p.min_capture_m);
        assert!(chosen.excess_turn_rad <= p.max_excess_turn_rad);
    }

    #[test]
    fn pointing_away_is_not_charged_for_the_turn_it_must_make() {
        // 180 degrees off: the craft has to turn 180 degrees whichever point it
        // aims at, so the comfort test must not blame the capture point for it.
        let route = north_route();
        let p = params();
        let pose = Pose2::new(DVec2::new(200.0, 20_000.0), -std::f64::consts::FRAC_PI_2);
        let rejoin = plan_rejoin(&route, pose, 20_000.0, &p, None).expect("plannable");
        assert!(
            rejoin.total_turn_rad > std::f64::consts::PI * 0.8,
            "a reversal really does need the turn"
        );
        assert!(
            rejoin.excess_turn_rad <= p.max_excess_turn_rad,
            "but it should not be charged as excess: {:.0} deg",
            rejoin.excess_turn_rad.to_degrees()
        );
    }

    #[test]
    fn on_course_and_aligned_captures_straight_ahead() {
        let route = north_route();
        let pose = Pose2::new(DVec2::new(0.0, 5_000.0), std::f64::consts::FRAC_PI_2);
        let rejoin = plan_rejoin(&route, pose, 5_000.0, &params(), None).expect("plannable");
        // Already on it: the flyable path is essentially the straight route.
        assert_abs_diff_eq!(rejoin.detour_ratio(), 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(rejoin.excess_turn_rad, 0.0, epsilon = 1e-6);
        // And the capture is the minimum distance ahead, not somewhere distant.
        assert_abs_diff_eq!(
            rejoin.capture_along_m,
            5_000.0 + params().min_capture_m,
            epsilon = 1.0
        );
    }

    #[test]
    fn a_small_offset_joins_gently_and_a_large_one_reaches_further_out() {
        let route = north_route();
        let p = params();
        let near = plan_rejoin(
            &route,
            Pose2::new(DVec2::new(300.0, 5_000.0), std::f64::consts::FRAC_PI_2),
            5_000.0,
            &p,
            None,
        )
        .expect("plannable");
        let far = plan_rejoin(
            &route,
            Pose2::new(DVec2::new(6_000.0, 5_000.0), std::f64::consts::FRAC_PI_2),
            5_000.0,
            &p,
            None,
        )
        .expect("plannable");
        assert!(
            far.capture_along_m > near.capture_along_m,
            "a bigger deviation must capture further along: near {} far {}",
            near.capture_along_m,
            far.capture_along_m
        );
        // Both stay within the comfort budget — that is the point of searching.
        assert!(near.excess_turn_rad <= p.max_excess_turn_rad);
        assert!(far.excess_turn_rad <= p.max_excess_turn_rad);
    }

    #[test]
    fn the_rejoin_ends_on_the_route_pointing_along_it() {
        let route = north_route();
        for (x, heading) in [
            (2_000.0, std::f64::consts::FRAC_PI_2),
            (-4_000.0, 0.0),
            (500.0, std::f64::consts::PI),
        ] {
            let pose = Pose2::new(DVec2::new(x, 6_000.0), heading);
            let rejoin = plan_rejoin(&route, pose, 6_000.0, &params(), None).expect("plannable");
            let end = rejoin
                .path
                .point_at(rejoin.path.length())
                .expect("non-empty");
            // On the route (x = 0) and travelling along it (north).
            assert_abs_diff_eq!(end.position.x, 0.0, epsilon = 1.0);
            assert_abs_diff_eq!(
                crate::wrap_angle(end.theta - std::f64::consts::FRAC_PI_2),
                0.0,
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn flying_away_from_the_route_still_produces_a_flyable_join() {
        // Pointing due south, well right of a northbound route: the only way
        // back is a turn, and it must still meet the route tangentially.
        let route = north_route();
        let pose = Pose2::new(DVec2::new(3_000.0, 20_000.0), -std::f64::consts::FRAC_PI_2);
        let rejoin = plan_rejoin(&route, pose, 20_000.0, &params(), None).expect("plannable");
        let end = rejoin
            .path
            .point_at(rejoin.path.length())
            .expect("non-empty");
        assert_abs_diff_eq!(end.position.x, 0.0, epsilon = 1.0);
        assert_abs_diff_eq!(
            crate::wrap_angle(end.theta - std::f64::consts::FRAC_PI_2),
            0.0,
            epsilon = 1e-6
        );
        assert!(rejoin.length_m > 0.0);
    }

    #[test]
    fn a_still_feasible_hint_is_kept_so_the_cue_does_not_twitch() {
        let route = north_route();
        let p = params();
        let pose = Pose2::new(DVec2::new(400.0, 5_000.0), std::f64::consts::FRAC_PI_2);
        let first = plan_rejoin(&route, pose, 5_000.0, &p, None).expect("plannable");
        // One second later, slightly further along and slightly closer in.
        let moved = Pose2::new(DVec2::new(380.0, 5_090.0), std::f64::consts::FRAC_PI_2);
        let second = plan_rejoin(&route, moved, 5_090.0, &p, Some(first.capture_along_m))
            .expect("plannable");
        assert_abs_diff_eq!(
            second.capture_along_m,
            first.capture_along_m,
            epsilon = 1e-9
        );
    }

    #[test]
    fn a_hint_the_craft_has_passed_is_discarded() {
        let route = north_route();
        let p = params();
        let pose = Pose2::new(DVec2::new(0.0, 12_000.0), std::f64::consts::FRAC_PI_2);
        let rejoin = plan_rejoin(&route, pose, 12_000.0, &p, Some(6_000.0)).expect("plannable");
        assert!(
            rejoin.capture_along_m >= 12_000.0 + p.min_capture_m,
            "captured behind the craft at {}",
            rejoin.capture_along_m
        );
    }

    #[test]
    fn the_aim_point_converges_to_the_route_course() {
        let route = north_route();
        let pose = Pose2::new(DVec2::new(1_500.0, 5_000.0), std::f64::consts::FRAC_PI_2);
        let rejoin = plan_rejoin(&route, pose, 5_000.0, &params(), None).expect("plannable");
        // Far along the rejoin the aim direction is the route's own course, so a
        // cue that points at it needs no mode change on capture.
        let (_, theta) = rejoin.aim(rejoin.length_m).expect("non-empty");
        assert_abs_diff_eq!(
            crate::wrap_angle(theta - std::f64::consts::FRAC_PI_2),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn near_the_end_of_the_route_it_captures_the_end_rather_than_nothing() {
        let route = north_route();
        let pose = Pose2::new(DVec2::new(200.0, 39_500.0), std::f64::consts::FRAC_PI_2);
        let rejoin = plan_rejoin(&route, pose, 39_500.0, &params(), None).expect("plannable");
        assert!(rejoin.capture_along_m <= route.length() + 1e-6);
        assert!(rejoin.capture_along_m > 39_500.0);
    }

    #[test]
    fn degenerate_inputs_return_none_not_panics() {
        let empty = LateralPath::default();
        let pose = Pose2::new(DVec2::ZERO, 0.0);
        assert!(plan_rejoin(&empty, pose, 0.0, &params(), None).is_none());
        let bad = RejoinParams {
            turn_radius_m: 0.0,
            ..params()
        };
        assert!(plan_rejoin(&north_route(), pose, 0.0, &bad, None).is_none());
    }
}
