//! Runway descriptors and the approach planner.
//!
//! A [`RunwayStrip`] is the navigation view of a paved strip: where it is, which
//! way it points, and how big it is. It is deliberately a plain value rather
//! than a handle into the game's structure registry, so the planner, its tests,
//! and the headless display preview all speak the same language.
//!
//! Every strip has **two** [`RunwayEnd`]s — you can land either way — and which
//! one you pick is the single most important choice in an approach. That is why
//! the selection is a first-class type with its own designator, not a boolean
//! buried in the plan.
//!
//! [`plan_approach`] produces a route that is flyable from wherever the craft
//! is: a straight final aligned with the landing heading, preceded by the
//! shortest bank-limited transition onto it ([`crate::dubins`]).

use glam::{DVec2, DVec3};

use crate::dubins::{DubinsWord, Pose2, plan_dubins};
use crate::path::{LateralPath, Leg};
use crate::vnav::{VerticalProfile, VnavParams};
use crate::waypoint::{
    RouteFrame, VerticalConstraint, Waypoint, WaypointKind, theta_of, theta_to_heading,
};

/// A paved strip, in body-fixed terms. Mirrors the game's
/// `StructureKind::Runway` + its site placement; the half-extents are the real
/// ones, because a navigation display that draws runways at a fixed symbol size
/// teaches the pilot nothing about whether they will fit on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunwayStrip {
    /// Stable identifier from whatever registry owns the strip, so a selection
    /// survives re-enumeration.
    pub id: u64,
    /// Unit body-fixed direction to the strip centre.
    pub center_dir: DVec3,
    /// Unit body-fixed tangent along the strip's nominal heading.
    pub heading_tangent: DVec3,
    pub half_length_m: f64,
    pub half_width_m: f64,
    /// Strip surface elevation (m above the body reference radius).
    pub elevation_m: f64,
    /// Body reference radius (m).
    pub body_radius_m: f64,
}

impl RunwayStrip {
    /// The strip centre as a body-fixed point.
    pub fn center_point(&self) -> DVec3 {
        self.center_dir * (self.body_radius_m + self.elevation_m)
    }

    /// Both landable ends, in `heading_tangent` order (forward end first).
    pub fn ends(&self) -> [RunwayEnd; 2] {
        [
            RunwayEnd {
                strip: *self,
                reciprocal: false,
            },
            RunwayEnd {
                strip: *self,
                reciprocal: true,
            },
        ]
    }
}

/// One landable direction of a strip: the strip plus which way you are landing.
///
/// `reciprocal == false` means landing **along** `heading_tangent` (so the
/// threshold is at the `−tangent` end); `true` is the other way round.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunwayEnd {
    pub strip: RunwayStrip,
    pub reciprocal: bool,
}

impl RunwayEnd {
    /// Unit body-fixed direction of travel when landing on this end.
    pub fn landing_dir(&self) -> DVec3 {
        if self.reciprocal {
            -self.strip.heading_tangent
        } else {
            self.strip.heading_tangent
        }
    }

    /// The threshold — the end you cross on the way in — as a body-fixed point.
    pub fn threshold_point(&self) -> DVec3 {
        self.strip.center_point() - self.landing_dir() * self.strip.half_length_m
    }

    /// The far end you roll out toward, as a body-fixed point.
    pub fn stop_end_point(&self) -> DVec3 {
        self.strip.center_point() + self.landing_dir() * self.strip.half_length_m
    }

    /// Full paved length (m).
    pub fn length_m(&self) -> f64 {
        2.0 * self.strip.half_length_m
    }

    /// A [`RouteFrame`] anchored at this end's threshold. Every approach to this
    /// end is planned in this frame.
    pub fn route_frame(&self) -> Option<RouteFrame> {
        RouteFrame::new(
            self.threshold_point().try_normalize()?,
            self.strip.body_radius_m,
            self.strip.elevation_m,
        )
    }

    /// Compass heading (rad, 0 = north) of the landing direction, measured at
    /// the threshold.
    pub fn landing_heading_rad(&self, frame: &RouteFrame) -> f64 {
        theta_to_heading(theta_of(frame.direction_to_local(self.landing_dir())))
    }

    /// Runway designator, `1..=36` — the landing heading in tens of degrees,
    /// the number painted on the threshold.
    pub fn designator(&self, frame: &RouteFrame) -> u8 {
        let deg = self
            .landing_heading_rad(frame)
            .to_degrees()
            .rem_euclid(360.0);
        let mut n = (deg / 10.0).round() as i32;
        if n <= 0 {
            n += 36;
        } else if n > 36 {
            n -= 36;
        }
        n as u8
    }

    /// The other end of the same strip.
    pub fn flipped(&self) -> Self {
        Self {
            strip: self.strip,
            reciprocal: !self.reciprocal,
        }
    }
}

/// Approach-planning parameters. Craft-dependent values (speed, bank limit) are
/// supplied by the caller from the actual vehicle — a spaceplane and a light
/// aircraft do not fly the same pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApproachParams {
    /// Maximum bank the planned turns may require (rad). Also the ceiling the
    /// guidance bank command is clamped to.
    pub bank_limit_rad: f64,
    /// Local gravitational acceleration (m/s²) — sizes the turn radius. Thalos
    /// is not Earth; a heavier world turns the same craft wider.
    pub gravity_m_s2: f64,
    /// Planning speed (m/s) for the turn radius: the speed you expect to be
    /// flying while maneuvering onto final.
    pub maneuver_speed_m_s: f64,
    /// Never plan a turn tighter than this (m), whatever the speed says.
    pub min_turn_radius_m: f64,
    /// Length of the straight final approach segment, threshold-ward from the
    /// final approach point (m).
    pub final_length_m: f64,
    /// How far past the threshold the aim (touchdown) point sits (m). The
    /// glideslope is aimed here, so this also sets the threshold crossing
    /// height: `aim_inset · tan(glideslope)`.
    pub aim_inset_m: f64,
    /// Shortest straight run onto the aim point the planner will leave itself
    /// when joining a final that is already under way (m). Below this it stops
    /// shortening and accepts a steeper join, because there is no room left.
    pub min_capture_run_m: f64,
    pub vnav: VnavParams,
}

impl Default for ApproachParams {
    fn default() -> Self {
        Self {
            bank_limit_rad: 25.0_f64.to_radians(),
            gravity_m_s2: crate::EARTH_G_M_S2,
            maneuver_speed_m_s: 110.0,
            min_turn_radius_m: 400.0,
            final_length_m: 9_000.0,
            aim_inset_m: 300.0,
            min_capture_run_m: 1_200.0,
            vnav: VnavParams::default(),
        }
    }
}

impl ApproachParams {
    /// Bank-limited turn radius (m) for the planning speed: `v² / (g · tan φ)`,
    /// floored at [`Self::min_turn_radius_m`].
    pub fn turn_radius_m(&self) -> f64 {
        let tan_phi = self.bank_limit_rad.tan();
        if !(tan_phi.is_finite()) || tan_phi <= 1e-6 || self.gravity_m_s2 <= 0.0 {
            return self.min_turn_radius_m.max(1.0);
        }
        let v = self.maneuver_speed_m_s.max(1.0);
        (v * v / (self.gravity_m_s2 * tan_phi)).max(self.min_turn_radius_m.max(1.0))
    }
}

/// Where along the route the craft currently is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproachPhase {
    /// Maneuvering to intercept the final approach segment.
    Transition,
    /// Established on the straight final, before the threshold.
    Final,
    /// Past the threshold — over the paved strip.
    Touchdown,
}

/// A planned approach: one lateral path, one vertical profile, and the waypoints
/// that name the interesting points on it.
#[derive(Debug, Clone, PartialEq)]
pub struct ApproachPlan {
    /// The end being landed on.
    pub end: RunwayEnd,
    /// Frame the geometry lives in (anchored at the threshold).
    pub frame: RouteFrame,
    /// Transition + final, as one continuous path ending at the aim point.
    pub path: LateralPath,
    /// Along-path distance where the straight final begins (m).
    pub final_start_along_m: f64,
    pub vertical: VerticalProfile,
    /// Compass heading (rad) of the landing direction.
    pub landing_heading_rad: f64,
    /// Runway designator (`1..=36`).
    pub designator: u8,
    /// Named points on the route, in fly order.
    pub waypoints: Vec<Waypoint>,
    /// Which Dubins word the transition used — `None` if the craft was already
    /// on the extended centerline and the plan is a straight-in.
    pub transition_word: Option<DubinsWord>,
    /// Turn radius the transition was planned with (m).
    pub turn_radius_m: f64,
    /// How far past the threshold the aim point sits (m) — the plan's own copy,
    /// so `dtg` readings stay interpretable without the params that built it.
    pub aim_inset_m: f64,
}

impl ApproachPlan {
    /// Total route length (m), transition plus final.
    pub fn length_m(&self) -> f64 {
        self.path.length()
    }

    /// Distance-to-go at the threshold. `dtg = 0` is the **aim** point, so
    /// crossing the threshold happens at `aim_inset_m` still to go — the reason
    /// a threshold-crossing readout must use this rather than zero.
    pub fn threshold_dtg_m(&self) -> f64 {
        self.aim_inset_m
    }
}

/// Plan an approach to `end` from the craft's current pose.
///
/// `craft_body_fixed` is the craft position in the body-fixed frame (metres from
/// the body centre) and `track_dir_body_fixed` its current ground-track
/// direction (velocity for a moving craft, nose for a stationary one). The
/// vertical profile is planned from the craft's current altitude, so a craft
/// already low gets a level intercept instead of a climb.
///
/// Returns `None` only for degenerate geometry (a strip at the body centre, a
/// zero-length track direction with no usable fallback).
pub fn plan_approach(
    end: RunwayEnd,
    craft_body_fixed: DVec3,
    track_dir_body_fixed: DVec3,
    params: &ApproachParams,
) -> Option<ApproachPlan> {
    let frame = end.route_frame()?;
    let landing_dir_local = frame
        .direction_to_local(end.landing_dir())
        .try_normalize()?;
    let landing_theta = theta_of(landing_dir_local);

    // Aim point: `aim_inset` down the strip from the threshold (which is the
    // frame origin), and the final approach point `final_length` before it.
    let aim_local = landing_dir_local * params.aim_inset_m;
    let fap_local = -landing_dir_local * params.final_length_m;

    let craft_local = frame.to_local(craft_body_fixed);
    // How much centreline is left between the craft and the aim point (positive
    // when the aim point is still ahead along the landing direction).
    let run_to_aim_m = (aim_local - craft_local).dot(landing_dir_local);
    let craft_theta = match frame
        .direction_to_local(track_dir_body_fixed)
        .try_normalize()
    {
        Some(d) => theta_of(d),
        // A craft with no usable ground track (straight up, or stationary with a
        // vertical nose) gets the landing heading as its assumed track: the
        // plan is then a straight-in from wherever it is, which is the least
        // surprising fallback and never a NaN.
        None => landing_theta,
    };

    // Where to join the final approach. Normally that is the final approach
    // point; but a craft already *inside* the final corridor must not be sent
    // back out to it — a Dubins path to a fix 6 km behind you is a full
    // turn-around, which is a correct answer to the wrong question when you are
    // three kilometres out and lined up. In that case the join point slides
    // forward along the centreline, leaving a stabilised run onto the aim point.
    let joining_late = run_to_aim_m > 0.0 && run_to_aim_m < params.final_length_m;
    let join_local = if joining_late {
        let run =
            (run_to_aim_m * 0.5).clamp(params.min_capture_run_m.min(run_to_aim_m), run_to_aim_m);
        aim_local - landing_dir_local * run
    } else {
        fap_local
    };

    let radius = params.turn_radius_m();
    let transition = plan_dubins(
        Pose2::new(craft_local, craft_theta),
        Pose2::new(join_local, landing_theta),
        radius,
    )?;

    // A transition that is already a single straight line along the landing
    // heading is a straight-in: report no word, so displays can say so.
    let straight_in = transition.path.legs.len() <= 1
        && crate::wrap_angle(craft_theta - landing_theta).abs() < 1.0e-3;

    let mut legs = transition.path.legs.clone();
    let final_start_along_m = transition.length_m;
    legs.push(Leg::Line {
        from: join_local,
        to: aim_local,
    });
    let path = LateralPath::new(legs);

    let craft_altitude_m = frame.altitude_of(craft_body_fixed);
    let aim_altitude_m = frame.origin_altitude_m;
    let vnav = VnavParams {
        final_dtg_m: params.final_length_m + params.aim_inset_m,
        ..params.vnav
    };
    let vertical = VerticalProfile::plan(&vnav, aim_altitude_m, craft_altitude_m);

    let designator = end.designator(&frame);
    let landing_heading_rad = end.landing_heading_rad(&frame);

    // The final-approach waypoint marks where the final is actually joined, with
    // the profile altitude for that distance-to-go — not the nominal capture
    // altitude, which would draw a waypoint the craft is nowhere near when
    // joining late.
    let join_dtg_m = (aim_local - join_local).length();
    let fap_altitude = vertical.target_altitude_m(join_dtg_m);
    let waypoints = vec![
        Waypoint {
            dir: frame.to_body_fixed(join_local, fap_altitude).normalize(),
            vertical: Some(VerticalConstraint::At(fap_altitude)),
            speed_m_s: Some(vnav.approach_speed_m_s),
            kind: WaypointKind::FinalApproach,
        },
        Waypoint {
            dir: frame.origin_dir,
            vertical: Some(VerticalConstraint::At(
                aim_altitude_m + params.aim_inset_m * vnav.glideslope_rad.tan(),
            )),
            speed_m_s: Some(vnav.approach_speed_m_s),
            kind: WaypointKind::Threshold,
        },
        Waypoint {
            dir: frame.to_body_fixed(aim_local, aim_altitude_m).normalize(),
            vertical: Some(VerticalConstraint::At(aim_altitude_m)),
            speed_m_s: None,
            kind: WaypointKind::Aim,
        },
    ];

    Some(ApproachPlan {
        end,
        frame,
        path,
        final_start_along_m,
        vertical,
        landing_heading_rad,
        designator,
        waypoints,
        transition_word: (!straight_in).then_some(transition.word),
        turn_radius_m: radius,
        aim_inset_m: params.aim_inset_m,
    })
}

/// Local-plane geometry of a strip, for drawing it to scale.
///
/// Returns the strip centre, the unit along-direction, and the half-extents, all
/// in `frame`'s local coordinates — everything a display needs to draw the real
/// rectangle rather than a fixed-size glyph.
pub fn strip_in_frame(strip: &RunwayStrip, frame: &RouteFrame) -> Option<StripGeometry> {
    let along = frame
        .direction_to_local(strip.heading_tangent)
        .try_normalize()?;
    Some(StripGeometry {
        center: frame.to_local(strip.center_point()),
        along,
        half_length_m: strip.half_length_m,
        half_width_m: strip.half_width_m,
    })
}

/// A strip projected into a local plane, in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StripGeometry {
    pub center: DVec2,
    /// Unit direction along the strip (the `heading_tangent` side).
    pub along: DVec2,
    pub half_length_m: f64,
    pub half_width_m: f64,
}

impl StripGeometry {
    /// The two strip ends in local coordinates: `(−along end, +along end)`.
    pub fn ends(&self) -> (DVec2, DVec2) {
        (
            self.center - self.along * self.half_length_m,
            self.center + self.along * self.half_length_m,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const R: f64 = 6.0e6;

    /// A 5 km × 90 m strip at lat 0, lon 0, pointing north-east-ish (the shape
    /// of the Thalos spaceport primary).
    fn strip() -> RunwayStrip {
        let center_dir = DVec3::X;
        // Local north at (1,0,0) is +Y; local east is... build via the frame so
        // the test does not hand-roll a basis.
        let frame = RouteFrame::new(center_dir, R, 700.0).expect("valid");
        let heading = (frame.north * 30.0_f64.to_radians().cos()
            + frame.east * 30.0_f64.to_radians().sin())
        .normalize();
        RunwayStrip {
            id: 1,
            center_dir,
            heading_tangent: heading,
            half_length_m: 2_500.0,
            half_width_m: 45.0,
            elevation_m: 700.0,
            body_radius_m: R,
        }
    }

    #[test]
    fn both_ends_are_reciprocal_designators() {
        let s = strip();
        let [a, b] = s.ends();
        let fa = a.route_frame().expect("valid");
        let fb = b.route_frame().expect("valid");
        let da = a.designator(&fa) as i32;
        let db = b.designator(&fb) as i32;
        // 30° heading → runway 03; the reciprocal is 210° → 21.
        assert_eq!(da, 3, "expected RWY 03, got {da}");
        assert_eq!(db, 21, "expected RWY 21, got {db}");
        assert_eq!(((da - db).abs()) % 36, 18, "ends must be 180° apart");
    }

    #[test]
    fn threshold_and_stop_end_are_a_full_length_apart() {
        let s = strip();
        let end = s.ends()[0];
        let d = end.threshold_point().distance(end.stop_end_point());
        assert_abs_diff_eq!(d, end.length_m(), epsilon = 1.0);
        assert_abs_diff_eq!(end.length_m(), 5_000.0, epsilon = 1e-9);
        // Flipping swaps them.
        assert_abs_diff_eq!(
            end.threshold_point()
                .distance(end.flipped().stop_end_point()),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn a_straight_in_plan_is_a_straight_line_plus_final() {
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        let params = ApproachParams::default();
        // Put the craft 25 km out on the extended centerline, on heading.
        let landing_local = frame.direction_to_local(end.landing_dir()).normalize();
        let craft_local = -landing_local * 25_000.0;
        let craft = frame.to_body_fixed(craft_local, 3_000.0);
        let plan = plan_approach(end, craft, end.landing_dir(), &params).expect("plannable");
        assert!(
            plan.transition_word.is_none(),
            "expected a straight-in, got {:?}",
            plan.transition_word
        );
        // Route ends at the aim point, past the threshold.
        let last = plan
            .path
            .point_at(plan.path.length())
            .expect("non-empty path");
        assert_abs_diff_eq!(
            last.position.distance(landing_local * params.aim_inset_m),
            0.0,
            epsilon = 1e-6
        );
        // And the total length is intercept + final + aim inset.
        assert_abs_diff_eq!(
            plan.length_m(),
            25_000.0 + params.aim_inset_m,
            epsilon = 1.0
        );
    }

    #[test]
    fn an_approach_from_behind_the_field_turns_around() {
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        let landing_local = frame.direction_to_local(end.landing_dir()).normalize();
        // Craft 6 km PAST the runway, flying away from it.
        let craft_local = landing_local * 6_000.0;
        let craft = frame.to_body_fixed(craft_local, 1_500.0);
        let plan = plan_approach(end, craft, end.landing_dir(), &ApproachParams::default())
            .expect("plannable");
        assert!(
            plan.transition_word.is_some(),
            "must maneuver, not straight-in"
        );
        // The route must be long enough to contain a reversal, and must still
        // finish exactly on the aim point pointing down the runway.
        assert!(
            plan.length_m() > 6_000.0 + plan.final_start_along_m * 0.0 + 9_000.0,
            "route too short to be a real turnaround: {}",
            plan.length_m()
        );
        let last = plan.path.point_at(plan.path.length()).expect("non-empty");
        assert_abs_diff_eq!(
            crate::wrap_angle(last.theta - theta_of(landing_local)),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn the_final_segment_is_aligned_with_the_landing_heading() {
        let s = strip();
        for end in s.ends() {
            let frame = end.route_frame().expect("valid");
            let landing_local = frame.direction_to_local(end.landing_dir()).normalize();
            // Approach from well off to one side.
            let craft_local = DVec2::new(-14_000.0, 9_000.0);
            let craft = frame.to_body_fixed(craft_local, 2_500.0);
            let plan = plan_approach(end, craft, end.landing_dir(), &ApproachParams::default())
                .expect("plannable");
            // The last leg is the final: sample just before the end.
            let p = plan
                .path
                .point_at(plan.path.length() - 100.0)
                .expect("non-empty");
            assert_abs_diff_eq!(
                crate::wrap_angle(p.theta - theta_of(landing_local)),
                0.0,
                epsilon = 1e-6
            );
            // `final_start_along_m` really is where the final begins: the point
            // there must sit ON the extended centreline (zero lateral offset)
            // and be heading down it. Its distance out is *not* fixed at the
            // nominal final length — a craft already inside the corridor joins
            // the final where it is (see
            // `a_craft_already_on_short_final_joins_it_instead_of_flying_back`).
            let join = plan
                .path
                .point_at(plan.final_start_along_m)
                .expect("non-empty");
            let lateral = crate::path::signed_cross_track(
                join.position - landing_local * plan.aim_inset_m,
                theta_of(landing_local),
            );
            assert_abs_diff_eq!(lateral, 0.0, epsilon = 1.0);
            assert_abs_diff_eq!(
                crate::wrap_angle(join.theta - theta_of(landing_local)),
                0.0,
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn a_craft_already_on_short_final_joins_it_instead_of_flying_back() {
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        let landing_local = frame.direction_to_local(end.landing_dir()).normalize();
        let params = ApproachParams::default();
        // 3 km before the threshold, 150 m left of centreline, on heading — the
        // situation where planning back to a 9 km final approach point would
        // command a 6 km turn-around.
        let right = DVec2::new(landing_local.y, -landing_local.x);
        let craft_local = landing_local * (params.aim_inset_m - 3_000.0) - right * 150.0;
        let craft = frame.to_body_fixed(craft_local, 850.0);
        let plan = plan_approach(end, craft, end.landing_dir(), &params).expect("plannable");
        // The whole route must be about as long as the distance left to fly, not
        // twice a 9 km final plus a reversal.
        assert!(
            plan.length_m() < 5_000.0,
            "expected a short join, got a {:.0} m route",
            plan.length_m()
        );
        // And it still ends at the aim point, on the landing heading.
        let last = plan.path.point_at(plan.path.length()).expect("non-empty");
        assert_abs_diff_eq!(
            last.position.distance(landing_local * params.aim_inset_m),
            0.0,
            epsilon = 1.0
        );
        assert_abs_diff_eq!(
            crate::wrap_angle(last.theta - theta_of(landing_local)),
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn joining_late_still_leaves_a_stabilised_run() {
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        let landing_local = frame.direction_to_local(end.landing_dir()).normalize();
        let params = ApproachParams::default();
        for run in [8_000.0, 4_000.0, 2_000.0, 900.0] {
            let craft_local = landing_local * (params.aim_inset_m - run);
            let craft = frame.to_body_fixed(craft_local, 300.0 + run * 0.05);
            let plan = plan_approach(end, craft, end.landing_dir(), &params).expect("plannable");
            let final_length = plan.length_m() - plan.final_start_along_m;
            assert!(
                final_length > 0.0,
                "no final segment left at {run} m to run"
            );
            assert!(
                final_length <= run + 1.0,
                "final segment {final_length} exceeds the {run} m available"
            );
        }
    }

    #[test]
    fn a_craft_past_the_aim_point_still_flies_the_full_pattern() {
        // Past the touchdown point, "join the final where you are" is not
        // available — there is no final left — so it must plan the way around.
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        let landing_local = frame.direction_to_local(end.landing_dir()).normalize();
        let craft_local = landing_local * 4_000.0;
        let craft = frame.to_body_fixed(craft_local, 900.0);
        let plan = plan_approach(end, craft, end.landing_dir(), &ApproachParams::default())
            .expect("plannable");
        assert!(
            plan.length_m() > 12_000.0,
            "expected a full pattern, got {:.0} m",
            plan.length_m()
        );
    }

    #[test]
    fn turn_radius_tracks_speed_and_gravity() {
        let slow = ApproachParams {
            maneuver_speed_m_s: 80.0,
            ..ApproachParams::default()
        };
        let fast = ApproachParams {
            maneuver_speed_m_s: 200.0,
            ..ApproachParams::default()
        };
        assert!(fast.turn_radius_m() > slow.turn_radius_m() * 3.0);
        let heavy = ApproachParams {
            gravity_m_s2: 2.0 * crate::EARTH_G_M_S2,
            ..slow
        };
        assert!(
            heavy.turn_radius_m() < slow.turn_radius_m(),
            "more gravity buys a tighter turn at the same bank"
        );
        // The floor holds for a crawling craft.
        let crawl = ApproachParams {
            maneuver_speed_m_s: 1.0,
            ..ApproachParams::default()
        };
        assert_abs_diff_eq!(crawl.turn_radius_m(), crawl.min_turn_radius_m);
    }

    #[test]
    fn degenerate_bank_limit_falls_back_to_the_radius_floor() {
        let p = ApproachParams {
            bank_limit_rad: 0.0,
            ..ApproachParams::default()
        };
        assert_abs_diff_eq!(p.turn_radius_m(), p.min_turn_radius_m);
    }

    #[test]
    fn a_vertical_track_direction_falls_back_to_a_straight_in() {
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        let craft = frame.to_body_fixed(DVec2::new(-20_000.0, 0.0), 4_000.0);
        // Track straight up: no ground track at all.
        let plan = plan_approach(end, craft, frame.origin_dir, &ApproachParams::default());
        assert!(plan.is_some(), "must not fail on a vertical track");
    }

    #[test]
    fn strip_geometry_carries_the_real_extents() {
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        let g = strip_in_frame(&s, &frame).expect("valid");
        assert_abs_diff_eq!(g.half_length_m, 2_500.0);
        assert_abs_diff_eq!(g.half_width_m, 45.0);
        let (a, b) = g.ends();
        assert_abs_diff_eq!(a.distance(b), 5_000.0, epsilon = 1.0);
        // The threshold end sits at the frame origin (that is what the frame is).
        assert_abs_diff_eq!(a.length(), 0.0, epsilon = 1.0);
    }
}
