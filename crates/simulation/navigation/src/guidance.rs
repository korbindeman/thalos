//! Per-frame guidance: how far off the route you are, and what to do about it.
//!
//! [`compute_guidance`] is a pure function of (plan, craft state) — no memory, no
//! integrators, no mode latches — so the ND, the PFD's deviation scales, and the
//! autopilot all read *the same numbers* by construction rather than by three
//! separate re-derivations that drift apart. Anything stateful (mode
//! engagement, selection, plan refresh policy) belongs to the caller.
//!
//! # Deviations: angular where a pilot expects angular
//!
//! Cross-track is metres, because that is what a map shows. The **localizer and
//! glideslope deviations are angles**, matching real ILS scale behaviour: the
//! same 40 m of lateral error is a full-scale deflection on short final and
//! nothing at all 20 km out. That sensitivity growth is the whole reason those
//! instruments work, so it is modelled rather than approximated with a linear
//! error and a fudged gain.

use glam::DVec3;

use crate::approach::{ApproachPhase, ApproachPlan};
use crate::path::signed_cross_track;
use crate::vnav::SpeedGate;
use crate::waypoint::{theta_of, theta_to_heading};

/// Full-scale localizer deflection (rad). ±2.5° is the ILS convention.
pub const LOC_FULL_SCALE_RAD: f64 = 0.043_633_231; // 2.5°
/// Full-scale glideslope deflection (rad). ±0.7° is the ILS convention.
pub const GS_FULL_SCALE_RAD: f64 = 0.012_217_305; // 0.7°

/// Lookahead floor for the lateral capture law (m). Below this the guidance
/// would demand an unflyably sharp intercept at low speed.
const MIN_LOOKAHEAD_M: f64 = 250.0;
/// Lookahead time constant (s): the capture law aims at a point this many
/// seconds ahead along the path.
const LOOKAHEAD_TIME_S: f64 = 8.0;
/// Maximum track-angle correction the lateral law will command (rad) — a 45°
/// intercept, the standard maximum for joining a course.
const MAX_INTERCEPT_RAD: f64 = std::f64::consts::FRAC_PI_4;
/// Proportional gain from track-angle error to commanded turn rate (1/s).
const TRACK_GAIN_PER_S: f64 = 0.45;
/// Proportional gain from altitude error to commanded vertical speed (1/s).
const ALTITUDE_GAIN_PER_S: f64 = 0.06;
/// Vertical-speed command clamp (m/s) — keeps the correction civilised.
const MAX_VS_CORRECTION_M_S: f64 = 8.0;

/// Craft state the guidance needs, in body-fixed terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuidanceInput {
    /// Craft position in the body-fixed frame (m from the body centre).
    pub position_body_fixed: DVec3,
    /// Ground-track direction, body-fixed. Velocity direction for a moving
    /// craft; the nose is an acceptable stand-in when nearly stationary.
    pub track_dir_body_fixed: DVec3,
    /// Ground speed (m/s) — sizes the lookahead and converts turn rate to bank.
    pub ground_speed_m_s: f64,
    /// Local gravitational acceleration (m/s²) for the turn-rate ↔ bank relation.
    pub gravity_m_s2: f64,
    /// Bank the command is clamped to (rad).
    pub bank_limit_rad: f64,
}

/// Everything the displays and the autopilot read.
///
/// Angles that a pilot reads (`*_heading_rad`) are **compass** (0 = north, CW);
/// deviations are signed with the pilot's own sense: positive cross-track and
/// positive localizer mean *you are right of where you should be*, positive
/// glideslope means *you are high*, positive bank command means *roll right*.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Guidance {
    pub phase: ApproachPhase,
    /// Signed lateral offset from the route (m, + = right of course).
    pub cross_track_m: f64,
    /// Route course at the closest point (compass rad).
    pub course_heading_rad: f64,
    /// Craft ground track (compass rad).
    pub track_heading_rad: f64,
    /// Track the craft should fly now to capture and hold the route (compass
    /// rad) — course plus the intercept correction.
    pub desired_heading_rad: f64,
    /// Distance still to fly along the route, to the aim point (m).
    pub dtg_m: f64,
    /// Distance flown along the route (m).
    pub along_m: f64,
    /// Straight-line distance to the threshold (m) — what a "RWY 3.2 km"
    /// readout means, as distinct from route distance.
    pub threshold_range_m: f64,
    pub altitude_m: f64,
    pub target_altitude_m: f64,
    /// `altitude − target` (m, + = high).
    pub altitude_error_m: f64,
    pub target_speed_m_s: Option<f64>,
    /// Next speed/configuration gate ahead.
    pub next_gate: Option<SpeedGate>,
    /// Angular localizer deviation (rad, + = right of the centerline). Full
    /// scale is [`LOC_FULL_SCALE_RAD`].
    pub loc_deviation_rad: f64,
    /// Angular glideslope deviation (rad, + = above the slope). Full scale is
    /// [`GS_FULL_SCALE_RAD`].
    pub gs_deviation_rad: f64,
    /// Commanded bank (rad, + = right), clamped to the bank limit. The flight
    /// director cue and the future autoland both read this.
    pub bank_command_rad: f64,
    /// Commanded vertical speed (m/s, + = up).
    pub vertical_speed_command_m_s: f64,
    /// Whether the craft is inside both full-scale deflections — "established".
    pub established: bool,
}

impl Guidance {
    /// Localizer deflection as a fraction of full scale, clamped to `[-1, 1]`.
    pub fn loc_deflection(&self) -> f64 {
        (self.loc_deviation_rad / LOC_FULL_SCALE_RAD).clamp(-1.0, 1.0)
    }

    /// Glideslope deflection as a fraction of full scale, clamped to `[-1, 1]`.
    pub fn gs_deflection(&self) -> f64 {
        (self.gs_deviation_rad / GS_FULL_SCALE_RAD).clamp(-1.0, 1.0)
    }
}

/// Compute guidance for `plan` at the craft state in `input`.
///
/// Returns `None` only when the plan's path is empty (which
/// [`crate::approach::plan_approach`] never produces).
pub fn compute_guidance(plan: &ApproachPlan, input: &GuidanceInput) -> Option<Guidance> {
    let frame = &plan.frame;
    let local = frame.to_local(input.position_body_fixed);
    let altitude_m = frame.altitude_of(input.position_body_fixed);

    let closest = plan.path.closest(local)?;
    let dtg_m = (plan.path.length() - closest.along_m).max(0.0);

    // Track direction: fall back to the route course when the craft has no
    // usable ground track, so a stationary craft reads zero track error rather
    // than NaN.
    let track_theta = match frame
        .direction_to_local(input.track_dir_body_fixed)
        .try_normalize()
    {
        Some(d) => theta_of(d),
        None => closest.theta,
    };

    // --- Lateral: L1-style capture. Aim at a point one lookahead ahead on the
    // path; the correction is the angle to it, capped at a 45° intercept.
    let lookahead = (input.ground_speed_m_s.max(0.0) * LOOKAHEAD_TIME_S).max(MIN_LOOKAHEAD_M);
    // Positive cross-track means the craft is RIGHT of course, so the desired
    // track must rotate to the left of the course — and left is CCW, i.e. a
    // positive θ offset (see `crate::waypoint`). Flipping this sign turns the
    // capture law into a divergence law.
    let intercept = (closest.cross_track_m / lookahead)
        .atan()
        .clamp(-MAX_INTERCEPT_RAD, MAX_INTERCEPT_RAD);
    let desired_theta = closest.theta + intercept;
    let heading_error = crate::wrap_angle(desired_theta - track_theta);

    // Turn rate → bank: `tan φ = ω·v/g`.
    let bank_command_rad = if input.gravity_m_s2 > 0.0 {
        let omega = TRACK_GAIN_PER_S * heading_error;
        let v = input.ground_speed_m_s.max(1.0);
        let tan_phi = omega * v / input.gravity_m_s2;
        // A left turn (CCW, positive ω) is a LEFT bank, i.e. negative in the
        // pilot's right-positive convention — hence the sign flip.
        let phi = -tan_phi.atan();
        phi.clamp(-input.bank_limit_rad.abs(), input.bank_limit_rad.abs())
    } else {
        0.0
    };

    // --- Vertical: profile altitude plus a proportional correction on top of
    // the profile's own descent rate.
    let target_altitude_m = plan.vertical.target_altitude_m(dtg_m);
    let altitude_error_m = altitude_m - target_altitude_m;
    let profile_vs = if dtg_m <= plan.vertical.capture_dtg_m {
        -input.ground_speed_m_s.max(0.0) * plan.vertical.glideslope_rad.tan()
    } else if dtg_m <= plan.vertical.top_of_descent_dtg_m() {
        -input.ground_speed_m_s.max(0.0) * plan.vertical.cruise_descent_rad.tan()
    } else {
        0.0
    };
    let vertical_speed_command_m_s = profile_vs
        + (-ALTITUDE_GAIN_PER_S * altitude_error_m)
            .clamp(-MAX_VS_CORRECTION_M_S, MAX_VS_CORRECTION_M_S);

    // --- ILS-style deviations, measured against the final approach centerline
    // (not the whole route — a localizer needle has no opinion about a base leg).
    let landing_local = frame
        .direction_to_local(plan.end.landing_dir())
        .try_normalize()
        .unwrap_or(crate::waypoint::dir_from_theta(closest.theta));
    let landing_theta = theta_of(landing_local);
    let aim_local = landing_local * plan.aim_inset_m;
    let to_aim = aim_local - local;
    // Along-centerline distance still to run to the aim point (negative once
    // past it).
    let centerline_distance_m = to_aim.dot(landing_local);
    let lateral_m = signed_cross_track(local - aim_local, landing_theta);
    let loc_deviation_rad = lateral_m.atan2(centerline_distance_m.max(1.0));
    let gs_deviation_rad = plan
        .vertical
        .glideslope_deviation_rad(centerline_distance_m.max(1.0), altitude_m);

    let threshold_range_m = input
        .position_body_fixed
        .distance(plan.end.threshold_point());

    let phase = if dtg_m <= plan.aim_inset_m {
        ApproachPhase::Touchdown
    } else if closest.along_m >= plan.final_start_along_m - 1.0 {
        ApproachPhase::Final
    } else {
        ApproachPhase::Transition
    };

    let established = loc_deviation_rad.abs() <= LOC_FULL_SCALE_RAD
        && gs_deviation_rad.abs() <= GS_FULL_SCALE_RAD;

    Some(Guidance {
        phase,
        cross_track_m: closest.cross_track_m,
        course_heading_rad: theta_to_heading(closest.theta),
        track_heading_rad: theta_to_heading(track_theta),
        desired_heading_rad: theta_to_heading(desired_theta),
        dtg_m,
        along_m: closest.along_m,
        threshold_range_m,
        altitude_m,
        target_altitude_m,
        altitude_error_m,
        target_speed_m_s: plan.vertical.target_speed_m_s(dtg_m),
        next_gate: plan.vertical.next_gate(dtg_m),
        loc_deviation_rad,
        gs_deviation_rad,
        bank_command_rad,
        vertical_speed_command_m_s,
        established,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approach::{ApproachParams, RunwayStrip, plan_approach};
    use crate::waypoint::RouteFrame;
    use approx::assert_abs_diff_eq;
    use glam::DVec2;

    const R: f64 = 6.0e6;

    fn strip() -> RunwayStrip {
        let center_dir = DVec3::X;
        let frame = RouteFrame::new(center_dir, R, 700.0).expect("valid");
        RunwayStrip {
            id: 1,
            center_dir,
            // Due north, so "right of course" is unambiguously east.
            heading_tangent: frame.north,
            half_length_m: 2_500.0,
            half_width_m: 45.0,
            elevation_m: 700.0,
            body_radius_m: R,
        }
    }

    struct Scene {
        plan: ApproachPlan,
        frame: RouteFrame,
        landing_local: DVec2,
    }

    /// A straight-in plan from 20 km out, on the centerline and on profile.
    fn scene() -> Scene {
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        let landing_local = frame.direction_to_local(end.landing_dir()).normalize();
        let craft = frame.to_body_fixed(-landing_local * 20_000.0, 2_000.0);
        let plan = plan_approach(end, craft, end.landing_dir(), &ApproachParams::default())
            .expect("plannable");
        Scene {
            plan,
            frame,
            landing_local,
        }
    }

    /// Craft state on the extended centerline `dist` before the aim point,
    /// offset `lateral` metres to the right and at `altitude`.
    fn state(sc: &Scene, dist: f64, lateral: f64, altitude: f64) -> GuidanceInput {
        let right = DVec2::new(sc.landing_local.y, -sc.landing_local.x);
        let local = sc.landing_local * (sc.plan.aim_inset_m - dist) + right * lateral;
        GuidanceInput {
            position_body_fixed: sc.frame.to_body_fixed(local, altitude),
            track_dir_body_fixed: sc.plan.end.landing_dir(),
            ground_speed_m_s: 80.0,
            gravity_m_s2: crate::EARTH_G_M_S2,
            bank_limit_rad: 25.0_f64.to_radians(),
        }
    }

    fn on_slope_altitude(sc: &Scene, dist: f64) -> f64 {
        sc.plan.vertical.target_altitude_m(dist)
    }

    #[test]
    fn on_course_and_on_slope_reads_zero_everywhere() {
        let sc = scene();
        let g = compute_guidance(&sc.plan, &state(&sc, 5_000.0, 0.0, on_slope_altitude(&sc, 5_000.0)))
            .expect("guidance");
        assert_abs_diff_eq!(g.cross_track_m, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(g.loc_deviation_rad, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(g.gs_deviation_rad, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(g.bank_command_rad, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(g.altitude_error_m, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(g.dtg_m, 5_000.0, epsilon = 1.0);
        assert!(g.established);
        assert_eq!(g.phase, ApproachPhase::Final);
    }

    #[test]
    fn right_of_course_commands_a_left_bank() {
        let sc = scene();
        let dist = 6_000.0;
        let g = compute_guidance(&sc.plan, &state(&sc, dist, 400.0, on_slope_altitude(&sc, dist)))
            .expect("guidance");
        assert!(g.cross_track_m > 0.0, "right of course is positive");
        assert!(g.loc_deviation_rad > 0.0, "localizer reads right");
        assert!(
            g.bank_command_rad < 0.0,
            "must roll LEFT to correct a right offset, got {}",
            g.bank_command_rad.to_degrees()
        );
        // Mirrored on the other side.
        let g2 = compute_guidance(&sc.plan, &state(&sc, dist, -400.0, on_slope_altitude(&sc, dist)))
            .expect("guidance");
        assert!(g2.cross_track_m < 0.0);
        assert!(g2.bank_command_rad > 0.0, "must roll right");
        assert_abs_diff_eq!(g.bank_command_rad, -g2.bank_command_rad, epsilon = 1e-9);
    }

    #[test]
    fn bank_command_respects_the_limit() {
        let sc = scene();
        // Wildly off course and pointing the wrong way.
        let mut st = state(&sc, 8_000.0, 20_000.0, 3_000.0);
        st.track_dir_body_fixed = -sc.plan.end.landing_dir();
        st.ground_speed_m_s = 250.0;
        let g = compute_guidance(&sc.plan, &st).expect("guidance");
        assert!(
            g.bank_command_rad.abs() <= 25.0_f64.to_radians() + 1e-12,
            "bank {} exceeded the limit",
            g.bank_command_rad.to_degrees()
        );
    }

    #[test]
    fn high_and_low_read_on_the_glideslope_and_the_vertical_command() {
        let sc = scene();
        let dist = 7_000.0;
        let on = on_slope_altitude(&sc, dist);
        let high = compute_guidance(&sc.plan, &state(&sc, dist, 0.0, on + 150.0)).expect("g");
        assert!(high.gs_deviation_rad > 0.0, "high reads positive");
        assert!(high.altitude_error_m > 0.0);
        let low = compute_guidance(&sc.plan, &state(&sc, dist, 0.0, on - 150.0)).expect("g");
        assert!(low.gs_deviation_rad < 0.0, "low reads negative");
        // Being low must command a shallower descent than being high.
        assert!(
            low.vertical_speed_command_m_s > high.vertical_speed_command_m_s,
            "low {} should sink less than high {}",
            low.vertical_speed_command_m_s,
            high.vertical_speed_command_m_s
        );
    }

    #[test]
    fn on_profile_vertical_command_matches_the_glideslope_geometry() {
        let sc = scene();
        let dist = 5_000.0;
        let g = compute_guidance(&sc.plan, &state(&sc, dist, 0.0, on_slope_altitude(&sc, dist)))
            .expect("g");
        // 80 m/s down a 3° slope is ~4.2 m/s of sink.
        let expected = -80.0 * 3.0_f64.to_radians().tan();
        assert_abs_diff_eq!(g.vertical_speed_command_m_s, expected, epsilon = 0.05);
    }

    #[test]
    fn deviations_grow_as_the_threshold_closes() {
        let sc = scene();
        // A fixed 60 m lateral error, at 15 km and at 1 km.
        let far = compute_guidance(
            &sc.plan,
            &state(&sc, 15_000.0, 60.0, on_slope_altitude(&sc, 15_000.0)),
        )
        .expect("g");
        let near = compute_guidance(
            &sc.plan,
            &state(&sc, 1_000.0, 60.0, on_slope_altitude(&sc, 1_000.0)),
        )
        .expect("g");
        assert!(near.loc_deviation_rad > far.loc_deviation_rad * 5.0);
        // Far out it is within full scale; up close it is pegged.
        assert!(far.loc_deflection().abs() < 1.0);
        assert_abs_diff_eq!(near.loc_deflection(), 1.0);
    }

    #[test]
    fn phase_progresses_transition_final_touchdown() {
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        let landing_local = frame.direction_to_local(end.landing_dir()).normalize();
        // Start well off to the side so the plan has a real transition.
        let craft = frame.to_body_fixed(DVec2::new(12_000.0, -9_000.0), 2_500.0);
        let plan =
            plan_approach(end, craft, end.landing_dir(), &ApproachParams::default()).expect("plan");
        let sc = Scene {
            plan,
            frame,
            landing_local,
        };
        let at_start = GuidanceInput {
            position_body_fixed: craft,
            track_dir_body_fixed: sc.plan.end.landing_dir(),
            ground_speed_m_s: 100.0,
            gravity_m_s2: crate::EARTH_G_M_S2,
            bank_limit_rad: 25.0_f64.to_radians(),
        };
        assert_eq!(
            compute_guidance(&sc.plan, &at_start).expect("g").phase,
            ApproachPhase::Transition
        );
        let on_final = state(&sc, 4_000.0, 0.0, on_slope_altitude(&sc, 4_000.0));
        assert_eq!(
            compute_guidance(&sc.plan, &on_final).expect("g").phase,
            ApproachPhase::Final
        );
        // Over the strip, past the aim point.
        let over = state(&sc, -100.0, 0.0, sc.plan.frame.origin_altitude_m + 2.0);
        assert_eq!(
            compute_guidance(&sc.plan, &over).expect("g").phase,
            ApproachPhase::Touchdown
        );
    }

    #[test]
    fn threshold_range_and_dtg_are_different_quantities() {
        let sc = scene();
        // Offset far to the side: route distance-to-go must exceed the straight
        // line to the threshold is NOT guaranteed, but they must not be equal.
        let g = compute_guidance(&sc.plan, &state(&sc, 3_000.0, 2_000.0, 1_200.0)).expect("g");
        assert!(
            (g.dtg_m - g.threshold_range_m).abs() > 100.0,
            "dtg {} and threshold range {} should differ off-centerline",
            g.dtg_m,
            g.threshold_range_m
        );
    }

    #[test]
    fn a_stationary_craft_yields_finite_numbers() {
        let sc = scene();
        let mut st = state(&sc, 4_000.0, 100.0, 900.0);
        st.ground_speed_m_s = 0.0;
        st.track_dir_body_fixed = DVec3::ZERO;
        let g = compute_guidance(&sc.plan, &st).expect("g");
        assert!(g.bank_command_rad.is_finite());
        assert!(g.vertical_speed_command_m_s.is_finite());
        assert!(g.desired_heading_rad.is_finite());
        assert!(g.loc_deviation_rad.is_finite());
    }

    #[test]
    fn headings_are_compass_and_agree_with_the_runway() {
        let sc = scene();
        let dist = 6_000.0;
        let g = compute_guidance(&sc.plan, &state(&sc, dist, 0.0, on_slope_altitude(&sc, dist)))
            .expect("g");
        // The strip points due north, so course/track/desired are all ~0/360.
        for h in [g.course_heading_rad, g.track_heading_rad, g.desired_heading_rad] {
            let deg = h.to_degrees();
            assert!(
                !(1.0..=359.0).contains(&deg),
                "expected a northerly heading, got {deg}"
            );
        }
        assert_eq!(sc.plan.designator, 36, "due north is RWY 36");
    }
}
