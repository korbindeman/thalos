//! Per-frame guidance: how far off the route you are, and what to do about it.
//!
//! [`compute_guidance`] is a pure function of (plan, craft state) — no memory, no
//! integrators, no mode latches — so the ND, the PFD's deviation scales, and the
//! autopilot all read *the same numbers* by construction rather than by three
//! separate re-derivations that drift apart. Anything stateful (mode
//! engagement, selection, plan refresh policy) belongs to the caller.
//!
//! # Two different questions, two different answers
//!
//! This module answers both, and they must not be confused:
//!
//! - **"Where should I point?"** — the steering director
//!   ([`Guidance::desired_heading_rad`], [`Guidance::fpa_command_rad`], and the
//!   `director_*` deflections). This is **route-relative**: it follows the
//!   active path, whatever leg the craft is on. The destination being a runway
//!   is incidental — the same cue works for any route.
//!
//!   There is exactly one active path, and it is the one the ND draws. When the
//!   craft is blown off course the flyable rejoin ([`crate::rejoin`]) is
//!   *spliced into* that path by the caller, not offered here as a competing
//!   cue. The distinction is the whole of INC-20260801T035551Z: a cue that
//!   disagrees with the drawn route makes the display a lie, and one recomputed
//!   every frame gives the craft a target that slides away as it advances.
//! - **"How far off the beam am I?"** — the ILS-style localizer and glideslope
//!   deviations. These are **runway-relative** by definition: they are measured
//!   against the final approach centreline, exactly like the ground equipment
//!   they imitate, and they are meaningless on a base leg. A display should show
//!   them only once established on final.
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
/// Fraction of the feedforward turn rate the follower is allowed to add or
/// subtract for error correction, beyond the plain rate limit.
///
/// The feedforward is what *holding* the path costs; the correction is what
/// *returning* to it costs, and the bank limit has to cover both. Capping the
/// correction keeps a large cross-track error from eating the entire bank
/// budget and leaving nothing to actually fly the turn with.
const MAX_CORRECTION_RATE_PER_S: f64 = 0.12;
/// Proportional gain from altitude error to commanded vertical speed (1/s).
const ALTITUDE_GAIN_PER_S: f64 = 0.06;
/// Vertical-speed command clamp (m/s) — keeps the correction civilised.
const MAX_VS_CORRECTION_M_S: f64 = 8.0;

/// Full-scale lateral director deflection (rad): 25° of bank *error* pegs the
/// cue — the whole bank budget, so the dot uses its full travel for corrections
/// a pilot actually makes rather than saturating instantly.
pub const DIRECTOR_BANK_FULL_SCALE_RAD: f64 = 0.436_332_313; // 25°
/// Full-scale vertical director deflection (rad): a 4° flight-path-angle error.
/// Sized against the 3° glideslope, so "one dot low" is a real correction.
pub const DIRECTOR_FPA_FULL_SCALE_RAD: f64 = 0.069_813_170; // 4°

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
    /// Vertical speed (m/s, + = climbing). With ground speed this gives the
    /// craft's current flight-path angle, which is what the vertical half of the
    /// director is measured against.
    pub vertical_speed_m_s: f64,
    /// Local gravitational acceleration (m/s²) for the turn-rate ↔ bank relation.
    pub gravity_m_s2: f64,
    /// Bank the command is clamped to (rad).
    pub bank_limit_rad: f64,
    /// The craft's current bank (rad, + = right wing down).
    ///
    /// Needed so the lateral director can be a *roll* cue — the difference
    /// between the bank the follower wants and the bank the craft has — rather
    /// than a heading cue that silently disagrees with the autopilot on every
    /// turn. See [`Guidance::director_lateral`].
    pub bank_rad: f64,
    /// Last frame's along-track position (m), so the projection onto the route
    /// follows the craft instead of jumping legs where the route doubles back —
    /// see [`crate::path::LateralPath::closest_from`]. `None` seeds a fresh plan.
    ///
    /// The caller holds it because this function stays pure
    /// (ADR-20260730T005746Z), the same arrangement as the rejoin's capture hint.
    pub track_hint_along_m: Option<f64>,
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
    /// Track the craft should fly now (compass rad) — **the steering answer**.
    /// The active path's own course at the closest point, rotated by a
    /// lookahead intercept proportional to how far off it the craft is.
    ///
    /// Note this is the *aim*, not the whole command: holding a curved segment
    /// also costs a standing turn rate, which is why `bank_command_rad` is not
    /// simply a function of this heading (see the follower in
    /// [`compute_guidance`]).
    pub desired_heading_rad: f64,
    /// The craft's current flight-path angle (rad, + = climbing).
    pub fpa_rad: f64,
    /// The flight-path angle to fly now (rad, + = climbing) — the vertical half
    /// of the steering answer, derived from the profile's vertical-speed command.
    pub fpa_command_rad: f64,
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
    /// Commanded bank (rad, + = right), clamped to the bank limit. **The single
    /// lateral command**: LAND flies it, the ND's steering dot shows it against
    /// the craft's actual bank, and the PFD's flight-director bar does the same.
    pub bank_command_rad: f64,
    /// The craft's current bank (rad, + = right wing down), echoed from the
    /// input so a display reading this struct never has to fetch attitude from a
    /// second source to interpret [`Self::bank_command_rad`].
    pub bank_rad: f64,
    /// Commanded vertical speed (m/s, + = up).
    pub vertical_speed_command_m_s: f64,
    /// Whether the craft is inside both full-scale deflections — "established".
    pub established: bool,
}

impl Guidance {
    /// Lateral steering deflection in `[-1, 1]`, **positive = roll right**.
    ///
    /// **This is the autopilot's own command, shown to a human.** It is the
    /// difference between the bank the follower would fly
    /// ([`Self::bank_command_rad`]) and the bank the craft actually has, so
    /// centring the cue by hand flies the trajectory LAND would have flown. One
    /// law, two consumers.
    ///
    /// It used to be a pure heading-error cue, which is a *different* law and a
    /// quietly wrong one on any curved leg: holding an arc costs a standing bank
    /// before there is any heading error to see, so a pilot who kept the dot
    /// centred through a turn flew straight ahead and slid off the outside of
    /// it. That is exactly the defect that was fixed in the autopilot's lateral
    /// law (INC-20260801T035551Z) and it survived here, in the cue the player
    /// actually flies, until the two were unified.
    ///
    /// Centred still means "you are doing the right thing": on a straight leg
    /// the command is zero bank, and on a curve it is the turn's own bank.
    pub fn director_lateral(&self) -> f64 {
        ((self.bank_command_rad - self.bank_rad) / DIRECTOR_BANK_FULL_SCALE_RAD).clamp(-1.0, 1.0)
    }

    /// Vertical steering deflection in `[-1, 1]`, **positive = pitch up**.
    pub fn director_vertical(&self) -> f64 {
        ((self.fpa_command_rad - self.fpa_rad) / DIRECTOR_FPA_FULL_SCALE_RAD).clamp(-1.0, 1.0)
    }

    /// Localizer deflection as a fraction of full scale, clamped to `[-1, 1]`.
    pub fn loc_deflection(&self) -> f64 {
        (self.loc_deviation_rad / LOC_FULL_SCALE_RAD).clamp(-1.0, 1.0)
    }

    /// Glideslope deflection as a fraction of full scale, clamped to `[-1, 1]`.
    pub fn gs_deflection(&self) -> f64 {
        (self.gs_deviation_rad / GS_FULL_SCALE_RAD).clamp(-1.0, 1.0)
    }
}

/// Compute guidance for `plan` at the craft state in `input`, tracking `path`.
///
/// `path` is the **active** route: normally `plan.path`, and after a rejoin has
/// been committed, `plan.path.splice_rejoin(...)`. It is passed separately
/// precisely so there is only ever one answer to "where should I go" — the
/// thing being followed is the thing being drawn. `plan` still supplies
/// everything runway-relative: the aim point, the ILS deviations, and the
/// vertical profile.
///
/// There is deliberately no steering-cue parameter any more. A rejoin used as a
/// cue is a second path authority that disagrees with the drawn route by
/// design; splicing it into `path` removes the disagreement instead of
/// annotating it (INC-20260801T035551Z).
///
/// Returns `None` only when `path` is empty (which
/// [`crate::approach::plan_approach`] never produces).
pub fn compute_guidance(
    plan: &ApproachPlan,
    path: &crate::path::LateralPath,
    input: &GuidanceInput,
) -> Option<Guidance> {
    let frame = &plan.frame;
    let local = frame.to_local(input.position_body_fixed);
    let altitude_m = frame.altitude_of(input.position_body_fixed);

    let closest = path.closest_from(local, input.track_hint_along_m)?;
    let dtg_m = (path.length() - closest.along_m).max(0.0);

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

    // --- Lateral steering: a path *follower*, not a path aimer.
    //
    // Two terms, and the first is the one that was missing. Holding a curved
    // segment costs a standing turn rate `v·κ` before any error correction
    // exists; a law built only on heading error cannot produce that rate until
    // it has first grown an error, which is a standing cross-track offset by
    // construction. On a 3 km-radius join that offset was kilometres
    // (INC-20260801T035551Z).
    //
    // The correction term is the classic one: positive cross-track means the
    // craft is RIGHT of course, so the desired track rotates to the LEFT of the
    // course — and left is CCW, i.e. a positive θ offset (see
    // `crate::waypoint`). Flipping this sign turns the capture law into a
    // divergence law.
    let v = input.ground_speed_m_s.max(1.0);
    let lookahead = (input.ground_speed_m_s.max(0.0) * LOOKAHEAD_TIME_S).max(MIN_LOOKAHEAD_M);
    let intercept = (closest.cross_track_m / lookahead)
        .atan()
        .clamp(-MAX_INTERCEPT_RAD, MAX_INTERCEPT_RAD);
    let desired_theta = closest.theta + intercept;
    let heading_error = crate::wrap_angle(desired_theta - track_theta);

    let feedforward_rate = v * closest.curvature;
    let correction_rate = (TRACK_GAIN_PER_S * heading_error)
        .clamp(-MAX_CORRECTION_RATE_PER_S, MAX_CORRECTION_RATE_PER_S);

    // Turn rate → bank: `tan φ = ω·v/g`.
    let bank_command_rad = if input.gravity_m_s2 > 0.0 {
        let omega = feedforward_rate + correction_rate;
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

    // Flight-path angles, from vertical speed over ground speed. Guarded at low
    // ground speed, where the ratio is meaningless rather than merely large.
    let speed_floor = input.ground_speed_m_s.max(1.0);
    let fpa_rad = (input.vertical_speed_m_s / speed_floor)
        .clamp(-1.0, 1.0)
        .asin();
    let fpa_command_rad = (vertical_speed_command_m_s / speed_floor)
        .clamp(-1.0, 1.0)
        .asin();

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

    // Phase from distance-to-go, not from along-track. A spliced rejoin shifts
    // every along-track distance and leaves every distance-to-go alone, so this
    // is the form that means the same thing on the plain route and the amended
    // one.
    let phase = if dtg_m <= plan.aim_inset_m {
        ApproachPhase::Touchdown
    } else if dtg_m <= plan.final_dtg_m() + 1.0 {
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
        fpa_rad,
        fpa_command_rad,
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
        bank_rad: input.bank_rad,
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
            // On a 3-degree slope at 80 m/s the craft is sinking ~4.2 m/s; the
            // helper's states are otherwise "on profile", so match it.
            vertical_speed_m_s: -80.0 * 3.0_f64.to_radians().tan(),
            gravity_m_s2: crate::EARTH_G_M_S2,
            bank_limit_rad: 25.0_f64.to_radians(),
            bank_rad: 0.0,
            track_hint_along_m: None,
        }
    }

    fn on_slope_altitude(sc: &Scene, dist: f64) -> f64 {
        sc.plan.vertical.target_altitude_m(dist)
    }

    #[test]
    fn on_course_and_on_slope_reads_zero_everywhere() {
        let sc = scene();
        let g = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, 5_000.0, 0.0, on_slope_altitude(&sc, 5_000.0)),
        )
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
        let g = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, dist, 400.0, on_slope_altitude(&sc, dist)),
        )
        .expect("guidance");
        assert!(g.cross_track_m > 0.0, "right of course is positive");
        assert!(g.loc_deviation_rad > 0.0, "localizer reads right");
        assert!(
            g.bank_command_rad < 0.0,
            "must roll LEFT to correct a right offset, got {}",
            g.bank_command_rad.to_degrees()
        );
        // Mirrored on the other side.
        let g2 = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, dist, -400.0, on_slope_altitude(&sc, dist)),
        )
        .expect("guidance");
        assert!(g2.cross_track_m < 0.0);
        assert!(g2.bank_command_rad > 0.0, "must roll right");
        assert_abs_diff_eq!(g.bank_command_rad, -g2.bank_command_rad, epsilon = 1e-9);
    }

    #[test]
    fn the_director_points_where_the_craft_should_turn() {
        let sc = scene();
        let dist = 6_000.0;
        // Right of course: the director must call for a LEFT turn.
        let right = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, dist, 500.0, on_slope_altitude(&sc, dist)),
        )
        .expect("g");
        assert!(
            right.director_lateral() < 0.0,
            "right of course should steer left, got {}",
            right.director_lateral()
        );
        let left = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, dist, -500.0, on_slope_altitude(&sc, dist)),
        )
        .expect("g");
        assert!(left.director_lateral() > 0.0);
        // On course and aligned: centred.
        let on = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, dist, 0.0, on_slope_altitude(&sc, dist)),
        )
        .expect("g");
        assert_abs_diff_eq!(on.director_lateral(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn the_vertical_director_points_where_the_craft_should_pitch() {
        let sc = scene();
        let dist = 6_000.0;
        let on_slope = on_slope_altitude(&sc, dist);
        // Low: the profile calls for less sink than the craft has, so the cue
        // says pitch up.
        let low = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, dist, 0.0, on_slope - 200.0),
        )
        .expect("g");
        assert!(
            low.director_vertical() > 0.0,
            "low should steer up, got {}",
            low.director_vertical()
        );
        let high = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, dist, 0.0, on_slope + 200.0),
        )
        .expect("g");
        assert!(high.director_vertical() < 0.0, "high should steer down");
        // On profile with the matching sink rate: centred.
        let on =
            compute_guidance(&sc.plan, &sc.plan.path, &state(&sc, dist, 0.0, on_slope)).expect("g");
        assert_abs_diff_eq!(on.director_vertical(), 0.0, epsilon = 0.02);
    }

    #[test]
    fn the_director_saturates_rather_than_running_away() {
        let sc = scene();
        let mut st = state(&sc, 8_000.0, 30_000.0, 4_000.0);
        st.track_dir_body_fixed = -sc.plan.end.landing_dir();
        st.vertical_speed_m_s = -60.0;
        let g = compute_guidance(&sc.plan, &sc.plan.path, &st).expect("g");
        assert!(g.director_lateral().abs() <= 1.0);
        assert!(g.director_vertical().abs() <= 1.0);
    }

    #[test]
    fn a_spliced_rejoin_puts_the_craft_on_the_path_it_is_drawn_flying() {
        // The old arrangement fed the rejoin in as a steering *cue* while still
        // measuring cross-track against the untouched route, so the numbers said
        // "4 km off course" for the whole time the craft was correctly flying
        // back. Spliced, the rejoin is the front of the route: the craft is on
        // the path, and the path leads to the runway.
        let sc = scene();
        let dist = 12_000.0;
        let st = state(&sc, dist, 4_000.0, on_slope_altitude(&sc, dist));
        let local = sc.plan.frame.to_local(st.position_body_fixed);
        let track = sc
            .plan
            .frame
            .direction_to_local(st.track_dir_body_fixed)
            .normalize();
        let pose = crate::dubins::Pose2::new(local, crate::waypoint::theta_of(track));
        let closest = sc.plan.path.closest(local).expect("on path");
        let params = crate::rejoin::RejoinParams::for_radius(1_400.0);
        let rejoin =
            crate::rejoin::plan_rejoin(&sc.plan.path, pose, closest.along_m, &params, None)
                .expect("rejoin");
        let spliced = sc
            .plan
            .path
            .splice_rejoin(rejoin.path.clone(), rejoin.capture_along_m);

        let on_route = compute_guidance(&sc.plan, &sc.plan.path, &st).expect("g");
        let on_spliced = compute_guidance(&sc.plan, &spliced, &st).expect("g");

        // Against the bare route the craft is kilometres off. Against the route
        // it is actually flying, it is on it.
        assert!(on_route.cross_track_m.abs() > 3_000.0);
        assert!(
            on_spliced.cross_track_m.abs() < 50.0,
            "the craft sits at the head of the spliced path, got {} m",
            on_spliced.cross_track_m
        );
        // And the splice does not invent or destroy distance to the runway: the
        // rejoin's length is the whole of the difference.
        let added = on_spliced.dtg_m - (sc.plan.length_m() - rejoin.capture_along_m);
        assert!(
            (added - rejoin.length_m).abs() < 1.0,
            "dtg gained {added} m but the rejoin is {} m",
            rejoin.length_m
        );
    }

    #[test]
    fn the_hand_flown_cue_commands_the_same_roll_the_autopilot_flies() {
        // The point of unifying them: a pilot who centres the dot flies what
        // LAND would have flown. So the cue must be zero exactly when the craft
        // is already at the commanded bank, and must call for the *difference*
        // otherwise — including on a turn, where the command is nonzero even
        // with no tracking error at all.
        let sc = scene();
        let dist = 6_000.0;
        let mut st = state(&sc, dist, 900.0, on_slope_altitude(&sc, dist));

        // Wings level while a correction is wanted: the cue calls for roll.
        st.bank_rad = 0.0;
        let level = compute_guidance(&sc.plan, &sc.plan.path, &st).expect("g");
        assert!(level.bank_command_rad < 0.0, "right of course rolls left");
        assert!(
            level.director_lateral() < 0.0,
            "the cue must call for the same left roll, got {}",
            level.director_lateral()
        );

        // Already banked at the command: the cue is centred, because the pilot
        // is doing exactly what the autopilot would.
        st.bank_rad = level.bank_command_rad;
        let matched = compute_guidance(&sc.plan, &sc.plan.path, &st).expect("g");
        assert_abs_diff_eq!(matched.director_lateral(), 0.0, epsilon = 1e-9);

        // Banked the wrong way: the cue saturates toward the correction.
        st.bank_rad = 20.0_f64.to_radians();
        let wrong = compute_guidance(&sc.plan, &sc.plan.path, &st).expect("g");
        assert!(wrong.director_lateral() < level.director_lateral());
        assert!(wrong.director_lateral() >= -1.0);
    }

    #[test]
    fn on_a_turn_the_cue_is_not_centred_by_flying_straight() {
        // The defect this unification removes. The old cue was heading error
        // only, so a pilot holding it centred through a curved leg flew wings
        // level and slid off the outside of the turn — the same standing-offset
        // failure the autopilot had, left behind in the human's instrument.
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        let craft = frame.to_body_fixed(DVec2::new(14_000.0, -11_000.0), 2_500.0);
        let plan =
            plan_approach(end, craft, end.landing_dir(), &ApproachParams::default()).expect("plan");
        let arc = (0..400)
            .map(|i| plan.path.length() * i as f64 / 400.0)
            .filter_map(|s| plan.path.point_at(s))
            .find(|p| p.curvature.abs() > 1e-6)
            .expect("the plan turns somewhere");

        // Exactly on the arc, exactly along it, wings level — every *error* is
        // zero, and the cue must still say "roll".
        let st = GuidanceInput {
            position_body_fixed: frame.to_body_fixed(arc.position, 2_000.0),
            track_dir_body_fixed: {
                let d = crate::waypoint::dir_from_theta(arc.theta);
                frame.east * d.x + frame.north * d.y
            },
            ground_speed_m_s: 90.0,
            vertical_speed_m_s: 0.0,
            gravity_m_s2: crate::EARTH_G_M_S2,
            bank_limit_rad: 25.0_f64.to_radians(),
            bank_rad: 0.0,
            track_hint_along_m: Some(arc.along_m),
        };
        let g = compute_guidance(&plan, &plan.path, &st).expect("g");
        assert_abs_diff_eq!(g.cross_track_m, 0.0, epsilon = 1.0);
        assert!(
            g.director_lateral().abs() > 0.1,
            "wings level on a {:.0} m arc must not read as centred, got {}",
            1.0 / arc.curvature.abs(),
            g.director_lateral()
        );
    }

    #[test]
    fn following_a_turn_costs_bank_before_any_error_exists() {
        // The defect the feedforward term exists for: perfectly on a curved
        // segment, a heading-error-only law commands zero bank and immediately
        // falls outside the turn. A follower banks into it.
        let s = strip();
        let end = s.ends()[0];
        let frame = end.route_frame().expect("valid");
        // Start well off to the side so the plan contains real turning arcs.
        let craft = frame.to_body_fixed(DVec2::new(14_000.0, -11_000.0), 2_500.0);
        let plan =
            plan_approach(end, craft, end.landing_dir(), &ApproachParams::default()).expect("plan");

        // Walk the route and find a point on an arc; sit exactly on it, tracking
        // exactly along it, so every error term is zero by construction.
        let arc = (0..400)
            .map(|i| plan.path.length() * i as f64 / 400.0)
            .filter_map(|s| plan.path.point_at(s))
            .find(|p| p.curvature.abs() > 1e-6)
            .expect("the plan turns somewhere");
        let st = GuidanceInput {
            position_body_fixed: frame.to_body_fixed(arc.position, 2_000.0),
            track_dir_body_fixed: {
                let d = crate::waypoint::dir_from_theta(arc.theta);
                frame.east * d.x + frame.north * d.y
            },
            ground_speed_m_s: 90.0,
            vertical_speed_m_s: 0.0,
            gravity_m_s2: crate::EARTH_G_M_S2,
            bank_limit_rad: 25.0_f64.to_radians(),
            bank_rad: 0.0,
            track_hint_along_m: Some(arc.along_m),
        };
        let g = compute_guidance(&plan, &plan.path, &st).expect("g");
        assert!(
            g.cross_track_m.abs() < 1.0,
            "test setup: should be on the path, got {} m",
            g.cross_track_m
        );
        // A left (CCW, positive curvature) arc must command left (negative) bank.
        assert!(
            g.bank_command_rad.abs() > 5.0_f64.to_radians(),
            "on a {:.0} m-radius arc the follower must already be banking, got {:.1}°",
            1.0 / arc.curvature.abs(),
            g.bank_command_rad.to_degrees()
        );
        assert_eq!(
            g.bank_command_rad < 0.0,
            arc.curvature > 0.0,
            "bank must lean into the turn"
        );
    }

    #[test]
    fn bank_command_respects_the_limit() {
        let sc = scene();
        // Wildly off course and pointing the wrong way.
        let mut st = state(&sc, 8_000.0, 20_000.0, 3_000.0);
        st.track_dir_body_fixed = -sc.plan.end.landing_dir();
        st.ground_speed_m_s = 250.0;
        let g = compute_guidance(&sc.plan, &sc.plan.path, &st).expect("guidance");
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
        let high = compute_guidance(&sc.plan, &sc.plan.path, &state(&sc, dist, 0.0, on + 150.0))
            .expect("g");
        assert!(high.gs_deviation_rad > 0.0, "high reads positive");
        assert!(high.altitude_error_m > 0.0);
        let low = compute_guidance(&sc.plan, &sc.plan.path, &state(&sc, dist, 0.0, on - 150.0))
            .expect("g");
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
        let g = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, dist, 0.0, on_slope_altitude(&sc, dist)),
        )
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
            &sc.plan.path,
            &state(&sc, 15_000.0, 60.0, on_slope_altitude(&sc, 15_000.0)),
        )
        .expect("g");
        let near = compute_guidance(
            &sc.plan,
            &sc.plan.path,
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
            vertical_speed_m_s: 0.0,
            gravity_m_s2: crate::EARTH_G_M_S2,
            bank_limit_rad: 25.0_f64.to_radians(),
            bank_rad: 0.0,
            track_hint_along_m: None,
        };
        assert_eq!(
            compute_guidance(&sc.plan, &sc.plan.path, &at_start)
                .expect("g")
                .phase,
            ApproachPhase::Transition
        );
        let on_final = state(&sc, 4_000.0, 0.0, on_slope_altitude(&sc, 4_000.0));
        assert_eq!(
            compute_guidance(&sc.plan, &sc.plan.path, &on_final)
                .expect("g")
                .phase,
            ApproachPhase::Final
        );
        // Over the strip, past the aim point.
        let over = state(&sc, -100.0, 0.0, sc.plan.frame.origin_altitude_m + 2.0);
        assert_eq!(
            compute_guidance(&sc.plan, &sc.plan.path, &over)
                .expect("g")
                .phase,
            ApproachPhase::Touchdown
        );
    }

    #[test]
    fn threshold_range_and_dtg_are_different_quantities() {
        let sc = scene();
        // Offset far to the side: route distance-to-go must exceed the straight
        // line to the threshold is NOT guaranteed, but they must not be equal.
        let g = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, 3_000.0, 2_000.0, 1_200.0),
        )
        .expect("g");
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
        let g = compute_guidance(&sc.plan, &sc.plan.path, &st).expect("g");
        assert!(g.bank_command_rad.is_finite());
        assert!(g.vertical_speed_command_m_s.is_finite());
        assert!(g.desired_heading_rad.is_finite());
        assert!(g.loc_deviation_rad.is_finite());
    }

    #[test]
    fn headings_are_compass_and_agree_with_the_runway() {
        let sc = scene();
        let dist = 6_000.0;
        let g = compute_guidance(
            &sc.plan,
            &sc.plan.path,
            &state(&sc, dist, 0.0, on_slope_altitude(&sc, dist)),
        )
        .expect("g");
        // The strip points due north, so course/track/desired are all ~0/360.
        for h in [
            g.course_heading_rad,
            g.track_heading_rad,
            g.desired_heading_rad,
        ] {
            let deg = h.to_degrees();
            assert!(
                !(1.0..=359.0).contains(&deg),
                "expected a northerly heading, got {deg}"
            );
        }
        assert_eq!(sc.plan.designator, 36, "due north is RWY 36");
    }
}
