//! Route navigation: what the pilot has selected to fly to, and the live
//! guidance to it.
//!
//! This is the Bevy glue around [`thalos_navigation`]. The pure crate owns the
//! geometry (approach paths, vertical profiles, deviations); this module owns the
//! *selection* (which runway end is armed), the *policy* (when a plan is
//! recomputed), and the *publication* of one [`RouteState`] that every consumer
//! reads — the ND widget, the PFD's deviation scales, and, later, the autoland
//! autopilot.
//!
//! **Not to be confused with [`crate::navigation`]**, which despite the name is
//! the attitude/SAS pointing-mode state (prograde, retrograde, target hold).
//! That module predates this one and is queued for a rename
//! (`docs/backlog.md`); nothing here touches attitude.
//!
//! # One authority, many projections
//!
//! `RouteState` is the single source of "where are we going and how are we
//! doing". Displays never re-plan and never re-derive a deviation: an ND that
//! computed its own approach path would eventually disagree with the PFD needle
//! the pilot is following, and the disagreement would be invisible until
//! someone flew into terrain over it.
//!
//! # One path: [`RouteState::active_path`]
//!
//! There is exactly one answer to "where should I go", and it is the line on the
//! ND. `plan.path` is the *nominal* route to the runway; `active_path` is what
//! the craft flies, drawn, and measured against — the plan with any committed
//! rejoin spliced onto its front ([`maybe_commit_rejoin`]).
//!
//! This used to be two things. The route was drawn while a per-frame rejoin was
//! flown, so the aircraft visibly took different turns from the path on the
//! display, and cross-track reported it as kilometres off course the whole time
//! it was correctly flying back. Splicing removes the disagreement rather than
//! annotating it (INC-20260801T035551Z).
//!
//! # When a plan is recomputed (and when it must not be)
//!
//! The plan is *not* rebuilt every frame — a path that jitters as airspeed
//! wobbles is unflyable and unreadable. It is rebuilt when the selection
//! changes, and while still maneuvering, when the craft has drifted more than
//! [`REPLAN_CROSS_TRACK_M`] from the planned path (rate-limited by
//! [`REPLAN_MIN_INTERVAL_S`]).
//!
//! **Once the craft passes the final approach point the plan freezes**
//! ([`RouteState::plan_frozen`]). This is not an optimisation: re-planning from
//! a position *past* the FAP would ask the Dubins planner to fly back to a fix
//! behind the craft, which it solves with a full turn-around — the plan would
//! loop the aircraft away from the runway it is 3 km from.
//!
//! **Frozen is not the same as established**, and the two must never share a
//! flag. Freezing is geometric and irreversible; being established is a live
//! statement about the needles that can stop being true. When one flag meant
//! both, an approach that reached the FAP 1.8 km off the centreline froze its
//! plan, declared itself established, and flew the unrecoverable result down to
//! 250 m before anything objected. [`RouteState::established`] is now the
//! honest needle test, and it is what the stabilisation gate in
//! [`crate::route_autopilot`] reads.

use bevy::math::{DVec2, DVec3};
use bevy::prelude::*;

use thalos_navigation::{
    ApproachParams, ApproachPhase, ApproachPlan, DestinationInput, DestinationParams,
    GuidanceInput, LateralPath, Pose2, RejoinParams, RouteFrame, RunwayStrip, VnavParams,
    angular_distance_rad, compute_destination_guidance, compute_guidance, plan_approach,
    plan_rejoin, theta_of,
};
use thalos_physics_canonical::body_fixed::inertial_to_body_fixed;
use thalos_physics_canonical::terrain_provider::TerrainProvider;
use thalos_world::BodyId;

use crate::GameTerrainRegistry;
use crate::aero::ShipAero;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::structures::{StructureKind, StructureRegistry};

pub use thalos_game_state::nav::{
    ArmedEnd, BANK_LIMIT_RAD, RouteDisplay, RouteRequest, RouteSelection, RouteState, RouteStatus,
    RunwayEndEntry,
};

// 25°

/// Never plan a turn tighter than this (m), whatever the speed suggests.
const MIN_TURN_RADIUS_M: f64 = 400.0;
/// Straight final approach segment length (m). At a 3° glideslope this puts the
/// capture altitude ~470 m above the threshold — a sane pattern altitude.
const FINAL_LENGTH_M: f64 = 9_000.0;
/// Aim point inset past the threshold (m). Also sets the threshold crossing
/// height: 450 m × tan 3° ≈ 24 m. The extra margin is deliberate: the recorded
/// approach tracked ~4 m below profile and flared to a touchdown only ~150 m
/// into the strip when aimed at 300 m. This moves the expected contact into the
/// touchdown zone without changing the glideslope or flare law.
const AIM_INSET_M: f64 = 450.0;
/// Shortest stabilised straight run onto the aim point the planner leaves itself
/// when the craft is already inside the final corridor (m) — see
/// `ApproachParams::min_capture_run_m`.
const MIN_CAPTURE_RUN_M: f64 = 1_200.0;
/// LAND hands its spherical ingress to the local terminal planner inside this
/// surface distance from the arrival fix.
const TERMINAL_CAPTURE_RANGE_M: f64 = 60_000.0;
/// Arrival fix behind the threshold. This gives the local planner room to make
/// a bank-limited join before the 9 km straight final.
const ARRIVAL_FIX_BEHIND_THRESHOLD_M: f64 = 35_000.0;
/// Conservative clearance over the body's published maximum elevation.
const DESTINATION_TERRAIN_CLEARANCE_M: f64 = 2_500.0;
const DESTINATION_DESCENT_DISTANCE_M: f64 = 180_000.0;
const DESTINATION_MAX_VS_M_S: f64 = 15.0;
/// Approach speed used when the craft's own stall speed cannot be derived (no
/// atmosphere sample, or a craft with no lift curve). Matches the short-final
/// spawn speed in [`crate::runway`].
const FALLBACK_APPROACH_SPEED_M_S: f64 = 80.0;
/// Multiple of stall speed flown on approach — the standard 1.3 Vs margin.
const APPROACH_STALL_MARGIN: f64 = 1.3;

/// Cross-track drift (m) that triggers a re-plan while still maneuvering **and
/// no rejoin can be flown**.
///
/// The qualifier is the whole point. Cross-track is measured against the route,
/// and a rejoin is by construction a path that leaves the route to get back onto
/// it — a bank-limited reversal swings the craft a full turn diameter clear.
/// Treating that as "drifted" made the two mechanisms fight: the rejoin flew the
/// craft out, the drift test called it lost, the plan was rebuilt from the
/// craft's current position, and the cycle repeated every ~47 s. That is the
/// loop in the recorded flight, and it is why the ND kept redrawing.
///
/// So drift only re-plans when the rejoin planner has already said there is no
/// flyable way back. That is the honest "this plan is unreachable" signal.
const REPLAN_CROSS_TRACK_M: f64 = 2_000.0;
/// Cross-track (m) at which the plan is rebuilt even though a rejoin exists.
///
/// A backstop for the pathological case only — a rejoin technically plannable
/// from 20 km away is a worse idea than a fresh approach. Sized well beyond any
/// legitimate reversal: a 25° bank at 130 m/s reverses inside a 7 km diameter.
const REPLAN_UNREACHABLE_CROSS_TRACK_M: f64 = 15_000.0;
/// Cross-track drift (m) off the **active** path that commits a rejoin into it.
///
/// Well above the tracking error a working follower leaves (tens of metres), so
/// normal flight never amends the route; well below the re-plan threshold, so
/// the cheap fix (fly back onto this route) is always tried before the expensive
/// one (build a different route).
const REJOIN_COMMIT_CROSS_TRACK_M: f64 = 400.0;
/// Minimum gap between committed rejoins (s). Committing is a decision the craft
/// then spends a minute or two *flying*; re-deciding at frame rate is what made
/// the old per-frame rejoin a target the craft could never reach.
const REJOIN_COMMIT_MIN_INTERVAL_S: f32 = 20.0;
/// Minimum wall-clock gap between automatic re-plans (s).
const REPLAN_MIN_INTERVAL_S: f32 = 2.0;

/// Ground speed (m/s) below which the craft's own track direction is unusable
/// and the nose is used instead.
const MIN_TRACK_SPEED_M_S: f64 = 2.0;

/// Body nose axis, matching the navball / PFD / ND conventions.
const BODY_NOSE: DVec3 = DVec3::Y;

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Published state
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct RoutePlugin;

impl Plugin for RoutePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RouteSelection>()
            .init_resource::<RouteState>()
            .add_message::<RouteRequest>()
            .add_systems(
                Update,
                // Requests first so a click acts on the same frame it happens,
                // then the state rebuild — which is what every display reads,
                // so it must be settled before the HUD systems run.
                (apply_route_requests, update_route_state)
                    .chain()
                    .after(crate::SimStage::Sync),
            );
    }
}

/// **Sole writer** of [`RouteSelection`].
fn apply_route_requests(
    mut requests: MessageReader<RouteRequest>,
    state: Res<RouteState>,
    mut selection: ResMut<RouteSelection>,
) {
    for request in requests.read() {
        let next = match *request {
            RouteRequest::Pick(strip) => match selection.armed {
                // Clicking the armed strip again flips the landing direction.
                Some(armed) if armed.strip == strip => Some(ArmedEnd {
                    strip,
                    reciprocal: !armed.reciprocal,
                }),
                // Otherwise take whichever end of that strip is listed first —
                // `ends` is sorted by threshold range, so that is the end whose
                // threshold the craft is nearer, i.e. the shorter approach.
                _ => state
                    .ends
                    .iter()
                    .find(|e| e.armed_end.strip == strip)
                    .map(|e| e.armed_end),
            },
            RouteRequest::Cycle(step) => {
                if state.ends.is_empty() {
                    None
                } else {
                    let len = state.ends.len() as i32;
                    let current = selection
                        .armed
                        .and_then(|a| state.ends.iter().position(|e| e.armed_end == a))
                        .map(|i| i as i32);
                    let next_index = match current {
                        Some(i) => (i + step).rem_euclid(len),
                        // Nothing armed: step forward from the nearest end.
                        None if step >= 0 => 0,
                        None => len - 1,
                    };
                    Some(state.ends[next_index as usize].armed_end)
                }
            }
            RouteRequest::Flip => selection.armed.map(|a| ArmedEnd {
                strip: a.strip,
                reciprocal: !a.reciprocal,
            }),
            RouteRequest::Clear => None,
        };
        if selection.armed != next {
            selection.armed = next;
        }
    }
}

/// **Sole writer** of [`RouteState`]: enumerate runway ends, maintain the plan
/// under the re-plan policy, and publish live guidance.
fn update_route_state(
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    terrain: Res<GameTerrainRegistry>,
    structures: Res<StructureRegistry>,
    ship_aero_q: Query<&ShipAero, With<thalos_physics_local::LocalCraftBody>>,
    time: Res<Time<Real>>,
    mut selection: ResMut<RouteSelection>,
    mut state: ResMut<RouteState>,
) {
    let s = &sim.simulation;
    let dominant = s.dominant_body();
    let Some(body_state) = solar.states.as_deref().and_then(|st| st.get(dominant)) else {
        clear_plan(&mut state, RouteStatus::Unavailable);
        return;
    };

    // Craft pose/velocity in the body-fixed frame, through the one canonical
    // conversion — never a hand-rolled rotation here.
    let craft = s.craft_state();
    let frame_state = inertial_to_body_fixed(body_state, craft.translation, craft.attitude);
    let position_bf = frame_state.translation_body.position;
    let Some(up) = position_bf.try_normalize() else {
        clear_plan(&mut state, RouteStatus::Unavailable);
        return;
    };

    // Ground track: the horizontal part of the surface-relative velocity, or the
    // nose when barely moving (a parked craft has no track).
    // The craft's own bank, so the steering dot can show the follower's roll
    // command against it. Same formula as `thalos_control::FlightState::bank` —
    // the up vector rotated into body axes — evaluated here because this is
    // where the body-fixed attitude already is.
    let up_body = frame_state.orientation_body.inverse() * up;
    let bank_rad = (-up_body.x).atan2(up_body.z);

    let velocity_bf = frame_state.translation_body.velocity;
    let horizontal = velocity_bf - up * velocity_bf.dot(up);
    let ground_speed_m_s = horizontal.length();
    let track_dir_bf = if ground_speed_m_s >= MIN_TRACK_SPEED_M_S {
        horizontal / ground_speed_m_s
    } else {
        let nose_bf = frame_state.orientation_body * BODY_NOSE;
        (nose_bf - up * nose_bf.dot(up))
            .try_normalize()
            .unwrap_or(up)
    };

    // --- Enumerate every landable end, nearest threshold first.
    let body = &sim.system.bodies[dominant];
    let body_radius_m = body.radius_m;
    state.ends.clear();
    for site in structures.sites_on(dominant) {
        let StructureKind::Runway {
            half_length_m,
            half_width_m,
        } = site.kind
        else {
            continue;
        };
        let strip = RunwayStrip {
            id: site.id.0,
            center_dir: site.anchor_dir,
            heading_tangent: site.heading_tangent,
            half_length_m: half_length_m as f64,
            half_width_m: half_width_m as f64,
            // A draped strip inherits its parent basin's level — see
            // `StructureRegistry::site_elevation_m`.
            elevation_m: structures.site_elevation_m(site),
            body_radius_m,
        };
        for end in strip.ends() {
            let Some(route_frame) = end.route_frame() else {
                continue;
            };
            state.ends.push(RunwayEndEntry {
                armed_end: ArmedEnd {
                    strip: site.id,
                    reciprocal: end.reciprocal,
                },
                end,
                designator: end.designator(&route_frame),
                landing_heading_rad: end.landing_heading_rad(&route_frame),
                threshold_range_m: position_bf.distance(end.threshold_point()),
            });
        }
    }
    state
        .ends
        .sort_by(|a, b| a.threshold_range_m.total_cmp(&b.threshold_range_m));

    if state.ends.is_empty() {
        clear_plan(&mut state, RouteStatus::NoRunways);
        // A stale selection on a body with no runways cannot be flown.
        if selection.armed.is_some() {
            selection.armed = None;
        }
        return;
    }

    let Some(armed) = selection.armed else {
        clear_plan(&mut state, RouteStatus::Idle);
        return;
    };
    // The armed strip may have been removed (editor deletion, body change).
    let Some(entry) = state.ends.iter().find(|e| e.armed_end == armed).copied() else {
        clear_plan(&mut state, RouteStatus::Idle);
        selection.armed = None;
        return;
    };

    state.armed = Some(entry);

    // --- Approach speed from the craft's own stall speed at the threshold.
    let approach_speed_m_s = approach_speed(
        &sim,
        dominant,
        entry.end.strip.elevation_m,
        ship_aero_q.iter().next(),
    );
    state.approach_speed_m_s = approach_speed_m_s;

    let params = ApproachParams {
        bank_limit_rad: BANK_LIMIT_RAD,
        gravity_m_s2: body.gm / (body_radius_m + entry.end.strip.elevation_m).powi(2),
        // Plan the turns for how fast the craft is actually going; a fast craft
        // needs a wider intercept than its approach speed would suggest.
        maneuver_speed_m_s: ground_speed_m_s.max(approach_speed_m_s),
        min_turn_radius_m: MIN_TURN_RADIUS_M,
        final_length_m: FINAL_LENGTH_M,
        aim_inset_m: AIM_INSET_M,
        min_capture_run_m: MIN_CAPTURE_RUN_M,
        vnav: VnavParams {
            approach_speed_m_s,
            final_dtg_m: FINAL_LENGTH_M + AIM_INSET_M,
            ..VnavParams::default()
        },
    };

    // The local approach frame is intentionally not global. Outside the
    // terminal region publish spherical same-body destination guidance to an
    // arrival fix behind the runway, then build the precise local plan only
    // after the craft reaches it.
    let Some(runway_frame) = entry.end.route_frame() else {
        clear_plan(&mut state, RouteStatus::Unavailable);
        return;
    };
    let landing_dir_local = runway_frame
        .direction_to_local(entry.end.landing_dir())
        .normalize_or_zero();
    let arrival_dir = runway_frame
        .to_body_fixed(
            -landing_dir_local * ARRIVAL_FIX_BEHIND_THRESHOLD_M,
            entry.end.strip.elevation_m,
        )
        .normalize_or_zero();
    let arrival_range_m = angular_distance_rad(up, arrival_dir) * body_radius_m;
    let selection_changed = state.planned_for != Some(armed);
    if selection_changed {
        state.plan_frozen = false;
        state.established = false;
        state.plan = None;
        state.guidance = None;
        state.display = RouteDisplay::default();
        state.track_hint_along_m = None;
    }
    let recovery_ingress = state.force_destination_ingress && arrival_range_m > 5_000.0;
    if !state.plan_frozen && (arrival_range_m > TERMINAL_CAPTURE_RANGE_M || recovery_ingress) {
        let altitude_m = position_bf.length() - body_radius_m;
        let arrival_altitude_m =
            entry.end.strip.elevation_m + FINAL_LENGTH_M * 3.0_f64.to_radians().tan();
        let cruise_altitude_m = (terrain.0.max_elevation_m(dominant)
            + DESTINATION_TERRAIN_CLEARANCE_M)
            .max(arrival_altitude_m + 1_500.0);
        state.destination_guidance = compute_destination_guidance(
            arrival_dir,
            &DestinationParams {
                body_radius_m,
                gravity_m_s2: params.gravity_m_s2,
                bank_limit_rad: BANK_LIMIT_RAD,
                cruise_altitude_m,
                arrival_altitude_m,
                descent_distance_m: DESTINATION_DESCENT_DISTANCE_M,
                cruise_speed_m_s: (approach_speed_m_s * 1.8).max(130.0),
                max_vertical_speed_m_s: DESTINATION_MAX_VS_M_S,
            },
            &DestinationInput {
                position_body_fixed: position_bf,
                track_dir_body_fixed: track_dir_bf,
                ground_speed_m_s,
                altitude_m,
            },
        );
        state.destination_arrival_dir = Some(arrival_dir);
        state.plan = None;
        state.active_path = LateralPath::default();
        state.rejoin_len_m = 0.0;
        state.guidance = None;
        state.planned_for = Some(armed);
        state.track_hint_along_m = None;
        state.established = false;
        state.display = RouteDisplay::default();
        state.status = if state.destination_guidance.is_some() {
            RouteStatus::Armed
        } else {
            RouteStatus::Unavailable
        };
        if selection_changed {
            info!(
                "route: destination RWY {:02} — spherical ingress {:.1} km",
                entry.designator,
                arrival_range_m / 1000.0
            );
        }
        return;
    }
    state.force_destination_ingress = false;
    state.destination_guidance = None;
    state.destination_arrival_dir = Some(arrival_dir);

    // --- Re-plan policy (see the module docs).
    let now = time.elapsed_secs();
    //
    // A committed rejoin means the craft is already flying a planned way back,
    // so cross-track from the nominal route is expected to be large and is not
    // evidence that the plan is unreachable.
    let cross_track_m = state.guidance.map(|g| g.cross_track_m.abs()).unwrap_or(0.0);
    let rejoin_flying = state.rejoin_len_m > 0.0;
    let unreachable = cross_track_m
        > if rejoin_flying {
            REPLAN_UNREACHABLE_CROSS_TRACK_M
        } else {
            REPLAN_CROSS_TRACK_M
        };
    let drifted =
        !state.plan_frozen && unreachable && now - state.planned_at_s >= REPLAN_MIN_INTERVAL_S;
    let needs_plan = selection_changed || state.plan.is_none() || drifted;
    if drifted {
        info!(
            target: "thalos::diagnostic::approach_ap",
            event = "route_replanned",
            cross_track_m,
            rejoin_flying,
            "route: rebuilding the approach"
        );
    }

    if needs_plan {
        match plan_approach(entry.end, position_bf, track_dir_bf, &params) {
            Some(plan) => {
                if selection_changed {
                    info!(
                        "route: armed RWY {:02} ({:.0}° / {:.1} km) — final {:.1} km, turn radius {:.0} m",
                        entry.designator,
                        entry.landing_heading_rad.to_degrees(),
                        entry.threshold_range_m / 1000.0,
                        FINAL_LENGTH_M / 1000.0,
                        plan.turn_radius_m,
                    );
                }
                // A fresh plan is its own active path; any committed rejoin
                // belonged to geometry that no longer exists.
                state.active_path = plan.path.clone();
                state.rejoin_len_m = 0.0;
                state.rejoin_committed_at_s = now;
                state.display = plan_display(&plan, &state.active_path);
                state.plan = Some(plan);
                state.planned_for = Some(armed);
                state.planned_at_s = now;
                // Both hints index into the path that was just replaced.
                // Carrying either across would pin the craft to an along-track
                // distance on geometry that no longer exists.
                state.track_hint_along_m = None;
            }
            None => {
                clear_plan(&mut state, RouteStatus::Unavailable);
                return;
            }
        }
    }

    if state.plan.is_none() {
        clear_plan(&mut state, RouteStatus::Idle);
        return;
    }

    // --- Commit a rejoin into the route, if the craft has drifted off it.
    //
    // This is the one place the route may be *amended* without being replanned.
    // A rejoin is not a steering cue running alongside the drawn route — that
    // arrangement gave the system two answers to "where should I go", and
    // because it was recomputed every frame its aim point slid forward as the
    // craft advanced, so the craft chased a receding target for 385 s without
    // ever capturing (INC-20260801T035551Z). Committed, it is simply the front
    // of the route: drawn, flown, and measured as one object.
    let vertical_speed_m_s = velocity_bf.dot(up);
    maybe_commit_rejoin(&mut state, position_bf, track_dir_bf, now);

    let (guidance, rejoin_points) = {
        let plan = state.plan.as_ref().expect("checked above");
        let frame = plan.frame;
        let guidance = compute_guidance(
            plan,
            &state.active_path,
            &GuidanceInput {
                position_body_fixed: position_bf,
                track_dir_body_fixed: track_dir_bf,
                ground_speed_m_s,
                vertical_speed_m_s,
                gravity_m_s2: params.gravity_m_s2,
                bank_limit_rad: BANK_LIMIT_RAD,
                bank_rad,
                track_hint_along_m: state.track_hint_along_m,
            },
        );

        // The committed rejoin is drawn in its own colour as the leading part of
        // the route — "this is the bit that gets you back on" — rather than as a
        // second line competing with it.
        let mut points = Vec::new();
        if state.rejoin_len_m > 1.0 {
            let elevation = frame.origin_altitude_m;
            let sag_m = (state.rejoin_len_m * 0.002).clamp(2.0, 60.0);
            let head = state.active_path.head_to(state.rejoin_len_m);
            points.extend(
                head.polyline(sag_m)
                    .into_iter()
                    .map(|point| frame.to_body_fixed(point, elevation)),
            );
        }
        (guidance, points)
    };

    state.display.rejoin_points = rejoin_points;
    state.track_hint_along_m = guidance.map(|g| g.along_m);
    // The rejoin is consumed once flown: from there the craft is on the nominal
    // route again, and a later drift is allowed to commit a fresh one.
    if let Some(g) = guidance
        && state.rejoin_len_m > 0.0
        && g.along_m >= state.rejoin_len_m
    {
        state.rejoin_len_m = 0.0;
    }
    if let Some(g) = guidance {
        // Two separate questions, deliberately kept apart.
        //
        // *May the plan still change?* — no, once past the final approach
        // point, because re-planning from there routes back to a fix behind the
        // craft. Purely geometric, and irreversible for this plan.
        if g.phase != ApproachPhase::Transition {
            state.plan_frozen = true;
        }
        // *Is the craft actually on the beam?* — the needles say, every frame,
        // and they are allowed to say no again. Freezing the plan does not make
        // an approach stable, and treating it as if it did is what flew an
        // approach 1.8 km off course down to the threshold.
        state.established = g.established && g.phase != ApproachPhase::Transition;
    }
    state.guidance = guidance;
    state.status = RouteStatus::Armed;
}

/// Amend the active route with a flyable rejoin when the craft has drifted off
/// it, and adopt the plan's own path when there is nothing to amend.
///
/// Committing is deliberately rare and sticky. The craft is expected to *fly*
/// the committed path, so re-deciding every frame would recreate the moving
/// target this design exists to remove: the trigger is a real drift off the
/// **active** path (not the nominal one, which the craft may legitimately be far
/// from while flying a committed rejoin), it is rate-limited, and it is refused
/// once the plan is frozen — reshaping the route past the final approach point
/// is the same mistake as re-planning there.
fn maybe_commit_rejoin(state: &mut RouteState, position_bf: DVec3, track_dir_bf: DVec3, now: f32) {
    let Some(plan) = state.plan.as_ref() else {
        return;
    };
    // A fresh or invalidated plan starts as its own active path.
    if state.active_path.is_empty() {
        state.active_path = plan.path.clone();
        state.rejoin_len_m = 0.0;
        state.track_hint_along_m = None;
    }

    let frame = plan.frame;
    let local = frame.to_local(position_bf);
    let Some(on_active) = state
        .active_path
        .closest_from(local, state.track_hint_along_m)
    else {
        return;
    };
    let drifted = on_active.cross_track_m.abs() > REJOIN_COMMIT_CROSS_TRACK_M;
    if state.plan_frozen
        || !drifted
        || now - state.rejoin_committed_at_s < REJOIN_COMMIT_MIN_INTERVAL_S
    {
        return;
    }

    // Plan the way back against the *nominal* route — that is what the craft is
    // ultimately trying to be on. A global projection is right here: commits are
    // rare, and this is exactly the "seeding a fresh track" case.
    let Some(track) = frame.direction_to_local(track_dir_bf).try_normalize() else {
        return;
    };
    let Some(on_plan) = plan.path.closest(local) else {
        return;
    };
    let Some(rejoin) = plan_rejoin(
        &plan.path,
        Pose2::new(local, theta_of(track)),
        on_plan.along_m,
        &RejoinParams::for_radius(plan.turn_radius_m),
        None,
    ) else {
        return;
    };
    if rejoin.path.length() <= 1.0 {
        return;
    }

    // Distance-to-go is the invariant across the swap — a splice adds length
    // ahead of the craft and leaves every distance-to-go alone — so carry the
    // track hint across as a dtg, not as an along-track distance. Dropping it
    // instead would reseed the projection globally and reintroduce exactly the
    // jump `closest_from` exists to prevent.
    let dtg_before = (state.active_path.length() - on_active.along_m).max(0.0);
    state.active_path = plan
        .path
        .splice_rejoin(rejoin.path.clone(), rejoin.capture_along_m);
    state.rejoin_len_m = rejoin.length_m;
    state.rejoin_committed_at_s = now;
    state.track_hint_along_m = Some((state.active_path.length() - dtg_before).max(0.0));
    state.display = plan_display(plan, &state.active_path);
    info!(
        target: "thalos::diagnostic::approach_ap",
        event = "route_rejoin_committed",
        cross_track_m = on_active.cross_track_m,
        rejoin_len_m = rejoin.length_m,
        capture_along_m = rejoin.capture_along_m,
        excess_turn_rad = rejoin.excess_turn_rad,
        "route: rejoin spliced into the active path"
    );
}

/// Drop the plan and guidance, keeping the enumerated ends (a display still
/// wants to draw runways with nothing armed).
fn clear_plan(state: &mut RouteState, status: RouteStatus) {
    if state.plan.is_some() {
        info!("route: disarmed");
    }
    state.plan = None;
    state.active_path = LateralPath::default();
    state.rejoin_len_m = 0.0;
    state.guidance = None;
    state.destination_guidance = None;
    state.destination_arrival_dir = None;
    state.armed = None;
    state.planned_for = None;
    state.plan_frozen = false;
    state.established = false;
    state.track_hint_along_m = None;
    state.force_destination_ingress = false;
    state.display = RouteDisplay::default();
    state.status = status;
}

/// Tessellate a plan into body-fixed display geometry.
///
/// Sag tolerance scales with the route length so a 200 km diversion does not
/// spend hundreds of points, while a 400 m turn radius on a 2 km-range plot
/// still reads as a curve rather than a corner.
///
/// Public because the headless ND preview (`examples/nav_preview.rs`) draws real
/// plans through this exact function — a preview that tessellated its own way
/// would be checking symbology against geometry the game never produces.
pub fn plan_display(plan: &ApproachPlan, active_path: &LateralPath) -> RouteDisplay {
    let frame: &RouteFrame = &plan.frame;
    let sag_m = (active_path.length() * 0.002).clamp(2.0, 60.0);
    let elevation = frame.origin_altitude_m;

    let mut display = RouteDisplay::default();
    // Leg by leg, so the final approach's first point is a known index rather
    // than the result of a float comparison (see `final_start_index`). The final
    // is always the last leg — `plan_approach` pushes it after the transition,
    // and a spliced rejoin only ever adds legs to the front.
    let final_leg = active_path.legs.len().saturating_sub(1);
    let mut along = 0.0;
    let mut prev: Option<DVec2> = None;
    for (leg_index, leg) in active_path.legs.iter().enumerate() {
        if leg_index == final_leg {
            display.final_start_index = display.path_points.len().saturating_sub(1);
        }
        for p in LateralPath::new(vec![*leg]).polyline(sag_m) {
            if let Some(previous) = prev {
                let step = previous.distance(p);
                // The shared endpoint between consecutive legs is emitted twice;
                // keep one copy.
                if step < 1e-6 {
                    continue;
                }
                along += step;
            }
            prev = Some(p);
            display.path_points.push(frame.to_body_fixed(p, elevation));
            display.path_along_m.push(along);
        }
    }
    display.waypoints = plan
        .waypoints
        .iter()
        .map(|w| {
            let altitude = w.vertical.map(|v| v.target_m()).unwrap_or(elevation);
            (w.dir * (frame.body_radius_m + altitude), w.kind)
        })
        .collect();
    display
}

/// Approach speed (m/s): `1.3 × Vs` from the craft's own lift curve, mass, and
/// the air density **at the threshold** — not at the craft's current altitude,
/// which would size the approach speed for cruise air.
///
/// Falls back to [`FALLBACK_APPROACH_SPEED_M_S`] for a craft with no lift curve
/// (a capsule), a body with no atmosphere, or a missing aero readout.
fn approach_speed(
    sim: &SimulationState,
    body_id: BodyId,
    threshold_elevation_m: f64,
    ship_aero: Option<&ShipAero>,
) -> f64 {
    let body = &sim.system.bodies[body_id];
    let Some(atmosphere) = body.terrestrial_atmosphere.as_ref() else {
        return FALLBACK_APPROACH_SPEED_M_S;
    };
    let sample = atmosphere.sample_at_altitude_m(
        threshold_elevation_m,
        body.surface_pressure_pa(),
        body.surface_gravity_m_s2(),
    );
    let density = sample.density_kg_m3;
    if density <= 1e-6 {
        return FALLBACK_APPROACH_SPEED_M_S;
    }
    let Some(config) = ship_aero.map(|a| a.config) else {
        return FALLBACK_APPROACH_SPEED_M_S;
    };
    if config.lift_slope <= 0.0 || config.reference_area_m2 <= 0.0 {
        return FALLBACK_APPROACH_SPEED_M_S;
    }
    // Landing configuration: clean CL_max plus the flap increment.
    let cl_max = config.cl0 + config.lift_slope * config.stall_alpha + config.flap_dcl;
    if cl_max <= 0.0 {
        return FALLBACK_APPROACH_SPEED_M_S;
    }
    let mass = sim.simulation.ship_mass_kg().max(1.0);
    let g = body.gm / (body.radius_m + threshold_elevation_m).powi(2);
    let vs = (2.0 * mass * g / (density * config.reference_area_m2 * cl_max)).sqrt();
    if vs.is_finite() && vs > 1.0 {
        APPROACH_STALL_MARGIN * vs
    } else {
        FALLBACK_APPROACH_SPEED_M_S
    }
}
