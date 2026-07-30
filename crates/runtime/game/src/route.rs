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
//! # When a plan is recomputed (and when it must not be)
//!
//! The plan is *not* rebuilt every frame — a path that jitters as airspeed
//! wobbles is unflyable and unreadable. It is rebuilt when the selection
//! changes, and while still maneuvering, when the craft has drifted more than
//! [`REPLAN_CROSS_TRACK_M`] from the planned path (rate-limited by
//! [`REPLAN_MIN_INTERVAL_S`]).
//!
//! **Once the craft is established on final the plan freezes.** This is not an
//! optimisation: re-planning from a position *past* the final approach point
//! would ask the Dubins planner to fly back to a fix behind the craft, which it
//! solves with a full turn-around — the plan would loop the aircraft away from
//! the runway it is 3 km from. Freezing on final is the correctness rule.

use bevy::math::{DVec2, DVec3};
use bevy::prelude::*;

use thalos_navigation::{
    ApproachParams, ApproachPhase, ApproachPlan, Guidance, GuidanceInput, LateralPath, RouteFrame,
    RunwayEnd, RunwayStrip, VnavParams, WaypointKind, compute_guidance, plan_approach,
};
use thalos_physics_canonical::body_fixed::inertial_to_body_fixed;
use thalos_world::BodyId;

use crate::aero::ShipAero;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::structures::{StructureId, StructureKind, StructureRegistry};

/// Maximum bank the planner may require and the guidance may command (rad).
/// 25° is the airliner standard for maneuvering in the terminal area — steep
/// enough to turn in reasonable airspace, shallow enough to be comfortable and
/// to leave stall margin at approach speed.
pub const BANK_LIMIT_RAD: f64 = 0.436_332_313; // 25°

/// Never plan a turn tighter than this (m), whatever the speed suggests.
const MIN_TURN_RADIUS_M: f64 = 400.0;
/// Straight final approach segment length (m). At a 3° glideslope this puts the
/// capture altitude ~470 m above the threshold — a sane pattern altitude.
const FINAL_LENGTH_M: f64 = 9_000.0;
/// Aim point inset past the threshold (m). Also sets the threshold crossing
/// height: 300 m × tan 3° ≈ 16 m, matching real ILS practice.
const AIM_INSET_M: f64 = 300.0;
/// Shortest stabilised straight run onto the aim point the planner leaves itself
/// when the craft is already inside the final corridor (m) — see
/// `ApproachParams::min_capture_run_m`.
const MIN_CAPTURE_RUN_M: f64 = 1_200.0;
/// Approach speed used when the craft's own stall speed cannot be derived (no
/// atmosphere sample, or a craft with no lift curve). Matches the short-final
/// spawn speed in [`crate::runway`].
const FALLBACK_APPROACH_SPEED_M_S: f64 = 80.0;
/// Multiple of stall speed flown on approach — the standard 1.3 Vs margin.
const APPROACH_STALL_MARGIN: f64 = 1.3;

/// Cross-track drift (m) that triggers a re-plan while still maneuvering.
const REPLAN_CROSS_TRACK_M: f64 = 2_000.0;
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

/// A specific landable runway direction: a strip plus which way you land on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArmedEnd {
    pub strip: StructureId,
    /// `true` = landing against the strip's `heading_tangent`.
    pub reciprocal: bool,
}

/// The pilot's armed destination. `None` = nothing selected, guidance idle.
///
/// **Sole writer:** [`apply_route_requests`]. Everything that wants to change
/// the selection sends a [`RouteRequest`] instead of writing here, so the two
/// selection paths (clicking the ND plot, the selector buttons) cannot race.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct RouteSelection {
    pub armed: Option<ArmedEnd>,
}

/// A request to change the selection. Both selection paths speak this.
#[derive(Debug, Clone, Copy, Message)]
pub enum RouteRequest {
    /// Pick a strip (a click on the ND): arms the end you would more sensibly
    /// land on given where the craft is, and **flips** the end if that strip is
    /// already armed — so repeated clicks toggle the landing direction.
    Pick(StructureId),
    /// Step through every landable end on the body, nearest first.
    Cycle(i32),
    /// Land the other way on the currently armed strip.
    Flip,
    /// Disarm.
    Clear,
}

// ---------------------------------------------------------------------------
// Published state
// ---------------------------------------------------------------------------

/// Why there is no active guidance, for the display to say so plainly rather
/// than showing a blank plot.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum RouteStatus {
    /// Runways are available, none armed.
    #[default]
    Idle,
    /// A plan and live guidance exist.
    Armed,
    /// No runway on the dominant body.
    NoRunways,
    /// Body/craft state not available this frame (loading, no dominant body).
    Unavailable,
}

/// One selectable runway end, as the selector and the ND see it.
#[derive(Debug, Clone, Copy)]
pub struct RunwayEndEntry {
    pub armed_end: ArmedEnd,
    /// Navigation-crate view of the end (carries the strip geometry).
    pub end: RunwayEnd,
    /// Designator, `1..=36`.
    pub designator: u8,
    /// Compass heading of the landing direction (rad).
    pub landing_heading_rad: f64,
    /// Straight-line distance from the craft to this end's threshold (m).
    pub threshold_range_m: f64,
}

/// Display-ready geometry of the active plan, in **body-fixed metres** — the
/// same frame the ND already projects runways from, so a display never has to
/// know about route frames.
#[derive(Debug, Default, Clone)]
pub struct RouteDisplay {
    /// The planned lateral path, tessellated (arcs included).
    pub path_points: Vec<DVec3>,
    /// Named waypoints with their kind, in fly order.
    pub waypoints: Vec<(DVec3, WaypointKind)>,
    /// Index into `path_points` of the first point on the final approach leg.
    ///
    /// Carried as an **index**, not derived by comparing `path_along_m` against
    /// `final_start_along_m`: the polyline accumulates *chord* lengths while the
    /// plan measures *arc* length, so the two disagree by metres over a curved
    /// transition and the comparison lands on the wrong side of the boundary.
    /// That silently dropped the final-approach highlight on straight-in
    /// approaches, where the boundary is an exact tie.
    pub final_start_index: usize,
}

/// Everything downstream reads: the plan, the live guidance, the selectable
/// ends, and display-ready geometry.
///
/// **Sole writer:** [`update_route_state`].
#[derive(Resource, Default)]
pub struct RouteState {
    pub status: RouteStatus,
    /// The active approach plan, if one is armed and plannable.
    pub plan: Option<ApproachPlan>,
    /// Live guidance against that plan.
    pub guidance: Option<Guidance>,
    /// Every landable end on the dominant body, **nearest threshold first**.
    pub ends: Vec<RunwayEndEntry>,
    pub display: RouteDisplay,
    /// Approach speed the plan was built with (m/s) — shown as the speed target
    /// and used by the speed gates.
    pub approach_speed_m_s: f64,
    /// Which end the current plan belongs to, so a selection change is detected.
    planned_for: Option<ArmedEnd>,
    /// Real time of the last (re)plan.
    planned_at_s: f32,
    /// Latched once the craft reaches the final segment: freezes the plan (see
    /// the module docs — re-planning on final flies you away from the runway).
    established: bool,
}

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

    // --- Re-plan policy (see the module docs).
    let now = time.elapsed_secs();
    let selection_changed = state.planned_for != Some(armed);
    if selection_changed {
        state.established = false;
    }
    let drifted = !state.established
        && state
            .guidance
            .is_some_and(|g| g.cross_track_m.abs() > REPLAN_CROSS_TRACK_M)
        && now - state.planned_at_s >= REPLAN_MIN_INTERVAL_S;
    let needs_plan = selection_changed || state.plan.is_none() || drifted;

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
                state.display = plan_display(&plan);
                state.plan = Some(plan);
                state.planned_for = Some(armed);
                state.planned_at_s = now;
            }
            None => {
                clear_plan(&mut state, RouteStatus::Unavailable);
                return;
            }
        }
    }

    let Some(plan) = state.plan.as_ref() else {
        clear_plan(&mut state, RouteStatus::Idle);
        return;
    };
    let guidance = compute_guidance(
        plan,
        &GuidanceInput {
            position_body_fixed: position_bf,
            track_dir_body_fixed: track_dir_bf,
            ground_speed_m_s,
            gravity_m_s2: params.gravity_m_s2,
            bank_limit_rad: BANK_LIMIT_RAD,
        },
    );
    if let Some(g) = guidance
        && g.phase != ApproachPhase::Transition
    {
        // Latch: from here on the plan is frozen.
        state.established = true;
    }
    state.guidance = guidance;
    state.status = RouteStatus::Armed;
}

/// Drop the plan and guidance, keeping the enumerated ends (a display still
/// wants to draw runways with nothing armed).
fn clear_plan(state: &mut RouteState, status: RouteStatus) {
    if state.plan.is_some() {
        info!("route: disarmed");
    }
    state.plan = None;
    state.guidance = None;
    state.planned_for = None;
    state.established = false;
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
pub fn plan_display(plan: &ApproachPlan) -> RouteDisplay {
    let frame: &RouteFrame = &plan.frame;
    let sag_m = (plan.length_m() * 0.002).clamp(2.0, 60.0);
    let elevation = frame.origin_altitude_m;

    let mut display = RouteDisplay::default();
    // Leg by leg, so the final approach's first point is a known index rather
    // than the result of a float comparison (see `final_start_index`). The final
    // is always the last leg — `plan_approach` pushes it after the transition.
    let final_leg = plan.path.legs.len().saturating_sub(1);
    let mut prev: Option<DVec2> = None;
    for (leg_index, leg) in plan.path.legs.iter().enumerate() {
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
            }
            prev = Some(p);
            display.path_points.push(frame.to_body_fixed(p, elevation));
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
