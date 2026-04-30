//! General ship autopilot.
//!
//! The autopilot owns ship-control primitives (warp, attitude target,
//! throttle) while executing a scheduled burn directive. Maneuver nodes
//! are only one producer of those directives: they provide an execution
//! time, burn direction, and planned Δv magnitude. The state machine
//! below does not inspect maneuver-node data or use the maneuver
//! pointing mode; future producers can publish the same directive shape
//! without changing the executor.
//!
//! Four execution states:
//!
//! - [`AutopilotState::Idle`] — no directive pending, autopilot doesn't act.
//! - [`AutopilotState::Armed`] — a directive exists. The player retains
//!   full control of throttle, attitude, and nav mode. The autopilot
//!   defensively clamps warp from above so a single sim-time advance
//!   can't overshoot the lead window. Manual warp-up is permitted
//!   per-frame; the next frame's clamp catches it. No user controls are
//!   locked.
//! - [`AutopilotState::Engaging`] — within the lead window. The
//!   autopilot now owns the ship: it forces warp to 1×, publishes a
//!   direct attitude target so the PD controller settles on the burn
//!   direction, and holds throttle at zero. User input is locked out for
//!   the keys/buttons the autopilot drives. Transitions to `Burn` once
//!   warp is at 1×, attitude is converged on the burn vector, and
//!   `now ≥ directive.center_time − duration/2`.
//! - [`AutopilotState::Burn`] — throttle held at 1.0 each frame,
//!   integrating `delivered_dv` until magnitude meets the planned Δv;
//!   then the directive is completed and the autopilot returns to
//!   `Idle`.
//!
//! Burns are centered on `center_time`: the throttle opens at
//! `center_time − duration/2`. The trajectory predictor in
//! [`thalos_physics::trajectory`] uses the same convention for maneuver
//! burns, so the displayed flight plan matches what the autopilot flies
//! for maneuver-provided directives.
//!
//! Engagement trigger. The autopilot only arms a directive while there
//! is still a full lead window available. Armed → Engaging fires when
//! sim-time crosses into that lead window. The warp cascade that ramps
//! speed down to 1× lives inside Armed (one level per frame, capped so
//! a single advance can't overshoot the lead window) so the player keeps
//! controls during the cascade itself. Lockout only kicks in once the
//! autopilot genuinely owns the ship.
//!
//! Maneuver editing. The maneuver directive publisher treats dirty or
//! actively-slid plans as planning-only and clears its published
//! directive unless the autopilot is already burning. The core autopilot
//! only sees "directive exists" vs. "no directive"; it has no maneuver
//! editing policy of its own.
//!
//! Pointing-before-burning. The transition Engaging → Burn requires the
//! ship's nose to be within [`POINTING_TOLERANCE_COS`] of the burn
//! direction *and* its angular velocity to be below
//! [`POINTING_OMEGA_TOL`]. Without this, the autopilot would open the
//! throttle at the nominal burn start regardless of attitude, and the
//! off-axis component of an unsettled pointing would be wasted as
//! lateral thrust.
//!
//! Slew-aware lead. The base lead from [`lead_seconds_for`] only budgets
//! the *small-angle* PD settling tail. For a large slew (90°-180° flip)
//! the controller saturates against `max_torque` and moves bang-bang:
//! accelerate then decelerate, total `t ≈ 2·sqrt(θ/α_min)`. We estimate
//! that time per-frame from the current ship-nose-vs-burn-vector angle
//! and the slowest pitch/yaw axis, then engage `t_slew` seconds earlier
//! so pointing finishes before burn start instead of the burn waiting it
//! out and firing late.
//!
//! Control lockout: while the autopilot is "active" ([`is_active`]
//! true, i.e. `Engaging` or `Burn`), the throttle/attitude/warp input
//! handlers gate themselves out — both keybind and HUD-button routes —
//! so the autopilot owns every primitive it cares about. `Armed`
//! deliberately does not lock anything: a directive placed a year out
//! shouldn't disable the player's controls for the entire wait.
//!
//! Disengage path: if the user toggles the autopilot checkbox mid-burn,
//! the autopilot writes throttle = 0 once and returns to `Idle`. Without
//! the explicit zero-write the previously-asserted `commanded = 1.0`
//! would persist and the engine would keep firing after the autopilot
//! stopped owning the throttle.
//!
//! [`is_active`]: Autopilot::is_active

use bevy::math::DVec3;
use bevy::prelude::*;

use crate::SimStage;
use crate::fuel::{ThrottleState, gate_throttle_on_fuel_availability, handle_throttle_input};
use crate::maneuver::{InteractionMode, ManeuverPlan};
use crate::navigation::{AUTOPILOT_SETTLE_S, SHIP_NOSE_BODY, maneuver_node_burn_direction};
use crate::rendering::SimulationState;

const MANEUVER_DIRECTIVE_NAMESPACE: &str = "maneuver";

/// Opaque id for an autopilot directive.
///
/// `namespace` lets each producer keep its own local id space without
/// making the core autopilot depend on producer-specific enums. The
/// maneuver directive adapter uses `"maneuver"` and the UI node id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AutopilotDirectiveId {
    namespace: &'static str,
    local_id: u64,
}

impl AutopilotDirectiveId {
    pub const fn new(namespace: &'static str, local_id: u64) -> Self {
        Self {
            namespace,
            local_id,
        }
    }

    pub const fn namespace(self) -> &'static str {
        self.namespace
    }

    pub const fn local_id(self) -> u64 {
        self.local_id
    }
}

/// A scheduled burn request for the autopilot.
///
/// This is deliberately not a maneuver node: producers resolve their own
/// domain data into timing, direction, and scalar burn size before the
/// executor sees it.
#[derive(Debug, Clone, Copy)]
pub struct AutopilotBurnDirective {
    pub id: AutopilotDirectiveId,
    /// Nominal burn center time, seconds from simulation epoch.
    pub center_time: f64,
    /// World-frame unit vector for the burn.
    pub direction: DVec3,
    /// Planned Δv magnitude, m/s.
    pub delta_v_magnitude: f64,
    /// Estimated finite-burn duration, seconds.
    pub duration_s: f64,
}

impl AutopilotBurnDirective {
    fn burn_start(self) -> f64 {
        self.center_time - self.duration_s / 2.0
    }
}

/// The next burn directive visible to the autopilot.
///
/// Today this is populated from the next maneuver node. Keeping it as a
/// separate resource makes the maneuver-to-autopilot adapter replaceable
/// by later guidance systems without touching the executor.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct AutopilotBurnSchedule {
    next: Option<AutopilotBurnDirective>,
}

impl AutopilotBurnSchedule {
    pub fn clear(&mut self) {
        self.next = None;
    }

    pub fn set_next(&mut self, directive: AutopilotBurnDirective) {
        self.next = Some(directive);
    }

    pub fn next(&self) -> Option<AutopilotBurnDirective> {
        self.next
    }

    fn get(&self, id: AutopilotDirectiveId) -> Option<AutopilotBurnDirective> {
        self.next.filter(|directive| directive.id == id)
    }
}

#[derive(Resource, Debug)]
pub struct Autopilot {
    pub enabled: bool,
    pub(crate) state: AutopilotState,
}

impl Default for Autopilot {
    fn default() -> Self {
        Self {
            enabled: true,
            state: AutopilotState::Idle,
        }
    }
}

impl Autopilot {
    /// Snapshot of the current executor state for UI/read-only systems.
    pub(crate) fn state(&self) -> AutopilotState {
        self.state
    }

    /// `true` when the autopilot is actively driving the ship — i.e.
    /// state is `Engaging` or `Burn`. Consumed by
    /// [`crate::controls::update_control_locks`] which translates it
    /// into the per-surface flags that input handlers actually read.
    pub(crate) fn is_active(&self) -> bool {
        matches!(
            self.state,
            AutopilotState::Engaging { .. } | AutopilotState::Burn { .. }
        )
    }

    /// `true` only while a scheduled burn is actively being executed.
    /// The flight-plan view uses this to hold the precomputed trajectory
    /// steady while thrust is being applied.
    pub(crate) fn is_burning(&self) -> bool {
        matches!(self.state, AutopilotState::Burn { .. })
    }

    /// Direct world-frame attitude target while the autopilot owns
    /// pointing. This bypasses `NavigationMode::ManeuverNode` so the
    /// executor can fly any directive producer, not only maneuver nodes.
    pub(crate) fn attitude_target(&self) -> Option<DVec3> {
        match self.state {
            AutopilotState::Engaging { direction, .. } | AutopilotState::Burn { direction, .. } => {
                Some(direction)
            }
            AutopilotState::Idle | AutopilotState::Armed { .. } => None,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum AutopilotState {
    #[default]
    Idle,
    /// A valid directive exists. Player retains full control of
    /// throttle, attitude, and nav mode; the autopilot defensively
    /// clamps warp from above so a single sim-time advance can't
    /// overshoot the lead window.
    Armed { directive_id: AutopilotDirectiveId },
    /// Within the lead window: ramping warp, pointing the ship, holding
    /// throttle at zero. User input is locked out.
    Engaging {
        directive_id: AutopilotDirectiveId,
        direction: DVec3,
    },
    /// Engine firing, integrating delivered Δv toward the planned
    /// magnitude.
    Burn {
        directive_id: AutopilotDirectiveId,
        direction: DVec3,
        /// Magnitude of the planned Δv, m/s.
        planned_dv: f64,
        /// Value of [`thalos_physics::simulation::Simulation::delivered_dv`]
        /// captured at the moment the burn started; subtracted from the
        /// live value each frame to get "Δv delivered since burn start."
        anchor_delivered_dv: f64,
    },
}

/// Emitted when a generic burn directive completes.
///
/// Producer adapters decide whether the id belongs to them. The
/// maneuver adapter uses this to retire the corresponding maneuver node.
#[derive(Debug, Clone, Copy, Message)]
pub struct AutopilotBurnCompleted {
    pub id: AutopilotDirectiveId,
}

pub struct AutopilotPlugin;

impl Plugin for AutopilotPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Autopilot>()
            .init_resource::<AutopilotBurnSchedule>()
            .add_message::<AutopilotBurnCompleted>()
            .add_systems(
                Update,
                publish_maneuver_autopilot_directive
                    .in_set(SimStage::Physics)
                    .before(autopilot_system),
            )
            .add_systems(
                Update,
                autopilot_system
                    .in_set(SimStage::Physics)
                    .after(crate::bridge::handle_warp_controls)
                    .after(handle_throttle_input)
                    .before(crate::bridge::handle_attitude_controls)
                    .before(gate_throttle_on_fuel_availability)
                    .before(crate::bridge::advance_simulation),
            )
            .add_systems(
                Update,
                consume_completed_maneuver_directives
                    .in_set(SimStage::Physics)
                    .after(autopilot_system)
                    .before(crate::bridge::advance_simulation),
            );
    }
}

/// How many real seconds before `directive.center_time` the autopilot
/// needs to be at 1× warp. Sized to cover one PD settling cycle (with
/// safety margin) plus the half-burn that precedes center time.
///
/// Also consumed by the warp-to-next-maneuver auto-warp in
/// [`crate::warp_to_maneuver`] when sizing its lead window — both
/// systems agree on the same safe time so they don't fight over warp
/// level inside the window.
pub(crate) fn lead_seconds_for(burn_duration_s: f64) -> f64 {
    AUTOPILOT_SETTLE_S * 1.5 + burn_duration_s / 2.0 + 0.5
}

/// Worst-case real-frame budget — must match
/// [`thalos_physics::simulation::SimulationConfig::max_real_delta`] so
/// our look-ahead captures the largest sim-time advance the upcoming
/// step might apply.
const FRAME_DT_BUDGET_S: f64 = 0.1;

/// `cos(θ_max)` for the pointing-converged check. 0.9998 ≈ cos(1.15°)
/// — tight enough that off-axis thrust is < 0.02% of total, loose
/// enough that a small residual ω from the PD controller can still
/// settle inside one frame at 60 Hz.
const POINTING_TOLERANCE_COS: f64 = 0.9998;

/// Maximum angular velocity (rad/s) for the pointing-converged check.
/// Without this, the dot-product check could pass on the way through
/// the target while ω is still high — opening throttle on a sweeping
/// nose would smear thrust across a band, not a point.
const POINTING_OMEGA_TOL: f64 = 0.05;

/// Multiplier on the bang-bang slew time. Real PD leaves torque
/// saturation slightly before the ideal switch point, so the actual
/// slew is a bit slower than `2·sqrt(θ/α)`. 1.2 absorbs that without
/// noticeably bloating the engagement window.
const SLEW_SAFETY_MARGIN: f64 = 1.2;

/// Hard ceiling on the slew-time estimate. An underactuated ship
/// (`max_torque / inertia → 0`) makes the bang-bang formula explode;
/// the cap keeps the lead window bounded. A ship that genuinely can't
/// slew within this is broken in ways the autopilot can't paper over.
const MAX_SLEW_ESTIMATE_S: f64 = 60.0;

/// Publish a generic burn directive from the next maneuver node.
///
/// This is the only maneuver-specific input edge into the autopilot:
/// it resolves the node into timing, direction, and planned Δv. The
/// executor itself consumes only [`AutopilotBurnDirective`].
pub(crate) fn publish_maneuver_autopilot_directive(
    mut schedule: ResMut<AutopilotBurnSchedule>,
    autopilot: Res<Autopilot>,
    sim: Res<SimulationState>,
    plan: Res<ManeuverPlan>,
    mode: Res<InteractionMode>,
) {
    if edit_blocks_maneuver_directive(&mode, plan.dirty, autopilot.state()) {
        schedule.clear();
        return;
    }

    let Some(node) = plan.nodes.first() else {
        schedule.clear();
        return;
    };
    let dv_mag = node.delta_v.length();
    let Some(duration) = burn_duration_for(dv_mag, &sim.simulation) else {
        schedule.clear();
        return;
    };
    let Some(direction) = maneuver_node_burn_direction(&sim.simulation, node) else {
        schedule.clear();
        return;
    };

    schedule.set_next(AutopilotBurnDirective {
        id: AutopilotDirectiveId::new(MANEUVER_DIRECTIVE_NAMESPACE, node.id.0),
        center_time: node.time,
        direction,
        delta_v_magnitude: dv_mag,
        duration_s: duration,
    });
}

/// Retire maneuver nodes whose generic autopilot directive completed.
///
/// This is the output adapter corresponding to
/// [`publish_maneuver_autopilot_directive`]. The core autopilot emits an
/// opaque directive id; this adapter recognizes the maneuver namespace
/// and reconciles the physics/UI maneuver schedule.
pub(crate) fn consume_completed_maneuver_directives(
    mut completed: MessageReader<AutopilotBurnCompleted>,
    mut sim: ResMut<SimulationState>,
) {
    for event in completed.read() {
        if event.id.namespace() != MANEUVER_DIRECTIVE_NAMESPACE {
            continue;
        }
        sim.simulation.consume_maneuver_node(event.id.local_id());
    }
}

/// `true` when the ship's nose lies within [`POINTING_TOLERANCE_COS`]
/// of `target_dir` *and* its angular velocity is below
/// [`POINTING_OMEGA_TOL`]. Both clauses are needed: the dot test alone
/// passes momentarily during a fast slew through the target.
fn pointing_converged(
    sim: &thalos_physics::simulation::Simulation,
    target_dir: bevy::math::DVec3,
) -> bool {
    let attitude = sim.attitude();
    let nose_world = attitude.orientation * SHIP_NOSE_BODY;
    let cos_err = nose_world.dot(target_dir);
    let omega = attitude.angular_velocity.length();
    cos_err >= POINTING_TOLERANCE_COS && omega <= POINTING_OMEGA_TOL
}

/// Bang-bang slew time for an angular distance `angle_rad` against the
/// slowest pitch/yaw axis, with the safety margin and cap applied. PD's
/// small-angle tail is *not* included here — `lead_seconds_for` already
/// covers it via `AUTOPILOT_SETTLE_S * 1.5`.
fn slew_time_for_angle(angle_rad: f64, sim: &thalos_physics::simulation::Simulation) -> f64 {
    if angle_rad < 1e-3 {
        return 0.0;
    }
    let params = sim.ship_params();
    let alpha_pitch = params.max_torque.x / params.moment_of_inertia.x.max(1e-9);
    let alpha_yaw = params.max_torque.z / params.moment_of_inertia.z.max(1e-9);
    let alpha = alpha_pitch.min(alpha_yaw).max(1e-6);
    (SLEW_SAFETY_MARGIN * 2.0 * (angle_rad / alpha).sqrt()).min(MAX_SLEW_ESTIMATE_S)
}

fn slew_time_estimate(
    target_dir: bevy::math::DVec3,
    sim: &thalos_physics::simulation::Simulation,
) -> f64 {
    let nose_world = sim.attitude().orientation * SHIP_NOSE_BODY;
    let angle = nose_world.dot(target_dir).clamp(-1.0, 1.0).acos();
    slew_time_for_angle(angle, sim)
}

fn burn_duration_for(dv_mag: f64, sim: &thalos_physics::simulation::Simulation) -> Option<f64> {
    if dv_mag <= 0.0 {
        return None;
    }
    let duration = sim.estimated_burn_duration(dv_mag);
    (duration > 0.0).then_some(duration)
}

fn lead_for_directive(
    directive: AutopilotBurnDirective,
    sim: &thalos_physics::simulation::Simulation,
) -> f64 {
    lead_seconds_for(directive.duration_s) + slew_time_estimate(directive.direction, sim)
}

fn has_preparation_margin(now: f64, directive: AutopilotBurnDirective, lead: f64) -> bool {
    now < directive.center_time - lead
}

fn edit_blocks_maneuver_directive(
    mode: &InteractionMode,
    plan_dirty: bool,
    state: AutopilotState,
) -> bool {
    (matches!(*mode, InteractionMode::SlidingNode) || plan_dirty)
        && !matches!(state, AutopilotState::Burn { .. })
}

pub(crate) fn autopilot_system(
    mut autopilot: ResMut<Autopilot>,
    mut sim: ResMut<SimulationState>,
    mut throttle: ResMut<ThrottleState>,
    schedule: Res<AutopilotBurnSchedule>,
    mut completed: MessageWriter<AutopilotBurnCompleted>,
) {
    if !autopilot.enabled {
        if matches!(autopilot.state, AutopilotState::Burn { .. }) {
            throttle.commanded = 0.0;
        }
        autopilot.state = AutopilotState::Idle;
        return;
    }

    match autopilot.state {
        AutopilotState::Idle => {
            let Some(directive) = schedule.next() else {
                return;
            };
            let now = sim.simulation.sim_time();
            let lead = lead_for_directive(directive, &sim.simulation);
            if !has_preparation_margin(now, directive, lead) {
                return;
            }
            // Park in Armed — observation only, no commands asserted.
            // Engagement and user lockout wait until the burn is
            // actually approaching.
            autopilot.state = AutopilotState::Armed {
                directive_id: directive.id,
            };
        }

        AutopilotState::Armed { directive_id } => {
            let Some(directive) = schedule.get(directive_id) else {
                autopilot.state = AutopilotState::Idle;
                return;
            };

            let now = sim.simulation.sim_time();
            // Slew time depends on where the ship is currently
            // pointing; recomputed each frame so hand-aiming toward the
            // burn shrinks the lead window and pointing away opens it up.
            let lead = lead_for_directive(directive, &sim.simulation);
            let safe_target = (directive.center_time - lead).max(now);

            // Defensive warp clamp: drop one level whenever a single
            // worst-case frame would overshoot `safe_target`. Runs in
            // Armed (no lockout) so the player keeps controls while the
            // cascade ramps speed down.
            while sim.simulation.warp.speed() > 1.0 {
                let warp = sim.simulation.warp.speed();
                if now + warp * FRAME_DT_BUDGET_S > safe_target {
                    sim.simulation.warp.decrease();
                } else {
                    break;
                }
            }

            if now >= safe_target {
                // Crossed into the lead window. Hand the ship over to
                // the autopilot from this frame on. The direct attitude
                // target is exposed via `Autopilot::attitude_target`.
                autopilot.state = AutopilotState::Engaging {
                    directive_id,
                    direction: directive.direction,
                };
            }
        }

        AutopilotState::Engaging { directive_id, .. } => {
            let Some(directive) = schedule.get(directive_id) else {
                throttle.commanded = 0.0;
                autopilot.state = AutopilotState::Idle;
                return;
            };

            // Keep the attitude target fresh in case the directive
            // producer refines it while we are engaging.
            autopilot.state = AutopilotState::Engaging {
                directive_id,
                direction: directive.direction,
            };

            // Hold throttle at zero while we wait for warp to drop and
            // attitude to settle.
            throttle.commanded = 0.0;

            let now = sim.simulation.sim_time();

            // We entered Engaging at `now ≥ safe_target`, so the
            // Armed-side cascade has already pulled warp down close to
            // 1×. Final drop in case any level remains, then require 1×
            // before checking attitude/burn-start.
            while sim.simulation.warp.speed() > 1.0 {
                sim.simulation.warp.decrease();
            }

            let warp = sim.simulation.warp.speed();
            let at_1x = (warp - 1.0).abs() < f64::EPSILON;
            if !at_1x || now < directive.burn_start() {
                return;
            }

            // At 1× and past burn start — only ignite if attitude has
            // actually settled on the burn vector. If not, sit here at
            // throttle = 0 until pointing converges; the burn starts
            // late but on-axis, which is the lesser of two evils.
            if !pointing_converged(&sim.simulation, directive.direction) {
                return;
            }

            autopilot.state = AutopilotState::Burn {
                directive_id,
                direction: directive.direction,
                planned_dv: directive.delta_v_magnitude,
                anchor_delivered_dv: sim.simulation.delivered_dv(),
            };
            throttle.commanded = 1.0;
        }

        AutopilotState::Burn {
            directive_id,
            planned_dv,
            anchor_delivered_dv,
            ..
        } => {
            let delivered = sim.simulation.delivered_dv() - anchor_delivered_dv;
            if delivered >= planned_dv {
                throttle.commanded = 0.0;
                completed.write(AutopilotBurnCompleted { id: directive_id });
                autopilot.state = AutopilotState::Idle;
            } else {
                throttle.commanded = 1.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directive(center_time: f64) -> AutopilotBurnDirective {
        AutopilotBurnDirective {
            id: AutopilotDirectiveId::new("test", 1),
            center_time,
            direction: DVec3::Y,
            delta_v_magnitude: 10.0,
            duration_s: 2.0,
        }
    }

    #[test]
    fn arming_requires_full_preparation_margin() {
        assert!(has_preparation_margin(10.0, directive(20.0), 9.999));
        assert!(!has_preparation_margin(10.0, directive(20.0), 10.0));
        assert!(!has_preparation_margin(10.0, directive(20.0), 12.0));
    }

    #[test]
    fn maneuver_edits_block_non_burning_autopilot_states() {
        let id = AutopilotDirectiveId::new("test", 1);
        assert!(edit_blocks_maneuver_directive(
            &InteractionMode::SlidingNode,
            false,
            AutopilotState::Armed { directive_id: id },
        ));
        assert!(edit_blocks_maneuver_directive(
            &InteractionMode::Idle,
            true,
            AutopilotState::Engaging {
                directive_id: id,
                direction: DVec3::Y,
            },
        ));
        assert!(!edit_blocks_maneuver_directive(
            &InteractionMode::SlidingNode,
            true,
            AutopilotState::Burn {
                directive_id: id,
                direction: DVec3::Y,
                planned_dv: 5.0,
                anchor_delivered_dv: 0.0,
            },
        ));
    }
}
