//! Navigation, planning, and guidance state: the maneuver plan, navigation
//! modes, autopilots, the orbit program, routes, warp-to-node, and the
//! velocity reference frame. Writers stay with their owning runtime modules.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_control::ControlDemand;
use thalos_navigation::approach::RunwayEnd;
use thalos_navigation::{ApproachPlan, DestinationGuidance, Guidance, WaypointKind};
use thalos_physics_canonical::maneuver::delta_v_to_world;
use thalos_physics_canonical::trajectory::Trajectory;
use thalos_physics_canonical::orbit_planner::{OrbitDirection, TargetOrbit, TargetPlane};
use thalos_physics_canonical::simulation::Simulation;
use thalos_physics_canonical::velocity_frame::VelocityReferenceFrame;

use crate::autoflight::{AutoflightLocks, BurnArm, SequenceEvent};
use crate::maneuver_plan::{GameNode, ManeuverPlan};
use crate::structures::StructureId;

/// Per-frame cap on warp level imposed by altitude above the dominant
/// body. Computed by [`enforce_warp_altitude_limits`] each frame and read
/// by [`handle_warp_controls`] to refuse manual escalation past the cap.
///
/// `max_level` is an index into `Simulation::warp.levels()`. The default
/// `usize::MAX` means "no constraint" — used on the first frame before
/// enforcement runs and whenever the craft is in a regime where canonical
/// step does not propagate translation (landed, in the local-rigid-body
/// bubble), so terrain phasing is impossible.
#[derive(Resource, Debug, Clone)]
pub struct WarpLimits {
    pub max_level: usize,
}

/// Currently selected target body (world ID, not entity id).
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct TargetBody {
    pub target: Option<usize>,
}

#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    Map,
    #[default]
    Ship,
}

/// The player craft's active navball speed mode plus the sticky-override
/// bookkeeping that drives auto-switching.
#[derive(Resource, Default, Debug)]
pub struct VelocityFrameState {
    /// Frame active this frame. **Sole writer:** [`update_velocity_frame`].
    pub active: VelocityReferenceFrame,
    /// Sticky user override (set by the readout click); cleared when the
    /// auto suggestion changes (situation-boundary crossing) or a Target
    /// override loses its target.
    pub override_choice: Option<VelocityReferenceFrame>,
    /// Previous frame's auto suggestion, for transition detection.
    pub last_suggested: Option<VelocityReferenceFrame>,
}

/// Next frame in the click cycle: Orbit → Surface → Target → Orbit,
/// skipping Target when no target is selected.
pub fn next_frame(
    current: VelocityReferenceFrame,
    target_available: bool,
) -> VelocityReferenceFrame {
    use VelocityReferenceFrame::*;
    match current {
        Orbit => Surface,
        Surface => {
            if target_available {
                Target
            } else {
                Orbit
            }
        }
        Target => Orbit,
    }
}

#[derive(Resource, Default)]
pub struct WarpToManeuver {
    /// `true` while the auto-warp is engaged. Cleared on arrival, when
    /// no upcoming maneuver remains, when an active burn takes over, or
    /// when the player nudges warp manually (handled in
    /// [`crate::bridge::handle_warp_controls`]).
    pub active: bool,
    /// Latest target as of the most recent system tick — drives the
    /// HUD readout. `None` whenever auto-warp is off.
    pub current: Option<ManeuverTarget>,
}

/// Soonest scheduled maneuver after `sim_time`. Skips past zero-Δv
/// nodes (placeholders that wouldn't fire). The strict `>` filter
/// rejects a node sitting at the current epoch — that node is already
/// mid-execution or stale.
pub fn find_next_maneuver(sim_time: f64, simulation: &Simulation) -> Option<ManeuverTarget> {
    for node in simulation.maneuvers().iter() {
        if node.time <= sim_time {
            continue;
        }
        let dv_mag = node.delta_v.length();
        if dv_mag <= 0.0 {
            continue;
        }
        return Some(ManeuverTarget {
            epoch: node.time,
            duration_s: simulation.estimated_burn_duration(dv_mag),
        });
    }
    None
}

#[derive(Clone, Copy, Debug)]
pub struct ManeuverTarget {
    pub epoch: f64,
    /// Tsiolkovsky burn duration computed when the target was selected.
    /// Lets the safe-target calculation reuse the autopilot's lead
    /// formula without re-querying the simulation.
    pub duration_s: f64,
}

/// Discrete ship-orientation modes the player can request.
///
/// `None` in [`NavigationState::mode`] means free flight (no auto-orient).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationMode {
    /// Hold current attitude (kill rotation).
    Stability,
    /// Point along orbital velocity.
    Prograde,
    /// Point against orbital velocity.
    Retrograde,
    /// Point along the orbital plane normal.
    Normal,
    /// Point against the orbital plane normal.
    AntiNormal,
    /// Point toward the parent body.
    RadialIn,
    /// Point away from the parent body.
    RadialOut,
    /// Point toward the selected target.
    Target,
    /// Point away from the selected target.
    AntiTarget,
    /// Point along the next maneuver node's burn direction.
    ManeuverNode,
}

/// Currently requested orientation mode.
///
/// `mode` selects the autopilot's pointing target (`None` = free
/// flight). Scheduled burn execution is owned by
/// [`crate::autopilot::Autopilot`], not by the maneuver/navigation
/// UI state.
#[derive(Resource, Debug, Default)]
pub struct NavigationState {
    pub mode: Option<NavigationMode>,
}

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
    pub fn burn_start(self) -> f64 {
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
    pub next: Option<AutopilotBurnDirective>,
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

    pub fn get(&self, id: AutopilotDirectiveId) -> Option<AutopilotBurnDirective> {
        self.next.filter(|directive| directive.id == id)
    }
}

/// The shared scheduled-burn executor.
///
/// **Producer-agnostic and always available.** [`BurnArm`] says on whose
/// behalf it is running — the pilot, for hand-placed maneuver nodes, or the
/// engaged [`FlightProgram`], for nodes that program installed. It used to
/// be gated on an `AutoflightMode` enum whose `Orbit` variant had to alias
/// `Maneuver` in three separate places so the ascent program could reuse
/// this executor; arming is the honest way to express that, and it removes
/// the aliasing.
#[derive(Resource, Debug, Default)]
pub struct Autopilot {
    pub arm: BurnArm,
    pub state: AutopilotState,
}

impl Autopilot {
    pub fn arm(&self) -> BurnArm {
        self.arm
    }

    /// Arm or disarm. Disarming always resets the executor: a directive
    /// half-flown by a disarmed executor is the state that leaves a
    /// throttle asserted with nobody owning it.
    pub fn set_arm(&mut self, arm: BurnArm) {
        if self.arm != arm {
            self.arm = arm;
            if !arm.armed() {
                self.state = AutopilotState::Idle;
            }
        }
    }

    pub fn disarm(&mut self) {
        self.set_arm(BurnArm::Off);
    }

    /// Snapshot of the current executor state for UI/read-only systems.
    pub fn state(&self) -> AutopilotState {
        self.state
    }

    /// `true` when the executor is actively driving the ship — i.e. state
    /// is `Engaging` or `Burn`.
    pub fn is_active(&self) -> bool {
        self.arm.armed()
            && matches!(
                self.state,
                AutopilotState::Engaging { .. } | AutopilotState::Burn { .. }
            )
    }

    /// What this executor requires locked out of pilot reach *right now*.
    ///
    /// `Armed` deliberately declares nothing: a directive placed a year out
    /// must not disable the player's controls for the entire wait. Only
    /// `Engaging`/`Burn` are time-critical enough that a warp advance would
    /// integrate straight through the burn.
    pub fn required_locks(&self) -> AutoflightLocks {
        if self.is_active() {
            AutoflightLocks::FULL_AUTHORITY
        } else {
            AutoflightLocks::NONE
        }
    }

    /// Direct world-frame attitude target while the autopilot owns
    /// pointing. This bypasses `NavigationMode::ManeuverNode` so the
    /// executor can fly any directive producer, not only maneuver nodes.
    pub fn attitude_target(&self) -> Option<DVec3> {
        match self.state {
            AutopilotState::Engaging { direction, .. } | AutopilotState::Burn { direction, .. } => {
                Some(direction)
            }
            AutopilotState::Idle | AutopilotState::Armed { .. } => None,
        }
    }

    /// The autopilot's attitude request for the fly-by-wire control bus
    /// ([`crate::control_bus`]). `PointNose` at the burn direction while
    /// engaging/burning, otherwise `Free` (it yields attitude to lower-
    /// priority sources). The arbiter ranks this above nav modes and the SAS
    /// hold but below an unlocked pilot stick.
    pub fn demand(&self) -> thalos_control::ControlDemand {
        if !self.arm.armed() {
            return thalos_control::ControlDemand::NONE;
        }
        match self.state {
            AutopilotState::Engaging { direction, .. } => {
                thalos_control::ControlDemand::autoflight(
                    thalos_control::AttitudeDemand::PointNose(direction),
                    Some(0.0),
                    None,
                    None,
                )
            }
            AutopilotState::Burn { direction, .. } => thalos_control::ControlDemand::autoflight(
                thalos_control::AttitudeDemand::PointNose(direction),
                Some(1.0),
                None,
                None,
            ),
            AutopilotState::Idle | AutopilotState::Armed { .. } => {
                thalos_control::ControlDemand::NONE
            }
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
        /// Value of [`thalos_physics_canonical::simulation::Simulation::delivered_dv`]
        /// captured at the moment the burn started; subtracted from the
        /// live value each frame to get "Δv delivered since burn start."
        anchor_delivered_dv: f64,
    },
}

/// Emitted when a generic burn directive starts.
///
/// Producer adapters decide whether the id belongs to them. The
/// maneuver adapter uses this to remove the executing node from future
/// flight-plan input.
#[derive(Debug, Clone, Copy, Message)]
pub struct AutopilotBurnStarted {
    pub id: AutopilotDirectiveId,
}

/// Emitted when a generic burn directive completes.
///
/// Producer adapters decide whether the id belongs to them. The
/// maneuver adapter treats this as an idempotent cleanup fallback; the
/// executing node is normally retired on [`AutopilotBurnStarted`].
#[derive(Debug, Clone, Copy, Message)]
pub struct AutopilotBurnCompleted {
    pub id: AutopilotDirectiveId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrbitShape {
    #[default]
    Circular,
    Elliptical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrbitPlaneChoice {
    /// Cheapest direct-ascent plane from the current site. On orbit this
    /// becomes `PreserveCurrent`, avoiding an unnecessary plane change.
    #[default]
    Auto,
    Preserve,
    Nearest,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitDraft {
    pub shape: OrbitShape,
    pub periapsis_altitude_m: f64,
    pub apoapsis_altitude_m: f64,
    pub inclination_rad: f64,
    pub direction: OrbitDirection,
    pub plane: OrbitPlaneChoice,
}

impl Default for OrbitDraft {
    fn default() -> Self {
        Self {
            shape: OrbitShape::Circular,
            periapsis_altitude_m: 200_000.0,
            apoapsis_altitude_m: 200_000.0,
            inclination_rad: 0.0,
            direction: OrbitDirection::Prograde,
            plane: OrbitPlaneChoice::Auto,
        }
    }
}

impl OrbitDraft {
    pub fn target(self, reference_body: usize) -> TargetOrbit {
        let periapsis_altitude_m = self.periapsis_altitude_m.min(self.apoapsis_altitude_m);
        let apoapsis_altitude_m = self.apoapsis_altitude_m.max(periapsis_altitude_m);
        TargetOrbit {
            reference_body,
            periapsis_altitude_m,
            apoapsis_altitude_m,
            plane: match self.plane {
                OrbitPlaneChoice::Auto | OrbitPlaneChoice::Preserve => TargetPlane::PreserveCurrent,
                OrbitPlaneChoice::Nearest => TargetPlane::Nearest {
                    inclination_rad: self.inclination_rad,
                    direction: self.direction,
                },
            },
        }
    }

    pub fn normalize(&mut self) {
        self.periapsis_altitude_m = self
            .periapsis_altitude_m
            .clamp(MIN_ORBIT_ALTITUDE_M, MAX_ORBIT_ALTITUDE_M);
        self.apoapsis_altitude_m = self
            .apoapsis_altitude_m
            .clamp(MIN_ORBIT_ALTITUDE_M, MAX_ORBIT_ALTITUDE_M);
        self.inclination_rad = self.inclination_rad.clamp(0.0, 90.0_f64.to_radians());
        if self.shape == OrbitShape::Circular {
            self.periapsis_altitude_m = self.apoapsis_altitude_m;
        } else if self.periapsis_altitude_m > self.apoapsis_altitude_m {
            std::mem::swap(
                &mut self.periapsis_altitude_m,
                &mut self.apoapsis_altitude_m,
            );
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OrbitProgramPhase {
    #[default]
    Idle,
    Planned,
    Preflight,
    Wait,
    Rise,
    Turn,
    Ascent,
    MainEngineCutoff,
    Coast,
    Circularize,
    Trim,
    Complete,
    Abort,
}

impl OrbitProgramPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "IDLE",
            Self::Planned => "PLANNED",
            Self::Preflight => "PREFLT",
            Self::Wait => "WAIT",
            Self::Rise => "RISE",
            Self::Turn => "TURN",
            Self::Ascent => "ASCENT",
            Self::MainEngineCutoff => "MECO",
            Self::Coast => "COAST",
            Self::Circularize => "CIRC",
            Self::Trim => "TRIM",
            Self::Complete => "COMPLETE",
            Self::Abort => "ABORT",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrbitPlanSummary {
    pub node_count: usize,
    pub total_delta_v_m_s: f64,
    pub predicted_periapsis_altitude_m: f64,
    pub predicted_apoapsis_altitude_m: f64,
    pub predicted_inclination_rad: f64,
}

#[derive(Resource, Debug)]
pub struct OrbitProgram {
    pub draft: OrbitDraft,
    pub phase: OrbitProgramPhase,
    pub target_body: Option<usize>,
    pub summary: Option<OrbitPlanSummary>,
    pub error: Option<String>,
    pub program_id: u64,
    pub surface_program: bool,
    pub demand: ControlDemand,
    pub launch_altitude_m: f64,
    pub phase_started_s: f64,
    pub diagnostic_s: f64,
    pub within_tolerance_s: f64,
    pub target_plane_normal: DVec3,
    pub idle_handoff_pending: bool,
    /// The next armed sequencing event, published for the annunciator.
    /// Written by the ascent guidance loop from the staging sequencer and
    /// the MECO criterion.
    pub sequence: SequenceEvent,
}

impl Default for OrbitProgram {
    fn default() -> Self {
        Self {
            draft: OrbitDraft::default(),
            phase: OrbitProgramPhase::Idle,
            target_body: None,
            summary: None,
            error: None,
            program_id: 1,
            surface_program: false,
            demand: ControlDemand::NONE,
            launch_altitude_m: 0.0,
            phase_started_s: 0.0,
            diagnostic_s: 0.0,
            within_tolerance_s: 0.0,
            target_plane_normal: DVec3::ZERO,
            idle_handoff_pending: false,
            sequence: SequenceEvent::None,
        }
    }
}

impl OrbitProgram {
    pub fn demand(&self) -> ControlDemand {
        self.demand
    }

    pub fn active(&self) -> bool {
        !matches!(
            self.phase,
            OrbitProgramPhase::Idle
                | OrbitProgramPhase::Planned
                | OrbitProgramPhase::Complete
                | OrbitProgramPhase::Abort
        )
    }

    /// What the ascent program requires locked, **per phase**.
    ///
    /// The distinction this method exists to make: powered flight and the
    /// staging sequence are time-critical (a warp advance integrates
    /// straight through thrust, and `activate_stage` is gated to 1×), while
    /// the ballistic coast to the circularisation node is not. The old lock
    /// table could not express that — it locked warp for the whole program
    /// — which killed warp-to-node during the one phase that is nothing but
    /// waiting. During `Circularize` the burn executor declares its own
    /// `FULL_AUTHORITY` and the union restores the warp lock, so this can
    /// stay honest about what the *program* needs.
    pub fn required_locks(&self) -> AutoflightLocks {
        match self.phase {
            OrbitProgramPhase::Idle
            | OrbitProgramPhase::Planned
            | OrbitProgramPhase::Complete
            | OrbitProgramPhase::Abort => AutoflightLocks::NONE,
            OrbitProgramPhase::Preflight
            | OrbitProgramPhase::Wait
            | OrbitProgramPhase::Rise
            | OrbitProgramPhase::Turn
            | OrbitProgramPhase::Ascent
            | OrbitProgramPhase::MainEngineCutoff => AutoflightLocks::FULL_AUTHORITY,
            OrbitProgramPhase::Coast
            | OrbitProgramPhase::Circularize
            | OrbitProgramPhase::Trim => AutoflightLocks::GUIDANCE_COAST,
        }
    }

    /// `true` while the program is flying a continuous guidance law rather
    /// than delegating to the burn executor. Drives the `GUID` annunciation
    /// and decides which source fills the autopilot slot.
    pub fn guidance_active(&self) -> bool {
        self.surface_program && self.active()
    }
}

#[derive(Debug, Clone, Copy, Message)]
pub enum OrbitTargetRequest {
    ToggleShape,
    AdjustPeriapsis(i8),
    AdjustApoapsis(i8),
    AdjustInclination(i8),
    ToggleDirection,
    TogglePlane,
    Plan,
    Execute,
    Cancel,
}

/// Maximum bank the planner may require and the guidance may command (rad).
/// 25° is the airliner standard for maneuvering in the terminal area — steep
/// enough to turn in reasonable airspace, shallow enough to be comfortable and
/// to leave stall margin at approach speed.
pub const BANK_LIMIT_RAD: f64 = 0.436_332_313;

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
    /// Along-path distance of each point (m), same length as `path_points`.
    ///
    /// Earns its place by answering "which of these points are still ahead of
    /// me": the plan freezes once established on final, so the points behind the
    /// craft would otherwise keep the ND framed on an intercept that was flown
    /// twenty kilometres ago.
    pub path_along_m: Vec<f64>,
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
    /// The flyable path back onto the route from where the craft actually is,
    /// tessellated like `path_points`. Empty while the craft tracks the route
    /// closely enough that drawing it would only thicken the line.
    ///
    /// This is **guidance, not a re-plan** — the route itself is untouched. See
    /// `thalos_navigation::rejoin`.
    pub rejoin_points: Vec<DVec3>,
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
    /// Spherical guidance while the selected runway is outside the terminal
    /// approach region. Exactly one of this and `guidance` is populated.
    pub destination_guidance: Option<DestinationGuidance>,
    /// Body-fixed arrival fix for the destination leg.
    pub destination_arrival_dir: Option<DVec3>,
    /// Every landable end on the dominant body, **nearest threshold first**.
    pub ends: Vec<RunwayEndEntry>,
    pub display: RouteDisplay,
    /// Approach speed the plan was built with (m/s) — shown as the speed target
    /// and used by the speed gates.
    pub approach_speed_m_s: f64,
    /// Which end the current plan belongs to, so a selection change is detected.
    pub planned_for: Option<ArmedEnd>,
    /// Real time of the last (re)plan.
    pub planned_at_s: f32,
    /// Latched once the craft reaches the final segment: freezes the plan (see
    /// the module docs — re-planning on final flies you away from the runway).
    pub established: bool,
    /// Where along the route the rejoin captured last frame. Fed back as a hint
    /// so the capture point holds still while the craft flies toward it —
    /// `plan_rejoin` is pure, so this frame-to-frame memory lives here
    /// (ADR-20260730T005746Z).
    pub rejoin_hint_along_m: Option<f64>,
    /// Recovery asks the destination leg to carry the craft back behind the
    /// runway even when that arrival fix is already inside the ordinary
    /// terminal-capture radius.
    pub force_destination_ingress: bool,
}

impl RouteState {
    /// Invalidate a frozen final after a go-around and require a fresh ingress
    /// to the selected runway's arrival fix.
    pub fn recover_to_destination_ingress(&mut self) {
        self.plan = None;
        self.guidance = None;
        self.established = false;
        self.rejoin_hint_along_m = None;
        self.display = RouteDisplay::default();
        self.force_destination_ingress = true;
    }
}

/// Stateful LAND executor. `demand` is private to keep this module the sole
/// state-machine writer; the control bus reads it through [`demand`].
#[derive(Resource, Debug)]
pub struct LandAutopilot {
    /// Whether the landing program is *selected*, independent of the
    /// phase it has reached. Previously this was carried by the
    /// `AutoflightMode::Land` enum variant, which is what forced the
    /// program's engagement to share a slot with the burn executor's.
    pub engaged: bool,
    pub phase: LandPhase,
    pub demand: ControlDemand,
    pub speed_integral: f64,
    pub contact_s: f64,
    pub airborne_s: f64,
    pub stopped_s: f64,
    pub go_arounds: u8,
    pub go_around_s: f64,
    pub diagnostic_s: f64,
}

/// Public phase for the HUD and diagnostics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LandPhase {
    #[default]
    Off,
    Enroute,
    TerminalCapture,
    Final,
    Flare,
    Rollout,
    GoAround,
    Stopped,
    Unable,
}

/// World-frame unit vector pointing along the next maneuver node's Δv.
/// Returns `None` when no node exists or the burn direction is
/// degenerate.
///
/// Pointing for a maneuver requires the ship and reference body states
/// *at burn time* — using "now" gets the wrong PRN frame for any
/// non-instant burn. Uses the cached prediction; when it's missing
/// (right after a node edit) falls back to *both* states at current
/// time so the PRN frame stays internally consistent rather than
/// mixing ship-now with body-future.
///
/// Shared by [`compute_target_direction`] (driving the
/// [`NavigationMode::ManeuverNode`] pointing target) and the burn-
/// directive publisher's direction calculation.
pub fn maneuver_burn_direction(sim: &Simulation, plan: &ManeuverPlan) -> Option<DVec3> {
    // Point at the next burn the autopilot would fly — a still-planned or
    // currently-executing node — never a spent one lingering for display.
    maneuver_node_burn_direction(sim, plan.nodes.iter().find(|n| n.drives_directive())?)
}

/// World-frame unit vector pointing along a maneuver node's Δv.
pub fn maneuver_node_burn_direction(sim: &Simulation, node: &GameNode) -> Option<DVec3> {
    let ship = sim.ship_state();
    let time = sim.sim_time();
    let prediction_state = sim
        .prediction()
        .and_then(|p| p.pre_burn_state_at(node.time, sim.ephemeris(), sim.bodies()))
        .map(|s| thalos_world::StateVector {
            position: s.position,
            velocity: s.velocity,
        })
        .or_else(|| sim.prediction().and_then(|p| p.state_at(node.time)));
    let (ship_pos, ship_vel, frame_time) = match prediction_state {
        Some(s) => (s.position, s.velocity, node.time),
        None => (ship.position, ship.velocity, time),
    };
    let body_state = sim.ephemeris().state(
        node.reference_body,
        thalos_physics_canonical::canonical::Epoch(frame_time),
    );
    let dv_world = delta_v_to_world(
        node.delta_v,
        ship_vel,
        ship_pos,
        body_state.position,
        body_state.velocity,
    );
    safe_normalize(dv_world)
}

pub const MIN_ORBIT_ALTITUDE_M: f64 = 10_000.0;

pub const MAX_ORBIT_ALTITUDE_M: f64 = 50_000_000.0;

/// `try_normalize` with an explicit `None` for degenerate vectors — shared
/// by the guidance math here and the runtime's pointing laws.
pub fn safe_normalize(v: DVec3) -> Option<DVec3> {
    if v.length_squared() < 1e-20 {
        None
    } else {
        Some(v.normalize())
    }
}

impl VelocityFrameState {
    /// Pin a manual override (called by the speed-readout click handler).
    pub fn set_override(&mut self, frame: VelocityReferenceFrame) {
        self.override_choice = Some(frame);
    }
}

impl ManeuverTarget {
    /// Short label such as "Maneuver in 1h 23m".
    pub fn label(&self, now: f64) -> String {
        let remaining = (self.epoch - now).max(0.0);
        format!("Maneuver in {}", format_duration(remaining))
    }
}

impl WarpToManeuver {
    pub fn cancel(&mut self) {
        self.active = false;
        self.current = None;
    }
}

impl LandAutopilot {
    pub fn phase(&self) -> LandPhase {
        self.phase
    }

    pub fn demand(&self) -> ControlDemand {
        self.demand
    }

    pub fn active(&self) -> bool {
        !matches!(
            self.phase,
            LandPhase::Off | LandPhase::Stopped | LandPhase::Unable
        )
    }

    /// What the landing program requires locked.
    ///
    /// Approach and landing is time-critical throughout — every phase is
    /// either flying a profile against terrain or rolling out on it — so
    /// unlike the ascent program there is no phase where warp may be handed
    /// back. It additionally owns the ground channels, which no other
    /// automation touches.
    pub fn required_locks(&self) -> AutoflightLocks {
        if !self.active() {
            return AutoflightLocks::NONE;
        }
        AutoflightLocks {
            ground_steer: true,
            wheel_brake: true,
            ..AutoflightLocks::FULL_AUTHORITY
        }
    }

    pub fn reset_for_engagement(&mut self, phase: LandPhase) {
        self.engaged = true;
        self.phase = phase;
        self.demand = ControlDemand::NONE;
        self.speed_integral = 0.0;
        self.contact_s = 0.0;
        self.airborne_s = 0.0;
        self.stopped_s = 0.0;
        self.go_arounds = 0;
        self.go_around_s = 0.0;
        self.diagnostic_s = 0.0;
    }

    pub fn set_phase(&mut self, next: LandPhase) {
        if self.phase == next {
            return;
        }
        info!(
            target: "thalos::diagnostic::approach_ap",
            event = "land_phase",
            from = ?self.phase,
            to = ?next,
            go_arounds = self.go_arounds,
            "LAND phase transition"
        );
        self.phase = next;
    }
}

impl Default for WarpLimits {
    fn default() -> Self {
        Self {
            max_level: usize::MAX,
        }
    }
}

impl Default for LandAutopilot {
    fn default() -> Self {
        Self {
            engaged: false,
            phase: LandPhase::Off,
            demand: ControlDemand::NONE,
            speed_integral: 0.0,
            contact_s: 0.0,
            airborne_s: 0.0,
            stopped_s: 0.0,
            go_arounds: 0,
            go_around_s: 0.0,
            diagnostic_s: 0.0,
        }
    }
}

pub fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.0}s", seconds)
    } else if seconds < 3600.0 {
        let m = (seconds / 60.0).floor();
        let s = seconds - m * 60.0;
        format!("{:.0}m {:02.0}s", m, s)
    } else if seconds < 86400.0 {
        let h = (seconds / 3600.0).floor();
        let m = ((seconds - h * 3600.0) / 60.0).floor();
        format!("{:.0}h {:02.0}m", h, m)
    } else {
        let d = (seconds / 86400.0).floor();
        let h = ((seconds - d * 86400.0) / 3600.0).floor();
        format!("{:.0}d {:02.0}h", d, h)
    }
}

pub fn in_map_view(view: Res<ViewMode>) -> bool {
    *view == ViewMode::Map
}

/// Body-frame "nose" axis for ship pointing. Apollo-style stacks have
/// their long axis along body Y, with the command pod at +Y; flipping
/// this would also flip the autopilot's pointing convention.
pub const SHIP_NOSE_BODY: DVec3 = thalos_control::NOSE_BODY;
