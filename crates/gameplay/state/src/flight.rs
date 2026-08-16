//! Flight-control and craft-systems state the HUD (and other feature crates)
//! read: throttle, control locks, SAS/realized control, flight config, gear,
//! aero readouts, the on-foot controller state. Writers stay with their
//! owning runtime modules; each resource names its sole writer.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_control::AssistStatus;
use thalos_physics_canonical::aero::{AeroConfig, ControlInputs};
use thalos_shipyard::StageSummary;
use thalos_world::BodyId;

use crate::autoflight::ThrottleChannel;

/// Canonical throttle position, persisted across frames so every controller
/// moves the same control surface.
///
/// `commanded` is the current throttle position: pilot input and the winning
/// automatic throttle demand both write it. `effective` is what the engines
/// actually receive after fuel and warp gating. Automatic control therefore
/// leaves the throttle where it last moved it instead of revealing a hidden,
/// stale pilot setpoint when it disengages.
#[derive(Resource, Debug, Default)]
pub struct ThrottleState {
    pub commanded: f64,
    pub effective: f64,
}

impl ThrottleState {
    /// Commit the control-bus winner to the canonical throttle position.
    ///
    /// A source that yields leaves the control where it last moved it.
    /// Scheduled burns are the exception: ending a burn is itself a cutoff
    /// command, so the `Burn -> non-automatic owner` edge closes the throttle
    /// instead of leaving the engine firing at the burn setting. A deliberate
    /// pilot movement on that edge remains authoritative.
    pub fn apply_arbitration(
        &mut self,
        winner: Option<f64>,
        automatic_winner: bool,
        previous_channel: ThrottleChannel,
        pilot_moved: bool,
    ) {
        if previous_channel == ThrottleChannel::Burn && !automatic_winner && !pilot_moved {
            self.commanded = 0.0;
        } else if let Some(winner) = winner {
            self.commanded = winner.clamp(0.0, 1.0);
        }
    }
}

/// Per-control-surface lockout flags. `true` = a programmatic system
/// is currently driving this surface and human input should be
/// dropped. Defaults are all `false` (everything free).
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct ControlLocks {
    /// Throttle setting — HOTAS absolute axis, Z/X snap, Shift/Ctrl ramp,
    /// future throttle slider. Gated in [`crate::fuel::handle_throttle_input`].
    pub throttle: bool,
    /// Attitude torque commands — W/A/S/D/Q/E. Gated in
    /// [`crate::bridge::handle_attitude_controls`] (player_torque is
    /// zeroed; the autopilot's pointing target still runs through
    /// `compute_attitude_control`). The T (SAS toggle) key is not
    /// gated — it just flips a state bool that's irrelevant while the
    /// autopilot owns attitude.
    pub attitude: bool,
    /// Warp level changes — `.` / `,` / `\` keys, HUD `<` / `>` /
    /// `→ Next` buttons. Pause (Space, HUD pause if added) stays
    /// available unconditionally; that exemption lives in the warp
    /// handler itself rather than as a separate flag here.
    pub warp: bool,
    /// Navigation mode buttons in the side panel (Stability, Prograde,
    /// …, Maneuver). The autopilot checkbox at the top of the same
    /// panel is *not* gated — it's the only override path while the
    /// autopilot is engaged.
    pub navigation_mode: bool,
    /// Nosewheel / tiller steering.
    pub ground_steer: bool,
    /// Wheel braking.
    pub wheel_brake: bool,
}

/// Number of flap lever detents past UP (1 = TAKEOFF, 2 = LANDING).
pub const FLAP_DETENTS: u8 = 2;

/// Flap lever + spoiler state. The lever detent is written by
/// [`update_flight_config`] (the `F`/`R` keys) and the HUD flap gate
/// (`hud::flight_config_panel::handle_clicks`, click a segment to set that
/// detent directly); the actuator fractions are solely
/// [`update_flight_config`]'s. Read by the aero force
/// system ([`crate::aero::apply_aero_forces`]), the control-surface visuals
/// (`ship_view`), and the HUD flight-config pills.
#[derive(Resource, Reflect, Clone, Copy, Default)]
#[reflect(Resource)]
pub struct FlightConfig {
    /// Flap lever detent, `0..=FLAP_DETENTS` (0 = UP).
    pub flap_setting: u8,
    /// Actual flap actuator position in `[0, 1]` (chases the lever).
    pub flap_fraction: f64,
    /// Actual spoiler position in `[0, 1]` (chases the brakes toggle).
    pub spoiler_fraction: f64,
}

/// Free-flight SAS toggle state (the `T` key / the HUD SAS button). When
/// enabled and nothing higher-priority is engaged, the controller holds the
/// current attitude — the "centered stick = hold current attitude" behaviour,
/// and the arming switch for the plane fly-by-wire assist.
///
/// **Defaults on**: every craft spawns with SAS engaged (spaceships hold
/// attitude, planes fly FBW with auto-trim + stall protection), and the flag
/// survives destruction/respawn. Toggling off is the deliberate act.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SasState {
    pub enabled: bool,
}

/// The realized control-surface command published each frame by
/// [`realize_control`].
///
/// **Sole writer:** [`realize_control`]. Read by the aero force system
/// ([`crate::aero::apply_aero_forces`]) for control-surface deflections. The
/// matching reaction-wheel command lands directly in the simulation's
/// `ControlInput::torque_command` (consumed by `apply_local_forces`), so it is
/// not mirrored here.
#[derive(Component, Debug, Default, Clone, Copy)]
pub struct RealizedControl {
    /// Aero control-surface deflections fed to `evaluate_aero`.
    pub aero: ControlInputs,
    /// The controller's normalized attitude command this frame, body frame
    /// `[-1, 1]` (`x` pitch, `y` roll, `z` yaw) — the arbitrated pilot / SAS /
    /// nav / autopilot effort *before* the reaction-wheel↔aero split. This is
    /// what the control-surface visuals deflect to: it shows commanded control
    /// effort at full scale, independent of how the allocator happens to divide
    /// the torque (the allocated `aero` fraction collapses toward zero when aero
    /// authority dwarfs the reaction-wheel torque, so it is not a usable visual
    /// signal).
    pub command: DVec3,
    /// Flight-assist status this frame: whether the plane fly-by-wire law is
    /// engaged and whether stall protection is actively clamping the pitch
    /// command. Read by the HUD's SAS button.
    pub assist: AssistStatus,
}

/// Per-stage readout, published each frame from the live parts and the
/// [`StagingPlan`] and consumed by the bottom-right HUD panel. Empty when
/// there is no staged vessel (e.g. EVA). Each [`StageSummary`] is the shared
/// type from [`thalos_shipyard::staging`], so the HUD and the shipyard
/// editor's preview render the same shape.
///
/// **Sole writer:** [`publish_staging_summaries`].
#[derive(Resource, Default)]
pub struct StagingSummaries(pub Vec<StageSummary>);

/// Every wheel on a craft, attached to its Avian rigid body so
/// [`apply_landing_gear_forces`] can find them.
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct WheelSet {
    pub wheels: Vec<Wheel>,
}

/// Latched brakes (KSP-style, the B key). When engaged,
/// [`apply_landing_gear_forces`] replaces free rolling with a high-gain
/// fore/aft hold (clamped to the tyre friction circle), so the craft stays
/// put under gravity, slopes, and the residual settle — though full takeoff
/// thrust still overpowers it — and the spoilers deploy
/// ([`crate::flight_config`]), so the same latch is the in-air speedbrake
/// and the rollout lift dump.
///
/// Defaults **off** (most spawns are airborne and must not start with
/// spoilers out); the parked runway placement engages it explicitly so a
/// freshly-spawned aircraft holds on the strip
/// (`runway::finish_runway_spawn`). Reflect-registered (for a future debug UI).
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct ParkingBrake {
    pub engaged: bool,
}

/// Whether any landing-gear wheel is currently bearing load on the ground
/// ("weight on wheels"). Set each frame by [`apply_landing_gear_forces`] from
/// its per-wheel suspension raycast, and read in the aero pass
/// ([`crate::aero::apply_aero_forces`]) to drop all aero on a grounded craft
/// below the taxi airspeed floor, where the AoA is degenerate (the velocity is
/// suspension settle, not flow). Above that floor a grounded craft flies the
/// full aero model — rotation authority and ground-roll damping are real
/// aerodynamics. Reflect-registered (for a future debug UI).
#[derive(Resource, Default, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct WeightOnWheels {
    pub grounded: bool,
}

/// Landing-gear up/down latch (KSP-style, the G key). When `down`,
/// [`apply_landing_gear_forces`] runs the suspension; when up it stands down
/// entirely (no contact, no weight on wheels) and the gear meshes are hidden
/// (`ship_view::sync_gear_visibility`). Binary — there is no retraction
/// animation, so a half-deployed load state never exists.
///
/// Defaults **down**: every ground/approach spawn (runway, final, descent) needs
/// gear extended, and orbit/EVA craft have no wheels so the state is moot.
/// Retraction is interlocked against weight-on-wheels (see [`toggle_gear`]).
/// Reflect-registered (for a future debug UI).
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct GearState {
    pub down: bool,
}

/// `tile_lod_m` hint for queries that want the finest CPU-synthesizable
/// terrain detail. GPU-backed height sources prefer the resident atlas
/// when populated; when they fall back to the CPU pipeline this hint
/// drives `compute_detail_height` to its full octave count.
pub const PHYSICS_QUERY_TILE_LOD_M: f32 = 0.5;

/// Whether the EVA player is walking on terrain or coasting like a craft.
///
/// EVA is a full craft (KSP-style): it can stand on a surface or sit in
/// orbit. The two regimes need opposite state flow, so this flag picks one:
///
/// - `Grounded`: [`step_eva_controller`] owns the capsule pose, running the
///   body-fixed character physics, and the canonical→Avian snap stands down.
/// - `Airborne`: Kepler owns canonical translation and the snap drives the
///   capsule from canonical (exactly like a ship coasting in vacuum); the
///   character controller stands down.
///
/// Set explicitly by the EVA teleport actions — surface teleports ground it,
/// orbit teleports make it airborne. (Suborbital ballistic flight — jumping,
/// walking off a cliff — stays *within* the grounded regime; this flag is the
/// coarse surface↔orbit switch, not the per-frame grounded/airborne state,
/// which lives in [`ActivePlayerController::grounded`].)
/// Defaults to `Grounded` to match the startup surface spawn.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
#[reflect(Component)]
pub enum EvaMode {
    #[default]
    Grounded,
    Airborne,
}

#[derive(Resource, Default, Debug, Clone)]
pub struct PlayerControllerState {
    pub active: Option<ActivePlayerController>,
}

/// Flight readout for the HUD.
#[derive(Component, Default, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct AeroReadout {
    pub airspeed_ms: f64,
    pub dynamic_pressure_pa: f64,
    pub mach: f64,
    pub density_kgm3: f64,
    /// Net aero force magnitude (N) and the angle of attack (deg), for debug.
    pub force_n: f64,
    pub alpha_deg: f64,
}

/// The aircraft's aerodynamic config, computed from its blueprint by `ship_view`
/// and consumed when the Avian body spawns. Replaced on each spawn.
#[derive(Resource, Default)]
pub struct ShipAeroLayout {
    pub config: AeroConfig,
}

/// One landing-gear wheel as a **raycast suspension**, in the craft body frame.
///
/// All directions/points are craft-local (`X=right, Y=nose, Z=dorsal`) — the
/// same frame `gear_mesh` authors in — so `Rotation.0 * p` maps them into the
/// body-centered inertial frame the Avian rigid body lives in. Built once at
/// spawn from the gear parts ([`build_wheel_set`]) and cached so the per-frame
/// system does no part-tree walking.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct Wheel {
    /// The gear part entity this leg belongs to, so per-gearbox state (the
    /// visual compression offset) can be keyed back to its rendered mesh.
    pub source: Entity,
    /// Strut top at the host skin — the suspension ray origin.
    pub strut_top_local: DVec3,
    /// Suspension axis (belly-ward `r̂`): the ray direction and spring line.
    pub susp_dir_local: DVec3,
    /// Roll axis (`fore`): brake / rolling resistance act along this.
    pub roll_dir_local: DVec3,
    /// Axle axis (`lateral`): lateral grip resists slip along this.
    pub axle_dir_local: DVec3,
    pub strut_length: f64,
    pub wheel_radius: f64,
    /// Nose (single-leg) gear steers; main pairs do not.
    pub steerable: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ActivePlayerController {
    pub body_entity: Entity,
    pub visual_entity: Entity,
    pub body_id: BodyId,
    pub inertial_position_m: DVec3,
    /// Player position relative to the body centre, in the **body-fixed**
    /// (co-rotating) frame. `inertial_offset = body.orientation * pos_bf`.
    /// `ZERO` is the "uninitialised" sentinel — re-seeded from the rigid body
    /// on the first grounded frame and after any teleport.
    pub pos_bf: DVec3,
    /// Surface-relative velocity in the body-fixed frame (walking + vertical).
    pub vel_bf: DVec3,
    /// Horizontal facing direction in the body-fixed frame, slewed toward the
    /// movement direction for a smooth third-person pivot.
    pub facing_bf: DVec3,
    /// The body-centred inertial offset this controller last wrote to Avian's
    /// `Position`. Used to detect an *external* teleport (F9 drop / map plant)
    /// — `Position` changing to something the controller didn't write — without
    /// mistaking the body's normal per-frame co-rotation for one.
    pub last_avian_offset: DVec3,
    pub grounded: bool,
    pub at_rest: bool,
    pub rest_timer_s: f64,
    pub surface_speed_m_s: f64,
}

impl PlayerControllerState {
    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub fn active_position_m(&self) -> Option<DVec3> {
        self.active.map(|active| active.inertial_position_m)
    }

    /// Whether the on-foot player is standing on the surface (vs airborne /
    /// falling). `false` when there is no active EVA player.
    pub fn is_grounded(&self) -> bool {
        self.active.map(|a| a.grounded).unwrap_or(false)
    }

    /// Whether the on-foot player has been stationary on the surface long
    /// enough to be warp-eligible. `false` when there is no active EVA player.
    pub fn is_at_rest(&self) -> bool {
        self.active.map(|a| a.at_rest).unwrap_or(false)
    }

    /// Surface-relative speed (m/s) of the on-foot player — walking + vertical,
    /// excluding the body's co-rotation. `0.0` when there is no active player.
    pub fn surface_speed_m_s(&self) -> f64 {
        self.active.map(|a| a.surface_speed_m_s).unwrap_or(0.0)
    }
}

impl EvaMode {
    pub fn is_grounded(self) -> bool {
        matches!(self, EvaMode::Grounded)
    }
}

impl FlightConfig {
    /// Configuration for an aircraft already parked on a runway and ready to
    /// begin its takeoff roll.
    ///
    /// The actuator starts at the commanded detent rather than travelling from
    /// UP after the loading screen clears.
    pub fn runway_takeoff() -> Self {
        let flap_fraction = TAKEOFF_FLAP_DETENT as f64 / FLAP_DETENTS as f64;
        Self {
            flap_setting: TAKEOFF_FLAP_DETENT,
            flap_fraction,
            spoiler_fraction: 0.0,
        }
    }

    /// HUD label for a flap lever detent (0 = UP, `FLAP_DETENTS` = LANDING).
    pub fn detent_label(detent: u8) -> &'static str {
        match detent {
            0 => "UP",
            1 => "T/O",
            _ => "LDG",
        }
    }
}

/// Flap lever detent used for a runway-ready takeoff configuration.
pub const TAKEOFF_FLAP_DETENT: u8 = 1;

impl Default for GearState {
    fn default() -> Self {
        Self { down: true }
    }
}

impl Default for SasState {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// The live aero config attached to the player's Avian body.
#[derive(Component)]
pub struct ShipAero {
    pub config: AeroConfig,
}

/// Apply a requested gear position through the weight-on-wheels interlock.
/// Shared by the key ([`toggle_gear`]) and the HUD pill so both honour the same
/// rule: extending is always allowed; retracting is refused while grounded.
pub fn set_gear_down(gear: &mut GearState, weight_on_wheels: &WeightOnWheels, down: bool) {
    if down || !weight_on_wheels.grounded {
        gear.down = down;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_throttle_moves_the_canonical_control() {
        let mut throttle = ThrottleState {
            commanded: 0.22,
            ..default()
        };
        throttle.apply_arbitration(Some(0.76), true, ThrottleChannel::Pilot, false);
        assert_eq!(throttle.commanded, 0.76);
    }

    #[test]
    fn automatic_release_keeps_the_last_position() {
        let mut throttle = ThrottleState {
            commanded: 0.76,
            ..default()
        };
        throttle.apply_arbitration(None, false, ThrottleChannel::Guidance, false);
        assert_eq!(throttle.commanded, 0.76);
    }

    #[test]
    fn scheduled_burn_release_commands_cutoff() {
        let mut throttle = ThrottleState {
            commanded: 1.0,
            ..default()
        };
        // The unlocked pilot source republishes the last value, but that must
        // not hide the automatic burn-complete cutoff edge.
        throttle.apply_arbitration(Some(1.0), false, ThrottleChannel::Burn, false);
        assert_eq!(throttle.commanded, 0.0);
    }

    #[test]
    fn pilot_movement_wins_during_burn_disconnect() {
        let mut throttle = ThrottleState {
            commanded: 1.0,
            ..default()
        };
        throttle.apply_arbitration(Some(0.35), false, ThrottleChannel::Burn, true);
        assert_eq!(throttle.commanded, 0.35);
    }

    #[test]
    fn arbitration_clamps_the_canonical_control() {
        let mut throttle = ThrottleState {
            commanded: 0.5,
            ..default()
        };
        throttle.apply_arbitration(Some(1.5), true, ThrottleChannel::Guidance, false);
        assert_eq!(throttle.commanded, 1.0);
    }
}
