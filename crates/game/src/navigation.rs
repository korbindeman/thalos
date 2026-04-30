//! Ship-orientation mode widget and attitude autopilot.
//!
//! [`NavigationState`] holds the player's current orientation request.
//! The widget toggles modes via the side panel; [`compute_attitude_control`]
//! turns the active mode into a [`ControlInput`] for the physics
//! simulation each frame, called from
//! [`crate::bridge::handle_attitude_controls`]. Player keyboard input
//! takes priority over any active mode.

use bevy::math::DVec3;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};
use thalos_physics::maneuver::{delta_v_to_world, orbital_frame};
use thalos_physics::simulation::Simulation;
use thalos_physics::trajectory::Trajectory;
use thalos_physics::types::ControlInput;

use crate::autopilot::{Autopilot, AutopilotBurnSchedule, AutopilotState};
use crate::controls::ControlLocks;
use crate::maneuver::{GameNode, ManeuverPlan};
use crate::photo_mode::not_in_photo_mode;
use crate::rendering::SimulationState;
use crate::target::TargetBody;

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

impl NavigationMode {
    fn label(self) -> &'static str {
        match self {
            Self::Stability => "Stability",
            Self::Prograde => "Prograde",
            Self::Retrograde => "Retrograde",
            Self::Normal => "Normal",
            Self::AntiNormal => "Anti-Normal",
            Self::RadialIn => "Radial-In",
            Self::RadialOut => "Radial-Out",
            Self::Target => "Target",
            Self::AntiTarget => "Anti-Target",
            Self::ManeuverNode => "Maneuver",
        }
    }
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

pub struct NavigationPlugin;

impl Plugin for NavigationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NavigationState>().add_systems(
            bevy_egui::EguiPrimaryContextPass,
            navigation_panel.run_if(not_in_photo_mode),
        );
    }
}

fn navigation_panel(
    mut contexts: EguiContexts,
    mut nav: ResMut<NavigationState>,
    mut autopilot: ResMut<Autopilot>,
    burn_schedule: Res<AutopilotBurnSchedule>,
    locks: Res<ControlLocks>,
    sim: Res<SimulationState>,
    target: Res<TargetBody>,
    plan: Res<ManeuverPlan>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    let has_target = target.target.is_some();
    let has_node = !plan.nodes.is_empty();
    // Mode buttons go read-only when the navigation_mode lock is set —
    // the autopilot (or any future locker) is asserting a programmatic
    // pointing target and a click would be ignored until control is
    // released.
    let modes_enabled = !locks.navigation_mode;

    // Drop the selected mode if the precondition that justified it is gone
    // (target deselected, last maneuver node deleted).
    match nav.mode {
        Some(NavigationMode::Target | NavigationMode::AntiTarget) if !has_target => {
            nav.mode = None;
        }
        Some(NavigationMode::ManeuverNode) if !has_node => {
            nav.mode = None;
        }
        _ => {}
    }

    let avail = ctx.available_rect();
    let initial_pos = egui::pos2(avail.right() - 140.0, avail.center().y - 130.0);

    egui::Window::new("Navigation")
        .default_pos(initial_pos)
        .resizable(false)
        .show(ctx, |ui| {
            ui.set_min_width(140.0);
            // Checkbox stays interactive at all times — this is the
            // only override path while the autopilot is locked on.
            ui.checkbox(&mut autopilot.enabled, "Autopilot");
            autopilot_status(ui, &autopilot, &burn_schedule, &sim);
            ui.separator();
            ui.add_enabled_ui(modes_enabled, |ui| {
                mode_button(ui, &mut nav.mode, NavigationMode::Stability, true);
                ui.separator();
                mode_button(ui, &mut nav.mode, NavigationMode::Prograde, true);
                mode_button(ui, &mut nav.mode, NavigationMode::Retrograde, true);
                ui.separator();
                mode_button(ui, &mut nav.mode, NavigationMode::Normal, true);
                mode_button(ui, &mut nav.mode, NavigationMode::AntiNormal, true);
                ui.separator();
                mode_button(ui, &mut nav.mode, NavigationMode::RadialIn, true);
                mode_button(ui, &mut nav.mode, NavigationMode::RadialOut, true);
                ui.separator();
                mode_button(ui, &mut nav.mode, NavigationMode::Target, has_target);
                mode_button(ui, &mut nav.mode, NavigationMode::AntiTarget, has_target);
                ui.separator();
                mode_button(ui, &mut nav.mode, NavigationMode::ManeuverNode, has_node);
            });
        });
}

/// Render the autopilot's live status under the autopilot checkbox.
///
/// - `Idle`: collapses to nothing.
/// - `Armed`: subtle gray "armed · T-XXs" — no lockout, just a hint
///   the autopilot is waiting in the wings.
/// - `Engaging`: yellow "ENGAGING · burn in T-XXs" — controls are
///   locked, warp is being ramped, attitude is being settled.
/// - `Burn`: bold red banner with delivered vs. planned Δv and a
///   progress bar so the player can't miss that the engine is firing.
fn autopilot_status(
    ui: &mut egui::Ui,
    autopilot: &Autopilot,
    burn_schedule: &AutopilotBurnSchedule,
    sim: &SimulationState,
) {
    let countdown_to_burn = |state: AutopilotState| -> Option<f64> {
        let directive_id = match state {
            AutopilotState::Armed { directive_id }
            | AutopilotState::Engaging { directive_id, .. }
            | AutopilotState::Burn { directive_id, .. } => directive_id,
            AutopilotState::Idle => return None,
        };
        let directive = burn_schedule.next().filter(|d| d.id == directive_id)?;
        let now = sim.simulation.sim_time();
        let burn_start = directive.center_time - directive.duration_s / 2.0;
        Some(burn_start - now)
    };

    match autopilot.state() {
        AutopilotState::Idle => {}
        state @ AutopilotState::Armed { .. } => {
            let Some(t_minus) = countdown_to_burn(state) else {
                return;
            };
            ui.add_space(4.0);
            ui.colored_label(
                egui::Color32::from_rgb(170, 170, 170),
                format!("armed \u{00B7} T{:+.0}s", t_minus),
            );
        }
        state @ AutopilotState::Engaging { .. } => {
            let Some(t_minus) = countdown_to_burn(state) else {
                return;
            };
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("ENGAGING")
                    .color(egui::Color32::from_rgb(255, 200, 60))
                    .strong(),
            );
            ui.label(format!("burn in T{:+.0}s", t_minus));
        }
        AutopilotState::Burn {
            planned_dv,
            anchor_delivered_dv,
            ..
        } => {
            let delivered = (sim.simulation.delivered_dv() - anchor_delivered_dv).max(0.0);
            let progress = if planned_dv > 0.0 {
                (delivered / planned_dv).clamp(0.0, 1.0) as f32
            } else {
                0.0
            };
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("\u{25CF} BURNING")
                    .color(egui::Color32::from_rgb(255, 80, 80))
                    .strong()
                    .size(16.0),
            );
            ui.label(format!(
                "\u{0394}v {:.1} / {:.1} m/s",
                delivered, planned_dv
            ));
            ui.add(egui::ProgressBar::new(progress).desired_width(130.0));
        }
    }
}

fn mode_button(
    ui: &mut egui::Ui,
    current: &mut Option<NavigationMode>,
    mode: NavigationMode,
    enabled: bool,
) {
    let active = *current == Some(mode);
    let resp = ui.add_enabled(enabled, egui::Button::selectable(active, mode.label()));
    if resp.clicked() {
        *current = if active { None } else { Some(mode) };
    }
}

// ---------------------------------------------------------------------------
// Autopilot
// ---------------------------------------------------------------------------

/// PD controller settling time, seconds. ω_n = π/T gives a quarter-period
/// of T/2 — the ship reaches the target attitude in ~T seconds when it
/// starts within the linear-torque regime, longer when the controller
/// saturates against `max_torque`. Read by the scheduled-burn autopilot
/// in [`crate::autopilot`] to size its lead time before a maneuver.
pub(crate) const AUTOPILOT_SETTLE_S: f64 = 2.0;

/// Body-frame "nose" axis for ship pointing. Apollo-style stacks have
/// their long axis along body Y, with the command pod at +Y; flipping
/// this would also flip the autopilot's pointing convention.
pub(crate) const SHIP_NOSE_BODY: DVec3 = DVec3::Y;

/// Resolve the active orientation request into a [`ControlInput`] for
/// [`Simulation::set_control`]. Priority order:
///
/// 1. **Player keyboard input** (any of W/A/S/D/Q/E pressed) overrides
///    navigation modes. Callers pass zero player torque while a
///    programmatic lock owns attitude.
/// 2. **Autopilot target** → direct scheduled-burn pointing target
///    supplied by [`crate::autopilot`].
/// 3. **Stability** mode → re-uses the integrator's SAS rate damper
///    (zero command, `sas_enabled: true`).
/// 4. **Directional modes** (Prograde/Retrograde/Normal/AntiNormal/
///    Radial/Target/AntiTarget/ManeuverNode) → run the PD autopilot
///    against the per-mode target direction.
/// 5. **Free flight** (no mode active, no player input) → either zero
///    torque or SAS damping per the T-key toggle state.
///
/// `target_unreachable` means the target was set but the world target
/// vector couldn't be resolved this frame (e.g. zero relative velocity
/// for prograde, missing prediction for ManeuverNode). In that case we
/// fall back to the free-flight branch instead of holding the last
/// command — better to do nothing than to point at the wrong thing.
pub fn compute_attitude_control(
    player_torque: DVec3,
    nav_mode: Option<NavigationMode>,
    autopilot_target: Option<DVec3>,
    target: &TargetBody,
    plan: &ManeuverPlan,
    sim: &Simulation,
    sas_toggle_state: bool,
) -> ControlInput {
    // Spread `..*sim.control()` to preserve fields this fn doesn't
    // own — currently `throttle`, written by the fuel-drain system in
    // a separate pass. Adding new orthogonal fields to `ControlInput`
    // (e.g. RCS later) will inherit the same protection automatically.
    let base = *sim.control();

    let player_active = player_torque.length_squared() > 1e-6;
    if player_active {
        return ControlInput {
            torque_command: player_torque,
            sas_enabled: false,
            ..base
        };
    }

    if let Some(target_dir) = autopilot_target {
        return ControlInput {
            torque_command: autopilot_command(target_dir, sim),
            sas_enabled: true,
            ..base
        };
    }

    let Some(mode) = nav_mode else {
        return ControlInput {
            torque_command: DVec3::ZERO,
            sas_enabled: sas_toggle_state,
            ..base
        };
    };

    if matches!(mode, NavigationMode::Stability) {
        return ControlInput {
            torque_command: DVec3::ZERO,
            sas_enabled: true,
            ..base
        };
    }

    let Some(target_dir) = compute_target_direction(mode, sim, target, plan) else {
        return ControlInput {
            torque_command: DVec3::ZERO,
            sas_enabled: sas_toggle_state,
            ..base
        };
    };

    // SAS=true is harmless while the autopilot is producing nonzero
    // torque (the integrator's rate damper only activates when the
    // command is ~zero) and cleanly takes over once the autopilot
    // converges, killing any residual ω the PD term doesn't catch.
    ControlInput {
        torque_command: autopilot_command(target_dir, sim),
        sas_enabled: true,
        ..base
    }
}

/// World-frame unit vector the ship's nose should point at, given the
/// active mode and the current sim/target/plan state. Returns `None`
/// when the target can't be computed (e.g. target body not selected
/// for [`NavigationMode::Target`], no maneuver node for
/// [`NavigationMode::ManeuverNode`], or a degenerate frame). Stability
/// is handled by the caller — this fn never returns a target for it.
fn compute_target_direction(
    mode: NavigationMode,
    sim: &Simulation,
    target: &TargetBody,
    plan: &ManeuverPlan,
) -> Option<DVec3> {
    let ship = sim.ship_state();
    let time = sim.sim_time();

    match mode {
        NavigationMode::Stability => None,

        NavigationMode::Prograde | NavigationMode::Retrograde => {
            let body = sim.dominant_body();
            let body_state = sim.ephemeris().query_body(body, time);
            let rel_vel = ship.velocity - body_state.velocity;
            let dir = safe_normalize(rel_vel)?;
            Some(if mode == NavigationMode::Prograde {
                dir
            } else {
                -dir
            })
        }

        NavigationMode::Normal | NavigationMode::AntiNormal => {
            let body = sim.dominant_body();
            let body_state = sim.ephemeris().query_body(body, time);
            let [_, normal, _] = orbital_frame(
                ship.position,
                ship.velocity,
                body_state.position,
                body_state.velocity,
            );
            Some(if mode == NavigationMode::Normal {
                normal
            } else {
                -normal
            })
        }

        NavigationMode::RadialIn | NavigationMode::RadialOut => {
            let body = sim.dominant_body();
            let body_state = sim.ephemeris().query_body(body, time);
            let radial_out = safe_normalize(ship.position - body_state.position)?;
            Some(if mode == NavigationMode::RadialOut {
                radial_out
            } else {
                -radial_out
            })
        }

        NavigationMode::Target | NavigationMode::AntiTarget => {
            let target_id = target.target?;
            let target_state = sim.ephemeris().query_body(target_id, time);
            let to_target = safe_normalize(target_state.position - ship.position)?;
            Some(if mode == NavigationMode::Target {
                to_target
            } else {
                -to_target
            })
        }

        NavigationMode::ManeuverNode => maneuver_burn_direction(sim, plan),
    }
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
pub(crate) fn maneuver_burn_direction(sim: &Simulation, plan: &ManeuverPlan) -> Option<DVec3> {
    maneuver_node_burn_direction(sim, plan.nodes.first()?)
}

/// World-frame unit vector pointing along a maneuver node's Δv.
pub(crate) fn maneuver_node_burn_direction(sim: &Simulation, node: &GameNode) -> Option<DVec3> {
    let ship = sim.ship_state();
    let time = sim.sim_time();
    let (ship_pos, ship_vel, frame_time) =
        match sim.prediction().and_then(|p| p.state_at(node.time)) {
            Some(s) => (s.position, s.velocity, node.time),
            None => (ship.position, ship.velocity, time),
        };
    let body_state = sim.ephemeris().query_body(node.reference_body, frame_time);
    let dv_world = delta_v_to_world(
        node.delta_v,
        ship_vel,
        ship_pos,
        body_state.position,
        body_state.velocity,
    );
    safe_normalize(dv_world)
}

/// PD controller mapping (target direction, current attitude, ship
/// inertia) → per-axis torque command in `[-1, 1]`. The pointing axis
/// is body +Y; roll about that axis is purely damped (no orientation
/// constraint about the nose).
fn autopilot_command(target_nose_world: DVec3, sim: &Simulation) -> DVec3 {
    let attitude = sim.attitude();
    let params = sim.ship_params();

    // Target in body frame so the cross product directly yields the
    // body-frame error axis ω needs to act about.
    let target_body = attitude.orientation.inverse() * target_nose_world;

    // `nose × target` gives axis × sin(angle). The Y component is
    // always zero (Y × anything has zero Y), so torque.y comes purely
    // from the −Kd·ω damping term — exactly what we want for roll.
    //
    // Near 180° error, sin(angle)→0 and the controller stalls. Inject
    // a kick about body X to break the symmetry; the integrator will
    // settle on a real axis once it starts moving.
    let dot_with_nose = target_body.y;
    let error_axis = if dot_with_nose < -0.99 {
        DVec3::X
    } else {
        SHIP_NOSE_BODY.cross(target_body)
    };

    let omega_n = std::f64::consts::PI / AUTOPILOT_SETTLE_S;
    let kp = params.moment_of_inertia * (omega_n * omega_n);
    let kd = params.moment_of_inertia * (2.0 * omega_n);

    let desired_torque = error_axis * kp - attitude.angular_velocity * kd;

    DVec3::new(
        normalize_axis(desired_torque.x, params.max_torque.x),
        normalize_axis(desired_torque.y, params.max_torque.y),
        normalize_axis(desired_torque.z, params.max_torque.z),
    )
}

fn normalize_axis(desired: f64, max: f64) -> f64 {
    if max > 0.0 {
        (desired / max).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn safe_normalize(v: DVec3) -> Option<DVec3> {
    if v.length_squared() < 1e-20 {
        None
    } else {
        Some(v.normalize())
    }
}
