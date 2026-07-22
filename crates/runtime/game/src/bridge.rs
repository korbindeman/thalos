//! Bridge between the physics simulation and Bevy ECS.
//!
//! All simulation state lives in [`Simulation`] (physics crate). This module
//! is a thin adapter that:
//!
//! 1. Calls [`Simulation::step`] each frame to advance the ship.
//! 2. Recomputes trajectory prediction synchronously on the main thread
//!    whenever the maneuver plan is dirty or the cached result is stale, and
//!    *only* while the ship is on an actual ballistic trajectory. When the
//!    ship is landed (`BodyFixed`) or in `LocalRigidBody` with active
//!    terrain contact, the live velocity carries Avian's contact reactions —
//!    feeding that into Keplerian propagation would produce a wobbling
//!    curve the ship will never follow, so the prediction is cleared and
//!    the renderer shows nothing. A single `propagate_flight_plan` pass
//!    terminates early (stable orbit, terrain impact, SOI transition),
//!    keeping the typical pass well under one frame. Running in-line means
//!    an edit on frame N produces the fresh trajectory on frame N — no
//!    worker-thread lag.
//! 3. Maps keyboard input to warp controls.

use bevy::prelude::*;
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::canonical::{AuthorityMode, Epoch};
use thalos_physics_canonical::maneuver::ManeuverNode;
use thalos_physics_canonical::regime::PredictionDisplay;
use thalos_physics_local::{ActiveLocalBubble, LocalCraftBody};

use crate::SimStage;
use crate::controls::ControlLocks;
use crate::maneuver::ManeuverPlan;
use crate::rendering::SimulationState;
use crate::sim_clock::SimClock;
use crate::warp_to_maneuver::{WarpToManeuver, find_next_maneuver};

pub fn advance_simulation(clock: Res<SimClock>, mut sim: ResMut<SimulationState>) {
    let _span = tracing::info_span!("advance_simulation").entered();

    let pre_pos = sim.simulation.ship_state().position;
    let pre_t = sim.simulation.sim_time();

    sim.simulation.step(clock.delta_secs_f64());

    // Diagnostic: log anything that looks physically impossible so we can
    // catch state corruption the instant it happens instead of noticing it
    // visually a few frames later.
    let post_pos = sim.simulation.ship_state().position;
    let post_t = sim.simulation.sim_time();
    let dt = post_t - pre_t;
    let dx = (post_pos - pre_pos).length();
    let warp = sim.simulation.warp.speed();
    // A ship in LEO around Thalos moves ~7 km/s relative to the body, but
    // the body itself drifts heliocentrically at ~30 km/s. Cap at
    // 100 km/s * dt as a rough sanity ceiling.
    let max_reasonable = 1.0e5 * dt.max(1.0);
    if dt > 0.0 && dx > max_reasonable {
        warn!(
            "ship jumped {:.3e} m in {:.2}s (warp={:.0}x, ratio={:.3}x reasonable); pre=({:.3e},{:.3e},{:.3e}) post=({:.3e},{:.3e},{:.3e})",
            dx,
            dt,
            warp,
            dx / max_reasonable,
            pre_pos.x,
            pre_pos.y,
            pre_pos.z,
            post_pos.x,
            post_pos.y,
            post_pos.z,
        );
    }
    if !post_pos.is_finite() {
        warn!(
            "ship position went non-finite: {:?} at sim_time {:.3} (warp={:.0}x)",
            post_pos, post_t, warp,
        );
    }

    // The propagator's terrain-aware collision check requires `prev_alt > 0`
    // at every step boundary, so a state that's already at or below mean
    // radius — typically warp-out from a deep valley, or Avian readback
    // drift letting the ship sink a few cm into the collider — will
    // silently run unbounded Kepler instead of terminating at impact and
    // the trajectory flies through the planet. Forcing 1× warp here breaks
    // the runaway: subsequent frames advance one sim-second at a time so
    // the player notices, and the throttle gate already keeps any active
    // burn off until they return to 1×. 1 m tolerance keeps Avian's
    // normal contact noise quiet.
    //
    // Only kicks in for warp > 1×. Pause (0×) and 1× are both safe — pause
    // doesn't advance sim time at all, and 1× advances one real-second per
    // frame so any Kepler runaway is visible. Without this guard, a player
    // on EVA in a valley below mean radius would have Space-pause stomped
    // back to 1× every frame.
    //
    // Only meaningful while the craft is being Kepler-coasted — that is the
    // only regime where `step()` propagates translation, so the only one where
    // a sub-surface state can run away unbounded. A craft resting on terrain
    // under `BodyFixed` (settled ship) or `LocalRigidBody` (grounded EVA, ship
    // on its collider) is legitimately at or below mean radius wherever local
    // terrain dips below it, and `step()` never coasts it. Stomping its warp
    // here every frame was the surface time-warp bug: anywhere terrain sits
    // below the mean radius, warp snapped back to 1× and the player could not
    // fast-forward on the ground.
    let coasting = matches!(sim.simulation.authority(), AuthorityMode::OnRails { .. });
    let soi_body = sim.simulation.dominant_body();
    let (body_radius, body_name) = {
        let body_def = &sim.simulation.bodies()[soi_body];
        (body_def.radius_m, body_def.name.clone())
    };
    if coasting && body_radius > 0.0 && sim.simulation.warp.target_speed() > 1.0 {
        let body_state = sim.simulation.ephemeris().state(soi_body, Epoch(post_t));
        let altitude_above_mean = (post_pos - body_state.position).length() - body_radius;
        if altitude_above_mean < -1.0 && post_pos.is_finite() {
            sim.simulation.warp.reset_immediate();
            warn!(
                "ship sits {:.1}m below mean radius of {} (warp={:.0}x dropped to 1x); \
                 terrain-aware coast cannot detect collision from a sub-surface state",
                -altitude_above_mean, body_name, warp,
            );
        }
    }
}

/// Drive the cached trajectory prediction from the regime record (A3 port
/// #3, `docs/simulation/regimes.md`): `PredictionDisplay::Hide` clears the plan —
/// landed (`BodyFixed`), in ground contact under the backend (the velocity
/// carries contact reactions Kepler can't follow), or walking on foot
/// (analytically glued to the rotating surface; predicting it at high warp
/// caused expensive bogus recomputes and terrain-residency churn from
/// impossible encounters). The classification lives in the resolver
/// (`thalos_physics_canonical::regime::prediction_display`); before the
/// bubble/record exists the plan stays visible, matching the legacy
/// "no bubble → ballistic" default.
fn update_prediction(
    mut sim: ResMut<SimulationState>,
    active: Res<ActiveLocalBubble>,
    craft_q: Query<&crate::regime::CraftRegimeState, With<LocalCraftBody>>,
) {
    let _span = tracing::info_span!("update_prediction").entered();

    let show = active
        .bubble
        .as_ref()
        .and_then(|bubble| craft_q.get(bubble.craft_entity).ok())
        .map(|state| matches!(state.regime.prediction, PredictionDisplay::Show))
        .unwrap_or(true);
    if !show {
        sim.simulation.clear_prediction();
        return;
    }

    if !sim.simulation.prediction_needs_refresh() {
        return;
    }

    sim.simulation.recompute_prediction();
}

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

impl Default for WarpLimits {
    fn default() -> Self {
        Self {
            max_level: usize::MAX,
        }
    }
}

/// Apply the regime record's warp policy (A4, `docs/simulation/regimes.md`): publish
/// `CraftRegime.warp.max_level` as [`WarpLimits`] for the input handler and
/// HUD, and clamp the current level down to it.
///
/// The policy *computation* — the per-level altitude ladder, the
/// in-atmosphere 1× clamp, the surface-resting exemptions (`BodyFixed` /
/// quiet grounded ship), and the on-foot at-rest rule — lives in the
/// unit-tested resolver (`thalos_physics_canonical::regime`); this system
/// only enforces the published decision, and the record's
/// `warp.constraint` tells the HUD *why* a cap binds.
///
/// There is deliberately no trajectory lookahead (a suborbital arc is fully
/// warpable while you're high on it); phasing through terrain at high warp
/// is prevented downstream by the propagator's swept collision detection,
/// and the two emergency warp resets (coast collision, sub-surface state)
/// remain in `Simulation::step` / [`advance_simulation`]. Before the
/// bubble/record exists the cap stays `usize::MAX` ("no constraint") — the
/// documented pre-enforcement default; every scenario spawns warp-paused
/// behind the loading screen, so nothing can over-warp in that window.
pub fn enforce_warp_altitude_limits(
    mut sim: ResMut<SimulationState>,
    mut limits: ResMut<WarpLimits>,
    active: Res<ActiveLocalBubble>,
    craft_q: Query<&crate::regime::CraftRegimeState, With<LocalCraftBody>>,
) {
    let cap = active
        .bubble
        .as_ref()
        .and_then(|bubble| craft_q.get(bubble.craft_entity).ok())
        .map(|state| state.regime.warp.max_level)
        .unwrap_or(usize::MAX);
    limits.max_level = cap;
    if sim.simulation.warp.level_index() > cap {
        sim.simulation.warp.clamp_to_level(cap);
    }
}

/// Handle keyboard input to adjust the warp multiplier.
///
/// - `.`      -- increase to next warp level
/// - `,`      -- decrease to previous warp level (0x = paused)
/// - `\`      -- reset to 1x
/// - `Space`  -- toggle pause (0x) / resume previous level
/// - `G`      -- toggle warp-to-next-maneuver auto-warp (see
///   [`crate::warp_to_maneuver`])
///
/// Warp-level changes are gated by [`ControlLocks::warp`] — when set,
/// some programmatic system (today: the scheduled-burn autopilot
/// during its lead-down) is driving warp and human nudges would just
/// get clobbered. Pause (Space) is always free; that exemption lives
/// here in the handler rather than as a separate lock flag, since the
/// throttle gate in
/// [`crate::fuel::gate_throttle_on_fuel_availability`] forces the
/// engine off at any non-1× warp anyway, so pausing mid-burn cleanly
/// suspends it and unpausing resumes. Pressing any manual warp key
/// cancels an in-progress auto-warp.
pub fn handle_warp_controls(
    input: Res<GameInputIntent>,
    locks: Res<ControlLocks>,
    limits: Res<WarpLimits>,
    mut sim: ResMut<SimulationState>,
    mut warp_to: ResMut<WarpToManeuver>,
) {
    if input.warp_to_maneuver {
        if warp_to.active {
            warp_to.cancel();
        } else if find_next_maneuver(sim.simulation.sim_time(), &sim.simulation).is_some() {
            warp_to.active = true;
        }
        return;
    }

    let manual_warp_key = input.warp_increase || input.warp_decrease || input.warp_reset;
    if manual_warp_key && warp_to.active {
        warp_to.cancel();
    }

    if locks.warp {
        return;
    }

    if input.warp_increase {
        // Refuse escalation past the altitude cap. The enforcement
        // system already clamps an over-cap current level downward;
        // checking here avoids the visible single-frame flicker of the
        // level going up and immediately back down.
        if sim.simulation.warp.level_index() < limits.max_level {
            sim.simulation.warp.increase();
        }
    } else if input.warp_decrease {
        // Comma steps down the warp ladder and into pause at the bottom
        // (level 0 = 0×), so pause is "in line with" time warp rather than a
        // separate key. `.` / increase steps back up and unpauses.
        if sim.simulation.warp.level_index() > 0 {
            sim.simulation.warp.decrease();
        }
    } else if input.warp_reset {
        sim.simulation.warp.reset();
    }
}

/// Sync the UI-side [`ManeuverPlan`] into the physics `ManeuverSequence`.
///
/// The UI plan is authoritative for the node lifecycle: physics never removes
/// nodes on its own. When `plan.dirty` (a user edit, or the autopilot advancing
/// a node's burn phase), the physics sequence is rebuilt from scratch, taking
/// only the nodes that still drive the prediction — i.e. [`NodeBurnPhase::Planned`]
/// ones. A burning node (`Executing`) is excluded so the planned Δv isn't
/// double-counted on top of the live thrust, and a spent node (`Executed`) is
/// excluded because its burn is already in the past; both linger in the UI plan
/// for display until the user dismisses them.
///
/// [`NodeBurnPhase::Planned`]: crate::maneuver::NodeBurnPhase::Planned
/// [`NodeBurnPhase::Executing`]: crate::maneuver::NodeBurnPhase::Executing
/// [`NodeBurnPhase::Executed`]: crate::maneuver::NodeBurnPhase::Executed
fn sync_maneuver_plan(mut plan: ResMut<ManeuverPlan>, mut sim: ResMut<SimulationState>) {
    if !plan.dirty {
        return;
    }
    let _span = tracing::info_span!("sync_maneuver_plan").entered();
    plan.dirty = false;

    let seq = sim.simulation.maneuvers_mut();
    seq.nodes.clear();
    for node in plan.nodes.iter().filter(|n| n.drives_prediction()) {
        seq.nodes.push(ManeuverNode {
            id: Some(node.id.0),
            time: node.time,
            delta_v: node.delta_v,
            reference_body: node.reference_body,
        });
    }
    seq.nodes.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ---------------------------------------------------------------------------
// CraftStateMirror — Reflect-friendly snapshot of canonical ship state
// ---------------------------------------------------------------------------

/// Reflect-registered mirror of the canonical `CraftState`, refreshed
/// once per frame by [`refresh_craft_state_mirror`]. The canonical
/// state lives in `thalos_physics_canonical` (no Bevy dependency), so it cannot
/// derive `Reflect` directly; this resource is a read-only Reflect-registered
/// projection (for the HUD / a future debug overlay).
#[derive(Resource, Reflect, Default, Clone, Debug)]
#[reflect(Resource)]
pub struct CraftStateMirror {
    pub sim_time_s: f64,
    pub warp_speed: f64,
    pub position_m: [f64; 3],
    pub velocity_m_s: [f64; 3],
    /// World-frame angular velocity of the craft (rad/s). In the local
    /// bubble this is the live Avian body rate (written back to canonical by
    /// `local_physics::readback_local_craft`); a diagnostic for attitude /
    /// control-stability work — a steady non-decaying oscillation here is
    /// SAS chatter. See `docs/simulation/control.md`.
    pub angular_velocity_rad_s: [f64; 3],
    pub mass_kg: f64,
    pub dominant_body_id: u32,
    /// Discriminant name of `AuthorityMode` (variant fields elided).
    pub authority: String,
    /// Whole-craft structural failure from a terrain impact. See
    /// `docs/simulation/surface.md`.
    pub destroyed: bool,
    /// Surface-relative approach speed (m/s) of the destroying impact;
    /// `0.0` unless `destroyed`.
    pub last_impact_speed_m_s: f64,
    /// Aggregate thrust currently pushed into the propagator (N). Zero
    /// means no engine is producing thrust this frame (e.g. air-breathing
    /// jets with no intake air). Diagnostic mirror of
    /// `ship_params().thrust_n`.
    pub thrust_n: f64,
    /// Altitude above the dominant body's reference radius (m).
    pub altitude_m: f64,
    /// Whether the dominant body has a `terrestrial_atmosphere` block.
    pub has_atmosphere: bool,
    /// Kármán line of the dominant body's atmosphere (m); 0 if none.
    pub karman_line_m: f64,
    /// Whether the ship is currently inside the breathable column (the
    /// gate air-breathing jets check). Diagnostic mirror of
    /// `fuel::ship_in_atmosphere`.
    pub in_atmosphere: bool,
    /// Number of engines that passed every propulsion gate this frame
    /// (enabled, positive thrust/isp, atmosphere ok, intake satisfied,
    /// reactants present). Zero with `in_atmosphere == true` means the
    /// gate that killed thrust is *not* the atmosphere.
    pub propulsion_engine_count: u32,
}

fn refresh_craft_state_mirror(
    sim: Res<SimulationState>,
    propulsion: Res<crate::fuel::ActivePropulsion>,
    mut mirror: ResMut<CraftStateMirror>,
) {
    let state = sim.simulation.ship_state();
    mirror.sim_time_s = sim.simulation.sim_time();
    mirror.warp_speed = sim.simulation.warp.speed();
    mirror.position_m = [state.position.x, state.position.y, state.position.z];
    mirror.velocity_m_s = [state.velocity.x, state.velocity.y, state.velocity.z];
    let omega = sim.simulation.attitude().angular_velocity;
    mirror.angular_velocity_rad_s = [omega.x, omega.y, omega.z];
    mirror.mass_kg = sim.simulation.ship_mass_kg();
    mirror.dominant_body_id = sim.simulation.dominant_body() as u32;
    mirror.authority = match sim.simulation.authority() {
        AuthorityMode::OnRails { .. } => "OnRails",
        AuthorityMode::LocalRigidBody { .. } => "LocalRigidBody",
        AuthorityMode::BodyFixed { .. } => "BodyFixed",
    }
    .to_string();
    mirror.destroyed = sim.simulation.is_destroyed();
    mirror.last_impact_speed_m_s = sim.simulation.last_impact_speed_m_s();
    mirror.thrust_n = sim.simulation.ship_params().thrust_n;
    let body_id = sim.simulation.dominant_body();
    let body = sim.system.bodies.get(body_id);
    mirror.altitude_m = body
        .map(|body| {
            let body_pos = sim
                .ephemeris
                .state(body_id, Epoch(sim.simulation.sim_time()))
                .position;
            (state.position - body_pos).length() - body.radius_m
        })
        .unwrap_or(f64::NAN);
    mirror.has_atmosphere = body
        .map(|b| b.terrestrial_atmosphere.is_some())
        .unwrap_or(false);
    mirror.karman_line_m = body
        .and_then(|b| b.terrestrial_atmosphere.as_ref())
        .map(|a| a.karman_line_m as f64)
        .unwrap_or(0.0);
    mirror.in_atmosphere = mirror.has_atmosphere
        && mirror.karman_line_m > 0.0
        && mirror.altitude_m >= 0.0
        && mirror.altitude_m <= mirror.karman_line_m;
    mirror.propulsion_engine_count = propulsion.engines.len() as u32;
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct BridgePlugin;

impl Plugin for BridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CraftStateMirror>()
            .init_resource::<WarpLimits>()
            .register_type::<CraftStateMirror>()
            .add_systems(
                Update,
                (
                    enforce_warp_altitude_limits,
                    handle_warp_controls,
                    advance_simulation,
                    sync_maneuver_plan,
                    update_prediction,
                    refresh_craft_state_mirror,
                )
                    .chain()
                    .in_set(SimStage::Physics),
            );
    }
}
