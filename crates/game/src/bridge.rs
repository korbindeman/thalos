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

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::canonical::{AuthorityMode, Epoch};
use thalos_physics_canonical::maneuver::ManeuverNode;
use thalos_physics_canonical::types::ControlInput;
use thalos_physics_local::{
    ActiveLocalBubble, LocalBubbleConfig, LocalCraftBody,
    avian::{AngularVelocity, ContactGraph, LinearVelocity},
    craft_contacts_terrain,
};

use crate::GameTerrainRegistry;
use crate::SimStage;
use crate::autopilot::Autopilot;
use crate::controls::ControlLocks;
use crate::fuel::ThrottleState;
use crate::maneuver::ManeuverPlan;
use crate::navigation::{NavigationState, compute_attitude_control};
use crate::player_controller::EvaMode;
use crate::rendering::SimulationState;
use crate::sim_clock::SimClock;
use crate::target::TargetBody;
use crate::velocity_frame::VelocityFrameState;
use crate::warp_to_maneuver::{WarpToManeuver, find_next_maneuver};
use thalos_physics_canonical::terrain_provider::TerrainProvider;
use thalos_physics_canonical::types::VesselKind;

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
    let coasting = matches!(
        sim.simulation.authority(),
        AuthorityMode::OnRails { .. }
            | AuthorityMode::WarpIntegrated { .. }
            | AuthorityMode::Docked { .. }
    );
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

/// Whether the ship is currently on a ballistic trajectory — coasting under
/// gravity (or thrusting) without any external authority overriding its
/// motion. `BodyFixed` is landed and `LocalRigidBody` with active terrain
/// contact carries Avian's contact reactions in its velocity; in both cases
/// Keplerian propagation produces a curve the ship will never follow, so
/// we hide it.
fn ship_is_ballistic(
    sim: &SimulationState,
    active: &ActiveLocalBubble,
    contact_graph: &ContactGraph,
) -> bool {
    match sim.simulation.authority() {
        AuthorityMode::BodyFixed { .. } => false,
        AuthorityMode::LocalRigidBody { .. } => {
            let Some(bubble) = active.bubble.as_ref() else {
                return true;
            };
            let Some(terrain_entity) = bubble.terrain_entity else {
                return true;
            };
            !craft_contacts_terrain(contact_graph, bubble.craft_entity, terrain_entity)
        }
        AuthorityMode::OnRails { .. }
        | AuthorityMode::WarpIntegrated { .. }
        | AuthorityMode::Docked { .. } => true,
    }
}

fn update_prediction(
    mut sim: ResMut<SimulationState>,
    active: Res<ActiveLocalBubble>,
    contact_graph: Res<ContactGraph>,
    eva_mode: Res<EvaMode>,
) {
    let _span = tracing::info_span!("update_prediction").entered();

    // Grounded EVA is analytically glued to the rotating surface by the
    // body-fixed player controller. It has no ballistic flight plan; treating
    // its collider-less LocalRigidBody as ballistic feeds a surface state into
    // Kepler prediction at high warp, causing expensive bogus recomputes and
    // terrain-residency churn from impossible encounters.
    if sim.simulation.vessel_kind() == VesselKind::Eva && eva_mode.is_grounded() {
        sim.simulation.clear_prediction();
        return;
    }

    if !ship_is_ballistic(&sim, &active, &contact_graph) {
        sim.simulation.clear_prediction();
        return;
    }

    if !sim.simulation.prediction_needs_refresh() {
        return;
    }

    sim.simulation.recompute_prediction();
}

/// Sample player attitude input + active navigation mode and push the
/// resulting [`ControlInput`] into the simulation.
///
/// Player keys (W/S pitch, A/D yaw, Q/E roll) override any active
/// [`NavigationMode`] for the duration they're held; T toggles SAS for
/// free-flight rate damping. Mode-specific autopilot logic lives in
/// [`compute_attitude_control`] — this system just collects inputs.
///
/// Player torque is zeroed while [`ControlLocks::attitude`] is set so
/// whatever programmatic system holds the lock (today: the autopilot's
/// direct burn-pointing target) wins.
/// `compute_attitude_control` still runs — it's the path that drives
/// the autopilot's PD command from its direct target.
pub fn handle_attitude_controls(
    input: Res<GameInputIntent>,
    nav: Res<NavigationState>,
    locks: Res<ControlLocks>,
    target: Res<TargetBody>,
    plan: Res<ManeuverPlan>,
    autopilot: Res<Autopilot>,
    velocity_frame: Res<VelocityFrameState>,
    mut sim: ResMut<SimulationState>,
    mut sas_enabled: Local<bool>,
) {
    // A destroyed craft accepts no attitude input: kill the command, drop
    // SAS, and skip the autopilot path so the wreck tumbles freely and the
    // HUD reads inert. (`apply_local_forces` also zeroes torque on its side.)
    if sim.simulation.is_destroyed() {
        *sas_enabled = false;
        sim.simulation.set_control(ControlInput::default());
        return;
    }

    if input.toggle_sas {
        *sas_enabled = !*sas_enabled;
    }

    let autopilot_target = autopilot.attitude_target();
    let mut player_torque = DVec3::ZERO;
    if !locks.attitude && autopilot_target.is_none() {
        player_torque = DVec3::new(
            input.attitude.x as f64,
            input.attitude.y as f64,
            input.attitude.z as f64,
        );
    }

    let control = compute_attitude_control(
        player_torque,
        nav.mode,
        autopilot_target,
        velocity_frame.active,
        &target,
        &plan,
        &sim.simulation,
        *sas_enabled,
    );
    sim.simulation.set_control(control);
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

/// Compute the highest warp level the craft can safely engage, clamp the
/// current level if it exceeds that cap, and publish the cap as
/// [`WarpLimits`] for the input handler and HUD.
///
/// Gating is purely a function of the craft's *current* altitude above
/// the dominant body, KSP-style: each warp level carries a minimum
/// altitude (in body-radii) in `WarpController`'s ladder, and the cap is
/// the highest level whose floor the craft currently clears.
///
/// There is deliberately no trajectory lookahead. Whether a future
/// periapsis dips below the surface is not this gate's concern — a
/// suborbital arc is fully warpable while you're high on it, and the cap
/// steps back down on its own as the craft descends and re-evaluates
/// each frame. Phasing through terrain at high warp is prevented
/// downstream, not here: `coast_segment` adaptively subdivides and the
/// swept-min Hermite check in `detect_step_crossings` flags a sub-surface
/// dip even when both ends of a warp step sit above ground, at which
/// point `Simulation::step` halts the ship at the impact and resets warp
/// to 1×. The gate decides *where you may warp*; the propagator
/// guarantees *you can't warp through the ground*.
///
/// The altitude floor is skipped for the two surface-resting regimes —
/// `BodyFixed` (settled ship) and grounded EVA — which instead get a flat
/// `SURFACE_WARP_MAX_SPEED` ceiling, since they can't phase through terrain but
/// do overrun the terrain streamer at very high warp. Every other regime gets
/// the altitude gate, so flying low in the Avian bubble (`LocalRigidBody`)
/// still drops to the pause/1× zone near terrain.
pub fn enforce_warp_altitude_limits(
    mut sim: ResMut<SimulationState>,
    terrain: Res<GameTerrainRegistry>,
    mut limits: ResMut<WarpLimits>,
    eva_mode: Res<EvaMode>,
    input: Res<GameInputIntent>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    active: Res<ActiveLocalBubble>,
    contact_graph: Res<ContactGraph>,
    config: Res<LocalBubbleConfig>,
    throttle: Res<ThrottleState>,
    craft_q: Query<(&LinearVelocity, &AngularVelocity), With<LocalCraftBody>>,
) {
    // Both BodyFixed (settled ship) and grounded EVA are stationary on the
    // surface and cannot phase through terrain, so they're exempt from the
    // altitude floor below. Terrain streaming is separately frozen at very
    // high stationary surface warp by `ground_terrain`, so the gameplay cap
    // can be the top of the configured warp ladder instead of the old 100×
    // renderer workaround.
    const SURFACE_WARP_MAX_SPEED: f64 = f64::INFINITY;
    let eva_grounded = sim.simulation.vessel_kind() == VesselKind::Eva && eva_mode.is_grounded();
    let ship_grounded_stationary = sim.simulation.vessel_kind() == VesselKind::Ship
        && matches!(
            sim.simulation.authority(),
            AuthorityMode::LocalRigidBody { .. }
        )
        && active.bubble.as_ref().is_some_and(|bubble| {
            bubble.terrain_entity.is_some_and(|terrain_entity| {
                craft_contacts_terrain(&contact_graph, bubble.craft_entity, terrain_entity)
                    && craft_q.get(bubble.craft_entity).is_ok_and(
                        |(linear_velocity, angular_velocity)| {
                            linear_velocity.length() < config.max_stable_speed_m_s
                                && angular_velocity.length() < config.max_stable_angular_speed_rad_s
                                && throttle.effective <= 1.0e-3
                        },
                    )
            })
        });
    if eva_grounded
        || ship_grounded_stationary
        || matches!(sim.simulation.authority(), AuthorityMode::BodyFixed { .. })
    {
        // KSP rule: you can only engage on-rails warp on foot once you've come
        // to a complete stop ("landed and stationary"). While the player is
        // walking, jumping, or falling, hold them at 1× (live). Movement intent
        // is read directly from input so pressing a move key while warping
        // drops warp immediately, rather than waiting on the rest debounce.
        let at_rest = player.as_deref().map(|p| p.is_at_rest()).unwrap_or(false);
        let wants_to_move = input.player_move.length_squared() > 1.0e-4 || input.player_jump;
        let eva_can_warp = eva_grounded && at_rest && !wants_to_move;
        let ceiling = if eva_grounded && !eva_can_warp {
            1.0
        } else {
            SURFACE_WARP_MAX_SPEED
        };
        let cap = {
            let levels = sim.simulation.warp.levels();
            levels
                .iter()
                .rposition(|&speed| speed <= ceiling)
                .unwrap_or(0)
        };
        limits.max_level = cap;
        if sim.simulation.warp.level_index() > cap {
            sim.simulation.warp.clamp_to_level(cap);
        }
        return;
    }

    let dominant = sim.simulation.dominant_body();
    let bodies = sim.simulation.bodies();
    if bodies[dominant].radius_m <= 0.0 {
        limits.max_level = usize::MAX;
        return;
    }

    // Current altitude above the dominant body, as a fraction of its
    // radius. Tracking the ratio rather than absolute metres lets each
    // level's floor be a single body-radius fraction that's meaningful
    // across the whole system — a 0.05-radius floor is ~160 km on Thalos,
    // a few hundred metres on a small moon. The conservative
    // `body_radius + max_terrain_elevation` buffer treats every direction
    // as the tallest authored peak, so the gate never over-reports
    // altitude near terrain. The radius is known > 0 from the early
    // return above.
    let sim_time = sim.simulation.sim_time();
    let body_pos = sim
        .simulation
        .ephemeris()
        .state(dominant, Epoch(sim_time))
        .position;
    let r = bodies[dominant].radius_m;
    let buffer = r + terrain.0.max_elevation_m(dominant);
    let altitude_m = (sim.simulation.ship_state().position - body_pos).length() - buffer;
    let alt_radii = altitude_m.max(0.0) / r;

    // Walk the levels in order; the highest one whose min-altitude floor
    // the craft currently clears is the cap. `alt_radii` saturated at 0.0
    // means we're inside the conservative terrain envelope and only
    // levels with `min_altitude_radii_for(i) == 0.0` qualify.
    let levels = sim.simulation.warp.levels();
    let mut max_level = 0usize;
    for i in 0..levels.len() {
        if sim.simulation.warp.min_altitude_radii_for(i) <= alt_radii {
            max_level = i;
        }
    }
    limits.max_level = max_level;

    if sim.simulation.warp.level_index() > max_level {
        sim.simulation.warp.clamp_to_level(max_level);
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

/// Sync the UI-side [`ManeuverPlan`] with the physics `ManeuverSequence`.
///
/// Lifecycle:
/// 1. Remove UI nodes that physics reports as consumed this frame. This is
///    the only way UI nodes retire — never by comparing time to `sim_time`,
///    which would silently drop nodes whose execution was skipped (e.g. at
///    observation warp). A UI node still sitting with `time <= sim_time`
///    means physics didn't burn it — a bug signal worth surfacing, not
///    hiding.
/// 2. When `plan.dirty` (user edit or consumption), push the current UI
///    list into physics, tagging each entry with its `NodeId` so the next
///    consumption cycle can round-trip.
fn sync_maneuver_plan(mut plan: ResMut<ManeuverPlan>, mut sim: ResMut<SimulationState>) {
    let consumed = sim.simulation.drain_consumed_node_ids();
    if !consumed.is_empty() {
        info!(
            "[bridge] physics consumed maneuver node ids: {:?} (sim_time={:.2})",
            consumed,
            sim.simulation.sim_time()
        );
        let before = plan.nodes.len();
        plan.nodes.retain(|n| !consumed.contains(&n.id.0));
        if plan.nodes.len() != before {
            plan.dirty = true;
        }
    }

    if !plan.dirty {
        return;
    }
    let _span = tracing::info_span!("sync_maneuver_plan").entered();
    plan.dirty = false;

    let seq = sim.simulation.maneuvers_mut();
    seq.nodes.clear();
    for node in &plan.nodes {
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
// CraftStateMirror — Reflect-friendly snapshot of canonical ship state for BRP
// ---------------------------------------------------------------------------

/// Reflect-registered mirror of the canonical `CraftState`, refreshed
/// once per frame by [`refresh_craft_state_mirror`]. The canonical
/// state lives in `thalos_physics_canonical` (no Bevy dependency), so it cannot
/// derive `Reflect` directly; this resource is the read-only projection
/// an agent inspects over BRP.
#[derive(Resource, Reflect, Default, Clone, Debug)]
#[reflect(Resource)]
pub struct CraftStateMirror {
    pub sim_time_s: f64,
    pub warp_speed: f64,
    pub position_m: [f64; 3],
    pub velocity_m_s: [f64; 3],
    pub mass_kg: f64,
    pub dominant_body_id: u32,
    /// Discriminant name of `AuthorityMode` (variant fields elided).
    pub authority: String,
    /// Whole-craft structural failure from a terrain impact. See
    /// `docs/surface.md`.
    pub destroyed: bool,
    /// Surface-relative approach speed (m/s) of the destroying impact;
    /// `0.0` unless `destroyed`.
    pub last_impact_speed_m_s: f64,
}

fn refresh_craft_state_mirror(sim: Res<SimulationState>, mut mirror: ResMut<CraftStateMirror>) {
    let state = sim.simulation.ship_state();
    mirror.sim_time_s = sim.simulation.sim_time();
    mirror.warp_speed = sim.simulation.warp.speed();
    mirror.position_m = [state.position.x, state.position.y, state.position.z];
    mirror.velocity_m_s = [state.velocity.x, state.velocity.y, state.velocity.z];
    mirror.mass_kg = sim.simulation.ship_mass_kg();
    mirror.dominant_body_id = sim.simulation.dominant_body() as u32;
    mirror.authority = match sim.simulation.authority() {
        AuthorityMode::OnRails { .. } => "OnRails",
        AuthorityMode::WarpIntegrated { .. } => "WarpIntegrated",
        AuthorityMode::LocalRigidBody { .. } => "LocalRigidBody",
        AuthorityMode::BodyFixed { .. } => "BodyFixed",
        AuthorityMode::Docked { .. } => "Docked",
    }
    .to_string();
    mirror.destroyed = sim.simulation.is_destroyed();
    mirror.last_impact_speed_m_s = sim.simulation.last_impact_speed_m_s();
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
                    handle_attitude_controls,
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
