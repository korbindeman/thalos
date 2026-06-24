//! The `CraftRegime` resolver, the authority executor, and the drift
//! checker (`docs/regimes.md` Phases A2–A3).
//!
//! [`resolve_regime`] is the **sole writer** of the per-craft
//! [`CraftRegimeState`] component: it gathers [`RegimeInputs`] from the ECS
//! once per frame (top of `SimStage::Physics`, before the bridge's warp
//! handling) and runs the pure classifier in
//! `thalos_physics_canonical::regime`.
//!
//! Consumer migration status (A3): **authority is live** —
//! [`apply_regime_authority`] is the single writer of canonical
//! `AuthorityMode` transitions — and **the Avian role is live** —
//! `local_physics::compute_avian_authority` projects the record onto
//! `AvianAuthority` via [`legacy_avian_role`], so the resolver is the one
//! classifier — and **prediction gating, the terrain-collider gate, and the
//! warp policy are live** (`bridge::update_prediction`, the patch
//! attach/detach/maintain gates, and `bridge::enforce_warp_altitude_limits`
//! all read the record). Every per-frame regime decision now flows from the
//! resolver; [`check_regime_drift`] remains as end-of-frame sanity checks
//! plus the BRP-readable record snapshot in [`RegimeDriftDiagnostics`].
//!
//! Input snapshot semantics (`docs/regimes.md` §3.2): physics-derived
//! signals (contacts, weight-on-wheels, collider presence, speeds) are
//! previous-frame; command inputs (warp, throttle) are current-frame. The
//! legacy systems compute their predicates mid-chain from this frame's
//! values, so single-frame mismatches are expected at regime *edges* (patch
//! attach/detach, throttle taps, settle crossings) — those are counted as
//! blips. A mismatch that persists two or more consecutive checked frames is
//! steady-state drift and a real classifier bug; the A2 acceptance criterion
//! is zero steady mismatches across the scenario matrix.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::canonical::{AuthorityMode, BodyFixedPose, EntityRef, Epoch};
use thalos_physics_canonical::regime::{
    AuthorityKind, CraftRegime, PredictionDisplay, RegimeInputs, RegimeMemory, TranslationOwner,
    WalkingInputs, WarpLevel, expected_authority, resolve,
};
use thalos_physics_canonical::surface_local::{SurfaceLocalState, surface_local_to_body_fixed};
use thalos_physics_canonical::terrain_provider::TerrainProvider;
use thalos_physics_canonical::types::VesselKind;
use thalos_physics_local::{
    ActiveLocalBubble, LocalBubbleConfig, LocalCraftBody,
    avian::{AngularVelocity, ContactGraph, LinearVelocity, Position, Rotation},
    craft_contacts_terrain,
};

use crate::GameTerrainRegistry;
use crate::SimStage;
use crate::bridge::WarpLimits;
use crate::fuel::ThrottleState;
use crate::local_physics::{AvianAuthority, AvianRole, WeightOnWheels};
use crate::player_controller::{EvaMode, PlayerControllerState};
use crate::rendering::SimulationState;
use crate::sim_clock::SimClock;

/// Per-craft regime record + resolver memory. **Sole writer:**
/// [`resolve_regime`]. Shadow-only in Phase A2 — downstream systems start
/// reading it in Phase A3.
#[derive(Component, Debug, Clone)]
pub struct CraftRegimeState {
    pub regime: CraftRegime,
    pub memory: RegimeMemory,
    /// Canonical authority the record projects for end of frame
    /// (`regime::expected_authority`), captured at resolve time so the
    /// drift checker compares against exactly what the resolver decided.
    pub expected_authority: AuthorityKind,
}

/// BRP-readable drift observability. `*_blips` count one-frame mismatches
/// (expected at regime edges, see module docs); `*_steady` count mismatches
/// that persisted ≥ 2 consecutive checked frames (classifier bugs). The
/// string fields snapshot the latest record for inspection.
#[derive(Resource, Reflect, Default, Clone, Debug)]
#[reflect(Resource)]
pub struct RegimeDriftDiagnostics {
    pub frames_checked: u64,
    /// Frames skipped because the warp speed lerp was in flight —
    /// classification during the 0.35 s cosmetic transition is
    /// timing-dependent and compares as noise.
    pub skipped_warp_transition_frames: u64,
    pub authority_blips: u64,
    pub authority_steady: u64,
    pub authority_consecutive: u32,
    pub prediction_blips: u64,
    pub prediction_steady: u64,
    pub prediction_consecutive: u32,
    pub last_mismatch: String,
    // Latest record snapshot (strings for Reflect/BRP friendliness).
    pub medium: String,
    pub ground: String,
    pub translation_owner: String,
    pub rotation_owner: String,
    pub backend_clock_runs: bool,
    pub warp_max_level: u64,
    pub warp_constraint: String,
    pub prediction: String,
    pub expected_authority: String,
}

pub struct RegimePlugin;

impl Plugin for RegimePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RegimeDriftDiagnostics>()
            .register_type::<RegimeDriftDiagnostics>()
            .add_systems(
                Update,
                resolve_regime
                    .in_set(SimStage::Physics)
                    .before(crate::bridge::enforce_warp_altitude_limits),
            )
            // End-of-frame comparison: by `Sync` the legacy chain (bridge +
            // local_physics, including the settle collapse) has produced its
            // final values for the frame.
            .add_systems(Update, check_regime_drift.in_set(SimStage::Sync));
    }
}

/// Sole writer of [`CraftRegimeState`].
#[allow(clippy::too_many_arguments)]
fn resolve_regime(
    mut commands: Commands,
    clock: Res<SimClock>,
    sim: Res<SimulationState>,
    throttle: Res<ThrottleState>,
    active: Res<ActiveLocalBubble>,
    config: Res<LocalBubbleConfig>,
    contact_graph: Res<ContactGraph>,
    weight_on_wheels: Res<WeightOnWheels>,
    terrain: Res<GameTerrainRegistry>,
    eva_mode: Res<EvaMode>,
    intent: Res<GameInputIntent>,
    player: Option<Res<PlayerControllerState>>,
    mut craft_q: Query<
        (
            Entity,
            &LinearVelocity,
            &AngularVelocity,
            Option<&mut CraftRegimeState>,
        ),
        With<LocalCraftBody>,
    >,
) {
    // No live local body, no regime — mirrors the legacy systems' bubble
    // early-returns (pre-bubble loading frames are unclassified).
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Ok((entity, linear_velocity, angular_velocity, state)) =
        craft_q.get_mut(bubble.craft_entity)
    else {
        return;
    };

    let vessel = sim.simulation.vessel_kind();
    let walking = (vessel == VesselKind::Eva && eva_mode.is_grounded()).then(|| WalkingInputs {
        grounded: player.as_deref().map(|p| p.is_grounded()).unwrap_or(false),
        at_rest: player.as_deref().map(|p| p.is_at_rest()).unwrap_or(false),
        wants_to_move: intent.player_move.length_squared() > 1.0e-4 || intent.player_jump,
    });

    let body_id = sim.simulation.dominant_body();
    let body = &sim.simulation.bodies()[body_id];
    let sim_time = sim.simulation.sim_time();
    let body_position = sim
        .simulation
        .ephemeris()
        .state(body_id, Epoch(sim_time))
        .position;
    let radial_distance_m = (sim.simulation.ship_state().position - body_position).length();
    let karman_line_m = body
        .terrestrial_atmosphere
        .as_ref()
        .map(|a| a.karman_line_m as f64)
        .unwrap_or(0.0);
    let terrain_buffer_m = body.radius_m + terrain.0.max_elevation_m(body_id);

    let warp = &sim.simulation.warp;
    let warp_ladder: Vec<WarpLevel> = warp
        .levels()
        .iter()
        .enumerate()
        .map(|(index, &speed)| WarpLevel {
            speed,
            min_altitude_radii: warp.min_altitude_radii_for(index),
        })
        .collect();

    let hull_contacts_terrain_patch = bubble
        .terrain_entity
        .map(|terrain_entity| {
            craft_contacts_terrain(&contact_graph, bubble.craft_entity, terrain_entity)
        })
        .unwrap_or(false);

    let inputs = RegimeInputs {
        sim_delta_s: clock.delta_secs_f64(),
        warp_speed: warp.speed(),
        warp_target_speed: warp.target_speed(),
        warp_ladder: &warp_ladder,
        throttle_effective: throttle.effective,
        throttle_commanded: throttle.commanded,
        authority: sim.simulation.authority().into(),
        walking,
        // Capability proxy: the EVA capsule's collider is removed at spawn.
        // Becomes a real per-craft capability once parts declare it.
        craft_has_collider: vessel != VesselKind::Eva,
        body_radius_m: body.radius_m,
        altitude_above_mean_m: radial_distance_m - body.radius_m,
        altitude_above_terrain_buffer_m: radial_distance_m - terrain_buffer_m,
        karman_line_m,
        terrain_collider_attached: bubble.terrain_entity.is_some(),
        hull_contacts_terrain_patch,
        weight_on_wheels: weight_on_wheels.grounded,
        linear_speed_m_s: linear_velocity.length(),
        angular_speed_rad_s: angular_velocity.length(),
        max_stable_speed_m_s: config.max_stable_speed_m_s,
        max_stable_angular_speed_rad_s: config.max_stable_angular_speed_rad_s,
        settle_dwell_s: config.stable_contact_time_s,
    };

    let memory = state.as_ref().map(|s| s.memory).unwrap_or_default();
    let (regime, next_memory) = resolve(&inputs, &memory);
    let projected_authority = expected_authority(&inputs, &regime);
    let next_state = CraftRegimeState {
        regime,
        memory: next_memory,
        expected_authority: projected_authority,
    };
    match state {
        Some(mut existing) => *existing = next_state,
        None => {
            commands.entity(entity).insert(next_state);
        }
    }
}

/// Project the record's owner/clock fields onto the legacy three-way
/// [`AvianRole`] (A3 port #2): clock off → `Paused`, Backend translation →
/// `Full`, Backend rotation under Canonical translation → `AttitudeOnly`.
/// Walking maps to `Paused` — the kinematic controller owns the capsule and
/// every backend-side system has its own EVA short-circuit; the legacy
/// classifier's incidental `Full`-in-atmosphere for grounded EVA gated
/// nothing (all `Full`-gated systems are vessel- or EVA-guarded).
pub(crate) fn legacy_avian_role(regime: &CraftRegime) -> AvianRole {
    if !regime.backend_clock_runs {
        AvianRole::Paused
    } else if regime.translation_owner == TranslationOwner::Backend {
        AvianRole::Full
    } else {
        AvianRole::AttitudeOnly
    }
}

/// Phase A3 port #1 — the single canonical-authority executor.
///
/// Applies [`CraftRegimeState::expected_authority`] to the canonical
/// `AuthorityMode`, replacing the three legacy authority writers:
/// `manage_authority` (the generic take/release, the grounded-EVA pin, and
/// the landed warp-request collapse), `release_landed_ship_on_throttle`
/// (the landed throttle release + its warp snap to 1×), and
/// `collapse_or_constrain_warp` (the timed settle collapse). The *decision*
/// lives in the unit-tested pure core
/// (`thalos_physics_canonical::regime::expected_authority`); this system
/// only realizes it — payload construction (bubble id / `BodyFixed` pose
/// capture) and transition side effects (the release's warp reset, the
/// handoff diagnostics).
///
/// Scenario teleports and respawn seeds still write authority directly
/// (event-driven, not per-frame); the resolver reads the seeded state next
/// frame and the executor continues from there — exactly as
/// `manage_authority` coexisted with them.
///
/// Pre-port behavior parity was established by the A2 shadow phase: ~51k
/// drift-checked frames across five scenarios with zero steady mismatches
/// between `expected_authority` and the legacy writers' end-of-frame
/// authority (see `docs/regimes.md`).
pub(crate) fn apply_regime_authority(
    active: Res<ActiveLocalBubble>,
    mut sim: ResMut<SimulationState>,
    mut diagnostics: ResMut<crate::local_physics::AvianHandoffDiagnostics>,
    craft_q: Query<(&CraftRegimeState, &Position, &Rotation), With<LocalCraftBody>>,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Ok((state, position, rotation)) = craft_q.get(bubble.craft_entity) else {
        return;
    };
    let current = AuthorityKind::from(sim.simulation.authority());
    let target = state.expected_authority;
    if current == target {
        return;
    }
    match target {
        AuthorityKind::OnRails => {
            let landed_release = current == AuthorityKind::BodyFixed;
            sim.simulation
                .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
            if landed_release {
                // Landed throttle release: hand live dynamics back at exactly
                // 1× so the throttle gate sees a live regime next frame.
                sim.simulation.warp.reset_immediate();
                info!("released landed ship on commanded throttle");
            } else {
                diagnostics.last_handoff_kind = "ReleasedTranslation".to_string();
                diagnostics.last_handoff_sim_time_s = sim.simulation.sim_time();
            }
        }
        AuthorityKind::LocalRigidBody => {
            sim.simulation
                .transition_authority(AuthorityMode::LocalRigidBody {
                    bubble: bubble.id,
                    root_entity: EntityRef(bubble.craft_entity.to_bits()),
                });
            // The take-translation diagnostics (kind/time/residual) are
            // recorded by `readback_local_craft` once Avian's converted
            // state is available.
        }
        AuthorityKind::BodyFixed => {
            // Settle / landed-warp collapse: freeze the craft's current pose
            // into the rotating body frame.
            //
            // Capture the pose from Avian's **surface-local** state, *not* from
            // canonical. This collapse always comes from `LocalRigidBody`, where
            // canonical's translation is only re-synced by `readback_local_craft`
            // at the end of the frame — so here (executor runs before readback)
            // it is one frame stale relative to `sim_time`. Pairing that stale
            // inertial translation with the current `body_state` mismatches their
            // epochs by one frame of the body's orbital motion (~hundreds of
            // metres at orbital speed), which froze the craft into a displaced
            // pose hovering off the surface after every re-settle. The
            // SLF→body-fixed conversion takes no `body_state` at all
            // (`surface_local_to_body_fixed` is a constant rotation + translation
            // in the co-rotating frame), so it is epoch-coherent by construction
            // — the mismatch cannot arise. (Ship-only path; EVA never settles to
            // `BodyFixed`. Velocity/spin are zero in the landed pose.)
            let slf = SurfaceLocalState {
                position_m: position.0,
                velocity_m_s: DVec3::ZERO,
                orientation_frame: rotation.0.normalize(),
                angular_velocity_body: DVec3::ZERO,
            };
            let body_fixed = surface_local_to_body_fixed(&bubble.frame, slf);
            let pose = BodyFixedPose {
                position_body_m: body_fixed.translation_body.position,
                orientation_body: body_fixed.orientation_body,
            };
            sim.simulation
                .transition_authority(AuthorityMode::BodyFixed {
                    body: bubble.body_id,
                    pose,
                });
            info!("collapsed stable landed craft to BodyFixed authority");
        }
    }
}

/// Outcome of one check's consecutive-mismatch bookkeeping.
enum Drift {
    Match,
    /// First mismatch frame — may resolve as a blip next frame.
    Pending,
    /// Second consecutive mismatch frame: steady-state drift onset.
    SteadyOnset,
    /// Persisting steady drift, periodic re-warn.
    SteadyOngoing,
    Silent,
}

fn track(consecutive: &mut u32, blips: &mut u64, steady: &mut u64, matched: bool) -> Drift {
    if matched {
        let previous = *consecutive;
        *consecutive = 0;
        if previous == 1 {
            // Resolved after exactly one frame: an edge blip, not drift.
            *blips += 1;
        }
        return Drift::Match;
    }
    *consecutive = consecutive.saturating_add(1);
    match *consecutive {
        1 => Drift::Pending,
        2 => {
            *steady += 1;
            Drift::SteadyOnset
        }
        n if n % 300 == 0 => Drift::SteadyOngoing,
        _ => Drift::Silent,
    }
}

/// Compare the shadow record against the legacy machinery's end-of-frame
/// values. Read-only over simulation state.
#[allow(clippy::too_many_arguments)]
fn check_regime_drift(
    sim: Res<SimulationState>,
    limits: Res<WarpLimits>,
    avian: Res<AvianAuthority>,
    active: Res<ActiveLocalBubble>,
    craft_q: Query<&CraftRegimeState, With<LocalCraftBody>>,
    mut diag: ResMut<RegimeDriftDiagnostics>,
) {
    let Some(bubble) = active.bubble.as_ref() else {
        return;
    };
    let Ok(state) = craft_q.get(bubble.craft_entity) else {
        return;
    };
    // Reborrow through `ResMut` once so the disjoint field `&mut`s below
    // don't each count as a whole-resource borrow.
    let diag = &mut *diag;

    // Snapshot the record for BRP inspection every frame, compared or not.
    diag.medium = format!("{:?}", state.regime.medium);
    diag.ground = format!("{:?}", state.regime.ground);
    diag.translation_owner = format!("{:?}", state.regime.translation_owner);
    diag.rotation_owner = format!("{:?}", state.regime.rotation_owner);
    diag.backend_clock_runs = state.regime.backend_clock_runs;
    diag.warp_max_level = state.regime.warp.max_level as u64;
    diag.warp_constraint = format!("{:?}", state.regime.warp.constraint);
    diag.prediction = format!("{:?}", state.regime.prediction);
    diag.expected_authority = format!("{:?}", state.expected_authority);

    // While the warp speed lerp is in flight, the resolver (frame-top speed)
    // and the legacy role classifier (post-step speed) see different
    // mid-transition values for ~20 frames — pure timing noise around a
    // cosmetic smoothing. Skip comparisons entirely until the lerp settles.
    let warp = &sim.simulation.warp;
    if (warp.speed() - warp.target_speed()).abs() > f64::EPSILON {
        diag.skipped_warp_transition_frames += 1;
        return;
    }
    diag.frames_checked += 1;

    // --- AvianRole ----------------------------------------------------
    // No longer compared: since A3 port #2 `compute_avian_authority` *is*
    // [`legacy_avian_role`] applied to this same record, so the comparison
    // is tautological. Sanity-assert the projection stayed in sync (catches
    // a reordered schedule, never expected to fire).
    debug_assert_eq!(legacy_avian_role(&state.regime), avian.role);
    let _ = &avian;

    // --- Canonical authority -------------------------------------------
    // Since the A3 port this is an executor sanity check, not a parity
    // check: `apply_regime_authority` writes the expected value, so a
    // mismatch here means an external writer (scenario teleport, respawn
    // seed) changed authority after the executor ran this frame — expected
    // as a blip on those events, never steady.
    let actual_authority: AuthorityKind = sim.simulation.authority().into();
    let matched = actual_authority == state.expected_authority;
    if !matched {
        diag.last_mismatch = format!(
            "authority: record expected {:?} vs canonical {:?}",
            state.expected_authority, actual_authority
        );
    }
    report(
        track(
            &mut diag.authority_consecutive,
            &mut diag.authority_blips,
            &mut diag.authority_steady,
            matched,
        ),
        "authority",
        &diag.last_mismatch,
    );

    // --- Warp cap --------------------------------------------------------
    // No longer compared: since A4 `enforce_warp_altitude_limits` *applies*
    // the record's warp policy, so `WarpLimits` is the record by
    // construction. Sanity-assert the application stayed in sync.
    debug_assert_eq!(limits.max_level, state.regime.warp.max_level);

    // --- Prediction gating -------------------------------------------------
    let legacy_shows = sim.simulation.prediction().is_some();
    let record_shows = matches!(state.regime.prediction, PredictionDisplay::Show);
    let matched = legacy_shows == record_shows;
    if !matched {
        diag.last_mismatch = format!(
            "prediction: record {:?} vs legacy cached-plan present {}",
            state.regime.prediction, legacy_shows
        );
    }
    report(
        track(
            &mut diag.prediction_consecutive,
            &mut diag.prediction_blips,
            &mut diag.prediction_steady,
            matched,
        ),
        "prediction",
        &diag.last_mismatch,
    );
}

fn report(drift: Drift, check: &str, detail: &str) {
    match drift {
        Drift::Match | Drift::Pending | Drift::Silent => {}
        Drift::SteadyOnset => {
            warn!("regime drift (steady, {check}): {detail}");
        }
        Drift::SteadyOngoing => {
            warn!("regime drift persists ({check}): {detail}");
        }
    }
}
