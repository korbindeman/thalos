//! [`FlightPlanView`]: the game-side cache derived from a physics [`FlightPlan`].
//!
//! Rebuilt whenever the prediction version or target body changes. Holds
//! [`GhostSpec`]s that drive ghost entity spawn/update in [`super::ghost`].

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::types::{BodyId, BodyKind, BodyState};

use crate::rendering::SimulationState;
use crate::target::TargetBody;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Lifecycle phase of a ghost body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GhostPhase {
    /// Fully visible.
    Active,
    /// Blending toward the real body. `progress` goes 0.0 → 1.0.
    Blending { progress: f32 },
    /// Handoff complete — will be despawned.
    Retired,
}

/// Specification for a ghost body, derived from encounters in the FlightPlan.
pub struct GhostSpec {
    /// Physics body ID.
    pub body_id: BodyId,
    /// SOI parent body. The ghost is anchored to this body's *current*
    /// world position so it stays near its parent regardless of how far
    /// the parent moves between now and the projection epoch.
    pub parent_id: BodyId,
    /// `body(t_enc) − parent(t_enc)` — body's offset from its parent at
    /// the projection epoch. Combined with the parent's live pin at render
    /// time this places the ghost at `parent_now + relative`.
    pub relative_position: DVec3,
    /// SOI entry time — used for lifecycle/handoff timing.
    pub encounter_epoch: f64,
    /// ECS entity (filled after spawn, preserved across rebuilds).
    pub entity: Option<Entity>,
    /// Current lifecycle phase.
    pub phase: GhostPhase,
}

/// Game-side view of the physics FlightPlan.
#[derive(Resource, Default)]
pub struct FlightPlanView {
    /// Prediction version this view was built from.
    pub(super) version: u64,
    /// Last target body we built for.
    pub(super) last_target: Option<usize>,
    /// Ghost body specifications.
    pub(super) ghost_specs: Vec<GhostSpec>,
}

impl FlightPlanView {
    /// Find the ghost spec for a given body_id (first match).
    pub fn ghost_for_body(&self, body_id: BodyId) -> Option<&GhostSpec> {
        self.ghost_specs.iter().find(|g| g.body_id == body_id)
    }

    /// World-space pin for a leg whose anchor is `body_id`.
    ///
    /// If the body has an active ghost, returns `parent_pin + relative` —
    /// the body's projected offset from its parent applied to the parent's
    /// *current* pin. This anchors the ghost (and any leg pinned to it)
    /// to the parent's live position rather than the body's heliocentric
    /// future position, so e.g. a Mira encounter stays near current
    /// Thalos instead of drifting along Thalos's heliocentric path.
    ///
    /// The lookup recurses through `parent_id`, so chained encounters
    /// (Thalos ghost → Mira ghost) compose correctly.
    ///
    /// During blending the pin lerps toward the body's current position
    /// so handoff to the live body is smooth. Falls back to the body's
    /// current heliocentric position when no ghost exists.
    pub fn pin_for_body(&self, body_id: BodyId, body_states: &[BodyState]) -> DVec3 {
        if let Some(ghost) = self.ghost_for_body(body_id) {
            let parent_pin = self.pin_for_body(ghost.parent_id, body_states);
            let ghost_pin = parent_pin + ghost.relative_position;
            match ghost.phase {
                GhostPhase::Active => return ghost_pin,
                GhostPhase::Blending { progress } => {
                    if let Some(real_state) = body_states.get(body_id) {
                        let t = progress.clamp(0.0, 1.0) as f64;
                        return ghost_pin * (1.0 - t) + real_state.position * t;
                    }
                    return ghost_pin;
                }
                GhostPhase::Retired => {}
            }
        }
        body_states
            .get(body_id)
            .map(|bs| bs.position)
            .unwrap_or(DVec3::ZERO)
    }
}

// ---------------------------------------------------------------------------
// Rebuild system
// ---------------------------------------------------------------------------

pub(super) fn rebuild_flight_plan_view(
    sim: Option<Res<SimulationState>>,
    target: Res<TargetBody>,
    mut view: ResMut<FlightPlanView>,
) {
    let Some(sim) = sim else { return };
    let Some(flight_plan) = sim.simulation.prediction() else {
        return;
    };

    let version = sim.simulation.prediction_version();
    let target_changed = view.last_target != target.target;
    if view.version == version && !target_changed {
        return;
    }
    view.version = version;
    view.last_target = target.target;

    let ephemeris = sim.ephemeris.as_ref();

    // Collect new ghost specs from aggregated encounters.
    let mut new_specs: Vec<GhostSpec> = Vec::new();

    for enc in flight_plan.encounters() {
        let body_id = enc.body;

        let Some(body_def) = sim.system.bodies.get(body_id) else {
            continue;
        };
        if body_def.kind == BodyKind::Star {
            continue;
        }

        // Projection epoch = SOI entry time.
        //
        // This is the exact moment the approach leg's last sample meets the
        // capture leg's first sample, so pinning the ghost here makes both
        // legs align on screen with no visible kink at the SOI boundary.
        // It is also stable across repredictions — SOI crossings are defined
        // by a zero-crossing of the SOI-radius signed distance, so numerical
        // refinement only shifts it by a sub-step amount.  Closest approach
        // and maneuver-on-leg epochs, by contrast, can drift visibly between
        // coarse and fine passes and cause the capture orbit to jump during
        // warp.
        let projection_epoch = enc.entry_epoch;

        let body_state = ephemeris.query_body(body_id, projection_epoch);
        let parent_id = body_def.parent.unwrap_or(0);
        let parent_state = ephemeris.query_body(parent_id, projection_epoch);

        new_specs.push(GhostSpec {
            body_id,
            parent_id,
            relative_position: body_state.position - parent_state.position,
            encounter_epoch: enc.entry_epoch,
            entity: None,
            phase: GhostPhase::Active,
        });
    }

    // Target body closest-approach ghost (if not already covered by an encounter).
    if let Some(target_id) = target.target
        && !new_specs.iter().any(|s| s.body_id == target_id)
        && let Some(ca) = flight_plan.closest_approach_to(target_id, ephemeris)
    {
        let Some(body_def) = sim.system.bodies.get(target_id) else {
            reconcile_entities(&mut view.ghost_specs, new_specs);
            return;
        };
        if body_def.kind != BodyKind::Star {
            let parent_id = body_def.parent.unwrap_or(0);
            let parent_state = ephemeris.query_body(parent_id, ca.epoch);
            new_specs.push(GhostSpec {
                body_id: target_id,
                parent_id,
                relative_position: ca.body_state.position - parent_state.position,
                encounter_epoch: ca.epoch,
                entity: None,
                phase: GhostPhase::Active,
            });
        }
    }

    reconcile_entities(&mut view.ghost_specs, new_specs);
}

/// Match new specs against existing ones by `body_id` (preferring the
/// old spec whose `encounter_epoch` is closest to the new one when there
/// are duplicates).  Preserves entity references so camera focus and
/// material state survive repredictions even when the encounter epoch
/// shifts significantly between passes.
fn reconcile_entities(old: &mut Vec<GhostSpec>, mut new: Vec<GhostSpec>) {
    for new_spec in &mut new {
        let best_match = old
            .iter()
            .enumerate()
            .filter(|(_, o)| o.body_id == new_spec.body_id && o.entity.is_some())
            .min_by(|(_, a), (_, b)| {
                let da = (a.encounter_epoch - new_spec.encounter_epoch).abs();
                let db = (b.encounter_epoch - new_spec.encounter_epoch).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        if let Some(pos) = best_match {
            new_spec.entity = old[pos].entity;
            if old[pos].phase != GhostPhase::Retired {
                new_spec.phase = old[pos].phase;
            }
            old.remove(pos);
        }
    }

    // Remaining old specs weren't matched — mark retired so lifecycle can
    // run focus handoff, then sync_ghost_bodies despawns.
    for orphan in old.iter_mut() {
        orphan.phase = GhostPhase::Retired;
    }

    new.append(old);
    *old = new;
}
