//! [`FlightPlanView`]: the game-side cache derived from a physics [`FlightPlan`].
//!
//! Rebuilt whenever the prediction version, target body, or render frame
//! focus changes. Holds [`Ghost`]s that mark future encounter positions
//! and serve as rendering pins for legs anchored to those bodies.
//!
//! # Pin resolution
//!
//! [`FlightPlanView::pin_for_body`] is the central API. It answers:
//! "Given a sample anchored to body `B` at time `t`, what physics-space
//! position should the renderer pin its leg to?"
//!
//! Resolution rules, in order:
//! 1. If `B` is the camera focus body, return the focus body's current
//!    heliocentric position. (No ghost — the focus sits at the visual
//!    center, with the rest of the plan composed around it.)
//! 2. If a ghost exists for `B` whose `encounter_epoch ≤ t + tolerance`,
//!    pick the latest such ghost and return `pin(parent) + relative`,
//!    recursing through the parent chain.
//! 3. Otherwise return `B`'s current heliocentric position.
//!
//! The recursion grounds out at the focus body (rule 1) or at a body
//! with no ghost (rule 3), so chained encounters (e.g. Sun → Thalos →
//! Mira) compose: Mira's ghost adds its parent-frame offset to Thalos's
//! ghost, which adds its own offset to the focus, and the trajectory
//! stays continuous across SOI boundaries within the chain.
//!
//! The kink at the focus-body's own SOI boundary is intentional: the
//! pre-SOI leg renders in heliocentric, the in-SOI leg renders in the
//! focus's frame (rule 1), and the offset between them is the focus
//! body's motion between now and the SOI crossing. Patched-conics
//! visualisation has no smooth answer here; an SOI-entry marker (added
//! in a follow-up change) makes the discontinuity legible.

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics::types::{BodyId, BodyKind, BodyState};

use crate::coords::RenderFrame;
use crate::rendering::SimulationState;
use crate::target::TargetBody;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Tolerance (sim-seconds) for the `encounter_epoch ≤ t` ghost selection
/// rule. Encounter epochs and leg-start sample times are computed
/// independently in physics; a small allowance prevents samples right
/// at the boundary from missing their ghost due to float drift. Well
/// within the spacing between distinct encounters in any practical
/// prediction window.
const GHOST_EPOCH_TOLERANCE_S: f64 = 1.0;

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

/// A future-encounter marker for a celestial body, anchored relative to
/// its SOI parent so the trajectory leg pinned to it stays glued to the
/// parent's live position rather than drifting along the body's
/// heliocentric path.
pub struct Ghost {
    pub body_id: BodyId,
    /// SOI parent of `body_id` at `encounter_epoch`. The pin is
    /// composed as `pin(parent_id) + relative_position`.
    pub parent_id: BodyId,
    /// `body(t_enc) − parent(t_enc)`. Combined with the parent's pin
    /// at render time, this places the ghost at
    /// `parent_pin + relative` — i.e. the body's offset from its
    /// parent at the encounter epoch, projected into the parent's
    /// current frame.
    pub relative_position: DVec3,
    /// SOI entry time (or closest-approach epoch for target ghosts
    /// that don't enter an SOI). Drives lifecycle and ghost selection.
    pub encounter_epoch: f64,
    /// ECS entity (filled after spawn, preserved across rebuilds where
    /// the same `(body_id, ~encounter_epoch)` survives).
    pub entity: Option<Entity>,
    pub phase: GhostPhase,
}

/// Game-side view of the physics FlightPlan.
#[derive(Resource, Default)]
pub struct FlightPlanView {
    pub(super) version: u64,
    pub(super) last_target: Option<usize>,
    pub(super) last_focus: Option<BodyId>,
    /// Frame focus body at the most recent rebuild. `pin_for_body`
    /// uses this to ground the parent-chain recursion (rule 1 above).
    pub(super) focus_body: BodyId,
    /// All ghost markers in the active plan. May contain multiple
    /// entries for the same `body_id` when the trajectory encounters
    /// the same body at different times.
    pub(super) ghosts: Vec<Ghost>,
}

impl FlightPlanView {
    pub fn ghosts(&self) -> &[Ghost] {
        &self.ghosts
    }

    pub(super) fn ghosts_mut(&mut self) -> &mut Vec<Ghost> {
        &mut self.ghosts
    }

    /// Latest non-Retired ghost for `body_id` whose `encounter_epoch ≤
    /// sample_time + GHOST_EPOCH_TOLERANCE_S`. This is the ghost that
    /// pins a leg containing samples at `sample_time`.
    fn ghost_for_anchor(&self, body_id: BodyId, sample_time: f64) -> Option<&Ghost> {
        self.ghosts
            .iter()
            .filter(|g| g.body_id == body_id && g.phase != GhostPhase::Retired)
            .filter(|g| g.encounter_epoch <= sample_time + GHOST_EPOCH_TOLERANCE_S)
            .max_by(|a, b| {
                a.encounter_epoch
                    .partial_cmp(&b.encounter_epoch)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// World-space pin for a leg whose anchor is `body_id` at sample
    /// time `t`. See module docstring for resolution rules.
    pub fn pin_for_body(
        &self,
        body_id: BodyId,
        sample_time: f64,
        body_states: &[BodyState],
    ) -> DVec3 {
        // Rule 1: focus body grounds out at its current heliocentric
        // position. Suppresses parent-chain composition for the focus
        // and the kink at the focus-body SOI boundary lives here.
        if body_id == self.focus_body {
            return body_states
                .get(body_id)
                .map(|s| s.position)
                .unwrap_or(DVec3::ZERO);
        }

        // Rule 2: ghost match → recurse through parent.
        if let Some(ghost) = self.ghost_for_anchor(body_id, sample_time) {
            // The parent recursion uses the body's encounter epoch —
            // this picks the parent's ghost that was active at the
            // moment our trajectory entered this body's SOI, which is
            // what guarantees continuity through chained encounters.
            let parent_pin =
                self.pin_for_body(ghost.parent_id, ghost.encounter_epoch, body_states);
            let composed = parent_pin + ghost.relative_position;

            return match ghost.phase {
                GhostPhase::Active => composed,
                GhostPhase::Blending { progress } => {
                    let real = body_states
                        .get(ghost.body_id)
                        .map(|s| s.position)
                        .unwrap_or(composed);
                    let t = (progress.clamp(0.0, 1.0)) as f64;
                    composed * (1.0 - t) + real * t
                }
                // Retired ghosts are filtered out in `ghost_for_anchor`,
                // so this branch is unreachable in practice. Fall back
                // gracefully if someone bypasses the filter.
                GhostPhase::Retired => body_states
                    .get(ghost.body_id)
                    .map(|s| s.position)
                    .unwrap_or(composed),
            };
        }

        // Rule 3: no ghost → body's current heliocentric position.
        body_states
            .get(body_id)
            .map(|s| s.position)
            .unwrap_or(DVec3::ZERO)
    }
}

// ---------------------------------------------------------------------------
// Rebuild system
// ---------------------------------------------------------------------------

pub(super) fn rebuild_flight_plan_view(
    sim: Option<Res<SimulationState>>,
    target: Res<TargetBody>,
    frame: Res<RenderFrame>,
    mut view: ResMut<FlightPlanView>,
) {
    let Some(sim) = sim else { return };
    let Some(flight_plan) = sim.simulation.prediction() else {
        return;
    };

    let version = sim.simulation.prediction_version();
    let target_changed = view.last_target != target.target;
    let focus_changed = view.last_focus != Some(frame.focus_body);
    if view.version == version && !target_changed && !focus_changed {
        return;
    }
    view.version = version;
    view.last_target = target.target;
    view.last_focus = Some(frame.focus_body);
    view.focus_body = frame.focus_body;

    let ephemeris = sim.ephemeris.as_ref();
    let focus_body = frame.focus_body;

    let mut new_ghosts: Vec<Ghost> = Vec::new();

    // One ghost per encounter in the active plan, except the focus body —
    // which is kept at the visual center via rule 1 in pin_for_body.
    for enc in flight_plan.encounters() {
        let body_id = enc.body;
        if body_id == focus_body {
            continue;
        }
        let Some(body_def) = sim.system.bodies.get(body_id) else {
            continue;
        };
        if body_def.kind == BodyKind::Star {
            continue;
        }

        // Projection epoch = SOI entry time. This is the exact moment
        // approach and capture legs meet, and it is stable across
        // repredictions (zero-crossing of the SOI signed distance), so
        // the ghost doesn't jitter on every reprediction the way a
        // closest-approach or maneuver-on-leg epoch would.
        let projection_epoch = enc.entry_epoch;
        let body_state = ephemeris.query_body(body_id, projection_epoch);
        let parent_id = body_def.parent.unwrap_or(0);
        let parent_state = ephemeris.query_body(parent_id, projection_epoch);

        new_ghosts.push(Ghost {
            body_id,
            parent_id,
            relative_position: body_state.position - parent_state.position,
            encounter_epoch: projection_epoch,
            entity: None,
            phase: GhostPhase::Active,
        });
    }

    // Target body closest-approach ghost — visual marker for "this is
    // where my target will be at closest pass" when no SOI encounter
    // covers it.
    if let Some(target_id) = target.target
        && target_id != focus_body
        && !new_ghosts.iter().any(|g| g.body_id == target_id)
        && let Some(ca) = flight_plan.closest_approach_to(target_id, ephemeris)
        && let Some(body_def) = sim.system.bodies.get(target_id)
        && body_def.kind != BodyKind::Star
    {
        let parent_id = body_def.parent.unwrap_or(0);
        let parent_state = ephemeris.query_body(parent_id, ca.epoch);
        new_ghosts.push(Ghost {
            body_id: target_id,
            parent_id,
            relative_position: ca.body_state.position - parent_state.position,
            encounter_epoch: ca.epoch,
            entity: None,
            phase: GhostPhase::Active,
        });
    }

    reconcile_entities(&mut view.ghosts, new_ghosts);
}

/// Pair each new ghost with the closest existing ghost of the same body,
/// preserving entity references so spawn/lifecycle state survives
/// repredictions. Existing ghosts left without a partner are marked
/// Retired so the lifecycle system can run camera handoff before
/// despawn.
fn reconcile_entities(old: &mut Vec<Ghost>, mut new: Vec<Ghost>) {
    for new_ghost in &mut new {
        let best = old
            .iter()
            .enumerate()
            .filter(|(_, o)| o.body_id == new_ghost.body_id && o.entity.is_some())
            .min_by(|(_, a), (_, b)| {
                let da = (a.encounter_epoch - new_ghost.encounter_epoch).abs();
                let db = (b.encounter_epoch - new_ghost.encounter_epoch).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        if let Some(pos) = best {
            new_ghost.entity = old[pos].entity;
            if old[pos].phase != GhostPhase::Retired {
                new_ghost.phase = old[pos].phase;
            }
            old.remove(pos);
        }
    }

    // Orphaned old ghosts: mark Retired so update_ghost_lifecycle can
    // run camera handoff, and sync_ghost_bodies despawns the entity on
    // the next pass.
    for orphan in old.iter_mut() {
        orphan.phase = GhostPhase::Retired;
    }

    new.append(old);
    *old = new;
}
