use bevy::math::{DVec3, Vec2, Vec3};
use bevy::prelude::*;
use thalos_physics_canonical::canonical::{CraftState, Epoch};
use thalos_physics_canonical::trajectory::{
    Encounter, FlightPlan, Trajectory, TrajectoryBranchStack,
};
use thalos_physics_canonical::types::{BodyDefinition, BodyId, BodyState, BodyStates};

use crate::coords::{RenderOrigin, WorldScale};
use crate::rendering::{SimulationState, SolarSystemState};
use crate::target::TargetBody;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct MapContext {
    pub origin: DVec3,
    pub scale: f64,
    pub focus_body: BodyId,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct DragInput {
    pub screen_delta: Vec2,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MapEdit {
    None,
}

pub trait MapProjection: Send + Sync {
    fn project_body(&self, body: &BodyState, ctx: &MapContext) -> Vec3;
    fn project_point(&self, point_inertial_m: DVec3, epoch: Epoch, ctx: &MapContext) -> Vec3;
    #[allow(dead_code)]
    fn unproject_drag(&self, drag: DragInput, ctx: &MapContext) -> MapEdit;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LinearMapProjection;

impl MapProjection for LinearMapProjection {
    fn project_body(&self, body: &BodyState, ctx: &MapContext) -> Vec3 {
        self.project_point(body.position, body.epoch, ctx)
    }

    fn project_point(&self, point_inertial_m: DVec3, _epoch: Epoch, ctx: &MapContext) -> Vec3 {
        ((point_inertial_m - ctx.origin) * ctx.scale).as_vec3()
    }

    fn unproject_drag(&self, _drag: DragInput, _ctx: &MapContext) -> MapEdit {
        MapEdit::None
    }
}

#[derive(Resource, Default, Clone)]
pub struct MapSnapshot {
    pub epoch: Epoch,
    pub body_states: BodyStates,
    pub body_defs: Vec<BodyDefinition>,
    pub crafts: Vec<CraftState>,
    pub flight_plan: Option<FlightPlan>,
    pub branch_stack: Option<TrajectoryBranchStack>,
    pub prediction_version: u64,
    pub target_body: Option<BodyId>,
    pub projected_body_states: Vec<ProjectedBodyState>,
    pub warp_speed: f64,
}

impl MapSnapshot {
    pub fn context(
        &self,
        origin: &RenderOrigin,
        scale: &WorldScale,
        focus_body: BodyId,
    ) -> MapContext {
        MapContext {
            origin: origin.position,
            scale: scale.0,
            focus_body,
        }
    }

    pub fn body_state_at(&self, body: BodyId, epoch: f64) -> Option<BodyState> {
        self.projected_body_states
            .iter()
            .find(|state| state.body == body && (state.epoch.0 - epoch).abs() <= 1.0)
            .map(|state| state.state)
            .or_else(|| {
                self.body_states
                    .get(body)
                    .copied()
                    .filter(|state| (state.epoch.0 - epoch).abs() <= 1.0)
            })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProjectedBodyState {
    pub body: BodyId,
    pub epoch: Epoch,
    pub state: BodyState,
}

pub fn update_map_snapshot(
    sim: Res<SimulationState>,
    body_cache: Res<SolarSystemState>,
    target: Res<TargetBody>,
    mut snapshot: ResMut<MapSnapshot>,
) {
    let Some(body_states) = body_cache.states.as_ref() else {
        return;
    };

    snapshot.epoch = Epoch(sim.simulation.sim_time());
    snapshot.body_states.clone_from(body_states);
    snapshot.body_defs.clone_from(&sim.system.bodies);
    snapshot.crafts.clear();
    snapshot.crafts.push(sim.simulation.craft_state().clone());
    snapshot.flight_plan = sim.simulation.prediction().cloned();
    snapshot.branch_stack = sim.simulation.trajectory_branches().cloned();
    snapshot.prediction_version = sim.simulation.prediction_version();
    snapshot.target_body = target.target;
    snapshot.warp_speed = sim.simulation.warp.speed();
    snapshot.projected_body_states.clear();

    if let Some(plan) = sim.simulation.prediction() {
        for encounter in plan.encounters() {
            push_encounter_window_projected_states(
                &mut snapshot.projected_body_states,
                &sim,
                plan,
                encounter,
            );
        }
        for approach in plan.approaches() {
            push_projected_body_state(
                &mut snapshot.projected_body_states,
                &sim,
                approach.body,
                approach.epoch,
            );
            if let Some(parent) = sim
                .system
                .bodies
                .get(approach.body)
                .and_then(|body| body.parent)
            {
                push_projected_body_state(
                    &mut snapshot.projected_body_states,
                    &sim,
                    parent,
                    approach.epoch,
                );
            }
        }
    }
}

fn push_encounter_window_projected_states(
    out: &mut Vec<ProjectedBodyState>,
    sim: &SimulationState,
    plan: &FlightPlan,
    encounter: &Encounter,
) {
    let Some(body_def) = sim.system.bodies.get(encounter.body) else {
        return;
    };
    let parent = body_def.parent;
    let (_, plan_end) = plan.epoch_range();
    let end = encounter.exit_epoch.unwrap_or(plan_end);
    let (start, end) = if encounter.entry_epoch <= end {
        (encounter.entry_epoch, end)
    } else {
        (end, encounter.entry_epoch)
    };

    let mut epochs = vec![encounter.entry_epoch, encounter.closest_epoch];
    if let Some(exit) = encounter.exit_epoch {
        epochs.push(exit);
    }
    for segment in plan.segments() {
        epochs.extend(
            segment
                .samples
                .iter()
                .map(|sample| sample.time)
                .filter(|time| *time >= start - 1e-6 && *time <= end + 1e-6),
        );
    }
    epochs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    epochs.dedup_by(|a, b| (*a - *b).abs() <= 1.0);

    for epoch in epochs {
        push_projected_body_state(out, sim, encounter.body, epoch);
        if let Some(parent) = parent {
            push_projected_body_state(out, sim, parent, epoch);
        }
    }
}

fn push_projected_body_state(
    out: &mut Vec<ProjectedBodyState>,
    sim: &SimulationState,
    body: BodyId,
    epoch: f64,
) {
    if out
        .iter()
        .any(|state| state.body == body && (state.epoch.0 - epoch).abs() <= 1.0)
    {
        return;
    }
    out.push(ProjectedBodyState {
        body,
        epoch: Epoch(epoch),
        state: sim.ephemeris.state(body, Epoch(epoch)),
    });
}

pub struct MapViewPlugin;

impl Plugin for MapViewPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MapSnapshot::default()).add_systems(
            Update,
            update_map_snapshot
                .after(crate::solar_system_state::sync_solar_system_state)
                .in_set(crate::SimStage::Sync),
        );
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::{DMat3, DQuat, DVec3};
    use thalos_physics_canonical::canonical::{
        AuthorityMode, MassState, ResourceState, TranslationalState,
    };
    use thalos_physics_canonical::types::AttitudeState;

    use super::*;

    fn craft() -> CraftState {
        CraftState {
            id: 1,
            epoch: Epoch::ZERO,
            translation: TranslationalState {
                position: DVec3::X,
                velocity: DVec3::Y,
            },
            attitude: AttitudeState {
                orientation: DQuat::IDENTITY,
                angular_velocity: DVec3::ZERO,
            },
            mass: MassState {
                wet_mass_kg: 10.0,
                dry_mass_kg: 5.0,
                inertia_body_kg_m2: DMat3::IDENTITY,
                center_of_mass_body_m: DVec3::ZERO,
            },
            resources: ResourceState,
            authority: AuthorityMode::OnRails { trajectory: 0 },
        }
    }

    #[test]
    fn mutating_map_snapshot_cannot_mutate_canonical_craft_state() {
        let canonical = craft();
        let mut snapshot = MapSnapshot {
            crafts: vec![canonical.clone()],
            ..default()
        };

        snapshot.crafts[0].translation.position = DVec3::splat(99.0);
        snapshot.crafts[0].authority = AuthorityMode::WarpIntegrated { integrator: 4 };

        assert_eq!(canonical.translation.position, DVec3::X);
        assert_eq!(
            canonical.authority,
            AuthorityMode::OnRails { trajectory: 0 }
        );
    }
}
