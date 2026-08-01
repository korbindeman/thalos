//! The map-view snapshot boundary: read-only projections of canonical
//! simulation state for map rendering. Map systems consume [`MapSnapshot`]
//! and never touch the live `Simulation` or real-space render entities.

use bevy::math::{DVec3, Vec3};
use bevy::prelude::*;
use thalos_physics_canonical::canonical::{CraftState, Epoch};
use thalos_physics_canonical::trajectory::{FlightPlan, TrajectoryBranchStack};
use thalos_physics_canonical::types::{BodyState, BodyStates};
use thalos_world::{BodyDefinition, BodyId};

use crate::coords::{RenderOrigin, WorldScale};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct MapContext {
    pub origin: DVec3,
    pub scale: f64,
    pub focus_body: BodyId,
}

pub trait MapProjection: Send + Sync {
    fn project_body(&self, body: &BodyState, ctx: &MapContext) -> Vec3;
    fn project_point(&self, point_inertial_m: DVec3, epoch: Epoch, ctx: &MapContext) -> Vec3;
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
}

/// Read-only projection of canonical simulation state for map rendering.
///
/// Map systems consume this snapshot and never touch the live `Simulation` or
/// real-space render entities. `body_states` is copied wholesale from
/// [`SolarSystemState::states`](crate::solar_system::SolarSystemState), so it
/// inherits the index-aligned `states[i].id == i` invariant the map's body
/// lookups rely on.
///
/// **Sole writer:** the runtime's `update_map_snapshot`. Every other map
/// system reads it.
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
