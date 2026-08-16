//! Validated startup composition for rendering applications.
//!
//! The plan is concrete and restart-time: it records which spatial adapter and
//! appearance implementations an application selected, validates the
//! combination before rendering starts, exposes it as a Bevy resource, and
//! emits one structured diagnostic summary. Concrete adapters remain plugins
//! owned by their applications; this crate does not hide them behind dynamic
//! dispatch.

use bevy::{log::info, prelude::*};

pub use thalos_render_model::{
    AtmosphereAdapter, CloudAdapter, FarBodyAdapter, LightingAdapter, OceanAdapter,
    RenderCapabilities, RenderPlan, RenderPlanError, SpatialAdapter, TerrainAdapter,
    ValidatedRenderPlan,
};

#[derive(Resource, Debug, Clone, Copy)]
pub struct ActiveRenderPlan(ValidatedRenderPlan);

impl ActiveRenderPlan {
    pub const fn new(plan: ValidatedRenderPlan) -> Self {
        Self(plan)
    }

    pub const fn validated(self) -> ValidatedRenderPlan {
        self.0
    }

    pub const fn plan(self) -> RenderPlan {
        self.0.plan()
    }
}

pub struct RenderPlanPlugin {
    plan: ValidatedRenderPlan,
}

impl RenderPlanPlugin {
    pub const fn new(plan: ValidatedRenderPlan) -> Self {
        Self { plan }
    }
}

impl Plugin for RenderPlanPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ActiveRenderPlan::new(self.plan))
            .add_systems(Startup, emit_render_plan);
    }
}

fn emit_render_plan(active: Res<ActiveRenderPlan>) {
    let plan = active.plan();
    info!(
        target: "thalos::diagnostic::render_plan",
        event = "selection",
        spatial = plan.spatial.as_str(),
        terrain = plan.terrain.as_str(),
        atmosphere = plan.atmosphere.as_str(),
        ocean = plan.ocean.as_str(),
        clouds = plan.clouds.as_str(),
        lighting = plan.lighting.as_str(),
        far_body = plan.far_body.as_str(),
        "validated render plan"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_publishes_the_exact_validated_plan() {
        let plan = RenderPlan::korsou_planar()
            .validate(RenderCapabilities::KORSOU)
            .unwrap();
        let mut app = App::new();
        app.add_plugins(RenderPlanPlugin::new(plan));

        assert_eq!(app.world().resource::<ActiveRenderPlan>().validated(), plan);
    }
}
