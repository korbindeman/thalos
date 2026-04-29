//! Unified flight plan view: ghost bodies, trajectory rendering, and lifecycle
//! management as a single coherent system.
//!
//! The physics crate produces a [`FlightPlan`] with legs, encounters, and
//! segments. This module builds a game-side view on top of that:
//!
//! - **Ghost bodies** — translucent spheres at future encounter positions,
//!   positioned at closest approach (or maneuver time if a node sits on the
//!   encounter leg). Ghost bodies serve as the rendering anchor for their
//!   encounter legs.
//! - **Trajectory rendering** — continuous gizmo line through all legs, each
//!   drawn relative to its SOI body. Future encounter legs pin to the ghost
//!   body's transform.
//! - **Lifecycle** — ghosts spawn when encounters appear in the prediction,
//!   persist stably across repredictions (diff-based, no churn), blend out
//!   as the real body catches up, and hand off camera focus on retirement.

mod ghost;
mod markers;
mod render;
mod view;

pub use ghost::GhostBody;
pub use view::FlightPlanView;

use bevy::prelude::*;

pub struct FlightPlanViewPlugin;

impl Plugin for FlightPlanViewPlugin {
    fn build(&self, app: &mut App) {
        // Lifecycle runs BEFORE sync_ghost_bodies so retired ghosts (either
        // from sim-time advance or reconcile churn) get a chance to hand off
        // camera focus before their entity is despawned.
        app.init_resource::<FlightPlanView>()
            .add_systems(Startup, markers::setup_trajectory_marker_assets)
            .add_systems(
                Update,
                (
                    view::rebuild_flight_plan_view,
                    ghost::update_ghost_lifecycle,
                    ghost::sync_ghost_bodies,
                    ghost::update_ghost_transforms,
                    render::render_trajectory.run_if(
                        crate::photo_mode::not_in_photo_mode
                            .and(crate::view::in_map_view),
                    ),
                    markers::manage_trajectory_markers.run_if(
                        crate::photo_mode::not_in_photo_mode
                            .and(crate::view::in_map_view),
                    ),
                )
                    .chain()
                    .after(crate::rendering::cache_body_states)
                    .after(crate::rendering::update_render_frame)
                    .in_set(crate::SimStage::Sync),
            );
    }
}
