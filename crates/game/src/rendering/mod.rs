//! Rendering module: celestial bodies, orbit lines, and ship marker.
//!
//! # Coordinate system
//! The physics simulation uses a heliocentric inertial frame with the ecliptic
//! as the XZ plane (Y up). All positions from the ephemeris are in metres.
//! We apply the current [`WorldScale`] to convert metres to render units so Bevy's f32
//! transforms don't lose precision on solar-system distances.
//!
//! 1 render unit = 1 / WorldScale metres. Map view uses 1e-6 (1 unit = 1000 km); ship view uses 1.0 (1 unit = 1 m).

mod body_lod;
mod generation;
mod lighting;
mod materials;
mod spawn;
mod trails;
mod transforms;
mod types;

use body_lod::{LastClick, double_click_focus_system, focus_camera_on_homeworld, sync_body_icons};
use generation::{finalize_planet_generation, patch_reference_cloud_covers};
use lighting::{
    sync_film_grain_to_exposure, update_camera_exposure, update_planet_light_dirs,
    update_solid_planet_params, update_sun_light,
};
use materials::{
    LastCloudBandUpdate, update_cloud_bands, update_gas_giant_params, update_ring_params,
};
use spawn::spawn_bodies;
use trails::{draw_orbits, recompute_orbit_trails};
use transforms::{
    update_body_positions, update_planet_orientations, update_ship_body_meshes,
    update_ship_position,
};
pub use transforms::{update_render_frame, update_render_origin};
pub use types::{
    CameraExposure, CelestialBody, FrameBodyStates, PlanetshineTints, PlayerShip, ShipMarker,
    SimulationState,
};

use bevy::prelude::*;
pub use thalos_planet_rendering::ReferenceClouds;
use thalos_planet_rendering::{convert_reference_clouds_when_ready, load_reference_cloud_sources};

use crate::SimStage;
// Re-export so existing `use crate::rendering::{RenderFrame, RenderOrigin}` sites keep working.
pub use crate::coords::{RenderFrame, RenderOrigin};

/// Radius of screen-stable marker billboards as a fraction of camera distance
/// (in render units). Bodies whose rendered sphere is smaller than this get
/// replaced by a fixed-size circle billboard, and map-view marker overlays
/// use the same value so every marker family has the same screen size.
pub(crate) const SCREEN_MARKER_RADIUS: f32 = 0.006;

/// Render-space radius for a marker that should keep the same screen size as
/// the body icon billboards.
#[inline]
pub(crate) fn screen_marker_radius(world_pos: Vec3, camera_pos: Vec3) -> f32 {
    (world_pos - camera_pos).length().max(1.0) * SCREEN_MARKER_RADIUS
}

pub fn cache_body_states(sim: Res<SimulationState>, mut cache: ResMut<FrameBodyStates>) {
    let t = sim.simulation.sim_time();
    if cache.states.is_some() && (t - cache.time).abs() < f64::EPSILON {
        return;
    }
    if let Some(states) = cache.states.as_mut() {
        sim.ephemeris.query_into(t, states);
    } else {
        let mut states = Vec::with_capacity(sim.ephemeris.body_count());
        sim.ephemeris.query_into(t, &mut states);
        cache.states = Some(states);
    }
    cache.time = t;
}

pub struct RenderingPlugin;

impl Plugin for RenderingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(LastClick::default())
            .insert_resource(RenderOrigin::default())
            .insert_resource(RenderFrame::default())
            .insert_resource(FrameBodyStates::default())
            .insert_resource(PlanetshineTints::default())
            .insert_resource(CameraExposure::default())
            .init_resource::<ReferenceClouds>()
            .init_resource::<LastCloudBandUpdate>()
            .add_systems(
                Startup,
                (
                    configure_gizmos,
                    spawn_bodies,
                    focus_camera_on_homeworld.after(spawn_bodies),
                    load_reference_cloud_sources,
                ),
            )
            .add_systems(
                Update,
                (
                    convert_reference_clouds_when_ready,
                    patch_reference_cloud_covers.after(convert_reference_clouds_when_ready),
                    finalize_planet_generation,
                    cache_body_states,
                    update_render_origin.after(cache_body_states),
                    update_render_frame.after(cache_body_states),
                    update_body_positions.after(update_render_origin),
                    update_ship_body_meshes.after(update_body_positions),
                    update_sun_light.after(cache_body_states),
                    update_camera_exposure.after(cache_body_states),
                    sync_film_grain_to_exposure.after(update_camera_exposure),
                    update_planet_light_dirs
                        .after(cache_body_states)
                        .after(update_camera_exposure)
                        .after(finalize_planet_generation),
                    update_planet_orientations
                        .after(cache_body_states)
                        .after(finalize_planet_generation),
                    update_gas_giant_params
                        .after(cache_body_states)
                        .after(update_camera_exposure),
                    update_solid_planet_params
                        .after(cache_body_states)
                        .after(update_camera_exposure),
                    update_ring_params
                        .after(cache_body_states)
                        .after(update_camera_exposure),
                    update_ship_position.after(update_render_origin),
                    recompute_orbit_trails.after(cache_body_states),
                )
                    .in_set(SimStage::Sync),
            )
            .add_systems(
                Update,
                (
                    draw_orbits
                        .after(recompute_orbit_trails)
                        .after(update_render_origin)
                        .run_if(crate::photo_mode::not_in_photo_mode.and(crate::view::in_map_view)),
                    sync_body_icons.run_if(crate::view::in_map_view),
                    double_click_focus_system
                        .after(update_ship_position)
                        .run_if(crate::view::in_map_view),
                    update_cloud_bands.after(finalize_planet_generation),
                )
                    .in_set(SimStage::Sync),
            );
    }
}

fn configure_gizmos(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.line.width = 2.0;
    // All current gizmos (orbit lines, trajectory previews, ghost trails)
    // are map-view overlays. Restrict the default group to MAP_LAYER so
    // the ship camera doesn't draw them. If ship-view gizmos are ever
    // needed, register a separate gizmo group with SHIP_LAYER.
    config.render_layers = bevy::camera::visibility::RenderLayers::layer(crate::coords::MAP_LAYER);
}
