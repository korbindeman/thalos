//! Rendering module: celestial bodies, orbit lines, and ship marker.
//!
//! # Coordinate system
//! The physics simulation uses a heliocentric inertial frame with the ecliptic
//! as the XZ plane (Y up). All positions from the ephemeris are in metres.
//! Map-view systems apply the current [`WorldScale`] to convert metres to
//! render units. Ship-view body grids and the player ship live under BigSpace
//! so Bevy transforms stay local to 1 km cells.
//!
//! 1 render unit = 1 / WorldScale metres. Map view uses 1e-6 (1 unit = 1000 km);
//! ship-view meshes use 1.0 (1 unit = 1 m) inside BigSpace cells.

mod body_lod;
mod clouds;
mod generation;
pub(crate) mod ground_terrain;
mod lighting;
mod materials;
pub(crate) mod real_space;
mod scene_depth;
mod spawn;
mod terrain_residency;
mod trails;
mod transforms;
mod types;

pub use crate::solar_system_state::{SimulationState, SolarSystemState};
use body_lod::{LastClick, double_click_focus_system, focus_camera_on_homeworld, sync_body_icons};
use generation::{PendingPlanetInstalls, patch_reference_cloud_covers, poll_planet_install_tasks};
use ground_terrain::{
    pause_surface_terrain_streaming_at_high_warp, sync_body_render_lod,
    update_body_terrain_atmosphere,
};
use lighting::{
    sync_film_grain_to_exposure, update_camera_exposure, update_planet_light_dirs,
    update_solid_planet_params, update_sun_light,
};
use materials::{
    LastCloudBandUpdate, update_cloud_bands, update_gas_giant_params, update_ring_params,
};
use real_space::{
    attach_player_ship_to_big_space, attach_ship_camera_to_big_space, setup_big_space,
    update_real_space_body_positions,
};
use scene_depth::{SceneDepthPlugin, setup_scene_depth_image};
use spawn::spawn_bodies;
use terrain_residency::TerrainResidencyPlugin;
use trails::{draw_orbits, recompute_orbit_trails};
use transforms::{update_body_positions, update_planet_orientations, update_ship_position};
pub use transforms::{update_render_frame, update_render_origin};
pub use types::{
    CameraExposure, CelestialBody, PlanetshineTints, PlayerShip, RealSpaceBody, ShipMarker,
};

use bevy::prelude::*;
pub use thalos_body_render::ReferenceClouds;
use thalos_body_render::{convert_reference_clouds_when_ready, load_reference_cloud_sources};

use crate::SimStage;
use crate::solar_system_state::sync_solar_system_state;
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

pub struct RenderingPlugin;

impl Plugin for RenderingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SceneDepthPlugin)
            .add_plugins(TerrainResidencyPlugin)
            .add_plugins(clouds::CloudsRenderPlugin)
            .insert_resource(LastClick::default())
            .insert_resource(RenderOrigin::default())
            .insert_resource(RenderFrame::default())
            .insert_resource(PlanetshineTints::default())
            .insert_resource(CameraExposure::default())
            .register_type::<CameraExposure>()
            .init_resource::<ReferenceClouds>()
            .init_resource::<LastCloudBandUpdate>()
            .init_resource::<PendingPlanetInstalls>()
            .add_systems(
                Startup,
                (
                    configure_gizmos,
                    setup_big_space,
                    // `spawn_bodies` reads `SceneDepthImage` to seed each
                    // body's permanent `BodySky` material. The resource is
                    // inserted by `setup_scene_depth_image` via Commands,
                    // so we need an explicit `.after` to force a deferred-
                    // command flush before this runs.
                    spawn_bodies
                        .after(setup_big_space)
                        .after(setup_scene_depth_image),
                    attach_ship_camera_to_big_space
                        .after(setup_big_space)
                        .after(crate::camera::spawn_camera),
                    attach_player_ship_to_big_space
                        .after(setup_big_space)
                        .after(crate::ship_view::spawn_player_ship),
                    focus_camera_on_homeworld.after(spawn_bodies),
                    load_reference_cloud_sources,
                ),
            )
            .add_systems(
                Update,
                (
                    convert_reference_clouds_when_ready,
                    update_render_origin.after(sync_solar_system_state),
                    update_render_frame.after(sync_solar_system_state),
                    update_body_positions
                        .after(update_render_origin)
                        .after(crate::map_view::update_map_snapshot),
                    update_real_space_body_positions.after(sync_solar_system_state),
                    poll_planet_install_tasks.after(update_real_space_body_positions),
                    patch_reference_cloud_covers
                        .after(convert_reference_clouds_when_ready)
                        .after(poll_planet_install_tasks),
                    update_sun_light.after(sync_solar_system_state),
                    update_camera_exposure.after(sync_solar_system_state),
                    sync_film_grain_to_exposure.after(update_camera_exposure),
                    // Planet materials are inserted synchronously by
                    // `spawn_bodies` (no more async finalize), so these
                    // per-frame updaters no longer need to order against a
                    // generation system.
                    update_planet_light_dirs
                        .after(sync_solar_system_state)
                        .after(update_camera_exposure),
                    update_planet_orientations.after(sync_solar_system_state),
                    // Unified per-body render-LOD: one pass toggles
                    // terrain ↔ impostor (surface LOD) and BodySky ↔ halo
                    // (atmosphere vantage) from a single camera-to-body
                    // distance. Must run after `update_real_space_body_positions`
                    // (for current body world positions).
                    pause_surface_terrain_streaming_at_high_warp,
                    sync_body_render_lod
                        .after(update_real_space_body_positions)
                        .after(pause_surface_terrain_streaming_at_high_warp),
                    // `update_body_terrain_atmosphere` moved to PostUpdate
                    // (see below) so it reads body GlobalTransforms after
                    // big_space's `TransformSystems::Propagate` finishes
                    // recentering them. Running in Update on a snap
                    // teleport left the material uniform's `planet_center`
                    // a frame behind the view uniform's camera position;
                    // the unified atmosphere shader then saw the camera as
                    // "outside the shell" and discarded every pixel.
                    update_gas_giant_params
                        .after(sync_solar_system_state)
                        .after(update_camera_exposure),
                    update_solid_planet_params
                        .after(sync_solar_system_state)
                        .after(update_camera_exposure),
                    update_ring_params
                        .after(sync_solar_system_state)
                        .after(update_camera_exposure),
                    update_ship_position.after(update_render_origin),
                    recompute_orbit_trails.after(sync_solar_system_state),
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
                    update_cloud_bands.after(sync_solar_system_state),
                )
                    .in_set(SimStage::Sync),
            )
            .add_systems(
                bevy::app::PostUpdate,
                update_body_terrain_atmosphere.after(bevy::transform::TransformSystems::Propagate),
            );
    }
}

fn configure_gizmos(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.line.width = 2.0;
    // All current gizmos (orbit lines, trajectory previews, ghost trails)
    // are map-view overlays. Restrict the default group to MAP_LAYER so
    // the ship camera doesn't draw them. Ship-view debug overlays register
    // separate gizmo groups with SHIP_LAYER.
    config.render_layers = bevy::camera::visibility::RenderLayers::layer(crate::coords::MAP_LAYER);
}
