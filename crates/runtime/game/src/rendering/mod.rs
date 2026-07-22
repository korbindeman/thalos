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
pub(crate) mod contact_shadow;
mod generation;
mod gpu_grass;
mod grass;
pub(crate) mod ground_terrain;
mod lighting;
mod map_terrain;
mod materials;
mod ocean;
pub(crate) mod real_space;
pub(crate) mod tile_cache;
// Scattered pebble/rock decoration is disabled — no rocks on the surface.
// Re-enable by uncommenting this and the `RockScatterPlugin` registration below.
// mod rocks;
mod scene_depth;
mod spawn;
pub(crate) mod ssao;
pub(crate) mod sun_shadow;
pub(crate) mod terrain_residency;
mod trails;
mod transforms;
mod types;
mod vegetation;
pub(crate) mod view_anchor;

/// Cascade count shared by **every** game `DirectionalLight` that can cast
/// shadows (the world `SunLight`, the shipyard editor's key light, and the
/// otherwise-shadowless `MoonLight`).
///
/// Bevy 0.19's `check_dir_light_mesh_visibility` reuses one
/// `Local<Parallel<Vec<Vec<Entity>>>>` thread-queue across frames *and* across
/// lights, resizing each participating worker's slot to the current light's
/// cascade count. If two shadow-casting directional lights disagree on that
/// count, a worker truncated by the smaller-cascade light and then skipped by
/// the larger-cascade light's `par_iter` gets over-indexed at collection —
/// panicking with `index out of bounds` (observed as a 2-vs-4 mismatch between
/// `SunLight` and the shipyard key light). Keeping one count everywhere makes
/// the thread-queue slots uniform so the over-index can never happen.
pub const SHADOW_CASCADE_COUNT: usize = 2;

pub use crate::solar_system_state::{SimulationState, SolarSystemState};
use body_lod::{LastClick, double_click_focus_system, focus_camera_on_homeworld, sync_body_icons};
use ground_terrain::{
    pause_surface_terrain_streaming_at_high_warp, sync_body_render_lod,
    update_body_terrain_atmosphere,
};
use lighting::{
    sync_film_grain_to_exposure, update_camera_exposure, update_moon_light,
    update_solid_planet_params, update_sun_light,
};
use materials::{
    LastCloudBandUpdate, update_cloud_bands, update_gas_giant_params, update_ring_params,
};
use real_space::{
    attach_player_ship_to_big_space, attach_ship_camera_to_big_space, setup_big_space,
    update_real_space_body_positions,
};
use scene_depth::SceneDepthPlugin;
use spawn::spawn_bodies;
use terrain_residency::TerrainResidencyPlugin;
use trails::{draw_orbits, recompute_orbit_trails};
use transforms::{update_body_positions, update_ship_position};
pub use transforms::{update_render_frame, update_render_origin};
pub use types::{
    ActiveCraft, CameraExposure, CelestialBody, PlanetshineTints, PlayerShip, RealSpaceBody,
    ShipMarker, track_active_craft,
};

use crate::SimStage;
use crate::solar_system_state::sync_solar_system_state;
use bevy::prelude::*;
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
        app.add_plugins(view_anchor::ViewAnchorPlugin)
            .add_plugins(SceneDepthPlugin)
            .add_plugins(ssao::SsaoPlugin)
            // Contact tier of the three-tier shadow split
            // (ADR-20260722T111848Z); reads the same `SceneDepthImage` as SSAO.
            .add_plugins(contact_shadow::ContactShadowPlugin)
            .add_plugins(sun_shadow::SunShadowPlugin)
            // Must precede the terrain plugins: they spawn terrain (and thus ask
            // for cached providers) and the registry has to exist by then.
            .add_plugins(tile_cache::TileCachePlugin)
            .add_plugins(TerrainResidencyPlugin)
            .add_plugins(map_terrain::MapTerrainPlugin)
            .add_plugins(clouds::CloudsRenderPlugin)
            .add_plugins(ocean::OceanRenderPlugin)
            .add_plugins(grass::GrassRenderPlugin)
            .add_plugins(gpu_grass::GpuGrassPlugin)
            .add_plugins(vegetation::VegetationRenderPlugin)
            // Rocks/pebbles disabled — see `mod rocks` above.
            // .add_plugins(rocks::RockScatterPlugin)
            .insert_resource(LastClick::default())
            .insert_resource(RenderOrigin::default())
            .insert_resource(RenderFrame::default())
            .insert_resource(PlanetshineTints::default())
            .insert_resource(CameraExposure::default())
            .register_type::<CameraExposure>()
            .register_type::<ground_terrain::AtmosphereTuning>()
            .init_resource::<ground_terrain::AtmosphereTuning>()
            .init_resource::<ground_terrain::OceanDebugSettings>()
            .init_resource::<LastCloudBandUpdate>()
            .init_resource::<ActiveCraft>()
            // The N-craft accessor seam: keep `ActiveCraft` mirroring the active
            // craft entity each frame (`SimStage::Sync`, before consumers read it).
            .add_systems(Update, track_active_craft.in_set(SimStage::Sync))
            .add_systems(
                Startup,
                (
                    configure_gizmos,
                    setup_big_space,
                    attach_ship_camera_to_big_space
                        .after(setup_big_space)
                        .after(crate::camera::spawn_camera),
                ),
            )
            // World spawn is keyed to `WorldState::Live`, not `Startup`: a
            // scenario boot QUEUES `Live` from a `Startup` system (never
            // `insert_state(Live)` at build — Bevy runs the initial
            // `StateTransition` BEFORE `PreStartup`, which would fire this
            // chain before `setup_big_space` / `setup_scene_depth_image`
            // commands — which `spawn_bodies` reads — are flushed; see
            // `main.rs`). The queued transition applies at the first regular
            // `StateTransition`, same frame, after `Startup`. A bare menu boot
            // stays `Absent` and spawns nothing until the menu starts a
            // scenario. The chain gives `focus_camera_on_homeworld` a sync
            // point so it sees the bodies `spawn_bodies` just queued. The
            // player ship gets its BigSpace attach from the per-frame
            // `attach_player_ship_to_big_space` pass below, same as
            // relaunch-built craft.
            .add_systems(
                OnEnter(crate::loading::WorldState::Live),
                (spawn_bodies, focus_camera_on_homeworld).chain(),
            )
            .add_systems(
                Update,
                (
                    // Runtime-built craft (relaunch / start-screen craft
                    // swaps) need the same BigSpace attach the boot craft
                    // gets at Startup; no-op once attached (see the fn docs).
                    attach_player_ship_to_big_space,
                    update_render_origin.after(sync_solar_system_state),
                    update_render_frame.after(sync_solar_system_state),
                    update_body_positions
                        .after(update_render_origin)
                        .after(crate::map_view::update_map_snapshot),
                    update_real_space_body_positions.after(sync_solar_system_state),
                    // F1: the Bevy sun is a projection of the spine's heliocentric
                    // flux, so it must read the per-frame exposure gain first.
                    update_sun_light
                        .after(sync_solar_system_state)
                        .after(update_camera_exposure),
                    update_moon_light.after(sync_solar_system_state),
                    update_camera_exposure.after(sync_solar_system_state),
                    sync_film_grain_to_exposure.after(update_camera_exposure),
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
                        .run_if(
                            crate::photo_mode::not_in_photo_mode.and_then(crate::view::in_map_view),
                        ),
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
