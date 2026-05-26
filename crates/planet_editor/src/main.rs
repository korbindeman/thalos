#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::asset::AssetPlugin;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use bevy::window::{PresentMode, WindowMode, WindowResolution};
use bevy_egui::egui;
use thalos_input::enhanced::{ActionSources, EnhancedInputSystems};
use thalos_input::planet_editor::{PlanetEditorInputIntent, PlanetEditorInputPlugin};
use thalos_input::settings::InputSettings;
use thalos_physics_canonical::body_trajectory_provider::BodyTrajectoryProvider;
use thalos_physics_canonical::canonical::Epoch;
use thalos_physics_canonical::parsing::load_solar_system_from_dir;
use thalos_physics_canonical::patched_conics::PatchedConics;
use thalos_physics_canonical::types::{BodyDefinition, BodyId, BodyKind, SolarSystemDefinition};
use thalos_planet_rendering::{
    AU_M, AtmosphereBlock, CLOUD_BAND_COUNT, GasGiantLayers, GasGiantMaterial,
    GasGiantMaterialHandle, GasGiantParams, LIGHT_AT_1AU, PlanetCoastlineParams,
    PlanetDetailParams, PlanetHaloMaterial, PlanetHaloMaterialHandle, PlanetMaterial,
    PlanetMaterialHandle, PlanetParams, PlanetRenderingPlugin, PlanetWaterParams, ReferenceClouds,
    RingLayers, RingMaterial, RingMaterialHandle, RingParams, SceneLighting, StarLight,
    bake_from_planet_surface, build_ring_mesh, cloud_cover_image_for_body,
    convert_reference_clouds_when_ready, load_reference_cloud_sources,
};
use thalos_terrain::{
    AirlessImpactProjectionConfig, AtmosphereSpec, AuthoredFeatureConfig, BodyArchetype,
    BoundaryKind, ColdDesertProjectionConfig, CompositionClass, DynamicSurfaceLayers,
    DynamicSurfaceState, FeatureId, FeatureLock, FeatureManifest, FeatureParamValue,
    FeatureProjectionConfig, FeatureSeed, FeatureSeedStream, FeatureTerrainConfig, HydrosphereSpec,
    IceInventory, MegabasinFeatureConfig, OceanTerrainConfig, PlanetSurface, PlateKind,
    StaticSurfaceData, TectonicActivity, TectonicConfig, TectonicSystem, TerrainCompileContext,
    TerrainCompileOptions, TerrainConfig, TerrainIntent, compile_terrain_config,
    plan_initial_compilation, sample_static_surface, sample_surface, sub_seed, surface_sample,
};
use thalos_terrain_render::{
    BodyTerrainMaterial, BodyTerrainShadow, PipelineTileProvider, ThalosTerrainPlugin,
    rendered_height_range,
};
use thalos_udlod::big_space::{BigSpaceCommands, FloatingOrigin, GridCell, ReferenceFrame};
use thalos_udlod::math::TileCoordinate;
use thalos_udlod::prelude::{
    AttachmentConfig, AttachmentFormat, TerrainBundle, TerrainConfig as UdlodTerrainConfig,
    TerrainViewComponents, TerrainViewConfig, TileAtlas, TileProvider, TileTree,
};

mod body_params;
mod camera;
mod materials;
mod preview;
mod sky_backdrop;
mod state;
mod surface_overlay;
mod tile_viewer;
mod ui;

use body_params::*;
use camera::*;
use materials::*;
use preview::*;
use sky_backdrop::SkyBackdropPlugin;
use state::*;
use surface_overlay::{
    SurfaceOverlayOrientation, SurfaceOverlayPlugin, SurfaceOverlayRenderRadius, activity_label,
};
use tile_viewer::*;
use ui::*;

fn main() {
    let body_arg = std::env::args().nth(1);
    let preferred = body_arg.as_deref().unwrap_or(DEFAULT_BODY_NAME);

    // Load from disk to include per-body detail files
    let system = {
        use std::path::Path;

        let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("assets"))
            .unwrap_or_else(|| "assets".into());

        load_solar_system_from_dir(&assets_dir).expect("load solar system from disk")
    };

    let find_body = |name: &str| {
        system
            .bodies
            .iter()
            .find(|b| b.name.eq_ignore_ascii_case(name))
    };
    let body = find_body(preferred)
        .or_else(|| find_body(DEFAULT_BODY_NAME))
        .unwrap_or(&system.bodies[0]);
    let selected_body = body.name.clone();
    let resolved = build_params_for_body(&system, body);

    let light_intensity = light_intensity_at(resolved.heliocentric_distance_m);

    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(
            InputSettings::load_from_path("assets/input.ron")
                .expect("Failed to load input bindings from assets/input.ron"),
        )
        .insert_resource(SystemData { system })
        .insert_resource(EditedPlanet {
            selected_body,
            radius_m: resolved.radius_m,
            gravity_m_s2: resolved.gravity_m_s2,
            axial_tilt_rad: resolved.axial_tilt_rad,
            mode: resolved.mode,
            rings: resolved.rings,
            atmosphere: resolved.atmosphere,
            atmosphere_enabled: true,
            heliocentric_distance_m: resolved.heliocentric_distance_m,
            light_intensity,
            sun_azimuth: 0.0,
            sun_orbital_elevation: resolved.sun_orbital_elevation,
            full_bright: false,
            ambient_light: false,
            terrain_dirty: false,
            uniforms_dirty: false,
            body_changed: false,
            last_edit: None,
            requested_bake: None,
            last_bake_mode: TerrainBakeMode::Preview,
            selected_feature_id: None,
            tool: ToolMode::default(),
        })
        .init_resource::<OrbitCamera>()
        .init_resource::<TileViewerState>()
        .init_resource::<EquirectViewerState>()
        .init_resource::<ActivePreviewSurface>()
        .init_resource::<TerrainGenStatus>()
        .init_resource::<PreviewAtmosphereClock>()
        .init_resource::<ReferenceClouds>()
        .add_plugins(
            DefaultPlugins
                .build()
                .disable::<bevy::transform::TransformPlugin>()
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Thalos — Planet Editor".into(),
                        present_mode: PresentMode::AutoVsync,
                        mode: WindowMode::Windowed,
                        resolution: WindowResolution::new(1920, 1080),
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: "../../assets".to_string(),
                    ..default()
                }),
        )
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(bevy_egui::EguiPlugin::default())
        .add_plugins(PlanetRenderingPlugin)
        .add_plugins(ThalosTerrainPlugin)
        .add_plugins(PlanetEditorInputPlugin)
        .add_plugins(SkyBackdropPlugin)
        .add_plugins(SurfaceOverlayPlugin)
        .insert_resource(SurfaceOverlayRenderRadius(RENDER_RADIUS))
        .init_resource::<SurfaceOverlayOrientation>()
        .add_systems(
            Startup,
            (
                load_reference_cloud_sources,
                spawn_camera,
                spawn_preview_planet,
            ),
        )
        .add_systems(bevy_egui::EguiPrimaryContextPass, editor_ui)
        .add_systems(
            PreUpdate,
            gate_editor_input_sources.before(EnhancedInputSystems::Update),
        )
        .add_systems(
            Update,
            (
                convert_reference_clouds_when_ready,
                camera_input,
                camera_zoom_smoothing.after(camera_input),
                camera_apply_transform.after(camera_zoom_smoothing),
                pick_planet_click.after(camera_input),
                toggle_fullbright_hotkey,
                apply_uniform_changes.after(toggle_fullbright_hotkey),
                handle_body_switch,
                dispatch_rebake
                    .after(handle_body_switch)
                    .after(pick_planet_click),
                finalize_terrain_bake.after(dispatch_rebake),
                sync_tile_viewer_preview_visibility,
                update_tile_viewer_terrain.after(finalize_terrain_bake),
                update_tile_viewer_materials.after(update_tile_viewer_terrain),
                patch_preview_reference_cloud_cover
                    .after(convert_reference_clouds_when_ready)
                    .after(finalize_terrain_bake),
                update_preview_atmosphere
                    .after(apply_uniform_changes)
                    .after(finalize_terrain_bake),
                sync_surface_overlay_orientation,
            ),
        )
        .run();
}
