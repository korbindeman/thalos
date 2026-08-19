mod camera;
mod capture;
mod cli;
mod clouds;
mod diagnostics;
mod foliage;
mod ocean;
mod photo_mode;
mod places;
mod spatial;
mod terrain;
mod viewpoint;
mod world;

use std::time::Duration;

use anyhow::Result;
use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    asset::{AssetApp, AssetPlugin, io::AssetSourceBuilder},
    prelude::*,
    render::RenderPlugin,
    window::ExitCondition,
    winit::WinitPlugin,
};
use thalos_diagnostics::renderer_lease::{RendererLease, RendererRole};
use thalos_render_kit::{RenderCapabilities, RenderPlan, RenderPlanPlugin};

use camera::TerrainCameraPlugin;
use capture::CapturePlugin;
use cli::{CliAction, RunConfig, SpatialMode};
use clouds::CloudsPlugin;
use diagnostics::DiagnosticsPlugin;
use foliage::FoliagePlugin;
use ocean::OceanPlugin;
use photo_mode::KorsouPhotoModePlugin;
use places::PlacesPlugin;
use terrain::TerrainPlugin;
use viewpoint::ViewpointPlugin;
use world::WorldPlugin;

const DEFAULT_ASSET_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets");
const KORSOU_ASSET_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

fn asset_plugin() -> AssetPlugin {
    AssetPlugin {
        file_path: DEFAULT_ASSET_ROOT.into(),
        ..default()
    }
}

fn main() -> AppExit {
    match run() {
        Ok(exit) => exit,
        Err(error) => {
            eprintln!("error: {error:#}");
            AppExit::error()
        }
    }
}

fn run() -> Result<AppExit> {
    let config = match cli::parse()? {
        CliAction::Run(config) => config,
        CliAction::Help => {
            println!("{}", cli::help_text());
            return Ok(AppExit::Success);
        }
    };
    let headless = config.is_headless();
    let spatial = config.spatial;
    let viewpoints_path = config.viewpoints_path.clone();
    let renderer_role = if headless {
        RendererRole::CaptureHost
    } else {
        RendererRole::InteractiveGame
    };
    let _renderer_lease = RendererLease::acquire(renderer_role)?;
    let render_plan = match config.spatial {
        SpatialMode::Planar => RenderPlan::korsou_planar(),
        SpatialMode::Ellipsoid => RenderPlan::korsou_geodetic(),
    }
    .validate(RenderCapabilities::KORSOU)?;

    let mut preferences = if headless {
        thalos_runtime::preferences::AppPreferences::default()
    } else {
        thalos_runtime::preferences::load()
    };
    let quality_overrides = if headless {
        thalos_runtime::preferences::QualityOverrides::default()
    } else {
        thalos_runtime::preferences::QualityOverrides::from_env()
    };
    if let Some(preset) = quality_overrides.preset {
        preferences.apply_named_preset(preset);
    }
    let window_overrides = thalos_runtime::preferences::overrides_from_env();
    let window = thalos_runtime::preferences::initial_window(
        "Kòrsou — real-world island explorer",
        &preferences.window,
        &window_overrides,
    );

    let mut app = App::new();
    app.register_asset_source(
        "korsou",
        AssetSourceBuilder::platform_default(KORSOU_ASSET_ROOT, None),
    );
    app.insert_resource(ClearColor(Color::BLACK))
        .insert_resource(preferences.window)
        .insert_resource(preferences.graphics)
        .insert_resource(quality_overrides)
        .insert_resource(window_overrides);
    if headless {
        app.add_plugins(
            DefaultPlugins
                .set(asset_plugin())
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .set(RenderPlugin {
                    synchronous_pipeline_compilation: true,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )));
    } else {
        app.add_plugins(DefaultPlugins.set(asset_plugin()).set(WindowPlugin {
            primary_window: Some(window),
            ..default()
        }));
    }

    app.add_plugins(
        thalos_runtime::preferences::PreferencesPlugin::new(!headless)
            .with_foliage(spatial == SpatialMode::Planar)
            .with_clouds(true),
    )
    .add_plugins(thalos_runtime::viewer::ViewerPlugin::new(
        !headless, "ACTIVE",
    ))
    .add_plugins(RenderPlanPlugin::new(render_plan))
    .insert_resource::<RunConfig>(config)
    .add_plugins((
        TerrainPlugin,
        WorldPlugin::new(!headless),
        PlacesPlugin::new(!headless),
        ViewpointPlugin::new(viewpoints_path, !headless),
        TerrainCameraPlugin,
        CapturePlugin,
    ));
    if spatial == SpatialMode::Planar {
        app.add_plugins((FoliagePlugin, OceanPlugin));
    }
    app.add_plugins(CloudsPlugin);
    if !headless {
        app.add_plugins((DiagnosticsPlugin, KorsouPhotoModePlugin));
    }
    Ok(app.run())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{DEFAULT_ASSET_ROOT, KORSOU_ASSET_ROOT};

    #[test]
    fn default_asset_root_contains_shared_ui_fonts() {
        for font in [
            "Inter-Light.ttf",
            "Inter-Regular.ttf",
            "Inter-SemiBold.ttf",
            "FiraCode-Regular.ttf",
        ] {
            let path = Path::new(DEFAULT_ASSET_ROOT).join("fonts").join(font);
            assert!(
                path.is_file(),
                "shared UI font is missing: {}",
                path.display()
            );
        }
    }

    #[test]
    fn korsou_asset_source_contains_its_ocean_shader() {
        let path = Path::new(KORSOU_ASSET_ROOT).join("shaders/ocean.wgsl");
        assert!(
            path.is_file(),
            "Kòrsou ocean shader is missing: {}",
            path.display()
        );
        assert_eq!(crate::ocean::OCEAN_SHADER, "korsou://shaders/ocean.wgsl");
    }

    #[test]
    fn korsou_asset_source_contains_its_cloud_shader() {
        let path = Path::new(KORSOU_ASSET_ROOT).join("shaders/cloud_composite.wgsl");
        assert!(
            path.is_file(),
            "Kòrsou cloud shader is missing: {}",
            path.display()
        );
        assert_eq!(
            crate::clouds::CLOUD_SHADER,
            "korsou://shaders/cloud_composite.wgsl"
        );
    }
}
