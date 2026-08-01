//! Navball — procedural attitude indicator.
//!
//! Pipeline:
//! 1. [`texture`] bakes an equirectangular RGBA8 image at startup.
//! 2. [`render`] spawns a unit sphere with [`material::NavballMaterial`]
//!    on its own render layer, plus a dedicated camera that renders the
//!    sphere to an off-screen `Image` (limb-darkened via the WGSL).
//! 3. [`ui`] displays the off-screen image as a `bevy_ui` `ImageNode`.
//!
//! Attitude rotation and direction markers come in later modules.

pub mod attitude;
pub mod markers;
pub mod material;
pub mod render;
pub mod texture;
pub mod ui;

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

use thalos_game_state::sched::SimStage;

pub struct NavballPlugin;

impl Plugin for NavballPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<material::NavballMaterial>::default())
            .init_resource::<attitude::NavballFrame>()
            .add_systems(Startup, render::setup_navball_render)
            .add_systems(
                Startup,
                ui::setup_navball_ui
                    .after(render::setup_navball_render)
                    .after(crate::theme::init_theme),
            )
            .add_systems(
                Startup,
                markers::setup_navball_markers.after(ui::setup_navball_ui),
            )
            .add_systems(
                Update,
                (
                    attitude::drive_navball_attitude,
                    markers::update_navball_markers.after(attitude::drive_navball_attitude),
                )
                    .after(SimStage::Sync),
            );
    }
}
