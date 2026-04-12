pub mod bake;
mod material;
pub mod shader_types;
mod texture;

pub use bake::bake_from_body_data;
pub use material::{PlanetDetailParams, PlanetMaterial, PlanetMaterialHandle, PlanetParams};
pub use shader_types::{GpuCellRange, GpuCrater, GpuMaterial};
pub use texture::PlanetTextures;

use bevy::prelude::*;

/// Bevy plugin for planet impostor rendering.
///
/// Add this plugin to any Bevy app that needs to render planets (game, editor, etc.).
/// It registers the `PlanetMaterial` asset type — callers are responsible for
/// spawning entities with the material and updating uniforms per frame.
pub struct PlanetRenderingPlugin;

impl Plugin for PlanetRenderingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<PlanetMaterial>::default());
    }
}
