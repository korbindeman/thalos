//! Minimal terrain material used by the Stage 1 playground binary. Renders a
//! height-based grayscale so the synthetic ridge field is visible without
//! authoring a gradient asset.
//!
//! Moves into the crate (not the example) so the embedded shader has a stable
//! crate-keyed asset path (`embedded://thalos_terrain/playground_terrain.wgsl`)
//! and any future test consumer can reuse it.

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct PlaygroundMaterial {}

impl Material for PlaygroundMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_terrain/playground_terrain.wgsl".into()
    }
}

pub(crate) fn embed_playground_shader(app: &mut App) {
    embedded_asset!(app, "playground_terrain.wgsl");
}
