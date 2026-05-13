//! Stage 2/3 terrain material for procedural bodies.
//!
//! Reads the height + albedo attachments produced by
//! [`crate::pipeline::PipelineTileProvider`] and shades raw albedo (no PBR
//! lighting yet). PBR + atmospheric optics land in M4; for M3 we just want
//! the surface to read as the same body the impostor was showing from far
//! away, with seamless LOD transitions as the camera approaches.

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct BodyTerrainMaterial {}

impl Material for BodyTerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_terrain/body_terrain.wgsl".into()
    }
}

pub(crate) fn embed_body_terrain_shader(app: &mut App) {
    embedded_asset!(app, "body_terrain.wgsl");
}
