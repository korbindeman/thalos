use bevy::{asset::embedded_asset, prelude::*};
use itertools::Itertools;

pub const DEFAULT_VERTEX_SHADER: &str = "embedded://thalos_udlod/shaders/render/vertex.wgsl";
pub const DEFAULT_FRAGMENT_SHADER: &str = "embedded://thalos_udlod/shaders/render/fragment.wgsl";

#[derive(Default, Resource)]
pub(crate) struct InternalShaders(Vec<Handle<Shader>>);

impl InternalShaders {
    pub(crate) fn load(app: &mut App, shaders: &[&'static str]) {
        let mut shaders = shaders
            .iter()
            .map(|&shader| app.world_mut().resource_mut::<AssetServer>().load(shader))
            .collect_vec();

        let mut internal_shaders = app.world_mut().resource_mut::<InternalShaders>();
        internal_shaders.0.append(&mut shaders);
    }
}

pub(crate) fn load_terrain_shaders(app: &mut App) {
    embedded_asset!(app, "types.wgsl");
    embedded_asset!(app, "attachments.wgsl");
    embedded_asset!(app, "bindings.wgsl");
    embedded_asset!(app, "functions.wgsl");
    embedded_asset!(app, "debug.wgsl");
    embedded_asset!(app, "render/vertex.wgsl");
    embedded_asset!(app, "render/fragment.wgsl");

    InternalShaders::load(
        app,
        &[
            "embedded://thalos_udlod/shaders/types.wgsl",
            "embedded://thalos_udlod/shaders/attachments.wgsl",
            "embedded://thalos_udlod/shaders/bindings.wgsl",
            "embedded://thalos_udlod/shaders/functions.wgsl",
            "embedded://thalos_udlod/shaders/debug.wgsl",
            "embedded://thalos_udlod/shaders/render/vertex.wgsl",
            "embedded://thalos_udlod/shaders/render/fragment.wgsl",
        ],
    );
}
