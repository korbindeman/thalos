//! Thalos integration of the forked `bevy_terrain` UDLOD renderer.
//!
//! M3 staging ([docs/terrain.md](../../docs/terrain.md), "Rendering: ground
//! LOD"):
//!
//! - **Stage 1**: wire the fork into the workspace + run a deterministic
//!   [`SyntheticTileProvider`] end-to-end via `examples/playground.rs`.
//! - **Stage 2**: [`PipelineTileProvider`] forwards tile requests to
//!   [`thalos_terrain_gen::sample_static_surface`], so the same `PlanetSurface`
//!   the impostor billboard bakes also drives the UDLOD surface.
//! - **Stage 3**: per-body wiring lives in `thalos_game`'s
//!   `rendering::ground_terrain` module.
//!
//! ## Plugin layering
//!
//! [`ThalosTerrainPlugin`] adds [`bevy_terrain::prelude::TerrainPlugin`] +
//! [`bevy::pbr::MaterialPlugin<BodyTerrainMaterial>`] and embeds the body
//! terrain shader. `bevy_terrain::TerrainPlugin` in turn adds
//! [`big_space::prelude::BigSpaceDefaultPlugins`] when the `high_precision`
//! feature is enabled, so consumers that already register
//! `BigSpaceDefaultPlugins` directly must drop that registration to avoid the
//! duplicate-plugin panic.

use bevy::prelude::*;
use bevy_terrain::prelude::{TerrainMaterialPlugin, TerrainPlugin};

mod body_material;
mod pipeline;
mod playground_material;
mod synthetic;

pub use body_material::BodyTerrainMaterial;
pub use pipeline::PipelineTileProvider;
pub use playground_material::PlaygroundMaterial;
pub use synthetic::SyntheticTileProvider;

pub struct ThalosTerrainPlugin;

impl Plugin for ThalosTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(TerrainPlugin);
        app.add_plugins(TerrainMaterialPlugin::<BodyTerrainMaterial>::default());
        body_material::embed_body_terrain_shader(app);
        playground_material::embed_playground_shader(app);
    }
}
