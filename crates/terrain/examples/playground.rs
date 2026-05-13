//! Stage 1 sanity check for M3: drives `bevy_terrain` end-to-end inside the
//! Thalos workspace using a deterministic [`SyntheticTileProvider`] so we
//! don't depend on preprocessed disk assets.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p thalos_terrain --example playground
//! ```
//!
//! This binary is intentionally self-contained — it spawns its own
//! `BigSpace` root rather than reusing the game's hierarchy. The
//! game-side integration (parenting the terrain entity to a body's
//! `Grid` so it inherits orbital motion) lands in Stage 2.

use bevy::{math::DVec3, prelude::*};
use bevy_terrain::prelude::*;
use thalos_terrain::{PlaygroundMaterial, SyntheticTileProvider, ThalosTerrainPlugin};

/// Mira-scale body so the synthetic ridge field reads at roughly the right
/// scale relative to the camera. Not loaded from the solar-system asset —
/// the playground intentionally avoids depending on Thalos's body catalog.
const RADIUS: f64 = 1_700_000.0;
const MIN_HEIGHT: f32 = -2_500.0;
const MAX_HEIGHT: f32 = 4_500.0;
const TEXTURE_SIZE: u32 = 512;
const LOD_COUNT: u32 = 16;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.build().disable::<bevy::transform::TransformPlugin>(),
            ThalosTerrainPlugin,
            TerrainMaterialPlugin::<PlaygroundMaterial>::default(),
            TerrainDebugPlugin,
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PlaygroundMaterial>>,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
) {
    let config = TerrainConfig {
        lod_count: LOD_COUNT,
        model: TerrainModel::sphere(DVec3::ZERO, RADIUS, MIN_HEIGHT, MAX_HEIGHT),
        path: "thalos_terrain/playground".to_string(),
        ..default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: TEXTURE_SIZE,
        border_size: 2,
        mip_level_count: 4,
        format: AttachmentFormat::R16,
    });

    let provider: Box<dyn TileProvider> =
        Box::new(SyntheticTileProvider::new(MIN_HEIGHT, MAX_HEIGHT));
    let tile_atlas = TileAtlas::with_provider(&config, provider);
    let view_config = TerrainViewConfig::default();
    let tile_tree = TileTree::new(&tile_atlas, &view_config);

    commands.spawn_big_space(ReferenceFrame::default(), |root| {
        let frame = root.grid().clone();

        let terrain = root
            .spawn_spatial((
                TerrainBundle::new(tile_atlas, &frame),
                MeshMaterial3d(materials.add(PlaygroundMaterial::default())),
            ))
            .id();

        let view = root
            .spawn_spatial(DebugCameraBundle::new(
                -DVec3::X * RADIUS * 3.0,
                RADIUS,
                &frame,
            ))
            .id();

        tile_trees.insert((terrain, view), tile_tree);

        let sun_position = DVec3::new(-1.0, 1.0, -1.0) * RADIUS * 10.0;
        let (sun_cell, sun_translation) = frame.translation_to_grid(sun_position);
        root.spawn_spatial((
            Mesh3d(meshes.add(Sphere::new(RADIUS as f32 * 2.0).mesh().build())),
            Transform::from_translation(sun_translation),
            sun_cell,
        ));
    });
}
