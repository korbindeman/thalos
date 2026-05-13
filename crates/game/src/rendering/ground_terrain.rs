//! Spawn ground-LOD terrain entities for procedural bodies.
//!
//! Called from `generation::finalize_planet_generation` once a body's
//! `PlanetSurface` task resolves. The terrain entity is parented to the
//! body's real-space `Grid` so it inherits orbital + rotational motion
//! automatically (`bevy_terrain`'s `high_precision` integration handles
//! surface-scale precision via its Taylor-series approximation; we don't
//! nest any additional cells).
//!
//! The same `PlanetSurface` that drives the impostor billboard's
//! `PlanetMaterial` is shared with the [`PipelineTileProvider`] via `Arc`,
//! so there is exactly one copy of the heavy cubemap data per body.

use std::sync::Arc;

use bevy::camera::visibility::RenderLayers;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy_terrain::prelude::*;
use big_space::grid::Grid;
use thalos_physics::types::BodyDefinition;
use thalos_terrain::{BodyTerrainMaterial, PipelineTileProvider};
use thalos_terrain_gen::PlanetSurface;

use crate::coords::SHIP_LAYER;

use super::real_space::REAL_SPACE_CELL_SIZE_M;

/// LOD depth for body terrains. 16 LODs over a Mira-scale body
/// (~10.7 Mm circumference) gives ~1.3 m at the deepest tile texel — well
/// past where the synthesized cubemap data carries meaningful detail. Stage
/// 4+ may pin this to per-body values once the v2 backlog provides true
/// arbitrary-resolution synthesis.
const LOD_COUNT: u32 = 16;

/// Tile attachment resolution. 512 is the upstream default and matches what
/// the playground example uses; bevy_terrain mandates power-of-two for
/// mipmap generation. Larger sizes trade tile latency for fewer LOD levels.
const TILE_TEXTURE_SIZE: u32 = 512;
const TILE_BORDER_SIZE: u32 = 2;
const TILE_MIP_LEVELS: u32 = 4;

/// Resident tiles per body. Upstream defaults to 1024, which is tuned for
/// one giant terrain. With four bodies the atlas memory adds up quickly
/// (one slot at 512² is ~750 KB across height + albedo); 256 is enough to
/// fit the typical visible-tile set of the focused body plus a few tiles
/// per distant body.
const ATLAS_SIZE: u32 = 256;

/// Marker on the terrain entity spawned for a procedural body.
#[derive(Component, Debug)]
pub(super) struct BodyTerrain;

/// Spawn the UDLOD terrain for one procedural body.
///
/// `ship_parent_entity` is the body's `RealSpaceBody` entity (the 1 km-cell
/// body grid). `ship_camera` is the entity carrying [`crate::camera::ShipCamera`];
/// the tile-tree resource is keyed by `(terrain, view)` so we have to know the
/// view at spawn time.
pub(super) fn spawn_body_terrain(
    commands: &mut Commands,
    body: &BodyDefinition,
    surface: Arc<PlanetSurface>,
    ship_parent_entity: Entity,
    materials: &mut Assets<BodyTerrainMaterial>,
    tile_trees: &mut TerrainViewComponents<TileTree>,
    ship_camera: Entity,
) {
    let radius_m = body.radius_m as f32;
    let height_range = surface.static_surface.height_range;

    let model = TerrainModel::sphere(
        DVec3::ZERO,
        body.radius_m,
        -height_range,
        height_range,
    );

    let config = TerrainConfig {
        lod_count: LOD_COUNT,
        model,
        path: format!("thalos/{}", body.name.to_lowercase()),
        atlas_size: ATLAS_SIZE,
        ..Default::default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: TILE_TEXTURE_SIZE,
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        format: AttachmentFormat::R16,
    })
    .add_attachment(AttachmentConfig {
        name: "albedo".to_string(),
        texture_size: TILE_TEXTURE_SIZE,
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        format: AttachmentFormat::Rgba8,
    });

    let provider = PipelineTileProvider::new(body.name.clone(), surface, radius_m, height_range);
    let tile_atlas = TileAtlas::with_provider(&config, Box::new(provider));
    let view_config = TerrainViewConfig::default();
    let tile_tree = TileTree::new(&tile_atlas, &view_config);

    // The body's `real_body_entity` (`ship_parent_entity`) carries a
    // `Grid::new(REAL_SPACE_CELL_SIZE_M, 0.0)`. Reconstructing it here is
    // cheaper than threading the actual `Grid` through call sites; they're
    // identical by construction.
    let body_grid = Grid::new(REAL_SPACE_CELL_SIZE_M, 0.0);

    let terrain_entity = commands
        .spawn((
            TerrainBundle::new(tile_atlas, &body_grid),
            MeshMaterial3d(materials.add(BodyTerrainMaterial::default())),
            RenderLayers::layer(SHIP_LAYER),
            // The scene's `SunLight` cascade is sized for the ship's local
            // neighbourhood (≤500 m); rendering a planet into it would be
            // wasteful and produce artefacts. Shadow infrastructure is
            // reworked alongside atmospheres in M4.
            NotShadowCaster,
            NotShadowReceiver,
            ChildOf(ship_parent_entity),
            Name::new(format!("{} Terrain", body.name)),
            BodyTerrain,
        ))
        .id();

    tile_trees.insert((terrain_entity, ship_camera), tile_tree);

    info!(
        "spawned ground terrain for '{}' (radius {:.0} km, height range ±{:.0} m, atlas size {})",
        body.name,
        radius_m / 1000.0,
        height_range,
        ATLAS_SIZE,
    );
}
