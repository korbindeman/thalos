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
use thalos_physics::types::{BodyDefinition, BodyId};
use thalos_terrain::{BodyTerrainMaterial, PipelineTileProvider};
use thalos_terrain_gen::PlanetSurface;

use crate::camera::ShipCamera;
use crate::coords::SHIP_LAYER;

use super::real_space::REAL_SPACE_CELL_SIZE_M;
use super::types::{RealSpaceBody, SimulationState};

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

/// Marker on the ground-LOD terrain entity spawned for a procedural body.
/// Carries the body id so the impostor↔terrain LOD swap can pair it with the
/// matching impostor without walking parents.
#[derive(Component, Debug)]
pub(super) struct BodyTerrain {
    pub(super) body_id: BodyId,
}

/// Marker on the ship-layer impostor billboard for a procedural body. Set in
/// `finalize_planet_generation` when the placeholder mesh is replaced with
/// the `PlanetMaterial` billboard, so the swap system can find both halves
/// of the LOD pair via component queries instead of parent lookups.
#[derive(Component, Debug)]
pub(crate) struct RealSpaceImpostor {
    pub body_id: BodyId,
}

/// Camera-to-body-centre distance, expressed as a multiple of body radius,
/// at which the impostor billboard hands off to the ground-LOD terrain.
///
/// 4× radius keeps the impostor active while the body covers less than ~28°
/// of the view (its silhouette is well-defined as a disc); below 4× the body
/// fills enough of the frame that the 3-D mesh + heightfield reads better.
///
/// Both halves of the swap are still using the same baked cubemap data, so
/// at the threshold the visible texture matches; the discontinuity is in
/// projection (flat billboard ↔ 3-D mesh) and lighting (impostor PBR ↔ flat
/// albedo until M4 wires lighting through the terrain shader).
///
/// Smooth crossfade requires opacity uniforms on `PlanetMaterial` and
/// `BodyTerrainMaterial`; deferred to M4 along with the rest of the
/// terrain-side PBR work.
const TERRAIN_HANDOFF_RADIUS_FACTOR: f32 = 4.0;

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

    let provider = PipelineTileProvider::new(body.name.clone(), surface);
    let tile_atlas = TileAtlas::with_provider(&config, Box::new(provider));
    let view_config = TerrainViewConfig::default();
    let tile_tree = TileTree::new(&tile_atlas, &view_config);

    // The body's `real_body_entity` (`ship_parent_entity`) carries a
    // `Grid::new(REAL_SPACE_CELL_SIZE_M, 0.0)`. Reconstructing it here is
    // cheaper than threading the actual `Grid` through call sites; they're
    // identical by construction.
    let body_grid = Grid::new(REAL_SPACE_CELL_SIZE_M, 0.0);

    // Start hidden so newly-spawned terrains don't pop into view at a
    // wildly wrong LOD before the swap system has a chance to run (it runs
    // in `SimStage::Sync`, the same tick as the spawn but strictly after
    // `finalize_planet_generation`). The swap system flips visibility
    // on/off based on camera distance from this frame onward.
    let mut bundle = TerrainBundle::new(tile_atlas, &body_grid);
    bundle.visibility = Visibility::Hidden;

    let terrain_entity = commands
        .spawn((
            bundle,
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
            BodyTerrain { body_id: body.id },
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

/// Hard-switch the ship-layer impostor billboard and the ground-LOD terrain
/// entity based on ship-camera distance to each procedural body.
///
/// The pair is logically one body at two LOD levels and the renderer should
/// only ever draw one half. Without this system both would render
/// simultaneously: wasteful (double-draw + tile streaming for invisible
/// terrain on distant bodies) and visually wrong (impostor PBR shading
/// fights with the terrain's flat albedo at the same screen pixels).
///
/// Threshold logic: distance < `TERRAIN_HANDOFF_RADIUS_FACTOR × radius` →
/// terrain visible, impostor hidden. Otherwise the reverse. The map-layer
/// impostor lives on `MAP_LAYER` and is not touched here.
pub(super) fn sync_terrain_impostor_swap(
    sim: Res<SimulationState>,
    ship_cam_q: Query<&GlobalTransform, With<ShipCamera>>,
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut terrains: Query<(&BodyTerrain, &mut Visibility), Without<RealSpaceImpostor>>,
    mut impostors: Query<(&RealSpaceImpostor, &mut Visibility), Without<BodyTerrain>>,
) {
    let Ok(cam_xform) = ship_cam_q.single() else {
        return;
    };
    let cam_pos = cam_xform.translation();

    // The world-render-space position of each procedural body is its
    // `RealSpaceBody` grid origin. Indexing the grids by body_id once per
    // frame keeps the inner loops O(N_bodies + N_terrains + N_impostors).
    let mut body_pos_by_id: std::collections::HashMap<BodyId, Vec3> =
        std::collections::HashMap::with_capacity(sim.system.bodies.len());
    for (rsb, xform) in &body_q {
        body_pos_by_id.insert(rsb.body_id, xform.translation());
    }

    let threshold_for = |body_id: BodyId| -> Option<f32> {
        let body = sim.system.bodies.get(body_id)?;
        Some(TERRAIN_HANDOFF_RADIUS_FACTOR * body.radius_m as f32)
    };

    for (terrain, mut vis) in &mut terrains {
        let Some(body_pos) = body_pos_by_id.get(&terrain.body_id) else {
            continue;
        };
        let Some(threshold) = threshold_for(terrain.body_id) else {
            continue;
        };
        let close = (cam_pos - *body_pos).length() < threshold;
        let want = if close {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
    }

    for (impostor, mut vis) in &mut impostors {
        let Some(body_pos) = body_pos_by_id.get(&impostor.body_id) else {
            continue;
        };
        let Some(threshold) = threshold_for(impostor.body_id) else {
            continue;
        };
        let close = (cam_pos - *body_pos).length() < threshold;
        let want = if close {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
        if *vis != want {
            *vis = want;
        }
    }
}
