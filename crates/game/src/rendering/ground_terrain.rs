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
use thalos_planet_rendering::{AtmosphereBlock, SceneLighting};
use thalos_terrain::{
    BodySkyExtra, BodySkyMaterial, BodyTerrainMaterial, PipelineTileProvider,
};
use thalos_terrain_gen::PlanetSurface;

use crate::camera::ShipCamera;
use crate::coords::SHIP_LAYER;

use super::real_space::REAL_SPACE_CELL_SIZE_M;
use super::types::{CameraExposure, FrameBodyStates, RealSpaceBody, SimulationState};

/// Sun irradiance constant matching the impostor lighting system.
const LIGHT_AT_1AU: f32 = 10.0;
const AU_M: f64 = 1.496e11;

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

/// Marker on the per-body fullscreen sky-dome entity. Hidden whenever the
/// camera sits outside the body's atmosphere shell (i.e. the player is in
/// space, not on the surface), so it only contributes a fullscreen pass on
/// the body whose atmosphere currently surrounds the camera.
#[derive(Component, Debug)]
pub(super) struct BodySky {
    pub(super) body_id: BodyId,
}

/// Marker on the per-body `PlanetHaloMaterial` billboard (ship layer).
/// Mutually exclusive with [`BodySky`]: when the camera is outside the
/// atmosphere shell the halo provides the rim-glow on the impostor's
/// silhouette; when inside, the fullscreen sky pass takes over and the
/// halo would double-contribute, so it's hidden.
#[derive(Component, Debug)]
pub(crate) struct BodyHalo {
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
///
/// `atmosphere` is the static Rayleigh + Mie block for this body, expressed in
/// ship-view render units (same scale used by the impostor's `ship_atmosphere`).
/// Airless bodies pass `AtmosphereBlock::default()` (all zeros) and no
/// sky-dome entity is spawned.
pub(super) fn spawn_body_terrain(
    commands: &mut Commands,
    body: &BodyDefinition,
    surface: Arc<PlanetSurface>,
    ship_parent_entity: Entity,
    materials: &mut Assets<BodyTerrainMaterial>,
    tile_trees: &mut TerrainViewComponents<TileTree>,
    ship_camera: Entity,
    atmosphere: AtmosphereBlock,
) {
    let radius_m = body.radius_m as f32;
    let height_range = surface.static_surface.height_range;

    let model = TerrainModel::sphere(DVec3::ZERO, body.radius_m, -height_range, height_range);

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
    })
    .add_attachment(AttachmentConfig {
        name: "roughness".to_string(),
        texture_size: TILE_TEXTURE_SIZE,
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        // bevy_terrain has no single-channel 8-bit format; the source
        // cubemap is u8 and we upscale to u16 in the tile provider.
        format: AttachmentFormat::R16,
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

    // `scene` is zeroed here; `update_body_terrain_atmosphere` writes the
    // correct sun direction, flux, occluders, and ambient on the first Sync
    // tick after spawn.
    let material = BodyTerrainMaterial {
        atmosphere,
        scene: SceneLighting::default(),
    };

    let terrain_entity = commands
        .spawn((
            bundle,
            MeshMaterial3d(materials.add(material)),
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

/// Unified per-body render-LOD visibility.
///
/// One pass over each body decides which of its four ship-layer render
/// entities are active for this frame, based on a single camera-to-body
/// distance. Two orthogonal axes:
///
/// * **Surface LOD** — `dist < TERRAIN_HANDOFF_RADIUS_FACTOR × radius`
///   selects the 3-D terrain mesh; otherwise the flat impostor billboard.
///   Logically the same body at two projections; never draw both.
/// * **Atmosphere LOD** — `dist < radius + karman_line` selects the
///   `BodySky` fullscreen volumetric pass (camera is inside the shell);
///   otherwise the `BodyHalo` billboard rim-glow (camera is outside).
///   The two integrate the same scattering, just with different geometries
///   — `BodySky` covers every screen pixel and clips at scene depth, the
///   halo covers only the shell silhouette. Drawing both at once is what
///   produces the visible "band" the user reported, because their
///   integrations double on sky pixels but the halo discards on rays that
///   hit the planet sphere.
///
/// Three altitude bands fall out for a body with an atmosphere:
///
/// | distance                          | surface  | atmosphere |
/// |-----------------------------------|----------|------------|
/// | `d < radius + karman`             | terrain  | BodySky    |
/// | `radius + karman ≤ d < 4 × radius`| terrain  | halo       |
/// | `d ≥ 4 × radius`                  | impostor | halo       |
///
/// Airless bodies skip the BodySky entity entirely (never spawned) and
/// the halo's shader early-outs to a no-op; we still toggle its visibility
/// for consistency.
///
/// The map-layer impostor and map halo live on `MAP_LAYER` and are not
/// touched here.
#[allow(clippy::type_complexity)]
pub(super) fn sync_body_render_lod(
    sim: Res<SimulationState>,
    ship_cam_q: Query<&GlobalTransform, With<ShipCamera>>,
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut terrains: Query<
        (&BodyTerrain, &mut Visibility),
        (
            Without<RealSpaceImpostor>,
            Without<BodySky>,
            Without<BodyHalo>,
        ),
    >,
    mut impostors: Query<
        (&RealSpaceImpostor, &mut Visibility),
        (Without<BodyTerrain>, Without<BodySky>, Without<BodyHalo>),
    >,
    mut skies: Query<
        (&BodySky, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodyHalo>,
        ),
    >,
    mut halos: Query<
        (&BodyHalo, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodySky>,
        ),
    >,
) {
    let Ok(cam_xform) = ship_cam_q.single() else {
        return;
    };
    let cam_pos = cam_xform.translation();

    // The world-render-space position of each procedural body is its
    // `RealSpaceBody` grid origin. Index by body_id once per frame so each
    // marker loop is O(N) in its own entity count.
    let mut body_pos_by_id: std::collections::HashMap<BodyId, Vec3> =
        std::collections::HashMap::with_capacity(sim.system.bodies.len());
    for (rsb, xform) in &body_q {
        body_pos_by_id.insert(rsb.body_id, xform.translation());
    }

    // Returns (distance, swap_threshold, shell_radius) for one body, or
    // None if the body or its render-space position is missing.
    // `shell_radius == radius` for airless bodies (no atmosphere shell),
    // which makes the `inside_shell` test below false for any camera
    // outside the planet — the correct degenerate behaviour.
    let body_metrics = |body_id: BodyId| -> Option<(f32, f32, f32)> {
        let body = sim.system.bodies.get(body_id)?;
        let body_pos = body_pos_by_id.get(&body_id)?;
        let radius = body.radius_m as f32;
        let karman = body
            .terrestrial_atmosphere
            .as_ref()
            .map(|a| a.karman_line_m)
            .unwrap_or(0.0);
        let dist = (cam_pos - *body_pos).length();
        Some((dist, TERRAIN_HANDOFF_RADIUS_FACTOR * radius, radius + karman))
    };

    let set_vis = |vis: &mut Visibility, want: Visibility| {
        if *vis != want {
            *vis = want;
        }
    };

    for (terrain, mut vis) in &mut terrains {
        let Some((dist, swap, _shell)) = body_metrics(terrain.body_id) else {
            continue;
        };
        let want = if dist < swap {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
    }

    for (impostor, mut vis) in &mut impostors {
        let Some((dist, swap, _shell)) = body_metrics(impostor.body_id) else {
            continue;
        };
        let want = if dist < swap {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        set_vis(&mut vis, want);
    }

    for (sky, mut vis) in &mut skies {
        let Some((dist, _swap, shell)) = body_metrics(sky.body_id) else {
            continue;
        };
        let want = if dist < shell {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
    }

    for (halo, mut vis) in &mut halos {
        let Some((dist, _swap, shell)) = body_metrics(halo.body_id) else {
            continue;
        };
        let want = if dist >= shell {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
    }
}

/// Update the per-frame dynamic data on every body's `BodyTerrainMaterial`
/// and `BodySkyMaterial`:
///
/// - terrain `planet_extra` (planet center + radius, terminator-wrap knob)
/// - terrain `scene` (`SceneLighting`: primary star, eclipse occluders,
///   ambient; planetshine left zero for now)
/// - sky `atmosphere_extra` (sun dir + flux, planet center + radius)
///
/// Must run after `cache_body_states` (for sun direction from ephemeris),
/// `update_camera_exposure` (for exposure gain), and
/// `update_real_space_body_positions` (for up-to-date body grid transforms).
pub(super) fn update_body_terrain_atmosphere(
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    terrain_q: Query<(&BodyTerrain, &MeshMaterial3d<BodyTerrainMaterial>)>,
    sky_q: Query<(&BodySky, &MeshMaterial3d<BodySkyMaterial>)>,
    sim: Res<SimulationState>,
    cache: Res<FrameBodyStates>,
    exposure: Res<CameraExposure>,
    mut terrain_materials: ResMut<Assets<BodyTerrainMaterial>>,
    mut sky_materials: ResMut<Assets<BodySkyMaterial>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    let star_pos = states.first().map(|s| s.position).unwrap_or_default();

    // Planet center in render space from each body's grid transform.
    let mut body_render_pos: std::collections::HashMap<BodyId, Vec3> =
        std::collections::HashMap::with_capacity(sim.system.bodies.len());
    for (rsb, xform) in &body_q {
        body_render_pos.insert(rsb.body_id, xform.translation());
    }

    // Eclipse occluders at SHIP_SCALE. The terrain lives in the BigSpace
    // SHIP_LAYER where 1 render unit = 1 m, so we can use the body grid
    // translations directly without an extra origin/scale transform.
    let mut occluders: Vec<(BodyId, Vec3, f32)> =
        Vec::with_capacity(sim.system.bodies.len());
    for (i, body) in sim.system.bodies.iter().enumerate() {
        if matches!(body.kind, thalos_physics::types::BodyKind::Star) || body.radius_m < 1.0 {
            continue;
        }
        let Some(pos) = body_render_pos.get(&i) else {
            continue;
        };
        occluders.push((i, *pos, body.radius_m as f32));
    }

    // Per-body sky data: sun direction, flux, planet center, radius.
    let mut sky_by_body: std::collections::HashMap<BodyId, BodySkyExtra> =
        std::collections::HashMap::with_capacity(sim.system.bodies.len());
    for (i, body) in sim.system.bodies.iter().enumerate() {
        let Some(body_state) = states.get(i) else {
            continue;
        };
        let Some(render_pos) = body_render_pos.get(&i) else {
            continue;
        };
        let offset = star_pos - body_state.position;
        let dist = offset.length();
        let sun_dir = if dist > 0.0 {
            (offset / dist).as_vec3()
        } else {
            Vec3::Y
        };
        let au_over_d = (AU_M / dist.max(1.0)) as f32;
        let flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;
        let planet_radius = body.radius_m as f32;
        sky_by_body.insert(
            i,
            BodySkyExtra {
                sun_dir_flux: Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, flux),
                planet_center_radius: Vec4::new(
                    render_pos.x,
                    render_pos.y,
                    render_pos.z,
                    planet_radius,
                ),
            },
        );
    }

    for (terrain, mat_handle) in &terrain_q {
        let Some(mat) = terrain_materials.get_mut(mat_handle) else {
            continue;
        };
        mat.scene = build_terrain_scene_lighting(
            terrain.body_id,
            states,
            &occluders,
            exposure.gain,
        );
    }

    for (sky, mat_handle) in &sky_q {
        let Some(extra) = sky_by_body.get(&sky.body_id) else {
            continue;
        };
        let Some(mat) = sky_materials.get_mut(mat_handle) else {
            continue;
        };
        mat.atmosphere_extra = *extra;
    }
}

/// Build a `SceneLighting` for one terrain body. Equivalent to
/// `build_scene_lighting` in `rendering::lighting`, but specialised so the
/// occluder vec is keyed by `BodyId` directly and uses the SHIP-frame
/// (1 m = 1 render unit) body grid positions cached above.
fn build_terrain_scene_lighting(
    body_id: BodyId,
    states: &thalos_physics::types::BodyStates,
    occluders: &[(BodyId, Vec3, f32)],
    gain: f32,
) -> thalos_planet_rendering::SceneLighting {
    use thalos_planet_rendering::{
        MAX_ECLIPSE_OCCLUDERS, SceneLighting, StarLight,
    };

    let mut scene = SceneLighting::default();
    let star_pos = states.first().map(|s| s.position).unwrap_or_default();
    let body_pos = states.get(body_id).map(|s| s.position).unwrap_or_default();
    let offset = star_pos - body_pos;
    let distance_m = offset.length();
    let to_star = if distance_m > 0.0 {
        (offset / distance_m).as_vec3()
    } else {
        Vec3::Y
    };
    let au_over_d = (AU_M / distance_m.max(1.0)) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * gain;

    scene.star_count = 1;
    scene.stars[0] = StarLight {
        dir_flux: Vec4::new(to_star.x, to_star.y, to_star.z, flux),
        color: Vec4::new(1.0, 1.0, 1.0, 0.0),
    };

    let mut count = 0usize;
    for (other_id, pos, radius) in occluders {
        if *other_id == body_id {
            continue;
        }
        if count >= MAX_ECLIPSE_OCCLUDERS {
            break;
        }
        scene.occluders[count] = Vec4::new(pos.x, pos.y, pos.z, *radius);
        count += 1;
    }
    scene.occluder_count = count as u32;

    // Planetshine for moons could fill `scene.planetshine_*` here using the
    // parent's render-space center, radius, and the
    // `PlanetshineTints` resource. Skipped at this stage because the LOD
    // swap happens far enough from the body that planetshine contributes
    // little to the direct-light path; revisit when we drop the swap radius.
    scene
}
