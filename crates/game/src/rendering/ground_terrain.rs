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
use bevy::light::NotShadowCaster;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy_terrain::prelude::*;
use big_space::grid::Grid;
use thalos_physics::types::{BodyDefinition, BodyId};
use thalos_planet_rendering::{AtmosphereBlock, SceneLighting};
use thalos_terrain::{
    BodySkyExtra, BodySkyMaterial, BodyTerrainMaterial, BodyTerrainShadow, BodyWaterMaterial,
    BodyWaterParams, PipelineTileProvider, rendered_height_range,
};
use thalos_terrain_gen::{DynamicSurfaceState, PlanetSurface};

use crate::camera::{CameraTargetOffset, ShipCamera};
use crate::coords::SHIP_LAYER;

use super::real_space::REAL_SPACE_CELL_SIZE_M;
use super::types::{CameraExposure, PlayerShip, RealSpaceBody, SimulationState, SolarSystemState};

/// Sun irradiance constant matching the impostor lighting system.
const LIGHT_AT_1AU: f32 = 10.0;
const AU_M: f64 = 1.496e11;
const CRAFT_SHADOW_RADIUS_M: f32 = 2.5;
const CRAFT_SHADOW_MIN_HALF_LENGTH_M: f32 = 4.0;
const CRAFT_SHADOW_STRENGTH: f32 = 0.88;
const CRAFT_SHADOW_PENUMBRA_M: f32 = 0.75;
const CRAFT_SHADOW_MAX_DISTANCE_M: f32 = 25_000.0;

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

/// Marker on the per-body fullscreen sky-dome entity. Visible while the
/// body's real terrain LOD is active, so the ground path gets the same
/// atmosphere/cloud layer in front of the surface that the impostor path
/// composites inline.
#[derive(Component, Debug)]
pub(super) struct BodySky {
    pub(super) body_id: BodyId,
}

/// Marker on the per-body `PlanetHaloMaterial` billboard (ship layer).
/// Mutually exclusive with [`BodySky`]: when the impostor is active the halo
/// provides the rim-glow outside the billboard's solid sphere; while terrain
/// is active, the fullscreen sky pass covers rim, haze, and cloud overlay.
#[derive(Component, Debug)]
pub(crate) struct BodyHalo {
    pub body_id: BodyId,
}

/// Marker on the per-body water sphere (ship layer). Only spawned when the
/// baked surface has `sea_level_m = Some(_)`; visibility is paired with
/// [`BodyTerrain`] so the impostor's inline water BRDF takes over outside
/// the LOD swap radius.
#[derive(Component, Debug)]
pub(super) struct BodyWater {
    pub(super) body_id: BodyId,
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
    dynamic_state: DynamicSurfaceState,
) {
    let radius_m = body.radius_m as f32;
    let height_range = rendered_height_range(&surface, &dynamic_state);

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

    let provider =
        PipelineTileProvider::new(body.name.clone(), surface, dynamic_state, height_range);
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
        craft_shadow: BodyTerrainShadow::default(),
    };

    let terrain_entity = commands
        .spawn((
            bundle,
            MeshMaterial3d(materials.add(material)),
            RenderLayers::layer(SHIP_LAYER),
            // The scene's `SunLight` cascade is sized for standard local
            // meshes. Ground terrain receives craft shadows through an
            // analytic proxy; rendering the planet itself into the shadow map
            // would be wasteful and noisy.
            NotShadowCaster,
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

/// Icosphere subdivision for the water surface mesh.
///
/// 6 levels = 81,920 triangles. At a typical low-altitude view (~10 km up
/// over a Mira-scale body) that puts ~30 triangles across the visible
/// horizon disc — fine for wave-normal-dominated water shading. Bump to
/// 7 if mesh facets start reading at the silhouette.
const WATER_MESH_SUBDIVISIONS: u32 = 6;

/// Tiny offset (in metres) lifting the water mesh above the bare body
/// sphere. Resolves z-fighting between the water mesh and the seafloor
/// terrain mesh at iso-height texels in favour of water. Sized at 2 m so
/// it's comfortably above the f32 ULP near body-radius scale (~0.4 m at
/// 3 Mm) on the bodies we currently ship.
const WATER_SURFACE_EPSILON_M: f32 = 2.0;

/// Default deep-water tint when the bake omits an explicit
/// `WaterAppearance`. Matches `PlanetWaterParams::from_static_surface`'s
/// fallback so the impostor and ground-LOD paths agree.
const FALLBACK_WATER_COLOR_DEPTH: [f32; 4] = [0.012, 0.040, 0.090, 120.0];

/// Spawn the per-body water sphere. No-op when the baked surface has no
/// `sea_level_m` (airless bodies); otherwise creates a single icosphere
/// entity parented to the body's grid, hidden at start so
/// `sync_body_render_lod` can flip it in step with [`BodyTerrain`].
pub(super) fn spawn_body_water(
    commands: &mut Commands,
    body: &BodyDefinition,
    surface: &PlanetSurface,
    ship_parent_entity: Entity,
    meshes: &mut Assets<Mesh>,
    water_materials: &mut Assets<BodyWaterMaterial>,
) {
    let baked = &surface.static_surface;
    let Some(sea_level_m) = baked.sea_level_m else {
        return;
    };

    let water_radius_m = (body.radius_m as f32 + sea_level_m + WATER_SURFACE_EPSILON_M).max(1.0);

    let color_depth = baked
        .water_appearance
        .map(|w| {
            Vec4::new(
                w.color_depth[0],
                w.color_depth[1],
                w.color_depth[2],
                w.color_depth[3],
            )
        })
        .unwrap_or_else(|| {
            Vec4::new(
                FALLBACK_WATER_COLOR_DEPTH[0],
                FALLBACK_WATER_COLOR_DEPTH[1],
                FALLBACK_WATER_COLOR_DEPTH[2],
                FALLBACK_WATER_COLOR_DEPTH[3],
            )
        });

    let mesh = meshes.add(
        Sphere::new(water_radius_m)
            .mesh()
            .ico(WATER_MESH_SUBDIVISIONS)
            .expect("water mesh ico subdivision is within bevy's supported range"),
    );

    let params = BodyWaterParams {
        color_depth,
        // xyz populated each frame by `update_body_terrain_atmosphere` from
        // the body's render-space grid origin; w is constant.
        planet_center_radius: Vec4::new(0.0, 0.0, 0.0, water_radius_m),
        time: Vec4::ZERO,
    };
    let material = BodyWaterMaterial {
        scene: SceneLighting::default(),
        params,
    };

    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(water_materials.add(material)),
        Transform::default(),
        Visibility::Hidden,
        RenderLayers::layer(SHIP_LAYER),
        NotShadowCaster,
        ChildOf(ship_parent_entity),
        Name::new(format!("{} Water", body.name)),
        BodyWater { body_id: body.id },
    ));

    info!(
        "spawned water surface for '{}' (radius {:.0} km, sea level {:+.1} m)",
        body.name,
        body.radius_m / 1000.0,
        sea_level_m,
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
/// * **Atmosphere LOD** — terrain uses `BodySky`, impostor uses `BodyHalo`.
///   `BodySky` covers every screen pixel and clips at scene depth, which is
///   what gives ground LOD terrain aerial perspective and the detached cloud
///   shell. `BodyHalo` covers only the billboard shell silhouette, matching
///   the impostor body's inline atmosphere/cloud composite.
///
/// Three altitude bands fall out for a body with an atmosphere:
///
/// | distance           | surface  | atmosphere |
/// |--------------------|----------|------------|
/// | `d < 4 × radius`   | terrain  | BodySky    |
/// | `d ≥ 4 × radius`   | impostor | halo       |
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
            Without<BodyWater>,
        ),
    >,
    mut impostors: Query<
        (&RealSpaceImpostor, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<BodySky>,
            Without<BodyHalo>,
            Without<BodyWater>,
        ),
    >,
    mut waters: Query<
        (&BodyWater, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodySky>,
            Without<BodyHalo>,
        ),
    >,
    mut skies: Query<
        (&BodySky, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodyHalo>,
            Without<BodyWater>,
        ),
    >,
    mut halos: Query<
        (&BodyHalo, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodySky>,
            Without<BodyWater>,
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
    // None if the body or its render-space position is missing. The shell
    // radius is kept here for diagnostics / future atmosphere-specific LOD
    // tuning; today's visibility choice keys atmosphere to the surface LOD
    // so terrain always gets the fullscreen in-front pass.
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
        Some((
            dist,
            TERRAIN_HANDOFF_RADIUS_FACTOR * radius,
            radius + karman,
        ))
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

    // Water pairs with terrain: only visible inside the LOD swap radius.
    // Outside it, the impostor's inline water BRDF takes over.
    for (water, mut vis) in &mut waters {
        let Some((dist, swap, _shell)) = body_metrics(water.body_id) else {
            continue;
        };
        let want = if dist < swap {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
    }

    for (sky, mut vis) in &mut skies {
        let Some((dist, swap, _shell)) = body_metrics(sky.body_id) else {
            continue;
        };
        let want = if dist < swap {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
    }

    for (halo, mut vis) in &mut halos {
        let Some((dist, swap, _shell)) = body_metrics(halo.body_id) else {
            continue;
        };
        let want = if dist >= swap {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
    }
}

/// Update the per-frame dynamic data on every body's `BodyTerrainMaterial`,
/// `BodyWaterMaterial`, and `BodySkyMaterial`:
///
/// - terrain `planet_extra` (planet center + radius, terminator-wrap knob)
/// - terrain `scene` (`SceneLighting`: primary star, eclipse occluders,
///   ambient; planetshine left zero for now)
/// - water `scene` (same `SceneLighting` as terrain), water `params.time`
///   (Time::elapsed_secs for wave scroll), water `params.planet_center_radius.xyz`
///   (body render-space centre)
/// - sky `atmosphere_extra` (sun dir + flux, planet center + radius)
///
/// Must run after `sync_solar_system_state` (for sun direction from ephemeris),
/// `update_camera_exposure` (for exposure gain), and
/// `update_real_space_body_positions` (for up-to-date body grid transforms).
pub(super) fn update_body_terrain_atmosphere(
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    terrain_q: Query<(&BodyTerrain, &MeshMaterial3d<BodyTerrainMaterial>)>,
    water_q: Query<(&BodyWater, &MeshMaterial3d<BodyWaterMaterial>)>,
    sky_q: Query<(&BodySky, &MeshMaterial3d<BodySkyMaterial>)>,
    ship_q: Query<(&GlobalTransform, Option<&CameraTargetOffset>), With<PlayerShip>>,
    sim: Res<SimulationState>,
    cache: Res<SolarSystemState>,
    exposure: Res<CameraExposure>,
    time: Res<Time>,
    mut terrain_materials: ResMut<Assets<BodyTerrainMaterial>>,
    mut water_materials: ResMut<Assets<BodyWaterMaterial>>,
    mut sky_materials: ResMut<Assets<BodySkyMaterial>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let craft_shadow = craft_shadow_from_player_ship(&ship_q);

    let star_pos = states.first().map(|s| s.position).unwrap_or_default();

    // Planet center and orientation in render space from each body's grid
    // transform. The real-space grid rotation is body-local → world, so the
    // cloud cubemap sampler wants its inverse.
    let mut body_render_pos: std::collections::HashMap<BodyId, Vec3> =
        std::collections::HashMap::with_capacity(sim.system.bodies.len());
    let mut world_to_body_orientation: std::collections::HashMap<BodyId, Quat> =
        std::collections::HashMap::with_capacity(sim.system.bodies.len());
    for (rsb, xform) in &body_q {
        body_render_pos.insert(rsb.body_id, xform.translation());
        world_to_body_orientation.insert(
            rsb.body_id,
            xform.compute_transform().rotation.inverse().normalize(),
        );
    }

    // Eclipse occluders at SHIP_SCALE. The terrain lives in the BigSpace
    // SHIP_LAYER where 1 render unit = 1 m, so we can use the body grid
    // translations directly without an extra origin/scale transform.
    let mut occluders: Vec<(BodyId, Vec3, f32)> = Vec::with_capacity(sim.system.bodies.len());
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
        let q = world_to_body_orientation
            .get(&i)
            .copied()
            .unwrap_or(Quat::IDENTITY);
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
                world_to_body_orientation: Vec4::new(q.x, q.y, q.z, q.w),
            },
        );
    }

    for (terrain, mat_handle) in &terrain_q {
        let Some(mat) = terrain_materials.get_mut(mat_handle) else {
            continue;
        };
        mat.scene =
            build_terrain_scene_lighting(terrain.body_id, states, &occluders, exposure.gain);
        mat.craft_shadow = craft_shadow;
    }

    // Use real (wall-clock) elapsed seconds for wave scroll — the simulation
    // clock pauses under warp and ticks faster than wall time during it, both
    // of which would make wave motion read as wrong against the visible
    // motion of the body itself.
    let wave_time = time.elapsed_secs();
    for (water, mat_handle) in &water_q {
        let Some(mat) = water_materials.get_mut(mat_handle) else {
            continue;
        };
        mat.scene = build_terrain_scene_lighting(water.body_id, states, &occluders, exposure.gain);
        let render_pos = body_render_pos
            .get(&water.body_id)
            .copied()
            .unwrap_or(Vec3::ZERO);
        // Preserve the spawn-time `.w` (water surface radius) and only
        // overwrite the centre + time fields.
        let water_radius = mat.params.planet_center_radius.w;
        mat.params.planet_center_radius =
            Vec4::new(render_pos.x, render_pos.y, render_pos.z, water_radius);
        mat.params.time.w = wave_time;
    }

    for (sky, mat_handle) in &sky_q {
        let Some(extra) = sky_by_body.get(&sky.body_id) else {
            continue;
        };
        let Some(mat) = sky_materials.get_mut(mat_handle) else {
            continue;
        };
        mat.atmosphere_extra = *extra;
        if let Some(clouds) = cache
            .environment
            .get(sky.body_id)
            .and_then(|env| env.cloud_bands.as_ref())
        {
            let p = clouds.phases;
            mat.atmosphere.cloud_bands_a =
                Vec4::new(p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32);
            mat.atmosphere.cloud_bands_b =
                Vec4::new(p[4] as f32, p[5] as f32, p[6] as f32, p[7] as f32);
            mat.atmosphere.cloud_bands_c =
                Vec4::new(p[8] as f32, p[9] as f32, p[10] as f32, p[11] as f32);
            mat.atmosphere.cloud_bands_d =
                Vec4::new(p[12] as f32, p[13] as f32, p[14] as f32, p[15] as f32);

            let scroll = clouds.scroll_rate_rad_s.abs();
            let period = if scroll > 1.0e-9 {
                std::f64::consts::TAU / scroll
            } else {
                86_400.0
            };
            mat.atmosphere.cloud_dynamics.y = sim.simulation.sim_time().rem_euclid(period) as f32;
        }
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
    use thalos_planet_rendering::{MAX_ECLIPSE_OCCLUDERS, SceneLighting, StarLight};

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

fn craft_shadow_from_player_ship(
    ship_q: &Query<(&GlobalTransform, Option<&CameraTargetOffset>), With<PlayerShip>>,
) -> BodyTerrainShadow {
    let Ok((ship_xform, offset)) = ship_q.single() else {
        return BodyTerrainShadow::default();
    };

    let ship_transform = ship_xform.compute_transform();
    let local_center = offset.map(|offset| offset.0).unwrap_or(Vec3::ZERO);
    let center = ship_transform.translation + ship_transform.rotation * local_center;
    let mut axis = ship_transform.rotation * Vec3::Y;
    if axis.length_squared() <= 1.0e-6 {
        axis = Vec3::Y;
    } else {
        axis = axis.normalize();
    }

    BodyTerrainShadow {
        caster_pos_radius: Vec4::new(center.x, center.y, center.z, CRAFT_SHADOW_RADIUS_M),
        caster_axis_half_len: Vec4::new(
            axis.x,
            axis.y,
            axis.z,
            (local_center.y.abs() * 1.35).max(CRAFT_SHADOW_MIN_HALF_LENGTH_M),
        ),
        params: Vec4::new(
            CRAFT_SHADOW_STRENGTH,
            CRAFT_SHADOW_PENUMBRA_M,
            CRAFT_SHADOW_MAX_DISTANCE_M,
            1.0,
        ),
    }
}
