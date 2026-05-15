//! Installs a loaded [`PlanetSurface`] into the scene.
//!
//! Procedural bodies' bakes (hundreds of MB compressed) are loaded off the
//! main thread by an async task dispatched in [`super::spawn::spawn_bodies`].
//! [`poll_planet_install_tasks`] runs every frame, drains tasks that have
//! finished their disk I/O + decompress + [`prepare_planet_bake`] CPU work,
//! and calls [`install_baked_planet`] to run the main-thread half
//! (`upload_prepared_bake`, impostor + halo + ground-terrain spawn). Each
//! completed install ticks the [`LoadingProgress`] counter that drives the
//! startup loading-screen overlay.
//!
//! [`patch_reference_cloud_covers`] remains a per-frame system because the
//! reference-cloud cubemap loads asynchronously from disk (image decoding,
//! not terrain).

use std::sync::Arc;

use bevy::camera::visibility::NoFrustumCulling;
use bevy::ecs::system::SystemParam;
use bevy::image::Image;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use bevy::tasks::{Task, block_on, poll_once};
use bevy_terrain::prelude::{TerrainViewComponents, TileTree};
use thalos_local_physics::TerrainSurfaceRegistry;
use thalos_physics::types::{BodyDefinition, BodyKind};
use thalos_planet_rendering::{
    AtmosphereBlock, PlanetCoastlineParams, PlanetDetailParams, PlanetHaloMaterial, PlanetMaterial,
    PlanetParams, PlanetWaterParams, PreparedPlanetBake, ReferenceClouds,
    cloud_cover_image_for_body, upload_prepared_bake,
};
use thalos_terrain::{BodySkyMaterial, BodyTerrainMaterial, BodyWaterMaterial};
use thalos_terrain_gen::{DynamicSurfaceState, PlanetSurface};

use crate::loading::LoadingProgress;

use super::ground_terrain::{
    BodyHalo, BodySky, RealSpaceImpostor, spawn_body_terrain, spawn_body_water,
};
use super::types::{
    BodyMesh, CelestialBody, PlanetMaterials, PlanetshineTints, SharedPlanetMeshes, ShipBodyMesh,
};
use crate::camera::ShipCamera;
use crate::coords::{MAP_LAYER, MAP_SCALE, SHIP_LAYER, SHIP_SCALE};
use crate::solar_system_state::{CloudBandEnvironmentState, SolarSystemState};

/// Bundles the material-asset registries [`install_baked_planet`]
/// writes into. Kept as a `SystemParam` so `spawn_bodies` can pass these
/// three handles through one argument and stay under Bevy 0.18's 16-param
/// limit.
#[derive(SystemParam)]
pub(super) struct PlanetMaterialAssets<'w> {
    pub(super) planet: ResMut<'w, Assets<PlanetMaterial>>,
    pub(super) planet_halo: ResMut<'w, Assets<PlanetHaloMaterial>>,
    pub(super) body_terrain: ResMut<'w, Assets<BodyTerrainMaterial>>,
    pub(super) body_water: ResMut<'w, Assets<BodyWaterMaterial>>,
}

/// Storage / index / registry handles `spawn_bodies` needs when installing
/// each procedural body's bake. Same 16-param-limit motivation as
/// [`PlanetMaterialAssets`].
#[derive(SystemParam)]
pub(super) struct ProceduralInstallExtras<'w, 's> {
    pub(super) storage_buffers: ResMut<'w, Assets<ShaderStorageBuffer>>,
    pub(super) tile_trees: ResMut<'w, TerrainViewComponents<TileTree>>,
    pub(super) terrain_surfaces: ResMut<'w, TerrainSurfaceRegistry>,
    pub(super) terrain_registry: Res<'w, crate::GameTerrainRegistry>,
    pub(super) ship_camera_q: Query<'w, 's, Entity, With<ShipCamera>>,
}

/// World-level state the install path writes into. Bundled to stay under
/// the 16-param limit.
#[derive(SystemParam)]
pub(super) struct WorldStateAssets<'w> {
    pub(super) solar_system: ResMut<'w, SolarSystemState>,
    pub(super) planetshine: ResMut<'w, PlanetshineTints>,
}

/// Map body kind + size to a surface roughness value (0 = smooth, 1 = very rough).
///
/// This drives the terminator wrap in the planet impostor shader.
/// On a smooth sphere (no normal map), wrap simulates *unresolved* scattering
/// that softens the macro terminator — primarily atmospheric scattering, not
/// surface craters.  Crater roughness creates a *textured* terminator boundary
/// (individual shadow/lit patches), which only makes sense once normal maps
/// provide that detail.
///
/// Terminator wrap factor (shader `light_dir.w`). 0 = razor-sharp Lambert
/// terminator (airless vacuum look); nonzero softens the edge to fake
/// unresolved sub-pixel roughness on atmospheric bodies.
fn body_surface_roughness(body: &BodyDefinition) -> f32 {
    match body.kind {
        BodyKind::Star => 0.0,
        BodyKind::Planet => 0.0,
        BodyKind::Moon => 0.0,
        BodyKind::DwarfPlanet => 0.0,
        BodyKind::Centaur => 0.0,
        BodyKind::Comet => 0.0,
    }
}

/// All the asset registries `install_baked_planet` needs to write into.
/// Bundled so the caller's signature stays manageable. Carries the
/// underlying `&mut Assets<_>` refs rather than a `PlanetMaterialAssets`
/// SystemParam, so the struct stays free of borrow-checker lifetime
/// gymnastics.
pub(super) struct InstallAssets<'a> {
    pub planet_materials: &'a mut Assets<PlanetMaterial>,
    pub planet_halo_materials: &'a mut Assets<PlanetHaloMaterial>,
    pub body_terrain_materials: &'a mut Assets<BodyTerrainMaterial>,
    pub body_water_materials: &'a mut Assets<BodyWaterMaterial>,
    pub meshes: &'a mut Assets<Mesh>,
    pub images: &'a mut Assets<Image>,
    pub storage_buffers: &'a mut Assets<ShaderStorageBuffer>,
    pub tile_trees: &'a mut TerrainViewComponents<TileTree>,
    pub terrain_surfaces: &'a mut TerrainSurfaceRegistry,
    pub planetshine: &'a mut PlanetshineTints,
    pub solar_system: &'a mut SolarSystemState,
}

/// Per-body entity handles that `spawn_bodies` already created before the
/// install call. The function reads them, spawns billboards and halos under
/// them, and inserts material components on `body_entity`.
pub(super) struct InstallEntities {
    pub body_entity: Entity,
    pub ship_parent_entity: Entity,
    pub ship_camera: Entity,
}

/// Install a freshly-loaded [`PlanetSurface`] into the scene.
///
/// Uploads cubemaps + SSBOs, builds map-layer and ship-layer
/// `PlanetMaterial`s and halo materials, spawns the impostor billboards
/// under `body_entity` / `ship_parent_entity`, attaches a
/// `PlanetMaterials` component, and seeds the ground-LOD terrain entity
/// via [`spawn_body_terrain`].
///
/// `surface`, `prepared`, and `dynamic_state` are produced off the main
/// thread by the task dispatched in `spawn_bodies`; this function is the
/// main-thread half that does GPU upload + entity spawn.
pub(super) fn install_baked_planet(
    commands: &mut Commands,
    body: &BodyDefinition,
    body_id: usize,
    render_radius: f32,
    surface: PlanetSurface,
    prepared: PreparedPlanetBake,
    dynamic_state: DynamicSurfaceState,
    shared: &SharedPlanetMeshes,
    reference_clouds: &ReferenceClouds,
    terrain_registry: &crate::GameTerrainRegistry,
    entities: InstallEntities,
    assets: InstallAssets<'_>,
) {
    let _span = tracing::info_span!("install_baked_planet", body = %body.name).entered();

    // Wrap in Arc so the same `PlanetSurface` can back both the impostor
    // (which only borrows) and the ground-LOD `PipelineTileProvider`
    // (which holds it across async tile requests). `StaticSurfaceData` is
    // not `Clone`, so this is the cheap path. Dynamic-layer runtime state
    // is installed into `SolarSystemState`, the canonical per-body
    // environment source for every projection.
    let surface = Arc::new(surface);
    let baked = &surface.static_surface;

    assets
        .solar_system
        .install_dynamic_surface_state(body_id, dynamic_state.clone());
    let cloud_cover_config = body
        .terrestrial_atmosphere
        .as_ref()
        .and_then(|a| a.clouds.as_ref());
    if let Some(clouds) = cloud_cover_config {
        assets.solar_system.install_cloud_band_state(
            body_id,
            CloudBandEnvironmentState::new(
                clouds.scroll_rate as f64,
                clouds.differential_rotation as f64,
            ),
        );
    }
    assets.terrain_surfaces.insert(body_id, surface.clone());
    // Mirror into the propagator's terrain registry so prediction and
    // live propagation collide against the same surface. Both registries
    // hold `Arc<PlanetSurface>` clones of the same data — `local_physics`
    // builds colliders from it, `thalos_physics` queries the height
    // cubemap for collision detection.
    terrain_registry.0.insert(body_id, surface.clone());
    let detail =
        PlanetDetailParams::from_body(&baked.detail_params, baked.cubemap_bake_threshold_m);
    let height_range = baked.height_range;
    assets
        .planetshine
        .by_body
        .insert(body_id, baked.mean_albedo);
    let textures = upload_prepared_bake(prepared, assets.images, assets.storage_buffers);

    let roughness = body_surface_roughness(body);
    // Two atmosphere blocks: scale-dependent fields (`rim_shape.x`,
    // `rim_shape.y`) are expressed in render units, so they differ
    // between MAP_SCALE and SHIP_SCALE instances.
    let map_atmosphere = body
        .terrestrial_atmosphere
        .as_ref()
        .map(|a| AtmosphereBlock::from_terrestrial(a, (1.0 / MAP_SCALE) as f32))
        .unwrap_or_default();
    let ship_atmosphere = body
        .terrestrial_atmosphere
        .as_ref()
        .map(|a| AtmosphereBlock::from_terrestrial(a, (1.0 / SHIP_SCALE) as f32))
        .unwrap_or_default();

    let (cloud_cover, uses_reference_cloud) =
        cloud_cover_image_for_body(&body.name, reference_clouds, assets.images);

    let coastline = PlanetCoastlineParams::from_static_surface(baked);
    let water = PlanetWaterParams::from_static_surface(baked);

    let map_radius = render_radius;
    let ship_radius = ((body.radius_m * SHIP_SCALE) as f32).max(0.005);

    let make_material = |radius: f32, atmosphere: AtmosphereBlock| PlanetMaterial {
        params: PlanetParams {
            radius,
            height_range,
            terminator_wrap: roughness,
            // Airless bodies leave `sea_level_m` at the default
            // sentinel; the shader's water BRDF never fires for them.
            sea_level_m: baked.sea_level_m.unwrap_or(-1.0e9),
            water_color_depth: water.color_depth,
            coastline_warp_amp_radians: coastline.warp_amp_radians,
            coastline_jitter_amp_m: coastline.jitter_amp_m,
            coastline_seed: coastline.seed,
            ..default()
        },
        albedo: textures.albedo.clone(),
        height: textures.height.clone(),
        detail: detail.clone(),
        roughness: textures.roughness.clone(),
        craters: textures.craters.clone(),
        cell_index: textures.cell_index.clone(),
        feature_ids: textures.feature_ids.clone(),
        radial_features: textures.radial_features.clone(),
        atmosphere,
        cloud_cover: cloud_cover.clone(),
        ice_caps: textures.ice_caps.clone(),
        active_dunes: textures.active_dunes.clone(),
        active_dune_height: textures.active_dune_height.clone(),
        active_dune_albedo: textures.active_dune_albedo.clone(),
    };

    let map_material = make_material(map_radius, map_atmosphere);
    let ship_material = make_material(ship_radius, ship_atmosphere);
    let map_halo_handle = assets
        .planet_halo_materials
        .add(PlanetHaloMaterial::from(&map_material));
    let ship_halo_handle = assets
        .planet_halo_materials
        .add(PlanetHaloMaterial::from(&ship_material));
    let map_handle = assets.planet_materials.add(map_material);
    let ship_handle = assets.planet_materials.add(ship_material);

    // Map-layer impostor billboard.
    commands.spawn((
        Mesh3d(shared.billboard.clone()),
        MeshMaterial3d(map_handle.clone()),
        BodyMesh,
        bevy::camera::visibility::RenderLayers::layer(MAP_LAYER),
        // Billboard's local AABB is a flat 2×2 quad; the vertex shader
        // re-orients it each frame. Disable frustum culling so Bevy
        // doesn't hide it at angles where the AABB misses the view
        // frustum.
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        ChildOf(entities.body_entity),
        Name::new(format!("{} Impostor (Map)", body.name)),
    ));

    // Ship-layer impostor billboard. Pair this impostor with the
    // ground-LOD terrain via `RealSpaceImpostor` so `sync_body_render_lod`
    // can hide one half at a time based on camera distance.
    commands.spawn((
        Mesh3d(shared.billboard.clone()),
        MeshMaterial3d(ship_handle.clone()),
        ShipBodyMesh,
        bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER),
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        RealSpaceImpostor { body_id },
        ChildOf(entities.ship_parent_entity),
        Name::new(format!("{} Impostor (Ship)", body.name)),
    ));

    commands.spawn((
        Mesh3d(shared.billboard.clone()),
        MeshMaterial3d(map_halo_handle.clone()),
        BodyMesh,
        bevy::camera::visibility::RenderLayers::layer(MAP_LAYER),
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        ChildOf(entities.body_entity),
        Name::new(format!("{} Halo (Map)", body.name)),
    ));

    commands.spawn((
        Mesh3d(shared.billboard.clone()),
        MeshMaterial3d(ship_halo_handle.clone()),
        ShipBodyMesh,
        bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER),
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
        // Visibility is owned by `sync_body_render_lod`: shown when the
        // camera is outside the atmosphere shell (it provides the rim
        // halo around the impostor / terrain silhouette), hidden when
        // inside (the `BodySky` fullscreen pass takes over). Start
        // hidden so it doesn't double-contribute on frame 0 if the
        // first camera position lands inside the shell.
        Visibility::Hidden,
        BodyHalo { body_id },
        ChildOf(entities.ship_parent_entity),
        Name::new(format!("{} Halo (Ship)", body.name)),
    ));

    {
        let mut entity_cmds = commands.entity(entities.body_entity);
        entity_cmds.insert(PlanetMaterials {
            map: map_handle,
            ship: ship_handle,
            map_halo: map_halo_handle,
            ship_halo: ship_halo_handle,
        });
        if cloud_cover_config.is_some() && uses_reference_cloud {
            entity_cmds.insert(ReferenceCloudTarget {
                body_name: body.name.clone(),
            });
        }
    }

    // Spawn the ground-LOD terrain entity alongside the impostor. The same
    // `PlanetSurface` backs both via `Arc`.
    spawn_body_terrain(
        commands,
        body,
        surface.clone(),
        entities.ship_parent_entity,
        assets.body_terrain_materials,
        assets.tile_trees,
        entities.ship_camera,
        ship_atmosphere,
        dynamic_state.clone(),
    );

    // Per-body water sphere. Ocean-only — `spawn_body_water` returns early
    // when the baked surface has no `sea_level_m`. Visibility is paired with
    // the terrain entity by `sync_body_render_lod` (impostor's inline water
    // BRDF takes over outside the LOD swap radius).
    spawn_body_water(
        commands,
        body,
        &surface,
        entities.ship_parent_entity,
        assets.meshes,
        assets.body_water_materials,
    );
}

/// Output of the off-thread bake-load task. Owns everything the main
/// thread needs to call [`install_baked_planet`]: the decoded surface,
/// the GPU-ready `PreparedPlanetBake`, and the seeded dynamic state.
pub(super) struct PlanetBakeOutput {
    pub surface: PlanetSurface,
    pub dynamic_state: DynamicSurfaceState,
    pub prepared: PreparedPlanetBake,
}

/// A single procedural body whose bake is loading. `spawn_bodies` pushes
/// one of these per procedural body, [`poll_planet_install_tasks`] drains
/// completed entries each frame.
pub(super) struct PendingPlanetInstall {
    pub body_id: usize,
    pub body_name: String,
    pub body_entity: Entity,
    pub real_body_entity: Entity,
    pub render_radius: f32,
    pub task: Task<PlanetBakeOutput>,
}

/// Resource holding all in-flight bake loads. Drained as tasks complete.
#[derive(Resource, Default)]
pub(super) struct PendingPlanetInstalls(pub Vec<PendingPlanetInstall>);

/// Per-frame poll: any task whose load + prepare has finished is handed
/// off to [`install_baked_planet`] (main-thread asset upload + entity
/// spawn). Each install ticks `LoadingProgress.completed`; once the
/// counter matches `total`, the loading overlay hides itself.
#[allow(clippy::too_many_arguments)]
pub(super) fn poll_planet_install_tasks(
    mut commands: Commands,
    mut pending: ResMut<PendingPlanetInstalls>,
    mut progress: ResMut<LoadingProgress>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    reference_clouds: Res<ReferenceClouds>,
    sim: Res<super::types::SimulationState>,
    shared: Option<Res<SharedPlanetMeshes>>,
    mut planet_material_assets: PlanetMaterialAssets,
    mut procedural_install_extras: ProceduralInstallExtras,
    mut world_state: WorldStateAssets,
) {
    if pending.0.is_empty() {
        return;
    }
    let Some(shared) = shared else {
        // `SharedPlanetMeshes` is inserted by `spawn_bodies`; if it isn't
        // here yet the startup system hasn't run, which means no tasks are
        // in flight either. Defensive in case ordering ever changes.
        return;
    };

    let ship_camera = match procedural_install_extras.ship_camera_q.single() {
        Ok(e) => e,
        Err(_) => {
            // Camera not spawned yet on the first frame; wait one tick.
            return;
        }
    };

    // Drain completed tasks in place. `block_on(poll_once(..))` is a
    // non-blocking check on the AsyncCompute thread pool's task handle —
    // returns `Some(output)` only when the task has finished.
    let mut i = 0;
    while i < pending.0.len() {
        let Some(output) = block_on(poll_once(&mut pending.0[i].task)) else {
            i += 1;
            continue;
        };
        // `swap_remove` is safe here: we don't increment `i`, so the entry
        // moved into slot `i` will be polled on the next iteration.
        let entry = pending.0.swap_remove(i);
        let body = &sim.system.bodies[entry.body_id];

        install_baked_planet(
            &mut commands,
            body,
            entry.body_id,
            entry.render_radius,
            output.surface,
            output.prepared,
            output.dynamic_state,
            &shared,
            &reference_clouds,
            &procedural_install_extras.terrain_registry,
            InstallEntities {
                body_entity: entry.body_entity,
                ship_parent_entity: entry.real_body_entity,
                ship_camera,
            },
            InstallAssets {
                planet_materials: &mut planet_material_assets.planet,
                planet_halo_materials: &mut planet_material_assets.planet_halo,
                body_terrain_materials: &mut planet_material_assets.body_terrain,
                body_water_materials: &mut planet_material_assets.body_water,
                meshes: &mut meshes,
                images: &mut images,
                storage_buffers: &mut procedural_install_extras.storage_buffers,
                tile_trees: &mut procedural_install_extras.tile_trees,
                terrain_surfaces: &mut procedural_install_extras.terrain_surfaces,
                planetshine: &mut world_state.planetshine,
                solar_system: &mut world_state.solar_system,
            },
        );

        progress.completed += 1;
        progress.label = entry.body_name;
    }
}

#[derive(Component)]
pub(super) struct ReferenceCloudTarget {
    pub(super) body_name: String,
}

pub(super) fn patch_reference_cloud_covers(
    clouds: Res<ReferenceClouds>,
    targets: Query<(&PlanetMaterials, &ReferenceCloudTarget, &CelestialBody)>,
    skies: Query<(&BodySky, &MeshMaterial3d<BodySkyMaterial>)>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    mut halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
    mut sky_materials: ResMut<Assets<BodySkyMaterial>>,
) {
    for (mats, target, body) in &targets {
        let Some(cube) = clouds.cube(&target.body_name) else {
            continue;
        };
        for handle in [&mats.map, &mats.ship] {
            if let Some(mat) = materials.get_mut(handle)
                && mat.cloud_cover != cube
            {
                mat.cloud_cover = cube.clone();
            }
        }
        for handle in [&mats.map_halo, &mats.ship_halo] {
            if let Some(mat) = halo_materials.get_mut(handle)
                && mat.cloud_cover != cube
            {
                mat.cloud_cover = cube.clone();
            }
        }
        for (sky, handle) in &skies {
            if sky.body_id != body.body_id {
                continue;
            }
            if let Some(mat) = sky_materials.get_mut(handle)
                && mat.cloud_cover != cube
            {
                mat.cloud_cover = cube.clone();
            }
        }
    }
}
