//! Polling and finalization of in-flight terrain generation.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use bevy::tasks::{block_on, poll_once};
use thalos_physics::types::BodyKind;
use thalos_planet_rendering::{
    AtmosphereBlock, PlanetCoastlineParams, PlanetDetailParams, PlanetHaloMaterial, PlanetMaterial,
    PlanetParams, PlanetWaterParams, ReferenceClouds, bake_from_body_data,
    cloud_cover_image_for_body,
};

use super::types::{
    BodyMesh, CloudBandState, PendingPlanetGeneration, PlanetMaterials, PlanetshineTints,
    SharedPlanetMeshes, ShipBodyMesh, SimulationState,
};
use crate::coords::{MAP_LAYER, MAP_SCALE, SHIP_LAYER, SHIP_SCALE};

// ---------------------------------------------------------------------------
// Per-frame: finalise async terrain generation
// ---------------------------------------------------------------------------

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
fn body_surface_roughness(body: &thalos_physics::types::BodyDefinition) -> f32 {
    match body.kind {
        BodyKind::Star => 0.0,
        BodyKind::Planet => 0.0,
        BodyKind::Moon => 0.0,
        BodyKind::DwarfPlanet => 0.0,
        BodyKind::Centaur => 0.0,
        BodyKind::Comet => 0.0,
    }
}

/// Poll in-flight terrain tasks. When one completes, bake the result into GPU
/// textures, build the `PlanetMaterial`, and swap the body's placeholder sphere
/// for the impostor billboard.
pub(super) fn finalize_planet_generation(
    mut commands: Commands,
    mut pending_q: Query<(Entity, &mut PendingPlanetGeneration)>,
    sim: Res<SimulationState>,
    shared: Res<SharedPlanetMeshes>,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut planet_halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    reference_clouds: Res<ReferenceClouds>,
    mut planetshine: ResMut<PlanetshineTints>,
) {
    for (entity, mut pending) in &mut pending_q {
        let _span = tracing::info_span!("finalize_planet_generation").entered();
        let Some(baked) = block_on(poll_once(&mut pending.task)) else {
            continue;
        };

        let body = &sim.system.bodies[pending.body_id];
        let detail =
            PlanetDetailParams::from_body(&baked.detail_params, baked.cubemap_bake_threshold_m);
        let height_range = baked.height_range;
        planetshine
            .by_body
            .insert(pending.body_id, baked.mean_albedo);
        let textures = bake_from_body_data(&baked, &mut images, &mut storage_buffers);

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

        let cloud_seed = body
            .terrestrial_atmosphere
            .as_ref()
            .and_then(|a| a.clouds.as_ref())
            .map(|c| c.seed);
        let (cloud_cover, uses_reference_cloud) =
            cloud_cover_image_for_body(&body.name, cloud_seed, &reference_clouds, &mut images);

        let coastline = PlanetCoastlineParams::from_body_data(&baked);
        let water = PlanetWaterParams::from_body_data(&baked);

        let map_radius = pending.render_radius;
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
            atmosphere,
            cloud_cover: cloud_cover.clone(),
        };

        let map_material = make_material(map_radius, map_atmosphere);
        let ship_material = make_material(ship_radius, ship_atmosphere);
        let map_halo_handle = planet_halo_materials.add(PlanetHaloMaterial::from(&map_material));
        let ship_halo_handle = planet_halo_materials.add(PlanetHaloMaterial::from(&ship_material));
        let map_handle = planet_materials.add(map_material);
        let ship_handle = planet_materials.add(ship_material);

        let mesh_entity = pending.mesh_entity;
        commands
            .entity(mesh_entity)
            .insert((
                Mesh3d(shared.billboard.clone()),
                MeshMaterial3d(map_handle.clone()),
                // The billboard's local AABB is a flat 2×2 quad; the
                // vertex shader re-orients it each frame. Disable
                // frustum culling so Bevy doesn't hide it at angles
                // where the flat AABB misses the view frustum.
                NoFrustumCulling,
            ))
            .remove::<MeshMaterial3d<StandardMaterial>>();

        let ship_mesh_entity = pending.ship_mesh_entity;
        commands
            .entity(ship_mesh_entity)
            .insert((
                Mesh3d(shared.billboard.clone()),
                MeshMaterial3d(ship_handle.clone()),
                NoFrustumCulling,
            ))
            .remove::<MeshMaterial3d<StandardMaterial>>();

        commands.spawn((
            Mesh3d(shared.billboard.clone()),
            MeshMaterial3d(map_halo_handle.clone()),
            BodyMesh,
            bevy::camera::visibility::RenderLayers::layer(MAP_LAYER),
            NoFrustumCulling,
            NotShadowCaster,
            NotShadowReceiver,
            ChildOf(entity),
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
            ChildOf(entity),
            Name::new(format!("{} Halo (Ship)", body.name)),
        ));

        let has_clouds = body
            .terrestrial_atmosphere
            .as_ref()
            .and_then(|a| a.clouds.as_ref())
            .is_some();
        let mut entity_cmds = commands.entity(entity);
        entity_cmds
            .insert(PlanetMaterials {
                map: map_handle,
                ship: ship_handle,
                map_halo: map_halo_handle,
                ship_halo: ship_halo_handle,
            })
            .remove::<PendingPlanetGeneration>();
        if has_clouds {
            entity_cmds.insert(CloudBandState::default());
        }
        if uses_reference_cloud {
            entity_cmds.insert(ReferenceCloudTarget {
                body_name: body.name.clone(),
            });
        }
    }
}

#[derive(Component)]
pub(super) struct ReferenceCloudTarget {
    pub(super) body_name: String,
}

pub(super) fn patch_reference_cloud_covers(
    clouds: Res<ReferenceClouds>,
    targets: Query<(&PlanetMaterials, &ReferenceCloudTarget)>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    mut halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
) {
    for (mats, target) in &targets {
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
    }
}
