//! Polling and finalization of in-flight terrain generation, plus the
//! reference-cloud (TEMP) scaffold that lets specific bodies use a
//! hand-picked equirect photo cube instead of the procedural Wedekind
//! cloud bake.

use std::collections::HashMap;

use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use bevy::tasks::{block_on, poll_once};
use thalos_physics::types::BodyKind;
use thalos_planet_rendering::{
    AtmosphereBlock, PlanetDetailParams, PlanetHaloMaterial, PlanetMaterial, PlanetParams,
    bake_cloud_cover_image, bake_from_body_data, blank_cloud_cover_image,
    equirect_to_cloud_cover_image,
};

use super::types::{
    BodyMesh, CloudBandState, PendingPlanetGeneration, PlanetMaterials, SharedPlanetMeshes,
    ShipBodyMesh, SimulationState,
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

        // Bake the cloud-cover cubemap when the body has a cloud layer.
        // Bodies without clouds get a 1×1 blank fallback; the shader
        // gates its cloud path on `cloud_albedo_coverage.w > 0` so the
        // blank cube is effectively free.
        //
        // TEMP: bodies listed in `REFERENCE_CLOUD_IMAGES` use a hand-
        // picked photo cube instead of the procedural Wedekind bake.
        // If the async image decode hasn't finished yet, fall back to
        // blank; `patch_reference_cloud_covers` will swap the real cube
        // in once it's ready.
        let uses_reference_cloud = reference_cloud_path(&body.name).is_some();
        let cloud_cover = if uses_reference_cloud {
            reference_clouds
                .entries
                .get(&body.name)
                .and_then(|e| e.cube.clone())
                .unwrap_or_else(|| blank_cloud_cover_image(&mut images))
        } else {
            body.terrestrial_atmosphere
                .as_ref()
                .and_then(|a| a.clouds.as_ref())
                .map(|c| {
                    let _span = tracing::info_span!("bake_cloud_cover").entered();
                    bake_cloud_cover_image(c.seed, &mut images)
                })
                .unwrap_or_else(|| blank_cloud_cover_image(&mut images))
        };

        // Canonical high-frequency terrain bands — only enable on bodies
        // with a sea level; airless bodies skip the per-fragment fbm to
        // keep the shader cheap. The warp displaces the cubemap sample
        // direction by ~1 texel of arc on the sphere, breaking the
        // texel-grid staircase visible from orbit. The jitter adds
        // sub-texel surface detail on close approach.
        //
        // Seed folds the body seed's high+low halves and xors a per-band
        // magic so the coastline fields decorrelate from bake-time fbm
        // fields that share the body seed.
        let body_seed = baked.detail_params.seed;
        let coastline_seed = (body_seed as u32) ^ ((body_seed >> 32) as u32) ^ 0xC0A5_71_1Eu32;
        let has_ocean = baked.sea_level_m.is_some();
        // ~1 texel of arc on a 2048² cube = 2π/(4·2048) ≈ 7.7e-4 rad. 8e-4
        // is one texel of fbm-amplitude headroom; the design point.
        let coastline_warp_amp_radians = if has_ocean { 8.0e-4 } else { 0.0 };
        let coastline_jitter_amp_m = if has_ocean { 30.0 } else { 0.0 };

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
                coastline_warp_amp_radians,
                coastline_jitter_amp_m,
                coastline_seed,
                ..default()
            },
            albedo: textures.albedo.clone(),
            height: textures.height.clone(),
            detail: detail.clone(),
            material_cube: textures.material_cube.clone(),
            craters: textures.craters.clone(),
            cell_index: textures.cell_index.clone(),
            feature_ids: textures.feature_ids.clone(),
            materials: textures.materials.clone(),
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

// ---------------------------------------------------------------------------
// Reference cloud textures (TEMP)
//
// Per-body mapping from body name to an equirectangular source image in
// `assets/`. Each source is loaded at startup, projected equirectangular
// → cubemap once Bevy's async decode finishes, then swapped into the
// body's material by `patch_reference_cloud_covers` (handles the case
// where the body materialises before its cube is ready).
//
// This is a scaffold used to give specific bodies a hand-picked weather
// look while the procedural Wedekind cloud pipeline is being redesigned.
// Bodies not listed here fall through to the procedural bake.
// ---------------------------------------------------------------------------

const REFERENCE_CLOUD_IMAGES: &[(&str, &str)] = &[
    ("Thalos", "australia_clouds_8k.jpg"),
    ("Pelagos", "storm_clouds_8k.jpg"),
];
const REFERENCE_CLOUD_CUBE_RES: u32 = 512;

fn reference_cloud_path(body_name: &str) -> Option<&'static str> {
    REFERENCE_CLOUD_IMAGES
        .iter()
        .find(|(name, _)| *name == body_name)
        .map(|(_, path)| *path)
}

#[derive(Default)]
pub struct ReferenceCloudEntry {
    source: Option<Handle<Image>>,
    pub cube: Option<Handle<Image>>,
}

#[derive(Resource, Default)]
pub struct ReferenceClouds {
    // Keyed by body name (matching `REFERENCE_CLOUD_IMAGES`).
    entries: HashMap<String, ReferenceCloudEntry>,
}

#[derive(Component)]
pub(super) struct ReferenceCloudTarget {
    pub(super) body_name: String,
}

pub(super) fn load_reference_cloud_sources(
    asset_server: Res<AssetServer>,
    mut clouds: ResMut<ReferenceClouds>,
) {
    for (body_name, path) in REFERENCE_CLOUD_IMAGES {
        clouds.entries.insert(
            (*body_name).to_string(),
            ReferenceCloudEntry {
                source: Some(asset_server.load(*path)),
                cube: None,
            },
        );
    }
}

pub(super) fn convert_reference_clouds_when_ready(
    mut clouds: ResMut<ReferenceClouds>,
    mut images: ResMut<Assets<Image>>,
) {
    for entry in clouds.entries.values_mut() {
        if entry.cube.is_some() {
            continue;
        }
        let Some(source_handle) = entry.source.clone() else {
            continue;
        };
        let Some(source) = images.get(&source_handle) else {
            continue;
        };
        let _span = tracing::info_span!("equirect_to_cloud_cover").entered();
        let cube_image = equirect_to_cloud_cover_image(source, REFERENCE_CLOUD_CUBE_RES);
        entry.cube = Some(images.add(cube_image));
        // Drop the source handle so bevy can free the 128 MB 8k decode.
        entry.source = None;
    }
}

pub(super) fn patch_reference_cloud_covers(
    clouds: Res<ReferenceClouds>,
    targets: Query<(&PlanetMaterials, &ReferenceCloudTarget)>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    mut halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
) {
    for (mats, target) in &targets {
        let Some(entry) = clouds.entries.get(&target.body_name) else {
            continue;
        };
        let Some(cube) = entry.cube.as_ref() else {
            continue;
        };
        for handle in [&mats.map, &mats.ship] {
            if let Some(mat) = materials.get_mut(handle)
                && mat.cloud_cover != *cube
            {
                mat.cloud_cover = cube.clone();
            }
        }
        for handle in [&mats.map_halo, &mats.ship_halo] {
            if let Some(mat) = halo_materials.get_mut(handle)
                && mat.cloud_cover != *cube
            {
                mat.cloud_cover = cube.clone();
            }
        }
    }
}
