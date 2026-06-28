//! Runtime body-install support + reference-cloud patching.
//!
//! Procedural bodies generate their terrain at runtime via
//! [`thalos_terrain::ProceduralSurface`] behind the `SurfaceQuery` seam — there
//! is no pre-baked artifact to load. [`super::spawn::spawn_bodies`] spawns the
//! body shell + a solid-color impostor synchronously and registers the runtime
//! surface for the near-surface height source ([`HeightSourceRegistry`]) and the
//! propagator ([`crate::GameTerrainRegistry`]); the ground-LOD terrain itself is
//! spawned on demand by [`crate::rendering::terrain_residency`].
//!
//! The [`ProceduralInstallExtras`] / [`WorldStateAssets`] `SystemParam`s bundle
//! the registries `spawn_bodies` writes into so it stays under Bevy 0.18's
//! 16-param limit.
//!
//! [`patch_reference_cloud_covers`] remains a per-frame system because the
//! reference-cloud cubemap loads asynchronously from disk (image decoding).

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use thalos_body_render::{BodySkyMaterial, PlanetHaloMaterial, PlanetMaterial, ReferenceClouds};
use thalos_physics_local::HeightSourceRegistry;

use super::ground_terrain::BodySky;
use super::types::{CelestialBody, PlanetMaterials, PlanetshineTints};

/// Registry handles `spawn_bodies` needs when registering each procedural
/// body's runtime surface. Bundled to stay under the 16-param limit.
#[derive(SystemParam)]
pub(super) struct ProceduralInstallExtras<'w> {
    pub(super) height_sources: ResMut<'w, HeightSourceRegistry>,
    pub(super) terrain_registry: Res<'w, crate::GameTerrainRegistry>,
}

/// World-level state the spawn path writes into. Bundled to stay under the
/// 16-param limit.
#[derive(SystemParam)]
pub(super) struct WorldStateAssets<'w> {
    pub(super) planetshine: ResMut<'w, PlanetshineTints>,
}

/// Marks a body whose impostor/sky cloud cover should track an
/// asynchronously-loaded reference-cloud cubemap.
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
