//! Canonical ship-view analytic-ocean projection.
//!
//! Ocean visibility follows resident ground terrain, but its material is a
//! sibling of the atmosphere and clouds. This keeps the analytic sphere and
//! signed sea field independent of the custom `BodySky` material while their
//! explicit composition order remains stable.

use bevy::prelude::*;
use thalos_body_render::{BodyOceanMaterial, BodySkyMaterial};
use thalos_world::BodyId;

/// Per-body fullscreen analytic-ocean projection.
#[derive(Component, Debug)]
pub(super) struct BodyOcean {
    pub(super) body_id: BodyId,
}

pub(super) struct OceanRenderPlugin;

impl Plugin for OceanRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            sync_ocean_materials.after(super::ground_terrain::update_body_terrain_atmosphere),
        );
    }
}

/// Mirror the one per-body optical projection into the dedicated ocean
/// material. Static images are cloned at spawn; only frame-varying sun/body/
/// phase data and the current terrain-lookup entity change here.
fn sync_ocean_materials(
    skies: Query<(
        &super::ground_terrain::BodySky,
        &MeshMaterial3d<BodySkyMaterial>,
    )>,
    oceans: Query<(&BodyOcean, &MeshMaterial3d<BodyOceanMaterial>)>,
    sky_materials: Res<Assets<BodySkyMaterial>>,
    mut ocean_materials: ResMut<Assets<BodyOceanMaterial>>,
) {
    let sky_state: std::collections::HashMap<BodyId, _> = skies
        .iter()
        .filter_map(|(sky, handle)| {
            sky_materials.get(handle).map(|material| {
                (
                    sky.body_id,
                    (
                        material.atmosphere,
                        material.atmosphere_extra,
                        material.terrain_entity,
                        material.cloud_shadow,
                        material.cloud_shadow_map.clone(),
                    ),
                )
            })
        })
        .collect();

    for (ocean, handle) in &oceans {
        let Some((atmosphere, extra, terrain_entity, cloud_shadow, cloud_shadow_map)) =
            sky_state.get(&ocean.body_id).cloned()
        else {
            continue;
        };
        let Some(mut material) = ocean_materials.get_mut(handle) else {
            continue;
        };
        material.optical.atmosphere = atmosphere;
        material.optical.atmosphere_extra = extra;
        material.optical.terrain_entity = terrain_entity;
        // Shafts over water shade through the same cascade as the sky pass.
        material.optical.cloud_shadow = cloud_shadow;
        material.optical.cloud_shadow_map = cloud_shadow_map;
    }
}
