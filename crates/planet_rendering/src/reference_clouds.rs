//! Temporary reference-cloud texture support shared by the game and editor.
//!
//! Bodies listed here use hand-picked equirectangular cloud textures while the
//! procedural cloud generator is being redesigned. Other bodies fall back to
//! the procedural cloud-cover bake driven by `TerrestrialAtmosphere::clouds`.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::bake::{
    bake_cloud_cover_image, blank_cloud_cover_image, equirect_to_cloud_cover_image_with_rotation,
};

#[derive(Clone, Copy)]
struct ReferenceCloudImage {
    body_name: &'static str,
    path: &'static str,
    orientation: ReferenceCloudOrientation,
}

#[derive(Clone, Copy, Default)]
enum ReferenceCloudOrientation {
    #[default]
    Identity,
    FrontToNorthPole,
}

impl ReferenceCloudOrientation {
    fn source_from_output_rotation(self) -> Quat {
        match self {
            Self::Identity => Quat::IDENTITY,
            // Source image front-center is +X in the equirect mapping.
            // Sampling output +Y from source +X moves that feature to the
            // north pole, which is the requested "rotate up" orientation.
            Self::FrontToNorthPole => Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2),
        }
    }
}

const REFERENCE_CLOUD_IMAGES: &[ReferenceCloudImage] = &[
    ReferenceCloudImage {
        body_name: "Thalos",
        path: "australia_clouds_8k.jpg",
        orientation: ReferenceCloudOrientation::Identity,
    },
    ReferenceCloudImage {
        body_name: "Pelagos",
        path: "storm_clouds_8k.jpg",
        orientation: ReferenceCloudOrientation::Identity,
    },
    ReferenceCloudImage {
        body_name: "Vaelen",
        path: "korriban_clouds_4k.png",
        orientation: ReferenceCloudOrientation::FrontToNorthPole,
    },
];
const REFERENCE_CLOUD_CUBE_RES: u32 = 512;

pub fn reference_cloud_path(body_name: &str) -> Option<&'static str> {
    REFERENCE_CLOUD_IMAGES
        .iter()
        .find(|spec| spec.body_name == body_name)
        .map(|spec| spec.path)
}

#[derive(Default)]
struct ReferenceCloudEntry {
    source: Option<Handle<Image>>,
    cube: Option<Handle<Image>>,
    orientation: ReferenceCloudOrientation,
}

#[derive(Resource, Default)]
pub struct ReferenceClouds {
    entries: HashMap<String, ReferenceCloudEntry>,
}

impl ReferenceClouds {
    pub fn cube(&self, body_name: &str) -> Option<Handle<Image>> {
        self.entries
            .get(body_name)
            .and_then(|entry| entry.cube.clone())
    }
}

pub fn load_reference_cloud_sources(
    asset_server: Res<AssetServer>,
    mut clouds: ResMut<ReferenceClouds>,
) {
    for spec in REFERENCE_CLOUD_IMAGES {
        clouds.entries.insert(
            spec.body_name.to_string(),
            ReferenceCloudEntry {
                source: Some(asset_server.load(spec.path)),
                cube: None,
                orientation: spec.orientation,
            },
        );
    }
}

pub fn convert_reference_clouds_when_ready(
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
        let cube_image = equirect_to_cloud_cover_image_with_rotation(
            source,
            REFERENCE_CLOUD_CUBE_RES,
            entry.orientation.source_from_output_rotation(),
        );
        entry.cube = Some(images.add(cube_image));
        entry.source = None;
    }
}

pub fn cloud_cover_image_for_body(
    body_name: &str,
    cloud_seed: Option<u64>,
    reference_clouds: &ReferenceClouds,
    images: &mut Assets<Image>,
) -> (Handle<Image>, bool) {
    if reference_cloud_path(body_name).is_some() {
        return (
            reference_clouds
                .cube(body_name)
                .unwrap_or_else(|| blank_cloud_cover_image(images)),
            true,
        );
    }

    (
        cloud_seed
            .map(|seed| bake_cloud_cover_image(seed, images))
            .unwrap_or_else(|| blank_cloud_cover_image(images)),
        false,
    )
}
