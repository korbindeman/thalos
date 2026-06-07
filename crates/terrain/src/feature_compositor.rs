//! Runtime terrain-feature compositor for the Query API.
//!
//! This is the first P2 Mira slice: discrete terrain-shaping features stored on
//! [`StaticSurfaceData`](crate::static_surface::StaticSurfaceData) are folded
//! into the queried geometric height instead of existing only in the impostor
//! SSBO path. Large craters that were already baked into the cubemap are skipped
//! here to avoid double-counting; unbaked craters become part of the single
//! mesh/collider surface.

use glam::Vec3;

use crate::crater_profile::{SubPeaks, crater_profile, degradation_factor, morphology_for_radius};
use crate::spatial_index::FeatureRef;
use crate::static_surface::StaticSurfaceData;
use crate::types::Crater;

/// Conservative vertical headroom needed for runtime-composed features that
/// are not already represented in the baked cubemap.
pub(crate) fn runtime_feature_height_margin_m(static_surface: &StaticSurfaceData) -> f32 {
    static_surface
        .craters
        .iter()
        .filter(|crater| crater.radius_m < static_surface.cubemap_bake_threshold_m)
        .map(|crater| crater.depth_m.abs().max(crater.rim_height_m.abs()))
        .fold(0.0, f32::max)
}

/// Compose all runtime terrain features that influence `dir` onto
/// `base_height_m`.
///
/// Determinism rule: feature refs are deduplicated, then sorted by their index
/// in `StaticSurfaceData::craters`. The cratering stage stores craters
/// oldest-first, so younger craters are applied later and stamp over older
/// terrain consistently regardless of bucket/neighbour iteration order.
pub(crate) fn compose_runtime_features_m(
    static_surface: &StaticSurfaceData,
    dir: Vec3,
    base_height_m: f32,
) -> f32 {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO || static_surface.craters.is_empty() {
        return base_height_m;
    }

    let mut crater_indices: Vec<u32> = static_surface
        .feature_index
        .lookup_with_neighbors(dir)
        .filter_map(|feature| match feature {
            FeatureRef::Crater(index) => Some(index),
            FeatureRef::Volcano(_) | FeatureRef::Channel(_) => None,
        })
        .collect();

    if crater_indices.is_empty() {
        return base_height_m;
    }

    crater_indices.sort_unstable();
    crater_indices.dedup();

    let mut height_m = base_height_m;
    for index in crater_indices {
        let Some(crater) = static_surface.craters.get(index as usize) else {
            continue;
        };
        height_m = compose_unbaked_crater_m(
            height_m,
            crater,
            static_surface.radius_m,
            static_surface.cubemap_bake_threshold_m,
            dir,
        );
    }
    height_m
}

fn compose_unbaked_crater_m(
    height_m: f32,
    crater: &Crater,
    body_radius_m: f32,
    cubemap_bake_threshold_m: f32,
    dir: Vec3,
) -> f32 {
    // Craters at/above the bake threshold are already in the static height
    // cubemap. Leave them alone here so the runtime compositor only fills the
    // feature band that used to be impostor-only.
    if crater.radius_m >= cubemap_bake_threshold_m || crater.radius_m <= 0.0 {
        return height_m;
    }

    let center = crater.center.normalize_or_zero();
    if center == Vec3::ZERO {
        return height_m;
    }

    let angular_dist = center.dot(dir).clamp(-1.0, 1.0).acos();
    let surface_dist_m = angular_dist * body_radius_m.max(1.0);
    if surface_dist_m > crater.influence_radius_m() {
        return height_m;
    }

    let degrad = degradation_factor(crater.radius_m, crater.age_gyr);
    let delta_m = crater_profile(
        surface_dist_m / crater.radius_m,
        crater.depth_m * degrad,
        crater.rim_height_m * degrad,
        crater.radius_m,
        morphology_for_radius(crater.radius_m),
        0.0,
        0.0,
        0.0,
        &SubPeaks::default(),
        1.0,
        0.0,
    );

    height_m + delta_m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crater(center: Vec3, radius_m: f32, age_gyr: f32) -> Crater {
        Crater {
            center,
            radius_m,
            depth_m: radius_m * 0.18,
            rim_height_m: radius_m * 0.035,
            age_gyr,
            material_id: 0,
        }
    }

    #[test]
    fn runtime_margin_ignores_baked_craters() {
        let mut surface = minimal_static_surface();
        surface.cubemap_bake_threshold_m = 1_500.0;
        surface.craters = vec![crater(Vec3::X, 1_000.0, 0.0), crater(Vec3::Y, 2_000.0, 0.0)];

        let margin = runtime_feature_height_margin_m(&surface);
        assert_eq!(margin, 180.0);
    }

    fn minimal_static_surface() -> StaticSurfaceData {
        use crate::cubemap::Cubemap;
        use crate::generic_terrestrial_field::RuntimeTerrainDetail;
        use crate::spatial_index::IcoBuckets;
        use crate::types::{DetailNoiseParams, Material};

        StaticSurfaceData {
            radius_m: 100_000.0,
            cubemap_bake_threshold_m: 1_500.0,
            height_cubemap: Cubemap::new(1),
            height_range: 1.0,
            albedo_cubemap: Cubemap::new(1),
            material_cubemap: Cubemap::new(1),
            biome_weights_cubemap: Cubemap::new(1),
            roughness_cubemap: Cubemap::new(1),
            normal_cubemap: Cubemap::new(1),
            craters: Vec::new(),
            volcanoes: Vec::new(),
            channels: Vec::new(),
            feature_index: IcoBuckets::empty(),
            runtime_detail: RuntimeTerrainDetail::LegacyHmf,
            detail_params: DetailNoiseParams::default(),
            materials: vec![Material {
                albedo: [0.1, 0.1, 0.1],
                roughness: 0.8,
            }],
            mean_albedo: [0.1, 0.1, 0.1],
            sea_level_m: None,
            water_appearance: None,
        }
    }

    #[test]
    fn baked_craters_are_not_composed_again() {
        let c = crater(Vec3::X, 2_000.0, 0.1);
        let h = compose_unbaked_crater_m(123.0, &c, 100_000.0, 1_500.0, Vec3::X);
        assert_eq!(h, 123.0);
    }

    #[test]
    fn unbaked_crater_carves_center() {
        let c = crater(Vec3::X, 1_000.0, 0.0);
        let h = compose_unbaked_crater_m(0.0, &c, 100_000.0, 1_500.0, Vec3::X);
        assert!(
            h < -50.0,
            "expected crater center to carve below base, got {h}"
        );
    }

    #[test]
    fn unbaked_crater_does_not_affect_distant_samples() {
        let c = crater(Vec3::X, 1_000.0, 0.0);
        let h = compose_unbaked_crater_m(42.0, &c, 100_000.0, 1_500.0, Vec3::Y);
        assert_eq!(h, 42.0);
    }
}
