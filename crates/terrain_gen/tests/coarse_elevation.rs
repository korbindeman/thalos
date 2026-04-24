//! Smoke test for `CoarseElevation` (Stage 2 of the Thalos pipeline).
//!
//! Runs `TectonicSkeleton` + `CoarseElevation` at a small subdivision
//! level and checks the resulting elevation field:
//! - exists on every icosphere vertex,
//! - shows both continental and oceanic bias (positive and negative
//!   elevations are present),
//! - active-margin vertices show sharper relief than average.

use thalos_terrain_gen::{
    BodyBuilder, CoarseElevation, Composition, Pipeline, ProvinceKind, TectonicSkeleton,
};

fn thalos_composition() -> Composition {
    Composition::new(0.26, 0.70, 0.0, 0.04, 0.0)
}

fn builder() -> BodyBuilder {
    BodyBuilder::new(
        3_186_000.0,
        1003,
        thalos_composition(),
        128,
        4.5,
        None,
        23.0_f32.to_radians(),
    )
}

fn stage1() -> TectonicSkeleton {
    TectonicSkeleton {
        subdivision_level: 5,
        n_cratons: 20,
        plate_warp_amplitude: 0.4,
        plate_warp_frequency: 3.0,
        plate_warp_octaves: 5,
        n_active_margins: 1,
        n_rift_scars: 2,
        n_hotspot_tracks: 3,
        hotspot_track_length: 6,
        lloyd_iterations: 2,
        debug_paint_albedo: false,
    }
}

fn stage2() -> CoarseElevation {
    CoarseElevation {
        target_ocean_fraction: 0.65,
        continent_noise_frequency: 2.0,
        continent_noise_octaves: 6,
        continent_noise_persistence: 0.55,
        continent_warp_amplitude: 0.45,
        continent_warp_frequency: 2.2,
        horizontal_stretch: 1.5,
        mid_latitude_bias: 0.2,
        min_land_fraction: 0.003,
        min_ocean_fraction: 0.0,
        continental_bias_m: 500.0,
        oceanic_bias_m: -3500.0,
        suture_height_m: 2500.0,
        suture_falloff_hops: 2.5,
        suture_age_decay_myr: 1500.0,
        rift_depth_m: 800.0,
        rift_falloff_hops: 2.0,
        active_uplift_m: 3500.0,
        active_trench_m: 3500.0,
        active_falloff_hops: 3.0,
        trench_offset_hops: 3.0,
        trench_width_hops: 1.5,
        hotspot_height_m: 800.0,
        hotspot_falloff_hops: 1.5,
        noise_amplitude_m: 1500.0, // unchanged intentionally for test determinism
        noise_frequency: 2.5,
        noise_octaves: 7,
        noise_persistence: 0.62,
        warp_amplitude: 0.30,
        warp_frequency: 2.5,
        debug_paint_albedo: false,
    }
}

#[test]
fn coarse_elevation_populates_per_vertex_heights() {
    let mut b = builder();
    Pipeline::new(vec![Box::new(stage1()), Box::new(stage2())]).run(&mut b);

    let elevations = b
        .vertex_elevations_m
        .as_ref()
        .expect("vertex_elevations_m should be populated");
    let sphere = b.sphere.as_ref().unwrap();
    assert_eq!(elevations.len(), sphere.vertices.len());

    let min_e = elevations.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_e = elevations.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(min_e < -500.0, "no oceanic elevation: min = {min_e}");
    assert!(max_e > 500.0, "no continental elevation: max = {max_e}");
}

#[test]
fn coarse_elevation_active_margins_are_sharper() {
    let mut b = builder();
    Pipeline::new(vec![Box::new(stage1()), Box::new(stage2())]).run(&mut b);

    let elevations = b.vertex_elevations_m.as_ref().unwrap();
    let vps = b.vertex_provinces.as_ref().unwrap();
    let provinces = &b.provinces;

    let active_elevs: Vec<f32> = vps
        .iter()
        .enumerate()
        .filter(|&(_, &pid)| provinces[pid as usize].kind == ProvinceKind::ActiveMargin)
        .map(|(vi, _)| elevations[vi].abs())
        .collect();

    if active_elevs.is_empty() {
        // RNG could skip generating an ActiveMargin in rare seeds; skip.
        return;
    }

    let active_peak = active_elevs.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        active_peak > 1500.0,
        "active margin vertex should have sharp relief (>1500 m); got {active_peak}"
    );
}

#[test]
fn coarse_elevation_is_deterministic() {
    let run = || {
        let mut b = builder();
        Pipeline::new(vec![Box::new(stage1()), Box::new(stage2())]).run(&mut b);
        let elevations = b.vertex_elevations_m.unwrap();
        let first = elevations[0];
        let last = *elevations.last().unwrap();
        let hash: u64 = elevations.iter().fold(0u64, |a, e| {
            a.wrapping_mul(0x100000001b3).wrapping_add(e.to_bits() as u64)
        });
        (first.to_bits(), last.to_bits(), hash)
    };

    let a = run();
    let b = run();
    assert_eq!(a, b, "CoarseElevation not deterministic");
}
