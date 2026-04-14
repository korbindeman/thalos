//! Diagnostic bench for the Mira pipeline, honoring the current dev profile
//! (per-package overrides apply). Prints per-stage wall time.
//!
//! Run with `cargo run --example bake_bench -p thalos_terrain_gen`.

use glam::Vec3;
use std::time::Instant;
use thalos_terrain_gen::*;
#[allow(unused_imports)]
use thalos_terrain_gen::Stage;

fn main() {
    let resolution: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let composition = Composition::new(0.85, 0.10, 0.0, 0.05, 0.0);

    let mut builder = BodyBuilder::new(
        869_000.0,
        42,
        composition,
        resolution,
        4.5,
        Some(Vec3::Z),
        0.0,
    );

    let stages: Vec<Box<dyn Stage>> = vec![
        Box::new(Differentiate),
        Box::new(Megabasin {
            basins: vec![
                BasinDef {
                    center_dir: Vec3::new(0.25, 0.35, 0.90).normalize(),
                    radius_m: 320_000.0,
                    depth_m: 7_500.0,
                    ring_count: 4,
                },
                BasinDef {
                    center_dir: Vec3::new(-0.35, -0.30, 0.85).normalize(),
                    radius_m: 220_000.0,
                    depth_m: 5_200.0,
                    ring_count: 3,
                },
            ],
            hemispheric_lowering_m: 2_000.0,
        }),
        Box::new(Cratering {
            total_count: 50_000,
            sfd_slope: 1.8,
            sfd_slope_small: Some(3.3),
            sfd_break_radius_m: Some(1_000.0),
            min_radius_m: 400.0,
            max_radius_m: 500_000.0,
            age_bias: 2.0,
            cubemap_bake_threshold_m: 2_500.0,
            secondary_parent_radius_m: 50_000.0,
            secondaries_per_parent: 25,
            saturation_fraction: 0.05,
            chain_count: 3,
            chain_segment_count: 10,
        }),
        Box::new(MareFlood {
            target_count: 5,
            additional_crater_count: 0,
            fill_fraction: 0.7,
            near_side_bias: 0.9,
            boundary_noise_amplitude_m: 700.0,
            boundary_noise_freq: 4.0,
            episode_count: 3,
            wrinkle_ridges: true,
            procellarum: None,
        }),
        Box::new(Regolith {
            amplitude_m: 3.0,
            characteristic_wavelength_m: 50.0,
            crater_density_multiplier: 1.0,
            bake_d_min_m: 600.0,
            bake_scale: 1.0,
            density_modulation: 0.25,
            density_wavelength_m: 250_000.0,
        }),
        Box::new(SpaceWeather {
            highland_mature_albedo: Some(0.10),
            highland_fresh_albedo: Some(0.18),
            mare_mature_albedo: 0.065,
            mare_fresh_albedo: 0.09,
            young_crater_age_threshold: 0.5,
            ray_age_threshold: 0.1,
            ray_extent_radii: 7.0,
            ray_count_per_crater: 8,
            ray_half_width: 0.06,
        }),
    ];

    println!("-- mira bake @ {}x{} --", resolution, resolution);
    let total = Instant::now();
    Pipeline::new(stages).run(&mut builder);
    let _ = builder.build();
    println!("  {:>14}  {:>8.2} s", "TOTAL", total.elapsed().as_secs_f32());
}
