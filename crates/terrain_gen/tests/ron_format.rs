//! Verifies the RON shape we expect for stage param structs and StageDef.
//!
//! These tests pin the file format. If they break, body files break too.

use thalos_terrain_gen::{GeneratorParams, StageDef};

const MIRA_GENERATOR: &str = r#"(
    seed: 1004,
    composition: (silicate: 0.95, iron: 0.05, ice: 0.0, volatiles: 0.0, hydrogen_helium: 0.0),
    cubemap_resolution: 1024,
    body_age_gyr: 4.5,
    pipeline: [
        Differentiate(()),
        Megabasin((
            basins: [
                (center_dir: ( 0.25,  0.35, 0.90), radius_m: 320000.0, depth_m: 7500.0, ring_count: 4),
                (center_dir: (-0.35, -0.30, 0.85), radius_m: 220000.0, depth_m: 5200.0, ring_count: 3),
            ],
            hemispheric_lowering_m: 2000.0,
        )),
        Cratering((
            total_count: 10000,
            sfd_slope: 2.0,
            min_radius_m: 1500.0,
            max_radius_m: 250000.0,
            cubemap_bake_threshold_m: 10000.0,
        )),
        MareFlood((
            target_count: 5,
            additional_crater_count: 0,
            fill_fraction: 0.7,
            near_side_bias: 0.9,
            boundary_noise_amplitude_m: 500.0,
            boundary_noise_freq: 3.0,
            episode_count: 3,
            wrinkle_ridges: true,
        )),
        Regolith((
            amplitude_m: 3.0,
            characteristic_wavelength_m: 50.0,
            crater_density_multiplier: 1.0,
        )),
        SpaceWeather((
            highland_mature_albedo: 0.10,
            highland_fresh_albedo: 0.18,
            mare_mature_albedo: 0.065,
            mare_fresh_albedo: 0.09,
            young_crater_age_threshold: 0.5,
            ray_age_threshold: 0.1,
            ray_extent_radii: 7.0,
            ray_count_per_crater: 8,
            ray_half_width: 0.06,
        )),
    ],
)"#;

#[test]
fn mira_generator_ron_parses() {
    let parsed: GeneratorParams =
        ron::from_str(MIRA_GENERATOR).expect("Mira generator RON failed to parse");
    assert_eq!(parsed.seed, 1004);
    assert_eq!(parsed.cubemap_resolution, 1024);
    assert!((parsed.body_age_gyr - 4.5).abs() < 1e-6);
    assert_eq!(parsed.pipeline.len(), 6);
    assert!(matches!(parsed.pipeline[0], StageDef::Differentiate(_)));
    assert!(matches!(parsed.pipeline[1], StageDef::Megabasin(_)));
    assert!(matches!(parsed.pipeline[5], StageDef::SpaceWeather(_)));
}
