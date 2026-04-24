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
            highland_mature_albedo: Some(0.10),
            highland_fresh_albedo: Some(0.18),
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

const THALOS_GENERATOR: &str = r#"(
    seed: 1003,
    composition: (silicate: 0.26, iron: 0.70, ice: 0.0, volatiles: 0.04, hydrogen_helium: 0.0),
    cubemap_resolution: 2048,
    body_age_gyr: 4.5,
    pipeline: [
        TectonicSkeleton((
            subdivision_level: 8,
            n_cratons: 20,
            plate_warp_amplitude: 0.4,
            plate_warp_frequency: 3.0,
            plate_warp_octaves: 5,
            n_active_margins: 1,
            n_rift_scars: 2,
            n_hotspot_tracks: 4,
            hotspot_track_length: 12,
            lloyd_iterations: 3,
            debug_paint_albedo: false,
        )),
        CoarseElevation((
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
            noise_amplitude_m: 2500.0,
            noise_frequency: 2.5,
            noise_octaves: 7,
            noise_persistence: 0.62,
            warp_amplitude: 0.45,
            warp_frequency: 3.0,
            debug_paint_albedo: true,
        )),
        HydrologicalCarving((
            erosion_iterations: 100,
            erosion_k: 2.0e-4,
            stream_power_m: 0.5,
            stream_power_n: 1.0,
            max_erosion_per_iter_m: 50.0,
            deposition_rate: 0.08,
            pit_fill_epsilon_m: 0.001,
            target_ocean_fraction: 0.65,
            river_accumulation_threshold_m2: 2.0e11,
            river_half_width_rad: 0.0008,
            debug_paint_albedo: false,
        )),
        SurfaceMaterials((
            shelf_depth_m: 200.0,
            floodplain_log10_accumulation_m2: 9.7,
            floodplain_min_sediment_m: 0.5,
            floodplain_max_elevation_m: 1500.0,
            bare_rock_min_elevation_m: 2500.0,
            bare_rock_max_sediment_m: 0.3,
            ancient_stable_max_sediment_m: 5.0,
            fresh_volcanic_max_age_myr: 30.0,
            iron_source_active_margin: 1.0,
            iron_source_hotspot_track: 1.0,
            iron_source_suture: 0.3,
            iron_fraction_threshold: 0.01,
            iron_fraction_bif_threshold: 0.03,
            debug_paint_albedo: true,
        )),
    ],
)"#;

#[test]
fn thalos_generator_ron_parses() {
    let parsed: GeneratorParams =
        ron::from_str(THALOS_GENERATOR).expect("Thalos generator RON failed to parse");
    assert_eq!(parsed.seed, 1003);
    assert_eq!(parsed.pipeline.len(), 4);
    assert!(matches!(parsed.pipeline[0], StageDef::TectonicSkeleton(_)));
    assert!(matches!(parsed.pipeline[1], StageDef::CoarseElevation(_)));
    assert!(matches!(parsed.pipeline[2], StageDef::HydrologicalCarving(_)));
    assert!(matches!(parsed.pipeline[3], StageDef::SurfaceMaterials(_)));
}
