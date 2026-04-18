//! Smoke test for the Thalos MVP pipeline.
//!
//! Runs Differentiate → Plates → Tectonics → Topography → Biomes at a tiny
//! cubemap resolution (so tests stay fast) and checks the resulting
//! artefacts: plate map populated with boundaries, height field has both
//! continental and oceanic cells, biome map splits across all three
//! biomes, and re-running produces identical state (determinism).

use glam::Vec3;
use thalos_terrain_gen::{
    BiomeParams, BiomeRule, BodyBuilder, Composition, CubemapFace, Differentiate, OrogenDla,
    Pipeline, Plates, Tectonics, Topography,
};

fn thalos_composition() -> Composition {
    Composition::new(0.26, 0.70, 0.0, 0.04, 0.0)
}

fn thalos_builder(resolution: u32) -> BodyBuilder {
    BodyBuilder::new(
        3_186_000.0,
        1003,
        thalos_composition(),
        resolution,
        4.5,
        None,
        23.0_f32.to_radians(),
    )
}

fn thalos_pipeline() -> Pipeline {
    Pipeline::new(vec![
        Box::new(Differentiate),
        Box::new(Plates {
            n_plates: 40,
            continental_area_fraction: 0.35,
            n_continental_seeds: 4,
            neighbour_continental_bias: 0.5,
            lloyd_iterations: 3,
        }),
        Box::new(Tectonics {
            active_boundary_fraction: 0.20,
            orogen_age_peak_gyr: 2.0,
        }),
        Box::new(OrogenDla {
            // Lower subdivision at test resolution so the DLA graph
            // matches the smaller cubemap density. Level 4 → 2562 verts
            // (~200 km spacing on Thalos); plenty for smoke coverage.
            subdivision_level: 4,
            seed_density: 4.0,
            walkers_per_seed: 100,
            walker_step_budget: 200,
            launch_cap_km: 500.0,
            ridge_falloff_km: 40.0,
            depth_saturation: 4.0,
            samples_per_edge: 4,
            midpoint_jitter_frac: 0.3,
        }),
        Box::new(Topography {
            continental_baseline_m: 500.0,
            oceanic_baseline_m: -3500.0,
            continentalness_bandwidth: 0.02,
            peak_orogen_m: 7000.0,
            roughness_m: 700.0,
            orogen_age_scale_myr: 1500.0,
            noise_frequency: 2.5,
            warp_amplitude: 0.16,
            warp_frequency: 1.8,
            warp_amplitude_mid: 0.10,
            warp_frequency_mid: 5.0,
            warp_amplitude_hi: 0.04,
            warp_frequency_hi: 14.0,
            regional_height_lo_m: 2000.0,
            regional_frequency_lo: 0.5,
            regional_height_m: 1500.0,
            regional_frequency: 1.5,
            regional_height_hi_m: 600.0,
            regional_frequency_hi: 4.0,
            coastal_detail_m: 140.0,
            coastal_detail_frequency: 18.0,
            coastal_detail_scale_m: 800.0,
            sea_level_percentile: 0.55,
        }),
        // Climate-driven ice/snow coverage is now applied by PaintBiomes
        // as a continuous overlay — not a discrete biome — so the test
        // pipeline doesn't need sea_ice / land_ice classes. The smoke
        // test just checks that continental and oceanic cells land on
        // different biome ids, which is the only invariant the rest of
        // the stage chain cares about.
        Box::new(thalos_terrain_gen::Biomes {
            biomes: vec![
                BiomeParams {
                    name: "ocean".to_string(),
                    albedo: 0.06,
                    fresh_albedo: Some(0.10),
                    tint: [0.35, 0.55, 1.10],
                    tonal_amp: 0.12,
                    roughness: 0.05,
                },
                BiomeParams {
                    name: "continent".to_string(),
                    albedo: 0.20,
                    fresh_albedo: Some(0.28),
                    tint: [1.05, 0.95, 0.75],
                    tonal_amp: 0.30,
                    roughness: 0.85,
                },
            ],
            rules: vec![
                BiomeRule::HeightBelow {
                    m: 0.0,
                    jitter_amp_m: 0.0,
                    jitter_frequency: 3.0,
                },
                BiomeRule::Default,
            ],
        }),
    ])
}

#[test]
fn thalos_mvp_pipeline_runs_end_to_end() {
    let mut builder = thalos_builder(128);
    thalos_pipeline().run(&mut builder);

    // Plates populated by Plates stage.
    let pm = builder
        .plates
        .as_ref()
        .expect("Plates stage should populate builder.plates");
    assert_eq!(pm.plates.len(), 40);
    // Tectonics fills in the boundary list.
    assert!(
        !pm.boundaries.is_empty(),
        "Tectonics should have produced some boundaries"
    );

    // Continental / oceanic split is close to the requested 35%.
    let continental_count = pm
        .plates
        .iter()
        .filter(|p| matches!(p.kind, thalos_terrain_gen::PlateKind::Continental))
        .count();
    assert!(
        (2..=20).contains(&continental_count),
        "unexpected continental plate count: {continental_count}"
    );

    // Height field has meaningful variation: peaks above sea level and
    // deep ocean basins below.
    let mut min_h = f32::MAX;
    let mut max_h = f32::MIN;
    for face in CubemapFace::ALL {
        for &v in builder.height_contributions.height.face_data(face) {
            min_h = min_h.min(v);
            max_h = max_h.max(v);
        }
    }
    assert!(min_h < -1000.0, "no deep ocean: min_h = {min_h}");
    assert!(max_h > 500.0, "no land above sea: max_h = {max_h}");

    // Biome map has ocean and continent slots represented. Ice and snow
    // are now applied by PaintBiomes as continuous coverage, not biomes.
    let mut counts = [0u32; 2];
    for face in CubemapFace::ALL {
        for &v in builder.biome_map.face_data(face) {
            if (v as usize) < counts.len() {
                counts[v as usize] += 1;
            }
        }
    }
    assert!(counts[0] > 0, "no ocean biome");
    assert!(counts[1] > 0, "no continent biome");
}

#[test]
fn thalos_pipeline_is_deterministic() {
    let run = || {
        let mut builder = thalos_builder(64);
        thalos_pipeline().run(&mut builder);
        let h = builder
            .height_contributions
            .height
            .get(CubemapFace::PosX, 20, 20);
        let plate = builder
            .plates
            .as_ref()
            .unwrap()
            .plate_id_cubemap
            .get(CubemapFace::PosY, 10, 10);
        let biome = builder.biome_map.get(CubemapFace::NegZ, 30, 30);
        // Sum the whole heightfield so this test catches non-determinism
        // anywhere on the sphere, not just at the three sampled points. A
        // past incident slipped through because hashmap-iteration-order
        // non-determinism in Tectonics varied `orogen_intensity` across
        // runs, but the specific sampled cells happened to fall in
        // regions where the variation was small.
        let height_hash: u64 = CubemapFace::ALL
            .iter()
            .flat_map(|f| builder.height_contributions.height.face_data(*f).iter())
            .fold(0u64, |a, h| {
                a.wrapping_mul(0x100000001b3)
                    .wrapping_add(h.to_bits() as u64)
            });
        (h, plate, biome, height_hash)
    };

    let a = run();
    let b = run();
    assert_eq!(a, b, "Thalos pipeline not deterministic");
}

/// Release-mode bake timing at Thalos's production cubemap resolution.
/// `#[ignore]`d so it doesn't slow down CI; run with:
///   `cargo test --release -p thalos_terrain_gen --test thalos_pipeline -- \
///    --include-ignored thalos_full_resolution_bake`.
#[test]
#[ignore = "long-running; run with --include-ignored to measure bake time"]
fn thalos_full_resolution_bake() {
    use std::time::Instant;

    let mut builder = thalos_builder(2048);

    let start = Instant::now();
    thalos_pipeline().run(&mut builder);
    let run_elapsed = start.elapsed();

    let build_start = Instant::now();
    let body = builder.build();
    let build_elapsed = build_start.elapsed();

    println!(
        "\n-- thalos bake @ 2048² --\n  pipeline:  {:>6.2} s\n  finalize:  {:>6.2} s\n  total:     {:>6.2} s\n  height_range: {} m\n",
        run_elapsed.as_secs_f32(),
        build_elapsed.as_secs_f32(),
        (run_elapsed + build_elapsed).as_secs_f32(),
        body.height_range,
    );

    assert!(body.height_range > 1000.0);
}

#[test]
fn thalos_ocean_floor_biased_below_sea_level() {
    // ~65% of the surface should be below sea level (the Topography
    // baseline + roughness hits this, modulo plate-count RNG variance).
    let mut builder = thalos_builder(128);
    thalos_pipeline().run(&mut builder);

    let mut below = 0u32;
    let mut total = 0u32;
    for face in CubemapFace::ALL {
        for &v in builder.height_contributions.height.face_data(face) {
            total += 1;
            if v < 0.0 {
                below += 1;
            }
        }
    }
    let frac = below as f32 / total as f32;
    assert!(
        (0.50..=0.80).contains(&frac),
        "ocean fraction out of range: {frac}"
    );

    // Keep reference unused warning away.
    let _ = Vec3::X;
}
