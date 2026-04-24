//! Smoke test for `SurfaceMaterials` (Stage 5 of the Thalos pipeline).
//!
//! Runs TectonicSkeleton → CoarseElevation → HydrologicalCarving →
//! SurfaceMaterials at small subdivision and checks the stage's output
//! contract: the materials palette is populated, the material cubemap
//! has multiple distinct values, at least a few of the key categories
//! (ocean, continental) show up, and re-running yields identical
//! assignments.

use thalos_terrain_gen::{
    BodyBuilder, CoarseElevation, Composition, CubemapFace, HydrologicalCarving,
    MAT_ABYSSAL_SEABED, MAT_CONTINENTAL_REGOLITH, MAT_WEATHERED_GRANITE, Pipeline, SurfaceMaterials,
    TectonicSkeleton,
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
        noise_amplitude_m: 1500.0,
        noise_frequency: 2.5,
        noise_octaves: 7,
        noise_persistence: 0.62,
        warp_amplitude: 0.30,
        warp_frequency: 2.5,
        debug_paint_albedo: false,
    }
}

fn stage3() -> HydrologicalCarving {
    HydrologicalCarving {
        // Light erosion budget keeps the test fast without changing
        // the output contract (vertex elevations, drainage graph,
        // sediment map, sea level all populated).
        erosion_iterations: 20,
        erosion_k: 2.0e-4,
        stream_power_m: 0.5,
        stream_power_n: 1.0,
        max_erosion_per_iter_m: 50.0,
        deposition_rate: 0.08,
        pit_fill_epsilon_m: 0.001,
        target_ocean_fraction: 0.65,
        river_accumulation_threshold_m2: 1.0e10,
        river_half_width_rad: 0.0015,
        debug_paint_albedo: false,
    }
}

fn stage5() -> SurfaceMaterials {
    SurfaceMaterials {
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
        debug_paint_albedo: false,
    }
}

fn run_full_pipeline(b: &mut BodyBuilder) {
    Pipeline::new(vec![
        Box::new(stage1()),
        Box::new(stage2()),
        Box::new(stage3()),
        Box::new(stage5()),
    ])
    .run(b);
}

#[test]
fn surface_materials_registers_palette() {
    let mut b = builder();
    run_full_pipeline(&mut b);

    assert_eq!(
        b.materials.len(),
        10,
        "surface materials palette should have 10 entries"
    );
    // Albedo values must be finite and in [0, 1] range for every
    // registered material — anything else silently breaks downstream
    // sRGB bake.
    for (i, m) in b.materials.iter().enumerate() {
        for (c, v) in m.albedo.iter().enumerate() {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(v),
                "material {i} albedo channel {c} out of range: {v}"
            );
        }
        assert!(
            m.roughness >= 0.0 && m.roughness <= 1.0,
            "material {i} roughness out of range: {}",
            m.roughness
        );
    }
}

#[test]
fn surface_materials_paints_cubemap_with_multiple_ids() {
    let mut b = builder();
    run_full_pipeline(&mut b);

    let mut seen = std::collections::BTreeSet::new();
    for face in CubemapFace::ALL {
        for &id in b.material_cubemap.face_data(face) {
            seen.insert(id);
        }
    }
    assert!(
        seen.len() >= 3,
        "material cubemap should paint at least 3 distinct IDs, saw {seen:?}"
    );

    // Abyssal seabed is by far the majority class after Stage 3 fills
    // 65 % of the surface with deep ocean; it should always show up.
    assert!(
        seen.contains(&MAT_ABYSSAL_SEABED),
        "expected abyssal seabed to be painted somewhere"
    );

    // At least one of the two continental-default IDs should show up —
    // weathered granite (Craton or aged Suture) or regolith (default).
    assert!(
        seen.contains(&MAT_CONTINENTAL_REGOLITH) || seen.contains(&MAT_WEATHERED_GRANITE),
        "expected at least one continental default material, saw {seen:?}"
    );
}

#[test]
fn surface_materials_is_deterministic() {
    let run = || {
        let mut b = builder();
        run_full_pipeline(&mut b);
        // Hash the material cubemap and the palette so we catch any
        // non-determinism in either the assignment or the palette
        // registration.
        let mut hash: u64 = 0;
        for face in CubemapFace::ALL {
            for &id in b.material_cubemap.face_data(face) {
                hash = hash.wrapping_mul(0x100000001b3).wrapping_add(id as u64);
            }
        }
        let palette_hash: u64 = b.materials.iter().fold(0u64, |acc, m| {
            let mut h = acc;
            for c in &m.albedo {
                h = h.wrapping_mul(0x100000001b3).wrapping_add(c.to_bits() as u64);
            }
            h.wrapping_mul(0x100000001b3)
                .wrapping_add(m.roughness.to_bits() as u64)
        });
        (hash, palette_hash)
    };

    let a = run();
    let b = run();
    assert_eq!(a, b, "SurfaceMaterials not deterministic");
}
