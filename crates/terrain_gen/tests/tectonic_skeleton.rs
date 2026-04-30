//! Smoke test for `TectonicSkeleton` (Stage 1 of the Thalos pipeline).
//!
//! Runs the stage at a small subdivision level and checks that it
//! produces a province table, per-vertex assignments, and representative
//! coverage of the boundary province kinds.

use thalos_terrain_gen::{
    BodyBuilder, Composition, Pipeline, ProvinceKind, TectonicSkeleton, province::PROVINCE_NONE,
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

fn stage() -> TectonicSkeleton {
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

#[test]
fn tectonic_skeleton_produces_province_map() {
    let mut b = builder();
    Pipeline::new(vec![Box::new(stage())]).run(&mut b);

    let sphere = b.sphere.expect("sphere should be populated");
    let vertex_provinces = b
        .vertex_provinces
        .expect("per-vertex provinces should be populated");

    assert_eq!(vertex_provinces.len(), sphere.vertices.len());
    assert!(
        !b.provinces.is_empty(),
        "province table should be non-empty"
    );
    // Every vertex must be assigned to a real province.
    for &pid in &vertex_provinces {
        assert_ne!(pid, PROVINCE_NONE);
        assert!((pid as usize) < b.provinces.len());
    }
}

#[test]
fn tectonic_skeleton_produces_representative_kinds() {
    let mut b = builder();
    Pipeline::new(vec![Box::new(stage())]).run(&mut b);

    let kinds: std::collections::HashSet<ProvinceKind> =
        b.provinces.iter().map(|p| p.kind).collect();

    // Plate interiors are Craton. Boundaries split across the three
    // requested kinds; hotspots added separately.
    assert!(kinds.contains(&ProvinceKind::Craton));
    assert!(kinds.contains(&ProvinceKind::Suture));
    assert!(kinds.contains(&ProvinceKind::RiftScar));
    assert!(kinds.contains(&ProvinceKind::ActiveMargin));
}

#[test]
fn tectonic_skeleton_is_deterministic() {
    let run = || {
        let mut b = builder();
        Pipeline::new(vec![Box::new(stage())]).run(&mut b);
        let vps = b.vertex_provinces.unwrap();
        let first = vps[0];
        let last = *vps.last().unwrap();
        let hash: u64 = vps.iter().fold(0u64, |a, &p| {
            a.wrapping_mul(0x100000001b3).wrapping_add(p as u64)
        });
        let n_provinces = b.provinces.len();
        (first, last, hash, n_provinces)
    };

    let a = run();
    let b = run();
    assert_eq!(a, b, "TectonicSkeleton not deterministic");
}
