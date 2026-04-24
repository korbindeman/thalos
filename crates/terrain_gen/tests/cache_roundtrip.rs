//! Round-trip test for the `cache` module: build a small `BodyData`,
//! store it, load it, and confirm the key+payload survive.

use glam::Vec3;
use thalos_terrain_gen::{
    BodyBuilder, Composition, Differentiate, Pipeline, cache,
    generator::{GeneratorParams, StageDef},
};

fn tiny_params() -> GeneratorParams {
    GeneratorParams {
        seed: 1234,
        composition: Composition::new(0.85, 0.10, 0.0, 0.05, 0.0),
        cubemap_resolution: 16,
        body_age_gyr: 4.0,
        pipeline: vec![StageDef::Differentiate(Differentiate)],
    }
}

#[test]
fn cache_roundtrip_preserves_cubemaps() {
    let tmp = std::env::temp_dir().join("thalos_terrain_gen_cache_roundtrip");
    let _ = std::fs::remove_dir_all(&tmp);

    let params = tiny_params();
    let radius_m = 100_000.0f32;
    let tidal = Some(Vec3::Z);
    let tilt = 0.1;

    // Build a BodyData using the same two stages.
    let mut builder = BodyBuilder::new(
        radius_m,
        params.seed,
        params.composition,
        params.cubemap_resolution,
        params.body_age_gyr,
        tidal,
        tilt,
    );
    let stages = params
        .pipeline
        .clone()
        .into_iter()
        .map(|s| s.into_stage())
        .collect::<Vec<_>>();
    Pipeline::new(stages).run(&mut builder);
    let original = builder.build();

    let key = cache::cache_key(&params, radius_m, tidal, tilt);
    let path = cache::cache_path(&tmp, "TestBody", key);

    cache::store(&path, key, &original).expect("store");
    let loaded = cache::load(&path, key).expect("load");

    // Compare cubemap bytes.
    assert_eq!(
        loaded.height_cubemap.resolution(),
        original.height_cubemap.resolution()
    );
    for face in thalos_terrain_gen::cubemap::CubemapFace::ALL {
        assert_eq!(
            loaded.height_cubemap.face_data(face),
            original.height_cubemap.face_data(face),
            "height face {face:?}"
        );
        assert_eq!(
            loaded.albedo_cubemap.face_data(face),
            original.albedo_cubemap.face_data(face),
            "albedo face {face:?}"
        );
        assert_eq!(
            loaded.material_cubemap.face_data(face),
            original.material_cubemap.face_data(face),
            "material face {face:?}"
        );
    }
    assert_eq!(loaded.radius_m, original.radius_m);
    assert_eq!(loaded.height_range, original.height_range);

    // Wrong key → miss.
    assert!(cache::load(&path, key ^ 0xDEAD).is_none());

    let _ = std::fs::remove_dir_all(&tmp);
}
