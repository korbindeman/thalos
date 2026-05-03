//! Round-trip test for the `cache` module: compile a tiny ocean body,
//! store it, load it, and confirm the key+payload survive.

use glam::Vec3;
use thalos_terrain_gen::{
    OceanTerrainConfig, TerrainCompileContext, TerrainCompileOptions, TerrainConfig, cache,
    compile_terrain_config,
};

fn tiny_terrain() -> TerrainConfig {
    TerrainConfig::Ocean(OceanTerrainConfig {
        seed: 1234,
        cubemap_resolution: 16,
        seabed_albedo: [0.02, 0.05, 0.10],
        water_roughness: 0.04,
        sea_level_m: 1.0,
    })
}

fn tiny_context() -> TerrainCompileContext {
    TerrainCompileContext {
        body_name: "TestBody".to_string(),
        radius_m: 100_000.0,
        gravity_m_s2: 1.5,
        rotation_hours: None,
        obliquity_deg: Some(5.0),
        tidal_axis: Some(Vec3::Z),
        axial_tilt_rad: 0.1,
    }
}

#[test]
fn cache_roundtrip_preserves_cubemaps() {
    let tmp = std::env::temp_dir().join("thalos_terrain_gen_cache_roundtrip");
    let _ = std::fs::remove_dir_all(&tmp);

    let terrain = tiny_terrain();
    let context = tiny_context();
    let options = TerrainCompileOptions::default();

    let original = compile_terrain_config(&terrain, &context, options).expect("compile");
    let key = cache::terrain_cache_key(&terrain, &context, options);
    let path = cache::cache_path(&tmp, "TestBody", key);

    cache::store(&path, key, &original).expect("store");
    let loaded = cache::load(&path, key).expect("load");

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
    assert_eq!(loaded.sea_level_m, original.sea_level_m);

    assert!(cache::load(&path, key ^ 0xDEAD).is_none());

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn terrain_cache_key_tracks_compile_inputs() {
    let terrain = tiny_terrain();
    let mut context = tiny_context();

    let dev_options = TerrainCompileOptions {
        crater_count_scale: 0.1,
    };
    let release_options = TerrainCompileOptions {
        crater_count_scale: 1.0,
    };

    let base = cache::terrain_cache_key(&terrain, &context, dev_options);
    assert_ne!(
        base,
        cache::terrain_cache_key(&terrain, &context, release_options)
    );

    context.gravity_m_s2 = 2.0;
    assert_ne!(
        base,
        cache::terrain_cache_key(&terrain, &context, dev_options)
    );
}
