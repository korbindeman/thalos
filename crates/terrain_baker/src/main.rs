//! Offline terrain-package baker.
//!
//! This is the first producer for ADR-0008's package contract. The current MVP
//! producer is the deterministic airless feature compiler; the expensive
//! hierarchical diffusion producer will replace this producer without changing
//! the package reader or the game's `SurfaceQuery` consumers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::Vec3;
use thalos_terrain::{
    CubemapFace, PackageNodeAddress, PackageProducer, TerrainCompileContext, TerrainCompileOptions,
    TerrainPackageManifest, cache, compile_terrain_config, load_static_package,
    write_static_package,
};
use thalos_world::{BodyDefinition, BodyKind, parsing::load_solar_system_from_dir};

fn context_for(body: &BodyDefinition) -> TerrainCompileContext {
    TerrainCompileContext {
        body_name: body.name.clone(),
        radius_m: body.radius_m as f32,
        gravity_m_s2: body.surface_gravity_m_s2() as f32,
        rotation_hours: (body.rotation_period_s > 0.0)
            .then_some((body.rotation_period_s / 3600.0) as f32),
        obliquity_deg: Some(body.axial_tilt_rad.to_degrees() as f32),
        tidal_axis: matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z),
        axial_tilt_rad: body.axial_tilt_rad as f32,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let first = args.next().unwrap_or_else(|| "Mira".into());
    let (validate_only, body_name) = if first == "validate" || first == "--validate" {
        (true, args.next().unwrap_or_else(|| "Mira".into()))
    } else {
        (false, first)
    };
    let assets = Path::new("assets");
    let system = load_solar_system_from_dir(assets).expect("load solar-system assets");
    let body_id = system
        .name_to_id
        .get(&body_name)
        .copied()
        .unwrap_or_else(|| panic!("unknown body {body_name:?}"));
    let body = &system.bodies[body_id];
    assert!(
        body.terrain.is_some(),
        "{} has no terrain config",
        body.name
    );

    let context = context_for(body);
    let options = TerrainCompileOptions::default();
    let key = cache::terrain_cache_key(&body.terrain, body.tectonics.as_ref(), &context, options);
    let out = PathBuf::from("assets/terrain_packages").join(format!("{}.bin", body.name));

    if validate_only {
        let started = Instant::now();
        let loaded = load_static_package(&out, &body.name, key)
            .unwrap_or_else(|error| panic!("validate {}: {error}", out.display()));
        let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
        println!(
            "Validated {}: schema v{}, producer {} {}, key {:016x}, {:.1} MiB, {} nodes / {} blobs in {:.2?}",
            out.display(),
            loaded.manifest.schema_version,
            loaded.manifest.producer.name,
            loaded.manifest.producer.version,
            loaded.manifest.content_key,
            bytes as f64 / (1024.0 * 1024.0),
            loaded.manifest.nodes.len(),
            loaded.manifest.blobs.len(),
            started.elapsed(),
        );
        print_height_stats(&loaded.manifest);
        return;
    }

    println!("Baking {} → {}", body.name, out.display());
    println!("  package key: {key:016x}");
    let started = Instant::now();
    let mut surface = compile_terrain_config(
        &body.terrain,
        body.tectonics.as_ref(),
        &context,
        options,
        None,
    )
    .unwrap_or_else(|error| panic!("bake {}: {error}", body.name));
    let producer = PackageProducer {
        name: "thalos-airless-compat".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        model_hash: None,
    };
    let manifest =
        write_static_package(&out, &body.name, key, producer, &mut surface.static_surface)
            .unwrap_or_else(|error| panic!("write {}: {error}", out.display()));
    let loaded = load_static_package(&out, &body.name, key)
        .unwrap_or_else(|error| panic!("validate {}: {error}", out.display()));
    assert_eq!(loaded.manifest.content_key, manifest.content_key);
    let metres_per_unit = surface.static_surface.height_range * 2.0 / f32::from(u16::MAX);
    let mut max_error_m = 0.0f32;
    let mut sum_squared_error_m = 0.0f64;
    let mut sample_count = 0usize;
    for face in CubemapFace::ALL {
        for (source, decoded) in surface
            .static_surface
            .height_cubemap
            .face_data(face)
            .iter()
            .zip(loaded.static_surface.height_cubemap.face_data(face))
        {
            let error_m = source.abs_diff(*decoded) as f32 * metres_per_unit;
            max_error_m = max_error_m.max(error_m);
            sum_squared_error_m += f64::from(error_m * error_m);
            sample_count += 1;
        }
    }
    let rms_error_m = (sum_squared_error_m / sample_count as f64).sqrt();
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "  done in {:.2?}: schema v{}, {:.1} MiB, {} indexed craters, ±{:.0} m",
        started.elapsed(),
        manifest.schema_version,
        bytes as f64 / (1024.0 * 1024.0),
        surface.static_surface.craters.len(),
        surface.static_surface.height_range,
    );
    println!(
        "  reconstruction: max {max_error_m:.3} m, RMS {rms_error_m:.3} m, artifact {:016x}",
        loaded.manifest.artifact_fingerprint(),
    );
    print_height_stats(&loaded.manifest);
}

fn print_height_stats(manifest: &TerrainPackageManifest) {
    let mut levels = BTreeMap::<u8, (usize, usize, u64, f32)>::new();
    let mut predictor_errors = BTreeMap::<u8, Vec<f32>>::new();
    let mut flat = (0usize, 0usize);
    let mut rough = (0usize, 0usize);
    for node in &manifest.nodes {
        let PackageNodeAddress::Cube { lod, .. } = node.address else {
            continue;
        };
        let entry = levels.entry(lod).or_default();
        entry.0 += 1;
        entry.3 = entry.3.max(node.geometric_error_m);
        if let Some(blob_index) = node.blob_index {
            entry.1 += 1;
            entry.2 += manifest.blobs[blob_index as usize].encoded_len;
        }
        if lod > 0 {
            predictor_errors
                .entry(lod)
                .or_default()
                .push(node.predictor_error_m);
        }
        if lod > 0 && node.predictor_error_m <= manifest.height_pyramid.max_fallback_error_m {
            flat.0 += 1;
            flat.1 += usize::from(node.blob_index.is_some());
        }
        if lod > 0 && node.predictor_error_m >= manifest.height_pyramid.max_fallback_error_m * 4.0 {
            rough.0 += 1;
            rough.1 += usize::from(node.blob_index.is_some());
        }
    }
    println!(
        "  height pyramid: {}→{} px, {} levels, {:.1} m fallback budget",
        manifest.height_pyramid.base_resolution,
        manifest.height_pyramid.source_resolution,
        manifest.height_pyramid.level_count,
        manifest.height_pyramid.max_fallback_error_m,
    );
    for (lod, (total, retained, bytes, max_error)) in levels {
        let mut errors = predictor_errors.remove(&lod).unwrap_or_default();
        errors.sort_by(f32::total_cmp);
        let percentile = |fraction: f32| {
            if errors.is_empty() {
                0.0
            } else {
                errors[((errors.len() - 1) as f32 * fraction).round() as usize]
            }
        };
        println!(
            "    L{lod}: {retained}/{total} payloads ({:.1}% pruned), {:.2} MiB, max declared {max_error:.3} m; predictor min/p10/p50/p90/max {:.1}/{:.1}/{:.1}/{:.1}/{:.1} m",
            (total - retained) as f64 / total as f64 * 100.0,
            bytes as f64 / (1024.0 * 1024.0),
            percentile(0.0),
            percentile(0.1),
            percentile(0.5),
            percentile(0.9),
            percentile(1.0),
        );
    }
    if flat.0 > 0 {
        println!(
            "    within-budget retention: {}/{} ({:.1}%; expected 0%)",
            flat.1,
            flat.0,
            flat.1 as f64 / flat.0 as f64 * 100.0,
        );
    }
    if rough.0 > 0 {
        println!(
            "    >4×-budget retention: {}/{} ({:.1}%; expected 100%)",
            rough.1,
            rough.0,
            rough.1 as f64 / rough.0 as f64 * 100.0,
        );
    }
}
