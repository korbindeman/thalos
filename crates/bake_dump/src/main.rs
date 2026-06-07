//! `bake_dump` — headless terrain bake.
//!
//! Two modes:
//!
//! - **Default (full)**: runs the terrain compiler at the body's full
//!   resolution and writes the local game bake (`target/bakes/<body>.bin`,
//!   what your local game loads), equirectangular PNG previews
//!   (`stage-bakes/<body>/full/*.png`), and the ground-scale shaded-relief
//!   patch set as per-biome tile columns
//!   (`stage-bakes/<body>/full/patch/<biome>/*.png`).
//! - **`--preview`**: runs at 512² for fast iteration. Writes the equirect PNG
//!   previews *and* the same per-biome patch tile columns (hill + plain sites,
//!   spans from 120 km down to 60 m) under
//!   `stage-bakes/<body>/preview/patch/<biome>/`. Never touches `target/bakes/`,
//!   so iterating on the compiler doesn't invalidate the local game bake.
//!
//! Per-bake PNG outputs: albedo, height (grayscale, normalized to the
//! body's ± range), roughness, and object-space normal. Feature
//! cold-desert bodies also emit biome/suture maps in `--debug` so process
//! regions can be evaluated before albedo hides the structure.
//!
//! Usage:
//!
//!   cargo run --release -p thalos_bake_dump -- <body_name|all>
//!                                              [--preview]
//!                                              [--force]
//!                                              [--out <dir>]
//!                                              [--solar-system <path>]
//!                                              [--equirect-width W]
//!                                              [--debug]
//!
//! Body name matching is case-insensitive. Pass `all` to bake every
//! body in the solar system that has terrain.
//!
//! In production (non-`--preview`) mode the local bake at
//! `target/bakes/<body>.bin` is checked against the current cache key
//! before compiling; if the key matches the body is skipped (no
//! recompile, no PNG re-dump). Pass `--force` to bypass the check and
//! rebake unconditionally.
//!
//! Defaults:
//!
//!   --solar-system    assets/solar_system.ron
//!   --out             stage-bakes/<body>/{full,preview}/
//!   --equirect-width  512

mod gpu;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use glam::{DVec3, Vec3};
use image::{ImageBuffer, RgbImage};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use thalos_physics_canonical::parsing::load_solar_system_from_dir;
use thalos_terrain::cubemap::{CubemapFace, dir_to_face_uv};
use thalos_terrain::{
    BodyArchetype, BoundaryKind, ColdDesertField, DynamicSurfaceState, FeatureId,
    FeatureProjectionConfig, PlanetSurface, PlateKind, TerrainCompileContext,
    TerrainCompileOptions, TerrainConfig, compile_dynamic_surface_layers,
    compile_static_terrain_config, compile_tectonics_from_config, generate_initial_manifest,
    surface_height_m, surface_normal, surface_sample,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    /// Raw body name from the CLI (possibly `all`, mixed case).
    body_arg: String,
    /// Explicit `--out DIR`; when absent, defaults are derived per body
    /// (with a `full/` or `preview/` subdirectory).
    out_dir: Option<PathBuf>,
    solar_system: PathBuf,
    /// Fast-iteration preview mode (`--preview`). 512² cubemap; PNG dumps
    /// only, no local game bake written. Default is full resolution + local
    /// bake write.
    preview: bool,
    /// Bypass the on-disk hash check and rebake even if the existing
    /// local bake's key matches. No-op in `--preview` mode (preview
    /// never touches `target/bakes/`).
    force: bool,
    equirect_width: u32,
    /// Emit debug-only dumps (biome / suture / material-id) alongside the
    /// production PBR set. Off by default — production bakes ship only the
    /// four cubemaps the impostor consumes.
    debug: bool,
}

/// Cubemap face resolution used in preview mode. Matches the long-standing
/// iteration default — fast to compile and enough texture to read the
/// continental form clearly in equirect PNGs.
const PREVIEW_CUBEMAP_RESOLUTION: u32 = 512;

/// Result of writing the bake to `target/bakes/`. Surfaces the
/// production / preview distinction plus an explicit failure case so the
/// log message accurately reflects what happened on disk.
#[derive(Clone, Copy, Debug)]
enum StoreStatus {
    Stored,
    Failed,
    /// `--preview` was passed; the bake was compiled but intentionally
    /// not written to `target/bakes/`. PNG dumps still flow.
    SkippedPreview,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || raw.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: bake_dump <body_name|all> [--preview] [--force] [--out DIR] [--solar-system PATH] [--equirect-width W] [--debug]"
        );
        std::process::exit(if raw.is_empty() { 1 } else { 0 });
    }

    let mut body_name: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut solar_system: PathBuf = PathBuf::from("assets/solar_system.ron");
    let mut preview = false;
    let mut force = false;
    let mut equirect_width: u32 = 512;
    let mut debug = false;

    let mut i = 0;
    while i < raw.len() {
        let a = &raw[i];
        match a.as_str() {
            "--out" => {
                i += 1;
                out_dir = Some(PathBuf::from(&raw[i]));
            }
            "--solar-system" => {
                i += 1;
                solar_system = PathBuf::from(&raw[i]);
            }
            "--preview" => preview = true,
            "--force" => force = true,
            "--equirect-width" => {
                i += 1;
                equirect_width = raw[i].parse().expect("--equirect-width needs an integer");
            }
            "--debug" => debug = true,
            s if s.starts_with("--") => panic!("unknown flag: {s}"),
            s if body_name.is_none() => body_name = Some(s.to_string()),
            s => panic!("unexpected positional arg: {s}"),
        }
        i += 1;
    }

    let body_arg = body_name.expect("body name is required");

    Args {
        body_arg,
        out_dir,
        solar_system,
        preview,
        force,
        equirect_width,
        debug,
    }
}

const DEFAULT_OUT_ROOT: &str = "stage-bakes";

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    // Initialise the headless GPU context once at the top of the run.
    // Future stages (mid-frequency detail, cratering rasterization, …)
    // dispatch compute against this. The smoke test below confirms the
    // wgpu plumbing — adapter / device / queue / shader compile /
    // dispatch / readback — before any real stage relies on it.
    let gpu = gpu::GpuContext::new().expect("initialising GPU context");
    eprintln!("GPU: {}", gpu.describe());
    {
        let smoke = gpu::smoke_test(&gpu, 128).expect("GPU smoke test");
        for (i, v) in smoke.iter().enumerate() {
            assert_eq!(
                *v, i as u32,
                "GPU smoke test mismatch at index {i}: got {v}, expected {i}",
            );
        }
    }

    // Load from the assets directory containing the solar system file
    let root_path = std::path::Path::new(&args.solar_system)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("assets"));
    let system = load_solar_system_from_dir(root_path).expect("parsing solar system");

    let targets: Vec<&thalos_physics_canonical::types::BodyDefinition> =
        if args.body_arg.eq_ignore_ascii_case("all") {
            let mut v: Vec<_> = system
                .bodies
                .iter()
                .filter(|b| b.terrain.is_some())
                .collect();
            if v.is_empty() {
                panic!(
                    "no bodies in '{}' have terrain",
                    args.solar_system.display()
                );
            }
            v.sort_by(|a, b| a.name.cmp(&b.name));
            v
        } else {
            let body = system
                .bodies
                .iter()
                .find(|b| b.name.eq_ignore_ascii_case(&args.body_arg))
                .unwrap_or_else(|| panic!("body '{}' not found", args.body_arg));
            vec![body]
        };

    let mode_subdir = if args.preview { "preview" } else { "full" };
    let is_all = targets.len() > 1;
    let jobs: Vec<_> = targets
        .into_iter()
        .map(|body| {
            let out_dir = match (&args.out_dir, is_all) {
                // Explicit --out with a single body: use it directly,
                // no auto subdir (caller takes responsibility).
                (Some(p), false) => p.clone(),
                // Explicit --out with `all`: treat as parent, subdirs per body.
                (Some(p), true) => p.join(&body.name).join(mode_subdir),
                // Default: stage-bakes/<body>/{full,preview}/.
                (None, _) => PathBuf::from(DEFAULT_OUT_ROOT)
                    .join(&body.name)
                    .join(mode_subdir),
            };
            (body, out_dir)
        })
        .collect();

    let multi = MultiProgress::new();
    let any_failed = AtomicBool::new(false);

    if is_all {
        // Bounded in-process parallelism. The pool caps simultaneous body
        // compiles at half the core count; `terrain`'s internal
        // `par_iter` calls inherit this pool while we're inside
        // `pool.install`, so total CPU usage stays at the cap and the
        // box remains responsive during `bake all`.
        let parallel = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let body_concurrency = (parallel / 2).max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(body_concurrency)
            .thread_name(|i| format!("bake-{i}"))
            .build()
            .expect("building body-level rayon pool");
        pool.install(|| {
            jobs.par_iter().for_each(|(body, out_dir)| {
                if bake_one(
                    body,
                    out_dir,
                    args.preview,
                    args.force,
                    args.equirect_width,
                    args.debug,
                    &multi,
                    &gpu,
                )
                .is_err()
                {
                    any_failed.store(true, Ordering::Relaxed);
                }
            });
        });
    } else {
        for (body, out_dir) in &jobs {
            if bake_one(
                body,
                out_dir,
                args.preview,
                args.force,
                args.equirect_width,
                args.debug,
                &multi,
                &gpu,
            )
            .is_err()
            {
                any_failed.store(true, Ordering::Relaxed);
            }
        }
    }

    if any_failed.load(Ordering::Relaxed) {
        std::process::exit(1);
    }
}

/// Returns `Err(())` if any stage of the bake failed in a way the caller
/// should surface as a non-zero exit. The progress bar already carries
/// the user-facing failure message, so no further output is needed here.
#[allow(clippy::too_many_arguments)]
fn bake_one(
    body: &thalos_physics_canonical::types::BodyDefinition,
    out_dir: &Path,
    preview: bool,
    force: bool,
    equirect_width: u32,
    debug: bool,
    multi: &MultiProgress,
    gpu: &gpu::GpuContext,
) -> Result<(), ()> {
    let bar = multi.add(ProgressBar::new_spinner());
    bar.set_style(progress_style());
    bar.set_prefix(body.name.clone());
    bar.enable_steady_tick(Duration::from_millis(100));

    let start = Instant::now();
    let route = body.terrain.route_label();
    let context = terrain_context(body);
    let options = TerrainCompileOptions {
        crater_count_scale: 1.0,
        cubemap_resolution_override: preview.then_some(PREVIEW_CUBEMAP_RESOLUTION),
    };

    // Production-mode skip: if the local bake's stored hash already
    // matches what we'd compute now, the bake (and its PNG dumps) are
    // already current — re-baking would just rewrite identical bytes.
    if !preview && !force && local_bake_is_up_to_date(body) {
        bar.finish_with_message(format!("up-to-date · {route} · pass --force to rebake"));
        return Ok(());
    }

    bar.set_message("compiling dynamic layers");
    let dynamic_layers = match compile_dynamic_surface_layers(&body.terrain, &context) {
        Ok(d) => d,
        Err(e) => {
            bar.finish_with_message(format!("FAILED · dynamic layer compile: {e}"));
            return Err(());
        }
    };

    bar.set_message("compiling tectonics");
    let tectonics = compile_tectonics_from_config(body.tectonics.as_ref(), &context);

    bar.set_message("compiling static surface");
    // Mid-frequency detail runner. Capture clones of the GPU device +
    // queue so the closure is `'static` (`MidFreqRunner` is
    // `Box<dyn FnOnce + Send>` with no lifetime parameter). Arc clones
    // are cheap — wgpu Device/Queue are internally refcounted.
    let device = gpu.device.clone();
    let queue = gpu.queue.clone();
    let mid_freq: Option<thalos_terrain::stages::MidFreqRunner> = Some(Box::new(
        move |height: &mut thalos_terrain::cubemap::Cubemap<f32>,
              radius_m: f32,
              params: &thalos_terrain::stages::MidFreqDetailParams|
              -> Result<(), String> {
            gpu::run_mid_freq(&device, &queue, height, radius_m, params).map_err(|e| e.to_string())
        },
    ));
    let static_surface = match compile_static_terrain_config(
        &body.terrain,
        tectonics.as_ref(),
        &context,
        options,
        mid_freq,
    ) {
        Ok(s) => s,
        Err(e) => {
            bar.finish_with_message(format!("FAILED · static surface compile: {e}"));
            return Err(());
        }
    };

    let store_status = if preview {
        StoreStatus::SkippedPreview
    } else {
        bar.set_message("writing local bake");
        let bake_dir = local_bake_dir();
        let key = thalos_terrain::cache::terrain_cache_key(
            &body.terrain,
            body.tectonics.as_ref(),
            &context,
            options,
        );
        let path = thalos_terrain::cache::cache_path(&bake_dir, &body.name);
        match thalos_terrain::cache::store(&path, key, &static_surface) {
            Ok(()) => StoreStatus::Stored,
            Err(e) => {
                let _ = multi.println(format!(
                    "bake write failed for {} → {}: {e}",
                    body.name,
                    path.display(),
                ));
                StoreStatus::Failed
            }
        }
    };

    let surface = PlanetSurface {
        static_surface,
        dynamic_layers,
        tectonics,
    };
    let n_craters = surface.static_surface.craters.len();

    fs::create_dir_all(out_dir).expect("creating out dir");
    remove_legacy_outputs(out_dir, debug);

    bar.set_message("writing PNG dumps");
    dump_all_in_parallel(&surface, body, out_dir, equirect_width, debug);
    dump_info(&surface, &route, out_dir);

    // Both full and preview emit the ground-scale patch set so each mode dir
    // carries its `patch/<biome>/` tile columns alongside the equirects —
    // orbital coloration (the equirects) plus on-foot relief (the patches),
    // which the equirects are far too coarse to read. The surface is already
    // compiled, so patches are CPU-cheap regardless of mode. Routed through
    // `MultiProgress::println` so the per-site centre summaries don't garble
    // the bars under `bake all`.
    bar.set_message("rendering ground patches");
    dump_patches(&surface, out_dir, &mut |s| {
        let _ = multi.println(s);
    });

    let elapsed = start.elapsed();
    bar.finish_with_message(format!(
        "done in {:.1}s · {route} · {} · {n_craters} craters",
        elapsed.as_secs_f32(),
        store_status_label(store_status),
    ));
    Ok(())
}

fn progress_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.cyan.bold} {prefix:<10} {msg}")
        .expect("static progress template parses")
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏✓")
}

/// Cheap staleness check: does `target/bakes/<body>.bin` already carry
/// the cache key we'd produce now? Returns `false` on any failure
/// (missing file, decode error, mismatched key) so callers fall through
/// to the full recompile + overwrite.
fn local_bake_is_up_to_date(body: &thalos_physics_canonical::types::BodyDefinition) -> bool {
    let context = terrain_context(body);
    // Production options — must match the values `run_terrain` uses in
    // non-preview mode, otherwise the keys diverge and we'd never see a
    // hit. Keep these in sync if `run_terrain`'s production options
    // change.
    let options = TerrainCompileOptions {
        crater_count_scale: 1.0,
        cubemap_resolution_override: None,
    };
    let expected = thalos_terrain::cache::terrain_cache_key(
        &body.terrain,
        body.tectonics.as_ref(),
        &context,
        options,
    );
    let path = thalos_terrain::cache::cache_path(&local_bake_dir(), &body.name);
    matches!(
        thalos_terrain::cache::peek_key(&path),
        Ok(stored) if stored == expected,
    )
}

fn terrain_context(
    body: &thalos_physics_canonical::types::BodyDefinition,
) -> TerrainCompileContext {
    TerrainCompileContext {
        body_name: body.name.clone(),
        radius_m: body.radius_m as f32,
        gravity_m_s2: (body.gm / (body.radius_m * body.radius_m)) as f32,
        rotation_hours: None,
        obliquity_deg: Some((body.axial_tilt_rad as f32).to_degrees()),
        tidal_axis: matches!(body.kind, thalos_physics_canonical::types::BodyKind::Moon)
            .then_some(Vec3::Z),
        axial_tilt_rad: body.axial_tilt_rad as f32,
    }
}

/// Output directory for local game bakes (`<workspace>/target/bakes`). Mirror
/// of the same path resolved by the game and the editor's full-bake path,
/// so all producers + the local game agree on one location.
fn local_bake_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/bakes")
}

fn store_status_label(status: StoreStatus) -> &'static str {
    match status {
        StoreStatus::Stored => "wrote local bake",
        StoreStatus::Failed => "bake write failed",
        StoreStatus::SkippedPreview => "preview only (no local game bake)",
    }
}

// ---------------------------------------------------------------------------
// Dump
// ---------------------------------------------------------------------------

/// All PNG dumps for a single body, written concurrently. Each `write_equirect`
/// also parallelizes internally over rows; rayon's work-stealing means the
/// two layers compose without manual scheduling.
///
/// Production set (always): albedo, height, roughness, normal.
/// Ocean bodies also get an orbit-color preview that composites the separate
/// water layer over the raw seabed/land albedo for visual iteration.
/// Tectonic set (when the body has tectonics): plate-id, boundary-type.
/// Debug set (`--debug`): material-id, plus biome/suture for cold-desert bodies.
fn dump_all_in_parallel(
    surface: &PlanetSurface,
    body_def: &thalos_physics_canonical::types::BodyDefinition,
    out: &Path,
    equirect_w: u32,
    debug: bool,
) {
    let static_surface = &surface.static_surface;
    let state = DynamicSurfaceState::for_layers(&surface.dynamic_layers);
    // Metres per equirect texel at the equator. The Query API seam takes a
    // linear metres-per-sample LOD (not log2), so this drives the detail
    // cascade directly.
    let lod_m =
        ((std::f32::consts::TAU * static_surface.radius_m) / equirect_w.max(1) as f32).max(1.0);
    let height_range = static_surface.height_range;
    // Precompute the cold-desert biome field here so the field outlives the
    // `par_iter` below — closures borrow it by reference inside the debug
    // branch.
    let biome_field = if debug {
        cold_desert_biome_field(body_def)
    } else {
        None
    };

    enum DumpKind {
        Albedo,
        Height,
        Roughness,
        Normal,
        OrbitColor,
        PlateId,
        BoundaryType,
        Material,
        Biome,
        Suture,
    }

    let mut dumps = vec![
        DumpKind::Albedo,
        DumpKind::Height,
        DumpKind::Roughness,
        DumpKind::Normal,
    ];
    if static_surface.sea_level_m.is_some() {
        dumps.push(DumpKind::OrbitColor);
    }
    if surface.tectonics.is_some() {
        dumps.push(DumpKind::PlateId);
        dumps.push(DumpKind::BoundaryType);
    }
    if debug {
        dumps.push(DumpKind::Material);
        if biome_field.is_some() {
            dumps.push(DumpKind::Biome);
            dumps.push(DumpKind::Suture);
        }
    }

    dumps.into_par_iter().for_each(|kind| match kind {
        DumpKind::Albedo => write_equirect(out.join("albedo-equirect.png"), equirect_w, |dir| {
            let sample = surface_sample(surface, &state, dir.as_dvec3(), lod_m);
            [
                linear_to_srgb8(sample.albedo_linear.x),
                linear_to_srgb8(sample.albedo_linear.y),
                linear_to_srgb8(sample.albedo_linear.z),
            ]
        }),
        DumpKind::Height => write_equirect(out.join("height-equirect.png"), equirect_w, |dir| {
            let sample = surface_sample(surface, &state, dir.as_dvec3(), lod_m);
            let g = ((sample.height_m / height_range.max(1.0) * 0.5 + 0.5) * 255.0)
                .clamp(0.0, 255.0)
                .round() as u8;
            [g, g, g]
        }),
        DumpKind::Roughness => {
            write_equirect(out.join("roughness-equirect.png"), equirect_w, |dir| {
                let sample = surface_sample(surface, &state, dir.as_dvec3(), lod_m);
                let g = (sample.roughness.clamp(0.0, 1.0) * 255.0).round() as u8;
                [g, g, g]
            })
        }
        DumpKind::Normal => write_equirect(out.join("normal-equirect.png"), equirect_w, |dir| {
            let normal = surface_normal(surface, &state, dir.as_dvec3(), lod_m);
            [
                normal_to_u8(normal.x),
                normal_to_u8(normal.y),
                normal_to_u8(normal.z),
            ]
        }),
        DumpKind::OrbitColor => {
            write_equirect(out.join("orbit-color-equirect.png"), equirect_w, |dir| {
                let sample = surface_sample(surface, &state, dir.as_dvec3(), lod_m);
                let mut color = sample.albedo_linear;
                if let Some(sea_level_m) = static_surface.sea_level_m {
                    let depth_m = sea_level_m - sample.height_m;
                    if depth_m > 0.0 {
                        color = orbital_preview_water_color(
                            sample.albedo_linear,
                            depth_m,
                            static_surface.water_appearance.map(|w| w.color_depth),
                        );
                    }
                }
                [
                    linear_to_srgb8(color.x),
                    linear_to_srgb8(color.y),
                    linear_to_srgb8(color.z),
                ]
            })
        }
        DumpKind::PlateId => {
            let tectonics = surface
                .tectonics
                .as_ref()
                .expect("PlateId dump is only enqueued when tectonics exist");
            write_equirect(out.join("plate-id-equirect.png"), equirect_w, |dir| {
                let sample = tectonics.sample(dir);
                plate_color_srgb(sample.plate_id.0, sample.plate_kind)
            });
        }
        DumpKind::BoundaryType => {
            let tectonics = surface
                .tectonics
                .as_ref()
                .expect("BoundaryType dump is only enqueued when tectonics exist");
            // 8% of body radius — wide enough that the boundary reads from
            // orbit at typical equirect resolutions; narrow enough that
            // interior cells stay clean.
            let threshold_m = tectonics.body_radius_m * 0.08;
            write_equirect(out.join("boundary-type-equirect.png"), equirect_w, |dir| {
                let sample = tectonics.sample(dir);
                let Some(kind) = sample.boundary_kind else {
                    return [10, 10, 14];
                };
                let d = sample.boundary_distance_m;
                if d > threshold_m {
                    return [10, 10, 14];
                }
                let intensity = 1.0 - (d / threshold_m).clamp(0.0, 1.0);
                let base = boundary_kind_linear(kind);
                [
                    linear_to_srgb8(base[0] * intensity),
                    linear_to_srgb8(base[1] * intensity),
                    linear_to_srgb8(base[2] * intensity),
                ]
            });
        }
        DumpKind::Material => {
            let mat = &static_surface.material_cubemap;
            write_equirect(out.join("material-equirect.png"), equirect_w, |dir| {
                let (face, u, v) = dir_to_face_uv(dir);
                let (x, y) = uv_to_texel(u, v, mat.resolution());
                hash_color(mat.get(face, x, y) as u32)
            });
        }
        DumpKind::Biome => {
            let field = biome_field
                .as_ref()
                .expect("Biome dump is only enqueued when the cold-desert field exists");
            write_equirect(out.join("biome-equirect.png"), equirect_w, |dir| {
                field.debug_biome_color_srgb(dir)
            });
        }
        DumpKind::Suture => {
            let field = biome_field
                .as_ref()
                .expect("Suture dump is only enqueued when the cold-desert field exists");
            write_equirect(out.join("suture-equirect.png"), equirect_w, |dir| {
                field.sample_suture_debug(dir).debug_color_srgb()
            });
        }
    });
}

fn linear_to_srgb8(linear: f32) -> u8 {
    let x = linear.clamp(0.0, 1.0);
    let srgb = if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (srgb.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn normal_to_u8(v: f32) -> u8 {
    ((v.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8
}

fn orbital_preview_water_color(
    seabed_linear: Vec3,
    depth_m: f32,
    water_color_depth: Option<[f32; 4]>,
) -> Vec3 {
    let water = water_color_depth.unwrap_or([0.012, 0.040, 0.090, 120.0]);
    let deep = Vec3::new(water[0], water[1], water[2]) * 1.05 + Vec3::new(0.0, 0.006, 0.018);
    let min_depth_m = water[3].max(1.0);
    let depth_m = depth_m.max(0.0);

    // Preview only: approximate the separate ocean renderer's optical column
    // so orbit equirects read like the game view while `albedo-equirect`
    // remains the raw land/seabed substrate.
    let absorption = Vec3::new(
        (-0.018 * depth_m).exp(),
        (-0.010 * depth_m).exp(),
        (-0.004 * depth_m).exp(),
    );
    let transmitted_bottom = seabed_linear * absorption * 0.72;
    let shallow_scatter = Vec3::new(0.006, 0.078, 0.105) * (1.0 - (-depth_m / 28.0).exp());
    let shallow = transmitted_bottom + shallow_scatter;
    let deep_t = smoothstep_scalar(18.0, min_depth_m * 1.35, depth_m);
    shallow
        .lerp(deep, deep_t)
        .clamp(Vec3::splat(0.0), Vec3::splat(1.0))
}

fn smoothstep_scalar(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 == edge1 {
        return (x >= edge1) as u8 as f32;
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn cold_desert_biome_field(
    body: &thalos_physics_canonical::types::BodyDefinition,
) -> Option<ColdDesertField> {
    let TerrainConfig::Feature(feature) = &body.terrain else {
        return None;
    };
    if feature.archetype != BodyArchetype::ColdDesertFormerlyWet {
        return None;
    }

    let context = terrain_context(body);
    let spec = feature.to_planet_spec(&context);
    let manifest = generate_initial_manifest(&spec);
    let crust_id = FeatureId::new(format!("{}.crustal_provinces", spec.body_id));
    let crust = manifest.get(&crust_id)?;
    let projection = match &feature.projection {
        FeatureProjectionConfig::ColdDesert(config) => config.clone(),
        FeatureProjectionConfig::Auto | FeatureProjectionConfig::AirlessImpact(_) => {
            Default::default()
        }
    };

    Some(ColdDesertField::with_style(
        crust.seed,
        projection,
        feature.cold_desert_style.clone().unwrap_or_default(),
    ))
}

/// Deterministic per-plate sRGB color. Continental plates land in warm hues
/// (browns/reds/oranges) at moderate brightness; oceanic plates land in
/// cool blue-greens at lower brightness so the land/sea split reads at a
/// glance.
fn plate_color_srgb(plate_id: u32, kind: PlateKind) -> [u8; 3] {
    let h = thalos_terrain::seeding::splitmix64(plate_id as u64 ^ 0xB1ADE0FF);
    let hue_unit = ((h & 0xFFFF) as f32) / 65535.0;
    let (hue_deg, sat, val) = match kind {
        // Land: warm wedge, hue ∈ [0°, 60°] ∪ [300°, 360°] mapped from a
        // single uniform sample.
        PlateKind::Continental => {
            let hue = if hue_unit < 0.5 {
                hue_unit * 120.0
            } else {
                300.0 + (hue_unit - 0.5) * 120.0
            };
            (hue, 0.55, 0.72)
        }
        // Ocean: cool wedge, hue ∈ [180°, 240°].
        PlateKind::Oceanic => (180.0 + hue_unit * 60.0, 0.60, 0.40),
    };
    let [r, g, b] = hsv_to_linear_rgb(hue_deg, sat, val);
    [linear_to_srgb8(r), linear_to_srgb8(g), linear_to_srgb8(b)]
}

fn boundary_kind_linear(kind: BoundaryKind) -> [f32; 3] {
    match kind {
        BoundaryKind::Convergent => [1.00, 0.18, 0.18],
        BoundaryKind::Divergent => [0.20, 0.55, 1.00],
        BoundaryKind::Transform => [1.00, 0.85, 0.20],
    }
}

/// Linear-RGB output from HSV. Standard reference formula.
fn hsv_to_linear_rgb(h_deg: f32, s: f32, v: f32) -> [f32; 3] {
    let h = (h_deg.rem_euclid(360.0)) / 60.0;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

fn remove_legacy_outputs(out: &Path, debug: bool) {
    let mut targets: Vec<&str> = vec![
        "domain-equirect.png",
        "albedo-cross.png",
        "height-cross.png",
        "material-cross.png",
    ];
    if !debug {
        // Debug-only outputs from a previous run shouldn't linger when the
        // current bake didn't request them.
        targets.extend([
            "material-equirect.png",
            "biome-equirect.png",
            "suture-equirect.png",
        ]);
    }
    for name in targets {
        match fs::remove_file(out.join(name)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => panic!("removing stale bake output {name:?}: {e}"),
        }
    }
}

fn dump_info(surface: &PlanetSurface, route: &str, out: &Path) {
    let body = &surface.static_surface;
    let mut s = String::new();
    s.push_str(&format!("radius_m:    {}\n", body.radius_m));
    s.push_str(&format!("height_range_m: {}\n", body.height_range));
    s.push_str(&format!(
        "cubemap_resolution: {}\n",
        body.albedo_cubemap.resolution()
    ));
    if let Some(sl) = body.sea_level_m {
        s.push_str(&format!("sea_level_m: {sl:.1}\n"));
    }
    s.push_str(&format!("route:       {route}\n"));
    s.push_str(&format!("craters:     {}\n", body.craters.len()));
    s.push_str(&format!("volcanoes:   {}\n", body.volcanoes.len()));
    s.push_str(&format!("channels:    {}\n", body.channels.len()));
    s.push_str(&format!(
        "dynamic_layers: ice_caps={}, active_dunes={}\n",
        surface.dynamic_layers.ice_caps.len(),
        surface.dynamic_layers.active_dunes.len()
    ));
    if let Some(tectonics) = surface.tectonics.as_ref() {
        let n_plates = tectonics.plates.len();
        let n_continental = tectonics
            .plates
            .iter()
            .filter(|p| p.kind == PlateKind::Continental)
            .count();
        let n_oceanic = n_plates - n_continental;
        let mut convergent = 0usize;
        let mut divergent = 0usize;
        let mut transform = 0usize;
        for b in &tectonics.boundaries {
            match b.kind {
                BoundaryKind::Convergent => convergent += 1,
                BoundaryKind::Divergent => divergent += 1,
                BoundaryKind::Transform => transform += 1,
            }
        }
        s.push_str(&format!(
            "tectonics: plates={n_plates} (continental={n_continental}, oceanic={n_oceanic}), \
             cells={}, boundaries={} (convergent={convergent}, divergent={divergent}, transform={transform}), \
             activity={:?}\n",
            tectonics.mesh.cells.len(),
            tectonics.boundaries.len(),
            tectonics.config.activity,
        ));
    }
    s.push_str(&format!("materials:   {}\n", body.materials.len()));
    if !body.materials.is_empty() {
        // Texel-level histogram so you can see which material rules
        // actually fire, per bake. Useful when iterating on Stage 5
        // thresholds — the albedo PNG shows what the shader will
        // render, but the count tells you whether a rare category
        // is literally one pixel or genuinely un-triggered.
        let mut counts = vec![0u64; body.materials.len()];
        let mut total: u64 = 0;
        for face in CubemapFace::ALL {
            for &id in body.material_cubemap.face_data(face) {
                if (id as usize) < counts.len() {
                    counts[id as usize] += 1;
                }
                total += 1;
            }
        }
        s.push_str("material histogram:\n");
        for (i, &n) in counts.iter().enumerate() {
            let pct = 100.0 * n as f64 / total.max(1) as f64;
            let m = &body.materials[i];
            s.push_str(&format!(
                "  {:2}  albedo=({:.2},{:.2},{:.2})  r={:.2}  texels={:>9}  {:5.1}%\n",
                i, m.albedo[0], m.albedo[1], m.albedo[2], m.roughness, n, pct,
            ));
        }
    }
    fs::write(out.join("info.txt"), s).expect("writing info.txt");
}

// ---------------------------------------------------------------------------
// Patch hillshade (ground-scale shaded relief)
// ---------------------------------------------------------------------------

fn latlon_to_dir(lat_deg: f32, lon_deg: f32) -> Vec3 {
    let (sl, cl) = lat_deg.to_radians().sin_cos();
    let (sln, cln) = lon_deg.to_radians().sin_cos();
    Vec3::new(cl * sln, sl, cl * cln).normalize()
}

fn dir_to_latlon(dir: Vec3) -> (f32, f32) {
    (
        dir.y.clamp(-1.0, 1.0).asin().to_degrees(),
        dir.x.atan2(dir.z).to_degrees(),
    )
}

/// East/north tangent unit vectors at a surface direction.
fn tangent_basis(dir: Vec3) -> (Vec3, Vec3) {
    let up_ref = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    let east = up_ref.cross(dir).normalize();
    let north = dir.cross(east);
    (east, north)
}

/// Scan low-latitude directions and return either the highest-relief site
/// (`seek_hills`) or the flattest usable land site (`!seek_hills`). Both must
/// sit comfortably above sea level — otherwise "plain" lands on the abyssal
/// basin floor (which is genuinely the flattest place on the planet) and
/// "hill" can land on a continental swell with no real ridge content.
///
/// Probe geometry: 9-sample 3×3 stencil at ±`probe_m`, sized to the
/// `hill_ridges` wavelength (~5 km) so the cross actually catches ridge crests
/// rather than averaging through them. Score: max−min of `surface_height_m`
/// across the stencil.
fn auto_find_relief_center(surface: &PlanetSurface, seek_hills: bool) -> Vec3 {
    let state = DynamicSurfaceState::for_layers(&surface.dynamic_layers);
    let radius = surface.static_surface.radius_m;
    let sea_level_m = surface.static_surface.sea_level_m.unwrap_or(0.0);
    // Min altitude above sea level for either site. Keeps both hill and plain
    // out of the wave zone / continental shelf so the picked tile is genuine
    // land character, not waterline transition.
    let min_land_m = sea_level_m + 60.0;
    // Probe spacing ~ hill_ridges wavelength so the 3×3 stencil straddles a
    // ridge crest at a hill site instead of averaging through it.
    let probe_m = 5_500.0;
    let n = 20_000usize;
    let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());

    let (best, _) = (0..n)
        .into_par_iter()
        .map(|k| {
            let lat = ((k as f32 + 0.5) / n as f32 - 0.5) * 2.0 * 45.0; // -45..45°
            let lon = ((k as f32) * golden).to_degrees().rem_euclid(360.0) - 180.0;
            let dir = latlon_to_dir(lat, lon);
            let (east, north) = tangent_basis(dir);
            let base = dir * radius;
            let h = |off: Vec3| {
                surface_height_m(surface, &state, (base + off).normalize().as_dvec3(), 0.5)
            };
            let s = [
                h(Vec3::ZERO),
                h(east * probe_m),
                h(-east * probe_m),
                h(north * probe_m),
                h(-north * probe_m),
                h((east + north) * probe_m),
                h((east - north) * probe_m),
                h((-east + north) * probe_m),
                h((-east - north) * probe_m),
            ];
            let lo = s.iter().cloned().fold(f32::MAX, f32::min);
            let hi = s.iter().cloned().fold(f32::MIN, f32::max);
            // Reject any candidate where the centre dips below the minimum land
            // altitude. Plain candidates would happily settle on the abyssal
            // basin (genuinely flatter than any continental plain); hill
            // candidates picked on the shelf show no ridges either.
            if s[0] < min_land_m {
                return (dir, f32::NEG_INFINITY);
            }
            let relief = hi - lo;
            (dir, if seek_hills { relief } else { -relief })
        })
        .reduce(
            || (Vec3::Z, f32::NEG_INFINITY),
            |a, b| if b.1 > a.1 { b } else { a },
        );
    best
}

/// Render a `span_m × span_m` shaded-relief patch centred on `center`, sampling
/// the runtime walkable height on a tangent-plane grid.
fn render_patch(surface: &PlanetSurface, center: Vec3, span_m: f32, res: u32, out_path: &Path) {
    let state = DynamicSurfaceState::for_layers(&surface.dynamic_layers);
    let radius = surface.static_surface.radius_m as f64;
    // Build the tangent-plane sample grid in f64. At planet scale (~3.2e6 m on
    // Thalos) an f32 grid quantises the body-local sample position to the
    // ~0.25 m f32 lattice; for the sub-metre-per-pixel fine patches that beats
    // against the pixel spacing and renders as a grid/checkerboard moiré the
    // field never contains. The game's own tile path (`pixel_direction`) builds
    // its sample directions in f64 for exactly this reason — the diagnostic must
    // match it, or it shows precision artifacts the runtime does not have.
    let center = center.normalize().as_dvec3();
    let up_ref = if center.y.abs() > 0.99 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let east = up_ref.cross(center).normalize();
    let north = center.cross(east);
    let base = center * radius;
    let span_m = span_m as f64;
    let n = res as usize;

    let mut h = vec![0f32; n * n];
    h.par_chunks_mut(n).enumerate().for_each(|(j, row)| {
        let ny = (0.5 - (j as f64 + 0.5) / res as f64) * span_m; // +north at top
        for (i, cell) in row.iter_mut().enumerate() {
            let nx = ((i as f64 + 0.5) / res as f64 - 0.5) * span_m; // +east at right
            let dir = (base + east * nx + north * ny).normalize();
            *cell = surface_height_m(surface, &state, dir, 0.5);
        }
    });

    let ds = (span_m / res as f64) as f32;
    // Sun in local (east, up, north): from the NW, moderate elevation.
    let sun = Vec3::new(-0.55, 0.62, 0.55).normalize();
    let sample = |i: i32, j: i32| -> f32 {
        let ii = i.clamp(0, n as i32 - 1) as usize;
        let jj = j.clamp(0, n as i32 - 1) as usize;
        h[jj * n + ii]
    };
    // Absolute tint keyed off sea level and an upland reference (~1500 m above
    // sea level). This must NOT stretch per-patch the way the old `hmin..hmax`
    // tint did — otherwise a 60 m patch with a millimetre of relief and a 120 km
    // patch with a kilometre of relief both fill the full green→tan gamut, and
    // every LOD looks identical from a colour perspective.
    let sea_level_m = surface.static_surface.sea_level_m.unwrap_or(0.0);
    let upland_m = sea_level_m + 1_500.0;
    let beach = Vec3::new(0.74, 0.66, 0.45);
    let lowland = Vec3::new(0.27, 0.40, 0.18);
    let upland = Vec3::new(0.55, 0.47, 0.34);
    let rock = Vec3::new(0.40, 0.36, 0.31);
    let shallow_sea = Vec3::new(0.18, 0.32, 0.42);
    let deep_sea = Vec3::new(0.06, 0.10, 0.18);
    // Slope (rise/run) above which the surface reads as rock/cliff rather than
    // soil. Tunes the cliff cutoff and the gradient between soil and exposed
    // rock so steep coastlines render as cliffs, not sand.
    let cliff_slope_lo = 0.18;
    let cliff_slope_hi = 0.45;
    // Beach band width above sea level. The old 25 m strip read as a fat ring
    // around every continent; real beaches only occupy the surf zone (a few
    // metres of altitude) on gentle-slope coasts.
    let beach_max_m = 3.0;
    let tint_for = |height_m: f32, slope: f32| -> Vec3 {
        if height_m < sea_level_m {
            // Water rendering. The field's seabed has real bathymetric
            // structure (eroded_ridged at 180 km plus abyssal noise) with
            // hundreds of metres of relief. Tinting by raw depth made those
            // ridges project into the visible water colour as swirly contours
            // at the shore/shallow/deep band edges. Real water absorbs light
            // long before continental-shelf bathymetry registers from above, so
            // the visible tint only varies over deep-scale depths — fine
            // seabed structure stays invisible. No shore band: it was the
            // single biggest source of artefacts, because the bathymetric
            // ridge network always crossed the 0..18 m line at a wiggly contour.
            let raw_depth = (sea_level_m - height_m).max(0.0);
            // Gradient only kicks in below 800 m and saturates at 5000 m.
            // Continental shelf (≤ ~380 m) and shelf bathymetric variance
            // (±420 m) both fall below the start of the gradient, so they
            // render as uniform shallow_sea regardless of local seabed
            // structure. Only true abyssal depth modulates the visible tint.
            let depth_t = smoothstep_scalar(800.0, 5_000.0, raw_depth);
            shallow_sea.lerp(deep_sea, depth_t)
        } else {
            let rock_t = smoothstep_scalar(cliff_slope_lo, cliff_slope_hi, slope);
            let upland_t = ((height_m - sea_level_m) / (upland_m - sea_level_m)).clamp(0.0, 1.0);
            // Soil ramp: only the narrow surf-zone is sand, beyond that it
            // grades from lowland to upland by absolute altitude.
            let soil = if height_m < sea_level_m + beach_max_m && rock_t < 0.5 {
                let beach_t = ((height_m - sea_level_m) / beach_max_m).clamp(0.0, 1.0);
                beach.lerp(lowland, beach_t)
            } else {
                lowland.lerp(upland, upland_t)
            };
            soil.lerp(rock, rock_t)
        }
    };

    let mut data = vec![0u8; n * n * 3];
    data.par_chunks_mut(n * 3).enumerate().for_each(|(j, row)| {
        for i in 0..n {
            let height_here = h[j * n + i];
            let dh_de =
                (sample(i as i32 + 1, j as i32) - sample(i as i32 - 1, j as i32)) / (2.0 * ds);
            // North increases as j decreases (top row is +north).
            let dh_dn =
                (sample(i as i32, j as i32 - 1) - sample(i as i32, j as i32 + 1)) / (2.0 * ds);
            let shade = if height_here < sea_level_m {
                // Underwater: skip terrain shading. The seabed has real
                // bathymetric slope, but rendering it would expose seabed
                // structure as swirly shaded contours on what should read as
                // flat water. Use a constant, slightly-below-1 base.
                0.95
            } else {
                let normal = Vec3::new(-dh_de, 1.0, -dh_dn).normalize();
                // Sharper shading curve so micro-slopes are visible: amplify
                // the signed slope component against the sun direction before
                // mapping to brightness. The old `0.25 + 0.75·dot` washed
                // sub-degree slopes into a flat tone, hiding the field's
                // content at finer LODs.
                let raw = normal.dot(sun);
                let lit = (raw - 0.5).clamp(-1.0, 1.0);
                (0.55 + 0.85 * lit).clamp(0.10, 1.20)
            };
            let slope = (dh_de * dh_de + dh_dn * dh_dn).sqrt();
            let c = tint_for(height_here, slope) * shade;
            let idx = i * 3;
            row[idx] = linear_to_srgb8(c.x);
            row[idx + 1] = linear_to_srgb8(c.y);
            row[idx + 2] = linear_to_srgb8(c.z);
        }
    });

    let img: RgbImage =
        ImageBuffer::from_raw(res, res, data).expect("patch dimensions match buffer length");
    img.save(out_path)
        .unwrap_or_else(|e| panic!("writing {out_path:?}: {e}"));
}

/// Render the standard ground-scale patch set into `out_dir/patch/<biome>/`.
/// Each site becomes a biome tile column — its own subdir holding the LOD
/// cascade (`context-120km.png` … `ultra-60m.png`). Renders both a hill site
/// (highest relief) and the flattest plain site, so the plains/hills balance
/// and the gentle plain rolls are both visible without launching the game.
/// Called by both the full and preview bakes.
///
/// CPU-only — the surface is already compiled. `log` receives the per-site
/// centre summaries via `MultiProgress::println` so the lines don't garble the
/// progress bars.
fn dump_patches(surface: &PlanetSurface, out_dir: &Path, log: &mut dyn FnMut(String)) {
    let sites: Vec<(String, Vec3)> = vec![
        ("hill".to_string(), auto_find_relief_center(surface, true)),
        ("plain".to_string(), auto_find_relief_center(surface, false)),
    ];

    let res = 1024;
    for (site, center) in &sites {
        let (lat, lon) = dir_to_latlon(*center);
        log(format!("patch: {site} centre = lat {lat:.2}°, lon {lon:.2}°"));
        // Each site is a "biome" tile column: its own `patch/<biome>/` subdir
        // holding the LOD cascade as bare `<span>.png` files. This mirrors the
        // planet editor's tile view, where one tile carries several zoom
        // levels — the patches *are* the tiles that exist on the planet.
        let biome_dir = out_dir.join("patch").join(site);
        fs::create_dir_all(&biome_dir).expect("creating patch biome dir");
        for (span_km, name) in [
            (120.0_f32, "context-120km"),
            (12.0, "close-12km"),
            (3.0, "micro-3km"),
            (0.3, "fine-300m"),
            (0.06, "ultra-60m"),
        ] {
            let path = biome_dir.join(format!("{name}.png"));
            render_patch(surface, *center, span_km * 1000.0, res, &path);
        }
    }
}

// ---------------------------------------------------------------------------
// Projections
// ---------------------------------------------------------------------------

fn uv_to_texel(u: f32, v: f32, res: u32) -> (u32, u32) {
    let x = (u * res as f32).clamp(0.0, (res - 1) as f32) as u32;
    let y = (v * res as f32).clamp(0.0, (res - 1) as f32) as u32;
    (x, y)
}

/// Equirectangular projection: `width` × `width/2` image, latitude runs
/// top (+π/2) to bottom (−π/2), longitude runs left (−π) to right (+π).
/// Center of image is direction `+Z`. Rows are filled in parallel via rayon.
fn write_equirect<F: Fn(Vec3) -> [u8; 3] + Sync>(path: PathBuf, width: u32, shade: F) {
    let height = width / 2;
    let stride = (width as usize) * 3;
    let mut data = vec![0u8; (height as usize) * stride];
    data.par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row)| {
            let lat = (0.5 - (y as f32 + 0.5) / height as f32) * std::f32::consts::PI;
            let (sl, cl) = lat.sin_cos();
            for x in 0..width as usize {
                let lon = ((x as f32 + 0.5) / width as f32 - 0.5) * std::f32::consts::TAU;
                let (sln, cln) = lon.sin_cos();
                let dir = Vec3::new(cl * sln, sl, cl * cln);
                let [r, g, b] = shade(dir);
                let i = x * 3;
                row[i] = r;
                row[i + 1] = g;
                row[i + 2] = b;
            }
        });
    let img: RgbImage = ImageBuffer::from_raw(width, height, data)
        .expect("equirect dimensions match buffer length");
    img.save(&path)
        .unwrap_or_else(|e| panic!("writing {path:?}: {e}"));
}

/// Deterministic per-ID color for material/biome masks. ID 0 renders
/// mid-grey so "unset" is visually obvious.
fn hash_color(id: u32) -> [u8; 3] {
    if id == 0 {
        return [60, 60, 60];
    }
    let h = thalos_terrain::seeding::splitmix64(id as u64 ^ 0xD3ADBEEF);
    [
        (h & 0xFF) as u8,
        ((h >> 8) & 0xFF) as u8,
        ((h >> 16) & 0xFF) as u8,
    ]
}
