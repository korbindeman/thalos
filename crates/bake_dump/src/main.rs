//! `bake_dump` — headless terrain bake.
//!
//! Two modes:
//!
//! - **Default (full)**: runs the terrain compiler at the body's full
//!   resolution and writes both the local game bake (`target/bakes/<body>.bin`,
//!   what your local game loads) and equirectangular PNG previews
//!   (`stage-bakes/<body>/full/*.png`).
//! - **`--preview`**: runs at 512² for fast iteration. Writes ONLY the
//!   PNG previews to `stage-bakes/<body>/preview/`. Never touches
//!   `target/bakes/`, so iterating on the compiler doesn't invalidate
//!   the local game bake.
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

use glam::Vec3;
use image::{ImageBuffer, RgbImage};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use thalos_physics::parsing::load_solar_system_from_dir;
use thalos_terrain_gen::cubemap::{CubemapFace, dir_to_face_uv};
use thalos_terrain_gen::{
    BodyArchetype, BoundaryKind, ColdDesertField, DynamicSurfaceState, FeatureId,
    FeatureProjectionConfig, PlanetSurface, PlateKind, TerrainCompileContext,
    TerrainCompileOptions, TerrainConfig, compile_dynamic_surface_layers,
    compile_static_terrain_config, compile_tectonics_from_config, generate_initial_manifest,
    sample_surface,
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

    let targets: Vec<&thalos_physics::types::BodyDefinition> =
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
        // compiles at half the core count; `terrain_gen`'s internal
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
    body: &thalos_physics::types::BodyDefinition,
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
    let mid_freq: Option<thalos_terrain_gen::stages::MidFreqRunner> = Some(Box::new(
        move |height: &mut thalos_terrain_gen::cubemap::Cubemap<f32>,
              radius_m: f32,
              params: &thalos_terrain_gen::stages::MidFreqDetailParams|
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
        let key = thalos_terrain_gen::cache::terrain_cache_key(
            &body.terrain,
            body.tectonics.as_ref(),
            &context,
            options,
        );
        let path = thalos_terrain_gen::cache::cache_path(&bake_dir, &body.name);
        match thalos_terrain_gen::cache::store(&path, key, &static_surface) {
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
fn local_bake_is_up_to_date(body: &thalos_physics::types::BodyDefinition) -> bool {
    let context = terrain_context(body);
    // Production options — must match the values `run_terrain` uses in
    // non-preview mode, otherwise the keys diverge and we'd never see a
    // hit. Keep these in sync if `run_terrain`'s production options
    // change.
    let options = TerrainCompileOptions {
        crater_count_scale: 1.0,
        cubemap_resolution_override: None,
    };
    let expected = thalos_terrain_gen::cache::terrain_cache_key(
        &body.terrain,
        body.tectonics.as_ref(),
        &context,
        options,
    );
    let path = thalos_terrain_gen::cache::cache_path(&local_bake_dir(), &body.name);
    matches!(
        thalos_terrain_gen::cache::peek_key(&path),
        Ok(stored) if stored == expected,
    )
}

fn terrain_context(body: &thalos_physics::types::BodyDefinition) -> TerrainCompileContext {
    TerrainCompileContext {
        body_name: body.name.clone(),
        radius_m: body.radius_m as f32,
        gravity_m_s2: (body.gm / (body.radius_m * body.radius_m)) as f32,
        rotation_hours: None,
        obliquity_deg: Some((body.axial_tilt_rad as f32).to_degrees()),
        tidal_axis: matches!(body.kind, thalos_physics::types::BodyKind::Moon).then_some(Vec3::Z),
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
/// Tectonic set (when the body has tectonics): plate-id, boundary-type.
/// Debug set (`--debug`): material-id, plus biome/suture for cold-desert bodies.
fn dump_all_in_parallel(
    surface: &PlanetSurface,
    body_def: &thalos_physics::types::BodyDefinition,
    out: &Path,
    equirect_w: u32,
    debug: bool,
) {
    let static_surface = &surface.static_surface;
    let state = DynamicSurfaceState::for_layers(&surface.dynamic_layers);
    let lod = ((std::f32::consts::TAU * static_surface.radius_m) / equirect_w.max(1) as f32)
        .max(1.0)
        .log2();
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
            let sample = sample_surface(surface, &state, dir, lod);
            [
                linear_to_srgb8(sample.albedo.x),
                linear_to_srgb8(sample.albedo.y),
                linear_to_srgb8(sample.albedo.z),
            ]
        }),
        DumpKind::Height => write_equirect(out.join("height-equirect.png"), equirect_w, |dir| {
            let sample = sample_surface(surface, &state, dir, lod);
            let g = ((sample.height / height_range.max(1.0) * 0.5 + 0.5) * 255.0)
                .clamp(0.0, 255.0)
                .round() as u8;
            [g, g, g]
        }),
        DumpKind::Roughness => {
            write_equirect(out.join("roughness-equirect.png"), equirect_w, |dir| {
                let sample = sample_surface(surface, &state, dir, lod);
                let g = (sample.roughness.clamp(0.0, 1.0) * 255.0).round() as u8;
                [g, g, g]
            })
        }
        DumpKind::Normal => write_equirect(out.join("normal-equirect.png"), equirect_w, |dir| {
            let sample = sample_surface(surface, &state, dir, lod);
            [
                normal_to_u8(sample.normal.x),
                normal_to_u8(sample.normal.y),
                normal_to_u8(sample.normal.z),
            ]
        }),
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

fn cold_desert_biome_field(
    body: &thalos_physics::types::BodyDefinition,
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
    let h = thalos_terrain_gen::seeding::splitmix64(plate_id as u64 ^ 0xB1ADE0FF);
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
    let h = thalos_terrain_gen::seeding::splitmix64(id as u64 ^ 0xD3ADBEEF);
    [
        (h & 0xFF) as u8,
        ((h >> 8) & 0xFF) as u8,
        ((h >> 16) & 0xFF) as u8,
    ]
}
