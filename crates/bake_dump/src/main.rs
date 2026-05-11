//! `bake_dump` — headless terrain bake + PNG exporter.
//!
//! Runs a body's terrain compiler and writes the resulting cubemaps as
//! equirectangular PNG images:
//!
//! - **Equirectangular** (2:1 lat/lon): a "map of the globe" view for
//!   reading at a glance.
//!
//! Three core layers are dumped per bake: albedo, height (grayscale,
//! normalized to the body's ± range), and material ID (deterministic
//! per-ID colors). Feature cold-desert bodies also emit biome/suture maps so
//! process regions can be evaluated before albedo hides the structure.
//!
//! Usage:
//!
//!   cargo run --release -p thalos_bake_dump -- <body_name|all>
//!                                              [--out <dir>]
//!                                              [--solar-system <path>]
//!                                              [--equirect-width W]
//!
//! Body name matching is case-insensitive. Pass `all` to bake every
//! body in the solar system that has terrain.
//!
//! Defaults:
//!
//!   --solar-system    assets/solar_system.ron
//!   --out             stage-bakes/<body>/
//!   --cubemap-resolution 512
//!   --equirect-width     512

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};

use glam::Vec3;
use image::{ImageBuffer, Rgb, RgbImage};
use thalos_physics::parsing::load_solar_system_from_dir;
use thalos_terrain_gen::cubemap::{CubemapFace, dir_to_face_uv};
use thalos_terrain_gen::{
    BodyArchetype, BoundaryKind, ColdDesertField, DynamicSurfaceState, FeatureId,
    FeatureProjectionConfig, PlanetSurface, PlateKind, StaticSurfaceData, TectonicSystem,
    TerrainCompileContext, TerrainCompileOptions, TerrainConfig, compile_dynamic_surface_layers,
    compile_static_terrain_config, compile_tectonics_from_config, generate_initial_manifest,
    sample_surface,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    /// Raw body name from the CLI (possibly `all`, mixed case).
    body_arg: String,
    /// Explicit `--out DIR`; when absent, defaults are derived per body.
    out_dir: Option<PathBuf>,
    solar_system: PathBuf,
    /// Explicit `--cubemap-resolution N` override. `None` defers to the body
    /// (per-body override → radius-derived default).
    cubemap_resolution: Option<u32>,
    equirect_width: u32,
    /// Emit debug-only dumps (biome / suture / material-id) alongside the
    /// production PBR set. Off by default — production bakes ship only the
    /// four cubemaps the impostor consumes.
    debug: bool,
}

#[derive(Clone, Copy, Debug)]
enum CacheStatus {
    Hit,
    Stored,
    StoreFailed,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || raw.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: bake_dump <body_name|all> [--out DIR] [--solar-system PATH] [--cubemap-resolution N | --full] [--equirect-width W] [--debug]"
        );
        std::process::exit(if raw.is_empty() { 1 } else { 0 });
    }

    let mut body_name: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut solar_system: PathBuf = PathBuf::from("assets/solar_system.ron");
    // Iteration default: a fixed 512² preview, regardless of the body's
    // authored or radius-derived resolution. Full-res bakes are paid for
    // explicitly via `--full` (defer to body) or `--cubemap-resolution N`
    // (force a specific size). The editor's Full button takes the same path.
    let mut cubemap_resolution: Option<u32> = Some(512);
    let mut explicit_resolution: Option<&'static str> = None;
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
            "--cubemap-resolution" => {
                if let Some(prior) = explicit_resolution {
                    panic!("--cubemap-resolution conflicts with prior {prior}");
                }
                i += 1;
                cubemap_resolution = Some(
                    raw[i]
                        .parse()
                        .expect("--cubemap-resolution needs an integer"),
                );
                explicit_resolution = Some("--cubemap-resolution");
            }
            "--full" => {
                if let Some(prior) = explicit_resolution {
                    panic!("--full conflicts with prior {prior}");
                }
                cubemap_resolution = None;
                explicit_resolution = Some("--full");
            }
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
        cubemap_resolution,
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

    let is_all = targets.len() > 1;
    let jobs: Vec<_> = targets
        .into_iter()
        .map(|body| {
            let out_dir = match (&args.out_dir, is_all) {
                // Explicit --out with a single body: use it directly.
                (Some(p), false) => p.clone(),
                // Explicit --out with `all`: treat as parent, subdirs per body.
                (Some(p), true) => p.join(&body.name),
                // Default: stage-bakes/<body>.
                (None, _) => PathBuf::from(DEFAULT_OUT_ROOT).join(&body.name),
            };
            (body, out_dir)
        })
        .collect();

    if is_all {
        bake_all_in_child_processes(&args, &jobs);
    } else {
        for (body, out_dir) in &jobs {
            bake_one(
                body,
                out_dir,
                args.cubemap_resolution,
                args.equirect_width,
                args.debug,
            );
        }
    }
}

fn bake_all_in_child_processes(
    args: &Args,
    jobs: &[(&thalos_physics::types::BodyDefinition, PathBuf)],
) {
    let exe = std::env::current_exe().expect("locating bake_dump executable");
    let mut children: Vec<(String, Child)> = Vec::with_capacity(jobs.len());

    for (body, out_dir) in jobs {
        let mut cmd = Command::new(&exe);
        cmd.arg(&body.name)
            .arg("--out")
            .arg(out_dir)
            .arg("--solar-system")
            .arg(&args.solar_system)
            .arg("--equirect-width")
            .arg(args.equirect_width.to_string());
        match args.cubemap_resolution {
            Some(res) => {
                cmd.arg("--cubemap-resolution").arg(res.to_string());
            }
            None => {
                cmd.arg("--full");
            }
        }
        if args.debug {
            cmd.arg("--debug");
        }
        let child = cmd
            .spawn()
            .unwrap_or_else(|e| panic!("spawning bake_dump for {}: {e}", body.name));
        children.push((body.name.clone(), child));
    }

    let mut failed = false;
    for (name, mut child) in children {
        let status = child
            .wait()
            .unwrap_or_else(|e| panic!("waiting for bake_dump child {name}: {e}"));
        if !status.success() {
            eprintln!("bake_dump child for {name} exited with {status}");
            failed = true;
        }
    }
    if failed {
        std::process::exit(1);
    }
}

fn bake_one(
    body: &thalos_physics::types::BodyDefinition,
    out_dir: &Path,
    cubemap_resolution: Option<u32>,
    equirect_width: u32,
    debug: bool,
) {
    let (surface, route, cache_status) = run_terrain(body, cubemap_resolution);
    let static_surface = &surface.static_surface;

    fs::create_dir_all(out_dir).expect("creating out dir");
    remove_legacy_outputs(out_dir, debug);

    println!(
        "{}: baked via {} ({}, {} craters) → {}",
        body.name,
        route,
        cache_status_label(cache_status),
        static_surface.craters.len(),
        out_dir.display(),
    );

    dump_pbr_set(&surface, out_dir, equirect_width);
    if let Some(tectonics) = surface.tectonics.as_ref() {
        dump_tectonic_set(tectonics, out_dir, equirect_width);
    }
    if debug {
        dump_debug_set(static_surface, body, out_dir, equirect_width);
    }
    dump_info(&surface, &route, out_dir);
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

fn run_terrain(
    body: &thalos_physics::types::BodyDefinition,
    cubemap_resolution: Option<u32>,
) -> (PlanetSurface, String, CacheStatus) {
    let route = body.terrain.route_label();
    let context = terrain_context(body);
    let options = TerrainCompileOptions {
        crater_count_scale: 1.0,
        cubemap_resolution_override: cubemap_resolution,
    };
    let cache_dir = terrain_cache_dir();
    let key = thalos_terrain_gen::cache::terrain_cache_key(
        &body.terrain,
        body.tectonics.as_ref(),
        &context,
        options,
    );
    let path = thalos_terrain_gen::cache::cache_path(&cache_dir, &body.name, key);

    let dynamic_layers = compile_dynamic_surface_layers(&body.terrain, &context)
        .unwrap_or_else(|e| panic!("dynamic layer compile failed for {}: {e}", body.name));
    // Tectonics regenerates on every load (it's not cached). Build on both
    // the cache-hit and cache-miss paths so the editor and equirect dumpers
    // always see a consistent layer; downstream archetypes that read the
    // tectonic graph (currently AgingOceanicHomeworld) consume the same
    // instance.
    let tectonics = compile_tectonics_from_config(body.tectonics.as_ref(), &context);
    if let Some(static_surface) = thalos_terrain_gen::cache::load(&path, key) {
        return (
            PlanetSurface {
                static_surface,
                dynamic_layers,
                tectonics,
            },
            route,
            CacheStatus::Hit,
        );
    }

    let static_surface =
        compile_static_terrain_config(&body.terrain, tectonics.as_ref(), &context, options)
            .unwrap_or_else(|e| panic!("terrain compile failed for {}: {e}", body.name));
    let cache_status = match thalos_terrain_gen::cache::store(&path, key, &static_surface) {
        Ok(()) => CacheStatus::Stored,
        Err(e) => {
            eprintln!("terrain cache write failed for {}: {e}", body.name);
            CacheStatus::StoreFailed
        }
    };
    (
        PlanetSurface {
            static_surface,
            dynamic_layers,
            tectonics,
        },
        route,
        cache_status,
    )
}

fn terrain_cache_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/terrain_cache")
}

fn cache_status_label(status: CacheStatus) -> &'static str {
    match status {
        CacheStatus::Hit => "cache hit",
        CacheStatus::Stored => "cache stored",
        CacheStatus::StoreFailed => "cache store failed",
    }
}

// ---------------------------------------------------------------------------
// Dump
// ---------------------------------------------------------------------------

/// Production PBR set: albedo, height, roughness, normal. The four cubemaps
/// the impostor shader actually consumes.
fn dump_pbr_set(surface: &PlanetSurface, out: &Path, equirect_w: u32) {
    let body = &surface.static_surface;
    let state = DynamicSurfaceState::for_layers(&surface.dynamic_layers);
    let lod = ((std::f32::consts::TAU * body.radius_m) / equirect_w.max(1) as f32)
        .max(1.0)
        .log2();
    let albedo_shade = |dir: Vec3| -> [u8; 3] {
        let sample = sample_surface(surface, &state, dir, lod);
        [
            linear_to_srgb8(sample.albedo.x),
            linear_to_srgb8(sample.albedo.y),
            linear_to_srgb8(sample.albedo.z),
        ]
    };
    write_equirect(out.join("albedo-equirect.png"), equirect_w, albedo_shade);

    let height_shade = |dir: Vec3| -> [u8; 3] {
        let sample = sample_surface(surface, &state, dir, lod);
        let g = ((sample.height / body.height_range.max(1.0) * 0.5 + 0.5) * 255.0)
            .clamp(0.0, 255.0)
            .round() as u8;
        [g, g, g]
    };
    write_equirect(out.join("height-equirect.png"), equirect_w, height_shade);

    let roughness_shade = |dir: Vec3| -> [u8; 3] {
        let sample = sample_surface(surface, &state, dir, lod);
        let g = (sample.roughness.clamp(0.0, 1.0) * 255.0).round() as u8;
        [g, g, g]
    };
    write_equirect(
        out.join("roughness-equirect.png"),
        equirect_w,
        roughness_shade,
    );

    let normal_shade = |dir: Vec3| -> [u8; 3] {
        let sample = sample_surface(surface, &state, dir, lod);
        [
            normal_to_u8(sample.normal.x),
            normal_to_u8(sample.normal.y),
            normal_to_u8(sample.normal.z),
        ]
    };
    write_equirect(out.join("normal-equirect.png"), equirect_w, normal_shade);
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

/// Debug overlays — material-id, biome, suture maps. Useful when iterating on
/// generation; not part of the production rendering path.
fn dump_debug_set(
    body: &StaticSurfaceData,
    body_def: &thalos_physics::types::BodyDefinition,
    out: &Path,
    equirect_w: u32,
) {
    let mat_shade = |dir: Vec3| -> [u8; 3] {
        let (face, u, v) = dir_to_face_uv(dir);
        let (x, y) = uv_to_texel(u, v, body.material_cubemap.resolution());
        let id = body.material_cubemap.get(face, x, y);
        hash_color(id as u32)
    };
    write_equirect(out.join("material-equirect.png"), equirect_w, mat_shade);

    let Some(field) = cold_desert_biome_field(body_def) else {
        return;
    };
    let biome_shade = |dir: Vec3| -> [u8; 3] { field.debug_biome_color_srgb(dir) };
    write_equirect(out.join("biome-equirect.png"), equirect_w, biome_shade);

    let suture_shade = |dir: Vec3| -> [u8; 3] { field.sample_suture_debug(dir).debug_color_srgb() };
    write_equirect(out.join("suture-equirect.png"), equirect_w, suture_shade);
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

// ---------------------------------------------------------------------------
// Tectonic equirects
// ---------------------------------------------------------------------------

/// Tectonic equirects: plate-id colored regions, and a boundary-type overlay
/// fading by distance. Both are written when a body has a tectonic layer,
/// regardless of `--debug`, because seeing the plates is the deliverable.
fn dump_tectonic_set(tectonics: &TectonicSystem, out: &Path, equirect_w: u32) {
    // Threshold for boundary-line rendering: 8% of the body radius. Wide
    // enough that the boundary reads from orbit at typical equirect
    // resolutions; narrow enough that interior cells stay clean. Tune
    // visually if it looks wrong.
    let threshold_m = tectonics.body_radius_m * 0.08;

    let plate_shade = |dir: glam::Vec3| -> [u8; 3] {
        let sample = tectonics.sample(dir);
        plate_color_srgb(sample.plate_id.0, sample.plate_kind)
    };
    write_equirect(out.join("plate-id-equirect.png"), equirect_w, plate_shade);

    let boundary_shade = |dir: glam::Vec3| -> [u8; 3] {
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
    };
    write_equirect(
        out.join("boundary-type-equirect.png"),
        equirect_w,
        boundary_shade,
    );
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
/// Center of image is direction `+Z`.
fn write_equirect<F: Fn(Vec3) -> [u8; 3] + Sync>(path: PathBuf, width: u32, shade: F) {
    let height = width / 2;
    let mut img: RgbImage = ImageBuffer::new(width, height);
    for y in 0..height {
        let lat = (0.5 - (y as f32 + 0.5) / height as f32) * std::f32::consts::PI;
        let (sl, cl) = lat.sin_cos();
        for x in 0..width {
            let lon = ((x as f32 + 0.5) / width as f32 - 0.5) * std::f32::consts::TAU;
            let (sln, cln) = lon.sin_cos();
            let dir = Vec3::new(cl * sln, sl, cl * cln);
            let [r, g, b] = shade(dir);
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
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
