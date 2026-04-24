//! `bake_dump` — headless terrain bake + PNG exporter.
//!
//! Runs a body's terrain pipeline (optionally up to a specific stage) and
//! writes the resulting cubemaps as PNG images in two layouts:
//!
//! - **Equirectangular** (2:1 lat/lon): a "map of the globe" view for
//!   reading at a glance.
//! - **Cube cross** (4:3): the 6 faces laid out in a T; for spotting
//!   seam defects across face boundaries.
//!
//! Three layers are dumped per bake: albedo, height (grayscale,
//! normalized to the body's ± range), and material ID (deterministic
//! per-ID colors).
//!
//! Usage:
//!
//!   cargo run --release -p thalos_bake_dump -- <body_name|all>
//!                                              [--up-to-stage N]
//!                                              [--out <dir>]
//!                                              [--solar-system <path>]
//!                                              [--equirect-width W]
//!
//! Body name matching is case-insensitive. Pass `all` to bake every
//! body in the solar system that has a generator block.
//!
//! Defaults:
//!
//!   --solar-system    assets/solar_system.ron
//!   --out             target/stage-bakes/<body>/
//!   --equirect-width  2048
//!
//! `--up-to-stage N` keeps only the first N stages of the generator's
//! pipeline (1-indexed). Use this to inspect output after each stage
//! individually while iterating.

use std::fs;
use std::path::{Path, PathBuf};

use glam::Vec3;
use image::{ImageBuffer, Rgb, RgbImage};
use thalos_physics::parsing::load_solar_system;
use thalos_terrain_gen::cubemap::{CubemapFace, dir_to_face_uv};
use thalos_terrain_gen::{BodyBuilder, BodyData, GeneratorParams, Pipeline, StageDef};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    /// Raw body name from the CLI (possibly `all`, mixed case).
    body_arg: String,
    up_to_stage: Option<usize>,
    /// Explicit `--out DIR`; when absent, defaults are derived per body.
    out_dir: Option<PathBuf>,
    solar_system: PathBuf,
    equirect_width: u32,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || raw.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: bake_dump <body_name|all> [--up-to-stage N] [--out DIR] [--solar-system PATH] [--equirect-width W]"
        );
        std::process::exit(if raw.is_empty() { 1 } else { 0 });
    }

    let mut body_name: Option<String> = None;
    let mut up_to_stage: Option<usize> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut solar_system: PathBuf = PathBuf::from("assets/solar_system.ron");
    let mut equirect_width: u32 = 2048;

    let mut i = 0;
    while i < raw.len() {
        let a = &raw[i];
        match a.as_str() {
            "--up-to-stage" => {
                i += 1;
                up_to_stage = Some(raw[i].parse().expect("--up-to-stage needs an integer"));
            }
            "--out" => {
                i += 1;
                out_dir = Some(PathBuf::from(&raw[i]));
            }
            "--solar-system" => {
                i += 1;
                solar_system = PathBuf::from(&raw[i]);
            }
            "--equirect-width" => {
                i += 1;
                equirect_width = raw[i].parse().expect("--equirect-width needs an integer");
            }
            s if s.starts_with("--") => panic!("unknown flag: {s}"),
            s if body_name.is_none() => body_name = Some(s.to_string()),
            s => panic!("unexpected positional arg: {s}"),
        }
        i += 1;
    }

    let body_arg = body_name.expect("body name is required");

    Args {
        body_arg,
        up_to_stage,
        out_dir,
        solar_system,
        equirect_width,
    }
}

const DEFAULT_OUT_ROOT: &str = "target/stage-bakes";

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    let source = fs::read_to_string(&args.solar_system)
        .unwrap_or_else(|e| panic!("reading {:?}: {e}", args.solar_system));
    let system = load_solar_system(&source).expect("parsing solar_system.ron");

    let targets: Vec<&thalos_physics::types::BodyDefinition> = if args.body_arg.eq_ignore_ascii_case("all") {
        let mut v: Vec<_> = system
            .bodies
            .iter()
            .filter(|b| b.generator.is_some())
            .collect();
        if v.is_empty() {
            panic!("no bodies in '{}' have a generator block", args.solar_system.display());
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
    for body in targets {
        let generator = body
            .generator
            .as_ref()
            .unwrap_or_else(|| panic!("body '{}' has no generator block", body.name))
            .clone();

        let out_dir = match (&args.out_dir, is_all) {
            // Explicit --out with a single body: use it directly.
            (Some(p), false) => p.clone(),
            // Explicit --out with `all`: treat as parent, subdirs per body.
            (Some(p), true) => p.join(&body.name),
            // Default: target/stage-bakes/<body>.
            (None, _) => PathBuf::from(DEFAULT_OUT_ROOT).join(&body.name),
        };

        bake_one(body, generator, args.up_to_stage, &out_dir, args.equirect_width);
    }
}

fn bake_one(
    body: &thalos_physics::types::BodyDefinition,
    generator: GeneratorParams,
    up_to_stage: Option<usize>,
    out_dir: &Path,
    equirect_width: u32,
) {
    let (body_data, stage_names) = run_pipeline(generator, body.radius_m as f32, up_to_stage);

    fs::create_dir_all(out_dir).expect("creating out dir");

    println!(
        "{}: baked {} stages ({} provinces, {} craters) → {}",
        body.name,
        stage_names.len(),
        body_data.provinces.len(),
        body_data.craters.len(),
        out_dir.display(),
    );

    dump_all(&body_data, out_dir, equirect_width);
    dump_info(&body_data, &stage_names, out_dir);
}

fn run_pipeline(
    mut generator: GeneratorParams,
    radius_m: f32,
    up_to_stage: Option<usize>,
) -> (BodyData, Vec<String>) {
    if let Some(n) = up_to_stage {
        generator.pipeline.truncate(n);
    }

    let mut builder = BodyBuilder::new(
        radius_m,
        generator.seed,
        generator.composition,
        generator.cubemap_resolution,
        generator.body_age_gyr,
        None,
        0.0,
    );

    let stage_names: Vec<String> = generator
        .pipeline
        .iter()
        .map(|s| stage_name(s).to_string())
        .collect();

    let stages: Vec<Box<dyn thalos_terrain_gen::Stage>> = generator
        .pipeline
        .into_iter()
        .map(|s| s.into_stage())
        .collect();
    Pipeline::new(stages).run(&mut builder);

    (builder.build(), stage_names)
}

fn stage_name(def: &StageDef) -> &'static str {
    match def {
        StageDef::Differentiate(_) => "Differentiate",
        StageDef::Biomes(_) => "Biomes",
        StageDef::Climate(_) => "Climate",
        StageDef::CoarseElevation(_) => "CoarseElevation",
        StageDef::HydrologicalCarving(_) => "HydrologicalCarving",
        StageDef::Megabasin(_) => "Megabasin",
        StageDef::Cratering(_) => "Cratering",
        StageDef::MareFlood(_) => "MareFlood",
        StageDef::OrogenDla(_) => "OrogenDla",
        StageDef::PaintBiomes(_) => "PaintBiomes",
        StageDef::Plates(_) => "Plates",
        StageDef::Regolith(_) => "Regolith",
        StageDef::Scarps(_) => "Scarps",
        StageDef::SpaceWeather(_) => "SpaceWeather",
        StageDef::SurfaceMaterials(_) => "SurfaceMaterials",
        StageDef::TectonicSkeleton(_) => "TectonicSkeleton",
        StageDef::Tectonics(_) => "Tectonics",
        StageDef::Topography(_) => "Topography",
    }
}

// ---------------------------------------------------------------------------
// Dump
// ---------------------------------------------------------------------------

fn dump_all(body: &BodyData, out: &Path, equirect_w: u32) {
    // Albedo: sRGB RGBA8 cubemap, alpha dropped.
    let albedo_shade = |dir: Vec3| -> [u8; 3] {
        let (face, u, v) = dir_to_face_uv(dir);
        let (x, y) = uv_to_texel(u, v, body.albedo_cubemap.resolution());
        let px = body.albedo_cubemap.get(face, x, y);
        [px[0], px[1], px[2]]
    };
    write_equirect(out.join("albedo-equirect.png"), equirect_w, albedo_shade);
    write_cross(
        out.join("albedo-cross.png"),
        body.albedo_cubemap.resolution(),
        albedo_shade,
    );

    // Height: u16 quantized around 0, decoded to meters via height_range.
    let range = body.height_range;
    let height_shade = |dir: Vec3| -> [u8; 3] {
        let (face, u, v) = dir_to_face_uv(dir);
        let (x, y) = uv_to_texel(u, v, body.height_cubemap.resolution());
        let raw = body.height_cubemap.get(face, x, y);
        // Decode to [-1, 1]*range, remap to [0, 1] grayscale.
        let normalized = (raw as f32 / 65535.0).clamp(0.0, 1.0);
        let luma = (normalized * 255.0) as u8;
        let _ = range; // keep it around for info.txt
        [luma, luma, luma]
    };
    write_equirect(out.join("height-equirect.png"), equirect_w, height_shade);
    write_cross(
        out.join("height-cross.png"),
        body.height_cubemap.resolution(),
        height_shade,
    );

    // Material: u8 ID cubemap, hash ID → color.
    let mat_shade = |dir: Vec3| -> [u8; 3] {
        let (face, u, v) = dir_to_face_uv(dir);
        let (x, y) = uv_to_texel(u, v, body.material_cubemap.resolution());
        let id = body.material_cubemap.get(face, x, y);
        hash_color(id as u32)
    };
    write_equirect(out.join("material-equirect.png"), equirect_w, mat_shade);
    write_cross(
        out.join("material-cross.png"),
        body.material_cubemap.resolution(),
        mat_shade,
    );
}

fn dump_info(body: &BodyData, stage_names: &[String], out: &Path) {
    let mut s = String::new();
    s.push_str(&format!("radius_m:    {}\n", body.radius_m));
    s.push_str(&format!("height_range_m: {}\n", body.height_range));
    s.push_str(&format!(
        "cubemap_resolution: {}\n",
        body.albedo_cubemap.resolution()
    ));
    if let Some(ev) = body.vertex_elevations_m.as_ref() {
        let min_e = ev.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_e = ev.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        s.push_str(&format!("vertex_elevation_m: [{min_e:.1}, {max_e:.1}]\n"));
    }
    if let Some(sl) = body.sea_level_m {
        s.push_str(&format!("sea_level_m: {sl:.1}\n"));
        if let Some(ev) = body.vertex_elevations_m.as_ref() {
            let below = ev.iter().filter(|&&e| e <= sl).count();
            let frac = below as f32 / ev.len() as f32;
            s.push_str(&format!("ocean_vertex_fraction: {frac:.3}\n"));
        }
    }
    if let Some(dg) = body.drainage_graph.as_ref() {
        let max_accum = dg
            .accumulation_m2
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);
        s.push_str(&format!("drainage_max_accumulation_m2: {max_accum:.3e}\n"));
    }
    if let Some(sed) = body.vertex_sediment_m.as_ref() {
        let max_s = sed.iter().cloned().fold(0.0f32, f32::max);
        let total_s: f32 = sed.iter().sum();
        s.push_str(&format!(
            "sediment_m: peak={max_s:.1} total={total_s:.1}\n"
        ));
        // Percentile thresholds — useful when tuning Stage 5's
        // floodplain sediment cutoff (what's the 80th percentile
        // sediment depth? etc.)
        let mut sorted: Vec<f32> = sed.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |p: f32| -> f32 {
            let idx = ((sorted.len() as f32) * p) as usize;
            sorted[idx.min(sorted.len().saturating_sub(1))]
        };
        s.push_str(&format!(
            "sediment_pct: 50%={:.2}  80%={:.2}  90%={:.2}  99%={:.2}\n",
            pct(0.50),
            pct(0.80),
            pct(0.90),
            pct(0.99),
        ));
    }
    s.push_str(&format!("stages ({}):\n", stage_names.len()));
    for (i, name) in stage_names.iter().enumerate() {
        s.push_str(&format!("  {:2}. {}\n", i + 1, name));
    }
    s.push_str(&format!("provinces:   {}\n", body.provinces.len()));
    s.push_str(&format!("craters:     {}\n", body.craters.len()));
    s.push_str(&format!("volcanoes:   {}\n", body.volcanoes.len()));
    s.push_str(&format!("channels:    {}\n", body.channels.len()));
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
    if !body.provinces.is_empty() {
        s.push_str("province kinds:\n");
        let mut counts: std::collections::BTreeMap<String, u32> = Default::default();
        for p in &body.provinces {
            *counts.entry(format!("{:?}", p.kind)).or_insert(0) += 1;
        }
        for (k, n) in counts {
            s.push_str(&format!("  {k}: {n}\n"));
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
    img.save(&path).unwrap_or_else(|e| panic!("writing {path:?}: {e}"));
}

/// Cube-cross layout:
/// ```text
///          [+Y]
///   [-X][+Z][+X][-Z]
///          [-Y]
/// ```
/// Output dimensions: `4n × 3n` where `n` is face resolution. Empty cells
/// are filled with mid-grey so the outline is visible.
fn write_cross<F: Fn(Vec3) -> [u8; 3]>(path: PathBuf, face_res: u32, shade: F) {
    let n = face_res;
    let mut img: RgbImage = ImageBuffer::from_pixel(4 * n, 3 * n, Rgb([30, 30, 30]));

    // Map: (col, row) → face orientation. Each cell shows that face.
    let cells: [(u32, u32, CubemapFace); 6] = [
        (1, 0, CubemapFace::PosY),
        (0, 1, CubemapFace::NegX),
        (1, 1, CubemapFace::PosZ),
        (2, 1, CubemapFace::PosX),
        (3, 1, CubemapFace::NegZ),
        (1, 2, CubemapFace::NegY),
    ];

    for (col, row, face) in cells {
        for py in 0..n {
            let v = (py as f32 + 0.5) / n as f32;
            for px in 0..n {
                let u = (px as f32 + 0.5) / n as f32;
                let dir = thalos_terrain_gen::cubemap::face_uv_to_dir(face, u, v);
                let [r, g, b] = shade(dir);
                img.put_pixel(col * n + px, row * n + py, Rgb([r, g, b]));
            }
        }
    }

    img.save(&path).unwrap_or_else(|e| panic!("writing {path:?}: {e}"));
}

/// Deterministic per-ID color for material/biome masks. ID 0 renders
/// mid-grey so "unset" is visually obvious.
fn hash_color(id: u32) -> [u8; 3] {
    if id == 0 {
        return [60, 60, 60];
    }
    let h = thalos_terrain_gen::seeding::splitmix64(id as u64 ^ 0xD3ADBEEF);
    [(h & 0xFF) as u8, ((h >> 8) & 0xFF) as u8, ((h >> 16) & 0xFF) as u8]
}

