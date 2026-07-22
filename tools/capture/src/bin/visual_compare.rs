//! Deterministic headless visual-comparison orchestrator.
//!
//! The game itself still renders exactly one canonical `ShipCamera`. This tool
//! sends typed variants to the persistent headless renderer, then assembles the
//! full-resolution captures into evidence an agent or human can inspect. The
//! `--cold` lane retains one clean game process per variant for final evidence.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{
    Rgba, RgbaImage,
    imageops::{FilterType, overlay, resize},
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::{SystemTime, UNIX_EPOCH},
};

const FONT_BYTES: &[u8] = include_bytes!("../../../../assets/fonts/Inter-SemiBold.ttf");
const DIFF_AMPLIFICATION: u16 = 4;
const KNOWN_PRESETS: &[&str] = &[
    "spaceport-aerial",
    "runway-atmosphere",
    "hub",
    "dry-belt",
    "earth-reference",
    "mira-orbit",
    "mira-surface",
    "mira-eva",
    "cloud-runway",
    "cloud-motion",
    "cloud-cruise",
    "cloud-interior",
    "cloud-limb",
    "cloud-sunset",
];
const INVARIANT_ENV_KEYS: &[&str] = &[
    "THALOS_SCREENSHOT_SIZE",
    "THALOS_SCREENSHOT_AZIMUTH",
    "THALOS_SCREENSHOT_ELEVATION",
    "THALOS_SCREENSHOT_DISTANCE",
    "THALOS_SCREENSHOT_WARMUP",
    "THALOS_SCREENSHOT_HUD",
    "THALOS_SSAO",
    "THALOS_TERRAIN_INSPECTION",
    "THALOS_TERRAIN_CULL",
    "THALOS_TILE_CACHE",
    "THALOS_WGPU_BACKEND",
];

#[derive(Clone, Copy, Debug)]
struct Variant {
    label: &'static str,
    value: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct Axis {
    name: &'static str,
    env_key: &'static str,
    variants: &'static [Variant],
}

const SSAO_VARIANTS: &[Variant] = &[
    Variant {
        label: "off",
        value: "off",
    },
    Variant {
        label: "on",
        value: "on",
    },
    Variant {
        label: "raw",
        value: "show",
    },
];

/// Shadow-tier axis (BL-37). Isolates the **contact tier** (W18a) against the
/// cascade rig alone — the single factor under test is whether the screen-space
/// contact march contributes, per
/// ADR-20260722T111848Z-shadows-three-tier-not-virtual-shadow-maps. `raw` paints
/// the contact term itself, which separates "the march is wrong" from "the
/// receiver applies it wrong".
const SHADOW_VARIANTS: &[Variant] = &[
    Variant {
        label: "cascade-only",
        value: "off",
    },
    Variant {
        label: "contact",
        value: "on",
    },
    Variant {
        label: "raw",
        value: "show",
    },
];

const TERRAIN_LIGHTING_VARIANTS: &[Variant] = &[
    Variant {
        label: "lit",
        value: "lit",
    },
    Variant {
        label: "fullbright",
        value: "fullbright",
    },
    Variant {
        label: "geometric-normal",
        value: "geo-normal",
    },
];

const TERRAIN_CULLING_VARIANTS: &[Variant] = &[
    Variant {
        label: "backface",
        value: "back",
    },
    Variant {
        label: "two-sided",
        value: "none",
    },
];

const TERRAIN_REGOLITH_FILTER_VARIANTS: &[Variant] = &[
    Variant {
        label: "legacy-unfiltered",
        value: "legacy-regolith",
    },
    Variant {
        label: "footprint-filtered",
        value: "lit",
    },
];

const CLOUD_RECONSTRUCTION_VARIANTS: &[Variant] = &[
    Variant {
        label: "raw",
        value: "raw",
    },
    Variant {
        label: "dense-history",
        value: "dense",
    },
    Variant {
        label: "sparse-history",
        value: "sparse",
    },
];

const AXES: &[Axis] = &[
    Axis {
        name: "ssao",
        env_key: "THALOS_SSAO",
        variants: SSAO_VARIANTS,
    },
    Axis {
        name: "shadow",
        env_key: "THALOS_CONTACT_SHADOW",
        variants: SHADOW_VARIANTS,
    },
    Axis {
        name: "terrain-lighting",
        env_key: "THALOS_TERRAIN_INSPECTION",
        variants: TERRAIN_LIGHTING_VARIANTS,
    },
    Axis {
        name: "terrain-culling",
        env_key: "THALOS_TERRAIN_CULL",
        variants: TERRAIN_CULLING_VARIANTS,
    },
    Axis {
        name: "terrain-regolith-filter",
        env_key: "THALOS_TERRAIN_INSPECTION",
        variants: TERRAIN_REGOLITH_FILTER_VARIANTS,
    },
    Axis {
        name: "cloud-reconstruction",
        env_key: "THALOS_SCREENSHOT_CLOUD_RECONSTRUCTION",
        variants: CLOUD_RECONSTRUCTION_VARIANTS,
    },
];

#[derive(Debug)]
struct Args {
    preset: String,
    axis: Axis,
    out_dir: Option<PathBuf>,
    game: Option<PathBuf>,
    cold: bool,
}

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    created_unix_s: u64,
    revision: String,
    working_tree_dirty: bool,
    preset: String,
    axis: String,
    axis_env_key: String,
    capture_mode: String,
    game_executable: String,
    invariant_environment: BTreeMap<String, Option<String>>,
    image_width: u32,
    image_height: u32,
    variants: Vec<ManifestVariant>,
    contact_sheet: String,
    comparisons: Vec<ManifestComparison>,
}

#[derive(Serialize)]
struct ManifestVariant {
    ordinal: usize,
    label: String,
    env_value: String,
    image: String,
}

#[derive(Clone, Serialize)]
struct DiffMetrics {
    mean_abs_rgb_255: f64,
    rms_rgb_255: f64,
    max_channel_delta: u8,
    changed_pixels: u64,
    changed_fraction: f64,
}

#[derive(Serialize)]
struct ManifestComparison {
    baseline: String,
    variant: String,
    diff_image: String,
    wipe_image: String,
    metrics: DiffMetrics,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("visual comparison failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let workspace = workspace_root();
    // Pipeline specialization reads terrain culling before a material pipeline
    // exists, so this structural axis cannot be changed safely in-process.
    let cold = args.cold || args.axis.name == "terrain-culling";
    if args.axis.name == "terrain-culling" && !args.cold {
        println!("terrain-culling specializes a render pipeline; using isolated cold captures");
    }
    let game = args.game.unwrap_or_else(game_binary_from_current_exe);
    let dynamic_library_path = if cold {
        if !game.is_file() {
            return Err(format!(
                "game binary not found at {}; run `just compare-cold` or pass --game <path>",
                game.display()
            ));
        }
        Some(dynamic_library_path(&game, &workspace)?)
    } else {
        None
    };

    let out_dir = absolutize(
        &workspace,
        args.out_dir.unwrap_or_else(|| {
            PathBuf::from("tools")
                .join("agent_scratch")
                .join("screenshots")
                .join("comparisons")
                .join(&args.preset)
                .join(args.axis.name)
        }),
    );
    fs::create_dir_all(&out_dir)
        .map_err(|error| format!("create {}: {error}", out_dir.display()))?;

    println!(
        "visual comparison: preset={} axis={} ({} {} captures)",
        args.preset,
        args.axis.name,
        args.axis.variants.len(),
        if cold { "isolated" } else { "persistent" },
    );

    let mut capture_paths = Vec::with_capacity(args.axis.variants.len());
    for (index, variant) in args.axis.variants.iter().enumerate() {
        let path = out_dir.join(format!("{:02}_{}.png", index + 1, variant.label));
        let report_path = path.with_extension("jsonl");
        if path.exists() {
            fs::remove_file(&path)
                .map_err(|error| format!("remove stale {}: {error}", path.display()))?;
        }
        if report_path.exists() {
            fs::remove_file(&report_path)
                .map_err(|error| format!("remove stale {}: {error}", report_path.display()))?;
        }
        println!(
            "[{}/{}] {}={}",
            index + 1,
            args.axis.variants.len(),
            args.axis.env_key,
            variant.value
        );
        let mut command = if cold {
            let mut command = Command::new(&game);
            command
                .env("THALOS_SCREENSHOT", &args.preset)
                .env("THALOS_SCREENSHOT_OUT", &path)
                .env("THALOS_SCREENSHOT_REPORT", &report_path)
                .env(args.axis.env_key, variant.value)
                .env(
                    dynamic_library_env_key(),
                    dynamic_library_path.as_ref().expect("cold path has dylibs"),
                );
            command
        } else {
            let mut command = Command::new(capture_cli_from_current_exe());
            command
                .arg("shot")
                .arg(&args.preset)
                .arg("--out")
                .arg(&path)
                .arg("--report")
                .arg(&report_path)
                .arg("--set")
                .arg(format!("{}={}", args.axis.env_key, variant.value));
            command
        };
        command.current_dir(&workspace);
        let status = command
            .status()
            .map_err(|error| format!("launch {}: {error}", game.display()))?;
        if !status.success() {
            return Err(format!("variant '{}' exited with {status}", variant.label));
        }
        if !path.is_file() {
            return Err(format!(
                "variant '{}' exited successfully without writing {}",
                variant.label,
                path.display()
            ));
        }
        capture_paths.push(path);
    }

    let images = capture_paths
        .iter()
        .map(|path| {
            image::open(path)
                .map(|image| image.into_rgba8())
                .map_err(|error| format!("read {}: {error}", path.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (width, height) = images[0].dimensions();
    if images
        .iter()
        .any(|image| image.dimensions() != (width, height))
    {
        return Err(
            "variant captures have different dimensions; comparisons require one fixed viewport"
                .into(),
        );
    }

    let font = FontRef::try_from_slice(FONT_BYTES).map_err(|error| error.to_string())?;
    let mut comparisons = Vec::new();
    let mut metrics_by_variant = vec![None; images.len()];
    for index in 1..images.len() {
        let (diff, metrics) = diff_image(&images[0], &images[index]);
        let diff_path = out_dir.join(format!("diff_01_vs_{:02}.png", index + 1));
        diff.save(&diff_path)
            .map_err(|error| format!("write {}: {error}", diff_path.display()))?;

        let wipe = wipe_image(&images[0], &images[index]);
        let wipe_path = out_dir.join(format!("wipe_01_vs_{:02}.png", index + 1));
        wipe.save(&wipe_path)
            .map_err(|error| format!("write {}: {error}", wipe_path.display()))?;

        metrics_by_variant[index] = Some(metrics.clone());
        comparisons.push(ManifestComparison {
            baseline: args.axis.variants[0].label.to_owned(),
            variant: args.axis.variants[index].label.to_owned(),
            diff_image: relative_display(&workspace, &diff_path),
            wipe_image: relative_display(&workspace, &wipe_path),
            metrics,
        });
    }

    let contact_sheet = make_contact_sheet(
        &images,
        args.axis.variants,
        &metrics_by_variant,
        &args.preset,
        args.axis.name,
        &font,
    );
    let contact_path = out_dir.join("contact_sheet.png");
    contact_sheet
        .save(&contact_path)
        .map_err(|error| format!("write {}: {error}", contact_path.display()))?;

    let revision =
        git_output(&workspace, &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let working_tree_dirty =
        git_output(&workspace, &["status", "--porcelain"]).is_some_and(|output| !output.is_empty());
    let invariant_environment = INVARIANT_ENV_KEYS
        .iter()
        .filter(|key| **key != args.axis.env_key)
        .map(|key| ((*key).to_owned(), env::var(key).ok()))
        .collect();
    let variants = args
        .axis
        .variants
        .iter()
        .enumerate()
        .map(|(index, variant)| ManifestVariant {
            ordinal: index + 1,
            label: variant.label.to_owned(),
            env_value: variant.value.to_owned(),
            image: relative_display(&workspace, &capture_paths[index]),
        })
        .collect();
    let manifest = Manifest {
        schema_version: 2,
        created_unix_s: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        revision,
        working_tree_dirty,
        preset: args.preset,
        axis: args.axis.name.to_owned(),
        axis_env_key: args.axis.env_key.to_owned(),
        capture_mode: if cold { "cold" } else { "persistent" }.to_owned(),
        game_executable: game.display().to_string(),
        invariant_environment,
        image_width: width,
        image_height: height,
        variants,
        contact_sheet: relative_display(&workspace, &contact_path),
        comparisons,
    };
    let manifest_path = out_dir.join("manifest.json");
    let manifest_json = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("serialize manifest: {error}"))?;
    fs::write(&manifest_path, manifest_json)
        .map_err(|error| format!("write {}: {error}", manifest_path.display()))?;

    println!("comparison artifacts: {}", out_dir.display());
    println!("contact sheet: {}", contact_path.display());
    Ok(())
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut positional = Vec::new();
    let mut out_dir = None;
    let mut game = None;
    let mut cold = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            "--list" => {
                print_axes();
                return Ok(None);
            }
            "--out" => {
                out_dir = Some(PathBuf::from(args.next().ok_or("--out requires a path")?));
            }
            "--game" => {
                game = Some(PathBuf::from(args.next().ok_or("--game requires a path")?));
            }
            "--cold" => cold = true,
            option if option.starts_with('-') => {
                return Err(format!("unknown option '{option}'"));
            }
            value => positional.push(value.to_owned()),
        }
    }
    if positional.len() > 2 {
        return Err("expected at most <preset> <axis>; use --help for usage".into());
    }

    let preset = canonical_slug(positional.first().map_or("earth-reference", String::as_str));
    if !KNOWN_PRESETS.contains(&preset.as_str()) {
        return Err(format!(
            "unknown preset '{preset}'; expected one of {}",
            KNOWN_PRESETS.join(", ")
        ));
    }
    let axis_name = canonical_slug(positional.get(1).map_or("ssao", String::as_str));
    let axis = AXES
        .iter()
        .copied()
        .find(|axis| axis.name == axis_name)
        .ok_or_else(|| {
            format!(
                "unknown axis '{axis_name}'; expected one of {}",
                AXES.iter()
                    .map(|axis| axis.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;

    Ok(Some(Args {
        preset,
        axis,
        out_dir,
        game,
        cold,
    }))
}

fn print_help() {
    println!(
        "\
Usage: visual_compare [preset] [axis] [--out DIR] [--cold] [--game PATH]\n\
       visual_compare --list\n\n\
Defaults: preset=earth-reference axis=ssao\n\
Artifacts: artifacts/visual/runs/comparisons/<preset>/<axis>/\n\
Run through: just compare [preset] [axis]"
    );
    print_axes();
}

fn print_axes() {
    println!("Presets: {}", KNOWN_PRESETS.join(", "));
    println!("Axes:");
    for axis in AXES {
        println!(
            "  {} ({}): {}",
            axis.name,
            axis.env_key,
            axis.variants
                .iter()
                .map(|variant| format!("{}={}", variant.label, variant.value))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

fn canonical_slug(raw: &str) -> String {
    raw.trim().to_ascii_lowercase().replace('_', "-")
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("capture tool lives under <workspace>/tools/capture")
        .to_path_buf()
}

fn game_binary_from_current_exe() -> PathBuf {
    let current = env::current_exe().expect("resolve visual_compare executable");
    let profile_dir = current
        .parent()
        .expect("visual_compare lives under <target>/<profile>");
    profile_dir.join(format!("thalos_capture_host{}", env::consts::EXE_SUFFIX))
}

fn capture_cli_from_current_exe() -> PathBuf {
    env::current_exe()
        .expect("resolve visual_compare executable")
        .parent()
        .expect("visual_compare lives under <target>/<profile>")
        .join(format!("thalos_capture{}", env::consts::EXE_SUFFIX))
}

/// Reproduce Cargo's dynamic-library launch environment when the comparison
/// recipe runs the already-built orchestrator directly. `bevy/dynamic_linking`
/// puts the shared library under the profile/deps directory, while Rust dylibs
/// depend on the dynamic standard library in rustc's target libdir.
/// Without all three paths Windows exits with `0xc0000135` before `main`, while
/// Unix loaders report a missing shared object/dylib.
fn dynamic_library_path(game: &Path, workspace: &Path) -> Result<std::ffi::OsString, String> {
    let profile_dir = game
        .parent()
        .ok_or_else(|| format!("game path has no parent: {}", game.display()))?;
    let key = dynamic_library_env_key();
    let mut paths = vec![profile_dir.to_path_buf(), profile_dir.join("deps")];
    paths.push(rustc_target_libdir(workspace)?);
    if let Some(existing) = env::var_os(key) {
        paths.extend(env::split_paths(&existing));
    }
    env::join_paths(paths)
        .map_err(|error| format!("construct {key} for dynamic game launch: {error}"))
}

fn rustc_target_libdir(workspace: &Path) -> Result<PathBuf, String> {
    let output = Command::new("rustc")
        .current_dir(workspace)
        .args(["--print", "target-libdir"])
        .output()
        .map_err(|error| format!("query rustc target-libdir: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "rustc --print target-libdir exited with {}",
            output.status
        ));
    }
    let raw = String::from_utf8_lossy(&output.stdout);
    let path = PathBuf::from(raw.trim());
    if !path.is_dir() {
        return Err(format!(
            "rustc target-libdir is not a directory: {}",
            path.display()
        ));
    }
    Ok(path)
}

#[cfg(target_os = "windows")]
fn dynamic_library_env_key() -> &'static str {
    "PATH"
}

#[cfg(target_os = "macos")]
fn dynamic_library_env_key() -> &'static str {
    "DYLD_FALLBACK_LIBRARY_PATH"
}

#[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
fn dynamic_library_env_key() -> &'static str {
    "LD_LIBRARY_PATH"
}

fn absolutize(workspace: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        workspace.join(path)
    }
}

fn git_output(workspace: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(workspace)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn relative_display(workspace: &Path, path: &Path) -> String {
    path.strip_prefix(workspace)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn diff_image(baseline: &RgbaImage, variant: &RgbaImage) -> (RgbaImage, DiffMetrics) {
    let mut diff = RgbaImage::new(baseline.width(), baseline.height());
    let mut absolute_sum = 0_u64;
    let mut squared_sum = 0_u64;
    let mut max_delta = 0_u8;
    let mut changed_pixels = 0_u64;

    for (x, y, out) in diff.enumerate_pixels_mut() {
        let a = baseline.get_pixel(x, y).0;
        let b = variant.get_pixel(x, y).0;
        let mut rgb = [0_u8; 3];
        let mut changed = false;
        for channel in 0..3 {
            let delta = a[channel].abs_diff(b[channel]);
            changed |= delta != 0;
            absolute_sum += u64::from(delta);
            squared_sum += u64::from(delta) * u64::from(delta);
            max_delta = max_delta.max(delta);
            rgb[channel] = (u16::from(delta) * DIFF_AMPLIFICATION).min(255) as u8;
        }
        changed_pixels += u64::from(changed);
        *out = Rgba([rgb[0], rgb[1], rgb[2], 255]);
    }

    let pixel_count = u64::from(baseline.width()) * u64::from(baseline.height());
    let sample_count = (pixel_count * 3).max(1) as f64;
    (
        diff,
        DiffMetrics {
            mean_abs_rgb_255: absolute_sum as f64 / sample_count,
            rms_rgb_255: (squared_sum as f64 / sample_count).sqrt(),
            max_channel_delta: max_delta,
            changed_pixels,
            changed_fraction: changed_pixels as f64 / pixel_count.max(1) as f64,
        },
    )
}

fn wipe_image(baseline: &RgbaImage, variant: &RgbaImage) -> RgbaImage {
    let mut wipe = baseline.clone();
    let split = wipe.width() / 2;
    for y in 0..wipe.height() {
        for x in split..wipe.width() {
            wipe.put_pixel(x, y, *variant.get_pixel(x, y));
        }
    }
    for x in split.saturating_sub(2)..=(split + 2).min(wipe.width().saturating_sub(1)) {
        for y in 0..wipe.height() {
            wipe.put_pixel(x, y, Rgba([255, 74, 156, 255]));
        }
    }
    wipe
}

fn make_contact_sheet(
    images: &[RgbaImage],
    variants: &[Variant],
    metrics: &[Option<DiffMetrics>],
    preset: &str,
    axis: &str,
    font: &FontRef<'_>,
) -> RgbaImage {
    const MARGIN: u32 = 24;
    const GAP: u32 = 18;
    const TITLE_H: u32 = 64;
    const LABEL_H: u32 = 50;
    const MAX_TILE_W: u32 = 900;
    const MAX_TILE_H: u32 = 560;

    let source_w = images[0].width();
    let source_h = images[0].height();
    let scale = (MAX_TILE_W as f64 / source_w as f64)
        .min(MAX_TILE_H as f64 / source_h as f64)
        .min(1.0);
    let tile_w = (source_w as f64 * scale).round().max(1.0) as u32;
    let tile_h = (source_h as f64 * scale).round().max(1.0) as u32;
    let columns = (images.len() as f64).sqrt().ceil() as u32;
    let rows = (images.len() as u32).div_ceil(columns);
    let sheet_w = MARGIN * 2 + columns * tile_w + columns.saturating_sub(1) * GAP;
    let sheet_h = MARGIN * 2 + TITLE_H + rows * (LABEL_H + tile_h) + rows.saturating_sub(1) * GAP;
    let mut sheet = RgbaImage::from_pixel(sheet_w, sheet_h, Rgba([15, 18, 24, 255]));
    draw_text(
        &mut sheet,
        font,
        MARGIN as f32,
        MARGIN as f32,
        30.0,
        &format!("{} / {}", preset, axis),
        Rgba([237, 242, 250, 255]),
    );

    for (index, image) in images.iter().enumerate() {
        let column = index as u32 % columns;
        let row = index as u32 / columns;
        let x = MARGIN + column * (tile_w + GAP);
        let y = MARGIN + TITLE_H + row * (LABEL_H + tile_h + GAP);
        fill_rect(&mut sheet, x, y, tile_w, LABEL_H, Rgba([28, 34, 45, 255]));
        let metric = metrics[index]
            .as_ref()
            .map(|metric| {
                format!(
                    " · MAE {:.2}/255 · {:.1}% px",
                    metric.mean_abs_rgb_255,
                    metric.changed_fraction * 100.0
                )
            })
            .unwrap_or_else(|| " · baseline".to_owned());
        draw_text(
            &mut sheet,
            font,
            (x + 14) as f32,
            (y + 10) as f32,
            22.0,
            &format!("{:02} {}{}", index + 1, variants[index].label, metric),
            Rgba([219, 228, 241, 255]),
        );
        let thumbnail = resize(image, tile_w, tile_h, FilterType::Lanczos3);
        overlay(&mut sheet, &thumbnail, i64::from(x), i64::from(y + LABEL_H));
    }
    sheet
}

fn fill_rect(image: &mut RgbaImage, x: u32, y: u32, width: u32, height: u32, color: Rgba<u8>) {
    for py in y..(y + height).min(image.height()) {
        for px in x..(x + width).min(image.width()) {
            image.put_pixel(px, py, color);
        }
    }
}

fn draw_text(
    image: &mut RgbaImage,
    font: &FontRef<'_>,
    x: f32,
    y: f32,
    size: f32,
    text: &str,
    color: Rgba<u8>,
) {
    let scale = PxScale::from(size);
    let scaled = font.as_scaled(scale);
    let mut pen_x = x;
    let baseline_y = y + scaled.ascent();
    let mut previous = None;
    for character in text.chars() {
        let id = scaled.glyph_id(character);
        if let Some(previous) = previous {
            pen_x += scaled.kern(previous, id);
        }
        let glyph = id.with_scale_and_position(scale, ab_glyph::point(pen_x, baseline_y));
        if let Some(outlined) = font.outline_glyph(glyph) {
            let bounds = outlined.px_bounds();
            outlined.draw(|gx, gy, coverage| {
                let px = bounds.min.x as i32 + gx as i32;
                let py = bounds.min.y as i32 + gy as i32;
                if px < 0 || py < 0 || px >= image.width() as i32 || py >= image.height() as i32 {
                    return;
                }
                let destination = image.get_pixel_mut(px as u32, py as u32);
                let alpha = coverage * (f32::from(color[3]) / 255.0);
                for channel in 0..3 {
                    destination[channel] = (f32::from(destination[channel]) * (1.0 - alpha)
                        + f32::from(color[channel]) * alpha)
                        .round() as u8;
                }
                destination[3] = 255;
            });
        }
        pen_x += scaled.h_advance(id);
        previous = Some(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_axis_has_unique_labels_and_multiple_variants() {
        for axis in AXES {
            assert!(axis.variants.len() >= 2);
            let mut labels = axis
                .variants
                .iter()
                .map(|variant| variant.label)
                .collect::<Vec<_>>();
            labels.sort_unstable();
            labels.dedup();
            assert_eq!(labels.len(), axis.variants.len());
        }
    }

    #[test]
    fn identical_images_have_zero_diff() {
        let image = RgbaImage::from_pixel(2, 2, Rgba([10, 20, 30, 255]));
        let (_, metrics) = diff_image(&image, &image);
        assert_eq!(metrics.mean_abs_rgb_255, 0.0);
        assert_eq!(metrics.rms_rgb_255, 0.0);
        assert_eq!(metrics.max_channel_delta, 0);
        assert_eq!(metrics.changed_pixels, 0);
        assert_eq!(metrics.changed_fraction, 0.0);
    }

    #[test]
    fn diff_metrics_count_changed_pixels_and_channels() {
        let baseline = RgbaImage::from_pixel(1, 1, Rgba([0, 10, 20, 255]));
        let variant = RgbaImage::from_pixel(1, 1, Rgba([3, 10, 24, 255]));
        let (diff, metrics) = diff_image(&baseline, &variant);
        assert_eq!(diff.get_pixel(0, 0), &Rgba([12, 0, 16, 255]));
        assert_eq!(metrics.max_channel_delta, 4);
        assert_eq!(metrics.changed_pixels, 1);
        assert_eq!(metrics.changed_fraction, 1.0);
        assert!((metrics.mean_abs_rgb_255 - 7.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn direct_launch_uses_the_platform_library_path() {
        let key = dynamic_library_env_key();
        if cfg!(target_os = "windows") {
            assert_eq!(key, "PATH");
        } else if cfg!(target_os = "macos") {
            assert_eq!(key, "DYLD_FALLBACK_LIBRARY_PATH");
        } else {
            assert_eq!(key, "LD_LIBRARY_PATH");
        }
    }

    #[test]
    fn direct_launch_includes_profile_deps_and_rust_target_libdir() {
        let workspace = workspace_root();
        let profile = workspace.join("target").join("debug");
        let game = profile.join(format!("thalos_game{}", env::consts::EXE_SUFFIX));
        let joined = dynamic_library_path(&game, &workspace).expect("construct loader path");
        let paths = env::split_paths(&joined).collect::<Vec<_>>();

        assert_eq!(paths[0], profile);
        assert_eq!(paths[1], profile.join("deps"));
        assert_eq!(
            paths[2],
            rustc_target_libdir(&workspace).expect("resolve rustc target libdir")
        );
    }
}
