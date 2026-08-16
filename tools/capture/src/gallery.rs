//! Zero-GPU visual discovery for the shared viewpoint catalog.
//!
//! Gallery generation is deliberately an offline image operation. A thumbnail
//! answers "is this framing relevant?" and is never capture evidence; the
//! original PNG and its receipt remain the authority for visual verification.

use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use image::{
    DynamicImage, Rgba, RgbaImage,
    imageops::{FilterType, overlay, resize},
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use thalos_capture_protocol::{CaptureSourceSnapshot, ViewpointCatalog};

const FONT_BYTES: &[u8] = include_bytes!("../../../assets/fonts/Inter-SemiBold.ttf");
const THUMBNAIL_WIDTH: u32 = 320;
const THUMBNAIL_HEIGHT: u32 = 180;
const CARD_WIDTH: u32 = 352;
const CARD_HEIGHT: u32 = 264;
const CARD_GAP: u32 = 16;
const SHEET_MARGIN: u32 = 20;
const SHEET_HEADER_HEIGHT: u32 = 74;
const MAX_COLUMNS: u32 = 4;

#[derive(Debug, Default, Deserialize)]
struct ReceiptSummary {
    #[serde(default)]
    completed_unix_ms: u128,
    #[serde(default)]
    source: CaptureSourceSnapshot,
    #[serde(default)]
    workspace_matches: bool,
}

#[derive(Clone, Debug, Serialize)]
struct GalleryEntry {
    id: String,
    name: String,
    kind: &'static str,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    driver: Option<String>,
    cache_state: &'static str,
    source_image: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_receipt: Option<String>,
    thumbnail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    completed_unix_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capture_source_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capture_git_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace_matched_at_capture: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    issue: Option<String>,
}

#[derive(Debug, Serialize)]
struct GalleryCounts {
    total: usize,
    current: usize,
    stale: usize,
    cached: usize,
    unattributed: usize,
    missing: usize,
    unreadable: usize,
}

#[derive(Debug, Serialize)]
struct GalleryIndex {
    schema: &'static str,
    generated_unix_ms: u128,
    catalog: String,
    latest_capture_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    current_source_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    contact_sheet: Option<String>,
    counts: GalleryCounts,
    entries: Vec<GalleryEntry>,
}

#[derive(Debug, Default, PartialEq)]
struct ListArgs {
    gallery: bool,
    json: bool,
    output: Option<PathBuf>,
    help: bool,
}

pub(crate) fn run_cli(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let target = args
        .next()
        .ok_or("list requires a target; currently supported: viewpoints")?;
    if target == "-h" || target == "--help" || target == "help" {
        print_help();
        return Ok(());
    }
    if target != "viewpoints" {
        return Err(format!(
            "unknown list target {target:?}; currently supported: viewpoints"
        ));
    }
    let args = parse_list_args(args)?;
    if args.help {
        return Ok(());
    }
    let catalog = super::read_viewpoint_catalog()?;
    let workspace = super::workspace_root();
    let latest = workspace.join("artifacts/visual/latest");
    let output = args
        .output
        .map(super::absolute)
        .unwrap_or_else(|| workspace.join("artifacts/visual/catalog/viewpoints"));
    let current_source = super::capture_source_snapshot().ok();

    if args.gallery {
        let index = build_gallery(
            &catalog,
            &super::viewpoint_catalog_path(),
            &latest,
            &output,
            current_source.as_ref(),
            &workspace,
        )?;
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&index).map_err(|error| error.to_string())?
            );
        } else {
            let contact_sheet = index
                .contact_sheet
                .as_deref()
                .unwrap_or("(contact sheet unavailable)");
            println!(
                "viewpoint gallery: {contact_sheet}\nindex: {}\n{} cached · {} missing · {} unreadable",
                portable_path(&output.join("index.json"), &workspace),
                index.counts.current
                    + index.counts.stale
                    + index.counts.cached
                    + index.counts.unattributed,
                index.counts.missing,
                index.counts.unreadable,
            );
        }
    } else {
        let index = inspect_catalog(
            &catalog,
            &super::viewpoint_catalog_path(),
            &latest,
            None,
            current_source.as_ref(),
            &workspace,
        );
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&index).map_err(|error| error.to_string())?
            );
        } else {
            print_text_listing(&index);
        }
    }
    Ok(())
}

fn parse_list_args(mut args: impl Iterator<Item = String>) -> Result<ListArgs, String> {
    let mut parsed = ListArgs::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--gallery" => parsed.gallery = true,
            "--json" => parsed.json = true,
            "--out" => {
                parsed.output = Some(PathBuf::from(
                    args.next().ok_or("--out requires a directory")?,
                ));
            }
            "-h" | "--help" | "help" => {
                print_help();
                parsed.help = true;
                return Ok(parsed);
            }
            option if option.starts_with('-') => {
                return Err(format!("unknown list option {option:?}"));
            }
            other => return Err(format!("unexpected list argument {other:?}")),
        }
    }
    if parsed.output.is_some() && !parsed.gallery {
        return Err("--out requires --gallery".into());
    }
    Ok(parsed)
}

fn print_help() {
    println!(
        "thalos_capture list viewpoints [--gallery] [--json] [--out DIR]\n\n\
         Without --gallery, lists catalog metadata and cached-capture state.\n\
         --gallery performs no rendering: it builds a contact sheet, one 320x180\n\
         thumbnail per viewpoint, and index.json from artifacts/visual/latest."
    );
}

fn print_text_listing(index: &GalleryIndex) {
    for entry in &index.entries {
        let subject = entry
            .body
            .as_deref()
            .or(entry.driver.as_deref())
            .unwrap_or("-");
        println!(
            "{:<22} {:<6} {:<12} {}",
            entry.id, entry.kind, entry.cache_state, subject
        );
    }
    println!(
        "{} viewpoints · {} cached · {} missing · {} unreadable\nuse `thalos_capture list viewpoints --gallery` for the visual index",
        index.counts.total,
        index.counts.current + index.counts.stale + index.counts.cached + index.counts.unattributed,
        index.counts.missing,
        index.counts.unreadable,
    );
}

fn build_gallery(
    catalog: &ViewpointCatalog,
    catalog_path: &Path,
    latest: &Path,
    output: &Path,
    current_source: Option<&CaptureSourceSnapshot>,
    workspace: &Path,
) -> Result<GalleryIndex, String> {
    let thumbnails = output.join("thumbnails");
    fs::create_dir_all(&thumbnails)
        .map_err(|error| format!("create {}: {error}", thumbnails.display()))?;
    let font = FontRef::try_from_slice(FONT_BYTES).map_err(|error| error.to_string())?;
    let mut index = inspect_catalog(
        catalog,
        catalog_path,
        latest,
        Some(&thumbnails),
        current_source,
        workspace,
    );
    let mut rendered = Vec::with_capacity(index.entries.len());

    for entry in &mut index.entries {
        let source = workspace.join(Path::new(&entry.source_image));
        let (thumbnail, issue) = match image::open(&source) {
            Ok(image) => (fit_thumbnail(&image), None),
            Err(_) if !source.exists() => (placeholder_thumbnail(&font, "NO CACHED CAPTURE"), None),
            Err(error) => (
                placeholder_thumbnail(&font, "UNREADABLE CAPTURE"),
                Some(format!("decode {}: {error}", source.display())),
            ),
        };
        if let Some(issue) = issue {
            entry.cache_state = "unreadable";
            entry.issue = Some(issue);
        }
        let thumbnail_path = workspace.join(Path::new(&entry.thumbnail));
        if let Some(parent) = thumbnail_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        thumbnail
            .save(&thumbnail_path)
            .map_err(|error| format!("write {}: {error}", thumbnail_path.display()))?;
        rendered.push(thumbnail);
    }

    index.counts = counts(&index.entries);
    let sheet = make_contact_sheet(&index.entries, &rendered, &font);
    let contact_path = output.join("contact_sheet.png");
    sheet
        .save(&contact_path)
        .map_err(|error| format!("write {}: {error}", contact_path.display()))?;
    index.contact_sheet = Some(portable_path(&contact_path, workspace));
    let index_path = output.join("index.json");
    let bytes = serde_json::to_vec_pretty(&index).map_err(|error| error.to_string())?;
    fs::write(&index_path, bytes)
        .map_err(|error| format!("write {}: {error}", index_path.display()))?;
    Ok(index)
}

fn inspect_catalog(
    catalog: &ViewpointCatalog,
    catalog_path: &Path,
    latest: &Path,
    thumbnail_dir: Option<&Path>,
    current_source: Option<&CaptureSourceSnapshot>,
    workspace: &Path,
) -> GalleryIndex {
    let mut entries = catalog
        .viewpoints
        .iter()
        .map(|viewpoint| {
            inspect_entry(
                &viewpoint.id,
                &viewpoint.name,
                "saved",
                &viewpoint.description,
                Some(viewpoint.frame.label()),
                None,
                latest.join(format!("{}.png", file_slug(&viewpoint.id))),
                thumbnail_dir,
                current_source,
                workspace,
            )
        })
        .chain(catalog.scripted_viewpoints.iter().map(|viewpoint| {
            inspect_entry(
                &viewpoint.id,
                &viewpoint.name,
                "agent",
                &viewpoint.description,
                None,
                Some(&viewpoint.driver),
                latest.join(format!("{}.png", file_slug(&viewpoint.driver))),
                thumbnail_dir,
                current_source,
                workspace,
            )
        }))
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.id.cmp(&right.id));

    GalleryIndex {
        schema: "thalos.viewpoint-gallery.v1",
        generated_unix_ms: timestamp_millis(),
        catalog: portable_path(catalog_path, workspace),
        latest_capture_dir: portable_path(latest, workspace),
        current_source_fingerprint: current_source.map(|source| source.fingerprint.clone()),
        contact_sheet: None,
        counts: counts(&entries),
        entries,
    }
}

#[allow(clippy::too_many_arguments)]
fn inspect_entry(
    id: &str,
    name: &str,
    kind: &'static str,
    description: &str,
    body: Option<&str>,
    driver: Option<&str>,
    source_image: PathBuf,
    thumbnail_dir: Option<&Path>,
    current_source: Option<&CaptureSourceSnapshot>,
    workspace: &Path,
) -> GalleryEntry {
    let receipt_path = receipt_path(&source_image);
    let receipt = fs::read(&receipt_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<ReceiptSummary>(&bytes).ok());
    let cache_state = if !source_image.exists() {
        "missing"
    } else if let Some(receipt) = &receipt {
        if receipt.source.fingerprint.is_empty() {
            "unattributed"
        } else if current_source
            .is_some_and(|current| current.fingerprint == receipt.source.fingerprint)
        {
            "current"
        } else if current_source.is_some() {
            "stale"
        } else {
            "cached"
        }
    } else {
        "unattributed"
    };
    let thumbnail = thumbnail_dir
        .map(|directory| directory.join(format!("{}.png", file_slug(id))))
        .unwrap_or_else(|| {
            workspace.join(format!(
                "artifacts/visual/catalog/viewpoints/thumbnails/{}.png",
                file_slug(id)
            ))
        });

    GalleryEntry {
        id: id.to_owned(),
        name: name.to_owned(),
        kind,
        description: description.to_owned(),
        body: body.map(str::to_owned),
        driver: driver.map(str::to_owned),
        cache_state,
        source_image: portable_path(&source_image, workspace),
        source_receipt: receipt_path
            .exists()
            .then(|| portable_path(&receipt_path, workspace)),
        thumbnail: portable_path(&thumbnail, workspace),
        completed_unix_ms: receipt.as_ref().and_then(|receipt| {
            (receipt.completed_unix_ms != 0).then_some(receipt.completed_unix_ms)
        }),
        capture_source_fingerprint: receipt.as_ref().and_then(|receipt| {
            (!receipt.source.fingerprint.is_empty()).then(|| receipt.source.fingerprint.clone())
        }),
        capture_git_revision: receipt.as_ref().and_then(|receipt| {
            (!receipt.source.git_revision.is_empty()).then(|| receipt.source.git_revision.clone())
        }),
        workspace_matched_at_capture: receipt.as_ref().map(|receipt| receipt.workspace_matches),
        issue: None,
    }
}

fn counts(entries: &[GalleryEntry]) -> GalleryCounts {
    GalleryCounts {
        total: entries.len(),
        current: entries
            .iter()
            .filter(|entry| entry.cache_state == "current")
            .count(),
        stale: entries
            .iter()
            .filter(|entry| entry.cache_state == "stale")
            .count(),
        cached: entries
            .iter()
            .filter(|entry| entry.cache_state == "cached")
            .count(),
        unattributed: entries
            .iter()
            .filter(|entry| entry.cache_state == "unattributed")
            .count(),
        missing: entries
            .iter()
            .filter(|entry| entry.cache_state == "missing")
            .count(),
        unreadable: entries
            .iter()
            .filter(|entry| entry.cache_state == "unreadable")
            .count(),
    }
}

fn fit_thumbnail(source: &DynamicImage) -> RgbaImage {
    let source = source.to_rgba8();
    let scale = (THUMBNAIL_WIDTH as f64 / source.width().max(1) as f64)
        .min(THUMBNAIL_HEIGHT as f64 / source.height().max(1) as f64);
    let width = (source.width() as f64 * scale).round().max(1.0) as u32;
    let height = (source.height() as f64 * scale).round().max(1.0) as u32;
    let resized = resize(&source, width, height, FilterType::Lanczos3);
    let mut thumbnail =
        RgbaImage::from_pixel(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT, Rgba([9, 12, 18, 255]));
    overlay(
        &mut thumbnail,
        &resized,
        i64::from((THUMBNAIL_WIDTH - width) / 2),
        i64::from((THUMBNAIL_HEIGHT - height) / 2),
    );
    thumbnail
}

fn placeholder_thumbnail(font: &FontRef<'_>, label: &str) -> RgbaImage {
    let mut thumbnail =
        RgbaImage::from_pixel(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT, Rgba([20, 25, 34, 255]));
    for offset in 0..4 {
        let color = Rgba([43, 51, 66, 255]);
        draw_diagonal(
            &mut thumbnail,
            offset,
            THUMBNAIL_HEIGHT.saturating_sub(1 + offset),
            color,
        );
        draw_diagonal(&mut thumbnail, offset, offset, color);
    }
    draw_text(
        &mut thumbnail,
        font,
        58.0,
        78.0,
        17.0,
        label,
        Rgba([159, 172, 192, 255]),
    );
    thumbnail
}

fn draw_diagonal(image: &mut RgbaImage, start_x: u32, start_y: u32, color: Rgba<u8>) {
    let descending = start_y > THUMBNAIL_HEIGHT / 2;
    for x in start_x..image.width() {
        let travel = x - start_x;
        let y = if descending {
            start_y.saturating_sub(travel * THUMBNAIL_HEIGHT / THUMBNAIL_WIDTH)
        } else {
            (start_y + travel * THUMBNAIL_HEIGHT / THUMBNAIL_WIDTH).min(THUMBNAIL_HEIGHT - 1)
        };
        image.put_pixel(x, y, color);
    }
}

fn make_contact_sheet(
    entries: &[GalleryEntry],
    thumbnails: &[RgbaImage],
    font: &FontRef<'_>,
) -> RgbaImage {
    let columns = (entries.len() as u32).clamp(1, MAX_COLUMNS);
    let rows = (entries.len() as u32).div_ceil(columns);
    let width = SHEET_MARGIN * 2 + columns * CARD_WIDTH + columns.saturating_sub(1) * CARD_GAP;
    let height = SHEET_MARGIN * 2
        + SHEET_HEADER_HEIGHT
        + rows * CARD_HEIGHT
        + rows.saturating_sub(1) * CARD_GAP;
    let mut sheet = RgbaImage::from_pixel(width, height, Rgba([10, 13, 19, 255]));
    draw_text(
        &mut sheet,
        font,
        SHEET_MARGIN as f32,
        SHEET_MARGIN as f32,
        30.0,
        "Thalos viewpoint catalog",
        Rgba([237, 242, 250, 255]),
    );
    let available = entries
        .iter()
        .filter(|entry| !matches!(entry.cache_state, "missing" | "unreadable"))
        .count();
    draw_text(
        &mut sheet,
        font,
        SHEET_MARGIN as f32,
        (SHEET_MARGIN + 39) as f32,
        15.0,
        &format!(
            "{} viewpoints · {available} cached · {} missing · composition hints, not evidence",
            entries.len(),
            entries.len().saturating_sub(available)
        ),
        Rgba([145, 159, 180, 255]),
    );

    for (index, (entry, thumbnail)) in entries.iter().zip(thumbnails).enumerate() {
        let column = index as u32 % columns;
        let row = index as u32 / columns;
        let x = SHEET_MARGIN + column * (CARD_WIDTH + CARD_GAP);
        let y = SHEET_MARGIN + SHEET_HEADER_HEIGHT + row * (CARD_HEIGHT + CARD_GAP);
        fill_rect(
            &mut sheet,
            x,
            y,
            CARD_WIDTH,
            CARD_HEIGHT,
            Rgba([24, 29, 39, 255]),
        );
        draw_text(
            &mut sheet,
            font,
            (x + 16) as f32,
            (y + 12) as f32,
            18.0,
            &ellipsize(&format!("{} · {}", entry.id, entry.name), 38),
            Rgba([225, 232, 243, 255]),
        );
        overlay(&mut sheet, thumbnail, i64::from(x + 16), i64::from(y + 47));
        let subject = entry
            .body
            .as_deref()
            .or(entry.driver.as_deref())
            .unwrap_or("-");
        let status_color = match entry.cache_state {
            "current" => Rgba([111, 219, 163, 255]),
            "missing" | "unreadable" => Rgba([231, 137, 132, 255]),
            "stale" => Rgba([236, 188, 104, 255]),
            _ => Rgba([158, 174, 197, 255]),
        };
        draw_text(
            &mut sheet,
            font,
            (x + 16) as f32,
            (y + 235) as f32,
            14.0,
            &ellipsize(
                &format!("{} · {} · {}", entry.kind, subject, entry.cache_state),
                46,
            ),
            status_color,
        );
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

fn ellipsize(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let mut result = text
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    result.push('…');
    result
}

fn receipt_path(image: &Path) -> PathBuf {
    let stem = image
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("capture");
    image.with_file_name(format!("{stem}.capture.json"))
}

fn file_slug(id: &str) -> String {
    id.replace('-', "_")
}

fn portable_path(path: &Path, workspace: &Path) -> String {
    path.strip_prefix(workspace)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_capture_protocol::{
        CameraOptics, ScriptedViewpoint, Viewpoint, ViewpointFrame, ViewpointSpawn,
    };
    use uuid::Uuid;

    fn sample_catalog() -> ViewpointCatalog {
        ViewpointCatalog {
            viewpoints: vec![Viewpoint {
                id: "saved-view".into(),
                name: "Saved view".into(),
                description: "A saved camera".into(),
                saved_unix_ms: 1,
                frame: ViewpointFrame::AuthoredBodyFixed {
                    body: "Thalos".into(),
                    spawn: ViewpointSpawn::Runway,
                    boots_hub: false,
                    sim_time_s: 59_100.0,
                },
                camera_position_m: [1.0, 2.0, 3.0],
                camera_rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                optics: CameraOptics::from_vertical_fov(1.0, [16, 9]).unwrap(),
            }],
            scripted_viewpoints: vec![ScriptedViewpoint {
                id: "agent-view".into(),
                name: "Agent view".into(),
                description: "A scripted camera".into(),
                driver: "spaceport-aerial".into(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn list_parser_keeps_gallery_generation_explicit() {
        assert_eq!(
            parse_list_args(
                ["--gallery", "--json", "--out", "gallery"]
                    .into_iter()
                    .map(str::to_owned)
            )
            .unwrap(),
            ListArgs {
                gallery: true,
                json: true,
                output: Some(PathBuf::from("gallery")),
                help: false,
            }
        );
        assert!(parse_list_args(["--out", "gallery"].into_iter().map(str::to_owned)).is_err());
    }

    #[test]
    fn gallery_uses_cached_images_and_keeps_missing_entries_visible() {
        let workspace = super::super::workspace_root().join(format!(
            "target/viewpoint-gallery-test-{}",
            Uuid::new_v4().simple()
        ));
        let latest = workspace.join("artifacts/visual/latest");
        let output = workspace.join("artifacts/visual/catalog/viewpoints");
        fs::create_dir_all(&latest).unwrap();
        RgbaImage::from_pixel(640, 360, Rgba([20, 80, 140, 255]))
            .save(latest.join("saved_view.png"))
            .unwrap();
        let source = CaptureSourceSnapshot {
            fingerprint: "current".into(),
            git_revision: "abc123".into(),
            ..Default::default()
        };
        let receipt = serde_json::json!({
            "completed_unix_ms": 42,
            "workspace_matches": true,
            "source": source,
        });
        fs::write(
            latest.join("saved_view.capture.json"),
            serde_json::to_vec(&receipt).unwrap(),
        )
        .unwrap();

        let index = build_gallery(
            &sample_catalog(),
            &workspace.join("assets/viewpoints.json"),
            &latest,
            &output,
            Some(&CaptureSourceSnapshot {
                fingerprint: "current".into(),
                ..Default::default()
            }),
            &workspace,
        )
        .unwrap();
        assert_eq!(index.counts.total, 2);
        assert_eq!(index.counts.current, 1);
        assert_eq!(index.counts.missing, 1);
        assert!(output.join("contact_sheet.png").exists());
        assert!(output.join("index.json").exists());
        assert!(output.join("thumbnails/saved_view.png").exists());
        assert!(output.join("thumbnails/agent_view.png").exists());
        let _ = fs::remove_dir_all(&workspace);
    }

    #[test]
    fn thumbnail_fit_preserves_composition_with_letterboxing() {
        let source =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(400, 400, Rgba([10, 20, 30, 255])));
        let thumbnail = fit_thumbnail(&source);
        assert_eq!(thumbnail.dimensions(), (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT));
        assert_eq!(thumbnail.get_pixel(160, 90), &Rgba([10, 20, 30, 255]));
        assert_eq!(thumbnail.get_pixel(0, 90), &Rgba([9, 12, 18, 255]));
    }

    #[test]
    fn long_labels_are_bounded_for_the_contact_sheet() {
        assert_eq!(ellipsize("short", 8), "short");
        assert_eq!(ellipsize("abcdefghij", 6), "abcde…");
    }
}
