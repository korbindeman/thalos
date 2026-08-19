//! Offline perf report: renders the `thalos::diagnostic::perf` lane of
//! `artifacts/diagnostics/runtime.jsonl` (plus rotated `*.rot*.jsonl`
//! siblings) into a self-contained HTML report and a machine-readable
//! `summary.json` per session.
//!
//! Usage (via `just perf-report [session]`):
//!
//! ```text
//! thalos_perfreport            # newest session with perf data
//! thalos_perfreport latest     # same
//! thalos_perfreport <session>  # a specific `<pid>-<unix_ms>` session id
//! thalos_perfreport --list     # enumerate sessions found in the stream
//! ```
//!
//! Output: `artifacts/diagnostics/reports/<session>/report.html` + `summary.json`.
//! Agents should read `summary.json` (or the JSONL itself) — the HTML is for
//! humans.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

const DIAGNOSTICS_DIR: &str = "artifacts/diagnostics";
const SCHEMA: &str = "thalos.runtime_diagnostic.v1";

#[derive(Default)]
struct Session {
    pid: u64,
    first_ms: u128,
    last_ms: u128,
    /// (ts, fields) for perf-lane events, in stream order.
    gauges: Vec<(u128, Value)>,
    spikes: Vec<(u128, Value)>,
    blocks: Vec<(u128, Value)>,
    slab_gauges: Vec<(u128, Value)>,
    headless_benchmarks: Vec<(u128, Value)>,
    headless_render_passes: Vec<(u128, Value)>,
    total_events: u64,
}

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| "latest".into());

    let mut sessions: BTreeMap<String, Session> = BTreeMap::new();
    for path in stream_files() {
        ingest(&path, &mut sessions);
    }
    if sessions.is_empty() {
        eprintln!("no `{SCHEMA}` events found under {DIAGNOSTICS_DIR}/ — run the game first");
        std::process::exit(1);
    }

    if arg == "--list" {
        println!(
            "{:<24} {:>9} {:>8} {:>7} {:>7}",
            "SESSION", "DURATION", "GAUGES", "SPIKES", "BLOCKS"
        );
        for (id, s) in sessions.iter().rev() {
            println!(
                "{:<24} {:>8.1}m {:>8} {:>7} {:>7}",
                id,
                (s.last_ms.saturating_sub(s.first_ms)) as f64 / 60_000.0,
                s.gauges.len(),
                s.spikes.len(),
                s.blocks.len(),
            );
        }
        return;
    }

    if arg == "--headless-matrix" {
        write_headless_matrix(&sessions);
        return;
    }
    if arg == "--headless-shadow-cascades" {
        write_headless_shadow_cascades(&sessions);
        return;
    }
    if arg == "--headless-terrain-material" {
        write_headless_terrain_material(&sessions);
        return;
    }
    if arg == "--headless-terrain-prepass" {
        write_headless_terrain_prepass(&sessions);
        return;
    }
    if arg == "--headless-terrain-index" {
        write_headless_terrain_index(&sessions);
        return;
    }
    if arg == "--headless-terrain-culling" {
        write_headless_terrain_culling(&sessions);
        return;
    }

    let (id, session) = if arg == "latest" {
        // Newest session that actually carries perf data; fall back to newest.
        sessions
            .iter()
            .filter(|(_, s)| !s.gauges.is_empty())
            .max_by_key(|(_, s)| s.last_ms)
            .or_else(|| sessions.iter().max_by_key(|(_, s)| s.last_ms))
            .map(|(id, s)| (id.clone(), s))
            .expect("non-empty checked above")
    } else {
        match sessions.get(&arg) {
            Some(s) => (arg.clone(), s),
            None => {
                eprintln!("session `{arg}` not found; `--list` shows what exists");
                std::process::exit(1);
            }
        }
    };

    let out_dir = Path::new(DIAGNOSTICS_DIR).join("reports").join(&id);
    fs::create_dir_all(&out_dir).expect("create report dir");

    let summary = build_summary(&id, session);
    let summary_path = out_dir.join("summary.json");
    fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("write summary.json");

    let html_path = out_dir.join("report.html");
    fs::write(&html_path, render_html(&id, session, &summary)).expect("write report.html");

    println!("{}", summary_path.display());
    println!("{}", html_path.display());
    if session.gauges.is_empty() {
        eprintln!("note: session has no perf gauges (pre-perf-lane build?); report is skeletal");
    }
}

/// runtime.jsonl plus any rotated runtime.rot<ms>.jsonl siblings, oldest first
/// so ingestion order stays chronological.
fn stream_files() -> Vec<PathBuf> {
    let dir = Path::new(DIAGNOSTICS_DIR);
    let mut rotated: Vec<PathBuf> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("runtime.rot") && n.ends_with(".jsonl"))
        })
        .collect();
    rotated.sort();
    let active = dir.join("runtime.jsonl");
    if active.is_file() {
        rotated.push(active);
    }
    rotated
}

fn ingest(path: &Path, sessions: &mut BTreeMap<String, Session>) {
    let Ok(file) = fs::File::open(path) else {
        return;
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let Ok(v) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if v["schema"] != SCHEMA {
            continue;
        }
        let Some(session_id) = v["session"].as_str() else {
            continue;
        };
        let ts = v["ts_unix_ms"].as_u64().unwrap_or(0) as u128;
        let s = sessions.entry(session_id.to_string()).or_default();
        s.pid = v["pid"].as_u64().unwrap_or(0);
        if s.first_ms == 0 || ts < s.first_ms {
            s.first_ms = ts;
        }
        s.last_ms = s.last_ms.max(ts);
        s.total_events += 1;

        let target = v["target"].as_str().unwrap_or_default();
        let event = v["fields"]["event"].as_str().unwrap_or_default();
        match (target, event) {
            ("thalos::diagnostic::perf", "frame_gauge") => {
                s.gauges.push((ts, v["fields"].clone()));
            }
            ("thalos::diagnostic::perf", "spike") => {
                s.spikes.push((ts, v["fields"].clone()));
            }
            ("thalos::diagnostic::perf", "frame_block") => {
                s.blocks.push((ts, v["fields"].clone()));
            }
            ("thalos::diagnostic::perf", "headless_benchmark_end") => {
                s.headless_benchmarks.push((ts, v["fields"].clone()));
            }
            ("thalos::diagnostic::perf", "headless_render_pass") => {
                s.headless_render_passes.push((ts, v["fields"].clone()));
            }
            ("thalos::diagnostic::gpu_mem", "mesh_slab_gauge") => {
                s.slab_gauges.push((ts, v["fields"].clone()));
            }
            _ => {}
        }
    }
}

fn f(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap_or(0.0)
}

fn gpu_timing_available(gauge: &Value) -> bool {
    gauge["gpu_timing_available"]
        .as_bool()
        // Older sessions did not carry the presence bit. A positive timing is
        // still unambiguous; zero remains an unknown rather than a free GPU.
        .unwrap_or_else(|| f(gauge, "gpu_ms_mean") > 0.0)
}

/// Parse a comma-joined ms string field ("12.10,11.95,…") from spike /
/// frame_block events.
fn parse_ms_list(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_str()
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.parse::<f64>().ok())
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn build_summary(id: &str, s: &Session) -> Value {
    let duration_s = s.last_ms.saturating_sub(s.first_ms) as f64 / 1000.0;

    let mean_of = |key: &str| -> f64 {
        if s.gauges.is_empty() {
            return 0.0;
        }
        s.gauges.iter().map(|(_, g)| f(g, key)).sum::<f64>() / s.gauges.len() as f64
    };
    let max_of = |key: &str| -> f64 { s.gauges.iter().map(|(_, g)| f(g, key)).fold(0.0, f64::max) };
    let last_of = |key: &str| -> f64 { s.gauges.last().map(|(_, g)| f(g, key)).unwrap_or(0.0) };
    let last = s.gauges.last().map(|(_, gauge)| gauge);
    let gpu_windows: Vec<f64> = s
        .gauges
        .iter()
        .filter(|(_, gauge)| gpu_timing_available(gauge))
        .map(|(_, gauge)| f(gauge, "gpu_ms_mean"))
        .collect();
    let gpu_ms_mean = (!gpu_windows.is_empty())
        .then(|| gpu_windows.iter().sum::<f64>() / gpu_windows.len() as f64);

    // Exact frame-level percentiles when full-rate blocks were recorded;
    // otherwise the honest window-level aggregates only.
    let full_rate = if s.blocks.is_empty() {
        Value::Null
    } else {
        let mut frames: Vec<f64> = s
            .blocks
            .iter()
            .flat_map(|(_, b)| parse_ms_list(b, "cpu_ms"))
            .collect();
        frames.sort_by(|a, b| a.total_cmp(b));
        json!({
            "frames": frames.len(),
            "cpu_ms_p50": percentile(&frames, 0.50),
            "cpu_ms_p95": percentile(&frames, 0.95),
            "cpu_ms_p99": percentile(&frames, 0.99),
            "cpu_ms_max": frames.last().copied().unwrap_or(0.0),
        })
    };

    json!({
        "session": id,
        "pid": s.pid,
        "start_unix_ms": s.first_ms as u64,
        "end_unix_ms": s.last_ms as u64,
        "duration_s": duration_s,
        "gauge_windows": s.gauges.len(),
        // Window-level aggregates (each gauge summarizes ~2 s of frames):
        // *_mean averages the windows; worst_* is the worst window seen.
        "fps_mean": mean_of("fps"),
        "cpu_ms_mean": mean_of("cpu_ms_mean"),
        "worst_window_p95_ms": max_of("cpu_ms_p95"),
        "worst_window_max_ms": max_of("cpu_ms_max"),
        "gpu_ms_mean": gpu_ms_mean,
        "gpu_timing_available": !gpu_windows.is_empty(),
        "configuration": {
            "foliage_enabled": last.and_then(|g| g["foliage_enabled"].as_bool()),
            "clouds_enabled": last.and_then(|g| g["clouds_enabled"].as_bool()),
            "grass_enabled": last.and_then(|g| g["grass_enabled"].as_bool()),
            "gpu_grass_enabled": last.and_then(|g| g["gpu_grass_enabled"].as_bool()),
            "msaa_samples": last.and_then(|g| g["msaa_samples"].as_u64()),
            "shadow_cascade_budget": last.and_then(|g| g["shadow_cascade_budget"].as_u64()),
            "shadow_quality": last.and_then(|g| g["shadow_quality"].as_str()),
            "shadow_map_size_px": last.and_then(|g| g["shadow_map_size_px"].as_u64()),
            "vsync_enabled": last.and_then(|g| g["vsync_enabled"].as_bool()),
            "has_primary_window": last.and_then(|g| g["has_primary_window"].as_bool()),
            "window_width_px": last.and_then(|g| g["window_width_px"].as_u64()),
            "window_height_px": last.and_then(|g| g["window_height_px"].as_u64()),
        },
        "headless_benchmark": s.headless_benchmarks.last().map(|(_, result)| result),
        "stage_ms_mean": {
            "physics": mean_of("physics_ms"),
            "sync": mean_of("sync_ms"),
            "camera": mean_of("camera_ms"),
        },
        "memory": {
            "tile_mib_last": last_of("tile_mib"),
            "tile_mib_max": max_of("tile_mib"),
            "slab_mib_last": last_of("slab_mib"),
            "slab_mib_max": max_of("slab_mib"),
            "entities_last": last_of("entities"),
            "entities_max": max_of("entities"),
            "main_meshes_last": last_of("main_meshes"),
            "main_images_last": last_of("main_images"),
        },
        "spikes": s.spikes.iter().map(|(ts, sp)| json!({
            "ts_unix_ms": *ts as u64,
            "offset_s": (ts.saturating_sub(s.first_ms)) as f64 / 1000.0,
            "spike_ms": f(sp, "spike_ms"),
            "median_ms": f(sp, "median_ms"),
        })).collect::<Vec<_>>(),
        "full_rate": full_rate,
    })
}

fn write_headless_matrix(sessions: &BTreeMap<String, Session>) {
    let mut cells = serde_json::Map::new();
    for (session_id, session) in sessions {
        for (ts, result) in &session.headless_benchmarks {
            let Some(variant) = result["variant"].as_str() else {
                continue;
            };
            let replace = cells
                .get(variant)
                .and_then(|cell| cell["end_unix_ms"].as_u64())
                .is_none_or(|current| *ts as u64 > current);
            if replace {
                cells.insert(
                    variant.to_string(),
                    json!({
                        "session": session_id,
                        "end_unix_ms": *ts as u64,
                        "result": result,
                        "render_passes": headless_render_passes(session, variant),
                    }),
                );
            }
        }
    }

    let mean = |variant: &str| {
        cells
            .get(variant)
            .and_then(|cell| cell["result"]["cpu_ms_mean"].as_f64())
    };
    let missing: Vec<&str> = ["baseline", "foliage-off", "shadows-off", "both-off"]
        .into_iter()
        .filter(|variant| !cells.contains_key(*variant))
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "headless performance matrix is incomplete; missing {}",
            missing.join(", ")
        );
        std::process::exit(2);
    }
    let effects = match (
        mean("baseline"),
        mean("foliage-off"),
        mean("shadows-off"),
        mean("both-off"),
    ) {
        (Some(baseline), Some(foliage_off), Some(shadows_off), Some(both_off)) => json!({
            "foliage_ms_with_shadows": baseline - foliage_off,
            "shadows_ms_with_foliage": baseline - shadows_off,
            "foliage_ms_without_shadows": shadows_off - both_off,
            "shadows_ms_without_foliage": foliage_off - both_off,
            "both_disabled_ms": baseline - both_off,
            "residual_frame_ms": both_off,
        }),
        _ => Value::Null,
    };
    let matrix = json!({
        "schema": "thalos.headless_perf_matrix.v1",
        "cells": cells,
        "effects": effects,
    });
    let path = Path::new(DIAGNOSTICS_DIR)
        .join("reports")
        .join("headless-matrix.json");
    fs::create_dir_all(path.parent().expect("matrix report parent"))
        .expect("create matrix report directory");
    fs::write(&path, serde_json::to_string_pretty(&matrix).unwrap())
        .expect("write headless matrix");
    println!("{}", path.display());
    println!("{}", serde_json::to_string_pretty(&matrix).unwrap());
}

fn write_headless_shadow_cascades(sessions: &BTreeMap<String, Session>) {
    let labels = [
        "cascades-0",
        "cascades-1",
        "cascades-2",
        "cascades-3",
        "cascades-4",
    ];
    let mut cells = serde_json::Map::new();
    for (session_id, session) in sessions {
        for (ts, result) in &session.headless_benchmarks {
            let Some(variant) = result["variant"].as_str() else {
                continue;
            };
            if !labels.contains(&variant) {
                continue;
            }
            let replace = cells
                .get(variant)
                .and_then(|cell| cell["end_unix_ms"].as_u64())
                .is_none_or(|current| *ts as u64 > current);
            if replace {
                cells.insert(
                    variant.to_string(),
                    json!({
                        "session": session_id,
                        "end_unix_ms": *ts as u64,
                        "result": result,
                        "render_passes": headless_render_passes(session, variant),
                    }),
                );
            }
        }
    }

    let missing: Vec<_> = labels
        .into_iter()
        .filter(|label| !cells.contains_key(*label))
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "headless shadow-cascade ladder is incomplete; missing {}",
            missing.join(", ")
        );
        std::process::exit(2);
    }
    let metric = |budget: usize, key: &str| {
        cells[&format!("cascades-{budget}")]["result"][key]
            .as_f64()
            .unwrap_or_else(|| panic!("validated shadow cell has {key}"))
    };
    let marginal_ms = (1..=4)
        .map(|budget| {
            (
                format!("cascade_{budget}_ms"),
                json!(metric(budget, "cpu_ms_mean") - metric(budget - 1, "cpu_ms_mean")),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let marginal_p50_ms = (1..=4)
        .map(|budget| {
            (
                format!("cascade_{budget}_ms"),
                json!(metric(budget, "cpu_ms_p50") - metric(budget - 1, "cpu_ms_p50")),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let report = json!({
        "schema": "thalos.headless_shadow_cascades.v1",
        "cells": cells,
        "effects": {
            "all_cascades_ms": metric(4, "cpu_ms_mean") - metric(0, "cpu_ms_mean"),
            "all_cascades_p50_ms": metric(4, "cpu_ms_p50") - metric(0, "cpu_ms_p50"),
            "medium_savings_vs_high_ms": metric(4, "cpu_ms_mean") - metric(3, "cpu_ms_mean"),
            "low_savings_vs_high_ms": metric(4, "cpu_ms_mean") - metric(2, "cpu_ms_mean"),
            "off_savings_vs_high_ms": metric(4, "cpu_ms_mean") - metric(0, "cpu_ms_mean"),
            "residual_frame_ms": metric(0, "cpu_ms_mean"),
            "residual_frame_p50_ms": metric(0, "cpu_ms_p50"),
            "marginal_ms": marginal_ms,
            "marginal_p50_ms": marginal_p50_ms,
        },
        "quality_tiers": {
            "off": { "shadow_cascade_budget": 0, "cell": "cascades-0" },
            "low": { "shadow_cascade_budget": 2, "cell": "cascades-2" },
            "medium": { "shadow_cascade_budget": 3, "cell": "cascades-3" },
            "high": { "shadow_cascade_budget": 4, "cell": "cascades-4" },
        },
    });
    let path = Path::new(DIAGNOSTICS_DIR)
        .join("reports")
        .join("headless-shadow-cascades.json");
    fs::create_dir_all(path.parent().expect("shadow report parent"))
        .expect("create shadow report directory");
    fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
        .expect("write headless shadow-cascade ladder");
    println!("{}", path.display());
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn write_headless_terrain_material(sessions: &BTreeMap<String, Session>) {
    let labels = [
        "terrain-lit-before",
        "terrain-fullbright",
        "terrain-base-color",
        "terrain-hidden",
        "terrain-lit-after",
    ];
    let Some((session_id, session)) = sessions
        .iter()
        .filter(|(_, session)| {
            labels.iter().all(|label| {
                session
                    .headless_benchmarks
                    .iter()
                    .any(|(_, result)| result["variant"].as_str() == Some(label))
            })
        })
        .max_by_key(|(_, session)| session.last_ms)
    else {
        eprintln!("headless terrain-material benchmark is incomplete");
        std::process::exit(2);
    };

    let mut cells = serde_json::Map::new();
    for label in labels {
        let (ts, result) = session
            .headless_benchmarks
            .iter()
            .rev()
            .find(|(_, result)| result["variant"].as_str() == Some(label))
            .expect("complete session checked above");
        cells.insert(
            label.to_string(),
            json!({
                "session": session_id,
                "end_unix_ms": *ts as u64,
                "result": result,
                "render_passes": headless_render_passes(session, label),
            }),
        );
    }

    let metric = |label: &str, key: &str| {
        cells[label]["result"][key]
            .as_f64()
            .unwrap_or_else(|| panic!("validated terrain-material cell has {key}"))
    };
    let effects_for = |key: &str| {
        let lit_before = metric("terrain-lit-before", key);
        let fullbright = metric("terrain-fullbright", key);
        let base_color = metric("terrain-base-color", key);
        let terrain_hidden = metric("terrain-hidden", key);
        let lit_after = metric("terrain-lit-after", key);
        let lit_midpoint = (lit_before + lit_after) * 0.5;
        json!({
            "lit_before_ms": lit_before,
            "lit_after_ms": lit_after,
            "lit_drift_ms": lit_after - lit_before,
            "fullbright_ms": fullbright,
            "base_color_ms": base_color,
            "terrain_hidden_ms": terrain_hidden,
            "terrain_base_render_ms": base_color - terrain_hidden,
            "procedural_layers_ms": fullbright - base_color,
            "pbr_and_post_ms": lit_midpoint - fullbright,
            "layers_plus_lighting_ms": lit_midpoint - base_color,
        })
    };
    let report = json!({
        "schema": "thalos.headless_terrain_material.v1",
        "session": session_id,
        "cells": cells,
        "effects": {
            "mean": effects_for("cpu_ms_mean"),
            "p50": effects_for("cpu_ms_p50"),
        },
    });
    let path = Path::new(DIAGNOSTICS_DIR)
        .join("reports")
        .join("headless-terrain-material.json");
    fs::create_dir_all(path.parent().expect("terrain report parent"))
        .expect("create terrain report directory");
    fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
        .expect("write headless terrain-material report");
    println!("{}", path.display());
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn write_headless_terrain_prepass(sessions: &BTreeMap<String, Session>) {
    let labels = [
        "terrain-base-prepass-before",
        "terrain-hidden-prepass",
        "terrain-base-no-prepass",
        "terrain-hidden-no-prepass",
        "terrain-base-prepass-after",
    ];
    let Some((session_id, session)) = sessions
        .iter()
        .filter(|(_, session)| {
            labels.iter().all(|label| {
                session
                    .headless_benchmarks
                    .iter()
                    .any(|(_, result)| result["variant"].as_str() == Some(label))
            })
        })
        .max_by_key(|(_, session)| session.last_ms)
    else {
        eprintln!("headless terrain-prepass benchmark is incomplete");
        std::process::exit(2);
    };

    let mut cells = serde_json::Map::new();
    for label in labels {
        let (ts, result) = session
            .headless_benchmarks
            .iter()
            .rev()
            .find(|(_, result)| result["variant"].as_str() == Some(label))
            .expect("complete session checked above");
        cells.insert(
            label.to_string(),
            json!({
                "session": session_id,
                "end_unix_ms": *ts as u64,
                "result": result,
                "render_passes": headless_render_passes(session, label),
            }),
        );
    }

    let identity_keys = [
        "entities",
        "main_meshes",
        "tile_resident",
        "offscreen_width_px",
        "offscreen_height_px",
        "foliage_enabled",
        "shadow_cascade_budget",
    ];
    let reference = &cells[labels[0]]["result"];
    for label in labels.iter().skip(1) {
        for key in identity_keys {
            if cells[*label]["result"][key] != reference[key] {
                eprintln!(
                    "headless terrain-prepass identity drift: {label}.{key}={:?}, expected {:?}",
                    cells[*label]["result"][key], reference[key]
                );
                std::process::exit(2);
            }
        }
    }

    let effects_for = |key: &str| terrain_prepass_effects(&cells, key);
    let report = json!({
        "schema": "thalos.headless_terrain_prepass.v1",
        "session": session_id,
        "identity": identity_keys
            .into_iter()
            .map(|key| (key.to_string(), reference[key].clone()))
            .collect::<serde_json::Map<String, Value>>(),
        "cells": cells,
        "effects": {
            "mean": effects_for("cpu_ms_mean"),
            "p50": effects_for("cpu_ms_p50"),
        },
    });
    let path = Path::new(DIAGNOSTICS_DIR)
        .join("reports")
        .join("headless-terrain-prepass.json");
    fs::create_dir_all(path.parent().expect("terrain-prepass report parent"))
        .expect("create terrain-prepass report directory");
    fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
        .expect("write headless terrain-prepass report");
    println!("{}", path.display());
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn terrain_prepass_effects(cells: &serde_json::Map<String, Value>, key: &str) -> Value {
    let metric = |label: &str| {
        cells[label]["result"][key]
            .as_f64()
            .unwrap_or_else(|| panic!("validated terrain-prepass cell has {key}"))
    };
    let base_before = metric("terrain-base-prepass-before");
    let hidden_prepass = metric("terrain-hidden-prepass");
    let base_no_prepass = metric("terrain-base-no-prepass");
    let hidden_no_prepass = metric("terrain-hidden-no-prepass");
    let base_after = metric("terrain-base-prepass-after");
    let base_prepass = (base_before + base_after) * 0.5;
    let terrain_with_prepass = base_prepass - hidden_prepass;
    let terrain_without_prepass = base_no_prepass - hidden_no_prepass;
    json!({
        "base_prepass_before_ms": base_before,
        "base_prepass_after_ms": base_after,
        "base_prepass_drift_ms": base_after - base_before,
        "hidden_prepass_ms": hidden_prepass,
        "base_no_prepass_ms": base_no_prepass,
        "hidden_no_prepass_ms": hidden_no_prepass,
        "terrain_with_prepass_ms": terrain_with_prepass,
        "terrain_without_prepass_ms": terrain_without_prepass,
        "terrain_prepass_net_ms": terrain_with_prepass - terrain_without_prepass,
        "whole_scene_prepass_net_ms": base_prepass - base_no_prepass,
        "nonterrain_prepass_net_ms": hidden_prepass - hidden_no_prepass,
    })
}

fn write_headless_terrain_index(sessions: &BTreeMap<String, Session>) {
    let labels = [
        "terrain-base-dense-before",
        "terrain-hidden-dense",
        "terrain-base-coarse",
        "terrain-hidden-coarse",
        "terrain-base-dense-after",
    ];
    let Some((session_id, session)) = sessions
        .iter()
        .filter(|(_, session)| {
            labels.iter().all(|label| {
                session
                    .headless_benchmarks
                    .iter()
                    .any(|(_, result)| result["variant"].as_str() == Some(label))
            })
        })
        .max_by_key(|(_, session)| session.last_ms)
    else {
        eprintln!("headless terrain-index benchmark is incomplete");
        std::process::exit(2);
    };

    let mut cells = serde_json::Map::new();
    for label in labels {
        let (ts, result) = session
            .headless_benchmarks
            .iter()
            .rev()
            .find(|(_, result)| result["variant"].as_str() == Some(label))
            .expect("complete session checked above");
        cells.insert(
            label.to_string(),
            json!({
                "session": session_id,
                "end_unix_ms": *ts as u64,
                "result": result,
                "render_passes": headless_render_passes(session, label),
            }),
        );
    }

    let identity_keys = [
        "entities",
        "main_meshes",
        "tile_resident",
        "offscreen_width_px",
        "offscreen_height_px",
        "foliage_enabled",
        "shadow_cascade_budget",
        "depth_prepass_enabled",
    ];
    let reference = &cells[labels[0]]["result"];
    for label in labels.iter().skip(1) {
        for key in identity_keys {
            if cells[*label]["result"][key] != reference[key] {
                eprintln!(
                    "headless terrain-index identity drift: {label}.{key}={:?}, expected {:?}",
                    cells[*label]["result"][key], reference[key]
                );
                std::process::exit(2);
            }
        }
    }
    for (label, expected_step) in [
        ("terrain-base-dense-before", 1),
        ("terrain-hidden-dense", 1),
        ("terrain-base-coarse", 4),
        ("terrain-hidden-coarse", 4),
        ("terrain-base-dense-after", 1),
    ] {
        if cells[label]["result"]["terrain_index_step"] != expected_step {
            eprintln!("headless terrain-index cell {label} has the wrong index step");
            std::process::exit(2);
        }
    }

    let effects_for = |key: &str| terrain_index_effects(&cells, key);
    let report = json!({
        "schema": "thalos.headless_terrain_index.v1",
        "session": session_id,
        "identity": identity_keys
            .into_iter()
            .map(|key| (key.to_string(), reference[key].clone()))
            .collect::<serde_json::Map<String, Value>>(),
        "cells": cells,
        "effects": {
            "mean": effects_for("cpu_ms_mean"),
            "p50": effects_for("cpu_ms_p50"),
        },
    });
    let path = Path::new(DIAGNOSTICS_DIR)
        .join("reports")
        .join("headless-terrain-index.json");
    fs::create_dir_all(path.parent().expect("terrain-index report parent"))
        .expect("create terrain-index report directory");
    fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
        .expect("write headless terrain-index report");
    println!("{}", path.display());
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn terrain_index_effects(cells: &serde_json::Map<String, Value>, key: &str) -> Value {
    let metric = |label: &str| {
        cells[label]["result"][key]
            .as_f64()
            .unwrap_or_else(|| panic!("validated terrain-index cell has {key}"))
    };
    let dense_before = metric("terrain-base-dense-before");
    let hidden_dense = metric("terrain-hidden-dense");
    let coarse = metric("terrain-base-coarse");
    let hidden_coarse = metric("terrain-hidden-coarse");
    let dense_after = metric("terrain-base-dense-after");
    let dense = (dense_before + dense_after) * 0.5;
    let dense_terrain = dense - hidden_dense;
    let coarse_terrain = coarse - hidden_coarse;
    let net = dense_terrain - coarse_terrain;
    json!({
        "base_dense_before_ms": dense_before,
        "base_dense_after_ms": dense_after,
        "base_dense_drift_ms": dense_after - dense_before,
        "hidden_dense_ms": hidden_dense,
        "base_coarse_ms": coarse,
        "hidden_coarse_ms": hidden_coarse,
        "hidden_control_drift_ms": hidden_coarse - hidden_dense,
        "dense_terrain_ms": dense_terrain,
        "coarse_terrain_ms": coarse_terrain,
        "terrain_index_density_net_ms": net,
        "terrain_index_density_reduction_frac": net / dense_terrain.max(f64::EPSILON),
        "whole_scene_index_density_net_ms": dense - coarse,
    })
}

fn write_headless_terrain_culling(sessions: &BTreeMap<String, Session>) {
    let labels = [
        "terrain-base-full-bounds-before",
        "terrain-hidden-full-bounds",
        "terrain-base-tight-bounds",
        "terrain-hidden-tight-bounds",
        "terrain-base-full-bounds-after",
    ];
    let Some((session_id, session)) = sessions
        .iter()
        .filter(|(_, session)| {
            labels.iter().all(|label| {
                session
                    .headless_benchmarks
                    .iter()
                    .any(|(_, result)| result["variant"].as_str() == Some(label))
            })
        })
        .max_by_key(|(_, session)| session.last_ms)
    else {
        eprintln!("headless terrain-culling benchmark is incomplete");
        std::process::exit(2);
    };

    let mut cells = serde_json::Map::new();
    for label in labels {
        let (ts, result) = session
            .headless_benchmarks
            .iter()
            .rev()
            .find(|(_, result)| result["variant"].as_str() == Some(label))
            .expect("complete session checked above");
        cells.insert(
            label.to_string(),
            json!({
                "session": session_id,
                "end_unix_ms": *ts as u64,
                "result": result,
                "render_passes": headless_render_passes(session, label),
            }),
        );
    }

    let identity_keys = [
        "entities",
        "main_meshes",
        "tile_resident",
        "offscreen_width_px",
        "offscreen_height_px",
        "foliage_enabled",
        "shadow_cascade_budget",
        "depth_prepass_enabled",
        "terrain_index_step",
    ];
    let reference = &cells[labels[0]]["result"];
    for label in labels.iter().skip(1) {
        for key in identity_keys {
            if cells[*label]["result"][key] != reference[key] {
                eprintln!(
                    "headless terrain-culling identity drift: {label}.{key}={:?}, expected {:?}",
                    cells[*label]["result"][key], reference[key]
                );
                std::process::exit(2);
            }
        }
    }
    for (label, expected_tight) in [
        ("terrain-base-full-bounds-before", false),
        ("terrain-hidden-full-bounds", false),
        ("terrain-base-tight-bounds", true),
        ("terrain-hidden-tight-bounds", true),
        ("terrain-base-full-bounds-after", false),
    ] {
        if cells[label]["result"]["tight_tile_bounds"] != expected_tight {
            eprintln!("headless terrain-culling cell {label} has the wrong bounds state");
            std::process::exit(2);
        }
    }

    let effects_for = |key: &str| terrain_culling_effects(&cells, key);
    let report = json!({
        "schema": "thalos.headless_terrain_culling.v1",
        "session": session_id,
        "identity": identity_keys
            .into_iter()
            .map(|key| (key.to_string(), reference[key].clone()))
            .collect::<serde_json::Map<String, Value>>(),
        "cells": cells,
        "effects": {
            "mean": effects_for("cpu_ms_mean"),
            "p50": effects_for("cpu_ms_p50"),
        },
    });
    let path = Path::new(DIAGNOSTICS_DIR)
        .join("reports")
        .join("headless-terrain-culling.json");
    fs::create_dir_all(path.parent().expect("terrain-culling report parent"))
        .expect("create terrain-culling report directory");
    fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
        .expect("write headless terrain-culling report");
    println!("{}", path.display());
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn terrain_culling_effects(cells: &serde_json::Map<String, Value>, key: &str) -> Value {
    let metric = |label: &str| {
        cells[label]["result"][key]
            .as_f64()
            .unwrap_or_else(|| panic!("validated terrain-culling cell has {key}"))
    };
    let full_before = metric("terrain-base-full-bounds-before");
    let hidden_full = metric("terrain-hidden-full-bounds");
    let tight = metric("terrain-base-tight-bounds");
    let hidden_tight = metric("terrain-hidden-tight-bounds");
    let full_after = metric("terrain-base-full-bounds-after");
    let full = (full_before + full_after) * 0.5;
    let full_terrain = full - hidden_full;
    let tight_terrain = tight - hidden_tight;
    let net = full_terrain - tight_terrain;
    json!({
        "base_full_before_ms": full_before,
        "base_full_after_ms": full_after,
        "base_full_drift_ms": full_after - full_before,
        "hidden_full_ms": hidden_full,
        "base_tight_ms": tight,
        "hidden_tight_ms": hidden_tight,
        "hidden_control_drift_ms": hidden_tight - hidden_full,
        "full_bounds_terrain_ms": full_terrain,
        "tight_bounds_terrain_ms": tight_terrain,
        "terrain_tight_bounds_net_ms": net,
        "terrain_tight_bounds_reduction_frac": net / full_terrain.max(f64::EPSILON),
        "whole_scene_tight_bounds_net_ms": full - tight,
    })
}

fn headless_render_passes(session: &Session, variant: &str) -> Value {
    let mut passes = serde_json::Map::new();
    for (_, fields) in &session.headless_render_passes {
        if fields["variant"].as_str() != Some(variant) {
            continue;
        }
        let Some(pass) = fields["pass"].as_str() else {
            continue;
        };
        let gpu_timing_available = fields["gpu_timing_available"].as_bool().unwrap_or(false);
        passes.insert(
            pass.to_string(),
            json!({
                "cpu_ms": fields["cpu_ms"].as_f64(),
                "gpu_ms": if gpu_timing_available {
                    fields["gpu_ms"].as_f64().map(Value::from).unwrap_or(Value::Null)
                } else {
                    Value::Null
                },
                "gpu_timing_available": gpu_timing_available,
            }),
        );
    }
    Value::Object(passes)
}

fn render_html(id: &str, s: &Session, summary: &Value) -> String {
    // Chart series, seconds since session start on x.
    let t0 = s.first_ms;
    let rel = |ts: u128| (ts.saturating_sub(t0)) as f64 / 1000.0;

    let gauge_series: Vec<Value> = s
        .gauges
        .iter()
        .map(|(ts, g)| {
            json!({
                "t": rel(*ts),
                "mean": f(g, "cpu_ms_mean"), "p95": f(g, "cpu_ms_p95"),
                "max": f(g, "cpu_ms_max"), "gpu": f(g, "gpu_ms_mean"),
                "fps": f(g, "fps"),
                "tile": f(g, "tile_mib"), "slab": f(g, "slab_mib"),
                "entities": f(g, "entities"), "meshes": f(g, "main_meshes"),
                "physics": f(g, "physics_ms"), "sync": f(g, "sync_ms"),
                "camera": f(g, "camera_ms"),
            })
        })
        .collect();
    let spike_series: Vec<Value> = s
        .spikes
        .iter()
        .map(|(ts, sp)| {
            json!({
                "t": rel(*ts),
                "spike_ms": f(sp, "spike_ms"),
                "median_ms": f(sp, "median_ms"),
                "post_frames": f(sp, "post_frames"),
                "cpu": parse_ms_list(sp, "cpu_ms"),
                "gpu": parse_ms_list(sp, "gpu_ms"),
            })
        })
        .collect();
    let block_series: Vec<Value> = s
        .blocks
        .iter()
        .map(|(ts, b)| json!({ "t": rel(*ts), "cpu": parse_ms_list(b, "cpu_ms") }))
        .collect();

    let data = json!({
        "session": id,
        "summary": summary,
        "gauges": gauge_series,
        "spikes": spike_series,
        "blocks": block_series,
    });

    let template = include_str!("report_template.html");
    template
        .replace("__TITLE__", &format!("Thalos perf — {id}"))
        .replace("\"__DATA__\"", &serde_json::to_string(&data).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_gpu_timing_is_a_gap_and_configuration_survives() {
        let mut session = Session::default();
        session.gauges.push((
            100,
            json!({
                "fps": 10.0,
                "cpu_ms_mean": 100.0,
                "gpu_ms_mean": 0.0,
                "gpu_timing_available": false,
                "foliage_enabled": false,
                "clouds_enabled": false,
                "grass_enabled": false,
                "gpu_grass_enabled": true,
                "msaa_samples": 1,
                "shadow_cascade_budget": 4,
                "shadow_quality": "High",
                "shadow_map_size_px": 4096,
                "vsync_enabled": false,
                "has_primary_window": true,
                "window_width_px": 1600,
                "window_height_px": 900,
            }),
        ));

        let summary = build_summary("test", &session);
        assert!(summary["gpu_ms_mean"].is_null());
        assert_eq!(summary["gpu_timing_available"], false);
        assert_eq!(summary["configuration"]["foliage_enabled"], false);
        assert_eq!(summary["configuration"]["shadow_cascade_budget"], 4);
        assert_eq!(summary["configuration"]["shadow_quality"], "High");
        assert_eq!(summary["configuration"]["shadow_map_size_px"], 4096);
        assert_eq!(summary["configuration"]["window_width_px"], 1600);
    }

    #[test]
    fn old_positive_gpu_samples_remain_readable() {
        let mut session = Session::default();
        session.gauges.push((100, json!({"gpu_ms_mean": 12.5})));

        let summary = build_summary("test", &session);
        assert_eq!(summary["gpu_ms_mean"], 12.5);
        assert_eq!(summary["gpu_timing_available"], true);
    }

    #[test]
    fn headless_matrix_uses_latest_variant_result() {
        let mut sessions = BTreeMap::new();
        let mut old = Session::default();
        old.headless_benchmarks.push((
            100,
            json!({
                "variant": "baseline",
                "cpu_ms_mean": 20.0,
            }),
        ));
        sessions.insert("old".to_string(), old);
        let mut new = Session::default();
        new.headless_benchmarks.push((
            200,
            json!({
                "variant": "baseline",
                "cpu_ms_mean": 10.0,
            }),
        ));
        sessions.insert("new".to_string(), new);

        let newest = sessions
            .values()
            .flat_map(|session| session.headless_benchmarks.iter())
            .max_by_key(|(ts, _)| *ts)
            .unwrap();
        assert_eq!(newest.1["cpu_ms_mean"], 10.0);
    }

    #[test]
    fn terrain_prepass_effects_subtract_the_hidden_control() {
        let mut cells = serde_json::Map::new();
        for (label, value) in [
            ("terrain-base-prepass-before", 40.0),
            ("terrain-hidden-prepass", 18.0),
            ("terrain-base-no-prepass", 31.0),
            ("terrain-hidden-no-prepass", 15.0),
            ("terrain-base-prepass-after", 42.0),
        ] {
            cells.insert(label.to_string(), json!({"result": {"cpu_ms_p50": value}}));
        }

        let effects = terrain_prepass_effects(&cells, "cpu_ms_p50");
        assert_eq!(effects["base_prepass_drift_ms"], 2.0);
        assert_eq!(effects["terrain_with_prepass_ms"], 23.0);
        assert_eq!(effects["terrain_without_prepass_ms"], 16.0);
        assert_eq!(effects["terrain_prepass_net_ms"], 7.0);
        assert_eq!(effects["nonterrain_prepass_net_ms"], 3.0);
    }

    #[test]
    fn terrain_index_effects_subtract_the_hidden_control() {
        let mut cells = serde_json::Map::new();
        for (label, value) in [
            ("terrain-base-dense-before", 34.0),
            ("terrain-hidden-dense", 15.0),
            ("terrain-base-coarse", 22.0),
            ("terrain-hidden-coarse", 16.0),
            ("terrain-base-dense-after", 36.0),
        ] {
            cells.insert(label.to_string(), json!({"result": {"cpu_ms_p50": value}}));
        }

        let effects = terrain_index_effects(&cells, "cpu_ms_p50");
        assert_eq!(effects["base_dense_drift_ms"], 2.0);
        assert_eq!(effects["dense_terrain_ms"], 20.0);
        assert_eq!(effects["coarse_terrain_ms"], 6.0);
        assert_eq!(effects["terrain_index_density_net_ms"], 14.0);
        assert_eq!(effects["terrain_index_density_reduction_frac"], 0.7);
    }

    #[test]
    fn terrain_culling_effects_subtract_the_hidden_control() {
        let mut cells = serde_json::Map::new();
        for (label, value) in [
            ("terrain-base-full-bounds-before", 42.0),
            ("terrain-hidden-full-bounds", 17.0),
            ("terrain-base-tight-bounds", 31.0),
            ("terrain-hidden-tight-bounds", 16.0),
            ("terrain-base-full-bounds-after", 40.0),
        ] {
            cells.insert(label.to_string(), json!({"result": {"cpu_ms_p50": value}}));
        }

        let effects = terrain_culling_effects(&cells, "cpu_ms_p50");
        assert_eq!(effects["base_full_drift_ms"], -2.0);
        assert_eq!(effects["full_bounds_terrain_ms"], 24.0);
        assert_eq!(effects["tight_bounds_terrain_ms"], 15.0);
        assert_eq!(effects["terrain_tight_bounds_net_ms"], 9.0);
        assert_eq!(effects["terrain_tight_bounds_reduction_frac"], 0.375);
    }
}
