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
        "gpu_ms_mean": mean_of("gpu_ms_mean"),
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
