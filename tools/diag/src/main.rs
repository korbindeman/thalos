//! `just diag` — read the diagnostics lane and say what needs a closer look.
//!
//! This is the reader half of the diagnostics system: the game and the tools
//! write typed events, and one command turns a window of them into a short,
//! ranked list of findings an agent (or a daily routine) can act on. It never
//! fails a build and never exits non-zero on a finding — it is a report, and
//! the decision about what to do belongs to whoever reads it.
//!
//! The design rule is signal-to-noise: a healthy window prints nothing but its
//! header, and every finding carries the number and denominator behind it.

use std::{path::PathBuf, process::ExitCode};

use thalos_diagnostics::{paths, reader};

mod checks;
mod finding;

use finding::Severity;

const DEFAULT_WINDOW_HOURS: f64 = 24.0;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("diagnostics triage failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut hours = DEFAULT_WINDOW_HOURS;
    let mut dir: Option<PathBuf> = None;
    let mut as_json = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--since" | "--hours" => {
                let raw = args.next().ok_or("--since requires hours")?;
                hours = raw
                    .trim()
                    .trim_end_matches('h')
                    .parse::<f64>()
                    .map_err(|_| format!("--since expects hours, got {raw:?}"))?;
                if !(hours.is_finite() && hours > 0.0) {
                    return Err(format!("--since expects a positive window, got {raw:?}"));
                }
            }
            "--dir" => dir = Some(PathBuf::from(args.next().ok_or("--dir requires a path")?)),
            "--json" => as_json = true,
            "-h" | "--help" | "help" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown option {other:?}; use --help")),
        }
    }

    let dir = dir.unwrap_or_else(|| paths::diagnostics_dir().to_path_buf());
    let now = reader::now_unix_ms();
    let since = now.saturating_sub((hours * 3_600_000.0) as u128);
    let stream = reader::load(&dir, since).map_err(|error| format!("read {dir:?}: {error}"))?;

    let activity = checks::activity(&stream);
    let findings = checks::run(&stream);

    if as_json {
        let report = serde_json::json!({
            "schema": "thalos.diag_report.v1",
            "window_hours": hours,
            "since_unix_ms": since as u64,
            "sessions": activity.sessions,
            "records": activity.records,
            "capture_shots": activity.shots,
            "capture_shots_failed": activity.shots_failed,
            "sources": stream
                .sources
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>(),
            "findings": findings.iter().map(finding::Finding::to_json).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
        );
        return Ok(());
    }

    println!(
        "diagnostics triage · last {hours}h · {dir}",
        hours = trim_float(hours),
        dir = dir.display()
    );
    println!(
        "  {} record{} in {} session{}{}",
        activity.records,
        plural(activity.records),
        activity.sessions,
        plural(activity.sessions),
        match (activity.oldest_ts, activity.newest_ts) {
            (Some(oldest), Some(newest)) => format!(
                " · {} → {}",
                checks::stamp(oldest),
                checks::stamp(newest)
            ),
            _ => String::new(),
        }
    );
    if activity.shots > 0 {
        println!(
            "  {} capture shot{}, {} failed",
            activity.shots,
            plural(activity.shots),
            activity.shots_failed
        );
    }
    if stream.skipped_lines > 0 {
        println!(
            "  ({} line{} from other recorders skipped)",
            stream.skipped_lines,
            plural(stream.skipped_lines)
        );
    }
    println!();

    if findings.is_empty() {
        println!("nothing crossed a threshold. no action needed.");
        return Ok(());
    }

    let mut current: Option<Severity> = None;
    for item in &findings {
        if current != Some(item.severity) {
            println!("{}", item.severity);
            current = Some(item.severity);
        }
        println!("  [{}] {}", item.id, item.headline);
        for line in &item.detail {
            println!("      {line}");
        }
        println!("      → {}", item.next);
    }
    println!();
    println!(
        "{} finding{}. Thresholds and their rationale: tools/diag/src/finding.rs",
        findings.len(),
        plural(findings.len())
    );
    Ok(())
}

/// `24` rather than `24.0`, so the header reads like a human wrote it.
fn trim_float(value: f64) -> String {
    if (value - value.round()).abs() < f64::EPSILON {
        format!("{}", value.round() as i64)
    } else {
        format!("{value}")
    }
}

fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

fn print_help() {
    println!(
        "thalos_diag — report what in the diagnostics lane needs a closer look

USAGE:
    just diag [hours]
    cargo run -q -p thalos_diag -- [--since <hours>] [--dir <path>] [--json]

OPTIONS:
    --since <hours>   window to analyze (default 24h)
    --dir <path>      diagnostics directory (default artifacts/diagnostics)
    --json            machine-readable report

Exit status is 0 whether or not there are findings; this is a report, not a gate."
    );
}
