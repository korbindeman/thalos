//! Thalos diagnostics: one event contract, one sink, for every process.
//!
//! Stdout/stderr is the operator log — short lifecycle messages and actionable
//! warnings. Numeric telemetry, periodic state snapshots, and investigation
//! traces go to the machine-readable lane instead:
//!
//! ```no_run
//! tracing::info!(
//!     target: "thalos::diagnostic::tile_terrain",
//!     event = "residency_gauge",
//!     resident_count = 1_280_u64,
//!     resident_mib = 434.0_f64,
//!     budget_mib = 2048.0_f64,
//!     "tile residency gauge"
//! );
//! ```
//!
//! The lane is deliberately usable outside the game. `thalos_runtime` installs
//! it as a Bevy log layer; offline tools install it with
//! [`install_tool_lane`] and get the same file format, the same session id, and
//! the same storage hygiene — which is what lets one reader answer questions
//! about the game and about the tooling that captures it.
//!
//! **Event shape** (the contract readers depend on):
//!
//! - `event = "<snake_case_noun>"` is the stable key; renaming one breaks every
//!   reader, so treat it as an API.
//! - Fields are flat scalars with the unit in the name (`_ms`, `_mib`, `_m`,
//!   `_hz`, `_frac`, `_count`) — never a pre-formatted string to re-parse. The
//!   trailing message is a human label, not the payload.
//! - Emit the denominator with the numerator (`resident_mib` beside
//!   `budget_mib`); a number that cannot separate two hypotheses is not a
//!   diagnostic (INC-20260725T012104Z).
//! - Every line carries `pid` and `session`; concurrent processes are normal
//!   here, so aggregate per session, never across a whole file.

use std::{io, sync::Arc};

pub mod layer;
pub mod paths;
pub mod reader;
pub mod run;
pub mod sink;

pub use layer::{JsonlDiagnosticLayer, is_diagnostic_target};
pub use reader::{Record, Stream};
pub use run::ToolRun;
pub use sink::{DiagnosticSink, SCHEMA, session_id, sink};

/// Target prefix for the machine-readable lane. Anything below it is written to
/// JSONL and kept off the human console.
pub const TARGET_PREFIX: &str = "thalos::diagnostic";

/// Default sink for a game-shaped process (game, capture host).
pub const RUNTIME_LOG_FILENAME: &str = "runtime.jsonl";
/// Default sink for developer tools. Separate from the runtime stream because
/// its lifecycle is per-invocation and its readers ask different questions —
/// "is the capture lane fast and stable this week?" should not require paging
/// past a session of frame gauges.
pub const TOOL_LOG_FILENAME: &str = "tools.jsonl";

/// Environment override for the runtime sink path (bare filename resolves under
/// `artifacts/diagnostics/`).
pub const RUNTIME_LOG_ENV: &str = "THALOS_RUNTIME_DIAGNOSTICS";
/// Environment override for the developer-tool sink path.
pub const TOOL_LOG_ENV: &str = "THALOS_TOOL_DIAGNOSTICS";

/// Session role for a game-shaped process.
pub const ROLE_RUNTIME: &str = "runtime";

/// Open and install the runtime sink, returning the tracing layer to register.
///
/// The caller owns subscriber registration because Bevy's log plugin wants the
/// layer rather than a global subscriber.
pub fn runtime_layer() -> io::Result<JsonlDiagnosticLayer> {
    let path = paths::jsonl_path_from_env_or(RUNTIME_LOG_ENV, RUNTIME_LOG_FILENAME);
    let sink = DiagnosticSink::install(DiagnosticSink::open(&path, ROLE_RUNTIME)?);
    Ok(JsonlDiagnosticLayer::new(Arc::clone(sink)))
}

/// Install the developer-tool lane as this process's global tracing subscriber.
///
/// `tool` is the binary's own name (`capture`, `bake`, …), recorded as this
/// session's role.
///
/// Best-effort by design: a tool that cannot open its diagnostic sink still
/// does its job, it just does it unobserved. Returns whether the lane is live,
/// for callers that want to say so.
pub fn install_tool_lane(tool: &str) -> bool {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    let path = paths::jsonl_path_from_env_or(TOOL_LOG_ENV, TOOL_LOG_FILENAME);
    let Ok(sink) = DiagnosticSink::open(&path, &format!("tool:{tool}")) else {
        return false;
    };
    let sink = DiagnosticSink::install(sink);
    tracing_subscriber::registry()
        .with(JsonlDiagnosticLayer::new(Arc::clone(sink)))
        .try_init()
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::sync::Arc;
    use tracing_subscriber::{layer::SubscriberExt, registry::Registry};

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "thalos-diagnostics-{tag}-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    fn records(path: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(path)
            .expect("read sink")
            .lines()
            .map(|line| serde_json::from_str(line).expect("valid JSON line"))
            .collect()
    }

    #[test]
    fn diagnostic_events_are_jsonl_and_other_targets_are_ignored() {
        let path = temp_path("layer");
        let sink = DiagnosticSink::open(&path, "test").expect("open sink");
        let subscriber = Registry::default().with(JsonlDiagnosticLayer::new(Arc::clone(&sink)));

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(
                target: "thalos::diagnostic::test",
                event = "sample",
                count = 3_u64,
                ratio = 0.5_f64,
                ready = true,
                "structured sample"
            );
            tracing::info!(target: "thalos::status", "human-only sample");
        });

        let lines = records(&path);
        std::fs::remove_file(&path).ok();
        // session_start, then the one diagnostic event.
        assert_eq!(lines.len(), 2, "only diagnostic targets are recorded");
        let record = &lines[1];
        assert_eq!(record["schema"], SCHEMA);
        assert_eq!(record["session"], sink.session());
        assert_eq!(record["target"], "thalos::diagnostic::test");
        assert_eq!(record["fields"]["event"], "sample");
        assert_eq!(record["fields"]["count"], 3);
        assert_eq!(record["fields"]["ratio"], 0.5);
        assert_eq!(record["fields"]["ready"], true);
    }

    /// The layer must be usable from a plain (non-Bevy) subscriber — that is
    /// the whole point of the extraction, so it is a test, not a comment.
    #[test]
    fn phases_and_counters_flatten_with_units_in_the_name() {
        let path = temp_path("run");
        let sink = DiagnosticSink::open(&path, "test").expect("open sink");
        // Not installed process-wide (another test may own that), so drive the
        // record shape through the same writer the tool lane uses.
        let mut fields = serde_json::Map::new();
        fields.insert("event".into(), Value::from("tool_run"));
        fields.insert("phase_render_ms".into(), Value::from(1200_u64));
        fields.insert("restart_count".into(), Value::from(1_u64));
        sink.write_event(run::TOOL_TARGET, "INFO", fields);

        let lines = records(&path);
        std::fs::remove_file(&path).ok();
        let record = &lines[1];
        assert_eq!(record["target"], run::TOOL_TARGET);
        assert_eq!(record["fields"]["phase_render_ms"], 1200);
        assert_eq!(record["fields"]["restart_count"], 1);
    }
}
