//! Structured runtime diagnostics.
//!
//! Stdout/stderr is the operator log: short lifecycle messages and actionable
//! warnings. Numeric telemetry, periodic state snapshots, and investigation
//! traces use the `thalos::diagnostic::*` tracing target instead. This layer
//! writes those events as JSONL and the formatter below keeps them out of the
//! human-facing console.

use std::{
    io::{self, Write},
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use bevy::{
    log::{
        BoxedFmtLayer, BoxedLayer,
        tracing_subscriber::{Layer, filter::FilterFn, layer::Context, registry::Registry},
    },
    prelude::*,
};
use serde_json::{Map, Value, json};
use tracing::{
    Event,
    field::{Field, Visit},
};

pub const TARGET_PREFIX: &str = "thalos::diagnostic";
const LOG_FILENAME: &str = "runtime.jsonl";
const SCHEMA: &str = "thalos.runtime_diagnostic.v1";

/// Build the JSONL tracing layer installed alongside capture error accounting.
pub fn jsonl_layer() -> io::Result<BoxedLayer> {
    let path =
        crate::artifact_paths::jsonl_path_from_env_or("THALOS_RUNTIME_DIAGNOSTICS", LOG_FILENAME);
    let file = crate::artifact_paths::open_jsonl_append(&path)?;
    let started_unix_ms = unix_ms();
    let session = format!("{}-{started_unix_ms}", std::process::id());
    let layer = JsonlDiagnosticLayer {
        writer: Mutex::new(file),
        session: session.clone(),
        write_error_reported: AtomicBool::new(false),
    };
    layer.write_value(&json!({
        "schema": SCHEMA,
        "session": session,
        "ts_unix_ms": started_unix_ms,
        "pid": std::process::id(),
        "event": "session_start",
    }))?;
    Ok(Box::new(layer))
}

/// Replace Bevy's console formatter with one that omits artifact-only events.
///
/// The shared `EnvFilter` still controls whether events exist at all, so
/// `RUST_LOG` remains an escape hatch. This per-layer filter only decides what
/// humans see in the terminal.
pub fn human_console_layer(_app: &mut App) -> Option<BoxedFmtLayer> {
    Some(Box::new(
        bevy::log::tracing_subscriber::fmt::Layer::default()
            .with_writer(std::io::stderr)
            .with_filter(FilterFn::new(|metadata| {
                !metadata.target().starts_with(TARGET_PREFIX)
                    || matches!(
                        *metadata.level(),
                        tracing::Level::WARN | tracing::Level::ERROR
                    )
            })),
    ))
}

struct JsonlDiagnosticLayer {
    writer: Mutex<std::fs::File>,
    session: String,
    write_error_reported: AtomicBool,
}

impl JsonlDiagnosticLayer {
    fn write_value(&self, value: &Value) -> io::Result<()> {
        let mut line = serde_json::to_vec(value)?;
        line.push(b'\n');
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| io::Error::other("runtime diagnostic writer lock poisoned"))?;
        writer.write_all(&line)?;
        writer.flush()
    }
}

impl Layer<Registry> for JsonlDiagnosticLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, Registry>) {
        if !event.metadata().target().starts_with(TARGET_PREFIX) {
            return;
        }

        let mut visitor = JsonVisitor::default();
        event.record(&mut visitor);
        let record = json!({
            "schema": SCHEMA,
            "session": self.session,
            "ts_unix_ms": unix_ms(),
            "pid": std::process::id(),
            "level": event.metadata().level().as_str(),
            "target": event.metadata().target(),
            "fields": visitor.fields,
        });
        if let Err(error) = self.write_value(&record) {
            // Do not recursively log from a tracing layer. A diagnostic sink
            // failure is actionable and should remain visible, once.
            if !self.write_error_reported.swap(true, Ordering::Relaxed) {
                eprintln!("runtime diagnostic JSONL write failed: {error}");
            }
        }
    }
}

#[derive(Default)]
struct JsonVisitor {
    fields: Map<String, Value>,
}

impl Visit for JsonVisitor {
    fn record_f64(&mut self, field: &Field, value: f64) {
        self.fields.insert(field.name().to_string(), json!(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields.insert(field.name().to_string(), json!(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields.insert(field.name().to_string(), json!(value));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields.insert(field.name().to_string(), json!(value));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields.insert(field.name().to_string(), json!(value));
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        self.fields
            .insert(field.name().to_string(), json!(value.to_string()));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.fields
            .insert(field.name().to_string(), json!(format!("{value:?}")));
    }
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::log::tracing_subscriber::prelude::*;

    #[test]
    fn diagnostic_events_are_jsonl_and_other_targets_are_ignored() {
        let path = std::env::temp_dir().join(format!(
            "thalos-runtime-diagnostics-{}-{}.jsonl",
            std::process::id(),
            unix_ms()
        ));
        let file = std::fs::File::create(&path).expect("create test JSONL");
        let layer = JsonlDiagnosticLayer {
            writer: Mutex::new(file),
            session: "test-session".to_string(),
            write_error_reported: AtomicBool::new(false),
        };
        let subscriber = Registry::default().with(layer);

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

        let contents = std::fs::read_to_string(&path).expect("read test JSONL");
        std::fs::remove_file(&path).expect("remove test JSONL");
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(lines.len(), 1);
        let record: Value = serde_json::from_str(lines[0]).expect("valid JSON object");
        assert_eq!(record["schema"], SCHEMA);
        assert_eq!(record["session"], "test-session");
        assert_eq!(record["target"], "thalos::diagnostic::test");
        assert_eq!(record["fields"]["event"], "sample");
        assert_eq!(record["fields"]["count"], 3);
        assert_eq!(record["fields"]["ratio"], 0.5);
        assert_eq!(record["fields"]["ready"], true);
    }
}
