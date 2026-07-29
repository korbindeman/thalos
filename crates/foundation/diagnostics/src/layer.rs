//! The `tracing` layer that routes `thalos::diagnostic::*` events to the sink.
//!
//! Generic over the subscriber so the same type serves Bevy's `Registry`-based
//! log plugin and a plain `tracing_subscriber` registry in an offline tool.

use std::sync::Arc;

use serde_json::{Map, Value, json};
use tracing::{
    Event, Subscriber,
    field::{Field, Visit},
};
use tracing_subscriber::{layer::Context, registry::LookupSpan};

use crate::{TARGET_PREFIX, sink::DiagnosticSink};

/// Routes events whose target starts with [`TARGET_PREFIX`] into a
/// [`DiagnosticSink`] as JSONL, and ignores every other target.
#[derive(Debug)]
pub struct JsonlDiagnosticLayer {
    sink: Arc<DiagnosticSink>,
}

impl JsonlDiagnosticLayer {
    /// Build a layer writing into `sink`.
    pub fn new(sink: Arc<DiagnosticSink>) -> Self {
        Self { sink }
    }
}

impl<S> tracing_subscriber::Layer<S> for JsonlDiagnosticLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        if !metadata.target().starts_with(TARGET_PREFIX) {
            return;
        }
        let mut visitor = JsonVisitor::default();
        event.record(&mut visitor);
        self.sink
            .write_event(metadata.target(), metadata.level().as_str(), visitor.fields);
    }
}

/// True when this target belongs to the machine-readable lane, i.e. when a
/// human console formatter should suppress its informational events.
pub fn is_diagnostic_target(target: &str) -> bool {
    target.starts_with(TARGET_PREFIX)
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
