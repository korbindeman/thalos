//! Fatal-error detection for headless captures.
//!
//! **Why this exists (BL-20).** Bevy logs shader compilation and render-pipeline
//! validation failures at `ERROR` and then carries on: the frame renders without
//! the offending pass, the screenshot is written, and the process exits **zero**.
//! An agent driving `just screenshot` therefore sees a PNG appear and a success
//! exit code for a run whose render graph was partly dead. The existence of an
//! output file is not evidence that the capture is valid.
//!
//! A capture that logged a pipeline error is not a slightly-worse capture, it is
//! a *void* one — comparing it against a baseline produces confident nonsense.
//! So the host counts `ERROR` events through a tracing layer and exits non-zero
//! if any were seen, which turns a silent bad run into a loud one.
//!
//! Scope note: this deliberately counts **all** `ERROR` events rather than
//! pattern-matching shader messages. Anything a subsystem considered
//! error-worthy invalidates a deterministic verification capture, and an
//! allowlist would rot as new failure modes appear. `LogPlugin`'s filter still
//! governs what reaches the layer at all.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

// `tracing_subscriber` comes via `bevy::log`'s re-export rather than a direct
// dependency, so the layer cannot drift from the subscriber Bevy actually
// builds.
use bevy::log::tracing_subscriber::Layer;
use bevy::log::tracing_subscriber::layer::Context;
use bevy::log::tracing_subscriber::registry::Registry;
use bevy::log::BoxedLayer;
use bevy::prelude::*;
use tracing::Event;
use tracing::field::{Field, Visit};

/// Number of distinct error messages retained for the exit summary. The first
/// few are what identify the failing pass; the rest are usually the same
/// pipeline failing again on later frames.
const RETAINED_MESSAGES: usize = 5;

static ERROR_COUNT: AtomicUsize = AtomicUsize::new(0);

fn retained() -> &'static Mutex<Vec<String>> {
    static RETAINED: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    RETAINED.get_or_init(|| Mutex::new(Vec::new()))
}

/// Total `ERROR` events observed since process start.
pub fn error_count() -> usize {
    ERROR_COUNT.load(Ordering::Relaxed)
}

/// First [`RETAINED_MESSAGES`] error messages, for the exit summary.
pub fn error_messages() -> Vec<String> {
    retained().lock().map(|m| m.clone()).unwrap_or_default()
}

/// Install into [`bevy::log::LogPlugin::custom_layer`].
pub fn error_capture_layer(_app: &mut App) -> Option<BoxedLayer> {
    Some(Box::new(ErrorCounterLayer))
}

struct ErrorCounterLayer;

impl Layer<Registry> for ErrorCounterLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, Registry>) {
        if *event.metadata().level() != tracing::Level::ERROR {
            return;
        }
        ERROR_COUNT.fetch_add(1, Ordering::Relaxed);

        let Ok(mut messages) = retained().lock() else {
            return;
        };
        if messages.len() >= RETAINED_MESSAGES {
            return;
        }
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let target = event.metadata().target();
        let body = if visitor.message.is_empty() {
            event.metadata().name().to_string()
        } else {
            visitor.message
        };
        messages.push(format!("{target}: {body}"));
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        }
    }
}
