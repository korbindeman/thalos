//! Bevy adapter for the shared diagnostics lane.
//!
//! The event contract, the JSONL sink, session ids, and storage hygiene live in
//! `thalos_diagnostics`, so the capture client and the offline tools record on
//! the same terms as the game. Only the two Bevy-shaped pieces stay here: the
//! layer handed to Bevy's `LogPlugin`, and the console formatter that keeps the
//! machine-readable lane out of the human terminal.

use std::io;

use bevy::log::{
    BoxedFmtLayer, BoxedLayer,
    tracing_subscriber::{Layer, filter::FilterFn},
};
use bevy::prelude::*;

pub use thalos_diagnostics::session_id;

/// Build the JSONL tracing layer installed alongside capture error accounting.
///
/// Opening the sink also runs the one process-start housekeeping pass:
/// oversized JSONL files rotate away and the diagnostics directory is pruned
/// back under its total budget.
pub fn jsonl_layer() -> io::Result<BoxedLayer> {
    Ok(Box::new(thalos_diagnostics::runtime_layer()?))
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
                !thalos_diagnostics::is_diagnostic_target(metadata.target())
                    || matches!(
                        *metadata.level(),
                        tracing::Level::WARN | tracing::Level::ERROR
                    )
            })),
    ))
}
