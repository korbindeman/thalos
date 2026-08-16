//! Shared, extensible F3 diagnostics surface for interactive Thalos apps.
//!
//! The core owns one frame-history authority, the common panel and graph, F3
//! state, and stable extension ordering. Applications contribute typed Bevy UI
//! sections and side effects through the exported roots and system sets.
//! Process event storage remains in the Bevy-free `thalos_diagnostics` crate.

mod graph;
mod panel;
mod samples;

pub use graph::{DiagnosticsGraphMaterial, DiagnosticsGraphMode};
pub use panel::{
    DiagnosticsPanelExtensions, DiagnosticsPanelGate, DiagnosticsPanelMemoryExtensions,
    DiagnosticsPanelPlugin, DiagnosticsPanelRoot, DiagnosticsPanelStartupSet,
    DiagnosticsPanelState, DiagnosticsPanelUpdateSet, entity_count, spawn_section_header,
    spawn_text_section,
};
pub use samples::{
    DiagnosticsPanelPostUpdateSet, FRAME_HISTORY_LEN, FrameSamples, FrameStats, gpu_frame_ms,
};

/// `"820 MiB"` / `"3.2 GiB"` with one stable unit switch.
pub fn format_mib(mib: f32) -> String {
    if mib >= 1024.0 {
        format!("{:.1} GiB", mib / 1024.0)
    } else {
        format!("{mib:.0} MiB")
    }
}

/// [`format_mib`] for a byte count.
pub fn format_bytes(bytes: u64) -> String {
    format_mib(bytes as f32 / (1024.0 * 1024.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_format_changes_unit_without_long_mib_values() {
        assert_eq!(format_bytes(820 * 1024 * 1024), "820 MiB");
        assert_eq!(format_bytes(3_277 * 1024 * 1024), "3.2 GiB");
    }
}
