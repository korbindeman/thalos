//! Per-shot capture telemetry.
//!
//! The headless capture lane is the throughput floor for agent work, so its
//! cost and reliability are measured rather than recalled. Every shot — a
//! `just screenshot`, one scene of a `just capture` batch, one variant of a
//! `just compare` matrix — records what it did, how long each phase took, and
//! how it ended, into `artifacts/diagnostics/tools.jsonl` through the shared
//! `thalos_diagnostics` contract.
//!
//! The record is process-global rather than threaded through every signature:
//! a capture client is a short-lived single-threaded CLI running exactly one
//! shot at a time, and the phases worth timing (source fingerprinting, host
//! start, shader reload, render, validation) are spread across call layers that
//! would otherwise all grow a parameter they do not otherwise need.

use std::{sync::Mutex, time::Duration};

use serde_json::Value;
use thalos_diagnostics::ToolRun;

static ACTIVE: Mutex<Option<ToolRun>> = Mutex::new(None);

/// Diagnostic target for capture-client events outside a shot record.
pub const TARGET: &str = "thalos::diagnostic::capture";

/// Install the shared tool lane. Best effort: an unwritable sink costs
/// observability, never the capture itself.
pub fn install() {
    thalos_diagnostics::install_tool_lane("capture");
}

/// Begin a shot record, replacing (and finishing as `abandoned`) any previous
/// one that was never closed.
pub fn begin(command: impl Into<String>) {
    let run = ToolRun::start("capture", command);
    if let Ok(mut active) = ACTIVE.lock() {
        *active = Some(run);
    }
}

/// Record time spent in a named phase; repeats accumulate.
pub fn phase(name: &str, elapsed: Duration) {
    with(|run| run.phase(name, elapsed));
}

/// Increment a named counter (restarts, retries, rebuild recoveries).
pub fn count(name: &str) {
    with(|run| run.count(name));
}

/// Attach a typed context field.
pub fn field(name: &str, value: impl Into<Value>) {
    with(|run| run.field(name, value));
}

/// Close the active shot record with the outcome of `result`.
pub fn finish<T, E: std::fmt::Display>(result: &Result<T, E>) {
    let Ok(mut active) = ACTIVE.lock() else {
        return;
    };
    let Some(run) = active.take() else { return };
    match result {
        Ok(_) => run.ok(),
        Err(error) => run.fail(&error.to_string()),
    }
}

fn with(body: impl FnOnce(&mut ToolRun)) {
    if let Ok(mut active) = ACTIVE.lock()
        && let Some(run) = active.as_mut()
    {
        body(run);
    }
}

/// Record the machine-wide capture lock wait — contention between parallel
/// agents, which is process-level cost and would be misattributed as one
/// shot's latency.
pub fn record_lock_wait(waited: Duration, owner_pid: Option<u32>, outcome: &str) {
    let waited_ms = waited.as_millis() as u64;
    match owner_pid {
        Some(owner_pid) => tracing::info!(
            target: TARGET,
            event = "capture_lock",
            outcome,
            wait_ms = waited_ms,
            queued = true,
            owner_pid,
            "capture lock wait"
        ),
        None => tracing::info!(
            target: TARGET,
            event = "capture_lock",
            outcome,
            wait_ms = waited_ms,
            queued = false,
            "capture lock wait"
        ),
    }
}
