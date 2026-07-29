//! One developer-tool invocation, timed by phase.
//!
//! Every agent-runnable tool (`just screenshot` / `capture` / `compare` /
//! `preview` / `map` / `bake` / `texgen`) is infrastructure that agent work
//! rests on, so each run leaves a machine-readable answer to **what it did, how
//! long each phase took, and how it ended**. An exit code plus stderr prose is
//! not that answer: Bevy can log a fatal pipeline error and still exit zero
//! (BL-20), and "capture felt slow today" is not a measurement.
//!
//! ```no_run
//! # use thalos_diagnostics::ToolRun;
//! # use std::time::Instant;
//! let mut run = ToolRun::start("capture", "shot mira-orbit");
//! run.field("preset", "mira-orbit");
//! let started = Instant::now();
//! // … acquire the machine lock …
//! run.phase("lock_wait", started.elapsed());
//! run.ok();
//! ```
//!
//! A record is emitted at both ends of the run: `tool_run_start` when it
//! begins, `tool_run` when it finishes. The start line is what survives a hard
//! kill, which is the case worth diagnosing.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use serde_json::{Map, Value, json};

use crate::sink;

/// Target for developer-tool run records. One target, with `tool` as a field,
/// keeps the target set closed and lets readers filter per tool.
pub const TOOL_TARGET: &str = "thalos::diagnostic::tool";

/// A timed developer-tool invocation. Emits its record on [`ToolRun::ok`],
/// [`ToolRun::fail`], or on drop (as `abandoned`).
#[derive(Debug)]
pub struct ToolRun {
    tool: &'static str,
    command: String,
    started: Instant,
    phases_ms: BTreeMap<String, u128>,
    counters: BTreeMap<String, u64>,
    fields: Map<String, Value>,
    finished: bool,
}

impl ToolRun {
    /// Begin a run and emit its `tool_run_start` record.
    pub fn start(tool: &'static str, command: impl Into<String>) -> Self {
        let command = command.into();
        if let Some(sink) = sink::sink() {
            let mut fields = Map::new();
            fields.insert("event".into(), json!("tool_run_start"));
            fields.insert("tool".into(), json!(tool));
            fields.insert("command".into(), json!(command));
            sink.write_event(TOOL_TARGET, "INFO", fields);
        }
        Self {
            tool,
            command,
            started: Instant::now(),
            phases_ms: BTreeMap::new(),
            counters: BTreeMap::new(),
            fields: Map::new(),
            finished: false,
        }
    }

    /// Record time spent in a named phase. Repeated names accumulate, so a
    /// retried phase reports the total cost the caller actually paid.
    pub fn phase(&mut self, name: &str, elapsed: Duration) {
        *self.phases_ms.entry(name.to_owned()).or_default() += elapsed.as_millis();
    }

    /// Time `body` as a phase and return its value.
    pub fn timed<R>(&mut self, name: &str, body: impl FnOnce() -> R) -> R {
        let started = Instant::now();
        let value = body();
        self.phase(name, started.elapsed());
        value
    }

    /// Increment a named counter (retries, restarts, rebuilds). A counter is
    /// always emitted once bumped, so `0` versus absent is never ambiguous for
    /// the run that recorded it.
    pub fn count(&mut self, name: &str) {
        *self.counters.entry(name.to_owned()).or_default() += 1;
    }

    /// Attach a typed context field (preset, fingerprint, renderer pid …).
    pub fn field(&mut self, name: &str, value: impl Into<Value>) {
        self.fields.insert(name.to_owned(), value.into());
    }

    /// Finish successfully.
    pub fn ok(mut self) {
        self.emit("ok", None);
    }

    /// Finish with a failure reason. The reason is the first line of the error:
    /// the full text belongs on stderr, the classifier belongs here.
    pub fn fail(mut self, reason: &str) {
        let first_line = reason.lines().next().unwrap_or(reason);
        self.emit("error", Some(first_line));
    }

    /// Finish from a `Result`, returning it unchanged.
    pub fn finish<T, E: std::fmt::Display>(self, result: Result<T, E>) -> Result<T, E> {
        match &result {
            Ok(_) => self.ok(),
            Err(error) => self.fail(&error.to_string()),
        }
        result
    }

    fn emit(&mut self, outcome: &str, error: Option<&str>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let Some(sink) = sink::sink() else { return };
        let mut fields = Map::new();
        fields.insert("event".into(), json!("tool_run"));
        fields.insert("tool".into(), json!(self.tool));
        fields.insert("command".into(), json!(self.command));
        fields.insert("outcome".into(), json!(outcome));
        fields.insert(
            "total_ms".into(),
            json!(self.started.elapsed().as_millis() as u64),
        );
        for (name, ms) in &self.phases_ms {
            fields.insert(format!("phase_{name}_ms"), json!(*ms as u64));
        }
        for (name, count) in &self.counters {
            fields.insert(format!("{name}_count"), json!(*count));
        }
        for (name, value) in std::mem::take(&mut self.fields) {
            fields.insert(name, value);
        }
        if let Some(error) = error {
            fields.insert("error".into(), json!(error));
        }
        sink.write_event(
            TOOL_TARGET,
            if outcome == "ok" { "INFO" } else { "WARN" },
            fields,
        );
    }
}

impl Drop for ToolRun {
    fn drop(&mut self) {
        // A panic or an early `?` return still leaves a record; only a hard
        // abort escapes, and then the `tool_run_start` line is the tell.
        self.emit("abandoned", None);
    }
}
