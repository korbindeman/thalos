//! The JSONL sink every diagnostic line goes through.
//!
//! One line per event, one schema for every producer:
//!
//! ```json
//! {"schema":"thalos.runtime_diagnostic.v1","session":"12345-1753",
//!  "ts_unix_ms":1753…, "pid":12345, "level":"INFO",
//!  "target":"thalos::diagnostic::tile_terrain", "fields":{…}}
//! ```
//!
//! Two producers write through it: the [`crate::layer`] tracing layer (the
//! `info!(target: "thalos::diagnostic::*", …)` path used across the game) and
//! [`crate::run::ToolRun`], which needs per-run dynamic field names that the
//! `tracing` macros cannot express. Both emit the same line shape, so a reader
//! never has to care which one produced a record.

use std::{
    io::{self, Write},
    path::Path,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Map, Value, json};

use crate::paths;

/// Schema tag on every line. Bump only for a breaking envelope change; adding a
/// field inside `fields` is not one.
pub const SCHEMA: &str = "thalos.runtime_diagnostic.v1";

/// Process-wide sink, set by whichever install path the process uses.
static SINK: OnceLock<Arc<DiagnosticSink>> = OnceLock::new();

/// The active process sink, if diagnostics were installed.
///
/// Returns `None` in processes that never installed a lane (unit tests, small
/// binaries), which makes every emit site a no-op rather than a panic.
pub fn sink() -> Option<&'static Arc<DiagnosticSink>> {
    SINK.get()
}

/// This process's diagnostic stream session id (`<pid>-<start-unix-ms>`) — the
/// value every line it writes carries, and the key a reader groups by. Shown on
/// the F3 debug view so a live run can be joined to its recorded stream.
pub fn session_id() -> &'static str {
    sink().map(|sink| sink.session()).unwrap_or("unstarted")
}

/// An append-only JSONL diagnostic stream.
#[derive(Debug)]
pub struct DiagnosticSink {
    writer: Mutex<std::fs::File>,
    session: String,
    write_error_reported: AtomicBool,
}

impl DiagnosticSink {
    /// Open (or create) `path` for append and stamp a `session_start` line.
    ///
    /// `role` names what kind of process this is (`runtime`, `tool:capture`).
    /// A reader needs it to judge silence: a `capture status` invocation that
    /// records nothing is correct, while a game session that records nothing
    /// died during boot.
    ///
    /// Rotation/prune of the diagnostics directory runs once here, so every
    /// process that opens a lane also pays its share of storage hygiene.
    pub fn open(path: &Path, role: &str) -> io::Result<Arc<Self>> {
        paths::rotate_and_prune_diagnostics();
        let file = paths::open_jsonl_append(path)?;
        let started_unix_ms = unix_ms();
        let sink = Arc::new(Self {
            writer: Mutex::new(file),
            session: format!("{}-{started_unix_ms}", std::process::id()),
            write_error_reported: AtomicBool::new(false),
        });
        sink.write_value(&json!({
            "schema": SCHEMA,
            "session": sink.session,
            "ts_unix_ms": started_unix_ms,
            "pid": std::process::id(),
            "event": "session_start",
            "role": role,
        }))?;
        Ok(sink)
    }

    /// Publish `sink` as the process sink. The first caller wins; a later call
    /// is ignored, because a second lane would silently split one session's
    /// events across two files.
    pub fn install(sink: Arc<Self>) -> &'static Arc<Self> {
        let _ = SINK.set(sink);
        SINK.get().expect("sink set above")
    }

    /// This stream's session id.
    pub fn session(&self) -> &str {
        &self.session
    }

    /// Write one event record with the shared envelope.
    pub fn write_event(&self, target: &str, level: &str, fields: Map<String, Value>) {
        let record = json!({
            "schema": SCHEMA,
            "session": self.session,
            "ts_unix_ms": unix_ms(),
            "pid": std::process::id(),
            "level": level,
            "target": target,
            "fields": fields,
        });
        if let Err(error) = self.write_value(&record) {
            // Never log from here: this can run inside a tracing layer, and a
            // recursive emit would deadlock the writer lock. A sink failure is
            // actionable, so it stays visible — once.
            if !self.write_error_reported.swap(true, Ordering::Relaxed) {
                eprintln!("runtime diagnostic JSONL write failed: {error}");
            }
        }
    }

    fn write_value(&self, value: &Value) -> io::Result<()> {
        let mut line = serde_json::to_vec(value)?;
        line.push(b'\n');
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| io::Error::other("runtime diagnostic writer lock poisoned"))?;
        // One `write_all` of a complete line under O_APPEND: the game, the
        // capture host, and the capture client share this directory, and a
        // per-line append is what keeps their interleaving readable.
        writer.write_all(&line)?;
        writer.flush()
    }
}

pub(crate) fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
