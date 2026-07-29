//! Reading the lane back.
//!
//! Every report tool and every triage pass loads records the same way, so a
//! question asked of the game stream and a question asked of the tool stream
//! are answered from one parser: schema-gated, rotation-aware, and grouped by
//! the session that produced them.
//!
//! Grouping matters more here than it looks. Concurrent Thalos processes are
//! normal (two game instances, a game plus a capture host), they share these
//! files, and a statistic taken across the whole file silently averages them
//! together — which is how a per-process memory budget once read as healthy
//! while the machine ran out of VRAM (INC-20260725T012104Z).

use std::{
    collections::BTreeMap,
    fs,
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
};

use serde_json::{Map, Value};

use crate::sink::SCHEMA;

/// One recorded event.
#[derive(Clone, Debug)]
pub struct Record {
    pub session: String,
    pub ts_unix_ms: u128,
    pub pid: u32,
    pub level: String,
    pub target: String,
    pub fields: Map<String, Value>,
}

impl Record {
    /// The `event` key, or `""` for envelope-only lines such as `session_start`.
    pub fn event(&self) -> &str {
        self.fields
            .get("event")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    pub fn f64(&self, field: &str) -> Option<f64> {
        self.fields.get(field).and_then(Value::as_f64)
    }

    pub fn u64(&self, field: &str) -> Option<u64> {
        self.fields.get(field).and_then(Value::as_u64)
    }

    pub fn str(&self, field: &str) -> Option<&str> {
        self.fields.get(field).and_then(Value::as_str)
    }

    /// Subsystem name after the `thalos::diagnostic::` prefix.
    pub fn subsystem(&self) -> &str {
        self.target
            .strip_prefix(&format!("{}::", crate::TARGET_PREFIX))
            .unwrap_or(&self.target)
    }

    pub fn is_error(&self) -> bool {
        self.level == "ERROR"
    }

    pub fn is_warn(&self) -> bool {
        self.level == "WARN"
    }
}

/// A loaded window of the lane.
#[derive(Debug, Default)]
pub struct Stream {
    /// Records inside the window, oldest first.
    pub records: Vec<Record>,
    /// Files that contributed, for provenance in a report.
    pub sources: Vec<PathBuf>,
    /// Sessions that started inside the window, in first-seen order.
    pub sessions: Vec<String>,
    /// Session id → role (`runtime`, `tool:capture`), for sessions whose
    /// `session_start` line is inside the window.
    pub session_roles: BTreeMap<String, String>,
    /// Lines skipped because they carried another schema (specialized
    /// recorders share the directory) or would not parse.
    pub skipped_lines: usize,
}

impl Stream {
    /// What kind of process produced a session, when known.
    pub fn role(&self, session: &str) -> &str {
        self.session_roles
            .get(session)
            .map(String::as_str)
            .unwrap_or("unknown")
    }

    /// Records belonging to one session.
    pub fn session(&self, session: &str) -> impl Iterator<Item = &Record> {
        self.records
            .iter()
            .filter(move |record| record.session == session)
    }

    /// Records with a given `event` value.
    pub fn events<'a>(&'a self, event: &'a str) -> impl Iterator<Item = &'a Record> {
        self.records
            .iter()
            .filter(move |record| record.event() == event)
    }

    /// Newest timestamp in the window, if any.
    pub fn newest_ts_unix_ms(&self) -> Option<u128> {
        self.records.last().map(|record| record.ts_unix_ms)
    }

    /// Group records by session, preserving encounter order.
    pub fn by_session(&self) -> BTreeMap<&str, Vec<&Record>> {
        let mut grouped: BTreeMap<&str, Vec<&Record>> = BTreeMap::new();
        for record in &self.records {
            grouped
                .entry(record.session.as_str())
                .or_default()
                .push(record);
        }
        grouped
    }
}

/// Load every lane file in `dir` (active sinks and their rotated siblings),
/// keeping records at or after `since_unix_ms`.
///
/// Files whose lines carry another schema are read and skipped rather than
/// rejected: specialized recorders legitimately share this directory, and a
/// triage pass must not fail because one of them exists.
pub fn load(dir: &Path, since_unix_ms: u128) -> io::Result<Stream> {
    let mut stream = Stream::default();
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(stream);
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().is_some_and(|extension| extension == "jsonl")
        })
        .collect();
    files.sort();

    for path in files {
        let Ok(file) = fs::File::open(&path) else {
            continue;
        };
        let mut used = false;
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            if line.trim().is_empty() {
                continue;
            }
            match parse_line(&line, since_unix_ms) {
                ParsedLine::Record(record) => {
                    used = true;
                    stream.records.push(record);
                }
                ParsedLine::SessionStart { session, ts, role } => {
                    used = true;
                    if ts >= since_unix_ms {
                        if !stream.sessions.contains(&session) {
                            stream.sessions.push(session.clone());
                        }
                        stream.session_roles.insert(session, role);
                    }
                }
                ParsedLine::OutsideWindow => used = true,
                ParsedLine::Skip => stream.skipped_lines += 1,
            }
        }
        if used {
            stream.sources.push(path);
        }
    }

    stream.records.sort_by_key(|record| record.ts_unix_ms);
    for record in &stream.records {
        if !stream.sessions.iter().any(|s| s == &record.session) {
            stream.sessions.push(record.session.clone());
        }
    }
    Ok(stream)
}

enum ParsedLine {
    Record(Record),
    SessionStart {
        session: String,
        ts: u128,
        role: String,
    },
    OutsideWindow,
    Skip,
}

fn parse_line(line: &str, since_unix_ms: u128) -> ParsedLine {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return ParsedLine::Skip;
    };
    if value.get("schema").and_then(Value::as_str) != Some(SCHEMA) {
        return ParsedLine::Skip;
    }
    let ts = value
        .get("ts_unix_ms")
        .and_then(Value::as_u64)
        .unwrap_or_default() as u128;
    let session = value
        .get("session")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let Some(fields) = value.get("fields").and_then(Value::as_object) else {
        // Envelope-only line: the per-process `session_start` stamp.
        return if value.get("event").and_then(Value::as_str) == Some("session_start") {
            ParsedLine::SessionStart {
                session,
                ts,
                role: value
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_owned(),
            }
        } else {
            ParsedLine::Skip
        };
    };
    if ts < since_unix_ms {
        return ParsedLine::OutsideWindow;
    }
    ParsedLine::Record(Record {
        session,
        ts_unix_ms: ts,
        pid: value.get("pid").and_then(Value::as_u64).unwrap_or_default() as u32,
        level: value
            .get("level")
            .and_then(Value::as_str)
            .unwrap_or("INFO")
            .to_owned(),
        target: value
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        fields: fields.clone(),
    })
}

/// Milliseconds since the Unix epoch, for window arithmetic.
pub fn now_unix_ms() -> u128 {
    crate::sink::unix_ms()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, lines: &[&str]) {
        fs::create_dir_all(dir).expect("create dir");
        fs::write(dir.join(name), lines.join("\n")).expect("write lane");
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "thalos-reader-{tag}-{}-{}",
            std::process::id(),
            crate::sink::unix_ms()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn foreign_schemas_are_skipped_not_fatal() {
        let dir = temp_dir("schema");
        write(
            &dir,
            "runtime.jsonl",
            &[
                r#"{"schema":"thalos.runtime_diagnostic.v1","session":"a","ts_unix_ms":100,"pid":1,"level":"INFO","target":"thalos::diagnostic::perf","fields":{"event":"frame_gauge","cpu_ms_p95":9.5}}"#,
                r#"{"schema":"some.other.recorder.v3","ts_unix_ms":100,"whatever":true}"#,
                "not json at all",
            ],
        );
        let stream = load(&dir, 0).expect("load");
        fs::remove_dir_all(&dir).ok();
        assert_eq!(stream.records.len(), 1);
        assert_eq!(stream.skipped_lines, 2);
        assert_eq!(stream.records[0].event(), "frame_gauge");
        assert_eq!(stream.records[0].f64("cpu_ms_p95"), Some(9.5));
        assert_eq!(stream.records[0].subsystem(), "perf");
    }

    #[test]
    fn the_window_excludes_older_records_but_reads_rotated_files() {
        let dir = temp_dir("window");
        write(
            &dir,
            "runtime.rot123.jsonl",
            &[
                r#"{"schema":"thalos.runtime_diagnostic.v1","session":"old","ts_unix_ms":10,"pid":1,"level":"INFO","target":"thalos::diagnostic::perf","fields":{"event":"frame_gauge"}}"#,
                r#"{"schema":"thalos.runtime_diagnostic.v1","session":"new","ts_unix_ms":500,"pid":2,"level":"WARN","target":"thalos::diagnostic::tool","fields":{"event":"tool_run","outcome":"error"}}"#,
            ],
        );
        let stream = load(&dir, 100).expect("load");
        fs::remove_dir_all(&dir).ok();
        assert_eq!(stream.records.len(), 1, "the older record is outside");
        assert_eq!(stream.records[0].session, "new");
        assert!(stream.records[0].is_warn());
        assert_eq!(stream.sources.len(), 1, "rotated siblings are read");
    }
}
