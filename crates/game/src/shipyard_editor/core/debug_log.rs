//! Opt-in JSONL trace of the selection decision path.
//!
//! "Shipyard selection is a bit broken" reports are hard to debug by eye:
//! picking, the press→release drag arbitration, and the deselect-on-empty
//! click all interact across frames. This module records every step as one
//! JSON object per line so a repro can be replayed after the fact.
//!
//! **Off by default.** Logging is enabled only when the environment variable
//! `THALOS_SHIPYARD_SELECT_LOG` names a writable file path; otherwise every
//! `event(..)` call is a cheap early return. House style is JSONL (see
//! `CLAUDE.md` — "JSONL is the house style for machine-readable runtime
//! data"). Example:
//!
//! ```bash
//! THALOS_SHIPYARD_SELECT_LOG=shipyard-select.jsonl cargo run -p thalos_game -- shipyard
//! ```
//!
//! then inspect `tools/diagnostics/shipyard-select.jsonl`.

use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::prelude::*;

use super::state::EditorState;

/// One JSON field value. Keeps call sites declarative — a slice of
/// `(key, value)` pairs — instead of hand-rolling format strings per event.
pub enum Field<'a> {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(&'a str),
    Owned(String),
    Null,
}

impl From<i64> for Field<'_> {
    fn from(v: i64) -> Self {
        Field::Int(v)
    }
}
impl From<f32> for Field<'_> {
    fn from(v: f32) -> Self {
        Field::Float(v as f64)
    }
}
impl From<u32> for Field<'_> {
    fn from(v: u32) -> Self {
        Field::Int(v as i64)
    }
}
impl From<bool> for Field<'_> {
    fn from(v: bool) -> Self {
        Field::Bool(v)
    }
}
impl<'a> From<&'a str> for Field<'a> {
    fn from(v: &'a str) -> Self {
        Field::Str(v)
    }
}
impl From<String> for Field<'_> {
    fn from(v: String) -> Self {
        Field::Owned(v)
    }
}
impl From<Entity> for Field<'_> {
    fn from(e: Entity) -> Self {
        Field::Int(e.index().index() as i64)
    }
}
impl From<Option<Entity>> for Field<'_> {
    fn from(e: Option<Entity>) -> Self {
        e.map_or(Field::Null, |e| Field::Int(e.index().index() as i64))
    }
}

impl Field<'_> {
    fn write_json(&self, out: &mut String) {
        match self {
            Field::Int(v) => {
                let _ = write!(out, "{v}");
            }
            Field::Float(v) => {
                if v.is_finite() {
                    let _ = write!(out, "{v}");
                } else {
                    out.push_str("null");
                }
            }
            Field::Bool(v) => {
                let _ = write!(out, "{v}");
            }
            Field::Null => out.push_str("null"),
            Field::Str(s) => write_json_string(out, s),
            Field::Owned(s) => write_json_string(out, s),
        }
    }
}

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Sink for the selection trace. A `None` writer means logging is disabled
/// (the common case) and every [`event`](Self::event) call returns immediately.
///
/// Interior-mutable (the file behind a `Mutex`, the sequence behind an
/// `AtomicU64`) so the picking *observers* can take it as a plain `Res` —
/// observers fire synchronously at pointer-event time and never overlap a
/// `ResMut` writer.
#[derive(Resource, Default)]
pub struct SelectionLog {
    writer: Option<Mutex<std::fs::File>>,
    seq: AtomicU64,
}

impl SelectionLog {
    /// Build from `THALOS_SHIPYARD_SELECT_LOG`. A bare filename is placed under
    /// `tools/diagnostics/`. Appends (never truncates) so a run's trace
    /// accumulates; delete the file between runs for a clean log.
    pub fn from_env() -> Self {
        let Some(path) = crate::artifact_paths::jsonl_path_from_env("THALOS_SHIPYARD_SELECT_LOG")
        else {
            return Self::default();
        };
        match crate::artifact_paths::open_jsonl_append(&path) {
            Ok(file) => {
                info!(target: "thalos::shipyard", "selection trace → {}", path.display());
                Self {
                    writer: Some(Mutex::new(file)),
                    seq: AtomicU64::new(0),
                }
            }
            Err(err) => {
                warn!(target: "thalos::shipyard", "selection trace disabled (cannot open {}): {err}", path.display());
                Self::default()
            }
        }
    }

    /// True when a sink file is open. Use to skip building expensive fields.
    pub fn enabled(&self) -> bool {
        self.writer.is_some()
    }

    /// Append one `{ "seq": N, <fields...> }` line. No-op when disabled.
    pub fn event(&self, fields: &[(&str, Field<'_>)]) {
        let Some(writer) = &self.writer else {
            return;
        };
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let mut line = String::with_capacity(96);
        let _ = write!(line, "{{\"seq\":{seq}");
        for (key, value) in fields {
            line.push(',');
            write_json_string(&mut line, key);
            line.push(':');
            value.write_json(&mut line);
        }
        line.push_str("}\n");
        if let Ok(mut file) = writer.lock() {
            let _ = file.write_all(line.as_bytes());
        }
    }
}

/// Watch [`EditorState::selected`] and record every transition. This is the
/// authoritative "what is selected now" signal — paired with the per-click
/// events from the picking observers it shows whether a click selected, a
/// deselect fired, or the value changed for some other reason.
pub(super) fn log_selection_changes(
    log: Res<SelectionLog>,
    state: Res<EditorState>,
    mut last: Local<Option<Option<Entity>>>,
) {
    if !log.enabled() {
        return;
    }
    if *last == Some(state.selected) {
        return;
    }
    let previous = last.flatten();
    *last = Some(state.selected);
    log.event(&[
        ("event", "selection_changed".into()),
        ("from", previous.into()),
        ("to", state.selected.into()),
        ("pending", state.pending.is_some().into()),
    ]);
}
