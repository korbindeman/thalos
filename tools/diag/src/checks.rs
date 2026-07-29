//! The checks a triage pass runs over a window of the lane.
//!
//! Each check answers one question, states its answer with a denominator, and
//! stays silent when the number is boring. Adding a check here is how the
//! diagnostics system grows a new opinion; adding an event without a check is
//! how it grows noise.

use std::collections::BTreeMap;

use thalos_diagnostics::Stream;

use crate::finding::*;

/// Headline activity for the report header. Always printed, so an empty window
/// reads as "nothing ran" instead of "everything is fine".
#[derive(Debug, Default)]
pub struct Activity {
    pub sessions: usize,
    pub records: usize,
    pub shots: usize,
    pub shots_failed: usize,
    pub oldest_ts: Option<u128>,
    pub newest_ts: Option<u128>,
}

pub fn activity(stream: &Stream) -> Activity {
    let capture = capture_stats(stream);
    Activity {
        sessions: stream.sessions.len(),
        records: stream.records.len(),
        shots: capture.total,
        shots_failed: capture.failed,
        oldest_ts: stream.records.first().map(|record| record.ts_unix_ms),
        newest_ts: stream.newest_ts_unix_ms(),
    }
}

/// Run every check, most severe first.
pub fn run(stream: &Stream) -> Vec<Finding> {
    let mut findings = Vec::new();
    if stream.records.is_empty() {
        findings.push(Finding::new(
            Severity::Watch,
            "empty_window",
            "no diagnostic records in the window — nothing ran, or the lane stopped writing",
            "check artifacts/diagnostics/ exists and a game or capture ran in this period",
        ));
        return findings;
    }

    findings.extend(errors_and_warnings(stream));
    findings.extend(capture_health(stream));
    findings.extend(frame_health(stream));
    findings.extend(memory_health(stream));
    findings.extend(silent_sessions(stream));
    findings.extend(lane_noise(stream));
    findings.sort_by_key(|finding| finding.severity);
    findings
}

// ── errors and warnings ─────────────────────────────────────────────────────

fn errors_and_warnings(stream: &Stream) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut errors: BTreeMap<String, (usize, u128)> = BTreeMap::new();
    let mut warnings: BTreeMap<String, (usize, u128)> = BTreeMap::new();
    for record in &stream.records {
        // Capture failures arrive as WARN on the tool target and have their own
        // checks below; counting them twice would inflate every report.
        if record.target.ends_with("::tool") {
            continue;
        }
        let key = format!("{}·{}", record.subsystem(), record.event());
        if record.is_error() {
            let entry = errors.entry(key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 = entry.1.max(record.ts_unix_ms);
        } else if record.is_warn() {
            let entry = warnings.entry(key).or_insert((0, 0));
            entry.0 += 1;
            entry.1 = entry.1.max(record.ts_unix_ms);
        }
    }

    if !errors.is_empty() {
        let total: usize = errors.values().map(|(count, _)| count).sum();
        let mut finding = Finding::new(
            Severity::Attention,
            "error_events",
            format!(
                "{total} error event{} across {} kind{}",
                plural(total),
                errors.len(),
                plural(errors.len())
            ),
            "runtime.jsonl · filter level=ERROR",
        );
        for (key, (count, last)) in top(&errors) {
            finding = finding.with_detail(format!("{key} ×{count}, last {}", stamp(last)));
        }
        findings.push(finding);
    }

    if !warnings.is_empty() {
        let total: usize = warnings.values().map(|(count, _)| count).sum();
        let mut finding = Finding::new(
            Severity::Watch,
            "warn_events",
            format!(
                "{total} warning event{} across {} kind{}",
                plural(total),
                warnings.len(),
                plural(warnings.len())
            ),
            "runtime.jsonl · filter level=WARN",
        );
        for (key, (count, last)) in top(&warnings) {
            finding = finding.with_detail(format!("{key} ×{count}, last {}", stamp(last)));
        }
        findings.push(finding);
    }
    findings
}

// ── capture lane ────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct CaptureStats {
    total: usize,
    failed: usize,
    abandoned: usize,
    retried: usize,
    booted: usize,
    ok_durations_ms: Vec<f64>,
    errors: BTreeMap<String, usize>,
    boot_reasons: BTreeMap<String, usize>,
    lock_wait_ms: f64,
    lock_queued: usize,
}

fn capture_stats(stream: &Stream) -> CaptureStats {
    let mut stats = CaptureStats::default();
    for record in stream.events("tool_run") {
        if record.str("tool") != Some("capture") {
            continue;
        }
        stats.total += 1;
        match record.str("outcome").unwrap_or_default() {
            "ok" => {
                if let Some(total_ms) = record.f64("total_ms") {
                    stats.ok_durations_ms.push(total_ms);
                }
            }
            "abandoned" => stats.abandoned += 1,
            _ => {
                stats.failed += 1;
                *stats
                    .errors
                    .entry(record.str("error").unwrap_or("unknown").to_owned())
                    .or_default() += 1;
            }
        }
        if record.u64("retry_count").unwrap_or(0) > 0 {
            stats.retried += 1;
        }
        let action = record.str("host_action").unwrap_or_default();
        if !action.is_empty() && action != "reuse" {
            stats.booted += 1;
            *stats.boot_reasons.entry(action.to_owned()).or_default() += 1;
        }
    }
    for record in stream.events("capture_lock") {
        stats.lock_wait_ms += record.f64("wait_ms").unwrap_or_default();
        if record.fields.get("queued").and_then(|q| q.as_bool()) == Some(true) {
            stats.lock_queued += 1;
        }
    }
    stats
}

fn capture_health(stream: &Stream) -> Vec<Finding> {
    let stats = capture_stats(stream);
    let mut findings = Vec::new();
    if stats.total == 0 {
        return findings;
    }
    let total = stats.total;

    if stats.failed >= CAPTURE_FAILURES_ATTENTION {
        let mut finding = Finding::new(
            Severity::Attention,
            "capture_failures",
            format!(
                "{} of {total} capture shot{} failed ({:.0}%)",
                stats.failed,
                plural(total),
                percent(stats.failed, total)
            ),
            "tools.jsonl · event=tool_run outcome=error",
        );
        for (message, count) in top_counts(&stats.errors) {
            finding = finding.with_detail(format!("{message:?} ×{count}"));
        }
        findings.push(finding);
    }

    if stats.abandoned >= CAPTURE_ABANDONED_WATCH {
        findings.push(Finding::new(
            Severity::Watch,
            "capture_abandoned",
            format!(
                "{} of {total} capture shots ended without a result (killed mid-run)",
                stats.abandoned
            ),
            "tools.jsonl · event=tool_run outcome=abandoned",
        ));
    }

    if total >= CAPTURE_RATE_MIN_SHOTS {
        let retry_fraction = stats.retried as f64 / total as f64;
        if retry_fraction >= CAPTURE_RETRY_FRACTION_ATTENTION {
            findings.push(Finding::new(
                Severity::Attention,
                "capture_retries",
                format!(
                    "{} of {total} shots needed a host retry ({:.0}%) — each one is a boot paid twice",
                    stats.retried,
                    retry_fraction * 100.0
                ),
                "tools.jsonl · event=tool_run retry_count>0, read retry_reason",
            ));
        }

        let boot_fraction = stats.booted as f64 / total as f64;
        if boot_fraction >= CAPTURE_BOOT_FRACTION_WATCH {
            let severity = if boot_fraction >= CAPTURE_BOOT_FRACTION_ATTENTION {
                Severity::Attention
            } else {
                Severity::Watch
            };
            let mut finding = Finding::new(
                severity,
                "capture_boot_rate",
                format!(
                    "{} of {total} shots booted a host ({:.0}%) instead of reusing one",
                    stats.booted,
                    boot_fraction * 100.0
                ),
                "tools.jsonl · event=tool_run host_action",
            );
            for (reason, count) in top_counts(&stats.boot_reasons) {
                finding = finding.with_detail(format!("{reason} ×{count}"));
            }
            findings.push(finding);
        }
    }

    if let Some(p95) = percentile(&stats.ok_durations_ms, 0.95)
        && p95 >= CAPTURE_P95_MS_WATCH
    {
        let severity = if p95 >= CAPTURE_P95_MS_ATTENTION {
            Severity::Attention
        } else {
            Severity::Watch
        };
        let p50 = percentile(&stats.ok_durations_ms, 0.50).unwrap_or_default();
        findings.push(
            Finding::new(
                severity,
                "capture_latency",
                format!(
                    "successful shots: p95 {:.0}s, p50 {:.0}s over {} shot{}",
                    p95 / 1000.0,
                    p50 / 1000.0,
                    stats.ok_durations_ms.len(),
                    plural(stats.ok_durations_ms.len())
                ),
                "tools.jsonl · compare phase_host_start_ms vs phase_render_ms",
            )
            .with_detail("a warm reuse is seconds; ~2 min means the shot rebuilt the host"),
        );
    }

    if stats.lock_wait_ms >= CAPTURE_LOCK_WAIT_MS_WATCH {
        findings.push(Finding::new(
            Severity::Watch,
            "capture_lock_contention",
            format!(
                "{:.0} min lost waiting for the machine capture lock across {} queued invocation{}",
                stats.lock_wait_ms / 60_000.0,
                stats.lock_queued,
                plural(stats.lock_queued)
            ),
            "tools.jsonl · event=capture_lock wait_ms",
        ));
    }
    findings
}

// ── frame health ────────────────────────────────────────────────────────────

fn frame_health(stream: &Stream) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut spikes: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    let mut slow: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    for record in &stream.records {
        match record.event() {
            "spike" => {
                let entry = spikes.entry(record.session.as_str()).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 = entry.1.max(record.f64("spike_ms").unwrap_or_default());
            }
            "frame_gauge" => {
                let p95 = record.f64("cpu_ms_p95").unwrap_or_default();
                if p95 > SLOW_FRAME_MS {
                    let entry = slow.entry(record.session.as_str()).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 = entry.1.max(p95);
                }
            }
            _ => {}
        }
    }

    if let Some((session, (count, worst))) = spikes
        .iter()
        .max_by_key(|(_, (count, _))| *count)
        .map(|(session, value)| (*session, *value))
        && count >= SPIKES_WATCH
    {
        findings.push(Finding::new(
            if count >= SPIKES_ATTENTION {
                Severity::Attention
            } else {
                Severity::Watch
            },
            "frame_spikes",
            format!("{count} frame spikes in one session, worst {worst:.0} ms"),
            format!("just perf-report {session} · runtime.jsonl event=spike"),
        ));
    }

    if let Some((session, (count, worst))) = slow
        .iter()
        .max_by_key(|(_, (count, _))| *count)
        .map(|(session, value)| (*session, *value))
        && count >= SLOW_GAUGES_WATCH
    {
        findings.push(Finding::new(
            if count >= SLOW_GAUGES_ATTENTION {
                Severity::Attention
            } else {
                Severity::Watch
            },
            "slow_frames",
            format!(
                "{count} × 2 s windows below 30 fps in one session (worst p95 {worst:.0} ms)",

            ),
            format!("just perf-report {session}"),
        ));
    }
    findings
}

// ── memory ──────────────────────────────────────────────────────────────────

fn memory_health(stream: &Stream) -> Vec<Finding> {
    let mut findings = Vec::new();

    // The tile budget brake: any engagement means a framing may have rendered
    // coarser than it was authored to, which silently weakens visual evidence.
    let mut braked: Option<(&str, f64, u64)> = None;
    for record in stream.events("residency_gauge") {
        let Some(scale) = record.f64("split_scale") else {
            continue;
        };
        if scale < TILE_BRAKE_SCALE {
            let instances = record.u64("instances").unwrap_or(1);
            let worse = braked.is_none_or(|(_, previous, _)| scale < previous);
            if worse {
                braked = Some((record.session.as_str(), scale, instances));
            }
        }
    }
    if let Some((session, scale, instances)) = braked {
        findings.push(
            Finding::new(
                Severity::Attention,
                "tile_budget_brake",
                format!(
                    "the tile memory brake engaged (split scale fell to {scale:.2} with {instances} renderer instance{})",
                    plural(instances as usize)
                ),
                format!("runtime.jsonl session={session} event=residency_gauge"),
            )
            .with_detail("captures taken while braked may show coarser terrain than authored"),
        );
    }

    // Growth across a session, the open question behind the tile OOM: tiles can
    // sit inside their budget while the process still climbs.
    let mut worst: Option<(&str, f64, f64)> = None;
    for (session, records) in stream.by_session() {
        let series: Vec<f64> = records
            .iter()
            .filter(|record| record.event() == "frame_gauge")
            .filter_map(|record| Some(record.f64("tile_mib")? + record.f64("slab_mib")?))
            .collect();
        let (Some(first), Some(peak)) = (
            series.first().copied(),
            series.iter().copied().fold(None, |acc: Option<f64>, value| {
                Some(acc.map_or(value, |current: f64| current.max(value)))
            }),
        ) else {
            continue;
        };
        let growth = peak - first;
        if growth >= MEMORY_GROWTH_MIB_ATTENTION
            && worst.is_none_or(|(_, previous, _)| growth > previous)
        {
            worst = Some((session, growth, peak));
        }
    }
    if let Some((session, growth, peak)) = worst {
        findings.push(
            Finding::new(
                Severity::Attention,
                "memory_growth",
                format!(
                    "tile + mesh-slab memory grew {growth:.0} MiB within one session (peak {peak:.0} MiB)"
                ),
                format!("just perf-report {session} · runtime.jsonl event=frame_gauge"),
            )
            .with_detail(
                "the open accumulation question behind the tile OOM — check whether tile_mib \
                 plateaus while slab_mib climbs",
            ),
        );
    }
    findings
}

// ── lane health ─────────────────────────────────────────────────────────────

fn silent_sessions(stream: &Stream) -> Vec<Finding> {
    // A tool invocation that records nothing is correct — `capture status` and
    // `--help` open the lane and exit. Only a process that declared itself a
    // runtime and then said nothing is suspicious; `unknown` (a session written
    // before roles existed) is not evidence of anything.
    let silent: Vec<&String> = stream
        .sessions
        .iter()
        .filter(|session| stream.role(session) == "runtime")
        .filter(|session| stream.session(session).next().is_none())
        .collect();
    if silent.is_empty() {
        return Vec::new();
    }
    let mut finding = Finding::new(
        Severity::Watch,
        "silent_sessions",
        format!(
            "{} game-shaped process{} opened the lane and recorded nothing",
            silent.len(),
            if silent.len() == 1 { "" } else { "es" }
        ),
        "runtime.jsonl · a process that died during boot looks exactly like this",
    );
    for session in silent.iter().take(2) {
        finding = finding.with_detail(format!("session {session}"));
    }
    vec![finding]
}

/// Signal-to-noise maintenance: one event that dominates the lane buries
/// everything else and makes every future read more expensive. This is the
/// check that keeps the system from rotting into a log file.
fn lane_noise(stream: &Stream) -> Vec<Finding> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for record in &stream.records {
        *counts
            .entry(format!("{}·{}", record.subsystem(), record.event()))
            .or_default() += 1;
    }
    let Some((event, count)) = counts
        .iter()
        .max_by_key(|(_, count)| **count)
        .map(|(event, count)| (event.clone(), *count))
    else {
        return Vec::new();
    };
    let share = count as f64 / stream.records.len() as f64;
    if count < NOISE_MIN_RECORDS || share < NOISE_SHARE_WATCH {
        return Vec::new();
    }
    vec![
        Finding::new(
            Severity::Watch,
            "lane_noise",
            format!(
                "one event is {:.0}% of the lane: {event} ×{count}",
                share * 100.0
            ),
            "either it answers a question worth its volume, or it should be sampled, demoted to a THALOS_* opt-in, or deleted",
        )
        .with_detail(format!(
            "{} records total in the window across {} session(s)",
            stream.records.len(),
            stream.sessions.len()
        )),
    ]
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn top(map: &BTreeMap<String, (usize, u128)>) -> Vec<(&str, (usize, u128))> {
    let mut entries: Vec<_> = map
        .iter()
        .map(|(key, value)| (key.as_str(), *value))
        .collect();
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.1.0));
    entries.truncate(MAX_DETAIL_LINES);
    entries
}

fn top_counts(map: &BTreeMap<String, usize>) -> Vec<(&str, usize)> {
    let mut entries: Vec<_> = map
        .iter()
        .map(|(key, value)| (key.as_str(), *value))
        .collect();
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    entries.truncate(MAX_DETAIL_LINES);
    entries
}

fn percent(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64 * 100.0
    }
}

fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

/// Nearest-rank percentile over an unsorted sample.
fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((sorted.len() as f64 * quantile).ceil() as usize).clamp(1, sorted.len());
    Some(sorted[rank - 1])
}

/// `YYYY-MM-DD HH:MMZ` from Unix milliseconds, without a date dependency.
pub fn stamp(unix_ms: u128) -> String {
    let seconds = (unix_ms / 1000) as i64;
    let days = seconds.div_euclid(86_400);
    let time_of_day = seconds.rem_euclid(86_400);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}Z",
        time_of_day / 3600,
        (time_of_day % 3600) / 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use thalos_diagnostics::Record;

    fn record(session: &str, ts: u128, level: &str, target: &str, fields: serde_json::Value) -> Record {
        Record {
            session: session.to_owned(),
            ts_unix_ms: ts,
            pid: 1,
            level: level.to_owned(),
            target: target.to_owned(),
            fields: fields.as_object().expect("object").clone(),
        }
    }

    fn stream(records: Vec<Record>) -> Stream {
        let sessions = {
            let mut seen: Vec<String> = Vec::new();
            for record in &records {
                if !seen.contains(&record.session) {
                    seen.push(record.session.clone());
                }
            }
            seen
        };
        Stream {
            records,
            sources: Vec::new(),
            session_roles: sessions
                .iter()
                .map(|session| (session.clone(), "runtime".to_owned()))
                .collect(),
            sessions,
            skipped_lines: 0,
        }
    }

    fn shot(session: &str, ts: u128, outcome: &str, extra: serde_json::Value) -> Record {
        let mut fields = json!({
            "event": "tool_run",
            "tool": "capture",
            "command": "shot spaceport-aerial",
            "outcome": outcome,
            "total_ms": 4_000,
            "host_action": "reuse",
        });
        for (key, value) in extra.as_object().expect("object") {
            fields[key] = value.clone();
        }
        record(
            session,
            ts,
            if outcome == "ok" { "INFO" } else { "WARN" },
            "thalos::diagnostic::tool",
            fields,
        )
    }

    /// A healthy day must print nothing. This is the check that keeps the tool
    /// worth reading: if a clean window produces findings, every real finding
    /// is buried.
    #[test]
    fn a_healthy_window_produces_no_findings() {
        let mut records = Vec::new();
        for index in 0..10 {
            records.push(shot("s1", 1_000 + index, "ok", json!({})));
            records.push(record(
                "s1",
                1_000 + index,
                "INFO",
                "thalos::diagnostic::perf",
                json!({"event": "frame_gauge", "cpu_ms_p95": 8.0, "tile_mib": 300.0, "slab_mib": 200.0}),
            ));
        }
        assert!(
            run(&stream(records)).is_empty(),
            "a clean window must stay silent"
        );
    }

    /// Every check must be shown to fire on the defect it exists for —
    /// otherwise it is decoration that will pass forever.
    #[test]
    fn each_check_fires_on_its_own_defect() {
        let ids = |records: Vec<Record>| -> Vec<&'static str> {
            run(&stream(records))
                .into_iter()
                .map(|finding| finding.id)
                .collect()
        };

        assert!(
            ids(vec![shot(
                "s",
                1,
                "error",
                json!({"error": "capture launcher exited"})
            )])
            .contains(&"capture_failures")
        );

        let mut retried = Vec::new();
        for index in 0..10 {
            let outcome = if index < 3 { 1 } else { 0 };
            retried.push(shot("s", index, "ok", json!({"retry_count": outcome})));
        }
        assert!(ids(retried).contains(&"capture_retries"));

        let mut booted = Vec::new();
        for index in 0..10 {
            booted.push(shot(
                "s",
                index,
                "ok",
                json!({"host_action": "restart_stale_source"}),
            ));
        }
        let booted = ids(booted);
        assert!(booted.contains(&"capture_boot_rate"));

        let slow: Vec<Record> = (0..10)
            .map(|index| shot("s", index, "ok", json!({"total_ms": 400_000})))
            .collect();
        assert!(ids(slow).contains(&"capture_latency"));

        let brake = vec![record(
            "s",
            1,
            "INFO",
            "thalos::diagnostic::tile_terrain",
            json!({"event": "residency_gauge", "split_scale": 0.6, "instances": 2}),
        )];
        assert!(ids(brake).contains(&"tile_budget_brake"));

        let growth = vec![
            record(
                "s",
                1,
                "INFO",
                "thalos::diagnostic::perf",
                json!({"event": "frame_gauge", "cpu_ms_p95": 8.0, "tile_mib": 100.0, "slab_mib": 100.0}),
            ),
            record(
                "s",
                2,
                "INFO",
                "thalos::diagnostic::perf",
                json!({"event": "frame_gauge", "cpu_ms_p95": 8.0, "tile_mib": 900.0, "slab_mib": 900.0}),
            ),
        ];
        assert!(ids(growth).contains(&"memory_growth"));

        let errors = vec![record(
            "s",
            1,
            "ERROR",
            "thalos::diagnostic::clouds",
            json!({"event": "composite_frame_override"}),
        )];
        assert!(ids(errors).contains(&"error_events"));

        let spikes: Vec<Record> = (0..21)
            .map(|index| {
                record(
                    "s",
                    index,
                    "INFO",
                    "thalos::diagnostic::perf",
                    json!({"event": "spike", "spike_ms": 120.0}),
                )
            })
            .collect();
        assert!(ids(spikes).contains(&"frame_spikes"));

        assert!(ids(Vec::new()).contains(&"empty_window"));
    }

    /// Capture failures are WARN-level on the tool target; counting them as
    /// generic warnings too would double every capture report.
    #[test]
    fn capture_failures_are_not_also_counted_as_warnings() {
        let findings = run(&stream(vec![shot(
            "s",
            1,
            "error",
            json!({"error": "capture timed out"}),
        )]));
        let ids: Vec<_> = findings.iter().map(|finding| finding.id).collect();
        assert!(ids.contains(&"capture_failures"));
        assert!(!ids.contains(&"warn_events"));
    }

    #[test]
    fn timestamps_render_as_utc_minutes() {
        // 2026-07-29T04:00:00Z
        assert_eq!(stamp(1_785_297_600_000), "2026-07-29 04:00Z");
    }
}
