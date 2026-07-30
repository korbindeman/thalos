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
    findings.extend(shadow_health(stream));
    findings.extend(planet_reflection_health(stream));
    findings.extend(gear_ground_health(stream));
    findings.extend(gpu_health(stream));
    findings.extend(memory_health(stream));
    findings.extend(silent_sessions(stream));
    findings.extend(flow_effect_health(stream));
    findings.extend(lane_noise(stream));
    findings.sort_by_key(|finding| finding.severity);
    findings
}

// ── shadow coherence ────────────────────────────────────────────────────────

fn shadow_health(stream: &Stream) -> Vec<Finding> {
    let samples: Vec<_> = stream
        .events("stability_gauge")
        .filter(|record| record.target.ends_with("::shadow"))
        .collect();
    if samples.is_empty() {
        return Vec::new();
    }
    let mut bad = 0usize;
    let mut worst = 0.0_f64;
    let mut session = "";
    for record in &samples {
        let error = record.f64("origin_frame_error_m").unwrap_or_default();
        if error > SHADOW_ORIGIN_ERROR_M_ATTENTION {
            bad += 1;
            if error > worst {
                worst = error;
                session = &record.session;
            }
        }
    }
    if bad == 0 {
        return Vec::new();
    }
    vec![
        Finding::new(
            Severity::Attention,
            "shadow_frame_desync",
            format!(
                "{bad} of {} shadow gauges used the wrong render origin (worst {worst:.2} m)",
                samples.len()
            ),
            format!("runtime.jsonl session={session} event=stability_gauge"),
        )
        .with_detail(
            "a cell crossing must not move cascade cameras and receivers in different frames",
        ),
    ]
}

// ── orbital reflection of the planet ────────────────────────────────────────

/// A polished hull in orbit reflects the planet through the impostor bake. Two
/// ways that goes wrong produce a plausible image and no error:
///
/// - the bake is **absent** while the planet fills much of the reflected sky,
///   so the hull mirrors a flat body tint (the wiring failure);
/// - the bake is **bound but not varying**, which is how a wrong body-fixed
///   rotation or a blank cube presents (the silent-content failure).
///
/// Both are invisible in a screenshot — a hull reflecting a flat disc and one
/// reflecting continents differ subtly at a glance — which is exactly why the
/// `planet_reflection` event exists and why it needs a reader.
fn planet_reflection_health(stream: &Stream) -> Vec<Finding> {
    let samples: Vec<_> = stream.events("planet_reflection").collect();
    if samples.is_empty() {
        return Vec::new();
    }

    let mut missing = 0usize;
    let mut flat = 0usize;
    let mut session = "";
    let mut worst_spread = f64::INFINITY;
    for record in &samples {
        let disc_frac = record.f64("disc_frac").unwrap_or_default();
        // The planet has to actually be in the reflected sky for its albedo to
        // matter; a framing that points away from it is not a defect.
        if disc_frac < REFLECTION_DISC_FRAC_MIN {
            continue;
        }
        if record.bool("impostor") != Some(true) {
            missing += 1;
            session = &record.session;
            continue;
        }
        let spread = record.f64("albedo_spread").unwrap_or_default();
        if spread < REFLECTION_ALBEDO_SPREAD_FLAT {
            flat += 1;
            if spread < worst_spread {
                worst_spread = spread;
                session = &record.session;
            }
        }
    }

    let mut findings = Vec::new();
    if missing > 0 {
        findings.push(
            Finding::new(
                Severity::Watch,
                "reflection_bake_missing",
                format!(
                    "{missing} of {} reflection paints had the planet filling the sky with no impostor bake bound",
                    samples.len()
                ),
                format!("runtime.jsonl session={session} event=planet_reflection impostor=false"),
            )
            .with_detail(
                "hulls mirror a flat body tint instead of continents; check ImpostorAlbedoRegistry has an entry for this body",
            ),
        );
    }
    if flat > 0 {
        findings.push(
            Finding::new(
                Severity::Attention,
                "reflection_bake_flat",
                format!(
                    "{flat} reflection paints sampled a bound bake that does not vary (spread {worst_spread:.4})"
                ),
                format!("runtime.jsonl session={session} event=planet_reflection albedo_spread"),
            )
            .with_detail(
                "a bound-but-constant bake is a blank cube or a wrong body-fixed rotation, not a bland planet",
            ),
        );
    }
    findings
}

// ── landing gear vs floor backstop ──────────────────────────────────────────

/// A wheeled craft (gear down) being carried by the terrain floor backstop
/// means every suspension ray failed to find ground while the hull sat at the
/// surface — the buried-ray / belly-slide defect (INC-20260729T073116Z). One
/// event can be a touchdown transient; sustained carrying is the defect.
fn gear_ground_health(stream: &Stream) -> Vec<Finding> {
    let carried: Vec<_> = stream
        .events("backstop_intervention")
        .filter(|record| {
            record.u64("gear_down") == Some(1)
                && record.u64("weight_on_wheels") == Some(0)
                && record.u64("destroyed") == Some(0)
        })
        .collect();
    if carried.len() < GEAR_BACKSTOP_CARRY_EVENTS_ATTENTION {
        return Vec::new();
    }
    let mut worst = 0.0_f64;
    let mut session = "";
    for record in &carried {
        let penetration = record.f64("penetration_m").unwrap_or_default();
        if penetration >= worst {
            worst = penetration;
            session = &record.session;
        }
    }
    vec![
        Finding::new(
            Severity::Attention,
            "gear_carried_by_backstop",
            format!(
                "a wheeled craft rode the floor backstop instead of its gear for ~{} s (worst hull penetration {worst:.2} m)",
                carried.len()
            ),
            format!(
                "runtime.jsonl session={session} event=backstop_intervention · compare event=gear_contact"
            ),
        )
        .with_detail(
            "gear down + zero weight-on-wheels while the backstop carries the hull = the \
             suspension rays found no ground; expect a belly slide with no brakes",
        ),
    ]
}

// ── errors and warnings ─────────────────────────────────────────────────────

fn errors_and_warnings(stream: &Stream) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut errors: BTreeMap<String, (usize, u128)> = BTreeMap::new();
    let mut warnings: BTreeMap<String, (usize, u128)> = BTreeMap::new();
    for record in &stream.records {
        // Capture failures arrive as WARN on the tool target and have their own
        // checks below; counting them twice would inflate every report.
        if record.target.ends_with("::tool") || record.target.ends_with("::gpu_health") {
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

// ── whole-card GPU health ──────────────────────────────────────────────────

fn gpu_health(stream: &Stream) -> Vec<Finding> {
    let samples: Vec<_> = stream
        .events("sample")
        .filter(|record| record.target.ends_with("::gpu_health"))
        .collect();
    let failures: Vec<_> = stream
        .events("sample_error")
        .filter(|record| record.target.ends_with("::gpu_health"))
        .collect();
    if samples.is_empty() && failures.is_empty() {
        return Vec::new();
    }

    let mut findings = Vec::new();
    if let Some(failure) = failures.last() {
        let prior = samples
            .iter()
            .rev()
            .find(|sample| sample.session == failure.session);
        let mut finding = Finding::new(
            Severity::Attention,
            "gpu_adapter_lost",
            format!(
                "NVIDIA whole-card telemetry failed in {} runtime session{}",
                failures
                    .iter()
                    .map(|record| record.session.as_str())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                plural(
                    failures
                        .iter()
                        .map(|record| record.session.as_str())
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                )
            ),
            format!(
                "runtime.jsonl session={} event=sample_error",
                failure.session
            ),
        )
        .with_detail(
            failure
                .str("error")
                .unwrap_or("NVML stopped seeing the adapter"),
        );
        if let Some(sample) = prior {
            finding = finding.with_detail(gpu_sample_summary(sample));
        }
        findings.push(finding);
    }

    if let Some(worst) = samples.iter().max_by(|a, b| {
        a.f64("memory_used_frac")
            .unwrap_or_default()
            .total_cmp(&b.f64("memory_used_frac").unwrap_or_default())
    }) {
        let used_frac = worst.f64("memory_used_frac").unwrap_or_default();
        if used_frac >= GPU_MEMORY_USED_FRAC_ATTENTION {
            findings.push(
                Finding::new(
                    Severity::Attention,
                    "gpu_memory_pressure",
                    format!(
                        "whole-card VRAM reached {:.1}% ({:.0} of {:.0} MiB)",
                        used_frac * 100.0,
                        worst.f64("memory_used_mib").unwrap_or_default(),
                        worst.f64("memory_total_mib").unwrap_or_default()
                    ),
                    format!(
                        "runtime.jsonl session={} event=sample · reduce resident GPU resources",
                        worst.session
                    ),
                )
                .with_detail(gpu_sample_summary(worst)),
            );
        }
    }

    if let Some(hottest) = samples.iter().max_by(|a, b| {
        a.f64("temperature_c")
            .unwrap_or_default()
            .total_cmp(&b.f64("temperature_c").unwrap_or_default())
    }) {
        let temperature_c = hottest.f64("temperature_c").unwrap_or_default();
        if temperature_c >= GPU_TEMPERATURE_C_ATTENTION {
            findings.push(
                Finding::new(
                    Severity::Attention,
                    "gpu_thermal_pressure",
                    format!(
                        "GPU temperature reached {temperature_c:.0} °C across {} whole-card sample{}",
                        samples.len(),
                        plural(samples.len())
                    ),
                    format!(
                        "runtime.jsonl session={} event=sample · inspect cooling and clocks",
                        hottest.session
                    ),
                )
                .with_detail(gpu_sample_summary(hottest)),
            );
        }
    }

    let mut thermal_by_session: BTreeMap<&str, Vec<_>> = BTreeMap::new();
    for sample in &samples {
        if sample.u64("clock_throttle_reasons").unwrap_or_default()
            & (GPU_SW_THERMAL_SLOWDOWN_MASK | GPU_HW_THERMAL_SLOWDOWN_MASK)
            != 0
        {
            thermal_by_session
                .entry(sample.session.as_str())
                .or_default()
                .push(*sample);
        }
    }
    if let Some((session, throttled)) = thermal_by_session
        .into_iter()
        .filter(|(_, records)| records.len() >= GPU_THERMAL_THROTTLE_SAMPLES_ATTENTION)
        .max_by_key(|(_, records)| records.len())
    {
        let session_samples = samples
            .iter()
            .filter(|sample| sample.session == session)
            .count();
        let sw_thermal = throttled
            .iter()
            .filter(|sample| {
                sample.u64("clock_throttle_reasons").unwrap_or_default()
                    & GPU_SW_THERMAL_SLOWDOWN_MASK
                    != 0
            })
            .count();
        let hw_thermal = throttled
            .iter()
            .filter(|sample| {
                sample.u64("clock_throttle_reasons").unwrap_or_default()
                    & GPU_HW_THERMAL_SLOWDOWN_MASK
                    != 0
            })
            .count();
        let hw_slowdown = throttled
            .iter()
            .filter(|sample| {
                sample.u64("clock_throttle_reasons").unwrap_or_default() & GPU_HW_SLOWDOWN_MASK != 0
            })
            .count();
        let hottest = throttled
            .iter()
            .max_by(|a, b| {
                a.f64("temperature_c")
                    .unwrap_or_default()
                    .total_cmp(&b.f64("temperature_c").unwrap_or_default())
            })
            .copied()
            .expect("threshold guarantees a thermal sample");
        findings.push(
            Finding::new(
                Severity::Attention,
                "gpu_thermal_throttle",
                format!(
                    "NVIDIA asserted thermal slowdown in {}/{} whole-card samples ({:.1}%)",
                    throttled.len(),
                    session_samples,
                    throttled.len() as f64 / session_samples as f64 * 100.0
                ),
                format!(
                    "runtime.jsonl session={session} event=sample · inspect GPU/memory cooling and driver"
                ),
            )
            .with_detail(format!(
                "NVML reason counts: software thermal {sw_thermal}, hardware thermal {hw_thermal}, hardware slowdown {hw_slowdown}"
            ))
            .with_detail(gpu_sample_summary(hottest)),
        );
    }

    findings
}

fn gpu_sample_summary(record: &thalos_diagnostics::Record) -> String {
    format!(
        "card sample: {:.0}/{:.0} MiB, {:.0} °C, {:.0}/{:.0} W, {:.0}% GPU, {} MHz, throttle mask 0x{:x}",
        record.f64("memory_used_mib").unwrap_or_default(),
        record.f64("memory_total_mib").unwrap_or_default(),
        record.f64("temperature_c").unwrap_or_default(),
        record.f64("power_w").unwrap_or_default(),
        record.f64("power_limit_w").unwrap_or_default(),
        record.f64("gpu_util_frac").unwrap_or_default() * 100.0,
        record.u64("graphics_clock_mhz").unwrap_or_default(),
        record.u64("clock_throttle_reasons").unwrap_or_default(),
    )
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
    /// One representative payload per failure reason. The reason groups the
    /// cause; this says which crate failed to compile or what the host panicked
    /// on, so a triage pass can act without opening the log.
    error_examples: BTreeMap<String, BTreeMap<String, usize>>,
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
        // A game-owned renderer lease is an intentional workstation boundary,
        // not a broken capture lane. The host exits before Bevy/wgpu starts;
        // counting that refusal as a failure would make normal user play an
        // ATTENTION finding and teach agents to work around the safety gate.
        if record.str("launcher_exit_kind") == Some("renderer busy") {
            continue;
        }
        match record.str("outcome").unwrap_or_default() {
            "ok" => {
                if let Some(total_ms) = record.f64("total_ms") {
                    stats.ok_durations_ms.push(total_ms);
                }
            }
            "abandoned" => stats.abandoned += 1,
            _ => {
                stats.failed += 1;
                let error = record.str("error").unwrap_or("unknown").to_owned();
                *stats.errors.entry(error.clone()).or_default() += 1;
                if let Some(detail) = record.str("launcher_exit_detail") {
                    *stats
                        .error_examples
                        .entry(error)
                        .or_default()
                        .entry(detail.to_owned())
                        .or_default() += 1;
                }
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
            // The reason names the cause; this names the thing that caused it.
            if let Some((example, _)) = stats
                .error_examples
                .get(message)
                .and_then(|examples| top_counts(examples).into_iter().next())
            {
                finding = finding.with_detail(format!("    e.g. {example}"));
            }
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
            format!("{count} × 2 s windows below 30 fps in one session (worst p95 {worst:.0} ms)",),
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
    // sit inside their budget while the process still climbs. Prefer whole-
    // process RSS where the session carries it (frame_gauge `rss_mib`, added
    // 2026-07-29 after a host died at 8.1 GiB with every GPU gauge at ~2 GiB);
    // older sessions fall back to the GPU-side tile + slab sum.
    let mut worst: Option<(&str, f64, f64, bool)> = None;
    for (session, records) in stream.by_session() {
        let gauges: Vec<_> = records
            .iter()
            .filter(|record| record.event() == "frame_gauge")
            .collect();
        let rss: Vec<f64> = gauges
            .iter()
            .filter_map(|record| record.f64("rss_mib").filter(|v| *v > 0.0))
            .collect();
        let (series, is_rss) = if rss.is_empty() {
            let gpu: Vec<f64> = gauges
                .iter()
                .filter_map(|record| Some(record.f64("tile_mib")? + record.f64("slab_mib")?))
                .collect();
            (gpu, false)
        } else {
            (rss, true)
        };
        let (Some(first), Some(peak)) = (
            series.first().copied(),
            series
                .iter()
                .copied()
                .fold(None, |acc: Option<f64>, value| {
                    Some(acc.map_or(value, |current: f64| current.max(value)))
                }),
        ) else {
            continue;
        };
        let growth = peak - first;
        if growth >= MEMORY_GROWTH_MIB_ATTENTION
            && worst.is_none_or(|(_, previous, _, _)| growth > previous)
        {
            worst = Some((session, growth, peak, is_rss));
        }
    }
    if let Some((session, growth, peak, is_rss)) = worst {
        let (what, detail) = if is_rss {
            (
                "process RSS",
                "read rss_mib against mesh_cpu_mib / image_cpu_mib / tile_mib / slab_mib in the \
                 same gauges — the unattributed remainder is the lead",
            )
        } else {
            (
                "tile + mesh-slab memory",
                "the open accumulation question behind the tile OOM — check whether tile_mib \
                 plateaus while slab_mib climbs",
            )
        };
        findings.push(
            Finding::new(
                Severity::Attention,
                "memory_growth",
                format!("{what} grew {growth:.0} MiB within one session (peak {peak:.0} MiB)"),
                format!("just perf-report {session} · runtime.jsonl event=frame_gauge"),
            )
            .with_detail(detail),
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

    fn record(
        session: &str,
        ts: u128,
        level: &str,
        target: &str,
        fields: serde_json::Value,
    ) -> Record {
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

    fn reflection(session: &str, impostor: bool, disc_frac: f64, spread: f64) -> Record {
        record(
            session,
            1_000,
            "INFO",
            "thalos::diagnostic::sky",
            json!({
                "event": "planet_reflection",
                "impostor": impostor,
                "body_id": 2,
                "disc_texels": (disc_frac * 393_216.0) as u64,
                "disc_frac": disc_frac,
                "albedo_mean_r": 0.05,
                "albedo_mean_g": 0.105,
                "albedo_mean_b": 0.067,
                "albedo_spread": spread,
                "planet_ang_deg": 70.2,
                "surface_blend": 0.5,
            }),
        )
    }

    /// The measured healthy case: Thalos from a 200 km orbit, impostor bound,
    /// spread 0.046. Must stay silent, or the check is noise on every session
    /// that works.
    #[test]
    fn a_working_orbital_reflection_is_silent() {
        let findings = planet_reflection_health(&stream(vec![reflection("s", true, 0.36, 0.046)]));
        assert!(findings.is_empty(), "{findings:?}");
    }

    /// The wiring failure: planet fills the reflected sky, no bake bound.
    #[test]
    fn a_missing_bake_is_reported() {
        let findings = planet_reflection_health(&stream(vec![reflection("s", false, 0.36, 0.0)]));
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].id, "reflection_bake_missing");
    }

    /// The silent-content failure: bake bound but constant — a blank cube or a
    /// wrong body-fixed rotation. This is the one a screenshot cannot show.
    #[test]
    fn a_bound_but_flat_bake_is_reported() {
        let findings = planet_reflection_health(&stream(vec![reflection("s", true, 0.36, 0.0004)]));
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].id, "reflection_bake_flat");
    }

    /// A framing pointed away from the planet is not a defect, whatever the
    /// bake state — otherwise every surface capture would fire this.
    #[test]
    fn a_planet_outside_the_reflected_sky_is_not_a_defect() {
        let findings = planet_reflection_health(&stream(vec![
            reflection("s", false, 0.001, 0.0),
            reflection("s", true, 0.0, 0.0),
        ]));
        assert!(findings.is_empty(), "{findings:?}");
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
            records.push(record(
                "s1",
                1_000 + index,
                "INFO",
                "thalos::diagnostic::shadow",
                json!({"event": "stability_gauge", "origin_frame_error_m": 0.0}),
            ));
        }
        assert!(
            run(&stream(records)).is_empty(),
            "a clean window must stay silent"
        );
    }

    /// A launcher exit is only useful if the window says *why*. The 2026-07-29
    /// triage saw 13 identical `capture launcher exited` strings and could not
    /// tell a build failure from a dead host (BL-20260729T070928Z).
    #[test]
    fn capture_failures_report_the_cause_and_its_payload() {
        let details = |records: Vec<Record>| -> Vec<String> {
            run(&stream(records))
                .into_iter()
                .filter(|finding| finding.id == "capture_failures")
                .flat_map(|finding| finding.detail)
                .collect()
        };

        let reported = details(vec![
            shot(
                "s",
                1,
                "error",
                json!({
                    "error": "capture launcher exited: workspace build failure",
                    "launcher_exit_kind": "workspace build failure",
                    "launcher_exit_detail":
                        "error: could not compile `thalos_body_render` (lib) · error[E0609]",
                }),
            ),
            shot(
                "s",
                2,
                "error",
                json!({
                    "error": "capture launcher exited: capture host panic",
                    "launcher_exit_kind": "capture host panic",
                    "launcher_exit_detail":
                        "panicked at wgpu_core.rs:2253:18 · Error in Buffer::get_mapped_range",
                }),
            ),
        ]);

        let joined = reported.join("\n");
        assert!(
            joined.contains("workspace build failure") && joined.contains("capture host panic"),
            "the two causes must not collapse into one bucket: {joined}"
        );
        assert!(
            joined.contains("thalos_body_render") && joined.contains("get_mapped_range"),
            "each cause must carry the payload that makes it actionable: {joined}"
        );
    }

    #[test]
    fn renderer_busy_is_an_expected_refusal_not_a_capture_failure() {
        let findings = run(&stream(vec![shot(
            "s",
            1,
            "error",
            json!({
                "error": "capture launcher exited: renderer busy",
                "launcher_exit_kind": "renderer busy",
                "launcher_exit_detail":
                    "GPU renderer lease unavailable: pid 42 owns the interactive game",
            }),
        )]));
        assert!(
            findings
                .iter()
                .all(|finding| finding.id != "capture_failures"),
            "playing the game must not make diagnostics report a broken capture lane"
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

        let shadow_desync = vec![record(
            "s",
            1,
            "INFO",
            "thalos::diagnostic::shadow",
            json!({"event": "stability_gauge", "origin_frame_error_m": 1000.0}),
        )];
        assert!(ids(shadow_desync).contains(&"shadow_frame_desync"));

        let gpu_loss = vec![
            record(
                "s",
                1,
                "INFO",
                "thalos::diagnostic::gpu_health",
                json!({
                    "event": "sample",
                    "memory_used_mib": 4096.0,
                    "memory_total_mib": 12288.0,
                    "memory_used_frac": 0.333,
                    "temperature_c": 70.0,
                    "power_w": 180.0,
                    "power_limit_w": 285.0,
                    "gpu_util_frac": 0.8,
                }),
            ),
            record(
                "s",
                2,
                "ERROR",
                "thalos::diagnostic::gpu_health",
                json!({"event": "sample_error", "error": "NVML error 15: GPU is lost"}),
            ),
        ];
        assert!(ids(gpu_loss).contains(&"gpu_adapter_lost"));

        let gpu_pressure = vec![record(
            "s",
            1,
            "INFO",
            "thalos::diagnostic::gpu_health",
            json!({
                "event": "sample",
                "memory_used_mib": 11800.0,
                "memory_total_mib": 12288.0,
                "memory_used_frac": 0.9603,
                "temperature_c": 90.0,
                "power_w": 270.0,
                "power_limit_w": 285.0,
                "gpu_util_frac": 1.0,
            }),
        )];
        let pressure_ids = ids(gpu_pressure);
        assert!(pressure_ids.contains(&"gpu_memory_pressure"));
        assert!(pressure_ids.contains(&"gpu_thermal_pressure"));

        let gpu_thermal_throttle = (1..=3)
            .map(|timestamp| {
                record(
                    "s",
                    timestamp,
                    "INFO",
                    "thalos::diagnostic::gpu_health",
                    json!({
                        "event": "sample",
                        "memory_used_mib": 5800.0,
                        "memory_total_mib": 12288.0,
                        "memory_used_frac": 0.472,
                        "temperature_c": 79.0,
                        "power_w": 200.0,
                        "power_limit_w": 285.0,
                        "gpu_util_frac": 0.99,
                        "graphics_clock_mhz": 2505,
                        "clock_throttle_reasons": 32,
                    }),
                )
            })
            .collect();
        assert!(ids(gpu_thermal_throttle).contains(&"gpu_thermal_throttle"));

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

        // A session carrying `rss_mib` is judged on whole-process RSS — the
        // massif-aerial OOM shape: GPU gauges flat, RSS runaway.
        let rss_growth = vec![
            record(
                "s",
                1,
                "INFO",
                "thalos::diagnostic::perf",
                json!({"event": "frame_gauge", "cpu_ms_p95": 8.0, "tile_mib": 100.0, "slab_mib": 100.0, "rss_mib": 2000.0}),
            ),
            record(
                "s",
                2,
                "INFO",
                "thalos::diagnostic::perf",
                json!({"event": "frame_gauge", "cpu_ms_p95": 8.0, "tile_mib": 100.0, "slab_mib": 100.0, "rss_mib": 8000.0}),
            ),
        ];
        let rss_findings = run(&stream(rss_growth));
        let rss_finding = rss_findings
            .iter()
            .find(|finding| finding.id == "memory_growth")
            .expect("rss growth fires memory_growth");
        assert!(
            rss_finding.headline.contains("process RSS"),
            "rss_mib series should be preferred over tile+slab: {}",
            rss_finding.headline
        );
        // And flat RSS stays silent even while the GPU-side sum would have
        // fired — RSS is the truth once it exists.
        let rss_flat = vec![
            record(
                "s",
                1,
                "INFO",
                "thalos::diagnostic::perf",
                json!({"event": "frame_gauge", "cpu_ms_p95": 8.0, "tile_mib": 100.0, "slab_mib": 100.0, "rss_mib": 3000.0}),
            ),
            record(
                "s",
                2,
                "INFO",
                "thalos::diagnostic::perf",
                json!({"event": "frame_gauge", "cpu_ms_p95": 8.0, "tile_mib": 900.0, "slab_mib": 900.0, "rss_mib": 3100.0}),
            ),
        ];
        assert!(!ids(rss_flat).contains(&"memory_growth"));

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

    fn backstop_event(ts: u128, gear_down: u64, weight_on_wheels: u64) -> Record {
        record(
            "s",
            ts,
            "INFO",
            "thalos::diagnostic::local_physics",
            json!({
                "event": "backstop_intervention",
                "penetration_m": 0.62,
                "excess_m": 0.12,
                "gear_down": gear_down,
                "weight_on_wheels": weight_on_wheels,
                "destroyed": 0,
            }),
        )
    }

    /// The buried-ray defect signature: a gear-down craft carried by the
    /// backstop with no weight on wheels, sustained past one throttle tick
    /// (INC-20260729T073116Z — the belly-slide with no brakes).
    #[test]
    fn gear_carried_by_backstop_fires_on_sustained_carry() {
        let findings = run(&stream(vec![
            backstop_event(1_000, 1, 0),
            backstop_event(2_000, 1, 0),
        ]));
        let ids: Vec<_> = findings.iter().map(|finding| finding.id).collect();
        assert!(ids.contains(&"gear_carried_by_backstop"));
    }

    /// Silent when the wheels are carrying (backstop merely deep-caught a
    /// transient), when the craft is gearless/gear-up (hull rest is the
    /// intended contact), or when the carry lasted a single tick.
    #[test]
    fn gear_backstop_check_stays_silent_off_signature() {
        // Wheels carrying alongside the backstop.
        let wheels_loaded = stream(vec![backstop_event(1_000, 1, 1), backstop_event(2_000, 1, 1)]);
        assert!(!run(&wheels_loaded)
            .iter()
            .any(|finding| finding.id == "gear_carried_by_backstop"));
        // Gear up: hull-on-backstop is the crash/belly path, not this defect.
        let gear_up = stream(vec![backstop_event(1_000, 0, 0), backstop_event(2_000, 0, 0)]);
        assert!(!run(&gear_up)
            .iter()
            .any(|finding| finding.id == "gear_carried_by_backstop"));
        // One tick = a touchdown transient.
        let transient = stream(vec![backstop_event(1_000, 1, 0)]);
        assert!(!run(&transient)
            .iter()
            .any(|finding| finding.id == "gear_carried_by_backstop"));
    }
}

/// Vehicle flow effects fitted to the wrong body.
///
/// `reentry_shell_lit` publishes the craft bounds the shell actually resolved.
/// This is the one thing a screenshot cannot tell you apart from a physics bug:
/// both look like "the glow is in the wrong place". Every wrong-looking shell so
/// far has been a bounds-resolution failure — first a bounding sphere standing in
/// for an elongated hull, then a sweep that measured the effects' own proxy meshes
/// and resolved a small cube on a long vehicle.
fn flow_effect_health(stream: &Stream) -> Vec<Finding> {
    let mut findings = Vec::new();
    let Some(record) = stream
        .records
        .iter()
        .rev()
        .find(|r| r.event() == "reentry_shell_lit")
    else {
        return findings;
    };

    let meshes = record.u64("measured_mesh_count").unwrap_or(0);
    if meshes < FLOW_BOUNDS_MESH_MIN {
        findings.push(
            Finding::new(
                Severity::Attention,
                "flow_bounds_unmeasured",
                format!(
                    "the craft-bounds sweep measured {meshes} mesh(es) — every attached flow effect is sized from a default body"
                ),
                "runtime.jsonl · event=reentry_shell_lit, read measured_mesh_count and body_half_*_m;                  check the sweep in rendering::flow walks the PlayerShip descendants and that FlowProxyMesh is not over-applied",
            )
            .with_detail(format!(
                "    resolved half-extents {:.2} x {:.2} x {:.2} m",
                record.f64("body_half_x_m").unwrap_or(0.0),
                record.f64("body_half_y_m").unwrap_or(0.0),
                record.f64("body_half_z_m").unwrap_or(0.0),
            )),
        );
        return findings;
    }

    let axes = [
        record.f64("body_half_x_m").unwrap_or(0.0),
        record.f64("body_half_y_m").unwrap_or(0.0),
        record.f64("body_half_z_m").unwrap_or(0.0),
    ];
    let longest = axes.iter().cloned().fold(0.0f64, f64::max);
    let shortest = axes.iter().cloned().fold(f64::INFINITY, f64::min);
    if shortest > 0.0 && longest / shortest < FLOW_BOUNDS_CUBIC_RATIO {
        findings.push(
            Finding::new(
                Severity::Watch,
                "flow_bounds_cubic",
                format!(
                    "the measured craft box is near-cubic ({longest:.2} m vs {shortest:.2} m) — real vehicles are elongated, so the sweep likely missed the hull"
                ),
                "runtime.jsonl · event=reentry_shell_lit, compare body_half_*_m against the craft you can see in a capture",
            )
            .with_detail(format!("    measured from {meshes} descendant mesh(es)")),
        );
    }
    findings
}
