# INC-20260731T012216Z-stale-log-quarantined-capture: six-day-old log line quarantined a healthy capture lane

- **Date:** 2026-07-31 · **Surface:** every `just screenshot` / `just capture` invocation, machine-wide

## Symptom

All captures refused with "capture renderer is quarantined after GPU device loss for
the rest of this OS boot" while the GPU was demonstrably healthy (the game and capture
hosts had rendered all evening). The decisive tell: the recorded fault's
`fault_detail` tail ended in a **successful** capture ("saved …runway_atmosphere.png"),
and the only "Device is lost" text in `visual_capture_server.log` was from
2026-07-25 — six days before the quarantine.

## Root cause

The client classifies resource faults by substring-scanning the host log from a
recorded start offset. When the `LauncherState` json could not be read, the offset
fell back to **0** — and the log file is append-accumulated across boots, so the scan
covered six days of history and matched a July-25 `DeviceLost` line as a fresh fatal
fault. The device-loss policy then blocked the lane until OS reboot.

## Fix

The fallback is now the *current* end of the log (losing at worst this boot's early
text from the scan), never byte 0. A reboot clears the standing quarantine via the
existing uptime check.

## Recurrence signal

A `visual_capture_resource_events.jsonl` record whose `fault_detail` shows a healthy
tail (e.g. a "saved …png" line) or whose matched signature text cannot be found in
the log **after** the recorded launcher offset. If quarantines with stale evidence
reappear, the log window derivation has regressed.
