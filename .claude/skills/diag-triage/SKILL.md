---
name: diag-triage
description: >-
  Daily diagnostics triage for Thalos. Use when the user asks to check/review the
  diagnostics, asks "anything wrong?" / "how is the capture lane doing?", when a
  scheduled routine fires, or after a play session that felt off. Runs `just diag`
  over the runtime + tool lanes, interprets the findings against known mechanisms,
  and turns the ones that survive scrutiny into backlog rows or incidents — while
  keeping the lane's signal-to-noise high by pruning what nobody acts on.
---

# Diagnostics triage

The lane records what the game and the tooling actually did; `just diag` turns a
window of it into a short list of findings. This skill is the judgment between
those findings and the backlog: **most findings should end in "no action, and
here's why"**, and the ones that don't should end in a row precise enough for a
future agent to act on without re-deriving anything.

Two failure modes to avoid, in order of cost:

1. **Filing noise.** A row per twitchy metric makes the backlog unreadable and
   trains everyone to ignore it. A finding is only work when it is a *pattern*
   with a denominator, or a single event with a mechanism.
2. **Silently dropping a real signal.** If you decide not to act, say so in the
   report with the reason. An unexplained finding is not a closed finding.

## Procedure

### 1 · Read the window

```bash
just diag 24
```

Use a longer window (`just diag 168`) when the routine has not run for a while,
or when judging whether something is a trend. `just diag 24 --json` when you want
to compute over the findings rather than read them.

The header is part of the report. `0 records` means *nothing ran*, not *all
clear* — say which. A window with no sessions is a fine outcome to report in one
line.

### 2 · Interpret each finding

Findings carry a stable `id`. What each usually means, and the first move:

| id | Usually means | First move |
|---|---|---|
| `capture_failures` | the lane broke, or the workspace does not build | read the `error` string; if it is `capture launcher exited`, check whether the tree compiles (`just check`) before suspecting the lane |
| `capture_retries` | host dying mid-shot — device loss, OOM, wedged renderer | read `retry_reason`; a repeated reason is an incident, a one-off is a watch |
| `capture_boot_rate` | reuse is not working: agents pay a rebuild per image | check `host_action` mix — `restart_stale_source` all day means someone edits Rust constantly (normal); `restart_incompatible_scene` means preset batching is misordered |
| `capture_latency` | mostly a consequence of the two above | attribute with `phase_host_start_ms` vs `phase_render_ms` before treating it as its own problem |
| `capture_lock_contention` | parallel agents serializing on one GPU | expected with several agents; only act if a single invocation blocked for a long stretch |
| `error_events` / `warn_events` | a subsystem is unhappy | group by `subsystem·event`; a *new* kind matters more than a high count of a known one |
| `frame_spikes` / `slow_frames` | a stutter or a slow scene | `just perf-report <session>` for the timeline before guessing at a cause |
| `memory_growth` | the open accumulation question behind the tile OOM | check whether `tile_mib` plateaus while `slab_mib` climbs — that split is the whole diagnosis (INC-20260725T012104Z) |
| `tile_budget_brake` | terrain rendered coarser than authored | treat any capture taken in that session as suspect evidence, and say so |
| `silent_sessions` | a game-shaped process died during boot | correlate with the console log; tool sessions are already excluded |
| `lane_noise` | one event is drowning the lane | § 4 |
| `empty_window` | nothing ran | report it as such; do not infer health |

**Before filing anything**, check whether it is already known:

```bash
rg -i '<keyword>' docs/backlog.md docs/incidents/
```

A row that already covers it gets the new evidence appended (dates, counts) —
not a second row.

### 3 · Decide, per finding

- **Fix now** — only if it is small, safe, inside this repo's normal scope, and
  you can verify it. Then no row: the commit is the record.
- **File a backlog row** — multi-step, needs the user, or needs a play session.
  State the claim with its numbers, the mechanism if known, and what would close
  it. `BL-<UTC>-<slug>` per CLAUDE.md.
- **Write an incident** — only when a diagnosis was non-obvious and is now
  understood. Never for "we saw a number".
- **No action** — say why in one clause ("two agents captured in parallel; the
  lock did its job").

Never open a row you cannot state as a falsifiable claim. "Captures feel slow"
is not one; "p95 shot time 210 s across 14 shots, 12 of them rebuilding" is.

### 4 · Keep the signal-to-noise high

This is half the job, and the half everyone skips.

- **`lane_noise`**: decide whether that event earns its volume. If it answers a
  live question, leave it and say so. If it does not, propose sampling it,
  demoting it to a `THALOS_*` opt-in mode, or deleting it.
- **A check that fires every day and is dismissed every day** is a wrong
  threshold or a dead check. Tune the constant in `tools/diag/src/finding.rs`
  (with its rationale comment) or remove the check.
- **A finding you cannot explain from the data** is the highest-value outcome of
  a triage pass: it means the lane is missing an instrument. File a row to *add
  the diagnostic*, naming the question it must answer and the field that would
  separate the hypotheses. This is how the system grows.
- **Do not add an event without a check** — an event nobody reads is cost with
  no signal.

### 5 · Report

Keep it to what changed and what you did:

```
diagnostics · 24h · 12 sessions · 31 shots, 0 failed

ATTENTION
  memory_growth — 3.8 GiB within one session (peak 4.6 GiB)
      → filed BL-…-slab-accumulation; the split says slabs, not tiles

WATCH
  capture_boot_rate — 61% of shots rebuilt
      → no action: three agents edited Rust all day, which is what this looks like

nothing else crossed a threshold.
```

Lead with what needs the user. If nothing does, say that in one line — a clean
day is a valid, useful report, and reporting it plainly is what makes the
non-clean days credible.

## Boundaries

- **Do not** launch the game to reproduce something (CLAUDE.md: the user runs
  the game). Headless capture is yours; use it when a finding is visual.
- **Do not** fix another agent's in-flight work you happen to see in the lane.
  Confirm it is outside your change, note it, move on.
- **Do not** let the triage pass turn into the investigation. Triage decides
  what deserves an investigation; the investigation gets its own session.
