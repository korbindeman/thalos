# INC-NNNN: <short title>

- **Status:** Investigating | Fixed | Recurred (see INC-XXXX)
- **Date:** YYYY-MM-DD (observed) / YYYY-MM-DD (fixed)
- **Severity:** crash / hang / visual / behavioral / data-loss / perf
- **Surface:** which scenario or mode it shows in (`just game hub`, runway boot, map view, …)

## Summary

One paragraph: what was observed, what actually failed, and the fix in plain language.

## Symptoms

- How it manifests (what the screenshot / play session showed)
- Repro steps, as reliable as you have them (`just game <mode>` / `just screenshot <preset>` + env vars)

## Evidence

Where the diagnosis came from — the minimum needed to re-derive the conclusion. Typical Thalos
channels: a `just screenshot` preset PNG, `just preview` output, a targeted `info!(target:
"thalos::…")` / JSONL log the user reproduced, `THALOS_SHADOW_LOG`-style diagnostic dumps, a
chrome trace via `scripts/analyze_trace.py`, a panic backtrace, or the user's screenshot +
description.

```
(paste the decisive log lines / numbers / stack — the "tell")
```

## Hypotheses considered

The candidate causes laid out per the CLAUDE.md bug-fixing loop, and how each was ruled in/out.
This is the section that saves the next agent the differential.

## Root cause

The mechanism, precisely.

## Fix

What changed, and why it addresses the mechanism (structurally where applicable), not the symptom.

## Prevention & recurrence signals

- The standing rule this teaches (add/extend a CLAUDE.md gotcha or spec-doc invariant and link it here)
- The observable "tell" that would identify a recurrence quickly
