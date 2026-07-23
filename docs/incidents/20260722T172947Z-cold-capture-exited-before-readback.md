# INC-20260722T172947Z-cold-capture-exited-before-readback: Cold capture exited before readback

- **Status:** Fixed
- **Date:** 2026-07-22 (observed) / 2026-07-22 (fixed)
- **Severity:** behavioral
- **Surface:** `just screenshot-cold <preset>`

## Summary

An authoritative cold screenshot could finish its fixed post-request frame tail before Bevy's
asynchronous GPU readback completed. App shutdown then closed the screenshot result channel,
correctly invalidating the run and leaving the previous PNG in place. The headless driver now
waits until no `Capturing` entity remains before it begins the exit tail.

## Symptoms

- `just screenshot-cold cloud-runway` rendered and reached the capture request, but exited 3.
- The expected PNG timestamp did not change.
- Repro: run any cold preset whose GPU readback takes longer than the configured 24 tail frames.

## Evidence

The clean process compiled and ran successfully, then the capture-health guard rejected the
run with the decisive log:

```text
bevy_render::view::window::screenshot: Failed to send screenshot: sending on a closed channel
capture INVALID: 1 error(s) logged during this run.
```

`ScreenshotDriver` set `captured = true` when it spawned `Screenshot::image(target)`, then
unconditionally emitted `AppExit::Success` after `tail_frames`; it did not observe Bevy's
`Capturing` marker.

## Hypotheses considered

- **Render-pipeline or shader failure:** ruled out because capture-health reported only the
  closed screenshot channel, after the capture request.
- **Output-path/write failure:** ruled out because the failure occurred while sending the
  readback result, before `save_to_disk` could own it.
- **Readback/exit race:** confirmed by the fixed frame countdown running independently of the
  live `Capturing` entity.

## Root cause

The cold driver treated “screenshot entity spawned” as “readback completed.” Those are separate
events. A fixed 24-frame tail happened to hide the race on faster runs but was not a completion
contract; when it elapsed first, app shutdown dropped the observer channel.

## Fix

Cold capture now waits while any `Capturing` entity exists. Only after Bevy finishes the
asynchronous screenshot does the normal tail countdown begin. Persistent captures are unchanged.

## Prevention & recurrence signals

- Headless shutdown must be gated by asynchronous operation state, never an assumed frame count.
- A recurrence is identified by a closed screenshot channel, an unchanged output timestamp, and
  an otherwise pipeline-clean cold run.
- The cold-lane completion invariant is recorded in `docs/development/visual_testing.md`.
