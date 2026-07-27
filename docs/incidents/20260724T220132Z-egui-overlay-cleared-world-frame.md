# INC-20260724T220132Z-egui-overlay-cleared-world-frame: the egui overlay cleared the game to black

- **Status:** Fixed
- **Date:** 2026-07-25 (observed) / 2026-07-25 (fixed)
- **Severity:** visual
- **Surface:** every interactive game view after the F8 manager overlay-camera change

## Summary

The dedicated viewpoint-manager camera attempted to fix egui being attached to
an inactive world camera, but its high-order pass cleared the window after the world rendered.
`ClearColorConfig::Custom(Color::NONE)` means “clear to transparent black”; it
does not mean “do not clear.” The window therefore remained black even before
the F8 manager opened. Changing it to `ClearColorConfig::None` was necessary but
did not restore live output in Thalos's renderer. The final fix removes the
second window camera and attaches egui directly to the canonical `ShipCamera`.

## Symptoms

- The game window appeared but the game frame never became visible.
- The regression began with the dedicated high-order manager camera.
- It occurred without pressing F8 because the overlay camera renders every frame.

## Evidence

Bevy's `ClearColorConfig` contract explicitly distinguishes the operations:

```text
Custom(Color) — clear using the given color
None          — draw on top of anything already in the viewport
```

The manager camera rendered at order 1000 with
`Custom(Color::NONE)`, after the ship/map camera at order 0. That ordering and
clear operation exactly produce the observed full-frame black output.

## Hypotheses considered

- **The game failed to boot or create a world camera.** Unnecessary to explain
  the symptom: the new camera deterministically clears after any world-camera
  result.
- **The egui window itself covered the viewport.** Ruled out because the manager
  defaults closed; the black frame occurred without pressing F8.
- **The overlay's transparent color would preserve prior pixels.** Ruled out by
  Bevy's API contract: a transparent custom clear remains a clear operation.

## Root cause

The implementation confused a transparent clear color with a load-preserving
render pass. Alpha does not preserve the prior camera's target contents.

## Fix

Remove the dedicated manager camera and attach `PrimaryEguiContext` explicitly
to the canonical `ShipCamera`. This avoids both the original implicit
first-camera selection and a second same-window presentation pass. If a future
feature genuinely needs a layered camera, `ClearColorConfig::None` remains
necessary, but it is not needed for this manager.

## Prevention & recurrence signals

- Same-target overlay cameras must use `ClearColorConfig::None`. Transparent
  custom clears are appropriate for standalone transparent image targets, not
  for preserving lower-order window-camera output.
- A whole frame disappearing immediately after adding a higher-order camera is
  the recurrence tell; inspect that camera's clear operation first.
- The invariant is recorded in `docs/development/visual_testing.md`.
