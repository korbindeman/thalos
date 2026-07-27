# INC-20260724T214426Z-f8-manager-attached-to-inactive-camera: F8 manager rendered into the inactive map camera

- **Status:** Fixed
- **Date:** 2026-07-24 (observed) / 2026-07-24 (fixed)
- **Severity:** behavioral
- **Surface:** interactive game in normal ship/free-camera view

## Summary

Pressing F8 correctly toggled the new viewpoint manager resource, but no window
appeared. `bevy_egui` automatically put its primary context on the first camera
Thalos created. That camera is the map camera, which is inactive during normal
ship view, so the complete manager UI was rendered into a disabled camera. The
first repair used a dedicated overlay camera, but that introduced a second
same-window presentation pass and blacked out the game. The final repair
attaches the context explicitly to the canonical `ShipCamera`.

## Symptoms

- F8 appeared to do nothing in a normal interactive game.
- The input binding and manager compiled cleanly, and there was no visible error.
- Repro: start any normal 3-D scenario in ship view and press F8.

## Evidence

- The input plugin resets and collects `GameInputIntent::save_perspective` in
  ordered `PreUpdate` systems; its existing F8 input test passes. The manager
  consumes that completed intent in `Update`, ruling out the initial scheduling
  hypothesis.
- `ViewpointManagerPlugin` used `EguiPlugin::default()`, whose
  `auto_create_primary_context` behavior attaches to the first added camera.
- `camera::spawn_camera` spawns `MapCamera` before `ShipCamera`; in the default
  view it sets `MapCamera.is_active = false` and `ShipCamera.is_active = true`.

The decisive shape was therefore:

```text
first camera = MapCamera (inactive) + PrimaryEguiContext
active camera = ShipCamera (no EguiContext)
```

## Hypotheses considered

- **The one-frame F8 intent was missed by unordered systems.** Ruled out: input
  collection is chained after enhanced-input application in `PreUpdate`, while
  the manager reads it later in `Update`.
- **The manager plugin was omitted from the interactive app.** Ruled out:
  `AppBuilder` adds it whenever headless screenshot configuration is absent.
- **The egui context rendered through an inactive camera.** Confirmed by the
  default auto-attachment rule and Thalos's deterministic camera spawn order.

## Root cause

The integration relied on `bevy_egui`'s generic “first camera wins” default in
an application where camera creation order does not mean presentation
ownership. Thalos intentionally creates two mutually exclusive world cameras,
and the first one is inactive in the most common view.

## Fix

The viewpoint plugin disables automatic primary-context creation. In
`PostStartup`, after the world cameras exist, it inserts `PrimaryEguiContext`
on the canonical `ShipCamera`. This fixes presentation ownership without adding
another camera to the window render stack.

## Prevention & recurrence signals

- Window-space developer UI must attach explicitly to its intended presentation
  camera; it must not use implicit first-camera attachment. This invariant is recorded in
  `docs/development/visual_testing.md`.
- A UI resource changing state while its pixels are absent, especially only in
  one camera mode, is the recurrence tell. Inspect `PrimaryEguiContext`
  placement and `Camera::is_active` before debugging the UI code itself.
