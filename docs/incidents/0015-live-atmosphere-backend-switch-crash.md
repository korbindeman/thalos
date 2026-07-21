# INC-0015: Live atmosphere-backend switching escaped the isolated comparison boundary

- **Status:** Fixed
- **Date:** 2026-07-21 (observed) / 2026-07-21 (fixed)
- **Severity:** crash
- **Surface:** Settings → Graphics → Legacy custom atmosphere (debug)

## Summary

Changing the legacy-atmosphere checkbox during a flight crashed the game. The
checkbox exposed a diagnostic renderer choice as an autosaved gameplay setting,
even though Bevy raymarching was already the sole canonical atmosphere and the
accepted comparison architecture requires every legacy/canonical variant to
run in a fresh headless process. The invalid live path was deleted; gameplay
now has one atmosphere backend and the retained `BodySky` path is capture-only.

## Symptoms

- Switching atmosphere modes from Settings → Graphics terminated the running
  game instead of changing the sky.
- Starting directly in either backend for a headless comparison remained a
  supported use case; the failure was the in-process transition.
- The choice was persisted in `user/settings.ron`, so a diagnostic selection
  could also leak into later normal sessions.

## Evidence

- `GraphicsSettings::legacy_body_sky` was written live by the settings menu and
  autosaved with normal user preferences.
- A change crossed several renderer-global contracts in one frame:
  `sync_stock_atmosphere` spawned/despawned the Bevy `Atmosphere` proxy,
  `suppress_body_sky_for_stock_atmosphere` changed the fullscreen atmosphere
  owner, cloud lighting changed between Bevy LUTs and analytic fallback, and
  the UDLOD terrain view layout gained/lost its `ATMOSPHERE` bindings.
- ADR-20260721T032343Z already rejected a permanent user-facing choice between
  the two renderers. ADR-20260721T032344Z separately rejected sequential
  variants inside one process because global render resources, caches, and
  temporal histories leak across the transition.
- The canonical `atmosphere` comparison axis already supplies the required A/B
  as isolated child processes without touching persisted settings.
- A deterministic live-switch headless probe could not reach the atmosphere
  transition in the then-dirty worktree because an unrelated photo-mode query
  raised its own startup `B0001`; no lower-level panic signature is claimed
  here. The architectural violation and its complete call surface were visible
  directly in the backend-selection code.

## Hypotheses considered

- **The settings checkbox or RON autosave itself panicked.** The writer only
  assigned a boolean and serialized the aggregate settings; neither operation
  had a fallible unwrap on this path.
- **The proxy's ECS queries aliased `Atmosphere`.** A focused system-access test
  with legacy-body and proxy archetypes initialized successfully, ruling out a
  static Bevy `B0001` in `sync_stock_atmosphere`.
- **The live transition was a supported rendering operation with one local
  defect.** Rejected by the accepted renderer and visual-comparison decisions:
  legacy selection was explicitly diagnostic, and the supported comparison
  boundary is process startup, not runtime mutation.
- **A capture-only backend selector had leaked into normal gameplay.** Confirmed
  by the persisted graphics field, settings-menu control, and normal render
  systems reading that field every frame.

## Root cause

Backend selection had two authorities: a process-scoped screenshot override
for controlled comparisons and an autosaved live graphics preference. The
second authority turned a retained diagnostic implementation into a parallel
production path and asked renderer-global resources to change identity inside
a running frame. That operation was outside the supported architecture and
unnecessary because isolated comparisons already owned the diagnostic use case.

## Fix

- Removed `legacy_body_sky` from `GraphicsSettings`, its Settings → Graphics
  checkbox, and every normal render-system read.
- Normal gameplay now unconditionally selects Bevy raymarching. Cloud LUT
  coupling follows the same canonical default.
- Kept `THALOS_SCREENSHOT_ATMOSPHERE` and `just compare … atmosphere` as the one
  diagnostic selection path; each variant starts in a fresh process.
- Added a migration test proving an existing RON file containing the removed
  key still deserializes to canonical defaults.

## Prevention & recurrence signals

- Capture-only/debug renderer variants must be selected through typed headless
  overrides, never through persisted user settings.
- Do not add a live atmosphere-backend toggle. If a future comparison needs a
  new variant, extend the `atmosphere` axis and keep process isolation.
- A new `GraphicsSettings` field that changes render-graph ownership, view bind
  layouts, or fullscreen-pass identity is a recurrence signal; require an ADR
  before exposing it live.
- See [atmosphere.md](../atmosphere.md), [visual_testing.md](../visual_testing.md),
  ADR-20260721T032343Z, and ADR-20260721T032344Z.
