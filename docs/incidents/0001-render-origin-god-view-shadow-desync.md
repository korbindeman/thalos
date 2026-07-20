# INC-0001: God-view shadows vanished — `RenderOrigin` tracked the focus target, not the camera

- **Status:** Fixed
- **Date:** 2026-07-05 (observed + fixed)
- **Severity:** visual
- **Surface:** space-center hub / base editor god-views (`just game hub`, `just screenshot hub`)

## Summary

In the hub and base-editor god-views, nothing over the base cast shadows (trees, buildings).
The sun-shadow cascade cameras — plain `Camera3d`s living *outside* big_space, placed via
`player_render = player_inertial − RenderOrigin.position` — were being positioned ~1.79 × 10⁹ m
from the casters, because `RenderOrigin` was derived from `CameraFocus.target` (still the
orbital placeholder craft, or the star → `DVec3::ZERO`) while the god-view drove the
`ShipCamera` directly. Fix: in sim-frozen contexts, derive `RenderOrigin` from the
`ShipCamera`'s heliocentric pose so it always mirrors the big_space `FloatingOrigin`.

## Symptoms

- Hub / base-editor god-view: base fully lit, zero shadows from trees or buildings; flight
  views unaffected.
- Repro: `just screenshot hub` (agent-runnable, read the PNG).

## Evidence

`just screenshot hub` with `THALOS_SHADOW_LOG=<file>`:

```
reason:"ok", active:true, strength:0.88, alt_m:3540, eye:[…, …, 1.79e9]
```

A sane camera altitude (`alt_m:3540`) paired with a heliocentric-scale cascade eye is only
possible if `RenderOrigin` isn't tracking the camera. **The `eye` magnitude in
`THALOS_SHADOW_LOG` is the decisive tell for this bug class.**

## Hypotheses considered

- Casters missing from `SHADOW_CASTER_LAYER` in god-view — ruled out: `active:true`,
  `strength` sane, rig running.
- Cascade centring/AGL wrong — ruled out by the log: altitude was correct; the *eye position*
  was planetary-scale wrong, which only `RenderOrigin` can cause.
- big_space entities desynced — ruled out: terrain/buildings/trees rendered correctly (they use
  the `FloatingOrigin`, not `RenderOrigin`); only non-big_space render-space consumers — the
  cascade cameras — were the casualty.

## Root cause

`update_render_origin` (`rendering/transforms.rs`) set `RenderOrigin` from `CameraFocus.target`.
In flight the camera orbits the ship so focus ≈ camera and the bug is latent. God-views decouple
the `ShipCamera` from the focus pivot by planetary distances, so `RenderOrigin` diverged from
the `FloatingOrigin` and every render-space entity outside big_space desynced with it.

## Fix

When `game_context::context_freezes_sim` (hub / base editor / VAB — the god-view/modal
contexts), `update_render_origin` derives `RenderOrigin.position` from the `ShipCamera`'s
heliocentric pose (`cell × REAL_SPACE_CELL_SIZE_M + transform.translation`). Safe there because
the sim is frozen, so the Sync-stale camera pose is coherent; flight keeps the fresh
focus-target path. Structural: corrects `RenderOrigin` for **all** non-big_space render-space
consumers in any camera mode (hub, base editor, launch-select, screenshot rig) with no per-mode
plumbing — the same principle as `ViewAnchor` (the *view* decides render space, never a focus
target or mode flag).

## Prevention & recurrence signals

- **Invariant:** `RenderOrigin` must equal the big_space `FloatingOrigin`. Any new camera mode
  that moves the `ShipCamera` without a matching focus target must drive `RenderOrigin` from
  the camera. Never assume `CameraFocus.target` ≈ the render camera.
- **Recurrence tell:** missing shadows (or any misplaced non-big_space render-space entity) in
  exactly one camera mode → check `THALOS_SHADOW_LOG`'s `eye` magnitude against `alt_m` via
  `just screenshot hub`.
