# INC-20260730T222419Z-stuck-maneuver-drag-ate-the-throttle-keys: a stranded maneuver drag silently ate Shift/Ctrl

- **Date:** 2026-07-30 · **Surface:** any flight mode, after using the map-view maneuver editor

## Symptom

"Shift and Ctrl don't work for throttle anymore." Every other control was fine —
Z/X still snapped the throttle full/cut, W/A/S/D/Q/E still flew, the HOTAS still
worked. Sticky for the rest of the session; a restart cleared it.

The distinguishing facts, which is what made it tractable: the sim was *running*
(so it was not the paused-clock path), and Z/X still worked (so the input layer
was alive and the flight context was active). Shift and Ctrl are the only keys
bound in a **higher-priority** input context than `GameFlightContext`, so
whatever broke had to be a context that consumes them.

## Root cause

`GameManeuverPrecisionContext` (Shift = 10× finer, Ctrl = 100× finer while
dragging a maneuver arrow) sits at `ContextPriority` 90 with
`consume_input: true`; `GameFlightContext` — which owns the `throttle_ramp`
axis — sits at 20. `sync_maneuver_precision_context` activates the precision
context whenever `InteractionMode` is `DraggingArrow` or `SlidingNode`. So a
drag mode that never ends starves the throttle ramp forever, and nothing else,
because no other flight binding collides with Shift/Ctrl.

`InteractionMode` had two ways back to `Idle`, and both could be missed:

1. The `Pointer<DragEnd>` observers (`arrow_drag_end`, `slide_sphere_drag_end`)
   only fire for a **live** hitbox entity. A handle despawned mid-drag —
   selection change, node deletion, leaving map view — takes the event with it.
   `manage_arrow_handles` already guards the one case its author hit
   (`!has_node` while dragging); it does not cover the others.
2. The release check inside `maneuver_input` sat **below that system's six
   early returns**: no window, `window.cursor_position()` → `None`, no
   `ActiveCamera`, no `SimulationState`, `sim.simulation.prediction()` → `None`,
   no `SolarSystemState`. Releasing the button with any of those true skipped
   the `*mode = Idle` entirely. The reachable ones in ordinary play: the cursor
   leaving the window mid-drag (very common — you drag past the window edge and
   let go), and **no orbital prediction at all**, which is the normal state for
   an atmospheric aircraft. The whole system also ran under
   `not_game_paused`, so a release with the pause menu up was lost too.

A wrong-but-instructive first hypothesis: the HOTAS throttle lever reclaiming
the axis from the keyboard each frame (`THROTTLE_LEVER_MOVE_EPS`, see
INC/`runway-shake` history — the stick has caused input bugs before). It was
ruled out by a single fact: the lever's `return` in `handle_throttle_input` sits
*above* the Z/X branch, so a reclaiming lever would kill Z/X too. It didn't.
The same fact ruled out the other candidate, the ramp's `SimClock` delta going
to zero while paused — that needs a paused sim, and the sim was running. (That
one is still a real defect: throttle is an input, not a simulated quantity, so
the ramp should not freeze at warp 0×. Filed separately.)

## Fix

The invariant — **a drag mode implies the primary button is held** — is now its
own system, `end_drag_on_release`, registered *outside* the `not_game_paused`
tuple and before `sync_maneuver_precision_context` / `update_camera_block` /
`maneuver_input`. It reads only `GameInputIntent` and the mode, so it has no
early returns to hide behind and no state in which it declines to run. The
per-arm release handling inside `maneuver_input` is deleted (that system now
only ever sees a live drag, and no longer takes a write lock on `ManeuverPlan`).

This removes the class rather than the instance: any *future* early return added
to `maneuver_input`, and any new way for a hitbox to be despawned mid-drag, is
now harmless.

## Recurrence signal

A modifier key silently doing nothing in flight while every non-modifier key
works. Check `InteractionMode` first — if it is not `Idle` outside a drag, the
same starvation is back. `BlockCameraInput` is the second tell from the same
resource: a stuck drag mode also blocks camera orbit, so "Shift/Ctrl dead" and
"right-drag orbit dead" appearing together is this bug and not an input-binding
problem.

Standing rule, from the shape of this one: **a mode resource that gates an input
context must be released by a system that cannot decline to run.** Putting the
release below a fallible lookup makes the mode leak, and a leaked mode that
consumes a key is invisible — nothing errors, nothing logs, one key just stops.
