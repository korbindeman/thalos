# INC-20260724T223259Z-viewpoint-teleport-bypassed-freecam-owner: View snapped back after one frame

- **Status:** Fixed
- **Date:** 2026-07-25 (observed) / 2026-07-25 (fixed)
- **Severity:** behavioral
- **Surface:** F8 viewpoint manager, saved and agent-scripted “View” action

## Summary

The manager posed the rendered `ShipCamera` directly. That produced a visible
one-frame teleport, but the normal orbit-camera system still owned the camera
and reconstructed its focus-derived transform on the next update. The action
now activates freecam through a canonical API that seeds freecam's persistent
body-fixed anchor from the resolved pose and preserves its authored base FOV.

## Symptoms

- Clicking “View” visibly jumped to the selected composition.
- The camera immediately snapped back to its previous craft/focus view.
- The selected lens would also have eased back toward the default FOV because
  freecam previously treated Bevy's default lens as its only unzoomed target.

## Evidence

- `camera_transform_system` writes every active orbit camera each update unless
  `FreeCam::owns_camera_transform()` returns true.
- The manager changed `Transform`, `CellCoord`, and `Projection` but did not
  change `FreeCam`; ownership therefore remained with the orbit-focus path.
- Freecam stores a private `FreeCamReferenceFrame::BodyFixed` pose and
  reprojects that anchor every frame. Direct transform writes cannot update it.

## Hypotheses considered

- **The body-fixed viewpoint conversion was wrong.** Ruled out by the initial
  visible jump to the intended view; conversion produced a valid pose.
- **Another egui frame undid the click.** Ruled out because egui does not own
  the 3-D transform.
- **The normal camera controller reclaimed ownership.** Confirmed by its
  unconditional next-frame writer when freecam was inactive.

## Root cause

The manager bypassed the canonical camera-ownership state and treated the
rendered camera transform as authoritative. In Thalos it is an output of either
the orbit-focus controller or freecam's private anchor.

## Fix

`FreeCam::activate_at_world_pose` now atomically:

- activates freecam;
- snapshots the normal warp policy;
- captures the resolved world pose into the selected body's fixed frame; and
- records the current perspective FOV as the session's unzoomed lens.

Both saved and scripted manager entries pose the camera, then route the result
through that API before the next camera update.

## Prevention & recurrence signals

- Teleports must update the owning controller's state, never only its rendered
  transform. The invariant is recorded in `docs/development/visual_testing.md`
  and `docs/gameplay/input.md`.
- A correct one-frame jump followed by an exact snap-back is the recurrence
  tell for controller ownership, not coordinate conversion.
