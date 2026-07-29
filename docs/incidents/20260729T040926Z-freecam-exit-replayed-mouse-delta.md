# INC-20260729T040926Z-freecam-exit-replayed-mouse-delta: Normal camera flipped after freecam

- **Status:** Fixed
- **Date:** 2026-07-29
- **Severity:** behavioral
- **Surface:** F4 freecam → normal ship-camera handoff

## Summary

Leaving freecam released camera ownership before the normal orbit-input system
ran. If the exit frame still carried a held-pointer camera delta, the normal
controller consumed the same delta that belonged to freecam. A sufficiently
large vertical sample clamped the normal orbit elevation near 90°, producing an
apparently top-down or sideways normal view.

The normal rig now restores its transform immediately but ignores camera input
for the one exit frame. Freecam leveling also uses an explicit surface-flight
envelope instead of treating a body-fixed anchor as plane-like at any distance:
the canonical Kármán line on atmospheric bodies, or a radius-scaled ceiling on
airless bodies.

## Symptoms

- F4 back to the normal ship view could leave it near-vertical and unusable.
- The failure followed a freecam level-lock interaction and looked as though
  freecam's orientation had leaked into the normal camera.
- Level lock continued operating above the visible atmosphere.

## Evidence

- `toggle_freecam_system` runs before `camera_input_system`.
- On exit it set `FreeCam::active = false`, so `camera_input_system` resumed in
  that same update and read `GameInputIntent::camera_motion`.
- The intent sample is shared by both controllers for the frame; normal-camera
  elevation clamps at ±89°, matching the near-vertical recurrence.
- The level-lock gate previously required only a body-fixed reference frame. It
  never consulted `TerrestrialAtmosphere::karman_line_m`.

## Hypotheses considered

- **Freecam's rendered transform remained authoritative after exit.** Ruled out:
  `camera_transform_system` runs on the exit frame and reconstructs the normal
  transform whenever freecam is inactive.
- **The normal orbit state was directly mutated by freecam.** Ruled out:
  freecam writes its private body-fixed pose and the camera transform, not
  `CameraFocus`.
- **The exit-frame input sample crossed controller ownership.** Confirmed by
  system ordering and the shared `GameInputIntent` read.

## Fix

- `FreeCam::blocks_flight_camera_input()` covers both active freecam and a
  one-frame exit handoff.
- The handoff does not own the transform, so the normal camera pose is restored
  in the same frame; `freecam_drive_system`, ordered after that writer, then
  clears the input guard.
- Effective level-up is available below the anchored body's authored Kármán
  line, or below `min(5% of body radius, 100 km)` for an airless body. A 95%
  re-entry threshold prevents boundary chatter without carrying an atmospheric
  lock above the authored atmosphere top.
- Body-fixed reprojection remains active outside the surface-flight envelope,
  so warp stability is unchanged.

## Prevention & recurrence signals

- Camera transform ownership and camera input ownership are related but not
  identical during a controller handoff; model both explicitly.
- A normal orbit snapping to its elevation clamp only when F4 is released during
  a drag is the recurrence tell for an exit-frame input leak.
- Surface-flight constraints must use the authored atmosphere boundary where
  one exists and a curvature-scaled near-surface boundary where it does not,
  rather than merely testing for a planetary reference frame.
