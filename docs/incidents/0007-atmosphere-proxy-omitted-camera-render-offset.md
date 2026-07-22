# INC-0007: Raymarched atmosphere detached from the visible planet

- **Status:** Fixed
- **Date:** 2026-07-20 (observed and fixed)
- **Severity:** visual
- **Surface:** `just screenshot earth-reference`

## Summary

The first Bevy raymarched comparison rendered a huge diagonal atmosphere wedge
instead of a limb around Thalos. The proxy center was expressed relative to an
assumed camera-at-zero, but the screenshot camera retained a non-zero render
translation around its parked-craft origin. Adding the camera's actual
`GlobalTransform` translation to the f64 camera-to-body vector put the atmosphere
and terrain into the same render frame.

## Symptoms

- `earth_reference_bevy_r0.png` and `r1.png` showed an atmosphere boundary that
  crossed the frame diagonally and did not follow the visible planetary curve.
- The custom capture at the identical scripted camera still wrapped the visible
  terrain, ruling out the camera preset and terrain geometry.
- Repro: `THALOS_SCREENSHOT_ATMOSPHERE=bevy just screenshot earth-reference`.

## Evidence

The screenshot camera's `GlobalTransform::translation()` was not zero. The
failed proxy position used only the body center relative to the camera; the
corrected capture added the camera translation and immediately changed the
diagonal wedge into a concentric planetary limb. Later density changes altered
only colour/opacity, not the recovered alignment.

```text
failed:    proxy_center = camera_to_planet
corrected: proxy_center = camera_global.translation() + camera_to_planet
```

## Hypotheses considered

- **Atmosphere density or shell height was too large.** Density changes affected
  opacity but could not make the diagonal boundary concentric with terrain; ruled
  out as the alignment cause.
- **The scripted camera angle targeted the wrong point.** The matched custom
  `BodySky` capture followed the planet at the same pose; ruled out.
- **The atmosphere and terrain used different render frames.** The wedge vanished
  when the camera render offset was included; confirmed.

## Root cause

`ViewAnchor.cam_body` provides the f64 body-fixed camera vector and therefore the
body center *relative to the camera*. BigSpace keeps coordinates small but does
not promise that the `ShipCamera` itself is at render-space zero. The screenshot
rig left the render origin near the parked craft, so assigning the relative
vector directly to the atmosphere proxy omitted the camera boom offset and moved
the shell away from the rendered terrain.

## Fix

`sync_stock_atmosphere` now rotates the f64 body-fixed vector into world axes,
negates it to obtain camera-to-body, converts only that local result to `Vec3`,
and adds `ShipCamera`'s current `GlobalTransform::translation()`. The proxy is
also updated after real-space body positions in the sync schedule.

## Prevention & recurrence signals

- A camera-relative effect must be placed in the same extracted render frame as
  the camera: `camera_global + relative_offset`; BigSpace does not imply
  `camera_global == 0`. The invariant is recorded in `docs/rendering/atmosphere.md`.
- A straight or diagonal atmosphere boundary that does not share the terrain's
  curvature is a frame-placement bug. Do not start by retuning density.
