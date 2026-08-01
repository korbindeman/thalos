# INC-20260731T202545Z-shadow-rig-raced-camera-system: the shadow rig raced Bevy's `camera_system`

- **Date:** 2026-07-31 · **Surface:** any 3-D view with ground shadows while the camera moves
  (`just game runway`, freecam, god view)

## Symptom

Grid-shaped shadow banding over the terrain and cast shadows visibly trailing their casters
**while the camera moved**; both settled the moment the camera was parked. Whole-frame
brightness also flickered between neighbouring frames at the same sim time. Nothing in the
console, no pipeline error, exit 0.

Static evidence is useless here: every headless capture parks the camera, so
`just screenshot` reproduces none of it. The reproduction is a play session with continuous
camera motion — a boom, a climb, or a zoom.

## Root cause

`update_sun_shadow_camera` rewrites each cascade camera's `Projection` every frame from the
current `footprint_scale`, and hand-builds the matching `block.view_proj[i]` that every
receiver samples with. The two are consistent **in the main world**.

The render world does not read `Projection`. `extract_cameras` reads
`Camera::clip_from_view()`, i.e. `camera.computed.clip_from_view`, and the only writer of
that field is Bevy's `camera_system` — registered plain in `PostUpdate`
(`bevy_render::camera`, `CameraUpdateSystems`) with just `.before(AssetEventSystems)` and
`.before(visibility::update_frusta)`. The rig's chain carried
`.after(CellCoord::recenter_large_transforms).before(TransformSystems::Propagate)` and
therefore **no ordering against `camera_system` at all**.

On the frames where `camera_system` ran first, the cached clip matrix was the one built from
the *previous* frame's projection. So the cascade depth map was rasterized at last frame's
extents while every receiver projected into it through this frame's `view_proj`. The camera
`GlobalTransform` was fresh either way (Propagate runs later), so the error is purely a
scale/extent mismatch — which is why it is invisible when parked and severe under motion:
`footprint_scale` tracks camera AGL and routinely swings 1.1 → 2.9 within a second of
ordinary flight (`stability_gauge.footprint_scale`, session `26008-1785481833998`).

INC-20260731T011523Z already named this stale-matrix path — "rendering stayed correct
because camera extraction uses `camera.computed.clip_from_view`, the matrix `camera_system`
cached from the *previous* replace" — but fixed only the *culling frustum* half of the race
(seating `area` at the write site). The extraction half stayed live. This is the third
ordering defect in the same chain (INC-20260730T223451Z was the first); all three come from
the chain declaring what it must run *after* and never what it must run *before*.

**Ruled out along the way**, both by matched captures rather than argument:

- *Terrain skirt curtains casting into the cascades* (the same commit switched tile skirts
  from a 150 m drop to a curtain reaching the body floor sphere). A curtain's top vertex is
  the terrain surface, so any ray it could block has already crossed that surface; and the
  grid does not appear in static captures where the curtains are equally present.
- *Low-sun over-bias detaching shadows.* Neutralizing `BIAS_MAX_M` /
  `NORMAL_OFFSET_MAX_M` / `BIAS_MIN_M` in `shadow.wgsl` and recapturing the same preset at
  a pinned sun produced acne on the casters' own lit faces (proof the edit took) and **no
  change in shadow length**. A sun-elevation sweep over uniform grass
  (`artifacts/visual/runs/shadowdiag/aerial_*.png`) shows shadows attached at the caster
  base and scaling correctly with elevation. There is no static low-sun shadow defect; two
  earlier readings of one were the near-black apron hiding the shadow it fell on.

## Fix

`.before(bevy::camera::CameraUpdateSystems)` on the rig's `PostUpdate` chain. Before, not
after: `camera_system` then caches `clip_from_view` from the projection written this frame,
so the rendered map and the sampled matrix are the same transaction. `ortho.update()` at the
write site stays — it keeps `update_frusta` correct on its own terms regardless of order.

No cycle: `camera_system` is unconstrained relative to `TransformSystems::Propagate`, and
`CellCoord::recenter_large_transforms` is plain in `PostUpdate`, so the chain can sit before
both.

## Recurrence signal

Shadows that lag their casters or band over the caster tiles **only while the camera moves**,
with `stability_gauge.origin_frame_error_m` at 0.0 (which rules out the
INC-20260730T223451Z cell-crossing race) and `footprint_scale` changing between samples.

**Standing rule, for any system that writes a camera component:** a hand-written
`Projection` is read by the render world only through the matrix `camera_system` caches, so
such a system must declare `.before(CameraUpdateSystems)`. Declaring only what it runs
*after* leaves the consumer edge unstated, which is how all three races in this chain
happened.
