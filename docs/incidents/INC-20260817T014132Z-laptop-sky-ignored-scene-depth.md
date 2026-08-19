# INC-20260817T014132Z: Laptop render scale made the sky ignore the scene

- **Date:** 2026-08-17 · **Surface:** `just game launch` on Laptop (0.50× 3D), camera pitched slightly above the pad so the geometric horizon cuts through the stack

## Symptom

The sky was a solid wall on every pixel above the geometric horizon. The upper
half of the rocket vanished into air, and distant mountains that should have
broken the skyline were gone. Ground below the line still drew, but as a
featureless sheet. The cut was ruler-straight and moved with camera pitch, not
with the ship.

## Root cause

Laptop 3D is a smaller main target, upscaled to the window. The sky composite
clips its raymarch against `SceneDepthImage`, a copy of that target's depth.
`copy_scene_depth` skips any frame whose copy source and destination differ in
size, leaving the image cleared to 0 (reverse-Z empty). The shader then falls
back to the mean-radius planet sphere: opaque sky above the geometric horizon,
thin haze below.

The size hang was extract order. `extract_cameras` queues a full-window
`ExtractedCamera` insert. The scale path mutated that component in
`ExtractSchedule` after `extract_cameras` had *queued* its insert, so the
queued insert won when commands applied. Every frame the 3D target stayed at
window size while `SceneDepthImage` resized to 0.50×, and the copy never ran.

Showcase (1.00×) never took the shrink path, so capture and a Showcase session
did not show it.

## Fix

Shrink `ExtractedCamera` / `ExtractedView` in `RenderSystems::PrepareViews`,
after extract has applied and before `prepare_view_targets` allocates color
and depth. `copy_scene_depth` logs `depth_copy_skipped` when sizes still
differ; `just diag` reports a hang of three or more records.

## Recurrence signal

Opaque sky replacing the top of a nearby craft, cut exactly on the geometric
horizon, only while 3D render scale is below 1. `runtime.jsonl` event
`depth_copy_skipped` with `src_width` ≠ `dst_width` is the tell. Probe:
`THALOS_SCREENSHOT_RENDER_SCALE=0.5 just screenshot runway-atmosphere`.
