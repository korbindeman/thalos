# INC-20260724T232104Z-shadow-cascades-measured-from-the-craft: sun-shadow cascades were placed in the craft's frame, not the render frame

- **Date:** 2026-07-24 · **Surface:** any view whose camera leaves the craft — freecam (F4), god views, boomed-out capture framings

## Symptom

Flying the freecam away from the ship, shadows of scattered detail (trees, rocks,
structures) stop appearing past a few kilometres: the shadowed region stays parked around
the craft and does not follow the camera. Near the craft everything looks right, so it
reads as "shadows only exist within a radius of the ship".

The same mechanism degrades the canonical capture presets: at `spaceport-aerial`'s 4.2 km
boom the crisp near cascades already miss the frame entirely and only the coarse far
cascade (metres per texel) still covers it — the long-standing "shadows past the near
cascade are blobs" complaint.

Decisive evidence: `THALOS_SHADOW_LOG` now publishes `centre_off_m`, the render-space
distance from the camera to the cascade centre. The cascades centre on the ground under
the camera, so it can only legitimately be about `alt_m`. Anything larger — it equals
|camera − craft| — is this bug.

## Root cause

`rendering::sun_shadow` builds its cascade centre in f64 world space (from `ViewAnchor`,
correctly the camera) and then projects it into render space to place the orthographic
cascade cameras, which live *outside* big_space. It projected through
`coords::RenderOrigin`.

`RenderOrigin` is not the render frame. It tracks the **camera focus pivot** — the craft,
in flight — for the scaled map/orbit projections. big_space renders every entity relative
to the **floating origin's grid cell origin** (`LocalFloatingOrigin::set(origin_cell,
ZERO, IDENTITY)`), and the floating origin is the `ShipCamera`. The two agree only while
the camera sits on top of the craft; they diverge by the entire camera↔craft separation
the moment the view leaves it. So the whole cascade set was placed |camera − craft| away
from the world it was supposed to cover: cascades 0 and 1 covered nothing visible, and
only cascade 2 — inflated by the footprint's `cam_dist` term, which was silently
compensating for this very error — still overlapped the frame, at its coarsest texels.

The bug was invisible at the architecture level because the anchoring had already been
fixed *once*: `ViewAnchor` (ADR-era work) made the cascade centre follow the camera in
world space, and every scatter driver follows it too. The residual craft anchoring was
hiding one layer down, in the world→render projection. The earlier hub fix — the
`context_freezes_sim` branch in `update_render_origin` that snaps `RenderOrigin` to the
god-view camera — was the same bug caught in one context only; it made the frozen god
views work and left flight, and therefore freecam, broken.

## Fix

`rendering::real_space::RealSpaceOrigin` — a resource holding the world point that renders
at render-space zero, written from the `FloatingOrigin`'s `CellCoord`. The shadow rig
projects through it; `RenderOrigin` keeps its map-space job, with both doc comments now
saying which is which. Structural rather than a compensating offset: any future entity
placed in real space from outside the big_space hierarchy has one correct thing to read.

Residual, documented at the writer: the camera's `CellCoord` is read at `SimStage::Sync`,
before the camera drivers rewrite it, so on a frame where the camera crosses a 1 km cell
boundary the origin trails the render frame by one cell. That is the ordinary one-frame
camera lag every `SimStage::Sync` consumer carries, not the unbounded craft-relative error
it replaces.

## Recurrence signal

`centre_off_m` in `THALOS_SHADOW_LOG` materially exceeding `alt_m`. Standing rule (owned by
`RealSpaceOrigin`'s doc): metre-scale placement among big_space content measures from the
floating origin's cell origin; `RenderOrigin` is for scaled map space only.
