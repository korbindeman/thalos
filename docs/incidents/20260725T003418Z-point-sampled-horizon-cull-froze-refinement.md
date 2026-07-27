# INC-20260725T003418Z-point-sampled-horizon-cull-froze-refinement: point-sampled horizon culling froze tile refinement at the MIN_LEVEL shell

- **Date:** 2026-07-25 · **Surface:** tile terrain refinement, any framing —
  found on `THALOS_TERRAIN=diffusion just screenshot massif-aerial`

## Symptom

Adding a horizon gate to `tiles::select_leaves_with_relief` made every framing
render as a smooth featureless sphere. The tell was in one log line:

```
tile terrain: first full coverage (384 tiles) — impostor handoff ready
massif terrain streamed (0.0 s) — starting warmup
```

**384 = 6 × 8², exactly the `MIN_LEVEL` shell** — not "too few tiles", but
*zero splits anywhere*. The `0.0 s` stream is the same fact from the other side:
the capture's LOD-plateau gate saw no refinement to wait for.

## Root cause

The gate tested visibility by sampling the tile's four corners and its centre,
lifting each to `radius + relief` and applying the exact sphere-tangent test
`p · c ≥ r²`. That test is right; sampling points to stand in for the tile is
not.

A tile at `MIN_LEVEL` spans a quarter-circumference / 8 — **589 km on Thalos**.
The first selection tick runs before the capture poses its camera, so the eye
was at 756 m altitude, where the horizon (with the full ±9.8 km relief
allowance) reaches ~260 km. Every corner and the centre of the level-3 tile
underfoot lie 300–420 km from the sub-camera point, so all five samples fail —
while the ground *directly beneath the camera*, inside that same tile, is in
plain view. The gate therefore refused to split level 3, the descent never got
to a level whose tiles were small enough for point samples to be meaningful,
and refinement was stuck at the shell forever.

Diagnosis went through two wrong guesses first, both cheap to rule out with a
single one-shot `info!` of the actual inputs (`radius`, `relief`, `|cam|`, the
dot, `r²`): that `SurfaceQuery::height_range_m` was returning something
degenerate, and that `cam_body` was not a centre-relative position. Both were
fine — `relief = 9797.6`, `|cam| = R + 756`. The numbers named the real
problem immediately: `dot = 1.01430e13` vs `r² = 1.01506e13`, a 0.07 % miss,
i.e. "just over the horizon", for a tile the camera was standing on.

## Fix

Bound the tile by its **cone**, not by samples: compare the angle between the
eye direction and the tile centre, minus the tile's angular radius (centre to
farthest corner), against the horizon half-angle
`acos(r² / ((r + relief) · |c|))`. Conservative for any tile size, so it cannot
reproduce as the quadtree gets coarser — which is the property point sampling
never had.

## Recurrence signal

A resident/desired tile count that is exactly `6 × 4^MIN_LEVEL` (384 today), or
a `massif terrain streamed (0.0 s)`. Both mean refinement never started, which
in this subsystem almost always means a per-tile predicate rejected a coarse
tile it had no business judging.

**Standing rule for this module:** any new per-tile visibility or rejection test
(frustum culling is the next one queued, `NTR-X12`) must bound the tile as a
region — cone, sphere, or AABB. A tile is up to 589 km across; point samples
inside it say nothing about the rest of it. `tiles::horizon_tests` gates this
directly: `horizon_gate_keeps_the_ground_underfoot` asserts the nadir tile
survives at every level from `MIN_LEVEL` down, at altitudes from 30 m to 200 km.
