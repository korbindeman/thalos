# INC-20260723T232652Z — impostor billboard covered the tile terrain; "most legible" landmark pointed at a plain

**Status:** fixed (both findings), one successor question open (tracked in
NTR-X1)
**Fix commit:** e45b7e4
**Area:** NTR-X1 standard-path tile renderer × body-LOD swap × airless
landmark metadata

## Symptom

With `THALOS_TILE_RENDERER=1`, Mira rendered as a featureless smooth sphere
with faint painted-on albedo splotches — "shading is quite strange", "no
detail up close" (user, live fly-through 2026-07-24), reproduced headlessly on
`mira-surface` and `mira-rim`. Meanwhile the tile telemetry showed the
streamed tiles carrying the package's full ±6 km height range.

## Differential and evidence

1. *Heights lost in the provider?* — No. Per-tile telemetry (`GenStats
   h [lo, hi]`) showed real relief in the streamed tiles.
2. *Mesher/transform flattens the mesh?* — No. Mesher is probe-verbatim with
   central-difference normals and a winding self-test.
3. *We're not looking at the tiles at all?* — **Yes.** Two discriminators:
   the albedo splotches vanished the moment the swap system was taught to
   hide the impostor (they were the impostor's own baked cubemap), and a
   red-tinted tile material turned the visible sphere red (the tiles).
4. *Then why still flat?* — the rim framing's landmark hovered over gentle
   plains: per-landmark transects showed the first 8 registry landmarks all
   real (sampled relief ≈ metadata), but the framing's `MostLegible` pick
   (claimed 5,759 m relief) sat on ±0.5 km-scale ground.

## Root causes (two, stacked)

**A. The terrain↔impostor swap keys on udlod residency only.**
`sync_body_render_lod` builds its `terrain_resident` set from `BodyTerrain` +
`TileAtlas::pinned_tiles_ready()`. A body rendered by the NTR-X1 tile path
has neither, so its `RealSpaceImpostor` stayed `Visibility::Inherited` at
every distance. The impostor is an analytically-shaded smooth sphere — it
*is* the "flat, painted, strange" look, and it drew over the (correct) tiles.

**B. Landmark `relief_m` was a model estimate consumed by a max-selector.**
`airless_landmarks` stored `depth_m × degradation_factor` — an analytic
guess. The rim framing picks `max_by(relief_m)` over ~128 candidates, which
selects **exactly the crater the model most overestimates** (an old basin the
bake has relaxed to a plain). Selection-on-model-error: the young "Typical"
half was fine only because degradation ≈ 1 there, so model ≈ surface.
Pre-existing — udlod's `mira-rim` hovered over the same empty plain.

## Fixes

- `TileTerrainRoot` latches first full coverage (`coverage_ready()`, the
  analogue of `pinned_tiles_ready`); `sync_body_render_lod` counts
  tile-rendered bodies as resident. Hole-free despawn keeps the latch honest.
- `airless_landmarks` now **measures** relief from the surface at registry
  load (centre + floor/rim ring samples through `SurfaceQuery`); the analytic
  rank only bounds the measured candidate pool.

## Prevention / standing rules

- **A renderer that takes ownership of a body must feed every consumer that
  keyed on the old renderer's residency.** The udlod gate was flipped
  per-body, but "resident" was still udlod-only — the same class of bug
  awaits in the GPU height mirror, halo/sky/ocean flips, and anything else
  reading `BodyTerrain`. When flipping a body to the tile path, grep for
  `BodyTerrain`/`pinned_tiles_ready` consumers.
- **Never store a model estimate where a max-selector reads it.** Ranking by
  an estimate turns its worst error into the winner. Measure from the
  authority (the surface) or validate the winner before use.
- The red-tint material discriminator ("is the visible surface this mesh?")
  is cheap and decisive for draws-over-draws confusion; pair it with a
  per-batch height-range log so "geometry vs shading vs occlusion" separates
  in one capture.

## Recurrence tells

- A tile-rendered body that looks analytically smooth with painted albedo →
  some consumer is still showing an impostor-path visual for it.
- A cinematic framing "containing" a landmark that shows empty ground → the
  landmark metadata drifted from the surface; check measured vs stored
  relief.

## Open successor question (tracked in backlog NTR-X1)

At the most-legible landmark the load-time measurement (via
`sample_height_m`, f32 dir, coarse `lod_m` ≈ 2 km) sees ~6.5 km of relief,
while the tile provider (`sample_d`, f64 dir, `lod_m` 40–80 m) samples flat
ground at the same direction — both paths funnel into `surface_sample`, so
the suspect is LOD-dependent package reconstruction. A probe comparing both
paths across `lod_m ∈ {2000, 830, 83, 20}` at that exact direction is staged
in `tile_terrain.rs` (uncommitted diagnostic) and blocked on an unrelated
in-flight cloud edit breaking the workspace build.
