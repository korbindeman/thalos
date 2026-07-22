# INC-0002: Terrain intermittently covered / z-fought the space center — LOD height error vs the flatten plane

- **Status:** Fixed
- **Date:** 2026-07-18 (observed + fixed)
- **Severity:** visual
- **Surface:** space-center hub / any god-view or flight view of a flattened base at LOD-transition distances (`just game hub`, `just screenshot hub`)

## Summary

At certain view distances the rendered terrain rose up through the spaceport's
paving and buildings — sometimes swallowing the base outright, otherwise
z-fighting the aprons/taxiways. The pad flatten is applied at tile *bake* time,
but the ground the player sees is the tile atlas: udlod's vertex-stage LOD
height **blend and morph** mix in coarse ancestor tiles whose km-scale texels
average natural terrain into the basin, and a runtime-installed flatten can
leave stale unflattened tiles resident. Either error is decimetres — more than
the 0.12 m asphalt lift. Fix: the terrain vertex shader now re-applies each
flatten pad **analytically per vertex**, pinning the rendered ground inside a
pad rectangle to the exact tangent-plane elevation at every LOD / morph / bake
state, so structures render above the ground *by construction*.

## Symptoms

- The whole base (runways, aprons, buildings) intermittently disappears under
  terrain or shimmers/z-fights against it; "sometimes" because it depends on
  which LOD/morph band the site falls in for the current camera distance.
- User could not capture it in a still — the flicker component is temporal —
  but partial coverage is visible in top-down captures at ~30 km-scale
  distances (parts of the strip/campus eaten by green).
- Repro: `just screenshot hub` with `THALOS_SCREENSHOT_ELEVATION=85`
  `THALOS_SCREENSHOT_DISTANCE=30000` (site renders from coarse tiles).

## Evidence

- The mechanism was already documented in-repo at the spaceport-build rebuild
  call ([runway.rs](../../crates/runtime/game/src/runway.rs) `rebuild.request`): the
  vertex-stage LOD blend/morph "pull[s] the rendered ground ~decimetres off the
  flat plane in the LOD-transition bands … enough to z-fight the flush
  aprons/taxiways". The rebuild only repairs the *stale-tile* subset; the
  blend/morph error is inherent to coarse-texel interpolation and re-appears at
  every LOD-transition distance regardless of bake state.
- The headless apron probe (`just screenshot` log line `apron probe`) shows the
  baked height drifting off the plane toward the basin edge (`+0.23 m` at
  ±1200 m offsets with 4.8 m texels — at coarser LODs the same interpolation
  error scales with texel size past the 0.12 m paving lift).

## Hypotheses considered

1. **Depth-buffer precision (paving lift < depth ulp at distance)** — ruled
   out: Bevy's reverse-Z gives ~mm-cm precision at these distances, well under
   the 0.12 m lift. The flicker needs *geometry* actually crossing the paving.
2. **Stale pre-flatten tiles resident after a runtime flatten** — real but
   already handled (`TerrainRebuildRequest` on spaceport build / base-editor
   flatten); does not explain recurrence after rebuild.
3. **Coarse-tile texel interpolation + vertex LOD blend/morph** — confirmed
   (and pre-documented at the rebuild site): coarse texels straddling the basin
   average natural terrain into it; the vertex blend mixes those heights into
   the rendered ground at LOD-transition bands. Decimetres > 0.12 m lift →
   coverage + z-fight. View-distance dependence matches "sometimes".

## Root cause

The flatten plane was enforced only at tile-bake time, but rendered height is a
LOD-dependent *resampling* of baked tiles (bilinear texel filtering, vertex
LOD blend between two atlas LODs, vertex morph toward the parent grid). No
bake-side fix can make that resampling exact at every LOD, so the rendered
ground under structures was only approximately the plane structures were built
against — with error proportional to texel size.

## Fix

Make the flatten plane a **render-time authority** (structural; see
docs/world/terrain.md "Structures render above the ground"):

- `body_terrain.wgsl` gains a custom vertex entry (`flattened_height`):
  inside a pad rectangle the vertex height is overridden to the exact
  tangent-plane elevation — the same `TerrainFlatten::plane_elevation_m` maths
  the bake/collider/placement use — feathered over a band just inside the rect
  edge. Interior-only on purpose: the smoothstep ramp *outside* the rect is
  already baked into tiles, and re-applying it would double-blend and pull the
  visual ramp metres off the collider. Formulated cancellation-free for f32
  (`1 − cosθ = |dir − c|²/2`; never `(R+E)/cosθ − R`, which quantises at
  ulp(R) ≈ 0.25 m).
- `BodyTerrainMaterial` packs the pads as `FlattenBlock` / `FlattenRegionGpu`
  (in `BodyTerrainExtras`, no new bind slot); `update_body_terrain_atmosphere`
  mirrors each body's `TerrainFlattenRegistry` regions into it per frame
  (nearest-camera pads beyond `MAX_FLATTEN_REGIONS`). The map terrain keeps a
  zero block (no cost, no behaviour change).
- The bake-side flatten + rebuild remain load-bearing for the collider/CPU
  height mirror, albedo/material layers, scatter, and the ramp band — the
  rebuild-site comment in runway.rs now says exactly which halves each
  mechanism owns.

Verified headlessly: `just screenshot hub` at 4 km, 9 km low-angle, and 30 km
top-down — base fully above ground in all three, no pad-edge crease.

## Prevention & recurrence signals

- **Invariant (docs/world/terrain.md):** rendered ground under structures equals the
  plane they were built against, by construction; any future ground renderer
  must honor the flatten handle at render time, not only at bake time.
- WGSL gotchas hit while landing it are in the `wgsl-bevy` skill: `meta` is a
  reserved word; a file serving as both vertex+fragment shader under udlod must
  gate both entry points on the `FRAGMENT` def (udlod's `Coordinate` changes
  field count under it, and naga_oil never prunes entry points).
- **Recurrence tell:** paving/buildings sinking or shimmering only at specific
  camera distances (LOD bands), while close-up views look flush — capture with
  `just screenshot hub` + `THALOS_SCREENSHOT_DISTANCE` sweeps before touching
  bake code.
