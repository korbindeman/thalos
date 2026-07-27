# INC-20260725T184654Z-tile-depth-bias-was-sort-order-only: the space center flickered in and out of the ground

- **Date:** 2026-07-25 · **Surface:** `just game hub` / any god view, moving the camera or the craft over the spaceport

## Symptom

The flat parts of the space center — runway asphalt, taxiways, aprons — **flickered in and
out of the terrain as the camera or craft moved**, and worse the further out the view got. In
a frame that had lost them, the runway survived only as its painted markings: two thin lines
and the centreline dashes, with the asphalt between them gone. Buildings and the launchpad
slabs stayed. A settled headless capture at the same framings (`hub` at 8 km and 15 km) drew
everything correctly — the defect only existed *during* a move.

That split — asphalt gone, markings kept — is the tell, and it is a measurement: paving sits
`RUNWAY_ASPHALT_LIFT_M` = 0.12 m over the pad and markings `RUNWAY_MARKING_LIFT_M` = 0.17 m.
Whatever was covering the base was standing between the two.

## Root cause

Two independent faults, one making the other visible.

**1. The tile renderer's "finer detail always wins" rule was never implemented.**
`tiles/mod.rs` set `StandardMaterial::depth_bias = level × 2.0` and documented it as a
hardware `DepthBiasState.constant`, so that a stale coarse tile lingering over its refined
replacement would lose the pixels. Bevy does not implement `depth_bias` that way. It folds it
into the render phase's **sort distance** only — `bevy_core_pipeline/src/core_3d/mod.rs`:
`rangefinder.distance(&mesh_center) + depth_bias` — and among opaque geometry, sort order
decides nothing; the depth test does. (`rendering/spawn.rs` already carried a comment saying
exactly this about the same field, on a different material.) So the bias was a no-op, and the
`DESPAWN_GRACE_S` window was *deliberately lax* on the strength of a guarantee that did not
exist.

**2. Tile selection was blind to structure pads.** `select_leaves_scaled` refined on camera
distance and measured ruggedness. Nothing told it a levelled basin existed at the space
center, so on a merge the coarse ancestor over the base was allowed to be arbitrarily coarse.
At the spaceport the flatten cuts **83 m** of terrain (`basin levelled to mean 609 m, terrain
537..692 m`), and a tile whose sample spacing exceeds the basin no longer resolves the flat
footprint at all: its mesh cuts straight across from the natural ground outside, putting tens
of metres of hillside back over paving that stands 0.12 m proud.

The two compose into exactly the observed behaviour. The despawn rule is hole-free **by
construction** — on a merge the coarse ancestor lands *before* its fine children retire — so
for the grace window both are drawn and interpenetrate. Fault 2 makes the coarse one wrong
over the base; fault 1 lets it win the pixels anyway. Camera motion is what drives merges, so
the artefact is exclusively a motion artefact, which is why every settled capture was clean.

## Fix

- `LEVEL_DEPTH_BIAS_STEP` (material sort key) → `LEVEL_RENDER_LIFT_M` (2 mm/level of real
  radius, baked into the tile mesh). Real geometry, so it holds at any distance and any depth
  precision, and is view-independent so it cannot pop. Bounded well under the 0.12 m paving
  lift at `max_level` 18 (36 mm). The height mirror publishes the *provider's* heights, so
  colliders, camera floor and HUD altitude are untouched.
- `RefinementSite`: authored ground publishes a resolution floor that selection honours ahead
  of both the distance rule and the residency brake (yielding only to `above_horizon`). The
  game driver derives one per flatten region, asking for four samples across the pad's blend
  ramp. Measured cost: nothing below ~30 km, +33 leaves from a 200 km orbit, +90 from
  2,000 km — under 1 % of the residency budget, asserted in `pad_sites_leave_the_rest_of_the_body_alone`.
- The per-level material set existed only to carry the dead bias; collapsed to one handle.

## The tell

- Something draped on the ground disappears while a *taller* or *thicker* thing on the same
  footprint survives → the ground rose by less than that thing's height; measure against the
  drape lifts to bound the error before theorising about a mechanism.
- The artefact exists only while the camera moves and no still reproduces it → suspect the
  LOD replacement window, not the steady-state selection.
- **`StandardMaterial::depth_bias` never biases depth.** If a comment claims a Bevy material
  field resolves z-fighting, check what the phase actually does with it: for opaque geometry
  the only things that decide a pixel are geometry and the depth test.
