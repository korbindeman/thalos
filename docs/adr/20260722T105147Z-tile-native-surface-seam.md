# ADR-20260722T105147Z-tile-native-surface-seam: The tile is the surface authority; the terrain-renderer rework is deferred behind a schema freeze

- **Status:** Accepted
- **Date:** 2026-07-22
- **Related:** ADR-20260722T105146Z-stay-on-bevy-reject-engine-migration ·
  ADR-20260720T211046Z-offline-terrain-packages

## Context

`SurfaceQuery::sample(dir: Vec3, lod_m: f32) -> SurfaceSample` is a **point
query**. That signature is native to `ProceduralSurface`, an analytic function
evaluable anywhere and band-limited by `lod_m`. It is foreign to everything the
terrain direction is moving toward: the diffusion producer works in windows with
overlap fusion, and the package is a residual pyramid of rasters.

The resulting path is **raster → point → raster**: `PackageSurface` materialises
a tile, `sample` point-samples it per tile *pixel*, and `compute_tile_pixels`
re-rasterises into `AttachmentData`. It also flattens two real hierarchies —
the package's parent/child residual structure and UDLOD's quadtree — into a
single scalar, then rebuilds each from the other side.

Three further observations:

- **The rasterised tile is already the de facto authority.** Colliders, EVA,
  camera floor, and HUD altitude read `HeightSourceRegistry`, described in
  `CLAUDE.md` as "a GPU-atlas height mirror over the same surface, with a CPU
  fallback." `SurfaceQuery` is the vestigial contract; the CPU fallback is the
  leak keeping two paths alive.
- **`terrain_lod_optimization.md` blocks GPU tile production** on the claim that
  a WGSL port "creates a second height authority." With a package there is no
  cascade to port — the CPU decodes a raster rather than evaluating a field —
  so the blocker is a property of the analytic producer, not of GPU production.
- **UDLOD is off Bevy's default render path** (its own pipeline plus a
  `queue_terrain::<M>` system that must be ordered against Bevy's
  `queue_material_meshes`). Meanwhile `craft.rs` already demonstrates the
  on-path pattern: `ShadowedStandardMaterial =
  ExtendedMaterial<StandardMaterial, ShadowReceiveExtension>` — standard
  pipeline, Thalos extension, our shadow rig.

The question raised was whether to rework the terrain renderer now: chunked
per-tile meshes with CDLOD geomorph under an `ExtendedMaterial`, replacing
UDLOD's shared-grid + height-atlas vertex displacement.

## Decision

Three parts.

**1. The tile becomes the surface contract; point queries are derived.**
Generation hands over multi-channel mipped tiles, not points. `SurfaceQuery`
survives for the small number of genuine point consumers (HUD altitude,
spawn-site search, propagator collision) but is *sampled from materialised
tiles*, not an independent evaluator. The tile seam is a superset of the point
seam — rasterising an analytic function over a tile footprint is what
`compute_tile_pixels` already does — so procedural bodies lose nothing. Filed as
**BL-34**.

The contract must answer "how do I get a tile the bake did not anticipate" as a
first-class case, not a fallback: package hit → ancestor upsample → client
synthesis → cache. Three producers, one format, indistinguishable downstream.

**2. The UDLOD → mesh/`ExtendedMaterial` rework is deferred, with an explicit
trigger.** It is *not* rejected; it is unschedulable today because its entire
justification rests on tiles coming from a baked package, and no production
package exists. Mira's artifact is a 512-face compatibility fixture, MIRA is at
the L2 gate, Thalos is still analytic, and two open items from the Terrain
Diffusion evaluation (latent vs. pixel storage, `B2` pruning depth) are
schema-level.

> **Trigger:** package schema frozen **and** one body producing real tiles from
> it. Then prototype on **Mira only** — airless, so Hapke-only, with no
> atmosphere, ocean, vegetation, flatten pads, or structures, and with
> `mira-orbit` / `mira-surface` / `mira-eva` already available as the
> verification harness. Thalos follows only if that slice holds.

A second precondition is attribution: the `verify` queue must be burned down
first (GF-CAL plus the standing `verify` rows). Reworking the most load-bearing
render subsystem on top of ten unverified landings makes every regression
un-diagnosable, which violates the bug-fixing rule in `CLAUDE.md`.

**3. Scale consistency is an architectural invariant, not a tuning goal.**
Every band below the package is a **conditional refinement of its parent; never
additive**. Approaching a surface reveals higher frequencies already implied by
the coarser levels — never independent content. `mira_airless_mvp.md` §3 already
states the generation-side half ("refinement adds bandwidth, not a new
skyline"); this records the renderer-side consequence: **the unconditioned
4 km-wrapped f32 shader detail noise is deleted, not tuned**, once `Rclient` can
replace it. It has no orbital counterpart and is the current violator.

Note this constrains *detail synthesis*, not *geometric LOD morphing*. Blending
vertex positions between tessellation levels so silhouettes do not pop is
mandatory and invisible; adding content that did not exist at distance is the
defect.

## Alternatives

**Rework the renderer now.** Rejected: builds a consumer against a moving
producer contract, and lands on top of an unverified stack.

**Keep the point-query seam.** Rejected: it forces raster → point → raster on
every package-backed tile, keeps GPU tile production artificially blocked, and
requires the generator to discard everything it knows beyond a few scalars.

**Continue unifying by porting crafts onto the `thalos::lighting` spine only
(status quo).** Retained for now — F6 landed this way and it works. But note
`ShadowedStandardMaterial` means the hybrid has *already* been chosen from the
craft side; the remaining question is narrowly whether terrain also becomes an
`ExtendedMaterial`, which is what the Mira prototype answers.

## Consequences

- BL-34 is independent of how tiles are *drawn*, so it pays off under either
  outcome of the deferred decision, and it removes the resampling round trip
  currently charged against the cold-stream budget.
- `terrain_lod_optimization.md`'s "GPU tile production is blocked" note becomes
  conditional on the analytic producer and should be re-stated when BL-34 lands.
- If Thalos also becomes package-backed, the case for the rework strengthens
  considerably and the trigger may fire sooner than planned.
- Observed tile popping — actually seen in a capture, not hypothesised — would
  promote geomorph from an architectural nicety to a bug fix with independent
  priority. That is a screenshot question and belongs in the verify sweep.
- The deferred fork is tracked in `backlog.md` → *Decisions pending*.
