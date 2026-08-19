# ADR-20260818T211204Z: Tile terrain uses shared GPU-displaced patches

- **Status:** Accepted
- **Date:** 2026-08-18
- **Related:** ADR-20260723T142945Z-neural-terrain-standard-renderer-keystone ·
  ADR-20260814T152332Z-larger-tiles-reduce-streaming-entity-cost ·
  ADR-20260814T201228Z-broad-terrain-shadows-use-coarse-caster-twins

## Context

The extracted renderer put terrain on Bevy's standard material, lighting,
visibility, prepass, and shadow paths, but represented every resident tile as a
unique 129² mesh. A settled `forest-stand` view held roughly 1,500 such assets.
Controlled density tests attributed most of the remaining base-terrain cost to
geometry traversal, while the CPU mesh copies also consumed about 1 MiB per
tile and prevented automatic instancing.

The selector, hole-free replacement rule, tile cache, and CPU height mirror are
sound. Replacing them with a clipmap would widen a geometry problem into a
streaming and surface-authority rewrite.

## Decision

Keep the cube-sphere quadtree, streaming lifecycle, and CPU height mirror.
Replace unique raster meshes with three shared 33², 65², and 129² patch meshes.
Each resident tile owns one layer in a fixed array atlas:

- `Rgba32Float` stores exact body-local positions built in f64, ecological
  altitude, the skirt curtain, and the precision anchors;
- `Rgba8Unorm` stores linear macro albedo and canopy coverage;
- Bevy's `MeshTag` carries the atlas slot and current/previous geomorph factors
  through ordinary per-instance mesh data.

The visible material and the bare terrain-caster material use the same vertex
shader for main, prepass, deferred, and shadow pipelines. The main pass can
therefore retain Equal-depth testing without displacement disagreement.

Patch density is selected from projected tile span with hysteresis. Transitions
are adjacent 33↔65↔129 CDLOD morphs: the finer patch first collapses onto the
next-coarser sample lattice, then reveals its extra samples; downgrade reverses
that path. Current and previous morph factors are both packed into `MeshTag` so
motion vectors see the geometric transition.

The atlas has 2,048 layers. Steady-state selection budgets against 1,792; the
remaining 256 are replacement headroom because a merge target must land before
its fine children may retire. Slot reservation happens at task admission and is
released on cancellation, retirement, cold-start reset, and body handoff.

## Alternatives

- **Keep unique meshes and tune constants.** Rejected. Measurement isolates
  geometry traversal and per-tile assets as the structural floor.
- **Adopt a camera-centred clipmap.** Rejected for this slice. It replaces the
  already-correct selector, authored refinement floors, cache identity, and
  hole-free handoff to solve a representation problem.
- **Reconstruct cube-sphere positions from height in f32.** Rejected. At
  multi-megametre radii it reintroduces the precision drift the f64 tile origin
  removed and can disagree with the CPU height authority.
- **Switch patch meshes without morphing.** Rejected. It makes topology changes
  visible and violates ADR-20260722T105147Z's scale-consistency invariant.

## Consequences

- Resident geometry payload falls from about 1,008 KiB to 343 KiB per tile;
  the atlas allocation is fixed at about 687 MiB and the three meshes are
  shared by every entity and shadow view.
- Tiles sharing mesh, material, and pipeline become eligible for Bevy's normal
  automatic batching; the per-instance atlas address remains distinct.
- The CPU still materialises authoritative tiles and publishes their heights.
  GPU displacement changes presentation ownership, not surface authority.
- Solari cannot execute this vertex displacement while building a BLAS. Any
  future terrain RT proxy remains separate geometry extracted from the same CPU
  tile payload and must be budgeted independently.
