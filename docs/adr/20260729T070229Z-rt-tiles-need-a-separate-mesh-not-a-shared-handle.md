# ADR-20260729T070229Z-rt-tiles-need-a-separate-mesh-not-a-shared-handle: terrain enters the RT scene as duplicated geometry, and only near the view

- **Status:** Accepted
- **Date:** 2026-07-29
- **Supersedes:** part 2 of [ADR-20260724T224242Z](20260724T224242Z-solari-scene-half-not-lighting-half.md), for terrain tiles only. That ADR's decision — take `RaytracingScenePlugin`, never `SolariLightingPlugin`; RT is a visibility service inside our shading — stands unchanged.

## Context

ADR-20260724T224242Z part 2 specified how surfaces enter the RT scene:

> each tile and craft part gets a lightweight proxy sharing the *same mesh
> handle* (one BLAS, no duplicated geometry)

Starting NTR-RT3 on the geometry layer showed the tile half of that sentence is
not achievable. The premise it rests on — that the gap between our meshes and
Solari's requirement is a *missing* TANGENT — is only half the gap.

`bevy_solari-0.19.0`'s `is_mesh_raytracing_compatible` (`scene/blas.rs:173`)
compares the mesh's whole attribute sequence for **equality**:

```rust
let vertex_attributes = mesh.attributes().map(|(attribute, _)| attribute.id).eq([
    Mesh::ATTRIBUTE_POSITION.id, Mesh::ATTRIBUTE_NORMAL.id,
    Mesh::ATTRIBUTE_UV_0.id,     Mesh::ATTRIBUTE_TANGENT.id,
]);
```

`Mesh` stores attributes in a `BTreeMap` keyed by id (POSITION 0, NORMAL 1,
UV_0 2, UV_1 3, TANGENT 4, COLOR 5), so the sequence is sorted and the test is
well-defined — and **an extra attribute disqualifies a mesh exactly as a missing
one does.**

The two Thalos surfaces land on opposite sides of that:

| Surface | Attributes written | With TANGENT added | RT-eligible? |
|---|---|---|---|
| Craft part (`thalos_shipyard` meshers) | POSITION, NORMAL, UV_0 | `{0,1,2,4}` | **Yes** |
| Terrain tile (`tiles::build_tile_mesh`) | POSITION, NORMAL, UV_0, UV_1, COLOR | `{0,1,2,3,4,5}` | **Never** |

A tile's COLOR and UV_1 carry the NTR-X4 spare-channel contract that
`tile_terrain.wgsl` reads (albedo, forest band weight, wrapped body-fixed z,
ecological altitude). They are not decoration that can be dropped.

So terrain cannot share a handle with anything. And the failure is **silent** —
no warning, no BLAS, every ray simply misses. Had this been discovered on
hardware instead of in a test, the observable would have been a stainless hull
reflecting sky with no ground in it, and the natural conclusion would have been
"RT reflections don't look good", which is the wrong conclusion drawn from a
mesh format mismatch. ADR-20260724T224242Z anticipated this failure mode in the
abstract; it did not follow it through to the shared-handle claim.

## Decision

**1. Terrain tiles get a separate RT-only mesh asset; craft parts share theirs.**

- Craft: `part_mesh::add_raytracing_tangents` at the end of every shipyard
  mesher. One BLAS, no duplicated geometry, exactly as the prior ADR intended.
- Tiles: `rt::rt_twin_of_tile` builds a POSITION/NORMAL/UV_0/TANGENT twin from
  the visible mesh's own arrays at tile-build time. Deriving it from the raster
  geometry rather than re-evaluating the surface means a reflection can never
  disagree with the ground beneath it about where the ground is. Skirts are
  kept: at LOD boundaries they are what stops a reflection ray slipping through
  a crack and returning sky from under the surface.

**2. RT terrain coverage is scoped by radius around `ViewAnchor`, not by
residency — and this is a VRAM constraint, not a tuning knob.**

The twin costs **312 KiB** against the visible tile's 347 KiB, so an RT-covered
tile is **1.90×** a plain one. Covering the whole 4 GiB resident set would want
**another ~3.6 GiB**, against a budget that is already machine-wide and divided
between concurrent instances (INC-20260725T012104Z — the user routinely runs
two). Near-radius scoping is what makes RT terrain affordable at all.

This is also why the tangents on the tile twin are **gate filler**, not shading
data: Solari reads a tangent only when the hit material carries a normal map,
and the tile RT proxy's `StandardMaterial` has none. UV_0 on a tile is a planar
body-fixed projection, not a surface parameterisation, so the frame would not be
meaningful anyway. The same is true of craft parts, whose UV_0 is a constant
`[0, 0]`.

**3. NTR-RT3 runs before NTR-RT2, and carries NTR-RT1's measurement.**

The backlog order implied shadows first. Reflections are the cheaper consumer:
a roughness-0.08 hull needs terrain within a few hundred metres to a few km,
where RT sun visibility wants the full cascade range in the structure. Proving
the streaming gate on the smaller working set is the cheaper experiment, and if
it fails there it would certainly have failed at cascade range.

**4. Eligibility is a tested property of our meshers, not a GPU discovery.**
`rt::is_raytracing_eligible` mirrors the upstream predicate, and both surfaces
are held to it in tests — including one asserting the visible tile mesh is
*not* eligible, so if a future change makes it so, the separate twin is flagged
as dead weight rather than quietly kept.

## Alternatives

- **Move albedo/bands off the vertex stream into per-tile textures**, making the
  visible tile mesh eligible and restoring the shared handle. Rejected: it
  rewrites the NTR-X4 layer stack to save a mesh whose cost near-radius scoping
  already bounds, and the RT scene's `StandardMaterial` could not read those
  channels anyway — the ray hit would still need an approximating material.
- **A coarser RT twin** (decimate the 65² grid). Not rejected — deferred. It
  trades reflection fidelity at grazing angles for VRAM and BLAS build time, and
  it is the first knob to reach for if NTR-RT1's measurement is close. Keeping
  the twin geometry-identical for the first cut means any mismatch observed in a
  capture is a *shading* problem, not an LOD problem, which is the cheaper thing
  to debug first.
- **Patch `bevy_solari` to accept a superset of attributes.** Rejected: a
  vendored fork of a fast-moving experimental crate, to avoid a mesh we can
  build in twenty lines.

## Consequences

- `TILE_MESH_BYTES` is unchanged; `RT_TILE_MESH_BYTES` is a **second**
  denominator, and any RT residency budget must count both. The figures quoted
  above are asserted in `tiles`' tests so prose and code cannot drift.
- Craft part meshes gain TANGENT unconditionally, on all hardware including
  non-RT — ~16 B/vertex over a small mesh set, taken as negligible rather than
  gated, because a mesh whose format depends on a feature flag is a worse trap
  than the bytes are a cost.
- `thalos_shipyard` states the eligibility contract in its own words
  (`part_mesh::is_raytracing_ready`) because it is a dependency of the renderer
  and cannot import from it. The duplication is pinned by an agreement test in
  `rt.rs`.
- Still unmeasured, and still NTR-RT1's job: BLAS build/compaction throughput
  and per-frame TLAS rebuild cost against the near-radius proxy set. Nothing
  here adopts RT — it establishes that the geometry can exist and what it costs.
