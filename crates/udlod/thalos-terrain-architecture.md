# Thalos: Planetary Terrain Rendering Architecture

**Status:** design draft, v2 (post-research)
**Scope:** rendering only — terrain synthesis is treated as an opaque dependency with a defined interface

## Overview

This document specifies the rendering stack for Thalos and the rest of the Pyros System. It is the second iteration of this design after researching Kurt Kühnert's repository ecosystem; the first iteration overestimated the fork scope and partially mis-attributed responsibilities between libraries.

The stack is built on three layers:

1. **`big_space`** — system-scale world coordinates and floating-origin precision for inter-body distances.
2. **Forked `thalos_udlod`** — the library that owns terrain rendering: UDLOD, ellipsoidal cubesphere model, Taylor-series GPU precision, tile storage and sampling, big_space integration. We fork it only to add a runtime tile provider trait.
3. **Existing terrain generation pipeline** — feature-based synthesis of height and material data, exposed to the renderer through the `TileProvider` interface defined here.

These layers solve non-overlapping problems and compose cleanly. `big_space` doesn't care about terrain; `thalos_udlod` doesn't care about world coordinates beyond the optional `high_precision` integration; the synthesis pipeline doesn't care about either, only about producing tile data for given coordinates.

## Goals and non-goals

### Goals

- Seamless camera traversal from orbital altitude to surface walking scale (~cm features) on any body in the Pyros System, with no LOD popping or precision artifacts.
- Distant terrain rendered at appropriate LOD with screen-space-error-driven triangle distribution (UDLOD).
- Per-body parameterization: every body has its own ellipsoid shape and height range, including realistic oblateness for fast rotators.
- Procedural-first: no baked planet-scale heightmaps. The terrain pipeline is queried on demand at the resolution required by the current LOD.
- Compatible with the patched-conics orbital model: planet local frames inherit orbital and rotational motion via nested big_space grids.

### Non-goals (this doc)

- Terrain *synthesis* — the feature-based pipeline is treated as a black box. Out of scope.
- Atmospheric scattering, ocean rendering, gas giant impostors, or sun rendering. Separate subsystems even if they ultimately render alongside terrain.
- Asset streaming for non-terrain content.
- Physics interaction with terrain.

## Repository landscape (relevant context)

Kurt Kühnert authors several related repositories. Understanding the relationships saves a lot of confusion:

- **`thalos_udlod`** — the library. Pinned, actively developed (301 stars, ~34 forks). Contains the entire terrain rendering stack including UDLOD, Chunked Clipmap, three terrain models (planar / spherical / ellipsoidal), cubesphere projection, Taylor-series GPU precision approximation, optional `big_space` integration, multi-view rendering, debug tooling. **This is the fork target.**
- **`planetary_terrain_renderer`** — Master's thesis demo app (~21 stars). Uses `thalos_udlod` to render real-world Earth from GeoTIFF datasets. Has its own preprocess CLI tuned for georeferenced raster files, plus a polished example with debug controls. **Reference only.** Its `examples/spherical.rs` and debug controls are useful templates for our integration; we do not depend on or fork the app itself.
- **`terrain_renderer`** — Bachelor's thesis demo app for planar Saxony rendering. Older. **Not relevant.**
- **`dtm`** / **`bevy_dtm`** — custom QOI-like 16-bit image format for shallow heightmaps. **Not relevant** for procedural use; we synthesize tiles in memory.

The Master's thesis novelties (ellipsoid model, Taylor-series GPU precision) were merged into `thalos_udlod` itself during Kurt's two years of professional work at Argeo (Oct 2023 – Jul 2025), where he was paid to build production geospatial visualization on top of the library. This means the thesis tech is in well-tested production code, not a one-off thesis prototype.

## World coordinates: `big_space`

### Why (narrower than the v1 framing)

A solar system spans ~10¹² meters. f32 alone cannot represent both system-scale positions and surface-scale offsets. We need a precision strategy.

**Important: `thalos_udlod` already handles surface-scale precision on its own**, via Taylor-series approximation of ellipsoid coordinates relative to the viewer. Within the surface of a single planet, the renderer maintains GPU precision down to centimeter scale without needing big_space cells. This was a major correction from the first version of this doc.

big_space's actual role is therefore:
- **System frame** — barycentric, where bodies orbit.
- **Per-body local frames** — child grids that move with each body's orbital and rotational state. Surface-attached entities (terrain, ships on the ground, structures) parent to the body's grid and inherit its motion in high precision automatically.
- **Camera floating origin** — keeps f32 transforms accurate near the camera regardless of where it is in the system.

big_space does *not* need to provide nested cells for the surface of a single body. The terrain renderer handles that internally.

### Configuration for Pyros

- **Grid precision:** `i64`. Effectively unlimited range for a solar system; cheap.
- **Cell size:** ~1 km in the system frame. Sub-micrometer cell-local f32 precision near the camera, comfortably more than adequate for ship physics, surface objects, and anything else outside terrain.
- **Hierarchy:**
  - Root `BigSpace`: system inertial frame, origin at Pyros barycenter.
  - One `Grid` per orbiting body. Position updated each frame from patched-conics integrator. Rotation from spin state.
  - Terrain entity (the `thalos_udlod` setup for that body) is parented to the body's grid. It inherits orbital and rotational motion automatically.
  - The `high_precision` feature in `thalos_udlod` wires the floating origin into the Taylor approximation — the renderer knows where the camera is in big_space coords and computes its Taylor coefficients accordingly.

### Caveats

- Per `big_space` docs: prefer applying *deltas* to entity transforms over absolute positions. Setting absolute positions causes the floating-origin system to constantly re-center, fighting controllers. For deterministic motion (orbits), compute the position in high precision and write directly to `(CellCoord, Transform)` via `Grid::translation_to_grid`.

## Terrain rendering: forked `thalos_udlod`

### Why fork at all

Almost everything we need is upstream. The single missing piece is a way to feed runtime-synthesized tile data into the renderer instead of disk-loaded preprocessed tiles. The current `terrain_data` module assumes a preprocess pipeline that takes source data (GeoTIFF, etc.), runs a 3-step transformation, and writes per-tile textures to disk that the runtime then loads via Bevy assets. There is no exposed seam for "synthesize tile on demand."

The fork adds that seam — a `TileProvider` trait — and is otherwise minimal.

### What we keep from upstream (verified, May 2025 indexed main)

- **UDLOD triangulation.** GPU-driven, screen-space-error-based bintree subdivision, vertex-shader morphing for seamless LOD transitions.
- **Three terrain models, including `ELLIPSOIDAL`.** Per-body semi-major / semi-minor axes; min/max height range.
- **Cubesphere projection** with C_SQR distortion correction. Standard, mature.
- **`TerrainModelApproximation`** — second-order Taylor series coefficients computed CPU-side per frame and uploaded for use in GPU shaders to position vertices on the ellipsoid in viewer-relative coords with high precision. This is the master-thesis novelty that lets us avoid f64 on the GPU.
- **`TileAtlas` + `TileTree`.** GPU-side tile storage and hierarchical sampling with LOD blending.
- **Attachment system.** Multi-channel tile data with configurable resolution and format per channel: R16 (height), RG16 (normals), RGBA8 (albedo / splat / custom). 1px borders for seamless filtering.
- **`high_precision` feature** for `big_space` integration. Already wired to the camera and floating origin.
- **Multi-view rendering, custom material plugin system, debug visualization tools.** Mature and useful as-is.

### What we change in the fork

- **Add `TileProvider` trait.** The single architectural change. See "Integration boundary" below.
- **Refactor tile loading.** Decouple "where data comes from" from "how it's stored / sampled / rendered." The existing disk-loading code becomes one impl of `TileProvider` (named `DiskTileProvider` or similar) that we keep working as a regression check.
- **Bevy version bump.** Upstream's last main indexing was May 2025, likely on Bevy 0.16. We're on 0.18. Two version migrations, both mechanical, with available migration guides.

### What we don't fork

- We do not modify the UDLOD shader pipeline, the Taylor approximation math, the bind groups, or the rendering systems. The fork is data-source-only at first, which keeps merging upstream improvements feasible.

## Integration boundary: the `TileProvider` interface

This is the contract between rendering and synthesis. It is the only point of coupling.

### Contract

The `TileProvider` is a trait the renderer calls to obtain tile data. The renderer holds a `TileTree` per (terrain, view) pair and decides which tiles need to be resident based on camera position and LOD. When a tile becomes needed, the renderer asks the provider for its data; the provider returns texture data for each configured attachment.

```rust
// Approximate shape — exact API to be designed during Phase 2

pub trait TileProvider: Send + Sync {
    /// Request tile data for the given coordinate. May be async; the renderer
    /// will sample the parent LOD until the tile is ready.
    fn request_tile(
        &self,
        coord: TileCoordinate,
        attachments: &[AttachmentSpec],
    ) -> TileRequest;
}

// `TileCoordinate` is upstream — already exists in thalos_udlod, with face/lod/x/y.
```

The trait should be designed to allow async/eventual delivery; tile latency is not zero and the renderer must tolerate it gracefully (it already does, via parent-LOD fallback).

### What the provider must produce per tile

For each attachment configured on the terrain, an N×N texture in the attachment's format. Defaults from upstream:

| Attachment | Format | Default size | Purpose |
|---|---|---|---|
| `height` | R16 | 512×512 | Required. 16-bit unorm; 0..1 maps to `[min_height, max_height]` configured per body. |
| `normal` | RG16 | 512×512 | Optional. May be derived from height if not provided. |
| `albedo` | RGBA8 | configurable | Optional surface color. |
| `splat` | RGBA8 | configurable | Optional material weight masks for shader-side blending. |
| custom | configurable | configurable | Anything else the shader needs. |

Sizes are powers of two (upstream limitation: mipmap generation only supports POT).

### Border requirement

Each tile texture has a 1-pixel border that overlaps neighbors. Border values must match neighbors *exactly* (same float bits) so texture filtering across tile boundaries is seamless. This is automatic when the synthesis pipeline is a pure function of position.

### Determinism requirement

`request_tile(coord)` and `request_tile(child_of_coord)` must agree exactly on shared edges, regardless of evaluation order or other tiles requested. This is satisfied iff the synthesis pipeline is a pure function of (ellipsoid position, body parameters). The pipeline is feature-based and queryable at arbitrary resolution, which gives this for free as long as features are evaluated by position.

### Coordinate input to the provider

The provider receives `TileCoordinate` (face id, lod, x, y on the cubesphere). It is responsible for converting to ellipsoid position when sampling its synthesis pipeline. The provider should evaluate in 3D ellipsoid position rather than face-UV or lat/lon, both to avoid pole singularities and to make the determinism property easier to maintain.

The conversion math (cubesphere face/UV → ellipsoid position) is already implemented in upstream `thalos_udlod::math` and can be called from the provider.

### Latency

Tile requests are on the critical path of LOD streaming. The renderer falls back to parent LOD while a child is in flight, but persistent multi-second latency means the player sees low-detail terrain at close range.

Rough budget: tiles available within a few hundred ms of being requested at typical traversal speeds. Hard real-time is not required. If the synthesis pipeline can't meet this, the provider can layer caching (LRU in-memory, optional disk cache for visited regions) on top.

## Per-body configuration

Each body in the Pyros System needs a terrain configuration alongside its physical parameters in the existing RON solar system spec.

### Schema additions

```ron
// Conceptual; actual RON syntax to match existing spec style
Body(
    name: "Thalos",
    // ... existing physical params: mass, orbit, spin, etc.

    shape: Ellipsoid(
        semi_major_m: 6_378_137.0,
        semi_minor_m: 6_356_752.0,
    ),

    terrain: TerrainConfig(
        min_height_m: -11_000.0,
        max_height_m:   9_000.0,
        attachments: [
            Attachment(name: "height",  format: R16,    size: 512),
            Attachment(name: "normal",  format: RG16,   size: 512),
            Attachment(name: "splat",   format: RGBA8,  size: 256),
        ],
        synthesis_seed: "thalos-v1",
        // any other params the synthesis pipeline needs
    ),
)
```

### Computing oblateness

For physically-modeled fluid bodies (gas giants, fast-rotating rocky), oblateness can be derived from rotation rate:

$$f \approx \frac{\omega^2 R^3}{GM}$$ (fluid limit)

Gas giants in Pyros should use this — they should look visibly squashed. Rocky bodies use a smaller value or are authored directly.

### Bodies without terrain

Small bodies (asteroids, irregular moons) may skip this stack — at small scales a chunked mesh or single LOD-instanced model is more appropriate. The terrain stack is for bodies large enough that ellipsoid + LOD'd surface is the right primitive.

## Implementation phases

### Phase 1: stand up the rendering stack on a single body

- Fork `thalos_udlod`. Bring it forward to Bevy 0.18. Mostly mechanical.
- Get the upstream-equivalent of `planetary_terrain_renderer/examples/spherical.rs` running on the fork — disk-loaded GeoTIFF data, big_space camera, debug controls.
- Wire into the Thalos game project. Confirm `high_precision` feature works correctly with our big_space hierarchy.
- **Exit criterion:** can fly around a real-world Earth dataset using the fork in our project, with our big_space hierarchy.

### Phase 2: introduce `TileProvider`

- Add the trait to the fork's `terrain_data` module. Refactor the tile loading path to call through it. Provide a `DiskTileProvider` impl that replicates current behavior, so the spherical example still works as a regression check.
- Document the trait, the border requirement, and the determinism requirement.
- **Exit criterion:** spherical example still works via the new code path.

### Phase 3: connect the synthesis pipeline

- Implement `PipelineTileProvider` wrapping the existing terrain generation pipeline.
- In-memory LRU tile cache. Disk cache deferred unless latency demands it.
- Convert pipeline to evaluate in 3D ellipsoid position if it doesn't already.
- **Exit criterion:** Thalos rendering using synthesized data, with seam-free LOD transitions across tile and face boundaries.

### Phase 4: scale to the system

- Onboard remaining Pyros bodies with per-body terrain configs.
- Verify big_space hierarchy correctly handles multi-body scenarios under camera transitions.

### Phase 5: polish

- Profile and tune latency, cache sizes, LOD thresholds.
- Material shader work for surface appearance.
- Re-evaluate non-power-of-two tile sizes or other upstream limitations as they bite.

## Open questions

These remain undecided.

1. **Async tile request shape.** Bevy task pool? Custom worker pool? GPU compute path for some attachments? Profile-driven.
2. **Cache strategy.** In-memory only is simplest; disk cache becomes worth it if synthesis latency is high or if we want consistent reload performance. Defer until measured.
3. **CPU vs GPU synthesis split.** This affects the provider interface (does it hand back CPU buffers or GPU textures already on the device?). Worth resolving before Phase 3.
4. **Surface-scale detail beyond tile resolution.** At highest LOD, even cm-scale features outrun synthesized tile resolution. Likely solved via shader-side detail textures and triplanar mapping, parameterized by splat masks. On the roadmap, out of scope here.
5. **Body-to-body LOD handoff.** Camera transitioning from interplanetary to surface scale of a target body — what's the seamless handoff? Probably a separate "approach" mode that progressively refines the target body's tile residency.
6. **Shadow casting from terrain.** Cascaded shadow maps with terrain-LOD-aware cascades is the likely answer. Bevy 0.18's atmospheric scattering improvements may help frame this.

## Fork roadmap and upstream relationship

The fork is intended to be long-lived but not divergent. Goals:

- Track upstream Bevy version bumps by rebasing our changes onto upstream's migrations.
- Pull in upstream rendering improvements (mipmap fixes, tile loading efficiency, etc.).
- Eventually upstream the `TileProvider` abstraction itself if Kurt is interested. It's a generally useful capability not specific to Thalos — production geospatial users would benefit from runtime-synthesized tiles too. Worth at least opening a Discord conversation before we start the fork.

We keep the diff against upstream small and well-organized. The fork is named and tagged so the relationship is clear.

---

*Doc owner: Korbin. Last updated: post-research, pre-implementation.*
