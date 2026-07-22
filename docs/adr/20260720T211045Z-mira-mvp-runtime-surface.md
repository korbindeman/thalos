<<<<<<<< HEAD:docs/adr/20260720T211045Z-mira-mvp-runtime-surface.md
# ADR-20260720T211045Z-mira-mvp-runtime-surface: Mira's airless MVP is a runtime `SurfaceQuery`

- **Status:** Superseded by ADR-20260720T211046Z-offline-terrain-packages
- **Date:** 2026-07-20

> **Superseded 2026-07-20:** the product direction explicitly requires offline
> diffusion baking and adaptive terrain packages. ADR-20260720T211046Z-offline-terrain-packages retains the one
========
# ADR-20260720T211045Z-mira-mvp-runtime-surface: Mira's airless MVP is a runtime `SurfaceQuery`

- **Status:** Superseded by ADR-20260720T211046Z-offline-terrain-packages
- **Date:** 2026-07-20

> **Superseded 2026-07-20:** the product direction explicitly requires offline
> diffusion baking and adaptive terrain packages. ADR-20260720T211046Z-offline-terrain-packages retains the one
>>>>>>>> origin/main:docs/adr/20260720T211045Z-mira-mvp-runtime-surface.md
> `SurfaceQuery` consumer path but replaces this ADR's runtime-only producer
> decision.

## Context

The airless-terrain research proposes a bake-first `PlanetTerrainPackage` with
offline cubemap generation and runtime local residuals. Since that proposal was
written, Thalos removed its bake/disk-artifact pipeline. The live terrain
system now generates UDLOD tiles at runtime through `SurfaceQuery`, and every
height consumer is required to read that same surface.

Mira currently has `TerrainConfig::None`. Its former
`Feature(AirlessImpactMoon)` config only selected the generic runtime
`ProceduralSurface`, which ignored the feature config and gave Mira Thalos's
continental/ocean generator. The old airless compiler still contains useful
morphometry, but the compiler and `PlanetSurface` path are dead code scheduled
for removal.

## Decision

Implement the Mira airless MVP as a new body-typed runtime `SurfaceQuery`
backing selected by one canonical per-body surface factory.

The airless surface owns both its finite macro-crater catalogue and its
random-access local crater/regolith field. Their diameter hand-off is internal
to one `sample(direction, lod)` operation. The same shared surface instance is
registered for UDLOD, map/impostor projections, near-surface height queries,
and canonical terrain collision.

Mira is the first authored instance of the airless family, not a hard-coded
generator branch. Offline packages, diffusion, and learned residuals are
deferred. If adopted later, a package is another `SurfaceQuery` producer, not a
second renderer/physics path.

## Consequences

- The MVP reuses the existing UDLOD paging, tile cache, renderer, and physics
  seams instead of first building a parallel asset pipeline.
- The body terrain schema gains a compact live airless variant and a stable
  parameter/cache fingerprint.
- Repeated concrete `ProceduralSurface::new(radius, body.id)` construction is
  replaced by one N-body factory/registry, improving Thalos as well as Mira.
- Pure crater-profile math may be extracted from the legacy compiler, but no
  live dependency on its stages, baked cubemaps, or `PlanetSurface` is allowed.
- The MVP does not prove package formats, compression, offline seam belts, or a
  neural producer. Those remain a later producer/backend decision.

## Alternatives rejected

1. **Implement the vault package MVP first.** Rejected for this slice because it
   reverses the project's no-bake decision and duplicates UDLOD's existing
   random-access runtime provider before Mira has proved its geology.
2. **Re-enable `Feature(AirlessImpactMoon)`.** Rejected because that compiler is
   a superseded path and does not currently author the runtime surface.
3. **Branch on Mira inside `ProceduralSurface`.** Rejected because it preserves
   the bare body-ID coupling, makes one generator own incompatible planetary
   families, and fails the project's N-by-default rule.
