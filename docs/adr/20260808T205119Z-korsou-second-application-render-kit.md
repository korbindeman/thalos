# ADR-20260808T205119Z-korsou-second-application-render-kit: Kòrsou is a second application over reusable rendering leaves

- **Status:** Accepted
- **Date:** 2026-08-08

## Context

Kòrsou began as a separate lightweight Bevy explorer for flat, projected
real-world Curaçao terrain. It deliberately omits Thalos's physics, orbital
simulation, gameplay, and canonical runtime, but it independently needs the
same class of atmosphere, water, terrain appearance, diagnostics, and capture
mechanisms. Keeping it outside the workspace encouraged copied rendering code;
making it a `thalos_runtime` scenario would destroy the product distinction and
would not test whether the rendering architecture is actually reusable.

Thalos also intends to support several spatial models: local planar maps,
cube-sphere planetary terrain, and eventually Earth-scale ellipsoidal terrain
with explicit horizontal and vertical datums. Those models should share visual
mechanisms where their inputs and outputs agree, but they should not be hidden
behind a universal renderer interface before two implementations establish a
real seam.

## Decision

Kòrsou lives in this workspace as `apps/korsou`, a permanent second application
that does not depend on `thalos_runtime`, simulation, or gameplay crates. The
canonical Thalos game and capture host continue to share the complete runtime.

Reusable rendering mechanisms are extracted into state-in/pixels-out leaf
crates when there are at least two concrete callers. The first extractions are
`thalos_atmosphere`, whose authored planetary block and concrete Bevy Earth
projection serve different spatial adapters; `thalos_ocean`, whose authored
`OceanState` projection, canonical wave clock, resolved surface waves, filtered
slope field, and coastal attenuation feed both water adapters; and
`thalos_vegetation`, whose procedural woody meshes and foliage atlases feed both
Thalos's cube-sphere scatter adapter and Kòrsou's planar cell adapter.
Placement, streaming, spatial precision, coastline data, shadows, and LOD
topology remain with each adapter. Kòrsou also participates in the machine-wide
renderer lease and provides its own headless visual capture path.

Spatial topology remains in application-appropriate adapters. Planar projected
terrain, cube-sphere tiles, and a future ellipsoid/height-datum adapter may
share appearance data, payload formats, caches, and diagnostics, but do not
implement a speculative all-purpose `Renderer` trait. New seams are extracted
from demonstrated duplication or matching contracts.

## Alternatives

- **Keep Kòrsou in a separate repository and consume Thalos crates by path or
  git revision** — rejected because private unstable crate edges, duplicated
  lockfiles, and unsynchronized changes make joint refactors and compile-time
  dependency guarantees unnecessarily fragile.
- **Make Kòrsou a `thalos_runtime` scenario** — rejected because the runtime's
  simulation and gameplay are intentionally outside the explorer's product and
  would preserve only one composition root rather than proving reuse.
- **Move all rendering behind one renderer trait first** — rejected because
  planar, cube-sphere, and ellipsoidal topology have materially different
  streaming and precision constraints. A trait designed before the second
  adapter exists would encode guesses and centralize churn.
- **Copy selected Thalos implementations into Kòrsou** — rejected because the
  ocean-spectrum implementation had already drifted into two copies, proving
  that lessons alone do not preserve a shared appearance contract.

## Consequences

- Thalos now has two application compositions but still one canonical game
  runtime and one acceptance capture path for that runtime.
- Rendering leaves must remain usable without simulation or gameplay imports;
  Kòrsou is the compiler-enforced second caller for that property.
- Kòrsou may stay direct and lightweight. Reuse is earned mechanism by
  mechanism, not imposed as a framework migration.
- Atmosphere and ocean adapter ownership is recorded in
  ADR-20260808T221912Z-atmosphere-and-ocean-mechanisms-use-spatial-adapters;
  `docs/rendering/atmosphere.md` and `docs/rendering/ocean.md` are their live
  contracts.
- Real-world Earth support still requires an explicit ellipsoid, coordinate
  reference system, geoid/vertical datum, and terrain adapter; this decision
  does not conflate those concerns into the rendering layer.
