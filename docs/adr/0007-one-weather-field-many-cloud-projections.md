# ADR-0007: Planet clouds use one weather field with regime-specific render projections

**Status:** Accepted
**Date:** 2026-07-20

## Context

Thalos had three descriptions of terrestrial clouds: constants owned by the
near-volume driver, an optional world-schema cloud block consumed by a dormant
sky shader, and separately selected reference images for orbital bodies. They
could disagree about whether a body had clouds at all, and they guaranteed a
visible authority swap between flight and orbit.

A planet-scale renderer still needs different representations at different
projected sizes. Marching the full near-camera volume for a body a few pixels
wide is wasteful, while a flat orbital shell cannot provide an aircraft flying
inside a cloud with extinction or parallax.

## Decision

- `thalos_world::CloudClimate` is the only authored terrestrial-cloud
  configuration. `clouds: None` means that body has no terrestrial clouds.
- Each cloudy body owns one body-fixed `CloudWeatherField` in environment
  state. Its seam-safe cube field carries coverage, cloud type, base height,
  and top height; renderers never invent a second weather pattern.
- The rendering mechanism lives in `thalos_body_render::clouds`. The game
  selects the active body/view and projects environment state into render
  inputs, but does not own cloud rendering algorithms.
- Near volume, orbital optical-depth layer, cloud-sun transmittance, and tools
  may use different cost-scaled projections, but all consume the same weather
  field and ultimately the same density definition.
- Cloud temporal reconstruction stays local to the cloud projection rather
  than waiting for whole-scene TAA.
- The initial weather producer is deterministic authored/procedural generation
  with advection hooks, not a fluid solver. A future simulation may replace the
  producer without changing the consumer contract.

The vendored `bevy-volumetric-clouds` fork is absorbed under body rendering and
keeps its upstream license. The old reference-image and dormant slab paths are
deleted rather than retained as alternative authorities.

## Consequences

- Surface, atmosphere, orbit, shadows, and environment response can converge
  without translating between independently authored cloud maps.
- Render regimes can evolve independently in cost and reconstruction quality,
  but their transitions must be tested for visual continuity.
- Painted equirectangular maps remain valid import/export tools; they are not
  the runtime topology or an additional source of truth.
- CLOUD-2 through CLOUD-8 build on this contract. Reintroducing renderer-owned
  weather or body-name-selected reference clouds requires a superseding ADR.

## Rejected alternatives

- **Keep the cloud renderer as a top-level crate.** This preserves short-term
  file layout but leaves atmosphere, terrain, impostors, and clouds with split
  mechanism ownership.
- **Keep equirectangular runtime weather.** It is easy to paint but concentrates
  samples and derivatives at the poles; cube topology matches existing
  body-fixed rendering infrastructure.
- **One render technique at every altitude.** It either makes flight clouds too
  cheap or orbital views far too expensive. Shared authority matters; identical
  projection does not.
- **Start with a fluid climate simulation.** Its complexity does not address
  the immediate density, lighting, temporal, and cross-regime failures.
