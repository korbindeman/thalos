# ADR-20260808T221912Z-atmosphere-and-ocean-mechanisms-use-spatial-adapters: Atmosphere and ocean mechanisms use spatial adapters

- **Status:** Accepted
- **Date:** 2026-08-08

## Context

Thalos and Kòrsou need recognizably related atmosphere and ocean rendering,
but they do not inhabit the same spatial model. Thalos renders rotating
planetary bodies through a floating-origin, scene-depth-aware custom material.
Kòrsou renders a local projected metre frame with Bevy's standard material
path, real-world coastline products, and a camera-centred displaced water
clipmap.

The initial Kòrsou implementation copied useful ocean wave logic and directly
constructed Bevy's Earth atmosphere. That proved the shared mechanism, but it
also created two wave clocks, two filtering implementations, and no durable
place to document why atmosphere projection differs. Hiding both applications
behind one universal renderer interface would erase real differences in
geometry, precision, coast data, and composition.

## Decision

`thalos_atmosphere` owns authored atmosphere projections. It provides
`AtmosphereBlock` for the custom planetary shaders and a concrete Bevy Earth
adapter for local planar worlds. Kòrsou uses the Bevy adapter with a local
density calibration; Thalos keeps `BodySkyMaterial` as its sole rocky-body
atmosphere implementation. The two adapters intentionally do not pretend to be
interchangeable backends.

`thalos_ocean` owns the mechanisms whose interfaces match in both
applications: `OceanState` projection, the canonical f64-reduced wave clock,
resolved surface-wave height/slope/crest, deterministic spectrum payload,
anisotropic footprint filtering, omitted-variance transfer, and coastal wave
attenuation. Its WGSL library is consumed by both water implementations.

Spatial adapters own what does not match:

- Kòrsou owns the camera-centred planar clipmap, vertex displacement,
  real-world signed coastline and coastal-property textures, and Bevy PBR
  integration.
- Thalos owns analytic planet-scale ray/sphere water, its canonical signed
  sea-height field, body-fixed precision frame, custom atmosphere/lighting
  composition, and planet render ordering. It currently consumes shared
  slope, crest, and omitted variance without changing the analytic horizon.

Local displaced geometry is allowed where bounded by the adapter. A
planet-scale water mesh is not.

## Alternatives

- **Keep the copied Kòrsou water and atmosphere code** — rejected because the
  already-divergent wave clocks and filters would continue to drift.
- **Move Kòrsou onto `thalos_runtime`** — rejected because simulation,
  gameplay, and planetary composition are outside the explorer's product.
- **Make one renderer trait cover planar, spherical, and future ellipsoidal
  terrain** — rejected because the second implementation demonstrates shared
  mechanisms, not shared topology. The speculative interface would centralize
  churn and obscure precision/coastline ownership.
- **Replace Thalos water with Kòrsou's planar clipmap everywhere** — rejected
  because Kòrsou's local geometry is better at nearby silhouette, while
  Thalos's analytic surface is the correct stable planet-scale authority.

## Consequences

- Improvements to wave state, phase precision, filtering, and resolved wave
  shape now reach both applications through one module.
- Each application can optimize or A/B its spatial adapter without forking the
  authored appearance mechanisms.
- Kòrsou remains a small compiler-enforced consumer of rendering leaves rather
  than a second Thalos runtime.
- Future ellipsoid/geoid rendering should add another explicit spatial adapter
  over these leaves. It does not require turning the current two adapters into
  a universal renderer first.
- `docs/rendering/atmosphere.md` and `docs/rendering/ocean.md` are the canonical
  live contracts; application READMEs link to them instead of restating the
  mechanism in full.
