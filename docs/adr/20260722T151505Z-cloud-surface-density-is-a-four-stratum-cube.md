# ADR-20260722T151505Z-cloud-surface-density-is-a-four-stratum-cube: Cloud surface density is a four-stratum cube

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

ADR-20260722T141000Z requires CLOUD-6's planet-wide density to be
surface-parameterized and shared back into the near volume. The existing
canonical weather cube already owns coverage, type, base, and top, so the new
contract must extend that authority without creating a far-only noise field.

Three storage shapes were considered. Packing more information into the
weather RGBA channels would either discard typed vertical data or require
interpolation-unsafe bit packing. A phase-only cube could decorrelate the near
Cartesian volume, but a second producer would still be needed for far optical
density. A surface-density cube can be produced beside weather from the same
climate evaluation and consumed directly by both projections.

## Decision

- `CloudWeatherField` owns a second RGBA8 cubemap payload. Its four channels
  are broad, dimensionless cloud density at four normalized-height strata.
  It is not an independent weather resource.
- Weather and surface density share face size, mip count, version, body-fixed
  direction convention, and one upload/synchronization path.
- The producer evaluates only seamless direction-space noise plus the
  canonical coverage/type/base/top profile. It never samples the periodic
  Cartesian near-volume tile.
- The near marcher samples the surface field as the selector/envelope for its
  local Perlin/Worley morphology. The local 3-D volume remains responsible for
  sub-cell shape and erosion.
- Orbital and grazing consumers sample the same height strata with footprint
  mips. Those samples replace analytic weather-column occupancy; weather
  remains responsible for climate, typed optical properties, and lighting.
- A clear body binds one shared zero cube for both weather and surface density.

## Alternatives

- **Repack weather RGBA.** Rejected: type/base/top are live canonical inputs,
  and filtering packed integer subfields would corrupt them.
- **Add only a phase/seed cube.** Rejected: it conditions near density but does
  not supply a density-derived far representation, leaving parallel producer
  work and ownership.
- **Store a 3-D spherical volume texture.** Rejected for this slice: the
  authored shell has modest vertical complexity, while a four-stratum cube is
  naturally mip-filterable, much smaller, and directly addresses the observed
  planet-scale Cartesian repetition.
- **Generate a far-only moment atlas.** Rejected by the one-density contract;
  the source must affect the near volume before it can replace the far tier.

## Consequences

- Persistent cloud memory gains one RGBA8 cubemap mip chain. Its exact cost is
  reported by `cloud_target_memory` and the screenshot probe.
- Near density pays one filterable cubemap sample per broad/full probe. This is
  accepted provisionally and must pass the existing runway/cruise timing gates.
- Four vertical samples are a foundation rather than the final CLOUD-6 moment
  set. Albedo, normal, and representative-height moments can be derived from
  the shared strata without changing weather ownership.
- The implementation must be rejected if cold planet/limb captures show cube
  seams, repeated bands, or if local coverage/timing leaves the accepted
  BL-33 envelope.
