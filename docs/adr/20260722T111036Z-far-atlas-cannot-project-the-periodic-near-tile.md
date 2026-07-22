# ADR-20260722T111036Z-far-atlas-cannot-project-the-periodic-near-tile: Far atlas cannot project the periodic near tile

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

ADR-20260722T102639Z established that the far cloud tier needs a prefiltered
density-derived representation. BL-33 then tested the direct implementation:
a GPU compute pass evaluated the exact broad-density function used by the near
marcher over a cubemap, integrated eight vertical samples, and emitted either
optical-depth/height moments or four vertical optical-depth strata.

Cold cruise and limb brackets covered 256² and 512² faces, sparse and true
tangent-footprint filters, analytic height overlap, continuous mean/variance
reconstruction, and 6- versus 24-sample atlas chords. The atlas was stable as a
texture and every pipeline validated, but the visual result was not faithful.
Weak filtering exposed the near volume's periodic 8 km basis as horizontal
combs and then as a planet-scale checker/grid. Strong filtering suppressed the
repeat only by averaging the field back into the smooth slab the atlas was
meant to replace.

## Decision

- The far-density producer may share weather, typed vertical profiles,
  formation thresholds, density response, and lighting semantics with the
  near volume, but it may not map the periodic near-volume tile verbatim over
  planet-fixed positions.
- The far producer must use a non-periodic or deterministically
  phase-decorrelated density basis conditioned by the canonical weather field.
  This is a projection of the same density *contract*, not a second weather
  authority.
- Optical-depth/albedo/normal/height moments and the reduced-detail limb tier
  are still required. Their producer must demonstrate both footprint stability
  and absence of planet-scale repetition before replacing the analytic
  weather-column fallback.
- The existing 6-sample weather-column band march remains the accepted
  fallback. The complete periodic-atlas experiment is reverted.

## Alternatives

- **Keep the weakly filtered atlas.** Rejected: it makes the periodic basis
  visible as combs or a regular planet-scale cell grid.
- **Increase the spatial filter until the grid disappears.** Rejected: the
  result converges back to a featureless slab and loses the desired towers.
- **Use more chord samples.** Rejected as a standalone fix: 24 samples removed
  polygon gaps but made the repeating atlas topology more legible.
- **Use the atlas only for a mean-height shell.** Rejected: it rendered broad,
  flat cutouts rather than a reduced volume.
- **Treat the far field as an independent procedural weather layer.** Rejected
  by ADR-20260720T212214Z; weather placement and evolution remain canonical.

## Consequences

- CLOUD-6 gains an explicit non-periodic-density producer task before storage
  format or compositor tuning.
- The near 64³ volume remains appropriate for local shape and erosion, but is
  no longer assumed to be a valid planet-scale atlas source by itself.
- Rejected captures remain under `artifacts/visual/runs/bl-33/step-6a/`
  through `step-6h/`; `step-4/` remains the accepted visual checkpoint.
