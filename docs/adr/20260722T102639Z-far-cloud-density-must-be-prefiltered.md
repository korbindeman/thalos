# ADR-20260722T102639Z-far-cloud-density-must-be-prefiltered: Far cloud density must be prefiltered

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

BL-33 extended the trustworthy near-volume handoff to 67.2 km, but cruise and
limb rays still traverse roughly 200 km or more of cloud shell beyond it. The
temporary CLOUD-6 far tier integrates six analytic weather columns. It is
stable and planet-scale, but it cannot reproduce the canonical 3-D density and
therefore reads as a smooth slab at grazing angles.

A direct reduced-detail alternative was tested: bind the exact generated 64³
Perlin/Worley basis to the far compositor, share the typed density-shaping
functions with the near marcher, filter erosion out, and evaluate 24
deterministic samples along the grazing shell chord. The cold cruise capture
did not produce faithful cells. Sparse point samples of the periodic 8 km
basis aliased into severe horizontal combs. Increasing the sample count enough
to resolve the chord would turn the far tier into the expensive horizon march
the architecture is intended to avoid.

## Decision

- The orbital/grazing far representation will be a slowly refreshed,
  footprint-prefiltered atlas of optical depth, albedo, normal, and height
  moments derived from the canonical weather and density function.
- A reduced-detail low-orbit limb representation may sample the canonical
  density, but only with an explicit band-limit/prefilter appropriate to its
  projected footprint. It may not point-sample the periodic near-volume basis
  at multi-kilometre intervals.
- The existing analytic weather-column band march remains the temporary
  fallback in partition-of-unity with the near march until the moment atlas is
  available. Its slab-like grazing result is a tracked CLOUD-6/BL-33 residual,
  not an accepted final look.
- Near reach may grow only within the measured step budget or from a true
  conservative interval bound under
  ADR-20260721T033055Z-cloud-skips-require-conservative-bounds. It does not
  replace the far representation.

## Alternatives

- **Direct 24-sample march of the exact 3-D basis.** Rejected by cold capture:
  deterministic grazing samples alias the periodic volume into horizontal
  combs.
- **Raise the direct far sample count until the combs disappear.** Rejected
  because a ~200 km chord would need near cell-scale sampling, making the far
  tier an unbudgeted full horizon raymarch.
- **Average or blur the sampled density in the compositor.** Rejected because
  it hides aliasing by changing the threshold-selected density distribution
  and still provides no stable footprint hierarchy.
- **Keep the weather-column band march as the final tier.** Rejected because it
  shares weather but not the density definition and visibly fails the
  cruise/limb fidelity target.
- **Delete the far tier and fade the near volume out.** Rejected because it
  recreates the planet-scale disappearance and surface/orbit mismatch BL-33 is
  intended to remove.

## Consequences

- CLOUD-6 needs explicit producer/storage/update work for the density-moment
  atlas rather than another compositor-only tuning pass.
- The current stable fallback and clean planet projection are preserved while
  that work proceeds.
- The rejected live-march artifact remains under
  `artifacts/visual/runs/bl-33/step-5/` as regression evidence; none of its code
  is retained.
- Future far-tier proposals must demonstrate footprint stability in cold
  cruise, limb, and planet captures before replacing the fallback.
