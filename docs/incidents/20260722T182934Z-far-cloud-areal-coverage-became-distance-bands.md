# INC-20260722T182934Z-far-cloud-areal-coverage-became-distance-bands: Far cloud areal coverage became distance bands

- **Status:** Fixed
- **Date:** 2026-07-22
- **Severity:** visual
- **Surface:** cruise, limb, and orbital cloud projection

## Summary

Clouds were discrete 3-D bodies nearby but became smooth, thick bands with
distance. The near distribution was healthy; the far surface projection blurred
resolved cells, accumulated areal fractions as depth layers, and omitted the
near tier's sub-cell morphology probability. Far filtering and opacity now
preserve the same projected areal semantics as the near field.

## Evidence

- Corrected `cloud-cruise / cloud-tier` cold captures: near-only contained
  separated puffs; far-only contained the entire slab; composite overlaid it.
- `cloud-far-filter` far-only A/B: projected-pixel mip restored coherent weather
  cells that chord-spacing mip erased.
- `cloud-far-aggregation` far-only A/B: prefiltered mean removed opacity growth
  from strongest-hit/stacked samples.
- Final cold `cloud-runway`, `cloud-cruise`, `cloud-limb`, and `cloud-planet`
  captures were complete and free of shader/pipeline/capture-health errors.

An early set of cloud comparisons reported only ~0.1/255 MAE and was invalid:
the typed environment keys existed in the comparison runner but were absent
from the runtime `CAPTURE_OVERRIDE_KEYS` allowlist, so every variant rendered
the default. Adding the keys produced the expected large, localized tier
differences.

## Hypotheses considered

- **Near volumetric integration caused the band:** rejected by tier isolation.
- **Too few far chord samples:** rejected; sample count did not address the
  distribution semantics and prior high-sample experiments aliased.
- **Far mip footprint:** confirmed as one cause.
- **Areal samples treated as independent opacity layers:** confirmed as one
  cause.
- **Surface envelope treated as completed cloud:** confirmed by the remaining
  far/near density mismatch after filtering and mean aggregation.

## Root cause

Three independent quantities were conflated: chord sampling cadence, projected
pixel footprint, and cloud coverage. Cadence selected an excessively coarse
mip; then strongest-hit/stacked aggregation made coverage increase with path
length; finally the far estimator skipped the near tier's 3-D morphology fill.

## Fix and prevention

- Select far mips from projected pixel footprint.
- Aggregate the prefiltered areal mean and apply the near morphology fill.
- Keep the shared surface density as the near selector/envelope.
- Every new typed comparison axis must add its environment key to
  `CAPTURE_OVERRIDE_KEYS`; invariant diagnostic overrides must be recorded in
  the comparison manifest. Near-zero differences are not evidence that an axis
  is ineffective until the variant value is proven to reach the runtime.
