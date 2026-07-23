# ADR-20260722T182853Z-far-clouds-preserve-prefiltered-areal-coverage: Far clouds preserve prefiltered areal coverage

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

The four-stratum surface-density cube established one body-fixed cloud
distribution for near and far projections, but the grazing/orbital consumer
still turned it into smooth opaque bands. It selected cubemap mip from the
spacing between six chord samples, accumulated filtered areal fractions as
repeated translucent slabs, and treated the broad surface envelope as finished
cloud even though the near tier still requires a local 3-D morphology hit.

Controlled cold captures isolated the real tiers. `cloud-cruise / cloud-tier`
showed discrete clouds in `near-only` and the entire band in `far-only`.
Additional far-only comparisons separated footprint choice from aggregation.

## Decision

- The far projection selects weather and density mips from the projected pixel
  footprint. Quadrature spacing controls integration accuracy, not image
  filtering, and must not blur multiple visible weather cells into one sample.
- Chord aggregation consumes the prefiltered areal mean. It does not stack
  samples or select the strongest encountered cell, because either operation
  makes opacity grow with chord length rather than projected coverage.
- The far areal mean is multiplied by the expected sub-cell morphology fill
  (`FAR_SUBCELL_MORPHOLOGY_FILL`). The surface cube is the broad selector and
  envelope; the independent near 3-D basis still decides realized shape inside
  it, so rendering the envelope directly overestimates occupied area.
- The near marcher continues to multiply its local morphology by the shared
  surface envelope. It adds fidelity inside the same occupied regions instead
  of selecting a second planet-scale distribution.

## Alternatives

- **Chord-spacing mip.** Rejected by the far-only filter A/B: it erased resolved
  weather cells into a uniform layer.
- **Strongest sample plus damped stacking.** Rejected by the far-only
  aggregation A/B: a single hit guaranteed coverage and grazing chords stayed
  opaque.
- **More chord samples.** Previously rejected: it does not repair the semantic
  mismatch and raises cost; direct periodic-density marching also aliases.
- **Global coverage reduction.** Rejected: it would change authored climate and
  near clouds instead of correcting the far estimator.

## Consequences

- Cruise, limb, and planet projections retain filtered large-scale structure
  without thickening solely with distance; runway retains its 3-D morphology.
- The morphology-fill constant is an explicit calibration point until a richer
  offline optical-depth/moment producer can encode the realized sub-cell fill.
- The remaining user verification is temporal: fly through the handoff and
  confirm no motion-visible pop. Static headless captures are accepted.
