# ADR-20260722T135123Z-weather-phase-warp-does-not-make-a-periodic-cloud-basis-aperiodic: Weather phase warp does not make a periodic cloud basis aperiodic

- **Status:** Accepted
- **Date:** 2026-07-22
- **Supersedes in part:** ADR-20260722T111036Z's phase-decorrelation option

## Context

ADR-20260722T111036Z rejected a far atlas that projected the near marcher's
periodic 64³ source tile verbatim and allowed either a non-periodic or a
weather-conditioned phase-decorrelated producer. BL-33 then tested the latter
interpretation in isolation.

The canonical density's broad and 21.6 km formation domains were displaced by
continuous weather coverage/base/top channels; cloud type was excluded so
categorical boundaries could not pop the volume. Matched cold runway and cruise
captures retained coverage and reduced the scanline contour proxy, so the warp
is useful local anti-tiling. A 512² RGBA16F cubemap was then baked from that exact
density using a 3×3 footprint, eight height samples grouped into four optical-
depth strata, and a 24-sample far chord.

The cold `cloud-planet` gate failed catastrophically: the disc exposed long
families of repeated curved/diagonal combs. The phase field bent the tile's
repetition but did not remove its spatial frequency. The atlas implementation
was fully reverted; the independently verified near-density warp remains.

## Decision

- A weather-conditioned coordinate displacement of one periodic 3-D source is
  not an acceptable CLOUD-6 producer. “Phase-decorrelated” now means a basis
  whose combined repeat distance exceeds the planet-scale projection, not a
  smooth warp of a single repeat.
- Before another far-atlas implementation, the shared density must gain a
  genuinely non-repeating broad basis. The next bounded candidate is a convex
  crossfade between independently transformed, incommensurate-period 3-D
  domains, selected by a low-frequency field and verified first in runway and
  cruise captures.
- The six-sample weather-column band march remains the far fallback until that
  source passes both local coverage/banding gates and planet/limb repetition
  gates.

## Alternatives

- **Tune the phase amplitude.** Rejected: amplitude changes bend or tighten the
  combs but cannot remove the source's fundamental repeat.
- **Filter the phase-warped atlas more strongly.** Rejected by the earlier
  bracket: filtering enough to hide repetition converges to the flat slab.
- **Keep the atlas and mask the stripes in reconstruction.** Rejected: the
  artifact is in the producer's spatial spectrum, not a shading-only defect.
- **Add an unrelated far noise texture.** Rejected: it would violate the one-
  density contract unless the same source also participates in near density.

## Consequences

- The accepted local density now has continuous weather phase offsets on its
  broad and formation domains.
- CLOUD-6's next experiment is a shared aperiodic-basis change, not another
  storage-format or chord-sampling bracket.
- Rejected evidence is retained at
  `artifacts/visual/runs/bl-33/step-8a/cloud-planet.png`; rollback proof is at
  `artifacts/visual/runs/bl-33/step-8-rollback/cloud-planet.png`.
