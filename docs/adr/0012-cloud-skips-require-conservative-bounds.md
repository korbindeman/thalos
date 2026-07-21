# ADR-0012: Cloud ray leaps require conservative density bounds

**Status:** Accepted
**Date:** 2026-07-21

## Context

CLOUD-3 needed to keep a 50 km spherical volume inside a 3.5 ms 1440p High
budget. Its first completion used three heuristic empty-space levels: a dilated
weather maximum, an estimated local base/top crossing, and a broad-shape
occupancy proxy. These reduced density fetches but were not upper bounds on the
actual typed, eroded density over the skipped interval. At cruise grazing
angles their nonuniform resume positions posterized clouds into stable
horizontal shelves (INC-0011).

Removing every leap restored continuous density but initially made the densest
sunset probe too expensive. The budget could instead be met by reducing
redundant frequency work without changing view-ray sample positions.

## Decision

- A cloud view ray may leap over an interval only when a representation stores
  a conservative upper bound on density for that complete interval, such as a
  max-density mip/hierarchy built from the canonical density field.
- Weather coverage, cloud base/top, and correlated noise proxies may gate
  density work at the current sample, but they may not choose a later resume
  position by themselves.
- Until a conservative hierarchy is justified by measurements, the near
  volume keeps uniform adaptive ray cadence, fades only sub-pixel fine erosion
  with range, and reuses deliberately low-frequency modulation per short ray.
- Meeting the measured budget without empty-space leaps closes CLOUD-3; a
  conservative hierarchy is an optional future optimization, not required
  infrastructure.

## Alternatives

- **Retune the existing leap thresholds/distances.** Rejected because a more
  cautious heuristic is still not a bound and can fail under another weather
  field, shell curvature, or camera angle.
- **Keep the base/top leap only.** Rejected because the envelope-only A/B was
  sufficient to reproduce the posterized strata.
- **Disable skipping and accept the regression in cost.** Rejected because the
  unoptimized sunset exceeded the program budget.
- **Build a max-density hierarchy now.** Deferred because cadence-preserving
  far-detail LOD plus per-ray macro reuse measures 2.471 ms mean / 2.476 ms p95
  at 2560×1440 High; extra volume ownership and update cost are not justified.

## Consequences

- Grazing-angle density remains continuous and the posterized hierarchy class
  is structurally removed.
- CLOUD-3 meets its GPU budget without another persistent 3-D allocation.
- Some clear-air rays do more ordinary steps than a correct future hierarchy
  would require.
- Any future empty-space accelerator must prove its bound and ship with matched
  cruise/runway temporal-off A/B captures as well as timings.
