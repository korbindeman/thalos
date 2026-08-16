# ADR-20260803T072800Z-procedural-tectonics-uses-grown-plates: Procedural tectonics uses irregular grown plates, not nearest-seed cells

- **Status:** Accepted
- **Date:** 2026-08-03
- **Decided by:** user, 2026-08-03 ("real plates don't look so voronoi ... take some fine details from [thalos_maps]")
- **Refines:** ADR-20260803T060232Z

## Context

The first tectonic structural prior put Euler-pole motion on the analytic
Worley cells already used by continentalness. Its process labels were correct,
but the matched map and height diff exposed the ownership geometry: broad
straight bisectors, sharp multi-cell junctions, and mountain sectors shaped
like polygons. Narrowing the fossil and active influence bands reduced the
amount of changed terrain but left the same boundary network visible. Width was
therefore not the root cause.

The sibling `thalos_maps` prototype avoids this by growing plates across a
spherical mesh. Growth cost varies by plate, direction, and spatial noise, and
small plates are injected into contested gaps. Its relief is then sampled and
smoothed rather than exposing nearest-region facets directly.

## Decision

`ProceduralSurface` tectonics uses a cached cube-sphere process field built by
weighted multi-source flood growth:

- nine major plates start from well-separated spherical seeds;
- each plate has a deterministic growth bias and preferred bearing;
- broad crustal fBm plus cell-scale edge noise perturb the growth cost;
- after 62 percent fill, three slow-growing microplates are seeded in contested
  gaps between major fronts;
- Euler-pole motion classifies the contacts, and a spherical distance solve
  carries those process labels into narrow fossil and active corridors;
- bilinear cross-face sampling hides the substrate cells from runtime terrain.

Continentalness remains a separate analytic field and the sole coastline
authority. This ADR supersedes only the nearest-Worley topology and the overly
broad 620/210 km influence widths in ADR-20260803T060232Z; its pre-diffusion
placement, process semantics, and coastline separation remain in force.

## Alternatives

- **Narrow the existing Worley bands.** Tested as the first comparison axis.
  It removed most of the oversized uplift but preserved the polygon network,
  so it was rejected as a symptom-level fix.
- **Increase domain warping around the Worley cells.** Rejected because it
  disguises straight bisectors without changing nearest-seed ownership or its
  junction topology.
- **Sample the general-purpose in-tree `TectonicSystem` directly.** Rejected
  for this runtime path: it owns the offline feature-compiled body model and
  performs mesh-wide nearest-cell sampling. The procedural/diffusion landcover
  path needs an O(1) random-access signal at every tile vertex. The cached
  cube-sphere field provides that contract while retaining grown ownership.
- **Let the grown plates own continents too.** Rejected here because it would
  combine a relief correction with a coastline rewrite and invalidate the
  established site/coast authority.

## Consequences

- Tectonic belts are curved, connected corridors rather than Voronoi sectors;
  the permanent causal atlas and matched terrain diff are the visual regression
  evidence.
- Field construction is deterministic and shared by `(radius, seed)`; runtime
  sampling is four cross-face texel reads and interpolation.
- The plate graph is connected by construction and regression-tested alongside
  deterministic, bounded process output and cube-face mapping.
- `GENERATOR_VERSION` advances to 30. Learned terrain must be re-exported from
  the new conditioning chart before it can evidence these ridges.
