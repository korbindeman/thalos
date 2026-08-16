# ADR-20260814T201228Z-broad-terrain-shadows-use-coarse-caster-twins: Broad terrain shadows use coarse caster twins

- **Status:** Accepted
- **Date:** 2026-08-14
- **Amends:** ADR-20260730T225312Z-a-near-cascade-and-cascade-boxes-square-on-the-ground

## Context

After bounding foliage shadows, the settled `forest-stand` scene still spent
24.42 ms in the four custom shadow cameras. A warmed 4→0 cascade ladder isolated
their median marginal costs:

- cascade 1: 2.71 ms;
- cascade 2: 2.99 ms;
- cascade 3: 11.15 ms;
- cascade 4: 8.94 ms.

The two broad cascades therefore cost 20.08 ms while the two near cascades cost
5.70 ms. The renderer had recently moved from 65² to 129² visible terrain
tiles. Every tile's shadow child reused that full mesh, so broad terrain views
multiplied approximately 32,768 surface triangles per visible tile even though
their shadow texels cannot resolve that sampling density.

This revises the 2026-07-30 ADR's prediction that an additional cascade draw
would be nearly free. That estimate preceded correct culling and the larger tile
mesh; current headless evidence is authoritative.

## Decision

Terrain has two explicit shadow representations:

- cascades 0–1 see the exact visible 129² mesh on a near-terrain layer;
- cascades 2–3 see a 33² position+normal twin on a far-terrain layer;
- every fourth visible-grid sample is reused exactly, including both tile
  boundaries;
- both representations retain the floor-sphere skirt and the visible surface's
  tight culling box;
- structures, craft, rocks, and bounded foliage proxies remain on the common
  caster layer seen by all four cameras.

The coarse twin has 16× fewer surface triangles. Its GPU bytes are included in
the tile residency denominator; a 1,542-tile forest adds about 51 MiB, visible
in both the capture receipt and performance lane.

## Alternatives

- **Use the coarse twin in all cascades.** Rejected: the first two cameras cost
  only about 5.7 ms and exist specifically for landing gear, craft panels, and
  near terrain detail.
- **Reduce all shadow-map resolutions.** Rejected: prior 65²-tile measurement
  found fill insensitive, and it would spend the near-detail quality protected
  by the amended ADR. Geometry density is the changed variable and the typed
  cascade ladder identifies only the broad passes as the target.
- **Disable cascades 2–3.** Rejected: it removes ridge/valley terrain shadows
  rather than representing them at the fidelity their texels can resolve.
- **Implement the W12 horizon term first.** Deferred, not rejected: that is the
  correct beyond-cascade mechanism, but does not replace mid-field relief
  shadows or justify full visible geometry inside the existing cameras.
- **Merge resident tiles into one broad shadow mesh.** Rejected for this slice:
  it could reduce draw count further but introduces incremental rebuild and
  seam ownership. The per-tile coarse twin obtains the measured geometry win
  without a second streaming lifecycle.

## Consequences

- In the matched cascade ladder, cascades 3–4 fall from 20.08 to 8.68 ms p50.
- Full four-cascade p50 falls from 66.33 to 55.05 ms. Mean-derived FPS rises
  from 15.38 to 17.59 in the same scene.
- Combined with bounded foliage proxies, the original 83.50 ms baseline becomes
  56.84 ms: 11.98→17.59 fps, a 46.9% increase.
- The shadows-off control rises from 40.55 to 40.87 ms p50. The added entity and
  residency cost is therefore small relative to the far-pass reduction.
- Exact-source `forest-stand` capture shows no missing caster class, light band,
  or tile-seam leak. A more rugged matched preset remains useful future evidence
  if the coarse resolution changes.
