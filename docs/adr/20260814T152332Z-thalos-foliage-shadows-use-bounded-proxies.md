# ADR-20260814T152332Z-thalos-foliage-shadows-use-bounded-proxies: Thalos foliage shadows use bounded crown proxies

- **Status:** Accepted
- **Date:** 2026-08-14
- **Amends:** ADR-20260810T201029Z-korsou-foliage-shadows-use-bounded-proxies

## Context

The `forest-stand` performance matrix measured the real 1600×900 offscreen game
renderer after terrain residency and mesh counts stabilized. With clouds, grass,
and MSAA disabled, the full scene took 83.50 ms. Disabling foliage alone saved
19.86 ms while disabling foliage with shadows already off saved only 1.57 ms.
The 18.29 ms interaction was the cost of submitting visible foliage batches to
four custom shadow cameras, not the cost of drawing the forest in the main view.

The post-cull census explained the multiplier. Cascades 2 and 3 contained about
1,007 and 1,612 non-terrain casters, primarily alpha-masked tree batches. The
main foliage representation and its material were therefore paid repeatedly in
views where neither leaf shading nor horizon-distance silhouettes justified the
cost.

Kòrsou had already encountered the same mechanism and adopted bounded opaque
crown proxies. The spatial adapter differs, but the representation decision is
shared: detailed visible foliage is not the correct shadow representation.

## Decision

Thalos keeps visible mesh and impostor foliage on the main render layer and
marks it `NotShadowCaster`. Its spherical vegetation adapter derives a separate
shadow-only crown proxy from deterministic ring-0 **tree** placements:

- one opaque octahedron with six vertices and eight triangles per tree;
- batched by the existing 200 m ring-0 vegetation tile;
- active within 760 m of the view anchor and removed beyond 900 m;
- admitted at at most eight proxy tiles per frame;
- assigned only to the custom shadow-caster layer;
- omitted for shrubs and every farther vegetation ring;
- removed with its visible tile and disabled by the same foliage preference.

The proxy uses the visual tile's body-fixed anchor and real-space transform path.
The headless performance lane reports active proxy cells and triangles in every
benchmark result.

## Alternatives

- **Keep the full foliage batches as casters.** Rejected: the headless matrix
  attributes 18.29 ms to their interaction with the shadow rig.
- **Remove foliage shadows entirely.** Rejected: nearby trees need a bounded
  grounding signal, and the proxy supplies it for little measured interaction
  cost.
- **Reduce visible foliage density or distance.** Rejected: direct foliage costs
  only 1.57 ms. This would spend visual quality on the wrong path.
- **Use a depth-only version of the same foliage batches.** Rejected: it retains
  thousands of per-cascade submissions and alpha coverage. The entity/view
  multiplier is the fundamental cost.
- **Share Kòrsou's planar implementation directly.** Rejected: Thalos placement
  is spherical and body-fixed. The mechanism is shared through an explicit
  adapter rather than pretending the spatial models are identical.

## Consequences

- The matched `forest-stand` baseline falls from 83.50 to 64.93 ms, or 11.98 to
  15.40 fps, while the main-view foliage population remains unchanged.
- Foliage/shadow interaction falls from 18.29 to 0.32 ms. The remaining custom
  shadow cost is independent of foliage and must be addressed at the terrain /
  cascade representation boundary.
- The measured forest uses 53 proxy batches and 8,152 proxy triangles. The
  shadow cost is bounded by camera radius rather than the 22 km visible carpet.
- Distant leaf-perfect cast shadows are intentionally gone; nearby crown-shaped
  grounding shadows remain. Shrubs do not cast.
- Interactive flight remains the acceptance gate for temporal shadow stability;
  matched headless capture is the acceptance gate for population and gross
  shadow artifacts.
