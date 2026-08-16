# ADR-20260803T060232Z-tectonics-is-a-pre-diffusion-structural-prior: Tectonics is a pre-diffusion structural prior and never a coastline authority

- **Status:** Accepted
- **Date:** 2026-08-03
- **Decided by:** user, 2026-08-03 ("go for it")

## Context

Thalos had cellular "plates" for continental layout but its mountains came from
an unrelated 420 km fractal mask. The result could contain rugged areas, but not
long mountain systems whose placement explained itself through plate motion.
The sibling `thalos_maps` prototype demonstrated the missing shape: plate Euler
poles, relative boundary motion, and convergent/divergent/transform regimes
produce connected orogen, rift, ridge, and trench systems that a downstream
terrain generator can elaborate.

Two existing constraints narrow the design. First,
ADR-20260729T054550Z makes `ProceduralSurface::macro_signed_height_m` the one
coastline authority for both procedural and learned terrain. Second, the neural
pipeline is conditioned from `ProceduralSurface::macro_signals`; structure must
exist before that export if diffusion is expected to detail it coherently.

## Decision

**Tectonics authors the macro relief structure before diffusion, while
continentalness continues to author the coastline alone.**

The existing deterministic Worley plate cells receive deterministic Euler
poles. At each sample, the two nearest cells identify the exact local Voronoi
bisector. Their relative tangent velocities classify the margin as convergent,
divergent, or transform and supply a motion strength. Thalos's declining
geology is represented by a broad fossil signature on every margin plus a
narrower active core on a deterministic minority of plate pairs.

Convergent land margins drive `orogeny`, which is already consumed by the
mountain cascade and `ConditioningChart`. Divergent ocean margins add mid-ocean
ridge relief, divergent land margins cut rifts, and submerged convergent
margins cut trenches. All signed contributions pass through
`combine_macro_and_relief`; none feeds `continentalness` or
`macro_signed_height_m`. Composed-height sign is regression-pinned to the
authored signed sea field.

## Alternatives

- **Add tectonic ridges after neural generation.** Rejected because the model
  would not organize valleys, drainage, or fine ridges around the orogen. It
  would be a decal in geometry rather than a generative cause.
- **Keep the independent fractal orogeny mask and merely stretch it.** Rejected
  because it can imitate long shapes but cannot distinguish collisions, rifts,
  transforms, or trenches. The causal field remains unavailable to conditioning.
- **Instantiate the full in-tree `TectonicSystem` and bake a global cubemap for
  every `ProceduralSurface`.** Deferred as unnecessary for this slice. Its mesh
  construction and nearest-cell scans are appropriate for offline feature
  compilation, but the runtime already evaluates a coherent plate partition
  analytically. Adding a second plate topology would create an alignment and
  lifecycle problem without improving the first visible result.
- **Let tectonics reshape continents immediately.** Rejected for this slice.
  It would move the established coastline and spaceport, conflate two authority
  changes, and force a new coast validation problem before ridge quality could
  be judged.

## Consequences

- Procedural terrain and every newly exported conditioning chart contain
  connected tectonic relief provinces. The permanent `tectonic_preview` example
  emits matched causal-regime and shaded-relief atlases.
- The checked-in learned Thalos chart was generated from the previous orogeny
  field. It remains valid old content, but does not contain this decision's
  ridges; regeneration through `export_thalos_macro` and `thalos_export.py` is
  required before neural evidence can close NTR-X2f.
- `GENERATOR_VERSION` advances because canonical procedural relief changed.
- Future island arcs, hotspot chains, and hydrology consume these process labels
  rather than inventing parallel plate fields.
