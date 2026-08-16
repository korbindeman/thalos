# ADR-20260810T201029Z-korsou-foliage-shadows-use-bounded-proxies: Kòrsou foliage shadows use bounded crown proxies

- **Status:** Accepted
- **Date:** 2026-08-10
- **Amends:** ADR-20260810T194324Z-woody-vegetation-is-impostor-first

## Context

The shared impostor-first decision correctly removed Kòrsou's dense authored
tree meshes from Bevy's cascaded shadow maps. Submitting every alpha-masked plant
had multiplied foliage coverage by shadow cascades and made the target Mac
unusable. Excluding all foliage shadows, however, left nearby vegetation
visually detached from the ground and materially below the fidelity of Thalos.

The shared visible representation cannot simply cast through Bevy's standard
opaque shadow pass. Each root is a degenerate hemisphere-octahedral card whose
vertex shader expands its bounds and whose atlas supplies the silhouette. The
standard shadow pass does not reproduce that atlas projection and would cast an
opaque rectangular card. Restoring source meshes would restore the cost curve
the impostor-first decision removed.

The missing signal is a bounded approximation of nearby canopy occlusion, not
leaf-perfect shadow geometry at every distance.

## Decision

Kòrsou keeps every visible shrub and tree impostor marked `NotShadowCaster`.
Its planar foliage adapter derives a separate coarse crown proxy from each
deterministic accepted **tree** root:

- one octahedron with six vertices and eight triangles per tree;
- batched by the existing 128 m foliage cell;
- assigned to a shadow-only render layer seen by the directional sun but not by
  the normal camera;
- active only while the cell lies within 760 m of the camera;
- removed with the visible cell and disabled by the same foliage preference;
- omitted for shrubs, whose projected grounding benefit does not justify the
  extra caster geometry.

The proxy receives no shadows and exists only to cast into Bevy's cascaded sun
maps. The F3 Kòrsou extension reports active shadow cells and triangles so the
bounded cost is visible during acceptance. Structural tests pin the proxy cost
and the reach boundary.

This amends only the Kòrsou no-shadow clause of the impostor-first ADR. Shared
visible atlas geometry and Thalos's planetary custom-shadow policy are
unchanged.

## Alternatives

- **Keep foliage shadowless.** Rejected: it preserves the performance win but
  leaves nearby trees floating and breaks the cross-adapter fidelity target.
- **Let the visible cards cast directly.** Rejected: Bevy's standard shadow pass
  would cast the expanded opaque card bounds rather than the atlas silhouette.
- **Restore authored tree meshes only for shadows.** Rejected: it reintroduces
  high vertex/upload/cascade cost for geometry the camera cannot resolve.
- **Add an alpha-aware custom Bevy shadow pipeline.** Deferred: it could produce
  finer silhouettes, but it is materially more pipeline complexity and fill
  cost than the grounding signal requires. It needs measured evidence before
  replacing the proxy.
- **Cast proxies to the horizon.** Rejected: distant terrain canopy colour is
  already the representation authority, while unbounded cascaded casters would
  scale with island coverage rather than visible shadow value.

## Consequences

- Nearby tree cover can cast stable, volume-shaped sun shadows without changing
  the compact visible representation.
- The approximation cannot produce leaf-shaped penumbrae, and shrubs cast no
  shadow. Those are explicit fidelity limits rather than silent omissions.
- Foliage shadow cost is bounded by camera radius and exposed in diagnostics.
- GPU capture and live target-Mac acceptance remain required to validate cascade
  layer behavior, the visual grounding, and frame time.
