# ADR-20260722T141000Z-far-cloud-density-is-surface-parameterized: Far cloud density is surface-parameterized

- **Status:** Accepted
- **Date:** 2026-07-22
- **Supersedes:** ADR-20260722T135123Z's incommensurate Cartesian-domain candidate

## Context

ADR-20260722T135123Z proved that smoothly phase-warping one periodic 3-D tile
does not make it a valid planet-wide far-cloud producer. Its next bounded
candidate was a convex crossfade between independently transformed domains at
incommensurate periods.

BL-33 implemented that candidate in the canonical density and verified it
locally. A weather/formation-selected crossfade retained runway and cruise
coverage, then a guaranteed 35% contribution from the second domain retained
formation while materially changing the cloud field. Both variants were baked
through the same 512², 3×3-filtered, four-stratum optical-depth atlas and the
same 24-sample chord as the single-domain control.

The `cloud-planet` comb survived. The guaranteed blend changed where bands
appeared, but not their long curved/diagonal topology. The controlled result
therefore rejects the broader premise: sampling any small Cartesian 3-D tile
over planet-radius positions makes a spherical shell cut coherent repeated
surfaces into planet-scale bands. More Cartesian domains obscure the local
repeat but do not create a valid spherical parameterization.

## Decision

- CLOUD-6's planet-wide density producer is parameterized over the body's
  surface (cubemap direction plus normalized layer height), not Cartesian
  planet-centred position.
- A future surface seed/shape field is canonical input to both projections:
  the near volume uses it to select/phase its local 3-D bodies, while the far
  atlas integrates optical-depth/albedo/normal/height moments directly from
  that same surface-space density contract.
- The local Cartesian Perlin/Worley volume remains the near-field shape and
  erosion source. It is never evaluated over the whole sphere to author an
  orbital texture.
- The current weather cubemap is the authoritative starting seam. A new
  surface-space cell/phase channel may extend that authority, but it must not
  become an unrelated far-only noise layer.
- Until this producer exists, the six-sample weather-column march remains the
  accepted far fallback.

## Alternatives

- **Add more incommensurate Cartesian domains.** Rejected by controlled
  planet captures: the band distribution changes but the spherical-shell comb
  remains, while near cost rises by 0.35–0.73 ms in the tested views.
- **Keep the dual domain only for local variety.** Rejected for this slice:
  the local visual delta was small (0.16/255 cruise MAE) and does not justify
  the measured extra fetch after its far-producer purpose failed.
- **Use a far-only spherical noise texture.** Rejected by the one-density
  contract; the surface-space field must also condition near density.
- **Project the Cartesian volume in local tangent charts during the bake.**
  Deferred: chart ownership/blending is effectively a surface parameterization
  and should be implemented as the canonical surface seed rather than hidden
  inside the atlas producer.

## Consequences

- All experimental atlas allocations, bindings, compute pipelines, and the
  second near-domain fetch are reverted. The validated weather phase offsets
  on the single local broad/formation domains remain.
- CLOUD-6 is blocked on a surface-space density seed/contract, not on atlas
  format tuning.
- Rejected controlled evidence is retained under `step-10a/` and `step-10b/`;
  the clean weather-fallback proof remains `step-8-rollback/`.
