# ADR-0008: Offline terrain packages are an authored `SurfaceQuery` backing

- **Status:** Accepted
- **Date:** 2026-07-20
- **Supersedes:** ADR-0007

## Context

ADR-0007 interpreted the deletion of Thalos's old feature compiler and startup
bake checks as a product decision against baking. The intended product direction
is different: terrain baking is a first-class offline authoring feature. A
hierarchical diffusion pipeline, inspired by InfiniteDiffusion / Terrain
Diffusion, should generate the expensive planetary and regional structure on
developer or CI hardware. Player devices should stream the result and reconstruct
only close-range detail.

The live game already has valuable invariants that must survive this reversal:
UDLOD's cube-sphere tile address, `SurfaceQuery` as the shared height seam, one
height authority for rendering and collision, and bounded RAM/disk tile caches.
The deleted bake pipeline does not satisfy the new model/package requirements
and must not be revived merely because both systems are called “baking.”

## Decision

Add a new offline terrain bakery whose output is an immutable, versioned,
content-addressed terrain package consumed through a `PackageSurface`
implementation of `SurfaceQuery`.

The package is a sparse cube-sphere quadtree of low-frequency height plus
Laplacian residuals and material/conditioning channels. Nodes subdivide only
when measured reconstruction error or surface complexity requires more detail;
flat/simple terrain uses constant or low-order predictors and terminates early.

The player does not run planetary diffusion. A versioned deterministic client
stage reconstructs the final close-range bands from stored height,
derivatives/conditioning, and seed. Only CPU/GPU parity-proven deterministic
output is collidable; other learned or shader detail is visual-only.

The shipped package is authored source data. The existing RAM/disk tile cache is
disposable memoization of decoded/reconstructed UDLOD payloads and is keyed by
package hash, reconstruction version/tier, attachment layout, body scale, and
dynamic flatten state.

## Consequences

- ADR-0013 selects one Rust/Burn model definition for training and inference;
  Rust also owns the package reader, validator, tile contract, and gameplay
  sampling.
- The package format, not the model implementation, is the durable boundary.
  New models can rebake packages without changing render/collision consumers.
- Adaptive residual storage spends bytes according to measured error and
  complexity, preserving crater rims and scarps while strongly compressing mare.
- Cross-face context, canonical borders, package validation, rate-distortion
  reports, and fixed model/tool hashes become required bakery outputs.
- The current runtime `ProceduralSurface` remains a valid provider for bodies
  without packages and as a development fallback.
- The old `Feature`/`PlanetSurface` compiler, dump tool, and startup bake check
  remain superseded. Pure geological math may be extracted; their architecture
  is not restored.

## Implementation status (2026-07-20)

Schema v1 is live in `thalos_terrain::package`: magic/version header, producer
and optional model identity, body/content bounds, sparse cube-node records,
typed blob descriptors, geometric-error/complexity metadata, stable checksums,
and full bounds/overlap/reference validation. `PackageSurface` is the runtime
backing selected by the N-body `BodySurfaceRegistry`; its content key also
names reconstructed UDLOD cache entries.

The Mira compatibility producer writes one global metadata blob plus a 32→512
five-level height pyramid: six authoritative `RawU16LE` roots and quantized
signed-residual children under canonical half-open texel ownership. All 2,047
logical nodes are indexed; 86 omit their payload and reconstruct from the
nearest ancestor within the declared 256 m compatibility budget. The validator
checks the complete address/parent set, payload contracts, fallback errors,
bounds, overlaps, and checksums before decoding. The metadata serializes only a
1×1 height placeholder, proving indexed package height is the authority. The
exact artifact fingerprints reconstructed tile-cache entries, so an encoder
change cannot reuse tiles from an older package with the same authored inputs.
MIRA-0 is complete; the trained airless diffusion producer remains MIRA-1/2
work, and this ADR does not treat compatibility output as diffusion output.

The production path is now specified in `docs/mira_airless_mvp.md`: pinned
global and regional lunar DEM teachers plus labelled process simulation feed a
physical-scale Laplacian S0–S3 patch cascade; the first campaign sphere targets
4096 texels per face with direction-seeded overlap-belt consensus. These are
producer choices behind this ADR's package boundary, not changes to the runtime
decision.

## Alternatives rejected

1. **Runtime-only analytic generation (ADR-0007).** Rejected because the product
   explicitly needs learned, offline-authored terrain fidelity and variable
   storage density.
2. **Run Terrain Diffusion on player GPUs.** Rejected because model/runtime
   requirements, latency, determinism, and hardware reach should not gate play.
3. **Ship one uniform dense cubemap pyramid.** Rejected because it spends equal
   storage on flat mare and complex crater relief and prevents useful
   rate-distortion control.
4. **Treat package files as the runtime cache.** Rejected because authored
   content must be immutable/reliable while caches must remain disposable,
   tier-specific, and invalidatable.
