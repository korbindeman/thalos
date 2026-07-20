# ADR-0007: Planet clouds use one weather field with regime-specific render projections

- **Status:** Accepted
- **Date:** 2026-07-20

## Context

Thalos currently has three cloud authorities: near-volume constants and a
coverage texture in the game, authored `CloudCover` data in the world schema,
and separate reference images for the orbital layer. The near renderer is a
fixed thin shell built from an extruded 2-D atlas; it cannot express the cloud
types, vertical structure, lighting, or surface-to-orbit continuity required by
the accepted visual target. Extending that split would make every later density,
lighting, shadow, weather, and orbital feature reconcile divergent sources.

The architecture fork gated CLOUD-1: where cloud rendering belongs, which
planetary weather topology becomes canonical, whether cloud temporal work waits
for whole-scene TAA, and whether the first weather model attempts fluid
simulation.

## Decision

- `thalos_world::CloudClimate` is the authored, quality-neutral per-body cloud
  configuration. Absence is authoritative: no climate means no cloud system.
- A per-body `CloudWeatherField` is the mutable simulation-time authority for
  coverage, type, base, top, storm/precipitation potential, and wind. Its first
  topology is a seam-safe cubemap/2-D texture array; equirectangular data is an
  import/export format, not the runtime contract.
- Cloud render mechanism belongs in `thalos_body_render::clouds`. The game owns
  orchestration and projects simulation state into render inputs; it does not
  own an independent density definition. The existing vendored cloud renderer
  is absorbed, with its attribution retained, instead of remaining a parallel
  top-level subsystem.
- Near and mid volume, orbital optical-depth representation, cloud-sun
  transmittance, and authoring preview are projections of the same weather and
  density definition. They may use different LODs, but none may invent separate
  cloud placement.
- Temporal reconstruction remains cloud-local and body-fixed. It does not wait
  for or depend on the whole-scene TAA decision.
- The first weather evolution model is authored/procedural fields with slow
  advection and analytic front or cyclone stamps. A fluid simulation may later
  replace the producer behind `CloudWeatherField`, but is not part of the first
  high-fidelity vertical.

## Alternatives

- **Keep `thalos_volumetric_clouds` as a peer render crate** — rejected because
  it preserves a second celestial-render ownership boundary and encourages
  private lighting, atmosphere, body selection, and orbital paths.
- **Retain equirectangular weather as the runtime contract** — rejected because
  polar concentration and the longitude seam become permanent density,
  advection, filtering, and authoring problems. Equirect import/export remains
  useful without defining runtime topology.
- **Block cloud reconstruction on whole-scene TAA** — rejected because clouds
  have body-fixed motion, optical depth, timewarp, and history rejection needs
  that differ from opaque geometry; it would also delay the dominant cloud cost
  reduction behind an unrelated renderer decision.
- **Start with a fluid or climate simulation** — rejected because it adds a
  large simulation and validation surface before the renderer can faithfully
  display much cheaper structured weather fields.
- **Keep separate near and orbital authored assets** — rejected because their
  placement, shadows, light, and transitions cannot be made reliably coherent.

## Consequences

CLOUD-1 must first establish ownership and delete superseded authorities before
fidelity work begins. Later projections share stable weather identity and
versioning, enabling deterministic temporal rejection and seamless regime
handoff. The cube topology and mechanism move require up-front plumbing and
tooling, and the shared density contract must be designed carefully enough for
both WGSL consumers and limited CPU-side probes. In return, every later cloud
feature can improve one world instead of synchronizing several approximations.
