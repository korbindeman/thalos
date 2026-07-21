# ADR-20260721T050036Z-ocean-composite-independent-of-atmosphere: The analytic ocean is independent of the atmosphere renderer

- **Status:** Accepted
- **Date:** 2026-07-21

## Context

ADR-20260720T185954Z kept the ship-view analytic ocean inside the custom
`BodySky` pass because independently sorted transparent body-centred quads had
been unstable. ADR-20260721T032343Z later made Bevy's raymarched atmosphere
canonical and force-hid `BodySky` to prevent a double sky. Clouds were moved to
an explicitly ordered dedicated compositor, but ocean remained in the hidden
material. Consequently the default renderer showed seabed where ocean should
be, and even the `ocean-slopes` diagnostic never entered the water shader.

Atmosphere selection must not change whether a physical surface exists. The
analytic-sphere and one-signed-field decisions remain valid; only ownership of
their screen projection is at issue.

## Decision

The ship-view analytic ocean is owned by one dedicated
`BodyOceanMaterial`, independent of the selected atmosphere renderer.

- `BodySkyMaterial` compiles the shared optical shader as atmosphere-only and
  remains an explicit legacy A/B path.
- `BodyOceanMaterial` compiles that shader as ocean-only. It delegates the
  exact same bind-group implementation, so scene depth, resident signed-height
  tiles, coast-atlas tail, spectral slope field, and foreground-air integration
  have one implementation rather than a copied fork.
- The game mirrors the canonical per-body optical state into the ocean material
  after terrain/atmosphere projection and gives the ocean the same resident-
  terrain visibility lifecycle.
- Body-centred transparent siblings have an explicit order: legacy atmosphere,
  ocean, then clouds. Bevy's canonical atmosphere renders before these surface
  composites.

This supersedes only ADR-20260720T185954Z's decision to place ship-view water
inside `BodySky`; its analytic-sphere/no-mesh invariant remains accepted.

## Alternatives

- **Force ocean screenshots/gameplay back to the custom atmosphere** — rejected
  because it would hide the integration bug and restore two normal atmosphere
  paths.
- **Keep `BodySky` visible but make it ocean-only under Bevy** — rejected
  because atmosphere selection would still mutate the water owner's behavior
  and preserve the monolithic ownership seam.
- **Copy the ocean code into a new standalone shader** — rejected because the
  signed-field tile walk and filtered spectral shading are large, load-bearing
  mechanisms that must not drift between two implementations.
- **Reintroduce a planet-scale water mesh** — rejected by
  ADR-20260720T185954Z; facet sag and map/ground sorting are surface-model
  failures, not compositor failures.

## Consequences

Ocean remains visible and identical under both atmosphere backends, while the
legacy atmosphere can be deleted later without moving a physical surface.
Cloud and ocean ordering is deterministic through explicit material depth bias
instead of incidental entity sorting.

The ocean material temporarily shares the legacy sky's resource contract and
shader source so there is one implementation of foreground optical integration.
F7/F9 may rename/split that shared contract when atmosphere-derived environment
lighting becomes the canonical surface input, but must preserve one analytic
ocean projection and one signed-field authority.
