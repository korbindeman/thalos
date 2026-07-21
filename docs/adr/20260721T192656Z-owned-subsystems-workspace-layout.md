# ADR-20260721T192656Z-owned-subsystems-workspace-layout: Own adapted subsystems in the workspace layout

- **Status:** Superseded by ADR-20260721T194628Z-role-based-agent-first-workspace
- **Date:** 2026-07-21

## Context

The flat `crates/` directory makes Thalos packages, offline production support,
and three in-tree dependencies appear equivalent. That was useful while the
dependencies were forks being evaluated, but it no longer describes their
actual role. `big_space` is the project-wide floating-origin precision
substrate; `udlod` is the Thalos-specific terrain-rendering backend, sealed
behind the render crate; and `bevy_erosion_filter` is authored by Thalos's
author and is a temporary terrain-generation implementation ahead of the
diffusion path.

The source tree should communicate ownership and lifecycle without turning a
filesystem hierarchy into a new Rust dependency rule.

## Decision

Remove the `vendor` category. Keep the existing package names and public APIs,
but organize the workspace source tree as follows when BL-32 lands:

```text
crates/
  core/
    big_space/
    world/
    celestial/
    terrain/
      erosion_filter/
    texgen/
    shipyard/
  simulation/
    physics_canonical/
    physics_local/
    control/
  rendering/
    body_render/
    udlod/
  interface/
    input/
    ui/
  offline/
    terrain_baker/
    terrain_learned/
  app/
    game/
tools/
  terrain_train/
```

`core` means durable Thalos-owned substrate, not a claim that every crate in
it is Bevy-free. `bevy_erosion_filter` remains a distinct package nested under
the terrain family until the diffusion implementation replaces it. Preserve
the upstream attribution and license material in `big_space` and `udlod` even
though they are owned project subsystems.

## Alternatives

- **Keep a `vendor/` directory** — rejected because it implies outside
  ownership and temporary integration for components now central to Thalos.
- **Fold each subsystem into a neighbouring crate** — rejected because it
  would erase useful package seams and greatly expand the scope beyond a
  navigational restructure.
- **Use `core/` as a strict dependency layer** — rejected because the current
  dependency graph intentionally includes aggregation and renderer-adjacent
  code there. The directory is an ownership/lifecycle grouping only.

## Consequences

The move updates workspace-member and local path dependencies, documentation
links, and directory-sensitive helpers in one mechanical change. It does not
rename packages or alter the `body_render` → `udlod` sealing boundary. Future
diffusion work removes or replaces `core/terrain/erosion_filter` through its
own scoped task rather than reviving a generic vendor category.
