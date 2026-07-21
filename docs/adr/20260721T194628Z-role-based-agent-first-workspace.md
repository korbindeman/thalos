# ADR-20260721T194628Z-role-based-agent-first-workspace: Organize the workspace by role for agent-first development

- **Status:** Accepted
- **Date:** 2026-07-21
- **Supersedes:** ADR-20260721T192656Z-owned-subsystems-workspace-layout

## Context

Thalos's flat `crates/` directory mixes reusable libraries, product binaries,
offline producers, visual laboratories, and adapted dependencies. It also hides
that the current `thalos_game` binary is both the interactive application and
the only composition root available to headless verification. The first layout
proposal over-corrected this by putting most Thalos-owned code under `core/`, a
name that implied a dependency layer the graph did not actually have.

The workspace should make a new agent's first questions cheap: what is a
product, what is reusable runtime code, what is an offline producer, what is a
visual lab, and what is vendored? The source tree should also support the
first-class capture architecture in ADR-20260721T194629Z-first-class-headless-capture-runtime.

## Decision

Use top-level lifecycle roots and role-grouped library roots:

- `apps/` contains thin runnable products (`game`, `capture_host`).
- `crates/` contains reusable libraries grouped as `domain`, `simulation`,
  `rendering`, `interface`, `runtime`, `capture`, `offline`, and `vendor`.
- `tools/` contains production and orchestration executables.
- `labs/` contains isolated visual-development environments.
- `artifacts/` contains ignored generated evidence; tool source and tool output
  no longer share a tree.

The target package tree is recorded in `docs/architecture.md`. In particular:

- `big_space` is a foundational but **vendored** dependency and lives under
  `crates/vendor/big_space`; retain its upstream provenance and licence.
- `thalos_udlod` is a fully adapted Thalos rendering subsystem and lives under
  `crates/rendering/udlod`.
- `bevy_erosion_filter` is author-owned, temporary terrain implementation code
  nested under `crates/domain/terrain/erosion_filter` until diffusion replaces it.
- `thalos_texgen` is an offline compiler. Runtime rendering consumes baked
  assets and does not depend on the generator.
- The current construction and render packages may become
  `thalos_construction` and `thalos_render` as their already-planned dependency
  inversions land.

## Alternatives

- **Keep a flat `crates/` directory** — rejected because lifecycle and ownership
  remain implicit and every search spans unrelated code.
- **Put all owned code under `core/`** — rejected because ownership is not a
  dependency layer and the label obscures domain and runtime responsibilities.
- **Treat every adapted dependency as vendor code** — rejected for `udlod` and
  the erosion filter because Thalos owns their direction; retained for
  `big_space` after reconsidering its upstream-derived identity.
- **Split every large module into a crate** — rejected because package seams
  should enforce dependency boundaries, not replace ordinary Rust modules.

## Consequences

The move is larger than a path-only cleanup: the application/capture split must
land first so the final directories describe real boundaries. Existing package
names remain until their semantic rename steps. Documentation links, manifests,
helper scripts, and generated-artifact paths must move atomically per migration
slice. The old BL-32 one-shot mechanical move is decomposed by the capture plan
and then completes the remaining relocations.
