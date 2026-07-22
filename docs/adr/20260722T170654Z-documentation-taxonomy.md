# ADR-20260722T170654Z-documentation-taxonomy: Documentation is organized by role and primary subsystem

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

The `docs/` root had grown to 37 Markdown files. Navigation, sprint strategy,
player-facing design, simulation contracts, rendering programs, dated baselines,
and developer operations all appeared as peers. The filenames were individually
reasonable, but the flat tree no longer communicated which documents were
foundational, active plans, current specifications, or supporting evidence.

The existing root index listed every file, but it could not make directory
browsing or path completion less noisy. Merging the large specifications would
also make ownership and review harder rather than improving navigation.

## Decision

Keep four canonical documents at the `docs/` root:

- `README.md` — the one documentation map and cross-reference convention.
- `backlog.md` — the execution queue.
- `architecture.md` — repository and dependency boundaries.
- `gameplay.md` — the product and gameplay vision.

Group every other live root document by its primary responsibility:

- `roadmap/` — active sprint strategy.
- `gameplay/` — player-facing systems and interaction specifications.
- `simulation/` — vehicle, physics, authority, and surface simulation.
- `world/` — celestial bodies, terrain, and vegetation.
- `rendering/` — atmosphere, clouds, oceans, and effects.
- `development/` — build, tooling, capture, and verification workflows.
- `reference/` — dated baselines, completed work orders, and focused audits that
  remain useful but are not the canonical subsystem specification.

The existing `adr/`, `incidents/`, `lore/`, and `archive/` collections retain
their established roles. The hierarchy remains one level deep: subsystem specs
do not each gain another folder. `docs/README.md` is the sole hand-maintained
index; category folders do not duplicate it with their own README files.

Filenames remain unchanged so the move expresses only taxonomy. Prose references
use repository-relative paths such as `docs/rendering/clouds.md`; Markdown links
remain relative to the document containing them.

## Alternatives

- **Keep the flat root and improve only the index** — rejected because filesystem
  browsing and path completion would remain noisy.
- **Put all specifications in one `systems/` folder** — rejected because it moves
  the same undifferentiated list down one level.
- **Use a deep hierarchy per subsystem** — rejected because most subsystems have
  one canonical spec; one-file directories add navigation without structure.
- **Merge related specifications into a few very large documents** — rejected
  because the current files have distinct owners, lifecycles, and verification
  contracts. Consolidation should happen only when two documents duplicate an
  authority, not merely to reduce file count.

## Consequences

- The root now exposes the project's four highest-altitude documents directly.
- New documents must choose a role and category instead of defaulting to the root.
- Existing links and agent guidance require a one-time path migration.
- A document whose responsibility changes may move categories without changing
  its filename or identity; accepted ADR and incident paths remain stable.
