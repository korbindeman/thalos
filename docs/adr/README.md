# Architecture Decision Records

This directory is the project's **decision log** — the persistent, git-committed record of
*why* Thalos is built the way it is. `CLAUDE.md` and the `docs/` specs describe *what* the
system is; ADRs capture the decisions behind it, including the alternatives that were rejected
and why. Read here before reopening a settled choice.

## How to use

- **Read first.** Before making or revisiting a non-trivial design choice, skim the index below
  (and the relevant ADRs) so you don't re-litigate a settled question or re-explore a rejected
  path.
- **Add one when a decision is made** — whenever you choose among alternatives, defer or cut
  scope, or reverse an earlier approach (architectural or not). Do it at decision time; the
  reasoning is otherwise lost to context compaction. Copy `0000-template.md` to
  `NNNN-kebab-title.md` (next number in sequence), fill it in, and add a row to the index.
  Resolving a fork from the backlog's "Decisions pending" table always gets one.
- **One decision per record.** Keep each ADR short and focused.
- **Immutable once Accepted.** Don't rewrite an accepted decision — write a new ADR that
  supersedes it, and flip the old ADR's `Status` to `Superseded by ADR-NNNN`. Update the index.
- **Commit ADRs alongside the code** that realizes the decision (same change/slice).
- Historical decisions being migrated out of auto-memory get their original decision date plus
  a "recorded YYYY-MM-DD" note.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-steering-harness.md) | Roadmap steering harness — backlog + steer skill + ADR / incident logs | Accepted |
| [0002](0002-analytic-planet-water-never-meshed.md) | Planet water is an analytic ray-traced sphere — never a mesh, at any scale | Accepted |
| [0003](0003-eva-outside-the-slf.md) | EVA stays a kinematic body-fixed controller, outside the SLF | Accepted |
| [0004](0004-replace-avian-owned-tgs-soft-solver.md) | Replace Avian with an owned TGS-Soft solver crate (`thalos_physics`) | Accepted |
| [0005](0005-coastline-as-authored-data.md) | The coastline is authored data — scene depth is occlusion-only for water | Partially superseded by ADR-0006 |
| [0006](0006-water-projects-one-signed-sea-field.md) | Water is a projection of one signed sea-height field — depth never decides coverage | Accepted |
| [0007](0007-one-weather-field-many-cloud-projections.md) | Planet clouds use one weather field with regime-specific render projections | Accepted |
| [0008](0008-cloud-skips-require-conservative-bounds.md) | Cloud ray leaps require conservative density bounds; cadence-preserving reuse is the default | Accepted |
