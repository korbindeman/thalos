# ADR-20260731T024003Z-crate-granularity-policy: crates are split on payoff, and the runtime monolith is dismantled

- **Status:** Accepted
- **Date:** 2026-07-31

## Context

The workspace's *edges* are healthy — pure crates CI-guarded, role folders,
one composition layer — but the mass distribution is not: `thalos_runtime` is
~93 kLOC, ~43% of all workspace Rust, while the median crate is ~2 kLOC.
Consequences, each multiplied by every agent and every iteration: almost any
Rust edit pays the monolith's rebuild inside the capture-host restart;
concurrent agents collide in one crate (merge conflicts, invalidated
incremental artifacts); no feature can be type-checked or given a preview
harness without compiling the whole game. The structural cause is specific:
the shared game-state resources (`SolarSystemState`, `ViewAnchor`,
`GameContext`, `SimClock`, `MapSnapshot`, mirrors…) are defined inside
runtime modules, so anything reading them must live in the same crate.

Meanwhile the existing rule — "ordinary feature size is handled with Rust
modules rather than a crate per feature" — gave no counterweight, so features
accreted into runtime by default. A decision on *when a crate earns
existence* was needed, plus a direction for the accumulated mass.

## Decision

**A crate is three things at once: a compile unit, an ownership boundary,
and an iteration harness. Split whenever a boundary buys at least one of
four payoffs:**

1. **A cheaper edit loop** — edits to X stop rebuilding Y (the capture-host
   restart compile is the most-multiplied cost in the repo).
2. **A compiler-enforced dependency guarantee** — no-Bevy, no-renderer,
   state-in/pixels-out. Guarantees in `Cargo.toml` don't decay; guarantees in
   docs do.
3. **A standalone harness** — the crate carries its own preview/example
   binary that compiles without the runtime (`object_preview`,
   `just ui-preview` are the precedents).
4. **Agent isolation** — concurrent agents own disjoint crates.

Guardrails:

- **Modules still handle ordinary feature size.** The bar is one of the four
  payoffs, not taxonomy. In practice, anything feature-shaped ≥ ~3–5 kLOC
  with a clean state seam clears it.
- **Feature crates never depend on each other** — only downward on domain
  crates and the game-state crate. Cross-feature communication goes through
  the blackboard; two feature crates wanting each other means the shared
  thing belongs a layer down.
- **The game-state crate is types-only and append-biased** — resources,
  components, accessors, single-writer doc comments; no systems. It is a hub
  every feature depends on, so churn there rebuilds the world.
- **Don't split what's scheduled for demolition** (`thalos_udlod`, the
  procedural terrain generation chain). A split is an investment in a
  future; those have none.

The layer picture this formalizes (details + migration phases in
`docs/architecture.md`):

```text
L0  foundation    diagnostics, big_space
L1  pure domain   world, terrain, celestial, physics_canonical, control,
                  navigation, texgen            (no Bevy — CI-guarded)
L2  Bevy leaves   input, ui, body_shading, physics_local, shipyard,
                  capture/*, + feature crates (hud, map, shipyard_editor,
                  structures, clouds, …)
L2.5 game state   thalos_game_state (new) — the blackboard L2 reads/writes
L3  composition   thalos_runtime — schedule ordering, plugin graph,
                  sim-coupled drivers, glue; target ~25–35 kLOC
L4  shells        apps/game, tools/capture_host
```

## Alternatives

- **Status quo (modules only, one big runtime)** — rejected: the four costs
  above are real and multiplied per agent; the monolith already forced the
  "dirty worktree is not your problem" workflow rule because everyone edits
  one crate.
- **Crate-per-feature confetti (no bar)** — rejected: cross-crate refactors
  cost more than cross-module ones (orphan rule, visibility, Cargo churn),
  per-crate fixed costs invert the compile win below a few kLOC, and a web
  of inter-feature deps recreates the monolith with extra plumbing. Hence
  the four-payoff bar and the no-lateral-deps rule.
- **Splitting `thalos_terrain` (33 kLOC) now** — rejected: the diffusion
  terrain rework (keystone, ADR-20260723T143155Z) replaces the generator;
  splitting is wasted work until that lands. Same for `thalos_udlod` (EOL).
- **Extracting features without the state crate** — rejected: tried mentally
  against the coupling data; `hud/` alone touches ~20 sibling modules,
  almost all state reads. Without the blackboard extraction every peel drags
  half the runtime with it.
- **Keying the split to Cargo features instead of crates** — rejected:
  features don't give ownership boundaries or standalone harnesses, and
  feature mixtures fork the build graph (the `dev-renderer` fingerprint rule
  exists precisely to prevent that).

## Consequences

- New feature work goes in a feature crate (or an existing one), not into
  `thalos_runtime`, whenever it clears the bar. Runtime accepts only
  composition, sim-coupled drivers, and glue.
- Commits us to the Phase 5 migration in `docs/architecture.md`: extract
  `thalos_game_state`, then peel `shipyard_editor`, `hud`, `map`,
  `structures` (with cleanup package D), later `clouds` out of
  `body_render`. Sequenced in `docs/backlog.md` (Track 1).
- Phase 1 (the state seam) is wide-but-shallow churn across hundreds of
  import sites; it wants a quiet worktree and one focused session
  (~200–300k tokens).
- The state crate's stability becomes load-bearing: adding a resource is
  cheap, reshaping one rebuilds everything — reshapes should batch.
