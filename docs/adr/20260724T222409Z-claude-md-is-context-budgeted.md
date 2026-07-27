# ADR-20260724T222409Z-claude-md-is-context-budgeted: CLAUDE.md is a context budget, not a codebase encyclopedia

- **Status:** Accepted
- **Date:** 2026-07-24

## Context

`CLAUDE.md` is loaded into **every** agent context, on every task, before the
agent knows what it is about to do. It had grown to **1,962 lines / 138 KB**
(~35k tokens) — roughly a quarter of it a module-by-module narration of the game
crate, a second annotated index of `docs/`, and full-text restatements of policy
that already had an owning document (`docs/development/build_speed.md`,
`docs/README.md`, the subsystem specs).

The growth was structural, not careless. Every convention in the file says
"record the lesson in CLAUDE.md": the incident template asks for a promoted
gotcha, the steering harness asks for a sprint status block, the architecture
rule asks for an announcement. Each addition was individually correct and the
sum was a file that spends a large, permanent share of every context window on
detail that the specific task almost never needs — while the rules that *do*
apply to every task (don't launch the game, screenshot before calling terrain
work done, one authority per concern) compete with it for attention.

## Decision

`CLAUDE.md` is budgeted. It carries only:

1. What Thalos is and the current sprint direction (with pointers, not plans).
2. How the project steers — backlog / ADR / incident conventions.
3. The standing quality bar and the bug-fixing loop.
4. Commands, and the **verification contract**: what an agent may run, must run,
   and may never run.
5. Build/iteration rules whose violation costs a rebuild or a wrong diagnosis.
6. The invariants list — one authority per concern, crate boundaries, and the
   traps that are silent when broken.

Everything else moves to the document that owns it, and `CLAUDE.md` links there.
Concretely, in this change:

- Module-level crate anatomy, the data flow, the assets list, and the terrain
  consumer detail → `docs/architecture.md` (new *Crate anatomy* section).
- Bevy 0.19 API notes → `docs/development/bevy.md` (new). The two non-negotiable
  render-ordering rules stay inline, because getting them wrong is a runtime
  regression an agent will not attribute correctly.
- The annotated `docs/` index → deleted; `docs/README.md` is the single map.
- Toolchain policy → already owned by `docs/development/build_speed.md`; only the
  load-bearing subset stays.

Result: 1,962 → 352 lines (138 KB → 20 KB), with no content deleted except the
duplicate documentation index.

**The standing rule this creates:** adding to `CLAUDE.md` is a claim that the
fact belongs in *every* agent's context. A lesson that applies to one subsystem
belongs in that subsystem's spec; a lesson that applies to one class of change
belongs in the incident or the skill. Promote to `CLAUDE.md` only what an agent
must know *before* it knows what it is working on — and when promoting, prefer
one line plus a pointer over a paragraph.

## Alternatives considered

- **Leave it.** Rejected: the file grows monotonically under the existing
  conventions, and the cost is paid on every task by every agent.
- **Split into `CLAUDE.md` + auto-loaded includes.** Rejected: same token cost,
  worse discoverability, and it hides the budget rather than enforcing it.
- **Delete the moved detail outright.** Rejected: the crate anatomy is genuinely
  useful when an agent is orienting in unfamiliar code — it just should be
  fetched on demand, not preloaded.

## Consequences

- Agents needing module-level orientation must Read `docs/architecture.md`. That
  is one tool call, on the tasks that actually need it.
- The `docs/README.md` map becomes load-bearing for navigation; it was already
  the declared canonical map.
- Incident/ADR conventions that say "promote the lesson to a CLAUDE.md gotcha"
  now mean "promote it *if* it clears the every-context bar; otherwise put it in
  the spec." The incident template and `docs/incidents/README.md` already phrase
  this as a choice between the two.
