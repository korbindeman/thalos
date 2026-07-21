# ADR-20260720T185953Z-steering-harness: Roadmap steering harness — backlog + steer skill + ADR / incident logs

- **Status:** Accepted
- **Date:** 2026-07-18

## Context

Thalos runs two long sprints (architecture cleanup, graphics fidelity) whose plans live in two
large docs with their own status markers, while decision rationale and bug forensics accumulated
in the agent's **private auto-memory** — invisible to other agents, other machines, and the git
history, and lossy under context compaction. There was no operational "what's next?" queue: each
session re-derived priority from ~900 lines of plan docs plus memory, and statuses in the plan
docs drifted (a "blocked" claim was already stale when re-checked). The pattern was borrowed from
the `ochre` repo's working setup (its ADR-0001).

## Decision

Adopt a four-part steering harness, all git-committed:

1. **`docs/backlog.md`** — the execution queue: status-tracked rows (`next`/`wip`/`verify`/
   `blocked`/`done`/`later`) reusing the plan docs' own IDs, with a "Decisions pending" table.
   The thalos-specific `verify` status captures the landed-but-not-runtime-verified state that
   dominates here (agents can't run the game).
2. **`steer` skill** (`.claude/skills/steer/SKILL.md`) — routes "what's next?" (propose + scope
   + stop for go-ahead), "add X / fix Y" (file then do), and vision talk (capture then decompose).
3. **`docs/adr/`** — durable decision records; write at decision time, immutable once accepted.
4. **`docs/incidents/`** — post-mortems for fixed non-obvious bugs (broadened from ochre's
   crash-only scope, since most hard Thalos bugs are visual/behavioral).

The plan docs remain the strategy layer and rationale home; the backlog holds status + pointers
only, and both flip in the same change.

## Alternatives

- **Keep plan-doc checkboxes as the only tracker** — rejected: checkboxes carry no queue
  semantics (no deps, no ranking, no verify-vs-done distinction), and "what's next?" stays a
  full-doc re-read.
- **Keep decisions in auto-memory only** — rejected: private to one user+machine, not versioned
  with the code that realizes the decision, and truncated summaries lose the "alternatives
  rejected" half that prevents re-exploration.
- **Ochre's work-lease scripts** (parallel agents in one working tree) — rejected: the jj
  workspace workflow already isolates parallel agents with separate working dirs and `target/`;
  leases would add machinery for a solved problem.
- **A parity ledger** — rejected: it tracks one capability × N frontends; Thalos has one
  frontend, and the nearest analogue (surface × lighting capability) already exists as
  `gfx §4`'s substrate tracking.

## Consequences

- "What's next?" becomes cheap and consistent across sessions; discovered work must become a
  backlog row (never a silent TODO), and landed work is honestly `verify` until observed.
- Small standing overhead: backlog row + plan-doc checkbox + spec doc move together in every
  landing change.
- Auto-memory narrows to genuinely session-personal facts; durable decisions and forensics
  migrate to ADRs / incidents over time (seeded with
  ADR-20260720T185954Z-analytic-planet-water-never-meshed through
  ADR-20260720T185956Z-replace-avian-owned-tgs-soft-solver, plus INC-0001).
