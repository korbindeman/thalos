# ADR-20260724T223339Z-lean-process-fast-agents: Records are exception-driven; agents work, then write

- **Status:** Accepted
- **Date:** 2026-07-24
- **Amends:** ADR-20260720T185953Z-steering-harness (the harness stands; its ceremony is cut)

## Context

The steering harness (ADR-20260720T185953Z) gave the project a coherent roadmap and durable
memory, and both are working. What it also produced, in roughly five days, was **54 ADRs and
41 incident post-mortems** — plus a rule that *every* fix gets a backlog row, a rule that
each change synchronize a backlog row, a plan-doc checkbox and a spec doc, and a rule to
read the ADR directory before any non-trivial choice.

Each rule was individually defensible; the sum is a tax paid on every task. The failure modes
are concrete:

- **Writing crowds out doing.** A one-line fix carried a row, a status flip, and a candidate
  post-mortem. The overhead did not scale with the work.
- **A log nobody can read is not memory.** "Read the timestamp-ordered records before a
  non-trivial choice" stops being executable at 54 records; agents skim or skip, and the
  genuinely load-bearing ADRs get the same attention as the routine ones.
- **The sync rule had already gone vestigial.** `rg '\[[ x]\]' docs/roadmap/*.md` matches
  **zero** checkboxes — the plan docs stopped carrying status, but the rule to keep it in
  sync stayed on the books.
- **Ceremony discourages the record we actually want.** A nine-section incident template with
  every section required is a reason to skip the post-mortem on a bug that deserved one.

## Decision

**Do the work; write the exception.** Concretely:

1. **The backlog tracks work that outlives the session** — multi-step items, anything handed
   back to the user, anything deliberately deferred. Found-and-fixed in the same change needs
   no row; the commit is the record. "Discovered work becomes a row, never a silent TODO"
   narrows to *deferred* work.
2. **The backlog is the only status authority.** Plan docs hold rationale and sequencing and
   carry no parallel checkboxes. The three-way sync collapses to one row, plus the spec doc
   when behavior actually changed.
3. **ADRs are for decisions expensive to reverse** and likely to be re-litigated or
   re-explored. Ordinary judgment calls go in the commit message. Not writing one is the
   default. Reopening a settled area means `rg '<topic>' docs/adr`, not reading the directory.
4. **Incidents keep their bar (non-obvious diagnosis) and lose their length.** Four sections —
   symptom, root cause, fix, recurrence signal — any of which may be dropped.
5. **`steer` stops gating.** "Add X / fix Y" means do it. The mandatory scoping brief shrinks
   to a few lines, and the one surviving hard stop is the "what's next?" go-ahead, where
   picking wrong costs a session.
6. **Infrastructure replacement no longer needs an announcement first** — only a trail after:
   what was replaced, why, with the docs updated in the same change.

**Unchanged, deliberately:** the verification contract (never launch the game; a visual change
that compiles but wasn't screenshotted is `verify`, not `done`), the invariants and traps
lists, diagnose-before-patching, and chronological identifiers. None of these is bureaucracy —
they are the rules whose violation costs a wrong diagnosis, a silent regression, or a
permanently wrong reference.

## Alternatives

- **Leave it and rely on judgment.** Rejected: the rules are written as unconditional
  ("every", "always", "in the same change"), and agents follow them literally — which is what
  produced the volume. Relaxing the *text* is the intervention.
- **Drop ADRs and incidents entirely; rely on git history.** Rejected: the incident log is the
  one artifact that has demonstrably stopped repeat bugs, and commit messages are not
  searchable by symptom. The user's constraint was explicit — keep the hard-earned bug
  knowledge.
- **Auto-expire or prune old records.** Rejected as premature: the cost is at *write* time and
  at *search* time, not storage; raising the bar fixes both without deleting history.
- **Fold incidents into a single append-only file.** Rejected: it becomes a merge hotspot and
  loses per-record identity, which is cited from Rust source and backlog rows.

## Consequences

- Fewer records, each carrying more weight. The ADR log stays readable by `rg` for longer.
- Some knowledge that would have become an ADR now lives only in commit messages — accepted:
  it was the class of decision nobody re-litigates.
- The backlog stops being a complete ledger of everything done; it is a queue of what's open
  or deferred. Git history is the ledger. This is the intended trade.
- Judgment calls move to the agent ("is this expensive to reverse?", "was this diagnosis
  non-obvious?"). Some will be called wrong in both directions. A missed post-mortem is
  cheaper than the throughput lost to writing all of them.
