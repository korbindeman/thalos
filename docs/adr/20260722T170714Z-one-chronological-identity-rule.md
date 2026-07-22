# ADR-20260722T170714Z-one-chronological-identity-rule: One chronological identity rule for every mintable record

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

ADR-20260721T034338Z solved distributed identifier collisions for ADRs by replacing the shared
sequential counter with a UTC creation timestamp plus a semantic slug, and by deleting the
hand-maintained directory index. The reasoning was general, but the change was applied only to
`docs/adr/`. Incident post-mortems and backlog items kept their shared counters, so the problem
it diagnosed simply moved to the systems that were left alone.

That is no longer hypothetical. At the time of writing, `INC-0019` denotes two unrelated
incidents in two live checkouts: `0019-sccache-activation-scoped-to-one-directory.md` on `main`
and `0019-terrain-trainer-large-canvas-selected-small-reference.md` in the terrain worktree.
`main` has since allocated `0020` and `0021`, so the terrain record must be renumbered to `0022`
at merge — renaming the file, its heading, its index row, and every reference to it. Backlog
counters have diverged the same way: `BL-37` on `main`, `BL-40` in the rebase worktree, `BL-33`
in terrain-status, all drawn from one nominal sequence. Seventeen `BL-` identifiers are cited in
Rust source, so a collided backlog number lands in permanent artifacts, not just a queue.

`docs/incidents/README.md` also retained the manually maintained index table that the ADR
directory deleted. It is a second merge hotspot on the same path, and it had already drifted out
of sync with its own directory (22 files, 21 rows).

The remaining question was not *whether* the timestamp rule works — it is ratified and in daily
use — but whether agents should hold one identity rule or three, and what to do with the
identifiers already in circulation.

## Decision

Every independently mintable record uses one identity form:

```text
<KIND>-<YYYYMMDDTHHMMSSZ>-<kebab-slug>
```

This covers ADRs (`ADR-`), incident post-mortems (`INC-`), and backlog items (`BL-`). Where the
record is a file, the filename is the identifier without its kind prefix, plus `.md`; the
`ADR-`/`INC-` prefix appears in headings and references, never in filenames, so lexical
directory order stays chronological recording order.

No record directory carries a hand-maintained index. The timestamp-sorted files are the index;
`rg` provides content search.

**Existing sequential identifiers are frozen, not migrated.** `INC-0001`–`INC-0021` and `BL-1`–
`BL-40` keep their numbers, headings, filenames, and every existing reference. They remain valid
citations permanently. Only newly minted records use the timestamp form, so the old numbers age
out of the working set naturally rather than through a reference-rewriting pass across three
live worktrees.

Plan-doc-owned identifiers (`CL-A`…`CL-G`, `F1`–`F9`, `W`-numbers, `TM`, `C1`) are unaffected:
they are allocated by a single owning document, not by branches racing a counter.

## Alternatives

- **Migrate existing incidents and backlog items to timestamps** — rejected for now. It would
  rewrite roughly 74 references across `docs/` and `crates/` plus 23 cross-links inside
  `docs/incidents/`, and would invalidate `INC-` references in all three live worktrees,
  guaranteeing merge conflicts in exactly the files the change is meant to protect. The
  consistency gain does not pay for churn inflicted on in-flight branches. A single migration
  pass once no branch is mid-flight remains available if mixed identifiers become a real
  drag.
- **Date-only identifiers (`BL-20260722-slug`)** — rejected. Shorter, and the slug makes
  same-day collisions unlikely, but it introduces a second rule that differs from ADR/INC in a
  way an agent must remember and can get wrong. One rule that is slightly verbose beats two
  rules that are each slightly convenient.
- **Leave `BL-n` sequential** — rejected. Backlog rows look transient, but 17 of them are cited
  in Rust source, so they outlive the queue and a collision becomes a wrong permanent
  reference.
- **Keep the incidents index table, generate it from the files** — rejected. It removes the
  drift but not the merge hotspot: every branch still rewrites the same generated rows. The ADR
  directory already demonstrates that no index is needed.
- **Renumber at merge** — rejected for the reasons in ADR-20260721T034338Z, now with direct
  evidence of the cost.

## Consequences

- Branches can add incidents and backlog items without touching a shared counter or a shared
  index, which was the last remaining source of identifier collisions between worktrees.
- The terrain worktree's `INC-0019` still has to be reconciled by hand, as the last collision
  under the old scheme. Reconcile it by giving that record a timestamp identifier rather than
  renumbering it to `0022`.
- Incident and backlog directories read mixed — short legacy numbers alongside timestamped ones
  — until the legacy set falls out of use. This is accepted deliberately in exchange for not
  breaking in-flight references.
- Backlog identifiers become long in table cells and in Rust comments. Accepted as the cost of a
  single rule; the slug keeps them readable at the reference site.
- `docs/incidents/README.md` loses its index, so "what incidents exist?" becomes an `rg` query
  rather than a table read. The same tradeoff the ADR directory already accepted.
