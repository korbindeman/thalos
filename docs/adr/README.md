# Architecture Decision Records

The persistent record of *why* Thalos is built the way it is — the choices, and the
alternatives that were rejected. `CLAUDE.md` and the `docs/` specs describe *what* the
system is.

## The bar

Write an ADR when a decision is **expensive to reverse** and would otherwise be
re-litigated or re-explored:

- an architecture seam or an authority boundary,
- a path you tried and rejected that a future agent will be tempted to retry,
- a constraint that looks arbitrary from the code alone,
- a fork resolved out of the backlog's "Decisions pending" table.

Everything else — ordinary judgment calls, local design, "I picked B because it was
simpler" — belongs in the commit message. **Not writing an ADR is the default.** The log
is only useful if reading it is cheap.

## How to use

- **Search, don't scan.** `rg '<topic>' docs/adr` before reopening a settled area;
  `rg --sort path '^# ADR-' docs/adr` lists titles in chronological order. There is no
  hand-maintained index — it would be a merge hotspot.
- **Write it at decision time.** The reasoning is otherwise lost to context compaction.
  Copy `template.md` to `YYYYMMDDTHHMMSSZ-kebab-title.md` using `date -u
  '+%Y%m%dT%H%M%SZ'`. One decision per record; keep it short — **Alternatives** is the
  section that earns its keep.
- **Commit it alongside the code** that realizes the decision.
- **Immutable once Accepted.** Don't rewrite one — write a new ADR that supersedes it and
  flip the old `Status` to `Superseded by ADR-YYYYMMDDTHHMMSSZ-short-title`.

## Identity

`ADR-YYYYMMDDTHHMMSSZ-short-title`; the filename is that without the `ADR-` prefix, so
lexical order is chronological. The timestamp is creation time and the slug keeps parallel
branches off the same path — never allocate "the next number." Causal order is explicit in
`supersedes` references. See `ADR-20260721T034338Z-distributed-chronological-identifiers`.
