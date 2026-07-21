# Architecture Decision Records

This directory is the project's **decision log** — the persistent, git-committed record of
*why* Thalos is built the way it is. `CLAUDE.md` and the `docs/` specs describe *what* the
system is; ADRs capture the decisions behind it, including the alternatives that were rejected
and why. Read here before reopening a settled choice.

## How to use

- **Read first.** Before making or revisiting a non-trivial design choice, scan the
  timestamp-ordered ADR files and search their headings/content so you don't re-litigate a
  settled question or re-explore a rejected path.
- **Add one when a decision is made** — whenever you choose among alternatives, defer or cut
  scope, or reverse an earlier approach (architectural or not). Do it at decision time; the
  reasoning is otherwise lost to context compaction. Copy `template.md` to
  `YYYYMMDDTHHMMSSZ-kebab-title.md`, using the current UTC time (`date -u
  '+%Y%m%dT%H%M%SZ'`), and fill it in. Resolving a fork from the backlog's "Decisions pending"
  table always gets one.
- **One decision per record.** Keep each ADR short and focused.
- **Immutable once Accepted.** Don't rewrite an accepted decision — write a new ADR that
  supersedes it, and flip the old ADR's `Status` to `Superseded by
  ADR-YYYYMMDDTHHMMSSZ-short-title`.
- **Commit ADRs alongside the code** that realizes the decision (same change/slice).
- Historical decisions being migrated out of auto-memory get their original decision date plus
  a "recorded YYYY-MM-DD" note.

## Identity and ordering

ADR identifiers have the form `ADR-YYYYMMDDTHHMMSSZ-short-title`. The UTC timestamp is the
record's creation time; lexical filename order is therefore chronological recording order. The
semantic slug avoids unrelated branches competing for the same path. Causal order is explicit in
`supersedes` references rather than inferred from adjacent timestamps.

There is deliberately no hand-maintained index: it would make this README a merge hotspot. The
timestamp-sorted files are the canonical index. Use `rg --sort path '^# ADR-' docs/adr` to list
titles in order or `rg '<term>' docs/adr` to search decisions. See
`ADR-20260721T034338Z-distributed-chronological-identifiers` for the rationale and migration rule.
