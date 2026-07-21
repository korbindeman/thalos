# ADR-20260721T034338Z-distributed-chronological-identifiers: ADR identifiers are distributed chronological timestamps

- **Status:** Accepted
- **Date:** 2026-07-21

## Context

Sequential ADR numbers require every branch to allocate from one shared counter. Concurrent
branches have repeatedly created the same identifier and filename for unrelated decisions,
forcing renumbering during integration. They also both append to the manually maintained table in
this directory's README, creating a second merge hotspot.

ADR filenames should retain chronological ordering, remain readable in references, and be
mintable independently. No decentralized identifier can guarantee a globally exact real-time
order: branch clocks can differ and branches cannot observe one another. A central allocator
would provide that guarantee at the cost of coordinated, online ADR creation.

## Decision

New ADR identifiers use their UTC creation timestamp, precise to one second, followed by a
semantic slug:

```text
ADR-YYYYMMDDTHHMMSSZ-short-title
```

The filename is the identifier without the `ADR-` prefix plus `.md`. UTC timestamps make the
directory's lexical order its chronological recording order; the slug makes independently
authored files distinct and keeps references meaningful. If two branches describe the same
decision at the same second with the same slug, that collision is meaningful and must be
reconciled rather than hidden.

The ADR README is instructions only. It does not contain a manually maintained index: the
timestamp-sorted files are the canonical index, while searching headings provides a content
index.

Existing sequential records are migrated using their first Git recording timestamp. Records
introduced in the same commit receive consecutive one-second timestamps in their established
numeric order. Their original decision dates remain in their metadata.

## Alternatives considered

- **Renumber at merge:** preserves compact identifiers, but churns filenames, headings, and all
  references whenever branches collide.
- **Reserve number ranges:** moves coordination into range allocation and produces misleading
  numeric order.
- **Central allocator:** guarantees total order, but prevents independent and offline creation.
- **UUID or ULID:** avoids collisions, but UUIDs do not sort chronologically and both forms are
  less readable than a timestamp plus semantic slug.

## Consequences

- Branches can normally add ADRs without touching the same path or shared index.
- Lexical filename order exactly matches recorded UTC timestamp order.
- Causal order is expressed explicitly with `supersedes` references, not inferred from adjacent
  identifiers.
- Clock skew can misorder records authored on different machines; exact global real-time order
  remains intentionally weaker than independent ADR creation.
