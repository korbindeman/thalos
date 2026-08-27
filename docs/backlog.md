# Backlog

The queue is [`backlog.jsonl`](backlog.jsonl) — the **only** status authority.
Read it with `just queue`. Do not read the jsonl cover-to-cover to pick work.

```bash
just queue                 # next / wip / blocked titles
just queue -- --json       # machine use
just queue HYD-1           # one record
just backlog done HYD-1    # close: strips the note, leaves a title-only line
just backlog later HYD-1
just backlog wip HYD-1
just backlog note HYD-1 "done-criteria, max 360 chars"
just backlog add --track ntr --title "…" --note "done-criteria"
```

**What gets a row:** work that **outlives the current session**. Same-session
finish: no row; the commit is the record.

**Live record:** `id`, `track`, `status` (`next` / `wip` / `blocked`), `title`,
optional `note` (done-criteria, **max 360 characters**), `est`, `deps`, `refs`.
Closing (`done` / `later`) deletes the note. Session narrative goes in the
commit or an incident, not here.

**IDs:** reuse a plan doc's own ID when it has one; otherwise mint
`BL-YYYYMMDDTHHMMSSZ-slug` from `date -u '+%Y%m%dT%H%M%SZ'`. Never allocate
the next `BL-n`. `BL-1`–`BL-40` are frozen. Landed work is `done`; user silence
is acceptance (ADR-20260819T065009Z).

**Search:** `rg ID docs/backlog.jsonl`. Essays from the old markdown tables
survive in git history, not in the queue.

**Refs:** `ntr §N` · `clean §N` · `gfx §N` = roadmap · `ADR-…` = adr/ ·
`INC-…` = incidents/.
