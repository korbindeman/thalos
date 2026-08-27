# ADR-20260819T070046Z-backlog-is-jsonl: The queue is one JSONL file, not a markdown table

- **Status:** Accepted
- **Date:** 2026-08-19
- **Amends:** ADR-20260819T065557Z-harness-has-a-reader (the queue format),
  ADR-20260720T185953Z-steering-harness (status authority path)

## Context

Splitting `docs/backlog.md` into a live table and a ledger still left two
failure modes that refill the queue:

1. Closing meant two-file table surgery, so agents left `done` in the live file
   or left `wip` forever.
2. Item cells had no length cap. Live `next`/`wip` rows were already
   session-length essays again (~26k tokens) the day after the split.

ADR-20260819T065557Z rejected a "YAML/JSON twin" because a second status
authority is the thing the markdown queue was invented to prevent. That still
holds. The format of the *one* authority can change.

## Decision

1. **`docs/backlog.jsonl` is the status authority.** One record per line. Same
   shape as `runtime.jsonl` / `tools.jsonl`: append, filter to read, rewrite
   one line to close.
2. **`just queue` reads only `next` / `wip` / `blocked` (and open forks).**
   `just queue <id>` prints one record. Do not read the file cover-to-cover.
3. **`just backlog done <id>` is how a row closes.** It strips `note` (and
   `est`/`deps`). Closed records stay as title + `refs` lines so IDs remain
   `rg`-able. There is no ledger file.
4. **A live `note` is capped at 360 characters.** Title + done-criteria. Session
   diary belongs in the commit or an incident. `just queue` reports overlong
   notes as hygiene.
5. **`docs/backlog.md` is a pointer**, not a table. Old essay cells survive in
   git history of the markdown, not in the jsonl.

## Alternatives

- **Keep markdown + `just queue`.** Rejected: the reader helps reading, not
  write incentives. The table will fill again.
- **Markdown live file + JSONL twin.** Rejected: two authorities. The 065557Z
  alternative still applies to a *twin*; it does not forbid replacing the
  format of the one file.
- **Append-only jsonl (never rewrite a line).** Rejected: status is mutable.
  Rewriting one line is the close. History is git.

## Consequences

- Records accumulate; essays do not. A closed line is ~150 bytes.
- Closing is one command, so "move the row to the ledger" cannot be skipped.
- Frozen `BL-1`–`BL-40` collisions remain as duplicate `id`s (both kept);
  `just queue <id>` prints an array when that happens.
