# ADR-20260819T065557Z-harness-has-a-reader: Always-on context is a budget; the queue has a reader

- **Status:** Accepted
- **Date:** 2026-08-19
- **Amends:** ADR-20260724T222409Z-claude-md-is-context-budgeted (enforced again),
  ADR-20260720T185953Z-steering-harness (the queue gained a reader and a ledger),
  ADR-20260819T065009Z-landed-work-is-done (done rows leave the live file)

## Context

The 2026-07-24 CLAUDE.md budget (352 lines) had grown back to 709 lines / 45 KB,
and Cursor injects both `AGENTS.md` and `CLAUDE.md` — a symlink made that **22k
tokens of identical manual** before any task. Steer said "read the backlog
first"; that file was 580 KB / ~144k tokens. `expert-review` was advertised in
every session and the skill directory was gone. `.agents/skills/steer` still
named architecture cleanup as the primary sprint.

The verification status had already been retired (ADR-20260819T065009Z). The
remaining failure was the same class: always-on context and "read this whole
file" instructions that were no longer executable.

## Decision

1. **`CLAUDE.md` is the operating manual.** Cut back to the 2026-07-24 budget:
   direction, a *where to look* table, steering pointers, quality bar,
   verification contract, load-bearing build rules, invariants/traps. Observability
   essay, crate catalog, command catalog, comparison-loop detail, and
   expert-review cadence stay in their owning docs.
2. **`AGENTS.md` is a pointer**, not a copy, so Cursor does not pay the manual
   twice. Agents whose product only loads `AGENTS.md` follow the pointer.
3. **`docs/backlog.md` is the live queue** (`next` / `wip` / `blocked`).
   **`docs/backlog-ledger.md` is search-only** `done` / `later`. IDs stay valid;
   `rg` both files. Closing a row means moving it.
4. **`just queue` is the reader**, analogous to `just diag`. Steer runs it
   instead of reading the backlog cover-to-cover.
5. **`.claude/skills` is canonical.** `.agents/skills` are symlinks. The
   `expert-review` skill is retired from the always-on harness; existing reports
   in `docs/reviews/` remain historical.

## Alternatives

- **Keep the AGENTS.md symlink.** Rejected: Cursor inlines both filenames, so a
  symlink is a 2× tax with no extra information.
- **YAML/JSON twin of the backlog.** Rejected: a second status authority, which
  is the thing the markdown queue was invented to prevent.
- **Restore expert-review in always-on context.** Rejected: a dangling skill
  path is noise; restore the skill first if the cadence comes back.

## Consequences

- A new agent session pays ~4k tokens of CLAUDE.md plus a ~500-byte AGENTS.md
  pointer, not 22k of duplicated encyclopedia.
- "What's next?" is `just queue` (~100 lines), then one row.
- Agents that mark `done` without moving the row leave stray live-file rows;
  `just queue` reports them as hygiene.
