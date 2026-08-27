# ADR-20260819T065009Z-landed-work-is-done: Landed work is `done`; user silence is acceptance

- **Status:** Accepted
- **Date:** 2026-08-19
- **Amends:** ADR-20260724T223339Z-lean-process-fast-agents (the verification *practice* stands; the backlog `verify` gate does not), ADR-20260720T185953Z-steering-harness (`verify` retired as a status)

## Context

`verify` existed because agents cannot run the game: landed-but-unobserved work sat in the
queue until a screenshot or a play session closed it. By 2026-08-19 that waiting room was
**163 of 420 backlog rows** — a median 1,561-character session diary per row, most asking
the user to look. Steer treated the pile as a standing candidate for a bundled pass; at
that size the pass is itself a project, so nobody took it, and every new landing added
another open row.

The user rule is *geen gehoor is goed gehoor*: no news is acceptance. Feedback that arrives
later is extra data, not a reason to leave the original item open-ended.

## Decision

1. **Retire `verify` as a backlog status.** Statuses are `next` / `wip` / `blocked` /
   `done` / `later`.
2. **Landed work is `done`.** Screenshot visual changes when you can — that is how the
   agent knows the change worked, not a queue state. If capture is blocked (dirty tree,
   no GPU), say so in the report and still mark `done`.
3. **User silence is acceptance.** A later complaint is a **new row** (or a note on the
   done item). It does not reopen the original as unfinished work.
4. **Unchanged:** do not launch the game; still run headless capture for visual work;
   still confirm scope with the user *before fixing a reported visual complaint*
   (wrong-read cost). Those are diagnosis and evidence rules, not a done-gate.

Existing `verify` rows flip to `done` in the same change.

## Alternatives

- **Keep `verify` with a TTL** (auto-`done` after N days). Rejected: the status itself is
  the attractor. Agents will still write the essay and the pile rebuilds until the TTL
  fires, which is the same policy with extra ceremony.
- **Keep `verify` only when this session could not screenshot.** Rejected: dirty-tree and
  no-GPU are the common case under parallel agents. That recreates the pile.
- **Leave the 163 open and only change the rule going forward.** Rejected: "what's next?"
  still has to skip a graveyard. Close them.

## Consequences

- Some `done` work will later be wrong. Cheap: a new row. Expensive was 163 open rows
  making the queue unreadable.
- A play-session ask in a report is optional extra data, never a status.
- ADR-20260724T223339Z's "unchanged, deliberately" verification *contract* now means the
  practices above, not a backlog state named `verify`.
