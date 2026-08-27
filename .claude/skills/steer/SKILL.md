---
name: steer
description: >-
  Roadmap steering harness for Thalos. Use when the user asks "what's next?" /
  "what should I work on" (no specific task named); when discussing future vision,
  long-horizon direction, or reopening a strategic fork that should be captured and
  turned into work; or when adding a feature / filing a bug that should be tracked.
  Runs `just queue` (execution) and reads the sprint plan docs (strategy).
---

# Steer — the roadmap harness

- **`just queue`** — the reader. Next/wip/blocked, truncated. Do **not** read
  `docs/backlog.jsonl` cover-to-cover to pick work.
- **`just queue <id>`** — the one record you take.
- **`docs/backlog.jsonl`** — the status authority (ADR-20260819T070046Z).
- **Sprint plans** (`docs/roadmap/`) — strategy. CLAUDE.md "Current focus"
  names which sprint is primary.

Bias toward doing.

## Modes

- **"what's next?"** → **§ Propose.**
- **"add X" / "fix Y"** → **§ Do.**
- **Direction / vision / reopening a fork** → **§ Capture.**

## § Propose — "what's next?"

1. **`just queue`.** Sanity-check against `git log` / `git status` only if a status
   looks obviously stale.
2. **Pick** from `next` with no unmet deps. Rank: primary sprint, then what
   unblocks the most, then smaller. Landed work is `done` — do not treat
   silence as unfinished (ADR-20260819T065009Z).
3. **Open that one record** (`just queue <id>`) plus the plan-doc section it
   points at.
4. **Propose in a few lines** — the item, why it beat the others, the shape
   of the change, what done looks like. Stop for the go-ahead. On go:
   `just backlog wip <id>`, start.

If the real next step is a decision, switch to **§ Capture**.

## § Do — concrete work

**Do it.** Diagnose before patching (CLAUDE.md "Bug fixing"), then implement.

- File a row only if the work won't finish now. **Note:** title +
  done-criteria, max 360 characters (`just backlog add --track … --title …`
  / `just backlog note <id> "…"`).
- Close the loop: **`just backlog done <id>`** (strips the note). Screenshot
  visual work when you can; that is evidence, not a second status. Later
  disagreement is a new row.
- Discovered-and-deferred → a row; discovered-and-fixed → no row.

## § Capture — vision and forks

1. Sharpen with the user.
2. Write it in the plan/spec. A resolved fork that would be expensive to
   reopen gets an ADR and `just backlog resolved <id>`.
3. Decompose into `next` / `later` (`just backlog add` / `just backlog later`).
   Don't skip this.
4. Don't implement unless the user then asks.

## Keeping it honest

- Itemize the active sprints only; later pools stay at plan-doc granularity
  until pulled into `next`.
- Status lives in the jsonl; rationale in plan docs / specs / ADRs.
