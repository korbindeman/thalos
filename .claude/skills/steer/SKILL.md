---
name: steer
description: >-
  Roadmap steering harness for Thalos. Use when the user asks "what's next?" /
  "what should I work on" (no specific task named); when discussing future vision,
  long-horizon direction, or reopening a strategic fork that should be captured and
  turned into work; or when adding a feature / filing a bug that should be tracked.
  Reads docs/backlog.md (execution) and the sprint plan docs (strategy) and routes
  to one of three modes — scope the next item, evolve the vision, or capture
  concrete work — keeping the backlog in sync as work lands.
---

# Steer — the roadmap harness

Two documents plus this skill:

- **`docs/backlog.md`** — *execution*. The status-tracked queue and the only status
  authority. Answers *what's the next thing*.
- **The sprint plan docs** (`docs/roadmap/`) — *strategy*: rationale, package designs,
  open forks. Slow-changing. Answers *why* and *in what order*. CLAUDE.md's "Current
  focus" names which sprint is primary.

**Read the backlog first** (plus the plan-doc section a candidate points into), then
route. The point of this skill is to keep one coherent direction — not to add steps
between the user and working code. Bias toward doing.

## Modes

- **"what's next?"** (no task named) → **§ Propose.**
- **"add X" / "fix Y"** (a concrete thing) → **§ Do.**
- **Direction / vision / reopening a fork** → **§ Capture.**

Genuinely ambiguous? Ask one question.

## § Propose — "what's next?"

1. **Sanity-check** the backlog against `jj log` / `jj st`: fix statuses that have
   obviously gone stale. A skim, not an audit.
2. **Pick.** Candidates = `next` with no unmet deps, **plus the `verify` queue as a
   standing candidate** — when several landed items await verification, one bundled
   verification pass is often the highest-value move (serve the agent half yourself via
   `just screenshot` / `just preview` / `just ui-preview` and read the PNGs; package
   what needs a live eye as one checklist for the user's next session). Rank: primary
   sprint first, then what unblocks the most, then smaller as a tiebreaker.
3. **Propose it in a few lines** — the item, why it beat the others, the shape of the
   change, and what "done" looks like (split agent-verifiable from user-verifiable).
   Long enough to greenlight, not a document.
4. **Stop** for the go-ahead — this is the one place stopping is the contract, since
   picking the wrong next thing wastes a whole session. On go: flip to `wip`, start.

If the real next step is a decision rather than code, say so and switch to **§ Capture**
on that fork.

## § Do — concrete work

**Do it.** Diagnose before patching (CLAUDE.md "Bug fixing"), then implement.

- **File a row only if the work won't finish now** — multi-step, handed back to the
  user, or deliberately deferred. A same-session fix needs no row.
- **Close the loop** on rows that exist: `verify` when it lands, `done` when observed
  working. Update the spec doc if behavior changed. That's the whole bookkeeping.
- Work you discover and defer becomes a row; work you discover and fix doesn't.

## § Capture — vision and forks

1. **Sharpen** the direction with the user.
2. **Write it where it belongs** — extend a plan doc's scope, revise a fork's options,
   or start a design doc in the right `docs/` category. A *resolved* fork that would be
   expensive to reopen gets an **ADR** and flips its row in the backlog's "Decisions
   pending" table; an ordinary call just gets written down.
3. **Decompose** into `next` / `later` rows. This is the step that turns vision into
   pickable work — don't skip it.
4. **Don't implement** here unless the user then asks.

## Keeping it honest

- Flip statuses when work lands — a stale backlog makes "what's next?" wrong. `verify` →
  `done` only on observed behavior (user confirmation or a screenshot you read).
- Status lives in the backlog, rationale in the plan docs / specs / ADRs. One place each.
- Itemize the active sprints only; later pools stay at doc granularity until pulled in.
