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

This project self-steers through these documents plus this skill:

- **`docs/backlog.md`** — *execution*. A rolling, status-tracked queue of concrete scoped
  items across the active sprints. Answers *what's the next thing*.
- **The sprint plan docs** — *strategy*: `docs/roadmap/architecture_cleanup.md` (`clean`, the primary
  sprint) and `docs/roadmap/graphics_fidelity.md` (`gfx`, secondary). They hold the rationale, package
  designs, and open forks. Slow-changing. Answers *why* and *in what order*. CLAUDE.md's
  "Current focus" section names which sprint is primary.
- **`docs/adr/`** — resolved decisions (why things are the way they are). A fork resolving in
  conversation gets an ADR, not just a chat.
- **This skill** — the routing + scoping procedure that ties them together.

**Always read the backlog at the start of an invocation** (plus the plan-doc section any
candidate item points into), then route to one of three modes.

## Detect the mode

- **Mode 3 — "what's next?"** — user asks what to work on / for the next item, no specific
  task named. → Scope the next item.
- **Mode 1 — concrete work** — user names a specific feature or bug ("add X", "fix Y").
  → Capture, then do.
- **Mode 2 — vision** — user talks about direction, aspiration, "where this should go", or
  reopens a fork. → Capture to the plan docs, then decompose into work.

When genuinely ambiguous, ask one clarifying question rather than guessing.

## Mode 3 — "what's next?" (propose + scope, then stop)

**Contract: propose the next item, scope it into an actionable brief, then stop for the
user's go-ahead.** Do **not** start implementing, and do **not** flip the item to `wip`,
until the user greenlights.

1. **Reconcile.** Skim `jj log` / `jj st` and the working tree against `backlog.md`. Correct
   any status that's gone stale (marked `next` but already landed; `verify` already confirmed
   in a memory note or doc checkbox). A quick sanity pass, not an audit.
2. **Candidate pool** = items with status `next` (no unmet deps), **plus the `verify` queue
   as a standing candidate**: when several landed items are waiting on runtime verification,
   bundling them into one verification session is often the highest-value next step —
   unverified work compounds risk under further churn. Prefer the agent-servable half first
   (`just screenshot` presets, `just preview`, `just ui-preview` — read the PNGs), and package
   what genuinely needs a live eye as one tight checklist for the user's next play session
   rather than N scattered asks.
3. **Rank**, in order: (a) primary sprint before secondary (CLAUDE.md "Current focus");
   (b) gating risk / unblocks the most downstream items; (c) explicit priority notes;
   (d) smaller estimate as a momentum tiebreaker.
4. **Pick the top item** and write the brief:
   - **Item** — id + title.
   - **Why now** — where it sits, what it unblocks, why it beat the others.
   - **Goal** — the outcome, one line.
   - **Approach** — the sketch: key files / modules, the shape of the change.
   - **Exit criteria** — split **agent-verifiable** (compile, clippy, unit tests, headless
     screenshots/previews) from **user-verifiable** (play-session checklist). An item whose
     user half is pending lands as `verify`, not `done`.
   - **Risks / unknowns.**
   - **Est.**
5. **Stop.** Hand the brief back. On go-ahead: flip the item to `wip` in `backlog.md`, then
   begin. If the user redirects, pick again.

If the honest answer is *"the real next step is a decision, not code"* (a "Decisions pending"
fork gates everything queued), say so and switch to Mode 2 on that fork.

## Mode 1 — concrete work (capture + do)

1. **File it** in `backlog.md` as a `next` item — right track, reuse the plan doc's ID if one
   exists, else mint `BL-YYYYMMDDTHHMMSSZ-slug` from the current UTC time (`date -u
   '+%Y%m%dT%H%M%SZ'`) — never a sequential `BL-n`, which collides across parallel branches
   (`ADR-20260722T170714Z-one-chronological-identity-rule`). Note deps / est. Even a quick fix
   gets a one-line row; the backlog is the record of what was done.
2. **Do the work** — normal implementation, obeying `CLAUDE.md` (diagnosis-before-patching
   for bugs; the sprint rules of engagement in `clean §4`).
3. **Close the loop** — mark the item `verify` (or `done` if fully observed working), update
   the plan doc's checkbox and the relevant spec doc **in the same change**. If the work
   surfaced new work, file those as `next`/`later` rows rather than dropping them — no
   deferred TODOs outside the backlog. If a bug's root cause was non-obvious, consider an
   incident post-mortem (`docs/incidents/`).

## Mode 2 — vision (capture to the plan docs, then decompose)

1. **Discuss** and sharpen the aspiration with the user.
2. **Record it at the right altitude** — extend a plan doc's scope, revise a fork's options,
   or (for a genuinely new area) start a design doc in the appropriate `docs/` category from
   `docs/README.md`; do not add another root doc by default. If a fork
   is *resolved*, write an **ADR** (`docs/adr/`, per its README) and flip the fork's row in
   the backlog's "Decisions pending" table.
3. **Decompose** the new direction into concrete `backlog.md` items (`next` or `later`).
   This is the step that turns vision into pickable work — don't skip it.
4. **Don't implement** in this mode unless the user then asks. Mode 2 produces plans + items,
   not code.

## Keeping the trackers honest (every mode)

- Flip statuses the moment work lands; a stale backlog makes "what's next?" wrong. `verify`
  → `done` only on observed behavior (user confirmation or a read screenshot), with the date.
- The backlog row, the plan doc's checkbox, and the spec doc move **in the same change** —
  never let them tell different stories.
- Respect the **rolling window** — itemize the active sprints; the graphics later-pool stays
  at doc granularity in `gfx §4` until pulled in.
- New work discovered mid-task → a row, never a silent TODO.
- One source of truth — rationale lives in the plan docs / specs / ADRs; the backlog holds
  status + pointers.
