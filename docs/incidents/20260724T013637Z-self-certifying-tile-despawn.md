# INC-20260724T013637Z — tile despawn certified coverage with the tile itself; black holes while approaching terrain

**Status:** fixed (simulation-verified; fly-through pending)
**Area:** NTR-X1 standard-path tile streaming — despawn/coverage logic

## Symptom

Flying toward the ground on the tile renderer (`THALOS_TILE_RENDERER=1`,
Mira), large black tile-shaped holes opened in the terrain — worst while
descending into craters — and persisted through two prior mitigation rounds
(a 0.15 s despawn grace, then 0.5 s + 20 frames + per-level depth bias). The
runtime coverage audit proved the holes **logical**: up to ~half the desired
set had no resident tile anywhere in its ancestor/descendant column.

## How it was pinned

Headless captures could not reproduce it (capture cameras teleport; the bug
needs selection churn from continuous motion). The streaming logic is pure
data, so the despawn decision (`despawn_ready`) and the coverage audit
(`uncovered_desired`) were extracted and driven by a **descent simulation
test** (`streaming_tests::descent_keeps_every_desired_tile_covered`): plunge
200 km → 30 m, 250 m/s sweep at 30 m altitude, climb-out and re-dive, with
the production landing rate (4 synthesis workers × ~32 ms/tile) and the
in-flight cap, asserting the coverage invariant every 60 Hz tick. First run
(vertical descent, 24-wide parallelism) stayed green; matching production
parallelism and adding the low sweep + re-dive reproduced the failure
deterministically in ~2 s, and a per-column event log gave the full history:
an L7 despawned **while its L8 child was still pending**, under a certificate
that could not possibly hold.

## Root cause

`covered_by_resident(key)` begins with `if resident.contains_key(&key)
{ return true; }` — correct for its other callers, but the despawn split-case
called it on **the tile being considered for removal, which is still in the
resident map**. Every stale tile whose desired level had moved below it
therefore certified coverage **with itself**, entered retirement, and
despawned after the grace period — abandoning its still-pending children.
Under approach (selection deepening faster than the 4-thread pool lands
tiles) that is precisely a hole factory; at rest everything desired is
resident, so settled captures never showed it.

The two prior mitigations failed because they only delayed a certificate
that was never valid. The "sound" transitivity argument for simultaneous
despawn silently rested on the same self-referential certificate.

## Fix

The split case now descends from the tile's **children** (`key.children()
.all(covered_by_resident)`), never the self; a max-level tile cannot certify
via split at all. The simulation test — plunge, low sweep, re-dive — passes
with the fix and is a permanent regression gate.

## Prevention / standing rules

- **A predicate reused across callers must state its frame of reference.**
  "Is this footprint covered by residents" quietly meant "including you" —
  fine for the audit (which pre-excludes self), fatal for the removal
  decision. When a coverage/validity check participates in removing the very
  thing it checks, exclude the subject explicitly.
- **Streaming invariants get simulation tests, not just settled captures.**
  Stills verify states; streaming bugs live in transitions. The extracted
  pure core (`despawn_ready` + `uncovered_desired`) plus a scripted camera
  path reproduced in 2 s what three user fly-throughs could only
  demonstrate. Match the simulation's throughput to production (worker
  count, latency) — the 24-wide first attempt hid the bug.
- **The runtime coverage audit stays** (2 s cadence, warn-level): it is the
  cheap discriminator between "selection/despawn bug" and "presentation
  problem", and it caught this one the first time a user flew with it.

## Recurrence tells

- `tile terrain: N desired tiles LOGICALLY uncovered` warnings during
  approach/descent.
- Black tile-shaped holes that appear only during motion toward terrain and
  heal once stationary.
