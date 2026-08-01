# INC-20260801T035551Z: autoland flew a loop, then went around over the threshold

- **Reported:** 2026-08-01, user, from a `runway-approach` play session (screenshots + `runtime.jsonl` session `26268-1785554343309`)
- **Fixed:** same change
- **Areas:** `thalos_navigation::path`, `crate::route`, `crate::route_autopilot`, ND header, autopilot panel

## Symptom

"I set up a landing on the ND, the path looks nice. I put on autoland. It does
some of the stuff good, but a bit through the turn, it diverts, and shows a new
green path, the original route changes, it all looks weird, we're not aligned
for landing, and then, it just goes full throttle and flies off."

## What the recorder showed

265 `approach_ap` records, one per second, covering the whole flight:

| t (s) | |
|---|---|
| 0 | engaged, `TerminalCapture`, 29.5 km to go, 588 m above the runway |
| 95–146 | bank pegged at the 25° limit for ~50 s while cross-track ran out to −2000 m |
| 147 | cross-track hit the re-plan threshold → new plan; **dtg jumped 16.10 → 23.42 km** |
| 147–193 | same again, out to −1768 m |
| 194 | phase → `Final` **1.76 km left of the centreline**; **dtg collapsed 20.24 → 7.17 km and altitude error went 26 → 148 m in one frame**; pitch slammed +2.9° → −4.4°, throttle to zero |
| 194–238 | localizer pinned at −13° (3.5× full scale) the entire final |
| 238 | height dropped below 250 m → unstable-approach test became eligible and tripped instantly → go-around |
| 243 | → `Enroute`: gear up, flaps up, full throttle, climbing to the terrain-clearing cruise altitude, toward a fix 35 km behind the runway |

## Mechanism

**Three independent defects, each of which needs the others to produce this.**

### 1. The path projection teleported (root cause)

`LateralPath::closest` was a **global** nearest-point search over every leg,
with no memory of where the craft already was. Along-track distance is not just
a plot coordinate: `dtg` is derived from it, and `dtg` is the argument to the
entire vertical profile, the speed gates, and the approach phase.

A bank-limited join doubles back on itself — the ND preview's "overflown,
turning back" and "crosswind strip" cases are exactly this shape. The moment
the return leg got marginally nearer than the outbound leg, the projection
hopped, and **everything downstream teleported with it**. That is the 20.24 →
7.17 km frame: the craft had not moved and the plan had not changed
(cross-track went −1768 → −1761), only the leg the craft projected onto. The
autopilot read the result as "you are suddenly 148 m high on final" and dumped
the nose.

**The tell:** a one-frame step in `dtg_m` with `cross_track_m` essentially
unchanged. Real motion cannot do that.

### 2. The re-plan trigger fought the rejoin

Cross-track is measured against the route. A *rejoin* is by construction a path
that leaves the route in order to get back onto it — a bank-limited reversal
swings a full turn diameter clear. The drift trigger read that as "lost", threw
the plan away, and rebuilt it from the craft's current position; cross-track
reset to ~0 and grew again. Cycle time ~47 s, twice, for the whole approach.

That is the loop the user saw, and the "new green path" was the rejoin —
correct, unlabelled, and indistinguishable from the route it appeared to be
replacing.

### 3. One flag meant both "frozen" and "established"

`RouteState::established` was latched from `guidance.phase != Transition`, a
purely along-track test, and used for two unrelated jobs: inhibiting re-plans
(correct and geometric) and asserting the approach was stable (not remotely
true). So the plan froze while 1.76 km off the centreline, and from that moment
recovery was geometrically impossible. `Guidance::established` — the honest
"inside both full-scale needles" test — already existed and was used for
nothing but an asterisk on the PFD.

The go-around gate then compounded it: it read the **localizer** during the
join, where `thalos_navigation::guidance` states in as many words that the beam
is meaningless; it had no dwell, so one frame decided; and it only became
eligible below 250 m, so an approach that was unrecoverable at the final
approach point was flown for another 44 s before anything objected.

## Fix

- `LateralPath::closest_from(p, hint)` windows the projection around the
  previous along-track position (250 m back, 1000 m ahead — orders of magnitude
  above a frame's travel, an order of magnitude below the 13 km snap). The
  caller holds the hint, so the function stays pure per ADR-20260730T005746Z,
  the same arrangement as the rejoin's capture hint. `closest` remains for
  seeding, and is now documented as the discontinuous one.
- The drift re-plan only fires when the rejoin planner has already said there
  is no flyable way back, with a 15 km backstop for the pathological case.
- `plan_frozen` (geometric, irreversible) split from `established` (live,
  needle-based). The go-around gate reads the latter, only on `Final`/`Flare`,
  with a 5 s dwell.
- `LandNotice` carries the reason to the screen and the lane from one place.

## Recurrence tells

- `just diag` → **`route_replan_churn`** (the loop), **`route_rejoin_churn`**
  (the follower is not holding a path it was given), **`land_go_around_churn`**
  (never landed).
- `appr_frame` now logs **achieved** `bank_rad`/`fpa_rad` beside the commanded
  values, plus `established`, `plan_frozen`, and `unstable_s`. A bank pegged at
  the limit is now separable into "the aircraft will not roll" and "the plan is
  steering the wrong way", which one number could not distinguish.
  **The first version of that instrument was itself wrong**: it took
  `normalize(slf_position_m)` as "up", but that is an offset *inside* the
  surface-local frame, not a radial from the body centre, and it reported 50° of
  bank against a 12° command. Radial up comes from
  `surface_local::radial_up(&bubble.frame, position)`, the same call
  `control_bus` makes to build `FlightState`. An instrument that has not been
  checked against a known value is a hypothesis, not evidence.
- Pinned by `a_global_projection_snaps_legs_where_the_route_doubles_back` and
  `a_hinted_projection_stays_on_the_leg_the_craft_is_flying` in `path.rs`.

## Round 2: it still never intercepted

The three fixes above held — session `20324-1785556887804` ran distance-to-go
57.20 → 8.94 km monotonically, re-planned **zero** times, and went around 5 s
after the final approach point with 8.9 km still to run. But cross-track went
`−750 → 0 → +1600 → +44 → +2700 → +1630`, arriving at the FAP 1.6 km off with
the localizer at 11.5°. The failure had become clean and early; it was still
the same failure.

Two more causes, both about *following* rather than *planning*:

### 4. The rejoin was a second path authority

`compute_guidance` took the rejoin as a steering **cue** while measuring
cross-track against the untouched route. So the ND drew one path and the
autopilot flew another — the user's report was literally "it takes turns
different than the ND path" — and the numbers called the craft kilometres off
course for the entire time it was correctly flying back.

Worse, the cue was replanned every frame, so its aim point advanced with the
craft. The recorded signature is unmistakable: 385 s at a **flat 12.4° bank**
while distance from the route grew to 2.7 km. A law trying to close on a fixed
target would have been at the 25° limit — the fallback intercept alone demands
68° at that offset. A steady moderate bank that never arrives is a craft
chasing something that keeps moving.

**Fix:** the rejoin is now *committed into* the route
(`LateralPath::splice_rejoin`) rather than run beside it. `RouteState::active_path`
is the one thing that is drawn, flown, and measured. Commits are rate-limited to
one per 20 s and refused once the plan is frozen, because a rejoin is a decision
the craft then spends a minute or two flying. Distance-to-go is the invariant
across a splice, which is what lets the vertical profile survive it and what the
track hint is carried across as.

### 5. The lateral law had no curvature feedforward

Bank came only from heading error. Holding a curved segment costs a standing
turn rate `v·κ` *before* any error exists, so a heading-only law has to grow an
error first in order to produce that bank — a standing cross-track offset by
construction, proportional to curvature. On the 3 km-radius joins this planner
produces, that is kilometres.

**Fix:** `PathPoint::curvature`, and `bank = f(v·κ + clamp(k·heading_error))`.
The feedforward is what holding the path costs; the correction is what returning
to it costs; the bank limit covers both.

## What this was *not*

The long level segment early in the approach looked like "the autopilot never
descends", and it is not a defect: the craft was already at platform altitude,
so `VerticalProfile::plan` correctly held it there until top of descent while
the autothrottle bled 113 → 86 m/s at idle. The height never came down because
*`dtg` never came down* — defects 1 and 2 — not because the vertical law was
wrong. No change was made to the descent profile.
