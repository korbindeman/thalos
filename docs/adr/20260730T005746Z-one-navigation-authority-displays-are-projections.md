# ADR-20260730T005746Z-one-navigation-authority-displays-are-projections: Route navigation is one authority; every display is a projection of it

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

The ND's runway symbology was decorative: strips drew at a fixed symbol size
regardless of their real 5 km length, and the "extended centerline" dashes were a
fixed-length line off an arbitrarily-chosen end of each strip — unrelated to where
the craft was, which way it would land, or any distance. There was no route, no
vertical guidance, and no alignment indication anywhere in the game.

Building that meant deciding where navigation *lives*. The tempting shape is the
one the codebase already had: each instrument computes what it needs from the
world. The ND knows about runways, so let it work out the approach; the PFD knows
attitude, so let it work out the deviation. That is how the old ND came to draw
dashes nobody could interpret, and it scales badly in a specific and dangerous
way — two instruments showing the *same* quantity, derived twice, agreeing at
first and drifting apart later. A pilot follows the needle, not the map, and a
disagreement between them is invisible until it matters.

The second decision was frames. Approaches span tens of kilometres on a sphere,
and the geometry is far easier in a plane — but altitude on a tangent plane is
wrong by `d²/2R` (≈ 75 m at 30 km), which is a ruined glideslope.

## Decision

**One authority, many projections.** `thalos_navigation` (pure, no Bevy) owns all
route geometry and the per-frame guidance function; `crate::route` owns selection
and re-plan policy and publishes exactly one `RouteState`; the ND and the PFD are
*projections* of that state and derive no navigation quantity of their own. A
future autoland reads the same commands the flight director already draws.

**Split the frame by axis.** Lateral geometry is planar, in a body-fixed local
tangent frame using a **gnomonic** projection (altitude-independent, an exact
inverse, straight lines are great circles). **Altitude is never a plane
coordinate** — always height above the body reference radius, measured radially.

**The path is bank-limited and actually flyable**: a Dubins `CSC` transition from
the craft's current pose onto a straight final, sized by `v²/(g·tan φ)` with the
*local* gravity.

## Alternatives

- **Let each display compute its own guidance** (the status quo) — rejected as
  above: duplicated derivations of the same quantity drift, and the failure is
  silent. This is the whole reason the ADR exists.
- **Put the geometry in `thalos_control`** — rejected. Control is "how do I make
  the craft do this"; navigation is "what should the craft do". Keeping them
  separate is what lets guidance be a pure function with no craft dynamics in it,
  and lets the same guidance drive a flight director, an autopilot, or nothing at
  all.
- **Put it in `thalos_physics_canonical`** — rejected. Navigation is a
  gameplay/UI-facing policy layer, not physics truth; the pure crates' Bevy-free
  discipline is worth keeping for a reason and this would have blurred what
  `physics_canonical` means.
- **A single 3-D route frame, altitude included as a plane coordinate** — rejected:
  the 75 m-at-30 km sagitta error lands directly on the glideslope, which is the
  one number an approach cannot get wrong.
- **A chord projection instead of gnomonic** — rejected after measuring: local
  coordinates drifted with the craft's own altitude by `d·Δalt/R` (~9 m over a
  25 km final), corrupting route length, distance-to-go, and glideslope deviation
  *together* so the symptom reads as noise rather than a bug.
- **Straight final plus a simple direct intercept, no turn geometry** — rejected
  by the user when scoping: the point of the feature is a path you can actually
  fly, including from behind or offset from the field.
- **Implement Dubins `CCC` words too** — rejected: they win only for very close
  poses, where the answer is a tight double-spiral that is the wrong shape for an
  approach, and `CSC` provably always has a solution (LSL's discriminant is a sum
  of squares), so nothing is lost.
- **Re-plan every frame** — rejected. Beyond the jitter, re-planning once past the
  final approach point asks for a path back to a fix *behind* the craft, which
  Dubins solves with a full turn-around: the plan would fly the aircraft away from
  the runway it was three kilometres from. The plan freezes on final for
  correctness, not performance.

## Consequences

- Adding a display, an autopilot mode, or a new vehicle type does not touch route
  geometry. Adding a *route kind* (arbitrary waypoints, a rover route) extends
  `thalos_navigation` in one place.
- Guidance being pure and stateless means engagement, latching, and hysteresis
  live in the caller. `crate::route` therefore carries the only mode state, and
  its `established` latch is load-bearing (see the re-plan rule above).
- `crate::navigation` now has a misleading name — it is the attitude/SAS pointing
  modes. Renaming it is queued (`BL-20260730T005746Z-rename-attitude-modes`); until
  then both modules carry a pointer to the other.
- The planner is pure geometry with **no terrain awareness**: it will route
  through a mountain. Terrain clearance is explicitly deferred, not overlooked.
- Displays gain a hard requirement to be checkable: the ND's uniform assembly and
  scene projection are pure functions over plain data, which is what makes the
  headless `just nd-preview` harness able to render the *real* pipeline instead of
  a look-alike.
