# ADR-20260730T232443Z-runway-autoland-is-destination-to-stop: LAND owns the trip from airborne position to parked aircraft

- **Status:** Accepted
- **Date:** 2026-07-30

## Context

The first autoland brief described an APPR controller over the already-built
terminal guidance. It could capture the bank-limited local approach, schedule
configuration, flare, and reach weight-on-wheels. Two boundaries were still
open: whether it could engage only near final or from elsewhere, and whether
touchdown completed the mode.

Those boundaries define the product more than the individual control gains do.
A mode that requires the player to deliver the aircraft to the approach and
hands it back while still rolling is an approach autopilot. The intended feature
is simpler to explain and materially more useful: choose a runway and have the
aircraft arrive there, stopped.

The current terminal planner cannot honestly stretch to global range. Its
runway-centred `RouteFrame` is a gnomonic projection, deliberately precise for
approach distances and singular at the horizon. On the ground, nosewheel
steering currently reads raw pilot yaw and the parking brake is a persistent
latch, so stopping on the runway also requires extending the canonical control
authority rather than writing those resources from an isolated mode.

## Decision

**LAND is destination-level autonomy.** From any normal airborne position on the
current atmospheric body, the player selects any runway on that body and engages
LAND. The aircraft navigates to the runway, captures the terminal approach,
configures, flares, touches down, tracks the runway centreline, brakes to a
complete stop, leaves the parking brake holding, and only then disengages.

Touchdown is a phase transition from flare to rollout, never normal completion.
A stable stopped condition is weight-on-wheels plus surface-relative ground
speed below a named threshold for a dwell.

The route is composite but has one authority:

- a body-fixed spherical great-circle ingress handles arbitrary same-body range;
- a conservative terrain-clearing vertical profile brings the aircraft to an
  arrival fix;
- the existing runway-local bank-limited approach owns terminal capture through
  touchdown.

The specialised ingress is built now because it is required by LAND. General
player-authored waypoint routes, route editing, and persistence remain later and
will reuse it.

Maneuver and LAND modes share one autoflight owner and are mutually exclusive.
Throttle, ground steering, and braking go through canonical demand arbitration.
A recoverable unstable approach enters go-around and retries; it does not
silently disconnect near the ground. Pilot override, explicit cancellation, an
invalid destination, destruction, or physical inability to continue are true
disengagements with a recorded reason.

“Anywhere” does not silently include cross-body atmospheric flight, autonomous
off-runway takeoff, teleportation, or flight after the craft has lost the
physical capability to fly. Those are rejected engagement states, not degraded
versions of LAND.

## Alternatives

- **Terminal APPR only; the player flies the ingress** — rejected by the user.
  It exposes an implementation seam as a player obligation and does not satisfy
  “select a runway and arrive there.”
- **Disengage at weight-on-wheels** — rejected. The aircraft is still moving,
  directional control still matters, and “autoland” has not delivered a parked
  aircraft.
- **Project every start into the runway-local plane** — rejected. The gnomonic
  frame diverges at the horizon and would turn a documented local representation
  into silently wrong global navigation.
- **Wait for the full arbitrary-route system** — rejected. LAND needs exactly
  one generated destination leg; a route editor and user-authored waypoint
  semantics do not unblock it.
- **Write `ParkingBrake` and nosewheel steering directly from LAND** — rejected.
  That would create a second command path beside pilot input and repeat the
  throttle-authority defect the control bus is being extended to remove.

## Consequences

- The item is larger than the original approach-law brief: global ingress and
  rollout control are part of the minimum feature, not follow-ups.
- The first implementation can use the body's published maximum terrain
  elevation plus clearance as a conservative cruise floor. More efficient
  terrain-aware route optimisation remains separate.
- The UI selects a destination runway and shows LAND phases such as `ENRT`,
  `APPR`, `FLARE`, and `ROLLOUT`; `APPR` is no longer the name of the whole mode.
- End-to-end verification must start outside the terminal region and end with a
  stationary, brakes-held aircraft on the selected strip.
- Later general routes reuse the spherical leg, route sequencing, and control
  laws without changing this destination-to-stop contract.
