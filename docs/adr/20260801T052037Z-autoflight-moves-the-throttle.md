# ADR-20260801T052037Z-autoflight-moves-the-throttle: automatic control moves the canonical throttle

- **Status:** Accepted
- **Date:** 2026-08-01

## Context

Throttle authority was split into three values: a persistent pilot setpoint,
the control-bus winner, and the fuel-gated engine input. This let an automatic
controller temporarily outrank the pilot without moving the pilot's stored
setpoint. The HUD honestly exposed both values, which made one throttle control
look like two conflicting controls whenever LAND or ORBIT changed power.

The split also made release surprising. Automatic control could disappear and
reveal an unrelated, stale pilot value, so completion paths needed a separate
"hold idle until pilot movement" latch to prevent a surge. That latch made the
first takeover gesture disconnect automation without necessarily moving the
throttle.

## Decision

There is one canonical throttle position: `ThrottleState::commanded`. Pilot
input and the winning automatic `ControlDemand` both move it. Control-bus
arbitration still decides who may move it, but its result is committed to the
canonical position rather than stored in a parallel `selected` value.

`ThrottleState::effective` remains distinct because it answers a different
question: how much thrust survives fuel, engine, destruction, and warp gating.
It is not another controller position.

When automation releases the throttle, the canonical position stays where
automation last moved it. A controller that must leave at idle commands zero as
part of its exit: LAND completion and safety aborts do so, and a scheduled burn
has an explicit cutoff edge. Deliberate pilot throttle movement is sampled
while the channel is locked, disconnects the program, and is already the new
canonical position when control returns.

This supersedes the persistent-pilot-setpoint throttle policy described by
ADR-20260730T232443Z's control-authority implementation notes. Its destination,
route, and shared-arbitration decisions remain accepted.

## Alternatives

- **Keep the split and hide one HUD value** — rejected. It conceals the
  conflicting state without fixing stale-value handoff or the extra idle latch.
- **Restore the stored pilot value on disconnect** — rejected. The visible
  throttle would jump to a position no controller had just commanded.
- **Let automatic systems bypass arbitration and write engines directly** —
  rejected. Authority, annunciation, regime release, and engine input would no
  longer share one winner.

## Consequences

- `ThrottleState::selected` and `hold_idle_until_pilot_move` are deleted.
- The HUD's commanded/effective comparison now means control position versus
  constrained engine response, never pilot versus autopilot.
- Regime selection and the fuel gate read the canonical command, which is also
  the arbitration winner.
- New automatic throttle sources get takeover semantics automatically by
  publishing a normal `ControlDemand`.
