# INC-20260731T021150Z-orbit-autoflight-pinned-on-pad: ORBIT commanded full thrust while BodyFixed kept the craft landed

- **Date:** 2026-07-31 · **Surface:** `just game launch`, ORBIT ground execution

## Symptom

ORBIT passed preflight, entered `RISE`, and reported throttle `1.0`, TWR `2.54`,
and commanded acceleration `23.0 m/s²`, yet altitude remained exactly
`857.982812 m` for more than five seconds. That combination rules out an
underpowered stage or a launch-guidance throttle limit: thrust was selected,
but canonical translation was still pinned.

## Root cause

At the time, the landed `BodyFixed` release gate read
`ThrottleState::commanded`, then the persistent **pilot** setpoint. ORBIT
published a `ControlDemand` whose winner lived separately in
`ThrottleState::selected`; the pilot field correctly remained zero. The regime
resolver therefore never released landed authority even though the engine
received full autoflight throttle.

## Fix

The immediate fix made `RegimeInputs` consume the selected, pre-fuel-gate
throttle. ADR-20260801T052037Z later removed that split: the control-bus winner
now moves canonical `commanded`, and the regime consumes that same value. Any
winner above the release threshold, pilot or autoflight, hands translation back
to live physics.

## Recurrence signal

`orbit_autoflight_guidance` showing nonzero throttle, TWR greater than one,
and unchanged altitude across several samples while the program is in `RISE`,
`TURN`, or `ASCENT`. The regime input must always come from the arbitration
winner, never from an individual control source.
