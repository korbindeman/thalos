# INC-20260731T021150Z-orbit-autoflight-pinned-on-pad: ORBIT commanded full thrust while BodyFixed kept the craft landed

- **Date:** 2026-07-31 · **Surface:** `just game launch`, ORBIT ground execution

## Symptom

ORBIT passed preflight, entered `RISE`, and reported throttle `1.0`, TWR `2.54`,
and commanded acceleration `23.0 m/s²`, yet altitude remained exactly
`857.982812 m` for more than five seconds. That combination rules out an
underpowered stage or a launch-guidance throttle limit: thrust was selected,
but canonical translation was still pinned.

## Root cause

The landed `BodyFixed` release gate read `ThrottleState::commanded`, the
persistent **pilot** setpoint. After control-authority unification, ORBIT
publishes a `ControlDemand` and the arbitration winner is
`ThrottleState::selected`; the pilot field correctly remained zero. The regime
resolver therefore never released landed authority even though the engine
received full autoflight throttle.

## Fix

`RegimeInputs` now names and receives the selected, pre-fuel-gate throttle.
Any control-bus winner above the release threshold, pilot or autoflight,
hands translation back to live physics.

## Recurrence signal

`orbit_autoflight_guidance` showing nonzero throttle, TWR greater than one,
and unchanged altitude across several samples while the program is in `RISE`,
`TURN`, or `ASCENT`. The regime input must always come from the arbitration
winner, never from an individual control source.
