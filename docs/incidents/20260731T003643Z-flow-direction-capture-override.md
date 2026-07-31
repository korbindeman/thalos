# INC-20260731T003643Z — The flow probe hid a live direction inversion in vacuum

## Symptom

A rising rocket wore a vapour bell that widened ahead of its nose. The
deterministic `vapor-cone` capture appeared correctly oriented, but showed the
effect on a craft visibly parked outside the atmosphere.

## Mechanism

`air_relative = ship_velocity - co_rotating_air_velocity` points upstream, toward
the air the craft is entering. `resolve_flow` negated it while publishing a field
documented as “the direction the freestream arrives from.” Both attached effects
then interpreted that field according to the documentation, so live flow was
reversed. The capture supplied its own local direction in the documented
convention and therefore hid the defect.

The same capture supplied artificial density while the craft remained in orbit.
`in_atmosphere` was derived from the overridden density, allowing the authoring
surface to manufacture an atmosphere and light both condensation and reentry
effects in vacuum.

## Recurrence tell

If a flow effect is correct in its headless probe but reversed in live flight,
compare the probe's `flow_from_local` override with the producer's
`air_relative` sign. If any atmospheric effect appears above the Kármán line,
check whether `in_atmosphere` depends on an overridden value.

## Fix

Publish `air_relative` itself as the upstream arrival direction. Derive
`in_atmosphere` only from the real atmosphere sample, before applying authoring
overrides. The vapour probe now boots atmospheric cruise (with an authored
nose-side direction because cold terrain waits erase the initial cruise
velocity); the reentry probe boots the atmospheric landing approach. The live
direction convention is pinned independently at the producer.
