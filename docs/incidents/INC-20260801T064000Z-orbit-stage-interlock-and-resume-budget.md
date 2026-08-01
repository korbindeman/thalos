# INC-20260801T064000Z-orbit-stage-interlock-and-resume-budget: ORBIT could neither separate nor resume

- **Date:** 2026-08-01 · **Surface:** `just game launch`, ORBIT ascent

## Symptom

ORBIT needed the player to ignite the first stage. Near first-stage depletion
it commanded cutoff but never separated. A brief pilot override at about 33 km
then made every re-arm report insufficient delta-v despite a healthy upper
stage.

Session `26360-1785565230583` is the discriminator: it records
`stage_cutoff`, then no `stage_commanded` / `stage_separated`; after
`pilot_override`, six consecutive aborts are `insufficient_delta_v`.

## Root cause

Three authority boundaries each used the wrong quantity:

1. Preflight checked live thrust before requesting the cold stack's first
   `StageDemand`, so automatic launch still depended on manual staging.
2. The cutoff interlock received `ActivePropulsion::total_thrust_n`, the
   engines' full-throttle **rating**, as if it were produced thrust. Closing
   the throttle could never bring that number below the settled threshold, so
   separation waited forever.
3. Re-arm classified the craft as an atmospheric ascent correctly, but compared
   remaining staged delta-v with the original pad-to-orbit estimate. It charged
   already-earned altitude and velocity again.

## Fix

The shared sequencer now requests and acknowledges first-stage activation,
and its interlock consumes `rated thrust × effective throttle`. Resume retains
the original launch frame and target, restores the phase from height already
gained, and gates remaining fuel against a live vis-viva estimate plus only the
remaining gravity/drag reserve.

## Recurrence signal

- A healthy cold launch emits `stage_ignition_commanded` → `stage_ignited`.
- Every `stage_cutoff` with another stage available must be followed by
  `stage_commanded` → `stage_separated`; rated thrust remaining nonzero is not
  a reason to wait when effective throttle is zero.
- If `orbit_delta_v_refused` appears, its `required_delta_v_m_s` must reflect
  the live aligned speed and altitude. Re-arm must not repeat the pad value.
