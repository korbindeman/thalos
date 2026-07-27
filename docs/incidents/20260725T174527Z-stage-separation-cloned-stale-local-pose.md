# INC-20260725T174527Z-stage-separation-cloned-stale-local-pose: the booster jumped one inertial frame out of view

- **Date:** 2026-07-25 · **Surface:** stage separation while the active craft uses local rigid-body translation

## Symptom

The separation log reported a real 72.4 t vessel with three parts and a
0.269 m/s relative kick, but the booster disappeared from the chase view.

A five-second structural trace falsified the rendering hypotheses: the root,
all three parts, and all three mesh children remained alive; parentage,
inherited visibility, low-precision-root tags, and ship/shadow render layers
were correct. The decisive signal was position: the detached root was already
708 m from the active craft on its first rendered frame.

## Root cause

Staging ran in `SimStage::Physics` without an order relative to local-physics
readback. `Simulation::step` advances the shared epoch but intentionally leaves
`LocalRigidBody` translation unchanged. Avian readback later installs the
active craft's current-epoch inertial position.

Separation could run in between: it cloned the active craft's previous-frame
position into a new `OnRails` vessel, while `create_vessel` stamped that state
with the already-advanced epoch. At the craft's heliocentric velocity, one
render frame is hundreds of metres. The booster rendered correctly—far outside
the chase camera.

## Fix

`activate_stage` is explicitly ordered after `readback_local_craft`. The graph
cut now clones the active vessel only after its current-epoch inertial pose is
installed. The equal/opposite impulse and rendering hierarchy are unchanged.

## Recurrence signal

`artifacts/diagnostics/stage_separation.jsonl` reports
`distance_from_active_m`. A healthy first sample is approximately the physical
gap opened by the decoupler; a frame-one distance of hundreds of metres means
the separation transaction crossed an authority/readback boundary again.
