# ADR-20260731T232619Z-autoflight-strategic-tactical-split: autoflight splits into flight programs and tactical channels

- **Status:** Accepted
- **Date:** 2026-07-31

## Context

Autoflight had one selector: `AutoflightMode = Off | Maneuver | Land | Orbit`,
four mutually exclusive slots. But `Orbit` is not a peer of `Maneuver` — the
ascent program *delegates* to the scheduled-burn executor for its
circularisation nodes. Three separate places had to encode "…but `Orbit` is
also sort of `Maneuver`":

- `autopilot_system` gated on `matches!(mode, Maneuver | Orbit)`;
- `control_bus::realize_control` hand-rolled a two-level priority inside the
  `Orbit` match arm;
- `Autopilot::maneuver_active` and the lock table each repeated the pair.

That aliasing produced three defects that are one defect:

1. **Warp was locked for the whole ascent program**, including the ballistic
   coast. `update_control_locks` read `warp: maneuver || landing || orbiting`,
   and `warp_to_maneuver_system` cancels itself on sight of `locks.warp` — so
   the HUD's WARP button silently did nothing for the several minutes of coast
   when it was most wanted.
2. **The panel could contradict the ship.** `nav_panel` called
   `toggle_mode(Maneuver)` unconditionally; one click during an ascent stole
   the slot and left `monitor_orbit_maneuver_program` early-returning forever
   on a mode check it could no longer satisfy — program stranded in `COAST`,
   widget still annunciating a live program.
3. **Staging was reactive.** The ascent program staged when
   `total_thrust_n <= 1.0` — after the engine had already flamed out — then
   throttled to zero and pointed at local up while it waited for an
   acknowledgement, throwing the vehicle off its gravity turn at every
   staging event. A request issued at warp > 1× was never served, parking the
   program in `Wait` at zero throttle indefinitely.

`thalos_control::arbitrate` already resolved attitude and throttle
independently and already returned `attitude_owner` / `throttle_owner`, with a
doc comment stating the intent: *"UI gating derived from the same decision,
rather than from a parallel lock flag."* `ControlLocks` was the parallel flag,
built anyway.

## Decision

Split autoflight into two layers, mirroring transport-category avionics.

**Strategic — `FlightProgram { None, Ascent, Landing }`.** Owns targets,
sequencing, node installation, and staging commands. Derived each frame from
the programs' own phases by one writer, so "which program is engaged" cannot
drift from "which program is running". Owns the arming policy: `Ascent` arms
the burn executor for its own nodes, `Landing` disarms it (an approach must
never fly a leftover maneuver node), `None` returns it to the pilot.

**Tactical — the channels.** The scheduled-burn executor becomes
producer-agnostic and always available, gated by `BurnArm { Off, Pilot,
Program }` instead of by the mode enum. `resolve_autoflight` is the one pure
function deciding which single source fills the `DemandSource::Autopilot`
slot: program guidance > burn executor > engaged-program idle hold. Its
`AutoflightResolution` also carries what to annunciate.

Three consequences follow structurally rather than by convention:

- **Locks are declared, not derived.** Each source answers `required_locks()`
  for itself and `update_control_locks` unions them. Nothing pattern-matches a
  mode enum, so a coast cannot lock warp — the ascent program declares
  `GUIDANCE_COAST` (throttle + attitude, no warp) and the burn executor
  declares `FULL_AUTHORITY` only while `Engaging`/`Burn`.
- **The panel emits intent.** `AutoflightRequest` messages are consumed by one
  runtime system, which is the single place `ProgramOverridePolicy` is applied.
  Panels can no longer mutate the executor or a program.
- **Staging is commanded.** `StageSequencer` predicts burnout from the active
  stage's remaining propellant and mass flow, commands cutoff, waits out
  tail-off, checks interlocks (thrust decayed, rates below limit), requests
  separation through the existing `StageDemand`, then ignites. Guidance keeps
  steering throughout; only throttle is surrendered. Thrust collapse survives
  as a **backup trigger** that logs `stage_unpredicted`.

`ProgramOverridePolicy { Refuse, ConfirmDisconnect, Immediate }` has one
consulting call site and defaults to `ConfirmDisconnect`. Real avionics is
`Immediate` — refusing a pilot input is considered more dangerous than obeying
a wrong one. A game trades differently: an accidental click during ascent costs
the whole launch, one extra click costs nothing, and unlike `Refuse` it never
leaves a player staring at a dead button with no explanation. The policy exists
so this stays a default change rather than a refactor.

## Alternatives

- **Fix the three symptoms in place** (relax the warp lock condition, add an
  `if orbit.active()` guard to the MNVR button, add a predictive staging
  branch). Rejected: each fix re-encodes the `Orbit`-aliases-`Maneuver` rule in
  a fourth, fifth, and sixth place. The lock table in particular has to know
  about every executor to stay correct, which is exactly why it was wrong.
- **Add an `Ascent`-shaped variant per program to `AutoflightMode`.** Rejected:
  it keeps one slot for two independent questions. Any program that delegates
  to the burn executor reintroduces the aliasing.
- **Push the program/guidance distinction into `DemandSource`.** Rejected: that
  enum's ordering *is* its priority, and program guidance vs. node burn is not
  a priority difference — the resolution between them is strategic. Encoding it
  there would imply an arbiter relationship that does not exist. The
  distinction lives in `AttitudeChannel`, which is annunciation-only.
- **Keep staging reactive and just skip the point-at-up.** Rejected: the thrust
  dropout and the unserved-request hang both come from using the confirmation
  as the trigger. Launch vehicles command cutoff and use thrust decay as an
  interlock precisely because the reverse cannot be made reliable.
- **Predict burnout by integrating the throttle forward.** Rejected: guidance
  re-evaluates throttle every frame from atmosphere and apoapsis error, so a
  one-shot integral diverges silently. A constant-throttle estimate refreshed
  at 60 Hz tracks better and cannot drift unnoticed.

## Consequences

- `AutoflightMode` is gone. `Autopilot` no longer carries a mode;
  `LandAutopilot` gains an explicit `engaged` flag that the enum variant used
  to carry implicitly.
- `OrbitProgram` sheds `stage_request_id`, `stage_settle_frames`, and
  `resume_phase` — all bookkeeping for the reactive staging path — and gains
  `sequence` for annunciation. It no longer takes `Autopilot` at all; ten
  helper signatures lost the parameter.
- New events on `thalos::diagnostic::staging` and
  `thalos::diagnostic::autoflight`. `stage_unpredicted` and `stage_refused` are
  read by `just diag` (`stage_prediction_missed`, `stage_request_refused`); the
  healthy count for both is zero, which is why the threshold is 1.
- The annunciator (`AutoflightAnnunciation`) is written from the *arbitration
  outcome*, so a pilot stick that overrides the autopilot annunciates `MAN`.
  Panels must read it rather than infer engagement from their own button state.
- Not yet done: a full three-column FMA row (program / attitude / throttle /
  next event) — the chip currently folds program and channel into one label.
  Docking and transfer programs will drop into `FlightProgram` without
  touching the seam; that is the point of the split.
