# Control — the fly-by-wire layer

Thalos routes **every** ship-control command through one layer:
`thalos_control`. Pilot input does not drive effectors directly, and no
autopilot drives effectors directly either. Both speak a single command
vocabulary, are arbitrated by priority, run through one controller, and
are allocated across whatever effectors the craft has. This is the
"fly-by-wire" model: raw input is a *request*, and the layer decides what
that request does.

This replaced three disconnected paths that nothing coordinated:

1. **Reaction-wheel / SAS torque** — the old
   `navigation::compute_attitude_control` produced a `ControlInput`
   carrying a torque command plus an `sas_enabled` flag. That flag drove a
   per-frame **deadbeat damper** (`−I·ω/dt`, clamped) in the game's
   `compute_angular_acceleration`.
2. **Aero control surfaces** — the aero system read the raw stick
   (`GameInputIntent.attitude`) straight into `evaluate_aero`, bypassing
   the autopilot, nav modes, and locks entirely.
3. **Throttle** — a separate setpoint path.

The deadbeat damper was the SAS micro-jitter: it tried to annihilate *all*
angular velocity every frame and limit-cycled against the continuous aero
moments, because the reaction wheels and the control surfaces were two
uncoordinated effectors chasing different inputs. The fly-by-wire layer
fixes this structurally — one controller commands one torque that both
effectors execute together.

## Pipeline

```
sources → demands → [arbitrate] → [AttitudeController] → torque → [allocate] → effectors
```

Each frame, [`crate::control_bus::realize_control`] (game-side glue) runs
the whole pipeline:

1. **Collect demands.** Every source emits a `ControlDemand` tagged with
   its `DemandSource`:
   - **Pilot** (highest) — stick deflection → `Rate(cmd)`; hands-off →
     `Free`. Suppressed to `Free` while `ControlLocks::attitude` is set, so
     the player can't fight an engaged autopilot (KSP behaviour).
   - **Autopilot** — `PointNose(burn_dir)` while engaging/burning, else
     `Free`.
   - **NavMode** — `Stability` → `Hold`; a directional mode (prograde,
     target, maneuver, …) → `PointNose(dir)`; unresolved → `Free`.
   - **Sas** (lowest) — the free-flight `T`-toggle hold → `Hold` when on.
2. **Arbitrate.** `thalos_control::arbitrate` picks the highest-priority
   *active* (non-`Free`) attitude demand. Throttle is arbitrated
   independently.
3. **Control.** The single stateful `AttitudeController` turns the
   resolved demand into a normalized body-frame torque in `[-1, 1]`:
   - `Hold` → **full-quaternion PD** to a captured target orientation
     (roll included). Critically damped — it settles, it does not chatter.
   - `PointNose(dir)` → nose-direction PD constraining `+Y` body, purely
     damping roll (roll is free during a burn).
   - `Rate(cmd)` → pass-through (deflected stick = direct command).
   - `Free` → zero.
4. **Allocate.** `thalos_control::allocate` hands the *same* torque to
   every effector: reaction wheels (`ControlInput::torque_command`,
   consumed by `apply_local_forces`) and aero control surfaces
   (`RealizedControl::aero`, read by the aero force system). Aero responds
   through dynamic pressure — nothing in vacuum or on the ground — and the
   wheels cover the rest; the controller closes the loop on the summed
   result via ω feedback.

   > **Known issue — over-actuation in atmosphere (TODO: dynamic-pressure
   > blend).** Feeding *both* effectors the full command is correct in
   > vacuum but over-drives in thick air: at cruise the aero control
   > surfaces alone have large authority, so adding the full reaction-wheel
   > command on top makes the SAS `Hold` loop over-actuate. Observed via BRP
   > on the Meridian (`CraftStateMirror.angular_velocity_rad_s`): under SAS,
   > **yaw** picks up a ~0.05 rad/s oscillation where aero alone holds it
   > near zero. The fix is to make `allocate` dynamic-pressure aware — lean
   > on aero surfaces in atmosphere and reaction wheels in vacuum, rather
   > than commanding both fully — so the effective loop gain stays at what
   > the PD assumes. `realize_control` has the body atmosphere + airspeed to
   > compute `q̄` and pass an authority split into `allocate`.

## SAS feel: centered stick holds attitude

The SAS/`Hold` controller captures the current orientation the first frame
it engages and PD-holds it. Deflecting the stick emits `Rate`, which clears
the captured target and applies the command directly; releasing the stick
returns to `Hold`, which recaptures the *new* orientation. So "centered
stick = hold current attitude" — push to rotate, release to hold where you
let go. Directional nav modes (`PointNose`) take over from `Hold` when
selected, and the pilot outranks both while the stick is touched.

## Body-axis convention

`X` = pitch, `Y` = nose (roll axis), `Z` = yaw. This matches
`GameInputIntent.attitude` (`x`/`y`/`z` = pitch/roll/yaw), the aero
`ControlInputs { pitch, roll, yaw }`, and `ControlInput::torque_command`.
`thalos_control::SETTLE_TIME_S` and `NOSE_BODY` are the shared constants;
`navigation::AUTOPILOT_SETTLE_S` / `SHIP_NOSE_BODY` re-export them so the
autopilot's lead-time sizing can't drift from the controller's gains.

## Crate layout

- **`thalos_control`** (pure Rust, no Bevy; depends one-way on
  `thalos_physics_canonical` for `ShipParameters` / `AttitudeState` /
  `ControlInput` / `aero::ControlInputs`):
  - `demand` — `AttitudeDemand`, `ControlDemand`, `DemandSource` priority.
  - `arbiter` — `arbitrate` + `Arbitration`.
  - `attitude` — `AttitudeController` (the control laws).
  - `allocator` — `allocate` + `Allocation`.
- **`thalos_game::control_bus`** — the Bevy glue: `ControlBusPlugin`, the
  `SasState` / `AttitudeControllerState` / `RealizedControl` resources, and
  the `realize_control` system, scheduled in `SimStage::Physics` after the
  demand producers (pilot throttle, warp, autopilot) and before
  `advance_simulation` and the effector systems.

## Scope and extension points

This pass covers **attitude** for ships — the fragmented, jittery surface.
The remaining controls are designed-in extension points wired the same way
later:

- **Throttle setpoint.** Throttle currently stays on its existing path:
  `ThrottleState::commanded` is the player's persistent setpoint, the
  autopilot overrides it directly during a burn, and `ControlLocks::throttle`
  gates the player. The arbiter already arbitrates a throttle demand
  generically; folding the *setpoint* through the bus (without losing the
  persistent-setpoint and fuel-gate semantics) is the next step.
- **Warp.** Warp arbitration (player / autopilot / auto-warp) is still in
  `bridge`/`warp_to_maneuver`; it can become a `DemandSource`.
- **EVA.** The on-foot controller owns its own kinematics and has no
  reaction-wheel torque; a jetpack would slot in as a new source/effector.
- **RCS / engine gimbal.** New effectors are new branches in
  `allocate` (and a dynamic-pressure blend between aero and wheels lives
  there too); new command authority is a new `DemandSource`.

See `docs/aerodynamics.md` for the aero force model the control surfaces
drive, and `docs/simulation.md` for the authority/integration context.
