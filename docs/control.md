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
   - **Sas** (lowest) — the free-flight hold (`T` key / HUD SAS button) →
     `Hold` when on. **Defaults on** for every craft (and survives
     destruction/respawn): a spaceship spawns holding attitude, a plane
     spawns flying FBW; switching SAS off is the deliberate act.
2. **Arbitrate.** `thalos_control::arbitrate` picks the highest-priority
   *active* (non-`Free`) attitude demand. Throttle is arbitrated
   independently.
3. **Control.** The single stateful `AttitudeController` turns the
   resolved demand into a normalized body-frame torque in `[-1, 1]`:
   - `Hold` → **regime-dispatched**. Spaceship (no `FlightState`, see
     *Flight assist* below): **full-quaternion PD** to a captured target
     orientation (roll included). Critically damped — it settles, it does
     not chatter. Assisted plane (`FlightState` supplied): the
     **fly-by-wire pitch/bank hold** instead.
   - `PointNose(dir)` → nose-direction PD constraining `+Y` body, purely
     damping roll (roll is free during a burn).
   - `Rate(cmd)` → pass-through (deflected stick = direct command); on an
     assisted plane the pitch axis additionally rides the auto-trim and is
     clamped by the AoA envelope.
   - `Free` → zero.

   > **Damping is body-frame.** `AttitudeState::angular_velocity` is already
   > body-frame (canonical convention), so the PD damps it directly. An earlier
   > cut rotated it by `orientation.inverse()` first — a spurious double
   > transform that misaimed the damping torque by the ship's orientation, so
   > SAS failed to settle and pointing modes spun up. The `Rate` (pilot)
   > path is unaffected; only the PD laws read ω.
   >
   > **Eased-in slews, no overshoot.** The PD laws (`Hold`, `PointNose`) carry
   > a **deceleration-limited rate cap**. The bare PD is critically damped only
   > in its linear region; because `kp` is sized for a small-angle settle, the
   > command saturates against the available authority at a small error, so a
   > large target change used to run open-loop at full torque, build up rate,
   > and overshoot (snap-and-bounce). `pd_to_normalized_torque` now caps the
   > PD's implied desired rate `(ω_n/2)·e` at the stopping rate `√(2·α·|e|)`
   > (`α = authority/I`, with a 0.9 decel margin): far from target the cap binds
   > (near time-optimal — eases in), near it the law *is* the original PD (no
   > sqrt chatter). Small slews and strong-wheel craft are unchanged.
4. **Allocate.** `thalos_control::allocate` hands the *same* torque to
   every effector: reaction wheels (`ControlInput::torque_command`,
   consumed by `apply_local_forces`) and aero control surfaces
   (`RealizedControl::aero`, read by the aero force system). Aero responds
   through dynamic pressure — nothing in vacuum or on the ground — and the
   wheels cover the rest; the controller closes the loop on the summed
   result via ω feedback.

   > **Over-actuation in atmosphere — resolved.** Feeding both effectors the
   > full command used to over-drive in thick air (a ~0.05 rad/s SAS yaw
   > oscillation on the Meridian at cruise). The fix lives in the
   > *controller*, not the allocator: `pd_to_normalized_torque` normalizes
   > the PD output by the **total** available authority (`max_torque` + the
   > aero control authority at the current dynamic pressure, supplied by
   > `control_bus::player_aero_authority`), so driving both effectors at
   > that fraction realizes exactly the PD's intended torque.
   >
   > Aircraft command pods author `reaction_wheel_torque: 0`
   > (`assets/parts.ron`) — a plane's only attitude effector is its control
   > surfaces, so its authority scales with dynamic pressure like a real
   > aircraft's, and there is no free torque on the runway. The allocator
   > needs no special case: the wheels term simply contributes nothing.

## SAS feel: centered stick holds attitude

The SAS/`Hold` controller captures the current orientation the first frame
it engages and PD-holds it. Deflecting the stick emits `Rate`, which clears
the captured target and applies the command directly; releasing the stick
returns to `Hold`, which recaptures the *new* orientation. So "centered
stick = hold current attitude" — push to rotate, release to hold where you
let go. Directional nav modes (`PointNose`) take over from `Hold` when
selected, and the pilot outranks both while the stick is touched.

## Flight assist: SAS is fly-by-wire on a plane

SAS means different things per regime. On a spaceship it is the KSP-style
hold above. On a **winged craft flying in atmosphere** the same SAS toggle
engages a fly-by-wire law instead (`thalos_control::flight` + the plane
branch of the controller). The regime signal is the `Option<FlightState>`
that `control_bus` passes into `AttitudeController::update`: it is `Some`
only when SAS is *armed* (`SasState`, which **defaults on**, toggled by the
`T` key and the HUD SAS button; the legacy Stability nav mode also arms
it), the craft's aero config has lift (`lift_slope > 0`), the local
density is nonzero, and airspeed is above a 15 m/s floor. Spaceships, vacuum, taxi
speeds, and SAS-off never construct one, so those paths are byte-for-byte
the old behaviour — a plane that climbs out of the atmosphere hands back
to the quaternion hold automatically.

The fly-by-wire law, all in the body frame (`FlightState` carries local-up
and the air-relative velocity rotated into it):

- **Centered stick holds pitch attitude + bank angle, heading free.** A
  quaternion hold is wrong for a plane: holding heading in a banked turn
  fights the natural turn with skidding yaw. The pitch/bank hold gives
  coordinated stick-free turns. Captured bank inside 5° snaps to
  wings-level; beyond ±60° it clamps, so releasing the stick in a steep
  roll recovers to a sustainable turn instead of holding a spiral. The
  per-axis PDs reuse the same deceleration-limited gain family
  (`slew_axis`) as the quaternion hold, and a mild sideslip damper on yaw
  adds turn coordination on top of the airframe's weathervane.
- **Auto-trim.** A pure PD holding attitude against the wing's restoring
  moment parks at a steady-state sag (~2° on the Meridian). A slow pitch
  integrator (`TRIM_RATE_PER_S`, clamped to `TRIM_AUTHORITY` = 0.4, frozen
  while the command saturates or protection clamps — that's the
  anti-windup) nulls it, so hands-off flight holds the attitude exactly.
  The trim survives stick deflections (the deflected stick *rides* it, so
  centered stick ≈ trimmed flight rather than zero surface) and resets
  when SAS disengages or a pointing mode takes attitude.
- **Stall (AoA envelope) protection.** Every realized pitch command — the
  hold law *and* the pilot's stick while SAS is on — is clamped by
  `pitch_command_envelope`: above 80% of the config's `stall_alpha` the
  available nose-up command fades linearly, reaching zero *at* the stall
  angle (full back stick buys exactly stall AoA, never past), and beyond
  it a firm nose-down override ramps to full push within one
  protection-band width. Mirrored for negative AoA. The envelope is
  evaluated at the **predictive** AoA `α + ω_pitch · 0.5 s`
  (`ALPHA_PROTECT_LEAD_S`): a static clamp only reacts once α is in the
  band, and flight test showed a full pull at high dynamic pressure
  building enough pitch rate to bust ~10° past the stall before it bit —
  leading with the rate fades authority while the pull is still building
  (at zero rate it is exactly the static law; sustained full pull settles
  ~13° vs the 15° stall). It also out-votes the
  hold itself — a hold whose speed bleeds off toward the stall gets pushed
  nose-down rather than held into departure. SAS off is fully manual
  (KSP behaviour): aerobatics and spins stay possible, deliberately.

The controller publishes an `AssistStatus { fbw_active, protection_active }`
each frame (`RealizedControl::assist`); the HUD's SAS button reads it,
relabelling to **FBW** while the plane law flies and warn-tinting while
stall protection is actively clamping.

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
  - `attitude` — `AttitudeController` (the control laws, including the
    plane fly-by-wire hold + auto-trim).
  - `flight` — `FlightState` (body-frame α/β/pitch/bank), the AoA
    envelope (`pitch_command_envelope`), `AssistStatus`.
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
- **Engine gimbal — wired.** Thrust vectoring is a third attitude effector:
  a rocket's gimballed engines contribute `gimbal_torque_full · throttle` of
  pitch/yaw authority, folded into the controller's `effector_authority`
  (alongside the aero surfaces) and realized in
  `local_physics::forces::compute_angular_acceleration`. It is what makes a
  launch-vehicle ascent steerable — see `docs/aerodynamics.md` *Thrust
  vectoring*. Roll stays on the reaction wheels.
- **RCS.** A new translation/attitude effector: a new branch in `allocate`
  (and a dynamic-pressure blend between aero and wheels lives there too);
  new command authority is a new `DemandSource`.

See `docs/aerodynamics.md` for the aero force model the control surfaces
drive, and `docs/simulation.md` for the authority/integration context.
