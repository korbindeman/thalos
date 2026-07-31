# Runway destination autopilot (LAND / autoland) — implementation brief

**Status: not built.** This is a brief for the agent that implements it, not a
description of existing code.

Backlog: `BL-20260730T005746Z-approach-autopilot`. Spec for the guidance it
flies: [gameplay/navigation.md](../gameplay/navigation.md). Control-layer
context: [control.md](control.md). Product boundary:
ADR-20260730T232443Z-runway-autoland-is-destination-to-stop.

## Product contract

**From any normal airborne position on the current atmospheric body, select any
runway on that body and engage LAND. The aircraft navigates to it, captures the
approach, configures, lands, tracks the runway centreline during rollout, brakes
to a complete stop, leaves the parking brake holding, and only then disengages.**

This is destination-level autonomy, not a flight-director follower that stops at
touchdown. `APPR` is one phase inside the LAND mode. Later arbitrary waypoint
routes reuse the enroute leg and the same control laws; they are not a
prerequisite for autoland and LAND does not invent a parallel route authority.

“Anywhere” means an airborne, powered, controllable aircraft on the same body as
the selected runway. Cross-body transfer, autonomous off-runway takeoff, and
continuing after loss of the physical ability to fly are not promises hidden in
this feature. An impossible engagement is refused with a reason; it never
pretends to be active.

**Verification is a play session, deliberately.** There is no headless
closed-loop flight harness in this repo, and building one is *not* a
prerequisite: the user flies a deterministic runway-destination scenario and
reports back. So the job is to land it implemented, unit-tested where the logic is pure, and
**instrumented well enough that "it flew badly" can be diagnosed from
`runtime.jsonl` without a second play session**. That last part is not optional
— it is what makes the feedback loop cost one session instead of five.

---

## What already exists (do not rebuild)

| Thing | Where | What it gives you |
|---|---|---|
| Guidance outputs | `thalos_navigation::Guidance`, published in `route::RouteState` | `bank_command_rad`, `vertical_speed_command_m_s`, `target_speed_m_s`, `next_gate`, `dtg_m`, `phase`, `established`, and the deviations. Already clamped to the same bank limit the path was planned with. |
| Plane fly-by-wire law | `thalos_control::attitude::AttitudeController::plane_hold` | Flies a `PlaneHoldTarget { pitch_rad, bank_rad }` with a critically-damped PD, coordinating yaw against sideslip. |
| Stall protection | `thalos_control::flight::pitch_command_envelope` | Every realized pitch command already passes the AoA envelope. **The autopilot inherits this for free** — it cannot command a stall — provided it goes through the plane law rather than around it. |
| Priority arbitration | `thalos_control::{DemandSource, arbitrate}` | `Sas < NavMode < Autopilot < Pilot`. Declaration order *is* priority. |
| Input lockout | `controls::ControlLocks` | The established way to gate player input while a programmatic source owns a channel. |
| Configuration state | `local_physics::gear::GearState`, `flight_config::FlightConfig` | `gear.down`, `flap_setting` (0..=`FLAP_DETENTS`), plus actuator positions that chase the lever at a real rate. |
| Height over the runway | `guidance.altitude_m − plan.frame.origin_altitude_m` | The route frame is anchored at the threshold, so **this is already the flare input** — no radar-altimeter plumbing needed. |
| Terrain ceiling | `TerrainRegistry::max_elevation_m(body)` | A conservative global safe-cruise floor for the first destination planner. It avoids pretending the local approach planner is terrain-aware. |
| Ground contact and braking | `WeightOnWheels`, `ParkingBrake`, landing-gear friction | Bounce-resistant touchdown state, full friction-circle braking, spoiler deployment, and a true parked hold already exist. |

What does **not** exist yet:

- a spherical enroute leg from arbitrary same-body positions to the runway's
  local arrival region;
- a shared maneuver/LAND mode owner;
- programmatic nosewheel steering or wheel-brake demand through the control bus;
- the LAND state machine and its recovery policy.

---

## Decisions already made

1. **LAND owns the whole trip to a stopped aircraft** (user, 2026-07-31).
   Touchdown is a phase transition, not completion. Weight-on-wheels transitions
   to rollout; completion requires a stable near-zero surface-relative ground
   speed with the parking brake engaged.
2. **Throttle gets folded into the control bus** (user, 2026-07-30) rather than
   the autopilot writing `ThrottleState` directly the way the maneuver autopilot
   does. This is work item A below. It is the "one canonical path per operation"
   rule, and it removes a real defect class — see the note in A.
3. **Guidance stays pure.** `compute_guidance` has no memory, no latches, no
   engagement state (ADR-20260730T005746Z). All autopilot mode state lives in
   the game layer.
4. **The plan freezes once established on final.** Do not re-plan mid-approach
   and do not add a "re-plan because the autopilot drifted" path — re-planning
   past the final approach point asks for a route back to a fix *behind* the
   craft, which the planner answers with a full turn-around. See
   navigation.md § Re-plan policy.
5. **Long-range ingress is a specialised destination route, not the future
   general-route editor.** LAND may create the one automatic great-circle leg it
   needs. Arbitrary player-authored waypoints, fly-by/fly-over semantics, and
   general route sequencing remain later work.

---

## A. Fold throttle into the control bus

`ControlDemand` already carries `throttle: Option<f64>` and `arbitrate` already
resolves it by priority — the plumbing exists and is unused. Today
`ThrottleState::commanded` is written directly by the player's input handler and
*overwritten* by the maneuver autopilot during a burn.

**Why this is worth doing first:** because the autopilot currently mutates the
player's own setpoint, disengaging leaves the throttle wherever the autopilot
left it rather than where the pilot set it. With the bus, `commanded` stays the
pilot's persistent setpoint — storage, not a command path — and a programmatic
source merely *outranks* it for as long as it is engaged.

Shape:

- `ThrottleState::commanded` remains the player's persistent setpoint.
- Each source emits a throttle demand; `realize_control` writes the arbitrated
  winner into the sim's control input. The fuel gate
  (`gate_throttle_on_fuel_availability` → `effective`) stays *after*
  arbitration, unchanged.
- The maneuver autopilot stops writing `ThrottleState` and emits a demand
  instead.

> **Trap that will silently break the feature.** If the pilot source emits
> `Some(commanded)` every frame, it wins every arbitration — `Pilot` is the
> highest priority — and the autothrottle's demand can never take effect. The
> pilot source must emit `None` while an autothrottle owns the channel, and
> emit `Some` only when the player actually *moves* the throttle that frame —
> which is also the signal that should disengage the autothrottle (real
> autothrottle behaviour, and it means the pilot never has to find a button to
> take manual control). Write the test for this before the law.

Today `handle_throttle_input` returns before sampling while
`ControlLocks::throttle` is set. That must change: sampling a deliberate pilot
movement and realizing a pilot command are separate operations. Sample always,
retain the HOTAS deadband/anchor behaviour, publish a one-frame takeover edge,
and let that edge disengage LAND before the pilot demand enters arbitration.

Rollout extends the same canonical-control rule to the ground:

- add programmatic ground-steer and wheel-brake demands (either fields on
  `ControlDemand` or a sibling demand resolved by the same source priority);
- stop `apply_landing_gear_forces` from reading raw yaw intent as its only
  steering authority;
- keep `ParkingBrake` as the persistent latch and stopped-aircraft hold, while
  the resolved brake demand is what owns braking during rollout;
- pilot yaw/brake movement is an override edge, not a second writer fighting the
  LAND controller.

Full rollout braking is sufficient for the current tyre model: it is already
clamped to the friction circle. Do not invent ABS or wheel-slip state merely for
this slice.

---

## B. A plane target and one autoflight mode owner

The current vocabulary is `AttitudeDemand::{Free, Hold, PointNose, Rate}`. None
of these expresses "hold this bank angle and this pitch attitude", which is
exactly what the guidance publishes and exactly what `plane_hold` already flies —
except that `plane_hold` *captures* its `PlaneHoldTarget` from the current
attitude (`get_or_insert_with`) instead of accepting one.

So: add a variant carrying a `PlaneHoldTarget` (name it for what it is — this is
a flight-path hold, not a quaternion hold), and let a supplied target override
the captured one. `PointNose` is **not** a substitute: it has no roll authority,
and an aircraft turns by banking.

Keep the change inside `thalos_control` — the controller, the demand enum, and
the arbiter are all in the pure crate, so this is unit-testable without Bevy.

The game also needs one shared autoflight owner. The existing `Autopilot`
resource is the maneuver-burn executor, `ControlLocks` derives only from it, and
`control_bus` has exactly one `DemandSource::Autopilot` slot. A second
independent LAND resource must not become another equal-priority writer.

Introduce one selected mode (`Off | Maneuver | Land`) that:

- makes MNVR and LAND mutually exclusive;
- publishes the one autopilot demand consumed by `control_bus`;
- derives the owned control surfaces and UI annunciation;
- delegates internal execution to the existing maneuver state or the new LAND
  state without merging their unrelated laws.

An explicit plane target must construct `FlightState` and enter the plane law
even when the player's SAS toggle is off. Today `control_bus` only constructs
that state while SAS/Stability is armed; keying it only to that toggle would
silently make LAND bypass the plane hold and its AoA envelope.

---

## C. Plan from anywhere to the terminal approach

The existing `plan_approach` is precise local terminal geometry, not a global
route. Its runway-centred gnomonic projection diverges at the horizon and is
explicitly documented as an approach frame. Do not feed a far-side craft into
it and call that “anywhere.”

LAND builds one composite destination plan:

1. a spherical, body-fixed great-circle ingress from the aircraft's current
   ground direction to an arrival fix behind the selected runway's final
   approach point;
2. a conservative enroute vertical profile: climb/hold above
   `TerrainRegistry::max_elevation_m(body) + clearance`, then descend early
   enough to meet the terminal capture altitude and speed;
3. the existing bank-limited local `ApproachPlan` from the arrival region
   through final and touchdown.

The enroute leg belongs in `thalos_navigation` and publishes guidance through
the same `RouteState` authority as the terminal plan. Displays and LAND consume
that state; neither re-derives a bearing. At the exact antipode, choose a
deterministic great-circle normal rather than returning NaN. Rebuild the
enroute plan while distant if needed, but freeze the terminal plan under the
existing established-on-final rule.

This is deliberately the smallest reusable seed for later general routes. It
adds a spherical leg and phase sequencing, but no route editor, arbitrary
waypoint list, persistence, or user-authored constraints.

---

## D. The LAND state machine

New module `crates/runtime/game/src/route_autopilot.rs`. One resource holds the
mode state and one system publishes the selected autopilot demand.

Suggested explicit phases:

```text
Off → Enroute → TerminalCapture → Final → Flare → Rollout → Stopped → Off
                    ↑                         ↘ GoAround ────────┘
```

`Stopped` is a completion edge used to leave the parking brake engaged and emit
the completion diagnostic before returning to `Off`; it need not persist for
more than one frame.

### The three laws

All airborne inputs come from `RouteState.guidance`; none of them re-derives
navigation.

**Lateral.** `bank_rad = guidance.bank_command_rad` — already clamped to the
planned bank limit. Straight through.

**Vertical.** Convert the commanded vertical speed to a pitch attitude:

```text
γ_cmd  = asin(clamp(vertical_speed_command_m_s / V, -1, 1))     // flight path angle
pitch  = γ_cmd + α                                              // α from FlightState::alpha()
```

`V` is airspeed; `α` is available from the `FlightState` the control bus already
builds. Clamp the result to a sane band (roughly −10°…+15°) so a bad guidance
frame cannot command a bunt. The AoA envelope then clamps it again downstream,
which is the safety net, not the primary limit.

**Speed.** A PI controller on `target_speed_m_s`, output to the throttle demand
from work item A. Integral term needs anti-windup (approach spends a long time
at idle). Feed-forward from the profile's descent gradient helps but is optional.

### Configuration scheduling

`guidance.next_gate` already carries `dtg_m` and a label (`FLAPS` / `GEAR` /
`VAPP`). Drive `FlightConfig::flap_setting` and `GearState::down` off gate
crossings. Both actuators travel at a real rate, so command them *at* the gate
rather than waiting to need them.

### Flare

Below a flare height (start with ~15 m over the runway, from
`guidance.altitude_m − plan.frame.origin_altitude_m`):

- blend the glideslope pitch target toward a nose-up flare attitude,
- retard the throttle toward idle,
- hold wings level (`bank_rad → 0`),
- transition to rollout only after weight-on-wheels is stable long enough to
  reject a bounce.

The flare is the part most likely to need tuning from play feedback. Keep its
constants named and together at the top of the module so a tuning round is a
one-line diff per constant, and log them (below) so the user's "it slammed
down" can be matched against what the law actually commanded.

### Rollout and completion

Touchdown does **not** disengage LAND.

- Command idle throttle, full wheel braking, spoilers through the existing
  brake-driven configuration, and the landing flap/gear state.
- Track the selected runway centreline with a bounded yaw demand. At high speed
  the rudder carries it; as speed falls, the existing steering fade naturally
  hands authority to the nosewheel.
- Do not chase the distant threshold after touchdown. Ground cross-track and
  heading error are measured against the finite selected strip centreline in
  its body-fixed runway frame.
- Declare completion only when weight-on-wheels is stable and
  surface-relative ground speed remains below a named stop threshold (start
  with `0.5 m/s`) for a short dwell (start with `1 s`).
- On completion, set the persistent parking brake, leave throttle at the
  pilot's stored setpoint but with zero realized command until disengagement,
  emit `land_completed`, and then disengage. Gear remains down and flaps remain
  in landing configuration for the pilot to clean up.

The stored pilot throttle must not surge in on the completion frame. The handoff
is stopped, brakes-held, then ownership releases; if a nonzero physical HOTAS
lever later moves deliberately, the normal takeover path applies.

### Recovery and go-around

LAND's promise is arrival, so a recoverable unstable approach does not silently
disconnect near the ground. Excessive localizer/glideslope deviation, an
unstable sink rate, loss of touchdown capture, or a bounce transitions to
`GoAround`: climb power, a protected positive pitch target, wings level until
clear, gear/flap cleanup on a conservative schedule, then rebuild the ingress or
terminal capture and try again.

Pilot stick/throttle/brake input, explicit LAND cancellation, loss of the
selected runway, destruction, or loss of the physical ability to fly are true
disengagements. A bounded retry count may end in a visible `Unable` state with a
reason; it must not loop invisibly forever.

### Engagement and disengagement

Engage: select any runway on the current body, then press LAND. Selection may be
through the existing runway selector even when the strip is off the current ND
range; the UI must identify the destination unambiguously by site/runway and
distance. Refuse engagement with a visible reason if the craft is not an
airborne controllable aircraft or a destination plan cannot be built.

The existing `AUTOPILOT` panel becomes a mutually-exclusive mode surface:
`MNVR` for a scheduled burn, `LAND` for destination autoland. LAND's live
annunciation shows its phase (`ENRT`, `APPR`, `FLARE`, `ROLLOUT`) rather than
calling the entire trip APPR.

- pilot stick input (attitude falls out of the priority ordering automatically —
  but disengage explicitly so the mode annunciation is honest),
- pilot throttle movement (see the trap in A),
- pilot brake/ground-steer override,
- explicit LAND cancellation,
- the runway being cleared or the plan becoming physically impossible,
- destruction.

Mirror the maneuver autopilot's `ControlLocks` pattern for whatever it owns, and
make sure disengage leaves the craft in a flyable trim rather than dropping every
command in one frame. Normal successful disengagement happens only after
`Stopped`.

---

## Unit tests to write

The laws are pure functions if you write them that way — do, and test them:

- **Global ingress:** near, horizon-adjacent, far-side, polar, and exactly
  antipodal starts all produce finite great-circle guidance toward the selected
  arrival fix; the terminal handoff is continuous; the cruise altitude clears
  the body's published maximum terrain elevation by the configured margin.
- **Vertical law:** zero commanded vertical speed → pitch ≈ α (level flight);
  a commanded descent → pitch below that; output clamped at the band edges;
  finite at zero airspeed.
- **Speed law:** above target → throttle decreases; below → increases; the
  integrator does not wind up while saturated at idle.
- **Flare:** above flare height the pitch target equals the glideslope target
  exactly (no discontinuity at handover); at touchdown height the target is
  nose-up and the throttle is at idle.
- **Gate scheduling:** crossing the `GEAR` gate commands gear down and does not
  command it again; gates never fire in reverse when distance-to-go grows.
- **Rollout:** a bounce does not enter rollout; rollout commands idle/brakes and
  steers toward the centreline; crossing the speed threshold for less than the
  dwell does not complete; a stable stop engages the parking brake and is the
  sole normal completion.
- **Recovery:** each unstable airborne trigger enters `GoAround`, not `Off`;
  each true override/failure trigger disengages with the right reason; a normal
  on-profile frame does neither.
- **Mode ownership:** MNVR and LAND cannot both own the autopilot demand or
  control locks in one frame.
- **Throttle arbitration (work item A):** an engaged autothrottle wins over the
  pilot's resting setpoint; a pilot throttle *movement* wins and disengages;
  a manual disengagement restores the pilot's setpoint rather than leaving the
  autopilot's, while successful stopped completion holds realized idle until
  the pilot deliberately moves the throttle.
- **Ground arbitration:** LAND steering/braking wins while active; deliberate
  pilot yaw/brake movement disconnects it and takes control; no raw-input side
  path bypasses the winner.

Bank needs no test of its own — it is the guidance value unchanged, and the
guidance signs are already pinned in `thalos_navigation`.

---

## Instrumentation (this is what makes the play-session loop cheap)

One periodic record on the existing lane, per CLAUDE.md § Observability:

```
info!(target: "thalos::diagnostic::approach_ap", event = "appr_frame",
      phase = …, destination_id = …, dtg_m = …,
      loc_dev_rad = …, gs_dev_rad = …,
      bank_cmd_rad = …, pitch_cmd_rad = …, throttle_cmd = …,
      airspeed_m_s = …, target_speed_m_s = …, height_over_rwy_m = …,
      ground_speed_m_s = …, runway_cross_track_m = …,
      brake_cmd = …, ground_steer_cmd = …, retry_count = …)
```

at ≥ 1 s while engaged (allocation-free, and it must not dominate the lane), plus
one event for `land_engaged`, every phase transition and go-around,
`land_completed`, and `land_disengaged` with the reason. Then add the matching
check in `tools/diag/src/checks.rs` — CLAUDE.md is explicit that an event nobody
reads is cost with no signal. Checks: an abnormal disengagement, excessive
touchdown sink rate, runway excursion during rollout, excessive go-around
count, and a LAND session that ended without either completion or an explicit
disengagement reason. Threshold constants go in `finding.rs` with the reasoning,
tested both ways.

With that in place, "the landing felt wrong" is answerable from the file.

---

## Traps

- **Do not re-derive navigation.** Bank, vertical speed, target speed, and
  distance-to-go all come from `RouteState`. An autopilot that computes its own
  cross-track error will disagree with the needle the pilot is watching
  (ADR-20260730T005746Z).
- **Do not stretch the terminal plane around the planet.** `RouteFrame` is
  gnomonic and deliberately local. The spherical ingress and the runway-local
  terminal plan are different leg representations under one route authority.
- **Sign conventions**: positive bank = roll right, positive cross-track and
  localizer = right of course, positive glideslope = high. The display layer got
  the vertical sign wrong once already because it mirrored the lateral one.
- **Ground steering currently bypasses the bus.** The nosewheel reads raw pilot
  yaw. LAND rollout is not complete until that path consumes the arbitrated
  ground-steer demand.
- **Gimbal authority scales with throttle** (`gimbal_torque_full · throttle`).
  Irrelevant for a jet on approach, but if an autothrottle ever drives a
  rocket to idle it also removes its steering.
- **Every spawn scenario starts paused** at warp 0×. An engagement test that
  never resumes the sim will look like a dead autopilot.
