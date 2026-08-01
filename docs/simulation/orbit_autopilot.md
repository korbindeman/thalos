# Orbit-target autopilot

Status: MVP implemented 2026-07-31; runtime verification and ascent tuning
pending.

The feature has one player-facing promise:

> Configure an orbit, engage ORBIT, and let the current vehicle reach that
> orbit. On the ground this includes launch, atmospheric ascent, staging,
> cutoff, coast, and circularisation. In space it modifies the existing orbit
> through ordinary maneuver nodes and the existing automatic burn executor.

The ground case is the MVP. Transfer helpers grow from the same planner, but
they do not delay a reliable launch-to-orbit path.

## 1. One target, two entry modes

`TargetOrbit` is the authority. It describes the desired state independently of
how the craft gets there:

```rust
pub struct TargetOrbit {
    pub reference_body: BodyId,
    pub periapsis_altitude_m: f64,
    pub apoapsis_altitude_m: f64,
    pub plane: TargetPlane,
}

pub enum TargetPlane {
    PreserveCurrent,
    Nearest {
        inclination_rad: f64,
        direction: OrbitDirection,
    },
    Fixed {
        inclination_rad: f64,
        ascending_node_longitude_rad: f64,
        direction: OrbitDirection,
    },
}
```

The current UI presents a circular-altitude field and expands it to equal
periapsis/apoapsis values. An elliptical target exposes both altitudes.
`AUTO SET` is a one-shot suggestion action, not a displayed target value. On
the surface it writes the minimum directly reachable inclination plus
`PROGRADE` into the draft, preserving the body's rotational boost. In space it
writes the live orbit's inclination and direction. The fields therefore always
show the concrete values that will be planned; editing either keeps the choice
explicit.
`Nearest` means the cheapest plane with the requested inclination; it does not
pretend that the longitude of ascending node is fixed. `Fixed` remains in the
canonical contract for launch-window-sensitive contracts, but the first
planner rejects it explicitly; the orbit widget currently offers `AUTO SET`
plus `NEAREST` / `PRESERVE`.

The first ground MVP supports the current dominant body, prograde/retrograde,
target periapsis and apoapsis, and a reachable inclination. A direct ascent
whose requested inclination is below the launch-site latitude is rejected
before ignition unless a later dogleg planner explicitly supports it. The
planner never silently substitutes a different orbit.

Both entry modes compile the target into an `OrbitProgram`:

- **Already in space:** generate an ordinary `ManeuverSequence`, publish it as
  normal game-side maneuver nodes, and let the existing prediction, editing,
  warp-to-node, and `AutopilotBurnDirective` executor do the work.
- **On the surface or in atmospheric ascent:** run continuous ascent guidance
  until the craft is on a safe coast with the target apoapsis established, then
  generate the remaining circularisation/trim nodes and hand them to the same
  maneuver executor.

There is no direct state-vector rewrite and no second burn executor.

## 2. Orbit planning and maneuver generators

The pure planner belongs beside the canonical orbital mechanics, not in the UI.
It consumes a body state, craft state, epoch, and `TargetOrbit`; it produces a
typed plan result with maneuver nodes, total delta-v, estimated final elements,
and an explicit infeasibility reason.

The initial helper catalog is deliberately small:

1. `circularize_at_next_apsis`
2. `set_apoapsis`
3. `set_periapsis`
4. `change_plane_at_next_node`
5. `retarget_orbit`, which composes the helpers and may combine energy and
   plane changes when that is cheaper

Every generated burn is expressed in the existing
prograde/normal/radial frame and is therefore visible and editable exactly like
a manually placed node. Generated nodes carry provenance so replacing a target
replaces only that program's unexecuted nodes, never unrelated player nodes.

The planner validates its own result by propagating the generated sequence and
comparing the resulting osculating elements with the target tolerances. A plan
that does not converge is an error, not a plausible-looking set of nodes.

Hohmann and bi-elliptic helpers between same-body circular orbits are the next
catalog additions. Patched-conic body-to-body transfers, launch-window search,
Lambert targeting, rendezvous/phasing, and mid-course correction come later;
they use the same typed result and ordinary maneuver-node output.

## 3. Why ascent is not a maneuver-node chain

The current maneuver model is a timed, body-relative delta-v interpreted by a
two-body trajectory predictor. A launch is a finite burn through changing
gravity, atmosphere, mass, thrust, and staging topology. Encoding it as a long
row of impulsive nodes would make the displayed prediction authoritative where
it is least truthful and would still leave throttle, max-Q, staging, and cutoff
outside the plan.

Ascent is therefore a continuous guidance program with these phases:

1. **Preflight** — validate target geometry, launch window, active engines,
   staged thrust-to-weight, control authority, propellant, and a conservative
   delta-v margin. Publish the reason if engagement is refused.
2. **Vertical rise** — clear the pad and terrain with a radial attitude target.
3. **Turn initiation** — steer onto the target plane and begin a bounded pitch
   program without commanding an aerodynamically impossible angle of attack.
4. **Closed-loop ascent** — drive the predicted cutoff apoapsis, plane error,
   time-to-apoapsis, dynamic-pressure limit, and acceleration limit. This is
   state feedback, not a hard-coded pitch-versus-time script.
5. **Main-engine cutoff** — throttle down and cut off when the target apoapsis
   is established with a positive coast to apoapsis. Do not continue burning
   merely because a schedule says so.
6. **Coast** — point and warp using the normal maneuver-autopilot preparation
   rules.
7. **Circularise and trim** — generate the ordinary target-orbit maneuver set,
   execute it through the existing burn executor, then verify the achieved
   orbit from live state.

The MVP law need not be globally optimal. It must be closed-loop, robust across
the staged launchers the player can build, and honest about inability to reach
the requested orbit. A later optimal-guidance pass can replace the pure
guidance law without changing `TargetOrbit`, the phase owner, the control
authority, or the maneuver handoff.

## 4. Staging and control authority

ORBIT is a **flight program**, not a mode peer of MNVR. Autoflight is two
layers (ADR-20260731T232619Z): the strategic `FlightProgram` owns targets,
sequencing, and events, while the tactical channels — attitude and throttle —
are resolved per frame by `thalos_control::arbitrate` among the pilot, SAS, nav
modes, and one autopilot slot. ORBIT publishes guidance through `ControlDemand`
like any other source; the control bus commits its winning throttle demand to
the canonical throttle position, and ORBIT does not bypass that authority or
treat the SAS toggle as a hidden prerequisite.

Because it is a program rather than a mode, ORBIT *delegates* to the shared
scheduled-burn executor for the circularisation nodes it installs at MECO. The
handoff is expressed by yielding — surface guidance stops publishing, so
`resolve_autoflight` falls through to the executor — never by changing a mode.

Control locks are **declared** by each source (`required_locks()`) and unioned,
never derived from a selection enum. That is what lets the program hold
throttle and attitude through the ballistic coast while leaving **warp with the
player**, so warp-to-node works during the wait for circularisation.

Automatic staging is **commanded, not reactive** — the launch-vehicle
convention, where guidance predicts depletion and commands cutoff, and thrust
decay is the *confirmation* rather than the trigger:

- the staging topology and next-stage transaction remain owned by
  `StagingPlan`, and `activate_stage` is still the one canonical separation
  operation for both the space bar and automation;
- a cold stack is activated through that same acknowledged transaction before
  live TWR is checked, so `EXEC ORBIT` owns first-stage ignition as well as
  later separation;
- `StageSequencer` predicts burnout from the active stage's remaining
  propellant and its full-throttle mass flow, re-evaluated every frame so a
  throttle change re-predicts rather than lying;
- it arms and annunciates a countdown, commands cutoff a short lead before
  depletion, waits out thrust tail-off, and only then checks the separation
  interlocks — **actual throttle/fuel-gated thrust** decayed below threshold,
  angular rate below the re-contact limit. The engine rating is not a thrust
  measurement and never enters this interlock;
- separation is requested through the edge-triggered, acknowledged
  `StageDemand`, so a held condition cannot fire two stages;
- **guidance keeps steering through the whole sequence.** Only throttle is
  surrendered, and only for the few hundred milliseconds the sequence needs;
  the vehicle holds its pitch program across separation instead of pitching to
  local up;
- thrust collapse remains a **backup trigger**: it still stages, and it logs
  `stage_unpredicted`, which `just diag` reports as `stage_prediction_missed`.
  The healthy count is zero — a nonzero one falsifies the prediction and is the
  reason the backup path logs rather than silently covering for it;
- the sequence fails explicitly (`stage_exhausted` / `stage_refused`) if no
  usable stage remains, rather than waiting forever on an acknowledgement that
  will not come.

The mode disengages on explicit cancel, deliberate pilot takeover,
destruction, target invalidation, loss of control authority, unrecoverable
guidance divergence, or propulsion exhaustion. Disengagement cuts commanded
autoflight throttle to zero and records the reason. It never claims success
until the live post-burn orbit is within tolerance.

A deliberate pilot takeover leaves the target and original launch frame
intact. The editor changes `EXEC ORBIT` to `RESUME ORBIT`; selecting it reruns
the safety gate against the **live** position, velocity, remaining stages, and
remaining loss reserve, then returns directly to `RISE`, `TURN`, or `ASCENT`
for the altitude already reached. It never restarts the pad profile or charges
the consumed pad-to-current-state delta-v a second time.

Sandbox may expose ORBIT immediately. Programme mode gates target knowledge,
node following, and autonomous staging through the existing guidance-technology
ladder; this feature does not create a second progression model.

## 5. The orbit widget is the configuration surface

ORBIT is configured in the existing top-centre orbital widget
(`hud/orbital_panel.rs`, currently the compact AP/PE readout). There is no
separate orbit-planner screen and the trajectory MFD does not become a second
owner of the target.

The interaction follows the ND's runway pattern:

- the compact widget remains the always-readable current-orbit summary;
- clicking the orbital half expands an editor anchored directly below it
  without moving the altitude readout off screen centre;
- target edits and action buttons emit `OrbitTargetRequest`s; one orbit-program
  system is the sole writer of the selected target and plan state, just as
  `RouteRequest` keeps runway selection out of the ND;
- the expanded widget shows the configured target and the plan status; the
  trajectory view and map are projections of that same state.

The expanded editor currently contains:

- `CIRC` / `ELLIP` shape selection;
- one altitude for a circular orbit, or periapsis and apoapsis for an
  elliptical orbit;
- inclination and prograde/retrograde direction;
- `AUTO SET`, which populates concrete inclination/direction values, plus
  nearest-plane or preserve-current-plane policy;
- target orbit, estimated delta-v, phase, and refusal/warning text;
- `PLAN`, `EXEC ORBIT` / `RESUME ORBIT`, and `CLR`/`CANCEL`.

Fields use the same compact in-widget interaction vocabulary as the ND:
clickable selectors and `−` / `+` adjustment. Direct numeric entry and a fixed
plane/launch-window editor remain follow-up UI work.
Changing a field invalidates the previous preview and sends a new request; it
never mutates maneuver nodes ad hoc from the widget.

Collapsed while a program is active, the widget keeps AP/PE readable and adds
the target plus the current phase. Clicking it reopens the same editor, where
`CANCEL` is always reachable.

In space, `PLAN` creates previewable ordinary maneuver nodes. The player may
edit them before execution. On the ground, the first MVP previews the target,
estimated total delta-v, and current phase; the circularisation nodes appear
after MECO. Rich launch-azimuth/window/staging preview remains follow-up work.
The widget does not draw a high-confidence patched-conic atmospheric ascent
rail that the predictor cannot support.

The MFD trajectory plot and map view may draw the generated nodes and target
orbit, but remain projections. They neither configure the target nor own
execution.

The phase annunciation is stable and compact:
`PREFLT`, `WAIT`, `RISE`, `TURN`, `ASCENT`, `MECO`, `COAST`, `CIRC`, `TRIM`,
`COMPLETE`, or `ABORT`.

## 6. Feasibility and success

Preflight distinguishes at least:

- target below terrain/atmosphere safety floor;
- unreachable inclination from the site under the selected launch policy;
- fixed plane not currently in its launch window;
- initial thrust-to-weight at or below one;
- no controllable active propulsion;
- insufficient conservative staged delta-v;
- no valid staging path;
- target body not equal to the current dominant body in the MVP.

Warnings may be conservative; false success is forbidden. Delta-v sufficiency
uses the canonical staged vehicle data against a live remaining-energy
estimate: vis-viva departure energy to the target apoapsis, the remaining
apsis burn, only velocity already aligned with the selected launch plane, and
a named 1,200 m/s gravity/drag reserve that tapers to zero at the atmosphere
boundary. Re-arm therefore judges the mission still ahead, not a second launch
from the pad.

`COMPLETE` requires a live, post-thrust dwell inside named tolerances for
periapsis, apoapsis, inclination, and residual radial velocity. Merely consuming
the last node is not completion.

## 7. Diagnostics and verification

Implementation adds structured events to the existing diagnostics lane:

- `orbit_plan_result` — target, node count, delta-v, predicted residuals, and
  infeasibility kind;
- `orbit_autoflight_transition` — old/new phase and reason;
- `orbit_autoflight_guidance` — sampled no faster than once per second, carrying
  apoapsis error, plane error, time to apoapsis, dynamic pressure and limit,
  acceleration and limit, throttle demand, active stage, and remaining staged
  delta-v;
- `orbit_autoflight_complete` / `orbit_autoflight_abort` — achieved elements or
  typed abort reason.

`just diag` gets checks for a stalled phase, repeated stage demand, sustained
guidance divergence, and false completion outside tolerance. The events stay
after tuning as recurrence tells.

Agent-verifiable gates:

- pure tests for orbit-element conversion, each maneuver generator, target
  convergence, unreachable targets, launch-plane geometry, phase transitions,
  staging acknowledgement, and cutoff/circularisation handoff;
- deterministic ascent simulations across at least a single-stage rocket, a
  two-stage rocket, underpowered refusal, premature stage exhaustion, and a
  non-equatorial launch site;
- compile and clippy for the affected crates;
- extend `just ui-preview` with collapsed, expanded-ground, expanded-space,
  planned, executing, infeasible, and aborted orbit-widget states.

User-verifiable gates:

- configure and execute a new orbit from an existing orbit;
- configure the target entirely in the top-centre orbit widget, collapse it,
  reopen it, and confirm the same target and plan remain selected;
- launch a two-stage rocket from the ground, observe one action per stage, coast
  and circularise, then confirm the achieved orbit matches the configured
  target;
- take over briefly during ascent, reopen the widget, select `RESUME ORBIT`,
  and confirm guidance continues from the current phase with the same target;
- cancel during ascent and during coast, confirm throttle returns to zero and
  manual control returns cleanly;
- request an impossible orbit and confirm ignition is refused with a useful
  reason.

## 8. Deliberately deferred

- globally optimal ascent guidance;
- dogleg launches to inclinations below launch-site latitude;
- autonomous pad operations, countdown, clamps, or recovery;
- rendezvous, docking, formation flight, and proximity operations;
- body-to-body Lambert planning and gravity assists;
- low-thrust continuous-transfer optimisation;
- automatic abort-to-recovery trajectories.

These are extensions of the target/planner or phase machine. None requires a
second control stack or a second maneuver executor.
