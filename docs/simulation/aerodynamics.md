# Aerodynamics

Atmospheric flight forces (drag, lift, control surfaces) for spacecraft **and
aircraft**. The force physics is a small **native** model,
[`thalos_physics_canonical::aero`] (`crates/simulation/physics_canonical/src/aero.rs`); this
doc is that model plus the consumer-side integration story.

Status: **native bubble-side flight model.** A wingless craft (rocket/capsule)
gets a bluff-body drag config with weathervane stability, so it aligns with the
wind instead of tumbling. An aircraft gets lift + control + stability derived from
its wing parts (area / chord / span → reference geometry; cambered lift), plus a
**transonic wave-drag wall** (drag-divergence Mach from the authored sweep /
thickness via Korn) with an **air-breathing thrust lapse**, a shallow
**flight-configuration layer** (a three-detent flap lever and brakes-driven
spoilers, both authored as wing control-surface windows), and **per-surface
control authority** (the per-axis control coefficients derive from the
authored aileron/elevator/rudder windows and their real moment arms about the
CoM). Flight controls (pitch/roll/yaw) are wired, and an F3 overlay draws
colliders + force/wind vectors. The moment coefficients are **derived from transport-category
stability derivatives** (see *Handling feel* below), so felt inertia scales with
the craft's real mass and geometry. The Meridian (`ships/meridian.ron`, a
narrow-body airliner) is the reference test aircraft on the runway scenarios; its
wings carry an authored `WingRole` (`Lift` main wing, `Stabilizer` empennage)
that the config builder classifies on. Verified in-game (`just game cruise`): the
Meridian flies with bounded angular rates and a sane angle of attack — no
spin-up, no escape.

> **History.** This replaced a vendored LGPL flight-dynamics crate (`avian_fdm`).
> That crate's zone physics was sound, but Thalos used only ~30% of it behind a
> real integration tax (SAE-frame reconciliation, an Earth-atmosphere fork, a
> ground-aero hack, an off-screen-gizmo reframe), it served bluff bodies poorly,
> and it kept aero out of trajectory prediction. The native model is force-only
> like before but lives in pure Rust, removes that tax, and was the project's
> sole copyleft dependency — the stack is now permissive end-to-end.

## The model — a surface/panel sum

[`evaluate_aero`] is a pure function (glam f64, no Bevy):

```rust
fn evaluate_aero(
    vel_body: DVec3,    // craft CoM velocity relative to the air, body frame
    omega_body: DVec3,  // angular velocity, body frame
    density: f64,       // air density (kg/m³)
    cfg: &AeroConfig,   // whole-body reference geometry + coefficients
    controls: ControlInputs,
) -> AeroOutput          // net force + torque about the CoM, body frame
```

It is a **whole-body** model. From the air-relative velocity it derives one angle
of attack `α = atan2(−v·up, v·fwd)` and one sideslip `β = atan2(v·right, v·fwd)`,
then:

- **Forces:** lift `CL = CL0 + CL_α·α` (clamped past stall) perpendicular to the
  flow toward the dorsal side, plus drag `CD = CD0 + CL²/(π·e·AR)` opposing it,
  both scaled by `q̄·S`. A wingless craft sets `CL_α = 0` → pure drag.
  Two configuration overlays and one compressibility term extend this:
  - **Flaps / spoilers** (`ControlInputs::flap` / `spoiler`, each 0–1): flaps
    add `flap_dcl·flap` to CL0 (which also raises the stall-clamp ceiling —
    that's the lower stall speed) and `flap_dcd·flap²` to CD0; spoilers add
    `spoiler_dcd` and a negative `spoiler_dcl` (lift dump). See *Flight
    configuration* below.
  - **Wave drag**: past the craft's critical Mach (`mach_drag_divergence −
    0.108`) drag rises as `20·(M − M_crit)⁴` — the transonic wall. See
    *Compressibility* below. `evaluate_aero` takes the local speed of sound
    (≤ 0 disables all Mach effects).
- **Moments** (about the CoM, body frame) are three explicit, **unconditionally
  stable** terms scaled by `q̄·S·(arm)`:
  - **Restoring** — `cm0 − stability·α` in pitch and `−stability·β` in yaw, which
    turns the nose toward the relative wind (weathervane static stability; works
    for a wingless capsule too). The `cm0` trim offset puts the hands-off pitch
    trim point at a small positive AoA (`α_trim = cm0 / pitch_stability` ≈ the
    cruise attitude) instead of α = 0, so level cruise needs no held stick.
  - **Damping** — `−damp·(ρ·V·S·L²)·ω` on each axis, which **always opposes the
    angular rate** (this is what makes a spin impossible to pump).
  - **Control** — pitch/roll/yaw control-surface deflection × authority. The
    deflection is **not** the raw stick: it is the command the fly-by-wire
    layer (`thalos_control`, see `docs/simulation/control.md`) allocates to the control
    surfaces — the same command the reaction wheels execute, so the two
    effectors pull together instead of fighting (the old raw-stick path was
    half of the SAS jitter). Sourced from `RealizedControl::aero`.

**Why explicit moments, not an emergent per-surface strip sum.** A strip sum
(forces summed at each surface's real aerodynamic centre, stability emergent from
geometry) is physically elegant, but its rotation coupling *pumps energy* under
the bubble's explicit, per-frame-constant force integration: a spinning craft
autorotates to absurd rates and the `ω×r` tip velocities then explode the forces.
We tried it; the Meridian span up to ~1600 rad/s and shot off the planet. The
restoring + damping form cannot add energy — a disturbed craft settles to trim —
which is exactly what "super simple and robust" needs. Coefficients are
non-dimensional first cuts tuned in-game.

## Force-only coupling (Thalos owns mass, inertia, gravity)

`crates/runtime/game/src/aero.rs` is the adapter. Each physics step
[`apply_aero_forces`] reads the craft's Avian `LinearVelocity`/`AngularVelocity`,
rotates them into the body frame, samples the body's atmosphere for density, calls
`evaluate_aero`, and writes the result into the craft's `ConstantForce` /
`ConstantTorque`. Thalos keeps owning:

- **Mass / inertia** — surfaces have no collider; the explicit `Mass` /
  `AngularInertia` from `spawn_local_craft_body` stand.
- **Gravity / thrust** — applied as acceleration accumulators in
  `apply_local_forces`. The force and acceleration families **sum in the solver**;
  the canonical→Avian snap only zeroes the *acceleration* accumulators, so it never
  clobbers the aero force.

Two integration invariants, plus a safety clamp:

1. **The rigid body rotates about the real CoM.** `spawn_player_avian_body` pins
   `CenterOfMass(params.center_of_mass)` + `NoAutoCenterOfMass` for *every* ship
   (`crates/runtime/game/src/local_physics.rs`); otherwise Avian uses the collider
   centroid. The net aero force is applied at the CoM (`ConstantForce`) and the
   moment as a pure couple (`ConstantTorque`), so this keeps both correct (and the
   gear pivot right).
2. **Airspeed is the body-fixed velocity; wind = 0.** Ships integrate in the
   body-fixed (rotating) frame, so the Avian `LinearVelocity` is already
   surface-relative and the co-rotating airmass is static. Do **not** add an `ω×r`
   wind — it would double-count the co-rotation and give a parked craft a phantom
   `ω·R` airspeed.

Plus an **inertia-relative safety clamp** in `apply_aero_forces`: the force and
torque are bounded to the craft's own `mass · MAX_LIN_ACCEL` / `min_MOI ·
MAX_ANG_ACCEL` (≈10 g / 4 rad/s²). A real craft never exceeds that, so it never
binds normal flight, but it makes a numerical blow-up impossible — no `dt`/`q̄`/
spawn transient can impart more. A pathological physics step (`> 0.25 s`, e.g. the
multi-second gap behind the loading screen) is skipped entirely, since even a sane
force integrated over a huge `dt` is a huge impulse.

## Per-body atmosphere (`thalos_world`)

The *physics* atmosphere is distinct from the *render* atmosphere
(`scattering`): one decides how the air pushes, the other how the sky looks. It is
a **physical exponential** model — nothing Earth-hardcoded; everything derived from
authored surface conditions + the body's own gravity.

`TerrestrialAtmosphere` (`crates/domain/world/src/atmosphere.rs`) carries an optional
`profile: Option<AtmosphereProfile>` with the surface **thermodynamics** —
`surface_temperature_k`, `specific_gas_constant` (default 287 = Earth air),
`gamma` (1.4). The vertical density structure is derived by:

```rust
fn sample_at_altitude_m(&self, agl_m, surface_pressure_pa, surface_gravity_m_s2) -> AtmosphereSample
```

returning density, pressure, temperature, and speed of sound. With surface
temperature T₀ and gas constant R: ρ₀ = P₀/(R·T₀), **scale height H = R·T₀/g**
(from the body's own surface gravity), ρ(h) = ρ₀·e^(−h/H), a = √(γ·R·T₀); vacuum
at/above the Kármán line.

**Pressure is single-sourced:** `BodyDefinition::surface_pressure_pa()` reads it
from the terrain `AtmosphereSpec`, and `surface_gravity_m_s2()` is GM/r². For
Thalos (1 bar, g ≈ 9.06) this gives ρ₀ ≈ 1.225 and H ≈ 9.1 km. Isothermal-from-
surface-T is the current vertical model; an ISA-style lapse rate is the planned
refinement, behind the same call site.

## Compressibility — the transonic wall

A subsonic airframe must not casually cross Mach 1; before this existed the
Meridian would happily blow through it on four 50 kN turbojets. Two physical
mechanisms close that hole — there is no artificial speed clamp:

1. **Wave drag** (in `evaluate_aero`): past the critical Mach,
   `ΔCD = 20·(M − M_crit)⁴` — the canonical transonic drag-rise shape (+20
   drag counts at the divergence Mach `M_dd = M_crit + 0.108`, then a steep
   wall: several × CD0 by M ≈ 1). The Mach excess is capped at 0.5 so a
   hypothetical hypersonic entry stays "very draggy" instead of numerically
   absurd; the inertia-relative force clamp still backstops everything.
2. **Jet thrust lapse** (in `fuel::refresh_active_propulsion`): air-breathing
   engines (`requires_atmosphere`) scale thrust — and therefore mass flow — by
   `ρ/1.225`, the ambient density over the catalog's sea-level rating. This
   kills the "climb into thin air and keep accelerating" exploit and makes
   service ceilings real. Rockets carry their own oxidizer and are unaffected.
   The factor varies continuously with altitude, so
   `propulsion_config_changed` compares thrust/mass-flow with a *relative*
   (1%) threshold — an absolute epsilon would re-dirty trajectory prediction
   every frame of a climb.

**`M_dd` is derived from the authored wing geometry, not tuned per craft**
(`build_ship_aero_config`): the Korn equation
`M_dd = κ/cosΛ − (t/c)/cos²Λ − CL/(10·cos³Λ)` on the area-weighted authored
sweep and thickness (κ = 0.87 conventional airfoil, CL = the camber CL0 as a
cruise proxy, clamped to [0.5, 0.95]). Sweeping the wing and thinning the
airfoil in the shipyard buys real transonic margin — the same trade the
historical designers made. The Meridian's 30°-swept, 11%-thick wing lands at
M_dd ≈ 0.82 and tops out around M 0.80–0.84 at every altitude (unit-pinned by
`meridian_cannot_sustain_transonic_flight`), right in the 707/Comet band.
Bluff bodies keep `mach_drag_divergence = 0` (disabled): a capsule's
blunt-body Cd already stands in for its transonic behaviour, and reentry
should not see a quartic surprise.

## Flight configuration — flaps & speedbrake

Deliberately shallow (KSP-style): two controls, no trim/mixture/cowl-flap
management. `crates/runtime/game/src/flight_config.rs` owns the state
(`FlightConfig`); `assets/input.ron` binds the keys.

- **Flap lever** — `F` extends a detent, `R` retracts: UP → T/O → LDG.
  The aero model scales flap lift linearly with deployment and flap drag
  *quadratically*, so the middle detent is automatically the high-lift/
  low-drag takeoff setting and full is the draggy landing one — no extra
  authoring. Per-craft `flap_dcl`/`flap_dcd` derive from the authored
  `Flap` control-surface windows (plain-flap theory: ΔCL from
  `CLα·τ(c_f)·η·δ·S_flapped/S_ref`, ΔCD from Roskam's
  `1.7·c_f^1.38·(S_f/S)·sin²δ`), so resizing the flaps in the shipyard
  changes landing performance. The Meridian's landing flaps add ΔCL ≈ 0.7 /
  ΔCD ≈ 0.06 — a ~17% lower stall speed (pinned by
  `meridian_flaps_buy_a_slow_approach`). **Flap load relief**: above
  ~10 kPa dynamic pressure the effective deployment fades as `q_relief/q`
  (`apply_aero_forces`), so slamming landing flaps at cruise speed produces
  a gentle balloon instead of a 10 g pull-up — no placard speeds to manage.
- **Brakes** — the existing `B` latch is now unified KSP-style brakes:
  wheel brakes on the ground *and* spoilers in the air. `Spoiler` windows
  deflect trailing-edge-up when engaged, dumping lift (`spoiler_dcl < 0`)
  and adding drag — the in-air deceleration tool, and it is already latched
  for the rollout at touchdown.

Both are authored as [`ControlSurface`] windows on the wing (roles `Flap` /
`Spoiler` next to `Aileron`/`Elevator`/`Rudder`), so they get hinged meshes
and animation through the same path as the attitude surfaces — but they
deflect from the `FlightConfig` *actuator positions*, not the fly-by-wire
command. Actuators chase their targets at real travel rates (flaps ~6 s full
travel, spoilers ~0.8 s) on `SimClock`, and the aero model consumes the same
smoothed positions the visuals show, so deployment forces build smoothly. A
freshly placed `Lift` wing gets a default inboard flap + outboard aileron
(`default_control_surfaces`).

The HUD shows a **capability-gated flight-config cluster**
(`hud/flight_config_panel.rs`, under the atmosphere readout, styled like the
nav panel's SAS/RCS toggle buttons): flaps are a **segmented gate**
`FLAPS [ UP · T/O · LDG ]` — clicking a segment drives the lever straight to
that detent (one click to any position, never the wrong direction, and it
doubles as a lever-position readout; the commanded detent is highlighted and
glows amber while the actuator is in transit). The gate appears only when the
craft's aero config derived flap authority from authored `Flap` windows. The
brakes pill is a single latched toggle (click toggles the latch, same as `B`)
and shows only when the craft has gear wheels or spoilers. A rocket shows
neither. (The flap gate replaced an earlier single one-directional cycling
pill — `UP → T/O → LDG → UP` — which couldn't express "retract" and needed
two clicks to clean up after takeoff.)

## Authority & warp coupling

Aerodynamic flight is a **bubble** concern: it runs only while Avian owns
translation, at 1× warp.

- **In-atmosphere is a `Full`-role trigger.** `avian_role_from_inputs`
  (`crates/runtime/game/src/local_physics.rs`) returns `Full` when the craft is below the
  Kármán line, so Avian owns translation across the *whole* atmospheric column
  (Kármán line → ground), not just the ~20 km terrain-collider band.
- **Warp clamps to 1× in atmosphere.** `enforce_warp_altitude_limits`
  (`crates/runtime/game/src/bridge.rs`) caps warp to 1× below the Kármán line. Aero only
  runs in the live bubble, so warping would silently skip it. `apply_aero_forces`
  lives in `PhysicsSchedule`, so it only executes while physics is stepping —
  never under warp/pause or the `BodyFixed` regime.

EVA is excluded throughout (no aero surfaces attached).

## Handling feel — coefficients from transport derivatives

The **stability and damping** coefficients (`crates/runtime/game/src/aero.rs`,
held in the `AeroTuning` resource) are mapped from standard
transport-category stability derivatives (Cm_α ≈ −1.2, Cm_q ≈ −25 including
the α̇ lag this model lacks, Cl_p ≈ −0.45, Cn_r ≈ −0.3). The **control**
coefficients are no longer constants: they derive per craft from the authored
control-surface windows (see *Per-surface control authority* below), with
`AeroTuning` carrying scale multipliers over the derived values. Two mapping
details: the model's damping term `coeff·ρ·V·S·L²·ω` is 4× the standard
`C_q·(ωL/2V)` non-dimensionalisation, so `coeff = C/4`; and the reference span is
the **full tip-to-tip wingspan** (2 × the largest half-panel — panels are single
half-wings), which also makes the aspect ratio (b²/S ≈ 9 for the Meridian) and
hence induced drag realistic.

What this buys: **felt inertia is real physics, not per-class tuning.** Rate
onset is `τ = I / (damp·ρ·V·S·L²)` — about 1 s in roll for the ~37 t Meridian
(rates build over a second-plus and coast to a stop, with a full-stick steady
roll rate of ~22°/s at approach speed from the derived aileron authority —
author bigger ailerons for more), while a fighter-sized airframe's small
inertia and span land it at a few tenths of a second and triple-digit roll
rates. Heavy planes feel heavy and small planes nimble through their actual
mass and geometry. Full deflection commands the craft's real physical capability
(an airliner *can* roll at ~35°/s and pull to stall AoA — its pilots just
don't), so gentle inputs fly gently. `crates/runtime/game/src/aero.rs` has unit tests
pinning the Meridian's aspect ratio, steady roll rate, roll-onset τ, and pitch
trim authority to these bands so a retune can't silently regress the feel.

**Aircraft have no reaction wheels.** The aircraft command pods (`cockpit`,
`flightdeck`, `cockpit_inline` in `assets/parts.ron`) author
`reaction_wheel_torque: 0`: a plane's only attitude authority is its control
surfaces, scaled by dynamic pressure, exactly like reality. Controls are mushy
below flying speed, dead when parked, and crisp at cruise — and there is no
free, airspeed-independent torque to roll the craft over on the runway or
rotate it below V_r. (Rocket pods keep their wheels; the fly-by-wire allocator
already drives whichever effectors exist.)

## Ground handling

Below a taxi airspeed floor (`GROUND_AERO_AIRSPEED_FLOOR_M_S`, 5 m/s) a craft
with weight on wheels gets **no aero at all** — at near-zero airspeed the AoA is
degenerate (the velocity is suspension settle, not flow), so the gear owns the
craft outright. Above the floor a grounded craft flies the **full aero model**:
elevator authority builds with q̄ until rotation at V_r, the wings damp roll
through the takeoff run, and the fin weathervanes the nose — all real
aerodynamics, with no discontinuity at liftoff or touchdown. (The old blanket
weight-on-wheels moment zeroing existed to protect against the previous
over-damped coefficients, which were strong enough at taxi speed to fight the
suspension.)

Nosewheel steering fades with ground speed
(`GearTuning::steer_fade_speed_m_s`, `crates/runtime/game/src/local_physics.rs`): full
tiller throw at taxi speed, a couple of degrees at takeoff speed — the
real-world tiller→pedals split — so a hard yaw input at speed cannot command
the lateral grip that would trip the craft over its main gear. Tire grip
(`GearTuning::mu` = 0.8) is a dry-tire value: a skidding craft slides before
the contact force grows a tipping moment.

## Thrust vectoring (engine gimbal)

A wingless rocket has **no aero control authority** (`pitch_control = yaw_control =
0` in the bluff-body config) — only its weathervane restoring moment and its
reaction wheels. On a launch-mass stack the wheels are hopelessly weak: a ~150 t,
~20 m rocket has a pitch/yaw MOI ≈ 5×10⁶ kg·m², so a 40 kN·m command pod buys
≈ 0.5°/s² — and at max-q the aero weathervane restoring moment (≈ 10⁵ N·m at a few
degrees AoA) *overpowers* the wheels entirely. Reaction wheels cannot fly a
gravity turn. Real launch vehicles steer by **vectoring the engine thrust**, and
so does Thalos.

**Authoring.** `EngineSpec::gimbal_range_deg` (degrees; `0` = fixed bell) is the
peak thrust-vector deflection from the nose axis. Jets and upper-stage verniers
can leave it `0`; booster and vacuum main engines carry a few degrees (`assets/
parts.ron`: Zephyr 6°, Boreas 5°, Typhon 8°).

**Model.** `staging::recompute_ship_inertia` aggregates every gimballed engine
into `ShipParameters::gimbal_torque_full` = Σ `thrust · sin(range) · arm`, the
attitude torque **at full thrust**, where `arm` is the engine's axial distance to
the live CoM. Pitch and yaw share it (an axisymmetric bell gimbals both ways);
roll stays `0` (a centred engine can't roll the stack — roll remains a
reaction-wheel / future-RCS job). It is recomputed every frame, so the arm tracks
the CoM as fuel burns.

**Effective authority scales with throttle.** The gimbal only steers while there
is exhaust to vector: `fuel::active_thrust_fraction` gates the full-thrust term by
the fraction of thrust actually firing (zero at idle throttle, out of fuel, and
during coast). That single helper feeds both consumers so they can't disagree:

- the fly-by-wire controller adds `gimbal_torque_full · throttle` to its **effector
  authority** (`thalos_control::attitude`, the same `effector_authority` slot the
  aero surfaces use), so the PD normalizes its command by the real total; and
- `local_physics::forces::compute_angular_acceleration` realizes the same
  `command · (max_torque + gimbal_effective)`, so the torque produced equals the
  torque the controller intended (the "drive every effector at one fraction,
  normalize by the total" rule the allocator already uses for wheels + surfaces).

**Gameplay.** This makes the ascent flyable: lift off under SAS (the quaternion
`Hold` pins the vertical attitude), pitch a few degrees off the pad, then hold
surface **Prograde** (`navigation`, which resolves through the Surface velocity
frame) and ride the gravity turn to orbit — the gimbal has the authority to set
the pitch program, while the weathervane keeps AoA small in between. The
reference vehicle is `ships/atlas.ron` (a two-stage methalox rocket, ~10 km/s Δv,
both stages gimballed); guarded by `blueprint::tests::atlas_sample_is_an_orbital_rocket`.

## Scope and roadmap

**Aircraft.** `aero::build_ship_aero_config` aggregates the blueprint's `Wing`
parts via `ShipBlueprint::wing_aero_panels` into one `AeroConfig`: reference area =
total lifting (non-vertical) panel area, chord = mean aerodynamic chord, span =
full wingspan (2 × max half-panel), aspect ratio = b²/S; cambered lift + trim +
the wing stability/damping coefficients. Engine thrust stays Thalos's
nose-forward throttle.

**Per-surface control authority** (`derive_control_coefficients`): the per-axis
*control* coefficients are derived from the authored `ControlSurface` windows,
not tuned constants. Each aileron/elevator/rudder window contributes its
deflection lift (the same plain-flap term the flaps use, `CLα·τ(c_f)·η·δ_max`
over the spanned strip) times its **real moment arm about the CoM**
(`|(r × n̂)·axis|` with `r` the window centroid and `n̂` the panel's lift
normal), summed per role and non-dimensionalised into the evaluator's
`coeff·q̄·S·L` control term. So a bigger or further-outboard aileron rolls
harder, a longer tail arm pitches harder, and a craft authored without a
rudder genuinely has no yaw authority — sizing and placement show up in
handling exactly like flap sizing shows up in approach speed. Each role feeds
only its own axis (cross-couplings like rudder-roll are deliberately dropped:
the whole-body model keeps control moments axis-diagonal so fly-by-wire
allocation stays unconditionally stable — the same "explicit, not emergent"
reasoning as the restoring/damping terms; the **forces** never became a
per-surface strip sum, which is the thing that pumped energy when tried). On
the Meridian the derived values land within ~10% of the previously hand-tuned
transport constants (pitch 0.48 vs 0.5, yaw 0.032 vs 0.04, roll 0.037),
pinned by `meridian_control_authority_derives_from_surfaces`. The `AeroTuning`
resource now carries `*_control_scale` multipliers (default 1) over the
derived values instead of absolute control overrides, so a feel-tweak
can't erase the difference between a big and a small aileron.

**Spacecraft (rockets, capsules).** A bluff-body config (reference area
`ShipStats::frontal_area_m2`, Cd a blunt-body constant), `CL_α = 0`, with
weathervane restoring + damping so it aligns nose-to-wind (prograde) instead of
tumbling. Attitude authority comes from the reaction wheels plus **engine gimbal**
(see *Thrust vectoring* above) — the gimbal is what makes a controllable ascent
possible. Trimming a blunt capsule heatshield-forward instead is future work.

**Planes from the construction editor (future).** When `docs/gameplay/construction.md`'s
wing **Modules** exist, the same `wing_aero_panels` aggregation generalises
(control surfaces become wing parameters).

## Debug view (F3)

`F3` toggles a game-wide overlay: **Avian collider wireframes** (every physics
body) plus the native aero **force / wind vectors** drawn at the *rendered* ship
pose (the net aero force from the origin and the relative-wind arrow). The forces
are computed in the
body-centered bubble frame (~planet-radius from the floating origin, where an f32
`GlobalTransform` would quantise), so they are mapped onto the rendered
`PlayerShip` — same rigid body, shared body frame — via the body rotation (exact
for directions). Both groups start disabled.

## Deferred

- **Aero in prediction / warp.** Bubble-only today; predicted trajectories don't
  yet account for drag. The evaluator is pure `thalos_physics_canonical`, so it can
  now be wired into the shared `ShipPropagator` — the natural next step.
- **Orbital decay under warp** (`Realistic` preset): a separate, simpler drag
  model outside the bubble.
- **Reentry heating / heat-flux destruction** — an extension of the impact-
  destruction model (`ShipParameters::impact_tolerance_m_s`).
- **ISA-style temperature lapse** — drops in behind `sample_at_altitude_m`.
- **Heatshield-forward capsule trim** — the weathervane aligns nose-to-wind
  (prograde); a blunt capsule that should re-enter base-first needs an offset
  trim angle / a per-craft "stable attitude".
- **Per-nose-shape drag coefficient** — Cd is a single blunt-body constant.
- **Supersonic regimes** — the wave-drag wall is authored for *subsonic*
  airframes (capped quartic). A craft meant to fly supersonic needs a
  proper supersonic drag/lift model and an area-rule-ish authored escape
  (e.g. higher κ via a supercritical/sharp airfoil flag).
- **Per-surface forces / cross-couplings** — control *authority* now derives
  from the authored surface geometry (see *Per-surface control authority*),
  but forces remain a whole-body sum and the axis cross-terms (rudder-roll,
  adverse yaw, differential-flap asymmetry after damage) are deliberately
  dropped for stability. An emergent per-surface strip-sum force model was
  tried and rejected — it pumps energy under per-frame-constant integration.

## File map

- `crates/simulation/physics_canonical/src/aero.rs` — the native model: `AeroConfig`,
  `ControlInputs` (incl. flap/spoiler deployment), `evaluate_aero` (incl.
  wave drag; + unit tests).
- `crates/runtime/game/src/aero.rs` — `GameAeroPlugin`, `build_ship_aero_config`
  (panels → config, incl. Korn `M_dd` + flap/spoiler coefficient derivation +
  `derive_control_coefficients` for per-surface control authority),
  `attach_ship_aero`, `apply_aero_forces` (atmosphere sample + body-frame
  velocity + inertia-relative clamp + force write), the F3 overlay.
- `crates/runtime/game/src/flight_config.rs` — the flap lever / spoiler actuator
  state (`FlightConfig`) and its input handling.
- `crates/runtime/game/src/hud/flight_config_panel.rs` — the capability-gated
  flaps/brakes HUD pills.
- `crates/runtime/game/src/fuel.rs` — `air_breathing_thrust_factor` (jet density
  lapse) inside `refresh_active_propulsion`; `active_thrust_fraction` (the
  throttle gate the gimbal authority scales by).
- `crates/runtime/game/src/staging.rs` — `recompute_ship_inertia` aggregates every
  gimballed engine into `ShipParameters::gimbal_torque_full`.
- `crates/runtime/game/src/local_physics/forces.rs` — `compute_angular_acceleration`
  realizes `command · (max_torque + gimbal_effective)`; `apply_local_forces`
  computes the throttle-scaled `gimbal_effective`.
- `crates/simulation/control/src/attitude.rs` — the controller normalizes by
  `max_torque + effector_authority`, where `effector_authority` = aero
  surfaces + engine gimbal (fed from `control_bus::realize_control`).
- `crates/domain/construction/src/stats.rs` — `ShipBlueprint::wing_aero_panels` +
  `WingAeroPanel` (per-wing aerodynamic geometry incl. sweep/thickness and
  the `AeroSurfaceWindow` control-surface windows with body-frame centroids
  for the moment arms).
- `crates/domain/world/src/atmosphere.rs` — `AtmosphereProfile` + `AtmosphereSample` +
  `TerrestrialAtmosphere::sample_at_altitude_m`.
- `crates/domain/world/src/body.rs` — `surface_pressure_pa` / `surface_gravity_m_s2`.
- `crates/runtime/game/src/local_physics.rs` — `CenterOfMass` pin, `craft_in_atmosphere`
  + the in-atmosphere `Full`-role trigger, `WeightOnWheels`.
- `crates/runtime/game/src/bridge.rs` — atmosphere warp clamp.
- `crates/runtime/game/src/hud/atmo_panel.rs` — TAS / dynamic-pressure / Mach readout
  (from `AeroReadout`).

## Verification

- `cargo test -p thalos_physics_canonical aero::` — lift/drag direction,
  speed-squared scaling, aft-drag weathervane restoring moment, aft-tail pitch
  damping.
- `cargo test -p thalos_world` — density-profile unit tests.
- `just game landing` / `just game final` — the capsule should decelerate to a
  terminal velocity (not lunar freefall) and trim retrograde instead of tumbling.
- `just game cruise` / `just game runway` — the Meridian should be controllable.
- `cargo test -p thalos_game aero` — Meridian handling-feel pins, the
  transonic-wall pin (`meridian_cannot_sustain_transonic_flight`), the
  flap approach-speed pin (`meridian_flaps_buy_a_slow_approach`), and the
  per-surface authority pins (`meridian_control_authority_derives_from_surfaces`:
  derived pitch/roll/yaw in the transport band, bigger ailerons roll harder,
  no rudder → no yaw authority).
- In `cruise`: `F`/`R` step FLAPS UP→T/O→LDG (or click a segment of the HUD
  flap gate to jump straight to a detent) and the inboard trailing edges
  visibly run out; `B` pops the mid-span spoilers ("BRAKES ON") and the craft
  decelerates; full throttle tops out around M 0.8 instead of punching through
  Mach 1. In `orbit` (Apollo), the flap gate and brakes pill must not appear at
  all.
- Verify in-atmosphere flight by watching the HUD during descent: airspeed
  and dynamic pressure (`AeroReadout`'s `airspeed_ms` / `dynamic_pressure_pa`)
  go non-zero, the flap/brake state tracks the `FlightConfig` lever, the
  regime resolves to in-atmosphere → `Full`, and warp pins to 1× below the
  Kármán line. Add an `info!` log on these resources if you need exact values
  in the console.
