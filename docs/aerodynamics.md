# Aerodynamics

Atmospheric flight forces (drag, lift, control surfaces) for spacecraft **and
aircraft**. The force physics is a small **native** model,
[`thalos_physics_canonical::aero`] (`crates/physics_canonical/src/aero.rs`); this
doc is that model plus the consumer-side integration story.

Status: **native bubble-side flight model.** A wingless craft (rocket/capsule)
gets a bluff-body drag config with weathervane stability, so it aligns with the
wind instead of tumbling. An aircraft gets lift + control + stability derived from
its wing parts (area / chord / span → reference geometry; cambered lift). Flight
controls (pitch/roll/yaw) are wired, and an F3 overlay draws colliders +
force/wind vectors. The moment coefficients are **derived from transport-category
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
    layer (`thalos_control`, see `docs/control.md`) allocates to the control
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

`crates/game/src/aero.rs` is the adapter. Each physics step
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
   (`crates/game/src/local_physics.rs`); otherwise Avian uses the collider
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

`TerrestrialAtmosphere` (`crates/world/src/atmosphere.rs`) carries an optional
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

## Authority & warp coupling

Aerodynamic flight is a **bubble** concern: it runs only while Avian owns
translation, at 1× warp.

- **In-atmosphere is a `Full`-role trigger.** `avian_role_from_inputs`
  (`crates/game/src/local_physics.rs`) returns `Full` when the craft is below the
  Kármán line, so Avian owns translation across the *whole* atmospheric column
  (Kármán line → ground), not just the ~20 km terrain-collider band.
- **Warp clamps to 1× in atmosphere.** `enforce_warp_altitude_limits`
  (`crates/game/src/bridge.rs`) caps warp to 1× below the Kármán line. Aero only
  runs in the live bubble, so warping would silently skip it. `apply_aero_forces`
  lives in `PhysicsSchedule`, so it only executes while physics is stepping —
  never under warp/pause or the `BodyFixed` regime.

EVA is excluded throughout (no aero surfaces attached).

## Handling feel — coefficients from transport derivatives

The moment coefficients (`crates/game/src/aero.rs`, live-tunable via the
`AeroTuning` resource over BRP) are mapped from standard transport-category
stability derivatives (Cm_α ≈ −1.2, Cm_q ≈ −25 including the α̇ lag this model
lacks, Cl_p ≈ −0.45, Cn_r ≈ −0.3; full-throw Cl_δa ≈ 0.06, Cm_δe ≈ 0.5). Two
mapping details: the model's damping term `coeff·ρ·V·S·L²·ω` is 4× the standard
`C_q·(ωL/2V)` non-dimensionalisation, so `coeff = C/4`; and the reference span is
the **full tip-to-tip wingspan** (2 × the largest half-panel — panels are single
half-wings), which also makes the aspect ratio (b²/S ≈ 9 for the Meridian) and
hence induced drag realistic.

What this buys: **felt inertia is real physics, not per-class tuning.** Rate
onset is `τ = I / (damp·ρ·V·S·L²)` — about 1.2 s in roll for the ~37 t Meridian
(rates build over a second-plus and coast to a stop, with a full-stick steady
roll rate of ~35°/s at approach speed), while a fighter-sized airframe's small
inertia and span land it at a few tenths of a second and triple-digit roll
rates. Heavy planes feel heavy and small planes nimble through their actual
mass and geometry. Full deflection commands the craft's real physical capability
(an airliner *can* roll at ~35°/s and pull to stall AoA — its pilots just
don't), so gentle inputs fly gently. `crates/game/src/aero.rs` has unit tests
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
(`GearTuning::steer_fade_speed_m_s`, `crates/game/src/local_physics.rs`): full
tiller throw at taxi speed, a couple of degrees at takeoff speed — the
real-world tiller→pedals split — so a hard yaw input at speed cannot command
the lateral grip that would trip the craft over its main gear. Tire grip
(`GearTuning::mu` = 0.8) is a dry-tire value: a skidding craft slides before
the contact force grows a tipping moment.

## Scope and roadmap

**Aircraft.** `aero::build_ship_aero_config` aggregates the blueprint's `Wing`
parts via `ShipBlueprint::wing_aero_panels` into one `AeroConfig`: reference area =
total lifting (non-vertical) panel area, chord = mean aerodynamic chord, span =
full wingspan (2 × max half-panel), aspect ratio = b²/S; cambered lift + trim +
the wing stability/damping/control coefficients. Engine thrust stays Thalos's
nose-forward throttle. The natural next refinement is deriving the per-axis
*control* coefficients from the authored `ControlSurface` geometry (span window ×
chord fraction × arm) instead of one whole-body constant, so aileron sizing in
the shipyard shows up in roll feel.

**Spacecraft (rockets, capsules).** A bluff-body config (reference area
`ShipStats::frontal_area_m2`, Cd a blunt-body constant), `CL_α = 0`, with
weathervane restoring + damping so it aligns nose-to-wind (prograde) instead of
tumbling. Trimming a blunt capsule heatshield-forward instead is future work.

**Planes from the construction editor (future).** When `docs/construction.md`'s
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
- **Compressibility / wave drag** — no Mach drag rise, so a full-throttle
  aircraft has no transonic wall and will happily exceed Mach 1. A
  drag-divergence term (`ΔCD ∝ (M − M_dd)²`) is the cheap fix once
  `evaluate_aero` is handed the Mach number.
- **Per-control-surface authority** — control coefficients are whole-body
  constants; deriving them from the authored `ControlSurface` geometry would
  make shipyard surface sizing show up in handling.

## File map

- `crates/physics_canonical/src/aero.rs` — the native model: `AeroConfig`,
  `ControlInputs`, `evaluate_aero` (+ unit tests).
- `crates/game/src/aero.rs` — `GameAeroPlugin`, `build_ship_aero_config`
  (panels → config), `attach_ship_aero`, `apply_aero_forces` (atmosphere sample +
  body-frame velocity + inertia-relative clamp + force write), the F3 overlay.
- `crates/shipyard/src/stats.rs` — `ShipBlueprint::wing_aero_panels` +
  `WingAeroPanel` (per-wing aerodynamic geometry, body frame).
- `crates/world/src/atmosphere.rs` — `AtmosphereProfile` + `AtmosphereSample` +
  `TerrestrialAtmosphere::sample_at_altitude_m`.
- `crates/world/src/body.rs` — `surface_pressure_pa` / `surface_gravity_m_s2`.
- `crates/game/src/local_physics.rs` — `CenterOfMass` pin, `craft_in_atmosphere`
  + the in-atmosphere `Full`-role trigger, `WeightOnWheels`.
- `crates/game/src/bridge.rs` — atmosphere warp clamp.
- `crates/game/src/hud/atmo_panel.rs` — TAS / dynamic-pressure / Mach readout
  (from `AeroReadout`).

## Verification

- `cargo test -p thalos_physics_canonical aero::` — lift/drag direction,
  speed-squared scaling, aft-drag weathervane restoring moment, aft-tail pitch
  damping.
- `cargo test -p thalos_world` — density-profile unit tests.
- `just game landing` / `just game final` — the capsule should decelerate to a
  terminal velocity (not lunar freefall) and trim retrograde instead of tumbling.
- `just game cruise` / `just game runway` — the Meridian should be controllable.
- BRP: read `thalos_game::aero::AeroReadout` on the ship for non-zero
  `dynamic_pressure_pa` / `airspeed_ms` during descent; read `AvianRole` to
  confirm in-atmosphere → `Full`; confirm warp pins to 1× below the Kármán line.
