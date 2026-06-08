# Aerodynamics

Atmospheric flight forces (drag, lift, control surfaces) for spacecraft **and
aircraft**. This is the **consumer-side contract** and the integration story;
the force *physics* is provided by the vendored [`avian_fdm`] crate
(`crates/avian_fdm/`).

Status: **bubble-side flight model.** Aircraft get a full set of lift + control
zones derived from their wing parts (cambered main wing, symmetric
stabiliser/fin; elevator/aileron/rudder control surfaces); spacecraft get a
bluff-body drag zone. Flight controls (pitch/roll/yaw → elevator/aileron/rudder)
are wired, and an F3 debug overlay draws colliders + force/moment vectors. The
**airfoil and control-authority constants are a first cut that needs in-game
tuning** (static margin, control sign/gain, stall, takeoff speed). The Meridian
(`ships/meridian.ron`, a narrow-body airliner) is the reference test aircraft on
the runway scenarios; its wings carry an authored `WingRole` (`Lift` main wing,
`Stabilizer` empennage) that the zone generator classifies on.

## Why a vendored crate (and the LGPL story)

[`avian_fdm`] (by Víctor Cuadrado Juan) is a zone-based 6-DoF flight-dynamics
model for Bevy 0.18 / Avian 0.6: per-zone tabulated CL/CD/CY vs angle-of-attack
and Reynolds number, Viterna-Corrigan post-stall extrapolation, Oswald induced
drag, per-zone damping, and a `FlightState` readout. Its physics is exactly what
Thalos wants, so we **reuse its force pipeline** rather than reinvent it.

It is **LGPL-3.0-or-later** (its J-3 Cub preset crate is GPL-3.0-only — we do not
depend on it and author our own zones). This is the only copyleft entry on an
otherwise permissive stack.

**The LGPL obligation is satisfied by full-source distribution.** Thalos is now
fully source-available (code under PolyForm Noncommercial, assets under CC BY —
see `LICENSING.md`). LGPL's relink requirement on a statically linked binary is
met automatically when all the application source is published: anyone can
already rebuild Thalos against a modified `avian_fdm`. So no relicense and no
replacement is required — even for the paid build. The noncommercial restriction
applies to *Thalos's own* code, not to `avian_fdm`, which stays LGPL and remains
extractable on its own terms.

Two hard rules keep this clean:

1. **Never add a GPL/AGPL (non-LGPL) dependency** — in particular `avian_fdm`'s
   GPL-3.0-only J-3 Cub preset crate (`avian_fdm_j3cub_jsbsim`). GPL is viral
   across the whole combined work and would force all of Thalos to GPL, voiding
   the noncommercial model. CI guards this (`.github/workflows/ci.yml`).
2. **Keep `avian_fdm` isolated** behind the Bevy-side crates (`game` /
   `physics_local`) and force-only, so it can be swapped out later without
   touching the simulator core if we ever want to.

Replacing it with a fresh implementation of the same (uncopyrightable) physics
remains an *option* (it would let the force math move into pure
`thalos_physics_canonical` and feed trajectory prediction — see "Deferred"), but
it is no longer a licensing obligation.

The crate is **vendored in-tree** (`crates/avian_fdm/`, like `crates/udlod/`)
rather than pulled from crates.io because Thalos must edit its environment
assumptions (below). Vendored modifications travel under LGPL-3.0-or-later; the
`LICENSE` is carried in the crate directory. Each Thalos change is flagged with a
"vendored-fork addition" comment so it can be upstreamed.

## The two environment mismatches `avian_fdm` assumes away

`avian_fdm` assumes it owns a single Earth-bound aircraft. Thalos flies arbitrary
vehicles in a **body-centered-inertial floating-origin bubble** over a
**rotating** planet. Two assumptions had to be redirected:

1. **Atmosphere.** Upstream `update_atmosphere` derives Earth-ISA density from the
   root's world-space **Y** coordinate — meaningless in the bubble (Y is a
   ±planet-radius offset, not altitude). A vendored-fork plugin flag,
   `AircraftFdmPlugin::manage_atmosphere = false`, skips that system so Thalos
   owns `AtmosphereState` instead (`crates/game/src/aero.rs::sync_aero_environment`),
   filling it from the dominant body's per-body density model at the craft's real
   altitude above the mean surface.
2. **Airspeed.** Upstream airspeed is the raw inertial `LinearVelocity`. The local
   airmass **co-rotates** with the planet, so true airspeed is `v − ω×r` (the
   co-rotation term reaches a few hundred m/s near a planet's equator). Thalos
   publishes `ω×r` into `avian_fdm`'s existing `WindResource`
   (`vel_world = lin_vel − wind`) — **no fork needed for this part**, the wind
   hook already exists.

## Force-only use (Thalos owns mass, inertia, gravity)

`avian_fdm` only writes `ConstantForce` / `ConstantTorque`. Thalos keeps owning:

- **Mass / inertia** — the ship's `AeroZone`s carry **no collider**, so Avian's
  mass-properties model is untouched (`compute_aero_forces` reads each zone's
  `area_m2` + `Transform`, not its collider). Thalos's explicit `Mass` /
  `AngularInertia` from `spawn_local_craft_body` stand.
- **Gravity** — applied as `−μr/r³` via `ConstantLinearAcceleration` in
  `apply_local_forces`; Avian's global gravity stays disabled.

The two accumulator families (`ConstantForce`/`ConstantTorque` for aero,
`ConstantLinearAcceleration`/`ConstantAngularAcceleration` for gravity/thrust)
**sum in the solver** — no conflict. The canonical→Avian snap only zeroes the
*acceleration* accumulators, so it never clobbers the aero force.

## Per-body atmosphere (`thalos_world`)

The *physics* atmosphere is distinct from the *render* atmosphere
(`scattering`): one decides how the air pushes, the other how the sky looks.
It is a **physical exponential** model — nothing is Earth-hardcoded; everything
is derived from authored surface conditions + the body's own gravity.

`TerrestrialAtmosphere` (`crates/world/src/atmosphere.rs`) carries an optional
`profile: Option<AtmosphereProfile>` with the surface **thermodynamics** —
`surface_temperature_k`, `specific_gas_constant` (default 287 = Earth air),
`gamma` (1.4). The vertical density structure is *not* authored; it is derived
by the pure helper:

```rust
fn sample_at_altitude_m(&self, agl_m, surface_pressure_pa, surface_gravity_m_s2) -> AtmosphereSample
```

which returns density, pressure, temperature, and speed-of-sound. With surface
temperature T₀ and gas constant R: ρ₀ = P₀/(R·T₀), **scale height H = R·T₀/g**
(from the body's own surface gravity), ρ(h) = ρ₀·e^(−h/H), and a = √(γ·R·T₀).
Returns vacuum at/above the Kármán line.

**Pressure is single-sourced**: `BodyDefinition::surface_pressure_pa()` reads it
from the terrain `AtmosphereSpec` (`Breathable(pressure_bar)` etc.) so it isn't
authored twice; `surface_gravity_m_s2()` is GM/r². For Thalos (1 bar, g ≈ 9.06)
this gives ρ₀ ≈ 1.225 and H ≈ 9.1 km — note H follows the *real* gravity, not a
borrowed render scale height. When no `profile` is authored, Earth-like surface
conditions are assumed.

Isothermal-from-surface-T is the current vertical model; an ISA-style lapse
rate (T decreasing with altitude → altitude-varying Mach) is the planned
refinement and drops in behind the same `sample_at_altitude_m` call site.

## Authority & warp coupling

Aerodynamic flight is a **bubble** concern: it runs only while Avian owns
translation, at 1× warp. Two coupling points make that correct:

- **In-atmosphere is a `Full`-role trigger.** `avian_role_from_inputs`
  (`crates/game/src/local_physics.rs`) returns `Full` when the craft is below the
  Kármán line, alongside the existing thrust / terrain-contact triggers. This is
  what makes Avian own translation across the **whole** atmospheric column
  (Kármán line → ground), not only inside the ~20 km terrain-collider band —
  otherwise a reentering craft would Kepler-coast drag-free through the upper
  atmosphere and only feel drag at 20 km.
- **Warp clamps to 1× in atmosphere.** `enforce_warp_altitude_limits`
  (`crates/game/src/bridge.rs`) caps the warp level to 1× whenever the craft is
  below the Kármán line (KSP-style). Aero only runs in the live bubble, so
  warping in atmosphere would silently skip it.

EVA is excluded throughout (no aero zones attached), exactly as it is excluded
from terrain contact.

## Body-frame reconciliation (`AeroFrame`)

`avian_fdm` works in **SAE** body axes (X = nose, Y = right, Z = down); Thalos
ships are **Y = nose, X = right, Z = up**. The vendored crate gained an
[`AeroFrame`] component carrying a fixed `sae_to_entity` rotation (a 180° turn
about (1,1,0)/√2), threaded through `update_flight_state`, `compute_aero_forces`,
and the debug gizmos. Thalos sets it on every aircraft, and **zone transforms are
authored in the SAE frame** (`crates/game/src/aero.rs::entity_to_sae`). Without
this, AoA and lift come out in the wrong frame and nothing flies.

## Scope and roadmap

**Aircraft (implemented, first cut).** `aero.rs::build_ship_aero_layout` walks
the blueprint's `Wing` parts via `ShipBlueprint::wing_aero_panels` (shipyard,
which returns each panel's AC position + airfoil basis in the body frame) and
emits, per panel: a **base lifting zone** (cambered for the main wing, symmetric
for stabiliser/fin — provides stability + damping) plus a **control zone**
(elevator on the aft surface, ailerons L/R on the main wing, rudder on the fin).
A fuselage bluff-body drag zone is always present. `sync_flight_controls` maps
`GameInputIntent` pitch/roll/yaw into `ControlInputs`. Engine thrust stays
Thalos's nose-forward throttle (no `EngineZone`). **The airfoil / control
constants at the top of `aero.rs` are tuned by eye and need in-game iteration**
(elevator sign, control gain, static margin, stall, takeoff speed).

**Spacecraft (rockets, capsules).** A single bluff-body drag zone; reference area
is per-vehicle (`ShipStats::frontal_area_m2`, pushed via
`ShipParameters::reference_area_m2`), Cd a blunt-body constant.

**Planes from the construction editor (future).** When `docs/construction.md`'s
wing **Modules** exist, the same `wing_aero_panels` path generalises (control
surfaces become wing parameters); the zone generator is already Module-shaped.

## Debug view (F3)

`F3` toggles a game-wide overlay: **Avian collider wireframes** (every physics
body — ship, terrain patch, runway, EVA — via `PhysicsDebugPlugin`) plus the
`avian_fdm` **force / moment vectors** (lift, drag, side force, thrust, weight,
resultant, CG/AC markers, relative wind) via the FDM crate's `AircraftFdmDebugPlugin`
(`FdmGizmos`). The FDM debug renderer was made f64-compatible (its `debug-plugin`
feature no longer forces the avian3d f32 backend). Both start disabled; the force
arrow scale is set for airliner-magnitude forces in `aero.rs::init_debug_overlay`.

## Deferred

- **Aero in prediction / warp.** Bubble-only by choice; predicted trajectories do
  not yet account for drag (a reentry line would Kepler-coast). Wiring aero into
  the shared `ShipPropagator` is possible only after the "replace" exit moves the
  force math into `thalos_physics_canonical`.
- **Orbital decay under warp** (`Realistic` preset): a separate, simpler drag
  model that runs outside the bubble.
- **Reentry heating / heat-flux destruction** — a natural extension of the
  existing impact-destruction model (`ShipParameters::impact_tolerance_m_s`).
- **ISA-style temperature lapse** — the profile is isothermal-from-surface-T
  today (correct ρ and a constant speed-of-sound); a troposphere lapse rate
  would make temperature and Mach vary with altitude. Drops in behind
  `sample_at_altitude_m`.
- **Per-nose-shape drag coefficient** — Cd is a single blunt-body constant; it
  should vary by nose part (streamlined rocket vs blunt capsule).

## File map

- `crates/avian_fdm/**` — vendored LGPL force model. Thalos edits: the
  `manage_atmosphere` plugin flag (`src/plugin.rs`), the `AeroFrame` component
  (`src/components/aircraft.rs`) threaded through `kinematics.rs` /
  `aerodynamics/mod.rs`, and an f64-compatible `debug-plugin`
  (`src/debug_render/gizmos.rs`, `Cargo.toml`).
- `crates/shipyard/src/stats.rs` — `ShipBlueprint::wing_aero_panels` +
  `WingAeroPanel` (per-wing aerodynamic geometry).
- `crates/game/src/aero.rs` — `entity_to_sae`, `build_ship_aero_layout`
  (wing → lift/control zones), `attach_ship_aero`, `sync_flight_controls`,
  `init_debug_overlay`/`toggle_debug_overlay` (F3).
- `crates/world/src/atmosphere.rs` — `AtmosphereProfile` + `AtmosphereSample` +
  `TerrestrialAtmosphere::sample_at_altitude_m`.
- `crates/world/src/body.rs` — `BodyDefinition::surface_pressure_pa` /
  `surface_gravity_m_s2`; `crates/terrain/src/feature_compiler.rs` —
  `AtmosphereSpec::pressure_bar`.
- `crates/shipyard/src/stats.rs` — `ShipStats::frontal_area_m2`;
  `crates/physics_canonical/src/types.rs` — `ShipParameters::reference_area_m2` /
  `drag_coefficient`; `crates/game/src/ship_view.rs` — pushes them.
- `crates/game/src/aero.rs` — `GameAeroPlugin`, `sync_aero_environment`
  (atmosphere + co-rotation wind).
- `crates/game/src/local_physics.rs` — `craft_in_atmosphere` + the in-atmosphere
  `Full`-role trigger.
- `crates/game/src/bridge.rs` — atmosphere warp clamp in
  `enforce_warp_altitude_limits`.
- `crates/game/src/hud/atmo_panel.rs` — TAS / dynamic-pressure / Mach readout.

## Verification

- `cargo test -p thalos_world` — density-profile unit tests.
- `just game landing` / `just game final` — the ship should decelerate and reach
  a terminal velocity instead of a lunar-style freefall (A/B by toggling the
  body's `profile` / terrain `pressure_bar` / Kármán line).
- BRP: read `FlightState` on the ship for non-zero `dynamic_pressure_pa` /
  `airspeed_ms` during descent; read `AvianRole` to confirm in-atmosphere →
  `Full`; confirm warp pins to 1× below the Kármán line.

[`avian_fdm`]: https://github.com/viccuad/avian_fdm
