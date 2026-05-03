# Simulation architecture - target design

This document describes the target simulation architecture for Thalos:
orbital mechanics, local rigidbody gameplay, time warp, map rendering,
and the boundaries between them.

The design intentionally separates "truth" from presentation. The map
is a scaled analytical view. The real-space scene is a local gameplay
view. Neither is allowed to become the only authoritative simulation
state.

## Goals

- Keep long-horizon orbital behavior deterministic, testable, and
  independent of Bevy rendering.
- Support high time warp with coast, finite burns, attitude propagation,
  angular velocity, resource use, and simplified perturbing forces.
- Support local rigidbody gameplay for ships, docking, landed craft,
  collision, part joints, ground contact, debris, and nearby vehicles.
- Allow Bevy-native physics through Avian without forcing orbital
  prediction and time warp to run through the Bevy physics schedule.
- Use high-precision Bevy transforms for the real-space gameplay scene
  where they help, while keeping the map as an intentionally scaled
  rendering of analytical data.
- Make authority handoffs explicit: every craft has exactly one current
  simulation authority.
- Keep save/load, replay, prediction, and tests grounded in canonical
  state, not renderer transforms.
- Give players a small, stable world-mode choice instead of a matrix of
  physics knobs.

## Non-goals

- Running high-warp orbital prediction through Avian.
- Treating Bevy `Transform` or Avian `Position` as the only source of
  truth for orbital state.
- Sharing one transform hierarchy between real-space gameplay and the
  map view.
- Simulating the entire solar system as rigidbodies.
- Preserving detailed part-level rigidbody state through arbitrary high
  warp. High warp uses simplified aggregate dynamics.
- Converting an existing save between world presets.

## Core principle

Thalos should have three separate worlds:

1. **Orbital truth**: f64, deterministic, pure Rust. This owns epochs,
   body ephemerides, craft orbital state, flight plans, warp
   integration, event detection, and prediction.
2. **Real-space gameplay**: Bevy scene, high-precision transform
   support, Avian rigidbodies for the active local bubble. This owns
   contacts, joints, docking, landing, local collisions, and low-speed
   interaction.
3. **Map view**: scaled presentation. This reads snapshots and
   trajectories from orbital truth and draws a useful navigation view.
   It is not the real world and should not share render transforms with
   the local physics scene.

The map and real-space scene are clients of the simulation. They are not
the simulation.

## World presets

Each save slot chooses one world preset at creation time. The preset is
stored in the save header and is immutable for normal gameplay. Changing
presets invalidates body trajectories, craft trajectories, event
history, maneuver predictions, and sometimes the meaning of "stable"
orbits, so save conversion between presets is not a supported path.

```rust
pub enum WorldPreset {
    Classic,
    Realistic,
}

pub struct WorldPhysicsConfig {
    pub preset: WorldPreset,
    pub gravity_backend: GravityBackendConfig,
    pub provider_policy: ProviderPolicyRef,
    pub epoch: EpochConfig,
}

pub enum GravityBackendConfig {
    PatchedConics(PatchedConicsConfig),
    NBodyEphemeris(NBodyEphemerisConfig),
}

pub struct ProviderPolicyRef {
    pub id: String,
    pub version: u32,
}
```

### Classic

`Classic` is the KSP-like mode:

- gravity backend: `PatchedConics`
- body motion: analytical patched-conic rails
- force-free craft coast: analytical under the current conic segment
- finite burns and mass flow: supported
- attitude and angular velocity: supported
- orbital drag: disabled
- solar radiation pressure: disabled
- oblateness/J2: disabled

`Classic` exists for a clean, forgiving planning model. It should remain
stable and easy to reason about.

### Realistic

`Realistic` is the main Thalos mode:

- gravity backend: `NBodyEphemeris`
- body motion: offline N-body ephemeris stored as Chebyshev segments
- craft gravity: summed from ephemeris bodies
- finite burns and mass flow: supported
- attitude and angular velocity: propagated through coast and warp
- simplified orbital drag: supported under warp for low-orbit decay
- solar radiation pressure: target passive coast provider
- oblateness/J2: target passive coast provider for relevant bodies
- full atmospheric flight: clamps or exits high warp

`Realistic` is still a game mode, not a research-grade flight dynamics
tool. It is where realism is tuned against playability.

Internally, each preset expands to a provider policy. Provider policies
are composable implementation details used by runtime simulation,
prediction, and tests. Players choose `Classic` or `Realistic`; they do
not assemble force-provider menus. Saves store the exact provider policy
version. Future changes such as `RealisticPolicyV2` either require a
save migration or apply only to newly created worlds.

## Dependency assumptions

As of 2026-05-01:

- `big_space 0.12.0` targets Bevy `0.18` and provides `BigSpace`,
  `Grid`, `CellCoord`, and `FloatingOrigin` for high-precision Bevy
  transform hierarchies.
- `avian3d 0.6.1` targets Bevy `0.18`, has f64 support via feature
  flags, supports Parry f64 collision detection via `parry-f64`, and
  provides optional `enhanced-determinism`.
- [`particular`](https://github.com/Canleskis/particular) is the
  intended offline N-body interaction calculator for ephemeris baking.
  The ephemeris baker owns time integration, sampling, Chebyshev
  fitting, and asset versioning around it.

The architecture should not depend on exact minor APIs. Wrap both crates
behind local adapter modules so changes in either dependency do not
bleed through the rest of the game.

## Crate boundaries

Target crate split:

```text
crates/
  sim_core/
    Pure Rust simulation types, clocks, states, frames, force traits,
    torque traits, integrators, event detection, prediction contracts.

  ephemeris/
    Body trajectory providers: analytical Kepler, patched conics,
    baked Chebyshev N-body ephemerides, test fixtures.

  flight/
    Craft models, maneuver plans, finite burns, attitude controllers,
    resources, staging, aggregate mass/inertia.

  local_physics/
    Bevy + Avian bridge. Hydrates canonical craft state into
    rigidbodies, reads Avian state back into canonical state, manages
    local physics bubbles.

  real_space_rendering/
    Bevy rendering of the local gameplay scene. Owns big_space
    integration, cameras, local body meshes, ships, lighting.

  map_view/
    Bevy rendering of scaled map projections, orbit lines, maneuver
    widgets, ghost bodies, labels, and planning overlays.

  game/
    Orchestration, app states, input, UI, save/load glue, feature
    composition.

  tools/
    Offline ephemeris baker. Uses particular for N-body interactions,
    advances body states with the selected offline integrator, samples
    the result, and fits Chebyshev segments.
```

The names can be adjusted, but the dependency direction is mandatory:

```text
sim_core            <- ephemeris
sim_core            <- flight
sim_core            <- local_physics
sim_core            <- real_space_rendering
sim_core            <- map_view

ephemeris           <- flight
flight               <- game
local_physics        <- game
real_space_rendering <- game
map_view             <- game
```

`sim_core`, `ephemeris`, and most of `flight` must not depend on Bevy.
Use `glam` and `serde` directly where needed.

## Coordinate spaces

### Inertial system frame

The canonical orbital frame. Positions are f64 meters. Velocities are
f64 meters per second. Epochs are f64 seconds from a fixed simulation
epoch.

This is the only frame used for long-horizon prediction, save/load,
mission planning, and high-warp integration.

### Body-centered inertial frame

A frame centered on a body at a specific epoch, with axes aligned to the
system inertial frame. Useful for local orbital analysis, encounter
plots, and integrator conditioning.

### Body-fixed frame

A rotating frame attached to a body's surface. Landed craft, launchpads,
surface bases, and terrain patch colliders use this frame.

Body-fixed state is not "just a transform". It must include:

- `body_id`
- latitude/longitude/altitude or body-local Cartesian position
- body-fixed orientation
- optional surface-relative velocity for sliding/rolling cases
- epoch or rotation model version

### Craft body frame

The craft's local structural frame. Parts, thrust vectors, torque
application points, RCS blocks, reaction wheels, and docking ports are
defined here.

### Real-space grid frame

The Bevy gameplay scene uses `big_space` only for the real-space view.
The active camera or active craft is the floating origin. Bodies and
craft are placed into grid cells from canonical f64 positions.

This frame is a rendering and local-physics convenience, not a save-file
or prediction format.

### Map projection frame

The map view owns its own projection. It may use logarithmic distance,
patched local frames, focus-body-relative views, split scales, ghost
epochs, or icon substitutions. It should not reuse real-space transforms.

## Canonical data model

### Epoch and clock

```rust
pub struct Epoch(pub f64);      // seconds since simulation epoch
pub struct DurationS(pub f64);  // seconds

pub struct SimClock {
    pub current: Epoch,
    pub scale: f64,
    pub paused: bool,
    pub mode: TimeMode,
}

pub enum TimeMode {
    Realtime,
    PhysicsWarp,
    RailsWarp,
    PlanningPreview,
}
```

`TimeMode` is not just UI. It controls which authority modes are allowed
and which integrators may run.

### Body state

```rust
pub type BodyId = usize;

pub struct BodyState {
    pub id: BodyId,
    pub epoch: Epoch,
    pub position: DVec3,
    pub velocity: DVec3,
    pub orientation: DQuat,
    pub angular_velocity: DVec3,
    pub gm: f64,
    pub radius_m: f64,
}

pub trait BodyTrajectoryProvider: Send + Sync {
    fn body_count(&self) -> usize;
    fn state(&self, body: BodyId, epoch: Epoch) -> BodyState;
    fn states_into(&self, epoch: Epoch, out: &mut Vec<BodyState>);
}
```

Providers can be analytical, patched-conic, or baked ephemeris. Ship
simulation must only see this trait.

`Classic` saves use a `PatchedConics` provider. Body states are
evaluated analytically at runtime.

`Realistic` saves use an `NBodyEphemeris` provider. Body states are
evaluated from offline-computed Chebyshev segments:

```text
solar system definition
  -> initial body states
  -> offline N-body interactions via particular
  -> offline time integration and sampling
  -> Chebyshev segment fitting
  -> versioned ephemeris asset
  -> runtime BodyTrajectoryProvider
```

At runtime, both modes are rails for body motion. The difference is how
the body rails were produced and evaluated.

### Navigation context

Navigation context is separate from physics authority. It answers
"which body is this trajectory meaningfully near or organized around?"
for map views, maneuver defaults, labels, ghost focus, and encounter
lists.

In `Classic`, navigation context can use patched-conic SOI windows
because the active gravity provider already uses those boundaries.

In `Realistic`, SOI has no force-switching meaning. The navigation
layer computes body context scores over the predicted trajectory:

```rust
pub struct NavigationContextSample {
    pub epoch: Epoch,
    pub primary: Option<BodyId>,
    pub confidence: f64,
    pub scores: Vec<BodyContextScore>,
}

pub struct BodyContextScore {
    pub body: BodyId,
    pub gravity_share: f64,
    pub range_m: f64,
    pub hill_fraction: f64,
    pub relative_energy_j_kg: f64,
}
```

The `primary` body is a navigation default, not a physics authority. It
may be omitted when several bodies have comparable scores. The UI should
prefer explicit user focus or selection over silently rebinding context
when confidence is low.

### Navigation encounters

Navigation encounters are annotations extracted from context samples and
trajectory events. They replace SOI-window UX in `Realistic`.

```rust
pub struct NavigationEncounter {
    pub body: BodyId,
    pub kind: NavigationEncounterKind,
    pub start: Epoch,
    pub end: Epoch,
    pub focus_epoch: Epoch,
    pub closest_approach: Option<ClosestApproach>,
    pub confidence: f64,
}

pub enum NavigationEncounterKind {
    Flyby,
    HillDwell,
    TemporaryCapture,
    BoundOrbit,
    AtmosphereEntry,
    Impact,
}
```

Classifier inputs should include:

- closest-approach distance and epoch
- relative velocity
- time spent below a body-specific encounter radius
- time spent inside a fraction or multiple of Hill radius
- acceleration or gravity-share dominance
- body-relative specific orbital energy
- body-relative periapsis/apoapsis cycles where available
- geometric atmosphere and impact thresholds

No single metric is sufficient. Use hysteresis and minimum dwell times
so encounters do not flicker in multi-body regions.

`Classic` SOI entries/exits should be adapted into
`NavigationEncounter` intervals so map rendering can consume one
encounter model. `Realistic` ghost focus pins to the encounter body at
`focus_epoch` -- usually closest approach, interval midpoint, impact, or
atmosphere entry -- with the body state evaluated from the active
provider at that epoch.

### Craft state

The canonical craft state is aggregate by default. Detailed part-level
rigidbody state exists only while local physics authority is active, or
when a save file explicitly stores a landed/docked local scene.

```rust
pub type CraftId = u64;

pub struct TranslationalState {
    pub position: DVec3,
    pub velocity: DVec3,
}

pub struct AttitudeState {
    pub orientation: DQuat,
    pub angular_velocity: DVec3,
}

pub struct MassState {
    pub wet_mass_kg: f64,
    pub dry_mass_kg: f64,
    pub inertia_body_kg_m2: DMat3,
    pub center_of_mass_body_m: DVec3,
}

pub struct CraftState {
    pub id: CraftId,
    pub epoch: Epoch,
    pub translation: TranslationalState,
    pub attitude: AttitudeState,
    pub mass: MassState,
    pub resources: ResourceState,
    pub authority: AuthorityMode,
}
```

### Authority mode

Every craft has exactly one current authority.

```rust
pub enum AuthorityMode {
    OnRails {
        trajectory: TrajectoryId,
    },
    WarpIntegrated {
        integrator: WarpIntegratorId,
    },
    LocalRigidBody {
        bubble: LocalBubbleId,
        root_entity: EntityRef,
    },
    BodyFixed {
        body: BodyId,
        pose: BodyFixedPose,
    },
    Docked {
        assembly: AssemblyId,
        port: DockingPortId,
    },
}
```

Authority meaning:

- `OnRails`: no active continuous forces. State is evaluated from a
  cached or analytical trajectory.
- `WarpIntegrated`: pure deterministic integration owns translation,
  attitude, resources, and events. Used for burns under warp,
  atmosphere, perturbations, spin propagation, and non-contact dynamics.
- `LocalRigidBody`: Avian owns detailed local rigidbody motion for the
  active bubble. Canonical state is sampled from Avian after physics
  steps.
- `BodyFixed`: craft is anchored to a rotating body frame. Used for
  stable landed or parked surface objects when local physics is asleep.
- `Docked`: craft is part of a larger assembly. The assembly owns the
  aggregate state.

Authority changes are events and must be logged.

## Simulation regimes

### Rails coast

Use for high warp when no continuous forces need integration. This is
primarily the `Classic` force-free coast path, where craft motion can be
an analytical patched-conic segment.

Expected properties:

- very cheap to evaluate
- deterministic
- easy to extend far ahead
- suitable for map and planning

In `Classic`, force-free coast should use analytical conic propagation
where possible.

### Passive perturbed coast

Use in `Realistic` when the craft is coasting without player-commanded
burns or local contacts, but passive effects still accumulate over long
durations:

- summed gravity from Chebyshev ephemeris bodies
- torque-free attitude and angular velocity propagation
- simplified orbital drag
- solar radiation pressure
- oblateness/J2 or other body-shape perturbations
- resource-neutral passive effects that do not require local physics

This is the normal high-warp coast path for `Realistic`. It runs in pure
Rust, can use larger/adaptive steps, and may cache the resulting
trajectory for map rendering and prediction. These effects are most
visible over long coasts, so this path should favor long-horizon
accuracy and deterministic event detection over frame-by-frame visual
detail.

### Warp integration

Use when active forces, active torques, or passive perturbations must
continue under time warp:

- continuous thrust
- RCS translation or rotation
- reaction wheels
- simplified orbital drag
- full atmospheric drag in low-warp aerodynamic flight
- lift approximations in low-warp aerodynamic flight
- solar radiation pressure
- J2 or other gravity perturbations
- mass flow
- engine cutoff events
- angular velocity propagation

This runs in pure Rust, outside Avian.

`Classic` only enters warp integration for active finite forces such as
burns, mass flow, and attitude changes. `Realistic` may also use warp
integration for passive perturbed coast: orbital drag, radiation
pressure, oblateness, angular velocity propagation, and other enabled
perturbing providers.

### Local rigidbody

Use for contact and detailed local interactions:

- docking
- landing and takeoff near terrain
- collision with debris
- part joints
- wheel/leg contact
- surface sliding and bouncing
- nearby vehicle interaction
- low-altitude manual flight where rigidbody contacts are possible

This runs through Avian at fixed timestep. It is limited to low or
moderate time scale. High warp exits local rigidbody authority or pauses
until the craft can safely collapse to aggregate dynamics.

### Body-fixed sleep

Use for stable landed craft and surface infrastructure. The craft stores
a body-fixed pose and does not need live Avian rigidbodies until:

- the player focuses it
- a nearby active bubble includes it
- an external event disturbs it
- launch, decouple, docking, or collision occurs

## Force and torque model

Force and torque providers are pure simulation contracts. They should be
usable by warp integration and prediction workers. Avian may use related
adapters, but Avian is not the primary force API.

Provider policies are internal, versioned compositions of these
providers. A world preset selects a provider policy, but the integration
loop only sees provider lists and tolerances. This keeps the system
composable without exposing a large physics-options matrix to players.

```rust
pub struct ProviderPolicy {
    pub id: String,
    pub version: u32,
    pub force_providers: Vec<ForceProviderSpec>,
    pub torque_providers: Vec<TorqueProviderSpec>,
    pub event_detectors: Vec<EventDetectorSpec>,
    pub integration: IntegrationPolicy,
    pub warp_limits: WarpLimitPolicy,
}
```

Expected initial policies:

- `ClassicPolicyV1`: patched-conic gravity, finite thrust, mass flow,
  attitude/spin, no orbital drag, no radiation pressure, no J2.
- `RealisticPolicyV1`: N-body ephemeris gravity, finite thrust, mass
  flow, attitude/spin, simplified orbital drag, and passive perturbed
  coast hooks. Later versions may add solar radiation pressure,
  oblateness/J2, and other perturbations.

```rust
pub struct ForceContext<'a> {
    pub epoch: Epoch,
    pub bodies: &'a [BodyState],
    pub craft: &'a CraftState,
    pub environment: &'a EnvironmentCache,
}

pub struct ForceSample {
    pub force_inertial_n: DVec3,
    pub application_point_body_m: Option<DVec3>,
    pub mass_flow_kg_s: f64,
}

pub trait ForceProvider: Send + Sync {
    fn sample(&self, ctx: &ForceContext) -> ForceSample;
}

pub struct TorqueSample {
    pub torque_body_n_m: DVec3,
    pub wheel_momentum_delta: DVec3,
}

pub trait TorqueProvider: Send + Sync {
    fn sample(&self, ctx: &ForceContext) -> TorqueSample;
}
```

Providers:

- `PointMassGravity`: summed body gravity.
- `CentralGravity`: single-body gravity for cheap rails or local
  approximations.
- `FiniteThrust`: throttle, Isp, mass flow, gimbal, thrust point.
- `RcsForces`: translational and rotational RCS.
- `ReactionWheelTorque`: stores wheel momentum and saturation.
- `AtmosphericDrag`: sampled from body atmosphere model.
- `OrbitalDragApproximation`: simplified drag used for low-orbit decay
  under warp.
- `LiftApproximation`: optional future aerodynamic force.
- `SolarRadiationPressure`: optional sail and panel forces.
- `J2Perturbation`: optional body oblateness.

The provider policy, simulation regime, craft capabilities, and
environment decide which providers are active. For example, high warp in
`Realistic` may use gravity plus finite thrust plus orbital drag, while
local rigidbody mode uses Avian contacts plus custom gravity and engine
forces.

### Atmospheric regimes

Atmosphere is split into two regimes:

- **Orbital drag regime**: simplified drag for low-orbit decay. This is
  allowed under warp in `Realistic` because it preserves the important
  strategic consequence of low atmospheric orbits without simulating
  full aerodynamic flight.
- **Aerodynamic flight regime**: full or near-full aero, lift, heating,
  control surfaces, terrain proximity, and contact risk. High warp is
  disallowed here, KSP-style. The game clamps or exits warp when
  dynamic pressure, density, altitude, heating, or contact rules say the
  craft has entered this regime.

The exact entry rule is policy data, not hardcoded into the integrator.
Initial implementation can use altitude and density thresholds; later
versions can add dynamic pressure and heating.

## Maneuver frames

Maneuver reference frames are navigation/editing frames, not SOI
assertions. A maneuver stores a frame-selection tag. Prediction
preserves that tag and re-evaluates the actual basis from the active
body provider and predicted craft state at the maneuver epoch.

```rust
pub enum ManeuverFrame {
    Inertial,
    BodyRelative { body: BodyId },
    NavigationContext,
    VesselLocal,
}

pub enum BurnFrameBehavior {
    FixedAtBurnStart,
    TrackFrameDuringBurn,
}

pub struct ManeuverNode {
    pub epoch: Epoch,
    pub frame: ManeuverFrame,
    pub components: ManeuverComponents,
    pub burn_behavior: BurnFrameBehavior,
}
```

For prograde/normal/radial editing in `BodyRelative { body }`, the
propagator evaluates:

1. predicted craft state at the maneuver epoch
2. reference body state at the maneuver epoch
3. relative position and velocity
4. radial/prograde/normal basis from that relative state
5. inertial thrust or delta-v vector from the stored maneuver components

The frame tag is frozen. The basis is re-evaluated. A node created as
Mira-relative remains Mira-relative until the user changes it; it does
not silently rebind because another body becomes dominant later.

Default frame selection is UI policy:

- In `Classic`, default to the current patched-conic SOI body.
- In `Realistic`, prefer the selected or focused body, then a
  high-confidence navigation-context primary body, then the strongest
  gravity contributor. If context is ambiguous, the UI should make the
  frame explicit rather than hiding the ambiguity.

Long finite burns must specify whether the burn direction is fixed at
burn start or tracks the selected frame through the burn. Short impulse
nodes can usually use `FixedAtBurnStart`; low-thrust and long-duration
burns often need `TrackFrameDuringBurn`.

## Warp integration

Warp integration owns both translation and attitude.

```rust
pub struct WarpState {
    pub epoch: Epoch,
    pub translation: TranslationalState,
    pub attitude: AttitudeState,
    pub mass: MassState,
    pub resources: ResourceState,
}

pub trait WarpIntegrator {
    fn step(
        &mut self,
        state: &mut WarpState,
        target_epoch: Epoch,
        forces: &[Box<dyn ForceProvider>],
        torques: &[Box<dyn TorqueProvider>],
        events: &mut EventSink,
    ) -> StepReport;
}
```

Integrator requirements:

- deterministic with fixed inputs
- event detection for impact, atmosphere boundary, SOI metadata changes,
  burn start/end, resource depletion, periapsis/apoapsis, and user
  alarms
- adaptive stepping allowed, but accepted steps and event roots must be
  deterministic
- no dependency on frame delta
- usable by prediction jobs and runtime warp

Initial recommendation:

- Use a high-order adaptive Runge-Kutta integrator for force-rich warp
  because thrust, drag, resource depletion, and attitude control make
  symplectic purity less important.
- Keep a simpler symplectic or Kepler stepper available for pure coast
  trajectories.
- Add integrator error budgets per regime: committed maneuver
  prediction can spend more CPU than background debris.

### Attitude under warp

Attitude state must persist through warp:

```rust
pub struct AttitudeState {
    pub orientation: DQuat,       // body to inertial
    pub angular_velocity: DVec3,  // inertial or body, choose one and document it
}
```

The preferred convention is angular velocity in body coordinates for
torque integration and control laws, with helper methods for inertial
conversion.

Attitude integration must support:

- torque-free spin preservation
- reaction wheel torques
- RCS torque
- engine gimbal torque
- SAS/autopilot target tracking
- time-warp stability for long coast

Autopilot is a controller that emits torque or desired torque. It is not
a magical transform setter.

## Local physics with Avian

Avian should be used for local rigidbody dynamics, not for long-horizon
orbital truth.

Recommended configuration:

- Use `avian3d` with f64 physics if performance allows.
- Enable `parry-f64` when using f64 colliders.
- Consider `enhanced-determinism` for replay, multiplayer, or
  cross-platform exactness.
- Disable or override Avian's global uniform gravity. Gravity is a field
  sampled from Thalos body state.
- Keep Avian in a local bubble centered on active gameplay.

The local physics bridge owns conversion in both directions:

```rust
pub trait LocalPhysicsBridge {
    fn hydrate(
        &mut self,
        craft: &CraftState,
        assembly: &AssemblyDefinition,
        bubble: &LocalBubble,
    ) -> LocalRigidBodyHandle;

    fn sample_canonical_state(
        &self,
        handle: LocalRigidBodyHandle,
        epoch: Epoch,
    ) -> CraftState;

    fn collapse(
        &mut self,
        handle: LocalRigidBodyHandle,
        policy: CollapsePolicy,
    ) -> CollapseResult;
}
```

### Hydration

When entering local rigidbody mode:

1. Choose a local bubble origin and reference frame.
2. Convert canonical inertial position, velocity, orientation, and
   angular velocity into bubble-local state.
3. Spawn part rigidbodies or a reduced rigidbody representation.
4. Spawn joints and constraints.
5. Spawn local terrain/body colliders as needed.
6. Initialize Avian velocities from canonical state.
7. Mark craft authority as `LocalRigidBody`.

Hydration must preserve:

- center-of-mass position and velocity
- orientation
- angular velocity
- total mass
- fuel/resource state
- docking topology

### Sampling back

After Avian fixed steps, the bridge samples:

- aggregate center of mass
- aggregate linear momentum
- aggregate angular momentum
- body orientation
- angular velocity
- contact state
- broken joints or topology changes
- resource changes from engines/RCS

This updates the canonical `CraftState`. The renderer may use Avian
components directly for smooth visuals, but save/load and map data still
come from canonical state.

### Collapse

When leaving local rigidbody mode:

1. Read aggregate state from Avian.
2. If the craft is in stable contact with a body, collapse to
   `BodyFixed`.
3. If the craft is docked to another assembly, collapse to `Docked`.
4. If the craft is free-flying with active forces, collapse to
   `WarpIntegrated`.
5. If the craft is free-flying without active forces, build or update a
   trajectory and collapse to `OnRails`.
6. Despawn or sleep Avian entities not needed by the active bubble.

Collapse must be round-trip tested. Hydrate and immediately collapse
should reproduce the input aggregate state within a tight tolerance.

### Part-level state

During local physics, a craft can be an assembly of rigidbodies. During
warp, it should normally be an aggregate rigidbody. Part-level detail is
preserved as structural data plus resource state, not as hundreds of
active rigidbody states.

Exceptions:

- separated debris becomes independent craft or debris entities
- docked assemblies may store multiple rigid groups
- landed bases may store body-fixed sub-poses for modules

## big_space usage

Use `big_space` for the real-space scene only.

Recommended pattern:

- One `BigSpace` for the active real-space scene.
- Active camera or active craft has `FloatingOrigin`.
- Bodies, craft, debris, and local markers are positioned by converting
  canonical f64 inertial positions into `CellCoord + Transform`.
- For absolute ephemeris placement, compute grid cell and local
  transform directly instead of assigning huge `Transform.translation`
  values.
- Keep normal Bevy UI and map view outside the high-precision hierarchy.

Nested grids may be useful:

- system grid: inertial positions
- body grid: body-centered orbital space
- body-fixed grid: rotating surface frame
- craft grid: large assemblies or interiors

Do not add nested grids until a use case requires them. Each extra frame
adds mental overhead and handoff risk.

## Map view

The map view is a projection of orbital truth. It should not share
entities, transforms, or cameras with the real-space scene.

Map view inputs:

- body states at selected epochs
- craft canonical states
- trajectories and maneuver plans
- navigation context samples
- navigation encounters and event metadata
- selected focus body, encounter, or ghost epoch

Map view outputs:

- meshes, gizmos, billboards, labels, handles, and interaction targets
- selected maneuvers and edit deltas
- focus/selection commands

Map projection should be explicit:

```rust
pub trait MapProjection {
    fn project_body(&self, body: &BodyState, ctx: &MapContext) -> Vec3;
    fn project_point(&self, point_inertial_m: DVec3, epoch: Epoch, ctx: &MapContext) -> Vec3;
    fn unproject_drag(&self, drag: DragInput, ctx: &MapContext) -> MapEdit;
}
```

Useful projection modes:

- focused-body local inertial
- system barycentric scaled
- logarithmic distance for outer-system navigation
- encounter ghost frame
- surface/body-fixed map for landed craft

The map can draw simplified or fake-scale bodies because it is not the
collision world.

Map code should consume `NavigationEncounter` intervals, not raw SOI
events. In `Classic`, SOI windows are converted into navigation
encounters. In `Realistic`, closest approaches, Hill dwell intervals,
temporary captures, atmosphere entries, and impacts produce navigation
encounters directly.

## Prediction and flight planning

Prediction must use the same pure simulation logic as warp integration.
It must not depend on Avian.

Prediction inputs:

- initial canonical craft state
- body trajectory provider
- maneuver sequence
- provider policy selected by the world preset
- resource state
- attitude control mode
- event rules

Prediction outputs:

- sampled trajectory
- burn arcs
- navigation context samples
- navigation encounters
- event list
- closest approaches
- impact/landing warnings
- resource estimates
- attitude/spin samples where relevant

Prediction should use the same provider policy as runtime simulation.
Interactive preview may use looser tolerances, shorter horizons, or
lower sample density, but it must not silently change physics semantics.
Vacuum trajectory prediction should be as accurate as practical. Future
atmospheric prediction should use the same orbital-drag and low-warp
aero providers as runtime, subject to the same warp limits.

The map renders prediction outputs. It does not recompute physics.

## Handoff rules

### OnRails to WarpIntegrated

Trigger when:

- a finite burn starts
- attitude control must change angular state
- drag or atmosphere becomes relevant
- simplified perturbations are enabled
- user requests physics warp with active forces

Procedure:

1. Evaluate trajectory at current epoch.
2. Initialize `WarpState`.
3. Attach selected force and torque providers.
4. Change authority to `WarpIntegrated`.

### WarpIntegrated to OnRails

Trigger when:

- all continuous forces and torques are inactive
- craft is free-flying
- state is outside local contact bubble
- trajectory can be cheaply regenerated

Procedure:

1. Generate a new trajectory from final warp state.
2. Store event history and residual spin state.
3. Change authority to `OnRails`.

If the craft has nonzero angular velocity, the trajectory may be rails
for translation while attitude remains analytically or numerically
propagated. Do not discard spin.

### Any free-flight mode to LocalRigidBody

Trigger when:

- player enters local/manual scene near craft
- collision/contact is possible soon
- docking or landing systems are armed
- another active local object is nearby
- time scale drops below local physics threshold

Procedure:

1. Evaluate canonical state at current epoch.
2. Create or reuse local bubble.
3. Hydrate Avian entities.
4. Change authority to `LocalRigidBody`.

### LocalRigidBody to BodyFixed

Trigger when:

- craft is landed and stable
- relative velocity is below threshold
- contact normals and support points are stable
- no active forces require rigidbody simulation

Procedure:

1. Compute body-fixed pose from sampled Avian state.
2. Store landed contact metadata.
3. Despawn or sleep rigidbodies.
4. Change authority to `BodyFixed`.

### BodyFixed to LocalRigidBody

Trigger when:

- launch starts
- craft is selected for local control
- local bubble enters range
- external collision or force wakes it

Procedure:

1. Evaluate body pose and rotation at current epoch.
2. Convert body-fixed pose to inertial craft state.
3. Hydrate local rigidbodies.
4. Change authority to `LocalRigidBody`.

## Event model

Events are part of simulation output, not UI side effects.

```rust
pub enum SimEvent {
    AuthorityChanged { craft: CraftId, from: AuthorityMode, to: AuthorityMode },
    BurnStarted { craft: CraftId, burn: BurnId },
    BurnEnded { craft: CraftId, burn: BurnId, reason: BurnEndReason },
    ResourceDepleted { craft: CraftId, resource: ResourceId },
    NavigationEncounterStarted { craft: CraftId, encounter: NavigationEncounterId },
    NavigationEncounterEnded { craft: CraftId, encounter: NavigationEncounterId },
    ContactPredicted { craft: CraftId, body: BodyId, epoch: Epoch },
    ContactStarted { craft: CraftId, other: ContactTarget },
    ContactEnded { craft: CraftId, other: ContactTarget },
    Impact { craft: CraftId, body: BodyId, epoch: Epoch, speed_m_s: f64 },
    SoiMetadataChanged { craft: CraftId, old: BodyId, new: BodyId },
    Docked { a: CraftId, b: CraftId },
    Undocked { assembly: AssemblyId, child: CraftId },
}
```

SOI is metadata unless the active trajectory provider explicitly uses a
patched-conic approximation. The UI can still show SOI changes, but
force calculations should not depend on SOI in the target architecture.
In `Realistic`, navigation encounters are the player-facing replacement
for SOI windows.

## Scheduling

Target schedule shape in Bevy:

```text
Input
  collect controls, maneuver edits, view commands

SimClock
  advance deterministic simulation epoch

Authority
  decide authority transitions

Ephemeris
  cache body states for current epoch and prediction jobs

WarpIntegration
  step pure Rust warp-integrated craft

LocalPhysicsPrepare
  apply gravity/thrust/control inputs to Avian entities

Avian FixedPostUpdate
  local rigidbody simulation

LocalPhysicsReadback
  sample Avian aggregate state into canonical craft state

Prediction
  launch/merge async prediction jobs

PresentationSync
  update real-space scene from canonical/local state
  update map scene from trajectories and snapshots

Camera
  update active cameras after presentation transforms
```

The exact Bevy schedules can change, but the data flow should remain:
input -> simulation -> physics readback -> presentation -> camera.

## Save/load

Save canonical simulation state, not renderer state.

Save:

- world preset (`Classic` or `Realistic`)
- gravity backend config and version
- provider policy ID and version
- patched-conics config hash for `Classic`
- N-body Chebyshev ephemeris asset ID/hash for `Realistic`
- simulation epoch and clock settings
- body provider identity and version
- craft canonical states
- authority mode for each craft
- maneuver plans
- resource state
- assembly topology
- landed body-fixed poses
- docked assembly graph
- active warp integrator version for warp-integrated craft
- event log checkpoints if needed for replay

Do not save as authoritative:

- Bevy entity IDs
- Bevy `Transform`s
- Avian internal broadphase state
- map-view projected positions
- camera-relative render positions

Local rigidbody scenes may need a supplementary save block for active
part-level state. That block is still converted back into canonical
state on load before gameplay resumes.

## Determinism requirements

- Pure simulation code uses f64 and deterministic iteration order.
- Prediction jobs receive immutable snapshots.
- Event root finding uses fixed tolerances.
- Floating-point tolerances are documented per subsystem.
- Avian local physics is treated as local gameplay authority, not as
  the source for long-term replay unless enhanced determinism is enabled
  and tested.
- Randomness uses explicit seeds.
- Saves include provider versions so ephemeris or physics upgrades can
  migrate or invalidate old predictions.

## Testing strategy

Unit tests:

- save header rejects world-preset changes
- frame conversions round-trip
- body provider state continuity
- maneuver frame tag is preserved while basis is re-evaluated
- force provider outputs
- torque provider outputs
- navigation context scoring and hysteresis
- attitude integration preserves torque-free angular momentum
- finite burn mass flow and cutoff
- event root finding

Golden tests:

- `ClassicPolicyV1` patched-conic coast
- `RealisticPolicyV1` Chebyshev body evaluation
- coast trajectory over fixed epochs
- finite burn under warp
- drag entry scenario
- spin through high warp
- maneuver prediction vs runtime warp integration

Handoff tests:

- `OnRails -> WarpIntegrated -> OnRails` preserves state within
  tolerance
- `WarpIntegrated -> LocalRigidBody -> WarpIntegrated` preserves center
  of mass, velocity, orientation, and angular velocity
- `BodyFixed -> LocalRigidBody -> BodyFixed` preserves landed pose
- hydrate then immediately collapse is nearly identity

Map tests:

- map projection never mutates simulation state
- ghost encounter frame remains coincident with navigation encounter
  samples
- map focus changes do not affect real-space authority
- `Classic` SOI windows adapt into navigation encounters
- `Realistic` closest approach and capture intervals produce navigation
  encounters

Local physics tests:

- local gravity points toward the selected body
- simple landing contact stabilizes and collapses to `BodyFixed`
- docking creates an assembly and updates aggregate mass/inertia
- decoupling creates independent craft states

## Implementation plan

### Phase 1: Canonical state and authority shell

- Introduce canonical `WorldPreset`, `WorldPhysicsConfig`,
  `CraftState`, `AuthorityMode`, `Epoch`, and `BodyTrajectoryProvider`
  types.
- Add an authority resource that can describe current craft mode even if
  existing systems still perform the work.
- Add tests for frame conversion and authority transition bookkeeping.

### Phase 2: Decouple map view

- Move map rendering behind explicit snapshot/trajectory inputs.
- Stop sharing real-space entity transforms with map bodies.
- Introduce `MapProjection`.
- Keep current visuals working while making the map a presentation-only
  client.

### Phase 3: Real-space scene with big_space

- Add a feature-gated prototype real-space scene using `big_space`.
- Convert body and craft canonical positions into grid cell plus local
  transform.
- Keep UI and map outside the `BigSpace`.
- Validate camera precision near planets, moons, and interplanetary
  distances.

### Phase 4: Avian local bubble

- Add Avian behind a local adapter crate/module.
- Create a single-craft local bubble with custom gravity and no map
  dependency.
- Implement hydrate, sample, and collapse for an aggregate rigidbody.
- Add terrain patch collider or simple body collider for contact tests.

### Phase 5: Part-level local physics

- Hydrate ship assemblies into multiple rigidbodies and joints.
- Read aggregate mass, momentum, and angular momentum back from Avian.
- Handle docking, decoupling, broken joints, and debris creation.

### Phase 6: Warp force integration

- Add pure Rust force/torque providers and internal provider policies.
- Integrate finite burns, mass flow, passive perturbations, and attitude
  under warp.
- Preserve angular velocity through coast and warp.
- Use the same integrator for runtime warp and prediction jobs.
- Add `ClassicPolicyV1` and `RealisticPolicyV1`.

### Phase 6.5: Navigation contexts and maneuver frames

- Add `NavigationContextSample` and `NavigationEncounter`.
- Adapt `Classic` SOI windows into navigation encounters.
- Add `Realistic` encounter classification from closest approach, Hill
  dwell, relative energy, and geometric atmosphere/impact thresholds.
- Replace maneuver `reference_body` semantics with `ManeuverFrame`.
- Preserve maneuver frame tags through prediction while re-evaluating
  bases from active provider state.

### Phase 7: Full handoff integration

- Implement authority transitions between rails, warp integration,
  local rigidbody, body-fixed, and docked states.
- Add event logs and handoff tests.
- Make time-scale changes drive authority policy.

### Phase 8: Retire old coupling

- Remove render-origin assumptions from simulation state.
- Keep any camera-relative or scaled-origin logic inside presentation
  crates only.
- Ensure save/load contains no Bevy transform authority.

## Open decisions

- Offline N-body ephemeris baker details: integrator family, sampling
  cadence, Chebyshev segment length, polynomial degree, and error
  budgets.
- Grid cell size for `big_space` real-space scene.
- Whether Avian f64 performance is acceptable for the active bubble on
  target hardware.
- Maximum time scale allowed while local rigidbody authority remains
  active.
- Integrator choice and tolerances for force-rich warp.
- Exact atmospheric regime boundaries: altitude, density, dynamic
  pressure, heating, terrain proximity, or a combined rule.
- Exact navigation encounter classifier thresholds and hysteresis.
- Default maneuver-frame selection order when `Realistic` navigation
  context is ambiguous.
- Whether replay needs cross-platform bit identity or same-machine
  deterministic consistency is enough.

## Architectural invariants

- One canonical state per craft.
- One authority mode per craft.
- The map does not own physical truth.
- Real-space transforms do not own orbital truth.
- Avian owns contact-rich local motion only while authority says it
  does.
- Time warp and prediction use pure deterministic simulation.
- Navigation context is a UX layer, not a physics boundary.
- Maneuver frame tags are preserved; frame bases are re-evaluated from
  active provider state.
- Handoffs are explicit, logged, and tested.
- Rendering is allowed to scale, project, and simplify. Simulation is
  not.
