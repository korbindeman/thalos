# New Physics — N-body Ship Propagation on Rails

Replaces the patched-conics ship propagator with a numerical N-body
integrator that treats the ship as a massless test particle under summed
gravity from all bodies. Bodies stay on Keplerian rails for now;
the trait surface is designed so a baked N-body ephemeris can drop in
later.

The architectural pivot from today's code: **everything is rails.** Both
bodies and ships expose the same `Trajectory` interface that answers
`position(t)` / `state_vector(t)`. The simulation clock is a query
parameter — there is no live integration loop. Ships are propagated
forward in async workers; the rendered "where the ship is now" is just
`trajectory.position(sim_time.current)`.

This buys uniform behaviour at any time-warp factor (the main thread
does the same work at 1× and 10⁶×), kills the live-vs-predicted
divergence problem by construction, and makes parallel ships trivial.

## 1. Goals

- N-body gravity for ships under all bodies in the system. Lagrange
  points, halo orbits, weak-stability transfers, secular perturbations
  emerge naturally — no patched-conics seams.
- Bodies on Keplerian rails today. Same trait surface accepts a baked
  ephemeris later without touching ship propagation or rendering.
- Time-warp by walking through a precomputed prediction. Async worker
  keeps the cache extended ahead of `sim_time.current`. Unbounded warp
  factor with bounded main-thread cost.
- Parallel ships sharing one timeline. One actively edited; many flying
  on rails simultaneously (booster landings, auto-flown missions). All
  predicted independently.
- Realistic burn model. Finite-time burns under thrust force with mass
  flow and dry-mass cutoff. Δv-based UI input; duration derived from
  the rocket equation. Burn model behind a trait so high-fidelity
  variants (throttle profiles, gimbal, atmospheric Isp) slot in later.
- Flight plan horizons hours to years. Drag-edit responsiveness in the
  100 ms range for typical local maneuvers.
- Maneuver editing recomputes only the affected suffix of the flight
  plan — closed-form-equivalent latency, in N-body land — by detecting
  timeline divergence and restarting integration from the last common
  event.
- SOI tracking as informational metadata for rendering and UI
  hierarchy. Doesn't affect physics.
- Determinism: same initial state + flight plan ⇒ bit-identical
  trajectory across runs.
- Pure Rust, no Bevy in `thalos_physics`.

## 2. Non-goals (initial scope)

- N-body integration of bodies. Deferred behind the same trait. No
  offline ephemeris generation pipeline, no Chebyshev/REBOUND tooling.
- Backward time propagation. Forward only. (Bodies on Keplerian rails
  are time-symmetric for free; the cached trajectory cache is not.)
- User-selectable integrator. Single hardcoded default
  (Dormand-Prince 8(7)). Picker can be added later if needed.
- Body culling. None initially. Adaptive integrator handles cost by
  growing step in smooth regions. Revisit if profiling shows it's
  needed.
- Ship-on-ship gravity. Ships are massless point particles in the
  gravity model.
- General relativity, J₂ oblateness, solar radiation pressure,
  atmospheric drag.
- High-fidelity burn variations beyond constant-thrust. Trait
  extensibility hook only.
- Trajectory branching / alternate timelines. Parallel ships share the
  same timeline; "fork" is a UX concept (control focus shift), not a
  physics concept.
- Cache pruning policy fully fleshed out. Initial implementation prunes
  conservatively behind `sim_time.current`; tune in profiling.

## 3. Architectural invariants

- One simulation clock (`SimulationTime::current`). All entities
  synchronise their position from `trajectory.position(current)`.
- No live integration. Ships are advanced by the async prediction
  worker writing into the cache; the main thread reads only.
- Bodies and ships expose the same `BoundedTrajectory + EvaluateTrajectory`
  interface. Rendering, picking, encounter analysis don't know which
  is which.
- Frames are composed at evaluation time. `RelativeTrajectory<T1, T2>`
  is a wrapper that returns `t1.position(t) − t2.position(t)`. No
  per-sample anchor is stored. Plot rendering chooses its anchor when
  the plot is built.
- The propagator and the cached trajectory are decoupled. The async
  worker holds a propagator; the ECS holds the trajectory. The two
  meet only when a snapshot is shipped from worker to main.
- One propagator per ship. No global propagator iterating ships
  serially — each ship's prediction is its own task.
- SOI information is post-hoc metadata, never input to physics.

## 4. System overview

```
                ┌─────────────────────────────────────────────┐
                │            thalos_physics (lib)             │
                │                                             │
  Per body:     │  KeplerianTrajectory ──┐                    │
  Keplerian     │  (analytic, infinite   │                    │
  rails         │   bounds)              │                    │
                │                        ▼                    │
  Per ship:     │                  dyn EvaluateTrajectory ◄───┼─── ShipPropagator
  CubicHermite  │  CubicHermiteSpline ◄──┤                    │    (sums gravity
  cache         │  (cached samples)      │                    │     from all
                │                        │                    │     bodies)
                │                        │                    │
                │  RelativeTrajectory<T1, T2> ─── composes at evaluation time
                └─────────────────────────────────────────────┘
                                  ▲
                                  │ trait surface
                                  ▼
                ┌─────────────────────────────────────────────┐
                │            thalos_game (Bevy)               │
                │                                             │
                │  SimulationTime::current ──► sync_position  │
                │                              (every entity) │
                │                                             │
                │  PredictionPropagator (Component) ─────┐    │
                │       │                                │    │
                │       ▼                                ▼    │
                │  AsyncComputeTask ──────► async_channel ──► merge into Trajectory
                │       (steps integrator)                    │
                │                                             │
                │  Auto-extend: trigger predictions ahead of  │
                │  current epoch by lookahead × time_scale    │
                └─────────────────────────────────────────────┘
```

## 5. Crate layout

Extends `crates/physics/`. Existing `body_state_provider.rs`,
`patched_conics.rs`, `orbital_math.rs`, `parsing.rs`, `types.rs`
(`BodyDefinition`, `StateVector`, `BodyState`, `BodyId`) remain as the
body-side foundation. `ship_propagator.rs` (1654 lines) and
`simulation.rs` (811 lines) are deleted. The `trajectory/` subdirectory
is rewritten.

```
crates/physics/src/
  lib.rs
  types.rs               // BodyDefinition, StateVector, BodyState, BodyId, ShipState
  body_state_provider.rs // unchanged (BodyStateProvider trait)
  patched_conics.rs      // unchanged (Keplerian rails impl)
  orbital_math.rs        // unchanged
  parsing.rs             // unchanged
  maneuver.rs            // ManeuverNode (Δv input), TNB frame helpers

  trajectory/
    mod.rs               // BoundedTrajectory + EvaluateTrajectory traits
    cubic_hermite.rs     // CubicHermiteSpline<DVec3> (ship cache)
    keplerian.rs         // KeplerianTrajectory (body adapter over BodyStateProvider)
    relative.rs          // RelativeTrajectory<T1, T2>

  integrator/
    mod.rs               // Integrator trait, ODE problem
    dormand_prince87.rs  // adaptive RK 8(7) with PI step controller

  burn/
    mod.rs               // BurnModel trait, BurnDynamics
    constant_thrust.rs   // ConstantThrust impl

  flight_plan.rs         // FlightPlan, ManeuverNode → Timeline conversion, divergence
  ship_propagator.rs     // ShipPropagator, ODE wiring, integrator stepping
  simulation.rs          // delete
```

## 6. Data model

### 6.0 Epoch

`Epoch` is `f64` seconds since J2000.0 (the existing convention in
`thalos_physics::types`). For v1 keep it as a type alias to avoid a
sweeping rename:

```rust
pub type Epoch = f64;  // seconds since J2000.0
```

A typed wrapper (à la `ftime::Epoch`) is not worth the churn now;
revisit if calendar-aware UI ("set epoch to 2030-06-01 00:00 UTC")
becomes a thing, at which point `Duration` and `Epoch` arithmetic
type-safety pays for itself.

### 6.1 Trajectory traits

```rust
pub trait BoundedTrajectory {
    fn start(&self) -> Epoch;
    fn end(&self) -> Epoch;
    fn contains(&self, t: Epoch) -> bool { /* default impl */ }
}

pub trait EvaluateTrajectory {
    fn position(&self, t: Epoch) -> Option<DVec3>;
    fn state_vector(&self, t: Epoch) -> Option<StateVector>;
}
```

Both bodies and ships implement these. Renderer / encounter analysis
holds a `&dyn BoundedTrajectory + EvaluateTrajectory`.

### 6.2 Body trajectory

`KeplerianTrajectory` wraps `&PatchedConics` + a `BodyId`:

```rust
pub struct KeplerianTrajectory {
    body_id: BodyId,
    provider: Arc<dyn BodyStateProvider>,
}

impl BoundedTrajectory for KeplerianTrajectory {
    fn start(&self) -> Epoch { Epoch::MIN }
    fn end(&self)   -> Epoch { Epoch::MAX }
}

impl EvaluateTrajectory for KeplerianTrajectory {
    fn state_vector(&self, t: Epoch) -> Option<StateVector> {
        Some(self.provider.query_body(self.body_id, t.into()).into())
    }
    // ...
}
```

When bodies graduate to a baked N-body ephemeris later, replace this
with `EphemerisTrajectory(UniformSpline<DVec3>)`. No call sites change.

### 6.3 Ship trajectory

```rust
pub struct CubicHermiteSpline {
    samples: Vec<(Epoch, StateVector)>,  // dense (time, pos, vel)
}
```

Cubic Hermite interpolation between samples — consistent C¹ across
coast segments, C⁰ across burn boundaries (which is correct: thrust
ignition/cutoff is a real discontinuity in acceleration, but velocity
is continuous). Append-only after the sample at index 0; `clear_after`
truncates a suffix; `clear_before` drops a prefix (for cache pruning).

Wrapped for ECS:

```rust
pub struct Trajectory(Arc<RwLock<CubicHermiteSpline>>);
```

Reads don't block other reads. The async worker holds the write lock
briefly during merge. Rendering and analysis are pure readers.

### 6.4 Relative trajectory

```rust
pub struct RelativeTrajectory<'a, T1: ?Sized, T2: ?Sized> {
    pub trajectory: &'a T1,
    pub reference: Option<&'a T2>,
}

impl EvaluateTrajectory for RelativeTrajectory<...> {
    fn state_vector(&self, t: Epoch) -> Option<StateVector> {
        let traj = self.trajectory.state_vector(t)?;
        let refr = self.reference.as_ref()
            .map(|r| r.state_vector(t)).transpose()?
            .unwrap_or_default();
        Some(traj - refr)
    }
}
```

Used by trajectory rendering: `RelativeTrajectory::new(&ship, Some(&earth))`
gives the ship's path in Earth's instantaneous frame.

### 6.5 Ship state

```rust
pub struct ShipState {
    pub state: StateVector,    // position + velocity, inertial frame
    pub mass_kg: f64,          // current wet mass
}
```

Mass is integrated alongside position+velocity during burns. The full
integration state is `[StateVector, mass]`.

### 6.6 Burn model

The extensibility hook. Initial impl is `ConstantThrust`; future
high-fidelity variants slot in behind the same trait.

```rust
pub trait BurnModel: Send + Sync {
    /// Inertial-frame thrust acceleration and propellant mass flow at
    /// time `t`, given the ship's current state and the body context.
    /// Returns None if propellant is exhausted (caller stops applying
    /// thrust for the remainder of the segment).
    fn dynamics(
        &self,
        t: Epoch,
        ship: &ShipState,
        bodies: &dyn BodyStateProvider,
    ) -> Option<BurnDynamics>;

    fn start_time(&self) -> Epoch;
    fn end_time(&self)   -> Epoch;
    fn dry_mass_kg(&self) -> f64;
}

// `Arc<dyn BurnModel>` is the carry type — burn models are immutable
// once built, so propagator branching costs only an atomic increment.
// No `dyn_clone` machinery needed.

pub struct BurnDynamics {
    pub accel: DVec3,        // m/s², inertial frame
    pub mass_flow: f64,      // kg/s, positive = consumption
}
```

Initial implementation:

```rust
pub struct ConstantThrust {
    pub start: Epoch,
    pub end: Epoch,
    pub thrust_n: f64,
    pub isp_s: f64,                  // specific impulse
    pub dry_mass_kg: f64,
    pub direction: BurnDirection,
}

pub enum BurnDirection {
    /// Δv axis in the local TNB frame (prograde / normal / radial),
    /// where the frame is recomputed each evaluation from the ship's
    /// state relative to `reference_body`.
    LocalTNB { unit_axis: DVec3, reference_body: BodyId },
    /// Fixed inertial direction.
    Inertial { unit_axis: DVec3 },
}
```

`dynamics` reads `ship.mass_kg`, returns `accel = thrust_n / mass × direction_inertial`
and `mass_flow = thrust_n / (isp_s × g_0)`. Returns `None` once
`mass_kg ≤ dry_mass_kg`.

Future high-fidelity types (`ThrottleCurveBurn`, `GimballedBurn`, etc.)
implement the same trait and slot into `Timeline` segments.

### 6.7 Maneuver node (UI input)

User-facing, Δv-based:

```rust
pub struct ManeuverNode {
    pub start_time: Epoch,
    pub delta_v_local: DVec3,        // m/s, prograde / normal / radial
    pub reference_body: BodyId,
    pub thrust_n: f64,               // ship's current engine
    pub isp_s: f64,
    pub dry_mass_kg: f64,
    pub enabled: bool,
}
```

Converted to a `ConstantThrust` by `FlightPlan::generate_timeline`
using the rocket equation. With initial mass `m₀` from prior state:

```
m_final = m₀ × exp(-|Δv| / (isp_s × g₀))
duration = (m₀ - m_final) × isp_s × g₀ / thrust_n
```

If `m_final < dry_mass_kg`, the burn is propellant-limited: cap
duration at the time when mass hits dry, and the realised Δv falls
short. Surface this in the UI ("burn cuts off, target Δv unattainable").

### 6.8 Timeline

Integrator-facing. Sorted segments, generated from the flight plan:

```rust
pub enum TimelineSegment {
    Coast { start: Epoch, end: Epoch },
    Burn  { start: Epoch, end: Epoch, model: Arc<dyn BurnModel> },
}

pub struct Timeline {
    segments: Vec<TimelineSegment>,
}

impl Timeline {
    pub fn segment_at(&self, t: Epoch) -> &TimelineSegment;

    /// Walk both timelines in parallel; return the first epoch where
    /// they differ. Used for incremental re-prediction on flight-plan
    /// edits — restart integration from there, reuse the prefix.
    pub fn divergence_time(&self, other: &Self, before: Epoch) -> Option<Epoch>;
}
```

### 6.9 Flight plan

```rust
pub struct FlightPlan {
    pub end: Epoch,                           // prediction horizon
    pub nodes: IndexMap<NodeId, ManeuverNode>,
    pub integrator: IntegratorConfig,         // tolerance, step bounds
}

impl FlightPlan {
    pub fn generate_timeline(&self, initial_mass: f64) -> Timeline;
}
```

## 7. Integrator

Adaptive Dormand-Prince 8(7), single hardcoded choice. Eight
function evaluations per step; PI step-size controller targets a user
tolerance `atol_pos`, `atol_vel`, `atol_mass` per state component.

ODE state is `[r, v, m]` (9 doubles per ship). Right-hand side:

```
dr/dt = v
dv/dt = Σ_b GM_b × (r_b(t) − r) / |r_b(t) − r|³  +  burn_acceleration
dm/dt = -burn_mass_flow
```

Where `r_b(t)` is queried from the `BodyStateProvider`. During burns,
`burn_acceleration` and `burn_mass_flow` come from the active
`BurnModel`. During coasts, both are zero.

At coast↔burn boundaries, the integrator is reset (cached k-vectors
and step-size memory are invalid since the RHS is discontinuous in its
derivatives). Step bound is set to the next segment boundary so the
integrator naturally lands on it.

Implementation: ~300 lines including the Butcher tableau coefficients
and the PI controller. Validated against published Dormand-Prince 8(7) reference
output and a two-body Kepler problem in tests.

## 8. Ship propagator

```rust
pub struct ShipPropagator {
    integrator: DormandPrince87,
    state: IntegrationState,         // [r, v, m] + time + step size
    timeline: Timeline,
    bodies: Arc<dyn BodyStateProvider>,
    cursor: usize,                   // current segment index
}

impl ShipPropagator {
    pub fn new(
        initial: ShipState,
        initial_time: Epoch,
        timeline: Timeline,
        bodies: Arc<dyn BodyStateProvider>,
        config: IntegratorConfig,
    ) -> Self;

    /// Advance one integrator step. Pushes one sample to `out`.
    pub fn step(&mut self, out: &mut CubicHermiteSpline)
        -> Result<(), StepError>;

    /// Step until `out.end() ≥ horizon`.
    pub fn step_to(&mut self, out: &mut CubicHermiteSpline, horizon: Epoch)
        -> Result<(), StepError>;

    /// Snapshot for incremental shipping (mem::replace pattern).
    pub fn branch(&self) -> CubicHermiteSpline;

    pub fn time(&self) -> Epoch;
    pub fn timeline(&self) -> &Timeline;
}
```

The propagator owns its integration state and timeline copy; it
doesn't hold or write to the ECS-side `Trajectory`. Snapshots cross
the thread boundary as plain `CubicHermiteSpline` values; the main
thread merges into the `Arc<RwLock<...>>`.

## 9. Concurrency model

### 9.1 Components

```rust
#[derive(Component)]
pub struct PredictionPropagator(pub ShipPropagator);

#[derive(Component)]
pub struct PredictionTracker {
    task: bevy::tasks::Task<()>,
    receiver: async_channel::Receiver<Snapshot>,
    paused: Arc<AtomicBool>,
    target_horizon: Epoch,
    progress: Epoch,
}

pub struct Snapshot {
    pub propagator: ShipPropagator,
    pub trajectory: CubicHermiteSpline,
    pub soi_transitions: SoiTransitions,
}
```

### 9.2 Worker protocol

Branching pattern (taken straight from ephemeris-explorer):

```rust
async fn prediction_task(
    mut propagator: ShipPropagator,
    mut trajectory: CubicHermiteSpline,
    mut transitions: SoiTransitions,
    horizon: Epoch,
    sender: async_channel::Sender<Snapshot>,
    sync: SyncPolicy,
) {
    loop {
        propagator.step(&mut trajectory)?;
        update_soi_transitions(&propagator, &mut transitions);

        if sync.is_ready() && sender.is_empty() {
            let snap = Snapshot {
                propagator: propagator.clone(),
                trajectory: mem::replace(&mut trajectory, propagator.branch()),
                soi_transitions: mem::take(&mut transitions),
            };
            if sender.send(snap).await.is_err() { break; }
            sync.reset();
        }

        if propagator.time() >= horizon { /* final send and break */ }
    }
}
```

Worker never blocks on the receiver. When the channel is full, the
worker keeps integrating; the next send replaces the queued snapshot.
Main thread `process_predictions` system pulls snapshots and merges
into the `Arc<RwLock<CubicHermiteSpline>>`.

### 9.3 Sync policy

Two strategies, exposed in `SyncPolicy`:
- `Steps(N)`: send every N integrator steps. Default 100.
- `Hertz(H)`: send every 1/H wall-clock seconds. Default 60 Hz.

Hertz preferred for interactive responsiveness; steps useful for
tests.

### 9.4 Trajectory merge

```rust
fn merge(world_traj: &mut CubicHermiteSpline, snap: CubicHermiteSpline) {
    world_traj.clear_after(snap.start());
    world_traj.extend(snap);
}
```

`clear_after` is necessary because the snapshot starts at
`branch_time`, which may be before any speculative samples that were
already in the cache (e.g. from a partially-completed previous
extension).

## 10. Time control

### 10.1 SimulationTime

```rust
pub struct SimulationTime {
    pub current: Epoch,
    pub time_scale: f64,    // warp factor; 0 = paused
    pub paused: bool,
    pub bounds_start: Epoch,
    pub bounds_end: Epoch,
}
```

Each frame:

```rust
fn advance_simulation_time(time: Res<Time>, mut sim: ResMut<SimulationTime>) {
    if sim.paused { return; }
    let delta = time.delta_seconds() * sim.time_scale;
    sim.current = (sim.current + delta).clamp(sim.bounds_start, sim.bounds_end);
}
```

`bounds_end` is the minimum trajectory `.end()` across all marked
"bounds" entities (typically the player's active ship — when its
prediction is the slowest, it's what gates time).

### 10.2 Auto-extend

Every frame, request that all active flight plans extend their
predictions to `current + lookahead × time_scale`:

```rust
fn auto_extend(...) {
    let lookahead = Duration::from_seconds(5.0);
    let target = sim.current + lookahead × sim.time_scale;
    for plan_entity in active_flight_plans.iter() {
        commands.trigger(ExtendRequest {
            entity: plan_entity,
            target,
            forced: time_scale_just_changed,
            buffer: lookahead × sim.time_scale,
        });
    }
}
```

The `forced` flag fires when `time_scale` changes so a sudden warp
jump doesn't wait for a stale extension to finish — the worker is
restarted with the new horizon.

### 10.3 Warp-to-epoch (UI affordance)

For "skip ahead a year": pause simulation, trigger extension to the
target, ease `sim.current` toward target with cubic ease-in-out over a
fixed wall-clock duration (~2 s). Resumes auto-extend when the
animation completes.

```rust
pub struct WarpRequest { pub target: Epoch, pub duration: Duration }
```

### 10.4 Parallel ships

Every ship is an entity with its own `FlightPlan`,
`PredictionPropagator`, and `Trajectory`. All are advanced
independently by the prediction system. A ship that's not actively
edited still has its prediction extended and its position synced — it
flies on rails alongside whatever the player is doing.

The "active ship" distinction lives only in input/UI systems
(`ActiveShip` resource pointing to one entity). Maneuver editing
affects the active ship's flight plan; physics doesn't care which
ship is which.

#### Bounds policy

`BoundsTime` is a marker component that says "this entity's prediction
must stay ahead of `sim_time.current`":

```rust
sim_time.bounds_end = world.query::<&Trajectory, With<BoundsTime>>()
    .map(|t| t.end()).min().unwrap_or(Epoch::MAX);
```

The marker is added to a ship when its `FlightPlan` is created, and
removed when the flight plan ends (the final maneuver completes and
the ship has no further scheduled action). This is the rule that
matches the user's "boost lands while Mars rocket flies" scenario:

- **Active ship is being edited:** `BoundsTime` is on it. `sim_time`
  cannot run past its prediction.
- **Booster is auto-flying its descent:** `BoundsTime` is also on it
  while its flight plan still has un-executed burns. Time can't
  outrun the descent prediction.
- **Mars rocket finished its trans-Mars injection and is coasting
  with no further nodes:** `BoundsTime` is removed. The ship still
  flies on rails (`sync_position_to_time` reads its `Trajectory`),
  but it doesn't gate sim time. If `sim_time` warps past the rocket's
  cached trajectory end, its position freezes; auto-extend continues
  to push the cache forward at lower priority.

Bodies on Keplerian rails are always `BoundsTime`-implicitly-infinite,
so they don't enter the `min()`. (If/when bodies become bounded
N-body trajectories, mark the body entities `BoundsTime` too.)

## 11. Maneuver editing

### 11.1 Edit flow

1. User drags a Δv handle, changes `start_time`, toggles `enabled`.
2. `FlightPlanChanged(entity)` event fires.
3. `apply_flight_plan` system:
   ```rust
   let new_timeline = flight_plan.generate_timeline(initial_mass);
   // Find the latest epoch unaffected by the change. None means the
   // change happened at or before the trajectory start (e.g. integrator
   // params changed) — fall back to a full recompute.
   let restart_epoch = new_timeline
       .divergence_time(&prev_timeline, flight_plan.end)
       .unwrap_or_else(|| trajectory.read().start());
   let restart_state = trajectory.read().state_at(restart_epoch)?;
   let new_propagator = ShipPropagator::new(
       restart_state, restart_epoch, new_timeline, bodies.clone(), config);
   commands.trigger(ComputePrediction::extend(
       entity,
       flight_plan.end - restart_epoch,
   ).with_propagator(new_propagator));
   ```
4. The async worker re-integrates only the suffix.
5. Trajectory `merge` clears samples after `restart_epoch` and appends
   new ones.

`divergence_time` walks both timelines comparing `(start, end, model)`
of each segment, returning the first mismatched epoch (or `None` if
the timelines are identical or the user changed something that
invalidates the entire run, like the integration tolerance). For the
`model` comparison: each `BurnModel` impl exposes a stable identity
(parameter-derived `PartialEq` or hash) so timeline diffing doesn't
need trait-object equality gymnastics.

### 11.2 Drag responsiveness

Targeted: <100 ms for a single-leg local maneuver (transfer of a few
hours, ~10 minutes of integration time on a typical orbit). With
Dormand-Prince 8(7) at moderate tolerance (~10⁻⁹ position, ~10⁻¹² velocity), this
fits.

If long-horizon plans degrade interactivity, the followups are:
- Coarser tolerance during drag, refine on release.
- Cap drag-prediction horizon shorter than the rendered horizon
  (predict 1 hour ahead during drag, then extend after release).
- Per-segment resumable integrator state (skip re-running integration
  for segments that didn't change).

None of these are required for v1.

### 11.3 Mass coupling

Editing burn N's Δv changes the mass at burn N+1's start, which
changes its duration, which changes the mass at burn N+2's start, etc.
`generate_timeline` computes durations sequentially, accepting the
mass cascade. If the user edits an early burn, the entire flight plan
re-derives its timeline; from the integrator's perspective this is
just a bigger divergence — the same divergence-detection re-prediction
flow handles it.

## 12. Trajectory rendering

### 12.1 Anchor selection

A plot is described by a source ship/body and an optional reference
body to anchor against:

```rust
pub struct PlotSource {
    pub entity: Entity,                  // the trajectory being drawn
    pub reference: Option<Entity>,       // body to anchor relative to
}
```

Reference selection rule (in order):
1. If the user has explicitly selected a focus body, use it.
2. Otherwise, infer from `SoiTransitions`: the body whose SOI contains
   the ship at `sim_time.current` is the default anchor.
3. Heliocentric / barycentric fallback if the ship is in no body's
   SOI.

When bodies become N-body later, "SOI" is replaced by Hill sphere; the
classification logic is unchanged.

### 12.2 Sampling

For each plot, build a `RelativeTrajectory` between source and
reference. Walk the trajectory between `min` and `max` epochs with
adaptive sampling — Principia's `PlotMethod3` algorithm
([reference](https://github.com/mockingbirdnest/Principia/blob/2024080411-Klein/ksp_plugin/planetarium_body.hpp)):

```
target_tan2_error = (angular_resolution × camera_fov)²
loop:
    delta = previous_delta × 0.9 × (target_error / estimated_error)^0.25
    extrapolated = previous_pos + previous_vel × delta
    actual = trajectory.state_vector(t + delta)
    error = angular_distance²(extrapolated, actual, camera_pos)
    if error ≤ target: emit sample, advance
    else: shrink delta, retry
```

Sample density adapts to camera angle — dense near the camera, sparse
far away. Cuts orbit-line vertex counts by 10–50× versus uniform-time
sampling at usable visual quality.

Output is a `Vec<(Epoch, Vec3)>` in world space, with a constant
translation by `reference.position(sim_time.current)` to lift the
relative trajectory into the visible scene. This is what makes a
stable orbit *look stable* — both the ship and its reference body
move during the rendered span, but plotted in a snapshot of the
reference's frame at "now," the curve stays still.

### 12.3 Burn segments

Visually distinguish burn segments from coast segments (different
colour or dashed line). The segment metadata is on the `FlightPlan`;
the renderer queries `timeline.segment_at(t)` for each plotted point
to decide the style.

## 13. SOI tracking

```rust
pub struct SoiTransitions(Vec<(Epoch, BodyId)>);
```

Sorted list of body changes for a ship. Updated alongside trajectory
samples by the prediction worker:

```rust
fn update_soi_transitions(p: &ShipPropagator, t: &mut SoiTransitions) {
    // After each step from t0 to t1:
    for body in bodies {
        if let Some(crossing) = bisect_soi_crossing(body, t0, t1, p0, p1) {
            t.insert(crossing.time, crossing.body);
        }
    }
}
```

Pure post-step bookkeeping. Bisection finds the exact epoch where the
ship crosses the SOI sphere (`|r_ship(t) - r_body(t)| = soi_radius`);
relative velocity sign decides enter vs. exit. ~50 lines of code.

Used by:
- Hierarchical UI ("ship → Earth's SOI → Moon's SOI").
- Default anchor selection in trajectory rendering.
- Encounter detection ("closest approach to body X" — ternary search
  on the `RelativeTrajectory<Ship, BodyX>` over the segment).

When bodies graduate to N-body, "SOI sphere" becomes "Hill sphere
radius computed from current mass + orbit." The detection algorithm
is unchanged.

## 14. Cache management

### 14.1 Pruning policy

Behind `sim_time.current`, retain at most `retention_window` of past
samples:

```rust
let retention_window = Duration::from_seconds(3600.0);  // 1 hour, configurable
trajectory.write().clear_before(sim_time.current - retention_window);
```

Run once per second of wall-clock time. Default 1 hour gives the
player ample scrubbing room; configurable via `SimulationConfig`.

If the player rewinds (via UI affordance not in v1) past the
retention window, the past portion of the trajectory is gone. For v1
this is fine; if scrubbing becomes a feature, revisit (options:
larger retention, lazy backward re-integration).

### 14.2 Forward bound

No hard cap on prediction length. Auto-extend's lookahead bounds
worker effort to `5 s × time_scale` of cache ahead of `current`.
At 10⁶× warp that's ~58 days of cache — acceptable memory.

### 14.3 Memory ballpark

A `(Epoch, StateVector)` is 56 bytes. For a year-long flight plan
with ~hour-scale adaptive steps in deep space and minute-scale near
periapses, expect 50k–200k samples per ship — ~3–11 MB per ship.
Multiple ships add linearly. Comfortable.

## 15. Determinism

- All integration uses `f64` throughout; no `f32` shortcuts.
- Dormand-Prince 8(7) step controller produces identical step sequences for
  identical inputs. No randomness, no wall-clock-dependent decisions.
- Snapshot cadence (`SyncPolicy::Hertz`) does NOT affect numerical
  output — it only changes when chunks cross the thread boundary. The
  same trajectory results regardless of snapshot rate.
- Body queries through `BodyStateProvider` are pure functions of time.
- Tests: spin two propagators with identical inputs, run to the same
  horizon, assert bit-equal trajectories.

## 16. Migration plan

Phased so the game stays playable through the rewrite. Each phase
ends in a working build with no half-finished code.

### Phase 1 — Trajectory traits and Keplerian adapter
- Add `BoundedTrajectory` + `EvaluateTrajectory` traits.
- Add `KeplerianTrajectory` wrapper over `PatchedConics`.
- Add `RelativeTrajectory<T1, T2>`.
- Add `CubicHermiteSpline`.
- No behaviour change. New types unused.

### Phase 2 — Integrator
- Implement `DormandPrince87` with PI controller in `integrator/`.
- Tests: known Dormand-Prince 8(7) reference output, two-body Kepler problem
  comparison against `orbital_math::propagate_kepler`.
- Still no integration with the game.

### Phase 3 — Burn model and timeline
- Define `BurnModel` trait, `ConstantThrust` impl, `Timeline`.
- Move `ManeuverNode` from current `maneuver.rs` to the new shape (Δv
  input).
- `FlightPlan::generate_timeline` and `divergence_time`.
- Tests: rocket-equation-derived duration, divergence detection on
  hand-built timeline pairs.

### Phase 4 — Ship propagator
- Wire Dormand-Prince 8(7) + BodyStateProvider + Timeline + BurnModel into
  `ShipPropagator`. ODE RHS, segment-boundary handling, sample
  emission.
- Tests: single-coast Kepler match (compare propagated trajectory to
  closed-form Kepler over multiple orbits), burn segment with mass
  flow, mass-cutoff behaviour.

### Phase 5 — Async prediction worker
- `PredictionPropagator` / `PredictionTracker` components,
  `ComputePrediction` event, `prediction_task` async loop, merge
  system.
- Game still uses old physics for live state. Async prediction runs
  alongside, output discarded.
- **Bevy 0.18 ECS API parity check.** ephemeris-explorer is on a
  newer Bevy with its own observer / `EntityEvent` / relationship
  conventions. Verify the equivalent surface in Thalos's Bevy 0.18
  before wiring (specifically: `bevy::tasks::AsyncComputeTaskPool`,
  `Task` futures, observer / event semantics for `ComputePrediction`,
  `FlightPlanChanged`). Adapt the protocol shape to whatever 0.18
  exposes; the underlying pattern (worker holds propagator, ships
  snapshots over a bounded async channel) is engine-agnostic.

### Phase 6 — Cut over live state
- `SimulationTime` becomes the single clock.
- `sync_position_to_time` reads `Trajectory.position(sim_time.current)`
  for every entity.
- Delete old `simulation.rs`, `bridge.rs` propagation paths,
  `ship_propagator.rs` (old).
- Auto-extend system on.
- This is the destabilising phase — worth a dedicated branch and
  careful testing of warp transitions, parallel ships.

### Phase 7 — Trajectory rendering
- Replace existing trajectory rendering with `RelativeTrajectory`
  + Principia-style sampling.
- Anchor selection from SOI metadata.
- Burn-segment styling.

### Phase 8 — Maneuver editing
- Wire up `FlightPlanChanged` event, `apply_flight_plan` re-prediction
  flow.
- UI: drag Δv handles, edit start time, toggle enable.
- Verify drag responsiveness against the targeted budget.

### Phase 9 — Polish
- Cache pruning.
- Warp-to-epoch animation.
- SOI-driven trajectory colouring.
- Whatever the previous phases revealed.

## 17. Open questions

These don't gate the spec but want decisions before each lands:

- **Cache pruning trigger:** every wall-clock second, or every Nth
  frame? Either fine; pick whichever is simpler.
- **Snapshot cadence default:** 60 Hz seems right for visible warp
  smoothness; revisit if main-thread merge cost becomes visible in
  profiling.
- **Encounter detection:** ternary search on `RelativeTrajectory` over
  the cached span — when does it run? On demand for a selected target,
  or continuously for a list of "interesting" bodies?
- **Floating origin / big-space integration:** ephemeris-explorer
  uses `big_space` for Bevy-side rendering at solar-system scales.
  Thalos already handles this differently; confirm what stays.
- **Worker throughput at extreme warp.** Auto-extend's `5 s × time_scale`
  lookahead means at 10⁶× warp the worker has to extend ~58 days of
  cache to keep up. Probably fine for one ship at hours-scale; may be
  tight at year-scale flight plans with multiple parallel ships.
  Numbers come from profiling — flag during phase 6, don't pre-optimise.

## 18. References

- ephemeris-explorer (Canleskis): the architectural model adopted here.
  https://github.com/Canleskis/ephemeris-explorer
- Principia's `PlotMethod3` for adaptive trajectory sampling.
  https://github.com/mockingbirdnest/Principia
- Hairer, Nørsett, Wanner — *Solving Ordinary Differential Equations I*,
  Section II.5 (Dormand-Prince 8(7) coefficients and step controller).
- `docs/stale/new_physics.md` — earlier (abandoned) design using
  Chebyshev-fit ephemerides + symplectic leapfrog. Kept for reference.
