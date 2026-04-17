# Sol: Design Document

A spaceflight simulator with an N-body orbital map view, maneuver planner, and
automatic encounter visualization.


## What We're Building

The map view is a 3D orbital diagram. Celestial bodies are 
rendered spheres. Orbits are the primary visual element. The player places maneuver
nodes on their predicted trajectory and drags handles to shape burns, seeing the
resulting orbit update in real time.

This is the entire game loop for now. No flight view, no planetary rendering, no
ship model. Just the map, the sim, and the planner.

### Core Loop

1. N-body sim ticks forward (Sun, Earth, Moon, ship)
2. Ship's future trajectory is predicted by propagating a ghost state forward
3. Predicted trajectory is drawn as a polyline, rendered in the locally correct
   reference frame per segment, color-coded by dominant body
4. Player places a maneuver node on that trajectory
5. Delta-v handle adjustments re-propagate from the node forward, showing the
   post-burn trajectory and updated encounter information
6. Player can chain multiple nodes; each propagation feeds into the next
7. When sim time reaches a node, the burn executes


---


## Architecture

Four layers. Each depends only on the one below it.

```
┌─────────────────────────────────────────────┐
│  UI / Rendering                             │
│  (egui panels, gizmo drawing, camera)       │
├─────────────────────────────────────────────┤
│  ECS Integration                            │
│  (Bevy resources, systems, sync)            │
├─────────────────────────────────────────────┤
│  Prediction Pipeline                        │
│  (ghost propagation, trail generation,      │
│   phase machine, encounter detection)       │
├─────────────────────────────────────────────┤
│  Simulation Core                            │
│  (integrator, gravity, orbital math,        │
│   burn models)                              │
└─────────────────────────────────────────────┘
```

The sim core never touches Bevy. The prediction pipeline uses only the sim core.
ECS integration owns resources and wires the pipeline to the frame loop. UI reads
resources and emits events.


---


## Layer 1: Simulation Core

Pure Rust, no Bevy dependency. Lives in its own crate (or module initially) so it
can be tested independently and eventually compiled to WASM or used headless.

Everything is SI: meters, seconds, kilograms.


### Types

```rust
pub struct SimState {
    pub positions: Vec<DVec3>,
    pub velocities: Vec<DVec3>,
    pub masses: Vec<f64>,
}
```

Clone-cheap by design. Prediction clones and propagates a ghost copy. If clone
cost ever matters, switch inner vecs to `Arc<[T]>` with copy-on-write.

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u64);
```

Monotonically increasing counter, assigned by the ECS layer when a node is created.


### Integrator trait

```rust
pub trait Integrator: Send + Sync {
    fn step(
        &self,
        state: &mut SimState,
        accel_fn: &dyn Fn(&SimState) -> Vec<DVec3>,
        dt: f64,
    );
}
```

V1: `RK4Integrator`. The trait exists so we can later swap in Dormand-Prince
(adaptive RK45), symplectic leapfrog, or anything else. The `accel_fn` parameter
decouples the integrator from the force model.


### Force model trait

```rust
pub trait ForceModel: Send + Sync {
    fn compute_accelerations(&self, state: &SimState) -> Vec<DVec3>;
}
```

V1: `NBodyGravity`. Pairwise gravitational acceleration with an optional exclusion
mask so the ship's negligible mass doesn't waste cycles accelerating the Sun.

Future: `OblateGravity` (J2), `AtmosphericDrag`, `SolarRadiationPressure`,
`CompositeForceModel` that chains multiple models.


### Burn model trait

A burn is a function that injects acceleration over a time interval. Not a
point event on the state.

```rust
pub trait BurnModel: Send + Sync {
    /// Thrust acceleration to apply this substep (world-space, m/s^2).
    /// `dominant_body_idx` is needed to compute the orbital frame for burns
    /// specified in orbital-frame coordinates.
    fn acceleration(
        &self,
        state: &SimState,
        ship_idx: usize,
        dominant_body_idx: usize,
        time: f64,
        dt: f64,
    ) -> Option<DVec3>;

    /// Time window during which this burn is active. (start, end).
    /// For impulse burns, start == end.
    fn time_window(&self) -> (f64, f64);

    /// Total delta-v magnitude (for UI display and budget tracking).
    fn total_delta_v(&self) -> f64;

    /// Clone into a new Box. Needed for prediction (cloning the maneuver plan)
    /// and future undo/redo.
    fn clone_box(&self) -> Box<dyn BurnModel>;
}
```

Note: `dominant_body_idx` is passed explicitly because the burn needs it to convert
orbital-frame delta-v (prograde/normal/radial) to world-space acceleration. The
prediction pipeline and the ECS execution system both know the dominant body and
pass it in.

**V1: `ImpulseBurn`**

```rust
#[derive(Clone)]
pub struct ImpulseBurn {
    pub time: f64,
    pub delta_v: DVec3,  // (prograde, normal, radial) in orbital frame
}
```

`acceleration()` returns `None` except during the single substep containing
`self.time`, where it converts delta-v from orbital frame to world frame and
returns `world_dv / dt`. `time_window()` returns `(time, time)`.

Important: when a substep straddles the impulse time, the system should split
the substep at the exact impulse time: integrate to `impulse_time`, apply the
kick, then integrate the remainder. This prevents burn-timing errors at high warp.

**V2 (future): `FiniteBurn`**

```rust
#[derive(Clone)]
pub struct FiniteBurn {
    pub start_time: f64,
    pub direction: DVec3,     // orbital frame, normalized
    pub thrust_newtons: f64,
    pub mass_flow_rate: f64,  // kg/s
    pub initial_mass: f64,
}
```

`acceleration()` returns `thrust / current_mass` for the burn's duration. Mass
decreases over time (Tsiolkovsky). The prediction pipeline doesn't care which
implementation it's running: it asks for acceleration each substep.


### Maneuver node

```rust
pub struct ManeuverNode {
    pub id: NodeId,
    pub burn: Box<dyn BurnModel>,
}

impl Clone for ManeuverNode {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            burn: self.burn.clone_box(),
        }
    }
}
```

Constructing a node from the UI always creates an `ImpulseBurn` for now.


### Orbital math (standalone functions)

```rust
/// Prograde/normal/radial basis for the ship's orbit around `center_pos`.
pub fn orbital_frame(pos: DVec3, vel: DVec3, center_pos: DVec3) -> DMat3

/// Classical Keplerian orbital elements.
pub fn orbital_elements(pos: DVec3, vel: DVec3, mu: f64) -> OrbitalElements

/// Hill sphere radius: a * (m_body / (3 * m_parent))^(1/3)
pub fn hill_radius(semi_major_axis: f64, body_mass: f64, parent_mass: f64) -> f64
```

`OrbitalElements` includes: semi_major_axis, eccentricity, inclination, period,
periapsis, apoapsis, specific_energy, longitude_of_ascending_node,
argument_of_periapsis, true_anomaly. Degenerate cases (circular, equatorial) must
not produce NaN; use fallback values (0.0 for undefined angles).

Pure functions, no state.


---


## Layer 2: Prediction Pipeline

No Bevy types. Takes sim core types in, produces trail data out. Contains the
prediction phase state machine, encounter detection, and adaptive step sizing.


### Output types

```rust
pub struct TrailSegment {
    pub points: Vec<DVec3>,
    pub body_positions: Vec<Vec<DVec3>>,  // [step_idx][body_idx]
    pub times: Vec<f64>,
    pub dominant_body: Vec<usize>,        // per-point
    pub phase: Vec<PredictionPhase>,      // per-point (for rendering style)
    pub after_node: Option<NodeId>,
}
```

`body_positions` stores every body's position at every trail timestep. Memory cost
with 4 bodies and 10,000 steps: 10,000 * 4 * 24 bytes = ~960 KB. Negligible. This
data is essential for frame-relative rendering (subtracting a body's position to
render the trail in that body's reference frame).

`dominant_body` is per-point because the ship can transition between gravitational
dominance mid-segment. The rendering color gradient follows this.

`times` maps trail points back to sim time (for node placement and encounter
correlation).

```rust
pub struct ClosestApproach {
    pub body_idx: usize,
    pub position: DVec3,
    pub body_position: DVec3,
    pub distance: f64,
    pub time: f64,
    pub is_collision: bool,
}

pub struct Encounter {
    pub body_idx: usize,
    pub entry_time: f64,
    pub exit_time: f64,
    pub closest_approach: f64,     // distance to body center
    pub closest_time: f64,
    pub relative_velocity: f64,    // at closest approach
    pub capture: CaptureStatus,
    pub periapsis_altitude: f64,   // above surface
    pub eccentricity: f64,         // osculating elements at closest approach
    pub inclination: f64,
}

pub enum CaptureStatus {
    Flyby,
    Captured,
    Impact,
    Graze { altitude: f64 },
}

pub struct PredictionResult {
    pub segments: Vec<TrailSegment>,
    pub encounters: Vec<Encounter>,
    pub approaches: Vec<ClosestApproach>,  // for non-encounter close passes
    pub termination: TerminationReason,
}
```

`ClosestApproach` and `Encounter` serve different purposes. `ClosestApproach` is a
geometric fact: the trail passed within X km of body Y. Every body gets one (the
minimum distance point). `Encounter` is richer: it exists only when the trail enters
a body's Hill sphere, and contains orbital elements, capture status, entry/exit
times. A close pass that doesn't enter the Hill sphere produces a `ClosestApproach`
but not an `Encounter`.


### Prediction phase state machine

The propagator tracks what phase the trajectory is in. This drives both termination
decisions and rendering style.

```rust
#[derive(Clone, Debug)]
pub enum PredictionPhase {
    /// Bound orbit. Tracking angular position for closure detection.
    Orbiting {
        body_idx: usize,
        start_angle: f64,
        crossed_half: bool,
    },

    /// Unbound or transitioning. Looking for encounter or escape.
    Transfer {
        origin_body: usize,
    },

    /// Inside a body's Hill sphere. Tracking closest approach.
    Encounter {
        body_idx: usize,
        entered_at: f64,
        closest_so_far: f64,
        passed_closest: bool,
    },

    /// Leaving the system. Propagating until visually clear.
    Escaping {
        from_body: usize,
        escape_distance: f64,
    },

    /// Prediction complete.
    Done {
        reason: TerminationReason,
    },
}

#[derive(Clone, Debug)]
pub enum TerminationReason {
    OrbitClosed,
    Collision { body_idx: usize },
    EncounterResolved,
    EscapeComplete,
    BudgetExhausted,
}
```

Transition diagram:

```
                    ┌──────────────────────────────┐
                    v                              │
Start ──> Orbiting ──> Transfer ──> Encounter ──> Orbiting (new body)
              │            │            │              │
              v            v            v              v
           Done         Escaping     Done           Done
          (closed)         │       (collision/      (closed)
                           v        resolved)
                         Done
                       (escaped)
```

Phase transitions:

- **Orbiting -> Transfer**: specific orbital energy relative to dominant body
  becomes positive, or dominant body changes.
- **Orbiting -> Done(OrbitClosed)**: ship crosses the initial radial line (same
  sign of radial velocity) after `crossed_half` is true. Continue for 5-10%
  overlap, then stop. Exception: don't close if maneuver nodes remain ahead or
  if a target is set and no encounter has been evaluated yet.
- **Transfer -> Encounter**: ship enters a body's Hill sphere.
- **Transfer -> Escaping**: ship is on a hyperbolic trajectory relative to its
  dominant body and distance is increasing.
- **Encounter -> Orbiting**: eccentricity of osculating orbit relative to
  encounter body drops below 1.0 (captured). Reset closure tracking for new body.
- **Encounter -> Transfer/Orbiting(original)**: ship exits the Hill sphere
  (flyby complete).
- **Encounter -> Done(Collision)**: distance to body center < body radius.
- **Escaping -> Done(EscapeComplete)**: distance exceeds 3-5x the Hill sphere of
  the body being escaped. Exception: if escaping into another body's domain
  (dominant body switches), transition to Orbiting or Transfer around the new
  dominant body instead.
- **Any -> Done(BudgetExhausted)**: step count reaches hard cap.


### Prediction configuration

```rust
pub struct PredictionConfig {
    pub max_steps: usize,           // hard cap, default 10,000
    pub base_dt: f64,               // seconds, default 60.0
    pub adaptive_dt: bool,          // default true
    pub target_body: Option<usize>,
    pub body_radii: Vec<f64>,
    pub body_hill_radii: Vec<f64>,
    pub body_parent: Vec<Option<usize>>,  // for Hill sphere computation
    pub extend_count: usize,        // "predict further" presses
}
```

`body_hill_radii` is precomputed at startup:
`r_hill = a * (m_body / (3 * m_parent))^(1/3)`. Earth's Hill sphere around the
Sun is ~1.5 million km. Moon's Hill sphere around Earth is ~60,000 km.


### Propagation algorithm

```rust
pub fn predict(
    state: &SimState,
    ship_idx: usize,
    nodes: &[ManeuverNode],
    force_model: &dyn ForceModel,
    integrator: &dyn Integrator,
    config: &PredictionConfig,
    t0: f64,
) -> PredictionResult
```

The algorithm:

```
clone state as ghost
sort nodes by time
initialize phase = determine_initial_phase(ghost, ship_idx, config)
current_segment = new TrailSegment

for step in 0..config.max_steps:
    // adaptive dt
    min_altitude = min distance from ship to any body
    dt = base_dt * clamp(sqrt(min_altitude / REFERENCE_ALTITUDE), 0.01, 10.0)
    if target_body is set and ship is approaching target:
        dt = min(dt, base_dt * 0.2)  // finer resolution near target

    // compute accelerations
    accel = force_model.compute_accelerations(ghost)
    dominant = body with strongest gravitational pull on ship

    // add active burn accelerations
    for node in nodes:
        if let Some(burn_accel) = node.burn.acceleration(ghost, ship_idx, dominant, t, dt):
            accel[ship_idx] += burn_accel

    // step
    integrator.step(ghost, combined_accel_fn, dt)
    t += dt

    // record trail point
    current_segment.points.push(ghost.positions[ship_idx])
    current_segment.body_positions.push(ghost.positions.clone())
    current_segment.times.push(t)
    current_segment.dominant_body.push(dominant)

    // segment splitting: when a burn window ends, start new segment
    if a node's burn just finished:
        finalize current_segment
        current_segment = new TrailSegment { after_node: Some(node.id) }

    // update phase machine
    phase = update_phase(phase, ghost, ship_idx, config, t)

    if phase is Done:
        if extend_count > 0:
            extend_count -= 1
            phase = Orbiting { reset for current dominant body }
        else:
            break

finalize current_segment
scan all segments for closest approaches to each body
compute Encounters from phases that entered Encounter state
return PredictionResult
```

### Encounter detection

Encounter data is gathered in two passes:

**During propagation**: the phase machine tracks Hill sphere entry/exit times and
closest approach distance/time for each encounter. This data is accumulated as
the ghost state is stepped forward.

**Post-propagation**: `detect_encounters()` takes the raw encounter data from the
phase machine and enriches it with orbital elements. For each encounter, it
computes osculating elements at closest approach (relative position and velocity
of ship to encounter body at that timestep, fed into `orbital_elements()`). This
produces the full `Encounter` struct with eccentricity, capture status, periapsis
altitude, inclination, and relative velocity.

```rust
pub fn detect_encounters(
    raw_encounters: &[RawEncounter],  // from phase machine
    segments: &[TrailSegment],
    body_radii: &[f64],
    body_masses: &[f64],
) -> Vec<Encounter>
```

`ClosestApproach` for non-encounter bodies is computed by a separate scan of the
trail segments after propagation:

```rust
pub fn find_closest_approaches(
    segments: &[TrailSegment],
    ship_idx: usize,
    body_radii: &[f64],
) -> Vec<ClosestApproach>
```

This checks whether the trajectory *between* consecutive trail points could have
passed closer than either endpoint (interpolation check), guarding against coarse
dt skipping over a close pass.


### Adaptive step size

`dt = base_dt * clamp(sqrt(min_altitude / REFERENCE_ALTITUDE), 0.01, 10.0)`

Where `min_altitude` is the ship's distance to the nearest body and
`REFERENCE_ALTITUDE` is a tuning constant (e.g. 1e7 meters, roughly Earth radius
scale).

When a target body is set and the ship is approaching it, the dt cap tightens
further to avoid skipping over the Hill sphere.

With a budget of 10,000 steps and base_dt of 60s:
- LEO orbit (~5,400s period): ~900 fine steps, good resolution
- Lunar transfer (~3 days): ~4,000 mixed steps
- Solar escape: ~6,000 coarse steps out to Jupiter distance

Later: use RK45 error estimates to drive dt directly (Dormand-Prince gives local
truncation error for free).


### Adaptive prediction length

The prediction runs until the phase machine reaches `Done`. The phase encodes what
"done" means for each situation:

| Situation                  | Phase sequence                          | Stops when                          |
|----------------------------|-----------------------------------------|-------------------------------------|
| Stable orbit, no target   | Orbiting -> Done(OrbitClosed)           | Radial line crossing + overlap      |
| Stable orbit, target set  | Orbiting (extended) -> ...              | Encounter resolved or planning horizon exceeded |
| Transfer trajectory        | Transfer -> Encounter -> Orbiting       | One orbit at destination            |
| Escape                     | Escaping -> Done(EscapeComplete)        | 3-5x Hill sphere distance           |
| Collision                  | Any -> Done(Collision)                  | Surface intersection                |
| Post-maneuver              | (continues past nodes) -> Done          | First applicable termination after last node |
| Budget exhausted           | Any -> Done(BudgetExhausted)            | Hard cap reached (10,000 default)   |

The player gets one manual control: "predict further" (key or button). Each press
increments `extend_count`, which overrides the next termination and re-enters
`Orbiting` for one more phase cycle. This is the escape hatch for edge cases
(wanting to see 10 orbits, exploring resonant encounter chains).


### Edge cases

**Chaotic/precessing orbits**: if the radial line is crossed multiple times with
the radius at each crossing within 5% of the initial radius, treat it as "close
enough" and terminate. If the crossings show monotonically changing radius, note
the trend ("orbit raising"/"orbit decaying") in the phase data for the UI.

**Very eccentric orbits (e > 0.95)**: adaptive dt handles periapsis resolution.
The radial-line-crossing closure method is robust regardless of eccentricity (no
angle-per-step threshold).

**Orbit transitions without maneuvers**: N-body perturbations can gradually shift
an orbit from stable Earth orbit to chaotic to lunar encounter. The phase machine
detects this naturally (Orbiting -> Transfer -> Encounter). This is a feature, not
a bug: it surfaces dynamics that patched conics hide.

**Multiple encounters in one prediction**: Encounter -> Orbiting -> Encounter.
Each encounter gets its own entry. The "predict further" button lets players
explore multi-encounter chains and resonant flyby sequences.


---


## Reference Frames and Encounter Visualization

### The Problem

Drawing the trajectory in one inertial frame makes most of it unreadable. A 400 km
LEO drawn in heliocentric coordinates is a tiny wobbling helix. KSP solves this
with patched conics (hard frame switch at SOI boundaries). Principia does N-body
but requires manual frame selection.

We want the readability of patched conics with the accuracy of N-body.

### Frame-per-segment rendering

The prediction trail is one continuous curve in inertial space but we *render*
each segment in the reference frame of its dominant body. "Relative to body X"
means: subtract body X's predicted position at each corresponding timestep from
the ship's predicted position.

This is why `body_positions` is stored in `TrailSegment`. At render time:

```
render_point[i] = (points[i] - body_positions[i][frame_body] - camera_offset) * RENDER_SCALE
```

A circular orbit around Earth becomes a circle. A lunar flyby becomes a clean
hyperbola. The player never thinks about reference frames.

### Visual per phase

| Phase       | Style                                                       |
|-------------|-------------------------------------------------------------|
| Orbiting    | Solid line, full opacity, frame = dominant body. Overlap past closure at reduced opacity. |
| Transfer    | Solid line, distinct color. Frame transitions from origin body to destination as dominant switches. |
| Encounter   | Highlighted (brighter/thicker). Frame = encounter body. Closest approach marked. |
| Escaping    | Fading line. Opacity decreases with distance.               |
| Collision   | Trail turns red/orange near impact. Ends at bold marker.    |

### Smooth frame transitions

At the boundary where dominant body changes from A to B, blend:

```
blended_pos[i] = lerp(
    points[i] - body_positions[i][A],
    points[i] - body_positions[i][B],
    t
)
```

where `t` ramps 0 to 1 over the transition zone. The zone width should be
proportional to the time spent in the overlap region (where both bodies exert
significant gravity), not a fixed point count. Quick flyby = short transition.
Slow approach = long transition.

A practical heuristic: the transition zone spans the time interval where both
bodies contribute >10% of the total gravitational acceleration on the ship.


### Targeting

The player can target a body (right-click body marker, or `T` while hovering).

**No target set:**
- Trail rendered in dominant body's frame per segment (auto-switching)
- Closest approach markers for all bodies
- Encounters detected for all bodies that the trail passes through

**Target set:**
- All of the above, plus:
- **Biased prediction**: finer dt near target, extended prediction length to
  resolve encounter
- **Ghost trail**: secondary trail in the target body's frame, rendered at lower
  opacity/dashed. Shows the trajectory from the target's perspective even when
  the ship is still in another body's gravity well.
- **Encounter info panel** (see below)
- **Target body highlighted**: larger marker, its own orbit trail visible


### Encounter info panel

When a target is set and an encounter exists:

```
┌─ Moon Encounter ──────────────────────────┐
│                                           │
│  Closest approach:  342 km (surface)      │
│  Relative velocity: 812 m/s               │
│  Time to encounter: 2d 14h 22m            │
│  Orbit type:        Captured (e = 0.41)   │
│  Inclination:       12.3°                 │
│                                           │
│  ┌─────────────────────────────┐          │
│  │     (mini orbit diagram     │          │
│  │      in target's frame)     │          │
│  └─────────────────────────────┘          │
│                                           │
└───────────────────────────────────────────┘
```

The mini diagram is a 2D projection of the encounter trajectory in the target's
frame. Target is a circle at center; ship trajectory curves around it. Periapsis,
apoapsis marked. Incoming/outgoing asymptotes visible for flybys.

When no encounter exists but a target is set:

```
┌─ Moon (targeted) ─────────────────────────┐
│  No encounter detected                    │
│  Distance at closest pass: 48,200 km      │
│  Phase angle: 62°                         │
└───────────────────────────────────────────┘
```

### Encounter-aware maneuver feedback

When the player adjusts a maneuver node while a target is set:

1. Player drags a handle
2. Prediction re-runs (< 1ms for 4 bodies, 10k steps)
3. Encounter detection runs on new trail
4. Panel updates: closest approach, capture status, eccentricity
5. Trail reshapes in target's frame

The player can drag prograde and watch eccentricity drop below 1.0 as they find
the capture threshold. That's the core "aha" moment.


### Planning aids (future, designed for)

- **Transfer window indicator**: background search for optimal burn time when
  target is set but no encounter exists
- **Intercept guidance**: show delta-v to match target's velocity at closest
  approach (just `relative_velocity` from encounter data)
- **Resonant orbit helper**: after flyby, show whether the post-flyby orbit
  produces a re-encounter
- **Multi-body encounter chain**: detect and display sequential encounters
  (Earth -> Moon -> escape -> Mars)


---


## Layer 3: ECS Integration

Thin glue. Owns the authoritative state as Bevy resources, runs systems that call
into layers 1 and 2.


### Resources

```rust
#[derive(Resource)]
pub struct PhysicsState {
    pub state: SimState,
    pub integrator: Box<dyn Integrator>,
    pub force_model: Box<dyn ForceModel>,
}

#[derive(Resource)]
pub struct SimClock {
    pub time: f64,
    pub warp: f64,
    pub warp_levels: Vec<f64>,    // [1, 10, 100, 1_000, 10_000, 100_000]
    pub warp_index: usize,
    pub paused: bool,
}

#[derive(Resource, Default)]
pub struct ManeuverPlan {
    pub nodes: Vec<ManeuverNode>,
    pub dirty: bool,              // set when nodes change, cleared after prediction
    next_id: u64,                 // monotonic counter for NodeId
}

impl ManeuverPlan {
    pub fn next_node_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }
}

#[derive(Resource, Default)]
pub struct PredictionCache {
    pub result: Option<PredictionResult>,
    pub computed_at: f64,
}

#[derive(Resource)]
pub struct CameraFocus {
    pub entity: Entity,
    pub body_index: usize,
    /// Which body's reference frame is used for trail rendering.
    /// Usually matches body_index, but can differ (e.g. camera on ship,
    /// frame on Earth). When camera focuses on ship, this is set to the
    /// ship's current dominant body.
    pub active_frame: usize,
}

#[derive(Resource)]
pub struct ShipConfig {
    pub sim_index: usize,
    pub thrust_newtons: f64,
    pub dry_mass: f64,
    pub fuel_mass: f64,
}

#[derive(Resource, Default)]
pub struct TargetBody {
    pub target: Option<usize>,
}

/// Dominant gravitational body for the ship in the live simulation.
/// Distinct from the per-point dominant_body in prediction trail data.
#[derive(Resource)]
pub struct LiveDominantBody {
    pub index: usize,
    pub entity: Entity,
}
```

`LiveDominantBody` is updated per frame based on the live simulation state. It's
used for: the orbital info panel, computing the orbital frame for active thrust
input, and setting `CameraFocus.active_frame` when focused on the ship.

The per-point `dominant_body` in `TrailSegment` is computed during prediction and
can differ from the live value (the prediction looks into the future where the
dominant body might change).


### Components

```rust
#[derive(Component)]
pub struct SimBody(pub usize);

#[derive(Component)]
pub struct CelestialBody {
    pub name: String,
    pub radius: f64,
    pub hill_radius: f64,
    pub color: Color,
}

#[derive(Component)]
pub struct Ship;
```


### Systems

**`step_simulation`** (every frame, not FixedUpdate; warp needs variable dt)

1. If paused, return.
2. `effective_dt = frame_dt * warp`, subdivided so no substep exceeds ~100s.
3. For each substep:
   a. Check if any maneuver node's impulse time falls within this substep.
      If so, split: integrate to the impulse time, apply burn, integrate
      the remainder.
   b. Compute combined acceleration: `force_model + any active burn contributions`.
   c. Call `integrator.step()`.
   d. Advance `sim_clock.time`.
4. Mark nodes whose time windows have fully passed as executed; remove from plan.

**`update_live_dominant_body`** (after step_simulation)

Find which body exerts the strongest gravitational acceleration on the ship.
Update `LiveDominantBody`. Update `CameraFocus.active_frame` if camera is
focused on the ship.

**`run_prediction`** (when `ManeuverPlan.dirty` or every ~0.5s)

1. Build `PredictionConfig` from current state (body radii, Hill radii, target).
2. Call `predict()` from layer 2.
3. Store result in `PredictionCache`.
4. Update `TargetBody.encounters` if target is set.
5. Clear `ManeuverPlan.dirty`.

**`sync_transforms`** (after step_simulation)

For each entity with `SimBody(idx)`, compute camera-relative f32 position:
`(sim_pos - camera_sim_pos) * RENDER_SCALE`, write to `Transform`.


### Events

```rust
pub enum ManeuverEvent {
    PlaceNode { trail_time: f64 },
    AdjustNode { id: NodeId, delta_v: DVec3 },
    DeleteNode { id: NodeId },
    SelectNode { id: NodeId },
    DeselectNode,
    WarpToNode { id: NodeId },
}
```

Input systems emit these. A handler system processes them, modifies `ManeuverPlan`,
sets `dirty = true`. This decouples input from state mutation for future undo/redo
and networking.


---


## Layer 4: UI / Rendering


### Camera

Orbit camera:
- `yaw: f32, pitch: f32` via right-drag
- `distance: f64` via scroll, logarithmic: `distance *= exp(scroll * 0.1)`
- Focus entity (Tab to cycle through bodies and ship)
- Camera sim-space position: `focus_sim_pos + direction * distance` (DVec3)
- Camera `Transform` is always at render origin; everything else moves relative
- Clip planes: `near = max(0.001, render_distance * 1e-4)`,
  `far = render_distance * 1e5`
- `RENDER_SCALE = 1e-6` maps sim meters to render units

When `CameraFocus.active_frame` differs from the focus entity's body (e.g. camera
focuses on ship, frame is Earth), the trail renders in the frame body's coordinates
while the camera tracks the focus entity. When the camera transitions between focus
targets, smoothly interpolate the frame origin over ~0.5s so the trail visually
morphs between reference frames.


### Body rendering

Small fixed-screen-size marker per body (ring or filled circle via Gizmos). Text
label beside it. Color per body (Sun: yellow, Earth: blue, Moon: grey).

When a target is set, the target body draws larger/brighter with its own orbit
trail visible.


### Trail rendering

Read `PredictionCache`. Per `TrailSegment`:

1. Determine the frame body for each point (usually `dominant_body[i]`, with
   blending at transitions).
2. Compute frame-relative position:
   `(points[i] - body_positions[i][frame_body] - camera_sim_offset) * RENDER_SCALE`
3. Draw with `Gizmos::linestrip`.
4. Color by dominant body per point, lerping at transitions.
5. Opacity fades toward the trail tail.
6. Phase-specific styling (see visual per phase table above).

If a target is set, draw a secondary ghost trail in the target's frame at lower
opacity.


### Maneuver node interaction

**Placement** (press `N`):
- Project trail points to screen space.
- Snap indicator follows the closest point to cursor.
- Click emits `ManeuverEvent::PlaceNode` with the corresponding sim time
  (looked up from `TrailSegment.times`).

**Handles** (selected node):
- Six arrows from node position along orbital frame axes.
- Fixed screen size (don't scale with zoom).
- Drag updates the corresponding delta-v component and sets `ManeuverPlan.dirty`.
- Prediction re-runs; trail reshapes live.

**Node marker**: diamond at burn start position. For future finite burns: also mark
burn end, connect with highlighted arc.


### Panels (egui)

**Top bar: Time Control**
- Pause/resume (Space)
- Warp level buttons (,/.)
- Mission elapsed time: `T+ Xd Xh`
- Warning when approaching a node (auto-drop warp to 1x)
- "Prediction limit reached" indicator if budget was exhausted

**Left panel: Orbital Info**
- Reference body name (from `LiveDominantBody`)
- Altitude (distance to reference body surface), velocity (relative to reference)
- Semi-major axis, eccentricity, inclination, period, periapsis, apoapsis
- Orbit trend indicator if the prediction showed precession ("orbit raising/decaying")

**Right panel: Encounter Info** (when target set and encounter exists)
- As described in encounter info panel section above
- Mini orbit diagram

**Bottom panel: Maneuver Node Editor** (when a node is selected)
- Three sliders + number inputs: prograde, normal, radial delta-v
- Total delta-v magnitude
- Estimated burn duration (from `ShipConfig.thrust_newtons`)
- Time until node
- Delete button, warp-to-node button
- Later: burn mode toggle (impulse vs finite)


---


## Data Flow

```
Input (keys, mouse)
  │
  ├─> ManeuverEvent
  │     │
  │     v
  │   ManeuverPlan (modified, dirty=true)
  │     │
  │     v
  │   run_prediction
  │     │  clone SimState
  │     │  propagate with phase machine + burns
  │     │  detect encounters
  │     │  find closest approaches
  │     v
  │   PredictionCache + TargetBody.encounters
  │     │
  │     v
  │   Rendering
  │     │  frame-relative trail drawing
  │     │  node handles
  │     │  encounter markers + panel
  │     │  closest approach markers
  │
  ├─> Camera controls (yaw, pitch, zoom, focus, target)
  │
  └─> Time controls (warp, pause)

Each frame:
  SimClock tick
    │
    v
  step_simulation
    │  integrator + force model + active burns
    │  substep splitting at impulse times
    v
  update_live_dominant_body
    │
    v
  sync_transforms (camera-relative f32 positions)
```


---


## Startup State

- Sun at origin, zero velocity
- Earth at 1 AU on +X, orbital velocity on +Z
- Moon at Earth + 384,400 km on +X, Earth velocity + 1,022 m/s on +Z
- Ship in 400 km LEO above Earth, circular orbit velocity

Camera focuses on ship, distance ~20,000 km. `active_frame` = Earth (index 1).
Time warp 1x, paused, so the player can orient before unpausing.


---


## Extension Roadmap

Out of scope now. The architecture supports all of these without restructuring.

- **Finite burns**: implement `FiniteBurn` behind `BurnModel`. Prediction already
  handles it. Add burn-arc rendering and duration indicator.
- **Fuel and staging**: expand `ShipConfig`. `FiniteBurn` checks fuel. Node editor
  shows delta-v budget.
- **Additional forces**: implement `ForceModel` for drag, J2, solar pressure.
  Compose with `CompositeForceModel`.
- **Adaptive integrator**: `DormandPrinceIntegrator` behind `Integrator` trait.
- **Multiple vessels**: more ship entries in `SimState`, each with own
  `ManeuverPlan` and `PredictionCache`.
- **Flight view**: second camera mode with meshes, surfaces, atmosphere. Sim
  layer untouched.
- **Save/load**: serialize `SimState`, `SimClock`, `ManeuverPlan`, body metadata.
  All plain data. `BurnModel::clone_box()` pattern extends to serialization.
- **Networking**: sim is deterministic. Send `ManeuverPlan` diffs.


---


## Testing


### Philosophy

| Layer               | Test type       | Bevy needed? | Difficulty | Value   |
|---------------------|-----------------|--------------|------------|---------|
| Simulation core     | Unit tests      | No           | Easy       | Highest |
| Prediction pipeline | Unit tests      | No           | Easy       | High    |
| ECS integration     | System tests    | Headless     | Medium     | Medium  |
| UI / Rendering      | Visual / manual | Full app     | Hard       | Low     |

80% of test effort goes to layers 1 and 2. They're pure Rust, fast to test, and
where bugs are most dangerous.


### Test helpers

```rust
/// Two-body: Earth at origin, test mass in circular orbit.
fn circular_orbit_state(altitude_km: f64) -> (SimState, f64)  // (state, period)

/// Two-body: Earth at origin, test mass in elliptical orbit.
fn elliptical_orbit_state(periapsis_km: f64, apoapsis_km: f64) -> (SimState, f64)

/// Three-body: Sun-Earth-Moon at known epoch.
fn sun_earth_moon_state() -> SimState

/// Full scenario: Sun-Earth-Moon + ship in LEO.
fn leo_scenario() -> (SimState, usize)  // (state, ship_index)

/// Full scenario: ship on trans-lunar trajectory.
fn translunar_scenario() -> (SimState, usize)
```

### Tolerance constants

```rust
const POSITION_TOLERANCE_M: f64 = 1.0;
const VELOCITY_TOLERANCE_MS: f64 = 0.001;
const ENERGY_RELATIVE_TOLERANCE: f64 = 1e-8;
const ANGLE_TOLERANCE_RAD: f64 = 0.01;
const ORBIT_CLOSURE_TOLERANCE_M: f64 = 1000.0;
```

Use as defaults. Individual tests may use tighter or looser tolerances as
parameters.


### Layer 1: Simulation core tests

#### Integrator

**Circular orbit conservation** (write first, highest value single test)

Two-body: Earth + test mass in circular LEO. Step for one orbital period. Assert
position returns to start (< 1 m error at dt=10s), radius stays bounded, specific
orbital energy conserved (< 1e-6 relative error). Run at dt = 1s, 10s, 100s.
Verify convergence rate: halving dt should reduce error by ~16x (4th order).

Catches: sign errors in gravity (immediate divergence), factor-of-two errors
(wrong orbit shape), integrator instability (energy growth), "does it work at all."

```
test_circular_orbit_conservation_dt_1s
test_circular_orbit_conservation_dt_10s
test_circular_orbit_conservation_dt_100s
test_circular_orbit_convergence_rate
```

**Elliptical orbit** (e = 0.5). Step for one period. Check energy and angular
momentum conservation. Catches errors at periapsis where acceleration changes
rapidly.

```
test_elliptical_orbit_energy_conservation
test_elliptical_orbit_angular_momentum_conservation
```

**Hyperbolic trajectory**. Step until far away. Assert: doesn't get captured,
asymptotic velocity approaches expected `v_inf = sqrt(-GM/a)`, energy conserved.

```
test_hyperbolic_trajectory
```

**Three-body sanity**. Sun-Earth-Moon, step for one lunar month. Assert: Moon
completes ~one orbit around Earth, Earth completes ~1/12 solar orbit, no body
gains/loses significant energy. Qualitative check only (no closed-form reference).

```
test_three_body_qualitative_behavior
```

**Parameterized integrator tests**. Structure all the above so they accept a
`&dyn Integrator` parameter, enabling reuse when Dormand-Prince is added.


#### Force model

```
test_gravity_symmetry            // Newton's third law: acc[0] = -acc[1] * m1/m0
test_gravity_analytical_leo      // known GM/r^2 at LEO altitude
test_gravity_analytical_geo      // at GEO altitude
test_gravity_analytical_moon     // at Moon distance
test_gravity_superposition       // three bodies in line, middle body acceleration
test_gravity_exclusion_mask      // masked bodies produce same result for non-excluded
test_gravity_precision_at_1au    // catastrophic cancellation check at large coords
```

The precision test is important: Earth at (1.496e11, 0, 0) and ship at
(1.496e11, 6.771e6, 0). The relative position computation must not lose
significant digits. Compare against analytical reference.


#### Burn model

```
test_impulse_before_window       // returns None
test_impulse_during_window       // returns correct world-space acceleration
test_impulse_after_window        // returns None
test_impulse_straddle_substep    // substep straddles impulse time
test_impulse_exact_boundary      // impulse time = substep boundary
```

**Orbital frame:**

```
test_orbital_frame_orthonormality      // three vectors are orthonormal
test_orbital_frame_prograde_parallel   // prograde is along velocity
test_orbital_frame_at_multiple_angles  // 0°, 90°, 180°, 270° of orbit
test_orbital_frame_eccentric_orbit     // correct at various true anomalies
```

**End-to-end burn tests:**

```
test_prograde_impulse_raises_apoapsis
test_retrograde_impulse_lowers_apoapsis
test_normal_impulse_changes_inclination
test_radial_impulse_rotates_apse_line
```

Set up circular orbit, apply impulse, verify resulting orbital elements. If
`test_prograde_impulse_raises_apoapsis` passes, the orbital frame, burn model,
and integrator are all working together.


#### Orbital math

```
test_keplerian_elements_circular
test_keplerian_elements_eccentric
test_keplerian_elements_polar
test_keplerian_elements_retrograde
test_keplerian_elements_degenerate_circular_equatorial  // must not produce NaN
test_keplerian_elements_degenerate_exactly_circular     // undefined ω
test_orbital_period_iss            // ~92 min
test_orbital_period_moon           // ~27.3 days
test_orbital_period_earth          // ~365.25 days
test_hill_radius_earth             // ~1.5 million km
test_hill_radius_moon              // ~60,000 km
```


### Layer 2: Prediction pipeline tests

Still pure Rust, no Bevy. Construct `SimState`, call `predict()` directly.


#### Trail generation

```
test_closed_orbit_trail
```
Circular orbit, no nodes, no target. Trail should terminate after ~one orbit.
Last point should be near first. Total trail time ≈ orbital period.

```
test_trail_segment_count_with_node
```
Single impulse node midway through orbit. Result should have two segments.
Spatial continuity at the boundary (last point of seg 0 ≈ first point of seg 1).

```
test_trail_body_positions_stored
```
Verify `body_positions[i][j]` matches an independently propagated body j at
time `times[i]`.

```
test_dominant_body_transitions
```
Trans-lunar trajectory. Verify `dominant_body` transitions from Earth to Moon at
approximately the right distance.

```
test_hohmann_transfer_prediction
```
Two nodes: prograde at LEO, retrograde at target altitude. Three segments.
Middle segment is elliptical transfer. Final segment is approximately circular
at target altitude. Integration test for the full planning pipeline.


#### Termination conditions

```
test_collision_terminates         // retrograde burn into Earth, stops at surface
test_escape_terminates            // hyperbolic trajectory, stops after Hill sphere
test_budget_exhaustion            // tiny budget, verify exact step count
test_precessing_orbit_still_closes // LEO with Moon perturbation, tolerance check
test_orbit_closure_with_pending_node // node ahead: don't close early
test_target_extends_prediction    // target set: prediction runs past orbit closure
```

The `test_precessing_orbit_still_closes` is critical. In N-body, no orbit truly
closes. The tolerance must be tuned so realistic LEO orbits (with lunar
perturbation) still trigger closure rather than running to budget.


#### Phase state machine

```
test_phase_initial_orbiting       // LEO starts in Orbiting phase
test_phase_orbit_to_transfer      // prograde burn to escape velocity
test_phase_transfer_to_encounter  // trajectory enters Moon's Hill sphere
test_phase_encounter_to_capture   // eccentricity < 1 at closest approach
test_phase_encounter_to_flyby     // exits Hill sphere on hyperbolic path
test_phase_encounter_collision    // periapsis below surface
test_phase_escape_to_new_orbit    // lunar escape transitions to Earth orbit
test_phase_orbit_closure          // radial line crossing with crossed_half guard
test_phase_extend_count           // "predict further" overrides closure
```


#### Encounter detection

```
test_lunar_encounter_detected     // trans-lunar trajectory finds encounter
test_no_false_encounter_leo       // stable LEO with Moon target: no encounter
test_encounter_elements_accuracy  // eccentricity, periapsis match analytical
test_encounter_capture_status     // correct Flyby vs Captured vs Impact
test_encounter_detection_coarse_dt // grazing flyby detected even at coarse dt
test_closest_approach_interpolation // between-point check catches skipped passes
```

The `test_encounter_detection_coarse_dt` guards against the most dangerous failure:
adaptive dt skipping over a Hill sphere. Set up a grazing flyby, force coarse dt,
verify detection still works.


#### Frame-relative rendering data

```
test_frame_relative_circular_orbit // subtract Earth positions from ship trail,
                                   // result should approximate a circle in 2D
test_frame_blend_continuous        // blended positions are spatially continuous
                                   // across transition zone
test_frame_blend_no_self_intersection // blended trail doesn't cross itself
```


#### Adaptive step size

```
test_adaptive_dt_periapsis_resolution // eccentric orbit: more points near periapsis
test_adaptive_dt_budget_respected     // total steps ≤ max_steps
test_adaptive_dt_target_bias          // finer dt when approaching target body
```


### Layer 3: ECS integration tests

Headless Bevy: `App::new()` + `MinimalPlugins`. Insert resources, add systems,
call `app.update()`, query world.

```rust
fn make_test_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app
}
```

```
test_physics_advances_with_warp      // sim_clock.time increases by warp * frame_dt
test_paused_no_advancement           // positions unchanged
test_burn_executes_at_correct_time   // impulse applied when sim_clock passes node time
test_burn_not_executed_before_time   // node ahead: velocity unchanged
test_substep_splitting_at_impulse    // impulse at t=100, substep 99..101: splits correctly
test_transform_sync_camera_relative  // entity Transform matches (sim_pos - cam_pos) * SCALE
test_dominant_body_near_earth        // ship at LEO: LiveDominantBody = Earth
test_dominant_body_near_moon         // ship near Moon: LiveDominantBody = Moon
test_active_frame_follows_dominant   // camera on ship: active_frame tracks dominant body
test_place_node_event                // PlaceNode -> ManeuverPlan gains entry
test_adjust_node_event               // AdjustNode -> delta_v updated, dirty = true
test_delete_node_event               // DeleteNode -> plan empty
test_dirty_flag_cleared_after_prediction // run_prediction clears dirty
test_prediction_reruns_on_dirty      // dirty=true triggers prediction update
```


### Known failure modes and high-value tests

Ordered by likelihood * severity. Tests should specifically target these.

**1. Energy drift (high likelihood, high severity)**

RK4 drifts over many orbits. Long time warps accumulate error.

Detection: energy conservation tests at multiple timescales. The convergence rate
test catches whether the integrator is truly 4th order.

Mitigation: if drift becomes a problem, switch live sim to symplectic integrator
(bounded energy error). Keep RK4 for short-term prediction (accuracy > conservation).

**2. Impulse timing granularity (high likelihood, medium severity)**

At high warp, substeps are large. A periapsis burn lands hundreds of km past the
intended point.

Detection: `test_substep_splitting_at_impulse`.

Mitigation: split substeps at exact impulse time. Implement from the start.

**3. Orbit closure false negatives (medium likelihood, medium severity)**

Precessing orbit never exactly closes. Too-tight tolerance = trail runs to budget
(messy spiral). Too-loose = truncates early.

Detection: `test_precessing_orbit_still_closes`.

Mitigation: radial-line-crossing with radial distance tolerance. Tune against
realistic test cases.

**4. Floating-point cancellation (medium likelihood, high severity)**

Subtracting near-equal large DVec3 values loses digits. Ship at (1.496e11, 6.4e6, 0)
relative to Earth at (1.496e11, 0, 0).

Detection: `test_gravity_precision_at_1au`.

Mitigation: f64 gives ~15 digits, so at 1 AU we get sub-mm precision. Should be
fine, but test proves it.

**5. Frame transition rendering artifacts (medium likelihood, low severity)**

Lerp between frames can produce visual glitches if transition zone is too short
or bodies are moving fast relative to each other.

Detection: `test_frame_blend_continuous`, `test_frame_blend_no_self_intersection`.

Mitigation: transition zone proportional to gravitational overlap region.

**6. Encounter detection misses (low likelihood, high severity)**

Trail skips over Hill sphere if dt is too coarse. Grazing flyby missed entirely.

Detection: `test_encounter_detection_coarse_dt`.

Mitigation: adaptive dt near bodies + between-point interpolation check in
`find_closest_approaches`.

**7. Node position drift under precession (low likelihood, medium severity)**

Node placed at specific orbital position. Orbit precesses as sim advances, so
the node's position on the re-predicted trail shifts.

Detection: place node, advance sim, re-predict, check node position stability.

Mitigation (future): store intended orbital geometry (true anomaly), adjust node
time to maintain geometry.


### Regression snapshots

For complex outputs (full prediction trails), save as JSON on first run. Compare
on subsequent runs. Catches unintentional behavioral changes during refactoring.

```
test_regression_leo_prediction
test_regression_translunar_prediction
test_regression_hohmann_transfer
```


### Test priority order

What to write first, by value-per-effort:

1. `test_circular_orbit_conservation_dt_10s` (if this fails, nothing works)
2. `test_gravity_symmetry` + `test_gravity_analytical_leo`
3. `test_prograde_impulse_raises_apoapsis` (end-to-end burn)
4. `test_closed_orbit_trail` (prediction termination)
5. `test_collision_terminates`
6. `test_phase_orbit_closure` + `test_precessing_orbit_still_closes`
7. `test_hohmann_transfer_prediction` (full planning pipeline)
8. `test_lunar_encounter_detected`
9. `test_trail_body_positions_stored` + `test_frame_relative_circular_orbit`
10. `test_physics_advances_with_warp` + `test_burn_executes_at_correct_time`

Tests 1-6 must exist before any Bevy code is written. They validate the sim core
and prediction pipeline. If an agent is implementing layer 1 or 2, these tests are
the acceptance criteria for "done."


---


## Implementation Order

A single ordered checklist. Each step has clear acceptance criteria. Steps marked
with test names must pass those tests before moving on.

**Phase A: Simulation core** (no Bevy, pure Rust + tests)

1. `SimState`, `Integrator` trait, `RK4Integrator`, `ForceModel` trait,
   `NBodyGravity`.
   Tests: `test_circular_orbit_conservation_*`, `test_circular_orbit_convergence_rate`,
   `test_elliptical_orbit_*`, `test_gravity_*`.

2. `BurnModel` trait, `ImpulseBurn`, `orbital_frame()`, `orbital_elements()`,
   `ManeuverNode`, `NodeId`.
   Tests: `test_impulse_*`, `test_orbital_frame_*`, `test_prograde_impulse_raises_apoapsis`,
   `test_keplerian_elements_*`.

3. `hill_radius()`. Verify against known values for Earth and Moon.

**Phase B: Prediction pipeline** (no Bevy, pure Rust + tests)

4. `PredictionPhase` state machine, `PredictionConfig`, `TrailSegment`
   (with `body_positions`), `PredictionResult`.
   `predict()` function with phase machine and termination conditions.
   Tests: `test_closed_orbit_trail`, `test_collision_terminates`,
   `test_escape_terminates`, `test_phase_*`.

5. Trail segmentation at maneuver nodes. Adaptive step size.
   Tests: `test_trail_segment_count_with_node`, `test_hohmann_transfer_prediction`,
   `test_adaptive_dt_*`.

6. Encounter detection: `detect_encounters()`, `find_closest_approaches()`.
   Tests: `test_lunar_encounter_detected`, `test_no_false_encounter_leo`,
   `test_encounter_*`.

**Phase C: Bevy scaffold** (visual, iterate manually)

7. Bevy app with `MinimalPlugins` (for ECS tests) and `DefaultPlugins` (for
   running). Spawn Sun/Earth/Moon/Ship entities. Wire `PhysicsState`, `SimClock`,
   `LiveDominantBody` resources. `step_simulation` system. Body markers and ship
   dot via Gizmos. Camera-relative transform pipeline.
   Tests: `test_physics_advances_with_warp`, `test_paused_no_advancement`,
   `test_transform_sync_camera_relative`.
   Visual check: Earth orbits Sun at high warp.

8. Trail rendering. Read `PredictionCache`, draw linestrips. Color by dominant
   body per point.
   Tests: `test_trail_body_positions_stored`, `test_dominant_body_transitions`.
   Visual check: ship traces clean ellipse around Earth.

**Phase D: Reference frame rendering**

9. Frame-relative trail rendering: subtract `body_positions[i][frame_body]`
   before drawing. `CameraFocus.active_frame`.
   Tests: `test_frame_relative_circular_orbit`.
   Visual check: orbit is a clean static ellipse when camera focuses on Earth.

10. Frame transition blending at dominant body switches.
    Tests: `test_frame_blend_continuous`, `test_frame_blend_no_self_intersection`.
    Visual check: trans-lunar trajectory smoothly morphs between Earth and Moon frames.

**Phase E: UI**

11. Time controls: warp, pause, egui top bar, mission elapsed time.
12. Orbital info panel: elements relative to `LiveDominantBody`.

**Phase F: Maneuver planning**

13. Node placement: trail-to-screen projection, click to place.
    Test: `test_place_node_event`.
14. Node handles: six directional arrows, drag to adjust, prediction re-runs.
    Tests: `test_adjust_node_event`, `test_dirty_flag_cleared_after_prediction`.
15. Multi-node chaining. Validation target: Hohmann transfer to the Moon.
16. Node execution: activate burns in `step_simulation` when time enters window.
    Substep splitting at impulse time.
    Tests: `test_burn_executes_at_correct_time`, `test_substep_splitting_at_impulse`.

**Phase G: Encounters**

17. Target system: right-click to target, biased prediction, ghost trail.
18. Encounter info panel with orbital elements at closest approach.
19. Mini orbit diagram (2D projected encounter in target's frame).
20. Interactive encounter feedback: panel updates during node adjustment.

**Phase H: Polish**

21. Trail fade, camera focus transitions, warp-to-node, auto-warp-drop near nodes.
22. Closest approach markers and collision warnings.
23. "Predict further" button.
24. Orbit trend indicator ("orbit raising"/"decaying").
