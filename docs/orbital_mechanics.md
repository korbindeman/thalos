# Thalos: Orbital Mechanics

The authoritative design for physics, trajectory prediction, maneuver
planning, and the orbital-map UI. Supersedes `docs/DESIGN.md`,
`docs/new_physics.md`, and `docs/stale/orbital-mechanics-mvp-spec.md` —
those exist only for reference.

---

## 1. Vision

Thalos is an orbital-mechanics sandbox. The player looks at a clean 3D map
of the solar system, sees their ship on a deterministic future trajectory,
and places maneuver nodes to reshape it. The trajectory is drawn as a single
continuous curve, each segment rendered in its locally correct reference
frame, so a low orbit reads as a clean ellipse around its planet and an
interplanetary transfer reads as a clean arc through space.

Encounters — flying past a body, capturing into its orbit — are surfaced
in the map as **ghost bodies**: translucent copies of the real body at the
position it will occupy when the ship arrives. The player plans burns
relative to the ghost. There is no separate encounter panel, no "target"
mode, no modal UI; the ghost is the UX.

### Core loop

1. Live simulation advances the ship each frame under the gravity of its
   current SOI body plus any active thrust.
2. Trajectory prediction propagates the ship forward through every
   scheduled maneuver, producing a chain of **legs** and a set of ghost
   bodies for every encounter along the way.
3. The player places a maneuver node by clicking the trajectory. A selected
   node exposes prograde / normal / radial handles in the local frame of
   whichever body (or ghost) owns the SOI at that point.
4. Editing a node re-triggers prediction from that node onward. The
   trajectory, the ghosts, and any downstream nodes update live.
5. When sim time reaches a node, its burn executes in the live simulation.
   The ship follows the predicted path to tight numerical agreement; the
   two differ only by the normal drift a finite-Δv burn produces against an
   impulsive-burn approximation — but we *don't* make that approximation:
   both live and predicted trajectories integrate the same finite burn with
   RK4, so they stay in lockstep.

---

## 2. Design invariants

These must not drift, regardless of future extensions.

- **Deterministic ship trajectory.** No stochastic terms, no adaptive state
  that depends on wall-clock time. A given maneuver plan always produces
  the same trajectory. There is no uncertainty cone in the UI — the
  trajectory is not uncertain.
- **One [`ShipPropagator`] used everywhere.** Live stepping and prediction
  both call the same propagator. If the two ever diverge numerically,
  "where the ship is" drifts away from "where it was predicted to be" and
  the planning UI becomes a lie.
- **[`BodyStateProvider`] is the boundary between body motion and everything
  else.** Physics, prediction, and rendering never assume how bodies move;
  they query `query_body(id, t)`. The current implementation is analytical
  Keplerian (`PatchedConics`).
- **Patched conics.** The ship feels gravity from exactly one body at a
  time — the body whose SOI currently contains it. This is a deliberate
  modelling choice (§3). Secular perturbations, Lagrange-point gameplay,
  and three-body behaviours are accepted out of scope until we re-introduce
  an N-body propagator behind the trait (§9).
- **Effects are pure functions of `(state, time, body_states)`.** The only
  non-gravity effect today is finite thrust during a burn window. Future
  drag and SRP slot in the same way.
- **Physics crate has no Bevy dependency.** All physics stays in
  `thalos_physics` and is testable headlessly. The game crate is
  presentation and input.

---

## 3. Physics model

### 3.1 Body motion: `BodyStateProvider`

All consumers query body positions through this trait:

```rust
trait BodyStateProvider {
    fn query_into(&self, time: f64, out: &mut BodyStates);
    fn query_body(&self, id: BodyId, time: f64) -> BodyState;
    fn body_count(&self) -> usize;
    fn time_span(&self) -> f64;
    // ... plus convenience methods with sensible defaults.
}
```

**Current implementation: `PatchedConics` (analytical Keplerian).** Each
non-root body has Keplerian elements around its parent; queries evaluate
the closed-form solution and chain through parents (moon → planet → star)
to produce a heliocentric state. Bodies are perfectly periodic, perfectly
deterministic, zero-cost to precompute (nothing is precomputed — every
query is a closed-form evaluation).

This is a deliberate choice. Bodies staying exactly on rails is a
*feature*: the player can plan multi-year missions against a stable
backdrop, and a saved game always resumes with the bodies where they were.

**Future: precomputed N-body ephemeris for bodies (§9).**

### 3.2 Ship state

```rust
struct StateVector {
    position: DVec3,   // heliocentric inertial, metres
    velocity: DVec3,   // m/s
}
```

Everything is SI, double precision, heliocentric inertial throughout the
physics. Coordinate conversions live only at the rendering boundary.

### 3.3 Ship propagation: `ShipPropagator`

```rust
trait ShipPropagator: Send + Sync {
    fn coast_segment(&self, req: CoastRequest) -> SegmentResult;
    fn burn_segment(&self, req: BurnRequest) -> SegmentResult;
    fn soi_body_of(
        &self,
        position: DVec3,
        time: f64,
        ephemeris: &dyn BodyStateProvider,
        bodies: &[BodyDefinition],
    ) -> BodyId;
}
```

Each call propagates **one segment** — either pure coast or a finite-burn
window — and terminates at the first of: target time, SOI entry, SOI exit,
collision, stable-orbit closure (coast only), or burn end (burn only).
The caller loops across terminators to traverse multi-SOI legs.

**Default implementation: `KeplerianPropagator`.**

- **Coast segments** are analytical. The ship's state is converted to
  Keplerian elements in the SOI body's frame, advanced by solving Kepler's
  equation (Newton–Raphson, ~1e-13 convergence), and converted back. An
  elliptic-vs-hyperbolic branch handles escape trajectories; the parabolic
  edge is a linear fallback (rare; upgradeable to Barker's equation).
- **Burn segments** are RK4 with a fixed substep (default 1.0 s) under
  SOI-body gravity plus constant thrust in the ship's local P/N/R frame.
  One sample per substep.
- **SOI crossings** are detected at each sampled boundary by a distance
  check, then refined by bisection on the analytic signed-distance function
  to ~1 µs precision.
- **Collisions** use the same bisection against `|ship − SOI_body| − radius`.

### 3.4 SOI determination

`soi_body_of` walks every body and returns the one with the smallest SOI
radius whose sphere currently contains the ship. The star's SOI is
infinite, so it always matches as a fallback. `BodyDefinition::soi_radius_m`
is precomputed at load time from `a · (m / M_parent)^(2/5)` (Hill sphere).

### 3.5 Live stepping

`Simulation::step` is event-driven: each iteration finds the nearest upcoming
boundary (burn start, burn end, next node, target frame time) and invokes
the propagator's coast or burn segment accordingly. Because coast segments
are analytical, a long warp tick (say, 10⁶× real time) completes in a
single call without any substep cost. Burns cap per-substep work, so a
long burn at high warp pays RK4 per substep — but burns are bounded in
sim time.

There is no "observation warp mode"; any warp speed is fine because coast
propagation is closed-form.

---

## 4. Trajectory prediction

Prediction runs synchronously on the main thread. Under analytical
propagation it is cheap enough that a background worker is not needed.

### 4.1 When prediction runs

- On maneuver-plan edits (node added / moved / deleted / Δv changed).
- When the cached prediction ages beyond `prediction_stale_after` seconds
  of sim time, so the drawn trail does not drift off the live ship.

### 4.2 Legs and segments

```rust
struct FlightPlan {
    initial_state: StateVector,
    initial_time: f64,
    legs: Vec<Leg>,                // one per maneuver interval
    segments: Vec<NumericSegment>, // flat mirror (burn + coast interleaved)
    events: Vec<TrajectoryEvent>,  // SOI crossings, apsides, impacts
    encounters: Vec<Encounter>,    // aggregated encounter windows
    approaches: Vec<ClosestApproach>,
    baseline: Option<NumericSegment>, // full coast with no maneuvers
}

struct Leg {
    start_state: StateVector,
    start_time: f64,
    applied_delta_v: Option<DVec3>,
    burn_segment: Option<NumericSegment>,
    coast_segment: NumericSegment,
}
```

For each leg the propagator is invoked twice: optionally for the burn
window, then for the coast up to the next node (or horizon). Inside each
call the propagator loops across SOI transitions, so a single
`NumericSegment` can carry samples from multiple SOIs; consecutive samples
whose `anchor_body` differs identify a Hill-sphere crossing.

### 4.3 Samples

```rust
struct TrajectorySample {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    anchor_body: BodyId,   // the SOI body at this sample
    ref_pos: DVec3,        // anchor_body's position at `time`, cached
}
```

`ref_pos` is written once at sample time and read by the renderer to place
each sample in its anchor-relative frame without a per-frame ephemeris
query.

### 4.4 Stable-orbit closure

A bound coast whose apoapsis fits inside the SOI and whose periapsis
clears the body surface is visualised as a closed loop by requesting
`stop_on_stable_orbit`. The propagator detects the condition analytically
from the orbital elements and terminates exactly after one period. The
last leg of a flight plan enables this so a parked orbit reads as a clean
closed ellipse.

### 4.5 Re-propagation scope

Editing node N re-propagates from leg N onward. Legs before N are
unchanged — their samples, encounters, and ghosts are preserved.

---

## 5. Maneuver nodes

A node is a finite burn at a scheduled time, editable in the dominant
body's (or ghost's) local frame.

### 5.1 Data model

```rust
struct ManeuverNode {
    id: Option<u64>,
    time: f64,                // burn start, sim seconds
    delta_v: DVec3,           // (prograde, normal, radial), m/s
    reference_body: BodyId,   // defines the local frame
}
```

A non-zero-Δv node becomes a `ScheduledBurn` at propagation time, which
the propagator integrates as a finite burn under RK4.

### 5.2 Burn model

Linear thrust: constant acceleration in the direction of `delta_v` (as a
P/N/R vector) for `|Δv| / a_thrust` seconds. The direction is recomputed
every substep from the ship's live state relative to `reference_body`, so
a burn whose duration is a meaningful fraction of the orbital period tracks
the rotating local frame correctly.

The trajectory curves smoothly across the burn window rather than kinking.
This is the same machinery in both prediction and execution, so the
trajectory the player plans is the trajectory they fly.

### 5.3 Placement, editing, frames

- **Place** — click on the trajectory; the node is created at the sample
  time nearest the click with `reference_body` set to the sample's anchor
  body (or a nearby ghost's body).
- **Slide** — a central handle drags the node along the trajectory in time.
- **Shape** — six directional handles in the node's local frame.
- **Shift by orbits** — on a stable coast, the node can be shifted forward
  by discrete orbital periods.
- **Delete** — merges the adjacent legs; prediction re-runs from the
  preceding node.

---

## 6. Ghost bodies — the encounter UX

Ghost bodies are the primary way encounters are communicated. A ghost is
a translucent rendering of a real body at the position it will occupy at
a specific future time, displayed alongside the corresponding segment of
the predicted trajectory.

### 6.1 When a ghost appears

A ghost is instantiated for every encounter the propagator produces — an
`Encounter` aggregated from the SOI-entry events emitted when a leg's
propagation crosses into a body's Hill sphere. The ghost's timestamp is
the SOI entry time.

Ghosts also appear for the player's selected target when the trajectory
passes near it without entering its SOI (`ClosestApproach`).

### 6.2 Positioning and visuals

A ghost's position is `BodyStateProvider::query_body(body, t_ghost)`. As
live time advances toward `t_ghost`, the real body converges on the ghost;
at `t_ghost` they coincide and the ghost dissolves.

Visually: the real body's mesh / colour, desaturated and translucent, with
a small label showing name + relative time. No connecting line to the real
body.

### 6.3 Ghosts and maneuver nodes

Clicking on the trajectory near a ghost anchors the new node to that
encounter. Handles are computed in the ghost's local frame (P/N/R relative
to the *future* body position and the ship's velocity relative to it at
that point). A burn planned at a ghost is thus "at Moon arrival", not "at
heliocentric time T + 3d 14h" — so the node's Δv survives upstream edits
that shift the encounter's absolute time.

---

## 7. Reference frames

The simulation runs in a single heliocentric inertial frame. All frames
below are presentation only.

### 7.1 Per-SOI anchoring

Each sample's **anchor body** is the body whose SOI currently contains the
ship; `ref_pos` is that anchor's heliocentric position at the sample's
time. The renderer computes

```
render_pos = (sample.position − sample.ref_pos) + pin(anchor)
```

where `pin(anchor)` is the anchor's current position, or — when the anchor
is an encounter body whose ghost is active — the ghost's fixed world
position. This keeps each SOI span's trajectory and the corresponding ghost
coincident in world space: a capture arc inside a moon renders as a clean
ellipse around the ghost moon, not as epicycles in the departure body's
frame.

When an SOI crossing flips `anchor_body` between two consecutive samples,
the renderer blends the line colour halfway between the two body colours
at the crossing — a soft handoff rather than a hard cut.

### 7.2 Camera

KSP-style orbit camera. Focus on any body, ghost, ship, or maneuver node.
Double-click or Tab cycles focus. Camera distance is logarithmic on the
scroll wheel; transitions interpolate the frame origin over ~0.5 s.

### 7.3 Float precision

World coordinates are f64 throughout physics and prediction. The render
transform pipeline does `(world_pos − camera_world_pos) * RENDER_SCALE` in
f64, then casts to f32 for Bevy. This preserves sub-metre precision at
interplanetary distances.

---

## 8. Performance targets

- **60 FPS** under all normal interaction (pan, zoom, node edit).
- **Prediction** completes synchronously on the main thread within ~1 ms
  for typical intra-system plans (analytical coast dominates).
- **Prediction horizon** reaches interplanetary transfers (weeks to months
  of game time) without sampling loss.
- **Body state query** < 1 µs — called thousands of times per prediction.
- **Live stepping** stays within frame budget at arbitrary warp because
  coast propagation is closed-form.

---

## 9. Future work (out of scope for MVP)

Each item slots in without restructuring.

- **Precomputed N-body ephemeris for bodies.** Offline: integrate the full
  N-body system forward, fit Chebyshev polynomials, serialise a binary
  asset. Runtime: a new `BodyStateProvider` implementation that
  binary-searches the segment table. Nothing else changes. Buys secular
  perturbations (moon inclination drift, resonant chains).
- **N-body ship propagator.** A second `ShipPropagator` implementation that
  integrates numerically under summed gravity. Would re-enable Lagrange
  points and three-body resonances for gameplay. The trait boundary is
  already in place; only `Simulation` and `trajectory/` consumers need
  swap to the new impl.
- **Additional effects:** atmospheric drag, solar radiation pressure, J₂
  oblateness — each adds another acceleration term inside the burn-segment
  integrator.
- **Richer burn models:** throttle curves, mass flow + Tsiolkovsky, gimbal,
  staging.
- **Intent-based nodes:** "circularise here", "capture at ghost" — solved
  analytically from the orbital elements and turned into Δv.
- **Multi-ship:** multiple vessels, each with its own plan and prediction.

---

## 10. Where to find things in code

- `crates/physics/src/types.rs` — `StateVector`, `TrajectorySample`, body
  definitions.
- `crates/physics/src/orbital_math.rs` — Cartesian ↔ Keplerian conversion,
  `propagate_kepler`.
- `crates/physics/src/ship_propagator.rs` — `ShipPropagator` trait,
  `KeplerianPropagator` with coast, burn, SOI transitions.
- `crates/physics/src/patched_conics.rs` — current `BodyStateProvider`.
- `crates/physics/src/simulation.rs` — `Simulation::step`, warp, prediction
  cache.
- `crates/physics/src/trajectory/` — flight-plan construction, phase
  machine, encounter aggregation.
- `crates/physics/src/maneuver.rs` — node data model, orbital-frame helper.
- `crates/game/src/bridge.rs` — ties `Simulation` into Bevy's frame loop.
- `crates/game/src/flight_plan_view/` — ghost-body lifecycle, trajectory
  rendering, view reconciliation.
- `crates/game/src/maneuver/` — node placement, editing handles, frame
  attachment.
