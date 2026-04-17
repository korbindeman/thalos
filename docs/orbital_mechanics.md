# Thalos: Orbital Mechanics

The authoritative design for physics, trajectory prediction, maneuver planning, and
the orbital map UI. Supersedes `docs/DESIGN.md`, `docs/new_physics.md`, and
`docs/stale/orbital-mechanics-mvp-spec.md` — those exist only for reference.

---

## 1. Vision

Thalos is an orbital mechanics sandbox. The player looks at a clean 3D map of the
solar system, sees their ship on a deterministic future trajectory, and places
maneuver nodes to reshape that trajectory. The trajectory is drawn as a single
continuous curve rendered per-segment in the locally correct reference frame, so
a low orbit reads as a clean ellipse around its planet and an interplanetary
transfer reads as a clean arc through space.

Encounters — flying past a body, capturing into its orbit, hitting a Lagrange
point — are surfaced directly in the map as **ghost bodies**: translucent copies
of the real body rendered at the position it will occupy when the ship arrives.
The player plans burns relative to the ghost. There is no separate encounter
panel, no "target" mode, no modal UI for encounters; the ghost is the UX.

### Core loop

1. Live simulation advances the ship each frame under summed gravity from all
   relevant bodies, plus any active thrust.
2. A background worker propagates the ship forward from the current state,
   through every scheduled maneuver, producing a trajectory and a set of ghost
   body markers for every encounter along the way.
3. The player places a maneuver node by clicking the trajectory. A selected node
   exposes prograde / normal / radial handles in the local frame of whichever
   body (or ghost) dominates the trajectory at that point.
4. Editing a node re-triggers prediction from that node onward. The trajectory,
   the ghosts, and any downstream nodes update live.
5. When sim time reaches a node, its burn executes in the live simulation. The
   ship follows the predicted path to within tiny numerical error.

---

## 2. Design invariants

These must not drift, regardless of future extensions.

- **Ship trajectory is deterministic.** No stochastic terms, no adaptive state
  that depends on wall-clock time. A given maneuver plan always produces the
  same trajectory. This is why there is no uncertainty cone in the UI — the
  trajectory isn't uncertain.
- **Same integrator and force model for live stepping and prediction.** If the
  two ever diverge numerically, "where the ship is" drifts away from "where it
  was predicted to be" and the planning UI becomes a lie. One `IntegratorConfig`,
  used everywhere.
- **`BodyStateProvider` is the boundary between body motion and everything
  else.** Physics, prediction, and rendering never assume how bodies move; they
  query `position(body_id, t)` and `velocity(body_id, t)`. The current
  implementation is analytical Keplerian (`PatchedConics`). A precomputed
  N-body ephemeris is a drop-in replacement (§8).
- **Effects are pure functions of `(state, time, body_states)`.** Gravity,
  thrust, and any future drag / SRP live behind the same interface and are
  called identically from live and prediction. This is what keeps live and
  predicted trajectories in lockstep.
- **Physics crate has no Bevy dependency.** All physics logic stays in
  `thalos_physics` and is testable headlessly. The game crate is presentation
  and input.
- **No sphere-of-influence patching in the physics.** The ship always feels
  summed gravity from every body whose contribution is above a culling
  threshold. Lagrange points, resonances, and three-body behaviors emerge from
  the integrator. SOI-like concepts (dominant body, Hill sphere) exist only as
  presentation metadata.
- **One rendering rule for the whole trajectory.** Every sample is drawn
  relative to the gravity-weighted barycenter at that sample — no segments,
  no phases, no transition zones in the renderer (§7.2). The phase machine
  (§4.2) decides *what samples are produced*; the renderer decides nothing.

---

## 3. Physics model

### 3.1 Body motion: `BodyStateProvider`

All consumers query body positions through this trait:

```rust
trait BodyStateProvider {
    fn position(&self, body: BodyId, t: f64) -> DVec3;
    fn velocity(&self, body: BodyId, t: f64) -> DVec3;
    fn mu(&self, body: BodyId) -> f64;
    fn parent(&self, body: BodyId) -> Option<BodyId>;
    fn bodies(&self) -> &[BodyId];
}
```

**MVP implementation: `PatchedConics` (analytical Keplerian).** Each body has a
Keplerian orbit around its parent. The provider evaluates the orbit equations
at the requested time and chains through parents (moon → planet → star) to
produce a heliocentric state. Bodies are perfectly periodic, perfectly
deterministic, and zero-cost to precompute (nothing is precomputed — every
query is a closed-form evaluation).

This is a deliberate choice. For gameplay, bodies staying exactly on their rails
is a *feature*: the player can plan multi-year missions against a stable
backdrop, and a saved game always resumes with the bodies where they were. The
lack of secular body-on-body perturbation is acceptable because the player's
trajectory still feels summed gravity from every body — so flybys, captures,
and L-point behavior all work correctly for the ship.

**Future: precomputed N-body ephemeris (§8).**

### 3.2 Ship state

```rust
struct StateVector {
    position: DVec3,   // heliocentric inertial, meters
    velocity: DVec3,   // m/s
    mass: f64,         // kg, decreases during burns (future)
}
```

Everything is SI, double precision, heliocentric inertial throughout the
physics. Coordinate conversions live only at the rendering boundary.

### 3.3 Forces

A force is a pure function of `(state, time, body_states)` returning an
acceleration. The simulation maintains an `EffectRegistry` with one distinguished
slot for gravity (which also returns the dominant body and perturbation ratio —
useful for presentation) and a list of additional effects.

**MVP effects:**
- **Gravity** — summed over all bodies above a culling threshold
  (acceleration contribution < 1e-9 m/s² culled).
- **Thrust** — linear constant-acceleration burn active during a maneuver's
  time window. See §5.

**Later:** atmospheric drag, solar radiation pressure, J₂ oblateness. Each is
another `Effect`. No integrator or prediction changes required.

### 3.4 Integrator

Hybrid, identical in live and prediction:

- **Symplectic leapfrog** as the default — fixed timestep, bounded energy
  drift over long coasts in a single gravity well.
- **Adaptive Dormand-Prince (RK45)** when the perturbation ratio crosses a
  threshold (≈0.01, with hysteresis) — shrinks the step automatically near
  encounters, Lagrange regions, and during burns.

Switching is per-step and transparent to callers. The integrator outputs a
sample per step carrying: time, state, dominant body, perturbation ratio, step
size.

### 3.5 Live stepping

The live simulation runs every frame with `dt = frame_dt * warp`, subdivided so
no substep exceeds the integrator's accuracy budget. When a substep would
straddle the end of a burn window, it is split exactly at the window boundary
so burn timing never slops into adjacent steps.

A hard maximum warp is enforced where the integrator can no longer complete a
frame's worth of stepping within budget. The warp UI never offers a level that
breaks the simulation.

---

## 4. Trajectory prediction

A background worker produces a deterministic forecast of the ship's future,
split into **legs** (one per inter-node interval, plus a final leg after the
last node). Each leg is a `Vec<TrajectorySample>`.

### 4.1 When prediction runs

- On maneuver plan edits (node added / moved / deleted / delta-v changed) —
  the current run cancels and restarts from the earliest affected node.
- When the live ship has drifted far enough from the start-of-prediction state
  that the forecast is stale (threshold: a few integrator steps of
  divergence).

Edits never block the main thread. Progressive refinement: a coarse pass
arrives fast (sub-100 ms) for visual feedback; finer passes replace it over
subsequent frames until the full plan is resolved.

### 4.2 Phase machine — how a leg terminates

Each leg propagates until one of a small set of terminal conditions fires.
This is what gives us "one revolution" for stable orbits and "just enough" for
transfers, without any visual-cone heuristic and without chasing tolerances.

```
Orbiting  → OrbitClosed          (one angular revolution wrt dominant body)
Orbiting  → Transfer             (specific energy wrt dominant body ≥ 0,
                                  or dominant body changes)
Transfer  → Encounter            (ship enters a body's Hill sphere)
Transfer  → Escaping             (monotonically departing, hyperbolic)
Encounter → Orbiting             (osculating eccentricity < 1 at closest
                                  approach — captured)
Encounter → Transfer             (exits Hill sphere on unbound path — flyby)
Encounter → Collision            (distance < body radius)
Escaping  → EscapeComplete       (beyond 3-5× Hill sphere of escaped body)
Any       → BudgetExhausted      (hard step cap — a safety net, not normal)
```

**Orbit closure is angular, not positional.** At entry to `Orbiting`, record
the radial unit vector from the dominant body to the ship. Terminate the leg
the second time that radial direction is crossed (after a half-orbit guard).
No position tolerance, no "is the ship back where it started" check.

In an N-body reality no orbit truly closes — the second crossing is generally
*not* at the same radius or along the same exact axis of the ellipse. That is
fine: we do not force the visual line to loop. We just draw from the current
start to the first angular return, and when the leg is re-predicted from a
new start state (because live time advanced, or the player edited a node),
the next "revolution" is drawn from the new anchor. Across many re-predictions
this reads as an orbit that is smoothly precessing / raising / decaying —
which is what is actually happening.

Edge cases:

- **Pending nodes downstream** — closure is suppressed until all scheduled
  nodes in this leg have been integrated through.
- **Manual extension** — a key press increments an extension counter; the
  next would-be terminal condition is ignored and the leg continues for one
  more phase cycle. The escape hatch for resonant flyby chains and
  multi-orbit views.

### 4.3 What prediction outputs

```rust
struct TrajectorySample {
    t: f64,
    state: StateVector,
    dominant_body: BodyId,
    perturbation_ratio: f64,
    step: f64,
    // Per-body gravitational weights at this sample,
    // wᵢ = aᵢ / Σⱼ aⱼ. Used by the renderer (§7.2).
    // Reuses the aᵢ values already computed by the gravity effect.
    body_weights: SmallVec<[(BodyId, f32); 4]>,
}

struct Leg {
    samples: Vec<TrajectorySample>,
    phase_trace: Vec<Phase>,      // per-sample, drives rendering style
    terminator: Terminator,        // OrbitClosed / Collision / Encounter / ...
    encounters: Vec<EncounterRef>, // ghost-body instantiation points, §6
    after_node: Option<NodeId>,
}

struct FlightPlan {
    legs: Vec<Leg>,
    ghosts: Vec<Ghost>,            // §6 — derived from encounters
}
```

The renderer and the maneuver UI consume `FlightPlan` directly. Nothing else
is needed; all the metadata is in the samples.

### 4.4 Adaptive step resolution

The adaptive integrator naturally densifies samples near periapsis, during
burns, and through encounters. No separate sampling pass is needed. A
between-sample interpolation check runs post-hoc to guarantee no grazing
Hill-sphere entry is missed by a coarse step.

---

## 5. Maneuver nodes

A node is a linear-thrust burn at a scheduled time, editable in the dominant
body's (or ghost's) local reference frame.

### 5.1 Data model

```rust
struct ManeuverNode {
    id: NodeId,
    t: f64,                        // burn start time, heliocentric
    dv_local: DVec3,               // (prograde, normal, radial) in frame at t
    frame: ManeuverFrame,          // which body/ghost this is anchored to
}

enum ManeuverFrame {
    Body(BodyId),                  // real body at time t
    Ghost(GhostId),                // ghost body, resolves to (BodyId, t_encounter)
    Lagrange(BodyPair, Point),     // future
}
```

The frame is **chosen at placement time** and stays attached to that body /
ghost as the player edits earlier nodes. If a ghost is invalidated by an
upstream edit (the encounter moves or disappears), the node's frame falls
back to the dominant body at the node's time, and the UI surfaces a warning.

### 5.2 Burn model (MVP)

Linear thrust: constant acceleration in the direction of `dv_local` converted
to world frame at burn start, for a duration computed from current mass and
thrust. Integrated through substep-by-substep like any other effect. The
trajectory curves smoothly across the burn window rather than kinking.

This matches the existing implementation. Future burn models (throttle curves,
mass flow, gimbal, finite-Isp) slot in behind a trait without touching the
integrator or prediction.

### 5.3 Placement, editing, frames

- **Place** — click on the trajectory line. The node is created at the sample
  time nearest the click; `frame` is set to whichever body dominates at that
  sample (or to a nearby ghost if the click is within its influence region).
- **Slide** — a central handle drags the node along the trajectory in time.
- **Shape** — six directional arrows (prograde/retrograde, normal/antinormal,
  radial-in/radial-out) in the node's local frame. Arrows facing toward the
  camera are culled to avoid occlusion. Drag to adjust the corresponding
  component of `dv_local`.
- **Shift by orbits** — in a stable leg, the node can be shifted forward by
  discrete orbital periods without moving visually, to schedule on a future
  pass.
- **Delete** — merges the adjacent legs; prediction re-runs from the preceding
  node.

### 5.4 Re-propagation scope

Editing node N re-propagates leg N and onward. Legs before N are unchanged —
their samples, encounters, and ghosts are preserved. Progressive refinement
applies to the re-propagated portion.

---

## 6. Ghost bodies — the encounter UX

Ghost bodies are the primary way encounters are communicated to the player.
A ghost is a translucent rendering of a real body at the position it will
occupy at a specific future time, displayed alongside the corresponding
segment of the predicted trajectory.

### 6.1 When a ghost appears

A ghost is instantiated for every encounter the phase machine identifies:
every time a leg's phase enters `Encounter { body_idx, .. }`. The ghost's
timestamp is the time of closest approach (or Hill-sphere entry, if the leg
terminates in collision or capture).

Ghosts also appear for **Lagrange points** associated with body pairs the
trajectory passes through. An L-point ghost is a non-physical marker showing
where L1/L2/L3/L4/L5 will be at the relevant time. This falls out of the same
machinery — a ghost is a `(anchor, t)` pair where anchor is either a body or
a Lagrange reference.

The same body can have **multiple ghosts** on a long trajectory (two flybys,
a flyby-and-capture chain). Each ghost is independent, timestamped, labeled
with `T+Δt`.

### 6.2 Positioning and visuals

A ghost's position is `BodyStateProvider::position(body, t_ghost)`. As live
time advances toward `t_ghost`, the real body converges on the ghost; at
`t_ghost` they coincide and the ghost dissolves (the ghost's encounter leg has
been consumed and re-predicted from the new state).

Visually: the real body's mesh / color, desaturated and translucent, with a
small label showing name + relative time. Dimmer than the real body; dimmer
than the trajectory line; brighter than background orbit lines. No connecting
line to the real body.

### 6.3 Ghosts and maneuver nodes

This is the mechanic the game is built around. **Ghosts are the anchors for
planning-frame maneuver nodes.**

- Clicking on the trajectory near a ghost snaps the new node's `frame` to
  `Ghost(ghost_id)`. The handles are computed in the ghost's local frame
  (prograde/normal/radial relative to the *future* body position and the
  ship's velocity relative to it at that point).
- A burn planned at a ghost is thus a burn "at Moon arrival" or "at the L4
  point encounter," not "at heliocentric time T + 3d 14h." It survives
  upstream edits that shift the encounter's absolute time, because the node
  re-resolves relative to its anchor.
- When the player looks at a ghost, they are looking at the geometry they'll
  actually fly through. A circularization burn planned on the ghost of the
  Moon produces a circular orbit around the Moon, not a heliocentric shape
  that happens to intersect lunar space.

### 6.4 Trajectory rendering near ghosts

Nothing special happens. The gravity-weighted rendering rule (§7.2) already
does the right thing: when the ship is inside the encounter body's Hill
sphere, that body's weight `w` approaches 1.0, so each sample renders
relative to the body's position *at the sample's time* — which, over the
short duration of an encounter, is essentially coincident with the ghost's
fixed position. The flyby reads as a clean hyperbola against the target,
the capture reads as a clean ellipse, and the ghost itself is the visual
anchor the player sees the trajectory curve around.

---

## 7. Reference frames

The simulation runs in a single heliocentric inertial frame. All frames below
are presentation only.

### 7.1 Dominant body

Computed per sample as the body contributing the largest gravitational
acceleration to the ship. Already on every `TrajectorySample`. Drives:

- Trajectory line color (body's defined color from `solar_system.ron`).
- Default maneuver-node frame when placed on that segment.
- Color blending at transitions between dominant bodies (smooth gradient
  driven by perturbation ratio — no hard SOI boundary).

The dominant body is a **labeling / coloring** signal. It does *not* drive
the rendering frame; the weighted rule below does that uniformly.

### 7.2 Rendering: gravity-weighted barycenter

The whole trajectory is drawn by one rule, applied per-sample, with no
segments, phases, or transition zones in the renderer:

```
wᵢ      = aᵢ / Σⱼ aⱼ                        // aᵢ = GMᵢ / |rᵢ − r_ship|²
ref(t)  = Σᵢ wᵢ · rᵢ(t)                     // gravity-weighted barycenter
render_pos(sample) = sample.pos − ref(sample.t)
```

The weights are exactly the per-body gravitational acceleration ratios on
the ship, already computed by the gravity effect during propagation and
cached on the sample (§4.3).

This is a single continuous transform that automatically does the right
thing in every regime:

| Regime                    | Dominant weight       | What you see                                |
|---------------------------|-----------------------|---------------------------------------------|
| Low orbit (e.g. LEO)      | w_primary ≈ 0.9994    | Clean ellipse around the primary            |
| Hill-sphere edge          | weights ~comparable   | Line smoothly transitions frames            |
| Interplanetary coast      | w_star ≈ 1            | Clean heliocentric arc                      |
| Encounter (inside Hill)   | w_target → 1          | Clean hyperbola / capture against target    |
| Lagrange region           | two weights ~equal    | Bounded halo / Lissajous wobble near L-pt   |

**Mathematical equivalence to per-frame blending.** If only bodies A and B
have non-negligible weight, the rule expands to

```
render_pos = w_A · (sample.pos − r_A) + w_B · (sample.pos − r_B)
```

— exactly the lerp between "rendered in A's frame" and "rendered in B's
frame." The weighted-barycenter rule is the principled, gravity-driven
generalization of transition-zone blending. There is no hand-authored zone
width, and the rule extends to three or more bodies for free (Lagrange
points, tight satellite systems).

**One-revolution rendering is a property of the leg, not the frame.** A
stable-orbit leg draws from its anchor state to the first angular return
(§4.2). Because bodies in N-body reality precess, the rendered ring may not
close; that is not patched over. When live time advances or a node is
edited, the leg re-anchors and the next revolution is drawn fresh.

**Optional toggle.** "Render everything relative to the focused body" — for
inspecting heliocentric geometry of a full transfer in one rigid frame.
Useful occasionally; the weighted rule is the default.

**What the rendered curve is and isn't.** The weighted frame is a
visualization; coordinates in it don't correspond to any one inertial frame.
Physics, orbital elements, and the ship's actual velocity all live in the
heliocentric inertial frame. The maneuver-node local frame (§5) is computed
from the dominant body's position and velocity, *not* from the weighted
ref — the ref is a rendering construct only.

### 7.3 Camera

KSP-style orbit camera. Focus on any body, ghost, ship, or maneuver node.
Double-click or Tab cycles focus. Camera distance is logarithmic on the
scroll wheel; transitions between focus targets smoothly interpolate the
frame origin over ~0.5 s so the trail visually morphs between frames rather
than jump-cutting.

### 7.4 Float precision

World coordinates are f64 throughout physics and prediction. The render
transform pipeline does `(world_pos - camera_world_pos) * RENDER_SCALE`
in f64, then casts to f32 for Bevy. This preserves sub-meter precision at
interplanetary distances.

---

## 8. Future: precomputed N-body ephemeris

The target end state for body motion. Not MVP.

The pipeline (offline):

1. Parse `solar_system.ron`, convert each body's Keplerian orbit at the epoch
   to a Cartesian state.
2. Integrate the full N-body system forward — star + planets + moons + minor
   bodies — for the gameplay time span (thousands of years) using a symplectic
   long-term integrator (e.g. WHFast via REBOUND, offline).
3. Fit piecewise Chebyshev polynomials to each body's trajectory (segment
   length and degree chosen per-body based on curvature; error-bounded).
4. Serialize as a single binary asset (~10–20 MB for a full solar system over
   10k years).

The runtime change is a **new `BodyStateProvider` implementation** that
binary-searches the segment table and evaluates Chebyshev coefficients. Nothing
else in the stack changes: physics, prediction, rendering, maneuver planning
continue calling `position(body, t)` and `velocity(body, t)` exactly as
before.

This is deliberately isolated behind the trait so the upgrade is a single
swap, not a refactor. What it buys us: secular perturbations (moon inclination
drift, resonant chains among small bodies), mission-realistic long-term
behavior, and a smooth story if we ever add additional massive bodies that
matter gravitationally to each other.

Determinism is preserved: the asset is immutable and the Chebyshev
evaluation is deterministic f64 math.

---

## 9. Performance targets

- **60 FPS** under all normal interaction (pan, zoom, node edit).
- **Coarse trajectory preview** within 100 ms of a maneuver edit.
- **Prediction budget** per frame: 2–4 ms on the worker thread. Prediction
  never blocks the render loop.
- **Prediction horizon** reaches interplanetary transfers (weeks to months of
  game time) comfortably.
- **Body state query** < 1 µs — called thousands of times per prediction.
- **Live stepping** stays within frame budget up to the maximum advertised
  warp; warp levels beyond that are not offered.

---

## 10. Out of scope for MVP

All deferred to later phases; the architecture accommodates each without
restructuring.

- Precomputed N-body ephemeris for bodies (§8).
- Drag, SRP, J₂ as additional effects.
- Richer burn models: throttle, mass flow + Tsiolkovsky, gimbal, staging.
- Intent-based nodes: "circularize here," "capture at ghost," "match target
  velocity." The node system is already a time + dv + frame, extensible to
  "intent → dv via solver."
- Multi-ship: multiple vessels, each with its own plan and prediction.
- Flight view: smooth zoom from map into a ship-level chase camera.
- Gravity-field overlays: Hill spheres, equipotentials, zero-velocity curves.
- Save / load of flight plans (the data is already plain; only serialization
  plumbing is missing).

---

## 11. Where to find things in code

- `crates/physics/src/types.rs` — `StateVector`, `TrajectorySample`, body
  definitions.
- `crates/physics/src/patched_conics.rs` — current `BodyStateProvider`.
- `crates/physics/src/integrator.rs` — hybrid leapfrog / RK45.
- `crates/physics/src/simulation.rs` — `Simulation::step`, effect registry,
  maneuver sequence.
- `crates/physics/src/trajectory/` — prediction worker, phase machine,
  progressive refinement.
- `crates/physics/src/maneuver.rs` — node data model.
- `crates/game/src/bridge/` — ties `Simulation` into Bevy's frame loop.
- `crates/game/src/trajectory_rendering.rs` — leg drawing, weighted-barycenter frame.
- `crates/game/src/ghost_bodies.rs` — ghost body rendering.
- `crates/game/src/maneuver/` — node placement, editing handles, frame
  attachment.
