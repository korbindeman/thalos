# Thalos: Orbital Mechanics MVP Spec

Realistic N-body trajectory simulation with an intuitive planning UI.

---

## 1. System Overview

The MVP is an N-body orbital mechanics sandbox. The solar system is defined by an external data file containing body definitions (mass, radius, orbital parameters, color, parent relationships) and a ship definition. The simulation loads this file and sets up the initial conditions accordingly. For the MVP, the system is validated against a three-body subset (star, homeworld, moon) but the architecture is body-count-agnostic.

The simulation computes gravitational acceleration from all bodies on the ship at every timestep. No patched conics, no sphere-of-influence boundaries in the physics. Lagrange points, three-body captures, and chaotic sensitivity near gravitational boundaries all emerge from the integrator, not from special-case detection.

The MVP proves out the core loop: the player views their trajectory, sees uncertainty grow via the cone, places maneuver nodes anchored to ghost bodies at future positions, and adds corrections where the cone gets too wide. If this works with a small body set, it scales to a full solar system by adding bodies to the definition file.

Ship propulsion is placeholder. The goal is to validate the simulation, trajectory rendering, and interaction model.

---

## 2. Body State Provider

All massive bodies are precomputed via a full N-body simulation run at startup. The star is fixed at the origin. All other bodies interact gravitationally with each other during precomputation, producing a high-fidelity ephemeris for the entire gameplay time span.

Adaptive per-body sampling drives ephemeris generation. Sampling density is driven by curvature and fitting error: denser near periapsis where velocity and curvature are high, sparser near apoapsis where the trajectory is nearly linear. Each body has independent knot densities tuned to its orbital characteristics.

Those adaptive samples are an intermediate accuracy model, not necessarily the final stored format. The runtime/on-disk ephemeris may compress each body's trajectory into error-bounded Chebyshev segments fit over those samples, preserving smooth position/velocity queries while keeping file size and memory use reasonable over long spans.

Queries use O(log n) segment lookup per body, then evaluate the segment representation to recover position and velocity in a heliocentric inertial frame. Hermite interpolation remains a valid fitting/intermediate tool, but the persisted query format does not need to be raw Hermite knots.

The provider exposes a single interface: given a timestamp, return position, velocity, and mass for every body in a heliocentric inertial frame.

### Time Model

Game time starts at t=0 and runs at variable warp speed (1x, 10x, 100x, 1000x, etc.). The provider doesn't know about warp. It takes an absolute timestamp in seconds and returns positions. Warp is handled by the simulation loop advancing the clock faster.

### Precomputation Scope

For a small body set, the ephemeris is computed on startup (seconds at most). The time span is configurable, defaulting to 100 years. For larger solar systems, disk caching keyed by a hash of the initial conditions will be added as an optimization.

### Immutability

The ship never affects the bodies. The ephemeris is a fixed lookup table for the entire game session.

---

## 3. Ship Propagator

The ship propagator integrates the ship's state vector (position + velocity) forward in time under the combined influence of all forces acting on it.

### Force Model

The propagator doesn't know what forces exist. It maintains a force registry: a list of force functions that each take (position, velocity, time, body states) and return an acceleration vector. The propagator sums all contributions at each timestep.

MVP forces:
- **Gravity:** Sum of GM/r^2 contributions from each body, evaluated at the ship's position using body positions from the ephemeris.
- **Thrust:** Constant acceleration in a specified direction for a computed duration. Active during maneuver burn windows. Modeled as a force in the registry, replaceable later with more sophisticated engine simulation.

Future forces (not MVP, but the architecture accommodates them with no integrator changes):
- Atmospheric drag
- Solar radiation pressure
- Advanced engine models (variable thrust, throttle curves, gimbal)

### Hybrid Integrator

**Symplectic (leapfrog/Stormer-Verlet):** The default for stable coasting in a single dominant gravity well. Fixed timestep, energy-conserving over long durations, cheap per step.

**Adaptive Runge-Kutta (Dormand-Prince RK45):** Activated when perturbations become significant. Variable timestep that shrinks automatically where dynamics are stiff: near SOI transitions, close approaches, Lagrange point regions, and during thrust burns.

### Switching Criterion

The ratio of the largest perturbing acceleration to the dominant body's acceleration. When this ratio crosses a threshold (tunable, starting around 0.01), switch from symplectic to adaptive. When it drops back below, switch back. A hysteresis band prevents rapid toggling near the boundary.

### Output

The propagator outputs the full trajectory as a sequence of timestamped samples. Each sample carries:
- Position (heliocentric)
- Velocity (heliocentric)
- Timestamp
- Dominant body ID
- Perturbation ratio
- Integrator step size

The adaptive integrator naturally produces denser samples where dynamics are complex, sparser where smooth. This maps directly to rendering needs and cone width.

### Operating Modes

- **Live mode:** Advances the ship's actual state at the current warp rate. Sub-stepped at high warp to maintain accuracy. Capped at a maximum warp rate where the propagator can no longer keep up within the per-frame budget.
- **Prediction mode:** Runs ahead from the current state through the maneuver node sequence, producing the predicted trajectory for rendering. Runs asynchronously on a background thread.

---

## 4. Trajectory Prediction

Trajectory prediction is the propagator running in prediction mode: starting from the ship's current state, applying each maneuver node's thrust at its scheduled time, and integrating forward to produce the full predicted path.

### Triggering

Re-triggered when the player edits a maneuver node (create, move, delete, change delta-v), or when the ship's actual state advances enough that the prediction's starting point is stale. Does not re-run every frame.

### Progressive Refinement

On trigger, a coarse pass runs first with larger timesteps for fast visual feedback. Refinement continues over subsequent frames with smaller timesteps. The trajectory line and cone update as refinement progresses. The player sees the rough shape immediately, then it sharpens.

### Prediction Termination

The prediction runs until one of:
- The cone exceeds the fade-out threshold
- A stable orbit is detected (one revolution drawn)
- A collision with a body is detected (trajectory terminates at the surface)
- The per-frame computation budget is exhausted (continues next frame)

### Partial Re-propagation

When the player edits node N in a sequence, only the trajectory from node N onward is re-propagated. Earlier legs are unchanged.

### Output

A list of trajectory segments, one per leg between maneuver nodes (plus the final leg after the last node). Each segment is the sequence of timestamped samples from the propagator with full metadata. These feed directly into the trajectory renderer and cone renderer.

### Threading

Prediction runs on a background thread. The renderer draws the latest completed prediction. If a new edit arrives while prediction is running, the current run is cancelled and restarted from the edited node. The player never waits for prediction to finish.

---

## 5. Cone Rendering

The cone represents trajectory uncertainty. It communicates how reliable the predicted path is at each point.

### What Drives Cone Width

The perturbation ratio and integrator step size from the propagator sample metadata. Where the adaptive integrator takes small steps, dynamics are sensitive to initial conditions and small burn execution errors compound fast. The cone widens. Where the integrator takes large steps, dynamics are smooth and predictable, and the cone stays narrow.

This is a gameplay-readable proxy for chaotic sensitivity, not a rigorous statistical uncertainty estimate.

### Visual Treatment

A translucent tube around the trajectory line. Circular cross-section, radius derived from the cone width signal. Smooth interpolation between samples. Same color as the trajectory but lighter and more diffuse. The tube never obscures the trajectory line itself.

### Behavior at Extremes

**Near-zero width (stable coasting):** The cone is effectively invisible. Just the trajectory line.

**Fade-out threshold:** The cone reaches a maximum radius. Both the line and cone fade to zero opacity over a short distance. Clean termination, nothing rendered beyond.

### Stable Orbits

In a stable orbit (default one-revolution draw), the cone is invisible. When the player extends the view beyond one revolution (for future maneuver planning or in response to an encounter warning), the cone becomes visible and grows slowly over successive orbits, reflecting timing uncertainty.

### What the Cone is Not

Not a collision radius. Not interactive. Purely a visual signal that drives the player toward placing corrections where needed.

---

## 6. Trajectory Draw Rules

### Stable Orbit

The propagator detects approximate periodicity by checking if the state vector returns close to its starting point (position and velocity within tunable thresholds). When detected, one full revolution is drawn as a closed loop. No cone visible, solid line in the dominant body's color.

### Extending Beyond One Revolution

The player can manually look ahead, e.g. when selecting a future orbit for maneuver placement. The encounter detection system can also trigger a warning that, when clicked, extends the visible trajectory. In both cases, the trajectory continues beyond the single loop and the cone becomes visible.

### Complex Path (Not Periodic)

The trajectory line is drawn with the cone growing based on propagator metadata. When the cone reaches the fade-out threshold, both line and cone fade to zero opacity. Nothing is rendered beyond.

### Maneuver Sequences

Each leg between maneuver nodes is drawn independently. A leg might be a stable orbit (one revolution until the node's scheduled time), or a complex transfer (drawn until cone fade-out or next node). The trajectory is visually continuous but each leg has its own draw behavior.

### Post-Maneuver Prediction

After the final maneuver node, prediction runs forward normally. Stable orbit: one revolution. Complex path: draw until cone threshold.

### Collision Termination

If the predicted trajectory intersects a body surface within the rendered portion, the trajectory terminates at the surface. This is the visible indication of a collision.

If a collision or close encounter is detected within reliable prediction bounds but beyond the default one-revolution draw of a stable orbit, a persistent UI warning appears. The player clicks it to extend the trajectory and see the encounter.

No collision detection past cone fade-out. If prediction is unreliable, collision detection is also unreliable.

### Trajectory Line Visual Treatment

Solid 3D curve, screen-space width (constant pixel thickness regardless of zoom). Colored by dominant body with smooth blending at transitions based on gravitational influence ratio.

### Time Ticks (Off by Default)

Togglable. Small marks along the trajectory at regular time intervals. Bunched at periapsis (fast), spread at apoapsis (slow). Subtle, subordinate to all other visual elements.

---

## 7. Ghost Bodies

Ghost bodies are translucent renderings of real bodies at their future positions along the ship's trajectory.

### When They Appear

A ghost body fades in when the ship's predicted trajectory enters a region where that body has significant gravitational influence. This uses the perturbation ratio already computed by the propagator. The star does not get a ghost (fixed at origin).

Gravitationally significant non-physical locations (Lagrange points L1-L5 for relevant body pairs) can also appear as ghost markers, computed from the ephemeris at the relevant future timestamp.

### Positioning

The body state provider is queried at the timestamp corresponding to the relevant point on the trajectory. The ghost shows the body where it will actually be at that future time. As the ship progresses and the prediction updates, the ghost position updates.

As the ship advances in real time and approaches the ghost's timestamp, the real body converges on the ghost's position and smoothly replaces it. Ghost bodies have no gravitational interaction with anything. They are purely visual.

### Visual Treatment

Same model as the real body (sphere or circle depending on zoom) but translucent. Label includes the body name plus T+ time (e.g. "Body T+3d 14h").

### Multiple Ghosts

On a long trajectory, the same body can appear as multiple ghosts at different timestamps (e.g. two flybys of the same moon). Each is an independent instance. Maneuver nodes attach to whichever ghost is nearest on the trajectory.

### Interaction with Maneuver Nodes

When the player places a maneuver node near a ghost body (or ghost Lagrange point), the node's reference frame locks to that body. The node's handles orient relative to an orbit around the ghost. The player plans burns relative to the body's future position.

### Trajectory Rendering Relative to Ghosts

Trajectory segments near a ghost body are rendered relative to the ghost's position (which is the body's future position from the ephemeris). This means an approach trajectory appears as a clean curve relative to the destination, even though the underlying data is heliocentric.

---

## 8. Maneuver Nodes

A maneuver node represents a thrust event at a specific point on the trajectory.

### Force Model

The node defines a delta-v that is applied as a constant acceleration over a computed burn duration based on the ship's thrust and current mass. The propagator integrates through the burn, producing a curved trajectory during the burn rather than a sharp kink. Thrust is a force in the force registry, active during the burn window.

### Creation

The player clicks on the trajectory line to place a node at a specific timestamp. In a stable orbit, the player can shift the node forward or backward in discrete orbit increments to schedule it on a future pass.

### Editing

The node exposes three pairs of directional arrows in the local reference frame of the dominant body (or ghost body / Lagrange point) at that trajectory point: prograde/retrograde, normal/antinormal, radial-in/radial-out. A central sphere handle slides the node along the trajectory. Delta-v magnitude and burn duration are shown in the UI panel when selected.

Arrows that are nearly aligned with the camera viewing angle are hidden (dot product between arrow direction and camera forward vector below a threshold) to prevent occlusion of useful handles.

### Reference Frame Attachment

The node's handles orient relative to the dominant body at that point on the trajectory. Determined at placement by which body has the largest gravitational influence. Snaps discretely to the dominant body (no blending between reference frames for node editing).

### Re-propagation

Any edit triggers re-propagation from that node onward. Earlier legs are unchanged. The prediction system cancels in-progress computation and restarts from the edited node. Progressive refinement provides fast visual feedback.

### Multiple Nodes

The player builds a sequence of nodes. The prediction chains through them: propagate to node 1, apply thrust, propagate to node 2, apply thrust, continue. Each node defines a leg of the trajectory.

### Deletion

Removing a node merges the adjacent legs. Re-propagation from the preceding node (or ship's current state if it was the first).

### Visual Treatment

A sphere on the trajectory line. Clicking opens the editor with directional arrows and the central slide handle. Unselected nodes are visible as spheres but without handles. The trajectory has a subtle visual distinction at node boundaries so the player reads the plan as a sequence of legs.

### Future Node Types (Not MVP)

The node system is designed to be extensible beyond raw delta-v:
- **Circularization:** "Make my orbit circular at this point."
- **Injection:** "Capture into orbit around this body."
- **Correction:** "Minimize cone width here" (solver computes optimal delta-v).
- **Station-keeping:** "Hold this position/orbit for a duration" (active over a time window).
- **Manual vector:** Full direct control over the thrust vector.

These capture intent rather than raw impulse. The system computes the required force from the objective. All produce the same thing for the propagator: a force over a time window.

---

## 9. Reference Frames & Coordinate Systems

### World Frame

Everything in the simulation runs in a single heliocentric inertial frame. Star at the origin, ecliptic plane as the XZ plane, Y up. The body state provider, ship propagator, and trajectory prediction all operate in this frame. No frame conversions in the physics.

### View Frame

The camera can focus on any body or the ship. When focused, the view is centered on the focus target and moves with it. This is a rendering offset only. The simulation is unaffected by camera state.

### Trajectory Rendering Frame

By default, each trajectory segment is rendered relative to its dominant body at each point. The renderer subtracts the dominant body's position (from the ephemeris at that sample's timestamp) from each sample. For segments near ghost bodies, the ghost's position (the body's future position) is used as the reference.

This means a ship in low orbit around a planet shows a clean ellipse relative to that planet, and an approach segment toward a moon shows a clean curve relative to the moon's future position.

A toggle switches to rendering everything relative to the focused body, showing the full path in one coherent frame.

### Maneuver Node Frame

When a node is selected, its handles are computed from the ship's velocity relative to the dominant body at that trajectory point. This is the ship's world-frame velocity minus the dominant body's world-frame velocity, producing the local prograde/radial/normal directions.

### Dominant Body Determination

At each trajectory sample, the dominant body is the one contributing the largest gravitational acceleration to the ship. Computed during propagation and stored in sample metadata. Drives trajectory color blending, maneuver node frame attachment, ghost body relevance, and default rendering frame.

### No Explicit SOI Boundaries

No sphere-of-influence boundaries in the simulation or UI. Transitions between dominant bodies are a smooth gradient of gravitational influence, reflected in the trajectory color blend.

---

## 10. 3D Navigation View

### Camera Model

KSP-style orbit camera. The player selects a focus target (body, ship, maneuver node, ghost body). The camera orbits the target with mouse drag for rotation, scroll for zoom. Double-click to switch focus target. Camera state is (focus target, distance, azimuth, elevation).

### Zoom Range

Continuous zoom from close enough to see low orbit detail to far enough to see the full system. Handles the massive scale difference between a ship in low orbit (hundreds of km) and the full heliocentric view.

### Body Rendering

Bodies are 3D spheres at real proportional sizes. When a body's screen-space diameter drops below a fixed pixel threshold, it smoothly crossfades to a fixed-diameter circle billboard in the body's defined color.

### Body Labels

Always visible, anchored to the body. Scale with distance to remain readable without dominating at close range. Ghost body labels are translucent/dimmed and include T+ time.

### Ship Rendering

The ship is a fixed-size marker (like the minimum-size body circle) with a label. No 3D ship model in the MVP.

### Body Orbit Lines

Each body's orbit around its parent is drawn as a thin line in the body's defined color. Sampled from the precomputed ephemeris for one revolution. Background reference, dimmer than the ship trajectory.

### Reference Plane

A subtle grid on the ecliptic plane at medium zoom levels, fading at very close or very far zoom. Provides spatial orientation for understanding inclination.

### Future: Seamless Ship View Transition

The intended design is a smooth continuous zoom from map view down to ship-level flight view, with the camera behavior morphing from orbit camera to chase camera. Deferred from MVP. When implemented, requires blending camera behaviors and handling both ship-level detail and system-level abstraction during the transition.

---

## 11. Visual Design Language

### Hierarchy of Visual Prominence

Bodies > ship trajectory and cone > maneuver nodes > ghost bodies > body orbit lines > time ticks > reference grid.

### Trajectory Line

Solid 3D curve, screen-space width. Colored by dominant body (from body color definitions in the solar system file) with smooth blending at transitions based on gravitational influence ratio. The most prominent drawn element after bodies.

### Cone

Translucent tube around the trajectory line. Same color family as the trajectory but lighter and more diffuse. Radius from propagator metadata. Fades to zero opacity at the termination threshold. Never obscures the trajectory line.

### Ghost Bodies

Same visual as the real body but translucent. Label with T+ time. No connecting line to the real body.

### Maneuver Nodes

Spheres on the trajectory line. Clicking opens the editor: three pairs of directional arrows (prograde/retrograde, normal/antinormal, radial-in/radial-out) plus a central sphere for sliding along the trajectory. Arrows aligned with the camera viewing angle are hidden. Unselected nodes show as spheres only. Downstream leg has a subtle visual distinction from the upstream leg.

### Body Orbit Lines

Thin lines in each body's defined color. Sampled from the precomputed ephemeris. Dimmer than the ship trajectory.

### Time Ticks (Off by Default)

Togglable. Small marks at regular time intervals. Subtle, subordinate to all other elements.

### Collision/Encounter Warning

If within the rendered trajectory: the line terminates at the body surface. If beyond the default draw (but within reliable prediction): a persistent UI element. The player clicks it to extend the trajectory and focus the camera on the encounter.

### Color System

Body colors from the solar system definition file drive everything:
- Body orbit lines: body color, dimmed
- Ship trajectory: dominant body color, full brightness, smooth blending at transitions
- Cone: dominant body color, lighter variant
- Ghost bodies: body color, desaturated and translucent

---

## 12. Performance Targets

### Frame Rate

60 FPS minimum during all normal interaction (zooming, panning, selecting and editing nodes).

### Trajectory Prediction Latency

Coarse trajectory preview within 100ms of a maneuver node edit. Full refinement continues progressively over subsequent frames.

### Propagation Budget

The prediction system gets a fixed per-frame time budget (target 2-4ms) for background refinement. Yields after the budget is spent. Prediction never blocks the render loop.

### Prediction Horizon

The system handles predictions spanning weeks to months of game time. Interplanetary transfers must be comfortably within reach.

### Time Warp

At high warp, the propagator sub-steps to maintain accuracy within the per-frame budget. Frame rate stays at 60 FPS. A maximum warp rate is enforced where the propagator can no longer keep up.

### Ephemeris Lookups

Under 1 microsecond per query. Called thousands of times per prediction update.

### Memory

Precomputed ephemeris with adaptive per-body sampling as the accuracy model and compressed segment encoding as the storage model. Compact due to curvature-driven density plus polynomial compression.

### Ephemeris Precomputation

For small body sets, computed on startup (seconds at most). Disk caching (keyed by hash of initial conditions) deferred to larger solar systems.

---

## 13. Out of Scope

Deferred from the MVP, noted for future design:

- **Atmospheric drag:** Force model is extensible. Drag is a future force function. Affects low orbit stationkeeping and landing trajectories.
- **Realistic ship definitions:** Propulsion types, Isp, fuel tracking, mass ratio, staging. MVP uses a placeholder ship with linear thrust.
- **Intent-based maneuver nodes:** Circularization, injection, station-keeping, correction solver, manual vector control.
- **Multi-ship:** Multiple vessels with independent trajectories and predictions.
- **Seamless zoom to ship view:** Smooth camera transition from map view to flight view with ship rendering.
- **Gravitational overlays:** Hill spheres, equipotential surfaces, zero-velocity curves, gravity gradient visualization. Togglable overlay system.
- **Ephemeris disk caching:** Precompute once, write to disk keyed by initial conditions hash.
- **Advanced burn modeling:** Variable thrust, throttle control, staging.
- **Landing and ascent trajectories:** Require atmosphere model, terrain collision, aerodynamic forces.
- **Advanced time tick modes:** Orbital parameters, velocity readouts, or other per-sample data along the trajectory.

---

## Project Structure

### Workspace: `thalos`

Two crates with a clean boundary between pure physics computation and the Bevy game.

**`crates/physics` (`thalos_physics`)**

Pure Rust library, no Bevy dependency. Testable in isolation.

- `ephemeris` - N-body precomputation, adaptive per-body sampling/error control, and compact runtime lookups via compressed trajectory segments (for example, Chebyshev fits).
- `integrator` - Hybrid integrator (symplectic + adaptive RK), switching logic, hysteresis.
- `forces/gravity` - Gravitational acceleration summation from all bodies.
- `forces/thrust` - Linear constant-acceleration thrust model.
- `forces` - Force registry, summation interface.
- `trajectory` - Trajectory prediction, progressive refinement, stable orbit detection, collision detection, cone width computation.
- `maneuver` - Maneuver node data model, sequencing, orbit-increment shifting.
- `types` - Shared types: state vectors, body data, sample metadata.

**`crates/game` (`thalos_game`)**

Bevy application. Depends on `thalos_physics`.

Internal modules organized by concern:
- Rendering (trajectory lines, cone, bodies, ghost bodies, orbit lines, grid)
- UI (maneuver node interaction, handle culling, warnings, toggles, delta-v readouts)
- Camera (orbit camera, focus targets, zoom)
- Bridge (systems that feed `thalos_physics` outputs into Bevy ECS for rendering and interaction)
