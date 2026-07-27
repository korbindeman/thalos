# Surface-local frame & terrain-anchored structures

**Status: implemented for ships (2026-06).** The surface-local frame (SLF)
is live for ship craft, the ground colliders are solid, the runway is a
terrain-anchored structure, and the render-shake/gizmo bugs are fixed.
Verified across the `runway`, `cruise`, `final`, and `landing` scenarios
(handoff residuals 0–1e-10 m, no NaN/panic, stable landings). Three things
deviate from the original design below and are called out in
§10 *Implementation status*: **EVA was intentionally left on its
body-centered kinematic seam** (it gains nothing from the SLF — see §10),
the **runway keeps a solid cuboid collider** rather than being folded into
the terrain heightfield, and the physics step is still **fixed-timestep**
(a per-frame variant is a staged optional follow-up). Sections 1–9 are the
original design narrative; read §10 for what actually shipped.

Supersedes the body-centered-inertial *ship* contact bubble formerly
described in `docs/simulation/surface.md` ("Landing & impact destruction"). The
prior design premise — that ships needed migrating *to* a body-fixed
frame — was already partly true in the code: ships integrated in a
planet-centered body-fixed rotating frame with exact gravity/centrifugal/
Coriolis and a static-in-that-frame terrain trimesh, and raycast
spring-damper landing gear already existed. What the rebase actually
changed (§10): swapped the conversion seam to an anchored, re-anchorable
SLF; made the ground a **solid** collider (heightfield / cuboid) instead
of a one-sided trimesh; filtered the wheeled-craft hull out of ground
contact so the gear is the sole interface; clamped gear forces; and fixed
two render-side shakes.

## 1. Motivation — why the current contact stack is fragile

Today, near-surface contact physics runs in the **body-centered
inertial frame**: positions are megameter-scale, and the planet rotates
under the craft. Every robustness problem at the surface traces back to
that choice:

- **The ground is a moving object.** The terrain collider is a
  kinematic trimesh whose pose is recomputed *every frame* as the body
  rotates (`attach_terrain_patch_when_close` /
  `sync_terrain_collider_pose` in `crates/runtime/game/src/local_physics.rs`).
  The contact solver never sees a static floor; contacts are re-derived
  against moving geometry each frame, and any error in the
  multi-Mm-magnitude pose chain shows up as jitter — the visible
  "terrain shaking under the craft."
- **The trimesh is rebuilt as tiles stream.** Because the craft
  co-rotates with the planet, the tile streamer constantly loads/evicts
  tiles, churning the GPU-atlas height-source `revision()`, and
  `maintain_terrain_patch` rebuilds the collider trimesh in response.
  This was ~11% of surface frame time and the cause of the EVA
  "unplayable stutter" — fixed for EVA by *removing its collider
  entirely*, which ships can't do.
- **One-sided trimesh, permanent fall-through.** A trimesh has no
  interior; a single missed contact frame (fast descent, edge of patch,
  not-yet-streamed tile) drops the hull through the surface forever.
  The analytic height-query backstop (`local_physics.rs`, the
  penetration-lift system) exists purely to paper over this — a
  load-bearing band-aid.
- **Parked craft have huge frame velocities.** A craft at rest on the
  surface moves at `ω×r` (hundreds of m/s) in the bubble frame.
  "At rest" is a derived condition instead of `v ≈ 0`, and integrator
  error acts on large state values.

The codebase's own history points at the fix: everything robust at the
surface already bypasses this frame. Grounded EVA works because it
moved to the **body-fixed frame** with direct height queries
(`step_eva_controller`). The runway works because it gets a hand-posed
*flat* collider at a fixed elevation. The general ship-contact path is
the only one still doing rotating-trimesh contact in the inertial
frame, and it is the broken one. This design generalizes the EVA/runway
fixes into the single near-surface regime.

## 2. The surface-local frame (SLF)

When a craft is below the surface handoff altitude, all local physics
runs in a **local tangent frame anchored to a body-fixed surface point,
rotating with the body**.

### 2.1 Definition

- **Anchor**: a body-fixed unit direction `d̂` plus elevation, chosen
  at/near the craft's surface projection. Stored exactly like
  `RunwaySite` stores its site today (`center_dir`, `elevation_m`).
- **Axes (ENU)**: `up = d̂`; `east = normalize(ω̂ × up)` (`ω̂` = the
  body's spin axis in the body-fixed frame); `north = up × east`. Near
  the poles (`|ω̂ × up|` degenerate) fall back to an arbitrary
  consistent tangent. The frame **rotates with the body** — it is a
  body-fixed frame with a tangent-aligned basis and a surface-point
  origin.
- **State**: craft position/velocity in f64 relative to the anchor, in
  ENU axes; attitude as a quaternion relative to the frame. Magnitudes
  are meters-to-kilometers, so the f32/f64 precision cliffs at Mm
  magnitudes vanish, and physics agrees with rendering.

### 2.2 Frame dynamics (exact, not approximated)

The SLF is a rotating frame; the force terms are closed-form and cheap,
so include all of them — no flat-earth approximation is needed:

- **Gravity**: `g(p) = −μ r̂ / |r|²` where `r` = body center → craft,
  computed exactly per position (the anchor offset + `p`), expressed in
  ENU. Local "down" curves correctly across the frame.
- **Centrifugal**: `−ω × (ω × r)` (~0.02 m/s² at Thalos scale — small
  but exact and free).
- **Coriolis**: `−2ω × v` (~0.01 m/s² at aircraft speeds).

These are evaluated analytically per integration step. Aero forces
(`thalos_physics_canonical::aero`) already work from airspeed and local
density; in the SLF the surface-relative velocity *is* the frame
velocity (the atmosphere co-rotates), which simplifies the wind-frame
derivation rather than complicating it.

### 2.3 Curvature

The frame is not flat-earth: positions are exact 3-D, gravity is exact
radial. The only place flatness appears is the terrain **heightfield
collider** (§3), whose samples bake in the curvature drop relative to
the tangent plane (~2.6 m at the edge of a 4 km patch on Thalos), so
the collider matches the rendered ground exactly.

### 2.4 Re-anchoring

When the craft's horizontal offset from the anchor exceeds a threshold
(~1–2 km), pick a new anchor under the craft and translate the state in
f64 — exact, no discontinuity, the same trick big_space uses for cell
crossings. Re-anchoring is also the natural point to rebuild the
terrain heightfield window (§3). A plane cruising at 15,000 ft
re-anchors every few seconds; each re-anchor is a constant-time state
translation plus an async collider rebuild, not a hitch.

### 2.5 Entry / exit

- **Enter** when AGL drops below the handoff altitude (today's 20 km
  bubble threshold is a reasonable start). Convert canonical inertial
  state → body-fixed (existing `body_fixed.rs` helpers) → ENU offset
  from the chosen anchor. One conversion at the boundary, mediated
  through canonical state — the same pattern as
  `rebase_bubble_to_dominant_body`.
- **Exit** when AGL climbs back above the threshold (with hysteresis):
  convert SLF state → canonical inertial, hand authority back to the
  existing regimes (OnRails / AttitudeOnly / Full).
- While in the SLF, the canonical `CraftState` is **derived** from the
  SLF state every frame via the same conversion, so the map, trajectory
  prediction, HUD, and every other canonical consumer keep working
  unchanged. One craft state, one authority — the SLF is just a new
  authority regime, not a parallel state owner.

## 3. Terrain collider: static heightfield

Replace the per-frame-re-posed, per-streaming-rebuild trimesh with a
**static heightfield collider in the SLF**:

- An N×N grid of heights centered on the anchor, sampled from the same
  `HeightSource` the renderer and CPU queries read, minus the tangent
  curvature drop (§2.3). Built once at frame entry / re-anchor.
- **Never re-posed.** In the SLF the planet does not rotate under the
  craft — the frame rotates with it. The collider is genuinely static
  geometry; no per-frame pose sync, no broadphase churn, no moving-floor
  jitter. This is the structural fix for the shaking.
- **Two-sided / solid semantics.** Heightfield colliders have a defined
  interior, so the missed-frame → permanent-fall-through failure class
  is gone. The analytic penetration backstop is demoted from
  load-bearing to a debug assertion.
- **Rebuild policy**: rebuild only on re-anchor, or when the
  height-source revision changes *within the patch window* (a tile that
  overlaps the patch finished streaming). Rebuilds are an atomic swap of
  the heightfield data, not per-frame work, and can be computed off the
  main thread.
- **Flatten pads come for free.** Structure pads
  (`thalos_terrain::TerrainFlatten` via `TerrainFlattenRegistry`)
  already flow through the `HeightSource`, so the heightfield
  automatically contains the flattened runway/building pad. The
  separate hand-posed flat runway collider and
  `sync_runway_collider_pose` retire.

## 4. Craft contact

- **Landing gear = raycast spring-damper suspension.** Per `Gear` part:
  a ray (or short shapecast) from the attachment point along local down
  against the heightfield + structure colliders, with a spring-damper
  preloaded at static sag (the existing runway-stability work carries
  over directly). Resting and rolling stability become tuning problems
  instead of rigid-contact-solver problems — this is how flight/driving
  sims do ground handling, and it sidesteps the class of contact
  instability we fought on the runway.
- **Hull contact** (belly landings, tipping, crashes): the craft's
  convex hull(s) against the heightfield via the normal contact solver.
  Fidelity here can stay low: the whole-craft impact destruction model
  (`ShipParameters::impact_tolerance_m_s` →
  `Simulation::mark_destroyed`) means a hard hull impact destroys the
  craft rather than needing realistic crash dynamics.
- **Rest / landed collapse simplifies.** "At rest" in the SLF is
  literally `|v| ≈ 0` — no co-rotation subtraction. A landed craft is a
  frozen SLF state, which is exactly what `AuthorityMode::BodyFixed`
  pose evaluation expresses today; the two converge, and the stable-
  contact collapse becomes "stop integrating, keep the SLF pose."

## 5. Authority, warp, EVA

- **The `AvianRole` four-way dance collapses.** Above the handoff:
  canonical regimes unchanged. Below it: the SLF owns translation and
  rotation; canonical state is a derived projection (§2.5). The
  "Paused / BodyFixed / AttitudeOnly / Full + per-frame snap +
  writeback fights" machinery in `local_physics.rs` reduces to one
  regime boundary.
- **Warp**: in the SLF, the EVA at-rest gating generalizes to ships —
  warp above 1× only when at rest (which, landed, is the frozen
  BodyFixed-equivalent state); moving in the SLF clamps to 1×. The
  existing 100× surface cap (UDLOD streaming limit) is unchanged.
- **EVA folds in.** The grounded EVA controller already runs in the
  body-fixed frame; it becomes a native SLF citizen — the character
  controller integrates in the same frame as ships, against the same
  static heightfield, and its special-case exemptions (collider
  removal, snap short-circuits, `apply_local_forces` early-returns)
  are deleted rather than maintained.
- `SimClock` pause semantics are unchanged.

## 6. Terrain-anchored structures

The runway is the prototype; generalize it into data-driven
**structures** so buildings — and eventually player-placed/edited
buildings — are instances, not bespoke plugins.

### 6.1 The structure record

```text
StructureSite {
    id:              StructureId,
    body_id:         BodyId,
    anchor_dir:      DVec3,        // unit body-fixed direction (as RunwaySite.center_dir)
    heading_tangent: DVec3,        // body-fixed orientation on the surface
    placement:       FlattenTo { elevation_m } | Drape,
    kind:            StructureKind, // Runway | Building(..) | Pad | ...
    terrain_mod:     Option<TerrainModifier>, // today: TerrainFlatten; later ramps/foundations
}
```

- A per-body `StructureRegistry` resource holds the sites
  (single-writer, per the project invariant). Authored structures load
  from RON; player-placed structures (future) are the same records
  written at runtime and persisted per save.
- `RunwaySite` becomes the first `StructureKind`; the runway's site
  search, flatten install, paving meshes, and markings become the
  runway kind's spawn logic, parameterized by the record.

### 6.2 Terrain coupling

Each structure may carry a **terrain modifier** installed through the
existing `TerrainFlattenRegistry` handle the tile provider reads — the
runway's flatten pad mechanism, unchanged. Because modifiers flow
through the `HeightSource`, the rendered ground, CPU height queries,
*and* the SLF heightfield collider (§3) all agree automatically.

### 6.3 Physics

Structures whose footprint intersects the SLF window contribute
**static colliders in the frame** — built once at frame entry /
re-anchor, exactly like the terrain heightfield. No per-frame pose
sync: `update_runway_transform` and `sync_runway_collider_pose` retire
*for physics*. (The per-frame f64 root-grid re-placement stays for
**rendering** distant/at-warp structures — the documented f32-quaternion
jitter fix — since render placement is independent of physics.)

### 6.4 Player placement / editing (built — see `base_building.md`)

> **Landed 2026-06-29** as the in-world base editor (`crates/runtime/game/src/base_editor/`).
> The region invalidation below took the **coarse-hammer** path (despawn +
> respawn the whole body terrain, reusing the persistent flatten handle) rather
> than scoped per-AABB invalidation; the scoped version (item 1) is the
> optimization follow-up. Full design in `base_building.md`.

Placement = write a `StructureSite` at runtime + install its terrain
modifier + spawn its visuals/colliders. The one genuinely new mechanism
this requires is **region tile invalidation**: the runway avoids it by
installing its flatten *before* tiles stream at the site, but a player
placing a building stands on already-streamed tiles. Required:

1. Scoped invalidation in the tile pipeline — bump a region-scoped
   revision when a `TerrainFlatten`/modifier is installed or edited, so
   UDLOD reloads only tiles overlapping the modified AABB. *(MVP shipped a
   whole-body despawn/respawn instead; scoped invalidation is deferred.)*
2. SLF heightfield rebuild for the affected window (already the §3
   revision-change path).
3. Collider/visual respawn for the edited structure.

Everything else (records, registry, modifiers, static colliders) is the
same machinery as authored structures. The registry was designed for this,
and the base editor now writes player-placed sites/buildings into it.

## 7. Relation to the Avian question

This design is **backend-agnostic** and should land *before* any
decision to replace Avian:

- Implement on Avian first: a static heightfield collider + a dynamic
  craft body in the SLF is squarely inside what Avian is good at, and
  most of the current pain (moving floor, rebuild churn, writeback
  fights against snapped state) was the *frame*, not the engine.
- If Avian is later dropped, the replacement problem has shrunk to
  "one rigid body vs. static heightfield + raycast gear in small local
  coordinates" — tractable on `parry3d-f64` directly. Docking/joints
  (which follow this phase) favor keeping a constraint solver; decide
  then, with the SLF already in place either way.

## 8. Migration plan

1. **Frame math in `thalos_physics_canonical`** (pure Rust, unit
   tested): `surface_local` module — anchor + ENU basis construction,
   SLF ↔ body-fixed ↔ inertial conversions (building on `body_fixed.rs`),
   gravity/centrifugal/Coriolis terms, re-anchor translation.
2. **SLF bubble for ships**: new authority regime in the game/
   `physics_local` layer; static heightfield collider in the frame;
   canonical state derived per frame. Keep the trimesh path behind a
   flag as fallback until the landing / final / runway scenarios verify.
3. **Raycast gear suspension**; demote the penetration backstop to an
   assertion; delete the per-frame collider pose syncs.
4. **EVA folds into the SLF**; delete its special-case exemptions.
5. **Structures registry**: port the runway to a `StructureSite`; add a
   second structure kind (a simple building) to prove generality. *(Done —
   `BaseSite`/`Building` kinds added by the base editor.)*
6. **Runtime placement** (region tile invalidation) when placement
   gameplay needs it. *(Done 2026-06-29 — the in-world base editor, via the
   coarse despawn/respawn invalidation MVP; see `base_building.md`.)*

## 9. Open questions

- **Handoff altitude**: keep 20 km AGL, or trigger on dynamic pressure /
  density so vacuum-world descents enter later? Hysteresis width?
- **Multiple craft in the local scene** (debris and other vehicles): the SLF is
  per-active-craft today (mirroring `ActiveLocalBubble`), but the target is now
  resolved rather than open. One dominant-body frame contains N independently
  authoritative vessel rigid bodies so separated stages share terrain and can
  collide/fall while another craft stays active. See
  [vessels.md](vessels.md) and ADR-20260724T230226Z.
- **Ocean/water contact**: the heightfield is seabed; floating craft
  need a separate water-surface interaction. Out of scope here, but the
  SLF is the right frame for it too.
- **Heightfield window size vs. gear shapecast length** at extreme
  attitudes (a plane nosing over the patch edge) — likely solved by a
  generous window + the AGL-banded attach logic that exists today.

## 10. Implementation status (what actually shipped)

The frame math and ship rebase landed as designed; the contact and EVA
details diverged. Concretely:

**SLF frame math** — `thalos_physics_canonical::surface_local`
(`SurfaceAnchor`, `SurfaceLocalFrame` with the Y-up ENU basis + pole
fallback, `SurfaceLocalState`, the `inertial_to_surface_local` /
`surface_local_to_inertial` conversions composed onto `body_fixed.rs`,
`surface_local_acceleration`, `altitude_asl_m`/`radial_up`, `reanchor`).
Unit-tested incl. a free-fall dynamics-equivalence test (SLF vs
body-centered inertial agree to <1e-6) that pins the Coriolis sign.

**Ships in the SLF** — `LocalBubble.frame: SurfaceLocalFrame`
(`crates/simulation/physics_local`); the ship conversion seam
(`inertial_to_ship_frame`/`ship_frame_to_inertial` in
`crates/runtime/game/src/local_physics.rs`) routes through the SLF;
`apply_local_forces` uses `surface_local_acceleration`;
`reanchor_surface_frame` re-anchors at 1.5 km horizontal drift (exact f64,
canonical untouched). Consumers migrated: aero/control_bus altitude →
`altitude_asl_m`, debug hitbox isometry, terrain-floor backstop and
surface-friction up-vectors. Handoff residuals measured 0–1e-10 m in every
scenario.

**Solid ground** — the one-sided kinematic trimesh became a solid
**heightfield** collider (`spawn_terrain_collider_patch`, authored in the
patch-tangent frame, posed via `patch_basis_rotation` + the SLF rotation),
and the runway's one-sided trimesh became a solid **cuboid slab**
(`crates/runtime/game/src/runway.rs`). The one-sided trimesh was the cause of the
landing craft being violently ejected off its gear (one-step
penetration-recovery on a surface with no interior).

**Gear as sole ground contact** — a wheeled craft's hull collider is
filtered out of solver contact with the ground via collision layers
(`thalos_physics_local::{GROUND_LAYER, CRAFT_LAYER,
ground_collision_layers, wheeled_craft_collision_layers}`); the raycast
spring-damper gear is its only ground interface, and that gear force/torque
is now **inertia-relative clamped** (mirroring the aero model) — together
these stopped the spin/jump on throttle-up. Gearless craft (landers) keep
default all-vs-all layers and rest on the heightfield directly (verified in
`final`). Crash detection and the stable-contact→`BodyFixed` collapse were
repointed to `weight_on_wheels.grounded || hull-contact`.

**Render-shake fixes (not in the original design)** — two pre-existing
bugs surfaced once contact was stable: (a) the ship-pose overstep
extrapolation in `ship_view.rs` double-counted the planet co-rotation
`ω×r` after the SLF migration, injecting a ~4 m ground-speed-independent
sawtooth that read as the *ground* shaking while rolling — fixed by
subtracting the full surface velocity; (b) `draw_debug_hitboxes` (F3) read
a one-frame-stale `GlobalTransform` under big_space — fixed by moving it to
`PostUpdate` after `TransformSystems::Propagate` (same fix `draw_aero_debug`
already had).

**Structures registry** — `crates/runtime/game/src/structures.rs`:
`StructureSite`/`StructureRegistry`/`StructurePlacement` +
`apply_structure_flatten` (the single "stick to terrain" flatten path). The
runway registers a `StructureSite { kind: Runway, placement: FlattenTo }`
and installs its pad through this path; a future building is a data entry
(`FlattenTo` or `Drape`) plus its own visuals.

### Deviations & deferred work
- **EVA stayed on its body-centered kinematic seam** (design §5 wanted it
  folded in). Rationale: grounded EVA is a kinematic character controller
  with no collider and no Avian integration — it computes its canonical
  state directly in body-fixed (`player_controller.rs`, the
  `install_local_rigid_body_state` at the end of `step_eva_controller`).
  The SLF exists to give Avian's *contact solver* stable small coordinates;
  EVA never touches that solver, so the fold-in is cosmetic, carries real
  risk (the controller's surface-slide-out reseed guard, warp co-rotation
  sweep, and teleport-reseed would all interact with re-anchoring), and is
  hard to verify without on-foot walk-testing. EVA's "special cases" are
  intrinsic to it being kinematic, not artifacts of the frame. Treat EVA as
  a deliberately separate kinematic path.
- **Runway keeps its own solid cuboid collider** rather than being folded
  into the terrain heightfield (design §3/§6 envisioned the flatten pad in
  the heightfield as the sole ground). The cuboid is simpler and guaranteed
  flat; consolidation is a possible cleanup but the terrain patch is
  currently skipped on the runway body to avoid double contact.
- **Fixed-timestep physics retained.** Avian still steps in `FixedPostUpdate`
  at 64 Hz. A per-render-frame variant (`PhysicsPlugins::new(Update)` +
  reordering the force/readback chain around `PhysicsSystems::First..Last`)
  would remove the residual high-speed render stutter and the gear's
  one-frame force latency (possibly letting the gear clamp relax), but
  carries warp-pause-ordering and gear-retune risk and was deferred while
  feel is acceptable.
- **Penetration backstop not demoted.** `terrain_floor_backstop` is still
  load-bearing (the heightfield is a surface, not a closed solid, so fast
  descents can still tunnel). Demotion to a warning needs an intervention
  counter + zero-intervention runs across the descent scenarios first.
- **Heightfield rebuilds are synchronous** on the main thread
  (on re-anchor / height-source revision change) — a possible hitch near
  terrain streaming; async rebuild with atomic swap is the planned upgrade.
- **Runtime structure placement / region tile invalidation** (design §6.4)
  is **built** (2026-06-29) as the in-world base editor — player-placed sites
  and buildings, with a coarse despawn/respawn invalidation MVP. Scoped
  per-AABB invalidation remains the optimization follow-up. See
  `base_building.md`.
