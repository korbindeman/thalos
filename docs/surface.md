# Surface

> **Direction update (2026-06): surface-local frame design.** The
> body-centered-inertial contact bubble described in the "Landing &
> impact destruction" part of this doc (per-frame re-posed terrain
> trimesh, `SweptCcd` + analytic fall-through backstop, the
> `AvianRole` snap machinery) is slated to be replaced by a
> **surface-local tangent frame** with a static heightfield collider
> and raycast-suspension gear — see `docs/surface_local.md`. That doc
> also generalizes the runway into terrain-anchored structures. This
> doc remains the accurate description of the *current* implementation
> until those phases land.

How Thalos behaves at and on a planetary surface. Two parts:

- **On foot (EVA)** — on-foot / walking / rover gameplay: ground
  physics, what "landed" means for the canonical simulation, how
  heightfield data flows from terrain to physics, and how the map /
  trajectory view treats a surface-bound craft. (§§1–7 below.)
- **Landing & impact destruction** — landed-*ship* mechanics: powered
  descent, terrain collision (`SweptCcd` anti-tunneling), and
  whole-craft impact destruction. (The part of the same name at the end
  of this doc; formerly `landing.md`.)

The on-foot part is a **diagnosis + plan** document. Sections 2–3
describe the state of the code today (with file:line citations);
sections 4–6 lay out the target architecture and the migration order.
Open questions are tracked in §7 and must be resolved before the
corresponding work items move from "research" to "implement."

> **Update (2026-05): EVA controller reimplemented.** The naive
> "kinematic terrain follower" described in §2.1 has been replaced by a
> real kinematic **character controller** (`step_eva_controller` in
> `crates/game/src/player_controller.rs`). It simulates the player in the
> body's **body-fixed (rotating) frame** — tracking a body-fixed position
> + surface-relative velocity — and runs a grounded/airborne state machine
> with surface gravity (`g = μ/r²`): camera-relative walk/sprint (WASD +
> Shift), jump (Space), walking off ledges into a ballistic fall, and
> landing. The body-fixed frame is the core fix — surface velocity is the
> player's walking speed (m/s) instead of the inertial co-rotation drag
> `ω×r` (km/s) — which eliminated the walk freeze, the height-query
> sea-level teleport, and the warp explosion. Rest detection
> (`PlayerControllerState::is_at_rest`) gates time warp KSP-style (warp
> above 1× only when standing still; movement drops warp; 100× surface
> cap) and drives the on-foot HUD pill (`hud/eva_panel.rs`). §2.1 below is
> retained as the pre-rewrite diagnosis; §§4–6 record the design intent
> this implementation follows.

Scope: on-foot gameplay only. In-cockpit / interior IVA is not in
scope for this pass — the EVA capsule is the only character system
covered. Ship-on-surface physics is touched only where it intersects
the shared infrastructure (height source, BodyFixed mode, trajectory
suppression); a separate pass will redesign landed-ship mechanics.

## 1. Goals

- Walking on the surface tracks the rendered ground, with no visible
  capsule-vs-terrain gap and no positional jitter.
- Time warp works while standing on the surface (KSP-style "inertial
  surface mode") without requiring a launch.
- The orbital trajectory line is suppressed when the player is
  body-fixed; a surface marker / ground track replaces it.
- Heightfield queries used by physics and the heightfield drawn by
  the renderer share a single source of truth, so they agree
  exactly. The interface tolerates a future migration to GPU-side
  generation without churning the call sites.
- The system is debuggable in production (`bevy_brp` queries against
  `AvianAuthority`, `AuthorityMode`, `Position`, `LinearVelocity`)
  and the few invariants that must hold (BodyFixed iff trajectory
  suppressed, etc.) are checkable from outside the running game.

## 2. Current state

### 2.1 Walking system — `crates/game/src/player_controller.rs`

`walk_eva_on_terrain` ([player_controller.rs:239](crates/game/src/player_controller.rs:239))
owns the EVA capsule's position directly. The body is a kinematic
Avian rigid body with `CustomPositionIntegration` + rotation locked
([local_physics.rs:406](crates/game/src/local_physics.rs:406)); the
walking system writes `Position`, `Rotation`, `LinearVelocity`, and
zeroes `AngularVelocity` each frame.

Per-frame loop:

1. Read camera-relative walk input, project onto the tangent plane at
   `position.0.normalize()` (radial-out from body centre).
2. Step `position` forward by `surface_velocity * sim_dt +
   walking_velocity * real_dt`. `surface_velocity = ω × position`
   carries the player with the rotating surface; walking adds on top
   in real time so warp doesn't speed it up.
3. Convert the stepped direction to body-fixed and query
   `rendered_height_m(surface, dynamic_state, dir, tile_lod_m=0.5)`
   ([player_controller.rs:318](crates/game/src/player_controller.rs:318)).
4. Glue altitude to `body.radius + terrain_h + half_height +
   foot_clearance` ([player_controller.rs:324](crates/game/src/player_controller.rs:324)).
5. Write `LinearVelocity = surface_velocity + walking_velocity` so the
   canonical readback reports correct ground velocity.

There is no gravity force, no airborne ballistic phase, no contact
resolution. The system is "kinematic terrain follower." A jump or
cliff fall would teleport instantly — explicitly called out as a
known limitation in the doc comment
([player_controller.rs:233–238](crates/game/src/player_controller.rs:233)).

### 2.2 Local-physics authority — `crates/game/src/local_physics.rs`

Three-way `AvianRole` enum ([local_physics.rs:98](crates/game/src/local_physics.rs:98)):
`Paused` (warp or BodyFixed), `AttitudeOnly` (1× coast, Kepler owns
translation), `Full` (1× thrust or terrain contact). `avian_role_from_inputs`
([local_physics.rs:189](crates/game/src/local_physics.rs:189)) is the
pure predicate.

For EVA, every local-physics force/snap system either short-circuits
or goes through harmlessly:

- `snap_avian_from_canonical` early-returns for `VesselKind::Eva`
  ([local_physics.rs:714](crates/game/src/local_physics.rs:714)).
- `apply_local_forces` early-returns for `VesselKind::Eva`
  ([local_physics.rs:984](crates/game/src/local_physics.rs:984)).
- `readback_local_craft` *always* mirrors Avian → canonical for EVA
  ([local_physics.rs:1135](crates/game/src/local_physics.rs:1135));
  for ships it gates on `owns_translation`.

The EVA capsule's `Collider` is removed at spawn
([local_physics.rs:408](crates/game/src/local_physics.rs:408)) so
Avian's contact graph never finds it. This was an explicit fix for
an earlier sliding bug
([local_physics.rs:381–386](crates/game/src/local_physics.rs:381)).

EVA never enters `AuthorityMode::BodyFixed`. The capsule is spawned
in body-centered inertial coordinates and stays there; `BodyFixed`
is reserved for ships that collapsed after `stable_contact` in
`collapse_or_constrain_warp` ([local_physics.rs:1251](crates/game/src/local_physics.rs:1251)).

### 2.3 Height query — `crates/terrain_render/src/pipeline.rs`

`rendered_height_m(surface, dynamic_state, dir, tile_lod_m)`
([pipeline.rs:500](crates/terrain_render/src/pipeline.rs:500)) is the
single CPU-side "what does UDLOD draw here?" entry point. Three
stages:

1. Nearest-cubemap base + bilinear from the baked cubemap.
2. Dynamic overlays (ice caps, aeolian bedforms).
3. LOD-adaptive procedural detail: 3-D fBm noise + 2-D erosion filter,
   octave count chosen by `detail_plan_for_lod(tile_lod_m)`
   ([pipeline.rs:150](crates/terrain_render/src/pipeline.rs:150)).

The renderer's `PipelineTileProvider::request_tile`
([pipeline.rs:210](crates/terrain_render/src/pipeline.rs:210)) calls
the same evaluation function (`compute_tile_pixels` →
`evaluate_pixel`) and bakes the result into the UDLOD attachment
atlas. Per-tile `tile_lod_m` comes from
`tile_lod_m(body, coord, size, border)`
([pipeline.rs:712](crates/terrain_render/src/pipeline.rs:712)):
`body.radius * face_radians / inner_texels`. At Thalos
(radius ≈ 3186 km, `LOD_COUNT=16`, 512² tiles, border 2) the deepest
LOD reaches `tile_lod_m ≈ 0.15 m`.

The player query passes `tile_lod_m=0.5` directly
([player_controller.rs:37](crates/game/src/player_controller.rs:37)).
The terrain collider patch passes `tile_lod_m = vertex_spacing`
([rendered_height.rs:108](crates/terrain_render/src/rendered_height.rs:108)).
A render tile is only forced to produce the *same* heights as the
player query when the renderer's chosen `tile_lod_m` matches the
player's 0.5.

### 2.4 Trajectory rendering — `crates/game/src/flight_plan_view/`

`render_trajectory` ([render.rs:35](crates/game/src/flight_plan_view/render.rs:35))
draws the predicted flight plan unconditionally whenever
`MapSnapshot::flight_plan` exists. It does not inspect
`MapSnapshot::crafts[0].authority`, so a landed or walking craft
still gets an orbital line drawn (frozen at whatever was last
predicted before authority transitioned).

### 2.5 Ground LOD swap — `crates/game/src/rendering/ground_terrain.rs`

The impostor → UDLOD terrain handoff fires at `4 × radius` camera
distance ([ground_terrain.rs:125](crates/game/src/rendering/ground_terrain.rs:125)).
Each body's UDLOD `TerrainConfig` has `LOD_COUNT=16` with 512²
tiles ([ground_terrain.rs:50–56](crates/game/src/rendering/ground_terrain.rs:50)),
atlas capacity 256 tiles ([ground_terrain.rs:64](crates/game/src/rendering/ground_terrain.rs:64)).

## 3. Diagnosed issues

Six observed problems. Each names a primary suspect, the evidence,
and what work it takes to verify or fix. Items 3.1 and 3.2 are
hypotheses requiring on-device verification before commitment.

### 3.1 Capsule above rendered terrain

**Symptom**: visible gap between the capsule's feet and the rendered
ground (clearer in screenshot 2 than 1; screenshot 1 is partially
camera-tilt + self-shadow).

**Hypothesis (primary)**: LOD mismatch between the player's height
query and the resident UDLOD tile.

- Player query at `tile_lod_m=0.5` engages the full 5-octave detail
  cascade ([pipeline.rs:162](crates/terrain_render/src/pipeline.rs:162)
  via `detail_plan_for_lod`).
- The rendered tile at the player's location uses whatever LOD UDLOD
  has loaded. If the resident tile is at a coarser LOD (small
  `LOD_COUNT` level), its `tile_lod_m` is larger, fewer octaves get
  evaluated, and the rendered surface sits *below* the player's
  query height by the missing detail amplitude.

**Secondary hypothesis**: atlas-residency lag. UDLOD requested the
fine-LOD tile but the async bake hasn't completed; the renderer is
still drawing the previous LOD. The player query, being CPU-side
and synchronous, sees the fresh value.

**Verification**: run `just game`, watch the
`PipelineTileProvider tile lod=… tile_lod_m=…` log lines for the
tiles UDLOD actually requests at EVA camera distance (~6 m). If
`tile_lod_m` near the player is consistently > 1, fix is in
UDLOD's LOD-selection / view config; if it bounces around or stays
coarse only briefly, fix is in async-bake gating.

**Tile-seam evidence**: the discontinuity in screenshot 2 (lower
right, where two yellow patches sit at clearly different heights) is
a known UDLOD artifact when adjacent tiles render at different
LODs. Same root cause as the player gap.

### 3.2 Walking jitters back to initial position

**Symptom**: pressing forward briefly displaces the player, who then
snaps back to roughly the original spot.

**Not yet diagnosed.** The walking math is internally consistent
(see derivation in the walk system's doc comment); the obvious
overwrite paths all early-return for EVA. Open candidates:

- **Avian solver writeback path**. `writeback_solver_bodies`
  ([avian3d-0.6.1/src/dynamics/solver/solver_body/plugin.rs:263])
  writes `pos.0 += solver_body.delta_position` and
  `lin_vel.0 = solver_body.linear_velocity` for *every* SolverBody —
  `CustomPositionIntegration` only excludes the *integrator's*
  position update, not solver writeback. With the collider removed,
  `delta_position` should be zero, but `linear_velocity` is rewritten
  every frame from solver state. Need to confirm what value the
  solver caches.
- **System-order race**. `walk_eva_on_terrain` (in
  `PlayerControllerPlugin::Update`) and the local-physics chain (in
  `GameLocalPhysicsPlugin::Update`) both sit in `SimStage::Physics`
  with no explicit `.before/.after` between them. If Avian's
  `PhysicsSchedule` runs between them, intermediate state may
  overwrite walking's `Position` write.
- **Camera-vs-input frame mismatch**. The walk direction is built
  from the camera's `Transform.rotation` projected onto the body
  tangent. If the camera frame and the body-centered inertial frame
  drift (BigSpace floating origin), the walk direction could be
  systematically wrong.

**Verification**: BRP-watch `Position` and `LinearVelocity` on the
player capsule across 10 frames while holding `W`. If Position
oscillates frame-to-frame around a fixed point: solver writeback. If
it advances then resets in a longer cycle: schedule race. If
LinearVelocity is zeroed between walk writes: solver overwrite.

### 3.3 Trajectory line drawn while on the ground

**Symptom**: orbital trajectory remains visible behind a walking /
landed player.

**Root cause**: `render_trajectory`
([render.rs:35](crates/game/src/flight_plan_view/render.rs:35)) has
no authority gate. The fix is a single conditional: skip when
`snapshot.crafts[0].authority` is `BodyFixed`.

For EVA specifically, *the player is also never in BodyFixed*
(§2.2), so even adding the gate today wouldn't suppress the line.
EVA must enter BodyFixed first (see §4.2) before the gate has
anything to act on. The fix lands as a single coupled change.

### 3.4 No inertial-surface mode (cannot warp while landed)

**Symptom**: standing on the surface, time warp either does nothing
useful or destabilises the player position.

**Status**: most infrastructure exists but the EVA case isn't wired
in.

- `AuthorityMode::BodyFixed` exists and is honoured: warp at
  BodyFixed forces `AvianRole::Paused`
  ([local_physics.rs:199](crates/game/src/local_physics.rs:199)),
  preventing Avian from numerically integrating the body's
  rotation under a large `dt`.
- `collapse_to_body_fixed` transitions ships after stable contact
  ([local_physics.rs:1251](crates/game/src/local_physics.rs:1251)).
- EVA bypasses all of this: the walking system writes inertial
  Position directly, canonical state mirrors it via `readback_local_craft`,
  authority stays `OnRails`.

**Fix**: route the EVA controller through the same `BodyFixed`
mechanism — the walking system mutates the body-fixed *pose*, not
the inertial position. Then the existing `Paused` gate handles warp
correctly for free. See §4.2.

### 3.5 Terrain reads as uniform yellow

**Not a physics bug — authoring.** The CPU color cascade
([pipeline.rs:603](crates/terrain_render/src/pipeline.rs:603)) at low
flat ground (`h ≈ sea_level + 0`, normal_y ≈ 1) collapses to a mix
of `SAND_COLOR (0.80, 0.70, 0.60)` and `GRASS_COLOR2 (0.40, 0.50,
0.20)` — exactly the yellow we see. This is the
`AgingOceanicHomeworld` archetype's authored low-altitude appearance.

If the visual target is "ground reads with local material variation"
the work is in `terrain_color_cascade` and the archetype's biome
authoring, not in the renderer. Out of scope for this doc; flagged
here because the user's "low res, uniform, bright" complaint conflates
this with the geometric LOD issue (§3.1) — they are independent.

### 3.6 Visible low geometric resolution

**Symptom**: flat tiles, visible at-scale relief is much coarser than
the player query implies.

Same root cause as §3.1. The geometry the renderer is drawing comes
from whatever LOD tile is resident in the atlas; if UDLOD isn't
requesting the deepest LOD at the camera distance, the mesh is
coarse regardless of how detailed the cubemap baker *could* make a
tile. Resolution diagnosis goes hand-in-hand with the gap diagnosis.

## 4. Architecture target

Four design moves. None is a complete rewrite of the simulation, but
together they unify the EVA case with the existing ship-on-surface
machinery, decouple the height interface from its current CPU
implementation, and make the trajectory / map view a clean function
of authority mode.

### 4.1 `HeightSource` interface

Today every physics consumer calls `rendered_height_m` directly. With
GPU-side generation arriving, the truth will live on GPU; some
consumers will want to read back, others may want to dispatch their
own compute job.

Introduce a single trait, in `thalos_terrain_render`:

```rust
pub trait HeightSource: Send + Sync {
    /// Height in metres above the body's reference radius, evaluated
    /// at `dir` (body-fixed unit vector). `tile_lod_m` is a hint at
    /// the spatial scale the caller cares about — implementations may
    /// use it to choose between LOD levels or to bound a compute
    /// kernel's octave count. Returns `None` if no value is available
    /// for this direction (e.g., GPU tile not yet resident); callers
    /// fall back to a coarse approximation.
    fn sample_height_m(&self, dir: Vec3, tile_lod_m: f32) -> Option<f32>;
}
```

Two impls today:

- `CpuPipelineHeightSource` — wraps `rendered_height_m`. Always
  returns `Some(_)`. Used until GPU generation lands.
- `BakedCubemapHeightSource` (fallback) — base cubemap only, no
  detail. Returned by other impls when the live data isn't ready.

Future impl:

- `GpuAtlasMirrorHeightSource` — maintains a CPU mirror of resident
  UDLOD atlas height attachments, populated by a download callback
  on tile-bake completion. Sample = bilinear lookup in the mirrored
  texel buffer. Falls back to `BakedCubemapHeightSource` when no
  tile is resident at `dir`.

**Consumer migration**: every call site that today takes a
`&PlanetSurface` + `&DynamicSurfaceState` + dir + tile_lod_m takes a
`&dyn HeightSource` + dir + tile_lod_m instead. Resource registry
gains `HeightSourceRegistry: SecondaryMap<BodyId, Arc<dyn HeightSource>>`
mirroring the existing `TerrainSurfaceRegistry` shape. Call sites
to migrate:

- `walk_eva_on_terrain` ([player_controller.rs:318](crates/game/src/player_controller.rs:318))
- `agl_above_rendered_surface` ([local_physics.rs:251](crates/game/src/local_physics.rs:251))
- `build_rendered_terrain_patch` ([rendered_height.rs:97](crates/terrain_render/src/rendered_height.rs:97))
- `spawn_player_avian_body`'s EVA spawn altitude ([local_physics.rs:327](crates/game/src/local_physics.rs:327))

**GPU-readback overhead** (answering the cost question): the dominant
cost of `GpuAtlasMirrorHeightSource` is the one-time texel download
per tile generation, not per-query lookup. Per-frame altitude query
is bilinear texel fetch — negligible. Collider-patch rebuild (16K
samples) is one parallel scan of mirror memory — sub-millisecond.
Memory budget for height-only mirroring: packed RG16 × 512² × 256 slots ≈
256 MB worst case for the game height path (`R16` and `R32Float` remain
supported for older/debug providers); reduced in practice by mirroring only the
focused-body / nearest-LODs slice that physics actually queries. Tile-residency
lag (~1 frame
after GPU bake) is the only correctness gotcha; the fallback to
`BakedCubemapHeightSource` covers it without a teleport.

### 4.2 EVA as a body-fixed pose

EVA's canonical state becomes `AuthorityMode::BodyFixed { body, pose
}` with `BodyFixedPose` carrying the player's body-fixed direction
+ heading. The walking system mutates `pose`, not inertial Position.

Effects:

- `avian_role_from_inputs`'s existing BodyFixed check (`Paused` for
  EVA) replaces the `terrain_attached`-driven Full role. Avian's
  integrator stops touching the capsule entirely.
- `walk_eva_on_terrain` no longer fights the solver writeback path:
  it mutates canonical's pose; the snap chain runs (now harmless,
  since `Paused` snaps Avian from canonical each frame).
- Warp works for free: BodyFixed honours warp via the existing
  `Paused` gate.
- The trajectory gate from §4.3 trips automatically: BodyFixed
  craft don't get a Kepler line drawn.
- Re-launch becomes the explicit reverse transition — same path a
  ship would take leaving the surface. Triggered by something like a
  jump or "stand up from rover" intent. Out of scope for this pass;
  re-launch from EVA is `unimplemented!()` until we have ascent
  mechanics for the on-foot case.

The walking-physics work itself stays largely the same — read
input, advance pose in tangent plane, sample height for altitude,
clamp. The difference is that "advance pose" now writes a
body-fixed direction into `BodyFixedPose` instead of an inertial
DVec3.

**Risk**: there is exactly one open question (§3.2's jitter) that
might survive this refactor. If the jitter root cause is in
camera-vs-input frames (the third candidate), it persists. The §3.2
investigation must complete before this design commits — otherwise
we ship the refactor on a guess and the user reports the same bug.

### 4.3 Surface mode for trajectory + map view

When `MapSnapshot::crafts[0].authority` is `BodyFixed`:

- `render_trajectory` ([render.rs:35](crates/game/src/flight_plan_view/render.rs:35))
  returns early before drawing the orbital line.
- Map view replaces the orbital track with a **surface marker** at
  the body-fixed lat/lon, plus a **ground track** trail of where the
  craft has been over the last N seconds (drawn on the body surface,
  not in 3-D orbital space).
- Maneuver-node UI is hidden (you can't plan a burn while landed).
  Re-enables on transition out of BodyFixed.

On launch (BodyFixed → OnRails or LocalRigidBody), the existing
flight-plan prediction system rebuilds and the trajectory line
returns naturally.

Detail-of-rendering decisions for the ground track (line style,
fadeout, what counts as "trail") live in the map-view follow-up;
this doc only commits to the BodyFixed → surface-mode rule.

### 4.4 Debuggability invariants

Two predicates that must hold and should be BRP-queryable:

- **`BodyFixed ⇔ no Kepler line drawn`**. If a render system is
  drawing a Kepler line for a BodyFixed craft, that's a bug; flag
  with a warn-once.
- **`EVA active ⇒ AuthorityMode::BodyFixed`**. After §4.2 lands,
  EVA in any other authority mode is a bug.

Register `BodyFixedPose`, `AvianRole`, `AvianAuthority` for `Reflect`
so a remote agent can inspect them via `world_get_resources` and
`world_get_components`. (Today only `CraftStateMirror` is mirrored;
see [docs/tooling.md](docs/tooling.md) for the registration policy.)

## 5. Migration plan

Ordered for lowest-risk wins first. Items marked **[research]** are
investigations that must complete and report before the corresponding
**[build]** item starts.

### Phase A — Diagnose and unblock

1. **[research]** Diagnose §3.1 / §3.6 LOD selection. Run `just
   game`, capture `PipelineTileProvider tile lod=… tile_lod_m=…` log
   for tiles UDLOD requests at EVA camera distance. Decision tree:
   - If `tile_lod_m` stays coarse (> 1 m) → fix `TerrainViewConfig`
     in [ground_terrain.rs:188](crates/game/src/rendering/ground_terrain.rs:188);
     UDLOD's LOD-error metric isn't accounting for sub-metre camera
     distance.
   - If `tile_lod_m` reaches 0.5 m only after a delay → async-bake
     prioritisation; pre-prioritise tiles inside a focus radius.
   - If `tile_lod_m` is correct but visual is still flat → cascade /
     erosion is at fault, not the LOD plan.
2. **[research]** Diagnose §3.2 jitter. BRP-watch `Position`,
   `LinearVelocity`, `LocalCraftBody` on the player capsule across
   10 frames while walking forward. Decision tree:
   - Position oscillates frame-to-frame → solver writeback path;
     mitigation = remove `SolverBody` from EVA entity (also kinematic
     should drop out of solver loop with collider removed; verify
     this is actually what Avian does).
   - Position advances then resets in cycles → SimStage::Physics
     race; add explicit `.after(crate::local_physics::*)` to walking
     systems.
   - LinearVelocity zeroed between walks → solver overwrite;
     mitigation as above.
   - None of the above → check movement_direction frame: print the
     resolved `right`/`forward` vectors and compare against expected
     camera basis.
3. **[build]** Add the §4.3 trajectory gate (BodyFixed → skip
   render). One conditional in [render.rs:42](crates/game/src/flight_plan_view/render.rs:42).
   Lands even before EVA enters BodyFixed — ships in BodyFixed
   already benefit.

### Phase B — HeightSource interface

4. **[build]** Define `HeightSource` trait in `thalos_terrain_render`,
   implement `CpuPipelineHeightSource` and `BakedCubemapHeightSource`,
   add `HeightSourceRegistry` resource populated alongside
   `TerrainSurfaceRegistry`.
5. **[build]** Migrate the four call sites in §4.1 to take
   `&dyn HeightSource`. No behaviour change — just routing through the
   trait. Test: bake check still passes; player altitude unchanged.

### Phase C — EVA as body-fixed pose

6. **[build]** Add `BodyFixedPose` carrying body-fixed direction +
   heading; extend `AuthorityMode::BodyFixed` if it doesn't already
   own it; register for `Reflect`.
7. **[build]** Replace the EVA spawn path in
   `spawn_player_avian_body` ([local_physics.rs:312–350](crates/game/src/local_physics.rs:312))
   with a `BodyFixed`-mode install: canonical authority transitions
   straight to BodyFixed, no inertial state install. The Avian
   capsule is still spawned (so the rendering / camera-focus path
   has an entity to track) but its Position is snapped from canonical
   each frame by the existing `Paused`-mode snap.
8. **[build]** Rewrite `walk_eva_on_terrain`
   ([player_controller.rs:239](crates/game/src/player_controller.rs:239))
   to mutate `BodyFixedPose` instead of inertial Position. Height
   query is unchanged (still via the HeightSource registry).
9. **[build]** Remove the `is_eva` early-return special case from
   `readback_local_craft` ([local_physics.rs:1115](crates/game/src/local_physics.rs:1115));
   it's no longer needed because EVA owns canonical state directly
   through pose mutation.

### Phase D — Surface map view

10. **[build]** Map view detects `BodyFixed` and renders the surface
    marker + ground track in place of the orbital line. Specifics
    deferred to a map-view follow-up; this item commits the
    detection + the orbital-line suppression only.
11. **[build]** Hide maneuver-node UI under BodyFixed; restore on
    transition out.

### Phase E — GPU readback (deferred until GPU gen lands)

12. **[build]** `GpuAtlasMirrorHeightSource`: mirror resident height
    attachments into CPU memory on download-completion callback.
    Bilinear sample in mirror buffer; fall back to
    `BakedCubemapHeightSource` for non-resident tiles.
13. **[build]** Swap the registry's default impl from
    `CpuPipelineHeightSource` to `GpuAtlasMirrorHeightSource` for
    bodies with GPU-generated terrain. CPU pipeline impl remains the
    fallback / test impl.

## 6. Dependency graph

```
Phase A:  1 ─┐
          2 ─┤
          3 ─┘  (independent; ships in any order)
            │
Phase B:    └── 4 ─ 5
                    │
Phase C:            └── 6 ─ 7 ─ 8 ─ 9
                                    │
Phase D:                            └── 10 ─ 11
                                          │
Phase E:                                  └── 12 ─ 13
```

Phase A items 1 and 2 are research; their outcomes shape detail in
later phases (item 1 may add a sub-item under Phase C; item 2 may
add a sub-item before Phase C or invalidate the design entirely if
the jitter root cause is something the BodyFixed refactor doesn't
address).

## 7. Open questions

- **What is the §3.2 jitter root cause?** Must resolve in Phase A
  item 2. If the cause is something the BodyFixed refactor doesn't
  address (e.g., camera-frame issue), Phase C does not fix it and
  needs an additional item.
- **What `tile_lod_m` is UDLOD actually selecting at EVA camera
  distance?** Phase A item 1. If the renderer isn't reaching the
  deepest LOD, the §4.1 HeightSource interface won't fix the visible
  gap on its own — the renderer needs its own fix to match the
  player query.
- **When does GPU-side terrain generation land?** Sets the urgency
  on Phase E. If "soon," Phase E moves up; if "after several other
  milestones," Phase E is genuinely deferred and `CpuPipelineHeightSource`
  stays the default for a long time.
- **Re-launch from EVA semantics.** Out of scope for this pass, but
  the BodyFixed → OnRails reverse transition has to be designed
  before EVA can become anything other than a one-way trip to the
  ground.
- **Ship-on-surface physics.** Ships in BodyFixed today are
  numerically frozen pose; rover wheels, sliding craft, scaled-warp
  physics on the surface are all unaddressed. Touched here only
  where the trajectory-suppression rule applies. Landed-ship
  mechanics — powered descent, terrain collision (`SweptCcd`
  anti-tunneling), and impact destruction — are covered in the
  **Landing & impact destruction** part below (formerly `landing.md`);
  rover wheels / sliding / surface-warp remain open there.

### Phase A findings (2026-05-17, static)

Live observation aborted — auto-bake's
`cargo run --release -p thalos_bake_dump` got rustc-SIGKILL'd, most
likely OOM under parallel release builds. Findings below are from
static traces of the cited code paths.

**Item 1 — LOD selection (§3.1 / §3.6).**
[pipeline.rs:720](crates/terrain_render/src/pipeline.rs:720) clamps
`tile_lod_m = body.radius * face_radians / inner_texels` to a **1 m
minimum** via `.max(1.0)`. On Thalos the natural value at LOD 15 is
0.30 m, so every requested tile bakes at `tile_lod_m ≥ 1.0` — the
§3.1 hypothesis ("LOD 15 → `tile_lod_m=0.15`") doesn't hold against
the current code. The detail cascade then caps at
`MAX_DETAIL_OCTAVES=5`
([pipeline.rs:89](crates/terrain_render/src/pipeline.rs:89)) for any
`tile_lod_m ≤ 25 m`; on Thalos that's LOD ≥ 9 (natural ≈ 19 m). So
the player query (`tile_lod_m=0.5`, full 5-octave cascade) and any
rendered tile at LOD ≥ 9 produce **identical** detail heights. For
the player to sit above the ground from a cascade-depth mismatch, the
resident tile must be at **LOD ≤ 8** — async lag of ≥ 7 levels below
the LOD 15 that
[tile_tree.rs:307](crates/udlod/src/terrain_data/tile_tree.rs:307)
requests at ~6 m view (`load_distance/2^lod = 7.965 Mm/2^15 ≈ 243
m`). None of the §5 item 1 branches matches cleanly; closest is (b)
"async-bake prioritisation," but the lag needed is several levels,
not "delay to 0.5 m." A possibility §3.1 doesn't enumerate:
sample_base_with_dynamic's `dynamic_lod = log2(tile_lod_m)`
([pipeline.rs:510](crates/terrain_render/src/pipeline.rs:510)) is
`−1.0` for the player vs `0.0` for the renderer, so the base cubemap
mip level differs even when the detail cascade matches — could
contribute a sub-metre offset. **Status**: live observation still
needed to commit; the structural mismatch above is the headline
result for the spec rewrite.

**Item 2 — Walking jitter (§3.2).** All three §3.2 candidates fail
under static trace.

(a) *Avian solver writeback.*
[`prepare_solver_bodies`](file:///Users/korbin/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/avian3d-0.6.1/src/dynamics/solver/solver_body/plugin.rs)
line 211 resets `delta_position = 0` each step; `integrate_positions`
skips `CustomPositionIntegration` (line 504); `integrate_velocities`
early-returns for kinematic (line 357). With no `Collider`, no
contacts touch `delta_position`. `writeback_solver_bodies` (line 263)
becomes `pos.0 += 0 + (old_com − new_com)`, and
`LockedAxes::ROTATION_LOCKED` keeps `delta_rotation ≈ identity` so
the CoM term ≈ 0. Avian cannot move the EVA capsule.

(b) *System-order race.* `walk_eva_on_terrain`
([player_controller.rs:62](crates/game/src/player_controller.rs:62))
is the **only** writer to Position matching `With<PlayerControllerBody>`.
Every other Position writer in the local-physics chain
(`snap_avian_from_canonical:714`, `apply_local_forces:984`,
`debug_surface_drop:818`, `rebase_bubble_to_dominant_body:468`)
early-returns for `VesselKind::Eva` or for non-body-change. Avian's
`PhysicsSchedule` runs in `FixedPostUpdate`, before `Update`'s
`SimStage::Physics`, so its no-op writeback can't sequence after
walk. No second writer → no race.

(c) *Camera-vs-input frame mismatch.* Camera runs in
`SimStage::Camera` (after Physics + Sync), so walk reads the
previous-frame Transform. Projection in `movement_direction`
([player_controller.rs:398](crates/game/src/player_controller.rs:398))
removes the radial component from `(rotation * NEG_Z)` and
normalises — no obvious drift mode that produces a snap-back to the
original spot.

**Result**: §4.2's Risk paragraph applies — Phase C is **blocked, not
advanced**. Phase C reshapes who owns canonical Position but doesn't
add new writers or reorder the schedule; whatever's actually causing
the jitter survives the refactor unchanged. Candidates a static pass
can't decide, for the live follow-up:

- `body_state.angular_velocity` frame for the `surface_velocity = ω
  × position` term
  ([player_controller.rs:301](crates/game/src/player_controller.rs:301))
  — at Thalos equator ~232 m/s, so an inertial-vs-body-fixed
  mismatch would dominate per-frame motion.
- Camera `Transform.rotation` source frame under BigSpace floating
  origin — cell hops could rotate the apparent forward direction.
- Time source under pause/warp is now explicit: canonical/local systems
  consume `SimClock`, while camera/presentation systems use `Time<Real>`.

Resolving item 2 needs `just game` running. Recovery path before
re-launch: `cargo clean --release` to wipe the SIGKILL'd partial
state, then pre-build with `cargo build --release -p
thalos_bake_dump -j 4` to limit memory pressure before the
auto-bake fires.

### Phase A resolution (2026-05-17, fix landed)

The async-bake-lag refinement of hypothesis (§3.1) is the one that
ships: when the player query passes a hard-coded fine `tile_lod_m =
0.5` while UDLOD's atlas still shows a coarser ancestor at the
player's bucket, the CPU evaluates octaves the GPU mesh doesn't have,
floating the character by the missing-octave amplitude. The 1m clamp
on `tile_lod_m` makes the 5-octave cap reach at LOD ≥ 9 (~19 m tile
spacing on Thalos), but the resident LOD during the first seconds
after a fresh atlas spin-up routinely drops below that — exactly the
"a few octaves missing, ≤ 10 m float" the screenshot showed.

Fix shape: the player's CPU height query now reads the resident
`atlas_lod` at its own direction and derives the *renderer's* current
`tile_lod_m` from it, so the two evaluations agree on octave count by
construction. Implementation:

- `TileTree::best_resident_atlas_lod(world_position, model)` returns
  the resident ancestor LOD ([crates/udlod/src/terrain_data/tile_tree.rs](crates/udlod/src/terrain_data/tile_tree.rs)).
- `TileAtlas::model()` / `lod_count()` / `attachment_configs()`
  expose the geometry that the renderer-side `tile_lod_m` formula
  needs, plus `TerrainModel::scale()` is now `pub` so callers can
  multiply through to metres-per-texel without re-deriving the
  cube-sphere math ([crates/udlod/src/terrain_data/tile_atlas.rs](crates/udlod/src/terrain_data/tile_atlas.rs),
  [crates/udlod/src/math/terrain_model.rs](crates/udlod/src/math/terrain_model.rs)).
- `thalos_terrain_render::renderer_tile_lod_m_at(atlas, tree,
  world_pos)` packages "what tile_lod_m does the renderer use here?"
  into a single call so consumers don't repeat the LOD →
  metres-per-texel arithmetic ([crates/terrain_render/src/pipeline.rs](crates/terrain_render/src/pipeline.rs)).
- Three consumers route through it: `walk_eva_on_terrain`
  ([crates/game/src/player_controller.rs](crates/game/src/player_controller.rs)),
  the EVA branch of `spawn_player_avian_body`
  ([crates/game/src/local_physics.rs](crates/game/src/local_physics.rs)),
  and the GND-altitude readout
  ([crates/game/src/hud/orbital_panel.rs](crates/game/src/hud/orbital_panel.rs)).
  All three fall back to `tile_lod_m = 10 000` (procedural detail
  disabled) when no tile is resident yet — the GPU is showing only
  the base cubemap during that warm-up window, so the CPU sample
  matches the visible mesh through the lag instead of pinning to a
  higher full-detail surface that hasn't been drawn yet.

This was *not* the §4.1 HeightSource interface — it was a narrower
fix that kept `rendered_height_m`'s signature unchanged and just fed
it the right LOD parameter. Phase E's resolution below supersedes
this as the primary gameplay height path; `renderer_tile_lod_m_at`
is now a diagnostic / scale-query helper.

§3.2's walking-jitter root cause is still unresolved; this fix
doesn't touch the walk math. The shoreline component of §3.5 / the
"seam" complaint is mitigated independently by bumping
`WATER_MESH_SUBDIVISIONS` from 6 to 7 (4× the triangle count) so the
water/terrain intersection no longer reads as a polygon shoreline at
walking distance.

Verified visually post-fix: capsule sits flush against the rendered
terrain on Thalos at the sub-stellar spawn point, the prior hard
horizontal seam under the player is gone, and the HUD GND readout
still settles at 1 m (capsule centre 0.98 m above terrain → 1 m
rounded; foot clearance ~8 cm).

Known limitation introduced by this approach:
`TileTree::best_resident_atlas_lod` uses a single deepest-LOD bucket
lookup. For positions within ~`(tree_size/2) × tile_size_at_deep_lod`
of the camera (i.e. EVA + HUD + spawn at low altitude) the modulo
wrap lands on the correct bucket. Routing higher-altitude or
camera-distant queries through this helper without revisiting the
walk-to-deeper-LOD logic from the GPU `lookup_best` shader will
silently return a wrong bucket's `atlas_lod`. Restore the full walk
if `agl_above_rendered_surface` or another consumer ever needs
correct results for ships further from the focus.

### Phase E resolution (2026-05-17, build landed)

Phase E moved ahead of the full GPU-generation milestone by treating
UDLOD's resident height atlas as the gameplay height source. The
renderer already uploads CPU-produced tiles and downloads GPU-written
tiles through the same `TileAtlas` slot path; gameplay now consumes a
mirror of that slot data instead of independently re-evaluating
`rendered_height_m` at an assumed LOD.

Implementation shape:

- `thalos_terrain_render::HeightSource` is the gameplay-facing
  interface. `CpuPipelineHeightSource` remains the direct
  `rendered_height_m` implementation; `BakedCubemapHeightSource`
  is the detail-free fallback.
- `GpuAtlasMirrorHeightSource` owns an `Arc<RwLock<GpuAtlasHeightMirror>>`
  plus a baked fallback. The mirror stores resident R16, packed RG16, or
  R32Float height tiles keyed by `TileCoordinate`, samples them bilinearly in
  body-fixed direction space, and falls back when no resident ancestor exists.
  The game path uses packed RG16 to avoid visible height-quantization
  contouring on broad, shallow terrain without requesting filterable float
  textures.
- `THALOS_TERRAIN_PROVIDER=flat` installs a `ConstantHeightSource(0 m)`
  instead of a GPU mirror and leaves the propagator's baked surface
  registry empty for that body. The diagnostic flat mesh, EVA spawn,
  terrain-collider patch, walking clamp, HUD GND height, and trajectory
  collision therefore all use the same reference sphere.
- `thalos_udlod::TileAtlas` now exposes loaded-tile metadata,
  attachment data, and per-slot revisions. The mirror sync system only
  clones height slots whose revision changed, so it does not copy the
  full atlas every frame.
- `HeightSourceRegistry` lives beside `TerrainSurfaceRegistry` in
  `thalos_physics_local`. `install_baked_planet` registers a
  `GpuAtlasMirrorHeightSource` for each procedural body, and
  `spawn_body_terrain` attaches the mirror handle to the UDLOD terrain
  entity so `ThalosTerrainPlugin` can refresh it after tile loads /
  GPU readbacks.
- EVA walking, EVA spawn altitude, debug surface drop, AGL terrain
  patch attach/detach/rebuild, and the HUD GND readout now route
  through `HeightSourceRegistry`. Terrain collider patches build from
  `&dyn HeightSource`, so future GPU-generated tiles drive contact
  geometry without changing the local-physics call site.

The older `renderer_tile_lod_m_at` seam remains exported for diagnostics
and any future caller that only needs the resident LOD scale. It is no
longer the primary gameplay height path.


## Landing & impact destruction

Spec for landing a **ship** on a planetary surface and for destroying a
craft that hits the surface too hard. This is the "landed-ship
mechanics" pass that [`surface.md`](surface.md)
explicitly defers (its scope is on-foot/EVA only; §7 lists
"Ship-on-surface physics" as needing its own spec). On-foot gameplay,
the `HeightSource` interface, and the surface map view live in that
document; this one covers ship descent, contact, and structural
failure.

This is a **diagnosis + plan** document in the same shape as
`surface.md`. §1–2 describe the code as it stands (with
file:line citations); §3–5 lay out the target and the first
implemented slice; §6 tracks open questions.

### 1. Goals

- A ship descending onto terrain **stops at the surface** instead of
  clipping through it, at any approach speed the player can reach
  (descent burns, botched suicide burns, ballistic re-entry).
- A ship that contacts the surface **above its impact tolerance is
  destroyed** rather than surviving as an indestructible tumbling
  rigid body.
- A gentle touchdown still settles and collapses to
  `AuthorityMode::BodyFixed` (the existing stable-contact path), so
  "landed" remains a real, warp-safe end state.
- Destruction is **legible**: the player can tell at a glance that the
  craft is dead, and control is locked until they recover.
- The model is a **first slice that generalises**: whole-craft
  destruction today, with a clear path to per-part crash tolerance
  reusing the staging part-graph.

### 2. Current state

#### 2.1 The local bubble already has collision geometry

`thalos_physics_local` spins up one Avian rigid body per craft and,
inside the 1x-only surface warp zone, attaches a `Kinematic` trimesh
terrain patch:

- The ship is a `RigidBody::Dynamic` compound collider built from the
  shipyard part graph in `build_ship_collider_primitives`: one primitive
  per rendered part, positioned from `Attachment`/`AttachNodes` rather
  than frame-late `GlobalTransform`s. Frustum-like pods, adapters, and
  engines use full-radius cylinders that envelop the visible geometry;
  tanks and decouplers use their native cylinder footprint.
  The same primitive list is stored for the F8 collider outline.
  `build_ship_collider_primitives`
  ([local_physics.rs:1688](crates/game/src/local_physics.rs:1688)),
  spawned by `spawn_local_craft_body`
  ([lib.rs:284](crates/physics_local/src/lib.rs:284)).
- `attach_terrain_patch_when_close`
  ([local_physics.rs:592](crates/game/src/local_physics.rs:592)) spawns
  a `RigidBody::Kinematic` trimesh patch (`spawn_terrain_collider_patch`,
  [lib.rs:236](crates/physics_local/src/lib.rs:236)) only when AGL is
  below `handoff_agl_m` (20 km) **and** `WarpLimits` says 1x is the
  highest legal warp level. Manually switching to 1x above that surface
  warp-lock zone does not build the terrain collider, and any already
  attached non-contact patch is detached when the craft leaves the zone.
  The collider body sits at the patch center, its mesh vertices are
  body-fixed offsets from that center, and its
  `Position`/`Rotation`/velocities track the rotating body each frame so
  `Position + Rotation * local_vertex` lands in the right
  body-centered-inertial position with metre-scale narrow-phase
  coordinates.
- Avian's contact solver runs only when Avian *owns translation*, i.e.
  `AvianRole::Full` — throttle active **or** terrain patch attached
  (`avian_role_from_inputs`,
  [local_physics.rs:234](crates/game/src/local_physics.rs:234)). The
  patch's mere existence inside the AGL band is the "contact is
  physically possible here" signal.

So the contact pair exists and is solved on approach. The gap is
**not** missing colliders.

#### 2.2 Clip-through is tunneling, not a missing collider

Avian's narrow phase enables **speculative collision** for every body
by default (`NarrowPhaseConfig::default_speculative_margin = Scalar::MAX`,
unbounded — the effective margin is velocity·dt). Speculative contacts
alone do **not** reliably stop a fast body against a thin trimesh:
Avian's own docs warn that speculative contacts "treat contact
surfaces like infinite planes" and "can still occasionally miss
contacts, especially for thin objects" — and a single trimesh patch is
exactly that thin object. A craft descending at tens-to-hundreds of m/s
moves metres per 1/60 s step and slips across/through the 64 m-spaced
triangles.

No body in the project sets `SweptCcd` (the opt-in geometric
time-of-impact sweep) — a grep finds CCD only in the unrelated Kepler
propagator. `SweptCcd` is the approach Avian's documentation
prescribes for "fast and small objects [that] can pass through thin
geometry such as triangle meshes."

Secondary contributors (not the headline):

- The patch is coarse: `patch_resolution = 65`,
  `patch_half_extent_m = 4096`
  ([lib.rs](crates/physics_local/src/lib.rs)). Sub-patch relief isn't in
  the collider.
- A known collider-vs-rendered-surface gap exists between the coarse
  collider patch and the GPU-rendered surface.

> **Performance note.** The terrain collider is a `Kinematic` trimesh
> whose pose is re-synced every frame (`sync_terrain_collider_pose`) so it
> co-rotates with the planet inside the body-centered Avian bubble. Avian
> therefore re-runs broad/narrow phase against every collider triangle each
> frame, and this is the **dominant CPU cost while on the surface** —
> measured at ~8 ms/frame (EVA on Thalos, 4K) with the old
> `patch_resolution = 129` (~32k triangles). The window covered far more
> ground than any resting craft contacts; `patch_resolution = 65`
> (~8k triangles, native-texel density preserved in a still-generous window
> around the craft) cuts that to ~2 ms with no gameplay change — the
> grounded-EVA capsule is placed kinematically by `step_eva_controller`,
> not by the collider. Diagnosed CPU-bound by toggling the ship camera's
> `is_active` (frame time unchanged → not GPU) and the sim pause (frame
> time dropped from ~18 ms to ~10 ms → the `SimStage::Physics` set, i.e.
> Avian). If finer collision is ever needed under a craft, prefer a small
> high-res window over a large one; the cost scales with triangle count.

#### 2.3 There is no structural-integrity model

`CraftState` ([canonical.rs:166](crates/physics_canonical/src/canonical.rs:166)),
`MassState`, and `ShipParameters`
([types.rs:57](crates/physics_canonical/src/types.rs:57)) carry no
health/integrity/tolerance of any kind. The ship collider sets no
`Restitution` (Avian default 0, so it is not a bounce). A hard impact
resolves as an ordinary offset contact impulse on a tall thin stack →
it tumbles, and because nothing can destroy it, it survives and spins.

`docs/simulation.md` already *specifies* the intended hook —
`SimEvent::Impact { craft, body, epoch, speed_m_s }`
([simulation.md:1262](docs/simulation.md:1262)) — but no `SimEvent`
pipeline is implemented yet.

#### 2.4 Gentle landing already works

`collapse_or_constrain_warp`
([local_physics.rs:1315](crates/game/src/local_physics.rs:1315)) tracks
stable contact and collapses to `AuthorityMode::BodyFixed` after 2.0 s
of continuous contact under 0.5 m/s linear / 0.05 rad/s angular with
throttle zero (`stable_contact_reached`,
[lib.rs:336](crates/physics_local/src/lib.rs:336)). So the *landed*
end-state is real; the missing pieces are surviving the descent
(§2.2) and a consequence for not surviving it (§2.3).

#### 2.5 Rendered pose is extrapolated across the physics overstep

Avian integrates the craft on a **fixed** timestep (`PhysicsPlugins::default()`),
while the render/main loop runs at a higher, variable rate. When Avian owns
translation (`AvianRole::Full` — powered descent / landing), the canonical
position read back each frame therefore *holds* for several render frames and
then jumps a whole fixed step. The ship camera rigidly follows the craft, so that
hold/jump reads as the **terrain juddering at the viewer's feet** while the ship
itself and the (parallax-free) sky look steady — and it worsens as the surface
nears, since the same sub-step offset subtends more pixels. (It is invisible in
the `OnRails`/`AttitudeOnly` coast above the 20 km handoff, where Kepler advances
the canonical state once per render frame.)

`update_player_ship_world_position`
([ship_view.rs](crates/game/src/ship_view.rs)) hides this by advancing the
*rendered* root position by the body-relative velocity across
`Time<Fixed>::overstep()` whenever `AvianAuthority::owns_translation()`. Only the
relative (descent) component is extrapolated: the heliocentric orbital velocity
(~30 km/s) already moves smoothly via the Kepler-evaluated body position in the
readback — only the body-relative descent stutters. Physics, the canonical state,
and the terrain collider are untouched, so the visual leads the collider by at
most one physics step.

Not covered by this: the orbit camera's spring-arm still judders once the boom is
clamped within `CAMERA_TERRAIN_MARGIN_M` of the surface (touchdown / a craft sat
below the terrain), because the clamp target chases the per-frame terrain height.
That is a separate camera-collision issue, not the fixed-step stutter.

### 3. Target architecture

#### 3.1 Anti-tunneling: SweptCcd on dynamic craft

Attach `SweptCcd::default()` (`SweepMode::NonLinear`,
`include_dynamic: true`, zero thresholds) to the ship's dynamic rigid
body, plus an explicit `Restitution(0.0)` so landings never bounce.
Non-linear sweep also covers a tumbling craft, not just pure
translation. CCD is cheap for a single player craft and is the
documented fix for the trimesh-tunneling case. Speculative collision
stays on as the cheap first line; swept CCD is the backstop the docs
recommend combining with it.

This is the single highest-leverage change: it converts "fall through
the planet" into "stop at the surface," which is the precondition for
*every* landing outcome (gentle settle or hard crash).

#### 3.2 Impact tolerance is a ship parameter

Add `impact_tolerance_m_s: f64` to `ShipParameters` — a KSP-style crash
tolerance in m/s of surface-relative approach speed. It is pure
physical data, consistent with the other fields there (thrust, MOI,
dry mass), and is the seam the future per-part model refines (the
craft tolerance becomes `min` over contacting parts).

First-slice value: a single forgiving constant pushed with the rest of
the ship stats. EVA is exempt (`f64::INFINITY`) — on-foot contact
damage is out of scope here and EVA does not use Avian contact
resolution anyway (`surface.md` §2.1).

#### 3.3 Whole-craft destruction is canonical state

The "destroyed" fact lives on `Simulation`, beside the other transient
craft bookkeeping (warp, prediction dirty flag, vessel kind):

```rust
// Simulation
hull_destroyed: bool,
last_impact_speed_m_s: f64,

pub fn is_destroyed(&self) -> bool;
pub fn mark_destroyed(&mut self, impact_speed_m_s: f64);
pub fn repair(&mut self);   // clears on respawn/teleport
```

Canonical ownership keeps the invariant "one craft state, one
authority": HUD, control gating, and the BRP mirror all read one
truth. It is deliberately *not* on `CraftState` (which is cloned and
serialised widely) — there is no save/load yet, and a transient flag
on `Simulation` avoids rippling through every `CraftState`
constructor.

#### 3.4 Detection: pre-contact surface-relative speed

The game layer is the only place that sees contacts, so detection
lives there (`detect_terrain_impact` in `local_physics.rs`). Method:

- Each frame compute the craft's **surface-relative** speed in
  body-centered inertial: `v_rel = lin_vel − ω × r`, where `ω` is the
  body's angular velocity and `r` the craft's position. The terrain
  collider is patch-centered, but its `LinearVelocity = ω × patch_origin`
  and `AngularVelocity = ω` produce that same rotating-surface velocity
  field at contact points. Keep a short peak window (~6 frames).
- On the **rising edge** of contact between the craft body and the
  terrain patch (via `ContactPair`/`ContactGraph`,
  [contact_types/mod.rs](https://docs.rs/avian3d)), read the *peak
  windowed* speed and compare to `impact_tolerance_m_s`. Above
  tolerance → `mark_destroyed`.

Why pre-contact speed rather than the contact impulse:
`SweptCcd` arrests the body at the time-of-impact *position*, so the
velocity drop (and thus `total_normal_impulse`) can land a frame after
the contact-start flag. The windowed pre-impact speed is immune to
which frame the solver books the arrest on, and maps directly to the
m/s tolerance the player reasons about. (Known approximation:
speculative pre-damping can shave the last frame; the peak window and a
slightly conservative tolerance absorb it. See §6.)

#### 3.5 Consequence: force-pause + locked control + scenario picker

On destruction (per the chosen first-slice behaviour — lock + mark, no
explosion FX yet):

- **Control is locked.** `apply_local_forces`
  ([local_physics.rs:1000](crates/game/src/local_physics.rs:1000))
  applies gravity only (no thrust, zero reaction-wheel torque) and the
  input systems (`handle_attitude_controls`, the throttle gate)
  short-circuit, so SAS, autopilot, and the throttle readout all go
  quiet. This is the canonical gate on `is_destroyed()`; in practice the
  force-pause below freezes the whole `SimStage` anyway.
- **The signal is a forced modal.** On destruction the game
  **force-pauses**: `ScenarioMenu::open` folds into `SimClock`'s explicit
  pause predicate and `pause_menu::not_game_paused`, zeroing canonical/local
  sim delta and gating the `SimStage` system sets exactly like the escape
  menu / warp / freecam. Bevy's global `Time<Virtual>` keeps running so
  presentation effects can continue. A centered overlay (`scenario_menu.rs`)
  shows "VESSEL DESTROYED — impact NN m/s" above the four start scenarios,
  plus a log line at destruction time. `destroyed` is mirrored into
  `CraftStateMirror` so it is BRP-queryable.
- **Recovery is an in-place respawn.** Each scenario button repairs the
  craft (`Simulation::repair()`) and rebuilds it for the chosen start
  without relaunching the process: the authored Thalos parking orbit, a
  landing / final-approach descent over daylight dry land, or a Ship→EVA
  disembark. The three ship scenarios reuse `spawn::orbit_parking_state`
  / `spawn::compute_descent_state` (shared with the startup spawn so the
  two never drift); EVA swaps the vessel kind and lets
  `spawn_player_avian_body` plant the on-foot capsule next frame. The
  wreck's Avian bubble is torn down so a clean body respawns. The
  existing debug teleports (F9 surface-drop, body-tree orbit) still call
  `repair()` and close the picker automatically — `open` mirrors
  `is_destroyed()`.

While the picker is up the wreck is held frozen by the force-pause (no
debris settling — that is deferred to the explosion/debris FX pass);
picking a scenario replaces it with a fresh craft. Re-boarding a ship
from EVA via the picker is intentionally not wired, but EVA can't be
destroyed, so the picker only ever opens on a wrecked ship.

#### 3.6 Collider geometry: built from the rendered GPU tiles

The first slice's collider was a tangent-grid trimesh resampled at a
fixed ~64 m spacing over an 8 km patch
([rendered_height.rs](crates/terrain_render/src/rendered_height.rs)).
Even though it reads the *same* height source EVA does, the 64 m facets
cut across the sub-meter rendered relief — tenting *above* dips
("invisible mountains" you crash into) and slicing *under* peaks. EVA
never shows this because it samples the height source per-frame at the
capsule's exact direction and clamps to it, with no interpolation
([player_controller.rs:271](crates/game/src/player_controller.rs:271)).

The fix builds the collider **from the resident GPU atlas tiles the
renderer meshes from**, so it matches the drawn surface by
construction. `HeightSource::build_collider_patch` — overridden by
`GpuAtlasMirrorHeightSource`
([height_source.rs](crates/terrain_render/src/height_source.rs)) — finds
the finest resident tile under the ship and emits one collider vertex
per height texel at the tile's native resolution, each placed at the
exact cube-sphere position the renderer uses:
`Coordinate::world_position(TileCoordinate::pixel_coordinate(texel), height)`.
The game's terrain model is body-centered (`TerrainModel::sphere(ZERO, …)`),
so that position is body-fixed directly. A square window of up to
`patch_resolution` texels is centered on the ship and clamped to the
tile's logical (border-excluded) region, so every vertex is a real texel
of one tile — no cross-tile stitching.

`spawn_terrain_collider_patch` prefers this path and falls back to the
tangent-grid resample when it returns `None`: sources with no tile
geometry (CPU pipeline, flat, baked cubemap), and the GPU mirror before
a tile is resident (e.g. just after the 20 km attach, before streaming).
As finer tiles stream in during descent the mirror's `revision()` bumps,
`maintain_terrain_patch` rebuilds the collider re-centered on the ship at
the new (finer) resolution, and by touchdown the collider sits at the
rendered LOD's native resolution under the craft.

Because that window is small (tens of metres), the global
`patch_rebuild_distance_m` is too coarse to keep the craft on it during
surface travel. The patch reports its metric extent
(`TerrainPatchMesh::half_extent_m` → `LocalBubble::patch_half_extent_m`)
and `maintain_terrain_patch` also rebuilds when the craft drifts past
~45% of that half-extent, re-centering the window — so it follows lateral
movement (a rover, a sliding touchdown), not just vertical descent. The
coarse tangent-grid fallback keeps its km-scale global threshold.

The terrain collider body is patch-centered, not planet-centered. The
mesh stores `(vertex_body_fixed - patch_center_body_fixed)` offsets, the
rigid body `Position` is `body_orientation * patch_center_body_fixed`,
and its `LinearVelocity` is `ω × Position`. Together with
`AngularVelocity = ω`, this gives contact points the same rotating-surface
velocity as the old body-centered formulation while keeping the trimesh's
local coordinates near zero for stable support contacts.

### 4. First implemented slice

What this pass ships (scoped by the three product decisions: whole-craft
destruction; lock+mark with no explosion FX; land on hull, no landing
legs):

1. `SweptCcd` + `Restitution(0)` on the dynamic ship body
   (`physics_local`).
2. `ShipParameters::impact_tolerance_m_s` (`physics_canonical`).
3. `Simulation` destroyed state + accessors (`physics_canonical`).
4. `detect_terrain_impact` (`game`).
5. Control lockout in `apply_local_forces` + input/throttle gates;
   `repair()` on respawn teleports.
6. `scenario_menu.rs`: on destruction, force-pause + an in-place
   scenario-respawn picker for the four start scenarios (this replaced
   the original passive `hud/destroyed_banner.rs`); `Simulation::repair()`
   on respawn; `CraftStateMirror.destroyed`.
7. Collider built from the resident GPU tiles for by-construction
   alignment with the rendered surface
   (`HeightSource::build_collider_patch`), with tangent-grid fallback,
   and a window-relative rebuild so the small tile window follows the
   craft across the surface (§3.6). Attachment and stale-patch rebuilds
   are gated by the surface warp-lock zone: effective warp must be 1x and
   the altitude gate must cap the ladder at 1x, so a manual reset to 1x
   higher in the descent does not allocate, keep, or refresh collider
   geometry unless the existing patch is still needed to finish a contact
   collapse.
8. F8 craft-collider debug view, drawn in the ship camera from the same
   compound collider primitives used by the local rigid body. It uses a
   dedicated ship-layer gizmo group because the default gizmos are
   intentionally map-only.

Explicitly **not** in this slice: per-part crash tolerance and
fragmentation; landing legs; explosion/debris VFX; an automatic
surface↔orbit transition; the `SimEvent` pipeline (destruction is a
direct canonical state change for now, not an emitted event).

### 5. Tuning knobs

- `ShipParameters::impact_tolerance_m_s` — crash speed threshold.
  Start forgiving (~12 m/s) and tighten.
- Descent spawn profiles live in
  [`crates/game/src/spawn.rs`](crates/game/src/spawn.rs). `just game
  landing` starts ~25 km AGL over daylight dry land so the player sees
  the on-rails-to-local-physics handoff. `just game final` (aliases
  `final-approach`, `final_approach`, `approach`) starts ~1.5 km AGL,
  low and slow, after scoring daylight dry sites by local height relief
  to find a flat touchdown-practice patch.
- `SweptCcd` mode (`NonLinear` vs `Linear`) and per-body
  `SpeculativeMargin` — if high-speed approaches produce "ghost"
  spinning from the unbounded default speculative margin, cap it on the
  ship body. Left at default in the first slice; first lever to reach
  for if ghost collisions appear.
- `LocalBubbleConfig::patch_resolution` — for the tile-based collider
  (§3.6) this caps the texel-window side (vertices = window²), so it
  trades contact-area coverage against rebuild cost at native
  resolution. For the tangent-grid fallback it is the patch grid
  resolution. `patch_half_extent_m` sizes the fallback patch only.
- F8 toggles the ship-view craft-collider debug view. It draws the
  compound collider primitives over the rendered ship using a high-contrast
  outline, which is the first place to look when a part footprint or
  collider orientation seems wrong.

### 6. Open questions

- **Speculative pre-damping vs. measured impact speed.** Does the
  windowed pre-contact speed match perceived impact speed across the
  range of approaches, or does the unbounded speculative margin shave
  it enough to mistune the threshold? Needs `just game` observation;
  if material, switch detection to peak single-frame
  `max_normal_impulse_magnitude / mass` over the first few contact
  frames, or cap the ship's `SpeculativeMargin`.
- **Per-part crash tolerance.** The natural next step: each part in the
  staging graph (`crates/game/src/staging.rs`,
  `thalos_shipyard`) carries a tolerance; a hard contact destroys the
  parts whose contact exceeds theirs and fragments the craft into
  controllable/debris subtrees (the same graph cut staging already
  performs). Whole-craft tolerance becomes `min` over contacting parts.
- **Landing legs.** A dedicated part with a wide footprint and impact
  absorption (higher tolerance + a suspension constraint) so tall
  stacks land upright without tipping. Aircraft landing *gear* (wheels)
  is now implemented as raycast suspension (§7); rocket landing *legs*
  (no roll, wide static footprint) can reuse the same per-contact force
  channel with rolling/steer disabled.
- **Landing on water / oceans.** Ocean bodies use a flat-water
  placeholder; splashdown vs. terrain impact is unaddressed.
- **`SimEvent` pipeline.** When the event model
  (`simulation.md` §"Event model") is built, destruction should emit
  `SimEvent::Impact` instead of being a bare state mutation, so map
  warnings / mission logic can subscribe.

## Ground physics frame: ships run body-fixed

**The ship local-physics bubble runs in the body-*fixed* (rotating)
frame, not body-centered inertial.** This is the load-bearing change that
makes ground contact (and wheels) stable. In the body-centered *inertial*
frame a craft "parked" on the surface is actually translating at the
surface co-rotation speed (`ω×r` ≈ **256 m/s** on Thalos), and the
terrain/runway collider is flung along with it and re-posed every frame —
doing dynamic rigid-body contact between two bodies moving at 256 m/s
against a thin trimesh jitters, spins the craft up, and tunnels it through
the floor. In the body-fixed frame the craft and the ground are both
~stationary (0–taxi speed), so the contact solver is stable. (Grounded EVA
already worked this way; ships now match.)

Implementation (`crates/game/src/local_physics.rs`):

- **Conversion seam branches on `VesselKind`.** `inertial_to_ship_frame` /
  `ship_frame_to_inertial` (wrapping `thalos_physics_canonical::body_fixed`)
  convert canonical inertial ↔ body-fixed for **ships**; EVA keeps the
  body-centered `inertial_to_bubble_frame`. `inertial_to_craft_frame` /
  `craft_frame_to_inertial` dispatch by kind at the snap/readback/spawn/
  rebase seams.
- **Forces** in `apply_local_forces` are body-fixed: gravity `−μr/r³`
  (radial form is rotation-invariant) **plus** centrifugal `−Ω×(Ω×r)` and
  Coriolis `−2Ω×v`, with `Ω = orientation⁻¹ · body.angular_velocity`
  (tiny on Thalos but keeps burns correct). Thrust is `rotation·ŷ` in the
  body-fixed frame.
- **Ground colliders are static** in this frame: `sync_terrain_collider_pose`
  and `runway::sync_runway_collider_pose` set `Position =
  center_surface_body_m`, identity rotation, zero velocity. No per-frame
  re-pose, no co-rotation velocity.
- Rendering is unaffected — it reads canonical (inertial) state, and
  `readback_local_craft` converts the ship's body-fixed Avian state back to
  inertial each frame.

## Surface friction (hull contact)

A craft *without* landing gear — a lander or rocket resting on its belly —
gets its tangential ground friction from `apply_surface_friction`
([`crates/game/src/local_physics.rs`](crates/game/src/local_physics.rs)),
which runs right **after** `terrain_floor_backstop` and just **before**
`readback_local_craft` in the local-physics chain. The backstop only ever
removes the *into-surface* (radial) velocity component; before this system
existed a landed gearless craft kept its full tangential velocity and slid
indefinitely, because nothing opposed surface-parallel motion.

- **Coulomb stick/slip, velocity-level.** Like the backstop, it edits
  `LinearVelocity` directly rather than pushing a force into the
  acceleration accumulator. A velocity-level stick/slip cancels exactly
  within the per-frame friction budget, so the craft reaches a *true* stop
  regardless of step size — an `∝ v` force law only decays asymptotically
  and creeps forever (the bug this fixes).
- **Static / kinetic.** If the per-frame tangential slip is below
  `mu_static · g · dt` the craft sticks (its surface-parallel velocity is
  zeroed); otherwise kinetic friction decelerates it at `mu_kinetic · g`
  along the slip direction. Gravity in the body-fixed frame is central
  (purely radial), so it contributes no tangential term: friction only has
  to remove residual slip, and the normal load per unit mass is just
  `g = μ/r²` (mass cancels in the velocity-level form).
- **Contact test.** Reuses `deepest_hull_radial` (shared with the
  backstop): the craft is "on its hull" when the deepest collider-primitive
  support point sits within `contact_margin_m` of the sampled terrain
  surface. Airborne craft fall outside the band and are untouched.
- **Wheeled craft are skipped.** When any wheel bears load
  (`WeightOnWheels.grounded`) the landing-gear model owns the tangential
  reaction and the suspension holds the hull clear, so hull friction stands
  down — the two models never double-count.
- **Gating.** Same as the backstop: only when Avian owns translation
  (`AvianRole::Full`) for a live, non-destroyed `Ship`.
- **Tuning.** `SurfaceFriction` (`mu_static`, `mu_kinetic`,
  `contact_margin_m`) is a Reflect-registered resource — live-tune over BRP.

## Functional landing gear (wheels)

Landing gear are real wheels you can roll and taxi on, not cosmetic
struts. The model is **raycast suspension** on the single craft rigid
body — no physical wheel colliders or joints (those are unstable at
planet scale). It lives in
[`crates/game/src/local_physics.rs`](crates/game/src/local_physics.rs)
as `apply_landing_gear_forces`, running in the local-physics chain right
**after** `apply_local_forces`, so wheel forces add on top of the gravity
+ thrust + reaction-wheel torque already written into the craft's
acceleration accumulators.

- **Where the wheels are.** At craft spawn, `build_wheel_set` walks the
  gear parts and, via `thalos_shipyard::gear_leg_frames` (the *same*
  per-leg geometry `build_gear_mesh` draws), caches a `WheelSet` of
  craft-local strut-top points and suspension/roll/axle axes. Collider
  wheels therefore sit exactly under the rendered wheels, for any craft,
  with no hand-placed constants.
- **Rotate about the real CoM.** A geared craft also gets Avian
  `CenterOfMass` set to the shipyard-computed CoM (`ShipParameters::
  center_of_mass`) plus `NoAutoCenterOfMass`. Without it the body rotates
  about the nose-pod origin, ahead of every wheel, and the upward wheel
  forces have no balancing torque — the craft tips onto its nose. Wheel
  torque arms are taken relative to this CoM.
- **Per wheel, per frame.** A ray is cast from the strut top down the
  suspension axis (`Rotation·r̂`, belly-ward) against every collider
  *except the craft* — which transparently finds the runway slab or the
  terrain patch, whichever is closest, and reports nothing when the wheel
  is airborne. From the hit: a one-way **spring + damper** along the strut
  (using the contact-*point* velocity; the ground is static in the
  body-fixed frame so slip is just that velocity — **no `ω×r` term**),
  **lateral grip** resisting sideways slip, and a **longitudinal** force
  (rolling resistance + brake) — both clamped to a friction circle so
  wheels only ever remove ground-relative speed, never propel. Forward
  motion comes from engine thrust (or, later, powered wheels). The free
  rolling resistance is **Coulomb, not viscous**: a stiff fore/aft hold
  clamped to a small `rolling_mu·N` cap, so the constant opposing force
  brings a coasting craft to a true stop in finite time and then holds it,
  instead of the old `∝ v` law that decayed asymptotically and let the
  craft creep forever.
- **Ride height (no clip).** The damper is sized from the craft's real
  per-wheel mass for a near-critical settle, and `k_spring` is stiff enough
  that the natural static sag `m·g/(n·k)` is ~cm-scale, so the rigid wheel
  meshes don't visibly clip the ground. (A uniform spring *preload* to
  cancel the sag was tried and reverted — it unbalances the per-wheel
  torque and tips the craft; the suspension must find its own
  load-balanced equilibrium.)
- **Steering.** Nosewheel steering reuses the yaw axis (A/D, KSP-style —
  no separate binding); it rotates the single-leg gear's roll/axle
  directions about the strut, and the resulting off-CoM lateral grip yaws
  the craft (emergent, friction-limited).
- **Parking brake.** A latched toggle (`B`, `flight.parking_brake`),
  **engaged at startup** (`ParkingBrake::default` → `engaged: true`) so a
  spawned aircraft holds on the runway instead of creeping. When engaged,
  the longitudinal channel switches from rolling resistance to a high-gain
  fore/aft hold (`parking_brake_stiffness`, clamped to the friction
  circle), pinning the craft against gravity, slopes, and the settle
  residual — though full takeoff thrust still overpowers it. Tap `B` to
  release and taxi. `toggle_parking_brake` flips the `ParkingBrake`
  resource on the key edge.
- **Wings are colliders too.** `build_ship_collider_primitives` gives each
  `Wing` a thin oriented-cuboid collider matching its planform (via
  `wing_panel_frame`), so a wingtip catches the ground on an over-banked
  landing instead of passing through.
- **Gating.** Only when Avian owns translation (`AvianRole::Full`) for a
  live, non-destroyed `Ship`. On the runway scenario the craft is already
  `Full` (a terrain patch attaches at AGL≈0), so taxiing works straight
  from the parked state. At warp≠1× the role is `Paused` and the system is
  inert, so no warp-scaled `dt` ever reaches the spring.
- **Belly colliders stay.** The compound fuselage/engine cylinder
  colliders are kept as the `SweptCcd`/impact backstop; with the craft
  spawned at gear-bottom clearance the belly rides above the surface and
  only engages on a gear-collapse/abnormal landing.
- **Tuning.** `GearTuning` (spring stiffness, damping ratio, friction
  `mu`, lateral stiffness, rolling `rolling_mu` + `rolling_hold_stiffness`,
  parking-brake stiffness, max steer, travel, ray margin) is a
  Reflect-registered resource — live-tune over BRP while taxiing rather
  than recompiling.

**Driving requires thrust, which requires staged + fuelled engines.** The
demo aircraft's engines only produce thrust once *activated* (staging —
Space); an unstaged craft sits still on its wheels with `thrust_n = 0`
even at full throttle. (Thalos has an 80 km atmosphere, so its
air-breathing jets work once staged; `DebugMode::jets_in_vacuum` can force
air-breathers to fire on genuinely airless bodies.)

Not yet covered: powered wheels (drive torque from a throttle — the
eventual goal so airless-body rovers can drive without thrust) and gear
retraction. The parking brake's hold is a high-gain damper rather than a
true static-friction latch, so under a sustained force above the friction
circle (e.g. full thrust) it slips rather than holding.
