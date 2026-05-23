# Surface gameplay

Spec for on-foot (EVA / walking / rover) gameplay on a planetary
surface: ground physics, what "landed" means for the canonical
simulation, how heightfield data flows from terrain to physics, and
how the map / trajectory view treats a surface-bound craft.

This is a **diagnosis + plan** document. Sections 2–3 describe the
state of the code today (with file:line citations); sections 4–6 lay
out the target architecture and the migration order. Open questions
are tracked in §7 and must be resolved before the corresponding work
items move from "research" to "implement."

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
The `should_log_tile()` diagnostic
([pipeline.rs:144](crates/terrain_render/src/pipeline.rs:144))
logs the LOD level and `tile_lod_m` of the first ~32 tiles produced
per session — useful for confirming what LOD UDLOD is actually
selecting at the camera.

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
Memory budget for height-only mirroring: R16 × 512² × 256 slots ≈
128 MB worst case; halved by mirroring only the focused-body / nearest-LODs
slice that physics actually queries. Tile-residency lag (~1 frame
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
  where the trajectory-suppression rule applies. The landed-ship
  redesign this doc defers now has its own spec —
  [landing.md](landing.md) — which covers ship descent, terrain
  collision (`SweptCcd` anti-tunneling), and impact destruction; rover
  wheels / sliding / surface-warp remain open there.

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
- `Time` resource source (Virtual vs Real) under pause/warp.

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
  plus a baked fallback. The mirror stores resident R16 height tiles
  keyed by `TileCoordinate`, samples them bilinearly in body-fixed
  direction space, and falls back when no resident ancestor exists.
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
