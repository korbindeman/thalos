# Landing & impact destruction

Spec for landing a **ship** on a planetary surface and for destroying a
craft that hits the surface too hard. This is the "landed-ship
mechanics" pass that [`surface_gameplay.md`](surface_gameplay.md)
explicitly defers (its scope is on-foot/EVA only; §7 lists
"Ship-on-surface physics" as needing its own spec). On-foot gameplay,
the `HeightSource` interface, and the surface map view live in that
document; this one covers ship descent, contact, and structural
failure.

This is a **diagnosis + plan** document in the same shape as
`surface_gameplay.md`. §1–2 describe the code as it stands (with
file:line citations); §3–5 lay out the target and the first
implemented slice; §6 tracks open questions.

## 1. Goals

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

## 2. Current state

### 2.1 The local bubble already has collision geometry

`thalos_physics_local` spins up one Avian rigid body per craft and,
within 20 km AGL, attaches a `Kinematic` trimesh terrain patch:

- The ship is a `RigidBody::Dynamic` compound collider built from the
  rendered parts (cones/cylinders) in
  `build_ship_collider_primitives`
  ([local_physics.rs:1688](crates/game/src/local_physics.rs:1688)),
  spawned by `spawn_local_craft_body`
  ([lib.rs:284](crates/physics_local/src/lib.rs:284)).
- `attach_terrain_patch_when_close`
  ([local_physics.rs:592](crates/game/src/local_physics.rs:592)) spawns
  a `RigidBody::Kinematic` trimesh patch (`spawn_terrain_collider_patch`,
  [lib.rs:236](crates/physics_local/src/lib.rs:236)) when AGL drops
  below `handoff_agl_m` (20 km). Its `Rotation`/`AngularVelocity` track
  the body each frame so its body-fixed vertices land in the right
  body-centered-inertial positions.
- Avian's contact solver runs only when Avian *owns translation*, i.e.
  `AvianRole::Full` — throttle active **or** terrain patch attached
  (`avian_role_from_inputs`,
  [local_physics.rs:234](crates/game/src/local_physics.rs:234)). The
  patch's mere existence inside the AGL band is the "contact is
  physically possible here" signal.

So the contact pair exists and is solved on approach. The gap is
**not** missing colliders.

### 2.2 Clip-through is tunneling, not a missing collider

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

- The patch is coarse: `patch_resolution = 129`,
  `patch_half_extent_m = 4096`
  ([lib.rs:47](crates/physics_local/src/lib.rs:47)) → ~64 m vertex
  spacing. Sub-64 m relief isn't in the collider.
- A known collider-vs-rendered-surface gap exists; `debug_log_terrain_gap`
  ([local_physics.rs:1468](crates/game/src/local_physics.rs:1468))
  tracks it.

### 2.3 There is no structural-integrity model

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

### 2.4 Gentle landing already works

`collapse_or_constrain_warp`
([local_physics.rs:1315](crates/game/src/local_physics.rs:1315)) tracks
stable contact and collapses to `AuthorityMode::BodyFixed` after 2.0 s
of continuous contact under 0.5 m/s linear / 0.05 rad/s angular with
throttle zero (`stable_contact_reached`,
[lib.rs:336](crates/physics_local/src/lib.rs:336)). So the *landed*
end-state is real; the missing pieces are surviving the descent
(§2.2) and a consequence for not surviving it (§2.3).

### 2.5 Rendered pose is extrapolated across the physics overstep

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

## 3. Target architecture

### 3.1 Anti-tunneling: SweptCcd on dynamic craft

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

### 3.2 Impact tolerance is a ship parameter

Add `impact_tolerance_m_s: f64` to `ShipParameters` — a KSP-style crash
tolerance in m/s of surface-relative approach speed. It is pure
physical data, consistent with the other fields there (thrust, MOI,
dry mass), and is the seam the future per-part model refines (the
craft tolerance becomes `min` over contacting parts).

First-slice value: a single forgiving constant pushed with the rest of
the ship stats. EVA is exempt (`f64::INFINITY`) — on-foot contact
damage is out of scope here and EVA does not use Avian contact
resolution anyway (`surface_gameplay.md` §2.1).

### 3.3 Whole-craft destruction is canonical state

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

### 3.4 Detection: pre-contact surface-relative speed

The game layer is the only place that sees contacts, so detection
lives there (`detect_terrain_impact` in `local_physics.rs`). Method:

- Each frame compute the craft's **surface-relative** speed in
  body-centered inertial: `v_rel = lin_vel − ω × r`, where `ω` is the
  body's angular velocity and `r` the craft's position (the terrain
  collider sits at the body centre and co-rotates, so this is the
  speed the contact actually sees). Keep a short peak window (~6
  frames).
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

### 3.5 Consequence: inert debris + locked control + clear signal

On destruction (per the chosen first-slice behaviour — lock + mark, no
explosion FX yet):

- **Control is locked.** `apply_local_forces`
  ([local_physics.rs:1000](crates/game/src/local_physics.rs:1000))
  applies gravity only (no thrust, zero reaction-wheel torque), so the
  craft becomes an inert rigid body that keeps falling and colliding —
  reading naturally as debris that tumbles and settles. Input systems
  (`handle_attitude_controls`, the throttle gate) short-circuit so SAS,
  autopilot, and the throttle readout all go quiet.
- **The signal is a HUD banner.** A prominent "⚠ VESSEL DESTROYED —
  impact NN m/s" overlay (`hud/destroyed_banner.rs`) is the explicit
  cue, plus a log line at destruction time. `destroyed` is mirrored
  into `CraftStateMirror` so it is BRP-queryable.
- **Recovery is the existing teleports.** F9 debug surface-drop and the
  body-tree orbit teleport call `Simulation::repair()`, clearing the
  flag and handing a fresh craft back. (No automatic respawn.)

A destroyed craft is still allowed to settle to `BodyFixed` via the
existing stable-contact collapse — debris at rest is fine, and control
stays locked regardless of authority.

### 3.6 Collider geometry: built from the rendered GPU tiles

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

## 4. First implemented slice

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
6. `hud/destroyed_banner.rs` + `CraftStateMirror.destroyed`.
7. Collider built from the resident GPU tiles for by-construction
   alignment with the rendered surface
   (`HeightSource::build_collider_patch`), with tangent-grid fallback,
   and a window-relative rebuild so the small tile window follows the
   craft across the surface (§3.6).

Explicitly **not** in this slice: per-part crash tolerance and
fragmentation; landing legs; explosion/debris VFX; an automatic
surface↔orbit transition; the `SimEvent` pipeline (destruction is a
direct canonical state change for now, not an emitted event).

## 5. Tuning knobs

- `ShipParameters::impact_tolerance_m_s` — crash speed threshold.
  Start forgiving (~12 m/s) and tighten.
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

## 6. Open questions

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
  stacks land upright without tipping. Adds a `Part` type +
  collider; deferred.
- **Landing on water / oceans.** Ocean bodies use a flat-water
  placeholder; splashdown vs. terrain impact is unaddressed.
- **`SimEvent` pipeline.** When the event model
  (`simulation.md` §"Event model") is built, destruction should emit
  `SimEvent::Impact` instead of being a bare state mutation, so map
  warnings / mission logic can subscribe.
