# Craft regimes — the per-craft regime resolver

**Status: design agreed 2026-06-11; Phase A2 (shadow mode)
implemented and acceptance met 2026-06-12.** The pure classifier
lives in `thalos_physics_canonical::regime` (unit-tested), the
game-side resolver + drift checker in `crates/game/src/regime.rs`
(`RegimePlugin`); the legacy machinery still drives everything and no
consumer reads the record yet.

A2 acceptance results (~51 k drift-checked frames over BRP, reading
`thalos_game::regime::RegimeDriftDiagnostics`): **zero steady-state
mismatches in every check** (role, authority, warp cap, prediction);
three single-frame blips total, all on the documented §3.2 edges
(pause→1× snap; the landed throttle release). Scenario coverage:
**orbit** (coast, warp ladder up/down, pause cycle), **eva** (stand /
walk / re-rest), **runway** (parked `BodyFixed`, surface warp, landed
release, full-throttle roll, liftoff), **final** (atmospheric `Full`
descent, gearless hull touchdown, the timed settle collapse →
`BodyFixed`), **runway-approach** (rails→bubble handoff, sustained
atmospheric flight — which also covers `cruise`'s classification
surface; `landing` is classification-equivalent to `final`).
**A3 is unblocked**: port consumers one at a time, starting with
`manage_authority` → record executor.

This doc is the spec for the simulation-unification pass: Phase A
(regime resolver), Phase B (vocabulary + module cleanup), and the
Phase C sketch (unified force model). It supersedes the
*target-design* role of the "Avian's three roles" /
authority-handoff narrative in `docs/simulation.md` — that doc still
accurately describes today's implementation; this one describes where
it goes.

Related: `docs/simulation.md` (canonical core, authority, warp),
`docs/surface_local.md` (the SLF ships integrate in),
`docs/control.md` (the fly-by-wire pattern this generalizes),
`docs/aerodynamics.md` (the aero model Phase C unifies into
prediction).

## 1. Motivation — what falls short today

The canonical core is healthy: one `CraftState`, one `AuthorityMode`
with a transition log, one `ShipPropagator` shared by live stepping
and prediction, and a `Simulation::step` that only advances time,
evaluates the `BodyFixed` pose, and Kepler-coasts the rails. The
problems are all in how the *game side* decides, each frame, who owns
what:

- **Two authority vocabularies, reconciled by hand.** Canonical
  `AuthorityMode` (5 variants, 2 never constructed) and game-side
  `AvianRole` (`Paused`/`AttitudeOnly`/`Full`) are parallel
  classifications kept aligned by `manage_authority`
  (`crates/game/src/local_physics.rs`), which carries two
  special-case pins jumped ahead of its generic match (grounded EVA
  pinned to `LocalRigidBody`; landed ship + warp request collapsed to
  `BodyFixed`). `docs/surface_local.md` §5 predicted this machinery
  would collapse when the SLF landed; it survived intact.
- **The same predicates re-derived in N places.** "Is the ship
  sitting quietly on the ground" (contact + speed below
  `max_stable` + throttle ≤ 1e-3) is computed independently three
  times — `manage_authority`'s landed-warp collapse,
  `enforce_warp_altitude_limits`' `ship_grounded_stationary`
  (`crates/game/src/bridge.rs`), and `collapse_or_constrain_warp`'s
  `stable_contact_reached` — with drift between them (one reads
  `warp.target_speed()`, another `warp.speed()`). "Is the craft
  ballistic" is re-derived again in `bridge::ship_is_ballistic` to
  gate prediction. Warp-is-1× epsilon checks appear in at least four
  systems.
- **Warp policy is scattered across ~7 enforcement points**: the
  altitude ladder (canonical `WarpController`), the atmosphere 1×
  clamp (bridge), the EVA at-rest rule (bridge), the landed collapse
  (`manage_authority`), the collision warp-reset (`Simulation::step`),
  the sub-surface warp-reset (`advance_simulation`), and the
  terrain-collider warp gate. No single place can answer "what warp
  is allowed right now, and why" — which also means the HUD cannot
  display the reason.
- **EVA is interleaved, not parallel**: ~9 `VesselKind::Eva`
  early-returns threaded through ship systems (`manage_authority`,
  `snap_avian_from_canonical`, `readback_local_craft`,
  `apply_local_forces`, `attach_terrain_patch_when_close`,
  `reanchor_surface_frame`, `update_prediction`,
  `enforce_warp_altitude_limits`, `detect_terrain_impact`).
- **Dead spec vocabulary**: `AuthorityMode::WarpIntegrated` and
  `::Docked` are never constructed outside tests; canonical
  `SimClock`/`TimeMode` (`crates/physics_canonical/src/canonical.rs`)
  are entirely unused and the former name-clashes with the real
  game-side `sim_clock::SimClock`.

## 2. The vehicle thesis

Thalos is an aerospace-engineering game in general: rockets, planes,
ground vehicles, EVA with RCS in vacuum (KSP-style jetpack), and
eventually watercraft and submersibles. That ambition fixes the
shape of the regime model:

**One dynamic-rigid-body path for every craft, with exactly one
kinematic exception: walking.**

- Planes, rovers, boats, subs, jetpack-EVA are all "one aggregate
  body + environment forces + raycast suspension + occasional
  ground/water contact". None of them needs a constraint solver:
  Thalos is aggregate-rigid-body by design, docking merges vessels
  KSP-style, decoupling splits into independent bodies, debris is
  independent bodies.
- *Walking* stays a kinematic character controller, per the
  `docs/surface_local.md` §10 rationale (it never touches the contact
  solver; folding it into the SLF is cosmetic and risky). But it is
  reframed from "EVA is a separate vessel universe" to "walking is
  one **locomotion mode** that takes translation ownership while
  active". Jetpack/vacuum EVA becomes a *normal craft* — a capsule
  rigid body with an RCS effector through the fly-by-wire allocator
  (`docs/control.md` already lists RCS and the EVA jetpack as
  designed-in extension points).
- Consequently the regime classifier must not branch on vessel
  *kind*. It classifies from **craft capabilities** (wheels/gear,
  control surfaces, RCS, thrust, buoyancy — mostly already declared
  in the parts catalog) **× environment** (medium + ground state).
  Most `VesselKind::Eva` branches become a single check:
  `translation_owner == Kinematic`.

## 3. The `CraftRegime` record

One per-frame decision record, resolved once by a sole-writer system,
consumed by everything downstream. Per-craft from day one: the record
is a **component on the craft root entity**, not a singleton resource
(exactly one craft exists today; the shape must not assume that).

Sketch (field names and exact enum shapes finalize at
implementation):

```rust
// thalos_physics_canonical::regime — pure, Bevy-free, unit-tested.

pub enum Medium {
    Vacuum,
    Atmosphere,        // below the body's Kármán line
    WaterSurface,      // reserved — boats (unimplemented)
    Submerged,         // reserved — submersibles (unimplemented)
}

pub enum GroundState {
    Airborne,          // no ground interaction
    Contact,           // touching (WoW or hull contact), not settled
    Settled,           // stable-contact criteria held for the dwell time
}

pub enum TranslationOwner {
    Canonical,         // rails coast or analytic BodyFixed pose
    Backend,           // the local rigid-body backend (Avian today)
    Kinematic,         // a locomotion controller (walking)
}

pub enum RotationOwner {
    Canonical,         // frozen/analytic (warp, BodyFixed)
    Backend,           // integrated under fly-by-wire torque
    Kinematic,
}

pub struct WarpPolicy {
    pub max_level: usize,            // index into the warp ladder
    pub constraint: WarpConstraint,  // why — for HUD + diagnostics
}

pub enum WarpConstraint {
    Unconstrained,
    AltitudeLadder,        // per-level min-altitude floor
    InAtmosphere,          // aero forces only run live at 1×
    MovingOnSurface,       // surface warp needs at-rest
    NotAtRestOnFoot,       // walking/jumping/falling clamps to 1×
}

pub enum PredictionDisplay {
    Show,
    Hide(HideReason),      // Landed | GroundContact | OnFoot
}

pub struct CraftRegime {
    pub medium: Medium,
    pub ground: GroundState,
    pub translation_owner: TranslationOwner,
    pub rotation_owner: RotationOwner,
    /// Should the backend's integrator clock step this frame?
    /// (Rotation-only integration is `rotation_owner == Backend`
    /// with `translation_owner == Canonical` — today's AttitudeOnly.)
    pub backend_clock_runs: bool,
    pub warp: WarpPolicy,
    pub prediction: PredictionDisplay,
    /// Warp/capability gate for the ground-collider systems; the
    /// attach/detach systems keep their AGL geometry + hysteresis.
    pub terrain_collider_allowed: bool,
}
```

Mapping today's regimes onto the record:

| Today | translation | rotation | clock |
|---|---|---|---|
| OnRails coast, 1×, vacuum (`AttitudeOnly`) | Canonical | Backend | on |
| Warp > 1× / paused (`Paused`) | Canonical | Canonical | off |
| Thrust/contact/atmosphere at 1× (`Full`) | Backend | Backend | on |
| Landed (`BodyFixed`) | Canonical | Canonical | off |
| Walking (grounded EVA) | Kinematic | Kinematic | off* |

\* today the EVA capsule keeps Avian unpaused while the controller
writes `Position` directly; the resolver makes "clock off, controller
owns" explicit instead.

### 3.1 Resolver structure

- **Pure core** in `thalos_physics_canonical::regime`:
  `fn resolve(inputs: &RegimeInputs, memory: &RegimeMemory)
  -> (CraftRegime, RegimeMemory)`. `RegimeMemory` holds the small
  stateful pieces the classification needs across frames — today's
  settle timer (`stable_contact_s`, absorbed from
  `collapse_or_constrain_warp` / `LocalBubble`) and the previous
  decision for hysteresis/edge detection. Unit-tested the way
  `avian_role_from_inputs` is today.
- **Game-side sole writer** gathers `RegimeInputs` from the ECS and
  writes the `CraftRegime` component: warp speed/levels + target,
  canonical authority, effective throttle, terrain-patch presence,
  craft↔ground contact (contact graph), `WeightOnWheels`,
  linear/angular speeds, locomotion mode (today `EvaMode`), craft
  capabilities, dominant-body atmosphere data + altitude, destroyed
  flag.
- **One executor applies canonical transitions**: the successor of
  `manage_authority` maps the record onto `AuthorityMode` transitions
  (`transition_authority`), preserving the canonical authority log.
  `AuthorityMode` stays — it is the canonical, persisted authority;
  `CraftRegime` is the frame-local derived decision.

### 3.2 Scheduling and input staleness

The resolver runs **once, at the top of `SimStage::Physics`**, before
the bridge's warp handling and `advance_simulation`. Input snapshot
semantics, made explicit instead of today's implicit mix:

- **Physics-derived inputs are previous-frame**: contact graph,
  weight-on-wheels, terrain-patch presence, body velocities. (Today's
  `enforce_warp_altitude_limits` already reads last-frame contact
  state; today's `compute_avian_authority` reads same-frame patch
  presence because it runs mid-chain — under the resolver this edge
  becomes one frame later.)
- **Command inputs are current-frame**: warp level, throttle.

The one-frame lag at the patch attach/detach edges is harmless by
construction: attach happens at ~20 km AGL (one frame at descent
speeds is metres of altitude), detach has 1.5× hysteresis, and the
handoff-edge snap machinery covers the ownership flip regardless of
which frame it lands on. The drift-check phase (§6, A2) must confirm
mismatches occur *only* on these documented edges, never steady-state.

### 3.3 What the resolver subsumes

`compute_avian_authority` + `AvianRole`/`AvianAuthority`,
`manage_authority`'s pins and generic match,
`bridge::ship_is_ballistic` + the prediction gate's EVA branch, the
three grounded-stationary predicates, the EVA at-rest warp rule, the
atmosphere warp clamp, the altitude-ladder cap computation, and the
terrain-collider warp gate. The two **propagator-level emergency
resets stay** as backstops and are documented as such — the collision
warp-reset in `Simulation::step` and the sub-surface warp-reset in
`advance_simulation` fire on states the resolver cannot see coming
(mid-coast terrain crossing, corrupted state), not on policy.

### 3.4 Multi-craft scope

The record, resolver, and executors are written against "a craft"
(component + query). What is deliberately **not** done now: canonical
multi-craft (`Simulation` still owns one `CraftState`) and per-craft
bubbles (`ActiveLocalBubble` stays a singleton; the likely
generalization — a shared per-body frame with per-craft anchors — is
sketched in `docs/surface_local.md` §9). The constraint on new code
is only that nothing *new* bakes in single-craft assumptions.

## 4. Backend seam — the Avian decision

**Decision: Avian stays through Phases A–B, behind a tightened seam;
re-evaluate at the start of Phase C.**

Inventory of what Avian actually does for Thalos today: integrate one
aggregate rigid body from our accumulated accelerations; solve
hull-vs-ground contact for gearless craft (wheeled hulls are filtered
out of ground contact — the raycast gear is the sole interface);
raycasts for gear; `SweptCcd`; contact-graph queries; collision
layers. Gravity, gear suspension, aero, surface friction, the terrain
floor backstop, impact destruction, and attitude control are already
custom. Parry (which Avian wraps) provides all the genuinely hard
collision-detection code — heightfields, compound shapes, manifolds,
ray/shape casts, TOI — and stays either way.

The case for eventually going parry-direct is unusually strong here:
the authority dance (snap/readback, role→clock mapping,
`Time<Physics>` gymnastics) is a permanent impedance mismatch between
two integrators co-owning one state; the vehicle thesis (§2) needs no
constraint solver; an owned integrator gives per-render-frame
variable-dt f64 stepping for free (the fixed-timestep deviation
deferred in `docs/surface_local.md` §10); and a sequential-impulse
resolver for "one body vs static ground, destruction above modest
impact speeds" is the easiest contact problem there is. The case
against doing it *now*: it serializes against Phase A (which shrinks
the Avian-facing surface to a small executor layer — the very thing
that makes a swap cheap), and it re-opens the won-and-verified
gearless-contact stability battles, which deserves a dedicated slice
with scenario re-verification, not a refactor side effect.

**Seam rule (Phase A enforces):** the executor layer — snap/readback,
force application, collider construction, clock control — is the
*only* code allowed to touch Avian types. Everything else reads the
`CraftRegime` record and canonical state. `thalos_physics_local` is
already nominally this boundary; Phase A tightens it from convention
to structure. Naming in new code says `Backend`, not `Avian`.

**Re-evaluation triggers at Phase C** — swap if: the executor layer
still needs role gymnastics after Phase A; per-frame stepping inside
Avian's schedule proves as risky as flagged; Avian f64 lags Bevy
upgrades; or per-craft SLF anchors fight Avian's world model. Keep
if: a roadmap feature genuinely wants a constraint solver (articulated
rovers, jointed cranes/trailers). Decide with a spike: prototype the
parry-direct path for the gearless-lander contact case on a branch
and compare stability and effort directly.

## 5. Vocabulary cleanup (Phase B)

- **Delete** canonical `SimClock` + `TimeMode`
  (`crates/physics_canonical/src/canonical.rs`) — unused, and the
  name collides with the real `crates/game/src/sim_clock.rs` clock.
- **Delete** `AuthorityMode::WarpIntegrated` and
  `AuthorityMode::Docked` — never constructed outside tests, no save
  format exists to break, and Phase C should design the perturbed
  coast (and a later docking pass should design docking) on their own
  merits rather than inherit placeholder variants. Pre-alpha teardown
  policy applies.
- **Split `crates/game/src/local_physics.rs`** (~3.6 k lines) into
  focused modules: `regime` (executors of the record), `snap_readback`,
  `forces`, `gear` (wheels, suspension, parking brake),
  `ground_contact` (friction, backstop, impact detection),
  `terrain_patch`, `frames` (the inertial↔SLF/body-centered seam),
  `colliders` (part-collider construction).
- **Rewrite `docs/simulation.md`** to separate "shipped architecture"
  from "target design" (the unbuilt provider-policy / ephemeris /
  crate-split material), and update CLAUDE.md per the
  announce-and-document policy.

## 6. Migration plan

Phase A — behavior parity at every step:

- **A1. This doc.** Plus pointers from `docs/simulation.md` and
  CLAUDE.md.
- **A2. Introduce the record + resolver in shadow mode.** The
  resolver computes `CraftRegime` alongside the existing machinery;
  a drift-check system derives the legacy values (`AvianRole`,
  authority transitions, warp caps, prediction gating) from the
  record and `warn!`s on mismatch. A Reflect-registered
  `RegimeDriftDiagnostics` resource makes mismatches BRP-readable.
  Acceptance: zero steady-state mismatches across the scenario
  matrix (`orbit`, `eva`, `landing`, `final`, `runway`,
  `runway-approach`, `cruise`); transient mismatches only on the
  documented one-frame edges (§3.2).
- **A3. Port consumers one at a time**, each a small parity-checked
  diff.
  - **Port #1 — authority executor: done (2026-06-12).**
    `crate::regime::apply_regime_authority` is the single writer of
    canonical `AuthorityMode` transitions, applying
    `regime::expected_authority` (payload construction + the release's
    warp reset + handoff diagnostics). Replaced and deleted:
    `manage_authority`, `release_landed_ship_on_throttle`,
    `collapse_or_constrain_warp` + `collapse_to_body_fixed`, the
    `LocalBubble::stable_contact_s`/`stable_landed` fields, and
    `thalos_physics_local::stable_contact_reached` (the settle timer
    now lives in `RegimeMemory`). Runtime-verified on the worktree
    harness: parked hold, landed throttle release (+ 1× warp snap),
    take/release translation handoffs through a full takeoff, the
    timed settle collapse through an autonomous landing, and the EVA
    walking pin — zero steady mismatches, and the release-edge
    authority blip from A2 is gone (the executor applies the
    expectation in-frame). The drift checker's authority check remains
    as an executor sanity check (it now only flags external authority
    writers — scenario teleports/respawn seeds — as blips).
  - **Port #2 — Avian role projection: done (2026-06-12).**
    `compute_avian_authority` now projects the record onto
    `AvianAuthority` via `regime::legacy_avian_role` (clock off →
    `Paused`, Backend translation → `Full`, else `AttitudeOnly`) —
    the resolver is the one classifier. Deleted:
    `avian_role_from_inputs`, `avian_role_for`, `craft_in_atmosphere`
    and their six unit tests (equivalent coverage lives in the pure
    `regime` tests). `AvianAuthority` survives as the distribution
    vehicle (its `previous_role` edge still drives the handoff snap),
    so every downstream reader — snap/readback, forces, gear,
    backstop, friction, impact, `sync_avian_time`, ship-view
    extrapolation — was ported in one diff at the source. One
    deliberate semantic change: walking projects `Paused` instead of
    the legacy incidental `Full`-in-atmosphere; verified safe because
    every `Full`-gated system is vessel- or EVA-guarded, and
    `sync_avian_time` keeps the capsule's clock via its
    `player_active` term. Runtime-verified: runway full-throttle
    release through liftoff; EVA at-rest 10× warp + walking clamping
    warp back to 1×, zero drift in all remaining checks.
  - **Port #3 — prediction gating: done (2026-06-12).**
    `bridge::update_prediction` clears/recomputes the trajectory plan
    from `CraftRegime.prediction`; deleted `ship_is_ballistic` and the
    grounded-EVA special case. Runtime-verified on the runway cycle:
    plan cleared while parked (`Hide(Landed)`), recomputed on
    release/liftoff (`Show`) — zero blips, including the formerly
    expected release-edge prediction blip (the record now *drives* the
    clear instead of shadowing it).
  - **Port #4 — impact-detector gates: subsumed by port #2
    (2026-06-12).** `detect_terrain_impact`'s only regime input is
    `owns_translation()`, which already flows from the record via the
    role projection; its remaining `VesselKind::Eva` check is a
    *capability* guard (no collider → no contacts), kept until parts
    declare capabilities.
  - **Port #5 — terrain-patch gates: done (2026-06-12).** The
    attach/detach/maintain systems gate on
    `CraftRegime.terrain_collider_allowed` (which folds the 1×
    warp-lock with the craft-has-a-collider capability — subsuming the
    old per-`VesselKind` EVA skip). Deleted
    `terrain_colliders_allowed_by_warp(_inputs)` + 2 tests (covered by
    the pure regime gate tests). The runway-body skip and the AGL
    geometry/hysteresis stay in the attach/detach systems — they are
    collider *placement*, not regime.
- **A4. Warp-policy consolidation: done (2026-06-12).**
  `enforce_warp_altitude_limits` is now a thin applier of
  `CraftRegime.warp` (publish `WarpLimits`, clamp the level); the
  ~100 lines of in-system predicates (altitude ladder, in-atmosphere
  1× clamp, surface-resting exemptions, on-foot at-rest rule) deleted
  — the one computation is the resolver's `warp_policy`, and
  `warp.constraint` is available for a HUD lock-reason readout.
  Runtime-verified: orbital ladder capped the climb at exactly the
  altitude-correct level (10 000× at ~200 km) and refused further
  escalation; on-foot at-rest 100× warp engaged and walking clamped
  it straight back to 1×; zero sanity-check failures. The drift
  checker's role/authority/warp comparisons are now tautological by
  construction and reduced to debug sanity asserts + the BRP record
  snapshot.
- **A4. Consolidate warp policy** into `CraftRegime.warp`:
  `enforce_warp_altitude_limits` shrinks to "apply the record";
  the HUD gains the constraint reason.
- **A5. Retire** `AvianAuthority`, the duplicated predicates, and
  the shadow-mode drift checker. Update CLAUDE.md's single-writer
  list (`CraftRegime` ← its resolver).

Phase B — §5, alongside or after A.

Phase C — capability (sketched only; gets its own design pass):

- **One force evaluation** — gravity + thrust + aero (+ buoyancy
  later), one signature usable by the backend executor *and* the
  `ShipPropagator`, buying drag-aware prediction and an honest
  perturbed-coast/warp-integration design (this is where a successor
  to `WarpIntegrated` is designed for real, if needed).
- **EVA-RCS**: the capsule becomes a normal dynamic craft with an RCS
  effector in the fly-by-wire allocator; walking remains the
  kinematic locomotion mode.
- **Per-render-frame f64 stepping** and the **Avian go/no-go** (§4).

## 7. Invariants

- One `CraftRegime` per craft, exactly one writer (the resolver).
- Downstream systems are executors: they read the record; they do
  not re-derive regime predicates from raw state.
- The classification core is pure Rust in `thalos_physics_canonical`,
  unit-tested, Bevy-free.
- Only the backend executor layer touches Avian types.
- Canonical `AuthorityMode` remains the persisted authority with its
  transition log; the record is frame-local and derived.
- The propagator-level collision and sub-surface warp resets remain
  as emergency backstops outside the resolver.
