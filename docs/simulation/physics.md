# Physics — the owned solver (`thalos_physics`)

> **Status (decided 2026-06-28):** Thalos is replacing **Avian** with an
> owned physics solver, **`thalos_physics`**, built on `parry3d-f64`. The
> solver family is **TGS-Soft** (substepped soft-constraint, the
> Box2D/Rapier lineage) — *not* literal XPBD (patent note below). This doc
> is the target design + phased roadmap. The legacy backend-seam analysis in
> `docs/simulation/regimes.md` §4 ("Avian stays through Phases A–B") is **superseded by
> this decision** — it correctly scoped the seam work, which becomes Phase 0
> here.

## 1. Why replace Avian

Three reasons, each independently sufficient, that compound:

1. **Dual-integrator impedance mismatch.** Avian and our canonical
   (Kepler) integrator co-own one craft state, reconciled every frame by a
   snap/readback dance plus `Time<Physics>` clock gymnastics. That
   reconciliation is the structural source of surface drift, pops, and the
   role/authority bookkeeping complexity. An *owned* integrator that *is*
   the surface-regime owner removes the reconciliation entirely.

2. **The BeamNG endgame.** The long-term goal is BeamNG-style soft-body
   damage/deformation. A BeamNG vehicle has **no rigid body** — it is a
   node-beam mass-spring network (nodes = point masses, beams =
   spring-dampers with plastic yield + breaking). No rigid-body engine
   (Avian, Rapier, PhysX, Jolt) gives this — their soft bodies are elastic
   FEM/XPBD blobs that spring back, not structural crumple. So a rigid-body
   engine is the **wrong substrate for vehicles** regardless; we'd tear it
   out either way. Owning the solver is the path that scales from today's
   rigid vehicle to tomorrow's deformable one.

3. **Crate cleanliness.** `parry3d-f64` (the collision layer Avian wraps)
   is pure Rust with no Bevy. An owned solver on top of it is also
   Bevy-free, so it lives outside the engine and `physics_local`'s only job
   — quarantining Avian's Bevy dependency — ends. (See §4.)

## 2. What we keep, what we build

- **Keep — `parry3d-f64`** (dimforge, Apache-2.0, pure Rust, no Bevy,
  native f64). It provides every hard collision primitive: heightfield
  colliders, compound/convex shapes, contact-manifold generation, raycasts,
  shape casts, TOI/CCD. This is "the underlying stuff." It is the *only*
  external physics dependency.

- **Build — `thalos_physics`** (new crate): the owned solver. Bodies,
  constraints, contacts, friction, integration. No Bevy.

- **Solver family — TGS-Soft**, not XPBD. Temporal Gauss-Seidel with soft
  constraints + substepping (Erin Catto / Box2D v3 lineage; what Rapier and
  current Avian both use). It delivers everything we wanted from XPBD —
  small substeps, soft/compliant constraints, stiff contacts that don't
  explode, stiffness decoupled from iteration count — and is the
  *more battle-tested* choice for our core case (one body resting/landing on
  static terrain). It is also **patent-clean** (see §6). The node-beam
  endgame is unaffected: beams become compliant soft distance constraints
  with yield/break, which is integrator-agnostic.

## 3. Surface-stability design invariants

The hard requirement: **on-surface gameplay is smooth and stable at all
times — no floating-point desync, rock-steady rest, stable
landing/collision.** These are the non-negotiable design rules that follow:

1. **Solver runs entirely in the Surface-Local Frame (SLF), in f64 — never
   at planet-radius coordinates.** This is *the* fix for floating-point on a
   planet. Planet-radius coords (millions of m) give ~0.5 m precision in f32
   and burn f64 headroom on stiff-contact math. The SLF anchors a body-fixed
   tangent frame *under the craft* (meters-to-km coordinates), re-anchored
   every ~1.5 km of drift, so f64 gives sub-micron precision at the contact.
   The solver must be structurally incapable of leaving this regime. (The
   SLF math already lives in `physics_canonical::surface_local`; see
   `docs/simulation/surface_local.md`.)

2. **Single owner of the surface state → no internal desync.** In the
   surface regime the canonical state *is* the solver state — nothing is
   reconciled per frame. Kepler↔surface handoff happens once, at a regime
   boundary, not every frame. (`docs/simulation/regimes.md` owns the regime classifier
   that fires these boundaries.)

3. **Fixed-timestep physics + accumulator + render interpolation.** A fixed
   physics `dt` (accumulator drains real frame time; render interpolates
   between the last two physics states). Variable-dt is a known cause of
   stiff-contact/spring blowups. **This overrides the `docs/simulation/regimes.md` §4
   note about "variable-dt stepping for free"** — the stability requirement
   settles that tradeoff toward fixed-dt + determinism.

4. **Position/soft-constraint contact → jitter-free rest.** A lander on the
   heightfield is the solver's easiest case: soft one-sided contact
   constraints push out of penetration with no bounce/sink; friction is a
   per-substep tangential clamp (≤ μ·normal); substepping keeps stiff
   contact stable. This *retires* the old gearless-lander rest battles
   structurally rather than band-aiding them.

5. **Anti-tunneling retained.** parry TOI/shape-casts (CCD) + the analytic
   `terrain_floor_backstop` keep fast descents from punching through.

6. **Kinematic bodies from day one.** Bodies whose pose is set externally
   (infinite mass, contact obstacle, not integrated). The hook for
   replicated remote craft (§5) — and useful for moving platforms/structures
   generally.

## 4. Crate shape

End state replaces `{physics_canonical, physics_local}` with
`{thalos_physics, physics_canonical}` — both Bevy-free, Avian gone:

- **`thalos_physics`** *(new)* — the solver. Depends only on `parry3d-f64`
  (+ glam/serde). No Bevy. Generic: bodies, constraints, contacts,
  integration; later the node-beam soft-body layer.
- **`physics_canonical`** — unchanged role: the *domain* layer (orbital
  mechanics, canonical craft state, SLF tangent-frame math, the regime
  resolver). Bridges to `thalos_physics` in the SLF.
- **`physics_local`** — **dissolves.** Once Avian is gone, its
  quarantine job is done; the remaining thin Bevy ECS glue (components,
  plugin wiring, collider construction) moves into `crates/runtime/game`.

This delivers the original "merge the physics crates" instinct: the
Avian-quarantine crate disappears. The split that remains —
generic-solver vs. domain — is the *right* one.

## 5. Determinism & the multiplayer model

**Determinism target: single-player stability, not bit-exact
cross-platform lockstep.** The sim, collider, and render never drift apart;
the craft is rock-steady; no FP blowups at planet scale. Two different
machines are *not* required to produce identical results.

This is sufficient because the intended multiplayer model is **client
authority + state replication**, not lockstep:

- Each machine **fully simulates only its own craft** (physics + deformation)
  and broadcasts its state: transform at high rate, linear/angular velocity
  for dead-reckoning between updates, deformation deltas at low rate / on
  damage events.
- Remote craft are **kinematic, position-driven copies** (§3.6),
  interpolated (render slightly in the past) or extrapolated (dead-reckon).
- **No craft is ever simulated on two machines**, so machines never
  co-simulate the same body — bit-exact determinism is unnecessary, and the
  solver math is not constrained by netcode.

**Inter-player collisions** are resolved locally and approximately: your
machine computes *your* craft's response (and crumple) against the remote
craft as a kinematic obstacle from its latest received state; the other
machine does the mirror. The two views won't agree perfectly on a crash —
inherent to all non-lockstep physics netcode, and accepted in shipping
games. **Prior art:** BeamMP runs exactly this model with node-beam
deformation, so the BeamNG endgame and this netcode are known-compatible.

The only solver requirement this adds is kinematic bodies, already in §3.6.
Networking is a later layer; nothing in the solver is built for it now
beyond that hook.

## 6. Patent note (XPBD vs TGS-Soft)

The specific **XPBD / "Small Steps" methods are NVIDIA-authored and
patent-encumbered** — this is the *stated reason* Avian dropped XPBD for
TGS-Soft ([Avian #346](https://github.com/Jondolf/avian/issues/346)) and
Rapier never adopted it. General position-based dynamics predates that work
(Jakobsen Verlet; Müller PBD 2007), so the exposure is narrow, but since the
flagship Rust engine deliberately walked away, we sidestep it at **zero
cost** by choosing TGS-Soft (Catto/Box2D lineage). *Not legal advice;*
revisit if scope changes.

## 7. Roadmap

- **Phase 0 — Tighten the backend seam.** *(in progress — 2026-06-29)* Shrink
  the Avian-facing surface so *only* the executor allowlist (§7.1) names backend
  types; everything else reads canonical state / the `CraftRegime` record / the
  Avian-free `LocalCraftKinematics` readout. Keeps Avian; behaviour-preserving;
  makes the swap surgical. Landed so far: the `LocalCraftKinematics` readout
  (published by `publish_local_craft_kinematics` at the end of the physics
  chain); `control_bus` redirected off Avian onto it; the runway slab
  generalised into the executor's `StructureCollider` /
  `spawn_structure_collider` / `sync_structure_collider_pose`
  (`runway.rs` no longer names Avian); and a CI guard enforcing §7.1. EVA
  (`player_controller`/`debug`) stays on its body-centered seam by design.

### 7.1 Executor allowlist (CI-enforced)

The **only** code permitted to name `thalos_physics_local::avian` types
(enforced by the `boundaries` job in `.github/workflows/ci.yml`):

| Module | Role |
|---|---|
| `crates/simulation/physics_local/**` | the boundary crate (Avian re-export + collider/body/readout construction, clock/plugin setup) |
| `crates/runtime/game/src/local_physics/**` | the per-frame executor systems: snap/readback, forces, gear, ground, terrain/structure collider lifecycle, clock |
| `crates/runtime/game/src/aero.rs` | aero force layer — runs *inside* the Avian substep (`PhysicsSchedule`), reads live kinematics, writes force/torque |
| `crates/runtime/game/src/regime.rs` | the authority executor + ground-contact classifier (reads `ContactGraph` + the settle SLF pose) |
| `crates/runtime/game/src/player_controller.rs`, `crates/runtime/game/src/debug.rs` | the **EVA exception** (body-centered kinematic seam; `surface_local.md` §10) + debug hitbox/teleport |

Everything else reads `LocalCraftKinematics` (SLF kinematics), `CraftStateMirror`
(inertial canonical), or the `CraftRegime` record. When `thalos_physics` lands,
this allowlist is what gets re-pointed at the owned solver — module by module.

- **Phase 1 — Stand up `thalos_physics` in shadow mode.** *(planned, **deferred
  2026-06-29** — detailed plan in §7.2, verification in §7.3.)* TGS-Soft rigid
  body (single aggregate craft) + parry contacts against static colliders
  (heightfield, runway cuboid), fixed-dt substep loop in the SLF. Validated
  **test-first** (analytic ground truth + headless-Avian parity), with a thin
  live shadow diff for integration. Zero gameplay risk while confidence builds.

- **Phase 2 — Cut over + collapse crates.** Flip scenarios one at a time,
  re-verifying each: orbit handoff → final → landing → runway → EVA. The
  gearless-lander ground-contact case gets its own verification slice (it
  re-opens the won stability battles). Then **delete Avian**, fold the
  remaining glue out of `physics_local` into `game`, and update `CLAUDE.md`
  to the new crate shape.

- **Phase 3 — Node-beam deformation (the BeamNG layer).** Vehicles become
  node graphs in the same solver; compliant beams + plastic yield + beam
  break = damage; nodes couple one-way to parry terrain (penalty/soft
  contact + Stribeck-Coulomb friction); wheels become node-beam
  sub-assemblies. Two sizable *new* problems land here, flagged now:
  **(a)** auto-generating a node-beam structure from the *parametric
  shipyard blueprint* (BeamNG hand-authors these), and **(b)** skinning part
  meshes to node positions so the visible craft deforms. This is a milestone
  of its own, not a tail of the rewrite.

  > **Narrowed 2026-07-30 by ADR-20260730T003005Z.** Phase 3 no longer owns
  > the damage model, and destruction gameplay does not wait for it. Damage
  > state is **canonical** (`VesselDamage` on `VesselRecord`, Bevy-free,
  > persisted) and reaches flight through one rebuild path; the solver only
  > ever produces *load events*. A structural graph over today's aggregate
  > rigid body ships first — see [damage.md](damage.md) — and node-beam later
  > becomes a better *producer* of that same state. What remains uniquely
  > Phase 3's: real crumple geometry, suspension ruin, and mesh-to-node
  > skinning.

### 7.2 Phase 1 detail (deferred plan)

**Crate `thalos_physics`** — pure Rust, depends only on `parry3d-f64` + glam
(add it to the bevy-free CI guard when it lands). Modules:
- `body` — owned `RigidBody`: mass + inverse mass, body-frame inertia + inverse,
  position (`DVec3`), orientation (`DQuat`), linear/angular velocity,
  dynamic/kinematic flag.
- `collider` — wrappers over `parry3d_f64::shape::SharedShape`, built from the
  *same* neutral `LocalPrimitiveCollider`/heightfield geometry the executor emits
  (so the shadow body and the Avian body have identical shapes).
- `contact` — parry narrow-phase (`contact_manifolds`) + persistent feature IDs
  for warm-starting.
- `solver` — the TGS-Soft contact constraint solver.
- `integrate` — symplectic integrator + the fixed-dt substep loop.
- `query` — raycast/shapecast wrappers (gear suspension, terrain backstop).
- `world` — container; one `step(dt, &force_inputs)` entry point.
- **Seam:** parry speaks nalgebra (`Point3`/`Isometry3`/`UnitQuaternion`), we
  speak glam (`DVec3`/`DQuat`) — a small, tested conversion layer at the parry
  boundary is the only friction.

**TGS-Soft step** — fixed `dt`, N substeps of `h = dt/N` (Macklin small steps).
Per substep: (1) integrate velocity from net force/torque; (2) warm-start
contacts by feature ID; (3) solve contacts with **soft constraints** — a
spring-damper `(hertz, damping)` yielding a bias velocity + mass/impulse scaling
(Catto biasRate/massScale/impulseScale) instead of a hard position snap; friction
= tangent impulse clamped to `μ·normalImpulse`, warm-started; (4) integrate
position; (5) relax pass *without* bias to bleed off bias energy. Static ground =
`inverse_mass 0`, so only the craft velocity changes — the per-substep solve is a
few points × a couple relaxation passes (the easiest contact problem there is).

**Sub-stages (lowest-risk first):**
- **1a — integrator parity (no contacts):** 6-DoF symplectic integrator in the
  SLF, shadowed while airborne; should match Avian within `~|accel|·dt²`.
- **1b — contacts (the hard part):** parry manifolds + the TGS-Soft resolver,
  shadowing the **gearless-lander rest + touchdown** (rock-still rest, no
  jitter/sink/launch). **~90% of Phase 1 risk lives here** — the soft-constraint
  tuning, warm-starting, and friction-at-rest.
- **1c — gear raycast:** parry raycast spring-damper for wheeled craft (the gear
  already applies forces; the solver just integrates), shadowing runway taxi/land.

**Out of scope for Phase 1:** no cutover, no Avian deletion, no node-beam, no
networking/kinematic-remote, no `physics_local` dissolution (those are Phase 2+).

### 7.3 Phase 1 verification (test-first, deferred)

Tests are the **primary** gate — runnable headless without the game, so the loop
doesn't depend on a human running it; the live shadow harness shrinks to an
integration sign-off.

- **Layer 1 — analytic tests *in `thalos_physics`*** (pure, no Avian; the real
  correctness proof — matching ground truth beats matching another approximator):
  free-fall vs `x₀+v₀t+½at²`; free rotation conserves angular momentum; rest on a
  plane settles with bounded penetration + **no creep/jitter** + energy
  non-increasing; high-speed-into-thin-collider doesn't tunnel (parry
  TOI/backstop); incline friction stays/slides at the friction angle.
- **Layer 2 — Avian-parity tests *in `physics_local/tests/`*** (dev-deps
  `thalos_physics`; a **headless** `MinimalPlugins + PhysicsPlugins` Bevy `App` —
  no window/GPU). Identical scenario in both engines, **bounded-divergence**
  asserts (rest pose ~cm, settle time ~%, terminal velocity), **not** equality —
  engines diverge, especially in/after contact, so contact leans on Layer-1
  invariants. **Crate-boundary rule:** this comparison must NOT live in
  `thalos_physics` (which stays Bevy/Avian-free on the CI guard); dev-deps don't
  count against `cargo tree -e normal`, so the test belongs in `physics_local`.
- **Layer 3 — thin live shadow** for what tests can't reach: real craft geometry,
  the real gear/aero force systems feeding the solver, SLF re-anchoring, regime
  handoffs, warp. Writes JSONL for a once-per-scenario integration sign-off (the
  only step needing a game run).
- **Fixtures:** one shared set (free-fall, rest-on-plane, drop-and-settle, spin,
  incline-friction, landing-decel) run through every layer, using **synthetic
  analytic colliders** (plane / cuboid / simple heightfield) — never the real
  `ProceduralSurface` (per the terrain-gen test ban in `CLAUDE.md`; keeps tests
  fast and dependency-light).
- **Acceptance:** green `cargo test -p thalos_physics` + `cargo test -p
  thalos_physics_local`, CI-gated.
- **Plumbing risks:** stepping headless Avian deterministically (configure its
  fixed timestep / drive `PhysicsSchedule` directly); not over-tightening parity
  tolerances (assert bands, not equality).

## 8. References

- Collision: [parry3d-f64](https://crates.io/crates/parry3d-f64) ·
  [parry.rs](https://parry.rs/)
- Solver lineage: Erin Catto "Soft Constraints" / Box2D v3 TGS-Soft;
  [Rapier](https://rapier.rs/) TGS-Soft; Macklin et al.
  ["Small Steps in Physics Simulation"](https://mmacklin.com/smallsteps.pdf)
  (substepping intuition).
- Fixed timestep: Gaffer
  ["Fix Your Timestep!"](https://gafferongames.com/post/fix_your_timestep/).
- Node-beam / soft body:
  [Rigs of Rods](https://github.com/RigsOfRods/rigs-of-rods) (open-source
  node-beam reference), Müller
  ["Ten Minute Physics"](https://matthias-research.github.io/pages/tenMinutePhysics/)
  L9/L10, Provot (1995) plastic mass-spring,
  [BeamMP](https://beammp.com/) (networked node-beam prior art).
- Patent context: [Avian #346](https://github.com/Jondolf/avian/issues/346).
