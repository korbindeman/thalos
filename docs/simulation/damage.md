# Craft damage and destruction

The target contract for graded, survivable craft damage — the BeamNG-flavoured
model, delivered on the *current* rigid body rather than waiting for the
node-beam solver. It specializes the vessel registry in [vessels.md](vessels.md),
the force model in [aerodynamics.md](aerodynamics.md), and the solver roadmap in
[physics.md](physics.md).

The player-facing rule is:

> Damage is graded and survivable. A part's **function** fails before its
> **structure** does, and a craft that has lost a wing is a craft you fly home,
> not a craft that has ended.

Decided 2026-07-30 (ADR-20260730T003005Z): staged delivery — a structural graph
over the existing aggregate rigid body now, node-beam later as a *better producer*
of the same damage state. Per-panel aero is in scope, and damage persists.

## 1. Three layers, deliberately separate

| Layer | Owns | Depends on |
|---|---|---|
| **Damage state** | per-part structural integrity, per-*function* condition, per-joint yield | nothing |
| **Consequence** | how that state changes flight: asymmetric lift, thrust loss, CoM shift, jammed hinges, leaks | per-panel aero |
| **Damage read** | how the player perceives it: deformation, debris, effects, HUD | procedural meshes |

The layers do not know about each other's internals, and **none of them lives in
the physics backend**. The solver produces *load events*; it never owns damage.
That is the seam that makes `thalos_physics` Phase 3 (node-beam) an
implementation swap instead of a rewrite of this feature.

## 2. Canonical ownership

`VesselDamage` lives in `physics_canonical` beside the rest of `VesselRecord` —
Bevy-free, `serde`, keyed by stable ids, never by `Entity`:

```rust
pub struct VesselDamage {
    parts:  BTreeMap<PartKey, PartDamage>,
    joints: BTreeMap<JointKey, JointDamage>,
}

pub struct PartDamage {
    /// Structure, 1.0 pristine → 0.0 failed. Monotone non-increasing
    /// outside a repair transaction.
    pub integrity: f32,
    /// One entry per function this part kind declares.
    pub functions: Vec<FunctionCondition>,
}

pub struct FunctionCondition {
    pub kind: PartFunction,
    /// 1.0 nominal → 0.0 dead. Degrades independently of `integrity`.
    pub condition: f32,
    /// Latched hard failure; a latched function never recovers without repair.
    pub failed: bool,
}

pub struct JointDamage {
    /// Accumulated plastic deformation, 0 = pristine.
    pub yielded: f32,
    pub severed: bool,
}
```

**Identity.** `PartKey(u32)` is the part's index in its `ShipBlueprint::parts`
([blueprint.rs:177](../../crates/domain/construction/src/blueprint.rs:177)), assigned at
spawn. Every part has exactly one parent in the attach tree, so `JointKey(PartKey)`
— "the joint attaching this part to its parent" — is unique without a second
index. Keys are **never re-indexed**: after a structural cut, each resulting
vessel keeps the subset of keys it inherited, so damage survives separation with
no remapping. Blueprint indices already round-trip through save files, so
persistence is a consequence of this choice rather than extra work.

Because damage is canonical it survives regime handoff, warp, collapse-to-rails,
save/load, and the local-scene hydrate/collapse cycle for free.

**`is_destroyed` becomes derived.** Today it is a `hull_destroyed: bool` flipped
by a whole-craft ~12 m/s approach-speed tolerance
([simulation.rs:898](../../crates/simulation/physics_canonical/src/simulation.rs:898),
[ship_view.rs:52](../../crates/runtime/game/src/ship_view.rs:52)). It becomes a verdict
computed from `VesselDamage` — no surviving command-pod function, or the
structure carrying the pod fully severed. The `is_destroyed()` accessor keeps its
signature, so `control_bus`, `fuel`, `bridge`, and `scenario_menu` need no change.

## 3. Load channels — how function fails before structure

Damage sources are **channels**. Each part kind declares, in its catalog spec, a
structural limit per channel and a *fraction of that limit* per declared
function. The functional fraction is what delivers the player-facing rule: the
turbopump quits at a g-load the engine bell shrugs off.

| Channel | Measured from | Structural target | Typical functional targets |
|---|---|---|---|
| **Contact impulse** | solver contact impulse / gear weight-on-wheels | contacting part `integrity`, parent joint `yielded` | anything on that part |
| **Sustained g-load** | CoM acceleration + `ω×(ω×r)` at the part's station | joint `yielded` (bending) | `Thrust` (turbopump), `ReactionWheel`, `Avionics` |
| **Aero load** | per-panel force magnitude vs. panel limit (§4) | panel `integrity`, root joint | `ControlSurface` (hinge jam) |
| **Dynamic pressure** | `q̄` past the part's rated limit | — | `Intake`, `GearStrut` (deploy over V_max) |
| **Thermal** | later, with reentry heating | — | — |

Failure is **deterministic plus a seeded per-part offset** drawn from
`(craft_id, part_key, function)` — never runtime RNG. Two reasons: the fleet
replays to the earliest contact time on a collision, and a save/load must not
re-roll a marginal part.

**Replay safety.** Plastic accumulation is history-dependent, so naive
accumulation is double-applied when the fleet replays an interval. Load events
are staged per epoch and **committed once, at interval close**, after the
replay resolution has settled the clock. A replayed interval discards its staged
events and re-accumulates them. This is the single subtlest correctness
requirement in the feature and it gets its own test.

## 4. Per-panel aero

The current model is whole-body: one `AeroConfig` with one reference area, one
lift slope, and roll moment only from control deflection
([aero.rs:70](../../crates/simulation/physics_canonical/src/aero.rs:70)). It has **no term
that can express asymmetry**, so no amount of damage state makes a one-winged
craft roll. This is the prerequisite, not an optional fidelity upgrade.

The panel geometry already exists and is already per-panel:
`WingAeroPanel` carries `center_body_m`, `fore_dir`, `thick_dir`, `span_dir`,
area, chord, span, and its control-surface windows
([stats.rs:577](../../crates/domain/construction/src/stats.rs:577)) —
`build_ship_aero_config` simply *collapses* the list. The work is to stop
collapsing it.

**The model.** `evaluate_aero` becomes a panel sum: for each panel, local flow
`v + ω×r` at the panel's aerodynamic centre, its own α from its own basis, lift
and drag at that centre, accumulated as `F` and `r×F`. The fuselage keeps a
whole-body bluff/residual term. Consequences worth stating:

- asymmetry is **emergent** — a missing or degraded panel produces rolling *and*
  yawing moment with no special case;
- roll damping and roll-control authority stop being tuned coefficients and
  become geometry, which removes tuning surface rather than adding it;
- a jammed control surface is a panel with zero authority and a **fixed offset
  deflection**, which is a real and interesting failure mode;
- panel `integrity` reduces effective area and lift efficiency, so partial
  damage is expressible.

**The calibration gate.** The Meridian's handling is tuned against the
whole-body model, so the refactor must not be judged by feel alone. A pure test
in `physics_canonical` asserts that, for the Meridian panel set at its cruise
point, the panel sum reproduces the whole-body config's CL / CD / Cm within a
stated band and trims within a stated angle. The retune then becomes a bounded,
testable job. The whole-body model survives **only as that test's fixture** — it
is not kept as a runtime path.

**Crate boundary.** `WingAeroPanel` lives in `thalos_shipyard`, which depends on
Bevy; `physics_canonical` may not. The panel array type the force model consumes
is therefore declared in `physics_canonical::aero` (glam only), and the runtime
converts. Same shape as every other pure/consumer seam in the workspace.

## 5. Consequence — one rebuild path

Every trigger — impact, joint sever, function failure, repair — routes through
one function, `rebuild_damaged_aggregates(craft_id)`, which recomputes the
derived craft from (blueprint ∪ damage):

- **aero** — the panel list with per-panel integrity and control-surface
  condition applied, then the panel-sum config;
- **mass, CoM, inertia** — the existing `recompute_ship_inertia`
  ([staging.rs:804](../../crates/runtime/game/src/staging.rs:804));
- **thrust** — `EngineThrust` gated by the `Thrust` condition; a degraded engine
  produces reduced thrust and is more likely to latch a hard failure;
- **containment** — a degraded `ResourceContainment` bleeds its `PartResources`
  at a rate proportional to the damage;
- **control authority** — `thalos_control` already allocates against the config's
  per-axis authority, so a dead aileron reduces roll authority with no
  damage-specific code in the allocator.

No consumer reads `VesselDamage` to decide flight behaviour. They read the
rebuilt aggregates. That keeps the standing "one canonical path per operation"
bar and means the node-beam swap touches only the *producer*.

## 6. Structural failure and debris

A severed joint fires the separation transaction that already exists —
`materialize_separated_vessels`
([staging.rs:438](../../crates/runtime/game/src/staging.rs:438)) — with a failure
impulse in place of a decoupler's authored one. The detached assembly becomes a
canonical vessel per ADR-20260724T230226Z: it falls, collides, persists through
warp, and appears in map view. There is no debris-only code path.

This retires [vessels.md](vessels.md) §7's "per-part structural breakup" deferral.

## 7. Damage read without art

Part meshes are **parametric and procedural** (`wing_mesh`, `fuselage_mesh`,
`gear_mesh`), which is strictly better than authored art for deformation: there
is no rig and no skin weights, and deformation is a parameter change on geometry
we generate anyway. Three mechanisms, in ascending cost:

1. **Detachment** — carries the strongest read and is already built (§6).
2. **Effects** — engine smoke, venting fuel, contact sparks and scrape. Art-free
   and disproportionately legible.
3. **Parametric deformation** — a yielded joint applies a kink to the child's
   generated mesh (a wing takes a dihedral/incidence offset and a break-station
   taper; a fuselage loft takes a station-local radius perturbation).

Real crumple geometry needs node-beam *and* good models, and reads weakest at
Thalos's usual camera distances. It is last for that reason, not merely because
it is expensive.

## 8. Persistence and repair

`VesselDamage` is `serde` on `VesselRecord`, so save/load is a round-trip test
rather than a feature. Damage **persists across flights**: a craft recovered with
a dead reaction wheel still has one.

Repair is scoped, never ambient: full repair at a base with the relevant
facility, nothing in flight. `scenario_menu` already tracks `is_destroyed()` and
closes the moment any path clears it
([scenario_menu.rs:167](../../crates/runtime/game/src/scenario_menu.rs:167)), so the
repair path has a home.

## 9. Delivery slices

| ID | Slice |
|---|---|
| **DMG-1** | Canonical damage state: `VesselDamage` in `VesselRecord`, `PartKey`/`JointKey`, serde persistence, `is_destroyed` re-derived behind the existing accessor. No behaviour change yet. |
| **DMG-2** | Per-panel aero (§4) + the calibration gate. Independently valuable and independently verifiable; earliest because it carries the most risk. |
| **DMG-3** | Load channels + catalog limits (§3): structural vs. functional limits per part kind, seeded offsets, replay-safe commit. |
| **DMG-4** | `rebuild_damaged_aggregates` (§5) — the one consequence path. |
| **DMG-5** | Structural sever → debris via the existing cut (§6). |
| **DMG-6** | Damage read (§7): effects, then parametric deformation, plus a HUD damage surface. |
| **DMG-7** | Repair scopes and facilities (§8). |

DMG-2 precedes DMG-3/4 because consequence has nothing to act on without it.

## 10. Acceptance

Agent-verifiable:

- panel-sum parity test at the Meridian cruise point within the stated band;
- a panel removed from one side produces rolling *and* yawing moment of the
  expected sign and rough magnitude;
- a functional limit fires below its structural limit for every part kind that
  declares both (table-driven test over the catalog);
- damage accumulation is idempotent across a replayed interval;
- a save/load round-trip preserves `VesselDamage` exactly;
- a severed joint conserves mass and momentum through the existing cut tests;
- `just check thalos_game`, `just test`, `just clippy` pass.

User-verifiable:

- land hard: the airframe survives, a system does not, and the HUD says which;
- shear a wing in flight and fly the craft home on asymmetric lift;
- lose an engine and finish the flight on the remainder;
- recover the craft, and the damage is still there next flight;
- repair at a base and the craft is nominal again.

## 11. Deferred

Node-beam deformation (physics.md §7 Phase 3) and everything that needs it: real
crumple geometry, suspension ruin, mesh skinning to nodes. Thermal/reentry
damage, structural fatigue over a craft's lifetime, and multiplayer damage
replication are separate features. None of them changes the contract above.
