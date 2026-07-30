# Multiple vessels and physical separation

This is the target contract for more than one craft in a Thalos save and for
stage separation that leaves both sides in the world. It specializes the
aggregate-craft model in [simulation.md](simulation.md), the authority resolver
in [regimes.md](regimes.md), and the local scene in
[surface_local.md](surface_local.md).

The player-facing rule is simple:

> Separation changes one vessel into two vessels. It never turns the detached
> side into a temporary visual effect.

Both sides retain mass, resources, geometry, motion, collision, and a stable
identity. A side with a command pod is controllable; a side without one is
debris, but it is simulated by the same vessel machinery and can fall, impact,
land, or remain in orbit.

## 1. Canonical fleet

`Simulation` evolves from one `CraftState` plus parallel singleton fields into
a deterministic registry keyed by stable `CraftId`:

```rust
pub struct FleetSimulation {
    vessels: BTreeMap<CraftId, VesselRecord>,
    active: Option<CraftId>,
    next_craft_id: CraftId,
    // shared clock, bodies, ephemeris, and propagator policy
}

pub struct VesselRecord {
    pub state: CraftState,
    pub authority: CraftAuthorityBook,
    pub parameters: ShipParameters,
    pub kind: VesselKind,
    pub controllable: bool,
    pub destroyed: bool,
    // resources, maneuver plan, and structural identity as they migrate
}
```

The exact type names may differ, but the ownership may not:

- every physical vessel has one canonical record;
- the active craft is a selection, not the only craft that advances;
- all vessels advance against the same simulation epoch and
  `BodyTrajectoryProvider`;
- deterministic systems iterate by `CraftId`, never Bevy entity order;
- existing single-craft methods may remain temporarily as active-craft
  wrappers, but new code takes a `CraftId`;
- `Entity` is a runtime projection and is never persisted as vessel identity.

Changing the active craft changes input, camera, HUD, prediction focus, and
which maneuver plan is edited. It does not wake, create, pause, or destroy any
vessel.

## 2. Runtime identity and ownership

Every flight root carries `CraftIdentity(CraftId)` and a general `CraftRoot`
marker. `PlayerShip`'s current double meaning—"a craft exists" and "this craft
is active"—is split. `ActiveCraft` resolves the selected `CraftId` to its
current root entity and is the only sanctioned active-craft lookup.

Parts belong to the nearest `CraftRoot` ancestor. Fuel, staging, inertia,
colliders, engines, gear, and HUD summaries scope their queries to that root;
no system may aggregate every non-editor `Part` in the world. Per-craft runtime
state becomes components on the root as each consumer is migrated.

`MapSnapshot` and real-space rendering project every canonical vessel. Camera
and detailed HUD remain active-only.

## 3. Separation transaction

Firing a decoupler is one atomic simulation transaction:

1. Compute the attach-graph cut before mutating ECS hierarchy.
2. Partition parts, resources, engines, colliders, and staging topology into
   the two resulting connected assemblies.
3. Keep the existing `CraftId` with the assembly containing the selected
   command pod/control point; allocate a new `CraftId` for the other assembly.
4. Recompute wet/dry mass, centre of mass, inertia, thrust, control authority,
   and remaining stage plan independently for both assemblies.
5. Derive each new centre-of-mass state from the pre-cut rigid motion:

   `v_child = v_parent + omega_world × r_world`

   then apply the authored decoupler impulse (`Decoupler::ejection_impulse`) as
   equal and opposite impulses along the separation axis.
6. Create/update both canonical records and both runtime roots before releasing
   the old aggregate physics body.

For a locally simulated craft, this transaction runs only after the current
frame's local-physics readback has installed the active vessel's inertial pose
at the current simulation epoch. `Simulation::step` advances the shared clock
but deliberately does not advance `LocalRigidBody` translation; cloning between
that clock advance and readback would create a detached `OnRails` vessel from
the previous frame's position while labelling it with the new epoch. At
heliocentric speed that one-frame skew is hundreds of metres and looks exactly
like the discarded stage disappearing.

Linear momentum and angular momentum must be conserved within numeric tolerance
apart from external forces integrated during the tick. The ejection impulse is
internal, so its net linear impulse is zero.

Catalog decoupler tuning must also open a visible gap across the supported
launch-vehicle mass range, and a fixed authored impulse cannot do that on its
own: the relative separation speed it buys falls off as 1/mass, so the standard
4 m decoupler's 8 kN·s gives a light probe metres per second and a fully fuelled
Saturn about 0.28 m/s. The applied impulse is therefore the authored spring
impulse **raised to a clearance floor**: the jettisoned assembly must fully clear
the geometry it was nested inside — its interstage shroud, or a nominal bare
decoupler face when there is none — within `SEPARATION_CLEARANCE_TIME_S`.
Relative speed is `impulse / reduced mass`, so the floor is
`reduced_mass × clearance / time`, which makes clearance mass-independent by
construction. The authored value still wins whenever it is already stronger, so
light separations stay as snappy as their hardware implies.

Each separation logs the detached mass, applied impulse, resulting relative
speed, and the clearance distance the floor was sized against; a persistent
vessel whose relative speed is only centimetres per second is a tuning failure,
not evidence that rendering or graph partitioning lost the stage.

The interstage shroud (`thalos_runtime::shrouds`) is a child of the decoupler, so
the graph cut carries it down with the jettisoned stage — the KSP convention, and
the reason clearance is measured against the shroud's height rather than the
decoupler's own. Separation strips the decoupler's `Attachment`, which would make
it stop qualifying for a shroud, so the cut stamps `ShroudFired` on the shroud and
the reconcile pass leaves it alone from then on.

Separation does not automatically switch control. The active identity follows
the selected command pod. If only one side has a command pod, that side remains
controllable; the other is debris. Multiple command-pod assemblies may later be
selected through the normal active-craft switch.

## 4. Local physics and falling stages

The local scene is shared per dominant body, not owned by one craft. It holds
one terrain/contact frame and N vessel rigid bodies:

```text
LocalScene(body, frame)
├── active vessel rigid body
├── detached stage rigid body
├── nearby landed craft
└── shared terrain and structure colliders
```

Selection affects input only. Every hydrated vessel receives gravity,
atmospheric/aerodynamic forces where supported, collision, and contact
resolution. A stage separated during ascent therefore drifts under its ejection
impulse, falls under gravity, and can strike the terrain while the player keeps
flying the upper stage.

Vessels hydrate and collapse independently. Leaving the local range collapses
that vessel to its canonical aggregate state; entering the range hydrates it
again. Warp never deletes debris. Outside the local scene, an inactive vessel
continues through the canonical propagator and contact/event detection. A
surface impact transitions it to the same destroyed or `BodyFixed` outcomes as
an active vessel.

The first implementation may use one aggregate rigid body and compound collider
per separated assembly. Per-part joints, bending, and fragmentation are later
fidelity; they are not prerequisites for real persistent separation.

## 5. Delivery slices

1. **Fleet kernel — landed 2026-07-25.** Canonical records are keyed by
   `CraftId`; complete per-vessel authority/parameters/control/maneuver/
   prediction/fuel/damage state moved into `VesselRecord`; active-craft
   compatibility wrappers remain; id-addressed state, authority, and local
   writeback APIs are available; all vessels advance deterministically over
   one world interval. A collision is treated as a world event: the fleet
   replays to the earliest contact time so every record and the shared clock
   keep the same epoch. `MapSnapshot` now copies every canonical craft.
2. **Runtime multiplicity — vertical slice landed 2026-07-25.**
   `CraftIdentity`/`CraftRoot`/`CraftPart` now bind rendered roots and flight
   parts to canonical identity. Real-space rendering syncs every root, while
   mass, propulsion, inertia, staging topology, and HUD summaries scope to the
   active craft. Map markers, active selection, and the remaining per-craft
   runtime resources still belong to CL-E2.
3. **Shared local scene.** Support two aggregate rigid bodies in one body-fixed
   frame with independent authority/readback and collision.
4. **Physical staging — visible persistence slice landed 2026-07-25.**
   Subtree despawn is gone. A graph cut reparents each detached assembly under
   a new visible root, creates its canonical OnRails record, partitions live
   resources by part ownership, recomputes both aggregates, and applies the
   authored impulse equally/oppositely (including the parent's Avian velocity
   mirror). Detached local compound colliders, angular `ω×r` inheritance, and
   impact/settling fidelity remain with the shared-local-scene follow-up.
5. **Control and persistence.** Craft switching, active-only UI focus,
   save/load of every vessel, and recovery/cleanup policy.

## 6. Acceptance

Agent-verifiable:

- pure tests show two vessels advance under one clock and active selection
  cannot change either trajectory;
- graph-cut tests preserve every part exactly once and partition resources;
- separation tests conserve mass and momentum and apply equal/opposite authored
  impulse;
- a local-scene test keeps two rigid bodies alive and writes each back to the
  matching `CraftId`;
- no flight aggregation query can see parts owned by another root;
- `just check thalos_game`, `just test`, and `just clippy` pass.

User-verifiable:

- launch a two-stage craft, stage during ascent, and keep flying the upper
  stage while the booster remains visible and falls;
- watch the booster impact or settle instead of disappearing;
- enter map view and see both trajectories;
- switch to another controllable separated assembly and back without either
  vessel teleporting or pausing;
- time-warp away and return; both objects still exist at their propagated
  positions.

## 7. Deferred fidelity

Docking/undocking assemblies, debris lifetime policy, parachutes, thermal
destruction, recovery value, and multiplayer are separate features. They build on
canonical vessel multiplicity; none should introduce a second debris-only
simulation path.

**Per-part structural breakup is no longer deferred** — it is owned by
[damage.md](damage.md) (ADR-20260730T003005Z), which fires §3's separation
transaction from a severed joint with a failure impulse in place of a
decoupler's authored one.
