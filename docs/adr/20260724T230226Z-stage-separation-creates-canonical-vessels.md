# ADR-20260724T230226Z-stage-separation-creates-canonical-vessels: Stage separation creates canonical vessels

- **Status:** Accepted
- **Date:** 2026-07-24

## Context

Thalos currently owns one canonical `CraftState`. Runtime staging already
computes the correct attach-graph cut, but implements separation by despawning
the detached subtree and reducing the surviving craft's aggregate mass. That
cannot support a booster that visibly falls, persists through warp, collides
with terrain, appears in map view, or becomes controllable when it contains a
command pod. A Bevy-only debris object would fix the immediate visual symptom
while creating a second, non-persistent physics authority outside the canonical
simulation.

## Decision

Canonical simulation becomes a deterministic registry of vessel records keyed
by stable `CraftId`, with active craft stored as a separate selection.

Every connected assembly produced by staging is a vessel record. The existing
identity follows the assembly containing the selected command pod/control
point; every other assembly receives a new identity. Assemblies with a command
pod are selectable, while assemblies without one are debris capabilities of
the same vessel model.

Separation partitions the existing part graph and resource state, recomputes
both aggregates, derives both centres of mass from the pre-cut rigid motion,
and applies the decoupler's authored ejection impulse equally and oppositely.
Near a body, co-located vessels share one local frame with one aggregate rigid
body per vessel. Outside it, each vessel advances independently through the
canonical propagator. Rendering, map view, save/load, contact, and warp all
project or operate on those same records.

## Alternatives

- **Keep detached stages as Bevy debris** — rejected because they would vanish
  or freeze across authority changes, warp, save/load, and map view, creating a
  second answer to "where is this object?"
- **Run full part-level physics for every vessel everywhere** — rejected because
  high warp and distant objects need cheap deterministic aggregate propagation.
  Structural topology persists, but detailed rigid bodies hydrate only in the
  local scene.
- **Simulate only the active craft and analytically fake other trajectories** —
  rejected because active selection would change physical truth and nearby
  collision could not be authoritative.
- **Keep one local bubble per craft** — rejected because two recently separated
  bodies must collide with the same terrain and each other in one coordinate
  frame. The local scene belongs to the dominant body; craft are occupants.

## Consequences

Single-craft APIs become temporary active-craft wrappers rather than the core
model. Part queries, runtime resources, authority records, maneuver plans, and
local-physics ownership must become vessel-scoped. The migration is larger than
retaining a debris entity, but it gives staging, switching, docking,
fragmentation, persistence, and future nearby vehicles one canonical path.

Deterministic iteration order and stable identity become explicit requirements.
The initial physical implementation may use one aggregate compound collider per
assembly; per-part deformation remains a later fidelity layer.
