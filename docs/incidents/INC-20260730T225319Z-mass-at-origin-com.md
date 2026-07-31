# INC-20260730T225319Z-mass-at-origin-com: every CoM aggregation weighted part mass at the part origin

- **Date:** 2026-07-30 · **Surface:** `just game runway` ground handling (tips over in ordinary turns); any craft's CoM/MoI

## Symptom

The Meridian flipped over its main gear in ordinary taxi turns. The user's
hypothesis was "wheel track too narrow" — true, but the dominant term was the
weight split: with CoM at y −9.8 m, nose gear at −5.6 and mains at −19.25, the
**nose carried 69 % of the weight**, pinching the tricycle tip axis to a 0.51 m
effective half-track under a 4.1 m-high CoM → static roll-over at **0.12 g**
lateral, while the tires grip up to μ = 0.8 g. It tipped at a sixth of the
lateral force the tires can transmit.

## Root cause

Every mass aggregation — `ShipBlueprint::stats` (spawn params), the runtime
`recompute_ship_inertia`, and the separation `aggregate_world_parts` — weighted
each part's mass at its **transform origin**, which for every axial part is its
*top/nose mating node* (`nodes_for`: body spans `[0, −length]` from the origin).
The 35 m fuselage loft's ~10 t was counted at its nose tip, ~16 m from its real
centroid. Corrected, the fueled CoM is y **−16.4 m**, not −9.8 — the
nose-heaviness was a phantom of the mass model, not the design. The same error
biased every rocket stack (each tank's mass a half-length high) and inflated
pitch/yaw MoI ~40 % via wrong parallel-axis arms.

Instructive dead end: an earlier live-BRP memory recorded "60/40 nose-heavy,
CoM −11.04" as a fact about the *craft*; it was a fact about the *bug*.

## Fix

Mirrored `part_centroid_offset` / `live_part_centroid_offset`
(`thalos_shipyard::stats`): centroid = half the **shared cylinder-model
length** (`part_cylinder_dims` / `live_part_cylinder_dims`) down the local
axis, so the centroid, the self-inertia, and both blueprint/live sides can
never disagree. Wings and gear return zero (wing sweep shift is a documented
second-order follow-up). All three aggregation sites apply it (runtime sites
rotate by the part's transform rotation).

With the mass model honest, the stance was then re-tuned to real numbers
(`meridian_balance` example prints the readout): mains station 0.55 → 0.51,
`gear_main_hd` track ±1.0 → ±2.2 · host radius with hull-anchored slanted
struts (`gear_mesh`) → 12 % nose / 88 % mains, roll-over threshold 0.78 g ≈
tire μ. Same change set fixed the tipped-state zombie (SAS power-slide +
frictionless angular hull contact) — see BL-20260730T225319Z-ground-stance-tipover.

## Recurrence signal

`cargo run -p thalos_shipyard --example meridian_balance` — the stance readout
prints the load split and roll-over threshold with a verdict line
("skids before it tips" / "TIPS BEFORE IT SKIDS"). A nose share drifting far
above ~25 %, or a threshold well below the tire μ, is the tell. The general
rule: any new mass consumer must use the centroid helpers, never a bare part
`Transform`/`geo.position`.
