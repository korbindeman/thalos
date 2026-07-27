# INC-20260726T040431Z-grass-under-pavement: blades grew under the connection drape and came up through it

- **Date:** 2026-07-26 · **Surface:** eye level anywhere on the base's paved network (`just screenshot paved-ground`)

## Symptom

Grass tufts scattered across every taxiway, apron and service road, each one sunk to its
tips — the blades read as growing *below* the tarmac rather than on it. Reported from a live
session; reproduced headlessly at 13 m AGL, where the pavement measured **3.96% dark specks**
against a smooth surface.

The runway itself was clean, which is the tell: runways are placed structures, connections
are generated ones.

## Root cause

One mechanism, both halves of the symptom.

Connections are drawn as a drape lifted `CONNECTION_LIFT_BASE_M` (0.12 m) above the flattened
ground. Nothing published that pavement to the scatter layers: `grass_scatter_regions` builds
its clear/lawn footprints from `StructureRegistry::sites_on`, and `site_scatter_region` matches
only `BaseSite` / `Runway` / `Building` / `Launchpad` / `Tank`. Taxiways, aprons, roads and
crawlerways are *generated* from the site's structure set by `base_editor::connections`, never
registered as sites — so `classify_scatter` returned `Natural` over all of them and the GPU
grass field grew a full sward underneath. The 12 cm of drape then hid the lower part of every
blade, leaving only tips: "grass below the surface".

The lift constant's own doc comment records the previous encounter with this — the lift was
*raised* from ~4 cm to 0.12 m in part because "short lawn grass poked through". That is a race
the drape cannot win: a taller blade, a lower lift or a bumpier pad brings the tufts straight
back, and the lift is bounded above by how far pavement can float before it reads as a lip.
It also treats the symptom while the blades keep being placed, shaded and drawn.

## Fix

Every connection now publishes the footprint it paves, in the same call that builds its mesh.
`PavedFootprints` (in `base_editor::connections`, single writer `spawn_connection_entity`) holds
per-site `ScatterRegion`s with `ScatterTreatment::Clear`; `grass_scatter_regions` chains them
onto the placed-structure regions, so the grass layers clear pavement exactly as they already
cleared buildings. Rectangles come off the same centreline and the same width the mesh is
extruded from — mesh and footprint cannot drift apart — and a rebuild drops the site's old
footprints alongside its old mesh, so a moved taxiway stops clearing its former route.

This removes the class: a future `ConnectionKind` (a rail spur, a pipe run) gets its clearing
for free, because the funnel every spawner already goes through is the thing that records it.

Measured on the `paved-ground` framing, matched before/after: pavement dark specks
3.96% → 1.43% (the residual is the grass verge inside the measurement box; the tarmac itself
is visibly clean). The verge between pavement and runway is preserved — the clear margin is
0.6 m plus a 1 m ramp, so grass meets the tarmac without standing in it.

## Recurrence signal

Any speck of vegetation on tarmac in `just screenshot paved-ground`. If it returns, check
first whether a new paved thing bypassed `spawn_connection_entity` — that funnel is what makes
the footprint mandatory. Do **not** reach for `CONNECTION_LIFT_BASE_M`: raising the drape hides
blades without removing them, and that is what deferred this defect the first time.
