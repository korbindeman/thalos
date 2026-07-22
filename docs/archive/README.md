# Archived terrain-generation docs

These documents describe **superseded terrain-*generation* internals**. They
were archived (2026-06) when the project reframed generation as a **black box
behind the tile contract**: from every consumer's perspective (renderer,
collider, shadows, dynamic features), generation is just "something that
produces tiles conforming to the tile contract in
[terrain](../world/terrain.md)."
The new generator is being built fresh against that contract.

They are kept as **reference**, not as live specs. Do not treat anything here
as the current design. The canonical, current terrain docs are:

- [Terrain](../world/terrain.md) — the tile contract, ground-LOD rendering,
  surface shadows, colliders, and dynamic features (the consumer side).
- [Atmosphere and lighting](../rendering/atmosphere.md) — atmospheric optics, ocean, clouds,
  reflections, and lighting/GI.

## What's here

- `planet-generation-method.md` — old authoring workflow.
- `terrain-generation-cascade.md` — old semantic layer/cascade model.
- `planet-generation-pipeline-spec.md` — old field-DAG target architecture.
- `planet-generation-pipeline-migration.md` — old brownfield migration sequencing.
- `terrain-generation-legacy.md` — the generation ("feature compiler") chapter
  extracted from the old unified `terrain.md`, plus its v2 backlog.
- `gen/` — research surveys, aesthetic targets, and per-body process notes
  (`dunes.md`, `planet_aesthetics.md`, `terrestrial_pipeline_research.md`,
  `vaelen_processes.md`). `planet_aesthetics.md` in particular still captures
  visual targets the new generator should aim at — mine it, don't follow its
  pipeline framing.

Legacy producer source comments point here explicitly so a historical reference
cannot be mistaken for the live world specification.
