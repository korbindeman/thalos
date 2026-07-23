# Documentation

Thalos keeps only its highest-altitude documents at the `docs/` root. Everything
else is grouped by role or primary subsystem; this file is the canonical map.

## Start here

| Document | Authority |
|----------|-----------|
| [Backlog](backlog.md) | **Execution:** the status-tracked queue and the answer to “what’s next?” |
| [Architecture](architecture.md) | **Codebase:** workspace layout, ownership, and dependency boundaries |
| [Gameplay](gameplay.md) | **Product:** pillars, core loop, progression, and long-horizon player experience |
| [Documentation map](README.md) | **Navigation:** where each kind of knowledge belongs |

Use the [roadmap](roadmap/) for active sprint strategy, a subsystem spec for the
current design, an [ADR](adr/) for why a choice was made, an
[incident](incidents/) for fixed-bug forensics, and the [archive](archive/) only
for superseded reference material.

## Categories

### [Roadmap](roadmap/)

Active strategy and sequencing, not implementation detail.

- [Neural terrain × standard-path renderer](roadmap/neural_terrain_renderer.md) — **keystone / primary sprint**: paired diffusion terrain + Bevy-standard-path renderer, probe milestones, extraction plan.
- [Architecture cleanup](roadmap/architecture_cleanup.md) — background consolidation sprint (demoted from primary 2026-07-23).
- [Graphics fidelity](roadmap/graphics_fidelity.md) — one-world rendering strategy; spine-port items frozen by the keystone triage, composites continue.
- [Mira learned terrain](roadmap/mira_learned_terrain.md) — L1–L6 milestones, visual gates, dataset growth, compute ledger, evidence; paused after L2 closure per the keystone.

### [Gameplay systems](gameplay/)

Player-facing flows, editors, controls, and interface contracts.

- [Base building](gameplay/base_building.md)
- [Boot and start screen](gameplay/boot.md)
- [Construction](gameplay/construction.md)
- [HUD widgets](gameplay/hud_widgets.md)
- [Input](gameplay/input.md)
- [UI flow and `GameContext`](gameplay/ui_flow.md)
- [UI kit](gameplay/ui.md)

### [Simulation](simulation/)

Vehicle dynamics, authority, physics, and near-surface behavior.

- [Simulation core](simulation/simulation.md)
- [Craft regimes](simulation/regimes.md)
- [Surface-local frame](simulation/surface_local.md)
- [Surface gameplay and impacts](simulation/surface.md)
- [Aerodynamics](simulation/aerodynamics.md)
- [Fly-by-wire control](simulation/control.md)
- [Owned physics solver](simulation/physics.md)

### [World](world/)

Celestial content and the physical surface of bodies.

- [Celestial sphere](world/celestial.md)
- [Terrain contract](world/terrain.md)
- [Macro terrain and biomes](world/terrain_macro.md)
- [Mira airless terrain MVP](world/mira_airless_mvp.md)
- [Vegetation](world/vegetation.md)

### [Rendering](rendering/)

Atmospheric and surface-adjacent rendering systems.

- [Atmospheres, oceans, and lighting](rendering/atmosphere.md)
- [Clouds](rendering/clouds.md)
- [Ocean](rendering/ocean.md)
- [Rocket plumes](rendering/plume.md)

### [Development](development/)

How to build, inspect, capture, and verify the project.

- [Build speed and agent workflow](development/build_speed.md)
- [Capture architecture](development/capture.md)
- [Tooling](development/tooling.md)
- [Visual testing](development/visual_testing.md)

### [Reference](reference/)

Dated baselines, completed work orders, and focused audits. These support the
canonical specs above; they do not replace them.

- [Cloud baseline](reference/cloud_baseline.md)
- [Planetary rendering baseline](reference/planetary_rendering_baseline.md)
- [Shadow-unification work order](reference/shadow_unification_prompt.md)
- [Terrain LOD optimization audit](reference/terrain_lod_optimization.md)

### Durable records and setting

- [Architecture decisions](adr/) — accepted choices and rejected alternatives.
- [Incident post-mortems](incidents/) — evidence, root cause, fix, and recurrence signals.
- [Lore](lore/) — solar-system and civilization references.
- [Archive](archive/) — explicitly superseded design material.

`CLAUDE.md` at the repository root is the agent operating manual. The `steer`
skill (`.claude/skills/steer/SKILL.md`) routes new work between the roadmap and
the backlog.

## Placement rules

- New docs go in the category that owns their primary responsibility; do not
  add another root document by default.
- Keep the hierarchy one level deep. A subsystem with one spec does not need a
  one-file directory.
- Keep one authority per fact. Reference docs may capture evidence or a dated
  snapshot, but the live contract stays in the subsystem spec.
- Do not add category README files. Update this map when a document is added,
  moved, archived, or made canonical.
- Use repository-relative paths in prose (`docs/rendering/clouds.md`) and links
  relative to the containing Markdown file.

## Cross-reference convention

| Prefix | Points to |
|--------|-----------|
| bare `§N` | a section of the current document |
| `ntr §N` | [neural terrain × standard-path renderer](roadmap/neural_terrain_renderer.md) |
| `clean §N` | [architecture cleanup](roadmap/architecture_cleanup.md) |
| `gfx §N` | [graphics fidelity](roadmap/graphics_fidelity.md) |
| `cloud §N` | [clouds](rendering/clouds.md) |
| `ADR-YYYYMMDDTHHMMSSZ-slug` | [architecture decisions](adr/) |
| `INC-YYYYMMDDTHHMMSSZ-slug` | [incidents](incidents/), plus frozen legacy `INC-NNNN` |
| anything else | its repository-relative path or unambiguous filename plus section |

## Identity convention

Every record an agent mints—ADR, incident post-mortem, or backlog item—uses
`<KIND>-<YYYYMMDDTHHMMSSZ>-<kebab-slug>`, stamped with the current UTC time. A
record filename drops the kind prefix so lexical order is chronological; record
directories do not keep hand-maintained indexes.

Never allocate “the next number.” `INC-0001`–`INC-0021` and `BL-1`–`BL-40` are
frozen legacy identifiers: they remain valid but are never extended or
renumbered. Plan-owned identifiers such as `CL-A`, `F1`, and `W12` are
unaffected because one document owns their namespace.

See ADR-20260722T170714Z-one-chronological-identity-rule and
ADR-20260721T034338Z-distributed-chronological-identifiers.
