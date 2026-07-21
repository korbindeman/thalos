# Docs

The project's design and planning docs. Start here to find the right one.

## Steering (agent-maintained)

| Doc | What it is | Altitude |
|-----|------------|----------|
| [backlog.md](backlog.md) | **Execution** — the rolling, status-tracked queue of concrete scoped items across the active sprints. The *what's next*. Driven by the `steer` skill. | Work queue |
| [architecture_cleanup.md](architecture_cleanup.md) | **Primary sprint plan** — the consolidation pass: packages A–G, the audit baseline, rules of engagement. | Strategy |
| [graphics_fidelity.md](graphics_fidelity.md) | **Secondary sprint plan** — one-world principle, the F1–F9 unification foundation, per-substrate workstreams (§4), open questions (§7). | Strategy |
| [gameplay.md](gameplay.md) | **Horizon vision** — the core loop, the setback-not-failure consequence economy, the diegetic control-authority ladder, the uncrewed-first opening arc, vehicle families and heritage, probes, contracts. Design capture; **no active sprint, no backlog rows** — kept at doc granularity until a track opens. | Vision (horizon) |
| [adr/](adr/) | **Decision log** — numbered Architecture Decision Records: *why* a choice was made and what was rejected. | Rationale |
| [incidents/](incidents/) | **Incident post-mortems** — numbered forensics for *fixed* non-obvious bugs: evidence, root cause, prevention, recurrence signals. | Incidents |

## System specs (one per major system)

| Doc | System |
|-----|--------|
| [architecture.md](architecture.md) | Target workspace and dependency boundaries: apps, runtime, domain/simulation/rendering libraries, tools, labs, and artifacts |
| [simulation.md](simulation.md) | Orbital mechanics, authority, warp, map decoupling, big_space, local bubble |
| [regimes.md](regimes.md) | Per-craft `CraftRegime` resolver — the one classification record |
| [surface_local.md](surface_local.md) | The SLF: body-fixed tangent-frame near-surface physics (§10 = shipped-vs-design) |
| [surface.md](surface.md) | Surface gameplay: EVA on foot; landing & impact destruction |
| [aerodynamics.md](aerodynamics.md) | Atmospheric flight forces: whole-body aero model, flight config, control authority |
| [control.md](control.md) | Fly-by-wire layer: demand vocabulary, one attitude controller, effector allocation |
| [boot.md](boot.md) | Boot pipeline: `AppState`, `LoadingTracker`, start screen, deferred placements |
| [ui_flow.md](ui_flow.md) | `GameContext` sub-state — the in-game mode authority + migration phases |
| [ui.md](ui.md) | The `thalos_ui` kit: tokens, glass panels, widgets, kitchen-sink loop |
| [hud_widgets.md](hud_widgets.md) | The MFD slot: contextual HUD widget framework |
| [input.md](input.md) | Enhanced-input contexts, binding files, intent resources |
| [construction.md](construction.md) | Next-gen shipyard construction model (Module primitive; M6 target) |
| [base_building.md](base_building.md) | In-world surface base editor, structures, ground scatter |
| [terrain.md](terrain.md) | Consumer-side terrain contract: the tile primitive, LOD, colliders |
| [terrain_macro.md](terrain_macro.md) | Large-scale terrain: scale ownership, macro landcover phases |
| [mira_airless_mvp.md](mira_airless_mvp.md) | Mira airless-terrain vertical slice: runtime surface, crater bands, regolith, verification |
| [terrain_lod_optimization.md](terrain_lod_optimization.md) | The udlod fork's Thalos optimization pass: tile cache, mips, admission |
| [vegetation.md](vegetation.md) | Planet-scale vegetation plan: cascades, placement, impostors, instancing |
| [atmosphere.md](atmosphere.md) | Atmosphere/cloud/ocean *rendering*; IBL/reflection probe |
| [ocean.md](ocean.md) | Ocean rendering: shipped analytic-surface slice, invariants, verification, and spectral/local-simulation path |
| [clouds.md](clouds.md) | Planet-scale volumetric cloud program: canonical weather, density, temporal reconstruction, lighting, shadows, and orbital LOD |
| [celestial.md](celestial.md) | Procedural sky model: sources, spectra, cubemap bake |
| [physics.md](physics.md) | The owned-solver revamp (`thalos_physics`, TGS-Soft) — see ADR-20260720T185956Z-replace-avian-owned-tgs-soft-solver |
| [shadow_unification_prompt.md](shadow_unification_prompt.md) | F6 one-shadow-world status block + tuning knobs |
| [planetary_rendering_baseline.md](planetary_rendering_baseline.md) | Rendering baseline reference |
| [tooling.md](tooling.md) | Toolchain policy; local-only compiler tuning recipes |
| [build_speed.md](build_speed.md) | Cross-platform build acceleration, sccache, and the per-environment agent build workflow |
| [visual_testing.md](visual_testing.md) | Deterministic headless screenshot A/B and multi-test workflow |
| [capture.md](capture.md) | Agent-first capture architecture: shared runtime, persistent/cold stills, comparisons, and deterministic video |
| [lore/](lore/) | Solar-system reference + civilization narrative |
| [archive/](archive/) | Superseded terrain-*generation* design — reference only, see its README |

`CLAUDE.md` (repo root) is the operating manual for agents; the `steer` skill
(`.claude/skills/steer/SKILL.md`) routes "what's next?" between the plan docs and the backlog.

## Cross-reference convention

Docs cite each other with a short prefix so a ref is unambiguous from any doc:

| Prefix | Points to |
|--------|-----------|
| bare `§N` | a section of **the current document** |
| `clean §N` | [architecture_cleanup.md](architecture_cleanup.md) |
| `gfx §N` | [graphics_fidelity.md](graphics_fidelity.md) |
| `cloud §N` | [clouds.md](clouds.md) |
| `ADR-YYYYMMDDTHHMMSSZ-slug` | [adr/](adr/) |
| `INC-NNNN` | [incidents/](incidents/) |
| anything else | by filename (`ui_flow.md §3`) |
