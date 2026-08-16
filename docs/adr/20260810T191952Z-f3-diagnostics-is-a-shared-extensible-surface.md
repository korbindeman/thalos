# ADR-20260810T191952Z-f3-diagnostics-is-a-shared-extensible-surface: F3 diagnostics is one shared surface with application extensions

- **Status:** Accepted
- **Date:** 2026-08-10

## Context

Thalos already had a mature F3 performance surface: one toggle for the live
panel, frame and GPU history, device and memory facts, application state, and
game-specific debug drawings. When Kòrsou gained F3 it reproduced the panel,
toggle, frame/GPU extraction, device formatting, and refresh lifecycle around
different terrain and position fields. The second implementation proved both
the common mechanism and the application-specific seam.

Keeping both panels would make the same player operation drift. Moving the
whole game overlay into the lightweight application runtime would be worse:
simulation stages, planetary terrain, craft state, aero gizmos, and capture
scenario types must not become dependencies of Kòrsou.

The existing `thalos_diagnostics` foundation is intentionally Bevy-free. It
owns the process event/sink contract and is used by offline tools, so adding UI
or renderer dependencies there would collapse a useful lower boundary.

## Decision

F3 diagnostics is one extensible Bevy surface in a new
`thalos_diagnostics_ui` interface crate. It depends downward on Bevy,
`thalos_ui`, and the Bevy-free `thalos_diagnostics` foundation. The lightweight
`thalos_runtime[interactive]` bundle exposes it without acquiring simulation,
gameplay, planetary, or capture dependencies.

The shared core owns:

- the F3 intent, requested-open state, application-supplied availability gate,
  root visibility, and stable system ordering;
- one wall-clock CPU/GPU frame-history resource, common summary statistics,
  renderer/window and process facts, and the common graph;
- the panel shell, common sections, visual language, refresh cadence, and an
  explicit extension root.

Application adapter plugins add typed ECS components beneath the extension
root and update only their own fields. Thalos retains simulation-stage timing,
warp, terrain/memory budgets, body/landcover position, hitbox and aero-gizmo
side effects, and its request-scoped capture override. Kòrsou retains planar
terrain/foliage streaming and UTM/AGL position. Each application supplies the
single availability-gate writer that folds its modal/photo-mode state into the
shared panel.

The extension seam is ECS composition, not a string-keyed metric registry:
the core exports marker components, spawn helpers, public state, and ordered
system sets. Extensions spawn normal Bevy UI children and own normal systems.
The game perf recorder reads the shared frame history, so F3 and
`runtime.jsonl` continue to use one frame authority without changing the
diagnostic event schema.

## Alternatives

- **Keep two panels** — rejected because F3 is one user operation and the
  duplicated toggle, sampling, layout, and formatting had already begun to
  differ before Kòrsou's first live verification.
- **Share only panel styling** — rejected because it leaves the correctness
  sensitive frame/GPU sampling and visibility lifecycle duplicated.
- **Move the game overlay unchanged into the shared runtime** — rejected
  because its simulation, planetary, debug-gizmo, and capture dependencies
  violate the lightweight application boundary.
- **Register arbitrary named metrics and formatter closures** — rejected as a
  service-container abstraction over Bevy's existing typed ECS composition.
- **Put the Bevy panel in `thalos_diagnostics`** — rejected because offline
  tools must retain a Bevy-free diagnostics dependency.

## Consequences

- Both applications have the same common F3 panel, graph, input, sampling, and
  behavior while displaying facts meaningful to their own world model.
- A new application gets useful diagnostics by selecting the lightweight
  interactive capability and supplying only its extension sections and gate.
- Thalos's frame-history ownership moves below the game runtime; its periodic
  event schema and deeper performance gauges remain game-owned.
- The common panel can be verified in one headless preview. Application
  extensions still require their own deterministic data and live integration
  checks.
- The capability guard must reject any game-runtime dependency from
  `thalos_diagnostics_ui` and Kòrsou.
