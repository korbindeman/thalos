# Incident post-mortems

Git-committed write-ups of **fixed** non-obvious bugs — visual, behavioral, crash, or perf —
so future agents can answer "why did that break?" without re-deriving the diagnosis. ADRs
capture design *decisions*; these capture *incident* forensics: symptoms, evidence, the
hypothesis differential, root cause, fix, and how to spot a recurrence.

Not every bug earns one. Write a post-mortem when the diagnosis was non-obvious (the CLAUDE.md
bug-fixing loop ran a real differential), when the root cause teaches a standing invariant, or
when the same class could plausibly recur. A typo-grade fix doesn't need one.

## Workflow

1. **Diagnose first** (CLAUDE.md "Bug fixing"): hypothesis set → targeted falsifiable tests →
   agreed root cause. Search here for a matching prior (`rg 'RenderOrigin|jitter|<symptom>'
   docs/incidents/`) before re-deriving.
2. **Fix the mechanism**, not the symptom.
3. **Write the post-mortem in the same change**: copy `0000-template.md` to the next
   `NNNN-kebab-title.md`, fill every section (especially **Evidence**, **Hypotheses
   considered**, and **Prevention**), add a row to the index below. If the lesson is a standing
   rule, add/extend the matching CLAUDE.md gotcha or spec-doc invariant and link it from
   **Prevention**.
4. **Reference later**: cite `INC-NNNN` in discussion and backlog rows.

Historical incidents migrated from auto-memory keep their original dates.

## Index

| INC | Title | Severity | Date |
|-----|-------|----------|------|
| [0001](0001-render-origin-god-view-shadow-desync.md) | God-view shadows vanished — `RenderOrigin` tracked the focus target, not the camera | visual | 2026-07-05 |
| [0002](0002-terrain-covers-structures-lod-flatten.md) | Terrain intermittently covered / z-fought the space center — LOD height error vs the flatten plane | visual | 2026-07-18 |
| [0003](0003-orbital-black-continents-coast-speckle.md) | Black continents from orbit + dotted land-through-water coast speckle | visual | 2026-07-19 |
| [0004](0004-coarse-lod-mask-step-clamp-grey-shiny.md) | Distant terrain grey/shiny/pixelated — mask stencil step clamped below coarse texel spacing | visual | 2026-07-20 |
| [0005](0005-smoothstep-epsilon-guard-inverted-forest.md) | Forest painted onto the driest ground — `.max(EPSILON)` guard inverted descending-edge smoothsteps | visual | 2026-07-20 |
| [0006](0006-parallel-rustc-poisoned-incremental-link.md) | Experimental parallel rustc ICE poisoned incremental objects and broke the next link | perf | 2026-07-20 |
| [0007](0007-atmosphere-proxy-omitted-camera-render-offset.md) | Raymarched atmosphere detached from the planet — proxy omitted the camera render offset | visual | 2026-07-20 |
| [0008](0008-direct-dynamic-game-launch-missed-library-path.md) | Direct dynamic game launch missed Cargo's library search path | crash | 2026-07-21 |
| [0009](0009-mira-horizon-regolith-normal-aliasing.md) | Mira's opaque horizon looked transparent — unresolved regolith normals aliased through Hapke | visual | 2026-07-21 |
| [0010](0010-cloud-detail-period-eighth-scale.md) | Cloud detail collapsed into stipple — authored cell size was treated as an eight-cell tile period | visual | 2026-07-21 |
| [0011](0011-cloud-hierarchy-resume-strata.md) | Cloud hierarchy posterized volumes — heuristic leaps resumed on repeated height isosurfaces | visual | 2026-07-21 |
| [0012](0012-ocean-gradient-worms-isotropic-detail-loss.md) | Ocean gradient worms + premature isotropic detail loss | visual | 2026-07-21 |
