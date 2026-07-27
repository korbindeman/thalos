# INC-20260726T013031Z-loading-gate-waited-on-a-renderer-that-stood-down: every boot paid the 120 s loading backstop

- **Date:** 2026-07-26 · **Surface:** every `just game <scenario>` boot on the default tile renderer

## Symptom

```
01:23:29  surface terrain settled (lod 4.8 m/texel, 12 stable frames, 1.0 s) — revealing
01:23:42  tile terrain: first full coverage (3957 tiles) — impostor handoff ready
01:25:21  WARN loading hard-timeout after 120 s — revealing with step 'Surface terrain' incomplete
```

The ground was fully streamed at **t+21 s**. The loading screen stayed up until the
`LOADING_HARD_TIMEOUT_S` backstop fired at **t+120 s** and revealed the world with the step
still marked incomplete. Silent apart from that one warning, and easy to read as "terrain
streaming is slow" rather than "the gate can never pass".

## Root cause

`initial_residency_loading_gate` completes `step::TERRAIN` once every `Near`-wanted body is
either udlod-**resident** or has no authored terrain. Neither is ever true for the body the tile
renderer owns: NTR-X1 made `try_spawn` stand the legacy udlod stack down for tile-rendered
bodies, so `BodyTerrainResidency` never gains an entry for it — while `Simulation::dominant_body`
keeps it permanently in the wanted set at `Near`. The gate was therefore waiting on a condition
the renderer it was written against had been deliberately retired from producing.

The tell is the ordering in the log above: a `first full coverage` line *before* a
`loading hard-timeout` naming `Surface terrain` means the gate, not the streaming.

This is the ordinary cost of a partial renderer migration — the two-renderer window means every
readiness predicate has to name which renderer it is asking about. `surface_settle`'s
`step::SETTLE` already reads the tile path directly, which is why the runway/launch scenarios
still felt like they loaded and made the stall look scenario-specific rather than universal.

## Fix

The gate accepts the tile path's own handoff criterion for bodies it owns:
`tile_terrain::tile_rendered(body)` plus `TileTerrainRoot::coverage_ready()` — the same
"desired selection has been fully resident at least once" test that gates the impostor↔terrain
handoff, and the exact analogue of udlod's `pinned_tiles_ready`.

## Recurrence signal

A `loading hard-timeout after 120 s` warning naming a step whose subsystem already logged its
own readiness line earlier in the same boot. When a readiness predicate is added for the ground,
it must name the renderer: `is_resident` answers only for udlod, `coverage_ready` only for tiles.
