# INC-20260817T003451Z: Surface terrain waited on a parking-orbit tile fill

- **Date:** 2026-08-17 · **Surface:** every `just game launch` / `runway` boot (the 4-step loading screen)

## Symptom

The loading screen sits on **Surface terrain** (step 2 / 4) for a long time on every boot, then the world appears at the pad. Runtime frame rate after reveal is unrelated — the wait is loading work that never reaches the first visible frame.

## Root cause

Deferred placement seeds a debug parking orbit behind the loading screen and only installs the pad once a height source exists. The tile streamer follows `TileEye` from the first frame the root is up, which is **one frame before** `finish_runway_spawn` can see that height source. That first stream freezes a cold-start bootstrap for the orbital camera (a whole-planet cover, often thousands of 129² tiles). Until `coverage_ready`, the streamer admits **only** that frozen set — even after the craft has sat down — so the loading gate pays for tiles the pad drop discards.

The tell is a `first_coverage` / `initial coverage ready` line whose `resident` count looks like an orbital working set, followed by the camera already being on the pad.

## Fix

- Hold `TileEye` while `step::PLACEMENT` is registered and incomplete, so the first tiles stream at the pad after the flatten exists.
- Discard a frozen bootstrap if the eye teleports (~200 km parking-orbit → pad) before first cover.
- On boots that also register `step::SETTLE`, complete `Surface terrain` once the tile root exists. Settle waits for `coverage_ready` of the **pad** view (needed for the impostor handoff) and for 50 m/texel under the camera; further splits past that floor no longer reset the hold.

Orbit boots are unchanged: no placement step, so the eye publishes immediately and the terrain step still waits on first cover of the real view.

## Recurrence signal

A launch/runway boot whose `tile terrain: initial coverage ready` line names thousands of tiles, or whose Surface terrain step outlasts `first full coverage` of a later, much smaller pad set. If a readiness predicate is added for boot ground, it must name **which camera** the cover is for.
