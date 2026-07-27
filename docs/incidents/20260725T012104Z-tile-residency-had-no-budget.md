# INC-20260725T012104Z-tile-residency-had-no-budget: the ground could allocate VRAM without limit

- **Date:** 2026-07-25 · **Surface:** `just game launch`, ~90 s of repeated surface↔orbit flight

## Symptom

`DeviceLost error: Unknown Out of memory`, then a cascade of downstream panics that are
*consequences*, not causes — `ssao_uniform` / `contact_shadow_uniform` "Buffer is invalid",
`clustering metadata staging buffer` invalid, and finally a `SwapchainAcquireSemaphore ...
still in use` panic during teardown. Exit code `0xC0000409`.

Decisive evidence from the same log:

- `tile terrain: first full coverage (3081 tiles)` — an *ordinary* launch-pad framing
- `19,200 tiles landed` over ~85 s across ~7 surface↔orbit transitions
- the crash landed at t≈86 s with the ship at **1,966 km AGL** — a framing that selects
  coarse tiles and needs very few of them

## Root cause

The tile renderer had **no residency budget of any kind**. The only cap in `tiles/mod.rs` was
`MAX_IN_FLIGHT = 24`, which throttles *generation concurrency*, not how much geometry may be
resident. Residency was whatever the geometric selector asked for, spawned unconditionally.

One tile mesh costs **347 KiB** of VRAM (4,481 verts × 56 B across POSITION/NORMAL/COLOR/
UV_0/UV_1, plus 26,112 U32 indices). So:

| resident tiles | VRAM |
|---|---|
| 3,081 (the log's own first-coverage line) | 1.02 GiB |
| 7,752 (measured for the 22 km god view before `RUGGED_SPACING_FLOOR_M`) | 2.57 GiB |

`SPLIT_FACTOR` had recently gone 3.0 → 6.0 (4× tiles at a given distance) with
`SPLIT_FACTOR_RUGGED` at 18.0, so the unbudgeted ceiling had just risen sharply.

Stacked underneath it: the legacy udlod near-tier atlas is one texture array allocated
**upfront** (`depth_or_array_layers: atlas_size`, 384 slots × 4 attachments with 4 mips ≈
**890 MB per body**). This run allocated it for Thalos too, because `terrain_residency`
spawned at 01:02:52.612 and the tile driver only claimed the body at 01:02:53.840 — the boot
race that `ensure_tile_root` papered over with a *post-hoc* rebuild request.

**A wrong hypothesis worth recording**, because it is the obvious one and it is wrong: that
the churned meshes leak. They do not. `RenderAssetUsages::RENDER_WORLD` meshes are freed
correctly — bevy_render keys render-world removal on `AssetEvent::Unused` (last handle
dropped), not on main-world removal — and the slab allocator does track and free empty slabs.
Despawning a tile does release its VRAM.

**What the log does not settle**, and could not: whether the working set alone explains the
crash. Peak residency was near the surface, minutes *before* the crash; the crash came at a
cheap framing after sustained churn. That shape suggests memory that did not come back, on
top of a working set already in the 1–3 GiB range. Two candidates remain live — mesh-allocator
slabs pinned by stragglers (a slab returns to the driver only when *every* allocation in it is
freed, and slabs run to 512 MiB), and per-visit scatter rebuilds. The reason this was
undecidable is itself a finding: the tile log reported only a **cumulative** landing counter,
so a 19,200-landing session was equally consistent with a 1 GiB working set and a 6 GiB one.

## Fix

1. **A gauge, first** — `tile terrain residency:` every 5 s with resident count, MiB, pending,
   desired, retiring, and the split scale, plus `tile_resident` / `tile_mib` /
   `tile_split_scale` in `mem_diag`'s JSONL. Periodic rather than per-landing so a *settled*
   scene still reports its footprint.
2. **A byte-denominated budget** (`TILE_MESH_BYTES`, default 4 GiB, `THALOS_TILE_BUDGET_MB`,
   `0` disables). Denominated in bytes for the reason `rendering::tile_cache`'s payload budget
   already records: a tile-*count* cap looks harmless and silently means gigabytes. Set above
   every working set we have actually measured — the hungriest is NTR-X12's 10,900-tile 30 m
   Mira descent (3.6 GiB) — so the brake cannot silently coarsen a framing that was already
   capture-verified. It is a runaway brake, not a quality cap; the gauge is what will tell us
   where it can be tightened.
3. **The budget bites in selection, never eviction.** The despawn rule is hole-free by
   construction — a tile may only leave once its footprint is served by other resident tiles —
   so evicting resident tiles would punch holes in the ground, and blocking *admission* would
   deadlock (nothing can retire until its replacement lands). Coarsening desire instead makes
   the coarse ancestor desired, and once it lands the fine tiles retire through the normal
   merge certificate, with no hole at any instant.
4. **The legacy atlas is no longer allocated and thrown away.** `try_spawn` holds the
   *dominant* body's near-tier udlod spawn while `tile_claim_pending()`, self-releasing after
   10 s so an anchor that never resolves degrades to legacy terrain instead of no ground.

Consequence of (3) to keep in mind: satisfying the brake costs a transient *increase* (the
ancestor lands before its children leave, ≤ ~1/3 pyramid overhead), which is why the budget
must sit below real VRAM with headroom rather than at it.

## Recurrence signal

`tile terrain residency:` is the line to read. `split scale` below 1.00 means the brake is
holding detail back — expected during a fast descent, a standing signal that the budget is too
low (or residency too hungry) if it never recovers to 1.00 in level flight.

Reading it *against* `mem_diag`'s `render_meshes` / `wgpu_buffers` is what separates the
remaining candidates: tile MiB plateauing while those keep climbing means the ground is inside
its budget and the growth is elsewhere.

**Standing rule for this renderer:** any new per-tile GPU resource joins `TILE_MESH_BYTES`, or
the budget silently under-counts — and an under-counting budget is worse than none, because it
reports headroom that does not exist. `tile_mesh_bytes_matches_the_built_mesh` asserts the
denominator against a really-built mesh so an added attribute fails the test instead of the
GPU (NTR-RT1 wants TANGENT — that is the next one).

## Recurrence 2026-07-25 20:08 UTC — the budget was per *process*

It happened again, ~19 h later, with the fix above in the tree. The diagnostics settle what the
first log could not, and the answer was not the accumulation hypothesis:

**Two full game instances were live.** `runtime.jsonl` carries interleaved sessions —
pid 10560 (19:53:20 → 20:08:34) and pid 8376 (19:57:48 → 20:08:37), overlapping ~10.8 min and
**dying three seconds apart**, which is the card giving out rather than one process
misbehaving. Neither ever braked: `split_scale` was 1.00 in both, correctly, because each was
individually far inside its own 4 GiB (735 MiB / 556 MiB of tiles at death). Jointly they were
entitled to 8 GiB of tile meshes alone on a 12 GB card. An earlier pair the same day peaked at
**3.57 GiB of tiles concurrently**.

So the budget worked exactly as specified and the specification was wrong: *a budget every
participant reads as if it were alone is not a budget.* Fixed by making
`DEFAULT_RESIDENCY_BUDGET_BYTES` a **machine-wide** figure divided by the live renderer count
(`tiles::vram_share`: mtime heartbeats under the temp dir, stale after 10 s — timestamps, not
locks, because the OOM ends in an *abort* that runs no destructor). One instance is bit-identical
to before, so no capture-verified framing moves; two instances get half each.

Also confirmed live in both processes: **BL-20260725T012104Z**, the ~890 MB Pelagos near-tier
atlas pulled for a body neither session visited — ~1.7 GB of pure waste across the pair. Its
cause is now fixed at the source: `compute_wanted_residency`'s prediction rule is gated on
`placement_settled`, because a deferred-placement scenario (`launch`, the runway family,
descents) boots on a debug *parking orbit* placeholder whose predicted encounters are fiction.
Note the debounce could never have saved it — `despawn_debounce_s` counts **sim** time, which
does not advance at the warp 0× every scenario starts at.

**What this does not settle.** Tiles were 1.29 GiB of ~12 GB at the moment of death, so tile
residency was *not* the dominant consumer either time. The `THALOS_MEM_DIAG=1` +
`--features gpu-counters` differential is still the open work: tile MiB plateauing while
`render_meshes` / `wgpu_buffers` climb means the growth is elsewhere.

**Second recurrence signal, added:** the gauge now reports `instances`. A budget that suddenly
halves is a peer starting, not the brake misbehaving.
