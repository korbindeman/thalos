# Terrain LOD (udlod) optimization — tailoring the fork to Thalos

`thalos_udlod` began as `kurtkuehnert/bevy_terrain`, whose design target is
**preprocessed real-world raster data** (GeoTIFF → on-disk tile pyramids). Thalos
replaced that fixed format with an owned `SurfaceQuery` contract between producer
and renderer. The current producer synthesizes at runtime; ADR-20260720T211046Z-offline-terrain-packages plans a new
adaptive offline terrain package as another producer. In both cases Thalos knows
the full content identity, so reconstructed tiles are safely memoizable. This
document records the optimization pass that spends that leverage, and what it
deliberately did not do.

## What the fork had dropped vs upstream (audit result)

Nothing gameplay-relevant. The removed pieces were all preprocessing: GeoTIFF/TIFF
ingest (`formats/`), the Split→Downsample→Stitch bake (`preprocess/`), `config.tc`
— and **`DiskTileProvider`**, which was the only persistence mechanism. Its
intended replacement, `MemoryTileCacheProvider`, existed but was **never wired
in**, so in practice the fork had *no* tile caching at all: every atlas eviction,
residency-tier swap, and flatten rebuild re-synthesized tiles from scratch (a cold
surface site costs ~15 s of field evaluation).

Upstream's post-2024 work moved to a **new repo**,
[`planetary_terrain_renderer`](https://github.com/kurtkuehnert/planetary_terrain_renderer)
(WGS84 ellipsoid, multi-dataset overlay, CLI preprocessing). Nothing there is
needed here, but it is the better diff target for pulling fixes to the
Taylor-series precision path, which upstream kept improving.

## What landed

### 1. Caching — the headline

```text
MemoryTileCacheProvider   ← survives terrain despawn/respawn (handle held in a resource)
  └── DiskTileCacheProvider   ← survives the process
        └── PipelineTileProvider   ← actually evaluates the surface
```

- **`SharedTileCache`** extracts the in-memory frecency cache behind an
  `Arc<Mutex<…>>` handle the game holds **per body, outside the `TileAtlas`**.
  That placement is the whole point: `TerrainRebuildRequest` despawns and respawns
  a body's entire terrain to apply a flatten, and the cache used to die with it.
  Budgeted in **bytes, not tiles** (`rendering::tile_cache`) — a near-tier payload
  is ~5 MB, so a tile-count cap silently becomes gigabytes.
- **`DiskTileCacheProvider`** — write-through memoization, one file per tile,
  storing the **full mip chain** exactly as the atlas uploads it. Format `TLC1`,
  little-endian and alignment-safe by construction (see below). Reads are
  validated against the requested attachment layout — wrong count, wrong format
  tag, or wrong mip-chain length is treated as a *miss*, so a truncated or
  format-drifted file can never feed a malformed upload to the GPU.
  `prune_tile_cache` caps total size at boot.

**The cache key is the design.** Everything that changes a tile's content is
folded into one namespace (`rendering::tile_cache::namespace_fn`): the generator
fingerprint (`thalos_terrain::GENERATOR_VERSION`), the body, the model scale, and
the body's terrain-flatten state. Package-backed terrain extends the static half
with package content hash, reconstruction version, and quality tier; the shipped
package remains authored source data, not part of this disposable cache.

The subtle part — and the bug that nearly shipped — is that the namespace is
resolved **per request**, not frozen when the provider is built. The flatten
handle is read *per tile pixel* during synthesis, so a pad installed *after* the
terrain entity spawned (exactly what `build_spaceport` does) still changes what
later tiles bake. A construction-time snapshot would have filed those flattened
tiles under the *un*-flattened key and served them as pristine terrain next
session. `NamespaceFn` (a `Fn() -> u64`) closes that hole: a tile is always stored
under a key describing the inputs it was actually baked from. Invalidation is
therefore *structural* — a stale key is unreachable rather than wrong — and there
is no invalidation pass anyone can forget to run. Locked down by
`namespace_change_never_serves_the_pre_edit_tile`.

> **If you add an input to tile synthesis, add it to the namespace.** And bump
> `GENERATOR_VERSION` when the generator's output changes, or a cached run will
> keep rendering the old terrain while the code says otherwise. `THALOS_TILE_CACHE=0`
> disables the disk tier entirely while iterating on generation itself.

**A provider wrapper must do its work inside the spawned task.**
`TileProvider::request_tile` is called from `TileAtlas::update` in `Last`, on the
**main thread**, up to `max_concurrent_tile_loads` times per frame. A tile payload
is multiple megabytes, so a cache lookup that reads a file — or even just clones a
hit — on the caller's thread converts a background stall into a frame hitch, which
is the exact opposite of the point. Both cache tiers therefore spawn first and
look up second, holding their inner provider as an `Arc` so the task can own a
handle to it. Only the (cheap) namespace hash runs on the caller's thread.

### 2. Mip generation moved off the main thread

The `TileProvider` contract now requires providers to return the **full mip
chain**; `TileAtlas::update` no longer regenerates it. Mip filtering was per-tile
CPU work running on the main thread at every tile completion; it now runs on the
synthesis pool. It also means the cache tiers store fully-mipped payloads, so a
cache hit costs neither synthesis *nor* mip filtering.

### 3. Per-attachment resolution — a memory win, honestly labelled

Height keeps the full grid (it is the geometry, and the only attachment anything
physical reads). Albedo/roughness/material bake at half linear resolution. The GPU
atlas already sized every attachment's texture array independently and the shader
already carried a per-attachment `scale`/`offset`/`size`, so this was purely a
provider-side assumption to relax.

Those three are 10 of the 14 bytes per texel, so the atlas drops from ~14 B to
~6.5 B per height-texel-equivalent — better than a 2× cut in the single largest
allocation in the game. **It is not a synthesis win**: the provider now evaluates a
second, coarser grid rather than encoding everything from one, so field evaluation
gets slightly *more* expensive. (The extra grid is cheap — band-limited to its own
resolution, so its `tile_lod_m` is coarser and the cascade resolves fewer octaves —
but it is not free, and pretending otherwise would misattribute the win.)
Evaluating it separately rather than box-filtering the height grid down is what
keeps borders bit-identical with neighbouring tiles.

### 4. Screen-space-error-aware refinement

Pure view-distance subdivision spends the same tile budget on glass-flat ocean as
on a mountainside. `TileProvider::subdivision_scale` is a new seam: the provider
probes the surface across a tile's footprint, converts *relative* relief (relief ÷
the tile's own arc length — absolute metres are meaningless without the footprint)
into a scale in `[0.6, 1]`, and memoizes it per coordinate. The tile tree clamps to
a floor and applies the same scale to **both** streaming and draw-set refinement,
so what is drawn and what is streamed refine on one consistent threshold.

Bounded by construction: the scale is ≤ 1, so this only ever *removes* detail
relative to today's baseline, never adds. And it is consulted only for tiles that
already pass the distance test, keeping the probe off the full ~6 k-slot tree sweep.

### 5. Frustum-aware requests + dead-prepass deletion

Standing on a surface, roughly half the near tiles fall behind the camera and were
being fully synthesized. `TerrainViewConfig::cull_behind_view` defers *streaming*
of tiles more than ~115° off the view axis, past a near keep radius. **Hole-free by
construction**: the pinned root LODs guarantee a resident ancestor, so a deferred
tile just draws from its coarse parent until the camera turns toward it. Off for
the map view, which sees the whole body at once.

The dead GPU tiling prepass is gone: `tiling_prepass.rs`, `culling_bind_group.rs`,
both prepass shaders, the `Parameters`/`refine_tiles`/`prepare_indirect` bind
groups, and the per-frame specialization of four compute pipelines per view that
was running every frame for a pass that never dispatched. The CPU draw-set path
(which enforces the 2:1 cube-seam balance the GPU's per-tile-independent predicate
could not) is now the sole, un-shadowed tile-selection authority.

## What did *not* land, and why

**GPU tile production.** The architecture note calls for providers enqueuing GPU
jobs that write atlas layers directly, and that is where the big near-field
synthesis win lives. Investigating it surfaced a blocker that is an architectural
decision, not a coding task:

Thalos's terrain invariant is that **every height consumer reads the same
surface** — rendered mesh, physics collider, CPU height queries (spawn-site search,
EVA, HUD altitude). Moving synthesis to a compute shader means porting the whole
`ProceduralSurface` cascade to WGSL, which creates a **second height authority**
that will drift from the CPU one the colliders still use. Resolving that means
choosing one of:

- **GPU generates, CPU mirrors.** The `GpuAtlasHeightMirror` already reads the
  atlas back for colliders, so the GPU could become the authority *for resident
  tiles*. But the CPU fallback (used wherever no tile is resident — spawn-site
  search, distant queries) would then run a *different* generator, turning today's
  known sub-metre "CPU/GPU bilinear stand-off" into full generator divergence.
- **Keep CPU authoritative, GPU for appearance only.** Safe, but nearly worthless:
  one `sample_d()` produces height *and* albedo/roughness together, so splitting
  them off saves no field evaluation.
- **Port to WGSL and hold the two implementations bit-comparable**, with a
  differential test as the gate.

This needs a decision and a GPU to verify against; it is not something to land
blind. The CPU-side items above are what speed the near field *today*.

## Verification boundary

- **Verified**: `cargo check --workspace --all-targets`, `cargo clippy --workspace`
  (no new warnings), and `cargo test -p thalos_udlod` — 21 tests, including new
  disk-format round-trip, layout-mismatch rejection, and namespace-invalidation
  coverage, plus the pre-existing 2:1 balance invariants that the tile-tree changes
  touched. Two bugs were caught this way and fixed rather than shipped: the disk
  codec tripping `bytemuck`'s alignment requirement (the tag/length header leaves
  the payload unaligned, so decode is explicit little-endian chunks), and the cache
  tiers initially doing their IO on the main thread.
- **Needs an in-game pass**: items 4 and 5 change what streams onto the screen.
  They are conservative and hole-free by construction, but the constants want
  tuning against a real view — `SSE_FLAT_SCALE` / `RELIEF_FULL_DETAIL_GRADE`
  (`ground/pipeline.rs`), `SSE_MIN_SCALE` / `BEHIND_VIEW_COS` (`tile_tree.rs`).
  Item 3 halves albedo/material resolution, which is a visual change worth a look
  on the ground.
