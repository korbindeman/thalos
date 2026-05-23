# thalos_udlod Architecture Note

This file used to carry the early fork plan. The canonical terrain rendering
spec now lives in [`docs/terrain.md`](../../docs/terrain.md).

Current direction:

- `thalos_udlod` is an in-tree, runtime-provider-first UDLOD renderer.
- `TileAtlas` owns sparse residency, atlas slots, parent fallback, and GPU
  uploads.
- `TileTree` owns view-dependent tile requests and the CPU-balanced draw set.
- `TileProvider` owns tile contents. The current providers return CPU
  attachment buffers; the next direct path should let providers enqueue GPU
  jobs that write atlas layers and mark the current slot generation ready.
- `MemoryTileCacheProvider` is the in-memory frecency wrapper for CPU tile
  payloads. Keep cache policy in provider/producer layers rather than in
  `TileAtlas`.
- Preprocessed Earth-style datasets, GeoTIFF loading, `config.tc`, and
  `DiskTileProvider` are intentionally gone. Persistent reuse should be a
  Thalos cache provider/wrapper, not a restored asset-tree source.

Keep the upstream license attribution in `README.md`, `LICENSE-MIT`, and
`LICENSE-APACHE`.
