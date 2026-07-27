# thalos_udlod

> **Status: LEGACY / end-of-life.** The default ground renderer is
> `thalos_body_render::tiles` — terrain as ordinary `Mesh` +
> `StandardMaterial` on Bevy's standard path (keystone ADR-20260723T142945Z).
> This crate survives as the A/B baseline (`THALOS_TILE_RENDERER=0`) and for
> bodies the tile driver has not installed on. Defect-driven fixes only; no new
> terrain features here.

`thalos_udlod` is Thalos's in-tree UDLOD terrain renderer. It began as a
fork of [`bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain) by
Kurt Kuhnert (MIT OR Apache-2.0), but is now shaped around Thalos's runtime
planet pipeline rather than upstream's preprocessed raster datasets.

The fork owns:

- sparse tile-atlas residency and parent-LOD fallback
- CPU-balanced draw-tile selection for cube-face seam correctness
- UDLOD vertex generation and attachment sampling
- the Taylor-series planet-scale precision path
- `TileProvider`, the runtime seam for synthesized or cached tile data
- `MemoryTileCacheProvider`, a small provider wrapper for in-memory
  frecency reuse of CPU tile payloads

The old GeoTIFF/preprocess/`DiskTileProvider` path has been removed. If tile
reuse needs persistence, add a Thalos cache provider or provider wrapper keyed
by body/source hash, tile coordinate, and attachment configuration.

The attribution and original dual license files travel with the source:
`LICENSE-MIT` and `LICENSE-APACHE`.
