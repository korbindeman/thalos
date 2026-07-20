# Mira terrain training data

Only provenance and selection manifests are committed. Raw lunar products,
processed tiles, synthetic samples, checkpoints, and validation outputs remain
local because they are large or reproducible.

The MIRA-1 smoke path needs no external data:

```bash
cargo run -p thalos_terrain_train -- smoke --config tools/terrain_train/configs/smoke.toml
```

This writes reproducible corpus files below `terrain_data/synthetic/` and model,
metric, SafeTensors, and contact-sheet artifacts below `terrain_runs/`; both are
ignored. The default backend is Burn Flex/CPU. Verify the portable GPU graph
with `cargo check -p thalos_terrain_train --no-default-features --features gpu`.

Real-data adoption is intentionally two-step. First select exact SLDEM2015 and
Kaguya assets and add their URLs, sizes, SHA-256 hashes, spatial extents, and
licence/citation records to `manifest.json`. Only then may the downloader fetch
them into `terrain_data/raw/`. A catalog URL is not an artifact pin.

Geographic train/validation/holdout blocks must not overlap. Record the split
at selection time rather than randomly splitting already-extracted patches.
