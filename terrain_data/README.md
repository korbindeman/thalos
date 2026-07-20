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

Real-data adoption is intentionally two-step. Exact Kaguya S3 artifacts and
bounded ranges from the 128 ppd SLDEM2015 PDS FLOAT product are pinned for the
train, validation, and holdout splits in `manifest.json`. Add every future
artifact's URL, size, SHA-256, spatial extent, and licence/citation record
before fetching it into `terrain_data/raw/`. A catalog URL is not an artifact
pin.

Geographic train/validation/holdout blocks must not overlap. Record the split
at selection time rather than randomly splitting already-extracted patches.
