# MIRA terrain training

Rust-native MIRA-1 corpus, training, and validation tool. The backend-generic
model and diffusion contract live in `thalos_terrain_learned`; this binary owns
large/reproducible data and run artifacts. It does not write gameplay packages
until the planar patch proof passes.

```bash
cargo run -p thalos_terrain_train -- smoke --config tools/terrain_train/configs/smoke.toml
```

The default `cpu` feature uses Burn Flex for deterministic smoke work. Build
with `--no-default-features --features gpu` for Burn/WGPU. Campaign runners may
later add CUDA, ROCm, or Burn's Candle backend without changing model code.

`configs/mira_s2.toml` records the production-shaped S2 campaign. It is a
measurement target, not a claim that its current capacity is final.

## Verified real DEM preparation

Exact Kaguya artifacts and splits are pinned in `terrain_data/manifest.json`.
The Rust preprocessor refuses a mismatched SHA-256, decodes float32 COG/GeoTIFF,
resamples to a common physical scale, rejects patches below 99% valid coverage,
removes per-patch vertical bias, and writes f32 little-endian patches plus an
index. Example:

```bash
cargo run -p thalos_terrain_train -- prepare-dem \
  --input terrain_data/raw/kaguya/copernicus_validation.tif \
  --output terrain_data/processed/kaguya_s3/validation \
  --source-id kaguya_copernicus_validation \
  --split validation \
  --sha256 811d67da100242913adbae1495770bf6c424c0cdadccab184a23b5c6813101b1 \
  --native-mpp 19.072880418585456 --target-mpp 40 \
  --patch-size 256 --stride 128
```
