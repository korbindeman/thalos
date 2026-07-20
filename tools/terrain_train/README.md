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
