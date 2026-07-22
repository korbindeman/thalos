# MIRA terrain training

Rust-native MIRA-1 corpus, training, and validation tool. The backend-generic
model and diffusion contract live in `thalos_terrain_learned`; this binary owns
large/reproducible data and run artifacts. It does not write gameplay packages
until the planar patch proof passes.

```bash
cargo run -p thalos_terrain_train -- smoke --config tools/terrain_train/configs/smoke.toml
```

The L1 visual gate deliberately overfits one fixed patch and writes
`validation_comparison.png` as target / coarse input / generated
reconstruction / signed error:

```bash
cargo run --release -p thalos_terrain_train --no-default-features \
  --features gpu -- smoke --config tools/terrain_train/configs/overfit.toml
```

The default `cpu` feature uses Burn Flex for deterministic smoke work. Build
with `--no-default-features --features gpu` for Burn/WGPU. Cloud campaigns use
native Burn/CUDA without changing model code:

```bash
cargo run --release -p thalos_terrain_train \
  --no-default-features --features cuda -- backend-info
```

The preflight must identify `burn-cuda-autodiff` and the expected GPU before a
paid run starts. ROCm or Burn's Candle backend can be added later without
forking the model definition.

## Campaign execution and transport

MIRA-1 pilots default to the persistent local RTX 4070 Ti after the same CUDA
preflight. CPU remains the small deterministic smoke path. Rent cloud hardware
only for measured local-VRAM overflow, cloud-specific evidence, or a batch of
frozen campaigns that amortizes provisioning and clean codegen.

A bare GPU driver is not enough for local CUDA: cubecl JIT-compiles kernels at
run time, which needs the NVRTC DLLs (`nvrtc64_*.dll` next to the binary or on
`PATH`) and the CUDA runtime *headers* (`cuda_runtime.h`, found through
`CUDA_PATH/include`). Without them the run fails with an NVRTC load panic or a
`cannot open source file "cuda_runtime.h"` compilation error. On a machine
without the full toolkit, fetch the three official NVIDIA redistributables
(`cuda_nvrtc`, `cuda_cudart`, and `cuda_crt` from
`https://developer.download.nvidia.com/compute/cuda/redist/`), copy the
`bin/x64/nvrtc*.dll` files beside `thalos_terrain_train.exe`, unpack the cudart
archive anywhere, merge the crt archive's `include/crt` into the cudart
`include` (the WMMA matmul kernels need `crt/mma.h`), and set `CUDA_PATH` to
the cudart archive root when launching. The PC's working copy keeps them under
`C:\Users\korbi\.thalos-cuda`.

Thunder uses a strict control/data-plane split. The agent provisions, runs,
monitors, verifies hashes, and deletes the instance. The user runs one `tnr scp`
source/data upload and one `tnr scp` evidence download. Do not embed private
archives in remote shell commands and do not recover large artifacts through
file-API chunks. Verify the source archive before extraction and verify both the
evidence archive hash and its internal manifest before deletion.

## Prediction target

`[diffusion].prediction` selects `epsilon` (the backward-compatible default) or
`velocity`. The enum and both conversions live in `thalos_terrain_learned`, so
training and offline sampling cannot drift. Checkpoints record the target and
refuse resume under a different objective.

The expanded epsilon campaign with a near-zero terminal SNR
(`mira_l2_kaguya_cuda_v3`) exposed the expected low-SNR conditioning failure:
epsilon-to-clean reconstruction divided residual prediction error by
`sqrt(alpha_bar)`, produced 104.93 m held-region RMS, and visibly collapsed into
diagonal noise. `configs/expanded_cuda_velocity.toml` is the controlled retry;
it keeps the v3 corpus, seed, schedule, architecture, optimizer, and epoch count
fixed and changes only the prediction target.

`configs/mira_s2.toml` records the production-shaped S2 campaign. It is a
measurement target, not a claim that its current capacity is final.

Prepared real patches enter training through `[[data.real_sources]]` entries.
Every sample carries an explicit train/validation/holdout split, source ID, and
effective metres-per-pixel. Holdout records are rejected by the loader. The
model conditions on physical scale, so Kaguya and SLDEM bands are not silently
treated as the same spatial frequency. `real_overfit.toml` and
`real_pilot.toml` are the reproducible L1 and held-region L2 probes;
`expanded_pilot.toml` adds the geographically separated expansion corpus.

The expansion is selected and downloaded from the official USGS Kaguya STAC
catalog by Rust code. Downloads use bounded timeouts, a `.partial` file,
content-length validation, SHA-256 provenance, and an idempotent shared patch
index. Copernicus and Tycho exclusion blocks remain unavailable to fitting:

```bash
cargo run --release -p thalos_terrain_train -- discover-kaguya \
  --raw-dir terrain_data/raw/kaguya_expansion \
  --processed-dir terrain_data/processed/kaguya_expansion_s3/train \
  --manifest terrain_data/kaguya_expansion.json \
  --per-region 2
```

Campaign machines materialize the frozen manifest instead of repeating the
live catalog query. Every artifact byte count, SHA-256, and prepared patch count
must match before training starts:

```bash
cargo run --release -p thalos_terrain_train -- materialize-kaguya \
  --raw-dir terrain_data/raw/kaguya_expanded \
  --processed-dir terrain_data/processed/kaguya_expanded_s3/train \
  --manifest terrain_data/kaguya_expansion.json
```

Training exports the EMA model as `model.safetensors` and the unaveraged model
as `raw_model.safetensors`. Epoch checkpoints include both weight sets, Adam
state, and `checkpoint.json`. Set `train.resume = true` to continue to the
configured total epoch count. Adam slots are remapped by stable parameter path
on load because Burn parameter IDs are process-local; a resumed run is expected
to match uninterrupted canonical tensor hashes exactly.

Existing checkpoints can be re-evaluated without retraining. The report covers
height error, slope quantiles, multiscale structure functions, radial spectral
slope, and crater-size proxies, and rewrites the fixed target/coarse/generated/
error sheet:

```bash
cargo run --release -p thalos_terrain_train --no-default-features \
  --features gpu -- evaluate \
  --config tools/terrain_train/configs/expanded_pilot.toml \
  --checkpoint terrain_runs/mira_l2_kaguya_expanded_pilot_v2/model.safetensors
```

## Whole-sphere preview (full-moon render)

`sphere-preview` is the L3 scouting step: it loads the authored Mira package,
decodes the six macro cube faces, runs the trained checkpoint over each face
with the validator's 64-px windows and overlap fusion, and writes
`equirect_{macro,enhanced}.png`, orthographic `moon_{macro,enhanced}.png`
discs, and a `sphere_manifest.json` next to the run directory:

```bash
CUDA_PATH=... cargo run --release -p thalos_terrain_train \
  --no-default-features --features cuda -- sphere-preview \
  --config tools/terrain_train/configs/expanded_cuda_fourier.toml
```

Optional: `--face-size` (default 512), `--body` (default Mira), `--assets`,
`--out`. It is a *stylized preview*, not a bake: faces are generated
independently (border seams are expected until L3 seam consensus), the
scale-condition channel is pinned to the SLDEM teacher scale because the macro
is ~2.7 km/px, the learned band is re-dimensionalized by the face/train coarse
ratio, and the mare conditioning is a low-elevation proxy. The module doc in
`src/sphere.rs` lists every approximation.

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

SLDEM2015 uses a 2.8 GB PDS FLOAT image. The manifest pins three 2° latitude
strips as HTTP byte ranges, keeping each fetch below 48 MB. `prepare-sldem`
verifies a strip, crops its longitude window, decodes PDS `PC_REAL`
little-endian kilometres into metres, and routes it through the same
validation/index/preview path:

```bash
cargo run -p thalos_terrain_train -- prepare-sldem \
  --input terrain_data/raw/sldem/lat24_26_strip.f32le \
  --output terrain_data/processed/sldem_s0/train \
  --source-id sldem_mare_contact_train --split train \
  --sha256 5f77d6761ee1f2f294b818724a88ea8fb12d8d9ca1f4c174b9dc5b14d364e2ec \
  --native-mpp 236.901 --target-mpp 236.901 \
  --patch-size 256 --stride 256 --source-width 46080 --crop-x 45312
```
