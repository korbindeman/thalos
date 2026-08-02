# Mira airless terrain MVP

**Status:** playable package-backed MVP landed 2026-07-20; production diffusion path specified · **Decision:** [ADR-20260720T211046Z-offline-terrain-packages](../adr/20260720T211046Z-offline-terrain-packages.md) ·
**Execution:** `MIRA-0`…`MIRA-4` in [backlog.md](../backlog.md) · **Completion
program:** [mira_learned_terrain.md](../roadmap/mira_learned_terrain.md)

### Landed compatibility vertical slice

The playable MVP deliberately lands before the trained diffusion producer:

- `just bake Mira` runs the standalone `thalos_terrain_baker` and emits a
  content-keyed 29.6 MiB schema-v1 `assets/terrain_packages/Mira.bin` (±13,004 m,
  24,179 indexed craters with the current authored seed). `just validate-bake
  Mira` independently validates schema/body/content identity, sparse node/blob
  references, bounds, overlap, checksums, and payload decoding;
- `BodySurfaceRegistry` loads one `Arc<dyn SurfaceQuery>` per body. Mira uses
  `PackageSurface`; procedural bodies use `ProceduralSurface`. Ground UDLOD,
  impostor/albedo projection, height mirror, collision/altitude, and propagator
  collision all clone that same surface;
- the package fingerprint participates in memory/disk tile-cache namespaces, so
  a rebake cannot reuse stale reconstructed tiles;
- `just game mira` / `just game mira-eva` parameterise the existing spawn path,
  and `mira-orbit` / `mira-surface` / `mira-eva` headless captures provide
  stable orbit, landmark-regolith, and canonical eye-level probes. The
  reference-matched framings `mira-disc` / `mira-approach` / `mira-rim` were
  added 2026-07-22 alongside them (see `roadmap/mira_learned_terrain.md`
  §"Visual targets"): they target the three visual goals rather than a
  regression surface, and each pins a distinct solar geometry, so relief and
  albedo can be judged separately instead of confounded in one frame;
- airless archetypes select the shared Hapke regolith path and suppress
  terrestrial vegetation/grass. Close detail is reconstructed deterministically
  by the existing client tile synthesis and cached on device.

The canonical `mira-eva` probe added 2026-07-21 exposed and now guards the
transparent-looking horizon stipple fixed in `BL-16` / `INC-0009`. It poses the
camera at the actual canonical EVA site, samples the live atlas-backed height,
and uses eye-height tangent-look semantics; `mira-surface` alone did not frame
the affected spawn site. The terrain was always opaque and completely
rasterised. Unfiltered metre-scale procedural colour and normal octaves fell
below the screen-space Nyquist limit on grazing ridges, and Hapke amplified the
normal aliasing. Airless shader detail must therefore fade each octave against
the fragment's body-space pixel footprint, not camera distance alone. The
compatibility crater compositor also keeps exact angular distance in f64;
downcasting a near-unit direction dot product before `acos` quantises Mira-scale
surface distance into visible height steps.

For this compatibility slice only, the offline producer puts non-height
`StaticSurfaceData` metadata in one global v1 node and authoritative height in
a 32→512 cube-face pyramid: six `RawU16LE` coarse roots plus independently
quantized signed-residual children. Every logical node is indexed; a missing
payload means the canonical parent predictor is within the declared error and
the reader falls back to that ancestor. The metadata contains only a 1×1 height
placeholder, so this path cannot silently fall back to monolithic baked height.
It is not the old runtime startup-bake flow, and the player never runs it.
MIRA-0 is complete; MIRA-1/2 replace the compatibility producer with trained
hierarchical diffusion. This keeps the playable integration
honest: the deployment boundary is complete, the ML terrain generator is not.

## 1. Outcome

Mira is the first body produced by Thalos's new terrain bakery: a hierarchical
diffusion pipeline runs on developer/CI hardware, emits a versioned adaptive
terrain package, and the game streams that package through the existing UDLOD
tile contract. The player's machine never runs the planetary diffusion stack.
It reconstructs only the final close-range bands from baked conditioning data,
using a bounded deterministic detail stage suitable for ordinary GPUs.

The result must read as the same moon in the map, in orbit, during descent, and
on foot. Complex regions such as fresh crater rims, ejecta, scarps, and rough
highlands receive more stored residual detail. Flat mare and old regolith stop
subdividing earlier and compress to a cheap coarse representation.

This is an **airless-family vertical slice configured for Mira**. No baker,
package reader, renderer, cache, or detail stage may branch on Mira's literal
name or body ID.

## 2. Design basis

The approach borrows four ideas from InfiniteDiffusion / Terrain Diffusion:

- overlapping diffusion windows are fused during generation rather than
  independently generated and blended at the end;
- a coarse-to-fine hierarchy carries planetary context into local synthesis;
- elevation is represented as low-frequency structure plus Laplacian residuals;
  and
- tiles are seed-consistent and independently addressable.

Terrain Diffusion demonstrates these properties on unbounded planar terrain.
Thalos changes the deployment and topology: inference happens offline, Mira is
a finite closed sphere, and the deliverable is an immutable cube-sphere package.
The paper's model hierarchy and residual representation are inspiration and
research input, not a drop-in runtime dependency.

The final bakery is a fresh producer targeting the live `SurfaceQuery`/UDLOD
contract. The compatibility MVP temporarily serializes `StaticSurfaceData` to
prove that boundary; no startup bake check or dump/editor flow was revived.

## 3. Frequency and authority contract

```text
final surface = B0 + B1 + B2 + Rclient + Dvisual

B0  planetary macro       baked: basins, highlands, mare, silhouette
B1  regional geology      baked: major craters, rings, ejecta, scarps
B2  stored local residual adaptive: retained only where error/complexity needs it
Rclient close reconstruction deterministic, conditioned, collidable where enabled
Dvisual micro shading     normals/grit only; never changes gameplay contact
```

The package is authoritative for `B0..B2`. `Rclient` is a versioned algorithm
with package-provided conditioning and seed. Every collidable term must be
reconstructable through the CPU `SurfaceQuery` path as well as the GPU tile
fill. If a future learned client decoder cannot meet deterministic parity, its
output is appearance-only and cannot move landing legs, EVA, or impacts.

The bakery and client agree on explicit wavelength cutoffs. Stored residuals
must average near zero at their parent scale; client reconstruction must do the
same at the finest stored scale. Refinement adds bandwidth, not a new skyline.

## 4. Authored airless schema

`TerrainConfig::Package` points at a package manifest and retains the frozen
generation intent needed for inspection and rebaking. Its airless-family params
include:

- seed, radius/body-scale validation, model family/version, and bake profile;
- crater density, SFD slope, maximum diameter, age distribution, fresh-impact
  bias, secondary/ejecta intensity, and gardening;
- mare fraction, near-side bias, basin/ring prevalence, and regolith depth;
- highland/mare/fresh-material albedo, roughness, and rock abundance; and
- client reconstruction band, amplitude budget, and deterministic seed salt.

The package header stores the exact normalized parameter snapshot, model and
tool hashes, training-dataset/adapter identity, coordinate convention, units,
channel schemas, and reconstruction contract. The game refuses incompatible
major versions and treats optional unknown channels as absent only when the
manifest marks them non-authoritative.

### 4.1 Mira v1 authoring baseline

Freeze a comparable starting profile before expensive full-face bakes. These
are tuning inputs, not a claim that Mira copies the real Moon's geography:

| Field | Mira v1 starting point |
|---|---|
| family | `airless` |
| radius | 869,000 m |
| surface gravity | 0.12 g |
| mare fraction | explore 0.15–0.35 |
| crater retention | mid-high highland density, conditioned SFD |
| gardening | medium; preserve fresh-rim outliers |
| local hand-off diameter | start at 2–3 km and remeasure at final H0 resolution |
| material vocabulary | highland, mare, crater floor/fresh excavation |
| identity | 64-bit seed + normalized params + model/tool/package versions |

Real lunar data teaches structure and process statistics. Mira's layout remains
a fiction seed and parameter vector; the producer must not crop or remap a
recognisable real lunar region.

## 5. Package architecture

### 5.1 Address space

Use the same cube-sphere quadtree address as `thalos_udlod`:
`(face, lod, x, y)`. One mapping implementation and one edge-neighbour table are
shared by the baker validator and Rust runtime. The model may infer overlapping
tangent patches internally, but package tiles are emitted in this canonical
address space.

Cube faces must not be generated as six unrelated images. Offline inference
requests cross-edge context in body-direction space, blends overlapping window
predictions, and writes shared border samples from one canonical edge owner.
Corners and poles are validated by geodesic sampling, not only face-image
comparison.

### 5.2 Progressive residual pyramid

Each node stores a reconstruction of its parent plus optional residual
children:

```text
body.tpkg
├── manifest
├── root faces / coarse mip chain
├── quadtree index
│   └── child bitmask, bounds, error metrics, blob references
└── content-addressed blobs
    ├── height low-pass or residual
    ├── material/albedo/roughness
    └── conditioning: age, unit, crater/rock density, optional latent features
```

A child is retained when reconstructing it from its parent exceeds a
rate-distortion threshold. The decision combines:

- maximum and RMS height error in metres;
- normal/slope error, weighted heavily on silhouettes and landing-scale scarps;
- curvature and crater-rim preservation;
- material-boundary error for mare/ejecta/fresh excavation; and
- authored importance overrides for future hero sites.

Flat mare can terminate with a plane/low-order predictor plus a tiny residual.
Rough crater walls may retain several more levels. Runtime lookup always falls
back to the nearest stored ancestor, so a sparse package is hole-free.

### 5.3 Encoding and compression

MVP height tiles use a low-order predictor plus quantized signed residuals.
Choose quantization per node from its recorded height range and maximum-error
budget; do not force the whole moon through one global R16 range. Compress
payloads independently with a practical lossless codec such as Zstandard so a
single requested tile does not require inflating a large archive.

Material channels use compact palette/weight encodings. Constant channels and
constant/planar height nodes have explicit representations instead of spending
a full texture on zeros. Blobs are content-addressed, allowing identical or
deduplicated payloads and patchable package revisions.

Codec selection is a measured bake output, not a hand-authored terrain-class
rule. The baker records raw bytes, compressed bytes, decode cost, and error for
each candidate encoding, then selects the cheapest option satisfying the node's
quality budget.

### 5.4 Source package versus runtime cache

The shipped package and the user's tile cache are different layers:

```text
RAM reconstructed-tile cache
  └── disk reconstructed-tile cache
        └── immutable package tile provider
              └── nearest stored residual chain + client reconstruction
```

- The **package** is authored content and may be required for gameplay.
- The **cache** is disposable memoization of decoded/reconstructed UDLOD tile
  payloads. Deleting it cannot change terrain.
- The cache namespace includes package content hash, reconstruction version and
  quality tier, attachment layout, body scale, and dynamic flatten state.
- Immutable decoded base data and dynamic structure flattens remain separable
  where practical, so placing a pad does not duplicate unrelated package bytes.
- The existing RAM/disk cache provider is extended rather than replaced. Cache
  reads, decompression, residual reconstruction, and writes stay off the main
  thread and obey byte budgets.

## 6. Offline bakery

The bakery is a separately runnable Rust toolchain. Per ADR-20260721T033713Z-rust-native-learned-terrain, Burn owns one
backend-generic model, diffusion, and sampling definition across offline
training and inference. Campaigns may select WGPU, CUDA, ROCm, CPU, or Burn's
Candle backend without forking model code. Learned crates remain independent of
Bevy and outside the game dependency graph until a measured optional runtime
feature needs them; normal play consumes packages and never requires planetary
diffusion.

### Production inputs and reproducibility

The first airless corpus uses complementary teachers:

- checksum-pinned SLDEM2015 regions at the 236.901 m/px macro training scale,
  with the global product retained as the later whole-sphere source;
- selected, aligned Kaguya Terrain Camera DTMs for fine high-pass examples;
  and
- parameter-labelled synthetic crater, ejecta, secondary-chain, gardening, and
  short mass-wasting surfaces to cover process extremes missing from the real
  sample.

Data preparation records source URLs, exact filenames, hashes, licences,
projection/pole handling, horizontal alignment, and vertical bias in a
`terrain_data/manifest.json`. Geographic holdout blocks must be disjoint from
training. Raw source DEMs are training inputs, not game-package payloads;
redistribution and derived-weight obligations are verified before data is
adopted rather than assumed from the source host.

### Stage A — spherical authoring prior

Generate or author the global airless controls in body-direction space: mare
and megabasin layout, crust/highland prior, crater-age/density fields, tidal
near-side axis, and optional hero masks. This top level is finite and sphere
native; it prevents six-face drift and gives the diffusion hierarchy global
control.

### Stage B — hierarchical diffusion

Run an airless-specific model/adapter trained from lunar/airless DEMs plus
process-generated crater surfaces. An Earth-terrain checkpoint is not accepted
as Mira's production model: river basins and terrestrial erosion statistics are
the wrong prior. Each finer model conditions on the coarser elevation,
materials, age/density fields, and stable world coordinates.

Use overlapping inference windows with shared noise and intermediate-path
fusion. Generation order must not affect output. Produce low-pass elevation and
Laplacian residual channels at explicit physical resolutions.

The cascade's output contract is **height only**. Albedo and material provinces
are an authored/procedural field conditioned on the same seed and normalized
parameter vector, layered over the learned height rather than emitted by the
model — see ADR-20260722T084154Z-airless-material-provinces-authored. That ADR
also records why: at the orbital full-disc framing, albedo province structure
carries the image and every learned height band falls below one pixel.

The minimum proof is a coarse-to-fine 2D U-Net cascade trained on 256–512 px
tangent patches. A practical starting ladder is:

| Stage | Approximate wavelength | Primary teacher |
|---|---:|---|
| S0 | 8–32 km | global DEM + synthetic basins |
| S1 | 2–8 km | global DEM mid bands |
| S2 | 0.5–2 km | global DEM fine bands + process simulation |
| S3 | 100–500 m | regional DTMs + synthetic high-pass |
| S4, optional | 30–100 m | best regional DTM islands only |

Each stage conditions on its low-frequency context, the airless parameter
vector, stable direction/seed features, and optional authored atlas channels.
Training emits resumable EMA checkpoints, a model card containing dataset and
code identities, validation hillshades, radial spectra, slope histograms, and
crater-count/SFD proxies. Prove the tensor and normalization path by overfitting
one to four tiles before scaling training.

### Stage C — physical/profile correction

Measure crater SFD, rim/floor ratios, slope distributions, and residual bias.
Apply bounded corrections where necessary so generative realism does not break
the authored gravity/scale envelope. This is validation/correction, not a
second procedural surface pasted over the generated output.

### Stage D — adaptive package build

Convert the dense/generated hierarchy into the sparse residual quadtree, build
borders and mips, choose codecs, compute per-node error bounds, content-hash
blobs, and emit previews/statistics. A bake is accepted only if the Rust package
validator can reconstruct it independently.

The campaign bake starts at 4096 texels per cube face (roughly 335 m/texel near
a face centre on Mira); 8192 is a measured hero-package option, not the default.
The current 512-face compatibility artifact remains an integration fixture.
Dense raw R16 height alone would cost about 192 MiB at 4096 or 768 MiB at 8192,
before metadata, which is why the shipped result must pass through the adaptive
residual encoder rather than becoming a uniform face pyramid.

Spherical inference uses `hash(seed, direction)` and overgenerated seam belts.
Predictions covering the same body directions are fused before quantisation;
the package then writes each shared boundary through its canonical owner.
Acceptance artifacts include six-face hillshades, an equirectangular preview,
limb renders, face/corner geodesic probes, and package composition/error maps.

## 7. Client reconstruction (“upscaling”)

The MVP client stage is a deterministic conditioned residual generator, not a
second planetary diffusion stack. It receives the finest stored height and
derivatives plus package conditioning such as material, crater age/density,
roughness class, and a stable seed. It adds only the final bounded wavelengths:

- sub-tile craterlets and softened regolith undulation;
- slope/curvature-conditioned talus and ejecta texture;
- rock-density masks for the existing scatter system; and
- centimetre-to-decimetre normal/roughness detail in the shader.

The collidable portion is implemented in pure Rust and mirrored by tile
production. Cosmetic GPU detail is explicitly separate. A later small learned
super-resolution decoder is allowed behind the same interface, but must either
pass CPU/GPU parity and determinism gates or remain visual-only.

That learned residual is a small single-pass CNN over stored height, slope,
material/age/roughness conditioning, and stable seed noise—not multi-step
diffusion on the player's GPU. It is attempted only after the deterministic
procedural `Rclient` ceiling is measured against the low-approach gate. Tier 0
must remain playable without learned residual weights.

Quality tiers change the finest reconstructed bandwidth and cosmetic density,
not package macro geology. The lowest tier remains the same Mira with less
close-up detail.

## 8. Implementation slices

### MIRA-0 — Package contract and tracer integration

Define the versioned manifest, sparse quadtree index, tile blob headers,
content/error metadata, and Rust reader/validator. Add a `PackageSurface`
backing behind one canonical per-body surface factory. Route ground terrain,
map terrain, impostor projections, height registries, and propagator collision
through the same body-keyed `Arc<dyn SurfaceQuery>`.

Use a tiny hand-built fixture package first: a grey, slightly relieved Mira.
This proves package loading, fallback-to-ancestor reconstruction, cache
namespacing, regolith shading, and N-body selection before ML output obscures
integration failures.

**Landed 2026-07-20:** the compatibility artifact has 2,047 logical nodes and
1,961 blobs. Its canonical half-open ownership gives every texel one package
address; 86 child payloads fall back to ancestors within a 256 m compatibility
budget. The exact artifact, not merely its authored-input key, fingerprints the
reconstructed tile-cache namespace.

**Exit:** build/clippy clean; fixture package round-trips through the standalone
validator; Mira renders from the package in named-body map/orbit captures; no
consumer directly constructs `ProceduralSurface` for a package-backed body.

### MIRA-1 — Airless diffusion patch proof

Build the offline model pipeline on representative planar/tangent airless
patches. Establish the training corpus, physical scale, signed elevation and
Laplacian channels, conditioning schema, overlapping-window inference, and
deterministic seed/coordinate protocol. Compare hierarchical output against
independent-tile blending and the current analytic crater reference. Pin the
global and regional lunar inputs, build their Gaussian/Laplacian pyramids and
geographic holdouts, add labelled synthetic process surfaces, then overfit a
single stage before training the S0–S3 ladder.

The implementation is split into `thalos_terrain_learned`, containing the
backend-generic Burn model/sampler contract, and `thalos_terrain_train`, the
offline corpus/training/validation binary. Checkpoints use portable model
records and record the Burn version/backend; package output remains independent
of the training backend.

**Tracer evidence, 2026-07-20:** the Rust smoke command generated 48
ChaCha8-seeded 64² patches at 250 m/px with labelled mare, crater density,
gardening, rim, ejecta, and secondary-chain controls, then decomposed them into
physical S0–S3 Laplacian bands. A compact eight-channel Burn denoiser completed
10 Flex/autodiff batches in 6.55 s (`0.999119 → 0.972475` noise MSE), wrote
SafeTensors with canonical tensor SHA-256
`fd3b2807bc03cee347b1c44852b5b5e24fb843d4291cacf5fa6ceaf4fab63b3c`,
and ran six shared-coordinate-noise 64² DDIM windows over a 128×96 canvas.
Repeated inference was bit-identical (`0 m` max delta); overlap predictions
disagreed by `1.674274 m RMS` before weighted fusion. CPU and WGPU feature
graphs both compile. The inspected corpus/Laplacian sheet has crater structure,
but the two-epoch sampled residual remains noise-like, so this proves the
data/model/checkpoint/overlap path only and does **not** satisfy MIRA-1's quality
exit.

Three exact CC0 USGS Kaguya/LOLA-aligned S3 teachers are now pinned by URL,
STAC item, byte size, SHA-256, shape, bounding box, physical resolution, and
split: a 30.33 m/px mare/highland contact for training, 19.07 m/px Copernicus
block for validation, and 38.82 m/px Tycho block for holdout. The Rust
preprocessor verifies the complete download before decoding its float32 COG,
resamples all three to 40 m/px, rejects patches below 99% valid coverage,
locally inpaints the remaining small no-data islands, removes vertical bias,
and produced 23 train / 5 validation / 28 holdout 256² patches. Their
hillshades were inspected and the split regions do not overlap. The partial
Tycho download was deliberately rejected by the checksum gate before the
complete 4,209,915-byte artifact was adopted.

The complementary macro teacher is now an exact subset of the official
SLDEM2015 128 ppd PDS FLOAT product (236.901 m/px map scale). Three 2° latitude
strips are pinned as inclusive HTTP byte ranges from the 2.8 GB source, keeping
each fetch at 47,185,920 bytes. The Rust `prepare-sldem` path checksum-gates the
range, crops a 256² longitude window, decodes little-endian kilometre samples
to metres, and reuses the Kaguya validation/index/preview path. The adopted
train, Copernicus-validation, and Tycho-holdout range hashes are recorded in
`terrain_data/manifest.json`; all three have 100% valid samples, zero >150 m
single-pixel impulses, and inspected structurally distinct hillshades.

Training now applies the configured EMA after every optimizer step and uses it
for validation/export. Each epoch checkpoint stores raw and EMA SafeTensors,
full-precision Adam state, deterministic progress, physical scales, and a
parameter-path-to-Burn-ID map. Resume remaps Adam slots by path because Burn
parameter IDs are randomized per process. In the smoke proof, a run resumed
from epoch 2 through epoch 3 exactly matched an uninterrupted 3-epoch run:
final loss `0.9612839`, EMA hash
`f982b142fc91b904a7103b72acb7a0b6fe9dcd6af8dbdf7dc0dd3127054c47ff`,
and raw hash
`41858e5b929da18af031f148065c0d68617fec987c09fc1528d40e946be943e6`.

The first expanded CUDA campaign (`mira_l2_kaguya_cuda_v3`) corrected the
diffusion terminal state to alpha-bar 2.14e-5 but retained epsilon prediction.
It converged to 0.0095 training loss in 328.79 s on an RTX A6000, yet the held
Copernicus gallery regressed to 104.93 m RMS and diagonal high-frequency noise.
A 25→100-step sampling differential did not remove the artifact. Instrumenting
the shared sampler showed the terminal epsilon→clean conversion exploding to
70.1 RMS normalized clean height and driving 9.0% of the final field into the
output clamp. The shared `thalos_terrain_learned` contract therefore now
supports velocity prediction for both training targets and DDIM reconstruction;
checkpoint metadata prevents resuming under the wrong objective. The controlled
v4 changed only that target and completed 2,280 A6000 batches in 291.15 s. It
reduced held Copernicus error from v3's 104.93 m to 26.67 m RMS and restored
recognisable morphology, validating the terminal-SNR diagnosis. The inspected
output still has dense worm-like high-frequency texture, its spectrum slope is
−2.83 versus the target's −3.68, and it does not beat v2's 24.77 m RMS, so L2
remains open. The next controlled local-CUDA diagnostics isolate transposed-
convolution upsampling and timestep encoding; more epochs are not justified.

MIRA-1 pilots now default to the persistent local RTX 4070 Ti. Cloud is reserved
for measured VRAM overflow or batched frozen campaigns, with user-run `tnr scp`
ingress/egress and agent-managed control-plane work, per
ADR-20260721T020849Z-local-cuda-first-mira-campaigns.

SLDEM distribution-rights confirmation for derived-weight release, combined
campaign training, spectral/slope/SFD validation, and GPU VRAM/timing remain
open.

**Exit:** a fixed-seed patch set shows plausible fresh/old crater morphology,
no terrestrial drainage signature, seamless overlap interiors, stable repeated
generation, and recorded dataset hashes, model card, inference time, VRAM,
spectral/slope metrics, crater proxy counts, and model hashes.

### MIRA-2 — Whole-sphere adaptive bake

Add sphere-native global priors, cross-face tangent-window scheduling, canonical
border ownership, full hierarchy generation, adaptive residual pruning, and
per-node codec selection. Start with a 4096-face fixed Mira seed and emit
whole-body maps,
face/corner seam probes, crater/SFD statistics, package-size composition, and
rate-distortion curves.

The schema must gain an explicit **material/province node kind** before the first
whole-sphere campaign. `PackageBlobKind::StaticSurfaceV1` is the compatibility
substrate this slice replaces, and it is what carries Mira's albedo today; the
diffusion producer emits no material channel, so replacing the substrate without
that node kind renders the body grey. See
ADR-20260722T084154Z-airless-material-provinces-authored.

Cross-face tangent-window scheduling is the same problem MultiDiffusion-style
overlapping-window fusion solves on unbounded domains; see the InfiniteDiffusion
notes in `roadmap/mira_learned_terrain.md` for the reference treatment.

**Exit:** the package reconstructs within its declared error at every retained
level; seam belts remain within one quantisation step; flat mare demonstrably
costs less than rough highlands; orbit/map captures show a coherent moon with
regional secondary chains and at least two roughness provinces but no cube
seams or face-specific style drift.

### MIRA-3 — Client reconstruction and cache

Implement deterministic close-band reconstruction conditioned on package
channels, plus regolith boulder placement and cosmetic micro normals. Extend
the existing memory/disk cache namespace to the package and reconstruction
identities; benchmark cold package decode, warm disk hit, and RAM hit.

**Exit:** distance-bracketed approach/surface captures add detail without
moving resolved macro features; CPU height and rendered/GPU-mirror height stay
inside the declared tolerance; repeated visits hit cache; cache deletion leaves
identical terrain; no main-thread decode/write hitch.

If those captures still miss sharp rims or 20–200 m ejecta hummocks, evaluate
the optional compact residual CNN behind the same tiered interface. Do not add
it merely because the global diffusion path exists.

### MIRA-4 — Playability and acceptance

Parameterise the existing spawn path by target body and expose stable Mira
orbit/EVA routes without copying placement logic. Tune the fixed package and
client detail budget, then verify camera floor, HUD altitude, EVA/SLF contact,
landing gear, and on-rails terrain impact against the package-backed surface.

**Exit:** all automated probes/headless captures pass; the user verifies both
Mira routes and orbit-to-ground continuity. Until that live pass, the slice is
`verify`, not `done`.

## 9. Verification and budgets

Terrain-generation tests remain paused during the current visual iteration
phase. Generator evidence is produced by standalone bakery validation commands,
reports, and headless captures rather than new per-body CI terrain tests.

| Evidence | Acceptance |
|---|---|
| Package validator | Hashes, bounds, codecs, ancestor fallback, borders, and declared reconstruction errors valid |
| Whole-body map + six-face preview | Recognisable airless macro identity; no Earth-like continents/drainage or face seams |
| Complexity/size heatmap | Fine storage correlates with measured error/curvature, not arbitrary biome labels |
| Rate-distortion report | Package size and max/RMS/normal error shown per level and terrain class |
| Crater report | SFD and morphology bands plausible across stored/reconstructed hand-off |
| Orbit/approach/surface captures | Stable identity, detail-only refinement, Hapke/vacuum lighting, no terrestrial layers |
| Cold/warm tile benchmark | Package decode/reconstruction and both cache tiers measured; no main-thread I/O |
| CPU/GPU-mirror probe | All collidable bands agree within the existing tile quantisation/LOD tolerance |
| User Mira orbit/EVA session | Loading, navigation, contact, scale, and visual continuity feel correct |

Visual acceptance is split deliberately:

- **Orbit gate:** multi-scale crater SFD, regional rather than wallpaper-like
  secondary chains, mare/highland roughness contrast, a continuous relieved
  limb/terminator, and bit-identical identity on rebake.
- **Low-approach gate:** adjacent fresh and degraded crater ages, non-empty
  ejecta energy in the 20–200 m band, plausible wall-slope distribution,
  stable silhouettes from 20 km, and bounded height error during a 50 m AGL
  geodesic traversal.
- **Engineering gate:** seam, CPU/GPU parity, Tier-0 fallback, package bytes,
  page latency, and frame time are all recorded for a dev draft and a campaign
  package.

Failure at orbit routes back to S0–S2 or spherical consensus. Failure only at
low approach routes to local reconstruction/residual work, not indiscriminate
global-resolution increases.

Initial budgets are measurements, not invented limits: MIRA-0 records current
Thalos tile payload/cache costs; MIRA-1 records offline model cost; MIRA-2 uses
the rate-distortion curve to propose a shipped Mira package budget; MIRA-3 sets
the client latency and cache budgets from cold/warm profiling. Quality gates are
fixed before choosing final compression thresholds.

### MIRA-0 measured baseline (2026-07-20)

| Metric | Result |
|---|---:|
| Package | 32.4 MiB; content key `4def278582a2df1d`; artifact fingerprint `d3d21da448e050cd`; SHA-256 `321bd224bccfc0e0f4759f88a6ed46675ceb7cb0f9d8ab7c58d041fcd9c9f395` |
| Hierarchy | 32→512 px, 5 levels, 2,047 logical nodes / 1,961 blobs |
| Sparse fallback | 86 residual payloads omitted; every >4×-budget node retained |
| Reconstruction | 246.288 m max, 1.352 m RMS against the dense compatibility source |
| Determinism | consecutive bakes produced identical key, artifact fingerprint, and SHA-256 |
| Validation | independent load/checksum/hierarchy/decode in 66 ms |
| Headless visual | cache-disabled `mira-surface` and `mira-orbit` inspected; no face seam, tile boundary, or fallback hole |
| Screenshot cache run | first normal-cache surface capture 61.4 s; repeat 53.5 s (fixed 1,200-frame probe dominates both) |

The adaptive height encoding alone is not yet a size win over six dense R16
faces: the compatibility generator has high-frequency energy in almost every
tile, and the large legacy metadata blob dominates the package. That is an
observed input to MIRA-2's per-node codec/compression work, not a reason to
remove sparsity. A learned low-pass H0 with genuinely flat mare is expected to
terminate much earlier, but its rate-distortion curve must prove that claim.

## 10. Risks and controls

- **Planar model on a sphere:** infer overlapping tangent patches with
  body-direction conditioning and validate geodesically across face boundaries.
- **Wrong geological prior:** train/condition an airless family from lunar DEMs
  and crater simulations; reject terrestrial drainage/erosion signatures.
- **Diffusion nondeterminism:** pin model/tool/runtime versions and deterministic
  settings; package bytes, not reproduction on the player's GPU, are authority.
- **Over-compressing important relief:** prune on measured height, normal,
  curvature, material, and silhouette errors with authorable importance masks.
- **Package/cache confusion:** immutable content-addressed package versus
  disposable reconstructed-tile cache, with separate paths and hashes.
- **Client detail breaks physics:** only parity-proven deterministic bands are
  collidable; learned or shader-only detail defaults to visual-only.
- **Old-pipeline gravity:** the new bakery targets the tile/package spec and may
  extract pure math, but cannot depend on the deleted startup bake/compiler flow.
- **Scope inflation:** learned client SR, hero sites, rays, secondaries, resource
  geology, footprints, and tracks follow only after the four-scale Mira proof.

## 11. Order and estimate

`MIRA-0` is complete. The first machine-learning slice is MIRA-1; each later
slice depends on an inspectable artifact from the previous one.

| Slice | Estimate | Depends on |
|---|---:|---|
| MIRA-0 | L | — |
| MIRA-1 | L–XL / research | MIRA-0 schema |
| MIRA-2 | XL | MIRA-1 |
| MIRA-3 | L | MIRA-2 package |
| MIRA-4 | M | MIRA-3 |

The completed package/consumer tracer prevents the ML pipeline from settling an
accidental format that the renderer, collider, or cache cannot consume.

## References

- [InfiniteDiffusion / Terrain Diffusion project](https://xandergos.github.io/terrain-diffusion/)
- [Terrain Diffusion paper (arXiv 2512.08309)](https://arxiv.org/abs/2512.08309)
- Vault design notes: `Work/Thalos/Terrain/Planetary Terrain Pipeline.md`,
  `Approach Explainer.md`, `MVP Cratered Moon.md`, and `Mira Diffusion Path.md`
- [SLDEM2015 product 54 (NASA PGDA)](https://pgda.gsfc.nasa.gov/products/54)
