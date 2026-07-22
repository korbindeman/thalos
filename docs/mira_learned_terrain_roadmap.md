# Mira learned-terrain completion roadmap

**Owner:** Codex execution thread · **Status:** active · **Started:** 2026-07-21
**Architecture:** [mira_airless_mvp.md](mira_airless_mvp.md) · **Decisions:**
[ADR-20260720T211046Z-offline-terrain-packages](adr/20260720T211046Z-offline-terrain-packages.md),
[ADR-20260721T033713Z-rust-native-learned-terrain](adr/20260721T033713Z-rust-native-learned-terrain.md),
[ADR-20260720T222139Z-mira-cloud-campaign](adr/20260720T222139Z-mira-cloud-campaign.md)

This is the operational route from the current diffusion tracer to a
photorealistic-stylized, package-backed Mira. The backlog remains the global
queue; this document owns the learned-terrain milestones, visual gates, cloud
budget, and evidence ledger.

## Definition of done

Mira is complete when one fixed authored seed:

1. has convincing airless macro geology, crater ages, ejecta, mare/highland
   contrast, and no terrestrial drainage signature;
2. survives map → orbit → approach → EVA without seams, feature movement, or
   loss of identity;
3. bakes deterministically into the adaptive package consumed through the one
   `SurfaceQuery` authority;
4. reconstructs bounded close detail with CPU/GPU collision parity and useful
   cold/warm cache performance;
5. renders through the shared Hapke/vacuum lighting path with regolith,
   boulders, shadows, and micro-relief; and
6. passes the automated evidence matrix plus a user orbit-to-surface session.

An attractive hillshade is a milestone, not completion. A technically valid
package that does not look compelling is also not completion.

## Baseline at roadmap start

- MIRA-0 package/runtime tracer is complete and visually verified.
- Rust/Burn 0.21 model, DDPM/DDIM schedule, overlap fusion, SafeTensors,
  canonical hashes, EMA, and exact cross-process Adam resume are implemented.
- Synthetic v1 can deterministically produce the configured 16,384-sample
  campaign.
- Real seed corpus: 23/5/28 Kaguya S3 train/validation/holdout patches and one
  SLDEM macro patch in each split, all checksum-pinned and preview-inspected.
- The shared sample contract now carries explicit source, split, physical scale,
  and weak process labels. The 11-channel tensor is compiled on CPU, WGPU, and
  native CUDA; holdout records are never returned by the training loader.
- A real Kaguya one-patch overfit passes L1. The first held-region pilots do not
  yet pass L2; their evidence is recorded below.

## Milestones and gates

| Milestone | Work | Required visual/evidence | Exit state |
|---|---|---|---|
| **L1 — Overfit proof** | Combined sample vocabulary; native CUDA build; 1–4 real/synthetic patches; deepen tracer into a multiscale denoiser | Ground truth / coarse input / generated residual / reconstruction / error hillshade sheet. Craters and rims recognisable without labels | **passed 2026-07-21** |
| **L2 — Patch model** | Expand geographically disjoint Kaguya regions; balanced synthetic process extremes; S0–S3 sampling; spectral/slope/crater metrics | Fixed-seed train/validation/holdout galleries at several diffusion steps; no obvious memorised real crop | MIRA-1 `done` |
| **L3 — Sphere bake** | Sphere-native controls, cross-face tangent scheduling, seam consensus, adaptive residual package | Equirectangular map, six faces, corner/seam belts, complexity heatmap, rate-distortion report | MIRA-2 `done` |
| **L4 — Close reconstruction** | Deterministic bounded client band, boulders and cosmetic micro; cache integration | Same site at orbit/20 km/2 km/50 m; macro features stationary while bandwidth increases | MIRA-3 `done` |
| **L5 — Photorealistic styling** | Hapke calibration, material provinces, rock abundance, contact shadows, exposure/framing | Headless orbit, oblique approach, crater-rim landing and EVA contact sheets | graphics gate passes |
| **L6 — Acceptance** | Height-authority probes, package/cache benchmarks, final user run | Full evidence matrix plus user orbit-to-ground continuity approval | MIRA-4 `done` |

Every generated gallery is stored under the ignored `terrain_runs/` directory;
the roadmap records its config, canonical model hash, representative metrics,
and inspected result. A milestone cannot advance on training loss alone.

## Dataset roadmap

| Source | Role | Start | L2 target |
|---|---|---:|---:|
| Synthetic airless v1/v2 | labelled craters, ejecta, secondaries, gardening, mare controls | deterministic generator | 16k–64k sampled patches |
| USGS Kaguya TC DTM | S2–S4 real morphology | **25 source DTMs / 183 patches** (22-source expansion manifest SHA-256 `57cff833…d2e9b`) | 100–300 geographically separated DTMs / several thousand patches |
| SLDEM2015 128/512 ppd | S0–S2 macro/mid teacher | 3 bounded macro regions | disjoint regional coverage across mare, contact, highland, basin provinces |

Selection occurs by geographic block before patch extraction. Holdout regions
never participate in fitting or normalization. Real data teaches morphology;
Mira's layout remains a fictional seed and conditioning vector.

## Compute plan and spend ledger

Cloud work follows ADR-20260720T222139Z-mira-cloud-campaign. Prices below were checked on 2026-07-21 against
[Thunder Compute pricing](https://www.thundercompute.com/pricing) and must be
rechecked at launch.

| Phase | Hardware | Maximum spend | Launch criterion | Actual |
|---|---|---:|---|---:|
| Local proof | local CPU/WGPU | $0 | L1 code and gallery command ready | $0 |
| Backend benchmark | 1× RTX A6000 + 1× A100, ≤1 h each | $1.44 | native CUDA graph compile-clean | — |
| Architecture/data pilots | best examples-per-dollar result | $8.00 | overfit visual gate passes | **~$0.75** — A6000 v3 + controlled velocity v4 |
| Medium campaign | measured winner | $15.00 | expanded corpus manifest frozen | — |
| Final campaign + one retry | measured winner | $20.00 | validation metrics and gallery regression-clean | — |
| Storage/error buffer | snapshot or short rerun | $5.00 | only as needed | **~$0.08** — two retry instances deleted after the tenant export guard denied the approved source transfer |
| **Hard ceiling** | | **$49.44** | user approval required to exceed | **~$0.83 estimated** |

The persistent local RTX 4070 Ti is now the default MIRA-1 pilot backend: v4
used only 552 MiB, far below its 12 GB capacity, while local CUDA avoids cold
provisioning, codegen, and transfer cost. Rent an A6000 only when measured VRAM
requires it, cloud-specific evidence is needed, or several frozen campaigns can
share one provisioned session. Thunder transfers use one user-run `tnr scp`
upload and one user-run evidence download; the agent verifies hashes and owns
the remaining control-plane work. See
ADR-20260721T020849Z-local-cuda-first-mira-campaigns.

Each cloud run must save config, git commit, dataset manifest hash, backend,
GPU name/VRAM, peak allocation, examples/second, elapsed time, billed minutes,
loss curves, canonical raw/EMA hashes, and the fixed visual gallery before the
instance is stopped. Checkpoints are downloaded before deletion.

## Estimated implementation scope

Research quality is evidence-driven. These are remaining agent-token ranges,
not calendar estimates:

- L2 dataset, metrics, optimization, and cloud campaign: 30k–100k tokens;
- L3 sphere scheduling, seam consensus, and package integration: 40k–120k;
- L4 close reconstruction and cache evidence: 25k–70k;
- L5 Hapke/material styling and visual iteration: 25k–80k;
- L6 acceptance audit and final tuning: 10k–35k.

A compelling static in-game Mira is required before the seam-free whole-body
package can be declared production-ready.

## Evidence ledger

Runs are local Burn 0.21 WGPU and cost $0 unless the row says otherwise.

| Run | Corpus / architecture | Result | Gate |
|---|---|---|---|
| `mira_l1_overfit_v0` | one synthetic train patch; four-layer tracer; 500 batches | loss 0.9702→0.0149; 23.75 m reconstruction RMS; exact deterministic repeat; recognisable crater reconstruction | plumbing proof |
| `mira_l1_kaguya_overfit_v0` | one Kaguya train patch; four-layer tracer; 1,000 batches | loss 0.9707→0.0143; **7.62 m RMS**, 41.77 m max; exact repeat; inspected target/coarse/generated/error sheet | **L1 pass** |
| `mira_l2_kaguya_pilot_v0` | 23 Kaguya train / 5 Copernicus validation; four-layer tracer; 900 batches | loss 0.9873→0.0292; 18.97 m RMS; structured held-region error remains | L2 fail |
| `mira_l1_kaguya_unet_overfit_v1` | one Kaguya train patch; 2-level residual U-Net; 500 batches | loss 0.9672→0.0164; 12.61 m RMS | architecture smoke pass |
| `mira_l2_kaguya_unet_pilot_v1` | 23 Kaguya train / 5 Copernicus validation; residual U-Net; 450 batches | loss 0.9976→0.0904; 31.75 m RMS; visible high-frequency noise | L2 fail; tune/data before spend |
| `mira_l2_kaguya_expanded_pilot_v2` | 150 geographically separated Kaguya train / 5 Copernicus validation patches; residual U-Net; 570 batches | loss 0.9976→0.0375; **24.77 m RMS**; slope, structure-function, and crater proxies approach the target, but the visual sheet remains over-sharpened | L2 fail; longer/tuned CUDA campaign justified |
| `mira_l2_kaguya_cuda_v3` | same 150/5 corpus and U-Net; native CUDA on 1× RTX A6000; epsilon target; 100-step β=0.0001→0.2 schedule; 120 epochs / 2,280 batches | 328.79 s (**54.75 examples/s**), loss 0.9987→0.0095; 552 MiB sampled peak VRAM, 60.4% mean / 88% peak utilization; EMA hash `e8cce493…f2c7db0`; **104.93 m RMS**, generated median slope 54.33 vs target 16.58, spectrum −2.05 vs −3.68; inspected gallery is diagonal high-frequency noise | **L2 fail**; estimated cloud cost ~$0.45 including recovery |
| `mira_l2_kaguya_cuda_velocity_v4_local` | exact v4 config/corpus/seed rerun on the persistent local RTX 4070 Ti (native CUDA; NVRTC + cudart headers installed from the official redistributables) | 264.36 s (**69.0 examples/s** — faster than the A6000's 62.65), loss 0.30370→0.011889; 574 MiB observed VRAM, 86–88% utilization; EMA hash `a8290622…9e8b138a8`; **26.62 m RMS**, generated median slope 23.05 vs target 16.58, spectrum −2.83 vs −3.68; repeat determinism delta 0.0 | local backend parity proven; the 4070 Ti is the pilot baseline |
| `mira_l2_kaguya_cuda_velocity_v4` | v3-controlled 150/5 corpus, U-Net, schedule, seed, and 120 epochs; velocity is the only model-contract change | 291.15 s (**62.65 examples/s**), loss 0.3037→0.01094; 552 MiB peak VRAM, 77.7% active mean / 91% peak utilization; EMA hash `5bbc245c…6bd231d`; **26.67 m RMS**, generated median slope 23.01 vs target 16.58, spectrum −2.83 vs −3.68; morphology returned but the inspected gallery has dense worm-like high-frequency texture | **L2 fail**; terminal-SNR diagnosis validated, architecture reopened |

| `mira_l2_kaguya_cuda_resize_v5a` | v4-controlled on the local 4070 Ti; the only model-contract change is `upsampling = "resize"` (nearest ×2 from reshape/repeat primitives + 3×3 conv replacing each stride-2 `ConvTranspose2d`) | 316.49 s (57.63 examples/s), loss 0.30375→0.0098743; EMA hash `f0c10a85…8eb37ef0`; **25.18 m RMS**, generated median slope 22.44 vs target 16.58, spectrum −2.87 vs −3.68, finest structure band 30.83 vs 20.67 m; repeat delta 0.0; the held gallery's worm texture is **visually unchanged, in the same locations** as v4's under the same sampling seed | **L2 fail**; transposed-conv/checkerboard hypothesis falsified as the primary cause — the artifact is seed-locked residual sampling noise, not upsampling-born |

| `mira_l2_kaguya_cuda_fourier_v5b` | v4-controlled on the local 4070 Ti; the only model-contract change is `time_conditioning = "fourier"` (the broadcast normalized-time input plane is zeroed and a Fourier-feature MLP timestep embedding is injected as per-channel biases into all five U-Net levels) | 295.12 s (61.8 examples/s), loss 0.30302→0.010993; EMA hash `e96ce198…ce7a23a5`; **19.24 m RMS** (best ever; prior best v2 24.77), generated median slope 17.00 vs target 16.58, spectrum −3.39 vs −3.68, structure functions matching at every band (finest 20.74 vs 20.67 m); repeat delta 0.0; the held gallery loses the worm texture — crater morphology reads cleanly | **timestep-conditioning hypothesis confirmed as the primary artifact cause**; L2 gate candidate |

| `mira_l2_kaguya_cuda_fourier_resize_v6` | controlled combination of both v5 changes (fourier time + resize upsampling), otherwise v4-identical | 327.25 s (55.74 examples/s), loss 0.30338→0.012300; EMA hash `acfde294…6412536f`; 19.53 m RMS, slope median 17.07, spectrum −3.37; crater proxies 369/104/20/10 vs target 345/99/18/9; repeat delta 0.0 | ties v5b within noise — resize adds nothing once time conditioning is fixed; **v5b (fourier + transposed) is the winning architecture** |

### v3/v4 failure diagnosis — resolved by v5

Increasing deterministic DDIM sampling from 25 to 100 steps left the same
noise pattern and only changed RMS 104.93→96.09 m and spectrum −2.05→−2.10,
ruling out coarse sampler discretisation as the primary cause. A targeted
terminal-SNR trace showed the epsilon model's first reconstructed clean sample
at 70.1 RMS / 264.5 max normalized units; the state grew to 2.7 RMS and 9.0% of
final pixels reached the ±2.5 output clamp. Near-zero terminal SNR therefore
fixed train/start distribution alignment but exposed the ill-conditioning of
epsilon-to-clean reconstruction (`1 / sqrt(alpha_bar)`).

`DiffusionPrediction::Velocity` is now the one shared training/sampling
contract. Its near-zero-SNR round-trip unit test and 128×96 CPU overlap smoke
pass. Controlled v4 changed only that target. It restored recognisable
Copernicus morphology and reduced RMS from 104.93 to 26.67 m, confirming the
epsilon terminal-SNR diagnosis, but retained a dense spatially periodic texture
and did not beat v2's 24.77 m. More epochs or paid scaling are not justified.

The remaining failure fingerprint pointed first at the stride-2 transposed
convolutions, with the single broadcast normalized-time channel as the second
hypothesis. The 2026-07-21 local ablations tested them separately and inverted
that ranking: **v5a** (resize+conv upsampling, all else v4-controlled) left the
worm texture visually unchanged in the same seed-locked locations (25.18 m RMS
— marginal), falsifying transposed convolution as the primary cause; **v5b**
(Fourier timestep embedding replacing the broadcast plane, all else
v4-controlled) removed the artifact and repaired every metric family at once
(19.24 m RMS, slope median 17.00 vs target 16.58, spectrum −3.39 vs −3.68).
The broadcast plane starved the network of usable timestep information — one
low-amplitude channel among eleven — so the denoiser applied a nearly
time-independent filter and left structured residual noise at low-noise steps.
Future Thunder campaigns use manual `tnr scp` ingress/egress; the agent handles
only the control plane.

## Immediate execution queue

1. ~~Local 4070 Ti CUDA baseline~~ — **done 2026-07-21**
   (`mira_l2_kaguya_cuda_velocity_v4_local`: 69.0 examples/s, fingerprint
   reproduced; NVRTC/cudart/crt redistributables documented in the trainer
   README).
2. ~~Isolated v5 ablations~~ — **done 2026-07-21**: v5a falsified transposed
   convolution; v5b confirmed timestep conditioning and removed the artifact
   (19.24 m RMS); the v6 combination showed resize adds nothing. **The
   Fourier-time / transposed-conv contract (v5b) is the architecture going
   forward.**
3. Complete the remaining L2 gate evidence on the v5b contract: fixed-seed
   train/validation/holdout galleries at several diffusion steps, a
   memorisation check against the nearest real training crops, and the
   residual fine-stipple assessment (the generated field is morphology-true
   but retains mild high-frequency stipple; spectrum −3.39 vs target −3.68).
4. Add the disjoint SLDEM macro strips before L3 sphere scheduling, preserving
   Tycho and Copernicus as untouched acceptance data.
5. Bake the winning checkpoint sphere-natively, then advance through the
   package, close-detail/cache, Hapke, and acceptance gates in order. **First
   L3 scouting render landed 2026-07-21**: the `sphere-preview` command
   (trainer README) ran the v5b checkpoint over all six Mira package macro
   faces (225 windows/face, 278 s local) and produced the first whole-body
   equirect + orthographic full-moon renders
   (`terrain_runs/mira_l2_kaguya_cuda_fourier_v5b_sphere_preview/`); authored
   basin/rille/crater structure survives and the learned band reads as
   coherent crater-field relief. It is a stylized preview, not a bake — its
   documented approximations (independent faces, pinned scale condition,
   proportional re-dimensionalization, mare proxy) are the L3 work list.
