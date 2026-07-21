# CLOUD-0 baseline and probe guide (2026-07-20)

This is the measured baseline for the planet-scale cloud program in
`docs/clouds.md`. It records the current renderer before its ownership,
targets, density, lighting, or orbital representation are replaced. The
headless probes are intended to survive those replacements so later phases can
compare like-for-like views and budgets.

Status: complete. The agent-verifiable CLOUD-0 criteria landed on
`codex/cloud-0`; on 2026-07-20 the user compared the result with a
Blackrack/KSP reference set and judged the current renderer uniformly mediocre.
That verdict closes the user-verifiable step and prioritizes structural
morphology, cloud-regime variety, atmosphere-coupled lighting, deep
self-shadowing, and surface-to-orbit continuity over further parameter tuning.

## Probe contract

Five named headless presets exercise distinct regimes:

| Preset | Camera | Site lighting | Primary evidence |
|---|---:|---:|---|
| `cloud-runway` | 850 m AGL at the real spaceport | simulation epoch | ground relationship, silhouette, depth occlusion |
| `cloud-cruise` | 4,600 m AGL | 35° local sun | deck top, repetition, horizon |
| `cloud-interior` | 2,650 m AGL | 35° local sun | inside-shell extinction and reconstruction |
| `cloud-limb` | 200 km AGL, tangent to the surface horizon | 3° local sun | near/orbit handoff and atmospheric composition |
| `cloud-planet` | ~14,000 km AGL, look at body centre (full disc) | 42° local sun | CLOUD-6 orbital impostor weather across the whole planet |
| `cloud-sunset` | 700 m AGL, looking toward the sun | 1° local sun | phase response and atmosphere-coupled colour |

The sun-relative presets select a deterministic body-fixed surface point whose
local solar elevation matches the request. They do not change canonical
simulation time. All presets use the real `ShipCamera`, scene depth,
atmosphere, post stack, and cloud/sky composite.

Run the default suite with:

```powershell
just cloud-baseline
```

Or run a single probe and override it without recompiling:

```powershell
$env:THALOS_SCREENSHOT='cloud-limb'
$env:THALOS_SCREENSHOT_SIZE='2560x1440'
$env:THALOS_SCREENSHOT_CLOUD_QUALITY='high'
$env:THALOS_SCREENSHOT_CLOUD_TEMPORAL='off'
just screenshot cloud-limb
```

Available cloud overrides are camera altitude, look elevation, local sun
elevation, coverage, quality (`low`, `baseline`, `high`, `reference`), warmup,
and temporal history. Every capture writes a PNG plus a same-named JSONL report
unless `THALOS_SCREENSHOT_OUT` / `THALOS_SCREENSHOT_REPORT` redirect them.

The quality ladder is a CLOUD-0 diagnostic projection over the current
renderer: view/shadow steps are 36/2, 60/4, 72/6, and 96/8. Baseline's 60 view
steps preserve the shader's previous hard-coded cap. This is not the scalable
viewport-relative ladder planned for CLOUD-2. Temporal-off now gates both the
static same-pixel history and the moving-camera history; previously the latter
remained at 90% even when reprojection strength was zero.

## Development hardware and method

- GPU: NVIDIA GeForce RTX 4070 Ti, 12,282 MiB
- Driver: 610.62
- Backend: Vulkan
- Build: optimized development profile, Bevy 0.19 / wgpu 29
- Timing: Bevy render timestamp queries around the cloud compute pass; each
  row summarizes the last 120 warmed frames
- Persistent memory: exact byte counts derived from the texture descriptors,
  excluding transient command/bind-group overhead and the screenshot target

The output and history remain fixed at 1920×1080 even for a 2560×1440
viewport. Consequently, viewport size changes composite/readback cost but not
cloud-march pixel count or persistent cloud texture memory.

## GPU baseline

| View | Viewport | Quality | Temporal | GPU mean | GPU p95 |
|---|---:|---:|---:|---:|---:|
| Runway | 1920×1080 | baseline (60/4) | on | 9.571 ms | 10.620 ms |
| Cruise/deck | 1920×1080 | baseline (60/4) | on | 4.342 ms | 4.347 ms |
| Interior | 1920×1080 | baseline (60/4) | on | 2.781 ms | 2.789 ms |
| Limb | 1920×1080 | baseline (60/4) | on | 4.361 ms | 4.370 ms |
| Sunset | 1920×1080 | baseline (60/4) | on | 10.431 ms | 10.463 ms |
| Limb | 2560×1440 | baseline (60/4) | on | 4.358 ms | 4.366 ms |
| Interior diagnostic | 1920×1080 | baseline (60/4) | off | 2.786 ms | 2.794 ms |
| Sunset | 2560×1440 | high (72/6) | on | 11.062 ms | 11.108 ms |

The equal 1080p/1440p limb timings confirm that the pass is viewport-independent
today. The current 1440p High sunset cost is 3.16× the provisional ≤3.5 ms
target. Even the no-visible-cloud limb pass costs 4.36 ms, so CLOUD-2 needs to
reduce scheduled pixel work and cheaply reject empty/distant rays rather than
only tune density evaluations.

## Persistent cloud texture memory

| Allocation | Format / extent | Bytes | MiB |
|---|---|---:|---:|
| Current colour | RGBA32F, 1920×1080 | 33,177,600 | 31.64 |
| Current distance | R32F, 1920×1080 | 8,294,400 | 7.91 |
| History colour | RGBA32F, 1920×1080 | 33,177,600 | 31.64 |
| History distance | R32F, 1920×1080 | 8,294,400 | 7.91 |
| Base-noise atlas | RGBA32F, 1920×1920 | 58,982,400 | 56.25 |
| Worley detail | RGBA32F, 32³ | 524,288 | 0.50 |
| Coverage map | R8, 512×256 | 131,072 | 0.13 |
| **Total** | | **142,581,760** | **135.98** |

The off-screen screenshot target adds 7.91 MiB at 1080p or 14.06 MiB at
1440p, but is probe infrastructure rather than cloud-renderer ownership.

## Artifact inventory

1. **Tiling / dimensionality.** Runway and interior views show repeated tall
   curtains and mushroom-like columns. The 2-D coverage field is being
   extruded through a thin fixed shell, while the square base atlas repeats in
   planet space; it does not read as a varied 3-D cloud volume.
2. **Horizon and orbital range.** The cruise deck becomes a bright, nearly
   uniform horizon band. At 200 km the volume disappears completely in both
   1080p and 1440p probes: `MAX_CLOUD_DIST = 25 km` rejects the shell long
   before an orbital tangent ray reaches it. The pass still costs ~4.36 ms.
3. **Temporal fringe.** Temporal-on softens the pattern but leaves a visible
   ordered screen-door fringe along cloud edges. Temporal-off makes the same
   interleaved/dither structure stark, confirming that history hides rather
   than reconstructs the missing samples robustly.
4. **Depth crossing.** Static runway terrain correctly occludes the cloud
   composite; no through-ground leak was observed. Craft/cloud disocclusion,
   fast camera motion, and timewarp rejection still require the interactive
   verification session because a settled still cannot validate them.
5. **Near/orbit mismatch.** Near views contain bright volumetric forms while
   the orbit view contains none. There is no overlapping reduced-detail or
   optical-depth representation, so a continuous spaceflight handoff is not
   currently possible.
6. **Lighting mismatch.** Sunset clouds remain clipped white/grey instead of
   acquiring atmosphere-filtered warm direct light and sky-derived ambient.
   This is direct visual evidence of the private cloud-lighting inputs called
   out in the architecture plan.
7. **Interior response.** The nominal in-shell camera can sit in a coverage
   gap with clear-sky background and nearby opaque curtains; there is no
   convincing continuous extinction/fog response around the camera.

Generated captures and reports live under `tools/screenshots/cloud0/` and are
ignored by Git by design. The preset definitions, machine-readable report
schema, exact memory accounting, and this inventory are the durable baseline.

## 2026-07-21 CLOUD-2/3 checkpoint delta

The first fidelity checkpoint deliberately keeps the CLOUD-0 captures above as
the immutable before-state. Its matched five-view captures retain the same
paths under `tools/screenshots/` and report:

| View | Internal target | Quality | GPU mean | GPU p95 |
|---|---:|---:|---:|---:|
| Runway | 1280×720 | baseline (80/3) | 8.344 ms | 8.381 ms |
| Cruise | 1280×720 | baseline (80/3) | 1.198 ms | 1.209 ms |
| Limb | 1280×720 | baseline (80/3) | 4.714 ms | 4.733 ms |

Persistent allocation is 42,631,168 bytes (40.66 MiB): two RGBA32F colour
targets and two R32F distance targets at 1280×720, one RGBA32F 64³ volume, and
the RGBA8 256²×6 weather cube. This is a 70.4% reduction from CLOUD-1's
137.35 MiB despite increasing the 3-D basis resolution.

## 2026-07-21 CLOUD-2/3 completion

The completed path sizes all four view/history targets from the physical
viewport, updates one rotating 3×3 pixel class per valid-history frame, rejects
history across camera/FOV/body/weather/simulation/target discontinuities, clamps
reprojected radiance to a 3×3 current neighborhood, and reconstructs with
bilinear history plus hit-aware full-resolution filtering. Reference mode is
full-resolution, full-frame, and temporal-off.

Matched corrected Baseline captures at a 1920×1080 viewport use a 1280×720
cloud target and one stable lobe-scale directional shadow probe:

| View | GPU mean | GPU p95 |
|---|---:|---:|
| Runway | 2.077 ms | 2.406 ms |
| Cruise | 1.011 ms | 1.166 ms |
| Interior | 0.256 ms | 0.259 ms |
| Limb | 0.877 ms | 1.038 ms |
| Sunset | 2.184 ms | 2.511 ms |

The budget gate is the densest sunset probe at 2560×1440 High: a 1712×960
cloud target, 96 view steps, one shadow probe, **2.471 ms mean / 2.476 ms p95**,
and 71,507,968 bytes (68.20 MiB) persistent allocation. This is inside the
provisional 3.5 ms High target; the old fixed-target High sunset was 11.06 ms.

CLOUD-3 retains typed stratus/cumulus/storm profiles, multi-domain 64³
base/detail noise, boundary erosion, and continuous weather sampling. The first
completion attempt added height remapping, a mid-scale formation domain, range
step stretching, and weather/base-top/broad-occupancy leaps. Matched A/B showed
that the heuristic leaps posterized shallow rays into stable horizontal strata,
while the other additions left the hierarchy-free density softer than the last
organic checkpoint. That path was removed rather than tuned around
([INC-0008](incidents/0008-cloud-hierarchy-resume-strata.md)). The corrected
range path fades only sub-pixel fine erosion from 10–22 km and reuses the 21.6
km macro modulation once per short ray; neither changes sample positions. Any
future empty-space leap requires a true max-density bound over the skipped
interval.

## Interactive regression checklist

These motion checks remain the interactive acceptance pass for the completed
CLOUD-2/3 renderer, but do not block the agent-verifiable phase. Run `just game cruise` and `just game orbit`,
then compare:

- edge screen-door/shimmer during pitch and roll;
- vertical curtain repetition and flat cloud bases;
- cloud disappearance during climb to orbit;
- white/grey sunset response;
- craft/cloud depth crossing and disocclusion;
- history behavior under camera cuts and timewarp.
