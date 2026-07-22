# Planet-scale volumetric clouds

**Status:** active program, 2026-07-22. CLOUD-0 through CLOUD-3 are complete;
CLOUD-4 and CLOUD-6 have second slices landed and headless-verified (see the
2026-07-22 checkpoint below).
This document is the strategy and technical plan;
[backlog.md](../backlog.md) is the execution queue, while
[atmosphere.md](atmosphere.md) remains the spec for what the renderer ships
today. Architecture choices are fixed by
[ADR-20260720T212214Z-one-weather-field-many-cloud-projections](../adr/20260720T212214Z-one-weather-field-many-cloud-projections.md).

The target is a Blackrack-class cloud system for a surface-to-orbit flight
camera: shaped volumes that can be entered, stable planetary weather seen from
orbit, cloud shadows and aerial shafts that agree with the visible volume, and
one atmosphere/light environment shared with terrain, craft, structures, and
water.

Planning input: the supplied *Planet-scale volumetric clouds for spaceflight*
note, backed by the public Nubis/Horizon Zero Dawn, Nubis Evolved, Frostbite,
Hillaire, Skybolt, and Blackrack feature-level references named there.

### Visual target (user references, 2026-07-20)

The user-supplied Blackrack/KSP reference captures are the program's visual
bar. They do not merely show denser white noise. Their defining qualities are:

- **Recognizable regimes in one sky:** scattered cumulus, congestus and
  cumulonimbus towers, broad anvils, layered stratus/overcast, and thin shelf
  structure coexist at distinct heights and scales.
- **Weather-system coherence:** kilometre-tall towers gather into fronts and
  storm systems that remain legible across a runway, an aircraft-scale scene,
  the horizon, and the planet. Individual lobes never read as repeated sprites
  or vertically extruded coverage texels.
- **Convincing volume:** crisp sunlit cauliflower detail transitions into soft
  wisps, deep blue-grey self-shadow, dark rain-bearing cores, continuous
  interior extinction, and occasional diffuse precipitation/virga shafts.
- **Atmospheric light:** cloud light follows the sky and air around it — cool
  shadow fill at noon, warm low-sun transmission and rims, aerial recession,
  and energy-bounded bright regions rather than clipped white slabs.
- **One scene:** terrain, craft, water, atmosphere, and clouds agree on depth,
  sun visibility, ambient response, reflections, and scale. Flying beside or
  through a cloud must feel like occupying the same world, not compositing a
  sky effect behind it.

These are acceptance properties, not a mandate to reproduce a particular mod's
assets or exact art direction. Morphology, scale hierarchy, lighting, temporal
stability, and surface-to-orbit continuity take precedence over matching a
single still image.

## 1. What Thalos already has

This is an upgrade, not a greenfield renderer. Preserve these foundations:

- A body-fixed, planet-centred spherical-shell march now owned by
  `thalos_body_render::clouds`; it already handles cameras below, inside, and
  above the layer without a flat-sky assumption.
- `big_space`-safe camera inputs and planet-fixed noise/weather coordinates.
- An authored `CloudClimate` and per-body `CloudWeatherField`, projected as a
  1024×1024×6 RGBA cubemap carrying coverage/type/base/top, plus an
  8-level footprint-filtering mip chain.
- Per-pixel cloud hit distance and deterministic composition through one
  fullscreen `CloudCompositeMaterial`, including opaque-scene occlusion and
  independence from the atmosphere material's cloud ownership.
- Static-view sparse accumulation plus fresh full-pixel raymarching during
  camera motion; body-fixed reprojection only stabilizes a freshly marched
  moving result.
- A first orbital projection in `SolidPlanetMaterial` sampling the same
  weather cubemap as the near renderer.

The current result falls short for structural reasons:

1. **CLOUD-2 reconstruction is complete.** Viewport-relative targets, a
   screen-static rotating 3×3 sparse march, full current-ray marching during
   camera motion, body-fixed hit-aware history stabilization, discontinuity
   rejection, neighborhood clamp, bilinear history, and hit-aware
   reconstruction form one scalable path. A single nearest-cloud depth must
   never substitute old radiance for an untraced moving-view pixel: it cannot
   represent a translucent interval (INC-0016). The far projection's remaining
   offline atlas/reduced limb work belongs to CLOUD-6, not temporal
   reconstruction.
2. **The density foundation is now genuinely volumetric.** The extruded 2-D
   atlas is gone; a 64³ Perlin/Worley basis, typed vertical profiles, local
   base/top, boundary-only erosion, a decorrelated macro threshold, and a 50 km
   spherical march produce coherent bodies from runway through cruise. Fine
   erosion fades once sub-pixel, while low-frequency macro modulation is reused
   per ray. Heuristic empty-space leaps were removed after they posterized
   grazing views (INC-0011).
3. **Clouds inhabit a private lighting universe.** The compute shader receives
   hand-scaled sun and top/bottom ambient colours. It does not consume the
   atmosphere transmittance, sky-view environment, eclipse, or the shared
   direct-sun visibility path.
4. **The cost model is now scalable.** Low/Baseline/High/Reference targets are
   viewport-relative, with 1-in-9 sparse scheduling outside Reference. The
   worst 1440p High probe (sunset, 1712×960 cloud target) is 2.471 ms mean /
   2.476 ms p95 on the development RTX 4070 Ti, inside the provisional 3.5 ms target;
   Baseline at 1080p retains 71.16 MiB of persistent cloud textures after the
   BL-33 weather-resolution increase; the cloud pass remains below its timing
   budget in all five accepted cold probes.
5. **Visible clouds do not interact with the world.** The terrain, craft,
   structures, atmosphere, and reflection environment do not receive one
   cloud-transmittance field. The impostor's shadow probe is an unrelated
   approximation.

## 2. Architecture invariants

The cloud program extends the graphics plan's one-world principle:

1. **One authored climate, one runtime weather field.** A per-body
   `CloudClimate` in `thalos_world` supplies stable presets. A per-body
   `CloudWeatherField` in simulation/environment state is the mutable source of
   coverage, type, base, top, precipitation/storm potential, and wind. No
   renderer owns an independent weather pattern.
2. **Many projections, one density definition.** Near volume, orbital layer,
   sun-transmittance field, and authoring preview all evaluate the same
   `cloud_density(position, weather)` function from a shared WGSL library (with
   a small CPU mirror only where tools require it). Representation may change
   with projected footprint; weather authority never does.
3. **Clouds are participating media inside the atmosphere.** Direct light uses
   the shared `SceneLighting` sun/moon/eclipses and atmosphere transmittance.
   Cloud radiance is attenuated from sample to camera; atmosphere in front of a
   cloud is not incorrectly attenuated by cloud behind it.
4. **One cloud-occlusion field feeds every consumer.** Terrain, foliage, craft,
   structures, water, the atmosphere march, and environment lighting sample a
   shared `CloudSunTransmittance` projection. A second screen-space godray mask
   that can disagree with the visible volume is not allowed.
5. **Cloud temporal reconstruction is view-local.** It uses body-fixed cloud
   depth/motion and is independent of whole-scene TAA (W13). Map and ship views
   may choose different projections of the same field.
6. **No hard altitude pop.** Near volume, reduced-detail volume, and orbital
   projection overlap and crossfade using projected cloud-texel footprint and
   confidence, not a single magic camera-distance branch.
7. **Optional means optional.** A body with no authored cloud climate does not
   silently receive default clouds. Quality settings alter projection cost, not
   the body's climate.

### Proposed ownership

```text
thalos_world::CloudClimate (authored RON)
                    |
                    v
game::CloudWeatherField (per-body, simulation-time state)
                    |
        +-----------+------------+----------------+
        |                        |                |
        v                        v                v
near/mid view march      orbital optical-   CloudSunTransmittance
                        depth/normal atlas   (near cascades + globe tail)
        |                        |                |
        +------------------------+----------------+
                                 |
                                 v
        CloudComposite + shared lighting/shadow/environment consumers
```

Render mechanism belongs under `thalos_body_render::clouds`; the game-side
driver only selects bodies/views and projects simulation state into render
inputs. The current vendored crate should be absorbed during the ownership
slice, retaining its MIT attribution and license, rather than growing a second
top-level rendering subsystem beside `thalos_body_render`.

## 3. Data and rendering model

### 3.1 Authored climate and runtime weather

The former `CloudCover`/constant split is replaced by `CloudClimate`. Its
shipping Rust layout covers:

- seed and mean coverage;
- permitted altitude range and type mix;
- wind/advection parameters in body-fixed metres per second;
- extinction, single-scatter albedo, phase controls, and precipitation/storm
  thresholds;
- per-body quality-neutral scale ranges (weather, base shape, erosion detail).

The first runtime field should be a seam-safe cube/2-D-array field rather than a
new lat-long-only contract. Each texel carries at least coverage, cloud type,
base height, and top height. A CPU-generated field is sufficient initially;
later advection or a GPU weather solver can replace the producer without
changing consumers. Equirectangular import/export remains useful for painted
maps and tools.

Do not start with a full fluid simulation. Slowly advected fields plus analytic
front/cyclone stamps create far more visible structure per unit of complexity.

### 3.2 Density

The first high-fidelity density domain is one spatially varying water-cloud
shell whose local base, top, and vertical profile blend can express stratus,
cumulus, and cumulonimbus. A thin cirrus layer may later be another projection
of the same weather field; it must not acquire independent weather authority.

Density combines:

- planetary weather coverage/type/base/top;
- a true low-frequency 3-D Perlin-Worley base volume;
- higher-frequency Worley erosion/detail;
- type-specific vertical profiles and anvil shaping;
- rotated/offset octave domains and a second noise space to suppress tiling.

Use filterable compact formats for generated noise. The former 1920² RGBA32F
base atlas has been removed; later format work should compact the retained 64³
RGBA32F basis. A future empty-space hierarchy is optional and may leap only
from a conservative density bound over the skipped interval
([ADR-20260721T033055Z-cloud-skips-require-conservative-bounds](../adr/20260721T033055Z-cloud-skips-require-conservative-bounds.md)).

**Physical-scale invariant:** an authored feature scale names the feature, not
the full period of a stored tile containing several cells. The current erosion
channel contains eight cells per volume axis, so its sampled tile period is
`detail_scale_m * 8`. Treating `detail_scale_m` itself as the period creates
~56 m cells from the authored 450 m scale, below the 200–500 m horizon step and
recurs as stipple/micro-cloudlets; see [INC-0010](../incidents/0010-cloud-detail-period-eighth-scale.md).

### 3.3 View march and reconstruction

Keep the exact spherical shell intersection. Replace the fixed 25 km reach with
regime-aware sampling:

- fine steps and full detail near/inside cloud;
- the same stable adaptive cadence with fewer detail octaves through the distant deck;
- empty-space leaps only where max-density data conservatively prove the full segment clear;
- early exit on optical depth;
- orbital projection once a volume sample is sub-pixel.

Render into viewport-relative internal targets. A screen-static view uses a
rotating 3×3 topology (one ninth of full pixel work per frame), then
same-pixel accumulation, neighborhood clamp, and bilateral full-resolution
reconstruction. During camera motion every pixel must raymarch its current
world ray; body-fixed reprojection may stabilize that fresh result only after
coherent colour/distance selection and old-camera depth validation. A single
nearest-cloud depth is not a conservative reconstruction of the full
translucent ray integral, so history cannot stand in for untraced current
radiance (INC-0016). History rejection must cover camera cuts, FOV/resolution
changes, body switches, disocclusion, weather-version changes, large wind
displacement, and timewarp jumps. A screenshot mode renders all samples at high
step counts with temporal reuse disabled.

The existing cloud-local reprojection is the seed for this work. It is not
blocked on W13's whole-scene TAA decision.

### 3.4 Lighting and atmosphere composition

The cloud shader imports shared atmosphere/light helpers rather than receiving
artist-authored `sun_color` and top/bottom ambient colours:

- sun-to-sample atmosphere transmittance and eclipse state;
- sample-to-camera atmosphere transmittance;
- dual-lobe phase and Beer extinction;
- powder/dark-core shaping;
- energy-conserving multiple-scatter octaves or an equivalent Nubis-style
  approximation;
- atmosphere-derived sky ambient, with local overcast modulation.

The compute output needs enough depth information to split the atmosphere
composite into foreground and background segments (at minimum first hit plus an
optical-depth centroid or back depth). The current `cloud over fully integrated
air` ordering is an approximation: it attenuates some foreground air as if it
were behind the cloud. The replacement contract must make the ordering
explicit before lighting polish begins.

**Slices landed (2026-07-21):** near-volume march applies analytic sun and
sample→camera transmittance from the shared authored atmosphere coefficients.
The former direct dependency on Bevy's private atmosphere LUT resources was
deleted with that renderer; binding the shared Thalos sky/transmittance LUT is
the remaining CLOUD-4 coupling slice. Dual-lobe multi-scatter
phase octaves, HZD powder, volumetric self-shadow, and a soft Reinhard peak sit
on that shared transport. One dedicated `CloudCompositeMaterial` now owns both
near-volume and weather-derived orbital composition, so the atmosphere
material cannot hide clouds or restore a parallel cloud path. Orbital
projection (SolidPlanet + the cloud composite) uses greyer albedo and the same
solar-elevation chromaticity so the full disc no longer clips pure white.
CLOUD-4 still needs an explicit foreground/background atmosphere split; the
current cloud-over-fully-integrated-air ordering remains an approximation.

### 3.5 Shadows, godrays, and environment response

`CloudSunTransmittance` is a cascaded field:

- a coarse body-fixed cube tail integrates vertical/slanted optical depth for
  planet-scale and orbital views;
- one or more view-anchored, sun-aligned near cascades integrate the actual
  density field for runway/aircraft-scale correspondence;
- cascades update incrementally and temporally because cloud shadows are soft.

The shared surface direct-light helper multiplies hard-object shadow, terrain
horizon visibility, eclipse, and cloud transmittance. The atmosphere march
samples the same field for crepuscular shafts. Local overhead optical depth also
modulates the atmosphere-derived ambient/IBL projection so a craft under solid
overcast does not retain a sunny belly.

A coverage-map-only shadow is acceptable as an early debug projection, not as
the finished W2 implementation: it cannot match convective height or the godray
field.

### 3.6 Orbital representation

The far path is a slowly refreshed optical-depth/albedo/normal/height-moment
cubemap derived from the same weather and density basis. It replaces the
hand-picked reference images. The normal and height moments keep lighting and
the limb silhouette coherent with the last volumetric regime. Low orbit may
blend the atlas with a reduced-detail limb march; far orbit/map view uses the
atlas alone.

The transition is tested in motion in both directions. Surface shadow and orbit
colour must not swap authority at the transition.

**First slice (2026-07-21):** until the offline atlas exists, both
`SolidPlanetMaterial` and the BodySky orbital path derive column moments live
from the canonical weather cubemap via shared helpers in `thalos::atmosphere`
(`weather_column_from_texel`, `orbital_cloud_altitude`, `orbital_cloud_shade`,
`sample_weather_soft`, `orbital_cloud_normal_body`). That gives height parallax,
optical-depth self-shadow, soft neighborhood filtering (kills 256² face
squares), and one shading contract for terrain-LOD and far-impostor views. A
later producer can bake the same moments into an atlas without changing the
consumer API.

## 4. Implementation phases

| ID | Phase | Outcome | Depends on | Est. |
|----|-------|---------|------------|------|
| CLOUD-0 | Baseline, probes, and budgets | Reproducible current captures, GPU/memory baseline, cloud-specific headless presets, acceptance matrix | — | S–M |
| CLOUD-1 | Canonical ownership and schema | `CloudClimate` + per-body `CloudWeatherField`; `None` is authoritative; mechanism moves under `body_render`; legacy slab/reference ownership deleted; first weather-derived orbit layer | CLOUD-0 | M–L |
| CLOUD-2 | Scalable targets and temporal reconstruction | Viewport-relative low-res targets; screen-static interleaved march; fresh full-pixel motion march with hit-aware body-fixed stabilization; robust history rejection/clamp/upscale; screenshot mode; quality ladder | CLOUD-1 | L |
| CLOUD-3 | Multi-scale density and range | Weather cube with type/base/top, true 3-D base/detail noise, vertical profiles, anti-tiling, cadence-safe detail LOD and low-frequency reuse | CLOUD-2 | L |
| CLOUD-4 | Atmosphere-coupled lighting | Shared sun/eclipses/atmosphere/sky inputs, powder + multiple scattering, correct foreground/background media ordering | CLOUD-3 · F3/F4 substrate | L |
| CLOUD-5 | One-world interactions | Cascaded cloud-sun transmittance, all surface receivers, matched godrays, overcast ambient/IBL response | CLOUD-4 · F6 shadow substrate | L |
| CLOUD-6 | Full orbital projection and seamless handoff | Density-derived optical-depth/normal moments, reduced-detail limb regime, invisible surface↔orbit↔map transition | CLOUD-3 | M–L |
| CLOUD-7 | Living weather and authoring | Sim-time advection/growth/decay, front/cyclone stamps, per-body presets, paint/import/debug tools | CLOUD-5 · CLOUD-6 | L |
| CLOUD-8 | Inside-volume and storm polish | Interior extinction, precipitation/rain shafts, lightning/emissive path, final quality tuning | CLOUD-4 · CLOUD-7 | M–L |

Recommended execution order is CLOUD-0 → CLOUD-1 → CLOUD-2 → CLOUD-3, then
CLOUD-4 and CLOUD-6 can proceed against the same density contract; CLOUD-5
follows lighting, and weather/polish come last. This prevents expensive visual
work from landing on the fixed-resolution or split-authority paths that must be
removed.

### CLOUD-0 exit criteria

Agent-verifiable:

- Headless presets for broken-cloud runway, cruise/deck, cloud interior,
  low-orbit limb, and sunset; each can override camera altitude, sun angle,
  quality, warmup, and temporal-disabled screenshot mode.
- A repeatable GPU timing and render-target memory report for the cloud pass at
  1080p and 1440p; provisional High target ≤3.5 ms at 1440p on the development
  GPU, adjusted after measurement rather than treated as dogma.
- Baseline captures and a short artifact inventory: tiling, horizon fade,
  temporal fringe, depth crossing, and near/orbit mismatch.

User-verifiable:

- One short current-build session confirms which visual failures are most
  objectionable before fidelity work is prioritized.

**Status (2026-07-20):** complete. Five regime presets,
temporal/quality/pose overrides, Vulkan
GPU timing, exact target-memory reporting, 1080p/1440p captures, and the
artifact inventory are recorded in `docs/reference/cloud_baseline.md`. The 1440p High
sunset probe measures 11.06 ms mean on the development RTX 4070 Ti versus the
provisional 3.5 ms target; persistent cloud textures total 135.98 MiB. The user
then judged the current result uniformly mediocre against the reference set,
confirming that morphology, lighting, regime variety, and scale continuity are
the priority failures. That verdict completes the user-verifiable criterion.

### CLOUD-1 exit criteria

- `CloudClimate` is the sole authored terrestrial-cloud configuration and
  `None` creates no runtime weather or visible clouds.
- Every cloudy body receives one deterministic, seam-safe
  `CloudWeatherField`; coverage, type, base, and top survive the CPU→GPU
  contract even where the current marcher does not yet consume all channels.
- The cloud mechanism lives under `thalos_body_render`; game code only selects
  the active body and projects environment state.
- Near and first orbital projections sample the same field. The reference-image
  selector and dormant `BodySky` slab are deleted.
- The workspace is compile-clean and every CLOUD-0 headless regime still
  initializes and captures without shader/pipeline errors.

**Status (2026-07-20):** complete on `codex/cloud-0`. `cargo check -p
thalos_game` and warning-denied scoped Clippy pass; the complete five-preset
headless cloud suite captured without RON, asset, WGSL, bind-group, or pipeline
failure. The richer weather cubemap increases the persistent cloud allocation
to 144,023,552 bytes (137.35 MiB). Captures still show the inherited vertical
sheet/repetition, weak mass hierarchy, over-bright transport, and poor limb
result; those are deliberately not disguised with tuning here and remain owned
by CLOUD-2 through CLOUD-6.

### First CLOUD-2/3 fidelity checkpoint

**Status (2026-07-21):** historical first checkpoint, captured and compile-clean
on `codex/cloud-0`. The completed CLOUD-2/3 measurements are recorded below and
in `docs/reference/cloud_baseline.md`.

- Cloud colour/distance/history targets moved from 1920×1080 to 1280×720;
  the 1920² base atlas was deleted, the generated volume moved 32³ → 64³, and
  persistent cloud allocation fell from 137.35 MiB after CLOUD-1 to 40.66 MiB
  (42,631,168 bytes).
- Density now uses a wrap-first trilinear 3-D Perlin/Worley volume, typed
  stratus/cumulus/storm profiles, spatial base/top, storm anvils, a second
  macro threshold domain, boundary-only 450 m erosion, and 50 km bounded
  spherical range. The aliased detail-scale failure and diagnosis are retained
  in INC-0010.
- The canonical weather cube now combines synoptic, mesoscale, and cellular
  coverage. Near volume and first BodySky/SolidPlanet orbital projections use
  that same field; the 200 km limb probe shows broken systems rather than a
  missing layer or a uniform white shell.
- Final baseline probes at 1920×1080 report 8.34 ms mean / 8.38 ms p95 for the
  densest runway view, 1.20 / 1.21 ms for cruise, and 4.71 / 4.73 ms for the
  limb on the development RTX 4070 Ti. These are checkpoint measurements, not
  the CLOUD-2 budget exit: viewport-relative targets, sparse scheduling,
  history clamp/moments, and empty-space skipping remain.
- Lighting has darker cores, deterministic sparse self-shadow, analytic shared-
  coefficient atmosphere coupling, and a first solar-elevation tint, but CLOUD-4 still
  owns correct foreground/background media ordering. CLOUD-5 still owns world
  cloud shadows. CLOUD-6's first orbital-moments slice is in (live weather-column
  projection on SolidPlanet + BodySky); the offline OD/normal atlas and
  reduced-detail limb volume remain.

### CLOUD-2/3 completion (2026-07-21)

Completed reconstruction and corrected density/range slice on `codex/cloud-0`:

- **Typed 3-D density:** stratus/cumulus/storm vertical profiles modulate a
  continuous Perlin/Worley mass field; 450 m boundary erosion cuts readable
  cauliflower detail while preserving solid optical cores.
- **Safe near→horizon LOD:** only fine boundary erosion fades when its authored
  feature is smaller than the full-density sampling footprint. The corrected
  BL-33 marcher samples smooth broad mass every 600 m, backs up one interval on
  a hit, and advances monotonically at 120 m through full density; no weather or
  profile hint is used as a skip/resume gate.
- **Low-frequency reuse:** the 21.6 km anti-tiling modulation is evaluated once
  per short view segment and reused by view/shadow density. It remains smooth
  across pixels without a redundant trilinear fetch at every 200–500 m step.
- **Conservative-skip invariant:** a marcher may leap only from a true
  max-density bound over the complete skipped interval. Weather maxima,
  broad-shape proxies, and estimated base/top crossings are correlated hints,
  not bounds; using them as resume gates produced stable height strata
  ([INC-0011](../incidents/0011-cloud-hierarchy-resume-strata.md)).
- Weather producer: slightly stronger cellular gaps (not so strong they shatter
  the limb); taller storm tops / thinner stratus decks so type channels read in
  both near volume and limb silhouettes.
- **CLOUD-2 reconstruction:** physical-viewport targets, rotating 3×3 sparse
  scheduling, body-fixed reprojection, camera/FOV/body/weather/simulation/epoch
  rejection, 3×3 neighborhood clamp, bilinear history colour/distance, and
  hit-aware full-resolution reconstruction. Low/Baseline/High/Reference is a
  real 1/2–1× viewport quality ladder; Reference disables temporal and sparse
  scheduling.
- Weather type/base/top is sampled continuously at ordinary view steps. Holding
  the tuple piecewise or advancing by proxy-driven leaps was rejected because
  it quantized grazing rays into visible distance/height slabs.
- Matched five-view Baseline re-capture at 1920×1080 / 1280×720 is clean of the
  posterized hierarchy artifact. The worst 2560×1440 High probe (sunset,
  1712×960 cloud target) is **2.471 ms mean / 2.476 ms p95**, inside the
  provisional ≤3.5 ms target.
- CLOUD-2 and CLOUD-3 are complete. CLOUD-4's shared-LUT atmosphere transport
  has landed; it retains the explicit foreground/background media split.
  CLOUD-6 retains the offline orbital atlas and
  reduced-detail limb volume.

### CLOUD-4/6 second slices (2026-07-22)

Landed together and verified across the five headless cloud presets
(planet / limb / sunset / cruise / runway, llvmpipe software Vulkan — visual
evidence only, no timing claims):

- **Lighting completion (CLOUD-4).** The single unfiltered shadow probe became
  a filtered multi-tap sun optical depth: taps sample only the smooth typed
  broad mass (`detail_weight = 0` — probing ~55 m erosion keyed the direct term
  on cellular noise and read as soot), the tap ladder is jittered per pixel
  (fixed ladders banded into strata; the near march's temporal pass absorbs
  jitter), and τ drives **per-octave** multi-scatter attenuation
  (`MS_OCTAVE_WEIGHTS`/`_EXTINCTION`) so deep shade keeps wide-lobe fill
  instead of multiplying toward charcoal. Powder away-darkening restrained
  (0.85 → 0.35). CPU sun chromaticity no longer duplicates the per-sample
  air-mass reddening. **Foreground airlight is restored analytically** in both
  estimators (the composite draws clouds over fully-integrated sky, so cloud
  opacity was deleting the air in front of the cloud — the "dirty distant
  deck"): the near march adds `airlight_radiance · (1 − T_view)`, the far
  overlay the matching analytic-β veil. Residual: replace the analytic β
  estimates with the shared Thalos sky/transmittance LUT.
- **Far estimator (CLOUD-6).** The weather cube now carries a 6-level box mip
  chain (`CloudWeatherField::rgba8_mip_chain`; built via `Image::new_uninit`
  because `Image::new`'s size assert is mip-unaware — INC-0018), and all far
  consumers fetch footprint-filtered (`sample_weather_soft(lod)` replaced the
  5-tap cross; the moment-normal stencil widens with lod). The composite's
  22–85 km altitude gate is replaced by a 6-sample **weather-column band
  march** over the ray's shell slab, in partition-of-unity with the near
  march: `march_reach` mirrors `get_ray`'s per-ray reach (keep the two in
  lockstep) and the marcher's old distance haze-out became an entry-distance
  hand-off, so far clouds are replaced, not deleted. The band march's vertical
  profile is an **analytic per-segment height-overlap integral** — its
  jittered point-sampled predecessor speckled the far disc with white-noise
  holes, because this material pass has no temporal accumulation to absorb
  per-pixel jitter (sheet dark-fleck fraction 1.30% → 0.82% after the
  integral, worst near-black minima eliminated). `weather_cloud_opacity` is
  recalibrated as an areal fraction against the near volume's formation gate;
  chords combine strongest-column + damped stacking, not independent slabs.
  Producer: band latitudes warped and gated (broken segments, not paint
  rings), zonally-elongated warped frontal ridges.
- **Known remaining weaknesses** (recorded as the CLOUD-6 residual): the far
  field at grazing regimes (limb probe, cruise horizon) still tends toward a
  slab with soft box-mip squares at coarse lod — the reduced-detail limb
  volume is the intended fix; the offline OD/normal atlas and the in-motion
  surface↔orbit handoff check (user, live session) remain open.

### BL-33 fidelity convergence checkpoint (2026-07-22)

Three independently verified slices landed; the full BL-33 item remains `wip`
because the far estimator still needs density-derived moments and a
reduced-detail limb volume.

- **Canonical weather resolution and hierarchy.** Correcting the old scale
  estimate showed that a 256 face represented about 19.5 km/texel at the
  centre of a Thalos cube face. The field is now 1024 per face (about
  4.9 km/texel), with eight RGBA8 mip levels. Mesoscale warp, warped cellular
  mass, and a second cellular-cut domain add broken fronts and gaps without
  creating an independent render-side pattern. The complete cube mip chain is
  33,553,920 bytes; total 1280×720 Baseline cloud allocation is 74,612,224
  bytes (71.16 MiB).
- **Adaptive cheap/full-density march.** Clear air evaluates only the smooth,
  typed broad mass at a 600 m cadence. A meaningful hit backs up one interval
  and refines at 120 m; four empty fine samples return to coarse mode. A
  monotonic `refined_until` frontier prevents a coarse hit from revisiting and
  reintegrating an already refined interval. The full density, lighting, and
  transmittance integrals always use the 120 m cadence. Fine erosion is filtered
  by its authored feature size relative to that footprint, not by a hard-coded
  camera-distance fade.
- **Rejected bracket.** Pure distance-proportional cadence was tested first.
  A coarse law reduced the runway contour-extrema metric by about 30% but erased
  the cruise foreground towers; reach-preserving variants either exposed a
  circular march frontier or restored the original terracing. No part of that
  path was retained. The adaptive marcher breaks the reach/detail tradeoff
  without treating correlated weather/profile hints as conservative bounds.
- **Bounded reach extension.** Baseline/production now spends 112 adaptive
  probes instead of 80. At the unchanged 600 m broad cadence this extends
  clear-air reach from 48.0 to 67.2 km; the shell segment cap and entry handoff
  moved from 50 to 75 km. The per-ray weather/macro context preserves the old
  midpoint on short segments but clamps to 25 km beyond shell entry on long
  segments, so extending range cannot cull a ray from weather sampled hundreds
  of kilometres downrange. `march_reach` mirrors all three numbers.
- **Cold evidence.** Relative to the enriched-field checkpoint, runway cloud
  coverage rose from 13.15% to 16.49% while normalized contour extrema fell
  19.6% (27.85 → 22.38); cruise coverage stayed stable (11.11% → 10.90%).
  The final 112-step Baseline GPU means on the development RTX 4070 Ti were
  1.474 ms runway, 1.004 ms cruise, 0.078 ms planet, 0.081 ms limb, and
  1.229 ms sunset. A second, fixed-mask reach comparison kept runway extrema
  stable (9.21 → 9.14 per cloud width) while cruise cloud coverage rose 20.2%;
  the extra extrema there are newly resolved cells, not recurring runway
  contour rings. All five
  cold logs were clean of shader, pipeline, panic, and capture-health failures.
- **Rejected live far march.** A direct 24-sample far march was wired to the
  exact generated 3-D basis and shared density shaping, then cold-tested.
  Across a ~200 km grazing chord its deterministic samples aliased the 8 km
  periodic basis into severe horizontal combs. The complete experiment was
  reverted. This confirms the planned far tier must prefilter the density into
  optical/albedo/normal/height moments; point-sampling more exact density at
  horizon range is not a substitute for that atlas
  ([ADR-20260722T102639Z](../adr/20260722T102639Z-far-cloud-density-must-be-prefiltered.md)).
- **Rejected periodic-density atlas.** The next experiment GPU-baked the exact
  broad-density function into a 512² cubemap, first as OD/height moments and
  then as four vertical OD strata. Tangent-footprint filters, continuous
  mean/variance reconstruction, analytic segment overlap, and a 24-sample
  atlas chord were cold-bracketed. Weak filtering exposed the 8 km periodic
  near tile as combs and a planet-scale checker/grid; strong filtering removed
  the grid only by averaging the result back into a smooth slab. Every code
  path and allocation was reverted; evidence remains in
  `artifacts/visual/runs/bl-33/step-6a/` through `step-6h/`. CLOUD-6 therefore
  needs a weather-conditioned, genuinely non-periodic far-density producer
  before moment/limb reconstruction can be faithful
  ([ADR-20260722T111036Z](../adr/20260722T111036Z-far-atlas-cannot-project-the-periodic-near-tile.md)).
- **Rejected single-tile phase warp.** Continuous coverage/base/top channels
  now displace both the broad and 21.6 km formation domains. This local change
  passed matched runway/cruise gates (runway coverage 20.32%; contour proxy
  6.22 extrema per 1,000 masked pixels; cruise coverage unchanged at 83.32%).
  Baking that exact weather-warped density into the same 512² four-stratum
  atlas still exposed long curved/diagonal comb families on `cloud-planet`:
  a phase warp bends one tile's repetition but does not remove its frequency.
  The atlas was fully reverted; evidence is under `step-8a/` and rollback under
  `step-8-rollback/`.
  ([ADR-20260722T135123Z](../adr/20260722T135123Z-weather-phase-warp-does-not-make-a-periodic-cloud-basis-aperiodic.md)).
- **Rejected incommensurate Cartesian basis.** A second independently
  transformed 3-D domain at an incommensurate period passed local runway and
  cruise coverage gates. Both a context-selected crossfade and a guaranteed
  35% contribution were then run through the exact same atlas control. The
  planet comb survived while its distribution changed, proving the spherical
  shell is cutting coherent Cartesian repeated surfaces into bands. CLOUD-6's
  next producer is therefore surface-parameterized (body cubemap direction +
  normalized layer height) and shared back into near density; the local
  Cartesian volume is not projected over the whole sphere
  ([ADR-20260722T141000Z](../adr/20260722T141000Z-far-cloud-density-is-surface-parameterized.md)).
- **Open residual.** Planet projection is seam-free and preserves broken
  synoptic systems, but grazing limb/cruise/sunset views still reveal the
  weather-column estimator as a smooth slab beyond the 67.2 km resolved handoff.
  Further reach remains gated on either more step budget or a true interval
  maximum-density bound. The next slice is one density-derived optical-depth/albedo/normal/
  height-moment atlas shared by planet and limb views, after the shared
  surface-space density contract exists; it is not another independent
  weather field. Its density basis must not project the periodic near tile
  verbatim across the sphere.

### Program acceptance matrix

| Scenario | Pass condition |
|----------|----------------|
| Runway under a storm system | A coherent tower/anvil/strata hierarchy dominates the sky without reading as a wallpaper slab; soft self-shadow, cool fill, and terrain/craft shadows align with visible cells |
| Climb through deck | Continuous extinction, wispy boundary detail, and changing visibility; no sheet pop; rapid yaw rejects history rather than smearing |
| Cruise beside convection | Kilometre-scale towers, anvils, rain-bearing cores, scattered cells, and local cauliflower detail coexist with no obvious repeated atlas cells |
| Low-orbit limb | Cloud line lies inside the aerosol limb; reduced-volume/orbit handoff is invisible in motion |
| High orbit/map | Planet-wide fronts, broken decks, and storm systems stay coherent and match the weather/shadows seen below |
| Noon to sunset | Deep blue-grey cores and bright rims remain energy-bounded; atmosphere transmittance produces natural warm edges, lit shafts, and aerial recession rather than white/grey clipping |
| Timewarp/body rotation | Weather advances in simulation time; history resets cleanly with no trails or ground slip |
| Solid overcast | A layered ceiling can coexist with openings and embedded towers; sun, terrain, hull, structures, water, and environment ambient all respond to the same cloud field |

## 5. Risks and decision gates

These choices were accepted on 2026-07-20 and are recorded in
[ADR-20260720T212214Z-one-weather-field-many-cloud-projections](../adr/20260720T212214Z-one-weather-field-many-cloud-projections.md) and form the
constraints for every later phase:

1. **Weather topology — recommended: cube/2-D array.** It costs more authoring
   plumbing than the current equirect map, but avoids polar concentration and
   matches existing body-fixed cube tooling. Keep equirect import/export.
2. **Renderer home — recommended: absorb the vendored crate into
   `thalos_body_render::clouds`.** Keeping it separate saves short-term moves but
   preserves split mechanism ownership through every later phase.
3. **Temporal scope — recommended: cloud-local resolve, independent of W13.**
   Whole-scene TAA may still improve edges later, but it must not gate the cloud
   budget or timewarp rejection work.
4. **Weather ambition — recommended: advected authored/procedural fields, not a
   fluid solver.** A simulation can be added behind `CloudWeatherField` if the
   gameplay case later justifies it.
5. **Hardware target.** CLOUD-0 measurements should set High/Medium budgets and
   the fallback floor before texture formats and cascade counts are frozen.

## 6. Explicit non-goals for the first vertical

- A general fluid or climate simulation.
- Local fog banks unrelated to the planetary weather field.
- Ray-traced cloud lighting.
- Multiple independently authored cloud systems for surface and orbit.
- Shipping precipitation/lightning before density, temporal stability,
  atmosphere coupling, and world shadows are correct.
