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

**Amount is not kind.** The weather cube answers "how much cloud is here, and
how tall" — it can never answer "what kind", because the morphology that
distinguishes roll streets from open-cell honeycomb from a lane-cut sheet lives
at 1–5 km and the cube's texels are ~5 km. That second question belongs to the
analytic cell field's **style** (`cloud_cell_style`, `thalos::atmosphere`), which
takes the cube's type channel plus one low-frequency organization field and
returns cell period, zonal elongation, roll tilt and lobe character. Two rules
hold it together, and both are load-bearing:

- **Style must not move the field's distribution.** The near tier's formation
  threshold is a single Monte-Carlo fit for the whole planet, so a style that
  changed the cell field's spread would make authored coverage mean different
  things in different places. Anisotropy, tilt and period are linear domain maps
  and preserve it exactly; anything that does not (the billow blend) is
  renormalized by a measured σ fit. `cell_field.rs` guards this with tests.
- **A style may never scale the sampling DOMAIN.** This is the one that cost
  three rounds. The field is sampled at `dir·(radius/period)`, so a spatially
  varying period (or aspect, which scales by √a) adds a second chain-rule term
  to the sampling gradient: the field runs at a higher local frequency than its
  period nominally sets, while the octave fade still band-limits against the
  nominal period. The octave is under-filtered by exactly that factor and it
  renders as fine feathered hatching — measured at **1.70× finer than nominal**
  for the version that shipped. Cell size therefore comes from the octave
  WEIGHTS (they multiply already-sampled values), anisotropy from a BLEND
  between two globally constant transforms, and billow from mixing two values
  at the same point. None of those can move a sample.
- **Constant cell area.** Elongation is applied as across ÷ √a, along × √a, so
  raising the aspect turns cells into rolls instead of enlarging everything.
- **Anisotropy belongs to the arrangement, not to the cloud.** A cloud street is
  a row of round cumulus, not a ribbon, so the aspect grades across octaves and
  the finest octave — the one carrying the silhouette the eye reads as "is this
  a cloud" — stays isotropic. Stretching every octave equally renders tapering
  comets and combed stripes.
- **Keep the aspect small, and never let the tilt vanish under it.** The
  diagonal map suppresses longitude by `1/a`, so a large `a` degenerates the
  field toward a function of latitude alone — and a function of latitude on a
  sphere has circular contours, which render as fingerprint whorls. Mixing
  longitude back into the lattice's `y` (the tilt shear) is the direct antidote,
  which is why the tilt carries a latitude term rather than being free to hit
  zero exactly where the elongation is strongest.

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

**Landed (slice 1, 2026-07-25): the near cascade on the tile renderer.**

- Producer: a `cloud_shadow` entry point in the same compute shader, off the
  same bind groups, so it integrates `get_cloud_map_density` — the identical
  field, config, weather cube and strata cube the view march samples. There is
  no second cloud distribution that could disagree with the visible sky.
  Extinction is the view march's own primary-beam octave (`exp(-τ)`), so a cell
  that reads opaque from the air lays an equally opaque shadow.
- Geometry: 512², parameterised over the tangent plane under the view anchor,
  body-fixed (`CloudShadowFrame`). A texel holds the transmittance of the beam
  *passing through that plane point*, so a receiver looks up by intersecting
  its own sun ray with the plane — the parallax that lays a low sun's shadow
  kilometres downwind of the cloud casting it, which is exactly what the
  rejected coverage projection cannot do. Half-extent climbs an octave ladder
  with anchor altitude (12× altitude, clamped to 8–64 km), keeping the texel
  between 31 m and 250 m — sized for COVERAGE, not resolution, because every
  framing that matters is oblique and an oblique camera at 1 km sees ground a
  hundred kilometres out (measured: at 1.5× altitude the cascade covered only
  the basin under the camera and the rest of the frame fell outside);
  every rung oversamples the authored erosion scale, so the map is not
  texel-snapped and the erosion detail retires through the same footprint fade
  the view march uses.
- Receiver: `thalos::cloud_shadow::cloud_sun_transmittance`, one shared sampler,
  folded into the tile material's existing direct-sun gate. Standing at the
  gate's `SHADOW_FLOOR` is deliberate on that path: it multiplies the whole lit
  colour, so a physical `exp(-τ)` would erase the sky fill with it.
- Capture axis: `just compare <scene> cloud-shadow` (off / on / raw). `raw`
  paints the transmittance the cascade holds, which is the split that matters
  here — producer and receiver reach the same lookup frame by different routes.

Not yet built, in priority order: the coarse cube tail (so the term survives
past the cascade instead of feathering to lit at its border), the overcast
ambient/IBL response, the atmosphere march's shafts, incremental/temporal
cascade updates (it currently re-marches every frame), and the scatter/craft
receivers — terrain darkens under a cloud while the grass and trees standing on
it do not.

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
- **CLOUD-6 static handoff completion (2026-07-22).** Tier-isolated cold
  captures pinned the slab to the far projection. Far mips now follow projected
  pixel footprint rather than six-sample chord spacing; the chord preserves the
  prefiltered areal mean rather than strongest-hit/stacked opacity; and the
  result includes the near tier's expected sub-cell 3-D morphology fill. The
  near volume is gated by the same four-stratum surface envelope. Runway,
  cruise, limb, and planet captures retain coherent gaps and no longer thicken
  solely with range. See ADR-20260722T182853Z and
  INC-20260722T182934Z. The in-motion handoff check remains user verification.

### BL-33 fidelity convergence checkpoint (2026-07-22)

The static BL-33/CLOUD-6 convergence slices are headless-verified; the item is
`verify` pending the user-run in-motion surface↔orbit handoff.

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

### Round 2 — user verdict and the regime/morphology rework (2026-07-23)

The user verified the 2026-07-22 state against the Blackrack/MSFS reference
set and **failed it** on three axes (orbit, high-aerial, and cruise
screenshots): the far estimator reads as a thick translucent veil that is
simultaneously everywhere and missing where the volumetrics are; the 67 km
near/far handoff is a visible seam even in stills; and the volumetric field is
monotone same-scale puffs with visible lattice rows at cruise. Root causes and
the round-2 response (CLOUD-6 round 2 + BL-20260723T165923Z + BL-33):

1. **Regime-structured weather producer** (`CloudWeatherField::from_climate`):
   a synoptic occupancy field *thresholded* into weather systems with genuine
   zero-coverage clear air between them (the old producer summed fixed-scale
   noises around one mean — statistically identical speckle everywhere); an
   intensity term for system cores; a regime partition honouring the authored
   `type_mix` (scattered-cumulus fields / stratus sheets / storm clusters,
   plus the retained frontal ridges); per-regime coverage texture, cloud type
   (congestus building inside cumulus fields), and per-regime base/top so
   thin decks and tall towers coexist. Round-2 calibration from the user's
   live pass: bases lifted (~0.10 shell fraction), coverage thinned, gaps
   widened, congestus no longer gated to deep systems only.
2. **Near-tier formation authority moved to the surface field**
   (`get_cloud_map_density`): the strata density drives the formation
   threshold (`mix(0.76, 0.30, env)`); the periodic Cartesian tile only
   sculpts sub-texel lobes inside that envelope, so the spherical shell can no
   longer cut its repeat into planet-visible rows (completes the
   ADR-20260722T141000Z direction). The legacy Cartesian-organized threshold
   is retained behind `surface_density_coupling = 0` for A/B attribution.
   A coarse-mip region probe protects rays whose 25 km context anchor lands
   in a clear lane. Tower morphology: tall columns hold mass with height
   (`column_tall` reduces `vertical_narrow`, mirrored in the CPU strata
   producer) and weight the broad-shape spectrum toward low frequencies so a
   tower reads as one coherent mass while fair-weather puffs stay small.
3. **Far tier renders morphology, not a veil** (`sample_orbital_cloud` +
   shared helpers): `weather_cloud_opacity` recalibrated from the old
   near-mean statistics remap (`smoothstep(0.45, 0.80, cov·1.22)` deleted
   moderate-coverage fields from the far tier entirely) to a
   soft-toe areal fraction; column optical depth no longer multiplies areal
   coverage (a cell in a 30 %-coverage field is as optically thick as one in
   overcast — the double-count made every moderate region simultaneously
   sparse and translucent, the grey-veil signature); and opacity is
   footprint-split — unresolved footprints keep the areal-mean alpha,
   resolved footprints sharpen density into near-opaque cells with clear
   gaps, with stronger moment-normal relief at range. `SolidPlanetMaterial`
   mirrors the thinness-only optical-depth response.

4. **Layer-relative strata (contract change).** The four surface-density
   strata were sampled at fixed shell heights (12.5/37.5/62.5/87.5 % of the
   10.5 km shell). Any layer thinner than the ~2.6 km stratum spacing could
   fall entirely between two sampling heights and read zero from every
   stratum — which is exactly what the round-2 base lift did to quiet cumulus
   decks (~13–33 % of the shell): the far tier showed clear sky over a solid
   near-volume deck (user screenshots, 2026-07-23). The strata are now
   authored at 1/8, 3/8, 5/8, 7/8 of the local **[base, top]** interval, and
   every consumer maps its shell height through the same weather base/top
   channels (`h_layer = (h − base)/(top − base)`); outside the layer the
   shared reconstruction returns a hard zero (clamping to edge-stratum values
   painted a halo above tops). A thin deck now keeps full vertical resolution
   wherever it sits, so layer altitude/thickness are free per regime. The
   strata *column max* (`cloud_surface_column_density`, used by
   `SolidPlanetMaterial`) is layer-invariant and unchanged.

**Round-3 capture iteration (2026-07-23, cold lane):** four defects found and
fixed against the fresh five-preset sweep. (a) The layer-relative chord lost
grazing clouds — averaging over all six segments diluted a thin layer's one
in-layer hit toward zero at the limb; replaced with an analytic per-segment
layer-overlap clip (only intersecting segments enter the mean, evaluated over
the clipped layer-relative span). (b) Sunset chroma keyed on the
relief-perturbed shading normal painted midday cells that tilt away from the
sun in orange; day/night and warm chroma now follow the geometric solar
elevation, with the warm band narrowed to < ~9°. (c) The far foreground-air
veil used a flat 60 km/µ horizontal path; from orbit top-down that stripped
~74 % of the blue and turned the disc's clouds beige — replaced with a
scale-height air-mass path (`8000 / (µ + 0.10)`). (d) The strata fetch now
shares `sample_weather_soft`'s 0.75 mip floor so mip-0 texel lattice stops
crunching cell borders. Far radiance prefactor 0.55 → 0.68 to match the near
volume at the handoff. Final planet/cruise/limb/runway/sunset captures: white
broken systems with real clear regions from orbit, near↔far handoff without an
occupancy or colour jump in stills, limb "pink slab" identified as laterite
terrain (not clouds).

**Quantile fill contract (round 3, from the user's ascent sequence):** the
near tier's areal fill must EQUAL the strata density — the same contract the
far tier reads directly. The env smoothstep saturated at sd 0.60, so any
moderate-density texel (an authored 30–50 % broken field) rendered ~90 %
solid near the camera while the impostor honestly drew the sparse patches;
the visible "transition arc" at the 67 km reach was exactly that fill step.
The threshold is now the exceedance quantile of the tile's shape
distribution, linear in sd (`mix(0.72, 0.32, sd)`), which also delivers the
"too dense" correction globally. Residual: per-texel agreement is
approximate (different bases); judged by the in-motion user gate.

**Round 4 — the ascent-mismatch root causes (2026-07-23, probe-driven):**
the user's ascent kept showing a solid near-tier deck under a cloudless
impostor. A probe/A-B chain (stage ladder → strata content → rotation
discriminators → same-framing tier A/B via a new ownership-bypassing
`far-only` diagnostic) found and fixed three real defects: (1) the composite
read **spawn-time per-body weather-cube copies whose GPU content had diverged
from the live cubes** the marcher samples — the far tier was reading
near-empty strata; the composite now binds the compute pass's live cubes for
the active body (one weather authority, one upload); (2) `sync_cloud_weather_map`
now replaces the cube assets wholesale through `cloud_weather_image` instead
of mutating `image.data` in place; (3) the marcher's exact world→body frame
(wind included) is published as `ActiveCloudFrame` and overrides the
composite's copy — registration by construction. Falsified along the way
(recorded in BL-20260723T214730Z): quat-convention inversion, wind-angle
misregistration in cold runs, and a Rust/WGSL uniform-layout mismatch.
**Resolution (2026-07-24, INC-20260723T221126Z):** the alignment failure's
true root cause was that the NEAR marcher sampled a runtime-updated copy of
the weather cubes whose GPU content the re-upload path had scrambled — the
volumetrics flew a corrupted field while impostor/composite rendered the
correct spawn-time upload. Runtime cube mutation is now eliminated:
`sync_cloud_weather_binding` handle-swaps the compute pass onto the active
body's spawn-uploaded cubes (`BodyCloudCubes`), so one correctly-uploaded
field serves every consumer. Tier A/B, cruise, runway, and disc captures
agree. **User live verdict 2026-07-24: positioning is consistent** — the
alignment defect is closed. Two residuals remain, owned by
BL-20260723T214730Z: the near volumetrics render *thinner* than the thick
impostor suggests (fill/optical-depth parity — re-measure from scratch, all
pre-fix parity numbers were taken against the corrupted cube), and the
near↔far transition is *coarse* (the handoff needs finer morphology blending,
not just occupancy continuity).

Verification: headless sweep passed agent-side as above; the remaining gates
are user-run — the ascent-site near/far agreement (pending
BL-20260723T214730Z), in-motion surface↔orbit handoff, and the local-view
morphology verdict (squat puffs vs congestus towers in one horizon) that
round-2's calibration targets.

### Round 5 — derived fill/opacity pairing + transition morphology (2026-07-24)

The two residuals of BL-20260723T214730Z, re-measured from scratch on the
fixed cube (every pre-fix number was against corrupted data). New protocol
artifacts: a CPU cloudy-site probe (`solar_system_state::cloud_site_probe` —
scans the authored field for broken-moderate sites near the runway's daylight
longitude, with sun elevation at the boot epoch) because the spaceport column
is authored clear; tier A/B at `THALOS_RUNWAY_SITE="22.0,153.0"`,
spaceport-aerial `ELEVATION=70 DISTANCE=25000`, pixel fills vs a
`CLOUD_COVERAGE=0` baseline (`artifacts/visual/runs/cloud_fill2/`).

**Baseline measurement (the defect, quantified):** near-only fill 0.044 /
mean amplitude 0.013 vs far-only fill 1.000 / 0.419 over the same
0.33-coverage authored region — the far tier's saturating resolved curve
(`smoothstep(0.06, 0.40, mean_c)·0.95`) painted a near-solid veil while the
near tier under-filled its authored contract by ~8×.

**Fill pairing is now DERIVED, not tuned** (`clouds::fill_lut`,
ADR-worthy rule: never hand-retune either tier's response):

1. A CPU mirror of the marcher's density math (same 64³ tileable noise
   volume, domain transforms, thresholds, profiles, erosion — statistical
   fidelity is sufficient, but WGSL `fract` is floor-based) Monte-Carlo
   marches ~16 k vertical columns through the body's actual weather cube at
   spawn (~0.5 s, logged per-bin).
2. The near tier's formation threshold becomes an 8-node piecewise-linear
   curve `T(env)` fitted by coordinate descent so simulated column fill
   tracks the strata mean (identity contract; low bins stay
   envelope-limited, thin-deck cross-talk keeps mid bins above target —
   recorded in the fit log). Monotonicity must be enforced BY CONSTRUCTION
   (top + non-negative deltas): clamp-after-move silently froze the fit.
3. The far tier renders a 16-node LUT of the *achieved* expected column
   opacity `E[1−T_column | strata mean]` — far thickness equals near
   thickness by construction, independent of fit quality. Plumbing:
   `BodyCloudFill` → `CloudsConfig::fill_threshold_nodes` (compute uniform) +
   `BodySkyExtra::fill_response` (composite uniform). The mirror was
   validated against the capture: predicted region fill 0.20 vs measured
   0.27.

**Transition morphology (the coarse-handoff half):** the residual far excess
was spatial — per-cloud amplitude matched (0.26 vs 0.31) but the far tier
covered ~3× the area, from strata-blur halos and no sub-texel gaps. Three
changes: (a) the far tier perturbs the LUT input with body-fixed value noise
at the near tier's cell scale, anchored at the occupancy-weighted mean chord
position (NOT the best-segment position — the argmax flips discontinuously
between rays and cut "torn seam" lines across every cell), faded out as the
pixel footprint approaches the noise period so disc framings keep the
accepted filtered look; (b) the strata fetch's 0.75-mip floor now applies
only to genuinely unresolved footprints — at handoff ranges it was pure
magnification blur that widened every cell by kilometres of low-alpha halo;
(c) the near tier's last kilometres (reach dissolve + far shell entry)
widen the formation edge and retire erosion detail (`soften` in
`get_cloud_map_density`), so puffs melt into the same soft masses the far
tier renders before the occupancy crossfade swaps them.

**Reliability tell (new):** one cold capture in the round rendered zero
clouds with exit 0 — `ActiveCloudBody` never activated for the entire run
(no "composite frame override" line in the log); the identical re-run was
fine. When a capture shows no clouds at a known-cloudy site, check that log
line before diagnosing shader logic (BL-20's capture-validity gap extends to
silent cloud-pipeline non-activation).

Remaining: port the derived LUT into `solid_planet.wgsl` (its
`surface_density × thinness` response is close to the LUT mid-range —
closer than the old saturating curve — but not identical across the
composite↔impostor swap), and the user's live gates.

### Round 6 — one representation across the visible range (2026-07-24)

The response to the round-5 verdict (BL-20260724T003705Z; Blackrack/MSFS
reference study in that row): the near volumetric march now carries ONE cloud
representation to a 300 km reach, LOD'd by footprint instead of handing off to
the far estimator mid-view. The march contract (band edges 42/90/180/300 km,
steps 600/1200/2400/4800 m, refine at 1/5, entry ownership 240–300 km, reach
dissolve over the last 15 %) lives in `thalos::atmosphere`, imported by BOTH
`clouds_compute.wgsl` and `cloud_composite.wgsl` — the partition lockstep is
structural, not a comment. The alias-safety inversion is the load-bearing
idea: the density field band-limits AHEAD of each step increase (erosion
retires as the refine cadence outgrows the detail scale; the shape spectrum
narrows to its low-frequency mix; past ~90 km the Cartesian shape term is
replaced by the DERIVED homogenized field `E[shaped | env]` — a third LUT
from the same spawn Monte-Carlo, mean-preserving by construction and cheaper
per probe than a near-field sample). This satisfies
ADR-20260721T033055Z's conservative-bounds rule rather than fighting it:
BL-33's moiré and INC-0011's isosurfaces came from stretching steps over the
UNFILTERED field. Steep rays clamp to a 350 m radial step (bounded — their
in-shell segments are geometrically short). Budgets: 112/176/192/224
(baseline reaches the full 300 km).

Whiteness track: the marcher's ambient now binds the physical F3/F4
`SkyAmbient` irradiance (`E_sky/π`; 0.45 view factor on undersides; the old
analytic pair survives only as the space stand-in), with a τ-correlated
ambient self-occlusion proxy so interiors keep shape — the directional sky
march (Blackrack-style) remains the CLOUD-5 upgrade. Two capture-round
fixes: the far tier's morph fine octave needed its own alias gate (dot band
at the relocated handoff), and the brighter contrast makes the pre-existing
per-pixel fringe jitter stipple more visible (a temporal-reconstruction
quality item, not a density defect). Verified: cruise shows one continuous
deck to the horizon (no ownership arc), mid-altitude composite keeps
registration with shaded, white cells; limb and disc unchanged. Open:
far-tier brightness prefactor re-match, GPU budget re-measurement, fringe
stipple, user live gates.

**Round 6b — the ascent regime (2026-07-24, user verdict on 6a):** three
defects specific to the 100–400 km ascent view, all fixed against the
ascent-altitude probes (`THALOS_SCREENSHOT_CAMERA_ALTITUDE`, 150/280/400 km):
(a) the entry-ownership crossfade rendered as a sharp nadir CIRCLE filled
with an over-count veil — the far tier weighted its accumulation *inputs*
(through the response LUT's nonlinear toe) while the near tier faded
linearly in alpha, so the two halves never summed to unity; ownership now
attenuates the far tier's CONVERGED opacity (output-linear partition, at the
occupancy-weighted chord position). (b) The ~5 km strata texels rendered as
rounded SQUARES wherever footprints resolve them; a shared tangential
domain warp (`cloud_strata_warp` in thalos::atmosphere — same function, same
amount convention in the marcher's homogenized bands and the far tier, so
the fields stay registered) turns the lattice organic. The warp is a
measure-preserving direction remap, so the derived LUT statistics are
unchanged. (c) The homogenized near band read as smooth blur against the far
tier's crisp mottle; the sub-texel morphology noise is now part of the
shared contract (`cloud_morph_noise`) and the marcher's coarse bands
re-mottle `env` with the same field. Probes after: 150 km and 280 km nadir
views read as ONE continuous organic field; cruise pixel-identical to 6a.

**User verdict on round 5 (2026-07-24, live screenshots):**
registration/thickness parity largely pass — "a lot of it is quite
accurate" — but the transition still fails: the far sheet terminates on the
camera-relative ownership arc (a hard scalloped curtain edge), thin sheet
skirts show swirl filaments (grazing layer-clip residue), clouds read
uniformly gray, and the near→far detail loss is a visible cliff. The
architectural conclusion (with a Blackrack/MSFS reference study) is that the
residual symptoms are the mid-air representation swap itself; the
superseding direction is one footprint-LOD'd representation across the
visible range — see BL-20260724T003705Z-cloud-single-representation-reach.
The derived `fill_lut` pairing survives as the contract for the remaining
(true-orbit) representation boundary.

### Round 7 — volumetric morphology: sculpted tops (2026-07-24)

**User verdict on round 6b (live ascent screenshots): "continuity is quite
acceptable now" — the transition gate passed.** The new front is the
volumetric clouds themselves: "mostly really some flat sheets"
(BL-20260724T022522Z-cloud-volumetric-morphology). Three coupled causes,
three fixes, all in the shared-shaping lockstep trio (`get_cloud_map_density`
in clouds_compute.wgsl · `march_column`/`sample_shaped` in fill_lut.rs ·
`cloud_surface_density_cpu` in solar_system_state.rs):

- **Tops were faded, not sculpted.** The convective vertical profiles bled
  density out over the top ~30% of every column — a uniform soft lid at each
  weather texel's top altitude, which is exactly the "flat sheet" read. Now
  the profiles keep only a thin condensation skin (cumulus 0.93 / storm 0.94
  top fades) and top *shape* comes from a quadratically height-rising
  threshold (`dome = h²` × 0.42 cumulus / 0.30 storm, ×(1 − 0.45·column_tall))
  — each lobe's top is the isosurface where its own shape noise dips under
  the rising bar, so strong lobes tower and weak lobes stay squat
  (MSFS/Nubis-style carved cauliflower domes). No transcendentals, so the
  per-sample perf invariant holds. The term is near zero at the base where
  areal fill is decided, and the spawn-time calibration re-derives the
  formation threshold + both response LUTs against the new math, so tier
  parity survives by construction.
- **Erosion had one character at all heights.** It now flips: wispy
  (inverted-Worley) shredding on undersides, cauliflower billow cuts on
  domes, slightly stronger up high (`×(0.80 + 0.55·h)`); fill_lut pre-folds
  the height factor into `SampleRecord::erode`.
- **Ordinary cumulus had no room to develop.** The weather producer's
  `top_cumulus` baseline gave plain fair-weather columns <1 km of a 10.5 km
  shell; raised to 0.14 + 0.58·(0.42·cell_broken + 0.58·congestus) +
  0.09·vertical_noise so broken fields read as mixed-height puffs and
  building cells carry real depth.

**Verification state: compile-clean, captures BLOCKED** — the GPU dropped
off the bus mid-session ("GPU is lost", nvidia-smi; reboot required), after
a capture-host boot OOM'd while the user's live game session held VRAM.
Capture script staged at `artifacts/visual/runs/cloud_morph/capture_r7.sh`
(cruise / interior / 20 km-above / runway framings); run it after reboot and
judge dome variety, base wisps, and that stratus regions still read as
sheets. Carried follow-ups: far prefactor (0.68) re-match, GPU budget
re-measure.

### Round 8 — cloud radiance: the missing multiple scattering (2026-07-24)

**User verdict on round 7's morphology: still "muddy, and not pure white"**, with MSFS
references. The defect turned out not to be morphology at all — it is that the near
march rendered **single-scattering radiance** and called it multiple scattering. Full
diagnosis, measurements, and the ruled-out hypotheses:
[INC-20260724T232256Z](../incidents/20260724T232256Z-cloud-multiscatter-octaves-carried-no-energy.md).

In one line: the octave sum was divided by `Σw`, making it a weighted *average* of
`1/4π`-normalised phase values — three probability densities averaged still carry one
scattering event's energy. A lit cell returned `0.051·E` where the diffusion limit is
`A/π ≈ 0.255·E`, so the physically-bound sky ambient out-competed the sun and clouds took
their chroma from it. Measured on `cloud_cruise`: near-tier clouds at 0.30 display luminance
against 0.49–0.73 for the sky *behind* them.

The source term is now split as the physics is — exact normalised phase for single
scattering (forward glare, silver lining), plus an isotropic-equivalent multiple-scatter
reservoir at the medium's diffusion albedo `CLOUD_MS_ALBEDO = 0.80`, with the wider octaves
demoted to supplying its depth response and a residual anisotropy. Three compensating
constants downstream were removed in the same change rather than left carrying the error:
the far prefactor (hand-raised 0.55 → 0.68 to hide the dark near tier) is now **derived**
from the same anchor at 0.302; the Reinhard white point moves 2.2 → 10.0 so it bounds the
glare spike instead of dimming every white top; and both tiers' per-channel "phase headroom"
tints are deleted, since stacked on the authored albedo they biased sunlit cloud ~12–15%
blue.

Two standing rules follow (§3.4):

- **A normalised phase function carries one scattering event.** An octave series
  approximating multiple scattering must SUM energy. Normalising it by `Σw` turns it into a
  lobe-widening filter with a physical-sounding name.
- **Both tiers anchor to `CLOUD_MS_ALBEDO`, never to each other.** A fully lit optically
  thick cell must land at `A · flux · SCENE_FLUX_SCALE · SURFACE_DIRECT_SCALE` — a Lambertian
  surface of albedo A facing the same sun, the anchor every spine surface already uses.
  Matching the far tier to the near tier by eye is what let a 4× error hide inside a
  "handoff continuity" fix.

The `fill_lut` fill/opacity pairing is untouched and needs no re-derivation: it governs how
much of the frame clouds cover, this governs radiance per covered pixel.

**Verification state: compile-check and captures BLOCKED** — the working tree carries
unrelated in-progress fleet-kernel work in `physics_canonical/src/simulation.rs` that does not
build. Run `just capture cloud-cruise cloud-interior cloud-sunset cloud-planet` once it does,
and check: (a) sunlit tops read white, not blue-grey; (b) cloud tops are brighter than the sky
beside them; (c) near↔far handoff shows no brightness step at cruise/aerial altitudes — this
change moves both tiers, so the handoff is the thing most at risk; (d) round 7's dome tops
now read as volumes under the restored lit/shadow contrast.

### Round 9 — occupancy is not opacity; convective anisotropy (2026-07-25)

**User verdict, with a cumulonimbus reference: "clouds are super ghost-ish… they should be
realistic, solid appearing clouds"**, plus an explicit ask for vertical development.
Reproduced first (`cloud-runway`, `massif-ridge`): from a level or below viewpoint the deck
rendered as flat grey-lavender pancakes *dimmer than the sky behind them*, with the sky
gradient showing through. Looking straight down (`cloud-cruise`, `massif-aerial`) the same
deck read acceptably solid — the tell that this was an optical-depth defect, not a shading
or coverage one.

Two causes, both structural.

**1. The near march scaled extinction by occupancy.** `env` — the strata cube's value — is
an **areal fraction**: the share of a ~5 km weather texel that is cloudy
(`cloud_surface_density_cpu`, whose 2026-07-25 rework made this explicit). The marcher spent
it twice: once correctly, as the input to the derived `formation_threshold(env)` that decides
*where* cloud forms, and once as `shared_envelope = smoothstep(0.04, 0.42, env)` multiplying
the returned density. The second use thins every cell in proportion to how sparse its
*neighbourhood* is, so a 30 %-coverage region renders as one 30 %-opaque film instead of
solid cells with clear air between them. This is precisely the grey-veil error
`weather_column_from_texel` warns about, reached from the near tier's side, and it is why
`E[column alpha] = strata_mean` was being satisfied the wrong way: the fit is indifferent
between *40 % of columns at alpha 1.0* and *100 % of columns at alpha 0.4*, and the envelope
multiply pushed it to the latter.

Occupancy now gates formation only — `formation_gate = smoothstep(0.02, 0.08, env)`, a
clear-air cut so the Cartesian shape noise cannot grow cloud where the producer authored
none. The toe matters as much as the removal: a gate ramping from *zero* turns a
3 %-occupancy lane into half-density fog and reintroduces the ghost by another door. The fill
contract is preserved by construction because the spawn-time Monte-Carlo re-fits the
threshold curve against this exact math — measured: node 0 moved 0.503 → 0.667, i.e. the
clear-air suppression the envelope used to do migrated into the threshold, exactly as
intended, while mid/high-occupancy formation is unchanged and those cells are now several
times more opaque.

**2. Convective cells were pancakes by construction.** One isotropic 8 km shape period over
a layer 1.5–6 km deep cannot produce a tower, no matter what round 7's dome threshold carves
out of it — the cell is wider than the shell is tall. Convective columns now stretch the
domain's **radial** axis (`CONVECTIVE_STRETCH = 1.9`): a metre of climb moves the sample
only 1/stretch of a metre through the noise, so one lobe stays coherent from base to top
instead of decorrelating into a stack of unrelated blobs. Sheets keep the isotropic period —
a stratus deck genuinely is wide and flat — and `convective_w` is continuous in the type
channel, so no type boundary pops the volume.

**Rejected in the same round, by capture:** narrowing the convective *horizontal* period to
0.42× (≈3.4 km cells) to fix the aspect ratio directly. It made the near field beautifully
solid and painted the cruise deck as a **regular lattice of identical puffs in rows** — the
64³ volume's Cartesian repeat, planet-visible once many periods fit on screen (the
ADR-20260722T141000Z "rows" family, arriving from the tile side this time). The standing rule:
**apparent cell size comes from the formation threshold picking peaks out of a wide period —
one lobe per period is what broken cumulus looks like — never from shortening the period.**
Narrower cells require breaking the repeat first (a second octave at a non-commensurate
period), which is a separate, measurable change.

Three lockstep sites as always: `get_cloud_map_density` (clouds_compute.wgsl) and
`march_column` / `sample_shaped` (fill_lut.rs); `cloud_surface_density_cpu` is unaffected
(it authors occupancy and never had the envelope). `FILL_LUT_VERSION` 1 → 5 across the round.

**Verification (agent captures, this repo):** `cloud-runway` — sunlit tops white with defined
cauliflower silhouettes and shaded undersides, replacing the grey pancakes; `cloud-cruise` —
broken cumulus streets, solid and white, no lattice; `cloud-interior` — dense interior, no
regression.

**Residual, NOT closed — the `massif-ridge` veil.** At that framing (camera ~6.8 km, looking
12° down across the 5.8 km ridge) the massif still reads through a milky film. Established by
capture, not assumed:

- It **is** cloud, not aerial perspective: `THALOS_SCREENSHOT_CLOUD_COVERAGE=0` renders the
  massif completely crisp (`artifacts/visual/runs/cloud-ghost/ridge_nocloud.png`).
- It is the **near** tier, not the far projection: `THALOS_SCREENSHOT_CLOUD_TIER=near-only` is
  indistinguishable from the production composite.
- It is **not** march-budget exhaustion on grazing rays: forcing the step limit to 900
  (diagnostic WGSL edit, reverted) produced a pixel-equivalent image.

Remaining suspects, in order: the far-field edge softening (`edge_softness` widens 0.055 →
0.30 across the 30–42 km `c1` ramp, keyed to the *broad* band step even though density is
only ever integrated at the 0.2× refine cadence — i.e. plausibly ~5× over-conservative), and
the possibility that the camera sits inside a tall column at that site, in which case the
interior is simply under-dense. Both are measurable; neither was chased in this round.

### Round 10 — cells are an analytic surface field (2026-07-25)

**User verdict on round 9, against the Blackrack/MSFS reference set:
"incoherent soupy blobby mess, with rough transitions between LOD."** Both
halves have one cause, and it is structural rather than tuning — full decision,
measurements, and standing rules in
[ADR-20260725T222409Z](../adr/20260725T222409Z-cloud-cells-are-an-analytic-surface-field-lod-never-renders-the-mean.md).

Diagnosis, from captures and two measured numbers:

- **Nothing could draw a 1–5 km cell at range.** The weather cube is 4.89 km per
  texel (~10 km Nyquist) and its authored content bottoms out at 15–25 km
  (`fbm3(dir_raw · 128, 3)` / `· 211, 2`). Cell morphology lived only in the 8 km
  periodic Cartesian volume, retired past ~42 km and *replaced* past ~90 km.
- **The LOD rendered the field's mean.** `E[shaped | env]` is mean-preserving,
  which was believed sufficient; it is not, because shading is nonlinear
  (`E[shade(σ)] ≠ shade(E[σ])`). A variance-destroying filter keeps a region's
  average brightness and deletes every cloud in it — the flat beige sheet from
  ~40 km to the horizon, with a contrast step at the band edge.

The response is one shared field, `cloud_cell_field` (thalos::atmosphere),
evaluated identically by the marcher, the far projection, and `fill_lut`'s CPU
mirror: an aperiodic hash-lattice **column** field in the body-fixed *direction*
domain (no tile, so no repeat for the shell to cut into rows — the general form
of ADR-20260722T141000Z). The Cartesian volume is demoted to a mean-preserving
sub-cell sculptor that retires with range. LOD is octave retirement inside one
field, so range degrades cells → coarser cells → strata and no tier swaps
representation.

Three capture rounds, each fixing a defect the previous one exposed:

1. **Uniform popcorn carpet.** The raw octave mix has std **0.115** (measured,
   1.5M directions) — the threshold is a cliff (0.40 → 0.60 swings coverage
   80 % → 20 %) and everything crosses it by the same margin.
   `CLOUD_CELL_GAIN = 3.2` restores std ≈ 0.20.
2. **Flat-topped mesas with vertical sides.** Applying that gain with a hard
   clamp saturated **9.6 %** of the sky to exactly 1.0; a constant region has no
   isosurface, so the dome threshold cut every core at one exact altitude. Fixed
   with a soft saturation `x / (k + |x|)`, `CLOUD_CELL_KNEE = 0.45` — odd-
   symmetric about 0.5, so still mean-preserving, and no transcendentals.
3. **Boxy right-angled silhouettes.** `cloud_strata_warp` was gated to the
   coarse bands, on the premise that the Cartesian sculptor hid the ~5 km texel
   lattice near the camera. That premise died here — the cell field is
   thresholded against `env` directly, so a texel edge cuts a vertical wall
   through a cloud. Now applied unconditionally in both tiers (measure-
   preserving, so the calibration is untouched).

Also in the round: the derived `shape_response` LUT is **deleted** (its only
consumer was the homogenized band) — from `fill_lut`, the compute uniform,
`CloudsConfig`, and the cache schema; `FILL_LUT_VERSION` 5 → 6. Range-keyed
edge softening drops 0.30 → 0.075: it was keyed to the *broad* band step
although density is only integrated at the 0.2× refine cadence (~5×
over-conservative), and it was round 9's prime remaining suspect for the
`massif-ridge` milky veil.

**Verification (agent captures, compile- and clippy-clean).** `cloud-cruise`:
cells resolved to the horizon with real openings — the beige LOD sheet and its
ring are gone. `cloud-runway`: cumulus with rounded tops and depth replacing the
milky white wash. `cloud-limb`: thousands of individually resolved cells out to
the limb, replacing grey dither — the reference-image read, and the clearest
single result of the round.

**Open, in priority order:**

- **Coverage reads too high** at oblique framings (limb/cruise): near-overcast
  where the references keep large clear areas. The per-column fill contract is
  satisfied (the fit log tracks strata mean), so this is the weather producer's
  authored occupancy plus slant-path overlap, not the fill pairing. Measure
  before touching either.
- **Whole-disc framings are unchanged and correctly so** — at ~14 km/pixel every
  cell octave is below the sampler, so the disc shows the producer's 15–25 km
  smears. Cell structure at that framing is a producer question, not a renderer
  one.
- Faint horizontal banding on some distant cells (suspect: radial march
  quantization now that the cell field is constant down a column, which removes
  the sculptor that used to mask it) and the pre-existing per-pixel fringe
  stipple.
- GPU budget re-measurement: the cell field adds up to 3 hash-lattice octaves per
  density sample, offset by the retired noise fetches in the coarse bands.

### Round 11 — footprint LOD, and the impostor question (2026-07-26)

**User verdict on round 10, from a live ascent:** the mid-field is good, but
mid-ascent cells carry a **horizontal comb**, above ~300 km the volumetrics
**fade out entirely**, and the impostor that replaces them renders **no clouds
over the space center** where the volumetric tier had just drawn them — plus the
direct question, *should we even have an impostor?*

One cause: **the LOD ladder was keyed to camera distance, and from orbit
distance is not footprint.** Full decision and measurements in
[ADR-20260726T000929Z](../adr/20260726T000929Z-cloud-lod-is-keyed-to-footprint-not-distance.md).
At 300 km a pixel covers **345 m** (0.00115 rad/px at Bevy's default 45° fov on
a 1280×720 target), so a 5.4 km cell is ~12 px across — yet the old ladder read
"300 km", switched to 4800 m steps (~10 samples through the deck: the comb) and
then handed the whole ray to the far estimator (`ray.start >
CLOUD_MARCH_ENTRY_FADE_END_M`, which for a nadir view is just altitude). The far
estimator renders `E[opacity | occupancy]`, so a 10 %-occupancy region became a
~0.09 alpha wash — nothing — where the near tier had drawn solid cells. That is
round 9's "occupancy is not opacity" arriving from the far side.

**Should there be an impostor?** Not in the flight envelope. `get_ray` already
skips clear air analytically to the shell entry, and a near-nadir orbital ray
crosses the 10.5 km shell in ~10–15 km of path — the expensive case is *grazing*
chords, a function of ray angle, not altitude. A projection is only *correct*
where cells are genuinely sub-pixel (whole-disc/map, ~14 km per pixel), and there
no transition can be visible. So the far tier survives, demoted to that regime
plus the per-distance tail past the marcher's probe frontier.

Reference practice, for the record: nobody estimates the far field from
*statistics*. Either they keep raymarching the same field and LOD by footprint
(Blackrack's scaled-space clouds re-raymarch the same coverage field; MSFS keeps
marching through its envelope), or they bake the far tier as a slowly-refreshed
render of the same field — which is what CLOUD-6 originally planned and what
ADR-20260722T102639Z requires. A mean-opacity veil is neither.

Landed:

- **Step law** `cloud_march_step_m(t, pixel_angle, radial_rate)`: footprint is a
  FLOOR (600 m near-field minimum, so runway cost is unchanged); a radial cap
  (350 m of *vertical* travel) and a cell cap (1800 m ≈ ⅓ of the coarsest cell
  period) are CAPS and win where they disagree.
- **Ownership** `cloud_far_ownership(footprint)`, 2.2 → 5.4 km — ~1900–4700 km
  altitude on Thalos, so the whole flight envelope is volumetric.
- **One filter scale**: `filter_m` = max(refine step, footprint) is the sole LOD
  input to `get_cloud_map_density`. The `coarse` ladder is deleted.

**Rejected in the round, by capture:** a chord-length budget floor on the step
(with the reach raised to 600 km so no frontier existed). It makes the step a
function of how long the ray happens to be: a ground-level horizon ray (chord
~600 km) got **~9.7 km steps through a 1.15 km deck** and the whole distant band
rendered as horizontal slabs — worse than the artifact being fixed. Budget
belongs in the REACH, not the step; the frontier is retained, now integrating
the footprint law in closed form.

**Verification (agent captures, compile-clean).** 150 / 400 / 900 km nadir
probes (`THALOS_SCREENSHOT_CAMERA_ALTITUDE` + `_LOOK_ELEVATION=-35`,
`artifacts/visual/runs/cloud-footprint-lod/`): individually resolved cells across
the whole visible hemisphere to the limb — no blanket, no fade-out, no ownership
seam. Runway and cruise unchanged; whole disc unchanged. GPU on the development
RTX 4070 Ti at 1280×720: cruise 1.98 ms mean / 3.33 ms p95, 400 km nadir
**1.31 ms / 1.94 ms** — the orbital framing is *cheaper* than cruise, as the
chord geometry predicts. Both inside the 3.5 ms target.

**Round 11b — the edge-on deck (2026-07-26).** The user's follow-up verdict
confirmed orbital ("this approach is working") and flagged the level view:
clouds "a bit ghost-ish when looking at them straight on… we can easily see
through them, with straight artifacts cutting through". Two symptoms, one
cause — the gaps between the artifacts *are* the see-through. Full diagnosis,
including four falsified hypotheses,
[INC-20260726T012035Z](../incidents/20260726T012035Z-grazing-ray-step-floor-bricked-the-deck.md).

- **Cause: `CLOUD_MARCH_MIN_STEP_M` was a cost-only floor that overrode the
  footprint step at the ranges that matter.** At 30–60 km a pixel covers
  35–70 m, so the honest broad step is ~100 m — the floor forced 600 m, and a
  grazing ray striding 600 m through a ~1.15 km deck under-resolves density
  along the ray, with the stride's phase against the deck varying smoothly in
  screen-y. Measured on the level probe: 600 m banded, **300 m clean**, 150 m
  clean. Cost at cruise: 1.98 / 3.33 → **2.36 / 4.38** → 2.74 / 5.12 ms
  (mean / p95). 300 m accepted; the 3.5 ms p95 target is explicitly provisional
  and the artifact is a correctness defect.
- **Found and fixed on the way:** `cloud_surface_shape` reconstructed the four
  strata **piecewise-linearly**, so its slope broke at layer heights 1/8, 3/8,
  5/8, 7/8; `env` feeds `formation_threshold`, so a deck viewed edge-on rendered
  four horizontal shelves per cloud. Now a C1 Catmull-Rom reconstruction in all
  three lockstep sites (`cloud_surface_shape`, `fill_lut::surface_shape`,
  `cloud_surface_density_cpu`); `FILL_LUT_VERSION` 6 → 7. It cleaned the near
  field but not the distant band, so it is not the same defect.
- **Trap worth remembering:** `THALOS_SCREENSHOT_CLOUD_QUALITY=reference` no
  longer discriminates sampling defects the way it used to. Since round 11 the
  step *size* comes from the footprint law, so `reference` raises the step
  COUNT and leaves the stride untouched — it reproduced the artifact identically
  and nearly sent the investigation the wrong way.

**Open:** coverage still reads high at oblique framings (BL-20260725T042404Z);
the whole-disc producer smears; and the cloud *underside* still reads flat grey
from below — that is the lighting budget (round 8's territory), not density,
and needs the user's eye to judge.

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
