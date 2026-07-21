# Planet-scale volumetric clouds

**Status:** active program, 2026-07-21. CLOUD-0 and CLOUD-1 are complete;
CLOUD-2 and CLOUD-3 are in progress, with the first density/range fidelity
checkpoint captured on `codex/cloud-0`. This document is the strategy and technical plan;
[backlog.md](backlog.md) is the execution queue, while
[atmosphere.md](atmosphere.md) remains the spec for what the renderer ships
today. Architecture choices are fixed by [ADR-0007](adr/0007-one-weather-field-many-cloud-projections.md).

The target is a Blackrack-class cloud system for a surface-to-orbit flight
camera: shaped volumes that can be entered, stable planetary weather seen from
orbit, cloud shadows and aerial shafts that agree with the visible volume, and
one atmosphere/light environment shared with terrain, craft, structures, and
water.

Planning input: the supplied *Planet-scale volumetric clouds for spaceflight*
note, backed by the public Nubis/Horizon Zero Dawn, Nubis Evolved, Frostbite,
Hillaire, Skybolt, and Blackrack feature-level references named there.

The supplied KSP/Blackrack captures are the visual acceptance bar: coherent
planetary storm systems rather than a repeating deck; distinct stratus,
cumulus, congestus, cumulonimbus, and anvil silhouettes; cauliflower-scale
detail riding on kilometre-scale mass; deep but coloured self-shadow; bright
energy-bounded rims; warm sunset transport; and continuity from runway to
inside-volume flight to orbit. A renderer that is merely clean or technically
volumetric does not meet this target.

## 1. What Thalos already has

This is an upgrade, not a greenfield renderer. Preserve these foundations:

- A body-fixed, planet-centred spherical-shell march now owned by
  `thalos_body_render::clouds`; it already handles cameras below, inside, and
  above the layer without a flat-sky assumption.
- `big_space`-safe camera inputs and planet-fixed noise/weather coordinates.
- An authored `CloudClimate` and per-body `CloudWeatherField`, projected as a
  256×256×6 RGBA cubemap carrying coverage/type/base/top.
- Per-pixel cloud hit distance and deterministic composition inside the
  fullscreen `BodySky` atmosphere pass, including opaque-scene occlusion.
- Static-view accumulation and moving-view reprojection in the body-fixed
  frame.
- A first orbital projection in `SolidPlanetMaterial` sampling the same
  weather cubemap as the near renderer.

The current result falls short for structural reasons:

1. **The first CLOUD-2/3 checkpoint is not the finished reconstruction.** Near
   density now consumes coverage/type/base/top and the orbit layer samples the
   same continuous multi-scale field, but history still lacks neighborhood
   clamp/moments and the far projection lacks density-derived height/normal
   moments.
2. **The density foundation is now genuinely volumetric.** The extruded 2-D
   atlas is gone; a 64³ Perlin/Worley basis, typed vertical profiles, local
   base/top, boundary-only erosion, a decorrelated macro threshold, and a
   50 km spherical march produce coherent bodies from runway through cruise.
   Empty-space hierarchy and a reduced-detail mid-distance regime remain.
3. **Clouds inhabit a private lighting universe.** The compute shader receives
   hand-scaled sun and top/bottom ambient colours. It does not consume the
   atmosphere transmittance, sky-view environment, eclipse, or the shared
   direct-sun visibility path.
4. **The cost model is improved but not complete.** The fixed cloud target is
   now 1280×720 and persistent allocation is 42.63 MB instead of 137.35 MB,
   but it is not yet viewport-relative and still lacks 3×3/1-in-9 scheduling,
   neighborhood-clamped history, empty-space hierarchy, and regime-specific
   detail.
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
              BodySky + shared lighting/shadow/environment consumers
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
RGBA32F basis. Weather and a cheap max-density hierarchy must reject empty rays
before detail sampling.

**Physical-scale invariant:** an authored feature scale names the feature, not
the full period of a stored tile containing several cells. The current erosion
channel contains eight cells per volume axis, so its sampled tile period is
`detail_scale_m * 8`. Treating `detail_scale_m` itself as the period creates
~56 m cells from the authored 450 m scale, below the 200–500 m horizon step and
recurs as stipple/micro-cloudlets; see [INC-0007](incidents/0007-cloud-detail-period-eighth-scale.md).

### 3.3 View march and reconstruction

Keep the exact spherical shell intersection. Replace the fixed 25 km reach with
regime-aware sampling:

- fine steps and full detail near/inside cloud;
- coarser steps and fewer detail octaves through the distant deck;
- empty-space leaps where weather/max-density data prove a segment clear;
- early exit on optical depth;
- orbital projection once a volume sample is sub-pixel.

Render into viewport-relative internal targets. High quality should begin with
a rotating 3×3 topology (one ninth of full pixel work per frame), then
body-fixed reprojection, depth/moment validation, neighborhood clamp, and
bilateral full-resolution reconstruction. History rejection must cover camera
cuts, FOV/resolution changes, body switches, disocclusion, weather-version
changes, large wind displacement, and timewarp jumps. A screenshot mode renders
all samples at high step counts with temporal reuse disabled.

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

## 4. Implementation phases

| ID | Phase | Outcome | Depends on | Est. |
|----|-------|---------|------------|------|
| CLOUD-0 | Baseline, probes, and budgets | Reproducible current captures, GPU/memory baseline, cloud-specific headless presets, acceptance matrix | — | S–M |
| CLOUD-1 | Canonical ownership and schema | `CloudClimate` + per-body `CloudWeatherField`; `None` is authoritative; mechanism moves under `body_render`; legacy slab/reference ownership deleted; first weather-derived orbit layer | CLOUD-0 | M–L |
| CLOUD-2 | Scalable targets and temporal reconstruction | Viewport-relative low-res/interleaved march, robust history rejection/clamp/upscale, screenshot mode, quality ladder | CLOUD-1 | L |
| CLOUD-3 | Multi-scale density and range | Weather cube with type/base/top, true 3-D base/detail noise, vertical profiles, anti-tiling, empty-space skip, near-to-horizon regime LOD | CLOUD-2 | L |
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
artifact inventory are recorded in `docs/cloud_baseline.md`. The 1440p High
sunset probe measures 11.06 ms mean on the development RTX 4070 Ti versus the
provisional 3.5 ms target; persistent cloud textures total 135.98 MiB. The user
confirmed that fidelity across the supplied captures—not one isolated artifact—is
the priority and supplied the Blackrack/KSP acceptance bar above.

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

**Status (2026-07-21):** captured and compile-clean on `codex/cloud-0`; both
phases remain `wip` because their complete exit criteria are larger than this
vertical slice.

- Cloud colour/distance/history targets moved from 1920×1080 to 1280×720;
  the 1920² base atlas was deleted, the generated volume moved 32³ → 64³, and
  persistent cloud allocation fell from 137.35 MiB after CLOUD-1 to 40.66 MiB
  (42,631,168 bytes).
- Density now uses a wrap-first trilinear 3-D Perlin/Worley volume, typed
  stratus/cumulus/storm profiles, spatial base/top, storm anvils, a second
  macro threshold domain, boundary-only 450 m erosion, and 50 km bounded
  spherical range. The aliased detail-scale failure and diagnosis are retained
  in INC-0007.
- The canonical weather cube now combines synoptic, mesoscale, and cellular
  coverage. Near volume and first BodySky/SolidPlanet orbital projections use
  that same field; the 200 km limb probe shows broken systems rather than a
  missing layer or a uniform white shell.
- Final baseline probes at 1920×1080 report 8.34 ms mean / 8.38 ms p95 for the
  densest runway view, 1.20 / 1.21 ms for cruise, and 4.71 / 4.73 ms for the
  limb on the development RTX 4070 Ti. These are checkpoint measurements, not
  the CLOUD-2 budget exit: viewport-relative targets, sparse scheduling,
  history clamp/moments, and empty-space skipping remain.
- Lighting has darker cores, deterministic sparse self-shadow, and a first
  solar-elevation tint, but CLOUD-4 still owns atmosphere LUT coupling and
  correct foreground/background media ordering. CLOUD-5 still owns world
  cloud shadows; CLOUD-6 still owns density-derived orbital optical-depth,
  height, and normal moments.

### Program acceptance matrix

| Scenario | Pass condition |
|----------|----------------|
| Runway under broken cumulus | Stable detail, soft self-shadow, blue/overcast fill, terrain/craft shadows align with visible cells |
| Climb through deck | Continuous extinction and visibility; no sheet pop; rapid yaw rejects history rather than smearing |
| Cruise beside convection | Kilometer-scale silhouettes plus local cauliflower detail; no obvious repeated atlas cells |
| Low-orbit limb | Cloud line lies inside the aerosol limb; reduced-volume/orbit handoff is invisible in motion |
| High orbit/map | Planet-wide fronts stay coherent and match the weather/shadows seen below |
| Noon to sunset | Dark bases and bright rims remain energy-bounded; atmosphere transmittance gives natural warm edges |
| Timewarp/body rotation | Weather advances in simulation time; history resets cleanly with no trails or ground slip |
| Solid overcast | Sun, terrain, hull, structures, water, and environment ambient all respond to the same cloud field |

## 5. Risks and decision gates

These choices are accepted in ADR-0007 and form the constraints for every
later phase:

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
