# Terrain Feature Compiler

This document describes the replacement terrain-generation architecture for
Thalos. The current stage pipeline remains useful as a migration target and
visual feedback loop, but the long-term model is feature-first: authored and
procedural terrain are represented as stable, seed-addressable features that
compile into render-, physics-, and editor-friendly products.

## Goals

- Generate decent initial bodies from physical and historical parameters.
- Let authors keep good generated results while rerolling bad local results.
- Represent terrain from planet-scale provinces down to local scatter.
- Keep generation and terrain queries independent of Bevy and any renderer.
- Feed the existing flat impostor renderer with compact baked products.
- Eventually replace the old ordered stage list as the source of truth.

## Pipeline Shape

```
PlanetTerrainSpec
  -> TerrainPrior
  -> FeatureManifest
  -> SurfaceField
  -> projections:
       - impostor bake
       - landed tile data
       - physics sampling
       - editor provenance
```

The old `Stage` trait mutates `BodyBuilder` directly. The replacement compiler
separates planning from terrain evaluation:

- The prior infers budgets and tendencies from physical input.
- Feature generation builds a semantic graph of accepted features.
- Compilation projects that graph into continuous sphere-native fields.
- Cubemap, tile, physics, and editor outputs are projections of those fields.
- Renderers consume projections, never the full generator.

## Input State

The initial authored input should be small but meaningful:

```ron
PlanetTerrainSpec(
    body_id: "vaelen",
    root_seed: 912831,
    physical: (
        radius_m: 1_130_000,
        gravity_m_s2: 2.06,
        age_gyr: 4.3,
        stellar_flux_earth: 0.33,
        rotation_hours: Some(28.0),
        obliquity_deg: Some(24.0),
        atmosphere: ThinCo2(pressure_bar: 0.015),
        hydrosphere: AncientLost,
        ice_inventory: Moderate,
        composition: BasalticSilicate,
    ),
    archetype: ColdDesertFormerlyWet,
    intent: [
        ReadAsFirstInterplanetarySurfaceWorld,
        ForgivingLandingTerrain,
        VisibleAncientWaterStory,
    ],
)
```

The generator should not convert this directly into noise. It should infer a
`TerrainPrior`: crater retention, erosion strength, resurfacing budgets,
sediment mobility, ice stability, material palette, and feature budgets.

## Feature Identity

Every meaningful terrain contribution has a stable feature ID and independent
seed streams:

```rust
FeatureSeed {
    identity,
    placement,
    shape,
    detail,
    children,
}
```

This is the core authoring mechanism. A basin can keep its placement while its
mare fill is rerolled. A crater can keep its shape while ray detail changes. A
generated rock cluster can be promoted to authored data without freezing an
entire planet.

Root seeds are only defaults. Authored overrides may replace any feature seed
or lock any feature at a chosen level.

## Feature Manifest

Feature generation produces a manifest, not immediate terrain mutation:

```rust
FeatureInstance {
    id,
    kind,
    parent,
    seed,
    era,
    footprint,
    scale_range_m,
    params,
    lock,
    children,
}
```

The manifest is semantic and inspectable. It can answer editor questions such
as "which feature created this ridge?" or "what children belong to this basin?"

## MVP Feature Kinds

The MVP set covers Mira, Vaelen, and the planned terrestrial bodies without
committing to the full long-term feature catalogue. Each kind is shared
across bodies; per-body variation comes from biome budgets and authored
overrides. Some kinds appear in the manifest as singletons, some as
populations — see *Populations and Promotion* below.

- **CraterPopulation / Crater** — unified across all bodies. The compiler
  emits every individual crater from one or more `CraterPopulation` nodes
  (parameterized by density, size distribution, `age_bin` distribution
  (fresh / mature / ancient / ghost), and per-biome density modulator). All
  emitted instances live in the SSBO + spatial hash. Promotion turns a
  specific emitted crater into a singleton `Crater` with `lock: Placement`.
  A sufficiently large authored Crater can register a biome around itself
  (subsuming the megabasin role by scale).
- **Channel** — incised drainage feature. A few seeded curves with branch
  detail; not a full hydrology graph for MVP. Singletons today (small
  numbers per body); a future `ChannelNetwork` population can take over
  once full hydrology lands.
- **Rift** — large hand-placed canyon or graben. Singleton. Used for
  Valles-Marineris-class features that channel cannot do.
- **DuneSeaPopulation / DuneSea** — anisotropic dune field bounded to a
  parent biome, evaluated with the layered-aeolian recipe in `dunes.md`.
  Each named region is its own population; multiple regions per body get
  separate populations rather than one giant generator.
- **ShieldVolcano** — Olympus-Mons-class shield. Singleton, hand-placed.
  Piecewise-radial profile (gentle flank, summit caldera depression, basal
  scarp cliff) plus a `RadialFlow` biome around the flanks.

Cratering is the shared spine that every body draws on (via
`CraterPopulation`). The other four are optional and per-body.

## Populations and Promotion

The manifest is a list of *named, hand-touchable* entries — not every pixel
contribution. A typical body has on the order of 10-20 manifest nodes,
regardless of how many surface features the compiler ends up emitting.

Manifest entries split into two shapes:

- **Singletons** — a single named feature with a specific placement.
  Authored by hand or promoted from a population. Examples: a megabasin you
  placed, the Olympus-class shield, a Valles-class rift, a notable crater
  you decided to keep.
- **Populations** — generators that emit many instances at compile time
  from a small parameter set. Examples: `CraterPopulation` (tens of
  thousands of craters from density × size distribution × age-bin
  distribution × seed), `DuneSeaPopulation`, `MareFlood`. Their emitted
  instances are baked into the height cube and analytic feature buffers
  (SSBO + spatial hash); they never become individual manifest entries.

A population owns the same four seed substreams as any other feature.
Reroll its `placement` substream and every emitted crater moves; reroll its
`shape` and every rim/depth profile changes; per-instance authoring stays
the business of singletons.

**Promotion** is how an individual emitted instance becomes a singleton.
Click the planet in the editor: provenance reports "this pixel is dominated
by Population P, instance #N". A *Promote* action:

1. Reads the resolved params for that instance (center direction, radius,
   age, depth, etc.).
2. Inserts a new singleton FeatureInstance carrying those params with
   `lock: Placement` (or further locks).
3. Adds an exclusion ID to the population so the next compile skips that
   index, leaving the authored copy in its place.

That is the only path by which an individual crater (or dune, or channel
segment) enters the manifest. Until promoted, individuals are governed
entirely by their population's seeds.

For Mira, a typical manifest is roughly: 3 biomes + 2 megabasin singletons
+ 1 mare-flood population + 1 crater population + 1 regolith population + 1
space-weather modifier ≈ 9 entries. For Vaelen: 5-6 biomes + 1 shield + 1
rift + 1 megabasin + 3-4 named channels + 1 crater population + 1 dune-sea
population + 1 dust mantle ≈ 15-18 entries.

## Terrain Biomes

Biomes are the broad substrate/process fields under the feature graph. They
are not limited to Earth climate classes: an airless moon can have highland
regolith, mare basalt, and fresh ejecta; a cold desert can have rust dust
plains, dune seas, evaporite basins, volcanic plains, and badlands.

Each planned biome owns:

- a stable biome id
- a height-generator stack
- a palette function `(altitude_in_biome, slope, curvature, feature_proximity)
  -> (albedo, roughness)` (see Render Projection)
- feature budgets for terrain that should be spawned inside or across it

`HeightGenerator` is the primitive layer. Per-biome stacks compose peer
variants:

- `DerivFBM` — IQ-style derivative fBM, the current default
- `FloodBasin` — low-amplitude smooth fill that settles in low pockets
- `ErodedPlain` — derivative-fBM with stronger valley bias
- `BroadSwell` — gentle large-wavelength dome
- `EtchedPlateau` — stratified raised blocks separated by deep narrow gaps
- `OrientedDune` — anisotropic dune-wave generator (see `dunes.md`)
- `RadialFlow` — radial domain warp, used for shield-volcano flanks

Later generators (DLA mountain networks, yardangs, channel roughness) should
be added as peer variants rather than special-case stages. Features remain
separate from biomes: biomes describe the broad field, while features are the
discrete or structured overprints that live in that field. Some features
(megabasin-scale craters, shield volcanoes) also register a biome around
themselves, so the biome graph is partly authored and partly feature-spawned.

Biome placement is evaluated by reusable mask plans. A `BiomeMaskPlan` samples
direction, deterministic seed streams, and archetype-provided scalar signals
into normalized biome weights. The archetype still decides which signals matter
for its geology, but the scoring language is shared: caps, fBM masks,
smoothsteps, weighted sums, products, clamps, and intermediate named scores.

## Authoring Posture

Authoring is layered. Each layer adds control without invalidating the
previous one:

1. **Parameter dial.** `PlanetTerrainSpec` (physical params, archetype, intent)
   compiles via the prior into a default manifest. Roll the root seed; get a
   plausible body. Most bodies start and end here.
2. **Sketch.** Open the body in `planet_editor`. Click on the planet to add or
   move authored anchors: biome centers, megabasin-scale craters, rifts,
   shield volcanoes, dune seas. Authored anchors enter the manifest as
   `lock: Placement` features; the next compile honors them.
3. **Roll and tune.** For any feature in the manifest (procedural or
   authored), reroll a single seed substream (placement / shape / detail /
   children) or adjust params in the inspector. Lock granularity is
   per-feature.

Promotion is global. When the compiled body looks right, save: the entire
manifest serializes back to the body's RON. There is no separate "promotions"
sidecar — the manifest is the source of truth for both the editor and the
headless `bake_dump`, so the editor and the bake CLI must produce identical
output from identical RON.

## Era Ordering

Features are ordered by geological era so later history can overprint earlier
history:

1. Crust formation
2. Heavy bombardment
3. Ancient tectonics and resurfacing
4. Ancient hydrology, ice, or sedimentation
5. Recent impacts and surface modification
6. Present local detail

Mira mostly preserves its impact record. Vaelen needs more overprinting:
ancient impacts may become lake basins, evaporite floors, dust mantles, or
wind-eroded plains.

## Render Projection

The first required projection is the existing flat impostor shader.

The shader should remain boring:

```text
direction on unit sphere
  -> sample cubemaps
  -> optionally iterate compact analytic buffers
  -> shade a mathematically flat sphere
```

It should not evaluate the feature graph. The compiler decides what becomes
baked cubemap data and what remains analytic.

Current migration shape:

```text
FeatureManifest
  -> SurfaceField::sample(dir, sample_scale_m)
  -> bake_surface_field_into_builder()
  -> BodyData cubemaps + analytic buffers
```

`SurfaceField` is the render-agnostic contract. For a direction on the unit
sphere it returns height, a body-local normal contribution (`normal_local`),
and `(albedo, roughness)`. Color is not a separately stored field but the
result of evaluating the active biome's palette function on local physical
fields (altitude-in-biome, slope, curvature, feature proximity). At biome
boundaries the SurfaceField blends adjacent palette evaluations weighted by
biome membership. There is no discrete material ID and no separate marbling
layer: visual variation emerges because the inputs to the palette functions
(altitude, slope, curvature) are themselves multi-scale fields driven by the
biome's height-generator stack and the features overlaid on it.

Curvature, when needed by a palette function during bake, is computed from
the height cubemap by finite differencing on a smoothed input (e.g., a coarser
mip of the height cube). It is not stored as its own channel for MVP. If
sharp curvature gating later produces visible speckling, fall back to baking
a smoothed-curvature R8 channel.

Impostor contract (consumed directly by `planet_impostor.wgsl`):

- `albedo_cubemap` (`Rgba8UnormSrgb`): primary surface color, sampled bilinearly.
  No discrete material indirection — biome boundaries blend continuously.
- `height_cubemap` (`R16Unorm`): elevation used for per-fragment normals via
  `perturb_normal_from_height` (finite-difference at fragment time, full f32
  precision) and the self-shadow ray march.
- `roughness_cubemap` (`R8Unorm`): per-texel microsurface response, wired into
  the Hapke BRDF's opposition surge width.
- `feature_buffers` + `feature_index`: SSBO craters and the spatial hash that
  walks them. Crater LOD fades sub-pixel features at far zoom.

Also baked into `BodyData` but reserved for non-impostor consumers:

- `material_cubemap` (R8Uint dominant ID): used internally by stages
  (`PaintBiomes`, `MareFlood`, `SpaceWeather`) and by the CPU `sample()` path.
  Not bound to the impostor's GPU bind group.
- `normal_cubemap` (`Rgba8Unorm` object-space): reserved for ground LOD where
  chunked geometry can't cheaply finite-difference height at runtime. 8-bit
  encoding crushes shallow slope angles, so the impostor reconstructs normals
  from the height cube at fragment time instead.
- `materials` palette: still authored by stages for CPU sampling and for
  ground-LOD detail-texture blending; no longer uploaded as a GPU storage buffer.

Silhouette displacement is intentionally out of scope for the first projection.
Good albedo, normals, roughness, and compact meso features are enough for
Mira and Vaelen to read from orbit.

## Example Bodies

### Mira

Mira is the first proof target because it is feature-rich but physically simple:
an airless, tidally locked, silicate moon with a visible near-side identity.

Expected root features:

- global crust
- near-side megabasin A
- near-side megabasin B
- far-side highlands
- crater population
- mare flooding
- regolith garden
- space weathering

The key authoring loop is seed-local rerolling: keep a good basin, reroll its
secondary craters, lock its mare fill, or promote a fresh ray crater.

### Vaelen

Vaelen is the second proof target because it exercises history: thin
atmosphere, ancient wet past, sedimentary basins, evaporites, buried ice,
aeolian modification, and moderate crater preservation.

Expected biomes (each carrying a palette function of physical fields):

- dark volcanic / impact-melt lowlands (`FloodBasin` generator)
- pale sediment / evaporite lowlands (`ErodedPlain`)
- rust-highland dust mantle (`BroadSwell` plus dust modifier)
- etched plateaus and mesas (`EtchedPlateau`)
- one or two dune seas (`OrientedDune`)
- shield-volcano flanks (`RadialFlow`, biome spawned by the volcano feature)

Expected features:

- crater population (degraded, density modulated per biome)
- a few channel/canyon systems sourced in pale lowlands
- one Valles-class rift
- one Olympus-class shield volcano (authored placement)

The key test is era overprinting. The same feature graph must express an
original basin, its degraded rim, later sediment fill, evaporite floor, and
present dust or aeolian erosion — without the biome boundaries reading as
flat color zones from orbit.

## Migration Plan

1. Add feature compiler data types and deterministic feature seeding.
2. Infer `TerrainPrior` from `PlanetTerrainSpec`.
3. Generate initial feature manifests for Mira and Vaelen.
4. Compile those manifests into the current `BodyData` render contract.
5. Switch body definitions from `pipeline: [...]` to feature specs one body at
   a time.
6. Retire old stages after all bodies are compiled through the feature graph.

During migration, `BodyData` remains the renderer handoff. The feature compiler
becomes the new source of truth before the renderer changes.

## Asset Schema

Bodies now have a normalized terrain route:

```ron
terrain: Feature((
    seed: 1004,
    cubemap_resolution: 2048,
    body_age_gyr: 4.5,
    archetype: AirlessImpactMoon,
    composition: SilicateDominated,
    environment: (
        stellar_flux_earth: 1.0,
        atmosphere: None,
        hydrosphere: None,
        ice_inventory: None,
    ),
    intent: [
        ReadAsMoon,
        DistinctNearSideFace,
        DifferentFarSide,
        FirstLandingWorld,
    ],
    authored_features: [
        Megabasin((
            id: "mira.near_side_megabasin_a",
            center_dir: (1.0, 0.12, 0.24),
            radius_km: 250.0,
            depth_km: 6.0,
            lock: Placement,
        )),
    ],
))
```

## Current Implementation Status

`AirlessImpactMoon` and `ColdDesertFormerlyWet` are wired up: a
`PlanetTerrainSpec` is expanded into a `FeatureManifest`, then compiled into the
existing `BodyData` contract using the current bake primitives. This keeps the
flat impostor renderer working while moving source-of-truth terrain identity and
seeding into the feature compiler. `AgingOceanicHomeworld` and
`GenericTerrestrial` are not yet implemented — Thalos and Pelagos render via the
flat-water `TerrainConfig::Ocean` placeholder until the terrestrial pipeline lands.
