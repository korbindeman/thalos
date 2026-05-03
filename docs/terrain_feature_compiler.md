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

## Terrain Biomes

Biomes are now the broad substrate/process fields under the feature graph.
They are not limited to Earth climate classes: an airless moon can have
highland regolith, mare basalt, and fresh ejecta; a cold desert can have rust
dust plains, dune seas, evaporite basins, volcanic plains, and badlands.

Each planned biome owns:

- a stable biome id and material class
- base albedo/roughness for projection
- a height-generator stack
- feature budgets for terrain that should be spawned inside or across it

`HeightGenerator` is the primitive layer. The current implemented generator is
IQ-style derivative fBM, wrapped by `HeightGeneratorStack`; later generators
such as DLA mountain networks, dune waves, yardangs, channel incision, or
crater roughness should be added as peer variants rather than special-case
terrain stages. Features remain separate from biomes: biomes describe the
broad field, while features are the discrete or structured overprints that
live in that field.

Biome placement is evaluated by reusable mask plans. A `BiomeMaskPlan` samples
direction, deterministic seed streams, and archetype-provided scalar signals
into normalized biome weights. The archetype still decides which signals matter
for its geology, but the scoring language is shared: caps, fBM masks,
smoothsteps, weighted sums, products, clamps, and intermediate named scores.

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

`SurfaceField` is the render-agnostic contract. It returns height, linear
albedo, roughness, and a weighted material mix for a direction on the unit
sphere. The existing impostor projection still collapses the material mix to a
dominant material ID because `BodyData` has only one material byte today; that
collapse is a compatibility adapter, not the long-term source model.

Initial impostor contract:

- `height_cubemap`: low/mid-frequency elevation used for normals and masks.
- `albedo_cubemap`: orbital-scale color.
- `material_cubemap`: semantic material palette index.
- `feature_buffers`: compact meso-scale analytic buffers.
- `feature_index`: spatial lookup for analytic buffers.
- `materials`: palette consumed by renderers and sampling.

Silhouette displacement is intentionally out of scope for the first projection.
Good albedo, normals, material masks, and compact meso features are enough for
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

Vaelen is the second proof target because it exercises history:
thin atmosphere, ancient wet past, sedimentary basins, evaporites, buried ice,
aeolian modification, and moderate crater preservation.

Expected root features:

- crustal provinces
- ancient highlands
- sedimentary lowlands
- impact basin archive
- ancient channel networks
- evaporite basins
- buried ice zones
- volcanic plains
- aeolian mantle
- recent craters

The key test is era overprinting. The same feature graph must express an
original basin, its degraded rim, later lake fill, evaporite floor, and present
dust or wind erosion.

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

During migration, legacy bodies are normalized as
`terrain: LegacyPipeline(...)` by the loader when they still author the old
`generator` block.

## Current Implementation Status

The first compatibility projection exists for `AirlessImpactMoon`, and Mira in
`assets/solar_system.ron` now uses `terrain: Feature(...)`. A
`PlanetTerrainSpec` is expanded into a `FeatureManifest`, then compiled into the
existing `BodyData` contract using the current bake primitives. This keeps the
flat impostor renderer working while moving source-of-truth terrain identity and
seeding into the feature compiler.
