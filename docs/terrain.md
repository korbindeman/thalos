# Terrain

The unified spec for Thalos terrain: generation, rendering, and the
contract that connects them.

This doc covers both halves of the terrain system, which were
previously split across two architecture documents. Per-body process
notes (`gen/vaelen_processes.md`) and the wider research surveys
(`gen/terrestrial_pipeline_research.md`,
`gen/planet_aesthetics.md`, `gen/dunes.md`) remain as standalone
references.

> **Status (2026-05): generation half superseded.** The
> generation architecture below (feature compiler: `PlanetTerrainSpec →
> TerrainPrior → FeatureManifest → SurfaceField`, populations/promotion,
> era ordering, the v2 backlog) is being replaced by the
> [planet generation pipeline spec](planet-generation-pipeline-spec.md),
> via the [migration plan](planet-generation-pipeline-migration.md).
> Its concepts are folded into the new pipeline (archetypes → field
> presets, manifest → feature catalog, v2 backlog → fields + feature
> types + synthesisers), not discarded — keep this section as the
> reference being ported until cutover completes. The **ground-LOD /
> `TileProvider` half (M3) and the cross-renderer projection contract
> remain current**, reframed by the migration as Query-API consumers.

## Overview

Two halves, one contract:

1. **Generation** (`thalos_terrain`, pure Rust, no Bevy). A
   feature-first compiler. Inputs: a `PlanetTerrainSpec`. Outputs:
   `PlanetSurface`, split into immutable `StaticSurfaceData` (cubemap
   textures + analytic feature buffers) and `DynamicSurfaceLayers`
   (authored/compiled layer definitions for ice caps, active dunes, and
   future time-varying overprints). Source of truth for terrain identity.
2. **Rendering**. Today: a flat impostor renderer
   (`planet_impostor.wgsl`) consumes static cubemaps plus dynamic layer
   buffers built from `PlanetSurface + DynamicSurfaceState`. Future (M3):
   a forked `thalos_udlod` provides UDLOD on an ellipsoidal cubesphere,
   with the same sampling pipeline plugged in via a runtime `TileProvider`
   trait.

The bridge between the two halves is the **`SurfaceField` sample
contract** on the generation side and the **`TileProvider` trait** on
the rendering side. Both will hold a `TileTree` per (terrain, view)
pair once ground LOD lands.

## Status

| Area | Today | Target |
|---|---|---|
| Feature compiler | `AirlessImpactMoon` and `ColdDesertFormerlyWet` archetypes wired (Mira, Vaelen). v1 has no first-class hydrology, no layered substrate, no climate fields. | Revamped compiler with v2 backlog landed; all four main bodies through it (M2) |
| Terrestrial bodies | Thalos, Pelagos use `Ocean` flat-water placeholder | `AgingOceanicHomeworld`, `GenericTerrestrial` archetypes (M2) |
| Renderer (orbital) | Flat impostor reads `StaticSurfaceData` cubemaps + crater SSBO | Same impostor remains the far-orbit projection |
| Renderer (surface) | In-tree `thalos_udlod` fork (Bevy 0.18 + runtime `TileProvider`); `PipelineTileProvider` reads the Query API surface into UDLOD tiles. Height tiles use packed `Rg16` for the game path because broad, shallow Thalos slopes made plain `R16Unorm` quantization visible as contour terraces; matching height queries mirror the resident atlas. | Mira/Vaelen/Thalos/Pelagos rendering at surface scale with future mid/high-frequency detail projection added to both renderer and collider source together |
| Big-space hierarchy | Not present | Per-body grids parented to system grid (M1, M3) |

## Goals

- Procedural-first: no baked planet-scale heightmaps. The synthesis
  pipeline is queried on demand at the resolution required by the
  current LOD.
- Seamless camera traversal from orbital altitude to surface walking
  scale (~cm features) on any body, with no LOD popping or precision
  artifacts.
- Per-body parameterization: every body has its own ellipsoid shape,
  height range, and feature manifest, including realistic oblateness
  for fast rotators.
- Deterministic and stable feature identity. Authored anchors and
  rerollable feature seeds let authors keep good generated results
  while rerolling bad local ones.
- Compatible with the patched-conics orbital model: planet local
  frames inherit orbital and rotational motion via nested `big_space`
  grids.
- Single sampling contract (`SurfaceField`) consumed by every
  projection: bake, tile, physics, editor.
- Hard cache boundary: immutable substrate is cached as `StaticSurfaceData`;
  dynamic layer definitions/state rebuild only dynamic buffers and do not
  invalidate the static terrain cache unless they also alter substrate.

## Non-goals (this doc)

- Atmospheric scattering, ocean rendering, gas-giant impostors, sun
  rendering. Separate subsystems even if they ultimately render
  alongside terrain. See [atmosphere.md](atmosphere.md).
- Asset streaming for non-terrain content.
- Rover wheels and high-fidelity surface interaction. The M5 first
  slice has only an aggregate landing collider patch for Thalos.

---

## Generation: feature compiler

The replacement terrain-generation architecture for Thalos. The
existing flat impostor renderer remains the migration target and
visual feedback loop. The long-term model is feature-first: authored
and procedural terrain are represented as stable, seed-addressable
features that compile into render-, physics-, and editor-friendly
products.

### Pipeline shape

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

The old `Stage` trait mutates `BodyBuilder` directly. The replacement
compiler separates planning from terrain evaluation:

- The prior infers budgets and tendencies from physical input.
- Feature generation builds a semantic graph of accepted features.
- Compilation projects that graph into continuous sphere-native
  fields.
- Cubemap, tile, physics, and editor outputs are projections of those
  fields.
- Renderers consume projections, never the full generator.

### Input state

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

The generator should not convert this directly into noise. It should
infer a `TerrainPrior`: crater retention, erosion strength,
resurfacing budgets, sediment mobility, ice stability, material
palette, and feature budgets.

### Feature identity

Every meaningful terrain contribution has a stable feature ID and
independent seed streams:

```rust
FeatureSeed {
    identity,
    placement,
    shape,
    detail,
    children,
}
```

This is the core authoring mechanism. A basin can keep its placement
while its mare fill is rerolled. A crater can keep its shape while ray
detail changes. A generated rock cluster can be promoted to authored
data without freezing an entire planet.

Root seeds are only defaults. Authored overrides may replace any
feature seed or lock any feature at a chosen level.

### Feature manifest

Feature generation produces a manifest, not immediate terrain
mutation:

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

The manifest is semantic and inspectable. It can answer editor
questions such as "which feature created this ridge?" or "what
children belong to this basin?"

### MVP feature kinds

The MVP set covers Mira, Vaelen, and the planned terrestrial bodies
without committing to the full long-term feature catalogue. Each kind
is shared across bodies; per-body variation comes from biome budgets
and authored overrides. Some kinds appear in the manifest as
singletons, some as populations — see *Populations and Promotion*
below.

- **CraterPopulation / Crater** — unified across all bodies. The
  compiler emits every individual crater from one or more
  `CraterPopulation` nodes (parameterized by density, size
  distribution, `age_bin` distribution
  (fresh / mature / ancient / ghost), and per-biome density
  modulator). All emitted instances live in the SSBO + spatial hash.
  Promotion turns a specific emitted crater into a singleton `Crater`
  with `lock: Placement`. A sufficiently large authored Crater can
  register a biome around itself (subsuming the megabasin role by
  scale).
- **Channel** — incised drainage feature. A few seeded curves with
  branch detail; not a full hydrology graph for MVP. Singletons today
  (small numbers per body); a future `ChannelNetwork` population can
  take over once full hydrology lands (see "v2 backlog").
- **Rift** — large hand-placed canyon or graben. Singleton. Used for
  Valles-Marineris-class features that channel cannot do.
- **DuneSeaPopulation / DuneSea** — anisotropic dune field bounded to
  a parent biome, evaluated with the layered-aeolian recipe in
  `gen/dunes.md`. Each named region is its own population; multiple
  regions per body get separate populations rather than one giant
  generator.
- **ShieldVolcano** — Olympus-Mons-class shield. Singleton,
  hand-placed. Piecewise-radial profile (gentle flank, summit caldera
  depression, basal scarp cliff) plus a `RadialFlow` biome around the
  flanks.

Cratering is the shared spine that every body draws on (via
`CraterPopulation`). The other four are optional and per-body.

### Populations and promotion

The manifest is a list of *named, hand-touchable* entries — not every
pixel contribution. A typical body has on the order of 10-20 manifest
nodes, regardless of how many surface features the compiler ends up
emitting.

Manifest entries split into two shapes:

- **Singletons** — a single named feature with a specific placement.
  Authored by hand or promoted from a population. Examples: a
  megabasin you placed, the Olympus-class shield, a Valles-class
  rift, a notable crater you decided to keep.
- **Populations** — generators that emit many instances at compile
  time from a small parameter set. Examples: `CraterPopulation` (tens
  of thousands of craters from density × size distribution × age-bin
  distribution × seed), `DuneSeaPopulation`, `MareFlood`. Their
  emitted instances are baked into the height cube and analytic
  feature buffers (SSBO + spatial hash); they never become individual
  manifest entries.

A population owns the same four seed substreams as any other feature.
Reroll its `placement` substream and every emitted crater moves;
reroll its `shape` and every rim/depth profile changes; per-instance
authoring stays the business of singletons.

**Promotion** is how an individual emitted instance becomes a
singleton. Click the planet in the editor: provenance reports "this
pixel is dominated by Population P, instance #N". A *Promote* action:

1. Reads the resolved params for that instance (center direction,
   radius, age, depth, etc.).
2. Inserts a new singleton FeatureInstance carrying those params with
   `lock: Placement` (or further locks).
3. Adds an exclusion ID to the population so the next compile skips
   that index, leaving the authored copy in its place.

That is the only path by which an individual crater (or dune, or
channel segment) enters the manifest. Until promoted, individuals are
governed entirely by their population's seeds.

For Mira, a typical manifest is roughly: 3 biomes + 2 megabasin
singletons + 1 mare-flood population + 1 crater population + 1
regolith population + 1 space-weather modifier ≈ 9 entries. For
Vaelen: 5-6 biomes + 1 shield + 1 rift + 1 megabasin + 3-4 named
channels + 1 crater population + 1 dune-sea population + 1 dust
mantle ≈ 15-18 entries.

### Terrain biomes

Biomes are the broad substrate/process fields under the feature
graph. They are not limited to Earth climate classes: an airless moon
can have highland regolith, mare basalt, and fresh ejecta; a cold
desert can have rust dust plains, dune seas, evaporite basins,
volcanic plains, and badlands.

Each planned biome owns:

- a stable biome id
- a height-generator stack
- a palette function `(altitude_in_biome, slope, curvature,
  feature_proximity) -> (albedo, roughness)` (see *Render Projection*)
- feature budgets for terrain that should be spawned inside or across
  it

`HeightGenerator` is the primitive layer. Per-biome stacks compose
peer variants:

- `DerivFBM` — IQ-style derivative fBM, the current default
- `FloodBasin` — low-amplitude smooth fill that settles in low pockets
- `ErodedPlain` — derivative-fBM with stronger valley bias
- `BroadSwell` — gentle large-wavelength dome
- `EtchedPlateau` — stratified raised blocks separated by deep narrow
  gaps
- `OrientedDune` — anisotropic dune-wave generator (see
  `gen/dunes.md`)
- `RadialFlow` — radial domain warp, used for shield-volcano flanks

Later generators (DLA mountain networks, yardangs, channel roughness)
should be added as peer variants rather than special-case stages.
Features remain separate from biomes: biomes describe the broad
field, while features are the discrete or structured overprints that
live in that field. Some features (megabasin-scale craters, shield
volcanoes) also register a biome around themselves, so the biome
graph is partly authored and partly feature-spawned.

Biome placement is evaluated by reusable mask plans. A
`BiomeMaskPlan` samples direction, deterministic seed streams, and
archetype-provided scalar signals into normalized biome weights. The
archetype still decides which signals matter for its geology, but the
scoring language is shared: caps, fBM masks, smoothsteps, weighted
sums, products, clamps, and intermediate named scores.

### Authoring posture

Authoring is layered. Each layer adds control without invalidating
the previous one:

1. **Parameter dial.** `PlanetTerrainSpec` (physical params,
   archetype, intent) compiles via the prior into a default manifest.
   Roll the root seed; get a plausible body. Most bodies start and
   end here.
2. **Sketch.** Open the body in `planet_editor`. Click on the planet
   to add or move authored anchors: biome centers, megabasin-scale
   craters, rifts, shield volcanoes, dune seas. Authored anchors
   enter the manifest as `lock: Placement` features; the next compile
   honors them.
3. **Roll and tune.** For any feature in the manifest (procedural or
   authored), reroll a single seed substream
   (placement / shape / detail / children) or adjust params in the
   inspector. Lock granularity is per-feature.

Promotion is global. When the compiled body looks right, save: the
entire manifest serializes back to the body's RON. There is no
separate "promotions" sidecar — the manifest is the source of truth
for both the editor and the headless `bake_dump`, so the editor and
the bake CLI must produce identical output from identical RON.

### Era ordering

Features are ordered by geological era so later history can overprint
earlier history:

1. Crust formation
2. Heavy bombardment
3. Ancient tectonics and resurfacing
4. Ancient hydrology, ice, or sedimentation
5. Recent impacts and surface modification
6. Present local detail

Mira mostly preserves its impact record. Vaelen needs more
overprinting: ancient impacts may become lake basins, evaporite
floors, dust mantles, or wind-eroded plains.

### Render projection (impostor contract)

The first required projection is the existing flat impostor shader.

The shader should remain boring:

```text
direction on unit sphere
  -> sample cubemaps
  -> optionally iterate compact analytic buffers
  -> shade a mathematically flat sphere
```

It should not evaluate the feature graph. The compiler decides what
becomes baked cubemap data and what remains analytic.

Current migration shape:

```text
FeatureManifest
  -> SurfaceField::sample(dir, sample_scale_m)
  -> bake_surface_field_into_builder()
  -> PlanetSurface {
       static_surface: StaticSurfaceData cubemaps + analytic buffers
       dynamic_layers: DynamicSurfaceLayers
     }
```

`SurfaceField` is the render-agnostic contract. For a direction on
the unit sphere it returns height, a body-local normal contribution
(`normal_local`), and `(albedo, roughness)`. Color is not a separately
stored field but the result of evaluating the active biome's palette
function on local physical fields (altitude-in-biome, slope,
curvature, feature proximity). At biome boundaries the SurfaceField
blends adjacent palette evaluations weighted by biome membership.
There is no discrete material ID and no separate marbling layer:
visual variation emerges because the inputs to the palette functions
(altitude, slope, curvature) are themselves multi-scale fields driven
by the biome's height-generator stack and the features overlaid on
it.

Curvature, when needed by a palette function during bake, is computed
from the height cubemap by finite differencing on a smoothed input
(e.g., a coarser mip of the height cube). It is not stored as its own
channel for MVP. If sharp curvature gating later produces visible
speckling, fall back to baking a smoothed-curvature R8 channel.

Impostor contract (consumed directly by `planet_impostor.wgsl`):

- `albedo_cubemap` (`Rgba8UnormSrgb`): primary surface color, sampled
  bilinearly. No discrete material indirection — biome boundaries
  blend continuously.
- `height_cubemap` (`R16Unorm`): elevation used for per-fragment
  normals via `perturb_normal_from_height` (finite-difference at
  fragment time, full f32 precision) and the self-shadow ray march.
- `roughness_cubemap` (`R8Unorm`): per-texel microsurface response,
  wired into the Hapke BRDF's opposition surge width.
- `feature_buffers` + `feature_index`: SSBO craters and the spatial
  hash that walks them. Crater LOD fades sub-pixel features at far
  zoom.

Runtime upload policy: the flat impostor caps its uploaded cubemap
faces at `1024×1024`, even when `StaticSurfaceData` keeps a higher
source resolution for ground terrain and colliders. The ship-view
handoff to terrain is currently `4 × body.radius_m`; at that distance
the body is ~29° tall, or roughly 695 px on a 1080p viewport and 927
px on a 1440p viewport with Bevy's 45° vertical FOV. A 1024 face is
therefore the right current target for the orbital impostor. Move this
to 2048 only if 4K orbital inspection becomes a hard target before the
handoff distance changes.

Build-time provenance fields may feed the final albedo/height bake
without becoming renderer resources. For airless moons, `MareFlood`
writes `mare_coverage_cubemap`, a continuous 0..1 resurfacing mask:
shorelines, ghost-crater burial, and highland/mare palette blending
come from this fractional field. `material_cubemap` keeps only the
dominant compatibility ID.

Also baked into `StaticSurfaceData` but reserved for non-impostor consumers:

- `material_cubemap` (R8Uint dominant ID): used by the CPU
  `sample_static_surface()` path and by compatibility consumers that
  need one material ID per texel. It is not the color authority for
  fractional processes such as mare flooding, and is not bound to the
  impostor's GPU bind group.
- `normal_cubemap` (`Rgba8Unorm` object-space): reserved for ground
  LOD where chunked geometry can't cheaply finite-difference height
  at runtime. 8-bit encoding crushes shallow slope angles, so the
  impostor reconstructs normals from the height cube at fragment
  time instead.
- `materials` palette: still authored by stages for CPU sampling and
  for ground-LOD detail-texture blending; no longer uploaded as a GPU
  storage buffer.

Dynamic surface layers are deliberately not part of `StaticSurfaceData`:

- `DynamicSurfaceLayers::ice_caps`: authored/compiled ice veneer
  definitions. All production ice caps use this path; there is no
  permanent baked polar ice stage in the production compiler.
- `DynamicSurfaceLayers::active_dunes`: active, unconsolidated aeolian
  bedforms. Static terrain keeps basin structure, lithified/ancient
  geology, margins, scarps, and substrate; mobile sand sheets, dune
  relief, crests, ripples, tint, roughness, phase, amplitude, and
  mobility are dynamic layer data/state. The current impostor projection
  uses a prefiltered dynamic dune texture layer plus a transform uniform
  for slow migration, rather than evaluating dune-wave noise per fragment.
- `DynamicSurfaceState`: mutable runtime/editor state keyed by stable
  layer IDs. The default state reproduces authored appearance. Runtime game
  state owns this in `SolarSystemState`, not on renderer entities, so the
  impostor, ground terrain, collider source, editor overlays, and later
  weather/tide/wind systems read one shared per-body environment state.

The shared Rust sampling entry points are:

```rust
sample_static_surface(&StaticSurfaceData, dir, lod) -> SurfaceSample
sample_surface(&PlanetSurface, &DynamicSurfaceState, dir, lod) -> SurfaceSample
```

Dynamic layer contributions include height, normal, albedo, roughness,
and later optional material override. Dynamic height feeds the shared
sampled height, normals, water tests, self-shadow where supported, and
future terrain vertices. The current impostor silhouette is still the
mathematical sphere; dynamic displacement does not move the disk edge.

Silhouette displacement is intentionally out of scope for the first
projection. Good albedo, normals, roughness, and compact meso
features are enough for Mira and Vaelen to read from orbit.

### Cross-renderer feature projection contract

The impostor and the ground terrain renderer are two projections of
the same generated surface. Feature-specific shader logic is allowed,
but it must be a renderer projection of generator-owned feature data,
not an independent terrain system.

Every feature kind that needs visible detail below the cubemap scale
must declare a projection policy:

- **Baked fields:** what goes into the impostor cubemaps and ground
  tile attachments: height, albedo, roughness, normal, splat/material.
- **Analytic impostor data:** optional compact descriptors uploaded to
  the impostor shader when cubemap resolution is too coarse for the
  feature to read from orbit.
- **Ground tile data:** how `PipelineTileProvider` samples the same
  feature through `SurfaceField::sample()` into tile attachments.
- **Fragment detail:** optional shader-side detail for frequencies
  below tile resolution, driven by generator-authored descriptors or
  material parameters.
- **LOD ownership:** explicit bake/analytic/detail thresholds and
  overlap fades, in meters or screen-space pixels.
- **Mirror requirement:** if a profile is evaluated on both CPU and
  GPU, the descriptor fields, shape function, albedo/roughness response,
  and fade windows must match. The crater contract in
  `crates/terrain_gen/src/sample.rs` is the reference pattern.

WGSL modules may be feature-specific - for example crater, linear
feature, radial feature, or material-detail evaluators - but those
modules must consume projection data emitted by the compiler. They must
not infer geology from body name, local shader constants, or unrelated
noise fields.

Typical projection policies:

| Feature kind | Impostor projection | Ground projection |
|---|---|---|
| Craters | Large craters bake into cubemaps; mid-size craters use compact SSBO descriptors; sub-resolution craters use generator-authored statistical params or are omitted. | `SurfaceField` evaluates the same crater profiles into tiles; terrain fragments may reuse the crater/detail evaluator only for sub-tile normal or albedo detail. |
| Dune seas | Static substrate keeps basin/margin geology and any lithified ancient dune-derived terrain. Active sand sheets and dune relief are uploaded as a dynamic texture layer and transform-sampled in the canonical height/normal/material path; the `active_dunes` buffer remains layer metadata, not a per-fragment procedural evaluator. | Tile provider evaluates the same `ActiveDuneLayer + ActiveDuneState` over the static sample; fragment ripples come from material detail params when needed. |
| Seasonal ice caps | Not baked into height, albedo, roughness, or material cubemaps. The compiler carries cap definitions in `DynamicSurfaceLayers::ice_caps`; the impostor shader evaluates dynamic ice height, normals, albedo, and roughness from shared state. | Tile provider/runtime applies the same ice state layer over the static terrain sample; later climate simulation can vary coverage without rebuilding terrain. |
| Channels and dry riverbeds | Main incision, sediment color, and roughness bake into cubemaps; sharp banks or levees may use a generic linear-feature buffer. | Tile provider evaluates signed distance to the channel network, banks, terraces, bed material, and exposed strata from the same descriptors. |
| Rifts and grabens | Regional scarps and troughs usually bake into height/roughness; analytic linear buffers are reserved for sharp fault edges that need orbital crispness. | Tile provider evaluates the exact fault profile, talus, floor material, and secondary cracks. |
| Shield volcanoes | Shield shape, caldera, lava-flow color, and roughness bake into cubemaps; optional radial descriptors preserve caldera rims or large lobes. | Tile provider evaluates radial height profile, caldera depression, flank channels, lava-flow splats, and volcanic material detail. |
| Coastlines and oceans | Shoreline height, shelf color, wet/dry roughness, and beach or cliff masks are generated fields that bake into cubemaps. | Tile provider samples the same shore fields for shelves, beaches, cliffs, deltas, and wet material masks. Shader-side shoreline warp/jitter is migration debt unless represented as generator-authored detail parameters. |
| Layered substrate and material detail | Orbit sees the palette result baked into albedo/roughness, plus any distant asset or vegetation tint the terrain needs to read correctly. | Tiles and fragment shaders use splat/material attachments plus generator-authored triplanar/detail params so close-up material color agrees with the orbital view. |

The handoff invariant is simple: at any camera altitude where both
renderers can show a feature, they must show the same feature shape
with compatible height, normal, color, roughness, and material response.
Transitions should refine or fade detail in; they should never replace
one terrain interpretation with another.

### End-to-end runtime model

The source of truth is the authored planet spec plus the compiled
feature manifest, biome graph, static layer definitions, and dynamic
surface state. Everything dense or renderer-specific is a projection or
an acceleration artifact: impostor cubemaps, UDLOD tile attachments,
GPU atlas slots, collider patches, editor overlays, scatter cells, and
memory/disk caches must be reproducible from the same source state.
They must not become independent terrain authorities.

The long-term runtime shape is:

```text
PlanetTerrainSpec
  -> TerrainPrior
  -> FeatureManifest + BiomeGraph + DynamicSurfaceLayers
  -> SurfaceField::sample(dir, sample_scale_m, dynamic_state)
  -> projections:
       - orbital impostor bake
       - UDLOD tile provider attachments
       - physics height queries
       - rendered-height collider patches
       - editor provenance
       - scatter / vegetation instance fields
```

Keep ownership distinct:

- **Features** are semantic terrain/process structures with stable IDs
  and seed streams: craters, volcanoes, rifts, channels, basin fills,
  dune seas, and other geology that can affect height, material,
  roughness, provenance, and sometimes spawned child regions.
- **Biomes** are broad substrate/process fields: height-generator
  stacks, palette functions, feature budgets, and mask plans. They are
  not only Earth climate classes; mare basalt, highland regolith,
  evaporite basins, volcanic plains, and badlands are all biomes in
  this sense.
- **Dynamic layers** are mutable overlays whose default state reproduces
  authored appearance but whose runtime state belongs to
  `SolarSystemState`: seasonal ice, active dunes, later tracks, weather,
  tide/wind state, or other mutable surface processes.
- **Scatter populations** are object/detail instance fields driven by
  stable population IDs and placement rules: boulders, rock clusters,
  pebbles, grass blades, shrubs, and later trees. They may project into
  albedo/roughness/normal/height at some scales, but individual tiny
  scatter objects are not terrain-height authority by default.
- **Caches** are disposable acceleration: GPU atlas residency for
  currently drawable tiles, in-memory frecency caches for recent CPU
  tile payloads, and optional future disk caches for persistent reuse.

`SurfaceField::sample` must be scale-aware. A 2 km orbital sample should
not evaluate pebble- or grass-scale detail; a 0.5 m ground tile sample
may include those detail fields if their projection policy says they are
representable at that resolution. `sample_scale_m` is both a quality and
anti-aliasing contract: each layer or feature evaluates only the
frequencies visible at the requested scale and folds smaller frequencies
into palette/roughness/statistical detail or omits them entirely. This
keeps orbit bakes stable, tile generation bounded, and physics queries
from accidentally paying for visual-only microdetail.

For performance, compiled terrain should build runtime acceleration
structures from the semantic manifest before serving dense projections:
spatial hashes or spherical quadtrees for craters and scatter cells,
curve/BVH-style bounds for channels and rifts, compact runtime biome
mask plans, and per-feature projection descriptors. Tile generation must
query the tile footprint first and evaluate only features/layers that can
affect that footprint at the requested scale; it must not scan the whole
manifest per texel.

The intended work split is:

```text
Per frame:
  - update view-dependent TileTree state
  - draw resident UDLOD tiles from the GPU atlas
  - upload a bounded budget of completed tiles
  - update local collider/scatter cells only under explicit budgets

Async/background:
  - synthesize requested tile attachments
  - populate provider-level memory caches
  - prepare scatter-cell payloads

Offline/load/editor:
  - infer prior and compile the feature manifest
  - build runtime acceleration structures
  - bake orbital impostor cubemaps and compact analytic buffers
```

Tile cache keys must include enough source state to make invalidation
boring: body/source hash, generator version, dynamic layer epoch or hash
when dynamic data contributes to the payload, tile coordinate, and
attachment layout. Disk caching remains optional and measurement-driven;
provider-level memory caching is the default runtime optimization.

Dense scatter needs its own renderer and LOD policy. Grass blades,
pebbles, and small rocks should be generated from deterministic
surface-space cells keyed by body + population + cell, batched or
GPU-driven, distance culled, and faded into material/detail textures at
range. Only large scatter instances that gameplay can touch should
become ECS entities or colliders; visual-only micro scatter should not
inflate the terrain collider or per-frame ECS workload.

### Example bodies

#### Mira

Mira is the first proof target because it is feature-rich but
physically simple: an airless, tidally locked, silicate moon with a
visible near-side identity.

Expected root features:

- global crust
- primary near-side mare basin
- secondary near-side megabasin
- irregular near-side mare province
- far-side highlands
- crater population
- mare flooding
- regolith garden
- space weathering

The key authoring loop is seed-local rerolling: keep a good basin,
reroll its secondary craters, lock its mare fill, or promote a fresh
ray crater.

Mare flooding is a fractional resurfacing/provenance pass, not a hard
material classifier: it writes continuous coverage, blends fill height
by that coverage, and leaves the dominant material ID as a compatibility
summary. Old craters inside maria should be partly filled, their rims
subdued, and their albedo overprint suppressed so they read as
mare-colored ghost craters instead of bright highland rings.

#### Vaelen

Vaelen is the second proof target because it exercises history: thin
atmosphere, ancient wet past, sedimentary basins, evaporites, buried
ice, aeolian modification, and moderate crater preservation.

Expected biomes (each carrying a palette function of physical
fields):

- dark volcanic / impact-melt lowlands (`FloodBasin` generator)
- pale sediment / evaporite lowlands (`ErodedPlain`)
- rust-highland dust mantle (`BroadSwell` plus dust modifier)
- etched plateaus and mesas (`EtchedPlateau`)
- one or two dune seas (`OrientedDune`)
- shield-volcano flanks (`RadialFlow`, biome spawned by the volcano
  feature)

Expected features:

- crater population (degraded, density modulated per biome)
- a few channel/canyon systems sourced in pale lowlands
- one Valles-class rift
- one Olympus-class shield volcano (authored placement)

The key test is era overprinting. The same feature graph must
express an original basin, its degraded rim, later sediment fill,
evaporite floor, and present dust or aeolian erosion — without the
biome boundaries reading as flat color zones from orbit. See
`gen/vaelen_processes.md` for the per-body process notes.

### Asset schema

Bodies have a normalized terrain route:

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

### Migration plan

1. Add feature compiler data types and deterministic feature seeding.
2. Infer `TerrainPrior` from `PlanetTerrainSpec`.
3. Generate initial feature manifests for Mira and Vaelen.
4. Compile those manifests into `PlanetSurface`: cached
   `StaticSurfaceData` substrate plus first-class dynamic layer
   definitions/state.
5. Switch body definitions from `pipeline: [...]` to feature specs
   one body at a time.
6. Retire old stages after all bodies are compiled through the
   feature graph.

During migration, `PlanetSurface` is the renderer handoff. Static
textures and analytic feature buffers remain stable, while dynamic layer
buffers can be rebuilt independently from `DynamicSurfaceState`.

### Current implementation status

`AirlessImpactMoon` and `ColdDesertFormerlyWet` are wired up: a
`PlanetTerrainSpec` is expanded into a `FeatureManifest`, then
compiled into `PlanetSurface` using the current bake primitives for
static substrate and dynamic layer definitions for changeable surface
state. This keeps the flat impostor renderer working while moving
source-of-truth terrain identity and seeding into the feature compiler.
Configured polar caps and active unconsolidated dunes are dynamic
surface layers rather than projected into static cache ownership.
`AgingOceanicHomeworld` and `GenericTerrestrial` are not yet implemented
— Thalos and Pelagos render via the flat-water `TerrainConfig::Ocean`
placeholder until the terrestrial pipeline lands (M2).

### M2 — Terrain pipeline revamp

M2 is a general overhaul of the feature compiler, not just a "two new
archetypes" pass. The four main bodies — Mira, Vaelen, Thalos,
Pelagos — should all come out the other end rendering through one
revamped pipeline at the right visual quality.

Two streams of work, landed together:

**v2 backlog.** Items from the terrestrial-pipeline research that
make a v1 pipeline look right rather than just look complete. See
the *v2 backlog* section below for the full list. Spinning these up
alongside the new archetypes is cheaper than retrofitting them
later — they change the shape of the manifest and the BiomeMaskPlan
DSL.

**Missing archetypes.** Replacing the flat-water `Ocean` placeholder
requires:

- **`AgingOceanicHomeworld`** for Thalos. Deeply ancient, oceanic,
  iron-rich, geologically declining. Relevant prior outputs:
  hemispheric land/water asymmetry, buried legacy tectonic structure,
  dust mantles in dry interiors, ice-stability bands. Stagnant-lid
  plate boundaries are control scaffolding only: they can influence
  broad priors or debug overlays, but raw plate polygons must not be
  readable in final height, albedo, roughness, or normal cubemaps.
- **`GenericTerrestrial`** for Pelagos and other thick-atmosphere
  oceanic moons. Volcanic island arcs from tidal heating,
  hydrothermal circulation, photosynthetic shelf ecosystems. Read
  from orbit as a softer, milkier ocean world (atmospheric optics
  handled in [atmosphere.md](atmosphere.md)).

Existing archetypes (`AirlessImpactMoon` for Mira,
`ColdDesertFormerlyWet` for Vaelen) get re-evaluated against the
revamped pipeline; the v2 backlog items affect them too (e.g. Vaelen
gets first-class hydrology for its ancient-wet story, layered
substrate for evaporite floors and dust mantles).

Other Pyros bodies (Auron's moon system, the asteroid belt, outer
gas giants and ice moons) follow incrementally — the revamp produces
a pipeline they slot into, not a milestone gate.

---

## Rendering: ground LOD (M3)

Thalos uses the in-tree `thalos_udlod` fork for surface-scale terrain
rendering. The fork began as Kurt Kühnert's `bevy_terrain`, which was
shaped around rendering finite preprocessed raster datasets. That was
useful as a starting point, but it is the wrong primary model for
Thalos: our terrain tiles are synthesized at runtime from body data,
may later be cached opportunistically, and should eventually be
generated directly on the GPU.

The fork is therefore **runtime-provider-first**. `TileAtlas` and
`TileTree` own residency, fallback, LOD balancing, and shader-visible
atlas state; `TileProvider` owns the source of tile contents. The
offline GeoTIFF/preprocess/`DiskTileProvider` path has been removed.
Persistent reuse should be implemented as a cache provider/wrapper
around the runtime pipeline, keyed by body config + tile coordinate +
attachment spec, not as an authored `assets/<terrain>/data` tree.

**Fork status:** in-tree and divergent by design. The Bevy 0.18 port,
unconditional `big_space` precision path, CPU-balanced draw tile
selection, and runtime `TileProvider` path are already in place. M3 is
no longer about proving an upstream-like preprocessed example; it is
about making the Thalos tile pipeline agree with the impostor, collider,
and gameplay height sources, then moving expensive tile production to
GPU jobs.

### Repository landscape

- **`thalos_udlod`** — the library. Forked in-tree and edited as part
  of the Thalos workspace.
  Contains the entire terrain rendering stack including UDLOD,
  Chunked Clipmap, three terrain models (planar / spherical /
  ellipsoidal), cubesphere projection, Taylor-series GPU precision
  approximation, unconditional `big_space` integration, multi-view
  rendering, debug tooling, and the runtime tile provider seam.
- **`planetary_terrain_renderer`** — Master's thesis demo app. Uses
  `thalos_udlod` to render real-world Earth from GeoTIFF datasets.
  **Reference only.** Its `examples/spherical.rs` and debug controls
  are useful templates for our integration; we do not depend on or
  fork the app itself.
- **`terrain_renderer`**, **`dtm`** / **`bevy_dtm`** — older or
  format-specific. Not relevant for procedural use.

The Master's thesis novelties (ellipsoid model, Taylor-series GPU
precision) were merged into `thalos_udlod` itself during Kurt's two
years of professional work at Argeo (Oct 2023 – Jul 2025), where he
was paid to build production geospatial visualization on top of the
library. This means the thesis tech is in well-tested production
code.

### big_space role for terrain

`thalos_udlod` already handles surface-scale precision via
Taylor-series approximation of ellipsoid coordinates relative to the
viewer. Within the surface of a single planet, the renderer maintains
GPU precision down to centimeter scale without needing big_space
cells.

big_space's role for terrain is therefore the same as its role
elsewhere in the sim (see [simulation.md](simulation.md), "big_space
usage"):

- **System frame** — barycentric, where bodies orbit.
- **Per-body local frames** — child grids that move with each body's
  orbital and rotational state. Surface-attached entities (terrain,
  ships on the ground, structures) parent to the body's grid and
  inherit its motion in high precision automatically.
- **Camera floating origin** — keeps f32 transforms accurate near the
  camera regardless of where it is in the system.

big_space does *not* need to provide nested cells for the surface of
a single body. The terrain renderer handles that internally.

#### Configuration for Pyros

- **Grid precision:** `i64`. Effectively unlimited range for a solar
  system; cheap.
- **Cell size:** ~1 km in the system frame. Sub-micrometer cell-local
  f32 precision near the camera, comfortably more than adequate for
  ship physics, surface objects, and anything else outside terrain.
- **Hierarchy:**
  - Root `BigSpace`: system inertial frame, origin at Pyros
    barycenter.
  - One `Grid` per orbiting body. Position updated each frame from
    patched-conics integrator. Rotation from spin state.
  - Terrain entity (the `thalos_udlod` setup for that body) is
    parented to the body's grid. It inherits orbital and rotational
    motion automatically.
  - `thalos_udlod` wires the floating origin into the Taylor
    approximation unconditionally — the renderer knows where the camera
    is in big_space coords and computes its Taylor coefficients
    accordingly. The upstream `high_precision` Cargo feature that gated
    this path has been removed; it's the only viable precision path at
    planet scale.

#### Caveats

- Per `big_space` docs: prefer applying *deltas* to entity transforms
  over absolute positions. Setting absolute positions causes the
  floating-origin system to constantly re-center, fighting
  controllers. For deterministic motion (orbits), compute the
  position in high precision and write directly to
  `(CellCoord, Transform)` via `Grid::translation_to_grid`.

### What's in the fork

#### Inherited from upstream

- **UDLOD triangulation.** GPU-driven, screen-space-error-based
  bintree subdivision, vertex-shader morphing for seamless LOD
  transitions.
- **Three terrain models, including `ELLIPSOIDAL`.** Per-body
  semi-major / semi-minor axes; min/max height range.
- **Cubesphere projection** with C_SQR distortion correction.
  Standard, mature.
- **`TerrainModelApproximation`** — second-order Taylor series
  coefficients computed CPU-side per frame and uploaded for use in
  GPU shaders to position vertices on the ellipsoid in
  viewer-relative coords with high precision. The master-thesis
  novelty that lets us avoid f64 on the GPU.
- **`TileAtlas` + `TileTree`.** GPU-side tile storage and
  hierarchical sampling with LOD blending.
- **Attachment system.** Multi-channel tile data with configurable
  resolution and format per channel: R16 / packed RG16 / R32Float
    (height), RG16 (normals), RGBA8 (albedo / splat / custom). 1px
    borders for seamless filtering. The game uses packed RG16 height
    (coarse + residual) to avoid visible contouring on very low-slope
    terrain without requiring filterable float textures.
- **`big_space` integration** (unconditional; the upstream
  `high_precision` Cargo feature was removed). Wired to the camera and
  floating origin so the Taylor-series relative-position path is always
  active.
- **Multi-view rendering, custom material plugin system, debug
  visualization tools.**

#### Added/changed by the fork

- **`TileProvider` trait.** The architectural seam. See
  *TileProvider interface* below.
- **Runtime-provider-first tile loading.** Decouples "where data comes
  from" from "how it's stored / sampled / rendered." The original
  disk-loading/preprocess code was removed because it assumed a finite
  pre-authored dataset under `assets/`; Thalos needs arbitrary
  generated or cached tiles.
- **CPU-balanced draw tile selection.** The old GPU refine pass made
  per-tile decisions and could emit LOD gaps across cube-face seams.
  The current CPU draw-set pass enforces the 2:1 balance invariant
  before uploading the tile list to the same buffer the vertex shader
  consumes.
- **Bevy 0.18 port.** Upstream's last main indexing was May 2025,
  likely on Bevy 0.16.

#### Intended next divergence

- **GPU tile production.** Keep CPU residency and draw balancing until
  a GPU global-balance path is justified, but let providers enqueue GPU
  jobs that write directly into atlas slots. The first GPU producer can
  mirror the CPU `PipelineTileProvider`; later producers can run
  diffusion or other heavier synthesis before marking the slot ready.
- **Thalos cache provider.** If tile latency or repeated visits demand
  persistence, add a cache wrapper around runtime providers. It should
  store generated tile payloads by body/source hash and coordinate, not
  restore the old preprocessed dataset model.

### TileProvider interface

This is the contract between rendering and synthesis.

#### Contract

The `TileProvider` is a trait the renderer calls to obtain tile
data. The renderer holds a `TileTree` per (terrain, view) pair and
decides which tiles need to be resident based on camera position and
LOD. When a tile becomes needed, the renderer asks the provider for
its data; the provider returns texture data for each configured
attachment.

```rust
// Approximate shape — exact API to be designed during M3.

pub trait TileProvider: Send + Sync {
    /// Request tile data for the given coordinate. May be async; the
    /// renderer will sample the parent LOD until the tile is ready.
    fn request_tile(
        &self,
        coord: TileCoordinate,
        model: &TerrainModel,
        attachments: &[AttachmentConfig],
    ) -> Task<Result<Vec<AttachmentData>>>;
}
```

The trait should be designed to allow async/eventual delivery; tile
latency is not zero and the renderer must tolerate it gracefully (it
already does, via parent-LOD fallback).

The current trait returns CPU attachment buffers because that is the
implemented path. CPU payloads can already be wrapped in
`MemoryTileCacheProvider`, which keeps a bounded in-memory frecency
cache outside `TileAtlas`. Use the wrapper with a namespace derived
from body/source hash; the key also includes tile coordinate and
attachment layout. This is deliberately a provider concern so atlas
residency stays focused on currently visible slots.

The next shape should separate atlas-slot residency from production
backend:

```rust
enum TileProduction {
    Cpu(Task<Result<Vec<AttachmentData>>>),
    Gpu(GpuTileJob),
}
```

`GpuTileJob` should carry tile coordinate, atlas slot/generation,
attachment layout, and body/pipeline uniforms. Its compute pass writes
the target atlas array layer directly, then reports the slot ready. The
atlas should only care that the coordinate's current slot generation
finished; it should not care whether bytes came from CPU, cache, or GPU.

#### What the provider must produce per tile

For each attachment configured on the terrain, an N×N texture in the
attachment's format. Defaults from upstream:

| Attachment | Format | Default size | Purpose |
|---|---|---|---|
| `height` | packed RG16 in game (`R16` and `R32Float` still supported) | 512×512 | Required. Stored as normalized 0..1, mapping to `[min_height, max_height]` configured per body. Packed RG16 stores a residual in the second channel to avoid visible R16 contouring on broad shallow slopes without requesting filterable float textures. |
| `normal` | RG16 | 512×512 | Optional. May be derived from height if not provided. |
| `albedo` | RGBA8 | configurable | Optional surface color. |
| `splat` | RGBA8 | configurable | Optional material weight masks for shader-side blending. |
| custom | configurable | configurable | Anything else the shader needs. |

Sizes are powers of two (upstream limitation: mipmap generation only
supports POT).

#### Border requirement

Each tile texture has a 1-pixel border that overlaps neighbors.
Border values must match neighbors *exactly* (same float bits) so
texture filtering across tile boundaries is seamless. This is
automatic when the synthesis pipeline is a pure function of position.

#### Determinism requirement

`request_tile(coord)` and `request_tile(child_of_coord)` must agree
exactly on shared edges, regardless of evaluation order or other
tiles requested. This is satisfied iff the synthesis pipeline is a
pure function of (ellipsoid position, body parameters). The pipeline
is feature-based and queryable at arbitrary resolution, which gives
this for free as long as features are evaluated by position.

#### Coordinate input to the provider

The provider receives `TileCoordinate` (face id, lod, x, y on the
cubesphere). It is responsible for converting to ellipsoid position
when sampling its synthesis pipeline. The provider should evaluate
in 3D ellipsoid position rather than face-UV or lat/lon, both to
avoid pole singularities and to make the determinism property easier
to maintain.

The conversion math (cubesphere face/UV → ellipsoid position) is
already implemented in upstream `thalos_udlod::math` and can be
called from the provider.

#### Latency

Tile requests are on the critical path of LOD streaming. The
renderer falls back to parent LOD while a child is in flight, but
persistent multi-second latency means the player sees low-detail
terrain at close range.

Rough budget: tiles available within a few hundred ms of being
requested at typical traversal speeds. Hard real-time is not
required. If the synthesis pipeline can't meet this, the provider
can layer a Thalos cache on top. That cache is a runtime
optimization, not an authored terrain source.

### Per-body configuration

Each body in the Pyros System needs a terrain configuration alongside
its physical parameters in the existing RON solar system spec.

#### Schema additions

```ron
// Conceptual; actual RON syntax to match existing spec style
Body(
    name: "Thalos",
    // ... existing physical params: mass, orbit, spin, etc.

    shape: Ellipsoid(
        semi_major_m: 6_378_137.0,
        semi_minor_m: 6_356_752.0,
    ),

    terrain: TerrainConfig(
        min_height_m: -11_000.0,
        max_height_m:   9_000.0,
        attachments: [
            Attachment(name: "height",  format: R16,    size: 512),
            Attachment(name: "normal",  format: RG16,   size: 512),
            Attachment(name: "splat",   format: RGBA8,  size: 256),
        ],
        synthesis_seed: "thalos-v1",
        // any other params the synthesis pipeline needs
    ),
)
```

#### Computing oblateness

For physically-modeled fluid bodies (gas giants, fast-rotating
rocky), oblateness can be derived from rotation rate:

`f ≈ ω² R³ / GM` (fluid limit)

Gas giants in Pyros should use this — they should look visibly
squashed. Rocky bodies use a smaller value or are authored directly.

#### Bodies without terrain

Small bodies (asteroids, irregular moons) may skip this stack — at
small scales a chunked mesh or single LOD-instanced model is more
appropriate. The terrain stack is for bodies large enough that
ellipsoid + LOD'd surface is the right primitive.

### Implementation stages (M3)

The fork itself is done. The remaining stages are about pulling it
into Thalos and wiring it to the revamped feature compiler.

#### Stage 1: pull the fork into the workspace

- Add the `thalos_udlod` fork as an in-tree workspace crate.
- Stand up a `thalos_terrain_render` crate that owns the integration: registers
  `thalos_udlod::TerrainPlugin`, exposes a deterministic
  `SyntheticTileProvider` (pure function of `Coordinate::world_position`
  so tile borders are bit-identical), and ships a `playground` example
  binary that drives the fork end-to-end against the synthetic provider
  on a Mira-scale sphere.
- **Exit criterion:** `cargo run -p thalos_terrain_render --example playground`
  renders a UDLOD sphere using `SyntheticTileProvider`, validating that
  the fork compiles and runs inside the Thalos workspace.

  Why this differs from earlier drafts of the spec: the original Stage 1
  exit criterion was "fly around the fork's `DiskTileProvider` spherical
  example in our project." That was written before the `TileProvider`
  trait existed and assumed validating the disk path was the cheapest
  proof. Now that the trait is in place, exercising the same seam Stage 2
  will reuse with a synthetic provider is a stronger Stage 1 deliverable
  and skips authoring preprocessed disk assets we'd never ship. The old
  upstream-compatible `DiskTileProvider` path has since been removed;
  the synthetic playground is the fork sanity check.

  The big_space hierarchy validation moves to Stage 2 — the playground
  spawns its own `BigSpace` root rather than threading through the
  game's `RealSpaceRoot` + per-body `Grid` tree. Parenting the terrain
  to a body's `Grid` so it inherits orbital motion is part of the
  `PipelineTileProvider` wiring, not the fork-compile check.

#### Stage 2: implement `PipelineTileProvider`

- Implement `PipelineTileProvider` wrapping the synthesis pipeline.
  Converts cubesphere `TileCoordinate` → body-local direction → reads
  a temporary low-pass sample of the baked height/albedo/roughness
  cubemaps from `StaticSurfaceData` → copies the result into the
  configured tile attachments (height into packed `Rg16` in the game path,
    with `R16` / `R32Float` still supported by the provider; albedo into
  sRGB-encoded `Rgba8`, roughness into linear `R16` upscaled from the
  source u8 cubemap by 257). This is deliberately a short-term visual
  bridge for the current Thalos terracing; the terrain rewrite should
  replace it with genuinely continuous local fields.
- The current height path is deliberately the rendered Query API/atlas
  source, not an independent `thalos_terrain::sample_static_surface()`
  detail path. The full
  sampler includes SSBO crater iteration and statistical detail that
  UDLOD does not render yet; using it for tiles or colliders would make
  physics disagree with the visible surface.
- Border determinism is automatic: directions come from
  `TileCoordinate::stitched_pixel_coordinate` → `Coordinate::world_position`,
  the same mapping the renderer samples with after applying the same
  neighbour-border orientation as UDLOD's offline `stitch.wgsl` pass.
- Diagnostic override: run `THALOS_TERRAIN_PROVIDER=analytic3d just game`
  to replace the game ground-LOD tile data with the face-independent
  `SyntheticTileProvider` analytic 3D field. This is a visual seam test:
  it bypasses baked cubemap sampling and dynamic layers, so CPU height
  queries and terrain colliders still follow the normal rendered-height
  path rather than the analytic surface. The local craft-shadow proxy is
  on by default so nearby ships/EVA cast a stable sun-ray shadow onto the
  custom UDLOD terrain receiver. Ships project per-part frustum/cylinder
  silhouettes from the same procedural part dimensions as the visible mesh;
  EVA keeps a small capsule proxy. Leave `THALOS_TERRAIN_CRAFT_SHADOW`
  unset or set it to `on|auto` for the normal behavior, and set it to
  `off` only when isolating material seams. If the real body's height range
  makes the analytic field look flat, set
  `THALOS_TERRAIN_ANALYTIC_RANGE_M=500` (or another positive metre value)
  to widen the visual-only diagnostic height range.
- Fully flat diagnostic: run `THALOS_TERRAIN_PROVIDER=flat just game` to
  force constant-height terrain vertices and constant albedo/roughness. If
  a hole remains here, the defect is in UDLOD geometry selection, strip
  assembly, culling, or transform precision rather than terrain data.
  Gameplay height queries also use a zero-height source in this mode, so
  EVA spawn, walking, terrain-collider patches, and trajectory collision
  all agree with the rendered reference sphere from the first frame.
- The provider holds the `PlanetSurface` behind an `Arc` so it shares
  data with the impostor billboard's `PlanetMaterial` and there is one
  copy of the cubemap-heavy `StaticSurfaceData` per body.
- In-memory tile reuse is provided by `thalos_udlod`'s `TileAtlas`;
  an explicit Thalos-side persistent cache is deferred unless tile
  latency proves to be a problem.
- **Implementation:** [crates/terrain_render/src/pipeline.rs](../crates/terrain_render/src/pipeline.rs).
- **Exit criterion (met):** body terrain entities are spawned by
  `crates/game/src/rendering/ground_terrain.rs` from
  `finalize_planet_generation` once the body's `PlanetSurface` task
  resolves; the synthesized cubemap drives both impostor + ground LOD
  from the same source data.

#### Stage 3: shared Hapke shading

- `BodyTerrainMaterial` binds `AtmosphereBlock` (static, per-body) and
  `SceneLighting` (per-frame, primary star + eclipse occluders +
  ambient + planetshine parent). The per-frame writer
  (`update_body_terrain_atmosphere`) populates `SceneLighting` from
  the same ephemeris snapshot the impostor's per-frame writer uses,
  so primary-star direction and flux match across the LOD swap.
- `body_terrain.wgsl` samples height + albedo + roughness from the
  thalos_udlod attachment atlases, derives a height finite-difference
  normal via `sample_normal`, then heavily damps it toward the geometric
  sphere normal before calling `thalos::lighting::shade_hapke_surface`.
  This is a temporary visual-smoothing choice for the pre-rewrite
  terrestrial terrain: the current Thalos macro field has broad height
  bands, and full-strength height normals make those bands read as dark
  contour steps. The terrain vertex stage also passes a camera-relative
  view vector into the fragment shader so low-altitude view terms do not
  subtract large absolute render-space positions, and the shared Hapke
  helper keeps surface visibility anchored to the geometric normal when
  a height normal points away from the camera at grazing angles. On
  atmospheric bodies, the terrain shader also derives a small
  sky-diffuse fill term from `AtmosphereBlock`'s Rayleigh/Mie column so
  nearby slopes under a daylight sky do not collapse to pure black when
  they are not directly sun-facing; airless bodies keep the vacuum-black
  floor. The local craft proxy still feeds `external_shadow =
  local_craft_shadow` from a sun-ray capsule test against the player's
  craft, while crater shadow / terrain self-shadow remain deferred.
- The impostor (`planet_impostor.wgsl`) calls the same
  `shade_hapke_surface` with
  `external_shadow = crater_shadow * self_shadow_term`. Atmosphere
  transmittance, cloud composite, water BRDF, and limb darkening are
  applied post-call on the impostor side.
- The terrain LOD path keeps camera-path atmosphere and clouds out of
  `body_terrain.wgsl`; `BodySkyMaterial` draws the in-front layer as a
  fullscreen pass while terrain is visible, clipping the raymarch at
  copied scene depth and sampling the same reference cloud cubemap on a
  fixed-altitude shell. Terrain-side cloud shadows and water BRDF remain
  deferred, but orbital/mid-altitude haze and cloud coverage match the
  impostor handoff.
- **Implementation:**
  [crates/planet_lighting/src/shaders/lighting.wgsl](../crates/planet_lighting/src/shaders/lighting.wgsl),
  [crates/terrain_render/src/body_terrain.wgsl](../crates/terrain_render/src/body_terrain.wgsl),
  [crates/game/src/rendering/ground_terrain.rs](../crates/game/src/rendering/ground_terrain.rs).
- **Exit criterion (met):** terrain ground-LOD pixels go through
  Hapke + eclipse + planetshine + ambient via the shared helper, with
  per-fragment roughness sampled from the third tile attachment.

### M5 rendered-height terrain colliders

The first landing slice exposes rendered-height helpers from
`thalos_terrain_render`:

```rust
rendered_height_m(surface: &StaticSurfaceData, dir: Vec3) -> f32
build_rendered_terrain_patch(surface, body_radius_m, center_dir, basis, config)
    -> TerrainPatchMesh
```

These helpers decode the same R16 cubemap interpretation used by
`PipelineTileProvider`: `real_meters = (texel / 65535 * 2 - 1) *
height_range`. Local physics builds one tangent-plane patch around the
active craft from this data and converts the mesh into an Avian static
trimesh. Current defaults are 4096 m half extent, 129 x 129 vertices,
and rebuild after the craft moves more than 1024 m laterally from the
patch center.

This is a fidelity choice: collision matches the visible UDLOD
surface. When the HMF detail cascade (below) is engaged, both
`PipelineTileProvider` and the collider source route the same
`rendered_height_m` call so they stay in lockstep.

### Procedural-detail cascade (HMF + domain warp)

UDLOD tile data is the macro cubemap plus a runtime detail cascade
evaluated by `PipelineTileProvider` per tile pixel.

The macro is the baked cubemap: at Thalos's 4096² resolution and
3186 km radius, ~1.2 km per equator texel. Until the revamped terrain
generator provides better continuous local fields, `PipelineTileProvider`
low-pass filters the macro height/color/roughness source before UDLOD
tiles and rendered-height queries see it. The ground terrain material
also blends height-derived normals strongly back toward the geometric
normal so residual bands read as smooth terrain rather than terraced
contour steps.

The cascade — since migration P0, it lives in the Query API seam
[crates/terrain/src/query.rs](../crates/terrain/src/query.rs) (moved out
of `terrain_render::pipeline`, which now delegates to it) — adds
high-frequency detail on top of the macro:

- **Musgrave ridged hybrid multifractal** (`hmf_ridged_3d` in
  [crates/terrain/src/noise.rs](../crates/terrain/src/noise.rs))
  evaluated in body-local 3D so the field is sphere-continuous —
  the same physical point returns the same value regardless of which
  cube face is generating it. The HMF's self-modulating weight
  (`weight *= signal` each octave) produces "rough peaks, smooth
  valleys" without any external biome mask, and the ridged shape
  concentrates signal at noise zero-crossings → ridge crests rather
  than dome tops.
- **Domain warping** via `fbm3_vec3` offsets the position before HMF
  sampling. Breaks the lattice-aligned look of plain ridged noise.
- **Continuous octave count.** `detail_plan_for_lod` returns a
  fractional octave count from the tile's Nyquist resolution; HMF
  weights its top octave by the fractional part so tiles cascading
  N → N+1 across an LOD boundary blend smoothly. At the deepest LOD,
  11 octaves from a 1 km base wavelength bottom out at ~0.49 m, with
  ~12 cm amplitude at the deepest octave (decimetre-scale displacement
  at sub-metre wavelength).
- **Positive-only contribution.** HMF is normalised to `[0, 1]` and
  scaled by `DETAIL_AMP_M`; the macro acts as the sediment / tectonic
  floor, with HMF orogeny accumulating in rough regions. The R16
  encoding budgets `DETAIL_HEIGHT_MARGIN_M` (currently 250 m) above
  the static + dynamic envelope.

Known limitation: **CPU/GPU bilinear stand-off.** Tile R16 data is
bilinearly sampled by the GPU; `rendered_height_m` evaluates the
cascade pointwise at the requested `dir`. Off pixel centres the two
values disagree by up to one peak-to-trough of the resolved detail —
O(10–20 cm) at sub-metre wavelength. Acceptable for v1; fixing it
means threading `TerrainModel` into the height query and bilinear-
mixing four texel-centre evaluations using UDLOD's stretched-cube
projection (`Coordinate::world_position`). Tracked separately.

v2 candidates the cascade does not address today:

- Per-region character (biome-driven blend of two or more HMF
  profiles). Single profile applied uniformly means roughness varies
  with altitude (HMF's natural behavior) but every region has the
  same character. Hooking into the macro `biome_weights_cubemap` is a
  drop-in change to `compute_detail_height`.
- Anisotropic ridges aligned to tectonic stress vectors. Requires
  threading `TectonicSystem` directions into the per-pixel evaluator.
- Drainage networks. Erosion adds geological character at high CPU
  cost and is sphere-discontinuous in 2D; a 3D-continuous variant is
  available in `bevy_erosion_filter` if drainage detail proves
  necessary.

#### Stage 3: onboard Mira, Vaelen, Thalos, Pelagos

- Per-body terrain configs are derived from the existing body data:
  `body.radius_m` drives the ellipsoid model, `static_surface.height_range`
  drives min/max height, and the synthesis archetype drives the surface
  itself. No additional fields in the RON spec — that lets the same
  configuration cascade from impostor → ground LOD without diverging.
  If we later need divergent params (e.g., different LOD count per
  body) those land as additions on `FeatureTerrainConfig`.
- Spawning is unconditional on the procedural branch of
  `finalize_planet_generation`. Mira (`AirlessImpactMoon`), Vaelen
  (`ColdDesertFormerlyWet`), and Thalos (`AgingOceanicHomeworld`) all
  feed real synthesis through to the tile provider. Pelagos still
  routes through the `TerrainConfig::Ocean` placeholder; until the
  `GenericTerrestrial` archetype lands (M2 v2 backlog) it renders the
  flat-water static surface — the wiring is correct, the input just
  doesn't carry interesting features yet.
- The terrain entity is parented to the body's real-space grid (1 km
  cells) and inherits orbital + rotational motion automatically.
  Tile-tree association is per-`(terrain, ship_camera)`; map view and
  photo mode views do not load body tiles.
- **Impostor ↔ terrain handoff.** Each procedural body has *one* visible
  representation at a time on the ship layer. A
  `sync_terrain_impostor_swap` system flips visibility hard at a
  distance threshold of `4 × body_radius` from the body centre: closer
  than that, the UDLOD terrain is visible and the ship-layer impostor
  is hidden; farther, the reverse. The map-layer impostor renders at
  every distance and is unaffected.
  Smooth opacity crossfade requires `PlanetMaterial`/`BodyTerrainMaterial`
  opacity uniforms + matching shader work and lands alongside terrain
  PBR + atmospheric optics in M4.
- **Exit criterion (met):** every procedural body renders ground LOD
  alongside its impostor (one visible at a time per camera distance);
  atmospheric optics + terrain-as-caster cascaded shadowing land in M4.

Other Pyros bodies (Auron's moon system, Ceryx, outer-system worlds)
follow incrementally — same pipeline, no separate stage.

### Fork relationship and upstream

The fork is now a long-lived Thalos subsystem, not a thin upstream
overlay. Keep the attribution and license lineage clear, but optimize
the code for Thalos's runtime and GPU-generation path.

Useful upstream rendering fixes can still be cherry-picked or ported,
especially around UDLOD math, mip handling, and Bevy version bumps.
Do not preserve the old GeoTIFF/preprocess architecture just to keep a
small diff; that model is no longer part of Thalos's terrain renderer.

---

## v2 backlog

Strongest items from
[gen/terrestrial_pipeline_research.md](gen/terrestrial_pipeline_research.md)
to land alongside the M2 terrestrial pipeline. Not a separate
milestone — these are the recipes that make `AgingOceanicHomeworld`
and `GenericTerrestrial` look right.

1. **First-class hydrology features.** Lift the manifest from
   "channel" generators to a proper hydrology subgraph: drainage
   tree, watershed polygons, paleo-channels, lake basins. Run a
   Cordonnier-2016 stream-power solve on a coarse Delaunay over the
   cubesphere at the era-3/4 boundary; produces dendritic ridges and
   valleys that pure noise cannot.
2. **Layered material column under each `BiomeId`.** A small (2-4
   layer) stratigraphic stack with thicknesses and material refs.
   Erosion/aeolian/glacial overprints in later eras *expose*
   subsurface materials rather than recoloring. Borrowed from
   Cordonnier 2017 / Arches 2009.
3. **Climate-field inputs to `BiomeMaskPlan`.** Named fields computed
   once during prior inference: `stellar_flux_field`,
   `obliquity_seasonal_extreme`, `precipitation_proxy`,
   `subsurface_ice_stability`. Generalizes biomes to all archetypes
   (including tidally locked exoplanets) without special cases.
4. **Provenance API + seed-promotion state machine.** Sample of
   `SurfaceField` returns top contributing FeatureInstance IDs and
   weights. Explicit lifecycle: `procedural → liked → locked →
   promoted-to-authored`.
5. **Werner-1995 dune CA**, **Hartmann-Neukum crater
   chronology with overprint rule**, **DLA + Gaussian pyramid for
   ridges**. Concrete `HeightGenerator` peer variants implementable
   in days each.
6. **Inverse procedural sketches** (Schott 2023). Hand-painted
   regions enter the manifest as constraint primitives; the rest of
   the tree re-solves against them.
7. **Fusion archetypes as parameter-space points.** No special
   branches for "tidally locked" or "active hydrocarbon cycle"; a
   single archetype-bias function over the parameter cube.
8. **Projection policy per feature.** Codify `bake_lod_cutoff` plus
   analytic-buffer and fragment-detail thresholds for each feature
   kind. Craters are the reference pattern: impostor, CPU sampler, and
   ground LOD must agree on descriptor layout, profile semantics, and
   fade windows.

---

## Open questions

Carried from both source docs; remain undecided.

1. **Async tile request shape.** Bevy task pool? Custom worker pool?
   GPU compute path for some attachments? Profile-driven.
2. **Cache strategy.** In-memory only is simplest; disk cache becomes
   worth it if synthesis latency is high or if we want consistent
   reload performance. Defer until measured.
3. **CPU vs GPU synthesis split.** This affects the provider
   interface (does it hand back CPU buffers or GPU textures already
   on the device?). Worth resolving before M3 stage 3.
4. **CPU/GPU evaluator mirroring.** For analytic features used by
   both impostor and terrain fragment shaders, do we maintain
   handwritten Rust/WGSL mirrors with tests, generate WGSL from shared
   profile definitions, or restrict GPU evaluators to simple descriptor
   families?
5. **Surface-scale detail beyond tile resolution.** At highest LOD,
   even cm-scale features outrun synthesized tile resolution. Likely
   solved via shader-side detail textures and triplanar mapping,
   parameterized by splat masks.
6. **Body-to-body LOD handoff.** Camera transitioning from
   interplanetary to surface scale of a target body — what's the
   seamless handoff? Probably a separate "approach" mode that
   progressively refines the target body's tile residency.
7. **Shadow casting from terrain.** Ground LOD can receive local craft
   shadows from an analytic craft proxy, but terrain itself still does
   not render into shadow maps. Cascaded shadow maps with
   terrain-LOD-aware cascades is the likely answer for terrain-as-caster
   shadows. Bevy 0.18's atmospheric scattering improvements may help
   frame this.

---

## References

- [gen/terrestrial_pipeline_research.md](gen/terrestrial_pipeline_research.md)
  — academic and industry survey, source for the v2 backlog.
- [gen/planet_aesthetics.md](gen/planet_aesthetics.md) — visual
  target reference for orbital + surface looks.
- [gen/dunes.md](gen/dunes.md) — algorithm spec for the
  `OrientedDune`/`DuneSeaPopulation` generator.
- [gen/vaelen_processes.md](gen/vaelen_processes.md) — per-body
  process notes for Vaelen.
- [lore/solar_system.md](lore/solar_system.md) — the bodies the
  compiler must handle.
- [simulation.md](simulation.md) — big_space hierarchy, save/load,
  body state providers.
- [atmosphere.md](atmosphere.md) — atmospheric optics, ocean
  rendering, IBL.

---

*Doc owner: Korbin. Roadmap milestones served: M2, M3.*
