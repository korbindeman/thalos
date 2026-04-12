# Surface Generator — Design Document

## 1. Scope

Generate the visible solid surface of bodies that are well-approximated as a sphere plus a height field. Output is a sampleable field consumable by both an impostor renderer and a UDLOD terrain renderer.

**In scope:** spherical rocky and icy bodies large enough to be in (or near) hydrostatic equilibrium — Mira, the hot rocky world, Venus analogue, homeworld, Mars analogue, large rocky/icy moons, spherical dwarf planets.

**Out of scope:**
- Atmospheres, clouds, weather, scattering, aurorae — handled by the **atmosphere system**.
- Gas giants and ice giants in their entirety — they have no solid surface; their visible "skin" is the atmosphere system's product.
- Small irregular bodies (asteroids, small moons, comet nuclei, contact binaries) — handled by the **mesh pipeline**.
- Bodies with significantly non-spherical equilibrium shapes (Haumea-class fast rotators in Jacobi-ellipsoid equilibrium) — **deferred**. These need a shape representation the surface generator doesn't currently support, and the right home for them (extend this system, or absorb into a sufficiently capable mesh pipeline) will be decided when the mesh pipeline exists and we know what it can do.

**Note on oblateness:** every rotating body in equilibrium is technically an oblate spheroid, but for every body in scope the flattening is below visual threshold (sub-km on bodies with km-scale terrain features, dominated by tidal locking on the moons and slow rotation on the planets). The surface generator treats all bodies as spheres. If a future feature ever needs sub-visual oblateness (precision orbital mechanics, J2 effects), it can be handled outside the surface representation.

## 2. Philosophy

- **Approximate physical processes, not noise stacks.** Each generation stage corresponds to a real process (impact cratering, mare flooding, regolith gardening, …). Stages compose in roughly chronological order. This is what makes the system generalize across body types.
- **Band-limited representation.** The output has no fixed resolution. Each layer covers a specific spatial-frequency band; the layers together span from body-scale (hundreds of km) down to regolith-scale (~1 m). The renderer asks for whatever LOD it needs and gets the appropriate bands evaluated.
- **One sampling contract.** Both renderers — flat impostor and UDLOD terrain — call the same `sample(dir, lod)` function. Renderer-specific code is purely about *where* to sample and *how dense*, never about *what* the surface looks like.
- **Everything tunable.** Every numeric parameter that controls a process is exposed in per-body config. Defaults are chosen to produce realistic results for the target body, but every value is reachable from the body's KDL definition.

## 3. The body taxonomy

A single struct, `BodyData`, covers all in-scope bodies. Composition (rocky vs. icy) is a *parameter*, not a type, because rocky and icy bodies share all the same processes — only the materials palette and parameter ranges differ. Cryovolcanism is volcanism with different materials; ice cratering uses the same morphology rules with different rim/floor parameters.

All bodies are spheres. The reference shape is `radius_m`, full stop. See section 1's note on oblateness for why this is sufficient across the in-scope inventory.

## 4. The BodyData contract

There are two structs: `BodyBuilder` is mutable and used during pipeline execution; `BodyData` is immutable and what the GPU sees. The pipeline runner finalizes a builder into a `BodyData` once all stages have run.

```rust
// Runtime, GPU-facing. Immutable after construction.
struct BodyData {
    // Identity & global params
    radius_m: f32,

    // Low-frequency baked layer (broad, smooth features)
    height_cubemap: Cubemap<R16>,      // displacement from radius_m, in meters
    albedo_cubemap: Cubemap<RGBA8>,    // base color + roughness in alpha

    // Mid-frequency analytic layer (discrete features)
    // Any of these may be empty for bodies that don't use that feature type;
    // empty SSBOs cost nothing at sample time.
    craters: Vec<Crater>,
    volcanoes: Vec<Volcano>,
    channels: Vec<Channel>,            // rifts, grabens, ancient riverbeds
    feature_index: IcoBuckets,         // shared spatial index over all feature SSBOs

    // High-frequency analytic layer (statistical detail)
    detail_params: DetailNoiseParams,  // includes statistical small-crater layer

    // Materials, indexed by feature.material_id
    materials: Vec<Material>,

    // Optional: global structures from coherent stages
    plates: Option<PlateMap>,          // homeworld only
    drainage: Option<DrainageNetwork>, // bodies with fluvial erosion
}

// Build-time only. Holds scratch state, intermediate accumulators,
// and parameters that don't need to live on the GPU.
struct BodyBuilder {
    radius_m: f32,
    seed: u64,
    composition: Composition,                  // drives Differentiate; not needed at runtime
    cubemap_resolution: u32,                   // per-body, see below

    // Scratch: accumulating cubemap contributions before bake
    height_contributions: CubemapAccumulator,
    albedo_contributions: CubemapAccumulator,

    // The fields that will become BodyData
    craters: Vec<Crater>,
    volcanoes: Vec<Volcano>,
    channels: Vec<Channel>,
    detail_params: DetailNoiseParams,
    materials: Vec<Material>,
    plates: Option<PlateMap>,
    drainage: Option<DrainageNetwork>,
}
```

The GPU never sees `BodyBuilder`; it only sees the finalized `BodyData`. `composition` lives on the builder because it drives `Differentiate` but has no purpose after the materials palette is built. `seed` lives on the builder for the same reason: stages consume it during generation but the runtime needs no RNG.

### Per-stage seeds

Stages do not use `builder.seed` directly. Each stage derives its own seed via `hash(builder.seed, stage_name)`. This means tweaking one stage's parameters doesn't reshuffle the RNG state of every other stage — change the crater count and only the craters change, not the maria and the noise. This is a small detail with a large quality-of-life payoff during tuning.

### Materials are immutable after Differentiate

`Differentiate` runs first in every stage list and is the only stage allowed to write to `builder.materials`. After it completes, the materials palette is frozen for the rest of the pipeline run. Features can append themselves with `material_id` references that will remain stable. Without this rule, a later stage could rebuild the materials vec in a different order and silently invalidate every existing feature's index.

### Cubemap baking

The cubemap is not written directly by stages. Stages write to `height_contributions` and `albedo_contributions` accumulators on the builder. After all stages run, the pipeline runner finalizes the accumulators into the `height_cubemap` and `albedo_cubemap` on `BodyData`. There is no `BakeCubemap` stage in the per-body lists — the bake is an implicit final step the runner performs.

### Layer responsibilities

| Layer | Frequency band | Storage | Sampling cost |
|---|---|---|---|
| Cubemap | > ~5 km | per-body resolution × 6 faces, R16 + RGBA8 | 1 texture fetch (×4 for normals) |
| Feature SSBOs | ~threshold to ~5 km | up to ~10k entries per body, tunable | iterate near buckets, ~20–60 features per sample |
| Detail noise | < ~threshold | analytic, no storage | a few octaves of noise per sample |

The threshold between feature SSBO and detail noise is set per-body. A tighter threshold = larger SSBO = more visible discrete features at the cost of bandwidth. Default for Mira: ~1.5 km, giving ~10k craters.

### Cubemap resolution is per-body

Cubemap resolution is a per-body parameter, not a system-wide constant. The default is whatever gives roughly 3 km/texel at the equator: ~512² per face for Mira (869 km radius), ~1024² for the homeworld (3186 km), ~2048² for the Venus analogue (4035 km). Smaller bodies don't benefit from higher resolution because the cubemap only carries the low-frequency band — finer detail belongs in the feature SSBO and noise layers, where it's free to push as fine as needed.

The resolution lives on `BodyBuilder` and is consumed by the bake step. It's exposed as a tunable in the body's KDL, with a sensible default derived from `radius_m`.

### Spatial index

Feature SSBOs are indexed by an icosahedral subdivision of the sphere (level 4 = 1280 cells, ~30 km characteristic spacing on Mira). Each cell stores indices into the feature arrays for features whose influence radius overlaps it. Sampling iterates the cell containing `dir` plus its immediate neighbors. The same icosahedron can drive UDLOD subdivision, sharing structure between systems.

## 5. The sampling function

```rust
fn sample(body: &BodyData, dir: Vec3, lod: f32) -> SurfaceSample;

struct SurfaceSample {
    height: f32,         // meters above radius_m
    normal: Vec3,        // world-space, derived from height gradient on the sphere
    albedo: Vec3,
    roughness: f32,
    material_id: u32,
}
```

`lod` is `log2(meters_per_sample)` at the query point. Larger = coarser.

**Branching by LOD:**

1. **Always**: read the cubemap layer (height + albedo). For the impostor this is the only step (4 fetches for derived normals).
2. **If `lod < cubemap_threshold`** (~12 for 512² on Mira): iterate features in nearby buckets. Each feature contributes only if its size is resolvable at this LOD; contributions are windowed by `smoothstep(2·sample_spacing, 4·sample_spacing, feature_radius)` so they fade in continuously rather than popping.
3. **If `lod < detail_threshold`** (~6, ~64 m/sample): evaluate detail noise. Octave count derived from LOD so high-frequency octaves drop out smoothly with distance. The noise must visually continue the SFD of the discrete feature SSBOs below their threshold — see section 7's regolith stage. Without this, the handoff between explicit craters and noise is visible as a seam.

**Continuity in LOD is the key invariant.** New frequency content fades in via smoothstep windows; it never appears or disappears abruptly. This is what prevents both aliasing at coarse LOD and popping across LOD transitions.

### Normals

Always derived, never baked. The shader samples `height` at `dir` and at four directions offset by a small angle in two orthogonal tangent directions on the sphere, then builds the normal from the height differences and the tangent frame at `dir`. Six lines of shader code, identical across impostor / UDLOD vertex / UDLOD fragment.

This means the normal automatically reflects whichever bands are active at the current LOD: cubemap-only when far, plus features when close enough, plus noise when closer still. No tangent-space blending of separately-computed normals, no chance of mismatch between height and normal.

If profiling later shows the impostor is texture-fetch bound, a baked normal cubemap can be added as a pure optimization without changing the contract.

### Renderer call sites

- **Impostor**: per fragment, `dir = sphere_direction_from_uv(uv, view); s = sample(body, dir, impostor_lod);` Returns immediately after the cubemap read because `impostor_lod` is large.
- **UDLOD vertex**: per vertex, `s = sample(body, vertex_dir, lod_from_triangle_size); displaced_pos = vertex_dir * (body.radius_m + s.height);`
- **UDLOD fragment**: optionally re-samples at finer LOD for per-pixel detail, depending on triangle density.

## 6. The pipeline

A `BodyData` is produced by running an ordered list of stages on a `BodyBuilder`. Each stage implements a common trait:

```rust
trait Stage {
    fn name(&self) -> &str;
    fn dependencies(&self) -> &[&str];           // names of stages that must run before this one
    fn apply(&self, builder: &mut BodyBuilder, params: &StageParams);
}
```

Stages contribute output in one of three forms:

1. **Cubemap contributions** (megabasin, mare flood, space weathering, broad volcanic plains). These write to the builder's `height_contributions` and `albedo_contributions` accumulators. The pipeline runner bakes them into the final cubemap textures after all stages complete.
2. **Feature SSBO appends** (cratering, discrete volcanoes, rift systems). These append entries to the relevant feature list on the builder.
3. **Global structures** (plate tectonics, drainage networks). These produce intermediate global data structures stored on the builder that later stages and the sampler consult.

Stages may also **read and modify** outputs of prior stages, not just append. `MareFlood` removes buried craters from the SSBO; `AeolianErosion` smooths near-threshold features and modifies cubemap contributions. The pipeline ordering is meaningful: erosive and modifying stages run after the stages whose output they consume.

### Dependency checking

Each stage declares the names of the stages it depends on. At body-load time, the pipeline runner validates that the configured stage list satisfies all declared dependencies in order — every dependency must appear before the stage that requires it. Invalid stage lists fail at startup with a clear error, rather than producing subtly wrong terrain at runtime.

Dependencies are *required prerequisites*, not soft hints. `MareFlood` declares a dependency on `Cratering`. `AeolianErosion` declares dependencies on whichever feature-producing stages it erodes.

### Failure handling within a stage

Stages should be **permissive about under-supply**. If `MareFlood` is configured to fill 5 basins but `Cratering` only produced 3 large enough, `MareFlood` fills 3 and logs a warning — it does not error. This means retuning earlier stages doesn't break later ones in surprising ways. Hard requirements are expressed via declared dependencies (which fail fast at startup), not via runtime errors deep in a stage.

### Cubemap bake (implicit final step)

After all stages run, the pipeline runner finalizes the builder into a `BodyData`: it bakes the cubemap accumulators into `height_cubemap` and `albedo_cubemap` at the per-body resolution, builds the `feature_index` from the populated SSBOs, and drops the build-time-only fields (`composition`, `seed`, scratch state). This bake step is not a stage and does not appear in the per-body stage lists.

The pipeline runs CPU-side in Rust at body-load time, caches the result, and the GPU only ever reads `BodyData` — it never runs the pipeline.

### Per-body stage lists

| Body | Stages |
|---|---|
| Mira | `Sphere → Differentiate → Megabasin → Cratering → MareFlood → Regolith → SpaceWeather` |
| Hot rocky (I) | `Sphere → Differentiate → Cratering → ThermalProcessing` |
| Mars analogue (V) | `Sphere → Differentiate → ShieldVolcanism → RiftSystems → Cratering → AeolianErosion → RelictFluvial → PolarCaps` |
| Homeworld (III) | `Sphere → Differentiate → PlateTectonics → Volcanism → Hydrosphere → FluvialErosion → AeolianErosion → Cratering → Biosphere` |
| Ice moon | `Sphere → Differentiate → Cratering → Cryovolcanism → TidalFractures → SpaceWeather` |

There is no separate "light cratering" stage. Bodies that want fewer craters (like the homeworld, where erosion has erased most impact features) just pass a smaller `total_count` to the same `Cratering` stage. Stage *behavior* is shared across bodies; stage *parameters* are per-body.

`PlateTectonics` is the only stage that fundamentally needs a global pre-computation pass. It is also the only homeworld-specific stage. Every other stage operates per-region or per-feature and is reusable across bodies.

## 7. Mira in detail

Mira's recipe and the parameter shape for each stage. These are the tunables that live in Mira's KDL.

- **Sphere**: `radius_m = 869_000`.
- **Differentiate**: reads the body's `composition` parameter and expands it into the concrete materials palette and per-stage parameter ranges used downstream. For Mira: `composition = silicate_dominated`, `iron_fraction = 0.10`. Sets the materials palette to highlands anorthosite + mare basalt + fresh ejecta + space-weathered regolith. This stage is where the high-level "what is this body made of" parameter becomes the concrete inputs the rest of the pipeline consumes.
- **Megabasin**: one or two large gaussian-on-sphere depressions imprinted into the cubemap height contribution. Parameters: `count`, `center_dir[]`, `radius_km[]`, `depth_km[]`, plus a hemispheric asymmetry bias.
- **Cratering**: SFD-driven population sampler. Parameters: `total_count` (default 10_000), `sfd_slope` (default −2.0), `min_radius_m` (default 1_500), `max_radius_m` (default 250_000), `morphology_thresholds` (radii at which simple → complex → peak-ring → multi-ring transitions occur), `age_distribution`. The largest few craters from this stage are the named basins; the rest fill the surface.
- **MareFlood**: pick the N largest basins from cratering output by depth, fill to a per-basin level. Parameters: `target_count`, `fill_fraction`, material switches to mare basalt. Buried craters below the fill line are removed from the SSBO and replaced by smooth fill in the cubemap.
- **Regolith**: configures `detail_params` for crater-shaped statistical noise. Parameters: `amplitude_m`, `characteristic_wavelength_m`, `crater_density_multiplier`. Must visually continue the SFD below the cratering threshold.
- **SpaceWeather**: writes the albedo cubemap. Old terrain darkens; recent crater rays bright. Parameters: `weathering_rate`, `ray_decay_age`, `highland_albedo`, `mare_albedo`.

Expected output for Mira: an 869 km silicate body with two ancient basins, ~10k explicit craters spanning 1.5 km to ~250 km, a few maria filling the largest old basins, statistical small craters down to regolith scale, and an albedo field showing fresh ray systems on the youngest features.

## 8. Tunability

All parameters live in the body's KDL definition, in a `generator { … }` block colocated with the existing physical parameters (mass, orbit, etc.). Stages are listed in order; each stage has its own parameter sub-block. Defaults exist for everything but can be overridden per body.

This colocation means a body's complete definition — orbital, physical, and procedural — is one file, one source of truth.

## 9. Forward-looking: what each future body adds

What needs to be built when each next body type is tackled. None of this is in scope for the Mira milestone, but the architecture preserves the seams.

### Mars analogue
- **New stages**: `ShieldVolcanism` (broad cones with summit calderas, new feature SSBO entry type `Volcano`), `RiftSystems` (curve-on-sphere features, new SSBO type `Channel`), `AeolianErosion` (read-modify-write on cubemap and on near-threshold features), `RelictFluvial` (also produces channels), `PolarCaps` (latitude/elevation-driven albedo modifier).
- **Architectural pressure**: confirms that the mid-frequency layer needs more than just craters. The multiple-SSBO design handles this. First read-modify-write stage forces clean ordering rules: erosion runs after the features it erodes.
- **Atmospheric coupling**: surface generator output is consumed by the atmosphere system for ground albedo and topography; no change to this contract.

### Homeworld
- **New stages**: `PlateTectonics` (the only stage that needs global pre-computation; produces a `PlateMap` global structure), `Hydrosphere` (sea-level scalar that gates erosion and modifies albedo/material below threshold), `FluvialErosion` (needs `DrainageNetwork` global structure, computed from coarse height field), `Biosphere` (vegetation as material/color modifier driven by latitude, elevation, moisture).
- **Architectural pressure**: validates the "global structures" category of stage output. `PlateMap` and `DrainageNetwork` are neither cubemap contributions nor SSBO appends — they're coarse global data the sampler consults. Worth budgeting time for: drainage networks in particular are iterative and expensive to compute, even at coarse resolution.

### Ice moons
- Mostly a reskin: `Cryovolcanism` is `Volcanism` with different materials and parameters, `TidalFractures` reuses the `Channel` SSBO from Mars. Confirms that composition-as-parameter (rather than rocky-vs-icy as separate types) is the right call.

## 10. System boundaries

Three parallel systems collectively handle all visible bodies:

| System | Handles | Output |
|---|---|---|
| **Surface generator** (this doc) | Spherical rocky and icy bodies | `BodyData`, sampled via `sample(dir, lod)` |
| **Atmosphere generator** | All atmospheres, clouds, weather, gas/ice giants in entirety | Volumetric + impostor representations, visually equivalent |
| **Mesh pipeline** | Small irregular bodies (asteroids, small moons, comet nuclei) | Meshes with normal maps, treated as regular game objects |

The systems meet at two points only:
1. **The lower atmospheric boundary**: the atmosphere system reads the surface via `sample(dir, lod)` for ground albedo, topography, and shadow casting. One-way query. The atmosphere system may read at its own generation time (baking ground appearance into atmospheric LUTs) and/or at render time (per-frame queries where ground truth matters). Both are supported by the same contract — `sample()` is pure and stateless, so it doesn't matter when it's called.
2. **Impostor compositing**: at far distances the surface impostor and atmosphere impostor render together. The renderer (not the generators) handles this — surface produces color + depth/coverage, atmosphere reads that and adds scattering and clouds on top.

Neither generator knows about the other. The renderer that uses them does.

## 11. Open questions and deferred decisions

- **Drainage network resolution and algorithm.** Coarse flow accumulation on a sphere is non-trivial; revisit when starting the homeworld.
- **Cubemap face seams.** Sampling at face boundaries needs careful UV handling. Standard problem with standard solutions; flag when implementing the sampler.
- **GPU vs. CPU pipeline execution.** Initial implementation runs the pipeline on CPU at body-load time. If body-load latency becomes a problem, individual stages can be ported to compute shaders. The stage trait should be designed to allow this without changing the call sites.
- **Time-varying surfaces.** Currently assumed static. If active volcanism, seasonal polar caps, or tidal flexing need to animate, the sampler grows a `time` parameter and certain layers become functions of it. Defer until a body needs it.
- **Crater bucket density worst case.** At 10k craters, level-4 icosahedron buckets average ~8 features each. If clustering is severe in practice, may need level 5 (5120 cells). Confirm during cratering implementation.
- **Haumea-class fast rotators in Jacobi-ellipsoid equilibrium.** Deferred. They're large, smooth, and in equilibrium (so they'd benefit from procedural processes), but their shape isn't a sphere. Two possible homes when revisited: (a) extend the surface generator with an ellipsoid reference shape, used by this one body; or (b) build the mesh pipeline capable enough to bake procedural cratering and weathering into authored ellipsoidal meshes. The right call depends on how sophisticated the mesh pipeline ends up being and whether other non-spherical equilibrium bodies appear. Don't decide until both systems exist.
- **Mesh pipeline interface.** Out of scope for this doc, but worth a separate short doc when starting on asteroids — particularly the size/shape threshold that decides whether a body goes to surface generator or mesh pipeline.
