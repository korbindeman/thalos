# Planet Generation Pipeline — Architectural Specification

**Status:** Draft. High-level architectural specification.

---

## 1. Scope

This document defines the architecture of the planet generation pipeline: how planets are described, stored, edited, sampled, and consumed by the renderer. It establishes the contract between authoring, generation, and rendering without committing to implementation decisions that belong in the codebase.

### In scope

- Data model for planets, fields, and features
- Authoring model (stamps, generators, overlay, promotion)
- Sampling and storage architecture
- Output contract to the renderer
- CPU/GPU placement principles
- Temporal dynamics integration
- Conceptual development roadmap

### Out of scope

- Specific rendering algorithms (UDLOD internals, impostor shader implementation, instancing strategies)
- ML model training and inference (aspirational; designed for but not specified)
- Editor UI design (later phase)
- Specific data structure implementations and APIs

### Scope boundary

This pipeline specifies heightfield terrain and heightfield colliders for **spherical bodies** — bodies large enough to be in hydrostatic equilibrium. Body classification follows real physics and is static per body (the game's scale is physically accurate).

Outside this boundary, and not defined here:

- **Irregular sub-equilibrium bodies** (asteroids, small moons) use a separate mesh path with arbitrary 3D geometry and its own trimesh/convex colliders.
- **Non-heightfield geometry on spherical bodies** (caves, lava tubes, overhanging cliffs, natural arches) is additive geometry outside the heightfield pipeline, with its own trimesh colliders.

These systems may consume this pipeline's outputs where they overlap, but their internals are out of scope.

### Goals

Procedural-yet-authored planet generation, from orbital scales to centimeters. The intended workflow: define a planet physically, shuffle seeds to find a starting point, paint intent masks, place individual features, sculpt non-destructively. Multiple bodies coexist in a solar system, each with its own definition and seed.

---

## 2. Architectural overview

The pipeline has two procedural stages with an editable layer between them:

1. **Stage 1 — Procedural intent generation.** Produces coarse intent: masks and the feature catalog (procedural generators plus seed-derived instance fields).
2. **Editable intent layer.** The author's working surface. Edits to fields and features are stored as a versioned diff overlay on top of Stage 1's output.
3. **Stage 2 — Procedural detail generation (detail stage).** Consumes the intent layer to produce final terrain at the LOD the renderer requests.

The detail stage emits two output streams: a heightfield (terrain mesh data) and scatter instances (objects placed on the surface). These are consumed by separate renderer subsystems through a single Query API.

---

## 3. Core data model

### Planet

A planet is a fully self-describing value. Two systems with the same planet definition produce identical worlds. The definition contains:

- Physical parameters (radius, gravity, axial tilt, atmosphere, surface temperature, age)
- Archetype identifier (a named preset bundling defaults)
- Master seed and subseed tree
- A bag of named fields, ordered by automatic topological sort
- A registry of feature type declarations
- A spatial index of explicit feature instances (authored and promoted)
- Edit history (versioned overlay operations)

### Fields

The intent layer is a bag of named fields. Each field has metadata:

- **Semantic.** How the field's values are interpreted. Recognised semantics include SDF (signed distance to a region boundary, eikonal-correct on materialisation), Density (non-negative scalar magnitude), Categorical (class label, optionally via multi-channel argmin), Scalar (arbitrary numeric value).
- **Default value.** Used where the field hasn't been explicitly authored or generated.
- **Composition operator.** How new contributions compose with existing values.
- **Invariants.** Whether the field is maintained as a true SDF (with redistancing) or not.
- **Role.** Either `Intermediate` (consumed by other fields or generators) or `Output` (consumed by the detail stage).

Field names are arbitrary. The system does not hard-code field semantics for specific features — feature behaviour is driven by feature type declarations that reference field names.

### Field ordering

Fields form a directed acyclic graph based on expression contents. When a field's expression references another field, an edge is added. Topological order is computed automatically. The author never specifies field order manually; ordering is a consequence of dependencies.

Edits that would create a cycle are rejected at edit time with a clear error identifying the offending references.

Cascade invalidation: when a field is edited, all fields downstream in the DAG are marked dirty in the affected region.

### Feature types

Feature types are declared schemas. Each declaration includes:

- Name
- Parameters (name, type, default, optionally a source field reference)
- Generator rules (kind, density field reference, seed handling, stamp template)
- Kind, one of:
  - `TerrainModification` — feature contributes to heightfield via the detail stage
  - `SurfaceInstance` — feature is exposed as an instance placement
  - `SurfaceDecal` — texture-domain overlay (deferred)
- For `TerrainModification` features, a **composition declaration** (see Section 7):
  - Composition operators (which of additive, carve / clamp-down, flood / clamp-up, preserve-and-erode, replace-within-mask)
  - Influence-radius function of the feature's parameters (drives the tile range query during compositing)
  - Falloff curve to identity at the influence radius (C1-continuous)
  - Ordering key (e.g., age) and a deterministic tiebreak rule

Feature types are not hard-coded. New feature types are added by declaration without engine changes.

### Feature instances

Two categories of feature instance:

- **Explicit instances.** Authored from scratch (origin: `Authored`) or promoted from procedural (origin: `PromotedFrom`). Each has a stable identifier, position, type, and committed parameter values. Stored in a sparse spatial index.
- **Procedural instances.** Computed on demand by feature generators. Not stored; generated deterministically from generator rules and seed.

Querying features in a region returns the union of procedural and explicit instances, with promoted instances removed from procedural results via an exclusion index.

---

## 4. Authoring model

### Stamps

The basic unit of authored or generator contribution is a stamp. A stamp is composed of:

- **Geometry.** A geometric primitive defined by points on the sphere: Point, Capsule, Polyline (with chosen interpolation, such as linear or Bezier), Bezier (control points), PointSet (sprinkled points without interpolation).
- **Scalars.** Associated values: radius, value to write, falloff curve. Each scalar may be a Constant, PerPoint (varying along the geometry), or FromField (sampled from another field at evaluation time).
- **Composition operator.** How the stamp composes with the existing field state.

The FromField scalar value type is the mechanism by which generator outputs respond to author edits without re-running generators.

### Composition operators

The vocabulary includes smooth-min, smooth-max, add, blend, and replace, parameterised where applicable (e.g., smooth-min radius). The set is extensible; specific operators are added as needs arise.

### Generators

Generators are expression-tree nodes. They consume field references and scalar parameters as inputs, hold a seed, and produce two kinds of output: stamps targeted at a specific field, and feature instances of a specific type.

Generators are deterministic given their seed and inputs. Reshuffling a generator's seed regenerates its procedural output. Author edits to the generator's referenced fields modulate its output without requiring reshuffle.

### Author overlay

Each field has two contributions:

- **Procedural value.** Result of evaluating the field's expression tree (including stamps and generators).
- **Author overlay.** A diff layer of explicit author edits, stored as a replayable operation log.

At sample time, the overlay is composed onto the procedural value via the field's composition operator. The procedural value and overlay are stored separately and materialised separately. Reshuffling regenerates procedural without disturbing overlay; painting modifies overlay without disturbing procedural.

### Promotion

Individual procedural feature instances can be promoted to explicit instances. Promotion:

1. Captures the procedural instance's current parameters into a new explicit instance.
2. Marks the origin as `PromotedFrom(generator_id, seed_state)`.
3. Adds the position to the generator's exclusion index, preventing future double-generation.

Promoted features survive reshuffles of their originating generator. The author may then edit their parameters individually. Demotion removes the explicit instance and the exclusion entry, returning the position to procedural control.

### Edit history

All author operations (stamp, erase, modify, promote, demote, delete) are versioned. Undo/redo and version-control-style diffs operate on this log. The procedural side is not part of edit history; reshuffles are a separate operation type that modifies seeds rather than the overlay.

---

## 5. Storage

### Source of truth

The expression tree per field, plus the author overlay log, are the source of truth. Both are small (kilobytes per field for most planets) and version-controllable.

### Materialisation

A sparse hierarchical quadtree is maintained per field as a cache of evaluated values. The quadtree exists on each face of a cube-sphere parameterisation. Each quadtree node is one of:

- A uniform-value node (compresses any constant region to a single value)
- A subdivided node with children
- A leaf tile (dense array at the maximum subdivision depth used in that region)

The leaf tile size is configurable; 64 pixels per side is the starting default, to be tuned empirically against the renderer's tile structure.

Uniform regions store as a single value with no further storage cost. Authored regions subdivide only as deep as needed. Resolution is therefore unbounded in principle — fine detail in one region does not require fine resolution elsewhere.

### Two-path composition

Per field, two caches are maintained: a procedural quadtree (materialisation of the expression tree) and an overlay quadtree (materialisation of the author overlay). Sampling composes them via the field's composition operator and writes the result to a final cache.

### Cube-sphere parameterisation

Storage and sampling are organised on a cube-sphere. Specific details (face addressing, seam handling, distortion characteristics) are implementation choices; the architectural contract is that the parameterisation supports six-face quadtree storage with neighbour awareness for redistancing and continuous sampling.

---

## 6. Output contract

### Field roles

Every field is labelled `Intermediate` or `Output`. Output fields define the interface between intent layer and detail stage. The initial output field set (subject to change as the pipeline matures):

- `elevation_intent` (Scalar) — coarse height bias
- `continent_sdf` (SDF) — distance to land boundary
- `structural_intent` (Scalar) — intensity of structured terrain (mountain belts, fault zones, volcanic regions); generalises across multiple physical regimes
- `biome` (Categorical) — region label
- `material_intent` (Categorical) — dominant surface material category at coarse scale
- `climate_temperature` (Scalar) — local temperature; may be time-varying
- `climate_humidity` (Scalar) — local humidity; may be time-varying

This set is not exhaustive and may grow or shrink. Adding or removing an output is a deliberate contract change.

### Feature kinds and routing

Feature types declare a kind that routes their contributions:

- `TerrainModification` features are consumed by the heightfield synthesiser as contributions to elevation.
- `SurfaceInstance` features are exposed as instance placements in the scatter stream.
- `SurfaceDecal` is reserved for surface texture overlays; not part of initial scope.

---

## 7. Detail stage

The detail stage consumes the intent layer outputs and produces renderable terrain data. It comprises two parallel paths.

### Heightfield synthesis — two geometric bands, one surface

The heightfield is the **single geometric surface** shared by the render mesh and the physics collider. It is composed of two frequency bands:

- **Low band — intent.** Sourced directly from intent layer output fields. No synthesis; sampling the intent fields is the contribution.
- **Mid band — character.** The band where terrain character lives (mountain ridge morphology, crater rim profiles, regional roughness patterns). For the initial implementation, this band is produced analytically (parametric noise conditioned on intent). The mid band is the eventual target for learned models; replacing it does not change the contract.

These two bands produce the band-limited heightfield. **Everything finer than the mid band is explicitly non-geometric** — surface rendering only (materials, normal maps, parallax-occlusion mapping, scattered instances, decals). Non-geometric detail affects neither the visible silhouette nor collision. There is no fine mesh-displacement band.

This is a hard invariant, and it is what makes two constraints hold simultaneously:

- **Collider and render mesh are imperceptibly identical.** They are not approximations of each other; they are generated from the same band-limited heightfield. The only residual difference is LOD, which near physics bodies is held at the render LOD.
- **Terrain is cleanly walkable or cleanly not.** Band-limiting guarantees no sub-capsule-scale obstructions. Local slope is well-defined everywhere, so the walkable/slide decision (a slope threshold) is crisp rather than speckled. Cosmetic roughness lives in the shading layer where it cannot catch a physics body.

The frequency at which the heightfield stops being geometry and becomes surface rendering is shared by the mesh and the collider. It is tunable per LOD, but it is a single shared boundary, not two independent ones.

### Unified synthesiser interface

The detail stage and its sub-synthesisers (base, per-feature-type) implement a common conceptual contract: given a region, an LOD, and conditioning inputs, produce a contribution at that LOD. The implementation behind this contract may be analytic noise, parametric models, or learned models. Swapping implementations is opaque to consumers.

### Feature composition

`TerrainModification` features are composited into the heightfield as **deltas relative to the accumulated terrain**, not as absolute shapes. A feature reads the terrain as already modified by the base and any earlier features at its location and produces a modification — so a crater on a slope inherits the slope, a volcano floods low ground at its base.

**Operator vocabulary.** A feature type declares which operators it uses. The defined set:

- **Additive delta** (rim, ejecta, volcanic dome). `acc += delta · weight`. Order-independent among themselves.
- **Carve / clamp-down** (crater bowl, valley incision). Pulls the accumulator toward an excavation profile: `acc = lerp(acc, smoothmin(acc, floor_profile), weight)`. Cuts through prior material.
- **Flood / clamp-up** (lava embayment, sediment fill). Raises the accumulator to at least a fill surface: `acc = lerp(acc, smoothmax(acc, fill_level), weight)`. Buries prior topography.
- **Preserve-and-erode** (mesa caprock, inselberg). Holds a level where a resistant unit is, lowers around it.
- **Replace-within-mask** (rare, for features that fully own their footprint).

Each operator is applied through a weight field that is 1 at the feature core and smoothly → 0 at the **influence radius** (typically larger than the visible footprint, e.g., ejecta and volcanic aprons extend far beyond rim/cone). The transition is C1-continuous to avoid both visible creases and walkability speckling at the boundary.

**Ordered application.** Terrain assembly is: start with the band-limited base, then apply each feature in the region **in order**, each operating on the accumulated result. The ordering key is per-feature-type and physically motivated (age for impact features — younger overprints older — following the geological superposition principle). Ties are broken deterministically by feature ID hash so every tile computes the identical sequence.

This single rule makes intersecting features correct without special-casing. Two craters whose bowls overlap resolve correctly because the younger one's carve operator cuts through the older one's rim, applied in age order. Ejecta blankets accumulate additively (constructive overlap is order-independent); destructive overprint is handled by the carve operators in age order. A shield volcano placed near pre-existing craters can bury them via its clamp-up embayment if it is younger.

**Tile range query.** To composite a tile correctly, the detail stage must consider every feature whose **influence radius** intersects the tile, not every feature centred in it. A large feature centred far outside a tile can still affect it. The feature catalog supports the range query by influence radius; the feature type's declared influence-radius function provides the radius from each instance's parameters. The dense case (millions of small craters) remains cheap because each has a small influence radius.

**Tiling determinism.** Adjacent tiles overlapping the same feature apply the identical operator sequence in the identical order, so feature composition is seamless across tile boundaries by construction. No tile makes its own ordering decision.

Feature composition happens within heightfield synthesis at the character band. The composed result is the single band-limited surface that is both render mesh and physics collider — `TerrainModification` features are automatically present in both. Features may introduce legitimately steep clean geometry (a crater rim you slide off); the cleanly-walkable constraint forbids sub-capsule noise, not steepness.

### Scatter routing

Features of kind `SurfaceInstance` are not part of the heightfield. They are exposed via the scatter output stream as instance placements: each placement carries a position on the surface and per-instance parameters. The renderer consumes these as GPU instancing inputs.

### Learned synthesis as aspirational

Learned synthesis (diffusion or successor methods) is designed for as an eventual replacement for the mid band, per-feature-type and base alike. It is not in the initial implementation. The pipeline contract supports both analytic and learned synthesis without architectural change.

When learned synthesis is eventually integrated, base and feature models are trained on different data: base synthesisers on feature-poor source data; feature synthesisers on isolated feature exemplars. Training data preparation is an open research area, deferred.

---

## 8. Temporal dynamics

Time is an optional dimension of sampling. The conceptual sampling interface accepts a time parameter; fields whose metadata indicates time-invariance ignore it. Three patterns of dynamic content are supported architecturally:

1. **Time-varying intent fields.** The field's expression tree includes time-dependent nodes (e.g., seasonal blending of ice extent). The detail stage's synthesisers consume the time-varying intent as conditioning. Polar ice caps and seasonal coloring are handled this way.
2. **Time-varying feature instances.** Feature instances may carry an optional trajectory: a function describing how their position or parameters evolve over time. Slowly drifting dune fields are handled this way.
3. **Transient features.** Discrete features with finite time windows (lifetime start and end). Useful for event-driven content. Deferred until needed.

Static is the default; time-varying is opt-in per field or per feature instance. Caching strategy varies accordingly: static caches normally; slowly-varying caches per time-bucket and interpolates; fast-varying or trajectory-driven content is computed on demand.

---

## 9. Runtime API and consumers

### Query API surface

The runtime sees a small set of conceptual operations against a `Planet`:

- Sample a named field at a position and LOD (optionally with a time parameter).
- Generate a terrain tile (see Tile contract below) covering a region at a specific LOD.
- Query feature instances in a region, filtered by feature type or kind.
- Query scatter instance placements in a region at an LOD, filtered by type.
- Look up a specific feature instance's full data by stable identifier.
- Pre-warm a region: hint the cache to materialise specified tiles asynchronously.

These operations are asynchronous by default. Cached results return synchronously. Cold paths return futures.

The interface is defined in full from the outset. Feature and scatter operations are reserved and stubbed during phases where their backing implementation is not yet built; consumers can be written against the final shape and start receiving real results without code changes when implementation lands.

### Tile contract

Every terrain tile carries a fixed channel set sufficient for PBR rendering and physics collision:

- **Heightmap** (1 channel, float). The geometric surface. Used by the renderer for vertex displacement, by physics for the heightfield collider.
- **Material splat weights** (4 channels, normalised). Up to four materials per sample, with smooth blending at boundaries. Driven by `material_intent` with smoothness from its categorical SDF semantics. The renderer uses these to blend per-material detail textures (each material has its own albedo, normal, roughness textures sampled in shader).
- **Macro albedo modulation** (3 channels, RGB). Low-frequency colour offset on top of material splat. Captures regional variation independent of material (sun-bleached patches, mineral staining). Optional in early implementations but part of the contract.

Normals are derived from the heightmap by central-difference in the shader and are not stored per tile. Per-material constants (roughness, specular, metallic, AO) are looked up in shader from material weights and are not per-tile outputs.

Time-varying tiles carry a generation timestamp metadata field used by the cache to detect staleness when underlying time-varying intent fields advance.

### Consumers

The pipeline serves four subsystems through the Query API:

- **UDLOD terrain system.** Pulls terrain tiles continuously as the camera moves, at the LOD its tessellation requires. Tile granularity should align with the pipeline's storage tile size so leaf tiles upload directly without resampling. Async by default; never blocks the frame.
- **Flat impostor shader.** Long-distance planet rendering. Either consumes a baked low-LOD cube-map (cheap, infrequent bake) or samples intent fields parametrically (handles dynamics directly). Choice per planet.
- **Scatter renderer.** Consumes the scatter instance stream for currently visible regions. Per scatter type, generates GPU instance buffers from the pipeline's instance-query results. Instance buffers stay on the GPU.
- **Physics collider.** Consumes terrain tiles to build heightfield colliders (see below).

All consumers query the same planet at different granularities and LODs and share the same cache. New consumers (mini-map, preview thumbnails, save-game previews) plug in the same way.

### Physics collider

The collider is a Query API consumer like the renderer, not a separate pipeline. It produces collision geometry from the same band-limited heightfield the renderer consumes.

- **Source and type.** Generated from `generate_terrain_tile` output — the same band-limited heightfield surface as the render mesh. Because the terrain is a displacement field with no overhangs, the collider is a **heightfield collider** (a 2D grid of heights), not a trimesh. The heightmap-to-collider conversion is engine-side; the pipeline needs no new operation.
- **Imperceptible by construction.** The collider is not an approximation of the render mesh; it is generated from the identical band-limited heightfield. Near physics bodies the collider LOD is held at the render LOD, so the two surfaces are the same. This is the single-geometric-surface invariant from Section 7, applied to physics.
- **Cleanly walkable.** Because the heightfield is band-limited (no sub-capsule-scale obstructions), local slope is well-defined and the walkable/slide decision is crisp everywhere. Cosmetic roughness is non-geometric and cannot catch a physics body.
- **Windowed streaming.** Colliders exist only where physics bodies can be. The collider system tracks active physics bodies and ensures collider tiles exist within a radius at render LOD, generated on demand from the shared cache, evicted when no body is near. Cost is bounded by body count and velocity, not planet size.
- **Determinism guarantees consistency.** A terrain tile for a region at an LOD is bit-identical whether the renderer or the collider requested it. Author edits and procedural content are both present because both flow through `generate_terrain_tile`.
- **Features and scatter.** `TerrainModification` features are in the heightfield, so they are in the collider automatically. Large collidable `SurfaceInstance` features (boulders, tree trunks) get per-instance colliders attached on proximity to a physics body, derived from instance parameters. Small scatter (pebbles, grass) has no collision.
- **Tile-edge seam-closing is a correctness requirement.** Adjacent collider tiles must not leave gaps at boundaries (a seam a body could fall through). Handled by slight overlap or edge-snapping, the same pattern as render-mesh stitching, but treated as correctness rather than visual polish.
- **Time-varying geometry.** Appearance-only dynamics (seasonal colouring) have no collider impact. Geometry-changing dynamics (drifting dune fields) require collider tiles near physics bodies to refresh on the relevant time-bucket cadence — only within the windowed region, not planet-wide.

---

## 10. Caching

A four-tier hierarchy underlies the Query API:

- **L1 — GPU.** Currently rendered tiles. Managed by the renderer.
- **L2 — CPU RAM.** Recently used tiles, not currently active. In-memory LRU.
- **L3 — Disk frecency cache.** Persistent per-planet cache on disk, surviving sessions. Tiles age out by combined frequency and recency. Size-capped, user-configurable. Indexed binary store.
- **L4 — Pipeline generation.** Materialise the expression tree for the requested region. The cold path.

The cache layer is shared across consumers and transparent to them. Cache invalidation is local — editing the planet definition in a region invalidates only the affected tiles, identified by overlap with the edit's footprint. A pipeline version hash in the cache key invalidates everything on detail-stage version changes.

### Bake at planet load

Intent layer materialisation is baked to a fixed LOD at planet load (small storage, always loaded). Higher LODs are materialised lazily through the cache.

### Detail-stage tiles

Detail-stage outputs (heightfield tiles, scatter placements) are cached by the same hierarchy. For the analytic detail stage, runtime cost is low and L3 is a modest optimisation; the MVP may omit L3 entirely and ship with L1/L2 only, adding L3 later when profiling justifies it or when learned synthesis arrives. For the aspirational learned detail stage, runtime cost is high and L3 becomes critical. The architecture supports both without change.

---

## 11. CPU/GPU split

The pipeline targets a deployable runtime across platforms. The placement principle:

- **Non-feature stages** (intent expression evaluation, mask compositing, quadtree management, SDF redistancing) run on the CPU by default. GPU acceleration is acceptable where it fits but is not required.
- **Feature generators** run on the GPU. Generators that produce dense scatter (pebbles, grass, debris) must dispatch as compute shaders to avoid CPU stalls and to scale to billions of instances visible per frame.

This split keeps the pipeline broadly portable while ensuring feature placement remains performant on any target with a GPU.

---

## 12. Phasing

Phases describe the order in which the pipeline is built. Each phase produces a working subset; later phases extend rather than replace earlier ones.

- **Phase A — Core data model and pipeline.** Field bag, DAG ordering, expression trees, stamps, generators, feature catalog, two-path composition, sparse quadtree storage, cache layer. No rendering, no editor. Foundation only. Validated by sampling and determinism tests.
- **Phase B — Analytic synthesis and temporal.** Two-band heightfield synthesis (analytic) producing the single mesh-and-collider surface, feature routing, time-varying intent fields supported. The first phase producing terrain a renderer and physics can consume.
- **Phase C — Authoring interface.** Editor for fields, masks, features. Brush tools, promotion, seed shuffling. Time-varying feature trajectories supported when content needs them.
- **Phase D — Persistent infrastructure.** Disk-backed frecency cache, planet serialisation (RON-based with binary asset bundles), edit history with undo/redo, version control friendliness. Transient features added if gameplay demands them.
- **Phase E — Per-feature learned synthesisers (aspirational).** Replace one feature synthesiser at a time with learned models. Validates ML integration on constrained problems.
- **Phase F — Learned base synthesiser (aspirational).** Replace the analytic base mid-band synthesiser. Largest training effort.

Phases A through D constitute a complete shipped terrain system. Phases E and F are aspirational upgrades on independent timelines. Each is per-planet — different bodies in a solar system can run on different synthesisers without architectural disruption.

---

## 13. Open questions

Items discussed and deliberately not decided in this specification:

- **Cube-sphere parameterisation specifics.** Face mapping, distortion handling at corners, redistancing across seams.
- **Tile size optimisation.** 64 pixels is the starting default; final value tuned empirically against the renderer's tile structure and cache behaviour.
- **Biome-specific synthesis structure.** Whether the base synthesiser internally selects per-biome sub-synthesisers blended at boundaries, or treats biome as conditioning input.
- **Specific generator kinds beyond Poisson scatter.** Discovered as feature types require them.
- **Bowl-profile slope-awareness for impact features.** v1 uses a slope-agnostic excavation profile and accepts mild unrealism on steep pre-existing slopes; the feature synthesiser has access to accumulated terrain so a slope-aware profile is a later refinement, not an architectural change.
- **Ordering key for non-impact feature types.** Age is obvious for craters and volcanics. Mesas and other erosional remnants may not "overprint" at all; their key may be irrelevant or explicit z-order. Decide per type as the type set grows.
- **Influence-radius cap for pathological cases.** A feature with an extremely large influence radius makes many tiles' range queries expensive. Per-type sane caps, empirical.
- **Training data preparation for learned synthesis.** Feature-poor curation for base, isolated patches for features, delta vs. absolute targets. Open research.
- **Game-time vs. planetary-time coupling.** How real game time maps to planetary calendars (seasons, day/night). Out of scope for the pipeline itself.
