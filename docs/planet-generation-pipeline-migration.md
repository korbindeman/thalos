# Planet Generation Pipeline — Migration Plan

**Status:** Draft. Companion to
[planet-generation-pipeline-spec.md](planet-generation-pipeline-spec.md).

The spec describes the *target* architecture as if from scratch. This
doc maps it onto the crates that exist today and sequences a
**brownfield** migration: the game loads pre-baked terrain and renders
Mira/Vaelen/Thalos through an impostor + UDLOD stack right now, and must
keep doing so at every step. No flag day.

It also serves the second goal driving this work — **streamlining planet
rendering** — because the same seam that de-risks the generation rewrite
(a single Query API) is what collapses today's two divergent surface
synthesisers into one.

## 1. Relationship to the other docs

- **[planet-generation-pipeline-spec.md](planet-generation-pipeline-spec.md)**
  is the architectural target and owns the *what*. This doc owns the
  *how and in what order*.
- **[terrain.md](terrain.md)**'s **generation half is superseded** by
  the spec. Its feature-compiler model (`PlanetTerrainSpec → TerrainPrior
  → FeatureManifest → SurfaceField`, populations + promotion, era
  ordering, the v2 backlog) is not thrown away — it is *folded in* as the
  spec's feature catalog, generators, and synthesisers. terrain.md's
  **ground-LOD / `TileProvider` half stays** as the consumer-side
  mechanics, reframed below as a Query-API consumer.
- **ROADMAP** M2 ("terrain pipeline revamp") is re-pointed at this
  migration; M3 ("ground LOD") is already wired and becomes a consumer of
  the new seam rather than a separate pipeline.

## 2. The four crates after migration

The crate boundaries do **not** change. What changes is what lives
behind them. (Dependency direction is unchanged and acyclic.)

| Crate | Bevy? | Today | After migration |
|---|---|---|---|
| `thalos_terrain` | no | Archetype stage pipeline → full-res baked `PlanetSurface` | **The pipeline + Query API.** Field-DAG intent layer, feature catalog, two-band detail stage, sparse quadtree storage, cache. Emits the Tile contract + scatter stream. Still Bevy-free. |
| `thalos_terrain_render` | yes | UDLOD integration; **owns its own HMF detail cascade**; collider patches | **Query-API consumer.** `PipelineTileProvider` and collider patches both pull `generate_terrain_tile`; the local detail cascade is deleted (moves into the detail stage). Hosts GPU scatter dispatch. |
| `thalos_planet_rendering` | yes | Impostor + gas/ring/solid materials; bakes from `PlanetSurface`; post-processing | **Far/impostor consumer.** Impostor bakes a low-LOD cube-map *via the Query API* (or samples intent fields for dynamic bodies). Gas/ring/solid/post unchanged. |
| `thalos_planet_lighting` | yes | Shared lighting/atmosphere types + WGSL + Hapke BRDF | **Unchanged.** The spec puts shading out of scope; this crate is already the single shading source of truth both renderers use. |

Net effect on rendering: one band-limited surface, produced once, baked
to the impostor cube-map and streamed to UDLOD tiles and the collider —
all through one contract, all shaded through one BRDF. That is the
streamline.

## 3. The gap: spec concept → today → action

`KEEP` = current code substantially matches the spec. `ADAPT` = reshape
existing code. `REPLACE` = different architecture, rebuild behind a seam.
`NEW` = does not exist.

| Spec concept (§) | Today | Action |
|---|---|---|
| Self-describing `Planet` value (§3) | `TerrainConfig` enum of archetype presets + per-body RON ([terrain_config.rs](../crates/terrain/src/terrain_config.rs)) | REPLACE schema, keep RON-on-disk |
| Named **fields** + automatic **DAG** ordering (§3) | none — archetype stages mutate `BodyBuilder` in fixed order | NEW |
| **Feature types** as declared schemas (§3) | `FeatureKind` hardcoded; only `Megabasin` authorable | REPLACE with declarations |
| Explicit + procedural **instances**, **promotion** (§4) | `AuthoredFeatureConfig` + `FeatureSeed`/`FeatureLock` seeds; promotion designed in terrain.md, not built | ADAPT/BUILD — terrain.md's populations+promotion model ≈ the spec |
| **Stamps / generators / author overlay** (§4) | none | NEW |
| **Sparse quadtree** storage on cube-sphere (§5) | full-res `Cubemap<T>` ≤ 4096² baked into `StaticSurfaceData` ([static_surface.rs](../crates/terrain/src/static_surface.rs)) | REPLACE storage; cube-sphere param is already there |
| **Two-path composition** (procedural + overlay) (§5) | none | NEW |
| **Output fields** (`continent_sdf`, `structural_intent`, …) (§6) | implicit, pre-composited into baked cubemaps | NEW (the intent layer) |
| **Two-band heightfield**, fine detail non-geometric (§7) | **three geometric layers** (cubemap + crater SSBO + statistical noise, all displace height) **plus** a separate geometric HMF cascade in ground LOD | REPLACE — see §4 |
| **Feature composition** as ordered deltas (§7) | crater bake + SSBO; era ordering specced in terrain.md | ADAPT into the detail stage |
| **Query API** — `generate_terrain_tile`, `sample_field`, `query_features`/`query_scatter`, `prewarm` (§9) | `sample_static_surface(dir, lod)` point sampler + UDLOD-specific `TileProvider::request_tile` | NEW seam (wrap, then swap — §5) |
| **Tile contract** — height + 4-ch splat + macro albedo (§9) | ad-hoc per-attachment R16/Rgba8 in `PipelineTileProvider`; single material-ID cubemap + palette | ADAPT (material model changes — §7) |
| **Scatter** stream + renderer (§7, §9) | none | NEW (stub the API early, fill in Phase B) |
| **Physics collider** as a Query consumer (§9) | `rendered_height_m` + `build_rendered_terrain_patch` already share the ground surface ([rendered_height.rs](../crates/terrain_render/src/rendered_height.rs)) | KEEP approach, re-point at Query API; gain band-limiting |
| Normals derived in-shader, not stored (§9) | impostor + ground LOD both finite-difference height; baked `normal_cubemap` largely unused | KEEP — drop the baked normal cube |
| **Four-tier cache** (§10) | single-file whole-planet bake ([cache.rs](../crates/terrain/src/cache.rs)) + UDLOD `TileAtlas` (GPU) | ADAPT: add L2 RAM + per-tile L3; keep a planet-load bake |
| **CPU/GPU split** — feature gens on GPU (§11) | all CPU (rayon) | DEFER; keep `terrain` Bevy-free, dispatch GPU scatter on the consumer side (§6) |
| **Temporal dynamics** (§8) | `DynamicSurfaceLayers` (ice, dunes) bolted onto the baked product | ADAPT into time-varying intent fields / feature trajectories |
| **Learned synthesis** (§7) | none | Aspirational (spec Phases E/F) |

## 4. The rendering problem, stated precisely

Today a single body is described by **two surfaces that do not agree**:

- **Impostor** ([sample.rs](../crates/terrain/src/sample.rs)): baked
  cubemap + screen-faded explicit crater SSBO + statistical crater noise
  below 500 m. All three displace height.
- **Ground LOD** ([pipeline.rs](../crates/terrain_render/src/pipeline.rs)):
  the same baked cubemap + a domain-warped **ridged-HMF cascade**
  (`DETAIL_AMP_M = 250 m`, down to ~0.49 m wavelength). Different noise,
  different character.

The module docstring already concedes the consequence: a CPU/GPU
"bilinear stand-off" where the EVA controller floats O(10–20 cm) relative
to the rendered mesh, and a visible seam at the `4 × radius` impostor↔terrain
swap because the two paths are different *interpretations*, not two LODs
of one surface.

The spec fixes this structurally (§7):

1. **One band-limited heightfield** is the single geometric surface for
   *both* the render mesh and the collider. Low band = intent fields;
   mid band = character (analytic now, learned later).
2. **Everything finer than the mid band is non-geometric** — POM /
   normal maps / scatter in the shader, affecting neither silhouette nor
   collision. This is what makes terrain *cleanly* walkable (no
   sub-capsule speckle) — a property today's geometric sub-metre HMF
   does **not** guarantee.
3. Impostor, ground LOD, and collider are **consumers** of that one
   surface via `generate_terrain_tile`. The `4 × radius` hard swap
   becomes a crossfade between two LODs of the same data.

So "streamline rendering" is not a cleanup pass — it is item (3),
delivered first, as the seam the rest of the migration hides behind.

## 5. Strategy: strangler-fig around the Query API

Build the **Query API seam first**, wrap the *current* pipeline behind
it, move all consumers onto it, then replace the backing body-by-body.
This front-loads the rendering win and means generation is never rewritten
in a big bang — un-migrated bodies keep working through the adapter.

```
            ┌─────────────────────── thalos_terrain ───────────────────────┐
consumers → │  Query API (generate_terrain_tile / sample_field / query_*)   │
            │        │                                   │                   │
            │   [adapter over CURRENT pipeline]   [NEW field-DAG + detail]   │
            │    (whole bodies, pre-cutover)       (bodies, post-cutover)    │
            └───────────────────────────────────────────────────────────────┘
   ▲ PipelineTileProvider (UDLOD)   ▲ impostor bake   ▲ collider patch   ▲ scatter
```

Caveat to be honest about: in the wrap step the *geometric* surface
(height/albedo/roughness) unifies across impostor-bake, ground tile, and
collider immediately. The impostor shader's **high-frequency statistical
crater synthesis** in `planet_impostor.wgsl` is the one piece that can't
fully reconcile until fine detail formally becomes non-geometric — that
reconciliation lands in Phase B, not the wrap. The wrap kills the
collider/mesh stand-off and the two-geometric-synth divergence; the
shader-detail match follows.

## 6. Migration phases

Each phase ships a working subset. Mapping to the spec's Phases A–F and to
ROADMAP milestones is explicit so three numbering schemes don't drift.

| Migration phase | Realizes spec | ROADMAP |
|---|---|---|
| **P0 — Seam + renderer unification** | (prelude to A) | M2/M3 |
| **P1 — Core data model behind the seam** | Phase A | M2 |
| **P2 — Analytic two-band detail + per-body cutover** | Phase B | M2/M3 |
| **P3 — Authoring** | Phase C | — |
| **P4 — Persistence** | Phase D | — |
| **P5 — Learned synthesis** | Phases E/F | — |

### P0 — Seam + renderer unification *(no generation change)*

**Status: first slice landed.** The seam exists, the game's geometric
surface is unified, and the single synthesiser now lives in the pure
crate. Remaining P0 cleanup is listed at the end.

What landed:

- The Query API seam lives in
  [`thalos_terrain::query`](../crates/terrain/src/query.rs): a
  `SurfaceQuery` trait (`sample` / `sample_height_m` / `radius_m` /
  `height_range_m` / `prewarm`), a `BakedSurface` implementation wrapping
  today's `PlanetSurface`, and free-function evaluators
  (`surface_sample` / `surface_height_m` / `surface_height_range_m`).
- The domain-warped ridged-HMF **detail cascade moved out of the Bevy
  crate** (`thalos_terrain_render::pipeline`) **into** `thalos_terrain`,
  so there is now exactly **one** surface synthesiser, in the crate the
  spec says the pipeline belongs to.
- Consumers re-pointed at the seam:
  [`PipelineTileProvider`](../crates/terrain_render/src/pipeline.rs)
  evaluates pixels via `surface_sample`; `rendered_height_m` /
  `rendered_height_range` are now thin wrappers over the seam, so the
  collider (`build_rendered_terrain_patch`), the character-controller
  height source, the camera boom, and debug readouts all converge with no
  call-site churn; `CpuPipelineHeightSource` now holds a `BakedSurface`
  trait object directly.
- Behaviour-preserving by construction: the cascade was relocated
  verbatim, so the rendered surface is bit-identical to before — this
  slice is a refactor that establishes the seam, not a visual change.

Two **deviations from the literal plan**, both defensible:

- **Per-direction seam, not a packed `generate_terrain_tile`.** Evaluating
  by `sample(dir, lod_m)` keeps `thalos_terrain` Bevy-free *and*
  cube-sphere-mapping-agnostic; the consumers already own the canonical
  UDLOD tiling mapping and pack their own attachments. The packed `Tile`
  type + `generate_terrain_tile` (4-channel splat + macro-albedo) arrive
  in P2 with the material-model change, when there is a consumer for them.
- **`query_features` / `query_scatter` not stubbed with throwaway types.**
  They are documented as backward-compatible default-method additions to
  `SurfaceQuery` for P1/P2 rather than defined now with types that would
  be redesigned. `prewarm` + `Region` exist today.
- **The impostor `bake` bridge was left as-is.** It already consumes the
  baked macro cubemap, which is exactly the low band the seam samples as
  its base — re-pointing it through the seam would be a wasteful no-op at
  the macro level. The impostor's *shader-side* high-frequency detail
  (statistical craters) is the one piece still divergent from the
  ground's non-geometric detail; reconciling it is deferred to P2 when
  fine detail formally becomes non-geometric (the caveat in §5).

Also landed: **`bake_dump` PNG dumps converged onto the seam.** The four
production dumps (albedo / height / roughness / normal) now call
`surface_sample` / `surface_normal` instead of the legacy three-layer
`sample.rs` sampler, so the offline previews show the surface the game's
ground LOD actually renders. `sample.rs` is now unwired from all consumers
(kept as the P2 crater-composition reference).

> **Finding surfaced by the convergence.** The seam (== the ground-LOD
> HMF cascade) synthesizes **no SSBO/statistical craters** — the ground LOD
> has never rendered them. Only the *impostor* shows mid/small craters, via
> shader synthesis mirrored from `sample.rs`. So the impostor (far) and the
> ground (near) currently disagree on fine features: craters vs HMF ridges.
> P0 unified the *geometric* surface and made this pre-existing divergence
> visible; closing it is **P2 feature composition** (craters become carve
> operators in the mid band, present in both views).

Remaining P0 cleanup:

- **Validate** via tile-edge determinism (bit-identical borders across
  LOD/order — UDLOD already requires this) and a visual `just game` /
  `just bake <body> --preview` pass. Note: equirect dumps are whole-planet
  views, so sub-km detail is sub-texel either way — the visible dump change
  is limited to km-scale SSBO craters (gone) on airless bodies like Mira.
- Optionally expose a legacy-sampler dump mode if crater-rich previews
  prove useful for generation iteration before P2 lands.

### P1 — Core data model behind the seam *(spec Phase A)*

**Status: complete.** Built in `thalos_terrain::pipeline`, foundation-only,
behind the seam — no editor, no rendering change; bodies still compile
through the old path. Validated by **37 sampling + determinism unit tests**
(the carve-out now documented in CLAUDE.md — fast, toy-field-bag tests, not
per-body bakes). All five increments landed:

- **Increment 1 — intent layer. ✅** A bag of named `Field`s
  (`pipeline::field`), each with a value `Expr`ession tree
  (`pipeline::expr`); an automatically derived evaluation DAG
  (`pipeline::dag` — topological order from expression references, with
  cycle / dangling-reference / duplicate-name rejection); and direct
  sampling of any field at a direction (`pipeline::planet::Planet`).
- **Increment 2 — stamps. ✅** The basic unit of authored/generator
  contribution (`pipeline::stamp`): geometry primitives (point, capsule,
  polyline, point-set) + scalars (radius/value, `Const`/`FromField`) +
  falloff + composition operator. Stamps fold onto a field's base value;
  `FromField` scalars form DAG edges. This is where `CompositionOp` starts
  doing real work. (Bezier geometry is the one deferred primitive.)
- **Increment 3 — generators + feature catalog. ✅** Declared feature types
  with kind + composition declaration (`pipeline::feature`); a
  deterministic density-gated `ScatterGenerator`; explicit (authored /
  promoted) instances; promotion/demotion via a per-generator exclusion
  index; region + kind queries. `query_features` wired onto `SurfaceQuery`
  (default empty on the baked backing).
- **Increment 4 — author overlay + two-path composition. ✅** Each field
  carries a separately-materialised `AuthorOverlay` (a replayable paint-op
  log) composed onto its procedural value via the field's operator,
  weighted by overlay coverage (spec §4–5). Verified independent of
  procedural where it fully covers.
- **Increment 5 — sparse quadtree storage + cache. ✅** Per-field
  cube-sphere quadtree (`pipeline::storage::FieldCache` over
  `pipeline::cubesphere`) that materialises the sampler lazily, collapses
  uniform regions to a single value, and caches tiles in RAM (L2 over
  L4-generation). Per-path procedural/overlay caches and disk (L3) are
  later refinements (L3 in P4).

### P2 — Analytic two-band detail + per-body cutover *(spec Phase B)*

**Current first slice: P2A — basic continental Thalos. ✅ Landed.** The
cutover started with Thalos intentionally simplified to an all-land,
single-biome `GenericTerrestrial` prototype. This gave the game a playable
end-to-end new-pipeline surface before tackling oceans, hydrology, scatter, or
the full terrain-feature compositor. Its ground LOD uses
`RuntimeTerrainDetail::BasicContinental`, which evaluates the same smooth
continental field used by the bake instead of layering the legacy P0 HMF
cascade on top. Runtime ground height is intentionally LOD-invariant for this
slice so parent/child tile handoffs do not produce contour-like relief steps.

**Current Thalos slice: P2A.5 — oceanic continents. ✅ Landed.** Thalos now
uses `BodyArchetype::OceanicTerrestrial` and
`RuntimeTerrainDetail::OceanicContinental`: a signed, ocean-bearing analytic
field that produces domain-warped continents, noisy coastlines, continental
shelves, seabed bathymetry, and deterministic archipelago bands. The bake and
runtime Query API evaluate the same field, so the terrain mesh and collider
continue to agree. `sea_level_m = 0` marks the water boundary, but water is not
a terrain material: underwater terrain keeps seabed materials (basalt,
sediment, shelf sand/carbonate), while ocean color/reflection/optical depth
come from the separate water renderers (`BodyWaterMaterial` at ground LOD and
the current impostor water branch as the far-LOD bridge). `just bake` writes
both the raw substrate (`albedo-equirect.png`) and, for ocean worlds, a visual
iteration aid (`orbit-color-equirect.png`) that composites the separate water
layer over that substrate without changing the baked terrain data.

**Iteration-speed slice: `FieldSurface` + continent intent cache. ✅ Landed.**
The oceanic route now bakes through `thalos_terrain::FieldSurface` — the
field-DAG-era live backing for the Query API seam. It owns the body's terrain
params + a **reusable continent-intent cache** (`build_continent_intent_cache`)
and produces both `StaticSurfaceData` (`bake`) and `SurfaceQuery` samples from
one source. `compile_oceanic_terrestrial` is a thin wrapper over
`FieldSurface::bake`. Motivation (measured): the Thalos bake is stage-free, and
~91% of its cost is the field eval, ~80% of *that* being the LOD-independent
12-shape continent kernel — re-run from scratch every bake. Splitting that into
`continent_kernel_base` (cacheable intent) + `continent_shape_from_base`
(LOD-gated coastline detail) and caching the base lets a re-bake after a
*non-shape* edit (palette, sea level, ocean fraction) reuse it and skip the
kernel: measured **1.48 s → 0.45 s (3.3×)** at 512² for cache reuse; a
continent-shape edit rebuilds the cache. The cache is materialised at output
resolution, so baked cubemaps stay bit-exact at texel centres — no new
impostor/ground divergence. Also landed alongside: `paint_surface_albedo`
parallelised (single-threaded → rayon, ~9×) and LOD band early-outs in the
field evaluator (skip noise eval when a band's LOD weight is 0). **Still
pending:** the editor holds a fresh config per edit, so it doesn't yet exploit
cross-edit reuse — wiring a persistent `FieldSurface` into the editor (plus
exposing shape/non-shape params) is the next step. The runtime/ground path keeps
the direct (uncached) kernel — exact and unchanged.

**Current next slice: P2B — Mira crater/feature compositor. In progress.**
The first compositor increment is wired into the Query API: unbaked crater
features from `StaticSurfaceData::craters` are range-queried through
`IcoBuckets`, deduplicated, sorted by crater array index (the cratering stage
stores oldest-first), and folded into runtime height before the legacy HMF
band. Craters at/above `cubemap_bake_threshold_m` stay skipped so the cubemap
and runtime compositor do not double-count. This makes Mira's sub-threshold
craters part of the same mesh/collider surface instead of impostor-only SSBO
detail. Remaining P2B work is to promote this from crater-specific additive
composition into the full `TerrainModification` compositor (carve / flood /
preserve-erode / replace), then proceed to Vaelen, full Thalos, and Pelagos.

- Implement the **two-band heightfield**: low band = intent output
  fields; mid band = analytic character conditioned on intent;
  **fine detail reclassified as non-geometric** (shader POM/normal/
  scatter). Implement ordered **feature composition** (additive / carve /
  flood / preserve-erode / replace), age-ordered with deterministic
  tiebreak, range-queried by influence radius. **Partial:** Mira craters now
  compose through the Query API as runtime geometric features.
- Land the **scatter** stream (fills the P0 stub) and a minimal scatter
  renderer in `terrain_render`.

  > **Scope + pattern note (fernweh).** Scatter is *only*
  > `SurfaceInstance` features — discrete objects resting **on** the
  > surface (boulders, vegetation, debris, structures). Terrain-shaping
  > features (craters, volcanoes, rifts, mesas) are `TerrainModification`
  > and belong in the band-limited heightfield via the feature compositor
  > in the bullet above — they are **never** spawned as entities (spawning
  > them would duplicate geometry and break the one-surface mesh=collider
  > invariant). Within scatter, two paths:
  >
  > - **Dense** (grass, pebbles) → GPU compute instancing, not ECS
  >   entities (spec §11). The pattern below does *not* apply.
  > - **Sparse, large, collidable** (boulders with colliders, structures)
  >   → ECS entities. For these, and for per-instance collider windowing
  >   (spec §9), lift the windowed-reconciliation pattern from
  >   [fernweh](https://codeberg.org/glocq/fernweh) — the *pattern*, not
  >   the dependency (it's Bevy-coupled, on a 0.19-rc vs our 0.18, and
  >   ~110 lines; reimplement on our terms). Each frame: diff the desired
  >   region set against (pending tasks ∪ spawned entities) → spawn
  >   missing, despawn extra; **cancel an in-flight task by dropping it**
  >   when its region leaves the window; hold pending tasks in a `Local`
  >   map and `poll_once` them; the async task produces a spawnable
  >   description and the main thread instantiates it; tag each spawned
  >   root with its id.
  >
  > Adapt, don't copy the toy: our neighbourhoods are cube-sphere caps /
  > resident-tile regions (not a flat grid); add **hysteresis** (outer
  > despawn radius > inner spawn radius) so camera jitter at a boundary
  > doesn't thrash; add scatter LOD (distance fade / instance-count); and
  > watch the O(N)-per-frame full-set rebuild at high instance counts.
  > Their "superchunk = enum variant" trick applies only to a *single
  > entity* large enough to span scatter tiles (a multi-tile structure),
  > never to terrain features. Runevision's **LayerProcGen** docs are a
  > good read on layered generation before building this.

- **Cut bodies over one at a time** behind the Query API. Revised order:
  basic Thalos vertical slice → Mira → Vaelen → full Thalos → Pelagos.
  Each cutover retires that body's old archetype stage and folds
  terrain.md's archetype/v2-backlog work (hydrology, layered material
  columns, climate fields) in as fields + feature types + synthesisers.
  Old and new pipelines coexist until the last body migrates.
- **Band-limiting now holds** ⇒ colliders are cleanly walkable by
  construction, and the impostor's shader high-freq detail reconciles
  against the ground's non-geometric detail (closes the P0 caveat).

#### P2C — Unified surface material model (colour + BRDF). In progress.

The orbital impostor, the UDLOD ground LOD, and the editor preview must show the
**same** surface coloration, material, and shading. Today they don't: a single
point on Thalos is coloured **four** different ways —

1. `surface_color.rs::paint_surface_albedo` grades biome palettes into the baked
   albedo cubemap (impostor source);
2. `generic_terrestrial_field.rs::sample_oceanic_continental` carries the field's
   own `biome_mix` + `material_id`;
3. `terrain_render::pipeline::material_masks` re-derives grass/soil/rock/wet from
   slope/curvature and packs the tile `material` attachment;
4. `body_terrain.wgsl::eval_material_stack` re-derives forest/grass/snow from
   altitude/slope and largely discards the sampled albedo.

The BRDF is forked too: the impostor calls the shared
`thalos_planet_lighting::shade_hapke_surface`, while the ground LOD runs a
hand-rolled "P2A temporary" diffuse+sky-fill path — violating the "one shading
base" invariant (§8).

**Target.** *One* per-direction surface evaluator in `thalos_terrain` defines
albedo + roughness + a material identity (splat weights over a small per-body
**material palette**, each entry tagged with a **shading model** — Hapke for
airless regolith, rough-dielectric for soil/grass/rock, Fresnel-specular for
ice/glass, metallic). Every consumer materialises or samples that one evaluator;
no renderer invents colour. *One* shading dispatch in `thalos_planet_lighting`
switches BRDF by shading model; every shader calls it. (The new shading-model
tag is distinct from `feature_compiler::SurfaceMaterialClass`, which is
geological provenance, not a BRDF.)

**Materials-first (the load-bearing inversion).** Materials are the substrate —
each a `SurfaceMaterial { shading, base_albedo, roughness, metalness }` in a
body-wide library. The resolved hierarchy:

1. **Climate → biome mix** (weighted; biomes blend at boundaries).
2. **Biome → its ground-cover material set** (+ colour-grade params). *This is
   the sense in which biomes "define the palette":* a desert biome's ground cover
   is {sand, desert pavement, dry soil}, a forest biome's is {forest soil,
   broadleaf vegetation, mossy rock}, tundra's is {tundra soil, lichen,
   permafrost}.
3. **Form → selects.** A small set of **universal, cross-cutting materials**
   (exposed rock on steep/convex faces, snow on cold high gentle ground, coastal
   sand, basalt/sediment below sea level) are selected by pure form regardless of
   biome — the "mostly" caveat. Within a biome, form also picks among its
   ground-cover materials (vegetation vs bare soil by moisture/flatness).
4. **Blend + grade.** Fold each biome's form-selected ground cover with the
   universal form materials, blend by biome weight, apply the biome's artistic
   tint → final splat + albedo + roughness + dominant shading model.

So the **biome both selects the candidate materials and tints them**; form
selects which of those appears at each point. The biome is *not* a direct colour
painter (today's `BiomeColorPalette` conflates per-biome identity *and*
slope/height grading — that conflation is what this split removes). Both
selection stages live in the **one evaluator**, fed by a context the bake builds
from neighbour cubemap samples and the runtime builds analytically, so selection
is identical in both. The field stops pre-deciding final colour; it supplies
height plus cheap climate signals. Airless bodies are the degenerate case: one
"biome" whose palette is Hapke regolith materials, so their look is preserved by
the same machinery.

**Biomes are the regional unit (the bigger frame).** A biome is the top of the
dependency chain, classified from **macro climate** — latitude, continentalness,
macro-elevation intent, temperature, moisture — *independent of local
micro-relief*, so it can drive what is generated rather than being read back from
it. A biome owns **three facets**:

1. **Terrain generation** — the height-synthesis recipe for the region (rolling
   plains vs montane ridges vs dune sea). Biome decides what shape generates.
2. **Material palette** — which materials the ground cover draws from, the
   form-selection rules, and the colour tint.
3. **Scatter** — which objects spawn: trees, grass, pebbles, debris.

Form (slope / curvature / micro-altitude) selects *within* the biome's palette;
the universal materials (rock, snow, sand, ice, seabed) cross-cut every biome —
so **snow and rock are materials, not biomes**. These three facets are
**sequenced, not built at once**: facet 2 is this P2C slice; facet 1 is the
field-DAG terrain-gen work (P2); facet 3 is the scatter stream (P2/P3 §6). To
keep them one coherent unit instead of parallel systems, the material palette is
modelled as **owned by a `BiomeDef`** now, with reserved slots for the
terrain-gen recipe and the scatter set so facets 1 and 3 attach to the same
structure later:

```text
BiomeDef {
    ground_cover: Vec<BiomeMaterialRule>,  // facet 2 — library materials + form selector
    tint: BiomeGrade,                      // facet 2 — artistic colour grade
    // terrain_gen: TerrainRecipe,         // facet 1 — reserved (P2)
    // scatter: Vec<ScatterClass>,         // facet 3 — reserved (P2/P3)
}
```

Evaluator (facet 2): `climate → biome mix → each biome form-selects its
ground_cover (+ universal materials) → tint → blend`.

**Decisions (locked).** CPU evaluator is the single source of truth, GPU samples
(avoids the bit-exact CPU↔WGSL mirror trap); material = splat weights over a
per-body palette, each entry carrying a shading-model tag; the editor keeps its
separate impostor and its heightfield tile preview is for iterating on a local
slice (it does **not** need to be UDLOD — a displaced grid sampling the Query API
is enough).

**Increment ladder.**

- **A — Single painter core.** Extract `paint_surface_albedo`'s per-texel logic
  into a pure `surface_color_at(…) -> [f32; 3]` the bake loop calls per texel.
  Behaviour-preserving: the bake stays bit-identical. (Landed.)
- **B — Route the Query API through it.** The oceanic `SurfaceQuery::sample`
  returns the graded albedo + material from the painter core (not the flat
  material colour); the tile provider evaluates it per-direction at tile
  resolution and bakes albedo + roughness + material-splat into the tile
  attachments. Retires `pipeline::material_masks` (#3) — the field's `biome_mix`
  is the intent. Impostor, ground, and collider then read identical colour.
- **C — Single shading dispatch.** Add `shade_surface(shading_model, …)` to
  `lighting.wgsl`; reroute `planet_impostor.wgsl` and `body_terrain.wgsl` through
  it; delete `eval_material_stack` + the temporary ground lighting (#4).
- **D — Shared non-geometric detail.** Lift the albedo breakup + detail-normal
  into one WGSL lib both shaders import, closing the P0/§5 impostor-vs-ground
  detail caveat.
- **E — Editor heightfield preview.** Replace the UDLOD tile viewer in
  `planet_editor` with a displaced-grid patch sampled from the Query API and
  shaded with the same material shader.

The altitude/slope/snow banding that lives only in `eval_material_stack` today
moves CPU-side into the evaluator (the field's `biome_mix` already encodes most
of it), so both near and far views inherit it.

### P3 — Authoring *(spec Phase C)*

- Editor for fields, masks, and features in `planet_editor`: brush tools,
  promotion/demotion, seed shuffling, time-varying feature trajectories.
  Replayable overlay op-log is the author's source of truth.

### P4 — Persistence *(spec Phase D)*

- Per-tile **L3 disk frecency cache** (the current whole-planet bake
  becomes the planet-load intent bake), planet serialisation (RON +
  binary bundles), edit history with undo/redo, a pipeline-version hash
  in the cache key for global invalidation on detail-stage changes.

### P5 — Learned synthesis *(spec Phases E/F, aspirational)*

- Replace the mid-band synthesiser per-feature, then the base, behind the
  unchanged unified-synthesiser contract. Per-planet; independent timeline.

## 7. Decisions to lock before/while building

These are the choices that will bite if deferred; several are in the
spec's own Open Questions (§13).

1. **One cube-sphere mapping. — RESOLVED (P0).** The seam evaluates **by
   direction** (`sample(dir, lod_m)`), so `thalos_terrain` is
   mapping-agnostic: the canonical tiling mapping stays UDLOD's
   (`Coordinate::world_position`), owned by the consumer, and `terrain`'s
   own `Cubemap` / `dir_to_face_uv` is demoted to an internal storage
   detail of the baked backing. Tile-border determinism remains the
   consumer's responsibility (UDLOD's `stitched_pixel_coordinate` already
   guarantees shared directions).
2. **Material model change.** Today: baked albedo + single material-ID
   cubemap + palette. Spec: 4-channel splat weights + per-material detail
   textures sampled in-shader + macro-albedo modulation. This is an
   art-pipeline change (per-material texture sets), not just code. Splat
   from one-hot material ID is the P0 bridge; real authored materials are
   P2.
3. **Mid-band cutoff frequency.** The single shared boundary where height
   stops being geometry. Must be expressible per-LOD and identical for
   mesh and collider. Today it's implicit in tile Nyquist; make it
   explicit in P2.
4. **Water is not a terrain material. — RESOLVED for Thalos P2A.5.** Sea
   level is a renderer/physics boundary over signed terrain. Underwater
   terrain remains ordinary terrain data with seabed material, roughness, and
   collision height; it is not painted blue or replaced by an "ocean" material.
   Water color, reflection, waves, and optical absorption belong only to ocean
   renderers. The far impostor still carries an inline water branch as a
   bridge, but the baked albedo below sea level must remain seabed albedo so a
   later separate orbital-ocean material can consume the same height/albedo
   substrate without repainting the terrain.
5. **CPU vs GPU synthesis, and the Bevy boundary.** Spec §11 wants dense
   scatter generators on the GPU, but `terrain` must stay Bevy-free
   (CI-guarded crate-boundary invariant). Resolution: `terrain` emits
   deterministic **generator descriptors**; the **GPU dispatch lives on
   the consumer side** (`terrain_render`), with a CPU reference path in
   `terrain` for determinism tests. Lock this before scatter (P2).
6. **Quadtree leaf size vs UDLOD tile size.** Spec §5 default 64 px,
   "tuned against the renderer's tile structure." Align so leaf tiles
   upload without resampling. Tune empirically in P1/P2.
7. **How much of the feature-compiler vocabulary survives.** terrain.md's
   `TerrainPrior`/`FeatureManifest`/archetypes map onto the spec as:
   archetype = a preset bundle of fields + feature-type defaults; prior =
   generators reading physical params; manifest = the feature catalog +
   instance index. Confirm this mapping in P1 so terrain.md content is
   ported, not reinvented.

## 8. Invariants the migration must not break

- **The game loads pre-baked terrain only and never compiles.** Every
  phase keeps a working bake path (`just bake`, `bake_check` auto-repair).
- **One surface for mesh + collider** (true today via `rendered_height_m`;
  preserve it through the seam, strengthen it with band-limiting in P2).
- **`thalos_terrain` stays Bevy-free** — CI `cargo tree` guard. GPU work
  is consumer-side (§7.4).
- **Tile determinism** — bit-identical borders across LOD and request
  order; pipeline-version hash in the cache key.
- **Per-body cutover** — old and new pipelines coexist behind the Query
  API until the last body is migrated. No flag day.
- **One shading base** — all consumers continue to route through
  `thalos_planet_lighting::shade_hapke_surface`; the migration changes the
  *data source*, never forks the BRDF.

---

*Doc owner: Korbin. Companion to the pipeline spec; serves ROADMAP M2/M3.*
