# Terrain

The consumer-side contract for Thalos terrain: the **tile** the rest of the
engine reads, and how it is rendered, shadowed, collided against, and varied
over time. Terrain **generation is a black box** behind this contract — it
produces tiles; nothing in this doc depends on *how*.

> **Generation is archived (2026-06).** The previous generation design (feature
> compiler, field-DAG pipeline spec, migration plan, research surveys) was moved
> under [`archive/`](archive/) and reframed as a black box. A new generator is
> being built against the tile contract below. This doc is the *consumer* side:
> tile contract → rendering → shadows → colliders → dynamic features.
> Atmospheric optics, ocean, clouds, reflections, and lighting/GI live in
> [atmosphere.md](atmosphere.md).

## Overview

Everything the engine does with terrain reads one primitive: the **tile**. A
tile is a quadtree node on a cube-sphere (cube face + LOD + x/y) carrying a
fixed set of attachment textures (height, albedo, normal, roughness, material).
Generation produces tiles; the renderer draws them; the collider, the relief
shadows, and the future RT reflection geometry all derive from the *same* tile
height authority.

- **Generation** — a black box (`thalos_terrain`, pure Rust, no Bevy). Input: a
  body's terrain config. Output: `PlanetSurface` (immutable `StaticSurfaceData`
  + `DynamicSurfaceLayers`), sampled into tiles on demand. The new generator is
  built fresh against the tile contract; its internals are not specified here.
- **Rendering** — the in-tree `thalos_udlod` fork: UDLOD on an ellipsoidal
  cube-sphere, fed tiles through the runtime `TileProvider` seam, shaded with
  the custom Hapke-family BRDF (`thalos::lighting`), not Bevy PBR. See
  *Rendering: ground LOD*.
- **Consumers** — colliders (Avian trimesh from tile height), relief shadows
  (baked horizon attachment), dynamic features (static substrate + cheap
  dynamic overprints), and eventually RT reflection geometry (a BLAS extracted
  from the same tiles).

## The tile contract

The black-box boundary: what generation must produce and what every consumer
may rely on.

**Address.** A `TileCoordinate` is a cube-sphere quadtree node: `side` (0–5 cube
face), `lod` (0 = coarsest, `2^lod` tiles per face edge), `x`/`y`. Tiles have
`parent()`, four `children()`, eight `neighbours()` (cross-face aware), and a
deterministic pixel→world mapping (`stitched_pixel_coordinate →
Coordinate::world_position`). This address space is good as-is and is kept
verbatim from the udlod fork.

**Payload.** Per configured attachment, an N×N texture (power-of-two) in the
attachment's format, including a border ring that overlaps neighbours:

| Attachment | Format | Purpose |
|---|---|---|
| `height` | packed RG16 (R16 / R32Float also supported) | Required. Normalized 0..1 → `[min_height, max_height]`. RG16 packs coarse + residual to avoid R16 contouring on shallow slopes. |
| `normal` | RG16 | Optional; may be derived from height. |
| `albedo` | RGBA8 | Optional surface color. |
| `roughness` | R16 / R8 | Optional; per-fragment BRDF input. |
| `material` / `splat` | RGBA8 | Optional weight masks for shader-side blending. |
| *(planned)* `horizon` | packed | Baked relief-shadow horizon angles — see *Surface shadows*. |

**Determinism = pure function of position.** `request_tile(coord)` and
`request_tile(child_of_coord)` must agree bit-exactly on shared edges,
regardless of evaluation order. This holds **iff synthesis is a pure function of
(ellipsoid position, body params)** — and Thalos deliberately keeps it so:
there are **no neighbour-aware tile passes**. Erosion is applied **analytically
via `bevy_erosion_filter`** as a field/heightfield transform, not as a per-tile
hydraulic pass that would require border exchange between generated neighbours.
Border coherence therefore falls out of position-purity for free; the tile
payload never needs cross-tile structure.

**One height authority.** The tile's height is the single source of truth for
**rendered geometry, the physics collider, and any future RT BLAS**. They must
read the same height — today LOD-invariant, so parent/child handoff has no step
— or LOD seams crack visually, in collision, and in reflections. Do not add a
second height path for any consumer. The collider-patch trimesh extraction (see
*M5 rendered-height terrain colliders*) is also the seed of the future RT
reflection BLAS.

**Production backend is orthogonal.** The atlas cares only that a coordinate's
slot finished; bytes may come from CPU synthesis, an in-memory cache
(`MemoryTileCacheProvider`), or (intended) a GPU job that writes the atlas slot
directly. See *TileProvider interface* under Rendering for the API.

## Generation (black box)

Generation produces tiles conforming to *The tile contract*. Its internals —
archetypes, fields, feature placement — are being rebuilt and are intentionally
**not specified here**. The superseded design is in [`archive/`](archive/)
(`terrain-generation-legacy.md`, the pipeline spec + migration, and the `gen/`
research surveys) and is reference only.

The current authored data surface is `PlanetSurface` = immutable
`StaticSurfaceData` (cached, expensive) + `DynamicSurfaceLayers` (cheap
overprints; see *Dynamic features*). The one hard rule generation owes its
consumers: **the static substrate is the expensive, cached layer; anything
time-varying is a cheap additive overprint that must not invalidate the static
cache.**
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
- Stand up a `thalos_body_render` crate that owns the integration: registers
  `thalos_udlod::TerrainPlugin`, exposes a deterministic
  `SyntheticTileProvider` (pure function of `Coordinate::world_position`
  so tile borders are bit-identical), and ships a `playground` example
  binary that drives the fork end-to-end against the synthetic provider
  on a Mira-scale sphere.
- **Exit criterion:** `cargo run -p thalos_body_render --example playground`
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
  custom UDLOD terrain receiver. Ships project per-part silhouettes from
  the same procedural part geometry as the visible mesh, using two caster
  primitives: tapered capsule segments for bodies of revolution (stacked
  pods/tanks/adapters/engines/intakes as their frustum/cylinder profiles,
  a fuselage loft as nose/barrel/tail segments sampled from its skin
  model, wing-pylon jet nacelles along their pod axes via
  `jet_nacelle_centers`), and *thin planform quads* for lifting surfaces
  (root/tip leading/trailing-edge corners from `wing_panel_frame`) — a
  wing modelled as a capsule reads chord-thick from the side and throws a
  huge slab at low sun, while the quad projects the true trapezoid at any
  sun angle and vanishes edge-on. Penumbra on both primitives widens with
  caster height at the star's angular diameter, so contact shadows are
  crisp and overflight shadows soften. EVA keeps a small capsule proxy.
  Leave `THALOS_TERRAIN_CRAFT_SHADOW`
  unset or set it to `on|auto` for the normal behavior, and set it to
  `off` only when isolating material seams.
  Bevy's cascade shadow maps are the *structure/hull* counterpart: the
  craft casts CSM onto `StandardMaterial` receivers (runway slab, future
  structures) and itself. The sun `DirectionalLight` must keep
  `SHIP_LAYER` in its `RenderLayers` — `view::propagate_view_render_layers`
  stamps the whole craft subtree onto `SHIP_LAYER`, and a light only
  renders shadow casters whose layers intersect its own (the bug where
  craft CSM shadows silently never existed). If the real body's height range
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
- **Implementation:** [crates/body_render/src/ground/pipeline.rs](../crates/body_render/src/ground/pipeline.rs).
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
  [crates/body_render/src/shading/shaders/lighting.wgsl](../crates/body_render/src/shading/shaders/lighting.wgsl),
  [crates/body_render/src/ground/body_terrain.wgsl](../crates/body_render/src/ground/body_terrain.wgsl),
  [crates/game/src/rendering/ground_terrain.rs](../crates/game/src/rendering/ground_terrain.rs).
- **Exit criterion (met):** terrain ground-LOD pixels go through
  Hapke + eclipse + planetshine + ambient via the shared helper, with
  per-fragment roughness sampled from the third tile attachment.

### M5 rendered-height terrain colliders

The first landing slice exposes rendered-height helpers from
`thalos_body_render`:

```rust
rendered_height_m(surface: &StaticSurfaceData, dir: Vec3) -> f32
build_rendered_terrain_patch(surface, body_radius_m, center_dir, basis, config)
    -> TerrainPatchMesh
```

These helpers decode the same R16 cubemap interpretation used by
`PipelineTileProvider`: `real_meters = (texel / 65535 * 2 - 1) *
height_range`. Local physics builds one patch around the active craft
from this data and converts the mesh into an Avian static trimesh.
Terrain colliders attach, stay attached, and refresh only in the 1x-only
surface warp zone (`WarpLimits` caps the ladder at 1x), except for an
already-contacting patch that is finishing the landed collapse. Manually
switching to 1x higher in the descent keeps the collider absent. Current
tangent-grid fallback defaults are 4096 m half extent, 65 x 65 vertices,
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
of `body_render::ground::pipeline`, which now delegates to it) — adds
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

## Surface shadows

Good lighting and shadows are a priority. At planet scale "shadows" is four
distinct problems, and Bevy's cascaded shadow maps (CSM) are the wrong primary
tool for the dominant one — cascades are sized to the camera frustum, the custom
UDLOD pipeline neither casts into nor receives from them for free, and shadow
maps swim under the floating origin. The model is therefore **layered, each tool
on the problem it actually fits**, all composing as a single gate on the
direct-sun term before it enters the Hapke BRDF:

```
direct_sun *= horizon_visibility * cloud_transmittance * ssao
```

This *macroscopic* visibility gate is distinct from Hapke's internal shadowing
(the grain-scale opposition surge), which stays inside the BRDF. Ambient +
planetshine + sky-diffuse fill the shadowed regions; airless bodies keep the
vacuum-black floor (with a small ambient floor for playability).

1. **Terrain self-shadow (relief)** — ridges into valleys, crater rims, long
   grazing-sun shadows: the dominant surface look, and *not* a shadow-map
   problem. Terrain is a heightfield, so self-shadowing is a horizon query.
   Target: a **baked horizon-angle map as another generated tile attachment**
   (per texel, the horizon elevation in K azimuth bins) — body-fixed,
   scale-free, floating-origin-immune, bakeable, and resolution-matched because
   it is sampled from the same tiles. Sampling it also yields terrain→object
   shadowing (an object queries the horizon at its ground point). A runtime
   height-atlas raymarch is the fast prototype to validate the look before
   committing the attachment.
2. **Object → terrain (and object → object)** — ship/EVA onto the ground: a
   tight, **world-stable near-field shadow map** scoped to ~hundreds of metres
   around the player (texel-snapped in a body-fixed frame so it doesn't swim).
   Standard meshes cast; terrain receives by sampling it in its fragment shader
   (the stock `MeshPipelineViewLayout` is already bound at group 0). This
   **retires the analytic capsule craft-shadow proxy** with something that
   handles arbitrary silhouettes.
3. **Volumetric cloud shadows** — a dynamic, low-res sun-projected cloud
   transmittance term (the `cloud_transmittance` factor above); it also feeds
   god rays through the `BodySky` atmosphere pass (depth-coupled via
   `SceneDepthImage`). Owned by the atmosphere/cloud system — see
   [atmosphere.md](atmosphere.md).
4. **Contact / crevice** — SSAO: screen-space, pipeline-agnostic, covers terrain
   and objects uniformly with no custom plumbing.
5. **Macro day/night + eclipse** — already in `SceneLighting` (terminator N·L,
   `eclipse_factor`); not a shadow-map concern.

The current shipping state is layer 2's analytic proxy plus layer 5; the baked
horizon attachment (layer 1) is the next high-value visual win and the reason
`horizon` is reserved in *The tile contract*. Reflective-ship reflections and
ray-traced GI are covered in [atmosphere.md](atmosphere.md); both depend on the
same single height authority (a future RT BLAS is the collider-patch trimesh
extended).

## Dynamic features

Time-varying surface features — seasons, shifting dunes, tides — are placed by
asking **what is the lowest layer that must observe the change**:

- **Height-bearing dynamics** (shifting dunes, snow depth) → **generation side,
  as a cheap dynamic tile-output layer over the cached static substrate**, so
  physics colliders and relief shadows inherit them automatically via the shared
  height authority: `height = static(coord) + dynamic_layer(coord, state)`. The
  dynamic term must be cheap and must never re-run the expensive static field.
- **Appearance-only dynamics** (seasonal color, frost sheen) → **render side**, a
  shader modulation off the dynamic-state uniform. No tile regeneration.
- **Tides** → **water-renderer parameter** (sea level over time) plus a
  render-side intertidal wet/dry albedo band on terrain (current tide vs static
  terrain height). Per the "water is not terrain" invariant, tides are **never**
  baked into a terrain tile; the seabed is static.

**Invariant: the static/dynamic boundary must coincide with the expensive/cheap
boundary.** If a "dynamic" feature forces re-running the expensive continent
field, it is misclassified.

**Existing scaffolding.** `StaticSurfaceData` (cached) / `DynamicSurfaceLayers`
(`ice_caps`, `active_dunes`) / `DynamicSurfaceState` (mutable per-body state,
owned by `SolarSystemState`). `dynamic_state` already flows into
`compute_tile_pixels`, so dunes/ice are evaluated per-tile-pixel over the static
substrate today.

**Gap.** `dynamic_state` is currently a snapshot captured when the provider is
built; advancing it (season progressing, dunes migrating) does not yet
invalidate and re-request affected tiles. The missing seam is **scoped,
rate-limited tile invalidation**: only resident tiles intersecting a changed
layer's spatial window regenerate (dune `DuneSea` region, polar latitude),
rate-limited and time-warp-aware, regenerating that tile's height + normal +
albedo (+ horizon) attachments together. `HeightSource::revision()` already
exists to keep the collider side in sync; GPU tile production is the natural
home for the cheap dynamic re-pass.

---

## Vegetation decoration layer (grass blades)

Near-camera grass blades on vegetated bodies (Thalos), shipped 2026-06 as a
**self-contained decoration layer on the consumer side of the tile contract** —
it reads only the runtime seams (`HeightSource`, the material-mask gate,
`sea_level_m`), so the coming generation revamp can replace everything behind
those seams without touching it.

**Shape.** A camera-local set of body-fixed *grass tiles* (~25 m tangent
squares on a self-contained cube-sphere lattice, `GRASS_TILE_SIZE_M`). Each
tile is one batched `Mesh` of up to a few thousand tapered five-vertex blade
strips, built on `AsyncComputeTaskPool` and anchored exactly like the runway:
a root-grid big_space child re-posed in f64 every frame, vertices stored as
small offsets from the tile's surface centre. No GPU instancing in v1.

**Placement = the terrain shader's own grass gate.** Blades sample the body's
registered `HeightSource` (GPU-atlas mirror with CPU fallback — the collider's
source, so blades sit on the rendered ground) and accept by the same
slope/curvature mask math the tile baker writes into the material attachment's
grass channel (`material_masks_from_heights`), plus: above `sea_level_m + 1 m`,
slope ≤ 0.45, fading out over 2 400–3 100 m altitude (the shader's
grass→alpine band), and excluded where a `TerrainFlatten` pad (the runway) has
weight. Blade tints mirror the shader's `C_FOREST`/`C_GRASS`/`C_DRYGRASS`
palette by altitude.

**Shading.** `GrassMaterial` (`ground/grass.wgsl`) mirrors the vegetated
terrain path's lighting constants (`DIRECT_SUN_STRENGTH`, day/night sky fill)
with a wrap-diffuse against the sun direction; blades carry the *terrain*
normal so they light like the ground they grow from. Wind sway is a per-blade
phase-shifted vertex displacement; the distance fade (70→100 m) is a
screen-space-dithered discard in the opaque pass. Ship-layer only, so the map
view never sees it.

**Staleness.** Tiles snapshot `HeightSource::revision()` at build; a periodic
scan re-samples stale tiles' centre height and rebuilds only the ones whose
ground actually moved (> 5 cm) — this re-seats grass after tile streaming and
removes it when a flatten pad is installed late.

**Code.** Engine side: `crates/body_render/src/ground/vegetation.rs` +
`grass.wgsl` (lattice, builder, material). Driver:
`crates/game/src/rendering/grass.rs` (`GrassRenderPlugin`: active-body pick,
tile lifecycle, f64 anchoring, wind/sun updates, rebuild scan).

**Deferred.** GPU instancing / compute placement; density falloff rings;
albedo-sampled per-blade tint; eclipse occluders + received shadows; blade
bending under the player/downwash; exact cube-face-seam coverage (a small
grass gap at face seams is accepted).

---

## Open questions

Consumer-side open questions. (Generation's own open questions live with the new
generator, not here.)

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
7. **Shadow casting from terrain.** *(Resolved — see Surface shadows.)*
   Relief self-shadowing is a baked horizon-angle tile attachment, not
   terrain-as-CSM-caster; object shadows use a tight, world-stable near-field
   shadow map. CSM is rejected as the primary planet-scale tool. Remaining
   detail: the horizon attachment's bin count and soft-penumbra encoding.

---

## References

- [atmosphere.md](atmosphere.md) — atmospheric optics, ocean rendering,
  clouds, reflections (brushed-stainless → mirror), and lighting/GI
  (bevy_solari). The other half of the surface look.
- [simulation.md](simulation.md) — big_space hierarchy, save/load, body
  state providers.
- [lore/solar_system.md](lore/solar_system.md) — the bodies terrain must
  handle.
- [archive/](archive/) — **superseded** terrain-generation design
  (legacy feature compiler, pipeline spec + migration, `gen/` research
  surveys and aesthetic targets). Reference only; the new generator is
  built against *The tile contract* above.

---

*Doc owner: Korbin. Roadmap milestones served: M2 (generation, now a black
box), M3 (ground LOD rendering).*
