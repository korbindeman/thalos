# Architecture: crate layout & system boundaries

This is the **target** crate architecture and the in-progress migration toward
it. It exists because the crate split had drifted: the "pure, foundational"
physics crate had grown to depend on terrain and atmosphere, world-definition
data had no single home, and the planet-rendering concern was spread across
three crates that must stay byte-for-byte consistent. Per the project rule
(CLAUDE.md): infrastructure changes are announced here before they land.

## Three organizing principles

1. **One authored source of truth for the world.** Every physical parameter of
   the system and its bodies lives in one pure crate, `thalos_world`, consumed
   *downward* by physics, terrain generation, and rendering. Nobody else parses
   the body RON or owns body parameters.

2. **One celestial-body appearance model, multiple render backends.** The
   surface BRDF + atmosphere + cloud + water *math* is defined once. The
   impostor billboard (distant) and the udlod ground mesh (close) are two
   consumers of it — a backend chooses geometry, never its own shading.

3. **One game runtime, multiple application shells.** Interactive play and
   headless capture compose the same plugin graph through `thalos_runtime`.
   `thalos_game` and `thalos_capture_host` are thin launchers; neither owns a
   parallel world or renderer. Capture is a first-class surface for stills,
   comparisons, deterministic video, and agent verification (see `capture.md`).

### The appearance invariant

> A body's look = one authored dataset (`thalos_world`) + one dynamic
> environment state (`SolarSystemState`, single-writer in `thalos_runtime`) + one
> shading library (`thalos_body_render::shading`). A render backend may choose
> its geometry (billboard vs LOD mesh) but never defines its own lighting,
> atmosphere, cloud, or water math.

This is what keeps a planet identical across the impostor↔terrain LOD swap, for
lighting, atmosphere, weather/clouds, and water alike. It is already true for
the Hapke BRDF (`shade_hapke_surface`) and cloud-band state; the crate merge
removes the seams that let a backend drift.

### The type-partition rule (prevents dependency cycles)

> **Authored data types → `thalos_world`. Algorithms and runtime state →
> `thalos_physics_canonical`.** `physics` depends on `world`, never the reverse.

So `BodyDefinition`, `OrbitalElements`, `StateVector`, and the subsystem config
aggregates live *down* in `world`; `Simulation`, the propagators,
`TrajectorySample`, `BodyState`, `CraftState` stay *up* in `physics`. As long
as `world` never references a physics runtime type, the dependency points one
way.

## Target workspace

The hierarchy separates the player-facing application, reusable libraries,
developer/offline executables, authored/runtime assets, and generated evidence.
Within `crates/`, folders express responsibility and intended dependency
direction; ordinary feature size is handled with Rust modules rather than a
crate per feature.

```text
apps/
  game/                         # bin: thalos_game

crates/
  vendor/
    big_space/                  # foundational, upstream-derived dependency
  domain/
    world/                      # authored body/system truth
    celestial/                  # physical far-field sky model
    terrain/                    # SurfaceQuery + terrain generation
      erosion_filter/           # author-owned temporary implementation
    construction/               # current shipyard package/model
  simulation/
    physics_canonical/
    physics_local/
    control/
  rendering/
    render/                     # current body_render; state-in/pixels-out
    udlod/                      # Thalos-owned terrain-render backend
  interface/
    input/
    ui/
  runtime/
    game/                       # lib: thalos_runtime, sole app composition
  capture/
    protocol/                   # Serde-only request/result types
    runtime/                    # Bevy capture state machine/readback
  offline/
    texgen/                     # compiler source, no runtime edge
    terrain_learned/

tools/
  capture/                      # CLI: shot/compare/record/verify/status/stop
  capture_host/                 # headless shell over the real game runtime
  terrain_baker/
  terrain_train/
  texgen/
  world_map/

assets/
  generated/
  terrain_packages/

artifacts/
  visual/latest/
  visual/runs/
  video/
  diagnostics/
```

Visual verification scenes belong to `thalos_runtime` and render through the
canonical capture host. The former object/UI previews remain temporary crate
examples until equivalent in-context runtime presets land; there is no parallel
`labs/` application layer. Exploratory browser tooling such as Terrain Lab stays
under `tools/` and is not an acceptance renderer.

`apps/` is intentionally narrow: it contains software shipped to and launched
directly by the player. Headless capture is a first-class product capability,
but its host process is developer/automation infrastructure and therefore lives
under `tools/` alongside its controller.

The target dependency direction is:

```text
apps/game → thalos_runtime → domain + simulation + rendering + interface
tools/capture_host → thalos_runtime + capture_runtime → capture_protocol
capture CLI → capture_protocol
rendering → construction
physics_local + rendering → terrain height/query contract
runtime → renderer-specific GPU height-mirror registry
```

The construction/rendering edge now points correctly: rendering consumes the
construction model and owns the material projection. The terrain seam is also
repaired: `thalos_terrain` owns `HeightSource` plus the shared patch contract;
local physics consumes that contract without a rendering dependency; runtime
composition separately tracks the renderer's GPU-atlas mirrors. `thalos_texgen`
is offline and rendering loads its baked assets.

See ADR-20260721T194628Z-role-based-agent-first-workspace and
ADR-20260721T194629Z-first-class-headless-capture-runtime.

## Migration phases

- **Phase 0** — this document + CLAUDE.md announcement. *(done)*
- **Phase 1** — extract `thalos_world` from `physics_canonical`:
  - Move `BodyDefinition`, `BodyKind`, `OrbitalElements`, `StateVector`, `BodyId`,
    `G`, `AU_TO_METERS`, the orbital-element→Cartesian helpers,
    `SolarSystemDefinition`, and `parsing` into `world`.
  - `physics_canonical` drops its `thalos_terrain`/`thalos_atmosphere`
    dependencies, depends on `world`, and re-exports the moved types
    transitionally so downstream call sites compile unchanged (a follow-up
    sweep points them at `thalos_world` directly).
  - `world` keeps the `atmosphere`/`terrain` deps for now (aggregation) and
    re-exports their config types.
  - **Cycle resolutions:**
    - The ship spawn state (`SolarSystemDefinition.ship` / `ShipDefinition`) is
      *derived*, not authored — it was a debug parking orbit computed at parse
      time. Removed from `world`'s `SolarSystemDefinition`, which now exposes
      `homeworld_id`. Consumers compute the spawn via
      `physics::debug_orbits::debug_parking_orbit_relative_state` from the
      homeworld body. This keeps orbital algorithms out of `world`.
    - `SharedTerrainRegistry` (the `PlanetSurface`-backed `TerrainProvider`
      impl) moves out of `physics` to `game` (its only construction site). The
      `TerrainProvider` *trait* + `FlatTerrain` stay in `physics`, which no
      longer needs a terrain dependency. (Terrain-height fetching is being
      redesigned independently, so the concrete provider's final home is left
      open.)
- **Phase 2** — render consolidation. *(done)*
  - `planet_lighting` + `planet_rendering` + `terrain_render` merged into
    `thalos_body_render` (`shading`/`impostor`/`ground` modules behind one
    `BodyRenderPlugin`). The three old crates are deleted. Runtime composition
    depends on `thalos_body_render`; local physics consumes only the shared
    `thalos_terrain::HeightSource`/patch contract and has no rendering dependency.
  - The `atmosphere` data crate folded into `thalos_world::atmosphere`
    (deleted; authored body data has one home). `world`'s pure-crate Bevy-guard
    entry covers it.
  - `udlod` sealed behind `body_render`: it is re-exported as
    `thalos_body_render::udlod` and is the fork's single consumer — `game` and
    `body_editor` no longer depend on `thalos_udlod` directly. Replacing the
    ground backend is now localized to the `ground` module + that re-export.
- **Runtime verification (pending):** Phases 1–2 are compile-green but have not
  been confirmed by launching the game. The render merge changed shader-load
  paths; a `just game` launch should confirm shaders load and bodies render.
- **Phase 3** — rename `planet_editor` → `body_editor`. *(done)* Migrating
  "planet" terminology in the render layer to celestial-body terms remains.
- **Phase 4** — render *mechanism* consolidation (state-in / pixels-out).
  *(planned — sequenced AFTER the graphics-fidelity foundation, `docs/graphics_fidelity.md`
  §3.)* The scene-level rendering mechanism is currently split between crates: the
  camera/post bundle (`body_render::impostor::post_stack`, mis-filed under
  `impostor/`), the `scene_depth` and `sun_shadow` render-graph nodes (in
  `game::rendering`), the env probe (`game::reflection_probe`), and the hull
  material (`thalos_shipyard::material`). These consolidate into
  `thalos_body_render` — the de-facto render crate, already consumed by both
  `game` and `shipyard` — which is then renamed **`thalos_render`**. **Drivers stay
  put:** `game::rendering` is ~90% sim-coupled systems (read `SolarSystemState` /
  `CraftState` → uniforms / LOD swaps / spawns) and remains in `game`. The boundary
  is **state-in / pixels-out**: the render crate owns *how* to shade; `game` owns
  *what* to shade this frame. Deferred so the full mechanism set — including the new
  AO / sky-view-LUT / env-IBL render-graph nodes the fidelity foundation adds — is
  extracted in one pass rather than twice.
- **Phase 4a** — decouple rendering from `thalos_shipyard`. The construction crate
  carries rendering only because it began as a self-contained egui editor with its
  own viewport; now that the in-game Bevy-UI editor is primary, `thalos_shipyard`
  should own *ship definition + construction* (parts, blueprints, geometry mesh
  builders, sizing/stats/staging, editor command/state logic) and **none of the
  shading**.
  - **Done (2026-06-30, compile + clippy clean):** the craft hull material
    (`ShipPartMaterial` / `ShipPartExtension` / `ShipPartParams` + the base
    constructors) moved from `thalos_shipyard::material` to
    `thalos_body_render::craft`; the duplicated dims→`ShipPartParams` derivation
    unified into one `thalos_shipyard::appearance::ship_part_params` (was copy-pasted
    in `game::ship_view` and the editor's `editor::visuals`). This **unblocks the
    craft hull *receiving* the shared sun-shadow cascade** (graphics-fidelity
    F6b/F7 — `docs/graphics_fidelity.md`), since `thalos::shadow` + the metallic
    BRDF branch now live alongside the material.
  - **Done (2026-07-01):** the editor was slimmed *out* of `thalos_shipyard`
    entirely. The `editor/` module (`ShipEditorCorePlugin` + placement / visuals /
    commands / shrouds / state) moved into its sole consumer, the game, at
    `thalos_runtime::shipyard_editor::core`; the standalone egui `ship_editor` binary
    was **deleted** (superseded by the native-UI in-game editor). This retired
    follow-up items (3) relocate the binary and (4) drop `bevy_egui` +
    `thalos_celestial` from `thalos_shipyard`. `thalos_shipyard` is now the
    construction *model* only (parts / blueprints / geometry / sizing / stats /
    staging), with deps `bevy`, `glam`, `serde`, `ron`, `thalos_body_render`.
  - **Completed 2026-07-21:** the dependency is now
    `thalos_body_render → thalos_shipyard`. Material types, registration, and the
    construction-dimensions→material projection live in rendering; the
    construction crate contains no renderer dependency or material re-export.
