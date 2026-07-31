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
   impostor billboard (distant) and the ground mesh (close — the standard-path
   `tiles` renderer, or legacy udlod for bodies it has not taken over) are
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

## Crate granularity (ADR-20260731T024003Z)

A crate is three things at once: **a compile unit, an ownership boundary, and
an iteration harness**. Split whenever a boundary buys at least one of four
payoffs:

1. **A cheaper edit loop** — edits to X stop rebuilding Y (the capture-host
   restart compile is the most-multiplied cost in the repo).
2. **A compiler-enforced dependency guarantee** — no-Bevy, no-renderer,
   state-in/pixels-out. Guarantees in `Cargo.toml` don't decay.
3. **A standalone harness** — the crate carries its own preview/example binary
   that compiles without the runtime (`object_preview`, `just ui-preview`).
4. **Agent isolation** — concurrent agents own disjoint crates: fewer merge
   conflicts, no shared incremental-artifact invalidation.

Guardrails: modules still handle ordinary feature size (the bar is a payoff,
not taxonomy — in practice anything feature-shaped ≥ ~3–5 kLOC with a clean
state seam clears it); **feature crates never depend on each other**, only
downward on domain crates and `thalos_game_state` — cross-feature talk goes
through the blackboard, and two feature crates wanting each other means the
shared thing belongs a layer down; the game-state crate is **types-only and
append-biased** (no systems — churn there rebuilds the world); and **don't
split what's scheduled for demolition** (`thalos_udlod`, the procedural
terrain generation chain).

The dependency layers:

```text
L0  foundation    diagnostics, big_space
L1  pure domain   world, terrain, celestial, physics_canonical, control,
                  navigation, texgen            (no Bevy — CI-guarded)
L2  Bevy leaves   input, ui, body_shading, physics_local, shipyard,
                  capture/*, + feature crates (hud, map, shipyard_editor,
                  structures, clouds, …)
L2.5 game state   thalos_game_state — the blackboard L2 reads/writes
L3  composition   thalos_runtime — schedule ordering, plugin graph,
                  sim-coupled drivers, glue; target ~25–35 kLOC
L4  shells        apps/game, tools/capture_host
```

New feature work goes in a feature crate whenever it clears the bar;
`thalos_runtime` accepts only composition, sim-coupled drivers, and glue.
The migration that dismantles the current 93 kLOC runtime is **Phase 5**
below.

## Target workspace

The hierarchy separates the player-facing application, reusable libraries,
developer/offline executables, authored/runtime assets, and generated evidence.
Within `crates/`, folders express responsibility and intended dependency
direction; ordinary feature size is handled with Rust modules rather than a
crate per feature — with the granularity bar above deciding when a feature
*does* earn a crate.

```text
apps/
  game/                         # bin: thalos_game

crates/
  vendor/
    big_space/                  # foundational, upstream-derived dependency
  foundation/
    diagnostics/                # event contract + JSONL sink, no Bevy;
                                #   used by the game, tools, and offline bins
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
                                #   tiles/ = the default ground renderer
    shading/                    # thalos_body_shading: shared shading model
    clouds/                     # (planned, Phase 5c) volumetric-cloud composite
    udlod/                      # LEGACY terrain-render backend (EOL)
  interface/
    input/
    ui/
  gameplay/
    state/                      # (planned, Phase 5a) thalos_game_state: the
                                #   types-only blackboard feature crates share
    hud/                        # (planned, Phase 5b) flight HUD + navball + MFD
    map/                        # (planned, Phase 5b) map view, maneuver
                                #   planning, flight-plan display
    shipyard_editor/            # (planned, Phase 5b) the in-game VAB editor
    structures/                 # (planned, Phase 5b w/ cleanup pkg D) terrain-
                                #   anchored structures, runway geometry, bases
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
  *(planned — sequenced AFTER the graphics-fidelity foundation, `docs/roadmap/graphics_fidelity.md`
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
    F6b/F7 — `docs/roadmap/graphics_fidelity.md`), since `thalos::shadow` + the metallic
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
- **Phase 5** — dismantle the runtime monolith (ADR-20260731T024003Z;
  sequenced as backlog Track 1 rows). Measured baseline 2026-07-31:
  `thalos_runtime` is ~93 kLOC, ~43% of workspace Rust; the blocker is that
  the shared game-state resources are defined inside runtime modules.
  - **5a — the state seam (prerequisite).** Extract **`thalos_game_state`**
    (`crates/gameplay/state`): `SolarSystemState`/`SimulationState`,
    `CraftStateMirror`, `ViewAnchor`, `GameContext` + `ContextHistory`,
    `SimClock`, `MapSnapshot`, `UnitsSettings`, `ActiveCraft`,
    regime/authority resources, and the small shared component vocabulary
    (`PlayerShip`/`CraftRoot`, `CelestialBody`, …). Types, accessors, and
    single-writer doc comments only — no systems. Pure move, compile-neutral.
    Single-writer invariants move with their doc comments; the sole writers
    stay in runtime (or in the feature crate that owns them after a peel).
  - **5b — peel the leaves** (each independent once 5a lands; runtime lands at
    ~50 kLOC of drivers, spawn/scenario machinery, and glue):
    - `thalos_shipyard_editor` (~6 kLOC) — cheapest, most self-contained;
      partner of `thalos_shipyard`.
    - `thalos_hud` (hud/ + navball + velocity_frame + units display,
      ~12 kLOC) — biggest single win; also the natural home for the
      nav-display preview harness.
    - `thalos_map` (maneuver/, flight_plan_view/, map_view,
      body_tree_panel, trails, ~8 kLOC) — already hard-bounded by the
      `MapSnapshot` invariant.
    - `thalos_structures` (base_editor/ + structures + runway geometry,
      ~7 kLOC) — this is cleanup package D wearing a crate boundary; do D as
      an extraction and get both for one price.
    - Capture presets (`screenshot.rs` + viewpoints, ~7 kLOC) — wants to move
      toward `capture/`, but is scenario-coupled; last in line, possibly
      staying in runtime.
  - **5c — split `body_render` along composite lines** (opportunistic):
    `thalos_clouds` first (self-contained: own shaders, uniforms, compute —
    the clearest "one thing" in the repo); `tiles` possibly becomes its own
    crate when the NTR M5 extraction lands (it is the landing pad anyway).
    This *composes with* Phase 4, it does not contradict it: Phase 4 moves
    render mechanisms out of runtime into the rendering layer; 5c organizes
    the rendering layer into focused composite crates. `thalos_udlod` is
    never split — it is deleted.
  - **Deliberately untouched:** `physics_canonical` (cohesive math),
    `thalos_terrain` (until diffusion replaces the generator — that rework is
    the natural split moment), the runtime `rendering/` drivers (the Phase-4
    state-in/pixels-out boundary governs them).

---

# Crate anatomy (module-level reference)

Moved out of `CLAUDE.md` on 2026-07-25 so the agent operating manual stays
context-cheap. This is the **module-level map of the codebase**: what each crate
owns and where a given concern lives. Where a subsystem also has a spec under
`docs/gameplay/`, `docs/simulation/`, `docs/world/`, or `docs/rendering/`, **that
spec is the authority** on behaviour and this section is only the locator.

## Workspace crates

The crate-distillation refactor's phase status is in *Migration phases* above.

Thalos is a planetary exploration / orbital mechanics sandbox in Rust
(edition 2024, Bevy 0.19, glam 0.32). Workspace crates:

- **`thalos_world`** — *(Phase 1, new)* authored source of truth for the system
  and its bodies: `BodyDefinition`, `OrbitalElements`, `StateVector`, the RON
  loader (`parsing`), and the body subsystem-config aggregate. Pure Rust, no
  Bevy. Consumed by physics, terrain gen, and rendering.
- **`thalos_physics_canonical`** — pure Rust orbital-mechanics algorithms +
  runtime simulation state; depends on `thalos_world`. (Name contrasts with
  `physics_local`/Avian, not a claim of being the foundation.) Also hosts the
  native atmospheric-aero force model (`aero`): a whole-body lift/drag +
  stability/damping/control evaluator the game drives force-only in the local
  bubble — see `docs/simulation/aerodynamics.md`. Also hosts `surface_local`: the
  body-fixed Y-up tangent-frame math (anchor + ENU basis, inertial↔SLF
  conversions composed on `body_fixed`, exact gravity/centrifugal/Coriolis,
  re-anchor) the ship local-physics bubble integrates in — see
  `docs/simulation/surface_local.md`.
- **`thalos_control`** — pure-Rust fly-by-wire control layer. The single
  command vocabulary every ship-control source speaks (`AttitudeDemand` /
  `ControlDemand` tagged with a `DemandSource` priority), the priority
  `arbitrate`, the one `AttitudeController` (full-quaternion `Hold` PD +
  nose-`PointNose` PD, replacing the old per-frame deadbeat SAS damper),
  and the effector `allocate` (one torque → reaction wheels + aero control
  surfaces, so they stop fighting). **SAS is regime-dispatched**: on a
  winged craft flying in atmosphere (a `flight::FlightState` is supplied)
  the same SAS hold becomes a plane fly-by-wire law — pitch-attitude +
  bank-angle hold (heading free, so stick-free turns coordinate) with
  pitch auto-trim, sideslip damping, and an AoA envelope that
  stall-protects every pitch command including the pilot's stick —
  while spaceships / vacuum / SAS-off keep the quaternion path unchanged.
  Depends one-way on `physics_canonical`;
  no Bevy. The game-side glue is `thalos_runtime::control_bus`. See
  `docs/simulation/control.md`.
- **`thalos_input`** — Bevy enhanced-input contexts, RON binding loader, and per-binary input intent resources
- **`thalos_ui`** — the **game UI kit** (Bevy): design tokens (palette /
  spacing / type scale, `UiTheme`), the frosted-glass panel surface
  (`GlassMaterial` + the `UiBackdropSource` scene-copy pass), and the widget
  library (buttons/menu rows, sliders, checkboxes, cycle pickers, text
  fields, scroll columns, toasts, headings/dividers). **Every screen composes
  this kit — no per-screen colours, fonts, or interaction styling**; the
  flight HUD's `HudTheme` is a projection of the same tokens. Iterate with
  the kitchen-sink testbed (`just ui-preview` → PNG). Fonts: Inter
  (interface, OFL) + Fira Code (numeric/mono; Δ-strings stay mono by
  convention). See `docs/gameplay/ui.md`.
- **`thalos_runtime`** — the canonical Bevy composition layer shared by the
  player app and headless capture host; owns gameplay, rendering integration,
  UI, scenarios, and deterministic capture presets.
- **`thalos_game`** — the player-facing binary under `apps/game`; a thin wrapper
  that launches `thalos_runtime::AppBuilder`.
- **`thalos_terrain`** — procedural terrain generation pipeline (no Bevy dependency)
- **`thalos_weather`** — planetary weather simulation (no Bevy dependency, `crates/domain/weather`): a seeded reduced-gravity shallow-water layer + advected moisture tracer on a lat-lon grid, supplying the **synoptic organization** of the cloud weather cube — jets, Rossby-wave cyclone trains with fronts and dry slots, ITCZ convergence rain, subsidence-carved clear belts. Deterministic per seed (bit-for-bit, unit-tested); consumed by `thalos_runtime::solar_system_state::CloudWeatherField::from_climate`, which spins it up ~24 model days (~4 s) at body spawn and samples its cloud + rain fields as the occupancy driver under the authored zonal climatology and the existing mesoscale/cellular texture layers. It replaced the painted synoptic layer (curl-warped noise, analytic vortex rotations, ridged-noise fronts), which plateaued at a marbled/spirograph look no tuning escaped (2026-07-31). Iterate via `cargo run --release -p thalos_weather --example sim_probe [days] [seed]` → PNGs in `artifacts/diagnostics/weather_sim/` — the crate compiles in seconds, which is the point.
- **`thalos_celestial`** — procedural sky model: stars, galaxies, nebulae as physical flux sources (no Bevy dependency)
- **`thalos_texgen`** — procedural texture generation (no Bevy): CPU-rasterizes
  `TextureData` (sRGBA8), today the **foliage atlas** (small multi-toned leaf
  clusters + conifer needles + full-colour painterly bark) the tree meshes
  sample, plus a companion **foliage material atlas** (`foliage_material_atlas`,
  linear `Rgba8Unorm`: bark tangent-space normal in RGB + roughness in A). Bark
  albedo, normal, and roughness are all derived from one shared `bark_height`
  field, so cracks/ridges line up across channels; the height field uses
  **gradient (Perlin) noise**, not value noise, so the derived normal shows no
  lattice "weave" (see the `wgsl-bevy` skill note). The Bevy-free generator is
  isolated under `crates/offline/texgen`; `just texgen` runs the thin production
  tool and writes versioned PNGs plus a manifest under
  `assets/generated/vegetation/`. Rendering embeds those baked assets and does
  not depend on the generator. The atlas layout + `leaf_code` packing are the
  stable producer/consumer contract. Rocks and future procedural textures use
  the same offline boundary.

  *(The former `thalos_atmosphere` data crate — gas-giant cloud decks, hazes, rings, terrestrial scattering schemas — is folded into `thalos_world::atmosphere`; authored body data has one home.)*
- **`thalos_physics_local`** — Bevy/Avian f64 local-physics boundary for M5; aggregate craft hydration, terrain collider patches, contact/collapse helpers. **Ships integrate in the surface-local frame (SLF)** — a body-fixed tangent frame anchored under the craft, Y-up, small (meters–km) coordinates near the anchor, re-anchored at ~1.5 km drift; the frame math is `thalos_physics_canonical::surface_local` and the design/implementation notes are in `docs/simulation/surface_local.md`. The Avian rigid body persists across every regime; what *role* Avian plays each frame is a three-way `AvianRole`: `Paused` under warp / `BodyFixed` (canonical owns everything), `AttitudeOnly` while coasting in vacuum at 1× (Kepler owns translation, Avian still integrates rotation + contact for player input and SAS), `Full` when there's a non-gravity force to integrate (throttle active, terrain collider attached, or inside the atmosphere shell). Since the A3 port the role is **classified by the `CraftRegime` resolver** (`thalos_physics_canonical::regime`) and merely projected onto `AvianAuthority` by `compute_avian_authority` (`crates/runtime/game/src/local_physics/mod.rs`), which keeps the `previous_role` edge the handoff snap reads. Coasting flight in vacuum stays under Kepler / `OnRails` so AP/PE do not drift. The role classifier (`compute_avian_authority`) lives in `crates/runtime/game/src/local_physics/mod.rs`; the resulting **canonical authority transitions are owned by the regime executor** (`crate::regime::apply_regime_authority`, applying the unit-tested `thalos_physics_canonical::regime::expected_authority` — it subsumed the former `manage_authority`, the landed throttle release, and the timed settle collapse; see `docs/simulation/regimes.md` Phase A3). **Ground colliders are solid and static in the SLF**: terrain is a parry **heightfield** (not a one-sided trimesh — the trimesh's one-step penetration recovery flung landing craft off their gear), the runway is a solid cuboid slab (`crates/runtime/game/src/runway.rs`). A **wheeled craft's hull is filtered out of solver contact with the ground** via collision layers (`GROUND_LAYER`/`CRAFT_LAYER`); its raycast spring-damper landing gear is the sole ground interface and its force/torque is inertia-relative clamped. Gearless craft (landers) keep all-vs-all layers and rest on the heightfield directly. Fast descents are kept from tunneling by `SweptCcd` + the analytic `terrain_floor_backstop`, and a too-hard contact destroys the craft via the whole-craft impact model (`detect_terrain_impact` → `Simulation::mark_destroyed`, gated on `ShipParameters::impact_tolerance_m_s`; the contact signal is `weight_on_wheels` for wheeled craft, hull contact for gearless). **EVA is a deliberately separate kinematic path** — it is *not* an SLF citizen: it has no collider and computes its canonical state directly in the body-fixed frame (`player_controller::step_eva_controller`), so it gains nothing from the SLF's contact-solver stability; do not "unify" it into the SLF without on-foot walk-testing (see `docs/simulation/surface_local.md` §10). On destruction the game force-pauses and shows an in-place scenario-respawn picker (`crates/runtime/game/src/scenario_menu.rs`) offering the four start scenarios (ship orbit / landing / final approach / EVA); see `docs/simulation/surface.md`.
- **`thalos_body_render`** — *(Phase 2, new)* unified celestial-body rendering, one appearance model + regime-scaled projections. Five modules behind one `BodyRenderPlugin`: `shading` (shared `SceneLighting`/`AtmosphereBlock`/Hapke `shade_hapke_surface` + the `thalos::lighting`/`thalos::atmosphere` WGSL libraries), `impostor` (distant billboard materials for planets, gas giants, rings, solid bodies), **`tiles` (the default ground renderer** — cube-sphere quadtree of ordinary `Mesh` + `StandardMaterial`/`TileTerrainMaterial` entities on Bevy's standard path, `SurfaceQueryProvider` content, `TileHeightMirror` for ground consumers), `ground` (**legacy**: the `thalos_udlod`-backed terrain LOD — `ThalosTerrainPlugin`, `PipelineTileProvider`, `BodyTerrainMaterial` — plus the *non*-legacy analytic `BodySkyMaterial`/`BodyOceanMaterial` projections and rendered-height patch utilities), and `clouds` (the absorbed MIT `bevy-volumetric-clouds` fork: spherical body-fixed volume march + cloud-local history targets). Merged from the former `planet_lighting`+`planet_rendering`+`terrain_render` and top-level `thalos_volumetric_clouds` mechanisms. A backend chooses geometry or cost, never its own lighting/atmosphere/weather authority. Terrestrial clouds obey ADR-20260720T212214Z-one-weather-field-many-cloud-projections: authored `CloudClimate` → one per-body `CloudWeatherField` → near/orbit/shadow projections.
- **`thalos_udlod`** — **LEGACY / end-of-life** (keystone ADR-20260723T142945Z): `thalos_body_render::tiles` is the default ground renderer; this crate streams only bodies the tile driver has not installed on, or the whole world under the `THALOS_TILE_RENDERER=0` A/B baseline. Defect-driven fixes only — the rules below are for keeping it alive, not for extending it. Thalos-owned UDLOD terrain renderer at `crates/rendering/udlod/`. It originated from [`kurtkuehnert/bevy_terrain`](https://github.com/kurtkuehnert/bevy_terrain) by Kurt Kühnert (MIT OR Apache-2.0); attribution + license files travel with the source. It is **runtime-provider-first**: it renders sparse tile atlases fed by `TileProvider` implementations, not preprocessed Earth-style asset trees. The old GeoTIFF/preprocess path is gone. (Upstream's own successor is now a *different repo*, [`planetary_terrain_renderer`](https://github.com/kurtkuehnert/planetary_terrain_renderer) — a useful diff reference for fixes to the Taylor-series precision path.) A 2026-07 optimization pass tailored the subsystem to Thalos; **see `docs/reference/terrain_lod_optimization.md`**, and note these load-bearing rules:
  - **Providers own mip generation.** `TileProvider::request_tile` must return the **full mip chain** (call `AttachmentData::generate_mipmaps` inside the task). The atlas does *not* regenerate mips — that kept per-tile mip filtering on the main thread and made cached payloads useless.
  - **Attachments may differ in resolution.** The GPU atlas sizes each attachment's texture array independently. Height keeps the full grid (it is the geometry, and the only attachment physics reads); albedo/roughness/material bake at half (`TierConfig::detail_texture_size`) — a >2× cut in the game's largest allocation.
  - **Tiles are cached, and the cache key is the contract** (`game::rendering::tile_cache`): memory (survives terrain despawn/respawn) over disk (survives the process) over synthesis. The namespace is a `NamespaceFn` resolved **per request**, not frozen at construction, because the flatten handle is read per tile *pixel* — a pad installed after spawn still changes what later tiles bake. **If you add an input to tile synthesis, fold it into the namespace, and bump `thalos_terrain::GENERATOR_VERSION` when generation output changes** — otherwise a cached run silently renders old terrain. `THALOS_TILE_CACHE=0` disables the disk tier while iterating on generation.
  - **CPU draw-tile selection is the sole tile-selection authority** (it enforces the 2:1 LOD balance across cube-face seams that the GPU's per-tile-independent predicate could not). The dead GPU tiling prepass has been **deleted** — do not reintroduce it. Refinement now also honours a screen-space-error hint (`TileProvider::subdivision_scale`, ≤ 1, so it can only *remove* detail on flat ground) and a hole-free behind-view streaming cull (`TerrainViewConfig::cull_behind_view`).
  - Tile **production** on the GPU remains the intended big win, but it is blocked on an architectural decision, not effort: porting the cascade to WGSL creates a *second height authority* that would drift from the CPU one the colliders and spawn-site search read. See the doc's "What did not land, and why". **`big_space` integration is unconditional** — the upstream `high_precision` Cargo feature has been removed, along with the runtime `DebugTerrain.high_precision` toggle and the `HIGH_PRECISION` shader define / pipeline flag. The Taylor-series relative-position path (`compute_relative_position` in `shaders/functions.wgsl`) is the only viable precision path at planet scale; gating it behind a feature only forced defensive `#[cfg]` plumbing in every consumer.
- **`big_space`** — vendored floating-origin / high-precision grid substrate
  at `crates/vendor/big_space/`.
  It originates from [`aevyrie/big_space`](https://github.com/aevyrie/big_space)
  0.12 (MIT OR Apache-2.0); retain its upstream attribution and licence. It is
  foundational despite being vendor code: real_space, udlod, and render all build on it.
  Consumed via `[workspace.dependencies] big_space { path, features = ["i64"] }`.
  The `i64` cell precision is the workspace-wide choice. udlod keeps its
  `big_space.rs` re-export shim.
- **`bevy_erosion_filter`** — author-owned temporary GPU/CPU erosion-noise filter
  at `crates/domain/terrain/erosion_filter/`. The `bevy` feature is optional so
  `thalos_terrain` uses the pure-glam `cpu` module with `default-features = false`
  and pulls no Bevy crate (the no-Bevy CI guard still holds); `thalos_body_render`
  uses the `bevy` feature for the shader-library plugin. It is expected to be
  replaced by diffusion terrain work, not maintained as a generic vendor fork.
- **`thalos_shipyard`** — parametric ship **construction model** (ECS attach tree, RON blueprints): part components + catalog, resources, blueprint (de)serialization + spawn, attach nodes / surface mounts / KSP linked symmetry, parametric sizing + mass/capacity recompute, stats / staging, and the geometry mesh builders (cockpit / engine / fuselage / gear / wing) shared with the game's flight-craft rendering. It owns *what a craft is*; it does **not** own the interactive editor or any UI. The **editor application** lives with its sole consumer, the game, at `thalos_runtime::shipyard_editor` — a UI-agnostic `core` submodule (`ShipEditorCorePlugin`: `EditorState` command/state hub, placement, live mesh rebuilds, tank-resize handle, placement-preview ghost, shrouds, blueprint save/load against `ships/*.ron`) plus the native Bevy-UI front-end (scene + panels). There is no standalone editor binary (the old egui `just shipyard` tool was deleted). Every editor-owned entity carries the `EditorPart` marker (defined in `shipyard_editor::core`) and every core query filters on it; host systems that aggregate the same part components for the *flight* craft (fuel, staging, gear, ship visuals, colliders) must filter `Without<EditorPart>` — that marker is the only thing separating the build world from the flying craft in the same ECS `World`. Resource storage is whitelist-driven from the parts catalog: any part kind can declare `storage` entries for fixed (`units`) or volume-scaled (`units_per_m3`) capacity, and blueprints may only activate resources whitelisted by that part. Omitted blueprint resources mean "use catalog defaults"; explicit resource maps mean the user's selected active pools. Do not restore hard-coded per-resource tank fields such as `methane_l_per_m3` / `lox_l_per_m3`; add real resources (for example `Kerosene`) to `Resource` and catalog storage lists instead. Air intake is ambient capture, not stored oxidizer: engines declare `intake_requirement`, nacelles may provide `builtin_intake`, and separate `Intake` parts can feed future engine-core layouts. See `docs/gameplay/construction.md`.

## Workspace source layout

The accepted agent-first target is in
ADR-20260721T194628Z-role-based-agent-first-workspace, as refined by
ADR-20260721T211446Z-player-facing-apps-developer-tools, and
`docs/architecture.md`: the player-facing game alone under `apps/`; reusable `vendor` / `domain` / `simulation` /
`rendering` / `interface` / `runtime` / `capture` / `offline` libraries under
`crates/`; headless, production, and offline developer executables under `tools/`; and generated evidence under
`artifacts/`. UI and object verification scenes belong in the canonical runtime
capture path, not separate lab applications. `big_space` is vendor;
`udlod` is Thalos-owned rendering; erosion remains author-owned terrain code
until diffusion replaces it; texgen becomes offline.

Headless verification is a first-class product capability, not a simplified
renderer or a player-facing app. CAP-1–CAP-4 (`docs/development/capture.md`) extract one
`thalos_runtime` shared by the game app and capture-host tool, a typed capture protocol/runtime, one
Rust CLI for persistent/cold stills and comparisons, and deterministic frame
sequence/video capture. Both applications must compose the same plugin graph.

Core separation: `world`, `physics_canonical`, `control`, `terrain`,
`celestial`, and `texgen` are pure Rust libraries; `input`, `runtime`,
`body_render`, `physics_local`, and `shipyard` are Bevy consumers. Within
`body_render`, the `shading` module is the single source of truth for
`SceneLighting`, `AtmosphereBlock`, and the surface BRDF that both the
`impostor` and `ground` backends consume — no field-by-field mirror types.
Avian lives behind `thalos_physics_local`; do not add Avian to
`thalos_physics_canonical`.
Semantic input for the Bevy binaries flows through `thalos_input`
contexts and intent resources, with checked-in defaults at
`assets/input.ron`. HOTAS support also lives there: buttons can be
added to the existing action bindings as `GamepadButton(...)`, while
continuous pitch/yaw/roll/throttle axes are opt-in under
`game.hotas` and feed the same `GameInputIntent` fields as keyboard
flight controls. Those continuous axes are **not** read through Bevy's
gamepad layer: Bevy's `bevy_gilrs` converter drops every axis gilrs
labels `Unknown` (a bare flight stick's twist/throttle slider), so they
never reach `Gamepad` state. `thalos_input::joystick` instead runs its
own `gilrs::Gilrs` instance, reads every axis by raw platform `Code`,
and snapshots `code -> value` into `RawJoystickState`; HOTAS axes bind a
raw `u32` `code` (platform-specific — discover with `cargo run -p
thalos_input --example gamepad_axes`), not a `GamepadAxis`. Do not
revert HOTAS axes to `GamepadAxis` bindings; the twist literally cannot
be expressed that way.

### Canonical physics crate (`crates/simulation/physics_canonical/`)

Two trait abstractions draw the boundaries:

- `BodyTrajectoryProvider` (`body_trajectory_provider.rs`) — answers
  "where is body `i` at epoch `t`?" Implemented by `PatchedConics`;
  could be swapped for a baked ephemeris.
- `ShipPropagator` (`ship_propagator.rs`) — propagates the ship across
  one segment of coast or burn. Implemented by `KeplerianPropagator`:
  analytical Kepler coast under a single SOI body + RK4 finite-burn.
  SOI transitions are detected per substep and refined by bisection
  (coast) or shortened RK4 (burn).

Key modules:

- `canonical` — world preset/config, `Epoch`, `CraftState`,
  `AuthorityMode`. `Simulation` owns a deterministic `CraftId`-keyed fleet;
  each `VesselRecord` contains one canonical state and authority plus its
  parameters, controls, maneuver/prediction state, fuel counters, vessel kind,
  and damage state. Active craft is a separate selection, with the former
  single-craft methods retained as compatibility wrappers.
- `types` — `StateVector`, `BodyDefinition`, `TrajectorySample` (each
  sample carries `anchor_body` + `ref_pos` so the renderer can draw
  without a per-sample ephemeris query).
- `orbital_math` — Cartesian↔Keplerian conversions, Kepler-equation
  solvers (elliptic + hyperbolic), `propagate_kepler`.
- `patched_conics` — `BodyTrajectoryProvider` impl. Bodies form a
  parent-child tree; queries walk the lineage and sum each ancestor's
  motion.
- `ship_propagator` — `ShipPropagator` trait + `KeplerianPropagator`
  impl. Segments terminate on the first of: target time, SOI exit,
  collision, SOI enter, burn end, or stable-orbit closure. Resolution
  order at boundaries: `exit > collision > enter`.
- `simulation` — `Simulation` struct: owns canonical craft state,
  authority bookkeeping, warp, `KeplerianPropagator` instance, and
  `ManeuverSequence`. `step()` is called each frame and consumes
  maneuver nodes as their start time arrives.
- `body_fixed` — pure inertial↔body-fixed frame helpers used by landed
  `BodyFixed` pose evaluation (the rotating-with-the-surface authority).
- `body_centered` — pure inertial↔body-centered (translates with body,
  inertial axes) frame helpers. The frame Avian's local rigid body lives
  in: gravity reduces to `−μr/r³` with no fictitious forces.
- `trajectory/` — Flight-plan prediction. `propagate_flight_plan` runs
  the same `ShipPropagator` across the maneuver sequence, producing
  `Leg`s of `(burn?, coast)` `NumericSegment`s. Includes event
  detection (SOI / apsis / impact), encounter aggregation,
  closest-approach scans.
  - `propagate_branch_stack` produces a `TrajectoryBranchStack` with
    one `BranchKind::Actual` branch (no-maneuver baseline) plus
    `BranchKind::Projected` branches that fork at each maneuver-node
    start; this is how the map view renders ghost/preview tracks for
    pending edits.
- `maneuver` — `ManeuverNode`: time, delta-v in local
  prograde/normal/radial frame, reference body. Plus frame-conversion
  helpers.
- `parsing` *(now in `thalos_world`; re-exported as
  `thalos_physics_canonical::parsing` during the migration)* — Loads
  `assets/solar_system.ron` plus per-body files at
  `assets/bodies/<lowercase_name>.ron` into `SolarSystemDefinition`.
  System-level file carries physical + orbital specs; per-body files
  carry `terrain: TerrainConfig` and `tectonics: TectonicConfig`. The
  authored `BodyDefinition`/`OrbitalElements`/`StateVector` types also
  live in `thalos_world` now (see `docs/architecture.md`); the player's
  debug spawn orbit is derived from `homeworld_id` via
  `debug_orbits::debug_parking_orbit_relative_state`, not stored on
  `SolarSystemDefinition`.

### Game crate (`crates/runtime/game/`)

- **Boot, loading, and the start screen** (see `docs/gameplay/boot.md`).
  `loading.rs` owns `AppState` (`Loading` → `MainMenu` | `Running`) and the
  **`LoadingTracker`** — a declarative step registry (`begin` registers the
  step set for a load; producers update their step by id; the reveal fires
  when all registered steps complete, including deferred craft placement).
  A bare `just game` boots to the **start screen** (`main_menu.rs`:
  scenario picker / SHIPYARD / SETTINGS / QUIT) with the **world deferred**
  (`loading::WorldState::Absent`): no bodies / ship / sky spawn, no terrain
  streams, the boot loading pass registers zero steps (near-instant menu),
  and the winit loop is throttled to reactive mode while the menu is up.
  The world-spawn systems hang off `OnEnter(WorldState::Live)` instead of
  `Startup`; naming a scenario (`just game runway`) or setting
  `THALOS_AUTO_RUN` skips the menu and inserts `Live` (same chain, first
  frame, boot unchanged). The menu's first start after a deferred boot *is*
  a boot: it arms the boot placement flags, registers the boot step set,
  flips `Live`, and re-enters `Loading` (no craft swap needed —
  `spawn_player_ship` builds the chosen scenario's blueprint directly).
  With the world already live (menu re-entered from flight), scenario
  starts reuse the respawn / relaunch machinery in place — the runway pair
  re-arms its deferred placement + settle gate and re-enters `Loading`.
  `SpawnSituation` is therefore **mutable at runtime**; deferred placements
  are explicitly armed (`DescentPlacement`, `RunwayPlacement`), never keyed
  off the situation with a `Local<bool>`.
- **Spawn situation is a flag: ship in orbit, EVA on the
  surface, a landing approach over land, a final approach over
  flat land, one of two surface-runway scenarios, or Saturn on a launchpad.**
  `crates/runtime/game/src/lib.rs` reads
  `just game [mode]` (passed as a CLI arg — default `menu`, the start
  screen; falls back to the `THALOS_SPAWN` env var for a direct
  `cargo run`) into a
  `spawn::SpawnSituation` resource (`ShipOrbit` | `PolarOrbit` | `Eva` |
  `Landing` | `FinalApproach` | `Runway` | `RunwayApproach` | `Launch` |
  `Cruise`).
  The canonical `CraftState` is the player either way — KSP-style: one
  craft, Ship or EVA, distinguished by `VesselKind`. The ship blueprint is
  chosen per scenario by `SpawnSituation::ship_blueprint_path`
  (`apollo.ron` by default, `meridian.ron` for the aircraft scenarios,
  `saturn.ron` for `launch`).
  **`orbit`**:
  `VesselKind::Ship` in a low Thalos parking orbit
  (`system.ship.initial_state`), nose along prograde;
  `ship_view::spawn_player_ship` loads the blueprint and pushes the real
  ship params. **`eva`**: the player on foot, `VesselKind::Eva` with
  `ShipParameters::eva()` (90 kg, no thrust, no torque), placed ~12 km
  above the Thalos sub-stellar point;
  `local_physics::spawn_player_avian_body` branches on `VesselKind` to
  spawn a 1.8 m capsule (rotation-locked, walking friction) carrying both
  `LocalCraftBody` and `PlayerControllerBody`, and `spawn_player_ship`
  early-returns (no rocket). Re-boarding a ship is not implemented yet;
  the `toggle_player_controller` input is unwired. **`land`** (aliases
  `landing` / `descent`): `VesselKind::Ship` on a vacuum powered-descent
  approach. Because placing it *over land* at a true above-ground
  altitude needs terrain heights (unknown until bakes load), the runtime builder
  seeds the parking orbit as a placeholder behind the loading screen and
  `spawn::refine_descent_spawn` installs the real state on the first
  `AppState::Running` frame: it searches the daylight hemisphere for a
  dry-land, ice-free site (terrain height > `sea_level_m`, low latitude),
  then drops the ship ~25 km AGL over it, descending, nose retrograde,
  coasting `OnRails` until it crosses the 20 km local-physics handoff for
  the powered touchdown. **`final`** (aliases `final-approach` /
  `final_approach` / `approach`) uses the same deferred terrain-aware
  path but scores daylight dry sites by local height relief, then starts
  the ship ~1.5 km AGL, low and slow, over the first sufficiently flat
  patch (or the flattest dry fallback). Thalos now has both a *visual*
  atmosphere + volumetric clouds (`terrestrial_atmosphere` in
  `assets/bodies/thalos.ron`; see `docs/rendering/atmosphere.md`) and a *physics*
  atmosphere (per-body density below the Kármán line; see
  `docs/simulation/aerodynamics.md`), so descents now experience aerodynamic **drag** —
  a descending/reentering ship decelerates toward a terminal velocity rather
  than free-falling lunar-style. **`runway`** (alias
  `rwy`) and **`runway-approach`** (aliases `rwy-approach` /
  `approach-runway`) put the `meridian.ron` aircraft on a fixed runway on
  the Thalos surface, owned by `crate::runway`. Like the descent modes
  these are deferred and terrain-aware: `runway::finish_runway_spawn`
  runs once per arming of `RunwayPlacement` (boot, or a start-screen
  runway launch) and places the runway at a **fixed body-fixed site**
  (constant lat/lon + heading, `runway::fixed_runway_site` from the
  `RUNWAY_SITE_*` constants, overridable at runtime with
  `THALOS_RUNWAY_SITE="lat,lon[,heading]"`). This replaced the former
  flat/dry/sunlit grid scan, whose night-side fallback could spawn the
  craft in the dark; the pad is flattened under the footprint regardless,
  so the coordinates only need to be dry land sunlit at the spawn epoch
  (the author's call — a below-sea-level sample is warned, not corrected).
  The scenario also seats the sim clock at a **morning boot epoch**
  (`runway::RUNWAY_MORNING_EPOCH_S`, via `Simulation::set_sim_time`) so the
  fixed site is lit by a low rising sun instead of the epoch-0 noon sun the
  sub-stellar point gives it — time-of-day is a planet-rotation lever, the
  authored site stays put. Re-derive the epoch with the `morning_probe`
  example if the site or Thalos's rotation changes. After placing, it tears
  down any live Avian
  bubble so the rebuild seeds from the placed state (a bubble seeded
  from the pre-placement placeholder orbit would leave the rendered
  craft coasting the wrong trajectory). **The terrain itself is
  flattened into a pad and the runway
  sits flush on it** (replacing the former raised-slab + skirt/runoff
  platform): a single fixed elevation `E = mean(natural terrain over the
  basin)` — levelling to the *mean*, not the max, so the wide basin balances
  cut against fill instead of becoming an all-fill plateau towering over its
  surroundings — is chosen, then a `thalos_terrain::TerrainFlatten` pad is
  installed through the body's shared
  `rendering::ground_terrain::TerrainFlattenRegistry` handle. The terrain
  tile provider (`PipelineTileProvider`, wrapped in
  `thalos_terrain::FlattenedSurface`) reads that handle as it bakes, so the
  *rendered* ground — and, via the GPU-atlas height mirror, the collider and
  CPU height queries — level out to `E` across the basin and smoothstep-blend
  back to natural terrain over a wide (~500 m) ramp.
  The handle is read per tile-pixel, so setting the region affects tiles
  baked afterward — no UDLOD tile-reload needed, because the pad is set
  before the aircraft/camera jump to the site, so the tiles that stream in
  there bake flattened from the start. The handle lives in a per-body
  registry so it survives terrain despawn/respawn churn. On top of the
  levelled ground the runway is just a paved strip + markings + posts (lifted
  a few cm so the paving reads on the grass). To stay rock-steady at high
  warp the runway is positioned like the player ship: a **root** big_space
  grid child re-placed in f64 every frame (`update_runway_transform`),
  *not* a fixed-cell child of the rotating body grid — whose multi-Mm cell
  offset rotated by an f32 quaternion jitters as the body spins fast (only
  the small child vertex offsets ride the f32 `Transform.rotation`). A
  **flat kinematic collider at `E`** (a trimesh posed each frame by
  `sync_runway_collider_pose`, exactly like the terrain collider patch via
  `terrain_patch_pose`) still backs the landing surface so it stays exactly
  flat regardless of tile-streaming timing.
  `runway` parks the aircraft at rest on the threshold (`BodyFixed`
  authority, the launch-clamp pattern), lifted by its own measured ground
  clearance — `craft_ground_clearance` walks the part-visual meshes
  (`Mesh::compute_aabb`, since they carry `NoFrustumCulling` and so have no
  Bevy `Aabb`), finds the lowest point in the craft body frame, and offsets
  the spawn so *any* craft rests on the surface rather than a hardcoded
  constant; `runway-approach` starts it on short final (`OnRails` → bubble
  handoff onto the flat collider). **The parked `runway` spawn happens *behind*
  the loading screen.** `finish_runway_spawn` runs during `AppState::Loading`
  (not the first `Running` frame), so the aircraft is parked + the flatten pad
  installed + the camera at the surface while the screen is up; `surface_settle`
  (`SurfaceSettlePlugin`) then holds the screen until the terrain under the view
  has streamed flush (resident LOD at `tile_tree.view_position()` plateaus past a
  target), so the first visible frame is already flush + settled instead of the
  ground heaving up to the strip. Do **not** pause the sim during this window —
  `readback_local_craft` stops following the craft when the integrator is idle,
  so a pause strands the camera in the placeholder orbit and the tiles never
  stream at the site. Only the *parked* runway is gated; the airborne approach
  flies its descent during loading, so gating it would fly it past the threshold.
  Cold tile streaming to a fresh surface site is slow (~15 s for Thalos) because
  each tile is an expensive CPU field bake; `body_render`'s `compute_tile_pixels`
  is rayon-parallelised on a bounded pool and `udlod`'s `TileAtlas::update`
  admits tile loads **coarse-first, then nearest-view-first within each LOD** —
  so a cold view fills the low-LOD pyramid (complete coarse coverage, cheaply)
  before refining the nearest tiles, instead of building outward from the single
  closest tile and leaving the rest of the view an empty hole. That progressive
  coarse→fine fill is what makes the wait tolerable rather than minutes — see the
  cold-streaming memory note.
- **EVA is a full craft with a real character controller and a
  grounded/airborne split.** The `EvaMode` resource
  (`player_controller.rs`, `Grounded` | `Airborne`, defaults `Grounded`)
  is the coarse surface↔orbit switch. **`Grounded`**: `step_eva_controller`
  runs a kinematic character controller in the body's **body-fixed
  (rotating) frame** — it tracks a body-fixed position + surface-relative
  velocity and runs a grounded/airborne state machine with surface gravity
  (`g = μ/r²`): camera-relative walking (Shift = sprint), jumping (Space),
  walking off ledges into a ballistic fall, and landing. Working in the
  body-fixed frame (not body-centred inertial) is the key fix: surface
  velocity is the player's walking speed (m/s), not the inertial
  co-rotation drag `ω×r` (hundreds of m/s → km/s). That keeps warp from
  exploding the integrator, and the terrain height query never
  sea-level-teleports (a `None`/missing-tile sample holds altitude instead
  of snapping to the reference radius). `snap_avian_from_canonical`,
  `apply_local_forces`, and the `readback_local_craft` translation all
  short-circuit for grounded EVA so the controller owns the capsule pose
  outright; canonical authority is pinned to `LocalRigidBody` by the
  regime executor's walking pin (`regime::expected_authority`, not
  `OnRails`) since grounded EVA co-rotates with the surface and
  `Simulation::step` must only advance sim-time.
  `sync_avian_time` still pauses Avian's clock under warp (the controller
  writes `Position` directly each frame; no integrator is needed and the
  km/s-scaled integration that broke it is gone).
  **Rest detection + warp gating (KSP-style):** the controller publishes
  `grounded` / `at_rest` / `surface_speed_m_s` on `PlayerControllerState`.
  `enforce_warp_altitude_limits` lets on-foot warp climb above 1× only once
  the player is *at rest* (standing, stopped); while walking / jumping /
  falling it clamps to 1×, and movement input drops warp back to 1×
  immediately. At rest it caps at `SURFACE_WARP_MAX_SPEED` = 100× (above
  that the UDLOD tile streamer can't follow the body rotating under the
  camera — a UDLOD limitation, not a sim one). The on-foot HUD pill
  (`hud/eva_panel.rs`) surfaces `MOVING` / `FALLING` / `STANDING — warp
  ready`. **`Airborne`**: Kepler owns translation, the snap drives the
  capsule from canonical (exactly like a ship coasting in vacuum), and
  `step_eva_controller` stands down. `apply_local_forces` short-circuits
  for EVA in both modes — EVA has no thrust yet (coast-only; a jetpack is
  the natural follow-up). The teleports mirror the ship's: body-tree
  cmd-click sends EVA to a low orbit (→ `Airborne`), and a row's `drop`
  button + map cursor (or F9) plants it on a surface point (→ `Grounded`,
  in place via `local_physics::place_eva_on_surface` — the bubble is never
  torn down, unlike ships). `EvaMode` flips only on these explicit
  teleports; suborbital ballistic flight (jump, walk off a cliff) stays
  *within* `Grounded`, switching the controller's internal grounded/airborne
  state, not `EvaMode`.
- Semantic player input is read from `thalos_input::game::GameInputIntent`.
  Keep raw Bevy input only for cursor positions, picking spatial data, and UI
  internals. See `docs/gameplay/input.md`.
- `solar_system_state` — canonical game-level per-frame source for evaluated
  solar-system state. `SimulationState` owns the long-lived simulation,
  ephemeris, and authored system definition; `SolarSystemState` is refreshed
  once per frame from it and owns the `BodyState` vector plus per-body
  environment state (`DynamicSurfaceState`, cloud-band dynamics/phases, and
  later wind, storms, tides, dune movement). Map, real-space, impostor,
  terrain, halo/sky, and material systems are projections of this resource;
  they may cache derived data but must not independently own body/environment
  state.
- `bridge` — Core adapter. Calls `Simulation::step()` each frame,
  recomputes trajectory prediction *synchronously* on the main thread
  when the cached plan is dirty/stale, syncs maneuver edits, handles
  warp controls. (Single early-terminating `propagate_flight_plan`
  pass keeps the typical rebuild well under a frame; running in-line
  means an edit on frame N produces the fresh trajectory on frame N.)
- `control_bus` — game-side glue for the `thalos_control` fly-by-wire
  layer. Each frame `realize_control` collects every attitude command
  source as a tagged `ControlDemand` (pilot stick, the `T`-key SAS hold,
  the directional nav modes via `navigation::nav_attitude_demand`, the
  scheduled-burn autopilot via `Autopilot::attitude_demand`), arbitrates
  by priority, runs the one `AttitudeController`, and allocates the
  resulting torque to *both* effectors — reaction wheels
  (`ControlInput::torque_command` → `apply_local_forces`) and aero control
  surfaces (`RealizedControl::aero` → the aero force system). Attitude is
  no longer set anywhere else; the old `bridge::handle_attitude_controls`
  + `navigation::compute_attitude_control` + raw-stick aero paths are
  gone, and the per-frame deadbeat SAS damper with them. `realize_control`
  also arms the plane flight assist: when SAS is engaged on a winged craft
  flying in atmosphere it builds the body-frame `FlightState` (local up via
  `surface_local::radial_up`, air-relative velocity, the config's
  `stall_alpha`) that switches the controller's SAS hold to the fly-by-wire
  pitch/bank law with auto-trim + stall protection, and publishes
  `RealizedControl::assist` for the HUD (the SAS button reads FBW /
  warn-tints while protection clamps). **SAS defaults on** (`SasState`,
  surviving destruction/respawn); the HUD SAS button toggles that same
  switch as the `T` key. Throttle still
  rides its own setpoint path (`ThrottleState::commanded`, autopilot
  override, `ControlLocks`) pending a later fold-in. See `docs/simulation/control.md`.
- `map_view` — snapshot/projection boundary for map rendering. Copies
  `CraftState`, body states, and `FlightPlan` into `MapSnapshot`; map
  systems consume the snapshot and never mutate canonical simulation
  state.
- `hud/` — Bevy-UI HUD panels (`HudPlugin`), one file per panel, sharing
  the `theme::HudTheme` style. Includes the **MFD slot** (`hud/mfd/`): a
  contextual, customizable ship-view widget slot that auto-selects the
  widget relevant to the current flight context (orbital `Trajectory`
  plot, airliner `NavDisplay`, with `Docking`/`Interplanetary` stubs)
  with a manual pin/hide override. Each widget exposes `build` /
  `relevance(&FlightContext)` / optional `update`; `select_active_widget`
  is the single owner of widget + slot visibility, and `HudPanel` lives on
  the slot container only. This replaced the old single hardcoded
  "TRAJECTORY" panel, which surfaced an orbital plot during atmospheric
  flight. See `docs/gameplay/hud_widgets.md`.
- `rendering/` — Every system that turns simulation state into
  rendered geometry. Submodules:
  - `types` — rendering-shared resources (`CameraExposure`,
    `PlanetshineTints`) and components (`CelestialBody`, `PlayerShip`,
    `PlanetMaterials`, `SolidPlanetMaterials`, etc.). It re-exports
    `SimulationState` / `SolarSystemState` for compatibility, but their
    definitions live in `solar_system_state`.
  - `real_space` — BigSpace root and per-body real-space grids. The
    ship camera carries `FloatingOrigin`; UI and map entities stay
    outside the BigSpace hierarchy.
  - `spawn` — startup system that creates map-side body entities plus
    real-space body grids. Procedural bodies get a `PlanetMaterial`
    impostor billboard; non-procedural bodies get a
    `SolidPlanetMaterial` solid-color billboard.
  - `generation` — polls the in-flight `PlanetSurface` async tasks,
    bakes the result into a `PlanetMaterial`, and handles
    reference-cloud loading.
  - `lighting` — `CameraExposure`, `SceneLighting` population, planet
    / solid-planet material light updates, sun-light direction.
  - `transforms` — map snapshot projection, body/ship transform sync,
    planet orientation (tidal lock + spin).
  - `materials` — per-frame parameter updates for gas-giant, ring,
    and cloud-band animation.
  - `trails` — orbit-line periodic recompute + gizmo draw with focus
    / sibling fade.
  - `body_lod` — screen-space LOD: icon ↔ impostor crossfade,
    moon-merge fade, double-click-to-focus picking, homeworld focus.
  - `scene_depth` — `SceneDepthImage` resource + copy pass. Copies the main
    view depth into a sample-able `Depth32Float` image for custom effects that
    still consume it. The canonical rocky-body atmosphere uses Bevy's own
    depth-aware raymarch; the retained `BodySkyMaterial` is a debug atmosphere,
    while ocean and clouds are dedicated backend-independent composites.
  - `tile_terrain` — driver for the **default** ground renderer
    (`body_render::tiles`): installs a `TileTerrainRoot` on the first
    anchor-resolved terrain body, republishes the selection eye from
    `ViewAnchor` each frame, and points the body's `RenderedGround` +
    `HeightSource` at the tile height mirror. `THALOS_TILE_RENDERER=0` is the
    legacy-udlod A/B baseline (boot `OnceLock`; needs a cold run).
  - `ground_terrain` — **legacy** UDLOD terrain spawn for procedural bodies +
    impostor↔terrain LOD swap (`sync_terrain_impostor_swap`) at
    `4 × radius`. It still spawns the legacy `BodySky` fullscreen quad for
    debug comparison; normal rendering force-hides it in favour of the Bevy
    raymarched atmosphere. `BodyOceanMaterial` and `CloudCompositeMaterial`
    remain visible independently and must never be gated by that toggle.
  - `view_anchor` — the **one per-frame answer to "where is the view?"**
    for every view-dependent detail system. `ViewAnchor` (sole writer
    `update_view_anchor`, top of `SimStage::Sync` before
    `sync_solar_system_state`) resolves the `ShipCamera`'s big_space pose
    against the nearest terrain-backed body, **body-fixed at a coherent
    epoch** (previous-frame pose paired with previous-frame body states, so
    re-projection with the current frame's `BodyState` is exact under
    co-rotation × warp — see the module doc). Surface scatter (trees /
    grass / GPU grass / rocks) and the sun-shadow cascade centre all read
    it; **nothing view-dependent may anchor to the player craft or a mode
    flag** — new camera modes (god views, freecam, screenshot rigs) get
    correct detail with zero per-mode plumbing. Replaced
    `scatter_view_center` + `ShadowFocusOverride`.
- `ghost_bodies` — Renders ghost planet positions during time warp
  preview.
- `sky_render` — Renders procedural sky from `thalos_celestial`
  catalog (stars, galaxies as GPU meshes).
- `maneuver/` — Maneuver node placement/editing UI. Delta-v handles
  in local reference frame.
- `flight_plan_view/` — Renders the trajectory branch stack as
  on-screen tracks.
- `camera` — KSP-style orbit camera.
- `shipyard_editor/` — the **in-game ship editor**: the UI-agnostic editor
  logic in the `core` submodule (`ShipEditorCorePlugin`, moved here from the
  former `thalos_shipyard::editor`) plus its native Bevy-UI front-end (`scene`
  + `ui`), over the `thalos_shipyard` construction model. A **separate scene**,
  not an `AppState`: `ShipyardEditor::open` is a `SimClock` pause source, and
  the three `SimStage` sets (Physics/Sync/Camera) + the HUD update systems are
  gated on `shipyard_editor::editor_closed` (configured in `lib.rs` /
  `hud/mod.rs`), so **no game logic runs while the editor is open** — the
  flight world is suspended, not just hidden. While open, the scene cameras
  deactivate and a dedicated editor camera renders the build world on
  `coords::EDITOR_LAYER` (layer 5) — build entities (marked `EditorPart`,
  layer-stamped each frame) never bleed into flight/map views and survive
  close/reopen. All gameplay input contexts deactivate and the
  `ShipyardContext` (orbit/click) activates (`input.rs` gate); Escape-close is
  owned by `pause_menu::handle_escape_input`'s priority chain. UI is native
  Bevy UI in `HudTheme` style (`shipyard_editor/ui/`):
  parts palette, parametric slider inspector, staging readout, top bar with
  ship-name text field + mirror/snap/layout toggles, a **▶ Launch** button,
  status bar. Entry: the pause menu's SHIPYARD button, or `just game
  shipyard`. When adding game systems that
  aggregate shipyard part components, filter `Without<EditorPart>` (see
  fuel/staging/local_physics).
- `relaunch` — fly the editor's current design without a process restart
  (the editor's **Launch**). Carries the live build's collected
  `ShipBlueprint` (no file round-trip) to a two-phase relaunch: despawn the
  old `PlayerShip` + bubble, reset the sim, seat the craft into a scenario
  (cruise for aircraft, orbit otherwise — reusing the destruction-respawn
  helpers), then rebuild the craft via the shared
  `ship_view::build_player_ship` core. The new craft carries no `EditorPart`,
  so staging/bubble rebuild through the existing systems. Runway relaunch is
  deferred (one-shot terrain-aware startup placement).
- `base_editor/` — the **in-world surface base editor** (Cities:Skylines-style):
  pick a building site on the real terrain, flatten the land into a level pad,
  then click-and-place / edit placeholder buildings on it. Unlike the shipyard
  editor it is an **in-world overlay**, not a separate scene — the planet stays
  visible, the sim pauses (`BaseEditor::open` is a `sim_clock` pause source —
  like a warp-0 pause, so only the **Camera** `SimStage` set gates on
  `base_editor::base_editor_closed`; Physics/Sync keep running so the ground
  stays streamed/lit, just frozen), and the existing `ShipCamera` is
  repositioned to a god-view (`camera.rs`, WASD-pan + orbit/zoom) rather than
  swapped. Reuses the `StructureSite`/`StructureRegistry`/`apply_structure_flatten`
  layer (`structures.rs`, grown with `BaseSite`/`Building`/`Launchpad`/`Tank`
  kinds + `remove`/`update`) and the runway's per-frame f64 anchoring. Entry: the pause menu's
  SURFACE BASE button; Esc closes. The one new mechanism is **live runtime
  terrain invalidation** — UDLOD has no per-tile re-bake path, so a flatten
  installed after tiles stream is applied by `TerrainRebuildRequest`
  (`rendering::terrain_residency`) despawning + respawning the body terrain
  (reusing the persistent `TerrainFlattenRegistry` handle). Tab toggles placing
  a building vs a **launchpad** (a craft can be placed on it with **L**, reusing
  the runway's parked-placement helpers — runway spawn is horizontal-on-gear, a
  launchpad spawn is **vertical** nose-up), and a **typed auto-connection
  network** (`base_editor::connections`: `ConnectionKind` = taxiway / apron /
  road / crawlerway, each its own width+material; line networks are MST or
  explicit-edge strips, aprons are filled rects) regenerates as structures
  change. Runways are now **parametric + multi-instance**
  (`StructureKind::Runway { half_length_m, half_width_m }`, rendered + collided
  from the registry by `runway.rs`, with **ICAO-font designator numbers 01–36
  painted from the true compass heading**; every runway lies on the
  one shared basin tangent plane via `RunwayFrame::center_offset`, so an offset
  strip stays flush with the flattened ground). The
  **default base is a spaceport on a flat basin**: `runway::build_spaceport`
  registers a wide `BaseSite` basin, registers **two runways in a V** (the 5 km
  primary + an angled crosswind secondary diverging 30° from near the primary's
  threshold, on the side opposite the launch complex; the divergence gives each
  strip its own designator numbers, so no L/R suffix pair) draping on it,
  then `base_editor::spawn_default_base` authors the rest coplanar as **one
  core campus** on the launch-complex side (nothing between the runways): two
  launchpads with clearing, per-pad flame diverters + tank/fuel farms (`Tank`
  cylinders), a VAB-scale building, a row of hangars standing on a **large
  apron auto-derived from the hangar row**, blockhouses/ops, and the typed
  networks — a straight full-length core parallel taxiway plus three **curved
  link taxiways** (`connections::spawn_authored_path`, waypoint polylines with
  circular corner fillets) that cross the primary strip **split at the runway
  edges** (stub stops at one side, curve resumes at the other — no paving under
  the strip) and sweep tangentially onto the secondary's parallel taxiway,
  straight connectors + threshold **holding pads**, a curved landside road, and
  VAB→pad crawlerways — so the surface scenarios present a whole base. Spawn
  points are **intrinsic** to runways/launchpads, and the **create-craft→fly
  flow** is wired: the shipyard/VAB **LAUNCH** rebuilds the craft into an orbit
  hold then drops into an in-world **launch-point picker** — a
  `BaseEditorMode::SelectLaunch` god-view (`base_editor/launch_select.rs`) where
  clicking a runway lands the craft horizontal-on-gear and a launchpad lands it
  vertical-nose-up, via the shared placement cores (`runway::place_on_runway` /
  `place::place_on_launchpad`). The spaceport is built lazily on the first launch
  (`runway::build_spaceport`, extracted from `finish_runway_spawn`) behind a brief
  loading pass, and persists thereafter. Every **launchpad now carries its own
  kinematic collider** (`spawn_structure_entity`) — required because a
  `RunwaySite` makes `local_physics::terrain_patch` skip the generic ground patch
  on that body, so a pad-launched craft would otherwise fall through. The base's
  flattened ground reads as a **grass lawn**
  (thick short `GrassProfile::lawn`) with its paved/built footprints (runway,
  launchpads, buildings, tanks) **cleared** — the building-terrain scatter layer:
  `body_render::ground::scatter`'s `ScatterRegion`/`ScatterTreatment`/`classify_scatter`,
  derived from the `StructureRegistry` by `rendering::grass` and honoured by the
  grass tile builder (the seam future base trees/props extend). See
  `docs/gameplay/base_building.md` *Ground scatter*.
- `space_center/` — the **KSP-style Space Center hub**: the god-view overview you
  land in when you press **PLAY** on the start screen. Like the base editor it is
  an **in-world modal mode** (a `sim_clock` pause source, `space_center_closed`
  gates only the **Camera** `SimStage`), not an `AppState`. It is the *hub* the
  facility editors hang off: a `space_center::ui` panel offers **EDIT BASE** (→
  base editor on the existing basin) and **VAB** (→ shipyard editor), plus a
  **building-selection framework** (`space_center::select`) — left-click raycasts
  the pad sphere for the nearest selectable structure, outlines it, and entering
  an already-selected **facility** building opens it. Facilities are tagged on the
  `StructureSite` via `structures::Facility` (only `Vab` wired today; the seam for
  runway/pad launch, tracking station, admin). Entry: start-screen PLAY (a
  **clean start** — builds the spaceport *base only*, `runway::build_spaceport`
  with **no craft parked**, behind the loading screen via
  `HubSpaceportBuild`/`finish_hub_spaceport`, then opens the hub on reveal), or
  the in-flight pause menu's **SPACE CENTER** button. The only way to fly is to
  launch a craft yourself from the **VAB** (→ `base_editor::launch_select`'s
  pick-a-runway/pad flow) — nothing is loaded on the pad by default. **EXIT /
  Escape** from a PLAY-opened hub returns to the **main menu**
  (`SpaceCenter::return_to_menu`, since nothing is flying yet); from a
  pause-menu-opened hub it returns to that flight. `ReturnToSpaceCenter` makes a
  facility entered *from* the hub return to the hub on close (a VAB **Launch**,
  which queues a `RelaunchRequest`, instead drops to flight). The god-view camera
  is the **shared** `god_view` module (`GodViewOrbit` + `drive_god_view`),
  extracted from the base editor so both modes share the exact 3/4 orbit/zoom/pan
  feel and one shadow-focus override. Start screen (`main_menu`) is now the usual
  **PLAY / SETTINGS / QUIT** with the direct-scenario shortcuts tucked in a
  collapsible **QUICK START / DEV** submenu.
- `settings` — **unified settings persistence**. All user preferences live in
  one `settings.ron` (`AppSettings { window, graphics, units }` — a section per
  domain). Storage location switches on build profile (Bevy 0.19's
  `bevy::platform::dirs::preferences_dir`): **debug** keeps it project-local at
  the gitignored `user/settings.ron` (easy to inspect/reset during dev),
  **release** uses the OS app-data dir `<preferences_dir>/thalos/settings.ron`.
  `settings::load()` runs in `main()` before the app (the window section shapes
  the initial `Window`) and **migrates the legacy per-domain files**
  (`user/{settings,graphics,units}.ron`) on first run — a sectioned vs flat-RON
  heuristic disambiguates the new unified file from the old window-only
  `settings.ron` at the same path. Each domain still gets its **own Bevy
  resource** (`WindowSettings`/`GraphicsSettings`/`UnitsSettings`, inserted in
  `main()`) so every consumer is unchanged; `AppSettingsPlugin`'s one autosave
  watches all three and rewrites the file when any value changes (value-compared,
  so an open settings tab doesn't churn it). The per-domain modules below own
  only their resource + runtime behaviour, not the file IO.
- `window_settings` — window/display preferences (mode, windowed resolution,
  vsync, fullscreen monitor, user UI scale; the `window` section). Edited live
  from the settings menu's Window tab (`settings_menu`).
  `THALOS_WINDOW_MODE` / `THALOS_WINDOW_SIZE` / `THALOS_VSYNC` are *session
  overrides* (`overrides_from_env`: they win for the run, grey out their UI
  control, never persist), and `THALOS_SCALE` stays a pure env diagnostic.
  `apply_window_settings` pushes settings onto the primary `Window`
  (value-compared) and writes back windowed drag-resizes (the unified autosave
  persists them); `apply_ui_scale` (which absorbed the former
  `compensate_fractional_ui_scale`) multiplies the user UI scale into the
  fractional-HiDPI crisp-text snap and drives `UiScale`. Caveat: runtime mode
  switches recreate the swapchain, which the known flaky Windows driver path
  (generic surface-acquire panic in Bevy's `prepare_windows`) can turn into a
  crash — pre-existing platform issue, newly reachable at runtime.
- `graphics_settings` — graphics/rendering preferences (the `graphics` section),
  edited live from the settings menu's Graphics tab (`settings_menu`). Knobs:
  the **volumetric-cloud toggle** (`GraphicsSettings::clouds`), read by
  `rendering::clouds::drive_clouds` — when off it parks the cloud raymarch via
  the no-cloud-body path (`ActiveCloudBody = None` → blank fallback bound onto
  `BodySkyMaterial`), so the sky renders clear at near-zero GPU cost; the grass
  toggle; and the MSAA level. Add new render knobs here as fields + a control in
  `show_graphics_tab`.
- `units_settings` — display-unit preference (the `units` section), edited from
  the settings menu's Units tab. **SI is always the internal/simulation unit;**
  this only changes how a value is formatted for display. Two fields, because one
  global switch is the wrong model: `system` (`Metric` | `Imperial`) and
  `aviation` (`Aeronautical` | `FollowGlobal`). Aviation is unit-conservative in
  a way the rest of the world is not — feet, knots, ft/min, and nautical miles
  are the instrument units in metric countries too — so the preference is
  resolved **per `UnitDomain`** via `UnitsSettings::system_for(domain)`, which is
  the only supported way to reach a display unit. All conversion lives in
  `hud::format` (one function per quantity, each taking a resolved
  `UnitSystem`); a surface that hardcodes `"{:.0} m/s"` or reads
  `UnitsSettings::system` directly is a defect.

  Three kinds of caller, and the distinction is the whole design:

  1. **Instruments that state their own domain.** `UnitDomain::Aviation` — the
     PFD tapes/readouts (`hud::pfd_panel`), the atmospheric TAS/q/Mach pill
     (`hud::atmo_panel`), the MFD navigation display
     (`hud::mfd::widgets::nav_display`). A PFD tape is an aviation instrument in
     orbit too, so it never switches. `UnitDomain::General` — Δv, staging
     masses, map scales, and the whole shipyard editor, which follow `system`
     exactly.
  2. **Readouts shared between spaceflight and aviation**, via
     `UnitDomain::shared(airplane_flight)`. The altitude cluster
     (`hud::orbital_panel`: current altitude *and* apoapsis/periapsis) and the
     velocity readout (`hud::flight_panel`) show approach height and airspeed on
     final but orbital height and velocity on a transfer — the *same widget*, so
     there is no static answer and the situation has to pick. The whole altitude
     cluster resolves through one call, so the panels sitting side by side can
     never disagree about their unit.
  3. **The situation itself** is `hud::mfd::FlightContext::airplane_flight()` =
     `in_atmosphere && winged` — the "type of thing **and** situation" test.
     `winged` comes from the live `AeroConfig::lift_slope > 0`, which the
     blueprint's lifting panels already produce; there is no craft class in the
     model and this is the honest proxy for one. A rocket climbing through the
     same air is *not* flying: it keeps metres, because knots are meaningless on
     an ascent profile. `hud::IN_ATMOSPHERE_DENSITY` is the single threshold
     behind `in_atmosphere`, shared by the atmo pill's visibility, MFD widget
     auto-selection, and this test — they must agree on where the atmosphere
     starts or a panel appears in units its neighbours don't share.

  A `Marine` domain (knots + nautical miles) is the obvious next one and is
  deliberately absent until there is a ship instrument to attach it to.
  Deliberately metric regardless: the EVA walking-speed pill, base-editor
  building dimensions, the MFD trajectory-plot scale bar, and freecam.

Systems run in `SimStage` order: `Physics → Sync → Camera`
(configured in `crates/runtime/game/src/lib.rs`), ensuring deterministic state flow each
frame. Enhanced input intent collection runs in `PreUpdate` before these
sets. Simulation pause is an explicit clock boundary, not a global Bevy
clock pause: `crates/runtime/game/src/sim_clock.rs` owns `SimClock`, whose sole
writer folds Escape pause, destruction scenario picker, freecam, the
shipyard editor, and warp pause into a zero sim delta. Canonical stepping, local physics ownership,
resource burn, and grounded EVA motion consume `SimClock`; presentation
and UI animation use `Time<Real>` directly. Do not reintroduce pausing
`Time<Virtual>`/plain `Res<Time>` as the game-wide sim pause switch.

### Terrain gen crate (`crates/domain/terrain/`)

Cubemap-based procedural surface generation. No Bevy dependency.

- `TerrainConfig` — top-level enum in `terrain_config.rs`: `None`,
  `Feature(FeatureTerrainConfig)` for archetype-driven bodies (Mira,
  Vaelen, Thalos, Pelagos…), `Ocean(OceanTerrainConfig)` for
  flat-water placeholders. Current P2A.5 migration state: Thalos is
  authored as an ocean-bearing `OceanicTerrestrial` prototype; its ground
  LOD uses `RuntimeTerrainDetail::OceanicContinental` so the same signed
  continent/seabed evaluator defines bake height, runtime mesh height, and
  collider height instead of the legacy P0 HMF detail cascade. Runtime
  ground height is LOD-invariant in this slice to avoid contour-like
  parent/child tile handoff steps. Water is not a terrain material:
  underwater terrain keeps seabed albedo/material/roughness, and ocean
  color/reflection/optical absorption come from the separate water renderers.
  The fuller tectonic/hydrology Thalos route is parked for a later P2 slice.
- `compile_terrain_config(...)` — single entry point. Builds the
  optional `TectonicSystem` from `TectonicConfig`, compiles the
  static surface (via `feature_compiler` or `compile_ocean`), and
  compiles dynamic layers. Returns a `PlanetSurface`.
- `PlanetSurface` (in `static_surface.rs`) — the top-level product:
  - `static_surface: StaticSurfaceData` — immutable, cacheable
    cubemaps + `IcoBuckets` spatial index + optional `sea_level_m`.
  - `dynamic_layers: DynamicSurfaceLayers` — terrain-owned
    time-varying overlays (seasonal ice, unconsolidated aeolian
    bedforms).
  - `tectonics: Option<TectonicSystem>` — plate graph regenerated on
    each load (cheap, kept outside the static-surface cache).
- `BodyBuilder` — mutable build-time state. Stages mutate this
  (height/albedo accumulators, craters, volcanoes, channels,
  materials, detail noise params).
- `Stage` trait — `name()`, `apply(&mut BodyBuilder)`. Each stage is a
  pure transform; the feature compiler invokes them directly and sets
  `stage_seed` before each `apply()` so there's no ambient state.
- `tectonics/` — plate graph (`Plate`, `PlateKind`, `Boundary`,
  `BoundaryKind`), spherical mesh, plate-derived fields. Currently
  consumed by `ColdDesertFormerlyWet` (Vaelen) and by the parked fuller
  `AgingOceanicHomeworld` Thalos route; the active P2A Thalos prototype
  omits tectonics.
- `surface_field` / `aging_oceanic_field` / `cold_desert_field` /
  `vaelen_field` — continuous archetype surface fields sampled into
  the cubemap accumulators.
- `sample()` / `sample_static_surface()` / `SurfaceSample` — sampling
  contracts for reading finished surface data.

### Celestial crate (`crates/domain/celestial/`)

Procedural sky model. Pure Rust, no Bevy. Works in physical
quantities (flux, temperature, SED) — never pre-baked RGB.

- `Universe` — collection of `Source` objects (stars, galaxies, nebulae)
- `Spectrum` — spectral energy distributions: `Blackbody`,
  `PowerLaw`, `Tabulated`; `Passband` filtering
- `generate/` — procedural star field and galaxy placement
- `render/` — cubemap baking and telescope PSF

### Shipyard crate (`crates/domain/construction/`)

Parametric ship **construction model** (Bevy, no UI). Owns *what a craft is* —
parts, resources, blueprints, attach tree, sizing/stats/staging, geometry mesh
builders — not the editor (that's `thalos_runtime::shipyard_editor`, see the Game
crate section). The full next-gen construction model (planes/ships/stations from
one Module primitive) is specced in `docs/gameplay/construction.md`; **Slices 1–2
(parametric wing, wing-pylon jet nacelle, and scalar intake flow) plus
visually-actuating control surfaces have landed** — see that doc's §0 for the
status boundary.

- `AttachNode` / `Ship` — ECS tree structure for ship assembly
- `Part` trait — `CommandPod`, `Engine`, `AirIntake`, `FuelTank`,
  `Decoupler`, `Adapter`, `Wing`
- **Two placement capabilities.** `Attachment` is end-node stacking (the
  rocket path: mate `top`/`bottom`, diameter propagates). `SurfaceMount`
  is surface/footprint placement: sit a part on a host's skin at
  `(station, angle)`, **opting out of diameter propagation** (wings,
  wing-pylon nacelles; later gear). It is a *parallel* component, not a
  fork of `Attachment` — graph traversals that need every part union both.
- **KSP linked symmetry** — `SymmetryGroup { id, role }`. Placing a
  footprint part under the editor's Mirror (2×) mode stamps a **real
  counterpart entity** (separate left/right parts) linked in a group;
  `sync_symmetry_groups` keeps the mirror in lockstep with its primary
  (params copied with handed fields like `Wing.incidence` negated, mount
  reflected across X = 0), and edits/deletes act on the whole group. Nested
  symmetry stamps a part placed on a mirrored host (a nacelle on a wing)
  onto both hosts. Downstream the mirror is a plain part: meshes are
  single-panel, stats/staging count each entity once (no ×2), and the
  blueprint persists a `symmetry_group` id only so the editor re-links on
  load (the game ignores it).
- `Wing` — tapered/swept/dihedral lifting surface (span, root/tip chord,
  sweep, dihedral, t/c, incidence). `wing_mesh::build_wing_mesh` builds it
  in the host-local frame, shared by the editor and the game's `ship_view`.
  **Control surfaces are `Wing` parameters** (`Wing::control_surfaces:
  Vec<ControlSurface>` — ailerons/elevator/rudder/flaps/spoilers, each a
  trailing-edge spanwise window + chord fraction + deflection limit), not
  separate parts:
  `build_wing_mesh` notches them out of the loft and `build_control_surface_mesh`
  meshes each as a separate hinged sub-mesh about a consistently-oriented
  hinge axis (`control_surface_geometry` is the shared seam a future
  per-surface force model reads). The game animates the attitude surfaces from
  the fly-by-wire command (`RealizedControl::command`, *not* the allocated
  aero fraction): elevators on pitch (symmetric), ailerons on roll
  (differential by mount side), rudder on yaw. `Flap`/`Spoiler` windows
  deflect instead from `thalos_runtime::flight_config::FlightConfig` — the
  three-detent flap lever (F extend / R retract) and the B brakes latch
  (wheel brakes + spoilers, KSP-style) — and their authored window geometry
  derives the craft's flap/spoiler ΔCL/ΔCD in `build_ship_aero_config`
  (`docs/simulation/aerodynamics.md` *Flight configuration*). **Per-surface control
  authority**: the per-axis control coefficients likewise derive from the
  authored aileron/elevator/rudder windows (deflection lift × real moment arm
  about the CoM — `derive_control_coefficients`), so surface sizing and
  placement show up in handling; the *moment structure* stays the whole-body
  stable model — forces are never an emergent per-surface strip sum, which
  pumps energy (`docs/simulation/aerodynamics.md` *Per-surface control authority*).
  Gear will be its own footprint part. No flight model yet (M6).
- `Engine` / `AirIntake` — rocket bells remain node-stacked; jet nacelles can
  surface-mount to wings as `WingPylon` mounts with generated pylons. Ambient
  flow is modeled separately from resources: engines declare required intake
  kind/area, nacelles may carry built-in capture, and standalone intakes
  provide capture for future separated inlet/core designs.
- `ShipBlueprint` — RON serialization. Node mates serialize as
  `connections`; surface mounts as a separate `surface_mounts` list
  (`#[serde(default)]`, so pre-wing saves load unchanged).
- `sizing` — parametric node sizing (adapters/tanks scale from parent);
  surface mounts are intentionally excluded.
- `stats` — aggregate mass / CoM / MOI / Δv, plus geometry-derived
  **wing area + mean aerodynamic chord** ("will it fly" feedback; no sim).
- `staging` — pure stage derivation from decoupler topology
  (`derive_stages`) + per-stage Δv/fuel accounting
  (`compute_stage_summaries`). Single home for the staging model: the
  game's live ECS staging systems (`crates/runtime/game/src/staging.rs`) and the
  editor's staging preview (`ShipBlueprint::stage_summaries`) both call it,
  so stage boundaries never diverge. Like `stats`, no Bevy beyond glam math.

### Body render crate (`crates/rendering/render/`)

Unified Bevy rendering for celestial bodies — one appearance model with
regime-scaled projections. No world generation logic. Added via a single
`BodyRenderPlugin` (which composes four module sub-plugins). Four modules:

**`shading`** — the single source of truth every body-surface material
reads from. No materials of its own.
- `PlanetLightingPlugin` — registers the `thalos::lighting` and
  `thalos::atmosphere` shader libraries. The `impostor` and `ground`
  sub-plugins also add it defensively (no-op if already added) so each
  works standalone.
- `SceneLighting` / `StarLight` / `MAX_STARS` / `MAX_ECLIPSE_OCCLUDERS`,
  `AtmosphereBlock` / `CLOUD_BAND_COUNT` — scene + per-body uniforms.
- `shaders/lighting.wgsl` — WGSL mirror of `SceneLighting` plus
  `eclipse_factor`, `planetshine_sample`, `hapke_brdf`, and
  `shade_hapke_surface`. The impostor always shades through
  `shade_hapke_surface`; the ground LOD picks a path per body via
  `TerrainShadingStyle` (`body_terrain.wgsl`, carried in
  `BodyTerrainExtras.inspection.y`, derived from the terrain archetype at
  spawn): airless `Regolith` bodies (Mira) route through the same
  `shade_hapke_surface` so they reconverge with the impostor across the LOD
  swap, while `Vegetated` bodies (Thalos) use the ground-only rough-dielectric
  BRDF + ecological albedo bands.
- `shaders/atmosphere.wgsl` — WGSL mirror of `AtmosphereBlock`,
  `integrate_atmosphere`, `composite_clouds`, `apply_limb_darkening`.

**`impostor`** — distant billboard materials.
- `PlanetRenderingPlugin` — registers `PlanetMaterial`,
  `GasGiantMaterial`, `RingMaterial`, `SolidPlanetMaterial`.
- `PlanetMaterial` (cubemap height/albedo, `assets/shaders/planet_impostor.wgsl`),
  `SolidPlanetMaterial` (`solid_planet.wgsl`), `GasGiantMaterial`
  (`gas_giant.wgsl`), `RingMaterial` (`ring.wgsl`). These four shaders
  are AssetServer-loaded from the workspace `assets/shaders/` dir.
- `bake_from_body_data()` — consumes `terrain::PlanetSurface` →
  `PlanetTextures` for upload into a `PlanetMaterial`.

**`tiles`** — **the default ground renderer** (NTR-X1; keystone
ADR-20260723T142945Z). Terrain as ordinary `Mesh` +
`StandardMaterial`/`TileTerrainMaterial` entities streamed by a camera-driven,
2:1-balanced cube-sphere quadtree, parented to the body's rotating big_space
grid, entirely on Bevy's standard render path.
- `TileTerrainRoot` / `TileEye` — per-body residency + the selection eye, which
  the game republishes every frame from `ViewAnchor`
  (`thalos_runtime::rendering::tile_terrain`, the driver + install gate).
- `SurfaceQueryProvider` — content from the body's canonical
  `Arc<dyn SurfaceQuery>` (wrapped in the shared `FlattenedSurface` so structure
  pads level tile ground exactly as they level udlod's).
- `height_mirror::TileHeightMirror` — CPU mirror of the resident tile grids,
  published as the body's `RenderedGround` so scatter, colliders, the camera
  floor, and HUD altitude read *the ground that is actually drawn*. Sampling the
  analytic surface instead is not equivalent: `SurfaceQuery` is band-limited by
  `lod_m` and the mesh is ~6 m/vertex at the deepest level.
- `material::TileTerrainMaterial` — `ExtendedMaterial` over `StandardMaterial`
  (`assets/shaders/tile_terrain.wgsl`): Hapke branch for airless bodies,
  `thalos::shadow` receive, and the shared `THALOS_TERRAIN_INSPECTION` lane.

**`ground`** — **LEGACY** terrain half + non-legacy composites. The
`thalos_udlod`-backed terrain LOD (former `terrain_render`) is end-of-life and
takes defect-driven fixes only; `BodySkyMaterial`, `BodyOceanMaterial`, the
impostor handoff, and the ground-patch utilities that share this module are the
ADR-20260723T142945Z carve-out and outlive it.
- `ThalosTerrainPlugin` — adds `thalos_udlod::TerrainPlugin` +
  `BodyTerrainMaterial`/`BodySkyMaterial`/`BodyOceanMaterial`/`GrassMaterial`,
  embedding their `src/ground/*.wgsl` via `embedded://thalos_body_render/ground/…`.
- `PipelineTileProvider`, `rendered_height_*`, the `HeightSource` family,
  and rendered-height patch utilities used by M5 colliders.
- `vegetation` — near-camera grass-blade decoration layer for vegetated
  bodies: cube-sphere grass-tile lattice + batched blade-mesh builder
  (placement reuses the tile baker's grass-mask gate against the body's
  `HeightSource`) + `GrassMaterial`. Driven per-frame by
  `thalos_runtime::rendering::grass` (runway-style f64 body-fixed anchoring,
  revision-based rebuilds). See `docs/world/terrain.md` *Vegetation decoration
  layer*.
- `ground_patch` — `GroundPatchMaterial` / `GroundPatchMaterialPlugin`: a flat,
  sky-model-lit ground plane (the shared `thalos::lighting` dielectric BRDF, not
  the UDLOD stack) that **receives** the same cascaded sun-shadows trees cast —
  the diorama ground for the object preview, and the seed for the planned larger
  composed scenes. Deliberately simple: a flat-ground analogue of
  `body_terrain.wgsl` for tooling, sharing `TreeMaterial`'s cascade binding
  layout so one shadow rig feeds both.

**`clouds`** — spherical, body-fixed volumetric cloud render mechanism.
- Owns the absorbed `bevy-volumetric-clouds` compute pipeline, generated
  Perlin-Worley/Worley textures, cloud colour/distance targets, and cloud-local
  temporal history; upstream MIT attribution lives beside the module.
- Consumes a cube `CloudWeatherMap` uploaded from the active body's canonical
  `CloudWeatherField`. It does not create weather or choose a body.
- The game-side `rendering::clouds` driver selects the nearest authored cloudy
  body and projects `CloudClimate`/environment state into view uniforms. Near
  composition stays in `BodySkyMaterial`; the first orbit projection is in
  `SolidPlanetMaterial`. See ADR-20260720T212214Z-one-weather-field-many-cloud-projections and `docs/rendering/clouds.md`.

`body_render` is the **sole consumer** of the Thalos-owned `thalos_udlod`,
re-exported as `thalos_body_render::udlod` (`{prelude, math, big_space}`); no
other crate depends on the fork directly. Replacing the ground backend stays
localized to the `ground` module + that re-export.

## Data flow

```
assets/solar_system.ron + assets/bodies/<body>.ron
  → [parsing] SolarSystemDefinition (physical + orbit + TerrainConfig + TectonicConfig)
  → [PatchedConics] body positions at any epoch t
  → [Simulation::step] per frame → canonical CraftState + authority, consumes ManeuverNodes
  → [solar_system_state::sync_solar_system_state] canonical per-frame BodyStates
       + per-body environment state (`CloudWeatherField`, dynamic surface;
         later weather evolution/wind/tides)
  → [propagate_flight_plan / propagate_branch_stack] synchronous prediction
       → FlightPlan / TrajectoryBranchStack (Actual + Projected branches)
  → [map_view] MapSnapshot → map rendering, maneuver UI, collision warnings
  → [rendering::real_space] BigSpace grids → ship-view body / camera transforms

[terrain::compile_terrain_config]
  → PlanetSurface { static_surface, dynamic_layers, tectonics }
  → [body_render::bake_from_body_data] PlanetTextures
  → PlanetMaterial uploaded to GPU; impostor billboard renders
```


## Terrain generation (procedural bodies + Mira offline-package MVP)

`BodySurfaceRegistry` constructs one canonical `Arc<dyn SurfaceQuery>` per body.
Most bodies use `ProceduralSurface`, a pure analytic function of `(direction,
lod)` in f64 body-local coordinates. Mira is the first offline-package body:
`just bake Mira` writes `assets/terrain_packages/Mira.bin`, and startup validates
its content key before loading `BakedSurface`. There is no runtime/startup bake
check: a missing or stale package fails with the explicit bake command. Every
terrain-height consumer reads the same selected surface:

- **Ground LOD render** — a `FlattenedSurface`-wrapped canonical surface feeds
  the UDLOD tile provider.
- **Collider / camera terrain floor / runway / descent site / HUD altitude / EVA**
  — the near-surface `HeightSourceRegistry` (a GPU-atlas height mirror over the
  same surface, with a CPU fallback), via
  `HeightSource::sample_height_m`.
- **Propagator collision** — `GameTerrainRegistry` mirrors the canonical surface.

For the current Thalos generator there is **no sea-level layer**: the continent mask puts the shoreline at height
0 (the reference radius), so "sea level" is the constant **0 m** wherever a datum
is needed (dry-land checks, the camera's terrain floor). Do not read sea level
from a baked surface — that path is gone.

Mira's compatibility baker currently wraps the retained deterministic airless
compiler into a schema-v1 package with an explicit node/blob manifest,
32→512 adaptive residual height pyramid, ancestor fallback, checksums, exact
artifact identity, validator, and `PackageSurface`; player devices
never run the producer. The production
direction is an offline hierarchical-diffusion bakery that emits adaptive,
versioned cube-sphere terrain packages; see `docs/world/mira_airless_mvp.md` and
ADR-20260720T211046Z-offline-terrain-packages. Player devices stream packages through a `PackageSurface` backing and
reconstruct only bounded close-range detail. The authored package and the
disposable runtime tile cache are separate layers. Package-backed and procedural
bodies must still converge on the one `SurfaceQuery` consumer path and one height
authority.

The old runtime baked pipeline
— `thalos_bake_dump` (the "dumps"), `thalos_body_editor`, and the game's startup
`bake_check` — has been **deleted**. The remaining baked-pipeline modules in
`thalos_terrain` (the `Feature`/`Ocean` compiler, `PlanetSurface`/
`StaticSurfaceData`, `cache`, stages/tectonics/fields) are retained only as the
temporary offline Mira producer/global package payload. Do not reconnect them
to a runtime startup bake or revive the dump/editor flow. MIRA-0's adaptive
package tracer is complete; MIRA-1/2 replace the producer with diffusion. New bakery
work follows ADR-20260720T211046Z-offline-terrain-packages and the package spec behind the same `SurfaceQuery` seam.
MIRA learned models are Rust-native and authored once with pinned Burn 0.21 per
ADR-20260721T032343Z-bevy-raymarched-rocky-atmosphere. `thalos_terrain_learned` is Bevy-independent shared model/sampler code;
training checkpoints carry raw/EMA weights plus path-remapped Adam state so
cross-process resume remains deterministic. `thalos_terrain_train` is the
offline producer tool. Candle may be selected as a
Burn backend or used as a diffusion reference, but must not become a second
model implementation. Keep both learned crates out of `thalos_game` until a
measured optional runtime feature needs them; gameplay remains package-first.


## Assets

- `assets/solar_system.ron` — system-level definition: bodies'
  physical + orbital specs. RON format with `#![enable(implicit_some)]`.
- `assets/bodies/<lowercase_name>.ron` — per-body terrain + tectonics
  configuration. Loaded alongside the system file by `parsing`.
- `assets/parts.ron` — shipyard parts catalog.
- `assets/shaders/planet_impostor.wgsl` — procedural-body impostor
  (3-layer: baked cubemap low-freq, SSBO mid-freq features,
  shader-synthesized high-freq detail).
- `assets/shaders/solid_planet.wgsl` — solid-color billboard impostor.
- `assets/shaders/gas_giant.wgsl` — gas giant cloud / haze / rim
  rendering.
- `assets/shaders/ring.wgsl` — planetary ring system shader.
