# Surface base building

The in-world **base editor** is a Cities:Skylines-style tool for laying out a
surface base on a planet: pick a building site on the real terrain, flatten the
land into a level pad, then click-and-place / edit buildings on it. It is the
first realization of the player-placement gameplay that `surface_local.md` §6.4
designed for and deferred. Code lives in `crates/game/src/base_editor/`.

> **Status (2026-07-04).** Compiles clean; **not yet runtime-verified** by a
> `just game` launch. Implemented: site pick + live flatten, building + launchpad
> place/select/move(drag)/delete/rotate, keyboard sizing, a **typed
> auto-connection network** (taxiways / aprons / roads / crawlerways — MST or
> explicit edges, plus **curved fillet paths**), and an **authored default base**
> (the runway scenario spawns a wide flat *basin* holding a spaceport — **two
> numbered runways in a V** (a 5 km primary plus an angled crosswind secondary,
> each with its own true-heading designator numbers), one **core campus** on the
> launch-complex side (runway → full-length parallel taxiway with connectors →
> a **large apron auto-derived from the hangar row standing on it** → landside →
> pads/VAB), threshold holding pads, the secondary reached through three
> **curved link taxiways** that cross the primary strip (each crossing split at
> the runway edges — stub stops at one side, curve resumes at the other, no
> paving under the strip) and sweep tangentially onto its parallel taxiway (no
> buildings between the strips), a launch complex of two pads
> with tank farms / flame diverters, a VAB, ops + blockhouses, a curved landside
> road, and VAB→pad crawlerways, all coplanar), plus **spawning** (craft seat on
> a runway horizontal / a launchpad vertical).
> Deferred follow-ups (see *Not built yet*): the shipyard create-craft→fly flow,
> a slider inspector, user-drawn routing + crawler animation, disk persistence,
> scoped tile invalidation.

## The base model — a flat basin

A **base** is one flattened **basin** (a wide `TerrainFlatten` `BaseSite`) plus a
set of structures **draped** on it at the basin's single elevation `E`. Because
every structure is `Drape` at that constant radius, they are **coplanar by
construction** — there is no per-structure leveling, so they can't drift to
different heights. `E` is computed once from the natural ground over the basin
and the basin smoothstep-blends back to terrain over a wide ramp. The runway
scenario levels to the **mean** terrain over the basin (balanced cut/fill, so a
wide basin sinks into rising ground and fills hollows by roughly equal amounts
instead of forming an all-fill plateau); the editor's pick-a-site flatten still
uses `max + margin` for a small player-chosen pad.

**The default base** (authored by the runway scenario, `runway::finish_runway_spawn`
→ `base_editor::spawn_default_base`) is a spaceport with **two numbered runways
in a V**: the primary 5 km strip plus a shorter crosswind secondary that
diverges from near the primary's `−along` threshold at
`SECONDARY_HEADING_OFFSET_DEG` (30°) toward the empty side, opposite the launch
complex — the classic main-plus-diagonal layout (Dulles' 12/30 beside its
parallels). The basin is a single wide flat area offset toward the airside
(`BASIN_OFFSET_ACROSS_M`) and sized to clear both strips plus the complex.
`runway::build_spaceport` registers the two runways: the primary (centred) plus
the angled secondary (near threshold at `SEC_NEAR_ALONG_M`/`SEC_NEAR_ACROSS_M`,
fanning away so the strips never intersect) on the shared basin plane; both are
plain parametric `StructureKind::Runway { half_length_m, half_width_m }`
registry entries that render + collide through one generalized geometry path
(`runway.rs`), each painted with its designator (01–36; the heading divergence
gives each strip its own numbers, so no L/R suffix — `RunwayFrame::pair_side`
stays `0`, the lone-runway case) at both ends, from its true compass heading
(`runway_heading_deg` → `runway_designator`), rendered from the real **ICAO
runway font** (`assets/fonts/ICAORWYID.ttf`, rasterized with `ab_glyph` to an
alpha decal quad — `rasterize_designator` / `spawn_runway_numbers`). `spawn_default_base` then lays
out everything else on the basin (coplanar `Drape` at `E`) as **one core
campus** on the launch-complex side of the primary — nothing is authored
between the runways: a **large apron auto-derived from the hangar row**, with
the **row of hangars standing on it** (ramp all around); behind it an
operations building and per-pad blockhouses; then **two launchpads** with
blast-clear rings, each flanked by a **flame diverter** and a **fuel/tank
farm** (`StructureKind::Tank` cylinders); and a **VAB**-scale assembly building
(tagged the enterable `Facility::Vab`). The paving between all of it is
generated as a **typed connection network** (below). The surface scenarios
(`runway`/`landing`/`final`) present this whole base; the in-world editor
edits/extends the same `StructureRegistry` records.

## Why in-world (not a hangar scene)

Unlike the shipyard editor — a separate void *scene* that hides the flight world
— the base editor is an **in-world overlay**: the planet stays visible, the sim
pauses, a god-view camera looks down at the build site, and buildings are placed
on the actual flattened terrain. This is the Cities:Skylines feel, and it is
what forces the one genuinely new mechanism below (live terrain invalidation).
It reuses the shipyard's *patterns* (modal pause source, run-condition gating,
`HudTheme` UI, ghost-preview placement) but not its scene-swap half.

## Entry / lifecycle

`BaseEditor { open, mode, active_site }` is a resource and a **sim-clock pause
source** (`sim_clock::sync_sim_clock` reads it), not an `AppState`. Open it from
the pause menu's **SURFACE BASE** button; **Esc** closes it
(`pause_menu::handle_escape_input`, before the pause menu opens). While open:

- only the **Camera** `SimStage` set is gated off (`base_editor_closed` in
  `main.rs`) so the editor's own ungated god-view camera owns the view. The
  editor is a *warp-0-style* pause: the sim freezes via `SimClock`, but
  Physics/Sync keep running so the world keeps rendering + streaming terrain
  (gating Sync off black-holed the ground — the world must stay frozen-but-live);
- all gameplay input contexts deactivate
  (`input::gate_enhanced_input_sources` folds `BaseEditor::open` into
  `gameplay_suppressed`); the editor reads raw mouse/keyboard directly;
- `ViewMode` is forced to `Ship` on the open edge (the god-view is a 3-D view)
  and restored on close (`apply_open_state`, edge-detected via a `Local`);
- the flight HUD hides; the editor's own overlay (`ui.rs`) shows.

The two `BaseEditorMode`s are **PickSite** → **PlaceBuildings**; confirming a
site advances the mode and sets `active_site`.

## Camera (`camera.rs`)

No second camera: the existing `ShipCamera` (which carries the `FloatingOrigin`)
is repositioned to a 3/4 god-view over the focus, exactly like
`runway::update_runway_transform` — compute the heliocentric world position,
convert to a big_space `(CellCoord, local)` via the root `Grid`, set the camera
transform. The flight camera systems (`SimStage::Camera`) are gated off while
open, so they don't fight; they resume — snapping back to the ship — on close.
**Right-drag orbits, scroll zooms, WASD pans** the focus across the ground (pan
offset scales with zoom). The focus is `compute_focus` (mod.rs): the active
site's flattened centre, or the surface point under the ship while picking.

## Site pick → flatten (`pick.rs`)

Aim at the surface, see a ghost footprint that tracks the cursor (rotate with
Q/E), left-click to confirm. The cursor→surface raycast is the same analytic
trick as `debug::raycast_debug_surface_cursor`: render space and the
heliocentric world frame share axes (big_space cells are pure translations), so
we ray-vs-sphere the cursor against the body's render-space sphere
(`RealSpaceBody` GlobalTransform centre) and read the hit direction straight off
as a body-fixed direction.

Confirming scans the natural terrain over the footprint for its max height,
picks `E = max + margin`, registers a `StructureKind::BaseSite` `FlattenTo`
structure, applies the flatten through the shared `TerrainFlattenRegistry`
handle, and requests a terrain rebuild (below).

## Structure placement (`place.rs`)

Two tools (`Tool`, default **Select**):

- **Select / Move** (default — never places): left-click picks the structure
  under the cursor; with one held, **drag** repositions it (`reposition_structure`
  rewrites its `PlacedVisual` frame + registry anchor); **X**/**Delete** removes
  the selected one. Picking the **Select / Move** palette item returns here.
- **Place** (armed by clicking a palette item): a ghost tracks the grid-snapped
  cursor (Q/E rotate, `[ ]` / `- =` resize, **Tab** toggles building/launchpad),
  left-click places (stays armed for more), **right-click** cancels back to
  Select.

The picker palette + a status hint live in `ui.rs` (native Bevy UI, `HudTheme`).
A placed structure is a `StructureKind` record
(placement `Drape` — it sits on the already-level pad, no terrain modification of
its own): `Building { half_x_m, half_z_m, height_m }` → a `Cuboid`, or
`Launchpad { radius_m }` → a `Cylinder` slab with a yellow ring marking. Both
carry a `PlacedVisual` (`{ center_body, basis_body, kind }`) anchored every frame
in the body-fixed frame by `update_placed_transforms` — a root-grid big_space
child posed in f64, the runway pattern, so it stays rock-steady at high warp.
`update_placed_transforms` is **ungated** so structures stay anchored in flight
too, not just in the editor.

## Spawning (`place.rs`)

Spawnable structures have an **intrinsic spawn pose** derived from the structure:

- **Runway** → at the threshold, **horizontal**, nose down-runway, on its gear
  (`runway::place_parked` — used by the surface scenarios to seat the craft).
- **Launchpad** → at the centre, **vertical** (nose to local-up), on its engine
  end. In the editor, selecting a launchpad and pressing **L** spawns the craft
  there: it mirrors `place_parked` but uses `vertical_attitude` + the generalised
  `runway::craft_extent_below(-Y)` (rest on the rocket's lower end, not its
  belly), sets canonical state + a frozen `AuthorityMode::BodyFixed` pose, zeroes
  throttle, resets warp, and tears down the Avian bubble
  (`scenario_menu::clear_bubble`) so it rebuilds from the placed pose. The craft
  leaves the frozen pose the instant the pilot throttles up.

**The eventual flow** (`create base → build craft → fly from base`, deferred):
the shipyard's **Launch** will pick a base + a spawnable structure, build the
craft from the blueprint (reusing `relaunch`/`ship_view::build_player_ship`), and
seat it at that structure's spawn pose. For now spawning relocates the current
craft (editor **L** for a launchpad; the scenario seating for the runway).

## Auto-connections (`connections.rs`)

Connections are a **typed network**: every paved link is one `ConnectionKind` —
**Taxiway**, **Apron**, **Road**, or **Crawlerway** — each with its own width,
material, and ground lift (`ConnectionKind::style` / `::material`). Line networks
(taxiway / road / crawlerway) are flat strips of the kind's width along a set of
edges — either a **minimum spanning tree** (Prim's; n is small) or explicit
`(from, to)` edges. An **apron** is a filled rectangle (a hangar parking pad).
Every mesh is built in the site's local tangent frame and carried by a
`ConnectionVisual` anchored each frame like the structures (ungated, so it
persists in flight).

- **Editor** (`rebuild_connections`) — regenerates one taxiway MST over all the
  active site's structures when `structures_rev` bumps (place / delete / move).
- **Authored default base** (`spawn_default_base`) — builds several typed
  networks at spawn via `spawn_authored_network` (explicit edges),
  `spawn_authored_apron`, and `spawn_authored_path` (**curved fillet paths**: a
  waypoint polyline whose corners are rounded into circular arcs —
  `fillet_path` — then extruded to the kind's width by `build_path_mesh`).
  Airside is a real airport layout: the **core parallel taxiway** runs the
  primary's full length (straight), and the angled secondary's system hangs
  off it through three **curved link taxiways** sweeping across the V interior
  — each crossing is **split at the runway edges** (the core-side stub stops
  at one edge, the curve resumes at the other; no paving ever spans the strip,
  and both ends tuck 1 m under the runway's higher paving so the joints are
  seamless), then curves tangentially onto the secondary's parallel-taxiway
  line (the threshold link *is* that line's start; the midfield links merge
  into it at a hair-lower lift — `spawn_authored_path`'s `lift_bias_m`).
  Straight perpendicular connectors: the core-side halves of the link
  crossings, a threshold connector at the east primary end (through the
  **holding pads**, which fill the band between runway edge and taxiway),
  evenly-spaced exits between them, and near/midfield/far connectors on the
  secondary. The **core apron is
  auto-derived from the hangar row** — one large ramp the hangars stand *on*
  (it spans the row plus a parking margin and fills from the taxiway to behind
  the hangars' rear wall). Landside is one curved **road** path (ops →
  blockhouse → around the apron → across the VAB's doors → blockhouse), and the
  VAB→pad **crawlerway** rides explicit edges.

Adding a new infrastructure type (a utility pipe run, a rail spur) is a new
`ConnectionKind` variant plus a style/material — the routers and mesh builders
are kind-agnostic. The future **crawler-transporter** animation rides the
`Crawlerway` geometry this already lays down.

## Terrain data model (`structures.rs`)

Buildings ride the existing `StructureSite`/`StructureRegistry`/
`apply_structure_flatten` layer (the runway's home). This slice grew it:

- `StructureKind::Runway { half_length_m, half_width_m }` is **parametric**, so a
  base carries several runways (each its own size + heading), all rendered +
  collided through one generalized path in `runway.rs`.
- `StructureKind` also gained `BaseSite` (owns the flatten pad), `Building { … }`,
  `Launchpad { radius_m }`, and `Tank { radius_m, height_m }` (a vertical
  cylinder — the tank-farm stand-in; authored today, editable via the generic
  select/move/delete paths).
- `StructureSite` gained `parent_site: Option<StructureId>` (a structure's site).
- `StructureRegistry` gained `update(id, f)` and `remove(id)`;
  `remove_structure_flatten(id, …)` reverts a pad.

### Multi-flatten (`thalos_terrain::query`)

The body's flatten handle changed from `Arc<RwLock<Option<TerrainFlatten>>>`
(one region) to `Arc<RwLock<Vec<FlattenRegion>>>` (id-keyed regions), so a base
site and the runway pad coexist. `FlattenedSurface` applies the **single
highest-weight** region at any direction (pads are assumed not to overlap, so
no ramp stacking). `apply_structure_flatten` upserts by `StructureId`.
`nearest_flatten(regions, dir)` picks one representative pad for the tree/rock
scatter exclusion (those drivers want a single pad per dispatch). Grass no
longer reads it — see *Ground scatter* below.

## Ground scatter — lawn + clearings

A base's flattened ground is grassland (flat ⇒ a high terrain grass mask), so
the spaceport reads as a **grass lawn between the structures, bare paving under
them** rather than the blanket dead zone the old flatten-wide exclusion left
(which made the basin read as bare dirt with a hard grass edge). This is the
"scatter on the building terrain" layer, and the seam future trees/props plug
into.

- **Engine** (`body_render::ground::scatter`): `ScatterRegion { footprint,
  treatment }` + `ScatterTreatment::{ Clear, Lawn }` + `classify_scatter(regions,
  dir)`. A footprint reuses the flatten pad's tangent-plane rectangle SDF
  (`TerrainFlatten::weight`; the elevation is irrelevant here — only the
  rectangle is read). `Clear` wins over `Lawn` (a building on the lawn clears the
  grass under it), so a clearing is checked across every region before a lawn
  applies. Empty regions ⇒ `Natural` everywhere (off-base terrain untouched).
- **Grass build** (`build_grass_tile_mesh`): each candidate is classified —
  `Clear` ⇒ skip; `Lawn` ⇒ force the cover (bypass the natural grass-mask /
  treeline / landcover-coverage gates) with the managed `lawn_profile`
  (`GrassProfile::lawn` — short, thick, fluffy; the one place to retune the base
  look is `GRASS_PROFILE_LAWN` in `game::rendering::grass`); `Natural` ⇒ the
  existing meadow path. A lawn tile also places at a **higher density** under a
  raised candidate ceiling (`lawn_density_per_m2` = ring density ×
  `GRASS_LAWN_DENSITY_MULT`, capped by `MAX_LAWN_BLADES_PER_TILE`) — the lawn
  force-accepts every point, so a denser grid is what closes the gaps that
  otherwise read as patchy; scoped to lawn tiles, so wild/far grass cost is
  untouched.
- **Driver** (`game::rendering::grass::drive_grass_tiles`): derives the regions
  each frame from the `StructureRegistry` near the camera —
  `BaseSite` → `Lawn` (its flatten rectangle), and `Runway` / `Building` /
  `Launchpad` / `Tank` → `Clear` (their footprint + a small margin;
  `site_scatter_region`). Authored and player-placed bases both apply, since it
  reads the live registry.

**Caveat / follow-ups:** (1) Grass tiles already built don't re-clear when a
*Drape* structure (building/pad) is placed at runtime, because nothing bumps the
height-source revision — only a `FlattenTo` confirm (which triggers the terrain
rebuild) refreshes them; a scatter-region revision that re-dispatches nearby
grass tiles is the clean fix. (2) The lawn greens the *blades*, not the ground —
if a base's ground material is bare soil rather than grass, that mismatch needs a
terrain-material override under the lawn footprint (a separate `body_terrain`
change). (3) Trees/rocks still exclude on the basin via the flatten registry; a
`Lawn` that wants parkland trees (or a `Clear` for tarmac roads) is the natural
extension — thread `scatter_regions` into `build_scatter_tile` too.

## Live terrain invalidation — the one new mechanism

UDLOD bakes each resident tile **exactly once** and has no per-tile re-bake
path, so a flatten written *after* a tile is resident is invisible until the
tile is rebuilt. The runway never hits this (it flattens before its tiles
stream); a player flattening terrain they're looking at does.

**MVP (this slice):** `TerrainRebuildRequest`
(`rendering::terrain_residency`) + an **ungated** `apply_terrain_rebuild_requests`
that despawns and respawns the body's terrain entity at its current tier. The
respawn re-streams every tile through the body's *persistent*
`TerrainFlattenRegistry` handle (which already carries the new region), and the
GPU-atlas height mirror + surface-local collider follow via their existing
revision chain — no collider code. The system is ungated (plain `Update`, not in
`SimStage::Sync`) because the editor pauses the sim but the UDLOD streaming +
mirror sync it depends on keep running in `Last`; the `Sync`-gated residency
planner does **not** run while the editor is open, so the rebuild respawns
inline rather than handing a request to that planner.

This is acceptable because a flatten-confirm is rare and the world is paused; the
~1–2 s cold re-stream is the residency planner's normal budget, and LOD-0 is
pinned so there is no void-hole window.

**Follow-up (not built):** scoped-AABB invalidation
(`TileAtlas::invalidate_overlapping`) that re-queues only the tiles overlapping
the modified region instead of the whole body. Localizes to the `udlod` tile
atlas + the mirror/collider revision chain (which already keys off slot
revision).

## Persistence

In-session only: the `StructureRegistry`, the flatten handle, and the building
visual entities all survive scenario respawns / editor open-close within a run.
There is **no disk artifact** — a process restart loses the base. Disk
persistence (a gitignored `user/bases.ron`, loaded during `AppState::Loading`
so sites bake flattened pre-stream à la the runway) is a clean follow-up; design
it so loaded sites never need the runtime invalidation path.

## Launch-point selection (fly from base)

The shipyard/VAB **LAUNCH** is the default flight flow's entry: design a craft →
pick where to launch it → fly. Implemented as a **launch-point picker** — a third
base-editor mode, `BaseEditorMode::SelectLaunch`, living in
`base_editor/launch_select.rs`. It reuses the base editor's shared `god_view`
camera + `SimClock`-pause gating + cursor→body-fixed pick math; the existing
place/pick systems early-return on their mode, and the palette collapses to a
one-line hint (`ui::sync_overlay_for_mode`).

Flow (all game-**UNVERIFIED**):

1. The shipyard's `top_bar.rs` **LAUNCH** sets `RelaunchRequest{ShipOrbit}` (rebuild
   the craft into an orbit hold) **and** `base_editor::SpaceportLaunchRequest.arm`.
2. `begin_launch_flow` (gated on `relaunch_idle`, so it waits for the rebuild) opens
   the picker: **Case A** — the spaceport is already built (`RunwaySite` present, e.g.
   after PLAY→space-center) → open the god-view immediately; **Case B** — first launch
   → a brief **PLACEMENT-only** loading pass runs `runway::build_spaceport`, then the
   picker opens on `OnEnter(Running)`. (No SETTLE step — the craft is in orbit, so a
   settle gate keyed on its body-fixed point never resolves; the site's ground streams
   in live under the god-view.)
3. `update_launch_pick` raycasts the pad sphere and hit-tests **each** `Runway`
   (per-site `half_length_m`/`half_width_m` rectangle — both crossing runways are
   pickable) and each `Launchpad` (radius circle); a left-click latches the target.
   `apply_launch_placement` measures clearance (retrying until the rebuilt craft's
   meshes/gear are resident) and places it: a **runway** via `runway::place_on_runway`
   (horizontal on gear, parked inset from *its* threshold — `place_on_runway` now takes
   `half_length_m`; brakes+gear set; a `LaunchRelightEngines` one-shot lights the jets
   for throttle-only flight, since `enable_runway_engines` is `is_runway()`-gated), a
   **launchpad** via `place::place_on_launchpad` (vertical nose-up). Then warp→1× (the
   cores are warp-neutral) and the picker closes back to flight.

Shared cores were extracted so both the dev runway scenario and the picker produce the
identical result: `runway::build_spaceport` (site build minus craft place),
`measure_runway_clearance`, `place_parked`→`place_on_runway`, and
`place::place_on_launchpad` (from the L-key `launch_from_pad`). `RunwaySite` gained a
`basin_id` field (idempotency key + picker `active_site`).

Composes with the space-center hub: a VAB LAUNCH queues a relaunch, so
`space_center::restore_after_facility`'s `relaunching` branch drops to flight rather
than reopening the hub, and `begin_launch_flow` then opens the picker.

## Not built yet (ordered follow-ups)

1. **Launch-picker polish** — the picker (above) is in. Remaining: pre-highlight the
   sensible default by craft type (aircraft→runway, rocket→pad); smooth the ~2–3 frame
   flash of the orbit-hold view between VAB-close and picker-open (the relaunch takes 2
   frames); and a proper "launching…" overlay during the retry-until-clearance wait.
2. **Slider inspector** — a `HudTheme` slider panel that live-edits a *selected*
   structure's footprint. The picker palette exists; today only the *pending*
   footprint resizes (keyboard), not a placed structure.
3. **More connection types** — the typed network (taxiway / apron / road /
   crawlerway) is in; still open are resource/pipe lines, rail, **user-drawn
   routing** (vs. the automatic MST), and animating a **crawler** along the
   `Crawlerway` geometry. Taxiways currently reach the runways via two fixed
   complex-side exit nodes near the primary strip; per-runway exit taxiways
   (using the passed `_sec_heading`) are the natural refinement.
4. **Disk persistence** (above) — the authored default base re-spawns each run;
   player-built bases are in-session only.
5. **Scoped-AABB tile invalidation** (above).
6. **Structure colliders** — **launchpads** now get a kinematic `Cylinder`
   collider in `spawn_structure_entity` (posed like the runway via
   `sync_structure_collider_pose`) — required so a pad-launched craft doesn't fall
   through once a `RunwaySite` makes `local_physics::terrain_patch` skip the generic
   ground patch on that body. **Buildings + tanks** are still visual-only (EVA/craft
   pass through them); give them the same treatment next.
7. **Far-distance hide** — mirror `runway::sync_runway_visibility` so structures
   don't poke through the orbital impostor when zoomed way out.
8. **Launchpad polish** — countdown / hold-down clamp, measured per-craft pad
   sizing.
