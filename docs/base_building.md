# Surface base building

The in-world **base editor** is a Cities:Skylines-style tool for laying out a
surface base on a planet: pick a building site on the real terrain, flatten the
land into a level pad, then click-and-place / edit buildings on it. It is the
first realization of the player-placement gameplay that `surface_local.md` §6.4
designed for and deferred. Code lives in `crates/game/src/base_editor/`.

> **Status (2026-06-29).** Compiles clean; **not yet runtime-verified** by a
> `just game` launch. Implemented: site pick + live flatten, building + launchpad
> place/select/move(drag)/delete/rotate, keyboard sizing, **auto-connections
> (tarmac MST)**, an **authored default base** (the runway scenario spawns a wide
> flat *basin* holding a small spaceport — the runway plus a launch complex of
> two pads, tank farms, flame diverters, a VAB and hangars, all coplanar), and
> **spawning** (craft seat on a runway horizontal / a launchpad vertical).
> Deferred follow-ups (see *Not built yet*): the shipyard create-craft→fly flow,
> a slider inspector, road/resource connection types, disk persistence, scoped
> tile invalidation.

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
→ `base_editor::spawn_default_base`) is a small spaceport. The basin is a single
wide flat area **offset toward the launch-complex side of the runway** (so the
strip sits near one edge and the flattened ground isn't wasted on the empty
side), sized to hold the whole layout. The runway is the *first structure on the
basin* — it stops computing its own pad and drapes on the basin at `E`. Beside it
the complex carries **two large launchpads** with clearing around them, each
flanked by a **flame diverter** (a low concrete trench stand-in) and a small
**propellant tank farm** (`StructureKind::Tank` cylinders); a **VAB**-scale
assembly building and a pair of **hangars** line the far edge, with an operations
building and per-pad blockhouses near the strip. A tarmac MST links the runway,
pads, big buildings and blockhouses (the satellite tanks/diverters stay off the
road network). Everything drapes coplanar on the basin at `E`. The surface
scenarios (`runway`/`landing`/`final`) present this whole base; the in-world
editor edits/extends the same `StructureRegistry` records.

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

A single combined **tarmac** mesh connects every building / launchpad along a
**minimum spanning tree** (Prim's; n is small) — the least paving that links them
all. Each edge is a flat strip from one structure's footprint edge to the next's
(inset by their bounding radii), built in the site's local tangent frame and
carried by a `ConnectionVisual` anchored every frame like the structures (ungated,
so it persists in flight). The in-editor `rebuild_connections` regenerates it when
the active site's `structures_rev` bumps (place / delete / move); the authored
default base builds its tarmac once at spawn via `spawn_authored` (sharing the
`BaseMaterials::tarmac` material + the same MST/strip code). One type (tarmac) for
now; roads / resource lines + user-drawn routing extend the same seam.

## Terrain data model (`structures.rs`)

Buildings ride the existing `StructureSite`/`StructureRegistry`/
`apply_structure_flatten` layer (the runway's home). This slice grew it:

- `StructureKind` gained `BaseSite` (owns the flatten pad), `Building { … }`,
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

## Not built yet (ordered follow-ups)

1. **Shipyard "fly from base" flow** — the big one: the shipyard's **Launch**
   picks a base + a spawnable structure (runway/pad), builds the craft from the
   blueprint (`relaunch`/`ship_view::build_player_ship`), and seats it at that
   structure's spawn pose via a unified `spawn_craft_at`. Today spawning relocates
   the *current* craft (editor **L** for pads; the scenario seating for runways);
   runways aren't yet selectable-for-spawn in the editor (they're not `PlacedVisual`).
2. **Slider inspector** — a `HudTheme` slider panel that live-edits a *selected*
   structure's footprint. The picker palette exists; today only the *pending*
   footprint resizes (keyboard), not a placed structure.
3. **More connection types** — roads / resource lines as distinct
   `ConnectionVisual` kinds, and user-drawn routing (vs. the automatic MST).
4. **Disk persistence** (above) — the authored default base re-spawns each run;
   player-built bases are in-session only.
5. **Scoped-AABB tile invalidation** (above).
6. **Structure colliders** — static `Collider::cuboid`/`cylinder` posed like the
   runway, so EVA/craft collide with buildings + launchpads (today visual only).
7. **Far-distance hide** — mirror `runway::sync_runway_visibility` so structures
   don't poke through the orbital impostor when zoomed way out.
8. **Launchpad polish** — countdown / hold-down clamp, measured per-craft pad
   sizing.
