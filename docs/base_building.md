# Surface base building

The in-world **base editor** is a Cities:Skylines-style tool for laying out a
surface base on a planet: pick a building site on the real terrain, flatten the
land into a level pad, then click-and-place / edit buildings on it. It is the
first realization of the player-placement gameplay that `surface_local.md` §6.4
designed for and deferred. Code lives in `crates/game/src/base_editor/`.

> **Status (2026-06-29).** Compiles clean; **not yet runtime-verified** by a
> `just game` launch. Implemented: site pick + live flatten, building + launchpad
> place/select/delete/rotate, keyboard sizing, **launch (place the ship on a
> launchpad)**, **auto-connections (tarmac MST between structures)**, in-session
> persistence. Deferred follow-ups (see *Not built yet*): a full slider inspector
> + parts palette, road/resource-line connection types, disk persistence, and
> scoped-AABB tile invalidation.

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

- the three `SimStage` sets are gated off (`base_editor_closed` in `main.rs`),
  freezing flight logic and the flight camera so the editor's own ungated
  god-view camera owns the view (the world is frozen-but-visible);
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
Right-drag orbits, scroll zooms. The focus is `compute_focus` (mod.rs): the
active site's flattened centre, or the surface point under the ship while
picking.

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

In `PlaceBuildings`, a ghost tracks the grid-snapped cursor on the pad (Q/E
rotate, `[ ]` / `- =` resize). **Tab** toggles the pending kind between a
**building** (box ghost) and a **launchpad** (ring ghost). Left-click on empty
pad places it; left-click on an existing structure selects it; **X** / **Delete**
removes the selected one. A placed structure is a `StructureKind` record
(placement `Drape` — it sits on the already-level pad, no terrain modification of
its own): `Building { half_x_m, half_z_m, height_m }` → a `Cuboid`, or
`Launchpad { radius_m }` → a `Cylinder` slab with a yellow ring marking. Both
carry a `PlacedVisual` (`{ center_body, basis_body, kind }`) anchored every frame
in the body-fixed frame by `update_placed_transforms` — a root-grid big_space
child posed in f64, the runway pattern, so it stays rock-steady at high warp.
`update_placed_transforms` is **ungated** so structures stay anchored in flight
too, not just in the editor.

## Launchpad (`place.rs`)

A launchpad is a `Launchpad` structure plus a **launch** action: with a launchpad
selected, **L** places the player ship on it at rest and closes the editor. This
mirrors `runway::place_parked` — it reuses the (now `pub(crate)`) runway helpers
`level_heading_attitude` and `craft_ground_clearance` (to rest *any* craft on the
pad), then sets the canonical ship state + a frozen `AuthorityMode::BodyFixed`
pose (via the public `body_fixed` helpers), zeroes throttle, resets warp, and
tears down the live Avian bubble (`scenario_menu::clear_bubble`) so it rebuilds
from the placed pose. The craft leaves the frozen pose the instant the pilot
advances throttle, exactly like the runway. Runtime-placement caveat: the pad
terrain must be streamed (it is — it's the dominant body the player is on); a
brief tile-settle is possible if the pad was just flattened.

## Auto-connections (`connections.rs`)

When the active site's structure set changes, a single combined **tarmac** mesh
is rebuilt connecting every building / launchpad along a **minimum spanning
tree** (Prim's; n is small) — the least paving that links them all. Each edge is
a flat strip from one structure's footprint edge to the next's (inset by their
bounding radii). The mesh is built in the site's local tangent frame and carries
its own `ConnectionVisual`, anchored every frame like the structures (and ungated,
so it persists in flight). Rebuild triggers on the structure *count* changing
(place/delete), the only edit the foundation editor makes to a site. One type
(tarmac) for now; roads / resource lines + user-drawn routing extend the same
MST + strip-mesh seam.

## Terrain data model (`structures.rs`)

Buildings ride the existing `StructureSite`/`StructureRegistry`/
`apply_structure_flatten` layer (the runway's home). This slice grew it:

- `StructureKind` gained `BaseSite` (owns the flatten pad), `Building { … }`, and
  `Launchpad { radius_m }`.
- `StructureSite` gained `parent_site: Option<StructureId>` (a structure's site).
- `StructureRegistry` gained `update(id, f)` and `remove(id)`;
  `remove_structure_flatten(id, …)` reverts a pad.

### Multi-flatten (`thalos_terrain::query`)

The body's flatten handle changed from `Arc<RwLock<Option<TerrainFlatten>>>`
(one region) to `Arc<RwLock<Vec<FlattenRegion>>>` (id-keyed regions), so a base
site and the runway pad coexist. `FlattenedSurface` applies the **single
highest-weight** region at any direction (pads are assumed not to overlap, so
no ramp stacking). `apply_structure_flatten` upserts by `StructureId`.
`nearest_flatten(regions, dir)` picks one representative pad for vegetation
exclusion (the grass/tree drivers, which want a single pad per dispatch).

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

1. **Full inspector + parts palette** — a `HudTheme` slider panel (mirroring
   `shipyard_editor/ui/`) that live-edits a *selected* structure's footprint and
   a palette of typed buildings, replacing the keyboard-only pending sizing.
2. **More connection types** — roads / resource lines as distinct
   `ConnectionVisual` kinds, and user-drawn routing (vs. the automatic MST).
3. **Disk persistence** (above).
4. **Scoped-AABB tile invalidation** (above).
5. **Structure colliders** — static `Collider::cuboid`/`cylinder` posed like the
   runway, so EVA/craft collide with buildings + launchpads (today visual only).
6. **Far-distance hide** — mirror `runway::sync_runway_visibility` so structures
   don't poke through the orbital impostor when zoomed way out.
7. **Launchpad polish** — countdown / hold-down clamp, measured per-craft pad
   sizing, and integrating relaunch (shipyard **Launch**) to seat onto a pad.
