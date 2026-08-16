# Kòrsou context

Kòrsou is a lightweight, playful explorer built over real Curaçao terrain. It
uses a flat projected world by default, plus a deliberate geodetic-ellipsoid
tracer over the same real dataset. The goal is to move around a familiar island
quickly while proving that spatial representation can change without forking
terrain truth or appearance mechanisms.

Kòrsou is orthogonal to the Thalos simulation while living in the same Cargo
workspace. Its target composition is the lightweight
`thalos_runtime[interactive]` shell plus reusable rendering modules and its
explicit planar adapter. Window/display/UI-scale settings, the settings modal,
and MSAA are already shared through `thalos_preferences`; the bespoke
camera movement, optics, and freecam panel have been replaced by
`thalos_viewer`, as have viewpoint validation, persistence, CRUD, and F8/F9 UI.
The projected camera/viewpoint adapters and capture module remain
application-owned until APP-4–APP-5 finish their seams. Simulation and
gameplay capabilities stay disabled and absent from its dependency graph, so it
remains a small second application rather than a reduced Thalos or a staging
area for simulation features. Thalos owns full-scale simulation; Kòrsou owns
direct, low-friction exploration of one real place.

## Product principles

- **Real place, honest data.** Preserve source attribution, coordinate reference
  systems, and the boundary between measured data and synthetic visual detail.
- **Spatial representation is explicit.** `planar` uses local EPSG:32619 metres;
  `ellipsoid` converts the same UTM + EGM2008 orthometric samples through WGS84
  geodetic/ECEF into local ENU. Both narrow to f32 only inside a bounded local
  frame.
- **Exploration first.** Fast movement, saved viewpoints, and reproducible
  captures matter more than simulation depth.
- **Lightweight and legible.** Prefer deterministic fields, shared assets, and
  cell batches over a general ecosystem or planetary streaming framework.
- **Visually continuous.** Nearby detail should hand off to cheaper distant
  representations instead of ending visibly.
- **Renderer fidelity is shared.** Kòrsou and Thalos may use different spatial
  adapters, but terrain/material detail, solar lighting, shadows, atmosphere,
  and exposure should read at mostly consistent fidelity. Product separation is
  not permission for a visibly second-class renderer.

## World contract

- Source elevation: Copernicus GLO-30 GeoTIFF/COG in EPSG:4326.
- Projected terrain: WGS 84 / UTM zone 19N, EPSG:32619.
- Ellipsoid: WGS 84 geographic 3D, EPSG:4979 / EPSG:7030.
- Vertical datum: EGM2008 height, EPSG:3855; `h = H + N` is explicit.
- Coastline: attributed OpenStreetMap geometry projected into the same CRS.
- Runtime axes: `+X` east, `+Y` up, `-Z` north; one unit is one metre.
- CPU terrain queries return DEM truth. Synthetic relief plus mipmapped
  albedo/normal detail change only the rendered surface and are explicitly
  documented as synthetic.
- Foliage is a visual canopy field. Dense nearby cards and distant terrain tint
  sample the same field; it is not an ecological simulation or ground color.
  Visible cards do not cast opaque rectangular shadows. Deterministic tree roots
  instead feed coarse shadow-only crowns inside a bounded 760 m radius.

## Module map

- `tools/korsou_terrain_baker`: reads georeferenced GeoTIFFs, reprojects them,
  and bakes terrain, LOD error, coastline distance, and explicit CRS/datum
  metadata.
- `thalos_geodetic` + `spatial`: typed UTM/geodetic/ECEF/ENU/EGM2008 conversion
  and the two concrete runtime placement adapters.
- `places`: owns the attributed WGS84 place catalog, its one-time projection
  into local UTM metres, current-area classification, waypoint state, and the
  lightweight destination readout/marker. It does not depend on game
  navigation or turn saved camera viewpoints into POIs.
- `terrain`: owns DEM truth, adaptive planar terrain, rendered multi-band relief,
  tangent-space material detail, and the shared canopy field.
- `foliage`: adapts `thalos_vegetation` woody appearance payloads into dense,
  deterministic planar plant batches driven by the local canopy field, plus the
  bounded planar crown-shadow adapter.
- `ocean`: adapts `thalos_ocean` wave geometry, spectrum filtering, and
  precision-safe phase projection onto a camera-centered displaced clipmap;
  the baked shoreline products remain the local coast authority.
- `thalos_preferences` through `thalos_runtime[interactive]`: the common
  persisted window/UI-scale/MSAA/foliage model and F10 settings host. Kòrsou's
  planar foliage adapter registers the shared foliage control; it contributes
  no game-only controls.
- `thalos_viewer` through `thalos_runtime[interactive]`: the common freecam
  intent/motion, physical optics, panel, level/ground/speed controls, and the
  frame-tagged viewpoint store plus F8/F9 UI.
- `camera`: Kòrsou's planar adapter around the shared viewer. It alone owns DEM
  floor, map bounds, and planar/ellipsoid render projection.
- `viewpoint`: Curaçao default compositions plus the projected-local
  snapshot/apply adapter; no schema, persistence, or UI.
- `thalos_diagnostics_ui` through `thalos_runtime[interactive]`: the one F3
  toggle, wall-clock CPU/GPU history, renderer/process/scene facts, panel, and
  graph used by both applications.
- `thalos_photo_mode` through `thalos_runtime[interactive]`: the one F1
  clean-view state and visibility arbiter shared with the game. Kòrsou owns
  only the raw-key adapter and keeps viewer movement active while overlays are
  hidden.
- `diagnostics`: Kòrsou's typed extension supplies only planar terrain/foliage
  streaming, projected position, modal availability, and the clean-capture
  marker. It imports no game diagnostics or simulation systems.
- `capture`: transitional application-shell capture implementation, replaced
  and deleted when APP-5 lands. All visible UI comes from the shared viewer,
  viewpoint, preferences, and diagnostics crates.
- `world`: uses the `thalos_atmosphere` Bevy Earth adapter and owns the
  Curaçao/date-aware solar clock. One direction drives the atmospheric sun,
  direct light, generated environment lighting, shadows, and water reflection;
  its F10 World page controls time, date, run state, and rate.

## Out of scope

Orbital mechanics, a general biome simulator, ecological succession, and a
generic GIS workbench belong elsewhere. Additional real-world
layers are welcome when they make Curaçao more recognizable without compromising
the small explorer-shaped architecture.
