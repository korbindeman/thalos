# Kòrsou

A lightweight, playful Bevy 0.19 explorer built over real Curaçao terrain.
Kòrsou is intentionally flat, direct, and small: it is a place to fly around the
island, enjoy the view, and try visual ideas without becoming a full simulation.
It lives in the Thalos workspace so both applications can reuse renderer leaves,
and it is moving onto `thalos_runtime`'s lightweight interactive bundle for the
normal freecam/UI, viewpoints, settings, diagnostics, and capture machinery.
Simulation and gameplay remain disabled and absent from its dependency graph,
so it does not become a reduced planetary game.

The real-world data path starts from public Copernicus GLO-30 COG GeoTIFFs.
The baker reads their georeferencing, reprojects horizontal positions into WGS
84 / UTM zone 19N (EPSG:32619), and records the source's EGM2008 orthometric
vertical datum (EPSG:3855). Runtime can place that same dataset in two explicit
spatial adapters: `planar` recenters UTM metres; `ellipsoid` converts UTM +
orthometric height through WGS84/EGM2008 → ECEF → a bounded local ENU frame.
Bevy `+X` is east, `+Y` is up, and `-Z` is north in both modes, so the world
remains right-handed. A 61.44 km quadtree selects
terrain from baked screen-space error and generates each leaf with an RTIN mesh.
The standard terrain material combines nonrepeating vertex-colour soil/rock
breakup with a mipmapped tangent-space detail normal, while several bounded
synthetic relief bands carry shape across the visible near LODs. All of that
detail exists only in the rendered surface; CPU height queries always return
the unmodified DEM height.

Foliage uses one deterministic canopy field derived from terrain slope and
shoreline distance. Nearby vegetation is dense, minimum-spacing scrub and
broadleaf scatter built from Thalos's shared procedural woody payload and
batched into planar 128 m cells. Every accepted root is rendered through the
shared hemisphere-octahedral impostor mechanism: four vertices and two triangles
regardless of source-tree complexity, with the atlas view selected from aerial
or oblique camera direction. The visible cards remain outside Bevy's cascaded
shadow caster set because an opaque card would cast its rectangular bounds.
Trees instead contribute one coarse shadow-only crown proxy inside a bounded
760 m camera radius; shrubs do not. Coarser terrain LODs retain a restrained
canopy proxy while close terrain yields to the real geometry, so green
represents foliage rather than green soil. Beaches and the immediate shoreline
remain clear. F3 reports root count, compact vertex count, atlas readiness, and
the active shadow-cell/triangle budget.

The shoreline is independent of the DEM. OpenStreetMap `natural=coastline`
ways are assembled into closed rings, reprojected to UTM 19N, and baked into a
15 m signed-distance field. Mixed coastline tiles always use the fixed visual
LOD. Their zero-contour edges are projected back onto that field and subdivided
to at most 3 m before the land triangles are clipped. Gentle measured ground
meets sea level; steep elevated ground ends in a dedicated coast face that
continues below wave troughs. Synthetic rock displacement fades in from 8 to
36 m inland instead of redrawing the authored coast lip. Camera LOD therefore
cannot change the island silhouette or create ocean wedges.

The sea is Kòrsou's planar adapter over `thalos_ocean`. Thirteen
camera-centered clipmap levels provide real nearby displacement, while the
shared module supplies the canonical sea-state projection, precision-safe wave
clock, resolved wave shape, filtered spectrum, and omitted-variance handoff.
Kòrsou retains the pieces specific to this place: the baked signed shoreline,
coastal exposure/cliff properties, shelf colour, run-up, impact foam, and Bevy
PBR integration. See the canonical [ocean rendering
contract](../../docs/rendering/ocean.md).

The sky and aerial perspective use the `thalos_atmosphere` Bevy Earth adapter
at a lower-density planar calibration. Bevy's atmosphere generates the
environment lighting used by terrain and water, so haze, ambient light, and
water reflections share the same sun and sky. A local solar clock evaluates the
real Curaçao latitude, longitude, date, and AST time, so the visible atmospheric
sun, directional shadows, generated environment lighting, and ocean reflection
all follow one trajectory. Interactive runs default to August 10 at 15:30 AST
and advance at 60×; headless capture freezes that deterministic instant unless
`--time HH:MM` overrides it. Thalos's planetary application
uses the same authored-atmosphere leaf through its custom scene-depth-aware
adapter; see the canonical [atmosphere
contract](../../docs/rendering/atmosphere.md).

## Run

```bash
just korsou
```

The equivalent direct command is
`cargo run -p korsou --features dev-renderer`.

The default is `--spatial planar`. Use `--spatial ellipsoid` to render the same
terrain through the datum-aware ellipsoid tracer; headless captures accept the
same flag.

Controls:

The app starts in the western low-flight viewpoint, about 620 m above terrain.

- Hold left mouse: look
- `WASD`: move in the camera frame; `R`/`F`: rise/descend; `Q`/`E`: roll
- Mouse wheel: change the persistent cruise speed
- Shift/Ctrl: fast/slow speed modifier
- `L`: toggle level-to-local-up; `C`: toggle the DEM ground floor
- Hold `Z`: spring zoom through the shared physical lens model
- `1`: island aerial; `2`: western low flight; `3`: eastern low flight
- `[` / `]`: previous / next geographic waypoint; `X`: clear the waypoint
- `F9`: quick-save the exact camera and lens; Enter accepts the suggested name
- `F8`: open the shared viewpoint manager; view, rename, replace, or delete entries
- `F1`: toggle photo mode, keeping the viewer movable while hiding all overlays
- `F2`: save a clean, UI-free PNG under `screenshots/`
- `F3`: toggle the shared Thalos frame/GPU diagnostics panel, extended with
  Kòrsou streaming and projected position
- `F10`: open the shared Window / Graphics / World settings; World controls the
  local date, time, running state, and cycle rate; Escape closes it

## Viewpoints and headless captures

The viewpoint manager starts with the three camera presets above plus five
coastal references used to evaluate water work:

- `Reference - Grote Knip beach`: sand, protected shallows, and beach run-up
- `Reference - Boka Tabla cliffs`: exposed rock and wave impact
- `Reference - Blue Bay reef`: shallow reef and depth-dependent water colour
- `Reference - Caracasbaai close coast`: close waterline and nearshore detail
- `Reference - North coast waves`: 8 m open-water wave silhouette and foam

The first saved camera updates `apps/korsou/viewpoints.json` by default and
includes those defaults. F9 suggests a Curaçao + altitude name; Enter accepts
it, or typing replaces it. The file uses the same frame-tagged v3 catalog as
the game, while each application keeps its own authored entries. It is plain
JSON so tools and agents can inspect or edit the same catalog as the player.
Use `--viewpoints FILE` to select a different catalog.

## Places and waypoints

The freecam panel's **Location** row follows the nearest curated named area:
specific places such as Punda, Otrobanda, Pietermaai, and Caracasbaai win over
their broader Willemstad or Bandabou region. These labels are presentation
regions around attributed OpenStreetMap place coordinates, not legal or
administrative boundaries.

`[` and `]` cycle a separate geographic waypoint list covering beaches, bays,
landmarks, and lookouts; `X` clears it. The waypoint panel reports horizontal
distance and UTM grid-north bearing in the stable local frame. Its amber marker
is projected from the geographic surface point through the active planar or
ellipsoid spatial adapter, so the same destination remains correct in either
mode. F2 continues to produce a clean image by hiding the place UI with the
other viewer surfaces.

The checked-in [`assets/places.json`](assets/places.json) catalog keeps WGS84
coordinates and OSM element identities as the auditable source. Runtime
projects the catalog once into the same EPSG:32619 local frame as the terrain.
Saved viewpoints remain camera compositions; a waypoint remains a real place.
The authored Grote Knip and Boka Tabla camera references now resolve their
anchors through this catalog. Blue Bay deliberately retains its distinct reef
survey point because the OSM beach centroid is on a different part of the bay.

Capture a saved viewpoint without creating a window or requiring a display
server:

```bash
cargo run -p korsou --features dev-renderer -- \
  capture artifacts/korsou/aerial.png --viewpoint "Island aerial" \
  --time 17:30
```

Capture an arbitrary camera by supplying its position and look-at point in
local metres:

```bash
cargo run -p korsou --features dev-renderer -- \
  capture artifacts/korsou/custom.png \
  --position -15000,8000,25000 \
  --look-at 0,50,0 \
  --spatial ellipsoid \
  --size 1920x1080
```

The headless command waits for terrain streaming and morphing to settle, writes
a clean PNG, and exits. Coordinates use `+X` east, `+Y` up, and `-Z` north.
Run `cargo run -p korsou -- --help` for the compact command reference.

## Rebuild the terrain assets

The checked-in source rasters are the two public Copernicus GLO-30 COG tiles
covering Curaçao. The checked-in Overpass JSON is the attributed OSM coastline
snapshot used by the baker. Rebuild the projected assets with:

The pure-Rust baker does not require GDAL. It reads GeoTIFF georeferencing,
crops to the Curaçao explorer bounds, reprojects to WGS 84 / UTM zone 19N
(EPSG:32619), emits little-endian `f32` orthometric-height grids and offline
LODs, and records the source, projected, ellipsoid, and vertical CRS plus the
explicit `h = H + N` height relation in `metadata.json`. It also
bakes the maximum approximation error for native terrain nodes, land/water
coverage through visual level 6, and the quantized shoreline field. Runtime LOD
selection projects the metre errors into pixels while keeping mixed coast tiles
at the fixed shoreline level.

An OpenTopography download of the same product can be used instead:

```bash
cargo run -p korsou_terrain_baker -- apps/korsou/assets/terrain/curacao \
  --coastline apps/korsou/data/source/curacao-coastline-osm.json \
  apps/korsou/data/source/Copernicus_DSM_COG_10_N12_00_W069_00_DEM.tif \
  apps/korsou/data/source/Copernicus_DSM_COG_10_N12_00_W070_00_DEM.tif
```

OpenTopography's clipped API requires a user key, so the repository source data
uses the credential-free public AWS mirror of the identical 2021 product.

## Honest limits

- GLO-30 is a 30 m digital surface model. Interpolation and mesh refinement do
  not create measured terrain detail.
- The erosion-like relief and material detail are explicitly synthetic. DEM
  grade and shoreline distance keep low beaches smooth while activating
  multi-scale fractured relief on steep ground. Relief fades through coarser
  LODs; repeating material textures are a visual-frequency layer rather than
  geographic evidence.
- RTIN reduces triangles according to height error. The closest visual level
  samples a 7.5 m grid, and a bounded refinement band preserves the authored
  shoreline through that mesh, so near ground does not collapse into visibly
  coarse triangles. Neither refinement creates measured sub-30 m elevation.
- The OSM coastline is substantially better than a DEM threshold, but its
  piecewise-linear rendered contour is limited by the 15 m shoreline field.
  Tides, exact beach profiles, cliffs, salt flats, and inland water still need
  dedicated data and material rules.
- Place labels use curated point coverage and specificity, not polygonal
  neighbourhood boundaries. They are meant to answer "what area am I near?",
  not to perform cadastral or administrative reverse geocoding.
- Vertex colours approximate brown soil and rock from height and slope. Sand is
  limited by shoreline distance, slope, and lowland dryness. A shared procedural
  canopy field drives both dense nearby woody geometry and restrained distant
  foliage color, but satellite imagery, vegetation surveys, or land-cover data
  should eventually replace that classification.
- Foliage shadows use deliberately coarse crown volumes within 760 m. They
  ground nearby trees without submitting every alpha card to every cascade, but
  they do not reproduce leaf-shaped penumbrae and shrubs cast no shadow.
- The turquoise shelf is a visual distance-to-land treatment, not measured
  bathymetry or simulated optical depth. Beach/cliff and wave-exposure signals
  are explicit heuristics derived from the 30 m DEM and coastline field; they
  should be replaced by surveyed coastal classes and bathymetry when available.
- The local recentered frame keeps render coordinates within tens of kilometres.
  That flat projected frame is a product decision, not an interim substitute for
  a planetary coordinate system.

See [LICENSING.md](LICENSING.md) for the required Copernicus and OpenStreetMap
attribution.

See [CONTEXT.md](CONTEXT.md) for the project boundary and module map.
The workspace decision is recorded in
[ADR-20260808T205119Z](../../docs/adr/20260808T205119Z-korsou-second-application-render-kit.md),
and atmosphere/ocean adapter ownership in
[ADR-20260808T221912Z](../../docs/adr/20260808T221912Z-atmosphere-and-ocean-mechanisms-use-spatial-adapters.md).
The lightweight shared-runtime direction superseding the first decision is
[ADR-20260809T201216Z](../../docs/adr/20260809T201216Z-light-runtime-capability-bundles.md).
The bounded standard-path foliage-shadow policy is
[ADR-20260810T201029Z](../../docs/adr/20260810T201029Z-korsou-foliage-shadows-use-bounded-proxies.md).
