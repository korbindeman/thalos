# Macro terrain: landcover, biomes, landmass

Plan of record for **large-scale terrain generation** — everything you see from
cruise altitude and orbit: the landcover mosaic, biomes/climate, and the
continent/island structure. Written 2026-07-18 after a diagnosis of the
"repeating leopard-print planet" problem; Phase 1 below is implemented, Phases
2–3 are design.

Companion docs: `terrain.md` (consumer-side tile contract),
`docs/vegetation.md` (scatter cascade), `docs/graphics_fidelity.md` (shading).
The generator itself is `thalos_terrain::procedural::ProceduralSurface`.

## 1. Diagnosis: why the planet read as one repeating texture

Screenshots from altitude showed a uniform mottled green pattern stamped
planet-wide (dark forest blotches on lighter grass at a fixed ~0.5–1 km scale,
with visible diagonal streaking). Root causes, in order of weight:

1. **The landcover field literally tiled every 4 km.** The visible mottle was
   the per-pixel moisture/vegetation field in `body_terrain.wgsl`, computed
   from f32 body-fixed coordinates wrapped modulo
   `DETAIL_COORD_PERIOD_M = 4000` (an f32-precision necessity at planet radius).
   Its largest wavelength was capped at ~1 km explicitly "to avoid visible
   tiling" — so the largest landcover feature on the planet was ~1 km,
   repeating exactly every 4 km.
2. **The baked macro albedo had no lateral variation.** The f64 side
   (`ProceduralSurface::albedo_at`), which has unlimited wavelengths and bakes
   into the tile albedo pyramid, was a pure altitude-band function. Two points
   at the same altitude 2,000 km apart baked identical colour, so nothing at
   the 10 km–1,000 km scale broke the wallpaper.
3. **Height relief was spatially uniform.** Hills/swell amplitudes are
   constant on all land; the only regional differentiator is the 420 km
   orogeny field.
4. **Value-noise lattice "weave".** The low-octave value-noise fBm produced
   axis/diagonal correlation (the same artifact texgen's bark hit; fixed there
   with gradient noise).

## 2. The principle: scale ownership

There are two noise domains with different powers:

| Domain | Precision | Wavelengths | LOD behaviour |
|---|---|---|---|
| CPU tile bake (`ProceduralSurface`, f64) | exact at planet scale | unlimited | baked into the tile pyramid → auto anti-aliased with distance |
| Shader per-pixel (f32, wrapped at 4 km) | 4 km period | honestly ≤ ~200 m | evaluated live, must fade itself out |

**Rule: everything ≥ ~250 m wavelength must come from the f64 bake; the
wrapped shader field may only carry fine detail (≤ ~125 m) as a *modulation*
of the baked value.** A wrapped tier is invisible from the ground (a 4 km
repeat of ≤200 m features doesn't register) and fades before the repeat could
be seen from the air. Asking the wrapped domain to carry km-scale biome
structure is a structural error — it cannot, at any tuning.

## 3. Phase 1 — macro landcover from the f64 field (IMPLEMENTED 2026-07-18)

The macro moisture/landcover field moved into `ProceduralSurface` and is
delivered to every consumer through the existing seams:

- **`ProceduralSurface::macro_moisture`** (f64, gradient noise — no weave):
  three decorrelated tiers — climatic provinces (~700 km), regional mosaic
  (~90 km), stand/valley patchiness (~9 km, LOD-aware octaves cascading to
  ~0.5 km). Output in `[-1, 1]` (+ wet, − dry), same semantics the shader
  already used.
- **`SurfaceSample.moisture`** carries it through the `SurfaceQuery` seam;
  `SurfaceQuery::landcover_moisture(dir)` (default 0) exposes point queries.
  `HeightSource::landcover_moisture` delegates to the wrapped surface, so
  both grass paths get it with no new plumbing.
- **Baked into the albedo attachment's alpha channel** by
  `PipelineTileProvider` (`Rgba8UnormSrgb` → alpha stays linear; mips average
  it correctly). The terrain shader decodes `moisture = albedo.a * 2 − 1` and
  adds only the remaining fine 125 m wrapped tier
  (`thalos::landcover::moisture_detail`). The shader forces output alpha to 1
  (the attachment alpha is no longer opacity).
- **`albedo_at` now varies laterally**: lowland palette blends lush ↔ dry by
  macro moisture plus a ~30 km value-tone mottle, so the distant-body
  impostor and the 10 % macro tint agree with the ground's regions.
- **WGSL/CPU mirrors slimmed**: the 1 km/500 m wrapped tiers are deleted from
  `thalos::landcover` and its CPU mirror (`ground/landcover.rs`); both keep
  only the fine tiers. The CPU mirror now takes the macro value as a
  parameter instead of re-deriving it — shrinking the keep-in-sync surface.
- **Grass agrees with the ground**: the CPU blade builder samples
  `HeightSource::landcover_moisture` per clump; the GPU grass window carries a
  per-window macro value (`GpuGrassParams.phase.w`, sampled at the anchor —
  the finest macro tier is ~20× the window size, so a scalar suffices).
- `GENERATOR_VERSION` bumped (tile/disk caches invalidate by key).

Verification: `just map` (the `world_map` example) renders the planet in the
**true in-game macro palette** (`sample_d().albedo_linear`) plus a flat
`MacroBiome` class map with area-weighted per-biome coverage stats —
web-mercator by default, `WORLD_PROJ=equirect` / `WORLD_MODE=hypso` for the
legacy frames; `just screenshot` presets for the in-game view. The class map
cannot drift from the render: `ProceduralSurface::sample_biome_d` classifies
from the same `macro_band_ts` evaluation `albedo_at` blends.

### Follow-ups deliberately left out of Phase 1

- **Tree placement now tracks moisture + treeline** (landed 2026-07-20, TM-P2r.1).
  The woody branch of `scatter.rs::build_scatter_tile` multiplies its noise stand
  field (`forest_coverage`) by `woody_biome_gate(layer, moisture, eco_altitude)`:
  a moisture dryness ramp (trees gone by the ground's bare-soil threshold, shrubs
  a touch hardier) × a cold-lift-descended treeline term (`height +
  climate_cold_lift`), keyed off the SAME eco-band constants the ground palette
  uses. So trees thin on the dry-tan steppe, vanish on the bare desert, and stop
  at the poles — agreeing with the ground's `vegetation_color`. Grass *type* per
  biome is the remaining scatter/biome coupling (TM-P2r).
- **GPU grass macro is per-window constant** (±420 m window). If the macro
  gradient ever reads across a window edge, bake a per-texel macro channel
  into the control window instead.
- The `macro_variation` fine tier (250 m mottle + snow-line jitter) stays
  wrapped-shader-side; its regional (~1 km) tier moved into the baked albedo
  tone mottle.

## 4. Phase 2 — climate model → biomes (MVP IMPLEMENTED 2026-07-18)

The first climate slice landed as two shared scalar fields rather than
explicit biome classes — the blend-everything spirit of Whittaker weights with
a fraction of the machinery:

- **Cold lift** (`thalos_terrain::climate_cold_lift_m`, WGSL mirror
  `thalos::landcover::climate_cold_lift` — keep in lockstep): metres the
  ecological altitude bands (lush belt, treeline, snowline) descend at a
  latitude. A late power curve (`((sin_lat − 0.45)/0.55)^2.6 × 3600 m`) keeps
  mid-latitudes green against Thalos's high-sitting land (~850 m of lift at
  50°, treeline under the lowlands ~66°+, snow at sea level ~75°+ → polar ice
  caps). Every band consumer passes `altitude + cold_lift`: the terrain
  shader (latitude reconstructed per fragment from the cube-sphere coordinate
  via `compute_local_position` — exact, no extra bake channel), `albedo_at`
  (impostor/world map), the CPU blade builder, and GPU grass
  (`window_meta.w`, per-window anchor value).
- **Warmth** (`climate_warmth(cold_lift)`): gates the hot-desert **sand**
  palette (`C_SAND`) — the driest ground is sand in warm climates, tan
  steppe/soil in cold ones.
- **Moisture geography** (CPU only, bakes into the same albedo-alpha channel):
  a wet-planet base bias, equatorial wet belt, subtropical dry belt,
  mid-latitude storm track, polar desert (`latitude_moisture`), plus
  **continentality** — interiors drier than coasts, from the continentalness
  value the height path already computes.
- The runway site (lat 7.6°) sits at cold-lift 0 in the equatorial wet belt,
  so the approved spaceport look is preserved (slightly lusher).
- Tuning note: the world_map example's tint now **mirrors the in-game
  transfer curves** (dryness→tan past ~0.55, sand past ~0.8 × warmth, forest
  ramp) and its land ramp matches the eco bands — earlier tunings chased a
  map artifact (a brown 900 m hypso band the game never renders). Trust the
  map only because it mirrors the shader; keep it mirrored.

### TM-P3 — biome rebalance, authored to lore (LANDED 2026-07-20, `GENERATOR_VERSION` 12)

The `just map` stats (BL-11) diagnosed why the planet read as grey slabs with
a green fringe: 59 % of land classified *upland*, steppe/desert/tundra ≈ 0 %.
Three compounding causes, all fixed in one pass, authored against
`lore/solar_system.md` §II (35 % land; geologically old; "looks lush", with
rust-red lateritic ground where cover thins):

- **Macro bands re-aligned to the ground's eco bands** (the one-world rule):
  upland now 1500–2400 m eco (= the `landcover.wgsl` lush→treeline fade),
  rock 2400–3000 (treeline band), snow 3000–3600 (saturates at
  `CLIMATE_COLD_LIFT_MAX_M`, so the caps close at the poles). The old bands
  (upland from **120 m**, rock from 900 m) claimed nearly all land from
  orbit while the ground below rendered lush — orbit and ground disagreed.
- **INC-0005**: the crate-local `smoothstep`'s `.max(EPSILON)` denominator
  guard inverted every descending-edge call — the forest term had painted
  canopy onto the *driest* ground since the palette landed. No belt tuning
  could produce deserts while forest claimed dry land first. Forest edges
  also re-aligned to the ground's window (0.28–0.58 dryness).
- **Chain order**: tundra moved out of the lowland palette to sit between
  upland and rock, so cold cover can claim high-latitude ground the
  eco-shifted upland band used to crush; a laterite bare-soil step
  (mirroring the ground's `C_SOIL`, dryness 0.88–0.98) gives the lore's
  rust-red thin-cover ground a macro presence; soil + sand classify Desert.
- **Land authored lower** (old world): platform 420→300 m, interior gain
  650→400 m; `CONTINENT_C0` 0.105→0.143 → land fraction 35.2 % (lore: 35 %).
  Note: this reshapes all coastlines — coast-atlas / water verification
  passes see new coasts.
- **Moisture geography given teeth**: subtropic belt 0.40→0.70 amplitude and
  slightly widened; continentality gate re-sized to the real continentalness
  range (0.30–0.90, was 0.45–1.15 which never fully applied) and 0.24→0.26;
  wet bias 0.08→0.06.
- **Ecotone mosaic gate** (`GENERATOR_VERSION` 13, from the user's orbital
  screenshot: the dry belt read as splotchy green/tan camo): the 90 km /
  9 km mosaic tiers are scaled by `1 − 0.65·smoothstep(0.20, 0.50,
  |latitude + continental − bias|)` — where the geographic trend has
  committed (desert-belt core, rainforest core, polar desert) the cover is
  coherent and only the 700 km province tier varies it; the patchwork lives
  at the climate transitions, which is where real ecotones are. Reduced
  variance was compensated by the deeper belt mean (0.62→0.70), so desert
  cores are now *coherently* dry rather than noise spikes. Mosaic weights
  local 0.30→0.24, stand 0.35→0.26.

Result (map-verified, v13): forest 29.6 / grassland 25.6 / steppe 19.5 /
desert 4.6 / tundra 2.1 / upland 7.4 / rock 5.5 / snow 3.6 / beach 2.1 % of
land — tropical forest at the equator, a *coherent* steppe/desert belt at
15–40° (steppe-dominant both bands), temperate forest at the storm track,
barrens→tundra→cap polewards. Runway site stays LAND (602 m, equatorial wet
belt). In-game live-eye pending (TM-P3 row).

### Remaining for a fuller Phase 2

- **Biome identities (TM-P3b)**: erg/reg desert character, savanna, taiga
  vs temperate forest tone, softer polar rock ring, sea ice.
- **Scatter/biome coupling**: **trees & shrubs now gated by moisture + treeline**
  (landed 2026-07-20, TM-P2r.1 — `woody_biome_gate` in `scatter.rs`, §3
  follow-up above); the remainder is **grass profiles that switch per biome**
  (fold the macro fields into the grass style choice), the systematic version of
  the Phase-1 follow-up.
- **Explicit biome weights** (savanna vs steppe vs tundra palettes, biome-
  driven material masks) if the two-scalar model stops being enough.
- Sea ice at the polar ocean (water renderer, not terrain).
- Regional relief character (hills/swell amplitude by climate/orogeny).

## 5. Phase 3 — landmass & islands (DESIGN)

`continentalness` / `orogeny` stay the two replaceable seams (their doc
comments already promise this). Short of full plate simulation:

- **Plate margins from the Worley structure**: use the F2−F1 edge distance of
  the existing continent cells as a margin field. Mountain belts follow
  land-side margins (orogeny becomes margin-correlated instead of independent
  blobs); **island arcs** follow ocean-side margins.
- **Hotspot chains**: sparse seeded points with age-decayed seamount trails —
  mid-ocean archipelagos.
- **Shelf fragmentation**: extra coastline detail octaves gated to the shelf
  band so continental margins shed skerries/archipelagos without churning
  interiors.
- The hypsometric remap and relief cascade stay untouched, as designed.

Also queued here: regional height-character variation (hills/swell amplitude
modulated by the climate/orogeny fields, so badlands vs smooth plains exist).
