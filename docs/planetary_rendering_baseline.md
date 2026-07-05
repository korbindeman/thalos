# Planetary rendering baseline (2026-07-04)

A snapshot of the techniques the Thalos renderer applies today, written as the
**baseline for research into a next-generation full-scale planetary renderer**.
It describes what exists and how it works, states the known gaps, and points at
the code. It is descriptive, not aspirational — for the forward plan see
`docs/graphics_fidelity.md`; for per-system specs see `docs/terrain.md`,
`docs/atmosphere.md`, `docs/vegetation.md`.

Scale context: Thalos (the homeworld) is a ~3.19 Mm-radius terrestrial body
with ocean, atmosphere, and vegetation; the system also contains airless
regolith bodies (Mira), thin-atmosphere bodies (Vaelen), and gas giants. The
renderer must hold up continuously from interplanetary distance to a boot
standing on the ground.

---

## 1. Architectural frame

Three ideas organise the renderer:

1. **One physical world (the "one-world principle").** Every surface —
   terrain, vegetation, rocks, water, crafts, structures — should obey the
   same light, cast into and receive the same shadows, occlude each other, and
   recede into the same air. A surface that opts out reads as a pasted-on
   cut-out. (`docs/graphics_fidelity.md` §2.3.)
2. **One shared shading spine.** All body-surface materials import a single
   set of WGSL libraries (`thalos::lighting`, `thalos::atmosphere`,
   `thalos::shadow`, `thalos::landcover`, `thalos::foliage`,
   `thalos::grass_displace`) registered by `PlanetLightingPlugin`. No
   material-local BRDF or palette forks; when a constant moves it moves once.
3. **The central debt: two lighting universes.** Terrain / vegetation / water
   / rocks / impostors shade through the spine; **crafts and structures still
   use Bevy stock PBR** (`StandardMaterial` / `ship_part.wgsl`), reconciled by
   a CPU projection layer. The F1–F9 unification foundation is collapsing this
   (F1/F2 runtime-verified; F3–F6 landed; F7–F9 pending — see §12).

One further structural fact matters for research: **both universes already
share the camera and post stack** (one HDR target, one tonemap). The
divergence is purely in shading inputs — BRDF, sun/sky/shadow/IBL evaluation.

Crate layout: rendering *mechanism* (materials, shaders, spine, LUTs) lives in
`thalos_body_render`; per-frame *drivers* (fill uniforms, LOD swaps, f64
anchoring) live in `thalos_game::rendering`. The terrain LOD engine is the
vendored `thalos_udlod` (fork of kurtkuehnert/bevy_terrain), consumed only by
`body_render`.

---

## 2. Planet geometry & LOD (UDLOD)

The ground is streamed by the vendored **UDLOD** renderer: a cube-sphere,
sparse-tile-atlas, runtime-provider-first terrain system.

**Tile address space.** 6 cube faces × quadtree LODs; LOD 0 is one tile per
face, LOD N is 4^N tiles per face (`crates/udlod/src/math/coordinate.rs`).
Cube-face UVs carry a C_SQR distortion correction. Tile borders overlap by
2 px carrying neighbour data, and the pixel→direction mapping is *stitched*
(bit-exact across neighbours) so bilinear filtering never seams.

**Per-body tile configuration** (`crates/game/src/rendering/ground_terrain.rs:73`):

| Tier | LOD count | Atlas slots | Tile res | Use |
|---|---|---|---|---|
| Near (ship view) | 16 | 384 | 512² | active body ground |
| Distant | 8 | 64 | 128² | other bodies with terrain |
| Map | 12 | 256 | 256² | map view |

At Thalos's radius LOD 16 puts the finest tile texels at roughly metre scale.

**Tile selection is CPU-side and 2:1 balanced.** The upstream GPU refine pass
was replaced by a per-frame CPU `compute_draw_set()` that enforces a
restricted quadtree (any tile's neighbours within one LOD level), including
across cube-face seams — this is the correctness path that killed elevation
shears at seams. Tile *production* is the intended GPU extension point (job
queue → atlas slots); today it is CPU.

**Residency.** `TileAtlas` allocates fixed slots, evicts LRU, pins LOD 0 as a
permanent fallback ancestor, and guards stale async loads with per-slot
generation counters. Loads admit **coarse-first, then nearest-view-first
within each LOD**, so a cold teleport fills a complete low-LOD pyramid before
refining — this is what makes ~15 s cold streaming to a fresh site tolerable.
4 concurrent load slots, 32-deep queue; each tile bake is rayon-parallelised
across rows on a bounded pool.

**Precision.** Rendering sits under vendored `big_space` (i64 cells, ~1 km
cell size, camera owns `FloatingOrigin`). Within a body, UDLOD's key novelty
is a **second-order Taylor-series relative-position path**: per-frame CPU f64
coefficients let the vertex shader reconstruct camera-relative positions with
sub-centimetre precision at planet radius, where naive f32 quantises to
~0.25 m (`crates/udlod/src/shaders/functions.wgsl`, `compute_relative_position`).
This path is unconditional (the upstream `high_precision` feature was removed).
Everything that must be rock-steady on a rotating planet (runway, grass tiles,
structures, shadow-cascade anchors) follows the same pattern: a root-grid
child **re-posed in f64 every frame**, with only small f32 vertex offsets
riding the f32 rotation.

**Vertex morphing.** UDLOD morphs vertices between LODs in the vertex shader
(screen-space-error driven), so geometric LOD transitions are continuous; the
remaining pops are in the *content* layers (vegetation, detail bands), not the
mesh.

---

## 3. Terrain generation (runtime, no bake)

Terrain is a **pure analytic function evaluated at runtime** — there is no
bake, no disk cache, no height asset. One `thalos_terrain::ProceduralSurface`
per body implements the `SurfaceQuery` seam:

```rust
fn sample_d(&self, dir: DVec3, lod_m: f32) -> SurfaceSample  // height, albedo, roughness
```

Every consumer reads the same surface: the UDLOD tile baker, the physics
collider (parry heightfield patches), camera terrain floor, HUD altitude, EVA,
runway placement. The GPU-atlas height mirror (`HeightSourceRegistry`) serves
CPU-side height queries from the same resident tiles the renderer draws, with
a direct-evaluation fallback.

**Height model.** A signed continent/seabed macro field (the shoreline is
height 0 at the reference radius — "sea level" is the constant 0 m, no ocean
layer) plus a ridged-multifractal detail cascade: base wavelength 1000 m,
amplitude 250 m, lacunarity 2, up to 11 octaves (bottoming out ~0.5 m
wavelength / ~12 cm amplitude), with 2-octave domain warp (4 km wavelength,
800 m displacement) to break the lattice (`crates/terrain/src/query.rs:57`).
Relief fades to zero in a ~60 m band at the waterline so the coastline is
LOD-invariant (the shore is a macro-field crossing, not a detail-noise
accident). Ground height is deliberately LOD-invariant in this slice to avoid
parent/child tile contour steps.

**Local modification.** `TerrainFlatten` regions (runway/base pads) level the
field to a tangent plane with a smoothstep ramp (~500 m), read per tile-pixel
at bake time through a per-body registry handle; a despawn/respawn rebuild
path (`TerrainRebuildRequest`) covers flattens installed after tiles streamed.

**Tile attachments.** Each baked tile carries multiple channels: `height`
(RG16 coarse+residual packing — plain R16 contours on shallow slopes),
`normal`, `albedo` (sRGB), `roughness`, and a `material` weight mask
(R=grass, G=soil, B=rock, A=wetness) derived from slope, altitude, and a
Laplacian hollow/convexity stencil at bake time. A `horizon` channel for baked
relief-shadow angles is reserved but unbuilt.

**Determinism.** Tile synthesis is position-pure (no neighbour-aware passes,
no per-tile hydraulics), so any tile at any LOD agrees bit-exactly with its
neighbours and parents on shared edges. This is a hard constraint the current
generator honours and any future generator must preserve or replace
deliberately (it currently rules out simulation-style erosion at bake time).

---

## 4. Surface shading

**The spine.** `thalos::lighting::shade_surface(ThalosSurface, …)` is the
canonical per-fragment entry (`crates/body_render/src/shading/shaders/lighting.wgsl`).
The caller fills `ThalosSurface { albedo, roughness, normal_ws, geo_normal_ws,
emissive, occlusion, metallic, translucency, style }` and dispatch routes on
style:

- **REGOLITH** → `shade_hapke_surface`: a Hapke (2002) radiative-transfer BRDF
  (opposition surge B₀=1, single-scatter albedo w=0.45, back-scattering HG
  phase g=−0.3, Chandrasekhar H-functions). Used by airless terrain *and* the
  distant impostor, so a body's orbital disc and ground LOD reconverge across
  the LOD swap.
- **DIELECTRIC / FOLIAGE / WATER** → Cook–Torrance GGX (Smith
  height-correlated visibility, Schlick Fresnel, F0=0.04) + Oren–Nayar rough
  diffuse + Kulla–Conty energy compensation + Karis split-sum env-BRDF for
  ambient specular. The `metallic` field is reserved but unread (F7);
  FOLIAGE/WATER branches are stubs — foliage and water still run parallel
  shading paths (F9).
- **Foliage** shades via `shade_foliage`: wrap diffuse (bias 0.40, ~23° past
  the terminator), a warm two-sided translucency lobe, and hemisphere ambient
  with canopy bleed controls.
- **Specular AA** (Kaplanyan-style screen-space normal-variance roughness
  widening) is always on; it is the main defence for shimmer since the game is
  SMAA-only (no TAA).

**Sun/sky inputs.** `compute_surface_sky` converts zenith optical depth + sun
elevation into `{ sun_color, sun_scale, sky_radiance, ground_radiance }` — an
analytic clear-sky model (white noon → orange sunset, blue dome, warm ground
bounce, cool night floor). One shared terminator definition
(`sun_daylight`/`surface_daylight`) is used by every surface *and* by the CPU
light projection (F1 collapsed three duplicate definitions into one).

**Terrain material** (`body_terrain.wgsl`) samples the tile atlases, then
layers analytic detail: slope-driven substrate, multi-octave albedo breakup
(~20 m patches), a synthesized detail normal field (~1.25 m relief), and
grass-band micro-detail gated on the material mask. Ecological palette bands
(forest → alpine scree → snow by altitude and slope) come from
`thalos::landcover`, which has a hand-maintained CPU mirror
(`ground/landcover.rs`) — a known drift hazard slated for retirement (TM3).

**The frequency-band material model (and its gap).** The macro band —
per-tile material weights + analytic ecological bands — ships. The detail
band is **noise-modulated flat colour**, not a tiling PBR material library:
there are no detail albedo/normal/roughness textures, no height-biased weight
blending, no stochastic/hex de-repetition, no triplanar cliffs. This is the
single biggest close-up realism gap on the ground (TM1/TM2).

---

## 5. Lighting environment

**SceneLighting** is one CPU-built uniform feeding every spine material: up to
4 stars (direction + flux), up to 8 eclipse occluders (soft-penumbra sphere
tests), planetshine (Lambert-sphere phase from the parent body), and moonlight
(direction + phase/size/albedo/distance-derived flux, night-gated). Sun flux
is **heliocentric**: `LIGHT_AT_1AU · (AU/d)² · exposure_gain` — every surface,
including (since F1) the Bevy-PBR hull, dims with distance from the star.

**Exposure** has a single authority (F2): `CameraExposure` — an artist
distance-gain curve (gain = (focus_distance/AU)^1.0) plus a fixed global
baseline (`GLOBAL_EXPOSURE_STOPS = 0.0`). The Bevy `AutoExposure` histogram
was removed. This is deliberately *not* physical EV100 metering; brightness is
authored, not metered.

**Ambient / IBL (F3/F4, landed).** One physical source: a CPU **sky-view LUT**
(`shading/sky_view.rs`, 64×96 azimuth-from-sun × zenith, 16-step raymarch of
the *same* single+multi-scatter model the terrain shades through) is baked per
(sun direction, altitude). It feeds:
- the **reflection probe** (`reflection_probe.rs`): a CPU-painted 256²×6
  RGBA16F cubemap (physical sky + analytic warm ground bounce + HDR sun disc +
  starfield), repainted every 2 s (0.5 s under warp), consumed by Bevy's
  `GeneratedEnvironmentMapLight` prefilter for hull/structure IBL;
- the **surface ambient**: the LUT's cosine-weighted SH DC irradiance drives
  `GlobalAmbientLight` at gain 0.2 (a residual — the env-map prefiltered
  diffuse already delivers sky irradiance to `StandardMaterial`, so full
  strength double-counts), with ~0.7 s temporal smoothing.
Space-regime ambient is still a flat hand-tuned stand-in; full SH-9 and a
spine (terrain) LUT-ambient port are pending. The probe is CPU-painted, not a
real scene render — terrain/objects do not appear in reflections.

**AO (F5, landed for terrain).** A custom half-res hemisphere SSAO node
(R16F, IGN-rotated, no blur yet) reads the copied scene depth — the only depth
that sees the forked-UDLOD terrain (Bevy's prepass GTAO is terrain-blind
here) — and feeds `ThalosSurface.occlusion` (ambient-only, 1-frame latency).
Vegetation/rock/hull receivers and a VBAO upgrade are pending.

**Moonlight** is a physical-ish secondary directional term
(`moonlight_radiance` in the spine + a Bevy `MoonLight` projection for the PBR
universe), phase-driven, night-gated by the shared terminator.

**Post stack** (shared by everything): HDR → AgX tonemap, bloom (0.35,
threshold 0.6), SMAA High (no MSAA/TAA on the main path), contrast-adaptive
sharpening, deband dither, mild chromatic aberration (0.3%), exposure-driven
film grain. AgX-vs-Khronos-PBR-Neutral is an open A/B (ACES rejected).

---

## 6. Shadows — one shadow world

A self-managed cascade rig (`rendering/sun_shadow.rs` + `thalos::shadow`)
replaced stock Bevy CSM entirely (disabled on the sun light since F6):

- **3 orthographic cascades**, 4096² each, half-extents 400 / 1500 / 4000 m
  (expandable by power-of-two footprint scaling with hysteresis), centred on
  the **craft** (not the camera).
- **Everything casts and receives the same rig**: terrain receives (casting is
  replaced by the horizon term below); trees, grass, rocks, craft hull, gear,
  buildings, tanks, and the runway all cast (`SHADOW_CASTER_LAYER = 8`) and
  receive — the hull via `ship_part.wgsl`, plain `StandardMaterial` surfaces
  via a `ShadowedStandardMaterial` extension. Flat paving is receive-only
  (coplanar caster-receivers acne at grazing sun).
- **Stability on a rotating planet** required re-deriving the flat-world
  stable-CSM tricks in the body-fixed frame: texel snapping is anchored
  body-fixed (a render-space grid slides as the planet co-rotates → flicker),
  the cascade centre uses true AGL from the height source, and footprints are
  quantized with hysteresis.
- **Bias**: receiver normal-offset (~1.2 texels, capped 1.5 m) + slope-scaled
  depth bias (capped 2.5 m — must stay below tree height), 3×3 PCF.
- **Orbit mode**: above ~50 km AGL the rig collapses to a craft-local single
  cascade so the hull keeps self-shadowing without ground coverage.
- **Planet-scale terrain shadowing** is split: terrain-on-terrain uses a
  height-atlas horizon march in `body_terrain.wgsl`; terrain-on-objects is a
  v1 CPU horizon march (f64, ~30 km reach along the sun azimuth) that dims the
  sun `DirectionalLight` for the craft — per-fragment horizon for spine
  materials, cloud shadows, and PCSS/contact shadows are pending.

---

## 7. Atmosphere & sky

One **per-body fullscreen pass** (`BodySkyMaterial` / `body_sky.wgsl`,
"BodySky") handles halo, sky dome, aerial perspective, clouds compositing, and
the ocean in a single depth-aware raymarch. It runs at every altitude — the
same pass renders the limb from orbit and the sky dome from the ground.

**Scattering model** (`thalos::atmosphere`): single-scattering Rayleigh +
Henyey-Greenstein Mie, 16 view steps (adaptive down to 4 by path/shell ratio)
× 8 sun-column steps, per-body authored parameters in `assets/bodies/*.ron`
(Thalos: Rayleigh vertical OD (0.046, 0.108, 0.264), H_R 8 km; Mie OD 0.025,
H_M 1.2 km, g 0.76; Kármán line 80 km). A **32×32 multi-scatter LUT**
(Hillaire-style infinite-series transfer, CPU-baked per body) adds the
blue-dominant horizon fill. Limb darkening is applied on the disc. Not
physical: an artistic `strength` multiplier (3.0) on in-scatter, a luminance
threshold that crushes stars on bright sky pixels, and a decoupled
**aerial-perspective strength** (0.15 of sky strength) so the daytime dome can
be bright without over-fogging the ground.

**Depth coupling.** A render-graph node copies the main-pass depth into a
sampleable image between the opaque and transparent passes
(`rendering/scene_depth.rs`, MSAA resolve variant included). The atmosphere
pass reads it to clip the raymarch at terrain/hull/impostor hits and to key
aerial perspective to **view optical depth** (τ_near 0.30 ≈ 8 km onset, τ_far
2.40 ≈ 70 km full veil at sea level) rather than Euclidean distance — which is
what keeps orbital nadir views un-fogged while limb views haze fully. Because
BodySky is depth-keyed and fullscreen, crafts and structures receive aerial
perspective "for free" — the one one-world invariant that already holds for
the PBR universe.

**Object recession.** Spine-shaded objects (foliage, rocks) additionally blend
toward the local sky haze between 1 km and 35 km (capped at 0.32, brighten
capped 1.5×) via `object_aerial_recession`, so cut-outs don't stay crisp
against hazed terrain.

**What the atmosphere is not**: no ozone term (no blue twilight wedge), no
froxel/Hillaire aerial-perspective LUT (the raymarch runs per pixel), sun-only
(moon/star scattering handled analytically), no refraction, no
polarisation.

---

## 8. Clouds

Vendored, heavily reworked **HZD-style volumetric raymarch**
(`crates/volumetric_clouds/`), run in the **body-fixed frame** of the nearest
terrestrial-atmosphere body: true ray-sphere shell intersection from the
camera's planet-centred position, wrap-first noise sampling that stays
f32-safe at planet-radius coordinates — so clouds are planet-fixed,
co-rotating, and horizon-correct at any altitude.

- **Density**: planet-fixed equirect coverage map (512×256, generated from
  per-body `CloudWeatherState` — ITCZ/subtropical/storm-track latitude bands +
  seeded noise, version-gated re-upload; the future weather system's write
  target) × vertical profile (rounded base, eroded top) × 3-octave value-noise
  FBM detail. Thalos deck: base 2 km, thickness 1.3 km.
- **March**: ~adaptive 8–32 view steps, 4-step sun self-shadow march,
  non-physical forward-biased phase (silver lining), temporal reprojection
  with per-pixel nearest-hit distance and disocclusion rejection.
- **Compositing**: the cloud in-scatter texture + nearest-hit distance
  composite *inside* the BodySky pass (bound as `cloud_layer` /
  `cloud_distance`), not as a separate quad — separate transparents sort
  unreliably against the fullscreen sky under big_space.
- **Gaps**: ~25 km reach cap, full-res (no half-res upscale), no Nubis-grade
  lighting (powder/multi-scatter octaves), no cloud shadows on the ground yet,
  reprojection ghosts under fast motion (needs body-fixed motion vectors),
  static weather, and the orbital impostor's cloud shell is a separate
  fixed reference texture (unaligned with the volumetric layer).

---

## 9. Water

Planet water is **never a mesh**. Two representations:

- **Ship-view ocean**: an **analytic ray-traced sphere inside the BodySky
  pass** (`thalos::water`): numerically-stable ray-sphere intersection
  (Vieta + CPU f64-fed exact camera height), tested against scene depth so
  terrain correctly occludes/emerges. Shading: Cook–Torrance GGX (α 0.10,
  F0 0.02), two octaves of scrolled finite-difference wave normals (67 m and
  14 m, fading out by 6 km so orbit sees a clean sphere), exponential depth
  absorption from the water column recovered from scene depth
  (shallow-cyan → deep tint over ~14 m), and shoreline feathering over ~3 m of
  depth. There is also a dormant mesh-based `BodyWaterMaterial`
  (`TERRAIN_PATH_WATER_ENABLED = false`).
- **Map/distant ocean**: baked into the procedural impostor's albedo cubemap
  (flat colour v1).

Gaps: no displacement/geometric waves, no foam/whitecaps, no SSR (reflections
come from the analytic sky/sun only), three hand-calibrated water BRDFs await
consolidation into the spine's WATER branch (F9/W19).

---

## 10. Vegetation & ground detail

Design (docs/vegetation.md): a **four-band representation cascade** per layer,
ending in the terrain albedo — geometry is a near-field detail layer over a
ground colour that already reads correctly from any distance. Placement is
**deterministic and view-independent** (body-global cube-sphere tile lattice +
hashed candidates + priority Poisson-disk elimination), so roots are identical
at every LOD and nothing is stored. Constant *coverage*, not count, is held
with distance: unresolvable elements (grass) enlarge as density drops;
resolvable ones (trees) keep natural size and thin by spacing.

**Grass** — the newest subsystem (GPU slice 1, landed 2026-07-04,
game-unverified): a Ghost-of-Tsushima-style **vertex-synthesized field**. One
static template mesh (~780k verts) is reinterpreted every frame by the vertex
shader from cell hashes: 5 concentric bands (0.3 m → 5.2 m cells, 26 → 4
blades/clump) reaching 340 m, gated by a scrolling 768² control window
(height + landcover mask) re-filled asynchronously as the anchor drifts.
Blades get Bézier gust bending + flutter from a shared wind field, rounded
cross-blade normals converging to the terrain normal with distance, sheen +
translucency, per-clump moisture-driven dry/lush blending from the landcover
field, view-dependent edge-on widening, and a screen-space minimum blade
width. Beyond the field: baked 4-variant **card-atlas** tufts, then band 2
(grass detail folded into terrain shading), then landcover albedo. O(1)
persistent memory — this replaced CPU mega-tile blade meshes whose rebuild
churn was an OOM class.

**Trees/shrubs** — procedural species (branch-skeleton broadleaf + egg-shell
canopy; conifer authored), meshed at 4 LODs, batched one mesh per tile.
Foliage colour has **one definition** (`thalos::foliage`, sampling a
procedurally generated texgen atlas: leaf clusters, needles, bark
albedo/normal/roughness all derived from one gradient-noise bark height
field), used identically by the near mesh and the **octahedral impostor bake**
(8×8 hemioctahedral views, 128 px cells, albedo + normal/depth channels,
baked at startup) — so near↔far colour cannot drift. A 4-ring clipmap carries
trees to ~22 km: ring 0 (mesh LODs → natural-size impostor cards), ring 1 a
strict *subset* of ring 0's grid (thinning without dissolve), rings 2–3
coarse independent grids, with mirrored cross-fades at boundaries.

**Rocks/pebbles** — deformed-icosphere scatter placed inversely to grass
weight (bare ground), ~100 m reach, one ring, shared ground BRDF.

**Scatter clearing** — a `ScatterRegion` layer derived from the structure
registry clears vegetation under paved/built footprints and forces lawn grass
on base sites.

All layers: f64 body-fixed per-tile anchoring, one placement gate sampling the
same stencil the terrain mask baker writes, residency-gated async builds,
altitude ceilings, revision-based rebuilds. Instancing today is **batched
meshes per tile** (Bevy auto-batching); GPU-driven culling/indirect draw is a
planned phase, and the three drivers (grass/vegetation/rocks) triplicate the
clipmap lifecycle (a known refactor, VEG-R).

---

## 11. LOD chain, distant bodies, celestial sky

**Orbit→ground chain today**: icon dot → flat billboard impostor → full UDLOD
terrain. The impostor↔terrain swap keys on **apparent screen size** (body's
rendered radius vs. one icon-dot radius, `ground_terrain.rs:274`), not a fixed
multiple of radius; distant-tier tiles stream at ~2× the swap distance so the
handoff finds resident tiles. The swap is a hard cut.

- **Procedural-body impostor**: an albedo cubemap baked at startup from the
  same `ProceduralSurface` (continents + flat-colour ocean), shaded through
  the spine's Hapke path so disc and ground reconverge; gas giants and rings
  have dedicated impostor materials (banded cloud dynamics, ring shader).
- **Known break**: the interim solid-colour impostor for some bodies breaks
  the Hapke↔Vegetated reconvergence, and there is no dithered cross-fade
  (gated on TAA). The declared end state (W17) is UDLOD at all ranges with the
  impostor branch deleted.

**Celestial sky**: `thalos_celestial` generates a physical-flux universe
(~50k stars, galaxies; blackbody/power-law SEDs — never pre-baked RGB),
rendered as additive HDR billboards whose brightness is divided by the
exposure gain (constant perceived brightness across regimes), suppressed by
sun elevation (twilight ramp) plus the per-pixel sky-luminance star crush.
Eclipses, planetshine, and moonlight all run through `SceneLighting` for every
spine surface.

---

## 12. Status of the unification foundation (F1–F9)

| Step | What | Status |
|---|---|---|
| F1 | One terminator; Bevy sun/moon/ambient as projections of `SceneLighting`; heliocentric hull flux | ✅ runtime-verified |
| F2 | Single exposure authority (AutoExposure removed) | ✅ runtime-verified |
| F3 | Physical CPU sky-view LUT (sun-only) feeding the reflection probe | ☑ landed |
| F4 | LUT → ambient irradiance for surface `GlobalAmbientLight` + atmosphere-painted env cubemap | ☑ landed (gain 0.2 residual; SH-9 + spine port pending) |
| F5 | Half-res SSAO → terrain occlusion | ☑ landed (blur + more receivers pending) |
| F6 | One shadow world (everything casts/receives the rig; Bevy CSM disabled) | ☑ landed |
| F7 | Metallic branch + shared view-level scene/atmosphere bind group + prefiltered env from the LUT | ☐ |
| F8 | Port structures (a) then hull (b) onto `shade_surface` | ☐ |
| F9 | Wire FOLIAGE/WATER branches; retire parallel foliage/water BRDFs | ☐ |

---

## 13. Where realism falls short (research targets)

The gap inventory, roughly ordered by how much they hold back "proper
full-scale planetary renderer" realism:

**Terrain content & materials**
1. **No tiling detail-material layer** — near-field ground is noise-modulated
   flat colour (no detail albedo/normal/roughness textures, no height-biased
   blending, no hex-tiling de-repetition, no triplanar cliffs).
2. **Generator realism** — the height model is fractal noise + continent mask:
   no erosion/hydrology/drainage networks, no tectonic structure in the active
   path, no aeolian/glacial landforms; landcover is altitude/slope/noise, not
   climate-derived. The position-pure determinism constraint currently
   excludes simulation-based approaches at bake time.
3. **Four overlapping palettes + a hand-maintained CPU↔WGSL landcover mirror**
   (brittleness, not just realism).

**Lighting & GI**
4. **Two lighting universes** until F7–F9 land (hull/structures on Bevy PBR:
   different BRDF, IBL, ambient; no metallic branch in the spine).
5. **No GI beyond flat sky ambient + SSAO** — no bent normals, no SSGI, no
   sky-visibility term (basins as bright as ridgetops), SSAO unblurred and
   terrain-only.
6. **Reflections don't reflect the real scene** — CPU-painted probe (sky +
   analytic ground only), no SSR, no terrain in any reflection.

**Atmosphere & volumetrics**
7. **Single scattering + tricks** — no ozone, no froxel aerial-perspective
   LUT, artistic strength multipliers standing in for physical radiance
   calibration; exposure is authored distance-gain, not luminance metering.
8. **Cloud quality** — value-noise detail (not Perlin-Worley at full quality),
   non-physical phase, no in-cloud multi-scatter octaves, 25 km reach, no
   cloud shadows, full-res cost, reprojection ghosting.

**Temporal & LOD**
9. **No TAA** — the single gate blocking dithered LOD cross-fades, cloud
   resolve quality, and vegetation shimmer reduction; needs body-fixed motion
   vectors under big_space/dual-camera (Open Q7).
10. **Hard impostor↔terrain cut** and an interim flat/solid-colour distant
    body view (Slice 6 / W17): no mid-range "whole-planet from 1000 km"
    representation with real relief — the chain jumps from baked cubemap disc
    to streamed tiles.
11. **Vegetation mid-band** — no HLOD forest-cluster impostors; canopy colour
    not yet baked into terrain albedo as a coupled fade (the orbit→ground
    descent pop, W1).

**Water**
12. Flat analytic ocean: no wave displacement, foam, or SSR; shore
    interaction is a depth fade only.

**Shadows**
13. Terrain-relief shadowing per-fragment for objects/vegetation, PCSS
    contact hardening, contact shadows, cloud shadows — all pending; the v1
    horizon term is per-craft CPU only.

**Platform posture**
14. Tile production, vegetation placement, and LUT bakes are CPU-side; no
    GPU-driven pipeline (indirect draw, GPU cull, compute tile synthesis), no
    bindless materials yet, no RT path (terrain has no BLAS).

---

## 14. Key code anchors

| System | Where |
|---|---|
| LOD engine | `crates/udlod/` (tile tree/atlas: `src/terrain_data/`, precision: `src/shaders/functions.wgsl`) |
| Terrain generator | `crates/terrain/src/` (`query.rs` seam, `procedural.rs`) |
| Tile bake → GPU | `crates/body_render/src/ground/pipeline.rs`, `body_terrain.wgsl`, `body_material.rs` |
| Lighting spine | `crates/body_render/src/shading/shaders/lighting.wgsl`, `shading/mod.rs` |
| Atmosphere | `shading/shaders/atmosphere.wgsl`, `shading/{multi_scatter,sky_view}.rs`, `ground/body_sky.wgsl` |
| Scene depth | `crates/game/src/rendering/scene_depth.rs` |
| Shadows | `crates/game/src/rendering/sun_shadow.rs`, `shading/shaders/shadow.wgsl` |
| SSAO | `crates/game/src/rendering/ssao.rs` + `ssao.wgsl` |
| Exposure/light projection | `crates/game/src/rendering/lighting.rs`, `impostor/post_stack.rs` |
| IBL probe | `crates/game/src/reflection_probe.rs` |
| Clouds | `crates/volumetric_clouds/`, `crates/game/src/rendering/clouds.rs` |
| Water | `shading/shaders/water.wgsl` (inside body_sky), `ground/water_material.rs` (dormant) |
| Vegetation | `crates/body_render/src/ground/{vegetation,scatter,gpu_grass,tree_impostor,landcover}.rs`, drivers in `crates/game/src/rendering/{grass,gpu_grass,vegetation}.rs` |
| Celestial | `crates/celestial/`, `crates/game/src/sky_render.rs` |
| Plan / specs | `docs/graphics_fidelity.md`, `docs/terrain.md`, `docs/atmosphere.md`, `docs/vegetation.md` |
