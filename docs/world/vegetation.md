# Vegetation

Planet-scale procedural vegetation for Thalos: grass and ground cover that
stretches to the horizon, plus shrubs and trees, all with smooth LOD and good
close-up fidelity. This document is the **unified spec and implementation
plan** for the whole vegetation stack. The near-camera grass blade layer that
shipped 2026-06 (described in `docs/world/terrain.md` *Vegetation decoration layer*)
is the **foundation** this plan generalizes — its tile lattice, deterministic
hashed placement, shared placement gate, f64 per-tile anchoring, and
async-build/revision-rebuild lifecycle are reused verbatim by every layer here.

> **Status: Phases 0–2 landed (2026-06-25), runtime-unverified.** Shipped:
> the shared `tile_lattice` + `placement_gate` + `scatter` foundation (Phase 0);
> grass blade-LOD geometry + the **clipmap rings that take grass to the horizon**
> (Phase 1a–c); and **procedural trees + shrubs** with a mesh-LOD chain,
> clumping, per-instance tint/scale variation, terrain-residency-gated builds and
> f64 anchoring (Phase 2 + Phase 3 shrub species), plus tree/shrub **wind** and
> sky-consistent lighting via a custom `TreeMaterial` (vertex sway + the shared
> `thalos::lighting` hemisphere model). Compile- and unit-test-green, **not yet
> visually verified in a `just game` run** (the user owns that). Built alongside a
> parallel grass **sky-lighting** track in the same tree.
>
> **Scaling (2026-06-25):** trees/shrubs now batch into **one mesh per tile**
> (the grass path), removing the per-tree entity ceiling; range extends to
> ~1.4 km with seamless scale-fade. The Option-A→B→indirect escalation was
> superseded by this lower-risk per-tile batch (see §6).
>
> **Octahedral impostors (Phase 2d) landed (2026-06-25), runtime-unverified.**
> The far tree band is now hemisphere octahedral impostors (camera-facing quads
> sampling a per-species angle atlas baked off-screen at startup) instead of the
> minimal LOD3 mesh blob: the LOD2 mesh hands off to impostors at ~1.2 km and
> they carry the forest out to ~3.6 km at one quad apiece, re-lit through the same
> `thalos::lighting` model, with the existing craft-anchor scale-fade for a
> seamless, zoom-independent handoff + far edge. See §7. (Built compile-/clippy-
> /test-green; awaits a `just game` screenshot pass — the v-flip / normal-sign
> conventions are the one likely tuning iteration.)
>
> **Remaining:** the geometry→terrain-albedo handoff polish (Phase 1d — grass
> detail normal in `body_terrain.wgsl`, root blend); impostor polish (true
> depth-channel parallax — the channel is baked but unused; per-tile frustum-cull
> Aabb if far-tile draw count bites; a coarse impostor clipmap ring to reach the
> horizon from altitude cheaply); GPU-driven cull + `draw_indirect` only if
> per-tile meshes ever become the ceiling; GPU-generated grass (Phase 5).
>
> **Impostor clipmap (far-band reach extension) landed (2026-06-27,
> runtime-unverified).** The tree far band was a single fixed 200 m lattice with
> one quad per tree, capped at ~4.8 km — far too short to fade believably to the
> terrain albedo (the horizon is tens of km from any altitude). It is now a
> **clipmap**: ring 0 keeps the near/mid mesh cascade + natural-size impostor
> band; coarse **impostor-only grove rings** (tile 500/1000/2000 m) carry the
> forest to **~22 km**. Each coarse ring scatters `1/spacing_scale²` as many
> trees (a coarser Poisson grid via `VegScatterInput::spacing_scale`) to bound
> the quad count — so reach grows ~5× at a *bounded* (and net-lower) quad count
> vs. the old all-200 m-tiles band. Rings cross-fade through their shared boundary
> via per-ring impostor materials (own `time_fade`), mirroring the grass clipmap.
> AGL build ceiling raised (2.5→6 km) so climbing aircraft keep the forest. Driver:
> `game/src/rendering/vegetation.rs` (`TREE_RINGS`).
>
> **Trees stay natural-size at every LOD (`grove_scale = 1`, 2026-07-01).** The
> coarse rings originally *enlarged* each impostor `grove_scale×` (2.5/5/10) to
> hold coverage while decimating count — the constant-coverage rule. That reads
> wrong for trees (individually resolvable → giant trees + a size snap at every
> ring boundary as you approach). The enlargement is removed: every ring draws
> trees at their true bounding-sphere size, and the coarse rings carry coverage
> through `spacing_scale`/`keep_fraction` decimation plus the terrain albedo, not
> by fattening survivors (see §5.1).
>
> **Cross-ring positional correspondence — shared grid near (2026-07-01).** With
> the sizes consistent, the next artifact surfaced: trees "appearing from nothing"
> at the medium ring and a near tree vanishing before a different close one
> appears. Cause: **each ring used an independent Poisson grid** (`spacing_scale`
> coarsens the grid → *different* tree positions), so a ring handoff wasn't a
> tree changing representation — it was one whole grid of trees dissolving into a
> different grid. Invisible for grass (sub-pixel blades) but glaring for discrete
> trees; it also violated the §14 "root identical at every LOD" invariant. Fix:
> **ring 1 now shares ring 0's grid** (`spacing_scale = 1`) and renders a
> **nested subset** of it (`VegScatterInput::keep_fraction`, a deterministic
> per-cell hash gate applied before the height gate), so its trees are exactly
> ring 0's at the same spots — the 2.4 km handoff keeps each shared tree in place
> and only the density-delta *infill* fades in. Rings 2–3 (≥ 6 km, sub-pixel /
> below the eye-line horizon) keep cheap coarse independent grids. The ring fade
> also switched from *overlap-full* to a **complementary cross-fade** (fixed
> `TREE_FADE_BAND_M`, scales sum ~1 at a boundary) so a shared tree isn't
> double-drawn. `keep_fraction` on ring 1 is the handoff dial (→1 = no infill
> pop, denser mid-field). *Residual:* the scale-fade still breathes a shared tree
> to ~½ size mid-band (a few px at 2.4 km); a true fix is an opacity cross-fade,
> deferred. **Still needs the correlated forest-albedo handoff** (§9 / Phase 1d)
> so the ~22 km geometry edge melts into forest-colored ground — the immediate
> follow-up — and a **near-mesh quality pass** ("close trees read poorly": flat
> leaf-cluster cards / thin canopy — a mesh/atlas art task, §10).
>
> **Grass redesign decision (2026-07-03).** The shipped grass clipmap tops out
> at **340 m** (three rings; the "~1–1.5 km" reach above was never extended),
> Phase 1d (the terrain handoff) was never implemented, and the CPU
> megamesh-per-tile model is the confirmed memory failure: the game **runs out
> of memory with grass enabled** (grass tiles + rebuild churn on top of the
> ~450-mesh-tile tree fleet). Aerially the grass reads as a small noisy card
> disc with a visible coverage ring at its edge. After a survey of shipped
> horizon-grass systems (Ghost of Tsushima GDC 2021 per-frame GPU generation;
> MSFS greenness-derived grass; HZD ecotope placement; hexaquo/AMD/Far Cry 5),
> the plan is restructured around the **four-band cascade** (§5.0) with two
> commitments, in order:
>
> 1. **Band 2 first — grass as terrain shading** (§5.0 band 2; elevates and
>    broadens Phase 1d): a landcover-driven grass detail layer *inside*
>    `body_terrain.wgsl` (albedo breakup + detail normal + grass-statistics
>    roughness/AO through the spine BRDF) from ~250 m to the horizon. This is
>    what makes fields read lush from the air; near-zero memory.
> 2. **GPU per-frame generation replaces the CPU megamesh tiles** (Phase 5
>    pulled forward, §13) — for **every scatter layer**, grass first, then the
>    tree impostor band, shrubs, rocks. Geometry is regenerated each frame from
>    deterministic seeds + resident control data (the GPU height/mask atlas the
>    gate already mirrors); nothing persistent per blade/quad. Memory becomes
>    O(visible), the revision-rebuild machinery dissolves (next frame simply
>    regenerates), and the OOM class is deleted rather than tuned.
>
> Until the GPU path lands, the shipped CPU rings stay as-is (band 0–1
> stand-ins); do not extend their reach — that scales the broken memory curve.
>
> **Status (2026-07-04):** band 2 (grass-as-terrain-shading) **landed** —
> two-scale footprint-faded field detail + `shade_surface` transmit lobe in
> `body_terrain.wgsl`/`lighting.wgsl`, user-verified in-game (amplitudes
> trimmed once). **GPU grass slices 1 + 1.5 landed, preview-verified /
> game-UNVERIFIED**: the vertex-synthesized grass field
> (`body_render::ground::gpu_grass` + `game::rendering::gpu_grass`) replaces
> the **entire CPU grass clipmap** behind `GraphicsSettings::gpu_grass`
> (default on; off = full CPU fallback). One template mesh (slot-encoding
> vertices, ~780 k blades over **five** density bands to **340 m** — band 4
> is the card-scale far ring, so the CPU card ring parks too; a
> screen-space minimum blade width, `GG_MIN_WIDTH_RAD`, keeps the far
> bands from stochastically rasterizing away — sub-pixel blades were the
> in-game "grass just stops" bug), blades
> derived per frame in the vertex shader from body-global cell hashes + a
> CPU-filled 768² height/mask control window (async, rebuilt on ~25 m
> drift / terrain revision); per-blade memory and the rebuild-churn
> machinery are gone. Altitude collapse raised to 250–500 m AGL (was
> 150–300) so climb-out keeps a live sward below.
>
> **The high-fidelity blade pass (2026-07-04, the Ghost-of-Tsushima recipe;
> preview-verified / game-UNVERIFIED).** All in `gpu_grass.wgsl`:
> - **Wind**: a scrolling two-octave value-noise **gust field** bends whole
>   Bézier blades from the root (visible rolling waves, arc-length-ish
>   preserved), per-style stiffness ± per-blade jitter; the shared
>   `grass_displace` sway stays on as reduced tip flutter.
> - **Shading**: rounded cross-blade normal blended toward the terrain
>   normal with distance (near = individual blades, far = one cohesive
>   lawn), specular sheen lobe, t-weighted translucency, root ambient
>   occlusion — both root-darkening and AO **fade with band** so far blades
>   converge on the terrain's own `vegetation_color` (the MSFS rule; this
>   is what killed the polka-dot far field).
> - **View-dependent widening** (up to +90 % edge-on) so blades never thin
>   to nothing while the camera moves.
> - **Clump coherence**: per-clump facing/height/hue over per-blade jitter;
>   clump footprint ≥ 45 % of the band's cell so coarse bands interleave
>   (the carpet fix). Medium-scale (~9 m) mottle patches + moisture-driven
>   **dry-straw blade mix** (luminance-normalized straw hue).
> - **Grass types are data**: the WGSL profile consts are gone — styles
>   (dry / lush / lawn) live in the `GpuGrassParams::style` uniform table,
>   authored via `GrassStyle` (a `GrassProfile` + dry_mix/sheen/stiffness)
>   in `gpu_grass.rs::gpu_grass_style_table`. Every fill site must install
>   the table (zeroed styles = zero-size blades).
> Slice 2 stays: compute cull/compact + `draw_indirect`, then the tree
> impostor band onto the same path. Exercised headlessly in `just preview`
> (`gpu_grass_field_{side,top,3q}` over a synthetic flat window).

> No silent rewrites — when a phase lands, fold its notes here and update the
> roadmap.
>
> **Realistic-tree rework (2026-06-25, runtime-unverified).** Two changes replace
> the original "blobby intersecting trees" look:
> 1. **Blue-noise placement** — `scatter::build_scatter_tile` previously placed
>    each candidate at a uniform-random `(u,v)` (complete spatial randomness), so
>    canopies routinely interpenetrated. It now does **Poisson-disk** placement on
>    a body-global hashed cell grid with priority elimination: one grid per
>    `VegLayer` (all tree species share it, drawn per point by `mix_weight`), sized
>    to the layer's largest `min_spacing_m`. Every tree is ≥ `min_spacing` from
>    every other tree (no interpenetration), canopies still touch into groves, and
>    it is deterministic + seamless across tiles (unit-tested). `VegSpeciesPlacement`
>    swapped `density_per_m2` → `min_spacing_m` + `mix_weight`.
> 2. **Leaf cards + translucent foliage shading** — the canopy is no longer a solid
>    ellipsoid blob; it's alpha-tested **leaf-cluster cards** (no solid core — each
>    card's texture has an opaque-ish centre so overlapping cards self-cover), each
>    carrying a (dappled) outward normal so the flat cards light like a soft volume
>    (SpeedTree "puffiness"). Crown shapes (`CanopyStyle`): the **broadleaf**
>    (`push_broadleaf_canopy`) grows a small recursive **branch skeleton** (trunk-top
>    fork → main limbs → Pipe-Model-tapered splits, biased outward + up via
>    phototropism — the core idea of Makowski et al.'s *Synthetic Silviculture*) and
>    hangs a rounded **foliage cluster** at every branch tip; only the trunk + thick
>    first limbs are drawn (twigs hide inside the canopy). A guaranteed **egg-shell**
>    of clusters + a `crown_profile` height envelope keep the crown a full rounded
>    mass regardless of how the skeleton grew, while the branch tips add the irregular
>    outer lobes. **Shrubs** (`Round`, `push_canopy`) are a Fibonacci-skin + lobe
>    ellipsoid clump. (The **conifer** `push_conifer` drooping-needle-spray path still
>    exists but is currently unused — the pine is being rebuilt from scratch.)
>    Cards/limbs/trunk sample the procedural **foliage atlas**
>    generated by `thalos_texgen` (small multi-toned leaves + needle spray + bark;
>    `build_foliage_atlas` wraps it into a GPU image) via a packed `cell·4+corner`
>    code in `UV_1.y`. The atlas carries the real leaf colour; `canopy_color` is a
>    light per-species tint × AO. `tree.wgsl` alpha-tests the atlas and adds a
>    **two-sided translucency** term (backlit leaves transmit a warm glow).
>
> **Shared foliage material model (2026-06-29).** A tree's leaf/bark colour now
> has **one definition** — the `thalos::foliage` WGSL library
> (`crates/rendering/render/src/shading/shaders/foliage.wgsl`, `foliage_base_albedo` +
> `foliage_hue_tint`), the *albedo* analogue of `thalos::lighting::shade_foliage`.
> Both the near mesh trees (`tree.wgsl`) and the **impostor bake**
> (`tree_bake.wgsl`) derive their colour from it, on the same atlas sample + baked
> AO, so the impostor captures **exactly** the near-tree colour and the
> mesh→impostor handoff cannot drift; change the look in one place and the near
> canopy and the (startup-rebaked) impostor band move together. This replaced the
> earlier split where the bake stored the raw atlas green × vertex tint while
> `tree.wgsl` graded it to a muted olive — so the far band read brighter/greener
> than the near trees. The grade is now view-independent (the sun-facing `env`
> brightening moved into the shared lighting), which is what makes it bakeable.
> **Enforced in `just preview`:** `object_preview` renders the broadleaf as both a
> mesh and its octahedral impostor (`tree_broadleaf` vs `tree_broadleaf_impostor`),
> so the parity is self-verifiable (read the PNGs) on every future tree/shrub
> change — the harness also exercises all three tree shaders + the shared library
> as a compile/render smoke test. *Verified by preview; not yet `just game`-verified.*
>
> **Still interim (Slice 3+):** the depth/shadow pass is alpha-unaware, so trees
> cast roughly-solid (un-dappled) shadows; lower mesh LODs read sparser than the
> LOD0-baked impostor (a *fluffiness*/coverage gap distinct from the colour one —
> the next tuning pass is constant-coverage on the LOD chain); no atlas mips (far
> cards may shimmer); conifer cones use needle-card texturing but aren't yet
> drooping needle cards; no real branch structure. See the "Realistic trees"
> slice plan.

---

## 1. Philosophy and scope

Three principles drive every decision here:

1. **Vegetation is a consumer-side layer over the tile contract.** It reads only
   the runtime terrain seams — `HeightSource` (GPU-atlas mirror with CPU
   fallback), the material-mask grass gate (`material_masks_from_heights`), and
   `sea_level_m`. The terrain *generation* revamp can replace everything behind
   those seams without touching vegetation. Vegetation never owns terrain state.

2. **"As far as the eye can see" ends in the terrain albedo, not infinite
   instances.** The vegetated terrain shader (`body_terrain.wgsl`) already
   paints grass-colored ground with the `C_FOREST`/`C_GRASS`/`C_DRYGRASS`
   palette. Every vegetation layer is a *representation cascade* that adds 3D
   detail while it is perceptible, then hands off **invisibly** to the ground
   color that is already there. The eye reads vegetation to the horizon because
   the ground is vegetated; geometry only adds texture where you are close
   enough to resolve it.

3. **Procedural = deterministic placement + per-instance variation, not
   per-object geometry synthesis.** A plant's existence, position, and
   appearance are a pure hash of its location. A small library of
   pre-generated species meshes is scattered with per-instance scale / yaw /
   tilt / tint / wind-phase variation, so you can never spot the same plant
   twice at once, with zero stored transforms. We do **not** generate unique
   geometry per plant at runtime — that is what makes batching possible.

**Layers.** One unified system, three layers with different payloads and
cascade tails:

| Layer | Payload | Cascade tail | Lattice |
|---|---|---|---|
| `GroundCover` (grass) | batched blade/clump megamesh per tile | clump cards → terrain albedo | clipmap (tile size grows per ring) |
| `Shrub` | instanced low-poly mesh | mesh LODs → fade out (no impostor) | finer clipmap |
| `Tree` | instanced mesh | mesh LODs → octahedral impostor → terrain fold | coarser clipmap |

**Out of scope (for now):** aquatic / underwater plants (water is disabled
until the generator grows a sea level — terrain rewrite Slice 1); crop fields
and authored placement; destructible / interactive vegetation beyond
wind/downwash bending; seasonal phenology beyond the existing altitude tint
bands.

---

## 2. Current state (shipped 2026-06)

The grass blade near-ring — the Phase 1.0 baseline. See `docs/world/terrain.md` for
the canonical description; summarized here because the plan extends it.

- **Lattice:** self-contained cube-sphere lattice, ~25 m tiles
  (`GRASS_TILE_SIZE_M`), `GrassTileKey { face, x, y }`.
- **Placement:** deterministic `blade_hash` jittered grid; gate = above
  `sea_level_m + 1 m`, slope ≤ 0.45, grass material-mask dominant, altitude
  fade 2400→3100 m, plus the **building-terrain scatter regions** (below). ~24
  candidate blades/m² before gates.

**Building-terrain scatter (`scatter::ScatterRegion`/`ScatterTreatment`/`classify_scatter`).**
A base's flattened ground is grassland, so the scatter layer treats it as
*managed* terrain instead of excluding it wholesale: each per-frame footprint
declares `Clear` (paving / building / pad / tank — skip every blade) or `Lawn`
(force a tidy short-thick `GrassProfile::lawn` cover, bypassing the natural
grass-mask / coverage gates). `Clear` always wins over `Lawn`. The grass driver
(`game::rendering::grass`) derives the regions from the `StructureRegistry`
(`BaseSite` → `Lawn`, runway/building/launchpad/tank → `Clear`), so a spaceport
reads as lawn between the structures with bare paving under them. This is the
seam the future tree/prop scatter on a base plugs into (thread the same regions
into `build_scatter_tile`); see `docs/gameplay/base_building.md` *Ground scatter*. Off-base
the region set is empty, so wild terrain is unaffected. Trees/rocks still use the
older `TerrainFlatten` `nearest_flatten` exclusion.
- **Payload:** one batched `Mesh` per tile of curved tapered 7-vertex blade
  strips, built on `AsyncComputeTaskPool`.
- **Anchoring:** root-grid big_space child re-posed in f64 every frame (runway
  pattern); f32 `Transform.rotation` only ever acts on ≤ ~20 m vertex offsets.
- **Shading:** `GrassMaterial` / `grass.wgsl` — wrap-diffuse against the sun
  mirroring the vegetated terrain constants; per-blade wind sway; dithered
  distance discard (70→100 m); ship-layer only.
- **Lifecycle:** active-body pick (nearest `Vegetated` body with a
  `HeightSource`), nearest-first build dispatch, hysteresis despawn, periodic
  revision-driven rebuild when ground under a tile moves > 5 cm.

**Hard ring at ~100 m** is the first thing this plan removes.

---

## 3. Architecture

### 3.1 Code layout

- **`crates/rendering/render/src/ground/tile_lattice.rs`** *(new, refactor)* — the
  cube-sphere lattice math (`cube_face_uv` / `cube_dir` / `tile_frame` /
  `tiles_per_side` / `tile_uv_span`), lifted out of `vegetation.rs` into a
  `TileLattice { tiles_per_side }`. Grass switches to it with no behavior
  change. One lattice definition shared by grass, shrubs, trees — they cannot
  drift.
- **`crates/rendering/render/src/ground/scatter.rs`** *(new, pure)* —
  `VegLayer`, `VegSpeciesPlacement`, `VegInstance`, `VegScatterTile`,
  `build_scatter_tile`, `clump_field`, the shared `placement_gate` helper, the
  instanced material (`VegInstancedMaterial`), and the foliage impostor material
  (`FoliageImpostorMaterial`). No Bevy beyond mesh/material types, like
  `vegetation.rs`.
- **`crates/rendering/render/src/ground/vegetation.rs`** *(existing)* — keeps the
  grass blade/clump megamesh builder + `GrassMaterial`. Gains blade-LOD
  variants (7-vert → 3-vert → crossed-quad clump).
- **`crates/runtime/game/src/rendering/vegetation.rs`** *(new, driver)* — the unified
  driver: `SpeciesLibrary` resource + drive / finalize / anchor / rebuild / LOD
  systems, a generalization of today's `grass.rs`. Hosts trees and shrubs
  first; grass folds in at Phase 4.
- **`crates/runtime/game/src/rendering/grass.rs`** *(existing)* — stays as the grass
  driver until Phase 4 folds it into the unified driver.
- **Impostor bake** lives near `scatter.rs` (foliage octahedral atlas), distinct
  from the top-level `body_render::impostor` module (distant *planet*
  billboards) to avoid confusion.

### 3.2 Shared foundations (reused by every layer)

- **Cube-sphere tile lattice**, keyed by `tiles_per_side` (per LOD ring).
- **Deterministic hashed placement** — the `blade_hash` integer-mix family,
  generalized to `veg_hash(seed, tile_key, species, candidate, salt)`.
- **Shared `placement_gate`** — the grass slope/mask/normal/altitude/sea-level
  probe block, factored into one function. *One* definition of "where
  vegetation can grow," so layers never disagree.
- **f64 per-tile anchoring** — `update_grass_transforms` verbatim: each tile is
  a root-grid big_space child, the multi-Mm body-fixed offset rotated in f64
  every frame, the f32 transform acting only on small local offsets.
- **Async build + revision rebuild** — build on `AsyncComputeTaskPool`,
  snapshot `HeightSource::revision()`, rebuild stale tiles whose ground moved.
- **Shared wind field** — one `VegWind` resource (today's grass wind
  computation) read by all layers, so grass, shrubs, and canopies move
  coherently.

### 3.3 Species library

Built once at startup into a `SpeciesLibrary` resource. Per species: an LOD
mesh chain (`Vec<Handle<Mesh>>`, near→far), the instanced material, the
octahedral impostor atlas, a shadow-cutoff LOD, and a placement-params struct
(`VegSpeciesPlacement`) that is also handed (as an `Arc<[…]>`, asset-handle-free)
to the async build. Procedural generation (space colonization, see §10) runs
here, at library-build time only.

---

## 4. Placement

### 4.1 Deterministic blue-noise (Poisson-disk) scatter

Per **layer** (not per species — all species of a layer share one grid): a
body-global hashed cell grid sized to the layer's largest `min_spacing_m`
(`cells = tiles_per_side(radius, spacing)`). Each cell hashes to one jittered
candidate + a priority; a candidate **survives** only if no higher-priority
candidate within `min_spacing` exists in its ±2-cell neighbourhood (Poisson-disk
elimination). The species at each surviving point is drawn by `mix_weight`, then
the gate + clump + accept roll thin it (thinning only removes points, so the
min-spacing guarantee holds). This guarantees no two plants of a layer
interpenetrate while canopies still touch into groves.

The grid is **global** (a function of the cube cell, not the tile), so the
candidate set and every elimination are identical from any tile that overlaps a
region: placement is deterministic, seamless across tile boundaries (no clipping
or double-placement at seams — unit-tested), and frame-coherent across LOD
transitions (a plant's root is identical at 10 m and 2 km, so transitions don't
swim). Each tile *owns* the cells whose jittered position falls inside it; halo
cells act only as elimination blockers. (Earlier this was uniform-random
per-tile scatter — complete spatial randomness — which is why trees grew into
each other.)

### 4.2 The shared gate

Every candidate passes the same gate, factored out of the grass builder:

- above `sea_level_m + 1 m` (no aquatic plants yet);
- slope ≤ species `slope_limit` (grass 0.45; trees lower);
- grass material-mask channel dominant (`material_masks_from_heights` — the
  exact stencil the tile baker writes, so vegetation matches what the ground
  *looks* like);
- altitude bands (lush → fade → none), per species;
- `TerrainFlatten` pad exclusion (runway, future structures);
- terrain normal returned by the gate orients the instance to local up.

### 4.3 Clumping field

Uniform scatter is the number-one tell of procedural vegetation. A
low-frequency `clump_field(dir, layer, affinity) -> [0,1]` breaks it:

The shipped model splits into a **position-only canopy field** (`forest_coverage`,
shared by every layer) and a **per-sample terrain coupling** (`woody_terrain_factor`,
applied in `build_scatter_tile` because it needs the height sample), so the forest
reads as a real ecosystem rather than a stamped patch:

- **`forest_coverage(dir)` — canopy potential, two scales.** A large-scale stand
  field with a **wide ecotone ramp** (`smoothstep(FOREST_LO-0.06, FOREST_HI+0.12,
  mask)`, window `0.52/0.72`) so density *feathers in over a broad band* instead of
  a hard edge, multiplied by a **medium-scale glade field** (~130 m) that carves
  real **internal clearings** and breaks the uniform interior. Lower `FOREST_LO`
  for more forest, raise it for emptier plains; `GLADE_FREQ_MUL` sets clearing size.
- **`woody_terrain_factor(sample)` — clumping tied to the landform.** Trees/shrubs
  thin toward **ridges and steep faces** (`slope_factor`, fading to 0 as slope
  approaches the species limit) and thicken in **sheltered concave hollows**
  (`curvature` boost; convex knolls get cut). `PlacementSample.curvature` is the
  gate's discrete Laplacian (free — it already samples the four neighbours). This
  is what makes a stand thin up a ridge and pool in a hollow, so ecotones hug
  terrain. Tuned by `CURVATURE_GAIN/STRENGTH` + `SLOPE_THIN_FRAC`.
- **Trees** run at `affinity = 1` so the plains stay **genuinely treeless** — the
  ecotone + glades supply the gradual plain→forest falloff and the non-uniform
  interior.
- **Shrubs** peak in the stand-margin (ecotone) band plus a few lone bushes on
  the plain, and share the terrain coupling.
- **Grass** density dips inside closed canopy and peaks in clearings. The far
  clipmap rings additionally **cull grass under canopy** (`forest_cull ×
  forest_coverage(dir)`, ramping 0 → 0.95 over rings 2–4): a distant grove's
  ground grass is occluded by the trees in front of it, so rendering it is pure
  overdraw — but because the cull reads the same `forest_coverage`, grass **returns
  in the internal glades**. Near rings (0–1) keep all grass, so the forest floor
  is still grassed where you actually see it.

**Process-first nuance (CLAUDE.md).** The "no naked macro fBM" rule governs
visible *terrain height/albedo*. Vegetation *distribution* breakup is exactly
the "masks, placement, breakup" the rule permits noise for — but the clump
field must still be **gated by the biome/slope/altitude masks**, never paint
vegetation onto rock or water. When the terrain contract grows a richer
biome/moisture query (see §11), species selection and clumping should read it
instead of approximating biome from the grass-mask channel + altitude.

---

## 5. LOD and the representation cascade

### 5.0 The four-band cascade (redesign 2026-07-03)

Every vegetation layer is one instance of the same four-band structure, and
**all four bands are driven by the one landcover function**
(`thalos::landcover` / its CPU mirror) — the MSFS lesson: grass density and
color are *derived from* the field that colors the far ground, so a descending
camera watches the same pixels grow blades and there is no boundary to hide.

| Band | Range (grass) | Representation | Memory |
|---|---|---|---|
| 0 | 0 → ~80 m | full per-blade geometry, GPU-generated per frame | none persistent |
| 1 | ~80 → ~300 m | degraded blades / cards, coverage-conserving thinning | none persistent |
| 2 | ~250 m → horizon | **terrain shading**: grass detail layer in `body_terrain.wgsl` | zero |
| 3 | aerial/orbital | landcover tint in the terrain albedo (already shipped) | zero |

- **Bands 0–1 are generated, never stored** (Ghost of Tsushima model): a
  compute pass per tile hashes blade seeds, samples the resident height/mask
  atlas, culls (distance/frustum/occlusion) in the same dispatch, and appends
  survivors to an instance buffer for one `draw_indirect`. GoT ships ~83k
  visible blades in ~2.5 ms on PS4 hardware; that is the budget envelope.
- **Band 2 is a material pass, not objects.** Past the card ring the terrain
  shader itself carries grass *statistics*: albedo breakup, a detail normal,
  root-AO, and grass-tuned roughness/translucency through `shade_surface` —
  so far grass inherits shadows, aerial recession, and sky ambient by
  construction (the one-world principle applied to grass).
- **Band transitions are coverage- and color-exact.** The classic aerial
  failure — a visible disc around the camera — is a coverage/brightness step
  at a band boundary. Band 1 thinning widens survivors (constant coverage),
  band 1→2 cross-fades over an overlap zone, and both sample the same
  landcover color and the same wind field (a boundary invisible at rest still
  shows in motion if only one side moves).

### 5.1 The constant-coverage rule (the one that prevents bald ground)

The classic failure is reducing instance *count* with distance and leaving the
ground showing through. Instead hold **coverage** (fraction of ground hidden by
green) roughly constant while count falls:

```
coverage ≈ density × element_footprint        →  keep ~constant
as density ρ drops with distance, footprint ∝ 1/ρ  (elements grow)
```

A far "blade" is not a blade — it is a **clump** representing N blades, wider
and shorter, so one clump card at 300 m hides as much ground as 20 blades did
up close. The builder drives both `density` and `clump_scale` from the tile's
LOD ring.

**This rule applies only to *unresolvable* elements (grass).** It depends on the
individual element being sub-pixel, so the eye reads only aggregate coverage and
can't tell a fat clump card from the blades it replaces. **Trees break that
assumption:** an individual tree stays resolvable to well past the near rings, so
growing it to hold coverage just makes a visibly *giant* tree, and its apparent
size snaps at every ring boundary as the grow factor steps (2026-07-01: this was
the "far impostors look huge and shrink as you approach" report). So the tree
cascade holds `grove_scale = 1` on every clipmap ring — trees keep their true
size at every LOD — and the coarse rings carry coverage by density
(`spacing_scale`, decimated only gently) plus the forest-tinted terrain albedo
underneath, **not** by enlarging the survivors. Grass keeps the enlarge-to-hold
rule; trees (and any future individually-resolvable scatter) do not.

### 5.2 Per-layer cascades

**Grass (`GroundCover`)** — never an impostor:

| Band (ground level) | Representation | Density | Element |
|---|---|---|---|
| 0–60 m | full 3D blade (7-vert curved strip) | ~24/m² | blade |
| 60–150 m | reduced blade (3-vert, no arc) | ~6/m² | wider blade |
| 150–500 m | clump card (1–2 crossed quads, tuft texture) | ~1/m² eq | tuft (~20 blades) |
| 500 m–~1.5 km | billboard sheet / large clump card | very low | patch |
| beyond | terrain albedo only | — | — |

At eye height the geometric horizon on Thalos is only ~3.4 km
(`√(2·R·h)`, h ≈ 1.8 m), most of it grazing-angle/sub-pixel, so the geometry
cascade need only reach ~1–1.5 km before the terrain color legitimately carries
the rest. From altitude, far grass is sub-pixel and the cascade costs nothing.

**Shrub** — mesh LODs → fade out (too small to read as a billboard):
0–120 m LOD0, 120–250 m LOD1, beyond → dithered fade to nothing.

**Tree** — mesh LODs → octahedral impostor → terrain fold:
0–150 m LOD0, 150–500 m LOD1, 500–1200 m LOD2, 1200 m–`TREE_IMPOSTOR_MAX`
hemisphere octahedral impostor, beyond → culled (already in terrain albedo).

**Rock (`Rock`)** — near-only, **no impostor, no clipmap** (landed 2026-06-29,
runtime-unverified). Scattered pebbles / cobbles resolve only up close, so the
whole cascade is one fine ring out to ~100 m (mesh fades out at the reach,
nothing beyond). Two things make rocks distinct from the woody layers:

- **Placed inversely to grass.** The `Rock` branch of `build_scatter_tile`
  weights acceptance by `bare = 1 − grass_w` (the terrain shader's grass mask):
  stones gather on bare / rocky ground and thin to a small floor density
  (`ROCK_GRASS_FLOOR`) under thick grass, which visually covers the smaller ones
  anyway — so pebbles show in the rocky gaps between tufts, where real ones do.
  A medium-scale `rock_scatter_field` gathers them into loose scree clusters;
  they tolerate much steeper slopes than plants and ignore the runway clearing
  margin (gravel may approach the apron). They take any orientation (worn stones
  lie / half-bury at a jaunty angle), unlike near-upright plants.
- **A deformed-icosphere mesh, not a card.** `ground::rock_mesh` deforms a
  subdivided icosphere by 3-D gradient noise (lumps) + an ellipsoid squash
  (flattened, water-worn pebbles), smooth or faceted normals, with a baked
  cavity-AO / sun-bleach gradient on the vertex colour. A small library of a few
  distinct shapes/tones is scattered through one Poisson grid (so no two stones
  interpenetrate) and combined into one batched mesh per tile
  (`combine_rock_tile_mesh`), lit through the shared `thalos::lighting`
  rough-dielectric BRDF (`RockMaterial` / `rock.wgsl`) — the same surface model
  the ground uses — and receiving (and casting into) the trees' sun-shadow
  cascade. The driver (`game::rendering::rocks`) is a trimmed single-ring copy
  of the grass driver.

### 5.3 Tile-LOD clipmap

A fixed-size tile ring cannot reach the horizon (a 25 m tile out to 1.5 km is
~5000 tile entities). Use a **clipmap on the cube-sphere**: concentric rings
where tile size doubles each ring, so each ring is a thin annulus of ~50–100
tiles and the total stays bounded (~a few hundred tiles to the horizon).

```rust
struct VegRing {
    lod: u32,                 // 0 = finest
    tile_size_m: f64,         // BASE << lod  (25, 50, 100, 200, 400, 800…)
    tiles_per_side: i64,      // tiles_per_side(radius, tile_size)
    inner_m: f64, outer_m: f64,
}
```

This mirrors the UDLOD terrain LOD pyramid, applied to the vegetation lattice.
A tile takes the LOD of the ring whose annulus contains it; the **same**
`build_scatter_tile` / blade builder serves every ring, parameterized by
`(density, mesh_lod, clump_scale)`. The driver maintains all rings.

### 5.4 Smooth transitions

Never hard-swap, and **no dither** — dither shimmers without TAA and reads as a
visible "fade." Instead the transitions are **geometry scale-fade**: plants
*grow from zero* as they enter range and shrink back at the far edge, so the edge
is seamless and a fully-collapsed plant is a degenerate (invisible) mesh — no
discard needed. (Landed 2026-06-25.)

- **Trees** scale the whole mesh about the trunk base, per instance, by a grow
  factor from the focus distance (`tree.wgsl`).
- **Grass** collapses each blade's height toward its root (`uv.x · color.a`),
  with the per-ring near/far cross-fade so adjacent clipmap rings grow/shrink
  through their shared boundary (`grass.wgsl`).
- **No build pop-in:** tiles are built a full tile *beyond* the fade-out edge, so
  a tile finishes building while its content is scaled to ~0; it then grows in on
  approach.
- **No re-LOD vanish:** a tree tile's mesh LOD is swapped **in place**
  (`relod_veg_tiles`) as it approaches — no despawn/rebuild gap.
- **Zoom-independent:** all fade/LOD distances are measured from the **craft
  anchor** (`GrassParams.anchor`), never the camera, so camera zoom/orbit can't
  change what's drawn.

---

## 6. Rendering and instancing

The placement, LOD, anchoring, and rebuild layers are **identical** regardless of
how instances become draws.

**Landed model: one batched mesh per tile** (2026-06-25) — every layer (grass,
shrubs, trees) bakes a tile's instances into a single mesh and spawns *one*
entity per tile. Grass bakes blades; trees/shrubs **combine** their pre-built
species `TreeMeshData` per instance (transform + append) via
`combine_tree_tile_mesh`, baking each tree's base into `UV_0`/`UV_1` so the
shader still scale-fades and wind-varies per tree. This removes the per-tree ECS
entity overhead (the real ceiling — draw calls were already fine via batching),
so forests scale to dense/far with no custom render pipeline. Re-LOD and
revision-rebuild bake a new tile mesh and swap on completion (old kept until
ready → no vanish).

> The earlier "Option A (entity per tree) → Option B (instance buffer) → GPU
> indirect" escalation was the original sketch; the per-tile batched mesh (the
> grass path, generalized to combine arbitrary species meshes) reached the same
> scaling goal with far less risk and no new render plumbing, so it superseded
> Option A/B. A true GPU-driven cull + `draw_indirect` path (Horizon-style)
> remains the option if per-tile meshes ever become the ceiling — but the entity
> ceiling, which was the actual limit, is gone.

**Why not meshlets/Nanite for vegetation.** Foliage is *aggregate geometry*
(disconnected leaf cards / thin twigs) — the worst case for cluster
simplification, which wins on contiguous opaque surfaces. Bevy's virtual
geometry (0.16) has no alpha-mask support and is not production-ready, and even
UE's Nanite foliage costs ~2× per masked triangle and still uses impostors for
distant LODs. Meshlets may eventually help Thalos's *hard surfaces* (terrain
rock, hull, stations) — not its plants. The cascade above is the correct path.

---

## 7. Octahedral impostors (tree far band)

A single camera-facing quad sampling a pre-baked atlas of the tree from many
angles, with a depth channel for view-blend parallax — looks volumetric, costs
one quad. Specifics:

- **Hemisphere** capture, not full sphere: you never see a tree from below, so
  spend the resolution on side views.
- Baked **once at library-build time** per species (albedo + normal + depth).
- Used only in the far tree band; breaks down up close (parallax), so it never
  overlaps the near mesh LODs.
- The impostor→culled handoff coincides with the tree already being baked into
  the terrain albedo, so a forested continent reads correctly from orbit with
  zero drawn instances.

This is the piece most worth prototyping carefully — it is what sells
mid-to-far forests.

### 7.1 As implemented (2026-06-25)

Engine side in `body_render/src/ground/tree_impostor.rs` (+ `tree_impostor.wgsl`,
`tree_bake.wgsl`); driver + bake orchestration in
`game/src/rendering/vegetation.rs`. The mesh-tree path is unchanged; the far
band (`lod_for_dist` LOD3, ≥ ~1.2 km) swaps mesh → impostor.

- **Atlas.** Per tree species, an `N×N` (default 8×8) grid of hemisphere views,
  species stacked vertically in **two** linear `Rgba16Float` atlases: albedo +
  coverage, and object-local normal + depth. One `ImpostorAtlasLayout`; adding
  species or raising `N` is data-driven (constants in the driver).
- **Bake (grid-of-rotated-copies, not N² cameras).** One instance of the
  recentred species LOD0 mesh is spawned per (species, view cell), **rotated** so
  a single orthographic camera's −Z view captures that cell's hemioctahedral
  direction; the instances tile the atlas grid, so **one ortho camera per
  channel** (two total) bakes everything in one pass. The off-screen rig
  (`ImpostorBakeRig`, on dedicated layers 6/7, `Hdr` + `Tonemapping::None`)
  renders for a fixed number of frames to cover async pipeline compilation, then
  `tick_impostor_bake` tears it down and flags the band ready; the atlases retain
  the captured content. Normals are stored **object-local** (not world) so the
  runtime re-lights each tree in its own terrain frame.
- **Runtime.** `combine_impostor_tile_mesh` emits one quad per tree (base in
  `POSITION` — degenerate for the standard prepass, so impostors never touch a
  custom prepass pipeline; terrain up in `NORMAL`; corner in `UV_0`;
  `(scale, species)` in `UV_1`; `(tint, yaw)` in `COLOR`). The vertex billboards
  the quad to the captured view basis, sizes it from the per-species bounding
  sphere, and applies the **same craft-anchor scale-fade** the mesh trees use.
  The fragment hemioctahedral-encodes the camera→tree direction, **bilinearly
  blends the 4 surrounding captured views** (coverage-weighted, so silhouettes
  don't ghost), alpha-tests, rotates the blended object-frame normal to world,
  and lights it through `thalos::lighting` — matching the mesh trees and ground.
- **Seamless / zoom-independent.** Both materials share one lighting + fade
  parameter set; the fade band moves to the impostor far edge once baked, so mesh
  trees (all well inside) never fade and only the far impostors grow/shrink at the
  edge. The mesh→impostor swap is the existing in-place re-LOD (spawn new,
  despawn old → no vanish). Impostor tiles are `NotShadowCaster`.
- **Coarse impostor clipmap ring** — ✅ landed 2026-06-27 (see the status note at
  the top). Coarse grove rings (tile 500/1000/2000 m, `spacing_scale ≈
  grove_scale ≈ tile/200`) extend reach ~4.8 → ~22 km at a bounded quad count.
- **Deferred (follow-ups):** true depth-channel parallax (the channel is baked
  but the runtime blend doesn't ray-offset yet); per-tile frustum-cull `Aabb`
  (impostor meshes are `RENDER_WORLD`, so currently never frustum-culled — fine
  while clumping keeps far clearings empty, revisit if draw count bites);
  reaching past ~22 km from high altitude is the **terrain-albedo handoff**'s job
  (§9), not more impostor rings.

---

## 8. Wind

One `VegWind` resource (today's grass wind: tangent-plane direction at the
camera, slow veer) read by every layer so the whole stack moves as one — the
single biggest "it's alive" multiplier.

- **Grass:** per-blade phase-shifted sway, UV.x-weighted (root→tip), as shipped.
- **Trees/shrubs:** **hierarchical** — per-vertex wind weights baked into the
  mesh (trunk ≈ 0 → branch → leaf ≈ 1) times the shared wind, with per-instance
  phase from the world-position hash. Two layers of gust noise (a coarse
  scrolling pattern + a finer one) shared across grass and canopy, the Ghost of
  Tsushima recipe.
- **Later:** localized bending under the player and engine downwash.

---

## 9. The terrain handoff (seam polish)

Reaching far is easy; making boundaries invisible is the work.

1. **Color match.** Far clump cards / billboard sheets converge to the *exact*
   terrain grass-band albedo at the same pixel (sample the same `material_masks`
   band the terrain shader uses), so there is no luminance step where geometry
   stops. Blade tints already track the palette — extend to the far elements.
2. **Ground detail in the mid band.** Where blades thin out (~150 m+), bare
   terrain looks flat. Fade a **grass detail normal + albedo breakup** into
   `body_terrain.wgsl` across ~150 m–1 km (in as blades fade out) so the ground
   reads as grassy texture, not a smooth surface. This is what actually hides
   the geometry→albedo seam.
3. **Root blend.** Fade each blade/clump toward the ground color at its base
   (vertex-color driven) so plants melt into the terrain instead of sitting on
   it as cards. Cheap, large readability win for dense fields.
4. **Two-scale ground sampling** in the terrain shader to kill visible tiling in
   the albedo that now carries the far field.

---

## 10. Procedural asset generation

At **library-build time only**, never per-instance at runtime:

- **Trees/shrubs:** space-colonization branch growth (scatter attraction points,
  grow toward them) → trunk mesh + leaf cards; bake the LOD chain + the
  hemisphere octahedral impostor + per-vertex wind weights. A handful of base
  trees per species.
- **Grass:** the existing parametric blade/clump builder; clump-card textures
  baked from dense blade renders.
- **Per-instance variation at runtime** is purely scale + yaw + tilt + tint +
  wind-phase from the position hash. A few base meshes per species cover a
  planet with no visible repetition.

SpeedTree-style parametric authoring or L-systems are viable alternatives to
space colonization for the offline step; the runtime contract (a baked LOD +
impostor + wind-weight bundle per species) is unchanged either way.

---

## 11. Dependencies and seams

- **HeightSource** — already consumed; the placement substrate.
- **Material-mask grass channel** — already consumed as the biome approximation.
- **Richer biome/moisture query (future).** Good species selection and clumping
  want more than height + grass-mask + altitude: a biome id and a moisture /
  aridity field. This is a **terrain-contract** addition (an extended
  `SurfaceQuery`), tracked with the terrain rewrite. Until then, approximate
  from the grass-mask channel + altitude bands. Vegetation must not reach behind
  the seam for it.
- **`sea_level_m`** — gates aquatic exclusion; water rendering remains separate
  (water is not a terrain material).
- **`TerrainFlatten` / structures** — placement exclusion under pads.

---

## 12. Shadows

- **Near mesh LODs** cast CSM shadows (`casts_shadows_to_lod` cutoff per
  species) — trees especially need contact with the ground to feel planted.
  Note the craft-shadow-caster-layer gotcha: casters must share the light's
  render layer.
- **Impostors** do not cast (or cast a cheap proxy) — past the cutoff.
- **Grass** stays `NotShadowCaster` (as shipped); receiving soft AO/contact
  shadows near roots is a later polish item.

---

## 13. GPU-generated grass (endgame)

The CPU async-bake model is great for the near ring and fine for the clump-card
far rings (few elements), but baking a full blade clipmap every time the camera
moves is the scaling wall for *dense* grass far out at low altitude. The real
"infinite, lush, smooth-LOD" answer is **GPU-generated grass** (the Ghost of
Tsushima model): a compute shader emits blades per visible cell each frame, with
density / blade-LOD / clump-size computed per cell from distance + the height
source, no CPU tile baking at all.

Treat this as a later rewrite of the **generation step only** — the clipmap
rings, the constant-coverage rule, the terrain handoff, the wind field, and the
f64 anchoring all carry straight over. It removes the async-bake bottleneck and
is the natural home for downwash/player bending done in-shader.

---

## 14. Design invariants

- **Vegetation reads only the terrain runtime seams.** `HeightSource`, the
  material-mask gate, `sea_level_m`, flatten pads. No reaching behind the tile
  contract; the generation revamp must be replaceable underneath vegetation.
- **One placement gate.** All layers accept candidates through the single
  `placement_gate` helper. No per-layer reimplementation.
- **One tile lattice.** All layers and rings use `TileLattice`; the cube math
  has exactly one definition.
- **Placement is deterministic and view-independent.** A plant's root position
  is a pure hash of location, identical at every distance and LOD, so
  transitions never swim and nothing is stored.
- **f64 per-tile anchoring is mandatory.** Every tile is a root-grid big_space
  child re-posed in f64 each frame; the f32 transform only ever rotates small
  local offsets. (The runway/grass pattern — anything else jitters under warp.)
- **Cascades end in the terrain albedo.** Far vegetation is the ground color,
  not drawn geometry; the geometry layer's only job is the near-to-mid detail
  and an invisible handoff.
- **Coverage, not count, is held constant with distance.** Far elements grow to
  keep ground coverage; never thin to bald.
- **Library at build time, variation at runtime.** No per-instance geometry
  synthesis; batching depends on shared meshes.
- **One foliage material definition.** A plant's intrinsic leaf/bark colour is
  computed by exactly one function — `thalos::foliage::foliage_base_albedo` — that
  the near mesh shader *and* the impostor bake both call. No representation
  (near LOD, impostor) re-implements the look; the impostor is derived from the
  near material, so it tracks every tree/shrub change automatically. The parity
  is checked in `just preview` (mesh vs impostor side-by-side). Anything
  view/sun-dependent stays in the lighting (`shade_foliage`), never the albedo,
  so the material is bakeable.
- **Ship-layer only.** Vegetation never appears in the map view.

---

## 15. Implementation roadmap

Each slice is screenshot-verifiable on its own. The user runs the game and
sends screenshots (agents do not launch it).

### Phase 0 — Foundations (no visible change) ✅ landed
- **0a** Extract cube math → `ground::tile_lattice::TileLattice`; grass switches
  over, behavior identical.
- **0b** Factor the grass slope/mask/normal/altitude/sea-level block into a
  shared `placement_gate` helper; grass uses it, behavior identical.
- **0c** Stand up `scatter.rs` (`VegLayer`, `VegSpeciesPlacement`,
  `VegInstance`, `VegScatterTile`, `build_scatter_tile`) and the unified driver
  skeleton `game::rendering::vegetation` (drive/finalize/anchor/rebuild),
  initially hosting nothing.

### Phase 1 — Grass to the horizon (extends shipped grass) ✅ 1a–1c landed · 1d open
- **1a** Blade-mesh LODs: add 3-vert blade and crossed-quad clump to the grass
  builder, selected by a `lod` field; drive `density` + `clump_scale` from it
  (constant-coverage rule).
- **1b** Clipmap rings: generalize the grass driver to N coarser-lattice rings;
  fill each ring's annulus; push reach to ~1–1.5 km.
- **1c** Per-ring dither cross-fade (extend the existing fade-band discard).
- **1d** Terrain handoff: grass detail normal/breakup in `body_terrain.wgsl`
  fading in 150 m–1 km; far-clump color match; root blend; two-scale ground
  sampling. *Goal: grass visibly stretches to the horizon with no seam.*

### Phase 2 — Trees ✅ 2a–2d landed (2d runtime-unverified)
- **2a** `SpeciesLibrary` with one tree species (LOD0 only); `build_scatter_tile`
  → Option A entity-per-instance; reuse gate + anchoring + rebuild. *Goal:
  anchored trees, rock-steady under warp.*
- **2b** Clumping field + altitude/biome gating + shader-hashed per-instance
  variation (tint/phase/scale jitter).
- **2c** Mesh LOD chain + dither cross-fade; CSM shadows on near LODs.
- **2d** ✅ Hemisphere octahedral impostor far band + off-screen startup bake
  (§7.1): far band → impostors out to ~3.6 km, seamless mesh→impostor handoff.
  Deferred polish: depth-parallax, far-tile frustum cull, impostor clipmap ring.

### Phase 3 — Shrubs / undergrowth ✅ 3a landed (basic species) · 3b open
- **3a** `VegLayer::Shrub` on a finer clipmap; clusters at forest edges (reads
  tree grove mask); mesh LODs → fade out (no impostor).
- **3b** Root blend + integration with the grass ground cover.

### Phase 4 — Unify and scale
- **4a** Fold grass into the unified `VegLayer` driver (shared ring/clipmap/
  anchoring; grass keeps its megamesh payload). Retire `grass.rs`.
- **4b** Option B instanced material for shrub/tree density.
- **4c** GPU-driven culling + indirect draw.

### Phase 5 — GPU-generated vegetation (generation rewrite) **← pulled forward (2026-07-03); slice 1 landed 2026-07-04**
- **Slice 1 ✅ (vertex-synthesized field, preview-verified/game-unverified):**
  `ground::gpu_grass` — template mesh + per-frame vertex generation from
  body-global cell hashes + a scrolling height/mask control window; replaces
  the CPU blade rings 0–1 (see the status note in the header). Cheap and
  simple, at the cost of always running the full template through the vertex
  stage (~2.3 M verts; rejected blades collapse to degenerate strips).
- **Slice 2 (next):** compute pass generates + culls (distance/frustum/
  occlusion) + compacts instances → `draw_indexed_indirect`; extend the bands
  through the card ring to ~340 m and retire the CPU card tiles.
- Compute-shader blade emission per visible cell; density/LOD/clump from
  distance + height source. Clipmap/coverage/handoff/anchoring/wind carry over.
- **Scope broadened to the whole stack:** the compute-generate → cull →
  compact → `draw_indirect` node is designed to host every scatter layer.
  Order: grass blades/cards (deletes the grass OOM), then the tree impostor
  band (one quad per tree is the ideal compute payload; deletes the coarse
  impostor tile meshes), then shrubs/rocks. Near tree *mesh* tiles (LOD0–2)
  stay CPU-batched for now — they are few and need real meshes — but their
  count/reach should shrink as the impostor band moves closer.
- Sequenced **after** band 2 (the §5.0 terrain-shading layer, formerly Phase
  1d), which is the visual win and needs no new infrastructure.

### Cross-cutting (as needed)
- Procedural library generation (space colonization) — gate of Phase 2a.
- Wind field unification (`VegWind`) — fold grass + new layers onto one source.
- Richer biome/moisture `SurfaceQuery` — coordinated with the terrain rewrite.

---

## 16. Deferred / non-goals (for now)

- Aquatic / underwater plants (no sea level yet).
- Authored / hand-placed vegetation and crop fields.
- Destructible vegetation; advanced player/downwash interaction beyond wind.
- Seasonal phenology beyond altitude tint bands.
- Exact cube-face-seam coverage (a small gap at seams is accepted, as in grass
  v1).

---

## 17. Code pointers

- Lattice/scatter (engine): `crates/rendering/render/src/ground/tile_lattice.rs`,
  `scatter.rs`, `vegetation.rs`, `grass.wgsl`.
- Driver (game): `crates/runtime/game/src/rendering/vegetation.rs` (new),
  `grass.rs` (until Phase 4).
- Seams: `crates/rendering/render/src/ground/height_source.rs` (`HeightSource`),
  `pipeline.rs` (`material_masks_from_heights`),
  `rendered_height.rs` (`TerrainPatchBasis`).
- Terrain shader handoff: `crates/rendering/render/src/ground/body_terrain.wgsl`.
- Anchoring reference: `crates/runtime/game/src/runway.rs`
  (`update_runway_transform`), `crates/runtime/game/src/rendering/grass.rs`
  (`update_grass_transforms`).
- Related specs: `docs/world/terrain.md` (tile contract + shipped grass layer),
  `docs/rendering/atmosphere.md` (wind/weather neighbor systems),
  `docs/simulation/surface.md` (on-foot context).
