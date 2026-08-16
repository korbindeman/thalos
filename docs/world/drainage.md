# Drainage

The canonical plan for deriving watersheds, lakes, river channels, terrain
conditioning, and inland water from Thalos's completed neural terrain. This
document owns the `HYD-*` namespace. The backlog is the status authority; this
document owns the contract, sequencing, open forks, and acceptance gates.

## §1 Purpose and boundary

Drainage is an **offline derived product of the final neural heightfield**. It
is not a parallel procedural approximation and it is not computed from runtime
`SurfaceQuery` point samples in production. The solve happens after every
immutable neural height band that can move a divide, pass, valley, or outlet has
been composed.

The current 2 km equirectangular bake is a preview adapter. It proves the
hydrology machinery on a whole planet, but it cannot be shipping authority: its
grid is too coarse to react to the final mountains and it visibly imprints D8
stair-steps at local scale.

Drainage has four separate responsibilities that must not collapse into one
field:

1. **Topology:** where water can flow, which cells share a basin, and which
   outlet or closed depression receives it.
2. **Water budget:** catchment area, runoff, seasonality, and discharge. A large
   arid basin is not necessarily a strong river.
3. **Geomorphology:** lakes, reach geometry, floodplains, incision, deposition,
   and the constrained height delta applied back to terrain.
4. **Presentation:** landcover response and actual water surfaces. Neither is
   the drainage solver itself.

Dynamic structure flattening, player construction, and transient weather do not
rebake global drainage. Authored permanent terrain changes belong in the source
height package before its hash is frozen; local gameplay drainage around later
construction is a separate small-area system.

## §2 Fixed decisions

- **Post-neural authority.** The completed neural DEM is the height source.
  Analytic or coarse preview terrain never substitutes for missing final bands.
- **One signed height authority.** Drainage, terrain carving, rendered ground,
  collision, and water placement agree on the same composed height package.
- **Offline and deterministic.** Identical source hashes and configuration
  produce byte-identical drainage artifacts independent of thread count and
  work ordering.
- **Global topology first.** A continuous whole-planet band establishes basin
  ownership and continental water budgets before any local refinement.
- **Conserved local refinement.** Finer neural windows may move tributaries and
  local divides, but must inherit and return explicit boundary flux so they
  cannot create or destroy continental water.
- **Catchment is not discharge.** Geometric upstream area and climate-weighted
  flow remain separate package channels.
- **Closed basins are explicit.** Depression filling is a routing tool, not a
  claim that every natural basin drains to the sea. Lakes, endorheic basins,
  salt pans, and numerical pits receive different policies.
- **Provenance is binding.** Every output records exact height, climate, datum,
  algorithm, configuration, and upstream-drainage hashes. A mismatch is a hard
  validation failure, never a warning.
- **Static first.** The base package carries climatological drainage. Seasonal
  presentation may vary flow strength, but real-time planet-wide fluid or
  erosion simulation is outside the base-planet bake.

## §3 Current foundation

`thalos_terrain::hydrology` now supplies a terrain-independent raster solver:
priority-flood depression routing, spherical equirectangular cell metrics,
gradient-based D8 receivers, geometric catchment, and climate-weighted
annual-mean discharge. Synthetic tests pin deterministic routing, input
immutability, depression escape, spherical area conservation, and runoff
conservation at ocean outlets.

The 50.1 Mpx diffusion preview reproduces the earlier macro topology: largest
catchment 4.238 million km² and mean Horton-Strahler bifurcation ratio 4.75.
The new discharge channel reaches p99 51 m³/s, p99.9 1,967 m³/s, and a maximum
19,979 m³/s. These are validation evidence, not final Thalos geography.

Known limitations of the foundation:

- every depression is presently connected to an ocean seed;
- the receiver graph is D8 and locally grid-visible;
- runoff is a deterministic macro-moisture proxy rather than final climate;
- the solver is in-memory and sized for the 2 km preview, not yet the production
  global band;
- no lake equilibrium, reach graph, channel carving, delta, or water surface is
  emitted;
- preview sidecars identify the backing but do not yet bind exact neural and
  climate content hashes.

## §4 Authoritative data contract

### Inputs

| Input | Contract |
|---|---|
| Composed height | Signed metres relative to the body's water datum, after all immutable neural residuals at that band. Includes every detail allowed to move drainage at the solve scale. |
| Grid geometry | Exact cell-to-body mapping, spherical cell area, neighbour metric, seam topology, and physical sample spacing. Production is cube-sphere or another package-native mapping, not implicitly equirectangular. |
| Ocean and outlet mask | Derived from signed height plus authored water connectivity rules. A below-datum closed basin is not silently treated as open ocean. |
| Climate | At minimum precipitation, temperature or potential evapotranspiration, and seasonality on a compatible grid. Missing fine climate inherits conservatively from its parent band. |
| Permanent masks | Authored exclusions or immutable terrain interventions that existed before the height hash was frozen. Runtime/player flatten regions are excluded. |
| Parent flux | For a refined window: water entering each boundary segment, parent basin identity, permitted exits, and the parent water budget that must be returned. |
| Provenance | Body, datum, source-band hashes, climate hashes, producer/model identity, algorithm version, configuration hash, and parent-drainage hash where applicable. |

### Outputs

| Output | Purpose |
|---|---|
| Receiver or fractional-flow graph | Downhill topology with explicit ocean, lake, endorheic, and boundary terminals. |
| Catchment | Geometric upstream area independent of climate. |
| Discharge | Climatological flow plus the chosen seasonal/intermittency representation. |
| Depression hierarchy | Nested basins, spill saddles, fill volume, contributing area, terminal type, and equilibrium water level where applicable. |
| Reach graph | Seam-free vector centre-lines with stable reach ids, order, upstream/downstream links, slope, width, depth, bankfull flow, and intermittency. |
| Terrain delta | Bounded carve/deposition residual with its own hash; the original neural height remains recoverable. |
| Water geometry | Lake polygons and river reach geometry suitable for LOD generation, with surface elevation and shoreline metadata. |
| Validation receipt | Source and output hashes, timings, peak memory, conservation residuals, topology metrics, gate verdicts, and failure reason. |

Derived raster caches may exist for fast tile sampling, but the reach graph and
basin hierarchy are the semantic authority for rivers and lakes. A display
raster alone cannot preserve connectivity or generate stable arbitrary-altitude
water geometry.

## §5 Work order

### HYD-1 — production source and provenance contract

Define the exact package-native height/climate views consumed by hydrology and
the validation receipt written by every bake. Include hard stale-artifact
rejection. Resolve the interface without waiting for final content; bind the
actual band hashes when NTR-X3 freezes them.

**Exit:** fixtures prove that changing any height, climate, datum, algorithm, or
configuration hash invalidates every dependent drainage artifact.

### HYD-2 — depression inventory and basin hierarchy

Extend priority flood to retain the information it currently discards: original
pit floor, spill saddle, nested basin parent, contributing area, fill volume,
maximum/mean fill depth, outlet, and provisional terminal type. Do not change
the shipping topology policy yet; first make every candidate measurable.

**Exit:** synthetic nested pits, coastal lagoons, through-flow lakes, and closed
basins reproduce their known hierarchy and water volume exactly.

### HYD-3 — routing kernel and angular-bias gate

Build rotation-equivalent synthetic mountains, cones, planes, saddles, and
cross-seam basins. Compare the existing spherical D8 baseline with D-infinity,
multiple-flow-direction accumulation, and continuous channel tracing. Choose
the simplest kernel that passes the angular-bias and conservation gates.

The likely production shape is fractional D-infinity or MFD for accumulation,
followed by one stable centre-line per extracted reach. This is a recommendation,
not a decision until the rotated fixtures distinguish it from D8.

**Exit:** receiver/flow results are invariant within a declared tolerance under
grid rotation and cube-face seams; no polar or face-edge preference remains.

### HYD-4 — climatological runoff and closed-basin water balance

Replace the macro-moisture proxy with the finalized neural climate channels.
Compute precipitation, potential evapotranspiration, infiltration/base loss,
snow storage/melt where climate requires it, and runoff. Classify each
depression by long-term balance:

- open through-flow lake;
- seasonally or permanently overflowing lake;
- stable endorheic lake;
- intermittent playa/salt pan;
- numerical pit to fill and route through.

Store mean flow and enough seasonality/intermittency to distinguish perennial,
seasonal, and ephemeral reaches. Monthly bins are the upper useful bound for the
base bake; use a smaller mean-plus-seasonality representation if it reproduces
the same classifications.

**Exit:** water is conserved, equilibrium lake levels converge, and deliberately
arid large basins do not become perennial rivers.

### HYD-5 — deterministic production-scale solver

Turn the in-memory preview solver into a bounded-memory, resumable offline tool
for the chosen global neural band. Partitioning may change storage and work
ordering, never results. Record phase timings, peak RSS, spill/reload counts,
source hashes, output hashes, and failure details in the tool diagnostics lane.

**Exit:** monolithic and partitioned fixtures are byte-identical; interrupted
runs resume without accepting partial output; the target full-planet band fits
the documented bake machine budget.

### HYD-6 — authoritative global solve

After the continuous neural height and climate bands are frozen, run the first
binding whole-planet solve. This assigns every land cell to an ocean outlet,
lake, endorheic terminal, or explicitly invalid unresolved sink. No unresolved
sink may be silently filled.

**Exit:** all global conservation, seam, topology, provenance, and statistical
gates in §6 pass. The artifact is rejected and regenerated whenever an input
hash changes.

### HYD-7 — fine neural-window re-solves

For each finer authored height window, import parent basin ids and boundary
flux, solve using the detailed mountains and valleys, then return flux through
declared exits. Reconcile overlaps deterministically. A local window may move a
tributary or divide inside its authority, but cannot reroute a continental basin
through an undeclared boundary.

**Exit:** a monolithic high-resolution fixture and the same DEM solved as parent
plus windows produce equivalent basin ownership, outlet discharge, and reach
connectivity within declared quantization tolerances.

### HYD-8 — reach extraction and hydraulic geometry

Convert the solved fields into a seam-free vector reach graph. Use discharge
and intermittency for channel initiation; use catchment and Strahler order for
hierarchy. Fit centre-lines independent of the raster's cardinal/diagonal
steps. Derive slope, bankfull width/depth, bed elevation, floodplain envelope,
and confluence geometry from explicit scaling laws and terrain constraints.

**Exit:** stable reach ids survive unrelated window rebakes; confluences are
topologically valid; channels never run uphill; line work remains smooth at
native inspection scale.

### HYD-9 — constrained carve, lake placement, and re-solve

Apply a bounded derived height residual rather than replacing the neural
terrain. The pass may incise channels, open required outlets, form banks and
floodplains, and add constrained alluvium. It must preserve the authored
coastline zero crossing unless an explicit delta/estuary stage owns the change.

Use a measured loop: solve → extract → carve/deposit → re-solve. Stop when the
receiver graph and longitudinal profiles meet convergence gates, not after an
arbitrary pass count. Preserve the original neural DEM and store the terrain
delta separately.

**Exit:** downstream bed profiles are monotone except at classified lakes or
waterfalls; no carve crosses a protected divide; the final solve agrees with
the carved heightfield it ships beside.

### HYD-10 — terrain-package and landcover integration

Package final catchment, discharge, basin, reach, lake, and carve products with
their dependency hashes. Replace the coarse preview channel. Drive riparian
albedo, soil moisture, gallery vegetation, scatter clearing, wetlands, and dry
wash material from reach type and discharge rather than catchment alone.

**Exit:** runtime loading refuses stale or mismatched drainage, tile borders are
continuous, humid and arid rivers remain visually distinct, and disabling the
water renderer still leaves a coherent landcover read.

### HYD-11 — arbitrary-altitude inland water renderer

Render river reaches and lake polygons at their solved elevations. This is
separate from the analytic sea sphere and shares its solution with sub-sea-level
inland seas. Generate LOD water meshes from the semantic reach/lake authority;
bind depth, flow direction, roughness, shoreline wetness, reflection, atmosphere,
and shadowing to the one-world rendering path.

**Exit:** rivers and lakes do not z-fight or disappear across terrain LOD;
water never climbs banks or crosses divides; ocean, inland water, terrain,
craft, and atmosphere obey the same light and depth universe.

### HYD-12 — mouths, floodplains, deltas, and deposition

Add estuaries, distributaries, deltas, fans, terraces, oxbows, wetlands, and
seasonal floodplain masks where discharge, sediment proxy, slope, and receiving
water permit them. This stage may alter the coastline only through an explicit
delta product whose change is visible in provenance.

**Exit:** deposition occurs only where transport capacity falls; distributary
graphs conserve flow; coast changes are bounded, attributed, and regenerate all
downstream shoreline products.

## §6 Verification gates

### Solver invariants

- exact source/config provenance and stale-output rejection;
- deterministic outputs across thread counts, work order, resume, and tiling;
- land area and runoff conserved at ocean, lake, endorheic, and window terminals;
- longitude wrap and cube-face seams preserve neighbour and basin continuity;
- rotated fixtures remain within the declared angular-bias tolerance;
- every non-terminal reach has a valid downstream connection;
- fine-window boundary inflow equals returned outflow plus classified storage;
- carve/re-solve converges and the shipped channels agree with shipped height.

### Whole-planet diagnostics

Report distributions, not one flattering basin:

- basin area, relief, elongation, hypsometry, and outlet class;
- drainage density by climate and relief province;
- Strahler stream counts and bifurcation ratios by scale;
- Hack-style reach length versus basin area;
- catchment versus discharge, including arid outliers;
- endorheic land fraction, lake area/volume, and overflow frequency;
- river slope, width, depth, intermittency, and confluence angle;
- conservation residuals and polar/face-edge directional bias;
- topology stability between global and refined bands.

Earth ranges are calibration references, not universal hard targets. Thalos may
differ where its radius, climate, relief, or lore says it should; every accepted
departure is named rather than hidden by a global average.

### Visual evidence

Every authoritative bake produces a frozen report with:

- global basin and discharge atlases;
- matched catchment-versus-discharge maps;
- representative humid, arid, polar, mountain, plain, lake, endorheic, delta,
  and cross-seam crops;
- receiver-direction and depression-class debug views;
- final height before/after carve plus difference and longitudinal profiles;
- runtime captures of riparian landcover, a river reach, a lake shore, a mouth,
  and terrain/water LOD transitions.

The review must explicitly answer: do channels follow the detailed neural
valleys and mountain passes, are any rivers visibly grid-stepped or uphill, and
does the water budget distinguish wet from merely large basins?

## §7 Open decisions

| Fork | Options | Recommendation / binding gate |
|---|---|---|
| Flow kernel | D8; D-infinity; MFD; continuous vector tracing | Keep D8 only as the baseline. Choose from HYD-3's rotated and seam fixtures; favor fractional accumulation plus stable single-reach extraction if it passes. |
| Seasonal representation | annual mean only; mean + seasonality/intermittency; 12 monthly bins | Mean only is insufficient for ephemeral rivers and lake balance. Prefer the smallest representation that reproduces the 12-bin classification fixture. |
| Production partitioning | full in-memory; striped external memory; hierarchical tiles | Results must be partition-independent. Choose after the final global band dimensions and bake-machine memory budget are known. |
| Carve authority | no carve; hydraulic procedural replacement; bounded residual over neural terrain | Bounded residual wins: neural terrain remains authored truth, while drainage repairs convergence and longitudinal profiles measurably. |
| Inland water geometry | raster mask in terrain tiles; vector reach/lake meshes; hybrid | Prefer semantic vectors with derived tile masks: vectors preserve topology and mesh LOD, masks cheaply drive landcover and shore materials. |
| Climate cadence | static climatology; seasonal presentation; live hydrology | Base package is static climatology. Seasonal rendering may modulate it; live global fluid simulation is a separate future program. |

Resolve a fork only with its named gate. A future implementation must not turn a
temporary preview choice into package authority merely because code already
exists.

## §8 Readiness sequence

The base planet is ready for the definitive drainage bake only when:

1. HYD-1 through HYD-5 and their fixtures are green;
2. the continuous neural height and climate bands are frozen and hashed;
3. HYD-6 produces a valid global authority;
4. HYD-7 incorporates every shipping fine neural window;
5. HYD-8 and HYD-9 produce a converged reach graph and carved height delta;
6. HYD-10 packages and validates the result.

HYD-11 and HYD-12 make that drainage visible as water and mature geography.
They do not authorize an earlier coarse or stale drainage bake.
