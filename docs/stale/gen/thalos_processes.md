# Thalos — Process Reference

Companion to the surface generator design doc and a sibling of `mira_processes.md`. The architecture doc specifies *how* the pipeline runs; this doc specifies *what each stage in Thalos's recipe is physically modeling and visually producing*.

Thalos's full recipe, chronologically:

```
Sphere → Differentiate → Plates → Tectonics → Hotspots → Topography
       → Hydrology → Climate → Biomes → Detail
(Atmosphere is a shader-side pass, not a bake stage.)
```

Thalos is a 3,186 km iron-cored homeworld with a thin silicate mantle, ~65% ocean cover, 0.85 atm N₂/O₂ atmosphere, 23° axial tilt, and a day close to Earth's. Plate tectonics was vigorous for the first ~2.5 Gyr, building continents, stabilising the carbon cycle, and enabling complex life; it is now declining as the planet's small thermal reservoir runs down and the lithosphere thickens into stagnant lid. The visible present is a handful of mature continents separated by old ocean basins, snowcapped peaks at the few remaining active margins, polar ice caps, and a weather system carrying pale blue skies and white-on-blue cloud cover.

The visual goal at impostor distance is "instantly readable as a habitable, Earth-lineage ocean world — but old and stately, not young and restless."

Thalos's recipe is fundamentally different from Mira's. Mira is an airless moon whose surface is an arrested impact record. Thalos is a tectonically sculpted world with a hydrological cycle: impacts are erased, continents are built, mountains are lifted and ground down, water and ice partition the surface into climatic zones. The stage list reflects that — no Cratering, no MareFlood, no SpaceWeather. Instead: tectonics, topography, drainage, climate, biomes. Where Mira needs thousands of analytic crater primitives, Thalos needs coherent geological structures (plates, mountain belts, drainage networks).

**Implementation status (2026-04-17):** the basic pass implements Differentiate + Biomes only. The rest of the pipeline is designed in this document but not yet coded. Thalos currently reads as "ocean world with fractal continents and polar ice" — correct in broad strokes, but flat, with no plate structure, no mountains, no rivers, no climate-derived biome variation. The remaining stages are the roadmap.

---

## Architectural decisions

Three decisions shape the rest of the pipeline and are worth making explicit before the per-stage detail.

### 1. The tectonic model is quasi-static, not integrated over time

Full plate-motion simulation over 4.6 Gyr is a genuine research problem — topology changes (plates merging and splitting) break naive kinematic integration, and the results are hard to tune toward specific visual goals. The Thalos pipeline skips integration entirely.

Instead, the Plates stage uses **Euler-pole kinematics** to produce *self-consistent relative motions* without ever stepping time forward. Each plate carries an Euler pole and an angular velocity. Boundary types (convergent / divergent / transform) are derived analytically from the pair of poles evaluated at each boundary midpoint. Ages — how long a boundary has been convergent, how long ago an orogen last peaked — are sampled from distributions rather than integrated, skewed to produce Thalos's "declining phase" narrative (most orogens old and worn, a minority young and active).

This gives up the ability to reproduce a specific historical sequence ("plate A collided with plate B 1.8 Gyr ago, producing mountain range X"). It keeps the ability to produce a geologically coherent *present state*: plate boundaries with consistent relative motions, orogens whose ages feel right, ocean-floor-age maps that correctly encode the spreading / subduction balance. For a playable planet, present state is what matters.

### 2. Erosion is not simulated — terrain is generated to look eroded

Stream-power-law hydraulic erosion on a cubemap is expensive and iterative, and it couples badly to the pipeline's pure-transform stage architecture. The Topography stage skips it entirely and uses **Íñigo Quílez's analytical-derivative noise** instead: each octave's amplitude is suppressed by the accumulated slope magnitude from previous octaves. This produces sharp ridges where slopes build up and smooth valleys where they cancel — the morphological signature of erosion, without running an erosion sim.

Drainage networks — the *other* output of physical erosion — are produced by a separate Hydrology stage using **Diffusion-Limited Aggregation** seeded on continental interiors. The DLA branches define river courses; the Hydrology stage carves valleys along them and deposits alluvial flats at river mouths.

One gap to paper over: neither IQ noise nor DLA produces the *sediment-flat lowlands* that physical erosion deposits in basins and river mouths. The Hydrology stage fakes this by flattening terrain along the last N kilometres of each DLA trunk and injecting a low-amplitude floor near confluences. The result reads as "a planet with coastal plains and river deltas" without any physical deposition.

### 3. Continental plates cluster loosely, not randomly

Fully random plate-type assignment (each plate independently continental or oceanic) produces a checkerboard that does not visually read as continents. One supercontinent is a specific phase of Earth's history, not a neutral default. **Loose clusters** — 3–5 continental groupings, each 1–3 adjacent continental plates — give Thalos the visual distribution of present-day Earth: distinct landmasses with their own characters, coherent enough to travel between.

The Plates stage implements this by picking N seed plates to be continental, then biasing each neighbour of a continental plate toward also being continental. Iterate until the target continental area fraction (~0.35) is met.

---

## The visual goal: a calm, old planet with a few dramatic anchors

Thalos's terrain hinges on a single aesthetic claim: the planet looks *tired*. Most of it is the landscape of deep time — rolling shield cratons, worn-down ancient mountain belts, wide alluvial plains, long slow rivers. Alpine drama is concentrated in one or two places; elsewhere the feel is stately rather than restless. What Earth might look like a billion years from now, after its heat engine has wound down.

Stage-by-stage, the visual character is roughly:

- **The ocean (65% of the surface)** is not a featureless blue. Depth shading runs from pale continental shelves to deep abyssal plains. Fewer and more muted mid-ocean ridges than Earth. Ocean floor on average *older* than Earth's because slowing subduction doesn't recycle it as fast — some seafloor is 500–800 Myr old, buried in thick sediment.

- **The old majority of continental land.** 60–70% of continental area is one of: shield cratons (vast rolling plains over ancient basement, a few hundred metres of relief, the size of continents themselves); ancient orogen belts (worn peaks 1500–2500 m, rounded, broadly forested — Appalachians, Urals, Scottish Highlands); passive margins (wide coastal plains, massive deltas).

- **The dramatic minority.** One or two active orogens where continents are still actively colliding (Himalayan-class, 7–8 km peaks, sharp ridges, ice-capped, glaciers). One or two Andean-style subduction margins (coastal chains 4–6 km with arc volcanoes). One active rift valley (linear lakes, volcanic activity, thin new crust). One or two hotspot chains (Hawaii-scale, young end still active as islands).

- **The distinctive Thalos signature — rust.** Exposed ancient orogen cores and shield crust — anywhere vegetation is thin over old bedrock, in semi-arid continental interiors or high plateaus — carry a subtle ruddy/ochre tint from oxidised iron. The metal-rich composition quietly staining the landscape. An iron world that looks the part.

- **Hydrology.** Long meandering rivers across old plains, bigger and older than Earth's average (no recent glacial reset has wiped them). Dendritic drainage everywhere. Enormous slow-accumulating deltas. Endorheic basins common in continental interiors.

- **Climate and biomes.** Earth-like latitudinal bands, slightly tamer weather from the 0.85 atm atmosphere. Rainforests, savannahs, temperate forests, grasslands, taiga, tundra, ice — placed by a zonal circulation + orographic precipitation model, not pure latitude heuristics.

The combined feel from orbit: a blue-white world with distinct continents that read as "lived-in" — a few dramatic mountain regions as visual anchors, most continental area reading as quiet deep-time landscape, polar caps, subtle rust undertone where old rock is exposed.

---

## Sphere

**Physical model:** the body's reference shape. Thalos is a rotating body in hydrostatic equilibrium; its flattening (rotational oblateness) is on the order of 1 part in 300, below visual threshold at impostor scale. The surface generator treats it as a sphere and leaves oblateness to whatever precision orbital dynamics needs it.

**What it sets:** the radius of the body and the coordinate frame for everything downstream.

**Parameters:** `radius_m = 3_186_000`.

**Visual contribution alone:** a featureless smooth sphere. Useful as a sanity check.

---

## Differentiate

**Physical model:** the thermal evolution of the body. Thalos accreted in the metal-rich inner disk at 1 AU from Pyros; the resulting body is a large iron core (~70% by volume) overlain by a thin silicate mantle and crust. The iron core segregated and remained molten under a solid inner core; its outer liquid shell drives Thalos's strong magnetic field, which protects the atmosphere from stellar-wind stripping. The thin mantle concentrated lithophile elements (U, Th, K) and produced a crustal system dominated by feldspathic continental crust (low-density, buoyant) and mafic oceanic basalt (dense, subductable). The dichotomy between continental and oceanic crust is the central fact of the surface: continents float, basins subduct.

Secondary materials emerge from the hydrological cycle: continental shelves and river deltas accumulate sediment; polar regions freeze out a thin water-ice cap that advances and retreats seasonally; in the current declining-tectonics era, continental interiors develop thick sedimentary cover.

**What it sets:** the materials palette. Downstream stages paint with these named materials.

**Materials defined for Thalos:**
- *Continental crust (felsic)*: bright, rough, granite-to-granodiorite bulk composition. The upper surface of continents. Albedo ~0.20, reddish-tan when dry, green-brown under vegetation.
- *Oceanic basin (basalt)*: dark, smooth under water. Albedo ~0.07 (dry); through deep water the visible albedo is dominated by water absorption + Rayleigh scattering, ~0.06.
- *Continental shelf / sediment*: intermediate tan. Shallow water, suspended sediment raises apparent albedo near coasts. ~0.12 effective.
- *Polar ice cap*: the brightest material. Fresh snow over sea or land ice. Albedo ~0.75.
- *Vegetation veneer*: a shading modifier, not a substrate. Applied to continental crust in habitable-climate latitudes. Tints base continental albedo toward muted olive-green. Canopy contribution to roughness.
- *Exposed-basement rust tint*: a red-oxide modifier applied to continental crust where vegetation is thin over ancient shield (see Biomes). Ochre/rust shift, a few percent of albedo.

**Parameters:** `composition = iron_core_silicate_crust`, per-material albedo ranges, roughness values, vegetation and ice and rust tint curves.

**Visual contribution alone:** still a featureless sphere, but downstream stages now know what to colour it with.

---

## Plates *(future stage)*

**Physical model:** the lithosphere's partition into discrete rigid plates. Thalos began with vigorous plate tectonics (~30–50 plates, active spreading, active subduction) and has since declined toward stagnant lid; most old boundaries have locked up, a minority remain active. The present-day plate geometry is what this stage produces.

Plates are treated as rigid Voronoi cells on the sphere. Each carries:
- an **Euler pole** (unit vector on the sphere — the axis around which it rotates relative to the mantle),
- an **angular velocity** (rad/Myr — positive or negative gives rotation sense).

The Euler-pole formulation is the key trick: the relative motion between any two plates at any boundary point is determined analytically from the two Euler poles, without integrating time. If plates A and B converge and plates B and C converge, the relative motion of A and C falls out automatically. Transforms and triple junctions emerge for free.

**Plate type.** Continental plates carry buoyant felsic crust; oceanic plates carry dense mafic crust. This distinction drives everything downstream — subduction is only possible between oceanic and continental (or oceanic-oceanic); orogens only form at convergent boundaries involving continental crust; ocean-floor age only applies to oceanic plates.

**Continental clustering.** Plate type is *not* assigned independently per plate. The stage picks 3–5 continental seed plates, then gives each neighbour of a continental plate a ~50% probability of also being continental, iterating until the target continental area fraction is hit. This produces loose continent-like clusters, matching present-day Earth's distribution.

**What it sets:** a `PlateMap` — plate ID per cubemap texel — plus a per-plate `Plate` record (id, kind, centroid, Euler pole, angular velocity) and the plate-adjacency graph with boundary polylines.

**Tidal asymmetry:** none. Plate tectonics is driven by internal heat flow, not rotation.

**Parameters:** `n_plates` (~30–50 for a 3186 km body), `continental_area_fraction` (~0.35 for Thalos), `n_continental_seeds` (3–5), `neighbour_continental_bias` (~0.5), `angular_velocity_distribution` (mostly 0.1–1°/Myr with a long left tail at ~0 for stagnant plates).

**Visual contribution alone:** none. Plates is a structural stage. Its output steers Tectonics, Topography, Hotspots, and Climate.

**Implementation approach:** spherical Voronoi over ~40 seeds spread by 2–3 Lloyd relaxation iterations. Either a dedicated spherical Voronoi crate or stereographic-projected planar Delaunay (`spade`) with dual-pole stitching. Per-cubemap-face planar Voronoi with seam handling is a fallback; more code for worse results.

---

## Tectonics *(future stage)*

**Physical model:** activity at plate boundaries, classified by relative motion and plate types. Each boundary segment falls into one of three categories:

- **Convergent** — plates moving toward each other.
  - Continental + Continental: collisional orogen (Himalayan-style), highest possible uplift.
  - Oceanic + Continental: subduction, arc-volcanic mountain belt on the overriding continental side (Andean-style).
  - Oceanic + Oceanic: subduction, island arc on overriding oceanic plate (Aleutian-style).
- **Divergent** — plates moving apart.
  - Oceanic + Oceanic: mid-ocean ridge, generates new ocean floor at the spreading rate.
  - Continental + Continental (rare, snapshot of active rifting): rift valley, linear lakes, thin new crust, volcanic activity.
- **Transform** — plates sliding past each other. Fault systems, minor topographic relief.

**Boundary ages.** Each boundary has an `establishment_age` — how long it's been in its current configuration. A currently-convergent boundary may have been convergent for 200 Myr or for 3 Gyr; older convergent boundaries have more cumulative uplift. The declining-phase throttle reduces intensity contributions after ~2.5 Gyr of body age, so boundaries active only in the last Gyr produce less uplift than ancient active ones.

**`is_active`** marks whether a boundary is currently moving. In Thalos's declining era only ~20% of boundaries are active; the rest are stagnant but still carry their historical record. A stagnant ancient orogen has worn peaks, a structural lineament, maybe a root — it just isn't rising any more.

**Ocean-floor age** for each oceanic cell is distance to nearest divergent boundary divided by spreading rate. Gives the young-near-ridges / old-near-trenches pattern. On Thalos, the modern spreading rate is lower than Earth's (~2 cm/yr vs Earth's ~5 cm/yr), so ocean-floor age distributions skew older — more 500–800 Myr seafloor than Earth has.

**What it sets:** the per-cell `TectonicCell` record:
- `plate_id`, `plate_kind`
- `dist_to_boundary` (km)
- `nearest_boundary_kind` (Convergent | Divergent | Transform)
- `nearest_boundary_age` (Myr)
- `ocean_floor_age` (Myr; None for continental)
- `orogen_intensity` (0..1)
- `orogen_age` (Myr since peak)
- `orogen_axis` (tangent direction, for ridge alignment in noise)
- `boundary_activity` (0..1; low = stagnant lid)

Every downstream topographic / climatic / biome decision reads some subset of this.

**Tidal asymmetry:** none.

**Parameters:** `active_boundary_fraction` (~0.2), `establishment_age_distribution` (peak ~2 Gyr, skewed old), `orogen_age_peak_gyr` (~2.0), `modern_spread_rate_cm_per_yr` (~2), `declining_phase_throttle_start_gyr` (~2.5), `orogen_intensity_weights` (speed × age × continental fraction).

**Visual contribution alone:** none directly. Tectonics authors fields that Topography and downstream stages consume.

**Implementation approach:** walk the plate adjacency graph, classify each boundary segment from Euler-pole differences at the midpoint, draw per-boundary attributes from the parametric distributions, then rasterise `TectonicCell` over the cubemap with distance falloffs from boundaries.

---

## Hotspots *(future stage)*

**Physical model:** intraplate volcanism driven by deep mantle plumes rising through the lithosphere. The Earth archetype is Hawaii: a stationary mantle plume, a drifting plate above it, a chain of volcanic islands marking the plate's motion over time (youngest over the plume, older in the direction of past plate motion).

Because Thalos's tectonics is declining, hotspots are proportionally *more* visually significant than on Earth — less competition from plate-boundary volcanism means plume tracks are a bigger fraction of the total volcanic signature. Expect a Hawaii-scale chain or two; the youngest ends are active islands and submerged seamounts, the older ends eroded flat-topped guyots or worn volcanic ridges.

**Large igneous provinces (LIPs)** — episodic giant eruptions that blanket regions with flood basalts — are the rarer, older cousin. A few ancient LIPs may be baked in as dark basalt provinces on continental interiors or as thick oceanic plateaus.

**What it sets:** per-texel volcanic contributions to topography (hotspot cones, seamounts, guyots, LIP flows), added on top of the Plates/Tectonics structural output.

**Tidal asymmetry:** none.

**Parameters:** `n_hotspot_chains` (2–3), `plume_rate_Myr` (cone spawning interval along the chain), `chain_length_km`, `LIP_count` (0–2), `LIP_radius_km`.

**Visual contribution alone:** small but distinctive. Linear chains of volcanic cones transitioning to submerged guyots across oceanic plates; one or two broad dark basaltic provinces on continental interiors.

**Implementation approach:** pick plume points on the sphere, use the overriding plate's Euler pole to compute plate motion vector at that point, stamp a chain of cones along that vector with ages increasing with distance from the plume. Amplitude and preservation decay with age.

---

## Topography *(future stage)*

**Physical model:** the final height field, produced by layering tectonic and hotspot contributions through a slope-suppressed noise synthesis that mimics the morphology of eroded terrain without an erosion pass.

**Base height (isostasy).**
- Continental cells: buoyant crust, baseline ~+500 m above sea level.
- Oceanic cells: dense crust, baseline depth depends on ocean-floor age (half-space cooling curve): ~−2500 m near ridges, ~−5500 m in old abyssal plains.

**Orogen contributions.** Orogen intensity from the Tectonics stage adds high-amplitude modulation along boundary lineaments. The orogen axis direction biases fractal noise toward ridge-parallel alignment. Orogen age trades amplitude for smoothness:
- young active orogens → high amplitude, high frequency, sharp ridges (IQ noise with low slope-suppression → spiky peaks);
- old worn orogens → low amplitude, low frequency, smooth ridges (high slope-suppression → rolling hills);
- cratonic interior (far from any boundary) → strong amplitude suppression, reads as rolling shield.

**Hotspots** are stamped on top.

**The IQ analytical-derivative noise.** Each octave's amplitude is suppressed by the accumulated slope magnitude from previous octaves:

```
accumulator  = 0
slope_accum  = vec2(0)
amplitude    = 1.0
for octave in 0..N:
    (value, grad) = noise_with_gradient(p * frequency[octave])
    slope_accum  += grad
    accumulator  += amplitude * value / (1.0 + dot(slope_accum, slope_accum))
    amplitude    *= gain
```

Where slopes have accumulated (steep terrain), higher-frequency contributions are damped — valleys stay smooth, ridges stay sharp. Where slopes cancel (flat terrain), all octaves contribute equally — plateaus accumulate full noise amplitude. The result reads as eroded without having run erosion.

Per-cell amplitude, frequency, and gain are modulated by the tectonic fields (orogen intensity, age, boundary activity, distance-to-boundary). Young-orogen cells ramp up amplitude and frequency; cratonic cells suppress them.

**Sea level** is set as a scalar to hit the 65% ocean cover target given the height distribution produced above. Cells below sea level become ocean; continental cells whose height dips below sea level become continental shelf.

**What it sets:** the height cubemap (core bake output) plus the land / shelf / ocean mask.

**Tidal asymmetry:** none at Thalos's rotation rate.

**Parameters:** `sea_level_percentile` (authored to hit 55% ocean), `continental_baseline_m` (~+500), `oceanic_baseline_m` (~−3500), `continentalness_bandwidth` (~0.02; controls plate-boundary softness), `octave_count`, `base_frequency`, `gain`, plus per-tectonic-regime amplitude/frequency/gain multiplier tables. Baseline is lerped by a continuous continentalness weight — a weighted-sum of every plate's kind by angular distance — so the height cubemap stays seam-free at cube-face boundaries.

**Visual contribution alone:** enormous. Thalos goes from "flat-continent placeholder" to "planet with mountains, ridges, basins, rolling plains." The single biggest visual jump beyond the basic pass.

**Implementation approach:** run the IQ noise sum per texel, using per-cell parameters looked up from the Tectonics field. The noise primitive wants analytical gradients so the slope-suppression loop doesn't pay 4× for central differences; `thalos_noise` is scheduled to provide this. Until then scalar Perlin with central differences is acceptable at bake time.

---

## Hydrology *(future stage)*

**Physical model:** drainage networks on continental interiors. On any wet planet with relief, precipitation routes downhill, combining into progressively larger rivers and culminating in major drainage systems reaching the ocean (or, in endorheic basins, a terminal lake). The pattern is *dendritic* — branching, fractal, self-similar.

Traditionally this network is extracted by flow-routing on a completed height field. Thalos inverts the dependency: the drainage network is **seeded** on the continents via **Diffusion-Limited Aggregation**, and the Topography stage's output is modified to carve valleys along the DLA branches. This gives dendritic drainage for free without flow-routing, and the network is available for downstream use (hydrology at UDLOD, biomes as a moisture-tracked variable).

DLA seeds a random walker at a random continental-interior cell; each walker random-walks until it touches an ocean outlet or a previously-grown branch, at which point it sticks. Iterating produces an outward-growing branching skeleton with realistic tributary hierarchy.

**Alluvial flats.** To compensate for the absence of physical erosion and deposition, the Hydrology stage flattens terrain along the last N kilometres of each DLA trunk (major river) and injects a low-amplitude noise floor near trunk confluences. This produces deltas and coastal plains as a visual fake rather than a physical simulation. Without this, Thalos's continents would all be spiky uplands with nothing in between.

**What it sets:** a `DrainageNetwork` structure (hierarchical river graph over continental texels) and a `hydrology_modulation` field added to the height cubemap — negative where valleys carve, small positive where alluvial floors build, zero elsewhere.

**Tidal asymmetry:** none.

**Parameters:** `dla_seed_density_per_continent`, `branch_carve_depth_m` (how deep to carve along DLA branches), `alluvial_flatten_length_km` (how far up each trunk to flatten), `delta_floor_amplitude_m`.

**Visual contribution alone:** subtle at impostor distance but cumulatively major. Continents read as *weathered* rather than *extruded*: ridges have smoothed flanks; valleys look like drainage paths rather than noise features; coastal plains pick up light sediment tones where rivers meet the sea. At close range (UDLOD), the drainage network is directly visible.

**Implementation approach:** DLA on each continent in tangent-plane projection (continents are small relative to Thalos's 3186 km radius, so distortion is minor). Seam handling at coasts is a non-issue since rivers reach the sea at coasts by construction. Valley carving is a Gaussian negative stamp along branch polylines; alluvial flats are a per-texel amplitude reduction near trunk ends.

---

## Climate *(future stage)*

**Physical model:** the distribution of temperature, precipitation, and surface wind across the body. Four drivers:

1. **Insolation by latitude.** Annual-mean solar flux falls off with cos(latitude), modified by axial tilt (23° on Thalos). Equator warm, poles cold.

2. **Zonal circulation.** The three-cell atmospheric model — Hadley (equator to ~30°), Ferrel (~30° to ~60°), Polar (~60° to pole) — gives alternating wet/dry latitudinal bands regardless of substrate. Equator is wet (rising air, convective precipitation); subtropics are dry (descending air, deserts); mid-latitudes are wet (poleward air masses precipitating on western margins); polar regions are dry (cold air holds little moisture).

3. **Orographic precipitation.** Where terrain gradients rise against prevailing wind directions, uplifted air cools and precipitates; leeward sides develop rain shadows. Interacts directly with Topography's height field: mountain ranges near coasts produce wet windward slopes and dry leeward interiors.

4. **Ocean currents.** A simplified rule model — warm currents flow poleward on western ocean margins, cold currents flow equatorward on eastern margins — modifies coastal climates. West coasts at mid-latitudes are mild and wet; east coasts are more continental.

**What it sets:** a `temperature_cubemap` (annual mean, °C) and `precipitation_cubemap` (annual total, mm/yr). Intermediate data — not visualised directly but drives Biomes.

**Tidal asymmetry:** none.

**Parameters:** `equator_temperature_c`, `polar_temperature_c`, `axial_tilt_deg` (23), a prevailing-wind-direction table per zonal cell, `orographic_gain`, `ocean_current_amplitude`.

**Visual contribution alone:** none directly; intermediate data.

**Implementation approach:** zonal circulation is ~200 lines of rule-based per-cell computation. Orographic precipitation is a dot product of wind vector against local terrain gradient (available from the IQ noise's analytical derivative). Ocean currents are a rule-table lookup keyed on latitude and coast-side.

---

## Biomes

**Physical model:** climatic and geological partitioning of the surface. The Biomes stage is the sole stage that directly authors albedo at impostor distance — everything upstream authors structure; Biomes is where structure becomes colour.

Inputs: temperature (Climate), precipitation (Climate), elevation (Topography), latitude, substrate type (Differentiate + Tectonics), boundary distance (Tectonics — for the rust signature).

**Partition hierarchy:**

1. **Ocean vs. land** — altitude below sea level → ocean.
2. **Polar ice cap** — temperature below freezing year-round → ice.
3. **Alpine zone** — elevation above a latitude-dependent snow line → permanent snow.
4. **Vegetated continental interior** — moderate temperature, sufficient precipitation, low-to-mid altitude.

Finer partitions within land: rainforest (hot + very wet), savannah (hot + moderate), desert (hot + very dry), temperate forest (moderate + wet), grassland (moderate + moderate), taiga (cold + moderate), tundra (cold + dry).

**The rust signature.** Continental cells that are *both* far from any tectonic boundary (shield craton) *and* in low-vegetation biomes (desert, tundra, alpine) pick up a subtle reddish tint in their albedo — the exposed Archean-equivalent basement showing its oxidised-iron character through thin soil. This is the "metal-rich world" signature: present everywhere vegetation doesn't hide it, absent where life covers the rock.

**Boundary irregularity.** All biome boundaries are made fractal-irregular by an Fbm jitter on the decision variable (already implemented in `biomes.rs`'s `Fbm` rule). Without this, iso-contours read as smooth curves that look painted-on rather than natural.

**What it sets:** the `biome_map` cubemap (biome ID per texel) and the final albedo cubemap.

**Tidal asymmetry:** none.

**Parameters:** `biomes[]` with per-biome material and albedo (including rust tint curve for exposed basement), `rules[]` with the partitioning-rule hierarchy, snow-line latitude curve.

**Current basic-pass rules (2026-04-17):**
- `Latitude { min_abs: 0.72, soft_width: 0.08 }` → ice cap.
- `Fbm { frequency: 1.6, threshold: 0.12, octaves: 5 }` → continental crust.
- `Default` → ocean.

These give the correct silhouette (blue world with fractal continents and polar caps) but no climate-derived biome variation — no forests, no deserts, no rust. The full rule table is the roadmap.

Rule order matters: ice cap evaluated first (a continental-like fragment inside a cap should be ice, not tan); continent next; ocean last.

**Visual contribution alone:** the final visible colour of every surface texel. Biomes carries the entire "Earth-analog" read at impostor distance.

---

## Detail *(future stage — equivalent to Regolith on airless bodies)*

**Physical model:** the sub-cubemap-resolution texture layer. On Thalos this is soil, canopy, waves, dune fields, fracture patterns — statistically different per biome. Continental interiors have quasi-random soil/vegetation texture; coasts have wave-front parallelism; deserts have aeolian dune periodicity; ocean surfaces have wavelength-dependent specular scatter. None of it is visible at impostor distance.

**What it sets:** the detail-noise parameters consumed by the UDLOD renderer at close range. No contribution to the impostor cubemap.

The IQ analytical-derivative noise from Topography can be reused here at higher frequency for the shader's third layer, giving consistent morphology between baked and in-shader layers.

**Visual contribution at impostor distance:** none.

**Visual contribution at close range:** surface texture between explicit features. Determines whether a meadow reads as grassland vs. forest, whether an ocean reads as glassy calm vs. storm swell.

---

## Atmosphere / Clouds *(shader-side, not a bake stage)*

**Physical model:** Thalos's ~0.85 atm N₂/O₂ atmosphere scatters sunlight preferentially at short wavelengths (Rayleigh), producing a pale blue sky and a blue halo visible around the silhouette from space. Cloud cover is persistent — water vapor condenses into cumulus, stratus, and cirrus decks covering roughly 50% of the disk at any moment.

The atmosphere is thin compared to Earth's, so the rim halo is subtler and the limb brightening less pronounced. From orbit, Thalos reads as a slightly desaturated version of Earth.

**What it sets:** nothing at bake time. The atmosphere is rendered by a runtime shader pass (see `TerrestrialAtmosphere` in `thalos_atmosphere_gen` and the atmosphere module imported by `planet_impostor.wgsl`). Parameters (scale height, atmosphere top altitude, rim halo colour/intensity, limb shading) live in the body's `terrestrial_atmosphere` RON block, not in the generator pipeline.

**Visual contribution:** a blue halo outside the silhouette, limb darkening on the disk, and eventually per-fragment Rayleigh scattering and shader-synthesised cloud fields. The basic pass implements rim halo + limb shading only — cloud fields deferred.

**Tidal asymmetry:** none.

---

## How the stages combine visually

Useful frame: "what does Thalos lose if this stage is turned off?"

- **No Sphere**: nothing renders.
- **No Differentiate**: downstream stages have no materials to paint with.
- **No Plates**: no plate map → Tectonics, Topography, Hotspots, and Climate have no structure to key off. Continents degenerate to isotropic noise blobs (current basic-pass behaviour).
- **No Tectonics**: Plates exist but carry no boundary ages, orogen intensities, or ocean-floor ages. Topography loses its ability to distinguish young from old mountains; everything reads as uniform-amplitude noise over the plate map.
- **No Hotspots**: no intraplate volcanic chains or LIPs. Minor visual loss — hotspots are distinctive but small-area features.
- **No Topography**: continents are flat plateaus, oceans are flat basins, no mountains at all. The second-biggest visual loss after Biomes.
- **No Hydrology**: no rivers, no valleys, no deltas. Continents read as "extruded from noise" rather than "weathered by water." Subtle but cumulative.
- **No Climate**: Biomes falls back to latitude heuristics only — no rain shadows, no monsoons, no coastal climate contrast. Biome distribution reads as concentric latitude bands with no longitudinal variation.
- **No Biomes**: continents stay raw continental-crust tan, no ice, no vegetation green, no rust. Reads as "airless, dry Earth." Largest single visual payoff stage.
- **No Detail**: no effect at impostor distance.
- **No Atmosphere**: planet reads as "rocky airless Earth-twin" rather than "living world."

Ranking most to least visible at impostor distance:

> **Biomes ≈ Topography > Atmosphere > Hydrology > Tectonics > Plates > Hotspots >> Climate (indirect) >> Differentiate ≈ Sphere >> Detail**

Tune accordingly.

---

## What "looks right" means for Thalos

Validation criterion: "does this read as a habitable, Earth-lineage world, with the specific aesthetic of a planet past its geological prime?" Specific things to check:

- The majority of the disk is blue (ocean). Continental coverage is roughly a third.
- Continents have fractal-irregular coastlines, not smooth curves or perfect circles. Loose-cluster arrangement — 3–5 distinct landmasses — not one supercontinent, not a checkerboard.
- Polar regions are white (ice cap), with a soft transition rather than a hard circle.
- At least one continent carries a visible bright mountain belt — the active orogen. *(Deferred to Topography.)*
- Most mountain ranges look *worn*, not sharp — wide, rounded, broadly forested. Only the active orogens and Andean margins look sharp. *(Deferred to Topography + Biomes.)*
- Long dark river traces cross continental interiors, terminating in alluvial fans at the coasts. *(Deferred to Hydrology.)*
- Subtle rust tint visible in exposed continental interiors (semi-arid zones, tundra, alpine). *(Deferred to Biomes.)*
- A subtle blue halo is visible just outside the planet's silhouette, brightest on the sunward side.
- The limb picks up a cool chromatic shift rather than going flat-grey.
- Cloud cover breaks up otherwise-uniform regions. *(Deferred.)*

Of these, the basic pass (2026-04-17) satisfies: blue ocean majority, fractal coastlines, polar ice cap, blue halo + limb shading. The rest are the roadmap.

If all of the above eventually come true, Thalos is done.
