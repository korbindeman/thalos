# Thalos — Terrain Generation Pipeline

A high-level spec for the procedural terrain pipeline for Thalos, the homeworld of the Pyros System. This document captures the planet's established character, the artistic targets the pipeline must serve, and the staged architecture for generating a cubemap-native planet surface.

---

## 1. Thalos in One Paragraph

Thalos is a shrunken Earth wrapped around an oversized iron heart. Half Earth's radius (3,186 km), roughly Earth's gravity (0.91 g), thin atmosphere (0.85 atm, breathable N₂/O₂), 65% ocean cover. The planet is **geologically aging**: plate tectonics has been winding down for hundreds of millions of years and is trending toward a stagnant lid regime. The metal-rich disk it formed from left siderophile elements abnormally abundant near the surface, which shows up everywhere as an **iron-stained palette** — rust-red sediments, dark iron-rich sand beaches, banded iron formations in old orogenic belts. The civilization has recently unified and is pushing outward toward space, which implicitly locates the planet in a **benign, cool-temperate climate window** analogous to Earth's Holocene.

The signature feeling, compressed to a single image: stand on a cliff in west Cornwall at dusk, but the cliffs are redder, the horizon is visibly closer, the sky is crisper, there are two moons rising, and the land under your feet has been quietly eroding for so long it has forgotten what it used to be.

---

## 2. Character Invariants

These are the constraints the pipeline must satisfy. If the output doesn't express these, the pipeline is wrong — no matter how technically correct each stage is in isolation.

- **Old continents.** Rounded uplands, wide floodplains, deep drainage networks, few sharp features. The Appalachians and the Scottish Highlands, not the Alps and the Andes.
- **Fossil tectonics.** Plate boundaries are visible but dead. Old arcs, old rifts, old orogenies, all heavily eroded. A small number of **active exceptions** (one or two live margins, a few hotspots) stand out against the stagnant majority.
- **Erosion dominates uplift.** On young worlds, tectonics builds faster than rivers can tear down. On Thalos, rivers have been winning for a long time. The hydrological stage should be aggressive.
- **Iron palette as consequence.** Iron-rich surface materials are not a shader tint. They come from where the sediment originated — eroded arc volcanics, weathered banded iron formations, mafic sand concentrations on beaches. Materials are consequences of geology and hydrology, not a separate layer on top.
- **Modest relief overall.** The small planet radius means feature scale matters. Avoid Earth-sized mountains crammed onto a half-sized sphere. Gentler landscapes fit both the "old and worn" brief and the physical scale.
- **Cool-temperate climate.** Small polar caps tight to the poles with irregular topography-respecting edges. Seasonal snow extends further but is seasonal. No continental-scale ice sheets. The planet reads as habitable and alive, not frozen.
- **Island character skews old.** More drowned ria coastlines, continental fragments, and old eroded arcs than young volcanic arcs. The rare young archipelagos are precious and narratively significant.
- **Sphere-native throughout.** Cubemap faces are a storage detail. Operations must be topologically aware that the sphere is continuous. Seamlessness is structural, not cosmetic.

---

## 3. Pipeline Overview

Six stages. Each produces a distinct kind of data. Each feeds the next.

```
Stage 1: Tectonic Skeleton
    ↓ province map (age, type, elevation bias)
Stage 2: Coarse Elevation
    ↓ mid-resolution heightfield + province mask
Stage 3: Hydrological Carving
    ↓ refined heightfield, drainage graph, sediment map, coastline
Stage 4: Fine-scale Roughness
    ↓ full-resolution heightfield
Stage 5: Surface Materials
    ↓ material ID cubemap + PBR parameters
Stage 6: Climate & Ice
    ↓ temperature field, precipitation field, ice/snow coverage
```

### Cross-cutting concerns

- **Resolution cascade.** Don't generate everything at final resolution and downsample. Generate at the lowest resolution each stage can tolerate, then upsample into the next. Stage 1 can be tiny (64×64 per face). Stage 3 is probably the most expensive. Stage 4 can be streamed per-chunk at playtime.
- **Per-stage seeds.** Each stage individually seedable so you can iterate on stage 4 without regenerating stage 1. Critical for tuning velocity.
- **Keep the graph data.** The drainage network from stage 3 isn't just useful for erosion — it's useful for gameplay (navigable rivers, settlements at confluences), ecology (moisture propagation), and materials (sediment provenance). Don't throw it away.
- **Sphere topology.** Every stage operates on the sphere natively. Flow routing, DLA walkers, erosion all need to handle the continuous topology. Fixing seams at the end is pain.

---

## 4. Stage-by-Stage

### Stage 1 — Tectonic Skeleton

**Purpose.** Establish the fossil record of the planet's first ~2.5 Gyr of tectonic activity. This is the skeleton everything else cascades from.

**Inputs.** Seed, target number of cratons, target active-margin count (1–2 globally), hotspot count (handful).

**Outputs.** Low-resolution cubemap of geological provinces. Each cell tagged with:
- Province type: `craton`, `suture`, `rift_scar`, `arc_remnant`, `active_margin`, `hotspot_track`, `oceanic_basin`
- Age (affects how much erosion applies downstream)
- Rough elevation bias (continental cratons positive, oceanic basins negative)

**Target result.** A globe-scale map that, if you color-coded it by province, would look like a geological province map of an old planet. A handful of ancient cratons forming continental cores. Curving belts of old sutures between them. Linear rift scars where continents pulled apart. One or two active margins marked for later special treatment. A few hotspot tracks, mostly in oceanic regions.

**Implementation notes.** Could be as simple as placing craton seeds on the sphere, Voronoi-partitioning, then running a plate-history simulation to generate sutures and rifts. Doesn't need to be physically rigorous — it needs to look like the output of a plate-tectonic history, not actually simulate one.

---

### Stage 2 — Coarse Elevation

**Purpose.** Turn the skeleton into a real heightfield at moderate resolution. This is where provinces become topography.

**Inputs.** Stage 1 province map, seed.

**Outputs.** Mid-resolution elevation cubemap plus a province mask (for later stages to know "this is an old arc" vs "this is a craton interior").

**Target result.** A globe whose coarse elevation respects the tectonic skeleton. Cratons are broad low plateaus. Sutures are worn mountain belts — think Appalachians, not Himalayas. Rift scars are basins or linear seas. Hotspot tracks are island chains or submerged seamount lines. Active margins are the one place sharp, young relief is allowed.

**Implementation notes.** DLA seeded along suture lines from stage 1 is a strong fit — it produces branching, dendritic uplift patterns that look like eroded orogenic belts. Discipline point: *don't over-amplify*. The temptation is to crank height variance for visual drama, but Thalos wants subdued relief with a few exceptional provinces. Active margins get their sharp relief here; everywhere else stays gentle.

---

### Stage 3 — Hydrological Carving

**Purpose.** Let water do to the landscape what it's been doing for a billion years. This is the character-defining stage.

**Inputs.** Stage 2 heightfield, province mask, seed, target sea level (tuned to hit 65% ocean cover).

**Outputs.**
- Refined elevation cubemap (carved by erosion, filled by deposition)
- Drainage network as a graph (nodes = basins, edges = rivers)
- Sediment thickness map (where deposition happened)
- Coastline (where refined elevation crosses sea level)

**Target result.** A globe with mature, deeply developed drainage. Dendritic networks on cratons. Trellis patterns on old fold belts. Braided rivers on floodplains. Deeply incised valleys cutting through uplands. Broad depositional lowlands filled with sediment. Coastlines are irregular and complex — ria systems where drowned river valleys meet the sea, not clean smooth boundaries. Fjord-analogues in formerly glaciated high latitudes.

**Implementation notes.** Run flow-accumulation on the sphere (there's established literature on graph-based flow routing on spheres — worth reading). Both carving *and* deposition must be first-class outputs; the sediment thickness map is critical for stage 5's material assignment. Run erosion hard — Thalos has had hundreds of millions of years of it. Pick sea level last, after the landscape has settled, to hit the 65% ocean target.

---

### Stage 4 — Fine-scale Roughness

**Purpose.** Add the high-frequency detail that makes local areas feel authentic without disturbing the large-scale story.

**Inputs.** Stage 3 outputs, province mask, seed.

**Outputs.** Full-resolution heightfield.

**Target result.** Local texture appropriate to each province: ridge-and-valley texture on old fold belts, fracture patterns on exposed craton bedrock, volcanic cones in active provinces, rare impact structures preserved in the most stable cratons (active erosion has obliterated most elsewhere), dune fields in arid basins, subtle terrace structures along major rivers.

**Implementation notes.** This stage is *additive modulation* on stage 3, not a replacement. You're adding detail while preserving the large-scale hydrology and topography. DLA works here too but for different features (local drainage texture, fracture networks). Multi-octave noise with domain warping fills in where structured processes don't apply. Biome-aware: the roughness applied to a desert basin is different from that applied to a temperate upland. This stage is the best candidate for deferred/streamed generation at playtime.

---

### Stage 5 — Surface Materials

**Purpose.** Make Thalos *look* like Thalos. This is where the iron palette enters the world.

**Inputs.** All previous stages — heightfield, province mask, drainage graph, sediment map.

**Outputs.** Material ID cubemap plus PBR parameters (albedo, roughness, etc.) per material.

**Target result.** Each surface cell has a material that is a *consequence* of its history:
- Weathered granite-analogue on craton interiors
- Banded iron formation exposures on old arc remnants
- Iron-rich rust-red sediment on floodplains and deltas downstream of old arcs
- Dark iron-sand beaches where mafic minerals concentrate
- Peat-analogue and wet soil in poorly drained lowlands
- Bare rock on recently scoured uplands (glacial or high-energy erosion)
- Fresh volcanic rock on the rare active provinces

**Implementation notes.** The rule is that material assignment should be *derivable* from upstream data. An iron-rich floodplain is iron-rich because stage 3 traced its sediment back to an arc remnant from stage 1. This is more work than a biome-based material assignment, but it produces a coherent palette that tells the planet's story. This stage also hooks into biome assignment later (temperature + precipitation + altitude → biome), but the material layer is more fundamental and should exist even for lifeless bodies.

---

### Stage 6 — Climate & Ice

**Purpose.** Generate the climate fields that drive ice, snow, and (later) biome distribution.

**Inputs.** Stage 3 heightfield and coastline, axial tilt (~23°), orbital parameters, optional season parameter.

**Outputs.**
- Temperature field (annual mean, annual range)
- Precipitation field
- Permanent ice coverage (where temperature is below freezing year-round with sufficient accumulation)
- Seasonal snow coverage (optional, if rendering a specific season)

**Target result.** A cool-temperate planet. Small permanent ice caps tight to the poles, occupying maybe 5–8% of the visible disc from orbit. Cap edges are *irregular* — following coastlines, dipping along mountain ranges, retreating where warm ocean currents reach. Asymmetry between hemispheres if the geography is asymmetric (continental pole accumulates more than oceanic pole). Mountain glaciers at lower latitudes on high terrain. If rendering winter, seasonal snow extends into mid-latitudes in continental interiors; if rendering summer, most of that melts off.

**Implementation notes.** Temperature is primarily latitude-driven, with elevation (−6.5°C per km, roughly) and ocean proximity as modifiers. Precipitation requires a more careful model — rain-shadow effects, continental dryness, orographic lift — but a simple version works for a first pass. Ice is then a threshold: cold enough + enough precipitation = permanent ice; cold winters but warmer summers = seasonal snow. This stage also produces the fields needed for biome assignment downstream, so it's worth getting right.

---

## 5. What to Build First

If starting from scratch, the build order should mirror the data flow, but the *earliest* payoff comes from stages that produce the most visible character:

1. **Stage 1** is foundational and cheap. Do it first.
2. **Stage 2** gives you a recognizable planet. Do it second.
3. **Stage 3** is where Thalos starts to look like Thalos rather than a generic planet. Third.
4. **Stage 6** (climate/ice) should come before stage 5 if you want a quickly-convincing planet render, because ice caps are a high-signal visual feature and a broken cap is very noticeable. Also cheap to prototype.
5. **Stage 5** (materials) is where the iron palette shows up. Big visual payoff but depends on good drainage data from stage 3.
6. **Stage 4** (fine detail) last. It's the most expensive to iterate on and the least impactful at the orbital view. Also the best candidate for deferred generation.

---

## 6. Open Questions for Later

- **Axial tilt variation and seasonality.** Worth implementing seasonal variation, or lock to a "nominal" season for the orbital view?
- **Tidal influence from Mira.** Mira is close enough to raise significant tides. Does this show up in the pipeline (coastal geometry, tidal flats) or is it purely a shader/gameplay concern?
- **Active-margin placement.** Is the placement of the one or two live margins a worldbuilding decision (hand-placed) or random per-seed? The civilization's geography probably reflects where the live geology has been historically.
- **Biome layer.** Not part of this pipeline but a natural downstream consumer of stages 3, 5, and 6.
- **Streaming and LOD.** How much of stage 4 can be regenerated on demand per-chunk, and what do we bake vs. compute live?
