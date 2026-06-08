# Planet Generation — The Authoring Method

**Status:** Draft. Defines the generation method *from the author's chair* —
how a person shapes a planet, and therefore what the system underneath must
provide. Companions:

- [terrain-generation-cascade.md](terrain-generation-cascade.md) — the technical
  layer / derivation model.
- [planet-generation-pipeline-spec.md](planet-generation-pipeline-spec.md) — the
  field-DAG substrate.
- [planet-generation-pipeline-migration.md](planet-generation-pipeline-migration.md)
  — the brownfield sequencing.

## 1. The principle

**Author the causes; derive the consequences.** A body's description is small:
physical parameters + a structural seed + a few generator knobs (+ optional
painted intent and placed features). Everything else — continents, sea level,
climate, biomes, rivers, materials, beaches, cliffs — is *derived* by the
generation cascade and is reproducible from that description.

The generator reads consequences off logical primitives (flat coast → beach,
steep coast → cliff), so the author shapes *intent*, never final geometry.

Two rules:

- **Intent fields bias, never replace.** Painted elevation/climate/structure
  fields bias the generator; the generator is the only thing that produces final
  terrain, materials, and scatter.
- **Everything is reproducible from the config.** The saved config is the source
  of truth; the game bakes and loads from it.

## 2. Two levels of shaping

| Level | What you do | Examples |
|---|---|---|
| **Definition** | Pick a preset, set physical params, roll a structural seed, tune generator sliders | "Oceanic terrestrial, 1.2 g, 70% ocean, young tectonics" |
| **Authoring** | Paint intent fields, place discrete features | "Coast here, a volcano there, drier in this basin" |

**Definition is parametric** (sliders and reroll); **authoring is spatial**
(brushes and placements). Definition sets the world; authoring sculpts it.

Changing definition parameters regenerates the cascade and may shift the terrain
your authored features sit on — the editor warns when this would strand a
placement. Save a snapshot first if you want to return.

## 3. Definition

### Preset & physical parameters

Pick an archetype preset (oceanic terrestrial, airless impact moon, cold desert,
ice world). This populates sensible defaults for:

- radius, mass / gravity, bulk composition, age
- insolation (distance + star luminosity), rotation period, obliquity
- volatile budget (water + atmosphere)

Tune any parameter from there. Derived readouts (surface gravity, equilibrium
temperature, day length, escape velocity) update live so you see consequences.

### Structural seed

One master seed drives the cascade. **Reroll** to get a different world —
continents rearrange, mountains shift, climate zones move. The seed is saved in
the config so the same seed always produces the same planet.

### Generator tuning

Per-layer sliders shape character without rerolling the seed:

- **Tectonics** — plate count, continental fraction, clustering
- **Topography** — relief scale, sea-level offset
- **Climate** — circulation strength, rain-shadow strength, seasonality
- **Erosion / hydrology** — erosion strength, river density
- **Biomes** — treeline / aridity thresholds, blend sharpness
- **Terrain character** — mountain ruggedness, hill frequency
- **Features / scatter** — crater density, scatter density per type

These are saved in the config alongside the seed.

## 4. Authoring

### Intent fields (paint)

Paint on the sphere to bias the generator:

- **Elevation** — raise / lower / flatten / smooth
- **Coastline** — push land or sea
- **Climate** — warmer / colder, wetter / drier
- **Structure** — mark a mountain belt, fault, or volcanic province

Brush params: size, falloff, strength, add / subtract / smooth. Each stroke
carries an **authority** — does it *bias* the procedural result or *override* it.

Painted overlays are saved as a replayable op-log composed onto the procedural
output. Rerolling the seed regenerates the procedural layer without disturbing
the overlay; painting modifies the overlay without disturbing the procedural
layer.

### Features (place)

Place discrete features on the sphere: volcanoes, craters, canyons, named peaks,
basins. Move, scale, rotate, and parameter-edit them.

Procedural features (craters from the age-driven density) can be **promoted** to
explicit — the system captures their current parameters, and they survive seed
rerolls. **Demote** returns them to procedural control.

Scatter (trees, grass, boulders) is authored as a density field per type. The
renderer handles the LOD continuum from far-field material to near instances; the
author sets what and how dense.

### Override management

Toggle between seeing the painted overlay, the derived result, and the
composite. Clear an override region, reset to derived.

## 5. Snapshots

Save the current config as a named snapshot. Load a snapshot to return to it.
Snapshots are flat — a list of named saves. If you need branching or diffing, use
git on the config files.

The game loads the active config directly; snapshots are an editor convenience.

## 6. Inspect & preview

- **Field inspector** — view any intermediate or output field as a sphere /
  equirect overlay (elevation, slope, temperature, moisture, biome, tectonic
  plates); toggle, opacity.
- **Preview** — orbital (impostor) ↔ ground (heightfield), real-time so edits
  show live.
- **Bake** — bake the current config to the game artifact (`just bake <body>`).

## 7. What's authored vs. derived

- **Authored** (the saved config): preset, physical params, structural seed,
  generator sliders, painted overlays, explicit features.
- **Derived** (never authored, always reproducible): bedrock & final elevation,
  sea level, slope / curvature, coastlines, temperature, moisture, rivers /
  lakes, biomes, materials, scatter distributions, beaches / cliffs / scree.

Derivation is the default; you reach into authoring only where the derived
result isn't what you want. **Authority** (bias vs. override) governs how hard
your edit overrides the derivation.

## 8. Relationship to other docs

- This doc = the **method / workflow** (author's chair).
- [terrain-generation-cascade.md](terrain-generation-cascade.md) = the
  **technical layer / derivation model**.
- [planet-generation-pipeline-spec.md](planet-generation-pipeline-spec.md) = the
  **field-DAG substrate**;
  [planet-generation-pipeline-migration.md](planet-generation-pipeline-migration.md)
  = the **build sequencing**.

---

*Doc owner: Korbin. The authoring method; see the cascade doc for the derivation
rules and the spec for the substrate.*
