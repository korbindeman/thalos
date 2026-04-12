# Thalos Planet Generation — Logic Spec v0.1

## Purpose

Define the logic for a physically-parameterized procedural surface generator. The generator takes a small set of authored physical parameters describing a celestial body and produces surface data (elevation, material, surface maturity, and auxiliary per-sample fields later stages need) sampled on arbitrary points on a sphere.

This spec covers **v0.1: airless rocky bodies only** — moons and small planets with no meaningful atmosphere, no liquid surface, no biosphere. Targets: Luna, Mercury, Callisto, Rhea, Deimos.

The architecture must be designed so later revisions can add atmospheric terrestrial planets, ice worlds with cryovolcanism, and gas/ice giants **without restructuring the pipeline** — only by adding new derived properties and new stages. Stages not relevant to a given body must no-op based on derived properties, never on a body "type" enum.

Rendering, file formats, LOD, meshing, and output layout are **out of scope**. The generator produces logical surface data; wiring into the existing pipeline is the caller's responsibility.

---

## Design principles

1. **Author physical inputs, derive everything else.** The caller specifies a minimal set of physical parameters. The generator computes derived properties (gravity, escape velocity, equilibrium temperature, heat budget, etc.) once, up front. All later stages read derived properties, never raw inputs.
2. **No body-type branching.** No `if is_moon` or `if is_gas_giant` anywhere. Stages are gated by continuous derived properties. This is what lets the pipeline extend.
3. **Stages are ordered and each stage is a pure function** of (previous surface state, derived properties, sub-seed). Testable, reproducible, cacheable stage-by-stage.
4. **Hierarchical seeding.** One top-level seed derives per-stage sub-seeds by hashing with a stable stage name. Re-running one stage must not disturb any other stage's randomness.
5. **Resolution- and layout-agnostic.** Every stage is expressed as a function over points on the unit sphere. The caller chooses sampling. The generator assumes nothing about grids, faces, or texel counts.
6. **Physical units throughout.** Meters, kilograms, seconds, Kelvin. No normalized 0–1 elevation. Conversions happen at the edges.

---

## Authored physical parameters (v0.1)

Minimum set for a rocky airless body. Future revisions add parameters without removing any.

- **Radius** (m)
- **Mass** (kg)
- **Semi-major axis** relative to the system primary (m) — drives equilibrium temperature; already needed in v0.1 for future-compatibility of the derived-properties pass.
- **Bond albedo** (0–1)
- **Bulk composition fractions** — silicate, iron, ice, volatiles, hydrogen-helium. Sum to 1. v0.1 meaningfully consumes only silicate/iron/ice, but all fields exist.
- **Age** (Gyr) — drives crater accumulation and radiogenic heat decay.
- **Rotation period** (s)
- **Axial tilt** (rad)
- **Orbital eccentricity** — for future tidal heating; stored now so the data model is stable.
- **Parent body reference** (optional) — needed later for tidal heating and impact-flux modulation. v0.1 may ignore dynamics but must accept the field.
- **Impact flux multiplier** — scalar knob, 1.0 = lunar baseline.
- **Thermal history scalar** (0–1) — "how much internal activity did this body experience before freezing out." Drives mare flooding in v0.1; drives tectonics and volcanism later.
- **Seed** (u64)

**Optional authored overrides** (when present, bypass the matching derivation): surface gravity, equilibrium temperature, heat budget, tectonic style, magnetic field strength, hydrosphere fraction. These are authoritative when set. This is the art-direction escape hatch.

---

## Derived properties

Computed once from authored parameters before any stage runs. Every stage reads these, never raw inputs. Adding atmospheric planets means adding derivations here and stages that consume them, not rewriting existing logic.

Required for v0.1:

- **Surface gravity** — from mass and radius.
- **Escape velocity** — from mass and radius.
- **Equilibrium temperature** — from semi-major axis, bond albedo, system primary luminosity. Caller supplies luminosity; the generator does not assume Sol.
- **Oblateness** — from rotation rate, mass, radius. Negligible for slow rotators but the derivation must exist.
- **Radiogenic heat budget** — function of mass and age; simple exponential decay against a chondritic reference.
- **Tidal heat budget** — from parent mass, orbital distance, eccentricity. Zero when no parent specified. v0.1 does not consume this directly in any stage but it must be derived, because the mare-flooding decision reads **total** heat budget and the architecture must not distinguish heat sources.
- **Total internal heat budget** — radiogenic + tidal + a small residual accretion term decaying with age.
- **Simple-to-complex crater transition diameter** — inversely proportional to surface gravity, calibrated so lunar gravity produces ~15 km. This single derivation is what makes small moons visually distinct from large ones at no extra cost.
- **Atmospheric retention verdict** — per candidate volatile (H₂, He, H₂O, N₂, CO₂), compare thermal velocity at equilibrium temperature against escape velocity with a ~6× safety factor. In v0.1 this is expected to return "retains nothing" for all targets; the (future) atmosphere stage would no-op. This is the extension hook: when the same derivation later returns "retains N₂/CO₂" or "retains H/He," newly-added stages activate automatically.

All derived properties computed in one up-front pass and passed as an immutable context to every stage.

---

## Pipeline

Stages run in the listed order. Each stage reads current surface state and derived properties, and writes to surface state. A stage that does not apply must skip by examining derived properties — never by a body-type check, and never by being omitted from the pipeline configuration.

Each stage uses a sub-seed derived as `hash(top_seed, stage_name)`.

### Stage 0 — Reference shape

Initialize every sample as a sphere of the given radius, adjusted by oblateness. Elevation = 0 (measured from reference radius). Default material = dominant silicate type from composition (anorthosite-like when highland-dominated, basalt-like otherwise). Baseline maturity = "old, weathered."

### Stage 1 — Primordial crustal topography

Low-frequency coherent noise on the sphere representing ancient crustal thickness variation. Amplitude scales inversely with surface gravity (low-g bodies support taller primordial relief) and with the square root of radius. Multiple octaves, low total amplitude — typically a few kilometers peak-to-peak for a lunar-scale body. Intentionally gentle; impact stages will dominate the visible relief on airless bodies.

No material change. No maturity change.

### Stage 2 — Giant basins

Distribute a small number of very large impacts (typically 3–15 for a lunar-scale body, scaled by surface area and impact flux multiplier). Diameters sampled from the high tail of the size distribution. These are **not** ordinary craters and must be treated separately because they exhibit:

- Deep excavation, a significant fraction of crustal thickness
- Broad rim uplift extending well past the nominal rim
- Peak rings or multi-ring structures above a size threshold
- Long-range ejecta

Each basin writes elevation (excavation + rim + ejecta), a "basin floor" material flag, and a fresh-impact maturity reset within the excavation zone.

Runs before the main crater population so smaller craters correctly overprint basin rims and floors.

### Stage 3 — Mare flooding

Gated on total internal heat budget and thermal history scalar. If the body had sufficient internal activity, select basin floors whose lowest elevations fall below a threshold and flood them to a level determined by available melt volume. Within flooded regions:

- Flatten elevation to the flood level (minor noise for wrinkle ridges)
- Set material to mare basalt (or the compositionally appropriate dark volcanic material)
- Reset maturity to a mid value (mare are younger than highlands but have weathered since emplacement)

Bodies with low heat budget or low thermal history skip this stage cleanly. Small cold moons (Rhea, Callisto) produce no mare and look uniformly cratered. This must emerge from the derived properties, not a type check.

### Stage 4 — Main crater population

The workhorse stage.

**Size-frequency distribution.** Sample crater diameters from a power-law approximating observed lunar cumulative SFD (roughly N(>D) ∝ D⁻² in the main size range, with documented breaks at small and large ends). Total count proportional to `surface_area × age × impact_flux_multiplier` against a lunar-calibrated reference.

**Ordering.** Sort all sampled craters oldest-to-youngest. Required so younger craters correctly overprint older. Assign each crater an age within the body's history.

**Placement.** Distribute positions on the sphere. Uniform random is acceptable for v0.1; a slight Poisson-disk relaxation is an acceptable enhancement but not required.

**Stamping.** For each crater, in age order, modify the surface within its area of effect. Crater profile is a function of normalized radial distance from the center and crater diameter, with morphology depending on whether the diameter exceeds the derived simple-to-complex transition:

- **Simple (below transition):** bowl-shaped excavation, depth-to-diameter ~0.2, raised rim peaking at the rim radius, ejecta blanket decaying as inverse cube of radial distance past the rim, extending roughly 2–3 crater radii.
- **Complex (above transition):** shallower flat floor, terraced walls, central peak (small gaussian), broader ejecta. Depth-to-diameter decreases with size.
- **Peak-ring (well above transition):** floor with a concentric ring of peaks instead of a single central peak.

Profiles must be **additive modifications** to existing elevation, not replacements. Ejecta adds, excavation subtracts. This is what makes overprinting read correctly.

Every stamped crater must perturb its own shape slightly: mild ellipticity, rim irregularity, asymmetric ejecta. Without this perturbation craters look stamped. Non-optional.

Record each crater's age in an auxiliary field so the maturity stage can use it.

**Material.** Crater floors and ejecta blankets write a transient "fresh excavation" material ID. The maturity stage later blends this toward the surrounding baseline based on crater age.

### Stage 5 — Secondaries and rays

For each large crater from Stage 4 above a size threshold, optionally spawn a cluster of small secondary craters in a radial pattern and mark radial "ray" streaks extending outward. Secondaries stamp like normal small craters. Rays do **not** modify elevation — they only write a maturity reset along the ray pattern, so they appear as bright streaks after Stage 6. Produces Tycho-like features.

May be skipped entirely with no structural consequence; purely visual enhancement.

### Stage 6 — Regolith and space weathering (maturity pass)

A pass over the whole surface that finalizes the maturity field. On airless bodies, maturity is the dominant source of visible variation — more than composition. Fresh surfaces are bright, old surfaces dark.

Combines:

- Global baseline maturity from body age
- Per-sample noise for natural variation
- Per-crater resetting: fresh craters have low maturity on floor and ejecta, blending to baseline based on the crater age recorded in Stage 4
- Ray contributions from Stage 5
- **Slope-based reset:** steep slopes have reduced maturity because loose regolith slides off and exposes fresh material. Cheap effect, significantly improves how crater rims and scarps read. Slope from local elevation gradients.

Maturity is per-sample; does not modify elevation or material.

### Stage 7 — Detail hook

Does **not** bake high-frequency detail into the surface data. Instead, records parameters (noise seeds, amplitude envelopes keyed to material and slope) that a runtime or later bake step can use to synthesize sub-sample detail on demand. Baking fine detail into the base surface is wasteful and conflicts with virtualized-geometry LOD; delegating to a parameterized hook keeps output compact and lets the consumer choose its own resolution.

No surface mutation. Only records parameters alongside the surface data.

---

## Stage gating summary (v0.1)

| Stage | Gating | Airless rocky behavior |
|---|---|---|
| 0 Reference shape | always | runs |
| 1 Primordial topography | always | runs, low amplitude |
| 2 Giant basins | count > 0 from flux × area | runs |
| 3 Mare flooding | heat budget + thermal history above threshold | runs only for large, thermally active bodies |
| 4 Main craters | always, on solid surfaces | runs, dominates relief |
| 5 Secondaries and rays | optional enhancement | runs |
| 6 Maturity | always, on airless bodies | runs |
| 7 Detail hook | always | runs |

**Future stages** — gating only, do not implement in v0.1:

- Tectonics — gated on heat budget above a higher threshold + solid surface
- Volcanism beyond mare — gated on active heat budget
- Hydrosphere — gated on atmospheric retention of water + surface temperature in liquid range
- Erosion — gated on atmospheric pressure above a threshold
- Ice caps — gated on local surface temperature below volatile freezing point
- Biosphere tint — gated on temperature, pressure, and liquid water all in habitable range
- Atmosphere layer — gated on atmospheric retention verdict
- Gas giant banded atmosphere — gated on H/He retention + absence of solid surface

Adding these must be purely additive against the v0.1 pipeline.

---

## Physical calibration notes

Numbers the implementer needs and should not have to rediscover. Expose as named constants so they can be tuned.

- **Simple-to-complex transition** for the Moon is ~15–20 km. Scale as `15 km × (g_moon / g_body)`. Approximate; real relationship has a weak dependence on target material but gravity alone is sufficient.
- **Lunar cumulative crater density** at 1 km diameter on a ~4 Gyr surface is on the order of 10⁻² per km². Calibrate the impact flux constant so `age = 4.5 Gyr, flux_multiplier = 1.0, lunar gravity, lunar radius` produces lunar-like density. All other bodies then scale naturally.
- **Crater depth-to-diameter:** ~0.2 for simple, decreasing to ~0.05 for the largest complex. Smooth interpolation, not a step.
- **Rim height:** ~4% of diameter for fresh simple craters. Rim erosion is not required in v0.1 — maturity darkening provides sufficient visual aging.
- **Ejecta extent:** continuous blanket to ~1 crater radius past the rim, discontinuous ejecta and secondaries out to several radii. A single inverse-cube falloff from the rim is adequate.

---

## Determinism and testing requirements

- Same parameters + seed → bit-identical output across runs.
- Hierarchical and stable sub-seeding: changing Stage 4's sub-seed must not alter Stage 2's output.
- Each stage independently testable: given a synthetic input surface state, calling the stage produces deterministic output.
- Derived-properties computation is a pure function with unit tests against Luna, Mercury, Callisto, Rhea, and Deimos. Expected values for gravity, escape velocity, equilibrium temperature, and simple-to-complex transition diameter for those five bodies should be included as test fixtures.
- End-to-end test for Luna parameters must produce: nonzero giant basin count, nonzero mare area, crater density within an order of magnitude of lunar observation at 1 km scale, and a maturity field containing both fresh-crater lows and weathered highs.

---

## What v0.1 explicitly does not do

No atmosphere. No liquid surface, no oceans, no rivers. No erosion, no sediment transport. No tectonics or plate simulation. No volcanism beyond basin mare flooding. No biosphere or vegetation signals. No ring systems. No tidal deformation beyond oblateness. No color or albedo synthesis — maturity and material are the physical channels; color is the consumer's job. No LOD, no meshing, no normals, no tangent space, no texture atlasing.

All of the above have a place in the architecture (stages, derived properties, or consumers of the output) and none require changes to the v0.1 stages when they are added.

---

## Implementation order (suggested)

1. Derived-properties computation as a pure function. Unit-test against the five reference bodies.
2. Sphere sampling abstraction — the generator operates on "points on a unit sphere," not any particular grid. Whatever the caller passes in is what gets written.
3. Crater stamping function in isolation: one crater, correct simple profile, correct rim, correct ejecta falloff. Verify visually against a reference before moving on. This single function most determines whether the output looks like a moon.
4. Simple-to-complex morphology branch and central peaks.
5. Giant basin stage.
6. Full crater population with age ordering and size-frequency sampling.
7. Mare flooding.
8. Maturity pass with slope-based resetting.
9. Secondaries and rays.
10. Detail hook parameters.

Step 3 is where output starts looking like a real body. Steps 1–2 are infrastructure. Step 6 is where calibration against the Moon becomes possible and should not be skipped.
