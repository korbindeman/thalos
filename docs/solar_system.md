# Solar System Configuration

## Scale Philosophy

The system uses a hybrid scaling approach optimized for the balance between physical plausibility and gameplay:

| Category | Scale | Rationale |
|---|---|---|
| Star | 1:1 | Anchors all orbital dynamics and habitable zone calculations |
| Orbital distances | 1:1 | Determined by stellar luminosity, not body size |
| Gas/ice giants | 1:1 | Preserves visual spectacle, moon systems, and orbital dynamics |
| Thalos (homeworld) | 1:2 | Sweet spot for reaching orbit while maintaining ~0.9g |
| Ashara (Venus analogue) | 2:3 | Harder to escape than Thalos |
| Vaelen (Mars analogue) | 1:3 | Easy to land on and launch from |
| Other rocky bodies/moons | 1:3 to 1:2 | Tuned per body for gameplay role |

## Formation: Metal-Rich Protoplanetary Disk

All rocky bodies share an elevated iron content, explained by a single formation scenario rather than per-body hand-waving. The star formed in a region enriched by nearby Type Ia supernovae, producing a protoplanetary disk with a high iron-to-silicate ratio. This is astrophysically observed: high-metallicity stars exist and their planets are expected to be metal-enriched.

The disk produces a natural density gradient. Inner bodies accreted from hotter regions where metals condensed preferentially, so they are denser. Outer rocky bodies incorporated more silicates and volatiles, so they are less dense. This is the same pattern as our solar system (Mercury denser than Mars), just shifted upward.

---

## Star — Pyros

| Property | Value |
|---|---|
| Spectral type | G2V |
| Mass | 1.0 M☉ |
| Radius | 696,000 km |
| Luminosity | 1.0 L☉ |
| Temperature | 5,778 K |
| Age | ~4.6 Gyr |

Solar twin. All habitable zone distances, orbital periods, and insolation values follow directly.

---

## I. Ignis (Hot Rocky World)

| Property | Value |
|---|---|
| Scale | 1:3 |
| Semi-major axis | 0.30 AU |
| Orbital period | 60 days |
| Eccentricity | ~0.02 |
| Radius | 813 km |
| Density | 7,000 kg/m³ |
| Mass | 0.0026 M🜨 |
| Surface gravity | 0.16 g |
| Escape velocity | 1.6 km/s |
| Atmosphere | None |
| Surface | Molten dayside, cratered nightside |
| Rotation | Tidally locked |

The innermost body and the densest (most iron-rich, formed where only metals could condense). Dayside exceeds 700 K. Potential mining target for refractory metals, but thermal management dominates any surface operations. Low gravity makes landing and departure trivial.

---

## II. Ashara (Venus Analogue)

| Property | Value |
|---|---|
| Scale | 2:3 |
| Semi-major axis | 0.65 AU |
| Orbital period | 191 days |
| Eccentricity | ~0.01 |
| Radius | 4,035 km |
| Density | 9,500 kg/m³ |
| Mass | 0.44 M🜨 |
| Surface gravity | 1.09 g |
| Escape velocity | 9.3 km/s |
| Atmosphere | ~50 atm, CO₂-dominant |
| Surface temp | ~700 K |
| Rotation | Slow retrograde (~180 days) |

The largest and most massive rocky body in the system. Its 2/3 scale and high density (justified by proximity to the star in the metal-rich disk) produce surface gravity exceeding Thalos's. Combined with a crushing 50-atmosphere greenhouse, this is the hardest body in the system to escape from.

### Gameplay reference

| Metric | Value | vs Homeworld |
|---|---|---|
| Orbital velocity (150 km) | ~7.0 km/s | 135% |
| Delta-v to orbit (est.) | ~9.5 km/s | 146% |
| Escape velocity | 9.3 km/s | 124% |

Surface operations are impractical (700 K, 50 atm). Colonization means aerostat habitats at 50-60 km altitude, where temperature and pressure approach habitable ranges. Launching from an aerostat platform avoids the deepest atmosphere but still requires fighting significant drag and gravity. Arrival is easier: the thick atmosphere enables aggressive aerobraking.

---

## III. Thalos (Homeworld)

| Property | Value |
|---|---|
| Scale | 1:2 |
| Semi-major axis | 1.00 AU |
| Orbital period | 365.25 days |
| Eccentricity | ~0.017 |
| Radius | 3,186 km |
| Density | 10,000 kg/m³ |
| Mass | 0.23 M🜨 |
| Surface gravity | 0.91 g (8.9 m/s²) |
| Escape velocity | 7.5 km/s |
| Atmosphere | ~0.85 atm, N₂/O₂ |
| Hydrosphere | ~65% ocean cover |
| Axial tilt | ~23° |
| Magnetic field | Strong |

### Gameplay reference

| Metric | Value | vs Real Earth |
|---|---|---|
| Orbital velocity (150 km) | 5.2 km/s | 67% |
| Delta-v to orbit (est.) | ~6.5 km/s | 69% |
| Escape velocity | 7.5 km/s | 67% |
| Surface gravity | 8.9 m/s² | 91% |

### Physical character

Thalos is essentially a massive exposed iron core with a thin silicate mantle and crust, ~70% iron core by volume. The result of forming in the most metal-rich region of the disk at this orbital distance. The enormous liquid iron outer core generates a strong magnetic field that protects the atmosphere from stellar wind stripping.

**Atmosphere** is ~0.85 atm, slightly thinner than Earth's. Breathable N₂/O₂ mix. Equivalent to roughly 1,500 m altitude on Earth: unnoticeable to most people. Weather systems and hydrological cycling function normally but storms carry slightly less energy.

**Geology** is shaped by the planet's thermal history. The thin silicate mantle contains concentrated lithophile elements (uranium, thorium, potassium), and the oversized core provides strong basal heating as the inner core crystallizes. Surface water weakens the lithosphere, enabling fracture into mobile plates. Together, these factors drove vigorous plate tectonics for the first ~2.5 billion years, building continents, establishing the carbon cycle, and stabilizing climate for complex life.

Plate tectonics has since entered a declining phase. The planet's small thermal reservoir (0.23 Earth masses) and high surface-to-volume ratio mean heat escapes faster than on Earth. A few active plate boundaries remain, but most of the lithosphere has thickened into stagnant regions. Volcanism is waning. The carbon cycle still functions (ocean buffering, remaining volcanic outgassing, weathering), but the planet is geologically aging faster than Earth. On a timescale of hundreds of millions of years, it will trend toward a stagnant lid regime.

**Surface resources**: the metal-rich composition means iron, nickel, and siderophile elements are abnormally abundant near the surface. This shaped the civilization's technological trajectory.

### Mira (Moon)

| Property | Value |
|---|---|
| Scale | 1:2 |
| Orbital radius | 192,000 km |
| Orbital period | ~16 days |
| Radius | 869 km |
| Density | 5,000 kg/m³ |
| Mass | 0.0023 M🜨 |
| Surface gravity | 0.12 g |
| Escape velocity | 1.5 km/s |
| Apparent angular diameter | ~0.52° |

Formed from a giant impact that ejected silicate mantle material, so the Moon is less dense than the planet (silicate-dominated rather than iron-dominated). Tidally locked, airless, heavily cratered. Angular size is nearly identical to our Moon as seen from Earth, preserving familiar eclipse geometry. Provides axial tilt stabilization.

### Selva (Minor Moon)

| Property | Value |
|---|---|
| Orbital radius | 269,000 km |
| Orbital period | ~13 days |
| Radius | 190 km |
| Mass | 3.6 × 10¹⁹ kg |
| Albedo | 0.25 |
| Inclination | 3.0° |

Small captured or co-accreted body in a circular orbit beyond Mira. Silicate-dominated composition similar to Mira. A potential early-game waypoint between Thalos and Auron.

---

## IV. Auron (Sub-Saturn)

| Property | Value |
|---|---|
| Scale | 1:1 |
| Semi-major axis | 1.31 AU |
| Orbital period | 547 days |
| Eccentricity | ~0.03 |
| Mass | 40 M🜨 (~0.13 M♃) |
| Radius | 44,600 km (~7.0 R🜨) |
| Ring system | Moderate, icy |
| Atmosphere | H₂/He, banded gold/cream |

### Orbital resonance

Locked in a 3:2 mean-motion resonance with Thalos. Conjunction occurs every ~3 years at a minimum separation of 0.31 AU (~46.4 million km). The resonance prevents chaotic perturbation accumulation and ensures long-term stability of both orbits.

### Appearance from Thalos

| Feature | Angular size | Comparison |
|---|---|---|
| Planet disk at conjunction | 6.6 arcmin (0.11°) | 21% of the Moon |
| Ring system at conjunction | ~13 arcmin (0.22°) | 42% of the Moon |
| Planet at opposition | ~0.9 arcmin | Bright point |

At conjunction: a striking naked-eye disk with visible gold coloration. The ring system, at favorable inclination, extends the apparent size to nearly half the Moon's diameter. The brightest object in the night sky by a wide margin. At opposition: shrinks to a brilliant point, still brighter than any star.

### IV-a. Pelagos (Ocean Moon)

| Property | Value |
|---|---|
| Scale | 1:2 |
| Orbital radius | 500,000 km |
| Orbital period | ~6.2 days |
| Radius | 2,350 km |
| Density | 8,000 kg/m³ |
| Mass | 0.073 M🜨 |
| Surface gravity | 0.54 g |
| Escape velocity | 5.0 km/s |
| Atmosphere | N₂-dominant, CO₂/H₂O, ~0.4 atm |
| Hydrosphere | ~85% ocean cover |
| Surface temp | 270-290 K (-3 to 17 °C) |

The most biologically significant body in the system. Dual energy input: stellar insolation from the habitable-zone-adjacent orbit and tidal heating from Auron. Tidal forces maintain active volcanism and hydrothermal circulation, cycling nutrients through the ocean.

At 0.54 g and 0.4 atm, this is a world you can stand on. The atmosphere is too thin and CO₂-rich for unassisted breathing, but pressure suits can be minimal: oxygen supply and thermal regulation, no full pressurization needed. Waves are long and slow. Rain falls gently. Volcanic island arcs break the ocean surface but there are no continents.

**Sky from the surface**: Auron dominates, subtending ~10° (20x the Moon from Earth). Regular stellar eclipses behind the gas giant. Thalos is the brightest point in the night sky.

#### Biology

Photosynthetic primary producers in the upper ocean drive a complex food web. Life has reached Cambrian-equivalent complexity over ~3 billion years of stable conditions:

- Diverse multicellular marine organisms with mineralized body plans (shells, exoskeletons)
- Segmented swimmers, armored grazers, tentacled predators
- Filter-feeders on volcanic island shelves
- Burrowing organisms in shallow sediments
- No land life (minimal emergent landmass, no soil formation)

The ecosystem is alien but recognizable. Convergent evolution produces body plans that echo Earth's Cambrian fauna. Biochemistry may differ (different amino acids, possibly different chirality), but ecological niches are universal.

#### Gameplay reference

| Metric | Value |
|---|---|
| Orbital velocity (100 km) | 3.6 km/s |
| Delta-v to orbit (est.) | ~4.2 km/s |
| Escape velocity | 5.0 km/s |

### IV-b, IV-c, IV-d. Minor Moons — Carpo, Theron, Maren

| Moon | Radius | Mass | Orbital radius | Eccentricity | Inclination | Albedo |
|---|---|---|---|---|---|---|
| Carpo (IV-b) | 180 km | 1.5 × 10¹⁹ kg | 799,000 km | 0.04 | 8.0° | 0.15 |
| Maren (IV-c) | 75 km | 9.5 × 10¹⁷ kg | 1,198,000 km | 0.12 | 47.0° | 0.08 |
| Theron (IV-d) | 120 km | 3.9 × 10¹⁸ kg | 2,094,000 km | 0.18 | 154.0° (retrograde) | 0.05 |

Three captured bodies in irregular orbits. Carpo is the innermost and most accessible; Maren orbits at moderate inclination; Theron is retrograde with a very dark carbonaceous surface. Mining and refueling waypoints.

---

## V. Vaelen (Mars Analogue)

| Property | Value |
|---|---|
| Scale | 1:3 |
| Semi-major axis | 1.70 AU |
| Orbital period | 809 days (2.22 years) |
| Eccentricity | ~0.20 (and rising) |
| Radius | 1,130 km |
| Density | 6,500 kg/m³ |
| Mass | 0.0066 M🜨 |
| Surface gravity | 0.21 g |
| Escape velocity | 2.2 km/s |
| Atmosphere | Thin, ~6 mbar CO₂-dominant |
| Surface temp | 160-245 K (varies with season and orbital position) |

### Gameplay reference

| Metric | Value |
|---|---|
| Orbital velocity (50 km) | 1.5 km/s |
| Delta-v to orbit (est.) | ~1.7 km/s |
| Escape velocity | 2.2 km/s |

The easiest major body to land on and launch from. Single-stage landers with generous margins. Transfer windows from Thalos vary dramatically due to high eccentricity: favorable windows are cheap, unfavorable ones significantly more expensive.

### Orbital instability

Secularly unstable. Perturbations from Auron and Teros slowly pump eccentricity. No damping mechanism exists (no massive moon, no thick atmosphere, no tidal interaction with a nearby giant).

**Geological history**: formed with low eccentricity (~0.03) and likely had surface liquid water for its first ~1.5 billion years. Ancient riverbeds, evaporite deposits, and sedimentary layering record this wetter past. As eccentricity climbed, surface water was lost to sublimation at perihelion and permanent freezing at aphelion. Subsurface water ice persists at higher latitudes and in sheltered craters.

**Current conditions**: eccentricity of ~0.20 produces perihelion insolation ~44% higher than aphelion. Extreme seasonal temperature swings. CO₂ cycles between polar caps and atmosphere. Dust storms triggered by perihelion heating.

**Prognosis**: within ~1 billion years, eccentricity climbs high enough that perihelion approaches enter Auron's sphere of influence. Most probable outcome: gravitational scattering and ejection from the system as a rogue planet.

### Colonization value

The doom timescale is irrelevant to civilization. Primary value:

- **Strategic position** as waypoint between inner system and outer gas giants
- **Perihelion water access**: seasonal melt of equatorial subsurface ice
- **Mineral wealth**: concentrated ore deposits from ancient hydrological cycling
- **Trivial gravity well**: cheap to operate from, ideal staging point
- **Cultural weight**: a world visibly dying on geological timescales, with ancient riverbeds and a doomed orbit

### Kael and Xxirt (Moons)

| Moon | Radius | Mass | Orbital radius | Eccentricity | Inclination |
|---|---|---|---|---|---|
| Kael | 14 km | 7.3 × 10¹³ kg | 94,000 km | 0.015 | 1.1° |
| Xxirt | 8 km | 1.7 × 10¹³ kg | 150,000 km | 0.002 | 1.8° |

Two tiny moons in near-circular, low-inclination orbits. Likely captured asteroids or co-accreted debris. Too small for meaningful gravity; useful as navigation landmarks or micro-refueling stops.

---

## VI. Asteroid Belt

| Property | Value |
|---|---|
| Inner edge | ~2.5 AU |
| Outer edge | ~3.8 AU |
| Total mass | ~0.001 M🜨 |

Sculpted by resonances with Teros. Kirkwood-style gaps present. Mix of metallic (inner belt), silicate, and carbonaceous (outer belt) bodies. Largest body: ~400-600 km diameter, possibly differentiated. The metal-rich disk origin means the belt's metallic fraction is higher than our asteroid belt.

---

## VII. Teros (Jupiter Analogue)

| Property | Value |
|---|---|
| Scale | 1:1 |
| Semi-major axis | 5.0 AU |
| Orbital period | 11.2 years |
| Eccentricity | ~0.04 |
| Mass | 1.2 M♃ (380 M🜨) |
| Radius | 74,000 km |
| Ring system | Faint, dusty |
| Atmosphere | H₂/He, ammonia cloud bands, ochre/white |
| Notable | Persistent storm systems, intense radiation belts |

Primary debris shield for inner system. Extensive moon system. Intense radiation environment complicates close operations.

### VII-a. Glacis (Europa Analogue)

| Property | Value |
|---|---|
| Scale | 1:2 |
| Orbital radius | 670,000 km |
| Orbital period | ~3.5 days |
| Radius | 780 km |
| Density | 4,500 kg/m³ |
| Mass | 0.0015 M🜨 |
| Surface gravity | 0.10 g |
| Escape velocity | 1.2 km/s |
| Surface | Water ice shell, ~15-25 km thick |
| Subsurface ocean | Global, ~80 km deep, saline |
| Surface temp | ~100 K |

Tidal heating maintains a liquid ocean beneath the ice shell. Hydrothermal vents on the ocean floor provide energy and mineral cycling. Cryovolcanic plumes occasionally breach the surface, depositing organic-rich material on the ice.

**Biology**: chemosynthetic microbial life. Biofilms around hydrothermal vents, free-swimming single-celled organisms in the water column. The energy budget of chemosynthesis imposes a complexity ceiling: diverse microbial ecosystem, but no multicellularity.

Exploration requires drilling through the ice shell or sampling cryovolcanic deposits. Teros's radiation belts are a significant hazard.

### VII-b through VII-d. Major Moons — Pyrith, Lithos, Calyx

| Moon | Radius | Mass | Orbital radius | Albedo | Character |
|---|---|---|---|---|---|
| Pyrith (VII-b) | 910 km | 1.14 × 10²² kg | 422,000 km | 0.40 | Io analogue, volcanic |
| Lithos (VII-c) | 1,315 km | 1.79 × 10²² kg | 1,070,000 km | 0.15 | Large icy, ancient cratered |
| Calyx (VII-d) | 1,205 km | 1.37 × 10²² kg | 1,880,000 km | 0.10 | Mixed ice-rock, dark |

- **Pyrith**: Io analogue. Volcanic, tidally tortured, sulfur-rich surface. Deep in the radiation belts.
- **Lithos**: Large icy moon, ancient cratered surface. Possible frozen fossil ocean. Geological record of early system bombardment.
- **Calyx**: Mixed ice-rock. Thin atmosphere possible. Most habitable of the outer moons due to lower radiation at its orbital distance.

---

## VIII. Seraph (Second Gas Giant)

| Property | Value |
|---|---|
| Scale | 1:1 |
| Semi-major axis | 8.5 AU |
| Orbital period | 24.8 years |
| Eccentricity | ~0.05 |
| Mass | 0.5 M♃ (159 M🜨) |
| Radius | 60,000 km |
| Ring system | Prominent, wide, icy |
| Atmosphere | H₂/He, muted blue-gold banding |

The visual showpiece of the outer system. Dense, bright icy rings with visible gaps and divisions, more spectacular than Auron's. Several mid-sized moons, including a Titan-class body with a thick nitrogen/methane atmosphere and hydrocarbon lakes.

### VIII-a. Tethyn (Titan-Class Moon)

| Property | Value |
|---|---|
| Orbital radius | 1,222,000 km |
| Orbital period | ~16 days |
| Radius | 1,288 km |
| Mass | 1.79 × 10²² kg |
| Eccentricity | 0.029 |
| Inclination | 0.3° |
| Albedo | 0.20 |

The largest of Seraph's moons. Thick nitrogen/methane atmosphere with hydrocarbon lakes on the surface. The only moon in the outer system with a substantial atmosphere, making it a high-value exploration target despite the distance.

---

## IX. Nereus (Inner Ice Giant)

| Property | Value |
|---|---|
| Scale | 1:1 |
| Semi-major axis | 18 AU |
| Orbital period | 76 years |
| Eccentricity | ~0.03 |
| Mass | 17 M🜨 (~1.0 M♆) |
| Radius | 25,000 km |
| Atmosphere | H₂/He/CH₄, deep blue-green |

Methane absorption gives it a striking color. A few small icy moons. Missions here are serious commitments.

---

## X. Pallos (Outer Ice Giant)

| Property | Value |
|---|---|
| Scale | 1:1 |
| Semi-major axis | 28 AU |
| Orbital period | 148 years |
| Eccentricity | ~0.04 |
| Mass | 12 M🜨 |
| Radius | 22,000 km |
| Atmosphere | H₂/He/CH₄, pale cyan |
| Axial tilt | ~82° |

Highly tilted, rolling along its orbit. Decades-long polar seasons. Irregular captured moons in chaotic orbits. The outermost major body.

---

## Small Bodies and Distant Objects

The major planets define the system's architecture. The small bodies define its texture: the debris, the survivors, the strays. Most individual asteroids and minor comets will be catalogued separately. What follows are the structurally and narratively significant populations and objects.

### System inclination

The ten major planets orbit within 2-3° of the invariable plane, a direct consequence of forming from a single disk. Departures from this flatness indicate bodies that were scattered, captured, or perturbed after formation. Inclination is a biographical marker: the more inclined an orbit, the more violent its history.

### Trojan populations

#### VII L4/L5 Trojans (Teros)

Thousands of small bodies trapped at Teros's leading and trailing Lagrange points, co-orbital at 5.0 AU. Compositionally more primitive than the main belt: volatile-rich, carbonaceous, dark surfaces. A few are large enough (100-200 km) to be individually significant. These are accessible from Teros without the delta-v cost of returning to the main belt, making them a convenient resource population for outer-system operations.

#### IV L4/L5 Trojans (Auron)

A smaller population trapped at Auron's Lagrange points at 1.31 AU. Less massive due to Auron's lower mass, but notable because they sit in the habitable zone. Small volatile-rich bodies at Earth-like stellar distances. Potential early-game targets: closer than the main belt, scientifically interesting, and a testbed for asteroid rendezvous missions before committing to the belt.

### Asteroid belt structure (VI, expanded)

The belt (2.5-3.8 AU) is sculpted by Teros's resonances. Kirkwood-style gaps at the 3:1 (~2.5 AU), 5:2, 7:3, and 2:1 (~3.8 AU) mean-motion resonances carve the belt into distinct zones. The metal-rich disk origin produces a higher metallic fraction than our asteroid belt:

- **Inner belt (2.5-2.8 AU)**: dominated by metallic (M-type) and silicate (S-type) bodies. Dense, iron-rich. Primary mining targets for platinum-group metals.
- **Mid belt (2.8-3.3 AU)**: mixed composition. Transitional zone.
- **Outer belt (3.3-3.8 AU)**: dominated by carbonaceous (C-type) bodies. Volatile-rich: water, organics, carbon compounds. Primary mining targets for Thalos's scarcity resources.

Largest body: ~500 km diameter, possibly differentiated with a metallic core and silicate mantle. A dwarf planet candidate with enough gravity to be roughly spherical. Potential site for a permanent belt operations hub.

Individual asteroids and minor bodies within the belt will be detailed separately.

### Kuiper belt analogue (30-55 AU)

A broad disk of icy bodies beyond Pallos. The population has structure carved by resonances with the ice giants:

- **Classical belt (38-48 AU)**: dynamically cold, low eccentricity (< 0.1), low inclination (< 5°). These objects have sat undisturbed since formation. The most pristine material in the accessible system.
- **Resonant populations**: objects locked in mean-motion resonances with Pallos (3:2, 2:1, 5:3). Protected from scattering by the resonance lock, some with significant eccentricity and inclination.
- **Scattered belt (45-55+ AU)**: objects that had encounters with the ice giants and were kicked onto eccentric, inclined orbits (inclinations up to 30°). The transition zone between the Kuiper belt and the scattered disk.

### Scattered disk (50-200+ AU)

Objects that were flung outward by past encounters with the giant planets. High eccentricity, high inclination, long orbital periods. The source population for centaurs (bodies that migrate inward and orbit among the giants). Sparsely populated but containing some large bodies on extreme orbits.

### Oort cloud analogue

A spherical shell of icy bodies at thousands to tens of thousands of AU. Effectively isotropic: objects at all inclinations, including retrograde. The source of long-period comets. Not directly reachable within the game's scope, but its existence is implied by the comets it sends inward.

---

## Named Small Bodies

### Erebos (Pluto Analogue)

| Property | Value |
|---|---|
| Semi-major axis | 42 AU |
| Orbital period | 272 years |
| Eccentricity | 0.25 |
| Inclination | 17° |
| Perihelion | 31.5 AU (crosses inside Pallos's orbit) |
| Aphelion | 52.5 AU |
| Radius | ~600 km |
| Surface gravity | ~0.03 g |
| Surface | Nitrogen/methane ices, reddish tholins |
| Atmosphere | Thin N₂, partially freezes out at aphelion |
| Resonance | 3:2 with Pallos |

Binary system. A large companion moon roughly 40% its diameter, tidally locked into a mutual orbit (both bodies face each other permanently, orbiting their common barycenter every ~6 days). The pair likely formed from a giant impact in the early Kuiper belt.

The 3:2 resonance with Pallos keeps it safe despite its orbit crossing inside the ice giant's path. At perihelion, the thin nitrogen atmosphere sublimates and expands. At aphelion, it freezes back onto the surface as frost. A world that breathes on a 272-year cycle.

Late-game exploration target. Reaching it is a statement of capability. The binary dynamics and seasonal atmosphere make it scientifically rich.

#### Nyx (Companion Moon)

| Property | Value |
|---|---|
| Orbital radius | 19,400 km |
| Radius | 240 km |
| Mass | 5.2 × 10¹⁹ kg |
| Albedo | 0.10 |
| Composition | 60% ice, 38% silicate |

The large companion in the Erebos binary system. Tidally locked into mutual orbit. Likely formed from the same giant impact that shaped the binary. Icy composition with a dirty surface.

### Vorra (Haumea Analogue)

| Property | Value |
|---|---|
| Semi-major axis | 43 AU |
| Orbital period | 282 years |
| Eccentricity | 0.19 |
| Inclination | 28° |
| Dimensions | ~900 x 550 x 450 km (triaxial ellipsoid) |
| Rotation period | 3.9 hours |
| Surface | Crystalline water ice, very high albedo |
| Ring system | Narrow, icy |

The fastest-spinning large body in the system. Its extreme rotation distorts it into an elongated egg shape. A past catastrophic collision nearly shattered it and produced a **collisional family**: a cluster of smaller icy fragments sharing similar orbits, identifiable by their matching surface composition and orbital elements. Two small moons remain in orbit, likely reaccreted debris from the same impact.

The narrow ring, orbiting close to the surface, makes it the smallest known ringed body in the system. High inclination (28°) is a relic of the collision or subsequent scattering.

### Acheron (Sedna Analogue)

| Property | Value |
|---|---|
| Semi-major axis | ~290 AU |
| Orbital period | ~4,900 years |
| Eccentricity | 0.73 |
| Inclination | 25° |
| Perihelion | 78 AU |
| Aphelion | ~500 AU |
| Radius | ~450 km |
| Surface | Dark red, ultraprocessed organics |
| Surface temp | 20-35 K |

The most distant known object in the system. Its perihelion at 78 AU is far beyond the influence of any known planet, making its origin a mystery. Possible explanations: perturbation by a passing star during the system's youth, an undiscovered distant planet, or gravitational influence from the star's birth cluster.

The surface has never been meaningfully warmed by the star. Ultraprocessed organic compounds give it a deep red color. It's a frozen relic of the original protoplanetary disk, effectively unchanged in 4.6 billion years.

Reaching it is the ultimate deep-space challenge. A mission to the Sedna analogue is a multi-decade commitment at minimum. The reward is the most pristine sample of primordial material accessible anywhere.

### Chiron (Centaur)

| Property | Value |
|---|---|
| Semi-major axis | ~13 AU |
| Orbital period | ~47 years |
| Eccentricity | 0.35 |
| Inclination | 35° |
| Perihelion | ~8.5 AU (near Planet VIII) |
| Aphelion | ~17.5 AU (near Planet IX) |
| Radius | ~120 km |
| Surface | Mixed ice and rock, intermittent cometary activity |

A Kuiper belt refugee. It migrated inward through gravitational encounters and now orbits between Nereus and Pallos on an unstable orbit. It will eventually be ejected from the system, captured by a giant, or scattered further inward. Dynamical lifetime: tens of millions of years at most.

Perihelion passages near Seraph trigger cometary outbursts: subsurface volatiles heat, pressurize, and vent, producing brief comas and dust tails. Unpredictable and spectacular. The high inclination (35°) takes it well out of the ecliptic plane, requiring dedicated mission design to reach.

Scientifically valuable as a Kuiper belt sample that's come to the player rather than requiring a trip to 40+ AU.

---

## Named Comets

### Wanderer (Short-period Comet)

| Property | Value |
|---|---|
| Semi-major axis | ~17 AU |
| Orbital period | ~52 years |
| Eccentricity | 0.92 |
| Inclination | 22° |
| Perihelion | 1.4 AU (between Thalos and Auron) |
| Aphelion | ~33 AU |
| Nucleus radius | ~6 km |

The civilization's Halley's comet. Returns within a lifetime but rarely: two or three appearances per century. Bright enough to be visible with the naked eye for weeks around perihelion, developing a prominent dust tail and a fainter ion tail. Deeply embedded in mythology, calendar systems, and cultural memory. Historical records of its appearances stretch back millennia.

Its perihelion at 1.4 AU brings it between Thalos and Auron, meaning it passes through the busiest part of the system. An intercept mission during perihelion approach is a dramatic early-to-mid game objective: time-limited (the window is narrow), scientifically valuable (pristine volatiles and organics), and culturally resonant (you're visiting the mythological wanderer).

The comet is Kuiper belt-derived, captured into its current orbit by past interactions with the ice giants. It has been losing mass each perihelion for thousands of orbits. The surface is a dark, processed crust with active jets on the sunward side.

### Great Comet (Once-in-a-civilization event)

| Property | Value |
|---|---|
| Semi-major axis | Effectively parabolic (~10,000+ AU) |
| Eccentricity | ~0.9999 |
| Inclination | 67° |
| Perihelion | 0.45 AU (inside Ashara's orbit) |
| Nucleus radius | ~30 km |

A long-period comet from the Oort cloud. Appears once during the game timeline, unpredicted. First detected as a faint smudge in deep-space survey data, it brightens over months as it falls sunward on a near-parabolic orbit.

At perihelion (0.45 AU), it is spectacularly bright: easily visible in daylight, with a tail spanning 30°+ of sky. The high inclination (67°) means it plunges through the ecliptic plane at a steep angle, reinforcing that Oort cloud objects come from a spherical distribution, not the flat disk.

The nucleus is enormous by cometary standards (30 km). This is likely its first pass through the inner system, so the surface is pristine: unprocessed primordial ices, pre-solar grains, organic compounds unaltered since the system formed. A one-time scientific opportunity.

An intercept mission is challenging: the high inclination demands a large plane-change maneuver (expensive in delta-v), and the trajectory is only known with sufficient precision a few months before perihelion. A time-pressured, resource-intensive mission with irreplaceable scientific payoff. The player can also simply watch it from Thalos, which is its own kind of experience.

After perihelion, it swings back outward and will not return for millions of years, if ever. The gravitational perturbations of the inner passage may alter its orbit enough to eject it from the system entirely.

---

> **Note:** Individual asteroids, minor Kuiper belt objects, and insignificant short-period comets will be catalogued separately. The bodies listed above are the structurally or narratively significant objects that define the system's character beyond the major planets.

---

## System Summary

### Difficulty gradient (rocky bodies)

| Body | Gravity | Delta-v to orbit | Character |
|---|---|---|---|
| I. Ignis | 0.16 g | ~0.9 km/s | Trivial landing, thermal hell |
| V. Vaelen | 0.21 g | ~1.7 km/s | Easy, friendly waypoint |
| IV-a. Pelagos | 0.54 g | ~4.2 km/s | Moderate, scientifically priceless |
| III. Thalos | 0.91 g | ~6.5 km/s | Home base |
| II. Ashara | 1.09 g | ~9.5 km/s | Punishing, thick atmosphere |

### Orbital distances

| Route | Distance (AU) | Character |
|---|---|---|
| Thalos to Mira | 0.0013 | First destination |
| Thalos to Auron (conjunction) | 0.31 | Closest neighbor, every ~3 years |
| Thalos to Ashara | ~0.35 | Short but punishing |
| Thalos to Vaelen (favorable) | ~0.5 | Varies with eccentricity |
| Thalos to Vaelen (unfavorable) | ~1.2 | Wait for a better window |
| Thalos to Teros | ~4.0 | Major expedition |
| Thalos to Seraph | ~7.5 | Ring tourism |
| Thalos to Pallos | ~27 | Deep space |
| Thalos to Erebos (perihelion) | ~30 | Kuiper belt, late-game |
| Thalos to Acheron (perihelion) | ~77 | Extreme deep space |

### System structure by distance

| Zone | Distance (AU) | Contents |
|---|---|---|
| Inner system | 0.3-1.7 | Ignis, Ashara, Thalos, Auron, Vaelen |
| Main belt | 2.5-3.8 | Asteroids, dwarf planet candidate |
| Giant planets | 5.0-28 | Teros, Seraph, Nereus, Pallos; Chiron |
| Kuiper belt | 30-55 | Erebos, Vorra, icy dwarfs |
| Scattered disk | 50-200+ | Acheron, detached objects |
| Oort cloud | 10,000+ | Source of long-period comets |

### System stability

| Component | Status |
|---|---|
| Thalos–Auron resonance lock | Stable indefinitely |
| Outer giants (Teros–Pallos) | Stable indefinitely |
| Erebos (3:2 with Pallos) | Resonance-protected, stable |
| Vaelen | Unstable, ejection in ~1 Gyr |
| Chiron | Unstable, ejection or capture in ~10 Myr |
| Ignis | Stable but slowly spiraling inward |

### Biological inventory

| Body | Life | Complexity | Energy source |
|---|---|---|---|
| Thalos | Yes | Technological civilization | Photosynthesis |
| Pelagos | Yes | Cambrian-equivalent marine | Photosynthesis + chemosynthesis |
| Glacis | Yes | Microbial | Chemosynthesis |
