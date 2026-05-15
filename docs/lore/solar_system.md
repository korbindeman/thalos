# Solar System

This document is the player-facing canon for the Pyros system. It is not
a complete astronomical catalog. Bodies are listed here because they define
gameplay, navigation, exploration, or the identity of the system.

The design goal is a readable system with rich local neighborhoods: a small
number of major destinations, with moons and minor bodies becoming relevant
once the player is operating in their parent system.

## Design Principles

### Map Readability

Every permanently named body should earn its place by being mechanically
distinct, visually distinct, or narratively central. The system should feel
like a campaign map first and a real catalog second.

Moons can be more numerous than planets because they are local texture.
They should not all compete as top-level destinations in the default map.
Planetary systems should collapse by default and expand when focused,
surveyed, or approached.

### Scale Philosophy

The system uses a hybrid scale model tuned for orbital gameplay:

| Category | Scale | Rationale |
|---|---:|---|
| Star | 1:1 | Anchors luminosity, orbital periods, and habitable-zone distances. |
| Orbital distances | 1:1 | Preserves transfer windows and long-range navigation. |
| Gas and ice giants | 1:1 | Keeps spectacle, moon systems, and gravity wells convincing. |
| Thalos | 1:2 | Keeps launch demanding while avoiding real-Earth orbital cost. |
| Ashara | 2:3 | Harder than Thalos, especially through atmosphere. |
| Vaelen | 1:3 | Easy surface operations and first practical interplanetary landing. |
| Rocky moons and minor worlds | 1:2 to 1:3 | Tuned per gameplay role. |

### Progression

The intended experiential ladder is:

1. **Mira** - first moon and first orbital transfer.
2. **Selva** - optional Minmus-style low-gravity target.
3. **Auron flyby or orbit** - first interplanetary spectacle.
4. **Vaelen landing and return** - first practical interplanetary surface mission.
5. **Pelagos landing and return** - first serious off-world world.
6. **Ceryx and the belt** - midgame logistics and resource operations.
7. **Teros and its moons** - outer-system expedition and hazard management.
8. **Nereus** - deep-space ice giant operations.
9. **Erebos/Nyx** - resonant icy-binary achievement.
10. **Vorra** - advanced optional oddball target.

## Formation

Pyros formed in a metal-rich stellar nursery. Nearby supernova enrichment
left the protoplanetary disk with an unusually high iron-to-silicate ratio.
The inner rocky bodies are dense, iron-rich worlds; farther out, volatile
and ice fractions rise.

This single premise explains the system's core flavor:

- Thalos is a small but high-gravity iron-rich homeworld.
- Ashara is an especially punishing inner rocky planet.
- Vaelen is lower-density, colder, and volatile-poor at the surface.
- The belt is rich in metals, water-bearing minerals, and carbon compounds.
- Outer small bodies preserve volatile-rich primordial material.

A wide-orbit companion star, Aetheros, formed in the same stellar nursery
and settled into a ~2,500 AU semi-major-axis orbit around Pyros during
the cluster's dispersal. Aetheros is a single old M-dwarf with its own
planetary system reserved for a later expansion. Its visible presence in
Thalos's sky has shaped the civilization's cosmic imagination since
prehistory.

## Pyros

| Property | Value |
|---|---:|
| Type | G2V star |
| Mass | 1.0 solar masses |
| Radius | 696,000 km |
| Luminosity | 1.0 solar luminosities |
| Age | ~4.6 billion years |

Pyros is a solar twin. It exists to make the rest of the system legible:
habitable-zone placement, transfer distances, solar power, and lighting all
follow familiar expectations.

## I. Ashara

| Property | Value |
|---|---:|
| Role | Inner-system hard mode, sulfur world |
| Semi-major axis | 0.65 AU |
| Orbital period | ~191 days |
| Radius | 4,035 km |
| Surface gravity | 1.09 g |
| Escape velocity | 9.3 km/s |
| Atmosphere | ~80 atm, CO2 dominant with significant SO2 and H2SO4 clouds |
| Surface temperature | ~720 K |
| Rotation | Slow prograde (~80-day day) |
| Surface signature | Yellow sulfur deposits over dark basalt, active volcanism |

Ashara is the system's single inner hazard world. It is not merely hot;
it is hard to operate from. The gravity well is deeper than Thalos's,
the atmosphere is crushing, and ascent from the lower atmosphere is a
serious engineering problem.

Visually, Ashara is the sulfur planet. The supernova that enriched
Pyros's nursery seeded sulfur and other volatiles into the inner zone
alongside iron. On Thalos and Vaelen the sulfur is buried or chemically
locked; on Ashara the heat and active geology have brought it to the
surface in massive quantities. From space the disk reads muddy yellow
to ochre with cream upper-atmosphere haze bands. The surface is dark
basaltic plains streaked with yellow sulfur deposits, ringed by active
outgassing vents and fresh lava flows, with highland regions where
sulfur vapor freezes out as pale sulfur snow. Yellow plumes of sulfur
ejecta and glowing lava seams are visible from orbit, especially
against the night side.

The atmospheric chemistry adds operational hazards beyond pressure and
heat. Sulfuric acid clouds at multiple altitudes attack metals — landers
need acid-resistant alloys, aerostats must fight chemical corrosion on
top of pressure. The sky from the surface is yellow-orange and hazy,
with acid drizzle at certain altitude bands. The night horizon glows
faintly with lava-light from active flows.

Arrival can be forgiving because the thick atmosphere enables aggressive
aerobraking. Departure is the punishment. Long-term presence means high
altitude aerostat infrastructure, not casual surface bases.

### Khalkos

| Property | Value |
|---|---:|
| Role | Captured metal moon |
| Orbital radius | ~25,000 km |
| Orbital period | ~17 hours |
| Radius | ~120 km |
| Surface gravity | ~0.02 g |
| Escape velocity | ~0.2 km/s |
| Atmosphere | None |
| Surface | Irregular, heavily cratered metallic regolith, iron-nickel-sulfide composition |

Khalkos is Ashara's only natural satellite: a captured metal-rich
asteroid in a tight orbit. As Pyros's nursery dispersed, several
iron-rich fragments fell toward the inner system; most were swallowed
by Ashara's deep gravity well, but Khalkos stayed in orbit.

Visually, Khalkos is irregular and dark gunmetal grey — similar in
composition to Lyssos but on a much smaller scale and without the
heavy space-weathering history. The Ashara-facing hemisphere is
scorched darker by Ashara's infrared radiation; the outward-facing
side reads slightly lighter.

Gameplay-wise, Khalkos is a low-stakes target near Ashara: a staging
point for descent missions, a metal-extraction target without
atmospheric hazards, and an observation platform for surveying
Ashara's surface and plume activity from above the acid clouds.

## II. Thalos

| Property | Value |
|---|---:|
| Role | Homeworld |
| Semi-major axis | 1.00 AU |
| Orbital period | 365.25 days |
| Radius | 3,186 km |
| Surface gravity | 0.91 g |
| Escape velocity | 7.5 km/s |
| Delta-v to low orbit | ~6.5 km/s |
| Atmosphere | ~0.85 atm, N2/O2 |
| Hydrosphere | ~65 percent ocean |
| Land cover | Heavy vegetation on iron-rich lateritic soils |
| Axial tilt | 23 deg |

Thalos is the inhabited homeworld: smaller than Earth, dense, metal-rich,
and geologically old. Its oversized iron core gives it a strong magnetic
field and a resource-rich surface history.

From orbit the 35 percent land area reads deep green: heavy native
vegetation on iron-rich lateritic soils, with rust-red ground showing
through where forest cover thins or breaks. Thalos looks lush. The soil
chemistry still resists cultivated agriculture, which is what keeps the
civilization coastal and ocean-fed (see `civilization.md`).

Thalos is not an easy launch site. Reaching orbit remains the first major
skill gate, but the planet is small enough that spaceflight is practical
without Earth-scale launch vehicles.

### Mira

| Property | Value |
|---|---:|
| Role | First moon |
| Orbital radius | 192,000 km |
| Orbital period | ~16 days |
| Radius | 869 km |
| Surface gravity | 0.12 g |
| Escape velocity | 1.5 km/s |
| Apparent size from Thalos | ~0.52 deg |

Mira is the obvious first off-world destination. It is large enough to feel
like a real moon, close enough to reach early, and visually important from
Thalos's surface.

Its role is orbital literacy: transfers, capture, landing, return, eclipses,
and basic surface operations.

### Selva

| Property | Value |
|---|---:|
| Role | Minmus-style optional moon |
| Orbital radius | 269,000 km |
| Radius | 190 km |
| Surface gravity | Very low |
| Inclination | 3 deg |
| Surface | Pale olivine-rich regolith, heavily cratered with sharp relief |

Selva is the early-game optional moon. It is easier to land on than Mira,
but less obvious to target: farther out, smaller, slightly inclined, and
less forgiving for sloppy encounters.

Visually, Selva is the system's small high-relief body. The surface is
pale green-grey olivine-rich regolith over an ancient cratered substrate;
billions of years of impacts have sharpened the topography into pronounced
crater rims, central peaks, and saw-toothed ridges between overlapping
basins. The body is too small to have eroded its impact history into the
smooth shapes typical of larger moons, so every crater is preserved in
high relief and the terminator reads jagged against the sky.

It gives players a low-gravity playground and a reason to practice
precise transfers before leaving the Thalos system.

## III. Auron

| Property | Value |
|---|---:|
| Role | Nearby spectacle and midgame system |
| Semi-major axis | 1.31 AU |
| Orbital period | ~547 days |
| Mass | ~40 Earth masses |
| Radius | 44,600 km |
| Atmosphere | H2/He, peach and coral base with cream high-altitude bands |
| Banding style | Saturn-style smooth diffuse zones, low contrast |
| Rings | Moderate icy ring system |

Auron is close on purpose. Its presence in the Thalos sky is a core identity
feature of the setting. At favorable conjunctions the disk and rings are
naked-eye objects, making it the first interplanetary target many players
will want to visit.

Visually, Auron is the warm giant: a peach-to-coral base with cream
high-altitude haze bands. Trace organic compounds and warmer upper-
atmosphere chemistry give the warm cast; a thick haze layer damps band
contrast Saturn-style, so the zones read as soft latitudinal transitions
rather than Jupiter's hard stripes. The rings are icy and read distinctly
cool-blue against the warm body — a photogenic complement that defines
Auron's silhouette.

Reaching Auron is not the same as conquering its moon system. A flyby or
high orbit is an early spectacle; serious operations around Pelagos are a
later milestone. Auron also shares its orbit with Lyssos, a Mercury-class
metallic world locked at the L4 Lagrange point 60 degrees ahead. Lyssos
is described separately below.

### Pagos

| Property | Value |
|---|---:|
| Role | Inner Auron moon, water-ice depot |
| Orbital radius | ~310,000 km |
| Orbital period | ~3.1 days |
| Radius | 540 km |
| Surface gravity | 0.05 g |
| Escape velocity | 0.8 km/s |
| Atmosphere | None |
| Surface | Heavily cratered water ice, bright bluish-white |

Pagos is the practical first stop in the Auron system: an airless ice moon
inside Pelagos's orbit, bright enough to read as a visible morning crescent
from Pelagos's surface. No atmosphere, no biosphere, straightforward
water-ice extraction for propellant, with sunlight still strong enough for
solar power to be viable.

It gives players a low-stakes intermediate before committing to Pelagos's
ocean-world landing campaign: arrive in the Auron system, refuel and stage
at Pagos, then descend through the blue haze.

### Pelagos

| Property | Value |
|---|---:|
| Role | Ocean-life moon and major midgame world |
| Orbital radius | 500,000 km |
| Orbital period | ~6.2 days |
| Radius | 2,350 km |
| Surface gravity | 0.54 g |
| Escape velocity | 5.0 km/s |
| Delta-v to low orbit | ~4.2 km/s |
| Atmosphere | ~1.4 atm, N2/CO2/H2O with blue aerosol haze |
| Hydrosphere | ~85 percent ocean |

Pelagos is the most biologically important body outside Thalos. Stellar
insolation and tidal heating from Auron sustain a cold ocean world with
volcanic island arcs, hydrothermal circulation, and complex marine life.
Its atmosphere is thick, humid, and optically deep without becoming
Titan-like: a nitrogen-dominated blue shell with CO2 greenhouse warming,
water clouds, sea-salt and sulfate aerosols, and trace organic haze. Sunlight
still reaches the volcanic shelves where photosynthetic ecosystems are
dense, but the sky and orbital limb read softer and milkier than Thalos.

It is harder than Vaelen in the ways that matter: moon-system targeting,
entry control, ocean-world landing constraints, weather, and a meaningful
ascent vehicle. Its atmosphere helps with arrival but adds engineering
requirements.

### Carpo

| Property | Value |
|---|---:|
| Role | Captured contact binary |
| Orbital radius | ~799,000 km |
| Extent | ~220 km long, two fused lobes (~100 km each) |
| Surface | Two-toned: sulfur-yellow lobe and clean grey silicate lobe |
| Atmosphere | None |

Carpo is not one body but two — a peanut-shaped contact binary captured
into Auron's orbit. The two lobes have visibly different compositions
and surface histories, suggesting they originated as separate bodies
that later coalesced under low-velocity contact and never fully merged.
One lobe reads sulfur-yellow from UV and radiation processing of
volatile-rich material; the other is clean grey silicate, the remnant
of a more refractory progenitor. The neck between them is dust-pooled
and slightly brighter.

Visually, Carpo is the system's distinctive small body: lumpy,
two-toned, identifiable from any approach geometry. Real-world analogs
are 486958 Arrokoth and comet 67P, scaled up.

Gameplay-wise, Carpo remains a convenient staging target. Landing
operations are easy (negligible gravity), the two-lobe geometry adds
visual interest, and the compositional split gives a reason to sample
both ends.

### Theron

| Property | Value |
|---|---:|
| Role | Captured tholin-coated outer moon |
| Orbital radius | ~2,094,000 km |
| Radius | 120 km |
| Orbit | Inclined retrograde |
| Surface | Pinkish-cream tholin coating, mottled regional variations |

Theron is the local challenge moon of the Auron system: distant,
inclined, retrograde, and visually distinctive. The surface is coated
in tholins — pinkish-cream organic compounds formed by UV processing
of methane and nitrogen on a captured icy substrate. Regional color
variations show as mottled patches: cream highlands, salmon mid-tones,
deeper red-brown around impact features and tectonic fractures.

The orbit and surface chemistry tell the story together: Theron was
captured from the outer system, probably from the trans-Nereus zone,
and brings its outer-system surface chemistry inward with it. It gives
the Auron system a high-skill navigation target and a sample of
distant-origin material without crowding the main progression.

## Lyssos (Auron L4 Co-orbital)

| Property | Value |
|---|---:|
| Role | Co-orbital Mercury-class destination |
| Semi-major axis | 1.31 AU (same as Auron, 60 deg ahead at L4) |
| Orbital period | ~547 days (matches Auron) |
| Radius | ~2,400 km |
| Surface gravity | ~0.42 g |
| Escape velocity | ~3.9 km/s |
| Atmosphere | None |
| Surface | Burnished metallic regolith, heavy cratering |

Lyssos co-orbits Auron at the L4 Lagrange point, 60 degrees ahead of
Auron in their shared orbit around Pyros. Both bodies have the same
~547-day year. Hohmann transfer windows from Thalos to Lyssos are offset
by roughly a quarter-orbit from Thalos-to-Auron windows, so Lyssos is
not a body you fold into an Auron campaign — it is its own dedicated
trip.

Visually, Lyssos is a stripped, metal-rich Mercury-class world: bare
iron-nickel regolith, heavy cratering from 4.6 Gyr of impacts, no
atmosphere or water to oxidize the surface. The base reads dark
gunmetal grey from solar-wind sputtering and micrometeorite weathering,
with bright radial crater rays where recent impacts have exposed
fresher metal. Subtle color variation from sulfide and silicate
inclusions: greys, deep browns, occasional copper-toned regions. The
closest real-world analog is the M-type asteroid 16 Psyche.

Lore: Lyssos is the iron-rich proto-Auron material that did not make it
into Auron's core. During the protoplanetary disk's dispersal it pooled
at the L4 Lagrange point and stabilized there. Composition-wise it is
genetically related to Auron's interior — chemically what Auron's deep
core is presumed to be — in solid, accessible form.

Lyssos is a mid-to-late game destination. The Δv is modest (1.31 AU,
no atmosphere, low surface gravity), but the orbital offset means
timing is its own discipline.

## IV. Vaelen

| Property | Value |
|---|---:|
| Role | First practical interplanetary surface world |
| Semi-major axis | 1.73 AU |
| Orbital period | ~2.28 years |
| Eccentricity | ~0.12 |
| Radius | 1,130 km |
| Surface gravity | 0.21 g |
| Escape velocity | 2.2 km/s |
| Delta-v to low orbit | ~1.7 km/s |
| Atmosphere | Thin CO2, iron-oxide dust-loaded |
| Sky color | Rust-red (Mie-dominant from suspended dust) |

Vaelen is the Mars-role world, but tuned for gameplay clarity. Auron may be
closer and more spectacular, but Vaelen is the first serious interplanetary
landing and return mission because its surface operations are forgiving.

Its eccentric orbit creates meaningful transfer-window variation without
letting the planet live in Auron's shadow. Favorable windows are cheap;
unfavorable windows teach players to wait.

The surface tells one story in two halves. Roughly half the planet is a
single continent-scale dune sea: a planet-scale erg of linear dunes, star
dunes, and barchan margins, pooled into the long topographic low that
records Vaelen's ancient ocean basin. The other half is rust-red cratered
terrain (Mars-like crater density) cut by dried channel networks, with
cyan evaporite floors (gypsum, sulfates, chloride pans) lining the
ancient lake beds and salt plains. Buried ice persists in scarps and
near the poles.

The thin atmosphere is loaded with iron-oxide fines mobilized off the
dune sea. The sky reads rust-red from the surface, the limb glows
ferrous-orange from orbit, and aerial perspective warms distant terrain
toward red. The dune basin is not just a feature, it sources the
atmospheric color; the two together are Vaelen's visual signature.

Vaelen is strategically useful as a staging world between the inner
system, Auron, the belt, and Teros.

### Kael

| Property | Value |
|---|---:|
| Role | Inner Vaelen moon |
| Orbital radius | ~94,000 km |
| Radius | 14 km |
| Character | Small rubble or captured rock |

Kael gives Vaelen local orbital gameplay: rendezvous, tiny-body landing,
depot placement, and moon-assisted mission planning.

### Xxirt

| Property | Value |
|---|---:|
| Role | Outer Vaelen moon |
| Orbital radius | ~150,000 km |
| Radius | 8 km |
| Character | Tiny outer companion |

Xxirt is not a major destination. It exists because Vaelen should feel like
a complete local system once the player arrives.

## V. Ceryx

| Property | Value |
|---|---:|
| Role | Main-belt dwarf and logistics hub |
| Semi-major axis | ~2.7 AU |
| Orbital period | ~4.4 years |
| Radius | ~470 km |
| Surface gravity | ~0.03 g |
| Escape velocity | ~0.5 km/s |
| Surface | Dark hydrated regolith, craters, bright salt deposits |

Ceryx is the named face of the asteroid belt. The rest of the belt can be
population and survey content, but Ceryx gives the region a clear destination.

Its value is logistical rather than dramatic: water, carbon compounds,
hydrated minerals, nearby metallic asteroids, low-gravity construction, and
midgame refueling. It is not hard to land on; it is hard to use efficiently
because the belt demands planning, windows, and infrastructure.

## VI. Teros

| Property | Value |
|---|---:|
| Role | Outer-system giant and hazard gate |
| Semi-major axis | 5.0 AU |
| Orbital period | ~11.2 years |
| Mass | ~1.2 Jupiter masses |
| Radius | 74,000 km |
| Atmosphere | H2/He with NH3 / NH4SH / H2O cloud decks; deep sienna and burnt-orange bands with bone-white zones |
| Banding style | Jupiter-style high-contrast turbulent bands, planet-scale cyclones |
| Signature feature | Great Dark Spot — slow-rotating mid-southern cyclone, ~15,000 km across |
| Rings | Faint dusty rings |
| Hazard | Intense radiation belts; vivid polar auroras driven by the active magnetosphere |

Teros is the first true outer-system expedition. It is massive, dangerous,
and operationally different from Auron. The player should feel the jump in
scale: long cruise, weak sunlight, radiation planning, and complex moon
encounters.

Visually, Teros is the stormy giant. Where Auron is calm peach with soft
banding, Teros reads as deep sienna and burnt-orange zones cut by
bone-white belts, with chaotic turbulent boundaries and planet-scale
cyclones. A permanent Great Dark Spot — a slow-rotating cyclone ~15,000
km across at mid-southern latitudes — is visible from most approach
trajectories and gives Teros a one-glance silhouette. Vivid auroras
flicker at the polar regions, driven by the active magnetosphere that
also produces the radiation belts.

### Pyrith

| Property | Value |
|---|---:|
| Role | Volcanic inner moon |
| Orbital radius | ~422,000 km |
| Radius | 910 km |
| Surface | Multicolored sulfur palette over dark basalt, active volcanism |
| Hazard | Deep in Teros's radiation belts |

Pyrith is the system's volcanic spectacle. Tidal heating from Teros
drives the most active surface vulcanism anywhere outside Ashara, and
the surface chemistry is a rich palette of sulfur compounds in different
oxidation states: yellow sulfur plains, orange and brown sulfur deposits,
red rings of sulfur compounds around hot volcanic centers, and white
sulfur-dioxide frost in cooler regions. Dark basalt shows through where
recent flows have not yet been re-coated. Visible eruption plumes and
glowing magma seams stand out at any zoom level.

Pyrith pairs thematically with Ashara: both are sulfur worlds, but
Pyrith's sulfur is mobilized by tidal heating rather than solar. Pyros's
metal-rich nursery seeded sulfur into both the inner zone and the
condensing outer disk, and Pyrith is its outer-system expression.

Pyrith is visually loud and mechanically hostile. It sits deep in
Teros's radiation environment and rewards players who can plan short,
efficient, high-risk operations.

### Glacis

| Property | Value |
|---|---:|
| Role | Europa-like microbial-life moon |
| Orbital radius | ~670,000 km |
| Radius | 780 km |
| Surface | Ice shell over global ocean |
| Life | Chemosynthetic microbial ecosystem |

Glacis is the science prize of Teros. Exploration means radiation shielding,
plume sampling, ice drilling, or carefully timed low-altitude passes.

### Khalin

| Property | Value |
|---|---:|
| Role | Titan-analog hydrocarbon moon |
| Orbital radius | ~1,300,000 km |
| Radius | 2,700 km |
| Surface gravity | 0.16 g |
| Escape velocity | 2.5 km/s |
| Atmosphere | ~1.5 bar N2/CH4 with thick tholin haze |
| Surface temperature | ~90 K |
| Surface | Methane/ethane lakes (polar), cryovolcanic flows; obscured from orbit by haze |

Khalin is Teros's Titan analog: a major moon wrapped in an opaque amber
tholin haze. From orbit the surface is invisible — radar and infrared
are the only ways to image what is underneath. The haze gives Khalin a
distinctive sunset-orange disk visible against the deep sienna of Teros.

Beneath the haze, the surface is cold (~90 K), with methane and ethane
lakes pooling in the polar regions and cryovolcanic flows scarring the
mid-latitudes. Organic chemistry is rich: the hydrocarbon inventory
across Khalin is the late-game organic feedstock for the civilization's
industrial supply chain.

Khalin is operationally distinct from Teros's other moons. The
atmosphere makes arrival forgiving (aerobraking, parachutes), but the
cold and the surface chemistry require specialized hardware. Pyrith is
hot and irradiated; Glacis is icy and biologically protected; Calyx is
the practical staging point. Khalin is the chemistry mission.

### Calyx

| Property | Value |
|---|---:|
| Role | Safer outer Teros moon, two-tone surface |
| Orbital radius | ~1,880,000 km |
| Radius | 1,205 km |
| Surface | Two-toned: bright water ice on the trailing hemisphere, dark carbonaceous on the leading hemisphere |

Calyx is Teros's distinctive outer moon: a body whose two hemispheres
read as visibly different worlds. The leading hemisphere is coated in
dark organic and carbonaceous material that Calyx has swept up moving
through Teros's outer system over geological time; the trailing
hemisphere preserves the original bright water-ice surface. From orbit
the boundary between the two is sharp enough to be visible at silhouette
range — a one-glance identifier from any approach trajectory.

Calyx is also the practical moon of the Teros system: farther from the
worst radiation, easier to operate around, and useful as a staging
point for the more dangerous inner moons. The two-tone surface makes
landing site selection a real choice — bright ice for water harvesting
versus the dark organic-rich terrain for surface chemistry.

## VII. Nereus

| Property | Value |
|---|---:|
| Role | Sole ice giant |
| Semi-major axis | 12 AU |
| Orbital period | ~41.6 years |
| Mass | ~12 Earth masses |
| Radius | 22,000 km |
| Atmosphere | H2/He/CH4, deep blue-green |
| Ring system | Sparse set of thin icy rings |
| Axial tilt | ~18 deg |

Nereus is the system's only ice giant. Methane absorption gives it a deep
blue-green disk, while its moderate axial tilt keeps the ring plane readable
without making the planet roll around Pyros. One slim bright inner ring
anchors clustered, dimmer outer ringlets that catch light at high phase
angles without turning the planet into a Saturn analogue.

Missions to Nereus are deep-space operations: low solar power, long travel
times, cold thermal design, and sparse rescue options.

### Orias

| Property | Value |
|---|---:|
| Role | Captured ice-giant moon |
| Orbital radius | ~900,000 km |
| Radius | ~950 km |
| Orbit | Inclined retrograde |
| Surface | Cantaloupe terrain, mottled greenish-tan-pinkish, nitrogen frost, cryovolcanic features |
| Signature feature | Dark reddish polar cap (tholin-deposited) |

Orias gives Nereus a local objective: a captured moon on an awkward
retrograde orbit, useful as both a science target and a navigation
test in the outer system.

Visually, Orias is the system's most chromatically distinctive moon.
The surface shows characteristic "cantaloupe terrain" — densely packed
shallow depressions in a regular pattern, formed by ice diapirism over
geological time — colored in mottled greenish-tan-pinkish hues from
organic surface processing. Bright patches of nitrogen frost migrate
seasonally across the surface as Nereus moves around Pyros over its
~42-year year.

A distinctive permanent feature anchors Orias's silhouette: a dark
reddish polar cap. Methane and nitrogen escaping Nereus's upper
atmosphere migrate to Orias's cold polar trap and accumulate there
over geological timescales, processed by UV into deep red tholin
deposits. The cap reads as a permanent dark crown visible from any
approach trajectory — *the moon with the bloody crown*.

## VIII. Erebos

| Property | Value |
|---|---:|
| Role | Resonant icy-binary achievement |
| Semi-major axis | ~15.7 AU |
| Resonance | 3:2 with Nereus |
| Radius | ~600 km |
| Surface gravity | ~0.03 g |
| Surface | Nitrogen and methane ices, reddish organics |

Erebos is the late-game icy-frontier destination. It is far enough that
reaching it is a statement of capability, but close enough to belong to the
normal outer-system ladder rather than pure prestige content.

Its resonance with Nereus protects it despite a more eccentric, inclined
orbit. Seasonal volatile transport gives it a surface that changes over
long timescales.

### Nyx

| Property | Value |
|---|---:|
| Role | Erebos companion |
| Orbital radius | ~19,400 km |
| Radius | 240 km |
| Composition | Dirty ice and silicate |

Nyx is not just a moon; together with Erebos it creates a binary-world
mission. Navigation, lighting, and surface operations should feel different
from a normal planet-moon pair.

## IX. Vorra

| Property | Value |
|---|---:|
| Role | Advanced optional oddball |
| Semi-major axis | ~24 AU |
| Inclination | High |
| Shape | Rapidly rotating triaxial body |
| Surface | Bright crystalline water ice |
| Rings | Narrow icy ring |

Vorra is for advanced players. It should not be a default early map peer.
It is discovered through survey progression and then becomes a high-skill
target: high inclination, fast rotation, irregular shape, tight ring system,
and a collisional-family backstory.

The appeal is not distance alone. Erebos already handles the resonant
icy-frontier achievement. Vorra is the weird precision mission after that.

## Aetheros (Companion Star)

| Property | Value |
|---|---:|
| Type | Old M-dwarf, gravitationally bound to Pyros |
| Spectral class | M3V |
| Mass | ~0.25 M_sun |
| Luminosity | ~0.041 L_sun |
| Age | ~8-10 Gyr (older than Pyros) |
| Semi-major axis from Pyros | ~2,500 AU |
| Orbital period around Pyros | ~110,000 years |
| Apparent magnitude from Thalos | ~-6.3 (naked-eye in daylight) |

Aetheros is the second star in Thalos's sky. Its wide orbit around Pyros
completes once every ~110,000 years, so on any human timescale it is a
fixed bright point against the background stars. Spectroscopy identified
its spectral class long before spaceflight; the species has known it as
a small old red companion for centuries. Being significantly older than
Pyros gives Aetheros a settled, low-flare temperament despite being an
M-dwarf.

**Expansion content, not base game.** Aetheros has its own planetary
system reserved for a later expansion, bundled with a new propulsion
tier (fusion torch or beyond) capable of crossing the ~2,500 AU gap on
multi-year timescales. Specifics of Aetheros's planets are intentionally
left vague at this stage so the expansion has room to design them. The
base game's tech tree stops short of the required propulsion ceiling by
design.

In the base game, Aetheros is a physically simulated body and a
prominent visible star, but it is not reachable.

**N-body note.** Once the simulation moves from patched conics to full
N-body, Aetheros's wide orbit around Pyros is propagated as a slow drift
over centuries; until then it is treated as a fixed point with no
operational consequence.

## System Summary

### Core Destinations

| Zone | Bodies | Purpose |
|---|---|---|
| Home system | Thalos, Mira, Selva | Training, first landings, low-gravity practice. |
| Inner system | Ashara, Khalkos | Sulfur-world hard mode and its small metal moon. |
| Nearby giant | Auron, Pagos, Pelagos, Carpo, Theron | Spectacle, ice depot, ocean life, local moon operations. |
| Auron co-orbital | Lyssos | Mercury-class metallic world, timing-discipline mission. |
| Mars analogue | Vaelen, Kael, Xxirt | First practical interplanetary surface campaign. |
| Belt | Ceryx | Midgame resources, depots, logistics. |
| Outer giant | Teros, Pyrith, Glacis, Khalin, Calyx | Radiation, outer science, hydrocarbon harvesting, moon-system mastery. |
| Ice giant | Nereus, Orias | Deep-space operations and ice-giant moon geometry. |
| Icy frontier | Erebos, Nyx | Late-game binary-world achievement. |
| Advanced optional | Vorra | Survey-gated precision challenge. |
| Companion star | Aetheros | Expansion target: visible since prehistory, reachable only with post-game propulsion. |

### Surface Difficulty

| Body | Character | Relative difficulty |
|---|---|---:|
| Selva | Tiny low-gravity moon | Very low |
| Pagos | Airless ice moon (Auron system) | Very low |
| Khalkos | Tiny metal moon (Ashara system) | Very low |
| Mira | First real moon | Low |
| Lyssos | Airless Mercury-class metal world | Low |
| Ceryx | Belt dwarf | Low landing, moderate logistics |
| Vaelen | Low-gravity planet with dust storms | Moderate |
| Khalin | Cryogenic hydrocarbon moon with thick atmosphere | High |
| Pelagos | Ocean moon with atmosphere | High |
| Thalos | Homeworld launch site | High |
| Ashara | Venus-like hard mode | Extreme |

### Biological Inventory

| Body | Life | Complexity | Energy source |
|---|---|---|---|
| Thalos | Yes | Technological civilization | Photosynthesis |
| Pelagos | Yes | Complex marine ecosystem | Photosynthesis and chemosynthesis |
| Glacis | Yes | Microbial ecosystem | Chemosynthesis |

### Default Map Policy

The default solar-system map should emphasize parent bodies and major
campaign targets. Moons appear when their parent system is focused. Vorra
appears after survey progression. Belt population beyond Ceryx is generated
or discovered as needed rather than listed as permanent named bodies.
