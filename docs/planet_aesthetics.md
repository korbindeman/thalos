# The Aesthetics of Procedural Planets: A Field Guide for the Pyros System

*A reference document for orbital-first planet rendering, with scope into surface transitions. Aesthetic-first, framed as "what looks good and why / what looks bad and why." Engine-agnostic, with notes from Outerra, Star Citizen, Space Engine, Elite Dangerous, No Man's Sky, Infinity (I-Novae), Sebastian Lague, Inigo Quilez, Sean O'Neil, Bruneton, Guerrilla's Nubis, NASA/ESA imagery, and academic graphics literature.*

---

## TL;DR

- **Procedural planets fail by looking *uniform*** — equal noise at every scale, isotropic detail, banded biomes, perfect spheres, and flat oceans are the visual tells. Real planets read as the **layered residue of distinct geological processes** (tectonics, fluvial, glacial, aeolian, impact), each writing its signature at a different frequency band. If your planet looks like one fBm + a colormap, it will look procedural even at 8K.
- **Orbital "good" is mostly atmosphere, terminator, and macro-composition** — believable limb-glow with proper Rayleigh/Mie falloff, a soft (not knife-edged) day/night terminator, hemispheric land/water asymmetry, continuous mountain belts, drainage networks that read as branching trees, and clouds that obey latitude bands and cast shadows on the surface. Get those right and the planet sells itself before you even look at the ground.
- **Believability beats realism.** Earth-likeness is a trap; internal consistency is the goal. A planet whose climate, geology, biome distribution, and weathering all tell the same story (axial tilt → ice caps → drainage → erosion patterns) will be more convincing than a photoreal Earth clone with one inconsistent feature. Restraint with saturation, restraint with exotic colors, and restraint with "sci-fi flair" is what separates Outerra/Space Engine from Shadertoy demos.

---

## Key Findings

1. **The signature failure mode of procedural planets is *frequency homogeneity*.** Pure fBm produces self-similar detail at every scale — but real terrain has *distinct* processes operating at *distinct* scales. Tectonics writes 1000 km–10 km features (continents, mountain belts, rifts); fluvial erosion writes 100 km–100 m features (drainage networks, valleys); glacial writes 50 km–10 m (cirques, U-valleys); aeolian writes 10 km–1 m (dunes, ventifacts); impact writes any scale but with a specific morphology (rim, ejecta, central peak). When all those bands look like the same noise, the eye reads "fake" instantly.

2. **Atmosphere does more aesthetic work than terrain from orbit.** Sean O'Neil's GPU Gems 2 shader and Bruneton's precomputed scattering (2008) are the canonical references; both produce the limb-glow, terminator softness, and aerial perspective that *make a sphere read as a planet* rather than a textured ball. The single biggest visual upgrade you can give a mediocre terrain is a correct atmosphere.

3. **Macro composition is a hand-authored or simulation-driven layer, not noise.** Outerra, Star Citizen Planet Tech v4/v5, Elite Dangerous Stellar Forge, and the work of practitioners like Tectonics.js, Ysaneya/Infinity, and Logan Schwartz/PlaTec all converge on the same conclusion: **continental shapes need a coarse "story" layer** (tectonic plates, hotspots, basin/range), and noise is only used to perturb and detail it.

4. **Color restraint is the difference between "candy planet" and "real planet."** NASA's Blue Marble: Next Generation, ESA's true-color Mars mosaic (HRSC, 2023), and Cassini imagery all show that real planetary albedo is *desaturated* and *narrow-gamut* — Earth oceans are not pure cyan, Mars is butterscotch not red, the Moon is gray-tan not white. No Man's Sky deliberately violates this for art-direction reasons (Grant Duncan's GDC talk: 70s sci-fi covers), and that choice is read instantly as "stylized," not "real."

5. **Scale cues are mostly cloud parallax, atmospheric perspective, and proportional feature sizing.** Mountains shouldn't look like wrinkles unless they're huge; rivers shouldn't be visible from orbit unless they're Amazon-scale; biome boundaries shouldn't be knife-edges. The Outerra blog repeatedly notes that getting *scale* right (using real-Earth elevation data and refining from there) was harder than any single technique.

---

## Details

## 1. Common Pitfalls That Scream "Procedural"

These are the visual tells that mark a planet as fake within seconds:

| Tell | Why it looks bad |
|---|---|
| **Uniform fBm at all scales** | Every zoom level looks the same. Real planets show *different processes* at different scales. Inigo Quilez's frequency analysis confirms: real mountain profiles measure as ~yellow noise (H≈1), but the signature of erosion is *non-self-similar* — fBm alone misses ridges, valleys, and drainage. |
| **Isotropic detail** | Real terrain has *direction*. Drainage flows downhill, dunes align with prevailing wind, fault scarps line up with stresses, mountain belts are linear, glacial striations are parallel. A planet with no preferred directions reads as static. |
| **No large-scale tectonic structure** | Continents look "blobby" or like Voronoi cells. Real continents have *shoulders, peninsulas, and rifts* tracing plate boundaries. |
| **Repeating texture tiles visible from orbit** | A killer for Outerra-class engines; their blog explicitly cites this as a problem solved by procedural color mixing. From orbit, any tiling pattern reads as wallpaper. |
| **Color banding** | Visible quantization in atmosphere, terminator, or ocean depth. Always render in HDR / linear and dither the final 8-bit conversion. |
| **Knife-edge biome boundaries** | Real biomes interdigitate over 10–100 km. Voronoi-cell biomes visible from orbit are an instant tell. |
| **"Noise soup"** | Domain-warped fBm composited with itself looks busy but featureless. No anchor structure, no clear silhouettes. |
| **Missing scale cues** | No clouds, no atmosphere, no curvature falloff — the planet looks like a billiard ball. |
| **Hard terminator (no atmosphere)** | If the planet *should* have air, a hard day/night line says "shading is wrong." Real Earth's terminator is a fuzzy ~half-degree gradient (atmospheric refraction + scattering). |
| **Perfect spheres** | Earth's equatorial bulge is ~1/298 (≈42 km); Jupiter and Saturn are noticeably oblate. Doesn't matter for terrestrials at most viewing distances, but for fast-rotators it's a visible flaw. |
| **Uniform crater distribution on airless bodies** | Real lunar mare are *resurfaced* and crater-poor; highlands are saturated. Just sprinkling craters uniformly looks like a Whitley filter, not a moon. |
| **Crater rims with no ejecta, no rays, no degradation gradient** | Tycho-fresh and ancient should look very different. Without a degradation/age dimension, all craters look the same. |

> **Don't:** Spray a single 8-octave fBm onto a sphere, slap a height-based colormap on it, and call it a planet.
> **Do:** Author a coarse "world" layer (continents/plates/basins), let process-aware noise (ridged, billow, domain-warped) live *inside* that layer, and reserve fBm for high-frequency texture only.

---

## 2. Macro-Scale Composition: Reading as a Coherent World

The orbital silhouette is the first thing the eye registers. A planet looks coherent when its *macro features tell a consistent geological story.*

**What makes Earth read as Earth from orbit:**

- **Hemispheric asymmetry.** Earth is ~70% ocean, but the land/water split is *not uniform* — most land is in the northern hemisphere. This asymmetry reads as "real history" (continental drift since Pangaea), not "noise threshold."
- **Continuous mountain belts.** The Andes, Rockies, Himalayas, and East African Rift all trace plate boundaries — they're *lines*, not random peaks. Even at 8 px/km, you can see them.
- **Visible drainage networks.** From the ISS, the Amazon, Nile, Mississippi, Lena, and Ganges read as tree-like dark threads. They imply gravity, hydrology, and erosion all at once. NASA Blue Marble images make these legible at very low resolution.
- **Ice cap shapes.** Antarctica is roughly circular (centered on a pole); Greenland is roughly elongated (centered on land). Mars's polar caps are layered and seasonal. These shapes follow rotational + climate logic.
- **Equatorial cloud bands.** Earth's ITCZ, Jupiter's belts/zones, Saturn's bands — all driven by Hadley/zonal circulation. Latitude-banded clouds say "this planet rotates and has weather"; random puffs say "noise."
- **Dust/aerosol haze.** Look at any DSCOVR/EPIC image of Earth: there's always Saharan dust drifting over the Atlantic, biomass-burning smoke over central Africa, sun-glint on the Pacific. These transient atmospheric layers add *life*.

**What practitioners do:**

- **Outerra** uses real 90 m elevation data as the macro layer and refines fractally below. The macro shape is *given*, not invented.
- **Elite Dangerous's Stellar Forge** simulates planetary formation from first principles — accretion, tidal evolution, plate tectonics — so that every planet's *macro shape is the residue of a process*.
- **Star Citizen Planet Tech v4/v5 + Genesis** drives biomes from temperature, humidity, geology, soil type, soil depth, nutrients, sunlight exposure and slope aspect — emergent biomes rather than painted Voronoi.
- **Tectonics.js, Ysaneya/Infinity, and the World Orogen tool** all run actual plate tectonic simulations to generate continent shapes, then layer fluvial/glacial erosion on top.

> **Do:** Author or simulate a coarse plate/basin layer (~ a few hundred km features) before any noise. Place mountain *chains* along plate boundaries, not mountain *fields*. Bias land toward one hemisphere if you want it to feel like a real history exists. Add hotspot trails (Hawaii-style chains) for visual interest.
> **Don't:** Threshold a single noise function and call the high values "land." That gives you blobs without continuity, peninsulas without coherence, and no story.

---

## 3. Color and Material Palettes

**The single most common failure mode:** oversaturation. Real planet colors are *muted*.

**Reference points (true-color from orbit):**

- **Earth (Blue Marble NG, MODIS):** Oceans are dark blue tending toward green-near-coast, *not cyan*. Vegetation is olive-to-dark-green, *not Kelly green*. Deserts are tan-to-ochre, not orange. Ice is bluish-white, not pure white. Clouds are warm white, not gray.
- **Mars (ESA HRSC mosaic, 2023):** Predominantly butterscotch / dark yellowish-brown ocher, with grey-black volcanic basalt sands and lighter sulfate/clay patches. ESA explicitly notes that processed maps usually *exaggerate* color contrast; the true muted palette is what the eye would actually see.
- **Moon / Mercury:** Gray-tan, not gray. Highlands brighter than mare. Fresh crater ejecta (Tycho, Copernicus) is brighter and bluish-white due to immature regolith — and that brightness *fades over millions of years* via space weathering.
- **Titan:** Orange-haze obscures most of the surface; what's visible is brownish with darker hydrocarbon seas. Cassini's specular sun-glint on Kraken Mare is one of the most cinematic planet images ever taken.
- **Jupiter:** Cream, salmon, rust, ochre, brown — *not* primary colors. The bands are subtle gradients, not sharp lines. Gerald Eichstädt and Seán Doran's processing of JunoCam data is a great reference for how subtle the contrasts actually are.
- **Pluto:** Pale tans, pinks, blues, deep reds. The "heart" (Tombaugh Regio) is bright nitrogen ice; the surrounding Cthulhu Regio is dark red-brown tholin. The contrast is *high* but the saturation is *low*.

**Aesthetic principles:**

- **Restraint in saturation.** A good rule: take whatever color you think is right, then desaturate ~30%. Then desaturate another 10%. Earth is not the color of a corporate brand-deck globe.
- **Subtle hue variation within biomes.** Real forests are not one green — they're a mosaic of greens, browns, and yellows with seasonal modulation. Use noise to vary *hue and saturation slightly* within each biome, not just brightness.
- **Vegetation vs. mineralogy vs. ice as three orthogonal layers.** Plants pull color toward green. Mineralogy gives rock its base color (basalt = dark gray, granite = pinkish, sandstone = ochre, iron oxides = red). Ice is bright but tinted by the underlying surface and overlying atmosphere.
- **Why "fully saturated greens" look fake.** Real vegetation absorbs strongly in the red and reflects mostly in the near-IR; in the *visible*, plants look surprisingly muted. A Kelly-green planet reads as cartoon.
- **Atmospheric color shift with viewing angle.** As you look toward the limb, you see through more atmosphere — colors desaturate, shift toward the dominant scattering color (blue for Earth/Rayleigh, butterscotch-pink for Mars/dust-Mie). This is automatic if your scattering shader is correct.
- **Rayleigh vs. Mie aesthetics.** Rayleigh (small particles, λ⁻⁴) gives Earth's blue sky and red sunsets. Mie (large particles, weak wavelength dependence) gives Titan's haze, Mars's dust glow, and the *forward-scattering* highlight near the sun. Pure Rayleigh planets look "Earthy"; pure Mie planets look "dusty/hazy."
- **The "blue marble" effect.** Earth's deep oceans + thin Rayleigh atmosphere + scattered white clouds give a specific saturated-but-balanced look. To get it, you need (a) a deep dark blue ocean, (b) a Rayleigh atmosphere with proper limb-glow, (c) layered clouds with shadows. Miss any one and it doesn't read.
- **Specular highlights on oceans.** The sun-glint spot moves with the sub-solar point and is a *huge* readability cue. Without it, oceans look like flat blue paint. With it, they look wet. Cassini's Titan sea-glint is the canonical demonstration that this works for non-water liquids too.
- **Wet vs. dry surfaces.** Wet surfaces have higher specular, lower diffuse, and slightly darker base color. A river system that doesn't darken its banks looks painted-on.
- **Regolith on airless bodies.** Vacuum-weathered surfaces are *darkened and reddened* over time (space weathering). Old craters are dimmer and pinker; fresh craters are brighter and bluer. Without this gradient, all of your craters look the same age, which they aren't.

> **Do:** Sample real planetary photography for your color anchors. Restrain your saturation. Vary hue within biomes. Use specular for liquids.
> **Don't:** Pick three saturated colors and lerp between them. Don't make oceans cyan. Don't make grass Kelly green. Don't let your alien planets be neon.

---

## 4. Detail at Multiple Scales (Frequency Content)

This is the deepest aesthetic principle and where most amateurs go wrong.

**The problem with pure fBm.** Inigo Quilez's analysis of mountain profile frequency content (Hurst exponent ~1, "yellow noise") shows fBm is *statistically* a decent first approximation. But statistically-correct ≠ visually-correct. fBm produces *self-similar* detail: the same kind of bump at every scale. Real planets do not.

**The principle: each geological process owns a frequency band.**

| Process | Band | Visual signature |
|---|---|---|
| Tectonics | 10,000 km – 100 km | Continents, mountain belts, rift valleys, plate boundary linearity |
| Volcanic | 1,000 km – 1 km | Shield volcanoes (gentle slopes), stratovolcanoes (steep cones), lava flows (lobate), calderas |
| Fluvial / hydraulic erosion | 1,000 km – 10 m | Dendritic drainage networks, V-shaped valleys, alluvial fans, deltas |
| Glacial | 100 km – 10 m | U-shaped valleys, cirques, fjords, drumlins, moraines, striations |
| Aeolian | 100 km – 1 m | Dune fields with directional alignment, ventifacts, deflation hollows |
| Impact | any scale | Rims, ejecta blankets, ray systems (when fresh), central peaks (large), secondaries |
| Coastal | 10 km – 1 m | Headland-bay alternation, wave-cut cliffs, sand spits |

These bands *interact*. A fluvial network erodes a tectonically-uplifted mountain belt. A glacier carves a U-valley out of a previously-fluvial V-valley. Aeolian dunes drape over an impact basin. **The richness of real planets is the *layered residue* of these interactions.**

**Why "noise + colormap" looks wrong.** It produces detail without *meaning*. The eye picks up that the high-frequency wiggles are not connected to the low-frequency story. Process-driven terrain has *coherent narratives* at every scale — a peak you see from orbit has visible drainage radiating from it, those drainages have V-valleys, those V-valleys have alluvial fans at their mouths.

**Practical recipes from practitioners:**

- **Inigo Quilez's "fBm with derivative-damped octaves"** (`a += b*n.x/(1.0+dot(d,d))`) approximates erosion: where the terrain is steep, higher octaves contribute less. It produces flatter valley floors and sharper ridges — instant erosion-look from a one-line modification.
- **Domain warping** (Quilez): warping noise space with another noise function produces the swirly, river-like patterns of real eroded terrain. Simply better-looking than vanilla fBm.
- **Ridged multifractal** (Musgrave, *Texturing & Modeling*): `1 - |noise|` creates sharp ridge networks that look like fault scarps and mountain crests. Ysaneya's Infinity engine cites a 16-octave ridged fBm as the primary mountain generator.
- **Hydraulic erosion simulation** (Mei et al. 2007 GPU; Št'ava et al. 2008): post-process the heightfield with a real water-flow + sediment-transport sim. Produces drainage networks, alluvial fans, and deposition patterns that no static noise function can match. Star Citizen Planet Tech v5 explicitly uses erosion simulation to drive rock and debris distribution.
- **Plate-tectonic preprocess** (Tectonics.js, PlaTec, World Orogen): generate the macro shape from a plate sim, then apply erosion + noise on top.

**Anisotropy.** Real-world detail has *direction*. Glacial valleys point downhill; dunes align with wind; fault scarps follow stress fields. If your terrain has no preferred directions anywhere, it looks fake. Even a small amount of axis-aligned anisotropy (rotated noise basis, directional warping) sells the planet.

> **Do:** Build a frequency budget. Decide which process owns which band. Use different noise types for different bands. Add anisotropy where geology demands it. Run an erosion pass on the heightfield, even if cheap.
> **Don't:** Treat fBm as the universal terrain generator. Don't let one noise function carry the whole frequency spectrum. Don't make every scale look like every other scale.

---

## 5. Atmosphere and Clouds (Orbital Aesthetics)

Atmosphere is where you get the *most aesthetic value per line of code* on a planet from orbit. A mediocre terrain with a great atmosphere reads as a great planet; a great terrain with a hard terminator reads as a textured ball.

**Limb glow.** As you look toward the planet's edge, your line of sight passes through a long atmospheric path. For Earth, this scatters strongly in blue (Rayleigh), producing the iconic glowing edge. The Sean O'Neil GPU Gems 2 model and Bruneton's 2008 *Precomputed Atmospheric Scattering* both nail this. The aesthetic key: the limb glow extends *beyond* the geometric edge of the planet, producing a soft halo. If your atmosphere stops at the geometric horizon, it looks pasted-on.

**Terminator softness.** A real atmosphere bends sunlight (~half a degree on Earth) and scatters it past the geometric terminator. This produces:
- A fuzzy day/night transition, not a knife-edge.
- A reddish/orange glow on the night-side near the terminator (twilight wedge).
- Backlit clouds that catch sunlight after the surface beneath them is in shadow.
Without these, the planet looks like a Lambertian ball with a hard terminator — instantly fake.

**Day/night boundary specifics:**
- **Earth-like:** Blue limb, soft red-orange terminator wedge, city lights on the night side.
- **Mars-like:** Pinkish limb (Mie scattering off dust), softer terminator, no city lights, dust-storm edges visible.
- **Venus-like:** Thick yellow-white haze, very fuzzy terminator, no surface visible.
- **Airless (Moon, Mercury, Iapetus):** Knife-edge terminator. *This is correct.* The hardness sells the airlessness.

**Cloud realism (orbital scale):**

- **Latitude bands.** ITCZ near equator (white, towering), Hadley descending zones (cloud-poor, where deserts form), mid-latitude westerlies (frontal cloud streaks), polar cloud caps. Without latitude structure, clouds look like noise.
- **Cyclones / hurricanes.** Spirals at ~10–30° latitude in summer hemispheres. Cassini's Saturn polar storm and Earth's hurricane imagery are the references. A "real-looking" cloud layer has *structure at multiple scales* — mesoscale fronts, synoptic spirals, cumulus clusters.
- **Transparency and shadow casting.** Clouds *must* cast shadows on the surface. Without shadows, clouds float as detached layers. Even a cheap projected-shadow approximation is enormously better than nothing.
- **Cloud parallax.** When the camera moves, clouds at altitude H should shift relative to the surface. This single effect sells "thick atmosphere" more than any shader.
- **Volumetric vs. impostor.** For orbital views, a 2D cloud impostor textured onto a slightly-larger-than-planet sphere is fine — and is what most planet-scale games do. Volumetric (Guerrilla's Nubis, Frostbite's physically-based sky) only matters once you're flying *through* the cloud layer.
- **Gas giants:** Bands are *latitude-correlated* (zonal jets), sometimes with the Saturn-hexagon-style polar structures. Within bands you get vortices (Great Red Spot), barges, and turbulent shear lines at band boundaries. Eichstädt/Doran's JunoCam processing is the visual benchmark.

**Aurora at poles.** Aesthetic-only feature for life-bearing worlds with magnetic fields; SpaceEngine explicitly notes "do not create aurora on airless bodies" as a realism guideline. Aurora glow should be at the *top* of the atmosphere, ringed at high latitudes, color-matching the dominant atmospheric gas (green/red for O₂, pink/violet for N₂).

**City lights on the night side.** The "Black Marble" / Suomi NPP imagery is the reference. Lights cluster along coastlines, rivers, and major roads — *not* uniformly. A planet with even-density night-side lights looks fake. Lights also reveal civilization patterns (a network connecting cities reads as roads).

**Scale height appearance.** Earth's atmosphere is visually thin from ISS (~the limb-glow is a thin band ~100 km wide on a 12,742 km diameter planet — about 1.5% the planet's radius). Make it too thick and the planet looks "encrusted"; too thin and it disappears. Get the ratio right and it sells the scale.

> **Do:** Implement Bruneton-style precomputed scattering. Make the limb glow extend beyond the planet's geometric edge. Use latitude-banded cloud noise. Cast cloud shadows on the surface. Distinguish Rayleigh (Earth) from Mie (Mars/Titan) palettes.
> **Don't:** Use a flat-color "atmosphere shell" with a fresnel falloff. Don't paint clouds as a detail texture without shadows. Don't make airless bodies have soft terminators.

---

## 6. Lighting and Shading

**The terminator is where lighting tells the truth.** At full phase, almost any shader looks OK. At crescent phase, errors are obvious.

**Why phong-only shading looks fake.**
- No subsurface scattering on ice/snow.
- No anisotropic reflection on water (specular blob, not specular streak).
- No fresnel-correct ocean (oceans get *more* reflective at grazing angles — that's why sun-glint is spectacular at low sun angles).
- No multiple scattering in atmosphere — dark side stays *truly black*, when it should pick up scattered earthshine.

**Self-shadowing on terrain.** Critical near the terminator. Long mountain shadows are the strongest scale cue at low sun angles. The bright peak / dark valley alternation tells you the terrain has relief. Without self-shadowing, mountains near the terminator look flat. Note that NASA uses terminator-line imagery to *measure* topography precisely because the shadows reveal heights.

**Ambient occlusion in craters and valleys.** Crater interiors, deep canyons (Valles Marineris from orbit is a great reference), and steep valleys should be darker than their surroundings. AO is critical for making airless body surfaces read as 3D rather than 2D-textured.

**Sub-pixel mountain shadows from orbit.** When a mountain is smaller than a pixel, its shadow can still be visible in the surrounding pixels (especially at low sun angles). This is partly a normal map / pre-filtered detail problem. If you discard sub-pixel detail entirely, terrain near the terminator looks suspiciously smooth. Outerra's blog discusses this directly: their fractal terrain uses analytical normals (per Quilez derivative-noise) to keep micro-relief alive even at orbital distances.

**Specular ocean glints.** As above. The classical look is a brilliant spot at the sub-solar point that *spreads into a glint streak* due to wave roughness. Robinson, Meadows & Crisp 2010 and McCullough 2006 both discuss the polarization signature and brightness. From a games perspective: model ocean roughness, do a proper microfacet specular, and you get cinematic results for free.

**Ice/snow BRDF.** Anisotropic, sub-surface, with strong forward-scattering. Antarctica from orbit is bluish-white and *not* fully Lambertian — there's a noticeable hot-spot opposition surge at low phase angles. Snow on slopes should be brighter than snow on flats (because slopes face the sun more directly when oriented correctly), and snow accumulates preferentially on flat or pole-facing surfaces (slope-and-aspect-driven).

**Low-angle terminator light.** Sunset-color (long-wavelength) light grazes the surface at the terminator. Mountain peaks catch it while their bases are still in shadow. This is *aesthetically* the most cinematic light in any planet rendering. Get it right and your screenshots look like Cassini photos.

**Earthshine / planetshine.** The dark side of Earth's Moon is faintly visible because Earth illuminates it. For game purposes this can be approximated as a low-intensity ambient term proportional to the host planet's albedo and apparent area — it's a small effect but it reads as "physically modeled."

> **Do:** Use proper PBR for ocean and ice. Self-shadow the terrain. Add ambient occlusion to craters and valleys. Make terminator light warm and grazing. Implement at least single-scattered atmospheric multiple-scattering for night-side ambient.
> **Don't:** Use a single Lambert + Phong term and call it shading. Don't let the night side go pure black if there's an atmosphere.

---

## 7. Believability vs. Realism (Designing Alien Planets)

The user's brief notes Pyros System (Thalos, Mira) — these will be alien planets, not Earth-clones. The aesthetic goal is **believability, not Earth-likeness.**

**Internal consistency is the keystone.** A planet whose features all tell the same story will read as real even if it's bizarre. A planet with one inconsistent feature reads as fake even if everything else is photoreal. Things that should agree:

- **Axial tilt ↔ ice cap shape & seasonal extent.** High tilt (Uranus, ~98°) gives extreme polar cycles. Low tilt (Mercury, ~0°) gives stable polar shadow zones with permanent ice. Earth's ~23.5° gives moderate seasons and roughly hemispherical caps.
- **Rotation rate ↔ atmospheric circulation.** Fast rotators (Jupiter, ~10h) get many narrow zonal bands; slow rotators (Venus, 243 days) get nearly hemispheric Hadley cells.
- **Insolation ↔ biome distribution.** Hot zones near the equator (or sub-stellar point on tidally locked worlds), cold at poles. Habitable zones are bands, not blobs. If you have lush jungle at the poles for no reason, the planet is broken.
- **Tectonic activity ↔ geology.** Active world (Earth, Io): mountain belts, volcanism, fresh terrain, few craters. Dead world (Moon, Callisto): saturated cratering, ancient relief, no tectonics. Pluto sits between these — surprisingly geologically active given its size, with Sputnik Planitia's convection cells at ~33 km scale.
- **Atmospheric composition ↔ color and weather.** CH₄/N₂ atmospheres (Titan): orange haze. CO₂ atmospheres (Mars, Venus): pinkish/yellowish dust scattering. Clear-O₂/N₂ atmospheres (Earth): blue Rayleigh-dominant.

**Restraint with exotic colors.** Sci-fi cliché: purple skies because alien. Real reason a sky might be a non-Earth color: different atmospheric composition + different stellar spectrum. Tatooine-style binary systems are fine; a purple sky on a G-star-orbiting nitrogen-atmosphere world is just bad. *Justify the color choices through composition.*

**Avoiding sci-fi clichés:**
- **Glowing crystals everywhere:** unjustified unless there's a thermodynamic reason (radioactive, bioluminescent, recently impacted).
- **Neon biomes:** real bioluminescence is dim and patchy. Avatar's Pandora reads as fantasy because of this.
- **Ringed terrestrial planets:** rings need a Roche-limit-distance moon disruption — possible but narratively heavy. If you give a planet rings, *commit* to them (shadow-banding on the surface, ring-plane crossing visible from the surface).
- **Multi-colored "stripes" of biomes around a tidally-locked planet:** real tidally-locked worlds would have a sub-stellar hot zone, terminator twilight ring, and frozen far side — the bands are *circular concentric*, not linear.

**The role of "weathered" detail.** A planet that looks too pristine reads as "freshly generated." Real planetary surfaces show the *evidence of time*:
- Crater rims softened by regolith and small-impact gardening.
- Drainage channels filled with sediment.
- Lava flows in various stages of weathering (fresh/black to old/red).
- Ejecta rays fading from bright to dark (Tycho-fresh vs. ancient).

Iapetus is the canonical "weathered detail" reference: its leading-hemisphere darkening from accumulated dust is *exactly* the kind of process-trace detail that makes a moon feel ancient and lived-in.

**Geological history readable in landscape.** The best alien planets in fiction (e.g., Solaris, Annihilation, Arrival's Heptapod world) have surfaces that *hint at past events* — old impact basins, dry seabeds, tectonic suture zones. Even abstract alien planets benefit: a dichotomy between northern lowlands (resurfaced young) and southern highlands (ancient crater-saturated) — Mars's signature feature — gives instant geological history.

> **Do:** Pick a planet's "story" first (axial tilt, rotation, age, atmosphere, life-bearing or not). Then derive features from the story. Use restraint with exotic flair — one weird thing is a feature, three weird things is a mess.
> **Don't:** Pick "alien-looking" colors and features arbitrarily. Don't make every planet equally weird. Don't violate insolation/climate logic for visual interest.

---

## 8. Scale Cues and "Feeling of a Planet"

This is what separates "a sphere with terrain" from "a *world*." Several cues stack to produce the sense of vast scale.

**Atmospheric perspective.** Distant terrain fades toward the atmospheric haze color. From orbit, this is subtle but always present — the limb of a continent is bluer/hazier than its center. Outerra's blog notes that without proper distance haze, the world feels small. Bruneton's aerial perspective term handles this.

**Cloud parallax.** Clouds at altitude move differently from the surface as the camera translates. If clouds and surface move together, the cloud layer reads as a decal. If they parallax correctly, the atmosphere has *thickness*.

**Curvature of the horizon.** From low orbit (~400 km on Earth), the horizon is visibly curved. From high orbit, the planet is small in the field of view but its terminator is a clear arc. The horizon curvature radius is a *direct* scale cue — the eye reads it instinctively. Don't break it with FOV tricks.

**Proportional feature sizing.** This is where amateur procedural planets fail catastrophically.
- A "mountain" 1 km wide and 100 m tall, when seen from 400 km up, is sub-pixel. It should not be a major visible feature.
- A continent 5,000 km across should fill most of a hemispheric view from 1,000 km up.
- Rivers should not be visible from orbit unless they're Amazon/Nile-scale (10s of km wide for the largest, hundreds of m wide for the rest — invisible above ~30 km altitude).
- Cities are barely visible by day even from low orbit; they're more visible as light clusters at night.

If your "mountains" are visible as wrinkles from orbit, they're either *enormous* (Olympus Mons-scale, 600 km across) or your scaling is off.

**Relative scale between features.** Rivers are tiny vs. mountains; mountains are tiny vs. continents; continents are tiny vs. the planet. Get the ratios right or the eye reads the planet as toy-sized.

**Cloud-to-surface interaction.** Clouds floating *above* the terrain (with parallax and shadow) sells altitude. Outerra and Star Citizen Planet Tech v4/v5 both emphasize cloud-light interaction with terrain (volumetric shadows from cloud layers).

> **Do:** Test your planet at multiple altitudes. Verify that mountains read appropriately at 100 km, 500 km, and 5,000 km. Implement aerial perspective. Make sure clouds and surface parallax independently.
> **Don't:** Tune feature scales by what looks good at one altitude. Don't make sub-orbital features visible from orbit unless they're genuinely planet-scale.

---

## 9. Transitions Between Scales (Orbit → Surface)

This is the technical heart of the Outerra/Star Citizen quality bar. Aesthetically, the goal is: **the planet should look right at every distance, with no surprise reveals.**

**What looks bad in transitions:**

- **Pop-in.** A feature suddenly appearing as you descend. Mitigated by fade-in or by ensuring a coarse representation is always present that gradually refines. Outerra explicitly mentions LOD and fractal refinement at every scale; the "ground texture is visible from outer space" was a hard-won feature in Star Citizen Planet Tech v4 (CitizenCon 2949).
- **Detail emerging too suddenly.** Watching procedural noise "boil into existence" as you zoom in is a tell. Better: slowly amplitude-fade higher octaves so detail *thickens* rather than *appears*.
- **Knife-edge biome boundaries on close approach.** A boundary that looked fine from orbit becomes a hard line on the surface. Solution: scale-dependent dithering or interdigitation pattern.
- **Mountains that look majestic from orbit but are flat polygons up close.** Two failure modes: (a) coarse heightmap with no fine detail, (b) detail noise that doesn't match the macro feature's geological character. A volcano should have volcanic-rock detail; a sedimentary mountain should have stratification.
- **Color/material disagreement between scales.** Texture from orbit shows green forest; landing on the surface shows beige sand. This is the deepest tell — the macro and micro data are decoupled. Star Citizen v5's planet shader explicitly *evaluates required terrain color when viewed at a distance — taking into account the distribution of assets that are not loaded in*, eliminating this mismatch.

**Aesthetic principles for ensuring features look right at every distance:**

- **Erosion patterns visible at every scale.** Mountains should have visible drainage from orbit (dark threads down their flanks) AND visible drainage on the surface (actual stream channels in the ground). Both should be the *same drainage network*, not two unrelated noises.
- **Self-similar features should match.** If from orbit a mountain has a particular ridge-line direction, that direction should still be visible up close. The macro silhouette should constrain the micro silhouette.
- **Smooth amplitude/frequency transitions in noise.** Adding octaves gradually as the camera approaches, rather than all-at-once. Outerra's bicubic-then-fractal subdivision is a clean version of this.
- **Distance-coherent biome blending.** Blending between biomes should follow the same noise-driven rule at every scale — the boundary on the surface should be the same shape as the boundary from orbit, just at a finer resolution.
- **Asset color baking into the terrain shader.** Star Citizen v5 explicitly bakes tree/grass color into the terrain shader so distant terrain has the right tint without rendering individual assets. This eliminates the "bright green from orbit, brown ground up close" inconsistency.

> **Do:** Make orbit and surface views share data and color. Make features look like the same feature at any zoom. Fade detail in gradually with altitude.
> **Don't:** Hand-author orbital colors separately from surface materials. Don't stop your noise frequency cascade at a specific LOD; it should keep going.

---

## 10. What Separates Top-Tier from Amateur

Concrete aesthetic differences between Outerra/Star Citizen/Space Engine and Shadertoy-class procedural planets:

| Top-tier | Amateur |
|---|---|
| Texture detail consistent from orbit to ground level (no orbital "flat painted texture" reveal) | Orbital view shows clean shading; close-up shows tiled noise |
| Biome blending uses 2D climate data (temperature × humidity, soil, slope) | Biomes are altitude-thresholded bands |
| Rivers actually follow terrain (lowest path) | Rivers are noise-textured strips that don't track topography |
| Snow accumulates on slopes correctly (slope + aspect + altitude + temperature) | Snow is a flat altitude threshold |
| Sky scattering is physically based (Bruneton, O'Neil) | Sky is a fresnel-shell with a flat color |
| Water rendering uses microfacet specular + foam at coastlines + depth-darkened color | Water is a flat blue plane |
| Clouds interact with terrain (cast shadows, terrain-aware coverage) | Clouds are a separate sphere that shows through mountains |
| Crater morphology varies with age (fresh→bright + rays + sharp rim; old→dark + softened + pit-bottomed) | All craters are identical |
| Color and saturation are restrained, sampled from real planetary photography | Saturated greens, cyans, and oranges; Photoshop-y |
| Macro composition follows tectonic logic (continuous belts, basin/range) | Continents are blobs; mountains are scattered peaks |
| Detail in terrain and texture analytic-derivative-coherent (Quilez "more noise") | Noise computed naively with finite differences for normals; aliasing |

**Concrete examples by reference:**

- **Outerra:** "Linear-space lighting + fractal mixing of three textures (grass/daisies/lighter-grass)" eliminated visible texture tiling — a single technique, huge aesthetic payoff. Shadow blurring + correct sRGB pipeline gave them the "almost real" look users frequently mistake for photographs.
- **Star Citizen Planet Tech v5 + Genesis:** Uses temperature, humidity, geology, soil type, soil depth, nutrients, sunlight exposure, slope aspect to drive *emergent* biomes. Each Flora competes for placement based on local conditions. Trees are tinted and shed based on seasons. Eliminates the "biome-of-the-week looks pasted on" problem.
- **Space Engine:** Uses biome presets keyed to planet class and host star spectrum (so a G-star world's vegetation differs from a K-star world's). Procedural color variation within each biome. Two-scale soil materials with PBR.
- **Infinity (Ysaneya):** Procedural ocean color as a function of depth (deep blue → coastal green). Procedural ring textures with multi-pass band compositing. Shadowing of rings from the planet.
- **Sebastian Lague's Coding Adventure: Procedural Moons and Planets** (Outer Wilds-inspired): A great reference for the *minimum viable* procedural planet aesthetic — restraint, clear silhouette, and good lighting can carry a low-detail planet aesthetically. His videos are a good calibration of what a small team can achieve.
- **Hello Games / No Man's Sky:** Grant Duncan's GDC talk on procedural art emphasizes *rule-based* generation rather than randomness. Templates with constraints, color palettes inspired by specific reference (70s sci-fi covers), and consistency *within* each generated planet are what makes their universe cohere. NMS deliberately stylizes; the lesson generalizes to any aesthetic.

---

## 11. Specific Anti-Patterns to Avoid

A consolidated checklist:

- **Flat oceans.** Oceans need: depth-darkened base color, microfacet specular, sun-glint streak at the sub-solar point, foam at coastlines (and breaking waves where shallow), fresnel reflectivity that increases at grazing angles. A flat-color ocean is the single biggest "this is fake" tell on a habitable planet.
- **Voronoi-blob biomes.** Cells visible from orbit. Solution: blend noisy boundaries with smaller-scale interdigitation; drive biomes from continuous climate fields, not discrete cells.
- **Visibly tiling textures.** The ground texture repeats every 100 m, visible from any altitude where you see more than one tile. Solutions: fractal mixing (Outerra), stochastic texture variation (Heitz/Neyret), unrepetitive blending across multiple scales (Star Citizen v5).
- **Candy-colored planets.** Saturated, primary-color palettes read as cartoon. Even "stylized" planets benefit from some hue variation and desaturation.
- **Unrealistic ring structures.** Rings should: be inside the Roche limit; have gap/band structure (Cassini Division-style); cast a shadow on the planet; receive a shadow from the planet on the night side. Saturn from Cassini is the reference.
- **Perfectly spherical fast rotators.** Use an oblate spheroid with appropriate flattening (Earth: f≈1/298, Saturn: f≈1/10).
- **Uniform crater distribution on airless bodies.** Real airless bodies have crater density variation: ancient highlands are saturated, young plains are sparsely cratered. The Moon's mare/highland dichotomy is the reference.
- **Missing ejecta rays on fresh craters.** A fresh crater (Tycho, Copernicus, Giordano Bruno) has a brilliant ray system extending many crater diameters out. The rays fade over millions of years via space weathering. Without rays, fresh craters look like old ones.
- **Lack of regolith softening on old craters.** Old craters are rounded, partly buried, and may have flat floors from sediment infill. If all craters have crisp rims, none of them look old.
- **Hard atmosphere shells.** A flat-blue ring around the planet without proper Rayleigh/Mie scattering. The blue should be *strongest* at the limb and *fade* through the disk.
- **Unscaled feature distribution.** Mountains every 50 km because that's the noise frequency, regardless of tectonic logic. Real mountains are *concentrated along plate boundaries*, with vast flat plains in between.
- **Ignoring the lit/unlit boundary aesthetics.** A planet rendered only as fully lit looks decorative, not real. Show the terminator. Show twilight glow. Show city lights.
- **No oblique-impact crater asymmetry.** Real impact craters from oblique impacts have asymmetric ejecta blankets and butterfly-shaped or "forbidden zone" patterns (e.g., Proclus on the Moon). Always-circular ejecta is unrealistic.
- **Using the same seed/noise basis for everything.** Visual rhythm forms when different processes use different basis functions. If everything is gradient-noise, the planet has a uniform "feel."
- **Thin-line river textures glued onto terrain.** Rivers must follow gradient. Use flow-accumulation post-processes or streamline integration; don't paint rivers.

---

## 12. Reference Points and Benchmarks

Real planetary imagery to anchor your aesthetic targets:

**Earth:**
- **Blue Marble (Apollo 17, 1972; Blue Marble Next Generation, MODIS, 2002–2004):** The canonical "what a habitable planet looks like from outside." Note the *muted* palette — deep navy oceans, olive-green vegetation, ochre deserts, warm-white clouds, blue limb glow.
- **DSCOVR/EPIC:** Continuous sunlit-disk imagery showing dust plumes, cyclones, and sun-glint over oceans. Reference for transient atmospheric features.
- **ISS time-lapse / "Earth at night":** Reference for terminator wedge, aurora, and city-light distribution.

**Moon:**
- **LRO (Lunar Reconnaissance Orbiter) imagery:** Reference for crater morphology, ray systems (Tycho, Copernicus), mare/highland color contrast, and regolith brightness gradients.
- **Apollo earthrise / surface panoramas:** Reference for airless-body shading — knife-edge terminator, deep shadows in crater interiors, harsh contrast.

**Mars:**
- **ESA Mars Express HRSC global mosaic (2023):** Best true-color reference. Note the muted yellow-orange-grey palette and how mineralogical color variation (basalt sands, sulfates, clays) reads at orbital scale.
- **HiRISE / Mars Reconnaissance Orbiter:** Surface-detail reference for dust devil tracks, dune fields, frost patterns.
- **Valles Marineris from orbit:** Canonical "huge canyon as seen from space" reference.
- **Olympus Mons:** Canonical shield-volcano-bigger-than-a-small-country reference. Note how it has gentle slopes that barely register at orbital distance — *not* a sharp peak.

**Jupiter (and gas giants generally):**
- **JunoCam imagery (especially Eichstädt/Doran processing):** Belts, zones, vortices, polar storms, the Great Red Spot. Note the *subtle* color contrast in true-color processing.
- **Hubble & Cassini Saturn imagery:** The Saturn hexagon (Wikipedia and Cassini observations) is the canonical "polar circulation creates geometric patterns" reference. Useful if you want striking gas-giant features that have a basis in real fluid dynamics.
- **Voyager Jupiter:** First-pass, high-contrast color is what most people remember; better as inspiration than reference.

**Saturn moons (Cassini):**
- **Iapetus:** Two-tone hemisphere dichotomy and the equatorial ridge — canonical "weird moon" reference. The dichotomy is darkening *only* on the leading hemisphere from accumulated debris; this is the kind of internally-consistent oddity that reads as real.
- **Mimas:** Herschel crater giving the "Death Star" silhouette — reference for impact-dominated icy bodies.
- **Enceladus:** Tiger-stripe fissures with cryovolcanic plumes — reference for active icy moon aesthetics.
- **Titan:** Orange haze enveloping a methane-hydrocarbon surface; specular sun-glint on Kraken Mare. Reference for thick-atmosphere moons and non-water specular surfaces.

**Pluto (New Horizons):**
- **Tombaugh Regio (the "heart"):** Sputnik Planitia's nitrogen ice convection cells (~33 km) bordered by water-ice mountains (al-Idrisi Montes). Reference for non-impact tectonics on small icy bodies.
- **Cthulhu Regio:** Dark red tholin terrain. Reference for organic-aerosol surface deposits.
- **Pluto color mosaic (S.A. Stern et al.):** Reference for low-saturation but high-hue-variation palettes — pales blues, yellows, oranges, deep reds all coexist without being garish.

**Exoplanet renders:**
- **SETI/STScI artist concepts (Robert Hurt et al.):** Stylized but science-anchored. Useful for showing how artists choose to depict planets we don't have direct imagery of.
- **Seán Doran's Twitter feed:** A practitioner who processes JunoCam, Cassini, and synthetic data into beautiful, restrained renders. Great calibration for "what does a real planet *actually* look like."

---

## A Pyros-Specific Cheat Sheet (Thalos & Mira)

A condensed checklist as you build planets:

**Macro layer (do first, hand-author or simulate):**
- [ ] Plate / basin / hotspot map
- [ ] Hemispheric land-water asymmetry (don't be uniform)
- [ ] Continuous mountain belts along convergent boundaries
- [ ] Rift structures along divergent boundaries
- [ ] Coastal interdigitation (peninsulas, islands, bays)

**Mid layer (process-aware noise):**
- [ ] Erosion-aware fBm (Quilez derivative-damped) for terrain
- [ ] Domain-warped noise for fluvial drainage patterns
- [ ] Ridged multifractal for mountain ridges
- [ ] Hydraulic erosion post-pass on the heightfield (even cheap)
- [ ] Glacial signature in high latitudes / high altitudes (cirques, U-valleys)
- [ ] Aeolian dunes in arid zones with directional alignment

**Micro layer (texture and material):**
- [ ] Slope/aspect/altitude-driven material blending (rock on steep, soil on flat, snow on cold-and-flat)
- [ ] Stochastic detail texture variation (no visible tiling)
- [ ] Color baked into terrain shader for distance views (orbit-surface color match)

**Atmosphere:**
- [ ] Bruneton-style precomputed scattering (Rayleigh + Mie)
- [ ] Limb glow extends past geometric edge
- [ ] Soft terminator with twilight wedge
- [ ] Multi-layer clouds with latitude-banded structure
- [ ] Cloud shadows on terrain
- [ ] Aerial perspective for distant features

**Lighting:**
- [ ] Self-shadowing terrain
- [ ] Ambient occlusion in valleys/craters
- [ ] Microfacet ocean with sun-glint streak
- [ ] Anisotropic snow/ice BRDF
- [ ] Earthshine / multi-scattered ambient on night side (if atmosphere)

**Anti-patterns to actively check against:**
- [ ] No flat-color oceans
- [ ] No knife-edge biome boundaries from orbit
- [ ] No saturation > "real Earth" unless deliberate
- [ ] No sub-orbital features visible from orbit (proportional scaling)
- [ ] No identical craters; vary age and degradation
- [ ] No perfect spheres on fast rotators
- [ ] No hard terminators on atmospheric bodies
- [ ] No surprise pop-in during orbital descent

**Pyros-specific differentiation (Thalos vs. Mira):**
- Pick a planet "story" before any rendering decisions — axial tilt, rotation, geological age, atmospheric composition, life-bearing or not.
- Pick distinct *dominant* processes for each (e.g., Thalos = young volcanic; Mira = old ice-aged) so they read differently at every scale.
- Restrain "alien flair" to one signature feature per planet — Iapetus has its dichotomy + ridge; Pluto has its heart; Mars has its hemispheric dichotomy + Valles Marineris. Don't pile features.

---

## Caveats

- **Source tiers vary.** Research from peer-reviewed papers (Bruneton 2008, Mei et al. 2007, Kallweit et al. 2017), GDC/SIGGRAPH talks (Schneider/Guerrilla, McKendrick/Hello Games), official NASA/ESA imagery captions, and the Outerra/Inigo Quilez/Ysaneya/Star Citizen Wiki blogs is high-confidence. Some details from secondary sources (game-news sites, fan wikis, third-party tutorials) are more approximate; treat *technique attributions* as directionally correct rather than precise.
- **"Real-time" vs "offline" boundaries are blurring.** Techniques like volumetric clouds, precomputed multi-scattering, and erosion simulation that were offline-only ten years ago are now real-time. The aesthetic principles in this document are stable; the specific implementations to reach them will keep changing — the Bruneton scattering paper is from 2008, Sean O'Neil from 2005, and they're still relevant.
- **Star Citizen Planet Tech v5, Genesis, and "Brave New Worlds" (CitizenCon 2954/2024)** are presented partly as in-development. Some described features (continuous biome variation, planetary-scale terrain features like the Adir obsidian scar, dynamic weather) were demoed and described in talks but were not fully implemented in shipping builds at the time of those announcements. Treat them as direction-of-travel for top-tier procedural planets, not as guaranteed-shipping benchmarks.
- **No Man's Sky's aesthetic is deliberately stylized** (Grant Duncan's 70s sci-fi-cover inspiration). It's not a model for photorealism, but its rule-based generation philosophy is broadly applicable.
- **Color rendering varies by display, sRGB/HDR pipeline, and tone-mapping.** "True color" planet imagery from NASA/ESA is itself a processed product; the ESA Mars HRSC mosaic explicitly notes that contrast was stretched to highlight surface variation. Calibrate against multiple references rather than one.
- **Real planetary surfaces have variability and exceptions** to every rule stated here. Iapetus's two-tone is *not* internally consistent in any local sense — it required exotic explanation (thermal segregation triggered by exogenic dust). The principle "internal consistency" doesn't mean "no surprises"; it means "surprises should be explicable in retrospect."
- **Bevy/wgpu specifics** were out of scope for this aesthetic guide. Most of the techniques here have published shader implementations in GLSL/HLSL; porting to wgsl is mechanical. The Bruneton reference implementation in particular is widely reimplemented; community Bevy ports of Sebastian Lague's planet shader and Sean O'Neil's atmospheric scattering exist.
- **The user's existing patched-conics + gas giant impostor work** assumed; this document focuses on the rocky/terrestrial planet pipeline (Thalos, Mira). Gas giant aesthetics (banded zonal flow, vortex structures, polar hexagons, ring shadowing) are touched on in §5 and §12 but warrant their own deeper treatment if the system contains additional gas giants.