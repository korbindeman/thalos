# Mira — Process Reference

Companion to the surface generator design doc. The architecture doc specifies *how* the pipeline runs; this doc specifies *what each stage in Mira's recipe is physically modeling and visually producing*. It is the reference for tuning, for understanding why a stage exists, and for spotting when an output looks wrong.

Mira's recipe, repeated for convenience:

```
Sphere → Differentiate → Megabasin → Cratering → MareFlood → Regolith → SpaceWeather
```

Mira is a 869 km tidally locked silicate moon, formed from giant-impact ejecta off the homeworld. Its near side faces the homeworld permanently. The visual goal is "instantly readable as a real moon, with a recognizable face from the homeworld and a meaningfully different far side."

---

## Sphere

**Physical model:** the body's reference shape. Mira is in hydrostatic equilibrium and slow-rotating (tidally locked to a 16-day orbit), so it is spherical to well below visual threshold. No oblateness, no ellipsoidal distortion.

**What it sets:** the radius of the body and the coordinate frame for everything downstream. All later stages express their output as offsets from this sphere.

**Parameters:** `radius_m = 869_000`.

**Visual contribution alone:** a featureless smooth sphere. This is what you see if every other stage is a no-op. Useful as a sanity check — if even the sphere doesn't render correctly, no later stage will help.

---

## Differentiate

**Physical model:** the early thermal evolution of the body. As Mira accreted from giant-impact ejecta, the interior heated, partially melted, and separated by density. Heavy iron sank toward the center; light silicates floated up to form a crust. For Mira specifically, the impact origin means the bulk composition is silicate-dominated (most of the iron stayed in the homeworld's core), so the crust is thick anorthositic-style highland material with relatively little iron.

The crust then froze. The top of the magma ocean crystallized first into a buoyant feldspar-rich layer (the highlands). Late-stage residual melts, denser and richer in iron and incompatible elements, sank or were trapped beneath the crust — they would later resurface as mare basalt when impacts cracked the crust open.

**What it sets:** the materials palette for the rest of the pipeline, and the parameter ranges that downstream stages will use. After Differentiate runs, the body has a defined set of named materials that features can reference.

**Materials defined for Mira:**
- *Highland anorthosite*: bright, rough, the primordial crust. The default surface material everywhere maria are absent. Albedo high (~0.12 for old highlands, brighter when fresh).
- *Mare basalt*: dark, smoother than highlands, the late-stage flood material. Only appears where MareFlood places it. Albedo low (~0.07 for old maria, slightly brighter when fresh).
- *Fresh ejecta*: brightest material, the freshly excavated subsurface thrown out by recent impacts. Albedo highest (~0.18). Decays toward highland or mare albedo as space weathering ages it.
- *Mature regolith*: the terminal state of any surface exposed to space for billions of years. Slightly darker than its parent material. Most of Mira is in this state.

**Parameters:** `composition = silicate_dominated`, `iron_fraction = 0.10`, plus per-material albedo ranges and roughness values.

**Visual contribution alone:** still a featureless sphere, but now downstream stages know what to color it with. Think of Differentiate as "loading the paint set" rather than painting anything.

---

## Megabasin

**Physical model:** the very largest impacts in the body's history, the kind that happened during the Late Heavy Bombardment when the inner system was being pummeled by leftover planetesimals. These impacts were big enough to excavate down through the entire crust and leave depressions hundreds of kilometers across that persist for billions of years.

For Mira, two of these dominate. They are not statistical features — they are bespoke ancient scars, hand-placed because they define the body's character. Real-Moon analogues are South Pole-Aitken (the largest basin on the Moon) and the Imbrium basin.

The basins themselves are smooth bowls on the largest scale: a broad depression with gently sloping walls, far too large for any individual crater rim to be visible at this scale. Smaller features (later impacts) are imprinted on top of them by Cratering.

**What it sets:** the largest-scale topography on the body. After Megabasin runs, the cubemap height field has two broad depressions; everything else (cratering, mare flooding) builds on top of this foundation. Megabasin also creates the conditions for the most dramatic mare flooding: the basin floors are the lowest places on the body, so when MareFlood looks for places to fill, the megabasin floors are the obvious targets.

**Tidal asymmetry:** Megabasin's basin centers default to placing both basins on the near side, near the sub-homeworld point. This is a hand-authored worldbuilding choice rather than a physical process — large impacts don't actually care about tidal locking. The reason for the choice is narrative: the player should see Mira's distinctive dark patches every night from the homeworld. If both megabasins were on the far side, Mira would look like a featureless white ball from home. The basin placements are also slightly offset from each other so Mira has visual structure rather than one big spot.

**Parameters:** `count` (default 2 for Mira), `center_dir[]` (the unit vectors pointing at each basin center, defaulting to near-side positions), `radius_km[]` (default ~250 km for the larger basin, ~180 km for the smaller), `depth_km[]` (default ~6 km and ~4 km respectively — these are the depressions relative to the surrounding terrain), and a hemispheric asymmetry bias for additional broad lowering of the near side.

**Visual contribution alone:** the sphere now has two large smooth depressions on the near side. From a distance the body looks like a slightly lumpy ball; you can't see the depressions clearly without lighting from a low angle. This is the first stage that breaks the spherical symmetry.

---

## Cratering

**Physical model:** the cumulative history of impact bombardment over billions of years. Anything orbiting in the system will eventually hit the moons; over geologic time the surface accumulates a population of craters spanning many orders of magnitude in size, from massive basins down to micrometeorite pits.

The size-frequency distribution (SFD) is a power law with cumulative slope around -2: there are roughly 100× as many craters of any given size as craters ten times larger. This means the surface is dominated visually by a few very large craters and statistically by an enormous number of small ones. The transition between "individually distinctive" and "blurs into texture" happens around the size where craters become smaller than the eye can resolve from the current viewing distance.

Crater morphology varies systematically with size:

- *Simple craters* (small): a clean bowl shape. Raised rim, depressed floor, smooth interior. These look the same from any angle. Everything below ~10 km on Mira is a simple crater.
- *Complex craters* (medium): a flat floor, terraced walls (the rim has slumped inward in steps), and a central peak where the rebound from the impact pushed material up from below. The central peak is the diagnostic feature. Roughly 10-100 km on Mira.
- *Peak-ring basins* (large): instead of a single central peak, the rebound forms a ring of peaks inside the crater. The floor between the ring and the outer rim is flat. Roughly 100-200 km.
- *Multi-ring basins* (very large): concentric rings of peaks, faulting in the surrounding terrain, ejecta blankets that extend hundreds of kilometers. These are the largest individual impact features and are essentially unique objects, each one different. The largest few craters on Mira from this stage may be in this category (the megabasins are even larger but were placed by the previous stage).

Every crater also produces an *ejecta blanket*: a ring of debris thrown out radially by the impact, brighter than the surrounding terrain when fresh, fading as space weathering darkens it. The ejecta blanket extends roughly one crater diameter outward from the rim, becoming sparser with distance. For the largest craters, the ejecta forms ray systems — radial streaks visible for many crater diameters away. These rays are how you spot fresh young craters from orbit; they fade over hundreds of millions of years.

Crater ages are distributed roughly uniformly across the body's history, with a slight bias toward older ages (because more time has passed during which old craters could form). The youngest few craters are the ones with bright ejecta and ray systems still visible.

**Tidal asymmetry:** none. Impacts come from random directions and hit both hemispheres roughly equally. The far side of Mira is just as cratered as the near side. (The near side does receive a small enhancement from gravitational focusing by the homeworld, on the order of 10-20%, but this is invisible against random scatter and not modeled.)

**What it sets:** the dominant mid-scale topography of the entire body. After Cratering runs, the body is covered in thousands of impact features. The largest of them are baked into the cubemap (visible at impostor distance); the rest sit in the feature SSBO for the UDLOD renderer to consume later.

**The two-output split:** Cratering does two things with its output. Every crater goes into the feature SSBO, including the smallest. *Additionally*, craters above the cubemap-resolvable size threshold (roughly 10 km radius for Mira's 512² cubemap, which gives ~3.4 km/texel) get rasterized into the cubemap accumulator so they're visible from far away. This means the impostor renders ~hundreds of explicit craters baked into the cubemap, and the UDLOD path will eventually render the full ~10k from the SSBO.

**Parameters:** `total_count` (default 10_000 — tunable, this is the SSBO budget), `sfd_slope` (default −2.0, the power-law exponent), `min_radius_m` (default 1500, the smallest crater stored explicitly — anything below this is handled by Regolith's statistical noise), `max_radius_m` (default 250_000, the largest crater this stage produces — note this is smaller than the megabasins, which are placed separately), `morphology_thresholds` (radii at which simple → complex → peak-ring → multi-ring transitions occur, default ~10 km / ~100 km / ~200 km), `age_distribution` (controls how many young vs. old craters), and `cubemap_bake_threshold_m` (the minimum crater radius that gets rasterized into the cubemap, default ~10_000 for Mira).

**Visual contribution alone:** the body now has a saturated cratered surface. The largest craters are individually visible as distinct features with raised rims, central peaks (on bigger ones), and ejecta blankets. The medium craters fill in between as textural density. The body now reads as "moon" rather than "sphere with depressions." This is the single biggest visual jump in the pipeline.

---

## MareFlood

**Physical model:** late-stage volcanic flooding of the largest impact basins. After the heavy bombardment ended, Mira's interior was still hot enough that pockets of basaltic magma existed beneath the crust. Where impacts had cracked the crust open — particularly the deep megabasin floors and the largest crater bottoms — this magma could find its way to the surface and flood the basin floors with dark basaltic lava.

The lava pooled at the lowest points and spread across the basin floors until it solidified, burying whatever older craters had been there before. The result is the famous mare/highland dichotomy: smooth, dark, low-elevation maria filling the bottoms of old basins, surrounded by bright, rough, heavily-cratered highlands.

Mare boundaries are not perfectly circular. The lava flowed to wherever the floor was low enough and ponded where it could; it left embayments where it flowed around obstructions and ghost features where it incompletely buried older craters. The boundary follows topography modulated by flow patterns, not basin geometry directly.

**Tidal asymmetry:** strong. This is *the* defining tidal asymmetry on the real Moon and on Mira. The reason: on a tidally locked body, the crust is asymmetrically thick — thinner on the near side (the side facing the parent body) and thicker on the far side. The mechanism is that the body's center of mass is offset slightly toward the parent during crust solidification, and the buoyant crust crystallized asymmetrically as a result.

The consequence for mare flooding is dramatic. On the near side, the thinner crust is easier for basaltic magma to penetrate when an impact basin cracks it open — so basins on the near side fill with lava. On the far side, the thicker crust holds the magma in even when basins are excavated, so far-side basins remain empty depressions and never become maria.

The result is Mira's defining visual feature: the near side is dominated by dark mare patches that are visible from the homeworld; the far side is essentially mare-free, a uniformly bright cratered highland surface. Cratering is the same on both sides; only the flooding differs.

**What it sets:** the maria themselves — flat dark fills inside the largest old basins and the biggest old craters on the near side. Buried craters (any crater whose interior is below the fill line) get removed from the SSBO and replaced by smooth fill in the cubemap. The cubemap accumulator's height value in the mare regions is set to the fill level; the material is switched to mare basalt; the albedo drops accordingly.

**Boundary irregularity:** the mare-fill region is not just "where altitude is below threshold." A low-frequency noise is added to the boundary so the maria have organic, irregular outlines instead of perfect topographic contours. Without this they look like flooded craters in a bathtub; with it they look like lava flows.

**Parameters:** `target_count` (how many basins to fill, default ~5-8 for Mira), `fill_fraction` (how high the lava reaches relative to the basin's depth, default ~0.7), `near_side_bias` (default ~0.9, the strength of the near-side preference; 1.0 = exclusively near side, 0 = uniform), `boundary_noise_amplitude` and `boundary_noise_wavelength` (controls the irregular shape of the mare edges).

**Visual contribution alone:** the body now has dark patches on the near side, concentrated in the largest old basins and the biggest old craters. The far side is unaffected. From a distance this is the moment Mira becomes recognizable as a specific moon rather than a generic cratered ball — it now has a *face*. The contrast between dark maria and bright highlands is the single largest source of visual structure in the impostor view.

---

## Regolith

**Physical model:** the surface layer of pulverized rock that accumulates on any airless body over billions of years of micrometeorite bombardment. Every tiny impact churns the topmost few meters of surface material, breaking down rocks into progressively finer particles. The result is a layer of fine dust and small rock fragments — *regolith* — covering essentially the entire body.

Regolith has several effects relevant to the surface generator. It smooths small-scale roughness (sharp rock edges get worn down). It produces a slight downhill creep over geologic time (mass wasting), softening crater rims and slopes. And it represents the population of *very small craters* — the micro-impacts that are too numerous and too small to track individually but that collectively give the surface its fine-grained texture.

Statistically, the small-crater population continues the same SFD as the larger craters: tens of millions of craters between 10 m and 1 km on Mira, billions below that. None of them are individually significant, but together they set the surface's texture.

**What it sets:** the high-frequency detail layer, which fills in everything below the explicit-crater threshold. It does not contribute to the cubemap (those frequencies are below the cubemap's resolution anyway). It is consumed only by the UDLOD renderer at close range.

The detail noise should be *crater-shaped*, not generic fBm. A surface covered in small craters looks specifically like small craters — circular depressions clustered at all scales — and it doesn't look like sand dunes or generic Perlin noise. The Regolith stage configures the detail noise parameters to mimic the SFD of the explicit craters at smaller sizes, so the visual handoff between explicit features and noise is seamless.

**Parameters:** `amplitude_m` (the typical depth of small-crater texture, default a few meters), `characteristic_wavelength_m` (the approximate spacing of the smallest visible crater-like features), `crater_density_multiplier` (how dense the small craters are relative to a continued SFD from the explicit cratering stage). These set up the parameters that the sampler will use at close range.

**Visual contribution at impostor distance:** none. Regolith operates entirely below the cubemap resolution. For milestone B (impostor only), this stage exists in the recipe but produces no visible effect. It is included so the recipe is complete and so its parameters are wired up for when UDLOD comes online.

**Visual contribution at close range (future UDLOD):** the surface texture between explicit craters. Without Regolith, close-up surface looks plasticky-smooth between named craters. With it, the surface has the salt-and-pepper density of real lunar regolith.

---

## SpaceWeather

**Physical model:** the slow alteration of any exposed surface on an airless body by the space environment. Solar wind ions, cosmic rays, micrometeorite impacts, and ultraviolet radiation all act on exposed material over time. The cumulative effect on silicate surfaces is to *darken* them — fresh anorthositic crust is brighter than old anorthositic crust by a meaningful factor — and to slightly *redden* them spectrally.

The relevant timescale for visual weathering is hundreds of millions of years. Any surface that's been exposed for ~1 Gyr has reached its mature albedo; freshly exposed surfaces (recent crater interiors, fresh ejecta) are noticeably brighter, fading toward maturity over time.

The most dramatic visual consequence is the *ray system* around fresh young craters. When a crater forms, it ejects material from below the weathered surface layer — material that has never been exposed to space and is therefore at its maximum brightness. This bright ejecta fans out radially in long streaks (rays) visible for many crater diameters from the impact point. As space weathering darkens the ejecta over hundreds of millions of years, the rays fade. On any moon, only the very youngest few craters still have visible ray systems; the vast majority of craters have lost their rays to weathering.

There's a secondary effect on mare albedo too. Fresh mare basalt is darker than fresh anorthosite, but old mare basalt has reached a similar mature dark albedo. The contrast between mare and highland is a contrast between two materials that have both reached their long-term equilibrium with the space environment.

**What it sets:** the albedo cubemap. SpaceWeather is the only stage in Mira's recipe that directly authors the albedo channel rather than the height channel. It reads the materials and ages already established by previous stages and writes brightness values into the albedo accumulator accordingly.

**The three things SpaceWeather does:**

1. *Maturity darkening.* Every region of the surface gets its base albedo reduced by an amount that depends on its exposure age. Most of the body has been exposed for billions of years and reaches its mature albedo. The exception is the youngest features.

2. *Fresh-crater brightening.* The youngest craters in the SSBO (those with `age` below a recency threshold) get their interiors and immediate surroundings boosted toward fresh-ejecta brightness. The brightening decays with distance from the crater rim and with the crater's age, so a slightly older young crater is dimmer than a brand-new one.

3. *Ray systems on the very youngest few craters.* For craters younger than a stricter threshold (the top few percent), SpaceWeather rasterizes radial ray patterns extending several crater radii outward from the rim. The rays are brighter than the surrounding terrain, oriented randomly per crater (real ray systems are roughly radially symmetric but with strong directional variations from the impact angle), and decay both radially and with the crater's age. Only one or two craters on Mira need rays for the body to read correctly — Tycho is enough on the real Moon. Don't try to give every crater rays.

**Tidal asymmetry:** none directly. Space weathering acts equally everywhere. However, since maria are concentrated on the near side and SpaceWeather gives maria their final dark color, the visual effect of weathering reinforces the existing near/far asymmetry. The near side weathers into "dark maria + bright highlands"; the far side weathers into "uniformly bright highlands with darker spots only inside young craters."

**Parameters:** `weathering_rate` (how fast the maturity darkening proceeds with age), `young_crater_age_threshold` (which craters get the freshness brightening), `ray_age_threshold` (which much smaller subset gets ray systems — should be very few), `ray_extent_radii` (how far rays extend from the rim, in crater radii, default ~5-10), `ray_count_per_crater` (default ~6-12), `highland_mature_albedo`, `highland_fresh_albedo`, `mare_mature_albedo`, `mare_fresh_albedo`.

**Visual contribution:** Mira goes from "two-tone gray ball with dark patches" to "real moon photograph." The tonal variety from maturity darkening gives the highlands depth; one or two ray systems on the near side give the eye a focal point and make the body feel like a specific place rather than a generic textured sphere. This stage has the highest visual-payoff-to-code-complexity ratio of any stage in the recipe.

---

## How the stages combine visually

A useful way to think about each stage's contribution is to ask "what does Mira lose if this stage is turned off?"

- **No Sphere**: nothing renders.
- **No Differentiate**: stages downstream don't know what materials to use; everything is undefined.
- **No Megabasin**: Mira loses its largest topographic features and its narrative recognizability from the homeworld. Maria still form in the largest random craters, but they're smaller and less dramatic.
- **No Cratering**: Mira becomes a smooth ball with just the megabasins. Reads as "alien planet" rather than "moon." This is the biggest single loss.
- **No MareFlood**: Mira is uniformly bright and cratered everywhere. The near side and far side look identical. Recognizable as a moon, but not as *this* moon. Loses its face.
- **No Regolith**: no effect at impostor distance. Loses surface detail at close range (UDLOD).
- **No SpaceWeather**: the body looks like a clay model. Two flat tones (highland and mare), no tonal variety, no ray systems, no sense of age. Recognizably a moon but uncanny.

The stages are not equal in visual contribution. From most to least visible at impostor distance: **MareFlood ≈ Cratering > SpaceWeather > Megabasin >> Differentiate ≈ Sphere >> Regolith**. Spend tuning effort accordingly.

## What "looks right" means for Mira

The validation criterion is "does this read as a real moon to someone who's looked at photos of the real Moon." Specific things to check:

- The near side is dominated by dark patches; the far side is uniformly bright. The two faces are obviously different.
- The dark patches are inside basin-shaped regions, not arbitrary blobs. Their boundaries are irregular but their general shape follows topography.
- The highlands (everything that isn't mare) are densely cratered, not smooth. You can see crater density at multiple scales — a few big craters, many medium craters, texture from small ones.
- A small number of bright young craters with ray systems are visible. Not many — one or two is correct, ten is too many.
- Crater morphology varies with size: small ones are bowls, medium ones have central peaks, big ones have peak rings.
- The terminator (day/night line) shows visible relief — crater rims cast long shadows when lit from a low angle.

If all of those are true, Mira is done. If any of them are wrong, the relevant stage is the place to look first.
