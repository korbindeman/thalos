# Procedural generation of realistic rocky moon surfaces

**The single most important factor separating convincing procedural moons from ugly ones is crater morphology fidelity** — specifically, getting the radial profile right for each size class, layering craters across a realistic size-frequency distribution with age-dependent degradation, and including ejecta blankets and ray systems that make craters look *impacted* rather than *stamped*. This report synthesizes planetary science literature, game-engine implementations (Elite Dangerous, I-Novae Infinity, Outerra), and procedural generation research to provide actionable technique references for each of your seven pipeline stages. Every technique is tagged as bake-appropriate (fidelity) or realtime-appropriate (performance), and common failure modes are called out explicitly.

---

## 1. Crater profile functions: the core of visual quality

This is where previous implementations most commonly fail. A convincing crater is not a simple inverted cone — it is a piecewise radial function with distinct zones that change character depending on diameter.

### The piecewise radial profile

The standard approach, descended from Musgrave's *Texturing and Modeling: A Procedural Approach* (1998) and refined by subsequent implementations, defines crater elevation as a function of normalized radial distance **t = r/R** where R is the rim-crest radius. The profile has five zones:

**Central peak** (complex craters only, t < t_peak): A parabolic or conical mound of uplifted bedrock at the crater center. Peak diameter scales linearly as **D_peak = 0.168 × D** (Krüger et al. 2018, from 5,505 pristine craters). Peak height follows a power law: **h_peak = 0.0049 × D^1.297** (in km, D in km). For a 50 km complex crater, this gives a peak ~8.4 km wide and ~520 m high. Peaks are rugged and irregular — never model them as smooth cones; add noise perturbation to break symmetry.

**Flat floor** (complex craters, t_peak < t < t_floor): A roughly level surface of impact melt and breccia. Floor diameter scales as **D_floor = 0.187 × D^1.249** (Pike 1977). Simple craters lack this zone entirely.

**Bowl/cavity wall** (simple craters: 0 < t < 1.0; complex craters: t_floor < t < 1.0): For simple craters, a parabolic bowl **h(t) = −depth × (1 − t²)** works well. For complex craters, the wall contains **terraces** — concentric stepped benches formed by gravitational slumping of normal-fault blocks. Model 2–4 steps of decreasing width from rim to floor, with noise-perturbed positions and heights. The wall transitions smoothly to the rim via a steep scarp.

**Rim crest** (t ≈ 1.0): A sharp topographic high. Rim height scales as **h_rim = 0.036 × D** for simple craters, **h_rim = 0.236 × D^0.399** for complex craters (Pike). Model as a narrow Gaussian ridge: **h(t) = h_rim × exp(−(t−1)² / 2σ²)** where σ controls sharpness. Fresh craters have tight σ; degraded craters have broad, softened σ.

**Ejecta blanket** (t > 1.0): The single most neglected zone in procedural craters, and a primary cause of the "ugly crater" problem. The canonical ejecta thickness equation (McGetchin, Settle & Head 1973) is: **thickness = 0.14 × R^0.74 × (r/R)^(−3.0)**. About **50% of ejecta volume falls within 1R of the rim** (2R from center), and ~90% within 5R. The continuous ejecta blanket extends to ~2–3 crater radii; beyond that, ejecta becomes patchy and discontinuous (secondary crater chains and rays). This zone must be additive — ejecta raises the surrounding terrain, it does not replace it.

**Bake vs. realtime**: The full piecewise profile with all five zones is cheap enough for realtime evaluation (it's a handful of branches and math ops per sample). Both the I-Novae Infinity engine and Outerra evaluate crater profiles analytically in shaders. For the bake pass, you can afford additional refinements like noise-modulated ejecta thickness and per-terrace geometry.

### Morphology class transitions

The profile function must switch behavior at physically-motivated diameter thresholds. On the Moon (gravity-dependent, so scale to your **~870 km radius** body accordingly):

| Class | Diameter | d/D ratio | Key features |
|---|---|---|---|
| Simple bowl | < ~15 km | ~0.20 | Parabolic bowl, sharp rim, no floor/peak |
| Transitional | ~15–25 km | ~0.10–0.15 | Incipient wall slumping, partial flat floor |
| Complex | ~25–137 km | ~0.05–0.10 | Flat floor, central peak, terraced walls |
| Peak-ring | ~137–300 km | ~0.03–0.05 | Peak ring at ~0.5D replaces central peak |
| Multi-ring | > 300 km | Very shallow | Concentric rings at ~√2 spacing ratio |

For your smaller body (870 km radius vs. Moon's 1,737 km), **the simple-to-complex transition diameter scales inversely with surface gravity**. At roughly half lunar gravity, the transition shifts to ~30 km. Apply this scaling to all thresholds.

**Critical failure mode**: Using a single bowl profile for all crater sizes. A 100 km crater rendered as a simple bowl (d/D = 0.2, giving 20 km depth) looks absurdly deep and wrong. Complex craters must be much shallower with flat floors and central peaks.

### What makes craters look ugly — and how to fix it

The five most damaging failure modes, ranked by visual impact:

**Missing ejecta blankets** is the #1 cause of ugly craters. Without the raised rim apron and power-law ejecta falloff, craters look like holes punched into a flat surface with a cookie cutter. The fix is to always include the r^(−3) ejecta term extending to at least 3–5 crater radii.

**Perfect circular symmetry** makes craters look machine-stamped. Real impacts are rarely perfectly vertical — most arrive at oblique angles, producing slight ellipticity (axis ratio typically **1.0–1.2**, rarely up to 1.5). Add 5–15% noise perturbation to rim height, ejecta thickness, and the radial profile itself. Vary the angular distribution of ejecta — for oblique impacts, create an uprange avoidance zone.

**Uniform degradation state** — all craters at the same freshness — breaks immersion instantly. A realistic surface is a palimpsest: most craters are ancient and subdued, very few are young and sharp. The degradation model is described in detail in the SFD section below.

**Wrong depth-to-diameter scaling** across the size range. Simple craters at d/D ≈ 0.2 transitioning abruptly to complex craters at d/D ≈ 0.05 creates a visual discontinuity. Interpolate smoothly through the transitional range.

**Cookie-cutter repetition** — all craters sharing the same profile shape. Vary the profile parameters (rim height, floor flatness, peak prominence) by ±20% around the empirical mean, and randomize the number and position of wall terraces.

---

## 2. Size-frequency distribution and crater population

### The production function

The standard lunar crater SFD is the **Neukum Production Function** (Neukum 1983, revised 2001) — an 11th-degree polynomial in log-log space valid from 10 m to 300 km. For procedural generation, a piecewise power-law approximation is more practical:

- **Large craters (D > ~2 km)**: cumulative slope **b ≈ −1.8 to −2.0** (N(>D) ∝ D^b)
- **Small craters (D < ~2 km)**: cumulative slope **b ≈ −3.0 to −3.8** (steep branch)

The knee at ~1–2 km is the most important feature. For a 3.5 Ga mare surface, the absolute density is approximately **N(D > 1 km) ≈ 5,500 per 10⁶ km²** (Robbins et al. 2014).

**Sampling algorithm**: Inverse CDF sampling from a bounded power law is O(1) per crater and trivially generates millions: **D = D_min / sqrt(1 − u × (1 − (D_min/D_max)²))** for cumulative slope −2 (u uniform on [0,1]). For a multi-slope SFD, use stratified sampling — compute expected counts per diameter bin, sample each bin independently with its local slope, then combine. This is a bake-time computation.

### Saturation and equilibrium

Geometric saturation — hexagonal close-packing of same-sized craters — occurs at **N_gs(>D) = 1.54 × D^(−2)** (~7.6% area coverage). Real surfaces never reach this; empirical equilibrium sits at **1–5% of geometric saturation** (Gault 1970, Xiao & Werner 2015). The equilibrium SFD always has a slope of **−2 cumulative**, regardless of the steeper production slope, because crater destruction balances creation.

For your pipeline: generate craters from the production function, then enforce a saturation cap. If the density at any size exceeds ~3% of geometric saturation, cull the oldest craters at that size. This naturally produces the observed slope break.

### Age assignment and degradation

Crater formation approximates a Poisson process, so ages are uniformly distributed in time. But survivorship is strongly size-dependent — **small craters are destroyed much faster than large ones**. The topographic diffusion model (Soderblom 1970, Fassett & Thomson 2014) treats degradation as: **∂h/∂t = κ∇²h**, with lunar diffusivity **κ ≈ 5.5 m²/Myr**. The degradation state is parameterized by **K = κt**. A crater's fractional depth reduction scales as approximately **exp(−C × K/D²)**, meaning a 300 m crater at 3 Ga is reduced to ~7% of its initial depth, while a 10 km crater retains most of its original form.

For rendering, map degradation to visual parameters using the Pohn & Offield 7-class system:

- **Class 7** (fresh, < ~100 Ma): Sharp rim, visible rays, full ejecta, d/D at full value
- **Class 4** (moderate, ~1–2 Ga): Rim softened ~50%, ejecta faded, floor partially infilled
- **Class 1** (ghost, > ~3.5 Ga): Rim barely discernible, nearly flat, ejecta absent

**Bake-time operation**: Apply diffusion-based degradation to each crater based on its assigned age before stamping into the heightmap. This is the most physically accurate approach and too expensive for realtime.

### Crater overlap

Sort craters oldest-first and rasterize sequentially. Neither pure addition nor pure minimum handles overlap correctly — the best documented approach (from the Astrolith procedural planet generator, GameDev.net) samples a **base elevation** when each crater is first placed, then blends between absolute (interior) and additive (exterior) modes: the interior uses base elevation plus crater profile, while the exterior lerps toward additive blending as distance increases. Young craters stamp into the existing terrain, correctly overprinting older features.

When a new crater's interior fully contains an older crater, the older crater is obliterated (**cookie-cutting**). When ejecta thickness exceeds a small crater's depth, that crater is buried (**gardening**). Both effects happen naturally with age-ordered stamping if ejecta is applied additively.

---

## 3. Mare flooding with realistic boundaries

### Why near-side bias exists

The near-side/far-side mare asymmetry on the real Moon results from two reinforcing factors: **crustal thickness dichotomy** (near-side crust ~30–40 km, far-side ~50–60 km from GRAIL data) and **KREEP concentration** (radioactive heat-producing elements concentrated on the near side sustaining mantle melting for billions of years). Thinner crust allows magma dikes to reach the surface; thicker far-side crust traps magma as intrusions (cryptomaria).

For your body, model a hemispheric crustal-thickness gradient. Set a threshold: basins where crust is thinned below a critical value (by impact excavation plus the hemispheric gradient) receive lava flooding; deeper basins get more fill.

### Making mare boundaries look like lava, not bathtubs

The single biggest failure mode for procedural maria is **filling below a flat elevation threshold**, which produces perfectly level, geometrically clean shorelines. Real mare boundaries exhibit irregular embayments, peninsulas of highland terrain, kipukas (isolated highland islands), and lobate flow fronts.

The recommended technique is a **noise-modulated flood fill**: compute a base fill level for each basin from its depth and crustal-thickness-derived eruption volume, then modulate this level spatially with **domain-warped Perlin noise** (amplitude ~50–200 m, frequency tuned to produce embayments at the 10–50 km scale). This creates organically irregular boundaries. Run **multiple flooding episodes** at slightly different noise-modulated levels to simulate sequential eruptions — each episode gets a subtly different albedo (high-Ti basalts are darker).

**Ghost craters** — partially buried pre-mare features visible as circular ridges within maria — are essential for realism. Place craters on the pre-mare terrain before the flooding step. Craters whose rims protrude just above the lava level become ghost craters with subdued circular ridges. Concentrate them along mare-highland boundaries where basalt is thinnest. The 661 buried craters identified globally on the Moon are preferentially distributed at mare margins.

**Wrinkle ridges** are low sinuous compressional ridges on mare surfaces. Measured dimensions: **~40 m height, ~290 m width, segments ~3.5 km long** (global Frontiers 2023 dataset). They follow basin-concentric patterns and correlate with thicker basalt fill (>500 m). Model as sinuous polylines extruded with narrow Gaussian profiles. Place circular wrinkle ridges over ghost crater rims.

**Bake-time operation**: All mare flooding, ghost crater identification, and wrinkle ridge placement should happen during the bake pass. These are low-frequency features that define the global appearance.

---

## 4. Space weathering and albedo control

### Albedo values and maturity mapping

Space weathering on airless silicate bodies produces **nanophase metallic iron** (npFe⁰, 1–15 nm particles) in grain rims through micrometeorite bombardment and solar wind implantation. This causes progressive **darkening and reddening** following approximately logarithmic kinetics.

| Surface type | Albedo (geometric) |
|---|---|
| Fresh anorthosite (highland ejecta) | 0.25–0.30 |
| Mature highland | 0.10–0.15 |
| Fresh basalt (mare ejecta) | 0.10–0.15 |
| Mature mare | 0.05–0.08 |
| Lunar global average | ~0.12 |

Map crater age to maturity using: **maturity = clamp(ln(age_Ma) / ln(1000), 0, 1)**, then **albedo = lerp(fresh_albedo, mature_albedo, maturity)**. The logarithmic relationship means rapid initial darkening (~20–30% reduction in the first ~100 Ma) with asymptotic approach to saturation by ~1 Ga. Craters younger than ~100 ka appear essentially fresh.

### Ray system generation

Rays are the most visually distinctive feature of young craters and critical for the "real Moon" impression. They form from **high-velocity ejecta** launched ballistically and from **secondary impact chains** that excavate fresh local material along radial paths. Tycho (85 km, ~108 Ma) produces rays extending >1,500 km with nearly **10⁶ secondary craters >63 m diameter**.

Key characteristics for procedural generation:

- **Ray count** scales with crater size relative to surface roughness wavelength. Assign 8–24 rays per young crater with random angular spacing.
- **Asymmetry is essential**: exclude a 30–90° angular sector to simulate oblique impact (Proclus is the textbook example). Vary ray intensity and width per spoke.
- **Radial falloff**: brightness decays as a power law with distance. Rays are thin (meters-thick) deposits, bright because they expose fresh immature material.
- **Two persistence mechanisms** (Hawke et al. 2004): immaturity rays fade over ~1 Ga as weathering matures the surface; compositional rays (highland material on mare) persist indefinitely.
- **Rendering**: For the bake pass, stamp ray patterns into the albedo cubemap using per-crater polar-coordinate noise (angular hash creates spokes, radial power-law controls falloff). Musgrave's original approach uses ridged fBm along the radial direction with a cutoff threshold, plus "splatter" turbulence for texture.

**Bake vs. realtime**: Ray systems are best baked into the albedo map during the SpaceWeather stage, since they are per-crater features that need to overlay the entire surface. Realtime evaluation is possible for a small number of prominent ray craters but expensive for the full population.

---

## 5. Megabasins as the topographic skeleton

### Reference basin parameters

Your hand-placed megabasins establish the large-scale topographic framework that all subsequent stages build upon. Key real-world reference basins:

**South Pole-Aitken**: ~2,500 km diameter, 6–8 km deep, 4 tentative ring structures. The Moon's largest basin; filled only ~3–4% by mare basalt despite extreme depth. Located on far side. Age ~4.3 Ga. In isostatic equilibrium (much shallower than original excavation).

**Orientale**: ~930 km outer ring (Cordillera), best-preserved multi-ring basin. Four rings at **930, 620, 480, 320 km** — verifying the **√2 ring spacing ratio** (930/620 = 1.50, 620/480 = 1.29, 480/320 = 1.50, average ≈ 1.43 ≈ √2). Relatively unflooded, exposing basin structure clearly. The best morphological reference for multi-ring basin profiles.

**Imbrium**: ~1,160 km, 3 major rings. Prominent radial ejecta sculpture ("Imbrium sculpture") extending across much of the near side. Largest nearside mascon. Inner ring peaks mostly submerged beneath mare basalt — only isolated peaks like Mons Pico and Mons La Hire protrude.

### The √2 ring spacing rule

The ratio between successive ring diameters is empirically **√2 ≈ 1.414 (±0.3)**, established from 296 rings across 67 multi-ring basins on Moon, Mercury, and Mars (Pike & Spudis 1987). This ratio is **target-invariant**, suggesting a fundamental formation mechanism. Numerical hydrocode simulations (Johnson et al. 2016, 2018) confirm this ratio emerges from the interaction between inward-collapsing cavity walls and central mantle uplift.

For procedural generation: define each megabasin by center, outermost ring diameter, and number of rings. Compute inner ring diameters by successive division by √2. Model each ring as a Gaussian ridge profile with noise-perturbed radius and amplitude. The inner depression is a broad negative paraboloid floored at the basin depth (corrected for isostatic compensation — ancient basins are much shallower than initial excavation depth). Depth/diameter ratios for relaxed basins range from ~1/100 to 1/400, far shallower than crater-scale d/D ratios.

**Bake-time operation**: Megabasins are placed first and define the low-frequency terrain shape. They should be authored or seeded deterministically.

---

## 6. Regolith and the "craters all the way down" problem

### Why fBm fails for lunar surfaces

Standard fractional Brownian motion produces smooth, rounded, Gaussian-distributed terrain that reads as **eroded terrestrial landscape** — hills and valleys with no characteristic depressions. Lunar surfaces are crater-saturated at every observable scale down to centimeters. The visual signature is overlapping circular depressions, not flowing terrain.

### Crater-shaped noise functions

The recommended approach replaces fBm with **multi-octave Voronoi/Worley-based crater noise**:

**Warped Voronoi cells** (realtime-capable): Compute F1 (distance to nearest Voronoi point) and apply a crater-shaped transfer function: `crater = smoothstep(0, crater_radius, F1)` remapped from depth to rim height. Layer multiple octaves at different cell densities, with amplitude following the SFD power law (each octave represents a different crater size decade). Warp input coordinates with low-frequency noise to break grid regularity. Inigo Quilez's "voronoise" technique (iquilezles.org) generalizes between Perlin noise and Voronoi, allowing smooth blending.

**The I-Novae Infinity approach** (Ysaneya's GLSL implementation) uses 3D cell noise evaluated on the unit sphere. Each octave computes the distance to the nearest cell point in a 3×3×3 neighborhood (27 lookups), applies a crater profile function, and accumulates. Three octaves at different frequencies produce craters spanning roughly three size decades. FBM noise warps the cell positions for non-uniform distribution. This runs in realtime on consumer GPUs.

**Hybrid strategy for your pipeline**: Bake explicit craters (with full morphology, ejecta, degradation) down to your resolution limit during the Cratering stage. Below that threshold, the Regolith stage adds crater-shaped Voronoi noise that continues the SFD seamlessly. The transition should be imperceptible — match the amplitude and density of the noise octaves to the tail of your explicit crater population.

**Bake vs. realtime**: The Voronoi crater noise is well-suited for realtime evaluation in fragment shaders (the I-Novae engine does exactly this). For the bake pass, you can afford explicit small-crater stamping with individual profiles, which gives better overlap behavior and degradation control.

---

## 7. Spatial indexing and realtime sampling architecture

### How production games handle this

The dominant architecture across Elite Dangerous, I-Novae Infinity, Outerra, and KSP is **cubemap with quadtree LOD**. Each of the 6 cube faces is a quadtree that recursively subdivides based on camera distance. Terrain patches are generated by GPU compute shaders evaluating noise functions, then cached as heightmap/normal map textures per quadtree node.

**Elite Dangerous** (Frontier Developments): Uses "noise graphs" — collections of noise equations taking position and planet parameters as input — evaluated per-patch on GPU compute. Planet-specific geological parameters (crater density, volcanism, etc.) from the StellarForge simulation are packed into a GPU buffer that modulates noise functions. Uses **64-bit and dual-float precision** (two 32-bit floats emulating 64-bit) because planet-scale coordinates reach tens of billions of millimeters. The Horizons-era approach was fully procedural; Odyssey introduced tiled heightmap segments which caused visible repetition artifacts — a cautionary tale about mixing procedural and tiled approaches.

**I-Novae Infinity engine** (Ysaneya): Uses 3D cell/Voronoi noise for craters evaluated analytically in fragment shaders. The `gpuCellCrater3DB` function evaluates crater profiles using the cell-noise approach described above. Reports that shaders can be "thousands of instructions" but GPU-bound on fillrate rather than ALU — modern GPUs handle the arithmetic well.

**Outerra**: Uses a "vector stage" for overlaying analytical features (including craters) onto generated terrain tiles. Craters are specified by center/diameter/depth and rasterized into terrain tiles as the viewer moves — a **hybrid analytical/rasterized approach**. Each crater definition is only 64 bits of storage.

### Spatial indexing for your crater database

For 10,000+ explicit craters on a sphere, the most practical approach is **cubemap-face quadtree indexing** — which you already have as part of your terrain LOD system. When subdividing quadtree nodes, cull the parent node's crater list to find which craters overlap each child node. This naturally builds a spatial index during LOD traversal. The Astrolith implementation (GameDev.net) documents this approach in detail.

For realtime shader evaluation of implicit craters (the Regolith stage), use the **3D cell approach**: divide space into cells using `floor()` of scaled 3D coordinates, check the current cell and 26 neighbors (3×3×3). This is exactly standard Worley noise — fixed cost of 27 hash lookups per octave regardless of total crater count. Using 3D coordinates on the unit sphere avoids cubemap seam artifacts entirely.

### LOD-windowed crater contribution

Each crater should contribute only at LOD levels where it subtends enough pixels to be visible. Large craters contribute at coarse LODs; small craters appear only at fine LODs. **Fade contributions smoothly** based on screen-space size to prevent popping — use a soft ramp over the transition range rather than a hard cutoff. For normal maps, pre-convert heightmap to normal map per LOD tile; this ensures lighting detail is always present even when polygon density is low, and remains stable during LOD transitions. The **CDLOD** (Continuous Distance-Dependent LOD) algorithm by Strugar (2009) provides smooth vertex morphing between LOD levels.

---

## 8. What reads as "real Moon" versus generic procedural body

The visual credibility of a lunar surface depends on a short list of non-negotiable features, roughly ordered by importance:

**Crater saturation at every scale** is the most important single factor. Every patch of surface, at every zoom level, must show craters. If zooming in reveals smooth terrain between large craters, the illusion breaks immediately. This is what the Regolith noise stage must provide.

**Highland/mare albedo dichotomy** with a roughly **2:1 to 3:1 brightness ratio** (highlands at 0.10–0.15, mare at 0.05–0.08) creates the iconic face-of-the-Moon pattern. The near side shows extensive dark maria; the far side is almost entirely bright highland. Without this, the body reads as a generic asteroid.

**Bright ray systems** crossing dark mare are visible to the naked eye on the real Moon. Tycho's rays span nearly the entire near side. Even one or two prominent ray craters dramatically increase realism.

**Sharp, perfectly black shadows** with no atmospheric fill. Any hint of blue haze, scattering, or soft shadow edges breaks the illusion of an airless body. Terminator lighting (low sun angles) is the most dramatic view — long shadows reveal topographic relief across the entire limb region.

**Opposition surge** — a **>40% brightness increase** as the phase angle approaches zero (Buratti et al., Clementine data) — is caused by coherent backscatter enhancement and shadow hiding. Implement as a phase-angle-dependent term in the surface BRDF. Without it, full-phase views look unnaturally flat.

**Degradation variation across the age spectrum**: the surface must read as ancient — a palimpsest of billions of years of bombardment where fresh sharp craters are rare punctuations against a background of subdued, overlapping ancient impacts. If all craters look equally fresh, the surface reads as artificially generated.

### Key reference imagery sources

For visual reference during development, the **Lunar Reconnaissance Orbiter Camera** (LROC) is the primary modern source — NAC images at **0.5 m/pixel** for surface detail, WAC mosaics at 100 m/pixel for global context. The interactive QuickMap3D viewer allows examining specific features. Apollo orbital photography (metric and panoramic cameras from missions 15–17) provides excellent oblique views. Kaguya/SELENE HDTV imagery provides the best full-disk far-side views. For terminator lighting reference, search LROC featured images for low-sun-angle shots of any highland region. For ray systems, full-Moon Earth-based astrophotography clearly shows the Tycho, Copernicus, and Aristarchus ray systems.

---

## Pipeline stage summary: bake vs. realtime allocation

| Pipeline stage | Bake (cubemap, fidelity) | Realtime (shader, performance) |
|---|---|---|
| **Sphere** | Base sphere shape | — |
| **Differentiate** | Composition/material map | Low-res texture lookup |
| **Megabasin** | Ring profiles, depth, ejecta sculpture | — |
| **Cratering** | Explicit craters >threshold with full morphology, ejecta, degradation | — |
| **MareFlood** | Noise-modulated flood fill, ghost craters, wrinkle ridges | — |
| **Regolith** | Could pre-bake into normal maps | Voronoi crater noise in fragment shader |
| **SpaceWeather** | Ray systems, maturity albedo map | Opposition surge BRDF term |

The general principle: **all discrete, age-ordered, overlap-sensitive features belong in the bake pass** (craters, mare, rays). **Self-similar, resolution-independent detail belongs in the realtime shader** (small-crater noise, albedo lookup, lighting).

---

## Key references and implementations

**Planetary science foundations**: Pike (1974, 1977) for all crater morphometric power laws. Krüger et al. (2018) "Deriving Morphometric Parameters" in JGR Planets for the most current central peak and depth relationships. McGetchin, Settle & Head (1973) for the ejecta thickness equation. Neukum (1983) and Neukum et al. (2001) for the production function. Fassett & Thomson (2014) for diffusion degradation with κ ≈ 5.5 m²/Myr. Hawke et al. (2004) for ray classification. Pike & Spudis (1987) for the √2 ring spacing ratio. Johnson et al. (2016, 2018) for multi-ring basin hydrocode simulations.

**Game engine references**: Elite Dangerous terrain generation (Doc Ross, 80.lv 2018 interview; also wccftech.com coverage of Horizons planet tech). I-Novae Infinity engine cell-noise crater shaders (Ysaneya's GLSL `gpuCellCrater3DB` function). Outerra crater blog post (outerra.blogspot.com, March 2013). No Man's Sky GDC 2017 talk by Innes McKendrick ("Continuous World Generation"). Strugar's CDLOD algorithm (2009) for smooth LOD transitions.

**Procedural generation**: Musgrave, *Texturing and Modeling: A Procedural Approach* (1998/2002) — the `luna.c` source code for the original procedural moon including crater profiles and ray systems. The Astrolith planet generator blog series on GameDev.net ("Planet Generation: Impact Craters") for practical overlap handling and quadtree crater culling. Inigo Quilez's articles on smooth Voronoi (iquilezles.org/articles/smoothvoronoi/) and voronoise for crater-shaped noise. The Cratermaker library (cratermaker.readthedocs.io) — a modern Python/Rust tool implementing scientifically accurate `crater_profile()`, `ejecta_profile()`, and `ray_pattern()` functions based on Richardson et al. models.

**Open-source implementations worth studying**: Cratermaker (cratermaker.readthedocs.io) for scientific crater morphology. TerraForge3D (github.com/Jaysmito101/TerraForge3D) for node-based procedural terrain with crater support. moon_gen (github.com/mbiselx/moon_gen) for simple crater placement with ejecta. OmniLRS (github.com/OmniLRS/OmniLRS) for semi-procedural lunar terrain at 2.5 cm resolution. Parallax KSP mod (github.com/Gameslinx/Tessellation) for tessellation and triplanar terrain shaders. Planet-LOD (github.com/sp4cerat/Planet-LOD) for adaptive spherical LOD.

## Conclusion

The path from "ugly craters" to convincing lunar terrain runs through three critical improvements. First, **the profile function must be piecewise with all five zones** — especially the ejecta blanket, which most implementations omit. Second, **morphology must vary with size** — complex craters need flat floors, central peaks, and terraces at physically-motivated diameter thresholds. Third, **degradation must span the full age spectrum** using diffusion-based rim softening so that fresh craters are rare bright punctuations against a weathered ancient surface. Get these three right, and the surface will read as a real world rather than a texture. The remaining stages — mare flooding with noise-modulated boundaries and ghost craters, ray systems baked into albedo, and Voronoi crater noise for the regolith detail below your explicit-crater threshold — build on this foundation to complete the illusion.