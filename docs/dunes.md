# A Real-Time, GPU-Friendly Algorithm for Procedural Sand Dune Heightmaps on Spherical Planets

This report synthesizes dune geomorphology, real-time procedural-noise techniques, and sphere-specific sampling concerns into a concrete algorithm you can implement as a WGSL/Bevy shader function `dune_height(p, base_h, params) -> (h, dh/du, dh/dv)`. The deliverable is intentionally an *algorithm + design rationale*, not finished code.

---

## A. Dune morphology and physics fundamentals (parameter grounding)

The morphology of a sand sea is controlled by four roughly independent inputs, and reading dunefield imagery as a function of them is what gives parametric control its physical legitimacy.

**1. Wind directionality regime (the dominant parameter).** Field and flume work codified by Rubin & Hunter (1987), with later generalizations by Courrech du Pont, Narteau, and Gao (Nature Comm. 2017; Sci. Reports 2015), gives the *gross bedform-normal transport* (GBNT) rule: in a multidirectional wind regime, dunes orient themselves to the crest direction α that maximizes Σᵢ |Qᵢ · n(α)| (the sum of normal sand-flux components from all winds). The rule has clean limits:
- **Unidirectional / narrow unimodal** → barchans (low sand) or transverse ridges (high sand); crest perpendicular to wind.
- **Bidirectional, divergence angle 90–135°** → linear/seif dunes; crest along the resultant transport direction. With unequal wind strengths the crests become oblique and meander (sinuous seifs); with equal strengths they're straighter ("vegetated linear" form with Y-junctions).
- **Multidirectional (RDP/DP → 0)** → star dunes, with three or more arms meeting at a high central peak; they grow vertically rather than migrating.
- **Bimodal at acute angles + low sand** → asymmetric barchans whose downwind limb extends and ultimately becomes a seif (Bagnold; Parteli et al. 2014).

**2. Sand availability.** Distinguishes erodible bed (continuous) from sand-on-bedrock (discontinuous, "fingering" growth). On bedrock with low supply you get isolated barchans with sharp dune-field boundaries (Namib coastal margin); on a fully covered bed you get connected transverse/linear/star dune *seas*. Crucially, Courrech du Pont's work shows availability also flips the dune *growth mechanism* (bed-instability vs. fingering elongation) and so changes orientation, even at fixed wind regime. The **sharp, almost step-function boundary** of a sand sea against pavement (visible in the Namib reference) is itself a key parameter — model it as a hard mask, not a smooth fade.

**3. Wavelength selection.** Andreotti, Claudin & co. established that the smallest dune wavelength λ₀ scales with the sand-flux saturation length Lsat ≈ (ρₛ/ρ_f)·d (the "drag length"), giving λ₀ on Earth ~10–20 m for sand. Above this, dune size is set by a *maximum-growth-rate* mode (validated experimentally in PNAS 2021). Andreotti et al. (Nature 2009) showed that **giant dunes (draa)** select their wavelength from the depth of the atmospheric boundary layer — typical 300 m – 3.5 km. So the natural dune-bedform hierarchy is three roughly decoupled wavelength bands:
- **Ripples** (0.05 – 1 m, set by saltation hop length)
- **Dunes** (10 – 500 m, set by Lsat / growth-rate maximum)
- **Megadunes / draa** (0.5 – 5 km, set by atmospheric boundary layer)

These are essentially modular and superpose in the real world (small dunes climbing a slow-moving draa). That is the single most useful structural fact for procedural generation.

**4. Asymmetric profile.** Each dune has a gentle stoss face (~10–15°), a sharp brink line, and a slip face at the angle of repose (32–34° dry sand). Symmetric sinusoidal noise *never* looks like sand; the asymmetry is the entire visual signature.

**5. Branching / Y-junctions.** Linear dunes commonly merge in tuning-fork "Y" patterns where two ridges converge downwind. This is a topological signature that pure ridged-noise will not produce on its own; you have to put it in via the warping field.

---

## B. Real-time procedural techniques

Here is the survey of GPU-friendly building blocks, scored for dune use.

### B.1 Anisotropic / oriented noise

- **Gabor noise (Lagae, Lefebvre, Drettakis & Dutré, SIGGRAPH 2009).** A sparse sum of Gaussian-windowed oriented sinusoids (Gabor kernels). It gives *direct* control over orientation, principal frequency, and bandwidth — exactly the knobs we want. The cost is non-trivial (a few dozen kernel evaluations per pixel) but the kernels are local, so a hash grid evaluation works on the GPU. The 2010 STAR survey (Lagae et al., *Computer Graphics Forum*) and Galerne et al.'s "Gabor Noise by Example" (2012) include practical implementations. **This is the most natural dune-crest primitive when you can afford it.**
- **Spectral / oriented noise via filtered isotropic noise.** Cheaper alternative: take Perlin/simplex noise sampled along a "wind aligned" direction, optionally with a sin(2π · u/λ) modulation along the cross-wind axis. The familiar `sin(dot(p, wind)/λ + fbm(p))` is a degenerate Gabor noise with one orientation and infinite bandwidth.
- **Flow noise (Perlin & Neyret, SIGGRAPH 2001 sketches; Gustavson & Strand, JCGT 2022 "psrdnoise").** Adds rotation of gradients and advection of scales over time; gives "swirling" temporal evolution at constant cost. Useful for evolving the cross-wind component without simulation.
- **Anisotropic Gabor with `2D gabor(p, freq=1/λ, omega=W(p))`** is what you want for the dune-scale layer. Set bandwidth low (narrow band) for transverse/linear (clean stripes) and high for messy/sinuous (barchanoid).

### B.2 Producing the sharp asymmetric dune profile

This is the part most procedural attempts get wrong. A pure cosine ridge gives ~equal slopes; for a dune you need ~10° stoss and ~32° slip.

The clean trick is to **map a band-limited oriented signal through an asymmetric 1-D shaping function aligned to the wind**. Given:
- `s = signed phase along wind` ∈ [−π, π]
- `n(p) = oriented (Gabor-like) crest field` returning crest *position* (zero crossings on crests)

Compute the local phase `φ` along wind, then a sawtooth/skewed-triangle: define ramp-up from 0→1 over a fraction `α` of the wavelength (stoss), then ramp-down from 1→0 over fraction (1−α) (slip). With α ≈ 0.85, the slope ratio is ~5.7:1, giving the right stoss/slip asymmetry for a dune amplitude that produces ~30° on the slip side. In WGSL-ish:

```
fn asym_ridge(phase: f32, alpha: f32) -> f32 {
    let t = fract(phase);                  // 0..1 along wind
    return select((1.0 - t)/(1.0 - alpha), t/alpha, t < alpha);
}
```

The brink (the sharp line) is the C¹ discontinuity at `t = α`. To soften the windward toe (which is rounded in reality, not pointed), pass through `smoothstep` near `t = 0` with a small width. To get the right behavior under fbm, this asymmetric shaping must be applied **after** the warping/orientation step, so that the steep face faces locally downwind everywhere.

This same trick (asymmetric triangle in the wind-aligned phase) is the workhorse used by:
- *Inigo Quilez*'s dune-style raymarched landscapes,
- The Houdini "Terrain Dunes" shelf tool (a heightfield triangle wave + warp),
- Most Shadertoy "Desert Sand" shaders (e.g., Shane's `ld3BzM`),
- Alan Zucconi's *Journey* sand-shader walkthrough.

### B.3 Domain warping for sinuous crests, branching, and Y-junctions

Iñigo Quilez's classic article on domain warping (`p → p + h(p)` where h is fbm) is the right tool for adding sinuosity. For dunes you want the warp to be **anisotropic**: warp strongly in the cross-wind direction, weakly along wind, so crests meander but spacing is preserved. A two-pass fbm warp with a per-pixel rotation matrix that aligns to the local wind tangent gives the dendritic / star megadune look in the Namib reference.

Y-junctions appear naturally where two warped crests of similar phase converge — to *encourage* them, modulate the wavelength itself by a low-frequency noise so crest spacing varies, and adjacent crests sometimes merge.

### B.4 Worley / Voronoi for star and barchan dunes

For star megadunes and isolated barchans, a Worley/Voronoi cellular noise is a better primitive than oriented noise:
- **Star dunes:** place Poisson-distributed cell centers; each center is a peak; the height field is `peak_amplitude · max(0, 1 − F1/R)` with three "arms" cut by minimum distance to a fan of N (3–6) directions emanating from the cell center. Where two cells meet, ridges form between them — exactly the look of a star-dune field.
- **Barchans:** at each cell center, place a parametric crescent template oriented along W(cell). The crescent can be encoded as `h = stoss(d_along_wind) − slip(d_along_wind) · mask_inside_crescent`, with horns extending downwind by enlarging the crescent's cross-wind support along the leeward side (Hersen 2003 derivation gives an explicit shape). This gives crisp horns that sin-noise approaches cannot.

Both are O(1) per pixel for a 3×3 neighborhood lookup and are very GPU-friendly.

### B.5 Reaction-diffusion / Turing patterns

Two-morphogen Turing systems (activator-inhibitor on a fixed grid) generate clean stripe patterns and zebra-like domains; they have been proposed for dune-like striping. They're elegant but they require iteration, so they're a fallback for **bake-time** generation of a low-frequency draa template — not for real-time per-pixel evaluation. For your real-time path, prefer Gabor/Worley.

### B.6 Multi-scale superposition

Combine three independent layers, each on its own wavelength band. Important detail: the **ratio of upper-to-lower wavelength** in real dunefields is typically 5–20, and the crests of the smaller scale are often *not* aligned with the larger scale (small dunes climb a draa obliquely). So apply different orientations or even a different regime field per layer. The Namib reference image is precisely this: a dune-scale linear/transverse field that climbs a slowly varying draa-scale envelope.

### B.7 Existing implementations to study

- **Houdini *Dune Solver* (Barrett Meeker)** — OpenCL Werner-style CA with bedrock+sand layered model; used by DNEG on *Dune* (2021/2024). Reference for offline ground truth, not real-time, but its `bedrock + sand_layer` architecture is exactly the pattern your shader should mimic statically.
- **DNEG's *Dune* pipeline (SideFX article).** Confirms heightfield + custom solver was preferred over physical sim for dune *shape* generation, even at film budget. They blocked dunes statically and only ran simulations for sandworm-induced motion. This validates a non-simulation approach for static dune fields.
- **thatgamecompany / *Journey* (John Edwards, GDC 2013; Alan Zucconi blog series).** Authoritative reference for *shading* sand (ocean-spec + glitter + view-dependent ripple normal map). The geometry was authored 3-D models, not procedural. So for your project, *Journey* informs the shading half but not the heightmap half.
- **No Man's Sky / Star Citizen.** Both rely on noise stacks with biome-specific masks but neither has published a dune-specific algorithm. The general approach is "fbm + domain warp + biome mask" which is the baseline you're trying to beat.
- **Inigo Quilez** — articles on warp, fbm, smin/sabs, and his terrain shaders (`Elevated`, `Rainforest`, `Snail`) are the canonical source for *raymarched* heightfields built from warped fbm with analytic derivatives. The analytic-derivative pattern (each noise returns `(value, ∇value)`) is what you want, because the dune asymmetry needs to know slope to place the brink line.
- **Shadertoy `ld3BzM` (Shane, "Desert Sand")** — a clean reference for cheap dune-look noise using sin(fbm-warped phase) with asymmetric clamping.

---

## C. Sphere-specific concerns

The dune algorithm must consume a position on `S²` and return a height. Three concerns dominate.

### C.1 Defining the wind field W(p) on the sphere

Start with a **prevailing-wind model** parameterized analytically. Let `p` be a unit vector. Define latitude `φ = asin(p.y)`. Earth-like Hadley/Ferrel/Polar cells give a piecewise-zonal pattern: trade winds (easterly) at |φ| < 30°, westerlies between 30°–60°, polar easterlies > 60°. Build the local wind in the tangent plane:

```
let east  = normalize(cross(vec3(0,1,0), p));   // +φ tangent (east)
let north = cross(p, east);                     // +φ tangent (north)
let zonal = cos(3.0 * phi);                     // alternates sign every 30°
let merid = 0.2 * sin(3.0 * phi);               // weak meridional component
let W_proto = zonal*east + merid*north;
```

This already gives smoothly varying, latitude-dependent prevailing winds with the right Earth-like signs. To make it more interesting:

- **Add a curl-free random potential** `Φ(p)` defined as low-band spherical noise (sample a 3-D simplex noise at `p` itself; it's intrinsically seamless on the sphere). Take its surface gradient: `∇_S Φ = ∇Φ − (∇Φ · p) p`. This adds large-scale low-pressure-system perturbations.
- **Add a divergence-free part** by taking `∇_S (Φ₂) × p` for a second potential. Helmholtz-Hodge decomposition guarantees coverage of any tangent field this way (Jun & Genton 2018, *JASA*, "Modeling Tangential Vector Fields on a Sphere"). Mix curl-free and div-free by user weights — gives you "convergent zones" (curl-free dominant, ITCZ-like) and "rotational zones" (div-free, cyclonic).
- **Bidirectional / multidirectional regimes.** A single tangent vector is unidirectional. For bidirectional you need two vectors `W₁`, `W₂` and a transport ratio. Define them as `R(p) · W(p)` for a small set of rotations that vary with latitude/biome, and bake their angular distribution into a *regime field* `R(p) ∈ {uni, bi(angle), multi}` (a smoothly interpolated scalar). This regime field is then queried at the dune layer to choose the morphology generator.

### C.2 Sampling oriented noise on a sphere without seams or pole artifacts

Two viable approaches, both common in real engines:

**(i) Solid noise in 3-D** — sample 3-D simplex/Worley/Gabor at the *world position* `p` (or `p · R_planet`). No UV unwrapping, no seams. This is what `bevy_terrain` and most planet-shader projects do. Drawback: 3-D Gabor noise is more expensive than 2-D, and it's harder to align kernels to a 2-D wind vector. **Solution:** for the asymmetric-ridge step, project locally into the tangent plane (using `east`/`north` from §C.1), evaluate a 2-D oriented noise in the tangent plane parameterization, then add to the height. This is exactly Lagae's "setup-free surface noise" idea (Gabor surface noise) — locally evaluate 2-D noise in the tangent frame.

**(ii) Cube-sphere face-local UVs** — bake/evaluate per-face. Works but introduces seams across cube edges, which is exactly where dunes look terrible. Not recommended.

**Pole artifacts** are avoided by *never* using a (lat, lon) parameterization for noise sampling; always use 3-D position or a per-face cube-sphere sample.

**Tangent-frame transport:** the wind direction must be parallel-transported smoothly on the sphere. The construction in §C.1 produces this automatically because `W(p)` is defined as a smooth function of `p ∈ S²`, not as a UV-parameterized angle. The local 2-D (`u`, `v`) plane for dune-noise evaluation is the tangent plane spanned by `(W(p)/|W|, p × W(p)/|W|)` — this frame is well-defined wherever |W| > 0 and continuous everywhere else.

### C.3 LOD and detail amplification

The classical detail-amplification pattern works perfectly for dunes because the three wavelength bands map onto LOD bands:

| Layer | Wavelength | Required mesh resolution | LOD activation |
|---|---|---|---|
| Draa / megadune | 500 m – 5 km | sample at low LOD on quadtree | always on |
| Dune | 30 – 500 m | mid-LOD chunks within ~10 km | proximity-gated |
| Ripples (normal-map only) | 0.05 – 1 m | screen-space; bake into normal | within a few hundred meters |

In practice: write `dune_height(p, lod)` to skip noise octaves whose Nyquist exceeds the local sampling frequency. The draa layer is *always* added, the dune layer is gated on planet-relative distance, the ripple layer never affects geometry — it's a procedural normal-map perturbation in the fragment shader (Journey-style).

For analytic anti-aliasing of the brink line at distance, fade the asymmetric-ridge sharpness with LOD: `alpha_eff = mix(0.5, alpha, lod_weight)`. At infinity, the dune becomes a symmetric soft hump (no aliasing); near the camera the brink line snaps to its 32° angle.

---

## D. Parameter design

A clean parameter taxonomy that supports both the Namib megadune look and the close-up barchan look:

### D.1 Per-planet (constant) parameters

- `seed: u32`
- `planet_radius: f32`
- Wind-field parameters: `hadley_strength`, `cyclone_count`, `cyclone_potential_seed`, `meridional_strength`
- Wavelength constants: `lambda_draa` (e.g., 1500 m), `lambda_dune` (e.g., 80 m), `lambda_ripple` (e.g., 0.4 m). Cf. Andreotti's Lsat scaling — for a desert planet with denser atmosphere or different gravity, *all three scale together* by the drag-length ratio, which is a great game-feel knob.
- Amplitude constants: `H_draa`, `H_dune` — set so that slope at the slip face is ~tan(32°) at peak amplitude.
- `alpha_stoss_slip` ≈ 0.8–0.9 (fraction of wavelength on the stoss side).

### D.2 Tangent-vector-field "live" parameters (sampled at `p`)

- `W(p)` — primary wind tangent vector (unit, on the tangent plane).
- `W₂(p)` — optional secondary wind for bimodal regimes.
- `directionality(p) ∈ [0, 1]` — 0 = unidirectional, 0.5 = bidirectional, 1 = multidirectional. Drives the morphology selector.
- `divergence_angle(p)` — only relevant when `directionality ≈ 0.5`; selects between transverse, oblique, and longitudinal seif.

### D.3 Sand-availability scalar field

- `A(p) ∈ [0, 1]` — sand cover. A *sharp* threshold mask is essential to reproduce the Namib edge: `A = step(t, sand_potential(p))` where `sand_potential` is a slow fbm. For diffuse-edge dune fields use `smoothstep(t-w, t+w, …)` with small w. Add a bedrock-elevation modulation so dunes accumulate downwind of topographic obstacles (Bagnold's classical observation).
- `vegetation(p) ∈ [0, 1]` — anchors crests, biases linear and parabolic morphologies, suppresses migration.

### D.4 Style controls (artistic)

- `branching: f32` — magnitude of cross-wind warp; high values give the Namib dendritic look.
- `crest_sharpness: f32 ∈ [0, 1]` — controls smoothstep width at the brink and stoss-toe rounding.
- `multiscale_alignment: f32` — how strongly the dune-scale wind aligns with the draa-scale wind (0 = independent → "messy" stacked patterns, 1 = perfectly aligned → simple textbook fields).
- `sinuosity: f32` — frequency-to-amplitude ratio of the warp field; high → meandering seifs; low → straight bars.

This is enough to dial between the two reference images. For the **Namib megadune look**: `directionality ≈ 0.6` (multi/bi mix), high `H_draa`, mid `H_dune`, large `branching`, sharp `A` mask, low `multiscale_alignment`. For the **golden barchan/transverse look**: `directionality ≈ 0.05`, low `H_draa` (no megadunes), high `crest_sharpness`, vegetation > 0 at margins, smooth `A`.

---

## E. Evolution / mutation over time

You explicitly want non-simulation animation. Three layered strategies, in order of fidelity vs. cost:

### E.1 Phase advection (cheapest, look right at any frame rate)

For each oriented-noise layer, the phase argument is `φ = (p · W) / λ`. Add a time term: `φ → φ + (c · t / λ)` where `c` is a migration speed. Because the asymmetric shaping function is built on `fract(φ)`, this produces dunes that **migrate downwind** at speed `c` while preserving their shape. Choose `c` from real physics — barchan migration is roughly `c ≈ Q / H` where `Q` is sand flux and `H` is dune height, so smaller dunes migrate faster (Bagnold's law). Implement as `c = c0 / (1 + H/H_ref)` to get the visually satisfying "small dunes overtake big ones" effect. The Perlin-Neyret flow-noise paper formalizes this gradient-rotation + scale-advection scheme and it composes naturally with fbm.

### E.2 Slow parameter-field evolution (drives morphological change)

Animate the *parameter fields* slowly. On a CPU/compute-shader tick (once every 5–60 simulated minutes), update:
- `W(p)` — small random perturbation of the cyclone potentials;
- `A(p)` — let sand "blow" between cells: a low-resolution height-field-only update that moves a tiny mass per tick downwind. This is *not* the Werner CA — there are no slabs, no per-frame stepping, no per-particle physics. It's just `A(p, t+Δt) = A(p, t) + ε · (A_upwind − A)`, executed once per minute on a 256×256 spherical texture. Cheap and gives drift of the sand-sea boundary over game time.

The procedural shader reads the slowly-updating `A(p, t)` and `W(p, t)` textures and produces the *current* dune field from them. So you get apparent morphological change (dune fields advancing into bedrock, retreating, swirling) without ever simulating a single grain.

### E.3 Cross-fade between regime samples

To allow a wind regime to *change* over time (storm season → calm trade winds), keep two precomputed regime fields and crossfade. Because dunes adapt slowly in reality, the visual "lag" of the procedural look matches a real planet's seasonal cycle.

### E.4 Noise-seed reseeding for catastrophic events

For dramatic events (sandstorm passes through), bump the per-cell seed of the dune layer in an affected region, with a smooth-blend window. The dune field locally re-randomizes its arrangement.

### E.5 What to *avoid*

- Translating the entire fbm domain by `t · W` — without per-layer phase resets this drifts the *coherent structure* indefinitely and looks artificial.
- Animating `λ` — wavelength is set by physics; changing it implies a global atmospheric change.
- Per-frame Gabor kernel reseed — kernel positions popping is very visible.

---

## F. Recommended algorithm — "Layered Aeolian Heightfield"

A sketch of the full per-pixel/per-vertex function.

```
fn dune_height(p_world: vec3<f32>, base_h: f32, params: DuneParams) -> SampleOut {
    // ---------- 0. Setup tangent frame and fields ----------
    let p = normalize(p_world);
    let east  = normalize(cross(vec3(0.0,1.0,0.0), p));
    let north = cross(p, east);
    let lat   = asin(p.y);

    // Wind field W(p): zonal Hadley pattern + low-band random potential (curl-free + div-free)
    let W = wind_field(p, east, north, params);
    let regime = regime_field(p, params);                  // {uni, bi, multi} blend
    let avail  = sand_availability(p, params);             // ∈ [0,1], can be sharp-edged
    let veg    = vegetation_field(p, params);

    // ---------- 1. Draa (megadune) layer ----------
    // Low-frequency oriented Gabor noise + asymmetric ridge along W
    var h = 0.0;
    var dh: vec2<f32> = vec2(0.0);
    let phase_draa = dot(tangent_uv(p, east, north), W) / params.lambda_draa
                   + 0.6 * fbm2(p_world * (1.0 / params.lambda_draa) * 0.3);
    let draa = asym_ridge(phase_draa, params.alpha_stoss_slip);
    h  += params.H_draa * avail * draa;

    // ---------- 2. Dune layer (morphology selector) ----------
    let dune = select_dune_morphology(
                 p, east, north, W, regime, avail, veg, params);
    h += dune.height * avail * (1.0 - 0.6 * veg);   // veg suppresses dune amplitude

    // Snap-to-base masking: dune amplitude tapers to 0 where avail < threshold
    h = smooth_mask(h, avail, params.edge_sharpness);

    // ---------- 3. Ripple layer (normal-map only, fragment shader) ----------
    // Not part of heightmap. Fragment perturbs normal with high-freq oriented noise.

    return SampleOut(base_h + h, dh /* analytic derivatives accumulated above */);
}
```

with the **morphology selector** being a soft-blend over morphology-specific generators:

```
fn select_dune_morphology(...) -> Layer {
    // Three primary generators, each producing height + analytic gradient:
    let H_uni  = transverse_or_barchan(...);     // cosine + asym ridge, or Worley crescents
    let H_bi   = linear_seif(...);               // along W_resultant + cross-wind warp
    let H_star = star_dune_voronoi(...);         // Worley cells with N-arm template

    // GBNT-driven blend weights from regime scalar:
    let w_uni  = max(0.0, 1.0 - 2.0 * regime);
    let w_bi   = 1.0 - abs(2.0 * regime - 1.0);
    let w_star = max(0.0, 2.0 * regime - 1.0);
    return w_uni*H_uni + w_bi*H_bi + w_star*H_star;
}
```

### F.1 The three morphology generators

**Unidirectional generator** — for each scale, evaluate phase along W, asymmetric-ridge it, modulate amplitude by a sand-supply mask. For sand-poor regions, multiply by a Worley `(1 - F1)` mask so ridges break into discrete crescents → barchans. For sand-rich regions, keep the continuous ridge → transverse dunes.

**Bidirectional generator** — compute the GBNT-optimal crest direction `n*` from the two wind vectors (closed form when there are exactly two: `n*` bisects the *complements* of W₁, W₂ — see Rubin & Hunter). Evaluate phase along `n*`. Add cross-wind warp by an fbm sized to the wavelength to introduce sinuosity and Y-junctions. For Y-junctions specifically, modulate the wavelength by a low-frequency multiplicative noise so adjacent crest spacings drift; merging happens automatically where two close crests share a half-wavelength shift.

**Star-dune generator** — Worley cells; at each cell center `cᵢ`, place a peak with `M=4 ± 1` arms at angles `θⱼ = 2πj/M + jitterᵢ`. Height contribution from cell `cᵢ` to point `p` is `H_star · max(0, 1 − dᵢ/Rᵢ) · max_j cos(angle(p-cᵢ) - θⱼ)^k` where `k` is an arm-sharpness exponent. Arm angles can be biased toward the local set of wind directions, giving "real" star dunes whose arms point toward the dominant winds.

### F.2 Computational cost estimate

For a typical pixel evaluation at LOD that includes both layers, the approximate work is:
- ~6 fbm taps for warp + regime fields,
- 1 Gabor or oriented-noise eval per scale (2 scales geometric),
- 1 Worley 3×3 lookup for morphology selector when `regime > 0`,
- ~30–60 ALU ops for shaping and blending.

That fits in a fragment-shader budget. Pre-baking the *parameter* fields (`W`, `A`, regime) into a low-resolution cubemap (256² per face) costs almost nothing per frame and removes the cyclone-potential evaluation from the hot path.

### F.3 Why this design

- **Physical plausibility comes from the GBNT-driven morphology blend** — at every point on the planet, the morphology is the one a real wind regime would produce.
- **Visual fidelity comes from the asymmetric ridge function applied late in the pipeline**, after warping and morphology selection — that is what gives the brink line and slip face their convincing signature.
- **The two reference images** are produced by the same code with different parameter fields: Namib = high `H_draa`, sharp `A` boundary, mid-range `regime` (some bi, some multi); golden barchan field = `H_draa = 0`, low regime, sand-poor mask, vegetation at margin.
- **Animation is decoupled from generation**: phase advection makes dunes migrate, slow parameter updates evolve the field, but the per-frame heightmap evaluation remains pure-procedural and stateless.
- **The architecture mirrors what production VFX uses for *Dune* (DNEG/SideFX article)** — a static heightfield generator plus optional sim-driven displacement — but compresses it into a real-time function suitable for evaluation on a sphere.

---

## Caveats

- **Gabor noise cost.** A high-quality Gabor evaluation can be 30–100 ALU per pixel depending on impulses-per-cell. For a slower GPU target, drop to oriented sin(phase + warp) — visually weaker but 5–10× cheaper. Lagae's STAR (2010) gives concrete cost numbers; Galerne et al. (2012) give a quantization scheme that helps.
- **Sphere wind-field design is open-ended.** The Hadley/Ferrel zonal model is a starting point; for game-relevant planets you may want artist-painted overrides per planet, or more sophisticated stream-function-based fields (vector spherical harmonics; cf. Le Gia et al. 2020, FaVeST). Make `W(p)` a swappable function.
- **GBNT closed form for bidirectional wind** is a useful approximation, not exact — the real "fingering" vs. "bed-instability" mode bifurcation (Courrech du Pont et al.) means the same wind regime can produce different orientations depending on sand availability. The proposed regime blend captures this only approximately.
- **The "dune solver" used by DNEG on *Dune* is a Werner-style cellular automaton in OpenCL** — explicitly the kind of physical sim you wanted to avoid. Nothing in the published film/game literature does what you're proposing in real time at full fidelity. The closest art-team analog is Houdini's `terrain_dunes` shelf tool, which is essentially the asym-ridge + warp recipe outlined here, baked offline.
- **Branching/Y-junction realism is the weakest part of any procedural approach.** Real Y-junctions arise from wavelength coarsening over time (Andreotti et al. 2009 on pattern coarsening). The wavelength-modulation hack proposed here gives the right *look* but not the right *statistics*; if statistical realism matters, do a one-time bake using the Houdini Dune Solver or a Werner CA, store the result as a sparse displacement texture, and use the live procedural function only for detail amplification.
- **Crest-sharpness anti-aliasing.** The C¹ discontinuity at the brink is precisely what aliases. Use the LOD-driven `alpha_eff` blending described in §C.3 and screen-space derivatives (`dpdx`, `dpdy` on the phase) to bandlimit the asymmetric ridge — otherwise distant dunes will shimmer.
- **Phase-animation continuity.** Pure phase advection eventually pushes the noise field through one full wavelength, at which point the pattern repeats. Periodic reseeding (every ~5 wavelengths) with a smooth crossfade hides this but can be visible if not carefully tuned.
- **Two reference images vs. parameter dialing.** The Namib satellite view is at draa scale (kilometers across); the golden aerial is at dune scale (hundreds of meters). Both have to come out of the same generator at different camera distances, which is exactly why the multi-scale layered design is mandatory rather than cosmetic.