# Atmospheres, Oceans, and Lighting

The unified spec for everything that sits between the camera and the
solid surface: gas-giant materials, rocky-body atmospheric scattering,
clouds, ocean rendering, and image-based lighting (IBL / reflection
probe).

What runs today: gas-giant materials, Thalos's shared `BodySky` raymarched
rocky-body atmosphere, Kòrsou's Bevy Earth atmosphere adapter, one dedicated
near/orbital cloud composite, two ocean spatial adapters over shared wave
mechanisms, and a CPU-painted reflection probe. Ocean and clouds remain
dedicated composites with explicit ordering around Thalos's sole planetary
atmosphere renderer.

## Status

| Area | Today | Future |
|---|---|---|
| Gas / ice giants | `GasGiantMaterial` + `atmosphere_gen::AtmosphereParams`: cloud deck, haze, rim halo, optional Rayleigh blue gap. Storm + aurora layers stubbed. | Navigable upper atmospheres with one exterior/interior weather authority, physical flight envelope, storms, and deeper layers; see [navigable_gas_giants.md](../world/navigable_gas_giants.md). |
| Rocky-body sky | **The shared `BodySkyMaterial` raymarch is canonical** (ADR-20260721T185221Z-custom-rocky-atmosphere). It reads the authored `AtmosphereBlock`, renders for every resident terrain view, and clips against shared opaque scene depth. `earth-reference` and `runway-atmosphere` are the orbital and surface regression probes. | Tune the orbital limb inside the shared optical model; complete terrain/atmosphere radiometric exposure unification without restoring a second renderer. |
| Local planar sky | Kòrsou consumes `thalos_atmosphere::add_bevy_earth_atmosphere` with a `0.45` density calibration. A Curaçao/date-aware solar clock drives the same directional light used by the atmospheric sun disc, cascaded shadows, generated environment map, terrain, and water reflection. Bevy supplies aerial perspective and standard-path PBR environment lighting. Interactive time/date/rate controls live on F10 World; capture accepts deterministic `--time HH:MM`. | Add another concrete adapter only when a local application has different demonstrated needs; do not turn the planetary and planar implementations into a universal backend trait. |
| Cloud rendering | **Dedicated cloud path:** the body-fixed compute march exports premultiplied radiance/transmittance plus hit depth; `CloudCompositeMaterial` is the sole near/orbital screen compositor and samples the same per-body weather field as `SolidPlanetMaterial`. Cloud lighting currently uses its analytic projection of the shared atmospheric coefficients. Gas-giant decks remain distinct. | Bind the shared Thalos sky/transmittance LUT explicitly while completing foreground/background atmosphere ordering, then shared cloud shadows and environment response. |
| Oceans | `thalos_ocean` supplies one authored sea-state/wave mechanism to two adapters: Thalos's analytic-sphere `BodyOceanMaterial` with signed-field coverage and custom atmosphere optics, and Kòrsou's displaced planar clipmap with real-world coast textures and Bevy PBR. See [ocean.md](ocean.md). | Dynamic spectral displacement/Jacobian cascades, persistent foam, and bounded local shore/wake solvers; Thalos keeps its analytic planet-scale surface while local adapters may use bounded geometry. |
| Reflection probe | CPU painter: 256²×6 cubemap, change-gated at 2 s real / 0.5 s under fast warp, with sun disc + Lambert planet hemisphere + dim starfield. A detached `GeneratedEnvironmentMapLight` producer filters it; a craft-local, specular-only `LightProbe` consumes the specular output at the canonical photometric scale and cancels inherited craft attitude so the world-authored cubemap stays world-aligned. | Real-scene cubemap capture once Bevy supports omnidirectional cameras (PR #13840), or self-implemented if it bites. **Not a Phase-1 priority.** |

> **Rendering vs physics.** This doc covers atmosphere *rendering* (how the sky
> and aerial perspective look). The *physical* atmosphere — air density vs
> altitude, and the aerodynamic drag/lift forces it produces — is a separate
> concern in `docs/simulation/aerodynamics.md`. They share one authored boundary, the
> `karman_line_m` on `TerrestrialAtmosphere`, but the physical profile
> (`TerrestrialAtmosphere::sample_at_altitude_m` → density / pressure /
> temperature / speed-of-sound) feeds forces, not shading.

### Canonical rocky-body render path (2026-07-21)

`BodySkyMaterial` is the sole rocky-body atmosphere projection. The game
spawns one body-centred fullscreen material for each terrestrial atmosphere;
the unified render-LOD system makes it visible whenever that body's resident
terrain is visible. Its `AtmosphereBlock` is built from the same authored
Rayleigh/Mie parameters consumed by the CPU `SkyViewLut`, terrain lighting,
ocean, and cloud analytic coupling.

The material raymarches in camera-relative planet space and samples the shared
opaque scene-depth copy, so terrain, structures, and craft clip one continuous
air segment without a second camera-local atmosphere entity. The removed Bevy
path required an extracted f32 proxy plus separate layout/suppression rules and
proved washed-out, incomplete, and distance-unstable in matched and live views
(ADR-20260721T185221Z-custom-rocky-atmosphere).

`just screenshot earth-reference` supplies the fixed 3:2 low-orbit regression
frame. `just screenshot runway-atmosphere` is the complementary low,
near-horizontal surface probe for sky, long slant-path haze, and terrain/
structure recession. There is no atmosphere backend environment override,
comparison axis, live selector, or persisted setting. Existing settings files
may still contain the removed `legacy_body_sky` key, which Serde ignores during
migration.

### Shared authoring, explicit spatial adapters (2026-08-08)

`thalos_atmosphere` is the reusable leaf. It owns the authored-to-GPU
`AtmosphereBlock` projection used by Thalos and the concrete Bevy Earth adapter
used by local planar applications. It does not own camera placement, scene
depth, planet composition, or application runtime state.

The adapters intentionally differ:

| Adapter | Owns | Why it stays distinct |
|---|---|---|
| Thalos planetary | `BodySkyMaterial`, camera-relative shell raymarch, opaque-scene-depth clipping, multi-body ordering, cloud/terrain/ocean optical composition | A rotating planet and floating origin need analytic shell geometry, render-unit conversion, and body-aware composition. |
| Kòrsou planar | Bevy `Atmosphere::earth`, a `0.45` local density calibration, generated environment lighting, and an application-owned astronomical clock at Curaçao's latitude/longitude | A recentered UTM metre frame can use Bevy's maintained standard-path sky; its long ground sightlines never curve out of the dense lower atmosphere. The adapter, rather than the reusable atmosphere leaf, owns local civil time and the east/up/north projection of the solar direction. |

The density calibration belongs to the Kòrsou adapter, not authored
`TerrestrialAtmosphere`: it compensates for planar geometry rather than
describing a different Earth. This seam lets a future ellipsoid/geoid adapter
reuse authored state without pretending the current two render
implementations are interchangeable. See
[ADR-20260808T221912Z](../adr/20260808T221912Z-atmosphere-and-ocean-mechanisms-use-spatial-adapters.md).

The solar clock is a render input, not atmosphere authorship. It evaluates the
sun from ordinal date, AST civil time, and Curaçao's real coordinates, then
projects that direction into the active local frame. The atmosphere, direct
light, shadow cascades, generated environment map, and ocean may not keep
parallel sun directions. Headless capture freezes its instant so day/night
comparisons remain reproducible.

## Goals

- Read the right way at orbit and in atmosphere. Earth-like worlds
  should look Earth-like; airless moons should keep knife-edge
  terminators; Venus/Ashara should read as oppressive.
- Per-body parameterization. Atmospheric optics are functions of the
  atmosphere variant chosen in the prior, never a hard branch.
- One Thalos body-shading vocabulary. BRDF choice (Hapke /
  Lommel-Seeliger / GGX) is a function of the atmosphere variant, not a
  per-pixel switch. A separate application may adapt the shared authored state
  to a maintained local renderer such as Bevy PBR.
- Cloud and atmosphere assets are authored or inferred at the
  `PlanetTerrainSpec` level so the same schema drives appearance from
  orbit through descent.
- Lighting via the same physical model the renderer already uses
  (`assets/shaders/lighting.wgsl`). New BRDFs slot into that library.

## Non-goals

- Volumetric in-atmosphere flight effects (god rays through cloud
  layers, mist banks). Worth a stretch goal but not in M4 scope.
- Realtime weather simulation. Cloud fields are static reference
  overlays with differential rotation only.
- Spectral simulation, displaced local geometry, and persistent foam in the
  BL-12 visual tracer. They are the next ocean program, not hidden inside the
  initial shader-only fidelity slice; see [ocean.md §6](ocean.md#6-production-path).
- Real-scene reflection probe capture. Deferred.

---

## Gas / ice giants (today)

The `atmosphere_gen` crate holds the *data* definition of a body's
atmosphere — not the renderer, not the shader, not the GPU uniforms.
It is the analogue of `terrain_gen` for the gaseous layer above a
body.

Two sibling schemas:

- **`AtmosphereParams`** — gas / ice giants. The cloud deck IS the
  visible disk; there is no solid surface. Rich schema: cloud
  palette, zonal banding, haze, rim halo, Rayleigh blue gaps, limb
  darkening.
- **`TerrestrialAtmosphere`** — terrestrial bodies with a thin gas
  shell over a baked solid surface. Sparse schema: rim halo + limb
  shading + optional limb darkening. Built to composite over the
  impostor.

Bodies carry one or the other, never both.

### Layer model

A gas giant's visible disk is composited from several optically
distinct layers. First-pass rendering supports the first three; the
remaining layers are wired as explicit stubs so fidelity can climb
without schema churn.

1. **Cloud deck** — the optically thick layer. Defines the visible
   colour at each latitude/longitude.
2. **Haze layer** — mid-altitude particulate layer above the cloud
   deck. Contributes a subtle chromatic shift, softens band edges,
   modulates the terminator.
3. **Rim halo** — upper-atmosphere forward-scattered light visible
   just outside the cloud-deck disk. Approximates a Rayleigh-like
   limb glow via an exponential density falloff with altitude.
4. **Storm features** *(future)* — discrete long-lived vortices like
   Jupiter's Great Red Spot. Will reuse the SSBO pattern from
   `terrain_gen::Crater` so GPU detail-layer code stays structurally
   consistent across body types.
5. **Aurora** *(future)* — polar emission ring. Separate layer so
   its additive blending and magnetic-field alignment can be tuned
   independently of cloud-deck shading.

### Rendering

`GasGiantMaterial` (`crates/planet_rendering/src/gas_giant.rs`)
consumes `AtmosphereParams` and feeds
`assets/shaders/gas_giant.wgsl`. `RingMaterial` is a sibling
(`assets/shaders/ring.wgsl`).

---

## Rocky atmospheres (shipping)

Single-scattering Rayleigh + Mie raymarch, per-channel Rayleigh
(Bucholtz 1995 sea-level coefficients) and scalar Mie
(Henyey-Greenstein phase function). The integral runs **once per
fragment in a single fullscreen pass per body** — the same shader
produces the rim halo, the lit-disk haze, the terminator orange
band, and the surface aerial perspective at every altitude from
orbit to ground.

### Architecture: shared optics, one render path

The atmosphere is a fullscreen quad per body (`BodySkyMaterial` in
`crates/rendering/render/src/ground/sky_material.rs`, shader at
`crates/rendering/render/src/ground/body_sky.wgsl`) while the real terrain LOD is
visible. It renders in `Transparent3d` with `depth_compare = Always`
so it rasterizes on every pixel, then clips the raymarch with two
intersections:

- **Atmosphere shell** — `radius + karman_line_m`. Rays missing the
  shell `discard` (cheap early-out covers most off-body pixels).
- **Scene depth** — sampled from a per-frame copy of the main pass's
  depth attachment (see "Scene-depth coupling" below). Clips the
  raymarch at terrain, the impostor body, the ship hull, or any
  other opaque geometry.

The ship-view LOD split is:

1. **Far** (impostor visible at ≥ 4× radius). The impostor shader
   composites surface transmittance, in-scatter, and clouds inline;
   its paired halo material draws the shell outside the solid sphere.
2. **Mid** (camera outside karman shell, terrain visible). `BodySky`
   is active even though the camera is in space, so terrain gets the
   same in-front atmospheric haze and a fixed-altitude reference cloud
   shell instead of a naked terrain disk with only a rim halo.
3. **Near** (camera inside karman shell). `BodySky` ray segments
   either terminate at terrain (aerial perspective on terrain) or exit
   the shell into the void (sky color).

The terrain surface shader itself still owns only surface-side light:
direct sunlight (rough-dielectric Oren–Nayar + Cook–Torrance GGX for wet
vegetated bodies like Thalos — see the BRDF table below; the impostor still
uses Hapke for airless bodies, pending a unified per-material shading
dispatch), eclipse/craft shadow factors, and a small atmosphere-derived
sky-diffuse fill computed from the bound `AtmosphereBlock`. `BodySky` owns
camera-path haze/transmittance/clouds.
That split keeps airless terrain vacuum-black while preventing daylight
terrain under a blue atmosphere from turning pure black on sun-opposed
nearby slopes.

### Draw order: the composites are pinned, not sorted

The atmosphere, the analytic ocean, the cloud composite and the celestial
backdrop all ride `Transparent3d`, which sorts by the view-space depth of each
mesh's centre. **That sort is meaningless for a fullscreen pass and must never
be relied on.** A composite parented to the body has the planet centre as its
sort point, and the sort key is not the distance to that centre — it is the
*projection of the offset onto the view axis*. Standing on the surface it
collapses through zero and, for any camera pitched above the horizontal, goes
positive (the centre is behind the eye), which drops the composite to the very
end of the phase where it paints over world transparency. That erased engine
plumes against the sky for as long as the rule was geometric
(INC-20260725T185440Z-plume-erased-by-the-sky).

The stack is therefore declared, in `thalos_body_render::composite_order`:

```text
celestial backdrop → atmosphere → ocean → clouds → world transparents
```

Each pass claims a slot as a `Material::depth_bias` (Bevy folds that straight
into the sort key) far enough out that no camera orientation can reorder the
stack or lift a composite past ordinary transparency. **A fullscreen composite
must claim a slot there — never bias 0.** The backdrop sits at the far end
precisely so the air is drawn over it: that is what lets the atmosphere's
`(1 − alpha)` transmittance perform the per-pixel star crush.

`just screenshot plume-skyline` is the regression probe — it holds the camera
pitched above the horizontal, the only regime in which the ordering can fail.

### Why this approach (not Bruneton)

Bruneton 2008 is the canonical "right" answer for high-fidelity
in-atmosphere flight (KSP2 / Scatterer / O'Neil GPU Gems), but its
4D LUT precompute is overkill at our current fidelity. The
single-scattering raymarch with per-pixel sample jitter delivers
~95% of the visual quality at a fraction of the implementation cost
and zero load-time precompute. When fidelity warrants it, the
`integrate_atmosphere` helper is the swap point: replace its body
with LUT lookups, leave the call sites untouched.

### Scene-depth coupling

WebGPU forbids sampling the live depth attachment from a fragment
shader, and our forked `thalos_udlod` does not queue into
`Opaque3dPrepass` (the standard prepass-depth path), so the built-in
prepass-depth texture is terrain-blind. The shared workaround lives in
`crates/rendering/foundation`:

- A `SceneDepthImage` resource owns a `Handle<Image>` whose GPU
  texture (`Depth32Float`, `COPY_DST | TEXTURE_BINDING |
  RENDER_ATTACHMENT`) is sized to the selected camera viewport.
- `SceneDepthView` marks the application-selected 3D camera; the foundation
  extracts that marker without importing `ShipCamera` or another application
  type.
- The `copy_scene_depth` render system runs between Bevy's main opaque and
  transparent passes and issues `copy_texture_to_texture` from the main pass's
  `ViewDepthTexture` (which has terrain depth written by then) into
  the Image. Under MSAA it instead runs a depth-only fullscreen resolve that
  copies sample 0 into the same single-sample image. Source and destination
  must match in size; a skipped copy leaves empty depth and the sky paints
  opaque air over every pixel above the geometric horizon
  (INC-20260817T014132Z). Laptop 0.50× 3D is the usual hang — the scale must
  land on `ExtractedCamera` in `PrepareViews`, after extract, not in
  `ExtractSchedule`.
- `BodySkyMaterial` binds the Image as `texture_depth_2d` at
  `@group(3) @binding(2)`; `body_sky.wgsl` reads it via
  `textureLoad` and reconstructs view-space distance with
  `view.view_from_clip`. When a depth hit exists, that reconstructed
  distance is the authoritative surface clip; the analytic mean-radius
  sphere is only a fallback for pixels with no opaque depth. Ground LOD
  relief can protrude over the reference-sphere horizon, and clipping it
  to the fallback first produces a dark horizon band.
- The same material binds the reference cloud cubemap at bindings
  `3/4`; the shader intersects a fixed-altitude cloud shell before
  scene depth so terrain LOD receives the same detached cloud layer as
  the impostor path. The cloud overlay is gated to body/terrain-hit
  rays and faded in across the geometric horizon, avoiding a visible
  fixed-altitude cloud-shell tangent band on sky-only pixels.
- The ship camera uses `scene_depth_view_texture_usages()`
  (`RENDER_ATTACHMENT | COPY_SRC | TEXTURE_BINDING`) so both paths are legal.
  `ShipCamera` remains separately extracted for Thalos-specific SSAO and contact
  shadows; the shared depth pass filters only on `SceneDepthView`, which the
  map camera and light/shadow views do not carry.
- **Near/far geometry classification is shell-segment membership, never a
  fixed distance** (INC-0003). A depth hit counts as "this body's surface"
  iff it lies inside the ray's atmosphere-shell segment (`t_scene <=` the
  shell far exit); anything past the shell exit is a background celestial
  body and gets the sky-pixel treatment (full-column in-scatter + the
  perceptual luminance opacity boost that crushes stars/daylit moons). A
  distance cutoff encodes a camera-altitude assumption that orbital views
  violate — the old `atmos_top_r * 4` cutoff turned the planet's own
  continents pure black beyond ~13,000 km while the analytic ocean stayed
  lit.
- **The analytic ocean's coverage/colour are direct samples of the one signed
  sea-height field — never a depth comparison, at any range** (ADR-20260720T185958Z-water-projects-one-signed-sea-field,
  superseding ADR-20260720T185957Z-coastline-as-authored-data's range-blend). The ocean branch samples signed sea
  height at the sphere-hit direction from a resolution cascade of the same
  field: the resident **udlod height-tile atlas** first (a WGSL port of the
  tile-tree walk, capped at the pixel's footprint LOD and mip-sampled within
  the tile — the exact texels the visible terrain mesh is displaced from;
  bound at material bindings 11–14 via `BodySkyMaterial`'s manual
  `AsBindGroup`, keyed by its `terrain_entity` field), then the per-body
  **coast/bathymetry cube** (`BodySkyMaterial::coast_atlas`, baked once at
  spawn by `bake_coast_bathymetry_cube` from the same `SurfaceQuery` surface)
  as the coarse tail — cold streaming, terrain despawned, beyond the impostor
  swap. Same field at two resolutions = no authority crossfade and no seam;
  the field's zero crossings are LOD-invariant (relief never crosses sea
  level), so the waterline cannot move with camera distance or streaming
  state. Coverage is a band around the zero crossing sized by the *sampled
  texel* (0.75 m wet-edge floor, never a range-scaled error model); colour is
  the field's bathymetry over the slant path. Scene depth's only remaining
  job is occlusion by resolvable geometry: terrain occludes via the *field's*
  height at the scene-hit direction (footprint-scaled thresholds), and
  non-terrain geometry (craft, structures) occludes when it stands in front
  of the water surface by more than a footprint-scaled margin. Never derive
  water coverage or colour from scene depth again — the depth-compare
  architecture regenerated coast speckle / translucent-wash artifacts three
  times (INC-0003, BL-5, BL-8) before ADR-20260720T185958Z-water-projects-one-signed-sea-field removed it.
- **Shore interaction is keyed on the same field** (BL-10, tier 1 —
  MSFS-class, normals + albedo only, no displaced geometry): near shore the
  sky pass takes two extra field taps for the tangent gradient, giving
  `shade_ocean` the vertical depth, distance-to-waterline, and shoreward
  direction. From those: wave shoaling (chop calms as depth → 0), breaker
  swell ridges + foam stripes whose phase is a function of *shore distance*
  (crest lines parallel to the beach by construction — refraction for free —
  marching shoreward with time), and a churned swash edge in the last metre.
  All of it fades out past ~9 km view distance; the map/impostor callers pass
  far sentinels and are untouched. Tier 2 (advected breaker fronts with
  crest shaping + foam decay trails, Sea-of-Thieves-class) is the follow-up
  seam.

Cost: one fullscreen depth copy per frame. Trivial on M2 Pro at
typical resolutions.

### Authoring (`AtmosphericScattering` + `TerrestrialAtmosphere`)

The Kármán line lives on the parent `TerrestrialAtmosphere`, not on
the scattering substructure — it's the rendering integration cutoff
AND the gameplay boundary for drag / heating / "in atmosphere"
state, independent of whether scattering is configured.

```ron
terrestrial_atmosphere: Some((
    // Single source of truth for atmosphere top. Replaces the old
    // implicit `5 × max(rayleigh, mie)` default that capped Earth at
    // ~40 km. Author generously — well above 5 × scale-height so the
    // raymarch captures the full column.
    karman_line_m: 80000.0,
    scattering: Some((
        vertical_optical_depth: (R, G, B),  // Rayleigh τ_v at zenith.
        rayleigh_scale_height_m: 8000.0,
        mie_optical_depth: 0.021,           // scalar (white).
        mie_scale_height_m: 1200.0,
        mie_asymmetry: 0.76,                // Henyey-Greenstein g ∈ [-1, 1].
        strength: 1.0,                      // artistic multiplier; 0 disables.
    )),
    clouds: ...,
    limb_darkening: ...,
))
```

Earth-like Rayleigh sea-level (Bucholtz 1995): `(0.046, 0.108,
0.264)`. Dust-loaded atmospheres invert the slope (red dominant —
see Vaelen). Mie: clean 0.02, hazy 0.10, dust storm 0.30+. Karman
line: Thalos 80 km, Pelagos 90 km, Vaelen 60 km.

### Visual targets achieved

- ✓ **Continuity from orbit to ground.** One shader, one integration
  path, no LOD seams between sky and terrain.
- ✓ **Soft terminator** with red/orange wedge from Rayleigh sun-column
  attenuation.
- ✓ **Limb glow extends beyond the geometric edge** — graze rays
  through the atmosphere shell.
- ✓ **Aerial perspective** on terrain and impostor body via
  scene-depth clipping, **decoupled from sky-dome brightness** (see
  *Aerial perspective vs sky-dome brightness* below).
- ✓ **Mie forward-peak haze** on the night-side rim — warm crescent
  on back-lit bodies.
- ✓ **Knife-edge terminator** preserved on airless bodies (Mira,
  Selva, asteroids — `karman_line_m == 0` or absent
  `TerrestrialAtmosphere` skips both spawn and raymarch).

### Aerial perspective vs sky-dome brightness

The `BodySky` pass produces both the **sky dome** (in-scatter along view
rays that miss the surface) and the **aerial perspective** (the airlight
in-scatter added on top of terrain/geometry pixels, attenuated by the same
integral's transmittance). Both come from one `integrate_atmosphere`
result, so they share the authored `strength` / `multi_scatter_gain`.

That sharing is a trap. Those knobs are tuned so the **sky dome** reads
bright and crushes stars at midday (Thalos sits at `strength: 3`,
`multi_scatter_gain: 3`). But applied unchanged as airlight, the in-scatter
is far brighter than the surface even over short paths, so the ground
washes into a uniform veil at *any* altitude — well above 400 m you can
barely see it, and nearby relief reads as fog. Crucially this is **not**
extinction: per-channel β is already Earth-clear-day correct (~50–90 km
Koschmieder visibility), and zeroing the in-scatter (`strength → 0`) leaves
the ground perfectly crisp under a black sky. The wash is purely the
*additive* airlight, and it is over-bright even at the physically "correct"
`strength: 1` (verified on Pelagos) — short-path airlight (especially the
altitude-dependent multi-scatter fill, strong where air is densest near the
ground) simply over-contributes relative to surface albedo in the engine's
flux units.

The fix decouples the two. `crates/runtime/game/src/rendering/ground_terrain.rs`
holds an `AtmosphereTuning` resource (`#[reflect(Resource)]`)
with an **absolute** `aerial_perspective_strength` (default `0.10` =
clear weather). Each frame the per-body shader multiplier is computed as
`aerial_perspective_strength / effective_sky_strength` and passed in
`BodySkyExtra::cloud_band_radii.z`; `body_sky.wgsl` scales the in-scatter
on surface/geometry-hit pixels by it (blended by `surface_fade` so it eases
to full sky-dome strength across the horizon — sky pixels are untouched).
Dividing by the sky strength makes the ground airlight land at the same
absolute strength on every body regardless of how bright its sky is
authored, while staying physically proportional to that body's β (thicker
atmospheres still haze more). Transmittance/extinction is left untouched,
so distance still fades contrast the same way — only the additive veil is
dimmed.

**Aerial veil driver: air mass, not Euclidean distance.** The artistic
distance-haze veil in `body_sky.wgsl` (the additive-strength + opacity boost
that lets *distant* terrain desaturate toward the haze colour, on top of the
absolute airlight above) is keyed on the **view-path optical depth** the
integrator already accumulates — `view_tau = -log(mean(transmittance))` — not on
raw camera→surface distance. Distance is a fine proxy for air mass *inside* the
atmosphere but decouples from it the moment the camera climbs out: from orbit
every surface pixel is hundreds of km away, so a distance-keyed ramp saturated
to full veil across the entire disc and washed even the crisp nadir (the terrain
"looks super washed out from orbit" bug). Optical depth is the physically honest
driver — a thin vertical column at nadir-from-orbit (near-clear), a long slant
column at the limb or along a low horizontal flight path (full veil). The tau
thresholds (`aerial_tau_near ≈ 0.30`, `aerial_tau_far ≈ 2.40`) are calibrated to
the old distance ramp *at sea level* (its 8 km onset / 70 km full-veil paths), so
the on-surface look is unchanged and altitude simply re-grades it correctly.

**Column consistency: the low-τ half of the veil.** The two paragraphs above left
a gap that read as "the atmosphere is too clear at high altitude". The airlight
ratio scales only the **additive** in-scatter; the dst-attenuation opacity stays
physical (`1 − T`). Below `aerial_tau_near` the artistic veil is off, so a surface
pixel there gets the physical `1 − T` of its own radiance *removed* and only
`aerial_perspective_strength / sky_strength` of the airlight that should replace
it handed back. Looking straight down from orbit through one full Thalos column
that is a ~15% dim against ~1/30th of the light returned: orbital land came out
**darker and more saturated**, the opposite of aerial perspective, and stayed
crisp to within a hair of the limb where ISS reference photos show a blue veil
developing over most of the disc. (The `aerial` ramp itself was never
inconsistent — it feeds *both* the in-scatter scale and the opacity. Only the
constant floor underneath it was.)

`body_sky.wgsl` now lifts that floor from the ground-calibrated clear-weather
value toward `column_airlight_exposure / atmos_geom.z` — `atmos_geom.z` being the
artistic sky-dome inflation that exists only to crush stars, so dividing it out
lands on the airlight the in-scatter would carry at `strength: 1`.

The lift is driven by **Rayleigh air mass**, not total τ, and that distinction is
load-bearing. The sea-level calibration is aerosol-dominated — an 8 km horizontal
path crosses ~6.7 Mie scale heights but only one Rayleigh one — while an orbital
ray barely crosses the 1.2 km Mie layer at all. Keying the lift on total τ would
therefore re-haze exactly the ground distance the Mie cut (0.06 → 0.025) was made
to keep crisp, and re-haze it with **grey** aerosol veil, which is what washed it
to a grey-tan band in the first place. The two columns separate cleanly out of the
**chromatic spread** of the transmittance the integrator already returns: Mie is
spectrally flat, so it cancels exactly in a channel difference and what survives
is pure Rayleigh. `rayleigh_air_mass = (τ_b − τ_r) / (τ_v,b − τ_v,r)`, where
`1.0` = one vertical column = nadir from orbit. The onset sits at a sixth of a
column so a ground observer's near field is untouched, and the far end well past
one column because an oblique orbital frame spans ~1–4 columns — a nearer ceiling
pins most of the frame at one value and reads as a flat wash instead of depth.

`column_airlight_exposure` is the same class of fudge as the two below: the
ground calibration's own 0.10 correction is far too aggressive once the column is
thick (applying it is what produced no veil at all), while the uncorrected value
over-veils orbital land to a flat blue-grey with no biome colour left. It is
screenshot-calibrated against ISS reference framings and retires with F1/F2.

**Impostor path.** The distant billboard (`solid_planet.wgsl`,
`SolidPlanetMaterial` body pass) previously added the in-scatter at full sky-dome
`strength`, so a terrestrial planet washed pale-blue from map/orbit distance
(continents vanished); cutting it to near-zero then made the planet read airless
(a hairline rim over a crisp disc). It now (1) runs the **full multi-scatter
integral** — `integrate_atmosphere_multiscatter` with the same per-body `ms_lut`
`BodySkyMaterial` binds (scale-invariant, so the SHIP_SCALE bake is reused for
the MAP_SCALE disc; airless bodies bind a 1×1 blank the gate never samples) — so
the *diffuse* second-order blue fill that makes a planet-from-space look
atmospheric is physical, not faked; and (2) keeps a `DISC_AIRLIGHT_FRACTION`
(shader const, `0.15`) of that in-scatter as the single overall airlight dial,
leaving physical transmittance and the separate rim-halo pass untouched. The
in-scatter is air-mass graded by chord length, so the sub-observer point stays
subtle while the limb glows — a real blue veil over the whole disc.

`DISC_AIRLIGHT_FRACTION` (and the ground's `aerial_perspective_strength`) are
still fudge factors: they exist only because surface radiance and airlight aren't
in a consistent exposure/flux scale (the in-scatter over-contributes even at the
physically "correct" `strength: 1`). The **F1/F2 unification foundation** (one
flux, one exposure) is what lets the physical `strength ≈ 1` be correct at every
altitude and retires the fraction entirely. Reference lineage (KSP2, KSA /
Blackrack's Scatterer): Bruneton/Hillaire precomputed scattering — transmittance
+ multi-scatter + sky-view + aerial-perspective LUTs, geometry-aware
(altitude/view-zenith/sun-zenith), never distance fog. This project already runs
the Hillaire-lite single + multi-scatter machinery on both the ground and (now)
the impostor.

This is the **clear-weather visibility knob weather will later drive**
(lower = clearer, higher = hazier/humid); it is the natural seam for a
future per-region visibility field. `AtmosphereTuning` also carries
`strength` / `multi_scatter_gain` dev overrides (sentinel `< 0` = keep the
authored value) for live sky-dome tuning — both are pure runtime
multipliers that do **not** feed the multi-scatter LUT bake, so overriding
them at runtime is exact (no LUT rebake).

### Object aerial recession (foliage/objects fade earlier than terrain)

`BodySky` applies aerial perspective to **every** opaque pixel uniformly,
keyed on scene depth — so trees, grass, impostors, and buildings already get
the *same* haze as terrain at the same distance (they are opaque geometry on
`SHIP_LAYER` that writes depth before the foundation copy pass). But surface
**objects** read more saturated / higher-contrast than the ground, so at the
same distance they pop out against terrain `BodySky` has hazed. `BodySky`
can't help — it has only depth and can't tell an object pixel from a terrain
pixel.

So each object material recedes its own lit colour toward the air **in the
shader**, via the shared `object_aerial_recession` helper in
`thalos::lighting` (`shading/shaders/lighting.wgsl`), called at the end of
`tree.wgsl`, `tree_impostor.wgsl`, and `grass.wgsl`. It blends the lit colour
toward the `SurfaceSky` sky-dome radiance by `smoothstep(NEAR, FAR, camera_dist)`,
deliberately starting **closer** than `BodySky`'s ~8 km terrain onset
(`OBJECT_AERIAL_NEAR_M` ≈ 1.5 km). The haze target is **clamped to a small
multiple of the object's own luminance** (`OBJECT_AERIAL_BRIGHTEN_CAP`) — the
analytic sky radiance runs several × brighter than lit foliage, so fading
straight toward it blows distant canopies out to white instead of hazing them;
the cap makes it read as haze (desaturate + gentle bluish lift). `MAX` is kept
low (≈0.32) so objects fade only a touch more than terrain, never a full
dissolve, and the ramp is stretched over a long distance (NEAR ≈ 1 km → FAR
≈ 35 km, well past the tree band) so the haze builds up gradually instead of
piling into a narrow transition band that reads as an abrupt "haze line". The
`OBJECT_AERIAL_*` consts are the tuning dials (tune from a
`just game` surface screenshot — the `just preview` camera sits inside NEAR and
shows no effect).
This stacks on top of the `BodySky` veil past 8 km, which is intended (objects
recede *more/earlier* than terrain). Buildings (`StandardMaterial`) don't get
this yet — they'd need an `ExtendedMaterial` to reach a fragment hook; deferred.

### Backlog

- **Unify impostor and terrain cloud shadows.** `planet_impostor.wgsl`
  still owns its receiver-side cloud shadow probe and water BRDF. The
  dedicated cloud compositor matches the visible layer beside the atmosphere,
  but terrain cloud shadows remain deferred until the ground path
  has an explicit receiver-side shadow term.
- **MSAA path.** Currently `Msaa::Off` on the ship camera —
  `copy_texture_to_texture` requires matching sample counts, and
  binding a multisampled depth as `texture_depth_2d` would force
  every consumer onto a `MULTISAMPLED` shader-def fork. Re-enable
  with a resolve pass + `texture_depth_multisampled_2d` if jaggies
  become the bottleneck.
- **Ozone absorption.** Two extra parameters (band altitude + per-
  channel absorption coefficient); per-sample multiplier on
  transmittance. Produces Earth's blue-purple twilight wedge.
- **Multi-scatter approximation.** Below visual threshold for
  orbital views. Matters at low altitudes if ground-scattered light
  becomes noticeable.
- **Sample-count LOD.** 8 view × 6 sun samples is comfortable on
  M2 Pro; tune if budget bites. Apparent body size is the natural
  LOD knob.
- **Bruneton swap.** When in-atmosphere flight fidelity demands it,
  replace the raymarch body with LUT lookups; call sites stay
  unchanged.

---

## Cloud rendering (M4)

> This section specifies the implementation that ships today. The future
> architecture, sequencing, and acceptance matrix live in
> [clouds.md](clouds.md); keep future rationale there rather than growing a
> second plan in this file.

### Today: canonical weather plus body-render cloud projections (CLOUD-1)

The shipping near-cloud mechanism is a vendored fork of
`bevy-volumetric-clouds` (MIT, evroon), absorbed—with its upstream license—at
`crates/rendering/render/src/clouds/`. It is reworked around Thalos's spherical,
`big_space`, dual-camera engine. The
raymarch runs entirely in the **body-fixed frame** of the active cloud
body, so clouds are planet-fixed: glued to the ground, co-rotating with
the surface, horizon-correct at any altitude and at the limb. Three
stages:

1. **Generation + raymarch (compute, `clouds_compute.wgsl`).** An `init`
   pass builds a Perlin-Worley 2-D atlas + a 3-D Worley volume (the HZD
   noise set); an `update` pass raymarches the cloud shell every frame
   into a 1920×1080 RGBA32F texture (rgb = premultiplied in-scatter,
   a = transmittance) plus an R32F **nearest cloud-hit distance** texture
   for the composite's depth occlusion. Density = atlas base shape eroded
   by the Worley detail, shaped by a vertical profile, gated by the
   planet-fixed coverage map (below); lighting is a dual-lobe
   Henyey-Greenstein phase + a Beer self-shadow march. Shell intersection
   is a true ray-sphere from the camera's actual body-fixed position, and
   the march uses a **bounded world-space step** (not a fixed step
   *count*): ~20 samples across a short band crossing, coarsening toward
   `MAX_RAY_STEPS` on long near-tangent segments, plus a distance
   haze-out — so horizon rays can't alias into radial "fountain" streaks
   and steep rays don't show dither moiré. Per-sample work is texture +
   FMA only: coverage and the triplanar weights are hoisted to once per
   ray (planet-scale smooth), and zonal wind advection is folded into
   the body-fixed frame game-side.
   Because body-fixed positions are ~3.2e6 m (f32 lattice ~0.25 m), all
   field sampling is **wrap-first** — positions are reduced into one
   world-space tile period before texel scaling — and the 2-D atlas is
   projected onto the sphere **triplanar** by the local surface normal
   (within one view the weights are constant; the blend only matters
   across the planet, where it avoids polar pinch). Zonal wind advects
   the whole field as a rotation about the body's spin axis.
2. **Body-fixed drive (game, `rendering/clouds.rs`).** `drive_clouds`
   picks the **active authored cloud body** (the nearest body whose
   terrestrial atmosphere has `CloudClimate`; published as the
   `ActiveCloudBody` resource, sole writer) and
   feeds the crate the camera's planet-centred position and view basis
   rotated by the inverse body orientation (`CameraMatrices`), plus the
   body-fixed sun (scene-matched flux) and planet radius. Static
   appearance (coverage/density/scale/heights) is projected from that climate
   (`CloudsConfig` is `Reflect`-registered). A landed/parked
   camera is *static* in this frame, so temporal reprojection converges
   exactly when the view is steady.
3. **Composite (body_render, `cloud_composite.wgsl`).** The cloud texture and
   hit-distance texture are bound to one `CloudCompositeMaterial`, the sole
   fullscreen owner of near-volume and weather-derived orbital clouds. It
   renders after the canonical `BodySky` atmosphere; both take pinned slots from
   `composite_order` (see "Draw order" above), which fixes their order relative
   to each other *and* keeps both behind world transparency. Occlusion against
   opaque geometry (ship hull / terrain) and the analytic ocean ramps
   `cloud_vis` from the **per-pixel nearest cloud-hit distance** to the band exit
   (ray-shell intersection, `cloud_band_radii` in the shared per-body
   parameters). The ocean hit comes from the same stable ray/sphere helper as
   `BodyOceanMaterial`, because the opaque scene-depth copy cannot contain a
   transparent fullscreen pass. This keeps geometry under a sparse deck from
   dimming far-behind clouds and lets the ship cross the cloud boundary without
   a hard pop. `rendering::clouds::sync_cloud_composite_materials`
   binds the live textures on the active cloud body and blank fallbacks
   everywhere else. The current ordering still treats the already-integrated
   atmospheric radiance as wholly behind the cloud; CLOUD-4 owns the explicit
   foreground/background segmentation.

**Weather field (the canonical hook).** `thalos_world::CloudClimate` is the
sole authored terrestrial-cloud configuration; `clouds: None` creates neither
a runtime field nor visible default clouds. At body spawn, it deterministically
produces one body-fixed `CloudWeatherField` in `SolarSystemState`: a seam-safe
RGBA8 cubemap with 256² texels per face (R coverage, G type, B normalized base,
A normalized top). `sync_cloud_weather_map` uploads only when `(body, version)`
changes. A future weather system mutates or replaces this field and increments
`version`; consumers keep the same contract.

The **first orbital projection** in `SolidPlanetMaterial` samples that same
weather cubemap by body-fixed normal and composites a surface-following cloud
layer. It establishes one authority across regimes; CLOUD-6 replaces it with
density-derived optical depth, normals, height moments, and a reduced-detail
limb handoff. The former body-name-selected reference cubemaps and dormant
in-shader `BodySky` slab march are deleted.

**Remaining approximations (deliberate — see Next):**

- The marched reach is capped (`MAX_CLOUD_DIST` ≈ 25 km) with a haze-out,
  so very distant decks dissolve rather than draw.
- Temporal accumulation runs in both regimes — same-pixel when the view is
  steady in the body-fixed frame, depth-validated reprojection through the
  previous frame's camera in motion — so the jittered march converges
  instead of boiling. Residual dither survives only at cloud↔sky
  silhouettes, where the disocclusion test rejects history; a ping-pong
  history buffer + neighborhood clamp is the upgrade if that fringe shows.
- The weather map is static between version bumps — no advection or
  evolution yet.
- The current volume marcher consumes only coverage from the richer weather
  texel. CLOUD-3 applies type/base/top to typed vertical profiles and range LOD.
- The orbital layer is a continuity scaffold, not the final high-fidelity
  orbit representation.

### Planned replacement

The cloud program is decomposed in [clouds.md §4](clouds.md): canonical
per-body state (CLOUD-1, landed), scalable temporal reconstruction (CLOUD-2),
multi-scale density (CLOUD-3), shared atmosphere lighting (CLOUD-4), one-world
cloud transmittance/godrays (CLOUD-5), the orbital projection (CLOUD-6), and
weather/authoring (CLOUD-7). CLOUD-2 is the next implementation slice.

For gas giants, "clouds" *are* the cloud deck and live inside
`AtmosphereParams` already.

---

## Ocean rendering

### Today

The canonical ship-view ocean is the analytic sphere composited by the
dedicated `BodyOceanMaterial`
(ADR-20260720T185954Z-analytic-planet-water-never-meshed and
ADR-20260721T050036Z-ocean-composite-independent-of-atmosphere), with coverage and
bathymetry sampled from the one signed sea field
(ADR-20260720T185958Z-water-projects-one-signed-sea-field). The ocean material samples
four body-fixed scales of one shared mipmapped
broadband slope texture, using the surface pixel's major/minor axes and 16×
anisotropic filtering so grazing views preserve resolvable cross-wave detail.
Mip-omitted slope energy becomes GGX roughness; exceptional resolved slopes
source sparse whitecaps; `thalos::water::shade_ocean_detailed` reflects the
atmosphere-derived sky and sun. Rust supplies f64-reduced wave phase plus
body-local wind axes so sub-metre detail does not swim at planetary coordinates
or floating-origin rebases.

This is the production architecture's shader-only visual tracer. It deliberately
does not claim spectral displacement, persistent foam history, or vessel/local
solver coupling; those stages and their handoff contracts live in
[ocean.md](ocean.md).

### Where ocean lives

Ocean is the renderer's job, but its *topology* is the signed sea-height field
selected by `SurfaceQuery`. The resident terrain atlas and coast/bathymetry cube
are two resolutions of that one field; both ground LOD and the analytic water
projection therefore agree on the shoreline. Future spectral and local fields
modify detail, never coverage authority.

---

## Reflective surfaces / IBL

### Goal and the two tiers

The goal is **gorgeous mirror-finish stainless ships with accurate
reflections** — ambitious, deferred. It splits into two tiers that share one
foundation (the single terrain height authority):

- **MVP (in progress): brushed/satin stainless** via **SSR + a ship-anchored
  reflection probe**. The probe captures terrain *for free* — terrain
  rasterizes normally into the probe faces; SSR adds the on-screen part; and
  brushed-stainless roughness blurs the reflection enough to hide probe
  parallax error and SSR holes. Real stainless (Starship-style) is satin, not a
  flawless mirror, and hulls are curved — both work in the MVP's favor. Keep
  stainless **roughness a parameter drivable toward zero**, and keep the
  reflection source **behind the material interface** so the SSR+probe backend
  can later swap to an RT backend without touching ship materials.
- **Dream: flawless mirror finish** with off-screen-complete, parallax-correct
  terrain reflections. This requires **terrain in the ray-tracing acceleration
  structure (a BLAS)** — there is no shortcut. RT *on the ship only* (terrain
  absent from the BLAS) reflects the sky but **not the ground**, exactly the
  wrong artifact for a grounded stainless vehicle. **Updated 2026-07-24:** the
  prerequisite is now satisfiable without the research project this paragraph
  once described. Terrain renders as ordinary meshes through the tile renderer
  (NTR-X1/X2a), so the BLAS source is those meshes — no collider-trimesh
  extraction, no separate RT geometry, and therefore no raster-vs-RT divergence
  beyond LOD seams. What remains open is cost, not feasibility: BLAS build and
  compaction against the tile streaming rate, and acceleration-structure
  precision under the floating origin. Scoped as **NTR-RT3** behind the
  **NTR-RT1** measurement gate; the shape is fixed by
  [ADR-20260724T224242Z](../adr/20260724T224242Z-solari-scene-half-not-lighting-half.md).
  The "keep the reflection source behind the material interface" contract above
  is exactly what lets that land without touching ship materials.

### Today

Environment-map source for metallic ship-part reflections. A detached
`GeneratedEnvironmentMapLight` producer filters the painted cubemap; a
craft-local Bevy `LightProbe` carries a plain `EnvironmentMapLight`
that shares only the generated specular handle. `ShipPartMaterial`
panels therefore read the sky from the selected ship's orbital vantage
without projecting its planet disc onto every `StandardMaterial` in
the camera view. The cubemap stores scene-flux radiance; probe intensity
derives from the same `LUX_PER_SPINE_FLUX` bridge as direct sunlight,
putting hull reflection and world lighting in one photometric scale.

CPU-authored cubemap, reconsidered every 2 real seconds (0.5 s under
fast warp, with direction and sim-time change gates) from the ship's
current state:

- Cubemap `Image` asset, 256² × 6 layers, `Rgba16Float`, cube view
  descriptor. `TEXTURE_BINDING | COPY_DST` usage — no
  render-attachment, since we're not rendering into it.
- Painter reads ship-to-sun and ship-to-planet directions from
  `SimulationState`, plus the planet's angular radius from its
  physical radius and range. Each frame that hits the refresh tick,
  all 6 faces get rewritten: sun disc (HDR hot spot), lit planet
  disc with a Lambert terminator, dim starfield tint everywhere else.
- **The orbital planet disc is the impostor bake, not a flat tint**
  (2026-07-29). Each texel inside the disc solves the exact ray-sphere
  hit, rotates the resulting normal into the body-fixed frame using the
  rotation the renderer *drew the body with* (`PreciseRotation`, so a
  tidally-locked body needs no special case), and samples the body's
  `ImpostorAlbedo` — the same bake `SolidPlanetMaterial` shows the
  player, held in `ImpostorAlbedoRegistry` so the two cannot disagree.
  A mirror hull in orbit therefore reflects continents, coastlines and
  ocean rather than one blue-grey constant. The flat `planet_color`
  survives as the fallback for solid-colour and degraded bodies.
  Clouds are **not** in the bake, so the reflected planet is a
  cloudless one — the next increment, and the reason this is not yet
  the whole orbital story.

  The hit-normal solve is expressed in units of the planet radius
  (`D/R = 1/planet_sin`) rather than absolute metres, because
  `|oc|² − R²` at low orbit is ≈ 4.9e13 − 4.4e13 and keeps barely six
  significant figures in f32. Verified against an f64 absolute-metre
  solve from 30 km to 30,000 km altitude.
- Mutates the image via `Assets<Image>::get_mut`, marking it changed;
  Bevy's detached runtime filter producer re-prefilters diffuse +
  specular mips downstream.
- The local consumer uses a separate black diffuse cubemap while sharing
  the producer's generated specular handle. The producer keeps its real
  diffuse storage target, and the `SkyAmbient` → `GlobalAmbientLight`
  projection remains the one diffuse-sky authority. This prevents the
  same sky irradiance being added twice inside the local probe volume
  without corrupting the compute filter's output binding.

Under a terrestrial atmosphere the painter blends by altitude (across
the Kármán line) from that orbital model into a **surface sky**: the
lower hemisphere is a warm analytic terrain ground-bounce, and the
**upper hemisphere is the physical `SkyViewLut`** (graphics-fidelity
**F3**) — a CPU raymarch of the *same* single + multiple-scattering
model the terrain shades through (`integrate_atmosphere_multiscatter`,
sharing the `body_render::shading::multi_scatter` primitives), baked
into a small `(azimuth-from-sun × view-zenith)` LUT for the current sun
direction + altitude and sampled per cubemap texel. This replaced a
hand-kept analytic `cpu_surface_sky` that had to be mirrored against the
spine's WGSL `compute_surface_sky` by hand — so the metallic hull and
dielectric structures now reflect the real atmosphere-derived sky, with
no CPU/WGSL drift hazard. `PHYSICAL_SKY_SCALE` (=1.0, the physical
baseline) is the one calibration dial. The multi-scatter LUT the bake
needs is static per body, so it is cached (keyed by body id) and only
the view-dependent sky-view LUT rebakes on a sun/altitude shift.

`SkyAmbient` already projects this same LUT into the flat
`GlobalAmbientLight` bridge for terrain and `StandardMaterial`
diffuse fill. Full directional SH remains future work; the current
craft probe deliberately does not compete with that diffuse authority.

Lives in `crates/runtime/game/src/reflection_probe.rs`; the sky-view LUT
mechanism in `crates/rendering/render/src/shading/sky_view.rs`.

### Why not render the actual scene into the cubemap

The "correct" path — six cameras rendering the real scene into
per-face views of a cubemap — is blocked in Bevy 0.18 by a layering
trap: the main-world `camera_system` resolves
`RenderTarget::TextureView(handle)` against the main-world
`ManualTextureViews` resource every frame and panics if the handle
isn't present. Populating that with real `TextureView`s requires a
GPU `Texture` from `GpuImage`, which only exists in the render
world. Workable paths exist (~200-line custom render-graph node, or
a render-graph subgraph akin to shadow maps) but neither was
justified given the visual target and the fact that ship rendering
was actively churning.

### Migration trigger

Switch to real-scene capture when:

- [Bevy PR #13840](https://github.com/bevyengine/bevy/pull/13840)
  (`OmnidirectionalCameraBundle`, `ActiveCubemapSides` for
  round-robin) merges into a release we can use, **or**
- The painted-planet divergence from the impostor reads wrong at
  screenshot distance (a consistent visual bug, not a one-off).

Craft-local `LightProbe` + specular `EnvironmentMapLight` is the stable
contract. Everything behind that consumer, including the detached
realtime filter producer, can be swapped without touching ship materials.

### Status

Not a Phase 1 priority. Revisit cadence: every 6 weeks or when a
major Bevy release ships, whichever is sooner. Known limits of the
current painter — the planet is a Lambert disc rather than the
actual impostor; stars are a flat tint; planet direction is keyed
off the homeworld and will need to re-pick the nearest body once
ships move far from Thalos.

### Real-time GI / ray tracing (bevy_solari)

Not wired in. **Rewritten 2026-07-24** — the previous text ("the terrain has no
mesh, so it is invisible to ray tracing") described the udlod era and is
obsolete: terrain renders as ordinary meshes through the tile renderer.

The decision is recorded in
[ADR-20260724T224242Z](../adr/20260724T224242Z-solari-scene-half-not-lighting-half.md):
**take Solari's raytracing *scene*, never its raytraced *lighting*.** Verified
against the 0.19 crate source, `SolariLightingPlugin` forces the opaque path
deferred app-wide (costing our Hapke BRDF), extracts only plain
`StandardMaterial` (excluding every surface we ship — all are `ExtendedMaterial`),
and has **no sky or environment lighting whatsoever**: rays that miss contribute
nothing. On an atmospheric body that deletes the dominant ambient term, so it is
a *replacement* for our lighting universe and a worse one. `RaytracingScenePlugin`
carries none of that — BLAS/TLAS plus scene bindings, with no opinion on shading —
and surfaces enter it through `Mesh3d`-less proxy entities that share the visible
entity's mesh handle, keeping their own materials. Consumers: RT sun visibility
(NTR-RT2) and mirror-hull reflections (NTR-RT3), both behind the NTR-RT1 cost
gate. It stays experimental and hardware-RT-only, so it is **not** a foundation
for planet lighting — the raster path remains the baseline. The
dominant surface indirect is already analytic/baked and more robust at scale:
**planetshine** (ground bounce), **atmospheric skylight** (in-scatter), and the
**baked horizon AO** tile attachment. If dynamic bounce that also covers
terrain is wanted without solving RT geometry, **SSGI** is the pipeline-agnostic
middle path — screen-space, sees terrain and objects, no BLAS.

---

## BRDFs by body type

The atmosphere variant in the prior decides which BRDF the renderer
uses for the solid surface (or, for gas giants, the cloud deck).
This is a switch on the body, not per-pixel.

| Body type | BRDF | Why |
|---|---|---|
| Airless (Mira, Selva, asteroids) | Hapke + opposition-surge width keyed by `roughness_cubemap` | Reads as lunar; the surge is what makes the surface brighten when the sun is behind the camera |
| Thin atmosphere, dry (Vaelen, Mars-likes) | Lommel-Seeliger | Cheap stand-in that captures the dust-dominant scatter; can upgrade to Hapke if it bites |
| Thick atmosphere, wet (Thalos, Pelagos, Earth-likes) | GGX microfacet surface + microfacet ocean | The single-scattering atmosphere march (not Bruneton — see *Why this approach*) does most of the aesthetic work |
| Thick atmosphere, dry (Ashara, Venus-likes) | PBR GGX with Mie-thick atmosphere overpowering everything | Ground BRDF barely matters; the atmosphere is the read |
| Gas / ice giants | Cloud-deck shading via `gas_giant.wgsl` | Not a surface BRDF in the usual sense — the cloud deck is layered transmittance |

---

## Open questions

1. **Cloud authoring schema.** Where does cloud noise / latitude band
   data live? Adjacent to `TerrestrialAtmosphere`, or its own
   schema? Decide before M4 starts.
2. **Ocean reflection tiers.** F7/F9 provide the shared prefiltered sky
   environment. Profile and judge that result before choosing whether SSR or a
   local planar/probe tier earns its cost; analytic sky/sun remains the stable
   fallback in every case.
3. **Atmosphere on Vaelen.** Pressure is ~0.015 bar; thick enough to
   register optically, thin enough that "Mars-like Lommel-Seeliger"
   may already be enough without full Bruneton. Decide whether
   Vaelen is a degenerate case of M4 or an M4 milestone target.
4. **Aerosol layers**, like Pelagos's blue-haze atmosphere or
   tholin/dust on outer-system bodies. Whether those need additional
   scattering coefficients beyond Rayleigh + Mie + ozone, or whether
   tuning Mie + composition is enough.

---

## References

- [ADR-20260808T221912Z](../adr/20260808T221912Z-atmosphere-and-ocean-mechanisms-use-spatial-adapters.md) —
  shared atmosphere/ocean mechanisms with explicit spatial adapters.
- [Ocean rendering](ocean.md) — shared wave mechanisms and the planar versus
  planetary water adapters.
- [archive/gen/planet_aesthetics.md](../archive/gen/planet_aesthetics.md) —
  visual target reference (archived). Read this before tuning M4.
- [terrain.md](../world/terrain.md) — supplies the surface heights and ocean
  topology that atmosphere/ocean read against.
- [simulation.md](../simulation/simulation.md) — ship state and floating origin
  the reflection probe reads.
- Bruneton 2008, *Precomputed Atmospheric Scattering*. Canonical
  scattering reference.
- Sean O'Neil, *GPU Gems 2*. Older but still useful intro.
- Hapke, *Theory of Reflectance and Emittance Spectroscopy*. For
  airless-body BRDF.

---

*Doc owner: Korbin. Roadmap milestones served: M4 (rocky atmospheres
+ ocean rendering). Reflection probe deferred.*
