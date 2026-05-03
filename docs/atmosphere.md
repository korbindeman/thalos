# Atmospheres, Oceans, and Lighting

The unified spec for everything that sits between the camera and the
solid surface: gas-giant materials, rocky-body atmospheric scattering,
clouds, ocean rendering, and image-based lighting (IBL / reflection
probe).

This doc is mostly future work (M4). What runs today is gas-giant
materials, a sparse terrestrial-atmosphere rim shader, an in-impostor
water BRDF, and a CPU-painted reflection probe.

## Status

| Area | Today | Target (M4) |
|---|---|---|
| Gas / ice giants | `GasGiantMaterial` + `atmosphere_gen::AtmosphereParams`: cloud deck, haze, rim halo, optional Rayleigh blue gap. Storm + aurora layers stubbed. | Storm and aurora layers; volumetric for cinematic close-ups. |
| Rocky-body sky | Sparse `TerrestrialAtmosphere`: rim halo + limb shading composited over the impostor. No path-traced scattering. | Bruneton 2008 precomputed scattering. Soft terminator with twilight wedge. Per-body parameters (Earth-like / Mars-like / Venus-like). |
| Cloud rendering | None on rocky bodies. Gas-giant cloud deck is part of the impostor. | 2D cloud shells with animated detail noise, latitude-banded structure, surface shadows. Optional volumetric layer for orbital cinematic moments. |
| Oceans | In-impostor water BRDF: triggered where `sample_height_m(dir) < sea_level`. Authored deep-water color + minimum column depth. Flat surface. | Microfacet ocean with sun-glint streak, depth-darkened color, fresnel reflectivity, foam at coastlines. Probably a dedicated material rather than the impostor. |
| Reflection probe | CPU painter: 256³ cubemap rewritten every 0.25 s with sun disc + Lambert planet hemisphere + dim starfield. Feeds Bevy's `GeneratedEnvironmentMapLight`. | Real-scene cubemap capture once Bevy supports omnidirectional cameras (PR #13840), or self-implemented if it bites. **Not a Phase-1 priority.** |

## Goals

- Read the right way at orbit and in atmosphere. Earth-like worlds
  should look Earth-like; airless moons should keep knife-edge
  terminators; Venus/Ashara should read as oppressive.
- Per-body parameterization. Atmospheric optics are functions of the
  atmosphere variant chosen in the prior, never a hard branch.
- One shading vocabulary. BRDF choice (Hapke / Lommel-Seeliger / GGX)
  is a function of the atmosphere variant, not a per-pixel switch.
- Cloud and atmosphere assets are authored or inferred at the
  `PlanetTerrainSpec` level so the same schema drives appearance from
  orbit through descent.
- Lighting via the same physical model the renderer already uses
  (`assets/shaders/lighting.wgsl`). New BRDFs slot into that library.

## Non-goals

- Volumetric in-atmosphere flight effects (god rays through cloud
  layers, mist banks). Worth a stretch goal but not in M4 scope.
- Realtime weather simulation. Cloud fields are procedural, not
  evolving.
- Sea state animation beyond a tile-able displacement / normal
  texture. No Tessendorf, no spectral wave sim.
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

## Rocky atmospheres (M4)

### Plan

Adopt Bruneton 2008 precomputed atmospheric scattering. This is the
canonical real-time approach and matches what KSP2 (and `Scatterer`
for KSP1) shipped. Four reasons we should use it as-is:

- LUTs are parameterized only by altitude and sun angles
  (latitude/longitude symmetry is exploited), so storage is cheap.
- Rayleigh + Mie are both supported, which we need: Earth-likes
  (Rayleigh-dominant, blue), Mars-likes (Mie-dominant, butterscotch),
  Venus-likes (Mie-thick, opaque) all fall out of the same model
  with different coefficient sets.
- Aerial perspective for distant terrain comes for free.
- Tested against ground truth in published papers and shipping
  games.

### Per-body parameters

Each body's `TerrestrialAtmosphere` (or a successor type) carries the
Bruneton parameter set:

- Rayleigh scale height (~8 km for Earth)
- Mie scale height (~1.2 km for Earth)
- Rayleigh scattering coefficient (RGB; per-body composition)
- Mie scattering coefficient (RGB)
- Mie phase asymmetry `g`
- Ozone-equivalent absorption coefficient and altitude band (Earth's
  ~25 km ozone layer is what gives the orange terminator)
- Top-of-atmosphere altitude (3-5 scale heights)

These are inferred by the prior from atmosphere variant + body
radius + composition, with author overrides.

### Visual targets

See [gen/planet_aesthetics.md](gen/planet_aesthetics.md) for the full
visual reference. The headline targets for M4:

- **Soft terminator** with twilight wedge on bodies that have
  atmospheres. Mountain peaks catch warm light while their bases
  shadow.
- **Limb glow extends beyond the geometric edge.** A flat disk with a
  fresnel ring reads as fake.
- **Aerial perspective** desaturates and tints distant terrain
  toward the limb color.
- **Knife-edge terminator** preserved on airless bodies (Mira, Selva,
  asteroids). The hardness sells the airlessness.

### Coupling to terrain

Bruneton scattering reads a depth/height field for the planet to
compute shadowing and aerial perspective. Once ground LOD lands
(M3), the atmosphere reads against the live terrain. Until then the
sphere geometry is enough for orbital views.

---

## Cloud rendering (M4)

Two layers, separate jobs:

1. **Cloud shells** — 2D shell sphere slightly larger than the body
   surface, textured with animated procedural noise. Latitude bands
   (ITCZ near equator, descending zones at ~30° lat, mid-latitude
   westerlies, polar caps). Cast shadows on the surface — without
   shadows, clouds float as detached layers.
2. **Volumetric layer** *(stretch goal)* — for cinematic atmospheric
   shots. Raymarched with temporal upscaling, similar to Blackrack's
   KSP1 volumetric clouds. Not required for M4 to ship.

Cloud parameters live alongside `TerrestrialAtmosphere` in the body
RON. The cloud noise field is procedural; storm structure (vortices,
fronts) is inferred from rotation rate + obliquity.

For gas giants, "clouds" *are* the cloud deck and live inside
`AtmosphereParams` already.

---

## Ocean rendering (M4)

### Today

The flat impostor has a built-in water BRDF triggered where sampled
height is below sea level. `PlanetWaterParams` carries:

- `water_color_depth: Vec4` — xyz is linear-RGB deep-water color, w
  is minimum optical column depth (in meters; prevents shelf-water
  artefacts on flat-ocean placeholders).
- `water_roughness: f32`.

This is good enough for the orbital impostor read. It is **not**
sufficient for descent or surface-level ocean.

### Target

A dedicated ocean material (not the impostor) once ground LOD ships.
Required:

- Microfacet specular with sun-glint streak. The sub-solar bright
  spot spreading into a glint streak with wave roughness is the
  single biggest "this is wet" cue.
- Depth-darkened color: deep blue offshore, green-cyan in shallows,
  driven by terrain depth at sample point.
- Fresnel reflectivity that increases at grazing angles.
- Foam at coastlines and where shallow.
- IBL contribution from the sky cubemap (so reflective oceans show
  the sky correctly, not just the sun).

### Where ocean lives

Ocean is the renderer's job, but the *topology* (where ocean is)
comes from the terrain feature compiler — sea level on
`AgingOceanicHomeworld` / `GenericTerrestrial` archetypes (M2). The
ocean material reads the terrain height cubemap (or tile, in M3) to
decide where it lives.

This is one of the reasons M4 sequences after M3: the same data flow
that drives ground LOD also drives ocean shoreline.

---

## Reflective surfaces / IBL

### Today

Environment-map source for metallic ship-part reflections. Feeds
Bevy's `GeneratedEnvironmentMapLight` on the main camera so
`ShipPartMaterial` panels read the sky from the ship's orbital
vantage.

CPU-authored cubemap, rewritten every `REFRESH_INTERVAL` seconds
(0.25 s) from the ship's current state:

- Cubemap `Image` asset, 256³, 6 layers, `Rgba16Float`, cube view
  descriptor. `TEXTURE_BINDING | COPY_DST` usage — no
  render-attachment, since we're not rendering into it.
- Painter reads ship-to-sun and ship-to-planet directions from
  `SimulationState`, plus the planet's angular radius from its
  physical radius and range. Each frame that hits the refresh tick,
  all 6 faces get rewritten: sun disc (HDR hot spot), lit-side
  planet hemisphere with a Lambert terminator, dim starfield tint
  everywhere else.
- Re-assigns the handle via `Assets<Image>::get_mut` which marks it
  changed; Bevy's runtime filter pipeline re-prefilters diffuse +
  specular mips downstream.

Lives in `crates/game/src/reflection_probe.rs`. ~300 lines.

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

`GeneratedEnvironmentMapLight` is the stable contract on the main
camera. Everything behind it can be swapped without touching ship
materials or camera setup.

### Status

Not a Phase 1 priority. Revisit cadence: every 6 weeks or when a
major Bevy release ships, whichever is sooner. Known limits of the
current painter — the planet is a Lambert disc rather than the
actual impostor; stars are a flat tint; planet direction is keyed
off the homeworld and will need to re-pick the nearest body once
ships move far from Thalos.

---

## BRDFs by body type

The atmosphere variant in the prior decides which BRDF the renderer
uses for the solid surface (or, for gas giants, the cloud deck).
This is a switch on the body, not per-pixel.

| Body type | BRDF | Why |
|---|---|---|
| Airless (Mira, Selva, asteroids) | Hapke + opposition-surge width keyed by `roughness_cubemap` | Reads as lunar; the surge is what makes the surface brighten when the sun is behind the camera |
| Thin atmosphere, dry (Vaelen, Mars-likes) | Lommel-Seeliger | Cheap stand-in that captures the dust-dominant scatter; can upgrade to Hapke if it bites |
| Thick atmosphere, wet (Thalos, Pelagos, Earth-likes) | PBR GGX + microfacet ocean | Standard; the atmosphere does most of the aesthetic work via Bruneton |
| Thick atmosphere, dry (Ashara, Venus-likes) | PBR GGX with Mie-thick atmosphere overpowering everything | Ground BRDF barely matters; the atmosphere is the read |
| Gas / ice giants | Cloud-deck shading via `gas_giant.wgsl` | Not a surface BRDF in the usual sense — the cloud deck is layered transmittance |

---

## Open questions

1. **Cloud authoring schema.** Where does cloud noise / latitude band
   data live? Adjacent to `TerrestrialAtmosphere`, or its own
   schema? Decide before M4 starts.
2. **Ocean material vs. impostor.** Keep the in-impostor water BRDF
   for orbit-only views and add a ground-LOD ocean material for
   close, or unify into one path? Probably unify — duplicate code
   between projections is a smell.
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

- [gen/planet_aesthetics.md](gen/planet_aesthetics.md) — visual
  target reference. Read this before tuning M4.
- [terrain.md](terrain.md) — supplies the surface heights and ocean
  topology that atmosphere/ocean read against.
- [simulation.md](simulation.md) — ship state and floating origin
  the reflection probe reads.
- Bruneton 2008, *Precomputed Atmospheric Scattering*. Canonical
  scattering reference.
- Sean O'Neil, *GPU Gems 2*. Older but still useful intro.
- Hapke, *Theory of Reflectance and Emittance Spectroscopy*. For
  airless-body BRDF.

---

*Doc owner: Korbin. Roadmap milestones served: M4 (rocky atmospheres
+ ocean rendering). Reflection probe deferred.*
