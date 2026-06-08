# Atmospheres, Oceans, and Lighting

The unified spec for everything that sits between the camera and the
solid surface: gas-giant materials, rocky-body atmospheric scattering,
clouds, ocean rendering, and image-based lighting (IBL / reflection
probe).

What runs today: gas-giant materials, a single-scattering Rayleigh +
Mie atmosphere on terrestrial impostors and terrain LOD, reference
cloud-cover overlays with shader-side differential rotation, an
in-impostor water BRDF, and a CPU-painted reflection probe. The
terrestrial atmosphere is production-ready for orbital views;
in-atmosphere flight is the remaining frontier.

## Status

| Area | Today | Future |
|---|---|---|
| Gas / ice giants | `GasGiantMaterial` + `atmosphere_gen::AtmosphereParams`: cloud deck, haze, rim halo, optional Rayleigh blue gap. Storm + aurora layers stubbed. | Storm and aurora layers; volumetric for cinematic close-ups. |
| Rocky-body sky | **Single-scattering Rayleigh + Mie raymarch** (`atmosphere.wgsl::integrate_atmosphere`). Per-body β + scale heights + Henyey-Greenstein g; one integral produces in-scatter, transmittance, rim halo, terminator orange, and aerial perspective. 8 view × 6 sun samples per fragment with per-pixel jitter. The terrain `BodySky` pass and the impostor path both use this orbital model so the surface stays readable and haze concentrates toward the limb instead of washing the whole disk blue. Sky pixels still boost alpha from local in-scatter luminance so bright sky crushes stars where an observer's eye would adapt away from them. Per-body params at `assets/bodies/<name>.ron::scattering`. | Ozone absorption (Earth's blue-purple twilight, two extra params); Bruneton 2008 / Hillaire-style multi-scatter LUTs once in-atmosphere flight justifies the precompute step. |
| Cloud rendering | **Terrain `BodySky` path: volumetric slab raymarch** (`body_sky.wgsl::cloud_volume_overlay`) between two concentric shells (`cloud_shape.x/y`). Density = reference coverage cube (16-band differential rotation) shaped by a vertical profile and 3-D detail noise; a short sun march gives self-shadow, a forward-scatter phase the silver lining, and a per-sample + segment terminator fade keeps the night side from punching black holes in the starfield. Camera-regime aware: clouds show over the disk from orbit and overhead from the surface. The impostor (orbital/far) still composites a flat lit reference shell + shadow probe. Bodies without a registered overlay bind a blank cube. Gas-giant cloud deck is part of the impostor. | Procedural coverage field (real weather patterns) to replace hand-picked equirects; half-res + temporal upscale for cost; surface-shadow projection from clouds onto terrain LOD; volumetric in the impostor for close orbital cinematics. |
| Oceans | In-impostor water BRDF: triggered where `sample_height_m(dir) < sea_level`. Authored deep-water color + minimum column depth. Sky-tint reflection now derives from the new β·H Rayleigh fields (was hand-authored). Flat surface. | Microfacet ocean with sun-glint streak, depth-darkened color, fresnel reflectivity, foam at coastlines. Probably a dedicated material rather than the impostor. |
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
- Realtime weather simulation. Cloud fields are static reference
  overlays with differential rotation only.
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

## Rocky atmospheres (shipping)

Single-scattering Rayleigh + Mie raymarch, per-channel Rayleigh
(Bucholtz 1995 sea-level coefficients) and scalar Mie
(Henyey-Greenstein phase function). The integral runs **once per
fragment in a single fullscreen pass per body** — the same shader
produces the rim halo, the lit-disk haze, the terminator orange
band, and the surface aerial perspective at every altitude from
orbit to ground.

### Architecture: shared optics, two render paths

The atmosphere is a fullscreen quad per body (`BodySkyMaterial` in
`crates/terrain_render/src/sky_material.rs`, shader at
`crates/terrain_render/src/body_sky.wgsl`) while the real terrain LOD is
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
direct Hapke sunlight, eclipse/craft shadow factors, and a small
atmosphere-derived sky-diffuse fill computed from the bound
`AtmosphereBlock`. `BodySky` owns camera-path haze/transmittance/clouds.
That split keeps airless terrain vacuum-black while preventing daylight
terrain under a blue atmosphere from turning pure black on sun-opposed
nearby slopes.

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
prepass-depth texture is terrain-blind. The workaround lives in
`crates/game/src/rendering/scene_depth.rs`:

- A `SceneDepthImage` resource owns a `Handle<Image>` whose GPU
  texture (`Depth32Float`, `COPY_DST | TEXTURE_BINDING`) is sized to
  the camera viewport.
- A render-graph node `CopySceneDepthNode` runs between
  `Node3d::MainOpaquePass` and `Node3d::MainTransparentPass` and
  issues `copy_texture_to_texture` from the main pass's
  `ViewDepthTexture` (which has terrain depth written by then) into
  the Image.
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
- The ship camera carries `depth_texture_usages = RENDER_ATTACHMENT
  | COPY_SRC` so the copy is legal, plus `Msaa::Off` so source and
  destination sample counts match. The `ShipCamera` component is
  extracted to the render world via `ExtractComponentPlugin` so the
  node's `ViewQuery` filters to that view only (the map camera and
  any future light / shadow views don't carry it).

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
  scene-depth clipping.
- ✓ **Mie forward-peak haze** on the night-side rim — warm crescent
  on back-lit bodies.
- ✓ **Knife-edge terminator** preserved on airless bodies (Mira,
  Selva, asteroids — `karman_line_m == 0` or absent
  `TerrestrialAtmosphere` skips both spawn and raymarch).

### Backlog

- **Unify impostor and terrain cloud shadows.** `planet_impostor.wgsl`
  still owns the full surface-side cloud composite, including the
  shadow probe and water BRDF. `BodySkyMaterial` now matches the
  visible cloud layer for terrain LOD, but terrain cloud shadows are
  still deferred until the ground path has an explicit receiver-side
  shadow term.
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

### Today: volumetric slab raymarch (terrain path)

The terrain `BodySky` fullscreen pass (`crates/terrain_render/src/body_sky.wgsl`)
raymarches a **volumetric cloud layer** as a thin slab between two
concentric spheres — cloud base `radius + cloud_shape.x` and cloud top
`base + cloud_shape.y`. This replaced the earlier fixed-altitude 2-D
shell intersection. The march reuses everything the atmosphere pass
already solved: the per-body fullscreen quad, the scene-depth clip (so
terrain and the ship hull correctly occlude clouds), and the
`AtmosphereBlock` cloud uniforms.

Per sample, density is built from three factors:

1. **Coverage** — the reference cloud-cover cubemap, sampled with the
   existing 16-band differential rotation (`sample_cloud_banded`). This
   is the large-scale weather map and the co-rotation source.
2. **Vertical profile** — rounded base, eroded top, so the slab reads as
   a deck rather than a uniform fog.
3. **Detail noise** — a few octaves of animated 3-D value-noise fbm in
   body-local space (so billows co-rotate with the surface), combined via
   a coverage-threshold erosion (`n + cov − 1`) that carves clear sky
   where coverage is low and keeps cloud where it is high.

Lighting is a short secondary march toward the sun for self-shadow
(Beer's law), a cheap forward-biased phase for the sun-facing silver
lining, and an ambient floor so undersides aren't black. A per-sample
and per-segment terminator fade keeps night-side clouds from occluding
stars as black blobs. The result is premultiplied and composited over
the atmosphere in-scatter exactly like the old overlay.

The pass is **camera-regime aware**: above the layer (orbit) clouds are
gated to the planet disk and faded across the geometric horizon to avoid
a hard limb tangent band; below the layer (standing on the surface) the
underside renders overhead on sky pixels too; inside the layer (descent)
the march runs from the camera to the nearest shell. Step count is
adaptive (target ~`thickness/16`, clamped 8–32) with per-pixel
interleaved-gradient jitter to hide banding.

The **orbital impostor** (≥ 4× radius) is unchanged — it still
composites a flat lit reference shell + shadow probe
(`planet_impostor.wgsl::composite_clouds`); a per-pixel volumetric march
on a body that small on screen isn't worth it yet.

Cloud parameters live on `CloudCover` alongside `TerrestrialAtmosphere`
in the body RON (`coverage`, `albedo`, `scroll_rate`,
`differential_rotation`, plus the new `base_altitude_m`, `thickness_m`,
`density`). They convert to render units and pack into
`AtmosphereBlock::cloud_shape` at the `from_terrestrial` boundary.

### Next

- **Procedural coverage field.** The coverage cube is still a hand-picked
  equirectangular reference image per body (`reference_clouds.rs`).
  Replace it with a procedural weather field (noise + the latitude bands
  the environment model already carries) — this is the hook for real,
  evolving weather patterns and storm structure inferred from rotation
  rate + obliquity.
- **Cost: half-res + temporal upscale.** The march currently runs
  full-res; on surface views it touches the whole sky hemisphere. Drop to
  half/quarter-res with temporal reprojection (Blackrack/HZD style) if it
  bites.
- **Surface cloud shadows (designed; terrain receiver pending).** A dynamic,
  low-res **sun-projected cloud transmittance** buffer, sampled as the
  `cloud_transmittance` multiplier on the direct-sun term by terrain, objects,
  and the in-scatter march alike — one shared slot (see
  [terrain.md](terrain.md) *Surface shadows*). Cheap because clouds are
  soft/low-frequency; updated per frame (or every few) from the cloud volume
  state in `SolarSystemState`, kept in a body-fixed / sun-aligned frame so it
  doesn't swim under the floating origin. The same transmittance, sampled per
  step in the `BodySky` march, gives **god rays / crepuscular shafts** (already
  depth-coupled via `SceneDepthImage`). The impostor's existing shadow probe is
  the orbital analogue; the terrain receiver is the remaining wiring.

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
  wrong artifact for a grounded stainless vehicle. Making terrain RT-visible
  means extending the collider-patch trimesh extraction (see
  [terrain.md](terrain.md) *The tile contract* / *M5 colliders*) into a BLAS
  region, accepting raster-vs-RT geometry divergence at LOD seams, and solving
  acceleration-structure precision under the floating origin. A research
  project, not a toggle — the single height authority is what keeps the door
  open.

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

### Real-time GI / ray tracing (bevy_solari)

Not wired in (Bevy 0.18). `bevy_solari` builds ray-tracing acceleration
structures from `Mesh3d`; the terrain has no mesh (GPU-procedural indirect
draw), so it is invisible to ray tracing without the BLAS-extraction work
above. Its natural fit is the **near-field mesh scene** — ship, EVA, future
interiors/stations — where dynamic RT bounce shines; it is experimental and
hardware-RT-only, so it is **not** a foundation for planet lighting. The
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

- [archive/gen/planet_aesthetics.md](archive/gen/planet_aesthetics.md) —
  visual target reference (archived). Read this before tuning M4.
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
