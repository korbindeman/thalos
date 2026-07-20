# Graphics fidelity plan

Living tracking doc for the push toward MSFS / KSP2-tier visuals.

**Visual direction:** *stylized but high fidelity* — lean on lighting, shading,
and a controlled-but-rich palette, *not* photo textures. MSFS is the reference
for **outdoor-HDR rendering technique** (atmosphere, aerial perspective,
exposure, soft shadows); KSP2 is the reference for **art intent** (readable,
slightly heightened palette). Not KSP1, not photoreal terrain.

This doc was restructured 2026-06-30 after a full architecture review (see
§7 Decision log). It is organised around the realisation that the renderer is
**one shared substrate with many consuming surfaces** — and that the biggest
structural debt is that crafts/structures don't yet consume that substrate.

## Status legend

- ☐ not started · ◐ in progress · ☑ landed (compile-verified) · ✅ runtime-verified by a `just game` screenshot

---

# 1. The renderer as a system

The pipeline is feature-rich; the flatness in current screenshots is a few
specific gaps + tuning + one structural split, not missing machinery.

**Already wired (the strong base):**

- **Shared lighting spine** — `thalos::lighting` (`shade_surface` / `ThalosSurface`)
  is a real single dispatch point. `body_terrain.wgsl` fills one `ThalosSurface`
  and the airless impostor (`solid_planet.wgsl`) shades through the *same*
  `shade_hapke_surface`, so a body's orbital disc and its ground LOD reconverge
  across the LOD swap. The BRDF kit is production-grade and `vec3`-ready (GGX D,
  Smith height-correlated vis, Schlick Fresnel, Oren–Nayar, Karis split-sum
  env-BRDF, Kulla–Conty energy compensation, Kaplanyan specular AA).
- **Scene lighting** — `SceneLighting` is one clean CPU mirror feeding every
  body material; eclipse occluders + planetshine come free to anything binding it.
- **Atmosphere** — custom per-body single-scatter raymarch (`body_sky.wgsl` /
  BodySky) with aerial perspective keyed to a copied **scene-depth** texture, a
  CPU-baked multi-scatter LUT, and a vendored volumetric cloud slab.
- **Shadows** — a self-managed **3-cascade** ortho sun-shadow rig
  (`rendering/sun_shadow.rs`, `SHADOW_CASTER_LAYER = 8`, 4096², copy-node →
  `Depth32Float`) sampled by terrain/trees/grass/rocks via `thalos::shadow`.
- **Terrain** — runtime-generated (`ProceduralSurface`, no bake), streamed by
  the vendored UDLOD renderer; per-tile height/albedo/roughness/**material-mask**
  atlases; rough-dielectric BRDF over analytic ecological albedo bands.
- **Vegetation** — meshed trees w/ octahedral far impostors, grass clipmap to
  ~1.5 km, scattered rocks/pebbles, wind, all shading through the spine's
  `shade_foliage`.
- **Camera/post (shared by both lighting paths)** — HDR target, **AgX** tonemap,
  bloom, SMAA, CAS, mild chromatic aberration, deband, exposure-driven film grain.

---

# 2. Architecture

## 2.1 The shared lighting spine

`thalos::lighting::shade_surface(s: ThalosSurface, …)` is the canonical
per-fragment shading entry. The caller fills `ThalosSurface { albedo, roughness,
normal_ws, geo_normal_ws, emissive, occlusion, metallic, translucency, style }`
and the function dispatches on `style` (DIELECTRIC / REGOLITH / FOLIAGE / WATER).
Terrain, vegetation, water, rock, and impostors all flow through it. **The
unification work is mostly extending and connecting to this spine, not building
a new one.**

## 2.2 The central fault line: two lighting universes

Terrain / vegetation / water / rock / impostors shade through the spine.
**Crafts and structures do not** — `assets/shaders/ship_part.wgsl` extends Bevy
`StandardMaterial` → `bevy_pbr::apply_pbr_lighting`; base-editor buildings, pads,
tanks, and the runway are plain `StandardMaterial`. These are a second
Cook–Torrance, a second shadow system, a second IBL, a second exposure authority,
and a third terminator definition.

The two universes are reconciled today only by a **CPU day/night scalar** in
`rendering/lighting.rs::update_sun_light`: it evaluates the terrain's terminator
function, then scales a Bevy `DirectionalLight` (`SUN_DAY_ILLUMINANCE = 10000`
lux — **no inverse-square**), a separate `MoonLight`, and `GlobalAmbientLight`
(`AMBIENT_NIGHT 4` → `AMBIENT_DAY 50`) to *mimic* what the spine computes
physically. Verified drift hazards:

- **Three terminator definitions** (`body_terrain.wgsl`, `update_sun_light`,
  `update_moon_light`) re-derive the same smoothstep from near-duplicate constants.
- **The hull doesn't track heliocentric distance** — flat 10k lux (lighting.rs:373)
  vs every spine surface scaling by `LIGHT_AT_1AU·(AU/d)²` (lighting.rs:126). A
  ship at Nyx is lit like one at Thalos.
- **Two exposure authorities** the code itself notes "compound": `CameraExposure`
  distance-gain × Bevy `AutoExposure` histogram, forced into agreement only by AgX.
- **Two moonlight models**, **two IBL systems** (spine analytic hemisphere vs the
  hull's crude CPU-painted `reflection_probe.rs` cubemap), **two shadow systems**.

**Important nuance:** the two universes *already share the camera* — same HDR
target, AgX, bloom, SMAA, post. The divergence is purely in **shading inputs**
(BRDF + how sun/sky/shadow/atmosphere/IBL are evaluated). That's why this is
fixable without touching the post stack.

## 2.3 The one-world principle — everything interacts

This is the organising invariant for the whole sprint. **Thalos is one physical
world; every renderable surface — terrain, terrain detail, vegetation, rocks,
crafts, landing gear, buildings, pads, tanks, runway, water — must obey the same
light, cast into the same shadows, occlude each other, and recede into the same
air.** A surface that opts out of any of these reads as a pasted-on cut-out.

Concretely, the invariants every surface must satisfy:

1. **One light environment.** One sun (with heliocentric flux + eclipse), one
   moon, one sky/ambient, one exposure, one tonemap. Time of day, eclipse,
   altitude, and weather change *every* surface identically. No surface carries
   its own private sun or ambient.
2. **One shadow world — every solid object is both caster and receiver.** Trees,
   rocks, **craft, gear, buildings, tanks, pads, runway** all cast into and
   receive from the *same* cascade rig. A ship shadows the grass; a hangar
   shadows the ship; a tree shadows the hull.
3. **Terrain occludes the sun for everything on it.** A mountain shadows the
   valley *and the ship parked in it*. This is a shared terrain-relief shadow
   term (horizon-angle), sampled by terrain **and** by every object standing on
   the terrain — not just terrain-on-terrain. (See §4.2.)
4. **Mutual contact/ambient occlusion.** Where any object meets the ground or
   another object, the crease darkens — regardless of which subsystem owns each
   surface. One screen-space occlusion field feeds every surface's ambient term.
5. **One atmosphere for everyone.** Aerial perspective + recession apply to every
   surface by camera distance, so nothing stays a crisp saturated cut-out against
   terrain the air has already hazed.
6. **Reflections reflect the real world.** Water and the hull reflect the actual
   sky/sun/terrain (driven from the atmosphere), not a fiction pinned to `t=0`.
7. **One view anchor — detail and view-dependent presentation exist around the
   camera, not the craft.** Every view-dependent system (terrain streaming,
   scatter clipmaps for trees/grass/rocks, the sun-shadow cascade centre,
   atmosphere/twilight visibility) is a function of **the render view**, so the
   same world renders consistently from any camera —
   flight orbit, freecam, god-view hub/base-editor, headless screenshot rig —
   with zero per-mode plumbing. The single authority is
   `rendering::view_anchor::ViewAnchor` (the camera pose resolved body-fixed at
   a coherent epoch; sole writer, all detail drivers read it). The player craft
   is just an object *in* the world; nothing may anchor detail to it. Landed
   2026-07-05, replacing the per-driver `scatter_view_center` fallback chain +
   `ShadowFocusOverride`. Regression probe: `just screenshot hub` (camera at
   the base, craft in orbit — anything craft-anchored goes missing there
   first). Extended 2026-07-20 to the celestial-backdrop twilight/Kármán term
   after freecam exposed its remaining `ship_state()` anchor (BL-12).

Today, items 1–6 hold *within* the spine (terrain ↔ vegetation ↔ rock), partially
for water, and **almost none of them hold for crafts/structures** (which only get
shared aerial perspective, because BodySky is a fullscreen depth-keyed pass the
hull happens to fall under). The foundation (§3) is what makes 1–6 hold for
*everything*.

## 2.4 The shared-substrate contract

Every surface must agree on **one** of each substrate. This table is the
load-bearing artifact of the plan: each bold cell is a parallel re-implementation
to be collapsed into the spine column.

| Surface | Lighting | Terminator | Exposure | Shadow | IBL | AO | Atmosphere |
|---|---|---|---|---|---|---|---|
| Terrain | spine | shared | gain | rig (recv) | analytic | analytic | BodySky |
| Vegetation | `shade_foliage` | shared | gain | rig (cast+recv) | analytic | baked vtx | recession |
| Rock | spine | shared | gain | rig (cast+recv) | analytic | baked vtx | recession |
| Water | own GGX | shared | manual | SceneLighting | tint hack | — | BodySky |
| Impostor | spine (Hapke) | shared | gain | — | analytic | — | own/BodySky |
| **Craft** | **Bevy PBR** | **CPU mirror** | **histogram** | **none** | **CPU cubemap** | **none** | BodySky (depth) |
| **Structures** | **Bevy PBR** | **CPU mirror** | **histogram** | **none** | **CPU cubemap** | **none** | BodySky (depth) |

The spine already covers ~4 of 8 columns; the craft path re-implements all 8 in
parallel and is missing shadow + AO entirely. **Foundation = collapse the
parallel path into projections of the spine** so the one-world invariants hold.

## 2.5 The frequency-band terrain-material model

"Terrain should create textures" is a **frequency-band** question, not yes/no.
Generating unique per-texel planet textures violates the no-bake invariant and is
petabyte-class — the wrong answer. The production-correct architecture (which
Thalos *already has the skeleton of*) is a band split:

- **Macro (low-freq):** per-tile material-weight mask (grass/soil/rock/wet) +
  analytic ecological bands by altitude/slope/moisture. *Already shipping.*
- **Detail (high-freq, near-field):** small **tiling material** library
  (albedo+normal+roughness), height-blended by the mask weights, faded in by
  distance, de-repeated by stochastic (hex) tiling. *This is the gap* — today the
  detail layer is noise-modulated *flat colour*, so close-ups read as "luminous
  ground" the shader fights with normal hacks.

The brittleness to retire: **four overlapping palettes** (`landcover.wgsl`,
`body_terrain.wgsl`, `procedural.rs::albedo_at`, `synthetic.rs`) and a
hand-maintained bit-for-bit `landcover.wgsl ↔ ground/landcover.rs` CPU/WGSL
mirror with no automated guard. See §4.6.

---

# 3. The unification foundation (do this first)

The structural moves that make every later fidelity item cheaper and make the
one-world invariants (§2.3) hold. **Decision: shared INPUTS first (B), shared
BRDF second (A)** — B is low-risk, retires the magic-number bridge immediately,
and is a prerequisite for A (a metallic hull on `shade_surface` leans hard on the
IBL that B builds). See §7 for the rationale.

| ID | Foundation step | Status | Effort | Sprint |
|---|---|---|---|---|
| **F1** | One `terminator(elev, alt)` helper + Bevy sun/moon/ambient driven as a *projection* of `SceneLighting` + heliocentric hull flux. Retires 3 terminators → 1 and the flat-lux bug. `rendering/lighting.rs` becomes sole writer. | ✅ | Med | THIS |
| **F2** | One exposure authority: **keep the artist `CameraExposure` distance curve, freeze/remove the Bevy `AutoExposure` histogram** (+ a fixed global-exposure baseline). Stops the distance-gain × histogram fight. *(Chose this over EV100 luminance metering — Q3.)* | ✅ | Med | THIS |
| **F3** | Hillaire-style **sky-view LUT** built from the existing `integrate_atmosphere` (multi-scatter LUT already exists, so the pattern is in-codebase). | ☑ | Med-High | THIS | **Landed (compile + clippy clean, 3 physical-invariant unit tests pass, 2026-07-01, awaiting screenshot).** New CPU `SkyViewLut` (`body_render::shading::sky_view`) raymarches the *same* single+multi-scatter model as `integrate_atmosphere_multiscatter` (reusing the shared `multi_scatter` primitives — `MultiScatterLut`/`compute_t_exit`/`sun_optical_depth` now `pub(crate)`), baked into a 2-D `(azimuth-from-sun × view-zenith)` LUT for the current sun dir + altitude. Sun-only light model (Open Q8 default). **Consumer wired:** `reflection_probe.rs` now sources the surface **sky** from the physical LUT (keeping the analytic ground-bounce + sun-disc reddening), retiring the hand-kept `cpu_surface_sky` WGSL-mirror hazard — so the metallic hull + dielectric structures reflect the real atmosphere-derived sky. `PHYSICAL_SKY_SCALE` (=1.0, physical baseline) is the calibration dial. **Next (F4):** project this LUT → SH for the terrain/`StandardMaterial` ambient + retire `GlobalAmbientLight`. **Runtime-verified 2026-07-02** after a 3-round debug: (1) never paint with `CameraExposure.gain = 0` (boot ordering → black-env stuck state; probe now defers + the change-gate watches sun-disc radiance); (2) repaint cadence must be warp-aware (2 s real / 0.5 s under warp + a 120 sim-s drift trigger — the sun crosses day/night between paints on a short-day planet); (3) see F4 for the sky **double-count** lesson |
| **F4** | Integrate the LUT → 9 SH coeffs feeding **both** the spine ambient *and* the `StandardMaterial` ambient from one physical source. Deletes the hand-tuned `GlobalAmbientLight`. *(Prototypable on the current analytic sky before F3 lands.)* | ◐ | Med | THIS | **Sub-step A landed (compile+clippy clean 2026-07-01, awaiting screenshot):** `reflection_probe.rs` now paints the env cubemap from the **atmosphere-derived surface sky** (CPU mirror of `compute_surface_sky`) blended by altitude into the orbital model, feeding the existing `GeneratedEnvironmentMapLight` — so the metallic hull reflects the real sky-dome + ground, and dielectric structures get real sky ambient (Bevy prefilters the cube → diffuse ≈ SH + specular). **Sub-step B — surface ambient landed (compile+clippy clean, unit-tested, 2026-07-01, awaiting screenshot):** the *surface* `GlobalAmbientLight` is now the **physical sky irradiance** (the SH DC term — `SkyViewLut::ambient_sky_irradiance`, the F3 LUT's cosine-weighted hemispherical irradiance), published by the reflection probe as `SkyAmbient` and consumed by `update_sun_light`. This retires the hand-tuned 700-lux day-fill + fixed sky-blue tint *for the surface* — brightness+chroma now track time-of-day/sun-elevation/atmosphere, with the flux→lux mapping shared with the sun (`AMBIENT_SKY_LUX_GAIN`=1 dial). **Runtime-verified 2026-07-02, with a load-bearing correction:** the flat surface ambient must be a **residual** (`AMBIENT_SKY_LUX_GAIN = 0.2`), not the full sky irradiance — the env cubemap's prefiltered diffuse *already* delivers the sky irradiance to every `StandardMaterial`, so a full-strength flat term counts the sky **twice** (washed-out hull by day, buildings glowing at dusk — exactly what the first cut produced). `GlobalAmbientLight` also gained ~0.7 s temporal smoothing so warp-crossing day/night repaints fade instead of popping. **Remaining:** (a) the **space** ambient still uses the flat stand-in (blended in by altitude — retires with env-map IBL at photometric intensity, W7/F7); (b) full SH-9 + **spine** (terrain) ambient port (the terrain still shades the sky analytically via `compute_surface_sky` — close, same atmosphere, but not yet the LUT); (c) the env cubemap still hard-cuts per repaint (visible in reflections under warp; a crossfade is a possible follow-up). |
| **F5** | Screen-space AO pass → `ThalosSurface.occlusion` (and the Bevy ambient). **Ship with F4** — sky-IBL ambient is flat in crevices without AO. | ☑ | Med | THIS | **Terrain landed (compile+clippy clean, `ssao.wgsl` naga-validated, 2026-07-01, screenshot-TUNING pending).** Custom half-res hemisphere-SSAO node (`rendering::ssao`: `AoImage` + a `Core3d` pass mirroring `CopySceneDepthNode`) reads `SceneDepthImage` — the one depth that sees the forked-udlod terrain (Bevy's prepass-based GTAO is terrain-blind here). View-space, f32-safe under big_space; 1-frame latency (sampled next frame). Applied into `body_terrain.wgsl`'s `surf.occlusion` (ambient-only), threaded via the sun-shadow-map pattern (patched onto the material, gated by `inspection.w`, white fallback). `SsaoConfig` = radius/bias/intensity/power dials. **Remaining:** veg/rock/grass materials + the hull (`ship_part.wgsl`) + Bevy StandardMaterial ambient (mechanical follow-ons — same `AoImage`); a blur pass (currently IGN-rotated, no blur). SSAO is eyeball-verified — first `just game runway`/`landing` screenshot is for tuning, not pass/fail. |
| **F6** | Shadow-rig unification: stamp craft + structures onto `SHADOW_CASTER_LAYER`; make the `StandardMaterial` path sample `thalos::shadow`. Now everything casts + receives one shadow. | ☑ | Low-Med | THIS | **Landed 2026-07-02 (compile+clippy clean, awaiting screenshots) — see W5/W6/W12 in §4.2 for the full inventory.** One shadow world: `ShadowedStandardMaterial` (new, `body_render::craft`) gives every former Bevy-CSM receiver the rig; runway casts; **stock Bevy CSM disabled on the sun light**; the analytic `BodyTerrainShadow` craft proxy deleted; stable-CSM slope-scaled/normal-offset bias added to `thalos::shadow`; craft-local orbit mode keeps hull self-shadow off-surface. F8b's "retire Bevy stock CSM" is hereby already done — its remainder is the `shade_surface` port itself. |
| **F7** | Metallic conductor branch in `surface_brdf` + **one shared view-level scene+atmosphere bind group** + roughness-mip prefiltered env from the F3 LUT. | ☐ | High | next |
| **F8a** | Port **structures** (runway/pads/buildings/tanks) onto `shade_surface` (low-risk half — simple metallic/dielectric, already f64-posed). | ☐ | Med | next |
| **F8b** | Port **hull** onto `shade_surface`; retire Bevy stock CSM, the CPU reflection probe, and the magic constants. | ☐ | Med-High | next |
| **F9** | Wire the stubbed `FOLIAGE` / `WATER` branches into `shade_surface`; retire the parallel `shade_foliage` + `body_water.wgsl` BRDFs; re-enable ground-LOD water (`TERRAIN_PATH_WATER_ENABLED`). | ☐ | Med | next |

**The keystone is F3+F4+F5+F6** — one atmosphere-derived environment + one
occlusion field + one shadow world is what delivers the one-world invariants.

**Caveats baked into the steps:**
- *F1:* `update_sun_light` currently rewrites illuminance + `GlobalAmbientLight`
  every frame; the projection must be the *sole writer* or runtime overrides get
  clobbered (see memory `craft-shadow-caster-layer`).
- *F3:* the standard sky-view LUT assumes one dominant light; Thalos has sun +
  moon + stars + eclipse. **Open Q8** — LUT the sun only (keep moon/star analytic)
  or rebuild per dominant light.
- *F6 / shadows under big_space:* cascade texel-snap and origin must be computed
  in **floating-origin-relative** coordinates, not planet-centric f64 (at Mm
  radius, f32 light-space ULPs are texel-sized). Preserve the existing
  craft-centred (not camera-centred) framing that dodges the frame-lagged crawl.
- *F7:* the ground module already fights Metal's 16-vertex-buffer cap +
  `AsBindGroup` forcing vertex visibility on every `#[uniform]`. A hull/structure
  material calling `shade_surface` re-trips exactly that gauntlet — plan the
  shared scene/atmosphere bind group *with* the metallic branch or the port
  stalls on per-material packing.

---

# 4. Workstream tracking

Grouped by **substrate**, not as a flat list. Each item: `status · effort ·
sprint · deps · technique/source`. THIS = surface/vegetation/atmosphere on Thalos
+ the §3 foundation. Items are ranked roughly by visual-impact-per-effort within
each group.

## 4.1 Lighting inputs

| ID | Item | Status | Effort | Sprint | Notes |
|---|---|---|---|---|---|
| W9 | Lighting-input unification (= F1): one terminator, `SceneLighting`-driven Bevy lights, heliocentric hull flux | ◐ | Med | THIS | compile+clippy clean 2026-06-30; sun→heliocentric flux (`LUX_PER_SPINE_FLUX`), `surface_daylight` helper unifies sun+moon terminators. **Awaiting noon screenshot to tune the calibration constant** |
| W10 | Single exposure authority (= F2) | ✅ | Med | THIS | **runtime-verified 2026-07-01** (runway surface A/B unchanged, as predicted — AutoExposure was ~neutral there, nothing propping it up). Model: *keep the artist `CameraExposure` distance curve, remove `AutoExposure`* (not EV100 metering). Retired the Bevy `AutoExposure` histogram (`post_stack.rs`) + the `tune_auto_exposure` vacuum/surface preset blend (`camera.rs`); brightness now = `CameraExposure` input gain × fixed `color_grading.exposure` baseline (`GLOBAL_EXPOSURE_STOPS`). Confirmed the exact split via Bevy 0.19 source: `AutoExposure`→`color_grading.exposure` (global tonemap multiply, all surfaces), `Exposure`→`view.exposure` (PBR hull only). **Next:** noon calibration (nudge `GLOBAL_EXPOSURE_STOPS`), then the airlight fudges (`DISC_AIRLIGHT_FRACTION` / `aerial_perspective_strength`) can retire on a consistent exposure |
| — | Moonlight onto ground + hull (`moonlight_radiance` + `MoonLight`) | ◐ | — | THIS | moon **night-gate** now shares F1's `surface_daylight` terminator; two moon models still to fully merge |

## 4.2 Shadows — the one-world shadow model

Two layers deliver "objects aren't lit when terrain/things block the sun":

- **Near cascade (object scale):** every solid object casts + receives in the
  same 3-cascade rig. *Object-on-object and object-on-ground.*
- **Terrain-relief shadow (planet scale):** a shared **horizon-angle** term so a
  mountain shadows the valley *and the objects standing in it*. Sampled by terrain
  and by every object, not just terrain.

| ID | Item | Status | Effort | Sprint | Notes / source |
|---|---|---|---|---|---|
| — | Cascaded sun-shadow rig (terrain/trees/grass/rocks receive; trees+rocks cast) | ✅ | — | done | `sun_shadow.rs` + `thalos::shadow`. Open tuning: region size, bias, PCF |
| W5 | **Caster + receiver unification** (= F6): craft, gear, buildings, pads, tanks, runway cast into and receive the rig | ☑ | Low | THIS | **Landed 2026-07-02 (compile + clippy clean, awaiting screenshots) — ONE SHADOW WORLD.** Every former Bevy-CSM receiver now samples `thalos::shadow`: the hull via `ship_part.wgsl` (F6b, as before), and everything else via the new **`ShadowedStandardMaterial`** (`body_render::craft::ShadowReceiveExtension` + `assets/shaders/shadowed_standard.wgsl` — stock PBR + rig receive, fanned per-frame by `apply_craft_shadow`): base buildings/pads/tanks/tarmac, runway paving+posts, plain craft parts (pod/engine/wing/control-surface/nacelle/gear), and the EVA capsule. Runway top/skirt/posts now also **cast** (`SHADOW_CASTER_LAYER`). **Stock Bevy CSM on the sun light is DISABLED** (`spawn.rs`, `shadow_maps_enabled: false`) — exactly one shadow definition exists. The old analytic terrain craft-shadow proxy (`BodyTerrainShadow` capsule/quad ray-test) is **deleted** (shader + `ground_terrain.rs` driver + `THALOS_TERRAIN_CRAFT_SHADOW`). In orbit / above 6 km AGL the rig switches to **craft-local mode** (cascade 0 centred on the craft, far cascades parked) so the hull keeps self-shadowing without ground cascades. F8's `shade_surface` port later retires the stylized `CRAFT_SHADOW_FLOOR`/`SHADOW_FLOOR` attenuation |
| W2 | **Cloud sun transmittance for every receiver** (= cloud §3.5 / CLOUD-5): a body-fixed globe tail + view/sun-aligned near cascades derived from the canonical density field; terrain, objects, water, atmosphere shafts, and ambient all consume it | ☐ | High | later | A coverage-map projection is a debug rung only; the final field must match visible 3-D density. Nubis / Skybolt / UE Volumetric Cloud |
| W6 | Stable CSM: bounding-sphere fit + **floating-origin-relative** texel snap; slope-scaled normal-offset bias per cascade | ☑ | Low-Med | THIS | **Landed 2026-07-02 (awaiting screenshots).** Per-cascade texel snapping in the light plane already existed (render-space/floating-origin-relative, craft-centred); added the **receiver normal-offset + slope-scaled depth bias** in `thalos::shadow`: `ShadowCascadeBlock` grew `sun_dir` + per-cascade texel size (`params.y`), and the new `sun_shadow_factor_nrm(pos, normal, …)` offsets the sample point ~1 texel along the surface normal and scales the bias by tanθ to the sun. Terrain (`height_n`), hull, and all `ShadowedStandardMaterial` surfaces use it; foliage/grass keep the normal-less path at `base × 2.5` (≈ the old tuned biases). Base biases dropped 0.6/2.5/10 m → 0.25/1/4 m (less peter-panning). Tuning knobs: `CASCADE_BIAS_M`, `NORMAL_OFFSET_TEXELS`, `MAX_SLOPE_SCALE`, `NO_NORMAL_BIAS_SCALE` |
| W12 | **Terrain horizon-angle self-shadow** sampled by *all* surfaces (mountain shadows valley + objects) | ◐ | Med-High | next | Terrain-side already ships (`terrain_self_shadow` height-atlas march in `body_terrain.wgsl`, bounded by resident tiles). **Object-side v1 landed 2026-07-02:** `body_render::horizon_sun_visibility` (pure f64 body-local march of the body's `HeightSource` along the sun azimuth, ~30 km reach, no bake) evaluated at the craft each frame in `update_sun_light` and multiplied into the sun `DirectionalLight` illuminance — a mountain between the low sun and the parked craft/base now pulls the direct term on hull/structures/EVA to 0 (ambient sky fill deliberately kept). **Remaining for full W12:** per-fragment horizon term for trees/grass/rocks (spine materials), longer-than-resident-tile terrain reach, max-mip acceleration |
| W18 | Contact shadows + PCSS contact-hardening | ☐ | Med | later | short screen-space depth march (reuse `SceneDepthImage`); grounds objects, fixes peter-panning |

## 4.3 Ambient occlusion & GI

| ID | Item | Status | Effort | Sprint | Notes / source |
|---|---|---|---|---|---|
| W8 | **Visibility-Bitmask AO (VBAO)** → `ThalosSurface.occlusion` + Bevy ambient | ◐ | Med | THIS | Therrien 2023 / XeGTAO. **First cut landed as F5 (2026-07-01):** custom half-res **hemisphere SSAO** node (`rendering::ssao`) reading `SceneDepthImage`, viewspace-only (f32-safe), small world radius, into the terrain `surf.occlusion` — the "custom node mirroring `CopySceneDepthNode`" plan. **Not yet VBAO** — it's classic hemisphere SSAO (no bitmask/thickness heuristic for thin grass/trunks); upgrading the shader to VBAO is a drop-in follow-up. **Remaining:** veg/rock/hull materials + Bevy StandardMaterial ambient + a spatial blur (currently IGN-rotated). Delivers one-world invariant #4 (mutual contact AO) for the dominant surface |
| W20 | Per-tile sky-visibility (basins ambiently darker) | ☐ | Med-High | later | horizon scan in `PipelineTileProvider`, new attachment channel (coarse/cached; adds cold-stream cost) |
| — | Bent normals / SSGI | ☐ | High | later | after VBAO + IBL prove out |

## 4.4 IBL / reflections

| ID | Item | Status | Effort | Sprint | Notes / source |
|---|---|---|---|---|---|
| W7 | **Atmosphere → SH ambient → prefiltered env** (= F3+F4): one environment for spine ambient + `StandardMaterial` ambient + hull specular | ◐ | Med-High | THIS | Hillaire 2020 / Frostbite. **Sub-step A landed** (2026-07-01, awaiting screenshot): the CPU-painted `reflection_probe.rs` no longer paints a *fake* orbital-only env — it paints the **atmosphere-derived surface sky** (blue dome + warm ground + reddened sun, CPU mirror of `compute_surface_sky`) blended by altitude into the orbital planet-disc, feeding the existing `GeneratedEnvironmentMapLight` prefilter (diffuse ≈ SH + specular). So the metallic hull reflects the world it's in and dielectric structures get real sky ambient. **Sub-step A′ / F3 landed (2026-07-01, awaiting screenshot):** the reflection probe's env cubemap now paints its **sky** from the physical `SkyViewLut` (a raymarch of the same single+multi-scatter model the terrain shades through) instead of the analytic `cpu_surface_sky` mirror — so the reflected sky is physical, not hand-tuned. **Next:** retire `GlobalAmbientLight` (Sub-step B), project the sky-view LUT → SH for the *terrain/`StandardMaterial`* ambient (F4 proper — the probe consumes it for the hull today), and a GPU cubemap-render can replace the CPU paint. Keystone — resolves three divergences |
| W19 | Water onto `shade_surface WATER` (sky/sun reflection) + re-enable ground-LOD water | ☐ | Med | next (F9) | retires 3 hand-calibrated water BRDFs; `TERRAIN_PATH_WATER_ENABLED=false` today |
| — | Screen-space reflections (SSR) | ☐ | Med | later | only if env-map reflections read too static on water/hull |

## 4.5 Atmosphere / aerial / clouds / water

The high-fidelity cloud direction is now a dedicated program in
[clouds.md](clouds.md). This table keeps the stable W-IDs, but cloud design,
phase dependencies, and acceptance criteria live there.

| ID | Item | Status | Effort | Sprint | Notes / source |
|---|---|---|---|---|---|
| — | Atmosphere Mie retune (clean continental haze) | ◐ | — | THIS | `assets/bodies/thalos.ron`; awaiting noon screenshot |
| — | Object aerial recession (foliage/rocks → sky haze) | ◐ | — | THIS | `object_aerial_recession`; fold inside `shade_surface` so it can't be forgotten (one-world invariant #5) |
| W11 | Hillaire aerial-perspective **froxel** LUT | ☐ | High | later | keep the raymarch as the space/upper-atmosphere fallback; 3-channel transmittance for sunset |
| W15 | Atmosphere-coupled Nubis cloud lighting (powder + multi-scatter octaves + shared sun/sky transmittance) (= CLOUD-4) | ☐ | High | later | cloud §3.4 · Schneider 2023 |
| W16 | View-relative 3×3 amortized cloud reconstruction, neighborhood-clamped body-fixed history, and screenshot mode (= CLOUD-2) | ☐ | High | later | cloud §3.3; cloud-local resolve is **not gated on W13** |

## 4.6 Terrain material

The mask + bands already ship; the upgrade is making weights select **tiling
materials** instead of constant colours.

| ID | Item | Status | Effort | Sprint | Notes / source |
|---|---|---|---|---|---|
| — | Landcover palette/altitude bands rework (green→grey→white, wider forest, `C_ALPINE`) | ◐ | — | THIS | in working tree; awaiting screenshot |
| TM1 | **Material-ID weight blending**: ~4–8 tiling PBR materials in `texgen`, bound as `texArray`, height-biased (not linear) weight blend | ☐ | Med | THIS-adj | COD MLTM / Star Citizen 0.5-midpoint heightlerp. Mind the Metal buffer budget (F7 note) |
| TM2 | Repetition + projection polish (near-field): hex-tiling on detail materials gated by `DETAIL_FADE`; selective triplanar on rock/cliff weight | ☐ | Med | next | Mikkelsen 2022 + Heitz-Neyret. WGSL: needs `textureSampleGrad` in the randomized-UV loop (check `wgsl-bevy` skill) |
| TM3 | **Collapse the palette/mirror debt**: moisture/landcover field onto the `SurfaceQuery` seam, baked into the material attachment; shader + grass read the baked field; retire `albedo_at` + the hand-mirror | ☐ | Med | THIS-adj | pairs with W1 (both read `forest_coverage`); also kills the W1 CPU/GPU hash mirror |
| TM4 | RVT-style cached tile synthesis | ☐ | High | **defer** | UDLOD is already a sparse-tile VT substrate, but **profile first** — surface frames are CPU/collision-bound; atlas array is not `STORAGE` yet (GPU tile production genuinely unbuilt) |

## 4.7 Vegetation (this sprint's headline surface work)

| ID | Item | Status | Effort | Sprint | Notes / source |
|---|---|---|---|---|---|
| W1 | **Aggregate canopy/grass colour baked into terrain albedo** | ☐ | Low | THIS | `forest_coverage`-driven `C_FOREST` tint, faded as geometry coverage ramps in. Kills the orbit→ground descent pop. Star Citizen Planet Tech. **Tint-out and geometry-in as one coupled curve** |
| W3 | **GoT grass shader tricks**: view-space blade widening + curved normals + fractional-width distance LOD | ☐ | Low | THIS | Wohllaib GDC 2021 / GPUOpen. Kills edge-on shimmer |
| W4 | **Two-sided foliage translucency** (reserved `translucency` field) + **bake the term into the impostor** | ☐ | Low | THIS | Habel 2007 / UE two-sided foliage. Backlit leaves glow |
| W21 | Foliage atlas mips (far leaf cards shimmer) | ☐ | Low | THIS | pull forward if W4 ships — impostor leaf shimmer compounds with translucency |
| — | Clump-card (billboard tuft) far/mid rings (`BladeKind::Card`) | ◐ | — | THIS | in working tree; verify visually |
| W14 | HLOD forest cluster impostors (mid-altitude band) | ☐ | Med-High | later | per coarse cube cell; proxy colour = W1 aggregate canopy |
| W22 | `#[bindless]` foliage materials (regain MDI/GPU-cull) | ☐ | Low-Med | later | verify GPU-cull composes with the big_space relative-position vertex path |
| VEG-R | **Driver unification**: fold `grass.rs`/`vegetation.rs`/`rocks.rs` into one `VegLayer` driver (they triplicate the clipmap lifecycle and **diverge on base-clearing**) | ☐ | Med | THIS? | pure refactor, no screenshot payoff — but prevents drift while we touch all three for W1/W3/W4. **Open Q5** |

## 4.8 Distant bodies & orbit-to-floor LOD chain

| ID | Item | Status | Effort | Sprint | Notes |
|---|---|---|---|---|---|
| W13 | **TAA / temporal resolve** in the **body-fixed** frame | ☐ | Med-High | later | unblocks whole-scene dithered LOD cross-fade + vegetation/edge stability. Cloud reconstruction owns a local depth/motion history (CLOUD-2) and no longer waits on this fork. Motion-vector story under big_space/dual-camera is the deferral reason. **Open Q7** |
| 6b | Dithered mesh↔impostor LOD cross-fade | ☐ | Med | later | gated on W13 |
| W17 | Slice-6 distant-body view: UDLOD at all ranges, delete the flat-colour impostor branch, re-home limb glow onto BodySky | ☐ | High | later | the interim solid-colour impostor breaks Hapke↔Vegetated reconvergence at the swap. **Open Q6** |

## 4.9 Color management & post

| ID | Item | Status | Effort | Sprint | Notes / source |
|---|---|---|---|---|---|
| C1 | **Tonemapper base decision**: keep AgX or switch to Khronos PBR Neutral (keeps an authored analytic palette faithful — i.e. `landcover`). **Reject ACES** (saturated sky + green veg + bright sun = textbook hue skew) | ☐ | Low | THIS | one A/B; **Open Q2** |
| C2 | CA + grain: reduce/gate for a long-sightline planet sim (the `just preview` tool already strips them) | ☐ | Low | THIS? | **Open Q4** |
| — | Single exposure authority | see W10/F2 | — | THIS | tonemapper A/B should run *after* W9 changes the inputs underneath it |

---

# 5. Shared shader library architecture

All surface shading routes through one set of libraries registered in
`PlanetLightingPlugin` (`shading/mod.rs`).

| Library | Import path | Provides |
|---|---|---|
| `lighting.wgsl` | `thalos::lighting` | `SceneLighting`, `ThalosSurface`, `shade_surface`, `shade_foliage`, `shade_hapke_surface`, `surface_brdf`, `compute_surface_sky`, `moonlight_radiance`, `object_aerial_recession`, `sun_daylight`, specular AA |
| `atmosphere.wgsl` | `thalos::atmosphere` | `AtmosphereBlock`, `integrate_atmosphere`, `composite_clouds` |
| `landcover.wgsl` | `thalos::landcover` | `vegetation_color`, `forest_coverage`, `snow_coverage` (CPU mirror: `ground/landcover.rs` — to be retired, see TM3) |
| `shadow.wgsl` | `thalos::shadow` | `ShadowCascadeBlock`, `sun_shadow_factor` |
| `foliage.wgsl` | `thalos::foliage` | foliage albedo model (near mesh + impostor bake) |
| `grass_displace.wgsl` | `thalos::grass_displace` | `grass_blade_world_pos` |

**Invariant:** every surface material derives lighting from these libraries. No
material-local BRDF or palette fork. When a parameter moves, it moves in one place.

**Planned growth (the foundation, §3):**
- **Metallic conductor branch** in `surface_brdf`: lerp F0 from `DIELECTRIC_F0`
  toward `albedo` by `ThalosSurface.metallic` (read *nowhere* today), make
  Fresnel/env-reflection `vec3`, zero the diffuse lobe as metallic→1.
- **One shared view-level scene+atmosphere bind group** so any material
  (incl. the `StandardMaterial`-derived hull/structure path) can call
  `shade_surface` without re-tripping the Metal buffer budget.
- **`ThalosSurface.occlusion`** fed by the screen-space AO pass (W8).
- **Prefiltered env IBL** from the sky-view LUT (W7) replacing the analytic
  hemisphere specular for low-roughness/metallic surfaces.
- **Wire `FOLIAGE` / `WATER` branches** and retire the parallel `shade_foliage`
  + `body_water.wgsl` paths.

## Crate boundary — mechanism vs drivers

Rendering splits into two concerns on a **state-in / pixels-out** boundary
(see `docs/architecture.md` Phase 4):

- **Mechanism — "how to shade":** materials, shaders, the `thalos::lighting`
  spine, the camera/post bundle, and the render-graph nodes (scene-depth, the
  sun-shadow rig, and the planned AO + sky-view-LUT + env-IBL passes). Sim-agnostic,
  reusable across binaries. **Home: `thalos_body_render`** — already the de-facto
  render crate (consumed by both `thalos_game` and `thalos_shipyard`). It grows
  toward a renamed `thalos_render`; moving the remaining stragglers (`scene_depth` /
  `sun_shadow` nodes from `game`, the env probe) + the rename is **one focused
  follow-up AFTER the foundation lands**, so the full mechanism set is extracted
  once. *(The craft **hull material** already moved here — `body_render::craft`,
  Phase 4a in `docs/architecture.md` — which is what unblocks F6b: the hull can now
  sample `thalos::shadow`. The editor itself has since moved out of
  `thalos_shipyard` into `thalos_game::shipyard_editor`; the remaining Phase-4a
  follow-up is the material-application split + dependency flip.)*
- **Drivers — "what to shade this frame":** systems that read `SolarSystemState` /
  `CraftState` / `Simulation` → fill uniforms, decide LOD swaps, spawn bodies,
  anchor grass/clouds/shadow in f64. Inherently sim-coupled (~90% of
  `game::rendering` today, by sim-state reference count). **Home: `thalos_game`** —
  these stay in `game`; they are *not* extracted.

**Rule for foundation (F1–F9) code:** new *mechanism* (AO node, sky-view LUT,
env-IBL, the shared scene/atmosphere bind group, the metallic branch) lands in
`thalos_body_render`; systems that read sim state (e.g. F1's `SceneLighting`
projection) stay in `thalos_game`. New code lands in its final home once — no
double churn.

---

# 6. Verification

I can't see the game — every visual item lands behind a `just game [mode]`
screenshot from the user. Structural changes are announced before they're made
(per CLAUDE.md) and reflected into `docs/terrain.md` / `vegetation.md` /
`atmosphere.md` / `control.md` as they land.

- **Headless regression check:** `just preview` renders the diorama gallery at a
  distance well inside `OBJECT_AERIAL_NEAR_M`, so close-up objects must look
  identical pre/post aerial-recession changes. Use it for geometry/material work.
- **Distance/atmosphere/shadow tuning:** needs a `just game runway` / `cruise` /
  `landing` screenshot.
- **The one-world checks** (§2.3): a ship should shadow the grass and be shadowed
  by a hangar/tree (W5); a ship parked behind a hill should be in shade (W12); a
  ship at Nyx should be dramatically dimmer than at Thalos (F1); the hull, ground,
  and buildings should all warm/cool together through a sunset (F1/F4).

---

# 7. Decision log & open questions

## Decisions taken (2026-06-30 architecture review)

- **Foundation before fidelity.** Land the §3 unification (F1–F6) + Tier-1
  vegetation (W1–W6) this sprint, rather than front-loading eye-candy on top of
  the two-universes split.
- **Shared inputs (B) before shared BRDF (A).** B is low-risk and a prerequisite
  for A. Within A, **structures port before the hull** (the low-risk half).
- **Terrain stays analytic-backbone + tiling-detail** (frequency-band model),
  *not* unique-per-texel textures and *not* runtime VT/GPU-synthesis yet.
- **One environment from the atmosphere** (sky-view LUT → SH → prefiltered env) is
  the keystone — it unifies ambient + IBL across both universes.

## Open questions for the lead

1. **Hull-port scope** — commit to the full hull `shade_surface` port this cycle,
   or land B + IBL + AO + the *structures* port and re-evaluate whether the hull's
   gap justifies losing `StandardMaterial`'s clearcoat/anisotropy?
2. **Tonemapper base** — keep AgX or A/B against Khronos PBR Neutral? (ACES
   rejected regardless.) Run the A/B before or after W9 changes the inputs?
3. **Exposure authority** — single EV100 luminance metering (needs re-tuning
   intensities to physical-ish EV + accepting eye-adaptation on orbit↔surface), or
   keep the artist `CameraExposure` curve and just *freeze* `AutoExposure`?
4. **CA + film grain** — reduce/gate for a long-sightline sim, or keep "mild CA +
   grain" as the intended sensor-sim identity?
5. **Vegetation driver unification (VEG-R)** — fold the three clipmap drivers into
   one `VegLayer` now (prevents fade/AGL/churn/clearing drift while we touch all
   three for W1/W3/W4), or defer the pure refactor?
6. **Slice-6 distant-body view (W17)** — in scope this cycle (UDLOD at all ranges
   + delete the impostor branch + ~2.5k lines of dead bake code), or ride the
   interim impostor and mask the swap with W1?
7. **TAA gate (W13)** — sanction it as a mid-term whole-scene track (dithered
   LOD and vegetation/edge stability; needs body-fixed motion vectors under
   big_space), or stay SMAA-only? Cloud-local reconstruction is decided inside
   the cloud program and does not wait on this choice.
8. **Sky-view LUT light model** — LUT the sun only (keep moon/star analytic;
   simpler, recommended) or rebuild sky-view per dominant light (correct moonlit
   reflections, more cost)? Blocks W7's parameterization.

---

*Key file anchors:* spine `crates/body_render/src/shading/shaders/lighting.wgsl`
(metallic stub @564, F0 @455); craft divergence `assets/shaders/ship_part.wgsl` +
`crates/game/src/rendering/lighting.rs` (magic constants @267/272/273/388,
physical flux @126, flat hull lux @373); terrain material
`crates/body_render/src/ground/body_terrain.wgsl` + `landcover.wgsl` ↔
`crates/body_render/src/ground/landcover.rs`; shadow rig
`crates/game/src/rendering/sun_shadow.rs` + `shading/shaders/shadow.wgsl`;
atmosphere `crates/body_render/src/shading/{atmosphere.rs,multi_scatter.rs,sky_view.rs}` +
`ground/body_sky.wgsl`; reflection probe `crates/game/src/reflection_probe.rs`;
post `crates/body_render/src/impostor/post_stack.rs` (AgX @37).
