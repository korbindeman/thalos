# Graphics fidelity plan

Living tracking doc for the push toward MSFS / KSP2-tier visuals. Visual
direction: **stylized but high fidelity** — lean on lighting, shading, and a
controlled-but-rich palette, *not* photo textures. KSP2 (not 1) and MSFS are
the references.

## Status legend

- ☐ not started · ◐ in progress · ☑ landed (compile-verified) · ✅ runtime-verified by a `just game` screenshot

## Where the renderer already is

The pipeline is feature-rich; the flatness in current screenshots is a few
specific gaps + tuning, not missing machinery. Already wired:

- TonyMcMapface tonemap, distance-driven `CameraExposure`, auto-exposure, film
  grain (`impostor/post_stack.rs`, `rendering/lighting.rs`).
- Bloom, SMAA, CAS sharpening, mild chromatic aberration, deband dither
  (`impostor/post_stack.rs`).
- Physically-integrated per-body atmosphere with **aerial perspective** keyed to
  scene depth, multi-scatter LUT, volumetric cloud slab (`ground/body_sky.wgsl`,
  `AtmosphereTuning`).
- Terrain: rough-dielectric BRDF (Oren–Nayar + Cook–Torrance + Kulla–Conty) with
  altitude/slope **ecological albedo bands**, material masks, ~20 m albedo
  breakup noise, ~1.25 m micro-relief normals (`ground/body_terrain.wgsl`).
- Vegetation: meshed trees w/ octahedral far impostors, 5-ring grass clipmap to
  ~1.5 km, wind on both, shared scene lighting (`ground/tree.wgsl`,
  `grass.wgsl`, `scatter.rs`, `vegetation.rs`).

## The gaps (ranked by visual impact)

1. **Nothing in the world casts shadows.** CSM is ship-only, 2 cascades, 500 m;
   all terrain/vegetation tagged `NotShadowCaster`/`NotShadowReceiver`. The
   forest is a uniformly-lit carpet → reads flat. *Dominant tell.*
2. **No ambient occlusion** (SSAO/GTAO off). Trees/aircraft float; canopies have
   no interior depth.
3. **Monochrome desaturated palette.** One olive green everywhere; all broadleaf
   trees share one leaf atlas, ±11% hue jitter only.
4. **Uniform forest composition.** Clump field exists but reads as even fill.
5. **Muddy aerial perspective + plastic aircraft.** Distance goes grey-green not
   blue-haze; hull has no IBL/reflection probe.

## Workstreams

### 1. World shadows — *highest impact*

**Architecture finding (2026-06-27).** Terrain does **not** receive shadows via
Bevy CSM — the UDLOD pass is custom and receives shadows only in its own shader:
the analytic craft proxy (`BodyTerrainShadow`, capped at 24 capsules + 8 quads,
CPU-uploaded each frame in `local_craft_shadow`, `rendering/ground_terrain.rs`)
plus a height-field self-shadow march (`terrain_self_shadow`,
`body_terrain.wgsl`). Trees/grass use fully custom WGSL lighting (wrap-diffuse +
hemisphere sky) and sample no shadow map, so they neither receive CSM nor cast
on each other/the ground visibly. Consequence: the originally-planned "extend the
craft proxy to trees" is unworkable — thousands of trees can't go through a
32-slot analytic uniform.

- ☐ **1a. Blob/contact shadow discs.** Soft dark disc baked into each tree's
  instanced draw at its base. Grounds trees, scales trivially, touches neither
  terrain shader nor CSM. Contact-AO, not directional. *Quick low-risk win.*
- ✅ **1b. Sun-aligned shadow texture sampled by the terrain shader.**
  *Runtime-verified 2026-06-27 — runway + craft shadows visible on the ground.*
  Tree-on-tree + grass-receiving **deferred** at user's call. Increment 1: tree
  **mesh** tiles cast; **terrain receives**. Tree-on-tree, grass-receiving, and
  craft-as-caster are follow-ups. Implemented in
  [`rendering/sun_shadow.rs`](crates/game/src/rendering/sun_shadow.rs) +
  `body_terrain.wgsl` (`sun_shadow_factor`) + `BodyTerrainExtras`/material
  binding. Matrix convention verified against `bevy_camera` source (orthographic
  reverse-z, `(far, near)` swap). Tunables: `SHADOW_STRENGTH`,
  `SHADOW_DEPTH_BIAS`, `SHADOW_REGION_HALF_EXTENT_M`, `SHADOW_MAX_ALTITUDE_M`,
  3×3 PCF. **Open risks for the screenshot:** compare-direction sign, acne/peter-
  pan bias, and whether `tree.wgsl` alpha-discards (leaf-shaped vs blocky
  shadows).

  **Design (chosen against the real code):**
  - New module `rendering/sun_shadow.rs`. A plain orthographic `Camera3d`
    *outside* big_space (like the map camera), on a dedicated
    `SHADOW_CASTER_LAYER = 8`, render target a 2048² color image, `Msaa::Off`,
    `order = -1`. Positioned each frame from the ship camera's render-space
    position + the active body's render-space sun direction (inertial dirs ==
    render-space dirs; no rotation). Active only near a vegetated surface
    (gated on `ActiveCloudBody` + camera altitude) to avoid a 2048² pass in
    orbit.
  - Reuse the `scene_depth` pattern: a render-graph `ViewNode`
    (`CopySunShadowDepth`, filtered to the `SunShadowCamera` marker) copies the
    shadow view's `ViewDepthTexture` into a `Depth32Float` `SunShadowImage`
    (`COPY_DST | TEXTURE_BINDING`), sampled as `texture_depth_2d` via
    `textureLoad` (manual PCF, no comparison sampler — matches `body_sky.wgsl`).
  - Caster tagging: tree **mesh** tiles (`vegetation.rs`) get
    `RenderLayers::from_layers(&[SHIP_LAYER, SHADOW_CASTER_LAYER])` so the same
    `TreeMaterial` draw (with its leaf alpha-discard) writes leaf-shaped depth.
    Impostor tiles stay non-casters. Craft parts added in a later pass.
  - The shadow `view_proj` (Bevy reverse-z ortho, `Mat4::orthographic_rh` with
    near/far swapped to match Bevy) + params packed into the existing
    `BodyTerrainExtras` uniform (avoids a new vertex buffer — Metal 16-slot
    cap). New `#[texture(3, sample_type="depth")]` binding on
    `BodyTerrainMaterial` bound to `SunShadowImage` on every instance (map
    terrain too, gated off via `params.x = 0`).
  - `body_terrain.wgsl`: project fragment render-space pos → shadow clip →
    `textureLoad` depth → biased reverse-z compare with PCF → multiply the
    direct sun term. Sign/bias are screenshot-tunable.
  - Convention risk (matrix/reverse-z/compare direction) is the main blind
    spot; built to match Bevy's ortho, flipped after the first screenshot.
- ✗ **1c. Wire custom shaders into Bevy CSM.** Rejected — couples the custom
  UDLOD/tree pipelines to Bevy shadow bind groups, exactly what the project
  deliberately sidestepped.

### 2. Ambient occlusion (GTAO)
- ☐ Add `ScreenSpaceAmbientOcclusion` to the ship camera; verify coexistence with
  the `scene_depth` copy node + depth prepass. Tune low/stylized.

### 3. Palette & lighting — *stylized-vivid target (user's call)*
- ◐ **Ground palette + lighting pass 1** (`body_terrain.wgsl` + `lighting.wgsl`).
  Pushed `C_*` anchor chroma back up (greens get a wider green-vs-red/blue gap;
  drab→alive without the old chartreuse), cut `BREAKUP_HUE_AMT` 0.07→0.035 +
  `BREAKUP_VALUE_AMT` 0.24→0.20 (kills the brown muddy smears), and in the shared
  lighting lib `SURFACE_DIRECT_SCALE` 0.20→0.23 (punchier sun) +
  `SURFACE_SKY_CHROMA_GAIN` 6.0→8.0 (more saturated blue sky/shadows). *Awaiting
  screenshot to dial in.* These lighting knobs are shared by grass/trees/impostor
  too, so the whole surface family stays in sync.
- ☐ Tree hue/value variance + species palettes + leaf-atlas variants. *(Deferred
  with the rest of the vegetation work.)*

### 5b. Atmosphere retune + aircraft IBL
- ◐ **Atmosphere pass 1** — the noon distance washed to a grey-tan band. Root
  cause: authored `mie_optical_depth: 0.06` ("hazier humid day"); Mie is
  wavelength-independent grey haze so it desaturates everything at range. Cut to
  `0.025` (clean continental) + `mie_scale_height_m` 1600→1200 (haze hugs
  valleys) in `thalos.ron`, and `aerial_perspective_strength` 0.15→0.10
  (`AtmosphereTuning`). *Awaiting noon screenshot.* (thalos.ron is a runtime
  asset — Mie tweaks only need a restart, not a rebuild.)
- ☐ Reflection probe / `EnvironmentMapLight` on the hull so it reads as metal,
  not grey clay; tune hull metallic/roughness.

### 4. Forest composition
- ☐ Strengthen clump contrast (clearings/meadows), tie density to moisture/slope
  masks, treeline falloff.

### 5. Atmosphere retune + aircraft IBL
- ☐ Retune Mie/Rayleigh + `aerial_perspective_strength` for crisp blue haze.
- ☐ Add reflection probe / `EnvironmentMapLight` to the ship; tune hull material.

## Verification

I can't see the game — every item lands behind a `just game [mode]` screenshot
from the user. Tuning items (3/4/5) need several screenshot rounds. Structural
changes (esp. 1b CSM caster re-tagging) get announced before they're made, per
CLAUDE.md, and the shadow/AO/atmosphere additions get reflected into
`docs/terrain.md` / `docs/vegetation.md` / `docs/atmosphere.md` as they land.
