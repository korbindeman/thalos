# Reflection probe — design

Environment-map source for metallic ship-part reflections. Feeds Bevy's
`GeneratedEnvironmentMapLight` on the main camera so `ShipPartMaterial`
panels read the sky (sun + planet + stars) from the ship's orbital
vantage.

## Current implementation

CPU-authored cubemap, rewritten every `REFRESH_INTERVAL` seconds
(0.25 s) from the ship's current state:

- Cubemap `Image` asset, 256³, 6 layers, `Rgba16Float`, cube view
  descriptor. `TEXTURE_BINDING | COPY_DST` usage — no render-attachment,
  since we're not rendering into it.
- Painter reads ship-to-sun and ship-to-planet directions from
  `SimulationState`, plus the planet's angular radius from its physical
  radius and range. Each frame that hits the refresh tick, all 6 faces
  get rewritten: sun disc (HDR hot spot), lit-side planet hemisphere
  with a Lambert terminator, dim starfield tint everywhere else.
- Re-assigns the handle via `Assets<Image>::get_mut` which marks it
  changed; Bevy's runtime filter pipeline re-prefilters diffuse +
  specular mips downstream.

Lives in `crates/game/src/reflection_probe.rs`. ~300 lines.

## Why not render the actual scene into the cubemap

The "correct" path — six cameras rendering the real scene into per-face
views of a cubemap — is blocked in Bevy 0.18 by a layering trap:

Bevy's main-world `camera_system` (in `bevy_render/src/camera.rs`)
resolves `RenderTarget::TextureView(handle)` against the main-world
`ManualTextureViews` resource every frame to read the target size. It
panics if the handle isn't present. Populating main-world
`ManualTextureViews` with real `TextureView`s requires a GPU `Texture`
— but the `Texture` comes from `GpuImage` which only exists in the
**render world**. So the naive pattern (main world creates `Image` →
render-world creates views from `GpuImage` → inserts into
`ManualTextureViews`) leaves main-world's copy empty on frame 1 and
panics.

Workable paths through, none trivial:

1. **Main-world-created `Texture`.** `Res<RenderDevice>` *is* exposed in
   the main world in Bevy 0.18 (`bevy_render/src/lib.rs:416`). We can
   create our own cubemap `Texture` in main world, build face views,
   insert them into `ManualTextureViews`. But `GeneratedEnvironmentMapLight`
   needs a `Handle<Image>` for the env-map binding — so we also need a
   normal `Image` asset and a copy-blit pass from our texture → the
   image's `GpuImage` texture each frame. Workable, ~200 extra lines,
   one render-graph node for the copy.
2. **Custom render-graph subgraph** that runs the 3D subgraph against
   per-face views of the cubemap, all inside the render world — no
   `ManualTextureViews` at all. This is what Bevy does for shadow maps.
   Architecturally right; days of work.

Neither was justified given the visual target (ship reflecting orbit
from Thalos) and the fact that `ship_view` is actively churning.

## Upstream status

- Bevy 0.17: shipped `GeneratedEnvironmentMapLight` — runtime
  pre-filtering from an arbitrary cubemap source. *We use this.*
- Bevy 0.18 release notes: no mention of reflection probes, omni
  cameras, or dynamic env maps.
- [PR #13840 — Implement omnidirectional cameras for real-time
  reflection probes](https://github.com/bevyengine/bevy/pull/13840)
  adds exactly the missing piece (`OmnidirectionalCameraBundle`,
  `ActiveCubemapSides` for round-robin). **Draft, not merged** —
  `S-Waiting-on-Author`, most recent activity Feb 2026.
- [Tracking issue #20212 — realtime env-map filtering](https://github.com/bevyengine/bevy/issues/20212)
  is about improvements to the existing filter pipeline, not the
  render-to-cubemap source side.

## Migration path when upstream lands

`GeneratedEnvironmentMapLight` is the stable contract on the main
camera. Everything behind it can be swapped without touching ship
materials or camera setup:

1. If PR #13840 merges into a release we can use: delete the CPU
   painter, spawn an `OmnidirectionalCameraBundle` parented to the
   ship with `ActiveCubemapSides` configured for round-robin, point
   `GeneratedEnvironmentMapLight.environment_map` at its cubemap.
2. If we implement path (1) or (2) from the previous section
   ourselves: same drop-in — the CPU painter is replaced by whatever
   writes the cubemap.

## Known limits of the current CPU painter

- Planet is a solid-colour Lambert disc, not the actual impostor
  shader. Reflections don't match the on-screen planet in detail
  (surface texture, atmosphere rim).
- Stars are a flat tint, not individual points. Sub-pixel at 256³
  anyway; would need per-star rendering at higher probe res to matter.
- Sun is a clamped-HDR disc. No bloom / flare in reflections — those
  live on the direct light path.
- Planet direction currently keyed off the sim homeworld (Thalos).
  Once ships move far from Thalos this needs to re-pick the nearest
  body.

## Revisit trigger

Switch to real-scene capture when:

- PR #13840 lands in a usable Bevy release, *or*
- The painted-planet divergence from the impostor reads wrong at
  screenshot distance (a consistent visual bug, not a one-off).

Revisit cadence: every 6 weeks or when a major Bevy release ships,
whichever is sooner.
