# INC-20260826T124420Z-tile-prepass-material-bind-group: GPU-displaced tiles killed the process at prepass pipeline creation

- **Date:** 2026-08-26 · **Surface:** every `just game` mode with tile ground, and
  every ground `just screenshot` preset (`spaceport-aerial`, `craft-stance`, …)

## Symptom

Fatal at the first frame that wanted to draw terrain — Bevy's render error handler
quits the app, followed by a wall of `CommandQueue has un-applied commands`:

```text
ERROR bevy_render::error_handler: Caught rendering error: Validation Error
  In Device::create_render_pipeline, label = 'pbr_prepass_pipeline'
    Error matching ShaderStages(VERTEX) shader requirements against the pipeline
      Shader global ResourceBinding { group: 3, binding: 111 } is not available
      in the pipeline layout
        Binding is missing from the pipeline layout
```

`just screenshot spaceport-aerial` reproduces it exactly and refuses the shot
("capture INVALID: 2 error(s) logged during this run") — the capture lane's error
gate caught it, so this never silently produced a plausible-but-wrong PNG.

Two red herrings worth naming, because both cost time:

- `pbr_prepass_pipeline` appears nowhere in `bevy_pbr` and nowhere in this repo.
  Bevy's label is `prepass_pipeline`; `StandardMaterial::specialize` prefixes it
  with `pbr_` (`pbr_material.rs`). So the failing pipeline is *some*
  `ExtendedMaterial<StandardMaterial, _>` — that's the only thing the label tells
  you, and it covers both the camera prepass and Bevy's shadow pass.
- Group 3 binding 111 is `tile_position_atlas`, which is bound and working in the
  main pass — so "the material forgot to declare it" is the wrong hypothesis.

## Root cause

Bevy 0.19 skips the material bind group entirely for a **depth-only opaque** pass.
`bevy_pbr::prepass::is_depth_only_opaque_prepass` puts `empty_layout` at group 3
and the phase is drawn by `PrepassOpaqueDepthOnlyDrawFunction` (and, for Bevy's
shadow maps, `ShadowsDepthOnlyDrawFunction` in `render/light.rs`), neither of
which binds it. The ship camera carries `DepthPrepass` and nothing else
(`rendering camera.rs`), so terrain's prepass key is exactly `DEPTH_PREPASS` —
the depth-only case.

`tile_displacement.wgsl` is the shared vertex stage for the visible, prepass,
deferred and shadow pipelines, and it *reads group 3* (bindings 111/112) to find
the vertex position it is meant to place. A pipeline whose vertex shader
references a binding its layout omits fails validation, and a fatal wgpu
validation error takes the process with it.

The opt-out Bevy checks for is `MeshPipelineKey::MAY_DISCARD` — or
`PREPASS_READS_MATERIAL`, which 0.19.0 *defines* and *reads* (in `ALL_PREPASS_BITS`
and in `light.rs`) but never sets anywhere, and which no `Material` impl can reach:
neither the prepass nor the shadow queue ORs `MaterialProperties::mesh_pipeline_key_bits`
into the mesh key. So on 0.19.0 the alpha mode is the only lever there is.

## Fix

`DISPLACED_PREPASS_ALPHA_MODE` in `crates/rendering/render/src/tiles/material.rs`:
both tile extensions override `MaterialExtension::alpha_mode()` to
`AlphaMode::Mask(0.5)`, which sets `MAY_DISCARD`, moves the material to the
alpha-mask phase, and gets the real material layout plus a group-3 bind in the
prepass and the shadow pass.

Nothing is actually masked. Only the *pipeline-level* alpha mode changes; the base
`StandardMaterial` stays `AlphaMode::Opaque`, so the GPU-side material flags still
say opaque and `alpha_discard` forces alpha to 1.0 in both `tile_terrain.wgsl` and
Bevy's prepass fragment. The main-pass pipeline is unchanged apart from a
`MAY_DISCARD` shader def — blend, depth write and depth compare come from the
blend bits, which stay opaque.

This removes the mechanism rather than the symptom: the class of bug is "a vertex
stage that serves a depth-only pass reads its own material bind group", and the
material now declares that it needs the bind group in exactly the way Bevy checks.

Verified by matched before/after on one preset: with the override removed,
`just screenshot spaceport-aerial` fails with the error above; with it in place the
same preset, plus `craft-stance` and `forest-stand`, capture clean with terrain,
contact shadow and depth ordering intact.

Known cost, if the perf lane ever shows it: the depth prepass now compiles a
fragment stage for terrain (Bevy's void `prepass_alpha_discard` entry), which
forfeits early-Z depth writes on that draw. The answer would be a no-op prepass
fragment shader of our own — not removing the alpha mode.

## Recurrence signal

`create_render_pipeline, label = 'pbr_prepass_pipeline'` + "Binding is missing from
the pipeline layout" for any binding ≥ 100. It means a material's prepass/shadow
vertex shader reads its own bind group while the pass is classified depth-only.

Standing rule, now in `.claude/skills/wgsl-bevy/SKILL.md`: **a material whose
`prepass_vertex_shader` reads the material bind group must not be
`AlphaMode::Opaque`.** Adding a displacement material — or "cleaning up" a
terrain material's alpha mode back to opaque — reintroduces a hard crash, not a
visual bug.
