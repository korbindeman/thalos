# Bevy 0.19 notes

The workspace migrated to **Bevy 0.19** on 2026-07-01. This file is the
load-bearing subset of that migration: the API changes that still bite anyone
editing this codebase, plus the two ordering rules that cost us runtime
regressions. The full write-up lives in the `bevy-019-migration` auto-memory and
the official [0.18→0.19 migration guide](https://bevy.org/learn/migration-guides/0-18-to-0-19/).

Version stack: glam **0.32**, wgpu **29**, avian3d **0.7**, bevy_egui **0.40**,
bevy_enhanced_input **0.26**, gilrs 0.11.

**Render graph is gone — passes are systems.** 0.19 replaced the node-based
`RenderGraph` with ECS schedules. Our custom passes (`scene_depth`,
`sun_shadow`, `film_grain`, the `volumetric_clouds` compute, udlod) are now
**systems in the `Core3d` schedule** (`bevy::core_pipeline::{Core3d,
Core3dSystems}`; sets `Prepass`/`MainPass`/`EarlyPostProcess`/`PostProcess`) or
the root `RenderGraph` schedule (`RenderGraphSystems` `Begin→Render→Submit`).
`RenderContext` + `ViewQuery<D,F>` are **SystemParams** now (`ViewQuery`
auto-skips non-matching views; there is no `render_device()` on `RenderContext`
— add `Res<RenderDevice>`). Order view passes with `.after(main_opaque_pass_3d)
.before(main_transparent_pass_3d)`.

**Two ordering rules are non-negotiable** (both cost us runtime regressions):
- **Any post pass that calls `post_process_write()` must sit in the exact chain
  slot its old node held** — set membership *and* a relative `.after()` are both
  load-bearing. 0.19's `ViewTarget` ping-pong parity index is a persistent
  `Arc<AtomicUsize>` reused across frames, so a mis-slotted flip makes the
  presented buffer alternate → global brightness flicker. `film_grain` must be
  `.in_set(Core3dSystems::PostProcess).after(…::cas)` — last inside PostProcess,
  before the after-PostProcess UI/upscaling consumers.
- **Retained binned render phases**: mutating a material every frame (e.g.
  udlod's per-frame lighting write to `BodyTerrainMaterial`) flags it dirty, and
  Bevy's `queue_material_meshes` runs `phase.remove(main_entity)` for dirty
  entities. A custom queue system must run **after** Bevy's:
  `queue_terrain::<M>.after(RenderSystems::QueueMeshes).before(RenderSystems::PhaseSort)`,
  or it gets dequeued after it adds itself and never draws.

**Resources are components now.** `#[derive(Resource)]` also implements
`Component`. Broad `EntityRef` / `Query<Entity>`-style queries can conflict with
resource access — our `PartQuery` (fuel.rs / staging.rs) filters
`Without<bevy::ecs::resource::IsResource>` to avoid the B0001 panic; keep that on
any broad part query. Also: 0.19 validates `Res<T>` at **fetch time** and panics
if absent, so a `RenderStartup` system reading a resource another `RenderStartup`
system creates must `.after()` it (udlod pins
`init_terrain_render_pipeline::<M>.after(bevy::pbr::init_mesh_pipeline_view_layouts)`).

**`Image::new` cannot carry mip chains.** Its size debug-assert compares
`data.len()` against the level-0 volume only, ignoring `mip_level_count` — so
building a mip-mapped image (e.g. the cloud weather cube) via `Image::new`
panics in dev builds ("Pixel data, size and format have to match"). Use
`Image::new_uninit`, set `texture_descriptor.mip_level_count`, then assign
`image.data`. Layout is `TextureDataOrder::LayerMajor` (layer0[mip0..], …),
matching Bevy's default upload.

**Text moved cosmic-text → Parley.** `TextFont.font` is a `FontSource` (not
`Handle<Font>`; `.into()` a handle or name a family), `font_size` is
`FontSize::Px(f32)` (bare `.into()` on an f32 literal mis-infers — write
`FontSize::Px(N)`). The shared `HudTheme.font` is a `FontSource`.

**Notable renames/moves** if you touch these areas: `bevy_scene` →
`bevy_world_serialization` (`Scene`→`WorldAsset`, `SceneRoot`→`WorldAssetRoot`;
we don't use runtime scenes), atmosphere moved `bevy_pbr`→`bevy_light`, `Hdr`
→`bevy::camera::Hdr`, light `shadows_enabled`→`shadow_maps_enabled`,
`ShaderStorageBuffer`→`ShaderBuffer`, `insert_non_send_resource`→`insert_non_send`.
wgpu 29: pipeline `push_constant_ranges`→`immediate_size: u32`,
`DepthStencilState.depth_write_enabled/depth_compare` are `Option<_>`.

**0.19 features we deliberately do NOT use** (we have custom replacements):
Bevy's Skybox, the new BSN / Next-Gen Scenes, rectangular area lights,
`EditableText`, and Bevy's built-in atmosphere. The custom `BodySky` raymarch
is the sole rocky-body atmosphere (ADR-20260721T185221Z-custom-rocky-atmosphere);
do not restore Bevy's camera-local proxy, a capture-only backend, or a live/
persisted selector. **Worth evaluating for the graphics sprint** (new in
0.19, not yet adopted): **contact shadows** (screen-space, kills close-geometry
peter-panning — complements our `thalos::shadow` rig), **physically-based SSR**,
**parallax-corrected cubemaps** (relevant to the F3/F4 IBL work), and the
vignette/lens-distortion post FX. See `docs/roadmap/graphics_fidelity.md` before pulling
any of these in — they must obey the one-world / spine rules, not bypass them.

