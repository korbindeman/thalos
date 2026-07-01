//! Startup system that spawns one entity tree per body in the solar
//! system: a `CelestialBody` root with impostor billboard, halo, icon,
//! ground-LOD terrain, and (where applicable) ring children.
//!
//! Stars, gas giants, and solid-impostor bodies spawn fully here.
//! Procedural bodies (`body.terrain.is_some()`) split their setup in
//! two: this system spawns the shell entities (root, optional
//! tidal-lock tags, icon, `BodySky`) and dispatches an
//! `AsyncComputeTaskPool` task that performs the heavy
//! `target/bakes/<name>.bin` decode and `prepare_planet_bake`. The
//! polling system in `super::generation::poll_planet_install_tasks`
//! drains those tasks each frame and finishes the install (GPU upload,
//! impostors, halos, terrain, water). Each completed install ticks the
//! [`crate::loading::LoadingTracker`] step that drives the startup
//! loading screen. Missing or stale bakes still panic with a message
//! pointing at `just bake <name>` — just from the task instead of the
//! main thread. Bodies with no authored terrain (`TerrainConfig::None`)
//! fall through to the [`SolidPlanetMaterial`] impostor tinted with
//! `body.color`.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::image::Image;
use bevy::light::cascade::CascadeShadowConfigBuilder;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::math::DQuat;
use bevy::prelude::*;
use big_space::prelude::Grid;
use thalos_body_render::udlod::prelude::PreciseRotation;
use thalos_body_render::{
    AtmosphereBlock, GasGiantLayers, GasGiantMaterial, GasGiantParams, GpuAtlasMirrorHeightSource,
    MULTI_SCATTER_LUT_HEIGHT, MULTI_SCATTER_LUT_WIDTH, ReferenceClouds, RingLayers, RingMaterial,
    RingParams, SceneLighting, SolidPlanetHaloMaterial, SolidPlanetMaterial, SolidPlanetParams,
    bake_impostor_albedo_cube, bake_multi_scatter_lut, blank_impostor_cube, build_ring_mesh,
    cloud_cover_image_for_body,
};
use thalos_body_render::{BodySkyExtra, BodySkyMaterial};
use thalos_physics_canonical::canonical::Epoch;
use thalos_terrain::{ProceduralSurface, TerrainConfig};
use thalos_world::BodyKind;

use super::generation::{ProceduralInstallExtras, WorldStateAssets};
use super::ground_terrain::{BodySky, RealSpaceImpostor};
use super::real_space::{RealSpaceRoot, real_space_grid};
use super::scene_depth::SceneDepthImage;
use super::types::{
    BodyIcon, BodyMesh, CelestialBody, GasGiantMaterials, MapRingMaterial, MoonLight, RealSpaceBody,
    ShipBodyMesh, ShipRingMaterial, SimulationState, SolidPlanetMaterials, SunLight, TidallyLocked,
};
use crate::coords::{MAP_LAYER, MAP_SCALE, SHIP_LAYER, SHIP_SCALE};
use crate::loading::LoadingTracker;
use crate::view::HideInShipView;
use std::sync::Arc;

// (Bake cache-key + local-bake-dir helpers removed in 0b-1 — procedural bodies
// generate terrain at runtime and need no bake.)

/// Bake the atmosphere's multi-scatter LUT and upload it as a small
/// linear-sampled image for `BodySkyMaterial` / `body_sky.wgsl`.
///
/// The bake (`thalos_body_render::bake_multi_scatter_lut`) emits RGBA f32,
/// but f32 textures are not linear-filterable without the `FLOAT32_FILTERABLE`
/// wgpu feature (which this project does not request). The LUT is a smooth,
/// low-dynamic-range radiance field, so we repack to `Rgba16Float` — filterable
/// everywhere and far more than enough precision at 32×32. Sampled with
/// `ImageSampler::linear()` (linear filtering + clamp-to-edge), matching the
/// `textureSampleLevel` lookup in `integrate_atmosphere_multiscatter`.
fn build_multi_scatter_lut(
    atmosphere: &AtmosphereBlock,
    planet_radius_render: f32,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    use bevy::asset::RenderAssetUsages;
    use bevy::image::ImageSampler;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

    let f32_bytes = bake_multi_scatter_lut(
        atmosphere,
        planet_radius_render,
        MULTI_SCATTER_LUT_WIDTH,
        MULTI_SCATTER_LUT_HEIGHT,
    );
    // RGBA f32 little-endian → RGBA f16 little-endian.
    let mut data = Vec::with_capacity(f32_bytes.len() / 2);
    for chunk in f32_bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        data.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }

    let mut image = Image::new(
        Extent3d {
            width: MULTI_SCATTER_LUT_WIDTH,
            height: MULTI_SCATTER_LUT_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::linear();
    images.add(image)
}

/// 1×1 "clear" cloud layer (RGBA32F `(0, 0, 0, 1)` → transmittance 1) used as
/// the [`BodySkyMaterial::cloud_layer`] fallback. The game swaps in the live
/// `thalos_volumetric_clouds` texture for the active cloud body; every other
/// body keeps this, so the cloud composite in `body_sky.wgsl` is a no-op there.
fn blank_cloud_layer(images: &mut Assets<Image>) -> Handle<Image> {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
    // RGBA32F (0.0, 0.0, 0.0, 1.0), little-endian (1.0f32 = 0x3F80_0000).
    let data = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x80, 0x3F];
    let mut image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    images.add(image)
}

/// Blank fallback cloud textures shared by every body's `BodySky`.
/// `ground_terrain::update_body_terrain_atmosphere` binds the live volumetric
/// textures on the active cloud body and rebinds these on every other body,
/// so a body that stops being active sheds its stale cloud layer.
#[derive(Resource, Clone)]
pub(super) struct BlankCloudTextures {
    pub layer: Handle<Image>,
    pub distance: Handle<Image>,
}

/// 1×1 far-sentinel cloud-distance fallback (R32F `1e9` = "no cloud on this
/// ray") for [`BodySkyMaterial::cloud_distance`]; same swap policy as
/// [`blank_cloud_layer`].
fn blank_cloud_distance(images: &mut Assets<Image>) -> Handle<Image> {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
    let data = 1.0e9_f32.to_le_bytes().to_vec();
    let mut image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    images.add(image)
}

/// 1×1 blank multi-scatter LUT (`Rgba16Float` `(0,0,0,1)`) bound by
/// [`SolidPlanetMaterial`] on airless bodies. Their vacuum atmosphere makes the
/// shader's `atmosphere_scattering_active` gate skip the sample entirely, so the
/// contents are never read — but the binding must still be a valid texture, and
/// it must match the real LUT's bind-group layout (`Rgba16Float`, linear).
fn blank_multi_scatter_lut(images: &mut Assets<Image>) -> Handle<Image> {
    use bevy::asset::RenderAssetUsages;
    use bevy::image::ImageSampler;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
    // Rgba16Float (0.0, 0.0, 0.0, 1.0): f16 0.0 = 0x0000, 1.0 = 0x3C00, LE bytes.
    let data = vec![0, 0, 0, 0, 0, 0, 0x00, 0x3C];
    let mut image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image.sampler = ImageSampler::linear();
    images.add(image)
}

pub(super) fn spawn_bodies(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut gas_giant_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
    mut solid_planet_materials: ResMut<Assets<SolidPlanetMaterial>>,
    mut solid_planet_halo_materials: ResMut<Assets<SolidPlanetHaloMaterial>>,
    mut sky_materials: ResMut<Assets<BodySkyMaterial>>,
    sim: Res<SimulationState>,
    real_root: Res<RealSpaceRoot>,
    scene_depth: Res<SceneDepthImage>,
    reference_clouds: Res<ReferenceClouds>,
    mut procedural_install: ProceduralInstallExtras,
    mut loading_tracker: ResMut<LoadingTracker>,
    mut world_state: WorldStateAssets,
) {
    let bodies = &sim.system.bodies;
    let initial_states = sim.ephemeris.states(Epoch::ZERO);

    // Shared meshes.
    let icon_mesh = meshes.add(Circle::new(1.0));
    // Unit rectangle (corners at ±1) shared across all planet billboards.
    // The vertex shader scales it by params.radius each frame.
    let billboard_mesh = meshes.add(Rectangle::new(2.0, 2.0));
    // Fallback cloud layer + cloud-distance for every body's BodySky; the live
    // volumetric cloud textures are bound per-frame for the active cloud body.
    let blank_cloud = blank_cloud_layer(&mut images);
    let blank_cloud_dist = blank_cloud_distance(&mut images);
    // 1×1 fallback impostor cube for solid-colour bodies (they use the flat
    // `albedo`, gated by `albedo.w < 0.5`, and never sample it).
    let blank_impostor = images.add(blank_impostor_cube());
    // 1×1 blank multi-scatter LUT for airless `SolidPlanetMaterial` impostors
    // (their vacuum atmosphere gate skips the sample; the binding just needs to
    // be a valid, layout-matching texture).
    let blank_ms_lut = blank_multi_scatter_lut(&mut images);
    commands.insert_resource(BlankCloudTextures {
        layer: blank_cloud.clone(),
        distance: blank_cloud_dist.clone(),
    });
    // Star icosphere — emissive star meshes still use a real sphere
    // (no impostor). Procedural bodies and gas giants render as
    // camera-facing quads (`billboard_mesh`); solid-impostor bodies
    // likewise. No body needs an icosphere any more.
    let unit_sphere_star = meshes.add(Sphere::new(1.0).mesh().ico(5).unwrap());

    for body in bodies {
        let state = &initial_states[body.id];
        let real_grid = real_space_grid();
        let (real_cell, real_offset) = real_grid.translation_to_grid(state.position);
        let real_body_entity = commands
            .spawn((
                Transform::from_translation(real_offset),
                real_cell,
                Grid::new(super::real_space::REAL_SPACE_CELL_SIZE_M, 0.0),
                Visibility::Inherited,
                RealSpaceBody { body_id: body.id },
                // f64 companion to this grid's f32 `Transform.rotation`, read by
                // udlod's high-precision Taylor vertex path. Set each frame
                // alongside the f32 rotation in `update_real_space_body_positions`;
                // identity until then matches the spawn-time f32 rotation.
                PreciseRotation(DQuat::IDENTITY),
                Name::new(format!("{} Real Space", body.name)),
                ChildOf(real_root.entity),
            ))
            .id();

        // Unified atmosphere fullscreen pass. Spawned here (not after
        // terrain bake) so it's live from frame one — otherwise a
        // cmd-shift-click teleport before the body's terrain task finishes
        // lands inside an atmosphere shell that has no `BodySky` entity to
        // render and the sky comes up black. The entity needs nothing from
        // the bake; the atmosphere block is read straight from the body's
        // authored config. `BodySky` visibility is hidden when the camera
        // is outside the shell (see `sync_body_render_lod`).
        let ship_atmosphere = body
            .terrestrial_atmosphere
            .as_ref()
            .map(|a| AtmosphereBlock::from_terrestrial(a, (1.0 / SHIP_SCALE) as f32))
            .unwrap_or_default();
        // Bake the multi-scatter LUT once from the static atmosphere block, and
        // share it across this body's `BodySky` fullscreen pass AND its
        // solid-planet impostor (map + ship). SHIP_SCALE == 1 (1 render unit =
        // 1 m), so the body's solid radius in render units is just `radius_m`;
        // this must match the units of `ship_atmosphere.atmos_geom.x` (already
        // scaled by `inv_m` in `from_terrestrial`), which it does at SHIP_SCALE.
        // The LUT is scale-invariant (a function of normalized altitude + sun
        // angle), so this SHIP_SCALE bake is reused unchanged for the MAP_SCALE
        // disc impostor. Airless bodies bind the shared 1×1 blank.
        let planet_radius_render = (body.radius_m * SHIP_SCALE) as f32;
        let multi_scatter_lut = if ship_atmosphere.atmos_geom.z > 0.0 {
            build_multi_scatter_lut(&ship_atmosphere, planet_radius_render, &mut images)
        } else {
            blank_ms_lut.clone()
        };
        if ship_atmosphere.atmos_geom.z > 0.0 {
            let (sky_cloud_cover, _) =
                cloud_cover_image_for_body(&body.name, &reference_clouds, &mut images);

            let sky_material = BodySkyMaterial {
                atmosphere: ship_atmosphere,
                atmosphere_extra: BodySkyExtra::default(),
                scene_depth: scene_depth.handle.clone(),
                cloud_cover: sky_cloud_cover,
                multi_scatter_lut: multi_scatter_lut.clone(),
                cloud_layer: blank_cloud.clone(),
                cloud_distance: blank_cloud_dist.clone(),
            };
            commands.spawn((
                Mesh3d(billboard_mesh.clone()),
                MeshMaterial3d(sky_materials.add(sky_material)),
                bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER),
                NoFrustumCulling,
                NotShadowCaster,
                NotShadowReceiver,
                // Start hidden — the visibility-culling system flips it on
                // while the body's ground terrain LOD is active.
                Visibility::Hidden,
                ChildOf(real_body_entity),
                Name::new(format!("{} Sky", body.name)),
                BodySky { body_id: body.id },
            ));
        }

        // Map-side bodies stay under normal Bevy transforms. Ship-layer
        // body meshes are children of the matching real-space BigSpace grid.
        let pos = (state.position * MAP_SCALE).as_vec3();

        let render_radius = ((body.radius_m * MAP_SCALE) as f32).max(0.005);
        let ship_render_radius = ((body.radius_m * SHIP_SCALE) as f32).max(0.005);

        let [r, g, b] = body.color;
        let base_color = Color::srgb(r, g, b);
        let is_star = body.kind == BodyKind::Star;

        // Icon material: unlit, emissive, double-sided flat circle. Alpha is
        // driven per-frame by `sync_body_icons` to crossfade against the
        // impostor mesh as the body shrinks through the icon threshold.
        let icon_material = std_materials.add(StandardMaterial {
            base_color: base_color.with_alpha(0.0),
            emissive: LinearRgba::new(r, g, b, 0.0) * 2.0,
            unlit: true,
            double_sided: true,
            alpha_mode: AlphaMode::Blend,
            // Sort tiebreak among Transparent3d items at the same
            // body-center depth. Note this only affects sort order, not
            // the actual fragment depth.
            depth_bias: 10.0,
            ..default()
        });

        let body_entity = if is_star {
            // Stars keep the simple emissive icosphere — no impostor needed.
            let star_material = std_materials.add(StandardMaterial {
                base_color,
                emissive: LinearRgba::WHITE * 5000.0,
                ..default()
            });

            let body_entity = commands
                .spawn((
                    Transform::from_translation(pos),
                    Visibility::Inherited,
                    CelestialBody {
                        body_id: body.id,
                        is_star,
                        render_radius,
                        radius_m: body.radius_m,
                    },
                    Name::new(body.name.clone()),
                ))
                .id();

            commands.spawn((
                Mesh3d(unit_sphere_star.clone()),
                MeshMaterial3d(star_material.clone()),
                Transform::from_scale(Vec3::splat(render_radius)),
                NotShadowCaster,
                NotShadowReceiver,
                BodyMesh,
                bevy::camera::visibility::RenderLayers::layer(MAP_LAYER),
                ChildOf(body_entity),
            ));
            commands.spawn((
                Mesh3d(unit_sphere_star.clone()),
                MeshMaterial3d(star_material),
                Transform::from_scale(Vec3::splat(ship_render_radius)),
                NotShadowCaster,
                NotShadowReceiver,
                ShipBodyMesh,
                bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER),
                ChildOf(real_body_entity),
            ));
            commands.spawn((
                Mesh3d(icon_mesh.clone()),
                MeshMaterial3d(icon_material),
                Transform::default(),
                Visibility::Hidden,
                BodyIcon,
                HideInShipView,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(body_entity),
            ));

            body_entity
        } else if body.terrain.is_some() {
            // Procedural body. Terrain is generated at runtime by
            // `ProceduralSurface` behind the `SurfaceQuery` seam (no bake), so
            // there is nothing heavy to load off-thread: spawn the shell + a
            // solid-color impostor synchronously, register the runtime surface
            // for the near-surface height source and the propagator, and let
            // `terrain_residency` spawn the ground-LOD terrain on demand.
            //
            // 0b-1 interim: the impostor is a lit solid-color sphere tinted by
            // the body colour. The real distant view (udlod everywhere visible
            // + unified atmosphere) is Slice 6, which deletes this stand-in.
            let tidal_axis = matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z);

            let body_entity = commands
                .spawn((
                    Transform::from_translation(pos),
                    Visibility::Inherited,
                    CelestialBody {
                        body_id: body.id,
                        is_star,
                        render_radius,
                        radius_m: body.radius_m,
                    },
                    Name::new(body.name.clone()),
                ))
                .id();

            // Tidally-locked moons get their local +Z axis as the parent
            // direction (matches the editor); both the body entity and the
            // real-space grid carry the tag.
            if tidal_axis.is_some()
                && let Some(parent_id) = body.parent
            {
                commands
                    .entity(body_entity)
                    .insert(TidallyLocked { parent_id });
                commands
                    .entity(real_body_entity)
                    .insert(TidallyLocked { parent_id });
            }

            // Icon child — visible at far distance, crossfaded by
            // `sync_body_icons`.
            commands.spawn((
                Mesh3d(icon_mesh.clone()),
                MeshMaterial3d(icon_material),
                Transform::default(),
                Visibility::Hidden,
                BodyIcon,
                HideInShipView,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(body_entity),
            ));

            // Interim solid-color lit impostor (map + ship). The ship-layer
            // billboard is tagged `RealSpaceImpostor` so `sync_body_render_lod`
            // hides it whenever the ground-LOD terrain is resident.
            let albedo_linear = Color::srgb(r, g, b).to_linear();
            // Procedural body: bake a low-frequency impostor albedo cube
            // (continents + oceans) from the runtime surface. `albedo.w = 1`
            // tells the shader to sample the cube (by the body-fixed normal, set
            // per frame in `update_solid_planet_params`) instead of the flat
            // colour; the xyz colour stays as the planetshine tint fallback.
            let albedo = Vec4::new(
                albedo_linear.red,
                albedo_linear.green,
                albedo_linear.blue,
                1.0,
            );
            let impostor_surface = ProceduralSurface::new(body.radius_m as f32, body.id as u32);
            let impostor_cube = images.add(bake_impostor_albedo_cube(&impostor_surface, 256));
            // Map-scale atmosphere optics for the rim glow + on-disc aerial
            // perspective. Airless procedural bodies have no
            // `terrestrial_atmosphere` → vacuum block → the shader early-outs.
            // The ship-layer impostor stays vacuum here: the in-context `BodySky`
            // fullscreen pass owns the ship-view atmosphere once terrain is
            // resident, and the far-impostor atmosphere is a Slice-6 concern.
            let map_atmosphere = body
                .terrestrial_atmosphere
                .as_ref()
                .map(|a| AtmosphereBlock::from_terrestrial(a, (1.0 / MAP_SCALE) as f32))
                .unwrap_or_default();
            let map_mat = solid_planet_materials.add(SolidPlanetMaterial {
                params: SolidPlanetParams {
                    radius: render_radius,
                    albedo,
                    scene: SceneLighting::default(),
                    atmosphere: map_atmosphere,
                    ..default()
                },
                albedo_cube: impostor_cube.clone(),
                multi_scatter_lut: multi_scatter_lut.clone(),
            });
            let ship_mat = solid_planet_materials.add(SolidPlanetMaterial {
                params: SolidPlanetParams {
                    radius: ship_render_radius,
                    albedo,
                    scene: SceneLighting::default(),
                    ..default()
                },
                albedo_cube: impostor_cube.clone(),
                multi_scatter_lut: multi_scatter_lut.clone(),
            });
            commands.spawn((
                Mesh3d(billboard_mesh.clone()),
                MeshMaterial3d(map_mat.clone()),
                BodyMesh,
                bevy::camera::visibility::RenderLayers::layer(MAP_LAYER),
                NoFrustumCulling,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(body_entity),
            ));
            commands.spawn((
                Mesh3d(billboard_mesh.clone()),
                MeshMaterial3d(ship_mat.clone()),
                ShipBodyMesh,
                bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER),
                NoFrustumCulling,
                NotShadowCaster,
                NotShadowReceiver,
                RealSpaceImpostor { body_id: body.id },
                ChildOf(real_body_entity),
            ));

            // Atmosphere rim glow on the map disc, for bodies that have one.
            // Premultiplied sibling billboard outside the solid silhouette;
            // `update_solid_planet_params` keeps its params in lockstep.
            let map_halo = if map_atmosphere.atmos_geom.z > 0.0 {
                let halo_mat = solid_planet_halo_materials.add(SolidPlanetHaloMaterial {
                    params: SolidPlanetParams {
                        radius: render_radius,
                        albedo,
                        scene: SceneLighting::default(),
                        atmosphere: map_atmosphere,
                        ..default()
                    },
                });
                commands.spawn((
                    Mesh3d(billboard_mesh.clone()),
                    MeshMaterial3d(halo_mat.clone()),
                    BodyMesh,
                    bevy::camera::visibility::RenderLayers::layer(MAP_LAYER),
                    NoFrustumCulling,
                    NotShadowCaster,
                    NotShadowReceiver,
                    ChildOf(body_entity),
                ));
                Some(halo_mat)
            } else {
                None
            };

            commands.entity(body_entity).insert(SolidPlanetMaterials {
                map: map_mat,
                ship: ship_mat,
                map_halo,
            });

            // Register the runtime surface for (a) the near-surface height
            // source — collider / character controller / HUD altitude, via the
            // GPU-atlas mirror with a CPU fallback — and (b) the propagator's
            // coarse orbital collision. Built from the same body params as the
            // ground tile provider (`spawn_body_terrain`) so all three agree.
            let proc_surface = ProceduralSurface::new(body.radius_m as f32, body.id as u32);
            procedural_install.height_sources.insert_gpu_mirror_source(
                body.id,
                GpuAtlasMirrorHeightSource::new(Arc::new(proc_surface)),
            );
            procedural_install
                .terrain_registry
                .0
                .insert(body.id, proc_surface);
            world_state.planetshine.by_body.insert(
                body.id,
                [albedo_linear.red, albedo_linear.green, albedo_linear.blue],
            );

            body_entity
        } else if let Some(atmos) = &body.atmosphere {
            // Gas / ice giant path. No terrain bake, no placeholder
            // swap: spawn the billboard + GasGiantMaterial directly.
            // Per-frame updates flow through `update_gas_giant_params`
            // exactly like `update_planet_light_dirs` does for baked
            // bodies.
            //
            // Two material instances per body: one baked at MAP_SCALE
            // for the map-layer billboard, one at SHIP_SCALE for the
            // ship-layer billboard. The cloud-deck / haze layers are
            // expressed in render units, so the meters-per-render-unit
            // factor differs between the two.
            // Average the cloud-deck palette as the gas-giant planetshine
            // tint. Palette colours are already linear-RGB.
            let palette = &atmos.cloud_deck.palette;
            let mean_cloud = if palette.is_empty() {
                [0.5, 0.5, 0.5]
            } else {
                let mut sum = [0.0f32; 3];
                for stop in palette {
                    sum[0] += stop.color[0];
                    sum[1] += stop.color[1];
                    sum[2] += stop.color[2];
                }
                let n = palette.len() as f32;
                [sum[0] / n, sum[1] / n, sum[2] / n]
            };
            world_state.planetshine.by_body.insert(body.id, mean_cloud);

            let map_layers =
                GasGiantLayers::from_params(atmos, body.rings.as_ref(), (1.0 / MAP_SCALE) as f32);
            let ship_layers =
                GasGiantLayers::from_params(atmos, body.rings.as_ref(), (1.0 / SHIP_SCALE) as f32);

            let map_gas_material = gas_giant_materials.add(GasGiantMaterial {
                params: GasGiantParams {
                    radius: render_radius,
                    ..default()
                },
                layers: map_layers,
            });
            let ship_gas_material = gas_giant_materials.add(GasGiantMaterial {
                params: GasGiantParams {
                    radius: ship_render_radius,
                    ..default()
                },
                layers: ship_layers,
            });

            let body_entity = commands
                .spawn((
                    Transform::from_translation(pos),
                    Visibility::Inherited,
                    CelestialBody {
                        body_id: body.id,
                        is_star,
                        render_radius,
                        radius_m: body.radius_m,
                    },
                    Name::new(body.name.clone()),
                ))
                .id();

            commands.spawn((
                Mesh3d(billboard_mesh.clone()),
                MeshMaterial3d(map_gas_material.clone()),
                BodyMesh,
                bevy::camera::visibility::RenderLayers::layer(MAP_LAYER),
                // Billboard's local AABB is a flat 2×2 quad; the vertex
                // shader re-orients it each frame. Disable frustum
                // culling so Bevy doesn't hide it at angles where the
                // flat AABB misses the view frustum.
                NoFrustumCulling,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(body_entity),
            ));

            commands.spawn((
                Mesh3d(billboard_mesh.clone()),
                MeshMaterial3d(ship_gas_material.clone()),
                ShipBodyMesh,
                bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER),
                NoFrustumCulling,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(real_body_entity),
            ));

            commands.spawn((
                Mesh3d(icon_mesh.clone()),
                MeshMaterial3d(icon_material),
                Transform::default(),
                Visibility::Hidden,
                BodyIcon,
                HideInShipView,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(body_entity),
            ));

            commands.entity(body_entity).insert(GasGiantMaterials {
                map: map_gas_material,
                ship: ship_gas_material,
            });

            body_entity
        } else {
            // Non-procedural body: solid-color billboard impostor. Same
            // camera-facing-quad architecture as the procedural impostor
            // and gas giant paths, so close approaches don't clip the
            // body against the camera near plane. Renders as a single
            // linear-RGB color (sRGB → linear for pipeline compatibility).
            let albedo_linear = Color::srgb(r, g, b).to_linear();
            let albedo = Vec4::new(
                albedo_linear.red,
                albedo_linear.green,
                albedo_linear.blue,
                0.0,
            );

            // Non-procedural solid bodies (asteroids, airless moons) have no
            // terrestrial atmosphere → vacuum block → no rim, no aerial tint.
            let map_mat = solid_planet_materials.add(SolidPlanetMaterial {
                params: SolidPlanetParams {
                    radius: render_radius,
                    albedo,
                    scene: SceneLighting::default(),
                    ..default()
                },
                albedo_cube: blank_impostor.clone(),
                multi_scatter_lut: multi_scatter_lut.clone(),
            });
            let ship_mat = solid_planet_materials.add(SolidPlanetMaterial {
                params: SolidPlanetParams {
                    radius: ship_render_radius,
                    albedo,
                    scene: SceneLighting::default(),
                    ..default()
                },
                albedo_cube: blank_impostor.clone(),
                multi_scatter_lut: multi_scatter_lut.clone(),
            });

            let body_entity = commands
                .spawn((
                    Transform::from_translation(pos),
                    Visibility::Inherited,
                    CelestialBody {
                        body_id: body.id,
                        is_star,
                        render_radius,
                        radius_m: body.radius_m,
                    },
                    Name::new(body.name.clone()),
                ))
                .id();

            commands.spawn((
                Mesh3d(billboard_mesh.clone()),
                MeshMaterial3d(map_mat.clone()),
                BodyMesh,
                bevy::camera::visibility::RenderLayers::layer(MAP_LAYER),
                NoFrustumCulling,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(body_entity),
            ));
            commands.spawn((
                Mesh3d(billboard_mesh.clone()),
                MeshMaterial3d(ship_mat.clone()),
                ShipBodyMesh,
                bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER),
                NoFrustumCulling,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(real_body_entity),
            ));
            commands.spawn((
                Mesh3d(icon_mesh.clone()),
                MeshMaterial3d(icon_material),
                Transform::default(),
                Visibility::Hidden,
                BodyIcon,
                HideInShipView,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(body_entity),
            ));

            commands.entity(body_entity).insert(SolidPlanetMaterials {
                map: map_mat,
                ship: ship_mat,
                map_halo: None,
            });

            body_entity
        };

        // ── Ring system ─────────────────────────────────────────────
        //
        // Body-level: any body with `rings: Some(_)` gets a ring annulus,
        // gas giant or rocky alike. Two ring children are spawned — a
        // map-layer child baked at MAP_SCALE and a ship-layer child under
        // the real-space body grid at SHIP_SCALE — each with its own
        // `RingMaterial` instance because the per-frame uniforms (planet
        // center, radii) differ between the two views.
        //
        // Ring child rotation is `Rx(-tilt)`. For gas giants, the cloud
        // shader treats `orientation = Rx(+tilt)` as world→body-local,
        // so the body's world-space equatorial plane normal is
        // `Rx(-tilt) * (0,1,0)` — the ring mesh (geometric normal +Y)
        // therefore needs `Rx(-tilt)` to align. If the cloud shader's
        // ring-shadow plane test in `gas_giant.wgsl` is changed, this
        // rotation must move with it.
        //
        // TODO(unimplemented): ring-shadow on rocky-body surfaces.
        // The gas-giant cloud-deck shader projects body points onto
        // the ring plane and darkens accordingly; the terrain
        // impostor shader (`planet_impostor.wgsl`) has no equivalent
        // pass yet. So a rocky body with rings renders the rings
        // themselves correctly (and the rings self-shadow against
        // the body via `ring.wgsl`'s planet-shadow ray-cast), but
        // the body's lit surface won't darken inside the ring
        // annulus. Wiring this requires adding ring uniforms to
        // `PlanetMaterial` and a matching shadow term to the
        // impostor shader.
        if let Some(rings) = &body.rings {
            // The cloud-deck ring-shadow term is only wired into
            // `GasGiantMaterial`, which is selected when a body has
            // an atmosphere and no terrain. Anything else
            // (terrain-baked, plain placeholder, or star) renders the
            // ring annulus correctly but won't darken the body's
            // surface inside it. Discriminator must match the branch
            // selection above — `atmosphere.is_some()` alone is not
            // sufficient because a body with both `terrain` and
            // `atmosphere` would take the terrain branch first.
            let renders_as_gas_giant =
                matches!(&body.terrain, TerrainConfig::None) && body.atmosphere.is_some();
            if !renders_as_gas_giant {
                warn!(
                    "body '{}' has rings but ring-shadow on its surface is not yet implemented \
                     (only gas-giant cloud decks receive a ring shadow today; \
                     see TODO in spawn_bodies / planet_impostor.wgsl)",
                    body.name
                );
            }

            let map_mpru = (1.0 / MAP_SCALE) as f32;
            let ship_mpru = (1.0 / SHIP_SCALE) as f32;
            let map_inner = rings.inner_radius_m / map_mpru;
            let map_outer = rings.outer_radius_m / map_mpru;
            let ship_inner = rings.inner_radius_m / ship_mpru;
            let ship_outer = rings.outer_radius_m / ship_mpru;

            let map_ring_mesh = meshes.add(build_ring_mesh(map_inner, map_outer, 512));
            let ship_ring_mesh = meshes.add(build_ring_mesh(ship_inner, ship_outer, 512));

            let map_ring_material = ring_materials.add(RingMaterial {
                params: RingParams {
                    planet_center_radius: Vec4::new(pos.x, pos.y, pos.z, render_radius),
                    inner_radius: map_inner,
                    outer_radius: map_outer,
                    ..default()
                },
                layers: RingLayers::from_system(rings),
            });
            let ship_ring_material = ring_materials.add(RingMaterial {
                params: RingParams {
                    planet_center_radius: Vec4::new(pos.x, pos.y, pos.z, ship_render_radius),
                    inner_radius: ship_inner,
                    outer_radius: ship_outer,
                    ..default()
                },
                layers: RingLayers::from_system(rings),
            });

            let tilt = body.axial_tilt_rad as f32;
            let tilt_rot = Transform::from_rotation(Quat::from_rotation_x(-tilt));

            commands.spawn((
                Mesh3d(map_ring_mesh),
                MeshMaterial3d(map_ring_material.clone()),
                tilt_rot,
                BodyMesh,
                bevy::camera::visibility::RenderLayers::layer(MAP_LAYER),
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(body_entity),
                MapRingMaterial(map_ring_material),
            ));

            commands.spawn((
                Mesh3d(ship_ring_mesh),
                MeshMaterial3d(ship_ring_material.clone()),
                tilt_rot,
                ShipBodyMesh,
                bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER),
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(real_body_entity),
                ShipRingMaterial(ship_ring_material),
            ));
        }
    }

    // The player ship's `ShipMarker` billboard is spawned alongside the
    // ship root in `ship_view::spawn_player_ship`, so all per-ship entities
    // are created in one place — multi-ship support later means calling
    // that spawn function per blueprint, not threading state through the
    // body-setup path.

    // Directional light simulating sunlight. Direction is updated per-frame
    // by `update_sun_light` to point from the star toward the camera focus body.
    // Using a DirectionalLight with cascaded shadow maps instead of a PointLight
    // because Bevy's point light can't handle solar-system-scale distances.
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            color: Color::WHITE,
            shadow_maps_enabled: true,
            ..default()
        },
        // The ship is the only shadow caster — every body mesh is tagged
        // `NotShadowCaster` / `NotShadowReceiver`. A ~10 m caster doesn't
        // need a 100 km cascade chain; two cascades sized for the ship's
        // local neighbourhood keep the shadow pass cheap. The count is the
        // shared `SHADOW_CASCADE_COUNT` (see its doc for the Bevy bug that
        // requires every directional light to agree on it).
        CascadeShadowConfigBuilder {
            num_cascades: super::SHADOW_CASCADE_COUNT,
            minimum_distance: 0.1,
            maximum_distance: 500.0,
            first_cascade_far_bound: 30.0,
            overlap_proportion: 0.2,
        }
        .build(),
        // The light must share a render layer with its shadow *casters*:
        // `check_dir_light_mesh_visibility` intersects the light's layers
        // with each mesh's before adding it to the cascade lists, and the
        // craft's part visuals are re-stamped onto SHIP_LAYER every frame
        // by `view::propagate_view_render_layers` (the `HideInMapView`
        // subtree stamp). Without SHIP_LAYER here the craft never enters
        // the shadow map and casts nothing on the runway or itself.
        bevy::camera::visibility::RenderLayers::from_layers(&[0, crate::coords::SHIP_LAYER]),
        Transform::default(),
        SunLight,
    ));

    // Secondary directional light for moonlight — driven each frame by
    // `lighting::update_moon_light` from the brightest child moon of the body
    // the craft is on (e.g. Mira over Thalos), so the `StandardMaterial` hull +
    // surface structures catch moonlight at night the way the terrain shader's
    // own moonlight term lights the ground. No shadows (soft fill) and no
    // cascade config; starts dark (illuminance 0) until a lit moon is up. Shares
    // SHIP_LAYER with the craft for the same reason `SunLight` does.
    commands.spawn((
        DirectionalLight {
            illuminance: 0.0,
            color: Color::WHITE,
            shadow_maps_enabled: false,
            ..default()
        },
        // Shadows are off, so this cascade config is never actually sampled —
        // but a bare `DirectionalLight` silently gets Bevy's default 4-cascade
        // config, which would mismatch `SunLight`'s count the moment someone
        // flips `shadow_maps_enabled` on. Pin it to `SHADOW_CASCADE_COUNT` so
        // that future change can't reintroduce the `check_dir_light_mesh_visibility`
        // over-index panic (see the constant's doc).
        CascadeShadowConfigBuilder {
            num_cascades: super::SHADOW_CASCADE_COUNT,
            ..default()
        }
        .build(),
        bevy::camera::visibility::RenderLayers::from_layers(&[0, crate::coords::SHIP_LAYER]),
        Transform::default(),
        MoonLight,
    ));

    // Dim ambient light so shadowed sides of planets aren't pitch black.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 50.0,
        ..default()
    });

    // No async bake installs anymore — procedural bodies generate terrain at
    // runtime. Seed the bake-install step's total to 0 so it completes
    // immediately and the loading screen advances to the terrain-residency gate.
    loading_tracker.set_total(crate::loading::step::BODIES, 0);
}
