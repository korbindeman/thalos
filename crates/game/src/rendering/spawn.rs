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
//! [`crate::loading::LoadingProgress`] counter that drives the startup
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
use thalos_udlod::prelude::PreciseRotation;
use thalos_physics_canonical::canonical::Epoch;
use thalos_physics_canonical::types::BodyKind;
use thalos_planet_rendering::{
    AtmosphereBlock, GasGiantLayers, GasGiantMaterial, GasGiantParams, ReferenceClouds, RingLayers,
    RingMaterial, RingParams, SceneLighting, SolidPlanetMaterial, SolidPlanetParams,
    build_ring_mesh, cloud_cover_image_for_body, prepare_planet_bake,
};
use thalos_terrain::{
    DynamicSurfaceState, PlanetSurface, TerrainCompileContext, TerrainCompileOptions,
    TerrainConfig, cache, compile_dynamic_surface_layers, compile_tectonics_from_config,
};
use thalos_terrain_render::{BodySkyExtra, BodySkyMaterial};

use super::generation::{
    PendingPlanetInstall, PendingPlanetInstalls, PlanetBakeOutput, WorldStateAssets,
};
use super::ground_terrain::BodySky;
use super::real_space::{RealSpaceRoot, real_space_grid};
use super::scene_depth::SceneDepthImage;
use super::types::{
    BodyIcon, BodyMesh, CelestialBody, GasGiantMaterials, MapRingMaterial, RealSpaceBody,
    SharedPlanetMeshes, ShipBodyMesh, ShipRingMaterial, SimulationState, SolidPlanetMaterials,
    SunLight, TidallyLocked,
};
use crate::coords::{MAP_LAYER, MAP_SCALE, SHIP_LAYER, SHIP_SCALE};
use crate::loading::LoadingProgress;
use crate::view::HideInShipView;
use bevy::tasks::AsyncComputeTaskPool;

/// Crater-count scale factor used in the cache-key computation. Must
/// match whatever `bake_dump` (and the editor's Full button) used when
/// the bake was produced; mismatch fails load with `HashMismatch`.
///
/// Local bakes always use 1.0 (full crater authoring), so the game
/// uses 1.0 unconditionally. The old `DEV_CRATER_SCALE` knob existed
/// to speed up the game's own bake in dev — obsolete now that the game
/// never compiles.
const BAKE_CRATER_COUNT_SCALE: f32 = 1.0;

/// Directory holding local bake artifacts. Mirror of
/// `crates/bake_dump/src/main.rs::local_bake_dir` — both must resolve to
/// the same workspace-relative path so producer and consumer agree.
fn local_bake_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/bakes")
}

pub(super) fn spawn_bodies(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut gas_giant_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
    mut solid_planet_materials: ResMut<Assets<SolidPlanetMaterial>>,
    mut sky_materials: ResMut<Assets<BodySkyMaterial>>,
    sim: Res<SimulationState>,
    real_root: Res<RealSpaceRoot>,
    scene_depth: Res<SceneDepthImage>,
    reference_clouds: Res<ReferenceClouds>,
    mut pending_installs: ResMut<PendingPlanetInstalls>,
    mut loading_progress: ResMut<LoadingProgress>,
    mut world_state: WorldStateAssets,
) {
    let bodies = &sim.system.bodies;
    let initial_states = sim.ephemeris.states(Epoch::ZERO);

    // Shared meshes.
    let icon_mesh = meshes.add(Circle::new(1.0));
    // Unit rectangle (corners at ±1) shared across all planet billboards.
    // The vertex shader scales it by params.radius each frame.
    let billboard_mesh = meshes.add(Rectangle::new(2.0, 2.0));
    // Star icosphere — emissive star meshes still use a real sphere
    // (no impostor). Procedural bodies and gas giants render as
    // camera-facing quads (`billboard_mesh`); solid-impostor bodies
    // likewise. No body needs an icosphere any more.
    let unit_sphere_star = meshes.add(Sphere::new(1.0).mesh().ico(5).unwrap());
    commands.insert_resource(SharedPlanetMeshes {
        billboard: billboard_mesh.clone(),
    });

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
        if ship_atmosphere.atmos_geom.z > 0.0 {
            let (sky_cloud_cover, _) =
                cloud_cover_image_for_body(&body.name, &reference_clouds, &mut images);

            let sky_material = BodySkyMaterial {
                atmosphere: ship_atmosphere,
                atmosphere_extra: BodySkyExtra::default(),
                scene_depth: scene_depth.handle.clone(),
                cloud_cover: sky_cloud_cover,
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
            // Procedural body. The pre-baked surface (`target/bakes/<name>.bin`)
            // can be hundreds of MB and several seconds to decompress; loading
            // it on the main thread froze startup. Spawn the always-present
            // shell entities here (body root, optional tidal-lock tags, icon)
            // and dispatch the bake load + `prepare_planet_bake` to
            // `AsyncComputeTaskPool`. The polling system
            // (`poll_planet_install_tasks`) finishes the install when each
            // task resolves: GPU upload, impostor billboards, halo, ground
            // terrain, water sphere. The loading screen counts these tasks.
            let body_name = body.name.clone();
            let body_radius_m = body.radius_m as f32;
            let gravity_m_s2 = (body.gm / (body.radius_m * body.radius_m)) as f32;
            // Tidally-locked moons get their local +Z axis as the parent
            // direction, matching the editor.
            let tidal_axis = matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z);
            let axial_tilt_rad = body.axial_tilt_rad as f32;

            let context = TerrainCompileContext {
                body_name: body_name.clone(),
                radius_m: body_radius_m,
                gravity_m_s2,
                rotation_hours: None,
                obliquity_deg: Some(axial_tilt_rad.to_degrees()),
                tidal_axis,
                axial_tilt_rad,
            };
            let options = TerrainCompileOptions {
                crater_count_scale: BAKE_CRATER_COUNT_SCALE,
                cubemap_resolution_override: None,
            };
            let key =
                cache::terrain_cache_key(&body.terrain, body.tectonics.as_ref(), &context, options);
            let bake_dir = local_bake_dir();
            let path = cache::cache_path(&bake_dir, &body_name);

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
                    Name::new(body_name.clone()),
                ))
                .id();

            // Moons with a tidal axis and a parent body render tidally
            // locked: the map/ship impostor material uploads world→body,
            // while the real-space body grid (and therefore ground terrain)
            // uses the inverse body→world rotation.
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

            // Icon child — visible at far distance, crossfaded with the
            // impostor by `sync_body_icons` once it spawns.
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

            // Off-thread half: disk I/O + bincode decode + dynamic-layer /
            // tectonics compile + CPU bake prep. Owns clones of the body's
            // terrain/tectonics config so the task outlives the borrow on
            // `sim.system.bodies`. The cache load and the two pure-CPU
            // compiles only share `context` immutably and have no data
            // dependency, so they run concurrently via `rayon::join`.
            let terrain_cfg = body.terrain.clone();
            let tectonics_cfg = body.tectonics.clone();
            let task = AsyncComputeTaskPool::get().spawn(async move {
                let body_name_for_panics = body_name.clone();
                let load_path = path.clone();
                let (static_surface, (dynamic_layers, tectonics_built)) = rayon::join(
                    || match cache::load(&load_path, key) {
                        Ok(s) => s,
                        Err(cache::LoadError::Missing { path }) => panic!(
                            "missing bake for body '{name}' at {path} (key {key:016x}). \
                             Run `just bake {name}` (or `just bake all`) to produce it.",
                            name = body_name_for_panics,
                            path = path.display(),
                        ),
                        Err(cache::LoadError::HashMismatch {
                            path,
                            expected,
                            found,
                        }) => panic!(
                            "stale bake for body '{name}' at {path}: stored key {found:016x}, \
                             expected {expected:016x}. The body config or `thalos_terrain` \
                             source has changed since the bake was produced. \
                             Run `just bake {name}` to regenerate.",
                            name = body_name_for_panics,
                            path = path.display(),
                        ),
                        Err(cache::LoadError::Decode { path, message }) => panic!(
                            "corrupt bake for body '{name}' at {path}: {message}. \
                             Delete and run `just bake {name}` to regenerate.",
                            name = body_name_for_panics,
                            path = path.display(),
                        ),
                    },
                    || {
                        rayon::join(
                            || {
                                compile_dynamic_surface_layers(&terrain_cfg, &context)
                                    .unwrap_or_else(|e| {
                                        panic!(
                                            "dynamic layer compile failed for {}: {e}",
                                            body_name
                                        )
                                    })
                            },
                            || compile_tectonics_from_config(tectonics_cfg.as_ref(), &context),
                        )
                    },
                );
                let surface = PlanetSurface {
                    static_surface,
                    dynamic_layers,
                    tectonics: tectonics_built,
                };
                let dynamic_state = DynamicSurfaceState::for_layers(&surface.dynamic_layers);
                let prepared = prepare_planet_bake(&surface, &dynamic_state);
                PlanetBakeOutput {
                    surface,
                    dynamic_state,
                    prepared,
                }
            });

            loading_progress.total += 1;
            pending_installs.0.push(PendingPlanetInstall {
                body_id: body.id,
                body_name: body.name.clone(),
                body_entity,
                real_body_entity,
                render_radius,
                task,
            });

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

            let map_mat = solid_planet_materials.add(SolidPlanetMaterial {
                params: SolidPlanetParams {
                    radius: render_radius,
                    albedo,
                    scene: SceneLighting::default(),
                },
            });
            let ship_mat = solid_planet_materials.add(SolidPlanetMaterial {
                params: SolidPlanetParams {
                    radius: ship_render_radius,
                    albedo,
                    scene: SceneLighting::default(),
                },
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
            shadows_enabled: true,
            shadow_depth_bias: 2.0,
            shadow_normal_bias: 2.0,
            ..default()
        },
        // The ship is the only shadow caster — every body mesh is tagged
        // `NotShadowCaster` / `NotShadowReceiver`. A ~10 m caster doesn't
        // need a 100 km cascade chain; two cascades sized for the ship's
        // local neighbourhood keep the shadow pass cheap.
        CascadeShadowConfigBuilder {
            num_cascades: 2,
            minimum_distance: 0.1,
            maximum_distance: 500.0,
            first_cascade_far_bound: 30.0,
            overlap_proportion: 0.2,
        }
        .build(),
        Transform::default(),
        SunLight,
    ));

    // Dim ambient light so shadowed sides of planets aren't pitch black.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 50.0,
        ..default()
    });

    // Tell the loading screen the totals are final. Without this the
    // `advance_to_running` system would fire on frame 0 (when
    // `completed == total == 0` is trivially true) and the state would
    // tick over before any bake had been dispatched.
    loading_progress.seeded = true;
}
