//! Startup system that spawns one entity tree per body in the solar
//! system: a CelestialBody root with map-layer mesh, ship-layer mesh,
//! flat icon, and (where applicable) ring children. Procedural bodies
//! are seeded with a placeholder sphere; `finalize_planet_generation`
//! later swaps in the impostor billboard once the background terrain
//! task finishes.

use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::cascade::CascadeShadowConfigBuilder;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::tasks::AsyncComputeTaskPool;
use thalos_physics::types::BodyKind;
use thalos_planet_rendering::{
    GasGiantLayers, GasGiantMaterial, GasGiantParams, RingLayers, RingMaterial, RingParams,
    SceneLighting, SolidPlanetMaterial, SolidPlanetParams, build_ring_mesh,
};
use thalos_terrain_gen::{
    TerrainCompileContext, TerrainCompileOptions, TerrainConfig, compile_terrain_config,
};

use super::types::{
    BodyIcon, BodyMesh, CelestialBody, GasGiantMaterials, MapRingMaterial, PendingPlanetGeneration,
    PlanetshineTints, SharedPlanetMeshes, ShipBodyMesh, ShipRingMaterial, SimulationState,
    SolidPlanetMaterials, SunLight, TidallyLocked,
};
use crate::coords::{MAP_LAYER, MAP_SCALE, SHIP_LAYER, SHIP_SCALE};
use crate::view::HideInShipView;

/// Dev-mode crater-count scale factor. Cratering + space_weather together
/// dominate the terrain bake and both scale linearly with crater count, so
/// cutting the authored count by 10× in dev brings bakes from minutes to
/// ~20 s. Release builds keep the full authored counts.
#[cfg(debug_assertions)]
const DEV_CRATER_SCALE: f32 = 0.1;
#[cfg(not(debug_assertions))]
const DEV_CRATER_SCALE: f32 = 1.0;

fn terrain_cache_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/terrain_cache")
}

pub(super) fn spawn_bodies(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut gas_giant_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
    mut solid_planet_materials: ResMut<Assets<SolidPlanetMaterial>>,
    sim: Res<SimulationState>,
    mut planetshine: ResMut<PlanetshineTints>,
) {
    let bodies = &sim.system.bodies;
    let initial_states = sim.ephemeris.query(0.0);

    // Shared meshes.
    let icon_mesh = meshes.add(Circle::new(1.0));
    // Unit rectangle (corners at ±1) shared across all planet billboards.
    // The vertex shader scales it by params.radius each frame.
    let billboard_mesh = meshes.add(Rectangle::new(2.0, 2.0));
    // Unit icosphere — each `BodyMesh` / `ShipBodyMesh` child uses
    // `Transform::scale` to size it for its layer.
    let unit_sphere_star = meshes.add(Sphere::new(1.0).mesh().ico(5).unwrap());
    let unit_sphere_body = meshes.add(Sphere::new(1.0).mesh().ico(4).unwrap());
    commands.insert_resource(SharedPlanetMeshes {
        billboard: billboard_mesh.clone(),
    });

    for body in bodies {
        let state = &initial_states[body.id];
        // Both the parent transform and `body.render_radius` are anchored
        // at MAP_SCALE: the body parent acts as the canonical map-side
        // anchor, and ShipBodyMesh siblings carry a compensating local
        // translation in `update_ship_body_meshes` so they sit at
        // `phys * SHIP_SCALE` in world space.
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
                // Local transform updated each frame by
                // `update_ship_body_meshes` to compensate for the
                // parent's MAP_SCALE translation.
                Transform::from_scale(Vec3::splat(ship_render_radius)),
                NotShadowCaster,
                NotShadowReceiver,
                ShipBodyMesh,
                bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER),
                ChildOf(body_entity),
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
            // Procedural body: dispatch the terrain_gen pipeline to a background
            // task so startup isn't blocked. Meanwhile show a plain placeholder
            // sphere; `finalize_planet_generation` swaps in the impostor
            // billboard with a baked `PlanetMaterial` once the task completes.
            let terrain = body.terrain.clone();
            let radius_m = body.radius_m as f32;
            let gravity_m_s2 = (body.gm / (body.radius_m * body.radius_m)) as f32;
            // Tidally-locked moons get their local +Z axis as the parent
            // direction, matching the editor.
            let tidal_axis = matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z);
            let axial_tilt_rad = body.axial_tilt_rad as f32;
            let body_name = body.name.clone();

            let task = AsyncComputeTaskPool::get().spawn(async move {
                let cache_dir = terrain_cache_dir();
                let context = TerrainCompileContext {
                    body_name: body_name.clone(),
                    radius_m,
                    gravity_m_s2,
                    rotation_hours: None,
                    obliquity_deg: Some(axial_tilt_rad.to_degrees()),
                    tidal_axis,
                    axial_tilt_rad,
                };
                let options = TerrainCompileOptions {
                    crater_count_scale: DEV_CRATER_SCALE,
                };
                let key = thalos_terrain_gen::cache::terrain_cache_key(&terrain, &context, options);
                let path = thalos_terrain_gen::cache::cache_path(&cache_dir, &body_name, key);
                if let Some(data) = thalos_terrain_gen::cache::load(&path, key) {
                    info!("terrain cache hit: {body_name}");
                    return data;
                }
                info!("terrain cache miss, baking: {body_name}");
                let data = compile_terrain_config(&terrain, &context, options)
                    .unwrap_or_else(|e| panic!("terrain compile failed for {body_name}: {e}"));
                match thalos_terrain_gen::cache::store(&path, key, &data) {
                    Ok(()) => info!("terrain cache wrote: {body_name}"),
                    Err(e) => warn!("terrain cache write failed for {body_name}: {e}"),
                }
                data
            });

            // Placeholder: same plain-sphere look as the non-procedural branch
            // so the body is visible immediately at roughly the right size and
            // colour while the terrain pipeline runs in the background.
            let placeholder_mat = std_materials.add(StandardMaterial {
                base_color: Color::srgb(r, g, b),
                perceptual_roughness: 0.9,
                metallic: 0.0,
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

            // Moons with a tidal axis and a parent body are rendered tidally
            // locked: `update_planet_orientations` rewrites the material's
            // orientation quaternion each frame so the baked near-side keeps
            // facing the parent.
            if tidal_axis.is_some()
                && let Some(parent_id) = body.parent
            {
                commands
                    .entity(body_entity)
                    .insert(TidallyLocked { parent_id });
            }

            let mesh_entity = commands
                .spawn((
                    Mesh3d(unit_sphere_body.clone()),
                    MeshMaterial3d(placeholder_mat.clone()),
                    Transform::from_scale(Vec3::splat(render_radius)),
                    BodyMesh,
                    bevy::camera::visibility::RenderLayers::layer(MAP_LAYER),
                    ChildOf(body_entity),
                ))
                .id();

            let ship_mesh_entity = commands
                .spawn((
                    Mesh3d(unit_sphere_body.clone()),
                    MeshMaterial3d(placeholder_mat),
                    Transform::from_scale(Vec3::splat(ship_render_radius)),
                    ShipBodyMesh,
                    bevy::camera::visibility::RenderLayers::layer(SHIP_LAYER),
                    ChildOf(body_entity),
                ))
                .id();

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

            commands
                .entity(body_entity)
                .insert(PendingPlanetGeneration {
                    task,
                    body_id: body.id,
                    render_radius,
                    mesh_entity,
                    ship_mesh_entity,
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
            planetshine.by_body.insert(body.id, mean_cloud);

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
                ChildOf(body_entity),
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
                ChildOf(body_entity),
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
        // map-layer child baked at MAP_SCALE and a ship-layer child at
        // SHIP_SCALE — each with its own `RingMaterial` instance because
        // the per-frame uniforms (planet center, radii) differ between
        // the two views. The ship-layer child carries [`ShipBodyMesh`]
        // so `update_ship_body_meshes` translates it correctly each
        // frame.
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
                ChildOf(body_entity),
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
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            minimum_distance: 0.1,
            maximum_distance: 100_000.0,
            first_cascade_far_bound: 10.0,
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
}

