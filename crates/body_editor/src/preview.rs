#![allow(clippy::too_many_arguments)]

use super::*;

// Preview spawning

#[cfg(debug_assertions)]
pub(crate) const DEV_CRATER_SCALE: f32 = 0.1;
#[cfg(not(debug_assertions))]
pub(crate) const DEV_CRATER_SCALE: f32 = 1.0;

/// Output directory for the editor's "Full" bake. Same location the game
/// loads from and `bake_dump` writes to — so pressing Full here produces
/// the local game artifact directly.
pub(crate) fn local_bake_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/bakes")
}

pub(crate) fn dispatch_terrain_bake(
    terrain: &TerrainConfig,
    tectonics: Option<&TectonicConfig>,
    radius_m: f64,
    gravity_m_s2: f32,
    tidal_axis: Option<Vec3>,
    axial_tilt_rad: f32,
    body_name: String,
    cubemap_resolution_override: Option<u32>,
) -> Task<Result<PlanetSurface, String>> {
    let radius_m = radius_m as f32;
    let mut terrain = terrain.clone();
    let tectonics = tectonics.cloned();
    if let Some(res) = cubemap_resolution_override {
        match &mut terrain {
            TerrainConfig::Feature(c) => c.cubemap_resolution = Some(res),
            TerrainConfig::Ocean(c) => c.cubemap_resolution = Some(res),
            TerrainConfig::None => {}
        }
    }
    AsyncComputeTaskPool::get().spawn(async move {
        let bake_dir = local_bake_dir();
        let route = terrain.route_label();
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
            cubemap_resolution_override: None,
        };
        // The editor never reads from the bake store so edits and compile
        // changes always show up; only full-res bakes write, producing the
        // local artifact the game loads from.
        let is_full_bake = cubemap_resolution_override.is_none();
        info!("baking {body_name} via {route}");
        // The editor's compile path doesn't wire up a GPU mid-frequency
        // runner yet (would need the Bevy `RenderDevice` plumbed into the
        // async task pool). Skip the stage for now — the editor preview
        // shows continental relief without mid-freq detail. Producing a
        // production-quality local bake still requires `just bake`.
        let mid_freq = None;
        let data =
            match compile_terrain_config(&terrain, tectonics.as_ref(), &context, options, mid_freq)
            {
                Ok(data) => data,
                Err(e) => return Err(format!("terrain compile failed for {body_name}: {e}")),
            };
        if is_full_bake {
            let key = thalos_terrain::cache::terrain_cache_key(
                &terrain,
                tectonics.as_ref(),
                &context,
                options,
            );
            let path = thalos_terrain::cache::cache_path(&bake_dir, &body_name);
            match thalos_terrain::cache::store(&path, key, &data.static_surface) {
                Ok(()) => info!("wrote local bake: {body_name} → {}", path.display()),
                Err(e) => warn!("local bake write failed for {body_name}: {e}"),
            }
        }
        Ok(data)
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_preview(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    std_materials: &mut Assets<StandardMaterial>,
    gas_giant_materials: &mut Assets<GasGiantMaterial>,
    ring_materials: &mut Assets<RingMaterial>,
    billboard: &BillboardMesh,
    planet: &mut EditedPlanet,
    status: &mut TerrainGenStatus,
) {
    let parent = commands
        .spawn((
            Transform::default(),
            Visibility::Inherited,
            PreviewPlanet,
            Name::new("Preview Planet"),
        ))
        .id();

    match &planet.mode {
        BodyMode::Terrain {
            terrain,
            tectonics,
            tidal_axis,
        } => {
            let placeholder_mesh = meshes.add(Sphere::new(RENDER_RADIUS).mesh().ico(4).unwrap());
            let placeholder_mat = std_materials.add(StandardMaterial {
                base_color: Color::srgb(0.4, 0.4, 0.45),
                perceptual_roughness: 0.9,
                metallic: 0.0,
                ..default()
            });

            let mesh_entity = commands
                .spawn((
                    Mesh3d(placeholder_mesh),
                    MeshMaterial3d(placeholder_mat),
                    ChildOf(parent),
                ))
                .id();

            let task = dispatch_terrain_bake(
                terrain,
                tectonics.as_ref(),
                planet.radius_m,
                planet.gravity_m_s2,
                *tidal_axis,
                planet.axial_tilt_rad,
                planet.selected_body.clone(),
                TerrainBakeMode::Preview.resolution_override(),
            );
            planet.last_bake_mode = TerrainBakeMode::Preview;
            status.current_started = Some(Instant::now());
            commands
                .entity(parent)
                .insert(PendingTerrainGen { task, mesh_entity });
        }
        BodyMode::GasGiant { layers } => {
            let scene = scene_lighting_for(planet);
            let tilt = body_orientation(planet);

            let mat_handle = gas_giant_materials.add(GasGiantMaterial {
                params: GasGiantParams {
                    radius: RENDER_RADIUS,
                    rotation_phase: 0.0,
                    elapsed_time: 0.0,
                    orientation: Vec4::new(tilt.x, tilt.y, tilt.z, tilt.w),
                    scene: scene.clone(),
                    ..default()
                },
                layers: *layers.clone(),
            });

            commands.spawn((
                Mesh3d(billboard.0.clone()),
                MeshMaterial3d(mat_handle.clone()),
                ChildOf(parent),
            ));

            commands
                .entity(parent)
                .insert(GasGiantMaterialHandle(mat_handle));
        }
        BodyMode::Star => {
            let star_mesh = meshes.add(Sphere::new(RENDER_RADIUS).mesh().ico(5).unwrap());
            let star_mat = std_materials.add(StandardMaterial {
                base_color: Color::BLACK,
                emissive: LinearRgba::new(1.0, 0.95, 0.8, 1.0) * 5000.0,
                ..default()
            });
            commands.spawn((Mesh3d(star_mesh), MeshMaterial3d(star_mat), ChildOf(parent)));
        }
    }

    // Ring system — body-level, decoupled from `BodyMode`. Any preview
    // body (terrain or gas giant) gets a ring annulus if `planet.rings`
    // is set. The ring shadow uniform on `GasGiantMaterial` is fed
    // separately at material build time; for terrain bodies the ring
    // renders correctly but the body surface doesn't yet darken inside
    // the annulus (see TODO in `spawn_bodies` / `planet_impostor.wgsl`).
    if let Some(rings) = &planet.rings {
        let scene = scene_lighting_for(planet);
        let meters_per_ru = planet.radius_m as f32 / RENDER_RADIUS;
        let inner_ru = rings.inner_radius_m / meters_per_ru;
        let outer_ru = rings.outer_radius_m / meters_per_ru;
        let ring_mesh = meshes.add(build_ring_mesh(inner_ru, outer_ru, 128));

        let ring_mat = ring_materials.add(RingMaterial {
            params: RingParams {
                planet_center_radius: Vec4::new(0.0, 0.0, 0.0, RENDER_RADIUS),
                inner_radius: inner_ru,
                outer_radius: outer_ru,
                scene,
                ..default()
            },
            layers: *rings.layers.clone(),
        });

        let ring_entity = commands
            .spawn((
                Mesh3d(ring_mesh),
                MeshMaterial3d(ring_mat.clone()),
                ChildOf(parent),
                PreviewRing,
            ))
            .id();

        commands
            .entity(ring_entity)
            .insert(RingMaterialHandle(ring_mat));
    }
}

pub(crate) fn spawn_preview_planet(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut gas_giant_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
    mut status: ResMut<TerrainGenStatus>,
    mut planet: ResMut<EditedPlanet>,
) {
    let billboard_mesh = meshes.add(Rectangle::new(2.0, 2.0));
    commands.insert_resource(BillboardMesh(billboard_mesh));

    let billboard = BillboardMesh(meshes.add(Rectangle::new(
        RENDER_RADIUS * 2.0 + 2.0,
        RENDER_RADIUS * 2.0 + 2.0,
    )));

    spawn_preview(
        &mut commands,
        &mut meshes,
        &mut std_materials,
        &mut gas_giant_materials,
        &mut ring_materials,
        &billboard,
        &mut planet,
        &mut status,
    );

    commands.insert_resource(billboard);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_terrain_bake(
    mut commands: Commands,
    mut pending_q: Query<(Entity, &mut PendingTerrainGen), With<PreviewPlanet>>,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut planet_halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut status: ResMut<TerrainGenStatus>,
    mut active_surface: ResMut<ActivePreviewSurface>,
    mut tile_viewer: ResMut<TileViewerState>,
    mut equirect_viewer: ResMut<EquirectViewerState>,
    billboard: Res<BillboardMesh>,
    planet: Res<EditedPlanet>,
    reference_clouds: Res<ReferenceClouds>,
    children_q: Query<&Children>,
    halo_q: Query<Entity, With<PreviewAtmosphereHalo>>,
) {
    for (entity, mut pending) in &mut pending_q {
        let Some(result) = block_on(poll_once(&mut pending.task)) else {
            continue;
        };
        let surface = match result {
            Ok(surface) => surface,
            Err(e) => {
                // Compile failure (e.g. a transient invalid edit). Log and
                // drop the pending bake without touching the entity's
                // existing PlanetMaterial — the previous terrain stays on
                // screen so the user can recover by undoing the edit.
                warn!("{e}");
                commands.entity(entity).remove::<PendingTerrainGen>();
                status.current_started = None;
                continue;
            }
        };
        let surface = Arc::new(surface);
        let body = &surface.static_surface;
        let dynamic_state = DynamicSurfaceState::for_layers(&surface.dynamic_layers);

        let detail =
            PlanetDetailParams::from_body(&body.detail_params, body.cubemap_bake_threshold_m);
        let height_range = body.height_range;
        let textures =
            bake_from_planet_surface(&surface, &dynamic_state, &mut images, &mut storage_buffers);
        let (_, _, wrap) = lighting_for(&planet);
        let scene = scene_lighting_for(&planet);

        let coastline = PlanetCoastlineParams::from_static_surface(body);
        let water = PlanetWaterParams::from_static_surface(body);
        let atmosphere = active_atmosphere(&planet);
        let cloud_cover = cloud_cover_for(&planet, &reference_clouds, &mut images);
        let q = body_orientation(&planet);

        let planet_material = PlanetMaterial {
            params: PlanetParams {
                radius: RENDER_RADIUS,
                height_range,
                terminator_wrap: wrap,
                fullbright: if planet.full_bright { 1.0 } else { 0.0 },
                orientation: Vec4::new(q.x, q.y, q.z, q.w),
                scene,
                sea_level_m: body.sea_level_m.unwrap_or(-1.0e9),
                water_color_depth: water.color_depth,
                coastline_warp_amp_radians: coastline.warp_amp_radians,
                coastline_jitter_amp_m: coastline.jitter_amp_m,
                coastline_seed: coastline.seed,
                ..default()
            },
            albedo: textures.albedo,
            height: textures.height,
            detail,
            roughness: textures.roughness,
            craters: textures.craters,
            cell_index: textures.cell_index,
            feature_ids: textures.feature_ids,
            radial_features: textures.radial_features,
            atmosphere,
            cloud_cover,
            ice_caps: textures.ice_caps,
            active_dunes: textures.active_dunes,
            active_dune_height: textures.active_dune_height,
            active_dune_albedo: textures.active_dune_albedo,
        };
        let halo_handle = planet_halo_materials.add(PlanetHaloMaterial::from(&planet_material));
        let mat_handle = planet_materials.add(planet_material);

        let mesh_entity = pending.mesh_entity;
        commands
            .entity(mesh_entity)
            .insert((
                Mesh3d(billboard.0.clone()),
                MeshMaterial3d(mat_handle.clone()),
                NoFrustumCulling,
            ))
            .remove::<MeshMaterial3d<StandardMaterial>>();

        if planet.atmosphere.is_some() {
            let existing_halo = children_q
                .get(entity)
                .ok()
                .and_then(|children| children.iter().find(|child| halo_q.get(*child).is_ok()));
            if let Some(halo_entity) = existing_halo {
                commands.entity(halo_entity).insert((
                    Mesh3d(billboard.0.clone()),
                    MeshMaterial3d(halo_handle.clone()),
                    NoFrustumCulling,
                ));
            } else {
                commands.spawn((
                    Mesh3d(billboard.0.clone()),
                    MeshMaterial3d(halo_handle.clone()),
                    ChildOf(entity),
                    PreviewAtmosphereHalo,
                    NoFrustumCulling,
                    NotShadowCaster,
                    NotShadowReceiver,
                    Name::new(format!("{} Atmosphere Halo", planet.selected_body)),
                ));
            }
        }

        active_surface.body_name = planet.selected_body.clone();
        active_surface.surface = Some(surface.clone());
        active_surface.dynamic_state = Some(dynamic_state.clone());
        tile_viewer.dirty = true;
        equirect_viewer.dirty = true;

        let mut entity_commands = commands.entity(entity);
        entity_commands
            .insert(PlanetMaterialHandle(mat_handle))
            .insert(PlanetHaloMaterialHandle(halo_handle))
            .insert(PreviewDynamicSurface {
                layers: surface.dynamic_layers.clone(),
                state: dynamic_state,
            })
            .remove::<PendingTerrainGen>();
        if planet
            .atmosphere
            .as_ref()
            .is_some_and(|atmos| atmos.block.cloud_albedo_coverage.w > 0.0)
        {
            commands
                .entity(entity)
                .insert(PreviewCloudBandState::default());
        } else {
            commands.entity(entity).remove::<PreviewCloudBandState>();
        }

        if let Some(started) = status.current_started.take() {
            status.last_duration = Some(started.elapsed());
        }
    }
}
