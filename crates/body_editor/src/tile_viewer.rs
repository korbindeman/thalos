#![allow(clippy::too_many_arguments)]

use super::*;

// Tile viewer

pub(crate) fn tile_viewer_center_dir(state: &TileViewerState) -> DVec3 {
    let lat = state.center_lat_deg.to_radians() as f64;
    let lon = state.center_lon_deg.to_radians() as f64;
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    DVec3::new(cos_lat * sin_lon, sin_lat, cos_lat * cos_lon).normalize()
}

pub(crate) fn sync_tile_viewer_preview_visibility(
    state: Res<TileViewerState>,
    mut preview_q: Query<&mut Visibility, With<PreviewPlanet>>,
) {
    let target = if state.enabled {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut visibility in &mut preview_q {
        if *visibility != target {
            *visibility = target;
        }
    }
}

pub(crate) fn tile_viewer_terrain_config(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
) -> UdlodTerrainConfig {
    let height_range = rendered_height_range(surface, dynamic_state);
    UdlodTerrainConfig {
        lod_count: TILE_VIEWER_LOD_COUNT,
        model: thalos_body_render::udlod::math::TerrainModel::sphere(
            DVec3::ZERO,
            surface.static_surface.radius_m as f64,
            -height_range,
            height_range,
        ),
        atlas_size: TILE_VIEWER_ATLAS_SIZE,
        ..Default::default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: TILE_VIEWER_TEXTURE_SIZE,
        border_size: TILE_VIEWER_TILE_BORDER_SIZE,
        mip_level_count: TILE_VIEWER_MIP_LEVELS,
        format: AttachmentFormat::Rg16,
    })
    .add_attachment(AttachmentConfig {
        name: "albedo".to_string(),
        texture_size: TILE_VIEWER_TEXTURE_SIZE,
        border_size: TILE_VIEWER_TILE_BORDER_SIZE,
        mip_level_count: TILE_VIEWER_MIP_LEVELS,
        format: AttachmentFormat::Rgba8,
    })
    .add_attachment(AttachmentConfig {
        name: "roughness".to_string(),
        texture_size: TILE_VIEWER_TEXTURE_SIZE,
        border_size: TILE_VIEWER_TILE_BORDER_SIZE,
        mip_level_count: TILE_VIEWER_MIP_LEVELS,
        format: AttachmentFormat::R16,
    })
    .add_attachment(AttachmentConfig {
        name: "material".to_string(),
        texture_size: TILE_VIEWER_TEXTURE_SIZE,
        border_size: TILE_VIEWER_TILE_BORDER_SIZE,
        mip_level_count: TILE_VIEWER_MIP_LEVELS,
        format: AttachmentFormat::Rgba8,
    })
}

pub(crate) fn tile_viewer_view_config(state: &TileViewerState, radius_m: f64) -> TerrainViewConfig {
    TerrainViewConfig {
        tree_size: state.tile_count.clamp(4, 32),
        grid_size: state.verts_per_tile.clamp(4, 128),
        precision_threshold_distance: 10_000.0 / radius_m.max(1.0),
        ..TerrainViewConfig::default()
    }
}

pub(crate) fn tile_viewer_stats(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    state: &TileViewerState,
) -> TileViewerStats {
    let center_dir = tile_viewer_center_dir(state);
    let basis = thalos_body_render::TerrainPatchBasis::from_normal(center_dir);
    let half_extent_m = state.tile_count.max(1) as f64 * state.tile_size_m.max(1.0) as f64 * 0.5;
    let samples_per_axis = 17;
    let lod_m = (half_extent_m * 2.0 / (samples_per_axis - 1) as f64).max(1.0) as f32;
    let center = center_dir * surface.static_surface.radius_m as f64;
    let mut min_height_m = f32::INFINITY;
    let mut max_height_m = f32::NEG_INFINITY;
    for z in 0..samples_per_axis {
        let local_z =
            -half_extent_m + z as f64 * (half_extent_m * 2.0 / (samples_per_axis - 1) as f64);
        for x in 0..samples_per_axis {
            let local_x =
                -half_extent_m + x as f64 * (half_extent_m * 2.0 / (samples_per_axis - 1) as f64);
            let dir = (center + basis.tangent_x * local_x + basis.tangent_z * local_z).normalize();
            let h = surface_sample(surface, dynamic_state, dir, lod_m).height_m;
            min_height_m = min_height_m.min(h);
            max_height_m = max_height_m.max(h);
        }
    }
    TileViewerStats {
        min_height_m,
        max_height_m,
        relief_m: max_height_m - min_height_m,
    }
}

pub(crate) fn update_tile_viewer_terrain(
    mut commands: Commands,
    mut materials: ResMut<Assets<BodyTerrainMaterial>>,
    mut tile_trees: ResMut<TerrainViewComponents<TileTree>>,
    active: Res<ActivePreviewSurface>,
    root: Res<EditorBigSpaceRoot>,
    planet: Res<EditedPlanet>,
    mut state: ResMut<TileViewerState>,
    camera_q: Query<Entity, With<EditorCamera>>,
    terrain_q: Query<Entity, With<TileViewerTerrain>>,
) {
    if !state.enabled {
        if let Ok(camera_entity) = camera_q.single() {
            for entity in &terrain_q {
                tile_trees.remove(&(entity, camera_entity));
                commands.entity(entity).despawn();
            }
        } else {
            for entity in &terrain_q {
                commands.entity(entity).despawn();
            }
        }
        return;
    }
    if !state.dirty {
        return;
    }

    let Ok(camera_entity) = camera_q.single() else {
        return;
    };
    for entity in &terrain_q {
        tile_trees.remove(&(entity, camera_entity));
        commands.entity(entity).despawn();
    }

    let (Some(surface), Some(dynamic_state)) = (&active.surface, &active.dynamic_state) else {
        state.stats = None;
        state.dirty = false;
        return;
    };
    let config = tile_viewer_terrain_config(surface, dynamic_state);
    let height_range = rendered_height_range(surface, dynamic_state);
    let provider: Box<dyn TileProvider> = Box::new(PipelineTileProvider::new(
        active.body_name.clone(),
        surface.clone(),
        dynamic_state.clone(),
        height_range,
    ));
    let mut tile_atlas = TileAtlas::with_provider(&config, provider);
    for side in 0..6 {
        tile_atlas.pin_tile(TileCoordinate::new(side, 0, 0, 0));
    }

    let view_config = tile_viewer_view_config(&state, surface.static_surface.radius_m as f64);
    let tile_tree = TileTree::new(&tile_atlas, &view_config);
    let frame = ReferenceFrame::default();
    let mut bundle = TerrainBundle::new(tile_atlas, &frame);
    bundle.visibility = Visibility::Visible;

    let material = BodyTerrainMaterial {
        atmosphere: if planet.atmosphere_enabled && !planet.full_bright {
            active_atmosphere(&planet)
        } else {
            AtmosphereBlock::default()
        },
        scene: scene_lighting_for(&planet),
        extras: BodyTerrainExtras {
            craft_shadow: BodyTerrainShadow::default(),
            debug: Default::default(),
            inspection: Vec4::new(if planet.full_bright { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0),
        },
    };
    let terrain = {
        let mut grid = commands.grid(root.0, ReferenceFrame::default());
        grid.spawn_spatial((
            bundle,
            MeshMaterial3d(materials.add(material)),
            NotShadowCaster,
            TileViewerTerrain,
            Name::new(format!("Tile Viewer Terrain — {}", active.body_name)),
        ))
        .id()
    };
    tile_trees.insert((terrain, camera_entity), tile_tree);

    state.stats = Some(tile_viewer_stats(surface, dynamic_state, &state));
    state.dirty = false;
    let half_extent_m = state.tile_count.max(1) as f32 * state.tile_size_m.max(1.0) * 0.5;
    state.orbit_distance = state.orbit_distance.max(half_extent_m * 2.5).max(100.0);
    if state.free_position.length_squared() < 1e-4 || state.free_position.y < 1.0 {
        state.free_position = Vec3::new(0.0, half_extent_m.max(100.0), half_extent_m * 2.0 + 100.0);
        state.free_speed_units_s = (state.tile_size_m * 0.5).clamp(10.0, 10_000.0);
    }
}

pub(crate) fn update_tile_viewer_materials(
    planet: Res<EditedPlanet>,
    active: Res<ActivePreviewSurface>,
    terrain_q: Query<&MeshMaterial3d<BodyTerrainMaterial>, With<TileViewerTerrain>>,
    mut materials: ResMut<Assets<BodyTerrainMaterial>>,
) {
    if terrain_q.is_empty() {
        return;
    }
    let scene = scene_lighting_for(&planet);
    let atmosphere = if planet.atmosphere_enabled && !planet.full_bright {
        active_atmosphere(&planet)
    } else {
        AtmosphereBlock::default()
    };
    let force_fullbright = planet.full_bright;
    for handle in &terrain_q {
        let Some(mat) = materials.get_mut(&handle.0) else {
            continue;
        };
        mat.scene = scene.clone();
        mat.atmosphere = atmosphere;
        mat.extras.inspection.x = if force_fullbright { 1.0 } else { 0.0 };
    }
    let _ = active;
}
