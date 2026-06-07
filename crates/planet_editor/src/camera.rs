#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) const CAMERA_VFOV: f32 = std::f32::consts::FRAC_PI_4;
pub(crate) const PLANET_VIEW_FRACTION: f32 = 0.40;
pub(crate) const SURFACE_MARGIN: f32 = 1.35;

#[derive(Component)]
pub(crate) struct EditorCamera;

#[derive(Resource)]
pub(crate) struct OrbitCamera {
    azimuth: f32,
    elevation: f32,
    distance: f32,
    target_distance: f32,
    min_distance: f32,
    max_distance: f32,
    planet_render_radius: f32,
}

impl OrbitCamera {
    pub(crate) fn from_render_radius(r: f32) -> Self {
        let min = r * SURFACE_MARGIN;
        let max = r / (0.5 * PLANET_VIEW_FRACTION * CAMERA_VFOV).sin();
        let initial = 5.0_f32.clamp(min, max);
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            distance: initial,
            target_distance: initial,
            min_distance: min,
            max_distance: max,
            planet_render_radius: r,
        }
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::from_render_radius(RENDER_RADIUS)
    }
}

pub(crate) fn spawn_camera(mut commands: Commands) {
    let mut root_entity = None;
    commands.spawn_big_space(ReferenceFrame::default(), |root| {
        root_entity = Some(root.id());
        root.spawn_spatial((
            Camera3d::default(),
            thalos_planet_rendering::space_camera_post_stack(),
            Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            FloatingOrigin,
            EditorCamera,
        ));
    });
    commands.insert_resource(EditorBigSpaceRoot(
        root_entity.expect("spawn_big_space closure sets root entity"),
    ));
}

pub(crate) fn camera_input(
    input: Res<PlanetEditorInputIntent>,
    mut orbit: ResMut<OrbitCamera>,
    mut tile_viewer: ResMut<TileViewerState>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut egui_ctx: bevy_egui::EguiContexts,
    planet: Res<EditedPlanet>,
) {
    if egui_ctx
        .ctx_mut()
        .is_ok_and(|ctx| ctx.wants_pointer_input())
    {
        return;
    }

    const ROTATE_SENSITIVITY: f32 = 0.005;
    const ZOOM_SENSITIVITY: f32 = 0.04;

    if tile_viewer.enabled {
        match tile_viewer.camera_mode {
            TileViewerCameraMode::Orbit => {
                if input.primary_pressed {
                    let delta = input.camera_motion;
                    tile_viewer.orbit_azimuth += delta.x * ROTATE_SENSITIVITY;
                    tile_viewer.orbit_elevation = (tile_viewer.orbit_elevation
                        - delta.y * ROTATE_SENSITIVITY)
                        .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
                }
                if input.camera_wheel.y != 0.0 {
                    let log_d = tile_viewer.orbit_distance.max(0.1).ln()
                        - input.camera_wheel.y * ZOOM_SENSITIVITY;
                    tile_viewer.orbit_distance = log_d.exp().clamp(0.05, 500.0);
                }
            }
            TileViewerCameraMode::Free => {
                if input.primary_pressed {
                    let delta = input.camera_motion;
                    tile_viewer.free_yaw -= delta.x * ROTATE_SENSITIVITY;
                    tile_viewer.free_pitch = (tile_viewer.free_pitch
                        - delta.y * ROTATE_SENSITIVITY)
                        .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
                }
                if input.camera_wheel.y != 0.0 {
                    let factor = (1.0 + input.camera_wheel.y * 0.1).max(0.1);
                    tile_viewer.free_speed_units_s =
                        (tile_viewer.free_speed_units_s * factor).clamp(0.05, 500.0);
                }

                let rot = Quat::from_rotation_y(tile_viewer.free_yaw)
                    * Quat::from_rotation_x(tile_viewer.free_pitch);
                let mut dir = Vec3::ZERO;
                if keys.pressed(KeyCode::KeyW) {
                    dir.z -= 1.0;
                }
                if keys.pressed(KeyCode::KeyS) {
                    dir.z += 1.0;
                }
                if keys.pressed(KeyCode::KeyA) {
                    dir.x -= 1.0;
                }
                if keys.pressed(KeyCode::KeyD) {
                    dir.x += 1.0;
                }
                if keys.pressed(KeyCode::KeyE) {
                    dir.y += 1.0;
                }
                if keys.pressed(KeyCode::KeyQ) {
                    dir.y -= 1.0;
                }
                if dir.length_squared() > 0.0 {
                    let sprint =
                        keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
                    let speed = tile_viewer.free_speed_units_s * if sprint { 4.0 } else { 1.0 };
                    tile_viewer.free_position += rot * dir.normalize() * speed * time.delta_secs();
                }
            }
        }
        return;
    }

    // While a placement tool is active, left-click is reserved for adding
    // features — don't also rotate the camera. Scroll-zoom stays usable.
    if input.primary_pressed && !planet.tool.placing() {
        let delta = input.camera_motion;
        orbit.azimuth += delta.x * ROTATE_SENSITIVITY;
        orbit.elevation = (orbit.elevation - delta.y * ROTATE_SENSITIVITY)
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    if input.camera_wheel.y != 0.0 {
        let surface = orbit.planet_render_radius;
        let min_h = (orbit.min_distance - surface).max(1e-4);
        let max_h = orbit.max_distance - surface;
        let h = (orbit.target_distance - surface).max(min_h);
        let log_h = h.ln() - input.camera_wheel.y * ZOOM_SENSITIVITY;
        let new_h = log_h.exp().clamp(min_h, max_h);
        orbit.target_distance = surface + new_h;
    }
}

pub(crate) fn gate_editor_input_sources(
    mut action_sources: ResMut<ActionSources>,
    mut egui_ctx: bevy_egui::EguiContexts,
) {
    let (pointer_busy, keyboard_busy) = egui_ctx
        .ctx_mut()
        .map(|ctx| (ctx.wants_pointer_input(), ctx.wants_keyboard_input()))
        .unwrap_or((false, false));
    thalos_input::gating::set_mouse_sources(&mut action_sources, !pointer_busy);
    thalos_input::gating::set_keyboard_source(&mut action_sources, !keyboard_busy);
}

/// `F` flips `full_bright` and forces atmosphere to the opposite state, so
/// the surface can be inspected unlit and unobscured in one keystroke.
pub(crate) fn toggle_fullbright_hotkey(
    input: Res<PlanetEditorInputIntent>,
    mut planet: ResMut<EditedPlanet>,
    mut egui_ctx: bevy_egui::EguiContexts,
) {
    if !input.toggle_fullbright {
        return;
    }
    if egui_ctx
        .ctx_mut()
        .is_ok_and(|ctx| ctx.wants_keyboard_input())
    {
        return;
    }
    planet.full_bright = !planet.full_bright;
    if planet.atmosphere.is_some() {
        planet.atmosphere_enabled = !planet.full_bright;
    }
    planet.uniforms_dirty = true;
}

pub(crate) fn camera_zoom_smoothing(
    mut orbit: ResMut<OrbitCamera>,
    tile_viewer: Res<TileViewerState>,
    time: Res<Time>,
) {
    if tile_viewer.enabled {
        return;
    }
    let speed = 10.0;
    let t = (speed * time.delta_secs()).min(1.0);
    let log_current = orbit.distance.ln();
    let log_target = orbit.target_distance.ln();
    orbit.distance = (log_current + (log_target - log_current) * t).exp();
}

pub(crate) fn tile_viewer_basis_and_center(
    state: &TileViewerState,
    active: &ActivePreviewSurface,
) -> Option<(DVec3, thalos_terrain_render::TerrainPatchBasis, DVec3)> {
    let surface = active.surface.as_ref()?;
    let dynamic_state = active.dynamic_state.as_ref()?;
    let center_dir = tile_viewer_center_dir(state);
    let basis = thalos_terrain_render::TerrainPatchBasis::from_normal(center_dir);
    let sample = surface_sample(surface, dynamic_state, center_dir, 1.0);
    let center = center_dir * (surface.static_surface.radius_m + sample.height_m) as f64;
    Some((center_dir, basis, center))
}

pub(crate) fn camera_apply_transform(
    orbit: Res<OrbitCamera>,
    tile_viewer: Res<TileViewerState>,
    active: Res<ActivePreviewSurface>,
    mut query: Query<(&mut Transform, &mut GridCell), With<EditorCamera>>,
) {
    let Ok((mut transform, mut cell)) = query.single_mut() else {
        return;
    };
    if tile_viewer.enabled {
        let Some((_center_dir, basis, center_body_m)) =
            tile_viewer_basis_and_center(&tile_viewer, &active)
        else {
            return;
        };
        let (position_body_m, look_target_body_m, up_body) = match tile_viewer.camera_mode {
            TileViewerCameraMode::Orbit => {
                let (sin_az, cos_az) = tile_viewer.orbit_azimuth.sin_cos();
                let (sin_el, cos_el) = tile_viewer.orbit_elevation.sin_cos();
                let local = DVec3::new(
                    (cos_el * sin_az * tile_viewer.orbit_distance) as f64,
                    (sin_el * tile_viewer.orbit_distance) as f64,
                    (cos_el * cos_az * tile_viewer.orbit_distance) as f64,
                );
                (
                    center_body_m + basis.local_to_body_vec(local),
                    center_body_m,
                    basis.normal.as_vec3(),
                )
            }
            TileViewerCameraMode::Free => {
                let local = tile_viewer.free_position.as_dvec3();
                let rot = Quat::from_rotation_y(tile_viewer.free_yaw)
                    * Quat::from_rotation_x(tile_viewer.free_pitch);
                let forward_local = (rot * -Vec3::Z).as_dvec3();
                let up_local = (rot * Vec3::Y).as_dvec3();
                let position = center_body_m + basis.local_to_body_vec(local);
                (
                    position,
                    position + basis.local_to_body_vec(forward_local),
                    basis.local_to_body_vec(up_local).as_vec3(),
                )
            }
        };

        let frame = ReferenceFrame::default();
        let (new_cell, translation) = frame.translation_to_grid(position_body_m);
        *cell = new_cell;
        *transform = Transform::from_translation(translation).looking_at(
            (look_target_body_m - position_body_m).as_vec3() + translation,
            up_body,
        );
        return;
    }

    *cell = GridCell::default();
    let (sin_az, cos_az) = orbit.azimuth.sin_cos();
    let (sin_el, cos_el) = orbit.elevation.sin_cos();
    let pos = Vec3::new(
        cos_el * sin_az * orbit.distance,
        sin_el * orbit.distance,
        cos_el * cos_az * orbit.distance,
    );
    *transform = Transform::from_translation(pos).looking_at(Vec3::ZERO, Vec3::Y);
}
