#![allow(clippy::too_many_arguments)]

use super::*;

/// Applies shader-uniform-only changes to the current material.
#[allow(clippy::too_many_arguments)]
/// Mirror the editor's world-to-body orientation quaternion into the resource
/// the surface overlays read. The overlay plugin inverts it for the shell
/// transform so metadata colors physically track the rendered body.
pub(crate) fn sync_surface_overlay_orientation(
    planet: Res<EditedPlanet>,
    mut orientation: ResMut<SurfaceOverlayOrientation>,
) {
    let q = body_orientation(&planet);
    if orientation.0 != q {
        orientation.0 = q;
    }
}

pub(crate) fn apply_uniform_changes(
    mut planet: ResMut<EditedPlanet>,
    equirect_viewer: Res<EquirectViewerState>,
    input: Res<BodyEditorInputIntent>,
    terrain_q: Query<&PlanetMaterialHandle, With<PreviewPlanet>>,
    halo_q: Query<&PlanetHaloMaterialHandle, With<PreviewPlanet>>,
    gas_q: Query<&GasGiantMaterialHandle, With<PreviewPlanet>>,
    ring_q: Query<&RingMaterialHandle, With<PreviewRing>>,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut planet_halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
    mut gas_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
    mut last_force: Local<bool>,
) {
    // Any of: the on-planet field overlay being on, OR space being held,
    // forces fullbright + atmosphere-off so debug views read cleanly. Press /
    // release of either source must rewrite uniforms, so we track the combined
    // flag with a Local — the overlay toggle and semantic input can both
    // mutably tick every frame, so `is_changed()` on them is unusable.
    let overlays_on = equirect_viewer.overlay_on_planet;
    let space_held = input.overlay_suppress;
    let force = overlays_on || space_held;
    let force_changed = *last_force != force;
    if force_changed {
        *last_force = force;
    }
    if !planet.uniforms_dirty && !force_changed {
        return;
    }
    planet.uniforms_dirty = false;

    let (_, _, wrap) = lighting_for(&planet);
    let scene = scene_lighting_for(&planet);
    let fullbright = if force || planet.full_bright {
        1.0
    } else {
        0.0
    };
    let atmosphere = if force {
        AtmosphereBlock::default()
    } else {
        active_atmosphere(&planet)
    };
    let q = body_orientation(&planet);
    let q4 = Vec4::new(q.x, q.y, q.z, q.w);

    match &planet.mode {
        BodyMode::Terrain { .. } => {
            for handle in &terrain_q {
                let Some(mat) = planet_materials.get_mut(&handle.0) else {
                    continue;
                };
                mat.params.terminator_wrap = wrap;
                mat.params.fullbright = fullbright;
                mat.params.orientation = q4;
                mat.params.scene = scene.clone();
                mat.atmosphere = atmosphere;
            }
            for handle in &halo_q {
                let Some(mat) = planet_halo_materials.get_mut(&handle.0) else {
                    continue;
                };
                mat.params.terminator_wrap = wrap;
                mat.params.fullbright = fullbright;
                mat.params.orientation = q4;
                mat.params.scene = scene.clone();
                mat.atmosphere = atmosphere;
            }
        }
        BodyMode::GasGiant { .. } => {
            for handle in &gas_q {
                let Some(mat) = gas_materials.get_mut(&handle.0) else {
                    continue;
                };
                mat.params.orientation = q4;
                mat.params.scene = scene.clone();
            }
        }
        BodyMode::Star => {}
    }

    // Ring scene lighting refresh runs regardless of body mode — rings
    // are now sibling to `BodyMode`, not nested inside it.
    if planet.rings.is_some() {
        for handle in &ring_q {
            let Some(mat) = ring_materials.get_mut(&handle.0) else {
                continue;
            };
            mat.params.scene = scene.clone();
        }
    }
}

pub(crate) fn patch_preview_reference_cloud_cover(
    clouds: Res<ReferenceClouds>,
    planet: Res<EditedPlanet>,
    terrain_q: Query<
        (&PlanetMaterialHandle, Option<&PlanetHaloMaterialHandle>),
        With<PreviewPlanet>,
    >,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut planet_halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
) {
    if planet.atmosphere.is_none() {
        return;
    }
    let Some(cube) = clouds.cube(&planet.selected_body) else {
        return;
    };

    for (body_handle, halo_handle) in &terrain_q {
        if let Some(mat) = planet_materials.get_mut(&body_handle.0)
            && mat.cloud_cover != cube
        {
            mat.cloud_cover = cube.clone();
        }
        if let Some(halo_handle) = halo_handle
            && let Some(mat) = planet_halo_materials.get_mut(&halo_handle.0)
            && mat.cloud_cover != cube
        {
            mat.cloud_cover = cube.clone();
        }
    }
}

pub(crate) fn write_cloud_animation(
    atmosphere: &mut AtmosphereBlock,
    elapsed_s: f64,
    bands: Option<(Vec4, Vec4, Vec4, Vec4)>,
) {
    atmosphere.cloud_dynamics.y = elapsed_s as f32;
    if let Some((bands_a, bands_b, bands_c, bands_d)) = bands {
        atmosphere.cloud_bands_a = bands_a;
        atmosphere.cloud_bands_b = bands_b;
        atmosphere.cloud_bands_c = bands_c;
        atmosphere.cloud_bands_d = bands_d;
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn update_preview_atmosphere(
    mut clock: ResMut<PreviewAtmosphereClock>,
    time: Res<Time>,
    planet: Res<EditedPlanet>,
    mut query: Query<
        (
            &PlanetMaterialHandle,
            Option<&PlanetHaloMaterialHandle>,
            Option<&mut PreviewCloudBandState>,
        ),
        With<PreviewPlanet>,
    >,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut planet_halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
) {
    clock.elapsed_s += time.delta_secs() as f64;
    if !planet.atmosphere_enabled {
        return;
    }

    for (handle, halo_handle, cloud_state) in &mut query {
        let Some(mat) = planet_materials.get(&handle.0) else {
            continue;
        };
        let scroll = mat.atmosphere.cloud_dynamics.x as f64;
        let diff = mat.atmosphere.cloud_shape.w.clamp(0.0, 1.0) as f64;
        let bands = if scroll.abs() >= 1e-12 {
            cloud_state.map(|mut state| {
                let dt = time.delta_secs() as f64;
                for i in 0..CLOUD_BAND_COUNT {
                    let sin2 = i as f64 / (CLOUD_BAND_COUNT - 1) as f64;
                    let lat_factor = 1.0 - diff * sin2;
                    let omega = scroll * lat_factor;
                    state.phases[i] =
                        (state.phases[i] + omega * dt).rem_euclid(std::f64::consts::TAU);
                }

                let p = &state.phases;
                (
                    Vec4::new(p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32),
                    Vec4::new(p[4] as f32, p[5] as f32, p[6] as f32, p[7] as f32),
                    Vec4::new(p[8] as f32, p[9] as f32, p[10] as f32, p[11] as f32),
                    Vec4::new(p[12] as f32, p[13] as f32, p[14] as f32, p[15] as f32),
                )
            })
        } else {
            None
        };

        if let Some(mat) = planet_materials.get_mut(&handle.0) {
            write_cloud_animation(&mut mat.atmosphere, clock.elapsed_s, bands);
        }
        if let Some(halo_handle) = halo_handle
            && let Some(mat) = planet_halo_materials.get_mut(&halo_handle.0)
        {
            write_cloud_animation(&mut mat.atmosphere, clock.elapsed_s, bands);
        }
    }
}

pub(crate) fn dispatch_rebake(
    mut commands: Commands,
    mut planet: ResMut<EditedPlanet>,
    mut status: ResMut<TerrainGenStatus>,
    preview_q: Query<(Entity, &Children), With<PreviewPlanet>>,
    pending_q: Query<&PendingTerrainGen, With<PreviewPlanet>>,
) {
    let requested_bake = planet.requested_bake;
    if requested_bake.is_none() && !planet.terrain_dirty {
        return;
    }
    // Debounce live edits so a slider drag doesn't queue throwaway tasks.
    // Explicit bake buttons bypass this so a deliberate request fires now.
    if requested_bake.is_none()
        && let Some(last) = planet.last_edit
        && last.elapsed().as_millis() < REBAKE_DEBOUNCE_MS
    {
        return;
    }
    // One bake at a time. The dirty flag stays set so we'll retry once the
    // current task finalizes.
    if !pending_q.is_empty() {
        return;
    }
    let BodyMode::Terrain {
        ref terrain,
        ref tectonics,
        tidal_axis,
    } = planet.mode
    else {
        planet.terrain_dirty = false;
        planet.requested_bake = None;
        return;
    };
    let Ok((entity, children)) = preview_q.single() else {
        return;
    };
    let Some(mesh_entity) = children.iter().next() else {
        return;
    };
    let terrain = terrain.clone();
    let tectonics = tectonics.clone();
    let radius_m = planet.radius_m;
    let gravity_m_s2 = planet.gravity_m_s2;
    let axial_tilt_rad = planet.axial_tilt_rad;
    let bake_mode = requested_bake.unwrap_or(TerrainBakeMode::Preview);
    let resolution_override = bake_mode.resolution_override();
    planet.terrain_dirty = false;
    planet.requested_bake = None;
    planet.last_bake_mode = bake_mode;

    let task = dispatch_terrain_bake(
        &terrain,
        tectonics.as_ref(),
        radius_m,
        gravity_m_s2,
        tidal_axis,
        axial_tilt_rad,
        planet.selected_body.clone(),
        resolution_override,
    );
    status.current_started = Some(Instant::now());
    commands
        .entity(entity)
        .insert(PendingTerrainGen { task, mesh_entity });
}
