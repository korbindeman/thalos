#![allow(clippy::too_many_arguments)]

use super::*;

pub(crate) fn sun_direction(azimuth: f32, elevation: f32) -> Vec3 {
    let (sa, ca) = azimuth.sin_cos();
    let (se, ce) = elevation.sin_cos();
    Vec3::new(ce * sa, se, ce * ca)
}

/// World→body orientation quaternion for the preview, matching the game's
/// `update_planet_orientations` at sim_time = 0 (free-spinning case): the
/// `Ry(phase) * Rx(tilt)` composition collapses to `Rx(tilt)` since phase = 0.
/// Stored in `PlanetParams.orientation` / `GasGiantParams.orientation`, where
/// the shaders use it to rotate world-space directions into body-local space.
pub(crate) fn body_orientation(planet: &EditedPlanet) -> Quat {
    Quat::from_rotation_x(planet.axial_tilt_rad)
}

// Body → editor params conversion

pub(crate) struct ResolvedBody {
    pub(crate) radius_m: f64,
    pub(crate) gravity_m_s2: f32,
    pub(crate) axial_tilt_rad: f32,
    pub(crate) mode: BodyMode,
    pub(crate) rings: Option<EditorRings>,
    pub(crate) atmosphere: Option<EditorAtmosphere>,
    pub(crate) heliocentric_distance_m: f64,
    pub(crate) sun_orbital_elevation: f32,
}

pub(crate) fn build_params_for_body(
    system: &SolarSystemDefinition,
    body: &thalos_physics_canonical::types::BodyDefinition,
) -> ResolvedBody {
    let mode = if body.kind == BodyKind::Star {
        BodyMode::Star
    } else if let Some(atmos) = &body.atmosphere {
        let layers = Box::new(GasGiantLayers::from_params(
            atmos,
            body.rings.as_ref(),
            body.radius_m as f32 / RENDER_RADIUS,
        ));
        BodyMode::GasGiant { layers }
    } else if body.terrain.is_some() {
        BodyMode::Terrain {
            terrain: body.terrain.clone(),
            tectonics: body.tectonics.clone(),
            tidal_axis: matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z),
        }
    } else {
        BodyMode::Terrain {
            terrain: placeholder_terrain_config(),
            tectonics: body.tectonics.clone(),
            tidal_axis: matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z),
        }
    };

    let rings = body.rings.as_ref().map(|rings| EditorRings {
        inner_radius_m: rings.inner_radius_m,
        outer_radius_m: rings.outer_radius_m,
        layers: Box::new(RingLayers::from_system(rings)),
    });
    let atmosphere = body.terrestrial_atmosphere.as_ref().map(|atmos| {
        let meters_per_render_unit = body.radius_m as f32 / RENDER_RADIUS;
        EditorAtmosphere {
            block: AtmosphereBlock::from_terrestrial(atmos, meters_per_render_unit),
        }
    });

    ResolvedBody {
        radius_m: body.radius_m,
        gravity_m_s2: (body.gm / (body.radius_m * body.radius_m)) as f32,
        axial_tilt_rad: body.axial_tilt_rad as f32,
        mode,
        rings,
        atmosphere,
        heliocentric_distance_m: heliocentric_sma(system, body),
        sun_orbital_elevation: orbital_sun_elevation(system, body),
    }
}

pub(crate) fn placeholder_terrain_config() -> TerrainConfig {
    TerrainConfig::Ocean(OceanTerrainConfig {
        seed: 0,
        cubemap_resolution: Some(64),
        seabed_albedo: [0.02, 0.05, 0.10],
        water_roughness: 0.04,
        sea_level_m: 1.0,
    })
}

pub(crate) fn heliocentric_sma(
    system: &SolarSystemDefinition,
    start: &thalos_physics_canonical::types::BodyDefinition,
) -> f64 {
    let mut current = start;
    for _ in 0..32 {
        match current.parent {
            None => return AU_M,
            Some(parent_id) => {
                let parent = &system.bodies[parent_id];
                if parent.kind == BodyKind::Star {
                    return current
                        .orbital_elements
                        .as_ref()
                        .map(|oe| oe.semi_major_axis_m)
                        .unwrap_or(AU_M);
                }
                current = parent;
            }
        }
    }
    AU_M
}

pub(crate) fn light_intensity_at(distance_m: f64) -> f32 {
    let ratio = AU_M / distance_m.max(1.0);
    LIGHT_AT_1AU * (ratio * ratio) as f32
}

pub(crate) fn orbital_sun_elevation(
    system: &SolarSystemDefinition,
    body: &thalos_physics_canonical::types::BodyDefinition,
) -> f32 {
    if body.kind == BodyKind::Star {
        return 0.0;
    }

    let Some(star_id) = system.bodies.iter().position(|b| b.kind == BodyKind::Star) else {
        return 0.0;
    };

    let ephemeris = PatchedConics::new(system, 1.0);
    let body_state = ephemeris.state(body.id, Epoch::ZERO);
    let star_state = ephemeris.state(star_id, Epoch::ZERO);
    let to_sun = star_state.position - body_state.position;
    let distance = to_sun.length();
    if distance <= f64::EPSILON {
        return 0.0;
    }

    (to_sun.y / distance).clamp(-1.0, 1.0).asin() as f32
}

pub(crate) fn lighting_for(planet: &EditedPlanet) -> (f32, f32, f32) {
    (
        planet.light_intensity,
        if planet.ambient_light {
            AMBIENT_INTENSITY
        } else {
            0.0
        },
        0.0,
    )
}

/// Build a `SceneLighting` for the preview. Single star, no eclipse
/// occluders, no planetshine — editor scenes are one body at a time.
pub(crate) fn scene_lighting_for(planet: &EditedPlanet) -> SceneLighting {
    let (light_intensity, ambient_intensity, _wrap) = lighting_for(planet);
    let dir = sun_direction(planet.sun_azimuth, planet.sun_orbital_elevation);
    let mut scene = SceneLighting {
        ambient_intensity,
        star_count: 1,
        ..default()
    };
    scene.stars[0] = StarLight {
        dir_flux: Vec4::new(dir.x, dir.y, dir.z, light_intensity),
        color: Vec4::new(1.0, 1.0, 1.0, 0.0),
    };
    scene
}

pub(crate) fn active_atmosphere(planet: &EditedPlanet) -> AtmosphereBlock {
    if !planet.atmosphere_enabled {
        return AtmosphereBlock::default();
    }
    planet
        .atmosphere
        .as_ref()
        .map(|atmos| atmos.block)
        .unwrap_or_default()
}

pub(crate) fn cloud_cover_for(
    planet: &EditedPlanet,
    reference_clouds: &ReferenceClouds,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    cloud_cover_image_for_body(&planet.selected_body, reference_clouds, images).0
}
