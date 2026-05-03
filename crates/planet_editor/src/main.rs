use std::collections::HashMap;
use std::time::{Duration, Instant};

use bevy::asset::AssetPlugin;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use bevy::window::PresentMode;
use bevy_egui::egui;
use thalos_physics::body_state_provider::BodyStateProvider;
use thalos_physics::parsing::load_solar_system;
use thalos_physics::patched_conics::PatchedConics;
use thalos_physics::types::{BodyDefinition, BodyId, BodyKind, SolarSystemDefinition};
use thalos_planet_rendering::{
    AtmosphereBlock, CLOUD_BAND_COUNT, GasGiantLayers, GasGiantMaterial, GasGiantMaterialHandle,
    GasGiantParams, PlanetDetailParams, PlanetHaloMaterial, PlanetHaloMaterialHandle,
    PlanetMaterial, PlanetMaterialHandle, PlanetParams, PlanetRenderingPlugin, ReferenceClouds,
    RingLayers, RingMaterial, RingMaterialHandle, RingParams, SceneLighting, StarLight,
    bake_from_body_data, build_ring_mesh, cloud_cover_image_for_body,
    convert_reference_clouds_when_ready, load_reference_cloud_sources,
};
use thalos_terrain_gen::{
    AirlessImpactProjectionConfig, AuthoredFeatureConfig, BodyData, ColdDesertProjectionConfig,
    FeatureId, FeatureManifest, FeatureProjectionConfig, FeatureSeed, FeatureSeedStream,
    GeneratorParams, TerrainCompileContext, TerrainCompileOptions, TerrainConfig,
    compile_terrain_config, plan_initial_compilation, sub_seed,
};

mod sky_backdrop;

use sky_backdrop::SkyBackdropPlugin;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LIGHT_AT_1AU: f32 = 10.0;
const AMBIENT_INTENSITY: f32 = 0.05;
const AU_M: f64 = 1.496e11;
const DEFAULT_BODY_NAME: &str = "Mira";
const RENDER_RADIUS: f32 = 1.5;

const SOLAR_SYSTEM_RON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/solar_system.ron"
));

// ---------------------------------------------------------------------------
// Body rendering mode
// ---------------------------------------------------------------------------

enum BodyMode {
    Terrain {
        terrain: TerrainConfig,
        tidal_axis: Option<Vec3>,
    },
    GasGiant {
        layers: Box<GasGiantLayers>,
    },
    Star,
}

/// Ring system parameters held alongside [`BodyMode`] on
/// [`EditedPlanet`]. Sibling, not nested, so any body can have a ring.
struct EditorRings {
    inner_radius_m: f32,
    outer_radius_m: f32,
    layers: Box<RingLayers>,
}

struct EditorAtmosphere {
    block: AtmosphereBlock,
    cloud_seed: Option<u64>,
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct SystemData {
    system: SolarSystemDefinition,
}

#[derive(Resource)]
struct EditedPlanet {
    selected_body: String,
    radius_m: f64,
    gravity_m_s2: f32,
    axial_tilt_rad: f32,
    mode: BodyMode,
    rings: Option<EditorRings>,
    atmosphere: Option<EditorAtmosphere>,
    atmosphere_enabled: bool,
    heliocentric_distance_m: f64,
    light_intensity: f32,
    sun_azimuth: f32,
    sun_orbital_elevation: f32,
    full_bright: bool,
    ambient_light: bool,
    terrain_dirty: bool,
    uniforms_dirty: bool,
    /// Body was switched — need to tear down and respawn the preview mesh.
    body_changed: bool,
}

#[derive(Resource, Default)]
struct TerrainGenStatus {
    current_started: Option<Instant>,
    last_duration: Option<Duration>,
}

#[derive(Resource)]
struct BillboardMesh(Handle<Mesh>);

#[derive(Component)]
struct PendingTerrainGen {
    task: Task<BodyData>,
    mesh_entity: Entity,
}

fn sun_direction(azimuth: f32, elevation: f32) -> Vec3 {
    let (sa, ca) = azimuth.sin_cos();
    let (se, ce) = elevation.sin_cos();
    Vec3::new(ce * sa, se, ce * ca)
}

// ---------------------------------------------------------------------------
// Body → editor params conversion
// ---------------------------------------------------------------------------

struct ResolvedBody {
    radius_m: f64,
    gravity_m_s2: f32,
    axial_tilt_rad: f32,
    mode: BodyMode,
    rings: Option<EditorRings>,
    atmosphere: Option<EditorAtmosphere>,
    heliocentric_distance_m: f64,
    sun_orbital_elevation: f32,
}

fn build_params_for_body(
    system: &SolarSystemDefinition,
    body: &thalos_physics::types::BodyDefinition,
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
            tidal_axis: matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z),
        }
    } else {
        BodyMode::Terrain {
            terrain: placeholder_terrain_config(),
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
            cloud_seed: atmos.clouds.as_ref().map(|clouds| clouds.seed),
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

fn placeholder_terrain_config() -> TerrainConfig {
    TerrainConfig::LegacyPipeline(GeneratorParams {
        seed: 0,
        composition: thalos_terrain_gen::Composition::new(1.0, 0.0, 0.0, 0.0, 0.0),
        cubemap_resolution: 64,
        body_age_gyr: 4.5,
        pipeline: Vec::new(),
    })
}

fn heliocentric_sma(
    system: &SolarSystemDefinition,
    start: &thalos_physics::types::BodyDefinition,
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

fn light_intensity_at(distance_m: f64) -> f32 {
    let ratio = AU_M / distance_m.max(1.0);
    LIGHT_AT_1AU * (ratio * ratio) as f32
}

fn orbital_sun_elevation(
    system: &SolarSystemDefinition,
    body: &thalos_physics::types::BodyDefinition,
) -> f32 {
    if body.kind == BodyKind::Star {
        return 0.0;
    }

    let Some(star_id) = system.bodies.iter().position(|b| b.kind == BodyKind::Star) else {
        return 0.0;
    };

    let ephemeris = PatchedConics::new(system, 1.0);
    let body_state = ephemeris.query_body(body.id, 0.0);
    let star_state = ephemeris.query_body(star_id, 0.0);
    let to_sun = star_state.position - body_state.position;
    let distance = to_sun.length();
    if distance <= f64::EPSILON {
        return 0.0;
    }

    (to_sun.y / distance).clamp(-1.0, 1.0).asin() as f32
}

fn lighting_for(planet: &EditedPlanet) -> (f32, f32, f32) {
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
fn scene_lighting_for(planet: &EditedPlanet) -> SceneLighting {
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

fn active_atmosphere(planet: &EditedPlanet) -> AtmosphereBlock {
    if !planet.atmosphere_enabled {
        return AtmosphereBlock::default();
    }
    planet
        .atmosphere
        .as_ref()
        .map(|atmos| atmos.block)
        .unwrap_or_default()
}

fn cloud_cover_for(
    planet: &EditedPlanet,
    reference_clouds: &ReferenceClouds,
    images: &mut Assets<Image>,
) -> Handle<Image> {
    let cloud_seed = planet
        .atmosphere
        .as_ref()
        .and_then(|atmos| atmos.cloud_seed);
    cloud_cover_image_for_body(&planet.selected_body, cloud_seed, reference_clouds, images).0
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Component)]
struct PreviewPlanet;

#[derive(Component)]
struct PreviewRing;

#[derive(Component)]
struct PreviewAtmosphereHalo;

#[derive(Component, Default)]
struct PreviewCloudBandState {
    phases: [f64; CLOUD_BAND_COUNT],
}

#[derive(Resource, Default)]
struct PreviewAtmosphereClock {
    elapsed_s: f64,
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

const CAMERA_VFOV: f32 = std::f32::consts::FRAC_PI_4;
const PLANET_VIEW_FRACTION: f32 = 0.40;
const SURFACE_MARGIN: f32 = 1.35;

#[derive(Component)]
struct EditorCamera;

#[derive(Resource)]
struct OrbitCamera {
    azimuth: f32,
    elevation: f32,
    distance: f32,
    target_distance: f32,
    min_distance: f32,
    max_distance: f32,
    planet_render_radius: f32,
}

impl OrbitCamera {
    fn from_render_radius(r: f32) -> Self {
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

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        thalos_planet_rendering::space_camera_post_stack(),
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        Msaa::Off,
        EditorCamera,
    ));
}

fn camera_input(
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut orbit: ResMut<OrbitCamera>,
    mut egui_ctx: bevy_egui::EguiContexts,
) {
    if egui_ctx
        .ctx_mut()
        .is_ok_and(|ctx| ctx.wants_pointer_input())
    {
        return;
    }

    const ROTATE_SENSITIVITY: f32 = 0.005;
    const ZOOM_SENSITIVITY: f32 = 0.04;

    if mouse.pressed(MouseButton::Left) {
        let delta = motion.delta;
        orbit.azimuth += delta.x * ROTATE_SENSITIVITY;
        orbit.elevation = (orbit.elevation - delta.y * ROTATE_SENSITIVITY)
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    if scroll.delta.y != 0.0 {
        let surface = orbit.planet_render_radius;
        let min_h = (orbit.min_distance - surface).max(1e-4);
        let max_h = orbit.max_distance - surface;
        let h = (orbit.target_distance - surface).max(min_h);
        let log_h = h.ln() - scroll.delta.y * ZOOM_SENSITIVITY;
        let new_h = log_h.exp().clamp(min_h, max_h);
        orbit.target_distance = surface + new_h;
    }
}

fn camera_zoom_smoothing(mut orbit: ResMut<OrbitCamera>, time: Res<Time>) {
    let speed = 10.0;
    let t = (speed * time.delta_secs()).min(1.0);
    let log_current = orbit.distance.ln();
    let log_target = orbit.target_distance.ln();
    orbit.distance = (log_current + (log_target - log_current) * t).exp();
}

fn camera_apply_transform(
    orbit: Res<OrbitCamera>,
    mut query: Query<&mut Transform, With<EditorCamera>>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };
    let (sin_az, cos_az) = orbit.azimuth.sin_cos();
    let (sin_el, cos_el) = orbit.elevation.sin_cos();
    let pos = Vec3::new(
        cos_el * sin_az * orbit.distance,
        sin_el * orbit.distance,
        cos_el * cos_az * orbit.distance,
    );
    *transform = Transform::from_translation(pos).looking_at(Vec3::ZERO, Vec3::Y);
}

// ---------------------------------------------------------------------------
// Preview spawning
// ---------------------------------------------------------------------------

#[cfg(debug_assertions)]
const DEV_CRATER_SCALE: f32 = 0.1;
#[cfg(not(debug_assertions))]
const DEV_CRATER_SCALE: f32 = 1.0;

fn terrain_cache_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/terrain_cache")
}

fn dispatch_terrain_bake(
    terrain: &TerrainConfig,
    radius_m: f64,
    gravity_m_s2: f32,
    tidal_axis: Option<Vec3>,
    axial_tilt_rad: f32,
    body_name: String,
) -> Task<BodyData> {
    let radius_m = radius_m as f32;
    let terrain = terrain.clone();
    AsyncComputeTaskPool::get().spawn(async move {
        let cache_dir = terrain_cache_dir();
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
        };
        let key = thalos_terrain_gen::cache::terrain_cache_key(&terrain, &context, options);
        let path = thalos_terrain_gen::cache::cache_path(&cache_dir, &body_name, key);
        if let Some(data) = thalos_terrain_gen::cache::load(&path, key) {
            info!("terrain cache hit: {body_name} via {route}");
            return data;
        }
        info!("terrain cache miss, baking {body_name} via {route}");
        let data = compile_terrain_config(&terrain, &context, options)
            .unwrap_or_else(|e| panic!("terrain compile failed for {body_name}: {e}"));
        match thalos_terrain_gen::cache::store(&path, key, &data) {
            Ok(()) => info!("terrain cache wrote: {body_name}"),
            Err(e) => warn!("terrain cache write failed for {body_name}: {e}"),
        }
        data
    })
}

#[allow(clippy::too_many_arguments)]
fn spawn_preview(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    std_materials: &mut Assets<StandardMaterial>,
    gas_giant_materials: &mut Assets<GasGiantMaterial>,
    ring_materials: &mut Assets<RingMaterial>,
    billboard: &BillboardMesh,
    planet: &EditedPlanet,
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
                planet.radius_m,
                planet.gravity_m_s2,
                *tidal_axis,
                planet.axial_tilt_rad,
                planet.selected_body.clone(),
            );
            status.current_started = Some(Instant::now());
            commands
                .entity(parent)
                .insert(PendingTerrainGen { task, mesh_entity });
        }
        BodyMode::GasGiant { layers } => {
            let scene = scene_lighting_for(planet);
            let tilt = Quat::from_rotation_x(planet.axial_tilt_rad);

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

fn spawn_preview_planet(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut gas_giant_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
    mut status: ResMut<TerrainGenStatus>,
    planet: Res<EditedPlanet>,
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
        &planet,
        &mut status,
    );

    commands.insert_resource(billboard);
}

#[allow(clippy::too_many_arguments)]
fn finalize_terrain_bake(
    mut commands: Commands,
    mut pending_q: Query<(Entity, &mut PendingTerrainGen), With<PreviewPlanet>>,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut planet_halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut status: ResMut<TerrainGenStatus>,
    billboard: Res<BillboardMesh>,
    planet: Res<EditedPlanet>,
    reference_clouds: Res<ReferenceClouds>,
    children_q: Query<&Children>,
    halo_q: Query<Entity, With<PreviewAtmosphereHalo>>,
) {
    for (entity, mut pending) in &mut pending_q {
        let Some(body) = block_on(poll_once(&mut pending.task)) else {
            continue;
        };

        let detail =
            PlanetDetailParams::from_body(&body.detail_params, body.cubemap_bake_threshold_m);
        let height_range = body.height_range;
        let textures = bake_from_body_data(&body, &mut images, &mut storage_buffers);
        let (_, _, wrap) = lighting_for(&planet);
        let scene = scene_lighting_for(&planet);

        let body_seed = body.detail_params.seed;
        let coastline_seed = (body_seed as u32) ^ ((body_seed >> 32) as u32) ^ 0xC0A5_711E_u32;
        let has_ocean = body.sea_level_m.is_some();
        let coastline_warp_amp_radians = if has_ocean { 8.0e-4 } else { 0.0 };
        let coastline_jitter_amp_m = if has_ocean { 30.0 } else { 0.0 };
        let atmosphere = active_atmosphere(&planet);
        let cloud_cover = cloud_cover_for(&planet, &reference_clouds, &mut images);

        let planet_material = PlanetMaterial {
            params: PlanetParams {
                radius: RENDER_RADIUS,
                height_range,
                terminator_wrap: wrap,
                fullbright: if planet.full_bright { 1.0 } else { 0.0 },
                scene,
                sea_level_m: body.sea_level_m.unwrap_or(-1.0e9),
                coastline_warp_amp_radians,
                coastline_jitter_amp_m,
                coastline_seed,
                ..default()
            },
            albedo: textures.albedo,
            height: textures.height,
            detail,
            roughness: textures.roughness,
            craters: textures.craters,
            cell_index: textures.cell_index,
            feature_ids: textures.feature_ids,
            atmosphere,
            cloud_cover,
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

        commands
            .entity(entity)
            .insert(PlanetMaterialHandle(mat_handle))
            .insert(PlanetHaloMaterialHandle(halo_handle))
            .remove::<PendingTerrainGen>();
        if planet
            .atmosphere
            .as_ref()
            .and_then(|atmos| atmos.cloud_seed)
            .is_some()
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

// ---------------------------------------------------------------------------
// Body switching — tear down old preview and spawn new one
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn handle_body_switch(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut gas_giant_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
    mut planet: ResMut<EditedPlanet>,
    mut status: ResMut<TerrainGenStatus>,
    billboard: Res<BillboardMesh>,
    preview_q: Query<Entity, With<PreviewPlanet>>,
) {
    if !planet.body_changed {
        return;
    }
    planet.body_changed = false;
    planet.terrain_dirty = false;

    for entity in &preview_q {
        commands.entity(entity).despawn();
    }

    spawn_preview(
        &mut commands,
        &mut meshes,
        &mut std_materials,
        &mut gas_giant_materials,
        &mut ring_materials,
        &billboard,
        &planet,
        &mut status,
    );
}

// ---------------------------------------------------------------------------
// Editor UI (egui)
// ---------------------------------------------------------------------------

fn render_body_tree_ui(
    ui: &mut egui::Ui,
    system: &SolarSystemDefinition,
    selected_body: Option<BodyId>,
) -> Option<BodyId> {
    let mut children_of: HashMap<BodyId, Vec<&BodyDefinition>> = HashMap::new();
    for body in &system.bodies {
        if let Some(parent) = body.parent {
            children_of.entry(parent).or_default().push(body);
        }
    }
    // Stable order: the file's listing order.
    for kids in children_of.values_mut() {
        kids.sort_by_key(|b| b.id);
    }

    let root = system.bodies.iter().find(|b| b.parent.is_none())?;
    let mut clicked: Option<BodyId> = None;

    // Major tree: star and its non-minor descendants.
    render_body_tree_row(ui, root, selected_body, &mut clicked, 0);
    if let Some(kids) = children_of.get(&root.id) {
        for child in kids.iter().filter(|b| !is_minor(b.kind)) {
            render_body_subtree(ui, child, &children_of, selected_body, &mut clicked, 1);
        }
    }

    // Minor bodies: collapsing group of dwarf planets / centaurs /
    // comets that orbit the star, with their own descendants nested.
    let minor: Vec<&BodyDefinition> = children_of
        .get(&root.id)
        .map(|kids| kids.iter().copied().filter(|b| is_minor(b.kind)).collect())
        .unwrap_or_default();
    if !minor.is_empty() {
        ui.collapsing("Minor bodies", |ui| {
            for body in minor {
                render_body_subtree(ui, body, &children_of, selected_body, &mut clicked, 0);
            }
        });
    }

    clicked
}

fn is_minor(kind: BodyKind) -> bool {
    matches!(
        kind,
        BodyKind::DwarfPlanet | BodyKind::Centaur | BodyKind::Comet
    )
}

fn render_body_subtree(
    ui: &mut egui::Ui,
    body: &BodyDefinition,
    children_of: &HashMap<BodyId, Vec<&BodyDefinition>>,
    selected_body: Option<BodyId>,
    clicked: &mut Option<BodyId>,
    depth: u32,
) {
    render_body_tree_row(ui, body, selected_body, clicked, depth);
    if let Some(kids) = children_of.get(&body.id) {
        for child in kids {
            render_body_subtree(ui, child, children_of, selected_body, clicked, depth + 1);
        }
    }
}

fn render_body_tree_row(
    ui: &mut egui::Ui,
    body: &BodyDefinition,
    selected_body: Option<BodyId>,
    clicked: &mut Option<BodyId>,
    depth: u32,
) {
    let is_selected = selected_body == Some(body.id);

    ui.horizontal(|ui| {
        ui.add_space(depth as f32 * 14.0);

        let [r, g, b] = body.color;
        let dot_color = egui::Color32::from_rgb(
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b.clamp(0.0, 1.0) * 255.0) as u8,
        );
        let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 4.0, dot_color);
        ui.add_space(4.0);

        let label = ui.add(egui::Button::selectable(is_selected, &body.name).frame(false));
        if label.clicked() {
            *clicked = Some(body.id);
        }
    });
}

fn select_body(planet: &mut EditedPlanet, system: &SolarSystemDefinition, body_id: BodyId) {
    let body = &system.bodies[body_id];
    if planet.selected_body == body.name {
        return;
    }

    let resolved = build_params_for_body(system, body);
    planet.radius_m = resolved.radius_m;
    planet.gravity_m_s2 = resolved.gravity_m_s2;
    planet.axial_tilt_rad = resolved.axial_tilt_rad;
    planet.mode = resolved.mode;
    planet.rings = resolved.rings;
    planet.atmosphere = resolved.atmosphere;
    planet.heliocentric_distance_m = resolved.heliocentric_distance_m;
    planet.light_intensity = light_intensity_at(resolved.heliocentric_distance_m);
    planet.sun_orbital_elevation = resolved.sun_orbital_elevation;
    planet.selected_body = body.name.clone();
    planet.body_changed = true;
    planet.uniforms_dirty = true;
}

fn fires(r: &egui::Response) -> bool {
    r.drag_stopped() || (r.changed() && !r.dragged())
}

fn draw_airless_projection_controls(
    ui: &mut egui::Ui,
    projection: &mut AirlessImpactProjectionConfig,
) -> bool {
    let mut changed = false;
    ui.collapsing("Projection", |ui| {
        changed |= fires(&ui.add(
            egui::Slider::new(&mut projection.base_crater_count, 0..=500_000).text("Base craters"),
        ));
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.min_crater_radius_m, 100.0..=5_000.0)
                    .text("Min crater m"),
            ),
        );
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.max_crater_radius_m, 10_000.0..=180_000.0)
                    .text("Max crater m"),
            ),
        );
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.cubemap_bake_threshold_m, 250.0..=5_000.0)
                    .text("Bake threshold m"),
            ),
        );
        changed |= fires(&ui.add(
            egui::Slider::new(&mut projection.mare_fill_fraction, 0.0..=1.0).text("Mare fill"),
        ));
        changed |= fires(
            &ui.add(
                egui::Slider::new(
                    &mut projection.mare_boundary_noise_amplitude_m,
                    0.0..=2_500.0,
                )
                .text("Mare edge noise m"),
            ),
        );
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.regolith_bake_d_min_m, 100.0..=2_000.0)
                    .text("Regolith bake min m"),
            ),
        );
    });
    changed
}

fn draw_cold_desert_projection_controls(
    ui: &mut egui::Ui,
    projection: &mut ColdDesertProjectionConfig,
) -> bool {
    let mut changed = false;
    ui.collapsing("Projection", |ui| {
        changed |= fires(&ui.add(
            egui::Slider::new(&mut projection.relief_scale_m, 0.25..=2.0).text("Relief scale"),
        ));
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.volcanic_dark_strength, 0.0..=2.0)
                    .text("Dark regions"),
            ),
        );
        changed |= fires(&ui.add(
            egui::Slider::new(&mut projection.pale_basin_strength, 0.0..=2.0).text("Pale basins"),
        ));
        changed |=
            fires(&ui.add(
                egui::Slider::new(&mut projection.channel_strength, 0.0..=2.0).text("Channels"),
            ));
        changed |= fires(
            &ui.add(egui::Slider::new(&mut projection.dune_strength, 0.0..=2.0).text("Dunes")),
        );
        changed |= fires(&ui.add(
            egui::Slider::new(&mut projection.base_crater_count, 0..=100_000).text("Base craters"),
        ));
        changed |= fires(
            &ui.add(
                egui::Slider::new(&mut projection.max_crater_radius_m, 5_000.0..=90_000.0)
                    .text("Max crater m"),
            ),
        );
    });
    changed
}

fn draw_projection_controls(ui: &mut egui::Ui, projection: &mut FeatureProjectionConfig) -> bool {
    match projection {
        FeatureProjectionConfig::Auto => {
            ui.label("Projection: Auto");
            false
        }
        FeatureProjectionConfig::AirlessImpact(config) => {
            draw_airless_projection_controls(ui, config)
        }
        FeatureProjectionConfig::ColdDesert(config) => {
            draw_cold_desert_projection_controls(ui, config)
        }
    }
}

fn reroll_authored_seed(
    root_seed: u64,
    id: &FeatureId,
    seed: &mut Option<FeatureSeed>,
    stream: FeatureSeedStream,
) {
    let current = seed.unwrap_or_else(|| FeatureSeed::derive(root_seed, id));
    *seed = Some(current.rerolled(stream, "planet_editor"));
}

fn draw_authored_feature_controls(
    ui: &mut egui::Ui,
    root_seed: u64,
    authored_features: &mut [AuthoredFeatureConfig],
) -> bool {
    let mut changed = false;
    ui.collapsing("Authored Features", |ui| {
        for feature in authored_features {
            match feature {
                AuthoredFeatureConfig::Megabasin(config) => {
                    let id = config.id.clone();
                    ui.horizontal(|ui| {
                        ui.label(id.as_str());
                        if ui.small_button("Shape").clicked() {
                            reroll_authored_seed(
                                root_seed,
                                &id,
                                &mut config.seed,
                                FeatureSeedStream::Shape,
                            );
                            changed = true;
                        }
                        if ui.small_button("Detail").clicked() {
                            reroll_authored_seed(
                                root_seed,
                                &id,
                                &mut config.seed,
                                FeatureSeedStream::Detail,
                            );
                            changed = true;
                        }
                    });
                }
            }
        }
    });
    changed
}

fn draw_feature_manifest(ui: &mut egui::Ui, manifest: &FeatureManifest) {
    ui.collapsing("Feature Manifest", |ui| {
        ui.label(format!("{} features", manifest.features.len()));
        let root_children = manifest
            .get(&manifest.root)
            .map(|root| root.children.clone())
            .unwrap_or_default();
        for child_id in root_children {
            draw_feature_manifest_node(ui, manifest, &child_id);
        }
    });
}

fn draw_feature_manifest_node(ui: &mut egui::Ui, manifest: &FeatureManifest, id: &FeatureId) {
    let Some(feature) = manifest.get(id) else {
        return;
    };
    let label = if feature.scale_range_m.max_m.is_finite() {
        format!(
            "{} · {:?} · {:.1}-{:.1} km",
            feature.id,
            feature.kind,
            feature.scale_range_m.min_m / 1_000.0,
            feature.scale_range_m.max_m / 1_000.0
        )
    } else {
        format!("{} · {:?} · global", feature.id, feature.kind)
    };

    if feature.children.is_empty() {
        ui.label(label);
    } else {
        let children = feature.children.clone();
        ui.collapsing(label, |ui| {
            for child_id in children {
                draw_feature_manifest_node(ui, manifest, &child_id);
            }
        });
    }
}

fn editor_ui(
    mut contexts: bevy_egui::EguiContexts,
    mut planet: ResMut<EditedPlanet>,
    system: Res<SystemData>,
    diagnostics: Res<DiagnosticsStore>,
    status: Res<TerrainGenStatus>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };

    let selected_body_id = system.system.name_to_id.get(&planet.selected_body).copied();
    let mut clicked_body = None;
    let initial_pos = ctx.available_rect().left_top() + egui::vec2(8.0, 8.0);
    egui::Window::new("Celestial bodies")
        .default_pos(initial_pos)
        .resizable(false)
        .show(ctx, |ui| {
            ui.set_min_width(180.0);
            clicked_body = render_body_tree_ui(ui, &system.system, selected_body_id);
        });
    if let Some(body_id) = clicked_body {
        select_body(&mut planet, &system.system, body_id);
    }

    let controls_pos = egui::pos2(
        (ctx.available_rect().right() - 340.0).max(ctx.available_rect().left()),
        ctx.available_rect().top() + 8.0,
    );
    egui::Window::new("Planet Editor")
        .default_pos(controls_pos)
        .show(ctx, |ui| {
            let fps = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(|d| d.smoothed())
                .unwrap_or(0.0);
            ui.label(format!("FPS: {:.0}", fps));
            ui.label(format!("Body: {}", planet.selected_body));
            ui.separator();

            // ---- Terrain gen status ----------------------------------------
            if matches!(planet.mode, BodyMode::Terrain { .. }) {
                match (status.current_started, status.last_duration) {
                    (Some(started), _) => {
                        let elapsed = started.elapsed().as_secs_f32();
                        ui.label(format!("Generating terrain for {:.2}s…", elapsed));
                    }
                    (None, Some(d)) => {
                        ui.label(format!("Last bake: {:.2}s", d.as_secs_f32()));
                    }
                    (None, None) => {}
                }
            }

            ui.separator();

            // ---- Read-only derived info ------------------------------------
            ui.label(format!("Radius: {:.1} km", planet.radius_m / 1000.0));
            ui.label(format!(
                "Heliocentric: {:.3} AU",
                planet.heliocentric_distance_m / AU_M
            ));
            ui.label(format!("Light intensity: {:.2}", planet.light_intensity));

            ui.separator();

            let mut terrain_changed = false;
            let mut uniforms_changed = false;
            let body_name = planet.selected_body.clone();
            let radius_m = planet.radius_m as f32;
            let gravity_m_s2 = planet.gravity_m_s2;
            let axial_tilt_rad = planet.axial_tilt_rad;

            if let BodyMode::Terrain {
                ref mut terrain,
                tidal_axis,
            } = planet.mode
            {
                ui.heading("Parameters");
                ui.label(format!("Terrain: {}", terrain.route_label()));
                match terrain {
                    TerrainConfig::Feature(config) => {
                        ui.horizontal(|ui| {
                            terrain_changed |= fires(
                                &ui.add(egui::Slider::new(&mut config.seed, 0..=9999).text("Seed")),
                            );
                            if ui.button("Reroll World").clicked() {
                                config.seed = sub_seed(config.seed, "planet_editor:world_seed");
                                terrain_changed = true;
                            }
                        });
                        terrain_changed |= draw_projection_controls(ui, &mut config.projection);
                        terrain_changed |= draw_authored_feature_controls(
                            ui,
                            config.seed,
                            &mut config.authored_features,
                        );

                        let compile_context = TerrainCompileContext {
                            body_name: body_name.clone(),
                            radius_m,
                            gravity_m_s2,
                            rotation_hours: None,
                            obliquity_deg: Some(axial_tilt_rad.to_degrees()),
                            tidal_axis,
                            axial_tilt_rad,
                        };
                        let spec = config.to_planet_spec(&compile_context);
                        let plan = plan_initial_compilation(&spec);
                        draw_feature_manifest(ui, &plan.manifest);
                    }
                    TerrainConfig::LegacyPipeline(generator) => {
                        terrain_changed |= fires(
                            &ui.add(egui::Slider::new(&mut generator.seed, 0..=9999).text("Seed")),
                        );
                    }
                    TerrainConfig::None => {}
                }
                ui.separator();
            }

            ui.heading("Shading");
            if planet.atmosphere.is_some() {
                uniforms_changed |= ui
                    .checkbox(&mut planet.atmosphere_enabled, "Atmosphere")
                    .changed();
            }
            uniforms_changed |= ui
                .checkbox(&mut planet.full_bright, "Full bright")
                .changed();
            uniforms_changed |= ui
                .checkbox(&mut planet.ambient_light, "Ambient light")
                .changed();
            uniforms_changed |= fires(
                &ui.add(
                    bevy_egui::egui::Slider::new(
                        &mut planet.sun_azimuth,
                        -std::f32::consts::PI..=std::f32::consts::PI,
                    )
                    .text("Sun azimuth"),
                ),
            );

            if terrain_changed {
                planet.terrain_dirty = true;
            }
            if uniforms_changed {
                planet.uniforms_dirty = true;
            }
        });
}

/// Applies shader-uniform-only changes to the current material.
#[allow(clippy::too_many_arguments)]
fn apply_uniform_changes(
    mut planet: ResMut<EditedPlanet>,
    terrain_q: Query<&PlanetMaterialHandle, With<PreviewPlanet>>,
    halo_q: Query<&PlanetHaloMaterialHandle, With<PreviewPlanet>>,
    gas_q: Query<&GasGiantMaterialHandle, With<PreviewPlanet>>,
    ring_q: Query<&RingMaterialHandle, With<PreviewRing>>,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut planet_halo_materials: ResMut<Assets<PlanetHaloMaterial>>,
    mut gas_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
) {
    if !planet.uniforms_dirty {
        return;
    }
    planet.uniforms_dirty = false;

    let (_, _, wrap) = lighting_for(&planet);
    let scene = scene_lighting_for(&planet);
    let atmosphere = active_atmosphere(&planet);

    match &planet.mode {
        BodyMode::Terrain { .. } => {
            for handle in &terrain_q {
                let Some(mat) = planet_materials.get_mut(&handle.0) else {
                    continue;
                };
                mat.params.terminator_wrap = wrap;
                mat.params.fullbright = if planet.full_bright { 1.0 } else { 0.0 };
                mat.params.scene = scene.clone();
                mat.atmosphere = atmosphere;
            }
            for handle in &halo_q {
                let Some(mat) = planet_halo_materials.get_mut(&handle.0) else {
                    continue;
                };
                mat.params.terminator_wrap = wrap;
                mat.params.fullbright = if planet.full_bright { 1.0 } else { 0.0 };
                mat.params.scene = scene.clone();
                mat.atmosphere = atmosphere;
            }
        }
        BodyMode::GasGiant { .. } => {
            for handle in &gas_q {
                let Some(mat) = gas_materials.get_mut(&handle.0) else {
                    continue;
                };
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

fn patch_preview_reference_cloud_cover(
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

fn write_cloud_animation(
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
fn update_preview_atmosphere(
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

fn dispatch_rebake(
    mut commands: Commands,
    mut planet: ResMut<EditedPlanet>,
    mut status: ResMut<TerrainGenStatus>,
    preview_q: Query<(Entity, &Children), With<PreviewPlanet>>,
) {
    if !planet.terrain_dirty {
        return;
    }
    let BodyMode::Terrain {
        ref terrain,
        tidal_axis,
    } = planet.mode
    else {
        planet.terrain_dirty = false;
        return;
    };
    let Ok((entity, children)) = preview_q.single() else {
        return;
    };
    let Some(mesh_entity) = children.iter().next() else {
        return;
    };
    let terrain = terrain.clone();
    let radius_m = planet.radius_m;
    let gravity_m_s2 = planet.gravity_m_s2;
    let axial_tilt_rad = planet.axial_tilt_rad;
    planet.terrain_dirty = false;

    let task = dispatch_terrain_bake(
        &terrain,
        radius_m,
        gravity_m_s2,
        tidal_axis,
        axial_tilt_rad,
        planet.selected_body.clone(),
    );
    status.current_started = Some(Instant::now());
    commands
        .entity(entity)
        .insert(PendingTerrainGen { task, mesh_entity });
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let body_arg = std::env::args().nth(1);
    let preferred = body_arg.as_deref().unwrap_or(DEFAULT_BODY_NAME);

    let system = load_solar_system(SOLAR_SYSTEM_RON).expect("parse solar_system.ron");

    let find_body = |name: &str| {
        system
            .bodies
            .iter()
            .find(|b| b.name.eq_ignore_ascii_case(name))
    };
    let body = find_body(preferred)
        .or_else(|| find_body(DEFAULT_BODY_NAME))
        .unwrap_or(&system.bodies[0]);
    let selected_body = body.name.clone();
    let resolved = build_params_for_body(&system, body);

    let light_intensity = light_intensity_at(resolved.heliocentric_distance_m);

    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(SystemData { system })
        .insert_resource(EditedPlanet {
            selected_body,
            radius_m: resolved.radius_m,
            gravity_m_s2: resolved.gravity_m_s2,
            axial_tilt_rad: resolved.axial_tilt_rad,
            mode: resolved.mode,
            rings: resolved.rings,
            atmosphere: resolved.atmosphere,
            atmosphere_enabled: true,
            heliocentric_distance_m: resolved.heliocentric_distance_m,
            light_intensity,
            sun_azimuth: 0.0,
            sun_orbital_elevation: resolved.sun_orbital_elevation,
            full_bright: false,
            ambient_light: false,
            terrain_dirty: false,
            uniforms_dirty: false,
            body_changed: false,
        })
        .init_resource::<OrbitCamera>()
        .init_resource::<TerrainGenStatus>()
        .init_resource::<PreviewAtmosphereClock>()
        .init_resource::<ReferenceClouds>()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Thalos — Planet Editor".into(),
                        present_mode: PresentMode::AutoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: "../../assets".to_string(),
                    ..default()
                }),
        )
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(bevy_egui::EguiPlugin::default())
        .add_plugins(PlanetRenderingPlugin)
        .add_plugins(SkyBackdropPlugin)
        .add_systems(
            Startup,
            (
                load_reference_cloud_sources,
                spawn_camera,
                spawn_preview_planet,
            ),
        )
        .add_systems(bevy_egui::EguiPrimaryContextPass, editor_ui)
        .add_systems(
            Update,
            (
                convert_reference_clouds_when_ready,
                camera_input,
                camera_zoom_smoothing.after(camera_input),
                camera_apply_transform.after(camera_zoom_smoothing),
                apply_uniform_changes,
                handle_body_switch,
                dispatch_rebake.after(handle_body_switch),
                finalize_terrain_bake.after(dispatch_rebake),
                patch_preview_reference_cloud_cover
                    .after(convert_reference_clouds_when_ready)
                    .after(finalize_terrain_bake),
                update_preview_atmosphere
                    .after(apply_uniform_changes)
                    .after(finalize_terrain_bake),
            ),
        )
        .run();
}
