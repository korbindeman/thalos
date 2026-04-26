use std::time::{Duration, Instant};

use bevy::asset::AssetPlugin;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use bevy::window::PresentMode;
use thalos_physics::parsing::load_solar_system;
use thalos_physics::types::{BodyKind, SolarSystemDefinition};
use thalos_planet_rendering::{
    GasGiantLayers, GasGiantMaterial, GasGiantMaterialHandle, GasGiantParams,
    PlanetDetailParams, PlanetMaterial, PlanetMaterialHandle, PlanetParams, PlanetRenderingPlugin,
    RingLayers, RingMaterial, RingMaterialHandle, RingParams,
    SceneLighting, StarLight, bake_from_body_data, blank_cloud_cover_image, build_ring_mesh,
};
use thalos_terrain_gen::{BodyBuilder, BodyData, GeneratorParams, Pipeline};

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
        generator: GeneratorParams,
        tidal_axis: Option<Vec3>,
    },
    GasGiant {
        layers: Box<GasGiantLayers>,
        has_rings: bool,
        ring_inner_m: f32,
        ring_outer_m: f32,
        ring_layers: Option<Box<RingLayers>>,
    },
    Star,
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
    axial_tilt_rad: f32,
    mode: BodyMode,
    heliocentric_distance_m: f64,
    light_intensity: f32,
    terminator_wrap: f32,
    sun_azimuth: f32,
    sun_elevation: f32,
    full_bright: bool,
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
    axial_tilt_rad: f32,
    mode: BodyMode,
    heliocentric_distance_m: f64,
}

fn build_params_for_body(
    system: &SolarSystemDefinition,
    body: &thalos_physics::types::BodyDefinition,
) -> ResolvedBody {
    let mode = if body.kind == BodyKind::Star {
        BodyMode::Star
    } else if let Some(atmos) = &body.atmosphere {
        let layers = Box::new(GasGiantLayers::from_params(atmos, body.radius_m as f32 / RENDER_RADIUS));
        let (has_rings, ring_inner_m, ring_outer_m, ring_layers) =
            if let Some(rings) = &atmos.rings {
                (
                    true,
                    rings.inner_radius_m,
                    rings.outer_radius_m,
                    Some(Box::new(RingLayers::from_system(rings))),
                )
            } else {
                (false, 0.0, 0.0, None)
            };
        BodyMode::GasGiant {
            layers,
            has_rings,
            ring_inner_m,
            ring_outer_m,
            ring_layers,
        }
    } else if let Some(g) = &body.generator {
        BodyMode::Terrain {
            generator: g.clone(),
            tidal_axis: matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z),
        }
    } else {
        BodyMode::Terrain {
            generator: GeneratorParams {
                seed: 0,
                composition: thalos_terrain_gen::Composition::new(1.0, 0.0, 0.0, 0.0, 0.0),
                cubemap_resolution: 64,
                body_age_gyr: 4.5,
                pipeline: Vec::new(),
            },
            tidal_axis: matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z),
        }
    };

    ResolvedBody {
        radius_m: body.radius_m,
        axial_tilt_rad: body.axial_tilt_rad as f32,
        mode,
        heliocentric_distance_m: heliocentric_sma(system, body),
    }
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

fn lighting_for(planet: &EditedPlanet) -> (f32, f32, f32) {
    (
        planet.light_intensity,
        AMBIENT_INTENSITY,
        planet.terminator_wrap,
    )
}

/// Build a `SceneLighting` for the preview. Single star, no eclipse
/// occluders, no planetshine — editor scenes are one body at a time.
fn scene_lighting_for(planet: &EditedPlanet) -> SceneLighting {
    let (light_intensity, ambient_intensity, _wrap) = lighting_for(planet);
    let dir = sun_direction(planet.sun_azimuth, planet.sun_elevation);
    let mut scene = SceneLighting::default();
    scene.ambient_intensity = ambient_intensity;
    scene.star_count = 1;
    scene.stars[0] = StarLight {
        dir_flux: Vec4::new(dir.x, dir.y, dir.z, light_intensity),
        color: Vec4::new(1.0, 1.0, 1.0, 0.0),
    };
    scene
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Component)]
struct PreviewPlanet;

#[derive(Component)]
struct PreviewRing;

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
    generator: &GeneratorParams,
    radius_m: f64,
    tidal_axis: Option<Vec3>,
    axial_tilt_rad: f32,
    body_name: String,
) -> Task<BodyData> {
    let radius_m = radius_m as f32;
    let mut g = generator.clone();
    g.scale_crater_count(DEV_CRATER_SCALE);
    AsyncComputeTaskPool::get().spawn(async move {
        let cache_dir = terrain_cache_dir();
        let key = thalos_terrain_gen::cache::cache_key(&g, radius_m, tidal_axis, axial_tilt_rad);
        let path = thalos_terrain_gen::cache::cache_path(&cache_dir, &body_name, key);
        if let Some(data) = thalos_terrain_gen::cache::load(&path, key) {
            info!("terrain cache hit: {body_name}");
            return data;
        }
        info!("terrain cache miss, baking: {body_name}");
        let mut builder = BodyBuilder::new(
            radius_m,
            g.seed,
            g.composition,
            g.cubemap_resolution,
            g.body_age_gyr,
            tidal_axis,
            axial_tilt_rad,
        );
        let stages = g
            .pipeline
            .into_iter()
            .map(|s| s.into_stage())
            .collect::<Vec<_>>();
        Pipeline::new(stages).run(&mut builder);
        let data = builder.build();
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
            generator,
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
                generator,
                planet.radius_m,
                *tidal_axis,
                planet.axial_tilt_rad,
                planet.selected_body.clone(),
            );
            status.current_started = Some(Instant::now());
            commands
                .entity(parent)
                .insert(PendingTerrainGen { task, mesh_entity });
        }
        BodyMode::GasGiant {
            layers,
            has_rings,
            ring_inner_m,
            ring_outer_m,
            ring_layers,
        } => {
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

            if *has_rings
                && let Some(rl) = ring_layers
            {
                let meters_per_ru = planet.radius_m as f32 / RENDER_RADIUS;
                let inner_ru = *ring_inner_m / meters_per_ru;
                let outer_ru = *ring_outer_m / meters_per_ru;
                let ring_mesh = meshes.add(build_ring_mesh(inner_ru, outer_ru, 128));

                let ring_mat = ring_materials.add(RingMaterial {
                    params: RingParams {
                        planet_center_radius: Vec4::new(0.0, 0.0, 0.0, RENDER_RADIUS),
                        inner_radius: inner_ru,
                        outer_radius: outer_ru,
                        scene: scene.clone(),
                        ..default()
                    },
                    layers: *rl.clone(),
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
        BodyMode::Star => {
            let star_mesh = meshes.add(Sphere::new(RENDER_RADIUS).mesh().ico(5).unwrap());
            let star_mat = std_materials.add(StandardMaterial {
                base_color: Color::BLACK,
                emissive: LinearRgba::new(1.0, 0.95, 0.8, 1.0) * 5000.0,
                ..default()
            });
            commands.spawn((
                Mesh3d(star_mesh),
                MeshMaterial3d(star_mat),
                ChildOf(parent),
            ));
        }
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

    let billboard = BillboardMesh(
        meshes.add(Rectangle::new(
            RENDER_RADIUS * 2.0 + 2.0,
            RENDER_RADIUS * 2.0 + 2.0,
        )),
    );

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
    mut images: ResMut<Assets<Image>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut status: ResMut<TerrainGenStatus>,
    billboard: Res<BillboardMesh>,
    planet: Res<EditedPlanet>,
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
        let coastline_seed = (body_seed as u32)
            ^ ((body_seed >> 32) as u32)
            ^ 0xC0A5_71_1Eu32;
        let has_ocean = body.sea_level_m.is_some();
        let coastline_warp_amp_radians = if has_ocean { 8.0e-4 } else { 0.0 };
        let coastline_jitter_amp_m = if has_ocean { 30.0 } else { 0.0 };

        let mat_handle = planet_materials.add(PlanetMaterial {
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
            material_cube: textures.material_cube,
            craters: textures.craters,
            cell_index: textures.cell_index,
            feature_ids: textures.feature_ids,
            materials: textures.materials,
            atmosphere: default(),
            cloud_cover: blank_cloud_cover_image(&mut images),
        });

        let mesh_entity = pending.mesh_entity;
        commands
            .entity(mesh_entity)
            .insert((
                Mesh3d(billboard.0.clone()),
                MeshMaterial3d(mat_handle.clone()),
            ))
            .remove::<MeshMaterial3d<StandardMaterial>>();

        commands
            .entity(entity)
            .insert(PlanetMaterialHandle(mat_handle))
            .remove::<PendingTerrainGen>();

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

fn editor_ui(
    mut contexts: bevy_egui::EguiContexts,
    mut planet: ResMut<EditedPlanet>,
    system: Res<SystemData>,
    diagnostics: Res<DiagnosticsStore>,
    status: Res<TerrainGenStatus>,
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    bevy_egui::egui::Window::new("Planet Editor").show(ctx, |ui| {
        let fps = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|d| d.smoothed())
            .unwrap_or(0.0);
        ui.label(format!("FPS: {:.0}", fps));
        ui.separator();

        // ---- Body picker ------------------------------------------------
        let mut new_body: Option<String> = None;
        bevy_egui::egui::ComboBox::from_label("Body")
            .selected_text(&planet.selected_body)
            .show_ui(ui, |ui| {
                for b in &system.system.bodies {
                    if ui
                        .selectable_label(b.name == planet.selected_body, &b.name)
                        .clicked()
                    {
                        new_body = Some(b.name.clone());
                    }
                }
            });
        if let Some(name) = new_body
            && let Some(&id) = system.system.name_to_id.get(&name)
        {
            let resolved = build_params_for_body(&system.system, &system.system.bodies[id]);
            planet.radius_m = resolved.radius_m;
            planet.axial_tilt_rad = resolved.axial_tilt_rad;
            planet.mode = resolved.mode;
            planet.heliocentric_distance_m = resolved.heliocentric_distance_m;
            planet.light_intensity = light_intensity_at(resolved.heliocentric_distance_m);
            planet.selected_body = name;
            planet.body_changed = true;
            planet.uniforms_dirty = true;
        }

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

        // ---- Editable fields --------------------------------
        fn fires(r: &bevy_egui::egui::Response) -> bool {
            r.drag_stopped() || (r.changed() && !r.dragged())
        }

        let mut terrain_changed = false;
        let mut uniforms_changed = false;

        if let BodyMode::Terrain { ref mut generator, .. } = planet.mode {
            ui.heading("Parameters");
            terrain_changed |=
                fires(&ui.add(
                    bevy_egui::egui::Slider::new(&mut generator.seed, 0..=9999).text("Seed"),
                ));
            ui.separator();
        }

        ui.heading("Shading");
        uniforms_changed |= ui
            .checkbox(&mut planet.full_bright, "Full bright")
            .changed();
        uniforms_changed |= fires(
            &ui.add(
                bevy_egui::egui::Slider::new(&mut planet.terminator_wrap, 0.0..=1.0)
                    .text("Terminator wrap"),
            ),
        );
        uniforms_changed |= fires(
            &ui.add(
                bevy_egui::egui::Slider::new(
                    &mut planet.sun_azimuth,
                    -std::f32::consts::PI..=std::f32::consts::PI,
                )
                .text("Sun azimuth"),
            ),
        );
        uniforms_changed |= fires(
            &ui.add(
                bevy_egui::egui::Slider::new(
                    &mut planet.sun_elevation,
                    -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
                )
                .text("Sun elevation"),
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
fn apply_uniform_changes(
    mut planet: ResMut<EditedPlanet>,
    terrain_q: Query<&PlanetMaterialHandle, With<PreviewPlanet>>,
    gas_q: Query<&GasGiantMaterialHandle, With<PreviewPlanet>>,
    ring_q: Query<&RingMaterialHandle, With<PreviewRing>>,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut gas_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
) {
    if !planet.uniforms_dirty {
        return;
    }
    planet.uniforms_dirty = false;

    let (_, _, wrap) = lighting_for(&planet);
    let scene = scene_lighting_for(&planet);

    match &planet.mode {
        BodyMode::Terrain { .. } => {
            for handle in &terrain_q {
                let Some(mat) = planet_materials.get_mut(&handle.0) else {
                    continue;
                };
                mat.params.terminator_wrap = wrap;
                mat.params.fullbright = if planet.full_bright { 1.0 } else { 0.0 };
                mat.params.scene = scene.clone();
            }
        }
        BodyMode::GasGiant { .. } => {
            for handle in &gas_q {
                let Some(mat) = gas_materials.get_mut(&handle.0) else {
                    continue;
                };
                mat.params.scene = scene.clone();
            }
            for handle in &ring_q {
                let Some(mat) = ring_materials.get_mut(&handle.0) else {
                    continue;
                };
                mat.params.scene = scene.clone();
            }
        }
        BodyMode::Star => {}
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
        ref generator,
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
    let generator = generator.clone();
    let radius_m = planet.radius_m;
    let axial_tilt_rad = planet.axial_tilt_rad;
    planet.terrain_dirty = false;

    let task = dispatch_terrain_bake(
        &generator,
        radius_m,
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
        .insert_resource(ClearColor(Color::srgb(0.02, 0.01, 0.04)))
        .insert_resource(SystemData { system })
        .insert_resource(EditedPlanet {
            selected_body,
            radius_m: resolved.radius_m,
            axial_tilt_rad: resolved.axial_tilt_rad,
            mode: resolved.mode,
            heliocentric_distance_m: resolved.heliocentric_distance_m,
            light_intensity,
            terminator_wrap: 0.2,
            sun_azimuth: 0.0,
            sun_elevation: 0.0,
            full_bright: false,
            terrain_dirty: false,
            uniforms_dirty: false,
            body_changed: false,
        })
        .init_resource::<OrbitCamera>()
        .init_resource::<TerrainGenStatus>()
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
        .add_systems(Startup, (spawn_camera, spawn_preview_planet))
        .add_systems(bevy_egui::EguiPrimaryContextPass, editor_ui)
        .add_systems(
            Update,
            (
                camera_input,
                camera_zoom_smoothing.after(camera_input),
                camera_apply_transform.after(camera_zoom_smoothing),
                apply_uniform_changes,
                handle_body_switch,
                dispatch_rebake.after(handle_body_switch),
                finalize_terrain_bake.after(dispatch_rebake),
            ),
        )
        .run();
}
