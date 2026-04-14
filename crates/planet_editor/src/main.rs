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
use thalos_terrain_gen::{BodyBuilder, BodyData, GeneratorParams, Pipeline};
use thalos_planet_rendering::{
    PlanetDetailParams, PlanetMaterial, PlanetMaterialHandle, PlanetParams,
    PlanetRenderingPlugin, bake_from_body_data,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LIGHT_AT_1AU: f32 = 10.0;
const AMBIENT_INTENSITY: f32 = 0.05;
/// Fullbright wrap value stuffed into `light_dir.w`. Shader computes
/// `wrap = light_dir.w * 0.08`, so 1250 → wrap ≈ 100 → diffuse ≈ 0.99 every
/// where on the sphere. Surge term still adds ≤40% on the sun-facing cap.
const FULLBRIGHT_WRAP: f32 = 1250.0;
/// Fullbright light intensity. Picks a value such that, with wrap ≈ 100,
/// `lit = albedo * light / PI ≈ albedo` on the night side. Bright side runs
/// ~40% hotter from surge — fine for debug readout.
const FULLBRIGHT_LIGHT: f32 = std::f32::consts::PI;
const AU_M: f64 = 1.496e11;
const DEFAULT_BODY_NAME: &str = "Mira";

const SOLAR_SYSTEM_RON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/solar_system.ron"));

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
    /// Direction toward the parent body in the body's local frame. Used as
    /// the tidal axis for tidally-locked moons. Currently fixed to +Z (the
    /// body's local frame is unauthored — the parent direction is just a
    /// per-body convention until we author it from orbital state).
    tidal_axis: Option<Vec3>,
    axial_tilt_rad: f32,
    /// The full generator block for the body. Cloned from the loaded
    /// `SolarSystemDefinition` and rebuilt into a Pipeline on every bake.
    generator: GeneratorParams,
    heliocentric_distance_m: f64,
    light_intensity: f32,
    terminator_wrap: f32,
    sun_azimuth: f32,
    sun_elevation: f32,
    full_bright: bool,
    /// Terrain pipeline inputs changed — requires an async rebake.
    terrain_dirty: bool,
    /// Only shader uniforms changed — cheap, applied in place.
    uniforms_dirty: bool,
}

/// Tracks the running / last-completed terrain generation for the UI.
#[derive(Resource, Default)]
struct TerrainGenStatus {
    current_started: Option<Instant>,
    last_duration: Option<Duration>,
}

/// Billboard mesh used by the preview planet, created once at startup.
#[derive(Resource)]
struct BillboardMesh(Handle<Mesh>);

/// In-flight terrain generation task attached to the preview planet entity.
/// Re-baking just inserts a fresh `PendingTerrainGen`; the dropped component's
/// `Task` is cancelled automatically.
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

/// Resolved per-body editor inputs: physical/orbital info plus the cloned
/// generator block. Bodies without a `generator` block in the file can't be
/// previewed yet — the caller is expected to filter those out of the picker.
struct ResolvedBody {
    radius_m: f64,
    tidal_axis: Option<Vec3>,
    axial_tilt_rad: f32,
    generator: GeneratorParams,
    heliocentric_distance_m: f64,
}

fn build_params_for_body(
    system: &SolarSystemDefinition,
    body: &thalos_physics::types::BodyDefinition,
) -> Option<ResolvedBody> {
    let generator = body.generator.as_ref()?.clone();
    Some(ResolvedBody {
        radius_m: body.radius_m,
        // For now, only moons get a tidal axis, and we treat it as +Z in the
        // body's local frame. Authored axes per body would replace this.
        tidal_axis: matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z),
        axial_tilt_rad: body.axial_tilt_rad as f32,
        generator,
        heliocentric_distance_m: heliocentric_sma(system, body),
    })
}

fn heliocentric_sma(system: &SolarSystemDefinition, start: &thalos_physics::types::BodyDefinition) -> f64 {
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

/// Resolve the `(light, ambient)` pair fed to `PlanetParams`. Full bright
/// collapses the directional contribution into ambient so the surface reads
/// evenly without sculpting shadows.
/// Returns `(light_intensity, ambient_intensity, wrap)`. Fullbright abuses the
/// shader's terminator-wrap term: `diffuse = (n·l + wrap)/(1+wrap)` → ~1.0
/// everywhere when `wrap` is huge, so the whole sphere reads as fully lit.
fn lighting_for(planet: &EditedPlanet) -> (f32, f32, f32) {
    if planet.full_bright {
        (FULLBRIGHT_LIGHT, AMBIENT_INTENSITY, FULLBRIGHT_WRAP)
    } else {
        (planet.light_intensity, AMBIENT_INTENSITY, planet.terminator_wrap)
    }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Component)]
struct PreviewPlanet;

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
        Self::from_render_radius(1.5)
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
    if egui_ctx.ctx_mut().is_ok_and(|ctx| ctx.wants_pointer_input()) {
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
        // Log-space zoom on *height above surface*, so near-surface steps
        // shrink to near-zero while far-away steps stay responsive.
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
    let Ok(mut transform) = query.single_mut() else { return };
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
// Planet preview
// ---------------------------------------------------------------------------

/// Dev-mode crater-count scale factor. Cratering + space_weather together
/// dominate the Mira bake (~95% of wall time at authored 500k craters), and
/// both scale linearly with total crater count. Cutting the count by 10×
/// in dev brings the editor bake from ~190 s down to ~20 s with a uniform
/// quality drop rather than visual artifacts.
#[cfg(debug_assertions)]
const DEV_CRATER_SCALE: f32 = 0.1;
#[cfg(not(debug_assertions))]
const DEV_CRATER_SCALE: f32 = 1.0;

/// Dispatches the terrain pipeline for `planet` onto the async compute pool.
fn dispatch_terrain_bake(planet: &EditedPlanet) -> Task<BodyData> {
    let radius_m = planet.radius_m as f32;
    let tidal_axis = planet.tidal_axis;
    let axial_tilt_rad = planet.axial_tilt_rad;
    let mut g = planet.generator.clone();
    g.scale_crater_count(DEV_CRATER_SCALE);
    AsyncComputeTaskPool::get().spawn(async move {
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
        builder.build()
    })
}

fn spawn_preview_planet(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut status: ResMut<TerrainGenStatus>,
    planet: Res<EditedPlanet>,
) {
    let billboard_mesh = meshes.add(Rectangle::new(2.0, 2.0));
    commands.insert_resource(BillboardMesh(billboard_mesh));

    // Placeholder sphere shown until the first async bake completes.
    let placeholder_mesh = meshes.add(Sphere::new(1.5).mesh().ico(4).unwrap());
    let placeholder_mat = std_materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.4, 0.45),
        perceptual_roughness: 0.9,
        metallic: 0.0,
        ..default()
    });

    let parent = commands
        .spawn((
            Transform::default(),
            Visibility::Inherited,
            PreviewPlanet,
            Name::new("Preview Planet"),
        ))
        .id();

    let mesh_entity = commands
        .spawn((
            Mesh3d(placeholder_mesh),
            MeshMaterial3d(placeholder_mat),
            ChildOf(parent),
        ))
        .id();

    let task = dispatch_terrain_bake(&planet);
    status.current_started = Some(Instant::now());
    commands
        .entity(parent)
        .insert(PendingTerrainGen { task, mesh_entity });
}

/// Poll in-flight terrain tasks. When one finishes, bake GPU textures, build
/// a fresh `PlanetMaterial`, and swap it onto the mesh entity.
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
        let dir = sun_direction(planet.sun_azimuth, planet.sun_elevation);
        let (light_intensity, ambient_intensity, wrap) = lighting_for(&planet);

        let mat_handle = planet_materials.add(PlanetMaterial {
            params: PlanetParams {
                radius: 1.5,
                light_intensity,
                ambient_intensity,
                height_range,
                light_dir: Vec4::new(dir.x, dir.y, dir.z, wrap),
                orientation: Vec4::new(0.0, 0.0, 0.0, 1.0),
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
        // Only bodies with a generator block are previewable.
        let mut new_body: Option<String> = None;
        bevy_egui::egui::ComboBox::from_label("Body")
            .selected_text(&planet.selected_body)
            .show_ui(ui, |ui| {
                for b in &system.system.bodies {
                    if b.generator.is_none() {
                        continue;
                    }
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
            && let Some(resolved) = build_params_for_body(&system.system, &system.system.bodies[id])
        {
            planet.radius_m = resolved.radius_m;
            planet.tidal_axis = resolved.tidal_axis;
            planet.axial_tilt_rad = resolved.axial_tilt_rad;
            planet.generator = resolved.generator;
            planet.heliocentric_distance_m = resolved.heliocentric_distance_m;
            planet.light_intensity = light_intensity_at(resolved.heliocentric_distance_m);
            planet.selected_body = name;
            planet.terrain_dirty = true;
            planet.uniforms_dirty = true;
        }

        // ---- Terrain gen status ----------------------------------------
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
        ui.heading("Parameters");
        let mut terrain_changed = false;
        let mut uniforms_changed = false;
        terrain_changed |= fires(&ui.add(
            bevy_egui::egui::Slider::new(&mut planet.generator.seed, 0..=9999).text("Seed"),
        ));

        ui.separator();
        ui.heading("Shading");
        uniforms_changed |= ui
            .checkbox(&mut planet.full_bright, "Full bright")
            .changed();
        uniforms_changed |= fires(&ui.add(
            bevy_egui::egui::Slider::new(&mut planet.terminator_wrap, 0.0..=1.0)
                .text("Terminator wrap"),
        ));
        uniforms_changed |= fires(&ui.add(
            bevy_egui::egui::Slider::new(
                &mut planet.sun_azimuth,
                -std::f32::consts::PI..=std::f32::consts::PI,
            )
            .text("Sun azimuth"),
        ));
        uniforms_changed |= fires(&ui.add(
            bevy_egui::egui::Slider::new(
                &mut planet.sun_elevation,
                -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
            )
            .text("Sun elevation"),
        ));

        if terrain_changed {
            planet.terrain_dirty = true;
        }
        if uniforms_changed {
            planet.uniforms_dirty = true;
        }
    });
}

/// Applies shader-uniform-only changes (lighting, terminator wrap) to the
/// current `PlanetMaterial` in place. Cheap — no rebake required.
fn apply_uniform_changes(
    mut planet: ResMut<EditedPlanet>,
    query: Query<&PlanetMaterialHandle, With<PreviewPlanet>>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
) {
    if !planet.uniforms_dirty {
        return;
    }
    planet.uniforms_dirty = false;

    let (light_intensity, ambient_intensity, wrap) = lighting_for(&planet);
    for handle in &query {
        let Some(mat) = materials.get_mut(&handle.0) else { continue };
        mat.params.light_intensity = light_intensity;
        mat.params.ambient_intensity = ambient_intensity;
        let dir = sun_direction(planet.sun_azimuth, planet.sun_elevation);
        mat.params.light_dir = Vec4::new(dir.x, dir.y, dir.z, wrap);
    }
}

/// Dispatches a new terrain bake task whenever terrain inputs change. The
/// currently-displayed material stays visible until the task finishes.
fn dispatch_rebake(
    mut commands: Commands,
    mut planet: ResMut<EditedPlanet>,
    mut status: ResMut<TerrainGenStatus>,
    preview_q: Query<(Entity, &Children), With<PreviewPlanet>>,
) {
    if !planet.terrain_dirty {
        return;
    }
    let Ok((entity, children)) = preview_q.single() else { return };
    let Some(mesh_entity) = children.iter().next() else { return };
    planet.terrain_dirty = false;

    let task = dispatch_terrain_bake(&planet);
    status.current_started = Some(Instant::now());
    commands
        .entity(entity)
        .insert(PendingTerrainGen { task, mesh_entity });
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let system = load_solar_system(SOLAR_SYSTEM_RON).expect("parse solar_system.ron");

    // Pick a body that has a generator block. Prefer Mira; fall back to the
    // first body with a pipeline.
    let (selected_body, resolved) = system
        .name_to_id
        .get(DEFAULT_BODY_NAME)
        .copied()
        .and_then(|id| {
            build_params_for_body(&system, &system.bodies[id])
                .map(|r| (system.bodies[id].name.clone(), r))
        })
        .or_else(|| {
            system.bodies.iter().find_map(|b| {
                build_params_for_body(&system, b).map(|r| (b.name.clone(), r))
            })
        })
        .expect("no body in solar_system.ron has a generator block");

    let light_intensity = light_intensity_at(resolved.heliocentric_distance_m);

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.02, 0.01, 0.04)))
        .insert_resource(SystemData { system })
        .insert_resource(EditedPlanet {
            selected_body,
            radius_m: resolved.radius_m,
            tidal_axis: resolved.tidal_axis,
            axial_tilt_rad: resolved.axial_tilt_rad,
            generator: resolved.generator,
            heliocentric_distance_m: resolved.heliocentric_distance_m,
            light_intensity,
            terminator_wrap: 0.2,
            sun_azimuth: 0.0,
            sun_elevation: 0.0,
            full_bright: false,
            terrain_dirty: false,
            uniforms_dirty: false,
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
                dispatch_rebake,
                finalize_terrain_bake.after(dispatch_rebake),
            ),
        )
        .run();
}
