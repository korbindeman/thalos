use bevy::asset::AssetPlugin;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::PresentMode;
use thalos_physics::parsing::load_solar_system;
use thalos_physics::types::{BodyKind, SolarSystemDefinition};
use thalos_terrain_gen::{BodyBuilder, GeneratorParams, Pipeline};
use bevy::render::storage::ShaderStorageBuffer;
use thalos_planet_rendering::{
    PlanetDetailParams, PlanetMaterial, PlanetMaterialHandle, PlanetParams,
    PlanetRenderingPlugin, PlanetTextures, bake_from_body_data,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LIGHT_AT_1AU: f32 = 12.0;
const AMBIENT_INTENSITY: f32 = 0.05;
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
    /// The full generator block for the body. Cloned from the loaded
    /// `SolarSystemDefinition` and rebuilt into a Pipeline on every bake.
    generator: GeneratorParams,
    heliocentric_distance_m: f64,
    light_intensity: f32,
    terminator_wrap: f32,
    sun_azimuth: f32,
    sun_elevation: f32,
    dirty: bool,
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
    #[allow(dead_code)]
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
    const ZOOM_SENSITIVITY: f32 = 0.1;

    if mouse.pressed(MouseButton::Left) {
        let delta = motion.delta;
        orbit.azimuth += delta.x * ROTATE_SENSITIVITY;
        orbit.elevation = (orbit.elevation - delta.y * ROTATE_SENSITIVITY)
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    if scroll.delta.y != 0.0 {
        let log_dist = orbit.target_distance.ln() - scroll.delta.y * ZOOM_SENSITIVITY;
        orbit.target_distance = log_dist.exp().clamp(orbit.min_distance, orbit.max_distance);
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

struct BakeResult {
    textures: PlanetTextures,
    height_range: f32,
    detail: PlanetDetailParams,
}

fn bake_preview(
    planet: &EditedPlanet,
    images: &mut Assets<Image>,
    storage_buffers: &mut Assets<ShaderStorageBuffer>,
) -> BakeResult {
    let g = &planet.generator;

    let mut builder = BodyBuilder::new(
        planet.radius_m as f32,
        g.seed,
        g.composition,
        g.cubemap_resolution,
        g.body_age_gyr,
        planet.tidal_axis,
    );

    let stages = g
        .pipeline
        .iter()
        .cloned()
        .map(|s| s.into_stage())
        .collect::<Vec<_>>();
    let pipeline = Pipeline::new(stages);
    pipeline.run(&mut builder);

    let body = builder.build();
    let detail = PlanetDetailParams::from_body(&body.detail_params, body.cubemap_bake_threshold_m);
    let height_range = body.height_range;
    let textures = bake_from_body_data(&body, images, storage_buffers);

    BakeResult { textures, height_range, detail }
}

fn spawn_preview_planet(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
    planet: Res<EditedPlanet>,
) {
    let billboard_mesh = meshes.add(Rectangle::new(2.0, 2.0));

    let bake = bake_preview(&planet, &mut images, &mut storage_buffers);

    let dir = sun_direction(planet.sun_azimuth, planet.sun_elevation);
    let mat_handle = planet_materials.add(PlanetMaterial {
        params: PlanetParams {
            radius: 1.5,
            rotation_phase: 0.0,
            light_intensity: planet.light_intensity,
            ambient_intensity: AMBIENT_INTENSITY,
            light_dir: Vec4::new(dir.x, dir.y, dir.z, planet.terminator_wrap),
            height_range: bake.height_range,
        },
        albedo: bake.textures.albedo,
        height: bake.textures.height,
        detail: bake.detail,
        material_cube: bake.textures.material_cube,
        craters: bake.textures.craters,
        cell_index: bake.textures.cell_index,
        feature_ids: bake.textures.feature_ids,
        materials: bake.textures.materials,
    });

    commands
        .spawn((
            Transform::default(),
            Visibility::Inherited,
            PlanetMaterialHandle(mat_handle.clone()),
            PreviewPlanet,
            Name::new("Preview Planet"),
        ))
        .with_children(|parent| {
            parent.spawn((
                Mesh3d(billboard_mesh),
                MeshMaterial3d(mat_handle),
            ));
        });
}

// ---------------------------------------------------------------------------
// Editor UI (egui)
// ---------------------------------------------------------------------------

fn editor_ui(
    mut contexts: bevy_egui::EguiContexts,
    mut planet: ResMut<EditedPlanet>,
    system: Res<SystemData>,
    diagnostics: Res<DiagnosticsStore>,
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
            planet.generator = resolved.generator;
            planet.heliocentric_distance_m = resolved.heliocentric_distance_m;
            planet.light_intensity = light_intensity_at(resolved.heliocentric_distance_m);
            planet.selected_body = name;
            planet.dirty = true;
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
        let mut changed = false;
        changed |= fires(&ui.add(
            bevy_egui::egui::Slider::new(&mut planet.generator.seed, 0..=9999).text("Seed"),
        ));

        ui.separator();
        ui.heading("Shading");
        changed |= fires(&ui.add(
            bevy_egui::egui::Slider::new(&mut planet.terminator_wrap, 0.0..=1.0)
                .text("Terminator wrap"),
        ));
        changed |= fires(&ui.add(
            bevy_egui::egui::Slider::new(
                &mut planet.sun_azimuth,
                -std::f32::consts::PI..=std::f32::consts::PI,
            )
            .text("Sun azimuth"),
        ));
        changed |= fires(&ui.add(
            bevy_egui::egui::Slider::new(
                &mut planet.sun_elevation,
                -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
            )
            .text("Sun elevation"),
        ));

        if changed {
            planet.dirty = true;
        }
    });
}

fn apply_descriptor_changes(
    mut planet: ResMut<EditedPlanet>,
    query: Query<&PlanetMaterialHandle, With<PreviewPlanet>>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    if !planet.dirty {
        return;
    }
    planet.dirty = false;

    for handle in &query {
        let Some(mat) = materials.get_mut(&handle.0) else { continue };
        mat.params.light_intensity = planet.light_intensity;
        mat.params.ambient_intensity = AMBIENT_INTENSITY;
        let dir = sun_direction(planet.sun_azimuth, planet.sun_elevation);
        mat.params.light_dir = Vec4::new(dir.x, dir.y, dir.z, planet.terminator_wrap);

        let bake = bake_preview(&planet, &mut images, &mut storage_buffers);
        mat.params.height_range = bake.height_range;
        mat.albedo = bake.textures.albedo;
        mat.height = bake.textures.height;
        mat.detail = bake.detail;
        mat.material_cube = bake.textures.material_cube;
        mat.craters = bake.textures.craters;
        mat.cell_index = bake.textures.cell_index;
        mat.feature_ids = bake.textures.feature_ids;
        mat.materials = bake.textures.materials;
    }
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
            generator: resolved.generator,
            heliocentric_distance_m: resolved.heliocentric_distance_m,
            light_intensity,
            terminator_wrap: 0.2,
            sun_azimuth: 0.0,
            sun_elevation: 0.0,
            dirty: false,
        })
        .init_resource::<OrbitCamera>()
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
                apply_descriptor_changes,
            ),
        )
        .run();
}
