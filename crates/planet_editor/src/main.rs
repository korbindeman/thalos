use bevy::asset::AssetPlugin;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::PresentMode;
use thalos_physics::parsing::load_solar_system;
use thalos_physics::types::{BodyKind, SolarSystemDefinition};
use thalos_terrain_gen::{BodyBuilder, Composition, Pipeline};
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

const SOLAR_SYSTEM_KDL: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/solar_system.kdl"));

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
    seed: u64,
    composition: Composition,
    cubemap_resolution: u32,
    heliocentric_distance_m: f64,
    light_intensity: f32,
    terminator_wrap: f32,
    dirty: bool,
}

// ---------------------------------------------------------------------------
// Body → editor params conversion
// ---------------------------------------------------------------------------

fn build_params_for_body(
    system: &SolarSystemDefinition,
    body: &thalos_physics::types::BodyDefinition,
) -> (f64, u64, Composition, f64) {
    let heliocentric = heliocentric_sma(system, body);

    let (composition, seed) = body
        .procedural
        .as_ref()
        .map(|profile| {
            (
                Composition::new(
                    profile.composition.silicate,
                    profile.composition.iron,
                    profile.composition.ice,
                    profile.composition.volatiles,
                    profile.composition.hydrogen_helium,
                ),
                profile.seed,
            )
        })
        .unwrap_or_else(|| {
            (
                Composition::new(0.95, 0.05, 0.0, 0.0, 0.0),
                hash_name_to_seed(&body.name),
            )
        });

    (body.radius_m, seed, composition, heliocentric)
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

fn hash_name_to_seed(name: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
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

fn bake_preview(
    planet: &EditedPlanet,
    images: &mut Assets<Image>,
) -> PlanetTextures {
    let builder = BodyBuilder::new(
        planet.radius_m as f32,
        planet.seed,
        planet.composition,
        planet.cubemap_resolution,
    );

    // No stages yet — pipeline is empty.
    let pipeline = Pipeline::new(vec![]);
    let mut builder = builder;
    pipeline.run(&mut builder);

    let body = builder.build();
    bake_from_body_data(&body, images)
}

fn spawn_preview_planet(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut images: ResMut<Assets<Image>>,
    planet: Res<EditedPlanet>,
) {
    let billboard_mesh = meshes.add(Rectangle::new(2.0, 2.0));

    let textures = bake_preview(&planet, &mut images);

    let mat_handle = planet_materials.add(PlanetMaterial {
        params: PlanetParams {
            radius: 1.5,
            rotation_phase: 0.0,
            light_intensity: planet.light_intensity,
            ambient_intensity: AMBIENT_INTENSITY,
            light_dir: Vec4::new(0.0, 0.0, 1.0, planet.terminator_wrap),
        },
        albedo: textures.albedo,
        height: textures.height,
        detail: PlanetDetailParams::default(),
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
) {
    let Ok(ctx) = contexts.ctx_mut() else { return };
    bevy_egui::egui::Window::new("Planet Editor").show(ctx, |ui| {
        // ---- Body picker ------------------------------------------------
        let mut new_body: Option<String> = None;
        bevy_egui::egui::ComboBox::from_label("Body")
            .selected_text(&planet.selected_body)
            .show_ui(ui, |ui| {
                for b in &system.system.bodies {
                    if b.kind == BodyKind::Star {
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
        {
            let body = &system.system.bodies[id];
            let (radius_m, seed, composition, hel) = build_params_for_body(&system.system, body);
            planet.radius_m = radius_m;
            planet.seed = seed;
            planet.composition = composition;
            planet.cubemap_resolution = 0; // auto
            planet.heliocentric_distance_m = hel;
            planet.light_intensity = light_intensity_at(hel);
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
        changed |= fires(&ui.add(bevy_egui::egui::Slider::new(&mut planet.seed, 0..=9999).text("Seed")));

        ui.separator();
        ui.heading("Shading");
        changed |= fires(&ui.add(
            bevy_egui::egui::Slider::new(&mut planet.terminator_wrap, 0.0..=1.0)
                .text("Terminator wrap"),
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
) {
    if !planet.dirty {
        return;
    }
    planet.dirty = false;

    for handle in &query {
        let Some(mat) = materials.get_mut(&handle.0) else { continue };
        mat.params.light_intensity = planet.light_intensity;
        mat.params.ambient_intensity = AMBIENT_INTENSITY;
        mat.params.light_dir = Vec4::new(
            mat.params.light_dir.x,
            mat.params.light_dir.y,
            mat.params.light_dir.z,
            planet.terminator_wrap,
        );

        let textures = bake_preview(&planet, &mut images);
        mat.albedo = textures.albedo;
        mat.height = textures.height;
        mat.detail = PlanetDetailParams::default();
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let system = load_solar_system(SOLAR_SYSTEM_KDL).expect("parse solar_system.kdl");

    let (radius_m, seed, composition, heliocentric_distance_m, selected_body) = {
        let id = system
            .name_to_id
            .get(DEFAULT_BODY_NAME)
            .copied()
            .or_else(|| {
                system
                    .bodies
                    .iter()
                    .find(|b| b.kind != BodyKind::Star)
                    .map(|b| b.id)
            })
            .expect("no non-star body in solar system");
        let body = &system.bodies[id];
        let (radius_m, seed, composition, hel) = build_params_for_body(&system, body);
        (radius_m, seed, composition, hel, body.name.clone())
    };

    let light_intensity = light_intensity_at(heliocentric_distance_m);

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.02, 0.01, 0.04)))
        .insert_resource(SystemData { system })
        .insert_resource(EditedPlanet {
            selected_body,
            radius_m,
            seed,
            composition,
            cubemap_resolution: 0,
            heliocentric_distance_m,
            light_intensity,
            terminator_wrap: 0.2,
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
