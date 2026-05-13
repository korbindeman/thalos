use std::collections::HashMap;
use std::time::{Duration, Instant};

use bevy::asset::AssetPlugin;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use bevy::window::PresentMode;
use bevy_egui::egui;
use thalos_input::enhanced::{ActionSources, EnhancedInputSystems};
use thalos_input::planet_editor::{PlanetEditorInputIntent, PlanetEditorInputPlugin};
use thalos_input::settings::InputSettings;
use thalos_physics::body_trajectory_provider::BodyTrajectoryProvider;
use thalos_physics::canonical::Epoch;
use thalos_physics::parsing::load_solar_system_from_dir;
use thalos_physics::patched_conics::PatchedConics;
use thalos_physics::types::{BodyDefinition, BodyId, BodyKind, SolarSystemDefinition};
use thalos_planet_rendering::{
    AtmosphereBlock, CLOUD_BAND_COUNT, GasGiantLayers, GasGiantMaterial, GasGiantMaterialHandle,
    GasGiantParams, PlanetCoastlineParams, PlanetDetailParams, PlanetHaloMaterial,
    PlanetHaloMaterialHandle, PlanetMaterial, PlanetMaterialHandle, PlanetParams,
    PlanetRenderingPlugin, PlanetWaterParams, ReferenceClouds, RingLayers, RingMaterial,
    RingMaterialHandle, RingParams, SceneLighting, StarLight, bake_from_planet_surface,
    build_ring_mesh, cloud_cover_image_for_body, convert_reference_clouds_when_ready,
    load_reference_cloud_sources,
};
use thalos_terrain_gen::{
    AirlessImpactProjectionConfig, AtmosphereSpec, AuthoredFeatureConfig, BodyArchetype,
    BoundaryKind, ColdDesertProjectionConfig, CompositionClass, DynamicSurfaceLayers,
    DynamicSurfaceState, FeatureId, FeatureLock, FeatureManifest, FeatureParamValue,
    FeatureProjectionConfig, FeatureSeed, FeatureSeedStream, FeatureTerrainConfig, HydrosphereSpec,
    IceInventory, MegabasinFeatureConfig, OceanTerrainConfig, PlanetSurface, PlateKind,
    TectonicActivity, TectonicConfig, TectonicSystem, TerrainCompileContext, TerrainCompileOptions,
    TerrainConfig, TerrainIntent, compile_terrain_config, plan_initial_compilation, sub_seed,
};

mod sky_backdrop;
mod surface_overlay;

use sky_backdrop::SkyBackdropPlugin;
use surface_overlay::{
    PreviewSurfaceOverlays, SurfaceOverlayOrientation, SurfaceOverlayPlugin,
    SurfaceOverlayRenderRadius, SurfaceOverlayState, activity_label,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LIGHT_AT_1AU: f32 = 10.0;
const AMBIENT_INTENSITY: f32 = 0.05;
const AU_M: f64 = 1.496e11;
const DEFAULT_BODY_NAME: &str = "Mira";
const RENDER_RADIUS: f32 = 1.5;

/// Live-edit rebakes wait this long after the last edit before kicking off,
/// so a slider drag doesn't queue dozens of throwaway bakes.
const REBAKE_DEBOUNCE_MS: u128 = 150;
/// Cubemap resolution used for live preview rebakes. Keep this at the same
/// resolution as the normal headless bake so coastline shaping reads in the
/// editor instead of being smoothed away by the preview texture.
const PREVIEW_CUBEMAP_RESOLUTION: u32 = 512;
/// Explicit mid-resolution bake for checking near-final terrain without paying
/// the full 2048² compile cost.
const HALF_CUBEMAP_RESOLUTION: u32 = 1024;

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
        /// Optional tectonic structural prior. Cloned from the body's
        /// `BodyDefinition.tectonics`. Threaded through to the bake task
        /// so the resulting `PlanetSurface` carries a `TectonicSystem`
        /// for downstream visualization.
        tectonics: Option<TectonicConfig>,
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
}

/// Active sketching tool. `Inspect` is the default — clicks on the planet
/// pick features instead of placing. The other variants enter placement mode:
/// the next planet click appends an authored feature of the matching kind.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum ToolMode {
    #[default]
    Inspect,
    AddMegabasin,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum TerrainBakeMode {
    #[default]
    Preview,
    Half,
    Full,
}

impl TerrainBakeMode {
    fn resolution_override(self) -> Option<u32> {
        match self {
            Self::Preview => Some(PREVIEW_CUBEMAP_RESOLUTION),
            Self::Half => Some(HALF_CUBEMAP_RESOLUTION),
            Self::Full => None,
        }
    }

    fn label(self) -> String {
        match self {
            Self::Preview => format!("preview {PREVIEW_CUBEMAP_RESOLUTION}²"),
            Self::Half => format!("half {HALF_CUBEMAP_RESOLUTION}²"),
            Self::Full => "full".to_string(),
        }
    }
}

impl ToolMode {
    fn label(self) -> &'static str {
        match self {
            Self::Inspect => "Inspect",
            Self::AddMegabasin => "+ Megabasin",
        }
    }

    fn placing(self) -> bool {
        !matches!(self, Self::Inspect)
    }
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
    /// Wall-clock time of the most recent terrain-affecting edit. Drives the
    /// debounced preview rebake so a slider drag doesn't spawn throwaway tasks.
    last_edit: Option<Instant>,
    /// Set by explicit bake buttons. Bypasses debounce, then resets after
    /// dispatch. Live edits use `Preview`.
    requested_bake: Option<TerrainBakeMode>,
    /// Last or in-flight bake mode, for status display.
    last_bake_mode: TerrainBakeMode,
    /// Currently-selected manifest feature (None = nothing selected). Drives
    /// the per-feature inspector panel.
    selected_feature_id: Option<FeatureId>,
    /// Active sketching tool. While `placing()`, planet clicks add features
    /// and the orbit camera ignores left-button drag.
    tool: ToolMode,
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
    /// Bake task. Returns `Err` rather than panicking so a transient compile
    /// failure (e.g. an in-progress edit that puts the spec into a state the
    /// compiler rejects) just logs and leaves the existing terrain on
    /// screen, instead of taking the editor down with the task pool.
    task: Task<Result<PlanetSurface, String>>,
    mesh_entity: Entity,
}

#[allow(dead_code)]
#[derive(Component, Clone)]
struct PreviewDynamicSurface {
    layers: DynamicSurfaceLayers,
    state: DynamicSurfaceState,
}

fn sun_direction(azimuth: f32, elevation: f32) -> Vec3 {
    let (sa, ca) = azimuth.sin_cos();
    let (se, ce) = elevation.sin_cos();
    Vec3::new(ce * sa, se, ce * ca)
}

/// World→body orientation quaternion for the preview, matching the game's
/// `update_planet_orientations` at sim_time = 0 (free-spinning case): the
/// `Ry(phase) * Rx(tilt)` composition collapses to `Rx(tilt)` since phase = 0.
/// Stored in `PlanetParams.orientation` / `GasGiantParams.orientation`, where
/// the shaders use it to rotate world-space directions into body-local space.
fn body_orientation(planet: &EditedPlanet) -> Quat {
    Quat::from_rotation_x(planet.axial_tilt_rad)
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

fn placeholder_terrain_config() -> TerrainConfig {
    TerrainConfig::Ocean(OceanTerrainConfig {
        seed: 0,
        cubemap_resolution: Some(64),
        seabed_albedo: [0.02, 0.05, 0.10],
        water_roughness: 0.04,
        sea_level_m: 1.0,
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
    let body_state = ephemeris.state(body.id, Epoch::ZERO);
    let star_state = ephemeris.state(star_id, Epoch::ZERO);
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
    cloud_cover_image_for_body(&planet.selected_body, reference_clouds, images).0
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
        EditorCamera,
    ));
}

fn camera_input(
    input: Res<PlanetEditorInputIntent>,
    mut orbit: ResMut<OrbitCamera>,
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

fn gate_editor_input_sources(
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
fn toggle_fullbright_hotkey(
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
            cubemap_resolution_override: None,
        };
        // The editor never reads from the cache so edits and compile changes
        // always show up; only full-res bakes write, producing artifacts for
        // downstream consumers.
        let is_full_bake = cubemap_resolution_override.is_none();
        info!("baking {body_name} via {route}");
        let data = match compile_terrain_config(&terrain, tectonics.as_ref(), &context, options) {
            Ok(data) => data,
            Err(e) => return Err(format!("terrain compile failed for {body_name}: {e}")),
        };
        if is_full_bake {
            let key = thalos_terrain_gen::cache::terrain_cache_key(
                &terrain,
                tectonics.as_ref(),
                &context,
                options,
            );
            let path = thalos_terrain_gen::cache::cache_path(&cache_dir, &body_name, key);
            match thalos_terrain_gen::cache::store(&path, key, &data.static_surface) {
                Ok(()) => info!("terrain cache wrote: {body_name}"),
                Err(e) => warn!("terrain cache write failed for {body_name}: {e}"),
            }
        }
        Ok(data)
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

fn spawn_preview_planet(
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

        let mut entity_commands = commands.entity(entity);
        let biome_weights = surface.static_surface.biome_weights_cubemap.clone();
        let overlays = PreviewSurfaceOverlays {
            tectonics: surface.tectonics,
            biome_weights,
        };
        entity_commands
            .insert(PlanetMaterialHandle(mat_handle))
            .insert(PlanetHaloMaterialHandle(halo_handle))
            .insert(overlays)
            .insert(PreviewDynamicSurface {
                layers: surface.dynamic_layers,
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
        &mut planet,
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
    planet.terrain_dirty = false;
    planet.last_edit = None;
    planet.requested_bake = None;
    planet.last_bake_mode = TerrainBakeMode::Preview;
    planet.selected_feature_id = None;
    planet.tool = ToolMode::default();
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

fn draw_spec_controls(ui: &mut egui::Ui, config: &mut FeatureTerrainConfig) -> bool {
    let mut changed = false;
    ui.collapsing("Spec", |ui| {
        let prev_arch = config.archetype;
        egui::ComboBox::from_label("Archetype")
            .selected_text(format!("{:?}", config.archetype))
            .show_ui(ui, |ui| {
                for arch in [
                    BodyArchetype::AirlessImpactMoon,
                    BodyArchetype::ColdDesertFormerlyWet,
                    BodyArchetype::AgingOceanicHomeworld,
                    BodyArchetype::GenericTerrestrial,
                ] {
                    ui.selectable_value(&mut config.archetype, arch, format!("{arch:?}"));
                }
            });
        if config.archetype != prev_arch {
            changed = true;
        }

        let prev_comp = config.composition;
        egui::ComboBox::from_label("Composition")
            .selected_text(format!("{:?}", config.composition))
            .show_ui(ui, |ui| {
                for comp in [
                    CompositionClass::SilicateDominated,
                    CompositionClass::BasalticSilicate,
                    CompositionClass::IronRichSilicate,
                    CompositionClass::IcySilicate,
                ] {
                    ui.selectable_value(&mut config.composition, comp, format!("{comp:?}"));
                }
            });
        if config.composition != prev_comp {
            changed = true;
        }

        changed |= fires(
            &ui.add(egui::Slider::new(&mut config.body_age_gyr, 0.5..=12.0).text("Age (Gyr)")),
        );

        ui.collapsing("Environment", |ui| {
            changed |= fires(
                &ui.add(
                    egui::Slider::new(&mut config.environment.stellar_flux_earth, 0.0..=3.0)
                        .text("Stellar flux (Earth)"),
                ),
            );
            changed |= draw_atmosphere(ui, &mut config.environment.atmosphere);
            changed |= draw_hydrosphere(ui, &mut config.environment.hydrosphere);

            let prev_ice = config.environment.ice_inventory;
            egui::ComboBox::from_label("Ice inventory")
                .selected_text(format!("{:?}", config.environment.ice_inventory))
                .show_ui(ui, |ui| {
                    for ice in [
                        IceInventory::None,
                        IceInventory::Trace,
                        IceInventory::Moderate,
                        IceInventory::High,
                    ] {
                        ui.selectable_value(
                            &mut config.environment.ice_inventory,
                            ice,
                            format!("{ice:?}"),
                        );
                    }
                });
            if config.environment.ice_inventory != prev_ice {
                changed = true;
            }
        });

        ui.collapsing("Intent", |ui| {
            for intent in [
                TerrainIntent::ReadAsMoon,
                TerrainIntent::DistinctNearSideFace,
                TerrainIntent::DifferentFarSide,
                TerrainIntent::FirstLandingWorld,
                TerrainIntent::ReadAsFirstInterplanetarySurfaceWorld,
                TerrainIntent::ForgivingLandingTerrain,
                TerrainIntent::VisibleAncientWaterStory,
                TerrainIntent::RustDustAndEvaporites,
                TerrainIntent::HomeworldIdentity,
            ] {
                let mut on = config.intent.contains(&intent);
                if ui.checkbox(&mut on, format!("{intent:?}")).changed() {
                    if on {
                        config.intent.push(intent);
                    } else {
                        config.intent.retain(|i| *i != intent);
                    }
                    changed = true;
                }
            }
        });
    });
    changed
}

/// Atmosphere variant selector + payload editor. Switching variants preserves
/// the current pressure_bar across variants that carry one.
fn draw_atmosphere(ui: &mut egui::Ui, atmos: &mut AtmosphereSpec) -> bool {
    let mut changed = false;
    let cur_disc = std::mem::discriminant(atmos);
    let pressure = match *atmos {
        AtmosphereSpec::None => 0.01,
        AtmosphereSpec::ThinCo2 { pressure_bar }
        | AtmosphereSpec::Breathable { pressure_bar }
        | AtmosphereSpec::Other { pressure_bar } => pressure_bar,
    };
    egui::ComboBox::from_label("Atmosphere")
        .selected_text(atmosphere_label(atmos))
        .show_ui(ui, |ui| {
            for candidate in [
                AtmosphereSpec::None,
                AtmosphereSpec::ThinCo2 {
                    pressure_bar: pressure,
                },
                AtmosphereSpec::Breathable {
                    pressure_bar: pressure,
                },
                AtmosphereSpec::Other {
                    pressure_bar: pressure,
                },
            ] {
                let selected = std::mem::discriminant(&candidate) == cur_disc;
                if ui
                    .selectable_label(selected, atmosphere_label(&candidate))
                    .clicked()
                    && !selected
                {
                    *atmos = candidate;
                    changed = true;
                }
            }
        });
    if let AtmosphereSpec::ThinCo2 { pressure_bar }
    | AtmosphereSpec::Breathable { pressure_bar }
    | AtmosphereSpec::Other { pressure_bar } = atmos
    {
        changed |= fires(
            &ui.add(
                egui::Slider::new(pressure_bar, 0.001..=10.0)
                    .logarithmic(true)
                    .text("Pressure (bar)"),
            ),
        );
    }
    changed
}

fn atmosphere_label(a: &AtmosphereSpec) -> &'static str {
    match a {
        AtmosphereSpec::None => "None",
        AtmosphereSpec::ThinCo2 { .. } => "ThinCo2",
        AtmosphereSpec::Breathable { .. } => "Breathable",
        AtmosphereSpec::Other { .. } => "Other",
    }
}

fn draw_hydrosphere(ui: &mut egui::Ui, hydro: &mut HydrosphereSpec) -> bool {
    let mut changed = false;
    let cur_disc = std::mem::discriminant(hydro);
    let fraction = match *hydro {
        HydrosphereSpec::OceanFraction(f) => f,
        _ => 0.7,
    };
    egui::ComboBox::from_label("Hydrosphere")
        .selected_text(hydrosphere_label(hydro))
        .show_ui(ui, |ui| {
            for candidate in [
                HydrosphereSpec::None,
                HydrosphereSpec::Trace,
                HydrosphereSpec::AncientLost,
                HydrosphereSpec::OceanFraction(fraction),
            ] {
                let selected = std::mem::discriminant(&candidate) == cur_disc;
                if ui
                    .selectable_label(selected, hydrosphere_label(&candidate))
                    .clicked()
                    && !selected
                {
                    *hydro = candidate;
                    changed = true;
                }
            }
        });
    if let HydrosphereSpec::OceanFraction(f) = hydro {
        changed |= fires(&ui.add(egui::Slider::new(f, 0.0..=1.0).text("Ocean fraction")));
    }
    changed
}

fn hydrosphere_label(h: &HydrosphereSpec) -> &'static str {
    match h {
        HydrosphereSpec::None => "None",
        HydrosphereSpec::Trace => "Trace",
        HydrosphereSpec::AncientLost => "AncientLost",
        HydrosphereSpec::OceanFraction(_) => "OceanFraction",
    }
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

/// Overlay panel: solid metadata layers from the most recent bake. These are
/// pure visualization controls and never trigger a rebake.
fn draw_surface_overlay_panel(
    ui: &mut egui::Ui,
    overlay: &mut SurfaceOverlayState,
    preview: Option<&PreviewSurfaceOverlays>,
) {
    ui.heading("Overlays");
    let has_plates = preview.and_then(|p| p.tectonics.as_ref()).is_some();
    let has_biomes = preview.is_some();

    ui.add_enabled(
        has_plates,
        egui::Checkbox::new(&mut overlay.show_plates, "Plate colors"),
    );
    ui.add_enabled(
        has_biomes,
        egui::Checkbox::new(&mut overlay.show_biomes, "Biome colors"),
    );
}

/// Tectonics panel: live stats from the most recent bake and a config
/// sub-section. The layer config is separated from overlay controls so its
/// edits (which *do* trigger rebakes) can't be confused with visualization
/// toggles.
///
/// `archetype_requires_tectonics` locks the layer-on/off checkbox: bodies
/// whose archetype requires a tectonic graph (currently
/// `AgingOceanicHomeworld`) cannot be toggled to None — disabling would put
/// the bake into a guaranteed-fail state.
///
/// Returns true if any edit should trigger a rebake.
fn draw_tectonics_panel(
    ui: &mut egui::Ui,
    tectonics: &mut Option<TectonicConfig>,
    preview: Option<&TectonicSystem>,
    archetype_requires_tectonics: bool,
) -> bool {
    let mut changed = false;
    ui.heading("Tectonics");

    // ── Layer presence ──
    // For required archetypes the toggle is shown disabled with a label so
    // the constraint is visible; for optional archetypes it lets you opt in
    // or out of the layer entirely.
    if archetype_requires_tectonics {
        ui.label("Tectonic layer: required by archetype");
        if tectonics.is_none() {
            // Defensive: an archetype that requires tectonics shouldn't be
            // sitting at None. Seed a default so the bake doesn't fail on
            // the next rebake. Mark as changed so the rebake fires.
            *tectonics = Some(default_tectonic_config());
            changed = true;
        }
    } else {
        let mut enabled = tectonics.is_some();
        if ui
            .checkbox(&mut enabled, "Tectonic layer")
            .on_hover_text("Spherical-Voronoi plate graph; drives the plate-color overlay and (for AgingOceanicHomeworld) terrain shape.")
            .changed()
        {
            if enabled {
                *tectonics = Some(default_tectonic_config());
            } else {
                *tectonics = None;
            }
            changed = true;
        }
    }

    let Some(config) = tectonics.as_mut() else {
        return changed;
    };

    // ── Stats from the most recent bake ──
    // `preview` is None during the brief gap between dispatch and finalize;
    // show "–" placeholders so the layout doesn't jump.
    if let Some(sys) = preview {
        let n_continental = sys
            .plates
            .iter()
            .filter(|p| p.kind == PlateKind::Continental)
            .count();
        let n_oceanic = sys.plates.len() - n_continental;
        let mut convergent = 0usize;
        let mut divergent = 0usize;
        let mut transform = 0usize;
        for b in &sys.boundaries {
            match b.kind {
                BoundaryKind::Convergent => convergent += 1,
                BoundaryKind::Divergent => divergent += 1,
                BoundaryKind::Transform => transform += 1,
            }
        }
        ui.label(format!(
            "Plates: {} (continental {}, oceanic {})",
            sys.plates.len(),
            n_continental,
            n_oceanic,
        ));
        ui.label(format!(
            "Boundaries: {} (convergent {}, divergent {}, transform {})",
            sys.boundaries.len(),
            convergent,
            divergent,
            transform,
        ));
        ui.label(format!("Mesh cells: {}", sys.mesh.cells.len()));
        ui.label(format!("Activity: {}", activity_label(sys.config.activity)));
    } else {
        ui.label("Plates: –");
        ui.label("Boundaries: –");
        ui.label("Mesh cells: –");
        ui.label("Activity: –");
    }

    ui.separator();

    // ── Layer config ──
    // Slider ranges are conservative enough that no value produces a
    // degenerate tectonic graph. plate_count should stay below mesh_cells;
    // we don't enforce that on slider clamp because it's unusual and a
    // deliberate footgun there is fine.
    ui.label("Configuration:");
    ui.horizontal(|ui| {
        changed |= fires(&ui.add(egui::Slider::new(&mut config.seed, 0..=99_999).text("Seed")));
        if ui.button("Reroll").clicked() {
            config.seed = sub_seed(config.seed, "planet_editor:tectonic_seed");
            changed = true;
        }
    });
    changed |= fires(&ui.add(egui::Slider::new(&mut config.plate_count, 1..=64).text("Plates")));
    changed |=
        fires(&ui.add(egui::Slider::new(&mut config.mesh_cells, 256..=8192).text("Mesh cells")));
    changed |= fires(&ui.add(
        egui::Slider::new(&mut config.continental_fraction, 0.0..=1.0).text("Continental fraction"),
    ));

    let prev_activity = config.activity;
    egui::ComboBox::from_label("Activity")
        .selected_text(activity_label(config.activity))
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut config.activity, TectonicActivity::Active, "Active");
            ui.selectable_value(
                &mut config.activity,
                TectonicActivity::StagnantLid,
                "Stagnant lid",
            );
            // Frozen carries an age; pin to a placeholder when toggling
            // from the dropdown. The age field gets its own slider when
            // Frozen is selected.
            ui.selectable_value(
                &mut config.activity,
                TectonicActivity::Frozen { age_my: 1000.0 },
                "Frozen",
            );
        });
    if config.activity != prev_activity {
        changed = true;
    }
    if let TectonicActivity::Frozen { age_my } = &mut config.activity {
        changed |= fires(&ui.add(egui::Slider::new(age_my, 0.0..=4500.0).text("Frozen age (Myr)")));
    }

    changed
}

/// Default tectonic config seeded when the user opts in via the panel.
/// Earth-like ratios, StagnantLid (no live motion) so it's safe on bodies
/// regardless of activity expectations.
fn default_tectonic_config() -> TectonicConfig {
    TectonicConfig {
        plate_count: 12,
        mesh_cells: 2000,
        activity: TectonicActivity::StagnantLid,
        continental_fraction: 0.30,
        seed: 1,
        seed_dirs: None,
        continental_clustering: 0.0,
        equatorial_bias: 0.0,
        primary_size_multiplier: 1.0,
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

/// Draw the manifest as a flat indented selectable list. Returns the feature
/// id newly clicked this frame, if any. Tree depth is small (≤3) so flat
/// rendering with manual indentation is more usable than nested collapsibles.
fn draw_feature_manifest(
    ui: &mut egui::Ui,
    manifest: &FeatureManifest,
    selected: Option<&FeatureId>,
) -> Option<FeatureId> {
    let mut clicked = None;
    ui.collapsing("Feature Manifest", |ui| {
        ui.label(format!("{} features", manifest.features.len()));
        let root_children = manifest
            .get(&manifest.root)
            .map(|root| root.children.clone())
            .unwrap_or_default();
        for child_id in &root_children {
            walk_manifest_flat(ui, manifest, child_id, selected, &mut clicked, 0);
        }
    });
    clicked
}

fn walk_manifest_flat(
    ui: &mut egui::Ui,
    manifest: &FeatureManifest,
    id: &FeatureId,
    selected: Option<&FeatureId>,
    clicked: &mut Option<FeatureId>,
    depth: usize,
) {
    let Some(feature) = manifest.get(id) else {
        return;
    };
    let indent: String = std::iter::repeat_n("  ", depth).collect();
    let scale = if feature.scale_range_m.max_m.is_finite() {
        format!(
            " · {:.1}-{:.1} km",
            feature.scale_range_m.min_m / 1_000.0,
            feature.scale_range_m.max_m / 1_000.0
        )
    } else {
        " · global".to_string()
    };
    let label = format!("{indent}{} · {:?}{scale}", feature.id, feature.kind);
    let is_selected = selected == Some(id);
    if ui.selectable_label(is_selected, label).clicked() {
        *clicked = Some(id.clone());
    }
    let children = feature.children.clone();
    for child_id in &children {
        walk_manifest_flat(ui, manifest, child_id, selected, clicked, depth + 1);
    }
}

/// Inspector panel for the selected manifest feature. Editable for authored
/// features (matched by id against `authored`); read-only for generated ones.
/// Sets `delete` to the feature id the user asked to remove, if any.
fn draw_selected_inspector(
    ui: &mut egui::Ui,
    selected_id: &FeatureId,
    manifest: &FeatureManifest,
    root_seed: u64,
    authored: &mut [AuthoredFeatureConfig],
    delete: &mut Option<FeatureId>,
) -> bool {
    let mut changed = false;
    let Some(feature) = manifest.get(selected_id) else {
        ui.label(format!("(missing feature: {selected_id})"));
        return false;
    };

    ui.heading(feature.id.as_str());
    ui.label(format!("Kind: {:?}", feature.kind));
    ui.label(format!("Era: {:?}", feature.era));
    if feature.scale_range_m.max_m.is_finite() {
        ui.label(format!(
            "Scale: {:.1}-{:.1} km",
            feature.scale_range_m.min_m / 1_000.0,
            feature.scale_range_m.max_m / 1_000.0
        ));
    } else {
        ui.label("Scale: global");
    }

    let authored_index = authored.iter().position(|a| match a {
        AuthoredFeatureConfig::Megabasin(c) => &c.id == selected_id,
    });

    if let Some(idx) = authored_index {
        ui.separator();
        ui.label("(authored)");
        match &mut authored[idx] {
            AuthoredFeatureConfig::Megabasin(config) => {
                changed |= fires(&ui.add(
                    egui::Slider::new(&mut config.radius_km, 50.0..=2000.0).text("Radius (km)"),
                ));
                changed |= fires(
                    &ui.add(egui::Slider::new(&mut config.depth_km, 0.5..=20.0).text("Depth (km)")),
                );

                let mut has_rings = config.ring_count.is_some();
                let mut ring_count = config.ring_count.unwrap_or(2);
                if ui.checkbox(&mut has_rings, "Concentric rings").changed() {
                    config.ring_count = if has_rings { Some(ring_count) } else { None };
                    changed = true;
                }
                if has_rings
                    && fires(&ui.add(egui::Slider::new(&mut ring_count, 1..=4).text("Ring count")))
                {
                    config.ring_count = Some(ring_count);
                    changed = true;
                }

                ui.separator();
                ui.label("Reroll seed:");
                ui.horizontal(|ui| {
                    for (label, stream) in [
                        ("Placement", FeatureSeedStream::Placement),
                        ("Shape", FeatureSeedStream::Shape),
                        ("Detail", FeatureSeedStream::Detail),
                        ("Children", FeatureSeedStream::Children),
                    ] {
                        if ui.small_button(label).clicked() {
                            reroll_authored_seed(root_seed, &config.id, &mut config.seed, stream);
                            changed = true;
                        }
                    }
                });

                let prev_lock = config.lock;
                egui::ComboBox::from_label("Lock")
                    .selected_text(format!("{:?}", config.lock))
                    .show_ui(ui, |ui| {
                        for lock in [
                            FeatureLock::Unlocked,
                            FeatureLock::Placement,
                            FeatureLock::Shape,
                            FeatureLock::Detail,
                            FeatureLock::ShapeAndPlacement,
                            FeatureLock::Full,
                        ] {
                            ui.selectable_value(&mut config.lock, lock, format!("{lock:?}"));
                        }
                    });
                if config.lock != prev_lock {
                    changed = true;
                }

                ui.separator();
                if ui.button("Delete").clicked() {
                    *delete = Some(config.id.clone());
                    changed = true;
                }
            }
        }
    } else {
        ui.label("(generated)");
        for param in &feature.params {
            let line = match &param.value {
                FeatureParamValue::Number(n) => format!("{}: {n:.3}", param.key),
                FeatureParamValue::Text(t) => format!("{}: {t}", param.key),
                FeatureParamValue::Bool(b) => format!("{}: {b}", param.key),
                FeatureParamValue::Direction(_) => format!("{}: <direction>", param.key),
            };
            ui.label(line);
        }
        ui.separator();
        ui.add_enabled(false, egui::Button::new("Promote (TODO)"));
    }

    changed
}

/// Generate a new authored-feature id like `user.megabasin.7` by scanning
/// existing authored features for the highest numeric suffix on `prefix`.
fn next_authored_id(authored: &[AuthoredFeatureConfig], prefix: &str) -> String {
    let mut max_n: u32 = 0;
    for a in authored {
        let id = match a {
            AuthoredFeatureConfig::Megabasin(c) => c.id.as_str(),
        };
        if let Some(rest) = id.strip_prefix(prefix)
            && let Ok(n) = rest.parse::<u32>()
        {
            max_n = max_n.max(n);
        }
    }
    format!("{prefix}{}", max_n + 1)
}

/// Ray-vs-sphere intersection (sphere centered at origin). Returns the
/// surface direction (unit) of the nearer hit, or `None` if the ray misses
/// or the sphere is behind the origin.
fn ray_vs_sphere(origin: Vec3, dir: Vec3, radius: f32) -> Option<Vec3> {
    let b = origin.dot(dir);
    let c = origin.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t < 0.0 {
        return None;
    }
    Some((origin + dir * t).normalize())
}

/// Convert a left-click on the 3D view into a new authored feature on the
/// planet surface. Inert when no placement tool is active or when the cursor
/// is over an egui panel.
fn pick_planet_click(
    input: Res<PlanetEditorInputIntent>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform), With<EditorCamera>>,
    mut planet: ResMut<EditedPlanet>,
    mut egui_ctx: bevy_egui::EguiContexts,
) {
    if !input.primary_started {
        return;
    }
    if !planet.tool.placing() {
        return;
    }
    if egui_ctx
        .ctx_mut()
        .is_ok_and(|ctx| ctx.wants_pointer_input())
    {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_transform)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor) else {
        return;
    };
    let Some(direction) = ray_vs_sphere(ray.origin, *ray.direction, RENDER_RADIUS) else {
        return;
    };

    let tool = planet.tool;
    let new_id = match tool {
        ToolMode::Inspect => return,
        ToolMode::AddMegabasin => {
            let BodyMode::Terrain {
                ref mut terrain, ..
            } = planet.mode
            else {
                return;
            };
            let TerrainConfig::Feature(config) = terrain else {
                return;
            };
            let id_str = next_authored_id(&config.authored_features, "user.megabasin.");
            let new_id = FeatureId::new(id_str);
            config
                .authored_features
                .push(AuthoredFeatureConfig::Megabasin(MegabasinFeatureConfig {
                    id: new_id.clone(),
                    parent: None,
                    center_dir: direction,
                    radius_km: 250.0,
                    depth_km: 5.0,
                    ring_count: None,
                    seed: None,
                    lock: FeatureLock::Placement,
                }));
            new_id
        }
    };

    planet.selected_feature_id = Some(new_id);
    planet.terrain_dirty = true;
    planet.last_edit = Some(Instant::now());
}

fn editor_ui(
    mut contexts: bevy_egui::EguiContexts,
    mut planet: ResMut<EditedPlanet>,
    system: Res<SystemData>,
    diagnostics: Res<DiagnosticsStore>,
    status: Res<TerrainGenStatus>,
    mut overlay_state: ResMut<SurfaceOverlayState>,
    surface_overlay_q: Query<&PreviewSurfaceOverlays, With<PreviewPlanet>>,
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
    let max_panel_height = (ctx.available_rect().height() - 32.0).max(200.0);
    egui::Window::new("Planet Editor")
        .default_pos(controls_pos)
        .default_width(340.0)
        .max_height(max_panel_height)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    let fps = diagnostics
                        .get(&FrameTimeDiagnosticsPlugin::FPS)
                        .and_then(|d| d.smoothed())
                        .unwrap_or(0.0);
                    ui.label(format!("FPS: {:.0}", fps));
                    ui.label(format!("Body: {}", planet.selected_body));
                    ui.separator();

                    // ---- Terrain gen status ----------------------------------------
                    if matches!(planet.mode, BodyMode::Terrain { .. }) {
                        let mode_label = planet.last_bake_mode.label();
                        match (status.current_started, status.last_duration) {
                            (Some(started), _) => {
                                let elapsed = started.elapsed().as_secs_f32();
                                ui.label(format!("Generating ({mode_label}) for {elapsed:.2}s…"));
                            }
                            (None, Some(d)) => {
                                ui.label(format!(
                                    "Last bake ({mode_label}): {:.2}s",
                                    d.as_secs_f32()
                                ));
                            }
                            (None, None) => {}
                        }
                        ui.horizontal(|ui| {
                            let busy = status.current_started.is_some();
                            if ui
                                .add_enabled(!busy, egui::Button::new("Bake half res"))
                                .clicked()
                            {
                                planet.requested_bake = Some(TerrainBakeMode::Half);
                            }
                            if ui
                                .add_enabled(!busy, egui::Button::new("Bake full res"))
                                .clicked()
                            {
                                planet.requested_bake = Some(TerrainBakeMode::Full);
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label("Tool:");
                            for tool in [ToolMode::Inspect, ToolMode::AddMegabasin] {
                                let selected = planet.tool == tool;
                                if ui.selectable_label(selected, tool.label()).clicked() {
                                    planet.tool = if selected { ToolMode::Inspect } else { tool };
                                }
                            }
                        });
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
                    let mut selected_id = planet.selected_feature_id.clone();
                    let mut delete_request: Option<FeatureId> = None;
                    let surface_overlays = surface_overlay_q.single().ok();

                    if let BodyMode::Terrain {
                        ref mut terrain,
                        ref mut tectonics,
                        tidal_axis,
                    } = planet.mode
                    {
                        ui.heading("Parameters");
                        ui.label(format!("Terrain: {}", terrain.route_label()));
                        match terrain {
                            TerrainConfig::Feature(config) => {
                                ui.horizontal(|ui| {
                                    terrain_changed |= fires(&ui.add(
                                        egui::Slider::new(&mut config.seed, 0..=9999).text("Seed"),
                                    ));
                                    if ui.button("Reroll World").clicked() {
                                        config.seed =
                                            sub_seed(config.seed, "planet_editor:world_seed");
                                        terrain_changed = true;
                                    }
                                });
                                terrain_changed |= draw_spec_controls(ui, config);
                                terrain_changed |=
                                    draw_projection_controls(ui, &mut config.projection);

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
                                if let Some(c) =
                                    draw_feature_manifest(ui, &plan.manifest, selected_id.as_ref())
                                {
                                    selected_id = Some(c);
                                }
                                if let Some(sel) = selected_id.clone() {
                                    ui.separator();
                                    ui.heading("Selected");
                                    terrain_changed |= draw_selected_inspector(
                                        ui,
                                        &sel,
                                        &plan.manifest,
                                        config.seed,
                                        &mut config.authored_features,
                                        &mut delete_request,
                                    );
                                    if let Some(del_id) = delete_request.clone() {
                                        config.authored_features.retain(|a| match a {
                                            AuthoredFeatureConfig::Megabasin(c) => c.id != del_id,
                                        });
                                        selected_id = None;
                                        terrain_changed = true;
                                    }
                                }
                            }
                            TerrainConfig::Ocean(ocean) => {
                                terrain_changed |= fires(&ui.add(
                                    egui::Slider::new(&mut ocean.seed, 0..=9999).text("Seed"),
                                ));
                                terrain_changed |= fires(
                                    &ui.add(
                                        egui::Slider::new(&mut ocean.sea_level_m, 0.0..=10.0)
                                            .text("Sea level (m)"),
                                    ),
                                );
                                terrain_changed |= fires(
                                    &ui.add(
                                        egui::Slider::new(&mut ocean.water_roughness, 0.0..=0.3)
                                            .text("Water roughness"),
                                    ),
                                );
                            }
                            TerrainConfig::None => {}
                        }

                        ui.separator();
                        draw_surface_overlay_panel(ui, &mut overlay_state, surface_overlays);
                        ui.separator();

                        let archetype_requires_tectonics = matches!(
                            terrain,
                            TerrainConfig::Feature(c)
                                if c.archetype == BodyArchetype::AgingOceanicHomeworld
                        );
                        terrain_changed |= draw_tectonics_panel(
                            ui,
                            tectonics,
                            surface_overlays.and_then(|o| o.tectonics.as_ref()),
                            archetype_requires_tectonics,
                        );
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
                    let mut sun_azimuth_deg = planet.sun_azimuth.to_degrees();
                    if ui
                        .add(
                            egui::DragValue::new(&mut sun_azimuth_deg)
                                .speed(0.25)
                                .prefix("Sun azimuth: ")
                                .suffix(" deg")
                                .custom_formatter(|n, _| format!("{:.1}", n.rem_euclid(360.0))),
                        )
                        .changed()
                    {
                        planet.sun_azimuth = sun_azimuth_deg.to_radians();
                        uniforms_changed = true;
                    }
                    // Axial tilt affects both shader orientation and terrain compile
                    // context. Apply the visible orientation immediately, then let
                    // terrain preview bakes use the normal debounce.
                    let mut tilt_deg = planet.axial_tilt_rad.to_degrees();
                    if ui
                        .add(
                            bevy_egui::egui::Slider::new(&mut tilt_deg, -180.0..=180.0)
                                .text("Axial tilt (deg)"),
                        )
                        .changed()
                    {
                        planet.axial_tilt_rad = tilt_deg.to_radians();
                        uniforms_changed = true;
                        if matches!(&planet.mode, BodyMode::Terrain { .. }) {
                            terrain_changed = true;
                        }
                    }

                    if terrain_changed {
                        planet.terrain_dirty = true;
                        planet.last_edit = Some(Instant::now());
                    }
                    if uniforms_changed {
                        planet.uniforms_dirty = true;
                    }
                    planet.selected_feature_id = selected_id;
                });
        });
}

/// Applies shader-uniform-only changes to the current material.
#[allow(clippy::too_many_arguments)]
/// Mirror the editor's world-to-body orientation quaternion into the resource
/// the surface overlays read. The overlay plugin inverts it for the shell
/// transform so metadata colors physically track the rendered body.
fn sync_surface_overlay_orientation(
    planet: Res<EditedPlanet>,
    mut orientation: ResMut<SurfaceOverlayOrientation>,
) {
    let q = body_orientation(&planet);
    if orientation.0 != q {
        orientation.0 = q;
    }
}

fn apply_uniform_changes(
    mut planet: ResMut<EditedPlanet>,
    overlay_state: Res<SurfaceOverlayState>,
    input: Res<PlanetEditorInputIntent>,
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
    // Any of: an overlay being on, OR space being held, forces fullbright
    // + atmosphere-off so debug views read cleanly. Press / release of
    // either source must rewrite uniforms, so we track the combined flag
    // with a Local — overlay toggles and semantic input can both
    // mutably tick every frame, so `is_changed()` on them is unusable.
    let overlays_on = overlay_state.show_plates || overlay_state.show_biomes;
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

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let body_arg = std::env::args().nth(1);
    let preferred = body_arg.as_deref().unwrap_or(DEFAULT_BODY_NAME);

    // Load from disk to include per-body detail files
    let system = {
        use std::path::Path;

        let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("assets"))
            .unwrap_or_else(|| "assets".into());

        load_solar_system_from_dir(&assets_dir).expect("load solar system from disk")
    };

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
        .insert_resource(
            InputSettings::load_from_path("assets/input.ron")
                .expect("Failed to load input bindings from assets/input.ron"),
        )
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
            last_edit: None,
            requested_bake: None,
            last_bake_mode: TerrainBakeMode::Preview,
            selected_feature_id: None,
            tool: ToolMode::default(),
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
        .add_plugins(PlanetEditorInputPlugin)
        .add_plugins(SkyBackdropPlugin)
        .add_plugins(SurfaceOverlayPlugin)
        .insert_resource(SurfaceOverlayRenderRadius(RENDER_RADIUS))
        .init_resource::<SurfaceOverlayOrientation>()
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
            PreUpdate,
            gate_editor_input_sources.before(EnhancedInputSystems::Update),
        )
        .add_systems(
            Update,
            (
                convert_reference_clouds_when_ready,
                camera_input,
                camera_zoom_smoothing.after(camera_input),
                camera_apply_transform.after(camera_zoom_smoothing),
                pick_planet_click.after(camera_input),
                toggle_fullbright_hotkey,
                apply_uniform_changes.after(toggle_fullbright_hotkey),
                handle_body_switch,
                dispatch_rebake
                    .after(handle_body_switch)
                    .after(pick_planet_click),
                finalize_terrain_bake.after(dispatch_rebake),
                patch_preview_reference_cloud_cover
                    .after(convert_reference_clouds_when_ready)
                    .after(finalize_terrain_bake),
                update_preview_atmosphere
                    .after(apply_uniform_changes)
                    .after(finalize_terrain_bake),
                sync_surface_overlay_orientation,
            ),
        )
        .run();
}
