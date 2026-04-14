//! Rendering module: celestial bodies, orbit lines, and ship marker.
//!
//! # Coordinate system
//! The physics simulation uses a heliocentric inertial frame with the ecliptic
//! as the XZ plane (Y up). All positions from the ephemeris are in metres.
//! We apply `RENDER_SCALE` to convert metres to render units so Bevy's f32
//! transforms don't lose precision on solar-system distances.
//!
//! 1 render unit = 1 / RENDER_SCALE metres = 1,000 km.

use std::sync::Arc;

use bevy::light::cascade::CascadeShadowConfigBuilder;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use bevy::window::PrimaryWindow;
use thalos_physics::{
    body_state_provider::BodyStateProvider,
    simulation::Simulation,
    types::{BodyKind, BodyStates, SolarSystemDefinition},
};

use bevy::render::storage::ShaderStorageBuffer;
use thalos_planet_rendering::{
    FilmGrain, GasGiantLayers, GasGiantMaterial, GasGiantMaterialHandle, GasGiantParams,
    PlanetDetailParams, PlanetMaterial, PlanetMaterialHandle, PlanetParams, RingLayers,
    RingMaterial, RingMaterialHandle, RingParams, bake_from_body_data, build_ring_mesh,
};
use thalos_terrain_gen::{BodyBuilder, BodyData, Pipeline, StageDef};

use crate::SimStage;
use crate::camera::{CameraFocus, OrbitCamera};
// Re-export so existing `use crate::rendering::{RENDER_SCALE, RenderOrigin}` sites keep working.
pub use crate::coords::{RENDER_SCALE, RenderOrigin, to_render_pos};

/// Radius of screen-stable body icon markers as a fraction of camera distance
/// (in render units). Bodies whose rendered sphere is smaller than this get
/// replaced by a fixed-size circle billboard.
const MARKER_RADIUS: f32 = 0.006;

/// Dev-mode crater-count scale factor. Cratering + space_weather together
/// dominate the terrain bake and both scale linearly with crater count, so
/// cutting the authored count by 10× in dev brings bakes from minutes to
/// ~20 s. Release builds keep the full authored counts.
#[cfg(debug_assertions)]
const DEV_CRATER_SCALE: f32 = 0.1;
#[cfg(not(debug_assertions))]
const DEV_CRATER_SCALE: f32 = 1.0;

/// Map body kind + size to a surface roughness value (0 = smooth, 1 = very rough).
///
/// This drives the terminator wrap in the planet impostor shader.
/// On a smooth sphere (no normal map), wrap simulates *unresolved* scattering
/// that softens the macro terminator — primarily atmospheric scattering, not
/// surface craters.  Crater roughness creates a *textured* terminator boundary
/// (individual shadow/lit patches), which only makes sense once normal maps
/// provide that detail.
///
/// Terminator wrap factor (shader `light_dir.w`). 0 = razor-sharp Lambert
/// terminator (airless vacuum look); nonzero softens the edge to fake
/// unresolved sub-pixel roughness on atmospheric bodies.
fn body_surface_roughness(body: &thalos_physics::types::BodyDefinition) -> f32 {
    match body.kind {
        BodyKind::Star => 0.0,
        BodyKind::Planet => 0.0,
        BodyKind::Moon => 0.0,
        BodyKind::DwarfPlanet => 0.0,
        BodyKind::Centaur => 0.0,
        BodyKind::Comet => 0.0,
    }
}

/// Sun irradiance at 1 AU in shader units (W/m² scaled). Editor uses the same
/// value — keep them in sync. Per-body intensity is scaled by focus-relative
/// exposure (see `update_planet_light_dirs`) rather than by raw inverse-square
/// falloff, so distant bodies stay legible when the camera focuses on them.
const LIGHT_AT_1AU: f32 = 10.0;

/// Ambient floor. Vacuum has no fill light — night sides are black.
const PLANET_AMBIENT: f32 = 0.0;

const AU_M: f64 = 1.496e11;

/// Points sampled along each orbit for gizmo line drawing.
const ORBIT_SAMPLES: usize = 256;

/// Minimum sim-time advance (seconds) before orbit trails are recomputed.
/// Trails are also recomputed on the first frame.
const ORBIT_TRAIL_RECOMPUTE_INTERVAL: f64 = 3600.0;

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

/// Central simulation state.
#[derive(Resource)]
pub struct SimulationState {
    pub simulation: Simulation,
    pub system: SolarSystemDefinition,
    pub ephemeris: Arc<dyn BodyStateProvider>,
}

/// Per-frame cache of all body states at the current sim time. Populated once
/// per frame by `cache_body_states` and read by multiple rendering systems to
/// avoid redundant ephemeris queries.
#[derive(Resource, Default)]
pub struct FrameBodyStates {
    pub states: Option<BodyStates>,
    pub time: f64,
}

/// Camera exposure model. Acts as the semantic "sensor" of the game camera:
/// it owns how focus distance maps to display brightness and how much grain
/// is added in consequence. Every system that cares about "how much flux
/// does the shader see" or "how much noise should the post stack add" reads
/// this resource rather than recomputing from focus distance.
///
/// Soft sqrt compensation: outer-system focus pulls distant bodies out of
/// black without fully erasing the distance cue. Concretely, the display
/// flux at the focus body scales as `LIGHT_AT_1AU * (1 AU / focus_d)^0.5`,
/// so Thalos focus lands at 10, Nyx focus at ~1.5, Acheron (perihelion
/// 78 AU) at ~1.1. Inverse-sqrt instead of inverse-square keeps the feeling
/// that deep space is dim while staying legible.
///
/// The gain applied to each body's raw inverse-square flux in the impostor
/// shader is `exposure.gain = (focus_d / 1 AU)^1.5`. Combined with the raw
/// `(AU/body_d)^2` falloff baked into `update_planet_light_dirs`, this
/// yields the display flux above.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct CameraExposure {
    /// Camera focus body's distance from the star, in meters.
    pub focus_dist_m: f64,
    /// Multiplicative gain applied to per-body raw inverse-square flux.
    pub gain: f32,
    /// Log2(gain). Positive = we're pushing dark outer-system scenes;
    /// negative = we're pulling down bright inner-system scenes. Drives
    /// film grain strength (and, later, lens flare intensity).
    pub ev: f32,
}

/// Exposure exponent. 2.0 = full compensation (distant bodies look identical
/// to focused Thalos — destroys distance cue). 0.0 = no compensation (Nyx is
/// black). 1.0 = linear-in-distance compensation: display flux at focus is
/// `LIGHT_AT_1AU / focus_d_AU`, so Nyx focus lands at ~0.24 — visibly dim,
/// leaves shadows dark, and doesn't collide with Bevy `AutoExposure` pulling
/// the scene up independently in the post stack.
const EXPOSURE_ALPHA: f64 = 1.0;

/// Maximum positive EV used to drive grain. Beyond this, grain saturates.
/// log2(42^1.0) ≈ 5.4 — Nyx is roughly here.
const EXPOSURE_EV_GRAIN_MAX: f32 = 6.0;

pub fn cache_body_states(sim: Res<SimulationState>, mut cache: ResMut<FrameBodyStates>) {
    let t = sim.simulation.sim_time();
    if cache.states.is_some() && (t - cache.time).abs() < f64::EPSILON {
        return;
    }
    if let Some(states) = cache.states.as_mut() {
        sim.ephemeris.query_into(t, states);
    } else {
        let mut states = Vec::with_capacity(sim.ephemeris.body_count());
        sim.ephemeris.query_into(t, &mut states);
        cache.states = Some(states);
    }
    cache.time = t;
}

/// Convert an sRGB component (0..1) to linear light.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Distance (in render units) the focus body must move from the last-set
/// origin before `update_render_origin` actually updates.  Prevents jitter
/// from tiny frame-to-frame position changes.
const ORIGIN_UPDATE_THRESHOLD: f64 = 1000.0; // ~1 000 000 km

/// Tracks which entity the origin was last locked to, so a focus-body change
/// always triggers an immediate origin update.
#[derive(Resource, Default)]
struct PreviousFocusEntity {
    entity: Option<Entity>,
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct CelestialBody {
    pub body_id: usize,
    pub is_star: bool,
    pub render_radius: f32,
    /// True physical radius in metres (not clamped like render_radius).
    pub radius_m: f64,
}

/// Marks a body whose baked surface is tidally locked to its parent. Each
/// frame the orientation uniform is recomputed so the baked near-side (local
/// +Z, where the mare/tidal asymmetry lives) keeps facing the parent body.
#[derive(Component)]
struct TidallyLocked {
    parent_id: usize,
}

#[derive(Component)]
pub struct ShipMarker;

/// Marker for the directional light that simulates sunlight toward the focus body.
#[derive(Component)]
struct SunLight;

/// Marker for the 3D sphere mesh child of a celestial body.
#[derive(Component)]
struct BodyMesh;

/// Marker for the flat circle icon child of a celestial body.
#[derive(Component)]
struct BodyIcon;

/// In-flight terrain generation task for a procedural body.
///
/// While this component is attached to the parent `CelestialBody` entity, the
/// body renders with a plain placeholder sphere. Once the background task
/// completes, `finalize_planet_generation` bakes the result into GPU textures,
/// swaps the child mesh to the impostor billboard with a `PlanetMaterial`, and
/// removes this component.
#[derive(Component)]
struct PendingPlanetGeneration {
    task: Task<BodyData>,
    body_id: usize,
    render_radius: f32,
    /// Child entity holding the placeholder mesh; gets swapped to the impostor
    /// billboard when the task finishes.
    mesh_entity: Entity,
}

/// Shared meshes reused across every procedural planet, cached once at
/// startup so `finalize_planet_generation` doesn't need to re-add them.
#[derive(Resource)]
struct SharedPlanetMeshes {
    billboard: Handle<Mesh>,
}

// ---------------------------------------------------------------------------
// Orbit trail data (recomputed periodically from simulation)
// ---------------------------------------------------------------------------

/// Stores orbit trail render data for each body. Recomputed periodically
/// as sim time advances so trails reflect the forward trajectory.
#[derive(Resource)]
struct OrbitLines {
    lines: Vec<Option<OrbitLine>>,
    /// Sim time when trails were last computed.
    last_compute_time: f64,
}

struct OrbitLine {
    points: Vec<Vec3>,
    color: Color,
    parent_id: usize,
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Tracks the last left-click time and screen position for double-click detection.
#[derive(Resource, Default)]
struct LastClick {
    time: f64,
    position: Vec2,
}

/// Toggle for drawing celestial body orbit trails.
#[derive(Resource)]
pub struct ShowOrbits(pub bool);

impl Default for ShowOrbits {
    fn default() -> Self {
        Self(true)
    }
}

const DOUBLE_CLICK_THRESHOLD: f64 = 0.4; // seconds
const DOUBLE_CLICK_RADIUS: f32 = 10.0; // pixels — tolerance for cursor drift between clicks

pub struct RenderingPlugin;

impl Plugin for RenderingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(LastClick::default())
            .insert_resource(RenderOrigin::default())
            .insert_resource(FrameBodyStates::default())
            .insert_resource(CameraExposure::default())
            .insert_resource(PreviousFocusEntity::default())
            .insert_resource(ShowOrbits::default())
            .add_systems(
                Startup,
                (
                    configure_gizmos,
                    spawn_bodies,
                    focus_camera_on_homeworld.after(spawn_bodies),
                ),
            )
            .add_systems(
                Update,
                (
                    finalize_planet_generation,
                    cache_body_states,
                    update_render_origin.after(cache_body_states),
                    update_body_positions.after(update_render_origin),
                    update_sun_light.after(cache_body_states),
                    update_camera_exposure.after(cache_body_states),
                    sync_film_grain_to_exposure.after(update_camera_exposure),
                    update_planet_light_dirs
                        .after(cache_body_states)
                        .after(update_camera_exposure)
                        .after(finalize_planet_generation),
                    update_planet_orientations
                        .after(cache_body_states)
                        .after(finalize_planet_generation),
                    update_gas_giant_params
                        .after(cache_body_states)
                        .after(update_camera_exposure),
                    update_ring_params
                        .after(cache_body_states)
                        .after(update_camera_exposure),
                    update_ship_position.after(update_render_origin),
                    recompute_orbit_trails.after(cache_body_states),
                    draw_orbits
                        .after(recompute_orbit_trails)
                        .after(update_render_origin),
                    sync_body_icons,
                    double_click_focus_system,
                )
                    .in_set(SimStage::Sync),
            );
    }
}

fn configure_gizmos(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.line.width = 2.0;
}

// ---------------------------------------------------------------------------
// Startup: spawn body meshes and ship marker
// ---------------------------------------------------------------------------

fn spawn_bodies(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut gas_giant_materials: ResMut<Assets<GasGiantMaterial>>,
    mut ring_materials: ResMut<Assets<RingMaterial>>,
    sim: Res<SimulationState>,
) {
    let bodies = &sim.system.bodies;
    let initial_states = sim.ephemeris.query(0.0);

    // Shared meshes.
    let icon_mesh = meshes.add(Circle::new(1.0));
    // Unit rectangle (corners at ±1) shared across all planet billboards.
    // The vertex shader scales it by params.radius each frame.
    let billboard_mesh = meshes.add(Rectangle::new(2.0, 2.0));
    commands.insert_resource(SharedPlanetMeshes {
        billboard: billboard_mesh.clone(),
    });

    for body in bodies {
        let state = &initial_states[body.id];
        let pos = to_render_pos(state.position);

        let radius_render = (body.radius_m * RENDER_SCALE) as f32;
        let render_radius = radius_render.max(0.005);

        let [r, g, b] = body.color;
        let base_color = Color::srgb(r, g, b);
        let is_star = body.kind == BodyKind::Star;

        // Icon material: unlit, emissive, double-sided flat circle. Alpha is
        // driven per-frame by `sync_body_icons` to crossfade against the
        // impostor mesh as the body shrinks through the icon threshold.
        let icon_material = std_materials.add(StandardMaterial {
            base_color: base_color.with_alpha(0.0),
            emissive: LinearRgba::new(r, g, b, 0.0) * 2.0,
            unlit: true,
            double_sided: true,
            alpha_mode: AlphaMode::Blend,
            // Modest positive bias: wins vs trajectory gizmos at the same
            // body-center depth, but small enough that planet impostors
            // (and other opaque surfaces in front) still occlude the icon.
            depth_bias: 10.0,
            ..default()
        });

        if is_star {
            // Stars keep the simple emissive icosphere — no impostor needed.
            let star_mesh = meshes.add(Sphere::new(render_radius).mesh().ico(5).unwrap());
            let star_material = std_materials.add(StandardMaterial {
                base_color,
                emissive: LinearRgba::WHITE * 5000.0,
                ..default()
            });

            commands
                .spawn((
                    Transform::from_translation(pos),
                    Visibility::Inherited,
                    CelestialBody {
                        body_id: body.id,
                        is_star,
                        render_radius,
                        radius_m: body.radius_m,
                    },
                    Name::new(body.name.clone()),
                ))
                .with_children(|parent| {
                    parent.spawn((
                        Mesh3d(star_mesh),
                        MeshMaterial3d(star_material),
                        NotShadowCaster,
                        NotShadowReceiver,
                        BodyMesh,
                    ));
                })
                .with_child((
                    Mesh3d(icon_mesh.clone()),
                    MeshMaterial3d(icon_material),
                    Transform::default(),
                    Visibility::Hidden,
                    BodyIcon,
                ));
        } else if let Some(gen_params) = &body.generator {
            // Procedural body: dispatch the terrain_gen pipeline to a background
            // task so startup isn't blocked. Meanwhile show a plain placeholder
            // sphere; `finalize_planet_generation` swaps in the impostor
            // billboard with a baked `PlanetMaterial` once the task completes.
            let mut gen_params = gen_params.clone();
            gen_params.scale_crater_count(DEV_CRATER_SCALE);

            let radius_m = body.radius_m as f32;
            let seed = gen_params.seed;
            let composition = gen_params.composition;
            let cubemap_resolution = gen_params.cubemap_resolution;
            let body_age_gyr = gen_params.body_age_gyr;
            // Tidally-locked moons get their local +Z axis as the parent
            // direction, matching the editor.
            let tidal_axis = matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z);
            let axial_tilt_rad = body.axial_tilt_rad as f32;
            let pipeline_defs: Vec<StageDef> = gen_params.pipeline;

            let task = AsyncComputeTaskPool::get().spawn(async move {
                let mut builder = BodyBuilder::new(
                    radius_m,
                    seed,
                    composition,
                    cubemap_resolution,
                    body_age_gyr,
                    tidal_axis,
                    axial_tilt_rad,
                );
                let stages = pipeline_defs
                    .into_iter()
                    .map(|s| s.into_stage())
                    .collect::<Vec<_>>();
                Pipeline::new(stages).run(&mut builder);
                builder.build()
            });

            // Placeholder: same plain-sphere look as the non-procedural branch
            // so the body is visible immediately at roughly the right size and
            // colour while the terrain pipeline runs in the background.
            let sphere_mesh = meshes.add(Sphere::new(render_radius).mesh().ico(4).unwrap());
            let placeholder_mat = std_materials.add(StandardMaterial {
                base_color: Color::srgb(r * body.albedo, g * body.albedo, b * body.albedo),
                perceptual_roughness: 0.9,
                metallic: 0.0,
                ..default()
            });

            let body_entity = commands
                .spawn((
                    Transform::from_translation(pos),
                    Visibility::Inherited,
                    CelestialBody {
                        body_id: body.id,
                        is_star,
                        render_radius,
                        radius_m: body.radius_m,
                    },
                    Name::new(body.name.clone()),
                ))
                .id();

            // Moons with a tidal axis and a parent body are rendered tidally
            // locked: `update_planet_orientations` rewrites the material's
            // orientation quaternion each frame so the baked near-side keeps
            // facing the parent.
            if tidal_axis.is_some()
                && let Some(parent_id) = body.parent
            {
                commands
                    .entity(body_entity)
                    .insert(TidallyLocked { parent_id });
            }

            let mesh_entity = commands
                .spawn((
                    Mesh3d(sphere_mesh),
                    MeshMaterial3d(placeholder_mat),
                    BodyMesh,
                    ChildOf(body_entity),
                ))
                .id();

            commands.spawn((
                Mesh3d(icon_mesh.clone()),
                MeshMaterial3d(icon_material),
                Transform::default(),
                Visibility::Hidden,
                BodyIcon,
                ChildOf(body_entity),
            ));

            commands
                .entity(body_entity)
                .insert(PendingPlanetGeneration {
                    task,
                    body_id: body.id,
                    render_radius,
                    mesh_entity,
                });
        } else if let Some(atmos) = &body.atmosphere {
            // Gas / ice giant path. No terrain bake, no placeholder
            // swap: spawn the billboard + GasGiantMaterial directly.
            // Per-frame updates flow through `update_gas_giant_params`
            // exactly like `update_planet_light_dirs` does for baked
            // bodies.
            let meters_per_render_unit = (1.0 / RENDER_SCALE) as f32;
            let layers = GasGiantLayers::from_params(atmos, meters_per_render_unit);

            let gas_material = gas_giant_materials.add(GasGiantMaterial {
                params: GasGiantParams {
                    radius: render_radius,
                    light_intensity: LIGHT_AT_1AU,
                    // Match the `PLANET_AMBIENT = 0.0` convention used
                    // by baked planet impostors so gas giants get the
                    // same physically-dark shadowed hemisphere.
                    ambient_intensity: PLANET_AMBIENT,
                    rotation_phase: 0.0,
                    light_dir: Vec4::new(0.0, 1.0, 0.0, 0.0),
                    orientation: Vec4::new(0.0, 0.0, 0.0, 1.0),
                },
                layers,
            });

            let body_entity = commands
                .spawn((
                    Transform::from_translation(pos),
                    Visibility::Inherited,
                    CelestialBody {
                        body_id: body.id,
                        is_star,
                        render_radius,
                        radius_m: body.radius_m,
                    },
                    Name::new(body.name.clone()),
                ))
                .id();

            commands.spawn((
                Mesh3d(billboard_mesh.clone()),
                MeshMaterial3d(gas_material.clone()),
                BodyMesh,
                NotShadowCaster,
                NotShadowReceiver,
                ChildOf(body_entity),
            ));

            commands.spawn((
                Mesh3d(icon_mesh.clone()),
                MeshMaterial3d(icon_material),
                Transform::default(),
                Visibility::Hidden,
                BodyIcon,
                ChildOf(body_entity),
            ));

            commands
                .entity(body_entity)
                .insert(GasGiantMaterialHandle(gas_material));

            // ── Ring system ─────────────────────────────────────
            //
            // If the atmosphere authors a ring system, spawn a
            // child entity carrying the annulus mesh and a
            // dedicated `RingMaterial`. The ring child inherits
            // the body's translation from the ECS hierarchy, then
            // applies the body's axial tilt locally so the ring
            // plane is aligned with the body equator. Per-frame
            // updates flow through `update_ring_params`.
            if let Some(rings) = &atmos.rings {
                let inner_r = rings.inner_radius_m / meters_per_render_unit;
                let outer_r = rings.outer_radius_m / meters_per_render_unit;
                let ring_mesh = meshes.add(build_ring_mesh(inner_r, outer_r, 512));

                let ring_layers = RingLayers::from_system(rings);
                let ring_material = ring_materials.add(RingMaterial {
                    params: RingParams {
                        light_dir: Vec4::new(0.0, 1.0, 0.0, 0.0),
                        planet_center_radius: Vec4::new(
                            pos.x,
                            pos.y,
                            pos.z,
                            render_radius,
                        ),
                        light_intensity: LIGHT_AT_1AU,
                        ambient_intensity: PLANET_AMBIENT,
                        inner_radius: inner_r,
                        outer_radius: outer_r,
                    },
                    layers: ring_layers,
                });

                // Ring child rotation is the INVERSE of the gas
                // giant's `orientation = Rx(+tilt)`, because the
                // cloud shader treats `orientation` as the
                // world→body-local transform. That means the body's
                // world-space equatorial plane normal is
                // `Rx(-tilt) * (0,1,0)`, and the ring mesh — built
                // with its geometric normal at +Y — needs `Rx(-tilt)`
                // applied so it aligns with that world-space plane.
                // If this is ever changed, update the ring-shadow
                // test in `gas_giant.wgsl` to match — both sides
                // must agree on the same plane.
                let tilt = body.axial_tilt_rad as f32;
                commands.spawn((
                    Mesh3d(ring_mesh),
                    MeshMaterial3d(ring_material.clone()),
                    Transform::from_rotation(Quat::from_rotation_x(-tilt)),
                    NotShadowCaster,
                    NotShadowReceiver,
                    ChildOf(body_entity),
                    RingMaterialHandle(ring_material),
                ));
            }
        } else {
            // Non-procedural body: plain icosphere with StandardMaterial.
            // No surface generator has been wired up for this body yet, so
            // it shows as a solid-color ball matching the RON `physical`
            // block — exactly the pre-migration behavior.
            let sphere_mesh = meshes.add(Sphere::new(render_radius).mesh().ico(4).unwrap());
            let sphere_material = std_materials.add(StandardMaterial {
                base_color: Color::srgb(r * body.albedo, g * body.albedo, b * body.albedo),
                perceptual_roughness: 0.9,
                metallic: 0.0,
                ..default()
            });

            commands
                .spawn((
                    Transform::from_translation(pos),
                    Visibility::Inherited,
                    CelestialBody {
                        body_id: body.id,
                        is_star,
                        render_radius,
                        radius_m: body.radius_m,
                    },
                    Name::new(body.name.clone()),
                ))
                .with_children(|parent| {
                    parent.spawn((
                        Mesh3d(sphere_mesh),
                        MeshMaterial3d(sphere_material),
                        BodyMesh,
                    ));
                })
                .with_child((
                    Mesh3d(icon_mesh.clone()),
                    MeshMaterial3d(icon_material),
                    Transform::default(),
                    Visibility::Hidden,
                    BodyIcon,
                ));
        }
    }

    // Ship marker: screen-stable billboard circle, white.
    let ship_pos = to_render_pos(sim.simulation.ship_state().position);
    let ship_icon = meshes.add(Circle::new(1.0));
    let ship_material = std_materials.add(StandardMaterial {
        base_color: Color::WHITE,
        emissive: LinearRgba::WHITE * 2.0,
        unlit: true,
        double_sided: true,
        // Push the ship marker in front of every planet/billboard so it
        // never z-fights with a body that happens to share its depth.
        depth_bias: 1.0e9,
        ..default()
    });

    commands.spawn((
        Mesh3d(ship_icon),
        MeshMaterial3d(ship_material),
        Transform::from_translation(ship_pos),
        ShipMarker,
        Name::new("Ship"),
    ));

    // Directional light simulating sunlight. Direction is updated per-frame
    // by `update_sun_light` to point from the star toward the camera focus body.
    // Using a DirectionalLight with cascaded shadow maps instead of a PointLight
    // because Bevy's point light can't handle solar-system-scale distances.
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            color: Color::WHITE,
            shadows_enabled: true,
            shadow_depth_bias: 2.0,
            shadow_normal_bias: 2.0,
            ..default()
        },
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            minimum_distance: 0.1,
            maximum_distance: 100_000.0,
            first_cascade_far_bound: 10.0,
            overlap_proportion: 0.2,
        }
        .build(),
        Transform::default(),
        SunLight,
    ));

    // Dim ambient light so shadowed sides of planets aren't pitch black.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 50.0,
        ..default()
    });
}

// ---------------------------------------------------------------------------
// Per-frame: finalise async terrain generation
// ---------------------------------------------------------------------------

/// Poll in-flight terrain tasks. When one completes, bake the result into GPU
/// textures, build the `PlanetMaterial`, and swap the body's placeholder sphere
/// for the impostor billboard.
fn finalize_planet_generation(
    mut commands: Commands,
    mut pending_q: Query<(Entity, &mut PendingPlanetGeneration)>,
    sim: Res<SimulationState>,
    shared: Res<SharedPlanetMeshes>,
    mut planet_materials: ResMut<Assets<PlanetMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    for (entity, mut pending) in &mut pending_q {
        let _span = tracing::info_span!("finalize_planet_generation").entered();
        let Some(baked) = block_on(poll_once(&mut pending.task)) else {
            continue;
        };

        let body = &sim.system.bodies[pending.body_id];
        let detail =
            PlanetDetailParams::from_body(&baked.detail_params, baked.cubemap_bake_threshold_m);
        let height_range = baked.height_range;
        let textures = bake_from_body_data(&baked, &mut images, &mut storage_buffers);

        let roughness = body_surface_roughness(body);
        let mat_handle = planet_materials.add(PlanetMaterial {
            params: PlanetParams {
                radius: pending.render_radius,
                light_intensity: LIGHT_AT_1AU,
                ambient_intensity: PLANET_AMBIENT,
                height_range,
                light_dir: Vec4::new(0.0, 1.0, 0.0, roughness),
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
                Mesh3d(shared.billboard.clone()),
                MeshMaterial3d(mat_handle.clone()),
            ))
            .remove::<MeshMaterial3d<StandardMaterial>>();

        commands
            .entity(entity)
            .insert(PlanetMaterialHandle(mat_handle))
            .remove::<PendingPlanetGeneration>();
    }
}

// ---------------------------------------------------------------------------
// Per-frame: update planet impostor light directions
// ---------------------------------------------------------------------------

/// Updates each planet material's `light_dir` uniform to point from the body
/// toward the star.  Must run after `cache_body_states`.
/// Update the `CameraExposure` resource from the current focus body. This is
/// the single source of truth for how much gain the "camera" applies to the
/// raw inverse-square solar flux each body sees. Runs once per frame after
/// `cache_body_states`, before any consumer reads `CameraExposure`.
fn update_camera_exposure(
    cache: Res<FrameBodyStates>,
    focus: Res<CameraFocus>,
    bodies: Query<&CelestialBody>,
    mut exposure: ResMut<CameraExposure>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let star_pos = states.first().map(|s| s.position).unwrap_or_default();

    let focus_dist_m = focus
        .target
        .and_then(|e| bodies.get(e).ok())
        .filter(|b| !b.is_star)
        .and_then(|b| states.get(b.body_id))
        .map(|s| (s.position - star_pos).length())
        .unwrap_or(AU_M);

    let focus_d_au = (focus_dist_m / AU_M).max(1.0e-3);
    let gain = focus_d_au.powf(EXPOSURE_ALPHA) as f32;

    exposure.focus_dist_m = focus_dist_m;
    exposure.gain = gain;
    exposure.ev = gain.max(1.0e-6).log2();
}

/// Drive per-camera film grain strength from the current exposure push. When
/// the exposure system is lifting a dark outer-system scene by several EV,
/// that's equivalent to running a real sensor at high ISO: the visible result
/// is more grain. We add grain proportional to the positive EV push so Nyx
/// reads as "dim, sensor-limited" rather than "just another 1 AU body in
/// weird light."
fn sync_film_grain_to_exposure(exposure: Res<CameraExposure>, mut grains: Query<&mut FilmGrain>) {
    // Only positive EV adds grain. Pulling bright scenes down (inner-system
    // focus) doesn't add noise in a real sensor.
    let push_ev = exposure.ev.max(0.0);
    let normalized = (push_ev / EXPOSURE_EV_GRAIN_MAX).clamp(0.0, 1.0);
    const BASE_INTENSITY: f32 = 0.020;
    const MAX_EXTRA: f32 = 0.100;
    let target = BASE_INTENSITY + normalized * MAX_EXTRA;
    for mut grain in &mut grains {
        grain.intensity = target;
    }
}

fn update_planet_light_dirs(
    query: Query<(&CelestialBody, &PlanetMaterialHandle)>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    cache: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    sim: Res<SimulationState>,
    exposure: Res<CameraExposure>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let star_pos = states.first().map(|s| s.position).unwrap_or_default();
    let body_defs = sim.simulation.bodies();
    let gain = exposure.gain;

    // Collect eclipse-occluder candidates once. Only non-star bodies with
    // non-trivial render radius count — tiny comets add cost without ever
    // producing a visible eclipse.
    let mut occluders: Vec<(usize, Vec3, f32)> = Vec::new();
    for (body, _) in &query {
        if body.is_star || body.render_radius < 0.001 {
            continue;
        }
        let Some(state) = states.get(body.body_id) else {
            continue;
        };
        let render_pos = to_render_pos(state.position - origin.position);
        occluders.push((body.body_id, render_pos, body.render_radius));
    }

    for (body, handle) in &query {
        let Some(mat) = materials.get_mut(&handle.0) else {
            continue;
        };
        let body_pos = states
            .get(body.body_id)
            .map(|s| s.position)
            .unwrap_or_default();
        let offset = star_pos - body_pos;
        let distance_m = offset.length();
        let to_star = if distance_m > 0.0 {
            (offset / distance_m).as_vec3()
        } else {
            Vec3::Y
        };
        // Preserve w (surface roughness) — only update the direction xyz.
        let roughness = mat.params.light_dir.w;
        mat.params.light_dir = Vec4::new(to_star.x, to_star.y, to_star.z, roughness);
        // Raw inverse-square flux × exposure gain. Exposure math lives in
        // `update_camera_exposure`; this stage just multiplies.
        let au_over_d = AU_M / distance_m.max(1.0);
        let raw = (au_over_d * au_over_d) as f32;
        mat.params.light_intensity = LIGHT_AT_1AU * raw * gain;

        // Fill eclipse occluder list from all other visible bodies.
        mat.params.occluders = [Vec4::ZERO; thalos_planet_rendering::MAX_ECLIPSE_OCCLUDERS];
        let mut count = 0usize;
        for (other_id, pos, radius) in &occluders {
            if *other_id == body.body_id {
                continue;
            }
            if count >= thalos_planet_rendering::MAX_ECLIPSE_OCCLUDERS {
                break;
            }
            mat.params.occluders[count] = Vec4::new(pos.x, pos.y, pos.z, *radius);
            count += 1;
        }
        mat.params.occluder_count = count as u32;

        // Planetshine: pick the orbital parent, skipping the star. The
        // parent's Bond albedo × color is the effective reflected tint; its
        // render-space position and radius go in `parent_pos`.
        mat.params.parent_pos = Vec4::ZERO;
        mat.params.parent_tint = Vec4::ZERO;
        let body_def = &body_defs[body.body_id];
        if let Some(parent_id) = body_def.parent {
            let parent_def = &body_defs[parent_id];
            if !matches!(parent_def.kind, thalos_physics::types::BodyKind::Star) {
                if let Some(parent_state) = states.get(parent_id) {
                    let parent_render_pos = to_render_pos(parent_state.position - origin.position);
                    let parent_radius = (parent_def.radius_m * RENDER_SCALE) as f32;
                    let tint = Vec3::new(
                        parent_def.color[0],
                        parent_def.color[1],
                        parent_def.color[2],
                    ) * parent_def.albedo;
                    mat.params.parent_pos = Vec4::new(
                        parent_render_pos.x,
                        parent_render_pos.y,
                        parent_render_pos.z,
                        parent_radius,
                    );
                    mat.params.parent_tint = Vec4::new(tint.x, tint.y, tint.z, 1.0);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame: tidal-lock planet orientations
// ---------------------------------------------------------------------------

/// For each tidally-locked body, rewrite its material's orientation
/// quaternion so the baked local +Z axis points toward the parent body. The
/// terrain pipeline places mare / tidal asymmetry on +Z (`BodyBuilder::tidal_axis`);
/// this system is what turns that baked data into a visible tidal lock.
fn update_planet_orientations(
    query: Query<(&CelestialBody, &TidallyLocked, &PlanetMaterialHandle)>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    cache: Res<FrameBodyStates>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    for (body, lock, handle) in &query {
        let Some(mat) = materials.get_mut(&handle.0) else {
            continue;
        };
        let Some(body_state) = states.get(body.body_id) else {
            continue;
        };
        let Some(parent_state) = states.get(lock.parent_id) else {
            continue;
        };

        let offset = parent_state.position - body_state.position;
        let len = offset.length();
        if len < 1.0 {
            continue;
        }
        let dir = (offset / len).as_vec3();
        // `from_rotation_arc` produces the shortest rotation that maps `dir`
        // onto +Z. glam's impl handles the antiparallel case with a stable
        // fallback axis, so there's no degenerate pole for the moon's orbit.
        let q = Quat::from_rotation_arc(dir, Vec3::Z);
        mat.params.orientation = Vec4::new(q.x, q.y, q.z, q.w);
    }
}

// ---------------------------------------------------------------------------
// Per-frame: gas giant parameter update
// ---------------------------------------------------------------------------

/// Push camera/light/rotation state into every `GasGiantMaterial` each
/// frame. Mirrors `update_planet_light_dirs` for baked planets but
/// operates on the smaller `GasGiantParams` uniform.
///
/// Keeping this in its own system lets the scheduler parallelise with
/// the terrestrial path — the two queries are disjoint.
fn update_gas_giant_params(
    query: Query<(&CelestialBody, &GasGiantMaterialHandle)>,
    mut materials: ResMut<Assets<GasGiantMaterial>>,
    cache: Res<FrameBodyStates>,
    sim: Res<SimulationState>,
    exposure: Res<CameraExposure>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let star_pos = states.first().map(|s| s.position).unwrap_or_default();
    let body_defs = sim.simulation.bodies();
    let sim_time = sim.simulation.sim_time();
    let gain = exposure.gain;

    for (body, handle) in &query {
        let Some(mat) = materials.get_mut(&handle.0) else {
            continue;
        };

        // Radius can change if something later rescales the render unit;
        // rewrite every frame to stay in sync.
        mat.params.radius = body.render_radius;

        // Light direction: unit vector from body toward star.
        let body_pos = states
            .get(body.body_id)
            .map(|s| s.position)
            .unwrap_or_default();
        let offset = star_pos - body_pos;
        let distance_m = offset.length();
        let to_star = if distance_m > 0.0 {
            (offset / distance_m).as_vec3()
        } else {
            Vec3::Y
        };
        // w carries raw sim seconds — the gas-giant shader uses it for
        // differential rotation scroll, edge-wave phase, and edge
        // vortex chain epoch hashing. Modulo a day-scale period so the
        // f32 stays precise over long sessions.
        let time_mod = (sim_time % 86_400.0) as f32;
        mat.params.light_dir = Vec4::new(to_star.x, to_star.y, to_star.z, time_mod);

        // Inverse-square flux × exposure gain — same contract as
        // `update_planet_light_dirs` so gas giants share the scene's
        // overall exposure calibration.
        let au_over_d = AU_M / distance_m.max(1.0);
        let raw = (au_over_d * au_over_d) as f32;
        mat.params.light_intensity = LIGHT_AT_1AU * raw * gain;

        // Rotation phase: advance bands at the body's real rotation
        // rate. sim_time is seconds, rotation_period_s is seconds, so
        // the modulo drops the large integer part before conversion
        // to f32 and keeps precision high at long run times.
        let body_def = &body_defs[body.body_id];
        let period = body_def.rotation_period_s.max(1.0);
        let phase = ((sim_time % period) / period) as f32 * std::f32::consts::TAU;
        mat.params.rotation_phase = phase;

        // Orientation: axial tilt around the X axis. Gas giants aren't
        // tidally locked, so rotation is already folded into the band
        // phase above; the quaternion here only carries the tilt.
        let tilt = body_def.axial_tilt_rad as f32;
        let q = Quat::from_rotation_x(tilt);
        mat.params.orientation = Vec4::new(q.x, q.y, q.z, q.w);
    }
}

/// Per-frame `RingParams` update.
///
/// Ring materials need two pieces of live state:
///
/// 1. **Sun direction** — the shader uses it for Lambert + forward
///    scatter and for the planet-shadow ray test.
/// 2. **Planet center in world space** — the ring mesh is a child of
///    the body entity, so the body moves, so the center used by the
///    shadow ray must move with it.
///
/// Light intensity is re-exposed each frame against the current
/// camera exposure gain, matching `update_gas_giant_params` so the
/// ring and disk stay photometrically consistent.
fn update_ring_params(
    ring_query: Query<(&ChildOf, &RingMaterialHandle)>,
    body_query: Query<&CelestialBody>,
    mut materials: ResMut<Assets<RingMaterial>>,
    origin: Res<RenderOrigin>,
    cache: Res<FrameBodyStates>,
    exposure: Res<CameraExposure>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let star_pos = states.first().map(|s| s.position).unwrap_or_default();
    let gain = exposure.gain;

    for (parent, handle) in &ring_query {
        let Ok(body) = body_query.get(parent.0) else {
            continue;
        };
        let Some(mat) = materials.get_mut(&handle.0) else {
            continue;
        };

        let body_pos_m = states
            .get(body.body_id)
            .map(|s| s.position)
            .unwrap_or_default();
        let offset = star_pos - body_pos_m;
        let distance_m = offset.length();
        let to_star = if distance_m > 0.0 {
            (offset / distance_m).as_vec3()
        } else {
            Vec3::Y
        };

        // Planet center in the render frame — same transform
        // `update_body_positions` uses, so the shadow ray tests the
        // right sphere regardless of the rolling render origin.
        let center_render = to_render_pos(body_pos_m - origin.position);
        mat.params.planet_center_radius = Vec4::new(
            center_render.x,
            center_render.y,
            center_render.z,
            body.render_radius,
        );
        mat.params.light_dir = Vec4::new(to_star.x, to_star.y, to_star.z, 0.0);

        let au_over_d = AU_M / distance_m.max(1.0);
        let raw = (au_over_d * au_over_d) as f32;
        mat.params.light_intensity = LIGHT_AT_1AU * raw * gain;
    }
}

// ---------------------------------------------------------------------------
// Periodic orbit trail recomputation
// ---------------------------------------------------------------------------

/// Recompute orbit trails when sim time has advanced enough, or on first frame.
fn recompute_orbit_trails(
    mut commands: Commands,
    sim: Res<SimulationState>,
    existing: Option<Res<OrbitLines>>,
) {
    let sim_time = sim.simulation.sim_time();

    if let Some(ref orbit_lines) = existing {
        let elapsed = sim_time - orbit_lines.last_compute_time;
        if elapsed < ORBIT_TRAIL_RECOMPUTE_INTERVAL {
            return;
        }
    }

    let trails = sim.simulation.body_orbit_trails(ORBIT_SAMPLES);
    let bodies = sim.simulation.bodies();

    let lines: Vec<Option<OrbitLine>> = trails
        .into_iter()
        .enumerate()
        .map(|(i, trail)| {
            let trail = trail?;
            let body = &bodies[i];
            let [r, g, b] = body.color;
            let orbit_color = Color::linear_rgba(
                srgb_to_linear(r) * 0.4,
                srgb_to_linear(g) * 0.4,
                srgb_to_linear(b) * 0.4,
                0.6,
            );
            Some(OrbitLine {
                points: trail.points.iter().map(|p| to_render_pos(*p)).collect(),
                color: orbit_color,
                parent_id: trail.parent_id,
            })
        })
        .collect();

    commands.insert_resource(OrbitLines {
        lines,
        last_compute_time: sim_time,
    });
}

// ---------------------------------------------------------------------------
// Per-frame: floating origin & transform updates
// ---------------------------------------------------------------------------

/// Sets the render origin to the camera focus body's position so that nearby
/// objects always have small render-space coordinates (full f32 precision).
/// Applies hysteresis: only updates when the focus body has moved more than
/// `ORIGIN_UPDATE_THRESHOLD` render units from the current origin, or when
/// the focus entity changes.
fn update_render_origin(
    cache: Res<FrameBodyStates>,
    focus: Res<CameraFocus>,
    bodies: Query<&CelestialBody>,
    mut origin: ResMut<RenderOrigin>,
    mut prev_focus: ResMut<PreviousFocusEntity>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    let candidate = focus
        .target
        .and_then(|e| bodies.get(e).ok())
        .and_then(|b| states.get(b.body_id))
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);

    let focus_changed = focus.target != prev_focus.entity;
    if focus_changed {
        prev_focus.entity = focus.target;
    }

    let delta_render = (candidate - origin.position) * RENDER_SCALE;
    if focus_changed || delta_render.length() > ORIGIN_UPDATE_THRESHOLD {
        origin.position = candidate;
    }
}

fn update_body_positions(
    cache: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    mut query: Query<(&CelestialBody, &mut Transform)>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    for (body, mut transform) in &mut query {
        if let Some(state) = states.get(body.body_id) {
            transform.translation = to_render_pos(state.position - origin.position);
        }
    }
}

/// Point the directional sun light from the star toward the camera's focus body.
fn update_sun_light(
    cache: Res<FrameBodyStates>,
    focus: Res<CameraFocus>,
    bodies: Query<&CelestialBody>,
    mut light_query: Query<&mut Transform, With<SunLight>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    // Find the focus body's physics-space position.
    let focus_pos = focus
        .target
        .and_then(|e| bodies.get(e).ok())
        .and_then(|b| states.get(b.body_id))
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);

    // Star is always at index 0.
    let star_pos = states
        .get(0)
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);

    let dir = (focus_pos - star_pos).normalize();
    if dir.length_squared() < 0.5 {
        return; // Focus is on the star itself; direction undefined.
    }

    let dir_f32 = dir.as_vec3();
    for mut transform in &mut light_query {
        // DirectionalLight shines along its local -Z, so we look in the light's travel direction.
        transform.look_to(dir_f32, Vec3::Y);
    }
}

fn update_ship_position(
    sim: Res<SimulationState>,
    origin: Res<RenderOrigin>,
    focus: Res<CameraFocus>,
    camera_query: Query<&Transform, With<OrbitCamera>>,
    mut query: Query<&mut Transform, (With<ShipMarker>, Without<OrbitCamera>)>,
) {
    let Ok(cam_tf) = camera_query.single() else {
        return;
    };
    let cam_render_dist = (focus.distance * RENDER_SCALE) as f32;
    let icon_radius = cam_render_dist * MARKER_RADIUS;

    for mut transform in &mut query {
        transform.translation =
            to_render_pos(sim.simulation.ship_state().position - origin.position);
        transform.rotation = cam_tf.rotation;
        transform.scale = Vec3::splat(icon_radius);
    }
}

// ---------------------------------------------------------------------------
// Per-frame: draw precomputed orbit lines with Gizmos
// ---------------------------------------------------------------------------

/// Distance-to-parent / camera-distance ratio at which orbit trails start fading.
const ORBIT_FADE_START: f64 = 20.0;
/// Ratio at which orbit trails are fully hidden.
const ORBIT_FADE_END: f64 = 100.0;

/// Aggressive fade for the focus body's own orbit and its siblings (bodies
/// sharing the same parent as the focus). Expressed as
/// `focus_orbit_radius / cam_dist` — when the camera is well inside the
/// focus body's orbit around its parent, these lines clutter the view.
const SIBLING_FADE_START: f64 = 3.0;
const SIBLING_FADE_END: f64 = 10.0;

fn draw_orbits(
    mut gizmos: Gizmos,
    orbit_lines: Option<Res<OrbitLines>>,
    cache: Res<FrameBodyStates>,
    origin: Res<RenderOrigin>,
    focus: Res<CameraFocus>,
    bodies: Query<&CelestialBody>,
    show_orbits: Res<ShowOrbits>,
    sim: Option<Res<SimulationState>>,
) {
    if !show_orbits.0 {
        return;
    }
    let Some(orbit_lines) = orbit_lines else {
        return;
    };
    let Some(ref states) = cache.states else {
        return;
    };
    let Some(sim) = sim else {
        return;
    };

    // Focus body id and its parent id, if any.
    let focus_body_id = focus
        .target
        .and_then(|e| bodies.get(e).ok())
        .map(|b| b.body_id);
    let focus_parent_id = focus_body_id.and_then(|id| sim.simulation.bodies()[id].parent);

    // Determine the camera focus position in metres.
    let focus_pos = focus_body_id
        .and_then(|id| states.get(id))
        .map(|s| s.position)
        .unwrap_or(bevy::math::DVec3::ZERO);

    let cam_dist = focus.distance;

    for (i, line) in orbit_lines.lines.iter().enumerate() {
        let Some(line) = line else { continue };

        let parent_pos_m = states
            .get(line.parent_id)
            .map(|s| s.position)
            .unwrap_or(bevy::math::DVec3::ZERO);
        let dist_to_parent = (parent_pos_m - focus_pos).length();
        let ratio = dist_to_parent / cam_dist;

        // Focus body's own orbit, or a sibling (same parent). Apply an
        // aggressive fade so close-up views aren't cluttered by the orbit
        // ring cutting through the focus body.
        let is_self_or_sibling = Some(i) == focus_body_id
            || (focus_parent_id.is_some() && Some(line.parent_id) == focus_parent_id);

        let (fade_start, fade_end) = if is_self_or_sibling {
            (SIBLING_FADE_START, SIBLING_FADE_END)
        } else {
            (ORBIT_FADE_START, ORBIT_FADE_END)
        };

        if ratio > fade_end {
            continue;
        }

        let parent_render_pos = to_render_pos(parent_pos_m - origin.position);

        if ratio > fade_start {
            let t = (ratio - fade_start) / (fade_end - fade_start);
            let alpha = (1.0 - t) as f32;
            let faded = line.color.with_alpha(line.color.alpha() * alpha);
            gizmos.linestrip(line.points.iter().map(|p| *p + parent_render_pos), faded);
        } else {
            gizmos.linestrip(
                line.points.iter().map(|p| *p + parent_render_pos),
                line.color,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame: toggle body mesh vs icon based on screen-space size
// ---------------------------------------------------------------------------

type IconFilter = (With<BodyIcon>, Without<CelestialBody>, Without<OrbitCamera>);

/// Width of the crossfade window as a multiple of `icon_radius`. When the
/// body's render radius is between `icon_radius` and `icon_radius * (1 +
/// ICON_FADE_WIDTH)`, the icon alpha smoothly ramps from 0 to 1 while the
/// impostor mesh stays on top. Below `icon_radius` the mesh is hidden and
/// the icon is already fully opaque, so the swap is invisible.
const ICON_FADE_WIDTH: f32 = 0.25;

/// Multiple of `focus.distance` beyond which a body's billboard is hidden.
/// Bodies farther than this from the focus target (in render units) are
/// considered "out of the current neighborhood" — e.g. when framing Thalos
/// at ~20,000 km zoom, Auron is tens of millions of km away and its
/// billboard would just add clutter.
const BILLBOARD_NEIGHBORHOOD: f32 = 30.0;

/// Width of the moon-vs-parent merge fade as a multiple of the combined
/// icon radii. Moon alpha reaches 0 when its world-space separation from
/// the parent drops below `parent_icon_r + moon_icon_r` (i.e. the two icon
/// discs overlap and the moon can no longer be clicked separately), and
/// reaches 1 at `(1 + SEPARATION_FADE_WIDTH)` times that threshold.
const SEPARATION_FADE_WIDTH: f32 = 1.5;

/// Alpha ramp for moon billboards based on angular separation from parent.
/// Returns 1.0 when moon is clearly separable, 0.0 when icons fully merged.
fn moon_separation_alpha(moon_pos: Vec3, parent_pos: Vec3, cam_pos: Vec3) -> f32 {
    let moon_r = (moon_pos - cam_pos).length().max(1.0) * MARKER_RADIUS;
    let parent_r = (parent_pos - cam_pos).length().max(1.0) * MARKER_RADIUS;
    let merged = moon_r + parent_r;
    let fade_end = merged * (1.0 + SEPARATION_FADE_WIDTH);
    let sep = (moon_pos - parent_pos).length();
    ((sep - merged) / (fade_end - merged)).clamp(0.0, 1.0)
}
type MeshFilter = (
    With<BodyMesh>,
    Without<BodyIcon>,
    Without<CelestialBody>,
    Without<OrbitCamera>,
);

fn sync_body_icons(
    bodies: Query<(Entity, &CelestialBody, &Transform, &Children)>,
    sim: Res<SimulationState>,
    focus: Res<CameraFocus>,
    camera_query: Query<&Transform, With<OrbitCamera>>,
    mut icons: Query<
        (
            &mut Transform,
            &mut Visibility,
            &MeshMaterial3d<StandardMaterial>,
        ),
        IconFilter,
    >,
    mut meshes: Query<&mut Visibility, MeshFilter>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
) {
    let Ok(cam_tf) = camera_query.single() else {
        return;
    };

    let cam_rotation = cam_tf.rotation;
    let cam_pos = cam_tf.translation;
    let body_defs = sim.simulation.bodies();

    // Per-body camera distance: each icon has a constant screen size, so the
    // icon's world-space radius must scale with that body's own distance to
    // the camera — not the focus distance shared across the whole scene.
    let body_cam_dist = |tf: &Transform| (tf.translation - cam_pos).length().max(1.0);

    // Billboard neighborhood: anything farther from the focus target than
    // `focus.distance * BILLBOARD_NEIGHBORHOOD` (all in render units) is
    // outside the current view's scope and gets hidden.
    let focus_pos = focus
        .target
        .and_then(|e| bodies.get(e).ok())
        .map(|(_, _, tf, _)| tf.translation);
    let neighborhood_radius = (focus.distance * RENDER_SCALE) as f32 * BILLBOARD_NEIGHBORHOOD;

    for (entity, body, body_tf, children) in &bodies {
        let icon_radius = body_cam_dist(body_tf) * MARKER_RADIUS;
        let is_focus = focus.target == Some(entity);

        // Fade moons when their icon disc overlaps the parent's: at that
        // point the user can no longer click the moon separately from the
        // parent, so rendering it adds clutter. Focus is exempt so zooming
        // out while focused on a moon doesn't hide the focus target.
        let mut hidden = false;
        let mut separation_alpha = 1.0f32;
        if !is_focus
            && matches!(body_defs[body.body_id].kind, BodyKind::Moon)
            && let Some(parent_id) = body_defs[body.body_id].parent
            && let Some((_, _, parent_tf, _)) =
                bodies.iter().find(|(_, b, _, _)| b.body_id == parent_id)
        {
            separation_alpha =
                moon_separation_alpha(body_tf.translation, parent_tf.translation, cam_pos);
            if separation_alpha <= 0.0 {
                hidden = true;
            }
        }

        // Hide distant billboards: if this body is far outside the current
        // focus neighborhood AND would only render as a billboard anyway
        // (impostor radius below icon size), drop it entirely. Bodies that
        // still resolve as a real impostor mesh stay visible — otherwise
        // zooming in on a moon would make its huge parent disappear.
        if !is_focus
            && !body.is_star
            && body.render_radius < icon_radius
            && let Some(fp) = focus_pos
            && (body_tf.translation - fp).length() > neighborhood_radius
        {
            hidden = true;
        }

        // Icon alpha ramps from 0 at `render_radius >= (1 + WIDTH) *
        // icon_radius` to 1 at `render_radius <= icon_radius`. The impostor
        // stays visible throughout the fade window so the two layers
        // crossfade instead of popping.
        let fade_start = icon_radius * (1.0 + ICON_FADE_WIDTH);
        let icon_alpha = if body.is_star || hidden {
            0.0
        } else {
            ((fade_start - body.render_radius) / (fade_start - icon_radius)).clamp(0.0, 1.0)
                * separation_alpha
        };
        let show_icon = !hidden && icon_alpha > 0.0;
        let show_mesh = !hidden && (body.is_star || body.render_radius >= icon_radius);

        for child in children.iter() {
            if let Ok((mut icon_tf, mut icon_vis, mat_handle)) = icons.get_mut(child) {
                let target_vis = if show_icon {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                };
                if *icon_vis != target_vis {
                    *icon_vis = target_vis;
                }
                if show_icon {
                    if icon_tf.rotation != cam_rotation {
                        icon_tf.rotation = cam_rotation;
                    }
                    let target_scale = Vec3::splat(icon_radius);
                    if icon_tf.scale != target_scale {
                        icon_tf.scale = target_scale;
                    }
                    if let Some(mat) = std_materials.get_mut(&mat_handle.0) {
                        let current = mat.base_color.alpha();
                        if (current - icon_alpha).abs() > 1e-3 {
                            mat.base_color.set_alpha(icon_alpha);
                            // Emissive ignores material alpha in the forward
                            // shader, so scale rgb directly to fade glow.
                            let lin = mat.base_color.to_linear();
                            mat.emissive =
                                LinearRgba::new(lin.red, lin.green, lin.blue, 1.0) * 2.0
                                    * icon_alpha;
                        }
                    }
                }
            }
            if let Ok(mut mesh_vis) = meshes.get_mut(child) {
                let target_vis = if show_mesh {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                };
                if *mesh_vis != target_vis {
                    *mesh_vis = target_vis;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Startup: focus camera on homeworld
// ---------------------------------------------------------------------------

fn focus_camera_on_homeworld(
    mut focus: ResMut<CameraFocus>,
    bodies: Query<(Entity, &CelestialBody, &Name)>,
) {
    // Find "Thalos" (homeworld) entity and focus the camera on it.
    for (entity, _body, name) in &bodies {
        if name.as_str() == "Thalos" {
            focus.target = Some(entity);
            focus.distance = 2e7; // 20,000 km — close enough to see the planet
            focus.target_distance = 2e7;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Double-click body to focus camera
// ---------------------------------------------------------------------------

/// Detects double-clicks and focuses the camera on the nearest body by
/// projecting every body's world position to screen space and picking the
/// closest one to the cursor.  Works for both 3D sphere meshes and billboard
/// icons because we test the parent entity's transform, not the mesh child.
fn double_click_focus_system(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<(&Camera, &GlobalTransform), With<OrbitCamera>>,
    bodies: Query<(Entity, &CelestialBody, &Transform)>,
    sim: Res<SimulationState>,
    mut focus: ResMut<CameraFocus>,
    mut last_click: ResMut<LastClick>,
) {
    let focus_target = focus.target;
    let focus_render_dist = (focus.distance * RENDER_SCALE) as f32;
    let neighborhood_radius = focus_render_dist * BILLBOARD_NEIGHBORHOOD;
    let focus_pos = focus_target
        .and_then(|e| bodies.get(e).ok())
        .map(|(_, _, t)| t.translation);
    if !mouse_buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, cam_gt)) = camera_q.single() else {
        return;
    };

    let now = time.elapsed_secs_f64();
    let is_double = (now - last_click.time) < DOUBLE_CLICK_THRESHOLD
        && cursor_pos.distance(last_click.position) < DOUBLE_CLICK_RADIUS;

    last_click.time = now;
    last_click.position = cursor_pos;

    if !is_double {
        return;
    }

    // Reset so a third click doesn't trigger another double-click.
    last_click.time = 0.0;

    // Two-pass pick:
    //   1. Bodies currently rendered as visible billboards (icon alpha > 0,
    //      not hidden as a faraway moon) take priority. Among those, the one
    //      closest to the camera wins — a visible billboard is always the
    //      topmost selection target for any cursor inside its disc.
    //   2. Otherwise, fall back to the nearest-center hit across all bodies.
    let body_defs = sim.simulation.bodies();
    let viewport_height = window.height();
    let fov = std::f32::consts::FRAC_PI_4;
    let half_fov_tan = (fov / 2.0).tan();
    let cam_world = cam_gt.translation();

    let fade_end_factor = 1.0 + ICON_FADE_WIDTH;

    let mut billboard_best: Option<(Entity, f32)> = None; // (entity, cam_dist)
    let mut fallback_best: Option<(Entity, f32)> = None; // (entity, cursor_dist)

    for (entity, body, transform) in &bodies {
        let Ok(screen) = camera.world_to_viewport(cam_gt, transform.translation) else {
            continue;
        };
        let cam_dist = cam_world.distance(transform.translation);
        if cam_dist <= 0.0 {
            continue;
        }
        let pixels_per_unit = viewport_height / (2.0 * half_fov_tan * cam_dist);
        let icon_radius_world = cam_dist * MARKER_RADIUS;
        let icon_radius_px = icon_radius_world * pixels_per_unit;
        let sphere_radius_px = body.render_radius * pixels_per_unit;

        // Same visibility rules as `sync_body_icons`: moons fade out as
        // their icon disc merges with the parent's, and far-away bodies
        // outside the focus neighborhood get dropped. A moon that's fully
        // merged is neither visible nor clickable separately.
        if focus_target != Some(entity)
            && matches!(body_defs[body.body_id].kind, BodyKind::Moon)
            && let Some(parent_id) = body_defs[body.body_id].parent
            && let Some((_, _, parent_tf)) =
                bodies.iter().find(|(_, b, _)| b.body_id == parent_id)
            && moon_separation_alpha(transform.translation, parent_tf.translation, cam_world)
                <= 0.0
        {
            continue;
        }

        // Also skip bodies hidden by the distant-billboard rule (matches
        // `sync_body_icons`): only drop when the body is billboard-sized.
        if focus_target != Some(entity)
            && !body.is_star
            && body.render_radius < icon_radius_world
            && let Some(fp) = focus_pos
            && (transform.translation - fp).length() > neighborhood_radius
        {
            continue;
        }

        let is_visible_billboard =
            !body.is_star && body.render_radius < icon_radius_world * fade_end_factor;

        let dist_px = screen.distance(cursor_pos);

        if is_visible_billboard && dist_px <= icon_radius_px {
            if billboard_best.map(|(_, d)| cam_dist < d).unwrap_or(true) {
                billboard_best = Some((entity, cam_dist));
            }
            continue;
        }

        // Fallback pass: mesh-sized hit circle for bodies not acting as a
        // billboard, plus a 12px minimum so tiny dots stay clickable.
        let hit_radius = sphere_radius_px.max(icon_radius_px).max(12.0);
        if dist_px > hit_radius {
            continue;
        }
        if fallback_best.map(|(_, d)| dist_px < d).unwrap_or(true) {
            fallback_best = Some((entity, dist_px));
        }
    }

    let Some((target_entity, _)) = billboard_best.or(fallback_best) else {
        return;
    };

    // Compute smooth transition offset.
    let old_pos = focus
        .target
        .and_then(|e| bodies.get(e).ok())
        .map(|(_, _, t)| t.translation)
        .unwrap_or(Vec3::ZERO)
        + focus.focus_offset;

    let new_pos = bodies
        .get(target_entity)
        .map(|(_, _, t)| t.translation)
        .unwrap_or(Vec3::ZERO);

    focus.focus_offset = old_pos - new_pos;
    focus.target = Some(target_entity);
}
