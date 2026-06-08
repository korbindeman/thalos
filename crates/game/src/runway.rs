//! A fixed, flat runway on the Thalos surface, plus the two spawn scenarios
//! that put the aircraft on it (`just game runway`, `just game runway-approach`).
//!
//! This is a deferred, terrain-aware spawn just like the descent scenarios in
//! [`crate::spawn`]: `main.rs` parks the ship in the placeholder orbit behind
//! the loading screen, and [`finish_runway_spawn`] installs the real runway +
//! aircraft state on the first `AppState::Running` frame, once the terrain
//! height source is resident.
//!
//! **The terrain itself is flattened into a pad; the runway sits flush on it.**
//! A coarse body-fixed search picks a flat dry low-latitude site, then a single
//! fixed elevation `E = max(natural terrain over the pad) + margin` is chosen.
//! A [`thalos_terrain::TerrainFlatten`] pad is installed via the body's shared
//! [`crate::rendering::ground_terrain::TerrainFlattenRegistry`] handle: the
//! terrain tile provider reads it as it bakes, so the *rendered* ground — and,
//! through the GPU-atlas height mirror, the collider and CPU height queries —
//! level out to `E` across the pad and smoothstep-blend back to natural terrain
//! over a ramp. The pad is set before the aircraft/camera move to the site, so
//! the tiles that stream in there bake flattened from the start.
//!
//! On top of the levelled ground the runway is just a paved strip + markings +
//! posts, lifted a few centimetres so the paving reads on the grass. A flat
//! kinematic collider at `E` (posed each frame like the terrain collider patch)
//! still backs the landing surface so it stays exactly flat regardless of tile
//! residency timing.

use bevy::camera::visibility::RenderLayers;
use bevy::light::NotShadowCaster;
use bevy::camera::primitives::MeshAabb;
use bevy::math::{DMat3, DQuat, DVec3, Vec3, Vec3A};
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};

use thalos_body_render::{HeightSource, TerrainPatchBasis};
use thalos_physics_canonical::body_fixed::{
    body_fixed_pose_from_inertial, body_fixed_surface_velocity,
};
use thalos_physics_canonical::canonical::{AuthorityMode, Epoch, TranslationalState};
use thalos_physics_canonical::types::{AttitudeState, BodyState};
use thalos_physics_local::avian::{
    AngularVelocity, Collider, LinearVelocity, Position, RigidBody, Rotation,
};
use thalos_physics_local::{HeightSourceRegistry, TerrainSurfaceRegistry};
use thalos_terrain::TerrainFlatten;
use thalos_world::{BodyId, StateVector};

use crate::SimStage;
use crate::camera::ShipCamera;
use crate::coords::SHIP_LAYER;
use crate::debug::{DebugLaunchMount, DebugLaunchMountState};
use crate::local_physics::PHYSICS_QUERY_TILE_LOD_M;
use crate::rendering::ground_terrain::TerrainFlattenRegistry;
use crate::rendering::{PlayerShip, RealSpaceBody};
use crate::rendering::real_space::{RealSpaceRoot, real_space_grid};
use crate::solar_system_state::{SimulationState, SolarSystemState};
use crate::spawn::{SpawnSituation, sample_site_relief_m};

// ---------------------------------------------------------------------------
// Runway dimensions (realistic large runway: 3 km × 60 m)
// ---------------------------------------------------------------------------

const RUNWAY_LENGTH_M: f64 = 3000.0;
const RUNWAY_HALF_LENGTH_M: f64 = RUNWAY_LENGTH_M * 0.5;
const RUNWAY_WIDTH_M: f64 = 60.0;
const RUNWAY_HALF_WIDTH_M: f64 = RUNWAY_WIDTH_M * 0.5;

/// The flat terrain pad is levelled to this height above the highest natural
/// terrain across the pad footprint, so nothing pokes through the paving.
const RUNWAY_PLATFORM_MARGIN_M: f64 = 0.4;
/// Flat margin levelled around the painted strip (the "shoulder" of the pad).
/// The terrain inside `runway + this` is flattened to `E`.
const RUNWAY_PAD_MARGIN_M: f64 = 50.0;
/// Width of the graded ramp that blends the flat pad back to natural terrain.
const RUNWAY_PAD_RAMP_M: f64 = 150.0;
/// Asphalt strip sits this far above the flat terrain pad so the paving reads
/// as a surface on the ground (and never z-fights the flattened tiles).
const RUNWAY_ASPHALT_LIFT_M: f64 = 0.12;
/// Markings sit just above the asphalt to avoid z-fighting.
const RUNWAY_MARKING_LIFT_M: f64 = 0.17;

/// Top tessellation: segments along the length / across the width. The slab is
/// flat, so this is only for lighting/curvature — it can be coarse.
const RUNWAY_TOP_SEGMENTS_LEN: usize = 120;
const RUNWAY_TOP_SEGMENTS_W: usize = 4;
/// Subdivision length for marking strips (kept fine so dashes read cleanly).
const RUNWAY_MARKING_SEG_LEN_M: f64 = 25.0;
/// Flat collider tessellation (a trimesh; coarse is fine for a flat surface).
const RUNWAY_COLLIDER_SEGMENTS_LEN: usize = 24;
const RUNWAY_COLLIDER_SEGMENTS_W: usize = 3;

// ---------------------------------------------------------------------------
// Site & heading search (deterministic, body-fixed)
// ---------------------------------------------------------------------------

/// Coarse LOD for the land/flatness search — the baked macro height that
/// defines coastlines and broad relief, not fine procedural texture.
const SITE_SEARCH_LOD_M: f32 = 2000.0;
/// Freeboard above sea level for a sample to count as dry land.
const SITE_FREEBOARD_M: f32 = 50.0;
/// Keep the site well away from the ice caps (sin(lat); ~44°).
const SITE_MAX_ABS_LAT_SIN: f64 = 0.70;
const SITE_LAT_MIN_DEG: f64 = -40.0;
const SITE_LAT_MAX_DEG: f64 = 40.0;
const SITE_LAT_STEP_DEG: f64 = 5.0;
const SITE_LON_STEP_DEG: f64 = 5.0;
/// Flatness probe radius around a candidate site (≈ runway half-length so the
/// whole footprint is judged).
const SITE_PROBE_RADIUS_M: f64 = RUNWAY_HALF_LENGTH_M;

/// Azimuths tried for the takeoff heading (0..π; a runway is symmetric).
const HEADING_AZIMUTH_STEPS: usize = 18;
/// Height samples along the strip per candidate azimuth.
const HEADING_SAMPLES: usize = 13;

/// Footprint grid sampled to find the platform elevation (max/min terrain).
const FOOTPRINT_SAMPLES_LEN: usize = 40;
const FOOTPRINT_SAMPLES_W: usize = 5;

// ---------------------------------------------------------------------------
// Orientation markers (raised edge posts)
// ---------------------------------------------------------------------------

const POST_SPACING_M: f64 = 300.0;
const POST_EDGE_OFFSET_M: f64 = 4.0;
const POST_HEIGHT_M: f32 = 4.0;
const POST_THRESHOLD_HEIGHT_M: f32 = 6.0;
const POST_SIZE_M: f32 = 0.5;

// ---------------------------------------------------------------------------
// Aircraft placement
// ---------------------------------------------------------------------------

/// Small gap left between the craft's lowest point and the platform surface so
/// the gear/belly rests just on top without z-fighting. The real clearance is
/// measured per-craft from its geometry (see [`craft_ground_clearance`]); this
/// is only the sliver above it.
const RUNWAY_GEAR_REST_MARGIN_M: f64 = 0.03;
/// Park this far in from the threshold end so the nose is on the numbers.
const PARK_THRESHOLD_INSET_M: f64 = 150.0;

const APPROACH_BACK_M: f64 = 1500.0;
const APPROACH_ALT_M: f64 = 250.0;
const APPROACH_SPEED_M_S: f64 = 60.0;
const APPROACH_SINK_M_S: f64 = 4.0;

/// Hide the runway beyond this multiple of the body radius (matches the
/// terrain/impostor LOD swap so it isn't a speck poking through the orbital
/// billboard).
const RUNWAY_VIS_RADIUS_FACTOR: f64 = 4.0;

/// The chosen runway, in the body-fixed frame. Inserted once by
/// [`finish_runway_spawn`]; kept around for UI / future reference.
#[derive(Resource, Debug, Clone, Copy)]
pub struct RunwaySite {
    pub body_id: BodyId,
    /// Unit body-fixed direction to the runway centre.
    pub center_dir: DVec3,
    /// Unit body-fixed tangent along the takeoff heading at the centre.
    pub heading_tangent: DVec3,
    /// Flat platform elevation (m above the body reference radius). The whole
    /// runway top is this constant radius — level, not draped.
    pub elevation_m: f64,
}

/// Marker on the runway platform entity (a root-grid big_space child).
/// Children (shoulder, runoff, markings, posts) inherit its visibility and
/// transform. [`update_runway_transform`] positions it in f64 each frame.
#[derive(Component, Debug)]
struct RunwayVisual {
    body_id: BodyId,
    swap_radius_m: f32,
    /// Body-fixed position of the platform centre at elevation `E`.
    center_surface_body: DVec3,
}

/// Marker on the flat kinematic collider entity. Posed each frame so it
/// co-rotates with the body, exactly like the terrain collider patch.
#[derive(Component, Debug)]
struct RunwayCollider {
    body_id: BodyId,
    /// Body-fixed position of the platform centre at elevation `E`.
    center_surface_body_m: DVec3,
}

pub struct RunwayPlugin;

impl Plugin for RunwayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                // Runs during `AppState::Loading` (not gated on `Running`): the
                // site selection + flatten install + aircraft placement happen
                // behind the loading screen so the camera reaches the surface
                // and the terrain streams/settles before the reveal. Self-gated
                // by its `done` local + the height-source residency check.
                finish_runway_spawn.before(SimStage::Physics),
                update_runway_transform
                    .in_set(SimStage::Sync)
                    .after(crate::solar_system_state::sync_solar_system_state),
                sync_runway_visibility
                    .in_set(SimStage::Sync)
                    .after(update_runway_transform),
            ),
        )
        .add_systems(
            Update,
            sync_runway_collider_pose
                .in_set(SimStage::Physics)
                .after(crate::bridge::advance_simulation),
        );
    }
}

/// Deferred finisher: pick the site, build the flat platform + collider, and
/// place the aircraft. Runs once, retrying each frame until the terrain height
/// source is resident.
#[allow(clippy::too_many_arguments)]
fn finish_runway_spawn(
    mut done: Local<bool>,
    situation: Res<SpawnSituation>,
    mut sim: ResMut<SimulationState>,
    mut settle: ResMut<crate::surface_settle::SurfaceSettle>,
    mut launch_mount: ResMut<DebugLaunchMount>,
    height_sources: Res<HeightSourceRegistry>,
    surfaces: Res<TerrainSurfaceRegistry>,
    mut flatten_registry: ResMut<TerrainFlattenRegistry>,
    root: Res<RealSpaceRoot>,
    ship_root_q: Query<(Entity, &GlobalTransform), With<PlayerShip>>,
    children_q: Query<&Children>,
    mesh_q: Query<(&GlobalTransform, &Mesh3d)>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if *done || !situation.is_runway() {
        return;
    }
    let body_id = sim.simulation.dominant_body();
    let Some(height_source) = height_sources.get(body_id) else {
        return; // terrain not resident yet — retry next frame
    };
    let hs = height_source.as_ref();

    // For the parked scenario, measure how far the craft's lowest point sits
    // below its origin so it rests on the surface for *any* craft. Computed
    // before building anything, so a craft whose meshes/AABBs aren't ready yet
    // just retries instead of double-spawning the runway.
    let park_clearance_m = if matches!(*situation, SpawnSituation::Runway) {
        let Ok((ship_entity, ship_gt)) = ship_root_q.single() else {
            return; // ship not spawned yet — retry
        };
        match craft_ground_clearance(ship_entity, ship_gt, &children_q, &mesh_q, &meshes) {
            Some(c) => c + RUNWAY_GEAR_REST_MARGIN_M,
            None => return, // craft geometry not ready yet — retry
        }
    } else {
        0.0
    };

    let body_radius_m = sim.system.bodies[body_id].radius_m;
    let surface = surfaces.get(body_id);
    let sea_level_m = surface.as_ref().and_then(|s| s.static_surface.sea_level_m);

    let (center_dir, relief_m) = find_runway_site(hs, sea_level_m, body_radius_m);
    let center_h = hs
        .sample_height_m(center_dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
        .unwrap_or(0.0) as f64;
    let heading_tangent = choose_runway_heading(hs, center_dir, body_radius_m, center_h);
    let across = center_dir.cross(heading_tangent).normalize();

    // Flat pad half-extents: the painted strip plus the levelled shoulder.
    let pad_half_along = RUNWAY_HALF_LENGTH_M + RUNWAY_PAD_MARGIN_M;
    let pad_half_across = RUNWAY_HALF_WIDTH_M + RUNWAY_PAD_MARGIN_M;

    // Flat pad elevation: above the highest natural terrain across the whole pad
    // (computed from natural terrain, *before* the flatten is installed below) so
    // the levelled ground never has to cut through a peak inside the pad.
    let (max_h, min_h) = footprint_extremes(
        hs,
        center_dir,
        heading_tangent,
        across,
        pad_half_along,
        pad_half_across,
        body_radius_m,
        center_h,
    );
    let elevation_m = max_h + RUNWAY_PLATFORM_MARGIN_M;

    // Install the terrain flatten pad. The terrain provider reads this handle as
    // it bakes tiles, so the rendered ground — and, via the GPU-atlas height
    // mirror, the collider and CPU height queries — level out to `elevation_m`
    // across the pad, smoothstep-blending back to natural terrain over the ramp.
    // Set before placing the aircraft / moving the camera so the tiles that
    // stream in at the site bake flattened from the start.
    let flatten = TerrainFlatten::new(
        center_dir,
        heading_tangent,
        across,
        pad_half_along,
        pad_half_across,
        RUNWAY_PAD_RAMP_M,
        elevation_m,
        body_radius_m,
    );
    if let Ok(mut guard) = flatten_registry.handle(body_id).write() {
        *guard = Some(flatten);
    }

    let site = RunwaySite {
        body_id,
        center_dir,
        heading_tangent,
        elevation_m,
    };
    let frame = RunwayFrame {
        body_id,
        center_dir,
        heading: heading_tangent,
        across,
        body_radius_m,
        elevation_m,
    };

    let lat_deg = center_dir.y.clamp(-1.0, 1.0).asin().to_degrees();
    let lon_deg = center_dir.z.atan2(center_dir.x).to_degrees();
    info!(
        "runway: {} on {} at lat {:.1}°, lon {:.1}° — flat pad at {:.0} m (terrain {:.0}..{:.0} m, site relief {:.0} m), {:.0} m × {:.0} m",
        if matches!(*situation, SpawnSituation::Runway) {
            "parked"
        } else {
            "on approach"
        },
        sim.system.bodies[body_id].name,
        lat_deg,
        lon_deg,
        elevation_m,
        min_h,
        max_h,
        relief_m,
        RUNWAY_LENGTH_M,
        RUNWAY_WIDTH_M,
    );

    // Read the body state once (immutable) before any sim mutation below.
    let epoch = Epoch(sim.simulation.sim_time());
    let body_state = sim.ephemeris.state(body_id, epoch);

    spawn_runway_geometry(
        &mut commands,
        &mut meshes,
        &mut materials,
        &frame,
        (body_radius_m * RUNWAY_VIS_RADIUS_FACTOR) as f32,
        root.entity,
        &body_state,
    );
    spawn_runway_collider(&mut commands, &frame, &body_state);

    match *situation {
        SpawnSituation::Runway => place_parked(
            &mut sim,
            &mut launch_mount,
            &body_state,
            &site,
            body_radius_m,
            park_clearance_m,
        ),
        SpawnSituation::RunwayApproach => {
            place_approach(&mut sim, &body_state, &site, body_radius_m)
        }
        _ => {}
    }

    commands.insert_resource(site);
    *done = true;
    // The surface state + flatten pad are installed and the aircraft is placed:
    // let the settle gate start timing the tile stream at the (now-known) site.
    settle.mark_placed();
}

// ---------------------------------------------------------------------------
// Site & heading search
// ---------------------------------------------------------------------------

fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()).normalize()
}

/// Scan a fixed low-latitude lat/lon grid for the flattest dry-land patch.
/// Returns `(center_dir, relief_m)`. Deterministic and epoch-independent.
fn find_runway_site(
    hs: &dyn HeightSource,
    sea_level_m: Option<f32>,
    body_radius_m: f64,
) -> (DVec3, f32) {
    let Some(sea_level_m) = sea_level_m else {
        return (DVec3::X, 0.0);
    };
    let land_threshold = sea_level_m + SITE_FREEBOARD_M;

    let mut best_dry: Option<(f32, DVec3)> = None; // (relief, dir)
    let mut best_any: Option<(f32, DVec3)> = None; // (height, dir)

    let mut lat = SITE_LAT_MIN_DEG;
    while lat <= SITE_LAT_MAX_DEG {
        let mut lon = 0.0;
        while lon < 360.0 {
            let dir = latlon_dir(lat, lon);
            lon += SITE_LON_STEP_DEG;
            if dir.y.abs() > SITE_MAX_ABS_LAT_SIN {
                continue;
            }
            let Some(h) = hs.sample_height_m(dir.as_vec3(), SITE_SEARCH_LOD_M) else {
                continue;
            };
            if best_any.is_none_or(|(bh, _)| h > bh) {
                best_any = Some((h, dir));
            }
            if h <= land_threshold {
                continue;
            }
            let relief = sample_site_relief_m(
                hs,
                dir,
                body_radius_m,
                SITE_PROBE_RADIUS_M,
                SITE_SEARCH_LOD_M,
            )
            .unwrap_or(f32::INFINITY);
            if best_dry.is_none_or(|(br, _)| relief < br) {
                best_dry = Some((relief, dir));
            }
        }
        lat += SITE_LAT_STEP_DEG;
    }

    if let Some((relief, dir)) = best_dry {
        return (dir, relief);
    }
    if let Some((_, dir)) = best_any {
        return (dir, f32::INFINITY);
    }
    (DVec3::X, 0.0)
}

/// Pick the takeoff heading whose along-strip height profile is flattest.
fn choose_runway_heading(
    hs: &dyn HeightSource,
    center_dir: DVec3,
    body_radius_m: f64,
    center_h: f64,
) -> DVec3 {
    let basis = TerrainPatchBasis::from_normal(center_dir);
    let center_point = center_dir * (body_radius_m + center_h);
    let mut best: Option<(f32, DVec3)> = None;
    for k in 0..HEADING_AZIMUTH_STEPS {
        let theta = std::f64::consts::PI * k as f64 / HEADING_AZIMUTH_STEPS as f64;
        let axis = (basis.tangent_x * theta.cos() + basis.tangent_z * theta.sin()).normalize();
        let mut min_h = f32::INFINITY;
        let mut max_h = f32::NEG_INFINITY;
        let mut ok = true;
        for s in 0..HEADING_SAMPLES {
            let t = s as f64 / (HEADING_SAMPLES as f64 - 1.0);
            let along = -RUNWAY_HALF_LENGTH_M + RUNWAY_LENGTH_M * t;
            let dir = (center_point + axis * along).normalize();
            match hs.sample_height_m(dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M) {
                Some(h) => {
                    min_h = min_h.min(h);
                    max_h = max_h.max(h);
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let relief = max_h - min_h;
        if best.is_none_or(|(br, _)| relief < br) {
            best = Some((relief, axis));
        }
    }
    best.map(|(_, a)| a).unwrap_or(basis.tangent_x)
}

/// Max/min terrain height over the flat-pad footprint (the painted strip plus
/// the levelled shoulder), sampled at fine LOD.
#[allow(clippy::too_many_arguments)]
fn footprint_extremes(
    hs: &dyn HeightSource,
    center_dir: DVec3,
    heading: DVec3,
    across: DVec3,
    half_along_m: f64,
    half_across_m: f64,
    body_radius_m: f64,
    center_h: f64,
) -> (f64, f64) {
    let center_point = center_dir * (body_radius_m + center_h);
    let mut max_h = center_h;
    let mut min_h = center_h;
    for i in 0..=FOOTPRINT_SAMPLES_LEN {
        let along = -half_along_m + 2.0 * half_along_m * (i as f64 / FOOTPRINT_SAMPLES_LEN as f64);
        for j in 0..=FOOTPRINT_SAMPLES_W {
            let across_m =
                -half_across_m + 2.0 * half_across_m * (j as f64 / FOOTPRINT_SAMPLES_W as f64);
            let dir = (center_point + heading * along + across * across_m).normalize();
            if let Some(h) = hs.sample_height_m(dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M) {
                let h = h as f64;
                max_h = max_h.max(h);
                min_h = min_h.min(h);
            }
        }
    }
    (max_h, min_h)
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// The body-fixed runway frame at the fixed pad elevation `E`. Projects a
/// `(along, across)` runway coordinate onto a level (constant-radius) point.
struct RunwayFrame {
    body_id: BodyId,
    center_dir: DVec3,
    heading: DVec3,
    across: DVec3,
    body_radius_m: f64,
    elevation_m: f64,
}

impl RunwayFrame {
    fn center_surface(&self) -> DVec3 {
        self.center_dir * (self.body_radius_m + self.elevation_m)
    }

    /// Body-fixed offset (from the pad centre) of a runway coordinate at radius
    /// `E + radius_offset`, plus the radial direction there. With
    /// `radius_offset = 0` this is the flat (level) pad top.
    fn level(&self, along_m: f64, across_m: f64, radius_offset: f64) -> (DVec3, DVec3) {
        let cs = self.center_surface();
        let dir = (cs + self.heading * along_m + self.across * across_m).normalize();
        let pos = dir * (self.body_radius_m + self.elevation_m + radius_offset);
        (pos - cs, dir)
    }
}

fn build_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn flat_runway_material(materials: &mut Assets<StandardMaterial>, color: Color, rough: f32) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: color,
        perceptual_roughness: rough,
        metallic: 0.0,
        double_sided: true,
        cull_mode: None,
        ..default()
    })
}

fn spawn_runway_geometry(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    frame: &RunwayFrame,
    swap_radius_m: f32,
    parent: Entity,
    body_state: &BodyState,
) {
    // The terrain itself is flattened to `E` across the pad (see the
    // `TerrainFlatten` installed in `finish_runway_spawn`), so the runway is just
    // a paved strip sitting flush on the levelled ground — no skirt or runoff
    // geometry. The asphalt is lifted a few cm so it reads as paving on the
    // grass and never z-fights the flattened tiles.
    let top = meshes.add(build_top_mesh(frame));
    let asphalt = flat_runway_material(materials, Color::srgb(0.055, 0.055, 0.062), 0.95);

    // --- Markings (white, one mesh) ---
    let markings = meshes.add(build_markings_mesh(frame));
    let paint = flat_runway_material(materials, Color::srgb(0.85, 0.85, 0.85), 0.8);

    // Anchor the pad centre via a root-grid big_space cell, positioned in f64
    // (heliocentric) so the multi-Mm surface offset never lands in an f32
    // translation. `update_runway_transform` re-derives this every frame; this
    // is just the first-frame value. The strip rides the f32 `Transform.rotation`
    // (surface orientation) — fine, because only the small child vertex offsets
    // are rotated by it, not the planet-radius position.
    let orientation = body_state.orientation.normalize();
    let center_world = body_state.position + orientation * frame.center_surface();
    let (cell, local) = real_space_grid().translation_to_grid(center_world);
    let runway_entity = commands
        .spawn((
            Mesh3d(top),
            MeshMaterial3d(asphalt),
            Transform {
                translation: local,
                rotation: orientation.as_quat(),
                scale: Vec3::ONE,
            },
            cell,
            Visibility::Inherited,
            RenderLayers::layer(SHIP_LAYER),
            NotShadowCaster,
            ChildOf(parent),
            RunwayVisual {
                body_id: frame.body_id,
                swap_radius_m,
                center_surface_body: frame.center_surface(),
            },
            Name::new("Thalos Runway"),
        ))
        .id();

    commands.spawn((
        Mesh3d(markings),
        MeshMaterial3d(paint),
        Transform::IDENTITY,
        Visibility::Inherited,
        RenderLayers::layer(SHIP_LAYER),
        NotShadowCaster,
        ChildOf(runway_entity),
        Name::new("Runway Markings"),
    ));

    spawn_runway_posts(commands, meshes, materials, frame, runway_entity);
}

fn build_top_mesh(frame: &RunwayFrame) -> Mesh {
    let nl = RUNWAY_TOP_SEGMENTS_LEN;
    let nw = RUNWAY_TOP_SEGMENTS_W;
    let mut positions = Vec::with_capacity((nl + 1) * (nw + 1));
    let mut normals = Vec::with_capacity((nl + 1) * (nw + 1));
    let mut uvs = Vec::with_capacity((nl + 1) * (nw + 1));
    let mut indices = Vec::with_capacity(nl * nw * 6);
    for i in 0..=nl {
        let along = -RUNWAY_HALF_LENGTH_M + RUNWAY_LENGTH_M * (i as f64 / nl as f64);
        let v = i as f32 / nl as f32;
        for j in 0..=nw {
            let across_m = -RUNWAY_HALF_WIDTH_M + RUNWAY_WIDTH_M * (j as f64 / nw as f64);
            let u = j as f32 / nw as f32;
            let (off, up) = frame.level(along, across_m, RUNWAY_ASPHALT_LIFT_M);
            positions.push([off.x as f32, off.y as f32, off.z as f32]);
            normals.push([up.x as f32, up.y as f32, up.z as f32]);
            uvs.push([u, v]);
        }
    }
    let row = (nw + 1) as u32;
    for i in 0..nl as u32 {
        for j in 0..nw as u32 {
            let a = i * row + j;
            let b = a + 1;
            let c = a + row;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    build_mesh(positions, normals, uvs, indices)
}

fn build_markings_mesh(frame: &RunwayFrame) -> Mesh {
    let mut p = Vec::new();
    let mut n = Vec::new();
    let mut u = Vec::new();
    let mut idx = Vec::new();

    let half_w = RUNWAY_HALF_WIDTH_M;
    let half_l = RUNWAY_HALF_LENGTH_M;

    // Side edge lines (1 m wide, set in 1.5 m from the edge).
    let edge_c = half_w - 1.5;
    for sign in [-1.0, 1.0] {
        let c = sign * edge_c;
        push_marking_strip(&mut p, &mut n, &mut u, &mut idx, frame, -half_l + 60.0, half_l - 60.0, c - 0.5, c + 0.5);
    }
    // Dashed centreline (1 m wide; 30 m dash / 20 m gap).
    let mut a = -half_l + 120.0;
    while a + 30.0 < half_l - 120.0 {
        push_marking_strip(&mut p, &mut n, &mut u, &mut idx, frame, a, a + 30.0, -0.5, 0.5);
        a += 50.0;
    }
    // Threshold bars (solid, ~10 m along, near each end).
    let bar_in = half_w - 3.0;
    push_marking_strip(&mut p, &mut n, &mut u, &mut idx, frame, -half_l + 30.0, -half_l + 40.0, -bar_in, bar_in);
    push_marking_strip(&mut p, &mut n, &mut u, &mut idx, frame, half_l - 40.0, half_l - 30.0, -bar_in, bar_in);
    // Touchdown aiming blocks (a pair flanking the centreline near each end).
    for end in [-1.0, 1.0] {
        let a0 = end * (half_l - 360.0);
        let a1 = end * (half_l - 280.0);
        let (lo, hi) = if a0 < a1 { (a0, a1) } else { (a1, a0) };
        for off in [-9.0, 5.0] {
            push_marking_strip(&mut p, &mut n, &mut u, &mut idx, frame, lo, hi, off, off + 4.0);
        }
    }

    build_mesh(p, n, u, idx)
}

#[allow(clippy::too_many_arguments)]
fn push_marking_strip(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    frame: &RunwayFrame,
    along0: f64,
    along1: f64,
    across0: f64,
    across1: f64,
) {
    let len = (along1 - along0).abs();
    let segs = ((len / RUNWAY_MARKING_SEG_LEN_M).ceil() as usize).max(1);
    let base = positions.len() as u32;
    for i in 0..=segs {
        let t = i as f64 / segs as f64;
        let along = along0 + (along1 - along0) * t;
        for (j, &ac) in [across0, across1].iter().enumerate() {
            let (off, up) = frame.level(along, ac, RUNWAY_MARKING_LIFT_M);
            positions.push([off.x as f32, off.y as f32, off.z as f32]);
            normals.push([up.x as f32, up.y as f32, up.z as f32]);
            uvs.push([j as f32, t as f32]);
        }
    }
    let row = 2u32;
    for i in 0..segs as u32 {
        let a = base + i * row;
        let b = a + 1;
        let c = a + row;
        let d = c + 1;
        indices.extend_from_slice(&[a, c, b, b, c, d]);
    }
}

fn post_material(color: Color) -> StandardMaterial {
    StandardMaterial {
        base_color: color,
        emissive: color.to_linear() * 0.25,
        perceptual_roughness: 0.6,
        metallic: 0.0,
        ..default()
    }
}

/// Raised edge posts at regular intervals down both sides — the 3D references
/// to orient around. Takeoff-threshold posts green, far-end posts red, the rest
/// white (aviation convention).
fn spawn_runway_posts(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    frame: &RunwayFrame,
    parent: Entity,
) {
    let post_mesh = meshes.add(Cuboid::new(POST_SIZE_M, POST_HEIGHT_M, POST_SIZE_M));
    let thresh_mesh = meshes.add(Cuboid::new(POST_SIZE_M, POST_THRESHOLD_HEIGHT_M, POST_SIZE_M));
    let white = materials.add(post_material(Color::srgb(0.9, 0.9, 0.9)));
    let green = materials.add(post_material(Color::srgb(0.1, 0.8, 0.2)));
    let red = materials.add(post_material(Color::srgb(0.9, 0.15, 0.1)));

    let edge = RUNWAY_HALF_WIDTH_M + POST_EDGE_OFFSET_M;
    let stations = (RUNWAY_LENGTH_M / POST_SPACING_M).floor() as i32;
    for i in 0..=stations {
        let along = -RUNWAY_HALF_LENGTH_M + POST_SPACING_M * i as f64;
        if along > RUNWAY_HALF_LENGTH_M + 1.0 {
            break;
        }
        let (mesh, mat, height) = if i == 0 {
            (thresh_mesh.clone(), green.clone(), POST_THRESHOLD_HEIGHT_M)
        } else if i == stations {
            (thresh_mesh.clone(), red.clone(), POST_THRESHOLD_HEIGHT_M)
        } else {
            (post_mesh.clone(), white.clone(), POST_HEIGHT_M)
        };
        for side in [-1.0, 1.0] {
            let (base, up) = frame.level(along, side * edge, 0.0);
            let base_f = Vec3::new(base.x as f32, base.y as f32, base.z as f32);
            let up_f = Vec3::new(up.x as f32, up.y as f32, up.z as f32);
            let translation = base_f + up_f * (height * 0.5);
            let rotation = Quat::from_rotation_arc(Vec3::Y, up_f);
            commands.spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(mat.clone()),
                Transform {
                    translation,
                    rotation,
                    scale: Vec3::ONE,
                },
                Visibility::Inherited,
                RenderLayers::layer(SHIP_LAYER),
                NotShadowCaster,
                ChildOf(parent),
                Name::new("Runway Post"),
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Collider — a flat kinematic trimesh at elevation E
// ---------------------------------------------------------------------------

/// Spawn the flat landing collider at the platform elevation. Kept near its own
/// origin (body-fixed offsets from the platform centre) and posed each frame by
/// [`sync_runway_collider_pose`], exactly like the terrain collider patch — so
/// the aircraft rests/lands on a flat surface independent of the bumpy terrain.
fn spawn_runway_collider(commands: &mut Commands, frame: &RunwayFrame, _body_state: &BodyState) {
    let (vertices, indices) = build_collider_trimesh(frame);
    let center_surface = frame.center_surface();
    // The ship Avian body lives in the **body-fixed (rotating) frame**, so the
    // runway collider is held static there: `Position = center_surface` (the
    // body-fixed platform centre), identity rotation, zero velocity. The
    // trimesh vertices are body-fixed offsets from that centre, so they land
    // on the level platform with no co-rotation speed — the aircraft rests and
    // taxis at ~0 m/s instead of the ~256 m/s the old inertial frame imposed,
    // which is what keeps ground contact stable. See `docs/surface.md`.
    let collider = Collider::trimesh(vertices, indices);
    commands.spawn((
        RigidBody::Kinematic,
        collider,
        Position(center_surface),
        Rotation(DQuat::IDENTITY),
        LinearVelocity(DVec3::ZERO),
        AngularVelocity(DVec3::ZERO),
        RunwayCollider {
            body_id: frame.body_id,
            center_surface_body_m: center_surface,
        },
        Name::new("Runway collider"),
    ));
}

/// Flat trimesh (level grid at `E`): body-fixed vertex offsets from the
/// platform centre plus triangle indices.
fn build_collider_trimesh(frame: &RunwayFrame) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let nl = RUNWAY_COLLIDER_SEGMENTS_LEN;
    let nw = RUNWAY_COLLIDER_SEGMENTS_W;
    // Cover the runway plus the levelled pad margin so the aircraft can roll
    // off the strip onto the flat shoulder without dropping off the collider.
    let half_l = RUNWAY_HALF_LENGTH_M + RUNWAY_PAD_MARGIN_M;
    let half_w = RUNWAY_HALF_WIDTH_M + RUNWAY_PAD_MARGIN_M;
    let mut vertices = Vec::with_capacity((nl + 1) * (nw + 1));
    let mut indices = Vec::with_capacity(nl * nw * 2);
    for i in 0..=nl {
        let along = -half_l + 2.0 * half_l * (i as f64 / nl as f64);
        for j in 0..=nw {
            let across_m = -half_w + 2.0 * half_w * (j as f64 / nw as f64);
            let (off, _) = frame.level(along, across_m, 0.0);
            vertices.push(off);
        }
    }
    let row = (nw + 1) as u32;
    for i in 0..nl as u32 {
        for j in 0..nw as u32 {
            let a = i * row + j;
            let b = a + 1;
            let c = a + row;
            let d = c + 1;
            indices.push([a, c, b]);
            indices.push([b, c, d]);
        }
    }
    (vertices, indices)
}

/// Hold the kinematic runway collider **static in the ship's body-fixed frame**
/// (mirrors `local_physics::sync_terrain_collider_pose`): `Position =
/// center_surface_body_m`, identity rotation, zero velocity. The collider's
/// trimesh vertices are body-fixed offsets from that centre, so it stays
/// exactly on the level platform with no co-rotation speed. Defensive each
/// frame; the pose is constant, so this is effectively a no-op after spawn.
fn sync_runway_collider_pose(
    mut q: Query<(
        &RunwayCollider,
        &mut Position,
        &mut Rotation,
        &mut LinearVelocity,
        &mut AngularVelocity,
    )>,
) {
    for (rc, mut position, mut rotation, mut linear_velocity, mut angular_velocity) in &mut q {
        position.0 = rc.center_surface_body_m;
        rotation.0 = DQuat::IDENTITY;
        linear_velocity.0 = DVec3::ZERO;
        angular_velocity.0 = DVec3::ZERO;
    }
}

// ---------------------------------------------------------------------------
// Aircraft placement (referenced to the fixed elevation E, never terrain)
// ---------------------------------------------------------------------------

/// Attitude with the nose along `heading_body` and the dorsal along `up_body`
/// — level on the ground, lined up with the runway.
fn level_heading_attitude(body_state: &BodyState, up_body: DVec3, heading_body: DVec3) -> AttitudeState {
    let dorsal = up_body.normalize();
    let nose = (heading_body - dorsal * heading_body.dot(dorsal))
        .try_normalize()
        .unwrap_or_else(|| {
            let seed = if dorsal.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
            (seed - dorsal * seed.dot(dorsal)).normalize()
        });
    let right = nose.cross(dorsal).normalize();
    let craft_to_body = DMat3::from_cols(right, nose, dorsal);
    AttitudeState {
        orientation: (body_state.orientation * DQuat::from_mat3(&craft_to_body)).normalize(),
        angular_velocity: DVec3::ZERO,
    }
}

/// Measure how far the craft's lowest visual point sits below its origin,
/// along the craft's ventral (−Z) axis — so the parked craft can be lifted by
/// exactly this much and rest on the surface for *any* craft (gear, belly,
/// pods, whatever is lowest), with no per-craft constant.
///
/// Walks every mesh under the player-ship root, computes each part's local AABB
/// straight from its **mesh vertices** (`Mesh::compute_aabb` — the part visuals
/// carry `NoFrustumCulling`, so they have no Bevy-computed `Aabb` component),
/// transforms its corners into the ship-root (craft body) frame, and takes the
/// most-negative Z. The craft frame is `+Y` nose / `+Z` dorsal (up), so the
/// lowest point is the minimum Z. Returns `None` if no descendant mesh is ready
/// yet so the caller can retry. Uses local-to-root affines, so the result is
/// independent of the craft's current (placeholder-orbit) pose.
fn craft_ground_clearance(
    root_entity: Entity,
    root_gt: &GlobalTransform,
    children_q: &Query<&Children>,
    mesh_q: &Query<(&GlobalTransform, &Mesh3d)>,
    meshes: &Assets<Mesh>,
) -> Option<f64> {
    let root_inv = root_gt.affine().inverse();
    let mut min_z = f32::INFINITY;
    let mut found = false;
    let mut stack: Vec<Entity> = Vec::new();
    if let Ok(c) = children_q.get(root_entity) {
        stack.extend(c.iter());
    }
    while let Some(e) = stack.pop() {
        if let Ok((gt, mesh3d)) = mesh_q.get(e)
            && let Some(aabb) = meshes.get(&mesh3d.0).and_then(Mesh::compute_aabb)
        {
            let local = root_inv * gt.affine();
            let c = aabb.center;
            let h = aabb.half_extents;
            for sx in [-1.0f32, 1.0] {
                for sy in [-1.0f32, 1.0] {
                    for sz in [-1.0f32, 1.0] {
                        let corner = Vec3A::new(c.x + sx * h.x, c.y + sy * h.y, c.z + sz * h.z);
                        min_z = min_z.min(local.transform_point3a(corner).z);
                        found = true;
                    }
                }
            }
        }
        if let Ok(c) = children_q.get(e) {
            stack.extend(c.iter());
        }
    }
    found.then(|| (-min_z as f64).max(0.0))
}

/// Park the aircraft frozen, level, on the runway as a **`BodyFixed` launch
/// clamp** — the documented "spawn stably, settle on the runway, go from there"
/// behaviour, and the fix for the tip-over.
///
/// The clamp is the whole point. The parked spawn happens during loading, while
/// the terrain under the craft is still streaming from coarse → fine and its
/// collider is heaving up toward the flat pad height. If the craft were under
/// the local-physics bubble (Avian) through that window it would integrate gear
/// contacts against that moving ground and tip itself over (it lands on its tail
/// by reveal). `BodyFixed` makes the pose analytic — Avian is held on it by
/// [`crate::local_physics::snap_avian_from_canonical`] and never integrates — so
/// the craft sits rock-steady, level, regardless of what the terrain is doing.
/// References the fixed elevation `E`, so it never sinks or heaves.
/// `clearance_m` lifts the origin so the lowest point rests on the surface.
///
/// Registering the clamp in [`DebugLaunchMount`] lets the existing
/// [`crate::local_physics::release_debug_launch_mount`] release it to `OnRails`
/// (→ bubble) when the player throttles up — i.e. the dynamic handoff to Avian
/// happens *after* the terrain has settled and on the player's command, on flat
/// ground, so it doesn't tip.
fn place_parked(
    sim: &mut SimulationState,
    launch_mount: &mut DebugLaunchMount,
    body_state: &BodyState,
    site: &RunwaySite,
    body_radius_m: f64,
    clearance_m: f64,
) {
    let surface_radius = body_radius_m + site.elevation_m;
    let center_surface = site.center_dir * surface_radius;
    let threshold_dir =
        (center_surface - site.heading_tangent * (RUNWAY_HALF_LENGTH_M - PARK_THRESHOLD_INSET_M))
            .normalize();

    let up_body = threshold_dir;
    let nose_body = (site.heading_tangent - up_body * site.heading_tangent.dot(up_body)).normalize();
    let position_body = threshold_dir * (surface_radius + clearance_m);

    let position = body_state.position + body_state.orientation * position_body;
    let velocity = body_fixed_surface_velocity(body_state, position_body);
    let state = StateVector { position, velocity };
    let attitude = level_heading_attitude(body_state, up_body, nose_body);

    // Freeze as a body-fixed launch clamp at the level pose.
    sim.simulation.set_ship_state(state);
    sim.simulation.set_attitude(attitude);
    let pose = body_fixed_pose_from_inertial(body_state, TranslationalState::from(state), attitude);
    sim.simulation
        .transition_authority(AuthorityMode::BodyFixed {
            body: site.body_id,
            pose,
        });
    sim.simulation.warp.reset();
    sim.simulation.set_throttle(0.0);
    sim.simulation.set_target_body(Some(site.body_id));
    // Arm the throttle-up release so the player can taxi/take off from the clamp.
    launch_mount.active = Some(DebugLaunchMountState {
        body_id: site.body_id,
        pose,
    });
}

/// Put the aircraft on short final, lined up with the centreline and sinking,
/// coasting on rails until the local-physics bubble takes over. Referenced to
/// the fixed elevation `E`.
fn place_approach(sim: &mut SimulationState, body_state: &BodyState, site: &RunwaySite, body_radius_m: f64) {
    let surface_radius = body_radius_m + site.elevation_m;
    let center_surface = site.center_dir * surface_radius;
    let threshold_point = center_surface - site.heading_tangent * RUNWAY_HALF_LENGTH_M;
    let approach_point = threshold_point - site.heading_tangent * APPROACH_BACK_M;
    let approach_dir = approach_point.normalize();

    let up = (body_state.orientation * approach_dir).normalize();
    let local_heading =
        (site.heading_tangent - approach_dir * site.heading_tangent.dot(approach_dir)).normalize();
    let heading_inertial = (body_state.orientation * local_heading).normalize();

    let radius = surface_radius + APPROACH_ALT_M;
    let position = body_state.position + up * radius;
    let r_rel = position - body_state.position;
    let surface_velocity = body_state.velocity + body_state.angular_velocity.cross(r_rel);
    let approach_vel = heading_inertial * APPROACH_SPEED_M_S - up * APPROACH_SINK_M_S;
    let velocity = surface_velocity + approach_vel;

    let nose = approach_vel.try_normalize().unwrap_or(heading_inertial);
    let dorsal = (up - nose * up.dot(nose)).try_normalize().unwrap_or(up);
    let right = nose.cross(dorsal).normalize();
    let basis = DMat3::from_cols(right, nose, dorsal);
    let attitude = AttitudeState {
        orientation: DQuat::from_mat3(&basis),
        angular_velocity: DVec3::ZERO,
    };

    sim.simulation.set_ship_state(StateVector { position, velocity });
    sim.simulation.set_attitude(attitude);
    sim.simulation
        .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
    sim.simulation.warp.reset();
    sim.simulation.set_throttle(0.0);
    sim.simulation.set_target_body(Some(site.body_id));
}

// ---------------------------------------------------------------------------
// Per-frame transform + visibility
// ---------------------------------------------------------------------------

/// Position the runway like the player ship: a root-grid big_space child placed
/// in **f64** every frame. Anchoring it as a fixed-cell child of the *rotating*
/// body grid (the natural-looking choice) makes big_space rotate its multi-Mm
/// cell offset by an f32 quaternion, which jitters frame-to-frame at high warp
/// as the body spins fast (≈ decimetre ULP at planet radius). Computing the
/// centre in f64 here and letting the f32 `Transform.rotation` act only on the
/// small (≤ ~1.6 km) child vertex offsets keeps it rock-steady — the same trick
/// udlod uses with `PreciseRotation` for the terrain itself.
fn update_runway_transform(
    solar: Res<SolarSystemState>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut runways: Query<(&RunwayVisual, &mut CellCoord, &mut Transform)>,
) {
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let Ok(grid) = root_grid.single() else {
        return;
    };
    for (rv, mut cell, mut transform) in &mut runways {
        let Some(state) = states.get(rv.body_id) else {
            continue;
        };
        // Thalos is free-spinning, so the surface orientation is the body
        // orientation (the free-spin branch of
        // `surface_body_to_world_orientation_f64` — the only kind a player
        // stands on). Tidally-locked bodies would need that helper's f64 path.
        let orientation = state.orientation.normalize();
        let center_world = state.position + orientation * rv.center_surface_body;
        let (next_cell, local) = grid.translation_to_grid(center_world);
        *cell = next_cell;
        transform.translation = local;
        transform.rotation = orientation.as_quat();
    }
}

/// Hide the runway when the camera is far from the body (mirrors the
/// terrain/impostor LOD swap), so it doesn't poke through the orbital impostor.
fn sync_runway_visibility(
    cam_q: Query<&GlobalTransform, With<ShipCamera>>,
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut runways: Query<(&RunwayVisual, &mut Visibility)>,
) {
    let Ok(cam) = cam_q.single() else {
        return;
    };
    let cam_pos = cam.translation();
    for (marker, mut vis) in &mut runways {
        let Some((_, body_tf)) = body_q.iter().find(|(b, _)| b.body_id == marker.body_id) else {
            continue;
        };
        let dist = (cam_pos - body_tf.translation()).length();
        let want = if dist < marker.swap_radius_m {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
    }
}
