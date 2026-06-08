//! A fixed runway on the Thalos surface, plus the two spawn scenarios that put
//! the aircraft on it (`just game runway`, `just game runway-approach`).
//!
//! This is a deferred, terrain-aware spawn just like the descent scenarios in
//! [`crate::spawn`]: `main.rs` parks the ship in the placeholder orbit behind
//! the loading screen, and [`finish_runway_spawn`] installs the real runway +
//! aircraft state on the first `AppState::Running` frame, once the terrain
//! height source is resident.
//!
//! The runway location is chosen deterministically in the body-fixed frame
//! (epoch-independent): a coarse dry-land flat-patch search picks the site, and
//! a fine along-strip relief scan picks the takeoff heading. The runway and its
//! markers are draped over the real terrain height (sampled at
//! [`PHYSICS_QUERY_TILE_LOD_M`], the same query that feeds the collider the
//! aircraft rolls on) and parented to the body's `RealSpaceBody` grid so they
//! co-rotate with the surface exactly like the terrain does.

use bevy::camera::visibility::RenderLayers;
use bevy::light::NotShadowCaster;
use bevy::math::{DMat3, DQuat, DVec3, Vec3};
// `Vec3` is also in the prelude; the explicit import keeps the D-types and Vec3
// grouped and is harmless (same glam type).
use bevy::prelude::*;

use thalos_body_render::{HeightSource, TerrainPatchBasis};
use thalos_physics_canonical::body_fixed::{
    body_fixed_pose_from_inertial, body_fixed_surface_velocity,
};
use thalos_physics_canonical::canonical::{AuthorityMode, Epoch, TranslationalState};
use thalos_physics_canonical::types::{AttitudeState, BodyState};
use thalos_physics_local::{HeightSourceRegistry, TerrainSurfaceRegistry};
use thalos_world::{BodyId, StateVector};

use crate::SimStage;
use crate::camera::ShipCamera;
use crate::coords::SHIP_LAYER;
use crate::loading::AppState;
use crate::local_physics::PHYSICS_QUERY_TILE_LOD_M;
use crate::rendering::RealSpaceBody;
use crate::rendering::real_space::real_space_grid;
use crate::solar_system_state::SimulationState;
use crate::spawn::{SpawnSituation, sample_site_relief_m};

// ---------------------------------------------------------------------------
// Runway dimensions (realistic large runway: 3 km × 60 m)
// ---------------------------------------------------------------------------

const RUNWAY_LENGTH_M: f64 = 3000.0;
const RUNWAY_HALF_LENGTH_M: f64 = RUNWAY_LENGTH_M * 0.5;
const RUNWAY_WIDTH_M: f64 = 60.0;
const RUNWAY_HALF_WIDTH_M: f64 = RUNWAY_WIDTH_M * 0.5;

/// Lift the runway ribbon this far above the sampled terrain. Covers the
/// ~0.2 m f32/f64 rotation slip between this static mesh (big_space far path)
/// and the udlod terrain near the camera (f64 `PreciseRotation` path).
const RUNWAY_DRAPE_M: f64 = 0.5;
/// Markings sit just above the ribbon to avoid z-fighting.
const RUNWAY_MARKING_DRAPE_M: f64 = 0.6;

/// Ribbon tessellation: segments along the length / across the width.
const RUNWAY_MESH_SEGMENTS_LEN: usize = 240;
const RUNWAY_MESH_SEGMENTS_W: usize = 8;
/// Subdivision length for marking strips so they drape with the terrain.
const RUNWAY_MARKING_SEG_LEN_M: f64 = 25.0;

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

/// Belly clearance above terrain for the parked aircraft (Skyhawk fuselage
/// radius + the ribbon drape, so it sits on the painted surface).
const PARK_CLEARANCE_M: f64 = 1.8;
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
    /// Terrain height (m) at the centre.
    pub surface_height_m: f64,
}

/// Marker on the runway ribbon entity. Children (markings, posts) inherit its
/// visibility, so the swap system only toggles this one entity per body.
#[derive(Component, Debug)]
struct RunwayVisual {
    body_id: BodyId,
    swap_radius_m: f32,
}

pub struct RunwayPlugin;

impl Plugin for RunwayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                finish_runway_spawn
                    .run_if(in_state(AppState::Running))
                    .before(SimStage::Physics),
                sync_runway_visibility.in_set(SimStage::Sync),
            ),
        );
    }
}

/// Deferred finisher: pick the site, build the runway, and place the aircraft.
/// Runs once, retrying each frame until the terrain height source is resident.
fn finish_runway_spawn(
    mut done: Local<bool>,
    situation: Res<SpawnSituation>,
    mut sim: ResMut<SimulationState>,
    height_sources: Res<HeightSourceRegistry>,
    surfaces: Res<TerrainSurfaceRegistry>,
    body_q: Query<(Entity, &RealSpaceBody)>,
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
    let Some(real_body_entity) = body_q
        .iter()
        .find(|(_, b)| b.body_id == body_id)
        .map(|(e, _)| e)
    else {
        return; // body grid not spawned yet
    };
    let hs = height_source.as_ref();

    let body_radius_m = sim.system.bodies[body_id].radius_m;
    let sea_level_m = surfaces
        .get(body_id)
        .and_then(|s| s.static_surface.sea_level_m);

    let (center_dir, relief_m) = find_runway_site(hs, sea_level_m, body_radius_m);
    let center_h = hs
        .sample_height_m(center_dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
        .unwrap_or(0.0) as f64;
    let heading_tangent = choose_runway_heading(hs, center_dir, body_radius_m, center_h);

    let site = RunwaySite {
        body_id,
        center_dir,
        heading_tangent,
        surface_height_m: center_h,
    };

    let lat_deg = center_dir.y.clamp(-1.0, 1.0).asin().to_degrees();
    let lon_deg = center_dir.z.atan2(center_dir.x).to_degrees();
    info!(
        "runway: {} on {} at lat {:.1}°, lon {:.1}°, ground {:.0} m, site relief {:.0} m ({:.0} m × {:.0} m)",
        if matches!(*situation, SpawnSituation::Runway) {
            "parked"
        } else {
            "on approach"
        },
        sim.system.bodies[body_id].name,
        lat_deg,
        lon_deg,
        center_h,
        relief_m,
        RUNWAY_LENGTH_M,
        RUNWAY_WIDTH_M,
    );

    spawn_runway_geometry(
        &mut commands,
        &mut meshes,
        &mut materials,
        hs,
        &site,
        body_radius_m,
        real_body_entity,
    );

    // Place the aircraft. Body state is read before mutating the sim.
    let epoch = Epoch(sim.simulation.sim_time());
    let body_state = sim.ephemeris.state(body_id, epoch);
    match *situation {
        SpawnSituation::Runway => place_parked(&mut sim, &body_state, hs, &site, body_radius_m),
        SpawnSituation::RunwayApproach => {
            place_approach(&mut sim, &body_state, hs, &site, body_radius_m)
        }
        _ => {}
    }

    commands.insert_resource(site);
    *done = true;
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
        // Airless body: any low-latitude point is land. Pick a fixed seed.
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

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Bundles the body-fixed runway frame so the mesh/marking/post builders can
/// project a `(along, across)` runway coordinate onto the draped surface.
struct RunwayFrame {
    center_point: DVec3,
    heading: DVec3,
    across: DVec3,
    body_radius_m: f64,
    center_h: f64,
}

impl RunwayFrame {
    /// Returns `(offset_from_center_body, surface_normal_body)` for a runway
    /// coordinate, draped `drape` metres above the sampled terrain. Both are in
    /// the body-fixed frame (the runway entity carries identity rotation; the
    /// parent grid applies the body's surface orientation).
    fn offset(
        &self,
        hs: &dyn HeightSource,
        along_m: f64,
        across_m: f64,
        drape: f64,
    ) -> (DVec3, DVec3) {
        let p = self.center_point + self.heading * along_m + self.across * across_m;
        let dir = p.normalize();
        let h = hs
            .sample_height_m(dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
            .unwrap_or(self.center_h as f32) as f64;
        let surf = dir * (self.body_radius_m + h + drape);
        (surf - self.center_point, dir)
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
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Append one draped marking strip (a thin quad subdivided along its length)
/// to the shared marking-mesh buffers.
#[allow(clippy::too_many_arguments)]
fn push_marking_strip(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    frame: &RunwayFrame,
    hs: &dyn HeightSource,
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
            let (off, up) = frame.offset(hs, along, ac, RUNWAY_MARKING_DRAPE_M);
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

fn spawn_runway_geometry(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    hs: &dyn HeightSource,
    site: &RunwaySite,
    body_radius_m: f64,
    parent: Entity,
) {
    let frame = RunwayFrame {
        center_point: site.center_dir * (body_radius_m + site.surface_height_m),
        heading: site.heading_tangent.normalize(),
        across: site.center_dir.cross(site.heading_tangent).normalize(),
        body_radius_m,
        center_h: site.surface_height_m,
    };

    // --- Ribbon (asphalt) ---
    let nl = RUNWAY_MESH_SEGMENTS_LEN;
    let nw = RUNWAY_MESH_SEGMENTS_W;
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
            let (off, up) = frame.offset(hs, along, across_m, RUNWAY_DRAPE_M);
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
    let ribbon = meshes.add(build_mesh(positions, normals, uvs, indices));
    let asphalt = materials.add(StandardMaterial {
        base_color: Color::srgb(0.055, 0.055, 0.062),
        perceptual_roughness: 0.95,
        metallic: 0.0,
        double_sided: true,
        cull_mode: None,
        ..default()
    });

    // --- Markings (white, combined into one mesh) ---
    let mut mp = Vec::new();
    let mut mn = Vec::new();
    let mut mu = Vec::new();
    let mut mi = Vec::new();
    build_markings(&mut mp, &mut mn, &mut mu, &mut mi, &frame, hs);
    let markings = meshes.add(build_mesh(mp, mn, mu, mi));
    let paint = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.85, 0.85),
        perceptual_roughness: 0.8,
        metallic: 0.0,
        double_sided: true,
        cull_mode: None,
        ..default()
    });

    // Anchor the ribbon at the runway centre via a big_space cell so the
    // multi-Mm surface offset never lands in an f32 translation.
    let (cell, local) = real_space_grid().translation_to_grid(frame.center_point);
    let runway_entity = commands
        .spawn((
            Mesh3d(ribbon),
            MeshMaterial3d(asphalt),
            Transform::from_translation(local),
            cell,
            Visibility::Inherited,
            RenderLayers::layer(SHIP_LAYER),
            NotShadowCaster,
            ChildOf(parent),
            RunwayVisual {
                body_id: site.body_id,
                swap_radius_m: (body_radius_m * RUNWAY_VIS_RADIUS_FACTOR) as f32,
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

    spawn_runway_posts(commands, meshes, materials, &frame, hs, runway_entity);
}

/// Build the painted markings: side edge lines, dashed centreline, threshold
/// bars at both ends, and a touchdown aiming block near each threshold.
fn build_markings(
    p: &mut Vec<[f32; 3]>,
    n: &mut Vec<[f32; 3]>,
    u: &mut Vec<[f32; 2]>,
    idx: &mut Vec<u32>,
    frame: &RunwayFrame,
    hs: &dyn HeightSource,
) {
    let half_w = RUNWAY_HALF_WIDTH_M;
    let half_l = RUNWAY_HALF_LENGTH_M;

    // Side edge lines (1 m wide, set in 1.5 m from the edge).
    let edge_c = half_w - 1.5;
    for sign in [-1.0, 1.0] {
        let c = sign * edge_c;
        push_marking_strip(
            p,
            n,
            u,
            idx,
            frame,
            hs,
            -half_l + 60.0,
            half_l - 60.0,
            c - 0.5,
            c + 0.5,
        );
    }

    // Dashed centreline (1 m wide; 30 m dash / 20 m gap).
    let mut a = -half_l + 120.0;
    while a + 30.0 < half_l - 120.0 {
        push_marking_strip(p, n, u, idx, frame, hs, a, a + 30.0, -0.5, 0.5);
        a += 50.0;
    }

    // Threshold bars (solid, ~10 m along, near each end).
    let bar_in = half_w - 3.0;
    push_marking_strip(
        p,
        n,
        u,
        idx,
        frame,
        hs,
        -half_l + 30.0,
        -half_l + 40.0,
        -bar_in,
        bar_in,
    );
    push_marking_strip(
        p,
        n,
        u,
        idx,
        frame,
        hs,
        half_l - 40.0,
        half_l - 30.0,
        -bar_in,
        bar_in,
    );

    // Touchdown aiming blocks (a pair flanking the centreline near each end).
    for end in [-1.0, 1.0] {
        let a0 = end * (half_l - 360.0);
        let a1 = end * (half_l - 280.0);
        let (lo, hi) = if a0 < a1 { (a0, a1) } else { (a1, a0) };
        for off in [-9.0, 5.0] {
            push_marking_strip(p, n, u, idx, frame, hs, lo, hi, off, off + 4.0);
        }
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
/// to orient around. Takeoff-threshold posts are green, far-end posts red,
/// the rest white (aviation convention).
fn spawn_runway_posts(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    frame: &RunwayFrame,
    hs: &dyn HeightSource,
    parent: Entity,
) {
    let post_mesh = meshes.add(Cuboid::new(POST_SIZE_M, POST_HEIGHT_M, POST_SIZE_M));
    let thresh_mesh = meshes.add(Cuboid::new(
        POST_SIZE_M,
        POST_THRESHOLD_HEIGHT_M,
        POST_SIZE_M,
    ));
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
            let (base, up) = frame.offset(hs, along, side * edge, RUNWAY_DRAPE_M);
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
// Aircraft placement
// ---------------------------------------------------------------------------

/// Attitude with the nose along `heading_body` and the dorsal along `up_body`
/// — level on the ground, lined up with the runway (the "level orbital flight"
/// convention shared by the navball and control axes).
fn level_heading_attitude(
    body_state: &BodyState,
    up_body: DVec3,
    heading_body: DVec3,
) -> AttitudeState {
    let dorsal = up_body.normalize();
    let nose = (heading_body - dorsal * heading_body.dot(dorsal))
        .try_normalize()
        .unwrap_or_else(|| {
            let seed = if dorsal.x.abs() < 0.9 {
                DVec3::X
            } else {
                DVec3::Z
            };
            (seed - dorsal * seed.dot(dorsal)).normalize()
        });
    let right = nose.cross(dorsal).normalize();
    let craft_to_body = DMat3::from_cols(right, nose, dorsal);
    AttitudeState {
        orientation: (body_state.orientation * DQuat::from_mat3(&craft_to_body)).normalize(),
        angular_velocity: DVec3::ZERO,
    }
}

/// Park the aircraft at rest on the runway threshold via body-fixed authority
/// (the launch-clamp pattern: pinned and stationary until throttle releases
/// the clamp). Mirrors `debug.rs`'s surface drop.
fn place_parked(
    sim: &mut SimulationState,
    body_state: &BodyState,
    hs: &dyn HeightSource,
    site: &RunwaySite,
    body_radius_m: f64,
) {
    let center_point = site.center_dir * (body_radius_m + site.surface_height_m);
    let threshold_point =
        center_point - site.heading_tangent * (RUNWAY_HALF_LENGTH_M - PARK_THRESHOLD_INSET_M);
    let threshold_dir = threshold_point.normalize();
    let threshold_h = hs
        .sample_height_m(threshold_dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
        .unwrap_or(site.surface_height_m as f32) as f64;

    let up_body = threshold_dir;
    let nose_body =
        (site.heading_tangent - up_body * site.heading_tangent.dot(up_body)).normalize();
    let position_body = threshold_dir * (body_radius_m + threshold_h + PARK_CLEARANCE_M);

    let position = body_state.position + body_state.orientation * position_body;
    let velocity = body_fixed_surface_velocity(body_state, position_body);
    let state = StateVector { position, velocity };
    let attitude = level_heading_attitude(body_state, up_body, nose_body);

    let pose = body_fixed_pose_from_inertial(body_state, TranslationalState::from(state), attitude);
    sim.simulation
        .transition_authority(AuthorityMode::BodyFixed {
            body: site.body_id,
            pose,
        });
    sim.simulation.set_ship_state(state);
    sim.simulation.set_attitude(attitude);
    sim.simulation.warp.reset();
    sim.simulation.set_throttle(0.0);
    sim.simulation.set_target_body(Some(site.body_id));
}

/// Put the aircraft on short final, lined up with the centreline and sinking,
/// coasting on rails until the local-physics bubble takes over.
fn place_approach(
    sim: &mut SimulationState,
    body_state: &BodyState,
    hs: &dyn HeightSource,
    site: &RunwaySite,
    body_radius_m: f64,
) {
    let center_point = site.center_dir * (body_radius_m + site.surface_height_m);
    let threshold_point = center_point - site.heading_tangent * RUNWAY_HALF_LENGTH_M;
    let approach_point = threshold_point - site.heading_tangent * APPROACH_BACK_M;
    let approach_dir = approach_point.normalize();
    let approach_h = hs
        .sample_height_m(approach_dir.as_vec3(), PHYSICS_QUERY_TILE_LOD_M)
        .unwrap_or(site.surface_height_m as f32) as f64;

    let up = (body_state.orientation * approach_dir).normalize();
    let local_heading =
        (site.heading_tangent - approach_dir * site.heading_tangent.dot(approach_dir)).normalize();
    let heading_inertial = (body_state.orientation * local_heading).normalize();

    let radius = body_radius_m + approach_h + APPROACH_ALT_M;
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

    sim.simulation
        .set_ship_state(StateVector { position, velocity });
    sim.simulation.set_attitude(attitude);
    sim.simulation
        .transition_authority(AuthorityMode::OnRails { trajectory: 0 });
    sim.simulation.warp.reset();
    sim.simulation.set_throttle(0.0);
    sim.simulation.set_target_body(Some(site.body_id));
}

// ---------------------------------------------------------------------------
// Visibility
// ---------------------------------------------------------------------------

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
