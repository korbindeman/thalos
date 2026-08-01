//! Auto-connections: the typed paved network linking the structures on a site.
//!
//! A base's fixed features (runways, hangars, pads, buildings) are *placed*; the
//! paving that links them is generated. This module owns that generation as a
//! **typed network**: every connection is one [`ConnectionKind`] — a taxiway, an
//! apron, a road, or a crawlerway — each with its own width, colour, and ground
//! lift. Line networks (taxiway / road / crawlerway) are strips along a set of
//! edges (a minimum spanning tree, or explicit edges); an apron is a filled
//! rectangle (a parking pad in front of a hangar).
//!
//! The meshes are built in the site's local tangent frame and anchored every
//! frame like the structures themselves (a root-grid big_space child posed in
//! f64), so they stay put at high warp and persist when the editor closes.
//!
//! Extensibility: adding a new connection type (a utility pipe run, a rail spur)
//! is a new [`ConnectionKind`] variant with a style — the router (`spawn_authored_*`
//! for the authored base, `rebuild_connections` for the editor) and the mesh
//! builders below are kind-agnostic. The future VAB→pad *crawler* animation rides
//! the [`ConnectionKind::Crawlerway`] geometry this already lays down.

use bevy::camera::visibility::RenderLayers;
use bevy::light::NotShadowCaster;
use bevy::math::{DMat3, DQuat, DVec2, DVec3};
use bevy::prelude::*;

use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_body_render::{ShadowedStandardMaterial, shadowed};
use thalos_world::BodyId;

use crate::SimStage;
use crate::coords::SHIP_LAYER;
use crate::rendering::real_space::RealSpaceRoot;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::solar_system_state::sync_solar_system_state;
use crate::structures::{StructureId, StructurePlacement, StructureRegistry};

use super::place::{BaseMaterials, kind_bounding_m};
use super::{BaseBuildState, BaseEditor, BaseEditorMode, base_editor_open};

use thalos_body_render::ScatterRegion;

// Connection geometry (paved regions, MST, strip/apron/path meshes) lives in
// `thalos_structures` (Phase 5b); the rebuild driver and transform sync stay
// here.
pub use thalos_structures::connection_geometry::PavedFootprints;
use thalos_structures::connection_geometry::{
    CONNECTION_LIFT_BASE_M, ConnectionStyle, Node as SiteNode, PavedRect, build_apron_mesh,
    build_path_mesh, build_strip_mesh, fillet_path, minimum_spanning_tree, network_paved_rects,
    path_paved_rects, paved_regions, site_anchor,
};

// ── Paved footprints ────────────────────────────────────────────────────────
//
// Connections are drawn as a *lifted drape*: `CONNECTION_LIFT_BASE_M` of clear
// air between the pavement and the flattened ground under it. Nothing told the
// scatter layers about that pavement, so grass kept growing on the ground
// beneath and its blades came up through the tarmac — visible as stubby tufts
// scattered over every taxiway, apron and road, sunk to their tips because the
// drape hides the lower part of each blade (INC-20260726T040431Z).
//
// The lift constant's own comment records the same symptom being met by
// *raising the drape* until the blades were hidden. That is a race the drape
// cannot win: a taller blade, a lower lift, or a bumpier pad brings the tufts
// straight back, and the lift is bounded above by how far pavement can float
// before it reads as a lip. The blades have to not be there.
//
// So every connection publishes the footprint it paves, in the same call that
// builds its mesh, and the scatter layers read it exactly as they already read
// building and runway footprints. Mesh and footprint come off the same
// centreline and the same width, so they cannot drift apart.

/// A kind of paved connection. Each carries its own width / colour / lift so the
/// network reads as distinct facility types. New variants extend the base's
/// infrastructure vocabulary (utility roads, rail, pipe runs, …).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ConnectionKind {
    /// Wide airside pavement linking runways to aprons/hangars. Dark asphalt.
    Taxiway,
    /// A filled parking pad (in front of a hangar). Dark asphalt, filled rect.
    Apron,
    /// Narrow landside service road linking buildings. Light concrete.
    Road,
    /// Very wide gravel haul-way from the assembly building to the launch pads —
    /// the future crawler-transporter route. Neutral gravel.
    Crawlerway,
}

impl ConnectionKind {
    fn style(self) -> ConnectionStyle {
        match self {
            // Wide, prominent airside pavement — a taxiway is roughly half a
            // runway's width and must read clearly, not as a thin line. Sits at
            // the top of the paving band so it reads over roads/crawlerways it
            // crosses.
            ConnectionKind::Taxiway => ConnectionStyle {
                width_m: 44.0,
                lift_m: CONNECTION_LIFT_BASE_M,
            },
            ConnectionKind::Apron => ConnectionStyle {
                width_m: 0.0,
                lift_m: CONNECTION_LIFT_BASE_M - 0.01,
            },
            ConnectionKind::Road => ConnectionStyle {
                width_m: 10.0,
                lift_m: CONNECTION_LIFT_BASE_M - 0.02,
            },
            ConnectionKind::Crawlerway => ConnectionStyle {
                width_m: 40.0,
                lift_m: CONNECTION_LIFT_BASE_M - 0.03,
            },
        }
    }

    /// The shared material for this connection kind, from [`BaseMaterials`].
    fn material(self, mats: &BaseMaterials) -> Handle<ShadowedStandardMaterial> {
        match self {
            // Taxiways and aprons are the same dark asphalt as the tarmac.
            ConnectionKind::Taxiway | ConnectionKind::Apron => mats.tarmac.clone(),
            ConnectionKind::Road => mats.road.clone(),
            ConnectionKind::Crawlerway => mats.crawlerway.clone(),
        }
    }
}

/// One connection mesh for a site, anchored at the site centre.
#[derive(Component)]
struct ConnectionVisual {
    site_id: StructureId,
    body_id: BodyId,
    center_body: DVec3,
    basis_body: DQuat,
}

/// Tracks what the editor's auto-taxiway mesh was last built for, so it only
/// rebuilds when the active site's structure set changes.
#[derive(Resource, Default)]
struct ConnectionsState {
    built: Option<(StructureId, u32)>,
    material: Option<Handle<ShadowedStandardMaterial>>,
}

pub(super) struct BaseEditorConnectionsPlugin;

impl Plugin for BaseEditorConnectionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConnectionsState>()
            .init_resource::<PavedFootprints>()
            .add_systems(Update, rebuild_connections.run_if(base_editor_open))
            // Same anchoring contract as `place::update_placed_transforms`: in
            // `SimStage::Sync` after `sync_solar_system_state` so it reads the
            // current-frame body pose (a bare unordered `Update` jittered the
            // paving at warp > 1×). Sync still runs while the editor is open.
            .add_systems(
                Update,
                update_connection_transforms
                    .in_set(SimStage::Sync)
                    .after(sync_solar_system_state),
            );
    }
}

#[allow(clippy::too_many_arguments)]
fn rebuild_connections(
    editor: Res<BaseEditor>,
    build: Res<BaseBuildState>,
    registry: Res<StructureRegistry>,
    sim: Res<SimulationState>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ShadowedStandardMaterial>>,
    mut state: ResMut<ConnectionsState>,
    mut paved: ResMut<PavedFootprints>,
    existing: Query<(Entity, &ConnectionVisual)>,
    root: Res<RealSpaceRoot>,
) {
    if editor.mode != BaseEditorMode::PlaceBuildings {
        return;
    }
    let Some(site_id) = editor.active_site else {
        return;
    };
    let Some(site) = registry.get(site_id).copied() else {
        return;
    };
    let StructurePlacement::FlattenTo { elevation_m, .. } = site.placement else {
        return;
    };
    let body_id = site.body_id;
    let Some(body) = sim.system.bodies.get(body_id) else {
        return;
    };
    let pad_r = body.radius_m + elevation_m;

    let up = site.anchor_dir;
    let heading = site.heading_tangent;
    let across = heading.cross(up).normalize();

    // Project each child structure onto the site tangent plane: (along, across,
    // bounding radius). Projection and the anchor basis below use the *same*
    // `across`, so the strip endpoints land exactly on the structures.
    let mut nodes: Vec<SiteNode> = Vec::new();
    for s in registry.sites_on(body_id) {
        if s.parent_site != Some(site_id) {
            continue;
        }
        let offset = (s.anchor_dir - up) * pad_r;
        nodes.push(SiteNode {
            along: offset.dot(heading),
            across: offset.dot(across),
            bounding: kind_bounding_m(&s.kind),
        });
    }

    // Rebuild when the site's structures change (place / delete / move), keyed
    // off the build-state revision the placement system bumps.
    let rev = build.structures_rev;
    if state.built == Some((site_id, rev)) {
        return;
    }
    state.built = Some((site_id, rev));

    // Drop the stale mesh for this site — and the footprints that went with
    // it, or a moved taxiway would keep clearing grass along its old route.
    for (entity, cv) in existing.iter() {
        if cv.site_id == site_id {
            commands.entity(entity).despawn();
        }
    }
    paved.clear_site(body_id, site_id);
    if nodes.len() < 2 {
        return; // nothing to connect
    }

    // The editor's generic auto-connection is a taxiway MST over everything on
    // the site (the foundation behaviour). The authored base builds richer typed
    // networks via `spawn_authored_*`.
    let edges = minimum_spanning_tree(&nodes);
    let style = ConnectionKind::Taxiway.style();
    let Some(mesh) = build_strip_mesh(&nodes, &edges, style.width_m, style.lift_m) else {
        return;
    };

    let basis_body = DQuat::from_mat3(&DMat3::from_cols(heading, up, across));
    let center_body = up * pad_r;
    let material = state
        .material
        .get_or_insert_with(|| {
            // Shadow-receiving (F6): the taxiway darkens under structures and
            // the craft like the flattened ground it paves over.
            materials.add(shadowed(StandardMaterial {
                base_color: Color::srgb(0.13, 0.13, 0.15),
                perceptual_roughness: 0.92,
                metallic: 0.0,
                ..default()
            }))
        })
        .clone();

    spawn_connection_entity(
        &mut commands,
        &mut meshes,
        &mut paved,
        material,
        root.entity,
        body_id,
        site_id,
        center_body,
        basis_body,
        mesh,
        paved_regions(
            &network_paved_rects(&nodes, &edges, style.width_m),
            up,
            heading,
            pad_r,
        ),
    );
}

/// Spawn a connection mesh entity anchored at the site centre. Shared by the
/// editor rebuild and the authored spawners.
#[allow(clippy::too_many_arguments)]
fn spawn_connection_entity(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    paved: &mut PavedFootprints,
    material: Handle<ShadowedStandardMaterial>,
    root: Entity,
    body_id: BodyId,
    site_id: StructureId,
    center_body: DVec3,
    basis_body: DQuat,
    mesh: Mesh,
    regions: Vec<ScatterRegion>,
) {
    // Recorded here rather than at each caller so a connection can never be
    // drawn without publishing the ground it covers.
    paved.extend(body_id, site_id, regions);
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(material),
        Transform::default(),
        Visibility::Inherited,
        CellCoord::ZERO,
        ChildOf(root),
        RenderLayers::layer(SHIP_LAYER),
        NotShadowCaster,
        ConnectionVisual {
            site_id,
            body_id,
            center_body,
            basis_body,
        },
        Name::new("Base Connection"),
    ));
}

/// Build a typed line network (taxiway / road / crawlerway) over authored nodes.
/// `nodes` are `(along, across, bounding)` in the site tangent frame at
/// `center_dir`/`heading`/`pad_r`. `edges` are explicit index pairs, or `None`
/// to link them all with a minimum spanning tree. Single spawn (no rev tracking);
/// shares the `site_id` so the editor's `rebuild_connections` replaces it if that
/// base is later edited.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_authored_network(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    paved: &mut PavedFootprints,
    mats: &BaseMaterials,
    root: Entity,
    body_id: BodyId,
    site_id: StructureId,
    center_dir: DVec3,
    heading: DVec3,
    pad_r: f64,
    kind: ConnectionKind,
    nodes: &[(f64, f64, f64)],
    edges: Option<&[(usize, usize)]>,
) {
    if nodes.len() < 2 {
        return;
    }
    let node_vec: Vec<SiteNode> = nodes
        .iter()
        .map(|&(a, c, b)| SiteNode {
            along: a,
            across: c,
            bounding: b,
        })
        .collect();
    let mst;
    let edges = match edges {
        Some(e) => e,
        None => {
            mst = minimum_spanning_tree(&node_vec);
            &mst
        }
    };
    let style = kind.style();
    let Some(mesh) = build_strip_mesh(&node_vec, edges, style.width_m, style.lift_m) else {
        return;
    };
    let (center_body, basis_body) = site_anchor(center_dir, heading, pad_r);
    spawn_connection_entity(
        commands,
        meshes,
        paved,
        kind.material(mats),
        root,
        body_id,
        site_id,
        center_body,
        basis_body,
        mesh,
        paved_regions(
            &network_paved_rects(&node_vec, edges, style.width_m),
            center_dir,
            heading,
            pad_r,
        ),
    );
}

/// Build an authored apron (a filled parking rectangle) at `(along, across)` in
/// the site frame with the given half-extents.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_authored_apron(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    paved: &mut PavedFootprints,
    mats: &BaseMaterials,
    root: Entity,
    body_id: BodyId,
    site_id: StructureId,
    center_dir: DVec3,
    heading: DVec3,
    pad_r: f64,
    along: f64,
    across: f64,
    half_along: f64,
    half_across: f64,
) {
    let style = ConnectionKind::Apron.style();
    let Some(mesh) = build_apron_mesh(along, across, half_along, half_across, style.lift_m) else {
        return;
    };
    let (center_body, basis_body) = site_anchor(center_dir, heading, pad_r);
    spawn_connection_entity(
        commands,
        meshes,
        paved,
        ConnectionKind::Apron.material(mats),
        root,
        body_id,
        site_id,
        center_body,
        basis_body,
        mesh,
        paved_regions(
            &[PavedRect {
                center: DVec2::new(along, across),
                dir: DVec2::X,
                half_along,
                half_across,
            }],
            center_dir,
            heading,
            pad_r,
        ),
    );
}

/// Build an authored **curved path** (taxiway / road / …): a waypoint polyline
/// whose interior corners are rounded into circular fillets of (up to)
/// `fillet_radius_m`, extruded to the kind's width. This is how the airside
/// taxiways read as real curved pavement instead of sharp zig-zag strips.
///
/// `lift_bias_m` nudges the strip's ground lift: a path that merges
/// tangentially into another (a link taxiway joining a parallel taxiway) sits
/// a few millimetres *lower*, so the overlap region renders as the one main
/// strip instead of z-fighting.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_authored_path(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    paved: &mut PavedFootprints,
    mats: &BaseMaterials,
    root: Entity,
    body_id: BodyId,
    site_id: StructureId,
    center_dir: DVec3,
    heading: DVec3,
    pad_r: f64,
    kind: ConnectionKind,
    points: &[DVec2],
    fillet_radius_m: f64,
    lift_bias_m: f32,
) {
    let style = kind.style();
    let center = fillet_path(points, fillet_radius_m);
    let Some(mesh) = build_path_mesh(&center, style.width_m, style.lift_m + lift_bias_m) else {
        return;
    };
    let (center_body, basis_body) = site_anchor(center_dir, heading, pad_r);
    spawn_connection_entity(
        commands,
        meshes,
        paved,
        kind.material(mats),
        root,
        body_id,
        site_id,
        center_body,
        basis_body,
        mesh,
        paved_regions(
            &path_paved_rects(&center, style.width_m),
            center_dir,
            heading,
            pad_r,
        ),
    );
}

/// Anchor the connection meshes in the body-fixed frame each frame, like
/// `place::update_placed_transforms` (in `SimStage::Sync` after
/// `sync_solar_system_state`, so it reads the current-frame body pose).
fn update_connection_transforms(
    solar: Res<SolarSystemState>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut connections: Query<(&ConnectionVisual, &mut CellCoord, &mut Transform)>,
) {
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let Ok(grid) = root_grid.single() else {
        return;
    };
    for (cv, mut cell, mut transform) in &mut connections {
        let Some(state) = states.get(cv.body_id) else {
            continue;
        };
        let orientation = state.orientation.normalize();
        let center_world = state.position + orientation * cv.center_body;
        let (next_cell, local) = grid.translation_to_grid(center_world);
        *cell = next_cell;
        transform.translation = local;
        transform.rotation = (orientation * cv.basis_body).as_quat();
    }
}
