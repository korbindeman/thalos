//! Auto-connections: tarmac strips linking the structures on a site.
//!
//! When the active site's structure set changes, this rebuilds a single combined
//! tarmac mesh that connects every building / launchpad on the pad along a
//! minimum spanning tree (so the network is connected with the least paving).
//! The mesh is built in the site's local tangent frame and anchored every frame
//! like the structures themselves (a root-grid big_space child posed in f64), so
//! it stays put at high warp and persists when the editor closes.
//!
//! Foundation slice: one connection type (tarmac). Roads / resource lines and
//! user-drawn routing are follow-ups; the MST + strip mesh here is the seam they
//! extend. Rebuild triggers on the *count* of structures changing (place /
//! delete), which is all the foundation editor can do to a site.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::light::NotShadowCaster;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_world::BodyId;

use crate::coords::SHIP_LAYER;
use crate::rendering::real_space::RealSpaceRoot;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::structures::{StructureId, StructurePlacement, StructureRegistry};

use super::place::kind_bounding_m;
use super::{BaseEditor, BaseEditorMode, base_editor_open};

/// Tarmac strip width, metres.
const STRIP_W: f64 = 3.0;
/// Lift above the pad surface so the tarmac reads on the ground without z-fight.
const STRIP_LIFT_M: f32 = 0.06;

/// The combined tarmac mesh for one site, anchored at the site centre.
#[derive(Component)]
struct ConnectionVisual {
    site_id: StructureId,
    body_id: BodyId,
    center_body: DVec3,
    basis_body: DQuat,
}

/// Tracks what the connection mesh was last built for, so it only rebuilds when
/// the active site's structure set changes.
#[derive(Resource, Default)]
struct ConnectionsState {
    built: Option<(StructureId, usize)>,
    material: Option<Handle<StandardMaterial>>,
}

pub(super) struct BaseEditorConnectionsPlugin;

impl Plugin for BaseEditorConnectionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConnectionsState>()
            .add_systems(Update, rebuild_connections.run_if(base_editor_open))
            // Ungated so the network stays anchored in flight too.
            .add_systems(Update, update_connection_transforms);
    }
}

#[allow(clippy::too_many_arguments)]
fn rebuild_connections(
    editor: Res<BaseEditor>,
    registry: Res<StructureRegistry>,
    sim: Res<SimulationState>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut state: ResMut<ConnectionsState>,
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
    let mut nodes: Vec<Node> = Vec::new();
    for s in registry.sites_on(body_id) {
        if s.parent_site != Some(site_id) {
            continue;
        }
        let offset = (s.anchor_dir - up) * pad_r;
        nodes.push(Node {
            along: offset.dot(heading),
            across: offset.dot(across),
            bounding: kind_bounding_m(&s.kind),
        });
    }

    let count = nodes.len();
    if state.built == Some((site_id, count)) {
        return;
    }
    state.built = Some((site_id, count));

    // Drop the stale mesh for this site.
    for (entity, cv) in existing.iter() {
        if cv.site_id == site_id {
            commands.entity(entity).despawn();
        }
    }
    if count < 2 {
        return; // nothing to connect
    }

    let edges = minimum_spanning_tree(&nodes);
    let Some(mesh) = build_connection_mesh(&nodes, &edges) else {
        return;
    };

    let basis_body = DQuat::from_mat3(&DMat3::from_cols(heading, up, across));
    let center_body = up * pad_r;
    let material = state
        .material
        .get_or_insert_with(|| {
            materials.add(StandardMaterial {
                base_color: Color::srgb(0.14, 0.14, 0.16),
                perceptual_roughness: 0.92,
                metallic: 0.0,
                ..default()
            })
        })
        .clone();

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(material),
        Transform::default(),
        Visibility::Inherited,
        CellCoord::ZERO,
        ChildOf(root.entity),
        RenderLayers::layer(SHIP_LAYER),
        NotShadowCaster,
        ConnectionVisual {
            site_id,
            body_id,
            center_body,
            basis_body,
        },
        Name::new("Base Connections"),
    ));
}

struct Node {
    along: f64,
    across: f64,
    bounding: f64,
}

/// Prim's MST over the node positions (O(n²) — n is small). Returns edges as
/// index pairs.
fn minimum_spanning_tree(nodes: &[Node]) -> Vec<(usize, usize)> {
    let n = nodes.len();
    let mut in_tree = vec![false; n];
    let mut edges = Vec::with_capacity(n.saturating_sub(1));
    in_tree[0] = true;
    let dist2 = |a: &Node, b: &Node| {
        let dx = a.along - b.along;
        let dz = a.across - b.across;
        dx * dx + dz * dz
    };
    for _ in 1..n {
        let mut best: Option<(usize, usize, f64)> = None;
        for (i, ni) in nodes.iter().enumerate() {
            if !in_tree[i] {
                continue;
            }
            for (j, nj) in nodes.iter().enumerate() {
                if in_tree[j] {
                    continue;
                }
                let d = dist2(ni, nj);
                if best.is_none_or(|(_, _, bd)| d < bd) {
                    best = Some((i, j, d));
                }
            }
        }
        let Some((i, j, _)) = best else { break };
        in_tree[j] = true;
        edges.push((i, j));
    }
    edges
}

/// Build the combined tarmac mesh in the site-local frame (X = along/heading,
/// Y = up, Z = across). Each edge is a flat strip from one structure's edge to
/// the next's, inset by their bounding radii so the tarmac meets the footprints
/// rather than overlapping them.
fn build_connection_mesh(nodes: &[Node], edges: &[(usize, usize)]) -> Option<Mesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for &(i, j) in edges {
        let (a, b) = (&nodes[i], &nodes[j]);
        let dx = b.along - a.along;
        let dz = b.across - a.across;
        let len = (dx * dx + dz * dz).sqrt();
        if len < 1e-3 {
            continue;
        }
        let dir = (dx / len, dz / len);
        // Inset endpoints to the structure edges.
        let p0 = (a.along + dir.0 * a.bounding, a.across + dir.1 * a.bounding);
        let p1 = (b.along - dir.0 * b.bounding, b.across - dir.1 * b.bounding);
        let seg = ((p1.0 - p0.0), (p1.1 - p0.1));
        if (seg.0 * seg.0 + seg.1 * seg.1).sqrt() < 1e-2 {
            continue; // footprints touch — no tarmac between them
        }
        let perp = (-dir.1, dir.0);
        let hw = STRIP_W * 0.5;
        let base = positions.len() as u32;
        let corners = [
            (p0.0 + perp.0 * hw, p0.1 + perp.1 * hw),
            (p0.0 - perp.0 * hw, p0.1 - perp.1 * hw),
            (p1.0 + perp.0 * hw, p1.1 + perp.1 * hw),
            (p1.0 - perp.0 * hw, p1.1 - perp.1 * hw),
        ];
        for (cx, cz) in corners {
            positions.push([cx as f32, STRIP_LIFT_M, cz as f32]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([0.0, 0.0]);
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base + 1, base + 2, base + 3]);
    }

    if indices.is_empty() {
        return None;
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    Some(mesh)
}

/// Anchor the connection mesh in the body-fixed frame each frame (ungated, like
/// `place::update_placed_transforms`).
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
