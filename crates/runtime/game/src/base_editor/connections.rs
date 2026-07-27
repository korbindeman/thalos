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

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::RenderLayers;
use bevy::light::NotShadowCaster;
use bevy::math::{DMat3, DQuat, DVec2, DVec3};
use bevy::platform::collections::HashMap;
use bevy::mesh::{Indices, PrimitiveTopology};
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

use thalos_body_render::{ScatterRegion, ScatterTreatment};
use thalos_terrain::TerrainFlatten;

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

/// Extra clearance (m) outside the pavement edge. Small — grass should meet the
/// tarmac, just not stand in it.
const PAVED_CLEAR_MARGIN_M: f64 = 0.6;
/// Ramp width (m) outside the cleared rectangle. `classify_scatter` clears
/// wherever the footprint weight exceeds `SCATTER_CLEAR_W` (0.05), i.e. across
/// almost all of this ramp, so it stays narrow.
const PAVED_CLEAR_RAMP_M: f64 = 1.0;

/// Every connection's paved footprint, as scatter regions the grass and
/// vegetation layers clear against.
///
/// Keyed by site so a rebuild (the editor re-running `rebuild_connections`
/// after a placement change) replaces that site's pavement wholesale instead of
/// accumulating stale strips.
///
/// **Single writer:** `spawn_connection_entity` — the one funnel every
/// connection spawner already goes through.
#[derive(Resource, Default)]
pub struct PavedFootprints {
    by_site: HashMap<(BodyId, StructureId), Vec<ScatterRegion>>,
}

impl PavedFootprints {
    /// Drop everything recorded for a site — called before its connections are
    /// respawned.
    pub fn clear_site(&mut self, body_id: BodyId, site_id: StructureId) {
        self.by_site.remove(&(body_id, site_id));
    }

    fn extend(
        &mut self,
        body_id: BodyId,
        site_id: StructureId,
        regions: impl IntoIterator<Item = ScatterRegion>,
    ) {
        self.by_site
            .entry((body_id, site_id))
            .or_default()
            .extend(regions);
    }

    /// Every paved region on a body, for the scatter layers' region list.
    pub fn regions_on(&self, body_id: BodyId) -> impl Iterator<Item = &ScatterRegion> {
        self.by_site
            .iter()
            .filter(move |((b, _), _)| *b == body_id)
            .flat_map(|(_, regions)| regions.iter())
    }
}

/// One paved rectangle in the site tangent frame: centre `(along, across)`,
/// unit direction `dir` of its long axis, and half-extents.
#[derive(Clone, Copy)]
pub(super) struct PavedRect {
    center: DVec2,
    dir: DVec2,
    half_along: f64,
    half_across: f64,
}

/// The paved rectangles a width-`width_m` strip along `center` covers — one per
/// polyline segment. A filleted path is already dense enough that per-segment
/// rectangles track the curve within the clear ramp.
pub(super) fn path_paved_rects(center: &[DVec2], width_m: f64) -> Vec<PavedRect> {
    center
        .windows(2)
        .filter_map(|seg| {
            let (a, b) = (seg[0], seg[1]);
            let delta = b - a;
            let len = delta.length();
            if len < 1.0e-6 {
                return None;
            }
            Some(PavedRect {
                center: (a + b) * 0.5,
                dir: delta / len,
                half_along: len * 0.5,
                half_across: width_m * 0.5,
            })
        })
        .collect()
}

/// The paved rectangles a strip network over `edges` covers.
fn network_paved_rects(nodes: &[Node], edges: &[(usize, usize)], width_m: f64) -> Vec<PavedRect> {
    edges
        .iter()
        .flat_map(|&(i, j)| {
            let a = DVec2::new(nodes[i].along, nodes[i].across);
            let b = DVec2::new(nodes[j].along, nodes[j].across);
            path_paved_rects(&[a, b], width_m)
        })
        .collect()
}

/// Convert site-frame paved rectangles into body-fixed scatter regions.
///
/// The frame matches `site_anchor` / the mesh builders exactly: local `x` is
/// `heading` (along), local `z` is `heading × center_dir` (across), and a
/// site-frame point sits at `center_dir · pad_r + heading · along + across · c`.
/// `pad_r` doubles as the footprint's reference radius and the elevation is
/// left at zero: a scatter region is only ever asked for `weight(dir)`, never
/// for a height, and `pad_r` is the exact radius the mesh's own site-frame
/// points were placed at — so the footprint converts directions to tangent
/// offsets on the same sphere the pavement was built on.
fn paved_regions(
    rects: &[PavedRect],
    center_dir: DVec3,
    heading: DVec3,
    pad_r: f64,
) -> Vec<ScatterRegion> {
    let across_v = heading.cross(center_dir).normalize();
    rects
        .iter()
        .map(|r| {
            let point = center_dir * pad_r + heading * r.center.x + across_v * r.center.y;
            let rect_center = point.normalize();
            // Long axis in the tangent plane at the rectangle's *own* centre,
            // re-orthogonalised there — the site frame's axes tilt by
            // offset/radius across a kilometre-scale base.
            let along_v = heading * r.dir.x + across_v * r.dir.y;
            let tangent_along = (along_v - rect_center * along_v.dot(rect_center)).normalize();
            let tangent_across = tangent_along.cross(rect_center).normalize();
            ScatterRegion {
                footprint: TerrainFlatten::new(
                    rect_center,
                    tangent_along,
                    tangent_across,
                    r.half_along + PAVED_CLEAR_MARGIN_M,
                    r.half_across + PAVED_CLEAR_MARGIN_M,
                    PAVED_CLEAR_RAMP_M,
                    0.0,
                    pad_r,
                ),
                treatment: ScatterTreatment::Clear,
            }
        })
        .collect()
}

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

/// Base lift (m) for the paving band, matched to the runway asphalt
/// (`runway::RUNWAY_ASPHALT_LIFT_M` = 0.12) — the proven "reads as paving on the
/// ground, not a floating lip" value. The old ~4 cm lift lost to depth-buffer
/// imprecision at the Space Center god-view distance and to the basin's residual
/// cut/fill height jitter (the runway skirt drops a full 0.6 m for the same
/// reason), so the pavement z-fought the flattened tiles and short lawn grass
/// poked through. Connections carry no skirt, so the surface itself has to clear
/// the ground outright. Each kind sits a hair above/below its neighbours (below)
/// so overlapping pavement sorts cleanly.
const CONNECTION_LIFT_BASE_M: f32 = 0.12;

/// Visual style of a connection kind.
struct ConnectionStyle {
    /// Strip width (m) for line networks (ignored by aprons, which are sized
    /// explicitly).
    width_m: f64,
    /// Lift above the pad surface (m) so the paving reads on the ground without
    /// z-fighting. Overlapping kinds are separated by a small delta so the more
    /// prominent airside pavement reads over what it crosses (taxiway over road).
    lift_m: f32,
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

/// Build a strip network mesh in the site-local frame (X = along/heading, Y =
/// up, Z = across). Each edge is a flat strip of `width_m` from one node's edge
/// to the next's, inset by their bounding radii so the paving meets the
/// footprints rather than overlapping them.
fn build_strip_mesh(
    nodes: &[Node],
    edges: &[(usize, usize)],
    width_m: f64,
    lift_m: f32,
) -> Option<Mesh> {
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
            continue; // footprints touch — no paving between them
        }
        let perp = (-dir.1, dir.0);
        let hw = width_m * 0.5;
        push_quad(
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut indices,
            lift_m,
            [
                (p0.0 + perp.0 * hw, p0.1 + perp.1 * hw),
                (p0.0 - perp.0 * hw, p0.1 - perp.1 * hw),
                (p1.0 + perp.0 * hw, p1.1 + perp.1 * hw),
                (p1.0 - perp.0 * hw, p1.1 - perp.1 * hw),
            ],
        );
    }

    finish_mesh(positions, normals, uvs, indices)
}

/// Build a filled apron rectangle centred at `(along, across)` in the site frame
/// with the given half-extents (m).
fn build_apron_mesh(
    along: f64,
    across: f64,
    half_along: f64,
    half_across: f64,
    lift_m: f32,
) -> Option<Mesh> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // Corner order matters: `push_quad` winds `[0,2,1, 1,2,3]`, so the pair
    // sharing `along0` must run `+across → −across` for the face to point up
    // (the previous order wound it downward — a backface-culled, invisible
    // apron).
    push_quad(
        &mut positions,
        &mut normals,
        &mut uvs,
        &mut indices,
        lift_m,
        [
            (along - half_along, across + half_across),
            (along - half_along, across - half_across),
            (along + half_along, across + half_across),
            (along + half_along, across - half_across),
        ],
    );
    finish_mesh(positions, normals, uvs, indices)
}

/// Tessellate a waypoint polyline into a dense centreline, replacing each
/// interior corner with a circular fillet of (up to) `radius_m` — the curved
/// taxiway/road geometry. The radius is clamped so an arc never eats more than
/// half of either adjacent segment (two nearby corners each keep their own
/// fillet). Points are `(along, across)` in the site frame.
fn fillet_path(points: &[DVec2], radius_m: f64) -> Vec<DVec2> {
    let mut out: Vec<DVec2> = Vec::with_capacity(points.len() * 8);
    let Some((&first, rest)) = points.split_first() else {
        return out;
    };
    out.push(first);
    for i in 1..points.len().saturating_sub(1) {
        let (a, p, b) = (points[i - 1], points[i], points[i + 1]);
        let u = (p - a).normalize_or_zero(); // incoming direction
        let v = (b - p).normalize_or_zero(); // outgoing direction
        if u == DVec2::ZERO || v == DVec2::ZERO {
            continue;
        }
        // Turn angle at the corner (0 = straight through).
        let turn = (-u).dot(v).clamp(-1.0, 1.0).acos();
        let turn = std::f64::consts::PI - turn;
        if turn < 1.0e-3 {
            out.push(p);
            continue;
        }
        // Tangent distance from the corner to the arc's start/end, clamped to
        // half of each adjacent segment; the effective radius follows.
        let t_max = 0.5 * (p - a).length().min((b - p).length());
        let half_tan = (turn * 0.5).tan();
        let t = (radius_m * half_tan).min(t_max);
        let r_eff = t / half_tan;
        let start = p - u * t;
        let end = p + v * t;
        // Arc centre sits perpendicular to the incoming direction, on the side
        // the path turns toward.
        let side = u.perp_dot(v).signum();
        let center = start + DVec2::new(-u.y, u.x) * (side * r_eff);
        let a0 = (start - center).to_angle();
        let steps = (turn / 0.12).ceil().max(1.0) as usize; // ~7° per step
        for k in 0..=steps {
            let ang = a0 + side * turn * (k as f64 / steps as f64);
            out.push(center + DVec2::from_angle(ang) * r_eff);
        }
        let _ = end; // the final arc sample lands on `end`
    }
    if let Some(&last) = rest.last() {
        out.push(last);
    }
    out
}

/// Extrude a tessellated centreline into a flat strip of `width_m` (a triangle
/// strip with one vertex pair per point; directions are averaged at interior
/// points, so the fillets' small angle steps join smoothly).
fn build_path_mesh(center: &[DVec2], width_m: f64, lift_m: f32) -> Option<Mesh> {
    if center.len() < 2 {
        return None;
    }
    let hw = width_m * 0.5;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(center.len() * 2);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(center.len() * 2);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(center.len() * 2);
    let mut indices: Vec<u32> = Vec::with_capacity((center.len() - 1) * 6);
    for (i, p) in center.iter().enumerate() {
        let dir_in = if i > 0 {
            (center[i] - center[i - 1]).normalize_or_zero()
        } else {
            DVec2::ZERO
        };
        let dir_out = if i + 1 < center.len() {
            (center[i + 1] - center[i]).normalize_or_zero()
        } else {
            DVec2::ZERO
        };
        let dir = (dir_in + dir_out).normalize_or(if dir_in != DVec2::ZERO {
            dir_in
        } else {
            dir_out
        });
        let perp = DVec2::new(-dir.y, dir.x);
        for side in [1.0, -1.0] {
            let q = *p + perp * (hw * side);
            positions.push([q.x as f32, lift_m, q.y as f32]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([0.0, 0.0]);
        }
        if i > 0 {
            let b = (2 * i) as u32;
            indices.extend_from_slice(&[b - 2, b, b - 1, b - 1, b, b + 1]);
        }
    }
    finish_mesh(positions, normals, uvs, indices)
}

/// Push one flat quad (Y-up) at `lift` from four `(along, across)` corners
/// ordered so `[0,2,1, 1,2,3]` winds consistently.
fn push_quad(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    lift: f32,
    corners: [(f64, f64); 4],
) {
    let base = positions.len() as u32;
    for (cx, cz) in corners {
        positions.push([cx as f32, lift, cz as f32]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([0.0, 0.0]);
    }
    indices.extend_from_slice(&[base, base + 2, base + 1, base + 1, base + 2, base + 3]);
}

fn finish_mesh(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
) -> Option<Mesh> {
    if indices.is_empty() {
        return None;
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    Some(mesh)
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

/// Body-fixed anchor basis + centre for a site's connection meshes (all built in
/// the site-local `(heading, up, across)` frame).
fn site_anchor(center_dir: DVec3, heading: DVec3, pad_r: f64) -> (DVec3, DQuat) {
    let across = heading.cross(center_dir).normalize();
    let basis = DQuat::from_mat3(&DMat3::from_cols(heading, center_dir, across));
    (center_dir * pad_r, basis)
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
    let node_vec: Vec<Node> = nodes
        .iter()
        .map(|&(a, c, b)| Node {
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
