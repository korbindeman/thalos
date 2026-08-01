//! Connection **geometry**: the taxiway/apron network — minimum spanning tree
//! over base sites, filleted centre paths, strip/apron meshes, and the paved
//! footprint regions scatter clears against.
//!
//! Pure appearance, same boundary as [`crate::runway_geometry`]: the
//! `rebuild_connections` driver and the per-frame transform sync stay in
//! `thalos_runtime`.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::math::{DMat3, DQuat, DVec2, DVec3};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;
use thalos_body_render::{ScatterRegion, ScatterTreatment};
use thalos_game_state::structures::StructureId;
use thalos_terrain::TerrainFlatten;
use thalos_world::BodyId;

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
    pub by_site: HashMap<(BodyId, StructureId), Vec<ScatterRegion>>,
}

/// Extra clearance (m) outside the pavement edge. Small — grass should meet the
/// tarmac, just not stand in it.
pub const PAVED_CLEAR_MARGIN_M: f64 = 0.6;

/// Ramp width (m) outside the cleared rectangle. `classify_scatter` clears
/// wherever the footprint weight exceeds `SCATTER_CLEAR_W` (0.05), i.e. across
/// almost all of this ramp, so it stays narrow.
pub const PAVED_CLEAR_RAMP_M: f64 = 1.0;

/// Base lift (m) for the paving band, matched to the runway asphalt
/// (`runway::RUNWAY_ASPHALT_LIFT_M` = 0.12) — the proven "reads as paving on the
/// ground, not a floating lip" value. The old ~4 cm lift lost to depth-buffer
/// imprecision at the Space Center god-view distance and to the basin's residual
/// cut/fill height jitter (the runway skirt drops a full 0.6 m for the same
/// reason), so the pavement z-fought the flattened tiles and short lawn grass
/// poked through. Connections carry no skirt, so the surface itself has to clear
/// the ground outright. Each kind sits a hair above/below its neighbours (below)
/// so overlapping pavement sorts cleanly.
pub const CONNECTION_LIFT_BASE_M: f32 = 0.12;

/// The paved rectangles a strip network over `edges` covers.
pub fn network_paved_rects(
    nodes: &[Node],
    edges: &[(usize, usize)],
    width_m: f64,
) -> Vec<PavedRect> {
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
pub fn paved_regions(
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

/// Visual style of a connection kind.
pub struct ConnectionStyle {
    /// Strip width (m) for line networks (ignored by aprons, which are sized
    /// explicitly).
    pub width_m: f64,
    /// Lift above the pad surface (m) so the paving reads on the ground without
    /// z-fighting. Overlapping kinds are separated by a small delta so the more
    /// prominent airside pavement reads over what it crosses (taxiway over road).
    pub lift_m: f32,
}

pub struct Node {
    pub along: f64,
    pub across: f64,
    pub bounding: f64,
}

/// Prim's MST over the node positions (O(n²) — n is small). Returns edges as
/// index pairs.
pub fn minimum_spanning_tree(nodes: &[Node]) -> Vec<(usize, usize)> {
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
pub fn build_strip_mesh(
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
pub fn build_apron_mesh(
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
pub fn fillet_path(points: &[DVec2], radius_m: f64) -> Vec<DVec2> {
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
pub fn build_path_mesh(center: &[DVec2], width_m: f64, lift_m: f32) -> Option<Mesh> {
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
pub fn push_quad(
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

pub fn finish_mesh(
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

/// Body-fixed anchor basis + centre for a site's connection meshes (all built in
/// the site-local `(heading, up, across)` frame).
pub fn site_anchor(center_dir: DVec3, heading: DVec3, pad_r: f64) -> (DVec3, DQuat) {
    let across = heading.cross(center_dir).normalize();
    let basis = DQuat::from_mat3(&DMat3::from_cols(heading, center_dir, across));
    (center_dir * pad_r, basis)
}

/// One paved rectangle in the site tangent frame: centre `(along, across)`,
/// unit direction `dir` of its long axis, and half-extents.
#[derive(Clone, Copy)]
pub struct PavedRect {
    pub center: DVec2,
    pub dir: DVec2,
    pub half_along: f64,
    pub half_across: f64,
}

/// The paved rectangles a width-`width_m` strip along `center` covers — one per
/// polyline segment. A filleted path is already dense enough that per-segment
/// rectangles track the curve within the clear ramp.
pub fn path_paved_rects(center: &[DVec2], width_m: f64) -> Vec<PavedRect> {
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

impl PavedFootprints {
    /// Drop everything recorded for a site — called before its connections are
    /// respawned.
    pub fn clear_site(&mut self, body_id: BodyId, site_id: StructureId) {
        self.by_site.remove(&(body_id, site_id));
    }

    pub fn extend(
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
