//! Sparse quadtree storage + L2 RAM cache (spec §5, §10).
//!
//! [`FieldCache`] is a per-field cache of materialised values over the
//! cube-sphere. Each face holds a quadtree whose nodes are one of:
//!
//! - **Uniform** — a constant region, stored as a single value (no per-texel
//!   cost). Detected by probing corners + centre against `uniform_eps`.
//! - **Leaf** — a dense `(leaf_size+1)²` corner-aligned tile, sampled at the
//!   node's natural LOD and bilinearly interpolated.
//! - **Subdivided** — four children.
//!
//! Materialisation is **lazy** (only nodes on a query path are built) and
//! **cached** (built nodes persist), so this is the L2/RAM tier sitting over
//! L4 generation (the [`Planet`] sampler). Uniform compression means a flat
//! region — a constant field, an unauthored area — costs one node regardless
//! of query LOD, so resolution is unbounded only where detail actually exists.
//!
//! The cache materialises the field's *final* (procedural ∘ overlay) value via
//! [`Planet::sample_field`]. Splitting into independently-invalidated
//! procedural and overlay quadtrees (spec §5) is a later refinement; the
//! storage mechanics — sparse, uniform-compressed, lazy — are the same.
//!
//! Disk (L3) and GPU (L1) tiers are out of scope for this increment; the
//! migration adds L3 in phase P4.

use std::f32::consts::FRAC_PI_2;

use glam::Vec3;

use crate::pipeline::cubesphere::{FACE_COUNT, dir_to_face_uv, face_uv_to_dir};
use crate::pipeline::planet::Planet;

/// A dense corner-aligned tile: `(size+1)²` values over a node's uv square.
#[derive(Debug, Clone)]
struct LeafTile {
    /// Cells per side; the value grid is `(size+1) × (size+1)`.
    size: u32,
    values: Vec<f32>,
}

impl LeafTile {
    fn bilinear(&self, lu: f32, lv: f32) -> f32 {
        let n = self.size;
        let stride = n + 1;
        let fx = lu.clamp(0.0, 1.0) * n as f32;
        let fy = lv.clamp(0.0, 1.0) * n as f32;
        let x0 = (fx.floor() as u32).min(n.saturating_sub(1));
        let y0 = (fy.floor() as u32).min(n.saturating_sub(1));
        let x1 = (x0 + 1).min(n);
        let y1 = (y0 + 1).min(n);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let at = |x: u32, y: u32| self.values[(y * stride + x) as usize];
        let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
        let bot = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
        top + (bot - top) * ty
    }
}

#[derive(Debug, Clone)]
enum Node {
    /// Not yet visited.
    Unmaterialized,
    /// Whole node is one value.
    Uniform(f32),
    /// Dense tile at this node's resolution.
    Leaf(LeafTile),
    /// Four children, quadrant order `[(-u,-v), (+u,-v), (-u,+v), (+u,+v)]`.
    Subdivided(Box<[Node; 4]>),
}

/// Immutable per-sample parameters threaded through node materialisation.
struct NodeCtx<'a> {
    planet: &'a Planet,
    field: &'a str,
    leaf_size: u32,
    uniform_eps: f32,
    radius_m: f32,
    target_depth: u32,
}

/// Per-field quadtree cache over the cube-sphere.
#[derive(Debug, Clone)]
pub struct FieldCache {
    field: String,
    leaf_size: u32,
    max_depth: u32,
    uniform_eps: f32,
    radius_m: f32,
    faces: [Node; FACE_COUNT as usize],
    materialized_nodes: u32,
}

impl FieldCache {
    /// Create a cache for `field` on a body of `radius_m`.
    ///
    /// `leaf_size` is cells per leaf tile (64 is the spec's starting default).
    /// `max_depth` caps subdivision (and so the finest resolution).
    /// `uniform_eps` is the value spread below which a region collapses to a
    /// single value.
    pub fn new(
        field: impl Into<String>,
        radius_m: f32,
        leaf_size: u32,
        max_depth: u32,
        uniform_eps: f32,
    ) -> Self {
        Self {
            field: field.into(),
            leaf_size: leaf_size.max(1),
            max_depth,
            uniform_eps,
            radius_m,
            faces: core::array::from_fn(|_| Node::Unmaterialized),
            materialized_nodes: 0,
        }
    }

    /// Number of nodes materialised so far (diagnostics / tests).
    pub fn materialized_nodes(&self) -> u32 {
        self.materialized_nodes
    }

    /// Sample the cached field at `dir`, materialising along the path as needed.
    /// Matches [`Planet::sample_field`] at leaf-grid corners and bilinearly
    /// interpolates between them.
    pub fn sample(&mut self, planet: &Planet, dir: Vec3, lod_m: f32) -> f32 {
        let (face, u, v) = dir_to_face_uv(dir);
        let ctx = NodeCtx {
            planet,
            field: &self.field,
            leaf_size: self.leaf_size,
            uniform_eps: self.uniform_eps,
            radius_m: self.radius_m,
            target_depth: self.target_depth(lod_m),
        };
        // Disjoint field borrows: faces vs. the counter.
        let counter = &mut self.materialized_nodes;
        let node = &mut self.faces[face as usize];
        sample_node(node, &ctx, counter, face, 0, [0.0, 0.0, 1.0, 1.0], u, v)
    }

    /// Quadtree depth whose leaf-texel arc spacing best matches `lod_m`.
    fn target_depth(&self, lod_m: f32) -> u32 {
        let face_arc_m = FRAC_PI_2 * self.radius_m;
        let ideal = face_arc_m / (self.leaf_size as f32 * lod_m.max(1e-3));
        let depth = ideal.max(1.0).log2().round();
        (depth.max(0.0) as u32).min(self.max_depth)
    }
}

/// Arc spacing (metres) between adjacent leaf texels at `depth`.
fn node_lod_m(ctx: &NodeCtx, depth: u32) -> f32 {
    let face_arc_m = FRAC_PI_2 * ctx.radius_m;
    (face_arc_m / ((1u32 << depth) as f32 * ctx.leaf_size as f32)).max(1e-3)
}

/// `[u0, v0, u1, v1]` bounds of a node in face-uv.
type Bounds = [f32; 4];

fn sample_node(
    node: &mut Node,
    ctx: &NodeCtx,
    counter: &mut u32,
    face: u32,
    depth: u32,
    bounds: Bounds,
    u: f32,
    v: f32,
) -> f32 {
    match node {
        Node::Uniform(value) => *value,
        Node::Leaf(tile) => {
            let (lu, lv) = local_uv(bounds, u, v);
            tile.bilinear(lu, lv)
        }
        Node::Subdivided(children) => {
            let (quadrant, child_bounds) = descend(bounds, u, v);
            sample_node(
                &mut children[quadrant],
                ctx,
                counter,
                face,
                depth + 1,
                child_bounds,
                u,
                v,
            )
        }
        Node::Unmaterialized => {
            *counter += 1;
            if depth >= ctx.target_depth {
                // Terminal: a dense leaf, collapsed to uniform if flat.
                let tile = sample_grid(ctx, face, depth, bounds);
                if let Some(uniform) = uniform_value(&tile.values, ctx.uniform_eps) {
                    *node = Node::Uniform(uniform);
                    uniform
                } else {
                    let (lu, lv) = local_uv(bounds, u, v);
                    let value = tile.bilinear(lu, lv);
                    *node = Node::Leaf(tile);
                    value
                }
            } else {
                // Interior: collapse if the region probes flat, else subdivide.
                let (flat, probe_value) = probe(ctx, face, depth, bounds);
                if flat {
                    *node = Node::Uniform(probe_value);
                    probe_value
                } else {
                    *node = Node::Subdivided(Box::new([
                        Node::Unmaterialized,
                        Node::Unmaterialized,
                        Node::Unmaterialized,
                        Node::Unmaterialized,
                    ]));
                    let Node::Subdivided(children) = node else {
                        unreachable!("just assigned Subdivided");
                    };
                    let (quadrant, child_bounds) = descend(bounds, u, v);
                    sample_node(
                        &mut children[quadrant],
                        ctx,
                        counter,
                        face,
                        depth + 1,
                        child_bounds,
                        u,
                        v,
                    )
                }
            }
        }
    }
}

/// Map a global `(u, v)` to the node-local `(0..1, 0..1)`.
fn local_uv(bounds: Bounds, u: f32, v: f32) -> (f32, f32) {
    let [u0, v0, u1, v1] = bounds;
    (
        ((u - u0) / (u1 - u0)).clamp(0.0, 1.0),
        ((v - v0) / (v1 - v0)).clamp(0.0, 1.0),
    )
}

/// Pick the quadrant of `(u, v)` and return its child bounds.
fn descend(bounds: Bounds, u: f32, v: f32) -> (usize, Bounds) {
    let [u0, v0, u1, v1] = bounds;
    let mu = 0.5 * (u0 + u1);
    let mv = 0.5 * (v0 + v1);
    let qx = (u >= mu) as usize;
    let qy = (v >= mv) as usize;
    let cb = [
        if qx == 0 { u0 } else { mu },
        if qy == 0 { v0 } else { mv },
        if qx == 0 { mu } else { u1 },
        if qy == 0 { mv } else { v1 },
    ];
    (qx + qy * 2, cb)
}

fn sample_grid(ctx: &NodeCtx, face: u32, depth: u32, bounds: Bounds) -> LeafTile {
    let [u0, v0, u1, v1] = bounds;
    let size = ctx.leaf_size;
    let corners = size + 1;
    let lod = node_lod_m(ctx, depth);
    let mut values = Vec::with_capacity((corners * corners) as usize);
    for gy in 0..corners {
        let v = u0_lerp(v0, v1, gy, size);
        for gx in 0..corners {
            let u = u0_lerp(u0, u1, gx, size);
            let dir = face_uv_to_dir(face, u, v);
            values.push(ctx.planet.sample_field(ctx.field, dir, lod).unwrap_or(0.0));
        }
    }
    LeafTile { size, values }
}

fn u0_lerp(a: f32, b: f32, step: u32, size: u32) -> f32 {
    a + (b - a) * (step as f32 / size as f32)
}

/// Probe corners + centre; return `(is_flat, mean)`.
fn probe(ctx: &NodeCtx, face: u32, depth: u32, bounds: Bounds) -> (bool, f32) {
    let [u0, v0, u1, v1] = bounds;
    let mu = 0.5 * (u0 + u1);
    let mv = 0.5 * (v0 + v1);
    let lod = node_lod_m(ctx, depth);
    let pts = [
        (u0, v0),
        (u1, v0),
        (u0, v1),
        (u1, v1),
        (mu, mv),
    ];
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0;
    for (u, v) in pts {
        let value = ctx
            .planet
            .sample_field(ctx.field, face_uv_to_dir(face, u, v), lod)
            .unwrap_or(0.0);
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }
    (max - min <= ctx.uniform_eps, sum / pts.len() as f32)
}

/// `Some(mean)` if the spread of `values` is within `eps`, else `None`.
fn uniform_value(values: &[f32], eps: f32) -> Option<f32> {
    if values.is_empty() {
        return Some(0.0);
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0;
    for &value in values {
        min = min.min(value);
        max = max.max(value);
        sum += value;
    }
    if max - min <= eps {
        Some(sum / values.len() as f32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::expr::Expr;
    use crate::pipeline::field::Field;
    use crate::pipeline::planet::{Planet, PlanetPhysical};

    fn planet(expr: Expr) -> Planet {
        Planet::new(
            PlanetPhysical { radius_m: 1000.0 },
            7,
            vec![Field::scalar("h", expr)],
        )
        .unwrap()
    }

    fn dirs() -> Vec<Vec3> {
        vec![
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::new(0.3, 0.5, -0.8).normalize(),
            Vec3::new(-0.6, 0.2, 0.7).normalize(),
            Vec3::new(0.1, -0.9, 0.4).normalize(),
        ]
    }

    #[test]
    fn constant_field_matches_exactly_and_collapses_to_uniform() {
        let p = planet(Expr::Const(5.0));
        let mut cache = FieldCache::new("h", 1000.0, 16, 8, 1e-4);
        for dir in dirs() {
            let v = cache.sample(&p, dir, 2.0);
            assert!((v - 5.0).abs() < 1e-5, "constant should be exact, got {v}");
        }
        // Each touched face collapses to a single Uniform node at depth 0 — no
        // descent, no dense tiles. At most one node per distinct face touched.
        assert!(
            cache.materialized_nodes() <= FACE_COUNT,
            "uniform field must not subdivide (got {} nodes)",
            cache.materialized_nodes()
        );
    }

    #[test]
    fn cached_sampling_matches_direct_within_tolerance() {
        // Smooth-ish field (long wavelength, few octaves).
        let expr = Expr::Ridged {
            wavelength_m: 4000.0,
            octaves: 2.0,
            seed: 11,
        };
        let p = planet(expr);
        let mut cache = FieldCache::new("h", 1000.0, 32, 10, 1e-3);
        for dir in dirs() {
            let cached = cache.sample(&p, dir, 1.0);
            let direct = p.sample_field("h", dir, 1.0).unwrap();
            assert!(
                (cached - direct).abs() < 0.1,
                "cached {cached} vs direct {direct} at {dir:?}"
            );
        }
    }

    #[test]
    fn repeat_sampling_reuses_materialized_nodes() {
        let p = planet(Expr::Ridged {
            wavelength_m: 1500.0,
            octaves: 4.0,
            seed: 3,
        });
        let mut cache = FieldCache::new("h", 1000.0, 16, 8, 1e-3);
        let dir = Vec3::new(0.2, 0.3, 0.9).normalize();
        let _ = cache.sample(&p, dir, 1.0);
        let after_first = cache.materialized_nodes();
        let _ = cache.sample(&p, dir, 1.0);
        assert_eq!(
            cache.materialized_nodes(),
            after_first,
            "re-sampling the same point must not materialise new nodes"
        );
    }

    #[test]
    fn materialization_is_lazy_per_region() {
        let p = planet(Expr::Ridged {
            wavelength_m: 1500.0,
            octaves: 4.0,
            seed: 3,
        });
        let mut cache = FieldCache::new("h", 1000.0, 16, 6, 1e-3);
        let _ = cache.sample(&p, Vec3::X, 1.0);
        let after_one_region = cache.materialized_nodes();
        // Sampling the opposite hemisphere touches previously-untouched faces.
        let _ = cache.sample(&p, Vec3::NEG_X, 1.0);
        assert!(
            cache.materialized_nodes() > after_one_region,
            "a far region should materialise additional nodes"
        );
    }

    #[test]
    fn sampling_is_deterministic_across_caches() {
        let p = planet(Expr::Ridged {
            wavelength_m: 2000.0,
            octaves: 3.0,
            seed: 99,
        });
        let dir = Vec3::new(0.5, -0.4, 0.76).normalize();
        let mut a = FieldCache::new("h", 1000.0, 16, 8, 1e-3);
        let mut b = FieldCache::new("h", 1000.0, 16, 8, 1e-3);
        assert_eq!(a.sample(&p, dir, 1.0), b.sample(&p, dir, 1.0));
    }
}
