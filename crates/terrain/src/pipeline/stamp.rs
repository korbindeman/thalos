//! [`Stamp`] — the basic unit of authored or generator contribution (spec §4).
//!
//! A stamp writes a value into a field within a footprint on the sphere. It has
//! three parts:
//!
//! - **Geometry** — a primitive defined by directions on the unit sphere
//!   ([`StampGeometry`]): point, great-circle capsule, polyline, point-set.
//!   (Bezier is a later addition.)
//! - **Scalars** — radius, value, and (implicitly) the falloff
//!   ([`Scalar`]): each is a constant or sampled `FromField` at evaluation time.
//!   `FromField` is the mechanism by which generator output responds to author
//!   edits without re-running the generator.
//! - **Composition operator** — how the stamp's contribution merges with the
//!   field's accumulated value ([`CompositionOp`]).
//!
//! Geometry distances are computed in metres (arc length on a body of the
//! given radius) so a stamp's `radius` reads naturally as a metric footprint.
//! Evaluation — resolving `FromField` scalars and folding stamps onto a field —
//! lives on [`super::planet::Planet`], which owns the field bag; this module
//! provides the geometry/falloff math and the [`compose`] operator.

use glam::Vec3;

use crate::pipeline::expr::{smooth_max, smooth_min};
use crate::pipeline::field::CompositionOp;

/// A stamp scalar: a constant, or the value of another field sampled at the
/// evaluation point.
#[derive(Debug, Clone)]
pub enum Scalar {
    Const(f32),
    /// Sample another field at the same direction. Creates a DAG edge.
    FromField(String),
}

impl Scalar {
    pub fn field_ref(&self) -> Option<&str> {
        match self {
            Scalar::Const(_) => None,
            Scalar::FromField(name) => Some(name),
        }
    }
}

/// Falloff from full weight at the footprint core to zero at the radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Falloff {
    /// 1 inside, hard 0 at `t >= 1`.
    Hard,
    /// Linear ramp `1 - t`.
    Linear,
    /// C1-continuous smoothstep `1 → 0` over `t ∈ [0, 1]`.
    Smoothstep,
    /// Gaussian `exp(-(t·s)²)`; never exactly zero but clamped at `t >= 1`.
    Gaussian { sharpness: f32 },
}

impl Falloff {
    /// Weight in `[0, 1]` at normalised radius `t = distance / radius`.
    pub fn weight(self, t: f32) -> f32 {
        if t <= 0.0 {
            return 1.0;
        }
        if t >= 1.0 {
            return 0.0;
        }
        match self {
            Falloff::Hard => 1.0,
            Falloff::Linear => 1.0 - t,
            Falloff::Smoothstep => {
                let s = t * t * (3.0 - 2.0 * t);
                1.0 - s
            }
            Falloff::Gaussian { sharpness } => {
                let s = (t * sharpness).max(0.0);
                (-s * s).exp()
            }
        }
    }
}

/// Geometry of a stamp: a set of directions on the unit sphere.
#[derive(Debug, Clone)]
pub enum StampGeometry {
    /// A single point.
    Point(Vec3),
    /// A great-circle segment between two endpoints.
    Capsule { a: Vec3, b: Vec3 },
    /// A chain of great-circle segments (distance is the min over segments).
    Polyline(Vec<Vec3>),
    /// A sprinkling of points without interpolation (distance is the min).
    PointSet(Vec<Vec3>),
}

impl StampGeometry {
    /// Arc-length distance in metres from `dir` to the nearest point of this
    /// geometry, on a body of radius `body_radius_m`.
    pub fn distance_m(&self, dir: Vec3, body_radius_m: f32) -> f32 {
        let dir = dir.normalize_or_zero();
        let angular = match self {
            StampGeometry::Point(p) => angular_distance(dir, *p),
            StampGeometry::Capsule { a, b } => arc_distance(dir, *a, *b),
            StampGeometry::Polyline(points) => {
                if points.is_empty() {
                    return f32::INFINITY;
                }
                if points.len() == 1 {
                    angular_distance(dir, points[0])
                } else {
                    points
                        .windows(2)
                        .map(|seg| arc_distance(dir, seg[0], seg[1]))
                        .fold(f32::INFINITY, f32::min)
                }
            }
            StampGeometry::PointSet(points) => points
                .iter()
                .map(|p| angular_distance(dir, *p))
                .fold(f32::INFINITY, f32::min),
        };
        angular * body_radius_m
    }
}

/// A stamp: geometry + radius/value scalars + falloff + composition.
#[derive(Debug, Clone)]
pub struct Stamp {
    pub geometry: StampGeometry,
    /// Footprint radius in metres.
    pub radius: Scalar,
    /// Value written at the core.
    pub value: Scalar,
    pub falloff: Falloff,
    pub composition: CompositionOp,
}

impl Stamp {
    /// Field names this stamp references through `FromField` scalars (radius or
    /// value). Contributes DAG edges for the field the stamp targets.
    pub fn field_refs(&self, out: &mut Vec<String>) {
        if let Some(name) = self.radius.field_ref() {
            out.push(name.to_string());
        }
        if let Some(name) = self.value.field_ref() {
            out.push(name.to_string());
        }
    }
}

/// Compose a stamp's `value` (at `weight`) onto the accumulated field value via
/// `op`. `weight` is the falloff in `[0, 1]`; it scales how strongly the
/// operator pulls `acc` toward its target, so the contribution fades to
/// identity at the footprint edge.
pub(crate) fn compose(acc: f32, value: f32, weight: f32, op: CompositionOp) -> f32 {
    let w = weight.clamp(0.0, 1.0);
    if w <= 0.0 {
        return acc;
    }
    match op {
        CompositionOp::Replace => acc + (value - acc) * w,
        CompositionOp::Add => acc + value * w,
        CompositionOp::Blend { t } => acc + (value - acc) * (w * t).clamp(0.0, 1.0),
        CompositionOp::SmoothMin { k } => {
            let target = smooth_min(acc, value, k);
            acc + (target - acc) * w
        }
        CompositionOp::SmoothMax { k } => {
            let target = smooth_max(acc, value, k);
            acc + (target - acc) * w
        }
    }
}

fn angular_distance(a: Vec3, b: Vec3) -> f32 {
    a.normalize_or_zero()
        .dot(b.normalize_or_zero())
        .clamp(-1.0, 1.0)
        .acos()
}

/// Angular distance from `dir` to the great-circle segment `a`–`b`.
fn arc_distance(dir: Vec3, a: Vec3, b: Vec3) -> f32 {
    let a = a.normalize_or_zero();
    let b = b.normalize_or_zero();
    let dir = dir.normalize_or_zero();

    let n = a.cross(b);
    let n_len = n.length();
    if n_len < 1e-6 {
        // Degenerate segment (coincident/antipodal endpoints): treat as point.
        return angular_distance(dir, a);
    }
    let n = n / n_len;

    // Project dir onto the segment's great-circle plane.
    let proj = dir - n * dir.dot(n);
    let proj_len = proj.length();
    if proj_len < 1e-6 {
        // dir is at the pole of the great circle — equidistant from all of it.
        return std::f32::consts::FRAC_PI_2;
    }
    let proj = proj / proj_len;

    // If the projection lies on the arc, the perpendicular distance is closest;
    // otherwise the nearer endpoint is.
    let ab = angular_distance(a, b);
    let a_to_proj = angular_distance(a, proj);
    let b_to_proj = angular_distance(b, proj);
    if a_to_proj + b_to_proj <= ab + 1e-4 {
        dir.dot(n).clamp(-1.0, 1.0).asin().abs()
    } else {
        angular_distance(dir, a).min(angular_distance(dir, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falloff_endpoints() {
        for fo in [
            Falloff::Hard,
            Falloff::Linear,
            Falloff::Smoothstep,
            Falloff::Gaussian { sharpness: 2.0 },
        ] {
            assert_eq!(fo.weight(0.0), 1.0, "{fo:?} core");
            assert_eq!(fo.weight(1.0), 0.0, "{fo:?} edge");
            assert_eq!(fo.weight(1.5), 0.0, "{fo:?} beyond");
        }
    }

    #[test]
    fn point_distance_is_zero_at_point() {
        let g = StampGeometry::Point(Vec3::X);
        assert!(g.distance_m(Vec3::X, 1000.0) < 1e-3);
    }

    #[test]
    fn point_distance_scales_with_radius() {
        let g = StampGeometry::Point(Vec3::X);
        // 90° away on a unit-radius body is a quarter great circle = π/2 metres.
        let d = g.distance_m(Vec3::Y, 1.0);
        assert!((d - std::f32::consts::FRAC_PI_2).abs() < 1e-4, "got {d}");
    }

    #[test]
    fn capsule_midpoint_is_on_the_arc() {
        let a = Vec3::X;
        let b = Vec3::new(0.0, 1.0, 0.0);
        let mid = (a + b).normalize();
        let g = StampGeometry::Capsule { a, b };
        // A point exactly on the arc has ~zero distance.
        assert!(g.distance_m(mid, 1000.0) < 1.0, "midpoint should lie on arc");
    }

    #[test]
    fn capsule_uses_endpoint_when_projection_is_off_segment() {
        let a = Vec3::X;
        let b = Vec3::new(0.9, 0.435, 0.0).normalize(); // short arc near +X
        let g = StampGeometry::Capsule { a, b };
        // A direction "behind" a, off the segment, should fall back to endpoint a.
        let probe = Vec3::new(0.9, -0.435, 0.0).normalize();
        let d_arc = g.distance_m(probe, 1.0);
        let d_a = angular_distance(probe, a);
        assert!((d_arc - d_a).abs() < 1e-3, "expected endpoint distance");
    }

    #[test]
    fn compose_add_scales_by_weight() {
        assert!((compose(10.0, 4.0, 0.5, CompositionOp::Add) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn compose_replace_lerps() {
        assert!((compose(0.0, 8.0, 0.25, CompositionOp::Replace) - 2.0).abs() < 1e-6);
    }
}
