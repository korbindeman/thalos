//! [`Planet`] — a self-describing field bag, and the increment-1 sampler.
//!
//! A planet is a fully self-describing value (spec §3): physical parameters, a
//! seed, and a bag of named [`Field`]s. Two planets with the same definition
//! sample identically. Construction validates the field graph and computes the
//! evaluation order ([`FieldDag`]); sampling evaluates a field at a direction
//! on the unit sphere by walking that graph.
//!
//! This increment evaluates the **procedural value** of each field (its
//! expression tree). The author overlay and two-path composition (spec §4–5)
//! layer on in a later increment; the field's [`crate::pipeline::CompositionOp`]
//! and `default` are stored for that purpose and inert here.

use std::collections::HashMap;

use glam::Vec3;

use crate::pipeline::dag::{DagError, FieldDag};
use crate::pipeline::expr::{Expr, smooth_max, smooth_min};
use crate::pipeline::field::{AuthorOverlay, Field};
use crate::pipeline::stamp::{Scalar, Stamp, compose};

/// Physical parameters of a planet. Minimal for increment 1 — grows as the
/// model needs more (gravity, axial tilt, atmosphere, age, …).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetPhysical {
    /// Reference radius in metres. Spatial noise leaves are placed at
    /// `dir * radius_m`.
    pub radius_m: f32,
}

/// A self-describing planet: physical parameters, a master seed, and a bag of
/// named fields with a validated evaluation order.
#[derive(Debug, Clone)]
pub struct Planet {
    physical: PlanetPhysical,
    seed: u64,
    fields: Vec<Field>,
    dag: FieldDag,
    index: HashMap<String, usize>,
}

impl Planet {
    /// Build a planet from its field bag, validating the field graph.
    ///
    /// Returns a [`DagError`] if the fields contain a duplicate name, a
    /// reference to an unknown field, or a cycle.
    pub fn new(physical: PlanetPhysical, seed: u64, fields: Vec<Field>) -> Result<Self, DagError> {
        let dag = FieldDag::build(&fields)?;
        let index = fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i))
            .collect();
        Ok(Self {
            physical,
            seed,
            fields,
            dag,
            index,
        })
    }

    pub fn physical(&self) -> PlanetPhysical {
        self.physical
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// Field indices in dependency-first evaluation order.
    pub fn evaluation_order(&self) -> &[usize] {
        self.dag.order()
    }

    /// Sample one field's procedural value at `dir` (a direction on the unit
    /// sphere). Returns `None` if the field name is unknown.
    ///
    /// `lod_m` is metres-per-sample; spatial leaves use it later for
    /// band-limiting, but increment-1 leaves evaluate at full resolution.
    pub fn sample_field(&self, name: &str, dir: Vec3, lod_m: f32) -> Option<f32> {
        let idx = *self.index.get(name)?;
        let dir = dir.normalize_or_zero();
        let mut memo = vec![None; self.fields.len()];
        Some(self.eval_field(idx, dir, lod_m, &mut memo))
    }

    /// Sample every field at `dir`, returning values indexed parallel to
    /// [`Planet::fields`]. Evaluates in dependency order so each field is
    /// computed once.
    pub fn sample_all(&self, dir: Vec3, lod_m: f32) -> Vec<f32> {
        let dir = dir.normalize_or_zero();
        let mut memo = vec![None; self.fields.len()];
        for &idx in self.dag.order() {
            let value = self.eval_field(idx, dir, lod_m, &mut memo);
            memo[idx] = Some(value);
        }
        memo.into_iter().map(|v| v.unwrap_or(0.0)).collect()
    }

    fn eval_field(&self, idx: usize, dir: Vec3, lod_m: f32, memo: &mut [Option<f32>]) -> f32 {
        if let Some(value) = memo[idx] {
            return value;
        }
        let field = &self.fields[idx];

        // Procedural path: base expression + procedural stamps, in order.
        let mut procedural = self.eval_expr(&field.expr, dir, lod_m, memo);
        for stamp in &field.stamps {
            procedural = self.apply_stamp(stamp, procedural, dir, lod_m, memo);
        }

        // Overlay path (materialised separately): composed onto procedural via
        // this field's operator, weighted by overlay coverage. Unpainted points
        // (coverage 0) read straight through to procedural.
        let value = if field.overlay.is_empty() {
            procedural
        } else {
            let (overlay_value, coverage) = self.eval_overlay(&field.overlay, dir, lod_m, memo);
            compose(procedural, overlay_value, coverage, field.composition)
        };

        memo[idx] = Some(value);
        value
    }

    /// Evaluate a field's author overlay independently of its procedural value,
    /// returning `(painted_value, coverage)`. Within-overlay ops stack via their
    /// own composition operators; coverage accumulates alpha-over.
    fn eval_overlay(
        &self,
        overlay: &AuthorOverlay,
        dir: Vec3,
        lod_m: f32,
        memo: &mut [Option<f32>],
    ) -> (f32, f32) {
        let mut value = 0.0;
        let mut coverage = 0.0;
        for op in &overlay.ops {
            let radius_m = self.resolve_scalar(&op.radius, dir, lod_m, memo).max(1e-3);
            let dist_m = op.geometry.distance_m(dir, self.physical.radius_m);
            let weight = op.falloff.weight(dist_m / radius_m);
            if weight <= 0.0 {
                continue;
            }
            let painted = self.resolve_scalar(&op.value, dir, lod_m, memo);
            value = compose(value, painted, weight, op.composition);
            coverage += weight * (1.0 - coverage);
        }
        (value, coverage)
    }

    fn apply_stamp(
        &self,
        stamp: &Stamp,
        acc: f32,
        dir: Vec3,
        lod_m: f32,
        memo: &mut [Option<f32>],
    ) -> f32 {
        let radius_m = self
            .resolve_scalar(&stamp.radius, dir, lod_m, memo)
            .max(1e-3);
        let dist_m = stamp.geometry.distance_m(dir, self.physical.radius_m);
        let weight = stamp.falloff.weight(dist_m / radius_m);
        if weight <= 0.0 {
            return acc;
        }
        let value = self.resolve_scalar(&stamp.value, dir, lod_m, memo);
        compose(acc, value, weight, stamp.composition)
    }

    fn resolve_scalar(
        &self,
        scalar: &Scalar,
        dir: Vec3,
        lod_m: f32,
        memo: &mut [Option<f32>],
    ) -> f32 {
        match scalar {
            Scalar::Const(value) => *value,
            Scalar::FromField(name) => {
                let dep = self.index[name];
                self.eval_field(dep, dir, lod_m, memo)
            }
        }
    }

    fn eval_expr(&self, expr: &Expr, dir: Vec3, lod_m: f32, memo: &mut [Option<f32>]) -> f32 {
        match expr {
            Expr::Const(value) => *value,
            Expr::Field(name) => {
                // Resolvable: the DAG builder rejected unknown references, so
                // construction would have failed if this name were missing.
                let dep = self.index[name];
                self.eval_field(dep, dir, lod_m, memo)
            }
            Expr::Add(children) => children
                .iter()
                .map(|child| self.eval_expr(child, dir, lod_m, memo))
                .sum(),
            Expr::Mul(children) => children
                .iter()
                .map(|child| self.eval_expr(child, dir, lod_m, memo))
                .product(),
            Expr::Min(a, b) => self
                .eval_expr(a, dir, lod_m, memo)
                .min(self.eval_expr(b, dir, lod_m, memo)),
            Expr::Max(a, b) => self
                .eval_expr(a, dir, lod_m, memo)
                .max(self.eval_expr(b, dir, lod_m, memo)),
            Expr::SmoothMin { a, b, k } => smooth_min(
                self.eval_expr(a, dir, lod_m, memo),
                self.eval_expr(b, dir, lod_m, memo),
                *k,
            ),
            Expr::SmoothMax { a, b, k } => smooth_max(
                self.eval_expr(a, dir, lod_m, memo),
                self.eval_expr(b, dir, lod_m, memo),
                *k,
            ),
            Expr::Clamp { x, lo, hi } => self.eval_expr(x, dir, lod_m, memo).clamp(*lo, *hi),
            Expr::Scale { x, factor } => self.eval_expr(x, dir, lod_m, memo) * factor,
            Expr::Ridged {
                wavelength_m,
                octaves,
                seed,
            } => {
                let wl = wavelength_m.max(1e-3);
                let p = dir * (self.physical.radius_m / wl);
                crate::noise::hmf_ridged_3d(p, *seed, *octaves, 0.5, 2.0, 1.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::field::{CompositionOp, Field};
    use crate::pipeline::stamp::{Falloff, Scalar, Stamp, StampGeometry};

    fn planet(fields: Vec<Field>) -> Planet {
        Planet::new(PlanetPhysical { radius_m: 1000.0 }, 42, fields).expect("valid field bag")
    }

    #[test]
    fn evaluates_through_references() {
        // c = b + 1, b = a * 2, a = 3  =>  c = 7.
        let p = planet(vec![
            Field::scalar("c", Expr::Add(vec![Expr::field("b"), Expr::Const(1.0)])),
            Field::scalar(
                "b",
                Expr::Scale {
                    x: Box::new(Expr::field("a")),
                    factor: 2.0,
                },
            ),
            Field::scalar("a", Expr::Const(3.0)),
        ]);
        let v = p.sample_field("c", Vec3::X, 1.0).unwrap();
        assert!((v - 7.0).abs() < 1e-6, "expected 7, got {v}");
    }

    #[test]
    fn unknown_field_samples_to_none() {
        let p = planet(vec![Field::scalar("a", Expr::Const(1.0))]);
        assert!(p.sample_field("missing", Vec3::X, 1.0).is_none());
    }

    #[test]
    fn diamond_dependency_evaluates_once_and_correctly() {
        // top = left + right; left = base * 2; right = base + 1; base = ridged.
        // Whatever `base` is, top = 2*base + (base + 1) = 3*base + 1.
        let p = planet(vec![
            Field::scalar(
                "top",
                Expr::Add(vec![Expr::field("left"), Expr::field("right")]),
            ),
            Field::scalar(
                "left",
                Expr::Scale {
                    x: Box::new(Expr::field("base")),
                    factor: 2.0,
                },
            ),
            Field::scalar(
                "right",
                Expr::Add(vec![Expr::field("base"), Expr::Const(1.0)]),
            ),
            Field::scalar(
                "base",
                Expr::Ridged {
                    wavelength_m: 500.0,
                    octaves: 4.0,
                    seed: 7,
                },
            ),
        ]);
        let dir = Vec3::new(0.3, 0.7, -0.2);
        let base = p.sample_field("base", dir, 1.0).unwrap();
        let top = p.sample_field("top", dir, 1.0).unwrap();
        assert!((top - (3.0 * base + 1.0)).abs() < 1e-5);
    }

    #[test]
    fn sampling_is_deterministic() {
        let p = planet(vec![Field::scalar(
            "h",
            Expr::Ridged {
                wavelength_m: 800.0,
                octaves: 6.0,
                seed: 1234,
            },
        )]);
        let dir = Vec3::new(1.0, 2.0, 3.0).normalize();
        let a = p.sample_field("h", dir, 1.0).unwrap();
        let b = p.sample_field("h", dir, 1.0).unwrap();
        assert_eq!(a, b, "same dir must sample identically");
    }

    #[test]
    fn sample_all_matches_individual_samples() {
        let p = planet(vec![
            Field::scalar("a", Expr::Const(2.0)),
            Field::scalar("b", Expr::Add(vec![Expr::field("a"), Expr::Const(5.0)])),
        ]);
        let dir = Vec3::Y;
        let all = p.sample_all(dir, 1.0);
        for (i, field) in p.fields().iter().enumerate() {
            let individual = p.sample_field(&field.name, dir, 1.0).unwrap();
            assert_eq!(all[i], individual);
        }
    }

    #[test]
    fn point_stamp_raises_field_within_footprint_and_not_outside() {
        // Base 0, plus a point stamp at +X adding 100 within a footprint.
        // Footprint radius 100 m on a 1000 m body ≈ 0.1 rad ≈ 5.7°.
        let stamp = Stamp {
            geometry: StampGeometry::Point(Vec3::X),
            radius: Scalar::Const(100.0),
            value: Scalar::Const(100.0),
            falloff: Falloff::Smoothstep,
            composition: CompositionOp::Add,
        };
        let p = planet(vec![
            Field::scalar("h", Expr::Const(0.0)).with_stamps(vec![stamp]),
        ]);

        // At the core, full weight → +100.
        let core = p.sample_field("h", Vec3::X, 1.0).unwrap();
        assert!(
            (core - 100.0).abs() < 1e-3,
            "core should be ~100, got {core}"
        );

        // Far away (90°), outside the footprint → unchanged base.
        let far = p.sample_field("h", Vec3::Y, 1.0).unwrap();
        assert!(far.abs() < 1e-6, "far should be ~0, got {far}");
    }

    #[test]
    fn stamp_from_field_creates_dependency_and_resolves() {
        // `src` = 5; `dst` base 0 with a stamp whose value is FromField(src).
        // dst depends on src through the stamp, so the DAG must order src first
        // and the sampled value at the core must equal src.
        let stamp = Stamp {
            geometry: StampGeometry::Point(Vec3::X),
            radius: Scalar::Const(200.0),
            value: Scalar::FromField("src".into()),
            falloff: Falloff::Hard,
            composition: CompositionOp::Replace,
        };
        let p = planet(vec![
            Field::scalar("dst", Expr::Const(0.0)).with_stamps(vec![stamp]),
            Field::scalar("src", Expr::Const(5.0)),
        ]);

        // Dependency ordering: src before dst.
        let order: Vec<&str> = p
            .evaluation_order()
            .iter()
            .map(|&i| p.fields()[i].name.as_str())
            .collect();
        let src_pos = order.iter().position(|n| *n == "src").unwrap();
        let dst_pos = order.iter().position(|n| *n == "dst").unwrap();
        assert!(src_pos < dst_pos, "src must be ordered before dst");

        // At the core, Replace with full weight writes src's value.
        let v = p.sample_field("dst", Vec3::X, 1.0).unwrap();
        assert!((v - 5.0).abs() < 1e-6, "expected 5 from FromField, got {v}");
    }

    fn paint_at_x(value: f32) -> Stamp {
        Stamp {
            geometry: StampGeometry::Point(Vec3::X),
            radius: Scalar::Const(150.0),
            value: Scalar::Const(value),
            falloff: Falloff::Hard,
            composition: CompositionOp::Replace,
        }
    }

    #[test]
    fn overlay_replaces_in_painted_region_only() {
        use crate::pipeline::field::AuthorOverlay;
        // Field is Replace; procedural = 10 everywhere; overlay paints 99 at +X.
        let field = Field::scalar("h", Expr::Const(10.0)).with_overlay(AuthorOverlay {
            ops: vec![paint_at_x(99.0)],
        });
        let p = planet(vec![field]);

        let painted = p.sample_field("h", Vec3::X, 1.0).unwrap();
        assert!(
            (painted - 99.0).abs() < 1e-3,
            "painted core should read overlay, got {painted}"
        );

        let unpainted = p.sample_field("h", Vec3::Y, 1.0).unwrap();
        assert!(
            (unpainted - 10.0).abs() < 1e-6,
            "unpainted should read procedural, got {unpainted}"
        );
    }

    #[test]
    fn overlay_is_independent_of_procedural_where_it_fully_covers() {
        use crate::pipeline::field::AuthorOverlay;
        // Two planets differing only in procedural value; a Replace overlay with
        // full coverage at +X must yield the same painted value in both —
        // demonstrating the overlay is materialised independently of procedural.
        let with_10 = planet(vec![Field::scalar("h", Expr::Const(10.0)).with_overlay(
            AuthorOverlay {
                ops: vec![paint_at_x(99.0)],
            },
        )]);
        let with_20 = planet(vec![Field::scalar("h", Expr::Const(20.0)).with_overlay(
            AuthorOverlay {
                ops: vec![paint_at_x(99.0)],
            },
        )]);
        let a = with_10.sample_field("h", Vec3::X, 1.0).unwrap();
        let b = with_20.sample_field("h", Vec3::X, 1.0).unwrap();
        assert_eq!(
            a, b,
            "overlay value must not depend on procedural where coverage is full"
        );
        assert!((a - 99.0).abs() < 1e-3);
    }

    #[test]
    fn overlay_from_field_creates_dependency() {
        use crate::pipeline::field::AuthorOverlay;
        // Overlay paints FromField("src") — `h` must depend on `src`.
        let paint = Stamp {
            geometry: StampGeometry::Point(Vec3::X),
            radius: Scalar::Const(150.0),
            value: Scalar::FromField("src".into()),
            falloff: Falloff::Hard,
            composition: CompositionOp::Replace,
        };
        let p = planet(vec![
            Field::scalar("h", Expr::Const(0.0)).with_overlay(AuthorOverlay { ops: vec![paint] }),
            Field::scalar("src", Expr::Const(42.0)),
        ]);
        let order: Vec<&str> = p
            .evaluation_order()
            .iter()
            .map(|&i| p.fields()[i].name.as_str())
            .collect();
        let src_pos = order.iter().position(|n| *n == "src").unwrap();
        let h_pos = order.iter().position(|n| *n == "h").unwrap();
        assert!(
            src_pos < h_pos,
            "overlay FromField must create a dependency edge"
        );
        let v = p.sample_field("h", Vec3::X, 1.0).unwrap();
        assert!(
            (v - 42.0).abs() < 1e-6,
            "overlay should paint src's value, got {v}"
        );
    }

    #[test]
    fn stamp_from_field_cycle_is_rejected() {
        // dst's stamp reads src; src's stamp reads dst → cycle through stamps.
        let dst = Field::scalar("dst", Expr::Const(0.0)).with_stamps(vec![Stamp {
            geometry: StampGeometry::Point(Vec3::X),
            radius: Scalar::Const(1.0),
            value: Scalar::FromField("src".into()),
            falloff: Falloff::Hard,
            composition: CompositionOp::Add,
        }]);
        let src = Field::scalar("src", Expr::Const(0.0)).with_stamps(vec![Stamp {
            geometry: StampGeometry::Point(Vec3::X),
            radius: Scalar::Const(1.0),
            value: Scalar::FromField("dst".into()),
            falloff: Falloff::Hard,
            composition: CompositionOp::Add,
        }]);
        assert!(Planet::new(PlanetPhysical { radius_m: 1000.0 }, 0, vec![dst, src]).is_err());
    }
}
