//! [`Expr`] — the per-field value expression tree.
//!
//! A field's value at a point on the sphere is computed by evaluating its
//! `Expr`. Expressions compose constants, spatial noise leaves, arithmetic,
//! and **references to other fields** ([`Expr::Field`]). Those references are
//! the edges of the evaluation DAG (see [`super::dag`]); the topological order
//! is derived from them automatically, never specified by hand (spec §3).
//!
//! This is the increment-1 vocabulary: enough to build non-trivial field
//! graphs and validate ordering + determinism. Stamps and generators (which
//! *write* to fields rather than computing a scalar) are separate node
//! families added in later increments.
//!
//! Evaluation itself lives on [`super::planet::Planet`], which owns the field
//! bag needed to resolve [`Expr::Field`] references and the radius needed to
//! place spatial noise.

/// A field-value expression node.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Position-independent constant.
    Const(f32),
    /// Value of another field at the same position. Creates a DAG edge.
    Field(String),
    /// Sum of children.
    Add(Vec<Expr>),
    /// Product of children.
    Mul(Vec<Expr>),
    /// Hard minimum.
    Min(Box<Expr>, Box<Expr>),
    /// Hard maximum.
    Max(Box<Expr>, Box<Expr>),
    /// Polynomial smooth-min with radius `k` (k ≤ 0 falls back to hard min).
    SmoothMin { a: Box<Expr>, b: Box<Expr>, k: f32 },
    /// Polynomial smooth-max with radius `k`.
    SmoothMax { a: Box<Expr>, b: Box<Expr>, k: f32 },
    /// Clamp a child to `[lo, hi]`.
    Clamp { x: Box<Expr>, lo: f32, hi: f32 },
    /// Scale a child by a constant factor.
    Scale { x: Box<Expr>, factor: f32 },
    /// Domain-continuous ridged hybrid multifractal in `[0, 1]`, sampled in
    /// body-local 3D at `dir * radius / wavelength_m`. Sphere-continuous: the
    /// same physical point returns the same value regardless of cube face.
    Ridged {
        wavelength_m: f32,
        octaves: f32,
        seed: u32,
    },
}

impl Expr {
    /// Constant leaf.
    pub fn constant(value: f32) -> Self {
        Expr::Const(value)
    }

    /// Reference another field by name.
    pub fn field(name: impl Into<String>) -> Self {
        Expr::Field(name.into())
    }

    /// Append the names of every field this expression references (with
    /// duplicates) to `out`. Used by the DAG builder to derive edges.
    pub fn collect_field_refs(&self, out: &mut Vec<String>) {
        match self {
            Expr::Const(_) | Expr::Ridged { .. } => {}
            Expr::Field(name) => out.push(name.clone()),
            Expr::Add(children) | Expr::Mul(children) => {
                for child in children {
                    child.collect_field_refs(out);
                }
            }
            Expr::Min(a, b)
            | Expr::Max(a, b)
            | Expr::SmoothMin { a, b, .. }
            | Expr::SmoothMax { a, b, .. } => {
                a.collect_field_refs(out);
                b.collect_field_refs(out);
            }
            Expr::Clamp { x, .. } | Expr::Scale { x, .. } => x.collect_field_refs(out),
        }
    }

    /// Names of every field this expression references, deduplicated.
    pub fn field_refs(&self) -> Vec<String> {
        let mut refs = Vec::new();
        self.collect_field_refs(&mut refs);
        refs.sort();
        refs.dedup();
        refs
    }
}

/// Polynomial smooth minimum. `k` is the blend radius; `k <= 0` is a hard min.
pub(crate) fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

/// Polynomial smooth maximum, defined as `-smooth_min(-a, -b, k)`.
pub(crate) fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    -smooth_min(-a, -b, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_refs_dedup_and_sorted() {
        let expr = Expr::Add(vec![
            Expr::field("b"),
            Expr::field("a"),
            Expr::Scale {
                x: Box::new(Expr::field("b")),
                factor: 2.0,
            },
            Expr::Const(1.0),
        ]);
        assert_eq!(expr.field_refs(), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn leaves_have_no_refs() {
        assert!(Expr::Const(3.0).field_refs().is_empty());
        assert!(
            Expr::Ridged {
                wavelength_m: 1000.0,
                octaves: 4.0,
                seed: 7,
            }
            .field_refs()
            .is_empty()
        );
    }

    #[test]
    fn smooth_min_max_bounds() {
        // Smooth min is <= hard min's operands and approaches min as k -> 0.
        assert!((smooth_min(2.0, 5.0, 0.0) - 2.0).abs() < 1e-6);
        assert!((smooth_max(2.0, 5.0, 0.0) - 5.0).abs() < 1e-6);
        // With a blend radius the smooth min dips slightly below the hard min.
        assert!(smooth_min(2.0, 2.0, 1.0) <= 2.0);
    }
}
