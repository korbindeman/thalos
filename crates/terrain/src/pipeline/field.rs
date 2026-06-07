//! [`Field`] and its metadata — the unit of the intent layer.
//!
//! The intent layer is a bag of named fields (spec §3). Each field carries a
//! value [`Expr`]ession plus metadata describing how its values are
//! interpreted, composed, and consumed. Field *names* are arbitrary strings;
//! the system does not hard-code semantics for specific names — feature and
//! detail behaviour reference fields by name.

use crate::pipeline::expr::Expr;
use crate::pipeline::stamp::Stamp;

/// How a field's values are interpreted (spec §3).
///
/// Stored on every field; consulted by materialisation and the detail stage in
/// later increments (e.g. SDF fields are kept eikonal-correct on
/// materialisation, categorical fields resolve via multi-channel argmin). The
/// increment-1 sampler treats every field as a plain scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldSemantic {
    /// Signed distance to a region boundary (negative inside).
    Sdf,
    /// Non-negative scalar magnitude.
    Density,
    /// Class label (optionally via multi-channel argmin).
    Categorical,
    /// Arbitrary numeric value.
    Scalar,
}

/// How a new contribution (author overlay edit, or a generator's stamp)
/// composes with the existing field value (spec §4).
///
/// Stored now; applied once stamps and the author overlay land. The
/// increment-1 sampler evaluates the expression tree directly and does not yet
/// layer contributions, so this is inert until then.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositionOp {
    /// Overwrite.
    Replace,
    /// Sum.
    Add,
    /// Linear blend toward the contribution by `t`.
    Blend { t: f32 },
    /// Polynomial smooth-min with radius `k`.
    SmoothMin { k: f32 },
    /// Polynomial smooth-max with radius `k`.
    SmoothMax { k: f32 },
}

/// Whether a field feeds other fields/generators or the detail stage (spec §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldRole {
    /// Consumed by other fields or generators.
    Intermediate,
    /// Consumed by the detail stage — part of the output contract (spec §6).
    Output,
}

/// One named field in the intent layer.
#[derive(Debug, Clone)]
pub struct Field {
    /// Arbitrary, unique within a planet's field bag.
    pub name: String,
    /// How values are interpreted.
    pub semantic: FieldSemantic,
    /// Value used where the field hasn't been authored or generated. (Inert in
    /// increment 1, where the expression always produces a value.)
    pub default: f32,
    /// How contributions compose. (Inert until stamps/overlay land.)
    pub composition: CompositionOp,
    /// Intermediate vs. output.
    pub role: FieldRole,
    /// The procedural value expression. Field references inside it form
    /// edges of the evaluation DAG.
    pub expr: Expr,
    /// Stamps folded onto the base expression value, in order, each via its own
    /// composition operator. These are the *procedural* contribution
    /// (expression tree + generator-emitted stamps). `FromField` scalars inside
    /// them form DAG edges — see [`Field::dependencies`].
    pub stamps: Vec<Stamp>,
    /// Author overlay: a separately-stored, separately-materialised log of
    /// explicit edits (spec §4–5). Composed *onto* the procedural value via
    /// this field's [`CompositionOp`] at sample time, weighted by overlay
    /// coverage. Stored apart so reshuffling procedural never disturbs edits
    /// and painting never disturbs procedural.
    pub overlay: AuthorOverlay,
}

/// A field's author overlay: a replayable log of paint operations (spec §4).
///
/// Each op is a [`Stamp`] (geometry + radius/value scalars + falloff +
/// within-overlay composition). Sampling the overlay yields a painted value and
/// a coverage weight; the planet composes that onto the procedural value via
/// the field's operator, so unpainted regions (coverage 0) read straight
/// through to procedural. The op list *is* the replayable edit log; full
/// undo/redo and version-control diffs land in a later phase.
#[derive(Debug, Clone, Default)]
pub struct AuthorOverlay {
    pub ops: Vec<Stamp>,
}

impl AuthorOverlay {
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Append a paint op to the log.
    pub fn paint(&mut self, op: Stamp) {
        self.ops.push(op);
    }

    /// Pop the most recent op (basic undo; full undo/redo is a later phase).
    pub fn undo(&mut self) -> Option<Stamp> {
        self.ops.pop()
    }

    pub(crate) fn field_refs(&self, out: &mut Vec<String>) {
        for op in &self.ops {
            op.field_refs(out);
        }
    }
}

impl Field {
    /// Convenience constructor for an intermediate scalar field.
    pub fn scalar(name: impl Into<String>, expr: Expr) -> Self {
        Self {
            name: name.into(),
            semantic: FieldSemantic::Scalar,
            default: 0.0,
            composition: CompositionOp::Replace,
            role: FieldRole::Intermediate,
            expr,
            stamps: Vec::new(),
            overlay: AuthorOverlay::default(),
        }
    }

    /// Attach procedural stamps to this field.
    pub fn with_stamps(mut self, stamps: Vec<Stamp>) -> Self {
        self.stamps = stamps;
        self
    }

    /// Attach an author overlay to this field.
    pub fn with_overlay(mut self, overlay: AuthorOverlay) -> Self {
        self.overlay = overlay;
        self
    }

    /// Every field this field depends on: references in the base expression,
    /// in its procedural stamps, and in its author-overlay ops. Deduplicated.
    /// This is what the DAG builder uses to derive edges.
    pub fn dependencies(&self) -> Vec<String> {
        let mut refs = Vec::new();
        self.expr.collect_field_refs(&mut refs);
        for stamp in &self.stamps {
            stamp.field_refs(&mut refs);
        }
        self.overlay.field_refs(&mut refs);
        refs.sort();
        refs.dedup();
        refs
    }

    /// Mark this field as part of the detail-stage output contract.
    pub fn as_output(mut self) -> Self {
        self.role = FieldRole::Output;
        self
    }

    /// Set the field's interpretation semantic.
    pub fn with_semantic(mut self, semantic: FieldSemantic) -> Self {
        self.semantic = semantic;
        self
    }

    /// Set the field's default value.
    pub fn with_default(mut self, default: f32) -> Self {
        self.default = default;
        self
    }

    /// Set the field's composition operator.
    pub fn with_composition(mut self, composition: CompositionOp) -> Self {
        self.composition = composition;
        self
    }
}
