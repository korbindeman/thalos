//! The field-DAG planet-generation pipeline (the spec's target architecture).
//!
//! This module is the new pipeline described by
//! [docs/archive/planet-generation-pipeline-spec.md] and built per
//! [docs/archive/planet-generation-pipeline-migration.md] (migration phase P1+). It is
//! deliberately separate from the legacy archetype compiler
//! ([`crate::feature_compiler`] + [`crate::stages`]): the two coexist behind
//! the Query API seam ([`crate::query`]) until bodies are cut over one at a
//! time (migration P2). Nothing here is wired into rendering yet — it is
//! foundation-only, validated by sampling + determinism unit tests.
//!
//! ## What lives here (and what's coming)
//!
//! Phase A of the spec is the core data model. It lands in increments:
//!
//! - **Increment 1 (here):** the **intent layer** — a bag of named [`Field`]s,
//!   each with a value [`Expr`]ession tree; an automatically derived
//!   evaluation [`FieldDag`] (topological order from expression references,
//!   with cycle + dangling-reference rejection); and direct sampling of any
//!   field at a direction on the sphere ([`Planet::sample_field`]).
//! - **Later increments:** stamps (`pipeline::stamp`), generators + the
//!   feature catalog with promotion (`pipeline::feature`), the author overlay +
//!   two-path composition, sparse quadtree storage on the cube-sphere
//!   (`pipeline::storage`), and the L2/L4 cache tiers.
//!
//! ## Source of truth
//!
//! The expression tree per field is the source of truth (spec §5). It is small
//! and deterministic: a [`Planet`] with the same fields and seed samples
//! identically anywhere, which is what makes the eventual quadtree a pure
//! cache and tile borders bit-identical.

pub mod cubesphere;
pub mod dag;
pub mod expr;
pub mod feature;
pub mod field;
pub mod planet;
pub mod stamp;
pub mod storage;

pub use dag::{DagError, FieldDag};
pub use expr::Expr;
pub use feature::{
    FeatureCatalog, FeatureComposition, FeatureInstance, FeatureInstanceId, FeatureKind,
    FeatureOrigin, FeatureParam, FeatureParams, FeatureType, GeneratorId, InfluenceRadius,
    ScatterGenerator,
};
pub use field::{AuthorOverlay, CompositionOp, Field, FieldRole, FieldSemantic};
pub use planet::{Planet, PlanetPhysical};
pub use stamp::{Falloff, Scalar, Stamp, StampGeometry};
pub use storage::FieldCache;
