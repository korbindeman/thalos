use crate::body_builder::BodyBuilder;

/// A generation stage that contributes to a body's surface.
///
/// Stages are mutating transforms over a `BodyBuilder`. The feature
/// compiler invokes them directly via `apply()`, setting `stage_seed`
/// on the builder beforehand so each stage has a deterministic per-run
/// RNG stream that doesn't collide with the body seed.
pub trait Stage {
    /// Unique name, used in seed derivation by the caller.
    fn name(&self) -> &str;

    /// Execute the stage, mutating the builder.
    fn apply(&self, builder: &mut BodyBuilder);
}
