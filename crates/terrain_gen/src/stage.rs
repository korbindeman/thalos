use crate::body_builder::BodyBuilder;
use crate::seeding::sub_seed;

/// A generation stage that contributes to a body's surface.
///
/// Stages are run in order by the `Pipeline`.  Each stage may:
/// - Write to the cubemap accumulators (low-frequency)
/// - Append features to the SSBO lists (mid-frequency)
/// - Set global structures (plates, drainage)
/// - Configure detail noise parameters (high-frequency)
/// - Read and modify outputs of prior stages
pub trait Stage {
    /// Unique name used for dependency declarations and seed derivation.
    fn name(&self) -> &str;

    /// Names of stages that must run before this one.  The pipeline runner
    /// validates this at construction time.
    fn dependencies(&self) -> &[&str];

    /// Execute the stage, mutating the builder.
    fn apply(&self, builder: &mut BodyBuilder);
}

/// An ordered sequence of stages with validated dependencies.
///
/// The pipeline validates at construction time that every stage's declared
/// dependencies appear earlier in the list.  Running a pipeline with zero
/// stages is valid and produces a default `BodyData`.
pub struct Pipeline {
    stages: Vec<Box<dyn Stage>>,
}

impl Pipeline {
    /// Create a pipeline, validating dependency order.
    ///
    /// # Panics
    /// Panics if any stage declares a dependency that doesn't appear
    /// earlier in the list.
    pub fn new(stages: Vec<Box<dyn Stage>>) -> Self {
        // Validate: for each stage, all deps must be names of earlier stages.
        let mut seen_names: Vec<&str> = Vec::with_capacity(stages.len());
        for stage in &stages {
            for dep in stage.dependencies() {
                assert!(
                    seen_names.contains(dep),
                    "stage '{}' depends on '{}', which must appear earlier in the pipeline",
                    stage.name(),
                    dep
                );
            }
            seen_names.push(stage.name());
        }
        Self { stages }
    }

    /// Run all stages in order on the builder.
    ///
    /// Sets a per-stage seed on the builder before each stage via
    /// `hash(builder.seed, stage.name())`.
    pub fn run(&self, builder: &mut BodyBuilder) {
        for stage in &self.stages {
            builder.stage_seed = sub_seed(builder.seed, stage.name());
            stage.apply(builder);
        }
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyStage {
        name: &'static str,
        deps: Vec<&'static str>,
    }
    impl Stage for DummyStage {
        fn name(&self) -> &str {
            self.name
        }
        fn dependencies(&self) -> &[&str] {
            &self.deps
        }
        fn apply(&self, _builder: &mut BodyBuilder) {}
    }

    #[test]
    fn empty_pipeline_is_valid() {
        let p = Pipeline::new(vec![]);
        assert_eq!(p.stage_count(), 0);
    }

    #[test]
    fn valid_dependency_order() {
        let stages: Vec<Box<dyn Stage>> = vec![
            Box::new(DummyStage {
                name: "a",
                deps: vec![],
            }),
            Box::new(DummyStage {
                name: "b",
                deps: vec!["a"],
            }),
            Box::new(DummyStage {
                name: "c",
                deps: vec!["a", "b"],
            }),
        ];
        let p = Pipeline::new(stages);
        assert_eq!(p.stage_count(), 3);
    }

    #[test]
    #[should_panic(expected = "depends on 'a'")]
    fn invalid_dependency_order_panics() {
        let stages: Vec<Box<dyn Stage>> = vec![
            Box::new(DummyStage {
                name: "b",
                deps: vec!["a"],
            }),
            Box::new(DummyStage {
                name: "a",
                deps: vec![],
            }),
        ];
        Pipeline::new(stages);
    }

    #[test]
    fn stage_seeds_are_distinct() {
        let seed_a = sub_seed(42, "a");
        let seed_b = sub_seed(42, "b");
        assert_ne!(seed_a, seed_b);
    }
}
