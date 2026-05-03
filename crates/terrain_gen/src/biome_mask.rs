//! Reusable biome mask evaluation for feature-path surface fields.
//!
//! A mask plan turns direction, deterministic seed streams, and a small set of
//! caller-provided scalar signals into normalized biome weights. Archetypes can
//! keep their own meaningful signals while sharing the same scoring machinery.

use glam::Vec3;

use crate::noise::fbm3;
use crate::seeding::sub_seed;
use crate::surface_field::smoothstep;

const MAX_BIOME_SIGNALS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiomeMaskSeedStream {
    Identity,
    Placement,
    Shape,
    Detail,
    Children,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiomeMaskSeeds {
    pub identity: u64,
    pub placement: u64,
    pub shape: u64,
    pub detail: u64,
    pub children: u64,
}

impl BiomeMaskSeeds {
    pub fn get(self, stream: BiomeMaskSeedStream) -> u64 {
        match stream {
            BiomeMaskSeedStream::Identity => self.identity,
            BiomeMaskSeedStream::Placement => self.placement,
            BiomeMaskSeedStream::Shape => self.shape,
            BiomeMaskSeedStream::Detail => self.detail,
            BiomeMaskSeedStream::Children => self.children,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BiomeMaskSignal {
    name: &'static str,
    value: f32,
}

#[derive(Clone, Debug)]
pub struct BiomeMaskContext {
    dir: Vec3,
    seeds: BiomeMaskSeeds,
    signals: [BiomeMaskSignal; MAX_BIOME_SIGNALS],
    signal_count: usize,
}

impl BiomeMaskContext {
    pub fn new(dir: Vec3, seeds: BiomeMaskSeeds) -> Self {
        Self {
            dir,
            seeds,
            signals: [BiomeMaskSignal {
                name: "",
                value: 0.0,
            }; MAX_BIOME_SIGNALS],
            signal_count: 0,
        }
        .with_signal("dir_x", dir.x)
        .with_signal("dir_y", dir.y)
        .with_signal("dir_z", dir.z)
    }

    pub fn with_signal(mut self, name: &'static str, value: f32) -> Self {
        self.set_signal(name, value);
        self
    }

    pub fn set_signal(&mut self, name: &'static str, value: f32) {
        if let Some(signal) = self.signals[..self.signal_count]
            .iter_mut()
            .find(|signal| signal.name == name)
        {
            signal.value = value;
            return;
        }

        assert!(
            self.signal_count < MAX_BIOME_SIGNALS,
            "too many biome mask signals"
        );
        self.signals[self.signal_count] = BiomeMaskSignal { name, value };
        self.signal_count += 1;
    }

    fn signal(&self, name: &'static str) -> f32 {
        self.signals[..self.signal_count]
            .iter()
            .find(|signal| signal.name == name)
            .map(|signal| signal.value)
            .unwrap_or(0.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WeightedBiomeMaskExpr {
    pub weight: f32,
    pub expr: BiomeMaskExpr,
}

impl WeightedBiomeMaskExpr {
    pub fn new(weight: f32, expr: BiomeMaskExpr) -> Self {
        Self { weight, expr }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BiomeMaskExpr {
    Constant(f32),
    Signal(&'static str),
    Fbm {
        seed_stream: BiomeMaskSeedStream,
        stream: &'static str,
        frequency: f32,
        octaves: u32,
        gain: f32,
    },
    Cap {
        center: Vec3,
        inner_rad: f32,
        outer_rad: f32,
    },
    Abs(Box<BiomeMaskExpr>),
    SmoothStep {
        edge0: f32,
        edge1: f32,
        x: Box<BiomeMaskExpr>,
    },
    Clamp {
        min: f32,
        max: f32,
        x: Box<BiomeMaskExpr>,
    },
    WeightedSum(Vec<WeightedBiomeMaskExpr>),
    Product(Vec<BiomeMaskExpr>),
}

impl BiomeMaskExpr {
    pub fn constant(value: f32) -> Self {
        Self::Constant(value)
    }

    pub fn signal(name: &'static str) -> Self {
        Self::Signal(name)
    }

    pub fn fbm(
        seed_stream: BiomeMaskSeedStream,
        stream: &'static str,
        frequency: f32,
        octaves: u32,
        gain: f32,
    ) -> Self {
        Self::Fbm {
            seed_stream,
            stream,
            frequency,
            octaves,
            gain,
        }
    }

    pub fn cap(center: Vec3, inner_rad: f32, outer_rad: f32) -> Self {
        Self::Cap {
            center: center.normalize(),
            inner_rad,
            outer_rad,
        }
    }

    pub fn abs(expr: BiomeMaskExpr) -> Self {
        Self::Abs(Box::new(expr))
    }

    pub fn smoothstep(edge0: f32, edge1: f32, x: BiomeMaskExpr) -> Self {
        Self::SmoothStep {
            edge0,
            edge1,
            x: Box::new(x),
        }
    }

    pub fn clamp(min: f32, max: f32, x: BiomeMaskExpr) -> Self {
        Self::Clamp {
            min,
            max,
            x: Box::new(x),
        }
    }

    pub fn sum(terms: Vec<(f32, BiomeMaskExpr)>) -> Self {
        Self::WeightedSum(
            terms
                .into_iter()
                .map(|(weight, expr)| WeightedBiomeMaskExpr::new(weight, expr))
                .collect(),
        )
    }

    pub fn product(terms: Vec<BiomeMaskExpr>) -> Self {
        Self::Product(terms)
    }

    pub fn eval(&self, context: &BiomeMaskContext) -> f32 {
        match self {
            Self::Constant(value) => *value,
            Self::Signal(name) => context.signal(name),
            Self::Fbm {
                seed_stream,
                stream,
                frequency,
                octaves,
                gain,
            } => {
                let p = context.dir * *frequency;
                fbm3(
                    p.x,
                    p.y,
                    p.z,
                    sub_seed(context.seeds.get(*seed_stream), stream) as u32,
                    *octaves,
                    *gain,
                    2.0,
                )
            }
            Self::Cap {
                center,
                inner_rad,
                outer_rad,
            } => smoothstep(outer_rad.cos(), inner_rad.cos(), context.dir.dot(*center)),
            Self::Abs(expr) => expr.eval(context).abs(),
            Self::SmoothStep { edge0, edge1, x } => smoothstep(*edge0, *edge1, x.eval(context)),
            Self::Clamp { min, max, x } => x.eval(context).clamp(*min, *max),
            Self::WeightedSum(terms) => terms
                .iter()
                .map(|term| term.weight * term.expr.eval(context))
                .sum(),
            Self::Product(terms) => terms.iter().map(|term| term.eval(context)).product(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BiomeMaskRule {
    pub biome_index: usize,
    pub signal_name: Option<&'static str>,
    pub expr: BiomeMaskExpr,
}

impl BiomeMaskRule {
    pub fn new(biome_index: usize, signal_name: Option<&'static str>, expr: BiomeMaskExpr) -> Self {
        Self {
            biome_index,
            signal_name,
            expr,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiomeMaskWeights<const N: usize> {
    pub weights: [f32; N],
}

impl<const N: usize> BiomeMaskWeights<N> {
    pub fn dominant_index(self) -> usize {
        self.weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BiomeMaskPlan<const N: usize> {
    rules: Vec<BiomeMaskRule>,
    fallback_index: usize,
}

impl<const N: usize> BiomeMaskPlan<N> {
    pub fn new(rules: Vec<BiomeMaskRule>, fallback_index: usize) -> Self {
        assert!(N > 0, "biome mask plans require at least one biome");
        assert!(fallback_index < N, "fallback biome index out of range");
        for rule in &rules {
            assert!(rule.biome_index < N, "biome rule index out of range");
        }
        Self {
            rules,
            fallback_index,
        }
    }

    pub fn sample(&self, context: &mut BiomeMaskContext) -> BiomeMaskWeights<N> {
        let mut scores = [0.0; N];
        for rule in &self.rules {
            let score = rule.expr.eval(context).max(0.0);
            scores[rule.biome_index] += score;
            if let Some(signal_name) = rule.signal_name {
                context.set_signal(signal_name, score);
            }
        }

        let total: f32 = scores.iter().sum();
        if total <= 1e-6 {
            scores[self.fallback_index] = 1.0;
            return BiomeMaskWeights { weights: scores };
        }

        for score in &mut scores {
            *score /= total;
        }
        BiomeMaskWeights { weights: scores }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeds() -> BiomeMaskSeeds {
        BiomeMaskSeeds {
            identity: 1,
            placement: 2,
            shape: 3,
            detail: 4,
            children: 5,
        }
    }

    #[test]
    fn plan_normalizes_scores_and_exposes_intermediate_signal() {
        let plan = BiomeMaskPlan::<2>::new(
            vec![
                BiomeMaskRule::new(0, Some("first"), BiomeMaskExpr::constant(2.0)),
                BiomeMaskRule::new(
                    1,
                    None,
                    BiomeMaskExpr::sum(vec![
                        (1.0, BiomeMaskExpr::signal("first")),
                        (1.0, BiomeMaskExpr::constant(2.0)),
                    ]),
                ),
            ],
            0,
        );
        let mut context = BiomeMaskContext::new(Vec3::Z, seeds());
        let weights = plan.sample(&mut context);

        assert!((weights.weights[0] - 1.0 / 3.0).abs() < 1.0e-6);
        assert!((weights.weights[1] - 2.0 / 3.0).abs() < 1.0e-6);
        assert_eq!(weights.dominant_index(), 1);
    }

    #[test]
    fn cap_mask_peaks_at_center() {
        let expr = BiomeMaskExpr::cap(Vec3::Z, 0.2, 0.6);
        let context = BiomeMaskContext::new(Vec3::Z, seeds());
        let away = BiomeMaskContext::new(Vec3::X, seeds());

        assert!(expr.eval(&context) > 0.99);
        assert_eq!(expr.eval(&away), 0.0);
    }
}
