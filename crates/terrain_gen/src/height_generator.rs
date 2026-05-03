//! Configurable height-generation functions for continuous surface fields.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::noise::value_noise_3d_derivative;
use crate::seeding::sub_seed;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum HeightGenerator {
    #[default]
    None,
    /// Inigo Quilez-style value-noise fBM with analytic derivative damping.
    IqDerivativeFbm(IqDerivativeFbmHeight),
}

impl HeightGenerator {
    pub fn sample_height_m(self, dir: Vec3, seed: u64, stream: &str) -> f32 {
        match self {
            Self::None => 0.0,
            Self::IqDerivativeFbm(config) => config.sample_height_m(dir, seed, stream),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HeightGeneratorStack {
    #[serde(default)]
    pub generators: Vec<HeightGenerator>,
}

impl HeightGeneratorStack {
    pub fn single(generator: HeightGenerator) -> Self {
        Self {
            generators: vec![generator],
        }
    }

    pub fn sample_height_m(&self, dir: Vec3, seed: u64, stream: &str) -> f32 {
        if let [generator] = self.generators.as_slice() {
            return generator.sample_height_m(dir, seed, stream);
        }

        self.generators
            .iter()
            .enumerate()
            .map(|(i, generator)| {
                let stream = format!("{stream}:{i}");
                generator.sample_height_m(dir, seed, &stream)
            })
            .sum()
    }
}

impl From<HeightGenerator> for HeightGeneratorStack {
    fn from(value: HeightGenerator) -> Self {
        Self::single(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IqDerivativeFbmHeight {
    #[serde(default)]
    pub amplitude_m: f32,
    #[serde(default = "default_frequency")]
    pub frequency: f32,
    #[serde(default = "default_octaves")]
    pub octaves: u32,
    #[serde(default = "default_gain")]
    pub gain: f32,
    #[serde(default = "default_lacunarity")]
    pub lacunarity: f32,
    #[serde(default = "default_derivative_damping")]
    pub derivative_damping: f32,
}

impl IqDerivativeFbmHeight {
    pub const fn new(amplitude_m: f32, frequency: f32, octaves: u32) -> Self {
        Self {
            amplitude_m,
            frequency,
            octaves,
            gain: 0.49,
            lacunarity: 1.98,
            derivative_damping: 1.0,
        }
    }

    pub fn sample_height_m(self, dir: Vec3, seed: u64, stream: &str) -> f32 {
        if self.amplitude_m == 0.0 || self.octaves == 0 {
            return 0.0;
        }
        let seed = sub_seed(seed, stream) as u32;
        iq_derivative_fbm3(
            dir * self.frequency,
            seed,
            self.octaves,
            self.gain,
            self.lacunarity,
            self.derivative_damping,
        ) * self.amplitude_m
    }
}

impl Default for IqDerivativeFbmHeight {
    fn default() -> Self {
        Self::new(0.0, default_frequency(), default_octaves())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColdDesertBiomeHeightGenerators {
    #[serde(default)]
    pub rust_dust_plain: HeightGeneratorStack,
    #[serde(default)]
    pub dune_basin: HeightGeneratorStack,
    #[serde(default)]
    pub pale_evaporite_basin: HeightGeneratorStack,
    #[serde(default)]
    pub dark_volcanic_province: HeightGeneratorStack,
    #[serde(default)]
    pub rugged_badlands: HeightGeneratorStack,
}

impl Default for ColdDesertBiomeHeightGenerators {
    fn default() -> Self {
        Self {
            rust_dust_plain: HeightGenerator::IqDerivativeFbm(IqDerivativeFbmHeight {
                amplitude_m: 180.0,
                frequency: 2.2,
                octaves: 8,
                gain: 0.49,
                lacunarity: 1.98,
                derivative_damping: 1.0,
            })
            .into(),
            dune_basin: HeightGenerator::IqDerivativeFbm(IqDerivativeFbmHeight {
                amplitude_m: 45.0,
                frequency: 3.0,
                octaves: 6,
                gain: 0.48,
                lacunarity: 1.96,
                derivative_damping: 1.4,
            })
            .into(),
            pale_evaporite_basin: HeightGenerator::IqDerivativeFbm(IqDerivativeFbmHeight {
                amplitude_m: 80.0,
                frequency: 1.7,
                octaves: 7,
                gain: 0.48,
                lacunarity: 1.98,
                derivative_damping: 1.25,
            })
            .into(),
            dark_volcanic_province: HeightGenerator::IqDerivativeFbm(IqDerivativeFbmHeight {
                amplitude_m: 260.0,
                frequency: 3.4,
                octaves: 8,
                gain: 0.50,
                lacunarity: 2.02,
                derivative_damping: 0.75,
            })
            .into(),
            rugged_badlands: HeightGenerator::IqDerivativeFbm(IqDerivativeFbmHeight {
                amplitude_m: 520.0,
                frequency: 4.8,
                octaves: 10,
                gain: 0.50,
                lacunarity: 2.03,
                derivative_damping: 0.55,
            })
            .into(),
        }
    }
}

fn default_frequency() -> f32 {
    2.0
}

fn default_octaves() -> u32 {
    8
}

fn default_gain() -> f32 {
    0.49
}

fn default_lacunarity() -> f32 {
    1.98
}

fn default_derivative_damping() -> f32 {
    1.0
}

/// IQ's derivative-aware fBM terrain construction, lifted to 3D.
///
/// Each octave contributes `value / (1 + derivative_damping * dot(d, d))`,
/// where `d` is the accumulated analytic value-noise derivative. The result is
/// normalized by the geometric amplitude sum so configured amplitudes remain
/// useful in meters.
pub fn iq_derivative_fbm3(
    p: Vec3,
    seed: u32,
    octaves: u32,
    gain: f32,
    lacunarity: f32,
    derivative_damping: f32,
) -> f32 {
    let gain = gain.max(0.0);
    let lacunarity = lacunarity.max(0.0001);
    let derivative_damping = derivative_damping.max(0.0);

    let mut value = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut amp_sum = 0.0;
    let mut derivative_sum = Vec3::ZERO;

    for octave in 0..octaves {
        let octave_seed = crate::noise::pcg_u32(seed.wrapping_add(octave));
        let n = value_noise_3d_derivative(p.x * freq, p.y * freq, p.z * freq, octave_seed);
        derivative_sum += n.derivative;
        let damping = 1.0 + derivative_damping * derivative_sum.length_squared();
        value += amp * n.value / damping;
        amp_sum += amp;
        amp *= gain;
        freq *= lacunarity;
    }

    if amp_sum > 0.0 { value / amp_sum } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq_derivative_fbm_is_deterministic_and_finite() {
        let p = Vec3::new(0.21, 0.54, 0.81).normalize();

        let a = iq_derivative_fbm3(p * 2.7, 42, 8, 0.49, 1.98, 1.0);
        let b = iq_derivative_fbm3(p * 2.7, 42, 8, 0.49, 1.98, 1.0);

        assert_eq!(a, b);
        assert!(a.is_finite());
        assert!((-1.0..=1.0).contains(&a));
    }

    #[test]
    fn height_generator_uses_stream_as_seed_namespace() {
        let config = HeightGenerator::IqDerivativeFbm(IqDerivativeFbmHeight::new(100.0, 2.0, 6));
        let p = Vec3::new(-0.12, 0.88, 0.46).normalize();
        let a = config.sample_height_m(p, 123, "a");
        let b = config.sample_height_m(p, 123, "b");

        assert_ne!(a, b);
        assert!(a.abs() <= 100.0);
        assert!(b.abs() <= 100.0);
    }

    #[test]
    fn height_generator_stack_combines_independent_layers() {
        let stack = HeightGeneratorStack {
            generators: vec![
                HeightGenerator::IqDerivativeFbm(IqDerivativeFbmHeight::new(80.0, 2.0, 5)),
                HeightGenerator::IqDerivativeFbm(IqDerivativeFbmHeight::new(20.0, 9.0, 3)),
            ],
        };
        let p = Vec3::new(0.24, -0.37, 0.89).normalize();
        let a = stack.sample_height_m(p, 55, "biome");
        let b = stack.sample_height_m(p, 55, "biome");

        assert_eq!(a, b);
        assert!(a.is_finite());
        assert!(a.abs() <= 100.0);
    }
}
