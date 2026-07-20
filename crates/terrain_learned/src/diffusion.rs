use serde::{Deserialize, Serialize};

/// Serializable linear DDPM schedule shared by training and baking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiffusionSchedule {
    beta: Vec<f32>,
    alpha_bar: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiffusionScheduleError {
    TooFewSteps,
    InvalidBetaRange,
}

impl std::fmt::Display for DiffusionScheduleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewSteps => {
                formatter.write_str("a diffusion schedule needs at least two steps")
            }
            Self::InvalidBetaRange => {
                formatter.write_str("betas must satisfy 0 < start <= end < 1")
            }
        }
    }
}

impl std::error::Error for DiffusionScheduleError {}

impl DiffusionSchedule {
    pub fn linear(
        steps: usize,
        beta_start: f32,
        beta_end: f32,
    ) -> Result<Self, DiffusionScheduleError> {
        if steps < 2 {
            return Err(DiffusionScheduleError::TooFewSteps);
        }
        if !(0.0 < beta_start && beta_start <= beta_end && beta_end < 1.0) {
            return Err(DiffusionScheduleError::InvalidBetaRange);
        }

        let mut beta = Vec::with_capacity(steps);
        let mut alpha_bar = Vec::with_capacity(steps);
        let mut product = 1.0f32;
        for step in 0..steps {
            let fraction = step as f32 / (steps - 1) as f32;
            let value = beta_start + (beta_end - beta_start) * fraction;
            product *= 1.0 - value;
            beta.push(value);
            alpha_bar.push(product);
        }
        Ok(Self { beta, alpha_bar })
    }

    pub fn len(&self) -> usize {
        self.beta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.beta.is_empty()
    }

    pub fn beta(&self, step: usize) -> f32 {
        self.beta[step]
    }

    pub fn alpha_bar(&self, step: usize) -> f32 {
        self.alpha_bar[step]
    }

    pub fn noise_scales(&self, step: usize) -> (f32, f32) {
        let alpha_bar = self.alpha_bar(step);
        (alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt())
    }
}
