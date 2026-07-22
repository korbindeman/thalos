use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffusionPrediction {
    #[default]
    Epsilon,
    Velocity,
}

impl DiffusionPrediction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Epsilon => "epsilon",
            Self::Velocity => "velocity",
        }
    }

    pub fn training_target(
        self,
        clean_scale: f32,
        noise_scale: f32,
        clean: f32,
        noise: f32,
    ) -> f32 {
        match self {
            Self::Epsilon => noise,
            Self::Velocity => clean_scale * noise - noise_scale * clean,
        }
    }

    pub fn reconstruct(self, alpha_bar: f32, noisy: f32, prediction: f32) -> (f32, f32) {
        let clean_scale = alpha_bar.sqrt();
        let noise_scale = (1.0 - alpha_bar).sqrt();
        match self {
            Self::Epsilon => ((noisy - noise_scale * prediction) / clean_scale, prediction),
            Self::Velocity => (
                clean_scale * noisy - noise_scale * prediction,
                noise_scale * noisy + clean_scale * prediction,
            ),
        }
    }
}

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

    pub fn terminal_alpha_bar(&self) -> f32 {
        self.alpha_bar[self.alpha_bar.len() - 1]
    }

    pub fn terminal_signal_to_noise(&self) -> f32 {
        let alpha_bar = self.terminal_alpha_bar();
        alpha_bar / (1.0 - alpha_bar)
    }
}

#[cfg(test)]
mod tests {
    use super::{DiffusionPrediction, DiffusionSchedule};

    #[test]
    fn hundred_step_generation_schedule_ends_near_pure_noise() {
        let schedule = DiffusionSchedule::linear(100, 0.0001, 0.2).unwrap();

        assert!(schedule.terminal_alpha_bar() < 0.0001);
        assert!(schedule.terminal_signal_to_noise() < 0.0001);
    }

    #[test]
    fn velocity_target_reconstructs_clean_and_noise_at_terminal_snr() {
        let alpha_bar = 0.000_021_399_666f32;
        let clean_scale = alpha_bar.sqrt();
        let noise_scale = (1.0 - alpha_bar).sqrt();
        let clean = -0.73;
        let noise = 1.42;
        let noisy = clean_scale * clean + noise_scale * noise;
        let velocity =
            DiffusionPrediction::Velocity.training_target(clean_scale, noise_scale, clean, noise);

        let (reconstructed_clean, reconstructed_noise) =
            DiffusionPrediction::Velocity.reconstruct(alpha_bar, noisy, velocity);

        assert!((reconstructed_clean - clean).abs() < 1e-5);
        assert!((reconstructed_noise - noise).abs() < 1e-5);
    }
}
