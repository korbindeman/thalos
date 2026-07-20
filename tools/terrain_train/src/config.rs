use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub run: Run,
    pub data: Data,
    pub model: Model,
    pub diffusion: Diffusion,
    pub train: Train,
    pub validation: Validation,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Run {
    pub name: String,
    pub seed: u64,
    pub output_dir: PathBuf,
    pub data_dir: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Data {
    pub sample_count: usize,
    pub patch_size: usize,
    pub metres_per_pixel: f32,
    pub validation_fraction: f32,
    pub crater_density: [f32; 2],
    pub mare_fraction: [f32; 2],
    pub gardening: [f32; 2],
    pub rim_sharpness: [f32; 2],
}

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub base_channels: usize,
    pub condition_dim: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Diffusion {
    pub timesteps: usize,
    pub sample_steps: usize,
    pub beta_start: f32,
    pub beta_end: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Train {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub ema_decay: f32,
    pub gradient_clip: f32,
    pub num_workers: usize,
    pub device: String,
    #[serde(default)]
    pub resume: bool,
    #[serde(default = "default_checkpoint_every_epochs")]
    pub checkpoint_every_epochs: usize,
}

fn default_checkpoint_every_epochs() -> usize {
    1
}

#[derive(Clone, Debug, Deserialize)]
pub struct Validation {
    pub canvas_width: usize,
    pub canvas_height: usize,
    pub stride: usize,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        let mut config: Self = toml::from_str(&text)?;
        // Config paths historically resolve from the tool crate directory.
        let base = path
            .parent()
            .and_then(Path::parent)
            .unwrap_or_else(|| Path::new("."));
        config.run.output_dir = base.join(&config.run.output_dir);
        config.run.data_dir = base.join(&config.run.data_dir);
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.data.patch_size < 16 || !self.data.patch_size.is_multiple_of(8) {
            return Err("patch_size must be at least 16 and divisible by 8".into());
        }
        if self.data.sample_count < 2 {
            return Err("sample_count must be at least 2".into());
        }
        if !(0.0..0.5).contains(&self.data.validation_fraction) {
            return Err("validation_fraction must be in [0, 0.5)".into());
        }
        if self.model.condition_dim != 4 {
            return Err("MIRA-1 conditioning contract currently requires condition_dim = 4".into());
        }
        if !(0.0..1.0).contains(&self.train.ema_decay) {
            return Err("ema_decay must be in [0, 1)".into());
        }
        if self.train.device.trim().is_empty() {
            return Err("train.device cannot be empty".into());
        }
        if self.train.num_workers > 256 {
            return Err("num_workers exceeds the supported configuration range".into());
        }
        if self.train.checkpoint_every_epochs == 0 {
            return Err("checkpoint_every_epochs must be positive".into());
        }
        Ok(())
    }
}
