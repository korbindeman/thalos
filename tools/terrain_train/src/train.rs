use std::path::Path;
use std::time::{Duration, Instant};

use burn::backend::Autodiff;
#[cfg(feature = "cpu")]
use burn::backend::Flex;
#[cfg(feature = "gpu")]
use burn::backend::Wgpu;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::TensorData;
use burn::tensor::backend::BackendTypes;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use safetensors::SafeTensors;
use serde::Serialize;
use sha2::{Digest, Sha256};
use thalos_terrain_learned::{
    AirlessDenoiser, AirlessDenoiserConfig, CONDITION_CHANNELS, DiffusionSchedule,
};

use crate::{config::Config, pyramid, synthetic::Sample};

#[cfg(feature = "cpu")]
type BaseBackend = Flex;
#[cfg(feature = "gpu")]
type BaseBackend = Wgpu;
type TrainBackend = Autodiff<BaseBackend>;
type TrainDevice = <TrainBackend as BackendTypes>::Device;

#[derive(Clone, Debug, Serialize)]
pub struct TrainingReport {
    pub backend: &'static str,
    pub burn_version: &'static str,
    pub epochs: usize,
    pub batches: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub elapsed_seconds: f64,
    pub target_scale_metres: f32,
    pub coarse_scale_metres: f32,
    pub requested_device: String,
    pub requested_ema_decay: f32,
    pub configured_workers: usize,
    pub model_tensor_sha256: String,
    pub validation: crate::validate::ValidationReport,
}

struct Prepared<'a> {
    sample: &'a Sample,
    coarse: crate::grid::Grid,
    target: crate::grid::Grid,
}

pub fn run(
    config: &Config,
    samples: &[Sample],
    output_dir: &Path,
) -> Result<TrainingReport, Box<dyn std::error::Error>> {
    let validation_count =
        ((samples.len() as f32 * config.data.validation_fraction).ceil() as usize).max(1);
    let training_count = samples.len() - validation_count;
    let prepared: Vec<_> = samples[..training_count]
        .iter()
        .map(|sample| {
            let pyramid = pyramid::build(&sample.height);
            Prepared {
                sample,
                coarse: pyramid.coarse_for_s3,
                target: pyramid.full_resolution_bands[3].clone(),
            }
        })
        .collect();
    let target_scale = prepared
        .iter()
        .map(|item| item.target.max_abs())
        .fold(1.0f32, f32::max);
    let coarse_scale = prepared
        .iter()
        .map(|item| item.coarse.max_abs())
        .fold(1.0f32, f32::max);

    let device = Default::default();
    TrainBackend::seed(&device, config.run.seed);
    let mut model: AirlessDenoiser<TrainBackend> = AirlessDenoiserConfig::new()
        .with_hidden_channels(config.model.base_channels)
        .init(&device);
    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(
            config.train.gradient_clip,
        )))
        .init();
    let schedule = DiffusionSchedule::linear(
        config.diffusion.timesteps,
        config.diffusion.beta_start,
        config.diffusion.beta_end,
    )?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.run.seed ^ 0xa110_e551_u64);
    let started = Instant::now();
    let mut losses = Vec::new();
    let mut batches = 0usize;

    for _epoch in 0..config.train.epochs {
        for chunk in prepared.chunks(config.train.batch_size.max(1)) {
            let (input, expected) = batch_tensors(
                chunk,
                config,
                &schedule,
                target_scale,
                coarse_scale,
                &device,
                &mut rng,
            );
            let prediction = model.forward(input);
            let loss = MseLoss::new().forward(prediction, expected, Reduction::Auto);
            let loss_value: f32 = loss.clone().into_scalar();
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(config.train.learning_rate, model, gradients);
            losses.push(loss_value);
            batches += 1;
        }
    }

    let inference_model = model.valid();
    let validation = crate::validate::run(
        &inference_model,
        config,
        &schedule,
        target_scale,
        coarse_scale,
        &device,
    );
    crate::output::save_height_u16(
        &output_dir.join("validation_height.png"),
        &validation.generated_height,
    )?;
    crate::output::save_hillshade_region(
        &output_dir.join("validation_contact_sheet.png"),
        &validation.generated_height,
        config.validation.canvas_width,
        config.validation.canvas_height,
    )?;
    let mut store = SafetensorsStore::from_file(output_dir.join("model.safetensors"))
        .overwrite(true)
        .metadata("thalos_model", "mira_airless_patch_v0")
        .metadata("burn_version", "0.21.0");
    inference_model.save_into(&mut store)?;
    let model_tensor_sha256 = canonical_safetensors_hash(&output_dir.join("model.safetensors"))?;

    Ok(TrainingReport {
        backend: backend_name(),
        burn_version: "0.21.0",
        epochs: config.train.epochs,
        batches,
        initial_loss: losses.first().copied().unwrap_or(f32::NAN),
        final_loss: losses.last().copied().unwrap_or(f32::NAN),
        elapsed_seconds: duration_seconds(started.elapsed()),
        target_scale_metres: target_scale,
        coarse_scale_metres: coarse_scale,
        requested_device: config.train.device.clone(),
        requested_ema_decay: config.train.ema_decay,
        configured_workers: config.train.num_workers,
        model_tensor_sha256,
        validation: validation.report,
    })
}

fn batch_tensors(
    batch: &[Prepared<'_>],
    config: &Config,
    schedule: &DiffusionSchedule,
    target_scale: f32,
    coarse_scale: f32,
    device: &TrainDevice,
    rng: &mut ChaCha8Rng,
) -> (Tensor<TrainBackend, 4>, Tensor<TrainBackend, 4>) {
    let size = config.data.patch_size;
    let area = size * size;
    let mut input = vec![0.0f32; batch.len() * CONDITION_CHANNELS * area];
    let mut expected = vec![0.0f32; batch.len() * area];

    for (batch_index, prepared) in batch.iter().enumerate() {
        let step = rng.random_range(0..schedule.len());
        let (clean_scale, noise_scale) = schedule.noise_scales(step);
        for y in 0..size {
            for x in 0..size {
                let pixel = y * size + x;
                let noise = standard_normal(rng);
                let clean = prepared.target.values[pixel] / target_scale;
                set_channel(
                    &mut input,
                    batch_index,
                    0,
                    pixel,
                    area,
                    clean * clean_scale + noise * noise_scale,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    1,
                    pixel,
                    area,
                    prepared.coarse.values[pixel] / coarse_scale,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    2,
                    pixel,
                    area,
                    prepared.sample.parameters.crater_density / 40.0,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    3,
                    pixel,
                    area,
                    prepared.sample.parameters.gardening,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    4,
                    pixel,
                    area,
                    prepared.sample.mare_mask.values[pixel],
                );
                set_channel(
                    &mut input,
                    batch_index,
                    5,
                    pixel,
                    area,
                    x as f32 / (size - 1) as f32 * 2.0 - 1.0,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    6,
                    pixel,
                    area,
                    y as f32 / (size - 1) as f32 * 2.0 - 1.0,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    7,
                    pixel,
                    area,
                    step as f32 / (schedule.len() - 1) as f32,
                );
                expected[batch_index * area + pixel] = noise;
            }
        }
    }

    (
        Tensor::from_data(
            TensorData::new(input, [batch.len(), CONDITION_CHANNELS, size, size]),
            device,
        ),
        Tensor::from_data(
            TensorData::new(expected, [batch.len(), 1, size, size]),
            device,
        ),
    )
}

fn set_channel(
    values: &mut [f32],
    batch: usize,
    channel: usize,
    pixel: usize,
    area: usize,
    value: f32,
) {
    values[(batch * CONDITION_CHANNELS + channel) * area + pixel] = value;
}

fn standard_normal(rng: &mut ChaCha8Rng) -> f32 {
    let u1 = rng.random::<f32>().max(f32::MIN_POSITIVE);
    let u2 = rng.random::<f32>();
    (-2.0f32 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn backend_name() -> &'static str {
    #[cfg(feature = "cpu")]
    {
        "burn-flex-autodiff"
    }
    #[cfg(feature = "gpu")]
    {
        "burn-wgpu-autodiff"
    }
}

fn canonical_safetensors_hash(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut names = tensors.names();
    names.sort_unstable();
    let mut hash = Sha256::new();
    for name in names {
        let tensor = tensors.tensor(name)?;
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(format!("{:?}", tensor.dtype()).as_bytes());
        for dimension in tensor.shape() {
            hash.update((*dimension as u64).to_le_bytes());
        }
        hash.update(tensor.data());
    }
    Ok(format!("{:x}", hash.finalize()))
}
