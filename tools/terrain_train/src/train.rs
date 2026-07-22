use std::path::Path;
use std::time::{Duration, Instant};
use std::{collections::BTreeMap, io};

use burn::backend::Autodiff;
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "cpu")]
use burn::backend::Flex;
#[cfg(feature = "gpu")]
use burn::backend::Wgpu;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, ParamId};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{DefaultRecorder, Recorder};
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::TensorData;
use burn::tensor::backend::BackendTypes;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thalos_terrain_learned::{
    AirlessDenoiser, AirlessDenoiserConfig, CONDITION_CHANNELS, DiffusionPrediction,
    DiffusionSchedule, TimeConditioning, Upsampling,
};

use crate::{
    config::Config,
    pyramid,
    sample::{Sample, Split},
};

#[cfg(feature = "cpu")]
type BaseBackend = Flex;
#[cfg(feature = "gpu")]
type BaseBackend = Wgpu;
#[cfg(feature = "cuda")]
type BaseBackend = Cuda;
type TrainBackend = Autodiff<BaseBackend>;
type TrainDevice = <TrainBackend as BackendTypes>::Device;
type InferenceDevice = <BaseBackend as BackendTypes>::Device;

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
    pub schedule_terminal_alpha_bar: f32,
    pub schedule_terminal_signal_to_noise: f32,
    pub prediction: DiffusionPrediction,
    pub upsampling: Upsampling,
    pub time_conditioning: TimeConditioning,
    pub requested_device: String,
    pub requested_ema_decay: f32,
    pub resumed_from_epoch: usize,
    pub configured_workers: usize,
    pub training_samples: usize,
    pub validation_samples: usize,
    pub synthetic_samples: usize,
    pub real_samples: usize,
    pub source_ids: Vec<String>,
    pub model_tensor_sha256: String,
    pub raw_model_tensor_sha256: String,
    pub validation: crate::validate::ValidationReport,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CheckpointState {
    schema_version: u32,
    run_name: String,
    seed: u64,
    patch_size: usize,
    hidden_channels: usize,
    completed_epochs: usize,
    batches: usize,
    initial_loss: f32,
    final_loss: f32,
    target_scale_metres: f32,
    coarse_scale_metres: f32,
    #[serde(default)]
    prediction: DiffusionPrediction,
    #[serde(default)]
    upsampling: Upsampling,
    #[serde(default)]
    time_conditioning: TimeConditioning,
    parameter_ids: BTreeMap<String, String>,
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
    let prepared: Vec<_> = samples
        .iter()
        .filter(|sample| sample.provenance.split == Split::Train)
        .map(|sample| {
            let pyramid = pyramid::build(&sample.height);
            Prepared {
                sample,
                coarse: pyramid.coarse_for_s3,
                target: pyramid.full_resolution_bands[3].clone(),
            }
        })
        .collect();
    if prepared.is_empty() {
        return Err("training corpus contains no train samples".into());
    }
    let mut source_ids: Vec<_> = samples
        .iter()
        .map(|sample| sample.provenance.source_id.clone())
        .collect();
    source_ids.sort();
    source_ids.dedup();
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
        .with_upsampling(config.model.upsampling)
        .with_time_conditioning(config.model.time_conditioning)
        .init(&device);
    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(
            config.train.gradient_clip,
        )))
        .init();
    let mut ema_model = model.valid();
    let schedule = DiffusionSchedule::linear(
        config.diffusion.timesteps,
        config.diffusion.beta_start,
        config.diffusion.beta_end,
    )?;
    let started = Instant::now();
    let mut losses = Vec::new();
    let mut batches = 0usize;
    let mut start_epoch = 0usize;
    let checkpoint_state_path = output_dir.join("checkpoint.json");

    if config.train.resume && checkpoint_state_path.exists() {
        let state: CheckpointState =
            serde_json::from_slice(&std::fs::read(&checkpoint_state_path)?)?;
        validate_checkpoint(config, &state, target_scale, coarse_scale)?;
        let mut raw_store =
            SafetensorsStore::from_file(output_dir.join("checkpoint_model.safetensors"));
        ensure_loaded(model.load_from(&mut raw_store)?, "raw model")?;
        let mut ema_store =
            SafetensorsStore::from_file(output_dir.join("checkpoint_ema.safetensors"));
        ensure_loaded(ema_model.load_from(&mut ema_store)?, "EMA model")?;
        let optimizer_record = load_record_like(
            optimizer.to_record(),
            &output_dir.join("checkpoint_optimizer"),
            &device,
        )?;
        let current_ids = parameter_ids(&model)?;
        let mut old_to_new = BTreeMap::new();
        for (path, old_id) in &state.parameter_ids {
            let current_id = current_ids.get(path).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("checkpoint parameter path {path:?} is missing"),
                )
            })?;
            old_to_new.insert(
                ParamId::deserialize(old_id),
                ParamId::deserialize(current_id),
            );
        }
        let mut remapped_record = optimizer.to_record();
        remapped_record.clear();
        for (old_id, record) in optimizer_record {
            let new_id = old_to_new.get(&old_id).copied().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("checkpoint optimizer references unknown parameter {old_id}"),
                )
            })?;
            remapped_record.insert(new_id, record);
        }
        optimizer = optimizer.load_record(remapped_record);
        start_epoch = state.completed_epochs;
        batches = state.batches;
        losses.push(state.initial_loss);
        losses.push(state.final_loss);
    }

    let batch_size = config.train.batch_size.max(1);
    let batches_per_epoch = prepared.len().div_ceil(batch_size);
    for epoch in start_epoch..config.train.epochs {
        for (batch_in_epoch, chunk) in prepared.chunks(batch_size).enumerate() {
            let global_batch = epoch * batches_per_epoch + batch_in_epoch;
            let mut rng = ChaCha8Rng::seed_from_u64(batch_seed(config.run.seed, global_batch));
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
            ema_model = ema_model.ema_update(&model.valid(), config.train.ema_decay);
            losses.push(loss_value);
            batches += 1;
        }
        let completed_epochs = epoch + 1;
        if completed_epochs.is_multiple_of(config.train.checkpoint_every_epochs)
            || completed_epochs == config.train.epochs
        {
            save_model(
                &model.valid(),
                &output_dir.join("checkpoint_model.safetensors"),
                "mira_airless_patch_checkpoint_raw",
            )?;
            save_model(
                &ema_model,
                &output_dir.join("checkpoint_ema.safetensors"),
                "mira_airless_patch_checkpoint_ema",
            )?;
            DefaultRecorder::default().record(
                optimizer.to_record(),
                output_dir.join("checkpoint_optimizer"),
            )?;
            let state = CheckpointState {
                schema_version: 3,
                run_name: config.run.name.clone(),
                seed: config.run.seed,
                patch_size: config.data.patch_size,
                hidden_channels: config.model.base_channels,
                completed_epochs,
                batches,
                initial_loss: losses.first().copied().unwrap_or(f32::NAN),
                final_loss: losses.last().copied().unwrap_or(f32::NAN),
                target_scale_metres: target_scale,
                coarse_scale_metres: coarse_scale,
                prediction: config.diffusion.prediction,
                upsampling: config.model.upsampling,
                time_conditioning: config.model.time_conditioning,
                parameter_ids: parameter_ids(&model)?,
            };
            std::fs::write(&checkpoint_state_path, serde_json::to_vec_pretty(&state)?)?;
        }
    }

    let inference_model = ema_model;
    let validation = crate::validate::run(
        &inference_model,
        config,
        &schedule,
        target_scale,
        coarse_scale,
        &device,
        validation_reference(config, samples)?,
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
    crate::output::save_contact_sheet(
        &output_dir.join("validation_comparison.png"),
        &[
            ("target", &validation.target_height),
            ("coarse", &validation.coarse_height),
            ("generated", &validation.generated_height),
            ("error", &validation.error_height),
        ],
    )?;
    for (name, grid) in [
        ("target", &validation.target_height),
        ("coarse", &validation.coarse_height),
        ("generated", &validation.generated_height),
        ("error", &validation.error_height),
    ] {
        crate::output::save_hillshade(&output_dir.join(format!("validation_{name}.png")), grid)?;
    }
    save_model(
        &inference_model,
        &output_dir.join("model.safetensors"),
        "mira_airless_patch_v0_ema",
    )?;
    save_model(
        &model.valid(),
        &output_dir.join("raw_model.safetensors"),
        "mira_airless_patch_v0_raw",
    )?;
    let model_tensor_sha256 = canonical_safetensors_hash(&output_dir.join("model.safetensors"))?;
    let raw_model_tensor_sha256 =
        canonical_safetensors_hash(&output_dir.join("raw_model.safetensors"))?;

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
        schedule_terminal_alpha_bar: schedule.terminal_alpha_bar(),
        schedule_terminal_signal_to_noise: schedule.terminal_signal_to_noise(),
        prediction: config.diffusion.prediction,
        upsampling: config.model.upsampling,
        time_conditioning: config.model.time_conditioning,
        requested_device: config.train.device.clone(),
        requested_ema_decay: config.train.ema_decay,
        resumed_from_epoch: start_epoch,
        configured_workers: config.train.num_workers,
        training_samples: samples
            .iter()
            .filter(|sample| sample.provenance.split == Split::Train)
            .count(),
        validation_samples: samples
            .iter()
            .filter(|sample| sample.provenance.split == Split::Validation)
            .count(),
        synthetic_samples: samples
            .iter()
            .filter(|sample| sample.provenance.synthetic)
            .count(),
        real_samples: samples
            .iter()
            .filter(|sample| !sample.provenance.synthetic)
            .count(),
        source_ids,
        model_tensor_sha256,
        raw_model_tensor_sha256,
        validation: validation.report,
    })
}

pub fn evaluate(
    config: &Config,
    samples: &[Sample],
    output_dir: &Path,
) -> Result<crate::validate::ValidationReport, Box<dyn std::error::Error>> {
    let prepared: Vec<_> = samples
        .iter()
        .filter(|sample| sample.provenance.split == Split::Train)
        .map(|sample| {
            let pyramid = pyramid::build(&sample.height);
            (
                pyramid.full_resolution_bands[3].clone(),
                pyramid.coarse_for_s3,
            )
        })
        .collect();
    if prepared.is_empty() {
        return Err("evaluation corpus contains no train samples".into());
    }
    let target_scale = prepared
        .iter()
        .map(|item| item.0.max_abs())
        .fold(1.0f32, f32::max);
    let coarse_scale = prepared
        .iter()
        .map(|item| item.1.max_abs())
        .fold(1.0f32, f32::max);
    let device: InferenceDevice = Default::default();
    BaseBackend::seed(&device, config.run.seed);
    let mut model: AirlessDenoiser<BaseBackend> = AirlessDenoiserConfig::new()
        .with_hidden_channels(config.model.base_channels)
        .with_upsampling(config.model.upsampling)
        .with_time_conditioning(config.model.time_conditioning)
        .init(&device);
    let mut store = SafetensorsStore::from_file(output_dir.join("model.safetensors"));
    ensure_loaded(model.load_from(&mut store)?, "evaluation model")?;
    let schedule = DiffusionSchedule::linear(
        config.diffusion.timesteps,
        config.diffusion.beta_start,
        config.diffusion.beta_end,
    )?;
    let reference = validation_reference(config, samples)?;
    let validation = crate::validate::run(
        &model,
        config,
        &schedule,
        target_scale,
        coarse_scale,
        &device,
        reference,
    );
    crate::output::save_contact_sheet(
        &output_dir.join("validation_comparison.png"),
        &[
            ("target", &validation.target_height),
            ("coarse", &validation.coarse_height),
            ("generated", &validation.generated_height),
            ("error", &validation.error_height),
        ],
    )?;
    std::fs::write(
        output_dir.join("evaluation_report.json"),
        serde_json::to_vec_pretty(&validation.report)?,
    )?;
    Ok(validation.report)
}

/// Load the exported EMA inference model from a completed run directory.
pub fn load_inference_model(
    config: &Config,
    run_dir: &Path,
) -> Result<(AirlessDenoiser<BaseBackend>, InferenceDevice), Box<dyn std::error::Error>> {
    let device: InferenceDevice = Default::default();
    BaseBackend::seed(&device, config.run.seed);
    let mut model: AirlessDenoiser<BaseBackend> = AirlessDenoiserConfig::new()
        .with_hidden_channels(config.model.base_channels)
        .with_upsampling(config.model.upsampling)
        .with_time_conditioning(config.model.time_conditioning)
        .init(&device);
    let mut store = SafetensorsStore::from_file(run_dir.join("model.safetensors"));
    ensure_loaded(model.load_from(&mut store)?, "sphere-preview model")?;
    Ok((model, device))
}

fn validation_reference<'a>(
    config: &Config,
    samples: &'a [Sample],
) -> Result<Option<&'a Sample>, Box<dyn std::error::Error>> {
    if let Some(source_id) = &config.validation.reference_source_id {
        let sample = samples
            .iter()
            .find(|sample| &sample.provenance.source_id == source_id)
            .ok_or_else(|| format!("validation reference source {source_id:?} was not loaded"))?;
        Ok(Some(sample))
    } else if config.validation.use_first_training_sample {
        let sample = samples
            .iter()
            .find(|sample| sample.provenance.split == Split::Train)
            .ok_or("validation requested the first training sample, but none was loaded")?;
        Ok(Some(sample))
    } else {
        Ok(None)
    }
}

fn ensure_loaded(
    result: burn::store::ApplyResult,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if result.is_success() {
        Ok(())
    } else {
        Err(format!("checkpoint {label} load failed: {result:?}").into())
    }
}

fn validate_checkpoint(
    config: &Config,
    state: &CheckpointState,
    target_scale: f32,
    coarse_scale: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let compatible = state.schema_version == 3
        && state.run_name == config.run.name
        && state.seed == config.run.seed
        && state.patch_size == config.data.patch_size
        && state.hidden_channels == config.model.base_channels
        && state.target_scale_metres.to_bits() == target_scale.to_bits()
        && state.coarse_scale_metres.to_bits() == coarse_scale.to_bits()
        && state.prediction == config.diffusion.prediction
        && state.upsampling == config.model.upsampling
        && state.time_conditioning == config.model.time_conditioning
        && state.completed_epochs <= config.train.epochs;
    if compatible {
        Ok(())
    } else {
        Err(format!("checkpoint is incompatible with current config: {state:?}").into())
    }
}

fn save_model<B: Backend>(
    model: &AirlessDenoiser<B>,
    path: &Path,
    identity: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = SafetensorsStore::from_file(path)
        .overwrite(true)
        .metadata("thalos_model", identity)
        .metadata("burn_version", "0.21.0");
    model.save_into(&mut store)?;
    Ok(())
}

fn parameter_ids<B: Backend>(
    model: &AirlessDenoiser<B>,
) -> Result<BTreeMap<String, String>, Box<dyn std::error::Error>> {
    model
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| {
            let path = snapshot.full_path();
            let id = snapshot.tensor_id.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("trainable parameter {path:?} has no Burn ParamId"),
                )
            })?;
            Ok((path, id.serialize()))
        })
        .collect()
}

fn load_record_like<B: Backend, R: burn::record::Record<B>>(
    prototype: R,
    path: &Path,
    device: &B::Device,
) -> Result<R, Box<dyn std::error::Error>> {
    drop(prototype);
    Ok(DefaultRecorder::default().load(path.to_path_buf(), device)?)
}

fn batch_seed(run_seed: u64, global_batch: usize) -> u64 {
    let mut value = run_seed
        ^ (global_batch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ 0xa110_e551_d1ff_0510;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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
                    prepared.sample.parameters.mare_fraction,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    4,
                    pixel,
                    area,
                    prepared.sample.parameters.gardening,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    5,
                    pixel,
                    area,
                    prepared.sample.parameters.rim_sharpness / 2.0,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    6,
                    pixel,
                    area,
                    prepared.sample.mare_mask.values[pixel],
                );
                set_channel(
                    &mut input,
                    batch_index,
                    7,
                    pixel,
                    area,
                    prepared.sample.scale_condition(),
                );
                set_channel(
                    &mut input,
                    batch_index,
                    8,
                    pixel,
                    area,
                    x as f32 / (size - 1) as f32 * 2.0 - 1.0,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    9,
                    pixel,
                    area,
                    y as f32 / (size - 1) as f32 * 2.0 - 1.0,
                );
                set_channel(
                    &mut input,
                    batch_index,
                    10,
                    pixel,
                    area,
                    step as f32 / (schedule.len() - 1) as f32,
                );
                expected[batch_index * area + pixel] = config.diffusion.prediction.training_target(
                    clean_scale,
                    noise_scale,
                    clean,
                    noise,
                );
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
    #[cfg(feature = "cuda")]
    {
        "burn-cuda-autodiff"
    }
}

pub fn backend_report() -> String {
    let device: TrainDevice = Default::default();
    format!("{} ({})", backend_name(), BaseBackend::name(&device))
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
