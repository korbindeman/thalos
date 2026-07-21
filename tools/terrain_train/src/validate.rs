use burn::prelude::*;
use burn::tensor::TensorData;
use serde::Serialize;
use thalos_terrain_learned::{AirlessDenoiser, CONDITION_CHANNELS, DiffusionSchedule};

use crate::{config::Config, grid::Grid, pyramid, synthetic};

#[derive(Clone, Debug, Serialize)]
pub struct ValidationReport {
    pub canvas: [usize; 2],
    pub window: usize,
    pub stride: usize,
    pub windows: usize,
    pub sample_steps: usize,
    pub repeated_generation_max_abs_delta_m: f32,
    pub overlap_prediction_rms_m: f32,
}

pub struct ValidationOutput {
    pub generated_height: Grid,
    pub report: ValidationReport,
}

pub fn run<B: Backend>(
    model: &AirlessDenoiser<B>,
    config: &Config,
    schedule: &DiffusionSchedule,
    target_scale: f32,
    coarse_scale: f32,
    device: &B::Device,
) -> ValidationOutput {
    let full_size = config
        .validation
        .canvas_width
        .max(config.validation.canvas_height);
    let mut data = config.data.clone();
    data.patch_size = full_size.next_multiple_of(8);
    let sample = synthetic::generate(&data, config.run.seed ^ 0x5ea1_cafe_d15c_a11e);
    let pyramid = pyramid::build(&sample.height);

    let first = generate_canvas(
        model,
        config,
        schedule,
        target_scale,
        coarse_scale,
        device,
        &sample,
        &pyramid.coarse_for_s3,
    );
    let second = generate_canvas(
        model,
        config,
        schedule,
        target_scale,
        coarse_scale,
        device,
        &sample,
        &pyramid.coarse_for_s3,
    );
    let max_delta = first
        .residual
        .values
        .iter()
        .zip(&second.residual.values)
        .fold(0.0f32, |maximum, (left, right)| {
            maximum.max((left - right).abs())
        });
    let mut height = first.residual.clone();
    for y in 0..height.size {
        for x in 0..height.size {
            let source_x = x.min(pyramid.coarse_for_s3.size - 1);
            let source_y = y.min(pyramid.coarse_for_s3.size - 1);
            height.values[y * height.size + x] += pyramid.coarse_for_s3.get(source_x, source_y);
        }
    }

    ValidationOutput {
        generated_height: height,
        report: ValidationReport {
            canvas: [
                config.validation.canvas_width,
                config.validation.canvas_height,
            ],
            window: config.data.patch_size,
            stride: config.validation.stride,
            windows: first.windows,
            sample_steps: config.diffusion.sample_steps.min(schedule.len()),
            repeated_generation_max_abs_delta_m: max_delta,
            overlap_prediction_rms_m: first.overlap_rms,
        },
    }
}

struct GeneratedCanvas {
    residual: Grid,
    windows: usize,
    overlap_rms: f32,
}

#[allow(clippy::too_many_arguments)]
fn generate_canvas<B: Backend>(
    model: &AirlessDenoiser<B>,
    config: &Config,
    schedule: &DiffusionSchedule,
    target_scale: f32,
    coarse_scale: f32,
    device: &B::Device,
    sample: &synthetic::Sample,
    coarse: &Grid,
) -> GeneratedCanvas {
    let width = config.validation.canvas_width;
    let height = config.validation.canvas_height;
    let window = config.data.patch_size;
    let origins_x = origins(width, window, config.validation.stride);
    let origins_y = origins(height, window, config.validation.stride);
    let mut sum = vec![0.0f32; width * height];
    let mut raw_sum = vec![0.0f32; width * height];
    let mut sum_squares = vec![0.0f32; width * height];
    let mut weights = vec![0.0f32; width * height];
    let mut counts = vec![0usize; width * height];

    for &origin_y in &origins_y {
        for &origin_x in &origins_x {
            let prediction = sample_window(
                model,
                config,
                schedule,
                target_scale,
                coarse_scale,
                device,
                sample,
                coarse,
                origin_x,
                origin_y,
            );
            for y in 0..window {
                for x in 0..window {
                    let global_x = origin_x + x;
                    let global_y = origin_y + y;
                    if global_x >= width || global_y >= height {
                        continue;
                    }
                    let index = global_y * width + global_x;
                    let value = prediction[y * window + x];
                    let weight = blend_weight(x, y, window);
                    sum[index] += value * weight;
                    raw_sum[index] += value;
                    sum_squares[index] += value * value;
                    weights[index] += weight;
                    counts[index] += 1;
                }
            }
        }
    }

    let mut residual = Grid {
        size: width.max(height),
        values: vec![0.0; width.max(height).pow(2)],
    };
    let mut variance_sum = 0.0f64;
    let mut overlap_count = 0usize;
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            residual.values[y * residual.size + x] = sum[index] / weights[index].max(1e-8);
            if counts[index] > 1 {
                let unweighted_mean = raw_sum[index] / counts[index] as f32;
                let variance = (sum_squares[index] / counts[index] as f32
                    - unweighted_mean * unweighted_mean)
                    .max(0.0);
                variance_sum += f64::from(variance);
                overlap_count += 1;
            }
        }
    }
    GeneratedCanvas {
        residual,
        windows: origins_x.len() * origins_y.len(),
        overlap_rms: (variance_sum / overlap_count.max(1) as f64).sqrt() as f32,
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_window<B: Backend>(
    model: &AirlessDenoiser<B>,
    config: &Config,
    schedule: &DiffusionSchedule,
    target_scale: f32,
    coarse_scale: f32,
    device: &B::Device,
    sample: &synthetic::Sample,
    coarse: &Grid,
    origin_x: usize,
    origin_y: usize,
) -> Vec<f32> {
    let size = config.data.patch_size;
    let area = size * size;
    let mut state = vec![0.0f32; area];
    for y in 0..size {
        for x in 0..size {
            state[y * size + x] = coordinate_normal(config.run.seed, origin_x + x, origin_y + y);
        }
    }
    let steps = inference_steps(schedule.len(), config.diffusion.sample_steps);
    for (step_index, &step) in steps.iter().enumerate() {
        let mut input = vec![0.0f32; CONDITION_CHANNELS * area];
        for y in 0..size {
            for x in 0..size {
                let pixel = y * size + x;
                let global_x = origin_x + x;
                let global_y = origin_y + y;
                set(&mut input, 0, pixel, area, state[pixel]);
                set(
                    &mut input,
                    1,
                    pixel,
                    area,
                    coarse.get(global_x, global_y) / coarse_scale,
                );
                set(
                    &mut input,
                    2,
                    pixel,
                    area,
                    sample.parameters.crater_density / 40.0,
                );
                set(&mut input, 3, pixel, area, sample.parameters.gardening);
                set(
                    &mut input,
                    4,
                    pixel,
                    area,
                    sample.mare_mask.get(global_x, global_y),
                );
                set(
                    &mut input,
                    5,
                    pixel,
                    area,
                    global_x as f32 / (config.validation.canvas_width - 1) as f32 * 2.0 - 1.0,
                );
                set(
                    &mut input,
                    6,
                    pixel,
                    area,
                    global_y as f32 / (config.validation.canvas_height - 1) as f32 * 2.0 - 1.0,
                );
                set(
                    &mut input,
                    7,
                    pixel,
                    area,
                    step as f32 / (schedule.len() - 1) as f32,
                );
            }
        }
        let tensor = Tensor::<B, 4>::from_data(
            TensorData::new(input, [1, CONDITION_CHANNELS, size, size]),
            device,
        );
        let epsilon = model
            .forward(tensor)
            .into_data()
            .to_vec::<f32>()
            .expect("Flex f32 output");
        let alpha = schedule.alpha_bar(step);
        let previous_alpha = steps
            .get(step_index + 1)
            .map(|previous| schedule.alpha_bar(*previous))
            .unwrap_or(1.0);
        for pixel in 0..area {
            let clean = (state[pixel] - (1.0 - alpha).sqrt() * epsilon[pixel]) / alpha.sqrt();
            state[pixel] =
                previous_alpha.sqrt() * clean + (1.0 - previous_alpha).sqrt() * epsilon[pixel];
        }
    }
    for value in &mut state {
        *value = value.clamp(-2.5, 2.5) * target_scale;
    }
    state
}

fn origins(extent: usize, window: usize, stride: usize) -> Vec<usize> {
    let last = extent.saturating_sub(window);
    let mut origins: Vec<_> = (0..=last).step_by(stride.max(1)).collect();
    if origins.last().copied() != Some(last) {
        origins.push(last);
    }
    origins
}

fn inference_steps(total: usize, requested: usize) -> Vec<usize> {
    let count = requested.clamp(2, total);
    (0..count)
        .map(|index| {
            ((total - 1) as f32 * (1.0 - index as f32 / (count - 1) as f32)).round() as usize
        })
        .collect()
}

fn blend_weight(x: usize, y: usize, size: usize) -> f32 {
    let wx = (std::f32::consts::PI * (x as f32 + 0.5) / size as f32)
        .sin()
        .powi(2);
    let wy = (std::f32::consts::PI * (y as f32 + 0.5) / size as f32)
        .sin()
        .powi(2);
    (wx * wy).max(0.01)
}

fn set(input: &mut [f32], channel: usize, pixel: usize, area: usize, value: f32) {
    input[channel * area + pixel] = value;
}

fn coordinate_normal(seed: u64, x: usize, y: usize) -> f32 {
    let left = hash(seed ^ (x as u64).wrapping_mul(0x9e37_79b9));
    let right = hash(left ^ (y as u64).wrapping_mul(0x85eb_ca6b));
    let u1 = (((left >> 40) as u32) as f32 / (1u32 << 24) as f32).max(f32::MIN_POSITIVE);
    let u2 = ((right >> 40) as u32) as f32 / (1u32 << 24) as f32;
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn hash(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
