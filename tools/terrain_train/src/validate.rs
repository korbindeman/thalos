use burn::prelude::*;
use burn::tensor::TensorData;
use serde::Serialize;
use thalos_terrain_learned::{AirlessDenoiser, CONDITION_CHANNELS, DiffusionSchedule};

use crate::{
    config::Config,
    grid::Grid,
    pyramid,
    sample::{Sample, Split},
    synthetic,
};

#[derive(Clone, Debug, Serialize)]
pub struct ValidationReport {
    pub canvas: [usize; 2],
    pub window: usize,
    pub stride: usize,
    pub windows: usize,
    pub sample_steps: usize,
    pub repeated_generation_max_abs_delta_m: f32,
    pub overlap_prediction_rms_m: f32,
    pub reconstruction_rms_m: f32,
    pub reconstruction_max_abs_m: f32,
    pub target_metrics: TerrainMetrics,
    pub generated_metrics: TerrainMetrics,
}

#[derive(Clone, Debug, Serialize)]
pub struct TerrainMetrics {
    pub slope_quantiles: [f32; 3],
    pub structure_rms_m: [f32; 5],
    pub radial_power_slope: f32,
    pub crater_proxy_counts: [usize; 4],
}

pub struct ValidationOutput {
    pub generated_height: Grid,
    pub target_height: Grid,
    pub coarse_height: Grid,
    pub error_height: Grid,
    pub report: ValidationReport,
}

pub fn run<B: Backend>(
    model: &AirlessDenoiser<B>,
    config: &Config,
    schedule: &DiffusionSchedule,
    target_scale: f32,
    coarse_scale: f32,
    device: &B::Device,
    reference: Option<&Sample>,
) -> ValidationOutput {
    let full_size = config
        .validation
        .canvas_width
        .max(config.validation.canvas_height);
    let generated_sample;
    let sample = match reference {
        Some(sample) => sample,
        None => {
            let mut data = config.data.clone();
            data.patch_size = full_size.next_multiple_of(8);
            generated_sample = synthetic::generate(
                &data,
                config.run.seed ^ 0x5ea1_cafe_d15c_a11e,
                Split::Validation,
            );
            &generated_sample
        }
    };
    assert!(sample.height.size >= full_size);
    let pyramid = pyramid::build(&sample.height);

    let first = generate_canvas(
        model,
        config,
        schedule,
        target_scale,
        coarse_scale,
        device,
        sample,
        &pyramid.coarse_for_s3,
    );
    let second = generate_canvas(
        model,
        config,
        schedule,
        target_scale,
        coarse_scale,
        device,
        sample,
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
    let mut target_height = Grid::zeros(height.size);
    let mut coarse_height = Grid::zeros(height.size);
    let mut error_height = Grid::zeros(height.size);
    let mut squared_error = 0.0f64;
    let mut maximum_error = 0.0f32;
    let mut error_count = 0usize;
    for y in 0..config.validation.canvas_height {
        for x in 0..config.validation.canvas_width {
            let index = y * height.size + x;
            let target = sample.height.get(x, y);
            let coarse = pyramid.coarse_for_s3.get(x, y);
            let error = height.get(x, y) - target;
            target_height.values[index] = target;
            coarse_height.values[index] = coarse;
            error_height.values[index] = error;
            squared_error += f64::from(error * error);
            maximum_error = maximum_error.max(error.abs());
            error_count += 1;
        }
    }

    let target_metrics = terrain_metrics(
        &target_height,
        config.validation.canvas_width,
        config.validation.canvas_height,
    );
    let generated_metrics = terrain_metrics(
        &height,
        config.validation.canvas_width,
        config.validation.canvas_height,
    );
    ValidationOutput {
        generated_height: height,
        target_height,
        coarse_height,
        error_height,
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
            reconstruction_rms_m: (squared_error / error_count as f64).sqrt() as f32,
            reconstruction_max_abs_m: maximum_error,
            target_metrics,
            generated_metrics,
        },
    }
}

fn terrain_metrics(grid: &Grid, width: usize, height: usize) -> TerrainMetrics {
    let mut slopes = Vec::with_capacity(width * height);
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let dx = (grid.get(x + 1, y) - grid.get(x - 1, y)) * 0.5;
            let dy = (grid.get(x, y + 1) - grid.get(x, y - 1)) * 0.5;
            slopes.push((dx * dx + dy * dy).sqrt());
        }
    }
    slopes.sort_by(f32::total_cmp);
    let quantile = |fraction: f32| {
        let index = ((slopes.len().saturating_sub(1)) as f32 * fraction).round() as usize;
        slopes.get(index).copied().unwrap_or(0.0)
    };
    TerrainMetrics {
        slope_quantiles: [quantile(0.5), quantile(0.9), quantile(0.99)],
        structure_rms_m: [1, 2, 4, 8, 16].map(|lag| structure_rms(grid, width, height, lag)),
        radial_power_slope: radial_power_slope(grid, width, height),
        crater_proxy_counts: [2, 4, 8, 12]
            .map(|radius| crater_proxy_count(grid, width, height, radius)),
    }
}

fn structure_rms(grid: &Grid, width: usize, height: usize, lag: usize) -> f32 {
    if width <= lag || height <= lag {
        return 0.0;
    }
    let mut squared = 0.0f64;
    let mut count = 0usize;
    for y in 0..height {
        for x in 0..width - lag {
            let delta = grid.get(x + lag, y) - grid.get(x, y);
            squared += f64::from(delta * delta);
            count += 1;
        }
    }
    for y in 0..height - lag {
        for x in 0..width {
            let delta = grid.get(x, y + lag) - grid.get(x, y);
            squared += f64::from(delta * delta);
            count += 1;
        }
    }
    (squared / count.max(1) as f64).sqrt() as f32
}

fn radial_power_slope(grid: &Grid, width: usize, height: usize) -> f32 {
    let size = width.min(height).min(64);
    if size < 8 {
        return 0.0;
    }
    let mut rows = vec![(0.0f64, 0.0f64); size * size];
    for y in 0..size {
        for kx in 0..size {
            let mut sum = (0.0, 0.0);
            for x in 0..size {
                let source_x = x * width / size;
                let source_y = y * height / size;
                let angle = -std::f64::consts::TAU * (kx * x) as f64 / size as f64;
                let value = f64::from(grid.get(source_x, source_y));
                sum.0 += value * angle.cos();
                sum.1 += value * angle.sin();
            }
            rows[y * size + kx] = sum;
        }
    }
    let bins = size / 2;
    let mut power = vec![0.0f64; bins];
    let mut counts = vec![0usize; bins];
    for ky in 0..size {
        for kx in 0..size {
            let mut sum = (0.0, 0.0);
            for y in 0..size {
                let angle = -std::f64::consts::TAU * (ky * y) as f64 / size as f64;
                let value = rows[y * size + kx];
                sum.0 += value.0 * angle.cos() - value.1 * angle.sin();
                sum.1 += value.0 * angle.sin() + value.1 * angle.cos();
            }
            let fx = kx.min(size - kx);
            let fy = ky.min(size - ky);
            let radius = ((fx * fx + fy * fy) as f64).sqrt().round() as usize;
            if (1..bins).contains(&radius) {
                power[radius] += sum.0 * sum.0 + sum.1 * sum.1;
                counts[radius] += 1;
            }
        }
    }
    let points: Vec<_> = (2..bins)
        .filter(|index| counts[*index] > 0 && power[*index] > 0.0)
        .map(|index| {
            (
                (index as f64).ln(),
                (power[index] / counts[index] as f64).ln(),
            )
        })
        .collect();
    linear_slope(&points) as f32
}

fn linear_slope(points: &[(f64, f64)]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let mean_x = points.iter().map(|point| point.0).sum::<f64>() / points.len() as f64;
    let mean_y = points.iter().map(|point| point.1).sum::<f64>() / points.len() as f64;
    let numerator = points
        .iter()
        .map(|point| (point.0 - mean_x) * (point.1 - mean_y))
        .sum::<f64>();
    let denominator = points
        .iter()
        .map(|point| (point.0 - mean_x).powi(2))
        .sum::<f64>();
    numerator / denominator.max(f64::EPSILON)
}

fn crater_proxy_count(grid: &Grid, width: usize, height: usize, radius: usize) -> usize {
    if width <= radius * 2 || height <= radius * 2 {
        return 0;
    }
    let threshold = grid.rms() * 0.04;
    let mut count = 0;
    for y in (radius..height - radius).step_by(radius.max(2)) {
        for x in (radius..width - radius).step_by(radius.max(2)) {
            let center = grid.get(x, y);
            let rim = [
                grid.get(x - radius, y),
                grid.get(x + radius, y),
                grid.get(x, y - radius),
                grid.get(x, y + radius),
            ]
            .iter()
            .sum::<f32>()
                * 0.25;
            if rim - center > threshold {
                count += 1;
            }
        }
    }
    count
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
    sample: &Sample,
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
    sample: &Sample,
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
                set(&mut input, 3, pixel, area, sample.parameters.mare_fraction);
                set(&mut input, 4, pixel, area, sample.parameters.gardening);
                set(
                    &mut input,
                    5,
                    pixel,
                    area,
                    sample.parameters.rim_sharpness / 2.0,
                );
                set(
                    &mut input,
                    6,
                    pixel,
                    area,
                    sample.mare_mask.get(global_x, global_y),
                );
                set(&mut input, 7, pixel, area, sample.scale_condition());
                set(
                    &mut input,
                    8,
                    pixel,
                    area,
                    global_x as f32 / (config.validation.canvas_width - 1) as f32 * 2.0 - 1.0,
                );
                set(
                    &mut input,
                    9,
                    pixel,
                    area,
                    global_y as f32 / (config.validation.canvas_height - 1) as f32 * 2.0 - 1.0,
                );
                set(
                    &mut input,
                    10,
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
        let prediction = model
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
            let (clean, epsilon) =
                config
                    .diffusion
                    .prediction
                    .reconstruct(alpha, state[pixel], prediction[pixel]);
            state[pixel] = previous_alpha.sqrt() * clean + (1.0 - previous_alpha).sqrt() * epsilon;
        }
    }
    for value in &mut state {
        *value = value.clamp(-2.5, 2.5) * target_scale;
    }
    state
}

pub(crate) fn origins(extent: usize, window: usize, stride: usize) -> Vec<usize> {
    let last = extent.saturating_sub(window);
    let mut origins: Vec<_> = (0..=last).step_by(stride.max(1)).collect();
    if origins.last().copied() != Some(last) {
        origins.push(last);
    }
    origins
}

pub(crate) fn inference_steps(total: usize, requested: usize) -> Vec<usize> {
    let count = requested.clamp(2, total);
    (0..count)
        .map(|index| {
            ((total - 1) as f32 * (1.0 - index as f32 / (count - 1) as f32)).round() as usize
        })
        .collect()
}

pub(crate) fn blend_weight(x: usize, y: usize, size: usize) -> f32 {
    let wx = (std::f32::consts::PI * (x as f32 + 0.5) / size as f32)
        .sin()
        .powi(2);
    let wy = (std::f32::consts::PI * (y as f32 + 0.5) / size as f32)
        .sin()
        .powi(2);
    (wx * wy).max(0.01)
}

pub(crate) fn set(input: &mut [f32], channel: usize, pixel: usize, area: usize, value: f32) {
    input[channel * area + pixel] = value;
}

pub(crate) fn coordinate_normal(seed: u64, x: usize, y: usize) -> f32 {
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
