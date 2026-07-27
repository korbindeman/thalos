//! Whole-sphere learned-terrain preview (the roadmap's first L3 scouting step).
//!
//! Samples the authored Mira package's per-face macro height as the coarse
//! conditioning field, runs the trained denoiser over each cube face with the
//! same 64-px windows and overlap fusion the validator uses, and renders an
//! equirectangular hillshade plus orthographic "full moon" discs (macro-only
//! versus learned-enhanced).
//!
//! Deliberate preview approximations, owned by L3 proper later:
//! - Faces are generated independently; residual-band seams at face borders
//!   are expected and are part of what L3 seam consensus must remove.
//! - The macro field is ~2.7 km/px, far coarser than the 40–237 m/px training
//!   band. The scale-condition channel is pinned to the SLDEM teacher value
//!   and the generated band is re-dimensionalized by the face/train coarse
//!   ratio, so the output is stylized proportional relief, not metric ground
//!   truth.
//! - The mare conditioning mask is a smooth low-elevation proxy of the macro,
//!   not the authored material map.

use std::path::{Path, PathBuf};

use burn::prelude::*;
use burn::tensor::TensorData;
use image::{Rgb, RgbImage};
use serde::Serialize;
use thalos_terrain::cache;
use thalos_terrain::cubemap::{CubemapFace, dir_to_face_uv};
use thalos_terrain::package::load_static_package;
use thalos_terrain::terrain_config::{TerrainCompileContext, TerrainCompileOptions};
use thalos_terrain_learned::{AirlessDenoiser, CONDITION_CHANNELS, DiffusionSchedule};
use thalos_world::{BodyKind, parsing::load_solar_system_from_dir};

use crate::{
    config::Config,
    grid::Grid,
    output, pyramid,
    validate::{blend_weight, coordinate_normal, inference_steps, origins, set},
};

/// SLDEM2015 metres-per-pixel, the coarsest scale the model trained on.
const SLDEM_METRES_PER_PIXEL: f32 = 236.901;
/// Conditioning controls for the fictional Mira seed (the expansion-source
/// defaults the model saw for real morphology).
const CRATER_DENSITY: f32 = 20.0;
const MARE_FRACTION: f32 = 0.2;
const GARDENING: f32 = 0.35;
const RIM_SHARPNESS: f32 = 1.0;
/// Screen-space relief exaggeration for the orthographic disc.
const DISC_RELIEF_GAIN: f32 = 5.0;

pub struct SphereOptions {
    pub assets_dir: PathBuf,
    pub body: String,
    pub face_size: usize,
    pub out_dir: Option<PathBuf>,
}

#[derive(Serialize)]
struct SphereManifest {
    run_name: String,
    body: String,
    package_content_key: String,
    package_resolution: u32,
    face_size: usize,
    window: usize,
    stride: usize,
    sample_steps: usize,
    windows_per_face: usize,
    train_target_scale_metres: f32,
    train_coarse_scale_metres: f32,
    face_coarse_scale_metres: f32,
    residual_gain: f32,
    scale_condition_metres_per_pixel: f32,
    controls: [f32; 4],
    elapsed_seconds: f64,
}

pub fn run<B: Backend>(
    model: &AirlessDenoiser<B>,
    config: &Config,
    schedule: &DiffusionSchedule,
    device: &B::Device,
    options: &SphereOptions,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let started = std::time::Instant::now();
    let run_dir = &config.run.output_dir;
    let (target_scale, coarse_scale) = checkpoint_scales(run_dir, config)?;

    // Load the authored package exactly the way the game does.
    let system = load_solar_system_from_dir(&options.assets_dir)?;
    let body_id = *system
        .name_to_id
        .get(&options.body)
        .ok_or_else(|| format!("unknown body {:?}", options.body))?;
    let body = &system.bodies[body_id];
    let context = TerrainCompileContext {
        body_name: body.name.clone(),
        radius_m: body.radius_m as f32,
        gravity_m_s2: body.surface_gravity_m_s2() as f32,
        rotation_hours: (body.rotation_period_s > 0.0)
            .then_some((body.rotation_period_s / 3600.0) as f32),
        obliquity_deg: Some(body.axial_tilt_rad.to_degrees() as f32),
        tidal_axis: matches!(body.kind, BodyKind::Moon).then_some(glam::Vec3::Z),
        axial_tilt_rad: body.axial_tilt_rad as f32,
    };
    let key = cache::terrain_cache_key(
        &body.terrain,
        body.tectonics.as_ref(),
        &context,
        TerrainCompileOptions::default(),
    );
    let package_path = options
        .assets_dir
        .join("terrain_packages")
        .join(format!("{}.bin", body.name));
    let loaded = load_static_package(&package_path, &body.name, key)?;
    let cubemap = &loaded.static_surface.height_cubemap;
    let resolution = cubemap.resolution();
    let height_range = loaded.static_surface.height_range;

    // Decode each face into metres and resize to the working canvas.
    let mut macro_faces = Vec::with_capacity(6);
    for face in CubemapFace::ALL {
        let data = cubemap.face_data(face);
        let mut grid = Grid::zeros(resolution as usize);
        for (index, value) in data.iter().enumerate() {
            grid.values[index] = (f32::from(*value) / 65535.0 * 2.0 - 1.0) * height_range;
        }
        macro_faces.push(resize(&grid, options.face_size));
    }

    let coarse_faces: Vec<Grid> = macro_faces
        .iter()
        .map(|face| pyramid::build(face).coarse_for_s3)
        .collect();
    let face_coarse_scale = coarse_faces
        .iter()
        .map(Grid::max_abs)
        .fold(1.0f32, f32::max);
    // The model works in normalized band space; re-dimensionalize its output
    // by the face/train coarse ratio so detail stays proportional relief.
    let residual_gain = target_scale * (face_coarse_scale / coarse_scale);
    let scale_condition = ((SLDEM_METRES_PER_PIXEL / 250.0).log2() / 4.0).clamp(-1.0, 1.0);

    let mut enhanced_faces = Vec::with_capacity(6);
    let mut windows_per_face = 0;
    for (face_index, coarse) in coarse_faces.iter().enumerate() {
        let (residual, windows) = generate_face(
            model,
            config,
            schedule,
            device,
            coarse,
            face_coarse_scale,
            scale_condition,
            face_index,
        );
        windows_per_face = windows;
        let mut enhanced = coarse.clone();
        for (value, generated) in enhanced.values.iter_mut().zip(&residual.values) {
            *value += generated * residual_gain;
        }
        enhanced_faces.push(enhanced);
        println!(
            "face {face_index}: {windows} windows fused ({:.1}s elapsed)",
            started.elapsed().as_secs_f64()
        );
    }

    let out_dir = options
        .out_dir
        .clone()
        .unwrap_or_else(|| run_dir.with_file_name(format!("{}_sphere_preview", config.run.name)));
    std::fs::create_dir_all(&out_dir)?;

    let equirect_width = options.face_size * 4;
    save_equirect(
        &out_dir.join("equirect_macro.png"),
        &macro_faces,
        equirect_width,
    )?;
    save_equirect(
        &out_dir.join("equirect_enhanced.png"),
        &enhanced_faces,
        equirect_width,
    )?;
    render_disc(
        &out_dir.join("moon_macro.png"),
        &macro_faces,
        face_coarse_scale,
        1200,
    )?;
    render_disc(
        &out_dir.join("moon_enhanced.png"),
        &enhanced_faces,
        face_coarse_scale,
        1200,
    )?;

    let manifest = SphereManifest {
        run_name: config.run.name.clone(),
        body: body.name.clone(),
        package_content_key: format!("{:016x}", loaded.manifest.content_key),
        package_resolution: resolution,
        face_size: options.face_size,
        window: config.data.patch_size,
        stride: config.validation.stride,
        sample_steps: config.diffusion.sample_steps.min(schedule.len()),
        windows_per_face,
        train_target_scale_metres: target_scale,
        train_coarse_scale_metres: coarse_scale,
        face_coarse_scale_metres: face_coarse_scale,
        residual_gain,
        scale_condition_metres_per_pixel: SLDEM_METRES_PER_PIXEL,
        controls: [CRATER_DENSITY, MARE_FRACTION, GARDENING, RIM_SHARPNESS],
        elapsed_seconds: started.elapsed().as_secs_f64(),
    };
    std::fs::write(
        out_dir.join("sphere_manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(out_dir)
}

/// Read the training normalization scales recorded in the run checkpoint and
/// refuse a checkpoint trained under a different diffusion contract.
fn checkpoint_scales(
    run_dir: &Path,
    config: &Config,
) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    let state: serde_json::Value =
        serde_json::from_slice(&std::fs::read(run_dir.join("checkpoint.json"))?)?;
    let prediction = state["prediction"].as_str().unwrap_or_default();
    if prediction != config.diffusion.prediction.as_str() {
        return Err(format!(
            "checkpoint prediction {prediction:?} does not match config {:?}",
            config.diffusion.prediction.as_str()
        )
        .into());
    }
    let scale = |name: &str| -> Result<f32, Box<dyn std::error::Error>> {
        state[name]
            .as_f64()
            .map(|value| value as f32)
            .ok_or_else(|| format!("checkpoint.json missing {name}").into())
    };
    Ok((scale("target_scale_metres")?, scale("coarse_scale_metres")?))
}

/// Generate the learned residual band for one face canvas with overlap fusion.
#[allow(clippy::too_many_arguments)]
fn generate_face<B: Backend>(
    model: &AirlessDenoiser<B>,
    config: &Config,
    schedule: &DiffusionSchedule,
    device: &B::Device,
    coarse: &Grid,
    face_coarse_scale: f32,
    scale_condition: f32,
    face_index: usize,
) -> (Grid, usize) {
    let canvas = coarse.size;
    let window = config.data.patch_size;
    let stride = config.validation.stride;
    let window_origins = origins(canvas, window, stride);
    let mut sum = vec![0.0f32; canvas * canvas];
    let mut weights = vec![0.0f32; canvas * canvas];
    let face_seed = config.run.seed ^ (face_index as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);

    for &origin_y in &window_origins {
        for &origin_x in &window_origins {
            let prediction = sample_face_window(
                model,
                config,
                schedule,
                device,
                coarse,
                face_coarse_scale,
                scale_condition,
                face_seed,
                origin_x,
                origin_y,
            );
            for y in 0..window {
                for x in 0..window {
                    let index = (origin_y + y) * canvas + (origin_x + x);
                    let weight = blend_weight(x, y, window);
                    sum[index] += prediction[y * window + x] * weight;
                    weights[index] += weight;
                }
            }
        }
    }
    let mut residual = Grid::zeros(canvas);
    for index in 0..canvas * canvas {
        residual.values[index] = sum[index] / weights[index].max(1e-8);
    }
    (residual, window_origins.len() * window_origins.len())
}

/// One DDIM window in normalized band space (returns the unscaled state).
#[allow(clippy::too_many_arguments)]
fn sample_face_window<B: Backend>(
    model: &AirlessDenoiser<B>,
    config: &Config,
    schedule: &DiffusionSchedule,
    device: &B::Device,
    coarse: &Grid,
    face_coarse_scale: f32,
    scale_condition: f32,
    face_seed: u64,
    origin_x: usize,
    origin_y: usize,
) -> Vec<f32> {
    let size = config.data.patch_size;
    let area = size * size;
    let canvas = coarse.size;
    let mut state = vec![0.0f32; area];
    for y in 0..size {
        for x in 0..size {
            state[y * size + x] = coordinate_normal(face_seed, origin_x + x, origin_y + y);
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
                let coarse_normalized = coarse.get(global_x, global_y) / face_coarse_scale;
                set(&mut input, 0, pixel, area, state[pixel]);
                set(&mut input, 1, pixel, area, coarse_normalized);
                set(&mut input, 2, pixel, area, CRATER_DENSITY / 40.0);
                set(&mut input, 3, pixel, area, MARE_FRACTION);
                set(&mut input, 4, pixel, area, GARDENING);
                set(&mut input, 5, pixel, area, RIM_SHARPNESS / 2.0);
                set(&mut input, 6, pixel, area, mare_proxy(coarse_normalized));
                set(&mut input, 7, pixel, area, scale_condition);
                set(
                    &mut input,
                    8,
                    pixel,
                    area,
                    global_x as f32 / (canvas - 1) as f32 * 2.0 - 1.0,
                );
                set(
                    &mut input,
                    9,
                    pixel,
                    area,
                    global_y as f32 / (canvas - 1) as f32 * 2.0 - 1.0,
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
            .expect("f32 model output");
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
        *value = value.clamp(-2.5, 2.5);
    }
    state
}

/// Smooth low-elevation mare proxy of the normalized coarse field.
fn mare_proxy(coarse_normalized: f32) -> f32 {
    let t = (0.35 - coarse_normalized * 1.4).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn resize(source: &Grid, size: usize) -> Grid {
    if source.size == size {
        return source.clone();
    }
    let mut output = Grid::zeros(size);
    for y in 0..size {
        for x in 0..size {
            let sx = x as f32 / (size - 1) as f32 * (source.size - 1) as f32;
            let sy = y as f32 / (size - 1) as f32 * (source.size - 1) as f32;
            let x0 = sx as usize;
            let y0 = sy as usize;
            let x1 = (x0 + 1).min(source.size - 1);
            let y1 = (y0 + 1).min(source.size - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let top = source.get(x0, y0) * (1.0 - fx) + source.get(x1, y0) * fx;
            let bottom = source.get(x0, y1) * (1.0 - fx) + source.get(x1, y1) * fx;
            output.values[y * size + x] = top * (1.0 - fy) + bottom * fy;
        }
    }
    output
}

/// Sample a set of face grids by unit direction with bilinear face-local taps.
fn sample_faces(faces: &[Grid], dir: glam::Vec3) -> f32 {
    let (face, u, v) = dir_to_face_uv(dir);
    let grid = &faces[face as usize];
    let size = grid.size;
    let sx = (u * (size - 1) as f32).clamp(0.0, (size - 1) as f32);
    let sy = (v * (size - 1) as f32).clamp(0.0, (size - 1) as f32);
    let x0 = sx as usize;
    let y0 = sy as usize;
    let x1 = (x0 + 1).min(size - 1);
    let y1 = (y0 + 1).min(size - 1);
    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;
    let top = grid.get(x0, y0) * (1.0 - fx) + grid.get(x1, y0) * fx;
    let bottom = grid.get(x0, y1) * (1.0 - fx) + grid.get(x1, y1) * fx;
    top * (1.0 - fy) + bottom * fy
}

fn save_equirect(
    path: &Path,
    faces: &[Grid],
    width: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let height = width / 2;
    let mut grid = Grid::zeros(width);
    for y in 0..height {
        let latitude =
            std::f32::consts::FRAC_PI_2 - (y as f32 + 0.5) / height as f32 * std::f32::consts::PI;
        for x in 0..width {
            let longitude =
                (x as f32 + 0.5) / width as f32 * std::f32::consts::TAU - std::f32::consts::PI;
            let dir = glam::Vec3::new(
                latitude.cos() * longitude.cos(),
                latitude.sin(),
                latitude.cos() * longitude.sin(),
            );
            grid.values[y * width + x] = sample_faces(faces, dir);
        }
    }
    output::save_hillshade_region(path, &grid, width, height)?;
    Ok(())
}

/// Orthographic full-disc render with screen-space relief shading.
fn render_disc(
    path: &Path,
    faces: &[Grid],
    height_scale: f32,
    size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let light = glam::Vec3::new(-0.45, 0.35, 0.82).normalize();
    let sample_screen = |sx: f32, sy: f32| -> Option<f32> {
        let radius_squared = sx * sx + sy * sy;
        if radius_squared >= 1.0 {
            return None;
        }
        let dir = glam::Vec3::new(sx, sy, (1.0 - radius_squared).sqrt()).normalize();
        Some(sample_faces(faces, dir))
    };
    let pixel_step = 2.0 / size as f32;
    let mut image = RgbImage::new(size as u32, size as u32);
    for py in 0..size {
        let sy = 1.0 - (py as f32 + 0.5) * pixel_step;
        for px in 0..size {
            let sx = (px as f32 + 0.5) * pixel_step - 1.0;
            let radius = (sx * sx + sy * sy).sqrt();
            if radius >= 1.0 {
                image.put_pixel(px as u32, py as u32, Rgb([4, 4, 6]));
                continue;
            }
            let center = sample_screen(sx, sy).unwrap_or(0.0);
            let step = pixel_step * 1.5;
            let dx = (sample_screen(sx + step, sy).unwrap_or(center)
                - sample_screen(sx - step, sy).unwrap_or(center))
                / (2.0 * step * height_scale);
            let dy = (sample_screen(sx, sy + step).unwrap_or(center)
                - sample_screen(sx, sy - step).unwrap_or(center))
                / (2.0 * step * height_scale);
            let sphere_normal = glam::Vec3::new(sx, sy, (1.0 - radius * radius).sqrt());
            let normal = (sphere_normal
                + glam::Vec3::new(-dx, -dy, 0.0) * DISC_RELIEF_GAIN * sphere_normal.z)
                .normalize();
            let diffuse = normal.dot(light).max(0.0);
            // Height-tinted albedo: low mare plains read darker.
            let albedo = 0.58 + 0.20 * (center / height_scale).clamp(-1.0, 1.0);
            let limb = (1.0 - radius * radius).sqrt().powf(0.35);
            let value = (diffuse * albedo * limb).clamp(0.0, 1.0);
            let tone = (value.powf(1.0 / 2.2) * 255.0) as u8;
            image.put_pixel(
                px as u32,
                py as u32,
                Rgb([tone, tone, (f32::from(tone) * 0.96) as u8]),
            );
        }
    }
    image.save(path)?;
    Ok(())
}
