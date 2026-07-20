use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};
use tiff::decoder::{Decoder, DecodingResult};

#[derive(Clone, Debug)]
struct Raster {
    width: usize,
    height: usize,
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct PrepareOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub source_id: String,
    pub split: String,
    pub expected_sha256: String,
    pub native_metres_per_pixel: f32,
    pub target_metres_per_pixel: f32,
    pub patch_size: usize,
    pub stride: usize,
}

#[derive(Serialize)]
struct PatchRecord {
    file: String,
    source_id: String,
    source_sha256: String,
    split: String,
    origin_xy: [usize; 2],
    patch_size: usize,
    metres_per_pixel: f32,
    valid_fraction: f32,
    mean_height_removed_m: f32,
    impulse_fraction: f32,
    max_laplacian_m: f32,
}

pub fn prepare(options: &PrepareOptions) -> Result<usize, Box<dyn std::error::Error>> {
    let actual_hash = sha256_file(&options.input)?;
    if actual_hash != options.expected_sha256.to_lowercase() {
        return Err(format!(
            "{} SHA-256 mismatch: expected {}, got {}",
            options.input.display(),
            options.expected_sha256,
            actual_hash
        )
        .into());
    }
    if options.patch_size == 0 || options.stride == 0 {
        return Err("patch-size and stride must be positive".into());
    }
    if options.native_metres_per_pixel <= 0.0 || options.target_metres_per_pixel <= 0.0 {
        return Err("pixel scales must be positive".into());
    }

    let source = read_float_tiff(&options.input)?;
    let raster = resample(
        &source,
        options.native_metres_per_pixel,
        options.target_metres_per_pixel,
    );
    if raster.width < options.patch_size || raster.height < options.patch_size {
        return Err(format!(
            "resampled raster {}x{} is smaller than patch size {}",
            raster.width, raster.height, options.patch_size
        )
        .into());
    }

    std::fs::create_dir_all(&options.output)?;
    let mut records = Vec::new();
    let mut preview = None;
    for origin_y in origins(raster.height, options.patch_size, options.stride) {
        for origin_x in origins(raster.width, options.patch_size, options.stride) {
            let Some((values, valid_fraction, mean, impulse_fraction, max_laplacian_m)) =
                extract_patch(&raster, origin_x, origin_y, options.patch_size)
            else {
                continue;
            };
            let file = format!("{}_{origin_x:05}_{origin_y:05}.f32le", options.source_id);
            write_f32le(&options.output.join(&file), &values)?;
            if preview.is_none() {
                preview = Some(values.clone());
            }
            records.push(PatchRecord {
                file,
                source_id: options.source_id.clone(),
                source_sha256: actual_hash.clone(),
                split: options.split.clone(),
                origin_xy: [origin_x, origin_y],
                patch_size: options.patch_size,
                metres_per_pixel: options.target_metres_per_pixel,
                valid_fraction,
                mean_height_removed_m: mean,
                impulse_fraction,
                max_laplacian_m,
            });
        }
    }
    std::fs::write(
        options.output.join("index.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    if let Some(values) = preview {
        crate::output::save_contact_sheet(
            &options.output.join("preview.png"),
            &[(
                "first_patch",
                &crate::grid::Grid {
                    size: options.patch_size,
                    values,
                },
            )],
        )?;
    }
    Ok(records.len())
}

fn read_float_tiff(path: &Path) -> Result<Raster, Box<dyn std::error::Error>> {
    let mut decoder = Decoder::new(BufReader::new(File::open(path)?))?;
    let (width, height) = decoder.dimensions()?;
    let values = match decoder.read_image()? {
        DecodingResult::F32(values) => values,
        other => return Err(format!("expected float32 GeoTIFF, decoded {other:?}").into()),
    };
    Ok(Raster {
        width: width as usize,
        height: height as usize,
        values,
    })
}

fn resample(source: &Raster, native_scale: f32, target_scale: f32) -> Raster {
    let width = ((source.width as f32 * native_scale / target_scale).round() as usize).max(1);
    let height = ((source.height as f32 * native_scale / target_scale).round() as usize).max(1);
    let mut values = vec![f32::NAN; width * height];
    for y in 0..height {
        for x in 0..width {
            let source_x = (x as f32 + 0.5) * target_scale / native_scale - 0.5;
            let source_y = (y as f32 + 0.5) * target_scale / native_scale - 0.5;
            values[y * width + x] = bilinear(source, source_x, source_y);
        }
    }
    Raster {
        width,
        height,
        values,
    }
}

fn bilinear(source: &Raster, x: f32, y: f32) -> f32 {
    let x0 = x.floor().clamp(0.0, (source.width - 1) as f32) as usize;
    let y0 = y.floor().clamp(0.0, (source.height - 1) as f32) as usize;
    let x1 = (x0 + 1).min(source.width - 1);
    let y1 = (y0 + 1).min(source.height - 1);
    let samples = [
        source.values[y0 * source.width + x0],
        source.values[y0 * source.width + x1],
        source.values[y1 * source.width + x0],
        source.values[y1 * source.width + x1],
    ];
    if samples.iter().any(|value| !is_valid(*value)) {
        return f32::NAN;
    }
    let tx = x - x.floor();
    let ty = y - y.floor();
    let top = samples[0] * (1.0 - tx) + samples[1] * tx;
    let bottom = samples[2] * (1.0 - tx) + samples[3] * tx;
    top * (1.0 - ty) + bottom * ty
}

fn is_valid(value: f32) -> bool {
    value.is_finite() && value > -32_000.0
}

fn extract_patch(
    raster: &Raster,
    origin_x: usize,
    origin_y: usize,
    size: usize,
) -> Option<(Vec<f32>, f32, f32, f32, f32)> {
    let mut values = Vec::with_capacity(size * size);
    let mut sum = 0.0f64;
    let mut valid = 0usize;
    for y in 0..size {
        for x in 0..size {
            let value = raster.values[(origin_y + y) * raster.width + origin_x + x];
            values.push(value);
            if is_valid(value) {
                sum += f64::from(value);
                valid += 1;
            }
        }
    }
    let valid_fraction = valid as f32 / values.len() as f32;
    if valid_fraction < 0.99 {
        return None;
    }
    let mean = (sum / valid as f64) as f32;
    inpaint_and_center(&mut values, size, mean);
    let (impulse_fraction, max_laplacian_m) = impulse_metrics(&values, size);
    Some((
        values,
        valid_fraction,
        mean,
        impulse_fraction,
        max_laplacian_m,
    ))
}

fn impulse_metrics(values: &[f32], size: usize) -> (f32, f32) {
    let mut impulses = 0usize;
    let mut count = 0usize;
    let mut maximum = 0.0f32;
    for y in 1..size - 1 {
        for x in 1..size - 1 {
            let centre = values[y * size + x];
            let neighbour_mean = 0.25
                * (values[y * size + x - 1]
                    + values[y * size + x + 1]
                    + values[(y - 1) * size + x]
                    + values[(y + 1) * size + x]);
            let laplacian = (centre - neighbour_mean).abs();
            maximum = maximum.max(laplacian);
            impulses += usize::from(laplacian > 150.0);
            count += 1;
        }
    }
    (impulses as f32 / count as f32, maximum)
}

fn inpaint_and_center(values: &mut [f32], size: usize, mean: f32) {
    for value in values.iter_mut() {
        *value = if is_valid(*value) {
            *value - mean
        } else {
            f32::NAN
        };
    }
    for _ in 0..32 {
        if values.iter().all(|value| value.is_finite()) {
            return;
        }
        let previous = values.to_vec();
        let mut changed = false;
        for y in 0..size {
            for x in 0..size {
                let index = y * size + x;
                if previous[index].is_finite() {
                    continue;
                }
                let mut sum = 0.0f32;
                let mut count = 0usize;
                for offset_y in -1..=1isize {
                    for offset_x in -1..=1isize {
                        let sample_x = x as isize + offset_x;
                        let sample_y = y as isize + offset_y;
                        if sample_x < 0
                            || sample_y < 0
                            || sample_x >= size as isize
                            || sample_y >= size as isize
                        {
                            continue;
                        }
                        let value = previous[sample_y as usize * size + sample_x as usize];
                        if value.is_finite() {
                            sum += value;
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    values[index] = sum / count as f32;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for value in values {
        if !value.is_finite() {
            *value = 0.0;
        }
    }
}

fn origins(extent: usize, size: usize, stride: usize) -> Vec<usize> {
    let last = extent - size;
    let mut values: Vec<_> = (0..=last).step_by(stride).collect();
    if values.last().copied() != Some(last) {
        values.push(last);
    }
    values
}

fn write_f32le(path: &Path, values: &[f32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut hash = Sha256::new();
    let mut reader = BufReader::new(File::open(path)?);
    std::io::copy(&mut reader, &mut HashWriter(&mut hash))?;
    Ok(format!("{:x}", hash.finalize()))
}

struct HashWriter<'a>(&'a mut Sha256);

impl std::io::Write for HashWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.0.update(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
