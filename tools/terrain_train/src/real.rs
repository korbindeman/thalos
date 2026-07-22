use std::path::Path;

use serde::Deserialize;

use crate::{
    config::Data,
    grid::Grid,
    sample::{Parameters, Provenance, Sample, Split},
};

#[derive(Deserialize)]
struct PatchRecord {
    file: String,
    source_id: String,
    split: Split,
    patch_size: usize,
    metres_per_pixel: f32,
}

pub fn load(data: &Data) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    for source in &data.real_sources {
        let records: Vec<PatchRecord> = serde_json::from_slice(&std::fs::read(&source.index)?)?;
        let directory = source
            .index
            .parent()
            .ok_or("real-data index has no parent directory")?;
        for record in records.into_iter().take(source.limit.unwrap_or(usize::MAX)) {
            if record.split == Split::Holdout {
                continue;
            }
            let values = read_f32le(&directory.join(&record.file), record.patch_size)?;
            let mut height = resize_square(&values, record.patch_size, data.patch_size);
            height.subtract_mean();
            let mut mare_mask = Grid::zeros(data.patch_size);
            mare_mask.values.fill(source.mare_fraction);
            samples.push(Sample {
                height,
                mare_mask,
                parameters: Parameters {
                    crater_density: source.crater_density,
                    mare_fraction: source.mare_fraction,
                    gardening: source.gardening,
                    rim_sharpness: source.rim_sharpness,
                    crater_count: 0,
                },
                provenance: Provenance {
                    source_id: record.source_id,
                    split: record.split,
                    metres_per_pixel: record.metres_per_pixel * record.patch_size as f32
                        / data.patch_size as f32,
                    synthetic: false,
                },
            });
        }
    }
    Ok(samples)
}

fn read_f32le(path: &Path, size: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != size * size * 4 {
        return Err(format!(
            "{} contains {} bytes; expected {} for a {size}x{size} f32 patch",
            path.display(),
            bytes.len(),
            size * size * 4
        )
        .into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn resize_square(values: &[f32], source_size: usize, target_size: usize) -> Grid {
    if source_size == target_size {
        return Grid {
            size: source_size,
            values: values.to_vec(),
        };
    }
    let mut output = Grid::zeros(target_size);
    for y in 0..target_size {
        for x in 0..target_size {
            let source_x = (x as f32 + 0.5) * source_size as f32 / target_size as f32 - 0.5;
            let source_y = (y as f32 + 0.5) * source_size as f32 / target_size as f32 - 0.5;
            let x0 = source_x.floor().clamp(0.0, (source_size - 1) as f32) as usize;
            let y0 = source_y.floor().clamp(0.0, (source_size - 1) as f32) as usize;
            let x1 = (x0 + 1).min(source_size - 1);
            let y1 = (y0 + 1).min(source_size - 1);
            let tx = source_x - source_x.floor();
            let ty = source_y - source_y.floor();
            let top =
                values[y0 * source_size + x0] * (1.0 - tx) + values[y0 * source_size + x1] * tx;
            let bottom =
                values[y1 * source_size + x0] * (1.0 - tx) + values[y1 * source_size + x1] * tx;
            output.values[y * target_size + x] = top * (1.0 - ty) + bottom * ty;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::resize_square;

    #[test]
    fn resize_preserves_a_constant_height() {
        let resized = resize_square(&vec![37.0; 16 * 16], 16, 32);
        assert!(resized.values.iter().all(|height| *height == 37.0));
    }
}
