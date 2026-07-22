use std::f32::consts::TAU;

use crate::{
    config::Data,
    grid::Grid,
    sample::{Parameters, Provenance, Sample, Split},
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub fn generate(data: &Data, seed: u64, split: Split) -> Sample {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let crater_density = range(&mut rng, data.crater_density);
    let mare_fraction = range(&mut rng, data.mare_fraction);
    let gardening = range(&mut rng, data.gardening);
    let rim_sharpness = range(&mut rng, data.rim_sharpness);
    let area_km2 = (data.patch_size as f32 * data.metres_per_pixel / 1000.0).powi(2);
    let crater_count = ((crater_density * area_km2 * 0.035).round() as usize).clamp(3, 160);

    let mut height = Grid::zeros(data.patch_size);
    add_macro_relief(&mut height, &mut rng, mare_fraction);
    let mare_mask = build_mare_mask(data.patch_size, &mut rng, mare_fraction);

    for _ in 0..crater_count {
        let x = rng.random_range(0.0..data.patch_size as f32);
        let y = rng.random_range(0.0..data.patch_size as f32);
        let radius =
            (1.4f32 * (rng.random::<f32>() * 2.2).exp()).min(data.patch_size as f32 * 0.16);
        let age = (gardening + rng.random_range(-0.25..0.25)).clamp(0.0, 1.0);
        stamp_crater(&mut height, x, y, radius, age, rim_sharpness, &mut rng);
    }
    add_secondary_chain(&mut height, &mut rng, gardening, rim_sharpness);

    for _ in 0..(gardening * 4.0).round() as usize {
        height = blur3(&height);
    }
    height.subtract_mean();

    Sample {
        height,
        mare_mask,
        parameters: Parameters {
            crater_density,
            mare_fraction,
            gardening,
            rim_sharpness,
            crater_count,
        },
        provenance: Provenance {
            source_id: format!("synthetic-v1-{seed:016x}"),
            split,
            metres_per_pixel: data.metres_per_pixel,
            synthetic: true,
        },
    }
}

fn range(rng: &mut ChaCha8Rng, range: [f32; 2]) -> f32 {
    rng.random_range(range[0]..=range[1])
}

fn add_macro_relief(height: &mut Grid, rng: &mut ChaCha8Rng, mare_fraction: f32) {
    let size = height.size as f32;
    for _ in 0..5 {
        let angle = rng.random_range(0.0..TAU);
        let frequency = rng.random_range(0.45..2.2);
        let phase = rng.random_range(0.0..TAU);
        let amplitude = rng.random_range(70.0..260.0) * (1.15 - mare_fraction);
        for y in 0..height.size {
            for x in 0..height.size {
                let projected = (x as f32 * angle.cos() + y as f32 * angle.sin()) / size;
                height.add(
                    x,
                    y,
                    amplitude * (TAU * frequency * projected + phase).sin(),
                );
            }
        }
    }
}

fn build_mare_mask(size: usize, rng: &mut ChaCha8Rng, fraction: f32) -> Grid {
    let centre_x = rng.random_range(0.2..0.8) * size as f32;
    let centre_y = rng.random_range(0.2..0.8) * size as f32;
    let radius = size as f32 * (0.22 + fraction * 0.65);
    let mut mask = Grid::zeros(size);
    for y in 0..size {
        for x in 0..size {
            let distance = ((x as f32 - centre_x).powi(2) + (y as f32 - centre_y).powi(2)).sqrt();
            mask.values[y * size + x] =
                (1.0 - (distance - radius * 0.72) / (radius * 0.28)).clamp(0.0, 1.0);
        }
    }
    mask
}

fn stamp_crater(
    height: &mut Grid,
    centre_x: f32,
    centre_y: f32,
    radius: f32,
    age: f32,
    rim_sharpness: f32,
    rng: &mut ChaCha8Rng,
) {
    let freshness = 1.0 - 0.72 * age;
    let depth = radius * rng.random_range(16.0..34.0) * freshness;
    let extent = (radius * 2.8).ceil() as isize;
    for offset_y in -extent..=extent {
        for offset_x in -extent..=extent {
            let x = centre_x.floor() as isize + offset_x;
            let y = centre_y.floor() as isize + offset_y;
            if x < 0 || y < 0 || x >= height.size as isize || y >= height.size as isize {
                continue;
            }
            let dx = x as f32 + 0.5 - centre_x;
            let dy = y as f32 + 0.5 - centre_y;
            let normalized = (dx * dx + dy * dy).sqrt() / radius;
            let bowl = if normalized < 1.0 {
                -depth * (1.0 - normalized * normalized).powi(2)
            } else {
                0.0
            };
            let rim_width = 0.10 + 0.12 / rim_sharpness.max(0.2);
            let rim = depth
                * 0.22
                * rim_sharpness
                * (-(normalized - 1.0).powi(2) / rim_width.powi(2)).exp();
            let ejecta = if (1.0..2.8).contains(&normalized) {
                depth * 0.035 * freshness / normalized.powf(2.4)
            } else {
                0.0
            };
            height.add(x as usize, y as usize, bowl + rim + ejecta);
        }
    }
}

fn add_secondary_chain(
    height: &mut Grid,
    rng: &mut ChaCha8Rng,
    gardening: f32,
    rim_sharpness: f32,
) {
    let angle = rng.random_range(0.0..TAU);
    let start_x = rng.random_range(0.1..0.5) * height.size as f32;
    let start_y = rng.random_range(0.1..0.9) * height.size as f32;
    for index in 0..rng.random_range(3..8) {
        let distance = index as f32 * rng.random_range(2.5..5.5);
        stamp_crater(
            height,
            start_x + angle.cos() * distance,
            start_y + angle.sin() * distance,
            rng.random_range(0.8..2.2),
            gardening,
            rim_sharpness,
            rng,
        );
    }
}

fn blur3(source: &Grid) -> Grid {
    let mut output = Grid::zeros(source.size);
    for y in 0..source.size {
        for x in 0..source.size {
            let mut sum = 0.0;
            let mut weight = 0.0;
            for offset_y in -1..=1 {
                for offset_x in -1..=1 {
                    let sx = (x as isize + offset_x).clamp(0, source.size as isize - 1) as usize;
                    let sy = (y as isize + offset_y).clamp(0, source.size as isize - 1) as usize;
                    let w = if offset_x == 0 && offset_y == 0 {
                        4.0
                    } else if offset_x == 0 || offset_y == 0 {
                        2.0
                    } else {
                        1.0
                    };
                    sum += source.get(sx, sy) * w;
                    weight += w;
                }
            }
            output.values[y * source.size + x] = sum / weight;
        }
    }
    output
}
