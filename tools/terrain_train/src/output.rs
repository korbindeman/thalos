use std::path::Path;

use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};

use crate::grid::Grid;

pub fn save_height_u16(path: &Path, grid: &Grid) -> Result<(), image::ImageError> {
    let maximum = grid.max_abs().max(1.0);
    let image: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::from_fn(grid.size as u32, grid.size as u32, |x, y| {
            let normalized = grid.get(x as usize, y as usize) / (2.0 * maximum) + 0.5;
            Luma([(normalized.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16])
        });
    image.save(path)
}

pub fn save_contact_sheet(path: &Path, grids: &[(&str, &Grid)]) -> Result<(), image::ImageError> {
    let tile = grids[0].1.size as u32;
    let display_tile = tile.max(256);
    let mut sheet = RgbImage::new(display_tile * grids.len() as u32, display_tile);
    for (index, (_, grid)) in grids.iter().enumerate() {
        let rendered = hillshade(grid);
        let rendered = if display_tile == tile {
            rendered
        } else {
            image::imageops::resize(
                &rendered,
                display_tile,
                display_tile,
                image::imageops::FilterType::CatmullRom,
            )
        };
        for y in 0..display_tile {
            for x in 0..display_tile {
                let value = rendered.get_pixel(x, y)[0];
                sheet.put_pixel(
                    index as u32 * display_tile + x,
                    y,
                    Rgb([value, value, value]),
                );
            }
        }
    }
    sheet.save(path)
}

pub fn save_hillshade(path: &Path, grid: &Grid) -> Result<(), image::ImageError> {
    let rendered = hillshade(grid);
    let display_size = (grid.size as u32).max(256);
    if display_size == grid.size as u32 {
        rendered.save(path)
    } else {
        image::imageops::resize(
            &rendered,
            display_size,
            display_size,
            image::imageops::FilterType::CatmullRom,
        )
        .save(path)
    }
}

pub fn save_hillshade_region(
    path: &Path,
    grid: &Grid,
    width: usize,
    height: usize,
) -> Result<(), image::ImageError> {
    let rendered = hillshade(grid);
    image::imageops::crop_imm(&rendered, 0, 0, width as u32, height as u32)
        .to_image()
        .save(path)
}

fn hillshade(grid: &Grid) -> GrayImage {
    let scale = grid.max_abs().max(1.0) / grid.size as f32;
    ImageBuffer::from_fn(grid.size as u32, grid.size as u32, |x, y| {
        let x = x as usize;
        let y = y as usize;
        let left = grid.get(x.saturating_sub(1), y);
        let right = grid.get((x + 1).min(grid.size - 1), y);
        let up = grid.get(x, y.saturating_sub(1));
        let down = grid.get(x, (y + 1).min(grid.size - 1));
        let nx = -(right - left) / (2.0 * scale);
        let ny = -(down - up) / (2.0 * scale);
        let nz = 1.0;
        let inverse_length = (nx * nx + ny * ny + nz * nz).sqrt().recip();
        let light = (nx * -0.45 + ny * -0.55 + nz * 0.70) * inverse_length;
        Luma([((light * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8])
    })
}
