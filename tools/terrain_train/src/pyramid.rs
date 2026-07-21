use crate::grid::Grid;

pub struct LaplacianPyramid {
    /// Coarse S0 followed by signed S1-S3 residuals, all at full patch size.
    pub full_resolution_bands: [Grid; 4],
    pub coarse_for_s3: Grid,
}

pub fn build(source: &Grid) -> LaplacianPyramid {
    let l1 = downsample(source);
    let l2 = downsample(&l1);
    let l3 = downsample(&l2);

    let s0 = resize_to(&l3, source.size);
    let s1 = subtract(&resize_to(&l2, source.size), &s0);
    let s2_base = resize_to(&l1, source.size);
    let s2 = subtract(&s2_base, &resize_to(&l2, source.size));
    let s3 = subtract(source, &s2_base);

    LaplacianPyramid {
        full_resolution_bands: [s0, s1, s2, s3],
        coarse_for_s3: resize_to(&l1, source.size),
    }
}

fn downsample(source: &Grid) -> Grid {
    let size = source.size / 2;
    let mut output = Grid::zeros(size);
    for y in 0..size {
        for x in 0..size {
            let sx = x * 2;
            let sy = y * 2;
            output.values[y * size + x] = 0.25
                * (source.get(sx, sy)
                    + source.get(sx + 1, sy)
                    + source.get(sx, sy + 1)
                    + source.get(sx + 1, sy + 1));
        }
    }
    output
}

fn resize_to(source: &Grid, size: usize) -> Grid {
    let mut current = source.clone();
    while current.size < size {
        current = upsample_2x(&current);
    }
    current
}

fn upsample_2x(source: &Grid) -> Grid {
    let size = source.size * 2;
    let mut output = Grid::zeros(size);
    for y in 0..size {
        for x in 0..size {
            let fx = x as f32 * 0.5;
            let fy = y as f32 * 0.5;
            let x0 = fx.floor() as usize;
            let y0 = fy.floor() as usize;
            let x1 = (x0 + 1).min(source.size - 1);
            let y1 = (y0 + 1).min(source.size - 1);
            let tx = fx - x0 as f32;
            let ty = fy - y0 as f32;
            let a = source.get(x0, y0) * (1.0 - tx) + source.get(x1, y0) * tx;
            let b = source.get(x0, y1) * (1.0 - tx) + source.get(x1, y1) * tx;
            output.values[y * size + x] = a * (1.0 - ty) + b * ty;
        }
    }
    output
}

fn subtract(left: &Grid, right: &Grid) -> Grid {
    let mut output = Grid::zeros(left.size);
    for (output, (left, right)) in output
        .values
        .iter_mut()
        .zip(left.values.iter().zip(&right.values))
    {
        *output = left - right;
    }
    output
}
