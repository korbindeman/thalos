//! Quick sanity probe for `ProceduralSurface` (Slice 0).
//!
//! Not a test (terrain-gen tests are disabled during iteration — see CLAUDE.md);
//! a throwaway example: samples the generator over a direction grid at a couple
//! of LODs and prints min/max/mean height, NaN count, and a few albedo values,
//! so we can confirm the field is sane before wiring visual verification.
//!
//! Run: `cargo run -p thalos_terrain --example procedural_probe`

use glam::DVec3;
use thalos_terrain::ProceduralSurface;
use thalos_terrain::query::SurfaceQuery;

fn main() {
    let radius_m = 3_186_000.0_f32; // Thalos
    let surface = ProceduralSurface::new(radius_m, 1);
    println!(
        "radius_m = {:.0}, height_range_m = ±{:.0}",
        surface.radius_m(),
        surface.height_range_m()
    );

    for &lod_m in &[0.5_f32, 50.0, 5_000.0] {
        let n = 64usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut nan = 0usize;
        let mut count = 0usize;
        // Fibonacci-sphere direction grid for roughly uniform coverage.
        let total = n * n;
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        for i in 0..total {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / total as f64;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * i as f64;
            let dir = DVec3::new(theta.cos() * r, y, theta.sin() * r);
            let h = surface.sample_height_m(dir.as_vec3(), lod_m);
            if h.is_nan() {
                nan += 1;
                continue;
            }
            min = min.min(h);
            max = max.max(h);
            sum += h as f64;
            count += 1;
        }
        println!(
            "lod {:>8.1} m: min {:>9.1}  max {:>9.1}  mean {:>9.1}  nan {}/{}",
            lod_m,
            min,
            max,
            sum / count.max(1) as f64,
            nan,
            total
        );
    }

    // A few full samples (height + albedo) at fixed directions.
    for dir in [DVec3::X, DVec3::Y, DVec3::new(0.3, 0.6, 0.7).normalize()] {
        let s = surface.sample_d(dir, 1.0);
        println!(
            "dir {:+.2?}: h {:>9.1} m  albedo_lin ({:.3}, {:.3}, {:.3})  rough {:.2}",
            dir.to_array(),
            s.height_m,
            s.albedo_linear.x,
            s.albedo_linear.y,
            s.albedo_linear.z,
            s.roughness
        );
    }
}
