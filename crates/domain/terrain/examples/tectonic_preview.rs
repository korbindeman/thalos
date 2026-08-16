//! Global tectonic atlas for the canonical Thalos macro field (NTR-X2f).
//!
//! Produces two matched equirectangular maps:
//!
//! - `target/tectonic_regimes.png`: convergent (orange), divergent (cyan), and
//!   transform (magenta) margin influence over authored land/ocean.
//! - `target/tectonic_relief.png`: shaded canonical relief at the same framing.
//! - `target/tectonic_provinces.png`: broad hinterland (ochre), foreland
//!   (green), and ocean-ridge swell (cyan) response fields.
//!
//! The first image is the causal field and the second is its terrain result.
//! Keeping both is the regression surface for "ridges exist because plates met",
//! rather than because a visually similar noise mask happened to be tuned.
//!
//! Run: `cargo run -p thalos_terrain --release --example tectonic_preview`

use glam::DVec3;
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use thalos_terrain::{ProceduralSurface, SurfaceQuery, TectonicSignals};

const WIDTH: usize = 1_440;
const HEIGHT: usize = WIDTH / 2;
const RADIUS_M: f64 = 3_186_000.0;
const SEED: u32 = 2;
const LOD_M: f32 = 14_000.0;
const RIDGE_SIZE: usize = 800;
const RIDGE_PX_M: f64 = 1_000.0;

fn direction(x: usize, y: usize) -> DVec3 {
    let lon = (x as f64 + 0.5) / WIDTH as f64 * std::f64::consts::TAU - std::f64::consts::PI;
    let lat = std::f64::consts::FRAC_PI_2 - (y as f64 + 0.5) / HEIGHT as f64 * std::f64::consts::PI;
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin())
}

fn to_u8(linear: f64) -> u8 {
    (linear.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8
}

fn mix(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t.clamp(0.0, 1.0))
}

fn tangent_basis(dir: DVec3) -> (DVec3, DVec3) {
    let reference = if dir.y.abs() < 0.9 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let east = reference.cross(dir).normalize();
    let north = dir.cross(east).normalize();
    (east, north)
}

fn local_direction(site: DVec3, east: DVec3, north: DVec3, x: usize, y: usize) -> DVec3 {
    let half = RIDGE_SIZE as f64 * 0.5;
    let du = (x as f64 + 0.5 - half) * RIDGE_PX_M;
    let dv = (half - y as f64 - 0.5) * RIDGE_PX_M;
    (site + (east * du + north * dv) / RADIUS_M).normalize()
}

fn render_ridge_closeup(surface: &ProceduralSurface, site: DVec3, target: &std::path::Path) {
    let (east, north) = tangent_basis(site);
    let samples: Vec<(f64, f64, TectonicSignals)> = (0..RIDGE_SIZE * RIDGE_SIZE)
        .into_par_iter()
        .map(|index| {
            let x = index % RIDGE_SIZE;
            let y = index / RIDGE_SIZE;
            let dir = local_direction(site, east, north, x, y);
            (
                surface.macro_signed_height_m(dir),
                f64::from(surface.sample_height_m(dir.as_vec3(), RIDGE_PX_M as f32)),
                surface.tectonic_signals(dir),
            )
        })
        .collect();
    let mut image = RgbImage::new((RIDGE_SIZE * 2) as u32, RIDGE_SIZE as u32);

    for y in 0..RIDGE_SIZE {
        for x in 0..RIDGE_SIZE {
            let index = y * RIDGE_SIZE + x;
            let (macro_h, height, tectonics) = samples[index];
            let left = y * RIDGE_SIZE + x.saturating_sub(1);
            let right = y * RIDGE_SIZE + (x + 1).min(RIDGE_SIZE - 1);
            let up = y.saturating_sub(1) * RIDGE_SIZE + x;
            let down = (y + 1).min(RIDGE_SIZE - 1) * RIDGE_SIZE + x;

            let mut regime = if macro_h > 0.0 {
                [0.12, 0.105, 0.075]
            } else {
                [0.025, 0.055, 0.095]
            };
            for (weight, color) in [
                (tectonics.convergence, [0.95, 0.19, 0.07]),
                (tectonics.divergence, [0.03, 0.62, 0.92]),
                (tectonics.transform, [0.72, 0.12, 0.88]),
            ] {
                regime = mix(regime, color, weight);
            }
            image.put_pixel(x as u32, y as u32, Rgb(regime.map(to_u8)));

            let dh_dx = (samples[right].1 - samples[left].1) / (2.0 * RIDGE_PX_M);
            let dh_dy = (samples[down].1 - samples[up].1) / (2.0 * RIDGE_PX_M);
            let normal = DVec3::new(-dh_dx, -dh_dy, 1.0).normalize();
            let sun = DVec3::new(-0.62, -0.48, 0.62).normalize();
            let shade = 0.24 + 0.76 * normal.dot(sun).max(0.0);
            let base = if height <= 0.0 {
                mix(
                    [0.08, 0.30, 0.48],
                    [0.012, 0.045, 0.14],
                    (-height / 900.0).clamp(0.0, 1.0),
                )
            } else if height < 700.0 {
                mix([0.16, 0.24, 0.09], [0.26, 0.23, 0.13], height / 700.0)
            } else if height < 2_600.0 {
                mix(
                    [0.26, 0.23, 0.13],
                    [0.36, 0.34, 0.31],
                    (height - 700.0) / 1_900.0,
                )
            } else {
                mix(
                    [0.36, 0.34, 0.31],
                    [0.82, 0.84, 0.87],
                    (height - 2_600.0) / 1_400.0,
                )
            };
            image.put_pixel(
                (RIDGE_SIZE + x) as u32,
                y as u32,
                Rgb(base.map(|channel| to_u8(channel * shade))),
            );
        }
    }

    image.save(target).expect("save ridge close-up");
}

fn main() {
    let surface = ProceduralSurface::new(RADIUS_M as f32, SEED);
    let samples: Vec<(f64, f64, TectonicSignals)> = (0..WIDTH * HEIGHT)
        .into_par_iter()
        .map(|index| {
            let x = index % WIDTH;
            let y = index / WIDTH;
            let dir = direction(x, y);
            (
                surface.macro_signed_height_m(dir),
                f64::from(surface.sample_height_m(dir.as_vec3(), LOD_M)),
                surface.tectonic_signals(dir),
            )
        })
        .collect();

    let mut regime_image = RgbImage::new(WIDTH as u32, HEIGHT as u32);
    let mut province_image = RgbImage::new(WIDTH as u32, HEIGHT as u32);
    let mut relief_image = RgbImage::new(WIDTH as u32, HEIGHT as u32);
    let mut weighted_regimes = [0.0_f64; 3];
    let mut weighted_active = 0.0_f64;
    let mut weighted_orogeny = 0.0_f64;
    let mut weighted_provinces = [0.0_f64; 3];
    let mut total_weight = 0.0_f64;
    let mut land_weight = 0.0_f64;
    let mut relief_population_weight = [0.0_f64; 2];
    let mut relief_population_slope = [0.0_f64; 2];

    for y in 0..HEIGHT {
        let lat =
            std::f64::consts::FRAC_PI_2 - (y as f64 + 0.5) / HEIGHT as f64 * std::f64::consts::PI;
        let area_weight = lat.cos();
        let ground_dx = (std::f64::consts::TAU * RADIUS_M * lat.cos() / WIDTH as f64).max(1.0);
        let ground_dy = std::f64::consts::PI * RADIUS_M / HEIGHT as f64;

        for x in 0..WIDTH {
            let index = y * WIDTH + x;
            let (macro_h, height, tectonics) = samples[index];
            total_weight += area_weight;
            weighted_regimes[0] += area_weight * tectonics.convergence;
            weighted_regimes[1] += area_weight * tectonics.divergence;
            weighted_regimes[2] += area_weight * tectonics.transform;
            weighted_active += area_weight * tectonics.activity;
            weighted_orogeny += area_weight * tectonics.orogeny;
            weighted_provinces[0] += area_weight * tectonics.hinterland;
            weighted_provinces[1] += area_weight * tectonics.foreland;
            weighted_provinces[2] += area_weight * tectonics.ridge_swell;

            let left = y * WIDTH + (x + WIDTH - 1) % WIDTH;
            let right = y * WIDTH + (x + 1) % WIDTH;
            let up = y.saturating_sub(1) * WIDTH + x;
            let down = (y + 1).min(HEIGHT - 1) * WIDTH + x;
            let coast = [left, right, up, down].into_iter().any(|neighbor| {
                samples[neighbor].0.is_sign_positive() != macro_h.is_sign_positive()
            });

            let mut regime = if macro_h > 0.0 {
                [0.12, 0.105, 0.075]
            } else {
                [0.025, 0.055, 0.095]
            };
            let influences = [
                (tectonics.convergence, [0.95, 0.19, 0.07]),
                (tectonics.divergence, [0.03, 0.62, 0.92]),
                (tectonics.transform, [0.72, 0.12, 0.88]),
            ];
            for (weight, color) in influences {
                regime = mix(regime, color, weight);
            }
            if coast {
                regime = mix(regime, [0.82, 0.84, 0.78], 0.72);
            }
            regime_image.put_pixel(x as u32, y as u32, Rgb(regime.map(to_u8)));

            let mut province = if macro_h > 0.0 {
                [0.16, 0.15, 0.12]
            } else {
                [0.025, 0.055, 0.095]
            };
            province = mix(province, [0.82, 0.47, 0.10], tectonics.hinterland);
            province = mix(province, [0.08, 0.48, 0.25], tectonics.foreland);
            province = mix(province, [0.03, 0.58, 0.92], tectonics.ridge_swell);
            if coast {
                province = mix(province, [0.82, 0.84, 0.78], 0.72);
            }
            province_image.put_pixel(x as u32, y as u32, Rgb(province.map(to_u8)));

            let dh_dx = (samples[right].1 - samples[left].1) / (2.0 * ground_dx);
            let dh_dy = (samples[down].1 - samples[up].1) / (2.0 * ground_dy);
            let slope = dh_dx.hypot(dh_dy);
            if macro_h > 100.0 {
                land_weight += area_weight;
                let population = if tectonics.orogeny >= 0.08 || tectonics.hinterland >= 0.08 {
                    Some(0)
                } else if tectonics.boundary_distance_m >= 650_000.0
                    && tectonics.orogeny < 0.03
                    && tectonics.hinterland < 0.03
                    && tectonics.foreland < 0.03
                {
                    Some(1)
                } else {
                    None
                };
                if let Some(population) = population {
                    relief_population_weight[population] += area_weight;
                    relief_population_slope[population] += area_weight * slope;
                }
            }
            let normal = DVec3::new(-dh_dx, -dh_dy, 1.0).normalize();
            let sun = DVec3::new(-0.62, -0.48, 0.62).normalize();
            let shade = 0.28 + 0.72 * normal.dot(sun).max(0.0);
            let base = if height <= 0.0 {
                let shallow = (-height / 900.0).clamp(0.0, 1.0);
                mix([0.08, 0.30, 0.48], [0.012, 0.045, 0.14], shallow)
            } else if height < 700.0 {
                mix([0.16, 0.24, 0.09], [0.26, 0.23, 0.13], height / 700.0)
            } else if height < 2_600.0 {
                mix(
                    [0.26, 0.23, 0.13],
                    [0.36, 0.34, 0.31],
                    (height - 700.0) / 1_900.0,
                )
            } else {
                mix(
                    [0.36, 0.34, 0.31],
                    [0.82, 0.84, 0.87],
                    (height - 2_600.0) / 1_400.0,
                )
            };
            relief_image.put_pixel(
                x as u32,
                y as u32,
                Rgb(base.map(|channel| to_u8(channel * shade))),
            );
        }
    }

    let target = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../target");
    let regimes_path = target.join("tectonic_regimes.png");
    let provinces_path = target.join("tectonic_provinces.png");
    let relief_path = target.join("tectonic_relief.png");
    let ridge_path = target.join("tectonic_ridge.png");
    regime_image.save(&regimes_path).expect("save regime atlas");
    province_image
        .save(&provinces_path)
        .expect("save province atlas");
    relief_image.save(&relief_path).expect("save relief atlas");

    let ridge_index = samples
        .iter()
        .enumerate()
        .filter(|(_, (macro_h, _, _))| *macro_h > 100.0)
        .max_by(|(_, a), (_, b)| a.2.orogeny.total_cmp(&b.2.orogeny))
        .map(|(index, _)| index)
        .expect("Thalos has tectonic land");
    let ridge_site = direction(ridge_index % WIDTH, ridge_index / WIDTH);
    render_ridge_closeup(&surface, ridge_site, &ridge_path);

    println!("area-weighted mean boundary influence:");
    println!("  convergent {:.3}", weighted_regimes[0] / total_weight);
    println!("  divergent  {:.3}", weighted_regimes[1] / total_weight);
    println!("  transform  {:.3}", weighted_regimes[2] / total_weight);
    println!("  activity   {:.3}", weighted_active / total_weight);
    println!("  orogeny    {:.3}", weighted_orogeny / total_weight);
    println!("  hinterland {:.3}", weighted_provinces[0] / total_weight);
    println!("  foreland   {:.3}", weighted_provinces[1] / total_weight);
    println!("  ridge swell {:.3}", weighted_provinces[2] / total_weight);
    let belt_slope = relief_population_slope[0] / relief_population_weight[0].max(1.0e-9);
    let quiet_slope = relief_population_slope[1] / relief_population_weight[1].max(1.0e-9);
    println!("macro relief populations at {:.0} km/px:", LOD_M / 1_000.0);
    println!(
        "  tectonic belt {:>5.1}% land  mean slope {:>7.4}",
        relief_population_weight[0] / land_weight.max(1.0e-9) * 100.0,
        belt_slope
    );
    println!(
        "  quiet interior {:>5.1}% land  mean slope {:>7.4}",
        relief_population_weight[1] / land_weight.max(1.0e-9) * 100.0,
        quiet_slope
    );
    println!(
        "  belt/interior slope contrast {:.2}x",
        belt_slope / quiet_slope.max(1.0e-9)
    );
    println!(
        "ridge close-up center: lat {:+.3} lon {:+.3}",
        ridge_site.y.asin().to_degrees(),
        ridge_site.z.atan2(ridge_site.x).to_degrees()
    );
    println!("wrote {}", regimes_path.display());
    println!("wrote {}", provinces_path.display());
    println!("wrote {}", relief_path.display());
    println!("wrote {}", ridge_path.display());
}
