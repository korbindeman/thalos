//! Where does the terrain's *slope* energy actually live?
//!
//! "The ground looks bumpy" is a statement about slope at some wavelength, and
//! the height cascade has a band at nearly every wavelength — so the useful
//! question is which shell of the cascade dominates the RMS slope at the
//! footprints we actually ship. This probe answers it without a render.
//!
//! Method: sample a ground transect twice at the same points, once at a fine
//! `lod_m` and once at a coarse one. Every band is footprint-gated, so the
//! difference is exactly the content between the two footprints. Reporting RMS
//! amplitude *and* RMS slope per shell separates "this band is tall" from "this
//! band is steep" — bumpiness is the second one.
//!
//! Run:
//!   cargo run --release -p thalos_terrain --example band_roughness_probe
//!   THALOS_TERRAIN=procedural cargo run --release -p thalos_terrain --example band_roughness_probe
//!
//! Output: a table on stdout. Nothing is written to disk.

use glam::DVec3;
use rayon::prelude::*;
use thalos_terrain::query::SurfaceQuery;
use thalos_terrain::{DiffusionSurface, ProceduralSurface};

const RADIUS_M: f64 = 3_186_000.0;
const SEED: u32 = 2;

/// Transect sampling. 2 m steps over ~8 km: fine enough to resolve the finest
/// band any shipped tile carries, long enough that the coarse shells have
/// several cycles to average over.
const STEP_M: f64 = 2.0;
const SAMPLES: usize = 4096;

/// Footprint ladder. Consecutive entries bound one shell; each is roughly a
/// factor of ~2.5 apart so the shells tile the cascade without overlapping
/// much. `footprint_gate` passes a wavelength fully at 4x the footprint and
/// kills it at 2x, so footprint `f` is a low-pass at roughly `3f`.
const FOOTPRINTS_M: [f32; 8] = [1.0, 4.0, 10.0, 25.0, 60.0, 150.0, 400.0, 1000.0];

/// Sites, as body-fixed camera positions copied from `assets/viewpoints.json`.
/// The probe walks the ground directly beneath each.
const SITES: [(&str, [f64; 3]); 3] = [
    (
        "mountain-close",
        [-3_140_582.632, 499_221.213, 263_835.490],
    ),
    ("small-valley", [-3_119_318.589, 645_154.374, 149_554.459]),
    // Roughly 1100 km east of the other two, so it is outside every learned
    // detail window. The fine band answers a different question there — the
    // base stops at the chart's finest 1.2 km octave rather than at a 90 m
    // raster, so the whole 1.2 km-to-90 m range is the band's to invent, and a
    // ladder tuned only on windowed ground would ship untested over most of
    // the planet.
    ("far-field", [-3_043_000.0, 499_000.0, -815_000.0]),
];

fn load_surface() -> (Box<dyn SurfaceQuery>, &'static str) {
    if thalos_terrain::thalos_terrain_prefers_diffusion() {
        let dir = std::path::Path::new("assets/terrain_packages/thalos_diffusion");
        match DiffusionSurface::load(dir, RADIUS_M as f32, 2) {
            Ok(surface) => return (Box::new(surface), "diffusion"),
            Err(error) => println!("diffusion load failed ({error}); falling back"),
        }
    }
    (
        Box::new(ProceduralSurface::new(RADIUS_M as f32, SEED)),
        "procedural",
    )
}

fn tangent_basis(dir: DVec3) -> (DVec3, DVec3) {
    let seed = if dir.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
    let east = seed.cross(dir).normalize();
    (east, dir.cross(east).normalize())
}

/// Heights along the transect at one footprint.
fn transect(surface: &dyn SurfaceQuery, centre: DVec3, east: DVec3, lod_m: f32) -> Vec<f64> {
    (0..SAMPLES)
        .into_par_iter()
        .map(|i| {
            let along = (i as f64 - SAMPLES as f64 * 0.5) * STEP_M;
            let dir = (centre + east * (along / RADIUS_M)).normalize();
            f64::from(surface.sample_height_m(dir.as_vec3(), lod_m))
        })
        .collect()
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

fn main() {
    let (surface, backing) = load_surface();
    println!("backing: {backing}");
    println!(
        "transect: {} samples x {STEP_M} m = {:.1} km\n",
        SAMPLES,
        SAMPLES as f64 * STEP_M / 1000.0
    );

    for (name, position) in SITES {
        let centre = DVec3::from_array(position).normalize();
        let (east, _) = tangent_basis(centre);

        let ladder: Vec<Vec<f64>> = FOOTPRINTS_M
            .iter()
            .map(|lod| transect(surface.as_ref(), centre, east, *lod))
            .collect();

        let ground = ladder[0].iter().sum::<f64>() / SAMPLES as f64;
        println!("=== {name} (mean ground {ground:.0} m) ===");
        println!(
            "{:>18}  {:>10}  {:>10}  {:>9}",
            "shell (footprint)", "amp RMS m", "slope RMS", "slope deg"
        );

        for w in FOOTPRINTS_M.windows(2).enumerate() {
            let (i, pair) = w;
            let (fine, coarse) = (pair[0], pair[1]);
            // Content the fine footprint admits and the coarse one does not.
            let shell: Vec<f64> = ladder[i]
                .iter()
                .zip(&ladder[i + 1])
                .map(|(f, c)| f - c)
                .collect();
            // Slope of that shell at its own finest resolvable lag.
            let lag = (fine as f64 * 2.0 / STEP_M).round().max(1.0) as usize;
            let slopes: Vec<f64> = shell
                .windows(lag + 1)
                .map(|w| (w[lag] - w[0]) / (lag as f64 * STEP_M))
                .collect();
            let slope = rms(&slopes);
            println!(
                "{:>7.0}-{:<10.0}  {:>10.2}  {:>10.4}  {:>9.1}",
                fine,
                coarse,
                rms(&shell),
                slope,
                slope.atan().to_degrees()
            );
        }

        // Total, for scale.
        let base = ladder.last().unwrap();
        let total: Vec<f64> = ladder[0]
            .iter()
            .zip(base)
            .map(|(f, c)| f - c)
            .collect();
        println!("{:>18}  {:>10.2}", "all shells", rms(&total));
        println!();

        base_slope_histogram(surface.as_ref(), centre);
        write_relief(surface.as_ref(), centre, name, backing);
    }
}

/// Distribution of the *base* slope — the quantity a regime-selecting fine
/// band has to key on. Thresholds picked without this are guesses: if the
/// steep tail of a real massif tops out at 0.4 then a "rock above 0.85" rule
/// never fires anywhere on the planet, and the band silently ships one regime.
///
/// Measured at 90 m, the footprint the learned detail raster resolves — the
/// finest scale the fine band is allowed to steer by.
const SLOPE_PROBE_SPAN_M: f64 = 6_000.0;
const SLOPE_PROBE_STEP_M: f64 = 90.0;

fn base_slope_histogram(surface: &dyn SurfaceQuery, centre: DVec3) {
    let (east, north) = tangent_basis(centre);
    let side = (SLOPE_PROBE_SPAN_M / SLOPE_PROBE_STEP_M) as usize;
    let at = |x: f64, y: f64| -> f64 {
        let dir = (centre + east * (x / RADIUS_M) + north * (y / RADIUS_M)).normalize();
        f64::from(surface.sample_height_m(dir.as_vec3(), SLOPE_PROBE_STEP_M as f32))
    };

    let mut slopes: Vec<f64> = (0..side * side)
        .into_par_iter()
        .map(|i| {
            let x = ((i % side) as f64 - side as f64 * 0.5) * SLOPE_PROBE_STEP_M;
            let y = ((i / side) as f64 - side as f64 * 0.5) * SLOPE_PROBE_STEP_M;
            let dx = (at(x + SLOPE_PROBE_STEP_M, y) - at(x - SLOPE_PROBE_STEP_M, y))
                / (2.0 * SLOPE_PROBE_STEP_M);
            let dy = (at(x, y + SLOPE_PROBE_STEP_M) - at(x, y - SLOPE_PROBE_STEP_M))
                / (2.0 * SLOPE_PROBE_STEP_M);
            (dx * dx + dy * dy).sqrt()
        })
        .collect();
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pct = |p: f64| slopes[((slopes.len() - 1) as f64 * p) as usize];
    println!(
        "base slope @{SLOPE_PROBE_STEP_M} m over {:.0} km: \
         p10 {:.2} ({:.0}d)  p50 {:.2} ({:.0}d)  p90 {:.2} ({:.0}d)  p99 {:.2} ({:.0}d)  max {:.2} ({:.0}d)",
        SLOPE_PROBE_SPAN_M / 1000.0,
        pct(0.10),
        pct(0.10).atan().to_degrees(),
        pct(0.50),
        pct(0.50).atan().to_degrees(),
        pct(0.90),
        pct(0.90).atan().to_degrees(),
        pct(0.99),
        pct(0.99).atan().to_degrees(),
        slopes[slopes.len() - 1],
        slopes[slopes.len() - 1].atan().to_degrees(),
    );
}

/// Shaded relief of a patch, so the *shape* of the fine cascade is visible
/// without a renderer in the loop: slope-lit at the same low sun the viewpoint
/// captures use, with the regional trend removed so only the sub-kilometre
/// bands remain.
const RELIEF_PX: usize = 700;
const RELIEF_PX_M: f64 = 2.0;

fn write_relief(surface: &dyn SurfaceQuery, centre: DVec3, name: &str, backing: &str) {
    let (east, north) = tangent_basis(centre);
    let dir_at = |x: f64, y: f64| -> DVec3 {
        (centre + east * (x / RADIUS_M) + north * (y / RADIUS_M)).normalize()
    };
    let offset = RELIEF_PX as f64 * 0.5;

    let heights: Vec<f64> = (0..RELIEF_PX * RELIEF_PX)
        .into_par_iter()
        .map(|i| {
            let x = (i % RELIEF_PX) as f64 - offset;
            let y = (i / RELIEF_PX) as f64 - offset;
            f64::from(surface.sample_height_m(
                dir_at(x * RELIEF_PX_M, y * RELIEF_PX_M).as_vec3(),
                RELIEF_PX_M as f32,
            ))
        })
        .collect();

    // Sun 20 deg above the horizon out of the west, the framing the viewpoints
    // were saved at — grazing light is what makes metre-scale bumps legible.
    let (sun_x, sun_y, sun_z) = {
        let el: f64 = 20f64.to_radians();
        (el.cos(), 0.0, el.sin())
    };
    let mut pixels = vec![0u8; RELIEF_PX * RELIEF_PX * 3];
    for y in 1..RELIEF_PX - 1 {
        for x in 1..RELIEF_PX - 1 {
            let h = |x: usize, y: usize| heights[y * RELIEF_PX + x];
            let dx = (h(x + 1, y) - h(x - 1, y)) / (2.0 * RELIEF_PX_M);
            let dy = (h(x, y + 1) - h(x, y - 1)) / (2.0 * RELIEF_PX_M);
            let inv = 1.0 / (dx * dx + dy * dy + 1.0).sqrt();
            let lambert = ((-dx * sun_x - dy * sun_y + sun_z) * inv).clamp(0.0, 1.0);
            let v = (lambert.powf(0.85) * 255.0) as u8;
            let o = (y * RELIEF_PX + x) * 3;
            pixels[o] = v;
            pixels[o + 1] = v;
            pixels[o + 2] = v;
        }
    }

    let tag = std::env::var("THALOS_TERRAIN_BANDS")
        .unwrap_or_default()
        .replace([',', '-'], "");
    let tag = if tag.is_empty() { "all".to_string() } else { format!("no{tag}") };
    let path = format!("target/relief_{name}_{backing}_{tag}.png");
    image::RgbImage::from_raw(RELIEF_PX as u32, RELIEF_PX as u32, pixels)
        .expect("relief buffer")
        .save(&path)
        .expect("write relief png");
    println!(
        "wrote {path} ({} m across at {RELIEF_PX_M} m/px)\n",
        RELIEF_PX as f64 * RELIEF_PX_M
    );
}
