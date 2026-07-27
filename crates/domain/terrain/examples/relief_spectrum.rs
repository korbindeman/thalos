//! Relief spectrum probe — how much *slope* the height field carries per
//! sampling scale.
//!
//! The question this answers: when a renderer resolves finer and finer samples
//! of the same field, does the ground get more detailed, or just rougher? An
//! fBm whose amplitude halves as its frequency doubles contributes the **same
//! slope at every octave**, so a mesh that resolves ten more octaves than the
//! one before it renders ten times the crumple from identical data — which is
//! how the same terrain reads smooth through a coarse mesher and like beaten
//! foil through a fine one.
//!
//! Walks a straight body-fixed transect at a site, sampling at a range of
//! `lod_m` values (each transect sampled at its own `lod_m` spacing, i.e. what a
//! mesh of that resolution would actually build), and reports RMS/p95 slope and
//! the metre-scale relief band. A field with an honest spectral roll-off shows
//! slope *flattening* as `lod_m` drops; one that keeps climbing is manufacturing
//! roughness the data never had.
//!
//! Usage: `cargo run -p thalos_terrain --release --example relief_spectrum
//! [-- lat_deg lon_deg]` (default: the canonical runway site plains).

use glam::DVec3;
use thalos_terrain::{ProceduralSurface, SurfaceQuery};

const THALOS_RADIUS_M: f32 = 3_186_000.0;
const THALOS_BODY_SEED: u32 = 2;
/// Sampling scales to walk, metres per sample. The low end is what the tile
/// renderer builds at ground range (`SPLIT_FACTOR` 6 → `d / 384`); the high end
/// is udlod's near-ground texel spacing.
const LODS_M: [f32; 9] = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
/// Transect length (m) — long enough to carry the 6 km hill band's shape, short
/// enough to stay on one landform.
const TRANSECT_M: f64 = 2_048.0;

fn latlon_dir(lat_deg: f64, lon_deg: f64) -> DVec3 {
    let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let lat = args
        .next()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(7.6);
    let lon = args
        .next()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(178.0);

    let surface = ProceduralSurface::new(THALOS_RADIUS_M, THALOS_BODY_SEED);
    let radius = f64::from(THALOS_RADIUS_M);
    let site = latlon_dir(lat, lon);
    // Local east/north basis at the site; the transect runs east.
    let north = DVec3::Y;
    let east = north.cross(site).normalize();

    println!("relief spectrum @ lat {lat} lon {lon}, transect {TRANSECT_M} m east");
    println!(
        "{:>8}  {:>7}  {:>9}  {:>9}  {:>9}  {:>10}",
        "lod_m", "samples", "rms_slope", "p95_slope", "rms_deg", "band_p2p_m"
    );

    // Sampling step is FIXED; only the field's `lod_m` band-limit varies. That
    // separates "how densely am I sampling" from "how much detail did the field
    // hand back", which is the question a mesher's LOD choice actually asks.
    const STEP_M: f64 = 0.5;
    for lod_m in LODS_M {
        let step = STEP_M;
        let n = (TRANSECT_M / step).round() as usize + 1;
        let heights: Vec<f64> = (0..n)
            .map(|i| {
                let offset = i as f64 * step;
                let dir = (site * radius + east * offset).normalize();
                f64::from(surface.sample_height_m(dir.as_vec3(), lod_m))
            })
            .collect();

        let mut slopes: Vec<f64> = heights
            .windows(2)
            .map(|w| ((w[1] - w[0]) / step).abs())
            .collect();
        let rms = (slopes.iter().map(|s| s * s).sum::<f64>() / slopes.len() as f64).sqrt();
        slopes.sort_by(|a, b| a.partial_cmp(b).expect("finite slopes"));
        let p95 = slopes[(slopes.len() as f64 * 0.95) as usize];

        // Metre-scale band: the transect minus its own 64 m running mean, i.e.
        // the relief a walking observer sees as "the ground is not flat here".
        let win = ((64.0 / step).round() as usize).max(1);
        let mut band_min = f64::INFINITY;
        let mut band_max = f64::NEG_INFINITY;
        for i in 0..heights.len() {
            let lo = i.saturating_sub(win / 2);
            let hi = (i + win / 2 + 1).min(heights.len());
            let mean = heights[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
            band_min = band_min.min(heights[i] - mean);
            band_max = band_max.max(heights[i] - mean);
        }

        println!(
            "{lod_m:>8.2}  {:>7}  {rms:>9.4}  {p95:>9.4}  {:>9.2}  {:>10.3}",
            n,
            rms.atan().to_degrees(),
            band_max - band_min,
        );
    }

    // Canonical forest band over a patch around the site. `tile_terrain.wgsl`
    // selects its canopy layer — tint, per-tree stipple, normal dimple — from
    // this weight via `smoothstep(0.15, 0.55, forest + 0.20·jitter)`, so how
    // much of an open plain that lands on is a question about the *data*, not
    // about the shader's taste.
    const PATCH_M: f64 = 4_096.0;
    const PATCH_N: usize = 64;
    let mut forest: Vec<f64> = Vec::with_capacity(PATCH_N * PATCH_N);
    for j in 0..PATCH_N {
        for i in 0..PATCH_N {
            let du = (i as f64 / (PATCH_N - 1) as f64 - 0.5) * PATCH_M;
            let dv = (j as f64 / (PATCH_N - 1) as f64 - 0.5) * PATCH_M;
            let dir = (site * radius + east * du + north * dv).normalize();
            let (_, bands) = surface.sample_bands_d(dir, 8.0);
            forest.push(f64::from(bands.canopy));
        }
    }
    forest.sort_by(|a, b| a.partial_cmp(b).expect("finite weights"));
    let pick = |q: f64| forest[((forest.len() - 1) as f64 * q) as usize];
    let mean = forest.iter().sum::<f64>() / forest.len() as f64;
    // The shader's selection curve at the jitter extremes: how much of the
    // patch the canopy layer claims at all, and how much it claims fully.
    let density = |w: f64| ((w - 0.15) / 0.40).clamp(0.0, 1.0);
    let any = forest.iter().filter(|w| density(**w + 0.20) > 0.0).count();
    let full = forest.iter().filter(|w| density(**w - 0.20) >= 1.0).count();
    println!(
        "\ncanonical forest weight over a {PATCH_M:.0} m patch: mean {mean:.3}  \
         p10 {:.3}  p50 {:.3}  p90 {:.3}  max {:.3}",
        pick(0.10),
        pick(0.50),
        pick(0.90),
        pick(1.0),
    );
    println!(
        "shader canopy claim: {:.1} % of the patch gets some canopy, {:.1} % gets it fully",
        any as f64 / forest.len() as f64 * 100.0,
        full as f64 / forest.len() as f64 * 100.0,
    );

    // Canonical linear albedo at the site — the vertex colour both renderers
    // start from, and the anchor for judging a rendered frame's chroma: a lit
    // ground whose blue/green ratio is far above this one is being fed blue by
    // its lighting, not by its paint.
    let (sample, _) = surface.sample_bands_d(site, 8.0);
    let a = sample.albedo_linear;
    println!(
        "\ncanonical albedo (linear): [{:.3}, {:.3}, {:.3}]  B/G {:.2}",
        a.x,
        a.y,
        a.z,
        a.z / a.y.max(1.0e-6),
    );

    // Height vs sampling scale at the site itself. The coarse end is what the
    // *water* authority sees: the coast/bathymetry cube bakes `sample_height_m`
    // at one texel arc (τ·R / 4·res — 4.9 km at 1024²/face on Thalos), and the
    // analytic ocean floods wherever that reads below sea level (0 m).
    println!("\nsite height vs sampling scale (sea level = 0 m)");
    for lod_m in [1.0f32, 32.0, 256.0, 1_024.0, 4_886.0, 23_040.0] {
        let h = surface.sample_height_m(site.as_vec3(), lod_m);
        println!("{lod_m:>10.0} m/sample  →  {h:>9.1} m");
    }
}
