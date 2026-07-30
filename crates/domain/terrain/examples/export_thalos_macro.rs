//! NTR-X2d: export Thalos's canonical macro terrain as diffusion-conditioning
//! rasters — **all five channels**.
//!
//! Samples the same `ProceduralSurface` the game renders (body id 2, radius
//! 3,186 km) onto an equirect grid at the producer's coarse cell scale
//! (23.04 km/px at the equator) via [`ConditioningChart`], and writes:
//!
//! - `thalos_macro_elev.f32`    — elevation, metres        (channel 0)
//! - `thalos_macro_temp.f32`    — temperature, °C          (channel 1)
//! - `thalos_macro_tempsd.f32`  — temperature sd, °C × 100 (channel 2, BIO4)
//! - `thalos_macro_precip.f32`  — precipitation, mm/yr     (channel 3, BIO12)
//! - `thalos_macro_precipcv.f32`— precipitation CV         (channel 4, BIO15)
//!
//! Channels 2 and 4 are new. Before NTR-X2d they fell back to the producer's
//! own Perlin, so seasonality — the signal that separates a maritime coast from
//! a continental interior at the same latitude — was uncontrolled.
//!
//! The derivation, the province classification, and the four `finalize`
//! transforms the import path bypasses all live in
//! `thalos_terrain::macro_conditioning`, not here: this example is I/O plus the
//! verification report. `thalos_export.py` in the terrain-diffusion checkout
//! consumes these through `set_custom_conditioning_import`.
//!
//! Convention notes: `dir = (cos lat · cos lon, sin lat, cos lat · sin lon)`
//! (matches `runway::latlon_dir`); raster row 0 = north pole edge, column 0 =
//! longitude 0, row-major little-endian f32 — the layout the Python side and
//! `DiffusionSurface` both assume.
//!
//! Usage: `cargo run -p thalos_terrain --release --example export_thalos_macro
//! [-- out_dir]` (default out_dir: `target/thalos_macro`).

use std::io::Write as _;
use std::path::{Path, PathBuf};

use thalos_terrain::ProceduralSurface;
use thalos_terrain::macro_conditioning::{COARSE_PX_M, ConditioningChart};

const THALOS_RADIUS_M: f32 = 3_186_000.0;
const THALOS_BODY_SEED: u32 = 2;
const SAMPLE_LOD_M: f32 = 23_040.0;

fn write_f32(path: &Path, data: &[f32]) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in data {
        out.write_all(&v.to_le_bytes())?;
    }
    out.flush()
}

/// Area-weighted percentiles of a channel over the cells `mask` selects, so a
/// bad export is visible in the log rather than three steps later in a
/// hillshade.
///
/// Land and ocean are always reported separately: the chart is ~58 % ocean, and
/// a whole-chart percentile is dominated by cells whose climate the producer
/// never details into terrain.
///
/// `weights` are per-cell surface area ([`ConditioningChart::cell_weight`]).
/// Percentiles over an equirect raster **must** be area-weighted — uniform rows
/// in latitude mean a raw sort over-represents polar cells. This exporter
/// originally did not, and reported a 0.301 land fraction against the 0.352
/// `just map` measures from real surface area.
fn percentiles(label: &str, data: &[f32], mask: &[bool], weights: &[f64]) {
    let mut v: Vec<(f32, f64)> = data
        .iter()
        .zip(mask)
        .zip(weights)
        .filter_map(|((&d, &m), &w)| m.then_some((d, w)))
        .collect();
    if v.is_empty() {
        println!("  {label:<14} (no cells)");
        return;
    }
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let total: f64 = v.iter().map(|(_, w)| w).sum();
    let at = |q: f64| {
        let target = total * q;
        let mut acc = 0.0;
        for &(value, w) in &v {
            acc += w;
            if acc >= target {
                return value;
            }
        }
        v[v.len() - 1].0
    };
    println!(
        "  {label:<14} p1 {:>9.1}  p25 {:>9.1}  p50 {:>9.1}  p75 {:>9.1}  p99 {:>9.1}",
        at(0.01),
        at(0.25),
        at(0.50),
        at(0.75),
        at(0.99)
    );
}

fn main() -> std::io::Result<()> {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/thalos_macro"));
    std::fs::create_dir_all(&out_dir)?;

    let width = (std::f64::consts::TAU * f64::from(THALOS_RADIUS_M) / COARSE_PX_M).round() as usize;
    let surface = ProceduralSurface::new(THALOS_RADIUS_M, THALOS_BODY_SEED);

    println!("sampling {width}x{} at {:.0} m/px …", width / 2, COARSE_PX_M);
    let chart = ConditioningChart::build(&surface, width, SAMPLE_LOD_M);

    for (name, data) in [
        ("thalos_macro_elev", &chart.elevation_m),
        ("thalos_macro_temp", &chart.temperature_c),
        ("thalos_macro_tempsd", &chart.temperature_sd_c100),
        ("thalos_macro_precip", &chart.precipitation_mm),
        ("thalos_macro_precipcv", &chart.precipitation_cv),
    ] {
        write_f32(&out_dir.join(format!("{name}.f32")), data)?;
    }

    let land_mask: Vec<bool> = chart.elevation_m.iter().map(|&h| h > 0.0).collect();
    let all_mask = vec![true; chart.elevation_m.len()];
    // Per-cell surface area. Every figure below is area-weighted; a raw cell
    // count over an equirect grid over-represents the poles.
    let weights: Vec<f64> = (0..chart.height)
        .flat_map(|y| std::iter::repeat_n(chart.cell_weight(y), chart.width))
        .collect();

    // Producer's own synthetic prior, whole-map (bench_conditioning.py
    // calibrate): temp p25 5.8 / p50 19.9 / p75 26.2; precip p25 293 / p50 578
    // / p75 1148; temp sd p25 501 / p50 686 / p75 1010; cv p25 28 / p50 48.
    println!("\nchannel distributions — LAND ONLY (compare against the producer's prior):");
    percentiles("elev m", &chart.elevation_m, &land_mask, &weights);
    percentiles("temp C", &chart.temperature_c, &land_mask, &weights);
    percentiles("temp sd x100", &chart.temperature_sd_c100, &land_mask, &weights);
    percentiles("precip mm", &chart.precipitation_mm, &land_mask, &weights);
    percentiles("precip cv", &chart.precipitation_cv, &land_mask, &weights);
    percentiles("relief m", &chart.relief_m, &land_mask, &weights);
    percentiles("coast km", &chart.coast_distance_km, &land_mask, &weights);
    percentiles("rain shadow", &chart.rain_shadow, &land_mask, &weights);

    println!("\nwhole chart (land + ocean), the raster the producer actually reads:");
    percentiles("temp C", &chart.temperature_c, &all_mask, &weights);
    percentiles("precip mm", &chart.precipitation_mm, &all_mask, &weights);

    // A rain shadow covering most of the planet would be wrong; so would one
    // covering none of it. Earth's real orographic deserts are a low-teens
    // percentage of land, so this line is the term's sanity check.
    let land_area: f64 = land_mask
        .iter()
        .zip(&weights)
        .filter_map(|(&m, &w)| m.then_some(w))
        .sum();
    let shadowed: f64 = chart
        .rain_shadow
        .iter()
        .zip(&land_mask)
        .zip(&weights)
        .filter_map(|((&s, &m), &w)| (m && s > 0.25).then_some(w))
        .sum();
    println!(
        "\nrain shadow: {:.1} % of land area shadowed > 0.25",
        shadowed / land_area.max(1e-9) * 100.0
    );

    println!("\nlandform provinces:");
    for (p, frac) in chart.province_fractions() {
        if frac > 0.0 {
            println!("  {:<9} {:>6.2} %", p.name(), frac * 100.0);
        }
    }
    let land = chart.land_fraction();
    println!("\nland fraction {land:.3}  (lore target ~0.35)");

    let meta = format!(
        "{{\"width\":{},\"height\":{},\"px_m_equator\":{COARSE_PX_M},\
         \"radius_m\":{THALOS_RADIUS_M},\"seed\":{THALOS_BODY_SEED},\
         \"land_fraction\":{land:.4},\"channels\":5}}\n",
        chart.width, chart.height
    );
    std::fs::write(out_dir.join("thalos_macro.json"), meta)?;
    println!("\nwrote 5 channels + thalos_macro.json to {}", out_dir.display());
    Ok(())
}
