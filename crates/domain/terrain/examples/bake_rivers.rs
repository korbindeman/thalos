//! NTR-X2q(a) — bake a global **river channel** by flow accumulation.
//!
//! Thalos's surface is runtime-analytic and `SurfaceQuery` is a per-point black
//! box with no neighbourhood, so drainage cannot be computed in the field: flow
//! accumulation is inherently a global, ordered, downhill traversal. This bakes
//! it once, offline, into a raster the landcover authority samples.
//!
//! # Why this is a landcover channel and not water
//!
//! The analytic ocean is one sphere at r=R and draws exactly one water level
//! (ADR-20260720T185954Z), so a river at 300 m cannot be *rendered* as water.
//! What it can be is **wetness**: a term that darkens and greens the valley
//! floor and pulls vegetation toward it. From the air that reads as a river
//! system; on the ground it reads as a wet, green valley bottom. Actual water
//! surfaces at arbitrary altitude are step (c) of NTR-X2q and a renderer
//! decision.
//!
//! # Equirect is not a grid
//!
//! Two corrections that a naive DEM flow accumulation gets wrong on a sphere,
//! both of which put rivers in visibly wrong places:
//!
//! * **Cell aspect.** An equirect cell is `Δlon·R·cos(lat)` wide and `Δlat·R`
//!   tall, so at 60° it is half as wide as it is tall. Steepest descent must
//!   compare *gradients* (rise over real run), not height differences, or every
//!   river above the tropics prefers to run east-west.
//! * **Cell area.** Contribution to accumulated catchment must be weighted by
//!   `cos(lat)`, or polar catchments are massively overstated.
//!
//! Longitude wraps; latitude clamps at the poles.
//!
//! Run (honours `THALOS_TERRAIN`, and records which backing it baked from —
//! rivers follow the terrain they were baked on and will run uphill on the
//! other one):
//!
//! ```text
//! cargo run -p thalos_terrain --release --example bake_rivers [px_m]
//! THALOS_TERRAIN=diffusion cargo run -p thalos_terrain --release --example bake_rivers
//! ```

use glam::DVec3;
use rayon::prelude::*;
use thalos_terrain::hydrology::{
    HydrologyConfig, NO_RECEIVER, annual_runoff_mm, solve_equirectangular,
};
use thalos_terrain::query::SurfaceQuery;
use thalos_terrain::{DiffusionSurface, ProceduralSurface};

const RADIUS_M: f64 = 3_186_000.0;
const BODY_SEED: u32 = 2;
/// Default ground scale. 2 km resolves continental trunk systems and their
/// major tributaries — the aerial read — without a 200 Mpx bake.
const DEFAULT_PX_M: f64 = 2_000.0;
/// Gradient added per cell when filling a depression — see the fill step.
const FILL_EPSILON_M: f32 = 1.0e-3;
/// Decades of catchment the shipped u8 spans (1 km2 .. 10^7 km2).
const LOG_DECADES: f32 = 7.0;
/// Encoded annual-mean discharge range, in log10(m³/s). Byte zero remains the
/// explicit "no runoff" value; bytes 1..=255 span this range.
const DISCHARGE_LOG_MIN: f32 = -3.0;
const DISCHARGE_LOG_MAX: f32 = 6.0;
/// Catchment above which a cell counts as a channel for the structure stats.
const CHANNEL_HEAD_KM2: f32 = 1_000.0;

fn encode_log_discharge(discharge_m3_s: f32) -> u8 {
    if discharge_m3_s <= 10.0f32.powf(DISCHARGE_LOG_MIN) {
        return 0;
    }
    let t = ((discharge_m3_s.log10() - DISCHARGE_LOG_MIN)
        / (DISCHARGE_LOG_MAX - DISCHARGE_LOG_MIN))
        .clamp(0.0, 1.0);
    1 + (t * 254.0).round() as u8
}

fn load_surface() -> (Box<dyn SurfaceQuery>, &'static str) {
    let diffusion = std::env::var("THALOS_TERRAIN")
        .map(|v| v.trim().eq_ignore_ascii_case("diffusion"))
        .unwrap_or(false);
    if diffusion {
        let dir = std::path::Path::new("assets/terrain_packages/thalos_diffusion");
        match DiffusionSurface::load(dir, RADIUS_M as f32, BODY_SEED) {
            Ok(s) => return (Box::new(s), "diffusion"),
            Err(e) => println!("diffusion load failed ({e}); falling back to procedural"),
        }
    }
    (
        Box::new(ProceduralSurface::new(RADIUS_M as f32, BODY_SEED)),
        "procedural",
    )
}

fn dir_of(x: usize, y: usize, w: usize, h: usize) -> DVec3 {
    let lon = (x as f64 + 0.5) / w as f64 * std::f64::consts::TAU;
    let lat = (0.5 - (y as f64 + 0.5) / h as f64) * std::f64::consts::PI;
    DVec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin())
}

fn main() {
    let px_m = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(DEFAULT_PX_M);
    let w = (std::f64::consts::TAU * RADIUS_M / px_m).round() as usize;
    let h = w / 2;
    let (surface, backing) = load_surface();
    let surface = surface.as_ref();
    println!(
        "bake_rivers: {backing} backing, {w}x{h} @ {px_m:.0} m/px ({:.1} Mpx)",
        (w * h) as f64 / 1e6
    );

    // --- 1. sample the surface -------------------------------------------
    let t0 = std::time::Instant::now();
    let lod = px_m as f32;
    let height: Vec<f32> = (0..h)
        .into_par_iter()
        .flat_map(|y| {
            (0..w)
                .map(|x| surface.sample_height_m(dir_of(x, y, w, h).as_vec3(), lod))
                .collect::<Vec<_>>()
        })
        .collect();
    println!("  sampled height in {:.1}s", t0.elapsed().as_secs_f64());

    // Annual runoff is a second extensive field on the same cells. Keep it
    // separate from geometric catchment: drainage *shape* comes from terrain,
    // while perennial strength comes from climate. Folding the two together
    // would make it impossible to distinguish "large basin" from "large
    // river" later in the landcover and water renderers.
    let t_climate = std::time::Instant::now();
    let runoff_mm: Vec<f32> = (0..h)
        .into_par_iter()
        .flat_map(|y| {
            (0..w)
                .map(|x| annual_runoff_mm(surface.landcover_moisture(dir_of(x, y, w, h))))
                .collect::<Vec<_>>()
        })
        .collect();
    println!(
        "  sampled runoff climate in {:.1}s",
        t_climate.elapsed().as_secs_f64()
    );

    // --- 2. solve hydrology on the sampled raster -------------------------
    // The solver is independent of this preview adapter. The final neural bake
    // passes its completed DEM here directly, at the authored band resolution.
    let t_hydrology = std::time::Instant::now();
    let solved = solve_equirectangular(
        HydrologyConfig {
            width: w,
            height: h,
            planet_radius_m: RADIUS_M,
            fill_epsilon_m: FILL_EPSILON_M,
        },
        &height,
        &runoff_mm,
    )
    .unwrap_or_else(|e| panic!("hydrology solve failed: {e}"));
    println!(
        "  solved {} land cells in {:.1}s; filled {} cells (mean {:.1} m, max {:.1} m)",
        solved.descending_land.len(),
        t_hydrology.elapsed().as_secs_f64(),
        solved.raised_cell_count,
        solved.mean_fill_depth_m,
        solved.max_fill_depth_m
    );
    let receiver = &solved.receiver;
    let order_idx = &solved.descending_land;
    let accum_km2 = &solved.catchment_km2;
    let discharge_m3_s = &solved.discharge_m3_s;

    // --- 5. report + write -------------------------------------------------
    let land: Vec<f32> = order_idx.iter().map(|&i| accum_km2[i as usize]).collect();
    let mut sorted = land.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    let q = |f: f64| sorted[((sorted.len() - 1) as f64 * f) as usize];
    println!(
        "  catchment km2 on land: p50 {:.0}  p90 {:.0}  p99 {:.0}  p99.9 {:.0}  max {:.0}",
        q(0.5),
        q(0.9),
        q(0.99),
        q(0.999),
        sorted[sorted.len() - 1]
    );
    for thr in [1_000.0f32, 10_000.0, 100_000.0] {
        let n = land.iter().filter(|v| **v >= thr).count();
        println!(
            "    >= {thr:>9.0} km2: {n:>8} cells ({:.3}% of land)",
            n as f64 / land.len() as f64 * 100.0
        );
    }
    let mut discharge_sorted: Vec<f32> = order_idx
        .iter()
        .map(|&i| discharge_m3_s[i as usize])
        .collect();
    discharge_sorted.sort_unstable_by(f32::total_cmp);
    let dq = |f: f64| discharge_sorted[((discharge_sorted.len() - 1) as f64 * f) as usize];
    println!(
        "  annual-mean discharge m3/s: p50 {:.3}  p90 {:.1}  p99 {:.0}  p99.9 {:.0}  max {:.0}",
        dq(0.5),
        dq(0.9),
        dq(0.99),
        dq(0.999),
        discharge_sorted[discharge_sorted.len() - 1]
    );
    for thr in [1.0f32, 10.0, 100.0, 1_000.0] {
        let n = discharge_sorted.iter().filter(|v| **v >= thr).count();
        println!(
            "    >= {thr:>7.0} m3/s: {n:>8} cells ({:.3}% of land)",
            n as f64 / discharge_sorted.len() as f64 * 100.0
        );
    }

    // --- Horton-Strahler structure -----------------------------------------
    //
    // "Does this look like a river network?" has a standard answer: stream
    // counts should fall off geometrically with Strahler order, at a
    // bifurcation ratio Rb of about 3-5 on Earth. A network that is merely
    // dense-and-branching scores very differently from one that is
    // hierarchical, so this is the statistic to tune against rather than the
    // look of one render.
    //
    // Order is assigned upstream-first — the SAME sweep as accumulation
    // (descending filled height), so every donor is resolved before its
    // receiver. Reversing it silently yields all-order-1: nothing is ever a
    // confluence because no donor has an order yet.
    let mut order = vec![0u8; w * h];
    let mut donor_max = vec![0u8; w * h];
    let mut donor_max_count = vec![0u16; w * h];
    let channel = |i: usize| accum_km2[i] >= CHANNEL_HEAD_KM2;
    for &i in order_idx.iter() {
        let i = i as usize;
        if !channel(i) {
            continue;
        }
        // No channel donor => a source, order 1.
        let o = if donor_max[i] == 0 {
            1
        } else if donor_max_count[i] >= 2 {
            donor_max[i] + 1
        } else {
            donor_max[i]
        };
        order[i] = o;
        let r = receiver[i];
        if r != NO_RECEIVER {
            let r = r as usize;
            if o > donor_max[r] {
                donor_max[r] = o;
                donor_max_count[r] = 1;
            } else if o == donor_max[r] {
                donor_max_count[r] += 1;
            }
        }
    }
    // Count **stream segments**, not cells. Horton's ratio is defined over the
    // number of streams of each order; counting cells instead weights by
    // channel length and reports a different (flatter) number. A segment ends
    // where its order changes or it reaches the sea, so counting those ends
    // counts the segments exactly once each.
    let max_order = *order.iter().max().unwrap_or(&0);
    let mut counts = vec![0usize; max_order as usize + 2];
    for i in 0..w * h {
        let o = order[i];
        if o == 0 {
            continue;
        }
        let ends = match receiver[i] {
            NO_RECEIVER => true,
            r => order[r as usize] != o,
        };
        if ends {
            counts[o as usize] += 1;
        }
    }
    println!("  Horton-Strahler (channel head {CHANNEL_HEAD_KM2:.0} km2), Earth Rb ~ 3-5:");
    let mut ratios = Vec::new();
    for o in 1..=max_order as usize {
        let n = counts[o];
        let rb = if o + 1 <= max_order as usize && counts[o + 1] > 0 {
            let r = n as f64 / counts[o + 1] as f64;
            ratios.push(r);
            format!("{r:5.2}")
        } else {
            "    -".to_string()
        };
        println!("    order {o:>2}: {n:>8} streams   Rb {rb}");
    }
    if !ratios.is_empty() {
        println!(
            "    mean Rb {:.2} over orders 1..{}",
            ratios.iter().sum::<f64>() / ratios.len() as f64,
            max_order
        );
    }

    // Ship as **u8 log-catchment**, not f32 km2. Catchment spans seven decades
    // and the landcover only ever asks "how wet, 0..1", so an f32 spends 32 bits
    // encoding a dynamic range nothing consumes: 200 MB against 50 MB, and the
    // u8 is what compresses. A full byte step is ~3 % of catchment, far finer
    // than any visible change in wetness.
    let out_dir = std::path::Path::new("assets/terrain_packages/thalos_rivers");
    std::fs::create_dir_all(out_dir).unwrap();
    let bytes: Vec<u8> = accum_km2
        .iter()
        .map(|&v| {
            if v <= 1.0 {
                0
            } else {
                ((v.log10() / LOG_DECADES).clamp(0.0, 1.0) * 255.0).round() as u8
            }
        })
        .collect();
    let discharge_bytes: Vec<u8> = discharge_m3_s
        .iter()
        .map(|&v| encode_log_discharge(v))
        .collect();
    let stem = out_dir.join(format!("thalos_rivers_{}m", px_m as u64));
    std::fs::write(stem.with_extension("u8"), &bytes).unwrap();
    std::fs::write(stem.with_extension("discharge.u8"), &discharge_bytes).unwrap();
    let meta = format!(
        "{{\"width\":{w},\"height\":{h},\"px_m_equator\":{px_m},\"planet_radius_m\":{RADIUS_M},\"backing\":\"{backing}\",\"log_decades\":{LOG_DECADES},\"units\":\"u8 = 255*log10(catchment_km2)/log_decades\",\"discharge_log_min\":{DISCHARGE_LOG_MIN},\"discharge_log_max\":{DISCHARGE_LOG_MAX},\"discharge_units\":\"annual_mean_m3_s; byte 0 = zero; bytes 1..255 linear in log10\",\"runoff_model\":\"canonical_macro_moisture_v1\",\"mapping\":\"equirect\"}}
"
    );
    std::fs::write(stem.with_extension("json"), meta).unwrap();
    let nz = bytes.iter().filter(|b| **b > 0).count();
    println!(
        "  wrote {} + discharge ({:.1} MB each, {:.1}% catchment non-zero, {:.1}% runoff non-zero)",
        stem.with_extension("u8").display(),
        bytes.len() as f64 / 1e6,
        nz as f64 / bytes.len() as f64 * 100.0,
        discharge_bytes.iter().filter(|b| **b > 0).count() as f64 / discharge_bytes.len() as f64
            * 100.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discharge_encoding_reserves_zero_and_is_monotone() {
        assert_eq!(encode_log_discharge(0.0), 0);
        assert_eq!(encode_log_discharge(0.001), 0);
        let values = [0.01, 1.0, 100.0, 10_000.0, 1_000_000.0];
        let encoded: Vec<u8> = values.into_iter().map(encode_log_discharge).collect();
        assert!(encoded.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(encoded[encoded.len() - 1], 255);
    }
}
