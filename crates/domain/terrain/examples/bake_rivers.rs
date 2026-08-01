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

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::Write as _;

use glam::DVec3;
use rayon::prelude::*;
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
/// Catchment above which a cell counts as a channel for the structure stats.
const CHANNEL_HEAD_KM2: f32 = 1_000.0;

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
    println!("  sampled in {:.1}s", t0.elapsed().as_secs_f64());

    // Per-row geometry. `cos(lat)` shrinks the east-west run *and* the cell
    // area; both matter and they are not the same correction.
    let lat_of = |y: usize| (0.5 - (y as f64 + 0.5) / h as f64) * std::f64::consts::PI;
    let dy_m = std::f64::consts::PI * RADIUS_M / h as f64;
    let row_dx_m: Vec<f64> = (0..h)
        .map(|y| std::f64::consts::TAU * RADIUS_M * lat_of(y).cos() / w as f64)
        .collect();
    let row_area_km2: Vec<f64> = (0..h).map(|y| row_dx_m[y] * dy_m / 1.0e6).collect();

    // --- 2. fill depressions (priority-flood) -----------------------------
    //
    // **Without this there are no rivers.** Analytic terrain is full of local
    // minima, and a raw steepest-descent router terminates at every one, so each
    // pit truncates its own catchment: the first bake of this peaked at
    // 20 505 km2 of drainage — smaller than the Rhine — because no flow ever
    // reached the sea. Priority-flood raises each pit to its spill elevation, so
    // every land cell gains a monotone downhill path to the ocean and catchments
    // compose all the way down.
    //
    // Routing uses the FILLED heights; the original field is untouched and is
    // still what the game renders. A filled cell just means "water would cross
    // here", which is exactly what a lake or a floodplain is.
    let t_fill = std::time::Instant::now();
    let key = |v: f32| -> u32 {
        // Monotone f32 -> u32 so the heap can order by height without floats.
        let b = v.to_bits();
        if v >= 0.0 { b | 0x8000_0000 } else { !b }
    };
    let mut filled = height.clone();
    let mut done = vec![false; w * h];
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    for i in 0..w * h {
        if height[i] <= 0.0 {
            done[i] = true;
            heap.push(Reverse((key(height[i]), i as u32)));
        }
    }
    let mut raised = 0usize;
    let mut raised_sum = 0.0f64;
    while let Some(Reverse((_, i))) = heap.pop() {
        let i = i as usize;
        let (y, x) = (i / w, i % w);
        for dyi in -1i64..=1 {
            for dxi in -1i64..=1 {
                if dxi == 0 && dyi == 0 {
                    continue;
                }
                let ny = y as i64 + dyi;
                if ny < 0 || ny >= h as i64 {
                    continue;
                }
                let nx = (x as i64 + dxi).rem_euclid(w as i64);
                let j = ny as usize * w + nx as usize;
                if done[j] {
                    continue;
                }
                done[j] = true;
                // **Epsilon fill, not flat fill.** Filling a depression to
                // exactly its spill height creates a *plateau*, and a
                // steepest-descent router finds no strictly-downhill neighbour
                // on a plateau — so every filled cell becomes a fresh sink and
                // drainage gets *worse*, not better. Measured: plain fill took
                // the largest catchment from 20 505 km2 down to 10 928. Adding
                // a hair of gradient per step guarantees a strict descent out
                // of every basin. Over a 1 000-cell flat this accumulates 1 m,
                // which is far below anything the terrain or the landcover
                // notices.
                // Randomised per cell, not constant: a constant epsilon makes
                // every flat drain in whichever direction the flood happened to
                // sweep, which prints as long straight streaks. A positive
                // random step keeps the descent strict but removes the
                // direction preference.
                let jit = {
                    let mut z = (j as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    z ^= z >> 29;
                    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    0.5 + ((z >> 40) as f32 / 16_777_216.0)
                };
                let target = filled[i] + FILL_EPSILON_M * jit;
                if filled[j] < target {
                    raised += 1;
                    raised_sum += (target - filled[j]) as f64;
                    filled[j] = target;
                }
                heap.push(Reverse((key(filled[j]), j as u32)));
            }
        }
    }
    println!(
        "  filled {raised} depressions (mean rise {:.1} m) in {:.1}s",
        if raised > 0 {
            raised_sum / raised as f64
        } else {
            0.0
        },
        t_fill.elapsed().as_secs_f64()
    );

    // --- 3. steepest-descent receiver, by GRADIENT not height drop --------
    let t1 = std::time::Instant::now();
    let receiver: Vec<i64> = (0..h)
        .into_par_iter()
        .flat_map(|y| {
            let dx_m = row_dx_m[y];
            (0..w)
                .map(|x| {
                    let i = y * w + x;
                    let hc = filled[i];
                    if height[i] <= 0.0 {
                        return -1; // ocean: a sink, not a router
                    }
                    let (mut best, mut best_slope) = (-1i64, 0.0f64);
                    for dyi in -1i64..=1 {
                        for dxi in -1i64..=1 {
                            if dxi == 0 && dyi == 0 {
                                continue;
                            }
                            let ny = y as i64 + dyi;
                            if ny < 0 || ny >= h as i64 {
                                continue;
                            }
                            let nx = (x as i64 + dxi).rem_euclid(w as i64);
                            let j = ny as usize * w + nx as usize;
                            let run =
                                ((dxi as f64 * dx_m).powi(2) + (dyi as f64 * dy_m).powi(2)).sqrt();
                            let slope = (hc - filled[j]) as f64 / run;
                            if slope > best_slope {
                                best_slope = slope;
                                best = j as i64;
                            }
                        }
                    }
                    best
                })
                .collect::<Vec<_>>()
        })
        .collect();
    println!("  receivers in {:.1}s", t1.elapsed().as_secs_f64());

    // --- 4. accumulate downhill, highest cell first -----------------------
    // Processing in descending height guarantees every donor is resolved before
    // its receiver, so one pass suffices — no iteration to convergence.
    let t2 = std::time::Instant::now();
    let mut order_idx: Vec<u32> = (0..(w * h) as u32)
        .filter(|&i| height[i as usize] > 0.0)
        .collect();
    order_idx.sort_unstable_by(|a, b| filled[*b as usize].total_cmp(&filled[*a as usize]));
    let mut accum_km2: Vec<f32> = vec![0.0; w * h];
    for &i in &order_idx {
        let i = i as usize;
        let own = row_area_km2[i / w] as f32;
        let total = accum_km2[i] + own;
        accum_km2[i] = total;
        let r = receiver[i];
        if r >= 0 {
            accum_km2[r as usize] += total;
        }
    }
    println!(
        "  accumulated {} land cells in {:.1}s",
        order_idx.len(),
        t2.elapsed().as_secs_f64()
    );

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
        if r >= 0 {
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
            r if r < 0 => true,
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
    let stem = out_dir.join(format!("thalos_rivers_{}m", px_m as u64));
    std::fs::write(stem.with_extension("u8"), &bytes).unwrap();
    let meta = format!(
        "{{\"width\":{w},\"height\":{h},\"px_m_equator\":{px_m},\"planet_radius_m\":{RADIUS_M},\"backing\":\"{backing}\",\"log_decades\":{LOG_DECADES},\"units\":\"u8 = 255*log10(catchment_km2)/log_decades\",\"mapping\":\"equirect\"}}
"
    );
    std::fs::write(stem.with_extension("json"), meta).unwrap();
    let nz = bytes.iter().filter(|b| **b > 0).count();
    println!(
        "  wrote {} ({:.1} MB u8, {:.1}% non-zero)",
        stem.with_extension("u8").display(),
        bytes.len() as f64 / 1e6,
        nz as f64 / bytes.len() as f64 * 100.0
    );
}
