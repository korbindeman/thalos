//! Per-tile synthesis **timing** harness for [`ProceduralSurface`].
//!
//! The sibling `procedural_probe` checks the field is *sane* (min/max/NaN); this
//! one measures how *expensive* it is, so we can optimise against numbers instead
//! of back-of-envelope octave counts. It evaluates a real 512²-sample tile-shaped
//! direction grid through the public `sample_d` (the exact per-pixel call the
//! UDLOD tile baker makes in `body_render`'s `compute_tile_pixels`) and reports:
//!
//!   * ms per tile and ns per sample, single-threaded (clean per-core cost) and
//!     via rayon (realistic wall-clock when a cold view bakes one tile);
//!   * a LOD sweep, and a least-squares fit of time vs per-pixel octave count
//!     that splits the cost into **fixed overhead**, the **LOD-invariant
//!     warp+continent floor**, and the **variable hills/mountain octaves** — i.e.
//!     exactly how much a "evaluate low-freq layers on a coarse sub-grid" change
//!     could save on a fine tile.
//!
//! Not a test (terrain-gen tests are disabled during iteration — see CLAUDE.md).
//!
//! Run (always `--release` — debug f64 noise is meaningless here):
//!   `cargo run --release -p thalos_terrain --example procedural_bench`
//!   `cargo run --release -p thalos_terrain --example procedural_bench -- 512 5`
//!     (args: tile size, repetitions)

use std::hint::black_box;
use std::time::Instant;

use glam::DVec3;
use rayon::prelude::*;
use thalos_terrain::ProceduralSurface;
use thalos_terrain::query::SurfaceQuery;

// --- Mirror of the private tuning in `crates/terrain/src/procedural.rs`. -------
// Used ONLY to label each LOD with its per-pixel octave count for the regression.
// The timings call the real `sample_d`, so if these drift the octave *labels*
// drift but the measured cost never does. Keep in sync with `procedural.rs`.
const HILLS_WL_M: f64 = 6_000.0;
const MOUNTAIN_WL_M: f64 = 20_000.0;
const MAX_OCTAVES: f64 = 11.0;
/// 3 warp fBm × 2 octaves each.
const WARP_PERLIN: f64 = 6.0;
/// `fbm(CONTINENT_OCTAVES=5)`, evaluated at full depth on every tile.
const CONTINENT_PERLIN: f64 = 5.0;

fn octaves_for_lod(lod_m: f64, base_wl_m: f64) -> f64 {
    if lod_m <= 0.0 {
        return MAX_OCTAVES;
    }
    let ratio = base_wl_m / (2.0 * lod_m);
    if ratio <= 1.0 {
        return 1.0;
    }
    (ratio.log2() + 1.0).clamp(1.0, MAX_OCTAVES)
}

/// Total `perlin3` calls per pixel at `lod_m` (warp + continents + hills + mtn).
fn perlin_per_pixel(lod_m: f64) -> f64 {
    WARP_PERLIN
        + CONTINENT_PERLIN
        + octaves_for_lod(lod_m, HILLS_WL_M)
        + octaves_for_lod(lod_m, MOUNTAIN_WL_M)
}

const RADIUS_M: f64 = 3_186_000.0; // Thalos
const INNER_TEXELS: f64 = 508.0; // 512 − 2·border, matches the real tile config

/// Real metres-per-texel for a Thalos tile at quadtree depth `lod` (mirrors
/// `pipeline::tile_lod_m`): face is π/2 rad split into `2^lod` tiles of
/// `INNER_TEXELS` each.
fn tile_lod_m_for_depth(lod: u32) -> f64 {
    let face_radians = std::f64::consts::FRAC_PI_2 / (1u64 << lod) as f64;
    (RADIUS_M * face_radians / INNER_TEXELS).max(1.0)
}

/// One timed pass over a `size×size` tile-shaped grid of directions centred on a
/// fixed base direction, spanning `lod_m × size` metres of surface (so lattice
/// cells are crossed at the same rate a real tile crosses them). Returns
/// (elapsed, checksum) — the checksum is black-boxed so the field eval can't be
/// optimised away.
fn time_pass(surface: &ProceduralSurface, lod_m: f64, size: usize, parallel: bool) -> (f64, f64) {
    // Orthonormal tile frame around a generic (non-axis-aligned) base direction.
    let base = DVec3::new(0.37, 0.51, 0.77).normalize();
    let tx = base.cross(DVec3::Y).normalize();
    let ty = base.cross(tx).normalize();
    let span_m = lod_m * size as f64;
    let step = span_m / size as f64;
    let half = size as f64 * 0.5;
    let center = base * RADIUS_M;

    let row_sum = |y: usize| -> f64 {
        let v = (y as f64 - half) * step;
        let mut s = 0.0;
        for x in 0..size {
            let u = (x as f64 - half) * step;
            let dir = (center + tx * u + ty * v).normalize();
            s += surface.sample_d(dir, lod_m as f32).height_m as f64;
        }
        s
    };

    let start = Instant::now();
    let checksum: f64 = if parallel {
        (0..size).into_par_iter().map(row_sum).sum()
    } else {
        (0..size).map(row_sum).sum()
    };
    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, black_box(checksum))
}

/// Minimum elapsed over `reps` passes (min = least scheduler/turbo noise).
fn best_of(
    surface: &ProceduralSurface,
    lod_m: f64,
    size: usize,
    reps: usize,
    parallel: bool,
) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let (t, _) = time_pass(surface, lod_m, size, parallel);
        best = best.min(t);
    }
    best
}

fn main() {
    let mut args = std::env::args().skip(1);
    let size: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(512);
    let reps: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(5);

    let surface = ProceduralSurface::new(RADIUS_M as f32, 1);
    let samples = (size * size) as f64;
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);

    println!(
        "ProceduralSurface tile-synthesis bench — Thalos R={:.0} km, {size}×{size} = {:.0} samples/tile, best-of-{reps}, {cores} cores",
        RADIUS_M / 1000.0,
        samples,
    );
    println!(
        "(measures the per-pixel FIELD eval — `sample_d`; cube-sphere projection's 2nd world_position, height encode, and the material-mask pass live in body_render and are not included)\n"
    );

    // Warm caches / branch predictors before the first measured LOD.
    let _ = time_pass(&surface, 1.0, size.min(128), false);

    // Real Thalos tile depths from LOD 0 (whole face) to ~13 (sub-metre).
    let depths = [0u32, 2, 4, 6, 8, 10, 11, 12, 13];

    println!(
        "{:>4} {:>12} {:>7} | {:>10} {:>9} {:>8} | {:>10} {:>7}",
        "LOD", "lod_m", "perlin", "ms/tile", "ns/samp", "Msmp/s", "ms(rayon)", "speedup",
    );
    println!("{}", "-".repeat(78));

    let mut fit_pts: Vec<(f64, f64)> = Vec::new(); // (perlin_per_pixel, ms_1thread)
    for &lod in &depths {
        let lod_m = tile_lod_m_for_depth(lod);
        let perlin = perlin_per_pixel(lod_m);

        let t1 = best_of(&surface, lod_m, size, reps, false);
        let tp = best_of(&surface, lod_m, size, reps, true);

        let ms1 = t1 * 1e3;
        let ns_samp = t1 / samples * 1e9;
        let msmps = samples / t1 / 1e6;
        let msp = tp * 1e3;

        println!(
            "{:>4} {:>10.1} m {:>7.1} | {:>10.2} {:>9.1} {:>8.1} | {:>10.2} {:>6.1}×",
            lod,
            lod_m,
            perlin,
            ms1,
            ns_samp,
            msmps,
            msp,
            t1 / tp,
        );
        fit_pts.push((perlin, ms1));
    }

    // Least-squares fit ms_1thread = a + b·(perlin/pixel).
    //   intercept a  → fixed per-tile overhead (projection.normalize + albedo +
    //                  loop), independent of octave count
    //   slope b      → ms per perlin-octave per tile
    //   floor 11·b   → warp(6) + continent(5): LOD-invariant, paid on every tile
    let n = fit_pts.len() as f64;
    let sx: f64 = fit_pts.iter().map(|p| p.0).sum();
    let sy: f64 = fit_pts.iter().map(|p| p.1).sum();
    let sxx: f64 = fit_pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = fit_pts.iter().map(|p| p.0 * p.1).sum();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let a = (sy - b * sx) / n;

    let fine_lod_m = tile_lod_m_for_depth(13);
    let fine_perlin = perlin_per_pixel(fine_lod_m);
    let t_fine = a + b * fine_perlin;
    let overhead = a;
    let floor = 11.0 * b; // warp + continent
    let variable = (fine_perlin - 11.0) * b; // hills + mountains at finest LOD

    println!(
        "\nCost decomposition (linear fit, finest tile = {:.1} perlin/px):",
        fine_perlin
    );
    println!(
        "  fixed overhead (projection+albedo+loop) : {:>6.2} ms  ({:>4.0}%)",
        overhead,
        100.0 * overhead / t_fine
    );
    println!(
        "  warp+continent floor (LOD-invariant)    : {:>6.2} ms  ({:>4.0}%)  <- multi-res target",
        floor,
        100.0 * floor / t_fine
    );
    println!(
        "  hills+mountains (variable, fine LOD)     : {:>6.2} ms  ({:>4.0}%)",
        variable,
        100.0 * variable / t_fine
    );
    println!(
        "  per-octave cost                          : {:>6.3} ms/octave/tile",
        b
    );

    if cores > 0 {
        let pool = cores.saturating_sub(2).max(2);
        println!(
            "\nProjected cold-view wall-clock for the finest tile on the real eval pool (~{pool} threads): ~{:.0} ms/tile.",
            t_fine / pool as f64,
        );
        println!(
            "The runtime bakes up to 4 such tiles concurrently; a fresh surface needs dozens → the multi-second cold-stream wait."
        );
    }
}
