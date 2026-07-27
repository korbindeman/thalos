//! **Streaming-throughput** harness for the tile renderer's synthesis pipeline.
//!
//! `procedural_bench` measures one tile in isolation. This one measures the
//! thing a cold surface load actually waits on: **tiles per second** out of the
//! two-pool shape in `body_render::tiles` —
//!
//!   * an outer [`TaskPool`]-equivalent of `OUTER` worker threads, each owning
//!     one tile's `TerrainTileProvider::request`;
//!   * an inner bounded rayon pool of `cores − 2` threads that every outer
//!     worker fans its tile's *rows* across.
//!
//! The outer width (`TILE_SYNTHESIS_THREADS = 4`) was sized for UDLOD's 512²
//! tiles — 262 k samples each. The tile renderer's tiles are 67² = 4.5 k
//! samples, **58× smaller**, so the same width may now be leaving the machine
//! idle while row fan-out overhead dominates. This sweeps outer width × inner
//! fan-out and reports tiles/s, so the constant is chosen against a measurement
//! instead of an inherited comment.
//!
//! Samples through `sample_bands_d` — the exact call `SurfaceQueryProvider`
//! makes — over the real 67×67 halo grid.
//!
//! Not a test (terrain-gen tests are disabled during iteration — see CLAUDE.md).
//!
//! Run (always `--release`):
//!   `cargo run --release -p thalos_terrain --example tile_stream_bench`
//!   `cargo run --release -p thalos_terrain --example tile_stream_bench -- 300`
//!     (arg: tiles per configuration)

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use glam::DVec3;
use thalos_terrain::ProceduralSurface;
use thalos_terrain::query::SurfaceQuery;

const RADIUS_M: f64 = 3_186_000.0; // Thalos

/// `tiles::TILE_RES` (65) + 2 × `TILE_HALO` (1) — the provider's sample grid.
const SIDE: usize = 67;

/// Level 12 on Thalos: a representative near-surface streaming tile.
/// `tile_arc / (TILE_RES − 1)` = the provider's `sample_spacing_m`.
fn spacing_at_level(level: u32) -> f64 {
    RADIUS_M * std::f64::consts::FRAC_PI_2 / (1u64 << level) as f64 / 64.0
}

/// One tile's worth of provider work, single-threaded (the `parallel_rows =
/// false` shape): 67 rows × 67 `sample_bands_d` calls.
fn eval_tile_serial(surface: &ProceduralSurface, base: DVec3, spacing: f64) -> f64 {
    (0..SIDE).map(|j| eval_row(surface, base, spacing, j)).sum()
}

fn eval_row(surface: &ProceduralSurface, base: DVec3, spacing: f64, j: usize) -> f64 {
    let tx = base.cross(DVec3::Y).normalize();
    let ty = base.cross(tx).normalize();
    let centre = base * RADIUS_M;
    let half = SIDE as f64 * 0.5;
    let v = (j as f64 - half) * spacing;
    let mut acc = 0.0;
    for i in 0..SIDE {
        let u = (i as f64 - half) * spacing;
        let dir = (centre + tx * u + ty * v).normalize();
        let (sample, bands) = surface.sample_bands_d(dir, spacing as f32);
        acc += sample.height_m as f64 + bands.canopy as f64;
    }
    acc
}

/// One tile's worth of provider work with the rows fanned across `pool` — the
/// production `SurfaceQueryProvider::request` shape.
fn eval_tile_rows_parallel(
    surface: &ProceduralSurface,
    base: DVec3,
    spacing: f64,
    pool: &rayon::ThreadPool,
) -> f64 {
    pool.install(|| {
        use rayon::prelude::*;
        (0..SIDE)
            .into_par_iter()
            .map(|j| eval_row(surface, base, spacing, j))
            .sum()
    })
}

/// Distinct tile centres, so no two tiles hit identical field lattice cells
/// (which would flatter the cache in a way real streaming never does).
fn tile_centres(n: usize) -> Vec<DVec3> {
    (0..n)
        .map(|i| {
            let a = 0.7 + i as f64 * 0.013;
            let b = 0.3 + i as f64 * 0.021;
            DVec3::new(a.cos() * 0.6, b.sin() * 0.5 + 0.37, 0.77).normalize()
        })
        .collect()
}

/// Run `tiles` through an outer pool of `outer` threads, each tile evaluated
/// either serially or with its rows fanned across `inner_pool`. Returns tiles/s.
fn measure(
    surface: &Arc<ProceduralSurface>,
    centres: &[DVec3],
    spacing: f64,
    outer: usize,
    inner_pool: Option<&rayon::ThreadPool>,
) -> f64 {
    let next = AtomicUsize::new(0);
    let checksum = AtomicUsize::new(0);
    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..outer {
            scope.spawn(|| {
                let mut local = 0.0f64;
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= centres.len() {
                        break;
                    }
                    local += match inner_pool {
                        Some(pool) => eval_tile_rows_parallel(surface, centres[idx], spacing, pool),
                        None => eval_tile_serial(surface, centres[idx], spacing),
                    };
                }
                checksum.fetch_add(local.abs() as usize, Ordering::Relaxed);
            });
        }
    });
    let elapsed = start.elapsed().as_secs_f64();
    black_box(checksum.load(Ordering::Relaxed));
    centres.len() as f64 / elapsed
}

fn main() {
    let tiles: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(240);

    let surface = Arc::new(ProceduralSurface::new(RADIUS_M as f32, 1));
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let inner_threads = cores.saturating_sub(2).max(2);
    let inner_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(inner_threads)
        .build()
        .expect("inner pool");

    let level = 12;
    let spacing = spacing_at_level(level);
    let centres = tile_centres(tiles);

    println!(
        "tile-stream throughput — Thalos R={:.0} km, level {level} (spacing {spacing:.1} m), \
         {SIDE}×{SIDE} = {} samples/tile, {tiles} tiles/config, {cores} cores",
        RADIUS_M / 1000.0,
        SIDE * SIDE,
    );
    println!(
        "inner rayon pool = {inner_threads} threads (production `tile_eval_pool` sizing)\n"
    );

    // Warm caches so the first configuration isn't charged for cold pages.
    let _ = measure(&surface, &centres[..16.min(tiles)], spacing, 4, Some(&inner_pool));

    println!(
        "{:>26} {:>6} {:>12} {:>12} {:>9}",
        "configuration", "outer", "tiles/s", "ms/tile", "vs prod"
    );
    println!("{}", "-".repeat(72));

    let mut baseline = 0.0;
    let configs: Vec<(&str, usize, bool)> = vec![
        ("PRODUCTION rows-parallel", 4, true),
        ("rows-parallel", 8, true),
        ("rows-parallel", inner_threads, true),
        ("serial tiles", 4, false),
        ("serial tiles", 8, false),
        ("serial tiles", inner_threads, false),
        ("serial tiles", cores, false),
    ];
    for (label, outer, rows_parallel) in configs {
        let rate = measure(
            &surface,
            &centres,
            spacing,
            outer,
            rows_parallel.then_some(&inner_pool),
        );
        if baseline == 0.0 {
            baseline = rate;
        }
        println!(
            "{:>26} {:>6} {:>12.0} {:>12.2} {:>8.2}×",
            label,
            outer,
            rate,
            1000.0 / rate,
            rate / baseline
        );
    }
}
