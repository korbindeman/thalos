//! Dedicated [`TaskPool`] for CPU-heavy tile synthesis.
//!
//! Routing tile synthesis through Bevy's shared
//! [`AsyncComputeTaskPool`](bevy::tasks::AsyncComputeTaskPool) was the cause of
//! periodic ~2.5–3 s frame hitches at a ~3.4 s cadence while terrain streamed
//! in: Avian's collider-tree optimisation (`block_on_optimize_trees`) spawns
//! its async work on `AsyncComputeTaskPool` and `block_on`s for it from the
//! main thread inside the physics schedule. With up to `TILE_LOAD_SLOTS = 4`
//! concurrent OceanicTerrestrial syntheses already saturating that pool, the
//! optimisation task queued behind them and the main thread waited a full
//! "wave" of tile completions per physics step.
//!
//! Giving tile synthesis its own pool removes the contention entirely; Avian
//! and every other consumer of `AsyncComputeTaskPool` keep their fair share.
//! Inner per-tile parallelism (rayon, where used) is unchanged — only the
//! outer worker threads move to a separate pool.

use std::sync::OnceLock;

use bevy::tasks::{TaskPool, TaskPoolBuilder};

/// Worker thread count for the tile-synthesis pool. Sized to match
/// `TILE_LOAD_SLOTS` in `crates/runtime/game/src/rendering/ground_terrain.rs` so the
/// streamer's concurrent-tile cap maps one-to-one to a worker; more threads
/// here would just over-subscribe cores against the renderer / scheduler.
///
/// **Widening this was tried and reverted (2026-07-26).** In isolation the
/// machine does prefer a wide outer stage over per-tile row fan-out —
/// `cargo run --release -p thalos_terrain --example tile_stream_bench` measures
/// 1,035 tiles/s at `cores − 2` whole-tile workers against 868 for this 4-wide
/// row-fanning shape. In the running game it went the other way: mean per-tile
/// wall time rose from ~6 ms to ~32 ms and the settle took *longer*, because the
/// bench models an idle machine and the real one is also running the render
/// thread, Avian, and legacy udlod streaming a second body. That contention is
/// the entire reason this pool is bounded and separate in the first place, so
/// the microbenchmark does not get to overrule it.
///
/// If this is revisited, the measurement that matters is `total_landed` per
/// second between `installed` and the first `resident == desired` residency
/// gauge in `artifacts/diagnostics/runtime.jsonl` — not a standalone harness.
const TILE_SYNTHESIS_THREADS: usize = 4;

static POOL: OnceLock<TaskPool> = OnceLock::new();

/// Global tile-synthesis [`TaskPool`]. Initialised lazily on first call.
pub fn tile_synthesis_pool() -> &'static TaskPool {
    POOL.get_or_init(|| {
        TaskPoolBuilder::new()
            .num_threads(TILE_SYNTHESIS_THREADS)
            .thread_name("Tile Synthesis".to_string())
            .build()
    })
}

/// Worker thread count for the scatter-build pool. Bounded for the same reason
/// as everything in this module — the render thread, Avian, and the tile pools
/// keep their cores — but *separate* from the tile pool so a cold tree fill
/// does not queue behind minutes of far-mountain terrain refinement (and vice
/// versa: scatter can't starve the ground it is waiting on).
const VEG_SCATTER_THREADS: usize = 4;

static VEG_POOL: OnceLock<TaskPool> = OnceLock::new();

/// Dedicated [`TaskPool`] for vegetation / rock scatter builds
/// ([`crate::ground::scatter::build_scatter_tile`] + the per-tile mesh
/// combines).
///
/// These used to run on Bevy's shared `AsyncComputeTaskPool`, which is both
/// narrow (a fraction of cores, capped small) and contended — the exact shape
/// that hitched the main thread when tile synthesis lived there (see the module
/// doc). A cold tree carpet is ~2,300 tile builds; on the shared pool it
/// drained at ~100 tiles/s with the queue permanently full, i.e. the pool
/// width, not the queue depth, was the throughput ceiling
/// (`runtime.jsonl` vegetation `drive_gauge`, 2026-07-29).
pub fn veg_scatter_pool() -> &'static TaskPool {
    VEG_POOL.get_or_init(|| {
        TaskPoolBuilder::new()
            .num_threads(VEG_SCATTER_THREADS)
            .thread_name("Veg Scatter".to_string())
            .build()
    })
}

static EVAL_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Bounded rayon pool for the *inner* per-tile pixel evaluation
/// ([`crate::ground::pipeline`]'s `compute_tile_pixels`).
///
/// A single Thalos tile is ~262 k expensive field samples; evaluating it on one
/// thread leaves a cold view (e.g. a teleport straight to a runway, where only a
/// handful of tiles are needed) sitting for tens of seconds while most cores
/// idle. Spreading each tile across cores cuts that to a few seconds. But the
/// whole reason tile synthesis has its own [`TaskPool`] (rather than
/// `AsyncComputeTaskPool`) is to *not* starve the renderer/main thread — so the
/// inner parallelism is bounded here too, leaving a couple of cores free, and
/// every tile bake shares this one pool so N concurrent bakes can't collectively
/// saturate every core. Using rayon's implicit global pool instead would defeat
/// that isolation.
pub fn tile_eval_pool() -> &'static rayon::ThreadPool {
    EVAL_POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let threads = cores.saturating_sub(2).max(2);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("Tile Eval {i}"))
            .build()
            .expect("build tile-eval rayon pool")
    })
}
