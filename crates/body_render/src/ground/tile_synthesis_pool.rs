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
/// `TILE_LOAD_SLOTS` in `crates/game/src/rendering/ground_terrain.rs` so the
/// streamer's concurrent-tile cap maps one-to-one to a worker; more threads
/// here would just over-subscribe cores against the renderer / scheduler.
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
