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
