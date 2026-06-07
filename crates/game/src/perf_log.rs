//! Slow-frame logging for agent-driven (BRP + chrome trace) profiling.
//!
//! Pairs with `scripts/analyze_trace.py`. When a frame exceeds
//! `SlowFrameThresholdMs`, this plugin:
//! - pushes a record into `SlowFrameLog` (a `Reflect` resource agents can
//!   read over BRP via `world_get_resources`, and watch via the standard
//!   resource accessors);
//! - emits a tracing `info_span!("slow_frame", frame_index, duration_ms)`
//!   so a `--features profile-chrome` run contains a span at the right
//!   `ts` to scope `analyze_trace.py --around-name slow_frame` to the
//!   bad window instead of the whole session.
//!
//! Threshold default 25 ms (vs 16.7 ms at 60 Hz, leaving headroom for
//! normal jitter). Override at startup with `THALOS_SLOW_FRAME_MS=…`,
//! or live via BRP `world_mutate_resources` on
//! `thalos_game::perf_log::SlowFrameThresholdMs`.

use std::time::Instant;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use tracing::info_span;

#[derive(Resource, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct SlowFrameThresholdMs(pub f32);

impl Default for SlowFrameThresholdMs {
    fn default() -> Self {
        const DEFAULT_MS: f32 = 25.0;
        let value = std::env::var("THALOS_SLOW_FRAME_MS")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(DEFAULT_MS);
        Self(value)
    }
}

#[derive(Debug, Clone, Copy, Reflect)]
pub struct SlowFrameRecord {
    pub frame_index: u32,
    pub duration_ms: f32,
    pub since_start_ms: f32,
}

#[derive(Resource, Debug, Clone, Reflect)]
#[reflect(Resource)]
pub struct SlowFrameLog {
    pub records: Vec<SlowFrameRecord>,
    pub capacity: usize,
    pub total_seen: u32,
}

impl Default for SlowFrameLog {
    fn default() -> Self {
        Self {
            records: Vec::with_capacity(64),
            capacity: 64,
            total_seen: 0,
        }
    }
}

impl SlowFrameLog {
    fn push(&mut self, rec: SlowFrameRecord) {
        if self.records.len() >= self.capacity {
            self.records.remove(0);
        }
        self.records.push(rec);
        self.total_seen = self.total_seen.saturating_add(1);
    }
}

#[derive(Resource)]
struct PerfLogState {
    process_start: Instant,
}

pub struct PerfLogPlugin;

impl Plugin for PerfLogPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<SlowFrameThresholdMs>()
            .register_type::<SlowFrameLog>()
            .register_type::<SlowFrameRecord>()
            .init_resource::<SlowFrameThresholdMs>()
            .init_resource::<SlowFrameLog>()
            .insert_resource(PerfLogState {
                process_start: Instant::now(),
            })
            // Run after Bevy publishes the FRAME_TIME measurement so we see
            // the current frame's delta rather than the previous one.
            .add_systems(
                Update,
                detect_slow_frames.after(FrameTimeDiagnosticsPlugin::diagnostic_system),
            );
    }
}

fn detect_slow_frames(
    diagnostics: Res<DiagnosticsStore>,
    threshold: Res<SlowFrameThresholdMs>,
    state: Res<PerfLogState>,
    frame_count: Res<bevy::diagnostic::FrameCount>,
    mut log: ResMut<SlowFrameLog>,
) {
    let Some(ms) = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.value())
    else {
        return;
    };
    let ms = ms as f32;
    if !ms.is_finite() || ms <= threshold.0 {
        return;
    }
    let rec = SlowFrameRecord {
        frame_index: frame_count.0,
        duration_ms: ms,
        since_start_ms: state.process_start.elapsed().as_secs_f64() as f32 * 1000.0,
    };
    // Lands in chrome trace as a "slow_frame" B/E pair at the trigger ts.
    // `scripts/analyze_trace.py --around-name slow_frame` uses this to scope
    // the top-N aggregation to the bad frame's neighbourhood.
    info_span!(
        "slow_frame",
        frame_index = rec.frame_index,
        duration_ms = rec.duration_ms,
    )
    .in_scope(|| {});
    info!(
        target: "thalos::perf",
        frame = rec.frame_index,
        ms = rec.duration_ms,
        "slow frame",
    );
    log.push(rec);
}
