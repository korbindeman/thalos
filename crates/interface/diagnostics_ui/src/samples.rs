use std::time::Instant;

use bevy::{diagnostic::DiagnosticsStore, prelude::*};

/// Roughly 8.5 seconds at 60 fps: enough pre-context for a visible hitch.
pub const FRAME_HISTORY_LEN: usize = 512;

/// Ordering seam for consumers that need the newest shared frame before they
/// aggregate or emit their own diagnostics.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticsPanelPostUpdateSet {
    SampleFrame,
}

/// One authority for wall-clock CPU and top-level GPU frame history.
#[derive(Resource)]
pub struct FrameSamples {
    head: usize,
    filled: usize,
    cpu_ms: [f32; FRAME_HISTORY_LEN],
    gpu_ms: [f32; FRAME_HISTORY_LEN],
}

impl Default for FrameSamples {
    fn default() -> Self {
        Self {
            head: 0,
            filled: 0,
            cpu_ms: [0.0; FRAME_HISTORY_LEN],
            gpu_ms: [0.0; FRAME_HISTORY_LEN],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStats {
    pub count: usize,
    pub cpu_mean_ms: f32,
    pub cpu_p50_ms: f32,
    pub cpu_p95_ms: f32,
    pub cpu_max_ms: f32,
    pub gpu_mean_ms: f32,
}

impl FrameSamples {
    fn push(&mut self, cpu_ms: f32, gpu_ms: f32) {
        self.cpu_ms[self.head] = cpu_ms;
        self.gpu_ms[self.head] = gpu_ms;
        self.head = (self.head + 1) % FRAME_HISTORY_LEN;
        self.filled = (self.filled + 1).min(FRAME_HISTORY_LEN);
    }

    pub fn frame_count(&self) -> usize {
        self.filled
    }

    /// The `count` newest samples, chronological (oldest first).
    pub fn recent(&self, count: usize) -> impl Iterator<Item = (f32, f32)> + '_ {
        let count = count.min(self.filled);
        (0..count).map(move |offset| {
            let index = (self.head + FRAME_HISTORY_LEN - count + offset) % FRAME_HISTORY_LEN;
            (self.cpu_ms[index], self.gpu_ms[index])
        })
    }

    pub fn latest(&self) -> Option<(f32, f32)> {
        self.recent(1).next()
    }

    pub fn stats(&self, window: usize) -> Option<FrameStats> {
        let mut cpu: Vec<f32> = self.recent(window).map(|(cpu_ms, _)| cpu_ms).collect();
        if cpu.is_empty() {
            return None;
        }
        let gpu_mean_ms =
            self.recent(window).map(|(_, gpu_ms)| gpu_ms).sum::<f32>() / cpu.len() as f32;
        cpu.sort_by(|left, right| left.total_cmp(right));
        let count = cpu.len();
        Some(FrameStats {
            count,
            cpu_mean_ms: cpu.iter().sum::<f32>() / count as f32,
            cpu_p50_ms: cpu[count / 2],
            cpu_p95_ms: cpu[(count * 95 / 100).min(count - 1)],
            cpu_max_ms: cpu[count - 1],
            gpu_mean_ms,
        })
    }
}

/// Sum top-level `render/<pass>/elapsed_gpu` diagnostics without double
/// counting nested spans. Values are milliseconds and lag by GPU readback.
pub fn gpu_frame_ms(store: &DiagnosticsStore) -> f32 {
    store
        .iter()
        .filter_map(|diagnostic| {
            let path = diagnostic.path().as_str();
            if !path.starts_with("render/")
                || !path.ends_with("/elapsed_gpu")
                || path.bytes().filter(|byte| *byte == b'/').count() != 2
            {
                return None;
            }
            diagnostic.value().filter(|value| value.is_finite())
        })
        .sum::<f64>() as f32
}

pub(crate) fn collect_frame_samples(
    store: Res<DiagnosticsStore>,
    mut previous: Local<Option<Instant>>,
    mut samples: ResMut<FrameSamples>,
) {
    let now = Instant::now();
    let Some(last) = previous.replace(now) else {
        return;
    };
    let cpu_ms = (now - last).as_secs_f32() * 1000.0;
    if cpu_ms > 0.0 {
        samples.push(cpu_ms, gpu_frame_ms(&store));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_samples_remain_chronological_after_ring_wrap() {
        let mut samples = FrameSamples::default();
        for value in 0..(FRAME_HISTORY_LEN + 3) {
            samples.push(value as f32, value as f32 * 2.0);
        }
        let recent = samples.recent(3).collect::<Vec<_>>();
        assert_eq!(
            recent,
            vec![
                (FRAME_HISTORY_LEN as f32, FRAME_HISTORY_LEN as f32 * 2.0),
                (
                    FRAME_HISTORY_LEN as f32 + 1.0,
                    (FRAME_HISTORY_LEN as f32 + 1.0) * 2.0
                ),
                (
                    FRAME_HISTORY_LEN as f32 + 2.0,
                    (FRAME_HISTORY_LEN as f32 + 2.0) * 2.0
                ),
            ]
        );
    }

    #[test]
    fn stats_are_computed_over_the_requested_tail() {
        let mut samples = FrameSamples::default();
        for cpu_ms in [5.0, 10.0, 15.0, 20.0] {
            samples.push(cpu_ms, cpu_ms / 2.0);
        }
        let stats = samples.stats(3).unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.cpu_mean_ms, 15.0);
        assert_eq!(stats.cpu_p50_ms, 15.0);
        assert_eq!(stats.cpu_p95_ms, 20.0);
        assert_eq!(stats.cpu_max_ms, 20.0);
        assert_eq!(stats.gpu_mean_ms, 7.5);
    }
}
