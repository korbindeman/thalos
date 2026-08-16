//! Continuous offscreen performance benchmark for the real game render graph.
//!
//! The screenshot host is normally paced at 60 Hz and exits after one readback,
//! which makes it visual evidence rather than a benchmark. This opt-in plugin
//! keeps its real offscreen camera and warmed world alive, runs a foliage ×
//! shadow matrix in that one process, emits one machine-readable result per
//! cell, and exits without requesting a PNG. Keeping the world alive is
//! load-bearing: a cold forest scene can spend several minutes reaching
//! `Running`, which must not be charged four times or mistaken for frame cost.

use std::time::{Duration, Instant};

use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use thalos_body_render::tiles::TileTerrainRoot;

use super::PerfSamples;

const MODE_ENV: &str = "THALOS_HEADLESS_PERF";
const FRAMES_ENV: &str = "THALOS_HEADLESS_PERF_FRAMES";
const TIMEOUT_ENV: &str = "THALOS_HEADLESS_PERF_TIMEOUT_S";
const DEFAULT_MEASURE_FRAMES: usize = 240;
const STABLE_FRAMES: u32 = 120;
const FLUSH_FRAMES: u32 = 120;
const EXIT_TAIL_FRAMES: u32 = 30;
const DEFAULT_TIMEOUT_S: u64 = 1_200;

#[derive(Debug, Clone, Copy)]
struct BenchmarkVariant {
    label: &'static str,
    foliage: bool,
    cascades: usize,
}

// Run foliage-on cells first, then foliage-off cells. That avoids rebuilding
// the whole scatter clipmap merely to measure a later cell. Each transition
// still gets its own settle and flush windows.
const ATTRIBUTION_VARIANTS: [BenchmarkVariant; 4] = [
    BenchmarkVariant {
        label: "baseline",
        foliage: true,
        cascades: 4,
    },
    BenchmarkVariant {
        label: "shadows-off",
        foliage: true,
        cascades: 0,
    },
    BenchmarkVariant {
        label: "both-off",
        foliage: false,
        cascades: 0,
    },
    BenchmarkVariant {
        label: "foliage-off",
        foliage: false,
        cascades: 4,
    },
];

// Keep foliage resident while stepping the live cascade ceiling. Adjacent
// cells then differ by exactly one shadow camera and expose that camera's
// marginal frame cost without another cold scene load.
const SHADOW_CASCADE_VARIANTS: [BenchmarkVariant; 5] = [
    BenchmarkVariant {
        label: "cascades-4",
        foliage: true,
        cascades: 4,
    },
    BenchmarkVariant {
        label: "cascades-3",
        foliage: true,
        cascades: 3,
    },
    BenchmarkVariant {
        label: "cascades-2",
        foliage: true,
        cascades: 2,
    },
    BenchmarkVariant {
        label: "cascades-1",
        foliage: true,
        cascades: 1,
    },
    BenchmarkVariant {
        label: "cascades-0",
        foliage: true,
        cascades: 0,
    },
];

#[derive(Debug, Clone, Copy)]
enum BenchmarkMode {
    Attribution,
    ShadowCascades,
}

impl BenchmarkMode {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim() {
            "matrix" => Some(Self::Attribution),
            "shadow-cascades" => Some(Self::ShadowCascades),
            _ => None,
        }
    }

    fn variants(self) -> &'static [BenchmarkVariant] {
        match self {
            Self::Attribution => &ATTRIBUTION_VARIANTS,
            Self::ShadowCascades => &SHADOW_CASCADE_VARIANTS,
        }
    }
}

pub(crate) fn requested() -> bool {
    std::env::var(MODE_ENV).is_ok_and(|mode| BenchmarkMode::parse(&mode).is_some())
}

#[derive(Debug, Clone)]
struct BenchmarkConfig {
    mode: BenchmarkMode,
    measure_frames: usize,
    timeout: Duration,
}

impl BenchmarkConfig {
    fn from_env() -> Result<Self, String> {
        let mode = std::env::var(MODE_ENV).map_err(|_| format!("{MODE_ENV} is required"))?;
        let mode = BenchmarkMode::parse(&mode).ok_or_else(|| {
            format!("{MODE_ENV} must be 'matrix' or 'shadow-cascades', got {mode:?}")
        })?;
        let measure_frames = parse_or(FRAMES_ENV, DEFAULT_MEASURE_FRAMES)?;
        if !(120..=thalos_diagnostics_ui::FRAME_HISTORY_LEN).contains(&measure_frames) {
            return Err(format!(
                "{FRAMES_ENV} must be 120..={}, got {measure_frames}",
                thalos_diagnostics_ui::FRAME_HISTORY_LEN
            ));
        }
        let timeout_s = parse_or(TIMEOUT_ENV, DEFAULT_TIMEOUT_S)?;
        Ok(Self {
            mode,
            measure_frames,
            timeout: Duration::from_secs(timeout_s),
        })
    }
}

fn parse_or<T>(name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse()
            .map_err(|error| format!("{name}={raw:?}: {error}")),
        Err(_) => Ok(default),
    }
}

enum BenchmarkPhase {
    Configure {
        variant: usize,
    },
    WaitingForSettle {
        variant: usize,
    },
    Flushing {
        variant: usize,
        frames_left: u32,
    },
    Measuring {
        variant: usize,
        frames_left: usize,
        started: Instant,
    },
    Finishing {
        frames_left: u32,
    },
    Finished,
}

#[derive(Resource)]
struct HeadlessBenchmark {
    config: BenchmarkConfig,
    phase: BenchmarkPhase,
    started: Instant,
    last_counts: Option<(u32, u32)>,
    stable_frames: u32,
}

impl HeadlessBenchmark {
    fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            phase: BenchmarkPhase::Configure { variant: 0 },
            started: Instant::now(),
            last_counts: None,
            stable_frames: 0,
        }
    }

    fn active_label(&self) -> &'static str {
        let index = match self.phase {
            BenchmarkPhase::Configure { variant }
            | BenchmarkPhase::WaitingForSettle { variant }
            | BenchmarkPhase::Flushing { variant, .. }
            | BenchmarkPhase::Measuring { variant, .. } => variant,
            BenchmarkPhase::Finishing { .. } | BenchmarkPhase::Finished => {
                self.config.mode.variants().len() - 1
            }
        };
        self.config.mode.variants()[index].label
    }
}

pub(crate) struct HeadlessPerfBenchmarkPlugin;

impl Plugin for HeadlessPerfBenchmarkPlugin {
    fn build(&self, app: &mut App) {
        let config = BenchmarkConfig::from_env()
            .unwrap_or_else(|error| panic!("invalid headless performance benchmark: {error}"));
        app.insert_resource(HeadlessBenchmark::new(config))
            .add_systems(Last, drive_headless_benchmark);
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_headless_benchmark(
    mut benchmark: ResMut<HeadlessBenchmark>,
    perf: Res<PerfSamples>,
    frames: Res<thalos_diagnostics_ui::FrameSamples>,
    diagnostics: Res<DiagnosticsStore>,
    tile_roots: Query<&TileTerrainRoot>,
    screenshot: Res<crate::screenshot::ScreenshotConfig>,
    graphics: Res<crate::graphics_settings::GraphicsSettings>,
    preferences: Res<thalos_preferences::GraphicsPreferences>,
    mut perf_overrides: ResMut<crate::graphics_settings::PerfRenderOverrides>,
    vegetation_shadows: Res<crate::rendering::VegetationShadowStats>,
    app_state: Res<State<crate::loading::AppState>>,
    mut exit: MessageWriter<AppExit>,
) {
    if matches!(benchmark.phase, BenchmarkPhase::Finished) {
        return;
    }
    if benchmark.started.elapsed() > benchmark.config.timeout {
        error!(
            target: "thalos::diagnostic::perf",
            event = "headless_benchmark_failed",
            variant = benchmark.active_label(),
            reason = "matrix_timeout",
            timeout_s = benchmark.config.timeout.as_secs(),
            tile_resident = u64::from(perf.tile_resident),
            main_meshes = u64::from(perf.main_meshes),
            "headless benchmark matrix timed out"
        );
        benchmark.phase = BenchmarkPhase::Finished;
        exit.write(AppExit::error());
        return;
    }
    if *app_state.get() != crate::loading::AppState::Running {
        return;
    }

    match benchmark.phase {
        BenchmarkPhase::Configure { variant } => {
            let cell = benchmark.config.mode.variants()[variant];
            perf_overrides.set_foliage(cell.foliage);
            crate::rendering::sun_shadow::set_cascade_budget_for_benchmark(cell.cascades);
            benchmark.last_counts = None;
            benchmark.stable_frames = 0;
            info!(
                target: "thalos::diagnostic::perf",
                event = "headless_benchmark_config",
                variant = cell.label,
                foliage_enabled = cell.foliage,
                shadow_cascade_budget = cell.cascades as u64,
                "headless benchmark cell configured"
            );
            benchmark.phase = BenchmarkPhase::WaitingForSettle { variant };
        }
        BenchmarkPhase::WaitingForSettle { variant } => {
            let mut roots = tile_roots.iter().peekable();
            let terrain_covered =
                roots.peek().is_some() && roots.all(TileTerrainRoot::coverage_ready);
            let counts = (perf.main_meshes, perf.tile_resident);
            let counts_stable = benchmark.last_counts == Some(counts);
            benchmark.last_counts = Some(counts);
            if terrain_covered && perf.main_meshes > 0 && counts_stable {
                benchmark.stable_frames += 1;
            } else {
                benchmark.stable_frames = 0;
            }
            if benchmark.stable_frames >= STABLE_FRAMES {
                let cell = benchmark.config.mode.variants()[variant];
                info!(
                    target: "thalos::diagnostic::perf",
                    event = "headless_benchmark_ready",
                    variant = cell.label,
                    tile_resident = u64::from(perf.tile_resident),
                    main_meshes = u64::from(perf.main_meshes),
                    "headless benchmark cell settled"
                );
                benchmark.phase = BenchmarkPhase::Flushing {
                    variant,
                    frames_left: FLUSH_FRAMES,
                };
            }
        }
        BenchmarkPhase::Flushing {
            variant,
            ref mut frames_left,
        } => {
            *frames_left = frames_left.saturating_sub(1);
            if *frames_left == 0 {
                let cell = benchmark.config.mode.variants()[variant];
                info!(
                    target: "thalos::diagnostic::perf",
                    event = "headless_benchmark_start",
                    variant = cell.label,
                    frames = benchmark.config.measure_frames as u64,
                    "headless benchmark measurement started"
                );
                benchmark.phase = BenchmarkPhase::Measuring {
                    variant,
                    frames_left: benchmark.config.measure_frames,
                    started: Instant::now(),
                };
            }
        }
        BenchmarkPhase::Measuring {
            variant,
            ref mut frames_left,
            started,
        } => {
            *frames_left = frames_left.saturating_sub(1);
            if *frames_left != 0 {
                return;
            }

            let measure_frames = benchmark.config.measure_frames;
            let mut cpu_ms: Vec<f32> = frames
                .recent(measure_frames)
                .map(|(cpu_ms, _)| cpu_ms)
                .collect();
            if cpu_ms.len() != measure_frames {
                let cell = benchmark.config.mode.variants()[variant];
                error!(
                    target: "thalos::diagnostic::perf",
                    event = "headless_benchmark_failed",
                    variant = cell.label,
                    reason = "insufficient_frame_samples",
                    expected_frames = measure_frames as u64,
                    actual_frames = cpu_ms.len() as u64,
                    "headless benchmark frame history was incomplete"
                );
                benchmark.phase = BenchmarkPhase::Finished;
                exit.write(AppExit::error());
                return;
            }
            let mut gpu_sum_ms = 0.0_f32;
            let mut gpu_samples = 0_u32;
            for (_, gpu_ms) in frames.recent(measure_frames) {
                if gpu_ms > 0.0 {
                    gpu_sum_ms += gpu_ms;
                    gpu_samples += 1;
                }
            }
            cpu_ms.sort_by(|left, right| left.total_cmp(right));
            let count = cpu_ms.len();
            let cpu_mean_ms = cpu_ms.iter().sum::<f32>() / count as f32;
            let gpu_timing_available = gpu_samples > 0;
            let gpu_mean_ms = if gpu_timing_available {
                gpu_sum_ms / gpu_samples as f32
            } else {
                0.0
            };
            let cell = benchmark.config.mode.variants()[variant];
            info!(
                target: "thalos::diagnostic::perf",
                event = "headless_benchmark_end",
                variant = cell.label,
                frames = count as u64,
                wall_ms = started.elapsed().as_secs_f64() * 1000.0,
                fps = f64::from(1000.0 / cpu_mean_ms.max(1.0e-3)),
                cpu_ms_mean = f64::from(cpu_mean_ms),
                cpu_ms_p50 = f64::from(cpu_ms[count / 2]),
                cpu_ms_p95 = f64::from(cpu_ms[(count * 95 / 100).min(count - 1)]),
                cpu_ms_max = f64::from(cpu_ms[count - 1]),
                gpu_ms_mean = f64::from(gpu_mean_ms),
                gpu_timing_available,
                foliage_enabled = cell.foliage,
                clouds_enabled = graphics.clouds,
                grass_enabled = graphics.grass,
                gpu_grass_enabled = graphics.gpu_grass,
                msaa_samples = u64::from(preferences.msaa.samples()),
                shadow_cascade_budget = cell.cascades as u64,
                offscreen_width_px = u64::from(screenshot.width),
                offscreen_height_px = u64::from(screenshot.height),
                entities = thalos_diagnostics_ui::entity_count(&diagnostics),
                main_meshes = u64::from(perf.main_meshes),
                tile_resident = u64::from(perf.tile_resident),
                vegetation_shadow_cells = u64::from(vegetation_shadows.active_cells),
                vegetation_shadow_triangles = vegetation_shadows.triangles,
                tile_mib = f64::from(perf.tile_mib),
                slab_mib = f64::from(perf.slab_mib()),
                physics_ms = f64::from(perf.stage_physics_ms),
                sync_ms = f64::from(perf.stage_sync_ms),
                camera_ms = f64::from(perf.stage_camera_ms),
                "headless benchmark measurement complete"
            );

            let next = variant + 1;
            benchmark.phase = if next < benchmark.config.mode.variants().len() {
                BenchmarkPhase::Configure { variant: next }
            } else {
                BenchmarkPhase::Finishing {
                    frames_left: EXIT_TAIL_FRAMES,
                }
            };
        }
        BenchmarkPhase::Finishing {
            ref mut frames_left,
        } => {
            *frames_left = frames_left.saturating_sub(1);
            if *frames_left == 0 {
                benchmark.phase = BenchmarkPhase::Finished;
                exit.write(AppExit::Success);
            }
        }
        BenchmarkPhase::Finished => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_frame_window_fits_the_shared_ring() {
        assert!(DEFAULT_MEASURE_FRAMES <= thalos_diagnostics_ui::FRAME_HISTORY_LEN);
        assert!(DEFAULT_MEASURE_FRAMES >= 120);
    }

    #[test]
    fn matrix_contains_each_foliage_shadow_cell_once() {
        let cells: std::collections::BTreeSet<_> = ATTRIBUTION_VARIANTS
            .iter()
            .map(|variant| (variant.foliage, variant.cascades))
            .collect();
        assert_eq!(cells.len(), 4);
        assert!(cells.contains(&(true, 4)));
        assert!(cells.contains(&(false, 4)));
        assert!(cells.contains(&(true, 0)));
        assert!(cells.contains(&(false, 0)));
    }

    #[test]
    fn shadow_ladder_contains_every_budget_once() {
        let budgets: std::collections::BTreeSet<_> = SHADOW_CASCADE_VARIANTS
            .iter()
            .map(|variant| variant.cascades)
            .collect();
        assert_eq!(budgets, [0, 1, 2, 3, 4].into_iter().collect());
        assert!(
            SHADOW_CASCADE_VARIANTS
                .iter()
                .all(|variant| variant.foliage)
        );
    }
}
