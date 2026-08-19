//! Continuous offscreen performance benchmark for the real game render graph.
//!
//! The screenshot host is normally paced at 60 Hz and exits after one readback,
//! which makes it visual evidence rather than a benchmark. This opt-in plugin
//! keeps its real offscreen camera and warmed world alive, runs a foliage ×
//! shadow matrix in that one process, emits one machine-readable result per
//! cell, and exits without requesting a PNG. Keeping the world alive is
//! load-bearing: cold scene readiness must be paid once per matrix, not once
//! per cell or mistaken for frame cost.

use std::time::{Duration, Instant};

use bevy::camera::primitives::Aabb;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::diagnostic::DiagnosticsStore;
use bevy::prelude::*;
use thalos_body_render::tiles::{
    TileBodyOrigin, TileCullingProbeBounds, TileIndexProbeMeshes, TileTerrainRoot,
    material::TileTerrainMaterial, tile_index_count_for_benchmark,
};

use super::PerfSamples;

const MODE_ENV: &str = "THALOS_HEADLESS_PERF";
const FRAMES_ENV: &str = "THALOS_HEADLESS_PERF_FRAMES";
const TIMEOUT_ENV: &str = "THALOS_HEADLESS_PERF_TIMEOUT_S";
const DEFAULT_MEASURE_FRAMES: usize = 240;
const SCENE_STABLE_FRAMES: u32 = 120;
const VARIANT_FLUSH_FRAMES: u32 = 30;
const EXIT_TAIL_FRAMES: u32 = 30;
const DEFAULT_TIMEOUT_S: u64 = 1_200;

#[derive(Debug, Clone, Copy)]
struct BenchmarkVariant {
    label: &'static str,
    foliage: bool,
    cascades: usize,
    terrain_inspection: u32,
    terrain_visible: bool,
    depth_prepass: bool,
    terrain_index_step: usize,
}

impl BenchmarkVariant {
    fn tight_tile_bounds(self) -> Option<bool> {
        match self.label {
            "terrain-base-full-bounds-before"
            | "terrain-hidden-full-bounds"
            | "terrain-base-full-bounds-after" => Some(false),
            "terrain-base-tight-bounds" | "terrain-hidden-tight-bounds" => Some(true),
            _ => None,
        }
    }
}

// Run foliage-on cells first, then foliage-off cells. That avoids rebuilding
// the whole scatter clipmap merely to measure a later cell. The matrix settles
// the complete foliage-on scene before cell 1; foliage only transitions off,
// while every live switch uses the shared short flush window.
const ATTRIBUTION_VARIANTS: [BenchmarkVariant; 4] = [
    BenchmarkVariant {
        label: "baseline",
        foliage: true,
        cascades: 4,
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "shadows-off",
        foliage: true,
        cascades: 0,
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "both-off",
        foliage: false,
        cascades: 0,
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "foliage-off",
        foliage: false,
        cascades: 4,
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
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
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "cascades-3",
        foliage: true,
        cascades: 3,
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "cascades-2",
        foliage: true,
        cascades: 2,
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "cascades-1",
        foliage: true,
        cascades: 1,
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "cascades-0",
        foliage: true,
        cascades: 0,
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
];

// Bracket two inspection ablations with the ordinary lit path. All cells keep
// the exact same geometry and disable custom shadow cameras: fullbright keeps
// the procedural material layers but skips PBR, while base-color returns the
// baked vertex albedo before the layer stack. The second lit cell exposes
// thermal or clock drift instead of letting it masquerade as shader cost.
const TERRAIN_MATERIAL_VARIANTS: [BenchmarkVariant; 5] = [
    BenchmarkVariant {
        label: "terrain-lit-before",
        foliage: true,
        cascades: 0,
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-fullbright",
        foliage: true,
        cascades: 0,
        terrain_inspection: 1,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-base-color",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-hidden",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: false,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-lit-after",
        foliage: true,
        cascades: 0,
        terrain_inspection: 0,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
];

// Measure the terrain portion of Bevy's depth prepass without charging the
// rest of the opaque scene to terrain. Visible and hidden controls run with
// the prepass both enabled and disabled; the ordinary visible path brackets
// the axis so drift cannot masquerade as prepass cost. Base-colour inspection
// removes procedural and PBR shader work from every visible cell.
const TERRAIN_PREPASS_VARIANTS: [BenchmarkVariant; 5] = [
    BenchmarkVariant {
        label: "terrain-base-prepass-before",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-hidden-prepass",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: false,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-base-no-prepass",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: true,
        depth_prepass: false,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-hidden-no-prepass",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: false,
        depth_prepass: false,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-base-prepass-after",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
];

// Main-pass geometry discriminator. The opt-in tile probe builds a compact
// full-attribute 33² twin from every fourth sample of the visible 129² mesh.
// No-prepass base-colour cells keep the measurement on the main opaque path;
// hidden controls remove the fixed non-terrain scene, and dense cells bracket
// the handle swap.
const TERRAIN_INDEX_VARIANTS: [BenchmarkVariant; 5] = [
    BenchmarkVariant {
        label: "terrain-base-dense-before",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: true,
        depth_prepass: false,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-hidden-dense",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: false,
        depth_prepass: false,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-base-coarse",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: true,
        depth_prepass: false,
        terrain_index_step: 4,
    },
    BenchmarkVariant {
        label: "terrain-hidden-coarse",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: false,
        depth_prepass: false,
        terrain_index_step: 4,
    },
    BenchmarkVariant {
        label: "terrain-base-dense-after",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: true,
        depth_prepass: false,
        terrain_index_step: 1,
    },
];

// Fidelity-free culling candidate. All cells retain the exact dense mesh and
// production prepass; only the local-space AABB switches between Bevy's full
// skirt-inflated extent and the already-validated tight surface band used by
// terrain shadow twins.
const TERRAIN_CULLING_VARIANTS: [BenchmarkVariant; 5] = [
    BenchmarkVariant {
        label: "terrain-base-full-bounds-before",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-hidden-full-bounds",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: false,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-base-tight-bounds",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-hidden-tight-bounds",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: false,
        depth_prepass: true,
        terrain_index_step: 1,
    },
    BenchmarkVariant {
        label: "terrain-base-full-bounds-after",
        foliage: true,
        cascades: 0,
        terrain_inspection: 4,
        terrain_visible: true,
        depth_prepass: true,
        terrain_index_step: 1,
    },
];

#[derive(Debug, Clone, Copy)]
enum BenchmarkMode {
    Attribution,
    ShadowCascades,
    TerrainMaterial,
    TerrainPrepass,
    TerrainIndex,
    TerrainCulling,
}

impl BenchmarkMode {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim() {
            "matrix" => Some(Self::Attribution),
            "shadow-cascades" => Some(Self::ShadowCascades),
            "terrain-material" => Some(Self::TerrainMaterial),
            "terrain-prepass" => Some(Self::TerrainPrepass),
            "terrain-index" => Some(Self::TerrainIndex),
            "terrain-culling" => Some(Self::TerrainCulling),
            _ => None,
        }
    }

    fn variants(self) -> &'static [BenchmarkVariant] {
        match self {
            Self::Attribution => &ATTRIBUTION_VARIANTS,
            Self::ShadowCascades => &SHADOW_CASCADE_VARIANTS,
            Self::TerrainMaterial => &TERRAIN_MATERIAL_VARIANTS,
            Self::TerrainPrepass => &TERRAIN_PREPASS_VARIANTS,
            Self::TerrainIndex => &TERRAIN_INDEX_VARIANTS,
            Self::TerrainCulling => &TERRAIN_CULLING_VARIANTS,
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
            format!(
                "{MODE_ENV} must be 'matrix', 'shadow-cascades', 'terrain-material', \
                 'terrain-prepass', 'terrain-index', or 'terrain-culling', got {mode:?}"
            )
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

fn benchmark_scene_ready(
    root_exists: bool,
    roots_covered: bool,
    roots_settled: bool,
    main_meshes: u32,
    counts_stable: bool,
) -> bool {
    root_exists && roots_covered && roots_settled && main_meshes > 0 && counts_stable
}

enum BenchmarkPhase {
    WaitingForScene {
        variant: usize,
    },
    Configure {
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
    active_terrain_index_step: usize,
}

impl HeadlessBenchmark {
    fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            phase: BenchmarkPhase::WaitingForScene { variant: 0 },
            started: Instant::now(),
            last_counts: None,
            stable_frames: 0,
            active_terrain_index_step: 1,
        }
    }

    fn active_label(&self) -> &'static str {
        let index = match self.phase {
            BenchmarkPhase::WaitingForScene { variant }
            | BenchmarkPhase::Configure { variant }
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
    mut commands: Commands,
    mut benchmark: ResMut<HeadlessBenchmark>,
    perf: Res<PerfSamples>,
    frames: Res<thalos_diagnostics_ui::FrameSamples>,
    diagnostics: Res<DiagnosticsStore>,
    tile_roots: Query<&TileTerrainRoot>,
    mut tile_meshes: ParamSet<(
        Query<&mut Visibility, With<TileBodyOrigin>>,
        Query<(&mut Mesh3d, &TileIndexProbeMeshes), With<TileBodyOrigin>>,
        Query<&ViewVisibility, With<TileBodyOrigin>>,
        Query<(&mut Aabb, &TileCullingProbeBounds), With<TileBodyOrigin>>,
    )>,
    mut tile_materials: ResMut<Assets<TileTerrainMaterial>>,
    ship_cameras: Query<Entity, With<crate::camera::ShipCamera>>,
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
        BenchmarkPhase::WaitingForScene { variant } => {
            let root_exists = tile_roots.iter().next().is_some();
            let roots_covered = tile_roots.iter().all(TileTerrainRoot::coverage_ready);
            let roots_settled = tile_roots.iter().all(TileTerrainRoot::settled);
            let counts = (perf.main_meshes, perf.tile_resident);
            let counts_stable = benchmark.last_counts == Some(counts);
            benchmark.last_counts = Some(counts);
            if benchmark_scene_ready(
                root_exists,
                roots_covered,
                roots_settled,
                perf.main_meshes,
                counts_stable,
            ) {
                benchmark.stable_frames += 1;
            } else {
                benchmark.stable_frames = 0;
            }
            if benchmark.stable_frames >= SCENE_STABLE_FRAMES {
                let cell = benchmark.config.mode.variants()[variant];
                info!(
                    target: "thalos::diagnostic::perf",
                    event = "headless_benchmark_ready",
                    variant = cell.label,
                    tile_resident = u64::from(perf.tile_resident),
                    main_meshes = u64::from(perf.main_meshes),
                    settle_frames = u64::from(benchmark.stable_frames),
                    "headless benchmark scene settled"
                );
                benchmark.phase = BenchmarkPhase::Configure { variant };
            }
        }
        BenchmarkPhase::Configure { variant } => {
            let cell = benchmark.config.mode.variants()[variant];
            let expected_indices = tile_index_count_for_benchmark(cell.terrain_index_step)
                .expect("benchmark variant has a valid tile index step");
            let change_index_step = benchmark.active_terrain_index_step != cell.terrain_index_step;
            perf_overrides.set_foliage(cell.foliage);
            crate::rendering::sun_shadow::set_cascade_budget_for_benchmark(cell.cascades);
            for (_, material) in tile_materials.iter_mut() {
                material.extension.params.inspect = cell.terrain_inspection;
            }
            for mut visibility in &mut tile_meshes.p0() {
                *visibility = if cell.terrain_visible {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
            if change_index_step {
                let mut changed_meshes = 0_u32;
                for (mut mesh, probe) in &mut tile_meshes.p1() {
                    mesh.0 = if cell.terrain_index_step == 1 {
                        probe.dense.clone()
                    } else {
                        probe.coarse.clone()
                    };
                    changed_meshes += 1;
                }
                if changed_meshes != perf.tile_resident {
                    error!(
                        target: "thalos::diagnostic::perf",
                        event = "headless_benchmark_failed",
                        variant = cell.label,
                        reason = "tile_index_probe_incomplete",
                        expected_tiles = u64::from(perf.tile_resident),
                        changed_meshes = u64::from(changed_meshes),
                        terrain_index_step = cell.terrain_index_step as u64,
                        "headless tile index mutation was incomplete"
                    );
                    benchmark.phase = BenchmarkPhase::Finished;
                    exit.write(AppExit::error());
                    return;
                }
                benchmark.active_terrain_index_step = cell.terrain_index_step;
            }
            if let Some(tight_bounds) = cell.tight_tile_bounds() {
                let mut changed_bounds = 0_u32;
                for (mut aabb, bounds) in &mut tile_meshes.p3() {
                    *aabb = if tight_bounds {
                        bounds.surface
                    } else {
                        bounds.full
                    };
                    changed_bounds += 1;
                }
                if changed_bounds != perf.tile_resident {
                    error!(
                        target: "thalos::diagnostic::perf",
                        event = "headless_benchmark_failed",
                        variant = cell.label,
                        reason = "tile_culling_probe_incomplete",
                        expected_tiles = u64::from(perf.tile_resident),
                        changed_bounds = u64::from(changed_bounds),
                        tight_tile_bounds = tight_bounds,
                        "headless tile bounds mutation was incomplete"
                    );
                    benchmark.phase = BenchmarkPhase::Finished;
                    exit.write(AppExit::error());
                    return;
                }
            }
            for entity in &ship_cameras {
                if cell.depth_prepass {
                    commands.entity(entity).insert(DepthPrepass);
                } else {
                    commands.entity(entity).remove::<DepthPrepass>();
                }
            }
            info!(
                target: "thalos::diagnostic::perf",
                event = "headless_benchmark_config",
                variant = cell.label,
                foliage_enabled = cell.foliage,
                shadow_cascade_budget = cell.cascades as u64,
                shadow_quality = crate::rendering::sun_shadow::quality_label_for_budget(cell.cascades),
                shadow_map_size_px = u64::from(crate::rendering::sun_shadow::SHADOW_MAP_SIZE),
                terrain_inspection = u64::from(cell.terrain_inspection),
                terrain_visible = cell.terrain_visible,
                depth_prepass_enabled = cell.depth_prepass,
                terrain_index_step = cell.terrain_index_step as u64,
                terrain_indices_per_tile = expected_indices as u64,
                terrain_bounds_probe = cell.tight_tile_bounds().is_some(),
                tight_tile_bounds = cell.tight_tile_bounds().unwrap_or(false),
                "headless benchmark cell configured"
            );
            benchmark.phase = BenchmarkPhase::Flushing {
                variant,
                frames_left: VARIANT_FLUSH_FRAMES,
            };
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
            let terrain_view_visible = tile_meshes
                .p2()
                .iter()
                .filter(|visibility| visibility.get())
                .count();
            emit_render_pass_diagnostics(&diagnostics, cell.label);
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
                shadow_quality = crate::rendering::sun_shadow::quality_label_for_budget(cell.cascades),
                shadow_map_size_px = u64::from(crate::rendering::sun_shadow::SHADOW_MAP_SIZE),
                terrain_inspection = u64::from(cell.terrain_inspection),
                terrain_visible = cell.terrain_visible,
                depth_prepass_enabled = cell.depth_prepass,
                terrain_index_step = cell.terrain_index_step as u64,
                terrain_indices_per_tile = tile_index_count_for_benchmark(cell.terrain_index_step)
                    .expect("benchmark variant has a valid tile index step") as u64,
                terrain_bounds_probe = cell.tight_tile_bounds().is_some(),
                tight_tile_bounds = cell.tight_tile_bounds().unwrap_or(false),
                terrain_view_visible = terrain_view_visible as u64,
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

/// Preserve Bevy's settled top-level render-pass diagnostics beside each
/// headless benchmark cell. GPU timestamps are unavailable on Metal, so every
/// event carries an explicit availability bit and the offline reader renders a
/// missing GPU value as `null`, never as a free pass.
fn emit_render_pass_diagnostics(diagnostics: &DiagnosticsStore, variant: &str) {
    for diagnostic in diagnostics.iter() {
        let path = diagnostic.path().as_str();
        let Some(pass) = path
            .strip_prefix("render/")
            .and_then(|path| path.strip_suffix("/elapsed_cpu"))
            .filter(|pass| !pass.contains('/'))
        else {
            continue;
        };
        let Some(cpu_ms) = diagnostic
            .value()
            .filter(|value| value.is_finite() && *value >= 0.0)
        else {
            continue;
        };
        let gpu_path = format!("render/{pass}/elapsed_gpu");
        let gpu_ms = diagnostics
            .iter()
            .find(|candidate| candidate.path().as_str() == gpu_path)
            .and_then(|candidate| candidate.value())
            .filter(|value| value.is_finite() && *value > 0.0);
        info!(
            target: "thalos::diagnostic::perf",
            event = "headless_render_pass",
            variant,
            pass,
            cpu_ms,
            gpu_ms = gpu_ms.unwrap_or(0.0),
            gpu_timing_available = gpu_ms.is_some(),
            "headless render pass diagnostic"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_frame_window_fits_the_shared_ring() {
        assert!(DEFAULT_MEASURE_FRAMES <= thalos_diagnostics_ui::FRAME_HISTORY_LEN);
        assert_eq!(DEFAULT_MEASURE_FRAMES, 240);
    }

    #[test]
    fn benchmark_settles_the_scene_before_configuring_the_first_cell() {
        let benchmark = HeadlessBenchmark::new(BenchmarkConfig {
            mode: BenchmarkMode::Attribution,
            measure_frames: DEFAULT_MEASURE_FRAMES,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_S),
        });

        assert!(matches!(
            benchmark.phase,
            BenchmarkPhase::WaitingForScene { variant: 0 }
        ));
        assert_eq!(VARIANT_FLUSH_FRAMES, 30);
        assert!(VARIANT_FLUSH_FRAMES < SCENE_STABLE_FRAMES);
    }

    #[test]
    fn scene_readiness_requires_exact_settlement_and_stable_nonempty_counts() {
        assert!(benchmark_scene_ready(true, true, true, 1, true));
        assert!(!benchmark_scene_ready(false, true, true, 1, true));
        assert!(!benchmark_scene_ready(true, false, true, 1, true));
        assert!(
            !benchmark_scene_ready(true, true, false, 1, true),
            "coverage is a latch; exact desired leaves may still be landing"
        );
        assert!(!benchmark_scene_ready(true, true, true, 0, true));
        assert!(!benchmark_scene_ready(true, true, true, 1, false));
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

    #[test]
    fn terrain_prepass_axis_has_both_visibility_controls_and_a_bracket() {
        let cells: std::collections::BTreeSet<_> = TERRAIN_PREPASS_VARIANTS
            .iter()
            .map(|variant| (variant.terrain_visible, variant.depth_prepass))
            .collect();
        assert!(cells.contains(&(true, true)));
        assert!(cells.contains(&(false, true)));
        assert!(cells.contains(&(true, false)));
        assert!(cells.contains(&(false, false)));
        assert_eq!(
            TERRAIN_PREPASS_VARIANTS
                .iter()
                .filter(|variant| variant.terrain_visible && variant.depth_prepass)
                .count(),
            2,
            "the production path must bracket the typed axis"
        );
        assert!(
            TERRAIN_PREPASS_VARIANTS
                .iter()
                .all(|variant| variant.terrain_inspection == 4 && variant.cascades == 0)
        );
    }

    #[test]
    fn terrain_index_axis_has_dense_and_coarse_hidden_controls() {
        let cells: std::collections::BTreeSet<_> = TERRAIN_INDEX_VARIANTS
            .iter()
            .map(|variant| (variant.terrain_visible, variant.terrain_index_step))
            .collect();
        assert!(cells.contains(&(true, 1)));
        assert!(cells.contains(&(false, 1)));
        assert!(cells.contains(&(true, 4)));
        assert!(cells.contains(&(false, 4)));
        assert_eq!(
            TERRAIN_INDEX_VARIANTS
                .iter()
                .filter(|variant| variant.terrain_visible && variant.terrain_index_step == 1)
                .count(),
            2,
            "the dense production geometry must bracket the typed axis"
        );
        assert!(TERRAIN_INDEX_VARIANTS.iter().all(|variant| {
            variant.terrain_inspection == 4 && variant.cascades == 0 && !variant.depth_prepass
        }));
    }

    #[test]
    fn terrain_culling_axis_brackets_full_bounds_and_keeps_dense_geometry() {
        assert_eq!(
            TERRAIN_CULLING_VARIANTS
                .iter()
                .filter(
                    |variant| variant.terrain_visible && variant.tight_tile_bounds() == Some(false)
                )
                .count(),
            2
        );
        assert!(TERRAIN_CULLING_VARIANTS.iter().any(|variant| {
            variant.terrain_visible && variant.tight_tile_bounds() == Some(true)
        }));
        assert!(TERRAIN_CULLING_VARIANTS.iter().any(|variant| {
            !variant.terrain_visible && variant.tight_tile_bounds() == Some(false)
        }));
        assert!(TERRAIN_CULLING_VARIANTS.iter().any(|variant| {
            !variant.terrain_visible && variant.tight_tile_bounds() == Some(true)
        }));
        assert!(TERRAIN_CULLING_VARIANTS.iter().all(|variant| {
            variant.terrain_index_step == 1
                && variant.depth_prepass
                && variant.terrain_inspection == 4
                && variant.cascades == 0
        }));
    }
}
