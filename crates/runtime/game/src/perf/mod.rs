//! Always-on lightweight performance telemetry.
//!
//! One collector ([`PerfSamples`]) feeds three consumers:
//!
//! - the **F3 debug view** ([`overlay`]) — live text stats + graph shaders;
//! - the **perf lane** of `artifacts/diagnostics/runtime.jsonl` — a 2 s
//!   `frame_gauge` aggregate plus per-frame `spike` dumps around stutters
//!   (target `thalos::diagnostic::perf`);
//! - an opt-in **full-rate recording** (`THALOS_PERF_RECORD=1`) that emits
//!   one `frame_block` per second carrying every frame's CPU/GPU ms, for
//!   runs where the offline report should show the complete timeline.
//!
//! Deep profiling stays Tracy / `profile-chrome`; this layer is the cheap
//! always-available tier that tells you when to reach for those. Offline
//! rendering of the recorded lane: `just perf-report` (tools/perfreport).

pub mod gpu_images;
pub mod overlay;

use std::time::Instant;

use bevy::diagnostic::{DiagnosticsStore, EntityCountDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::diagnostic::{MeshAllocatorDiagnosticPlugin, RenderDiagnosticsPlugin};

use crate::SimStage;

/// Frame-history ring length: ~8.5 s at 60 fps. Also the window the graph
/// shader displays and the pre-context a spike dump carries.
pub const RING_LEN: usize = 512;
/// Memory-history ring length; sampled every [`MEM_SAMPLE_EVERY_FRAMES`]
/// frames, so 256 entries ≈ 2 minutes — long enough to see a leak's slope.
pub const MEM_RING_LEN: usize = 256;
/// Cadence of memory-ring samples and gauge refreshes (~0.5 s at 60 fps).
const MEM_SAMPLE_EVERY_FRAMES: u64 = 30;
/// Cadence of the aggregated `frame_gauge` JSONL event (~2 s at 60 fps).
const GAUGE_EVERY_FRAMES: u64 = 120;
/// Full-rate recording emits one `frame_block` event per this many frames.
const BLOCK_FRAMES: usize = 60;
/// A spike dump waits this many frames after the trigger so the dump carries
/// post-context, not just the run-up.
const SPIKE_POST_FRAMES: u32 = 60;
/// Minimum frames between spike dumps (~5 s): a sustained stutter storm
/// produces a few representative dumps, not megabytes of them.
const SPIKE_COOLDOWN_FRAMES: u64 = 300;

/// One authority for "how is this process performing right now".
///
/// Written only by [`collect_frame`] / [`sample_gauges`]; the F3 overlay and
/// the JSONL recorder are readers.
#[derive(Resource)]
pub struct PerfSamples {
    /// Index the next frame sample is written to.
    head: usize,
    /// Number of valid samples (saturates at [`RING_LEN`]).
    filled: usize,
    cpu_ms: [f32; RING_LEN],
    gpu_ms: [f32; RING_LEN],

    /// Wall time of the SimStage regions last frame (upper bounds: the
    /// parallel executor may interleave unrelated systems into a region).
    pub stage_physics_ms: f32,
    pub stage_sync_ms: f32,
    pub stage_camera_ms: f32,

    /// Latest slow gauges (refreshed every [`MEM_SAMPLE_EVERY_FRAMES`]).
    pub main_meshes: u32,
    pub main_images: u32,
    pub tile_resident: u32,
    pub tile_mib: f32,
    /// Estimated GPU bytes of all `GpuImage` asset textures
    /// ([`gpu_images::GpuImageBytesDiagnosticPlugin`]). Asset textures only —
    /// render targets and pass-owned textures are not render assets.
    pub texture_mib: f32,
    /// Whole-process resident set. The total the GPU-side gauges must be read
    /// against: a host was killed at 8.1 GiB RSS while tile + slab summed to
    /// ~2 GiB, and nothing measured the other six (2026-07-29, massif-aerial).
    pub rss_mib: f32,
    /// CPU bytes held by main-world `Assets<Mesh>` (vertex + index buffers).
    pub mesh_cpu_mib: f32,
    /// CPU bytes held by main-world `Assets<Image>` pixel data.
    pub image_cpu_mib: f32,

    mem_head: usize,
    mem_filled: usize,
    tile_mib_ring: [f32; MEM_RING_LEN],
    slab_mib_ring: [f32; MEM_RING_LEN],

    /// Rolling median of the last gauge window; 0 until first computed.
    /// Drives the spike threshold.
    median_cpu_ms: f32,
}

impl Default for PerfSamples {
    fn default() -> Self {
        Self {
            head: 0,
            filled: 0,
            cpu_ms: [0.0; RING_LEN],
            gpu_ms: [0.0; RING_LEN],
            stage_physics_ms: 0.0,
            stage_sync_ms: 0.0,
            stage_camera_ms: 0.0,
            main_meshes: 0,
            main_images: 0,
            tile_resident: 0,
            tile_mib: 0.0,
            texture_mib: 0.0,
            rss_mib: 0.0,
            mesh_cpu_mib: 0.0,
            image_cpu_mib: 0.0,
            mem_head: 0,
            mem_filled: 0,
            tile_mib_ring: [0.0; MEM_RING_LEN],
            slab_mib_ring: [0.0; MEM_RING_LEN],
            median_cpu_ms: 0.0,
        }
    }
}

impl PerfSamples {
    fn push_frame(&mut self, cpu_ms: f32, gpu_ms: f32) {
        self.cpu_ms[self.head] = cpu_ms;
        self.gpu_ms[self.head] = gpu_ms;
        self.head = (self.head + 1) % RING_LEN;
        self.filled = (self.filled + 1).min(RING_LEN);
    }

    fn push_mem(&mut self, tile_mib: f32, slab_mib: f32) {
        self.tile_mib_ring[self.mem_head] = tile_mib;
        self.slab_mib_ring[self.mem_head] = slab_mib;
        self.mem_head = (self.mem_head + 1) % MEM_RING_LEN;
        self.mem_filled = (self.mem_filled + 1).min(MEM_RING_LEN);
    }

    pub fn frame_count(&self) -> usize {
        self.filled
    }

    /// The `n` most recent frame samples, oldest first. `n` is clamped to the
    /// available history.
    pub fn recent(&self, n: usize) -> impl Iterator<Item = (f32, f32)> + '_ {
        let n = n.min(self.filled);
        (0..n).map(move |i| {
            let idx = (self.head + RING_LEN - n + i) % RING_LEN;
            (self.cpu_ms[idx], self.gpu_ms[idx])
        })
    }

    /// The `n` most recent memory samples (tile MiB, slab MiB), oldest first.
    pub fn recent_mem(&self, n: usize) -> impl Iterator<Item = (f32, f32)> + '_ {
        let n = n.min(self.mem_filled);
        (0..n).map(move |i| {
            let idx = (self.mem_head + MEM_RING_LEN - n + i) % MEM_RING_LEN;
            (self.tile_mib_ring[idx], self.slab_mib_ring[idx])
        })
    }

    /// Seed the slow gauges with representative values.
    ///
    /// **Preview and test use only** — it is not a second writer of the live
    /// resource ([`sample_gauges`] remains the sole one). It exists so a
    /// headless preview can render a readout against plausible numbers instead
    /// of a screen full of zeros, which would be evidence about nothing.
    pub fn seed_gauges(
        &mut self,
        main_meshes: u32,
        main_images: u32,
        tile_resident: u32,
        tile_mib: f32,
        slab_mib: f32,
        texture_mib: f32,
        rss_mib: f32,
        mesh_cpu_mib: f32,
        image_cpu_mib: f32,
    ) {
        self.main_meshes = main_meshes;
        self.main_images = main_images;
        self.tile_resident = tile_resident;
        self.tile_mib = tile_mib;
        self.texture_mib = texture_mib;
        self.rss_mib = rss_mib;
        self.mesh_cpu_mib = mesh_cpu_mib;
        self.image_cpu_mib = image_cpu_mib;
        self.push_mem(tile_mib, slab_mib);
    }

    /// Latest slab size gauge (mirrors the last memory-ring sample).
    pub fn slab_mib(&self) -> f32 {
        if self.mem_filled == 0 {
            return 0.0;
        }
        self.slab_mib_ring[(self.mem_head + MEM_RING_LEN - 1) % MEM_RING_LEN]
    }
}

/// Instants marking the SimStage region boundaries within one `Update` run.
/// Written by the four `mark_*` systems, folded into [`PerfSamples`] by the
/// last one.
#[derive(Resource, Default)]
struct StageMarks {
    frame_start: Option<Instant>,
    physics_end: Option<Instant>,
    sync_end: Option<Instant>,
}

/// Recorder bookkeeping: frame counter, spike state, full-rate buffers.
#[derive(Resource)]
struct PerfRecorder {
    frame: u64,
    /// Frames remaining until a triggered spike dump is emitted (post-context
    /// countdown). 0 = no dump pending.
    spike_countdown: u32,
    /// Peak frame ms observed since the pending dump was triggered.
    spike_peak_ms: f32,
    last_spike_frame: u64,
    /// `THALOS_PERF_RECORD` was set: emit per-frame `frame_block` events.
    full_rate: bool,
}

impl Default for PerfRecorder {
    fn default() -> Self {
        Self {
            frame: 0,
            spike_countdown: 0,
            spike_peak_ms: 0.0,
            last_spike_frame: 0,
            full_rate: std::env::var_os("THALOS_PERF_RECORD").is_some(),
        }
    }
}

pub struct PerfPlugin;

impl Plugin for PerfPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PerfSamples>()
            .init_resource::<StageMarks>()
            .init_resource::<PerfRecorder>()
            .add_plugins(EntityCountDiagnosticsPlugin::default())
            .add_plugins(MeshAllocatorDiagnosticPlugin)
            .add_plugins(gpu_images::GpuImageBytesDiagnosticPlugin)
            // GPU pass timings (`render/<pass>/elapsed_gpu` in the
            // DiagnosticsStore). Cheap timestamp queries; previously only the
            // headless capture app added this.
            .add_plugins(RenderDiagnosticsPlugin)
            .add_systems(
                Update,
                (
                    mark_frame_start.before(SimStage::Physics),
                    mark_physics_end
                        .after(SimStage::Physics)
                        .before(SimStage::Sync),
                    mark_sync_end.after(SimStage::Sync).before(SimStage::Camera),
                    fold_stage_marks.after(SimStage::Camera),
                ),
            )
            .add_systems(PostUpdate, (collect_frame, sample_gauges, record).chain())
            .add_plugins(overlay::DebugViewPlugin);
    }
}

fn mark_frame_start(mut marks: ResMut<StageMarks>) {
    marks.frame_start = Some(Instant::now());
}

fn mark_physics_end(mut marks: ResMut<StageMarks>) {
    marks.physics_end = Some(Instant::now());
}

fn mark_sync_end(mut marks: ResMut<StageMarks>) {
    marks.sync_end = Some(Instant::now());
}

fn fold_stage_marks(mut marks: ResMut<StageMarks>, mut samples: ResMut<PerfSamples>) {
    let now = Instant::now();
    if let (Some(t0), Some(t1), Some(t2)) = (marks.frame_start, marks.physics_end, marks.sync_end) {
        samples.stage_physics_ms = (t1 - t0).as_secs_f32() * 1000.0;
        samples.stage_sync_ms = (t2 - t1).as_secs_f32() * 1000.0;
        samples.stage_camera_ms = (now - t2).as_secs_f32() * 1000.0;
    }
    *marks = StageMarks::default();
}

/// Sum of the top-level `render/<pass>/elapsed_gpu` diagnostics, in ms.
///
/// Only 3-component paths are summed: nested pass spans would double-count
/// their parents. Values lag a frame or two behind (GPU timestamp readback).
pub fn gpu_frame_ms(store: &DiagnosticsStore) -> f32 {
    let mut total = 0.0f64;
    for diag in store.iter() {
        let path = diag.path().as_str();
        if !path.starts_with("render/") || !path.ends_with("/elapsed_gpu") {
            continue;
        }
        if path.bytes().filter(|&b| b == b'/').count() != 2 {
            continue;
        }
        if let Some(v) = diag.value()
            && v.is_finite()
        {
            total += v;
        }
    }
    total as f32
}

/// Frame cost is a **wall-clock** quantity, so it is measured with `Instant`
/// like the stage marks above — never from `Time<Real>`, which the offline
/// render drives to a fixed step (`sim_clock::SimClockDrive`). Under a driven
/// clock this gauge would otherwise report a flat 16.67 ms while frames
/// genuinely took 300 ms, and the perf lane would be confidently wrong.
fn collect_frame(
    store: Res<DiagnosticsStore>,
    mut previous: Local<Option<Instant>>,
    mut samples: ResMut<PerfSamples>,
) {
    let now = Instant::now();
    let Some(last) = previous.replace(now) else {
        return; // first frame
    };
    let cpu_ms = (now - last).as_secs_f32() * 1000.0;
    if cpu_ms <= 0.0 {
        return;
    }
    let gpu_ms = gpu_frame_ms(&store);
    samples.push_frame(cpu_ms, gpu_ms);
}

fn sample_gauges(
    recorder: Res<PerfRecorder>,
    store: Res<DiagnosticsStore>,
    meshes: Res<Assets<Mesh>>,
    images: Res<Assets<Image>>,
    tile_roots: Query<&thalos_body_render::tiles::TileTerrainRoot>,
    mut samples: ResMut<PerfSamples>,
) {
    if !recorder.frame.is_multiple_of(MEM_SAMPLE_EVERY_FRAMES) {
        return;
    }
    samples.main_meshes = meshes.len() as u32;
    samples.main_images = images.len() as u32;
    samples.tile_resident = tile_roots.iter().map(|r| r.resident_count()).sum::<usize>() as u32;
    let tile_mib =
        tile_roots.iter().map(|r| r.resident_bytes()).sum::<usize>() as f32 / (1024.0 * 1024.0);
    samples.tile_mib = tile_mib;
    samples.texture_mib = store
        .get(gpu_images::GpuImageBytesDiagnosticPlugin::diagnostic_path())
        .and_then(|d| d.value())
        .unwrap_or(0.0) as f32
        / (1024.0 * 1024.0);
    // CPU-side accounting, so `rss_mib` growth can be attributed rather than
    // guessed at. Byte sums over borrowed buffers — no allocation.
    let mib = |bytes: usize| bytes as f32 / (1024.0 * 1024.0);
    samples.rss_mib = thalos_diagnostics::process::self_resident_bytes()
        .map(|b| mib(b as usize))
        .unwrap_or(0.0);
    // `try_*`, not the plain accessors: a RENDER_WORLD-only mesh has had its
    // CPU data moved out at extraction, and the plain accessors panic on it.
    // Counting it as 0 is the truth this gauge wants — its CPU copy is gone.
    samples.mesh_cpu_mib = mib(meshes
        .iter()
        .map(|(_, mesh)| {
            let vertex_bytes = mesh
                .try_attributes()
                .map(|attributes| {
                    attributes
                        .map(|(_, values)| values.get_bytes().len())
                        .sum::<usize>()
                })
                .unwrap_or(0);
            let index_bytes = match mesh.try_indices_option() {
                Ok(Some(bevy::render::mesh::Indices::U16(v))) => v.len() * 2,
                Ok(Some(bevy::render::mesh::Indices::U32(v))) => v.len() * 4,
                _ => 0,
            };
            vertex_bytes + index_bytes
        })
        .sum());
    samples.image_cpu_mib = mib(images
        .iter()
        .map(|(_, image)| image.data.as_ref().map_or(0, Vec::len))
        .sum());
    let slab_mib = store
        .get(MeshAllocatorDiagnosticPlugin::slabs_size_diagnostic_path())
        .and_then(|d| d.value())
        .unwrap_or(0.0) as f32
        / (1024.0 * 1024.0);
    samples.push_mem(tile_mib, slab_mib);
}

/// `"820 MiB"` / `"3.2 GiB"` — one unit switch so a five-digit MiB figure never
/// pushes a readout column out of line.
///
/// Shared by the F3 debug view and the loading screen so the same quantity is
/// never spelled two ways on two screens.
pub fn fmt_mib(mib: f32) -> String {
    if mib >= 1024.0 {
        format!("{:.1} GiB", mib / 1024.0)
    } else {
        format!("{mib:.0} MiB")
    }
}

/// [`fmt_mib`] for a byte count.
pub fn fmt_bytes(bytes: u64) -> String {
    fmt_mib(bytes as f32 / (1024.0 * 1024.0))
}

/// Read the entity-count diagnostic (registered by
/// [`EntityCountDiagnosticsPlugin`]) out of the store.
pub fn entity_count(store: &DiagnosticsStore) -> u64 {
    store
        .get(&EntityCountDiagnosticsPlugin::ENTITY_COUNT)
        .and_then(|d| d.value())
        .unwrap_or(0.0) as u64
}

/// Format ring samples as a compact comma-joined string field for JSONL
/// (arrays of numbers aren't a native tracing field type; one short string
/// keeps a 180-frame dump to ~1.3 KB).
fn join_ms(values: impl Iterator<Item = f32>) -> String {
    let mut out = String::with_capacity(8 * 64);
    for (i, v) in values.enumerate() {
        if i > 0 {
            out.push(',');
        }
        // 2 decimals ≈ 10 µs resolution — plenty for frame times.
        out.push_str(&format!("{v:.2}"));
    }
    out
}

fn record(
    store: Res<DiagnosticsStore>,
    mut samples: ResMut<PerfSamples>,
    mut recorder: ResMut<PerfRecorder>,
) {
    recorder.frame += 1;
    let frame = recorder.frame;

    let latest_cpu_ms = samples.recent(1).next().map(|(cpu, _)| cpu).unwrap_or(0.0);

    // ── Tier A: aggregated gauge every ~2 s ─────────────────────────────
    if frame.is_multiple_of(GAUGE_EVERY_FRAMES) && samples.frame_count() > 0 {
        let window = GAUGE_EVERY_FRAMES as usize;
        let mut cpu: Vec<f32> = samples.recent(window).map(|(c, _)| c).collect();
        let gpu_mean = samples.recent(window).map(|(_, g)| g).sum::<f32>() / cpu.len() as f32;
        cpu.sort_by(|a, b| a.total_cmp(b));
        let n = cpu.len();
        let mean = cpu.iter().sum::<f32>() / n as f32;
        let median = cpu[n / 2];
        let p95 = cpu[(n * 95 / 100).min(n - 1)];
        let max = cpu[n - 1];
        samples.median_cpu_ms = median;
        info!(
            target: "thalos::diagnostic::perf",
            event = "frame_gauge",
            fps = f64::from(1000.0 / mean.max(1e-3)),
            cpu_ms_mean = f64::from(mean),
            cpu_ms_p50 = f64::from(median),
            cpu_ms_p95 = f64::from(p95),
            cpu_ms_max = f64::from(max),
            gpu_ms_mean = f64::from(gpu_mean),
            physics_ms = f64::from(samples.stage_physics_ms),
            sync_ms = f64::from(samples.stage_sync_ms),
            camera_ms = f64::from(samples.stage_camera_ms),
            entities = entity_count(&store),
            main_meshes = u64::from(samples.main_meshes),
            main_images = u64::from(samples.main_images),
            tile_resident = u64::from(samples.tile_resident),
            tile_mib = f64::from(samples.tile_mib),
            slab_mib = f64::from(samples.slab_mib()),
            texture_mib = f64::from(samples.texture_mib),
            rss_mib = f64::from(samples.rss_mib),
            mesh_cpu_mib = f64::from(samples.mesh_cpu_mib),
            image_cpu_mib = f64::from(samples.image_cpu_mib),
            "perf frame gauge"
        );
    }

    // ── Tier B: spike dumps ─────────────────────────────────────────────
    // Armed only once the ring is warm and a median exists, so boot frames
    // and loading hitches before steady state don't spam dumps.
    if recorder.spike_countdown > 0 {
        recorder.spike_peak_ms = recorder.spike_peak_ms.max(latest_cpu_ms);
        recorder.spike_countdown -= 1;
        if recorder.spike_countdown == 0 {
            let dump_len = 180usize.min(samples.frame_count());
            let cpu_str = join_ms(samples.recent(dump_len).map(|(c, _)| c));
            let gpu_str = join_ms(samples.recent(dump_len).map(|(_, g)| g));
            info!(
                target: "thalos::diagnostic::perf",
                event = "spike",
                spike_ms = f64::from(recorder.spike_peak_ms),
                median_ms = f64::from(samples.median_cpu_ms),
                // The spike sits SPIKE_POST_FRAMES from the end of the arrays.
                post_frames = u64::from(SPIKE_POST_FRAMES),
                cpu_ms = cpu_str.as_str(),
                gpu_ms = gpu_str.as_str(),
                "frame spike"
            );
            recorder.last_spike_frame = frame;
        }
    } else if samples.frame_count() == RING_LEN
        && samples.median_cpu_ms > 0.0
        && frame.saturating_sub(recorder.last_spike_frame) > SPIKE_COOLDOWN_FRAMES
    {
        let threshold = (3.0 * samples.median_cpu_ms).max(50.0);
        if latest_cpu_ms > threshold {
            recorder.spike_countdown = SPIKE_POST_FRAMES;
            recorder.spike_peak_ms = latest_cpu_ms;
        }
    }

    // ── Full-rate blocks (opt-in) ───────────────────────────────────────
    if recorder.full_rate
        && frame.is_multiple_of(BLOCK_FRAMES as u64)
        && samples.frame_count() >= BLOCK_FRAMES
    {
        let cpu_str = join_ms(samples.recent(BLOCK_FRAMES).map(|(c, _)| c));
        let gpu_str = join_ms(samples.recent(BLOCK_FRAMES).map(|(_, g)| g));
        info!(
            target: "thalos::diagnostic::perf",
            event = "frame_block",
            frames = BLOCK_FRAMES as u64,
            cpu_ms = cpu_str.as_str(),
            gpu_ms = gpu_str.as_str(),
            "full-rate frame block"
        );
    }
}
