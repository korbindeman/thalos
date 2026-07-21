//! GPU-memory leak diagnostic (temporary — remove once the leak is pinned).
//!
//! Enabled by setting the `THALOS_MEM_DIAG` env var (any value). Every ~2 s it
//! appends one JSONL line per world to `artifacts/diagnostics/mem_diag.jsonl`. A
//! leak shows up as a monotonically climbing count; the number that
//! *stays flat* localizes the category:
//!
//! The asset/entity counts need no build feature. For real wgpu driver-object
//! counters, also launch with `--features gpu-counters`; the feature is opt-in
//! so normal game/screenshot/preview builds share one Bevy/wgpu dylib.
//!
//!   - `render_meshes` climbs while `main_meshes` stays flat
//!         → the churned `RenderAssetUsages::RENDER_WORLD` scatter meshes
//!           (grass / trees / GPU-grass tiles) are not having their GPU copies
//!           freed on despawn. (Main-world count stays flat because RENDER_WORLD
//!           assets are dropped from the main world right after extraction.)
//!   - `render_images` climbs
//!         → image / texture leak (env-map prefilter, an atlas, a render target).
//!   - both counts flat but VRAM still climbs
//!         → the leak is below the asset layer (pipeline-cache specialization,
//!           mesh-allocator slabs, or a wgpu resource a custom pass retains).
//!
//! The two worlds write to the same file; each line is self-contained and
//! tagged with `"world"`, so interleaving is harmless — sort/grep by world.

use std::io::Write;
use std::sync::LazyLock;

use bevy::prelude::*;
use bevy::render::mesh::RenderMesh;
use bevy::render::render_asset::RenderAssets;
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};

/// Output filename under the canonical diagnostics directory.
const LOG_FILENAME: &str = "mem_diag.jsonl";
/// Log cadence in frames (~2 s at 60 fps). Both worlds keep their own counter.
const LOG_EVERY_FRAMES: u64 = 120;

/// `THALOS_NO_SCATTER=1` parks the tree/shrub + rock scatter drivers (grass has
/// its own Settings→Graphics toggle). A bisection knob for the GPU-memory leak:
/// with grass already off, if VRAM stops climbing when this parks trees + rocks
/// too, the continuous churn of tree/rock tile meshes through the mesh allocator
/// is the source. Checked once (cached).
pub fn scatter_killed() -> bool {
    static KILLED: LazyLock<bool> = LazyLock::new(|| std::env::var("THALOS_NO_SCATTER").is_ok());
    *KILLED
}

fn append_line(line: &str) {
    let path = crate::artifact_paths::default_jsonl_path(LOG_FILENAME);
    if let Ok(mut f) = crate::artifact_paths::open_jsonl_append(&path) {
        let _ = writeln!(f, "{line}");
    }
}

pub struct MemDiagPlugin;

impl Plugin for MemDiagPlugin {
    fn build(&self, app: &mut App) {
        // Compiled always; active only when the env var is set, so a normal run
        // never touches the disk.
        if std::env::var("THALOS_MEM_DIAG").is_err() {
            return;
        }
        let path = crate::artifact_paths::default_jsonl_path(LOG_FILENAME);
        info!(target: "thalos::mem", "GPU-memory diagnostic ON → {}", path.display());
        app.add_systems(Update, log_main_assets);
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(Render, log_render_assets.in_set(RenderSystems::Cleanup));
        }
    }
}

/// Main-world asset + entity counts.
fn log_main_assets(
    mut frame: Local<u64>,
    time: Res<Time>,
    meshes: Res<Assets<Mesh>>,
    images: Res<Assets<Image>>,
    mesh_entities: Query<(), With<Mesh3d>>,
) {
    *frame += 1;
    if *frame % LOG_EVERY_FRAMES != 0 {
        return;
    }
    let line = format!(
        "{{\"world\":\"main\",\"t_s\":{:.1},\"frame\":{},\"main_meshes\":{},\"main_images\":{},\"mesh_entities\":{}}}",
        time.elapsed_secs(),
        *frame,
        meshes.len(),
        images.len(),
        mesh_entities.iter().count(),
    );
    append_line(&line);
}

/// Render-world GPU asset + driver-object counts.
///
/// `render_meshes` / `render_images` are the Bevy-asset layer. `wgpu_*` are the
/// wgpu driver-object counters (buffers, textures, and device memory
/// allocations) — the layer *below* assets. If `wgpu_buffers` /
/// `wgpu_mem_allocs` climb while `render_meshes` stays bounded, the leak is the
/// `MeshAllocator` growing GPU buffer slabs from continuous variable-size scatter
/// mesh churn (slabs free only when fully empty). The wgpu counters read 0 unless
/// wgpu's `counters` feature is active (`thalos_game/gpu-counters`) — a 0 line
/// just means "use the external VRAM meter instead".
fn log_render_assets(
    mut frame: Local<u64>,
    meshes: Res<RenderAssets<RenderMesh>>,
    images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
) {
    *frame += 1;
    if *frame % LOG_EVERY_FRAMES != 0 {
        return;
    }
    let c = device.wgpu_device().get_internal_counters();
    let line = format!(
        "{{\"world\":\"render\",\"frame\":{},\"render_meshes\":{},\"render_images\":{},\
\"wgpu_buffers\":{},\"wgpu_textures\":{},\"wgpu_mem_allocs\":{}}}",
        *frame,
        meshes.iter().count(),
        images.iter().count(),
        c.hal.buffers.read(),
        c.hal.textures.read(),
        c.hal.memory_allocations.read(),
    );
    append_line(&line);
}
