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
use bevy::render::mesh::allocator::{MeshAllocator, MeshAllocatorSettings};
use bevy::render::render_asset::RenderAssets;
use bevy::render::renderer::RenderDevice;
use bevy::render::slab_allocator::SlabAllocatorSettings;
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
    let path = thalos_diagnostics::paths::default_jsonl_path(LOG_FILENAME);
    if let Ok(mut f) = thalos_diagnostics::paths::open_jsonl_append(&path) {
        let _ = writeln!(f, "{line}");
    }
}

pub struct MemDiagPlugin;

impl Plugin for MemDiagPlugin {
    fn build(&self, app: &mut App) {
        // The mesh-slab gauge is **always on**. It is the one number that
        // separates "the ground is over budget" from "the ground is fine and
        // GPU memory is going somewhere else", and needing to know to set an env
        // var *before* the run is precisely why INC-20260725T012104Z could not
        // be decided from its own crash log. It costs one `len()` and one sum
        // over a handful of slabs every ~10 s.
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(Render, log_mesh_slab_gauge.in_set(RenderSystems::Cleanup));
        }
        apply_slab_cap_override(app);

        // The JSONL file, by contrast, stays behind the env var so a normal run
        // never touches the disk.
        if std::env::var("THALOS_MEM_DIAG").is_err() {
            return;
        }
        let path = thalos_diagnostics::paths::default_jsonl_path(LOG_FILENAME);
        info!(target: "thalos::mem", "GPU-memory diagnostic ON → {}", path.display());
        app.add_systems(Update, log_main_assets);
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(Render, log_render_assets.in_set(RenderSystems::Cleanup));
        }
    }
}

/// Main-world asset + entity counts, plus tile-terrain residency.
///
/// `tile_resident` / `tile_mib` are the ground's own accounted VRAM. Reading
/// them *against* `render_meshes` is what separates the two failure modes: if
/// tile residency plateaus while `render_meshes` and `wgpu_buffers` keep
/// climbing, the ground is inside its budget and the growth is elsewhere
/// (scatter churn, or slabs pinned by stragglers).
fn log_main_assets(
    mut frame: Local<u64>,
    time: Res<Time>,
    meshes: Res<Assets<Mesh>>,
    images: Res<Assets<Image>>,
    mesh_entities: Query<(), With<Mesh3d>>,
    tile_roots: Query<&thalos_body_render::tiles::TileTerrainRoot>,
) {
    *frame += 1;
    if *frame % LOG_EVERY_FRAMES != 0 {
        return;
    }
    let tile_resident: usize = tile_roots.iter().map(|r| r.resident_count()).sum();
    let tile_bytes: usize = tile_roots.iter().map(|r| r.resident_bytes()).sum();
    // One root today (the tile driver takes one body per session); summing keeps
    // the line correct when per-body install lands.
    let split_scale = tile_roots
        .iter()
        .map(|r| r.split_scale())
        .fold(1.0, f64::min);
    let line = format!(
        "{{\"world\":\"main\",\"t_s\":{:.1},\"frame\":{},\"main_meshes\":{},\"main_images\":{},\
\"mesh_entities\":{},\"tile_resident\":{},\"tile_mib\":{:.1},\"tile_split_scale\":{:.3}}}",
        time.elapsed_secs(),
        *frame,
        meshes.len(),
        images.len(),
        mesh_entities.iter().count(),
        tile_resident,
        tile_bytes as f64 / (1024.0 * 1024.0),
        split_scale,
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
    allocator: Res<MeshAllocator>,
) {
    *frame += 1;
    if *frame % LOG_EVERY_FRAMES != 0 {
        return;
    }
    let c = device.wgpu_device().get_internal_counters();
    let line = format!(
        "{{\"world\":\"render\",\"frame\":{},\"render_meshes\":{},\"render_images\":{},\
\"wgpu_buffers\":{},\"wgpu_textures\":{},\"wgpu_mem_allocs\":{},\
\"mesh_slabs\":{},\"mesh_slab_mib\":{:.1}}}",
        *frame,
        meshes.iter().count(),
        images.iter().count(),
        c.hal.buffers.read(),
        c.hal.textures.read(),
        c.hal.memory_allocations.read(),
        allocator.slab_count(),
        allocator.slabs_size() as f64 / (1024.0 * 1024.0),
    );
    append_line(&line);
}

/// Frames between mesh-slab gauge lines (~10 s at 60 fps).
const SLAB_GAUGE_EVERY_FRAMES: u64 = 600;

/// `THALOS_MESH_SLAB_MB=<n>` caps Bevy's per-slab buffer size (default 512 MiB).
///
/// This is a **falsifier, deliberately not a default**. Bevy frees a mesh slab
/// only when it is fully empty and never shrinks one, so heavy tile churn can in
/// principle strand near-empty 512 MiB buffers — a mechanism that would look
/// exactly like the OOM being chased while every asset count stays flat. Capping
/// the slab size bounds what one stranded slab can cost.
///
/// It is a knob rather than a new default because the hypothesis is *not yet
/// measured*: changing the allocator's shape for every run on the strength of a
/// plausible story is how a symptom disappears without an explanation. Run one
/// session with it set, one without, and read `mesh slab gauge` in each. Smaller
/// slabs also mean more of them, and meshes in different slabs cannot batch into
/// one draw — so if this turns out to be the cause, the value wants choosing
/// against measured frame time, not simply minimising.
fn apply_slab_cap_override(app: &mut App) {
    let Ok(raw) = std::env::var("THALOS_MESH_SLAB_MB") else {
        return;
    };
    let Ok(mb) = raw.trim().parse::<u64>() else {
        warn!("THALOS_MESH_SLAB_MB={raw:?} is not a MiB count; ignoring");
        return;
    };
    if mb == 0 {
        warn!("THALOS_MESH_SLAB_MB=0 is meaningless; ignoring");
        return;
    }
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    // `MeshAllocatorPlugin` already ran `init_resource` in the render app, so
    // this overwrites its default rather than racing it.
    let defaults = SlabAllocatorSettings::default();
    render_app.insert_resource(MeshAllocatorSettings {
        slab_allocator_settings: SlabAllocatorSettings {
            max_slab_size: mb * 1024 * 1024,
            // Keep a large mesh out of a general slab whenever it would fill an
            // outsized share of one; leaving the stock 256 MiB threshold above a
            // small cap would defeat the cap for exactly the big allocations.
            large_threshold: defaults.large_threshold.min(mb * 1024 * 1024),
            ..defaults
        },
        ..default()
    });
    info!(target: "thalos::mem", "mesh slab cap → {mb} MiB (default 512)");
}

/// **Always-on** gauge of what Bevy's [`MeshAllocator`] is actually holding on
/// the GPU.
///
/// `slab_mib` is the real allocation — the sum of every slab *buffer*, not the
/// live mesh bytes inside them. The gap between it and the tile renderer's own
/// `resident_mib` is the number that has been missing from every OOM
/// post-mortem so far, because Bevy frees a slab only when it becomes **fully
/// empty** and never shrinks one: a slab that grew toward its 512 MiB ceiling
/// and then had all but one allocation freed still holds the whole buffer. With
/// terrain tiles churning by the thousand per session across six element
/// layouts (five vertex attributes plus indices, each with its own slab family),
/// retained-but-near-empty slabs are a mechanism that looks exactly like a leak
/// while every asset count stays flat.
///
/// Read it against `tile terrain residency:`. Both flat → the growth is neither
/// the ground nor mesh slabs (look at textures). `slab_mib` climbing while
/// `resident_mib` is flat → this is it, and `MeshAllocatorSettings.max_slab_size`
/// is the lever.
fn log_mesh_slab_gauge(
    mut frame: Local<u64>,
    allocator: Res<MeshAllocator>,
    meshes: Res<RenderAssets<RenderMesh>>,
) {
    *frame += 1;
    if *frame % SLAB_GAUGE_EVERY_FRAMES != 0 {
        return;
    }
    info!(
        target: "thalos::diagnostic::gpu_mem",
        event = "mesh_slab_gauge",
        slabs = allocator.slab_count(),
        slab_mib = allocator.slabs_size() as f64 / (1024.0 * 1024.0),
        render_meshes = meshes.iter().count(),
        "mesh slab gauge"
    );
}
