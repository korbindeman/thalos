//! GPU texture-memory gauge: estimated bytes held by `RenderAssets<GpuImage>`.
//!
//! Bevy's own `RenderAssetDiagnosticPlugin` counts *assets*; nothing in the
//! tree measured their *bytes*, which left every asset texture — terrain
//! package charts, vegetation atlases, cloud LUTs — inside the VRAM bar's
//! unattributed "other" segment. This plugin closes that gap with the same
//! shape as `MeshAllocatorDiagnosticPlugin`: a render-world system stores the
//! figure into an atomic each sample, and a `PreUpdate` system publishes it to
//! the main-world [`DiagnosticsStore`] where `sample_gauges` reads it.
//!
//! **Estimated**, not driver-reported: the sum over each texture's descriptor
//! (mip chain, array layers, block-compressed formats) of its tightly-packed
//! size. The driver may pad or page-align individual allocations, so the true
//! figure is slightly higher — close enough to attribute a segment, not a
//! substitute for the whole-card reading.
//!
//! **Asset textures only.** Render targets, shadow maps, the swapchain, and
//! textures custom passes create straight on the device are not render assets
//! and stay in "other" — which is exactly what that segment now means for this
//! process: non-asset GPU memory.

use bevy::diagnostic::{Diagnostic, DiagnosticPath, Diagnostics, RegisterDiagnostic};
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::texture::GpuImage;
use bevy::render::{Extract, ExtractSchedule, RenderApp};
use std::sync::atomic::{AtomicU64, Ordering};

/// Estimated bytes of all `GpuImage` textures, in the main-world store.
static GPU_IMAGE_BYTES: DiagnosticPath = DiagnosticPath::const_new("gpu_image_bytes");

/// Sampling cadence in render frames (~0.5 s at 60 fps, matching the
/// `MEM_SAMPLE_EVERY_FRAMES` cadence of the consumer gauge). Walking a few
/// hundred descriptors is cheap, but it is a periodic gauge, not per-frame
/// work.
const SAMPLE_EVERY_FRAMES: u64 = 30;

pub struct GpuImageBytesDiagnosticPlugin;

impl GpuImageBytesDiagnosticPlugin {
    pub fn diagnostic_path() -> &'static DiagnosticPath {
        &GPU_IMAGE_BYTES
    }
}

impl Plugin for GpuImageBytesDiagnosticPlugin {
    fn build(&self, app: &mut App) {
        app.register_diagnostic(Diagnostic::new(GPU_IMAGE_BYTES.clone()).with_suffix(" bytes"))
            .init_resource::<GpuImageBytesMeasurement>()
            .add_systems(PreUpdate, publish_measurement);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(ExtractSchedule, measure_gpu_images);
        }
    }
}

/// Main-world mailbox the render world stores into (the
/// `MeshAllocatorDiagnosticPlugin` pattern).
#[derive(Resource, Default)]
struct GpuImageBytesMeasurement {
    bytes: AtomicU64,
}

fn publish_measurement(
    mut diagnostics: Diagnostics,
    measurement: Res<GpuImageBytesMeasurement>,
) {
    diagnostics.add_measurement(&GPU_IMAGE_BYTES, || {
        measurement.bytes.load(Ordering::Relaxed) as f64
    });
}

fn measure_gpu_images(
    mut frame: Local<u64>,
    measurement: Extract<Res<GpuImageBytesMeasurement>>,
    images: Res<RenderAssets<GpuImage>>,
) {
    *frame += 1;
    if !frame.is_multiple_of(SAMPLE_EVERY_FRAMES) {
        return;
    }
    let total: u64 = images
        .iter()
        .map(|(_, image)| {
            let t = &image.texture;
            texture_bytes(
                t.size(),
                t.dimension(),
                t.format(),
                t.mip_level_count(),
                t.sample_count(),
            )
        })
        .sum();
    measurement.bytes.store(total, Ordering::Relaxed);
}

/// Tightly-packed size of a texture: every mip of every layer, in the
/// format's block units, times the sample count.
fn texture_bytes(
    size: Extent3d,
    dimension: TextureDimension,
    format: TextureFormat,
    mip_level_count: u32,
    sample_count: u32,
) -> u64 {
    let (block_w, block_h) = format.block_dimensions();
    // Depth-stencil and multi-planar formats report no single copy size;
    // 4 B/texel is the right order for every such format wgpu exposes, and
    // none of them appear as image assets in practice.
    let block_bytes = format.block_copy_size(None).unwrap_or(4);
    let mut total = 0u64;
    for mip in 0..mip_level_count {
        let w = (size.width >> mip).max(1);
        let h = (size.height >> mip).max(1);
        // 3D depth shrinks with the mip chain; array layers do not.
        let d = match dimension {
            TextureDimension::D3 => (size.depth_or_array_layers >> mip).max(1),
            _ => size.depth_or_array_layers,
        };
        total += u64::from(w.div_ceil(block_w))
            * u64::from(h.div_ceil(block_h))
            * u64::from(d)
            * u64::from(block_bytes);
    }
    total * u64::from(sample_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> Extent3d {
        Extent3d {
            width,
            height,
            depth_or_array_layers,
        }
    }

    #[test]
    fn rgba8_mip_chain_sums_every_level() {
        // 4×4 + 2×2 + 1×1 texels at 4 B each.
        assert_eq!(
            texture_bytes(
                extent(4, 4, 1),
                TextureDimension::D2,
                TextureFormat::Rgba8UnormSrgb,
                3,
                1,
            ),
            (16 + 4 + 1) * 4
        );
    }

    #[test]
    fn bc7_counts_blocks_not_texels() {
        // 2×2 blocks of 4×4 texels, 16 B per block.
        assert_eq!(
            texture_bytes(
                extent(8, 8, 1),
                TextureDimension::D2,
                TextureFormat::Bc7RgbaUnormSrgb,
                1,
                1,
            ),
            4 * 16
        );
    }

    #[test]
    fn array_layers_do_not_shrink_with_mips() {
        // Mip 0: 2×2×6, mip 1: 1×1×6 — layers constant.
        assert_eq!(
            texture_bytes(
                extent(2, 2, 6),
                TextureDimension::D2,
                TextureFormat::Rgba8Unorm,
                2,
                1,
            ),
            (4 * 6 + 6) * 4
        );
    }

    #[test]
    fn volume_depth_shrinks_with_mips() {
        // 4×4×4 + 2×2×2 + 1×1×1 texels at 1 B.
        assert_eq!(
            texture_bytes(
                extent(4, 4, 4),
                TextureDimension::D3,
                TextureFormat::R8Unorm,
                3,
                1,
            ),
            64 + 8 + 1
        );
    }
}
