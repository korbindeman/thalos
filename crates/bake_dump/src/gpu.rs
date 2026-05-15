//! Headless GPU compute context for `bake_dump`.
//!
//! Spins up a minimal wgpu Instance / Adapter / Device / Queue without a
//! window or display surface. Used by per-texel bake stages (the
//! mid-frequency-detail kernel, and future migrations) to dispatch
//! compute shaders rather than running the same arithmetic on the CPU.
//!
//! Lifecycle: construct once at the top of `bake_one`, share across
//! every GPU-aware stage, drop at the end. The wgpu objects are cheap
//! to clone (internally arc'd), so this struct hands out `Arc`s to the
//! device/queue if a stage needs to outlive the borrow.
//!
//! Backend choice: `Backends::all()` lets wgpu pick the best available
//! per-platform — Metal on macOS, Vulkan / DX12 / Vulkan / GL elsewhere.
//! No fallback adapter; if a real GPU isn't available the constructor
//! errors out and the caller decides what to do (today, panic).

use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use thalos_terrain_gen::cubemap::{Cubemap, CubemapFace};
use thalos_terrain_gen::stages::MidFreqDetailParams;
use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferAddress, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, DeviceDescriptor, Extent3d, Features, Instance, InstanceDescriptor,
    MapMode, Origin3d, PipelineLayoutDescriptor, PollType, PowerPreference,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};

/// A live wgpu Instance/Adapter/Device/Queue tuple, owned for the
/// duration of a bake run. Cheap to share by reference.
pub struct GpuContext {
    #[allow(dead_code)] // kept alive for adapter / device lifetime
    instance: Instance,
    #[allow(dead_code)] // exposed for future stages, unused in Phase 1
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize a headless GPU context.
    ///
    /// Picks the system's highest-performance adapter from any available
    /// backend. No window surface; no display required. Returns an
    /// error if no GPU adapter is available — `bake_dump`'s caller
    /// decides how loud to be about that.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .context(
                "no GPU adapter available — `bake_dump` requires a real GPU. On a headless \
                 Linux box install lavapipe (Mesa) and re-run.",
            )?;

        let adapter_info = adapter.get_info();
        // Use the adapter's full reported limits, not the portable
        // downlevel defaults. `bake_dump` runs only on dev / build
        // machines (the player never bakes), so we want all the
        // capability the local GPU offers. The mid-freq cubemap
        // download alone needs a >256 MB staging buffer for a 4096²
        // R32Float texture (384 MB), well above the downlevel
        // 256 MB cap and well within what every actual GPU supports.
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("bake_dump device"),
                required_features: Features::empty(),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .context("failed to acquire GPU device")?;

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    /// Human-readable adapter identifier, e.g.
    /// `"Apple M3 Max (IntegratedGpu, backend=Metal)"`. Used in startup
    /// log lines so a wrong-GPU pick is visible.
    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, backend={:?})",
            self.adapter_info.name, self.adapter_info.device_type, self.adapter_info.backend,
        )
    }
}

// ---------------------------------------------------------------------------
// Smoke test
// ---------------------------------------------------------------------------

/// Run a trivial compute dispatch and verify the result. Validates that
/// adapter / device / queue / shader-compile / dispatch / readback are
/// all wired up before any real stage relies on them. Returns the
/// per-element values written by the shader so the caller can assert
/// what it likes.
///
/// The kernel: a 1D buffer of `N` u32s, each set to its global
/// invocation index. Smallest thing that exercises the full pipeline.
pub fn smoke_test(ctx: &GpuContext, n: u32) -> Result<Vec<u32>> {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let buffer_size = (n as BufferAddress) * std::mem::size_of::<u32>() as BufferAddress;

    // Storage buffer the shader writes into.
    let storage = device.create_buffer(&BufferDescriptor {
        label: Some("smoke storage"),
        size: buffer_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Staging buffer for readback.
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("smoke staging"),
        size: buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("smoke shader"),
        source: ShaderSource::Wgsl(SMOKE_SHADER.into()),
    });

    // Single bind group: one read-write storage buffer at @binding(0).
    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("smoke pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("smoke pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("smoke bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("smoke encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("smoke pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // 64-wide workgroup. Ceil-divide so the last group covers any
        // remainder; the shader bounds-checks against `n`.
        let workgroups = n.div_ceil(64);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, buffer_size);
    queue.submit([encoder.finish()]);

    map_and_read::<u32>(device, &staging, buffer_size)
}

/// Trivial compute kernel for the smoke test. Each invocation writes
/// its global ID into the output buffer.
const SMOKE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&data)) {
        data[i] = i;
    }
}
"#;

/// Block-map a `COPY_DST | MAP_READ` buffer and decode it as a `Vec<T>`.
/// Caller is responsible for sizing — `byte_size` must equal
/// `count * size_of::<T>()` and `T` must be `Pod`.
fn map_and_read<T: bytemuck::Pod>(
    device: &wgpu::Device,
    staging: &Buffer,
    byte_size: BufferAddress,
) -> Result<Vec<T>> {
    let slice = staging.slice(..byte_size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    // Block until the queue catches up to the map fence.
    device
        .poll(PollType::wait_indefinitely())
        .context("device poll failed during staging buffer map")?;
    rx.recv()
        .context("staging buffer map channel closed")?
        .context("staging buffer map failed")?;

    let data = slice.get_mapped_range();
    let out: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Cubemap <-> wgpu::Texture
// ---------------------------------------------------------------------------

/// Pixel size in bytes for `R32Float`. Kept named for readability at the
/// few sites that compute byte strides.
const R32F_BPP: u32 = 4;

/// Upload a `Cubemap<f32>` to a new `R32Float` 2D-array texture with 6
/// layers. Layer order matches `CubemapFace::ALL`: +X, -X, +Y, -Y, +Z,
/// -Z. The texture is usable as a `storage_texture_2d_array<r32float,
/// read_write>` binding in a compute shader.
///
/// Requires the cubemap's row stride (`res * 4` bytes) to be a multiple
/// of 256 — wgpu's buffer→texture copy alignment requirement. Holds for
/// every power-of-two resolution from 64 upward (64 * 4 = 256), which
/// covers every resolution the bake actually produces.
pub fn upload_cubemap_f32(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    cube: &Cubemap<f32>,
) -> Texture {
    let res = cube.resolution();
    assert!(
        (res * R32F_BPP) % 256 == 0,
        "cubemap upload: row stride {} bytes for res {res} is not a multiple of 256",
        res * R32F_BPP,
    );

    let texture = device.create_texture(&TextureDescriptor {
        label: Some("mid-freq cubemap (R32F, D2Array, 6 layers)"),
        size: Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for face in CubemapFace::ALL {
        let face_idx = face as u32;
        let face_data = cube.face_data(face);
        let bytes: &[u8] = bytemuck::cast_slice(face_data);
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: face_idx,
                },
                aspect: TextureAspect::All,
            },
            bytes,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(res * R32F_BPP),
                rows_per_image: Some(res),
            },
            Extent3d {
                width: res,
                height: res,
                depth_or_array_layers: 1,
            },
        );
    }

    texture
}

/// Read an `R32Float` 2D-array texture (6 layers) back into a
/// `Cubemap<f32>`. Layer order matches the upload helper. Blocks on
/// `Queue::submit` until the readback completes.
pub fn download_cubemap_f32(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    texture: &Texture,
) -> Result<Cubemap<f32>> {
    let extent = texture.size();
    let res = extent.width;
    assert_eq!(extent.height, res, "cubemap download: texture is not square");
    assert_eq!(
        extent.depth_or_array_layers, 6,
        "cubemap download: expected 6 array layers"
    );
    assert_eq!(
        texture.format(),
        TextureFormat::R32Float,
        "cubemap download: expected R32Float, got {:?}",
        texture.format()
    );
    assert!(
        (res * R32F_BPP) % 256 == 0,
        "cubemap download: row stride for res {res} is not a multiple of 256",
    );

    let bytes_per_row = res * R32F_BPP;
    let face_bytes = (bytes_per_row * res) as BufferAddress;
    let total_bytes = face_bytes * 6;

    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("mid-freq cubemap staging"),
        size: total_bytes,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("mid-freq cubemap download"),
    });
    for face in CubemapFace::ALL {
        let face_idx = face as u32;
        let buffer_offset = face_bytes * face_idx as BufferAddress;
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: face_idx,
                },
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &staging,
                layout: TexelCopyBufferLayout {
                    offset: buffer_offset,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(res),
                },
            },
            Extent3d {
                width: res,
                height: res,
                depth_or_array_layers: 1,
            },
        );
    }
    queue.submit([encoder.finish()]);

    let bytes = map_and_read::<u8>(device, &staging, total_bytes)?;

    let mut cube = Cubemap::<f32>::new(res);
    for face in CubemapFace::ALL {
        let face_idx = face as usize;
        let start = face_idx * face_bytes as usize;
        let end = start + face_bytes as usize;
        let face_floats: &[f32] = bytemuck::cast_slice(&bytes[start..end]);
        cube.face_data_mut(face).copy_from_slice(face_floats);
    }
    Ok(cube)
}

// ---------------------------------------------------------------------------
// Mid-frequency detail dispatch
// ---------------------------------------------------------------------------

/// GPU-side mirror of [`thalos_terrain_gen::stages::MidFreqDetailParams`].
/// std140-compatible: 8 × 4 bytes = 32 bytes, all scalar fields naturally
/// aligned, no padding holes other than the explicit `_pad` slot kept in
/// sync with the WGSL struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct MidFreqParamsGpu {
    body_radius_m: f32,
    base_wl_m: f32,
    noise_amp_m: f32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    seed: u32,
    _pad: u32,
}

impl MidFreqParamsGpu {
    fn from_cpu(body_radius_m: f32, p: &MidFreqDetailParams) -> Self {
        Self {
            body_radius_m,
            base_wl_m: p.base_wl_m,
            noise_amp_m: p.noise_amp_m,
            octaves: p.octaves,
            persistence: p.persistence,
            lacunarity: p.lacunarity,
            seed: p.seed,
            _pad: 0,
        }
    }
}

/// WGSL source for the mid-frequency-detail kernel. Loaded at compile
/// time so the binary is self-contained.
const MID_FREQ_DETAIL_WGSL: &str =
    include_str!("../../terrain_gen/shaders/mid_freq_detail.wgsl");

/// WGSL source for the bevy_erosion_filter erosion kernel. Workspace
/// `bevy_erosion_filter` lives at `~/dev/bevy_erosion_filter`; we point
/// directly at its asset shader. The file uses one naga_oil directive
/// (`#define_import_path ...`) that raw wgpu/naga doesn't understand —
/// stripped at runtime by `strip_naga_oil_directives` below.
const EROSION_WGSL: &str =
    include_str!("../../../../bevy_erosion_filter/assets/shaders/erosion.wgsl");

/// Strip naga_oil preprocessor directives (`#define_import_path ...`,
/// `#import ...`) so raw wgpu/naga can compile a shader that was
/// authored for Bevy's `naga_oil` composition layer. Operates on whole
/// lines; the directives must appear as the first non-whitespace token
/// on their line. The directive line is replaced with a blank line so
/// downstream error messages still report meaningful line numbers.
fn strip_naga_oil_directives(src: &str) -> String {
    src.lines()
        .map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("#define_import_path") || trimmed.starts_with("#import") {
                ""
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build the full WGSL source the wgpu pipeline compiles: the erosion
/// library (with naga_oil directives stripped) prepended to the
/// mid-frequency-detail kernel so the kernel's `main` can call
/// `erosion_filter_surface3d` etc. directly.
fn build_mid_freq_shader_source() -> String {
    let mut out = String::with_capacity(EROSION_WGSL.len() + MID_FREQ_DETAIL_WGSL.len() + 256);
    out.push_str("// === BEGIN bevy_erosion_filter::erosion ===\n");
    out.push_str(&strip_naga_oil_directives(EROSION_WGSL));
    out.push_str("\n// === END bevy_erosion_filter::erosion ===\n");
    out.push_str("\n// === BEGIN mid_freq_detail kernel ===\n");
    out.push_str(MID_FREQ_DETAIL_WGSL);
    out
}

/// Dispatch the mid-frequency-detail kernel against an in-memory
/// `Cubemap<f32>` height accumulator. The cubemap is uploaded, the
/// compute pass runs across all 6 faces in parallel, and the result is
/// downloaded back into the same `Cubemap` (in-place via assignment).
///
/// This is the function the pipeline's `MidFreqRunner` closure wraps in
/// `bake_dump`. The runtime tile provider will eventually share the
/// same WGSL source via `#import` (Phase 3); the wgpu pipeline objects
/// stay distinct because the Device instances differ.
pub fn run_mid_freq(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    height: &mut Cubemap<f32>,
    body_radius_m: f32,
    params: &MidFreqDetailParams,
) -> Result<()> {
    let res = height.resolution();

    // 1. Upload current height to a storage texture.
    let texture = upload_cubemap_f32(device, queue, height);
    let view = texture.create_view(&TextureViewDescriptor {
        label: Some("mid-freq cubemap view"),
        format: Some(TextureFormat::R32Float),
        dimension: Some(TextureViewDimension::D2Array),
        usage: None,
        aspect: TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(6),
    });

    // 2. Params UBO.
    let params_gpu = MidFreqParamsGpu::from_cpu(body_radius_m, params);
    let params_buf = device.create_buffer(&BufferDescriptor {
        label: Some("mid-freq params"),
        size: std::mem::size_of::<MidFreqParamsGpu>() as BufferAddress,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params_gpu));

    // 3. Bind group layout + pipeline.
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("mid-freq bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::R32Float,
                    view_dimension: TextureViewDimension::D2Array,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("mid-freq pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Compose the WGSL: bevy_erosion_filter's erosion library
    // (naga_oil directives stripped) concatenated with our kernel. One
    // shader module, single compile.
    let composed = build_mid_freq_shader_source();
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("mid-freq shader"),
        source: ShaderSource::Wgsl(composed.into()),
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("mid-freq pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("mid-freq bg"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    // 4. Dispatch. 8×8 workgroup over res×res per face × 6 faces.
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("mid-freq encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("mid-freq pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = res.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 6);
    }
    queue.submit([encoder.finish()]);

    // 5. Read back into the same Cubemap.
    let updated = download_cubemap_f32(device, queue, &texture)?;
    *height = updated;
    Ok(())
}
