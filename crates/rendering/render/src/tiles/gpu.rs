//! GPU storage and shared topology for standard-path terrain tiles.
//!
//! The CPU still samples the authoritative [`SurfaceTile`] and builds exact
//! body-local positions in f64. What changes is ownership: those positions and
//! surface channels are uploaded into one array-texture slot, while every tile
//! entity reuses one of three small patch meshes. [`MeshTag`] carries the slot
//! through Bevy's ordinary mesh instance data, so automatic batching remains
//! available and the material stays on the standard PBR path.

use std::sync::{Arc, Mutex};

use bevy::asset::RenderAssetUsages;
use bevy::camera::primitives::Aabb;
use bevy::math::DVec3;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDimension, TextureFormat,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};

use super::{
    LEVEL_RENDER_LIFT_M, SurfaceTile, TILE_HALO, TILE_RES, TILE_SKIRT_VERTS, TILE_WRAP_M,
    build_tile_strip_indices, debug_height_scale, skirt_drop_m,
};

/// Fixed number of array layers. Metal supports 2,048 layers on the target
/// hardware, matching the legacy terrain atlas's array-texture model.
pub const TILE_GPU_ATLAS_SLOTS: u32 = 2_048;
/// Steady-state occupancy target. The remaining layers are replacement
/// headroom: a coarse merge target must land before its fine children may
/// retire, or the hole-free streamer can deadlock at a full atlas.
pub const TILE_GPU_USABLE_SLOTS: usize = 1_792;

pub const TILE_GPU_POSITION_WIDTH: u32 = (TILE_RES + 2 * TILE_HALO) as u32;
pub const TILE_GPU_POSITION_HEIGHT: u32 = 135;
pub const TILE_GPU_SURFACE_WIDTH: u32 = TILE_GPU_POSITION_WIDTH;
pub const TILE_GPU_SURFACE_HEIGHT: u32 = TILE_GPU_POSITION_WIDTH;

const TILE_GPU_GRID_SIDE: usize = TILE_RES + 2 * TILE_HALO;
const TILE_GPU_SURFACE_TEXELS: usize = TILE_GPU_GRID_SIDE * TILE_GPU_GRID_SIDE;
const TILE_GPU_ORIGIN_BODY_TEXEL: usize = TILE_GPU_SURFACE_TEXELS + TILE_SKIRT_VERTS;
const TILE_GPU_ORIGIN_WRAPPED_TEXEL: usize = TILE_GPU_ORIGIN_BODY_TEXEL + 1;
const TILE_GPU_POSITION_TEXELS: usize =
    TILE_GPU_POSITION_WIDTH as usize * TILE_GPU_POSITION_HEIGHT as usize;

pub const TILE_GPU_POSITION_BYTES: usize = TILE_GPU_POSITION_TEXELS * 16;
pub const TILE_GPU_SURFACE_BYTES: usize = TILE_GPU_SURFACE_TEXELS * 4;
/// Occupied payload bytes per resident tile. Shared patch meshes are a fixed
/// ~1 MiB total and therefore do not belong in this per-tile denominator.
pub const TILE_GPU_SLOT_BYTES: usize = TILE_GPU_POSITION_BYTES + TILE_GPU_SURFACE_BYTES;
pub const TILE_GPU_ALLOCATED_BYTES: usize = TILE_GPU_SLOT_BYTES * TILE_GPU_ATLAS_SLOTS as usize;

const _: () = assert!(TILE_GPU_ORIGIN_WRAPPED_TEXEL < TILE_GPU_POSITION_TEXELS);
const _: () = assert!(TILE_GPU_USABLE_SLOTS < TILE_GPU_ATLAS_SLOTS as usize);

#[derive(Clone, Default)]
pub struct TileGpuImages {
    pub position: Handle<Image>,
    pub surface: Handle<Image>,
}

pub(super) struct TileGpuPayload {
    pub(super) position_bytes: Vec<u8>,
    pub(super) surface_bytes: Vec<u8>,
    pub origin: DVec3,
    pub mesh_h: (f32, f32),
    pub surface_aabb: Aabb,
    pub full_aabb: Aabb,
}

struct TileGpuUpload {
    slot: u32,
    position_bytes: Vec<u8>,
    surface_bytes: Vec<u8>,
}

type UploadQueue = Arc<Mutex<Vec<TileGpuUpload>>>;

/// Main-world owner of atlas slots and the upload bridge.
#[derive(Resource)]
pub struct TileGpuStore {
    images: TileGpuImages,
    free_slots: Vec<u32>,
    uploads: UploadQueue,
}

impl FromWorld for TileGpuStore {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();
        let position = images.add(array_image(
            "tile_gpu_positions",
            TILE_GPU_POSITION_WIDTH,
            TILE_GPU_POSITION_HEIGHT,
            TextureFormat::Rgba32Float,
        ));
        let surface = images.add(array_image(
            "tile_gpu_surface",
            TILE_GPU_SURFACE_WIDTH,
            TILE_GPU_SURFACE_HEIGHT,
            TextureFormat::Rgba8Unorm,
        ));
        let free_slots = (0..TILE_GPU_ATLAS_SLOTS).rev().collect();
        Self {
            images: TileGpuImages { position, surface },
            free_slots,
            uploads: Arc::default(),
        }
    }
}

impl TileGpuStore {
    pub fn images(&self) -> TileGpuImages {
        self.images.clone()
    }

    pub fn free_slot_count(&self) -> usize {
        self.free_slots.len()
    }

    pub fn usable_budget_bytes(&self) -> usize {
        TILE_GPU_USABLE_SLOTS * TILE_GPU_SLOT_BYTES
    }

    pub(super) fn allocate(&mut self) -> Option<u32> {
        self.free_slots.pop()
    }

    pub fn release(&mut self, slot: u32) {
        debug_assert!(
            !self.free_slots.contains(&slot),
            "terrain atlas slot {slot} released twice"
        );
        self.free_slots.push(slot);
    }

    pub(super) fn upload(&self, slot: u32, payload: TileGpuPayload) {
        let upload = TileGpuUpload {
            slot,
            position_bytes: payload.position_bytes,
            surface_bytes: payload.surface_bytes,
        };
        self.uploads
            .lock()
            .expect("tile GPU upload queue poisoned")
            .push(upload);
    }
}

fn array_image(label: &'static str, width: u32, height: u32, format: TextureFormat) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width,
            height,
            depth_or_array_layers: TILE_GPU_ATLAS_SLOTS,
        },
        TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.label = Some(label);
    image
}

/// Render-world view of the main-world upload queue. The queue itself is
/// shared, so landing a tile does not clone its ~343 KiB payload during
/// extraction.
#[derive(Resource)]
struct TileGpuUploadBridge {
    images: TileGpuImages,
    uploads: UploadQueue,
}

impl ExtractResource for TileGpuUploadBridge {
    type Source = TileGpuStore;

    fn extract_resource(source: &Self::Source) -> Self {
        Self {
            images: source.images.clone(),
            uploads: Arc::clone(&source.uploads),
        }
    }
}

fn upload_tiles(
    bridge: Res<TileGpuUploadBridge>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    queue: Res<RenderQueue>,
) {
    let (Some(position), Some(surface)) = (
        gpu_images.get(&bridge.images.position),
        gpu_images.get(&bridge.images.surface),
    ) else {
        // Image preparation can trail extraction by one frame at startup.
        // Keep the uploads queued; drawing cannot begin before a tile lands.
        return;
    };
    let mut uploads = bridge
        .uploads
        .lock()
        .expect("tile GPU upload queue poisoned");
    for upload in uploads.drain(..) {
        write_layer(
            &queue,
            position,
            upload.slot,
            &upload.position_bytes,
            TILE_GPU_POSITION_WIDTH,
            TILE_GPU_POSITION_HEIGHT,
            16,
        );
        write_layer(
            &queue,
            surface,
            upload.slot,
            &upload.surface_bytes,
            TILE_GPU_SURFACE_WIDTH,
            TILE_GPU_SURFACE_HEIGHT,
            4,
        );
    }
}

fn write_layer(
    queue: &RenderQueue,
    image: &GpuImage,
    slot: u32,
    bytes: &[u8],
    width: u32,
    height: u32,
    bytes_per_texel: u32,
) {
    debug_assert_eq!(bytes.len(), (width * height * bytes_per_texel) as usize);
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &image.texture,
            mip_level: 0,
            origin: Origin3d {
                x: 0,
                y: 0,
                z: slot,
            },
            aspect: TextureAspect::All,
        },
        bytes,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * bytes_per_texel),
            rows_per_image: Some(height),
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PatchResolution {
    R33,
    R65,
    R129,
}

impl PatchResolution {
    pub(super) fn side(self) -> usize {
        match self {
            Self::R33 => 33,
            Self::R65 => 65,
            Self::R129 => 129,
        }
    }
}

#[derive(Resource)]
pub(super) struct TilePatchMeshes {
    r33: Handle<Mesh>,
    r65: Handle<Mesh>,
    r129: Handle<Mesh>,
}

impl FromWorld for TilePatchMeshes {
    fn from_world(world: &mut World) -> Self {
        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        Self {
            r33: meshes.add(build_patch_mesh(PatchResolution::R33)),
            r65: meshes.add(build_patch_mesh(PatchResolution::R65)),
            r129: meshes.add(build_patch_mesh(PatchResolution::R129)),
        }
    }
}

impl TilePatchMeshes {
    pub(super) fn handle(&self, resolution: PatchResolution) -> Handle<Mesh> {
        match resolution {
            PatchResolution::R33 => self.r33.clone(),
            PatchResolution::R65 => self.r65.clone(),
            PatchResolution::R129 => self.r129.clone(),
        }
    }
}

pub(super) fn build_patch_mesh(resolution: PatchResolution) -> Mesh {
    let res = resolution.side();
    let source_step = (TILE_RES - 1) / (res - 1);
    debug_assert!((TILE_RES - 1).is_multiple_of(res - 1));
    let vertex_count = res * res + 4 * res - 4;
    let mut positions = Vec::with_capacity(vertex_count);
    let mut patch_data = Vec::with_capacity(vertex_count);

    for j in 0..res {
        for i in 0..res {
            let sample_x = TILE_HALO + i * source_step;
            let sample_y = TILE_HALO + j * source_step;
            positions.push([
                sample_x as f32,
                sample_y as f32,
                (sample_y * TILE_GPU_GRID_SIDE + sample_x) as f32,
            ]);
            patch_data.push([source_step as f32, 0.0]);
        }
    }

    let mut border = Vec::with_capacity(4 * res - 4);
    for i in 0..res {
        border.push(i as u16);
    }
    for j in 1..res {
        border.push((j * res + res - 1) as u16);
    }
    for i in (0..res - 1).rev() {
        border.push(((res - 1) * res + i) as u16);
    }
    for j in (1..res - 1).rev() {
        border.push((j * res) as u16);
    }
    let skirt_base = positions.len() as u16;
    for (border_index, &top_vertex) in border.iter().enumerate() {
        let top_sample = positions[top_vertex as usize][2];
        let packed_index = TILE_GPU_SURFACE_TEXELS + border_index * source_step;
        positions.push([
            (packed_index % TILE_GPU_POSITION_WIDTH as usize) as f32,
            (packed_index / TILE_GPU_POSITION_WIDTH as usize) as f32,
            top_sample,
        ]);
        patch_data.push([source_step as f32, 1.0]);
    }
    let indices = build_tile_strip_indices(res, &border, skirt_base);

    // Keep Bevy's ordinary StandardMaterial shader definitions/layout. The
    // custom vertex stage ignores these dummy attributes and fills their
    // outputs from the atlas, but retaining the standard attribute set means
    // the fragment and prepass contracts need no private pipeline fork.
    let count = positions.len();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleStrip,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; count]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0, 1.0, 1.0, 1.0]; count]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, patch_data);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.0, 0.0]; count]);
    mesh.insert_indices(Indices::U16(indices));
    mesh
}

pub(super) fn build_tile_payload(
    tile: &SurfaceTile,
    radius_m: f64,
    relief_m: f64,
) -> TileGpuPayload {
    let key = tile.key;
    let side = TILE_GPU_GRID_SIDE;
    let step = 1.0 / (TILE_RES - 1) as f64;
    let h_scale = debug_height_scale();
    let lift = key.level as f64 * LEVEL_RENDER_LIFT_M;

    let mut body_positions = Vec::with_capacity(TILE_GPU_SURFACE_TEXELS);
    for j in 0..side {
        for i in 0..side {
            let s = (i as f64 - TILE_HALO as f64) * step;
            let t = (j as f64 - TILE_HALO as f64) * step;
            let height = tile.heights_m[j * side + i] as f64 * h_scale;
            body_positions.push(key.dir_at(s, t) * (radius_m + height + lift));
        }
    }
    let origin =
        key.center_dir() * (radius_m + tile.heights_m[(side / 2) * side + side / 2] as f64);
    let wrap_anchor = DVec3::new(
        (origin.x / TILE_WRAP_M).floor() * TILE_WRAP_M,
        (origin.y / TILE_WRAP_M).floor() * TILE_WRAP_M,
        (origin.z / TILE_WRAP_M).floor() * TILE_WRAP_M,
    );

    let mut position_texels = vec![[0.0_f32; 4]; TILE_GPU_POSITION_TEXELS];
    let mut surface_bytes = vec![0_u8; TILE_GPU_SURFACE_BYTES];
    let mut mesh_h = (f32::INFINITY, f32::NEG_INFINITY);
    let mut surface_lo = [f32::INFINITY; 3];
    let mut surface_hi = [f32::NEG_INFINITY; 3];

    for (index, &position_body) in body_positions.iter().enumerate() {
        let local = position_body - origin;
        let eco_altitude = tile.bands[index][0];
        position_texels[index] = [local.x as f32, local.y as f32, local.z as f32, eco_altitude];

        let color = tile.albedo_linear[index];
        let canopy = tile.bands[index][1];
        let dst = &mut surface_bytes[index * 4..index * 4 + 4];
        dst.copy_from_slice(&[
            unorm8(color[0]),
            unorm8(color[1]),
            unorm8(color[2]),
            unorm8(canopy),
        ]);
    }

    for j in 0..TILE_RES {
        for i in 0..TILE_RES {
            let sample = (j + TILE_HALO) * side + i + TILE_HALO;
            let local = position_texels[sample];
            let height = (body_positions[sample].length() - radius_m - lift) as f32;
            mesh_h.0 = mesh_h.0.min(height);
            mesh_h.1 = mesh_h.1.max(height);
            for axis in 0..3 {
                surface_lo[axis] = surface_lo[axis].min(local[axis]);
                surface_hi[axis] = surface_hi[axis].max(local[axis]);
            }
        }
    }

    let mut border_samples = Vec::with_capacity(TILE_SKIRT_VERTS);
    for i in 0..TILE_RES {
        border_samples.push(TILE_HALO * side + i + TILE_HALO);
    }
    for j in 1..TILE_RES {
        border_samples.push((j + TILE_HALO) * side + TILE_HALO + TILE_RES - 1);
    }
    for i in (0..TILE_RES - 1).rev() {
        border_samples.push((TILE_HALO + TILE_RES - 1) * side + i + TILE_HALO);
    }
    for j in (1..TILE_RES - 1).rev() {
        border_samples.push((j + TILE_HALO) * side + TILE_HALO);
    }

    let floor_radius_m = radius_m - relief_m.max(0.0);
    let legacy_drop_m = skirt_drop_m(tile.sample_spacing_m, radius_m) as f64;
    let down = -key.center_dir();
    let mut full_lo = surface_lo;
    let mut full_hi = surface_hi;
    for (skirt_index, &sample) in border_samples.iter().enumerate() {
        let top = position_texels[sample];
        let top_local = DVec3::new(top[0] as f64, top[1] as f64, top[2] as f64);
        let top_body = origin + top_local;
        let bottom = if relief_m.is_finite() {
            top_body.normalize() * floor_radius_m
        } else {
            top_body + down * legacy_drop_m
        } - origin;
        let texel = TILE_GPU_SURFACE_TEXELS + skirt_index;
        position_texels[texel] = [bottom.x as f32, bottom.y as f32, bottom.z as f32, top[3]];
        for axis in 0..3 {
            full_lo[axis] = full_lo[axis].min(position_texels[texel][axis]);
            full_hi[axis] = full_hi[axis].max(position_texels[texel][axis]);
        }
    }

    position_texels[TILE_GPU_ORIGIN_BODY_TEXEL] = [
        origin.x as f32,
        origin.y as f32,
        origin.z as f32,
        radius_m as f32,
    ];
    let origin_wrapped = origin - wrap_anchor;
    position_texels[TILE_GPU_ORIGIN_WRAPPED_TEXEL] = [
        origin_wrapped.x as f32,
        origin_wrapped.y as f32,
        origin_wrapped.z as f32,
        0.0,
    ];

    if (h_scale - 1.0).abs() < 1.0e-9 {
        let p0 = body_positions[TILE_HALO * side + TILE_HALO + TILE_RES - 1] - origin;
        let p1 = body_positions[(TILE_HALO + 1) * side + TILE_HALO + TILE_RES - 1] - origin;
        let p2 = body_positions[TILE_HALO * side + TILE_HALO + TILE_RES - 2] - origin;
        debug_assert!(
            (p1 - p0).cross(p2 - p0).dot(key.center_dir()) > 0.0,
            "tile {key:?}: shared patch first triangle winds inward"
        );
    }

    TileGpuPayload {
        position_bytes: bytemuck::cast_slice(&position_texels).to_vec(),
        surface_bytes,
        origin,
        mesh_h,
        surface_aabb: Aabb::from_min_max(
            Vec3::from_array(surface_lo),
            Vec3::from_array(surface_hi),
        ),
        full_aabb: Aabb::from_min_max(Vec3::from_array(full_lo), Vec3::from_array(full_hi)),
    }
}

fn unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub(super) struct TileGpuPlugin;

impl Plugin for TileGpuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TileGpuStore>()
            .init_resource::<TilePatchMeshes>()
            .add_plugins(ExtractResourcePlugin::<TileGpuUploadBridge>::default());
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(Render, upload_tiles.in_set(RenderSystems::PrepareResources));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_patch_resolution_addresses_exact_source_samples() {
        for resolution in [
            PatchResolution::R33,
            PatchResolution::R65,
            PatchResolution::R129,
        ] {
            let mesh = build_patch_mesh(resolution);
            let positions = mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .and_then(|values| values.as_float3())
                .expect("patch positions");
            let res = resolution.side();
            let source_step = (TILE_RES - 1) / (res - 1);
            for j in 0..res {
                for i in 0..res {
                    let position = positions[j * res + i];
                    assert_eq!(position[0] as usize, TILE_HALO + i * source_step);
                    assert_eq!(position[1] as usize, TILE_HALO + j * source_step);
                }
            }
            assert_eq!(mesh.count_vertices(), res * res + 4 * res - 4);
        }
    }

    #[test]
    fn atlas_payload_size_matches_the_budget_denominator() {
        assert_eq!(TILE_GPU_POSITION_BYTES, 131 * 135 * 16);
        assert_eq!(TILE_GPU_SURFACE_BYTES, 131 * 131 * 4);
        assert_eq!(TILE_GPU_SLOT_BYTES, 351_604);
        assert_eq!(TILE_GPU_ALLOCATED_BYTES, 720_084_992);
    }
}
