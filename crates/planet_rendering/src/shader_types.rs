//! GPU-facing struct definitions mirrored from WGSL.
//!
//! These structs are uploaded as raw bytes into storage buffers and read by
//! the planet impostor shader. Layout MUST match the WGSL definitions in
//! `assets/shaders/planet_impostor.wgsl`.
//!
//! ## std430 alignment notes
//!
//! We use `#[repr(C)]` + `bytemuck::Pod` so the slices can be cast to bytes
//! directly. Field ordering is chosen so `#[repr(C)]` layout matches WGSL's
//! std430 layout exactly — in particular, placing `Vec3` fields first and
//! following them with f32 fields avoids alignment gaps (offset 12 is
//! 4-aligned, so an f32 packs tight against a vec3).
//!
//! **Do not** derive `encase::ShaderType` on these structs. `ShaderType`
//! encodes structs with vec3 padded to 16 bytes, which diverges from the
//! `#[repr(C)]` layout. We upload via `bytemuck::cast_slice`, so bytemuck
//! alignment is the source of truth.

use bevy::math::Vec3;
use bytemuck::{Pod, Zeroable};

/// A discrete mid-frequency crater feature. Mirrors `Crater` in
/// `thalos_terrain_gen::types` but with a fixed GPU-compatible layout.
///
/// WGSL layout (std430):
/// ```wgsl
/// struct Crater {
///     center: vec3<f32>,    // offset 0,  size 12
///     radius_m: f32,        // offset 12, size 4
///     depth_m: f32,         // offset 16, size 4
///     rim_height_m: f32,    // offset 20, size 4
///     age_gyr: f32,         // offset 24, size 4
///     material_id: u32,     // offset 28, size 4
/// }; // total 32 bytes, align 16
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuCrater {
    pub center: Vec3,
    pub radius_m: f32,
    pub depth_m: f32,
    pub rim_height_m: f32,
    pub age_gyr: f32,
    pub material_id: u32,
}

/// A `(start, count)` range into the flattened `feature_ids` buffer.
/// One entry per ico cell in the spatial index.
///
/// WGSL layout: `struct CellRange { start: u32, count: u32 }` (8 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuCellRange {
    pub start: u32,
    pub count: u32,
}

/// A material palette entry.
///
/// WGSL layout (std430):
/// ```wgsl
/// struct Material {
///     albedo: vec3<f32>,   // offset 0,  size 12
///     roughness: f32,      // offset 12, size 4
/// }; // total 16 bytes, align 16
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMaterial {
    pub albedo: Vec3,
    pub roughness: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_match_wgsl_std430() {
        assert_eq!(std::mem::size_of::<GpuCrater>(), 32);
        assert_eq!(std::mem::size_of::<GpuCellRange>(), 8);
        assert_eq!(std::mem::size_of::<GpuMaterial>(), 16);
    }

    #[test]
    fn struct_alignments_are_compatible() {
        // `#[repr(C)]` alignment is the max field alignment. For these
        // structs that's 4 (Vec3 is repr(C) of 3 f32s, align 4). The WGSL
        // side declares align 16 for vec3, but since the per-field byte
        // offsets match, the layout is binary-compatible when the host
        // writes raw Pod bytes and the shader reads std430.
        assert_eq!(std::mem::align_of::<GpuCrater>(), 4);
        assert_eq!(std::mem::align_of::<GpuCellRange>(), 4);
        assert_eq!(std::mem::align_of::<GpuMaterial>(), 4);
    }
}
