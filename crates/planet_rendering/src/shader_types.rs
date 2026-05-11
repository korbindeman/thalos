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

/// A hand-anchored active dune region as seen by the impostor's per-fragment
/// dynamic layer synthesis.
///
/// WGSL layout (std430), `align 16` due to `vec3<f32>`:
/// ```wgsl
/// struct DuneSea {
///     center:            vec3<f32>,  // offset 0,  size 12
///     radius_rad:        f32,        // offset 12, size 4
///     axis_tangent:      vec3<f32>,  // offset 16, size 12
///     feather_rad:       f32,        // offset 28, size 4
///     albedo_crest_lin:  vec3<f32>,  // offset 32, size 12
///     crest_strength:    f32,        // offset 44, size 4
///     lambda_draa_m:     f32,        // offset 48
///     amplitude_draa_m:  f32,        // offset 52
///     lambda_dune_m:     f32,        // offset 56
///     amplitude_dune_m:  f32,        // offset 60
///     alpha_skew:        f32,        // offset 64
///     warp_amp_unit:     f32,        // offset 68
///     warp_freq:         f32,        // offset 72
///     coverage_scale:    f32,        // offset 76
///     phase_offset_m:    f32,        // offset 80
///     amplitude_scale:   f32,        // offset 84
///     mobility:          f32,        // offset 88
///     seed:              u32,        // offset 92
/// };                                  // total 96 bytes (multiple of 16)
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuDuneSea {
    pub center: Vec3,
    pub radius_rad: f32,
    pub axis_tangent: Vec3,
    pub feather_rad: f32,
    pub albedo_crest_lin: Vec3,
    pub crest_strength: f32,
    pub lambda_draa_m: f32,
    pub amplitude_draa_m: f32,
    pub lambda_dune_m: f32,
    pub amplitude_dune_m: f32,
    pub alpha_skew: f32,
    pub warp_amp_unit: f32,
    pub warp_freq: f32,
    pub coverage_scale: f32,
    pub phase_offset_m: f32,
    pub amplitude_scale: f32,
    pub mobility: f32,
    pub seed: u32,
}

/// Seasonal polar ice cap overlay. This is intentionally not part of the
/// baked terrain cubemaps; the impostor shader applies it as a runtime surface
/// layer over static terrain.
///
/// WGSL layout (std430):
/// ```wgsl
/// struct IceCap {
///     axis:                 vec3<f32>, // offset 0,  size 12
///     flags:                u32,       // offset 12, size 4
///     albedo_linear:        vec3<f32>, // offset 16, size 12
///     edge_latitude_deg:    f32,       // offset 28, size 4
///     dust_albedo_linear:   vec3<f32>, // offset 32, size 12
///     solid_latitude_deg:   f32,       // offset 44, size 4
///     edge_noise_deg:       f32,       // offset 48, size 4
///     edge_sharpness:       f32,       // offset 52, size 4
///     noise_frequency:      f32,       // offset 56, size 4
///     max_thickness_m:      f32,       // offset 60, size 4
///     albedo_strength:      f32,       // offset 64, size 4
///     roughness:            f32,       // offset 68, size 4
///     roughness_strength:   f32,       // offset 72, size 4
///     obliquity_response:   f32,       // offset 76, size 4
///     coverage_scale:       f32,       // offset 80, size 4
///     edge_offset_deg:      f32,       // offset 84, size 4
///     thickness_scale:      f32,       // offset 88, size 4
///     dustiness:            f32,       // offset 92, size 4
///     seed:                 u32,       // offset 96, size 4
///     _pad0:                u32,       // offset 100, size 4
///     _pad1:                u32,       // offset 104, size 4
///     _pad2:                u32,       // offset 108, size 4
/// };                                   // total 112 bytes
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuIceCap {
    pub axis: Vec3,
    pub flags: u32,
    pub albedo_linear: Vec3,
    pub edge_latitude_deg: f32,
    pub dust_albedo_linear: Vec3,
    pub solid_latitude_deg: f32,
    pub edge_noise_deg: f32,
    pub edge_sharpness: f32,
    pub noise_frequency: f32,
    pub max_thickness_m: f32,
    pub albedo_strength: f32,
    pub roughness: f32,
    pub roughness_strength: f32,
    pub obliquity_response: f32,
    pub coverage_scale: f32,
    pub edge_offset_deg: f32,
    pub thickness_scale: f32,
    pub dustiness: f32,
    pub seed: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// A radial volcanic feature for impostor-side erosion/color synthesis.
///
/// The broad shape is already baked into the cubemaps. This descriptor carries
/// the local frame and tuning needed to add feature-local gully normals,
/// exposed-material color, and roughness modulation in the shader.
///
/// WGSL layout (std430):
/// ```wgsl
/// struct RadialFeature {
///     center:          vec3<f32>, // offset 0,  size 12
///     radius_m:        f32,       // offset 12, size 4
///     east:            vec3<f32>, // offset 16, size 12
///     height_m:        f32,       // offset 28, size 4
///     north:           vec3<f32>, // offset 32, size 12
///     erosion_scale_m: f32,       // offset 44, size 4
///     seed:            u32,       // offset 48, size 4
///     material_id:     u32,       // offset 52, size 4
///     _pad0:           u32,       // offset 56, size 4
///     _pad1:           u32,       // offset 60, size 4
/// };                              // total 64 bytes
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuRadialFeature {
    pub center: Vec3,
    pub radius_m: f32,
    pub east: Vec3,
    pub height_m: f32,
    pub north: Vec3,
    pub erosion_scale_m: f32,
    pub seed: u32,
    pub material_id: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_match_wgsl_std430() {
        assert_eq!(std::mem::size_of::<GpuCrater>(), 32);
        assert_eq!(std::mem::size_of::<GpuCellRange>(), 8);
        assert_eq!(std::mem::size_of::<GpuDuneSea>(), 96);
        assert_eq!(std::mem::size_of::<GpuIceCap>(), 112);
        assert_eq!(std::mem::size_of::<GpuRadialFeature>(), 64);
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
        assert_eq!(std::mem::align_of::<GpuDuneSea>(), 4);
        assert_eq!(std::mem::align_of::<GpuIceCap>(), 4);
        assert_eq!(std::mem::align_of::<GpuRadialFeature>(), 4);
    }

    #[test]
    fn dune_sea_field_offsets_match_wgsl_std430() {
        // std430 places vec3<f32> at multiples of 16. Field byte offsets
        // here mirror the WGSL declaration; mismatching them silently
        // shifts data when the SSBO is read.
        let z = GpuDuneSea {
            center: Vec3::ZERO,
            radius_rad: 0.0,
            axis_tangent: Vec3::ZERO,
            feather_rad: 0.0,
            albedo_crest_lin: Vec3::ZERO,
            crest_strength: 0.0,
            lambda_draa_m: 0.0,
            amplitude_draa_m: 0.0,
            lambda_dune_m: 0.0,
            amplitude_dune_m: 0.0,
            alpha_skew: 0.0,
            warp_amp_unit: 0.0,
            warp_freq: 0.0,
            coverage_scale: 0.0,
            phase_offset_m: 0.0,
            amplitude_scale: 0.0,
            mobility: 0.0,
            seed: 0,
        };
        let base = (&z as *const GpuDuneSea) as usize;
        let off = |p: usize| p - base;
        assert_eq!(off((&z.center as *const Vec3) as usize), 0);
        assert_eq!(off((&z.radius_rad as *const f32) as usize), 12);
        assert_eq!(off((&z.axis_tangent as *const Vec3) as usize), 16);
        assert_eq!(off((&z.feather_rad as *const f32) as usize), 28);
        assert_eq!(off((&z.albedo_crest_lin as *const Vec3) as usize), 32);
        assert_eq!(off((&z.crest_strength as *const f32) as usize), 44);
        assert_eq!(off((&z.lambda_draa_m as *const f32) as usize), 48);
        assert_eq!(off((&z.amplitude_draa_m as *const f32) as usize), 52);
        assert_eq!(off((&z.lambda_dune_m as *const f32) as usize), 56);
        assert_eq!(off((&z.amplitude_dune_m as *const f32) as usize), 60);
        assert_eq!(off((&z.alpha_skew as *const f32) as usize), 64);
        assert_eq!(off((&z.warp_amp_unit as *const f32) as usize), 68);
        assert_eq!(off((&z.warp_freq as *const f32) as usize), 72);
        assert_eq!(off((&z.coverage_scale as *const f32) as usize), 76);
        assert_eq!(off((&z.phase_offset_m as *const f32) as usize), 80);
        assert_eq!(off((&z.amplitude_scale as *const f32) as usize), 84);
        assert_eq!(off((&z.mobility as *const f32) as usize), 88);
        assert_eq!(off((&z.seed as *const u32) as usize), 92);
    }

    #[test]
    fn ice_cap_field_offsets_match_wgsl_std430() {
        let z = GpuIceCap {
            axis: Vec3::ZERO,
            flags: 0,
            albedo_linear: Vec3::ZERO,
            edge_latitude_deg: 0.0,
            dust_albedo_linear: Vec3::ZERO,
            solid_latitude_deg: 0.0,
            edge_noise_deg: 0.0,
            edge_sharpness: 0.0,
            noise_frequency: 0.0,
            max_thickness_m: 0.0,
            albedo_strength: 0.0,
            roughness: 0.0,
            roughness_strength: 0.0,
            obliquity_response: 0.0,
            coverage_scale: 0.0,
            edge_offset_deg: 0.0,
            thickness_scale: 0.0,
            dustiness: 0.0,
            seed: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let base = (&z as *const GpuIceCap) as usize;
        let off = |p: usize| p - base;
        assert_eq!(off((&z.axis as *const Vec3) as usize), 0);
        assert_eq!(off((&z.flags as *const u32) as usize), 12);
        assert_eq!(off((&z.albedo_linear as *const Vec3) as usize), 16);
        assert_eq!(off((&z.edge_latitude_deg as *const f32) as usize), 28);
        assert_eq!(off((&z.dust_albedo_linear as *const Vec3) as usize), 32);
        assert_eq!(off((&z.solid_latitude_deg as *const f32) as usize), 44);
        assert_eq!(off((&z.edge_noise_deg as *const f32) as usize), 48);
        assert_eq!(off((&z.edge_sharpness as *const f32) as usize), 52);
        assert_eq!(off((&z.noise_frequency as *const f32) as usize), 56);
        assert_eq!(off((&z.max_thickness_m as *const f32) as usize), 60);
        assert_eq!(off((&z.albedo_strength as *const f32) as usize), 64);
        assert_eq!(off((&z.roughness as *const f32) as usize), 68);
        assert_eq!(off((&z.roughness_strength as *const f32) as usize), 72);
        assert_eq!(off((&z.obliquity_response as *const f32) as usize), 76);
        assert_eq!(off((&z.coverage_scale as *const f32) as usize), 80);
        assert_eq!(off((&z.edge_offset_deg as *const f32) as usize), 84);
        assert_eq!(off((&z.thickness_scale as *const f32) as usize), 88);
        assert_eq!(off((&z.dustiness as *const f32) as usize), 92);
        assert_eq!(off((&z.seed as *const u32) as usize), 96);
        assert_eq!(off((&z._pad0 as *const u32) as usize), 100);
        assert_eq!(off((&z._pad1 as *const u32) as usize), 104);
        assert_eq!(off((&z._pad2 as *const u32) as usize), 108);
    }

    #[test]
    fn radial_feature_field_offsets_match_wgsl_std430() {
        let z = GpuRadialFeature {
            center: Vec3::ZERO,
            radius_m: 0.0,
            east: Vec3::ZERO,
            height_m: 0.0,
            north: Vec3::ZERO,
            erosion_scale_m: 0.0,
            seed: 0,
            material_id: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let base = (&z as *const GpuRadialFeature) as usize;
        let off = |p: usize| p - base;
        assert_eq!(off((&z.center as *const Vec3) as usize), 0);
        assert_eq!(off((&z.radius_m as *const f32) as usize), 12);
        assert_eq!(off((&z.east as *const Vec3) as usize), 16);
        assert_eq!(off((&z.height_m as *const f32) as usize), 28);
        assert_eq!(off((&z.north as *const Vec3) as usize), 32);
        assert_eq!(off((&z.erosion_scale_m as *const f32) as usize), 44);
        assert_eq!(off((&z.seed as *const u32) as usize), 48);
        assert_eq!(off((&z.material_id as *const u32) as usize), 52);
        assert_eq!(off((&z._pad0 as *const u32) as usize), 56);
        assert_eq!(off((&z._pad1 as *const u32) as usize), 60);
    }
}
