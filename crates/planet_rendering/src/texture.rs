use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;

/// GPU resources consumed by a single [`crate::PlanetMaterial`].
///
/// Produced by [`crate::bake_from_body_data`] and plugged directly into the
/// material's fields.
pub struct PlanetTextures {
    // --- Layer 1: baked cubemaps ------------------------------------------
    /// sRGB albedo cubemap (Rgba8UnormSrgb). Primary surface color — the
    /// shader samples it directly.
    pub albedo: Handle<Image>,
    /// R16Unorm displacement cubemap.
    pub height: Handle<Image>,
    /// R8Unorm roughness cubemap. Per-texel microsurface response, sampled
    /// bilinearly by the shader for the PBR lighting term.
    pub roughness: Handle<Image>,

    // --- Layer 2: feature SSBOs -------------------------------------------
    /// `array<Crater>` — mid-frequency discrete craters.
    pub craters: Handle<ShaderStorageBuffer>,
    /// `array<CellRange>` — one entry per ico cell, `(start, count)` into
    /// `feature_ids`.
    pub cell_index: Handle<ShaderStorageBuffer>,
    /// `array<u32>` — concatenated crater indices referenced by cells.
    pub feature_ids: Handle<ShaderStorageBuffer>,
}
