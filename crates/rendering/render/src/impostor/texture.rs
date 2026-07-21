use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;

/// GPU resources consumed by a single [`crate::PlanetMaterial`].
///
/// Produced by [`crate::bake_from_planet_surface`] and plugged directly into the
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
    /// R16Unorm active-dune displacement cubemap. This is a dynamic-layer
    /// overlay, encoded as `height_m / params.height_range`.
    pub active_dune_height: Handle<Image>,
    /// Rgba8UnormSrgb active-dune material overlay. RGB is linear sand color
    /// sampled as sRGB, alpha is blend strength.
    pub active_dune_albedo: Handle<Image>,

    // --- Layer 2: feature SSBOs -------------------------------------------
    /// `array<Crater>` — mid-frequency discrete craters.
    pub craters: Handle<ShaderBuffer>,
    /// `array<CellRange>` — one entry per ico cell, `(start, count)` into
    /// `feature_ids`.
    pub cell_index: Handle<ShaderBuffer>,
    /// `array<u32>` — concatenated crater indices referenced by cells.
    pub feature_ids: Handle<ShaderBuffer>,
    /// `array<RadialFeature>` — feature-local radial volcano detail.
    pub radial_features: Handle<ShaderBuffer>,
    /// `array<IceCap>` — dynamic seasonal cap overlays rendered by the
    /// impostor shader over static terrain.
    pub ice_caps: Handle<ShaderBuffer>,
    /// `array<DuneSea>` — dynamic active dune overlays rendered by the
    /// impostor shader over static terrain.
    pub active_dunes: Handle<ShaderBuffer>,
}
