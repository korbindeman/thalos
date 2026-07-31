//! Octahedral tree impostors — the far band of the tree LOD cascade.
//!
//! A far tree is drawn as a single camera-facing quad that samples a pre-baked
//! **hemisphere octahedral atlas** of the species: the tree captured from an
//! `N×N` grid of view directions (equator = ground-level views, pole = top-down
//! aerial views, since the player flies). The fragment maps the camera→tree view
//! direction to atlas coords, bilinearly blends the surrounding captured views,
//! alpha-tests, and lights the blended object-frame normal through the *same*
//! `thalos::lighting` sky model the mesh trees and ground use — so the
//! mesh→impostor handoff reads as one continuous forest.
//!
//! This module is the engine side: the runtime [`TreeImpostorMaterial`] +
//! `tree_impostor.wgsl`, the [`TreeBakeMaterial`] + `tree_bake.wgsl` used to
//! render the atlas at startup, and the pure helpers (atlas image, bounding
//! sphere, recentre, hemioctahedral decode, per-cell bake rotation) the
//! game-side library build orchestrates the bake with. The bake *rig* (cameras +
//! per-cell instances) lives in `thalos_runtime::rendering::vegetation`, next to the
//! `SpeciesLibrary`, exactly like the mesh-tree builders live here but their
//! driver lives in the game crate.

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::image::ImageSampler;
use bevy::math::{Mat3, Quat, Vec2, Vec3, Vec4};
use bevy::mesh::{Indices, Mesh, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    TextureDimension, TextureFormat, TextureUsages,
};
use bevy::shader::ShaderRef;

use crate::ground::tree_mesh::TreeMeshData;
use crate::ground::vegetation::GrassParams;

/// Maximum tree species the impostor atlas / uniform can hold. Trees beyond this
/// fall back to mesh LODs (no impostor). Keep in sync with the WGSL
/// `array<vec4<f32>, 4>` in `tree_impostor.wgsl`.
pub const IMPOSTOR_MAX_SPECIES: usize = 4;

// ---------------------------------------------------------------------------
// Atlas layout + image
// ---------------------------------------------------------------------------

/// Layout of an octahedral impostor atlas: an `cells × cells` grid of hemisphere
/// views per species, species stacked vertically. The albedo+coverage atlas and
/// the normal+depth atlas share this layout.
#[derive(Debug, Clone, Copy)]
pub struct ImpostorAtlasLayout {
    /// Views per octahedral axis (`N`).
    pub cells: u32,
    /// Pixels per captured cell.
    pub cell_px: u32,
    /// Tree species captured into the atlas.
    pub species: u32,
}

impl ImpostorAtlasLayout {
    pub fn width(&self) -> u32 {
        self.cells * self.cell_px
    }
    pub fn height(&self) -> u32 {
        self.cells * self.species.max(1) * self.cell_px
    }
}

/// Create one empty HDR atlas image sized for `layout`, configured as an
/// off-screen render target cleared transparent each bake. Linear
/// `Rgba16Float` so the baked vertex-colour albedo and the encoded
/// normal/depth round-trip without sRGB conversions, and filterable so the
/// runtime sampler can blend across captured views.
pub fn make_impostor_atlas(layout: ImpostorAtlasLayout) -> Image {
    let mut img = Image::new_fill(
        Extent3d {
            width: layout.width(),
            height: layout.height(),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 8], // one Rgba16Float texel (4×f16) = 8 bytes of transparent black
        TextureFormat::Rgba16Float,
        RenderAssetUsages::default(),
    );
    img.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    img.sampler = ImageSampler::linear();
    img
}

// ---------------------------------------------------------------------------
// Geometry helpers (bounding sphere, recentre, hemioctahedral, bake rotation)
// ---------------------------------------------------------------------------

/// Bounding sphere `(centre, radius)` of a species mesh in its authored frame
/// (trunk base at the origin, +Y up). The impostor card is sized from the
/// radius and centred at `base + up · centre.y · scale` at runtime.
pub fn tree_bounding_sphere(data: &TreeMeshData) -> (Vec3, f32) {
    if data.positions.is_empty() {
        return (Vec3::ZERO, 1.0);
    }
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for p in &data.positions {
        let v = Vec3::from_array(*p);
        min = min.min(v);
        max = max.max(v);
    }
    let center = (min + max) * 0.5;
    let mut r = 0.0f32;
    for p in &data.positions {
        r = r.max((Vec3::from_array(*p) - center).length());
    }
    (center, r.max(1.0e-3))
}

/// A `Mesh` copy of `data` recentred so its bounding-sphere centre sits at the
/// origin — the mesh handed to the bake instances, which are rotated about the
/// origin per captured view direction and scaled to fill their atlas cell.
/// `UV_1.y` carries the foliage-atlas leaf code (as in [`crate::ground::tree_mesh`]'s
/// standalone mesh) so the bake shader samples the *same* leaf shape + colour the
/// mesh trees do — without it the bake captures only the near-white vertex tint
/// and the impostors render as solid pale quads.
pub fn recenter_tree_mesh(data: &TreeMeshData, center: Vec3) -> Mesh {
    let positions: Vec<[f32; 3]> = data
        .positions
        .iter()
        .map(|p| (Vec3::from_array(*p) - center).to_array())
        .collect();
    let count = positions.len();
    let uv0 = vec![[0.0f32; 2]; count];
    let uv1: Vec<[f32; 2]> = data.leaf_code.iter().map(|&c| [0.0, c]).collect();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, data.colors.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv0);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, uv1);
    mesh.insert_indices(Indices::U32(data.indices.clone()));
    mesh
}

/// Decode a hemisphere-octahedral atlas coordinate `uv ∈ [0,1]²` to a unit
/// direction on the upper (`y ≥ 0`) hemisphere. Used to enumerate the captured
/// view directions at bake time; the WGSL `hemioct_encode` in
/// `tree_impostor.wgsl` is its exact inverse.
pub fn hemioct_decode(uv: Vec2) -> Vec3 {
    let f = uv * 2.0 - Vec2::ONE;
    let t = Vec2::new(f.x + f.y, f.x - f.y) * 0.5;
    Vec3::new(t.x, 1.0 - t.x.abs() - t.y.abs(), t.y).normalize()
}

/// Rotation that maps the object so that view direction `d` (object→camera) lines
/// up with the bake camera's +Z (the camera looks down −Z at the cell). The card
/// basis derived from `d` (`right = up_ref × d`, `up = d × right`) maps to screen
/// X/Y, identically to the runtime `view_basis`, so a sampled cell's card UV
/// matches the captured projection.
pub fn impostor_bake_rotation(d: Vec3) -> Quat {
    let fwd = d.normalize_or(Vec3::Z);
    let up_ref = if fwd.y.abs() < 0.999 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let right = up_ref.cross(fwd).normalize();
    let up = fwd.cross(right);
    // basis maps camera-space → object-space (cols right/up/fwd); object→camera
    // is its inverse (transpose for an orthonormal basis).
    Quat::from_mat3(&Mat3::from_cols(right, up, fwd)).inverse()
}

// ---------------------------------------------------------------------------
// Runtime impostor material
// ---------------------------------------------------------------------------

/// Per-impostor-atlas uniform: grid dimensions + the per-species bounding
/// geometry the billboard sizes from. Mirrors `ImpostorParams` in
/// `tree_impostor.wgsl` — field order is load-bearing.
#[derive(Clone, Copy, ShaderType)]
pub struct ImpostorParams {
    /// x = cells (`N`), y = species count, z = alpha cutoff, w = v-flip (0/1).
    pub grid: Vec4,
    /// x = cell fill fraction (the bounding sphere occupies this fraction of the
    /// cell; the rest is the anti-bleed gutter); y/z/w reserved.
    pub atlas: Vec4,
    /// Per species: x = bounding-sphere radius (authored units), y = bounding
    /// centre height along +up (authored units); z/w reserved.
    pub species_geo: [Vec4; IMPOSTOR_MAX_SPECIES],
}

impl Default for ImpostorParams {
    fn default() -> Self {
        Self {
            grid: Vec4::new(8.0, 1.0, 0.4, 0.0),
            atlas: Vec4::new(0.84, 0.0, 0.0, 0.0),
            species_geo: [Vec4::new(1.0, 0.0, 0.0, 0.0); IMPOSTOR_MAX_SPECIES],
        }
    }
}

/// Extension for far tree impostors: billboards one quad per tree (baked by
/// [`combine_impostor_tile_mesh`](crate::ground::scatter::combine_impostor_tile_mesh))
/// and octahedral-samples the atlas. Reuses [`GrassParams`] for the
/// craft-anchor scale-fade, so the mesh→impostor handoff and the far cull are
/// seamless and zoom-independent, exactly like the mesh `TreeMaterial`.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct TreeImpostorExtension {
    #[uniform(100)]
    pub params: GrassParams,
    #[uniform(101)]
    pub impostor: ImpostorParams,
    #[texture(102)]
    #[sampler(103)]
    pub albedo: Handle<Image>,
    #[texture(104)]
    #[sampler(105)]
    pub normal: Handle<Image>,
    /// Cloud sun-transmittance cascade — the same map/block the tile ground and
    /// the mesh trees sample, fanned in per frame by the game's tree driver.
    /// NEW versus the spine impostor, which never dimmed under the deck.
    #[texture(106)]
    #[sampler(107)]
    pub cloud_shadow_map: Handle<Image>,
    #[uniform(108)]
    pub cloud_shadow: crate::clouds::CloudShadowBlock,
    /// Cascaded sun-shadow receive — impostor rings 0–1 CAST into cascades 1–2
    /// (they sit on `SHADOW_CASTER_LAYER` from 1.2 km out, and the cascades are
    /// pinned to cover exactly that band), so they must SAMPLE them too or a
    /// tree casts a shadow it cannot itself receive: bright trees on dark
    /// ground across a shadowed valley, and a shadowed→lit pop at the 1.2 km
    /// mesh↔impostor swap (reviews/20260730T011353Z §8). Past cascade 2's
    /// reach the sampler fades out and the W12 horizon term owns the far
    /// field, as before.
    #[uniform(109)]
    pub shadow: crate::ShadowCascadeBlock,
    /// Cascade 0 — the ±64 m near box added 2026-07-31. Bound at the END of
    /// this material's range rather than renumbered into the near→far slot:
    /// only the ARGUMENT order at the `thalos::shadow` call site is
    /// ordering-significant, so shifting live binding indices would be risk
    /// with no payoff. Field name still equals the cascade index.
    #[texture(113, sample_type = "depth")]
    pub sun_shadow_map_0: Handle<Image>,
    #[texture(110, sample_type = "depth")]
    pub sun_shadow_map_1: Handle<Image>,
    #[texture(111, sample_type = "depth")]
    pub sun_shadow_map_2: Handle<Image>,
    #[texture(112, sample_type = "depth")]
    pub sun_shadow_map_3: Handle<Image>,
}

/// Far-band impostor card on the standard path — `ExtendedMaterial` like the
/// mesh trees ([`tree_base_material`](crate::ground::tree_material::tree_base_material)
/// supplies the shared foliage base: double-sided, diffuse-transmitting), so
/// the mesh→impostor handoff stays inside the one lighting universe.
pub type TreeImpostorMaterial =
    bevy::pbr::ExtendedMaterial<bevy::pbr::StandardMaterial, TreeImpostorExtension>;

/// Wrap a [`TreeImpostorExtension`] into the full impostor material.
pub fn tree_impostor_material(extension: TreeImpostorExtension) -> TreeImpostorMaterial {
    TreeImpostorMaterial {
        base: crate::ground::tree_material::tree_base_material(),
        extension,
    }
}

impl bevy::pbr::MaterialExtension for TreeImpostorExtension {
    fn vertex_shader() -> ShaderRef {
        "shaders/tree_impostor_standard.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/tree_impostor_standard.wgsl".into()
    }

    // Opaque (depth-written) with a manual coverage `discard` in the fragment.
    // Opaque keeps the prepass binding-agnostic: the standard prepass uses
    // POSITION (all four quad corners share the tree base → degenerate, draws
    // nothing), so impostors never touch the prepass bind group.
    fn alpha_mode() -> Option<AlphaMode> {
        Some(AlphaMode::Opaque)
    }
}

// ---------------------------------------------------------------------------
// Bake material
// ---------------------------------------------------------------------------

/// Bake uniform. `mode.x`: 0 = albedo+coverage, 1 = object-local normal + depth.
/// `mode.y` = depth scale (`0.5 / cell-fit`), mapping the cell-space view depth
/// into `[0,1]`.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct BakeParams {
    pub mode: Vec4,
}

/// Material used only at startup to render the octahedral atlas. One instance
/// per (species, view cell) renders the recentred species mesh; `mode` selects
/// the albedo atlas (leaf colour × tint + leaf-shaped coverage) or the normal
/// atlas (object-local normal + depth). The same procedural foliage atlas the
/// mesh trees sample is bound so the captured leaf shape/colour matches the
/// mesh→impostor handoff. Object-local — not world — normals are stored so the
/// runtime impostor re-lights each tree in its terrain frame.
#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct TreeBakeMaterial {
    #[uniform(0)]
    pub params: BakeParams,
    /// Procedural foliage atlas (leaf clusters + shell + bark), shared with
    /// [`TreeMaterial`](crate::ground::TreeMaterial).
    #[texture(1)]
    #[sampler(2)]
    pub atlas: Handle<Image>,
}

impl Material for TreeBakeMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/tree_bake.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_body_render/ground/tree_bake.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

pub(crate) fn embed_tree_impostor_shaders(app: &mut App) {
    // The runtime impostor shader moved to the standard path — asset-loaded
    // (hot-reloadable) `shaders/tree_impostor_standard.wgsl`. Only the startup
    // bake shader stays embedded.
    embedded_asset!(app, "tree_bake.wgsl");
}

/// Lean plugin to render [`TreeImpostorMaterial`] (and run the [`TreeBakeMaterial`]
/// off-screen bake) in a standalone / headless app — the object preview, examples
/// — **without** the full UDLOD terrain stack. Mirrors
/// [`TreeMaterialPlugin`](crate::ground::TreeMaterialPlugin): registers both
/// materials, embeds their shaders, and ensures the shared shader libraries
/// (`thalos::lighting` / `thalos::foliage`) are present. The game gets all this
/// from `ThalosTerrainPlugin`; this is the minimal entry point for tools that want
/// to verify the mesh↔impostor handoff in isolation.
pub struct TreeImpostorMaterialPlugin;

impl Plugin for TreeImpostorMaterialPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::shading::PlanetLightingPlugin>() {
            app.add_plugins(crate::shading::PlanetLightingPlugin);
        }
        app.add_plugins(bevy::pbr::MaterialPlugin::<TreeImpostorMaterial>::default());
        app.add_plugins(bevy::pbr::MaterialPlugin::<TreeBakeMaterial>::default());
        embed_tree_impostor_shaders(app);
    }
}
