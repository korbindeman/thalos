//! Forward-rendered star field.
//!
//! Replaces the cubemap skybox path. A single mesh carries one quad
//! per star in `thalos_celestial::Universe`; a custom WGSL material
//! billboards each quad in screen space and synthesises an HDR PSF
//! per fragment. Stars are forced to the reverse-Z far plane so they
//! always sit behind solar-system bodies in the depth buffer.
//!
//! The star mesh is built once at startup and attached to the scene;
//! the shader expresses positions relative to the camera every frame
//! so no per-frame CPU work is needed.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{Indices, Mesh, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use thalos_celestial::Universe;
use thalos_celestial::generate::{DefaultGenParams, generate_default};

use crate::rendering::CameraExposure;

/// Uniform buffer for the stars shader. Matches the `StarsParams`
/// struct in `assets/shaders/stars.wgsl`.
#[derive(Clone, Copy, ShaderType)]
pub struct StarsParams {
    /// Half-width of a magnitude-0 star's quad in pixels.
    pub pixel_radius: f32,
    /// Overall flux multiplier applied to every star.
    pub brightness: f32,
    /// Per-star size exponent. Larger → brighter stars grow more.
    pub size_gamma: f32,
    pub _pad0: f32,
}

impl Default for StarsParams {
    fn default() -> Self {
        Self {
            pixel_radius: 4.0,
            brightness: StarsParams::BASE_BRIGHTNESS,
            size_gamma: 0.50,
            _pad0: 0.0,
        }
    }
}

impl StarsParams {
    /// Star brightness at focus gain = 1 (~1 AU focus). The per-frame
    /// exposure system divides this by `CameraExposure.gain` so the
    /// stars keep a constant *perceived* brightness as focus moves
    /// outward — same trick the old `Skybox.brightness` used, just
    /// applied to our material's uniform.
    pub const BASE_BRIGHTNESS: f32 = 140.0;
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct StarsMaterial {
    #[uniform(0)]
    pub params: StarsParams,
}

impl Material for StarsMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/stars.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/stars.wgsl".into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/stars_prepass.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        "shaders/stars_prepass.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        // Additive blend: each star's PSF adds to the existing frame.
        AlphaMode::Add
    }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Custom vertex layout — only POSITION (direction), UV_0
        // (quad corner), and COLOR (linear rgb + magnitude flux).
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(2),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];

        // Stars are "at infinity." The vertex shader emits clip.z = 0
        // (reverse-Z far plane); `GreaterEqual` passes against the
        // cleared depth buffer (also 0) so stars fill empty sky, but
        // fails against any real body (whose NDC z is strictly > 0), so
        // planets occlude stars without any magic offset. Never write
        // depth — stars must not occlude one another or later geometry.
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = false;
            depth.depth_compare = CompareFunction::GreaterEqual;
        }

        Ok(())
    }
}

pub struct SkyRenderPlugin;

impl Plugin for SkyRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<StarsMaterial>::default())
            .add_plugins(MaterialPlugin::<GalaxyMaterial>::default())
            .add_systems(Startup, spawn_stars)
            .add_systems(
                Update,
                (update_stars_brightness, update_galaxy_uniform).in_set(crate::SimStage::Sync),
            );
    }
}

/// Marker component so the per-frame exposure system can find the
/// stars material handle without scanning every material entity.
#[derive(Component)]
struct StarsMesh;

/// Mirrors the old `update_skybox_brightness` system: scale the stars
/// material's `brightness` inversely with `CameraExposure.gain` so
/// star visibility stays constant as the camera focus moves between
/// bright inner-system bodies and dim outer ones. Without this, the
/// auto-exposure post step pulls the dim scene up and the constant
/// star flux blows out the frame.
fn update_stars_brightness(
    exposure: Res<CameraExposure>,
    handles: Query<&MeshMaterial3d<StarsMaterial>, With<StarsMesh>>,
    mut materials: ResMut<Assets<StarsMaterial>>,
) {
    let gain = exposure.gain.max(1.0e-3);
    let brightness = StarsParams::BASE_BRIGHTNESS / gain;
    for handle in &handles {
        if let Some(material) = materials.get_mut(&handle.0) {
            material.params.brightness = brightness;
        }
    }
}

/// Build the Universe, bake it into a mesh, and spawn a standalone
/// entity that the vertex shader places relative to the camera.
///
/// The entity does NOT parent to the camera — it lives in world space
/// with an identity transform. Every frame, the vertex shader reads
/// `view.world_position` and fans star quads out to a virtual infinity
/// from there, so camera movement is handled entirely in the shader.
fn spawn_stars(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut stars_materials: ResMut<Assets<StarsMaterial>>,
    mut galaxy_materials: ResMut<Assets<GalaxyMaterial>>,
) {
    let universe = generate_default(&DefaultGenParams::default());

    let stars_mesh = meshes.add(build_star_mesh(&universe));
    let stars_material = stars_materials.add(StarsMaterial {
        params: StarsParams::default(),
    });
    commands.spawn((
        StarsMesh,
        Mesh3d(stars_mesh),
        MeshMaterial3d(stars_material),
        Transform::IDENTITY,
        NoFrustumCulling,
    ));

    let galaxy_mesh = meshes.add(build_galaxy_mesh(&universe));
    let galaxy_material = galaxy_materials.add(GalaxyMaterial {
        params: GalaxyParams::default(),
    });
    commands.spawn((
        GalaxyMesh,
        Mesh3d(galaxy_mesh),
        MeshMaterial3d(galaxy_material),
        Transform::IDENTITY,
        NoFrustumCulling,
    ));
}

// ---------------------------------------------------------------------------
// Galaxy material
// ---------------------------------------------------------------------------

/// Uniform buffer for the galaxy shader. Matches `GalaxyParams` in
/// `assets/shaders/galaxy.wgsl`.
#[derive(Clone, Copy, ShaderType)]
pub struct GalaxyParams {
    /// Conversion from angular radius (radians) to pixel radius. The
    /// per-frame exposure system recomputes this from the current
    /// viewport + vertical FOV so galaxies keep a consistent
    /// on-screen size independent of resolution.
    pub pixel_radius_scale: f32,
    /// Clamp for unresolved galaxies so they don't vanish below 1 px.
    pub min_pixel_radius: f32,
    /// Overall flux multiplier (scales with camera exposure).
    pub brightness: f32,
    pub _pad0: f32,
}

impl Default for GalaxyParams {
    fn default() -> Self {
        Self {
            pixel_radius_scale: 2000.0,
            min_pixel_radius: 1.2,
            brightness: GalaxyParams::BASE_BRIGHTNESS,
            _pad0: 0.0,
        }
    }
}

impl GalaxyParams {
    // Galaxies spread their magnitude flux over many pixels in the
    // shader, but the Sérsic-concentrated normalisation lets the
    // bulge spike well above the disk. This multiplier is tuned so
    // featured spiral cores glow warmly above the tonemap knee
    // without clipping the halo.
    pub const BASE_BRIGHTNESS: f32 = 1_500.0;
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct GalaxyMaterial {
    #[uniform(0)]
    pub params: GalaxyParams,
}

impl Material for GalaxyMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/galaxy.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "shaders/galaxy.wgsl".into()
    }
    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/galaxy_prepass.wgsl".into()
    }
    fn prepass_fragment_shader() -> ShaderRef {
        "shaders/galaxy_prepass.wgsl".into()
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Add
    }

    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(1),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(2),
            Mesh::ATTRIBUTE_TANGENT.at_shader_location(3),
            Mesh::ATTRIBUTE_COLOR.at_shader_location(4),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        // See `StarsMaterial::specialize` — same reverse-Z far-plane
        // treatment.
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = false;
            depth.depth_compare = CompareFunction::GreaterEqual;
        }
        Ok(())
    }
}

#[derive(Component)]
struct GalaxyMesh;

/// Update the per-frame galaxy uniform: brightness follows
/// `CameraExposure.gain`, and the angular→pixel scale factor is
/// recomputed from the current primary window height plus the
/// scene's vertical FOV. Viewport changes (resize, aspect change)
/// therefore keep galaxies the same angular size on-screen.
fn update_galaxy_uniform(
    exposure: Res<CameraExposure>,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    projections: Query<&Projection, With<crate::camera::ActiveCamera>>,
    handles: Query<&MeshMaterial3d<GalaxyMaterial>, With<GalaxyMesh>>,
    mut materials: ResMut<Assets<GalaxyMaterial>>,
) {
    let Ok(window) = windows.single() else { return };
    let height_px = window.resolution.physical_height() as f32;
    let Ok(projection) = projections.single() else {
        return;
    };
    let vertical_fov = match projection {
        Projection::Perspective(p) => p.fov,
        _ => return,
    };
    // Pixels per radian along the vertical axis: screen-space size
    // of a source of angular radius `θ` in pixels is `θ · px_per_rad`.
    let px_per_rad = height_px / vertical_fov;

    let gain = exposure.gain.max(1.0e-3);
    let brightness = GalaxyParams::BASE_BRIGHTNESS / gain;

    for handle in &handles {
        if let Some(material) = materials.get_mut(&handle.0) {
            material.params.pixel_radius_scale = px_per_rad;
            material.params.brightness = brightness;
        }
    }
}

/// Build the galaxy mesh. Each galaxy uses 4 vertices + 6 indices.
/// Attribute packing:
///   POSITION : direction unit vector (3)
///   UV_0     : quad corner (2)
///   NORMAL   : (angular_radius_rad, sersic_n, unused) (3)
///   TANGENT  : (axis_ratio, cos_pa, sin_pa, 0) (4)
///   COLOR    : (rgb linear, magnitude_flux) (4)
fn build_galaxy_mesh(universe: &Universe) -> Mesh {
    let n = universe.galaxies.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut tangents: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    const CORNERS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

    for (i, galaxy) in universe.galaxies.iter().enumerate() {
        let dir = galaxy.position.normalize();
        let rgb = galaxy.linear_srgb();
        let flux = galaxy.magnitude_flux();
        let (sin_pa, cos_pa) = galaxy.position_angle_rad.sin_cos();
        for corner in CORNERS {
            positions.push([dir.x, dir.y, dir.z]);
            uvs.push(corner);
            normals.push([galaxy.effective_radius_rad, galaxy.sersic_n, 0.0]);
            tangents.push([galaxy.axis_ratio, cos_pa, sin_pa, 0.0]);
            colors.push([rgb[0], rgb[1], rgb[2], flux]);
        }
        let base = (i * 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

/// Build a mesh containing one quad per star.
///
/// Four vertices per star. `POSITION` stores the direction unit
/// vector (same for all four corners); `UV_0` stores the corner
/// offset in [-1, +1]² so the vertex shader can billboard the quad;
/// `COLOR` stores the linear-sRGB chromaticity in `rgb` and the
/// magnitude flux factor in `a`.
fn build_star_mesh(universe: &Universe) -> Mesh {
    let n = universe.stars.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 4);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * 4);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 6);

    const CORNERS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

    for (i, star) in universe.stars.iter().enumerate() {
        let dir = star.position.normalize();
        let rgb = star.linear_srgb();
        let flux = star.magnitude_flux();
        for corner in CORNERS {
            positions.push([dir.x, dir.y, dir.z]);
            uvs.push(corner);
            colors.push([rgb[0], rgb[1], rgb[2], flux]);
        }
        let base = (i * 4) as u32;
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
