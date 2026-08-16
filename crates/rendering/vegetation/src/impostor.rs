//! Shared octahedral foliage impostors.
//!
//! The mechanism is topology-independent: callers provide local tree roots,
//! terrain-up vectors, and an ordered species library. This crate owns the
//! atlas layout, one-shot bake rig, standard-path material, and four-vertex
//! batch representation. Planar and planetary adapters remain responsible for
//! choosing and anchoring the tiles that feed it.

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::camera::visibility::RenderLayers;
use bevy::camera::{Hdr, ImageRenderTarget, RenderTarget, ScalingMode};
use bevy::core_pipeline::tonemapping::Tonemapping;
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

use crate::TreeMeshData;

/// Maximum species addressable by one atlas/material block.
pub const IMPOSTOR_MAX_SPECIES: usize = 4;

/// One selected woody root in the caller's bounded local tile frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImpostorInstance {
    pub position: Vec3,
    pub up: Vec3,
    pub scale: f32,
    pub species: u32,
    pub tint: Vec3,
    pub yaw: f32,
}

/// Build one batched impostor mesh: exactly four degenerate vertices and two
/// triangles per root. The vertex shader expands each root into a view-facing
/// card; no authored tree vertices are duplicated into streamed tiles.
pub fn combine_impostor_mesh(instances: &[ImpostorInstance]) -> Option<Mesh> {
    const CORNERS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut positions = Vec::with_capacity(instances.len() * 4);
    let mut normals = Vec::with_capacity(instances.len() * 4);
    let mut colors = Vec::with_capacity(instances.len() * 4);
    let mut uv0 = Vec::with_capacity(instances.len() * 4);
    let mut uv1 = Vec::with_capacity(instances.len() * 4);
    let mut indices = Vec::with_capacity(instances.len() * 6);

    for instance in instances {
        if instance.species as usize >= IMPOSTOR_MAX_SPECIES {
            continue;
        }
        let start = positions.len() as u32;
        let yaw01 = (instance.yaw / std::f32::consts::TAU).rem_euclid(1.0);
        for corner in CORNERS {
            positions.push(instance.position.to_array());
            normals.push(instance.up.normalize_or(Vec3::Y).to_array());
            colors.push([instance.tint.x, instance.tint.y, instance.tint.z, yaw01]);
            uv0.push(corner);
            uv1.push([instance.scale.max(0.0), instance.species as f32]);
        }
        indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    }

    if positions.is_empty() {
        return None;
    }
    Some(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uv0)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, uv1)
        .with_inserted_indices(Indices::U32(indices)),
    )
}

#[derive(Debug, Clone, Copy)]
pub struct ImpostorAtlasLayout {
    pub cells: u32,
    pub cell_px: u32,
    pub species: u32,
}

impl ImpostorAtlasLayout {
    pub fn width(self) -> u32 {
        self.cells * self.cell_px
    }

    pub fn height(self) -> u32 {
        self.cells * self.species.max(1) * self.cell_px
    }
}

pub fn make_impostor_atlas(layout: ImpostorAtlasLayout) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: layout.width(),
            height: layout.height(),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 8],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    image.sampler = ImageSampler::linear();
    image
}

pub fn tree_bounding_sphere(data: &TreeMeshData) -> (Vec3, f32) {
    if data.positions.is_empty() {
        return (Vec3::ZERO, 1.0);
    }
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for position in &data.positions {
        let position = Vec3::from_array(*position);
        min = min.min(position);
        max = max.max(position);
    }
    let center = (min + max) * 0.5;
    let radius = data
        .positions
        .iter()
        .map(|position| (Vec3::from_array(*position) - center).length())
        .fold(0.0, f32::max);
    (center, radius.max(1.0e-3))
}

pub fn recenter_tree_mesh(data: &TreeMeshData, center: Vec3) -> Mesh {
    let positions: Vec<_> = data
        .positions
        .iter()
        .map(|position| (Vec3::from_array(*position) - center).to_array())
        .collect();
    let count = positions.len();
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, data.colors.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0; 2]; count])
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_UV_1,
        data.leaf_code
            .iter()
            .map(|code| [0.0, *code])
            .collect::<Vec<_>>(),
    )
    .with_inserted_indices(Indices::U32(data.indices.clone()))
}

pub fn hemioct_decode(uv: Vec2) -> Vec3 {
    let f = uv * 2.0 - Vec2::ONE;
    let t = Vec2::new(f.x + f.y, f.x - f.y) * 0.5;
    Vec3::new(t.x, 1.0 - t.x.abs() - t.y.abs(), t.y).normalize()
}

pub fn impostor_bake_rotation(direction: Vec3) -> Quat {
    let forward = direction.normalize_or(Vec3::Z);
    let up_reference = if forward.y.abs() < 0.999 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let right = up_reference.cross(forward).normalize();
    let up = forward.cross(right);
    Quat::from_mat3(&Mat3::from_cols(right, up, forward)).inverse()
}

#[derive(Clone, Copy, ShaderType)]
pub struct ImpostorParams {
    pub grid: Vec4,
    pub atlas: Vec4,
    pub species_geo: [Vec4; IMPOSTOR_MAX_SPECIES],
}

impl Default for ImpostorParams {
    fn default() -> Self {
        Self {
            grid: Vec4::new(8.0, 1.0, 0.35, 0.0),
            atlas: Vec4::new(0.84, 0.0, 0.0, 0.0),
            species_geo: [Vec4::new(1.0, 0.0, 0.0, 0.0); IMPOSTOR_MAX_SPECIES],
        }
    }
}

/// View-dependent handoff owned by an adapter. `fade` is
/// `(near_m, far_m, half_band_m, enabled)` and `anchor.xyz` offsets the render
/// camera to the stable distance reference.
#[derive(Clone, Copy, ShaderType)]
pub struct ImpostorViewParams {
    pub fade: Vec4,
    pub anchor: Vec4,
}

impl Default for ImpostorViewParams {
    fn default() -> Self {
        Self {
            fade: Vec4::new(-1.0e9, 1.0e9, 1.0, 0.0),
            anchor: Vec4::ZERO,
        }
    }
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct FoliageImpostorExtension {
    #[uniform(100)]
    pub view: ImpostorViewParams,
    #[uniform(101)]
    pub impostor: ImpostorParams,
    #[texture(102)]
    #[sampler(103)]
    pub albedo: Handle<Image>,
    #[texture(104)]
    #[sampler(105)]
    pub normal: Handle<Image>,
}

pub type FoliageImpostorMaterial =
    bevy::pbr::ExtendedMaterial<bevy::pbr::StandardMaterial, FoliageImpostorExtension>;

pub fn foliage_impostor_material(extension: FoliageImpostorExtension) -> FoliageImpostorMaterial {
    FoliageImpostorMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            alpha_mode: AlphaMode::Opaque,
            double_sided: true,
            cull_mode: None,
            perceptual_roughness: 0.95,
            reflectance: 0.32,
            diffuse_transmission: 0.35,
            ..default()
        },
        extension,
    }
}

impl bevy::pbr::MaterialExtension for FoliageImpostorExtension {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_vegetation/impostor_standard.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_vegetation/impostor_standard.wgsl".into()
    }

    fn alpha_mode() -> Option<AlphaMode> {
        Some(AlphaMode::Opaque)
    }

    fn enable_prepass() -> bool {
        false
    }
}

#[derive(Clone, Copy, ShaderType, Default)]
pub struct BakeParams {
    pub mode: Vec4,
}

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct TreeBakeMaterial {
    #[uniform(0)]
    pub params: BakeParams,
    #[texture(1)]
    #[sampler(2)]
    pub atlas: Handle<Image>,
}

impl Material for TreeBakeMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://thalos_vegetation/impostor_bake.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://thalos_vegetation/impostor_bake.wgsl".into()
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

pub struct FoliageImpostorMaterialPlugin;

impl Plugin for FoliageImpostorMaterialPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<FoliageImpostorBakePlugin>() {
            app.add_plugins(FoliageImpostorBakePlugin);
        }
        app.add_plugins(MaterialPlugin::<FoliageImpostorMaterial>::default());
        embedded_asset!(app, "impostor_standard.wgsl");
    }
}

/// Minimal atlas-bake mechanism for adapters that provide their own runtime
/// material (the planetary renderer does this to add cloud and sun shadows).
pub struct FoliageImpostorBakePlugin;

impl Plugin for FoliageImpostorBakePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<thalos_body_shading::PlanetLightingPlugin>() {
            app.add_plugins(thalos_body_shading::PlanetLightingPlugin);
        }
        app.add_plugins(MaterialPlugin::<TreeBakeMaterial>::default());
        embedded_asset!(app, "impostor_bake.wgsl");
    }
}

#[derive(Component)]
pub struct ImpostorBakeRig;

#[derive(Debug, Clone, Copy)]
pub struct ImpostorBakeConfig {
    pub layout: ImpostorAtlasLayout,
    pub cell_fill: f32,
    pub alpha_cutoff: f32,
    pub albedo_layer: usize,
    pub normal_layer: usize,
}

#[derive(Clone)]
pub struct ImpostorAtlas {
    pub albedo: Handle<Image>,
    pub normal: Handle<Image>,
    pub params: ImpostorParams,
    pub max_extent_m: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_impostor_bake_rig(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    meshes: &mut Assets<Mesh>,
    bake_materials: &mut Assets<TreeBakeMaterial>,
    species: &[&TreeMeshData],
    foliage_atlas: Handle<Image>,
    config: ImpostorBakeConfig,
) -> ImpostorAtlas {
    let species_count = species.len().min(IMPOSTOR_MAX_SPECIES) as u32;
    let layout = ImpostorAtlasLayout {
        species: species_count,
        ..config.layout
    };
    let albedo = images.add(make_impostor_atlas(layout));
    let normal = images.add(make_impostor_atlas(layout));
    let mut species_geo = [Vec4::ZERO; IMPOSTOR_MAX_SPECIES];
    let mut max_extent_m = 0.0f32;

    for (layer, data) in species.iter().take(IMPOSTOR_MAX_SPECIES).enumerate() {
        let (center, radius) = tree_bounding_sphere(data);
        species_geo[layer] = Vec4::new(radius, center.y, 0.0, 0.0);
        max_extent_m = max_extent_m.max(center.y + radius);
    }

    let params = ImpostorParams {
        grid: Vec4::new(
            layout.cells as f32,
            species_count.max(1) as f32,
            config.alpha_cutoff,
            0.0,
        ),
        atlas: Vec4::new(config.cell_fill, 0.0, 0.0, 0.0),
        species_geo,
    };

    if species_count == 0 {
        return ImpostorAtlas {
            albedo,
            normal,
            params,
            max_extent_m,
        };
    }

    let cell_fit = config.cell_fill * 0.5;
    let depth_scale = 0.5 / cell_fit.max(1.0e-4);
    let albedo_material = bake_materials.add(TreeBakeMaterial {
        params: BakeParams {
            mode: Vec4::new(0.0, depth_scale, 0.0, 0.0),
        },
        atlas: foliage_atlas.clone(),
    });
    let normal_material = bake_materials.add(TreeBakeMaterial {
        params: BakeParams {
            mode: Vec4::new(1.0, depth_scale, 0.0, 0.0),
        },
        atlas: foliage_atlas,
    });

    for (layer, data) in species.iter().take(IMPOSTOR_MAX_SPECIES).enumerate() {
        let (center, radius) = tree_bounding_sphere(data);
        let mesh = meshes.add(recenter_tree_mesh(data, center));
        let scale = Vec3::splat(cell_fit / radius);
        for row in 0..layout.cells {
            for column in 0..layout.cells {
                let uv = Vec2::new(
                    (column as f32 + 0.5) / layout.cells as f32,
                    (row as f32 + 0.5) / layout.cells as f32,
                );
                let transform = Transform {
                    translation: Vec3::new(
                        column as f32 + 0.5,
                        (layer as u32 * layout.cells + row) as f32 + 0.5,
                        0.0,
                    ),
                    rotation: impostor_bake_rotation(hemioct_decode(uv)),
                    scale,
                };
                commands.spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(albedo_material.clone()),
                    transform,
                    Visibility::Visible,
                    RenderLayers::layer(config.albedo_layer),
                    ImpostorBakeRig,
                    Name::new("Foliage impostor bake albedo"),
                ));
                commands.spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(normal_material.clone()),
                    transform,
                    Visibility::Visible,
                    RenderLayers::layer(config.normal_layer),
                    ImpostorBakeRig,
                    Name::new("Foliage impostor bake normal"),
                ));
            }
        }
    }

    let grid_width = layout.cells as f32;
    let grid_height = (layout.cells * species_count.max(1)) as f32;
    let center = Vec3::new(grid_width * 0.5, grid_height * 0.5, 0.0);
    let camera = |order, layer, target, name| {
        (
            Camera3d::default(),
            Camera {
                order,
                clear_color: ClearColorConfig::Custom(Color::NONE),
                ..default()
            },
            Hdr,
            Tonemapping::None,
            RenderTarget::Image(ImageRenderTarget::from(target)),
            Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::Fixed {
                    width: grid_width,
                    height: grid_height,
                },
                near: 0.1,
                far: 100.0,
                ..OrthographicProjection::default_3d()
            }),
            Transform::from_translation(center + Vec3::Z * 10.0).looking_at(center, Vec3::Y),
            RenderLayers::layer(layer),
            ImpostorBakeRig,
            Name::new(name),
        )
    };
    commands.spawn(camera(
        -20,
        config.albedo_layer,
        albedo.clone(),
        "Foliage impostor bake camera albedo",
    ));
    commands.spawn(camera(
        -19,
        config.normal_layer,
        normal.clone(),
        "Foliage impostor bake camera normal",
    ));

    ImpostorAtlas {
        albedo,
        normal,
        params,
        max_extent_m,
    }
}

pub fn despawn_impostor_bake_rig(
    commands: &mut Commands,
    rig: &Query<Entity, With<ImpostorBakeRig>>,
) {
    for entity in rig {
        commands.entity(entity).despawn();
    }
}

#[cfg(test)]
mod tests {
    use bevy::mesh::VertexAttributeValues;

    use super::*;

    #[test]
    fn impostor_batch_cost_is_four_vertices_per_root() {
        let instances = (0..32)
            .map(|index| ImpostorInstance {
                position: Vec3::new(index as f32, 0.0, 0.0),
                up: Vec3::Y,
                scale: 1.0,
                species: 0,
                tint: Vec3::ONE,
                yaw: 0.0,
            })
            .collect::<Vec<_>>();
        let mesh = combine_impostor_mesh(&instances).unwrap();
        let VertexAttributeValues::Float32x3(positions) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        else {
            panic!("impostor positions must be Float32x3")
        };
        assert_eq!(positions.len(), instances.len() * 4);
        assert_eq!(mesh.indices().unwrap().len(), instances.len() * 6);
    }

    #[test]
    fn hemisphere_mapping_covers_equator_and_pole() {
        let center = hemioct_decode(Vec2::splat(0.5));
        assert!(center.distance(Vec3::Y) < 1.0e-5);
        for uv in [Vec2::ZERO, Vec2::X, Vec2::Y, Vec2::ONE] {
            assert!(hemioct_decode(uv).y.abs() < 1.0e-5);
        }
    }
}
