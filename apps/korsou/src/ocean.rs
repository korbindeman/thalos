use bevy::{
    asset::RenderAssetUsages,
    color::LinearRgba,
    image::{ImageSampler, ImageSamplerDescriptor},
    light::NotShadowCaster,
    mesh::{Indices, PrimitiveTopology},
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::{AsBindGroup, Extent3d, TextureDimension, TextureFormat},
    shader::ShaderRef,
};
use thalos_ocean::{
    OceanMechanismPlugin, RenderFrameTime, bake_ocean_slope_texture, project_ocean_frame,
};
use thalos_world::OceanState;

use crate::{
    camera::{TerrainCamera, TerrainCameraSet},
    terrain::TerrainDataset,
};

pub(crate) const OCEAN_SHADER: &str = "korsou://shaders/ocean.wgsl";
const OCEAN_GRID_CELLS: u32 = 128;
const OCEAN_CLIP_LEVELS: u32 = 13;
const OCEAN_BASE_CELL_SIZE_M: f32 = 2.0;
const COAST_TEXTURE_SIZE: usize = 1024;
const COAST_DISTANCE_RANGE_M: f32 = 4_000.0;

pub struct OceanPlugin;

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<OceanMechanismPlugin>() {
            app.add_plugins(OceanMechanismPlugin);
        }
        app.insert_resource(KorsouOceanState(OceanState::MODERATE))
            .add_plugins(MaterialPlugin::<OceanMaterial>::default())
            .add_systems(Startup, setup_ocean)
            .add_systems(
                Update,
                (
                    center_ocean_on_camera.after(TerrainCameraSet::Movement),
                    update_ocean_time,
                ),
            );
    }
}

type OceanMaterial = ExtendedMaterial<StandardMaterial, OceanExtension>;

#[derive(Component)]
struct OceanSurface;

#[derive(Resource)]
struct OceanMaterialHandle(Handle<OceanMaterial>);

#[derive(Resource)]
struct KorsouOceanState(OceanState);

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct OceanExtension {
    #[uniform(100)]
    deep_color: LinearRgba,
    #[uniform(100)]
    shelf_color: LinearRgba,
    #[uniform(100)]
    shallow_color: LinearRgba,
    #[uniform(100)]
    slope_amplitudes: Vec4,
    #[uniform(100)]
    low_phase: Vec4,
    #[uniform(100)]
    high_phase: Vec4,
    #[uniform(100)]
    surface_wavelengths_m: Vec4,
    #[uniform(100)]
    surface_amplitudes_m: Vec4,
    #[uniform(100)]
    surface_phases_rad: Vec4,
    #[uniform(100)]
    previous_surface_phases_rad: Vec4,
    #[uniform(100)]
    wind_and_foam: Vec4,
    #[uniform(100)]
    coast_bounds: Vec4,
    #[uniform(100)]
    shore_bounds: Vec4,
    #[uniform(100)]
    shore_params: Vec4,
    #[uniform(100)]
    time: Vec4,
    #[texture(101)]
    #[sampler(102)]
    slope_texture: Handle<Image>,
    #[texture(103)]
    #[sampler(104)]
    coast_distance_texture: Handle<Image>,
    #[texture(105)]
    #[sampler(106)]
    shore_properties_texture: Handle<Image>,
}

impl MaterialExtension for OceanExtension {
    fn vertex_shader() -> ShaderRef {
        OCEAN_SHADER.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        OCEAN_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        OCEAN_SHADER.into()
    }
}

fn setup_ocean(
    mut commands: Commands,
    dataset: Res<TerrainDataset>,
    sea_state: Res<KorsouOceanState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<OceanMaterial>>,
) {
    let frame_time = RenderFrameTime::new(0.0, 0.0).expect("zero ocean frame time is valid");
    let projection = project_ocean_frame(&sea_state.0, 9.81, frame_time);
    let slope_texture = images.add(bake_ocean_slope_texture());
    let coast_distance_texture = images.add(bake_coast_distance_texture(&dataset));
    let shore_properties_texture = images.add(bake_shore_properties_texture(&dataset));
    let bounds = dataset.metadata.grid_bounds_local_m;
    let coast_bounds = Vec4::new(
        bounds[0] as f32,
        bounds[1] as f32,
        (bounds[2] - bounds[0]) as f32,
        (bounds[3] - bounds[1]) as f32,
    );
    let shore = &dataset.metadata.coastline;
    let shore_bounds = Vec4::new(
        shore.distance_bounds_local_m[0] as f32,
        shore.distance_bounds_local_m[1] as f32,
        (shore.distance_bounds_local_m[2] - shore.distance_bounds_local_m[0]) as f32,
        (shore.distance_bounds_local_m[3] - shore.distance_bounds_local_m[1]) as f32,
    );

    let material = materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.018, 0.115, 0.16),
            perceptual_roughness: 0.28,
            metallic: 0.0,
            // Water's normal-incidence F0 is about 0.02. Bevy maps this as
            // F0 = 0.16 * reflectance^2.
            reflectance: 0.35,
            ..default()
        },
        extension: OceanExtension {
            deep_color: Color::srgb(0.012, 0.105, 0.15).into(),
            shelf_color: Color::srgb(0.008, 0.30, 0.38).into(),
            shallow_color: Color::srgb(0.035, 0.48, 0.50).into(),
            slope_amplitudes: projection.spectrum.slope_amplitudes,
            low_phase: projection.spectrum.low_phase,
            high_phase: projection.spectrum.high_phase,
            surface_wavelengths_m: projection.current_surface.wavelengths_m,
            surface_amplitudes_m: projection.current_surface.amplitudes_m,
            surface_phases_rad: projection.current_surface.phases_rad,
            previous_surface_phases_rad: projection.previous_surface.phases_rad,
            // Prevailing wave axis in local XZ, then foam slope onset and
            // maximum represented coast distance.
            wind_and_foam: Vec4::new(
                -0.91,
                0.414,
                sea_state.0.foam_slope_onset,
                COAST_DISTANCE_RANGE_M,
            ),
            coast_bounds,
            shore_bounds,
            shore_params: Vec4::new(shore.distance_clamp_m as f32, 3_000.0, 6_000.0, 0.0),
            time: Vec4::ZERO,
            slope_texture,
            coast_distance_texture,
            shore_properties_texture,
        },
    });
    commands.insert_resource(OceanMaterialHandle(material.clone()));

    for level in 0..OCEAN_CLIP_LEVELS {
        let cell_size_m = OCEAN_BASE_CELL_SIZE_M * (1u32 << level) as f32;
        commands.spawn((
            OceanSurface,
            Mesh3d(meshes.add(build_ocean_clip_mesh(cell_size_m, level > 0))),
            MeshMaterial3d(material.clone()),
            Transform::IDENTITY,
            NotShadowCaster,
            Name::new(format!("Caribbean Sea clip level {level}")),
        ));
    }
}

fn update_ocean_time(
    time: Res<Time>,
    sea_state: Res<KorsouOceanState>,
    material: Res<OceanMaterialHandle>,
    mut materials: ResMut<Assets<OceanMaterial>>,
) {
    if let Some(mut material) = materials.get_mut(&material.0) {
        let now_s = time.elapsed_secs_f64();
        let previous_s = now_s - time.delta_secs_f64();
        let frame_time = RenderFrameTime::new(previous_s, now_s)
            .expect("Bevy elapsed time must be finite and monotonic");
        let projection = project_ocean_frame(&sea_state.0, 9.81, frame_time);
        material.extension.slope_amplitudes = projection.spectrum.slope_amplitudes;
        material.extension.low_phase = projection.spectrum.low_phase;
        material.extension.high_phase = projection.spectrum.high_phase;
        material.extension.surface_wavelengths_m = projection.current_surface.wavelengths_m;
        material.extension.surface_amplitudes_m = projection.current_surface.amplitudes_m;
        material.extension.surface_phases_rad = projection.current_surface.phases_rad;
        material.extension.previous_surface_phases_rad = projection.previous_surface.phases_rad;
        material.extension.time.x = frame_time.current_epoch_s() as f32;
        material.extension.time.y = frame_time.delta_s() as f32;
    }
}

fn center_ocean_on_camera(
    camera: Single<&TerrainCamera>,
    mut ocean: Query<&mut Transform, With<OceanSurface>>,
) {
    for mut transform in &mut ocean {
        transform.translation.x = camera.position_m.x as f32;
        transform.translation.z = camera.position_m.z as f32;
    }
}

fn build_ocean_clip_mesh(cell_size_m: f32, ring: bool) -> Mesh {
    let side = OCEAN_GRID_CELLS + 1;
    let half_extent_m = OCEAN_GRID_CELLS as f32 * cell_size_m * 0.5;
    let mut positions = Vec::with_capacity((side * side) as usize);
    let mut normals = Vec::with_capacity((side * side) as usize);
    for z in 0..side {
        for x in 0..side {
            positions.push([
                -half_extent_m + x as f32 * cell_size_m,
                0.0,
                -half_extent_m + z as f32 * cell_size_m,
            ]);
            normals.push([0.0, 1.0, 0.0]);
        }
    }

    let hole_start = OCEAN_GRID_CELLS / 4;
    let hole_end = OCEAN_GRID_CELLS - hole_start;
    let mut indices = Vec::with_capacity((OCEAN_GRID_CELLS * OCEAN_GRID_CELLS * 6) as usize);
    for z in 0..OCEAN_GRID_CELLS {
        for x in 0..OCEAN_GRID_CELLS {
            if ring && x >= hole_start && x < hole_end && z >= hole_start && z < hole_end {
                continue;
            }
            let quad = z * side + x;
            indices.extend_from_slice(&[
                quad + side + 1,
                quad + 1,
                quad + side,
                quad,
                quad + side,
                quad + 1,
            ]);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_indices(Indices::U32(indices))
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
}

fn bake_coast_distance_texture(dataset: &TerrainDataset) -> Image {
    let bounds = dataset.metadata.grid_bounds_local_m;
    let width_m = bounds[2] - bounds[0];
    let height_m = bounds[3] - bounds[1];
    let mut land = vec![false; COAST_TEXTURE_SIZE * COAST_TEXTURE_SIZE];
    for z in 0..COAST_TEXTURE_SIZE {
        let local_z = bounds[1] + height_m * z as f64 / (COAST_TEXTURE_SIZE - 1) as f64;
        for x in 0..COAST_TEXTURE_SIZE {
            let local_x = bounds[0] + width_m * x as f64 / (COAST_TEXTURE_SIZE - 1) as f64;
            land[z * COAST_TEXTURE_SIZE + x] = dataset.is_land(local_x, local_z);
        }
    }
    let pixel_size_m = ((width_m + height_m) * 0.5 / (COAST_TEXTURE_SIZE - 1) as f64) as f32;
    let data = encode_distance_from_land(
        &land,
        COAST_TEXTURE_SIZE,
        COAST_TEXTURE_SIZE,
        pixel_size_m,
        COAST_DISTANCE_RANGE_M,
    );
    let mut image = Image::new(
        Extent3d {
            width: COAST_TEXTURE_SIZE as u32,
            height: COAST_TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    image
}

fn bake_shore_properties_texture(dataset: &TerrainDataset) -> Image {
    let coast = &dataset.metadata.coastline;
    let distances = dataset.coast_distance_samples();
    let width = coast.distance_width;
    let height = coast.distance_height;
    let distance_at =
        |x: usize, z: usize| f32::from(distances[z * width + x]) / coast.distance_units_per_metre;
    let wave_direction = Vec2::new(-0.91, 0.414).normalize();
    let range_m = coast.distance_clamp_m as f32;
    let spacing_m = coast.distance_spacing_m;
    let bounds = coast.distance_bounds_local_m;
    let mut data = Vec::with_capacity(width * height * 4);

    for z in 0..height {
        for x in 0..width {
            let distance_m = distance_at(x, z);
            let west = distance_at(x.saturating_sub(1), z);
            let east = distance_at((x + 1).min(width - 1), z);
            let north = distance_at(x, z.saturating_sub(1));
            let south = distance_at(x, (z + 1).min(height - 1));
            let gradient = Vec2::new(east - west, south - north);
            let landward = gradient.try_normalize().unwrap_or(Vec2::ZERO);

            let local_x = bounds[0] + x as f64 * spacing_m;
            let local_z = bounds[1] + z as f64 * spacing_m;
            let inland_offset_m = f64::from((-distance_m).max(0.0)) + 45.0;
            let inland_height_m = if landward == Vec2::ZERO {
                0.0
            } else {
                dataset.dem_height(
                    local_x + f64::from(landward.x) * inland_offset_m,
                    local_z + f64::from(landward.y) * inland_offset_m,
                )
            };
            let cliff = smoothstep(6.0, 22.0, inland_height_m);
            let exposure = if landward == Vec2::ZERO {
                if distance_m < 0.0 { 1.0 } else { 0.0 }
            } else {
                let seaward = -landward;
                let open_fetch = [(350.0, 0.25), (800.0, 0.30), (1_600.0, 0.45)]
                    .into_iter()
                    .filter(|(fetch_m, _)| {
                        !dataset.is_land(
                            local_x + f64::from(seaward.x) * *fetch_m,
                            local_z + f64::from(seaward.y) * *fetch_m,
                        )
                    })
                    .map(|(_, weight)| weight)
                    .sum::<f32>();
                wave_direction.dot(landward).clamp(0.0, 1.0) * open_fetch
            };

            data.extend_from_slice(&[
                encode_unorm(distance_m / range_m * 0.5 + 0.5),
                encode_unorm(cliff),
                encode_unorm(exposure),
                255,
            ]);
        }
    }

    let mut image = Image::new(
        Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    image
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn encode_unorm(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn encode_distance_from_land(
    land: &[bool],
    width: usize,
    height: usize,
    pixel_size_m: f32,
    range_m: f32,
) -> Vec<u8> {
    assert_eq!(land.len(), width * height);
    let mut distance = land
        .iter()
        .map(|is_land| if *is_land { 0.0 } else { f32::INFINITY })
        .collect::<Vec<_>>();
    let diagonal = std::f32::consts::SQRT_2;

    for z in 0..height {
        for x in 0..width {
            let index = z * width + x;
            if x > 0 {
                distance[index] = distance[index].min(distance[index - 1] + 1.0);
            }
            if z > 0 {
                distance[index] = distance[index].min(distance[index - width] + 1.0);
                if x > 0 {
                    distance[index] = distance[index].min(distance[index - width - 1] + diagonal);
                }
                if x + 1 < width {
                    distance[index] = distance[index].min(distance[index - width + 1] + diagonal);
                }
            }
        }
    }
    for z in (0..height).rev() {
        for x in (0..width).rev() {
            let index = z * width + x;
            if x + 1 < width {
                distance[index] = distance[index].min(distance[index + 1] + 1.0);
            }
            if z + 1 < height {
                distance[index] = distance[index].min(distance[index + width] + 1.0);
                if x + 1 < width {
                    distance[index] = distance[index].min(distance[index + width + 1] + diagonal);
                }
                if x > 0 {
                    distance[index] = distance[index].min(distance[index + width - 1] + diagonal);
                }
            }
        }
    }

    distance
        .into_iter()
        .map(|pixels| ((pixels * pixel_size_m / range_m).clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use bevy::mesh::VertexAttributeValues;

    use super::*;

    #[test]
    fn ocean_clipmap_has_a_dense_center_and_matching_rings() {
        let center = build_ocean_clip_mesh(OCEAN_BASE_CELL_SIZE_M, false);
        let ring = build_ocean_clip_mesh(OCEAN_BASE_CELL_SIZE_M * 2.0, true);
        let expected_vertices = ((OCEAN_GRID_CELLS + 1) * (OCEAN_GRID_CELLS + 1)) as usize;
        let center_triangles = OCEAN_GRID_CELLS as usize * OCEAN_GRID_CELLS as usize * 2;
        let inner_cells = OCEAN_GRID_CELLS as usize / 2;
        let ring_triangles = center_triangles - inner_cells * inner_cells * 2;

        assert_eq!(mesh_position_count(&center), expected_vertices);
        assert_eq!(mesh_position_count(&ring), expected_vertices);
        assert_eq!(mesh_triangle_count(&center), center_triangles);
        assert_eq!(mesh_triangle_count(&ring), ring_triangles);

        let center_half_extent = OCEAN_GRID_CELLS as f32 * OCEAN_BASE_CELL_SIZE_M * 0.5;
        let ring_hole_half_extent = OCEAN_GRID_CELLS as f32 * (OCEAN_BASE_CELL_SIZE_M * 2.0) * 0.25;
        assert_eq!(center_half_extent, ring_hole_half_extent);

        let outer_half_extent = OCEAN_GRID_CELLS as f32
            * OCEAN_BASE_CELL_SIZE_M
            * (1u32 << (OCEAN_CLIP_LEVELS - 1)) as f32
            * 0.5;
        assert!(outer_half_extent > 500_000.0);
    }

    #[test]
    fn coast_distance_starts_at_land_and_increases_outward() {
        let mut land = vec![false; 25];
        land[12] = true;
        let distance = encode_distance_from_land(&land, 5, 5, 100.0, 1_000.0);
        assert_eq!(distance[12], 0);
        assert!(distance[11] > distance[12]);
        assert!(distance[10] > distance[11]);
        assert_eq!(distance[2], distance[10]);
    }

    fn mesh_position_count(mesh: &Mesh) -> usize {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap() {
            VertexAttributeValues::Float32x3(positions) => positions.len(),
            _ => panic!("ocean positions must be Float32x3"),
        }
    }

    fn mesh_triangle_count(mesh: &Mesh) -> usize {
        match mesh.indices().unwrap() {
            Indices::U32(indices) => indices.len() / 3,
            _ => panic!("ocean indices must be U32"),
        }
    }
}
