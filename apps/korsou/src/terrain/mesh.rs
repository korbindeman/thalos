use std::collections::HashMap;

use bevy::{
    asset::RenderAssetUsages,
    color::LinearRgba,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    math::{DVec2, DVec3},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use super::{
    data::TerrainDataset,
    quadtree::{EdgeStitch, RTIN_TOLERANCE_M, TILE_CELLS, TileKey},
    rtin::Rtin,
    surface::canopy_coverage,
};
use crate::spatial::TerrainSpatialFrame;

const EDGE_MORPH_CELLS: usize = 8;
const SEA_LEVEL_M: f32 = 0.12;
const DETAIL_TEXTURE_SIZE: usize = 256;
const DETAIL_REPEAT_M: f32 = 96.0;
const DETAIL_NORMAL_STRENGTH: f32 = 0.015;
const MAX_SYNTHETIC_DETAIL_M: f32 = 7.0;
const FINE_TILE_CELLS: usize = TILE_CELLS * 2;
const COAST_EDGE_MAX_LENGTH_M: f64 = 3.0;
const COAST_HEIGHT_SAMPLE_INLAND_M: f64 = 4.0;
const COAST_SKIRT_DEPTH_M: f32 = 10.0;
const COAST_CLIFF_GRADE_START: f32 = 0.06;
const COAST_CLIFF_GRADE_END: f32 = 0.18;
const COAST_CLIFF_HEIGHT_START_M: f32 = 1.5;
const COAST_CLIFF_HEIGHT_END_M: f32 = 8.0;
const DEM_GRADIENT_SAMPLE_M: f64 = 30.0;
const NORMAL_FILTER_SOURCE_RADII: f64 = 2.0;
const RESOLVED_GRADE_START: f32 = 0.03;
const RESOLVED_GRADE_END: f32 = 0.12;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum VertexRef {
    Grid(u32),
    CoastEdge { a: u32, b: u32 },
    CoastSample(u32),
}

impl VertexRef {
    fn is_coast(self) -> bool {
        !matches!(self, Self::Grid(_))
    }
}

#[derive(Clone, Copy, Debug)]
struct ClipPoint {
    vertex: VertexRef,
    distance_m: f32,
}

impl ClipPoint {
    fn grid(index: u32, distance_m: f32) -> Self {
        Self {
            vertex: VertexRef::Grid(index),
            distance_m,
        }
    }

    fn coast_intersection(self, other: Self) -> Self {
        let (VertexRef::Grid(a), VertexRef::Grid(b)) = (self.vertex, other.vertex) else {
            unreachable!("coast clipping only intersects original triangle edges")
        };
        Self {
            vertex: VertexRef::CoastEdge {
                a: a.min(b),
                b: a.max(b),
            },
            distance_m: 0.0,
        }
    }
}

fn clip_triangle_to_land(triangle: [ClipPoint; 3]) -> Vec<[ClipPoint; 3]> {
    let mut polygon = Vec::with_capacity(4);
    let mut previous = triangle[2];
    for current in triangle {
        let previous_inside = previous.distance_m >= 0.0;
        let current_inside = current.distance_m >= 0.0;
        match (previous_inside, current_inside) {
            (true, true) => polygon.push(current),
            (true, false) => polygon.push(previous.coast_intersection(current)),
            (false, true) => {
                polygon.push(previous.coast_intersection(current));
                polygon.push(current);
            }
            (false, false) => {}
        }
        previous = current;
    }

    if polygon.len() < 3 {
        return Vec::new();
    }
    (1..polygon.len() - 1)
        .map(|index| [polygon[0], polygon[index], polygon[index + 1]])
        .collect()
}

fn refine_coast_triangle(
    triangle: [ClipPoint; 3],
    dataset: &TerrainDataset,
    bounds: [f64; 4],
    spacing: f64,
    side: usize,
    shore_distances: &[f32],
    coast_samples: &mut Vec<[f64; 2]>,
) -> Vec<[ClipPoint; 3]> {
    let Some(pivot_index) = triangle.iter().position(|point| !point.vertex.is_coast()) else {
        return vec![triangle];
    };
    if triangle
        .iter()
        .filter(|point| point.vertex.is_coast())
        .count()
        != 2
    {
        return vec![triangle];
    }

    let pivot = triangle[pivot_index];
    let start = triangle[(pivot_index + 1) % 3];
    let end = triangle[(pivot_index + 2) % 3];
    let start_position = coast_position(
        dataset,
        start.vertex,
        bounds,
        spacing,
        side,
        shore_distances,
        coast_samples,
    );
    let end_position = coast_position(
        dataset,
        end.vertex,
        bounds,
        spacing,
        side,
        shore_distances,
        coast_samples,
    );
    let segments = ((end_position - start_position).length() / COAST_EDGE_MAX_LENGTH_M)
        .ceil()
        .max(1.0) as usize;
    if segments == 1 {
        return vec![[pivot, start, end]];
    }

    let mut edge = Vec::with_capacity(segments + 1);
    edge.push(start);
    for step in 1..segments {
        let t = step as f64 / segments as f64;
        let position = project_to_shoreline(dataset, start_position.lerp(end_position, t));
        let sample = coast_samples.len() as u32;
        coast_samples.push(position.to_array());
        edge.push(ClipPoint {
            vertex: VertexRef::CoastSample(sample),
            distance_m: 0.0,
        });
    }
    edge.push(end);

    edge.windows(2)
        .map(|edge| [pivot, edge[0], edge[1]])
        .collect()
}

fn coast_position(
    dataset: &TerrainDataset,
    vertex: VertexRef,
    bounds: [f64; 4],
    spacing: f64,
    side: usize,
    shore_distances: &[f32],
    coast_samples: &[[f64; 2]],
) -> DVec2 {
    match vertex {
        VertexRef::Grid(_) => unreachable!("only clipped coast vertices have coast positions"),
        VertexRef::CoastEdge { a, b } => {
            let a = a as usize;
            let b = b as usize;
            let denominator = shore_distances[a] - shore_distances[b];
            let t = if denominator.abs() <= f32::EPSILON {
                0.5
            } else {
                (shore_distances[a] / denominator).clamp(0.0, 1.0)
            } as f64;
            let position = |index: usize| {
                DVec2::new(
                    bounds[0] + (index % side) as f64 * spacing,
                    bounds[1] + (index / side) as f64 * spacing,
                )
            };
            project_to_shoreline(dataset, position(a).lerp(position(b), t))
        }
        VertexRef::CoastSample(index) => DVec2::from_array(coast_samples[index as usize]),
    }
}

fn project_to_shoreline(dataset: &TerrainDataset, mut position: DVec2) -> DVec2 {
    let sample_step = dataset.metadata.coastline.distance_spacing_m * 0.5;
    for _ in 0..4 {
        let distance = f64::from(dataset.shore_distance_m(position.x, position.y));
        if distance.abs() < 0.01 {
            break;
        }
        let gradient = DVec2::new(
            f64::from(
                dataset.shore_distance_m(position.x + sample_step, position.y)
                    - dataset.shore_distance_m(position.x - sample_step, position.y),
            ),
            f64::from(
                dataset.shore_distance_m(position.x, position.y + sample_step)
                    - dataset.shore_distance_m(position.x, position.y - sample_step),
            ),
        ) / (2.0 * sample_step);
        let gradient_squared = gradient.length_squared();
        if gradient_squared < 1.0e-8 {
            break;
        }
        let correction = gradient * (distance / gradient_squared);
        position -= correction.clamp_length_max(sample_step);
    }
    position
}

fn coast_boundary_segment(triangle: [ClipPoint; 3]) -> Option<[VertexRef; 2]> {
    let pivot = triangle.iter().position(|point| !point.vertex.is_coast())?;
    let start = triangle[(pivot + 1) % 3].vertex;
    let end = triangle[(pivot + 2) % 3].vertex;
    (start.is_coast() && end.is_coast()).then_some([start, end])
}

pub struct BuiltTerrainMesh {
    pub mesh: Mesh,
    pub high_positions: Vec<[f32; 3]>,
    pub parent_positions: Vec<[f32; 3]>,
    pub source_positions_m: Vec<[f64; 2]>,
    pub origin_render_m: DVec3,
    pub triangles: usize,
}

pub fn collapse_positions(
    dataset: &TerrainDataset,
    spatial: &TerrainSpatialFrame,
    key: TileKey,
    source_positions_m: &[[f64; 2]],
    target_level: u8,
) -> Vec<[f32; 3]> {
    let bounds = dataset.quadtree_bounds(key.level, key.x, key.z);
    let origin_render_m = spatial.tile_origin(bounds[0], bounds[1]);
    source_positions_m
        .iter()
        .map(|position| {
            let world_x = position[0];
            let world_z = position[1];
            let height = if dataset.shore_distance_m(world_x, world_z).abs() <= 0.11 {
                SEA_LEVEL_M
            } else {
                grid_surface_height(dataset, world_x, world_z, target_level).max(SEA_LEVEL_M)
            };
            (spatial.project(DVec3::new(world_x, f64::from(height), world_z)) - origin_render_m)
                .as_vec3()
                .to_array()
        })
        .collect()
}

pub fn build_tile_mesh(
    dataset: &TerrainDataset,
    spatial: &TerrainSpatialFrame,
    key: TileKey,
    stitch: EdgeStitch,
) -> BuiltTerrainMesh {
    let bounds = dataset.quadtree_bounds(key.level, key.x, key.z);
    let origin_render_m = spatial.tile_origin(bounds[0], bounds[1]);
    let tile_size = bounds[2] - bounds[0];
    let cells = tile_cells_for_level(dataset, key.level);
    let spacing = tile_size / cells as f64;
    let side = cells + 1;
    let mut heights = vec![0.0; side * side];
    let mut shore_distances = vec![0.0; side * side];

    for z in 0..side {
        for x in 0..side {
            let world_x = bounds[0] + x as f64 * spacing;
            let world_z = bounds[1] + z as f64 * spacing;
            let index = z * side + x;
            heights[index] = rendered_height(dataset, world_x, world_z, key.level).max(SEA_LEVEL_M);
            shore_distances[index] = dataset.shore_distance_m(world_x, world_z);
        }
    }
    stitch_edges(dataset, key, bounds, spacing, cells, stitch, &mut heights);

    let mut constrained = vec![false; heights.len()];
    let coast_refine_band_m = dataset.metadata.coastline.distance_spacing_m as f32 * 4.0;
    for z in 0..side {
        for x in 0..side {
            let index = z * side + x;
            if x == 0 || z == 0 || x == cells || z == cells {
                constrained[index] = true;
            }
            constrained[index] |= shore_distances[index].abs() <= coast_refine_band_m;
            let value = shore_distances[index] >= 0.0;
            for (dx, dz) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let nx = x as isize + dx;
                let nz = z as isize + dz;
                let neighbour = if nx >= 0 && nz >= 0 && nx < side as isize && nz < side as isize {
                    shore_distances[nz as usize * side + nx as usize] >= 0.0
                } else {
                    let world_x = bounds[0] + nx as f64 * spacing;
                    let world_z = bounds[1] + nz as f64 * spacing;
                    dataset.shore_distance_m(world_x, world_z) >= 0.0
                };
                constrained[index] |= neighbour != value;
            }
        }
    }

    let rtin = Rtin::new(side, &heights, &constrained);
    let tolerance = RTIN_TOLERANCE_M[key.level.min(6) as usize];
    let source_triangles = rtin.triangles(tolerance);
    let mut clipped_triangles = Vec::with_capacity(source_triangles.len());
    let mut coast_samples = Vec::new();
    let mut coast_segments = Vec::new();
    for triangle in source_triangles {
        let points = triangle.map(|index| ClipPoint::grid(index, shore_distances[index as usize]));
        for clipped in clip_triangle_to_land(points) {
            for refined in refine_coast_triangle(
                clipped,
                dataset,
                bounds,
                spacing,
                side,
                &shore_distances,
                &mut coast_samples,
            ) {
                if let Some(segment) = coast_boundary_segment(refined) {
                    coast_segments.push(segment);
                }
                clipped_triangles.push(refined);
            }
        }
    }

    // The 7.5 m visual grid oversamples the 30 m GLO-30 source. Differentiate
    // over the source footprint so its bilinear cell boundaries do not become
    // lighting bands; the mesh positions and synthetic silhouette stay exact.
    let normals = grid_normals(
        &heights,
        side,
        spacing,
        dataset.metadata.levels[0].sample_spacing_m * NORMAL_FILTER_SOURCE_RADII,
    );
    let mut remap = HashMap::new();
    let mut high_positions = Vec::new();
    let mut parent_positions = Vec::new();
    let mut source_positions_m = Vec::new();
    let mut compact_normals = Vec::new();
    let mut compact_tangents = Vec::new();
    let mut colors = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::with_capacity(clipped_triangles.len() * 3);

    for triangle in clipped_triangles {
        for point in triangle {
            let compact = emit_vertex(
                point.vertex,
                &mut remap,
                dataset,
                spatial,
                key,
                bounds,
                origin_render_m,
                spacing,
                side,
                &heights,
                &shore_distances,
                &coast_samples,
                &normals,
                &mut high_positions,
                &mut parent_positions,
                &mut source_positions_m,
                &mut compact_normals,
                &mut compact_tangents,
                &mut colors,
                &mut uvs,
            );
            indices.push(compact);
        }
    }

    emit_coast_skirts(
        &coast_segments,
        dataset,
        spatial,
        key,
        bounds,
        origin_render_m,
        spacing,
        side,
        &shore_distances,
        &coast_samples,
        &mut high_positions,
        &mut parent_positions,
        &mut source_positions_m,
        &mut compact_normals,
        &mut compact_tangents,
        &mut colors,
        &mut uvs,
        &mut indices,
    );

    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, high_positions.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, compact_normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, compact_tangents)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices));

    BuiltTerrainMesh {
        mesh,
        high_positions,
        parent_positions,
        source_positions_m,
        origin_render_m,
        triangles: 0,
    }
    .with_triangle_count()
}

pub(super) fn tile_cells_for_level(dataset: &TerrainDataset, level: u8) -> usize {
    if level == dataset.metadata.quadtree.visual_max_level {
        FINE_TILE_CELLS
    } else {
        TILE_CELLS
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_vertex(
    vertex: VertexRef,
    remap: &mut HashMap<VertexRef, u32>,
    dataset: &TerrainDataset,
    spatial: &TerrainSpatialFrame,
    key: TileKey,
    bounds: [f64; 4],
    origin_render_m: DVec3,
    spacing: f64,
    side: usize,
    heights: &[f32],
    shore_distances: &[f32],
    coast_samples: &[[f64; 2]],
    normals: &[Vec3],
    high_positions: &mut Vec<[f32; 3]>,
    parent_positions: &mut Vec<[f32; 3]>,
    source_positions_m: &mut Vec<[f64; 2]>,
    compact_normals: &mut Vec<[f32; 3]>,
    compact_tangents: &mut Vec<[f32; 4]>,
    colors: &mut Vec<[f32; 4]>,
    uvs: &mut Vec<[f32; 2]>,
) -> u32 {
    if let Some(index) = remap.get(&vertex) {
        return *index;
    }

    let (source_high, source_parent, normal, color_height, shore_distance) = match vertex {
        VertexRef::Grid(source_index) => {
            let source_index = source_index as usize;
            let x = source_index % side;
            let z = source_index / side;
            let high = [
                x as f32 * spacing as f32,
                heights[source_index],
                z as f32 * spacing as f32,
            ];
            let world_x = bounds[0] + x as f64 * spacing;
            let world_z = bounds[1] + z as f64 * spacing;
            let parent_height = if key.level == 0 {
                high[1]
            } else {
                grid_surface_height(dataset, world_x, world_z, key.level - 1).max(SEA_LEVEL_M)
            };
            (
                high,
                [high[0], parent_height, high[2]],
                normals[source_index],
                dataset.dem_height(world_x, world_z),
                shore_distances[source_index],
            )
        }
        VertexRef::CoastEdge { .. } | VertexRef::CoastSample(_) => {
            let position = coast_position(
                dataset,
                vertex,
                bounds,
                spacing,
                side,
                shore_distances,
                coast_samples,
            );
            let high = [
                (position.x - bounds[0]) as f32,
                coast_top_height(dataset, position),
                (position.y - bounds[1]) as f32,
            ];
            (
                high,
                high,
                rendered_surface_normal(dataset, position, key.level, spacing),
                high[1],
                0.0,
            )
        }
    };

    let world_x = bounds[0] + f64::from(source_high[0]);
    let world_z = bounds[1] + f64::from(source_high[2]);
    let source_grade = dem_grade(dataset, world_x, world_z);
    let shading_normal = terrain_shading_normal(normal, source_grade);
    let high_world = DVec3::new(world_x, f64::from(source_high[1]), world_z);
    let parent_world = DVec3::new(world_x, f64::from(source_parent[1]), world_z);
    let high = (spatial.project(high_world) - origin_render_m)
        .as_vec3()
        .to_array();
    let parent = (spatial.project(parent_world) - origin_render_m)
        .as_vec3()
        .to_array();
    let render_normal = spatial
        .project_direction(high_world, shading_normal.as_dvec3())
        .as_vec3()
        .normalize_or(Vec3::Y);
    let render_east = spatial.project_direction(high_world, DVec3::X).as_vec3();
    let render_north = spatial.project_direction(high_world, DVec3::Z).as_vec3();
    let tangent =
        (render_east - render_normal * render_east.dot(render_normal)).normalize_or(Vec3::X);
    let handedness = if render_normal.cross(tangent).dot(render_north) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let index = high_positions.len() as u32;
    high_positions.push(high);
    parent_positions.push(parent);
    source_positions_m.push([world_x, world_z]);
    compact_normals.push(render_normal.to_array());
    compact_tangents.push([tangent.x, tangent.y, tangent.z, handedness]);
    colors.push(terrain_color(
        world_x,
        world_z,
        color_height,
        shore_distance,
        source_grade,
        key.level,
    ));
    uvs.push([
        world_x as f32 / DETAIL_REPEAT_M,
        world_z as f32 / DETAIL_REPEAT_M,
    ]);
    remap.insert(vertex, index);
    index
}

fn coast_top_height(dataset: &TerrainDataset, position: DVec2) -> f32 {
    let sample = coast_land_sample(dataset, position);
    let height = dataset.dem_height(sample.x, sample.y).max(SEA_LEVEL_M);
    let grade = dem_grade(dataset, sample.x, sample.y);
    let cliff_weight = smoothstep(COAST_CLIFF_GRADE_START, COAST_CLIFF_GRADE_END, grade)
        * smoothstep(COAST_CLIFF_HEIGHT_START_M, COAST_CLIFF_HEIGHT_END_M, height);
    SEA_LEVEL_M + (height - SEA_LEVEL_M) * cliff_weight
}

fn coast_land_sample(dataset: &TerrainDataset, position: DVec2) -> DVec2 {
    let landward = shore_gradient(dataset, position).normalize_or(DVec2::X);
    position + landward * COAST_HEIGHT_SAMPLE_INLAND_M
}

fn rendered_surface_normal(
    dataset: &TerrainDataset,
    position: DVec2,
    level: u8,
    grid_spacing_m: f64,
) -> Vec3 {
    let center = coast_land_sample(dataset, position);
    let step = grid_spacing_m * 0.5;
    let west = rendered_height(dataset, center.x - step, center.y, level).max(SEA_LEVEL_M);
    let east = rendered_height(dataset, center.x + step, center.y, level).max(SEA_LEVEL_M);
    let north = rendered_height(dataset, center.x, center.y - step, level).max(SEA_LEVEL_M);
    let south = rendered_height(dataset, center.x, center.y + step, level).max(SEA_LEVEL_M);
    Vec3::new(
        (west - east) / (2.0 * step) as f32,
        1.0,
        (north - south) / (2.0 * step) as f32,
    )
    .normalize()
}

fn shore_gradient(dataset: &TerrainDataset, position: DVec2) -> DVec2 {
    let step = dataset.metadata.coastline.distance_spacing_m * 0.5;
    DVec2::new(
        f64::from(
            dataset.shore_distance_m(position.x + step, position.y)
                - dataset.shore_distance_m(position.x - step, position.y),
        ),
        f64::from(
            dataset.shore_distance_m(position.x, position.y + step)
                - dataset.shore_distance_m(position.x, position.y - step),
        ),
    ) / (2.0 * step)
}

#[allow(clippy::too_many_arguments)]
fn emit_coast_skirts(
    coast_segments: &[[VertexRef; 2]],
    dataset: &TerrainDataset,
    spatial: &TerrainSpatialFrame,
    key: TileKey,
    bounds: [f64; 4],
    origin_render_m: DVec3,
    spacing: f64,
    side: usize,
    shore_distances: &[f32],
    coast_samples: &[[f64; 2]],
    high_positions: &mut Vec<[f32; 3]>,
    parent_positions: &mut Vec<[f32; 3]>,
    source_positions_m: &mut Vec<[f64; 2]>,
    normals: &mut Vec<[f32; 3]>,
    tangents: &mut Vec<[f32; 4]>,
    colors: &mut Vec<[f32; 4]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
) {
    let mut vertices = HashMap::new();
    for &[start, end] in coast_segments {
        let top_start = emit_coast_skirt_vertex(
            start,
            true,
            &mut vertices,
            dataset,
            spatial,
            key,
            bounds,
            origin_render_m,
            spacing,
            side,
            shore_distances,
            coast_samples,
            high_positions,
            parent_positions,
            source_positions_m,
            normals,
            tangents,
            colors,
            uvs,
        );
        let bottom_start = emit_coast_skirt_vertex(
            start,
            false,
            &mut vertices,
            dataset,
            spatial,
            key,
            bounds,
            origin_render_m,
            spacing,
            side,
            shore_distances,
            coast_samples,
            high_positions,
            parent_positions,
            source_positions_m,
            normals,
            tangents,
            colors,
            uvs,
        );
        let top_end = emit_coast_skirt_vertex(
            end,
            true,
            &mut vertices,
            dataset,
            spatial,
            key,
            bounds,
            origin_render_m,
            spacing,
            side,
            shore_distances,
            coast_samples,
            high_positions,
            parent_positions,
            source_positions_m,
            normals,
            tangents,
            colors,
            uvs,
        );
        let bottom_end = emit_coast_skirt_vertex(
            end,
            false,
            &mut vertices,
            dataset,
            spatial,
            key,
            bounds,
            origin_render_m,
            spacing,
            side,
            shore_distances,
            coast_samples,
            high_positions,
            parent_positions,
            source_positions_m,
            normals,
            tangents,
            colors,
            uvs,
        );
        indices.extend_from_slice(&[
            top_start,
            bottom_start,
            bottom_end,
            top_start,
            bottom_end,
            top_end,
        ]);
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_coast_skirt_vertex(
    vertex: VertexRef,
    top: bool,
    remap: &mut HashMap<(VertexRef, bool), u32>,
    dataset: &TerrainDataset,
    spatial: &TerrainSpatialFrame,
    key: TileKey,
    bounds: [f64; 4],
    origin_render_m: DVec3,
    spacing: f64,
    side: usize,
    shore_distances: &[f32],
    coast_samples: &[[f64; 2]],
    high_positions: &mut Vec<[f32; 3]>,
    parent_positions: &mut Vec<[f32; 3]>,
    source_positions_m: &mut Vec<[f64; 2]>,
    normals: &mut Vec<[f32; 3]>,
    tangents: &mut Vec<[f32; 4]>,
    colors: &mut Vec<[f32; 4]>,
    uvs: &mut Vec<[f32; 2]>,
) -> u32 {
    if let Some(index) = remap.get(&(vertex, top)) {
        return *index;
    }

    let position = coast_position(
        dataset,
        vertex,
        bounds,
        spacing,
        side,
        shore_distances,
        coast_samples,
    );
    let top_height = coast_top_height(dataset, position);
    let height = if top {
        top_height
    } else {
        SEA_LEVEL_M - COAST_SKIRT_DEPTH_M
    };
    let world = DVec3::new(position.x, f64::from(height), position.y);
    let render_position = (spatial.project(world) - origin_render_m).as_vec3();
    let landward = shore_gradient(dataset, position).normalize_or(DVec2::X);
    let outward = DVec3::new(-landward.x, 0.0, -landward.y);
    let render_normal = spatial
        .project_direction(world, outward)
        .as_vec3()
        .normalize_or(Vec3::X);
    let render_up = spatial.project_direction(world, DVec3::Y).as_vec3();
    let tangent = (render_up - render_normal * render_up.dot(render_normal)).normalize_or(Vec3::Y);
    let alongshore = spatial
        .project_direction(world, DVec3::new(-outward.z, 0.0, outward.x))
        .as_vec3();
    let handedness = if render_normal.cross(tangent).dot(alongshore) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let land_sample = coast_land_sample(dataset, position);
    let color = terrain_color(
        position.x,
        position.y,
        top_height,
        0.0,
        dem_grade(dataset, land_sample.x, land_sample.y),
        key.level,
    );

    let index = high_positions.len() as u32;
    high_positions.push(render_position.to_array());
    parent_positions.push(render_position.to_array());
    source_positions_m.push(position.to_array());
    normals.push(render_normal.to_array());
    tangents.push([tangent.x, tangent.y, tangent.z, handedness]);
    colors.push(color);
    uvs.push([
        height / DETAIL_REPEAT_M,
        ((position.x + position.y) * 0.5) as f32 / DETAIL_REPEAT_M,
    ]);
    remap.insert((vertex, top), index);
    index
}

impl BuiltTerrainMesh {
    fn with_triangle_count(mut self) -> Self {
        self.triangles = self.mesh.indices().map_or(0, |indices| indices.len() / 3);
        self
    }
}

fn stitch_edges(
    dataset: &TerrainDataset,
    key: TileKey,
    bounds: [f64; 4],
    spacing: f64,
    cells: usize,
    stitch: EdgeStitch,
    heights: &mut [f32],
) {
    if key.level == 0 {
        return;
    }
    let side = cells + 1;
    for z in 0..side {
        for x in 0..side {
            let mut weight = 1.0f32;
            if stitch.has(EdgeStitch::NORTH) {
                weight = weight.min(edge_morph_weight(z));
            }
            if stitch.has(EdgeStitch::SOUTH) {
                weight = weight.min(edge_morph_weight(cells - z));
            }
            if stitch.has(EdgeStitch::WEST) {
                weight = weight.min(edge_morph_weight(x));
            }
            if stitch.has(EdgeStitch::EAST) {
                weight = weight.min(edge_morph_weight(cells - x));
            }
            if weight < 1.0 {
                let world_x = bounds[0] + x as f64 * spacing;
                let world_z = bounds[1] + z as f64 * spacing;
                let coarse =
                    grid_surface_height(dataset, world_x, world_z, key.level - 1).max(SEA_LEVEL_M);
                let index = z * side + x;
                heights[index] = coarse + (heights[index] - coarse) * weight;
            }
        }
    }
}

fn edge_morph_weight(distance_cells: usize) -> f32 {
    let t = (distance_cells as f32 / EDGE_MORPH_CELLS as f32).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn grid_surface_height(dataset: &TerrainDataset, x: f64, z: f64, level: u8) -> f32 {
    let root = dataset.metadata.quadtree.domain_bounds_local_m;
    let spacing = (root[2] - root[0]) / ((1u32 << level) as f64 * TILE_CELLS as f64);
    let gx = ((x - root[0]) / spacing).clamp(0.0, (1u32 << level) as f64 * TILE_CELLS as f64);
    let gz = ((z - root[1]) / spacing).clamp(0.0, (1u32 << level) as f64 * TILE_CELLS as f64);
    let x0 = gx.floor();
    let z0 = gz.floor();
    let tx = (gx - x0) as f32;
    let tz = (gz - z0) as f32;
    let x0 = root[0] + x0 * spacing;
    let z0 = root[1] + z0 * spacing;
    let h00 = rendered_height(dataset, x0, z0, level);
    let h10 = rendered_height(dataset, x0 + spacing, z0, level);
    let h01 = rendered_height(dataset, x0, z0 + spacing, level);
    let h11 = rendered_height(dataset, x0 + spacing, z0 + spacing, level);
    if tx + tz <= 1.0 {
        h00 + tx * (h10 - h00) + tz * (h01 - h00)
    } else {
        h11 + (1.0 - tz) * (h10 - h11) + (1.0 - tx) * (h01 - h11)
    }
}

fn grid_normals(heights: &[f32], side: usize, spacing: f64, filter_radius_m: f64) -> Vec<Vec3> {
    let radius = (filter_radius_m / spacing).ceil() as usize;
    let radius = radius.clamp(1, side - 1);
    let mut normals = Vec::with_capacity(heights.len());
    for z in 0..side {
        for x in 0..side {
            let west_x = x.saturating_sub(radius);
            let east_x = (x + radius).min(side - 1);
            let north_z = z.saturating_sub(radius);
            let south_z = (z + radius).min(side - 1);
            let west = heights[z * side + west_x];
            let east = heights[z * side + east_x];
            let north = heights[north_z * side + x];
            let south = heights[south_z * side + x];
            let dx = (east_x - west_x) as f64 * spacing;
            let dz = (south_z - north_z) as f64 * spacing;
            normals.push(
                Vec3::new((west - east) / dx as f32, 1.0, (north - south) / dz as f32).normalize(),
            );
        }
    }
    normals
}

pub(crate) fn rendered_height(dataset: &TerrainDataset, x: f64, z: f64, level: u8) -> f32 {
    let base = dataset.dem_height(x, z);
    if !dataset.is_land(x, z) {
        return base;
    }
    base + synthetic_detail(
        x,
        z,
        base,
        dataset.shore_distance_m(x, z),
        dem_grade(dataset, x, z),
        level,
    )
}

fn dem_grade(dataset: &TerrainDataset, x: f64, z: f64) -> f32 {
    let sample = DEM_GRADIENT_SAMPLE_M;
    let dx = (dataset.dem_height(x + sample, z) - dataset.dem_height(x - sample, z))
        / (2.0 * sample as f32);
    let dz = (dataset.dem_height(x, z + sample) - dataset.dem_height(x, z - sample))
        / (2.0 * sample as f32);
    dx.hypot(dz)
}

fn terrain_shading_normal(mesh_normal: Vec3, source_grade: f32) -> Vec3 {
    Vec3::Y
        .lerp(
            mesh_normal,
            smoothstep(RESOLVED_GRADE_START, RESOLVED_GRADE_END, source_grade),
        )
        .normalize()
}

pub(crate) fn build_terrain_detail_normal() -> Image {
    let mut normal = Vec::with_capacity(DETAIL_TEXTURE_SIZE * DETAIL_TEXTURE_SIZE * 4);
    let texel_m = DETAIL_REPEAT_M / DETAIL_TEXTURE_SIZE as f32;

    for z in 0..DETAIL_TEXTURE_SIZE as i32 {
        for x in 0..DETAIL_TEXTURE_SIZE as i32 {
            let dx =
                (detail_height_texel(x + 1, z) - detail_height_texel(x - 1, z)) / (2.0 * texel_m);
            let dz =
                (detail_height_texel(x, z + 1) - detail_height_texel(x, z - 1)) / (2.0 * texel_m);
            let n = Vec3::new(
                -dx * DETAIL_NORMAL_STRENGTH,
                -dz * DETAIL_NORMAL_STRENGTH,
                1.0,
            )
            .normalize();
            normal.extend_from_slice(&[
                encode_unorm(n.x * 0.5 + 0.5),
                encode_unorm(n.y * 0.5 + 0.5),
                encode_unorm(n.z * 0.5 + 0.5),
                255,
            ]);
        }
    }

    rgba8_image_with_mips(normal, TextureFormat::Rgba8Unorm, true)
}

fn rgba8_image_with_mips(mut level: Vec<u8>, format: TextureFormat, normal_map: bool) -> Image {
    let mut data = level.clone();
    let mut size = DETAIL_TEXTURE_SIZE;
    let mut mip_count = 1;
    while size > 1 {
        let next_size = size / 2;
        let mut next = Vec::with_capacity(next_size * next_size * 4);
        for z in 0..next_size {
            for x in 0..next_size {
                let mut sum = Vec4::ZERO;
                for dz in 0..2 {
                    for dx in 0..2 {
                        let index = ((z * 2 + dz) * size + (x * 2 + dx)) * 4;
                        let sample = Vec4::new(
                            f32::from(level[index]) / 255.0,
                            f32::from(level[index + 1]) / 255.0,
                            f32::from(level[index + 2]) / 255.0,
                            f32::from(level[index + 3]) / 255.0,
                        );
                        sum += sample;
                    }
                }
                let mut average = sum * 0.25;
                if normal_map {
                    let n = (average.truncate() * 2.0 - Vec3::ONE).normalize_or(Vec3::Z);
                    average = (n * 0.5 + Vec3::splat(0.5)).extend(1.0);
                }
                next.extend_from_slice(&[
                    encode_unorm(average.x),
                    encode_unorm(average.y),
                    encode_unorm(average.z),
                    encode_unorm(average.w),
                ]);
            }
        }
        data.extend_from_slice(&next);
        level = next;
        size = next_size;
        mip_count += 1;
    }

    let mut image = Image::new_uninit(
        Extent3d {
            width: DETAIL_TEXTURE_SIZE as u32,
            height: DETAIL_TEXTURE_SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = mip_count;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        anisotropy_clamp: 16,
        ..ImageSamplerDescriptor::linear()
    });
    image
}

fn detail_height_texel(x: i32, z: i32) -> f32 {
    let x = x.rem_euclid(DETAIL_TEXTURE_SIZE as i32) as f32;
    let z = z.rem_euclid(DETAIL_TEXTURE_SIZE as i32) as f32;
    detail_noise_texel(x, z, 64, 0x29) * 0.20
        + detail_noise_texel(x, z, 32, 0x47) * 0.28
        + detail_noise_texel(x, z, 8, 0x71) * 0.52
}

fn detail_noise_texel(x: f32, z: f32, wavelength_px: i32, seed: u32) -> f32 {
    let wavelength = wavelength_px as f32;
    let cells = DETAIL_TEXTURE_SIZE as i32 / wavelength_px;
    let grid_x = x / wavelength;
    let grid_z = z / wavelength;
    let x0 = grid_x.floor() as i32;
    let z0 = grid_z.floor() as i32;
    let tx = smooth_fraction(grid_x.fract());
    let tz = smooth_fraction(grid_z.fract());
    let sample = |ix: i32, iz: i32| hash_unit(ix.rem_euclid(cells), iz.rem_euclid(cells), seed);
    let lower = sample(x0, z0) + (sample(x0 + 1, z0) - sample(x0, z0)) * tx;
    let upper = sample(x0, z0 + 1) + (sample(x0 + 1, z0 + 1) - sample(x0, z0 + 1)) * tx;
    lower + (upper - lower) * tz
}

fn encode_unorm(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn synthetic_detail(
    x: f64,
    z: f64,
    base_height: f32,
    shore_distance: f32,
    grade: f32,
    level: u8,
) -> f32 {
    let level_weight = match level {
        6.. => 1.0,
        5 => 0.40,
        4 => 0.08,
        _ => 0.0,
    };
    if level_weight == 0.0 {
        return 0.0;
    }

    let beach = (1.0 - smoothstep(0.05, 0.20, grade))
        * (1.0 - smoothstep(6.0, 16.0, base_height))
        * (1.0 - smoothstep(70.0, 150.0, shore_distance));
    let coast_lip_weight = smoothstep(8.0, 36.0, shore_distance);
    let land_weight = smoothstep(1.5, 12.0, base_height) * (1.0 - beach);
    let cliff_weight = smoothstep(0.05, 0.26, grade) * (1.0 - beach);

    let warp_x = value_noise(x, z, 720.0, 11) as f64 * 95.0;
    let warp_z = value_noise(x, z, 680.0, 29) as f64 * 95.0;
    let landform = value_noise(x + warp_x, z + warp_z, 420.0, 37);
    let broad = value_noise(x + warp_x, z + warp_z, 144.0, 47);
    let fine = value_noise(x - warp_z, z + warp_x, 48.0, 71);
    let drainage = (1.0 - broad.abs()).powi(7);
    let erosion_shape =
        (0.34 * landform + 0.58 * broad + 0.24 * fine - 0.82 * (drainage - 0.16)) * 2.8;

    let rock_warp_x = value_noise(x, z, 210.0, 181) as f64 * 24.0;
    let rock_warp_z = value_noise(x, z, 190.0, 193) as f64 * 24.0;
    let blocks = value_noise(x + rock_warp_x, z + rock_warp_z, 112.0, 211) * 1.25;
    let broad_fractures = ridged_noise(x - rock_warp_z, z + rock_warp_x, 54.0, 227) * 1.8;
    let fine_fractures = ridged_noise(x + rock_warp_z, z - rock_warp_x, 27.0, 233) * 1.3;
    let chips = value_noise(x - rock_warp_x, z + rock_warp_z, 18.0, 239) * 0.55;
    let rock_shape = blocks + broad_fractures + fine_fractures + chips;

    (erosion_shape * land_weight * (1.0 - cliff_weight * 0.58) + rock_shape * cliff_weight)
        .clamp(-MAX_SYNTHETIC_DETAIL_M, MAX_SYNTHETIC_DETAIL_M)
        * level_weight
        * coast_lip_weight
}

fn ridged_noise(x: f64, z: f64, wavelength: f64, seed: u32) -> f32 {
    let ridge = 1.0 - value_noise(x, z, wavelength, seed).abs();
    ridge.powi(4) * 2.0 - 0.55
}

const CANOPY_PROXY_STRENGTH: [f32; 7] = [0.18, 0.17, 0.15, 0.12, 0.08, 0.04, 0.02];

fn terrain_color(
    x: f64,
    z: f64,
    height: f32,
    shore_distance: f32,
    grade: f32,
    terrain_level: u8,
) -> [f32; 4] {
    let normal_y = 1.0 / (1.0 + grade * grade).sqrt();
    let slope = (1.0 - normal_y).clamp(0.0, 0.55) / 0.55;
    let flatness = 1.0 - smoothstep(0.025, 0.12, 1.0 - normal_y);
    let elevation = smoothstep(12.0, 260.0, height);
    let dry_variation = value_noise(x, z, 520.0, 101) * 0.5 + 0.5;
    let geology = value_noise(x + 171.0, z - 83.0, 1_450.0, 119) * 0.5 + 0.5;
    let local_tone = value_noise(x, z, 118.0, 149) * 0.5 + 0.5;
    let drainage = (1.0 - value_noise(x, z, 210.0, 163).abs()).powi(5);
    let soil = Vec3::new(0.30, 0.255, 0.125);
    let dry_soil = Vec3::new(0.46, 0.39, 0.225);
    let limestone = Vec3::new(0.56, 0.525, 0.43);
    let dark_rock = Vec3::new(0.315, 0.30, 0.255);
    let sand = Vec3::new(0.72, 0.635, 0.455);
    let canopy = Vec3::new(0.125, 0.205, 0.075);

    let mut color = soil.lerp(
        dry_soil,
        (dry_variation * 0.58 + local_tone * 0.18 + elevation * 0.16).clamp(0.0, 1.0),
    );
    let rock = dark_rock.lerp(limestone, geology);
    let rock_exposure = smoothstep(0.10, 0.58, slope + elevation * 0.16);
    color = color.lerp(rock, rock_exposure * (0.48 + geology * 0.32));
    color *= 1.0 - drainage * (1.0 - flatness) * 0.12;
    color = color.lerp(
        canopy,
        canopy_coverage(x, z, height, shore_distance, grade) * canopy_proxy_strength(terrain_level),
    );

    color = color.lerp(
        sand,
        sand_coverage(height, shore_distance, flatness, dry_variation),
    );
    LinearRgba::from(Color::srgb(color.x, color.y, color.z)).to_f32_array()
}

fn canopy_proxy_strength(terrain_level: u8) -> f32 {
    CANOPY_PROXY_STRENGTH[usize::from(terrain_level).min(CANOPY_PROXY_STRENGTH.len() - 1)]
}

fn sand_coverage(height: f32, shore_distance: f32, flatness: f32, dry_variation: f32) -> f32 {
    let beach = (1.0 - smoothstep(30.0, 110.0, shore_distance)) * flatness;
    let low_plain = (1.0 - smoothstep(3.0, 15.0, height))
        * flatness
        * smoothstep(0.76, 0.94, dry_variation)
        * (1.0 - beach);
    (beach * 0.76 + low_plain * 0.28).min(0.76)
}

fn value_noise(x: f64, z: f64, wavelength: f64, seed: u32) -> f32 {
    let x = x / wavelength;
    let z = z / wavelength;
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let tx = smooth_fraction((x - x.floor()) as f32);
    let tz = smooth_fraction((z - z.floor()) as f32);
    let a = hash_unit(x0, z0, seed);
    let b = hash_unit(x0 + 1, z0, seed);
    let c = hash_unit(x0, z0 + 1, seed);
    let d = hash_unit(x0 + 1, z0 + 1, seed);
    let lower_z = a + (b - a) * tx;
    let upper_z = c + (d - c) * tx;
    lower_z + (upper_z - lower_z) * tz
}

fn hash_unit(x: i32, z: i32, seed: u32) -> f32 {
    let mut value = (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (z as u32).wrapping_mul(0x85EB_CA77)
        ^ seed.wrapping_mul(0xC2B2_AE3D);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^= value >> 16;
    value as f32 / u32::MAX as f32 * 2.0 - 1.0
}

fn smooth_fraction(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use bevy::mesh::VertexAttributeValues;

    use super::*;

    #[test]
    fn synthetic_detail_is_deterministic_and_coast_safe() {
        assert_eq!(
            synthetic_detail(1234.0, -5678.0, 42.0, 200.0, 0.35, 6),
            synthetic_detail(1234.0, -5678.0, 42.0, 200.0, 0.35, 6)
        );
        assert_eq!(synthetic_detail(1234.0, -5678.0, 1.0, 25.0, 0.01, 6), 0.0);
        assert_eq!(synthetic_detail(1234.0, -5678.0, 42.0, 0.0, 0.35, 6), 0.0);
        assert_ne!(synthetic_detail(1234.0, -5678.0, 42.0, 80.0, 0.35, 6), 0.0);
        assert_ne!(synthetic_detail(1234.0, -5678.0, 42.0, 200.0, 0.35, 4), 0.0);
    }

    #[test]
    fn synthetic_detail_is_bounded_across_visible_lods() {
        for level in 4..=6 {
            for z in -12..=12 {
                for x in -12..=12 {
                    let detail = synthetic_detail(
                        f64::from(x) * 37.0,
                        f64::from(z) * 41.0,
                        60.0,
                        200.0,
                        0.55,
                        level,
                    );
                    assert!(detail.abs() <= MAX_SYNTHETIC_DETAIL_M);
                }
            }
        }
    }

    #[test]
    fn caracasbaai_cliff_mesh_has_finer_geometry_than_the_shoreline_grid() {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap();
        let level = dataset.metadata.quadtree.visual_max_level;
        let root = dataset.metadata.quadtree.domain_bounds_local_m;
        let tile_size = (root[2] - root[0]) / f64::from(1u32 << level);
        let cliff = [8_345.1, 14_734.4];
        let key = TileKey {
            level,
            x: ((cliff[0] - root[0]) / tile_size).floor() as u32,
            z: ((cliff[1] - root[1]) / tile_size).floor() as u32,
        };
        let spatial = TerrainSpatialFrame::new(&dataset, crate::cli::SpatialMode::Planar).unwrap();
        let built = build_tile_mesh(&dataset, &spatial, key, EdgeStitch::default());
        let shoreline_spacing = dataset.metadata.coastline.distance_spacing_m;
        let has_interior_half_step = built.source_positions_m.iter().any(|position| {
            if dataset.shore_distance_m(position[0], position[1]) < shoreline_spacing as f32 * 2.0 {
                return false;
            }
            position.iter().any(|coordinate| {
                let grid = (coordinate - root[0]) / shoreline_spacing;
                (grid - grid.round()).abs() > 0.2
            })
        });

        assert!(
            has_interior_half_step,
            "the closest Caracasbaai cliff mesh must resolve geometry below the 15 m shoreline grid"
        );
    }

    #[test]
    fn caracasbaai_cliff_coast_uses_a_vertical_face_instead_of_a_triangular_ramp() {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap();
        let level = dataset.metadata.quadtree.visual_max_level;
        let root = dataset.metadata.quadtree.domain_bounds_local_m;
        let tile_size = (root[2] - root[0]) / f64::from(1u32 << level);
        let cliff = [8_345.1, 14_734.4];
        let key = TileKey {
            level,
            x: ((cliff[0] - root[0]) / tile_size).floor() as u32,
            z: ((cliff[1] - root[1]) / tile_size).floor() as u32,
        };
        let spatial = TerrainSpatialFrame::new(&dataset, crate::cli::SpatialMode::Planar).unwrap();
        let built = build_tile_mesh(&dataset, &spatial, key, EdgeStitch::default());
        let indices: Vec<usize> = built.mesh.indices().unwrap().iter().collect();

        let has_vertical_coast_face = indices.chunks_exact(3).any(|triangle| {
            let all_on_coast = triangle.iter().all(|index| {
                let position = built.source_positions_m[*index];
                dataset.shore_distance_m(position[0], position[1]).abs() < 1.0
            });
            let a = built.source_positions_m[triangle[0]];
            let b = built.source_positions_m[triangle[1]];
            let c = built.source_positions_m[triangle[2]];
            let projected_twice_area =
                (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            let min_height = triangle
                .iter()
                .map(|index| built.high_positions[*index][1])
                .fold(f32::INFINITY, f32::min);
            let max_height = triangle
                .iter()
                .map(|index| built.high_positions[*index][1])
                .fold(f32::NEG_INFINITY, f32::max);
            all_on_coast && projected_twice_area.abs() < 0.01 && max_height - min_height > 3.0
        });

        assert!(
            has_vertical_coast_face,
            "the Caracasbaai cliff must end in a coast face; dropping the land surface to sea level creates the visible triangular fan"
        );
    }

    #[test]
    fn caracasbaai_height_profile_separates_smooth_beach_from_broken_cliff() {
        fn mean_profile_curvature(dataset: &TerrainDataset, center: [f64; 2], level: u8) -> f32 {
            let step = 7.5;
            let mut curvature = 0.0;
            let mut samples = 0;
            for offset in -4..=4 {
                let offset = f64::from(offset) * step;
                for axis in 0..2 {
                    let mut positions = [center, center, center];
                    positions[0][axis] += offset - step;
                    positions[1][axis] += offset;
                    positions[2][axis] += offset + step;
                    let heights = positions
                        .map(|position| rendered_height(dataset, position[0], position[1], level));
                    curvature += (heights[0] - 2.0 * heights[1] + heights[2]).abs();
                    samples += 1;
                }
            }
            curvature / samples as f32
        }

        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap();
        let level = dataset.metadata.quadtree.visual_max_level;
        let beach_curvature = mean_profile_curvature(&dataset, [8_800.0, 14_550.0], level);
        let cliff_curvature = mean_profile_curvature(&dataset, [8_345.1, 14_734.4], level);

        assert!(
            beach_curvature < 0.10,
            "the Caracasbaai foreground beach must remain smooth"
        );
        assert!(
            cliff_curvature > 0.65,
            "the Caracasbaai cliff needs metre-scale breaks instead of one rounded sheet"
        );
    }

    #[test]
    fn caracasbaai_beach_meets_water_without_a_retaining_wall() {
        fn nearest_shore(dataset: &TerrainDataset, seed: DVec2) -> DVec2 {
            let spacing = dataset.metadata.coastline.distance_spacing_m;
            let radius_steps = (500.0 / spacing) as i32;
            let mut nearest = seed;
            let mut nearest_distance = f32::INFINITY;
            for z in -radius_steps..=radius_steps {
                for x in -radius_steps..=radius_steps {
                    let candidate = seed + DVec2::new(f64::from(x), f64::from(z)) * spacing;
                    let distance = dataset.shore_distance_m(candidate.x, candidate.y).abs();
                    if distance < nearest_distance {
                        nearest = candidate;
                        nearest_distance = distance;
                    }
                }
            }
            project_to_shoreline(dataset, nearest)
        }

        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap();
        let beach = nearest_shore(&dataset, DVec2::new(8_800.0, 14_550.0));

        assert!(dataset.shore_distance_m(beach.x, beach.y).abs() < 0.02);
        assert!(
            coast_top_height(&dataset, beach) < 0.5,
            "the gentle Caracasbaai beach must meet the water instead of forming a retaining wall"
        );
    }

    #[test]
    fn material_detail_normal_tiles_exactly() {
        assert_eq!(detail_height_texel(0, 37), detail_height_texel(256, 37));
        assert_eq!(detail_height_texel(91, 0), detail_height_texel(91, 256));
    }

    #[test]
    fn gentle_source_grade_suppresses_unresolved_mesh_normals() {
        let noisy = Vec3::new(0.25, 1.0, -0.18).normalize();
        let filtered = terrain_shading_normal(noisy, 0.01);

        assert!(filtered.dot(Vec3::Y) > 0.999);
    }

    #[test]
    fn resolved_source_grade_preserves_mesh_normals() {
        let resolved = Vec3::new(0.25, 1.0, -0.18).normalize();
        let filtered = terrain_shading_normal(resolved, 0.30);

        assert!(filtered.dot(resolved) > 0.999);
    }

    #[test]
    fn edge_morph_is_exact_then_smoothly_releases() {
        assert_eq!(edge_morph_weight(0), 0.0);
        assert!(edge_morph_weight(4) > 0.0 && edge_morph_weight(4) < 1.0);
        assert_eq!(edge_morph_weight(EDGE_MORPH_CELLS), 1.0);
    }

    #[test]
    fn sand_is_limited_to_beaches_and_sparse_low_flat_ground() {
        let beach = sand_coverage(1.0, 0.0, 1.0, 0.5);
        let dry_plain = sand_coverage(4.0, 120.0, 1.0, 0.9);

        assert_eq!(beach, 0.76);
        assert!(dry_plain > 0.0 && dry_plain < beach);
        assert_eq!(sand_coverage(1.0, 0.0, 0.0, 0.9), 0.0);
        assert_eq!(sand_coverage(40.0, 120.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn canopy_proxy_yields_to_real_geometry_at_close_lod() {
        assert!(canopy_proxy_strength(0) > canopy_proxy_strength(6));
        assert_eq!(canopy_proxy_strength(u8::MAX), canopy_proxy_strength(6));
    }

    #[test]
    fn coastline_clips_a_mixed_triangle_instead_of_keeping_the_ocean_wedge() {
        let triangle = [
            ClipPoint::grid(0, 8.0),
            ClipPoint::grid(1, 3.0),
            ClipPoint::grid(2, -5.0),
        ];

        let clipped = clip_triangle_to_land(triangle);

        assert_eq!(clipped.len(), 2);
        assert!(
            clipped
                .iter()
                .flatten()
                .all(|point| point.distance_m >= 0.0)
        );
        let coast_vertices: std::collections::HashSet<_> = clipped
            .iter()
            .flatten()
            .filter_map(|point| match point.vertex {
                VertexRef::CoastEdge { a, b } => Some((a.min(b), a.max(b))),
                VertexRef::Grid(_) | VertexRef::CoastSample(_) => None,
            })
            .collect();
        assert_eq!(coast_vertices.len(), 2);
    }

    #[test]
    fn real_coastline_boundary_is_watertight_and_subdivided() {
        let asset_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/terrain/curacao");
        let dataset = TerrainDataset::load(&asset_dir).unwrap();
        let level = dataset.metadata.quadtree.visual_max_level;
        let side = 1u32 << level;
        let key = (0..side)
            .flat_map(|z| (0..side).map(move |x| TileKey { level, x, z }))
            .find(|key| {
                dataset.quadtree_coverage(key.level, key.x, key.z)
                    == (super::super::data::COVERAGE_LAND | super::super::data::COVERAGE_WATER)
            })
            .unwrap();
        let spatial = TerrainSpatialFrame::new(&dataset, crate::cli::SpatialMode::Planar).unwrap();
        let built = build_tile_mesh(&dataset, &spatial, key, EdgeStitch::default());
        let VertexAttributeValues::Float32x3(positions) =
            built.mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        else {
            panic!("terrain positions must be Float32x3")
        };
        let indices: Vec<usize> = built.mesh.indices().unwrap().iter().collect();
        let mut edges = HashMap::new();
        for triangle in indices.chunks_exact(3) {
            for (a, b) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                *edges.entry((a.min(b), a.max(b))).or_insert(0usize) += 1;
            }
        }

        let bounds = dataset.quadtree_bounds(key.level, key.x, key.z);
        let tile_size = (bounds[2] - bounds[0]) as f32;
        let on_same_tile_edge = |a: [f32; 3], b: [f32; 3]| {
            let epsilon = 0.01;
            (a[0].abs() < epsilon && b[0].abs() < epsilon)
                || ((a[0] - tile_size).abs() < epsilon && (b[0] - tile_size).abs() < epsilon)
                || (a[2].abs() < epsilon && b[2].abs() < epsilon)
                || ((a[2] - tile_size).abs() < epsilon && (b[2] - tile_size).abs() < epsilon)
        };
        let coast_edges: Vec<(usize, usize, f32)> = edges
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .filter_map(|((a, b), _)| {
                let position_a = positions[a];
                let position_b = positions[b];
                (!on_same_tile_edge(position_a, position_b)).then_some((
                    a,
                    b,
                    (position_a[0] - position_b[0])
                        .hypot(position_a[1] - position_b[1])
                        .hypot(position_a[2] - position_b[2]),
                ))
            })
            .collect();
        assert!(!coast_edges.is_empty());
        let max_coast_length = coast_edges
            .iter()
            .map(|(_, _, length)| *length)
            .fold(0.0, f32::max);
        assert!(max_coast_length <= COAST_EDGE_MAX_LENGTH_M as f32 + 0.01);
        for (a, b, _) in coast_edges {
            for index in [a, b] {
                let source = built.source_positions_m[index];
                assert!(
                    dataset.shore_distance_m(source[0], source[1]).abs() < 0.02,
                    "every subdivided boundary vertex must stay on the authored zero contour"
                );
            }
        }
    }
}
