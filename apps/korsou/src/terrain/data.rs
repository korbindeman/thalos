use std::{fs, path::Path};

use super::coast::CoastPolylines;
use anyhow::{Context, Result, ensure};
use bevy::prelude::Resource;
use serde::Deserialize;

pub const COVERAGE_LAND: u8 = 1;
pub const COVERAGE_WATER: u8 = 2;

#[derive(Resource)]
pub struct TerrainDataset {
    pub metadata: TerrainMetadata,
    levels: Vec<TerrainLevel>,
    coast_distances: Vec<i16>,
    coast_polylines: CoastPolylines,
    land_bounds_local_m: [f64; 4],
}

impl TerrainDataset {
    pub fn load(asset_dir: &Path) -> Result<Self> {
        let metadata_bytes = fs::read(asset_dir.join("metadata.json"))
            .with_context(|| format!("read {}/metadata.json", asset_dir.display()))?;
        let metadata: TerrainMetadata = serde_json::from_slice(&metadata_bytes)?;
        ensure!(
            metadata.format_version == 6,
            "unsupported terrain format; rebuild with `korsou_terrain_baker`"
        );
        ensure!(!metadata.levels.is_empty(), "terrain has no height levels");
        ensure!(!metadata.name.is_empty() && !metadata.source_product.is_empty());
        ensure!(!metadata.source_doi.is_empty() && !metadata.attribution.is_empty());
        ensure!(metadata.source_crs.starts_with("EPSG:4326"));
        ensure!(metadata.projected_crs.starts_with("EPSG:32619"));
        ensure!(metadata.ellipsoid_crs.starts_with("EPSG:4979"));
        ensure!(metadata.vertical_crs.starts_with("EPSG:3855"));
        ensure!(metadata.height_relation.contains("h ="));
        ensure!(
            metadata
                .geoid_model
                .to_ascii_lowercase()
                .contains("egm2008")
        );
        ensure!(metadata.crop_bounds_wgs84[0] < metadata.crop_bounds_wgs84[2]);
        ensure!(metadata.crop_bounds_wgs84[1] < metadata.crop_bounds_wgs84[3]);
        ensure!(metadata.grid_bounds_utm_m[0] < metadata.grid_bounds_utm_m[2]);
        ensure!(metadata.grid_bounds_utm_m[1] < metadata.grid_bounds_utm_m[3]);
        ensure!(!metadata.synthetic_detail.is_empty());
        ensure!(!metadata.coastline.representation.is_empty());
        ensure!(!metadata.coastline.method.is_empty());
        ensure!(!metadata.coastline.limitation.is_empty());
        ensure!(!metadata.coastline.source_file.is_empty());
        ensure!(!metadata.coastline.source_timestamp.is_empty());
        ensure!(!metadata.coastline.source_url.is_empty());
        ensure!(!metadata.coastline.attribution.is_empty());
        ensure!(!metadata.coastline.license.is_empty());
        ensure!(metadata.coastline.way_count > 0);
        ensure!(metadata.coastline.node_count > 0);
        ensure!(!metadata.coastline.polyline_file.is_empty());
        ensure!(metadata.coastline.polyline_vertex_count > 0);
        ensure!(metadata.coastline.distance_units_per_metre > 0.0);
        ensure!(metadata.coastline.distance_spacing_m > 0.0);
        ensure!(metadata.coastline.distance_width > 1);
        ensure!(metadata.coastline.distance_height > 1);
        ensure!((metadata.quadtree.tile_grid_size - 1).is_power_of_two());
        ensure!(metadata.quadtree.tile_grid_size == 65);
        ensure!(metadata.quadtree.visual_max_level == metadata.quadtree.native_max_level + 1);
        let native_node_count =
            (4usize.pow(u32::from(metadata.quadtree.native_max_level) + 1) - 1) / 3;
        ensure!(metadata.quadtree.geometric_errors_m.len() == native_node_count);
        let visual_node_count =
            (4usize.pow(u32::from(metadata.quadtree.visual_max_level) + 1) - 1) / 3;
        ensure!(metadata.quadtree.coverage.len() == visual_node_count);
        for (index, level) in metadata.levels.iter().enumerate() {
            ensure!(level.level == index, "terrain levels must be sequential");
            ensure!(level.min_height_m <= level.max_height_m);
        }

        let levels: Vec<TerrainLevel> = metadata
            .levels
            .iter()
            .map(|level| TerrainLevel::load(asset_dir, level))
            .collect::<Result<_>>()?;
        let coast_bytes = fs::read(asset_dir.join(&metadata.coastline.distance_file))?;
        ensure!(
            coast_bytes.len()
                == metadata.coastline.distance_width
                    * metadata.coastline.distance_height
                    * size_of::<i16>(),
            "{} has the wrong byte length",
            metadata.coastline.distance_file
        );
        let coast_distances: Vec<i16> = coast_bytes
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let mut min_land_x = usize::MAX;
        let mut min_land_z = usize::MAX;
        let mut max_land_x = 0;
        let mut max_land_z = 0;
        for z in 0..metadata.coastline.distance_height {
            for x in 0..metadata.coastline.distance_width {
                if coast_distances[z * metadata.coastline.distance_width + x] >= 0 {
                    min_land_x = min_land_x.min(x);
                    min_land_z = min_land_z.min(z);
                    max_land_x = max_land_x.max(x);
                    max_land_z = max_land_z.max(z);
                }
            }
        }
        ensure!(min_land_x != usize::MAX, "terrain land mask is empty");
        let min_grid_x = metadata.coastline.distance_bounds_local_m[0];
        let min_grid_z = metadata.coastline.distance_bounds_local_m[1];
        let spacing = metadata.coastline.distance_spacing_m;
        let land_bounds_local_m = [
            min_grid_x + min_land_x as f64 * spacing,
            min_grid_z + min_land_z as f64 * spacing,
            min_grid_x + max_land_x as f64 * spacing,
            min_grid_z + max_land_z as f64 * spacing,
        ];

        let coast_polylines =
            CoastPolylines::load(&asset_dir.join(&metadata.coastline.polyline_file))?;
        ensure!(
            coast_polylines.vertex_count() == metadata.coastline.polyline_vertex_count,
            "coastline polyline vertex count does not match metadata"
        );

        Ok(Self {
            metadata,
            levels,
            coast_distances,
            coast_polylines,
            land_bounds_local_m,
        })
    }

    pub fn nearest_coast_point(&self, local_x: f64, local_z: f64) -> [f64; 2] {
        self.coast_polylines.nearest_point(local_x, local_z)
    }

    pub fn coast_path(&self, start: [f64; 2], end: [f64; 2], max_edge_m: f64) -> Vec<[f64; 2]> {
        self.coast_polylines.path(start, end, max_edge_m)
    }

    pub fn distance_to_coast_line_m(&self, local_x: f64, local_z: f64) -> f64 {
        self.coast_polylines.distance_m(local_x, local_z)
    }

    pub fn land_bounds_local_m(&self) -> [f64; 4] {
        self.land_bounds_local_m
    }

    pub fn dem_height(&self, local_x: f64, local_z: f64) -> f32 {
        self.sample_height_level(local_x, local_z, 0)
    }

    pub fn sample_height_level(&self, local_x: f64, local_z: f64, level: usize) -> f32 {
        let level = &self.levels[level.min(self.levels.len() - 1)];
        let min_x = self.metadata.grid_bounds_local_m[0];
        let min_z = self.metadata.grid_bounds_local_m[1];
        let x = (local_x - min_x) / level.sample_spacing_m;
        let z = (local_z - min_z) / level.sample_spacing_m;
        if x < 0.0 || z < 0.0 || x > (level.width - 1) as f64 || z > (level.height - 1) as f64 {
            return 0.0;
        }
        let x0 = x.floor() as usize;
        let z0 = z.floor() as usize;
        let x1 = (x0 + 1).min(level.width - 1);
        let z1 = (z0 + 1).min(level.height - 1);
        let tx = (x - x0 as f64) as f32;
        let tz = (z - z0 as f64) as f32;
        let h00 = level.heights[z0 * level.width + x0];
        let h10 = level.heights[z0 * level.width + x1];
        let h01 = level.heights[z1 * level.width + x0];
        let h11 = level.heights[z1 * level.width + x1];
        let north = h00 + (h10 - h00) * tx;
        let south = h01 + (h11 - h01) * tx;
        north + (south - north) * tz
    }

    pub fn is_land(&self, local_x: f64, local_z: f64) -> bool {
        self.shore_distance_m(local_x, local_z) >= 0.0
    }

    pub fn shore_distance_m(&self, local_x: f64, local_z: f64) -> f32 {
        let coast = &self.metadata.coastline;
        let min_x = coast.distance_bounds_local_m[0];
        let min_z = coast.distance_bounds_local_m[1];
        let x = (local_x - min_x) / coast.distance_spacing_m;
        let z = (local_z - min_z) / coast.distance_spacing_m;
        if x < 0.0
            || z < 0.0
            || x > (coast.distance_width - 1) as f64
            || z > (coast.distance_height - 1) as f64
        {
            return -(coast.distance_clamp_m as f32);
        }
        let x0 = x.floor() as usize;
        let z0 = z.floor() as usize;
        let x1 = (x0 + 1).min(coast.distance_width - 1);
        let z1 = (z0 + 1).min(coast.distance_height - 1);
        let tx = (x - x0 as f64) as f32;
        let tz = (z - z0 as f64) as f32;
        let sample = |x: usize, z: usize| {
            f32::from(self.coast_distances[z * coast.distance_width + x])
                / coast.distance_units_per_metre
        };
        let north = sample(x0, z0) + (sample(x1, z0) - sample(x0, z0)) * tx;
        let south = sample(x0, z1) + (sample(x1, z1) - sample(x0, z1)) * tx;
        north + (south - north) * tz
    }

    pub(crate) fn coast_distance_samples(&self) -> &[i16] {
        &self.coast_distances
    }

    pub fn quadtree_error_m(&self, level: u8, x: u32, z: u32) -> f32 {
        if level > self.metadata.quadtree.native_max_level {
            return 2.8;
        }
        self.metadata.quadtree.geometric_errors_m[quadtree_index(level, x, z)]
    }

    pub fn quadtree_coverage(&self, level: u8, x: u32, z: u32) -> u8 {
        self.metadata.quadtree.coverage[quadtree_index(level, x, z)]
    }

    pub fn quadtree_bounds(&self, level: u8, x: u32, z: u32) -> [f64; 4] {
        let root = self.metadata.quadtree.domain_bounds_local_m;
        let size = (root[2] - root[0]) / (1u32 << level) as f64;
        let min_x = root[0] + x as f64 * size;
        let min_z = root[1] + z as f64 * size;
        [min_x, min_z, min_x + size, min_z + size]
    }

    pub fn max_height_m(&self) -> f64 {
        f64::from(self.levels[0].max_height_m)
    }
}

fn quadtree_index(level: u8, x: u32, z: u32) -> usize {
    let side = 1usize << level;
    let offset = (4usize.pow(u32::from(level)) - 1) / 3;
    offset + z as usize * side + x as usize
}

struct TerrainLevel {
    width: usize,
    height: usize,
    sample_spacing_m: f64,
    heights: Vec<f32>,
    max_height_m: f32,
}

impl TerrainLevel {
    fn load(asset_dir: &Path, metadata: &TerrainLevelMetadata) -> Result<Self> {
        let bytes = fs::read(asset_dir.join(&metadata.height_file))?;
        ensure!(
            bytes.len() == metadata.width * metadata.height * 4,
            "{} has the wrong byte length",
            metadata.height_file
        );
        let heights = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let mask = fs::read(asset_dir.join(&metadata.mask_file))?;
        ensure!(
            mask.len() == metadata.width * metadata.height,
            "{} has the wrong byte length",
            metadata.mask_file
        );
        Ok(Self {
            width: metadata.width,
            height: metadata.height,
            sample_spacing_m: metadata.sample_spacing_m,
            heights,
            max_height_m: metadata.max_height_m,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct TerrainMetadata {
    pub format_version: u32,
    pub name: String,
    pub source_product: String,
    pub source_doi: String,
    pub attribution: String,
    pub source_crs: String,
    pub projected_crs: String,
    pub ellipsoid_crs: String,
    pub vertical_crs: String,
    pub height_relation: String,
    pub geoid_model: String,
    pub crop_bounds_wgs84: [f64; 4],
    pub grid_bounds_utm_m: [f64; 4],
    pub local_origin_utm_m: [f64; 2],
    pub grid_bounds_local_m: [f64; 4],
    pub quadtree: QuadtreeMetadata,
    pub coastline: CoastlineMetadata,
    pub synthetic_detail: String,
    pub levels: Vec<TerrainLevelMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct QuadtreeMetadata {
    pub domain_bounds_local_m: [f64; 4],
    pub tile_grid_size: usize,
    pub native_max_level: u8,
    pub visual_max_level: u8,
    pub geometric_errors_m: Vec<f32>,
    pub coverage: Vec<u8>,
}

#[derive(Debug, Deserialize)]
pub struct CoastlineMetadata {
    pub representation: String,
    pub method: String,
    pub limitation: String,
    pub source_file: String,
    pub source_timestamp: String,
    pub source_url: String,
    pub attribution: String,
    pub license: String,
    pub polyline_file: String,
    pub polyline_vertex_count: usize,
    pub distance_file: String,
    pub distance_width: usize,
    pub distance_height: usize,
    pub distance_spacing_m: f64,
    pub distance_bounds_local_m: [f64; 4],
    pub distance_units_per_metre: f32,
    pub distance_clamp_m: f64,
    pub way_count: usize,
    pub node_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct TerrainLevelMetadata {
    pub level: usize,
    pub width: usize,
    pub height: usize,
    pub sample_spacing_m: f64,
    pub height_file: String,
    pub mask_file: String,
    pub min_height_m: f32,
    pub max_height_m: f32,
}
