use std::{
    env,
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use thalos_geodetic::{GeographicPosition, UtmPosition, wgs84_to_utm_north};
use tiff::{
    decoder::{Decoder, DecodingResult, Limits},
    tags::Tag,
};

mod coastline;

use coastline::{Coastline, DISTANCE_CLAMP_M, DISTANCE_DECIMETRES_PER_METRE};

const CROP_WEST: f64 = -69.20;
const CROP_SOUTH: f64 = 12.00;
const CROP_EAST: f64 = -68.70;
const CROP_NORTH: f64 = 12.43;
const SAMPLE_SPACING_M: f64 = 30.0;
const CHUNK_CELLS: usize = 64;
const QUADTREE_TILE_CELLS: usize = 64;
const MIP_COUNT: usize = 4;
const UTM_ZONE: u8 = 19;
const COAST_DISTANCE_FILE: &str = "coast_distance_l6.i16";
const COAST_POLYLINE_FILE: &str = "coast_polylines.bin";

const ATTRIBUTION: &str = "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved";
const OSM_ATTRIBUTION: &str =
    "© OpenStreetMap contributors, data available under the Open Database License (ODbL)";
const SENTINEL_ATTRIBUTION: &str = "Contains modified Copernicus Sentinel-2 data (ESA), processed to L2A COGs by Element 84 / AWS Earth Search";
const COAST_ATTRIBUTION: &str = "© OpenStreetMap contributors, data available under the Open Database License (ODbL); Contains modified Copernicus Sentinel-2 data (ESA), processed to L2A COGs by Element 84 / AWS Earth Search";

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let output_dir = args.next().map(PathBuf::from).context(
        "usage: korsou_terrain_baker OUTPUT_DIR --coastline COAST.json INPUT.tif [INPUT.tif ...]",
    )?;
    let mut coastline_path = None;
    let mut input_paths = Vec::new();
    while let Some(argument) = args.next() {
        if argument == "--coastline" {
            coastline_path = Some(
                args.next()
                    .map(PathBuf::from)
                    .context("--coastline requires a coastline-rings or Overpass JSON path")?,
            );
        } else {
            input_paths.push(PathBuf::from(argument));
        }
    }
    let coastline_path = coastline_path.context("--coastline COAST.json is required")?;
    ensure!(
        !input_paths.is_empty(),
        "at least one georeferenced Copernicus GeoTIFF is required"
    );

    println!("reading {} source raster(s)", input_paths.len());
    let sources: Vec<SourceRaster> = input_paths
        .iter()
        .map(|path| SourceRaster::read(path))
        .collect::<Result<_>>()?;

    for source in &sources {
        println!(
            "  {}: {}×{}, {:.8}° pixels, elevation {:.2}..{:.2} m, {}",
            source.path.display(),
            source.width,
            source.height,
            source.pixel_scale[0],
            source.min_elevation_m,
            source.max_elevation_m,
            if source.raster_is_point {
                "RasterPixelIsPoint"
            } else {
                "RasterPixelIsArea"
            }
        );
    }

    let projected_corners = [
        utm_forward(CROP_SOUTH, CROP_WEST, UTM_ZONE),
        utm_forward(CROP_SOUTH, CROP_EAST, UTM_ZONE),
        utm_forward(CROP_NORTH, CROP_WEST, UTM_ZONE),
        utm_forward(CROP_NORTH, CROP_EAST, UTM_ZONE),
    ];
    let raw_min_e = projected_corners
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min);
    let raw_max_e = projected_corners
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let raw_min_n = projected_corners
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min);
    let raw_max_n = projected_corners
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    let chunk_size_m = SAMPLE_SPACING_M * CHUNK_CELLS as f64;
    let min_e = (raw_min_e / chunk_size_m).floor() * chunk_size_m;
    let max_e = (raw_max_e / chunk_size_m).ceil() * chunk_size_m;
    let min_n = (raw_min_n / chunk_size_m).floor() * chunk_size_m;
    let max_n = (raw_max_n / chunk_size_m).ceil() * chunk_size_m;
    let width = ((max_e - min_e) / SAMPLE_SPACING_M).round() as usize + 1;
    let height = ((max_n - min_n) / SAMPLE_SPACING_M).round() as usize + 1;
    ensure!((width - 1) % CHUNK_CELLS == 0);
    ensure!((height - 1) % CHUNK_CELLS == 0);

    let local_origin = [(min_e + max_e) * 0.5, (min_n + max_n) * 0.5];
    let grid_bounds_local_m = [
        min_e - local_origin[0],
        local_origin[1] - max_n,
        max_e - local_origin[0],
        local_origin[1] - min_n,
    ];

    println!(
        "sampling EPSG:32619 grid: {}×{} posts, {:.1}×{:.1} km",
        width,
        height,
        (max_e - min_e) / 1000.0,
        (max_n - min_n) / 1000.0
    );

    let mut heights = vec![0.0f32; width * height];
    for row in 0..height {
        let local_z = grid_bounds_local_m[1] + row as f64 * SAMPLE_SPACING_M;
        for col in 0..width {
            let local_x = grid_bounds_local_m[0] + col as f64 * SAMPLE_SPACING_M;
            let (easting, northing) = local_to_utm(local_x, local_z, local_origin);
            let (lat, lon) = utm_inverse(easting, northing, UTM_ZONE);
            heights[row * width + col] = sample_sources(&sources, lon, lat).unwrap_or(0.0);
        }
        if row % 256 == 0 || row + 1 == height {
            println!("  sampled row {}/{}", row + 1, height);
        }
    }

    let coastline = Coastline::read(&coastline_path, local_origin)?;
    println!(
        "OSM coastline: {} ways, {} nodes, {} segments, timestamp {}",
        coastline.way_count,
        coastline.node_count,
        coastline.segment_count,
        coastline.source_timestamp
    );
    let mask = coastline.rasterize_land(grid_bounds_local_m, width, height, SAMPLE_SPACING_M);
    let land_posts = mask.iter().filter(|value| **value != 0).count();
    ensure!(land_posts > 0, "OSM coastline land mask is empty");

    println!("baking planar quadtree refinement errors");
    let mut quadtree = bake_quadtree_metadata(&heights, &mask, width, height, grid_bounds_local_m);

    fs::create_dir_all(&output_dir).with_context(|| format!("create {}", output_dir.display()))?;

    let mut level_metadata = Vec::with_capacity(MIP_COUNT);
    let mut level_heights = heights;
    let mut level_mask = mask;
    let mut level_width = width;
    let mut level_height = height;

    for level in 0..MIP_COUNT {
        let height_file = format!("height_l{level}.f32");
        let mask_file = format!("landmask_l{level}.u8");
        write_f32(&output_dir.join(&height_file), &level_heights)?;
        fs::write(output_dir.join(&mask_file), &level_mask)?;

        let (min_height_m, max_height_m) = finite_range(&level_heights);
        level_metadata.push(LevelMetadata {
            level,
            width: level_width,
            height: level_height,
            sample_spacing_m: SAMPLE_SPACING_M * (1usize << level) as f64,
            height_file,
            mask_file,
            min_height_m,
            max_height_m,
        });

        if level + 1 < MIP_COUNT {
            let next = downsample_height(&level_heights, level_width, level_height);
            let next_mask = downsample_mask(&level_mask, level_width, level_height);
            level_width = (level_width - 1) / 2 + 1;
            level_height = (level_height - 1) / 2 + 1;
            level_heights = next;
            level_mask = next_mask;
        }
    }

    let coast_bounds = quadtree.domain_bounds_local_m;
    let coast_width = (1usize << quadtree.visual_max_level) * QUADTREE_TILE_CELLS + 1;
    let coast_height = coast_width;
    let coast_spacing_m = (coast_bounds[2] - coast_bounds[0]) / (coast_width - 1) as f64;
    println!(
        "baking OSM shoreline field: {coast_width}×{coast_height}, {coast_spacing_m:.1} m posts"
    );
    let coast_mask =
        coastline.rasterize_land(coast_bounds, coast_width, coast_height, coast_spacing_m);
    quadtree.coverage = bake_quadtree_coverage(
        &coast_mask,
        coast_width,
        coast_height,
        quadtree.visual_max_level,
    );
    let coast_distance = coastline.signed_distance_field(
        &coast_mask,
        coast_bounds,
        coast_width,
        coast_height,
        coast_spacing_m,
    );
    write_i16(&output_dir.join(COAST_DISTANCE_FILE), &coast_distance)?;
    coastline.write_polylines(&output_dir.join(COAST_POLYLINE_FILE))?;

    let source_rasters = sources.iter().map(SourceRaster::metadata).collect();
    let metadata = Metadata {
        format_version: 6,
        name: "Curaçao Copernicus GLO-30 terrain explorer",
        source_product: "Copernicus DEM GLO-30, 2021 release",
        source_doi: "https://doi.org/10.5270/ESA-c5d3d65",
        attribution: ATTRIBUTION,
        source_crs: "EPSG:4326 (WGS 84)",
        projected_crs: "EPSG:32619 (WGS 84 / UTM zone 19N)",
        ellipsoid_crs: "EPSG:4979 (WGS 84 geographic 3D; EPSG:7030 ellipsoid)",
        vertical_crs: "EPSG:3855 (EGM2008 height)",
        height_relation: "WGS 84 ellipsoid height h = EGM2008 orthometric height H + geoid undulation N",
        geoid_model: "regional 5x5 GeographicLib egm2008-1 samples; bilinear; measured midpoint error <= 0.30 m",
        crop_bounds_wgs84: [CROP_WEST, CROP_SOUTH, CROP_EAST, CROP_NORTH],
        grid_bounds_utm_m: [min_e, min_n, max_e, max_n],
        local_origin_utm_m: local_origin,
        grid_bounds_local_m,
        chunk_cells: CHUNK_CELLS,
        chunk_size_m,
        quadtree,
        coastline: CoastlineMetadata {
            representation: "OSM land/sea rings densified with Sentinel-2 NDWI waterline crossings, kept as polylines and rasterized as a signed-distance field",
            method: "OSM natural=coastline rings stay the land/sea topology; Sentinel-2 B03/B08 NDWI zero-crossings densify long chords; runtime clips mixed triangles to those polylines",
            limitation: "Sentinel-2 waterline is a photographed instant inside a 40 m OSM corridor; tides, beach/cliff profiles, and inland water still need dedicated data. GLO-30 remains 30 m inland of the waterline",
            source_file: coastline
                .source_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            source_timestamp: coastline.source_timestamp,
            source_url: "https://www.openstreetmap.org",
            attribution: COAST_ATTRIBUTION,
            license: "Open Data Commons Open Database License (ODbL) 1.0; Copernicus Sentinel data (ESA)",
            polyline_file: COAST_POLYLINE_FILE,
            polyline_vertex_count: coastline.node_count,
            distance_file: COAST_DISTANCE_FILE,
            distance_width: coast_width,
            distance_height: coast_height,
            distance_spacing_m: coast_spacing_m,
            distance_bounds_local_m: coast_bounds,
            distance_units_per_metre: DISTANCE_DECIMETRES_PER_METRE,
            distance_clamp_m: DISTANCE_CLAMP_M,
            way_count: coastline.way_count,
            node_count: coastline.node_count,
        },
        synthetic_detail: "not baked; runtime-only visual displacement, never returned by DEM queries",
        source_rasters,
        levels: level_metadata,
    };

    let metadata_file = File::create(output_dir.join("metadata.json"))?;
    serde_json::to_writer_pretty(BufWriter::new(metadata_file), &metadata)?;
    println!(
        "wrote {} ({:.1}% land posts)\n{}",
        output_dir.display(),
        land_posts as f64 * 100.0 / (width * height) as f64,
        format!("{ATTRIBUTION}\n{OSM_ATTRIBUTION}\n{SENTINEL_ATTRIBUTION}")
    );
    Ok(())
}

#[derive(Debug)]
struct SourceRaster {
    path: PathBuf,
    width: usize,
    height: usize,
    pixel_scale: [f64; 2],
    tiepoint_raster: [f64; 2],
    tiepoint_model: [f64; 2],
    raster_is_point: bool,
    no_data: Option<f32>,
    data: Vec<f32>,
    min_elevation_m: f32,
    max_elevation_m: f32,
}

impl SourceRaster {
    fn read(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut decoder = Decoder::new(BufReader::new(file))?.with_limits(Limits::unlimited());
        let (width, height) = decoder.dimensions()?;
        let scale = decoder
            .get_tag_f64_vec(Tag::ModelPixelScaleTag)
            .context("GeoTIFF is missing ModelPixelScaleTag")?;
        let tie = decoder
            .get_tag_f64_vec(Tag::ModelTiepointTag)
            .context("GeoTIFF is missing ModelTiepointTag")?;
        ensure!(
            scale.len() >= 2 && tie.len() >= 6,
            "invalid GeoTIFF transform"
        );
        let raster_is_point = decoder
            .get_tag_u16_vec(Tag::GeoKeyDirectoryTag)
            .ok()
            .and_then(|keys| raster_type_is_point(&keys))
            .unwrap_or(false);
        let no_data = decoder
            .get_tag_ascii_string(Tag::GdalNodata)
            .ok()
            .and_then(|value| value.trim_matches('\0').parse::<f32>().ok());
        let data = decode_f32(decoder.read_image()?)?;
        ensure!(
            data.len() == width as usize * height as usize,
            "only one-band height GeoTIFFs are supported"
        );
        let (min_elevation_m, max_elevation_m) = finite_range_without_nodata(&data, no_data);
        Ok(Self {
            path: path.to_owned(),
            width: width as usize,
            height: height as usize,
            pixel_scale: [scale[0], scale[1]],
            tiepoint_raster: [tie[0], tie[1]],
            tiepoint_model: [tie[3], tie[4]],
            raster_is_point,
            no_data,
            data,
            min_elevation_m,
            max_elevation_m,
        })
    }

    fn sample(&self, lon: f64, lat: f64) -> Option<f32> {
        let center_offset = if self.raster_is_point { 0.0 } else { 0.5 };
        let x = (lon - self.tiepoint_model[0]) / self.pixel_scale[0] + self.tiepoint_raster[0]
            - center_offset;
        let y = (self.tiepoint_model[1] - lat) / self.pixel_scale[1] + self.tiepoint_raster[1]
            - center_offset;
        // The public AWS COG conversion removes the original shared east and
        // south border posts. Accept one clamped sample beyond each file so
        // adjacent COGs still form a continuous mosaic at their hand-off.
        if x < -1.001
            || y < -1.001
            || x > self.width as f64 + 0.001
            || y > self.height as f64 + 0.001
        {
            return None;
        }
        let x = x.clamp(0.0, self.width as f64 - 1.0);
        let y = y.clamp(0.0, self.height as f64 - 1.0);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = (x - x0 as f64) as f32;
        let ty = (y - y0 as f64) as f32;
        let samples = [
            self.data[y0 * self.width + x0],
            self.data[y0 * self.width + x1],
            self.data[y1 * self.width + x0],
            self.data[y1 * self.width + x1],
        ];
        if samples.iter().any(|value| self.is_nodata(*value)) {
            return samples.into_iter().find(|value| !self.is_nodata(*value));
        }
        let top = samples[0] + (samples[1] - samples[0]) * tx;
        let bottom = samples[2] + (samples[3] - samples[2]) * tx;
        Some(top + (bottom - top) * ty)
    }

    fn is_nodata(&self, value: f32) -> bool {
        !value.is_finite() || self.no_data.is_some_and(|no_data| value == no_data)
    }

    fn metadata(&self) -> SourceMetadata {
        let center_offset = if self.raster_is_point { 0.0 } else { 0.5 };
        let west = self.tiepoint_model[0]
            + (center_offset - self.tiepoint_raster[0]) * self.pixel_scale[0];
        let north = self.tiepoint_model[1]
            - (center_offset - self.tiepoint_raster[1]) * self.pixel_scale[1];
        SourceMetadata {
            file: self
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            width: self.width,
            height: self.height,
            pixel_scale_degrees: self.pixel_scale,
            sample_bounds_wgs84: [
                west,
                north - (self.height - 1) as f64 * self.pixel_scale[1],
                west + (self.width - 1) as f64 * self.pixel_scale[0],
                north,
            ],
            raster_type: if self.raster_is_point {
                "RasterPixelIsPoint"
            } else {
                "RasterPixelIsArea"
            },
            no_data: self.no_data,
            min_elevation_m: self.min_elevation_m,
            max_elevation_m: self.max_elevation_m,
        }
    }
}

fn raster_type_is_point(keys: &[u16]) -> Option<bool> {
    if keys.len() < 4 {
        return None;
    }
    let key_count = keys[3] as usize;
    for entry in keys[4..].chunks_exact(4).take(key_count) {
        if entry[0] == 1025 && entry[1] == 0 {
            return Some(entry[3] == 2);
        }
    }
    None
}

fn decode_f32(decoded: DecodingResult) -> Result<Vec<f32>> {
    Ok(match decoded {
        DecodingResult::F32(values) => values,
        DecodingResult::F64(values) => values.into_iter().map(|value| value as f32).collect(),
        DecodingResult::I16(values) => values.into_iter().map(f32::from).collect(),
        DecodingResult::I32(values) => values.into_iter().map(|value| value as f32).collect(),
        DecodingResult::U16(values) => values.into_iter().map(f32::from).collect(),
        other => bail!("unsupported GeoTIFF sample type: {other:?}"),
    })
}

fn sample_sources(sources: &[SourceRaster], lon: f64, lat: f64) -> Option<f32> {
    sources.iter().find_map(|source| source.sample(lon, lat))
}

fn downsample_height(source: &[f32], width: usize, height: usize) -> Vec<f32> {
    let next_width = (width - 1) / 2 + 1;
    let next_height = (height - 1) / 2 + 1;
    let mut result = vec![0.0; next_width * next_height];
    let weights = [1.0f32, 2.0, 1.0];
    for y in 0..next_height {
        for x in 0..next_width {
            let source_x = x * 2;
            let source_y = y * 2;
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for (wy, dy) in (-1isize..=1).enumerate() {
                for (wx, dx) in (-1isize..=1).enumerate() {
                    let sx = (source_x as isize + dx).clamp(0, width as isize - 1) as usize;
                    let sy = (source_y as isize + dy).clamp(0, height as isize - 1) as usize;
                    let weight = weights[wx] * weights[wy];
                    sum += source[sy * width + sx] * weight;
                    weight_sum += weight;
                }
            }
            result[y * next_width + x] = sum / weight_sum;
        }
    }
    result
}

fn downsample_mask(source: &[u8], width: usize, height: usize) -> Vec<u8> {
    let next_width = (width - 1) / 2 + 1;
    let next_height = (height - 1) / 2 + 1;
    let mut result = vec![0; next_width * next_height];
    for y in 0..next_height {
        for x in 0..next_width {
            let sx = x * 2;
            let sy = y * 2;
            let mut land = false;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let px = (sx as isize + dx).clamp(0, width as isize - 1) as usize;
                    let py = (sy as isize + dy).clamp(0, height as isize - 1) as usize;
                    land |= source[py * width + px] != 0;
                }
            }
            result[y * next_width + x] = if land { 255 } else { 0 };
        }
    }
    result
}

fn bake_quadtree_metadata(
    heights: &[f32],
    mask: &[u8],
    width: usize,
    height: usize,
    grid_bounds_local_m: [f64; 4],
) -> QuadtreeMetadata {
    let root_cells = (width - 1).max(height - 1).next_power_of_two();
    assert_eq!(root_cells % QUADTREE_TILE_CELLS, 0);
    let native_max_level = (root_cells / QUADTREE_TILE_CELLS).ilog2() as u8;
    let visual_max_level = native_max_level + 1;
    let root_size_m = root_cells as f64 * SAMPLE_SPACING_M;
    let domain_bounds_local_m = [
        -root_size_m * 0.5,
        -root_size_m * 0.5,
        root_size_m * 0.5,
        root_size_m * 0.5,
    ];
    let offset_x =
        ((domain_bounds_local_m[0] - grid_bounds_local_m[0]) / SAMPLE_SPACING_M).round() as isize;
    let offset_z =
        ((domain_bounds_local_m[1] - grid_bounds_local_m[1]) / SAMPLE_SPACING_M).round() as isize;

    let node_count = ((4usize.pow(native_max_level as u32 + 1)) - 1) / 3;
    let mut geometric_errors_m = Vec::with_capacity(node_count);
    let mut coverage = Vec::with_capacity(node_count);

    for level in 0..=native_max_level {
        let nodes_per_side = 1usize << level;
        let node_cells = root_cells / nodes_per_side;
        let stride = node_cells / QUADTREE_TILE_CELLS;
        for node_z in 0..nodes_per_side {
            for node_x in 0..nodes_per_side {
                let start_x = node_x * node_cells;
                let start_z = node_z * node_cells;
                let mut flags = 0u8;
                let mut max_error = 0.0f32;
                for z in start_z..=start_z + node_cells {
                    for x in start_x..=start_x + node_cells {
                        let land = domain_mask(mask, width, height, x, z, offset_x, offset_z);
                        flags |= if land { 1 } else { 2 };
                        if !land || stride == 1 {
                            continue;
                        }
                        let actual =
                            domain_height(heights, width, height, x, z, offset_x, offset_z);
                        let rel_x = x - start_x;
                        let rel_z = z - start_z;
                        let cell_x = (rel_x / stride).min(QUADTREE_TILE_CELLS - 1);
                        let cell_z = (rel_z / stride).min(QUADTREE_TILE_CELLS - 1);
                        let x0 = start_x + cell_x * stride;
                        let z0 = start_z + cell_z * stride;
                        let x1 = x0 + stride;
                        let z1 = z0 + stride;
                        let tx = (rel_x - cell_x * stride) as f32 / stride as f32;
                        let tz = (rel_z - cell_z * stride) as f32 / stride as f32;
                        let h00 = domain_height(heights, width, height, x0, z0, offset_x, offset_z);
                        let h10 = domain_height(heights, width, height, x1, z0, offset_x, offset_z);
                        let h01 = domain_height(heights, width, height, x0, z1, offset_x, offset_z);
                        let h11 = domain_height(heights, width, height, x1, z1, offset_x, offset_z);
                        let approximation = if tx + tz <= 1.0 {
                            h00 + tx * (h10 - h00) + tz * (h01 - h00)
                        } else {
                            h11 + (1.0 - tz) * (h10 - h11) + (1.0 - tx) * (h01 - h11)
                        };
                        max_error = max_error.max((actual - approximation).abs());
                    }
                }
                geometric_errors_m.push(max_error);
                coverage.push(flags);
            }
        }
        println!(
            "  quadtree level {level}: {} nodes, {:.0} m grid spacing",
            nodes_per_side * nodes_per_side,
            stride as f64 * SAMPLE_SPACING_M
        );
    }

    QuadtreeMetadata {
        domain_bounds_local_m,
        tile_grid_size: QUADTREE_TILE_CELLS + 1,
        native_max_level,
        visual_max_level,
        geometric_errors_m,
        coverage,
    }
}

fn bake_quadtree_coverage(mask: &[u8], width: usize, height: usize, max_level: u8) -> Vec<u8> {
    assert_eq!(width, height);
    assert_eq!(mask.len(), width * height);
    let finest_side = 1usize << max_level;
    let tile_cells = (width - 1) / finest_side;
    assert_eq!(tile_cells, QUADTREE_TILE_CELLS);

    let mut levels = vec![Vec::new(); max_level as usize + 1];
    let mut finest = vec![0; finest_side * finest_side];
    for tile_z in 0..finest_side {
        for tile_x in 0..finest_side {
            let mut flags = 0;
            let start_x = tile_x * tile_cells;
            let start_z = tile_z * tile_cells;
            for z in start_z..=start_z + tile_cells {
                for x in start_x..=start_x + tile_cells {
                    flags |= if mask[z * width + x] == 0 { 2 } else { 1 };
                }
            }
            finest[tile_z * finest_side + tile_x] = flags;
        }
    }
    levels[max_level as usize] = finest;

    for level in (0..max_level).rev() {
        let side = 1usize << level;
        let child_side = side * 2;
        let mut coverage = vec![0; side * side];
        for z in 0..side {
            for x in 0..side {
                let child_x = x * 2;
                let child_z = z * 2;
                coverage[z * side + x] = levels[level as usize + 1][child_z * child_side + child_x]
                    | levels[level as usize + 1][child_z * child_side + child_x + 1]
                    | levels[level as usize + 1][(child_z + 1) * child_side + child_x]
                    | levels[level as usize + 1][(child_z + 1) * child_side + child_x + 1];
            }
        }
        levels[level as usize] = coverage;
    }

    levels.into_iter().flatten().collect()
}

fn domain_height(
    source: &[f32],
    width: usize,
    height: usize,
    domain_x: usize,
    domain_z: usize,
    offset_x: isize,
    offset_z: isize,
) -> f32 {
    let x = domain_x as isize + offset_x;
    let z = domain_z as isize + offset_z;
    if x < 0 || z < 0 || x >= width as isize || z >= height as isize {
        0.0
    } else {
        source[z as usize * width + x as usize]
    }
}

fn domain_mask(
    source: &[u8],
    width: usize,
    height: usize,
    domain_x: usize,
    domain_z: usize,
    offset_x: isize,
    offset_z: isize,
) -> bool {
    let x = domain_x as isize + offset_x;
    let z = domain_z as isize + offset_z;
    x >= 0
        && z >= 0
        && x < width as isize
        && z < height as isize
        && source[z as usize * width + x as usize] != 0
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn write_i16(path: &Path, values: &[i16]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn finite_range(values: &[f32]) -> (f32, f32) {
    finite_range_without_nodata(values, None)
}

fn finite_range_without_nodata(values: &[f32], no_data: Option<f32>) -> (f32, f32) {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && !no_data.is_some_and(|no_data| *value == no_data))
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        })
}

#[derive(Serialize)]
struct Metadata<'a> {
    format_version: u32,
    name: &'a str,
    source_product: &'a str,
    source_doi: &'a str,
    attribution: &'a str,
    source_crs: &'a str,
    projected_crs: &'a str,
    ellipsoid_crs: &'a str,
    vertical_crs: &'a str,
    height_relation: &'a str,
    geoid_model: &'a str,
    crop_bounds_wgs84: [f64; 4],
    grid_bounds_utm_m: [f64; 4],
    local_origin_utm_m: [f64; 2],
    grid_bounds_local_m: [f64; 4],
    chunk_cells: usize,
    chunk_size_m: f64,
    quadtree: QuadtreeMetadata,
    coastline: CoastlineMetadata<'a>,
    synthetic_detail: &'a str,
    source_rasters: Vec<SourceMetadata>,
    levels: Vec<LevelMetadata>,
}

#[derive(Serialize)]
struct QuadtreeMetadata {
    domain_bounds_local_m: [f64; 4],
    tile_grid_size: usize,
    native_max_level: u8,
    visual_max_level: u8,
    geometric_errors_m: Vec<f32>,
    coverage: Vec<u8>,
}

#[derive(Serialize)]
struct CoastlineMetadata<'a> {
    representation: &'a str,
    method: &'a str,
    limitation: &'a str,
    source_file: String,
    source_timestamp: String,
    source_url: &'a str,
    attribution: &'a str,
    license: &'a str,
    polyline_file: &'a str,
    polyline_vertex_count: usize,
    distance_file: &'a str,
    distance_width: usize,
    distance_height: usize,
    distance_spacing_m: f64,
    distance_bounds_local_m: [f64; 4],
    distance_units_per_metre: f32,
    distance_clamp_m: f64,
    way_count: usize,
    node_count: usize,
}

#[derive(Serialize)]
struct SourceMetadata {
    file: String,
    width: usize,
    height: usize,
    pixel_scale_degrees: [f64; 2],
    sample_bounds_wgs84: [f64; 4],
    raster_type: &'static str,
    no_data: Option<f32>,
    min_elevation_m: f32,
    max_elevation_m: f32,
}

#[derive(Serialize)]
struct LevelMetadata {
    level: usize,
    width: usize,
    height: usize,
    sample_spacing_m: f64,
    height_file: String,
    mask_file: String,
    min_height_m: f32,
    max_height_m: f32,
}

fn utm_forward(lat_deg: f64, lon_deg: f64, zone: u8) -> (f64, f64) {
    let projected = wgs84_to_utm_north(
        GeographicPosition::new(lat_deg, lon_deg).expect("baker coordinates must be valid"),
        zone,
    )
    .expect("baker coordinates must be inside northern UTM");
    (projected.easting_m, projected.northing_m)
}

fn utm_inverse(easting: f64, northing: f64, zone: u8) -> (f64, f64) {
    let geographic = UtmPosition::new_north(zone, easting, northing)
        .and_then(UtmPosition::to_wgs84)
        .expect("baker UTM coordinates must be valid");
    (geographic.latitude_deg, geographic.longitude_deg)
}

fn utm_to_local(easting: f64, northing: f64, origin: [f64; 2]) -> [f64; 2] {
    [easting - origin[0], origin[1] - northing]
}

fn local_to_utm(local_x: f64, local_z: f64, origin: [f64; 2]) -> (f64, f64) {
    (local_x + origin[0], origin[1] - local_z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utm_central_meridian_has_standard_false_easting() {
        let (easting, northing) = utm_forward(0.0, -69.0, 19);
        assert!((easting - 500_000.0).abs() < 1.0e-6);
        assert!(northing.abs() < 1.0e-6);
    }

    #[test]
    fn curacao_projection_round_trips() {
        for (lat, lon) in [(12.1696, -68.99), (12.38, -69.15), (12.04, -68.75)] {
            let (easting, northing) = utm_forward(lat, lon, 19);
            let (round_lat, round_lon) = utm_inverse(easting, northing, 19);
            assert!((round_lat - lat).abs() < 1.0e-7);
            assert!((round_lon - lon).abs() < 1.0e-7);
        }
    }

    #[test]
    fn negative_local_z_is_geographic_north() {
        let origin = [505_920.0, 1_349_760.0];
        assert_eq!(
            utm_to_local(origin[0], origin[1] + 1.0, origin),
            [0.0, -1.0]
        );
        assert_eq!(
            local_to_utm(0.0, -1.0, origin),
            (origin[0], origin[1] + 1.0)
        );
    }
}
