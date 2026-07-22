use std::{
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::dem::{self, PrepareOptions};

const SEARCH_URL: &str = "https://stac.astrogeology.usgs.gov/api/search";
const COLLECTION: &str = "kaguya_terrain_camera_usgs_dtms";

pub struct Options {
    pub raw_dir: PathBuf,
    pub processed_dir: PathBuf,
    pub manifest: PathBuf,
    pub per_region: usize,
}

#[derive(Deserialize)]
struct FeatureCollection {
    features: Vec<Feature>,
}

#[derive(Deserialize)]
struct Feature {
    id: String,
    bbox: [f64; 4],
    properties: Properties,
    assets: Assets,
}

#[derive(Deserialize)]
struct Properties {
    gsd: f32,
    #[serde(rename = "proj:shape")]
    shape: [usize; 2],
}

#[derive(Deserialize)]
struct Assets {
    dtm: Asset,
}

#[derive(Deserialize)]
struct Asset {
    href: String,
    #[serde(rename = "raster:bands")]
    bands: Vec<RasterBand>,
}

#[derive(Deserialize)]
struct RasterBand {
    statistics: Statistics,
}

#[derive(Deserialize)]
struct Statistics {
    valid_percent: Option<f32>,
}

#[derive(Deserialize, Serialize)]
struct ExpansionManifest {
    schema_version: u32,
    collection: String,
    split: String,
    target_metres_per_pixel: f32,
    entries: Vec<ManifestEntry>,
}

#[derive(Deserialize, Serialize)]
struct ManifestEntry {
    source_id: String,
    url: String,
    local_path: PathBuf,
    sha256: String,
    bytes: u64,
    bbox_lon_lat: [f64; 4],
    native_metres_per_pixel: f32,
    valid_percent: f32,
    prepared_patches: usize,
}

pub fn discover_and_prepare(options: &Options) -> Result<usize, Box<dyn std::error::Error>> {
    if options.per_region == 0 || options.per_region > 4 {
        return Err("per-region must be between 1 and 4".into());
    }
    std::fs::create_dir_all(&options.raw_dir)?;
    std::fs::create_dir_all(&options.processed_dir)?;
    if let Some(parent) = options.manifest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut selected = Vec::new();
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(60))
        .build();
    for latitude in [-60, -30, 0, 30] {
        for longitude in [-180, -120, -60, 0, 60, 120] {
            let response: FeatureCollection = agent
                .post(SEARCH_URL)
                .send_json(serde_json::json!({
                    "collections": [COLLECTION],
                    "bbox": [longitude, latitude, longitude + 60, latitude + 30],
                    "limit": 100
                }))?
                .into_json()?;
            selected.extend(
                response
                    .features
                    .into_iter()
                    .filter(acceptable)
                    .take(options.per_region),
            );
        }
    }
    selected.sort_by(|left, right| left.id.cmp(&right.id));
    selected.dedup_by(|left, right| left.id == right.id);
    eprintln!(
        "selected {} geographically distributed Kaguya DTMs",
        selected.len()
    );

    let mut entries = Vec::with_capacity(selected.len());
    for feature in selected {
        let path = options.raw_dir.join(format!("{}.tif", feature.id));
        if !path.exists() {
            eprintln!("downloading {}", feature.id);
            download(&agent, &feature.assets.dtm.href, &path)?;
        }
        let checksum = sha256_file(&path)?;
        let bytes = std::fs::metadata(&path)?.len();
        let prepared_patches = dem::prepare(&PrepareOptions {
            input: path.clone(),
            output: options.processed_dir.clone(),
            source_id: feature.id.clone(),
            split: "train".into(),
            expected_sha256: checksum.clone(),
            native_metres_per_pixel: feature.properties.gsd,
            target_metres_per_pixel: 40.0,
            patch_size: 256,
            stride: 256,
        })?;
        if prepared_patches == 0 {
            eprintln!("skipping {}: no valid complete patches", feature.id);
            continue;
        }
        eprintln!(
            "prepared {}: {} patches at {:.2} m/px",
            feature.id, prepared_patches, feature.properties.gsd
        );
        entries.push(ManifestEntry {
            source_id: feature.id,
            url: feature.assets.dtm.href,
            local_path: path,
            sha256: checksum,
            bytes,
            bbox_lon_lat: feature.bbox,
            native_metres_per_pixel: feature.properties.gsd,
            valid_percent: feature.assets.dtm.bands[0]
                .statistics
                .valid_percent
                .expect("accepted feature has valid-percent metadata"),
            prepared_patches,
        });
    }
    std::fs::write(
        &options.manifest,
        serde_json::to_vec_pretty(&ExpansionManifest {
            schema_version: 1,
            collection: COLLECTION.into(),
            split: "train".into(),
            target_metres_per_pixel: 40.0,
            entries,
        })?,
    )?;
    Ok(selected_count(&options.manifest)?)
}

pub fn materialize(options: &Options) -> Result<usize, Box<dyn std::error::Error>> {
    let manifest: ExpansionManifest = serde_json::from_slice(&std::fs::read(&options.manifest)?)?;
    if manifest.schema_version != 1
        || manifest.collection != COLLECTION
        || manifest.split != "train"
        || manifest.target_metres_per_pixel.to_bits() != 40.0f32.to_bits()
    {
        return Err("unsupported Kaguya expansion manifest contract".into());
    }
    std::fs::create_dir_all(&options.raw_dir)?;
    std::fs::create_dir_all(&options.processed_dir)?;
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(60))
        .build();
    for entry in &manifest.entries {
        let path = options.raw_dir.join(format!("{}.tif", entry.source_id));
        if !path.exists() {
            eprintln!("downloading frozen {}", entry.source_id);
            download(&agent, &entry.url, &path)?;
        }
        let bytes = std::fs::metadata(&path)?.len();
        let checksum = sha256_file(&path)?;
        if bytes != entry.bytes || checksum != entry.sha256 {
            return Err(format!(
                "frozen source {} mismatch: expected {} bytes / {}, got {bytes} bytes / {checksum}",
                entry.source_id, entry.bytes, entry.sha256
            )
            .into());
        }
        let patches = dem::prepare(&PrepareOptions {
            input: path,
            output: options.processed_dir.clone(),
            source_id: entry.source_id.clone(),
            split: manifest.split.clone(),
            expected_sha256: entry.sha256.clone(),
            native_metres_per_pixel: entry.native_metres_per_pixel,
            target_metres_per_pixel: manifest.target_metres_per_pixel,
            patch_size: 256,
            stride: 256,
        })?;
        if patches != entry.prepared_patches {
            return Err(format!(
                "frozen source {} patch-count mismatch: expected {}, got {patches}",
                entry.source_id, entry.prepared_patches
            )
            .into());
        }
        eprintln!("materialized {}: {patches} patches", entry.source_id);
    }
    Ok(manifest.entries.len())
}

fn acceptable(feature: &Feature) -> bool {
    let valid = feature
        .assets
        .dtm
        .bands
        .first()
        .and_then(|band| band.statistics.valid_percent)
        .map(|valid_percent| valid_percent >= 99.0)
        .unwrap_or(false);
    let centre = [
        (feature.bbox[0] + feature.bbox[2]) * 0.5,
        (feature.bbox[1] + feature.bbox[3]) * 0.5,
    ];
    let outside_fixed_blocks = [[-19.9, 9.2], [-11.6, -42.9]]
        .iter()
        .all(|block| (centre[0] - block[0]).abs() > 4.0 || (centre[1] - block[1]).abs() > 4.0);
    let resampled_short_side = feature.properties.shape[0].min(feature.properties.shape[1]) as f32
        * feature.properties.gsd
        / 40.0;
    valid
        && (12.0..=45.0).contains(&feature.properties.gsd)
        && resampled_short_side >= 256.0
        && outside_fixed_blocks
}

fn download(agent: &ureq::Agent, url: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let partial = path.with_extension("tif.partial");
    if partial.exists() {
        std::fs::remove_file(&partial)?;
    }
    let response = agent.get(url).call()?;
    let expected_bytes = response
        .header("Content-Length")
        .and_then(|value| value.parse::<u64>().ok());
    let mut reader = response.into_reader();
    let mut file = File::create(&partial)?;
    let written = io::copy(&mut reader, &mut file)?;
    if expected_bytes.is_some_and(|expected| expected != written) {
        return Err(format!(
            "incomplete download for {url}: wrote {written} of {} bytes",
            expected_bytes.expect("checked above")
        )
        .into());
    }
    std::fs::rename(partial, path)?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn selected_count(path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    Ok(value["entries"].as_array().map_or(0, Vec::len))
}
