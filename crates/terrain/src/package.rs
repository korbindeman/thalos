//! Versioned authored terrain-package container (ADR-20260720T211046Z-offline-terrain-packages).
//!
//! The package is deliberately producer-agnostic. Revision 1 can carry the
//! compatibility `StaticSurfaceData` substrate, while its manifest already
//! models the sparse cube-sphere node/blob boundary that the diffusion bakery
//! will populate. Runtime consumers see [`PackageSurface`], never the producer.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{DVec3, Vec3};
use serde::{Deserialize, Serialize};

use crate::cubemap::{Cubemap, CubemapFace};
use crate::query::SurfaceSample;
use crate::{
    BakedSurface, DynamicSurfaceState, PlanetSurface, Region, StaticSurfaceData, SurfaceQuery,
};

const MAGIC: &[u8; 8] = b"THLSPK01";
pub const SCHEMA_VERSION: u32 = 1;
const HEADER_LEN: usize = MAGIC.len() + 8;
const HEIGHT_BASE_RESOLUTION: u32 = 32;
const HEIGHT_TILE_RESOLUTION: u32 = 32;
// The compatibility cubemap is 2.7 km/texel at its finest Mira level. A
// quarter-kilometre vertical budget is conservative enough to preserve its
// resolved silhouette while proving sparse fallback; learned campaign bakes
// will choose this from their measured rate-distortion profile.
const HEIGHT_PRUNE_ERROR_M: f32 = 256.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TerrainPackageManifest {
    pub schema_version: u32,
    pub body_name: String,
    pub body_radius_m: f32,
    pub height_range_m: f32,
    /// Hash of authored terrain inputs + producer source identity.
    pub content_key: u64,
    pub producer: PackageProducer,
    pub height_pyramid: HeightPyramidSpec,
    /// Sparse hierarchy. Revision 1 emits one global compatibility node; the
    /// production bakery adds cube-sphere nodes without changing the container.
    pub nodes: Vec<PackageNode>,
    pub blobs: Vec<PackageBlob>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageProducer {
    pub name: String,
    pub version: String,
    /// Model/checkpoint identity for learned producers; `None` for the
    /// deterministic compatibility producer.
    pub model_hash: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HeightPyramidSpec {
    pub source_resolution: u32,
    pub base_resolution: u32,
    pub tile_resolution: u32,
    /// Includes the base level. Level `n` has `2^n` tiles per face edge.
    pub level_count: u8,
    pub max_fallback_error_m: f32,
    pub border_rule: PackageBorderRule,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackageBorderRule {
    /// Tiles own half-open texel rectangles. Face-edge ownership is resolved
    /// by the cube address before a producer writes package samples.
    CanonicalHalfOpen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PackageNodeAddress {
    Global,
    Cube { face: u8, lod: u8, x: u32, y: u32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageNode {
    pub address: PackageNodeAddress,
    pub parent: Option<u32>,
    /// `None` means the parent's predictor satisfies this node's error budget.
    pub blob_index: Option<u32>,
    /// Maximum declared geometric reconstruction error for this node.
    pub geometric_error_m: f32,
    /// Error before this node's optional residual is applied. This drives the
    /// adaptive retention decision and rate-distortion reporting.
    pub predictor_error_m: f32,
    /// Normalized producer-measured complexity, used for adaptive retention.
    pub complexity: f32,
    pub min_wavelength_m: f32,
    pub max_wavelength_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackageBlobKind {
    /// Compatibility substrate. Replaced by macro/residual node kinds in the
    /// diffusion producer, but retained as a readable v1 kind.
    StaticSurfaceV1,
    /// One complete cube-face base-height plane. Later revisions layer sparse
    /// residual children over these base nodes.
    HeightBase,
    HeightResidual,
    Conditioning,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackageCodec {
    RawBincode,
    RawU16Le,
    /// Four-byte little-endian f32 scale followed by signed i16 residuals.
    QuantizedI16Le,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageBlob {
    pub kind: PackageBlobKind,
    pub codec: PackageCodec,
    /// Byte offset relative to the start of the blob region.
    pub offset: u64,
    pub encoded_len: u64,
    pub decoded_len: u64,
    /// Stable FNV-1a checksum of encoded bytes.
    pub checksum: u64,
}

pub struct LoadedTerrainPackage {
    pub manifest: TerrainPackageManifest,
    pub static_surface: StaticSurfaceData,
}

impl TerrainPackageManifest {
    /// Identity of the exact validated artifact, including encoder output.
    /// `content_key` instead identifies authored inputs and rejects stale bakes.
    pub fn artifact_fingerprint(&self) -> u64 {
        bincode::serde::encode_to_vec(self, bincode_config())
            .map_or(self.content_key, |bytes| checksum64(&bytes))
    }
}

#[derive(Debug)]
pub enum PackageError {
    Missing {
        path: PathBuf,
    },
    Io {
        path: PathBuf,
        message: String,
    },
    Invalid {
        path: PathBuf,
        message: String,
    },
    ContentKeyMismatch {
        path: PathBuf,
        expected: u64,
        found: u64,
    },
}

impl fmt::Display for PackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { path } => write!(f, "terrain package not found: {}", path.display()),
            Self::Io { path, message } => {
                write!(f, "terrain package I/O at {}: {message}", path.display())
            }
            Self::Invalid { path, message } => {
                write!(
                    f,
                    "invalid terrain package at {}: {message}",
                    path.display()
                )
            }
            Self::ContentKeyMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "stale terrain package at {}: content key {found:016x}, expected {expected:016x}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for PackageError {}

type EncodedBlob = (PackageBlobKind, PackageCodec, Vec<u8>);

/// Write a v1 adaptive package atomically.
pub fn write_static_package(
    path: &Path,
    body_name: &str,
    content_key: u64,
    producer: PackageProducer,
    surface: &mut StaticSurfaceData,
) -> io::Result<TerrainPackageManifest> {
    let height = std::mem::replace(&mut surface.height_cubemap, Cubemap::new(1));
    let metadata_result = bincode::serde::encode_to_vec(&*surface, bincode_config());
    surface.height_cubemap = height;
    let metadata =
        metadata_result.map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;

    let mut encoded_blobs = Vec::new();
    encoded_blobs.push((
        PackageBlobKind::StaticSurfaceV1,
        PackageCodec::RawBincode,
        metadata,
    ));
    let (height_pyramid, nodes) = encode_height_pyramid(
        &surface.height_cubemap,
        surface.radius_m,
        surface.height_range,
        &mut encoded_blobs,
    )?;

    let mut blobs = Vec::with_capacity(encoded_blobs.len());
    let mut offset = 0u64;
    for (kind, codec, bytes) in &encoded_blobs {
        blobs.push(PackageBlob {
            kind: *kind,
            codec: *codec,
            offset,
            encoded_len: bytes.len() as u64,
            decoded_len: bytes.len() as u64,
            checksum: checksum64(bytes),
        });
        offset += bytes.len() as u64;
    }

    let manifest = TerrainPackageManifest {
        schema_version: SCHEMA_VERSION,
        body_name: body_name.to_owned(),
        body_radius_m: surface.radius_m,
        height_range_m: surface.height_range,
        content_key,
        producer,
        height_pyramid,
        nodes,
        blobs,
    };
    let manifest_bytes = bincode::serde::encode_to_vec(&manifest, bincode_config())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let blob_bytes_len: usize = encoded_blobs.iter().map(|(_, _, bytes)| bytes.len()).sum();
    let mut bytes = Vec::with_capacity(HEADER_LEN + manifest_bytes.len() + blob_bytes_len);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&manifest_bytes);
    for (_, _, blob) in encoded_blobs {
        bytes.extend_from_slice(&blob);
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("bin.tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(tmp, path)?;
    Ok(manifest)
}

fn encode_height_pyramid(
    height: &Cubemap<u16>,
    radius_m: f32,
    height_range_m: f32,
    blobs: &mut Vec<EncodedBlob>,
) -> io::Result<(HeightPyramidSpec, Vec<PackageNode>)> {
    let source_resolution = height.resolution();
    if source_resolution < HEIGHT_BASE_RESOLUTION
        || !source_resolution.is_power_of_two()
        || !HEIGHT_BASE_RESOLUTION.is_power_of_two()
        || !source_resolution.is_multiple_of(HEIGHT_BASE_RESOLUTION)
        || !(source_resolution / HEIGHT_BASE_RESOLUTION).is_power_of_two()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "height resolution must be a power-of-two multiple of the package base resolution",
        ));
    }
    let level_count = (source_resolution / HEIGHT_BASE_RESOLUTION).ilog2() as u8 + 1;
    let spec = HeightPyramidSpec {
        source_resolution,
        base_resolution: HEIGHT_BASE_RESOLUTION,
        tile_resolution: HEIGHT_TILE_RESOLUTION,
        level_count,
        max_fallback_error_m: HEIGHT_PRUNE_ERROR_M,
        border_rule: PackageBorderRule::CanonicalHalfOpen,
    };
    let metres_per_unit = height_range_m * 2.0 / f32::from(u16::MAX);
    let circumference = radius_m * std::f32::consts::TAU;
    let mut nodes = vec![PackageNode {
        address: PackageNodeAddress::Global,
        parent: None,
        blob_index: Some(0),
        geometric_error_m: 0.0,
        predictor_error_m: 0.0,
        complexity: 1.0,
        min_wavelength_m: 0.0,
        max_wavelength_m: circumference,
    }];
    let mut indices = HashMap::new();
    let mut reconstructed: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());

    for (face_index, face) in CubemapFace::ALL.into_iter().enumerate() {
        let base = resample_grid(
            height.face_data(face),
            source_resolution,
            spec.base_resolution,
        );
        let values = base
            .iter()
            .map(|value| value.round().clamp(0.0, f32::from(u16::MAX)) as u16)
            .collect::<Vec<_>>();
        reconstructed[face_index] = values.iter().map(|value| f32::from(*value)).collect();
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let blob_index = blobs.len() as u32;
        blobs.push((PackageBlobKind::HeightBase, PackageCodec::RawU16Le, bytes));
        let address = PackageNodeAddress::Cube {
            face: face_index as u8,
            lod: 0,
            x: 0,
            y: 0,
        };
        let node_index = nodes.len() as u32;
        indices.insert(address, node_index);
        nodes.push(PackageNode {
            address,
            parent: Some(0),
            blob_index: Some(blob_index),
            geometric_error_m: metres_per_unit * 0.5,
            predictor_error_m: 0.0,
            complexity: 1.0,
            min_wavelength_m: circumference / (4.0 * spec.base_resolution as f32),
            max_wavelength_m: circumference / 4.0,
        });
    }

    for lod in 1..spec.level_count {
        let edge = 1u32 << lod;
        let resolution = spec.tile_resolution * edge;
        for (face_index, face) in CubemapFace::ALL.into_iter().enumerate() {
            let target = resample_grid(height.face_data(face), source_resolution, resolution);
            let predictor = resample_grid(&reconstructed[face_index], resolution / 2, resolution);
            let mut next = predictor.clone();
            for tile_y in 0..edge {
                for tile_x in 0..edge {
                    let mut residuals =
                        Vec::with_capacity((spec.tile_resolution * spec.tile_resolution) as usize);
                    let mut max_abs = 0.0f32;
                    let mut sum_abs = 0.0f32;
                    for local_y in 0..spec.tile_resolution {
                        let y = tile_y * spec.tile_resolution + local_y;
                        for local_x in 0..spec.tile_resolution {
                            let x = tile_x * spec.tile_resolution + local_x;
                            let index = (y * resolution + x) as usize;
                            let residual = target[index] - predictor[index];
                            max_abs = max_abs.max(residual.abs());
                            sum_abs += residual.abs();
                            residuals.push(residual);
                        }
                    }
                    let fallback_error_m = max_abs * metres_per_unit;
                    let retained = fallback_error_m > spec.max_fallback_error_m;
                    let mut measured_error_m = fallback_error_m;
                    let blob_index = if retained {
                        let scale = (max_abs / f32::from(i16::MAX)).max(f32::EPSILON);
                        let mut bytes = Vec::with_capacity(4 + residuals.len() * 2);
                        bytes.extend_from_slice(&scale.to_le_bytes());
                        let mut quantized = Vec::with_capacity(residuals.len());
                        for residual in residuals {
                            let value = (residual / scale)
                                .round()
                                .clamp(f32::from(i16::MIN), f32::from(i16::MAX))
                                as i16;
                            bytes.extend_from_slice(&value.to_le_bytes());
                            quantized.push(value);
                        }
                        let mut quantization_error = 0.0f32;
                        let mut cursor = 0usize;
                        for local_y in 0..spec.tile_resolution {
                            let y = tile_y * spec.tile_resolution + local_y;
                            for local_x in 0..spec.tile_resolution {
                                let x = tile_x * spec.tile_resolution + local_x;
                                let index = (y * resolution + x) as usize;
                                next[index] += f32::from(quantized[cursor]) * scale;
                                quantization_error =
                                    quantization_error.max((target[index] - next[index]).abs());
                                cursor += 1;
                            }
                        }
                        measured_error_m = quantization_error * metres_per_unit;
                        let index = blobs.len() as u32;
                        blobs.push((
                            PackageBlobKind::HeightResidual,
                            PackageCodec::QuantizedI16Le,
                            bytes,
                        ));
                        Some(index)
                    } else {
                        None
                    };
                    let address = PackageNodeAddress::Cube {
                        face: face_index as u8,
                        lod,
                        x: tile_x,
                        y: tile_y,
                    };
                    let parent_address = PackageNodeAddress::Cube {
                        face: face_index as u8,
                        lod: lod - 1,
                        x: tile_x / 2,
                        y: tile_y / 2,
                    };
                    let parent = *indices.get(&parent_address).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "missing encoded parent node")
                    })?;
                    let node_index = nodes.len() as u32;
                    indices.insert(address, node_index);
                    let mean_abs = sum_abs / (spec.tile_resolution * spec.tile_resolution) as f32;
                    nodes.push(PackageNode {
                        address,
                        parent: Some(parent),
                        blob_index,
                        geometric_error_m: measured_error_m,
                        predictor_error_m: fallback_error_m,
                        complexity: (mean_abs * metres_per_unit
                            / (spec.max_fallback_error_m * 2.0))
                            .clamp(0.0, 1.0),
                        min_wavelength_m: circumference / (4.0 * resolution as f32),
                        max_wavelength_m: circumference / (4.0 * edge as f32),
                    });
                }
            }
            reconstructed[face_index] = next;
        }
    }

    Ok((spec, nodes))
}

fn resample_grid(
    source: &[impl Copy + Into<f32>],
    source_resolution: u32,
    target_resolution: u32,
) -> Vec<f32> {
    let mut result = Vec::with_capacity((target_resolution * target_resolution) as usize);
    let scale = source_resolution as f32 / target_resolution as f32;
    for y in 0..target_resolution {
        let source_y =
            ((y as f32 + 0.5) * scale - 0.5).clamp(0.0, source_resolution.saturating_sub(1) as f32);
        let y0 = source_y.floor() as u32;
        let y1 = (y0 + 1).min(source_resolution - 1);
        let fy = source_y - y0 as f32;
        for x in 0..target_resolution {
            let source_x = ((x as f32 + 0.5) * scale - 0.5)
                .clamp(0.0, source_resolution.saturating_sub(1) as f32);
            let x0 = source_x.floor() as u32;
            let x1 = (x0 + 1).min(source_resolution - 1);
            let fx = source_x - x0 as f32;
            let value = |x, y| source[(y * source_resolution + x) as usize].into();
            let top = value(x0, y0) * (1.0 - fx) + value(x1, y0) * fx;
            let bottom = value(x0, y1) * (1.0 - fx) + value(x1, y1) * fx;
            result.push(top * (1.0 - fy) + bottom * fy);
        }
    }
    result
}

/// Decode and fully validate a package before any surface becomes visible.
pub fn load_static_package(
    path: &Path,
    expected_body: &str,
    expected_content_key: u64,
) -> Result<LoadedTerrainPackage, PackageError> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(PackageError::Missing {
                path: path.to_path_buf(),
            });
        }
        Err(error) => {
            return Err(PackageError::Io {
                path: path.to_path_buf(),
                message: error.to_string(),
            });
        }
    };
    if bytes.len() < HEADER_LEN || &bytes[..MAGIC.len()] != MAGIC {
        return invalid(path, "magic/version header mismatch");
    }
    let manifest_len = u64::from_le_bytes(bytes[MAGIC.len()..HEADER_LEN].try_into().unwrap());
    let manifest_len = usize::try_from(manifest_len)
        .map_err(|_| invalid_error(path, "manifest length exceeds address space"))?;
    let manifest_end = HEADER_LEN
        .checked_add(manifest_len)
        .ok_or_else(|| invalid_error(path, "manifest length overflow"))?;
    if manifest_end > bytes.len() {
        return invalid(path, "manifest extends beyond file");
    }
    let (manifest, consumed): (TerrainPackageManifest, usize) =
        bincode::serde::decode_from_slice(&bytes[HEADER_LEN..manifest_end], bincode_config())
            .map_err(|error| invalid_error(path, format!("manifest decode failed: {error}")))?;
    if consumed != manifest_len {
        return invalid(path, "manifest has trailing bytes");
    }
    validate_manifest(
        path,
        &manifest,
        expected_body,
        expected_content_key,
        bytes.len() - manifest_end,
    )?;
    for (index, descriptor) in manifest.blobs.iter().enumerate() {
        let start = manifest_end + descriptor.offset as usize;
        let end = start + descriptor.encoded_len as usize;
        if checksum64(&bytes[start..end]) != descriptor.checksum {
            return invalid(path, format!("blob {index} checksum mismatch"));
        }
    }

    let global = manifest
        .nodes
        .iter()
        .find(|node| node.address == PackageNodeAddress::Global)
        .ok_or_else(|| invalid_error(path, "missing global substrate node"))?;
    let descriptor = manifest
        .blobs
        .get(
            global
                .blob_index
                .ok_or_else(|| invalid_error(path, "global substrate node has no blob"))?
                as usize,
        )
        .ok_or_else(|| invalid_error(path, "global node references missing blob"))?;
    if descriptor.kind != PackageBlobKind::StaticSurfaceV1
        || descriptor.codec != PackageCodec::RawBincode
    {
        return invalid(path, "global substrate has unsupported kind/codec");
    }
    let start = manifest_end + descriptor.offset as usize;
    let end = start + descriptor.encoded_len as usize;
    let blob = &bytes[start..end];
    let (mut static_surface, consumed): (StaticSurfaceData, usize) =
        bincode::serde::decode_from_slice(blob, bincode_config())
            .map_err(|error| invalid_error(path, format!("substrate decode failed: {error}")))?;
    if consumed != blob.len() {
        return invalid(path, "global substrate has trailing bytes");
    }
    static_surface.height_cubemap = decode_height_pyramid(path, &manifest, &bytes, manifest_end)?;
    if (static_surface.radius_m - manifest.body_radius_m).abs() > 0.5 {
        return invalid(path, "manifest and substrate radii disagree");
    }
    Ok(LoadedTerrainPackage {
        manifest,
        static_surface,
    })
}

fn decode_height_pyramid(
    path: &Path,
    manifest: &TerrainPackageManifest,
    bytes: &[u8],
    blob_region_start: usize,
) -> Result<Cubemap<u16>, PackageError> {
    let spec = manifest.height_pyramid;
    let addresses = manifest
        .nodes
        .iter()
        .map(|node| (node.address, node))
        .collect::<HashMap<_, _>>();
    let mut reconstructed: [Vec<f32>; 6] = std::array::from_fn(|_| Vec::new());
    for face in 0..6u8 {
        let address = PackageNodeAddress::Cube {
            face,
            lod: 0,
            x: 0,
            y: 0,
        };
        let node = addresses[&address];
        let descriptor = &manifest.blobs[node.blob_index.unwrap() as usize];
        let payload = blob_payload(bytes, blob_region_start, descriptor);
        let (values, remainder) = payload.as_chunks::<2>();
        if !remainder.is_empty() {
            return invalid(path, "height base payload is not u16-aligned");
        }
        reconstructed[face as usize] = values
            .iter()
            .map(|value| f32::from(u16::from_le_bytes(*value)))
            .collect();
    }

    for lod in 1..spec.level_count {
        let edge = 1u32 << lod;
        let resolution = spec.tile_resolution * edge;
        for face in 0..6u8 {
            let mut next = resample_grid(&reconstructed[face as usize], resolution / 2, resolution);
            for tile_y in 0..edge {
                for tile_x in 0..edge {
                    let address = PackageNodeAddress::Cube {
                        face,
                        lod,
                        x: tile_x,
                        y: tile_y,
                    };
                    let node = addresses[&address];
                    let Some(blob_index) = node.blob_index else {
                        continue;
                    };
                    let descriptor = &manifest.blobs[blob_index as usize];
                    let payload = blob_payload(bytes, blob_region_start, descriptor);
                    let scale = f32::from_le_bytes(payload[..4].try_into().unwrap());
                    if !scale.is_finite() || scale <= 0.0 {
                        return invalid(
                            path,
                            format!("height residual {address:?} has invalid scale"),
                        );
                    }
                    let (values, remainder) = payload[4..].as_chunks::<2>();
                    debug_assert!(remainder.is_empty());
                    let mut cursor = 0usize;
                    for local_y in 0..spec.tile_resolution {
                        let y = tile_y * spec.tile_resolution + local_y;
                        for local_x in 0..spec.tile_resolution {
                            let x = tile_x * spec.tile_resolution + local_x;
                            let index = (y * resolution + x) as usize;
                            next[index] += f32::from(i16::from_le_bytes(values[cursor])) * scale;
                            cursor += 1;
                        }
                    }
                }
            }
            reconstructed[face as usize] = next;
        }
    }

    let mut height = Cubemap::new(spec.source_resolution);
    for (face_index, values) in reconstructed.into_iter().enumerate() {
        for (destination, value) in height
            .face_data_mut(CubemapFace::ALL[face_index])
            .iter_mut()
            .zip(values)
        {
            *destination = value.round().clamp(0.0, f32::from(u16::MAX)) as u16;
        }
    }
    Ok(height)
}

fn blob_payload<'a>(
    bytes: &'a [u8],
    blob_region_start: usize,
    descriptor: &PackageBlob,
) -> &'a [u8] {
    let start = blob_region_start + descriptor.offset as usize;
    &bytes[start..start + descriptor.encoded_len as usize]
}

fn validate_manifest(
    path: &Path,
    manifest: &TerrainPackageManifest,
    expected_body: &str,
    expected_content_key: u64,
    blob_region_len: usize,
) -> Result<(), PackageError> {
    if manifest.schema_version != SCHEMA_VERSION {
        return invalid(
            path,
            format!(
                "schema {} is unsupported (expected {})",
                manifest.schema_version, SCHEMA_VERSION
            ),
        );
    }
    if manifest.body_name != expected_body {
        return invalid(
            path,
            format!(
                "package body {:?} does not match {:?}",
                manifest.body_name, expected_body
            ),
        );
    }
    if manifest.content_key != expected_content_key {
        return Err(PackageError::ContentKeyMismatch {
            path: path.to_path_buf(),
            expected: expected_content_key,
            found: manifest.content_key,
        });
    }
    if !manifest.body_radius_m.is_finite()
        || manifest.body_radius_m <= 0.0
        || !manifest.height_range_m.is_finite()
        || manifest.height_range_m <= 0.0
    {
        return invalid(path, "non-finite or non-positive body bounds");
    }
    let spec = manifest.height_pyramid;
    if spec.source_resolution == 0
        || spec.base_resolution == 0
        || spec.tile_resolution == 0
        || spec.level_count == 0
        || !spec.source_resolution.is_power_of_two()
        || !spec.base_resolution.is_power_of_two()
        || spec.base_resolution != spec.tile_resolution
        || !spec.source_resolution.is_multiple_of(spec.base_resolution)
        || !(spec.source_resolution / spec.base_resolution).is_power_of_two()
        || spec.level_count != (spec.source_resolution / spec.base_resolution).ilog2() as u8 + 1
        || !spec.max_fallback_error_m.is_finite()
        || spec.max_fallback_error_m < 0.0
    {
        return invalid(path, "invalid height-pyramid specification");
    }
    let mut addresses = HashMap::with_capacity(manifest.nodes.len());
    for (index, node) in manifest.nodes.iter().enumerate() {
        if addresses.insert(node.address, index).is_some() {
            return invalid(path, format!("node {index} duplicates an address"));
        }
        if let Some(blob_index) = node.blob_index
            && blob_index as usize >= manifest.blobs.len()
        {
            return invalid(path, format!("node {index} references missing blob"));
        }
        if let PackageNodeAddress::Cube { face, lod, x, y } = node.address {
            let edge = 1u32.checked_shl(lod as u32).unwrap_or(0);
            if face >= 6 || lod >= spec.level_count || edge == 0 || x >= edge || y >= edge {
                return invalid(path, format!("node {index} has invalid cube address"));
            }
        }
        if let Some(parent) = node.parent
            && parent as usize >= manifest.nodes.len()
        {
            return invalid(path, format!("node {index} references missing parent"));
        }
        if !node.geometric_error_m.is_finite()
            || node.geometric_error_m < 0.0
            || !node.predictor_error_m.is_finite()
            || node.predictor_error_m < 0.0
            || !node.complexity.is_finite()
            || !(0.0..=1.0).contains(&node.complexity)
            || !node.min_wavelength_m.is_finite()
            || node.min_wavelength_m < 0.0
            || !node.max_wavelength_m.is_finite()
            || node.max_wavelength_m < node.min_wavelength_m
        {
            return invalid(path, format!("node {index} has invalid metrics"));
        }
    }
    let global_index = *addresses
        .get(&PackageNodeAddress::Global)
        .ok_or_else(|| invalid_error(path, "missing global substrate node"))?;
    if manifest.nodes[global_index].parent.is_some()
        || manifest.nodes[global_index].blob_index.is_none()
    {
        return invalid(path, "global substrate node has invalid parent/blob");
    }
    let mut expected_addresses = HashSet::new();
    expected_addresses.insert(PackageNodeAddress::Global);
    for face in 0..6u8 {
        for lod in 0..spec.level_count {
            let edge = 1u32 << lod;
            for y in 0..edge {
                for x in 0..edge {
                    let address = PackageNodeAddress::Cube { face, lod, x, y };
                    expected_addresses.insert(address);
                    let index = *addresses.get(&address).ok_or_else(|| {
                        invalid_error(path, format!("missing height node {address:?}"))
                    })?;
                    let node = &manifest.nodes[index];
                    let expected_parent = if lod == 0 {
                        global_index
                    } else {
                        *addresses
                            .get(&PackageNodeAddress::Cube {
                                face,
                                lod: lod - 1,
                                x: x / 2,
                                y: y / 2,
                            })
                            .unwrap()
                    };
                    if node.parent != Some(expected_parent as u32) {
                        return invalid(path, format!("height node {address:?} has wrong parent"));
                    }
                    match (lod, node.blob_index) {
                        (0, Some(blob_index)) => {
                            let blob = &manifest.blobs[blob_index as usize];
                            let expected_len = u64::from(spec.base_resolution).pow(2) * 2;
                            if blob.kind != PackageBlobKind::HeightBase
                                || blob.codec != PackageCodec::RawU16Le
                                || blob.encoded_len != expected_len
                                || blob.decoded_len != expected_len
                            {
                                return invalid(
                                    path,
                                    format!("height base {address:?} has invalid payload"),
                                );
                            }
                        }
                        (0, None) => {
                            return invalid(path, format!("height base {address:?} is pruned"));
                        }
                        (_, Some(blob_index)) => {
                            let blob = &manifest.blobs[blob_index as usize];
                            let expected_len = 4 + u64::from(spec.tile_resolution).pow(2) * 2;
                            if blob.kind != PackageBlobKind::HeightResidual
                                || blob.codec != PackageCodec::QuantizedI16Le
                                || blob.encoded_len != expected_len
                                || blob.decoded_len != expected_len
                            {
                                return invalid(
                                    path,
                                    format!("height residual {address:?} has invalid payload"),
                                );
                            }
                        }
                        (_, None)
                            if node.geometric_error_m
                                <= spec.max_fallback_error_m + f32::EPSILON => {}
                        (_, None) => {
                            return invalid(
                                path,
                                format!("pruned node {address:?} exceeds fallback budget"),
                            );
                        }
                    }
                }
            }
        }
    }
    if addresses.len() != expected_addresses.len() {
        return invalid(path, "manifest contains unexpected non-height nodes");
    }
    let mut ranges = Vec::with_capacity(manifest.blobs.len());
    for (index, blob) in manifest.blobs.iter().enumerate() {
        let start = usize::try_from(blob.offset)
            .map_err(|_| invalid_error(path, format!("blob {index} offset overflow")))?;
        let len = usize::try_from(blob.encoded_len)
            .map_err(|_| invalid_error(path, format!("blob {index} length overflow")))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| invalid_error(path, format!("blob {index} range overflow")))?;
        if end > blob_region_len {
            return invalid(path, format!("blob {index} extends beyond file"));
        }
        ranges.push((start, end, index));
    }
    ranges.sort_unstable_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return invalid(
                path,
                format!("blob {} overlaps blob {}", pair[0].2, pair[1].2),
            );
        }
    }
    Ok(())
}

fn checksum64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn invalid<T>(path: &Path, message: impl Into<String>) -> Result<T, PackageError> {
    Err(invalid_error(path, message))
}

fn invalid_error(path: &Path, message: impl Into<String>) -> PackageError {
    PackageError::Invalid {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn bincode_config()
-> bincode::config::Configuration<bincode::config::LittleEndian, bincode::config::Fixint> {
    bincode::config::standard().with_fixed_int_encoding()
}

/// Runtime `SurfaceQuery` projection of a validated authored package.
#[derive(Clone)]
pub struct PackageSurface {
    manifest: Arc<TerrainPackageManifest>,
    inner: BakedSurface,
}

impl PackageSurface {
    pub fn new(
        manifest: TerrainPackageManifest,
        surface: PlanetSurface,
        dynamic_state: DynamicSurfaceState,
    ) -> Self {
        Self {
            manifest: Arc::new(manifest),
            inner: BakedSurface::new(Arc::new(surface), dynamic_state),
        }
    }

    pub fn manifest(&self) -> &TerrainPackageManifest {
        &self.manifest
    }

    pub fn surface(&self) -> &Arc<PlanetSurface> {
        self.inner.surface()
    }
}

impl SurfaceQuery for PackageSurface {
    fn sample(&self, dir: Vec3, lod_m: f32) -> SurfaceSample {
        self.inner.sample(dir, lod_m)
    }

    fn sample_d(&self, dir: DVec3, lod_m: f32) -> SurfaceSample {
        self.inner.sample_d(dir, lod_m)
    }

    fn sample_height_m(&self, dir: Vec3, lod_m: f32) -> f32 {
        self.inner.sample_height_m(dir, lod_m)
    }

    fn radius_m(&self) -> f32 {
        self.inner.radius_m()
    }

    fn height_range_m(&self) -> f32 {
        self.inner.height_range_m()
    }

    fn prewarm(&self, region: Region, lod_m: f32) {
        self.inner.prewarm(region, lod_m)
    }
}
