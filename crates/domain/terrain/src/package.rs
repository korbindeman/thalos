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

use crate::cubemap::{Cubemap, CubemapFace, face_uv_to_dir};
use crate::generic_terrestrial_field::RuntimeTerrainDetail;
use crate::query::{SurfacePatch, SurfaceSample};
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
    load_static_package_inner(path, expected_body, Some(expected_content_key))
}

/// Decode and fully validate a shipped package without applying the bakery's
/// source-freshness check.
///
/// Release artifacts already select exact package bytes through their source
/// revision. Player startup validates schema, body identity, bounds, graph
/// structure, blob ranges, checksums, and decoding; it must not reject those
/// bytes because the compiler checkout used different text line endings or
/// another developer-only source signature. Tools and debug builds use
/// [`load_static_package`] to retain the stricter rebake signal.
pub fn load_static_package_artifact(
    path: &Path,
    expected_body: &str,
) -> Result<LoadedTerrainPackage, PackageError> {
    load_static_package_inner(path, expected_body, None)
}

fn load_static_package_inner(
    path: &Path,
    expected_body: &str,
    expected_content_key: Option<u64>,
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
    expected_content_key: Option<u64>,
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
    if let Some(expected_content_key) = expected_content_key
        && manifest.content_key != expected_content_key
    {
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
    refinement: Arc<PackageRefinementMetadata>,
}

impl PackageSurface {
    pub fn new(
        manifest: TerrainPackageManifest,
        surface: PlanetSurface,
        dynamic_state: DynamicSurfaceState,
    ) -> Self {
        let refinement = Arc::new(PackageRefinementMetadata::new(&manifest, &surface));
        Self {
            manifest: Arc::new(manifest),
            inner: BakedSurface::new(Arc::new(surface), dynamic_state),
            refinement,
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

    fn refinement_error_m(&self, patch: SurfacePatch, refined_spacing_m: f32) -> Option<f32> {
        self.refinement.error_m(patch, refined_spacing_m)
    }

    fn prewarm(&self, region: Region, lod_m: f32) {
        self.inner.prewarm(region, lod_m)
    }
}

/// Runtime index for the package's authored refinement bounds.
///
/// Package residuals do not describe geometry composed after the package is
/// decoded. Airless regolith has a small global amplitude we can bound; local
/// runtime craters are marked in a compact cubemap mask and deliberately make
/// the answer unknown for overlapping patches. Other runtime-detail strategies
/// and dynamic layers retain the selector's existing heuristic wholesale.
#[derive(Clone)]
struct PackageRefinementMetadata {
    body_radius_m: f32,
    base_resolution: u32,
    level_count: u8,
    node_errors_m: HashMap<PackageNodeAddress, f32>,
    runtime_error_m: f32,
    runtime_features: RuntimeFeatureMask,
    complete: bool,
}

impl PackageRefinementMetadata {
    fn new(manifest: &TerrainPackageManifest, surface: &PlanetSurface) -> Self {
        let mut runtime_features = RuntimeFeatureMask::default();
        for crater in surface.static_surface.craters.iter().filter(|crater| {
            crater.radius_m < surface.static_surface.cubemap_bake_threshold_m
                && crater.radius_m > 0.0
        }) {
            runtime_features.mark_cap(
                crater.center,
                (crater.influence_radius_m() / manifest.body_radius_m.max(1.0))
                    .min(std::f32::consts::PI),
            );
        }
        runtime_features.finish();

        let (runtime_error_m, supported_detail) = match surface.static_surface.runtime_detail {
            // A coarse interpolation and a refined sample can land at opposite
            // ends of the signed detail range, hence 2× amplitude.
            RuntimeTerrainDetail::AirlessRegolith(params) => (2.0 * params.amplitude_m.abs(), true),
            RuntimeTerrainDetail::LegacyHmf
            | RuntimeTerrainDetail::BasicContinental(_)
            | RuntimeTerrainDetail::OceanicContinental(_) => (0.0, false),
        };

        let node_errors_m = manifest
            .nodes
            .iter()
            .map(|node| {
                // A pruned residual is absent from the reconstructed runtime
                // surface, so refining cannot reveal it. For retained nodes,
                // predictor error bounds the authored residual and geometric
                // error covers its quantized reconstruction.
                let error = if node.blob_index.is_some() {
                    node.predictor_error_m + node.geometric_error_m
                } else {
                    0.0
                };
                (node.address, error)
            })
            .collect();

        Self {
            body_radius_m: manifest.body_radius_m,
            base_resolution: manifest.height_pyramid.base_resolution,
            level_count: manifest.height_pyramid.level_count,
            node_errors_m,
            runtime_error_m,
            runtime_features,
            complete: supported_detail && surface.dynamic_layers.is_empty(),
        }
    }

    fn error_m(&self, patch: SurfacePatch, refined_spacing_m: f32) -> Option<f32> {
        if !self.complete
            || !refined_spacing_m.is_finite()
            || refined_spacing_m <= 0.0
            || self.runtime_features.overlaps(patch)?
        {
            return None;
        }

        let face_resolution = self.body_radius_m * std::f32::consts::FRAC_PI_2 / refined_spacing_m;
        let target_lod = if face_resolution <= self.base_resolution as f32 {
            0
        } else {
            (face_resolution / self.base_resolution as f32)
                .log2()
                .ceil() as u8
        };

        // Residuals at and below the refined sampling scale can all move a
        // newly introduced vertex. Max within a level, sum across levels:
        // residuals compose additively down the hierarchy.
        let mut package_error_m = 0.0f32;
        for lod in target_lod.max(1)..self.level_count {
            package_error_m += self.max_error_at_lod(patch, lod)?;
        }
        Some(package_error_m + self.runtime_error_m)
    }

    fn max_error_at_lod(&self, patch: SurfacePatch, lod: u8) -> Option<f32> {
        if patch.face >= 6
            || patch.x >= (1u32.checked_shl(patch.level.into())?)
            || patch.y >= (1u32.checked_shl(patch.level.into())?)
        {
            return None;
        }

        if lod < patch.level {
            let shift = u32::from(patch.level - lod);
            return self
                .node_errors_m
                .get(&PackageNodeAddress::Cube {
                    face: patch.face,
                    lod,
                    x: patch.x >> shift,
                    y: patch.y >> shift,
                })
                .copied();
        }

        let scale = 1u32.checked_shl(u32::from(lod - patch.level))?;
        let x0 = patch.x.checked_mul(scale)?;
        let y0 = patch.y.checked_mul(scale)?;
        let mut max_error_m = 0.0f32;
        for y in y0..y0 + scale {
            for x in x0..x0 + scale {
                max_error_m =
                    max_error_m.max(*self.node_errors_m.get(&PackageNodeAddress::Cube {
                        face: patch.face,
                        lod,
                        x,
                        y,
                    })?);
            }
        }
        Some(max_error_m)
    }
}

/// Integral-image occupancy of runtime-only crater influence at a fixed cube
/// level. False positives only preserve the old selector; false negatives
/// would make package error claim authority over geometry it does not bound.
#[derive(Clone)]
struct RuntimeFeatureMask {
    cells: [Vec<u8>; 6],
    sums: [Vec<u32>; 6],
}

impl Default for RuntimeFeatureMask {
    fn default() -> Self {
        let side = 1usize << Self::LEVEL;
        Self {
            cells: std::array::from_fn(|_| vec![0; side * side]),
            sums: std::array::from_fn(|_| Vec::new()),
        }
    }
}

impl RuntimeFeatureMask {
    const LEVEL: u8 = 8;

    fn mark_cap(&mut self, center: Vec3, angular_radius: f32) {
        let center = center.normalize_or_zero();
        if center == Vec3::ZERO {
            return;
        }
        for face in 0..6u8 {
            self.mark_patch(face, 0, 0, 0, center, angular_radius.max(0.0));
        }
    }

    fn mark_patch(
        &mut self,
        face: u8,
        level: u8,
        x: u32,
        y: u32,
        cap_center: Vec3,
        cap_radius: f32,
    ) {
        let Some((center, radius)) = patch_cone(face, level, x, y) else {
            return;
        };
        let separation = center.dot(cap_center).clamp(-1.0, 1.0).acos();
        if separation > radius + cap_radius + 1.0e-5 {
            return;
        }
        if level == Self::LEVEL {
            let side = 1usize << Self::LEVEL;
            self.cells[face as usize][y as usize * side + x as usize] = 1;
            return;
        }
        let next = level + 1;
        for dy in 0..2 {
            for dx in 0..2 {
                self.mark_patch(face, next, x * 2 + dx, y * 2 + dy, cap_center, cap_radius);
            }
        }
    }

    fn finish(&mut self) {
        let side = 1usize << Self::LEVEL;
        let stride = side + 1;
        for face in 0..6 {
            let mut sum = vec![0u32; stride * stride];
            for y in 0..side {
                let mut row = 0u32;
                for x in 0..side {
                    row += u32::from(self.cells[face][y * side + x]);
                    sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + row;
                }
            }
            self.sums[face] = sum;
        }
        self.cells = std::array::from_fn(|_| Vec::new());
    }

    fn overlaps(&self, patch: SurfacePatch) -> Option<bool> {
        if patch.face >= 6 || self.sums[patch.face as usize].is_empty() {
            return None;
        }
        let patch_side = 1u32.checked_shl(patch.level.into())?;
        if patch.x >= patch_side || patch.y >= patch_side {
            return None;
        }
        let mask_side = 1u32 << Self::LEVEL;
        let (x0, y0, x1, y1) = if patch.level <= Self::LEVEL {
            let scale = 1u32 << (Self::LEVEL - patch.level);
            (
                patch.x * scale,
                patch.y * scale,
                (patch.x + 1) * scale,
                (patch.y + 1) * scale,
            )
        } else {
            let shift = patch.level - Self::LEVEL;
            let x = patch.x >> shift;
            let y = patch.y >> shift;
            (x, y, x + 1, y + 1)
        };
        debug_assert!(x1 <= mask_side && y1 <= mask_side);
        let stride = mask_side as usize + 1;
        let sum = &self.sums[patch.face as usize];
        let at = |x: u32, y: u32| sum[y as usize * stride + x as usize];
        Some(at(x1, y1) + at(x0, y0) > at(x1, y0) + at(x0, y1))
    }
}

fn patch_cone(face: u8, level: u8, x: u32, y: u32) -> Option<(Vec3, f32)> {
    let face = *CubemapFace::ALL.get(face as usize)?;
    let side = 1u32.checked_shl(level.into())? as f32;
    let u0 = x as f32 / side;
    let v0 = y as f32 / side;
    let u1 = (x + 1) as f32 / side;
    let v1 = (y + 1) as f32 / side;
    let center = face_uv_to_dir(face, (u0 + u1) * 0.5, (v0 + v1) * 0.5);
    // In unnormalised cube coordinates the patch half-diagonal is √2/side
    // and every face point has length >= 1. Therefore sin(theta) cannot exceed
    // that ratio. This is deliberately looser than sampling four corners: the
    // analytic bound covers the whole curved patch, so crater masks may retain
    // too much old refinement but can never miss runtime geometry.
    let sin_bound = std::f32::consts::SQRT_2 / side;
    let radius = if sin_bound >= 1.0 {
        std::f32::consts::PI
    } else {
        sin_bound.asin()
    };
    Some((center, radius))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_loader_validates_package_without_bakery_freshness() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/terrain_packages/Mira.bin");
        let artifact = load_static_package_artifact(&path, "Mira").unwrap();
        let wrong_key = artifact.manifest.content_key ^ 1;

        assert!(matches!(
            load_static_package(&path, "Mira", wrong_key),
            Err(PackageError::ContentKeyMismatch { .. })
        ));
    }

    #[test]
    fn refinement_error_sums_levels_and_takes_spatial_maxima() {
        let mut runtime_features = RuntimeFeatureMask::default();
        runtime_features.finish();
        let mut node_errors_m = HashMap::new();
        // Patch L1 (0,0) contains four L2 and sixteen L3 nodes. Every address
        // must exist, just as package validation guarantees at runtime.
        for lod in 2..=3u8 {
            let scale = 1u32 << (lod - 1);
            for y in 0..scale {
                for x in 0..scale {
                    node_errors_m.insert(PackageNodeAddress::Cube { face: 0, lod, x, y }, 1.0);
                }
            }
        }
        *node_errors_m
            .get_mut(&PackageNodeAddress::Cube {
                face: 0,
                lod: 2,
                x: 1,
                y: 0,
            })
            .unwrap() = 7.0;
        *node_errors_m
            .get_mut(&PackageNodeAddress::Cube {
                face: 0,
                lod: 3,
                x: 3,
                y: 2,
            })
            .unwrap() = 11.0;

        let radius_m = 100_000.0;
        let metadata = PackageRefinementMetadata {
            body_radius_m: radius_m,
            base_resolution: 32,
            level_count: 4,
            node_errors_m,
            runtime_error_m: 3.0,
            runtime_features,
            complete: true,
        };
        let target_resolution = 128.0;
        let spacing_m = radius_m * std::f32::consts::FRAC_PI_2 / target_resolution;
        let error = metadata
            .error_m(
                SurfacePatch {
                    face: 0,
                    level: 1,
                    x: 0,
                    y: 0,
                },
                spacing_m,
            )
            .unwrap();

        assert_eq!(error, 7.0 + 11.0 + 3.0);
    }

    #[test]
    fn runtime_feature_mask_invalidates_only_overlapping_patches() {
        let center = face_uv_to_dir(CubemapFace::PosX, 0.5, 0.5);
        let mut mask = RuntimeFeatureMask::default();
        mask.mark_cap(center, 0.02);
        mask.finish();

        assert_eq!(
            mask.overlaps(SurfacePatch {
                face: CubemapFace::PosX as u8,
                level: 8,
                x: 128,
                y: 128,
            }),
            Some(true)
        );
        assert_eq!(
            mask.overlaps(SurfacePatch {
                face: CubemapFace::NegX as u8,
                level: 8,
                x: 128,
                y: 128,
            }),
            Some(false)
        );
    }

    #[test]
    fn runtime_feature_mask_is_conservative_across_cube_seams() {
        let center = Vec3::new(1.0, 0.0, 1.0).normalize();
        let mut mask = RuntimeFeatureMask::default();
        mask.mark_cap(center, 0.03);
        mask.finish();

        let marked_faces = mask
            .sums
            .iter()
            .filter(|sum| sum.last().copied().unwrap_or(0) > 0)
            .count();
        assert!(marked_faces >= 2, "seam cap must mark both adjacent faces");
    }

    #[test]
    fn mira_package_has_bounded_regions_outside_runtime_craters() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/terrain_packages/Mira.bin");
        let loaded = load_static_package_artifact(&path, "Mira").unwrap();
        let surface = PlanetSurface {
            static_surface: loaded.static_surface,
            dynamic_layers: Default::default(),
            tectonics: None,
        };
        let metadata = PackageRefinementMetadata::new(&loaded.manifest, &surface);
        let mask_side = 1usize << RuntimeFeatureMask::LEVEL;
        let marked = metadata
            .runtime_features
            .sums
            .iter()
            .map(|sum| sum.last().copied().unwrap_or(0) as usize)
            .sum::<usize>();
        let total = 6 * mask_side * mask_side;

        assert!(
            marked > 0,
            "fixture no longer exercises runtime-crater fallback"
        );
        assert!(
            marked < total,
            "runtime craters cover every package patch, leaving no safe optimization surface"
        );
        assert!(metadata.complete);
    }

    #[test]
    fn flatten_decorator_preserves_refinement_metadata() {
        struct BoundedSurface;
        impl SurfaceQuery for BoundedSurface {
            fn sample(&self, _dir: Vec3, _lod_m: f32) -> SurfaceSample {
                SurfaceSample {
                    height_m: 0.0,
                    albedo_linear: Vec3::ZERO,
                    roughness: 1.0,
                    moisture: 0.0,
                }
            }

            fn radius_m(&self) -> f32 {
                1_000.0
            }

            fn height_range_m(&self) -> f32 {
                10.0
            }

            fn refinement_error_m(
                &self,
                _patch: SurfacePatch,
                _refined_spacing_m: f32,
            ) -> Option<f32> {
                Some(7.5)
            }
        }

        let surface =
            crate::FlattenedSurface::new(Arc::new(BoundedSurface), crate::flatten_handle());
        assert_eq!(
            surface.refinement_error_m(
                SurfacePatch {
                    face: 0,
                    level: 3,
                    x: 0,
                    y: 0,
                },
                10.0,
            ),
            Some(7.5)
        );
    }
}
