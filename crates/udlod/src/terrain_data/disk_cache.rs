//! [`DiskTileCacheProvider`] — write-through, on-disk memoization of
//! CPU-produced tile payloads.
//!
//! This is the persistent replacement for upstream's `DiskTileProvider`, but
//! shaped for Thalos's *runtime* generator rather than a preprocessed asset
//! tree. It is a plain [`TileProvider`] wrapper: on request it looks for a
//! cached file, and on a miss forwards to the inner provider and writes the
//! result back.
//!
//! # Correct invalidation for free
//!
//! Tiles are keyed by [`CachedTileKey`] — `(namespace, coord, attachment-layout
//! hash)`. The caller folds everything that changes the generator's output into
//! the [`NamespaceFn`] — generator version, body, and the runtime-mutable terrain
//! flatten state (see `thalos_game`'s `tile_cache`). Because the namespace is
//! resolved **per request**, a tile is always stored under a key describing the
//! inputs it was actually baked from: edit a pad, and subsequent tiles land under
//! a different key while the pre-edit ones become unreachable. There is no
//! invalidation pass to forget to run, and nothing stale can be served — a stale
//! key is simply never addressed. Disk space is reclaimed lazily by
//! [`prune_tile_cache`] on size pressure.
//!
//! # On-disk format (`TLC1`)
//!
//! One file per tile, holding the **full mip chain** exactly as the atlas
//! uploads it (the provider generates mips before caching — see the
//! [`TileProvider`] mip contract). Layout:
//!
//! ```text
//! magic  "TLC1"            (4 bytes)
//! count  u32 LE            (attachment count)
//! per attachment:
//!   tag  u8                (0 None, 1 Rgba8, 2 R16, 3 R32Float, 4 Rg16)
//!   len  u64 LE            (byte length of this attachment's mip chain)
//!   data [len] bytes
//! ```
//!
//! Reads are validated against the requested [`AttachmentConfig`] (count, per-
//! attachment format tag, and exact expected mip-chain byte length); any
//! mismatch is treated as a cache miss and the tile is re-synthesized, so a
//! truncated or format-drifted file can never feed malformed data to the GPU.

use crate::{
    math::{TerrainModel, TileCoordinate},
    terrain_data::{
        tile_provider::{CachedTileKey, NamespaceFn, TileProvider},
        AttachmentConfig, AttachmentData, AttachmentFormat,
    },
};
use anyhow::{anyhow, Result};
use bevy::tasks::{AsyncComputeTaskPool, Task};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

const MAGIC: &[u8; 4] = b"TLC1";

/// Write-through disk cache wrapping any [`TileProvider`].
pub struct DiskTileCacheProvider {
    /// `Arc`, not `Box`, so the async task can own a handle to it — the disk read
    /// must not happen on the caller's thread. See [`Self::request_tile`].
    inner: Arc<dyn TileProvider>,
    root: PathBuf,
    /// Resolved per request, not frozen at construction — see [`NamespaceFn`].
    namespace: NamespaceFn,
}

impl DiskTileCacheProvider {
    /// Wrap `inner`, caching tiles under `<root>/<namespace hex>/`. The namespace
    /// must fold in everything that changes the generator's output (generator
    /// version, body, and any runtime-mutable input such as terrain flattening);
    /// tiles from different namespaces never collide.
    pub fn new(
        inner: Box<dyn TileProvider>,
        root: impl AsRef<Path>,
        namespace: NamespaceFn,
    ) -> Self {
        Self {
            inner: Arc::from(inner),
            root: root.as_ref().to_path_buf(),
            namespace,
        }
    }
}

impl TileProvider for DiskTileCacheProvider {
    fn request_tile(
        &self,
        coord: TileCoordinate,
        model: &TerrainModel,
        attachments: &[AttachmentConfig],
    ) -> Task<Result<Vec<AttachmentData>>> {
        // `request_tile` is called from `TileAtlas::update` on the **main thread**,
        // so the file IO below must live inside the spawned task: a tile payload is
        // multiple megabytes, and reading one synchronously here (up to
        // `max_concurrent_tile_loads` times a frame) would turn a background stall
        // into a frame hitch — the exact opposite of the point. Only the path/key
        // derivation runs on the caller's thread.
        let namespace = (self.namespace)();
        let key = CachedTileKey::new(namespace, coord, attachments);
        let path = tile_path(&self.root.join(format!("{namespace:016x}")), &key);
        let attachments = attachments.to_vec();
        let inner = Arc::clone(&self.inner);
        let model = model.clone();

        AsyncComputeTaskPool::get().spawn(async move {
            // `read_tile` validates the payload against the requested layout and
            // returns `None` on any mismatch (missing / truncated /
            // format-drifted), so a bad file degrades to a miss, never to a bad
            // upload.
            if let Some(datas) = read_tile(&path, &attachments) {
                return Ok(datas);
            }

            let result = inner.request_tile(coord, &model, &attachments).await;
            if let Ok(datas) = &result {
                // Best-effort write-back; a failed write just means the next visit
                // re-synthesizes. Never fail the tile on an IO error.
                if let Err(e) = write_tile(&path, &attachments, datas) {
                    bevy::log::trace!("tile disk-cache write failed for {path:?}: {e}");
                }
            }
            result
        })
    }

    fn subdivision_scale(&self, coord: TileCoordinate, model: &TerrainModel) -> f64 {
        self.inner.subdivision_scale(coord, model)
    }
}

fn tile_path(namespace_dir: &Path, key: &CachedTileKey) -> PathBuf {
    let c = key.coord;
    namespace_dir.join(format!(
        "{}_{}_{}_{}_{:016x}.tile",
        c.side, c.lod, c.x, c.y, key.attachment_layout_hash
    ))
}

fn format_tag(format: AttachmentFormat) -> u8 {
    match format {
        AttachmentFormat::Rgb8 => 5,
        AttachmentFormat::Rgba8 => 1,
        AttachmentFormat::R16 => 2,
        AttachmentFormat::R32Float => 3,
        AttachmentFormat::Rg16 => 4,
    }
}

fn tag_matches(format: AttachmentFormat, tag: u8) -> bool {
    format_tag(format) == tag
}

/// Decode one attachment's payload from its on-disk bytes.
///
/// Deliberately *not* `AttachmentData::from_bytes`: that goes through
/// `bytemuck::cast_slice`, which requires the byte slice to be aligned to the
/// target element (2 bytes for `u16`, 4 for `f32`). Our records are preceded by
/// a 1-byte tag + 8-byte length, so the payload lands at an arbitrary offset in
/// the file buffer and the cast would panic. Converting element-wise from
/// explicit little-endian chunks is alignment-agnostic and pins the on-disk byte
/// order, so a cache file is not a hostage to host endianness.
fn decode_attachment(bytes: &[u8], format: AttachmentFormat) -> Option<AttachmentData> {
    let data = match format {
        AttachmentFormat::Rgba8 => AttachmentData::Rgba8(
            bytes
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect(),
        ),
        AttachmentFormat::R16 => AttachmentData::R16(
            bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
        ),
        AttachmentFormat::R32Float => AttachmentData::R32Float(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        AttachmentFormat::Rg16 => AttachmentData::Rg16(
            bytes
                .chunks_exact(4)
                .map(|c| {
                    [
                        u16::from_le_bytes([c[0], c[1]]),
                        u16::from_le_bytes([c[2], c[3]]),
                    ]
                })
                .collect(),
        ),
        AttachmentFormat::Rgb8 => return None,
    };
    Some(data)
}

/// Encode one attachment's payload to its on-disk bytes (little-endian, matching
/// [`decode_attachment`]).
fn encode_attachment(data: &AttachmentData) -> Vec<u8> {
    match data {
        AttachmentData::Rgba8(v) => v.iter().flat_map(|p| *p).collect(),
        AttachmentData::R16(v) => v.iter().flat_map(|p| p.to_le_bytes()).collect(),
        AttachmentData::R32Float(v) => v.iter().flat_map(|p| p.to_le_bytes()).collect(),
        AttachmentData::Rg16(v) => v
            .iter()
            .flat_map(|p| {
                let [a, b] = *p;
                let a = a.to_le_bytes();
                let b = b.to_le_bytes();
                [a[0], a[1], b[0], b[1]]
            })
            .collect(),
        AttachmentData::None => Vec::new(),
    }
}

/// Total byte length of the full mip chain for one attachment.
fn mip_chain_bytes(cfg: &AttachmentConfig) -> usize {
    let pixel_size = cfg.format.pixel_size() as usize;
    let mut total = 0usize;
    for mip in 0..cfg.mip_level_count {
        let size = (cfg.texture_size >> mip) as usize;
        total += size * size * pixel_size;
    }
    total
}

fn write_tile(
    path: &Path,
    attachments: &[AttachmentConfig],
    datas: &[AttachmentData],
) -> Result<()> {
    if datas.len() != attachments.len() {
        return Err(anyhow!(
            "attachment count mismatch: {} data vs {} configs",
            datas.len(),
            attachments.len()
        ));
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&(attachments.len() as u32).to_le_bytes());
    for (cfg, data) in attachments.iter().zip(datas) {
        match data {
            AttachmentData::None => {
                buf.push(0);
                buf.extend_from_slice(&0u64.to_le_bytes());
            }
            _ => {
                let bytes = encode_attachment(data);
                buf.push(format_tag(cfg.format));
                buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                buf.extend_from_slice(&bytes);
            }
        }
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // Write to a unique temp file, then rename, so a concurrent reader never
    // observes a partially written tile. The temp name is derived from the
    // target plus the OS thread id (unique among the pool threads that could
    // race on the same tile).
    let tmp = path.with_extension(format!("tmp-{:?}", std::thread::current().id()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.flush()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Reads and validates a cached tile. Returns `None` on any mismatch — a
/// missing file, wrong magic, wrong attachment count, a format tag that doesn't
/// match the requested config, or a mip-chain length that doesn't match the
/// requested resolution — so the caller falls back to synthesis.
fn read_tile(path: &Path, attachments: &[AttachmentConfig]) -> Option<Vec<AttachmentData>> {
    let raw = fs::read(path).ok()?;
    let mut cursor = 0usize;

    let magic = raw.get(cursor..cursor + 4)?;
    if magic != MAGIC {
        return None;
    }
    cursor += 4;

    let count = u32::from_le_bytes(raw.get(cursor..cursor + 4)?.try_into().ok()?) as usize;
    cursor += 4;
    if count != attachments.len() {
        return None;
    }

    let mut datas = Vec::with_capacity(count);
    for cfg in attachments {
        let tag = *raw.get(cursor)?;
        cursor += 1;
        let len = u64::from_le_bytes(raw.get(cursor..cursor + 8)?.try_into().ok()?) as usize;
        cursor += 8;
        let bytes = raw.get(cursor..cursor + len)?;
        cursor += len;

        if tag == 0 {
            // Stored as absent; only valid where the config produced no data.
            datas.push(AttachmentData::None);
            continue;
        }
        // The stored payload must match the requested layout exactly, or a
        // stale/drifted file could feed the GPU a wrong-sized upload.
        if !tag_matches(cfg.format, tag) || len != mip_chain_bytes(cfg) {
            return None;
        }
        datas.push(decode_attachment(bytes, cfg.format)?);
    }

    // Reject trailing garbage — a sign of a format we don't understand.
    if cursor != raw.len() {
        return None;
    }
    Some(datas)
}

/// Caps the total size of a tile-cache `root` directory at `max_bytes` by
/// deleting whole-tile files oldest-first (by modification time). Best-effort:
/// IO errors are ignored. Intended to run once at boot — the fixed-site access
/// pattern means the cache is naturally bounded within a session, but this keeps
/// it bounded *across* sessions and bodies. Returns the number of bytes freed.
pub fn prune_tile_cache(root: impl AsRef<Path>, max_bytes: u64) -> u64 {
    let root = root.as_ref();
    let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
    let mut total: u64 = 0;

    // Two-level walk: <root>/<namespace>/<tile>. Avoids pulling in a recursive
    // walk dependency for a known-shallow layout.
    let Ok(namespaces) = fs::read_dir(root) else {
        return 0;
    };
    for ns in namespaces.flatten() {
        let Ok(tiles) = fs::read_dir(ns.path()) else {
            continue;
        };
        for tile in tiles.flatten() {
            let path = tile.path();
            if path.extension().and_then(|e| e.to_str()) != Some("tile") {
                continue;
            }
            let Ok(meta) = tile.metadata() else { continue };
            let len = meta.len();
            let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            total += len;
            files.push((path, len, mtime));
        }
    }

    if total <= max_bytes {
        return 0;
    }

    // Oldest first.
    files.sort_by_key(|(_, _, mtime)| *mtime);
    let mut freed = 0u64;
    for (path, len, _) in files {
        if total <= max_bytes {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            total -= len;
            freed += len;
        }
    }
    freed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain_data::{static_namespace, AttachmentData};
    use bevy::math::DVec3;
    use bevy::tasks::{futures_lite::future, AsyncComputeTaskPool, Task, TaskPool};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn init_pool() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
    }

    fn unique_dir(tag: &str) -> PathBuf {
        // Unique per test + process, no rand/clock dependency needed for
        // isolation because each test uses a distinct `tag`.
        std::env::temp_dir().join(format!(
            "thalos_udlod_disk_cache_{}_{}",
            std::process::id(),
            tag
        ))
    }

    fn cfg(name: &str, size: u32, mips: u32, format: AttachmentFormat) -> AttachmentConfig {
        AttachmentConfig {
            name: name.to_string(),
            texture_size: size,
            border_size: 2,
            mip_level_count: mips,
            format,
        }
    }

    /// Build a payload with a valid mip chain for the given configs (the shape
    /// the atlas expects: base + downsampled levels concatenated).
    fn synth(configs: &[AttachmentConfig], seed: u16) -> Vec<AttachmentData> {
        configs
            .iter()
            .map(|c| {
                let base = (c.texture_size * c.texture_size) as usize;
                let mut data = match c.format {
                    AttachmentFormat::R16 => AttachmentData::R16(vec![seed; base]),
                    AttachmentFormat::Rg16 => AttachmentData::Rg16(vec![[seed, seed / 2]; base]),
                    AttachmentFormat::R32Float => AttachmentData::R32Float(vec![0.25; base]),
                    AttachmentFormat::Rgba8 => {
                        AttachmentData::Rgba8(vec![[seed as u8, 1, 2, 255]; base])
                    }
                    AttachmentFormat::Rgb8 => AttachmentData::None,
                };
                data.generate_mipmaps(c.texture_size, c.mip_level_count);
                data
            })
            .collect()
    }

    struct CountingProvider {
        calls: Arc<AtomicUsize>,
        configs: Vec<AttachmentConfig>,
    }

    impl TileProvider for CountingProvider {
        fn request_tile(
            &self,
            _coord: TileCoordinate,
            _model: &TerrainModel,
            _attachments: &[AttachmentConfig],
        ) -> Task<Result<Vec<AttachmentData>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let datas = synth(&self.configs, 9);
            AsyncComputeTaskPool::get().spawn(async move { Ok(datas) })
        }
    }

    #[test]
    fn round_trips_multi_attachment_mip_chain() {
        let dir = unique_dir("roundtrip");
        let _ = fs::remove_dir_all(&dir);
        let configs = vec![
            cfg("height", 8, 4, AttachmentFormat::Rg16),
            cfg("albedo", 8, 4, AttachmentFormat::Rgba8),
            cfg("roughness", 8, 4, AttachmentFormat::R16),
        ];
        let key = CachedTileKey::new(7, TileCoordinate::new(1, 3, 2, 5), &configs);
        let path = tile_path(&dir.join("00..7"), &key);
        let datas = synth(&configs, 42);

        write_tile(&path, &configs, &datas).unwrap();
        let read = read_tile(&path, &configs).expect("valid tile should decode");

        assert_eq!(read.len(), datas.len());
        assert_eq!(read[0].as_rg16().unwrap(), datas[0].as_rg16().unwrap());
        assert_eq!(read[2].as_r16().unwrap(), datas[2].as_r16().unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn layout_mismatch_is_a_miss_not_a_crash() {
        let dir = unique_dir("mismatch");
        let _ = fs::remove_dir_all(&dir);
        let written = vec![cfg("height", 8, 4, AttachmentFormat::Rg16)];
        let key = CachedTileKey::new(1, TileCoordinate::new(0, 2, 1, 1), &written);
        let path = tile_path(&dir.join("ns"), &key);
        write_tile(&path, &written, &synth(&written, 3)).unwrap();

        // Reading with a different resolution must reject the stale payload.
        let requested = vec![cfg("height", 16, 4, AttachmentFormat::Rg16)];
        assert!(read_tile(&path, &requested).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    /// The property the whole cache design rests on: when a *runtime-mutable*
    /// synthesis input changes (in Thalos, a terrain-flatten pad), the namespace
    /// changes with it and the pre-edit tile is never served again.
    ///
    /// This is why the namespace is a [`NamespaceFn`] resolved per request rather
    /// than a `u64` frozen at construction — the provider here is built *once*,
    /// exactly as the game builds it once per terrain spawn, and the pad is
    /// "installed" afterwards.
    #[test]
    fn namespace_change_never_serves_the_pre_edit_tile() {
        init_pool();
        let dir = unique_dir("invalidate");
        let _ = fs::remove_dir_all(&dir);
        let configs = vec![cfg("height", 8, 4, AttachmentFormat::Rg16)];
        let calls = Arc::new(AtomicUsize::new(0));
        let model = TerrainModel::sphere(DVec3::ZERO, 1.0, 0.0, 1.0);
        let coord = TileCoordinate::new(0, 1, 0, 0);

        // A namespace that flips when the "pad" is installed, like the game's
        // closure re-hashing the live flatten handle.
        let pad_installed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&pad_installed);
        let namespace: NamespaceFn = Arc::new(move || {
            if flag.load(Ordering::SeqCst) {
                0xBBBB
            } else {
                0xAAAA
            }
        });

        let provider = DiskTileCacheProvider::new(
            Box::new(CountingProvider {
                calls: Arc::clone(&calls),
                configs: configs.clone(),
            }),
            &dir,
            namespace,
        );

        future::block_on(provider.request_tile(coord, &model, &configs)).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Warm: still the same terrain, so the cache answers.
        future::block_on(provider.request_tile(coord, &model, &configs)).unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "unchanged inputs should hit"
        );

        // The pad lands. The same tile coordinate now means different ground, and
        // must be re-synthesized rather than served from the pre-pad file.
        pad_installed.store(true, Ordering::SeqCst);
        future::block_on(provider.request_tile(coord, &model, &configs)).unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "a changed synthesis input must invalidate, not serve the stale tile"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn second_request_hits_disk_without_calling_inner() {
        init_pool();
        let dir = unique_dir("hit");
        let _ = fs::remove_dir_all(&dir);
        let configs = vec![cfg("height", 8, 4, AttachmentFormat::Rg16)];
        let calls = Arc::new(AtomicUsize::new(0));
        let model = TerrainModel::sphere(DVec3::ZERO, 1.0, 0.0, 1.0);
        let coord = TileCoordinate::new(0, 1, 0, 0);

        let provider = DiskTileCacheProvider::new(
            Box::new(CountingProvider {
                calls: Arc::clone(&calls),
                configs: configs.clone(),
            }),
            &dir,
            static_namespace(0xABCD),
        );

        let first = future::block_on(provider.request_tile(coord, &model, &configs)).unwrap();
        let second = future::block_on(provider.request_tile(coord, &model, &configs)).unwrap();

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "second request must hit disk"
        );
        assert_eq!(first[0].as_rg16().unwrap(), second[0].as_rg16().unwrap());
        let _ = fs::remove_dir_all(&dir);
    }
}
