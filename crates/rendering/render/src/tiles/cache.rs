//! Memoization tiers for the tile renderer's [`TerrainTileProvider`] seam.
//!
//! Field evaluation is the floor on a cold surface load. Measured on the dev box
//! (`cargo run --release -p thalos_terrain --example tile_stream_bench`), the
//! synthesis pipeline tops out at ~870 tiles/s and the pool shape is already
//! within ~20 % of the machine ceiling — so there is no scheduling trick left
//! that meaningfully shortens the wait. The only way past it is to **not
//! evaluate the field twice**, which is what this module buys.
//!
//! The legacy udlod ground has had exactly this since it was the default
//! renderer (`thalos_game`'s `rendering::tile_cache`, wrapping udlod's
//! `MemoryTileCacheProvider` / `DiskTileCacheProvider`). When `tiles` took over
//! as the default ground renderer the memoization did **not** come with it,
//! because those wrappers are typed to udlod's own `TileProvider` trait. This is
//! the same two-tier design re-stated for [`TerrainTileProvider`]:
//!
//! ```text
//! CachedTileProvider
//!   ├── memory tier  ← survives tile despawn / terrain rebuild (held in a resource)
//!   ├── disk tier    ← survives the process
//!   └── inner        ← actually evaluates the surface
//! ```
//!
//! # The cache key is the whole design
//!
//! Everything that can change a tile's *content* is folded into one `namespace`
//! resolved **per request** — generator version, body, and the body's live
//! flatten regions (see `thalos_game`'s `rendering::tile_cache`, which builds the
//! closure for both renderers). So invalidation is *structural*, never a pass
//! somebody has to remember to run: install a pad and subsequent tiles land under
//! a different key while the pre-edit ones become unreachable. Nothing stale can
//! be served because a stale key is never addressed.
//!
//! The payload *shape* (`TILE_RES`, `TILE_HALO`, the body radius that fixes each
//! key's sample spacing) is validated by the disk tier itself on read, so a
//! format drift or a truncated file degrades to a miss rather than to wrong
//! geometry.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use super::{SurfaceTile, TILE_HALO, TILE_RES, TerrainTileProvider, TileKey};

/// On-disk payload format. Bump when the byte layout below changes; the
/// generator's own output changes ride on `GENERATOR_VERSION` through the
/// caller's namespace instead.
const TILE_CACHE_VERSION: u32 = 1;

const MAGIC: &[u8; 4] = b"STC1";

/// Resolves the current cache namespace. Called **per request**, never frozen at
/// construction: the flatten layer it hashes can change without the terrain
/// entity being rebuilt (`build_spaceport` installs its pad after the ground has
/// already spawned), and a frozen key would file those flattened tiles under the
/// un-flattened namespace and serve them as pristine terrain next session.
pub type TileNamespaceFn = Arc<dyn Fn() -> u64 + Send + Sync>;

/// Bytes one cached payload occupies: heights + albedo + material bands over the
/// halo grid.
pub const fn tile_payload_bytes() -> usize {
    let n = (TILE_RES + 2 * TILE_HALO) * (TILE_RES + 2 * TILE_HALO);
    n * (4 + 12 + 8)
}

/// The provider-seam payload, without the parts that are pure functions of the
/// key (`TileKey`, `sample_spacing_m`) and so are never stored.
#[derive(Clone)]
struct Payload {
    heights_m: Arc<Vec<f32>>,
    albedo_linear: Arc<Vec<[f32; 3]>>,
    bands: Arc<Vec<[f32; 2]>>,
}

impl Payload {
    fn from_tile(tile: &SurfaceTile) -> Self {
        Self {
            heights_m: Arc::new(tile.heights_m.clone()),
            albedo_linear: Arc::new(tile.albedo_linear.clone()),
            bands: Arc::new(tile.bands.clone()),
        }
    }

    fn into_tile(self, key: TileKey, radius_m: f64) -> SurfaceTile {
        SurfaceTile {
            key,
            sample_spacing_m: key.sample_spacing_m(radius_m),
            heights_m: Arc::unwrap_or_clone(self.heights_m),
            albedo_linear: Arc::unwrap_or_clone(self.albedo_linear),
            bands: Arc::unwrap_or_clone(self.bands),
        }
    }
}

// --- memory tier ---------------------------------------------------------------

/// LRU-by-bytes retention of decoded payloads, shared across terrain rebuilds.
///
/// Held behind an `Arc` in a Bevy resource rather than inside the terrain entity
/// — precisely so a `TerrainRebuildRequest` (which despawns and respawns a body's
/// whole ground to apply a flatten) does not throw away everything synthesized so
/// far.
#[derive(Default)]
pub struct SurfaceTileCache {
    entries: Mutex<MemoryTier>,
}

#[derive(Default)]
struct MemoryTier {
    map: HashMap<(u64, TileKey), (Payload, u64)>,
    /// Monotonic use counter — the LRU stamp. Not wall-clock, so it stays
    /// deterministic and needs no `Time` access from the synthesis threads.
    clock: u64,
    bytes: usize,
}

impl SurfaceTileCache {
    fn get(&self, ns: u64, key: TileKey) -> Option<Payload> {
        let mut tier = self.entries.lock().ok()?;
        tier.clock += 1;
        let stamp = tier.clock;
        let (payload, used) = tier.map.get_mut(&(ns, key))?;
        *used = stamp;
        Some(payload.clone())
    }

    fn insert(&self, ns: u64, key: TileKey, payload: Payload, budget_bytes: usize) {
        let Ok(mut tier) = self.entries.lock() else {
            return;
        };
        tier.clock += 1;
        let stamp = tier.clock;
        if tier.map.insert((ns, key), (payload, stamp)).is_none() {
            tier.bytes += tile_payload_bytes();
        }
        if tier.bytes <= budget_bytes {
            return;
        }
        // Evict in one batch down to 90 % of budget: eviction is O(n log n) in
        // the entry count, and a cold stream inserts thousands of tiles, so
        // trimming one entry per overflow would sort the whole map per tile.
        let target = budget_bytes / 10 * 9;
        let mut stamps: Vec<((u64, TileKey), u64)> =
            tier.map.iter().map(|(k, (_, u))| (*k, *u)).collect();
        stamps.sort_unstable_by_key(|(_, used)| *used);
        for (k, _) in stamps {
            if tier.bytes <= target {
                break;
            }
            if tier.map.remove(&k).is_some() {
                tier.bytes = tier.bytes.saturating_sub(tile_payload_bytes());
            }
        }
    }
}

// --- disk tier -----------------------------------------------------------------

/// `<root>/<namespace hex>/<face>_<level>_<x>_<y>.tile`
fn tile_path(root: &Path, ns: u64, key: TileKey) -> PathBuf {
    root.join(format!("{ns:016x}")).join(format!(
        "{}_{}_{}_{}.tile",
        key.face, key.level, key.x, key.y
    ))
}

/// Encode a payload. Layout:
///
/// ```text
/// magic   "STC1"    (4 bytes)
/// version u32 LE
/// side    u32 LE    (halo grid side — validated against this build's TILE_RES)
/// radius  f64 LE    (the radius the samples were taken at)
/// heights [side²]   f32 LE
/// albedo  [side²·3] f32 LE
/// bands   [side²·2] f32 LE
/// ```
fn encode(payload: &Payload, radius_m: f64) -> Vec<u8> {
    let side = (TILE_RES + 2 * TILE_HALO) as u32;
    let mut out = Vec::with_capacity(16 + tile_payload_bytes());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&TILE_CACHE_VERSION.to_le_bytes());
    out.extend_from_slice(&side.to_le_bytes());
    out.extend_from_slice(&radius_m.to_le_bytes());
    out.extend_from_slice(bytemuck::cast_slice(payload.heights_m.as_slice()));
    out.extend_from_slice(bytemuck::cast_slice(payload.albedo_linear.as_slice()));
    out.extend_from_slice(bytemuck::cast_slice(payload.bands.as_slice()));
    out
}

/// Decode, validating every length and shape field. Any mismatch returns `None`
/// (a miss), so a truncated write or a format drift re-synthesizes instead of
/// feeding malformed geometry to the mesher.
fn decode(bytes: &[u8], radius_m: f64) -> Option<Payload> {
    let n = (TILE_RES + 2 * TILE_HALO) * (TILE_RES + 2 * TILE_HALO);
    if bytes.len() != 20 + tile_payload_bytes() {
        return None;
    }
    if &bytes[0..4] != MAGIC {
        return None;
    }
    if u32::from_le_bytes(bytes[4..8].try_into().ok()?) != TILE_CACHE_VERSION {
        return None;
    }
    if u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize != TILE_RES + 2 * TILE_HALO {
        return None;
    }
    // The key alone does not fix the sample spacing — the body radius does. A
    // rescaled model (the map view) must never read this body's tiles.
    if f64::from_le_bytes(bytes[12..20].try_into().ok()?).to_bits() != radius_m.to_bits() {
        return None;
    }

    let mut off = 20;
    let heights: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&bytes[off..off + n * 4]).to_vec();
    off += n * 4;
    let albedo: Vec<[f32; 3]> =
        bytemuck::cast_slice::<u8, [f32; 3]>(&bytes[off..off + n * 12]).to_vec();
    off += n * 12;
    let bands: Vec<[f32; 2]> =
        bytemuck::cast_slice::<u8, [f32; 2]>(&bytes[off..off + n * 8]).to_vec();

    Some(Payload {
        heights_m: Arc::new(heights),
        albedo_linear: Arc::new(albedo),
        bands: Arc::new(bands),
    })
}

/// Write through a temp file + rename, so a torn write (crash, concurrent
/// instance) can never leave a half-file under a live key. The reader validates
/// lengths anyway; this keeps it from having to.
///
/// Deliberately **no** `sync_data`: this is a cache, and the only thing a lost
/// write costs is one re-evaluated tile. An fsync per tile is not free — with
/// the synthesis pool writing thousands of ~107 KB files it is a measurable
/// share of the stream it exists to accelerate. The rename is atomic without it,
/// so the failure mode stays "miss", never "corrupt".
fn write_tile(root: &Path, ns: u64, key: TileKey, bytes: &[u8]) {
    let path = tile_path(root, ns, key);
    let Some(dir) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("tmp{:x}", std::process::id()));
    let write = (|| -> std::io::Result<()> {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(bytes)
    })();
    if write.is_err() || std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

// --- the wrapper ---------------------------------------------------------------

/// Wraps any [`TerrainTileProvider`] in the memory + disk memoization tiers.
///
/// Every tier lookup happens on the synthesis worker that is already off the main
/// thread (see `tiles::stream_tile_terrain`), so the disk read needs no async
/// hop of its own — unlike udlod's equivalent, which is called from the schedule.
pub struct CachedTileProvider {
    inner: Arc<dyn TerrainTileProvider>,
    namespace: TileNamespaceFn,
    memory: Arc<SurfaceTileCache>,
    memory_budget_bytes: usize,
    /// `None` disables the persistent tier (`THALOS_TILE_CACHE=0`).
    disk_root: Option<PathBuf>,
}

impl CachedTileProvider {
    pub fn new(
        inner: Arc<dyn TerrainTileProvider>,
        namespace: TileNamespaceFn,
        memory: Arc<SurfaceTileCache>,
        memory_budget_bytes: usize,
        disk_root: Option<PathBuf>,
    ) -> Self {
        Self {
            inner,
            namespace,
            memory,
            memory_budget_bytes,
            disk_root,
        }
    }
}

impl TerrainTileProvider for CachedTileProvider {
    fn height_range_m(&self) -> f32 {
        self.inner.height_range_m()
    }

    fn request(&self, key: TileKey, radius_m: f64) -> SurfaceTile {
        let ns = (self.namespace)();

        if let Some(payload) = self.memory.get(ns, key) {
            return payload.into_tile(key, radius_m);
        }

        if let Some(root) = &self.disk_root {
            if let Ok(bytes) = std::fs::read(tile_path(root, ns, key)) {
                if let Some(payload) = decode(&bytes, radius_m) {
                    self.memory
                        .insert(ns, key, payload.clone(), self.memory_budget_bytes);
                    return payload.into_tile(key, radius_m);
                }
            }
        }

        let tile = self.inner.request(key, radius_m);
        let payload = Payload::from_tile(&tile);
        if let Some(root) = &self.disk_root {
            write_tile(root, ns, key, &encode(&payload, radius_m));
        }
        self.memory.insert(ns, key, payload, self.memory_budget_bytes);
        tile
    }
}

/// Delete least-recently-used namespaces until the cache is under `budget_bytes`.
///
/// Whole namespace directories, not individual tiles: a namespace is a
/// *generation* of the world (generator version + flatten state), so the useful
/// unit of reclamation is "the run before the pad moved", never a scattering of
/// tiles from the current one.
pub fn prune_disk_cache(root: &Path, budget_bytes: u64) -> u64 {
    let Ok(dirs) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut namespaces: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
    let mut total = 0u64;
    for entry in dirs.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let mut bytes = 0u64;
        let mut newest = std::time::SystemTime::UNIX_EPOCH;
        let Ok(files) = std::fs::read_dir(&path) else {
            continue;
        };
        for file in files.flatten() {
            let Ok(meta) = file.metadata() else {
                continue;
            };
            bytes += meta.len();
            if let Ok(modified) = meta.modified() {
                newest = newest.max(modified);
            }
        }
        total += bytes;
        namespaces.push((path, bytes, newest));
    }
    if total <= budget_bytes {
        return 0;
    }
    namespaces.sort_by_key(|(_, _, newest)| *newest);
    let mut freed = 0u64;
    for (path, bytes, _) in namespaces {
        if total.saturating_sub(freed) <= budget_bytes {
            break;
        }
        if std::fs::remove_dir_all(&path).is_ok() {
            freed += bytes;
        }
    }
    freed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(seed: f32) -> Payload {
        let n = (TILE_RES + 2 * TILE_HALO) * (TILE_RES + 2 * TILE_HALO);
        Payload {
            heights_m: Arc::new((0..n).map(|i| seed + i as f32).collect()),
            albedo_linear: Arc::new(vec![[0.1, 0.2, 0.3]; n]),
            bands: Arc::new(vec![[12.0, 0.5]; n]),
        }
    }

    const R: f64 = 3_186_000.0;

    #[test]
    fn round_trips_through_the_disk_encoding() {
        let p = payload(7.0);
        let decoded = decode(&encode(&p, R), R).expect("decode");
        assert_eq!(*decoded.heights_m, *p.heights_m);
        assert_eq!(*decoded.albedo_linear, *p.albedo_linear);
        assert_eq!(*decoded.bands, *p.bands);
    }

    #[test]
    fn a_truncated_or_mismatched_file_is_a_miss_not_a_bad_tile() {
        let bytes = encode(&payload(1.0), R);
        assert!(decode(&bytes[..bytes.len() - 4], R).is_none(), "truncated");
        assert!(decode(&bytes, R * 2.0).is_none(), "different body radius");

        let mut wrong_version = bytes.clone();
        wrong_version[4] = 0xEE;
        assert!(decode(&wrong_version, R).is_none(), "format version");

        let mut wrong_magic = bytes.clone();
        wrong_magic[0] = b'X';
        assert!(decode(&wrong_magic, R).is_none(), "magic");
    }

    #[test]
    fn the_memory_tier_evicts_least_recently_used_and_honours_its_budget() {
        let cache = SurfaceTileCache::default();
        let budget = tile_payload_bytes() * 4;
        let key = |i: u32| TileKey {
            face: 0,
            level: 5,
            x: i,
            y: 0,
        };
        for i in 0..4 {
            cache.insert(0, key(i), payload(i as f32), budget);
        }
        // Touch 0 so it is no longer the least-recently-used, then overflow.
        assert!(cache.get(0, key(0)).is_some());
        cache.insert(0, key(4), payload(4.0), budget);

        assert!(cache.get(0, key(0)).is_some(), "recently used survives");
        assert!(cache.get(0, key(1)).is_none(), "LRU victim evicted");
        assert!(
            cache.entries.lock().unwrap().bytes <= budget,
            "budget honoured"
        );
    }

    #[test]
    fn namespaces_do_not_collide() {
        let cache = SurfaceTileCache::default();
        let key = TileKey {
            face: 1,
            level: 6,
            x: 3,
            y: 4,
        };
        cache.insert(1, key, payload(1.0), usize::MAX);
        assert!(cache.get(2, key).is_none(), "other namespace must miss");
        assert!(cache.get(1, key).is_some());
    }
}
