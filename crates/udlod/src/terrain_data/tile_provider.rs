//! The [`TileProvider`] trait — the seam between the renderer and the source
//! of tile data.
//!
//! A [`TileAtlas`](crate::terrain_data::tile_atlas::TileAtlas) holds a single
//! `Box<dyn TileProvider>` and calls [`TileProvider::request_tile`] whenever a
//! tile becomes resident. The provider returns a [`Task`] that resolves to the
//! base data for every attachment of that tile, in the same order as the
//! supplied [`AttachmentConfig`] slice. Mipmaps and uploads are handled
//! downstream.
//!
//! Runtime providers synthesize or fetch tile payloads on demand — see
//! [`TileCoordinate::stitched_pixel_coordinate`](crate::math::TileCoordinate::stitched_pixel_coordinate)
//! for the canonical pixel→position mapping that keeps tile borders aligned
//! with neighbours.
//!
//! Tile latency is tolerated by the renderer via parent-LOD fallback. Failed
//! requests leave the tile in a perpetual loading state — providers are
//! expected to surface real errors only for unrecoverable situations.

use crate::{
    math::{TerrainModel, TileCoordinate},
    terrain_data::{AttachmentConfig, AttachmentData, AttachmentFormat},
};
use anyhow::Result;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex, MutexGuard},
};

/// Source of tile attachment data for a
/// [`TileAtlas`](crate::terrain_data::tile_atlas::TileAtlas).
///
/// Implementations are held behind `Box<dyn TileProvider>`, so different
/// terrains in the same world can use different providers (procedurally
/// synthesized, cache-backed, test stubs, etc.).
///
/// # Border alignment
///
/// Each tile texture has a border that overlaps adjacent tiles. For seamless
/// filtering the value at a shared texel must be **bit-identical** between
/// the two tiles that contain it. Providers must therefore evaluate their
/// data source as a pure function of position: same input → same output,
/// regardless of tile evaluation order. Computing texel positions via
/// [`TileCoordinate::stitched_pixel_coordinate`](crate::math::TileCoordinate::stitched_pixel_coordinate)
/// — which mirrors the offline stitch pass for border texels — is the
/// easiest way to satisfy this.
pub trait TileProvider: Send + Sync {
    /// Produce the base attachment data for `coord`.
    ///
    /// The returned task must resolve to one [`AttachmentData`] per entry in
    /// `attachments`, in the same order, in the layout described by
    /// [`AttachmentFormat`](crate::terrain_data::AttachmentFormat).
    ///
    /// **Mips are the provider's job.** Each payload must carry its **full mip
    /// chain** — call
    /// [`AttachmentData::generate_mipmaps`](crate::terrain_data::AttachmentData::generate_mipmaps)
    /// once, inside this task, before returning. The atlas does not regenerate
    /// them. This keeps the (per-tile, non-trivial) mip filtering on whatever
    /// pool the provider synthesizes on rather than on the main thread in
    /// `TileAtlas::update`, and it means the cache wrappers store fully-mipped
    /// payloads, so a cache hit costs neither synthesis nor mip filtering.
    ///
    /// Attachments may declare **different resolutions**; each payload is sized
    /// by its own `texture_size`, not the tile's largest.
    fn request_tile(
        &self,
        coord: TileCoordinate,
        model: &TerrainModel,
        attachments: &[AttachmentConfig],
    ) -> Task<Result<Vec<AttachmentData>>>;

    /// Per-tile subdivision-threshold scale in `(0, 1]` — the screen-space-error
    /// seam.
    ///
    /// The tile tree's refinement is pure view-distance by default: a tile is
    /// split when the view is nearer than `subdivision_distance / 2^lod`. That
    /// spends the same tile budget on glass-flat terrain as on a mountain range.
    /// A provider that owns its generator can return a value `< 1` for tiles
    /// whose footprint carries little relief, shrinking the split threshold so
    /// flat regions refine less (fewer tiles synthesized, drawn, and resident).
    ///
    /// **Contract:** the returned value must be **≤ 1** (never request *more*
    /// detail than distance alone would), cheap (called per candidate tile each
    /// frame — memoize any real evaluation), and monotone-ish in relief so LOD
    /// doesn't flicker. The default `1.0` reproduces stock UDLOD behaviour.
    fn subdivision_scale(&self, _coord: TileCoordinate, _model: &TerrainModel) -> f64 {
        1.0
    }
}

/// Resolves the cache namespace **at request time**.
///
/// Not a plain `u64` fixed at construction, and that is load-bearing: some
/// inputs to tile synthesis are themselves mutable at runtime behind a shared
/// handle (Thalos's terrain-flatten regions are read per tile *pixel*, so a pad
/// installed after the terrain entity spawned still changes what subsequent
/// tiles bake). Freezing the namespace when the provider is built would let
/// those tiles be cached under the pre-edit key and served later as if pristine
/// — the one failure mode caching must not have. Evaluating per request means
/// the key always describes the inputs the tile was actually baked from.
///
/// Must be cheap: it runs on every tile request, hit or miss.
pub type NamespaceFn = Arc<dyn Fn() -> u64 + Send + Sync>;

/// A namespace that never changes (tests, and producers with no mutable inputs).
pub fn static_namespace(namespace: u64) -> NamespaceFn {
    Arc::new(move || namespace)
}

/// A shareable, reference-counted handle to a [`MemoryTileCache`].
///
/// The cache is keyed by `namespace` (see [`CachedTileKey`]), so **one** handle
/// can back **many** [`MemoryTileCacheProvider`]s with different namespaces —
/// e.g. every reconstruction of a body's terrain provider after a
/// despawn/respawn. Holding this handle outside the `TileAtlas` (which is
/// dropped on respawn) is what lets a body's synthesized tiles survive
/// flatten-invalidation and residency-tier swaps instead of being re-baked from
/// scratch each time.
#[derive(Clone)]
pub struct SharedTileCache(Arc<Mutex<MemoryTileCache>>);

impl SharedTileCache {
    /// A fresh, empty shared cache.
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(MemoryTileCache::default())))
    }

    /// Number of tile payloads currently retained.
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Returns `true` when no tile payloads are retained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops every retained tile payload across all providers sharing this cache.
    pub fn clear(&self) {
        self.lock().entries.clear();
    }

    fn lock(&self) -> MutexGuard<'_, MemoryTileCache> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Default for SharedTileCache {
    fn default() -> Self {
        Self::new()
    }
}

/// In-memory frecency cache for CPU-produced tile payloads.
///
/// This is a provider wrapper, not part of [`TileAtlas`](crate::terrain_data::tile_atlas::TileAtlas).
/// `TileAtlas` owns current residency; this wrapper owns reuse across
/// visibility churn or repeated visits. The namespace is caller supplied so
/// Thalos can key by body/source hash without making [`TerrainModel`] pretend
/// to be a stable cache key.
///
/// Payloads are stored **with their full mip chain** (the provider generates
/// mips before this wrapper caches them — see the [`TileProvider`] contract), so
/// a cache hit skips both synthesis and mip filtering.
pub struct MemoryTileCacheProvider {
    /// `Arc`, not `Box`, so the async task can own a handle to it — see
    /// [`Self::request_tile`], which must not touch the cache on the main thread.
    inner: Arc<dyn TileProvider>,
    namespace: NamespaceFn,
    capacity_tiles: usize,
    cache: SharedTileCache,
}

impl MemoryTileCacheProvider {
    /// Wrap `inner` with a private cache that stores at most `capacity_tiles`
    /// tile payloads. A capacity of zero disables caching and forwards requests.
    pub fn new(inner: Box<dyn TileProvider>, namespace: NamespaceFn, capacity_tiles: usize) -> Self {
        Self::with_shared_cache(inner, namespace, capacity_tiles, SharedTileCache::new())
    }

    /// Wrap `inner` with a caller-owned [`SharedTileCache`], so the retained
    /// payloads outlive this provider (and the `TileAtlas` holding it). Use this
    /// for terrain that is despawned/respawned — hold the handle in a resource
    /// keyed by body and pass a fresh provider each spawn.
    pub fn with_shared_cache(
        inner: Box<dyn TileProvider>,
        namespace: NamespaceFn,
        capacity_tiles: usize,
        cache: SharedTileCache,
    ) -> Self {
        Self {
            inner: Arc::from(inner),
            namespace,
            capacity_tiles,
            cache,
        }
    }

    /// Number of tile payloads currently retained by the backing cache.
    pub fn len(&self) -> usize {
        self.lock_cache().entries.len()
    }

    /// Returns `true` when no tile payloads are retained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops every retained tile payload.
    pub fn clear(&self) {
        self.lock_cache().entries.clear();
    }

    fn lock_cache(&self) -> MutexGuard<'_, MemoryTileCache> {
        self.cache.lock()
    }
}

impl TileProvider for MemoryTileCacheProvider {
    fn request_tile(
        &self,
        coord: TileCoordinate,
        model: &TerrainModel,
        attachments: &[AttachmentConfig],
    ) -> Task<Result<Vec<AttachmentData>>> {
        if self.capacity_tiles == 0 {
            return self.inner.request_tile(coord, model, attachments);
        }

        // Everything below happens **inside** the spawned task, on purpose.
        // `request_tile` is called from `TileAtlas::update` on the main thread, and
        // a cache hit means cloning a multi-megabyte payload — doing that here
        // would trade a background stall for a frame hitch. Only the (cheap)
        // namespace hash runs on the caller's thread.
        let key = CachedTileKey::new((self.namespace)(), coord, attachments);
        let cache = self.cache.clone();
        let capacity_tiles = self.capacity_tiles;
        let inner = Arc::clone(&self.inner);
        let model = model.clone();
        let attachments = attachments.to_vec();

        AsyncComputeTaskPool::get().spawn(async move {
            if let Some(data) = cache.lock().get(key) {
                return Ok(data);
            }

            let result = inner.request_tile(coord, &model, &attachments).await;
            if let Ok(data) = &result {
                cache.lock().insert(key, data.clone(), capacity_tiles);
            }
            result
        })
    }

    fn subdivision_scale(&self, coord: TileCoordinate, model: &TerrainModel) -> f64 {
        // Caching does not change the geometry the underlying generator
        // produces, so the relief-based refinement hint is the inner provider's.
        self.inner.subdivision_scale(coord, model)
    }
}

/// Identity of a cached tile payload: the source `namespace` (body + generator
/// version + flatten revision, folded by the caller), the tile coordinate, and a
/// hash of the requested attachment layout. Shared with the disk cache so both
/// tiers key tiles identically.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CachedTileKey {
    pub(crate) namespace: u64,
    pub(crate) coord: TileCoordinate,
    pub(crate) attachment_layout_hash: u64,
}

impl CachedTileKey {
    pub(crate) fn new(
        namespace: u64,
        coord: TileCoordinate,
        attachments: &[AttachmentConfig],
    ) -> Self {
        Self {
            namespace,
            coord,
            attachment_layout_hash: hash_attachment_layout(attachments),
        }
    }
}

#[derive(Default)]
struct MemoryTileCache {
    entries: std::collections::HashMap<CachedTileKey, MemoryTileCacheEntry>,
    clock: u64,
}

impl MemoryTileCache {
    fn get(&mut self, key: CachedTileKey) -> Option<Vec<AttachmentData>> {
        let now = self.tick();
        let entry = self.entries.get_mut(&key)?;
        entry.hits = entry.hits.saturating_add(1);
        entry.last_access = now;
        Some(entry.data.clone())
    }

    fn insert(&mut self, key: CachedTileKey, data: Vec<AttachmentData>, capacity_tiles: usize) {
        if capacity_tiles == 0 {
            return;
        }

        let now = self.tick();
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.data = data;
            entry.hits = entry.hits.saturating_add(1).max(1);
            entry.last_access = now;
            return;
        }

        while self.entries.len() >= capacity_tiles {
            let Some(victim) = self.victim(now) else {
                break;
            };
            self.entries.remove(&victim);
        }

        self.entries.insert(
            key,
            MemoryTileCacheEntry {
                data,
                hits: 1,
                last_access: now,
            },
        );
    }

    fn victim(&self, now: u64) -> Option<CachedTileKey> {
        self.entries
            .iter()
            .min_by_key(|(_, entry)| entry.frecency_score(now))
            .map(|(key, _)| *key)
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }
}

struct MemoryTileCacheEntry {
    data: Vec<AttachmentData>,
    hits: u64,
    last_access: u64,
}

impl MemoryTileCacheEntry {
    fn frecency_score(&self, now: u64) -> u64 {
        let age = now.saturating_sub(self.last_access);
        let recency = 1024 / (age + 1);
        self.hits.saturating_mul(1024).saturating_add(recency)
    }
}

pub(crate) fn hash_attachment_layout(attachments: &[AttachmentConfig]) -> u64 {
    let mut hasher = DefaultHasher::new();
    attachments.len().hash(&mut hasher);
    for attachment in attachments {
        attachment.name.hash(&mut hasher);
        attachment.texture_size.hash(&mut hasher);
        attachment.border_size.hash(&mut hasher);
        attachment.mip_level_count.hash(&mut hasher);
        attachment_format_hash(attachment.format).hash(&mut hasher);
    }
    hasher.finish()
}

fn attachment_format_hash(format: AttachmentFormat) -> u8 {
    match format {
        AttachmentFormat::Rgb8 => 0,
        AttachmentFormat::Rgba8 => 1,
        AttachmentFormat::R16 => 2,
        AttachmentFormat::R32Float => 3,
        AttachmentFormat::Rg16 => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::DVec3;
    use bevy::tasks::futures_lite::future;
    use bevy::tasks::TaskPool;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingProvider {
        calls: Arc<AtomicUsize>,
        value: u16,
    }

    impl TileProvider for CountingProvider {
        fn request_tile(
            &self,
            _coord: TileCoordinate,
            _model: &TerrainModel,
            attachments: &[AttachmentConfig],
        ) -> Task<Result<Vec<AttachmentData>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let configs = attachments.to_vec();
            let value = self.value;

            AsyncComputeTaskPool::get().spawn(async move {
                Ok(configs
                    .iter()
                    .map(|cfg| {
                        let count = (cfg.texture_size * cfg.texture_size) as usize;
                        match cfg.format {
                            AttachmentFormat::R16 => AttachmentData::R16(vec![value; count]),
                            AttachmentFormat::R32Float => {
                                AttachmentData::R32Float(vec![0.5; count])
                            }
                            AttachmentFormat::Rgba8 => {
                                AttachmentData::Rgba8(vec![[value as u8, 0, 0, 255]; count])
                            }
                            AttachmentFormat::Rg16 => {
                                AttachmentData::Rg16(vec![[value, value]; count])
                            }
                            AttachmentFormat::Rgb8 => AttachmentData::None,
                        }
                    })
                    .collect())
            })
        }
    }

    fn r16_attachment(name: &str, texture_size: u32) -> AttachmentConfig {
        AttachmentConfig {
            name: name.to_string(),
            texture_size,
            border_size: 1,
            mip_level_count: 1,
            format: AttachmentFormat::R16,
        }
    }

    fn init_task_pool() {
        AsyncComputeTaskPool::get_or_init(TaskPool::default);
    }

    #[test]
    fn memory_cache_returns_hits_without_calling_inner_provider() {
        init_task_pool();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = MemoryTileCacheProvider::new(
            Box::new(CountingProvider {
                calls: Arc::clone(&calls),
                value: 7,
            }),
            static_namespace(42),
            8,
        );
        let model = TerrainModel::sphere(DVec3::ZERO, 1.0, 0.0, 1.0);
        let attachments = vec![r16_attachment("height", 4)];
        let coord = TileCoordinate::new(0, 1, 0, 0);

        let first = future::block_on(provider.request_tile(coord, &model, &attachments)).unwrap();
        let second = future::block_on(provider.request_tile(coord, &model, &attachments)).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.len(), 1);
        assert_eq!(first[0].as_r16().unwrap(), second[0].as_r16().unwrap());
    }

    #[test]
    fn memory_cache_keys_include_attachment_layout() {
        init_task_pool();
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = MemoryTileCacheProvider::new(
            Box::new(CountingProvider {
                calls: Arc::clone(&calls),
                value: 11,
            }),
            static_namespace(42),
            8,
        );
        let model = TerrainModel::sphere(DVec3::ZERO, 1.0, 0.0, 1.0);
        let coord = TileCoordinate::new(0, 1, 0, 0);

        let small = vec![r16_attachment("height", 4)];
        let large = vec![r16_attachment("height", 8)];

        future::block_on(provider.request_tile(coord, &model, &small)).unwrap();
        future::block_on(provider.request_tile(coord, &model, &large)).unwrap();
        future::block_on(provider.request_tile(coord, &model, &small)).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(provider.len(), 2);
    }
}
