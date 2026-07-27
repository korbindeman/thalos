//! Ground-LOD tile caching — the memoization layer over the runtime terrain
//! generator.
//!
//! Thalos *owns* both sides of the tile contract: it generates the tiles and it
//! knows exactly what makes a tile's content change. That makes tiles safely
//! memoizable, which the upstream fork never exploited — before this, every
//! atlas eviction, residency-tier swap, and flatten rebuild re-synthesized tiles
//! from scratch (a cold surface site costs ~15 s of field evaluation).
//!
//! Two tiers wrap the synthesizing [`PipelineTileProvider`]:
//!
//! ```text
//! MemoryTileCacheProvider  ← survives terrain despawn/respawn (handle held here)
//!   └── DiskTileCacheProvider  ← survives the process
//!         └── PipelineTileProvider  ← actually evaluates the surface
//! ```
//!
//! # The cache key is the whole design
//!
//! Everything that can change a tile's *content* is folded into one `namespace`
//! ([`namespace_for`]): the generator fingerprint
//! ([`thalos_terrain::GENERATOR_VERSION`]), the body, and a content hash of the
//! body's current flatten regions. The attachment layout (tile resolution,
//! formats, mip count) is keyed separately by the cache itself.
//!
//! So invalidation is *structural*, not a pass we have to remember to run: a
//! flatten edit rebuilds the terrain, which rebuilds the provider with a **new
//! namespace**, and the previous tiles are simply never addressed again. Nothing
//! stale can be served, because a stale key is unreachable rather than wrong.
//! Disk space is reclaimed lazily by [`prune_tile_cache`] at boot.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use bevy::prelude::*;
use thalos_body_render::tiles::{
    CachedTileProvider, SurfaceTileCache, TerrainTileProvider, TileNamespaceFn,
};
use thalos_body_render::udlod::prelude::{
    AttachmentConfig, DiskTileCacheProvider, MemoryTileCacheProvider, NamespaceFn, SharedTileCache,
    TerrainConfig, TileProvider, prune_tile_cache, static_namespace,
};
use thalos_terrain::{FlattenHandle, FlattenRegion, GENERATOR_VERSION};
use thalos_world::BodyId;

/// RAM budget for retained tile payloads, across all bodies.
///
/// Sized in **bytes, not tiles**, because a near-tier tile is ~5 MB (512² across
/// four attachments, with mips) — a tile-count cap would silently turn into
/// gigabytes. This tier only has to smooth short-lived churn (a tier swap, a
/// flatten rebuild); the disk tier below it is what makes a revisit cheap, and it
/// costs no RAM at all.
const MEMORY_CACHE_BUDGET_BYTES: usize = 192 * 1024 * 1024;

/// Ceiling for the on-disk tile cache. Generous — tiles are only written for
/// places the player actually went, and the fixed-site scenarios (the spaceport)
/// re-address the same few hundred tiles every run.
const DISK_CACHE_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// RAM budget for retained tile-renderer payloads, across all bodies.
///
/// A tile-renderer payload is ~107 KB (67² halo grid × heights + albedo + bands),
/// so this retains ~1,800 tiles. It only has to smooth churn — the disk tier is
/// what makes a revisit or a second boot cheap, and the OS page cache backs that
/// at no cost in this process's footprint. Deliberately modest: the box has
/// already OOM'd once on tile residency (INC-20260725T012104Z) and the user runs
/// concurrent instances.
const SURFACE_MEMORY_CACHE_BUDGET_BYTES: usize = 192 * 1024 * 1024;

/// Ceiling for the tile-renderer's on-disk cache. Larger than udlod's because
/// its payloads are ~50× smaller per tile while a surface view wants thousands
/// of them (~790 MB for one settled spaceport view).
const SURFACE_DISK_CACHE_BUDGET_BYTES: u64 = 6 * 1024 * 1024 * 1024;

/// Per-body in-memory tile caches, plus the disk-cache location.
///
/// Held as a resource — *outside* the `TileAtlas` / `TileTerrainRoot` — precisely
/// so the retained payloads outlive the terrain entity. `TerrainRebuildRequest`
/// despawns and respawns a body's whole terrain to apply a flatten; without this
/// handle living out here, that would throw away every synthesized tile.
///
/// One registry serves **both** ground renderers. The tiers are separate (the
/// payloads are different types with different formats) but the namespace
/// discipline that makes them safe is shared, which is the part worth having one
/// home for.
#[derive(Resource)]
pub struct TileCacheRegistry {
    memory: HashMap<BodyId, SharedTileCache>,
    /// `None` disables disk caching (`THALOS_TILE_CACHE=0`).
    disk_root: Option<PathBuf>,
    /// Tile-renderer (`body_render::tiles`) equivalents of the two above.
    surface_memory: HashMap<BodyId, Arc<SurfaceTileCache>>,
    surface_disk_root: Option<PathBuf>,
}

impl Default for TileCacheRegistry {
    fn default() -> Self {
        // One resolution, so `THALOS_TILE_CACHE=0` logs once and can never end up
        // disabling one renderer's cache but not the other's.
        let disk_root = disk_cache_root();
        let surface_disk_root = disk_root.as_ref().map(|root| {
            root.parent()
                .map(|parent| parent.join("surfacetilecache"))
                .unwrap_or_else(|| PathBuf::from("user/surfacetilecache"))
        });
        Self {
            memory: HashMap::new(),
            disk_root,
            surface_memory: HashMap::new(),
            surface_disk_root,
        }
    }
}

impl TileCacheRegistry {
    /// The body's shared in-memory cache, created empty on first use.
    fn memory_cache(&mut self, body_id: BodyId) -> SharedTileCache {
        self.memory.entry(body_id).or_default().clone()
    }

    /// Wrap a synthesizing provider in the memory + disk cache tiers.
    ///
    /// `flatten` is the body's **live** flatten handle, not a snapshot. The
    /// namespace closure re-reads it on every tile request, because the tile
    /// provider itself reads it per tile *pixel*: a pad installed after the
    /// terrain entity spawned (which is exactly what `build_spaceport` does)
    /// changes what subsequent tiles bake. Freezing the flatten hash at spawn
    /// would file those flattened tiles under the un-flattened key and serve them
    /// as pristine terrain next session. Pass `None` for terrain with no flatten
    /// layer (the map view).
    pub fn wrap_provider(
        &mut self,
        body_id: BodyId,
        inner: Box<dyn TileProvider>,
        config: &TerrainConfig,
        flatten: Option<FlattenHandle>,
        surface_fingerprint: u64,
    ) -> Box<dyn TileProvider> {
        let namespace = self.namespace_fn(body_id, config, flatten, surface_fingerprint);

        let disked: Box<dyn TileProvider> = match &self.disk_root {
            Some(root) => Box::new(DiskTileCacheProvider::new(inner, root, namespace.clone())),
            None => inner,
        };

        // Budget the memory tier in tiles derived from this config's actual
        // payload size, so a small-tile distant body retains many more entries
        // than a near body for the same RAM.
        let per_tile = tile_payload_bytes(config).max(1);
        let capacity_tiles = MEMORY_CACHE_BUDGET_BYTES / per_tile;

        Box::new(MemoryTileCacheProvider::with_shared_cache(
            disked,
            namespace,
            capacity_tiles,
            self.memory_cache(body_id),
        ))
    }

    /// Wrap the **tile renderer's** provider in the memory + disk tiers.
    ///
    /// Same contract as [`Self::wrap_provider`]: `flatten` is the body's live
    /// handle, not a snapshot, because the provider samples through
    /// `FlattenedSurface` per tile *vertex* — a pad installed after the ground
    /// spawned changes what subsequent tiles bake, and freezing the key would
    /// file those flattened tiles under the un-flattened namespace.
    ///
    /// `radius_m` is not part of the namespace: it is validated by the payload
    /// itself on read (a rescaled model is a different body's tile, and the disk
    /// tier rejects it as a miss rather than trusting the key).
    pub fn wrap_tile_provider(
        &mut self,
        body_id: BodyId,
        inner: Arc<dyn TerrainTileProvider>,
        flatten: Option<FlattenHandle>,
        surface_fingerprint: u64,
    ) -> Arc<dyn TerrainTileProvider> {
        let namespace = self.tile_namespace_fn(body_id, flatten, surface_fingerprint);
        let memory = Arc::clone(self.surface_memory.entry(body_id).or_default());
        Arc::new(CachedTileProvider::new(
            inner,
            namespace,
            memory,
            SURFACE_MEMORY_CACHE_BUDGET_BYTES,
            self.surface_disk_root.clone(),
        ))
    }

    /// [`Self::namespace_fn`] for the tile renderer.
    ///
    /// Deliberately **not** shared with udlod's: that one folds in
    /// `config.model.scale()` and is consumed by a cache whose payloads have a
    /// different shape entirely. Sharing the hash would make the two renderers'
    /// tiles collide in name while differing in content — the one failure this
    /// design exists to make impossible. What *is* shared is the discipline and
    /// the flatten hash below.
    ///
    /// **If you add an input to tile synthesis, add it here.** A missing input
    /// means stale tiles get served as if fresh.
    fn tile_namespace_fn(
        &self,
        body_id: BodyId,
        flatten: Option<FlattenHandle>,
        surface_fingerprint: u64,
    ) -> TileNamespaceFn {
        let mut hasher = DefaultHasher::new();
        "tiles".hash(&mut hasher);
        GENERATOR_VERSION.hash(&mut hasher);
        surface_fingerprint.hash(&mut hasher);
        (body_id as u64).hash(&mut hasher);
        let static_hash = hasher.finish();

        let Some(flatten) = flatten else {
            return Arc::new(move || static_hash);
        };

        Arc::new(move || {
            let mut hasher = DefaultHasher::new();
            static_hash.hash(&mut hasher);
            match flatten.read() {
                Ok(regions) => hash_flatten_regions(&regions).hash(&mut hasher),
                // A poisoned lock collapsing to "no pads" would be a *wrong* key
                // rather than a missing one; the sentinel keeps a flattened tile
                // from ever landing under the un-flattened namespace.
                Err(_) => u64::MAX.hash(&mut hasher),
            }
            hasher.finish()
        })
    }

    /// Build the per-request namespace resolver.
    ///
    /// The static half (generator version, body, model scale) is hashed once; the
    /// live half (flatten regions) is re-read on every tile request, since it can
    /// change without the terrain entity being rebuilt.
    ///
    /// **If you add an input to tile synthesis, add it to one of these halves.** A
    /// missing input means stale tiles get served as if fresh — the one failure
    /// mode this whole design exists to make impossible.
    fn namespace_fn(
        &self,
        body_id: BodyId,
        config: &TerrainConfig,
        flatten: Option<FlattenHandle>,
        surface_fingerprint: u64,
    ) -> NamespaceFn {
        let mut hasher = DefaultHasher::new();
        GENERATOR_VERSION.hash(&mut hasher);
        surface_fingerprint.hash(&mut hasher);
        (body_id as u64).hash(&mut hasher);
        // The model's scale (body radius + height envelope) feeds the encoded
        // height range, so a rescaled model (the map view) is a different tile
        // even at the same coordinate.
        config.model.scale().to_bits().hash(&mut hasher);
        let static_hash = hasher.finish();

        let Some(flatten) = flatten else {
            return static_namespace(static_hash);
        };

        Arc::new(move || {
            let mut hasher = DefaultHasher::new();
            static_hash.hash(&mut hasher);
            match flatten.read() {
                Ok(regions) => hash_flatten_regions(&regions).hash(&mut hasher),
                // A poisoned lock would otherwise silently collapse to "no pads",
                // which is a *wrong* key rather than a missing one. Hash a
                // distinct sentinel so we can never write a flattened tile under
                // the un-flattened namespace.
                Err(_) => u64::MAX.hash(&mut hasher),
            }
            hasher.finish()
        })
    }
}

/// Content hash of a body's flatten regions. Order-independent: the vector order
/// follows structure spawn order, which is not a property of the terrain.
fn hash_flatten_regions(regions: &[FlattenRegion]) -> u64 {
    let mut digests: Vec<u64> = regions
        .iter()
        .map(|region| {
            let mut h = DefaultHasher::new();
            let f = &region.flatten;
            region.id.hash(&mut h);
            for v in [
                f.center_dir.x,
                f.center_dir.y,
                f.center_dir.z,
                f.tangent_along.x,
                f.tangent_along.y,
                f.tangent_along.z,
                f.tangent_across.x,
                f.tangent_across.y,
                f.tangent_across.z,
                f.half_along_m,
                f.half_across_m,
                f.offset_along_m,
                f.offset_across_m,
                f.ramp_m,
                f.elevation_m,
                f.radius_m,
            ] {
                v.to_bits().hash(&mut h);
            }
            h.finish()
        })
        .collect();
    digests.sort_unstable();

    let mut hasher = DefaultHasher::new();
    digests.hash(&mut hasher);
    hasher.finish()
}

/// Total CPU bytes one cached tile payload occupies for this config: every
/// attachment's full mip chain.
fn tile_payload_bytes(config: &TerrainConfig) -> usize {
    config.attachments.iter().map(attachment_bytes).sum()
}

fn attachment_bytes(cfg: &AttachmentConfig) -> usize {
    let pixel_size = match cfg.format {
        thalos_body_render::udlod::prelude::AttachmentFormat::Rgb8 => 3,
        thalos_body_render::udlod::prelude::AttachmentFormat::Rgba8 => 4,
        thalos_body_render::udlod::prelude::AttachmentFormat::R16 => 2,
        thalos_body_render::udlod::prelude::AttachmentFormat::R32Float => 4,
        thalos_body_render::udlod::prelude::AttachmentFormat::Rg16 => 4,
    };
    (0..cfg.mip_level_count)
        .map(|mip| {
            let size = (cfg.texture_size >> mip) as usize;
            size * size * pixel_size
        })
        .sum()
}

/// Where the on-disk tile cache lives, mirroring `settings`' split: project-local
/// and easy to nuke in debug, OS app-data in release. `THALOS_TILE_CACHE=0`
/// disables it (useful while iterating on the generator itself, where a bumped
/// [`GENERATOR_VERSION`] would otherwise be the only thing standing between you
/// and stale terrain).
fn disk_cache_root() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("THALOS_TILE_CACHE") {
        let v = v.trim().to_ascii_lowercase();
        if matches!(v.as_str(), "0" | "false" | "no" | "off") {
            info!("ground-LOD tile disk cache disabled (THALOS_TILE_CACHE={v})");
            return None;
        }
    }

    #[cfg(debug_assertions)]
    let root = PathBuf::from("user/tilecache");
    #[cfg(not(debug_assertions))]
    let root = bevy::platform::dirs::preferences_dir()
        .map(|dir| dir.join("thalos").join("tilecache"))
        .unwrap_or_else(|| PathBuf::from("user/tilecache"));

    Some(root)
}

/// Cap the disk cache at boot. Cheap (a shallow directory walk) and keeps the
/// cache from growing without bound across sessions and bodies.
fn prune_disk_cache(registry: Res<TileCacheRegistry>) {
    let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
    if let Some(root) = &registry.disk_root {
        let freed = prune_tile_cache(root, DISK_CACHE_BUDGET_BYTES);
        if freed > 0 {
            info!(
                "pruned {:.1} MB from the ground-LOD tile disk cache at {}",
                mib(freed),
                root.display(),
            );
        }
    }
    if let Some(root) = &registry.surface_disk_root {
        let freed =
            thalos_body_render::tiles::cache::prune_disk_cache(root, SURFACE_DISK_CACHE_BUDGET_BYTES);
        if freed > 0 {
            info!(
                "pruned {:.1} MB from the surface tile disk cache at {}",
                mib(freed),
                root.display(),
            );
        }
    }
}

pub struct TileCachePlugin;

impl Plugin for TileCachePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TileCacheRegistry>()
            .add_systems(Startup, prune_disk_cache);
    }
}
