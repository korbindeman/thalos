//! Spawn ground-LOD terrain entities for procedural bodies.
//!
//! Called from `generation::finalize_planet_generation` once a body's
//! `PlanetSurface` task resolves. The terrain entity is parented to the
//! body's real-space `Grid` so it inherits orbital + rotational motion
//! automatically (`thalos_udlod`'s `big_space` integration handles
//! surface-scale precision via its Taylor-series approximation; we don't
//! nest any additional cells).
//!
//! The same `PlanetSurface` that drives the impostor billboard's
//! `PlanetMaterial` is shared with the [`PipelineTileProvider`] via `Arc`,
//! so there is exactly one copy of the heavy cubemap data per body.

use std::sync::Arc;

use bevy::camera::visibility::RenderLayers;
use bevy::light::NotShadowCaster;
use bevy::math::DVec3;
use bevy::prelude::*;
use big_space::grid::Grid;
use big_space::prelude::CellCoord;
use thalos_body_render::udlod::math::TileCoordinate;
use thalos_body_render::udlod::prelude::*;
use thalos_body_render::{AU_M, AtmosphereBlock, LIGHT_AT_1AU, SceneLighting};
use thalos_body_render::{
    BodySkyExtra, BodySkyMaterial, BodyTerrainDebug, BodyTerrainExtras, BodyTerrainMaterial,
    CASCADE_COUNT, FlattenBlock, GpuAtlasHeightMirrorComponent, GpuAtlasMirrorHandle,
    MAX_FLATTEN_REGIONS, PipelineTileProvider, SyntheticTerrainMode, SyntheticTileProvider,
    TerrainShadingStyle, ocean_wave_frame, project_ocean_spectrum, rendered_height_range,
};
use thalos_physics_canonical::canonical::AuthorityMode;
use thalos_physics_canonical::types::VesselKind;
use thalos_terrain::{FlattenHandle, FlattenedSurface, SurfaceQuery, flatten_handle};
use thalos_world::{BodyDefinition, BodyId};

use std::collections::HashMap;

use super::SCREEN_MARKER_RADIUS;
use super::ocean::BodyOcean;
use super::tile_cache::TileCacheRegistry;
use crate::camera::ShipCamera;
use crate::coords::SHIP_LAYER;
use crate::player_controller::{EvaMode, PlayerControllerState};

use super::real_space::REAL_SPACE_CELL_SIZE_M;
use super::types::{CameraExposure, RealSpaceBody, SimulationState, SolarSystemState};

/// Per-body shared [`FlattenHandle`]s for local terrain flattening (e.g. the
/// runway pad). The handle is created once per body and reused across terrain
/// despawn/respawn churn, so a flatten region set by gameplay survives a
/// residency reload: the next provider wraps the same handle and re-bakes the
/// affected tiles flattened. Empty by default (no flattening).
///
/// **Writers:** [`crate::runway`] sets a body's region. **Readers:**
/// [`spawn_body_terrain`] wraps the tile-provider surface with it.
#[derive(Resource, Default)]
pub struct TerrainFlattenRegistry {
    handles: HashMap<BodyId, FlattenHandle>,
}

impl TerrainFlattenRegistry {
    /// Fetch the body's flatten handle, creating an empty one on first use so a
    /// writer and the terrain provider share the same object regardless of which
    /// runs first.
    pub fn handle(&mut self, body_id: BodyId) -> FlattenHandle {
        self.handles
            .entry(body_id)
            .or_insert_with(flatten_handle)
            .clone()
    }

    /// Read-only lookup for consumers that must not create a handle (e.g. the
    /// per-frame material driver mirroring pads to the GPU). `None` simply
    /// means "no flattening on this body yet".
    pub fn get(&self, body_id: BodyId) -> Option<&FlattenHandle> {
        self.handles.get(&body_id)
    }
}

/// LOD depth for body terrains. 16 LODs over a Mira-scale body
/// (~10.7 Mm circumference) gives ~1.3 m at the deepest tile texel — well
/// past where the synthesized cubemap data carries meaningful detail. Stage
/// 4+ may pin this to per-body values once the v2 backlog provides true
/// arbitrary-resolution synthesis.
const LOD_COUNT: u32 = 16;

/// Tile attachment resolution. 512 is the upstream default and matches what
/// the playground example uses; thalos_udlod mandates power-of-two for
/// mipmap generation. Larger sizes trade tile latency for fewer LOD levels.
const TILE_TEXTURE_SIZE: u32 = 512;
const TILE_BORDER_SIZE: u32 = 2;
const TILE_MIP_LEVELS: u32 = 4;
/// CPU tile synthesis concurrency for Thalos' runtime terrain provider.
///
/// Each tile is now evaluated with rayon across all cores (see
/// `PipelineTileProvider`), so the synthesis pool's worker count is the real
/// concurrency limiter; this matches it (`tile_synthesis_pool` = 4 threads).
/// The admission queue is kept deeper than the slot count so the
/// nearest-view-first admission in `TileAtlas::update` has a wide candidate set
/// to draw the immediate tiles from on a cold view.
const TILE_LOAD_SLOTS: u32 = 4;
const TILE_LOAD_QUEUE_SIZE: u32 = TILE_LOAD_SLOTS * 8;

/// Resident tiles per body. Upstream defaults to 1024, which is tuned for
/// one giant terrain. With four bodies the atlas memory adds up quickly
/// (one slot at 512² is ~750 KB across height + albedo).
///
/// Sized to the focused-body request set with headroom for the 2:1 balance
/// pass in `thalos_udlod::TileTree::balance_lod_gaps`. The balance pass adds
/// a stair-step ring of forced cross-face requests at every active cube-face
/// seam — under the worst-case viewpoint (player near a cube corner, three
/// seams in near-field) this adds ~50–80 tiles on top of the in-window set.
/// 384 leaves room without ballooning per-body memory.
const ATLAS_SIZE: u32 = 384;

// ── Distant tier ─────────────────────────────────────────────────────────
//
// A second, deliberately tiny atlas configuration for bodies that are only
// *visible* (bigger than the icon dot in the ship view) but not gameplay-
// relevant — not the dominant SOI body and not a predicted encounter. The
// near tier above is ~1.8 GB/body (`ATLAS_SIZE` × 512² × 4 attachments), so
// "every visible planet at full resolution" does not scale; the distant tier
// trades detail for footprint so a dozen bodies can stay resident at once.
//
// Memory ≈ `ATLAS_SIZE_DISTANT` × `TILE_TEXTURE_SIZE_DISTANT`² × 14 B × ~1.33
// (mips) ≈ 64 × 128² × 14 × 1.33 ≈ 20 MB/body. The body is a small disc on
// screen at distant range, so 128² tiles and an 8-LOD cap resolve its
// silhouette and broad relief without the deep cascade the near tier needs for
// the player standing on the surface. Attachment *set* is unchanged (the
// shrink that drops baked albedo/roughness attachments touches the shared
// shader + pipeline and is deferred), so the same `BodyTerrainMaterial` and
// `body_terrain.wgsl` render both tiers with no shader branch.
const LOD_COUNT_DISTANT: u32 = 8;
const TILE_TEXTURE_SIZE_DISTANT: u32 = 128;
const ATLAS_SIZE_DISTANT: u32 = 64;

// ── Map tier ───────────────────────────────────────────────────────────────
//
// The orbital map renders the *focused* body filling the screen — a full sphere
// seen from outside, not the small disc the `Distant` tier targets. Covering a
// whole sphere even at LOD 2 already needs ~96 tiles (6 faces × 4²), so the
// 64-slot `Distant` atlas is slot-starved for this view and gets stuck on a
// coarse LOD whose 128² texels read as visible pixels. This tier is sized for a
// single full-screen body instead: enough slots to cover the visible hemisphere
// at a fine LOD (with the back side and the 2:1 balance ring on coarser tiles),
// a deeper LOD cap so it keeps refining as the map zooms in, and 256² tiles for
// crisper texels. Only ever one body uses it at a time (the map focus), so the
// footprint is a single-body cost, not multiplied across every visible planet.
//
// Memory ≈ `ATLAS_SIZE_MAP` × `TILE_TEXTURE_SIZE_MAP`² × 14 B × ~1.33 (mips)
//        ≈ 256 × 256² × 14 × 1.33 ≈ 310 MB for the one focused body.
const LOD_COUNT_MAP: u32 = 12;
const TILE_TEXTURE_SIZE_MAP: u32 = 256;
const ATLAS_SIZE_MAP: u32 = 256;

/// Detail tier for a body's ground-LOD terrain. Assigned by the residency
/// planner ([`crate::rendering::terrain_residency`]) from how gameplay-relevant
/// the body is: `Near` for the dominant SOI body and predicted encounters
/// (collider-backed, deep LODs, full atlas), `Distant` for bodies that are only
/// big enough on screen to deserve real terrain instead of the icon dot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TerrainTier {
    Near,
    Distant,
    /// The orbital-map focused body: a full-screen single sphere. Larger atlas /
    /// deeper LODs / bigger tiles than `Distant` so it isn't slot-starved.
    Map,
}

/// Concrete atlas/LOD knobs for one [`TerrainTier`]. The attachment *border* and
/// *mip count* are shared across tiers; only the slot count, LOD depth, and tile
/// resolution differ.
struct TierConfig {
    lod_count: u32,
    atlas_size: u32,
    tile_texture_size: u32,
    load_slots: u32,
    load_queue: u32,
}

impl TierConfig {
    /// Resolution for the *appearance* attachments (albedo / roughness /
    /// material), as opposed to height.
    ///
    /// Height carries the silhouette — it is the geometry, it is what the
    /// collider and the shader's normal derivation read, and it is the one
    /// attachment anything physical depends on — so it keeps the full grid. The
    /// other three are the macro-colour anchor and the material masks, and at
    /// half the linear resolution they are visually indistinguishable at the
    /// distances their tier is used at.
    ///
    /// **This is a memory trade, not a synthesis one.** Those three are 10 of the
    /// 14 bytes per texel; at quarter the texels the atlas drops from ~14 B to
    /// ~6.5 B per height-texel-equivalent — better than a 2× cut in the dominant
    /// memory cost of the whole terrain system (the near tier is the biggest
    /// allocation in the game). Synthesis gets *slightly* more expensive, not
    /// less: the provider now evaluates a second, coarser grid rather than
    /// encoding everything from one. That grid is cheap — it is band-limited to
    /// its own resolution, so its `tile_lod_m` is coarser and the detail cascade
    /// resolves fewer octaves per sample — but it is not free, and pretending
    /// otherwise would misattribute where the near-field win comes from.
    ///
    /// Evaluating it separately (rather than box-filtering the height grid down)
    /// is what keeps the borders correct: each attachment's border texels must be
    /// bit-identical with its neighbour tile's, which means they have to come from
    /// that grid's own `stitched_pixel_coordinate`, not from a downsample of a
    /// differently-bordered grid.
    fn detail_texture_size(&self) -> u32 {
        (self.tile_texture_size / 2).max(64)
    }
}

impl TerrainTier {
    fn config(self) -> TierConfig {
        match self {
            TerrainTier::Near => TierConfig {
                lod_count: LOD_COUNT,
                atlas_size: ATLAS_SIZE,
                tile_texture_size: TILE_TEXTURE_SIZE,
                load_slots: TILE_LOAD_SLOTS,
                load_queue: TILE_LOAD_QUEUE_SIZE,
            },
            TerrainTier::Distant => TierConfig {
                lod_count: LOD_COUNT_DISTANT,
                atlas_size: ATLAS_SIZE_DISTANT,
                tile_texture_size: TILE_TEXTURE_SIZE_DISTANT,
                load_slots: TILE_LOAD_SLOTS,
                load_queue: TILE_LOAD_QUEUE_SIZE,
            },
            TerrainTier::Map => TierConfig {
                lod_count: LOD_COUNT_MAP,
                atlas_size: ATLAS_SIZE_MAP,
                tile_texture_size: TILE_TEXTURE_SIZE_MAP,
                load_slots: TILE_LOAD_SLOTS,
                load_queue: TILE_LOAD_QUEUE_SIZE,
            },
        }
    }

    fn label(self) -> &'static str {
        match self {
            TerrainTier::Near => "near",
            TerrainTier::Distant => "distant",
            TerrainTier::Map => "map",
        }
    }
}

/// Marker on the ground-LOD terrain entity spawned for a procedural body.
/// Carries the body id so the impostor↔terrain LOD swap can pair it with the
/// matching impostor without walking parents.
#[derive(Component, Debug)]
pub(crate) struct BodyTerrain {
    pub(crate) body_id: BodyId,
}

/// Marker on the ship-layer impostor billboard for a procedural body. Set in
/// `finalize_planet_generation` when the placeholder mesh is replaced with
/// the `PlanetMaterial` billboard, so the swap system can find both halves
/// of the LOD pair via component queries instead of parent lookups.
#[derive(Component, Debug)]
pub(crate) struct RealSpaceImpostor {
    pub body_id: BodyId,
}

/// Marker on the per-body fullscreen sky-dome entity. Visible while the
/// body's real terrain LOD is active, so the ground path gets the same
/// atmosphere/cloud layer in front of the surface that the impostor path
/// composites inline.
#[derive(Component, Debug)]
pub(super) struct BodySky {
    pub(super) body_id: BodyId,
}

/// Marker on the per-body `PlanetHaloMaterial` billboard (ship layer).
/// Mutually exclusive with [`BodySky`]: when the impostor is active the halo
/// provides the rim-glow outside the billboard's solid sphere; while terrain
/// is active, the fullscreen sky pass covers rim, haze, and cloud overlay.
#[derive(Component, Debug)]
pub(crate) struct BodyHalo {
    pub body_id: BodyId,
}

/// Terrain↔impostor handoff, in units of the icon-dot radius: ground-LOD
/// terrain takes over once the body's ship-view rendered radius reaches this
/// many [`super::screen_marker_radius`]s.
///
/// Slice 6 goal: "anything bigger than a dot of light gets terrain, not a
/// billboard." So the handoff is keyed to apparent *screen size*, not a fixed
/// multiple of body radius — the old `4 × radius` kept the impostor active until
/// the body filled ~28° of the view, far closer than a small disc. At `1.0` the
/// swap happens exactly when the body grows past the icon dot, which for a
/// Thalos-size body is ~530,000 km (vs the old ~12,700 km). The residency
/// planner promotes the body to a resident terrain entity earlier still
/// (`RESIDENT_SCREEN_MARGIN`, ~2× this distance) so the tiles have streamed by
/// the time the swap fires.
///
/// The handoff distance is `radius_m / (SCREEN_MARKER_RADIUS × this)`; `dist <
/// that` ⟺ `radius_m > screen_marker_radius × this`.
///
/// The swap is still a hard cut (flat billboard ↔ 3-D mesh); a smooth crossfade
/// needs opacity uniforms on both materials and is deferred. At dot size the
/// body is only a few pixels across, so the cut is barely visible.
const TERRAIN_HANDOFF_SCREEN_MARKERS: f32 = 1.0;

/// Selects which provider feeds tile data to the UDLOD renderer. `Pipeline` is
/// the production path; `Analytic3d` and `Flat` are diagnostic stand-ins via
/// `SyntheticTileProvider`.
///
/// **Flat verified seamless (2026-05-17).** Every R16 height = 0.5, decoding
/// to 0 m, so the mesh is a smooth sphere at `body.radius_m`. No cross-tile
/// seams at any LOD, which proves UDLOD's vertex stage produces consistent
/// world positions across tile boundaries — the mesh-level baseline is clean.
/// Skip re-testing flat unless vertex math in
/// `crates/rendering/udlod/src/shaders/{vertex,functions}.wgsl` changes. Cross-tile
/// seams seen in `Analytic3d` are therefore caused by height data, not by the
/// vertex stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TerrainTileProviderMode {
    Pipeline,
    Analytic3d,
    Flat,
}

impl TerrainTileProviderMode {
    fn from_env() -> Self {
        let Ok(value) = std::env::var("THALOS_TERRAIN_PROVIDER") else {
            return Self::Pipeline;
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "pipeline" | "default" => Self::Pipeline,
            "analytic" | "analytic3d" | "synthetic" => Self::Analytic3d,
            "flat" | "constant" | "zero" => Self::Flat,
            other => {
                warn!("unknown THALOS_TERRAIN_PROVIDER={other:?}; using pipeline terrain provider");
                Self::Pipeline
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Pipeline => "pipeline",
            Self::Analytic3d => "analytic3d",
            Self::Flat => "flat",
        }
    }
}

pub(super) fn terrain_tile_provider_mode() -> TerrainTileProviderMode {
    TerrainTileProviderMode::from_env()
}

/// Fullbright terrain is a capture-only diagnostic that separates missing
/// raster coverage from Hapke/shadow/BRDF output. It never changes geometry,
/// tile synthesis, or attachment data.
fn terrain_inspection_flag() -> f32 {
    let Ok(value) = std::env::var("THALOS_TERRAIN_INSPECTION") else {
        return 0.0;
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "lit" | "default" | "off" => 0.0,
        "fullbright" | "albedo" | "on" => 1.0,
        "geo-normal" | "geometric-normal" | "smooth-normal" => 2.0,
        "legacy-regolith" | "unfiltered-regolith" => 3.0,
        other => {
            warn!("unknown THALOS_TERRAIN_INSPECTION={other:?}; using lit terrain");
            0.0
        }
    }
}

fn analytic_provider_height_range() -> f32 {
    // Mountain-scale relief regardless of the body's authored range. With
    // the multi-octave fBm signal sharpened to favour peaks (see
    // `synthetic.rs`), encoded ±2.5 km gives a usable mix of ~few-hundred-
    // metre plains and 2 km-class mountains. Override per-launch via
    // `THALOS_TERRAIN_ANALYTIC_RANGE_M` if a session needs more or less.
    const ANALYTIC_DEFAULT_RANGE_M: f32 = 2500.0;

    let Ok(value) = std::env::var("THALOS_TERRAIN_ANALYTIC_RANGE_M") else {
        return ANALYTIC_DEFAULT_RANGE_M;
    };

    match value.trim().parse::<f32>() {
        Ok(range) if range.is_finite() && range > 0.0 => range,
        _ => {
            warn!(
                "invalid THALOS_TERRAIN_ANALYTIC_RANGE_M={value:?}; using analytic default range"
            );
            ANALYTIC_DEFAULT_RANGE_M
        }
    }
}

/// Per-vertex morph band width, expressed as a fraction of one LOD step.
///
/// `morph_range = 0.2` (UDLOD default, tuned for the upstream Earth-scale
/// example) snaps fine-tile edge vertices onto coarse-tile vertex positions
/// in the last 20% of the tile's render range. On a 3.2 Mm body with the
/// camera close to the surface, this transition zone is too narrow — the
/// player walks through it in a few steps and visibly pops between LODs.
///
/// 0.5 spreads the morph over half a LOD step (roughly 1.4× more visual
/// distance per band), which makes the transition substantially smoother
/// while staying under the morph-distance / range invariant the upstream
/// docs call out (`morph_distance >= ~6 tile sizes`).
const TERRAIN_MORPH_RANGE: f32 = 0.5;

/// Same widening for the fragment / vertex height blend band so colours and
/// heights cross-fade over the same window the geometry morphs across.
const TERRAIN_BLEND_RANGE: f32 = 0.5;

/// Distance at which the vertex shader switches from UDLOD's Taylor-series
/// relative-position path (sub-mm precision near the view anchor) to the
/// plain `position_local_to_world` path (f32 ulp ≈ 0.4 m at planet scale).
/// Upstream's default of `0.001` (× body radius ≈ 3 km on Thalos) leaves
/// most of the visible surface on the low-precision path, where the noise
/// floor sits right at the size of the 1 m flat-mode debug checker.
///
/// 10 km is the sweet spot: well past the ~1 km range where 1 m cells
/// are individually resolvable on screen, and short of the ~50 km where
/// Taylor's 2nd-order truncation error (`R · Δst³ / 6`) starts climbing
/// past the f32 direct-path floor. Cost is ~30 fmas per vertex on the
/// high path; with O(10 k) visible vertices that's negligible.
///
/// `TerrainViewConfig::precision_threshold_distance` is a dimensionless
/// multiplier of body radius (UDLOD multiplies by `scale` in
/// `TileTree::new` before the value reaches the shader), so the conversion
/// to a per-body ratio lives in [`body_terrain_view_config`].
const TERRAIN_PRECISION_THRESHOLD_M: f64 = 10_000.0;

/// Above this stationary surface-warp target, keep the current UDLOD request
/// window instead of chasing the inertial camera motion every frame.
///
/// This replaces the old gameplay-level 100× cap: the player can now use the
/// full warp ladder while standing still, and the terrain renderer avoids the
/// one-time request storm/stall that used to happen as warp crossed into the
/// high surface levels.
const SURFACE_WARP_STREAMING_PAUSE_SPEED: f64 = 100.0;

/// UDLOD view config for body terrain. Wider morph + blend bands than the
/// upstream default plus a much larger high-precision threshold so the
/// Taylor branch actually fires; everything else stays at the upstream
/// tuning.
pub(crate) fn body_terrain_view_config(body_radius_m: f64) -> TerrainViewConfig {
    TerrainViewConfig {
        morph_range: TERRAIN_MORPH_RANGE,
        blend_range: TERRAIN_BLEND_RANGE,
        precision_threshold_distance: TERRAIN_PRECISION_THRESHOLD_M / body_radius_m.max(1.0),
        // Standing on a surface, roughly half the near tiles fall behind the
        // camera; synthesizing them is pure waste (upstream's distance-only
        // selection has no notion of where the view is looking). Deferring them
        // is hole-free: the pinned root LODs always leave a resident ancestor, so
        // a deferred tile just draws coarser until the camera turns toward it.
        // The map view turns this off — it sees the whole body at once.
        cull_behind_view: true,
        ..TerrainViewConfig::default()
    }
}

/// Pick the ground-LOD shading style for a body from its authored terrain
/// archetype. Airless impact moons (Mira) reconverge on the orbital impostor's
/// gray Hapke regolith look; everything else keeps the wet, vegetated
/// terrestrial path. Future airless archetypes get added to this match.
pub(crate) fn terrain_shading_style_for(body: &BodyDefinition) -> TerrainShadingStyle {
    match &body.terrain {
        thalos_terrain::TerrainConfig::Feature(cfg) => match cfg.archetype {
            thalos_terrain::BodyArchetype::AirlessImpactMoon => TerrainShadingStyle::Regolith,
            _ => TerrainShadingStyle::Vegetated,
        },
        _ => TerrainShadingStyle::Vegetated,
    }
}

/// Build the UDLOD `TerrainConfig` (atlas dimensions + the four attachments)
/// for a body terrain at a given detail [`TerrainTier`] and geometric `model`.
///
/// The `model` carries the scale (sphere radius + height range): the ship-view
/// terrain passes a true-metre model, the map-view terrain passes a `MAP_SCALE`
/// model. Everything else — attachment formats, atlas size, LOD count — is a
/// pure function of the tier, so both views stream the *same* procedural surface
/// through identical attachments (no shader branch). Shared by
/// [`spawn_body_terrain`] and `rendering::map_terrain`.
pub(crate) fn build_terrain_config(model: TerrainModel, tier: TerrainTier) -> TerrainConfig {
    let tc = tier.config();
    TerrainConfig {
        lod_count: tc.lod_count,
        model,
        atlas_size: tc.atlas_size,
        max_concurrent_tile_loads: tc.load_slots,
        max_queued_tile_loads: tc.load_queue,
        ..Default::default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: tc.tile_texture_size,
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        // Height uses two UNORM16 channels: x stores the coarse normalized
        // height and y stores a residual decoded in the shader. The shader
        // must decode each texel before bilinear filtering; hardware-filtering
        // the packed channels directly is invalid because the residual wraps at
        // every coarse LSB and creates false contour/terrace bands. Plain R16
        // made Thalos' broad, shallow slopes visible as quantized rings;
        // R32Float would need the optional FLOAT32_FILTERABLE wgpu feature
        // that the game intentionally does not request.
        format: AttachmentFormat::Rg16,
    })
    // The three appearance attachments below bake at `detail_texture_size` —
    // half of height's grid. See `TierConfig::detail_texture_size`.
    .add_attachment(AttachmentConfig {
        name: "albedo".to_string(),
        texture_size: tc.detail_texture_size(),
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        format: AttachmentFormat::Rgba8,
    })
    .add_attachment(AttachmentConfig {
        name: "roughness".to_string(),
        texture_size: tc.detail_texture_size(),
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        // thalos_udlod has no single-channel 8-bit format; the source
        // cubemap is u8 and we upscale to u16 in the tile provider.
        format: AttachmentFormat::R16,
    })
    .add_attachment(AttachmentConfig {
        name: "material".to_string(),
        texture_size: tc.detail_texture_size(),
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        // R = grass/vegetation, G = soil/peat, B = exposed rock, A = wetness.
        // These masks drive near-ground material blending; the albedo atlas
        // remains the macro/body-colour anchor for orbital continuity.
        // (Grass *placement* does not read this attachment — it re-derives the
        // same gate on the CPU in `body_render::ground::vegetation` — so the
        // half-res bake here only affects shading, not where blades land.)
        format: AttachmentFormat::Rgba8,
    })
}

/// Pin LOD 0 on every cube face for the atlas's entire lifetime. This gives
/// `get_best_tile` a guaranteed resident ancestor for any tile coordinate, so
/// the GPU shader never samples `INVALID_ATLAS_INDEX` (which decodes to 0 →
/// `min_height` → vertex sits at `-terrain_height_range` below the ellipsoid:
/// visible "void" holes wherever a draw tile has no streamed ancestor).
///
/// `is_spherical` is true for body terrain (`TerrainModel::sphere`), giving six
/// faces; the planar fallback (single side) is harmless to pin defensively.
pub(crate) fn pin_root_tiles(tile_atlas: &mut TileAtlas) {
    let side_count = if tile_atlas.model().is_spherical() {
        6
    } else {
        1
    };
    for side in 0..side_count {
        tile_atlas.pin_tile(TileCoordinate::new(side, 0, 0, 0));
    }
}

/// Spawn the UDLOD terrain for one procedural body.
///
/// `ship_parent_entity` is the body's `RealSpaceBody` entity (the 1 km-cell
/// body grid). `ship_camera` is the entity carrying [`crate::camera::ShipCamera`];
/// the tile-tree resource is keyed by `(terrain, view)` so we have to know the
/// view at spawn time.
///
/// `atmosphere` is the static Rayleigh + Mie block for this body, expressed in
/// ship-view render units (same scale used by the impostor's `ship_atmosphere`).
/// Airless bodies pass `AtmosphereBlock::default()` (all zeros) and no
/// sky-dome entity is spawned.
///
/// `tier` picks the atlas footprint (see [`TerrainTier`]): `Near` for the body
/// the player is on/heading toward (full atlas, deep LODs), `Distant` for a body
/// that is merely visible (tiny atlas so many stay resident cheaply).
pub(crate) fn spawn_body_terrain(
    commands: &mut Commands,
    body: &BodyDefinition,
    ship_parent_entity: Entity,
    materials: &mut Assets<BodyTerrainMaterial>,
    tile_trees: &mut TerrainViewComponents<TileTree>,
    ship_camera: Entity,
    atmosphere: AtmosphereBlock,
    height_mirror: Option<GpuAtlasMirrorHandle>,
    flatten: FlattenHandle,
    tier: TerrainTier,
    sun_shadow_maps: [Handle<Image>; CASCADE_COUNT],
    tile_cache: &mut TileCacheRegistry,
    base_surface: Arc<dyn SurfaceQuery>,
    surface_fingerprint: u64,
) -> Entity {
    let radius_m = body.radius_m as f32;
    let tc = tier.config();
    // The tile cache keys on the *live* flatten handle (its content is hashed per
    // tile request, not snapshotted here) — the provider reads it per tile pixel,
    // so a pad installed after this spawn still changes what later tiles bake.
    let cache_flatten = flatten.clone();
    // The construction site is the one place that names the concrete generation
    // type: wrap it as `Arc<dyn SurfaceQuery>` so the provider and the
    // height-range query see only the black-box seam. The flatten decorator sits
    // on top so a runtime-set pad (the runway) levels the rendered tiles — and,
    // via the GPU-atlas height mirror, the collider and CPU height queries too.
    //
    // Slice 0: the runtime procedural generator replaces the baked cubemap. It
    // is built from the same body params as the near-surface height source
    // (`install_baked_planet`), so the drawn ground and the collider stay in
    // lockstep without sharing an `Arc`.
    let terrain_surface: Arc<dyn SurfaceQuery> =
        Arc::new(FlattenedSurface::new(base_surface, flatten));
    let height_range = rendered_height_range(terrain_surface.as_ref());
    let provider_mode = terrain_tile_provider_mode();
    let terrain_height_range = match provider_mode {
        TerrainTileProviderMode::Pipeline => height_range,
        TerrainTileProviderMode::Analytic3d => analytic_provider_height_range(),
        TerrainTileProviderMode::Flat => 0.0,
    };

    let model = TerrainModel::sphere(
        DVec3::ZERO,
        body.radius_m,
        -terrain_height_range,
        terrain_height_range,
    );

    let config = build_terrain_config(model, tier);

    let provider: Box<dyn TileProvider> = match provider_mode {
        TerrainTileProviderMode::Pipeline => Box::new(PipelineTileProvider::new(
            body.name.clone(),
            terrain_surface,
        )),
        TerrainTileProviderMode::Analytic3d => {
            info!(
                "using analytic 3D terrain provider override for '{}'; visible ground no longer \
                 matches baked terrain or CPU height queries",
                body.name
            );
            Box::new(SyntheticTileProvider::new(
                -terrain_height_range,
                terrain_height_range,
            ))
        }
        TerrainTileProviderMode::Flat => {
            info!(
                "using flat terrain provider override for '{}'; visible ground is a constant \
                 sphere; gameplay height queries use the same zero-height surface",
                body.name
            );
            Box::new(SyntheticTileProvider::with_mode(
                0.0,
                0.0,
                SyntheticTerrainMode::Flat,
            ))
        }
    };
    // Memoize the synthesizing provider: in RAM (surviving the despawn/respawn
    // that flatten-invalidation and residency-tier swaps perform) and on disk
    // (surviving the process). See `rendering::tile_cache` — this is what turns a
    // cold ~15 s surface site into a warm one.
    let provider = tile_cache.wrap_provider(
        body.id,
        provider,
        &config,
        Some(cache_flatten),
        surface_fingerprint,
    );

    let mut tile_atlas = TileAtlas::with_provider(&config, provider);
    pin_root_tiles(&mut tile_atlas);
    let view_config = body_terrain_view_config(body.radius_m);
    let tile_tree = TileTree::new(&tile_atlas, &view_config);

    // The body's `real_body_entity` (`ship_parent_entity`) carries a
    // `Grid::new(REAL_SPACE_CELL_SIZE_M, 0.0)`. Reconstructing it here is
    // cheaper than threading the actual `Grid` through call sites; they're
    // identical by construction.
    let body_grid = Grid::new(REAL_SPACE_CELL_SIZE_M, 0.0);

    // Start hidden so newly-spawned terrains don't pop into view at a
    // wildly wrong LOD before the swap system has a chance to run (it runs
    // in `SimStage::Sync`, the same tick as the spawn but strictly after
    // `finalize_planet_generation`). The swap system flips visibility
    // on/off based on camera distance from this frame onward.
    let mut bundle = TerrainBundle::new(tile_atlas, &body_grid);
    bundle.visibility = Visibility::Hidden;

    // Enable the per-fragment checkerboard for the flat debug provider —
    // doing it in the texture is hopeless at this body's scale (1 m tiles
    // alias into moiré at every reasonable viewing distance), so the
    // shader synthesises the pattern from the geometric normal × radius.
    let debug = match provider_mode {
        // `view_phase` and `world_to_body_rot` are refreshed every frame
        // by `update_body_terrain_atmosphere`; spawn-time values are
        // placeholders that will be overwritten on the first tick.
        // `params = (checker_mode=1, _, checker_cell_size_m=1, _)`.
        TerrainTileProviderMode::Flat => BodyTerrainDebug {
            params: Vec4::new(1.0, 0.0, 1.0, 0.0),
            ..Default::default()
        },
        _ => BodyTerrainDebug::default(),
    };

    // `scene` is zeroed here; `update_body_terrain_atmosphere` writes the
    // correct sun direction, flux, occluders, and ambient on the first Sync
    // tick after spawn.
    // Surface shading style is body-static (derived from the terrain
    // archetype), so set it once here; the per-frame `extras` writer
    // (`update_body_terrain_atmosphere`) only touches `shadow`/`debug`
    // and leaves `inspection` alone.
    let shading_style = terrain_shading_style_for(body);
    let material = BodyTerrainMaterial {
        atmosphere,
        scene: SceneLighting::default(),
        extras: BodyTerrainExtras {
            debug,
            // w = 1.0 enables screen-space AO (F5) on the surface terrain; the
            // live `ao` handle is patched in each frame (see the terrain loop).
            inspection: Vec4::new(
                terrain_inspection_flag(),
                shading_style.shader_flag(),
                0.0,
                1.0,
            ),
            ..Default::default()
        },
        sun_shadow_map_0: sun_shadow_maps[0].clone(),
        sun_shadow_map_1: sun_shadow_maps[1].clone(),
        sun_shadow_map_2: sun_shadow_maps[2].clone(),
        // White fallback until the terrain loop patches the live AO image.
        ao: Handle::default(),
        // Likewise for the contact-shadow term (W18a); the gate lives in
        // `extras.shadow.gate.z`, written by the sun-shadow rig.
        contact_shadow: Handle::default(),
    };

    let mut terrain = commands.spawn((
        bundle,
        MeshMaterial3d(materials.add(material)),
        RenderLayers::layer(SHIP_LAYER),
        // Ground terrain receives craft/tree/structure shadows through the
        // custom `thalos::shadow` cascade rig (stock Bevy CSM on the sun is
        // disabled — one shadow world); rendering the planet itself into a
        // shadow map would be wasteful and noisy. Terrain relief self-shadow
        // comes from the shader's own height-atlas horizon march.
        NotShadowCaster,
        ChildOf(ship_parent_entity),
        Name::new(format!("{} Terrain", body.name)),
        BodyTerrain { body_id: body.id },
    ));
    if let Some(mirror) = height_mirror {
        terrain.insert(GpuAtlasHeightMirrorComponent::new(mirror));
    }
    let terrain_entity = terrain.id();

    tile_trees.insert((terrain_entity, ship_camera), tile_tree);

    info!(
        "spawned ground terrain for '{}' ({} tier, provider {}, radius {:.0} km, height range ±{:.0} m, atlas size {}, lod {}, tile {})",
        body.name,
        tier.label(),
        provider_mode.label(),
        radius_m / 1000.0,
        terrain_height_range,
        tc.atlas_size,
        tc.lod_count,
        tc.tile_texture_size,
    );

    terrain_entity
}

/// Icosphere subdivision for the water surface mesh.
///
/// 7 levels = 327,680 triangles. The on-foot EVA path renders the water
/// mesh from a few metres away at the shoreline, where 6-level mesh
/// facets read as polygonal seams between water and terrain. 7 keeps the
/// shoreline tessellation tight enough that the water/terrain
/// intersection looks like a continuous line at human scale; bump to 8 if
/// Unified per-body render-LOD visibility.
///
/// One pass over each body decides which of its four ship-layer render
/// entities are active for this frame, based on a single camera-to-body
/// distance. Two orthogonal axes:
///
/// * **Surface LOD** — the body's rendered radius exceeds the icon dot
///   (`radius_m > screen_marker_radius × TERRAIN_HANDOFF_SCREEN_MARKERS`,
///   equivalently `dist < radius_m / (SCREEN_MARKER_RADIUS × markers)`) selects
///   the 3-D terrain mesh; otherwise the flat impostor billboard / icon. Slice 6
///   keys this to apparent screen size, not a fixed `4 × radius`, so anything
///   bigger than a dot of light gets real terrain. Logically the same body at
///   two projections; never draw both.
/// * **Atmosphere LOD** — terrain uses `BodySky`, impostor uses `BodyHalo`.
///   `BodySky` covers every screen pixel and clips at scene depth, which is
///   what gives ground LOD terrain aerial perspective and the detached cloud
///   shell — and (Slice 6) the limb/rim glow once it is visible from outside the
///   shell too. `BodyHalo` covers only the billboard shell silhouette; it is not
///   spawned today, so beyond the swap a sub-dot body shows no halo (just the
///   impostor disc).
///
/// Two bands fall out for a body with an atmosphere (`markers` =
/// `TERRAIN_HANDOFF_SCREEN_MARKERS`):
///
/// | apparent size                    | surface  | atmosphere |
/// |----------------------------------|----------|------------|
/// | bigger than `markers` icon dots  | terrain  | BodySky    |
/// | smaller (sub-dot)                | impostor | halo/none  |
///
/// Airless bodies skip the BodySky entity entirely (never spawned) and
/// the halo's shader early-outs to a no-op; we still toggle its visibility
/// for consistency.
///
/// **Terrain residency gate.** Ground-LOD terrain entities are spawned lazily by
/// [`crate::rendering::terrain_residency`] only for bodies that are
/// gameplay-relevant or big enough on screen. For a body without a resident
/// `BodyTerrain` entity, the impostor stays visible at all distances — otherwise
/// the body would silently vanish once it grew past the dot with no terrain to
/// take over.
///
/// The map-layer impostor and map halo live on `MAP_LAYER` and are not
/// touched here.
pub(super) fn pause_surface_terrain_streaming_at_high_warp(
    mut commands: Commands,
    sim: Res<SimulationState>,
    eva_mode: Res<EvaMode>,
    player: Option<Res<PlayerControllerState>>,
    terrains: Query<(Entity, &BodyTerrain, Option<&TerrainStreamingPaused>)>,
) {
    let surface_stationary = match sim.simulation.authority() {
        AuthorityMode::BodyFixed { .. } => true,
        _ if sim.simulation.vessel_kind() == VesselKind::Eva && eva_mode.is_grounded() => player
            .as_deref()
            .map(|state| state.is_at_rest())
            .unwrap_or(false),
        _ => false,
    };
    let pause = surface_stationary
        && sim.simulation.warp.target_speed() > SURFACE_WARP_STREAMING_PAUSE_SPEED;
    let dominant = sim.simulation.dominant_body();

    for (entity, terrain, paused) in &terrains {
        let should_pause = pause && terrain.body_id == dominant;
        match (should_pause, paused.is_some()) {
            (true, false) => {
                commands.entity(entity).insert(TerrainStreamingPaused);
            }
            (false, true) => {
                commands.entity(entity).remove::<TerrainStreamingPaused>();
            }
            _ => {}
        }
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn sync_body_render_lod(
    sim: Res<SimulationState>,
    ship_cam_q: Query<&GlobalTransform, With<ShipCamera>>,
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut terrains: Query<
        (&BodyTerrain, &TileAtlas, &mut Visibility),
        (
            Without<RealSpaceImpostor>,
            Without<BodySky>,
            Without<BodyHalo>,
            Without<BodyOcean>,
        ),
    >,
    mut impostors: Query<
        (&RealSpaceImpostor, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<BodySky>,
            Without<BodyHalo>,
            Without<BodyOcean>,
        ),
    >,
    mut skies: Query<
        (&BodySky, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodyHalo>,
            Without<BodyOcean>,
        ),
    >,
    mut oceans: Query<
        (&BodyOcean, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodySky>,
            Without<BodyHalo>,
        ),
    >,
    mut halos: Query<
        (&BodyHalo, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodySky>,
            Without<BodyOcean>,
        ),
    >,
) {
    let Ok(cam_xform) = ship_cam_q.single() else {
        return;
    };
    let cam_pos = cam_xform.translation();

    // The world-render-space position of each procedural body is its
    // `RealSpaceBody` grid origin. Index by body_id once per frame so each
    // marker loop is O(N) in its own entity count.
    let mut body_pos_by_id: std::collections::HashMap<BodyId, Vec3> =
        std::collections::HashMap::with_capacity(sim.system.bodies.len());
    for (rsb, xform) in &body_q {
        body_pos_by_id.insert(rsb.body_id, xform.translation());
    }

    // Bodies with a resident ground-LOD terrain entity *whose pinned root
    // LODs have finished loading*. With lazy residency (see
    // `terrain_residency`), most bodies have no `BodyTerrain` at all — the
    // impostor / halo must then stay visible at every distance, otherwise
    // the body silently disappears once it grows past the icon dot. The same
    // applies briefly at spawn while LOD 0 streams: until the pinned tiles are
    // ready, treat the body as non-resident so the impostor stays up; the
    // terrain entity is only flipped to `Inherited` once it has a complete
    // resident-ancestor chain (otherwise the GPU samples
    // `INVALID_ATLAS_INDEX` → vertices drop to `min_height` → visible "void"
    // holes during the load window).
    let terrain_resident: std::collections::HashSet<BodyId> = terrains
        .iter()
        .filter_map(|(t, atlas, _)| atlas.pinned_tiles_ready().then_some(t.body_id))
        .collect();

    // Returns (distance, swap_threshold, shell_radius) for one body, or
    // None if the body or its render-space position is missing. The swap
    // threshold is the camera distance at which the body's rendered radius
    // equals `TERRAIN_HANDOFF_SCREEN_MARKERS` icon-dot radii: terrain shows
    // inside it, impostor/icon outside. `screen_marker_radius(p, cam) =
    // |p - cam| × SCREEN_MARKER_RADIUS`, so `radius_m > marker × markers`
    // ⟺ `dist < radius_m / (SCREEN_MARKER_RADIUS × markers)`. The shell radius
    // is kept for diagnostics / future atmosphere-specific LOD tuning; today's
    // visibility choice keys atmosphere to the surface LOD so terrain always
    // gets the fullscreen in-front pass.
    let body_metrics = |body_id: BodyId| -> Option<(f32, f32, f32)> {
        let body = sim.system.bodies.get(body_id)?;
        let body_pos = body_pos_by_id.get(&body_id)?;
        let radius = body.radius_m as f32;
        let karman = body
            .terrestrial_atmosphere
            .as_ref()
            .map(|a| a.karman_line_m)
            .unwrap_or(0.0);
        let dist = (cam_pos - *body_pos).length();
        let swap_distance = radius / (SCREEN_MARKER_RADIUS * TERRAIN_HANDOFF_SCREEN_MARKERS);
        Some((dist, swap_distance, radius + karman))
    };

    let set_vis = |vis: &mut Visibility, want: Visibility| {
        if *vis != want {
            *vis = want;
        }
    };

    for (terrain, atlas, mut vis) in &mut terrains {
        let Some((dist, swap, _shell)) = body_metrics(terrain.body_id) else {
            continue;
        };
        // Stay hidden until the atlas's pinned root tiles (LOD 0 per face)
        // have finished loading. Without this gate, a freshly-spawned
        // terrain becomes `Inherited` the instant the camera enters the
        // swap radius — but the atlas is still empty, so every vertex
        // samples `INVALID_ATLAS_INDEX` and the mesh sits one full
        // `terrain_height_range` below the ellipsoid (visible void).
        let want = if dist < swap && atlas.pinned_tiles_ready() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
    }

    for (impostor, mut vis) in &mut impostors {
        let Some((dist, swap, _shell)) = body_metrics(impostor.body_id) else {
            continue;
        };
        // Only hand off to terrain if a ground-LOD entity actually exists
        // for this body. Non-resident bodies stay impostor-visible at every
        // distance.
        let resident = terrain_resident.contains(&impostor.body_id);
        let want = if resident && dist < swap {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        set_vis(&mut vis, want);
    }

    for (sky, mut vis) in &mut skies {
        let Some((dist, swap, _shell)) = body_metrics(sky.body_id) else {
            continue;
        };
        // `BodySky` is always spawned at startup (per `spawn.rs`) for the
        // cmd-shift-click-teleport-into-atmosphere case, so we can flip it
        // independently of terrain residency. But it should only take over
        // from the halo when a terrain entity is up — otherwise the halo
        // is the body's atmosphere representation.
        let resident = terrain_resident.contains(&sky.body_id);
        let want = if resident && dist < swap {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
    }

    for (ocean, mut vis) in &mut oceans {
        let Some((dist, swap, _shell)) = body_metrics(ocean.body_id) else {
            continue;
        };
        let resident = terrain_resident.contains(&ocean.body_id);
        let want = if resident && dist < swap {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
    }

    for (halo, mut vis) in &mut halos {
        let Some((dist, swap, _shell)) = body_metrics(halo.body_id) else {
            continue;
        };
        // Halo is the atmosphere visual for the impostor. Show it whenever
        // we're showing the impostor: beyond the screen-size swap, OR at any
        // distance if terrain isn't resident. (No `BodyHalo` entity is spawned
        // today, so this loop is a no-op; `BodySky`'s limb now covers the
        // from-space rim glow.)
        let resident = terrain_resident.contains(&halo.body_id);
        let want = if !resident || dist >= swap {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
    }
}

/// Aerial-perspective (distance-haze) controls + sky-dome dev overrides
/// (Reflect-registered via `#[reflect(Resource)]` for a future debug UI).
///
/// **Why this exists.** A body's authored `scattering.strength` /
/// `multi_scatter_gain` brighten the **sky dome** so a clear midday sky reads
/// bright and crushes stars. But the `BodySky` fullscreen pass adds the *same*
/// in-scatter as the airlight (aerial perspective) on terrain/geometry pixels —
/// and that airlight is far brighter than the surface even over short paths, so
/// the ground washes out into a uniform veil at any altitude. Extinction
/// (transmittance) is already Earth-clear-day correct (~50–90 km visibility);
/// the wash is purely the additive airlight. See `docs/atmosphere.md` *Aerial
/// perspective*.
///
/// **The fix.** `aerial_perspective_strength` is an **absolute** in-scatter
/// strength used only for the terrain airlight, decoupled from the (much
/// higher) sky-dome strength. The per-body shader multiplier is
/// `aerial_perspective_strength / effective_sky_strength`, so every body's
/// ground haze lands at the same absolute strength regardless of how bright its
/// sky is authored — and remains physically proportional to that body's β
/// (thicker atmospheres still haze more). This is the **clear-weather
/// visibility knob weather will later modulate** (lower = clearer, higher =
/// hazier/humid). Plumbed to the shader via `BodySkyExtra::cloud_band_radii.z`.
///
/// **Dev overrides.** `strength` / `multi_scatter_gain`, when `>= 0`, override
/// `atmos_geom.z` / `atmos_geom.w` on every body's sky+terrain material this
/// frame (`< 0` keeps the authored spawn-time value). Both are pure runtime
/// multipliers — they do **not** feed the multi-scatter LUT bake — so changing
/// them is exact (no LUT rebake). Handy for sky tuning; not needed for normal
/// play. Edit the default below (Reflect-registered for a future debug UI) and
/// rebuild to change them.
#[derive(Resource, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct AtmosphereTuning {
    pub strength: f32,
    pub multi_scatter_gain: f32,
    pub aerial_perspective_strength: f32,
}

impl Default for AtmosphereTuning {
    fn default() -> Self {
        Self {
            // Negative sentinels: keep each body's authored sky-dome strength /
            // gain until explicitly overridden.
            strength: -1.0,
            multi_scatter_gain: -1.0,
            // Clear-weather default. Tuned against the in-atmosphere flight view:
            // ground reads crisply from altitude with only a subtle haze building
            // toward the horizon, instead of a uniform veil. Lowered from 0.15
            // alongside the Mie cut (thalos.ron) — together they keep the noon
            // distance crisp instead of washing to a grey-tan band. Weather will
            // later drive this up for hazy/humid conditions.
            aerial_perspective_strength: 0.10,
        }
    }
}

/// Ocean field inspection controls. Production defaults are inert; the
/// headless screenshot harness enables these explicitly for deterministic
/// topology/phase captures.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub(crate) struct OceanDebugSettings {
    pub(crate) slope_view: bool,
    pub(crate) phase_time_override_s: Option<f64>,
}

/// Update the per-frame dynamic data on every body's `BodyTerrainMaterial`
/// and `BodySkyMaterial`:
///
/// - terrain `planet_extra` (planet center + radius, terminator-wrap knob)
/// - terrain `scene` (`SceneLighting`: primary star, eclipse occluders,
///   ambient; planetshine left zero for now)
/// - sky `atmosphere_extra` (sun dir + flux, planet center + radius)
///
/// Must run after `sync_solar_system_state` (for sun direction from ephemeris),
/// `update_camera_exposure` (for exposure gain), and
/// `update_real_space_body_positions` (for up-to-date body grid transforms).
pub(super) fn update_body_terrain_atmosphere(
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    terrain_q: Query<(&BodyTerrain, &MeshMaterial3d<BodyTerrainMaterial>)>,
    sky_q: Query<(&BodySky, &MeshMaterial3d<BodySkyMaterial>)>,
    ship_cam_q: Query<(&CellCoord, &Transform), With<ShipCamera>>,
    sim: Res<SimulationState>,
    cache: Res<SolarSystemState>,
    exposure: Res<CameraExposure>,
    ocean_debug: Res<OceanDebugSettings>,
    // Combined into one tuple param to stay under Bevy's 16-arg system limit:
    // .0 = cloud config (heights for occlusion), .1 = which body the cloud
    // raymarch is currently rendered for (sole writer: `clouds::drive_clouds`),
    // .2 = live atmosphere airlight tuning,
    // .3 = sun-shadow map handle + view_proj + params (folded in to stay under
    //      Bevy's 16-system-arg limit).
    // .4 = screen-space AO image (F5); patched onto each terrain material so the
    //      shader can multiply it into the ambient occlusion.
    // .5 = AO config: drives the per-material AO gate/debug flag (inspection.w).
    // .6 = screen-space contact-shadow image (W18a); patched on alongside the AO
    //      image. Its gate travels inside `.3`'s block (`gate.z`), published by
    //      the shadow rig, so there is no separate config arg here.
    cloud_io: (
        Option<Res<thalos_body_render::CloudsConfig>>,
        Res<super::clouds::ActiveCloudBody>,
        Res<AtmosphereTuning>,
        Res<super::sun_shadow::SunShadowState>,
        Option<Res<super::ssao::AoImage>>,
        Res<super::ssao::SsaoConfig>,
        Option<Res<super::contact_shadow::ContactShadowImage>>,
    ),
    flatten_registry: Res<TerrainFlattenRegistry>,
    // ADR-20260720T185958Z-water-projects-one-signed-sea-field: resident-height-tile lookup inputs for the sky material's
    // analytic-ocean branch. .0 finds each body's udlod terrain entity + its
    // `TileAtlas` (lod count, height decode range, attachment-0 UV layout);
    // .1 supplies the per-(terrain, view) tile tree's window size.
    tile_lookup_io: (
        Query<(
            Entity,
            &BodyTerrain,
            &thalos_body_render::udlod::prelude::TileAtlas,
        )>,
        Option<
            Res<
                thalos_body_render::udlod::terrain_view::TerrainViewComponents<
                    thalos_body_render::udlod::prelude::TileTree,
                >,
            >,
        >,
    ),
    mut terrain_materials: ResMut<Assets<BodyTerrainMaterial>>,
    mut sky_materials: ResMut<Assets<BodySkyMaterial>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };

    let star_pos = states.first().map(|s| s.position).unwrap_or_default();
    // Atmosphere airlight tuning (see `AtmosphereTuning`).
    let tuning = &cloud_io.2;

    // Planet center and orientation in render space from each body's grid
    // transform. The real-space grid rotation is body-local → world, so the
    // cloud cubemap sampler wants its inverse.
    let mut body_render_pos: std::collections::HashMap<BodyId, Vec3> =
        std::collections::HashMap::with_capacity(sim.system.bodies.len());
    let mut world_to_body_orientation: std::collections::HashMap<BodyId, Quat> =
        std::collections::HashMap::with_capacity(sim.system.bodies.len());
    for (rsb, xform) in &body_q {
        body_render_pos.insert(rsb.body_id, xform.translation());
        world_to_body_orientation.insert(
            rsb.body_id,
            xform.compute_transform().rotation.inverse().normalize(),
        );
    }

    // Eclipse occluders at SHIP_SCALE. The terrain lives in the BigSpace
    // SHIP_LAYER where 1 render unit = 1 m, so we can use the body grid
    // translations directly without an extra origin/scale transform.
    let mut occluders: Vec<(BodyId, Vec3, f32)> = Vec::with_capacity(sim.system.bodies.len());
    for (i, body) in sim.system.bodies.iter().enumerate() {
        if matches!(body.kind, thalos_world::BodyKind::Star) || body.radius_m < 1.0 {
            continue;
        }
        let Some(pos) = body_render_pos.get(&i) else {
            continue;
        };
        occluders.push((i, *pos, body.radius_m as f32));
    }

    // Camera position in heliocentric inertial coords (f64), reconstructed from
    // the ship camera's big_space cell + local translation so it stays
    // f64-precise at planet radius. Used both for the analytic-ocean camera
    // altitude below and the flat-mode debug checker's `view_phase` further down.
    let camera_inertial = ship_cam_q
        .single()
        .map(|(cell, transform)| {
            DVec3::new(cell.x as f64, cell.y as f64, cell.z as f64) * REAL_SPACE_CELL_SIZE_M as f64
                + transform.translation.as_dvec3()
        })
        .unwrap_or_else(|_| sim.simulation.ship_state().position);

    // ADR-20260720T185958Z-water-projects-one-signed-sea-field: per-body height-tile lookup parameters for the sky material's
    // ocean branch, which samples signed sea height from the same udlod atlas
    // the terrain mesh displaces from. Bodies without a terrain (or whose
    // tile tree hasn't spawned yet) keep `tile_lookup.x = 0` — the shader
    // falls back to the coast atlas, and the material's bind-group prepare
    // also force-disables the flag if the render-world resources are missing.
    let (terrain_atlas_q, tile_trees) = &tile_lookup_io;
    let mut tile_lookup_by_body: std::collections::HashMap<BodyId, (Entity, Vec4, Vec4)> =
        std::collections::HashMap::new();
    for (terrain_entity, terrain, atlas) in terrain_atlas_q {
        // Attachment 0 is the height attachment by construction
        // (`build_terrain_config` adds it first); verify by name anyway so a
        // future re-ordering fails safe (coast-atlas fallback) instead of
        // decoding albedo texels as heights.
        let Some(height_cfg) = atlas.attachment_configs().first() else {
            continue;
        };
        if height_cfg.name != "height" {
            continue;
        }
        let Some(tree_size) = tile_trees.as_ref().and_then(|trees| {
            trees
                .iter()
                .find(|((t, _view), _)| *t == terrain_entity)
                .map(|(_, tree)| tree.tree_size())
        }) else {
            continue;
        };
        let texture = height_cfg.texture_size as f32;
        let border = height_cfg.border_size as f32;
        let center = texture - 2.0 * border;
        let (min_h, max_h) = atlas.model().height_range();
        tile_lookup_by_body.insert(
            terrain.body_id,
            (
                terrain_entity,
                Vec4::new(1.0, atlas.lod_count() as f32, tree_size as f32, center),
                Vec4::new(center / texture, border / texture, min_h, max_h),
            ),
        );
    }

    // Per-body sky data: sun direction, flux, planet center, radius.
    let mut sky_by_body: std::collections::HashMap<BodyId, BodySkyExtra> =
        std::collections::HashMap::with_capacity(sim.system.bodies.len());
    for (i, body) in sim.system.bodies.iter().enumerate() {
        let Some(body_state) = states.get(i) else {
            continue;
        };
        let Some(render_pos) = body_render_pos.get(&i) else {
            continue;
        };
        let offset = star_pos - body_state.position;
        let dist = offset.length();
        let sun_dir = if dist > 0.0 {
            (offset / dist).as_vec3()
        } else {
            Vec3::Y
        };
        let au_over_d = (AU_M / dist.max(1.0)) as f32;
        let flux = LIGHT_AT_1AU * au_over_d * au_over_d * exposure.gain;
        let planet_radius = body.radius_m as f32;
        let q = world_to_body_orientation
            .get(&i)
            .copied()
            .unwrap_or(Quat::IDENTITY);
        // Cloud band radii (render units) for the body the cloud raymarch is
        // rendered for, so the sky pass can keep the cloud layer from painting
        // over closer geometry.
        let cloud_band_radii = if Some(i) == cloud_io.1.0 {
            cloud_io
                .0
                .as_ref()
                .map(|cfg| {
                    Vec4::new(
                        planet_radius + cfg.clouds_bottom_height,
                        planet_radius + cfg.clouds_top_height,
                        0.0,
                        // w = composite-enable flag: 1.0 only on the body whose
                        // live cloud texture is bound below. The sky shader skips
                        // the cloud composite when this is 0.0 — every other body
                        // carries the 1×1 blank fallback, which the screen-space
                        // `textureLoad` would otherwise read out of bounds (→
                        // (0,0,0,0), an opaque black sky). `drive_clouds` clears
                        // `ActiveCloudBody` when clouds are disabled in graphics
                        // settings, so this flag goes 0.0 there too.
                        1.0,
                    )
                })
                .unwrap_or(Vec4::ZERO)
        } else {
            Vec4::ZERO
        };
        // Analytic ocean: a math sphere at `planet_radius + sea_level_m` shaded
        // as water inside the BodySky pass (see `body_sky.wgsl`). Render space on
        // SHIP_LAYER is 1 unit = 1 m, so the radius is just metres. Sea level is
        // 0 for runtime procedural oceans (shoreline pinned at the reference
        // radius); `None` (airless / ancient-dry) disables the branch.
        let (
            ocean,
            ocean_color_depth,
            ocean_camera_phase,
            ocean_low_phase,
            ocean_high_phase,
            ocean_slope_amplitudes,
            ocean_spectrum,
            ocean_wind_basis,
            ocean_crosswind_basis,
        ) = match body.terrain.ocean_sea_level_m() {
            Some(sea_level_m) => {
                // w = camera height above the sea sphere, computed in f64 so the
                // shader's ray-sphere intersection is stable at planet radius
                // (an f32 `b² − c` there catastrophically cancels → the surface
                // jitters as the camera moves). The shader rebuilds the near root
                // from this precise altitude instead.
                let sea_r = body.radius_m + sea_level_m as f64;
                let cam_alt_sea = ((camera_inertial - body_state.position).length() - sea_r) as f32;
                let state = body.ocean.unwrap_or_default();
                let wave_time_s = ocean_debug
                    .phase_time_override_s
                    .unwrap_or_else(|| sim.simulation.sim_time());
                let projection =
                    project_ocean_spectrum(&state, body.surface_gravity_m_s2(), wave_time_s);
                let camera_body =
                    body_state.orientation.inverse() * (camera_inertial - body_state.position);
                let frame = ocean_wave_frame(camera_body, &state);
                let (deep_r, deep_g, deep_b) = state.deep_water_color;
                (
                    // x = ocean radius (m), y = enable, z = shore-wave time
                    // reduced to its 14 s repeat period in f64, w = camera
                    // altitude above sea. Spectral packet phases are separate
                    // below; no large epoch is uploaded as f32.
                    Vec4::new(
                        planet_radius + sea_level_m,
                        1.0,
                        wave_time_s.rem_euclid(14.0) as f32,
                        cam_alt_sea,
                    ),
                    Vec4::new(deep_r, deep_g, deep_b, state.optical_depth_m.max(0.1)),
                    frame.camera_phase_m,
                    projection.low_phase,
                    projection.high_phase,
                    projection.slope_amplitudes,
                    Vec4::new(
                        frame.swell_angle_rad,
                        state.swell_energy.clamp(0.0, 1.0),
                        state.foam_slope_onset.max(0.01),
                        if ocean_debug.slope_view { 1.0 } else { 0.0 },
                    ),
                    frame.wind_basis,
                    frame.crosswind_basis,
                )
            }
            None => (
                Vec4::ZERO,
                Vec4::ZERO,
                Vec4::ZERO,
                Vec4::ZERO,
                Vec4::ZERO,
                Vec4::ZERO,
                Vec4::ZERO,
                Vec4::ZERO,
                Vec4::ZERO,
            ),
        };
        let (tile_lookup, tile_atlas_uv) = tile_lookup_by_body
            .get(&i)
            .map(|(_, lookup, uv)| (*lookup, *uv))
            .unwrap_or((Vec4::ZERO, Vec4::ZERO));
        sky_by_body.insert(
            i,
            BodySkyExtra {
                sun_dir_flux: Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, flux),
                planet_center_radius: Vec4::new(
                    render_pos.x,
                    render_pos.y,
                    render_pos.z,
                    planet_radius,
                ),
                world_to_body_orientation: Vec4::new(q.x, q.y, q.z, q.w),
                cloud_band_radii,
                ocean,
                ocean_color_depth,
                ocean_camera_phase,
                ocean_low_phase,
                ocean_high_phase,
                ocean_slope_amplitudes,
                ocean_spectrum,
                ocean_wind_basis,
                ocean_crosswind_basis,
                tile_lookup,
                tile_atlas_uv,
                cloud_march: if Some(i) == cloud_io.1.0 {
                    cloud_io
                        .0
                        .as_ref()
                        .map(|cfg| {
                            Vec4::new(cfg.clouds_raymarch_steps_count as f32, 0.0, 0.0, 0.0)
                        })
                        .unwrap_or(Vec4::ZERO)
                } else {
                    Vec4::ZERO
                },
            },
        );
    }

    // `camera_inertial` (f64) was computed above the sky-data loop — the render
    // camera's heliocentric position, the same `view.world_position` the shader
    // differences fragments against. The flat-mode debug checker's `view_phase`
    // (below) must reference it, not the craft: using the craft position slides
    // the checker across the surface whenever the camera orbits a stationary
    // player, because the phase reference and the shader's camera reference then
    // disagree by the orbit offset.
    for (terrain, mat_handle) in &terrain_q {
        let Some(mut mat) = terrain_materials.get_mut(mat_handle) else {
            continue;
        };
        mat.scene =
            build_terrain_scene_lighting(terrain.body_id, states, &occluders, exposure.gain);
        // Moonlight: the brightest child moon (e.g. Mira over Thalos) reflecting
        // the star back down, so a full moon lights the night landscape. Filled
        // only for the surface terrain path — the orbital map terrain skips it.
        let (moon_dir_flux, moon_color) =
            compute_moonlight(terrain.body_id, &sim.system.bodies, states, exposure.gain);
        mat.scene.moonlight_dir_flux = moon_dir_flux;
        mat.scene.moonlight_color = moon_color;
        // Sun-shadow map: the camera + matrix are owned by `sun_shadow`; bind
        // the handle and the render-space → shadow-clip transform. `params.x`
        // is 0 when the pass is inactive (orbit / off-surface), so the shader
        // skips sampling entirely.
        let sun_shadow = &cloud_io.3;
        mat.sun_shadow_map_0 = sun_shadow.images[0].clone();
        mat.sun_shadow_map_1 = sun_shadow.images[1].clone();
        mat.sun_shadow_map_2 = sun_shadow.images[2].clone();
        mat.extras.shadow = sun_shadow.block;
        // Screen-space AO (F5): bind the live half-res AO image so the shader can
        // multiply it into the ambient occlusion. The gate/debug flag rides
        // `inspection.w` (0 = off via THALOS_SSAO, 1 = apply, 2 = paint raw AO).
        if let Some(ao) = &cloud_io.4 {
            mat.ao = ao.handle.clone();
        }
        mat.extras.inspection.w = cloud_io.5.terrain_flag();
        // Contact shadows (W18a): bind the live full-res image; the gate came in
        // with `extras.shadow` above (`gate.z`), so nothing extra is set here.
        if let Some(contact) = &cloud_io.6 {
            mat.contact_shadow = contact.handle.clone();
        }
        // Live strength/gain override (keeps the terrain's atmosphere-driven
        // ambient sky fill in step with the sky dome). `< 0` = keep authored.
        if tuning.strength >= 0.0 {
            mat.atmosphere.atmos_geom.z = tuning.strength;
        }
        if tuning.multi_scatter_gain >= 0.0 {
            mat.atmosphere.atmos_geom.w = tuning.multi_scatter_gain;
        }
        // Body-fixed camera phase for shader-side procedural detail and the
        // optional flat-mode debug checker. The nearby terrain shader works in
        // camera-relative render metres for precision, then adds this f64-
        // computed body-fixed phase so metre-scale noise stays glued to the
        // rotating planet instead of sliding with floating-origin shifts/time
        // warp. Keep this period in sync with `DETAIL_COORD_PERIOD_M` in
        // `body_terrain.wgsl`.
        let period = 4000.0_f64;
        let (view_phase, world_to_body_q) = states
            .get(terrain.body_id)
            .map(|body_state| {
                // `body.orientation` is body-local → inertial; we want the
                // inverse to bring inertial deltas into the body-fixed frame.
                let world_to_body = body_state.orientation.inverse();
                let delta_inertial = camera_inertial - body_state.position;
                let delta_body = world_to_body * delta_inertial;
                let phase = bevy::math::DVec3::new(
                    delta_body.x.rem_euclid(period),
                    delta_body.y.rem_euclid(period),
                    delta_body.z.rem_euclid(period),
                );
                (phase.as_vec3(), world_to_body.as_quat().normalize())
            })
            .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
        mat.extras.debug.view_phase = Vec4::new(view_phase.x, view_phase.y, view_phase.z, 0.0);
        mat.extras.debug.world_to_body_rot = Vec4::new(
            world_to_body_q.x,
            world_to_body_q.y,
            world_to_body_q.z,
            world_to_body_q.w,
        );
        // Analytic vertex-stage pad flatten (the structural "structures render
        // above the ground" invariant — see `FlattenRegionGpu` in body_render).
        // Mirror this body's flatten pads into the material every frame so the
        // terrain vertex shader pins the rendered ground to the exact pad plane
        // at every LOD / morph / bake state. No handle or zero pads disables
        // the override (and the map terrain, updated elsewhere, never sets it).
        // `THALOS_FLATTEN_VERTEX=0` zeroes the block — a live A/B diagnostic
        // lever for attributing base-ground artifacts to this override vs the
        // baked tiles.
        static VERTEX_FLATTEN_ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            !matches!(
                std::env::var("THALOS_FLATTEN_VERTEX").as_deref(),
                Ok("0") | Ok("false") | Ok("off")
            )
        });
        if !*VERTEX_FLATTEN_ENABLED {
            mat.extras.flatten = FlattenBlock::default();
            continue;
        }
        mat.extras.flatten = flatten_registry
            .get(terrain.body_id)
            .and_then(|handle| {
                handle.read().ok().map(|regions| {
                    if regions.len() <= MAX_FLATTEN_REGIONS {
                        FlattenBlock::pack(regions.iter().map(|r| &r.flatten))
                    } else {
                        // More pads than uniform slots: keep those nearest the
                        // camera — only pads near the view can show LOD error.
                        let cam_dir_body = states
                            .get(terrain.body_id)
                            .map(|bs| bs.orientation.inverse() * (camera_inertial - bs.position))
                            .and_then(|v| v.try_normalize())
                            .unwrap_or(DVec3::Y);
                        let mut sorted: Vec<_> = regions.iter().collect();
                        sorted.sort_by(|a, b| {
                            b.flatten
                                .center_dir
                                .dot(cam_dir_body)
                                .total_cmp(&a.flatten.center_dir.dot(cam_dir_body))
                        });
                        FlattenBlock::pack(
                            sorted
                                .into_iter()
                                .take(MAX_FLATTEN_REGIONS)
                                .map(|r| &r.flatten),
                        )
                    }
                })
            })
            .unwrap_or_default();
    }

    for (sky, mat_handle) in &sky_q {
        let Some(extra) = sky_by_body.get(&sky.body_id) else {
            continue;
        };
        let Some(mut mat) = sky_materials.get_mut(mat_handle) else {
            continue;
        };
        mat.atmosphere_extra = *extra;
        // ADR-20260720T185958Z-water-projects-one-signed-sea-field: point the material at this body's live terrain entity so
        // its bind-group prepare can resolve the height atlas + tile tree in
        // the render world. Refreshed every frame, so terrain despawn/respawn
        // (residency tiers, flatten invalidation) can never leave it stale.
        mat.terrain_entity = tile_lookup_by_body
            .get(&sky.body_id)
            .map(|(entity, _, _)| *entity);
        // Sky-dome dev overrides (`< 0` keep the authored value).
        if tuning.strength >= 0.0 {
            mat.atmosphere.atmos_geom.z = tuning.strength;
        }
        if tuning.multi_scatter_gain >= 0.0 {
            mat.atmosphere.atmos_geom.w = tuning.multi_scatter_gain;
        }
        // Aerial-perspective decoupling: the shader multiplies the in-scatter on
        // terrain/geometry pixels by `cloud_band_radii.z`. We want the terrain
        // airlight to land at an *absolute* strength independent of the sky
        // dome, so divide by the effective sky strength (the in-scatter scales
        // linearly with `atmos_geom.z`). Result: ground haze stays at
        // `aerial_perspective_strength` on every body — proportional to its β,
        // not its authored sky brightness.
        let sky_strength = mat.atmosphere.atmos_geom.z.max(1.0e-3);
        mat.atmosphere_extra.cloud_band_radii.z =
            (tuning.aerial_perspective_strength / sky_strength).max(0.0);
        if let Some(clouds) = cache
            .environment
            .get(sky.body_id)
            .and_then(|env| env.cloud_bands.as_ref())
        {
            let p = clouds.phases;
            mat.atmosphere.cloud_bands_a =
                Vec4::new(p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32);
            mat.atmosphere.cloud_bands_b =
                Vec4::new(p[4] as f32, p[5] as f32, p[6] as f32, p[7] as f32);
            mat.atmosphere.cloud_bands_c =
                Vec4::new(p[8] as f32, p[9] as f32, p[10] as f32, p[11] as f32);
            mat.atmosphere.cloud_bands_d =
                Vec4::new(p[12] as f32, p[13] as f32, p[14] as f32, p[15] as f32);

            let scroll = clouds.scroll_rate_rad_s.abs();
            let period = if scroll > 1.0e-9 {
                std::f64::consts::TAU / scroll
            } else {
                86_400.0
            };
            mat.atmosphere.cloud_dynamics.y = sim.simulation.sim_time().rem_euclid(period) as f32;
        }
    }
}

/// Build a `SceneLighting` for one terrain body. Equivalent to
/// `build_scene_lighting` in `rendering::lighting`, but specialised so the
/// occluder vec is keyed by `BodyId` directly and uses the SHIP-frame
/// (1 m = 1 render unit) body grid positions cached above.
///
/// The star direction and flux are computed from the real (metre) ephemeris
/// positions, so they are scale-invariant: the map-view terrain
/// (`rendering::map_terrain`) reuses this with an empty occluder list (the map
/// schematic skips eclipse shadows) and a `MAP_SCALE` body.
pub(crate) fn build_terrain_scene_lighting(
    body_id: BodyId,
    states: &thalos_physics_canonical::types::BodyStates,
    occluders: &[(BodyId, Vec3, f32)],
    gain: f32,
) -> thalos_body_render::SceneLighting {
    use thalos_body_render::{MAX_ECLIPSE_OCCLUDERS, SceneLighting, StarLight};

    let mut scene = SceneLighting::default();
    let star_pos = states.first().map(|s| s.position).unwrap_or_default();
    let body_pos = states.get(body_id).map(|s| s.position).unwrap_or_default();
    let offset = star_pos - body_pos;
    let distance_m = offset.length();
    let to_star = if distance_m > 0.0 {
        (offset / distance_m).as_vec3()
    } else {
        Vec3::Y
    };
    let au_over_d = (AU_M / distance_m.max(1.0)) as f32;
    let flux = LIGHT_AT_1AU * au_over_d * au_over_d * gain;

    scene.star_count = 1;
    scene.stars[0] = StarLight {
        dir_flux: Vec4::new(to_star.x, to_star.y, to_star.z, flux),
        color: Vec4::new(1.0, 1.0, 1.0, 0.0),
    };

    let mut count = 0usize;
    for (other_id, pos, radius) in occluders {
        if *other_id == body_id {
            continue;
        }
        if count >= MAX_ECLIPSE_OCCLUDERS {
            break;
        }
        scene.occluders[count] = Vec4::new(pos.x, pos.y, pos.z, *radius);
        count += 1;
    }
    scene.occluder_count = count as u32;

    // Planetshine for moons could fill `scene.planetshine_*` here using the
    // parent's render-space center, radius, and the
    // `PlanetshineTints` resource. Skipped at this stage because the LOD
    // swap happens far enough from the body that planetshine contributes
    // little to the direct-light path; revisit when we drop the swap radius.
    scene
}

/// Moonlight contribution for one body's surface: the brightest child moon
/// reflecting the star back down, packed for `SceneLighting.moonlight_*`. This
/// is the reverse of planetshine (parent → moon) — a child moon lighting its
/// parent's night side, so a full Mira overhead lights the Thalos landscape.
///
/// Returns `(moonlight_dir_flux, moonlight_color)`:
///   - `dir_flux.xyz` = unit direction from the body toward the moon in world
///     render space (= the inertial direction; the big_space render frame shares
///     inertial axes, so a normalised inertial direction is already render-space).
///     `.w` = artistic flux folded with the moon's Lambert phase and a per-moon
///     relative-brightness from its size / albedo / distance.
///   - `color.xyz` = the moon's normalised linear hue, `.w` = enable flag.
///
/// Physical moonlight is ~1e-6 of sunlight — invisible after tonemapping — so
/// the flux is an artistic night-lift, not lux. `MOON_FULL_FLUX` is the single
/// brightness knob (the lift from a full, bright moon overhead); tune it from a
/// night screenshot.
fn compute_moonlight(
    body_id: BodyId,
    bodies: &[thalos_world::BodyDefinition],
    states: &thalos_physics_canonical::types::BodyStates,
    gain: f32,
) -> (Vec4, Vec4) {
    // Full-moon night lift in final surface-radiance units (~0..1). The shader
    // multiplies it by ground albedo × cosine × night/horizon gates, so ~0.12
    // lands clearly above the SURFACE_NIGHT_AMBIENT starlight floor (~0.01)
    // without reading as daylight.
    const MOON_FULL_FLUX: f32 = 0.12;
    // Reference "bright moon" reflectance shape (albedo_luminance × (R/d)²) that
    // maps to full brightness. Mira ≈ 0.14 × (8.69e5 / 1.91e8)² ≈ 2.9e-6.
    const MOON_REF_SHAPE: f64 = 3.0e-6;

    let none = (Vec4::ZERO, Vec4::ZERO);
    let Some(body_state) = states.get(body_id) else {
        return none;
    };
    let star_pos = states.first().map(|s| s.position).unwrap_or_default();
    let body_pos = body_state.position;

    let mut best_flux = 0.0f32;
    let mut best_dir = Vec3::Y;
    let mut best_tint = Vec3::ONE;

    for moon in bodies {
        if !matches!(moon.kind, thalos_world::BodyKind::Moon) || moon.parent != Some(body_id) {
            continue;
        }
        let Some(moon_state) = states.get(moon.id) else {
            continue;
        };
        let to_moon = moon_state.position - body_pos;
        let d = to_moon.length();
        if d <= 0.0 {
            continue;
        }
        // Lambert phase of the moon as seen from the body: the angle AT the moon
        // between the star and the body (full moon → 0 → phase 1; new moon → π → 0).
        let to_star_from_moon = (star_pos - moon_state.position).normalize_or_zero();
        let to_body_from_moon = (body_pos - moon_state.position).normalize_or_zero();
        let cos_g = to_star_from_moon.dot(to_body_from_moon).clamp(-1.0, 1.0);
        let g = cos_g.acos();
        let phase =
            ((g.sin() + (std::f64::consts::PI - g) * cos_g) / std::f64::consts::PI).clamp(0.0, 1.0);

        // Per-moon reflectance shape: albedo luminance × (angular radius)².
        let color_lin = Color::srgb(moon.color[0], moon.color[1], moon.color[2]).to_linear();
        let albedo_lum =
            (0.2126 * color_lin.red + 0.7152 * color_lin.green + 0.0722 * color_lin.blue) as f64;
        let ang = moon.radius_m / d;
        let shape = albedo_lum * ang * ang;
        let rel = (shape / MOON_REF_SHAPE).clamp(0.0, 1.5);

        let flux = MOON_FULL_FLUX * (phase * rel) as f32 * gain;
        if flux > best_flux {
            best_flux = flux;
            best_dir = (to_moon / d).as_vec3();
            // Normalise hue so flux carries brightness and the tint only colour.
            let max_c = color_lin
                .red
                .max(color_lin.green)
                .max(color_lin.blue)
                .max(1.0e-4);
            best_tint = Vec3::new(
                color_lin.red / max_c,
                color_lin.green / max_c,
                color_lin.blue / max_c,
            );
        }
    }

    if best_flux <= 0.0 {
        return none;
    }
    (
        Vec4::new(best_dir.x, best_dir.y, best_dir.z, best_flux),
        Vec4::new(best_tint.x, best_tint.y, best_tint.z, 1.0),
    )
}
