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
    BodyTerrainShadow, BodyWaterMaterial, BodyWaterParams, CASCADE_COUNT, GpuAtlasHeightMirrorComponent,
    GpuAtlasMirrorHandle, MAX_TERRAIN_SHADOW_CASTERS, MAX_TERRAIN_SHADOW_QUADS,
    PipelineTileProvider, SyntheticTerrainMode, SyntheticTileProvider, TerrainShadingStyle,
    rendered_height_range,
};
use thalos_physics_canonical::canonical::AuthorityMode;
use thalos_physics_canonical::types::VesselKind;
use thalos_shipyard::editor::EditorPart;
use thalos_shipyard::{
    Adapter, AirIntake, AttachNodes, CommandPod, Decoupler, Engine, EngineGeometry, FuelTank,
    Fuselage, JetNacelleMount, Part, PodGeometry, SurfaceMount, SurfaceMountKind, Wing,
    fuselage_skin_radius, fuselage_v_offset_at, host_mount_geometry, jet_nacelle_centers,
    jet_nacelle_length, wing_panel_frame,
};
use thalos_terrain::{
    FlattenHandle, FlattenedSurface, PlanetSurface, ProceduralSurface, SurfaceQuery, flatten_handle,
};
use thalos_world::{BodyDefinition, BodyId};

use std::collections::HashMap;

use super::SCREEN_MARKER_RADIUS;
use crate::camera::ShipCamera;
use crate::coords::SHIP_LAYER;
use crate::player_controller::{EvaMode, PlayerControllerState, PlayerControllerVisual};

use super::real_space::REAL_SPACE_CELL_SIZE_M;
use super::types::{CameraExposure, PlayerShip, RealSpaceBody, SimulationState, SolarSystemState};

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
}

const EVA_SHADOW_RADIUS_M: f32 = 0.42;
const EVA_SHADOW_HALF_LENGTH_M: f32 = 0.72;
const CRAFT_SHADOW_STRENGTH: f32 = 0.84;
const SHIP_SHADOW_MIN_PENUMBRA_M: f32 = 0.08;
const EVA_SHADOW_MIN_PENUMBRA_M: f32 = 0.06;
const CRAFT_SHADOW_MAX_DISTANCE_M: f32 = 350.0;

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

/// Marker on the per-body water sphere (ship layer). Only spawned when the
/// baked surface has `sea_level_m = Some(_)`; visibility is paired with
/// [`BodyTerrain`] so the impostor's inline water BRDF takes over outside
/// the LOD swap radius.
#[derive(Component, Debug)]
pub(super) struct BodyWater {
    pub(super) body_id: BodyId,
}

type ShadowPartQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static GlobalTransform,
        &'static AttachNodes,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Fuselage>,
        Option<&'static Wing>,
        Option<&'static Engine>,
        Option<&'static AirIntake>,
        Option<&'static SurfaceMount>,
    ),
    // The shipyard editor's build world shares this ECS; its parts sit near
    // the render origin and would cast phantom shadows onto the terrain.
    (With<Part>, Without<EditorPart>),
>;

#[derive(Clone, Copy)]
struct PartShadowShape {
    height: f32,
    radius_top: f32,
    radius_bottom: f32,
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
/// `crates/udlod/src/shaders/{vertex,functions}.wgsl` changes. Cross-tile
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

fn terrain_craft_shadow_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let Ok(value) = std::env::var("THALOS_TERRAIN_CRAFT_SHADOW") else {
            return true;
        };

        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => true,
            "1" | "on" | "true" | "yes" => true,
            "0" | "off" | "false" | "no" => false,
            other => {
                warn!("unknown THALOS_TERRAIN_CRAFT_SHADOW={other:?}; enabling craft shadow");
                true
            }
        }
    })
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
    .add_attachment(AttachmentConfig {
        name: "albedo".to_string(),
        texture_size: tc.tile_texture_size,
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        format: AttachmentFormat::Rgba8,
    })
    .add_attachment(AttachmentConfig {
        name: "roughness".to_string(),
        texture_size: tc.tile_texture_size,
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        // thalos_udlod has no single-channel 8-bit format; the source
        // cubemap is u8 and we upscale to u16 in the tile provider.
        format: AttachmentFormat::R16,
    })
    .add_attachment(AttachmentConfig {
        name: "material".to_string(),
        texture_size: tc.tile_texture_size,
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        // R = grass/vegetation, G = soil/peat, B = exposed rock, A = wetness.
        // These masks drive near-ground material blending; the albedo atlas
        // remains the macro/body-colour anchor for orbital continuity.
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
) -> Entity {
    let radius_m = body.radius_m as f32;
    let tc = tier.config();
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
    let terrain_surface: Arc<dyn SurfaceQuery> = Arc::new(FlattenedSurface::new(
        Arc::new(ProceduralSurface::new(body.radius_m as f32, body.id as u32)),
        flatten,
    ));
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
    // (`update_body_terrain_atmosphere`) only touches `craft_shadow`/`debug`
    // and leaves `inspection` alone.
    let shading_style = terrain_shading_style_for(body);
    let material = BodyTerrainMaterial {
        atmosphere,
        scene: SceneLighting::default(),
        extras: BodyTerrainExtras {
            craft_shadow: BodyTerrainShadow::default(),
            debug,
            inspection: Vec4::new(0.0, shading_style.shader_flag(), 0.0, 0.0),
            ..Default::default()
        },
        sun_shadow_map_0: sun_shadow_maps[0].clone(),
        sun_shadow_map_1: sun_shadow_maps[1].clone(),
        sun_shadow_map_2: sun_shadow_maps[2].clone(),
    };

    let mut terrain = commands.spawn((
        bundle,
        MeshMaterial3d(materials.add(material)),
        RenderLayers::layer(SHIP_LAYER),
        // The scene's `SunLight` cascade is sized for standard local
        // meshes. Ground terrain receives craft shadows through an
        // analytic proxy; rendering the planet itself into the shadow map
        // would be wasteful and noisy.
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
/// it still aliases at sub-metre stand-off.
const WATER_MESH_SUBDIVISIONS: u32 = 7;

/// Tiny offset (in metres) lifting the water mesh above the bare body
/// sphere. Resolves z-fighting between the water mesh and the seafloor
/// terrain mesh at iso-height texels in favour of water. Sized at 2 m so
/// it's comfortably above the f32 ULP near body-radius scale (~0.4 m at
/// 3 Mm) on the bodies we currently ship.
const WATER_SURFACE_EPSILON_M: f32 = 2.0;

/// Temporary kill switch for ground-LOD water. The impostor path still renders
/// its inline water BRDF outside the terrain handoff.
const TERRAIN_PATH_WATER_ENABLED: bool = false;

/// Default deep-water tint when the bake omits an explicit
/// `WaterAppearance`. Matches `PlanetWaterParams::from_static_surface`'s
/// fallback so the impostor and ground-LOD paths agree.
const FALLBACK_WATER_COLOR_DEPTH: [f32; 4] = [0.012, 0.040, 0.090, 120.0];

/// Spawn the per-body water sphere. No-op when the baked surface has no
/// `sea_level_m` (airless bodies); otherwise creates a single icosphere
/// entity parented to the body's grid, hidden at start so
/// `sync_body_render_lod` can flip it in step with [`BodyTerrain`].
// 0b-1: water is disabled until the generator grows a sea level (Slice 1);
// this is rewritten/removed then. Uncalled for now.
#[allow(dead_code)]
pub(crate) fn spawn_body_water(
    commands: &mut Commands,
    body: &BodyDefinition,
    surface: &PlanetSurface,
    ship_parent_entity: Entity,
    meshes: &mut Assets<Mesh>,
    water_materials: &mut Assets<BodyWaterMaterial>,
) -> Option<Entity> {
    if !TERRAIN_PATH_WATER_ENABLED {
        return None;
    }

    let baked = &surface.static_surface;
    let sea_level_m = baked.sea_level_m?;

    let water_radius_m = (body.radius_m as f32 + sea_level_m + WATER_SURFACE_EPSILON_M).max(1.0);

    let color_depth = baked
        .water_appearance
        .map(|w| {
            Vec4::new(
                w.color_depth[0],
                w.color_depth[1],
                w.color_depth[2],
                w.color_depth[3],
            )
        })
        .unwrap_or_else(|| {
            Vec4::new(
                FALLBACK_WATER_COLOR_DEPTH[0],
                FALLBACK_WATER_COLOR_DEPTH[1],
                FALLBACK_WATER_COLOR_DEPTH[2],
                FALLBACK_WATER_COLOR_DEPTH[3],
            )
        });

    let mesh = meshes.add(
        Sphere::new(water_radius_m)
            .mesh()
            .ico(WATER_MESH_SUBDIVISIONS)
            .expect("water mesh ico subdivision is within bevy's supported range"),
    );

    let params = BodyWaterParams {
        color_depth,
        // xyz populated each frame by `update_body_terrain_atmosphere` from
        // the body's render-space grid origin; w is constant.
        planet_center_radius: Vec4::new(0.0, 0.0, 0.0, water_radius_m),
        time: Vec4::ZERO,
    };
    let material = BodyWaterMaterial {
        scene: SceneLighting::default(),
        params,
    };

    let water_entity = commands
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(water_materials.add(material)),
            Transform::default(),
            Visibility::Hidden,
            RenderLayers::layer(SHIP_LAYER),
            NotShadowCaster,
            ChildOf(ship_parent_entity),
            Name::new(format!("{} Water", body.name)),
            BodyWater { body_id: body.id },
        ))
        .id();

    info!(
        "spawned water surface for '{}' (radius {:.0} km, sea level {:+.1} m)",
        body.name,
        body.radius_m / 1000.0,
        sea_level_m,
    );

    Some(water_entity)
}

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
            Without<BodyWater>,
        ),
    >,
    mut impostors: Query<
        (&RealSpaceImpostor, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<BodySky>,
            Without<BodyHalo>,
            Without<BodyWater>,
        ),
    >,
    mut waters: Query<
        (&BodyWater, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodySky>,
            Without<BodyHalo>,
        ),
    >,
    mut skies: Query<
        (&BodySky, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodyHalo>,
            Without<BodyWater>,
        ),
    >,
    mut halos: Query<
        (&BodyHalo, &mut Visibility),
        (
            Without<BodyTerrain>,
            Without<RealSpaceImpostor>,
            Without<BodySky>,
            Without<BodyWater>,
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

    // Water pairs with terrain: only visible inside the LOD swap radius
    // (and only if terrain is resident — water only spawns alongside
    // terrain, so this is mainly a safety check). Outside the swap radius
    // the impostor's inline water BRDF takes over.
    for (water, mut vis) in &mut waters {
        let Some((dist, swap, _shell)) = body_metrics(water.body_id) else {
            continue;
        };
        let resident = terrain_resident.contains(&water.body_id);
        let want = if resident && dist < swap {
            Visibility::Inherited
        } else {
            Visibility::Hidden
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

/// Update the per-frame dynamic data on every body's `BodyTerrainMaterial`,
/// `BodyWaterMaterial`, and `BodySkyMaterial`:
///
/// - terrain `planet_extra` (planet center + radius, terminator-wrap knob)
/// - terrain `scene` (`SceneLighting`: primary star, eclipse occluders,
///   ambient; planetshine left zero for now)
/// - water `scene` (same `SceneLighting` as terrain), water `params.time`
///   (`Time<Real>::elapsed_secs` for wave scroll), water `params.planet_center_radius.xyz`
///   (body render-space centre)
/// - sky `atmosphere_extra` (sun dir + flux, planet center + radius)
///
/// Must run after `sync_solar_system_state` (for sun direction from ephemeris),
/// `update_camera_exposure` (for exposure gain), and
/// `update_real_space_body_positions` (for up-to-date body grid transforms).
pub(super) fn update_body_terrain_atmosphere(
    body_q: Query<(&RealSpaceBody, &GlobalTransform)>,
    terrain_q: Query<(&BodyTerrain, &MeshMaterial3d<BodyTerrainMaterial>)>,
    water_q: Query<(&BodyWater, &MeshMaterial3d<BodyWaterMaterial>)>,
    sky_q: Query<(&BodySky, &MeshMaterial3d<BodySkyMaterial>)>,
    ship_q: Query<(), With<PlayerShip>>,
    eva_q: Query<&GlobalTransform, With<PlayerControllerVisual>>,
    part_q: ShadowPartQuery,
    ship_cam_q: Query<(&CellCoord, &Transform), With<ShipCamera>>,
    sim: Res<SimulationState>,
    cache: Res<SolarSystemState>,
    exposure: Res<CameraExposure>,
    time: Res<Time<Real>>,
    // Combined into one tuple param to stay under Bevy's 16-arg system limit:
    // .0 = live cloud render texture, .1 = cloud config (heights for occlusion),
    // .2 = per-pixel cloud-hit distance texture, .3 = which body the cloud
    // raymarch is currently rendered for (sole writer: `clouds::drive_clouds`),
    // .4 = blank fallbacks to rebind on bodies that are not the active one,
    // .5 = live atmosphere airlight tuning,
    // .6 = sun-shadow map handle + view_proj + params (folded in to stay under
    //      Bevy's 16-system-arg limit).
    cloud_io: (
        Option<Res<thalos_volumetric_clouds::CloudRenderTexture>>,
        Option<Res<thalos_volumetric_clouds::CloudsConfig>>,
        Option<Res<thalos_volumetric_clouds::CloudDistanceTexture>>,
        Res<super::clouds::ActiveCloudBody>,
        Option<Res<super::spawn::BlankCloudTextures>>,
        Res<AtmosphereTuning>,
        Res<super::sun_shadow::SunShadowState>,
    ),
    mut terrain_materials: ResMut<Assets<BodyTerrainMaterial>>,
    mut water_materials: ResMut<Assets<BodyWaterMaterial>>,
    mut sky_materials: ResMut<Assets<BodySkyMaterial>>,
) {
    let Some(ref states) = cache.states else {
        return;
    };
    let craft_shadow = if terrain_craft_shadow_enabled() {
        local_craft_shadow(&ship_q, &eva_q, &part_q)
    } else {
        BodyTerrainShadow::default()
    };

    let star_pos = states.first().map(|s| s.position).unwrap_or_default();
    // Atmosphere airlight tuning (see `AtmosphereTuning`).
    let tuning = &cloud_io.5;

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
        let cloud_band_radii = if Some(i) == cloud_io.3.0 {
            cloud_io
                .1
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
            },
        );
    }

    // Camera position in heliocentric inertial coords (f64), reconstructed from
    // the ship camera's big_space cell + local translation so it stays
    // f64-precise at planet radius. The flat-mode debug checker's `view_phase`
    // is the camera's body-fixed position mod-(2 × cell_size) per axis; it must
    // reference the *actual* render camera — the same `view.world_position` the
    // shader differences fragments against — not the craft. Using the craft
    // position slides the checker across the surface whenever the camera orbits
    // a stationary player, because the phase reference and the shader's camera
    // reference then disagree by the orbit offset.
    let camera_inertial = ship_cam_q
        .single()
        .map(|(cell, transform)| {
            DVec3::new(cell.x as f64, cell.y as f64, cell.z as f64) * REAL_SPACE_CELL_SIZE_M as f64
                + transform.translation.as_dvec3()
        })
        .unwrap_or_else(|_| sim.simulation.ship_state().position);
    for (terrain, mat_handle) in &terrain_q {
        let Some(mat) = terrain_materials.get_mut(mat_handle) else {
            continue;
        };
        mat.scene =
            build_terrain_scene_lighting(terrain.body_id, states, &occluders, exposure.gain);
        mat.extras.craft_shadow = craft_shadow;
        // Sun-shadow map: the camera + matrix are owned by `sun_shadow`; bind
        // the handle and the render-space → shadow-clip transform. `params.x`
        // is 0 when the pass is inactive (orbit / off-surface), so the shader
        // skips sampling entirely.
        let sun_shadow = &cloud_io.6;
        mat.sun_shadow_map_0 = sun_shadow.images[0].clone();
        mat.sun_shadow_map_1 = sun_shadow.images[1].clone();
        mat.sun_shadow_map_2 = sun_shadow.images[2].clone();
        mat.extras.shadow = sun_shadow.block;
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
    }

    // Use real (wall-clock) elapsed seconds for wave scroll — canonical sim
    // time pauses at warp-pause and ticks faster than wall time during warp,
    // both of which would make wave motion read as wrong against the visible
    // motion of the body itself.
    let wave_time = time.elapsed_secs();
    for (water, mat_handle) in &water_q {
        let Some(mat) = water_materials.get_mut(mat_handle) else {
            continue;
        };
        mat.scene = build_terrain_scene_lighting(water.body_id, states, &occluders, exposure.gain);
        let render_pos = body_render_pos
            .get(&water.body_id)
            .copied()
            .unwrap_or(Vec3::ZERO);
        // Preserve the spawn-time `.w` (water surface radius) and only
        // overwrite the centre + time fields.
        let water_radius = mat.params.planet_center_radius.w;
        mat.params.planet_center_radius =
            Vec4::new(render_pos.x, render_pos.y, render_pos.z, water_radius);
        mat.params.time.w = wave_time;
    }

    for (sky, mat_handle) in &sky_q {
        let Some(extra) = sky_by_body.get(&sky.body_id) else {
            continue;
        };
        let Some(mat) = sky_materials.get_mut(mat_handle) else {
            continue;
        };
        mat.atmosphere_extra = *extra;
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
        // Bind the live volumetric cloud + cloud-distance textures for the
        // active cloud body (the one `drive_clouds` is rendering); other
        // bodies keep the blank fallbacks so their atmosphere pass composites
        // no clouds.
        if Some(sky.body_id) == cloud_io.3.0 {
            if let Some(ref cr) = cloud_io.0 {
                mat.cloud_layer = cr.handle.clone();
            }
            if let Some(ref cd) = cloud_io.2 {
                mat.cloud_distance = cd.handle.clone();
            }
        } else if let Some(ref blanks) = cloud_io.4 {
            mat.cloud_layer = blanks.layer.clone();
            mat.cloud_distance = blanks.distance.clone();
        }
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

fn local_craft_shadow(
    ship_q: &Query<(), With<PlayerShip>>,
    eva_q: &Query<&GlobalTransform, With<PlayerControllerVisual>>,
    part_q: &ShadowPartQuery,
) -> BodyTerrainShadow {
    if ship_q.single().is_ok() {
        let mut shadow = BodyTerrainShadow::default();
        let mut count = 0usize;
        let mut quad_count = 0usize;
        for (xform, nodes, pod, dec, adapter, tank, fuselage, wing, engine, intake, surface_mount) in
            part_q.iter()
        {
            // Aircraft-shaped parts get casters derived from the same shared
            // shipyard geometry the renderer lofts, so the shadow silhouette
            // tracks the visible craft by construction.
            if let (Some(wing), Some(mount)) = (wing, surface_mount) {
                push_wing_quad(&mut shadow, &mut quad_count, xform, wing, mount, part_q);
                continue;
            }
            if let Some(fus) = fuselage {
                push_fuselage_casters(&mut shadow, &mut count, xform, fus, nodes);
                continue;
            }
            if let (Some(engine), Some(mount)) = (engine, surface_mount)
                && engine.geometry == EngineGeometry::JetNacelle
                && mount.kind == SurfaceMountKind::WingPylon
            {
                push_nacelle_caster(&mut shadow, &mut count, xform, engine, mount, part_q);
                continue;
            }
            let Some(shape) = part_shadow_shape(nodes, pod, dec, adapter, tank, engine, intake)
            else {
                continue;
            };
            let transform = xform.compute_transform();
            let height_scale = transform.scale.y.abs().max(1.0e-4);
            let radius_scale = transform
                .scale
                .x
                .abs()
                .max(transform.scale.z.abs())
                .max(1.0e-4);
            let top = transform.translation;
            let bottom = transform.translation
                + transform.rotation * Vec3::new(0.0, -shape.height * height_scale, 0.0);
            push_shadow_caster(
                &mut shadow,
                &mut count,
                top,
                shape.radius_top * radius_scale,
                bottom,
                shape.radius_bottom * radius_scale,
            );
        }

        if count > 0 || quad_count > 0 {
            shadow.params = Vec4::new(
                CRAFT_SHADOW_STRENGTH,
                SHIP_SHADOW_MIN_PENUMBRA_M,
                CRAFT_SHADOW_MAX_DISTANCE_M,
                count as f32,
            );
            shadow.quad_params.x = quad_count as f32;
            return shadow;
        }
    }

    if let Ok(eva_xform) = eva_q.single() {
        let eva_transform = eva_xform.compute_transform();
        let axis = (eva_transform.rotation * Vec3::Y).normalize_or_zero();
        let axis = if axis.length_squared() > 0.0 {
            axis
        } else {
            Vec3::Y
        };
        let center = eva_transform.translation;
        let mut shadow = BodyTerrainShadow::default();
        let mut count = 0usize;
        push_shadow_caster(
            &mut shadow,
            &mut count,
            center + axis * EVA_SHADOW_HALF_LENGTH_M,
            EVA_SHADOW_RADIUS_M,
            center - axis * EVA_SHADOW_HALF_LENGTH_M,
            EVA_SHADOW_RADIUS_M,
        );
        shadow.params = Vec4::new(
            CRAFT_SHADOW_STRENGTH,
            EVA_SHADOW_MIN_PENUMBRA_M,
            CRAFT_SHADOW_MAX_DISTANCE_M,
            count as f32,
        );
        return shadow;
    }

    BodyTerrainShadow::default()
}

/// Largest |scale| axis of a part transform, for scaling caster radii the way
/// `GlobalTransform::transform_point` scales the endpoint positions.
fn max_abs_scale(xform: &GlobalTransform) -> f32 {
    let scale = xform.compute_transform().scale;
    scale.x.abs().max(scale.y.abs()).max(scale.z.abs()).max(1.0e-4)
}

/// Skin radius a surface-mounted part sits at on its host — the same
/// `host_mount_geometry` lookup `ship_view` feeds the mesh builders, so the
/// caster frame matches the rendered loft. Falls back to a plain cylinder
/// radius when the host can't be resolved (mid-despawn).
fn shadow_host_mount_radius(part_q: &ShadowPartQuery, mount: &SurfaceMount) -> f32 {
    match part_q.get(mount.parent) {
        Ok((_, nodes, _, _, _, _, fuselage, _, _, _, _)) => {
            let top_d = nodes.get("top").map(|n| n.diameter).unwrap_or(2.0);
            host_mount_geometry(fuselage, top_d, mount.station, mount.angle).0
        }
        Err(_) => 1.0,
    }
}

/// One thin planform quad per lifting surface: corners at the root/tip
/// leading and trailing edges from the same `wing_panel_frame` the mesh is
/// lofted in. The shader projects the quad along the sun ray, so the shadow
/// is the true planform at any sun elevation — sweep, taper, and dihedral
/// included — and collapses to (nearly) nothing edge-on instead of the
/// chord-thick slab a capsule proxy throws at low sun.
fn push_wing_quad(
    shadow: &mut BodyTerrainShadow,
    quad_count: &mut usize,
    xform: &GlobalTransform,
    wing: &Wing,
    mount: &SurfaceMount,
    part_q: &ShadowPartQuery,
) {
    if *quad_count >= MAX_TERRAIN_SHADOW_QUADS {
        return;
    }
    let parent_radius = shadow_host_mount_radius(part_q, mount);
    let frame = wing_panel_frame(wing, mount.angle, parent_radius);
    let root_half = frame.fore_dir * (wing.root_chord * 0.5);
    let tip_half = frame.fore_dir * (wing.tip_chord * 0.5);
    // Wound root-LE -> tip-LE -> tip-TE -> root-TE so consecutive corners
    // trace the outline (the shader's edge test relies on the order).
    let corners = [
        frame.root_center + root_half,
        frame.tip_center + tip_half,
        frame.tip_center - tip_half,
        frame.root_center - root_half,
    ];
    let world = corners.map(|c| xform.transform_point(c));
    shadow.quad_a[*quad_count] = world[0].extend(0.0);
    shadow.quad_b[*quad_count] = world[1].extend(0.0);
    shadow.quad_c[*quad_count] = world[2].extend(0.0);
    shadow.quad_d[*quad_count] = world[3].extend(0.0);
    *quad_count += 1;
}

/// Three tapered segments tracing the fuselage loft — nose cap, barrel, tail
/// cone — sampled from the same skin model the mesh is lofted from, so a
/// pointed radome shadows to a point and an upswept tailcone lifts off the
/// ground line. The part origin is the `top` attach node; the loft runs down
/// local −Y (see `fuselage_mesh` frame docs).
fn push_fuselage_casters(
    shadow: &mut BodyTerrainShadow,
    count: &mut usize,
    xform: &GlobalTransform,
    fus: &Fuselage,
    nodes: &AttachNodes,
) {
    let diameter = nodes.get("top").map(|n| n.diameter).unwrap_or(fus.max_width);
    let nose_end = fus.nose_fraction.clamp(0.0, 0.49);
    let tail_start = (1.0 - fus.tail_fraction.clamp(0.0, 0.95)).max(nose_end);
    let stations = [0.0, nose_end, tail_start, 1.0];
    let scale = max_abs_scale(xform);
    let point_at = |s: f32| {
        xform.transform_point(Vec3::new(
            0.0,
            -s * fus.length,
            fuselage_v_offset_at(fus, diameter, s),
        ))
    };
    // Planform half-width: the skin radius along the lateral (+X) radial.
    let radius_at =
        |s: f32| fuselage_skin_radius(fus, diameter, s, std::f32::consts::FRAC_PI_2) * scale;
    for window in stations.windows(2) {
        if window[1] - window[0] < 1.0e-3 {
            continue;
        }
        push_shadow_caster(
            shadow,
            count,
            point_at(window[0]),
            radius_at(window[0]),
            point_at(window[1]),
            radius_at(window[1]),
        );
    }
}

/// One segment along each podded nacelle's axis, placed by the same
/// `jet_nacelle_centers` math that builds the pylon mesh (the mesh is in the
/// engine part's local frame, so the part transform places both alike).
fn push_nacelle_caster(
    shadow: &mut BodyTerrainShadow,
    count: &mut usize,
    xform: &GlobalTransform,
    engine: &Engine,
    mount: &SurfaceMount,
    part_q: &ShadowPartQuery,
) {
    let Ok((_, _, _, _, _, _, _, wing, _, _, wing_mount)) = part_q.get(mount.parent) else {
        return;
    };
    let (Some(wing), Some(wing_mount)) = (wing, wing_mount) else {
        return;
    };
    let nacelle_mount = JetNacelleMount {
        wing,
        wing_mount_angle: wing_mount.angle,
        parent_radius: shadow_host_mount_radius(part_q, wing_mount),
        span_fraction: mount.station,
        chord_fraction: mount.angle,
    };
    let half_length = jet_nacelle_length(engine) * 0.5;
    let scale = max_abs_scale(xform);
    let radius = engine.diameter * 0.5 * scale;
    for center in jet_nacelle_centers(engine, nacelle_mount) {
        push_shadow_caster(
            shadow,
            count,
            xform.transform_point(center + Vec3::Y * half_length),
            radius,
            xform.transform_point(center - Vec3::Y * half_length),
            radius,
        );
    }
}

fn part_shadow_shape(
    nodes: &AttachNodes,
    pod: Option<&CommandPod>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    engine: Option<&Engine>,
    intake: Option<&AirIntake>,
) -> Option<PartShadowShape> {
    if let Some(pod) = pod {
        // An inline cockpit has no body mesh of its own (`visual_spec`
        // returns None) — it must not cast a phantom shadow either.
        if matches!(pod.geometry, PodGeometry::Inline) {
            return None;
        }
        let (radius_top, radius_bottom, height) =
            thalos_shipyard::pod_visual_profile(pod.diameter, pod.geometry);
        Some(PartShadowShape {
            height,
            radius_top,
            radius_bottom,
        })
    } else if dec.is_some() {
        let radius = nodes.get("top").map(|n| n.diameter * 0.5).unwrap_or(0.5);
        Some(PartShadowShape {
            height: 0.2,
            radius_top: radius,
            radius_bottom: radius,
        })
    } else if let Some(adapter) = adapter {
        let top_diameter = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let bottom_diameter = adapter.target_diameter;
        Some(PartShadowShape {
            height: ((top_diameter + bottom_diameter) * 0.5).max(0.4),
            radius_top: top_diameter * 0.5,
            radius_bottom: bottom_diameter * 0.5,
        })
    } else if let Some(tank) = tank {
        let radius = nodes.get("top").map(|n| n.diameter * 0.5).unwrap_or(0.5);
        Some(PartShadowShape {
            height: tank.length,
            radius_top: radius,
            radius_bottom: radius,
        })
    } else if let Some(intake) = intake {
        Some(PartShadowShape {
            height: intake.length,
            radius_top: intake.diameter * 0.5,
            radius_bottom: intake.diameter * 0.5,
        })
    } else {
        engine.map(|engine| PartShadowShape {
            height: match engine.geometry {
                EngineGeometry::RocketBell => engine.diameter * 0.9,
                EngineGeometry::JetNacelle => thalos_shipyard::jet_nacelle_length(engine),
            },
            radius_top: match engine.geometry {
                EngineGeometry::RocketBell => engine.diameter * 0.35,
                EngineGeometry::JetNacelle => engine.diameter * 0.5,
            },
            radius_bottom: engine.diameter * 0.5,
        })
    }
}

fn push_shadow_caster(
    shadow: &mut BodyTerrainShadow,
    count: &mut usize,
    a: Vec3,
    radius_a: f32,
    b: Vec3,
    radius_b: f32,
) {
    if *count >= MAX_TERRAIN_SHADOW_CASTERS {
        return;
    }

    shadow.caster_a_radius[*count] = Vec4::new(a.x, a.y, a.z, radius_a.max(0.0));
    shadow.caster_b_radius[*count] = Vec4::new(b.x, b.y, b.z, radius_b.max(0.0));
    *count += 1;
}
