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
    BodyTerrainShadow, BodyWaterMaterial, BodyWaterParams, GpuAtlasHeightMirrorComponent,
    GpuAtlasMirrorHandle, MAX_TERRAIN_SHADOW_CASTERS, PipelineTileProvider, SyntheticTerrainMode,
    SyntheticTileProvider, rendered_height_range,
};
use thalos_physics_canonical::canonical::AuthorityMode;
use thalos_physics_canonical::types::VesselKind;
use thalos_shipyard::{
    Adapter, AirIntake, AttachNodes, CommandPod, Decoupler, Engine, EngineGeometry, FuelTank, Part,
    SurfaceMount, SurfaceMountKind,
};
use thalos_terrain::{BakedSurface, DynamicSurfaceState, PlanetSurface, SurfaceQuery};
use thalos_world::{BodyDefinition, BodyId};

use crate::camera::ShipCamera;
use crate::coords::SHIP_LAYER;
use crate::player_controller::{EvaMode, PlayerControllerState, PlayerControllerVisual};

use super::real_space::REAL_SPACE_CELL_SIZE_M;
use super::types::{CameraExposure, PlayerShip, RealSpaceBody, SimulationState, SolarSystemState};

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
/// The production provider currently evaluates expensive oceanic terrain on
/// CPU. Keep only a small number of provider tasks active so tile streaming
/// does not starve Bevy/rendering/input while the sim is paused.
const TILE_LOAD_SLOTS: u32 = 4;
const TILE_LOAD_QUEUE_SIZE: u32 = TILE_LOAD_SLOTS * 4;

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
        Option<&'static Engine>,
        Option<&'static AirIntake>,
        Option<&'static SurfaceMount>,
    ),
    With<Part>,
>;

#[derive(Clone, Copy)]
struct PartShadowShape {
    height: f32,
    radius_top: f32,
    radius_bottom: f32,
}

/// Camera-to-body-centre distance, expressed as a multiple of body radius,
/// at which the impostor billboard hands off to the ground-LOD terrain.
///
/// 4× radius keeps the impostor active while the body covers less than ~28°
/// of the view (its silhouette is well-defined as a disc); below 4× the body
/// fills enough of the frame that the 3-D mesh + heightfield reads better.
///
/// Both halves of the swap are still using the same baked cubemap data, so
/// at the threshold the visible texture matches; the discontinuity is in
/// projection (flat billboard ↔ 3-D mesh) and lighting (impostor PBR ↔ flat
/// albedo until M4 wires lighting through the terrain shader).
///
/// Smooth crossfade requires opacity uniforms on `PlanetMaterial` and
/// `BodyTerrainMaterial`; deferred to M4 along with the rest of the
/// terrain-side PBR work.
const TERRAIN_HANDOFF_RADIUS_FACTOR: f32 = 4.0;

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
fn body_terrain_view_config(body_radius_m: f64) -> TerrainViewConfig {
    TerrainViewConfig {
        morph_range: TERRAIN_MORPH_RANGE,
        blend_range: TERRAIN_BLEND_RANGE,
        precision_threshold_distance: TERRAIN_PRECISION_THRESHOLD_M / body_radius_m.max(1.0),
        ..TerrainViewConfig::default()
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
pub(crate) fn spawn_body_terrain(
    commands: &mut Commands,
    body: &BodyDefinition,
    surface: Arc<PlanetSurface>,
    ship_parent_entity: Entity,
    materials: &mut Assets<BodyTerrainMaterial>,
    tile_trees: &mut TerrainViewComponents<TileTree>,
    ship_camera: Entity,
    atmosphere: AtmosphereBlock,
    dynamic_state: DynamicSurfaceState,
    height_mirror: Option<GpuAtlasMirrorHandle>,
) -> Entity {
    let radius_m = body.radius_m as f32;
    // The construction site is the one place that names the concrete generation
    // type: wrap it as `Arc<dyn SurfaceQuery>` so the provider and the
    // height-range query see only the black-box seam.
    let terrain_surface: Arc<dyn SurfaceQuery> =
        Arc::new(BakedSurface::new(surface, dynamic_state));
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

    let config = TerrainConfig {
        lod_count: LOD_COUNT,
        model,
        atlas_size: ATLAS_SIZE,
        max_concurrent_tile_loads: TILE_LOAD_SLOTS,
        max_queued_tile_loads: TILE_LOAD_QUEUE_SIZE,
        ..Default::default()
    }
    .add_attachment(AttachmentConfig {
        name: "height".to_string(),
        texture_size: TILE_TEXTURE_SIZE,
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
        texture_size: TILE_TEXTURE_SIZE,
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        format: AttachmentFormat::Rgba8,
    })
    .add_attachment(AttachmentConfig {
        name: "roughness".to_string(),
        texture_size: TILE_TEXTURE_SIZE,
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        // thalos_udlod has no single-channel 8-bit format; the source
        // cubemap is u8 and we upscale to u16 in the tile provider.
        format: AttachmentFormat::R16,
    })
    .add_attachment(AttachmentConfig {
        name: "material".to_string(),
        texture_size: TILE_TEXTURE_SIZE,
        border_size: TILE_BORDER_SIZE,
        mip_level_count: TILE_MIP_LEVELS,
        // R = grass/vegetation, G = soil/peat, B = exposed rock, A = wetness.
        // These masks drive near-ground material blending; the albedo atlas
        // remains the macro/body-colour anchor for orbital continuity.
        format: AttachmentFormat::Rgba8,
    });

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
    // Pin LOD 0 on every cube face for the atlas's entire lifetime. This
    // gives `get_best_tile` a guaranteed resident ancestor for any tile
    // coordinate, so the GPU shader never samples `INVALID_ATLAS_INDEX`
    // (which decodes to 0 → `min_height` → vertex sits at
    // `-terrain_height_range` below the ellipsoid: visible "void" holes
    // wherever a draw tile has no streamed ancestor).
    //
    // `is_spherical` is true for body terrain (`TerrainModel::sphere`),
    // giving six faces; the planar fallback (single side) is harmless to
    // pin defensively.
    let side_count = if tile_atlas.model().is_spherical() {
        6
    } else {
        1
    };
    for side in 0..side_count {
        tile_atlas.pin_tile(TileCoordinate::new(side, 0, 0, 0));
    }
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
    let material = BodyTerrainMaterial {
        atmosphere,
        scene: SceneLighting::default(),
        extras: BodyTerrainExtras {
            craft_shadow: BodyTerrainShadow::default(),
            debug,
            inspection: Vec4::ZERO,
        },
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
        "spawned ground terrain for '{}' (provider {}, radius {:.0} km, height range ±{:.0} m, atlas size {})",
        body.name,
        provider_mode.label(),
        radius_m / 1000.0,
        terrain_height_range,
        ATLAS_SIZE,
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
/// * **Surface LOD** — `dist < TERRAIN_HANDOFF_RADIUS_FACTOR × radius`
///   selects the 3-D terrain mesh; otherwise the flat impostor billboard.
///   Logically the same body at two projections; never draw both.
/// * **Atmosphere LOD** — terrain uses `BodySky`, impostor uses `BodyHalo`.
///   `BodySky` covers every screen pixel and clips at scene depth, which is
///   what gives ground LOD terrain aerial perspective and the detached cloud
///   shell. `BodyHalo` covers only the billboard shell silhouette, matching
///   the impostor body's inline atmosphere/cloud composite.
///
/// Three altitude bands fall out for a body with an atmosphere:
///
/// | distance           | surface  | atmosphere |
/// |--------------------|----------|------------|
/// | `d < 4 × radius`   | terrain  | BodySky    |
/// | `d ≥ 4 × radius`   | impostor | halo       |
///
/// Airless bodies skip the BodySky entity entirely (never spawned) and
/// the halo's shader early-outs to a no-op; we still toggle its visibility
/// for consistency.
///
/// **Terrain residency gate.** Ground-LOD terrain entities are spawned
/// lazily by [`crate::rendering::terrain_residency`] only for bodies the
/// player is plausibly going to encounter. For a body without a resident
/// `BodyTerrain` entity, the impostor stays visible at all distances and
/// the halo stays visible at all distances — otherwise the body would
/// silently vanish when the camera passed inside `4 × radius` with no
/// terrain to take over.
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
    // the body silently disappears inside `4 × radius`. The same applies
    // briefly at spawn while LOD 0 streams: until the pinned tiles are
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
    // None if the body or its render-space position is missing. The shell
    // radius is kept here for diagnostics / future atmosphere-specific LOD
    // tuning; today's visibility choice keys atmosphere to the surface LOD
    // so terrain always gets the fullscreen in-front pass.
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
        Some((
            dist,
            TERRAIN_HANDOFF_RADIUS_FACTOR * radius,
            radius + karman,
        ))
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
        // we're showing the impostor: outside `4 × radius`, OR at any
        // distance if terrain isn't resident.
        let resident = terrain_resident.contains(&halo.body_id);
        let want = if !resident || dist >= swap {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        set_vis(&mut vis, want);
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
    // .0 = live cloud render texture, .1 = cloud config (heights for occlusion).
    cloud_io: (
        Option<Res<thalos_volumetric_clouds::CloudRenderTexture>>,
        Option<Res<thalos_volumetric_clouds::CloudsConfig>>,
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
        // Cloud band radii (render units) for the body the player is at, so the
        // sky pass can keep the cloud layer from painting over closer geometry.
        let cloud_band_radii = if i == sim.system.homeworld_id {
            cloud_io
                .1
                .as_ref()
                .map(|cfg| {
                    Vec4::new(
                        planet_radius + cfg.clouds_bottom_height,
                        planet_radius + cfg.clouds_top_height,
                        0.0,
                        0.0,
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
        // Bind the live volumetric cloud texture for the body the player is at
        // (driven relative to the homeworld); other bodies keep the blank
        // fallback so their atmosphere pass composites no clouds.
        if sky.body_id == sim.system.homeworld_id {
            if let Some(ref cr) = cloud_io.0 {
                mat.cloud_layer = cr.handle.clone();
            }
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
fn build_terrain_scene_lighting(
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
        for (xform, nodes, pod, dec, adapter, tank, engine, intake, surface_mount) in part_q.iter()
        {
            if matches!(
                (engine, surface_mount.map(|m| m.kind)),
                (Some(engine), Some(SurfaceMountKind::WingPylon))
                    if engine.geometry == EngineGeometry::JetNacelle
            ) {
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

        if count > 0 {
            shadow.params = Vec4::new(
                CRAFT_SHADOW_STRENGTH,
                SHIP_SHADOW_MIN_PENUMBRA_M,
                CRAFT_SHADOW_MAX_DISTANCE_M,
                count as f32,
            );
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
