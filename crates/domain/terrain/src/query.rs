//! Query API — the seam between terrain generation and its consumers.
//!
//! This is P0 of the planet-generation pipeline migration
//! ([docs/archive/planet-generation-pipeline-migration.md]). It establishes a single
//! contract every renderer and the physics collider evaluate the surface
//! through, so there is exactly **one geometric surface** ("one synth") shared
//! across the impostor, the UDLOD ground tiles, and the collider — replacing
//! the divergent cascades that previously lived in [`crate::sample`] (PNG
//! dumps) and `thalos_terrain_render::pipeline` (the game surface).
//!
//! ## Why this lives here
//!
//! The unified surface evaluator used to live in the Bevy crate
//! `thalos_terrain_render`. The spec puts the generation pipeline in the
//! pure-Rust `thalos_terrain` crate, so the evaluator moved here. The seam
//! evaluates **by direction** (`sample(dir, lod_m)`), which keeps this crate
//! both Bevy-free and cube-sphere-mapping-agnostic: the canonical tiling
//! mapping stays UDLOD's, owned by the consumer, and this crate's
//! [`crate::cubemap::Cubemap`] is only an internal storage detail.
//!
//! ## What backs it
//!
//! Today the default backing is the baked [`PlanetSurface`] plus the legacy
//! analytic detail cascade in this module ([`surface_sample`] /
//! [`BakedSurface`]). P2A `GenericTerrestrial` bodies opt into a direct runtime
//! evaluator for their smooth continental field, using the same terrain
//! function that produced the bake. Runtime ground geometry currently samples
//! that evaluator at full detail regardless of tile LOD to avoid parent/child
//! handoff contouring during the vertical slice. Later migration phases (field-DAG intent
//! layer + two-band detail stage) swap more backings behind the [`SurfaceQuery`]
//! trait without consumers noticing.
//!
//! ## Reserved for later phases
//!
//! `query_features`, `query_scatter`, and the full Tile contract (4-channel
//! material splat + macro-albedo modulation, spec §9) land in P2 alongside
//! the material-model change. They are intentionally omitted here rather than
//! stubbed with throwaway types; [`SurfaceQuery`] gains them as
//! default-method additions (backward-compatible) when their types exist.

use std::sync::{Arc, RwLock};

use glam::{DVec3, Vec3};

use crate::canopy::CanopyClimate;
use crate::cubemap::{Cubemap, dir_to_face_uv};
use crate::feature_compositor::{compose_runtime_features_m, runtime_feature_height_margin_m};
use crate::generic_terrestrial_field::{AirlessRegolithParams, RuntimeTerrainDetail};
use crate::sample::apply_dynamic_surface_layers;
use crate::static_surface::PlanetSurface;
use crate::types::DynamicSurfaceState;

// ---------------------------------------------------------------------------
// Detail-cascade tuning constants (moved verbatim from
// thalos_terrain_render::pipeline so the game surface is bit-identical).
// ---------------------------------------------------------------------------

/// Hash seed for the high-frequency detail noise. Decoupled from the body
/// generator's seed so changing terrain gen doesn't reshuffle ground-LOD
/// detail and vice versa.
const DETAIL_NOISE_SEED: u32 = 0x1E_E0_57_07;

/// Base (octave 0) wavelength of the HMF cascade, in metres.
const DETAIL_BASE_WL_M: f32 = 1000.0;

/// Peak amplitude of the HMF height contribution, in metres. HMF output is
/// normalised to `[0, 1]`, so the additive contribution to macro height stays
/// in `[0, DETAIL_AMP_M]`.
const DETAIL_AMP_M: f32 = 250.0;

const DETAIL_PERSISTENCE: f32 = 0.5;
const DETAIL_LACUNARITY: f32 = 2.0;

/// Musgrave ridged-multifractal offset.
const DETAIL_OFFSET: f32 = 1.0;

/// Cascade depth at the finest LOD. Eleven octaves from a 1 km base bottoms
/// out at `1 km / 2^11 ≈ 0.49 m`.
const MAX_DETAIL_OCTAVES: f32 = 11.0;

/// Hash seed for the domain-warp vector field. Independent from the HMF seed.
const WARP_NOISE_SEED: u32 = 0x77_C0_DE_42;

/// Wavelength of the warp field's octave 0, in metres.
const WARP_WAVELENGTH_M: f32 = 4000.0;

/// Maximum positional displacement of the domain warp, in metres.
const WARP_AMP_M: f32 = 800.0;

const WARP_OCTAVES: u32 = 2;

/// Additional height-range margin reserved for procedural detail. Matches
/// [`DETAIL_AMP_M`] so the R16 quantisation has room for the full positive HMF
/// contribution above the static + dynamic envelope.
const DETAIL_HEIGHT_MARGIN_M: f32 = DETAIL_AMP_M;

/// Minimum below-sea-level margin (m) preserved after capping detail uplift.
const SEA_LEVEL_CAP_EPSILON_M: f32 = 0.5;

// ---------------------------------------------------------------------------
// Sample type
// ---------------------------------------------------------------------------

/// One evaluated surface sample.
///
/// `height_m` is the band-limited geometric surface shared by the render mesh
/// and the physics collider. The remaining fields are shading channels. (The
/// spec's full Tile contract adds material splat weights and macro-albedo
/// modulation in P2; today's consumers read the channels below.)
#[derive(Debug, Clone, Copy)]
pub struct SurfaceSample {
    /// Height above the reference sphere, in metres.
    pub height_m: f32,
    /// Linear-space albedo.
    pub albedo_linear: Vec3,
    /// PBR roughness, 0..1.
    pub roughness: f32,
    /// Macro landcover moisture in `[-1, 1]` (+ wet, − dry) — the planet-scale
    /// f64 field the terrain shader's wrapped fine noise modulates
    /// (docs/world/terrain_macro.md). Baked into the tile albedo attachment's alpha
    /// channel. `0.0` for backings without a landcover model.
    pub moisture: f32,
}

/// Canonical landcover inputs for material-layer selection (NTR-X4).
///
/// Produced by the same macro band evaluation that produces
/// [`SurfaceSample::albedo_linear`] — same climate model, same fields — so a
/// material shader's treeline, snow line, and forest grain can never drift
/// from the palette. Zero for backings without a landcover model (airless
/// bodies, plain oceans).
///
/// The altitude bands travel as the **ecological altitude** they are compared
/// against rather than as pre-collapsed weights: one interpolated scalar then
/// yields *every* altitude line (treeline and snowline both, plus whatever a
/// later layer needs), it interpolates linearly across a triangle where a
/// smoothstepped weight does not, and the thresholds are already mirrored in
/// `thalos::landcover` for the shader side. The moisture-driven forest weight
/// has no such altitude form and stays a weight.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MaterialBands {
    /// Climate-shifted altitude in metres: geometric height plus the
    /// latitude cold lift (`procedural::climate_cold_lift_m`). Compare
    /// against `procedural::{ALPINE_LO_M, ALPINE_HI_M, SNOWLINE_LO_M,
    /// SNOWLINE_HI_M}` — or their `thalos::landcover` mirrors — for the
    /// treeline / snowline the macro palette actually paints.
    pub eco_altitude_m: f32,
    /// Absolute **canopy coverage** in `[0, 1]` — the single canopy authority
    /// (see [`crate::canopy`]): climate envelope × stand structure, confined to
    /// the lowland band (the treeline is already inside it).
    ///
    /// This is the exact weight the macro albedo bake mixed its dark canopy
    /// anchor at, which is what lets a material shader both drive aerial canopy
    /// grain from it *and* algebraically un-mix the anchor to recover understory
    /// colour. It is also what [`SurfaceQuery::canopy_coverage`] returns, so
    /// vegetation placement and the ground palette cannot disagree about where
    /// the forest is.
    pub canopy: f32,
}

// ---------------------------------------------------------------------------
// Query trait + region
// ---------------------------------------------------------------------------

/// A spherical-cap region on the body, for range queries and pre-warming.
#[derive(Debug, Clone, Copy)]
pub struct Region {
    /// Unit direction at the cap centre.
    pub center: Vec3,
    /// Angular radius of the cap, in radians.
    pub angular_radius_rad: f32,
}

/// One quadtree patch on the canonical cube-sphere.
///
/// This is deliberately a terrain-domain address rather than a renderer tile
/// key. Providers whose authored data is hierarchical can use it to answer
/// refinement questions without making [`SurfaceQuery`] depend on a rendering
/// crate. Face order matches [`crate::cubemap::CubemapFace`]; `y = 0` is the
/// top edge of the face, as in the package height pyramid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfacePatch {
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

/// The seam every consumer evaluates the surface through.
///
/// Consumers hold a `&dyn SurfaceQuery` so the backing implementation can be
/// swapped (baked surface today; field-DAG + detail stage later) without
/// touching call sites. The geometric surface (`sample`/`sample_height_m`) is
/// the single source of truth shared by the render mesh and the collider.
pub trait SurfaceQuery: Send + Sync {
    /// Evaluate the surface at unit direction `dir`, at `lod_m` metres per
    /// sample. Deterministic: equal `(dir, lod_m)` yield equal samples
    /// regardless of caller or order.
    fn sample(&self, dir: Vec3, lod_m: f32) -> SurfaceSample;

    /// Evaluate the surface at an **f64** direction. The precision-critical
    /// render path (per-tile-pixel synthesis at planet scale) must not downcast
    /// the direction to f32 before multiplying by the body radius, or the
    /// sample position quantises to a ~0.25 m body-local lattice and terraces
    /// the ground on foot. The default upcasts the f32 [`Self::sample`];
    /// backings that synthesise in f64 (the baked detail cascade, the
    /// continental field) override this to keep full precision.
    fn sample_d(&self, dir: DVec3, lod_m: f32) -> SurfaceSample {
        self.sample(dir.as_vec3(), lod_m)
    }

    /// Geometric height only — the common case for colliders, camera
    /// ray-casts, and altitude readouts.
    fn sample_height_m(&self, dir: Vec3, lod_m: f32) -> f32 {
        self.sample(dir, lod_m).height_m
    }

    /// Macro landcover moisture in `[-1, 1]` (+ wet, − dry) at unit direction
    /// `dir`, full detail — the point-query companion of
    /// [`SurfaceSample::moisture`] for consumers that don't need a full sample
    /// (grass builders, scatter). `0.0` for backings without a landcover model.
    fn landcover_moisture(&self, _dir: DVec3) -> f32 {
        0.0
    }

    /// Canopy coverage in `[0, 1]` at unit direction `dir` — the point-query
    /// companion of [`MaterialBands::canopy`], returning the *same* quantity for
    /// consumers that need the canopy field without a full sample (tree / shrub
    /// / ground-cover placement, the grass far-ring cull).
    ///
    /// `height_m` is supplied by the caller rather than re-sampled: every real
    /// consumer is placing something *on the ground* and already holds the
    /// height from its own gate, and the altitude chain (treeline, snow) needs
    /// it. Mirrors [`ProceduralSurface::macro_albedo_bands_for`]'s
    /// caller-supplies-height shape, so alternative height backings get a
    /// consistent canopy for free.
    ///
    /// **This is the one answer to "is there forest here."** Vegetation must not
    /// derive its own forest field behind this seam: that is exactly what used
    /// to make the ground palette and the tree scatter disagree from the air
    /// (see [`crate::canopy`]). `0.0` for backings without a landcover model.
    ///
    /// **Per-tile callers want [`Self::canopy_climate`] instead.** This is the
    /// convenience form for one-off queries: it re-evaluates the expensive
    /// climate terms every call, which is fine for a site search and ruinous in
    /// a per-candidate placement loop (grass evaluates coverage thousands of
    /// times per tile — doing it this way stalled terrain streaming outright).
    ///
    /// [`ProceduralSurface::macro_albedo_bands_for`]: crate::procedural::ProceduralSurface::macro_albedo_bands_for
    fn canopy_coverage(&self, dir: DVec3, height_m: f32, lod_m: f32) -> f32 {
        self.canopy_climate(dir, lod_m).coverage(dir, height_m)
    }

    /// The slowly-varying climate half of canopy coverage at `dir` — hoist this
    /// **once per tile**, then call [`CanopyClimate::coverage`] per candidate.
    /// See [`CanopyClimate`] for the cost split and why the altitude chain
    /// deliberately stays per-candidate.
    ///
    /// Default is a zero climate, whose coverage is `0.0` — correct for backings
    /// with no landcover model (airless bodies, plain oceans).
    fn canopy_climate(&self, _dir: DVec3, _lod_m: f32) -> CanopyClimate {
        CanopyClimate::default()
    }

    /// Sample plus the canonical [`MaterialBands`] in one evaluation — the
    /// tile renderer's per-vertex query (NTR-X4 material layers). The default
    /// returns the plain sample with zero bands; backings with a landcover
    /// model override this to reuse the band evaluation their albedo already
    /// performs (one landcover authority, no second biome model).
    fn sample_bands_d(&self, dir: DVec3, lod_m: f32) -> (SurfaceSample, MaterialBands) {
        (self.sample_d(dir, lod_m), MaterialBands::default())
    }

    /// Body reference radius, metres.
    fn radius_m(&self) -> f32;

    /// Vertical envelope (metres) the surface can occupy: static + dynamic +
    /// procedural-detail headroom. Consumers size their height encoding from
    /// this.
    fn height_range_m(&self) -> f32;

    /// Conservative maximum surface displacement (metres) that one more
    /// refinement step can reveal inside `patch` at `refined_spacing_m`.
    ///
    /// `Some(error)` lets a screen-space selector omit a split once that error
    /// projects below its pixel threshold. `None` means the backing cannot
    /// bound every geometric contributor there, so callers must retain their
    /// existing heuristic. The default is intentionally conservative.
    fn refinement_error_m(&self, _patch: SurfacePatch, _refined_spacing_m: f32) -> Option<f32> {
        None
    }

    /// Hint the backing to materialise `region` at `lod_m` asynchronously.
    /// No-op for the baked backing (everything is already resident).
    fn prewarm(&self, _region: Region, _lod_m: f32) {}

    /// Feature instances in `region` at `lod_m` — the union of procedural and
    /// explicit placements (spec §9). The baked backing has no feature catalog
    /// and returns empty; a field-DAG backing forwards to its
    /// [`crate::pipeline::feature::FeatureCatalog`].
    fn query_features(
        &self,
        _region: Region,
        _lod_m: f32,
    ) -> Vec<crate::pipeline::feature::FeatureInstance> {
        Vec::new()
    }
}

/// [`SurfaceQuery`] backed by the baked [`PlanetSurface`] and the analytic
/// detail cascade. The P0 implementation; later phases add field-DAG backings.
#[derive(Clone)]
pub struct BakedSurface {
    surface: Arc<PlanetSurface>,
    dynamic_state: DynamicSurfaceState,
}

impl BakedSurface {
    pub fn new(surface: Arc<PlanetSurface>, dynamic_state: DynamicSurfaceState) -> Self {
        Self {
            surface,
            dynamic_state,
        }
    }

    pub fn surface(&self) -> &Arc<PlanetSurface> {
        &self.surface
    }

    pub fn dynamic_state(&self) -> &DynamicSurfaceState {
        &self.dynamic_state
    }
}

impl SurfaceQuery for BakedSurface {
    // The trait takes an f32 `dir` for its general consumers (physics, camera,
    // HUD), which near the surface read the resident atlas anyway. The
    // precision-critical render path calls the free `surface_*` functions
    // directly with an f64 direction; here we promote so far/coarse trait
    // queries still resolve.
    fn sample(&self, dir: Vec3, lod_m: f32) -> SurfaceSample {
        surface_sample(&self.surface, &self.dynamic_state, dir.as_dvec3(), lod_m)
    }

    fn sample_d(&self, dir: DVec3, lod_m: f32) -> SurfaceSample {
        surface_sample(&self.surface, &self.dynamic_state, dir, lod_m)
    }

    fn sample_height_m(&self, dir: Vec3, lod_m: f32) -> f32 {
        surface_height_m(&self.surface, &self.dynamic_state, dir.as_dvec3(), lod_m)
    }

    fn radius_m(&self) -> f32 {
        self.surface.static_surface.radius_m
    }

    fn height_range_m(&self) -> f32 {
        surface_height_range_m(&self.surface, &self.dynamic_state)
    }
}

/// Borrowing [`SurfaceQuery`] over `&PlanetSurface` + `&DynamicSurfaceState`,
/// for consumers that already hold borrows and don't want an `Arc` round-trip
/// (camera ray-casts, the editor's tile config). Mirrors [`BakedSurface`]
/// without owning, so call sites can pass `&dyn SurfaceQuery` without threading
/// an `Arc` through their plumbing.
pub struct SurfaceRef<'a> {
    pub surface: &'a PlanetSurface,
    pub dynamic_state: &'a DynamicSurfaceState,
}

impl SurfaceQuery for SurfaceRef<'_> {
    fn sample(&self, dir: Vec3, lod_m: f32) -> SurfaceSample {
        surface_sample(self.surface, self.dynamic_state, dir.as_dvec3(), lod_m)
    }

    fn sample_d(&self, dir: DVec3, lod_m: f32) -> SurfaceSample {
        surface_sample(self.surface, self.dynamic_state, dir, lod_m)
    }

    fn sample_height_m(&self, dir: Vec3, lod_m: f32) -> f32 {
        surface_height_m(self.surface, self.dynamic_state, dir.as_dvec3(), lod_m)
    }

    fn radius_m(&self) -> f32 {
        self.surface.static_surface.radius_m
    }

    fn height_range_m(&self) -> f32 {
        surface_height_range_m(self.surface, self.dynamic_state)
    }
}

// ---------------------------------------------------------------------------
// Local terrain flattening
// ---------------------------------------------------------------------------

/// A flat rectangular pad stamped into the terrain heightfield — used to level
/// the ground under a runway (or any built site) without re-baking the body.
///
/// The pad is defined in the body-fixed frame: a rectangle of half-extents
/// `(half_along_m, half_across_m)` on the tangent plane at `center_dir`, raised
/// (or cut) to the **flat tangent plane** through `center_dir` at radial height
/// `elevation_m` — a true plane, not a constant-radius spherical cap (see
/// [`Self::plane_elevation_m`]). A flat pad matters at scale: anything built on
/// the same footprint as a flat plane (e.g. the runway's flat asphalt slab and
/// collider) stays coplanar with the ground, instead of the flat strip floating
/// off a curved cap by the curvature drop (~1 m at the end of a 5 km runway).
/// Outside the rectangle the flatten smoothstep-blends back to the natural
/// terrain over `ramp_m` metres, so there is no cliff at the pad edge.
///
/// [`FlattenedSurface`] reads this through a shared handle when synthesising
/// tiles, so the *rendered* terrain, the GPU-atlas height mirror (collider +
/// height source), and any CPU height query through the wrapped surface all see
/// the same flattened ground by construction.
#[derive(Debug, Clone, Copy)]
pub struct TerrainFlatten {
    /// Unit body-fixed direction to the pad centre.
    pub center_dir: DVec3,
    /// Unit body-fixed tangent along the pad's long axis.
    pub tangent_along: DVec3,
    /// Unit body-fixed tangent across the pad's short axis.
    pub tangent_across: DVec3,
    /// Half-length of the flat region along `tangent_along`, metres.
    pub half_along_m: f64,
    /// Half-width of the flat region along `tangent_across`, metres.
    pub half_across_m: f64,
    /// Rectangle-centre offset from `center_dir` along `tangent_along`, metres.
    /// The **plane** stays tangent at `center_dir` (see
    /// [`Self::plane_elevation_m`]); only the levelled rectangle shifts within
    /// it. Lets an asymmetric footprint (the spaceport basin, offset toward its
    /// secondary runway) share one plane with everything built at the anchor —
    /// anchoring the plane at the rectangle centre instead would tilt the ground
    /// ~`offset/R` relative to the pavement and bury/float it by decimetres at
    /// the far structures. Set via [`Self::with_rect_offset`], default `0`.
    pub offset_along_m: f64,
    /// Rectangle-centre offset along `tangent_across`, metres. See
    /// [`Self::offset_along_m`].
    pub offset_across_m: f64,
    /// Width of the blend-to-terrain ramp outside the flat region, metres.
    pub ramp_m: f64,
    /// Radial height (m above the reference radius) of the pad **at its centre**.
    /// Away from the centre the pad follows the flat tangent plane through this
    /// point, so the level it flattens to is [`Self::plane_elevation_m`], not a
    /// constant `elevation_m` everywhere.
    pub elevation_m: f64,
    /// Body reference radius (m), used to convert directions to tangent-plane
    /// offsets.
    pub radius_m: f64,
    /// `cos` of the largest angle from `center_dir` the flatten can reach
    /// (rectangle diagonal + ramp). Precomputed for a cheap per-sample reject.
    cos_max: f64,
}

impl TerrainFlatten {
    pub fn new(
        center_dir: DVec3,
        tangent_along: DVec3,
        tangent_across: DVec3,
        half_along_m: f64,
        half_across_m: f64,
        ramp_m: f64,
        elevation_m: f64,
        radius_m: f64,
    ) -> Self {
        // Largest lateral reach: rectangle half-diagonal plus the ramp.
        let reach = ((half_along_m * half_along_m + half_across_m * half_across_m).sqrt() + ramp_m)
            .max(0.0);
        // Tangent-plane reach → angle off the centre direction.
        let cos_max = (reach / radius_m.max(1.0)).atan().cos();
        Self {
            center_dir: center_dir.normalize(),
            tangent_along: tangent_along.normalize(),
            tangent_across: tangent_across.normalize(),
            half_along_m,
            half_across_m,
            ramp_m,
            elevation_m,
            radius_m,
            offset_along_m: 0.0,
            offset_across_m: 0.0,
            cos_max,
        }
    }

    /// Offset the levelled rectangle within the tangent plane (metres along
    /// `tangent_along` / `tangent_across`) without moving the plane's anchor.
    /// See [`Self::offset_along_m`].
    pub fn with_rect_offset(mut self, offset_along_m: f64, offset_across_m: f64) -> Self {
        self.offset_along_m = offset_along_m;
        self.offset_across_m = offset_across_m;
        // Re-derive the angular reject for the shifted rectangle.
        let reach_along = offset_along_m.abs() + self.half_along_m;
        let reach_across = offset_across_m.abs() + self.half_across_m;
        let reach = ((reach_along * reach_along + reach_across * reach_across).sqrt()
            + self.ramp_m)
            .max(0.0);
        self.cos_max = (reach / self.radius_m.max(1.0)).atan().cos();
        self
    }

    /// Blend weight in `[0, 1]` at body-fixed unit direction `dir`: `1` inside
    /// the flat pad, smoothstep down to `0` across `ramp_m`, `0` beyond.
    pub fn weight(&self, dir: DVec3) -> f64 {
        // Cheap angular reject so the 99.99% of tile pixels far from the pad pay
        // only one dot product.
        if dir.dot(self.center_dir) < self.cos_max {
            return 0.0;
        }
        // Tangent-plane offset from the pad centre, in metres. For a region this
        // small relative to the radius the chord matches the arc to well under a
        // millimetre, so the simple projection is exact enough.
        let offset = (dir - self.center_dir) * self.radius_m;
        let along =
            (offset.dot(self.tangent_along) - self.offset_along_m).abs() - self.half_along_m;
        let across =
            (offset.dot(self.tangent_across) - self.offset_across_m).abs() - self.half_across_m;
        if along <= 0.0 && across <= 0.0 {
            return 1.0;
        }
        // Distance from the rectangle's edge (exterior SDF).
        let dist = (along.max(0.0) * along.max(0.0) + across.max(0.0) * across.max(0.0)).sqrt();
        if dist >= self.ramp_m {
            return 0.0;
        }
        let t = 1.0 - dist / self.ramp_m;
        t * t * (3.0 - 2.0 * t)
    }

    /// Radial height (m above the reference radius) of the **flat tangent plane**
    /// the pad levels to, at body-fixed unit direction `dir`.
    ///
    /// The pad is a true flat plane — the tangent plane at `center_dir`, at
    /// radial height `elevation_m` directly above the centre — not a
    /// constant-radius spherical cap. A terrain point rendered at this radial
    /// height lands exactly on that plane, so the flattened ground stays coplanar
    /// with anything else built flat over the same footprint (the runway's flat
    /// asphalt slab, collider, and parked-craft rest pose). Away from the centre
    /// it rises above `elevation_m` by the curvature drop (~1 m at the end of a
    /// 5 km pad on a 3,186 km body) — exactly the gap that left the flat runway
    /// floating above a constant-`elevation_m` cap.
    pub fn plane_elevation_m(&self, dir: DVec3) -> f64 {
        // The plane is `{ P : P·center_dir = radius_m + elevation_m }`. A point in
        // direction `dir` at radial height `h` has `P = dir·(radius_m + h)`, so
        // `(radius_m + h)·cosθ = radius_m + elevation_m` where `cosθ = dir·center_dir`.
        let cos_theta = dir.dot(self.center_dir).max(1e-6);
        (self.radius_m + self.elevation_m) / cos_theta - self.radius_m
    }
}

/// One identified flatten region in a body's shared handle. The `id` is the
/// owning [`StructureSite`](../../game/src/structures.rs)'s id (so a structure
/// can update or remove exactly its own pad); the terrain crate treats it as an
/// opaque tag.
#[derive(Debug, Clone, Copy)]
pub struct FlattenRegion {
    pub id: u64,
    pub flatten: TerrainFlatten,
}

/// Shared, runtime-settable set of flatten regions for one body. Empty means
/// "no flattening", so a [`FlattenedSurface`] wrapping an empty handle is a
/// transparent passthrough. Multiple regions coexist (e.g. the runway pad plus
/// a player-placed base site); they are assumed not to overlap, so the surface
/// applies the single highest-weight region at any direction rather than
/// stacking ramps. The handle is read on every tile-pixel synthesis, so writing
/// a region takes effect on tiles baked afterward without rebuilding the
/// provider.
pub type FlattenHandle = Arc<RwLock<Vec<FlattenRegion>>>;

/// Create an empty [`FlattenHandle`] (no flattening).
pub fn flatten_handle() -> FlattenHandle {
    Arc::new(RwLock::new(Vec::new()))
}

/// Pick the flatten region whose pad centre is nearest the body-fixed unit
/// direction `dir`, for consumers that want a single representative pad rather
/// than the full set (e.g. vegetation exclusion, which tests one pad per
/// dispatch). Pads don't overlap, so the nearest pad is the only one that can
/// matter near `dir`. Returns `None` when there are no regions.
pub fn nearest_flatten(regions: &[FlattenRegion], dir: DVec3) -> Option<TerrainFlatten> {
    regions
        .iter()
        .max_by(|a, b| {
            a.flatten
                .center_dir
                .dot(dir)
                .total_cmp(&b.flatten.center_dir.dot(dir))
        })
        .map(|r| r.flatten)
}

/// [`SurfaceQuery`] decorator that overlays an optional [`TerrainFlatten`] on a
/// wrapped surface. The geometric height is levelled onto the pad's tangent
/// plane, and the pad's RAMP band wears the albedo toward bare soil in noise-
/// broken patches (see [`Self::worn_albedo`]) — the ground around a built site
/// reads as trafficked margin instead of untouched meadow running to the
/// asphalt edge. Roughness and all other metadata pass through unchanged.
pub struct FlattenedSurface {
    inner: Arc<dyn SurfaceQuery>,
    flatten: FlattenHandle,
}

/// Worn-margin band over the flatten weight: zero on wild ground, peaked on
/// the ramp, zero again well inside the pad (the levelled interior is the
/// managed lawn, not dirt).
const WORN_BAND_LO_W: f64 = 0.04;
const WORN_BAND_UP_W: f64 = 0.30;
const WORN_BAND_HI_START_W: f64 = 0.55;
const WORN_BAND_HI_END_W: f64 = 0.92;
/// Patch wavelength (m) and strength of the worn breakup. The noise gates the
/// wear into patches (bare scars between surviving grass) rather than painting
/// a uniform ring — a solid band reads as a decal, not as use.
const WORN_PATCH_WL_M: f64 = 22.0;
const WORN_BASE: f32 = 0.18;
const WORN_PATCH: f32 = 0.55;

impl FlattenedSurface {
    pub fn new(inner: Arc<dyn SurfaceQuery>, flatten: FlattenHandle) -> Self {
        Self { inner, flatten }
    }

    /// Blend weight + tangent-plane elevation of the single strongest region at
    /// `dir`, or `None` off-pad. Pads are assumed not to overlap (runway plus a
    /// distant base site), so max-weight selection avoids stacking two ramps
    /// into a double blend. The common case (zero or one region) costs one
    /// cheap angular reject per region.
    fn flatten_at(&self, dir: DVec3) -> Option<(f64, f64)> {
        let guard = self.flatten.read().ok()?;
        let mut best_w = 0.0_f64;
        let mut best_elev = 0.0_f64;
        for region in guard.iter() {
            let w = region.flatten.weight(dir);
            if w > best_w {
                best_w = w;
                // The flat *tangent plane* level at `dir`, not the constant centre
                // `elevation_m` — so the levelled ground is parallel to (and a
                // uniform asphalt-lift below) the flat runway slab built on it,
                // instead of a curved cap the flat strip floats above.
                best_elev = region.flatten.plane_elevation_m(dir);
            }
        }
        (best_w > 0.0).then_some((best_w, best_elev))
    }

    fn flatten_height(&self, dir: DVec3, natural_m: f32) -> f32 {
        match self.flatten_at(dir) {
            Some((w, elev)) => (natural_m as f64 * (1.0 - w) + elev * w) as f32,
            None => natural_m,
        }
    }

    /// Wear the albedo toward the shared bare-soil anchor on the pad's ramp
    /// band, broken into ~[`WORN_PATCH_WL_M`] patches so the margin reads as
    /// trafficked ground. Pure function of the body-fixed position, so tiles
    /// and any other albedo consumer agree, and deterministic across sessions.
    fn worn_albedo(&self, dir: DVec3, w: f64, albedo: Vec3) -> Vec3 {
        let band = smoothstep64(WORN_BAND_LO_W, WORN_BAND_UP_W, w)
            * (1.0 - smoothstep64(WORN_BAND_HI_START_W, WORN_BAND_HI_END_W, w));
        if band <= 0.0 {
            return albedo;
        }
        let p = dir * self.inner.radius_m() as f64 / WORN_PATCH_WL_M;
        let n = crate::noise::fbm3(
            p.x as f32, p.y as f32, p.z as f32,
            0x57EA12, // arbitrary fixed seed — the wear pattern is authored once
            2, 0.5, 2.0,
        );
        // Positive-tail threshold: roughly the upper half of the noise wears
        // through; the rest keeps its cover, so the band is patchy.
        let patchy = ((n - 0.45) * 2.5).clamp(0.0, 1.0);
        let wear = (band as f32) * (WORN_BASE + WORN_PATCH * patchy);
        albedo.lerp(
            crate::procedural::LATERITE_SOIL_ALBEDO,
            wear.clamp(0.0, 1.0),
        )
    }

    /// Height levelling + worn margin for one sample, shared by every entry.
    fn apply(&self, dir: DVec3, s: &mut SurfaceSample) {
        if let Some((w, elev)) = self.flatten_at(dir) {
            s.height_m = (s.height_m as f64 * (1.0 - w) + elev * w) as f32;
            s.albedo_linear = self.worn_albedo(dir, w, s.albedo_linear);
        }
    }
}

fn smoothstep64(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl SurfaceQuery for FlattenedSurface {
    fn sample(&self, dir: Vec3, lod_m: f32) -> SurfaceSample {
        let mut s = self.inner.sample(dir, lod_m);
        self.apply(dir.as_dvec3(), &mut s);
        s
    }

    fn sample_d(&self, dir: DVec3, lod_m: f32) -> SurfaceSample {
        let mut s = self.inner.sample_d(dir, lod_m);
        self.apply(dir, &mut s);
        s
    }

    fn sample_bands_d(&self, dir: DVec3, lod_m: f32) -> (SurfaceSample, MaterialBands) {
        let (mut s, bands) = self.inner.sample_bands_d(dir, lod_m);
        self.apply(dir, &mut s);
        (s, bands)
    }

    fn sample_height_m(&self, dir: Vec3, lod_m: f32) -> f32 {
        self.flatten_height(dir.as_dvec3(), self.inner.sample_height_m(dir, lod_m))
    }

    // ── Pass-through: everything that is not height ───────────────────────
    //
    // A flatten pad displaces **height** and nothing else — landcover, canopy,
    // climate, and features are all properties of the *place*, unchanged by
    // levelling the ground there.
    //
    // **Add every new [`SurfaceQuery`] method to this block.** A defaulted seam
    // method that this decorator forgets to forward does not fail loudly: it
    // silently returns the trait default for the whole body, because Thalos's
    // surface is wrapped here for the spaceport pad. That is how
    // `canopy_coverage` shipped as a constant 0.0 on its first run — every tree
    // vanished planet-wide and the capture site finder fell back to the
    // sub-stellar point, with nothing in the log to say why.
    fn landcover_moisture(&self, dir: DVec3) -> f32 {
        self.inner.landcover_moisture(dir)
    }

    fn canopy_climate(&self, dir: DVec3, lod_m: f32) -> CanopyClimate {
        self.inner.canopy_climate(dir, lod_m)
    }

    fn radius_m(&self) -> f32 {
        self.inner.radius_m()
    }

    fn height_range_m(&self) -> f32 {
        self.inner.height_range_m()
    }

    fn refinement_error_m(&self, patch: SurfacePatch, refined_spacing_m: f32) -> Option<f32> {
        self.inner.refinement_error_m(patch, refined_spacing_m)
    }

    fn prewarm(&self, region: Region, lod_m: f32) {
        self.inner.prewarm(region, lod_m);
    }

    fn query_features(
        &self,
        region: Region,
        lod_m: f32,
    ) -> Vec<crate::pipeline::feature::FeatureInstance> {
        self.inner.query_features(region, lod_m)
    }
}

// ---------------------------------------------------------------------------
// Free-function evaluator (the implementation)
//
// Consumers that already hold `&PlanetSurface` + `&DynamicSurfaceState` call
// these directly to avoid an `Arc` round-trip; `BakedSurface` delegates here.
// ---------------------------------------------------------------------------

/// Evaluate the full surface (height + albedo + roughness) at `dir`.
///
/// Stages, in order:
/// 1. Cubemap base, bilinearly sampled.
/// 2. Dynamic layers (ice caps, aeolian bedforms).
/// 3. Runtime geometric feature composition: legacy crater-backed bodies fold
///    unbaked crater features into height before procedural detail.
/// 4. Runtime geometric detail selected by the baked surface: legacy bodies
///    get the P0 HMF cascade; P2A `GenericTerrestrial` bodies evaluate their
///    smooth continental field directly at full detail for LOD-invariant runtime geometry.
pub fn surface_sample(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    dir: DVec3,
    lod_m: f32,
) -> SurfaceSample {
    let dir = dir.normalize_or_zero();
    if dir == DVec3::ZERO {
        return SurfaceSample {
            height_m: 0.0,
            albedo_linear: Vec3::ZERO,
            roughness: 0.5,
            moisture: 0.0,
        };
    }
    let dynamic_lod = lod_m.max(1e-6).log2();
    let base = sample_base_with_dynamic(surface, dynamic_state, dir, dynamic_lod);
    let height_m = runtime_height_m(surface, dir, lod_m, base.height_m);
    SurfaceSample {
        height_m,
        albedo_linear: base.albedo_linear,
        roughness: base.roughness,
        moisture: 0.0,
    }
}

/// Canonical "what does the ground LOD render at this direction?" height query.
///
/// This is the single source of truth shared between the atlas baker and every
/// system that must agree with the rendered ground — terrain colliders,
/// character controllers, camera boom ray-casts, HUD altitude readouts.
///
/// Pass a small `lod_m` (e.g. `0.5`) for full procedural detail near the
/// camera; pass the patch's vertex spacing when building a coarser collider
/// mesh so the mesh resolution matches the represented detail.
pub fn surface_height_m(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    dir: DVec3,
    lod_m: f32,
) -> f32 {
    let dir = dir.normalize_or_zero();
    if dir == DVec3::ZERO {
        return 0.0;
    }
    let dynamic_lod = lod_m.max(1e-6).log2();
    let base = sample_base_with_dynamic(surface, dynamic_state, dir, dynamic_lod);
    runtime_height_m(surface, dir, lod_m, base.height_m)
}

/// Vertical range (metres) the surface must encode: static + dynamic +
/// procedural detail headroom.
pub fn surface_height_range_m(surface: &PlanetSurface, state: &DynamicSurfaceState) -> f32 {
    let base = surface.static_surface.height_range + dynamic_height_margin(surface, state);
    match surface.static_surface.runtime_detail {
        RuntimeTerrainDetail::LegacyHmf => (base
            + runtime_feature_height_margin_m(&surface.static_surface)
            + DETAIL_HEIGHT_MARGIN_M)
            .max(1.0),
        RuntimeTerrainDetail::AirlessRegolith(params) => (base + params.amplitude_m.abs()).max(1.0),
        RuntimeTerrainDetail::BasicContinental(params) => {
            base.max(params.height_range_hint_m()).max(1.0)
        }
        RuntimeTerrainDetail::OceanicContinental(params) => {
            base.max(params.height_range_hint_m()).max(1.0)
        }
    }
}

/// Object-space surface normal at `dir`, via LOD-aware finite differences of
/// [`surface_height_m`].
///
/// Renderers derive their own normals from height in the fragment path (the
/// spec keeps normals out of the per-tile output); this CPU helper exists for
/// offline consumers — PNG dumps, the editor — that need a normal without a
/// GPU. The four probes re-enter [`surface_height_m`], so every band active at
/// `lod_m` feeds the normal.
pub fn surface_normal(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    dir: DVec3,
    lod_m: f32,
) -> Vec3 {
    let dir = dir.normalize_or_zero();
    if dir == DVec3::ZERO {
        return Vec3::Y;
    }
    let radius_m = surface.static_surface.radius_m.max(1.0) as f64;
    let up = if dir.y.abs() > 0.99 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let tangent = dir.cross(up).normalize();
    let bitangent = tangent.cross(dir);

    // LOD-aware offset: at least the cubemap texel scale, growing with lod_m so
    // the derivative reflects only the bands resolvable at that LOD.
    let res = surface.static_surface.height_cubemap.resolution().max(1) as f64;
    let texel_offset = 1.5 / res;
    let pixel_offset = lod_m as f64 / radius_m;
    let offset = texel_offset.max(pixel_offset);

    let h_at = |probe: DVec3| surface_height_m(surface, state, probe.normalize(), lod_m);
    let h_east = h_at(dir + tangent * offset);
    let h_west = h_at(dir - tangent * offset);
    let h_north = h_at(dir + bitangent * offset);
    let h_south = h_at(dir - bitangent * offset);

    let ds = (radius_m * offset * 2.0) as f32;
    let dh_dt = (h_east - h_west) / ds;
    let dh_db = (h_north - h_south) / ds;
    (dir.as_vec3() - tangent.as_vec3() * dh_dt - bitangent.as_vec3() * dh_db).normalize()
}

// ---------------------------------------------------------------------------
// Detail plan
// ---------------------------------------------------------------------------

/// Per-tile detail plan: a continuous octave count. Fractional values blend
/// the top octave in smoothly so a tile cascading from N → N+1 across an LOD
/// boundary does not pop. `0.0` disables detail.
#[derive(Clone, Copy, Debug)]
struct DetailPlan {
    octaves: f32,
}

/// Choose the cascade depth for a sample at `lod_m` metres per sample.
fn detail_plan_for_lod(lod_m: f32, base_wl_m: f32) -> DetailPlan {
    if lod_m <= 0.0 {
        return DetailPlan {
            octaves: MAX_DETAIL_OCTAVES,
        };
    }
    let ratio = base_wl_m / (2.0 * lod_m);
    if ratio <= 1.0 {
        return DetailPlan { octaves: 0.0 };
    }
    let octaves = (ratio.log2() + 1.0).clamp(0.0, MAX_DETAIL_OCTAVES);
    DetailPlan { octaves }
}

// ---------------------------------------------------------------------------
// Base sample (cubemap + dynamic layers)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct BaseSample {
    height_m: f32,
    albedo_linear: Vec3,
    roughness: f32,
}

fn sample_base_with_dynamic(
    surface: &PlanetSurface,
    dynamic_state: &DynamicSurfaceState,
    dir: DVec3,
    dynamic_lod: f32,
) -> BaseSample {
    let body = &surface.static_surface;
    // The base cubemap is low-resolution; f32 addressing is already far below
    // its texel size, so the radial direction downcasts here without loss.
    let dir = dir.as_vec3();
    let mut height = sample_height_bilinear(&body.height_cubemap, dir, body.height_range);
    let mut albedo = srgb_rgba8_to_linear_rgb(cubemap_texel_nearest(&body.albedo_cubemap, dir));
    let mut roughness = static_roughness(surface, dir);
    if !surface.dynamic_layers.is_empty() {
        apply_dynamic_surface_layers(
            surface,
            dynamic_state,
            dir,
            dynamic_lod,
            &mut height,
            &mut albedo,
            &mut roughness,
        );
    }
    BaseSample {
        height_m: height,
        albedo_linear: albedo,
        roughness,
    }
}

/// Bilinear sampling on a R16 height cubemap, decoded to metres.
fn sample_height_bilinear(cube: &Cubemap<u16>, dir: Vec3, height_range: f32) -> f32 {
    let (face, u, v) = dir_to_face_uv(dir);
    let res = cube.resolution();
    let res_f = res as f32;
    let px = (u * res_f - 0.5).clamp(0.0, res_f - 1.001);
    let py = (v * res_f - 0.5).clamp(0.0, res_f - 1.001);
    let x0 = px.floor() as u32;
    let y0 = py.floor() as u32;
    let x1 = (x0 + 1).min(res - 1);
    let y1 = (y0 + 1).min(res - 1);
    let fx = px - px.floor();
    let fy = py - py.floor();

    let h00 = decode_height(cube.get(face, x0, y0), height_range);
    let h10 = decode_height(cube.get(face, x1, y0), height_range);
    let h01 = decode_height(cube.get(face, x0, y1), height_range);
    let h11 = decode_height(cube.get(face, x1, y1), height_range);

    let top = h00 + (h10 - h00) * fx;
    let bot = h01 + (h11 - h01) * fx;
    top + (bot - top) * fy
}

fn static_roughness(surface: &PlanetSurface, dir: Vec3) -> f32 {
    let body = &surface.static_surface;
    let texel = cubemap_texel_nearest(&body.roughness_cubemap, dir);
    if texel > 0 {
        texel as f32 / 255.0
    } else {
        let material_id = cubemap_texel_nearest(&body.material_cubemap, dir) as usize;
        body.materials
            .get(material_id)
            .map(|m| m.roughness)
            .unwrap_or(0.5)
    }
}

// ---------------------------------------------------------------------------
// Runtime geometric detail
// ---------------------------------------------------------------------------

fn runtime_height_m(surface: &PlanetSurface, dir: DVec3, lod_m: f32, base_height_m: f32) -> f32 {
    match surface.static_surface.runtime_detail {
        RuntimeTerrainDetail::LegacyHmf => {
            // Legacy cascade is f32 (orbital-scale bake heritage); downcast here.
            let dir_f = dir.as_vec3();
            let feature_base_m =
                compose_runtime_features_m(&surface.static_surface, dir, base_height_m);
            let plan = detail_plan_for_lod(lod_m, DETAIL_BASE_WL_M);
            let detail_h = compute_detail_height(dir_f, surface.static_surface.radius_m, plan);
            combine_base_and_detail(feature_base_m, detail_h, surface.static_surface.sea_level_m)
        }
        RuntimeTerrainDetail::AirlessRegolith(params) => {
            // Airless regolith: the feature layer (craters/basins/mare) is the
            // macro relief; add only a gentle rounded fBM undulation on top. No
            // ridged HMF, no domain warp — those read as jagged mountains the
            // moon shouldn't have. LOD-invariant (sampled at full detail
            // regardless of tile LOD) like the other non-legacy arms, so a
            // parent→child tile handoff doesn't terrace the ground mesh.
            let dir_f = dir.as_vec3();
            let feature_base_m =
                compose_runtime_features_m(&surface.static_surface, dir, base_height_m);
            let detail_h =
                compute_regolith_detail_height(dir_f, surface.static_surface.radius_m, params);
            feature_base_m + detail_h
        }
        RuntimeTerrainDetail::BasicContinental(params) => {
            let _ = lod_m;
            // For the P2A Thalos prototype, keep runtime geometry invariant
            // across tile LODs. Otherwise the same world-space point receives
            // a different height when a parent tile hands off to a child,
            // which reads as contour-like terracing in the ground mesh. The
            // f64 direction keeps the sample position precise at planet scale.
            params.sample_height_dm(surface.static_surface.radius_m as f64, dir, 1.0)
        }
        RuntimeTerrainDetail::OceanicContinental(params) => {
            let _ = lod_m;
            // Same LOD-invariant rule as BasicContinental: the signed land / seabed
            // evaluator is the geometric surface for mesh and collider, while the
            // separate water renderer sits at `sea_level_m` above it.
            params.sample_height_dm(surface.static_surface.radius_m as f64, dir, 1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Legacy procedural detail cascade (domain-warped ridged HMF)
// ---------------------------------------------------------------------------

/// Domain-warped ridged hybrid multifractal in metres. Evaluated in body-local
/// 3D so the field is sphere-continuous (the same physical point returns the
/// same value regardless of which cube face is generating it). Returns `0.0`
/// when the LOD plan disables detail.
fn compute_detail_height(dir: Vec3, radius_m: f32, plan: DetailPlan) -> f32 {
    if plan.octaves <= 0.0 {
        return 0.0;
    }

    let p_3d_m = dir * radius_m;

    let warp_sample = crate::noise::fbm3_vec3(
        p_3d_m / WARP_WAVELENGTH_M,
        WARP_NOISE_SEED,
        WARP_OCTAVES,
        DETAIL_PERSISTENCE,
        DETAIL_LACUNARITY,
    );
    let warped_m = p_3d_m + warp_sample * WARP_AMP_M;

    let hmf = crate::noise::hmf_ridged_3d(
        warped_m / DETAIL_BASE_WL_M,
        DETAIL_NOISE_SEED,
        plan.octaves,
        DETAIL_PERSISTENCE,
        DETAIL_LACUNARITY,
        DETAIL_OFFSET,
    );

    hmf * DETAIL_AMP_M
}

/// Gentle rounded fBM undulation for airless regolith, in metres. Signed
/// (`± amplitude_m`) and evaluated in body-local 3D so it's sphere-continuous.
/// Unlike [`compute_detail_height`] there is no domain warp and no ridged
/// transform — plain fBM gives soft hummocks between the feature-layer craters
/// rather than the jagged mountains the ridged cascade produces.
fn compute_regolith_detail_height(dir: Vec3, radius_m: f32, params: AirlessRegolithParams) -> f32 {
    let wl = params.base_wavelength_m.max(1.0);
    let p = dir * radius_m / wl;
    // fbm3 returns ~[0,1]; center to ±1 then scale to the signed amplitude.
    // 5 octaves of rounded fBM: enough to texture the surface across the LOD
    // range without the high-frequency aliasing the ridged cascade had.
    let n = crate::noise::fbm3(
        p.x,
        p.y,
        p.z,
        params.seed,
        5,
        DETAIL_PERSISTENCE,
        DETAIL_LACUNARITY,
    );
    (n - 0.5) * 2.0 * params.amplitude_m
}

/// Combine macro height and HMF detail uplift, capping in shallow bathymetry so
/// positive-only detail never breaches the water surface on continental
/// shelves.
fn combine_base_and_detail(base_height_m: f32, detail_h: f32, sea_level_m: Option<f32>) -> f32 {
    let Some(sea) = sea_level_m else {
        return base_height_m + detail_h;
    };
    if base_height_m >= sea {
        return base_height_m + detail_h;
    }
    let max_uplift = (sea - base_height_m - SEA_LEVEL_CAP_EPSILON_M).max(0.0);
    base_height_m + detail_h.min(max_uplift)
}

// ---------------------------------------------------------------------------
// Dynamic-layer height margin
// ---------------------------------------------------------------------------

fn dynamic_height_margin(surface: &PlanetSurface, state: &DynamicSurfaceState) -> f32 {
    let mut margin = 0.0;

    for (index, layer) in surface.dynamic_layers.ice_caps.iter().enumerate() {
        let coverage_scale = state
            .ice_cap_state(index, layer)
            .map(|s| s.coverage_scale)
            .unwrap_or(1.0);
        let thickness_scale = state
            .ice_cap_state(index, layer)
            .map(|s| s.thickness_scale)
            .unwrap_or(1.0);
        if coverage_scale > 0.0 {
            margin += layer.spec.max_thickness_m.max(0.0) * thickness_scale.max(0.0);
        }
    }

    for (index, layer) in surface.dynamic_layers.active_dunes.iter().enumerate() {
        let Some(dune_state) = state.active_dune_state(index, layer) else {
            let region = &layer.region;
            margin += region.amplitude_draa_m.max(0.0) + region.amplitude_dune_m.max(0.0);
            continue;
        };
        if dune_state.coverage_scale > 0.0 {
            let region = &layer.region;
            margin += dune_state.amplitude_scale.max(0.0)
                * (region.amplitude_draa_m.max(0.0) + region.amplitude_dune_m.max(0.0));
        }
    }

    margin
}

// ---------------------------------------------------------------------------
// Cubemap / colour helpers
// ---------------------------------------------------------------------------

fn cubemap_texel_nearest<T>(cube: &Cubemap<T>, dir: Vec3) -> T
where
    T: Copy + Default,
{
    let (face, u, v) = dir_to_face_uv(dir);
    let res = cube.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    cube.get(face, x, y)
}

fn decode_height(texel: u16, range: f32) -> f32 {
    (texel as f32 / 65535.0 * 2.0 - 1.0) * range
}

fn srgb_rgba8_to_linear_rgb(texel: [u8; 4]) -> Vec3 {
    Vec3::new(
        srgb8_to_linear(texel[0]),
        srgb8_to_linear(texel[1]),
        srgb8_to_linear(texel[2]),
    )
}

fn srgb8_to_linear(srgb: u8) -> f32 {
    let srgb = f32::from(srgb) / 255.0;
    if srgb <= 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}
