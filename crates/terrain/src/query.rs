//! Query API — the seam between terrain generation and its consumers.
//!
//! This is P0 of the planet-generation pipeline migration
//! ([docs/planet-generation-pipeline-migration.md]). It establishes a single
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

use std::sync::Arc;

use glam::{DVec3, Vec3};

use crate::cubemap::{Cubemap, dir_to_face_uv};
use crate::feature_compositor::{compose_runtime_features_m, runtime_feature_height_margin_m};
use crate::generic_terrestrial_field::RuntimeTerrainDetail;
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

    /// Body reference radius, metres.
    fn radius_m(&self) -> f32;

    /// Vertical envelope (metres) the surface can occupy: static + dynamic +
    /// procedural-detail headroom. Consumers size their height encoding from
    /// this.
    fn height_range_m(&self) -> f32;

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
        };
    }
    let dynamic_lod = lod_m.max(1e-6).log2();
    let base = sample_base_with_dynamic(surface, dynamic_state, dir, dynamic_lod);
    let height_m = runtime_height_m(surface, dir, lod_m, base.height_m);
    SurfaceSample {
        height_m,
        albedo_linear: base.albedo_linear,
        roughness: base.roughness,
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
                compose_runtime_features_m(&surface.static_surface, dir_f, base_height_m);
            let plan = detail_plan_for_lod(lod_m, DETAIL_BASE_WL_M);
            let detail_h = compute_detail_height(dir_f, surface.static_surface.radius_m, plan);
            combine_base_and_detail(feature_base_m, detail_h, surface.static_surface.sea_level_m)
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
