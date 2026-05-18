//! Three-layer LOD contract for Mira-class bodies.
//!
//! This is the single source of truth for the layer boundaries. All agents —
//! CPU sampler, shader, rendering upload — must agree with these numbers.
//!
//! ```text
//! ┌────────────────┬──────────────────┬────────────────────────────────────┐
//! │ Layer          │ Radius range     │ Storage                            │
//! ├────────────────┼──────────────────┼────────────────────────────────────┤
//! │ Cubemap (baked)│ ≥ 5 km          │ height_cubemap (R16), albedo_cube  │
//! │                │                  │ (sRGB8), material_cubemap (R8).    │
//! │                │                  │ One fetch per sample.              │
//! ├────────────────┼──────────────────┼────────────────────────────────────┤
//! │ Feature SSBO   │ 500 m – 5 km    │ StaticSurfaceData.craters + feature_index.  │
//! │                │                  │ Iterated per fragment via spatial  │
//! │                │                  │ index. ~18 features/cell at L4     │
//! │                │                  │ ico, ~125/fragment worst case.     │
//! ├────────────────┼──────────────────┼────────────────────────────────────┤
//! │ Shader hash    │ < 500 m         │ Analytic crater noise from         │
//! │ (statistical)  │                  │ DetailNoiseParams. No identity —   │
//! │                │                  │ pure statistical tail.             │
//! └────────────────┴──────────────────┴────────────────────────────────────┘
//! ```
//!
//! ## Continuity invariants
//!
//! 1. **Smoothstep fade-in by screen-space size.** Every explicit crater
//!    fades in over `smoothstep(0.5 px, 4 px, feature_diameter_px)`. This
//!    prevents pop-in when the camera zooms and a crater crosses the
//!    "resolvable" threshold.
//!
//! 2. **SFD continuity at 500 m boundary.** The shader hash layer's
//!    `DetailNoiseParams` must produce the same expected SFD slope and
//!    density as the tail of the explicit `craters` population below its
//!    minimum radius. Without this, the handoff is visible as a seam.
//!
//! 3. **Normals derived from height gradient only, never baked.** The normal
//!    at a sample reflects whichever bands are active at the current LOD;
//!    six lines of shader / Rust code, identical across impostor and UDLOD.
//!
//! ## Two-view coverage
//!
//! - **Map view**: impostor billboard. Reads cubemap + iterates SSBO +
//!   shader hash. Map-view zoom can push pixel_size_m down to ~100 m, at
//!   which point mid-size SSBO craters (1-5 km) become resolvable and must
//!   render crisply. The SSBO iteration in the fragment shader delivers
//!   this.
//!
//! - **Ship view (UDLOD, not built)**: tessellated terrain. Vertex path
//!   samples cubemap displacement; fragment path iterates the same SSBO
//!   with finer pixel_size_m. Same contract — the impostor is the first
//!   consumer, UDLOD is the second.

use glam::Vec3;

use crate::crater_profile::{
    SubPeaks, crater_profile, degradation_factor, degradation_softness, morphology_for_radius,
    smoothstep_range,
};
use crate::cubemap::dir_to_face_uv;
use crate::spatial_index::FeatureRef;
use crate::static_surface::{PlanetSurface, StaticSurfaceData};
use crate::types::{ActiveDuneState, Crater, DetailNoiseParams, DynamicSurfaceState, IceCapState};

/// Result of sampling the surface at a point.
pub struct SurfaceSample {
    /// Height above the reference sphere, in meters.
    pub height: f32,
    /// World-space normal, derived from the height gradient.
    pub normal: Vec3,
    /// Linear-space albedo color.
    pub albedo: Vec3,
    /// PBR roughness, 0..1.
    pub roughness: f32,
    /// Index into `StaticSurfaceData::materials`.
    pub material_id: u32,
}

/// Sample the body surface at a direction on the unit sphere.
///
/// `lod` is `log2(meters_per_sample)` at the query point.
/// Larger = coarser.
///
/// ## LOD branching
/// 1. **Always**: read the cubemap layer (height + albedo).
/// 2. **Always (filtered by fade)**: iterate nearby features via the spatial
///    index. Each crater's contribution is weighted by a screen-space-size
///    smoothstep so sub-pixel features drop out continuously.
/// 3. **If `pixel_size_m < d_max_m`**: evaluate statistical detail noise.
pub fn sample_static_surface(body: &StaticSurfaceData, dir: Vec3, lod: f32) -> SurfaceSample {
    let dir = dir.normalize();

    let height = sample_height_only(body, dir, lod);
    let normal = compute_normal(body, dir, lod);
    let albedo = sample_albedo(body, dir);

    let material_id = body.material_cubemap.sample_nearest(dir) as u32;
    let roughness = sample_roughness(body, dir, material_id);

    SurfaceSample {
        height,
        normal,
        albedo,
        roughness,
        material_id,
    }
}

/// Sample the static substrate plus dynamic surface layers.
///
/// Dynamic displacement contributes to the shared height and normal paths so
/// the impostor projection and future ground tiles can mirror one canonical
/// surface evaluation. The impostor still uses a sphere silhouette; this API
/// only defines sampled surface height.
pub fn sample_surface(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    dir: Vec3,
    lod: f32,
) -> SurfaceSample {
    let dir = dir.normalize();
    let body = &surface.static_surface;

    let mut height = sample_height_only(body, dir, lod);
    let mut albedo = sample_albedo(body, dir);
    let material_id = body.material_cubemap.sample_nearest(dir) as u32;
    let mut roughness = sample_roughness(body, dir, material_id);
    apply_dynamic_surface_layers(
        surface,
        state,
        dir,
        lod,
        &mut height,
        &mut albedo,
        &mut roughness,
    );

    SurfaceSample {
        height,
        normal: compute_surface_normal(surface, state, dir, lod),
        albedo,
        roughness,
        material_id,
    }
}

// ---------------------------------------------------------------------------
// Height (Layer 1 + 2 + 3), no normal.  Used directly by `sample_static_surface()` and
// recursively by `compute_normal` for finite-difference offsets.
// ---------------------------------------------------------------------------

/// Evaluate the full three-layer height at a direction.
///
/// This function does not compute a normal and does not recurse into
/// `compute_normal`. The finite-difference normal derivation is the only
/// thing that calls this directly with offset directions.
fn sample_height_only(body: &StaticSurfaceData, dir: Vec3, lod: f32) -> f32 {
    // Layer 1: baked cubemap.
    let mut h = sample_cubemap_height(body, dir);

    // Layer 2: explicit crater features.
    h += sample_layer2_craters(body, dir, lod);

    // Layer 3: statistical detail noise.
    if lod < detail_threshold_lod(&body.detail_params) {
        let (dh, _grad) = sample_detail_noise(&body.detail_params, dir, lod);
        h += dh;
    }

    h
}

/// LOD threshold above which the statistical detail layer contributes.
/// Equivalent to `log2(d_max_m)` — once pixel_size_m >= d_max_m, every octave
/// is below the per-crater fade-in cutoff.
fn detail_threshold_lod(params: &DetailNoiseParams) -> f32 {
    if params.d_max_m <= 0.0 {
        return f32::NEG_INFINITY;
    }
    params.d_max_m.log2()
}

// ---------------------------------------------------------------------------
// Layer 1: cubemap
// ---------------------------------------------------------------------------

/// Decode a height texel from the R16Unorm cubemap.
fn decode_height(texel: u16, range: f32) -> f32 {
    (texel as f32 / 65535.0 * 2.0 - 1.0) * range
}

/// Sample height from the cubemap via bilinear interpolation.
fn sample_cubemap_height(body: &StaticSurfaceData, dir: Vec3) -> f32 {
    let (face, u, v) = dir_to_face_uv(dir);
    let res = body.height_cubemap.resolution() as f32;
    let px = (u * res - 0.5).clamp(0.0, res - 1.001);
    let py = (v * res - 0.5).clamp(0.0, res - 1.001);
    let x0 = px.floor() as u32;
    let y0 = py.floor() as u32;
    let x1 = (x0 + 1).min(body.height_cubemap.resolution() - 1);
    let y1 = (y0 + 1).min(body.height_cubemap.resolution() - 1);
    let fx = px - px.floor();
    let fy = py - py.floor();

    let h00 = decode_height(body.height_cubemap.get(face, x0, y0), body.height_range);
    let h10 = decode_height(body.height_cubemap.get(face, x1, y0), body.height_range);
    let h01 = decode_height(body.height_cubemap.get(face, x0, y1), body.height_range);
    let h11 = decode_height(body.height_cubemap.get(face, x1, y1), body.height_range);

    let top = h00 + (h10 - h00) * fx;
    let bot = h01 + (h11 - h01) * fx;
    top + (bot - top) * fy
}

/// Sample albedo from the cubemap.  Returns linear-space color.
fn sample_albedo(body: &StaticSurfaceData, dir: Vec3) -> Vec3 {
    let (face, u, v) = dir_to_face_uv(dir);
    let res = body.albedo_cubemap.resolution();
    let x = ((u * res as f32) as u32).min(res - 1);
    let y = ((v * res as f32) as u32).min(res - 1);
    let texel = body.albedo_cubemap.get(face, x, y);
    Vec3::new(
        srgb_to_linear(texel[0]),
        srgb_to_linear(texel[1]),
        srgb_to_linear(texel[2]),
    )
}

fn sample_roughness(body: &StaticSurfaceData, dir: Vec3, material_id: u32) -> f32 {
    let texel = body.roughness_cubemap.sample_nearest(dir);
    if texel > 0 {
        texel as f32 / 255.0
    } else {
        body.materials
            .get(material_id as usize)
            .map(|m| m.roughness)
            .unwrap_or(0.5)
    }
}

/// Apply terrain-owned dynamic overlays to an existing static sample.
///
/// `height` is in metres above the reference sphere, `albedo` is linear RGB,
/// and `roughness` is normalized 0..1. This deliberately does not evaluate
/// the static crater/detail stack; callers that already have a cheap static
/// cubemap sample can use this to keep dynamic ice and dune overlays shared
/// with the full [`sample_surface`] path.
pub fn apply_dynamic_surface_layers(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    dir: Vec3,
    lod: f32,
    height: &mut f32,
    albedo: &mut Vec3,
    roughness: &mut f32,
) {
    for (index, layer) in surface.dynamic_layers.ice_caps.iter().enumerate() {
        let fallback;
        let state = match state.ice_cap_state(index, layer) {
            Some(state) => state,
            None => {
                fallback = IceCapState {
                    id: layer.id.clone(),
                    ..IceCapState::default()
                };
                &fallback
            }
        };
        apply_ice_cap(layer, state, dir, height, albedo, roughness);
    }

    for (index, layer) in surface.dynamic_layers.active_dunes.iter().enumerate() {
        let fallback;
        let state = match state.active_dune_state(index, layer) {
            Some(state) => state,
            None => {
                fallback = ActiveDuneState {
                    id: layer.id.clone(),
                    mobility: layer.mobility,
                    ..ActiveDuneState::default()
                };
                &fallback
            }
        };
        apply_active_dune(
            layer,
            state,
            surface.static_surface.radius_m,
            dir,
            lod,
            height,
            albedo,
            roughness,
        );
    }
}

fn dynamic_height_delta(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    dir: Vec3,
    lod: f32,
) -> f32 {
    let mut height = 0.0;
    let mut albedo = Vec3::ZERO;
    let mut roughness = 0.0;
    apply_dynamic_surface_layers(
        surface,
        state,
        dir,
        lod,
        &mut height,
        &mut albedo,
        &mut roughness,
    );
    height
}

fn apply_ice_cap(
    layer: &crate::types::IceCapLayer,
    state: &IceCapState,
    dir: Vec3,
    height: &mut f32,
    albedo: &mut Vec3,
    roughness: &mut f32,
) {
    let spec = layer.spec;
    if state.coverage_scale <= 0.0 || state.thickness_scale <= 0.0 || spec.max_thickness_m <= 0.0 {
        return;
    }

    let axis = safe_normalize(spec.axis, Vec3::Y);
    let latitude_deg = dir.dot(axis).clamp(-1.0, 1.0).asin().to_degrees();
    let mut coverage: f32 = 0.0;
    if spec.north {
        coverage = coverage.max(ice_pole_coverage(latitude_deg, &spec, state));
    }
    if spec.south {
        coverage = coverage.max(ice_pole_coverage(-latitude_deg, &spec, state));
    }
    coverage = (coverage * state.coverage_scale).clamp(0.0, 1.0);
    if coverage <= 0.0 {
        return;
    }

    *height += spec.max_thickness_m * state.thickness_scale.max(0.0) * coverage;

    let clean = Vec3::from_array(spec.albedo_linear);
    let dusty = Vec3::from_array(spec.dust_albedo_linear);
    let ice_albedo = clean.lerp(dusty, state.dustiness.clamp(0.0, 1.0));
    let albedo_t = (coverage * spec.albedo_strength).clamp(0.0, 1.0);
    *albedo = albedo.lerp(ice_albedo, albedo_t);

    let roughness_t = (coverage * spec.roughness_strength).clamp(0.0, 1.0);
    *roughness += (spec.roughness.clamp(0.0, 1.0) - *roughness) * roughness_t;
}

fn ice_pole_coverage(
    pole_latitude_deg: f32,
    spec: &crate::types::IceCapSpec,
    state: &IceCapState,
) -> f32 {
    let sharpness = spec.edge_sharpness.clamp(0.0, 1.0);
    let transition_deg = (1.25 + (0.32 - 1.25) * sharpness).max(0.25);
    let edge = (spec.edge_latitude_deg + state.edge_offset_deg).clamp(0.0, 89.5);
    let solid = (spec.solid_latitude_deg + state.edge_offset_deg)
        .max(edge + transition_deg + 0.35)
        .clamp(edge + 0.6, 90.0);
    let coverage = smoothstep(edge, edge + transition_deg, pole_latitude_deg);
    let interior = smoothstep(edge + transition_deg, solid, pole_latitude_deg);
    (coverage * (0.62 + 0.38 * interior)).clamp(0.0, 1.0)
}

fn apply_active_dune(
    layer: &crate::types::ActiveDuneLayer,
    state: &ActiveDuneState,
    body_radius_m: f32,
    dir: Vec3,
    lod: f32,
    height: &mut f32,
    albedo: &mut Vec3,
    roughness: &mut f32,
) {
    if state.coverage_scale <= 0.0 || state.amplitude_scale <= 0.0 {
        return;
    }

    let region = &layer.region;
    let center = safe_normalize(region.center, dir);
    let angular_distance = dir.dot(center).clamp(-1.0, 1.0).acos();
    let outer = region.radius_rad + region.feather_rad.max(0.0);
    let mut weight = 1.0
        - smoothstep(
            region.radius_rad,
            outer.max(region.radius_rad + 1e-5),
            angular_distance,
        );
    weight = (weight * state.coverage_scale).clamp(0.0, 1.0);
    if weight <= 0.0 {
        return;
    }

    let pixel_size_m = 2.0_f32.powf(lod).max(1.0);
    let draa_lod = smoothstep(4.0, 9.0, region.lambda_draa_m / pixel_size_m);
    let dune_lod = smoothstep(2.0, 5.0, region.lambda_dune_m / pixel_size_m);
    if draa_lod.max(dune_lod) <= 0.001 {
        let tint = Vec3::from_array(region.albedo_crest_lin);
        let broad_t = (weight * region.crest_strength * 0.34).clamp(0.0, 0.18);
        *albedo = albedo.lerp(tint * 0.68, broad_t * 0.42);
        *roughness += (0.78 - *roughness) * (weight * 0.18).clamp(0.0, 0.24);
        return;
    }

    let axis = safe_normalize(
        region.axis_tangent - center * region.axis_tangent.dot(center),
        {
            let up = if center.y.abs() > 0.99 {
                Vec3::X
            } else {
                Vec3::Y
            };
            center.cross(up).normalize()
        },
    );
    let local = dir - center * dir.dot(center);
    let along_m = local.dot(axis) * body_radius_m + state.phase_offset_m;
    let across = safe_normalize(center.cross(axis), Vec3::Z);
    let cross_m = local.dot(across) * body_radius_m;
    let broad_warp = simple_value_noise(
        cross_m / body_radius_m * region.warp_freq * 0.52,
        region.seed ^ 0x6D2B_79F5,
    );
    let lace_warp = simple_value_noise(
        cross_m / body_radius_m * region.warp_freq * 2.2 + 13.7,
        region.seed ^ 0x9E37_79B9,
    );
    let meander_m = (broad_warp * 0.75 + lace_warp * 0.25) * region.warp_amp_unit * body_radius_m;
    let wind_m = along_m + meander_m;

    let lobe = smoothstep(-0.20, 0.52, broad_warp * 0.62 + weight * 0.10);
    let wavelength_jitter = (1.0
        + simple_value_noise(
            cross_m / body_radius_m * region.warp_freq * 0.8 + 19.3,
            region.seed ^ 0xA24B_AED5,
        ) * 0.24)
        .clamp(0.72, 1.42);
    let draa = asymmetric_ridge(
        wind_m / (region.lambda_draa_m.max(1.0) * wavelength_jitter) + broad_warp * 0.35,
        region.alpha_skew,
    ) * draa_lod;
    let dune = asymmetric_ridge(
        wind_m / region.lambda_dune_m.max(1.0) + lace_warp * 0.42,
        region.alpha_skew,
    ) * dune_lod;
    let body = (draa * (0.16 + lobe * 1.05)).clamp(0.0, 1.0);
    let crest = (0.55 * body + 0.45 * dune).clamp(0.0, 1.0);
    let amp = state.amplitude_scale.max(0.0);
    *height += weight
        * amp
        * (region.amplitude_draa_m.max(0.0) * body + region.amplitude_dune_m.max(0.0) * dune);

    let tint = Vec3::from_array(region.albedo_crest_lin);
    let tint_t = (weight * crest * region.crest_strength).clamp(0.0, 1.0);
    *albedo = albedo.lerp(tint, tint_t);
    *roughness += (0.76 - *roughness) * (weight * (0.25 + 0.35 * crest)).clamp(0.0, 1.0);
}

fn asymmetric_ridge(phase: f32, alpha_skew: f32) -> f32 {
    let t = phase - phase.floor();
    let alpha = alpha_skew.clamp(0.05, 0.95);
    let tri = if t < alpha {
        t / alpha
    } else {
        1.0 - (t - alpha) / (1.0 - alpha)
    };
    tri.clamp(0.0, 1.0).powf(1.35)
}

fn simple_value_noise(x: f32, seed: u64) -> f32 {
    let i0 = x.floor() as i32;
    let i1 = i0 + 1;
    let t = smoothstep(0.0, 1.0, x - x.floor());
    let a = hash_cell(i0, 0, 0, 0, seed as u32, (seed >> 32) as u32);
    let b = hash_cell(i1, 0, 0, 0, seed as u32, (seed >> 32) as u32);
    (u32_to_unit(a) * 2.0 - 1.0) * (1.0 - t) + (u32_to_unit(b) * 2.0 - 1.0) * t
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() <= f32::EPSILON {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn safe_normalize(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.length_squared() > 1e-12 {
        v.normalize()
    } else {
        fallback
    }
}

fn srgb_to_linear(srgb: u8) -> f32 {
    let s = srgb as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

// ---------------------------------------------------------------------------
// Layer 2: explicit crater SSBO iteration
// ---------------------------------------------------------------------------

/// Accumulate crater height contributions from the spatial index.
///
/// Iterates the cell containing `dir` plus its neighbors (~13 cells), filters
/// to `FeatureRef::Crater`, and for each crater whose influence region
/// contains the sample point, evaluates the Pike/Krüger profile from
/// `crater_profile` and applies the screen-space-size fade.
fn sample_layer2_craters(body: &StaticSurfaceData, dir: Vec3, lod: f32) -> f32 {
    if body.craters.is_empty() {
        return 0.0;
    }

    let pixel_size_m = 2_f32.powf(lod);
    let body_radius = body.radius_m;
    let mut acc = 0.0_f32;

    for feat in body.feature_index.lookup_with_neighbors(dir) {
        let FeatureRef::Crater(idx) = feat else {
            continue;
        };
        let Some(crater) = body.craters.get(idx as usize) else {
            continue;
        };

        // Skip craters already rasterized into the cubemap — their height
        // contribution is in the Layer 1 texel. Iterating them here would
        // double the displacement.
        if crater.radius_m >= body.cubemap_bake_threshold_m {
            continue;
        }

        // Fade by screen-space size. Matches the shader's SSBO + hash
        // window (0.5 → 8 px) so sub-pixel features still contribute to
        // population-level shading. See the contract at the top of file.
        let diameter_m = 2.0 * crater.radius_m;
        let diameter_px = diameter_m / pixel_size_m.max(1e-6);
        let weight = smoothstep_range(0.5, 8.0, diameter_px);
        if weight <= 0.0 {
            continue;
        }

        if let Some(h) = crater_profile_at(crater, dir, body_radius) {
            acc += h * weight;
        }
    }

    acc
}

/// Evaluate a single crater's radial profile at a sample direction.
///
/// Returns `None` if the sample lies beyond the ejecta cutoff (5R). Uses the
/// same math as the Cratering stage's bake path: angular distance on the
/// sphere, age-based diffusion degradation, then `crater_profile` dispatch.
fn crater_profile_at(crater: &Crater, dir: Vec3, body_radius_m: f32) -> Option<f32> {
    let center = crater.center.normalize();
    let cos_theta = dir.dot(center).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let surface_dist = theta * body_radius_m;
    let t = surface_dist / crater.radius_m;
    if t > 5.0 {
        return None;
    }

    let degrad = degradation_factor(crater.radius_m, crater.age_gyr);
    let depth = crater.depth_m * degrad;
    let rim_h = crater.rim_height_m * degrad;
    let morph = morphology_for_radius(crater.radius_m);
    let softness = degradation_softness(crater.radius_m, crater.age_gyr);
    // SSBO sample path: no per-texel sub-peak rubble (the bake path
    // hashes from the crater seed; here we just want the smooth profile).
    let no_subs: SubPeaks = Default::default();
    Some(crater_profile(
        t,
        depth,
        rim_h,
        crater.radius_m,
        morph,
        0.0,
        0.0,
        0.0,
        &no_subs,
        1.0,
        softness,
    ))
}

// ---------------------------------------------------------------------------
// Layer 3: statistical detail noise
// ---------------------------------------------------------------------------
//
// CPU mirror of the shader's `synthesize_small_craters` in
// `assets/shaders/planet_impostor.wgsl`. Must stay bit-for-bit equivalent in
// shape so the seam between impostor and UDLOD is invisible — differences
// here are visible as stripes when the camera crosses the transition.
//
// The profile family is intentionally simpler than the explicit-crater Pike
// profiles: constant depth/rim ratios and a `pow(r, n)` interior. The shader
// cannot afford Pike morphometry per cell, and explicit craters don't need
// to match this simpler family because they're above the detail threshold.

const SIMPLE_DEPTH_RATIO: f32 = 0.2;
const SIMPLE_RIM_RATIO: f32 = 0.04;
const SIMPLE_INTERIOR_EXPONENT: f32 = 2.5;
const EJECTA_EXTENT: f32 = 2.5;
const COMPLEX_FLOOR_FRACTION: f32 = 0.55;
const COMPLEX_PEAK_HEIGHT_FRAC: f32 = 0.15;
const COMPLEX_PEAK_BASE_FRAC: f32 = 0.15;
const COMPLEX_MIN_DEPTH_RATIO: f32 = 0.05;

/// Evaluate the statistical detail layer at `dir` for LOD `lod`.
///
/// Returns `(height_delta_m, grad_tangent)` where `grad_tangent` is the
/// tangent-plane gradient of the accumulated field on the unit sphere.
/// The renderer tangent-projects it to perturb the shading normal; for the
/// CPU sampler we drop the gradient in `sample_height_only` (finite
/// differences pick it up).
pub fn sample_detail_noise(params: &DetailNoiseParams, dir: Vec3, lod: f32) -> (f32, Vec3) {
    let mut height = 0.0_f32;
    let mut grad_tangent = Vec3::ZERO;

    if params.global_k_per_km2 <= 0.0 || params.d_min_m <= 0.0 {
        return (height, grad_tangent);
    }

    let p_unit = dir.normalize();
    let pixel_size_m = 2_f32.powf(lod);
    let body_r = params.body_radius_m;
    let seed_lo = (params.seed & 0xFFFF_FFFF) as u32;
    let seed_hi = (params.seed >> 32) as u32;

    // 11 octaves, each doubling `d_lo`. Matches the shader loop bound.
    for oi in 0u32..11 {
        let d_lo = params.d_min_m * (1u32 << oi) as f32;
        let d_hi = (params.d_min_m * (1u32 << (oi + 1)) as f32).min(params.d_max_m);
        if d_hi <= d_lo {
            break;
        }

        // Whole-octave LOD cull: if even the largest diameter in this octave
        // falls below the per-crater fade threshold, skip all 27 hashes.
        if d_hi < 0.5 * pixel_size_m {
            continue;
        }

        let d_lo_km = d_lo * 1e-3;
        let d_hi_km = d_hi * 1e-3;
        let per_km2 = params.global_k_per_km2
            * (d_lo_km.powf(-params.sfd_alpha) - d_hi_km.powf(-params.sfd_alpha));

        let cell_size_m = 2.0 * d_hi;
        let cell_area_km2 = (cell_size_m * 1e-3) * (cell_size_m * 1e-3);
        let expected_per_cell = per_km2 * cell_area_km2 / 3.0;
        if expected_per_cell <= 0.0 {
            continue;
        }

        let cell_size_unit = cell_size_m / body_r;
        let inv_cell = 1.0 / cell_size_unit;
        let cx = (p_unit.x * inv_cell).floor() as i32;
        let cy = (p_unit.y * inv_cell).floor() as i32;
        let cz = (p_unit.z * inv_cell).floor() as i32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    visit_detail_cell(
                        p_unit,
                        cx + dx,
                        cy + dy,
                        cz + dz,
                        oi,
                        cell_size_unit,
                        d_lo,
                        d_hi,
                        expected_per_cell,
                        pixel_size_m,
                        params,
                        seed_lo,
                        seed_hi,
                        &mut height,
                        &mut grad_tangent,
                    );
                }
            }
        }
    }

    // Cap the accumulated tangent so a cell with many overlapping craters
    // can't blow up the gradient (matches the shader's safety clamp).
    let grad_len = grad_tangent.length();
    if grad_len > 2.0 {
        grad_tangent *= 2.0 / grad_len;
    }

    (height, grad_tangent)
}

#[allow(clippy::too_many_arguments)]
fn visit_detail_cell(
    p_unit: Vec3,
    ix: i32,
    iy: i32,
    iz: i32,
    octave: u32,
    cell_size_unit: f32,
    d_lo: f32,
    d_hi: f32,
    expected_per_cell: f32,
    pixel_size_m: f32,
    params: &DetailNoiseParams,
    seed_lo: u32,
    seed_hi: u32,
    height_acc: &mut f32,
    grad_acc: &mut Vec3,
) {
    let h_cell = hash_cell(ix, iy, iz, octave, seed_lo, seed_hi);
    let u_exists = u32_to_unit(h_cell);
    if u_exists >= expected_per_cell {
        return;
    }

    let u_diam = u32_to_unit(pcg(h_cell ^ 0x68E3_1DA4));
    let u_px = u32_to_unit(pcg(h_cell ^ 0xB529_7A4D));
    let u_py = u32_to_unit(pcg(h_cell ^ 0xBE54_66CF));
    let u_pz = u32_to_unit(pcg(h_cell ^ 0x1B87_3593));
    let u_age = u32_to_unit(pcg(h_cell ^ 0xD2A9_C4B1));
    let u_ellip = u32_to_unit(pcg(h_cell ^ 0xA1C4_E9F2));
    let u_orient = u32_to_unit(pcg(h_cell ^ 0x3F7B_8C21));
    let u_rim_ph = u32_to_unit(pcg(h_cell ^ 0x9D2E_5A73));
    let u_rim_lob = u32_to_unit(pcg(h_cell ^ 0x54F1_D8B6));

    let cell_origin = Vec3::new(ix as f32, iy as f32, iz as f32) * cell_size_unit;
    let cand = cell_origin + Vec3::new(u_px, u_py, u_pz) * cell_size_unit;
    let cand_len = cand.length();
    if cand_len < 1e-6 {
        return;
    }
    let center = cand / cand_len;

    let diameter_m = sample_diameter(u_diam, d_lo, d_hi, params.sfd_alpha);
    let diameter_px = diameter_m / pixel_size_m.max(1e-6);
    // Shader uses smoothstep(0.5, 8.0, diameter_px) for the detail layer's
    // fade — wider than the explicit-crater 0.5..4.0 range so sub-pixel
    // octaves still contribute as statistical texture. Match exactly.
    let lod_weight = smoothstep_range(0.5, 8.0, diameter_px);
    if lod_weight <= 0.0 {
        return;
    }

    let radius_m = 0.5 * diameter_m;

    let cos_theta = p_unit.dot(center).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let s_arc_m = theta * params.body_radius_m;
    let r0 = s_arc_m / radius_m;
    if r0 >= EJECTA_EXTENT {
        return;
    }

    // Tangent-plane projection for azimuth + gradient direction.
    let proj = center - cos_theta * p_unit;
    let proj_len2 = proj.length_squared();
    let azimuth = if proj_len2 > 1e-12 {
        (proj.x).atan2(proj.y + proj.z * 0.7)
    } else {
        0.0
    };

    let ellipticity = u_ellip * 0.2;
    let ellip_angle = u_orient * std::f32::consts::TAU;
    let ellip_factor = 1.0 + ellipticity * (2.0 * (azimuth - ellip_angle)).cos();
    let r = r0 / ellip_factor;

    let rim_lobes = (u_rim_lob * 4.0 + 3.0).floor();
    let rim_phase = u_rim_ph * std::f32::consts::TAU;
    let rim_irregular = if r > 0.85 && r < 1.15 {
        let wave = (rim_lobes * azimuth + rim_phase).sin();
        let band = (1.0 - 4.0 * (r - 1.0) * (r - 1.0)).max(0.0);
        0.35 * wave * band
    } else {
        0.0
    };

    let d_over_dsc = diameter_m / params.d_sc_m;
    let depth_ratio = if d_over_dsc >= 1.0 {
        complex_depth_ratio(d_over_dsc)
    } else {
        SIMPLE_DEPTH_RATIO
    };
    let age_gyr = u_age * params.body_age_gyr;
    let degradation = degradation_factor(radius_m, age_gyr);
    let depth = diameter_m * depth_ratio * degradation;
    let rim = diameter_m * SIMPLE_RIM_RATIO * degradation;

    let (h_m, dh_dr) = if d_over_dsc >= 1.0 {
        detail_complex_profile(r, depth, rim)
    } else {
        detail_simple_profile(r, depth, rim)
    };
    let h_total = h_m + rim_irregular * rim;

    *height_acc += h_total * lod_weight;

    let grad_proj_len = proj_len2.sqrt();
    if grad_proj_len < 1e-8 {
        return;
    }
    let t_hat = proj / grad_proj_len;
    // dh/dr is derivative with respect to r (unitless, normalized radius).
    // To convert to a per-arc-length gradient: dh/ds = dh/dr * (1/radius_m).
    // Sign flips because moving from the center outward decreases `center
    // - cos_theta * p_unit` in the direction we projected.
    let grad = -(dh_dr) / radius_m * t_hat;
    *grad_acc += grad * lod_weight;
}

#[inline]
fn complex_depth_ratio(d_over_dsc: f32) -> f32 {
    let t = (-((d_over_dsc - 1.0).max(0.0)) / 3.0).exp();
    COMPLEX_MIN_DEPTH_RATIO + (SIMPLE_DEPTH_RATIO - COMPLEX_MIN_DEPTH_RATIO) * t
}

/// Shader-side simple profile. Returns (height, dh/dr).
fn detail_simple_profile(r: f32, depth: f32, rim: f32) -> (f32, f32) {
    if r <= 1.0 {
        let n = SIMPLE_INTERIOR_EXPONENT;
        let h = -depth + (depth + rim) * r.powf(n);
        let dh = (depth + rim) * n * r.powf(n - 1.0);
        (h, dh)
    } else {
        let span = EJECTA_EXTENT - 1.0;
        let t = ((r - 1.0) / span).clamp(0.0, 1.0);
        let s_taper = t * t * (3.0 - 2.0 * t);
        let fade = 1.0 - s_taper;
        let dfade_dr = -6.0 * t * (1.0 - t) / span;

        let base = rim / (r * r * r);
        let dbase_dr = -3.0 * rim / (r * r * r * r);

        (base * fade, dbase_dr * fade + base * dfade_dr)
    }
}

/// Shader-side complex profile.
fn detail_complex_profile(r: f32, depth: f32, rim: f32) -> (f32, f32) {
    let (base_h, base_dh) = if r <= 1.0 {
        if r <= COMPLEX_FLOOR_FRACTION {
            (-depth, 0.0)
        } else {
            let span = 1.0 - COMPLEX_FLOOR_FRACTION;
            let t = (r - COMPLEX_FLOOR_FRACTION) / span;
            let s = t * t * (3.0 - 2.0 * t);
            let ds_dr = 6.0 * t * (1.0 - t) / span;
            let h_total = depth + rim;
            (-depth + h_total * s, h_total * ds_dr)
        }
    } else {
        let span = EJECTA_EXTENT - 1.0;
        let t = ((r - 1.0) / span).clamp(0.0, 1.0);
        let s_taper = t * t * (3.0 - 2.0 * t);
        let fade = 1.0 - s_taper;
        let dfade_dr = -6.0 * t * (1.0 - t) / span;
        let raw = rim / (r * r * r);
        let draw_dr = -3.0 * rim / (r * r * r * r);
        (raw * fade, draw_dr * fade + raw * dfade_dr)
    };
    let sigma = COMPLEX_PEAK_BASE_FRAC;
    let g = (-(r * r) / (2.0 * sigma * sigma)).exp();
    let peak = COMPLEX_PEAK_HEIGHT_FRAC * depth * g;
    let dpeak = -COMPLEX_PEAK_HEIGHT_FRAC * depth * g * (r / (sigma * sigma));
    (base_h + peak, base_dh + dpeak)
}

/// Inverse-CDF sample of a power-law SFD.
fn sample_diameter(u: f32, d_lo: f32, d_hi: f32, alpha: f32) -> f32 {
    let lo = d_lo.powf(-alpha);
    let hi = d_hi.powf(-alpha);
    let y = lo + (hi - lo) * u;
    y.powf(-1.0 / alpha)
}

// ---------------------------------------------------------------------------
// Hash primitives (mirror assets/shaders/planet_impostor.wgsl)
// ---------------------------------------------------------------------------

fn pcg(x: u32) -> u32 {
    let state = x.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

fn hash_cell(ix: i32, iy: i32, iz: i32, octave: u32, seed_lo: u32, seed_hi: u32) -> u32 {
    let ux = ix as u32;
    let uy = iy as u32;
    let uz = iz as u32;
    let mut h = ux.wrapping_mul(73856093);
    h ^= uy.wrapping_mul(19349663);
    h ^= uz.wrapping_mul(83492791);
    h = pcg(h);
    h ^= octave.wrapping_mul(2654435769);
    h ^= seed_lo;
    h = pcg(h);
    h ^= seed_hi.wrapping_mul(1540483477);
    pcg(h)
}

#[inline]
fn u32_to_unit(x: u32) -> f32 {
    x as f32 / 4294967296.0
}

// ---------------------------------------------------------------------------
// Normal via LOD-aware finite differences
// ---------------------------------------------------------------------------

/// Compute the surface normal via finite differences on the full height field.
///
/// LOD-aware: at coarse LOD the sample offset stays at the cubemap texel
/// scale; at near LOD it tracks `pixel_size_m / body.radius_m` so the
/// derivative reflects only the bands actually resolvable at that LOD.
///
/// Each probe re-enters `sample_height_only`, so Layer 2 and Layer 3
/// contributions feed the normal just like Layer 1 does. Recursion is
/// bounded — `sample_height_only` never calls `compute_normal`.
fn compute_normal(body: &StaticSurfaceData, dir: Vec3, lod: f32) -> Vec3 {
    // Build a continuous tangent frame on the sphere at `dir`.
    // The `dir.y > 0.99` branch is a coarse fallback — fine for the offsets
    // we actually use; the UV artifact from flipping tangent is bounded to
    // four probe samples and averages out in the finite difference.
    let up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    let tangent = dir.cross(up).normalize();
    let bitangent = tangent.cross(dir);

    // LOD-aware offset: at coarse LOD, use the cubemap texel scale so the
    // gradient reflects the baked layer. At near LOD, track the sample
    // spacing so the derivative resolves the finer bands (SSBO + detail).
    let texel_offset = 1.5 / body.height_cubemap.resolution() as f32;
    let pixel_size_m = 2_f32.powf(lod);
    let pixel_offset = pixel_size_m / body.radius_m;
    let offset = texel_offset.max(pixel_offset);

    let h_east = sample_height_only(body, (dir + tangent * offset).normalize(), lod);
    let h_west = sample_height_only(body, (dir - tangent * offset).normalize(), lod);
    let h_north = sample_height_only(body, (dir + bitangent * offset).normalize(), lod);
    let h_south = sample_height_only(body, (dir - bitangent * offset).normalize(), lod);

    // Convert the angular offset to a world-space arc length.
    let ds = body.radius_m * offset * 2.0;
    let dh_dt = (h_east - h_west) / ds;
    let dh_db = (h_north - h_south) / ds;

    (dir - tangent * dh_dt - bitangent * dh_db).normalize()
}

fn compute_surface_normal(
    surface: &PlanetSurface,
    state: &DynamicSurfaceState,
    dir: Vec3,
    lod: f32,
) -> Vec3 {
    let body = &surface.static_surface;
    let up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    let tangent = dir.cross(up).normalize();
    let bitangent = tangent.cross(dir);

    let texel_offset = 1.5 / body.height_cubemap.resolution() as f32;
    let pixel_size_m = 2_f32.powf(lod);
    let pixel_offset = pixel_size_m / body.radius_m;
    let offset = texel_offset.max(pixel_offset);

    let height_at = |probe: Vec3| {
        let probe = probe.normalize();
        sample_height_only(body, probe, lod) + dynamic_height_delta(surface, state, probe, lod)
    };

    let h_east = height_at(dir + tangent * offset);
    let h_west = height_at(dir - tangent * offset);
    let h_north = height_at(dir + bitangent * offset);
    let h_south = height_at(dir - bitangent * offset);

    let ds = body.radius_m * offset * 2.0;
    let dh_dt = (h_east - h_west) / ds;
    let dh_db = (h_north - h_south) / ds;

    (dir - tangent * dh_dt - bitangent * dh_db).normalize()
}
