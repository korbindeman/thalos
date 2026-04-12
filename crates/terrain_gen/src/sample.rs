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
//! │ Feature SSBO   │ 500 m – 5 km    │ BodyData.craters + feature_index.  │
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

use crate::body_data::BodyData;
use crate::crater_profile::{
    crater_profile, degradation_factor, morphology_for_radius, smoothstep_range,
};
use crate::cubemap::dir_to_face_uv;
use crate::spatial_index::FeatureRef;
use crate::types::{Crater, DetailNoiseParams};

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
    /// Index into `BodyData::materials`.
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
pub fn sample(body: &BodyData, dir: Vec3, lod: f32) -> SurfaceSample {
    let dir = dir.normalize();

    let height = sample_height_only(body, dir, lod);
    let normal = compute_normal(body, dir, lod);
    let albedo = sample_albedo(body, dir);

    let material_id = body.material_cubemap.sample_nearest(dir) as u32;
    let roughness = body
        .materials
        .get(material_id as usize)
        .map(|m| m.roughness)
        .unwrap_or(0.5);

    SurfaceSample {
        height,
        normal,
        albedo,
        roughness,
        material_id,
    }
}

// ---------------------------------------------------------------------------
// Height (Layer 1 + 2 + 3), no normal.  Used directly by `sample()` and
// recursively by `compute_normal` for finite-difference offsets.
// ---------------------------------------------------------------------------

/// Evaluate the full three-layer height at a direction.
///
/// This function does not compute a normal and does not recurse into
/// `compute_normal`. The finite-difference normal derivation is the only
/// thing that calls this directly with offset directions.
fn sample_height_only(body: &BodyData, dir: Vec3, lod: f32) -> f32 {
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
fn sample_cubemap_height(body: &BodyData, dir: Vec3) -> f32 {
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
fn sample_albedo(body: &BodyData, dir: Vec3) -> Vec3 {
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

fn srgb_to_linear(srgb: u8) -> f32 {
    let s = srgb as f32 / 255.0;
    if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
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
fn sample_layer2_craters(body: &BodyData, dir: Vec3, lod: f32) -> f32 {
    if body.craters.is_empty() {
        return 0.0;
    }

    let pixel_size_m = 2_f32.powf(lod);
    let body_radius = body.radius_m;
    let mut acc = 0.0_f32;

    for feat in body.feature_index.lookup_with_neighbors(dir) {
        let FeatureRef::Crater(idx) = feat else { continue };
        let Some(crater) = body.craters.get(idx as usize) else { continue };

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
    Some(crater_profile(t, depth, rim_h, crater.radius_m, morph))
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

const FRESH_AGE_GYR: f32 = 0.1;
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
pub fn sample_detail_noise(
    params: &DetailNoiseParams,
    dir: Vec3,
    lod: f32,
) -> (f32, Vec3) {
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
    let depth = diameter_m * depth_ratio;
    let rim = diameter_m * SIMPLE_RIM_RATIO;

    let (h_m, dh_dr) = if d_over_dsc >= 1.0 {
        detail_complex_profile(r, depth, rim)
    } else {
        detail_simple_profile(r, depth, rim)
    };
    let h_total = h_m + rim_irregular * rim;

    // Age blend: young craters (age < FRESH_AGE_GYR) retain their crisp
    // maturity; mature craters are tuned down. This doesn't affect height
    // directly in our CPU sampler (we only track height + gradient) but
    // we compute it so the structure stays symmetric with the shader.
    let _age_gyr = u_age * params.body_age_gyr;
    let _ = FRESH_AGE_GYR;

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
fn compute_normal(body: &BodyData, dir: Vec3, lod: f32) -> Vec3 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body_builder::BodyBuilder;
    use crate::types::{Composition, Crater, Material};

    fn test_composition() -> Composition {
        Composition::new(0.9, 0.05, 0.0, 0.05, 0.0)
    }

    #[test]
    fn sample_default_body_returns_flat_surface() {
        let builder = BodyBuilder::new(
            869_000.0,
            42,
            test_composition(),
            4, // small resolution for test speed
            4.5,
            None,
        );
        let body = builder.build();

        let s = sample(&body, Vec3::X, 10.0);
        assert!(s.height.abs() < 1.0, "expected ~0 height, got {}", s.height);
        assert!(s.normal.dot(Vec3::X) > 0.99, "normal should point outward");
    }

    #[test]
    fn sample_near_crater_returns_depressed_height() {
        // Build a body with no stages, then inject a single explicit crater
        // so the feature spatial index contains one element pointing at
        // +X. Sampling at the crater center must land in the bowl below
        // the flat cubemap background.
        let mut builder = BodyBuilder::new(
            869_000.0,
            7,
            test_composition(),
            8,
            4.5,
            None,
        );
        builder.materials.push(Material {
            albedo: [0.5, 0.5, 0.5],
            roughness: 0.5,
        });

        // 2 km radius simple crater (diameter 4000 m). At lod=5 (32 m/sample)
        // diameter_px = 4000/32 = 125, well above the 4 px fade-in saturation.
        builder.craters.push(Crater {
            center: Vec3::X,
            radius_m: 2_000.0,
            depth_m: 800.0,
            rim_height_m: 150.0,
            age_gyr: 0.0, // zero diffusion degradation
            material_id: 0,
        });

        let body = builder.build();

        // Background: the cubemap is all zeros, so sampling off-crater must
        // be ~0. This is our reference.
        let off = sample(&body, Vec3::Y, 5.0);
        assert!(off.height.abs() < 1.0, "off-crater background height: {}", off.height);

        let s = sample(&body, Vec3::X, 5.0);
        assert!(
            s.height < -100.0,
            "expected strong depression from crater bowl, got {}",
            s.height,
        );
    }
}
