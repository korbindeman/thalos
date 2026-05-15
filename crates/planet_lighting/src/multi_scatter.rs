//! CPU-baked multi-scattering LUT for the atmosphere raymarch.
//!
//! Single-scattering on its own produces a physically dim daytime sky and an
//! exaggerated warm horizon: only the residual red transmits through the long
//! horizontal path. On Earth, the missing brightness and the blue fill-in
//! both come from light that has scattered two or more times before reaching
//! the camera. We add an approximation of that contribution by storing, for
//! each (sun-zenith angle, altitude) cell, the average single-scattered
//! radiance arriving at the cell from every direction on the sphere.
//!
//! The runtime atmosphere shader samples this LUT at every view step and
//! adds `σ_s · ρ · L_ms · T_view` to the in-scatter integral. Because the
//! LUT is precomputed per body (parameters don't change at runtime), the
//! shader-side cost is one texture sample per view step — eight samples on
//! the current 8-step raymarch.
//!
//! Following Hillaire 2020 (§5.2). Each cell is approximated by a single
//! bounce, `L_ms ≈ L_2`, rather than the full geometric series
//! `L_2 / (1 − F)` — sufficient to recover the perceptual blue cast of a
//! midday sky and to balance the in-scatter against star brightness without
//! the extra cost of computing F per cell.
//!
//! The LUT axes are linear in `μ_s = cos(sun_zenith)` and altitude. 32×32 is
//! plenty for a smooth output; higher resolution buys nothing visible at this
//! number of view steps.

use crate::AtmosphereBlock;
use glam::Vec3;

const PI: f32 = std::f32::consts::PI;

/// View-ray steps along each outgoing sphere direction during the bake. Mirrors
/// the runtime raymarch step count — going higher inside the bake doesn't help
/// because the per-cell average is already smoothing across 32 directions.
const BAKE_VIEW_STEPS: usize = 8;
/// Sun-column steps per sample point. Same rationale as `BAKE_VIEW_STEPS`.
const BAKE_SUN_STEPS: usize = 6;
/// Outgoing sphere directions per LUT cell. Fibonacci sampling keeps the
/// distribution near-uniform with no clumping at the poles.
const BAKE_SPHERE_DIRECTIONS: usize = 32;

/// Default LUT resolution. Hillaire's reference is 32×32; we match it.
pub const MULTI_SCATTER_LUT_WIDTH: u32 = 32;
pub const MULTI_SCATTER_LUT_HEIGHT: u32 = 32;

/// Bake the multi-scatter LUT for one body's atmosphere.
///
/// Output layout: row-major `(width × height)` cells, RGBA f32 per cell in
/// little-endian byte order, ready to be handed straight to `Image::new` with
/// `TextureFormat::Rgba32Float`. Each cell stores the estimated isotropic
/// incoming radiance (per unit sun flux × strength) at the cell's (μ_s, h)
/// coordinate; alpha is unused (set to 1.0).
///
/// `planet_radius_render` is the body's solid radius in the same render units
/// as `atmos.atmos_geom.x` — i.e., post-`AtmosphereBlock::from_terrestrial`
/// scaling. Caller is responsible for using the same `meters_per_render_unit`
/// when building the atmosphere block and when supplying the planet radius
/// here, or the LUT will be sampled at the wrong scale.
pub fn bake_multi_scatter_lut(
    atmos: &AtmosphereBlock,
    planet_radius_render: f32,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let h_top = atmos.atmos_geom.x;
    let h_r = atmos.rayleigh_beta_h.w.max(1e-3);
    let h_m = atmos.atmos_geom.y.max(1e-3);
    let beta_r = Vec3::new(
        atmos.rayleigh_beta_h.x,
        atmos.rayleigh_beta_h.y,
        atmos.rayleigh_beta_h.z,
    );
    let beta_m = atmos.mie_beta_g.x;
    let g = atmos.mie_beta_g.w;
    let atmos_top_r = planet_radius_render + h_top;

    let cell_count = (width * height) as usize;
    let mut data = Vec::with_capacity(cell_count * 4 * 4);

    for j in 0..height {
        let v = (j as f32 + 0.5) / height as f32;
        let h = v * h_top;
        for i in 0..width {
            let u = (i as f32 + 0.5) / width as f32;
            let mu_s = u * 2.0 - 1.0;

            let l_ms = bake_cell(
                mu_s,
                h,
                planet_radius_render,
                atmos_top_r,
                beta_r,
                beta_m,
                h_r,
                h_m,
                g,
            );

            data.extend_from_slice(&l_ms.x.to_le_bytes());
            data.extend_from_slice(&l_ms.y.to_le_bytes());
            data.extend_from_slice(&l_ms.z.to_le_bytes());
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }
    data
}

fn bake_cell(
    mu_s: f32,
    h: f32,
    planet_r: f32,
    atmos_top_r: f32,
    beta_r: Vec3,
    beta_m: f32,
    h_r: f32,
    h_m: f32,
    g: f32,
) -> Vec3 {
    // Place P on the +Y axis at altitude h. The cell is symmetric in azimuth
    // around the local zenith so picking any axis is fine — the sun direction
    // we construct below uses Y as the zenith.
    let p = Vec3::Y * (planet_r + h);
    let sun_dir = Vec3::new((1.0 - mu_s * mu_s).max(0.0).sqrt(), mu_s, 0.0);

    let mut l_2 = Vec3::ZERO;
    for i in 0..BAKE_SPHERE_DIRECTIONS {
        let s_dir = fibonacci_sphere(i, BAKE_SPHERE_DIRECTIONS);
        let t_exit = compute_t_exit(p, s_dir, planet_r, atmos_top_r);
        if t_exit <= 1e-3 {
            continue;
        }
        l_2 += single_scatter_integral(
            p,
            s_dir,
            sun_dir,
            0.0,
            t_exit,
            planet_r,
            atmos_top_r,
            beta_r,
            beta_m,
            h_r,
            h_m,
            g,
        );
    }
    // Isotropic-phase average: ∫_sphere L_1(ω) · (1 / 4π) dω
    // → (1 / N) · Σ L_1(ω_i) with uniform sphere sampling. Conveniently, the
    // 4π solid angle and 1 / 4π phase cancel.
    l_2 / BAKE_SPHERE_DIRECTIONS as f32
}

fn fibonacci_sphere(i: usize, n: usize) -> Vec3 {
    let golden = PI * (3.0 - 5.0_f32.sqrt());
    let y = 1.0 - (i as f32 / (n as f32 - 1.0)) * 2.0;
    let r = (1.0 - y * y).max(0.0).sqrt();
    let theta = golden * i as f32;
    Vec3::new(theta.cos() * r, y, theta.sin() * r)
}

fn compute_t_exit(p: Vec3, dir: Vec3, planet_r: f32, atmos_top_r: f32) -> f32 {
    let b = p.dot(dir);
    let c_a = p.length_squared() - atmos_top_r * atmos_top_r;
    let disc_a = b * b - c_a;
    if disc_a < 0.0 {
        return -1.0;
    }
    let mut t_exit = -b + disc_a.sqrt();

    let c_p = p.length_squared() - planet_r * planet_r;
    let disc_p = b * b - c_p;
    if disc_p > 0.0 {
        let t_p = -b - disc_p.sqrt();
        if t_p > 1e-3 {
            t_exit = t_exit.min(t_p);
        }
    }
    t_exit.max(0.0)
}

#[allow(clippy::too_many_arguments)]
fn single_scatter_integral(
    p_origin: Vec3,
    ray_dir: Vec3,
    sun_dir: Vec3,
    t_enter: f32,
    t_exit: f32,
    planet_r: f32,
    atmos_top_r: f32,
    beta_r: Vec3,
    beta_m: f32,
    h_r: f32,
    h_m: f32,
    g: f32,
) -> Vec3 {
    let path = t_exit - t_enter;
    if path <= 0.0 {
        return Vec3::ZERO;
    }

    let cos_theta = ray_dir.dot(sun_dir).clamp(-1.0, 1.0);
    let p_r = (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).max(1e-6);
    let p_m = (1.0 / (4.0 * PI)) * (1.0 - g2) / denom.powf(1.5);

    let ds = path / BAKE_VIEW_STEPS as f32;
    let mut sum_r = Vec3::ZERO;
    let mut sum_m = Vec3::ZERO;
    let mut od_r = 0.0f32;
    let mut od_m = 0.0f32;

    for i in 0..BAKE_VIEW_STEPS {
        let t = t_enter + (i as f32 + 0.5) * ds;
        let p_pt = p_origin + ray_dir * t;
        let h = (p_pt.length() - planet_r).max(0.0);
        let rho_r = (-h / h_r).exp();
        let rho_m = (-h / h_m).exp();
        od_r += rho_r * ds;
        od_m += rho_m * ds;

        let tau_view = beta_r * od_r + Vec3::splat(beta_m) * od_m;
        let trans_view = Vec3::new(
            (-tau_view.x).exp(),
            (-tau_view.y).exp(),
            (-tau_view.z).exp(),
        );

        let tau_sun = sun_optical_depth(
            p_pt,
            sun_dir,
            planet_r,
            atmos_top_r,
            beta_r,
            beta_m,
            h_r,
            h_m,
        );
        let trans_sun = Vec3::new((-tau_sun.x).exp(), (-tau_sun.y).exp(), (-tau_sun.z).exp());

        let weight = trans_view * trans_sun * ds;
        sum_r += rho_r * weight;
        sum_m += rho_m * weight;
    }

    beta_r * (sum_r * p_r) + Vec3::splat(beta_m) * (sum_m * p_m)
}

#[allow(clippy::too_many_arguments)]
fn sun_optical_depth(
    p: Vec3,
    sun_dir: Vec3,
    planet_r: f32,
    atmos_top_r: f32,
    beta_r: Vec3,
    beta_m: f32,
    h_r: f32,
    h_m: f32,
) -> Vec3 {
    let half_b = p.dot(sun_dir);
    let oc_len_sq = p.length_squared();

    let c_p = oc_len_sq - planet_r * planet_r;
    let disc_p = half_b * half_b - c_p;
    if disc_p > 0.0 {
        let t_p = -half_b - disc_p.sqrt();
        if t_p > 1e-3 {
            // exp(-40) ≈ 0 — sun is occluded by the body.
            return Vec3::splat(40.0);
        }
    }

    let c_a = oc_len_sq - atmos_top_r * atmos_top_r;
    let disc_a = half_b * half_b - c_a;
    if disc_a < 0.0 {
        return Vec3::ZERO;
    }
    let t_a = -half_b + disc_a.sqrt();
    if t_a <= 0.0 {
        return Vec3::ZERO;
    }

    let ds = t_a / BAKE_SUN_STEPS as f32;
    let mut od_r = 0.0f32;
    let mut od_m = 0.0f32;
    for i in 0..BAKE_SUN_STEPS {
        let t = (i as f32 + 0.5) * ds;
        let q = p + sun_dir * t;
        let h = (q.length() - planet_r).max(0.0);
        od_r += (-h / h_r).exp() * ds;
        od_m += (-h / h_m).exp() * ds;
    }
    beta_r * od_r + Vec3::splat(beta_m) * od_m
}
