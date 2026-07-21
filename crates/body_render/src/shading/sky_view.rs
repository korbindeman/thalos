//! CPU sky-view LUT — a physically-raymarched sky radiance field (graphics F3).
//!
//! Hillaire 2020's sky-view LUT: for a fixed camera altitude and sun direction,
//! the sky radiance along any view ray is a smooth 2-D function of
//! `(azimuth relative to the sun, view zenith)`. We bake that function once into
//! a small LUT by running the **same** single- + multiple-scattering integral the
//! GPU runs in `integrate_atmosphere_multiscatter` (`atmosphere.wgsl`), reusing
//! the shared raymarch primitives in [`super::multi_scatter`]. A consumer then
//! reads the whole sky with one bilinear [`SkyViewLut::sample`] per direction
//! instead of a full raymarch per direction.
//!
//! # Why CPU / why here
//!
//! This is the **mechanism** half of graphics-fidelity F3 (see
//! `docs/graphics_fidelity.md` §3), so it lives in `thalos_body_render`. Its first
//! consumer is the game's reflection probe (`crates/game/src/reflection_probe.rs`),
//! which already paints its environment cubemap on the CPU on a slow cadence: it
//! swaps its hand-kept analytic `cpu_surface_sky` (a WGSL mirror that had to be
//! kept in lockstep with `compute_surface_sky`) for a sample of this **physical**
//! LUT, so the metallic hull and dielectric structures reflect the real
//! atmosphere-derived sky — one atmosphere, one environment. F4 then projects the
//! same LUT to SH for the terrain/`StandardMaterial` ambient.
//!
//! # Units
//!
//! Radiance is returned in the caller's `sun_flux` units: pass the scene-flux sun
//! irradiance (`LIGHT_AT_1AU·(AU/d)²·gain`) and the LUT is in the same scene-flux
//! space as every other spine surface, so it shares the scene exposure. The bake
//! is scale-invariant (optical depths are dimensionless), so any *consistent*
//! length unit for `planet_radius_render` / `altitude` / the atmosphere block
//! works — the reflection probe bakes in meters (`meters_per_render_unit = 1`).

use crate::shading::AtmosphereBlock;
use crate::shading::multi_scatter::{MultiScatterLut, compute_t_exit, sun_optical_depth};
use glam::Vec3;

const PI: f32 = std::f32::consts::PI;

/// Default azimuth resolution (columns): `u ∈ [0,1] ↔ azimuth-from-sun `[0, π]`.
/// The field is symmetric about the sun-vertical plane, so half the circle is
/// enough. 64 is smooth for a reflection source.
pub const SKY_VIEW_LUT_WIDTH: u32 = 64;
/// Default zenith resolution (rows): `v ∈ [0,1] ↔ view-zenith angle `[0, π]``
/// (0 = straight up, 1 = straight down). Below-horizon rows integrate a short
/// occluded path and read ~0; consumers fill those with a ground term.
pub const SKY_VIEW_LUT_HEIGHT: u32 = 96;

/// View raymarch steps per LUT cell. Mirrors the GPU `ATMOS_VIEW_STEPS`.
const SKY_VIEW_STEPS: usize = 16;

/// Physically-raymarched sky radiance for one `(sun direction, altitude)`,
/// stored as a 2-D LUT over `(azimuth-relative-to-sun, view zenith)`.
///
/// Baked in a local frame (zenith = +Y, the sun's horizontal projection = +X);
/// [`sample`](SkyViewLut::sample) maps an arbitrary world view direction back
/// through the stored world `up` / `sun_dir` — the field is rotationally
/// symmetric about the zenith, so only the azimuth *relative to the sun* matters.
#[derive(Clone)]
pub struct SkyViewLut {
    width: u32,
    height: u32,
    /// Row-major `width × height` sky radiance cells (caller's `sun_flux` units).
    cells: Vec<Vec3>,
    /// World zenith the LUT was baked for (unit).
    up: Vec3,
    /// World sun direction the LUT was baked for (unit).
    sun_dir: Vec3,
}

impl SkyViewLut {
    /// Bake the sky-view LUT for a camera at `altitude` above the surface, looking
    /// out under sun direction `sun_dir` (world), with local zenith `up` (world).
    ///
    /// `ms` is the body's multi-scatter LUT ([`MultiScatterLut::bake`]); it is
    /// baked once per body (static) and reused across sun/altitude changes.
    /// `sun_flux` sets the output radiance units (see the module note).
    #[allow(clippy::too_many_arguments)]
    pub fn bake(
        atmos: &AtmosphereBlock,
        planet_radius_render: f32,
        altitude: f32,
        sun_dir: Vec3,
        up: Vec3,
        sun_flux: f32,
        ms: &MultiScatterLut,
        width: u32,
        height: u32,
    ) -> Self {
        let up = up.normalize_or(Vec3::Y);
        let sun_dir = sun_dir.normalize_or(Vec3::Y);

        let beta_r = atmos.rayleigh_beta_h.truncate();
        let beta_m = atmos.mie_beta_g.x;
        let h_r = atmos.rayleigh_beta_h.w.max(1e-3);
        let h_m = atmos.atmos_geom.y.max(1e-3);
        let atmos_top_alt = atmos.atmos_geom.x.max(1e-3);
        let atmos_top_r = planet_radius_render + atmos_top_alt;
        let strength = atmos.atmos_geom.z;
        let multi_gain = atmos.atmos_geom.w;
        let g = atmos.mie_beta_g.w;

        // Local frame: zenith = +Y, sun's horizontal projection along +X.
        // `s = sin(sun elevation)` = cos(sun zenith).
        let s = up.dot(sun_dir).clamp(-1.0, 1.0);
        let sun_local = Vec3::new((1.0 - s * s).max(0.0).sqrt(), s, 0.0);
        let p = Vec3::Y * (planet_radius_render + altitude.max(0.0));

        let mut cells = Vec::with_capacity((width * height) as usize);
        for j in 0..height {
            // v ↔ view-zenith angle θ ∈ [0, π] (0 = up, π = down).
            let v = (j as f32 + 0.5) / height as f32;
            let theta = v * PI;
            let (sin_t, cos_t) = theta.sin_cos();
            for i in 0..width {
                // u ↔ azimuth φ ∈ [0, π] from the sun's horizontal direction.
                let u = (i as f32 + 0.5) / width as f32;
                let (sin_a, cos_a) = (u * PI).sin_cos();
                let view = Vec3::new(sin_t * cos_a, cos_t, sin_t * sin_a);
                cells.push(sky_ray_radiance(
                    p,
                    view,
                    sun_local,
                    planet_radius_render,
                    atmos_top_r,
                    atmos_top_alt,
                    beta_r,
                    beta_m,
                    h_r,
                    h_m,
                    g,
                    strength,
                    multi_gain,
                    sun_flux,
                    ms,
                ));
            }
        }

        Self {
            width,
            height,
            cells,
            up,
            sun_dir,
        }
    }

    /// Cosine-weighted hemispherical **irradiance** from the sky above the local
    /// horizon — the illuminance a flat, up-facing surface receives from the sky,
    /// in the LUT's `sun_flux` (scene-flux) units. This is the SH DC term: the
    /// single physical number that drives a flat sky-fill ambient (graphics F4).
    /// It already encodes time-of-day (→ ~0 at night), sun elevation, and the
    /// atmosphere. Monte-Carlo estimate over the sphere:
    /// `E = ∫ L(ω)·max(cosθ,0) dω ≈ (4π/N)·Σ L(ωᵢ)·max(dot(ωᵢ, up), 0)`.
    pub fn ambient_sky_irradiance(&self) -> Vec3 {
        const N: usize = 128;
        let mut acc = Vec3::ZERO;
        for i in 0..N {
            let dir = fibonacci_sphere(i, N);
            let cos = dir.dot(self.up).max(0.0);
            if cos <= 0.0 {
                continue;
            }
            acc += self.sample(dir) * cos;
        }
        acc * (4.0 * PI / N as f32)
    }

    /// Sky radiance along world view direction `view_dir` (need not be
    /// normalized). Below-horizon directions read ~0 (occluded short path) — the
    /// caller supplies a ground term there.
    pub fn sample(&self, view_dir: Vec3) -> Vec3 {
        let view = view_dir.normalize_or(self.up);
        let cos_z = view.dot(self.up).clamp(-1.0, 1.0);
        let v = cos_z.acos() / PI;

        // Azimuth relative to the sun, from the horizontal projections.
        let view_h = (view - self.up * cos_z).normalize_or_zero();
        let sun_cz = self.sun_dir.dot(self.up);
        let sun_h = (self.sun_dir - self.up * sun_cz).normalize_or_zero();
        let cos_az = view_h.dot(sun_h).clamp(-1.0, 1.0);
        // `dot` of two zeroed projections is 0 → az = π/2, u = 0.5 (harmless: the
        // field is azimuth-flat at the zenith/nadir where the projection vanishes).
        let u = cos_az.acos() / PI;

        bilinear_clamp(&self.cells, self.width, self.height, u, v)
    }
}

/// Single- + multiple-scattering radiance along one view ray from `p`. CPU twin
/// of the `integrate_atmosphere_multiscatter` loop in `atmosphere.wgsl` (center
/// at the origin). Returns radiance in `sun_flux` units.
#[allow(clippy::too_many_arguments)]
fn sky_ray_radiance(
    p: Vec3,
    view: Vec3,
    sun_dir: Vec3,
    planet_r: f32,
    atmos_top_r: f32,
    atmos_top_alt: f32,
    beta_r: Vec3,
    beta_m: f32,
    h_r: f32,
    h_m: f32,
    g: f32,
    strength: f32,
    multi_gain: f32,
    sun_flux: f32,
    ms: &MultiScatterLut,
) -> Vec3 {
    if strength <= 0.0 {
        return Vec3::ZERO;
    }
    let t_exit = compute_t_exit(p, view, planet_r, atmos_top_r);
    if t_exit <= 1e-3 {
        return Vec3::ZERO;
    }

    let cos_theta = view.dot(sun_dir).clamp(-1.0, 1.0);
    let p_r = (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).max(1e-6);
    let p_m = (1.0 / (4.0 * PI)) * (1.0 - g2) / denom.powf(1.5);

    let n = SKY_VIEW_STEPS;
    let ds = t_exit / n as f32;
    let beta_m_v = Vec3::splat(beta_m);

    let mut sum_r = Vec3::ZERO;
    let mut sum_m = Vec3::ZERO;
    let mut sum_ms = Vec3::ZERO;
    let mut od_r = 0.0f32;
    let mut od_m = 0.0f32;

    for i in 0..n {
        let t = (i as f32 + 0.5) * ds;
        let pt = p + view * t;
        let r_pt = pt.length();
        let h = (r_pt - planet_r).max(0.0);
        let rho_r = (-h / h_r).exp();
        let rho_m = (-h / h_m).exp();
        od_r += rho_r * ds;
        od_m += rho_m * ds;

        let tau_view = beta_r * od_r + beta_m_v * od_m;
        let trans_view = vec3_exp(-tau_view);

        let tau_sun =
            sun_optical_depth(pt, sun_dir, planet_r, atmos_top_r, beta_r, beta_m, h_r, h_m);
        let trans_sun = vec3_exp(-tau_sun);

        let weight = trans_view * trans_sun * ds;
        sum_r += rho_r * weight;
        sum_m += rho_m * weight;

        // Multi-scatter fill: σ_s·ρ·L_ms·T_view (bake already integrated the
        // isotropic phase). Local zenith at the sample gives the sun-zenith cosine.
        let zenith = pt / r_pt.max(1e-3);
        let mu_s = sun_dir.dot(zenith).clamp(-1.0, 1.0);
        let h_norm = (h / atmos_top_alt).clamp(0.0, 1.0);
        let l_ms = ms.sample(mu_s, h_norm);
        let beta_rho = beta_r * rho_r + beta_m_v * rho_m;
        sum_ms += trans_view * beta_rho * l_ms * ds;
    }

    let single = sun_flux * strength * (beta_r * (sum_r * p_r) + beta_m_v * (sum_m * p_m));
    let multi = sun_flux * strength * multi_gain * sum_ms;
    single + multi
}

fn vec3_exp(v: Vec3) -> Vec3 {
    Vec3::new(v.x.exp(), v.y.exp(), v.z.exp())
}

/// The `i`-th of `n` near-uniform points on the unit sphere (Fibonacci
/// spiral) — for the hemispherical-irradiance Monte-Carlo integral.
fn fibonacci_sphere(i: usize, n: usize) -> Vec3 {
    let golden = PI * (3.0 - 5.0_f32.sqrt());
    let y = 1.0 - (i as f32 / (n as f32 - 1.0)) * 2.0;
    let r = (1.0 - y * y).max(0.0).sqrt();
    let theta = golden * i as f32;
    Vec3::new(theta.cos() * r, y, theta.sin() * r)
}

/// Bilinear lookup into a row-major `width × height` `Vec3` grid, clamp-to-edge,
/// at normalized `(u, v) ∈ [0, 1]²` (texel centers at `(i+0.5)/w, (j+0.5)/h`).
fn bilinear_clamp(cells: &[Vec3], width: u32, height: u32, u: f32, v: f32) -> Vec3 {
    if cells.is_empty() {
        return Vec3::ZERO;
    }
    let w = width.max(1);
    let h = height.max(1);
    let fx = (u.clamp(0.0, 1.0) * w as f32 - 0.5).clamp(0.0, (w - 1) as f32);
    let fy = (v.clamp(0.0, 1.0) * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
    let x0 = fx.floor() as u32;
    let y0 = fy.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;
    let at = |x: u32, y: u32| cells[(y * w + x) as usize];
    let top = at(x0, y0).lerp(at(x1, y0), tx);
    let bot = at(x0, y1).lerp(at(x1, y1), tx);
    top.lerp(bot, ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec4;

    /// An Earth-like atmosphere block in meters (`meters_per_render_unit = 1`).
    fn earth_atmos() -> AtmosphereBlock {
        let h_r = 8000.0_f32;
        let h_m = 1200.0_f32;
        AtmosphereBlock {
            // β_R per meter (Bucholtz), H_R in meters.
            rayleigh_beta_h: Vec4::new(5.8e-6, 13.5e-6, 33.1e-6, h_r),
            // β_M per meter, HG asymmetry g.
            mie_beta_g: Vec4::new(21.0e-6, 21.0e-6, 21.0e-6, 0.76),
            // top altitude, H_M, strength, multi-scatter gain.
            atmos_geom: Vec4::new(60_000.0, h_m, 1.0, 1.0),
            ..Default::default()
        }
    }

    const EARTH_R: f32 = 6.371e6;

    fn bake(sun_dir: Vec3, altitude: f32) -> SkyViewLut {
        let atmos = earth_atmos();
        let ms = MultiScatterLut::bake(&atmos, EARTH_R, 32, 32);
        SkyViewLut::bake(
            &atmos,
            EARTH_R,
            altitude,
            sun_dir,
            Vec3::Y,
            10.0,
            &ms,
            48,
            64,
        )
    }

    #[test]
    fn daytime_zenith_is_blue_and_positive() {
        // Sun high overhead; look straight up.
        let lut = bake(Vec3::new(0.15, 1.0, 0.0).normalize(), 0.0);
        let zenith = lut.sample(Vec3::Y);
        assert!(
            zenith.x.is_finite() && zenith.y.is_finite() && zenith.z.is_finite(),
            "sky radiance must be finite, got {zenith:?}"
        );
        assert!(zenith.length() > 0.0, "daytime zenith sky must be lit");
        // Rayleigh + multi-scatter → blue dominates red at the zenith.
        assert!(
            zenith.z > zenith.x,
            "zenith sky should be blue-dominant, got {zenith:?}"
        );
    }

    #[test]
    fn night_is_far_dimmer_than_day() {
        let day = bake(Vec3::new(0.15, 1.0, 0.0).normalize(), 0.0)
            .sample(Vec3::Y)
            .length();
        // Sun well below the horizon → view zenith sees no direct sun column.
        let night = bake(Vec3::new(0.3, -1.0, 0.0).normalize(), 0.0)
            .sample(Vec3::Y)
            .length();
        assert!(
            night < 0.1 * day,
            "night zenith ({night}) should be ≪ day zenith ({day})"
        );
    }

    #[test]
    fn ambient_irradiance_is_positive_by_day_and_dark_at_night() {
        let day = bake(Vec3::new(0.15, 1.0, 0.0).normalize(), 0.0).ambient_sky_irradiance();
        assert!(
            day.x.is_finite() && day.length() > 0.0,
            "daytime sky irradiance must be positive, got {day:?}"
        );
        // Blue-dominant (Rayleigh sky fill).
        assert!(
            day.z > day.x,
            "sky irradiance should be blue-dominant: {day:?}"
        );
        let night = bake(Vec3::new(0.3, -1.0, 0.0).normalize(), 0.0)
            .ambient_sky_irradiance()
            .length();
        assert!(
            night < 0.1 * day.length(),
            "night irradiance ({night}) should be ≪ day ({})",
            day.length()
        );
    }

    #[test]
    fn vacuum_strength_zero_is_black() {
        let mut atmos = earth_atmos();
        atmos.atmos_geom.z = 0.0; // strength off
        let ms = MultiScatterLut::bake(&atmos, EARTH_R, 32, 32);
        let lut = SkyViewLut::bake(&atmos, EARTH_R, 0.0, Vec3::Y, Vec3::Y, 10.0, &ms, 48, 64);
        assert_eq!(lut.sample(Vec3::Y), Vec3::ZERO);
    }
}
