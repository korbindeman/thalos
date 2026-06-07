//! Per-body atmosphere uniform.
//!
//! `AtmosphereBlock` carries the single-scattering Rayleigh + Mie set,
//! plus cloud-band and limb-darkening parameters, shared by every
//! planet-surface material's atmosphere pass. Authored quantities are
//! converted from meters into render units at the
//! `from_terrestrial` boundary, so the GPU side never has to divide by
//! the scale factor.
//!
//! The WGSL mirror lives at `shaders/atmosphere.wgsl` and is registered
//! by [`crate::PlanetLightingPlugin`]. Field order, widths, and padding
//! MUST match across both sides.

use bevy::math::Vec4;
use bevy::render::render_resource::ShaderType;
use thalos_world::TerrestrialAtmosphere;

/// Atmosphere uniform consumed by every body-surface material that
/// integrates atmospheric scattering, cloud bands, or limb darkening.
///
/// `AtmosphereBlock::default()` produces a vacuum: every scalar that
/// gates a layer (`scattering_strength`, `cloud_coverage`,
/// `limb_strength`) is zero, so the shader early-outs.
#[derive(Clone, Copy, ShaderType)]
pub struct AtmosphereBlock {
    /// Rayleigh sea-level scattering coefficient β_R, per render unit.
    /// xyz = R/G/B; w = Rayleigh scale height H_R in render units.
    /// Computed from authored τ_v and H via β = τ_v / H, then converted
    /// from per-meter to per-render-unit. β_R = 0 disables Rayleigh.
    pub rayleigh_beta_h: Vec4,
    /// Mie scattering parameters.
    /// xyz = β_M at sea level (per render unit, R = G = B for spectrally
    /// neutral aerosols; authoring may colour it for specific dust),
    /// w = Henyey-Greenstein asymmetry g in [-1, 1].
    pub mie_beta_g: Vec4,
    /// Atmosphere geometry + global gates.
    /// x = atmosphere top altitude above the surface (render units —
    ///     view raymarch terminates here),
    /// y = Mie scale height H_M (render units),
    /// z = strength multiplier (artistic; 0 disables the entire
    ///     scattering raymarch and the surface renders as if in vacuum),
    /// w = reserved for ozone band altitude (M4 follow-up).
    pub atmos_geom: Vec4,
    /// xyz = per-channel Minnaert exponents (R, G, B), w = strength.
    /// Pure artistic limb darkening on the lit surface; independent
    /// of the scattering model.
    pub limb_exponents: Vec4,
    /// xyz = sunlit-cloud albedo, w = coverage fraction in [0, 1].
    pub cloud_albedo_coverage: Vec4,
    /// Cloud layer shape.
    /// x = layer base altitude above the surface (render units),
    /// y = layer thickness (render units) — the volumetric slab depth,
    /// z = optical-density multiplier (scales raymarch extinction),
    /// w = differential-rotation coefficient.
    pub cloud_shape: Vec4,
    /// x = equatorial scroll rate (rad/s), y = sim time seconds
    /// (wrapped; written per frame by `update_planet_light_dirs`),
    /// zw = reserved for future cloud dynamics controls.
    pub cloud_dynamics: Vec4,
    /// Cloud main-deck band phases 0..=3. 16 total phases packed into
    /// four `Vec4`s carry the per-latitude-strip rotation state for
    /// the banded cloud decomposition. See `CLOUD_BAND_COUNT` /
    /// `CloudBandEnvironmentState` on the CPU side and
    /// `sample_cloud_banded` in `planet_impostor.wgsl` for usage.
    pub cloud_bands_a: Vec4,
    /// Cloud main-deck band phases 4..=7.
    pub cloud_bands_b: Vec4,
    /// Cloud main-deck band phases 8..=11.
    pub cloud_bands_c: Vec4,
    /// Cloud main-deck band phases 12..=15.
    pub cloud_bands_d: Vec4,
}

impl Default for AtmosphereBlock {
    fn default() -> Self {
        Self {
            rayleigh_beta_h: Vec4::ZERO,
            mie_beta_g: Vec4::ZERO,
            atmos_geom: Vec4::ZERO,
            limb_exponents: Vec4::ZERO,
            cloud_albedo_coverage: Vec4::ZERO,
            cloud_shape: Vec4::ZERO,
            cloud_dynamics: Vec4::ZERO,
            cloud_bands_a: Vec4::ZERO,
            cloud_bands_b: Vec4::ZERO,
            cloud_bands_c: Vec4::ZERO,
            cloud_bands_d: Vec4::ZERO,
        }
    }
}

/// Number of latitudinal cloud rotation bands. Each band has its own
/// rigid rotation speed `ω_i = scroll_rate × (1 − diff × sin²(lat_i))`
/// where `sin²(lat_i) = i / (CLOUD_BAND_COUNT − 1)`. Per-band phases are
/// accumulated on the CPU, mod `TAU` in f64, uploaded as four `Vec4`s
/// into `AtmosphereBlock`, and consumed by `sample_cloud_banded` in
/// `planet_impostor.wgsl`. Because each per-band phase wraps
/// independently mod TAU, there is no latitude at which rotation seams —
/// rotation is seamless forever. State persists trivially as 16 × f64
/// per body.
pub const CLOUD_BAND_COUNT: usize = 16;

impl AtmosphereBlock {
    /// Build from a `TerrestrialAtmosphere` and the body's
    /// meters-per-render-unit ratio. Any layer not present in the
    /// source struct is left at zero, which the shader interprets as
    /// "skip this layer entirely." The `cloud_dynamics.y` (sim time)
    /// field is left at zero here; per-frame writers populate it.
    ///
    /// Authored quantities are converted from meters into render units
    /// at this boundary, so the GPU side never has to divide by the
    /// scale factor. The Rayleigh / Mie β coefficients are derived
    /// from the authored vertical optical depth and scale height
    /// (β = τ_v / H, in 1/m), then scaled into 1/render-unit.
    pub fn from_terrestrial(atmos: &TerrestrialAtmosphere, meters_per_render_unit: f32) -> Self {
        let mut out = Self::default();
        let inv_m = 1.0 / meters_per_render_unit.max(1.0);

        if let Some(ld) = &atmos.limb_darkening {
            out.limb_exponents = Vec4::new(
                ld.red.max(0.0),
                ld.green.max(0.0),
                ld.blue.max(0.0),
                ld.strength.clamp(0.0, 1.0),
            );
        }

        if let Some(sc) = &atmos.scattering {
            let h_r_m = sc.rayleigh_scale_height_m.max(1.0);
            let h_m_m = sc.mie_scale_height_m.max(1.0);
            let beta_r_per_m = [
                sc.vertical_optical_depth[0].max(0.0) / h_r_m,
                sc.vertical_optical_depth[1].max(0.0) / h_r_m,
                sc.vertical_optical_depth[2].max(0.0) / h_r_m,
            ];
            let beta_m_per_m = sc.mie_optical_depth.max(0.0) / h_m_m;
            let m_per_unit = meters_per_render_unit.max(1.0);
            out.rayleigh_beta_h = Vec4::new(
                beta_r_per_m[0] * m_per_unit,
                beta_r_per_m[1] * m_per_unit,
                beta_r_per_m[2] * m_per_unit,
                h_r_m * inv_m,
            );
            out.mie_beta_g = Vec4::new(
                beta_m_per_m * m_per_unit,
                beta_m_per_m * m_per_unit,
                beta_m_per_m * m_per_unit,
                sc.mie_asymmetry.clamp(-0.999, 0.999),
            );
            out.atmos_geom = Vec4::new(
                atmos.karman_line_m.max(0.0) * inv_m,
                h_m_m * inv_m,
                sc.strength.max(0.0),
                0.0,
            );
        }

        if let Some(clouds) = &atmos.clouds {
            out.cloud_albedo_coverage = Vec4::new(
                clouds.albedo[0],
                clouds.albedo[1],
                clouds.albedo[2],
                clouds.coverage.clamp(0.0, 1.0),
            );
            // Altitudes are authored in meters; convert to render units at
            // this boundary so the volumetric raymarch in `body_sky.wgsl`
            // never has to divide by the scale factor. `density` is unitless.
            out.cloud_shape = Vec4::new(
                clouds.base_altitude_m.max(0.0) * inv_m,
                clouds.thickness_m.max(0.0) * inv_m,
                clouds.density.max(0.0),
                clouds.differential_rotation,
            );
            out.cloud_dynamics = Vec4::new(clouds.scroll_rate, 0.0, 0.0, 0.0);
        }
        out
    }
}
