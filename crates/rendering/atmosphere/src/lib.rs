//! Authored atmosphere projections shared by Thalos rendering applications.
//!
//! The authored [`thalos_world::TerrestrialAtmosphere`] is projected into the
//! compact [`AtmosphereBlock`] consumed by Thalos's planetary shaders. A flat,
//! metre-scale application may instead use [`add_bevy_earth_atmosphere`] to
//! install Bevy's maintained Earth atmosphere without importing the Thalos
//! runtime or its body compositor.
//!
//! These are concrete adapters, not interchangeable renderer backends. The
//! planetary adapter owns floating-origin, multi-body, and scene-depth
//! integration; Bevy owns the local tangent-world sky and its environment map.

use bevy::{
    light::{Atmosphere, atmosphere::ScatteringMedium},
    math::{Vec3, Vec4},
    prelude::{Assets, Handle},
    render::render_resource::ShaderType,
};
use thalos_world::TerrestrialAtmosphere;

/// The Earth radius used by Bevy's atmosphere preset.
pub const BEVY_EARTH_RADIUS_M: f32 = 6_360_000.0;
/// The outer radius used by Bevy's atmosphere preset.
pub const BEVY_EARTH_ATMOSPHERE_RADIUS_M: f32 = 6_460_000.0;

/// Install Bevy's Earth atmosphere medium and return its render adapter.
///
/// `density_multiplier` is an adapter calibration. Kòrsou uses `0.45` because
/// its flat terrain never curves out of the dense lower atmosphere along long
/// ground sightlines. It is deliberately not written into shared authored
/// atmosphere state.
pub fn add_bevy_earth_atmosphere(
    scattering_media: &mut Assets<ScatteringMedium>,
    density_multiplier: f32,
) -> Atmosphere {
    let medium = scattering_media.add(
        ScatteringMedium::earth(256, 256).with_density_multiplier(density_multiplier.max(0.0)),
    );
    Atmosphere::earth(medium)
}

/// Packed terrestrial-atmosphere state consumed by Thalos's planetary shaders.
///
/// `Default` is vacuum: every scalar that gates a layer is zero. Authored
/// quantities are converted from metres into render units by
/// [`from_terrestrial`](Self::from_terrestrial).
#[derive(Clone, Copy, ShaderType)]
pub struct AtmosphereBlock {
    /// Rayleigh sea-level scattering coefficient and scale height.
    /// xyz = beta per render unit; w = scale height in render units.
    pub rayleigh_beta_h: Vec4,
    /// Mie sea-level scattering coefficient and HG asymmetry.
    /// xyz = beta per render unit; w = g.
    pub mie_beta_g: Vec4,
    /// x = atmosphere top, y = Mie scale height, z = strength,
    /// w = multi-scatter gain.
    pub atmos_geom: Vec4,
    /// xyz = per-channel Minnaert exponents; w = strength.
    pub limb_exponents: Vec4,
    /// xyz = cloud albedo; w = coverage.
    pub cloud_albedo_coverage: Vec4,
    /// x = cloud base, y = thickness, z = density,
    /// w = differential rotation.
    pub cloud_shape: Vec4,
    /// x = equatorial scroll rate, y = simulation time.
    pub cloud_dynamics: Vec4,
    /// Cloud main-deck rotation phases 0..=3.
    pub cloud_bands_a: Vec4,
    /// Cloud main-deck rotation phases 4..=7.
    pub cloud_bands_b: Vec4,
    /// Cloud main-deck rotation phases 8..=11.
    pub cloud_bands_c: Vec4,
    /// Cloud main-deck rotation phases 12..=15.
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

/// Band count for the authored banded-atmosphere model.
pub use thalos_world::CLOUD_BAND_COUNT;

impl AtmosphereBlock {
    /// Project authored state into the planetary GPU representation.
    ///
    /// `meters_per_render_unit` must match the geometry adapter. The simulation
    /// time field remains zero; the application's per-frame projection writes
    /// it alongside other dynamic atmosphere state.
    pub fn from_terrestrial(
        atmosphere: &TerrestrialAtmosphere,
        meters_per_render_unit: f32,
    ) -> Self {
        let mut out = Self::default();
        let meters_per_unit = meters_per_render_unit.max(1.0);
        let units_per_meter = 1.0 / meters_per_unit;

        if let Some(limb) = &atmosphere.limb_darkening {
            out.limb_exponents = Vec4::new(
                limb.red.max(0.0),
                limb.green.max(0.0),
                limb.blue.max(0.0),
                limb.strength.clamp(0.0, 1.0),
            );
        }

        if let Some(scattering) = &atmosphere.scattering {
            let rayleigh_height_m = scattering.rayleigh_scale_height_m.max(1.0);
            let mie_height_m = scattering.mie_scale_height_m.max(1.0);
            let rayleigh_beta_per_m = [
                scattering.vertical_optical_depth[0].max(0.0) / rayleigh_height_m,
                scattering.vertical_optical_depth[1].max(0.0) / rayleigh_height_m,
                scattering.vertical_optical_depth[2].max(0.0) / rayleigh_height_m,
            ];
            let mie_beta_per_m = scattering.mie_optical_depth.max(0.0) / mie_height_m;
            out.rayleigh_beta_h = Vec4::new(
                rayleigh_beta_per_m[0] * meters_per_unit,
                rayleigh_beta_per_m[1] * meters_per_unit,
                rayleigh_beta_per_m[2] * meters_per_unit,
                rayleigh_height_m * units_per_meter,
            );
            out.mie_beta_g = Vec4::new(
                mie_beta_per_m * meters_per_unit,
                mie_beta_per_m * meters_per_unit,
                mie_beta_per_m * meters_per_unit,
                scattering.mie_asymmetry.clamp(-0.999, 0.999),
            );
            out.atmos_geom = Vec4::new(
                atmosphere.karman_line_m.max(0.0) * units_per_meter,
                mie_height_m * units_per_meter,
                scattering.strength.max(0.0),
                scattering.multi_scatter_gain.max(0.0),
            );
        }

        if let Some(clouds) = &atmosphere.clouds {
            out.cloud_albedo_coverage = Vec4::new(
                clouds.albedo[0],
                clouds.albedo[1],
                clouds.albedo[2],
                clouds.coverage.clamp(0.0, 1.0),
            );
            out.cloud_shape = Vec4::new(
                clouds.base_altitude_m.max(0.0) * units_per_meter,
                clouds.thickness_m.max(0.0) * units_per_meter,
                clouds.density.max(0.0),
                clouds.differential_rotation,
            );
            out.cloud_dynamics = Vec4::new(clouds.scroll_rate, 0.0, 0.0, 0.0);
        }
        out
    }
}

/// Bevy's atmosphere stores its medium separately from its geometry. This is
/// useful to callers that need to inspect or replace the medium after install.
pub fn bevy_atmosphere_medium(atmosphere: &Atmosphere) -> &Handle<ScatteringMedium> {
    &atmosphere.medium
}

/// Return the ground albedo of Bevy's Earth adapter for documentation/tests.
pub fn bevy_earth_ground_albedo() -> Vec3 {
    Vec3::splat(0.3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bevy_earth_adapter_preserves_expected_geometry() {
        let mut media = Assets::<ScatteringMedium>::default();
        let atmosphere = add_bevy_earth_atmosphere(&mut media, 0.45);
        assert_eq!(atmosphere.inner_radius, BEVY_EARTH_RADIUS_M);
        assert_eq!(atmosphere.outer_radius, BEVY_EARTH_ATMOSPHERE_RADIUS_M);
        assert_eq!(atmosphere.ground_albedo, bevy_earth_ground_albedo());
        assert!(media.get(bevy_atmosphere_medium(&atmosphere)).is_some());
    }

    #[test]
    fn default_gpu_projection_is_vacuum() {
        let block = AtmosphereBlock::default();
        assert_eq!(block.atmos_geom, Vec4::ZERO);
        assert_eq!(block.rayleigh_beta_h, Vec4::ZERO);
        assert_eq!(block.cloud_albedo_coverage, Vec4::ZERO);
    }
}
