use crate::coords::UnitVector3;
use crate::spectrum::Spectrum;

use super::{AngularProfile, Source};

/// Extended source with a Sérsic profile.
///
/// Each galaxy is a point on the celestial sphere plus an elliptical
/// 2-D profile in the tangent plane. The renderer splats a quad
/// oriented by `position_angle_rad`, scaled by `effective_radius_rad`
/// along the major axis and `effective_radius_rad · axis_ratio`
/// along the minor axis, and evaluates `exp(-b_n · (r^(1/n) - 1))`
/// per fragment — Sérsic with concentration index `n`. Spiral-like
/// galaxies use `n ≈ 1` (exponential disk), ellipticals use `n ≈ 4`
/// (de Vaucouleurs).
#[derive(Debug, Clone)]
pub struct Galaxy {
    pub position: UnitVector3,
    pub spectrum: Spectrum,
    pub apparent_magnitude: f32,
    pub effective_radius_rad: f32,
    pub sersic_n: f32,
    pub axis_ratio: f32,
    pub position_angle_rad: f32,
    pub redshift: f32,
}

impl Galaxy {
    /// Linear flux factor, same convention as [`crate::Star`].
    pub fn magnitude_flux(&self) -> f32 {
        10f32.powf(-0.4 * self.apparent_magnitude)
    }

    /// Linear sRGB colour derived from the underlying spectrum. For
    /// blackbody approximations we reuse the star temperature table;
    /// for other spectra we fall back to a neutral warm-white.
    pub fn linear_srgb(&self) -> [f32; 3] {
        match &self.spectrum {
            Spectrum::Blackbody { temperature_k, .. } => {
                crate::sources::star::temperature_to_linear_srgb_public(*temperature_k)
            }
            _ => [1.0, 0.94, 0.86],
        }
    }
}

impl Source for Galaxy {
    fn position(&self) -> UnitVector3 {
        self.position
    }
    fn spectrum(&self) -> &Spectrum {
        &self.spectrum
    }
    fn angular_profile(&self) -> AngularProfile {
        AngularProfile::Sersic {
            effective_radius_rad: self.effective_radius_rad,
            sersic_n: self.sersic_n,
            axis_ratio: self.axis_ratio,
            position_angle_rad: self.position_angle_rad,
        }
    }
}
