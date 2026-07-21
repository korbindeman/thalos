//! Authored physical and visual state for an ocean-bearing body.
//!
//! This is world data, not a render-material preset: simulation and rendering
//! project the same per-body state into whatever representation they need.

use serde::{Deserialize, Serialize};

/// The stable, slowly varying state of one planetary ocean.
///
/// Wave phases are deliberately absent. They are derived from the canonical
/// simulation epoch so pausing, time warp, and deterministic captures all see
/// one clock. Directions are body-local axes that consumers project onto the
/// local sea tangent plane.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OceanState {
    /// Dominant wind direction as a body-local axis.
    pub wind_axis_body: (f32, f32, f32),
    /// Wind speed at the standard 10 m reference height.
    pub wind_speed_10m_m_s: f32,
    /// Significant wave height used to scale the resolved slope spectrum.
    pub significant_wave_height_m: f32,
    /// Dominant open-water wavelength used to distribute energy between the
    /// long and medium spectral bands.
    pub dominant_wavelength_m: f32,
    /// Independent swell direction as a body-local axis.
    pub swell_axis_body: (f32, f32, f32),
    /// Fraction of long-wave energy carried by the independent swell, 0..1.
    pub swell_energy: f32,
    /// Deep-water linear RGB absorption/scatter tint.
    pub deep_water_color: (f32, f32, f32),
    /// Water-column depth at which the deep-water tint is effectively reached.
    pub optical_depth_m: f32,
    /// Resolved slope magnitude where open-water whitecaps begin.
    pub foam_slope_onset: f32,
}

impl OceanState {
    /// Calm/moderate reference sea used by Thalos and as a safe compatibility
    /// fallback for programmatically constructed ocean bodies.
    pub const MODERATE: Self = Self {
        wind_axis_body: (0.21, 0.93, 0.31),
        wind_speed_10m_m_s: 11.0,
        significant_wave_height_m: 2.4,
        dominant_wavelength_m: 150.0,
        swell_axis_body: (-0.37, 0.78, 0.50),
        swell_energy: 0.34,
        deep_water_color: (0.012, 0.040, 0.090),
        optical_depth_m: 120.0,
        foam_slope_onset: 0.22,
    };

    /// Validate authored values before they become shared world state.
    pub fn validate(self) -> Result<(), &'static str> {
        let finite_positive = |value: f32| value.is_finite() && value > 0.0;
        let axis_valid = |axis: (f32, f32, f32)| {
            axis.0.is_finite()
                && axis.1.is_finite()
                && axis.2.is_finite()
                && axis.0 * axis.0 + axis.1 * axis.1 + axis.2 * axis.2 > 1.0e-6
        };
        if !axis_valid(self.wind_axis_body) || !axis_valid(self.swell_axis_body) {
            return Err("wind and swell axes must be finite and non-zero");
        }
        if !finite_positive(self.wind_speed_10m_m_s)
            || !finite_positive(self.significant_wave_height_m)
            || !finite_positive(self.dominant_wavelength_m)
            || !finite_positive(self.optical_depth_m)
        {
            return Err("wind speed, wave height/wavelength, and optical depth must be positive");
        }
        if !self.swell_energy.is_finite() || !(0.0..=1.0).contains(&self.swell_energy) {
            return Err("swell energy must be finite and in 0..=1");
        }
        if !self.foam_slope_onset.is_finite() || !(0.0..=1.0).contains(&self.foam_slope_onset) {
            return Err("foam slope onset must be finite and in 0..=1");
        }
        let (r, g, b) = self.deep_water_color;
        if !r.is_finite() || !g.is_finite() || !b.is_finite() || r < 0.0 || g < 0.0 || b < 0.0 {
            return Err("deep-water colour must be finite and non-negative");
        }
        Ok(())
    }
}

impl Default for OceanState {
    fn default() -> Self {
        Self::MODERATE
    }
}
