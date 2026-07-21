use crate::coords::UnitVector3;
use crate::spectrum::Spectrum;

use super::{AngularProfile, Source};

/// A single point source.
///
/// Stars are stored as blackbodies plus an apparent magnitude. The
/// magnitude gives the renderer an absolute brightness scale that
/// survives the choice of passband — we just multiply the
/// band-integrated spectrum by the magnitude-derived flux factor.
#[derive(Debug, Clone)]
pub struct Star {
    pub position: UnitVector3,
    pub spectrum: Spectrum,
    /// Apparent visual magnitude. Lower = brighter; 0 ≈ Vega.
    pub apparent_magnitude: f32,
    /// Distance from the observer in parsecs. Used by the telescope
    /// imager to produce parallax between exposures taken at different
    /// points along the player's orbit.
    pub distance_pc: f32,
    /// Proper motion, radians per year (projected on sky). Zero for
    /// now — reserved for the variable/dynamic layer.
    pub proper_motion: [f32; 2],
}

impl Star {
    /// Linear flux factor derived from apparent magnitude, relative
    /// to a reference magnitude of zero.
    ///
    /// `f = 10^(-0.4 · m)`
    pub fn magnitude_flux(&self) -> f32 {
        10f32.powf(-0.4 * self.apparent_magnitude)
    }

    /// Linear-sRGB chromaticity for this star's temperature.
    ///
    /// Uses an anchored lookup of values adapted from Mitchell Charity's
    /// public-domain B-V → sRGB table
    /// (<https://vendian.org/mncharity/dir3/starcolor/>), converted to
    /// linear space. Stars outside the tabulated range clamp to the
    /// nearest entry. The returned colour is unit-scaled (max channel
    /// = 1.0); apply the magnitude flux separately.
    ///
    /// Only defined for blackbody spectra. Non-blackbody sources get a
    /// neutral white fall-back — a future extended layer (emission-line
    /// nebulae, powerlaw quasars) should carry its own colour.
    pub fn linear_srgb(&self) -> [f32; 3] {
        match &self.spectrum {
            Spectrum::Blackbody { temperature_k, .. } => temperature_to_linear_srgb(*temperature_k),
            _ => [1.0, 1.0, 1.0],
        }
    }
}

/// Shared re-export for other source types (e.g. galaxies) that want
/// to colour their emission from a blackbody-equivalent temperature.
pub(crate) fn temperature_to_linear_srgb_public(t: f32) -> [f32; 3] {
    temperature_to_linear_srgb(t)
}

/// Anchored interpolation from effective temperature to linear sRGB.
///
/// Anchors (approximate, from Charity's table + sRGB gamma):
///   40,000 K → bluish-white
///   10,000 K → near-white blue
///    7,500 K → white
///    6,000 K → warm white (~Sun)
///    4,500 K → yellow-orange
///    3,000 K → deep orange-red
fn temperature_to_linear_srgb(t: f32) -> [f32; 3] {
    // Table of (t, r_linear, g_linear, b_linear).
    // Values derived offline by un-gamma'ing Charity's sRGB anchors and
    // normalising so the brightest channel equals 1.
    const TABLE: &[(f32, [f32; 3])] = &[
        (3_000.0, [1.000, 0.404, 0.135]),
        (4_000.0, [1.000, 0.608, 0.344]),
        (5_000.0, [1.000, 0.787, 0.579]),
        (5_778.0, [1.000, 0.891, 0.775]),
        (6_500.0, [1.000, 0.953, 0.912]),
        (7_500.0, [0.961, 0.961, 1.000]),
        (9_000.0, [0.832, 0.893, 1.000]),
        (12_000.0, [0.670, 0.791, 1.000]),
        (20_000.0, [0.560, 0.720, 1.000]),
        (40_000.0, [0.490, 0.680, 1.000]),
    ];
    if t <= TABLE[0].0 {
        return TABLE[0].1;
    }
    let last = TABLE.last().unwrap();
    if t >= last.0 {
        return last.1;
    }
    for pair in TABLE.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t >= t0 && t <= t1 {
            let u = (t - t0) / (t1 - t0);
            return [
                c0[0] + (c1[0] - c0[0]) * u,
                c0[1] + (c1[1] - c0[1]) * u,
                c0[2] + (c1[2] - c0[2]) * u,
            ];
        }
    }
    [1.0, 1.0, 1.0]
}

impl Source for Star {
    fn position(&self) -> UnitVector3 {
        self.position
    }
    fn spectrum(&self) -> &Spectrum {
        &self.spectrum
    }
    fn angular_profile(&self) -> AngularProfile {
        AngularProfile::Point
    }
}
