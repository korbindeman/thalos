//! Source types for the celestial sphere.
//!
//! Each source carries physical properties (position + spectrum +
//! shape). Renderers project the spectrum through a passband to get
//! per-channel flux; they never see pre-computed RGB.

use crate::coords::UnitVector3;
use crate::spectrum::{Passband, Spectrum};

mod star;
mod galaxy;
mod nebula;

pub use galaxy::Galaxy;
pub use nebula::NebulaField;
pub use star::Star;

/// Angular profile of a source on the celestial sphere.
#[derive(Debug, Clone)]
pub enum AngularProfile {
    /// Unresolved point — renderer splats a PSF.
    Point,
    /// Extended elliptical profile (galaxies). `effective_radius` is
    /// in radians; `sersic_n` controls concentration; `axis_ratio`
    /// ∈ (0, 1] is minor/major.
    Sersic {
        effective_radius_rad: f32,
        sersic_n: f32,
        axis_ratio: f32,
        position_angle_rad: f32,
    },
    /// Volumetric field (nebulae). Renderer uses the source-specific
    /// sampling function — this variant just signals "not a point".
    Volumetric,
}

/// Shared read-only interface over any source type.
///
/// The typed-layer design in [`crate::Universe`] means renderers
/// normally iterate concrete `Vec<Star>` etc. directly for speed; this
/// trait exists for code that wants a uniform view (tests, dumps,
/// future generic renderers).
pub trait Source {
    fn position(&self) -> UnitVector3;
    fn spectrum(&self) -> &Spectrum;
    fn flux_in_band(&self, band: &Passband) -> f32 {
        band.integrate(self.spectrum())
    }
    fn angular_profile(&self) -> AngularProfile;
}
