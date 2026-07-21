use crate::coords::UnitVector3;
use crate::spectrum::Spectrum;

use super::{AngularProfile, Source};

/// Bounded volumetric emission field.
///
/// Placeholder for phase 4. The center is represented as the
/// `position` unit vector (direction on the celestial sphere); the
/// `angular_extent_rad` bounds how far off-axis a ray may be and still
/// contribute. Rendering will sample `noise_params` with a source-side
/// evaluator once that module exists.
#[derive(Debug, Clone)]
pub struct NebulaField {
    pub position: UnitVector3,
    pub spectrum: Spectrum,
    pub angular_extent_rad: f32,
    pub brightness: f32,
    pub noise_seed: u64,
}

impl Source for NebulaField {
    fn position(&self) -> UnitVector3 {
        self.position
    }
    fn spectrum(&self) -> &Spectrum {
        &self.spectrum
    }
    fn angular_profile(&self) -> AngularProfile {
        AngularProfile::Volumetric
    }
}
