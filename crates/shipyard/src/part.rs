use crate::resource::Resource;
use bevy::prelude::*;

/// Marker — entity is a ship part.
#[derive(Component, Debug, Clone, Copy)]
#[require(Transform, Visibility)]
pub struct Part;

#[derive(Component, Debug, Clone)]
pub struct CommandPod {
    pub model: String,
    pub diameter: f32,
    pub dry_mass: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Decoupler {
    pub ejection_impulse: f32,
    pub dry_mass: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Adapter {
    pub target_diameter: f32,
    pub dry_mass: f32,
}

/// Pure geometry — contents live in [`crate::PartResources`]. A tank can
/// hold any resource; this part does not restrict which.
#[derive(Component, Debug, Clone)]
pub struct FuelTank {
    pub length: f32,
    pub dry_mass: f32,
}

/// A mass fraction of a single reactant relative to the engine's total
/// mass flow. For a methalox engine at O/F = 3.6:
/// `[(Methane, 0.217), (Lox, 0.783)]`.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReactantRatio {
    pub resource: Resource,
    pub mass_fraction: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Engine {
    pub model: String,
    pub diameter: f32,
    /// Thrust in vacuum, N.
    pub thrust: f32,
    /// Specific impulse in vacuum, s.
    pub isp: f32,
    pub dry_mass: f32,
    /// Which reactants this engine consumes, as mass fractions of its
    /// total mass flow. Fractions must sum to 1.0; resources must all be
    /// mass-bearing. Non-mass-bearing resources (like electricity) belong
    /// in `power_draw_kw`, not here.
    pub reactants: Vec<ReactantRatio>,
    /// Continuous electrical draw while the engine is firing, kW.
    /// 0 for chemical engines.
    pub power_draw_kw: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineValidationError {
    /// Reactant mass fractions don't sum to 1.0 within tolerance.
    ReactantFractionsNotNormalized,
    /// A reactant references a non-mass-bearing resource (e.g. Electricity).
    ReactantNotMassBearing,
    /// Reactants list is empty — every engine must have at least one.
    NoReactants,
    /// A fraction is zero or negative.
    NonPositiveFraction,
}

impl Engine {
    /// Check invariants that the stats aggregator relies on. Call from
    /// tests, editors, or loaders; stats computation assumes these hold
    /// and will produce meaningless numbers otherwise.
    pub fn validate(&self) -> Result<(), EngineValidationError> {
        if self.reactants.is_empty() {
            return Err(EngineValidationError::NoReactants);
        }
        let mut sum = 0.0_f32;
        for r in &self.reactants {
            if r.mass_fraction <= 0.0 {
                return Err(EngineValidationError::NonPositiveFraction);
            }
            if !r.resource.is_mass_bearing() {
                return Err(EngineValidationError::ReactantNotMassBearing);
            }
            sum += r.mass_fraction;
        }
        if (sum - 1.0).abs() > 1e-4 {
            return Err(EngineValidationError::ReactantFractionsNotNormalized);
        }
        Ok(())
    }
}
