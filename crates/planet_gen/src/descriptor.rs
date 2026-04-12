//! Authored physical parameters for a celestial body (spec v0.1).
//!
//! All units SI: metres, kilograms, seconds, Kelvin, radians.  Every field
//! here is a *physical input*; anything that can be computed (gravity, heat
//! budgets, transition diameters, atmospheric retention) lives in
//! [`crate::derived`] and is derived once, up-front, before any generation
//! stage runs.  Stages read derived properties — never raw inputs — which is
//! what lets future body types (atmospheric planets, ice worlds, gas giants)
//! be added by extending derivations and adding stages, not by rewriting
//! existing logic.

/// Bulk composition as mass fractions.  v0.1 stages meaningfully consume only
/// `silicate`, `iron`, and `ice`, but all fields exist so the data model stays
/// stable as later stages are added.  Fractions must sum to 1.0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Composition {
    pub silicate: f64,
    pub iron: f64,
    pub ice: f64,
    pub volatiles: f64,
    pub hydrogen_helium: f64,
}

impl Composition {
    pub const SUM_TOLERANCE: f64 = 1e-6;

    pub fn new(
        silicate: f64,
        iron: f64,
        ice: f64,
        volatiles: f64,
        hydrogen_helium: f64,
    ) -> Self {
        let total = silicate + iron + ice + volatiles + hydrogen_helium;
        assert!(
            (total - 1.0).abs() < Self::SUM_TOLERANCE,
            "composition mass fractions must sum to 1.0, got {total}"
        );
        Self { silicate, iron, ice, volatiles, hydrogen_helium }
    }
}

/// Dynamical parent body.  Only the fields actually consumed by v0.1
/// derivations (tidal heating) live here; more can be added additively.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParentBody {
    pub mass_kg: f64,
    /// Semi-major axis of *this* body's orbit around its parent, in metres.
    pub orbit_semi_major_axis_m: f64,
}

/// Art-direction escape hatch.  When a field is `Some`, the corresponding
/// derivation is bypassed and this value is authoritative.  Only fields whose
/// overrides the v0.1 pipeline actually reads are present.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Overrides {
    pub surface_gravity: Option<f64>,
    pub equilibrium_temperature: Option<f64>,
    pub total_heat_budget: Option<f64>,
}

/// Complete authored description of a celestial body's surface.
///
/// Deterministic: same descriptor + same sphere samples → identical output,
/// always.  This is the serializable artifact the editor exports and the
/// generator consumes.
#[derive(Clone, Debug)]
pub struct PlanetDescriptor {
    pub name: String,

    /// Physical radius, metres.
    pub radius_m: f64,
    /// Mass, kilograms.
    pub mass_kg: f64,
    /// Semi-major axis relative to the luminous primary (Sun, for bodies in
    /// the solar system), metres.  Drives equilibrium temperature.
    pub semi_major_axis_m: f64,
    /// Bond albedo, 0..1.
    pub bond_albedo: f64,
    /// Bulk composition mass fractions.
    pub composition: Composition,
    /// Age in gigayears.  Drives crater accumulation and radiogenic decay.
    pub age_gyr: f64,
    /// Sidereal rotation period, seconds.
    pub rotation_period_s: f64,
    /// Axial tilt relative to the orbital plane, radians.
    pub axial_tilt_rad: f64,
    /// Orbital eccentricity around the dynamical parent, 0..1.
    pub eccentricity: f64,
    /// Dynamical parent, if any.  Required for tidal heating; future stages
    /// may additionally use it for impact-flux modulation.
    pub parent: Option<ParentBody>,
    /// Impact flux multiplier; 1.0 = lunar baseline.
    pub impact_flux_multiplier: f64,
    /// Thermal history scalar, 0..1 — "how much internal activity did this
    /// body experience before freezing out."  Drives mare flooding in v0.1.
    pub thermal_history: f64,
    /// Master seed.  Stages derive sub-seeds as `hash(seed, stage_name)`.
    pub seed: u64,
    /// Art-direction overrides.
    pub overrides: Overrides,
}
