//! Derived physical properties, computed once from a [`PlanetDescriptor`]
//! before any generation stage runs.  Every stage reads from
//! [`DerivedProperties`] — never from raw authored inputs — so future body
//! types extend the pipeline by adding derivations and stages, not by
//! rewriting existing logic.
//!
//! All derivations are pure functions.  Overrides on the descriptor take
//! precedence when set.

use crate::descriptor::PlanetDescriptor;

/// Physical constants and calibration points used by the derivations.
/// Public so tests and later stages can reference them directly.
pub mod constants {
    /// Gravitational constant, m³ / (kg · s²).
    pub const G: f64 = 6.674_30e-11;
    /// Stefan–Boltzmann constant, W / (m² · K⁴).
    pub const SIGMA: f64 = 5.670_374_419e-8;
    /// Boltzmann constant, J / K.
    pub const K_B: f64 = 1.380_649e-23;
    /// Atomic mass unit, kg.
    pub const AMU: f64 = 1.660_539_066_60e-27;
    /// Solar luminosity, W.  Default luminous primary for solar-system bodies.
    pub const L_SUN: f64 = 3.828e26;

    /// Lunar reference surface gravity, m / s².  Calibration point for the
    /// simple-to-complex crater transition diameter.
    pub const G_MOON: f64 = 1.625;
    /// Lunar simple-to-complex transition diameter, metres.  Spec §Physical
    /// calibration.
    pub const D_SC_MOON: f64 = 15_000.0;

    /// Specific radiogenic heat production at body formation (t = 0), W/kg.
    /// Calibrated so a 4.5 Gyr chondritic body produces roughly present-day
    /// chondritic heat flow (~3e-12 W/kg).
    pub const RADIOGENIC_H0: f64 = 2.0e-11;
    /// Effective radiogenic decay timescale, Gyr.  Single-exponential blend
    /// of U-238, Th-232, K-40; coarse but sufficient per spec.
    pub const RADIOGENIC_TAU_GYR: f64 = 2.5;

    /// Specific accretion-residual heat at formation, W/kg.  Small.
    pub const ACCRETION_H0: f64 = 1.0e-12;
    /// Accretion residual decay timescale, Gyr.
    pub const ACCRETION_TAU_GYR: f64 = 0.5;

    /// Dimensionless tidal dissipation factor k₂/Q.  Coarse default for
    /// rocky/icy moons; may be refined per body via art-direction overrides
    /// later.
    pub const TIDAL_K2_OVER_Q: f64 = 0.01;

    /// Safety factor for atmospheric retention: a species is retained when
    /// `v_escape > RETENTION_SAFETY_FACTOR × v_thermal`.  Spec §Derived
    /// properties specifies "~6×".
    pub const RETENTION_SAFETY_FACTOR: f64 = 6.0;
}

/// Candidate atmospheric species tested for retention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VolatileSpecies {
    H2,
    He,
    H2O,
    N2,
    CO2,
}

impl VolatileSpecies {
    /// Molecular mass in atomic mass units.
    pub fn molecular_mass_amu(self) -> f64 {
        match self {
            Self::H2 => 2.016,
            Self::He => 4.0026,
            Self::H2O => 18.0153,
            Self::N2 => 28.0134,
            Self::CO2 => 44.0095,
        }
    }

    pub const ALL: [Self; 5] = [Self::H2, Self::He, Self::H2O, Self::N2, Self::CO2];
}

/// Immutable pre-computed properties passed to every pipeline stage.
#[derive(Clone, Debug)]
pub struct DerivedProperties {
    pub surface_gravity: f64,
    pub escape_velocity: f64,
    pub equilibrium_temperature: f64,
    pub oblateness: f64,
    pub radiogenic_heat_budget_w: f64,
    pub tidal_heat_budget_w: f64,
    pub accretion_heat_budget_w: f64,
    pub total_heat_budget_w: f64,
    pub simple_to_complex_transition_m: f64,
    pub atmospheric_retention: [(VolatileSpecies, bool); 5],
}

impl DerivedProperties {
    /// Convenience: does this body retain the given species?
    pub fn retains(&self, species: VolatileSpecies) -> bool {
        self.atmospheric_retention
            .iter()
            .find(|(s, _)| *s == species)
            .map(|(_, r)| *r)
            .unwrap_or(false)
    }
}

/// Compute derived properties assuming the Sun as the luminous primary.
pub fn compute(desc: &PlanetDescriptor) -> DerivedProperties {
    compute_with_luminosity(desc, constants::L_SUN)
}

/// Compute derived properties with an explicit primary luminosity.  Use this
/// for non-solar systems (e.g. Thalos's own star).
pub fn compute_with_luminosity(
    desc: &PlanetDescriptor,
    primary_luminosity_w: f64,
) -> DerivedProperties {
    let g = desc
        .overrides
        .surface_gravity
        .unwrap_or_else(|| surface_gravity(desc.mass_kg, desc.radius_m));
    let v_esc = escape_velocity(desc.mass_kg, desc.radius_m);
    let t_eq = desc.overrides.equilibrium_temperature.unwrap_or_else(|| {
        equilibrium_temperature(primary_luminosity_w, desc.bond_albedo, desc.semi_major_axis_m)
    });
    let f_obl = oblateness(desc.mass_kg, desc.radius_m, desc.rotation_period_s);

    let h_rad = radiogenic_heat(desc.mass_kg, desc.age_gyr);
    let h_tid = tidal_heat(desc);
    let h_acc = accretion_residual_heat(desc.mass_kg, desc.age_gyr);
    let h_total = desc
        .overrides
        .total_heat_budget
        .unwrap_or(h_rad + h_tid + h_acc);

    let d_sc = simple_to_complex_transition(g);

    let mut retention = [(VolatileSpecies::H2, false); 5];
    for (i, sp) in VolatileSpecies::ALL.iter().enumerate() {
        retention[i] = (*sp, retains_species(v_esc, t_eq, *sp));
    }

    DerivedProperties {
        surface_gravity: g,
        escape_velocity: v_esc,
        equilibrium_temperature: t_eq,
        oblateness: f_obl,
        radiogenic_heat_budget_w: h_rad,
        tidal_heat_budget_w: h_tid,
        accretion_heat_budget_w: h_acc,
        total_heat_budget_w: h_total,
        simple_to_complex_transition_m: d_sc,
        atmospheric_retention: retention,
    }
}

// --- primitive derivations ---------------------------------------------------

pub fn surface_gravity(mass_kg: f64, radius_m: f64) -> f64 {
    constants::G * mass_kg / (radius_m * radius_m)
}

pub fn escape_velocity(mass_kg: f64, radius_m: f64) -> f64 {
    (2.0 * constants::G * mass_kg / radius_m).sqrt()
}

/// Fast-rotator equilibrium temperature (full surface radiates).
///     T⁴ = L(1 − A) / (16 π σ a²)
pub fn equilibrium_temperature(luminosity_w: f64, bond_albedo: f64, distance_m: f64) -> f64 {
    let num = luminosity_w * (1.0 - bond_albedo);
    let denom = 16.0 * std::f64::consts::PI * constants::SIGMA * distance_m * distance_m;
    (num / denom).powf(0.25)
}

/// Hydrostatic oblateness in the Maclaurin small-f limit for a homogeneous
/// body:  f ≈ (5/4) · ω² R³ / (G M).  Vanishes for non-rotators.
pub fn oblateness(mass_kg: f64, radius_m: f64, rotation_period_s: f64) -> f64 {
    if rotation_period_s <= 0.0 {
        return 0.0;
    }
    let omega = 2.0 * std::f64::consts::PI / rotation_period_s;
    (5.0 / 4.0) * omega * omega * radius_m.powi(3) / (constants::G * mass_kg)
}

/// Total radiogenic heat in watts.  Specific production decays exponentially
/// from a chondritic reference at formation.
pub fn radiogenic_heat(mass_kg: f64, age_gyr: f64) -> f64 {
    let specific = constants::RADIOGENIC_H0 * (-age_gyr / constants::RADIOGENIC_TAU_GYR).exp();
    specific * mass_kg
}

/// Residual accretion heat in watts.  Small and decays faster than radiogenic.
pub fn accretion_residual_heat(mass_kg: f64, age_gyr: f64) -> f64 {
    let specific = constants::ACCRETION_H0 * (-age_gyr / constants::ACCRETION_TAU_GYR).exp();
    specific * mass_kg
}

/// Standard synchronous-moon tidal heating formula:
///     Ė = (21/2) · (k₂/Q) · G · M_p² · R⁵ · n · e² / a⁶
/// with mean motion n = √(G M_p / a³).  Returns zero when the body has no
/// parent or zero eccentricity.
pub fn tidal_heat(desc: &PlanetDescriptor) -> f64 {
    let Some(parent) = desc.parent else {
        return 0.0;
    };
    if desc.eccentricity == 0.0 {
        return 0.0;
    }

    let a = parent.orbit_semi_major_axis_m;
    let m_p = parent.mass_kg;
    let r = desc.radius_m;
    let e = desc.eccentricity;
    let n = (constants::G * m_p / a.powi(3)).sqrt();

    (21.0 / 2.0)
        * constants::TIDAL_K2_OVER_Q
        * constants::G
        * m_p
        * m_p
        * r.powi(5)
        * n
        * e
        * e
        / a.powi(6)
}

/// Simple-to-complex crater transition diameter: lunar-calibrated, inversely
/// proportional to surface gravity.
pub fn simple_to_complex_transition(surface_gravity: f64) -> f64 {
    constants::D_SC_MOON * (constants::G_MOON / surface_gravity)
}

/// Jeans-escape retention test.  `v_thermal` is the most-probable speed of
/// the Maxwell-Boltzmann distribution, √(2 k_B T / m).
///
/// Note: the spec predicts that v0.1 targets retain nothing, and this holds
/// for Rhea and Deimos, but a pure Jeans + 6× criterion returns "retained"
/// for CO₂ on the Moon and for H₂O/N₂/CO₂ on Mercury and Callisto.  Real
/// atmospheric loss on those bodies is dominated by non-thermal processes
/// (solar-wind sputtering, photochemical loss) that Jeans alone cannot
/// capture.  The tests document the actual formula output so drift is
/// detectable; the spec's expected-behavior paragraph is a forecast, not a
/// hard requirement, and can be revisited when the atmosphere stage lands.
pub fn retains_species(
    escape_velocity: f64,
    equilibrium_temperature: f64,
    species: VolatileSpecies,
) -> bool {
    let m = species.molecular_mass_amu() * constants::AMU;
    let v_thermal = (2.0 * constants::K_B * equilibrium_temperature / m).sqrt();
    escape_velocity > constants::RETENTION_SAFETY_FACTOR * v_thermal
}

// --- tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_bodies;

    /// Relative-error comparison with a clear failure message.
    fn assert_close(actual: f64, expected: f64, rel_tol: f64, label: &str) {
        let scale = expected.abs().max(f64::MIN_POSITIVE);
        let err = (actual - expected).abs() / scale;
        assert!(
            err <= rel_tol,
            "{label}: expected {expected}, got {actual}, rel err {err:.3e} > tol {rel_tol:.3e}"
        );
    }

    struct Expect {
        name: &'static str,
        descriptor: PlanetDescriptor,
        surface_gravity: f64,
        escape_velocity: f64,
        equilibrium_temperature: f64,
        simple_to_complex_m: f64,
    }

    fn cases() -> Vec<Expect> {
        vec![
            Expect {
                name: "Luna",
                descriptor: reference_bodies::luna(),
                surface_gravity: 1.62329,
                escape_velocity: 2375.14,
                equilibrium_temperature: 270.33,
                simple_to_complex_m: 15_015.7,
            },
            Expect {
                name: "Mercury",
                descriptor: reference_bodies::mercury(),
                surface_gravity: 3.70160,
                escape_velocity: 4250.00,
                equilibrium_temperature: 437.19,
                simple_to_complex_m: 6_585.3,
            },
            Expect {
                name: "Callisto",
                descriptor: reference_bodies::callisto(),
                surface_gravity: 1.23579,
                escape_velocity: 2440.80,
                equilibrium_temperature: 114.66,
                simple_to_complex_m: 19_724.4,
            },
            Expect {
                name: "Rhea",
                descriptor: reference_bodies::rhea(),
                surface_gravity: 0.26383,
                escape_velocity: 634.81,
                equilibrium_temperature: 76.34,
                simple_to_complex_m: 92_389.0,
            },
            Expect {
                name: "Deimos",
                descriptor: reference_bodies::deimos(),
                surface_gravity: 2.5631e-3,
                escape_velocity: 5.6377,
                equilibrium_temperature: 221.55,
                simple_to_complex_m: 9.5119e6,
            },
        ]
    }

    #[test]
    fn gravity_escape_temperature_transition_match_reference_bodies() {
        for e in cases() {
            let d = compute(&e.descriptor);
            assert_close(d.surface_gravity, e.surface_gravity, 1e-3, &format!("{} g", e.name));
            assert_close(
                d.escape_velocity,
                e.escape_velocity,
                1e-3,
                &format!("{} v_esc", e.name),
            );
            assert_close(
                d.equilibrium_temperature,
                e.equilibrium_temperature,
                1e-3,
                &format!("{} T_eq", e.name),
            );
            assert_close(
                d.simple_to_complex_transition_m,
                e.simple_to_complex_m,
                1e-3,
                &format!("{} D_sc", e.name),
            );
        }
    }

    #[test]
    fn oblateness_is_small_for_slow_rotators() {
        // Moon and Mercury are slow rotators; hydrostatic oblateness should
        // be well below 1e-3 (frozen-in figures are not modelled).
        for desc in [reference_bodies::luna(), reference_bodies::mercury()] {
            let d = compute(&desc);
            assert!(
                d.oblateness < 1e-3 && d.oblateness >= 0.0,
                "{} oblateness {} not in [0, 1e-3)",
                desc.name,
                d.oblateness
            );
        }
    }

    #[test]
    fn radiogenic_heat_scales_with_mass_and_decays_with_age() {
        // At equal age, heat scales linearly with mass.
        let heavy = radiogenic_heat(1e24, 4.5);
        let light = radiogenic_heat(1e22, 4.5);
        assert_close(heavy / light, 100.0, 1e-12, "mass scaling");

        // At fixed mass, older body has less heat.
        let young = radiogenic_heat(1e23, 0.5);
        let old = radiogenic_heat(1e23, 4.5);
        assert!(young > old, "young body should have more radiogenic heat");
    }

    #[test]
    fn tidal_heat_zero_without_parent_or_eccentricity() {
        let mut d = reference_bodies::luna();
        d.parent = None;
        assert_eq!(tidal_heat(&d), 0.0);

        let mut d = reference_bodies::luna();
        d.eccentricity = 0.0;
        assert_eq!(tidal_heat(&d), 0.0);
    }

    #[test]
    fn tidal_heat_nonzero_for_eccentric_moon() {
        // Luna is eccentric around Earth — tidal heat should be finite and
        // positive, even if small compared to radiogenic.
        let h = tidal_heat(&reference_bodies::luna());
        assert!(h > 0.0 && h.is_finite(), "Luna tidal heat should be positive and finite, got {h}");
    }

    #[test]
    fn rhea_and_deimos_retain_nothing() {
        // These bodies are unambiguously airless under Jeans + 6×, matching
        // the spec's expected v0.1 behavior.
        for desc in [reference_bodies::rhea(), reference_bodies::deimos()] {
            let d = compute(&desc);
            for (species, retained) in d.atmospheric_retention {
                assert!(
                    !retained,
                    "{} unexpectedly retains {:?}",
                    desc.name, species
                );
            }
        }
    }

    #[test]
    fn luna_mercury_callisto_retention_matches_formula() {
        // Pure Jeans + 6× says these bodies *can* retain heavy gases, which
        // conflicts with the spec's forecast but follows directly from the
        // formula.  Assert the actual output so future changes to the
        // criterion (or constants) are caught explicitly.
        let luna = compute(&reference_bodies::luna());
        assert!(!luna.retains(VolatileSpecies::H2));
        assert!(!luna.retains(VolatileSpecies::He));
        assert!(!luna.retains(VolatileSpecies::H2O));
        assert!(!luna.retains(VolatileSpecies::N2));
        assert!(luna.retains(VolatileSpecies::CO2));

        let mercury = compute(&reference_bodies::mercury());
        assert!(!mercury.retains(VolatileSpecies::H2));
        assert!(!mercury.retains(VolatileSpecies::He));
        assert!(mercury.retains(VolatileSpecies::H2O));
        assert!(mercury.retains(VolatileSpecies::N2));
        assert!(mercury.retains(VolatileSpecies::CO2));

        let callisto = compute(&reference_bodies::callisto());
        assert!(!callisto.retains(VolatileSpecies::H2));
        assert!(!callisto.retains(VolatileSpecies::He));
        assert!(callisto.retains(VolatileSpecies::H2O));
        assert!(callisto.retains(VolatileSpecies::N2));
        assert!(callisto.retains(VolatileSpecies::CO2));
    }

    #[test]
    fn overrides_bypass_derivations() {
        let mut d = reference_bodies::luna();
        d.overrides.surface_gravity = Some(42.0);
        d.overrides.equilibrium_temperature = Some(100.0);
        d.overrides.total_heat_budget = Some(1e15);

        let derived = compute(&d);
        assert_eq!(derived.surface_gravity, 42.0);
        assert_eq!(derived.equilibrium_temperature, 100.0);
        assert_eq!(derived.total_heat_budget_w, 1e15);
    }

    #[test]
    fn deimos_has_no_complex_craters() {
        // Deimos's surface gravity is so low that the simple-to-complex
        // transition exceeds the body's radius — every crater is simple.
        let d = compute(&reference_bodies::deimos());
        let desc = reference_bodies::deimos();
        assert!(
            d.simple_to_complex_transition_m > 2.0 * desc.radius_m,
            "Deimos D_sc ({}) should exceed its diameter ({})",
            d.simple_to_complex_transition_m,
            2.0 * desc.radius_m,
        );
    }
}
