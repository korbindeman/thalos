//! Canonical [`PlanetDescriptor`]s for the five v0.1 reference bodies:
//! Luna, Mercury, Callisto, Rhea, Deimos.  These are the spec's calibration
//! targets — any change to the derivation code or physical constants must
//! keep [`crate::derived`]'s tests passing against these inputs.
//!
//! Values are drawn from standard planetary fact sheets (NASA / Wikipedia).
//! Bond albedos favour commonly-cited bond (not geometric) values where they
//! exist; composition, thermal history, and impact-flux fields are coarse
//! ballparks sufficient for v0.1 and will be refined as later stages land.

use crate::descriptor::{Composition, Overrides, ParentBody, PlanetDescriptor};

const GYR: f64 = 4.5;

/// Earth's Moon.
pub fn luna() -> PlanetDescriptor {
    PlanetDescriptor {
        name: "Luna".to_string(),
        radius_m: 1_737_400.0,
        mass_kg: 7.342e22,
        semi_major_axis_m: 1.496e11,
        bond_albedo: 0.11,
        composition: Composition::new(0.98, 0.02, 0.0, 0.0, 0.0),
        age_gyr: GYR,
        rotation_period_s: 2.360_6e6, // 27.32 d, synchronous with Earth
        axial_tilt_rad: 0.117, // 6.7° to orbital plane
        eccentricity: 0.0549,
        parent: Some(ParentBody {
            mass_kg: 5.9722e24,
            orbit_semi_major_axis_m: 3.844e8,
        }),
        impact_flux_multiplier: 1.0,
        thermal_history: 0.6,
        seed: 0x4C554E41, // "LUNA"
        overrides: Overrides::default(),
    }
}

/// Mercury.
pub fn mercury() -> PlanetDescriptor {
    PlanetDescriptor {
        name: "Mercury".to_string(),
        radius_m: 2_439_700.0,
        mass_kg: 3.3011e23,
        semi_major_axis_m: 5.7909e10,
        bond_albedo: 0.088,
        composition: Composition::new(0.30, 0.70, 0.0, 0.0, 0.0),
        age_gyr: GYR,
        rotation_period_s: 5.0670e6, // 58.646 d sidereal
        axial_tilt_rad: 5.9e-4,      // 0.034°
        eccentricity: 0.2056,
        parent: None, // Heliocentric; tidal heating from the Sun is negligible and not modelled.
        impact_flux_multiplier: 1.5,
        thermal_history: 0.5,
        seed: 0x4D455243,
        overrides: Overrides::default(),
    }
}

/// Callisto, Jupiter's outer Galilean moon.
pub fn callisto() -> PlanetDescriptor {
    PlanetDescriptor {
        name: "Callisto".to_string(),
        // a_sma is Jupiter's heliocentric mean distance — the luminous
        // primary for equilibrium temperature is the Sun, not Jupiter.
        radius_m: 2_410_300.0,
        mass_kg: 1.0759e23,
        semi_major_axis_m: 7.785e11,
        bond_albedo: 0.22,
        composition: Composition::new(0.50, 0.0, 0.50, 0.0, 0.0),
        age_gyr: GYR,
        rotation_period_s: 1.4421e6, // 16.689 d, synchronous with Jupiter
        axial_tilt_rad: 0.0,
        eccentricity: 0.0074,
        parent: Some(ParentBody {
            mass_kg: 1.898e27,
            orbit_semi_major_axis_m: 1.8827e9,
        }),
        impact_flux_multiplier: 2.0,
        thermal_history: 0.1,
        seed: 0x43414C4C,
        overrides: Overrides::default(),
    }
}

/// Rhea, Saturn's second-largest moon.
pub fn rhea() -> PlanetDescriptor {
    PlanetDescriptor {
        name: "Rhea".to_string(),
        radius_m: 763_800.0,
        mass_kg: 2.306e21,
        semi_major_axis_m: 1.434e12, // Saturn heliocentric
        bond_albedo: 0.48,
        composition: Composition::new(0.25, 0.0, 0.75, 0.0, 0.0),
        age_gyr: GYR,
        rotation_period_s: 4.518e5, // 4.518 d, synchronous with Saturn
        axial_tilt_rad: 0.0,
        eccentricity: 0.001,
        parent: Some(ParentBody {
            mass_kg: 5.6834e26,
            orbit_semi_major_axis_m: 5.2704e8,
        }),
        impact_flux_multiplier: 1.5,
        thermal_history: 0.1,
        seed: 0x52484541,
        overrides: Overrides::default(),
    }
}

/// Deimos, Mars's outer moon.
pub fn deimos() -> PlanetDescriptor {
    PlanetDescriptor {
        name: "Deimos".to_string(),
        radius_m: 6_200.0, // mean; Deimos is not spherical
        mass_kg: 1.4762e15,
        semi_major_axis_m: 2.2794e11, // Mars heliocentric
        bond_albedo: 0.068,
        composition: Composition::new(0.95, 0.0, 0.05, 0.0, 0.0),
        age_gyr: GYR,
        rotation_period_s: 1.0926e5, // 1.263 d, synchronous with Mars
        axial_tilt_rad: 0.0,
        eccentricity: 0.00033,
        parent: Some(ParentBody {
            mass_kg: 6.4171e23,
            orbit_semi_major_axis_m: 2.3463e7,
        }),
        impact_flux_multiplier: 1.0,
        thermal_history: 0.0,
        seed: 0x4445494D,
        overrides: Overrides::default(),
    }
}
