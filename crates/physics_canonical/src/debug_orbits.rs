//! Shared debug orbit constructors.
//!
//! These are not save-file semantics. They exist so development-only ship
//! spawn and teleport paths produce the same low, near-circular parking orbit.

use glam::DVec3;

use crate::orbital_math::near_circular_parking_speed;
use crate::types::{BodyDefinition, BodyState, StateVector};

/// Debug parking-orbit altitude used by initial ship spawn and debug teleports.
pub const DEBUG_PARKING_ORBIT_ALTITUDE_M: f64 = 200_000.0;

/// Fraction of a body's SOI that debug parking orbits should fit inside.
const DEBUG_PARKING_ORBIT_SOI_FRACTION: f64 = 0.3;

/// Altitude for the shared debug parking orbit around `body`.
pub fn debug_parking_orbit_altitude(body: &BodyDefinition) -> f64 {
    let mut altitude = DEBUG_PARKING_ORBIT_ALTITUDE_M;
    let max_altitude =
        (body.soi_radius_m * DEBUG_PARKING_ORBIT_SOI_FRACTION - body.radius_m).max(0.0);
    if max_altitude > 0.0 {
        altitude = altitude.min(max_altitude);
    }
    altitude
}

/// Relative state for the shared debug parking orbit around `body`.
///
/// Places the craft on `-X` with prograde velocity along `-Z` (equatorial,
/// inclination 0 in the XZ reference plane). Callers can add the body's
/// current state to get an absolute heliocentric state.
pub fn debug_parking_orbit_relative_state(body: &BodyDefinition) -> StateVector {
    let orbit_radius = body.radius_m + debug_parking_orbit_altitude(body);
    let orbital_speed = near_circular_parking_speed(body.gm, orbit_radius);

    StateVector {
        position: DVec3::new(-orbit_radius, 0.0, 0.0),
        velocity: DVec3::new(0.0, 0.0, -orbital_speed),
    }
}

/// Relative state for a near-circular **polar** debug parking orbit around
/// `body` (same altitude as [`debug_parking_orbit_relative_state`]).
///
/// Craft on `-X` with northbound velocity along `+Y` so specific angular
/// momentum lies in the XZ plane → inclination ≈ 90° under the house
/// convention (`orbital_math::cartesian_to_elements`).
pub fn debug_polar_parking_orbit_relative_state(body: &BodyDefinition) -> StateVector {
    let orbit_radius = body.radius_m + debug_parking_orbit_altitude(body);
    let orbital_speed = near_circular_parking_speed(body.gm, orbit_radius);

    StateVector {
        position: DVec3::new(-orbit_radius, 0.0, 0.0),
        velocity: DVec3::new(0.0, orbital_speed, 0.0),
    }
}

/// Absolute state for the shared debug parking orbit around `body`.
pub fn debug_parking_orbit_state(body: &BodyDefinition, body_state: &BodyState) -> StateVector {
    let rel = debug_parking_orbit_relative_state(body);
    StateVector {
        position: body_state.position + rel.position,
        velocity: body_state.velocity + rel.velocity,
    }
}

/// Absolute state for the polar debug parking orbit around `body`.
pub fn debug_polar_parking_orbit_state(
    body: &BodyDefinition,
    body_state: &BodyState,
) -> StateVector {
    let rel = debug_polar_parking_orbit_relative_state(body);
    StateVector {
        position: body_state.position + rel.position,
        velocity: body_state.velocity + rel.velocity,
    }
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;
    use crate::orbital_math::{NEAR_CIRCULAR_PARKING_ECCENTRICITY, cartesian_to_elements};
    use crate::types::{BodyKind, G};

    const EARTH_GM: f64 = G * 5.972e24;

    #[test]
    fn debug_parking_orbit_relative_state_uses_shared_altitude() {
        let body = BodyDefinition {
            id: 0,
            name: "Test".to_string(),
            kind: BodyKind::Planet,
            parent: None,
            mass_kg: EARTH_GM / G,
            radius_m: 3_186_000.0,
            color: [1.0, 1.0, 1.0],
            rotation_period_s: 86_400.0,
            axial_tilt_rad: 0.0,
            gm: EARTH_GM,
            soi_radius_m: 1.0e9,
            orbital_elements: None,
            terrain: thalos_world::TerrainConfig::None,
            tectonics: None,
            atmosphere: None,
            terrestrial_atmosphere: None,
            rings: None,
            surface_frame_ceiling_m: None,
        };

        let state = debug_parking_orbit_relative_state(&body);
        let altitude = state.position.length() - body.radius_m;
        let el = cartesian_to_elements(state, body.gm).unwrap();

        assert!((altitude - DEBUG_PARKING_ORBIT_ALTITUDE_M).abs() < 1e-6);
        assert_eq!(state.position.normalize(), DVec3::NEG_X);
        assert_eq!(state.velocity.normalize(), DVec3::NEG_Z);
        assert!((el.eccentricity - NEAR_CIRCULAR_PARKING_ECCENTRICITY).abs() < 1e-12);
        assert!(el.inclination_rad.abs() < 1e-9);
    }

    #[test]
    fn debug_polar_parking_orbit_is_near_circular_and_polar() {
        let body = BodyDefinition {
            id: 0,
            name: "Test".to_string(),
            kind: BodyKind::Planet,
            parent: None,
            mass_kg: EARTH_GM / G,
            radius_m: 3_186_000.0,
            color: [1.0, 1.0, 1.0],
            rotation_period_s: 86_400.0,
            axial_tilt_rad: 0.0,
            gm: EARTH_GM,
            soi_radius_m: 1.0e9,
            orbital_elements: None,
            terrain: thalos_world::TerrainConfig::None,
            tectonics: None,
            atmosphere: None,
            terrestrial_atmosphere: None,
            rings: None,
            surface_frame_ceiling_m: None,
        };

        let state = debug_polar_parking_orbit_relative_state(&body);
        let altitude = state.position.length() - body.radius_m;
        let el = cartesian_to_elements(state, body.gm).unwrap();

        assert!((altitude - DEBUG_PARKING_ORBIT_ALTITUDE_M).abs() < 1e-6);
        assert_eq!(state.position.normalize(), DVec3::NEG_X);
        assert_eq!(state.velocity.normalize(), DVec3::Y);
        assert!((el.eccentricity - NEAR_CIRCULAR_PARKING_ECCENTRICITY).abs() < 1e-12);
        assert!((el.inclination_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }
}
