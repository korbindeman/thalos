use glam::DVec3;
use std::collections::HashMap;

/// Gravitational constant in m^3 kg^-1 s^-2.
pub const G: f64 = 6.674_30e-11;

/// Minimum distance squared (m²) below which gravitational interactions are
/// skipped to avoid singularities. Equivalent to 100 m.
pub const MIN_DISTANCE_SQ: f64 = 1e4;

/// Minimum distance (m) for gravitational interaction cutoff.
pub const MIN_DISTANCE_M: f64 = 100.0;

/// Minimum distance squared (m²) for body-body gravitational interactions.
/// Set to 1 km² — appropriate for N-body integration where bodies have
/// meaningful radii.
pub const MIN_BODY_DISTANCE_SQ: f64 = 1e6;

// TODO: Uncomment after adding `pub mod parsing;` to lib.rs
pub use crate::parsing::load_solar_system;

#[cfg(test)]
const AU_TO_METERS: f64 = 1.496e11;

/// Unique identifier for a celestial body.
pub type BodyId = usize;

/// Position + velocity in heliocentric inertial frame.
#[derive(Debug, Clone, Copy)]
pub struct StateVector {
    pub position: DVec3,
    pub velocity: DVec3,
}

/// Static properties of a celestial body (immutable after load).
#[derive(Debug, Clone)]
pub struct BodyDefinition {
    pub id: BodyId,
    pub name: String,
    pub kind: BodyKind,
    pub parent: Option<BodyId>,
    pub mass_kg: f64,
    pub radius_m: f64,
    pub color: [f32; 3],
    pub gm: f64, // G * mass, precomputed
    pub orbital_elements: Option<OrbitalElements>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyKind {
    Star,
    Planet,
    Moon,
    DwarfPlanet,
    Centaur,
    Comet,
}

/// Keplerian orbital elements at epoch.
#[derive(Debug, Clone, Copy)]
pub struct OrbitalElements {
    pub semi_major_axis_m: f64,
    pub eccentricity: f64,
    pub inclination_rad: f64,
    pub lon_ascending_node_rad: f64,
    pub arg_periapsis_rad: f64,
    pub true_anomaly_rad: f64,
}

/// A timestamped state for a body — used in ephemeris samples.
#[derive(Debug, Clone, Copy)]
pub struct BodyState {
    pub position: DVec3,
    pub velocity: DVec3,
    pub mass_kg: f64,
}

/// Snapshot of all body states at a given time.
pub type BodyStates = Vec<BodyState>;

/// A single sample of the ship's propagated trajectory.
#[derive(Debug, Clone, Copy)]
pub struct TrajectorySample {
    pub time: f64,
    pub position: DVec3,
    pub velocity: DVec3,
    pub dominant_body: BodyId,
    pub perturbation_ratio: f64,
    pub step_size: f64,
}

/// Ship definition — placeholder for MVP.
#[derive(Debug, Clone)]
pub struct ShipDefinition {
    pub initial_state: StateVector,
    /// Thrust acceleration magnitude in m/s².
    pub thrust_acceleration: f64,
}

/// Full solar system definition loaded from file.
#[derive(Debug, Clone)]
pub struct SolarSystemDefinition {
    pub name: String,
    pub bodies: Vec<BodyDefinition>,
    pub ship: ShipDefinition,
    /// Map from body name to BodyId for convenience.
    pub name_to_id: HashMap<String, BodyId>,
}

impl SolarSystemDefinition {
    pub fn body_by_name(&self, name: &str) -> Option<&BodyDefinition> {
        self.name_to_id.get(name).map(|&id| &self.bodies[id])
    }
}

/// Parse a hex color string like "#FFE54D" into [r, g, b] floats in 0..1.
pub fn parse_hex_color(hex: &str) -> [f32; 3] {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(255) as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(255) as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(255) as f32 / 255.0;
    [r, g, b]
}

/// Convert orbital elements to Cartesian state vector relative to parent.
pub fn orbital_elements_to_cartesian(elements: &OrbitalElements, parent_gm: f64) -> StateVector {
    let a = elements.semi_major_axis_m;
    let e = elements.eccentricity;
    let i = elements.inclination_rad;
    let omega = elements.lon_ascending_node_rad;
    let w = elements.arg_periapsis_rad;
    let nu = elements.true_anomaly_rad;

    // Semi-latus rectum.
    let p = a * (1.0 - e * e);

    // Distance from focus.
    let r = p / (1.0 + e * nu.cos());

    // Position in orbital plane.
    let x_orb = r * nu.cos();
    let y_orb = r * nu.sin();

    // Velocity in orbital plane.
    let mu_over_p = parent_gm / p;
    let vx_orb = -mu_over_p.sqrt() * nu.sin();
    let vy_orb = mu_over_p.sqrt() * (e + nu.cos());

    // Rotation matrix from orbital plane to 3D (ecliptic) frame.
    // Using the standard aerospace convention: XZ ecliptic, Y up.
    let cos_o = omega.cos();
    let sin_o = omega.sin();
    let cos_w = w.cos();
    let sin_w = w.sin();
    let cos_i = i.cos();
    let sin_i = i.sin();

    let px = cos_o * cos_w - sin_o * sin_w * cos_i;
    let py = sin_i * sin_w;
    let pz = sin_o * cos_w + cos_o * sin_w * cos_i;

    let qx = -cos_o * sin_w - sin_o * cos_w * cos_i;
    let qy = sin_i * cos_w;
    let qz = -sin_o * sin_w + cos_o * cos_w * cos_i;

    // Note: Y is up in our coordinate system (ecliptic is XZ plane).
    let position = DVec3::new(
        x_orb * px + y_orb * qx,
        x_orb * py + y_orb * qy,
        x_orb * pz + y_orb * qz,
    );

    let velocity = DVec3::new(
        vx_orb * px + vy_orb * qx,
        vx_orb * py + vy_orb * qy,
        vx_orb * pz + vy_orb * qz,
    );

    StateVector { position, velocity }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_orbit_speed() {
        let elements = OrbitalElements {
            semi_major_axis_m: AU_TO_METERS,
            eccentricity: 0.0,
            inclination_rad: 0.0,
            lon_ascending_node_rad: 0.0,
            arg_periapsis_rad: 0.0,
            true_anomaly_rad: 0.0,
        };
        let sun_gm = G * 1.989e30;
        let sv = orbital_elements_to_cartesian(&elements, sun_gm);

        // For a circular orbit, |v| should equal sqrt(GM/r).
        let expected_speed = (sun_gm / AU_TO_METERS).sqrt();
        let actual_speed = sv.velocity.length();
        let rel_error = (actual_speed - expected_speed).abs() / expected_speed;
        assert!(rel_error < 1e-10, "Speed error: {rel_error}");

        // Position should be at (r, 0, 0) for zero angles.
        let rel_pos_error = (sv.position.length() - AU_TO_METERS).abs() / AU_TO_METERS;
        assert!(rel_pos_error < 1e-10, "Position error: {rel_pos_error}");
    }

    #[test]
    fn test_hex_color() {
        let c = parse_hex_color("#FF8000");
        assert!((c[0] - 1.0).abs() < 0.01);
        assert!((c[1] - 0.502).abs() < 0.01);
        assert!((c[2] - 0.0).abs() < 0.01);
    }
}
