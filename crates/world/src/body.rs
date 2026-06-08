//! Body and system data types — the authored physical/orbital definition.

use crate::atmosphere::{AtmosphereParams, RingSystem, TerrestrialAtmosphere};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thalos_terrain::{TectonicConfig, TerrainConfig};

/// Gravitational constant in m^3 kg^-1 s^-2.
pub const G: f64 = 6.674_30e-11;

pub const AU_TO_METERS: f64 = 1.496e11;

/// Unique identifier for a celestial body.
pub type BodyId = usize;

/// Position + velocity in heliocentric inertial frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StateVector {
    pub position: DVec3,
    pub velocity: DVec3,
}

/// Static physical properties of a celestial body (immutable after load).
///
/// Built from a `BodyFile` at parse time. `id`, `parent`, and `gm` are
/// resolved/computed by the loader; everything else mirrors the file.
#[derive(Debug, Clone)]
pub struct BodyDefinition {
    pub id: BodyId,
    pub name: String,
    pub kind: BodyKind,
    pub parent: Option<BodyId>,
    pub mass_kg: f64,
    pub radius_m: f64,
    /// sRGB hex color, used for UI icons and fallback rendering only.
    /// Real shading is driven by terrain/atmosphere definitions, not
    /// this field.
    pub color: [f32; 3],
    pub rotation_period_s: f64,
    pub axial_tilt_rad: f64,
    pub gm: f64, // G * mass, precomputed
    /// Sphere-of-influence radius (m).  Computed at load time from
    /// `a * (m / M_parent)^(2/5)`.  The star (no parent) gets `f64::INFINITY`
    /// so any point in the system falls inside it as a fallback anchor.
    pub soi_radius_m: f64,
    pub orbital_elements: Option<OrbitalElements>,
    pub terrain: TerrainConfig,
    /// Optional tectonic structural prior. When present, bodies carry a
    /// plate graph (mesh + plates + boundaries + per-cell distance fields)
    /// that the editor visualizes and a future `SurfaceField` height
    /// contribution will read. Independent of `terrain` — bodies on the
    /// flat-water `Ocean` placeholder can still carry tectonics, and
    /// bodies with feature terrain can opt into or out of tectonics
    /// independently.
    pub tectonics: Option<TectonicConfig>,
    /// Gas / ice giant atmosphere definition. A body with
    /// `atmosphere: Some(_)` and no `terrain` is rendered as a gas
    /// giant (optically thick all the way down, no solid surface).
    /// Mutually exclusive with `terrestrial_atmosphere` — a body has at
    /// most one atmosphere schema attached.
    pub atmosphere: Option<AtmosphereParams>,
    /// Thin atmosphere over a solid surface. Paired with `terrain`:
    /// a body with both set renders the baked impostor with an
    /// atmosphere shell composited over it (rim halo, limb shading).
    /// Mutually exclusive with `atmosphere` (the gas-giant schema).
    pub terrestrial_atmosphere: Option<TerrestrialAtmosphere>,
    /// Optional ring system, independent of body type. A ring annulus
    /// is rendered around any body that authors this — gas giants,
    /// rocky bodies, dwarf planets, all alike. Note: cloud-deck
    /// ring-shadow is wired only for gas-giant bodies; surfaces of
    /// terrain-baked bodies don't yet receive a ring shadow.
    pub rings: Option<RingSystem>,
    /// Altitude (m above the reference radius) at or below which the
    /// navball auto-selects the Surface velocity frame. Bodies with a
    /// `terrestrial_atmosphere` use the Kármán line at the call site;
    /// this authored value is the airless fallback / explicit override.
    /// `None` ⇒ the game derives a radius-fraction default.
    pub surface_frame_ceiling_m: Option<f64>,
}

impl BodyDefinition {
    /// Surface gravity g = GM/r² (m/s²) at the mean surface. Used by the
    /// aerodynamic atmosphere model to derive the density scale height.
    pub fn surface_gravity_m_s2(&self) -> f64 {
        if self.radius_m > 0.0 {
            self.gm / (self.radius_m * self.radius_m)
        } else {
            0.0
        }
    }

    /// Surface atmospheric pressure (Pa), the single authored source for the
    /// aerodynamic atmosphere model. Read from the terrain environment's
    /// `AtmosphereSpec` (so pressure isn't authored twice). Falls back to ~1 bar
    /// for a body that carries a `terrestrial_atmosphere` shell but no
    /// terrain-authored pressure, and to 0 (vacuum) for an explicit `None`
    /// atmosphere.
    pub fn surface_pressure_pa(&self) -> f64 {
        const ONE_BAR_PA: f64 = 101_325.0;
        match &self.terrain {
            TerrainConfig::Feature(feature) => {
                let bar = feature.environment.atmosphere.pressure_bar() as f64;
                if bar > 0.0 {
                    bar * ONE_BAR_PA
                } else {
                    0.0
                }
            }
            // No feature terrain to read pressure from: assume ~1 bar if a
            // terrestrial atmosphere shell is present, else vacuum.
            _ => {
                if self.terrestrial_atmosphere.is_some() {
                    ONE_BAR_PA
                } else {
                    0.0
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum BodyKind {
    Star,
    Planet,
    Moon,
    DwarfPlanet,
    Centaur,
    Comet,
}

/// Keplerian orbital elements at epoch. Stored in radians; the file format
/// uses degrees and converts at load time.
#[derive(Debug, Clone, Copy)]
pub struct OrbitalElements {
    pub semi_major_axis_m: f64,
    pub eccentricity: f64,
    pub inclination_rad: f64,
    pub lon_ascending_node_rad: f64,
    pub arg_periapsis_rad: f64,
    pub true_anomaly_rad: f64,
}

/// Full solar system definition loaded from file.
///
/// Pure authored data: the named bodies and a name→id index. The player's
/// initial spawn state is *not* stored here — it is a derived, debug-only
/// parking orbit computed by the consumer from `homeworld_id` via
/// `thalos_physics_canonical::debug_orbits`.
#[derive(Debug, Clone)]
pub struct SolarSystemDefinition {
    pub name: String,
    pub bodies: Vec<BodyDefinition>,
    /// Map from body name to BodyId for convenience.
    pub name_to_id: HashMap<String, BodyId>,
    /// The body the player's default spawn orbits — resolved from the file's
    /// `homeworld` name at load time.
    pub homeworld_id: BodyId,
}

impl SolarSystemDefinition {
    pub fn body_by_name(&self, name: &str) -> Option<&BodyDefinition> {
        self.name_to_id.get(name).map(|&id| &self.bodies[id])
    }

    /// The homeworld body — the default spawn anchor.
    pub fn homeworld(&self) -> &BodyDefinition {
        &self.bodies[self.homeworld_id]
    }
}

/// Orbital-plane → ecliptic basis (P, Q) for a set of Keplerian elements.
///
/// `P` points toward periapsis in the inertial frame; `Q` is in the orbital
/// plane, perpendicular to `P`, in the direction of motion at periapsis. The
/// pair forms an orthonormal basis sufficient to lift any (x_orb, y_orb)
/// orbital-plane coordinate into the inertial XZ-ecliptic frame.
pub fn keplerian_basis(elements: &OrbitalElements) -> (DVec3, DVec3) {
    let cos_o = elements.lon_ascending_node_rad.cos();
    let sin_o = elements.lon_ascending_node_rad.sin();
    let cos_w = elements.arg_periapsis_rad.cos();
    let sin_w = elements.arg_periapsis_rad.sin();
    let cos_i = elements.inclination_rad.cos();
    let sin_i = elements.inclination_rad.sin();

    let p = DVec3::new(
        cos_o * cos_w - sin_o * sin_w * cos_i,
        sin_i * sin_w,
        sin_o * cos_w + cos_o * sin_w * cos_i,
    );
    let q = DVec3::new(
        -cos_o * sin_w - sin_o * cos_w * cos_i,
        sin_i * cos_w,
        -sin_o * sin_w + cos_o * cos_w * cos_i,
    );
    (p, q)
}

/// Convert orbital elements to Cartesian state vector relative to parent.
pub fn orbital_elements_to_cartesian(elements: &OrbitalElements, parent_gm: f64) -> StateVector {
    let a = elements.semi_major_axis_m;
    let e = elements.eccentricity;
    let nu = elements.true_anomaly_rad;

    // Semi-latus rectum.
    let p_slr = a * (1.0 - e * e);

    // Distance from focus.
    let r = p_slr / (1.0 + e * nu.cos());

    // Position in orbital plane.
    let x_orb = r * nu.cos();
    let y_orb = r * nu.sin();

    // Velocity in orbital plane.
    let mu_over_p = parent_gm / p_slr;
    let vx_orb = -mu_over_p.sqrt() * nu.sin();
    let vy_orb = mu_over_p.sqrt() * (e + nu.cos());

    // Lift orbital-plane coords into the inertial (XZ-ecliptic, Y up) frame.
    let (basis_p, basis_q) = keplerian_basis(elements);

    StateVector {
        position: basis_p * x_orb + basis_q * y_orb,
        velocity: basis_p * vx_orb + basis_q * vy_orb,
    }
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
}
