use glam::{DQuat, DVec3};
use serde::Deserialize;
use std::collections::HashMap;
use thalos_atmosphere_gen::{AtmosphereParams, TerrestrialAtmosphere};
use thalos_terrain_gen::GeneratorParams;

/// Gravitational constant in m^3 kg^-1 s^-2.
pub const G: f64 = 6.674_30e-11;

pub const AU_TO_METERS: f64 = 1.496e11;

/// Unique identifier for a celestial body.
pub type BodyId = usize;

/// Position + velocity in heliocentric inertial frame.
#[derive(Debug, Clone, Copy)]
pub struct StateVector {
    pub position: DVec3,
    pub velocity: DVec3,
}

/// Ship attitude state. Kept separate from [`StateVector`] so trajectory
/// prediction (which doesn't care about orientation) stays cheap.
///
/// `orientation` is the body→world quaternion; `angular_velocity` is
/// expressed in the **body frame** (rad/s) — convention `Iω̇ = τ` plays
/// out cleanly when both `ω` and `τ` are in body coordinates.
#[derive(Debug, Clone, Copy)]
pub struct AttitudeState {
    pub orientation: DQuat,
    pub angular_velocity: DVec3,
}

impl Default for AttitudeState {
    fn default() -> Self {
        Self {
            orientation: DQuat::IDENTITY,
            angular_velocity: DVec3::ZERO,
        }
    }
}

/// Static physical properties needed to integrate ship attitude and thrust.
///
/// `moment_of_inertia` is the principal-axis MOI tensor's diagonal in
/// kg·m², expressed in the body frame. Off-diagonal terms are assumed
/// zero — adequate for axially-symmetric ship stacks. `max_torque` is
/// the per-axis torque cap from all reaction-wheel-providing parts
/// summed, in N·m.
///
/// `thrust_n`, `mass_flow_kg_per_s`, and `dry_mass_kg` are constants for
/// v1 (no staging). Current ship mass is tracked separately on
/// [`crate::Simulation`] because it changes as fuel burns; once it
/// reaches `dry_mass_kg` thrust cuts off cleanly (the propellant tanks
/// are empty).
#[derive(Debug, Clone, Copy)]
pub struct ShipParameters {
    pub moment_of_inertia: DVec3,
    pub max_torque: DVec3,
    pub thrust_n: f64,
    pub mass_flow_kg_per_s: f64,
    /// Dry mass — the floor under which `Simulation::ship_mass_kg` cannot
    /// fall, and the threshold below which thrust stops being applied.
    /// "Out of fuel" in physical units rather than an arbitrary numerical
    /// safety floor.
    pub dry_mass_kg: f64,
}

impl Default for ShipParameters {
    fn default() -> Self {
        // Sentinel values: nonzero MOI to avoid divide-by-zero, zero
        // torque so a ship with no parameters set can't accidentally
        // accept attitude commands. Zero thrust = drifting until a real
        // ship is spawned and pushes its blueprint stats in. Dry mass
        // sits at the safety floor so the integrator's mass never
        // divides by zero before a real ship has been pushed in.
        Self {
            moment_of_inertia: DVec3::ONE,
            max_torque: DVec3::ZERO,
            thrust_n: 0.0,
            mass_flow_kg_per_s: 0.0,
            dry_mass_kg: MIN_SHIP_MASS_KG,
        }
    }
}

/// Hard numerical floor on ship mass — keeps the integrator from dividing
/// by zero before a real ship has been spawned and `dry_mass_kg` set. Once
/// a ship is spawned, its actual `dry_mass_kg` is the operative floor.
pub(crate) const MIN_SHIP_MASS_KG: f64 = 1.0;

/// Player attitude + thrust command sampled each frame and pushed into
/// the simulation via [`crate::simulation::Simulation::set_control`].
///
/// `torque_command` is in body frame, components in `[-1, 1]`. Each
/// axis is multiplied by the matching [`ShipParameters::max_torque`]
/// component to produce the actual torque applied. `throttle` is the
/// player's commanded engine throttle, in `[0, 1]`. The bridge gates
/// this on fuel availability before sending; the simulation trusts the
/// value it receives and applies thrust along the body nose direction.
#[derive(Debug, Clone, Copy, Default)]
pub struct ControlInput {
    pub torque_command: DVec3,
    pub sas_enabled: bool,
    pub throttle: f64,
}

/// Static properties of a celestial body (immutable after load).
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
    pub color: [f32; 3],
    pub albedo: f32,
    pub rotation_period_s: f64,
    pub axial_tilt_rad: f64,
    pub gm: f64, // G * mass, precomputed
    /// Sphere-of-influence radius (m).  Computed at load time from
    /// `a * (m / M_parent)^(2/5)`.  The star (no parent) gets `f64::INFINITY`
    /// so any point in the system falls inside it as a fallback anchor.
    pub soi_radius_m: f64,
    pub orbital_elements: Option<OrbitalElements>,
    pub generator: Option<GeneratorParams>,
    /// Gas / ice giant atmosphere definition. A body with
    /// `atmosphere: Some(_)` and no `generator` is rendered as a gas
    /// giant (optically thick all the way down, no solid surface).
    /// Mutually exclusive with `terrestrial_atmosphere` — a body has at
    /// most one atmosphere schema attached.
    pub atmosphere: Option<AtmosphereParams>,
    /// Thin atmosphere over a solid surface. Paired with `generator`:
    /// a body with both set renders the baked impostor with an
    /// atmosphere shell composited over it (rim halo, limb shading).
    /// Mutually exclusive with `atmosphere` (the gas-giant schema).
    pub terrestrial_atmosphere: Option<TerrestrialAtmosphere>,
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
///
/// Under the analytical patched-conics propagator there is one gravitational
/// source per sample — the SOI body — so rendering, colouring, and encounter
/// detection all share the single `anchor_body` field. `ref_pos` is the
/// anchor body's heliocentric position at `time`, cached on the sample so
/// the renderer can compute the anchor-relative position without an
/// ephemeris query per sample per frame.
#[derive(Debug, Clone, Copy)]
pub struct TrajectorySample {
    pub time: f64,
    pub position: DVec3,
    pub velocity: DVec3,
    pub anchor_body: BodyId,
    /// `anchor_body`'s position at `time`, cached for cheap rendering.
    pub ref_pos: DVec3,
}

/// Ship definition — placeholder for MVP. Holds only the spawn state;
/// thrust/mass/mass-flow come from the ship blueprint at spawn time and
/// are pushed into [`crate::Simulation`] via setters.
#[derive(Debug, Clone)]
pub struct ShipDefinition {
    pub initial_state: StateVector,
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
