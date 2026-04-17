use glam::DVec3;
use serde::Deserialize;
use std::collections::HashMap;
use thalos_atmosphere_gen::AtmosphereParams;
use thalos_terrain_gen::GeneratorParams;

pub use crate::effects::gravity::{BODY_WEIGHTS_CAP, EMPTY_BODY_WEIGHTS};

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

pub const AU_TO_METERS: f64 = 1.496e11;

/// Unique identifier for a celestial body.
pub type BodyId = usize;

/// Position + velocity in heliocentric inertial frame.
#[derive(Debug, Clone, Copy)]
pub struct StateVector {
    pub position: DVec3,
    pub velocity: DVec3,
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
    /// Gas / ice giant atmosphere definition. Mutually meaningful with
    /// `generator`: a body with `atmosphere: Some(_)` and no `generator`
    /// is rendered as a gas giant (optically thick atmosphere all the
    /// way down, no solid surface). Bodies with both set — terrestrials
    /// with a thin atmosphere — will eventually composite the
    /// atmosphere shell over the baked surface, but that path is not
    /// wired up yet.
    pub atmosphere: Option<AtmosphereParams>,
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

/// Compute the gravity-weighted barycenter `Σᵢ wᵢ · rᵢ` over the bodies
/// listed in `body_weights`, looking up each body's position in `bodies`.
/// Missing-id entries (slot weight 0, or id ≥ len) contribute nothing.
#[inline]
pub fn weighted_barycenter(
    body_weights: &[(BodyId, f32); BODY_WEIGHTS_CAP],
    bodies: &[BodyState],
) -> DVec3 {
    let mut acc = DVec3::ZERO;
    for &(id, w) in body_weights.iter() {
        if w > 0.0 && id < bodies.len() {
            acc += bodies[id].position * w as f64;
        }
    }
    acc
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
    /// Body with the largest gravitational pull on the ship at this sample.
    /// Used for color tinting and the perturbation cone signal — *not* for
    /// rendering frame.  Can flicker between samples.
    pub dominant_body: BodyId,
    pub perturbation_ratio: f64,
    pub step_size: f64,
    /// Rendering anchor: the hierarchical parent used for drawing the
    /// trajectory.  For moons, this is stepped up to the parent planet so
    /// the trajectory stays in the planet's reference frame.
    ///
    /// Retained for legacy consumers (label-coloring, encounter detection).
    /// The renderer itself uses `body_weights` + `ref_pos` via the
    /// gravity-weighted barycenter rule (§7.2).
    pub anchor_body: BodyId,
    /// Gravity-weighted barycenter `Σᵢ wᵢ · rᵢ(sample.t)` at this sample's
    /// time, computed from `body_weights`. The renderer subtracts this to
    /// place the sample in the barycenter-relative frame, then adds the
    /// current-time barycenter to pin the trajectory to where the dominant
    /// bodies are now (§7.2).
    pub ref_pos: DVec3,
    /// SOI-level body: the smallest sphere-of-influence that contains the
    /// ship.  Used for encounter detection (SOI entry/exit, periapsis)
    /// independently of the rendering anchor.
    pub soi_body: BodyId,
    /// Top-K gravity weights at this sample, `wᵢ = aᵢ / Σⱼ aⱼ` renormalised
    /// across the stored entries. Drives the renderer's gravity-weighted
    /// barycenter rule: `render_pos = sample.pos - Σ wᵢ · rᵢ(t)`. Unused
    /// slots have weight 0.0.
    pub body_weights: [(BodyId, f32); BODY_WEIGHTS_CAP],
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
}
