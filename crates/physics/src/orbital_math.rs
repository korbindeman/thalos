//! Cartesian → Keplerian conversion and related orbital helpers.
//!
//! Consumed by trajectory event enrichment: given a relative state vector of
//! ship vs body at closest approach, produce osculating orbital elements so
//! the UI can show eccentricity, periapsis altitude, inclination, etc.

use glam::DVec3;

use crate::types::StateVector;

/// Osculating Keplerian elements of a state vector relative to a central
/// body with gravitational parameter `mu`.
///
/// Handles elliptic, parabolic, and hyperbolic trajectories.  Degenerate
/// cases (circular / equatorial / exactly parabolic) fall back to `0.0` for
/// undefined angles rather than producing NaN.
#[derive(Debug, Clone, Copy)]
pub struct OsculatingElements {
    /// Semi-major axis (m).  Negative for hyperbolic trajectories; infinite
    /// for exactly parabolic (returned as `f64::INFINITY`).
    pub semi_major_axis_m: f64,
    /// Eccentricity.  `< 1` bound, `≥ 1` unbound.
    pub eccentricity: f64,
    /// Inclination (radians), measured from the reference plane (XZ).
    pub inclination_rad: f64,
    /// Longitude of ascending node (radians).  `0.0` for equatorial orbits.
    pub lon_ascending_node_rad: f64,
    /// Argument of periapsis (radians).  `0.0` for circular orbits.
    pub arg_periapsis_rad: f64,
    /// True anomaly at epoch (radians).
    pub true_anomaly_rad: f64,
    /// Periapsis radius (m), measured from the central body.
    pub periapsis_m: f64,
    /// Apoapsis radius (m).  Infinite for unbound orbits.
    pub apoapsis_m: f64,
    /// Specific orbital energy (J / kg).  Negative bound, positive unbound.
    pub specific_energy: f64,
}

/// Convert a relative state vector (position and velocity of craft w.r.t. the
/// central body) into osculating Keplerian elements.
///
/// Returns `None` if `mu <= 0` or the state is degenerate enough to be
/// numerically meaningless (zero position or zero angular momentum with
/// non-radial motion).  Radial trajectories with zero angular momentum are
/// reported as rectilinear: eccentricity = 1, inclination / node undefined
/// (set to `0.0`).
pub fn cartesian_to_elements(rel: StateVector, mu: f64) -> Option<OsculatingElements> {
    if mu <= 0.0 {
        return None;
    }

    let r = rel.position;
    let v = rel.velocity;
    let r_mag = r.length();
    if r_mag <= 0.0 {
        return None;
    }
    let v_mag_sq = v.length_squared();

    // Specific orbital energy: ε = v²/2 - μ/r.
    let specific_energy = 0.5 * v_mag_sq - mu / r_mag;

    // Semi-major axis from energy.  Parabolic (ε ≈ 0) → infinity.
    let semi_major_axis_m = if specific_energy.abs() < 1e-20 {
        f64::INFINITY
    } else {
        -mu / (2.0 * specific_energy)
    };

    // Specific angular momentum h = r × v.
    let h = r.cross(v);
    let h_mag = h.length();

    // Eccentricity vector: e = v × h / μ - r̂.
    let e_vec = v.cross(h) / mu - r.normalize_or_zero();
    let eccentricity = e_vec.length();

    // Inclination: angle between h and the reference prograde normal.
    // Thalos uses the XZ ecliptic plane with Y up; `orbital_elements_to_cartesian`
    // places a zero-inclination orbit with h pointing along −Y (v̂ × r̂ on the
    // orbital plane for a circular prograde orbit), so inclination is
    // acos(−h·Ŷ / |h|).
    let inclination_rad = if h_mag > 0.0 {
        (-h.y / h_mag).clamp(-1.0, 1.0).acos()
    } else {
        0.0
    };

    // Node vector: n = (−Y) × h.  Points toward ascending node in the XZ plane.
    let n_vec = (-DVec3::Y).cross(h);
    let n_mag = n_vec.length();
    let lon_ascending_node_rad = if n_mag > 1e-12 {
        // RAAN measured around Y from +X.  X-component = cos, -Z component
        // = sin (right-handed around +Y).
        let raan = (-n_vec.z).atan2(n_vec.x);
        raan.rem_euclid(std::f64::consts::TAU)
    } else {
        0.0
    };

    let arg_periapsis_rad = if n_mag > 1e-12 && eccentricity > 1e-9 {
        let cos_w = (n_vec.dot(e_vec) / (n_mag * eccentricity)).clamp(-1.0, 1.0);
        let mut w = cos_w.acos();
        // Quadrant: if e has +Y component, periapsis is above the node.
        if e_vec.y < 0.0 {
            w = std::f64::consts::TAU - w;
        }
        w
    } else {
        0.0
    };

    let true_anomaly_rad = if eccentricity > 1e-9 {
        let cos_nu = (e_vec.dot(r) / (eccentricity * r_mag)).clamp(-1.0, 1.0);
        let mut nu = cos_nu.acos();
        if r.dot(v) < 0.0 {
            nu = std::f64::consts::TAU - nu;
        }
        nu
    } else if n_mag > 1e-12 {
        // Circular inclined: argument of latitude.
        let cos_u = (n_vec.dot(r) / (n_mag * r_mag)).clamp(-1.0, 1.0);
        let mut u = cos_u.acos();
        if r.y < 0.0 {
            u = std::f64::consts::TAU - u;
        }
        u
    } else {
        // Circular equatorial: true longitude.
        let cos_l = (r.x / r_mag).clamp(-1.0, 1.0);
        let mut l = cos_l.acos();
        if r.z > 0.0 {
            l = std::f64::consts::TAU - l;
        }
        l
    };

    let (periapsis_m, apoapsis_m) = if eccentricity < 1.0 && semi_major_axis_m.is_finite() {
        (
            semi_major_axis_m * (1.0 - eccentricity),
            semi_major_axis_m * (1.0 + eccentricity),
        )
    } else if eccentricity >= 1.0 && semi_major_axis_m.is_finite() {
        // Hyperbolic: a < 0; periapsis = a(1-e).
        (semi_major_axis_m * (1.0 - eccentricity), f64::INFINITY)
    } else {
        // Parabolic: use h² / μ as p, rp = p/2.
        let p = if mu > 0.0 { h_mag * h_mag / mu } else { 0.0 };
        (0.5 * p, f64::INFINITY)
    };

    Some(OsculatingElements {
        semi_major_axis_m,
        eccentricity,
        inclination_rad,
        lon_ascending_node_rad,
        arg_periapsis_rad,
        true_anomaly_rad,
        periapsis_m,
        apoapsis_m,
        specific_energy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AU_TO_METERS, G, orbital_elements_to_cartesian};

    const SUN_GM: f64 = G * 1.989e30;
    const EARTH_GM: f64 = G * 5.972e24;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn circular_equatorial_roundtrip() {
        let r = AU_TO_METERS;
        let v = (SUN_GM / r).sqrt();
        let state = StateVector {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, v),
        };
        let el = cartesian_to_elements(state, SUN_GM).unwrap();
        assert!(approx_eq(el.semi_major_axis_m, r, 1.0));
        assert!(el.eccentricity < 1e-6);
        assert!(el.inclination_rad.abs() < 1e-6);
        assert!(el.specific_energy < 0.0);
    }

    #[test]
    fn elliptical_periapsis_apoapsis() {
        // Build a known ellipse, round-trip check.
        let els = crate::types::OrbitalElements {
            semi_major_axis_m: 1.5e11,
            eccentricity: 0.4,
            inclination_rad: 0.2,
            lon_ascending_node_rad: 0.3,
            arg_periapsis_rad: 0.5,
            true_anomaly_rad: 0.7,
        };
        let sv = orbital_elements_to_cartesian(&els, SUN_GM);
        let back = cartesian_to_elements(sv, SUN_GM).unwrap();
        assert!(approx_eq(back.semi_major_axis_m, els.semi_major_axis_m, 1e3));
        assert!(approx_eq(back.eccentricity, 0.4, 1e-6));
        assert!(approx_eq(back.inclination_rad, 0.2, 1e-6));
        assert!(approx_eq(
            back.periapsis_m,
            1.5e11 * (1.0 - 0.4),
            1e3
        ));
        assert!(approx_eq(
            back.apoapsis_m,
            1.5e11 * (1.0 + 0.4),
            1e3
        ));
    }

    #[test]
    fn hyperbolic_escape() {
        let r = 7_000_000.0;
        let v_circ = (EARTH_GM / r).sqrt();
        let v = v_circ * 1.5; // well above escape
        let state = StateVector {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, v),
        };
        let el = cartesian_to_elements(state, EARTH_GM).unwrap();
        assert!(el.eccentricity > 1.0);
        assert!(el.specific_energy > 0.0);
        assert!(el.semi_major_axis_m < 0.0);
        assert!(el.periapsis_m > 0.0);
        assert!(el.apoapsis_m.is_infinite());
    }

    #[test]
    fn degenerate_inputs_do_not_nan() {
        let state = StateVector {
            position: DVec3::ZERO,
            velocity: DVec3::X,
        };
        assert!(cartesian_to_elements(state, EARTH_GM).is_none());

        let state2 = StateVector {
            position: DVec3::X * 7_000_000.0,
            velocity: DVec3::ZERO,
        };
        let el = cartesian_to_elements(state2, EARTH_GM).unwrap();
        assert!(el.eccentricity.is_finite());
        assert!(el.inclination_rad.is_finite());
    }
}
