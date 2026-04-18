//! Cartesian → Keplerian conversion and related orbital helpers.
//!
//! Consumed by trajectory event enrichment: given a relative state vector of
//! ship vs body at closest approach, produce osculating orbital elements so
//! the UI can show eccentricity, periapsis altitude, inclination, etc.

use glam::DVec3;

use crate::types::{OrbitalElements, StateVector, orbital_elements_to_cartesian};

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

    // Node vector: n = Ŷ × h. Points toward the ascending node in the XZ
    // plane. This matches `orbital_elements_to_cartesian`'s convention: for
    // i=π/4, Ω=0, ω=0, ν=0 the constructed orbit places periapsis at +X
    // (position = (a, 0, 0)), which equals the ascending node direction and
    // is exactly what Ŷ × h recovers here.
    let n_vec = DVec3::Y.cross(h);
    let n_mag = n_vec.length();
    let lon_ascending_node_rad = if n_mag > 1e-12 {
        // Ω is the angle of the node in the XZ plane measured from +X toward
        // +Z — the same sense `orbital_elements_to_cartesian` uses to build
        // P/Q. So Ω = atan2(n.z, n.x).
        let raan = n_vec.z.atan2(n_vec.x);
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

/// Propagate a state vector analytically under the gravity of a central body
/// with gravitational parameter `mu`, advancing by `dt` seconds.
///
/// `state_rel` is the state of the craft expressed relative to the central
/// body (position and velocity measured from the body, not heliocentric).
/// The result is in the same frame.
///
/// Handles elliptic (e < 1) and hyperbolic (e > 1) trajectories. Parabolic
/// trajectories (e == 1, exactly) are handled via a linear fallback — rare
/// enough that MVP accuracy is fine; a Barker-equation solver can replace
/// this without touching callers.
///
/// Degenerate inputs (zero position, zero angular momentum on a non-radial
/// state) are returned unchanged. Callers should treat these as pathological
/// and not propagate them further.
///
/// Numerical quality: Kepler's equation is solved with Newton-Raphson to
/// machine precision (typically 4-6 iterations for e < 0.8, up to ~16 for
/// e near 1). Energy and angular momentum are conserved to ~1e-12 relative
/// error over typical propagation spans.
pub fn propagate_kepler(state_rel: StateVector, mu: f64, dt: f64) -> StateVector {
    if dt == 0.0 || mu <= 0.0 {
        return state_rel;
    }

    let Some(elements) = cartesian_to_elements(state_rel, mu) else {
        return state_rel;
    };

    let e = elements.eccentricity;
    let a = elements.semi_major_axis_m;

    // Parabolic or near-parabolic: linear fallback for MVP.
    if !a.is_finite() || (e - 1.0).abs() < 1e-9 {
        return StateVector {
            position: state_rel.position + state_rel.velocity * dt,
            velocity: state_rel.velocity,
        };
    }

    let nu1 = if e < 1.0 {
        advance_elliptic(e, a, mu, elements.true_anomaly_rad, dt)
    } else {
        advance_hyperbolic(e, a, mu, elements.true_anomaly_rad, dt)
    };

    let updated = OrbitalElements {
        semi_major_axis_m: a,
        eccentricity: e,
        inclination_rad: elements.inclination_rad,
        lon_ascending_node_rad: elements.lon_ascending_node_rad,
        arg_periapsis_rad: elements.arg_periapsis_rad,
        true_anomaly_rad: nu1,
    };
    orbital_elements_to_cartesian(&updated, mu)
}

fn advance_elliptic(e: f64, a: f64, mu: f64, nu0: f64, dt: f64) -> f64 {
    let mean_motion = (mu / a.powi(3)).sqrt();
    let big_e0 = eccentric_from_true_elliptic(e, nu0);
    let m0 = big_e0 - e * big_e0.sin();
    let m1 = m0 + mean_motion * dt;
    let big_e1 = solve_kepler_elliptic(e, m1);
    true_from_eccentric_elliptic(e, big_e1)
}

fn advance_hyperbolic(e: f64, a: f64, mu: f64, nu0: f64, dt: f64) -> f64 {
    // Hyperbolic mean motion: n = sqrt(mu / (-a)^3). Negative `a` is the
    // convention for hyperbolic orbits.
    let mean_motion = (mu / (-a).powi(3)).sqrt();
    let h0 = hyperbolic_from_true(e, nu0);
    let n0 = e * h0.sinh() - h0;
    let n1 = n0 + mean_motion * dt;
    let h1 = solve_kepler_hyperbolic(e, n1);
    true_from_hyperbolic(e, h1)
}

/// E from ν for elliptic orbits using tan(E/2) = √((1-e)/(1+e)) · tan(ν/2).
/// atan2 form keeps the quadrant unambiguous for all ν.
pub(crate) fn eccentric_from_true_elliptic(e: f64, nu: f64) -> f64 {
    let (s, c) = (nu / 2.0).sin_cos();
    let y = (1.0 - e).sqrt() * s;
    let x = (1.0 + e).sqrt() * c;
    2.0 * y.atan2(x)
}

fn true_from_eccentric_elliptic(e: f64, big_e: f64) -> f64 {
    let (s, c) = (big_e / 2.0).sin_cos();
    let y = (1.0 + e).sqrt() * s;
    let x = (1.0 - e).sqrt() * c;
    2.0 * y.atan2(x)
}

/// Solve Kepler's equation E - e·sin(E) = M for E.
pub(crate) fn solve_kepler_elliptic(e: f64, m: f64) -> f64 {
    // Wrap M to [-π, π] for a well-conditioned initial guess.
    let pi = std::f64::consts::PI;
    let tau = std::f64::consts::TAU;
    let m = (m + pi).rem_euclid(tau) - pi;

    let mut big_e = if e < 0.8 {
        m
    } else if m >= 0.0 {
        pi
    } else {
        -pi
    };

    for _ in 0..32 {
        let (s, c) = big_e.sin_cos();
        let f = big_e - e * s - m;
        let fp = 1.0 - e * c;
        if fp.abs() < 1e-14 {
            break;
        }
        let delta = f / fp;
        big_e -= delta;
        if delta.abs() < 1e-13 {
            break;
        }
    }
    big_e
}

/// H from ν for hyperbolic orbits. Requires |ν| < arccos(-1/e) (the
/// asymptote angle); true for any state actually on the hyperbola.
pub(crate) fn hyperbolic_from_true(e: f64, nu: f64) -> f64 {
    // tanh(H/2) = √((e-1)/(e+1)) · tan(ν/2)
    let arg = ((e - 1.0) / (e + 1.0)).sqrt() * (nu / 2.0).tan();
    // Clamp to avoid atanh(±1) = ∞ from float noise exactly at the asymptote.
    let arg = arg.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    2.0 * arg.atanh()
}

fn true_from_hyperbolic(e: f64, h: f64) -> f64 {
    // ν = 2·atan2(√(e+1)·sinh(H/2), √(e-1)·cosh(H/2))
    let sh = (h / 2.0).sinh();
    let ch = (h / 2.0).cosh();
    let y = (e + 1.0).sqrt() * sh;
    let x = (e - 1.0).sqrt() * ch;
    2.0 * y.atan2(x)
}

/// Solve the hyperbolic Kepler equation e·sinh(H) - H = N for H.
pub(crate) fn solve_kepler_hyperbolic(e: f64, n: f64) -> f64 {
    // Battin's initial guess: works across small and large |N| uniformly.
    let mut h = n.signum() * (2.0 * n.abs() / e + 1.8).ln();
    if !h.is_finite() {
        h = 0.0;
    }

    for _ in 0..64 {
        let sh = h.sinh();
        let ch = h.cosh();
        let f = e * sh - h - n;
        let fp = e * ch - 1.0;
        if fp.abs() < 1e-14 {
            break;
        }
        let delta = f / fp;
        h -= delta;
        if delta.abs() < 1e-13 {
            break;
        }
    }
    h
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

    // -----------------------------------------------------------------
    // propagate_kepler tests
    // -----------------------------------------------------------------

    fn energy(state: StateVector, mu: f64) -> f64 {
        0.5 * state.velocity.length_squared() - mu / state.position.length()
    }

    fn ang_mom(state: StateVector) -> DVec3 {
        state.position.cross(state.velocity)
    }

    #[test]
    fn propagate_zero_dt_is_identity() {
        let state = StateVector {
            position: DVec3::new(8.0e6, 1.0e6, 2.0e6),
            velocity: DVec3::new(1000.0, 5000.0, 3000.0),
        };
        let result = propagate_kepler(state, EARTH_GM, 0.0);
        assert_eq!(result.position, state.position);
        assert_eq!(result.velocity, state.velocity);
    }

    #[test]
    fn propagate_circular_one_period_returns_to_start() {
        let r = 7.0e6;
        let v = (EARTH_GM / r).sqrt();
        let state = StateVector {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, v),
        };
        let period = std::f64::consts::TAU * (r * r * r / EARTH_GM).sqrt();
        let end = propagate_kepler(state, EARTH_GM, period);
        let pos_err = (end.position - state.position).length() / r;
        let vel_err = (end.velocity - state.velocity).length() / v;
        assert!(pos_err < 1e-8, "position drift after period: {pos_err}");
        assert!(vel_err < 1e-8, "velocity drift after period: {vel_err}");
    }

    #[test]
    fn propagate_elliptic_half_period_reaches_apoapsis() {
        // Start at periapsis on a 0.3-eccentric ellipse; half a period later
        // the craft should be at apoapsis, distance a(1+e) from the focus.
        let a = 1.0e8;
        let e = 0.3;
        let rp = a * (1.0 - e);
        let vp = (EARTH_GM * (1.0 + e) / rp).sqrt();
        let state = StateVector {
            position: DVec3::new(rp, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, vp),
        };
        let period = std::f64::consts::TAU * (a.powi(3) / EARTH_GM).sqrt();
        let end = propagate_kepler(state, EARTH_GM, period / 2.0);
        let ra = a * (1.0 + e);
        let r_end = end.position.length();
        let rel_err = (r_end - ra).abs() / ra;
        assert!(rel_err < 1e-8, "apoapsis distance: {r_end}, expected {ra}");
        // At apoapsis the radial velocity is zero — v is perpendicular to r.
        let radial = end.velocity.dot(end.position.normalize());
        assert!(radial.abs() / vp < 1e-8, "radial velocity at apoapsis: {radial}");
    }

    #[test]
    fn propagate_conserves_energy_and_angular_momentum() {
        let state = StateVector {
            position: DVec3::new(8.0e6, 1.0e6, 2.0e6),
            velocity: DVec3::new(1000.0, 5000.0, 3000.0),
        };
        let e0 = energy(state, EARTH_GM);
        let l0 = ang_mom(state);

        let end = propagate_kepler(state, EARTH_GM, 10_000.0);
        let e1 = energy(end, EARTH_GM);
        let l1 = ang_mom(end);

        assert!((e1 - e0).abs() / e0.abs() < 1e-10, "energy drift");
        assert!((l1 - l0).length() / l0.length() < 1e-10, "angular momentum drift");
    }

    #[test]
    fn propagate_round_trip_inverts() {
        let state = StateVector {
            position: DVec3::new(8.0e6, 1.0e6, 2.0e6),
            velocity: DVec3::new(1000.0, 5000.0, 3000.0),
        };
        let dt = 3600.0;
        let forward = propagate_kepler(state, EARTH_GM, dt);
        let back = propagate_kepler(forward, EARTH_GM, -dt);
        let pos_err = (back.position - state.position).length() / state.position.length();
        let vel_err = (back.velocity - state.velocity).length() / state.velocity.length();
        assert!(pos_err < 1e-9, "position round-trip error: {pos_err}");
        assert!(vel_err < 1e-9, "velocity round-trip error: {vel_err}");
    }

    #[test]
    fn propagate_retrograde_returns_after_period() {
        // Same circular radius, but orbiting the other way.
        let r = 7.0e6;
        let v = (EARTH_GM / r).sqrt();
        let state = StateVector {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, -v),
        };
        let period = std::f64::consts::TAU * (r * r * r / EARTH_GM).sqrt();
        let end = propagate_kepler(state, EARTH_GM, period);
        let pos_err = (end.position - state.position).length() / r;
        let vel_err = (end.velocity - state.velocity).length() / v;
        assert!(pos_err < 1e-8, "retrograde position drift: {pos_err}");
        assert!(vel_err < 1e-8, "retrograde velocity drift: {vel_err}");
    }

    #[test]
    fn propagate_inclined_conserves_plane() {
        // Inclined orbit: angular momentum direction must not drift.
        let els = OrbitalElements {
            semi_major_axis_m: 1.0e7,
            eccentricity: 0.2,
            inclination_rad: 0.7,
            lon_ascending_node_rad: 0.5,
            arg_periapsis_rad: 1.1,
            true_anomaly_rad: 0.3,
        };
        let state = orbital_elements_to_cartesian(&els, EARTH_GM);
        let h0 = ang_mom(state).normalize();
        let end = propagate_kepler(state, EARTH_GM, 5000.0);
        let h1 = ang_mom(end).normalize();
        let dot = h0.dot(h1);
        assert!(dot > 0.999_999_999, "orbital plane drift: dot = {dot}");
    }

    #[test]
    fn propagate_hyperbolic_conserves_energy() {
        // Escape trajectory: positive energy must be preserved exactly.
        let r = 7.0e6;
        let v_circ = (EARTH_GM / r).sqrt();
        let v = v_circ * 1.5;
        let state = StateVector {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, v),
        };
        let e0 = energy(state, EARTH_GM);
        assert!(e0 > 0.0);
        let end = propagate_kepler(state, EARTH_GM, 1000.0);
        let e1 = energy(end, EARTH_GM);
        assert!((e1 - e0).abs() / e0 < 1e-9, "hyperbolic energy drift");
        // Craft should be farther from the focus after 1000 s outbound.
        assert!(end.position.length() > r);
    }

    #[test]
    fn propagate_hyperbolic_round_trips() {
        let r = 7.0e6;
        let v_circ = (EARTH_GM / r).sqrt();
        let state = StateVector {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, v_circ * 1.5),
        };
        let dt = 500.0;
        let forward = propagate_kepler(state, EARTH_GM, dt);
        let back = propagate_kepler(forward, EARTH_GM, -dt);
        let pos_err = (back.position - state.position).length() / r;
        let vel_err = (back.velocity - state.velocity).length() / v_circ;
        assert!(pos_err < 1e-8, "hyperbolic round-trip pos err: {pos_err}");
        assert!(vel_err < 1e-8, "hyperbolic round-trip vel err: {vel_err}");
    }

    #[test]
    fn propagate_near_circular_is_stable() {
        // e = 1e-10 — near-circular, tests degenerate-element handling.
        let r = 7.0e6;
        let v = (EARTH_GM / r).sqrt() * (1.0 + 1e-10);
        let state = StateVector {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, v),
        };
        let end = propagate_kepler(state, EARTH_GM, 3600.0);
        assert!(end.position.is_finite());
        assert!(end.velocity.is_finite());
        let r_end = end.position.length();
        // Near-circular: radius should stay close to r.
        assert!((r_end - r).abs() / r < 1e-3);
    }
}
