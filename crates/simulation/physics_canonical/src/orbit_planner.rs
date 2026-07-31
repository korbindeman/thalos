//! Pure target-orbit planning.
//!
//! This module turns an osculating two-body state into ordinary
//! [`ManeuverNode`](crate::maneuver::ManeuverNode)s. It deliberately does not
//! execute burns or know about UI state: generated plans travel through the
//! same prediction and autopilot path as hand-authored nodes.

use std::f64::consts::{PI, TAU};

use glam::DVec3;
use thalos_world::{BodyId, StateVector};

use crate::maneuver::{ManeuverNode, orbital_frame};
use crate::orbital_math::{
    OsculatingElements, cartesian_to_elements, eccentric_from_true_elliptic, propagate_kepler,
};

/// Minimum time left before the first generated burn.
///
/// The runtime burn executor needs time to leave warp and slew. A plan made
/// closer to an apsis waits one orbit rather than publishing a node that cannot
/// be prepared honestly.
pub const MIN_FIRST_BURN_LEAD_S: f64 = 120.0;
pub const APSIS_RADIUS_TOLERANCE_M: f64 = 25.0;
pub const INCLINATION_TOLERANCE_RAD: f64 = 1.0e-5;
const MEANINGFUL_DELTA_V_M_S: f64 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitDirection {
    Prograde,
    Retrograde,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetPlane {
    PreserveCurrent,
    /// Cheapest plane with the requested absolute inclination. The angle is
    /// measured away from the equator and must lie in `[0, π/2]`; direction
    /// selects the prograde or retrograde solution.
    Nearest {
        inclination_rad: f64,
        direction: OrbitDirection,
    },
    /// Exact plane targeting is part of the public contract but requires a
    /// launch-window/node solver. The first planner rejects it explicitly.
    Fixed {
        inclination_rad: f64,
        ascending_node_longitude_rad: f64,
        direction: OrbitDirection,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetOrbit {
    pub reference_body: BodyId,
    pub periapsis_altitude_m: f64,
    pub apoapsis_altitude_m: f64,
    pub plane: TargetPlane,
}

impl TargetOrbit {
    pub fn circular(reference_body: BodyId, altitude_m: f64) -> Self {
        Self {
            reference_body,
            periapsis_altitude_m: altitude_m,
            apoapsis_altitude_m: altitude_m,
            plane: TargetPlane::PreserveCurrent,
        }
    }

    pub fn target_inclination_rad(self, current: f64) -> Result<f64, OrbitPlanError> {
        match self.plane {
            TargetPlane::PreserveCurrent => Ok(current),
            TargetPlane::Nearest {
                inclination_rad,
                direction,
            } => resolve_inclination(inclination_rad, direction),
            TargetPlane::Fixed { .. } => Err(OrbitPlanError::FixedPlaneUnsupported),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OrbitPlanRequest {
    /// Craft state relative to `target.reference_body`.
    pub state: StateVector,
    pub epoch_s: f64,
    pub mu: f64,
    pub body_radius_m: f64,
    pub target: TargetOrbit,
}

#[derive(Debug, Clone)]
pub struct OrbitPlan {
    pub target: TargetOrbit,
    pub nodes: Vec<ManeuverNode>,
    pub total_delta_v_m_s: f64,
    pub predicted_elements: OsculatingElements,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrbitPlanError {
    InvalidGravity,
    InvalidBodyRadius,
    InvalidTargetAltitude,
    PeriapsisAboveApoapsis,
    TargetIntersectsBody,
    DegenerateState,
    CurrentOrbitNotBound,
    InvalidInclination,
    FixedPlaneUnsupported,
    NoSafeTransfer,
    ValidationFailed {
        periapsis_error_m: f64,
        apoapsis_error_m: f64,
        inclination_error_rad: f64,
    },
}

#[derive(Debug, Clone, Copy)]
enum Apsis {
    Periapsis,
    Apoapsis,
}

#[derive(Debug, Clone)]
struct Candidate {
    nodes: Vec<ManeuverNode>,
    total_delta_v_m_s: f64,
    final_state: StateVector,
    final_epoch_s: f64,
}

/// Plan an ordinary maneuver-node sequence from the current bound orbit to the
/// requested target orbit.
///
/// The target does not constrain argument of periapsis. The planner therefore
/// evaluates transfers between both current apsides and both target apsides,
/// keeping the lowest-delta-v safe result. Inclination changes are performed at
/// the next equator crossing after the energy transfer.
pub fn plan_target_orbit(request: OrbitPlanRequest) -> Result<OrbitPlan, OrbitPlanError> {
    validate_request(request)?;
    let current =
        cartesian_to_elements(request.state, request.mu).ok_or(OrbitPlanError::DegenerateState)?;
    if current.eccentricity >= 1.0
        || !current.semi_major_axis_m.is_finite()
        || current.semi_major_axis_m <= 0.0
    {
        return Err(OrbitPlanError::CurrentOrbitNotBound);
    }

    let target_periapsis_m = request.body_radius_m + request.target.periapsis_altitude_m;
    let target_apoapsis_m = request.body_radius_m + request.target.apoapsis_altitude_m;
    let target_inclination_rad = request
        .target
        .target_inclination_rad(current.inclination_rad)?;

    let mut best: Option<Candidate> = None;
    for start in [Apsis::Periapsis, Apsis::Apoapsis] {
        for finish in [Apsis::Periapsis, Apsis::Apoapsis] {
            let Some(candidate) = energy_transfer_candidate(
                request,
                current,
                start,
                finish,
                target_periapsis_m,
                target_apoapsis_m,
            ) else {
                continue;
            };
            let candidate = add_inclination_change(candidate, request, target_inclination_rad)?;
            if best
                .as_ref()
                .is_none_or(|incumbent| candidate.total_delta_v_m_s < incumbent.total_delta_v_m_s)
            {
                best = Some(candidate);
            }
        }
    }

    let best = best.ok_or(OrbitPlanError::NoSafeTransfer)?;
    let predicted_elements = cartesian_to_elements(best.final_state, request.mu)
        .ok_or(OrbitPlanError::DegenerateState)?;
    validate_result(
        predicted_elements,
        target_periapsis_m,
        target_apoapsis_m,
        target_inclination_rad,
    )?;

    Ok(OrbitPlan {
        target: request.target,
        nodes: best.nodes,
        total_delta_v_m_s: best.total_delta_v_m_s,
        predicted_elements,
    })
}

/// Tangential speed at `radius_m` on an ellipse whose other apsis is
/// `opposite_apsis_radius_m`.
pub fn apsis_speed(mu: f64, radius_m: f64, opposite_apsis_radius_m: f64) -> f64 {
    let semi_major_axis_m = 0.5 * (radius_m + opposite_apsis_radius_m);
    (mu * (2.0 / radius_m - 1.0 / semi_major_axis_m))
        .max(0.0)
        .sqrt()
}

/// Prograde-frame delta-v that changes the opposite apsis while burning at the
/// current apsis.
pub fn set_opposite_apsis_delta_v(
    mu: f64,
    radius_m: f64,
    current_tangential_speed_m_s: f64,
    desired_opposite_apsis_radius_m: f64,
) -> f64 {
    apsis_speed(mu, radius_m, desired_opposite_apsis_radius_m) - current_tangential_speed_m_s
}

/// Prograde-frame delta-v that circularizes at the current radius.
pub fn circularize_delta_v(mu: f64, radius_m: f64, current_tangential_speed_m_s: f64) -> f64 {
    (mu / radius_m).sqrt() - current_tangential_speed_m_s
}

fn validate_request(request: OrbitPlanRequest) -> Result<(), OrbitPlanError> {
    if !request.mu.is_finite() || request.mu <= 0.0 {
        return Err(OrbitPlanError::InvalidGravity);
    }
    if !request.body_radius_m.is_finite() || request.body_radius_m <= 0.0 {
        return Err(OrbitPlanError::InvalidBodyRadius);
    }
    let pe = request.target.periapsis_altitude_m;
    let ap = request.target.apoapsis_altitude_m;
    if !pe.is_finite() || !ap.is_finite() {
        return Err(OrbitPlanError::InvalidTargetAltitude);
    }
    if pe > ap {
        return Err(OrbitPlanError::PeriapsisAboveApoapsis);
    }
    if pe <= 0.0 {
        return Err(OrbitPlanError::TargetIntersectsBody);
    }
    Ok(())
}

fn resolve_inclination(
    inclination_rad: f64,
    direction: OrbitDirection,
) -> Result<f64, OrbitPlanError> {
    if !inclination_rad.is_finite() || !(0.0..=PI * 0.5).contains(&inclination_rad) {
        return Err(OrbitPlanError::InvalidInclination);
    }
    Ok(match direction {
        OrbitDirection::Prograde => inclination_rad,
        OrbitDirection::Retrograde => PI - inclination_rad,
    })
}

#[allow(clippy::too_many_arguments)]
fn energy_transfer_candidate(
    request: OrbitPlanRequest,
    current: OsculatingElements,
    start: Apsis,
    finish: Apsis,
    target_periapsis_m: f64,
    target_apoapsis_m: f64,
) -> Option<Candidate> {
    let wait_s = time_to_next_apsis(current, request.mu, start, MIN_FIRST_BURN_LEAD_S)?;
    let first_epoch_s = request.epoch_s + wait_s;
    let first_state = propagate_kepler(request.state, request.mu, wait_s);
    let first_radius_m = first_state.position.length();
    let destination_radius_m = match finish {
        Apsis::Periapsis => target_periapsis_m,
        Apsis::Apoapsis => target_apoapsis_m,
    };
    if first_radius_m.min(destination_radius_m) <= request.body_radius_m {
        return None;
    }

    let first_speed_m_s = tangential_speed(first_state);
    let transfer_speed_m_s = apsis_speed(request.mu, first_radius_m, destination_radius_m);
    let first_delta_v_m_s = transfer_speed_m_s - first_speed_m_s;
    let first_delta_v = DVec3::new(first_delta_v_m_s, 0.0, 0.0);
    let mut after_first = apply_orbital_delta_v(first_state, first_delta_v);

    let transfer_a_m = 0.5 * (first_radius_m + destination_radius_m);
    if !transfer_a_m.is_finite() || transfer_a_m <= 0.0 {
        return None;
    }
    let coast_s = PI * (transfer_a_m.powi(3) / request.mu).sqrt();
    after_first = propagate_kepler(after_first, request.mu, coast_s);

    let destination_speed_m_s = tangential_speed(after_first);
    let target_other_radius_m = match finish {
        Apsis::Periapsis => target_apoapsis_m,
        Apsis::Apoapsis => target_periapsis_m,
    };
    let target_speed_m_s = apsis_speed(request.mu, destination_radius_m, target_other_radius_m);
    let second_delta_v_m_s = target_speed_m_s - destination_speed_m_s;
    let second_delta_v = DVec3::new(second_delta_v_m_s, 0.0, 0.0);
    let after_second = apply_orbital_delta_v(after_first, second_delta_v);
    let second_epoch_s = first_epoch_s + coast_s;

    let mut nodes = Vec::with_capacity(3);
    push_meaningful_node(
        &mut nodes,
        first_epoch_s,
        first_delta_v,
        request.target.reference_body,
    );
    push_meaningful_node(
        &mut nodes,
        second_epoch_s,
        second_delta_v,
        request.target.reference_body,
    );
    let total_delta_v_m_s = nodes.iter().map(|node| node.delta_v.length()).sum();

    Some(Candidate {
        nodes,
        total_delta_v_m_s,
        final_state: after_second,
        final_epoch_s: second_epoch_s,
    })
}

fn add_inclination_change(
    mut candidate: Candidate,
    request: OrbitPlanRequest,
    target_inclination_rad: f64,
) -> Result<Candidate, OrbitPlanError> {
    let elements = cartesian_to_elements(candidate.final_state, request.mu)
        .ok_or(OrbitPlanError::DegenerateState)?;
    let inclination_error = target_inclination_rad - elements.inclination_rad;
    if inclination_error.abs() <= INCLINATION_TOLERANCE_RAD {
        return Ok(candidate);
    }

    let ascending_nu = (-elements.arg_periapsis_rad).rem_euclid(TAU);
    let descending_nu = (PI - elements.arg_periapsis_rad).rem_euclid(TAU);
    let ascending_wait =
        time_until_true_anomaly(elements, request.mu, ascending_nu).unwrap_or(f64::INFINITY);
    let descending_wait =
        time_until_true_anomaly(elements, request.mu, descending_nu).unwrap_or(f64::INFINITY);
    let wait_s = ascending_wait.min(descending_wait);
    if !wait_s.is_finite() {
        return Err(OrbitPlanError::NoSafeTransfer);
    }
    let node_state = propagate_kepler(candidate.final_state, request.mu, wait_s);
    let speed_m_s = node_state.velocity.length();
    let angle = inclination_error.abs();

    let prograde_delta = speed_m_s * (angle.cos() - 1.0);
    let normal_delta = speed_m_s * angle.sin();
    let positive = DVec3::new(prograde_delta, normal_delta, 0.0);
    let negative = DVec3::new(prograde_delta, -normal_delta, 0.0);
    let positive_state = apply_orbital_delta_v(node_state, positive);
    let negative_state = apply_orbital_delta_v(node_state, negative);
    let positive_error = inclination_residual(positive_state, request.mu, target_inclination_rad);
    let negative_error = inclination_residual(negative_state, request.mu, target_inclination_rad);
    let (delta_v, final_state) = if positive_error <= negative_error {
        (positive, positive_state)
    } else {
        (negative, negative_state)
    };

    let node_epoch_s = candidate.final_epoch_s + wait_s;
    push_meaningful_node(
        &mut candidate.nodes,
        node_epoch_s,
        delta_v,
        request.target.reference_body,
    );
    candidate.total_delta_v_m_s += delta_v.length();
    candidate.final_state = final_state;
    candidate.final_epoch_s = node_epoch_s;
    Ok(candidate)
}

fn validate_result(
    elements: OsculatingElements,
    target_periapsis_m: f64,
    target_apoapsis_m: f64,
    target_inclination_rad: f64,
) -> Result<(), OrbitPlanError> {
    let periapsis_error_m = (elements.periapsis_m - target_periapsis_m).abs();
    let apoapsis_error_m = (elements.apoapsis_m - target_apoapsis_m).abs();
    let inclination_error_rad = (elements.inclination_rad - target_inclination_rad).abs();
    if periapsis_error_m > APSIS_RADIUS_TOLERANCE_M
        || apoapsis_error_m > APSIS_RADIUS_TOLERANCE_M
        || inclination_error_rad > INCLINATION_TOLERANCE_RAD
    {
        return Err(OrbitPlanError::ValidationFailed {
            periapsis_error_m,
            apoapsis_error_m,
            inclination_error_rad,
        });
    }
    Ok(())
}

fn time_to_next_apsis(
    elements: OsculatingElements,
    mu: f64,
    apsis: Apsis,
    min_lead_s: f64,
) -> Option<f64> {
    if elements.eccentricity >= 1.0 || elements.semi_major_axis_m <= 0.0 {
        return None;
    }
    let eccentric_anomaly =
        eccentric_from_true_elliptic(elements.eccentricity, elements.true_anomaly_rad);
    let mean_anomaly = eccentric_anomaly - elements.eccentricity * eccentric_anomaly.sin();
    let mean_motion = (mu / elements.semi_major_axis_m.powi(3)).sqrt();
    let target_mean_anomaly = match apsis {
        Apsis::Periapsis => 0.0,
        Apsis::Apoapsis => PI,
    };
    let period_s = TAU / mean_motion;
    let mut wait_s = ((target_mean_anomaly - mean_anomaly).rem_euclid(TAU)) / mean_motion;
    if wait_s < min_lead_s {
        wait_s += period_s;
    }
    Some(wait_s)
}

fn time_until_true_anomaly(
    elements: OsculatingElements,
    mu: f64,
    target_true_anomaly_rad: f64,
) -> Option<f64> {
    if elements.eccentricity >= 1.0 || elements.semi_major_axis_m <= 0.0 {
        return None;
    }
    let current_e = eccentric_from_true_elliptic(elements.eccentricity, elements.true_anomaly_rad);
    let target_e = eccentric_from_true_elliptic(elements.eccentricity, target_true_anomaly_rad);
    let current_m = current_e - elements.eccentricity * current_e.sin();
    let target_m = target_e - elements.eccentricity * target_e.sin();
    let mean_motion = (mu / elements.semi_major_axis_m.powi(3)).sqrt();
    Some((target_m - current_m).rem_euclid(TAU) / mean_motion)
}

fn apply_orbital_delta_v(state: StateVector, delta_v: DVec3) -> StateVector {
    let [prograde, normal, radial] =
        orbital_frame(state.position, state.velocity, DVec3::ZERO, DVec3::ZERO);
    StateVector {
        position: state.position,
        velocity: state.velocity + delta_v.x * prograde + delta_v.y * normal + delta_v.z * radial,
    }
}

fn tangential_speed(state: StateVector) -> f64 {
    let radial = state.position.normalize_or_zero();
    (state.velocity - radial * state.velocity.dot(radial)).length()
}

fn inclination_residual(state: StateVector, mu: f64, target: f64) -> f64 {
    cartesian_to_elements(state, mu).map_or(f64::INFINITY, |elements| {
        (elements.inclination_rad - target).abs()
    })
}

fn push_meaningful_node(
    nodes: &mut Vec<ManeuverNode>,
    time: f64,
    delta_v: DVec3,
    reference_body: BodyId,
) {
    if delta_v.length() >= MEANINGFUL_DELTA_V_M_S {
        nodes.push(ManeuverNode {
            id: None,
            time,
            delta_v,
            reference_body,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU: f64 = 3.986_004_418e14;
    const RADIUS: f64 = 6_371_000.0;

    fn state_at_periapsis(periapsis_m: f64, apoapsis_m: f64, inclination_rad: f64) -> StateVector {
        let elements = thalos_world::OrbitalElements {
            semi_major_axis_m: 0.5 * (periapsis_m + apoapsis_m),
            eccentricity: (apoapsis_m - periapsis_m) / (apoapsis_m + periapsis_m),
            inclination_rad,
            lon_ascending_node_rad: 0.0,
            arg_periapsis_rad: 0.0,
            true_anomaly_rad: 0.0,
        };
        thalos_world::orbital_elements_to_cartesian(&elements, MU)
    }

    #[test]
    fn circularization_helper_matches_vis_viva() {
        let radius = RADIUS + 200_000.0;
        let ellipse_speed = apsis_speed(MU, radius, RADIUS + 800_000.0);
        let dv = circularize_delta_v(MU, radius, ellipse_speed);
        assert!((ellipse_speed + dv - (MU / radius).sqrt()).abs() < 1.0e-9);
    }

    #[test]
    fn raises_circular_orbit_with_two_validated_burns() {
        let initial_radius = RADIUS + 200_000.0;
        let request = OrbitPlanRequest {
            state: state_at_periapsis(initial_radius, initial_radius, 0.0),
            epoch_s: 1_000.0,
            mu: MU,
            body_radius_m: RADIUS,
            target: TargetOrbit::circular(0, 500_000.0),
        };
        let plan = plan_target_orbit(request).expect("plan");
        assert_eq!(plan.nodes.len(), 2);
        assert!(plan.total_delta_v_m_s > 0.0);
        assert!((plan.predicted_elements.periapsis_m - (RADIUS + 500_000.0)).abs() < 1.0);
        assert!((plan.predicted_elements.apoapsis_m - (RADIUS + 500_000.0)).abs() < 1.0);
        assert!(plan.nodes[0].time - request.epoch_s >= MIN_FIRST_BURN_LEAD_S);
    }

    #[test]
    fn targets_elliptical_apsides() {
        let initial_radius = RADIUS + 250_000.0;
        let request = OrbitPlanRequest {
            state: state_at_periapsis(initial_radius, initial_radius, 0.0),
            epoch_s: 0.0,
            mu: MU,
            body_radius_m: RADIUS,
            target: TargetOrbit {
                reference_body: 0,
                periapsis_altitude_m: 400_000.0,
                apoapsis_altitude_m: 900_000.0,
                plane: TargetPlane::PreserveCurrent,
            },
        };
        let plan = plan_target_orbit(request).expect("plan");
        assert!((plan.predicted_elements.periapsis_m - (RADIUS + 400_000.0)).abs() < 1.0);
        assert!((plan.predicted_elements.apoapsis_m - (RADIUS + 900_000.0)).abs() < 1.0);
    }

    #[test]
    fn changes_to_requested_nearest_inclination() {
        let radius = RADIUS + 300_000.0;
        let request = OrbitPlanRequest {
            state: state_at_periapsis(radius, radius, 0.1),
            epoch_s: 0.0,
            mu: MU,
            body_radius_m: RADIUS,
            target: TargetOrbit {
                reference_body: 0,
                periapsis_altitude_m: 500_000.0,
                apoapsis_altitude_m: 500_000.0,
                plane: TargetPlane::Nearest {
                    inclination_rad: 0.6,
                    direction: OrbitDirection::Prograde,
                },
            },
        };
        let plan = plan_target_orbit(request).expect("plan");
        assert_eq!(plan.nodes.len(), 3);
        assert!((plan.predicted_elements.inclination_rad - 0.6).abs() < 1.0e-6);
    }

    #[test]
    fn rejects_body_intersection_and_fixed_plane() {
        let radius = RADIUS + 200_000.0;
        let mut request = OrbitPlanRequest {
            state: state_at_periapsis(radius, radius, 0.0),
            epoch_s: 0.0,
            mu: MU,
            body_radius_m: RADIUS,
            target: TargetOrbit::circular(0, 0.0),
        };
        assert!(matches!(
            plan_target_orbit(request),
            Err(OrbitPlanError::TargetIntersectsBody)
        ));
        request.target = TargetOrbit {
            reference_body: 0,
            periapsis_altitude_m: 200_000.0,
            apoapsis_altitude_m: 200_000.0,
            plane: TargetPlane::Fixed {
                inclination_rad: 0.0,
                ascending_node_longitude_rad: 0.0,
                direction: OrbitDirection::Prograde,
            },
        };
        assert!(matches!(
            plan_target_orbit(request),
            Err(OrbitPlanError::FixedPlaneUnsupported)
        ));
    }
}
