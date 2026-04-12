use std::f64::consts::{PI, TAU};

use glam::DVec3;

use crate::body_state_provider::BodyStateProvider;
use crate::types::{
    BodyId, BodyState, BodyStates, OrbitalElements, SolarSystemDefinition, StateVector,
    orbital_elements_to_cartesian,
};

#[derive(Debug, Clone, Copy)]
struct FixedRelativeState {
    parent_id: BodyId,
    state: StateVector,
}

#[derive(Debug, Clone, Copy)]
struct KeplerianOrbit {
    parent_id: BodyId,
    semi_major_axis_m: f64,
    eccentricity: f64,
    sqrt_one_minus_e_sq: f64,
    epoch_mean_anomaly_rad: f64,
    mean_motion_rad_per_s: f64,
    period_s: f64,
    basis_p: DVec3,
    basis_q: DVec3,
}

#[derive(Debug, Clone, Copy)]
enum BodyMotion {
    Static,
    FixedRelative(FixedRelativeState),
    Keplerian(KeplerianOrbit),
}

#[derive(Debug, Clone, Copy)]
struct PatchedConicBody {
    parent_id: Option<BodyId>,
    mass_kg: f64,
    motion: BodyMotion,
}

/// Deterministic patched-conics body motion provider.
///
/// Each non-root body follows fixed Keplerian elements around its parent.
/// Queries are evaluated directly at the requested time without precomputing or
/// loading an ephemeris.
pub struct PatchedConics {
    bodies: Vec<PatchedConicBody>,
    eval_order: Vec<BodyId>,
    time_span: f64,
}

impl PatchedConics {
    pub fn new(system: &SolarSystemDefinition, time_span: f64) -> Self {
        let bodies = system
            .bodies
            .iter()
            .map(|body| {
                let motion = match (body.parent, body.orbital_elements) {
                    (None, _) => BodyMotion::Static,
                    (Some(parent_id), Some(elements))
                        if is_supported_keplerian(&elements)
                            && system.bodies[parent_id].gm > 0.0 =>
                    {
                        BodyMotion::Keplerian(build_keplerian_orbit(
                            parent_id,
                            &elements,
                            system.bodies[parent_id].gm,
                        ))
                    }
                    (Some(parent_id), Some(elements)) => {
                        let state =
                            orbital_elements_to_cartesian(&elements, system.bodies[parent_id].gm);
                        BodyMotion::FixedRelative(FixedRelativeState { parent_id, state })
                    }
                    (Some(parent_id), None) => BodyMotion::FixedRelative(FixedRelativeState {
                        parent_id,
                        state: StateVector {
                            position: DVec3::ZERO,
                            velocity: DVec3::ZERO,
                        },
                    }),
                };

                PatchedConicBody {
                    parent_id: body.parent,
                    mass_kg: body.mass_kg,
                    motion,
                }
            })
            .collect();

        Self {
            eval_order: compute_eval_order(system),
            bodies,
            time_span,
        }
    }

    fn clamp_time(&self, time: f64) -> f64 {
        if time < 0.0 || time > self.time_span {
            #[cfg(debug_assertions)]
            eprintln!(
                "[patched_conics] query time {:.0}s clamped to [0, {:.0}]",
                time, self.time_span
            );
            time.clamp(0.0, self.time_span)
        } else {
            time
        }
    }
}

impl BodyStateProvider for PatchedConics {
    fn query_into(&self, time: f64, out: &mut BodyStates) {
        let t = self.clamp_time(time);
        if out.len() != self.bodies.len() {
            out.resize(
                self.bodies.len(),
                BodyState {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                    mass_kg: 0.0,
                },
            );
        }

        for &body_id in &self.eval_order {
            let body = self.bodies[body_id];
            out[body_id] = match body.motion {
                BodyMotion::Static => BodyState {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                    mass_kg: body.mass_kg,
                },
                BodyMotion::FixedRelative(fixed) => {
                    let parent = out[fixed.parent_id];
                    BodyState {
                        position: parent.position + fixed.state.position,
                        velocity: parent.velocity + fixed.state.velocity,
                        mass_kg: body.mass_kg,
                    }
                }
                BodyMotion::Keplerian(orbit) => {
                    let parent = out[orbit.parent_id];
                    let relative = keplerian_relative_state(&orbit, t);
                    BodyState {
                        position: parent.position + relative.position,
                        velocity: parent.velocity + relative.velocity,
                        mass_kg: body.mass_kg,
                    }
                }
            };
        }
    }

    fn query_body(&self, body_id: BodyId, time: f64) -> BodyState {
        let t = self.clamp_time(time);

        let mut lineage = [usize::MAX; 16];
        let mut len = 0usize;
        let mut current = Some(body_id);
        let mut fallback: Vec<BodyId> = Vec::new();

        while let Some(id) = current {
            if len < lineage.len() {
                lineage[len] = id;
            } else {
                if fallback.is_empty() {
                    fallback.extend_from_slice(&lineage[..len]);
                }
                fallback.push(id);
            }
            len += 1;
            current = self.bodies[id].parent_id;
        }

        let mut position = DVec3::ZERO;
        let mut velocity = DVec3::ZERO;
        let mass_kg = self.bodies[body_id].mass_kg;

        if fallback.is_empty() {
            for &id in lineage[..len].iter().rev() {
                match self.bodies[id].motion {
                    BodyMotion::Static => {}
                    BodyMotion::FixedRelative(fixed) => {
                        position += fixed.state.position;
                        velocity += fixed.state.velocity;
                    }
                    BodyMotion::Keplerian(orbit) => {
                        let relative = keplerian_relative_state(&orbit, t);
                        position += relative.position;
                        velocity += relative.velocity;
                    }
                }
            }
        } else {
            for &id in fallback.iter().rev() {
                match self.bodies[id].motion {
                    BodyMotion::Static => {}
                    BodyMotion::FixedRelative(fixed) => {
                        position += fixed.state.position;
                        velocity += fixed.state.velocity;
                    }
                    BodyMotion::Keplerian(orbit) => {
                        let relative = keplerian_relative_state(&orbit, t);
                        position += relative.position;
                        velocity += relative.velocity;
                    }
                }
            }
        }

        BodyState {
            position,
            velocity,
            mass_kg,
        }
    }

    fn body_count(&self) -> usize {
        self.bodies.len()
    }

    fn time_span(&self) -> f64 {
        self.time_span
    }

    #[allow(clippy::only_used_in_recursion)] // Not recursion — calls the trait default impl.
    fn detect_period(&self, body_id: BodyId, parent_id: BodyId, start_time: f64) -> f64 {
        match self.bodies[body_id].motion {
            BodyMotion::Keplerian(orbit)
                if orbit.parent_id == parent_id && orbit.period_s > 0.0 =>
            {
                orbit.period_s
            }
            // Fall through to the sampling-based default impl on the trait.
            _ => BodyStateProvider::detect_period(self, body_id, parent_id, start_time),
        }
    }

    fn body_orbit_trail(
        &self,
        body_id: BodyId,
        parent_id: BodyId,
        start_time: f64,
        num_samples: usize,
    ) -> Vec<DVec3> {
        let t0 = self.clamp_time(start_time);

        match self.bodies[body_id].motion {
            BodyMotion::Keplerian(orbit) if orbit.parent_id == parent_id => {
                let span = orbit.period_s.min((self.time_span - t0).max(0.0));
                if num_samples == 0 {
                    return vec![keplerian_relative_state(&orbit, t0).position];
                }

                (0..=num_samples)
                    .map(|i| {
                        let t = t0 + (i as f64 / num_samples as f64) * span;
                        keplerian_relative_state(&orbit, t).position
                    })
                    .collect()
            }
            BodyMotion::FixedRelative(fixed) if fixed.parent_id == parent_id => {
                vec![fixed.state.position; num_samples.saturating_add(1).max(1)]
            }
            // Fall through to the sampling-based default impl on the trait.
            _ => BodyStateProvider::body_orbit_trail(self, body_id, parent_id, t0, num_samples),
        }
    }
}

fn compute_eval_order(system: &SolarSystemDefinition) -> Vec<BodyId> {
    fn depth_of(
        body_id: BodyId,
        bodies: &[crate::types::BodyDefinition],
        cache: &mut [Option<usize>],
    ) -> usize {
        if let Some(depth) = cache[body_id] {
            return depth;
        }

        let depth = match bodies[body_id].parent {
            Some(parent_id) => 1 + depth_of(parent_id, bodies, cache),
            None => 0,
        };
        cache[body_id] = Some(depth);
        depth
    }

    let mut cache = vec![None; system.bodies.len()];
    let mut order: Vec<BodyId> = (0..system.bodies.len()).collect();
    order.sort_by_key(|&body_id| depth_of(body_id, &system.bodies, &mut cache));
    order
}

fn is_supported_keplerian(elements: &OrbitalElements) -> bool {
    elements.semi_major_axis_m > 0.0 && elements.eccentricity >= 0.0 && elements.eccentricity < 1.0
}

fn build_keplerian_orbit(
    parent_id: BodyId,
    elements: &OrbitalElements,
    parent_gm: f64,
) -> KeplerianOrbit {
    let (basis_p, basis_q) = keplerian_basis(elements);
    let mean_motion_rad_per_s = mean_motion_rad_per_s(elements.semi_major_axis_m, parent_gm);
    let period_s = if mean_motion_rad_per_s > 0.0 {
        TAU / mean_motion_rad_per_s
    } else {
        0.0
    };
    KeplerianOrbit {
        parent_id,
        semi_major_axis_m: elements.semi_major_axis_m,
        eccentricity: elements.eccentricity,
        sqrt_one_minus_e_sq: (1.0 - elements.eccentricity * elements.eccentricity)
            .max(0.0)
            .sqrt(),
        epoch_mean_anomaly_rad: mean_anomaly_from_true_anomaly(
            elements.eccentricity,
            elements.true_anomaly_rad,
        ),
        mean_motion_rad_per_s,
        period_s,
        basis_p,
        basis_q,
    }
}

fn keplerian_basis(elements: &OrbitalElements) -> (DVec3, DVec3) {
    let cos_o = elements.lon_ascending_node_rad.cos();
    let sin_o = elements.lon_ascending_node_rad.sin();
    let cos_w = elements.arg_periapsis_rad.cos();
    let sin_w = elements.arg_periapsis_rad.sin();
    let cos_i = elements.inclination_rad.cos();
    let sin_i = elements.inclination_rad.sin();

    let basis_p = DVec3::new(
        cos_o * cos_w - sin_o * sin_w * cos_i,
        sin_i * sin_w,
        sin_o * cos_w + cos_o * sin_w * cos_i,
    );
    let basis_q = DVec3::new(
        -cos_o * sin_w - sin_o * cos_w * cos_i,
        sin_i * cos_w,
        -sin_o * sin_w + cos_o * cos_w * cos_i,
    );

    (basis_p, basis_q)
}

fn mean_motion_rad_per_s(semi_major_axis_m: f64, parent_gm: f64) -> f64 {
    if semi_major_axis_m <= 0.0 || parent_gm <= 0.0 {
        0.0
    } else {
        (parent_gm / semi_major_axis_m.powi(3)).sqrt()
    }
}

fn mean_anomaly_from_true_anomaly(eccentricity: f64, true_anomaly_rad: f64) -> f64 {
    if eccentricity.abs() < 1e-12 {
        return true_anomaly_rad.rem_euclid(TAU);
    }

    let cos_nu = true_anomaly_rad.cos();
    let sin_nu = true_anomaly_rad.sin();
    let denom = 1.0 + eccentricity * cos_nu;
    let cos_e = (eccentricity + cos_nu) / denom;
    let sin_e = ((1.0 - eccentricity * eccentricity).max(0.0).sqrt() * sin_nu) / denom;
    let eccentric_anomaly = sin_e.atan2(cos_e);
    (eccentric_anomaly - eccentricity * sin_e).rem_euclid(TAU)
}

fn solve_kepler_equation(eccentricity: f64, mean_anomaly_rad: f64) -> f64 {
    let mean = (mean_anomaly_rad + PI).rem_euclid(TAU) - PI;
    let mut eccentric_anomaly = if eccentricity < 0.8 {
        mean
    } else if mean >= 0.0 {
        PI
    } else {
        -PI
    };

    for _ in 0..16 {
        let sin_e = eccentric_anomaly.sin();
        let cos_e = eccentric_anomaly.cos();
        let residual = eccentric_anomaly - eccentricity * sin_e - mean;
        let slope = 1.0 - eccentricity * cos_e;
        if slope.abs() < 1e-12 {
            break;
        }
        let delta = residual / slope;
        eccentric_anomaly -= delta;
        if delta.abs() < 1e-13 {
            break;
        }
    }

    eccentric_anomaly
}

fn keplerian_relative_state(orbit: &KeplerianOrbit, time: f64) -> StateVector {
    let mean_anomaly =
        (orbit.epoch_mean_anomaly_rad + orbit.mean_motion_rad_per_s * time).rem_euclid(TAU);
    let eccentric_anomaly = solve_kepler_equation(orbit.eccentricity, mean_anomaly);
    let cos_e = eccentric_anomaly.cos();
    let sin_e = eccentric_anomaly.sin();
    let denom = 1.0 - orbit.eccentricity * cos_e;

    let x = orbit.semi_major_axis_m * (cos_e - orbit.eccentricity);
    let y = orbit.semi_major_axis_m * orbit.sqrt_one_minus_e_sq * sin_e;

    let factor = if denom.abs() > 1e-12 {
        orbit.mean_motion_rad_per_s / denom
    } else {
        0.0
    };
    let vx = -orbit.semi_major_axis_m * sin_e * factor;
    let vy = orbit.semi_major_axis_m * orbit.sqrt_one_minus_e_sq * cos_e * factor;

    StateVector {
        position: orbit.basis_p * x + orbit.basis_q * y,
        velocity: orbit.basis_p * vx + orbit.basis_q * vy,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::types::{BodyDefinition, BodyKind, G, ShipDefinition};

    const AU: f64 = 1.496e11;
    const SUN_GM: f64 = 1.327_124_4e20;
    const JULIAN_YEAR: f64 = 3.156e7;

    fn make_two_body_system() -> SolarSystemDefinition {
        let sun_mass = SUN_GM / G;
        let sun = BodyDefinition {
            id: 0,
            name: "Sun".to_string(),
            kind: BodyKind::Star,
            parent: None,
            mass_kg: sun_mass,
            radius_m: 6.957e8,
            color: [1.0, 1.0, 0.0],
            albedo: 1.0,
            rotation_period_s: 0.0,
            axial_tilt_rad: 0.0,
            gm: SUN_GM,
            orbital_elements: None,
            generator: None,
        };

        let earth = BodyDefinition {
            id: 1,
            name: "Earth".to_string(),
            kind: BodyKind::Planet,
            parent: Some(0),
            mass_kg: 5.972e24,
            radius_m: 6.371e6,
            color: [0.0, 0.5, 1.0],
            albedo: 0.3,
            rotation_period_s: 86_400.0,
            axial_tilt_rad: 0.0,
            gm: G * 5.972e24,
            orbital_elements: Some(OrbitalElements {
                semi_major_axis_m: AU,
                eccentricity: 0.0,
                inclination_rad: 0.0,
                lon_ascending_node_rad: 0.0,
                arg_periapsis_rad: 0.0,
                true_anomaly_rad: 0.0,
            }),
            generator: None,
        };

        let mut name_to_id = HashMap::new();
        name_to_id.insert("Sun".to_string(), 0);
        name_to_id.insert("Earth".to_string(), 1);

        SolarSystemDefinition {
            name: "Test".to_string(),
            bodies: vec![sun, earth],
            ship: ShipDefinition {
                initial_state: StateVector {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                },
                thrust_acceleration: 0.5,
            },
            name_to_id,
        }
    }

    #[test]
    fn query_body_matches_query_into() {
        let system = make_two_body_system();
        let provider = PatchedConics::new(&system, 2.0 * JULIAN_YEAR);
        let mut all_states = Vec::new();
        provider.query_into(0.25 * JULIAN_YEAR, &mut all_states);
        let earth = provider.query_body(1, 0.25 * JULIAN_YEAR);

        assert!((earth.position - all_states[1].position).length() < 1e-6);
        assert!((earth.velocity - all_states[1].velocity).length() < 1e-9);
    }

    #[test]
    fn orbit_trail_uses_analytic_relative_motion() {
        let system = make_two_body_system();
        let provider = PatchedConics::new(&system, 2.0 * JULIAN_YEAR);
        let trail = provider.body_orbit_trail(1, 0, 0.0, 16);
        assert_eq!(trail.len(), 17);
        for point in trail {
            let rel_error = (point.length() - AU).abs() / AU;
            assert!(rel_error < 1e-10, "relative radius error: {rel_error}");
        }
    }
}
