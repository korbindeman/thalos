use glam::DVec3;

use crate::types::BodyId;

/// A single impulsive maneuver: a delta-v applied at a scheduled time.
///
/// The delta-v is expressed in the local orbital frame relative to
/// `reference_body`:
/// - `x` — prograde (along velocity relative to reference body)
/// - `y` — normal (orbit-plane north)
/// - `z` — radial (away from reference body)
#[derive(Debug, Clone)]
pub struct ManeuverNode {
    /// Opaque tag set by the caller (e.g. UI node id). Physics doesn't
    /// interpret it — it's reported back when the node is consumed so the
    /// caller can reconcile UI state with physics execution.
    pub id: Option<u64>,
    /// Simulation time (seconds from epoch) at which the maneuver is executed.
    pub time: f64,
    /// Delta-v in the local prograde/normal/radial frame (m/s).
    pub delta_v: DVec3,
    /// The body whose frame is used for the local-to-world conversion.
    pub reference_body: BodyId,
}

/// An ordered collection of maneuver nodes, sorted by time.
#[derive(Debug, Clone, Default)]
pub struct ManeuverSequence {
    pub nodes: Vec<ManeuverNode>,
}

impl ManeuverSequence {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a node in sorted order by time.
    pub fn add(&mut self, node: ManeuverNode) {
        let pos = self
            .nodes
            .binary_search_by(|n| {
                n.time
                    .partial_cmp(&node.time)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|i| i);
        self.nodes.insert(pos, node);
    }

    pub fn remove(&mut self, index: usize) {
        self.nodes.remove(index);
    }

    pub fn get(&self, index: usize) -> Option<&ManeuverNode> {
        self.nodes.get(index)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ManeuverNode> {
        self.nodes.iter()
    }
}

/// Compute the prograde/normal/radial orbital frame axes.
///
/// Returns `[prograde, normal, radial]` as an orthonormal basis:
/// - prograde = normalized(ship_vel - body_vel)
/// - normal   = perpendicular to orbital plane (radial × prograde)
/// - radial   = recomputed as prograde × normal for orthogonality
pub fn orbital_frame(
    ship_position: DVec3,
    ship_velocity: DVec3,
    body_position: DVec3,
    body_velocity: DVec3,
) -> [DVec3; 3] {
    let rel_vel = ship_velocity - body_velocity;
    let rel_pos = ship_position - body_position;

    let prograde = if rel_vel.length_squared() > 1e-20 {
        rel_vel.normalize()
    } else {
        DVec3::X
    };

    let radial_raw = if rel_pos.length_squared() > 1e-20 {
        rel_pos.normalize()
    } else {
        DVec3::Y
    };

    let normal_raw = radial_raw.cross(prograde);
    let normal = if normal_raw.length_squared() > 1e-20 {
        normal_raw.normalize()
    } else {
        let arbitrary = if prograde.dot(DVec3::Y).abs() < 0.9 {
            DVec3::Y
        } else {
            DVec3::X
        };
        prograde.cross(arbitrary).normalize()
    };

    let radial = prograde.cross(normal);

    [prograde, normal, radial]
}

/// Convert delta-v from local orbital frame to world frame.
pub fn delta_v_to_world(
    delta_v: DVec3,
    ship_velocity: DVec3,
    ship_position: DVec3,
    body_position: DVec3,
    body_velocity: DVec3,
) -> DVec3 {
    let [prograde, normal, radial] =
        orbital_frame(ship_position, ship_velocity, body_position, body_velocity);
    delta_v.x * prograde + delta_v.y * normal + delta_v.z * radial
}

/// Burn time required to achieve `delta_v_magnitude` of Δv given constant
/// thrust and propellant mass flow, with mass decreasing as fuel burns
/// (Tsiolkovsky inverse). Capped at the time when propellant exhausts:
/// a burn that asks for more Δv than the ship can deliver returns the
/// time to drain the tanks, not the unachievable Tsiolkovsky time.
///
/// Derivation: under constant thrust F and mass-flow ṁ, exhaust velocity
/// `ve = F/ṁ`. Mass at time `t` after burn start is `m(t) = m0 - ṁ·t`.
/// Tsiolkovsky gives `Δv = ve · ln(m0 / m(t))`, so
///   `t = (m0 / ṁ) · (1 - exp(-Δv / ve))`.
/// Propellant exhausts when `m(t) = m_dry`, i.e. at
///   `t_max = (m0 - m_dry) / ṁ`.
///
/// Returns 0 for any degenerate input (no thrust, no flow, no mass, or
/// already at/below dry mass).
pub fn burn_duration(
    delta_v_magnitude: f64,
    thrust_n: f64,
    mass_kg: f64,
    mass_flow_kg_per_s: f64,
    dry_mass_kg: f64,
) -> f64 {
    if thrust_n <= 0.0 || mass_flow_kg_per_s <= 0.0 || mass_kg <= dry_mass_kg {
        return 0.0;
    }
    let ve = thrust_n / mass_flow_kg_per_s;
    if ve <= 0.0 {
        return 0.0;
    }
    let requested = (mass_kg / mass_flow_kg_per_s) * (1.0 - (-delta_v_magnitude / ve).exp());
    let max_until_dry = (mass_kg - dry_mass_kg) / mass_flow_kg_per_s;
    requested.min(max_until_dry).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_insertion() {
        let mut seq = ManeuverSequence::new();
        seq.add(ManeuverNode {
            id: None,
            time: 300.0,
            delta_v: DVec3::X,
            reference_body: 0,
        });
        seq.add(ManeuverNode {
            id: None,
            time: 100.0,
            delta_v: DVec3::Y,
            reference_body: 0,
        });
        seq.add(ManeuverNode {
            id: None,
            time: 200.0,
            delta_v: DVec3::Z,
            reference_body: 0,
        });

        assert_eq!(seq.len(), 3);
        assert!((seq.nodes[0].time - 100.0).abs() < f64::EPSILON);
        assert!((seq.nodes[1].time - 200.0).abs() < f64::EPSILON);
        assert!((seq.nodes[2].time - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_delta_v_prograde_only() {
        // Ship moving in +X relative to body → prograde = +X.
        let dv = delta_v_to_world(
            DVec3::new(10.0, 0.0, 0.0),   // 10 m/s prograde
            DVec3::new(1000.0, 0.0, 0.0), // ship velocity
            DVec3::new(1e8, 0.0, 0.0),    // ship position
            DVec3::ZERO,                  // body position
            DVec3::ZERO,                  // body velocity
        );
        assert!((dv.x - 10.0).abs() < 1e-10);
        assert!(dv.y.abs() < 1e-10);
        assert!(dv.z.abs() < 1e-10);
    }

    #[test]
    fn test_burn_duration_zero_dv_is_instant() {
        assert!(burn_duration(0.0, 250_000.0, 9000.0, 75.0, 3200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_burn_duration_degenerate_inputs_return_zero() {
        assert_eq!(burn_duration(100.0, 0.0, 9000.0, 75.0, 3200.0), 0.0);
        assert_eq!(burn_duration(100.0, 250_000.0, 9000.0, 0.0, 3200.0), 0.0);
        // Already at dry mass — no propellant left to burn.
        assert_eq!(burn_duration(100.0, 250_000.0, 3200.0, 75.0, 3200.0), 0.0);
        assert_eq!(burn_duration(100.0, 250_000.0, 100.0, 75.0, 3200.0), 0.0);
    }

    #[test]
    fn test_burn_duration_at_one_ve_matches_tsiolkovsky_inverse() {
        // At Δv = ve, expect t = (m0 / ṁ) · (1 - 1/e). Pick dry low enough
        // that propellant doesn't bind.
        let thrust = 250_000.0;
        let flow = 75.0;
        let m0 = 9000.0;
        let dry = 100.0;
        let ve = thrust / flow;
        let expected = (m0 / flow) * (1.0 - (-1.0_f64).exp());
        let got = burn_duration(ve, thrust, m0, flow, dry);
        assert!((got - expected).abs() < 1e-9, "got {got}, want {expected}");
    }

    #[test]
    fn test_burn_duration_small_dv_matches_constant_accel_limit() {
        // Δv ≪ ve → burn duration ≈ Δv / (F/m₀) (mass change negligible).
        let thrust = 250_000.0;
        let flow = 75.0;
        let m0 = 9000.0;
        let dry = 3200.0;
        let dv = 1.0; // very small
        let constant_accel_estimate = dv / (thrust / m0);
        let got = burn_duration(dv, thrust, m0, flow, dry);
        assert!(
            (got - constant_accel_estimate).abs() / constant_accel_estimate < 1e-3,
            "got {got}, want ≈ {constant_accel_estimate}"
        );
    }

    #[test]
    fn test_burn_duration_caps_at_propellant_exhaustion() {
        // Apollo-ish ship: 250 kN, ṁ = 75 kg/s, m0 = 9000, m_dry = 3200.
        // ve = 3333 m/s, Δv_max = ve · ln(9000/3200) ≈ 3447 m/s.
        // Asking for 10x that should clip to (m0 − m_dry)/ṁ = 5800/75 ≈ 77.33 s.
        let thrust = 250_000.0;
        let flow = 75.0;
        let m0 = 9000.0;
        let dry = 3200.0;
        let expected_max = (m0 - dry) / flow;
        let got = burn_duration(40_000.0, thrust, m0, flow, dry);
        assert!(
            (got - expected_max).abs() < 1e-9,
            "got {got}, want {expected_max} (propellant-limited)"
        );
    }
}
