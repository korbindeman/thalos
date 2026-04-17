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

/// Compute burn duration given delta-v magnitude and thrust acceleration.
pub fn burn_duration(delta_v_magnitude: f64, thrust_acceleration: f64) -> f64 {
    if thrust_acceleration <= 0.0 {
        return 0.0;
    }
    delta_v_magnitude / thrust_acceleration
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
    fn test_burn_duration_basic() {
        assert!((burn_duration(100.0, 10.0) - 10.0).abs() < f64::EPSILON);
        assert!((burn_duration(0.0, 10.0)).abs() < f64::EPSILON);
        assert!((burn_duration(100.0, 0.0)).abs() < f64::EPSILON);
    }

}
