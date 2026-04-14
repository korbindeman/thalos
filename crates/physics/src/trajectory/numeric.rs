//! `NumericSegment`: a discrete-sample trajectory leg with cubic Hermite
//! interpolation.
//!
//! The integrator produces one `TrajectorySample` per step.  Because each
//! sample carries both position and velocity, we can fit a C¹-continuous
//! cubic Hermite between any two consecutive samples and read back states at
//! arbitrary times — no re-integration needed.
//!
//! `NumericSegment` also caches termination metadata (stable orbit, collision)
//! that the renderer uses to draw closed ellipses or collision markers.

use glam::DVec3;

use super::Trajectory;
use crate::types::{BodyId, StateVector, TrajectorySample};

/// Proxy for trajectory cone width at a sample point (metres).
///
/// Uses perturbation ratio scaled by *inverse* step size: smaller accepted
/// adaptive steps mean the integrator found the region numerically sensitive,
/// so uncertainty grows as `step_size` shrinks.
pub fn cone_width(sample: &TrajectorySample) -> f64 {
    const NOMINAL_STEP_SIZE: f64 = 60.0;
    const NOMINAL_CONE_WIDTH_M: f64 = 1.0e6;

    let step_factor = NOMINAL_STEP_SIZE / sample.step_size.max(1e-3);
    sample.perturbation_ratio * step_factor * NOMINAL_CONE_WIDTH_M
}

/// One propagated leg of a [`FlightPlan`](super::FlightPlan).
///
/// Samples are stored in strictly increasing `time` order.  Termination
/// metadata (`is_stable_orbit`, `collision_body`, `stable_orbit_start_index`)
/// travels with the segment so the renderer can draw closed ellipses or
/// collision markers without re-deriving the information.
#[derive(Debug, Clone)]
pub struct NumericSegment {
    pub samples: Vec<TrajectorySample>,
    /// Set to true if a full revolution around the dominant body was detected.
    pub is_stable_orbit: bool,
    /// First sample belonging to the closed-loop portion of a stable orbit.
    pub stable_orbit_start_index: Option<usize>,
    /// Body ID if the ship dropped below a body's surface during this leg.
    pub collision_body: Option<BodyId>,
}

impl NumericSegment {
    pub fn empty() -> Self {
        Self {
            samples: Vec::new(),
            is_stable_orbit: false,
            stable_orbit_start_index: None,
            collision_body: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn start_time(&self) -> Option<f64> {
        self.samples.first().map(|s| s.time)
    }

    pub fn end_time(&self) -> Option<f64> {
        self.samples.last().map(|s| s.time)
    }

    pub fn last_state(&self) -> Option<StateVector> {
        self.samples.last().map(|s| StateVector {
            position: s.position,
            velocity: s.velocity,
        })
    }

    /// Find index `i` such that `samples[i].time <= time <= samples[i+1].time`.
    /// Returns `None` if `time` is outside the segment or fewer than 2 samples.
    fn bracket_index(&self, time: f64) -> Option<usize> {
        let n = self.samples.len();
        if n < 2 {
            return None;
        }
        let start = self.samples[0].time;
        let end = self.samples[n - 1].time;
        if time < start || time > end {
            return None;
        }
        // Binary search for first sample with time > query.  `i-1` then
        // brackets the query on the left.
        let mut lo = 0usize;
        let mut hi = n - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.samples[mid].time <= time {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(lo)
    }
}

impl Trajectory for NumericSegment {
    fn state_at(&self, time: f64) -> Option<StateVector> {
        let n = self.samples.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            let s = &self.samples[0];
            if (time - s.time).abs() < 1e-6 {
                return Some(StateVector {
                    position: s.position,
                    velocity: s.velocity,
                });
            }
            return None;
        }

        let i = self.bracket_index(time)?;
        let a = &self.samples[i];
        let b = &self.samples[i + 1];
        let h = b.time - a.time;
        if h <= 0.0 {
            return Some(StateVector {
                position: a.position,
                velocity: a.velocity,
            });
        }

        // Normalized parameter tau in [0, 1].
        let tau = ((time - a.time) / h).clamp(0.0, 1.0);
        let tau2 = tau * tau;
        let tau3 = tau2 * tau;

        // Hermite basis functions on tau.
        let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
        let h10 = tau3 - 2.0 * tau2 + tau;
        let h01 = -2.0 * tau3 + 3.0 * tau2;
        let h11 = tau3 - tau2;

        // Tangents in tau-space: m_i = v_i * h (since d tau / d time = 1/h).
        let position =
            a.position * h00 + a.velocity * (h10 * h) + b.position * h01 + b.velocity * (h11 * h);

        // Derivatives of the basis wrt tau.
        let dh00 = 6.0 * tau2 - 6.0 * tau;
        let dh10 = 3.0 * tau2 - 4.0 * tau + 1.0;
        let dh01 = -6.0 * tau2 + 6.0 * tau;
        let dh11 = 3.0 * tau2 - 2.0 * tau;

        // dp/dtime = (dp/dtau) / h.
        let velocity = a.position * (dh00 / h)
            + a.velocity * dh10
            + b.position * (dh01 / h)
            + b.velocity * dh11;

        Some(StateVector { position, velocity })
    }

    fn epoch_range(&self) -> (f64, f64) {
        let start = self.start_time().unwrap_or(0.0);
        let end = self.end_time().unwrap_or(0.0);
        (start, end)
    }

    fn anchor_body_at(&self, time: f64) -> Option<BodyId> {
        let n = self.samples.len();
        if n == 0 {
            return None;
        }
        // Pick the sample closest in time.
        let i = match self.samples.binary_search_by(|s| {
            s.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) if i >= n => n - 1,
            Err(i) => {
                let lo = &self.samples[i - 1];
                let hi = &self.samples[i];
                if (time - lo.time).abs() <= (hi.time - time).abs() {
                    i - 1
                } else {
                    i
                }
            }
        };
        Some(self.samples[i].anchor_body)
    }
}

/// Helper used by flight-plan rendering: body-relative position preserved
/// through interpolation, so callers can query at any time without tearing.
impl NumericSegment {
    pub fn relative_position_at(
        &self,
        time: f64,
        body_states: &[crate::types::BodyState],
    ) -> Option<DVec3> {
        let state = self.state_at(time)?;
        let body = self.anchor_body_at(time)?;
        let body_pos = body_states.get(body)?.position;
        Some(state.position - body_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(time: f64, pos: DVec3, vel: DVec3) -> TrajectorySample {
        TrajectorySample {
            time,
            position: pos,
            velocity: vel,
            dominant_body: 0,
            perturbation_ratio: 0.0,
            step_size: 1.0,
            anchor_body: 0,
            anchor_body_pos: DVec3::ZERO,
        }
    }

    #[test]
    fn hermite_interpolates_straight_line() {
        let seg = NumericSegment {
            samples: vec![
                sample(0.0, DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)),
                sample(10.0, DVec3::new(10.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
            ],
            is_stable_orbit: false,
            stable_orbit_start_index: None,
            collision_body: None,
        };
        let mid = seg.state_at(5.0).unwrap();
        assert!((mid.position.x - 5.0).abs() < 1e-9);
        assert!((mid.velocity.x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hermite_endpoints_exact() {
        let a = sample(0.0, DVec3::ZERO, DVec3::new(1.0, 2.0, 3.0));
        let b = sample(1.0, DVec3::new(4.0, 5.0, 6.0), DVec3::new(7.0, 8.0, 9.0));
        let seg = NumericSegment {
            samples: vec![a, b],
            is_stable_orbit: false,
            stable_orbit_start_index: None,
            collision_body: None,
        };
        let s0 = seg.state_at(0.0).unwrap();
        assert!((s0.position - a.position).length() < 1e-9);
        assert!((s0.velocity - a.velocity).length() < 1e-9);
        let s1 = seg.state_at(1.0).unwrap();
        assert!((s1.position - b.position).length() < 1e-9);
        assert!((s1.velocity - b.velocity).length() < 1e-9);
    }

    #[test]
    fn out_of_range_returns_none() {
        let seg = NumericSegment {
            samples: vec![
                sample(0.0, DVec3::ZERO, DVec3::ZERO),
                sample(1.0, DVec3::ZERO, DVec3::ZERO),
            ],
            is_stable_orbit: false,
            stable_orbit_start_index: None,
            collision_body: None,
        };
        assert!(seg.state_at(-0.1).is_none());
        assert!(seg.state_at(1.1).is_none());
    }
}
