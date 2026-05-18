use std::f64::consts::TAU;

use glam::DVec3;

use crate::canonical::Epoch;
use crate::types::{BodyId, BodyState, BodyStates};

/// Shared interface for deterministic body-state sources.
///
/// Runtime systems use this trait so they can switch between patched conics,
/// precomputed ephemerides, or future N-body sources without changing call
/// sites.
pub trait BodyTrajectoryProvider: Send + Sync {
    /// Fill `out` with all body states at `epoch`.
    ///
    /// Implementations should reuse `out` when possible to avoid per-query
    /// allocations on hot paths.
    fn states_into(&self, epoch: Epoch, out: &mut BodyStates);

    /// Return the state of one body at `epoch`.
    fn state(&self, body_id: BodyId, epoch: Epoch) -> BodyState;

    /// Number of bodies exposed by this provider.
    fn body_count(&self) -> usize;

    /// Maximum supported time span, in seconds.
    fn time_span(&self) -> f64;

    /// Convenience wrapper that allocates a `Vec` for the result.
    ///
    /// Hot paths should call [`Self::states_into`] with a reused buffer
    /// instead — every call to this method allocates fresh, which adds up
    /// fast in the per-sample event-detection scans.
    fn states(&self, epoch: Epoch) -> BodyStates {
        let mut out = Vec::with_capacity(self.body_count());
        self.states_into(epoch, &mut out);
        out
    }

    /// Detect the orbital period of `body_id` relative to `parent_id`.
    ///
    /// The default implementation samples queries and works for any provider.
    fn detect_period(&self, body_id: BodyId, parent_id: BodyId, start_time: f64) -> f64 {
        let rel_pos_at = |t: f64| -> DVec3 {
            let body_state = self.state(body_id, Epoch(t));
            let parent_state = self.state(parent_id, Epoch(t));
            body_state.position - parent_state.position
        };

        let start_pos = rel_pos_at(start_time);
        let start_vel = {
            let body_vel = self.state(body_id, Epoch(start_time)).velocity;
            let parent_vel = self.state(parent_id, Epoch(start_time)).velocity;
            body_vel - parent_vel
        };
        let remaining = (self.time_span() - start_time).max(0.0);

        let orbit_radius = start_pos.length();
        let orbital_speed = start_vel.length();
        let approx_period = if orbital_speed > 1e-12 {
            TAU * orbit_radius / orbital_speed
        } else {
            return 0.0;
        };

        let scan_end = (approx_period * 2.0).min(remaining);
        if scan_end <= 0.0 {
            return 0.0;
        }

        let steps = 1000usize;
        let dt = scan_end / steps as f64;
        let mut cumulative_angle = 0.0;
        let mut prev_pos = start_pos;

        for i in 1..=steps {
            let t = start_time + dt * i as f64;
            let pos = rel_pos_at(t);

            let prev_len = prev_pos.length();
            let cur_len = pos.length();
            if prev_len > 1e-12 && cur_len > 1e-12 {
                let cos_angle = (prev_pos.dot(pos) / (prev_len * cur_len)).clamp(-1.0, 1.0);
                cumulative_angle += cos_angle.acos();
            }

            if cumulative_angle >= TAU {
                let last_step_angle = {
                    let prev_len = prev_pos.length();
                    let cur_len = pos.length();
                    if prev_len <= 1e-12 || cur_len <= 1e-12 {
                        0.0
                    } else {
                        (prev_pos.dot(pos) / (prev_len * cur_len))
                            .clamp(-1.0, 1.0)
                            .acos()
                    }
                };
                let overshoot = cumulative_angle - TAU;
                let frac = if last_step_angle > 1e-15 {
                    1.0 - overshoot / last_step_angle
                } else {
                    1.0
                };
                return dt * ((i - 1) as f64 + frac);
            }

            prev_pos = pos;
        }

        approx_period
    }

    /// Sample the forward orbit trail for `body_id` relative to `parent_id`.
    fn body_orbit_trail(
        &self,
        body_id: BodyId,
        parent_id: BodyId,
        start_time: f64,
        num_samples: usize,
    ) -> Vec<DVec3> {
        if num_samples == 0 {
            let body_state = self.state(body_id, Epoch(start_time));
            let parent_state = self.state(parent_id, Epoch(start_time));
            return vec![body_state.position - parent_state.position];
        }

        let period = self.detect_period(body_id, parent_id, start_time);
        let remaining = (self.time_span() - start_time).max(0.0);
        let span = period.min(remaining);

        (0..=num_samples)
            .map(|i| {
                let t = start_time + (i as f64 / num_samples as f64) * span;
                let body_state = self.state(body_id, Epoch(t));
                let parent_state = self.state(parent_id, Epoch(t));
                body_state.position - parent_state.position
            })
            .collect()
    }
}
