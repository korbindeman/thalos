//! Ephemeris: precomputed N-body trajectories for all celestial bodies.
//!
//! # Design
//!
//! At construction time, the full N-body system is integrated forward using
//! RK4 with a fixed 1-hour timestep for the requested time span.  The star is
//! pinned to the origin.  All other bodies interact gravitationally with every
//! other body (including the star).
//!
//! Instead of storing every integration step, each body keeps an adaptively
//! sampled list of `EphemerisSample` records.  A new sample is committed
//! whenever the body's actual position deviates from a linear extrapolation of
//! the previous sample by more than `CURVATURE_THRESHOLD` metres.  This gives
//! dense coverage near periapsis and sparse coverage near apoapsis.
//!
//! Lookups use O(log n) binary search to bracket the query time, then cubic
//! Hermite interpolation using the bracketing samples' positions and velocities
//! to produce a C1-continuous result.

use crate::types::{
    orbital_elements_to_cartesian, BodyId, BodyState, BodyStates, SolarSystemDefinition,
};
use glam::DVec3;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Default precomputation time span: 100 Julian years in seconds.
pub const DEFAULT_TIME_SPAN: f64 = 3.156e9;

/// RK4 integration timestep (seconds).  1 hour is accurate enough for a
/// 25-body solar system and finishes precomputation well under a second.
pub(crate) const DT: f64 = 3600.0;

/// Adaptive-sampling threshold (metres).  When the deviation of the current
/// position from linear extrapolation exceeds this value a new sample is
/// stored.  1 km gives a modest sample count per orbit for most planets.
pub(crate) const CURVATURE_THRESHOLD: f64 = 1_000.0;

// ---------------------------------------------------------------------------
// Internal sample type
// ---------------------------------------------------------------------------

/// A single committed sample for one body.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct EphemerisSample {
    pub(crate) time: f64,
    pub(crate) position: DVec3,
    pub(crate) velocity: DVec3,
}

// ---------------------------------------------------------------------------
// Per-body track
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BodyTrack {
    pub(crate) mass_kg: f64,
    pub(crate) samples: Vec<EphemerisSample>,
}

impl BodyTrack {
    /// Return the state at the given time using cubic Hermite interpolation.
    /// Times outside [t0, t_last] are clamped to the nearest endpoint.
    fn query(&self, time: f64) -> (DVec3, DVec3) {
        let samples = &self.samples;

        // Edge cases: empty or single sample.
        if samples.is_empty() {
            return (DVec3::ZERO, DVec3::ZERO);
        }
        if samples.len() == 1 {
            return (samples[0].position, samples[0].velocity);
        }

        // Clamp to recorded range.
        let t = time.clamp(samples.first().unwrap().time, samples.last().unwrap().time);

        // Binary search for the right bracket: find the largest index whose
        // time <= t.
        let idx = match samples.binary_search_by(|s| s.time.partial_cmp(&t).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };

        // Ensure we have a valid right neighbour.
        let i0 = idx.min(samples.len() - 2);
        let i1 = i0 + 1;

        let s0 = &samples[i0];
        let s1 = &samples[i1];

        hermite_interp(
            s0.time, s0.position, s0.velocity, s1.time, s1.position, s1.velocity, t,
        )
    }
}

// ---------------------------------------------------------------------------
// Cubic Hermite interpolation
// ---------------------------------------------------------------------------

/// Cubic Hermite interpolation between two samples with known derivatives.
///
/// Returns `(position, velocity)` at time `t` within `[t0, t1]`.
#[inline]
fn hermite_interp(
    t0: f64,
    p0: DVec3,
    v0: DVec3,
    t1: f64,
    p1: DVec3,
    v1: DVec3,
    t: f64,
) -> (DVec3, DVec3) {
    let dt = t1 - t0;
    // Guard against degenerate intervals (shouldn't happen in practice).
    if dt.abs() < 1e-12 {
        return (p0, v0);
    }

    let u = (t - t0) / dt;
    let u2 = u * u;
    let u3 = u2 * u;

    // Hermite basis polynomials.
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;

    let position = h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1;

    // Derivatives of the Hermite basis (for velocity reconstruction).
    let dh00 = (6.0 * u2 - 6.0 * u) / dt;
    let dh10 = (3.0 * u2 - 4.0 * u + 1.0) / dt;
    let dh01 = (-6.0 * u2 + 6.0 * u) / dt;
    let dh11 = (3.0 * u2 - 2.0 * u) / dt;

    let velocity = dh00 * p0 + dh10 * dt * v0 + dh01 * p1 + dh11 * dt * v1;

    (position, velocity)
}

// ---------------------------------------------------------------------------
// N-body acceleration
// ---------------------------------------------------------------------------

/// Compute the gravitational acceleration on body `i` due to all other bodies.
///
/// `positions` and `gms` must be indexed by `BodyId`.
#[inline]
fn acceleration(i: usize, positions: &[DVec3], gms: &[f64]) -> DVec3 {
    let mut acc = DVec3::ZERO;
    let ri = positions[i];
    for (j, &rj) in positions.iter().enumerate() {
        if j == i {
            continue;
        }
        let delta = rj - ri;
        let dist2 = delta.length_squared();
        if dist2 < 1e6 {
            // Bodies at effectively the same point — skip to avoid singularity.
            continue;
        }
        let dist = dist2.sqrt();
        acc += (gms[j] / (dist2 * dist)) * delta;
    }
    acc
}

// ---------------------------------------------------------------------------
// RK4 step over the full N-body state
// ---------------------------------------------------------------------------

/// The full mutable integration state: positions + velocities for every body.
pub(crate) struct NBodyState {
    pub(crate) positions: Vec<DVec3>,
    pub(crate) velocities: Vec<DVec3>,
}

/// Pre-allocated scratch buffers for zero-allocation RK4 stepping.
pub(crate) struct Rk4Scratch {
    // k-stage velocities and accelerations.
    k1_v: Vec<DVec3>,
    k1_a: Vec<DVec3>,
    k2_v: Vec<DVec3>,
    k2_a: Vec<DVec3>,
    k3_v: Vec<DVec3>,
    k3_a: Vec<DVec3>,
    k4_v: Vec<DVec3>,
    k4_a: Vec<DVec3>,
    // Temporary position buffer for evaluating accelerations at mid/endpoints.
    tmp_pos: Vec<DVec3>,
}

impl Rk4Scratch {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            k1_v: vec![DVec3::ZERO; n],
            k1_a: vec![DVec3::ZERO; n],
            k2_v: vec![DVec3::ZERO; n],
            k2_a: vec![DVec3::ZERO; n],
            k3_v: vec![DVec3::ZERO; n],
            k3_a: vec![DVec3::ZERO; n],
            k4_v: vec![DVec3::ZERO; n],
            k4_a: vec![DVec3::ZERO; n],
            tmp_pos: vec![DVec3::ZERO; n],
        }
    }
}

/// Compute all body accelerations from `positions` into `out`, reusing the
/// output buffer.
#[inline]
fn accels_into(out: &mut [DVec3], positions: &[DVec3], gms: &[f64], star_id: usize) {
    for (i, acc) in out.iter_mut().enumerate() {
        *acc = if i == star_id {
            DVec3::ZERO
        } else {
            acceleration(i, positions, gms)
        };
    }
}

/// Advance the N-body state by one RK4 step of `dt` seconds.
/// Uses pre-allocated `scratch` buffers — zero heap allocations per call.
/// The body at `star_id` is kept fixed at the origin throughout.
pub(crate) fn rk4_step(
    state: &mut NBodyState,
    scratch: &mut Rk4Scratch,
    gms: &[f64],
    dt: f64,
    star_id: usize,
) {
    let n = state.positions.len();
    let half_dt = dt * 0.5;

    // k1: slopes at current state.
    scratch.k1_v.copy_from_slice(&state.velocities);
    accels_into(&mut scratch.k1_a, &state.positions, gms, star_id);

    // k2: slopes at midpoint using k1.
    for i in 0..n {
        if i == star_id {
            scratch.tmp_pos[i] = DVec3::ZERO;
            scratch.k2_v[i] = DVec3::ZERO;
        } else {
            scratch.tmp_pos[i] = state.positions[i] + scratch.k1_v[i] * half_dt;
            scratch.k2_v[i] = state.velocities[i] + scratch.k1_a[i] * half_dt;
        }
    }
    accels_into(&mut scratch.k2_a, &scratch.tmp_pos, gms, star_id);

    // k3: slopes at midpoint using k2.
    for i in 0..n {
        if i == star_id {
            scratch.tmp_pos[i] = DVec3::ZERO;
            scratch.k3_v[i] = DVec3::ZERO;
        } else {
            scratch.tmp_pos[i] = state.positions[i] + scratch.k2_v[i] * half_dt;
            scratch.k3_v[i] = state.velocities[i] + scratch.k2_a[i] * half_dt;
        }
    }
    accels_into(&mut scratch.k3_a, &scratch.tmp_pos, gms, star_id);

    // k4: slopes at endpoint using k3.
    for i in 0..n {
        if i == star_id {
            scratch.tmp_pos[i] = DVec3::ZERO;
            scratch.k4_v[i] = DVec3::ZERO;
        } else {
            scratch.tmp_pos[i] = state.positions[i] + scratch.k3_v[i] * dt;
            scratch.k4_v[i] = state.velocities[i] + scratch.k3_a[i] * dt;
        }
    }
    accels_into(&mut scratch.k4_a, &scratch.tmp_pos, gms, star_id);

    // Combine: weighted average of the four slopes.
    let sixth_dt = dt / 6.0;
    for i in 0..n {
        if i == star_id {
            continue;
        }
        state.positions[i] += sixth_dt
            * (scratch.k1_v[i]
                + 2.0 * scratch.k2_v[i]
                + 2.0 * scratch.k3_v[i]
                + scratch.k4_v[i]);
        state.velocities[i] += sixth_dt
            * (scratch.k1_a[i]
                + 2.0 * scratch.k2_a[i]
                + 2.0 * scratch.k3_a[i]
                + scratch.k4_a[i]);
    }
}

// ---------------------------------------------------------------------------
// Initial conditions
// ---------------------------------------------------------------------------

/// Build heliocentric initial states for every body by recursively resolving
/// parent chains.  Uses an iterative fixpoint so it tolerates any body
/// ordering in the definition file.
pub(crate) fn build_initial_states(system: &SolarSystemDefinition) -> (Vec<DVec3>, Vec<DVec3>, usize) {
    let n = system.bodies.len();
    let mut positions = vec![DVec3::ZERO; n];
    let mut velocities = vec![DVec3::ZERO; n];
    let mut resolved = vec![false; n];

    // The star is the body with no parent; it sits at the origin.
    let star_id = system
        .bodies
        .iter()
        .find(|b| b.parent.is_none())
        .map(|b| b.id)
        .unwrap_or(0);
    resolved[star_id] = true;

    // Iterative passes until no more bodies can be resolved.
    let mut progress = true;
    while progress {
        progress = false;
        for body in &system.bodies {
            if resolved[body.id] {
                continue;
            }
            let parent_id = match body.parent {
                Some(id) => id,
                None => {
                    // A second parentless body (unusual) — fix at origin.
                    resolved[body.id] = true;
                    progress = true;
                    continue;
                }
            };
            if !resolved[parent_id] {
                continue; // Parent not yet resolved; try again next pass.
            }

            let sv = match &body.orbital_elements {
                Some(elements) => {
                    orbital_elements_to_cartesian(elements, system.bodies[parent_id].gm)
                }
                None => crate::types::StateVector {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                },
            };

            // Heliocentric = parent heliocentric + body-relative.
            positions[body.id] = positions[parent_id] + sv.position;
            velocities[body.id] = velocities[parent_id] + sv.velocity;
            resolved[body.id] = true;
            progress = true;
        }
    }

    (positions, velocities, star_id)
}

// ---------------------------------------------------------------------------
// Public Ephemeris type
// ---------------------------------------------------------------------------

/// Precomputed N-body ephemeris for a solar system.
///
/// Constructed once and then queried at arbitrary times.  All states are in
/// the heliocentric inertial frame with the star pinned to the origin.
#[derive(Serialize, Deserialize)]
pub struct Ephemeris {
    pub(crate) tracks: Vec<BodyTrack>,
    time_span: f64,
}

impl Ephemeris {
    /// Precompute the ephemeris for `system` over `[0, time_span]` seconds.
    ///
    /// Uses N-body RK4 at a 1-hour fixed timestep with adaptive per-body
    /// sampling based on positional curvature.
    pub fn new(system: &SolarSystemDefinition, time_span: f64) -> Self {
        Self::new_with_progress(system, time_span, |_, _| {})
    }

    /// Like [`Self::new`] but calls `on_progress(step, total_steps)` periodically.
    pub fn new_with_progress(
        system: &SolarSystemDefinition,
        time_span: f64,
        on_progress: impl FnMut(u64, u64),
    ) -> Self {
        let n = system.bodies.len();
        let (init_positions, init_velocities, star_id) = build_initial_states(system);

        let gms: Vec<f64> = system.bodies.iter().map(|b| b.gm).collect();
        let masses: Vec<f64> = system.bodies.iter().map(|b| b.mass_kg).collect();

        // Seed each track with the t=0 sample.
        let mut tracks: Vec<BodyTrack> = (0..n)
            .map(|i| BodyTrack {
                mass_kg: masses[i],
                samples: vec![EphemerisSample {
                    time: 0.0,
                    position: init_positions[i],
                    velocity: init_velocities[i],
                }],
            })
            .collect();

        let mut state = NBodyState {
            positions: init_positions,
            velocities: init_velocities,
        };
        let mut scratch = Rk4Scratch::new(n);

        let steps = (time_span / DT).ceil() as u64;
        let progress_interval = (steps / 1000).max(1);
        let mut on_progress = on_progress;

        for step in 1..=steps {
            rk4_step(&mut state, &mut scratch, &gms, DT, star_id);
            let t = step as f64 * DT;
            let is_last = step == steps;

            for i in 0..n {
                if i == star_id {
                    continue;
                }

                let last = tracks[i].samples.last().unwrap();
                let dt_since = t - last.time;

                // Linear extrapolation from the last committed sample.
                let extrapolated = last.position + last.velocity * dt_since;
                let deviation = (state.positions[i] - extrapolated).length();

                if deviation > CURVATURE_THRESHOLD || is_last {
                    tracks[i].samples.push(EphemerisSample {
                        time: t,
                        position: state.positions[i],
                        velocity: state.velocities[i],
                    });
                }
            }

            if step % progress_interval == 0 || is_last {
                on_progress(step, steps);
            }
        }

        Ephemeris { tracks, time_span }
    }

    /// Convenience constructor using the default 100-year time span.
    pub fn new_default(system: &SolarSystemDefinition) -> Self {
        Self::new(system, DEFAULT_TIME_SPAN)
    }

    /// Returns position, velocity, and mass for every body at `time`.
    ///
    /// `time` is clamped to `[0, time_span]`.
    pub fn query(&self, time: f64) -> BodyStates {
        let t = time.clamp(0.0, self.time_span);
        self.tracks
            .iter()
            .map(|track| {
                let (position, velocity) = track.query(t);
                BodyState { position, velocity, mass_kg: track.mass_kg }
            })
            .collect()
    }

    /// Returns the state of a single body at `time`.
    ///
    /// `time` is clamped to `[0, time_span]`.
    ///
    /// # Panics
    /// Panics if `body_id >= body_count()`.
    pub fn query_body(&self, body_id: BodyId, time: f64) -> BodyState {
        let t = time.clamp(0.0, self.time_span);
        let track = &self.tracks[body_id];
        let (position, velocity) = track.query(t);
        BodyState { position, velocity, mass_kg: track.mass_kg }
    }

    /// Number of bodies in this ephemeris.
    pub fn body_count(&self) -> usize {
        self.tracks.len()
    }

    /// The time span this ephemeris was precomputed for, in seconds.
    pub fn time_span(&self) -> f64 {
        self.time_span
    }

    /// Number of stored samples for the given body.  Useful for diagnostics.
    pub fn sample_count(&self, body_id: BodyId) -> usize {
        self.tracks[body_id].samples.len()
    }

    /// Total number of samples across all bodies.
    pub fn total_sample_count(&self) -> usize {
        self.tracks.iter().map(|t| t.samples.len()).sum()
    }

    /// Serialize to bytes via bincode.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = bincode::serialize(self).map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Deserialize from a bincode file on disk.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }
}

// ---------------------------------------------------------------------------
// Energy validation
// ---------------------------------------------------------------------------

/// Result of validating energy conservation for a single body.
#[derive(Debug)]
pub struct BodyEnergyReport {
    pub body_id: BodyId,
    pub initial_energy: f64,
    pub final_energy: f64,
    pub max_drift_ppm: f64,
    pub passed: bool,
}

/// Result of the full energy conservation validation.
#[derive(Debug)]
pub struct EnergyValidationReport {
    pub bodies: Vec<BodyEnergyReport>,
    pub all_passed: bool,
}

/// Compute the specific orbital energy of body `i` in the full N-body system.
/// E = 0.5 * v² - Σ(GM_j / |r_i - r_j|) for all j ≠ i.
pub(crate) fn body_specific_energy(i: usize, positions: &[DVec3], velocities: &[DVec3], gms: &[f64]) -> f64 {
    let ke = 0.5 * velocities[i].length_squared();
    let mut pe = 0.0;
    for (j, &rj) in positions.iter().enumerate() {
        if j == i {
            continue;
        }
        let dist = (positions[i] - rj).length();
        if dist > 1e3 {
            pe -= gms[j] / dist;
        }
    }
    ke + pe
}

/// Validate energy conservation by re-integrating the system for `duration`
/// seconds and tracking per-body specific energy drift.
///
/// Returns a report with per-body results. A body "passes" if its maximum
/// energy drift stays under `threshold_ppm` parts per million.
pub fn validate_energy_conservation(
    system: &SolarSystemDefinition,
    duration: f64,
    threshold_ppm: f64,
) -> EnergyValidationReport {
    validate_energy_conservation_with_progress(system, duration, threshold_ppm, |_, _| {})
}

pub fn validate_energy_conservation_with_progress(
    system: &SolarSystemDefinition,
    duration: f64,
    threshold_ppm: f64,
    on_progress: impl FnMut(u64, u64),
) -> EnergyValidationReport {
    let n = system.bodies.len();
    let (init_positions, init_velocities, star_id) = build_initial_states(system);
    let gms: Vec<f64> = system.bodies.iter().map(|b| b.gm).collect();

    // Compute initial energies.
    let initial_energies: Vec<f64> = (0..n)
        .map(|i| body_specific_energy(i, &init_positions, &init_velocities, &gms))
        .collect();

    let mut state = NBodyState {
        positions: init_positions,
        velocities: init_velocities,
    };
    let mut scratch = Rk4Scratch::new(n);

    let steps = (duration / DT).ceil() as u64;
    let check_interval = (steps / 100).max(1); // sample ~100 times
    let progress_interval = (steps / 1000).max(1);
    let mut max_drifts = vec![0.0_f64; n];
    let mut on_progress = on_progress;

    for step in 1..=steps {
        rk4_step(&mut state, &mut scratch, &gms, DT, star_id);

        if step % check_interval == 0 || step == steps {
            for i in 0..n {
                if i == star_id {
                    continue;
                }
                let e = body_specific_energy(
                    i,
                    &state.positions,
                    &state.velocities,
                    &gms,
                );
                if initial_energies[i].abs() > 1e-30 {
                    let drift_ppm =
                        ((e - initial_energies[i]) / initial_energies[i]).abs() * 1e6;
                    max_drifts[i] = max_drifts[i].max(drift_ppm);
                }
            }
        }

        if step % progress_interval == 0 || step == steps {
            on_progress(step, steps);
        }
    }

    let final_energies: Vec<f64> = (0..n)
        .map(|i| body_specific_energy(i, &state.positions, &state.velocities, &gms))
        .collect();

    let bodies: Vec<BodyEnergyReport> = (0..n)
        .filter(|&i| i != star_id)
        .map(|i| {
            let passed = max_drifts[i] < threshold_ppm;
            BodyEnergyReport {
                body_id: i,
                initial_energy: initial_energies[i],
                final_energy: final_energies[i],
                max_drift_ppm: max_drifts[i],
                passed,
            }
        })
        .collect();

    let all_passed = bodies.iter().all(|b| b.passed);
    EnergyValidationReport { bodies, all_passed }
}

// ---------------------------------------------------------------------------
// Stability analysis (run during generation)
// ---------------------------------------------------------------------------

/// An event detected during ephemeris generation.
#[derive(Debug)]
pub struct StabilityEvent {
    pub time_s: f64,
    pub body_id: BodyId,
    pub description: String,
}

/// Analyse a freshly-built ephemeris for notable events:
///  - Close approaches between non-star bodies (< 10× sum of radii)
///  - Orbital radius changes > 10% from initial value
pub fn check_stability(
    system: &SolarSystemDefinition,
    ephemeris: &Ephemeris,
    check_points: usize,
) -> Vec<StabilityEvent> {
    let mut events = Vec::new();
    let n = system.bodies.len();
    let dt = ephemeris.time_span / check_points as f64;

    // Initial orbital radii (distance from parent).
    let initial_radii: Vec<Option<f64>> = system
        .bodies
        .iter()
        .map(|b| {
            b.parent.map(|pid| {
                let bs = ephemeris.query_body(b.id, 0.0);
                let ps = ephemeris.query_body(pid, 0.0);
                (bs.position - ps.position).length()
            })
        })
        .collect();

    // Track which events we've already reported to avoid flooding.
    let mut reported_close_approach: Vec<Vec<bool>> = vec![vec![false; n]; n];
    let mut reported_radius_drift: Vec<bool> = vec![false; n];

    for step in 0..=check_points {
        let t = step as f64 * dt;
        let states = ephemeris.query(t);

        // Close approach check.
        for i in 0..n {
            if system.bodies[i].parent.is_none() {
                continue;
            }
            for j in (i + 1)..n {
                if system.bodies[j].parent.is_none() {
                    continue;
                }
                if reported_close_approach[i][j] {
                    continue;
                }
                let dist = (states[i].position - states[j].position).length();
                let threshold =
                    10.0 * (system.bodies[i].radius_m + system.bodies[j].radius_m);
                if dist < threshold {
                    reported_close_approach[i][j] = true;
                    events.push(StabilityEvent {
                        time_s: t,
                        body_id: i,
                        description: format!(
                            "Close approach: {} and {} within {:.2e} m (threshold {:.2e} m)",
                            system.bodies[i].name,
                            system.bodies[j].name,
                            dist,
                            threshold,
                        ),
                    });
                }
            }

            // Orbital radius drift check.
            if !reported_radius_drift[i]
                && let (Some(pid), Some(init_r)) =
                    (system.bodies[i].parent, initial_radii[i])
            {
                let current_r =
                    (states[i].position - states[pid].position).length();
                let drift = (current_r - init_r).abs() / init_r;
                if drift > 0.10 {
                    reported_radius_drift[i] = true;
                    events.push(StabilityEvent {
                        time_s: t,
                        body_id: i,
                        description: format!(
                            "Orbital radius drift: {} changed {:.1}% from initial {:.2e} m",
                            system.bodies[i].name,
                            drift * 100.0,
                            init_r,
                        ),
                    });
                }
            }
        }
    }

    events
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        BodyDefinition, BodyKind, OrbitalElements, ShipDefinition, SolarSystemDefinition,
        StateVector, G,
    };
    use std::collections::HashMap;

    const AU: f64 = 1.496e11;
    const SUN_GM: f64 = 1.327_124_4e20; // m^3/s^2

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
            gm: SUN_GM,
            orbital_elements: None,
        };

        // Earth-like circular orbit at 1 AU.
        let earth = BodyDefinition {
            id: 1,
            name: "Earth".to_string(),
            kind: BodyKind::Planet,
            parent: Some(0),
            mass_kg: 5.972e24,
            radius_m: 6.371e6,
            color: [0.0, 0.5, 1.0],
            gm: G * 5.972e24,
            orbital_elements: Some(OrbitalElements {
                semi_major_axis_m: AU,
                eccentricity: 0.0,
                inclination_rad: 0.0,
                lon_ascending_node_rad: 0.0,
                arg_periapsis_rad: 0.0,
                true_anomaly_rad: 0.0,
            }),
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
    fn star_stays_at_origin() {
        let system = make_two_body_system();
        let eph = Ephemeris::new(&system, 3.156e7);
        let state = eph.query_body(0, 1.5e7);
        assert!(
            state.position.length() < 1.0,
            "Star drifted: {:?}",
            state.position
        );
    }

    #[test]
    fn earth_orbit_radius_conserved() {
        let system = make_two_body_system();
        // Two years.
        let eph = Ephemeris::new(&system, 2.0 * 3.156e7);

        // Sample at several times and check the orbital radius stays near 1 AU.
        let times = [0.0, 1.0e7, 2.0e7, 3.0e7, 5.0e7, 6.0e7];
        for &t in &times {
            let state = eph.query_body(1, t);
            let r = state.position.length();
            let rel_err = (r - AU).abs() / AU;
            assert!(
                rel_err < 1e-4,
                "Orbital radius off at t={}: r={:.3e}, expected {:.3e} (err={:.2e})",
                t,
                r,
                AU,
                rel_err
            );
        }
    }

    #[test]
    fn query_clamps_to_time_span() {
        let system = make_two_body_system();
        let span = 3.156e7;
        let eph = Ephemeris::new(&system, span);

        // Querying beyond the span should not panic and should equal the endpoint.
        let at_end = eph.query_body(1, span);
        let beyond = eph.query_body(1, span * 2.0);
        assert_eq!(at_end.position, beyond.position);

        // Same for before t=0.
        let at_start = eph.query_body(1, 0.0);
        let before = eph.query_body(1, -1000.0);
        assert_eq!(at_start.position, before.position);
    }

    #[test]
    fn query_returns_all_bodies() {
        let system = make_two_body_system();
        let eph = Ephemeris::new(&system, 3.156e7);
        let states = eph.query(1.0e6);
        assert_eq!(states.len(), 2);
    }

    /// Build a system with a highly eccentric orbit (e=0.9, comet-like).
    /// Near apoapsis the body moves very slowly so linear extrapolation stays
    /// accurate over many steps — giving genuinely sparse sampling there.
    fn make_eccentric_system() -> SolarSystemDefinition {
        let sun_mass = SUN_GM / G;
        let sun = BodyDefinition {
            id: 0,
            name: "Sun".to_string(),
            kind: BodyKind::Star,
            parent: None,
            mass_kg: sun_mass,
            radius_m: 6.957e8,
            color: [1.0, 1.0, 0.0],
            gm: SUN_GM,
            orbital_elements: None,
        };

        // Comet-like orbit: e=0.9, a=5 AU.
        // Apoapsis = a*(1+e) = 9.5 AU, periapsis = a*(1-e) = 0.5 AU.
        let comet = BodyDefinition {
            id: 1,
            name: "Comet".to_string(),
            kind: BodyKind::Comet,
            parent: Some(0),
            mass_kg: 1e13,
            radius_m: 5e3,
            color: [0.8, 0.8, 0.8],
            gm: G * 1e13,
            orbital_elements: Some(OrbitalElements {
                semi_major_axis_m: 5.0 * AU,
                eccentricity: 0.9,
                inclination_rad: 0.0,
                lon_ascending_node_rad: 0.0,
                arg_periapsis_rad: 0.0,
                // Start near apoapsis (true anomaly = π) where the body is
                // moving slowly and adaptive sampling should compress well.
                true_anomaly_rad: std::f64::consts::PI * 0.99,
            }),
        };

        let mut name_to_id = HashMap::new();
        name_to_id.insert("Sun".to_string(), 0);
        name_to_id.insert("Comet".to_string(), 1);

        SolarSystemDefinition {
            name: "Test".to_string(),
            bodies: vec![sun, comet],
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
    fn adaptive_sampling_reduces_storage() {
        // Use a highly eccentric orbit where adaptive sampling genuinely
        // compresses: near apoapsis the body moves slowly so the linear
        // extrapolation stays accurate over many steps.
        //
        // Orbital period T = 2π * sqrt(a³/GM) ≈ 35.4 years ≈ 1.116e9 s.
        // At 1-hour steps that is ~310,000 steps per orbit.
        // Near apoapsis (where we start) a comet spends ~90% of its time,
        // so we expect far fewer than 310k samples over one orbit.
        let system = make_eccentric_system();
        let one_orbit = 1.116e9; // approx one orbital period
        let steps = (one_orbit / DT) as usize;
        let eph = Ephemeris::new(&system, one_orbit);
        let samples = eph.sample_count(1);

        // Adaptive sampling must produce fewer samples than the number of steps.
        assert!(
            samples < steps,
            "Expected adaptive sampling to produce fewer than {steps} samples, got {samples}"
        );
        assert!(samples > 0, "Expected at least one sample");
    }

    #[test]
    fn dvec3_bincode_is_raw_f64s() {
        // Our streaming finalize assumes DVec3 serializes as 3 consecutive
        // LE f64s with no length prefix.  Verify this.
        let v = DVec3::new(1.0, 2.0, 3.0);
        let bytes = bincode::serialize(&v).unwrap();
        let mut raw = Vec::new();
        raw.extend_from_slice(&1.0_f64.to_le_bytes());
        raw.extend_from_slice(&2.0_f64.to_le_bytes());
        raw.extend_from_slice(&3.0_f64.to_le_bytes());
        assert_eq!(bytes, raw, "DVec3 bincode format changed — streaming finalize will break");
    }
}
