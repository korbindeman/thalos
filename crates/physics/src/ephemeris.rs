//! Ephemeris: precomputed N-body trajectories for all celestial bodies.
//!
//! # Design
//!
//! The full N-body system is integrated forward with RK4 at a fixed 1-hour
//! timestep. Instead of storing sampled states directly, each body is split
//! into fixed-duration segments and each segment is encoded as Chebyshev
//! polynomials over time for both position and velocity. This keeps lookup
//! cheap while reducing disk usage by orders of magnitude compared to raw
//! samples over millennial spans.

use crate::types::{
    orbital_elements_to_cartesian, BodyId, BodyState, BodyStates, SolarSystemDefinition,
    MIN_BODY_DISTANCE_SQ,
};
use glam::DVec3;
use std::io::{self, Read, Write};

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Default precomputation time span: 100 Julian years in seconds.
pub const DEFAULT_TIME_SPAN: f64 = 3.156e9;

/// RK4 integration timestep (seconds). 1 hour.
pub(crate) const DT: f64 = 3600.0;

/// Polynomial degree used for each Chebyshev segment.
pub(crate) const CHEBYSHEV_DEGREE: usize = 12;
pub(crate) const CHEBYSHEV_COEFFICIENTS: usize = CHEBYSHEV_DEGREE + 1;

/// Default segment span in integration steps. Seven days at 1-hour steps.
pub(crate) const SEGMENT_STEPS: usize = 24 * 7;

/// Binary file magic for the custom ephemeris format.
const FILE_MAGIC: [u8; 8] = *b"THEPHM02";

/// One serialized segment payload in bytes.
pub(crate) const SEGMENT_BYTES: usize =
    2 * 8 + 6 * CHEBYSHEV_COEFFICIENTS * 8;

/// Header bytes before the per-track payloads.
pub(crate) const FILE_HEADER_BYTES: u64 = 8 + 4 + 4 + 8;

/// Per-track bytes excluding segment payloads.
pub(crate) const TRACK_HEADER_BYTES: u64 = 8 + 8;

// ---------------------------------------------------------------------------
// Internal sample + segment types
// ---------------------------------------------------------------------------

/// A single RK4 sample used while fitting a segment.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EphemerisSample {
    pub(crate) time: f64,
    pub(crate) position: DVec3,
    pub(crate) velocity: DVec3,
}

/// A Chebyshev-encoded time segment for one body.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChebyshevSegment {
    pub(crate) start_time: f64,
    pub(crate) duration: f64,
    pub(crate) position_coeffs: [[f64; CHEBYSHEV_COEFFICIENTS]; 3],
    pub(crate) velocity_coeffs: [[f64; CHEBYSHEV_COEFFICIENTS]; 3],
}

impl ChebyshevSegment {
    fn query(&self, time: f64) -> (DVec3, DVec3) {
        if self.duration <= 0.0 {
            let position = DVec3::new(
                self.position_coeffs[0][0],
                self.position_coeffs[1][0],
                self.position_coeffs[2][0],
            );
            let velocity = DVec3::new(
                self.velocity_coeffs[0][0],
                self.velocity_coeffs[1][0],
                self.velocity_coeffs[2][0],
            );
            return (position, velocity);
        }

        let x = (((time - self.start_time) / self.duration) * 2.0 - 1.0).clamp(-1.0, 1.0);
        let position = DVec3::new(
            eval_chebyshev(&self.position_coeffs[0], x),
            eval_chebyshev(&self.position_coeffs[1], x),
            eval_chebyshev(&self.position_coeffs[2], x),
        );
        let velocity = DVec3::new(
            eval_chebyshev(&self.velocity_coeffs[0], x),
            eval_chebyshev(&self.velocity_coeffs[1], x),
            eval_chebyshev(&self.velocity_coeffs[2], x),
        );

        (position, velocity)
    }
}

// ---------------------------------------------------------------------------
// Per-body track
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct BodyTrack {
    pub(crate) mass_kg: f64,
    pub(crate) segments: Vec<ChebyshevSegment>,
}

impl BodyTrack {
    /// Return the state at the given time using Chebyshev segment lookup.
    /// Times outside [t0, t_last] are clamped to the nearest endpoint.
    fn query(&self, time: f64) -> (DVec3, DVec3) {
        if self.segments.is_empty() {
            return (DVec3::ZERO, DVec3::ZERO);
        }
        if self.segments.len() == 1 {
            let seg = &self.segments[0];
            return seg.query(time.clamp(seg.start_time, seg.start_time + seg.duration));
        }

        let first = &self.segments[0];
        let last = self.segments.last().unwrap();
        let t = time.clamp(first.start_time, last.start_time + last.duration);

        let idx = match self
            .segments
            .binary_search_by(|seg| seg.start_time.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };

        let seg = &self.segments[idx.min(self.segments.len() - 1)];
        seg.query(t)
    }
}

// ---------------------------------------------------------------------------
// Chebyshev helpers
// ---------------------------------------------------------------------------

#[inline]
fn eval_chebyshev(coeffs: &[f64; CHEBYSHEV_COEFFICIENTS], x: f64) -> f64 {
    let mut b_kplus1 = 0.0;
    let mut b_kplus2 = 0.0;

    for &coeff in coeffs.iter().skip(1).rev() {
        let b_k = 2.0 * x * b_kplus1 - b_kplus2 + coeff;
        b_kplus2 = b_kplus1;
        b_kplus1 = b_k;
    }

    coeffs[0] + x * b_kplus1 - b_kplus2
}

fn chebyshev_nodes() -> [f64; CHEBYSHEV_COEFFICIENTS] {
    std::array::from_fn(|j| {
        (std::f64::consts::PI * j as f64 / CHEBYSHEV_DEGREE as f64).cos()
    })
}

fn chebyshev_coefficients(values: &[f64; CHEBYSHEV_COEFFICIENTS]) -> [f64; CHEBYSHEV_COEFFICIENTS] {
    let n = CHEBYSHEV_DEGREE as f64;
    let mut coeffs = [0.0; CHEBYSHEV_COEFFICIENTS];

    for (k, coeff) in coeffs.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (j, value) in values.iter().enumerate() {
            let weight = if j == 0 || j == CHEBYSHEV_DEGREE { 0.5 } else { 1.0 };
            let angle = std::f64::consts::PI * k as f64 * j as f64 / n;
            sum += weight * value * angle.cos();
        }
        *coeff = 2.0 * sum / n;
    }

    coeffs[0] *= 0.5;
    coeffs[CHEBYSHEV_DEGREE] *= 0.5;
    coeffs
}

fn sample_at_time(samples: &[EphemerisSample], time: f64) -> EphemerisSample {
    debug_assert!(!samples.is_empty());

    if samples.len() == 1 {
        return samples[0];
    }

    let first = samples[0];
    let last = *samples.last().unwrap();
    let t = time.clamp(first.time, last.time);
    if (t - last.time).abs() < 1e-9 {
        return last;
    }

    let idx = (((t - first.time) / DT).floor() as usize).min(samples.len() - 2);
    let s0 = samples[idx];
    let s1 = samples[idx + 1];
    let (position, velocity) = hermite_interp(
        s0.time, s0.position, s0.velocity, s1.time, s1.position, s1.velocity, t,
    );

    EphemerisSample { time: t, position, velocity }
}

pub(crate) fn fit_chebyshev_segment(samples: &[EphemerisSample]) -> ChebyshevSegment {
    debug_assert!(!samples.is_empty());

    let start_time = samples[0].time;
    let end_time = samples.last().unwrap().time;
    let duration = (end_time - start_time).max(0.0);

    if duration <= 0.0 {
        let sample = samples[0];
        let mut position_coeffs = [[0.0; CHEBYSHEV_COEFFICIENTS]; 3];
        let mut velocity_coeffs = [[0.0; CHEBYSHEV_COEFFICIENTS]; 3];
        position_coeffs[0][0] = sample.position.x;
        position_coeffs[1][0] = sample.position.y;
        position_coeffs[2][0] = sample.position.z;
        velocity_coeffs[0][0] = sample.velocity.x;
        velocity_coeffs[1][0] = sample.velocity.y;
        velocity_coeffs[2][0] = sample.velocity.z;
        return ChebyshevSegment {
            start_time,
            duration: 0.0,
            position_coeffs,
            velocity_coeffs,
        };
    }

    let nodes = chebyshev_nodes();
    let mut px = [0.0; CHEBYSHEV_COEFFICIENTS];
    let mut py = [0.0; CHEBYSHEV_COEFFICIENTS];
    let mut pz = [0.0; CHEBYSHEV_COEFFICIENTS];
    let mut vx = [0.0; CHEBYSHEV_COEFFICIENTS];
    let mut vy = [0.0; CHEBYSHEV_COEFFICIENTS];
    let mut vz = [0.0; CHEBYSHEV_COEFFICIENTS];

    let mid = start_time + duration * 0.5;
    let half = duration * 0.5;
    for (i, &node) in nodes.iter().enumerate() {
        let sample = sample_at_time(samples, mid + half * node);
        px[i] = sample.position.x;
        py[i] = sample.position.y;
        pz[i] = sample.position.z;
        vx[i] = sample.velocity.x;
        vy[i] = sample.velocity.y;
        vz[i] = sample.velocity.z;
    }

    ChebyshevSegment {
        start_time,
        duration,
        position_coeffs: [
            chebyshev_coefficients(&px),
            chebyshev_coefficients(&py),
            chebyshev_coefficients(&pz),
        ],
        velocity_coeffs: [
            chebyshev_coefficients(&vx),
            chebyshev_coefficients(&vy),
            chebyshev_coefficients(&vz),
        ],
    }
}

pub(crate) fn write_segment(w: &mut impl Write, segment: &ChebyshevSegment) -> io::Result<()> {
    w.write_all(&segment.start_time.to_le_bytes())?;
    w.write_all(&segment.duration.to_le_bytes())?;

    for coeffs in segment
        .position_coeffs
        .iter()
        .chain(segment.velocity_coeffs.iter())
    {
        for coeff in coeffs {
            w.write_all(&coeff.to_le_bytes())?;
        }
    }

    Ok(())
}

pub(crate) fn read_segment(r: &mut impl Read) -> io::Result<ChebyshevSegment> {
    let read_f64 = |reader: &mut dyn Read| -> io::Result<f64> {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        Ok(f64::from_le_bytes(bytes))
    };

    let start_time = read_f64(r)?;
    let duration = read_f64(r)?;
    let mut position_coeffs = [[0.0; CHEBYSHEV_COEFFICIENTS]; 3];
    let mut velocity_coeffs = [[0.0; CHEBYSHEV_COEFFICIENTS]; 3];

    for coeffs in position_coeffs.iter_mut().chain(velocity_coeffs.iter_mut()) {
        for coeff in coeffs {
            *coeff = read_f64(r)?;
        }
    }

    Ok(ChebyshevSegment {
        start_time,
        duration,
        position_coeffs,
        velocity_coeffs,
    })
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
    if dt.abs() < 1e-12 {
        return (p0, v0);
    }

    let u = (t - t0) / dt;
    let u2 = u * u;
    let u3 = u2 * u;

    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;

    let position = h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1;

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
        if dist2 < MIN_BODY_DISTANCE_SQ {
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
    k1_v: Vec<DVec3>,
    k1_a: Vec<DVec3>,
    k2_v: Vec<DVec3>,
    k2_a: Vec<DVec3>,
    k3_v: Vec<DVec3>,
    k3_a: Vec<DVec3>,
    k4_v: Vec<DVec3>,
    k4_a: Vec<DVec3>,
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
pub(crate) fn rk4_step(
    state: &mut NBodyState,
    scratch: &mut Rk4Scratch,
    gms: &[f64],
    dt: f64,
    star_id: usize,
) {
    let n = state.positions.len();
    let half_dt = dt * 0.5;

    scratch.k1_v.copy_from_slice(&state.velocities);
    accels_into(&mut scratch.k1_a, &state.positions, gms, star_id);

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
/// parent chains.
pub(crate) fn build_initial_states(
    system: &SolarSystemDefinition,
) -> (Vec<DVec3>, Vec<DVec3>, usize) {
    let n = system.bodies.len();
    let mut positions = vec![DVec3::ZERO; n];
    let mut velocities = vec![DVec3::ZERO; n];
    let mut resolved = vec![false; n];

    let star_id = system
        .bodies
        .iter()
        .find(|b| b.parent.is_none())
        .map(|b| b.id)
        .unwrap_or(0);
    resolved[star_id] = true;

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
                    resolved[body.id] = true;
                    progress = true;
                    continue;
                }
            };
            if !resolved[parent_id] {
                continue;
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
pub struct Ephemeris {
    pub(crate) tracks: Vec<BodyTrack>,
    pub(crate) time_span: f64,
}

impl Ephemeris {
    /// Precompute the ephemeris for `system` over `[0, time_span]` seconds.
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

        let mut tracks: Vec<BodyTrack> = (0..n)
            .map(|i| BodyTrack { mass_kg: masses[i], segments: Vec::new() })
            .collect();

        let mut buffers: Vec<Vec<EphemerisSample>> = (0..n)
            .map(|i| {
                vec![EphemerisSample {
                    time: 0.0,
                    position: init_positions[i],
                    velocity: init_velocities[i],
                }]
            })
            .collect();

        tracks[star_id].segments.push(fit_chebyshev_segment(&buffers[star_id]));

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

                buffers[i].push(EphemerisSample {
                    time: t,
                    position: state.positions[i],
                    velocity: state.velocities[i],
                });

                let completed_steps = buffers[i].len() - 1;
                if completed_steps >= SEGMENT_STEPS || is_last {
                    let segment = fit_chebyshev_segment(&buffers[i]);
                    tracks[i].segments.push(segment);

                    let last_sample = *buffers[i].last().unwrap();
                    buffers[i].clear();
                    buffers[i].push(last_sample);
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
    pub fn query(&self, time: f64) -> BodyStates {
        let t = if time < 0.0 || time > self.time_span {
            #[cfg(debug_assertions)]
            eprintln!("[ephemeris] query time {:.0}s clamped to [0, {:.0}]", time, self.time_span);
            time.clamp(0.0, self.time_span)
        } else {
            time
        };
        self.tracks
            .iter()
            .map(|track| {
                let (position, velocity) = track.query(t);
                BodyState { position, velocity, mass_kg: track.mass_kg }
            })
            .collect()
    }

    /// Returns the state of a single body at `time`.
    pub fn query_body(&self, body_id: BodyId, time: f64) -> BodyState {
        let t = if time < 0.0 || time > self.time_span {
            #[cfg(debug_assertions)]
            eprintln!("[ephemeris] query_body({}) time {:.0}s clamped to [0, {:.0}]", body_id, time, self.time_span);
            time.clamp(0.0, self.time_span)
        } else {
            time
        };
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

    /// Number of stored segments for the given body.
    pub fn segment_count(&self, body_id: BodyId) -> usize {
        self.tracks[body_id].segments.len()
    }

    /// Total number of segments across all bodies.
    pub fn total_segment_count(&self) -> usize {
        self.tracks.iter().map(|t| t.segments.len()).sum()
    }

    /// Serialize using the custom binary format.
    pub fn save(&self, path: &std::path::Path) -> io::Result<()> {
        let mut out = io::BufWriter::new(std::fs::File::create(path)?);
        self.write_to(&mut out)?;
        out.flush()
    }

    /// Deserialize from a custom ephemeris file on disk.
    pub fn load(path: &std::path::Path) -> io::Result<Self> {
        let mut reader = io::BufReader::new(std::fs::File::open(path)?);
        Self::read_from(&mut reader)
    }

    pub(crate) fn write_to(&self, out: &mut impl Write) -> io::Result<()> {
        out.write_all(&FILE_MAGIC)?;
        out.write_all(&(1u32).to_le_bytes())?;
        out.write_all(&(self.tracks.len() as u32).to_le_bytes())?;
        out.write_all(&self.time_span.to_le_bytes())?;

        for track in &self.tracks {
            out.write_all(&track.mass_kg.to_le_bytes())?;
            out.write_all(&(track.segments.len() as u64).to_le_bytes())?;
            for segment in &track.segments {
                write_segment(out, segment)?;
            }
        }

        Ok(())
    }

    pub(crate) fn read_from(input: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        if magic != FILE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unrecognized ephemeris file format",
            ));
        }

        let version = read_u32(input)?;
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported ephemeris version: {version}"),
            ));
        }

        let track_count = read_u32(input)? as usize;
        let time_span = read_f64(input)?;
        let mut tracks = Vec::with_capacity(track_count);

        for _ in 0..track_count {
            let mass_kg = read_f64(input)?;
            let segment_count = read_u64(input)? as usize;
            let mut segments = Vec::with_capacity(segment_count);
            for _ in 0..segment_count {
                segments.push(read_segment(input)?);
            }
            tracks.push(BodyTrack { mass_kg, segments });
        }

        Ok(Self { tracks, time_span })
    }
}

fn read_u32(input: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(input: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    input.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f64(input: &mut impl Read) -> io::Result<f64> {
    let mut bytes = [0u8; 8];
    input.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
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

    let initial_energies: Vec<f64> = (0..n)
        .map(|i| body_specific_energy(i, &init_positions, &init_velocities, &gms))
        .collect();

    let mut state = NBodyState {
        positions: init_positions,
        velocities: init_velocities,
    };
    let mut scratch = Rk4Scratch::new(n);

    let steps = (duration / DT).ceil() as u64;
    let check_interval = (steps / 100).max(1);
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
                let e = body_specific_energy(i, &state.positions, &state.velocities, &gms);
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
        .map(|i| BodyEnergyReport {
            body_id: i,
            initial_energy: initial_energies[i],
            final_energy: final_energies[i],
            max_drift_ppm: max_drifts[i],
            passed: max_drifts[i] < threshold_ppm,
        })
        .collect();

    let all_passed = bodies.iter().all(|b| b.passed);
    EnergyValidationReport { bodies, all_passed }
}

// ---------------------------------------------------------------------------
// Stability analysis
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct StabilityEvent {
    pub time_s: f64,
    pub body_id: BodyId,
    pub description: String,
}

pub fn check_stability(
    system: &SolarSystemDefinition,
    ephemeris: &Ephemeris,
    check_points: usize,
) -> Vec<StabilityEvent> {
    let mut events = Vec::new();
    let n = system.bodies.len();
    let dt = ephemeris.time_span / check_points as f64;

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

    let mut reported_close_approach: Vec<Vec<bool>> = vec![vec![false; n]; n];
    let mut reported_radius_drift: Vec<bool> = vec![false; n];

    for step in 0..=check_points {
        let t = step as f64 * dt;
        let states = ephemeris.query(t);

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

            if !reported_radius_drift[i]
                && let (Some(pid), Some(init_r)) = (system.bodies[i].parent, initial_radii[i])
            {
                let current_r = (states[i].position - states[pid].position).length();
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
    const SUN_GM: f64 = 1.327_124_4e20;

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
        assert!(state.position.length() < 1.0, "Star drifted: {:?}", state.position);
    }

    #[test]
    fn earth_orbit_radius_conserved() {
        let system = make_two_body_system();
        let eph = Ephemeris::new(&system, 2.0 * 3.156e7);
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

        let at_end = eph.query_body(1, span);
        let beyond = eph.query_body(1, span * 2.0);
        assert_eq!(at_end.position, beyond.position);

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

    #[test]
    fn chebyshev_coefficients_reconstruct_linear_function() {
        let values = std::array::from_fn(|i| {
            let x = chebyshev_nodes()[i];
            3.0 + 2.0 * x
        });
        let coeffs = chebyshev_coefficients(&values);
        for &x in &chebyshev_nodes() {
            let y = eval_chebyshev(&coeffs, x);
            assert!((y - (3.0 + 2.0 * x)).abs() < 1e-10);
        }
    }

    #[test]
    fn save_roundtrip_preserves_queries() {
        let system = make_two_body_system();
        let eph = Ephemeris::new(&system, 30.0 * 24.0 * 3600.0);
        let path = std::env::temp_dir().join("thalos_ephemeris_roundtrip.bin");
        eph.save(&path).unwrap();
        let loaded = Ephemeris::load(&path).unwrap();
        std::fs::remove_file(path).ok();

        let t = 11.5 * 24.0 * 3600.0;
        let a = eph.query_body(1, t);
        let b = loaded.query_body(1, t);
        assert!((a.position - b.position).length() < 1e-6);
        assert!((a.velocity - b.velocity).length() < 1e-9);
    }
}
