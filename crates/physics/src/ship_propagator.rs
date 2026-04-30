//! Ship propagation: analytical Kepler coasts + RK4 finite burns, with
//! SOI transition detection via root-finding.
//!
//! # Architecture
//!
//! [`ShipPropagator`] is the boundary between ship motion and everything else
//! in the physics stack. Today's implementation ([`KeplerianPropagator`]) uses
//! patched conics: the ship feels gravity from a single "SOI body" at a time,
//! transitions between SOIs are detected analytically, and bound coasts
//! propagate in closed form. A future N-body implementation can slot in
//! behind the same trait without touching `Simulation::step` or trajectory
//! prediction.
//!
//! # Coast vs burn
//!
//! Each call propagates a single segment of one flavour:
//!
//! - [`ShipPropagator::coast_segment`] — no thrust. Propagation is exact Kepler
//!   in the SOI body's frame. Samples are generated at uniform-time intervals
//!   and SOI crossings are bisected to machine precision.
//! - [`ShipPropagator::burn_segment`] — constant thrust acceleration in the
//!   ship's local frame relative to `reference_body`. RK4 with fixed substep
//!   of a few seconds; one sample per substep. SOI crossings are caught on
//!   the same per-substep checks and refined by bisection against the RK4
//!   hermite between substep endpoints.
//!
//! Both methods terminate at the first of: target time, SOI entry, SOI exit,
//! or collision with the current SOI body. Callers handle re-entry into the
//! propagator with the new SOI body.

use glam::DVec3;

use crate::body_state_provider::BodyStateProvider;
use crate::maneuver::delta_v_to_world;
use crate::orbital_math::{
    cartesian_to_elements, eccentric_from_true_elliptic, hyperbolic_from_true, propagate_kepler,
    solve_kepler_elliptic, solve_kepler_hyperbolic,
};
use crate::types::{BodyDefinition, BodyId, BodyState, G, StateVector, TrajectorySample};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Finite-burn parameters for a single segment.
///
/// The thrust direction is specified in the ship's local prograde/normal/
/// radial frame relative to `reference_body`. The frame is recomputed from
/// the ship's live state at every substep — the only honest interpretation
/// of a local Δv when the burn duration is a meaningful fraction of the
/// orbital period.
///
/// Thrust is modeled as constant force `thrust_n` with linear mass loss
/// at `mass_flow_kg_per_s`, so acceleration grows over the burn:
/// `m(t) = initial_mass_kg − mass_flow_kg_per_s · (t − start_time)`.
/// Once `m(t)` reaches `dry_mass_kg` propellant is exhausted and thrust
/// cuts off cleanly — the integrator continues coasting under gravity
/// alone for the remainder of `[start_time, end_time]`.
#[derive(Debug, Clone, Copy)]
pub struct BurnParams {
    /// Direction in the prograde/normal/radial frame. Magnitude is not used
    /// for the force (that's `thrust_n`) — only the direction matters.
    pub delta_v_local: DVec3,
    pub reference_body: BodyId,
    pub thrust_n: f64,
    pub initial_mass_kg: f64,
    pub mass_flow_kg_per_s: f64,
    /// Floor under which thrust stops applying — physical "out of fuel".
    pub dry_mass_kg: f64,
    pub start_time: f64,
    pub end_time: f64,
}

/// Why a propagated segment terminated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentTerminator {
    /// Reached the requested target time without any event.
    Horizon,
    /// A bound coast completed one full orbital period around the SOI body
    /// (only when the caller asked for `stop_on_stable_orbit`).
    StableOrbit,
    /// Ship crossed into `body`'s SOI at `time`.
    SoiEnter { body: BodyId, time: f64 },
    /// Ship crossed out of its current SOI, falling into the parent SOI.
    SoiExit { from: BodyId, to: BodyId, time: f64 },
    /// Ship reached the surface of `body` at `time`.
    Collision { body: BodyId, time: f64 },
    /// Burn window ended (only in `burn_segment`). Time is exact burn end.
    BurnEnd { time: f64 },
}

/// Result of propagating a single segment.
#[derive(Debug, Clone)]
pub struct SegmentResult {
    /// One `TrajectorySample` per rendered point, including the starting
    /// sample. For coasts: uniform in time. For burns: one per RK4 substep.
    /// The last sample sits exactly at `end_time`.
    pub samples: Vec<TrajectorySample>,
    pub terminator: SegmentTerminator,
    pub end_state: StateVector,
    pub end_time: f64,
    /// The SOI body the ship is in at `end_time`. Differs from the caller's
    /// input when the segment terminated in an SOI transition.
    pub end_soi_body: BodyId,
}

/// Inputs to a coast propagation.
#[derive(Clone, Copy)]
pub struct CoastRequest<'a> {
    pub state: StateVector,
    pub time: f64,
    pub soi_body: BodyId,
    pub target_time: f64,
    /// If true and the orbit is bound within the SOI, terminates after one
    /// complete period and marks [`SegmentTerminator::StableOrbit`].
    pub stop_on_stable_orbit: bool,
    /// Hint for how many samples to generate. Actual count may be clipped by
    /// `min_coast_samples` and rounded.
    pub sample_count_hint: usize,
    pub ephemeris: &'a dyn BodyStateProvider,
    pub bodies: &'a [BodyDefinition],
}

/// Inputs to a burn propagation.
#[derive(Clone, Copy)]
pub struct BurnRequest<'a> {
    pub state: StateVector,
    pub time: f64,
    pub soi_body: BodyId,
    /// Usually `burn.end_time.min(leg_end)` — whichever fires first.
    pub target_time: f64,
    pub burn: BurnParams,
    pub ephemeris: &'a dyn BodyStateProvider,
    pub bodies: &'a [BodyDefinition],
}

/// Pluggable propagation engine.
pub trait ShipPropagator: Send + Sync {
    /// Propagate a coast segment. Terminates at the first of: target time,
    /// SOI entry/exit, collision, or stable-orbit closure.
    fn coast_segment(&self, req: CoastRequest<'_>) -> SegmentResult;

    /// Propagate a finite-burn segment. Terminates at the first of: target
    /// time, SOI entry/exit, collision, or burn end.
    fn burn_segment(&self, req: BurnRequest<'_>) -> SegmentResult;

    /// Find the innermost body whose SOI contains the ship at `time`.
    ///
    /// Algorithm: walk all bodies, collect any whose SOI currently contains
    /// the ship, return the one with the smallest SOI radius. The root body
    /// (typically the star, SOI = ∞) always matches as a fallback.
    fn soi_body_of(
        &self,
        position: DVec3,
        time: f64,
        ephemeris: &dyn BodyStateProvider,
        bodies: &[BodyDefinition],
    ) -> BodyId;
}

// ---------------------------------------------------------------------------
// KeplerianPropagator
// ---------------------------------------------------------------------------

/// Default Keplerian patched-conics propagator.
#[derive(Debug, Clone)]
pub struct KeplerianPropagator {
    /// RK4 substep size for finite burns, seconds. Default 1.0 — accurate
    /// enough for typical chemical-burn magnitudes without flooding the
    /// sample buffer.
    pub burn_substep_s: f64,
    /// Minimum samples per coast segment. Default 2. Short segments (e.g. a
    /// burn window that coincides with target_time) still produce at least
    /// start + end.
    pub min_coast_samples: usize,
    /// Soft cap on coast samples per segment. Default 512. Guards against
    /// absurdly long coasts with dense sampling hints.
    pub max_coast_samples: usize,
}

impl Default for KeplerianPropagator {
    fn default() -> Self {
        Self {
            burn_substep_s: 1.0,
            min_coast_samples: 2,
            max_coast_samples: 512,
        }
    }
}

impl ShipPropagator for KeplerianPropagator {
    fn coast_segment(&self, req: CoastRequest<'_>) -> SegmentResult {
        self.coast_segment_impl(req)
    }

    fn burn_segment(&self, req: BurnRequest<'_>) -> SegmentResult {
        self.burn_segment_impl(req)
    }

    fn soi_body_of(
        &self,
        position: DVec3,
        time: f64,
        ephemeris: &dyn BodyStateProvider,
        bodies: &[BodyDefinition],
    ) -> BodyId {
        innermost_soi_body(position, time, ephemeris, bodies)
    }
}

// ---------------------------------------------------------------------------
// Coast segment
// ---------------------------------------------------------------------------

impl KeplerianPropagator {
    fn coast_segment_impl(&self, req: CoastRequest<'_>) -> SegmentResult {
        let CoastRequest {
            state,
            time,
            soi_body,
            mut target_time,
            stop_on_stable_orbit,
            sample_count_hint,
            ephemeris,
            bodies,
        } = req;

        let body_t0 = ephemeris.query_body(soi_body, time);
        let mu = body_t0.mass_kg * G;

        // Relative state in the SOI body's frame.
        let rel0 = StateVector {
            position: state.position - body_t0.position,
            velocity: state.velocity - body_t0.velocity,
        };

        // Stable-orbit detection: a bound orbit whose apoapsis fits inside the
        // SOI and whose periapsis clears the surface can be visualised as a
        // closed loop over exactly one period.
        let mut is_stable_orbit = false;
        if stop_on_stable_orbit && mu > 0.0 {
            if let Some(el) = cartesian_to_elements(rel0, mu) {
                let body_radius = bodies.get(soi_body).map(|b| b.radius_m).unwrap_or(0.0);
                let soi_radius = bodies
                    .get(soi_body)
                    .map(|b| b.soi_radius_m)
                    .unwrap_or(f64::INFINITY);
                if el.eccentricity < 1.0
                    && el.semi_major_axis_m.is_finite()
                    && el.apoapsis_m < soi_radius
                    && el.periapsis_m > body_radius
                {
                    let period = std::f64::consts::TAU * (el.semi_major_axis_m.powi(3) / mu).sqrt();
                    let period_end = time + period;
                    if period_end < target_time {
                        target_time = period_end;
                        is_stable_orbit = true;
                    }
                }
            }
        }

        if target_time <= time {
            // Degenerate: zero-length segment. Emit a single sample at t0.
            let sample = build_sample(time, state, soi_body, ephemeris);
            return SegmentResult {
                samples: vec![sample],
                terminator: SegmentTerminator::Horizon,
                end_state: state,
                end_time: time,
                end_soi_body: soi_body,
            };
        }

        // Sample count, clipped.
        let n = sample_count_hint
            .max(self.min_coast_samples)
            .min(self.max_coast_samples);

        // Sample times for the coast. Uniform-time equals uniform mean
        // anomaly, which sparse-samples the periapsis side of eccentric
        // orbits and makes the rendered curve look blocky there (and
        // artificially straight on the apoapsis arm). Uniform *eccentric*
        // anomaly gives arc-length-ish density, dense at both apsides,
        // which is what the player actually sees. Hyperbolic trajectories
        // use the hyperbolic-anomaly analogue.
        let sample_times = coast_sample_times(rel0, mu, time, target_time, n);

        // Helper: evaluate the ship's heliocentric state at sim time `t`.
        let eval_at = |t: f64| -> StateVector {
            let body_t = ephemeris.query_body(soi_body, t);
            let rel_t = propagate_kepler(rel0, mu, t - time);
            StateVector {
                position: body_t.position + rel_t.position,
                velocity: body_t.velocity + rel_t.velocity,
            }
        };

        // Enumerate candidate "threat" bodies: siblings under the same parent
        // plus the soi_body's parent (for exit). A child body only has a
        // chance of intercepting us if we orbit its parent; we simply scan
        // every body and filter out those whose ambient geometry forbids a
        // crossing (see `can_intercept`).
        let threat_bodies: Vec<BodyId> = (0..bodies.len())
            .filter(|&b| b != soi_body && can_intercept(b, soi_body, bodies))
            .collect();

        let mut samples: Vec<TrajectorySample> = Vec::with_capacity(n + 2);

        // Push start sample.
        samples.push(build_sample(time, state, soi_body, ephemeris));

        let mut prev_t = time;
        let mut prev_state = state;

        let soi_radius = bodies[soi_body].soi_radius_m;
        let body_radius = bodies[soi_body].radius_m;

        // Worklist of pending sample times. Subdivision pushes back, so we
        // need a mutable stack — `Vec::pop` returns the last element, so we
        // store in reverse order of intended processing.
        let mut work: Vec<f64> = sample_times.into_iter().skip(1).collect();
        work.reverse();

        // Step cap: bounds the cubic Hermite's accuracy in
        // [`detect_step_crossings`] — the swept-min check assumes the cubic
        // tracks the actual trajectory, which holds as long as the
        // trajectory doesn't swing wildly within one step. Two
        // complementary triggers, both relative to the smaller endpoint
        // altitude:
        //   - altitude-change cap, for steps that obviously cross many
        //     altitude bands;
        //   - relative-speed × h cap, for steps whose endpoints land at
        //     similar altitudes but cover a long path through periapsis
        //     (the canonical "symmetric dip" — `|cur_alt − prev_alt| ≈ 0`,
        //     so the altitude cap alone doesn't fire).
        const MAX_ALT_CHANGE_RATIO: f64 = 0.25;
        const MAX_PATH_RATIO: f64 = 0.25;
        // Lower bound on subdivided step duration. Stops runaway recursion if
        // a cap becomes unsatisfiable (e.g. ship grazing the surface).
        const MIN_STEP_S: f64 = 1e-3;

        while let Some(t) = work.pop() {
            let cur_state = eval_at(t);

            let prev_body = ephemeris.query_body(soi_body, prev_t);
            let cur_body = ephemeris.query_body(soi_body, t);
            let prev_alt = (prev_state.position - prev_body.position).length();
            let cur_alt = (cur_state.position - cur_body.position).length();
            let min_alt = prev_alt.min(cur_alt);
            let alt_change = (cur_alt - prev_alt).abs();
            let rel_speed = (prev_state.velocity - prev_body.velocity)
                .length()
                .max((cur_state.velocity - cur_body.velocity).length());
            let path = rel_speed * (t - prev_t);
            let needs_subdivide =
                alt_change > MAX_ALT_CHANGE_RATIO * min_alt || path > MAX_PATH_RATIO * min_alt;
            if needs_subdivide && (t - prev_t) > MIN_STEP_S {
                let mid = 0.5 * (prev_t + t);
                work.push(t);
                work.push(mid);
                continue;
            }

            let xings = detect_step_crossings(
                prev_state,
                prev_t,
                cur_state,
                t,
                soi_body,
                soi_radius,
                body_radius,
                &threat_bodies,
                ephemeris,
                bodies,
            );

            // Resolve crossings in `exit > collision > enter` order. Exit
            // and collision are geometrically mutually exclusive (one is
            // outward through the SOI radius, the other is inward through
            // the body radius); collision wins over enter because hitting
            // the SOI body wrecks the ship, so any near-miss into a child
            // SOI in the same step is moot. `refine_crossing` returns
            // `None` when the swept-extremum check turned out to be a
            // Hermite false positive — fall through to the next event in
            // that case instead of aborting the segment. The burn path
            // uses the same priority — see `burn_segment_impl`.
            if xings.exit
                && let Some(t_cross) = refine_crossing(
                    prev_t, t, soi_body, soi_radius, rel0, mu, time, soi_body, ephemeris,
                )
            {
                let cross_state = eval_at(t_cross);
                samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                return SegmentResult::exit(samples, cross_state, t_cross, soi_body, bodies);
            }
            if xings.collision
                && let Some(t_cross) = refine_crossing(
                    prev_t,
                    t,
                    soi_body,
                    body_radius,
                    rel0,
                    mu,
                    time,
                    soi_body,
                    ephemeris,
                )
            {
                let cross_state = eval_at(t_cross);
                samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                return SegmentResult::collision(samples, cross_state, t_cross, soi_body);
            }
            if let Some(child) = xings.enter {
                let child_soi = bodies[child].soi_radius_m;
                if let Some(t_cross) = refine_crossing(
                    prev_t, t, child, child_soi, rel0, mu, time, soi_body, ephemeris,
                ) {
                    let cross_state = eval_at(t_cross);
                    samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                    return SegmentResult::enter(samples, cross_state, t_cross, child);
                }
            }

            samples.push(build_sample(t, cur_state, soi_body, ephemeris));
            prev_t = t;
            prev_state = cur_state;
        }

        let end_state = prev_state;
        let end_time = target_time;
        SegmentResult {
            samples,
            terminator: if is_stable_orbit {
                SegmentTerminator::StableOrbit
            } else {
                SegmentTerminator::Horizon
            },
            end_state,
            end_time,
            end_soi_body: soi_body,
        }
    }
}

// ---------------------------------------------------------------------------
// Burn segment
// ---------------------------------------------------------------------------

impl KeplerianPropagator {
    fn burn_segment_impl(&self, req: BurnRequest<'_>) -> SegmentResult {
        let BurnRequest {
            state,
            time,
            soi_body,
            target_time,
            burn,
            ephemeris,
            bodies,
        } = req;

        let mut samples: Vec<TrajectorySample> = Vec::new();
        samples.push(build_sample(time, state, soi_body, ephemeris));

        if target_time <= time {
            return SegmentResult {
                samples,
                terminator: SegmentTerminator::BurnEnd { time },
                end_state: state,
                end_time: time,
                end_soi_body: soi_body,
            };
        }

        let mut cur_state = state;
        let mut cur_time = time;

        // SOI body mass cached for this segment. Patched conics: during a
        // burn the ship still feels only the SOI body's gravity; the thrust
        // adds on top.
        let body_mass = ephemeris.query_body(soi_body, time).mass_kg;
        let mu = body_mass * G;
        let soi_radius = bodies[soi_body].soi_radius_m;
        let body_radius = bodies[soi_body].radius_m;
        let threat_bodies: Vec<BodyId> = (0..bodies.len())
            .filter(|&b| b != soi_body && can_intercept(b, soi_body, bodies))
            .collect();

        // Linear refinement of a crossing at fraction `frac` into the current
        // substep, via a shortened RK4 step from (cur_state, cur_time).
        // Sub-second accuracy is fine at typical burn substeps.
        let refine_burn_crossing =
            |cur_state: StateVector, cur_time: f64, h: f64, frac: f64| -> (f64, StateVector) {
                let t_cross = cur_time + frac * h;
                let (cross_state, _) = rk4_burn_step(
                    cur_state,
                    cur_time,
                    frac * h,
                    soi_body,
                    mu,
                    &burn,
                    ephemeris,
                );
                (t_cross, cross_state)
            };

        while cur_time < target_time {
            let h = self.burn_substep_s.min(target_time - cur_time);

            let (next_state, _) =
                rk4_burn_step(cur_state, cur_time, h, soi_body, mu, &burn, ephemeris);
            let next_time = cur_time + h;

            let xings = detect_step_crossings(
                cur_state,
                cur_time,
                next_state,
                next_time,
                soi_body,
                soi_radius,
                body_radius,
                &threat_bodies,
                ephemeris,
                bodies,
            );

            // Pick the refinement fraction. When endpoints bracket the
            // target (one inside, one outside) `inv_lerp` is the standard
            // first-order estimate. When only the cubic Hermite swept-min
            // flagged the crossing (both endpoints on the same side), drop
            // back to the midpoint — burn substeps are 1 s by default so
            // the worst-case error is bounded. The verification step below
            // rejects the crossing if the refined state didn't actually
            // cross the threshold (Hermite false positive).
            let pick_inward_frac = |target_body: BodyId, target_distance: f64| -> f64 {
                let prev_d = (cur_state.position
                    - ephemeris.query_body(target_body, cur_time).position)
                    .length();
                let next_d = (next_state.position
                    - ephemeris.query_body(target_body, next_time).position)
                    .length();
                if prev_d > target_distance && next_d <= target_distance {
                    inv_lerp(prev_d, next_d, target_distance)
                } else {
                    0.5
                }
            };
            let pick_outward_frac = |target_body: BodyId, target_distance: f64| -> f64 {
                let prev_d = (cur_state.position
                    - ephemeris.query_body(target_body, cur_time).position)
                    .length();
                let next_d = (next_state.position
                    - ephemeris.query_body(target_body, next_time).position)
                    .length();
                if prev_d < target_distance && next_d >= target_distance {
                    inv_lerp(prev_d, next_d, target_distance)
                } else {
                    0.5
                }
            };
            let crossed_inward =
                |state: StateVector, t: f64, target_body: BodyId, target_distance: f64| -> bool {
                    let d =
                        (state.position - ephemeris.query_body(target_body, t).position).length();
                    d <= target_distance
                };
            let crossed_outward =
                |state: StateVector, t: f64, target_body: BodyId, target_distance: f64| -> bool {
                    let d =
                        (state.position - ephemeris.query_body(target_body, t).position).length();
                    d >= target_distance
                };

            // Resolve crossings in the same `exit > collision > enter`
            // order as `coast_segment_impl`; see the comment there for the
            // rationale. Verification rejects Hermite false positives.
            if xings.exit {
                let frac = pick_outward_frac(soi_body, soi_radius);
                let (t_cross, cross_state) = refine_burn_crossing(cur_state, cur_time, h, frac);
                if crossed_outward(cross_state, t_cross, soi_body, soi_radius) {
                    samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                    return SegmentResult::exit(samples, cross_state, t_cross, soi_body, bodies);
                }
            }
            if xings.collision {
                let frac = pick_inward_frac(soi_body, body_radius);
                let (t_cross, cross_state) = refine_burn_crossing(cur_state, cur_time, h, frac);
                if crossed_inward(cross_state, t_cross, soi_body, body_radius) {
                    samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                    return SegmentResult::collision(samples, cross_state, t_cross, soi_body);
                }
            }
            if let Some(child) = xings.enter {
                let child_soi = bodies[child].soi_radius_m;
                let frac = pick_inward_frac(child, child_soi);
                let (t_cross, cross_state) = refine_burn_crossing(cur_state, cur_time, h, frac);
                if crossed_inward(cross_state, t_cross, child, child_soi) {
                    samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                    return SegmentResult::enter(samples, cross_state, t_cross, child);
                }
            }

            cur_state = next_state;
            cur_time = next_time;
            samples.push(build_sample(cur_time, cur_state, soi_body, ephemeris));
        }

        // Reached target_time — which for burns is typically the burn end.
        let terminator = if (cur_time - burn.end_time).abs() < 1e-6 {
            SegmentTerminator::BurnEnd { time: cur_time }
        } else {
            SegmentTerminator::Horizon
        };
        SegmentResult {
            samples,
            terminator,
            end_state: cur_state,
            end_time: cur_time,
            end_soi_body: soi_body,
        }
    }
}

// ---------------------------------------------------------------------------
// RK4 step for finite-burn propagation
// ---------------------------------------------------------------------------

/// One RK4 step under SOI-body gravity plus constant-acceleration thrust in
/// the ship's local prograde/normal/radial frame.
fn rk4_burn_step(
    state: StateVector,
    t: f64,
    h: f64,
    soi_body: BodyId,
    mu: f64,
    burn: &BurnParams,
    ephemeris: &dyn BodyStateProvider,
) -> (StateVector, DVec3) {
    let body_at = |tt: f64| -> BodyState { ephemeris.query_body(soi_body, tt) };
    let ref_at = |tt: f64| -> BodyState { ephemeris.query_body(burn.reference_body, tt) };

    let accel = |pos: DVec3, vel: DVec3, tt: f64| -> DVec3 {
        let body = body_at(tt);
        let r = pos - body.position;
        let r_mag_sq = r.length_squared();
        let grav = if r_mag_sq > 1e4 {
            let r_mag = r_mag_sq.sqrt();
            -mu * r / (r_mag_sq * r_mag)
        } else {
            DVec3::ZERO
        };
        // Inclusive at both ends so RK4's k4 evaluation at `burn.end_time`
        // (when a substep lands exactly there) still picks up the boundary
        // thrust — otherwise the integrator silently under-delivers Δv on
        // the final substep. The burn-segment loop never asks us to step
        // past `burn.end_time`, so doubling up with a following coast
        // segment is not a concern here.
        let thrust = if tt >= burn.start_time && tt <= burn.end_time {
            let rb = ref_at(tt);
            let dir = delta_v_to_world(burn.delta_v_local, vel, pos, rb.position, rb.velocity);
            // Linear mass model: thrust cuts off cleanly once propellant
            // exhausts (`mass <= dry_mass_kg`), so an over-budget burn
            // coasts under gravity alone for the rest of the window
            // instead of diverging.
            let mass = burn.initial_mass_kg - burn.mass_flow_kg_per_s * (tt - burn.start_time);
            if dir.length_squared() > 0.0 && mass > burn.dry_mass_kg {
                let accel_mag = burn.thrust_n / mass;
                dir.normalize() * accel_mag
            } else {
                DVec3::ZERO
            }
        } else {
            DVec3::ZERO
        };
        grav + thrust
    };

    let k1_p = state.velocity;
    let k1_v = accel(state.position, state.velocity, t);

    let p2 = state.position + 0.5 * h * k1_p;
    let v2 = state.velocity + 0.5 * h * k1_v;
    let k2_p = v2;
    let k2_v = accel(p2, v2, t + 0.5 * h);

    let p3 = state.position + 0.5 * h * k2_p;
    let v3 = state.velocity + 0.5 * h * k2_v;
    let k3_p = v3;
    let k3_v = accel(p3, v3, t + 0.5 * h);

    let p4 = state.position + h * k3_p;
    let v4 = state.velocity + h * k3_v;
    let k4_p = v4;
    let k4_v = accel(p4, v4, t + h);

    let new_pos = state.position + (h / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p);
    let new_vel = state.velocity + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);

    (
        StateVector {
            position: new_pos,
            velocity: new_vel,
        },
        k1_v,
    )
}

// ---------------------------------------------------------------------------
// SOI determination & crossing root-finding
// ---------------------------------------------------------------------------

/// Innermost body whose SOI currently contains the ship.
pub fn innermost_soi_body(
    position: DVec3,
    time: f64,
    ephemeris: &dyn BodyStateProvider,
    bodies: &[BodyDefinition],
) -> BodyId {
    let mut best: Option<(BodyId, f64)> = None;
    for (id, def) in bodies.iter().enumerate() {
        let body = ephemeris.query_body(id, time);
        let d = (position - body.position).length();
        if d <= def.soi_radius_m {
            let smaller = best.map(|(_, r)| def.soi_radius_m < r).unwrap_or(true);
            if smaller {
                best = Some((id, def.soi_radius_m));
            }
        }
    }
    best.map(|(id, _)| id).unwrap_or(0)
}

/// Can body `candidate` geometrically intercept the ship while the ship is in
/// `soi`'s SOI? Only bodies whose own SOI lies (partly) inside `soi`'s SOI
/// are relevant — i.e. children of `soi`. This filter keeps the per-sample
/// threat-body loop small in systems with many bodies.
fn can_intercept(candidate: BodyId, soi: BodyId, bodies: &[BodyDefinition]) -> bool {
    // A body can only be entered from within its parent's SOI. So candidate
    // must be a descendant of `soi` (or equal to it — excluded upstream).
    let mut cur = Some(candidate);
    while let Some(id) = cur {
        if id == soi {
            return true;
        }
        cur = bodies.get(id).and_then(|b| b.parent);
    }
    false
}

/// Refine the time in [t_lo, t_hi] where the ship's distance to
/// `target_body` crosses `target_distance`. Used for SOI exit (target =
/// soi_body, distance = soi_radius), SOI entry (target = child, distance =
/// child_soi), and collision (target = soi_body, distance = body_radius).
///
/// Two cases:
/// 1. Endpoints bracket the target distance (one above, one below) — the
///    classic inward/outward sign change. Bisect.
/// 2. Endpoints sit on the same side. The Hermite swept-min/max in
///    [`detect_step_crossings`] flagged a crossing in between, so we
///    golden-section search for the extremum and bisect on the half that
///    actually brackets. Returns `None` if the extremum confirms no real
///    crossing — i.e. Hermite was a false positive — and the caller should
///    treat the step as no-event.
#[allow(clippy::too_many_arguments)]
fn refine_crossing(
    t_lo: f64,
    t_hi: f64,
    target_body: BodyId,
    target_distance: f64,
    rel0: StateVector,
    mu: f64,
    time0: f64,
    soi_body: BodyId,
    ephemeris: &dyn BodyStateProvider,
) -> Option<f64> {
    let f = |t: f64| -> f64 {
        let rel = propagate_kepler(rel0, mu, t - time0);
        let soi_bs = ephemeris.query_body(soi_body, t);
        let ship_pos = soi_bs.position + rel.position;
        let target = ephemeris.query_body(target_body, t);
        (ship_pos - target.position).length() - target_distance
    };
    let f_lo = f(t_lo);
    let f_hi = f(t_hi);

    if f_lo.signum() != f_hi.signum() {
        return Some(bisect_signs(t_lo, t_hi, &f, f_lo));
    }

    // Same sign: f_lo > 0 → both outside target sphere, look for inward dip
    // (min); f_lo < 0 → both inside, look for outward bulge (max). Equal-zero
    // is degenerate — treat as crossing exactly at t_lo.
    if f_lo == 0.0 {
        return Some(t_lo);
    }
    let seek_min = f_lo > 0.0;
    let t_extremum = golden_section_extremum(t_lo, t_hi, &f, seek_min);
    let f_ext = f(t_extremum);
    if f_ext.signum() == f_lo.signum() {
        return None;
    }
    Some(bisect_signs(t_lo, t_extremum, &f, f_lo))
}

/// Standard bisection assuming `f(t_lo)` and `f(t_hi)` have opposite signs.
fn bisect_signs(t_lo: f64, t_hi: f64, f: &impl Fn(f64) -> f64, f_lo: f64) -> f64 {
    let f_lo_sgn = f_lo.signum();
    let mut lo = t_lo;
    let mut hi = t_hi;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let f_mid = f(mid);
        if f_mid.signum() == f_lo_sgn {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1e-6 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Golden-section search on [t_lo, t_hi]. `seek_min = true` finds the
/// minimum; `false` finds the maximum.
fn golden_section_extremum(t_lo: f64, t_hi: f64, f: &impl Fn(f64) -> f64, seek_min: bool) -> f64 {
    const INV_PHI: f64 = 0.618_033_988_749_894_9; // 1/φ
    let mut a = t_lo;
    let mut b = t_hi;
    let mut c = b - INV_PHI * (b - a);
    let mut d = a + INV_PHI * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..60 {
        if (b - a).abs() < 1e-6 {
            break;
        }
        let pick_left = if seek_min { fc < fd } else { fc > fd };
        if pick_left {
            b = d;
            d = c;
            fd = fc;
            c = b - INV_PHI * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

// ---------------------------------------------------------------------------
// Shared step-boundary crossing detection
// ---------------------------------------------------------------------------

/// Which boundaries were crossed between two adjacent sample/substep points.
/// Refinement (locating the exact crossing time) and termination are the
/// caller's responsibility — coast uses [`refine_crossing`], burn uses linear
/// interpolation plus a shortened RK4 step — but the "did we cross?" test is
/// identical in both paths.
#[derive(Debug, Clone, Copy, Default)]
struct StepCrossings {
    /// Ship passed from inside `soi_body`'s SOI to outside.
    exit: bool,
    /// Ship passed through `soi_body`'s surface.
    collision: bool,
    /// Ship entered this child body's SOI (first match wins if several
    /// children are entered in the same step).
    enter: Option<BodyId>,
}

/// Detect SOI/surface crossings on the segment between two propagator
/// samples.
///
/// Two layers:
/// 1. Endpoint distances catch any crossing where one end is inside and the
///    other outside the target sphere — the common case.
/// 2. A cubic Hermite of relative position (using ship − body velocities at
///    both endpoints) gets sampled at three interior points; if the swept
///    minimum dips below the body radius (or below a child SOI) while both
///    endpoints sit outside, we still flag the crossing. This catches the
///    "skip-over-periapsis" failure mode where uniform-time sampling misses
///    a brief sub-surface dip.
///
/// SOI exit only uses the endpoint test — a swept-max excursion outside the
/// SOI that returns inside isn't physically an exit (the ship is still
/// inside at both endpoints), so we don't flag it.
#[allow(clippy::too_many_arguments)]
fn detect_step_crossings(
    prev_state: StateVector,
    prev_t: f64,
    next_state: StateVector,
    next_t: f64,
    soi_body: BodyId,
    soi_radius: f64,
    body_radius: f64,
    threat_bodies: &[BodyId],
    ephemeris: &dyn BodyStateProvider,
    bodies: &[BodyDefinition],
) -> StepCrossings {
    let mut out = StepCrossings::default();
    let h = next_t - prev_t;

    let prev_soi_bs = ephemeris.query_body(soi_body, prev_t);
    let next_soi_bs = ephemeris.query_body(soi_body, next_t);
    let q0 = prev_state.position - prev_soi_bs.position;
    let q1 = next_state.position - next_soi_bs.position;
    let qv0 = prev_state.velocity - prev_soi_bs.velocity;
    let qv1 = next_state.velocity - next_soi_bs.velocity;
    let prev_d_sq = q0.length_squared();
    let next_d_sq = q1.length_squared();
    let (interior_min_sq, _) = swept_dist_sq_extremes(q0, qv0, q1, qv1, h);
    let segment_min_sq = prev_d_sq.min(next_d_sq).min(interior_min_sq);

    if soi_radius.is_finite() {
        let r_sq = soi_radius * soi_radius;
        if prev_d_sq < r_sq && next_d_sq >= r_sq {
            out.exit = true;
        }
    }

    if body_radius > 0.0 {
        let r_sq = body_radius * body_radius;
        if prev_d_sq > r_sq && (next_d_sq <= r_sq || segment_min_sq <= r_sq) {
            out.collision = true;
        }
    }

    for &child in threat_bodies {
        let child_soi = bodies[child].soi_radius_m;
        if !child_soi.is_finite() || child_soi <= 0.0 {
            continue;
        }
        let r_sq = child_soi * child_soi;
        let prev_child = ephemeris.query_body(child, prev_t);
        let next_child = ephemeris.query_body(child, next_t);
        let cq0 = prev_state.position - prev_child.position;
        let cq1 = next_state.position - next_child.position;
        let cqv0 = prev_state.velocity - prev_child.velocity;
        let cqv1 = next_state.velocity - next_child.velocity;
        let prev_child_d_sq = cq0.length_squared();
        let next_child_d_sq = cq1.length_squared();
        let (cinterior_min_sq, _) = swept_dist_sq_extremes(cq0, cqv0, cq1, cqv1, h);
        let child_min_sq = prev_child_d_sq.min(next_child_d_sq).min(cinterior_min_sq);
        if prev_child_d_sq >= r_sq && (next_child_d_sq < r_sq || child_min_sq < r_sq) {
            out.enter = Some(child);
            break;
        }
    }

    out
}

/// Cubic Hermite interpolant of `(p0, v0)` at s=0 and `(p1, v1)` at s=1,
/// where `h` is the time interval (so the velocities are honored as
/// position/time, not position/parameter).
#[inline]
fn hermite_cubic(p0: DVec3, v0: DVec3, p1: DVec3, v1: DVec3, h: f64, s: f64) -> DVec3 {
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    h00 * p0 + (h10 * h) * v0 + h01 * p1 + (h11 * h) * v1
}

/// Min and max of `|q(s)|²` along the cubic Hermite over the open interval
/// `s ∈ (0, 1)`. Sampled at three interior points (s = 0.25, 0.5, 0.75) —
/// dense enough to catch any reasonable curvature given that step caps
/// upstream keep altitude change bounded per step. Endpoints aren't included
/// because callers already have `|q0|²` and `|q1|²`.
#[inline]
fn swept_dist_sq_extremes(q0: DVec3, qv0: DVec3, q1: DVec3, qv1: DVec3, h: f64) -> (f64, f64) {
    let mut min_sq = f64::INFINITY;
    let mut max_sq = 0.0_f64;
    for s in [0.25_f64, 0.5, 0.75] {
        let q = hermite_cubic(q0, qv0, q1, qv1, h, s);
        let d_sq = q.length_squared();
        if d_sq < min_sq {
            min_sq = d_sq;
        }
        if d_sq > max_sq {
            max_sq = d_sq;
        }
    }
    (min_sq, max_sq)
}

// ---------------------------------------------------------------------------
// SegmentResult constructors for the three crossing outcomes
// ---------------------------------------------------------------------------

impl SegmentResult {
    fn exit(
        samples: Vec<TrajectorySample>,
        end_state: StateVector,
        t_cross: f64,
        from: BodyId,
        bodies: &[BodyDefinition],
    ) -> Self {
        let parent = bodies[from].parent.unwrap_or(from);
        Self {
            samples,
            terminator: SegmentTerminator::SoiExit {
                from,
                to: parent,
                time: t_cross,
            },
            end_state,
            end_time: t_cross,
            end_soi_body: parent,
        }
    }

    fn collision(
        samples: Vec<TrajectorySample>,
        end_state: StateVector,
        t_cross: f64,
        body: BodyId,
    ) -> Self {
        Self {
            samples,
            terminator: SegmentTerminator::Collision {
                body,
                time: t_cross,
            },
            end_state,
            end_time: t_cross,
            end_soi_body: body,
        }
    }

    fn enter(
        samples: Vec<TrajectorySample>,
        end_state: StateVector,
        t_cross: f64,
        child: BodyId,
    ) -> Self {
        Self {
            samples,
            terminator: SegmentTerminator::SoiEnter {
                body: child,
                time: t_cross,
            },
            end_state,
            end_time: t_cross,
            end_soi_body: child,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sample times for a coast segment, distributed to give smooth rendering at
/// all parts of the orbit.
///
/// Uniform-time sampling equals uniform mean anomaly, which is sparse near
/// periapsis (where the ship moves fastest) — this makes an eccentric orbit
/// look like a V near the body and a ruler line on the apoapsis arm.
/// Sampling uniformly in *eccentric* anomaly places samples proportional to
/// √((a sin E)² + (b cos E)²), i.e. dense where the curve actually curves:
/// at both apsides. Hyperbolic orbits get the analogous distribution in H.
///
/// For degenerate cases (parabolic, span beyond one orbit, degenerate state)
/// we fall back to uniform-time.
fn coast_sample_times(
    rel0: StateVector,
    mu: f64,
    time0: f64,
    target_time: f64,
    n: usize,
) -> Vec<f64> {
    let uniform_time = |n: usize| -> Vec<f64> {
        (0..=n)
            .map(|i| time0 + (i as f64 / n as f64) * (target_time - time0))
            .collect()
    };
    if n == 0 || target_time <= time0 {
        return vec![time0, target_time];
    }
    let Some(el) = cartesian_to_elements(rel0, mu) else {
        return uniform_time(n);
    };
    let e = el.eccentricity;
    let a = el.semi_major_axis_m;
    let nu0 = el.true_anomaly_rad;
    if !a.is_finite() || (e - 1.0).abs() < 1e-9 {
        return uniform_time(n);
    }

    if e < 1.0 && a > 0.0 {
        // Elliptic. Advance by mean anomaly via uniform E stepping.
        let mean_motion = (mu / a.powi(3)).sqrt();
        if mean_motion <= 0.0 {
            return uniform_time(n);
        }
        let big_e0 = eccentric_from_true_elliptic(e, nu0);
        let m0 = big_e0 - e * big_e0.sin();
        let total_dm = mean_motion * (target_time - time0);
        // Bail out if the span covers more than ~one orbit; the unwrap is
        // annoying to get right and the stable-orbit path already caps us
        // at exactly one period.
        if total_dm.abs() > std::f64::consts::TAU * 1.01 {
            return uniform_time(n);
        }
        let m1 = m0 + total_dm;
        let big_e1 = solve_kepler_elliptic_continuous(e, big_e0, m0, m1);
        (0..=n)
            .map(|i| {
                let frac = i as f64 / n as f64;
                let big_e_i = big_e0 + frac * (big_e1 - big_e0);
                let m_i = big_e_i - e * big_e_i.sin();
                time0 + (m_i - m0) / mean_motion
            })
            .collect()
    } else if e > 1.0 && a < 0.0 {
        // Hyperbolic: uniform H.
        let mean_motion = (mu / (-a).powi(3)).sqrt();
        if mean_motion <= 0.0 {
            return uniform_time(n);
        }
        let h0 = hyperbolic_from_true(e, nu0);
        let n0 = e * h0.sinh() - h0;
        let total_dn = mean_motion * (target_time - time0);
        let n1 = n0 + total_dn;
        let h1 = solve_kepler_hyperbolic(e, n1);
        (0..=n)
            .map(|i| {
                let frac = i as f64 / n as f64;
                let h_i = h0 + frac * (h1 - h0);
                let n_i = e * h_i.sinh() - h_i;
                time0 + (n_i - n0) / mean_motion
            })
            .collect()
    } else {
        uniform_time(n)
    }
}

/// Elliptic Kepler solve that stays on the revolution following the anchor
/// `big_e0`.
///
/// [`solve_kepler_elliptic`] wraps `M` into `[-π, π]` for Newton-Raphson
/// convergence — the right call for any single-shot solve, since the result
/// modulo `2π` is unique. But [`coast_sample_times`] linearly interpolates
/// `E` between an anchor `big_e0` and a horizon `big_e1`, so it needs the
/// same revolution at both endpoints; otherwise samples jump across a
/// `2π` discontinuity mid-orbit. We snap the wrapped result to the `2π`
/// multiple closest to `big_e0 + (m1 - m0)` (the linear estimate).
fn solve_kepler_elliptic_continuous(e: f64, big_e0: f64, m0: f64, m1: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let big_e1_wrapped = solve_kepler_elliptic(e, m1);
    let expected = big_e0 + (m1 - m0);
    let shift = ((expected - big_e1_wrapped) / tau).round();
    big_e1_wrapped + shift * tau
}

/// Build a sample with `ref_pos` already cached. The ephemeris query here is
/// cheap (single Kepler chain) and saves the renderer a per-frame query per
/// sample; it is also the natural place to compute it, since the propagator
/// has the SOI body at hand.
#[inline]
fn build_sample(
    time: f64,
    state: StateVector,
    soi_body: BodyId,
    ephemeris: &dyn BodyStateProvider,
) -> TrajectorySample {
    let ref_pos = ephemeris.query_body(soi_body, time).position;
    TrajectorySample {
        time,
        position: state.position,
        velocity: state.velocity,
        anchor_body: soi_body,
        ref_pos,
    }
}

#[inline]
fn inv_lerp(a: f64, b: f64, target: f64) -> f64 {
    let denom = b - a;
    if denom.abs() < 1e-30 {
        0.5
    } else {
        ((target - a) / denom).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patched_conics::PatchedConics;
    use crate::types::{
        BodyDefinition, BodyKind, OrbitalElements, ShipDefinition, SolarSystemDefinition,
    };
    use std::collections::HashMap;

    const SUN_GM: f64 = 1.327_124_4e20;
    const EARTH_GM: f64 = G * 5.972e24;
    const AU: f64 = 1.496e11;

    fn sun_earth_system() -> (SolarSystemDefinition, PatchedConics) {
        let sun_mass = SUN_GM / G;
        let earth_mass = 5.972e24;
        let sun = BodyDefinition {
            id: 0,
            name: "Sun".into(),
            kind: BodyKind::Star,
            parent: None,
            mass_kg: sun_mass,
            radius_m: 6.957e8,
            color: [1.0, 1.0, 0.0],
            albedo: 1.0,
            rotation_period_s: 0.0,
            axial_tilt_rad: 0.0,
            gm: SUN_GM,
            soi_radius_m: f64::INFINITY,
            orbital_elements: None,
            generator: None,
            atmosphere: None,
            terrestrial_atmosphere: None,
            rings: None,
        };
        let earth = BodyDefinition {
            id: 1,
            name: "Earth".into(),
            kind: BodyKind::Planet,
            parent: Some(0),
            mass_kg: earth_mass,
            radius_m: 6.371e6,
            color: [0.0, 0.5, 1.0],
            albedo: 0.3,
            rotation_period_s: 86_400.0,
            axial_tilt_rad: 0.0,
            gm: EARTH_GM,
            soi_radius_m: AU * (earth_mass / sun_mass).powf(0.4),
            orbital_elements: Some(OrbitalElements {
                semi_major_axis_m: AU,
                eccentricity: 0.0,
                inclination_rad: 0.0,
                lon_ascending_node_rad: 0.0,
                arg_periapsis_rad: 0.0,
                true_anomaly_rad: 0.0,
            }),
            generator: None,
            atmosphere: None,
            terrestrial_atmosphere: None,
            rings: None,
        };
        let mut name_to_id = HashMap::new();
        name_to_id.insert("Sun".into(), 0);
        name_to_id.insert("Earth".into(), 1);
        let system = SolarSystemDefinition {
            name: "Test".into(),
            bodies: vec![sun, earth],
            ship: ShipDefinition {
                initial_state: StateVector {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                },
            },
            name_to_id,
        };
        let pc = PatchedConics::new(&system, 3.156e9);
        (system, pc)
    }

    #[test]
    fn soi_body_of_finds_earth_from_leo() {
        let (system, pc) = sun_earth_system();
        let earth = pc.query_body(1, 0.0);
        let leo = earth.position + DVec3::new(7.0e6, 0.0, 0.0);
        let propagator = KeplerianPropagator::default();
        let id = propagator.soi_body_of(leo, 0.0, &pc, &system.bodies);
        assert_eq!(id, 1);
    }

    #[test]
    fn soi_body_of_falls_back_to_star_far_from_everything() {
        let (system, pc) = sun_earth_system();
        let deep = DVec3::new(1e13, 0.0, 0.0);
        let propagator = KeplerianPropagator::default();
        let id = propagator.soi_body_of(deep, 0.0, &pc, &system.bodies);
        assert_eq!(id, 0);
    }

    #[test]
    fn coast_circular_orbit_closes_after_period() {
        let (system, pc) = sun_earth_system();
        let earth = pc.query_body(1, 0.0);
        let r = 7.0e6;
        let v = (EARTH_GM / r).sqrt();
        let state = StateVector {
            position: earth.position + DVec3::new(r, 0.0, 0.0),
            velocity: earth.velocity + DVec3::new(0.0, 0.0, v),
        };
        let period = std::f64::consts::TAU * (r.powi(3) / EARTH_GM).sqrt();
        let propagator = KeplerianPropagator::default();
        let result = propagator.coast_segment(CoastRequest {
            state,
            time: 0.0,
            soi_body: 1,
            target_time: period * 5.0,
            stop_on_stable_orbit: true,
            sample_count_hint: 64,
            ephemeris: &pc,
            bodies: &system.bodies,
        });
        assert!(matches!(result.terminator, SegmentTerminator::StableOrbit));
        // End time is approximately one period.
        assert!((result.end_time - period).abs() / period < 1e-3);
        // Closure must be checked in Earth's frame — Earth itself has moved
        // along its heliocentric orbit during that period.
        let earth_end = pc.query_body(1, result.end_time);
        let rel_end = result.end_state.position - earth_end.position;
        let rel_start = state.position - earth.position;
        let pos_err = (rel_end - rel_start).length() / r;
        assert!(pos_err < 1e-3, "stable orbit didn't close: err={pos_err}");
    }

    #[test]
    fn coast_collision_detected() {
        let (system, pc) = sun_earth_system();
        let earth = pc.query_body(1, 0.0);
        // Highly-eccentric orbit whose periapsis lies inside Earth: starts
        // at 1e7 m with 50% of circular speed → periapsis ≈ r/7 ≈ 1.4e6 m
        // (well below the 6.37e6 m surface). Tangential (non-radial) so the
        // propagator's Kepler path handles it — radial plunges hit the
        // parabolic fallback and are a separate case.
        let r = 1.0e7;
        let v = 0.5 * (EARTH_GM / r).sqrt();
        let state = StateVector {
            position: earth.position + DVec3::new(r, 0.0, 0.0),
            velocity: earth.velocity + DVec3::new(0.0, 0.0, v),
        };
        let propagator = KeplerianPropagator::default();
        let result = propagator.coast_segment(CoastRequest {
            state,
            time: 0.0,
            soi_body: 1,
            target_time: 20_000.0,
            stop_on_stable_orbit: false,
            sample_count_hint: 64,
            ephemeris: &pc,
            bodies: &system.bodies,
        });
        match result.terminator {
            SegmentTerminator::Collision { body, .. } => assert_eq!(body, 1),
            other => panic!("expected collision, got {:?}", other),
        }
    }

    #[test]
    fn coast_hyperbolic_exits_soi() {
        let (system, pc) = sun_earth_system();
        let earth = pc.query_body(1, 0.0);
        let r = 7.0e6;
        let v_circ = (EARTH_GM / r).sqrt();
        let state = StateVector {
            position: earth.position + DVec3::new(r, 0.0, 0.0),
            velocity: earth.velocity + DVec3::new(0.0, 0.0, v_circ * 2.0),
        };
        let propagator = KeplerianPropagator::default();
        let result = propagator.coast_segment(CoastRequest {
            state,
            time: 0.0,
            soi_body: 1,
            target_time: 1.0e8,
            stop_on_stable_orbit: false,
            sample_count_hint: 128,
            ephemeris: &pc,
            bodies: &system.bodies,
        });
        match result.terminator {
            SegmentTerminator::SoiExit { from, to, .. } => {
                assert_eq!(from, 1);
                assert_eq!(to, 0);
            }
            other => panic!("expected SoiExit, got {:?}", other),
        }
    }

    #[test]
    fn eccentric_coast_samples_densify_near_periapsis() {
        // Highly eccentric bound orbit. Uniform-time sampling puts most
        // samples near apoapsis (slow motion); uniform-E inverts that. We
        // verify by checking that the minimum sample-to-sample distance
        // occurs near periapsis, not on the apoapsis arm.
        let (system, pc) = sun_earth_system();
        let earth = pc.query_body(1, 0.0);
        // Periapsis 8e6 m (comfortably above Earth surface), apoapsis 7.2e7
        // → e = (ra − rp) / (ra + rp) = 0.8. a = (ra + rp)/2 = 4e7.
        let a = 4.0e7;
        let e = 0.8;
        let rp = a * (1.0 - e);
        let vp = (EARTH_GM * (1.0 + e) / rp).sqrt();
        let state = StateVector {
            position: earth.position + DVec3::new(rp, 0.0, 0.0),
            velocity: earth.velocity + DVec3::new(0.0, 0.0, vp),
        };
        let propagator = KeplerianPropagator::default();
        let period = std::f64::consts::TAU * (a.powi(3) / EARTH_GM).sqrt();
        let result = propagator.coast_segment(CoastRequest {
            state,
            time: 0.0,
            soi_body: 1,
            target_time: period * 2.0,
            stop_on_stable_orbit: true,
            sample_count_hint: 64,
            ephemeris: &pc,
            bodies: &system.bodies,
        });
        assert!(matches!(result.terminator, SegmentTerminator::StableOrbit));

        // Sample distances to Earth, pick the sample closest to periapsis.
        let dists: Vec<f64> = result
            .samples
            .iter()
            .map(|s| {
                let body = pc.query_body(1, s.time);
                (s.position - body.position).length()
            })
            .collect();
        let peri_idx = dists
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let apo_idx = dists
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        // Neighbouring-sample spacing near periapsis vs near apoapsis.
        let spacing_at = |i: usize| -> f64 {
            let prev = if i == 0 {
                result.samples.len() - 1
            } else {
                i - 1
            };
            (result.samples[i].position - result.samples[prev].position).length()
        };
        let peri_spacing = spacing_at(peri_idx);
        let apo_spacing = spacing_at(apo_idx);

        // Under uniform-E sampling the two spacings should be of the same
        // order (arc-length-ish). Under uniform-time they'd differ by a
        // factor of (1+e)²/(1-e)² ≈ 81 for e=0.8. Require the ratio be
        // well below that uniform-time factor.
        let ratio = apo_spacing / peri_spacing;
        assert!(
            ratio < 10.0,
            "periapsis undersampled: apo/peri spacing ratio = {ratio}"
        );
    }

    #[test]
    fn burn_raises_orbit() {
        let (system, pc) = sun_earth_system();
        let earth = pc.query_body(1, 0.0);
        let r = 7.0e6;
        let v = (EARTH_GM / r).sqrt();
        let state = StateVector {
            position: earth.position + DVec3::new(r, 0.0, 0.0),
            velocity: earth.velocity + DVec3::new(0.0, 0.0, v),
        };
        let propagator = KeplerianPropagator::default();
        // Pick `thrust_n` and `initial_mass_kg` so the starting acceleration
        // is 10 m/s² (matches the constant-accel value the legacy test used).
        let burn = BurnParams {
            delta_v_local: DVec3::new(100.0, 0.0, 0.0), // prograde
            reference_body: 1,
            thrust_n: 100_000.0,
            initial_mass_kg: 10_000.0,
            mass_flow_kg_per_s: 30.0,
            dry_mass_kg: 1_000.0,
            start_time: 0.0,
            end_time: 10.0,
        };
        let result = propagator.burn_segment(BurnRequest {
            state,
            time: 0.0,
            soi_body: 1,
            target_time: 10.0,
            burn,
            ephemeris: &pc,
            bodies: &system.bodies,
        });
        assert!(matches!(
            result.terminator,
            SegmentTerminator::BurnEnd { .. }
        ));
        // Speed increased — orbit raised.
        let v0 = (state.velocity - earth.velocity).length();
        let v1 = (result.end_state.velocity - earth.velocity).length();
        assert!(v1 > v0, "burn did not increase orbital speed: {v0} -> {v1}");
    }

    /// End-to-end check that the rocket equation is honored: a long
    /// straight-line burn (no gravity) should achieve `Δv = ve · ln(m0/mf)`
    /// when integrated for the duration that [`crate::maneuver::burn_duration`]
    /// returns for that target Δv.
    #[test]
    fn burn_segment_achieves_target_delta_v() {
        use crate::maneuver::burn_duration;

        // Build a synthetic massless body so gravity is effectively zero —
        // we only want to measure thrust integration accuracy.
        let mut bodies = vec![BodyDefinition {
            id: 0,
            name: "Origin".into(),
            kind: BodyKind::Star,
            parent: None,
            mass_kg: 0.0,
            radius_m: 0.0,
            color: [1.0; 3],
            albedo: 1.0,
            rotation_period_s: 0.0,
            axial_tilt_rad: 0.0,
            gm: 0.0,
            soi_radius_m: f64::INFINITY,
            orbital_elements: None,
            generator: None,
            atmosphere: None,
            terrestrial_atmosphere: None,
            rings: None,
        }];
        bodies[0].id = 0;
        let mut name_to_id = HashMap::new();
        name_to_id.insert("Origin".into(), 0);
        let system = SolarSystemDefinition {
            name: "Massless".into(),
            bodies,
            ship: ShipDefinition {
                initial_state: StateVector {
                    position: DVec3::new(1e9, 0.0, 0.0),
                    velocity: DVec3::new(0.0, 0.0, 1.0),
                },
            },
            name_to_id,
        };
        let pc = PatchedConics::new(&system, 3.156e9);

        let target_dv = 500.0;
        let thrust_n = 250_000.0;
        let initial_mass_kg = 10_000.0;
        let mass_flow_kg_per_s = 75.0;
        let dry_mass_kg = 1_000.0;
        let duration = burn_duration(
            target_dv,
            thrust_n,
            initial_mass_kg,
            mass_flow_kg_per_s,
            dry_mass_kg,
        );
        assert!(duration > 0.0);

        let state = StateVector {
            position: DVec3::new(1e9, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, 1.0), // small prograde so the local frame is well-defined
        };
        let propagator = KeplerianPropagator::default();
        let burn = BurnParams {
            delta_v_local: DVec3::new(1.0, 0.0, 0.0), // direction only — prograde
            reference_body: 0,
            thrust_n,
            initial_mass_kg,
            mass_flow_kg_per_s,
            dry_mass_kg,
            start_time: 0.0,
            end_time: duration,
        };
        let result = propagator.burn_segment(BurnRequest {
            state,
            time: 0.0,
            soi_body: 0,
            target_time: duration,
            burn,
            ephemeris: &pc,
            bodies: &system.bodies,
        });
        assert!(matches!(
            result.terminator,
            SegmentTerminator::BurnEnd { .. }
        ));
        let achieved_dv = (result.end_state.velocity - state.velocity).length();
        let err = (achieved_dv - target_dv).abs() / target_dv;
        assert!(
            err < 1e-3,
            "Δv error {err} (got {achieved_dv}, target {target_dv})"
        );
    }

    /// Multi-period prediction with periapsis below the surface: forces the
    /// `coast_sample_times` uniform-time fallback. Without swept-min CCD or
    /// step capping the sub-surface dip can fall between two outside
    /// endpoints and the propagator continues straight through the body.
    #[test]
    fn coast_collision_detected_on_multi_period_horizon() {
        let (system, pc) = sun_earth_system();
        let earth = pc.query_body(1, 0.0);

        // Highly eccentric orbit. Periapsis 3e6 m (well inside the 6.37e6 m
        // surface), apoapsis 5e7 m → e ≈ 0.887.
        let r_apo = 5.0e7_f64;
        let r_per = 3.0e6_f64;
        let a = 0.5 * (r_apo + r_per);
        let e = (r_apo - r_per) / (r_apo + r_per);
        let v_apo = (EARTH_GM * (1.0 - e) / (a * (1.0 + e))).sqrt();
        let period = std::f64::consts::TAU * (a.powi(3) / EARTH_GM).sqrt();

        let state = StateVector {
            position: earth.position + DVec3::new(r_apo, 0.0, 0.0),
            velocity: earth.velocity + DVec3::new(0.0, 0.0, v_apo),
        };
        let propagator = KeplerianPropagator::default();
        let result = propagator.coast_segment(CoastRequest {
            state,
            time: 0.0,
            soi_body: 1,
            // > 1.01 periods so `coast_sample_times` returns uniform-time
            // (not uniform-E) — the failure mode we're guarding against.
            target_time: period * 2.5,
            stop_on_stable_orbit: false,
            sample_count_hint: 16,
            ephemeris: &pc,
            bodies: &system.bodies,
        });
        match result.terminator {
            SegmentTerminator::Collision { body, time } => {
                assert_eq!(body, 1);
                // Surface impact has to be on the first periapsis pass —
                // half an orbit from apoapsis.
                assert!(
                    time < period,
                    "collision should be on first orbit (t < {period}), got t = {time}"
                );
            }
            other => panic!("expected collision, got {:?}", other),
        }
    }

    /// `swept_dist_sq_extremes` should report the cubic Hermite's interior
    /// minimum below a threshold even when both endpoints sit above it.
    #[test]
    fn swept_dist_sq_extremes_catches_interior_dip() {
        // Construct a relative-position curve that starts at altitude 10,
        // reaches a minimum near s = 0.5 well below 5, and ends at altitude
        // 10 again. Use velocities to drive the cubic toward and away from
        // the body.
        let q0 = DVec3::new(10.0, 0.0, 0.0);
        let q1 = DVec3::new(-10.0, 0.0, 0.0);
        let qv0 = DVec3::new(-30.0, 0.0, 0.0);
        let qv1 = DVec3::new(-30.0, 0.0, 0.0);
        let h = 1.0;
        let (min_sq, max_sq) = swept_dist_sq_extremes(q0, qv0, q1, qv1, h);
        // At s = 0.5 the cubic passes near the origin; the swept-min should
        // be well under both endpoints (|q0|² = |q1|² = 100).
        assert!(min_sq < 25.0, "interior min² should be small, got {min_sq}");
        assert!(max_sq <= 100.0 + 1e-6);
    }
}
