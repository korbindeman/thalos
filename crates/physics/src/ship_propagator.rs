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
#[derive(Debug, Clone, Copy)]
pub struct BurnParams {
    /// Direction in the prograde/normal/radial frame. Magnitude is not used
    /// for the force (that's `acceleration`) — only the direction matters.
    pub delta_v_local: DVec3,
    pub reference_body: BodyId,
    pub acceleration: f64,
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
                    let period =
                        std::f64::consts::TAU * (el.semi_major_axis_m.powi(3) / mu).sqrt();
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

        // Track whether the ship has already entered the SOI of a candidate
        // (so we don't re-trigger on the next sample).
        let active_children: Vec<bool> = vec![false; bodies.len()];
        let _ = active_children; // reserved for future per-body re-entry logic

        // Skip index 0 (it equals `time`, already emitted as the start sample).
        for t in sample_times.into_iter().skip(1) {
            let cur_state = eval_at(t);

            // Check for SOI exit between prev_t and t.
            let soi_radius = bodies[soi_body].soi_radius_m;
            if soi_radius.is_finite() {
                let prev_rel = prev_state.position - ephemeris.query_body(soi_body, prev_t).position;
                let cur_rel = cur_state.position - ephemeris.query_body(soi_body, t).position;
                let prev_out = prev_rel.length() - soi_radius;
                let cur_out = cur_rel.length() - soi_radius;
                if prev_out < 0.0 && cur_out >= 0.0 {
                    let t_cross = bisect_distance_sign_change(
                        prev_t,
                        t,
                        soi_body,
                        soi_radius,
                        rel0,
                        mu,
                        time,
                        ephemeris,
                    );
                    let cross_state = eval_at(t_cross);
                    let parent = bodies[soi_body].parent.unwrap_or(soi_body);
                    samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                    return SegmentResult {
                        samples,
                        terminator: SegmentTerminator::SoiExit {
                            from: soi_body,
                            to: parent,
                            time: t_cross,
                        },
                        end_state: cross_state,
                        end_time: t_cross,
                        end_soi_body: parent,
                    };
                }
            }

            // Check for SOI entry into any threat body.
            for &child in &threat_bodies {
                let child_soi = bodies[child].soi_radius_m;
                if !child_soi.is_finite() || child_soi <= 0.0 {
                    continue;
                }
                let prev_child = ephemeris.query_body(child, prev_t);
                let cur_child = ephemeris.query_body(child, t);
                let prev_rel = prev_state.position - prev_child.position;
                let cur_rel = cur_state.position - cur_child.position;
                let prev_d = prev_rel.length();
                let cur_d = cur_rel.length();
                if prev_d >= child_soi && cur_d < child_soi {
                    let t_cross = bisect_body_distance(
                        prev_t,
                        t,
                        child,
                        child_soi,
                        rel0,
                        mu,
                        time,
                        soi_body,
                        ephemeris,
                        DistanceTarget::EnterFrom,
                    );
                    let cross_state = eval_at(t_cross);
                    samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                    return SegmentResult {
                        samples,
                        terminator: SegmentTerminator::SoiEnter {
                            body: child,
                            time: t_cross,
                        },
                        end_state: cross_state,
                        end_time: t_cross,
                        end_soi_body: child,
                    };
                }
            }

            // Check for collision with the SOI body.
            let body_radius = bodies[soi_body].radius_m;
            if body_radius > 0.0 {
                let prev_rel = prev_state.position - ephemeris.query_body(soi_body, prev_t).position;
                let cur_rel = cur_state.position - ephemeris.query_body(soi_body, t).position;
                let prev_d = prev_rel.length();
                let cur_d = cur_rel.length();
                if prev_d > body_radius && cur_d <= body_radius {
                    let t_cross = bisect_body_distance(
                        prev_t,
                        t,
                        soi_body,
                        body_radius,
                        rel0,
                        mu,
                        time,
                        soi_body,
                        ephemeris,
                        DistanceTarget::EnterFrom,
                    );
                    let cross_state = eval_at(t_cross);
                    samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                    return SegmentResult {
                        samples,
                        terminator: SegmentTerminator::Collision {
                            body: soi_body,
                            time: t_cross,
                        },
                        end_state: cross_state,
                        end_time: t_cross,
                        end_soi_body: soi_body,
                    };
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

        while cur_time < target_time {
            let h = self.burn_substep_s.min(target_time - cur_time);

            let body_cur = ephemeris.query_body(soi_body, cur_time);
            eprintln!(
                "[rk4.before] t={:.6} h={:.6} pos_rel=({:.3e},{:.3e},{:.3e}) vel_rel=({:.3e},{:.3e},{:.3e})",
                cur_time, h,
                cur_state.position.x - body_cur.position.x,
                cur_state.position.y - body_cur.position.y,
                cur_state.position.z - body_cur.position.z,
                cur_state.velocity.x - body_cur.velocity.x,
                cur_state.velocity.y - body_cur.velocity.y,
                cur_state.velocity.z - body_cur.velocity.z,
            );
            let (next_state, _) = rk4_burn_step(
                cur_state,
                cur_time,
                h,
                soi_body,
                mu,
                &burn,
                ephemeris,
            );
            let next_time = cur_time + h;
            let body_next = ephemeris.query_body(soi_body, next_time);
            eprintln!(
                "[rk4.after]  t={:.6}      pos_rel=({:.3e},{:.3e},{:.3e}) vel_rel=({:.3e},{:.3e},{:.3e})",
                next_time,
                next_state.position.x - body_next.position.x,
                next_state.position.y - body_next.position.y,
                next_state.position.z - body_next.position.z,
                next_state.velocity.x - body_next.velocity.x,
                next_state.velocity.y - body_next.velocity.y,
                next_state.velocity.z - body_next.velocity.z,
            );

            // SOI / collision checks at substep boundary.
            let prev_soi_rel =
                cur_state.position - ephemeris.query_body(soi_body, cur_time).position;
            let next_soi_rel =
                next_state.position - ephemeris.query_body(soi_body, next_time).position;

            // Exit?
            if soi_radius.is_finite()
                && prev_soi_rel.length() < soi_radius
                && next_soi_rel.length() >= soi_radius
            {
                // Simple linear interpolation — sub-second accuracy is fine.
                let frac = inv_lerp(
                    prev_soi_rel.length(),
                    next_soi_rel.length(),
                    soi_radius,
                );
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
                samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                let parent = bodies[soi_body].parent.unwrap_or(soi_body);
                return SegmentResult {
                    samples,
                    terminator: SegmentTerminator::SoiExit {
                        from: soi_body,
                        to: parent,
                        time: t_cross,
                    },
                    end_state: cross_state,
                    end_time: t_cross,
                    end_soi_body: parent,
                };
            }

            // Collision?
            if body_radius > 0.0
                && prev_soi_rel.length() > body_radius
                && next_soi_rel.length() <= body_radius
            {
                let frac = inv_lerp(
                    prev_soi_rel.length(),
                    next_soi_rel.length(),
                    body_radius,
                );
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
                samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                return SegmentResult {
                    samples,
                    terminator: SegmentTerminator::Collision {
                        body: soi_body,
                        time: t_cross,
                    },
                    end_state: cross_state,
                    end_time: t_cross,
                    end_soi_body: soi_body,
                };
            }

            // SOI entry into a child body?
            for &child in &threat_bodies {
                let child_soi = bodies[child].soi_radius_m;
                if !child_soi.is_finite() || child_soi <= 0.0 {
                    continue;
                }
                let prev_rel =
                    cur_state.position - ephemeris.query_body(child, cur_time).position;
                let next_rel =
                    next_state.position - ephemeris.query_body(child, next_time).position;
                if prev_rel.length() >= child_soi && next_rel.length() < child_soi {
                    let frac = inv_lerp(prev_rel.length(), next_rel.length(), child_soi);
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
                    samples.push(build_sample(t_cross, cross_state, soi_body, ephemeris));
                    return SegmentResult {
                        samples,
                        terminator: SegmentTerminator::SoiEnter {
                            body: child,
                            time: t_cross,
                        },
                        end_state: cross_state,
                        end_time: t_cross,
                        end_soi_body: child,
                    };
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
        let thrust = if tt >= burn.start_time && tt < burn.end_time {
            let rb = ref_at(tt);
            let dir =
                delta_v_to_world(burn.delta_v_local, vel, pos, rb.position, rb.velocity);
            eprintln!(
                "[thrust] tt={:.6} pos_rel=({:.3e},{:.3e},{:.3e}) vel_rel=({:.3e},{:.3e},{:.3e}) dir=({:.3e},{:.3e},{:.3e}) dv_local=({:.3},{:.3},{:.3})",
                tt,
                pos.x - rb.position.x, pos.y - rb.position.y, pos.z - rb.position.z,
                vel.x - rb.velocity.x, vel.y - rb.velocity.y, vel.z - rb.velocity.z,
                dir.x, dir.y, dir.z,
                burn.delta_v_local.x, burn.delta_v_local.y, burn.delta_v_local.z,
            );
            if dir.length_squared() > 0.0 {
                dir.normalize() * burn.acceleration
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

/// Bisect to find the time in [t_lo, t_hi] where `|ship_rel_soi| - soi_radius`
/// changes sign. Assumes a single sign change in the interval (checked by
/// the caller).
#[allow(clippy::too_many_arguments)]
fn bisect_distance_sign_change(
    t_lo: f64,
    t_hi: f64,
    soi_body: BodyId,
    soi_radius: f64,
    rel0: StateVector,
    mu: f64,
    time0: f64,
    ephemeris: &dyn BodyStateProvider,
) -> f64 {
    let mut lo = t_lo;
    let mut hi = t_hi;
    let distance_minus_radius = |t: f64| -> f64 {
        let rel = propagate_kepler(rel0, mu, t - time0);
        let body = ephemeris.query_body(soi_body, t);
        let ship_pos = body.position + rel.position;
        let d = (ship_pos - body.position).length();
        d - soi_radius
    };
    let f_lo = distance_minus_radius(lo);

    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let f_mid = distance_minus_radius(mid);
        if f_mid.signum() == f_lo.signum() {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DistanceTarget {
    /// Sign changes from > target to < target (entering).
    EnterFrom,
}

/// Bisect to find the time in [t_lo, t_hi] where the ship's distance to
/// `target_body` crosses `target_distance`. Used for both SOI entry and
/// collision detection. The sign convention is embedded via the caller's
/// pre-check (we assume exactly one crossing in the interval).
#[allow(clippy::too_many_arguments)]
fn bisect_body_distance(
    t_lo: f64,
    t_hi: f64,
    target_body: BodyId,
    target_distance: f64,
    rel0: StateVector,
    mu: f64,
    time0: f64,
    soi_body: BodyId,
    ephemeris: &dyn BodyStateProvider,
    _target: DistanceTarget,
) -> f64 {
    let mut lo = t_lo;
    let mut hi = t_hi;
    let distance_minus_target = |t: f64| -> f64 {
        let rel = propagate_kepler(rel0, mu, t - time0);
        let soi_bs = ephemeris.query_body(soi_body, t);
        let ship_pos = soi_bs.position + rel.position;
        let target = ephemeris.query_body(target_body, t);
        (ship_pos - target.position).length() - target_distance
    };
    let f_lo = distance_minus_target(lo);

    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let f_mid = distance_minus_target(mid);
        if f_mid.signum() == f_lo.signum() {
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
/// `big_e0`. `solve_kepler_elliptic` wraps `M` into `[-π, π]` for
/// convergence, which would drop `E1` onto the wrong revolution when
/// `M1 - M0` crosses a wrap boundary. We snap the wrapped result back to
/// the 2π multiple closest to `big_e0 + (m1 - m0)` (the linear estimate).
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
    use crate::types::{BodyDefinition, BodyKind, OrbitalElements, ShipDefinition, SolarSystemDefinition};
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
                thrust_acceleration: 0.5,
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
            let prev = if i == 0 { result.samples.len() - 1 } else { i - 1 };
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
        let burn = BurnParams {
            delta_v_local: DVec3::new(100.0, 0.0, 0.0), // prograde
            reference_body: 1,
            acceleration: 10.0,
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
        assert!(matches!(result.terminator, SegmentTerminator::BurnEnd { .. }));
        // Speed increased — orbit raised.
        let v0 = (state.velocity - earth.velocity).length();
        let v1 = (result.end_state.velocity - earth.velocity).length();
        assert!(v1 > v0, "burn did not increase orbital speed: {v0} -> {v1}");
    }

}
