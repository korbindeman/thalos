//! Trajectory events: encounter detection and closest-approach search.
//!
//! Two different mechanisms operate on a propagated trajectory:
//!
//! 1. **Walking encounter detection** on a built [`NumericSegment`] — scans
//!    consecutive samples for SOI entries (anchor-body id change), surface
//!    impacts (distance-to-body < radius), and periapsis/apoapsis (sign flip
//!    of `r · v` in the anchor-body frame). Uses the segment's own Hermite
//!    interpolation to refine the event epoch by bisection.
//!
//! 2. [`closest_approach`] — golden-section search over a [`Trajectory`] vs a
//!    body's ephemeris trajectory, for ghost-body projection.
//!
//! Encounters carry the craft and body states at the event time so the
//! renderer can place ghosts without re-querying.

use super::numeric::NumericSegment;
use super::{FlightPlan, Trajectory};
use crate::body_state_provider::BodyStateProvider;
use crate::types::{BodyDefinition, BodyId, StateVector};

/// Opaque, unique identifier for an encounter within a flight plan.
pub type EncounterId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncounterKind {
    SoiEntry,
    SoiExit,
    Periapsis,
    Apoapsis,
    SurfaceImpact,
    /// Minimum distance approach to a specific target body; not tied to SOI.
    ClosestApproach,
}

/// A detected event along a flight plan.
#[derive(Clone, Copy, Debug)]
pub struct Encounter {
    pub id: EncounterId,
    pub body: BodyId,
    pub epoch: f64,
    pub kind: EncounterKind,
    pub craft_state: StateVector,
    pub body_state: StateVector,
}

/// Alias kept for parity with ~/dev/space naming.
pub type TrajectoryEvent = Encounter;

// ---------------------------------------------------------------------------
// Encounter detection on a numeric segment
// ---------------------------------------------------------------------------

/// Walk a propagated segment and emit encounter records.
///
/// Detection sources:
/// - **SoiEntry / SoiExit**: consecutive samples disagree on `anchor_body`.
/// - **SurfaceImpact**: distance to anchor body drops below its radius
///   (the segment is also flagged via `collision_body`; the event carries the
///   exact epoch).
/// - **Periapsis / Apoapsis**: sign flip of `(r_ship - r_body) · (v_ship -
///   v_body)`. Transition − → + is periapsis, + → − is apoapsis.
///
/// Each event epoch is refined by bisection on the interpolated segment
/// plus a fresh ephemeris query for the body state.
pub(super) fn detect_segment_events(
    segment: &NumericSegment,
    bodies: &[BodyDefinition],
    ephemeris: &dyn BodyStateProvider,
    starting_id: &mut EncounterId,
) -> Vec<Encounter> {
    let mut events = Vec::new();
    if segment.samples.len() < 2 {
        return events;
    }

    let mut body_buf = Vec::with_capacity(ephemeris.body_count());

    for pair in segment.samples.windows(2) {
        let a = pair[0];
        let b = pair[1];

        // 1. SOI transitions.
        if a.anchor_body != b.anchor_body {
            let kind = if is_child_of(bodies, b.anchor_body, a.anchor_body) {
                EncounterKind::SoiEntry
            } else if is_child_of(bodies, a.anchor_body, b.anchor_body) {
                EncounterKind::SoiExit
            } else {
                // Sibling SOI swap — count as an entry into the new body.
                EncounterKind::SoiEntry
            };
            let epoch = 0.5 * (a.time + b.time);
            if let (Some(craft), Some(body)) = (
                segment.state_at(epoch),
                query_body(ephemeris, b.anchor_body, epoch, &mut body_buf),
            ) {
                events.push(Encounter {
                    id: next_id(starting_id),
                    body: b.anchor_body,
                    epoch,
                    kind,
                    craft_state: craft,
                    body_state: body,
                });
            }
        }

        // 2. Periapsis / apoapsis on the anchor body.
        if a.anchor_body == b.anchor_body {
            let body_id = a.anchor_body;
            let rv_a = radial_velocity(&a, ephemeris, &mut body_buf, body_id, a.time);
            let rv_b = radial_velocity(&b, ephemeris, &mut body_buf, body_id, b.time);
            if let (Some(rv_a), Some(rv_b)) = (rv_a, rv_b)
                && rv_a.signum() != rv_b.signum()
                && rv_a != 0.0
            {
                let epoch = bisect_radial_velocity(segment, ephemeris, body_id, a.time, b.time);
                let kind = if rv_a < 0.0 {
                    EncounterKind::Periapsis
                } else {
                    EncounterKind::Apoapsis
                };
                if let (Some(craft), Some(body)) = (
                    segment.state_at(epoch),
                    query_body(ephemeris, body_id, epoch, &mut body_buf),
                ) {
                    events.push(Encounter {
                        id: next_id(starting_id),
                        body: body_id,
                        epoch,
                        kind,
                        craft_state: craft,
                        body_state: body,
                    });
                }
            }
        }
    }

    // 3. Surface impact (if the segment terminated in collision).
    if let Some(hit_id) = segment.collision_body {
        // Walk backwards from the final sample until distance > radius; bisect
        // in between for the first crossing.
        let body_radius = bodies
            .iter()
            .find(|b| b.id == hit_id)
            .map(|b| b.radius_m)
            .unwrap_or(0.0);

        if body_radius > 0.0 && segment.samples.len() >= 2 {
            let n = segment.samples.len();
            let last = &segment.samples[n - 1];
            let prev = &segment.samples[n - 2];
            let epoch = bisect_surface(
                segment,
                ephemeris,
                hit_id,
                body_radius,
                prev.time,
                last.time,
            );
            if let (Some(craft), Some(body)) = (
                segment.state_at(epoch),
                query_body(ephemeris, hit_id, epoch, &mut body_buf),
            ) {
                events.push(Encounter {
                    id: next_id(starting_id),
                    body: hit_id,
                    epoch,
                    kind: EncounterKind::SurfaceImpact,
                    craft_state: craft,
                    body_state: body,
                });
            }
        }
    }

    events
}

fn next_id(counter: &mut EncounterId) -> EncounterId {
    let id = *counter;
    *counter = counter.wrapping_add(1);
    id
}

fn is_child_of(bodies: &[BodyDefinition], child: BodyId, parent: BodyId) -> bool {
    bodies
        .iter()
        .find(|b| b.id == child)
        .and_then(|b| b.parent)
        .map(|p| p == parent)
        .unwrap_or(false)
}

fn query_body(
    ephemeris: &dyn BodyStateProvider,
    body: BodyId,
    time: f64,
    buf: &mut Vec<crate::types::BodyState>,
) -> Option<StateVector> {
    ephemeris.query_into(time, buf);
    let bs = buf.get(body)?;
    Some(StateVector {
        position: bs.position,
        velocity: bs.velocity,
    })
}

fn radial_velocity(
    sample: &crate::types::TrajectorySample,
    ephemeris: &dyn BodyStateProvider,
    buf: &mut Vec<crate::types::BodyState>,
    body_id: BodyId,
    time: f64,
) -> Option<f64> {
    ephemeris.query_into(time, buf);
    let bs = buf.get(body_id)?;
    let r = sample.position - bs.position;
    let v = sample.velocity - bs.velocity;
    Some(r.dot(v))
}

fn bisect_radial_velocity(
    segment: &NumericSegment,
    ephemeris: &dyn BodyStateProvider,
    body_id: BodyId,
    mut lo: f64,
    mut hi: f64,
) -> f64 {
    let mut buf = Vec::new();
    let f = |t: f64, buf: &mut Vec<crate::types::BodyState>| -> f64 {
        let Some(state) = segment.state_at(t) else {
            return 0.0;
        };
        ephemeris.query_into(t, buf);
        let Some(bs) = buf.get(body_id) else {
            return 0.0;
        };
        let r = state.position - bs.position;
        let v = state.velocity - bs.velocity;
        r.dot(v)
    };
    let mut f_lo = f(lo, &mut buf);
    for _ in 0..40 {
        if (hi - lo).abs() < 1e-3 {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let f_mid = f(mid, &mut buf);
        if f_mid == 0.0 {
            return mid;
        }
        if f_lo.signum() == f_mid.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn bisect_surface(
    segment: &NumericSegment,
    ephemeris: &dyn BodyStateProvider,
    body_id: BodyId,
    body_radius: f64,
    mut lo: f64,
    mut hi: f64,
) -> f64 {
    let mut buf = Vec::new();
    // f(t) = |craft - body| - radius. Negative means inside.
    let f = |t: f64, buf: &mut Vec<crate::types::BodyState>| -> f64 {
        let Some(state) = segment.state_at(t) else {
            return 0.0;
        };
        ephemeris.query_into(t, buf);
        let Some(bs) = buf.get(body_id) else {
            return 0.0;
        };
        (state.position - bs.position).length() - body_radius
    };
    let mut f_lo = f(lo, &mut buf);
    for _ in 0..40 {
        if (hi - lo).abs() < 1e-3 {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let f_mid = f(mid, &mut buf);
        if f_mid.abs() < 1.0 {
            return mid;
        }
        if f_lo.signum() == f_mid.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// Closest approach: flight plan vs target body
// ---------------------------------------------------------------------------

/// Find the minimum-distance encounter between a [`FlightPlan`] and a target
/// body's ephemeris, using a coarse sweep + golden-section refinement.
///
/// Returns `None` if the flight plan has no samples, the target body is
/// unknown, or the refinement fails.  The returned `Encounter` carries
/// `EncounterKind::ClosestApproach`.
pub fn closest_approach(
    plan: &FlightPlan,
    target: BodyId,
    ephemeris: &dyn BodyStateProvider,
) -> Option<Encounter> {
    let (start, end) = plan.epoch_range();
    if end <= start {
        return None;
    }

    let mut buf = Vec::new();
    let distance_at = |t: f64, buf: &mut Vec<crate::types::BodyState>| -> Option<f64> {
        let craft = plan.state_at(t)?;
        ephemeris.query_into(t, buf);
        let bs = buf.get(target)?;
        Some((craft.position - bs.position).length())
    };

    // Coarse sweep.
    let samples = 256usize;
    let step = (end - start) / samples as f64;
    let mut best_t = start;
    let mut best_d = f64::MAX;
    for i in 0..=samples {
        let t = start + step * i as f64;
        if let Some(d) = distance_at(t, &mut buf)
            && d < best_d
        {
            best_d = d;
            best_t = t;
        }
    }
    if best_d == f64::MAX {
        return None;
    }

    // Ternary-ish refinement within the bracketing interval.
    let mut lo = (best_t - step).max(start);
    let mut hi = (best_t + step).min(end);
    for _ in 0..40 {
        if (hi - lo) < 0.5 {
            break;
        }
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        let d1 = distance_at(m1, &mut buf).unwrap_or(f64::MAX);
        let d2 = distance_at(m2, &mut buf).unwrap_or(f64::MAX);
        if d1 < d2 {
            hi = m2;
            if d1 < best_d {
                best_d = d1;
                best_t = m1;
            }
        } else {
            lo = m1;
            if d2 < best_d {
                best_d = d2;
                best_t = m2;
            }
        }
    }

    let craft_state = plan.state_at(best_t)?;
    ephemeris.query_into(best_t, &mut buf);
    let bs = buf.get(target)?;
    // ID is derived from the encounter list length + a sentinel offset so
    // transient closest-approach queries don't alias the walk-emitted ids
    // stored on the flight plan.
    let id = plan.encounters.len() as u64 + u32::MAX as u64;
    Some(Encounter {
        id,
        body: target,
        epoch: best_t,
        kind: EncounterKind::ClosestApproach,
        craft_state,
        body_state: StateVector {
            position: bs.position,
            velocity: bs.velocity,
        },
    })
}
