//! Trajectory events, aggregated encounters, and closest-approach scans.
//!
//! Three layers, in order of richness:
//!
//! 1. [`TrajectoryEvent`] — low-level discrete event detected while walking a
//!    propagated segment: SOI entry/exit, periapsis/apoapsis crossings, surface
//!    impact.  Produced by [`detect_segment_events`].
//!
//! 2. [`Encounter`] — aggregated SOI window.  An `SoiEntry` event (optionally
//!    paired with a later `SoiExit`) plus the minimum-distance sample scanned
//!    between them, enriched with osculating orbital elements and a
//!    [`CaptureStatus`].  This is what the UI shows to the player ("captured
//!    flyby, e = 0.41, periapsis altitude = 342 km").  Built by
//!    [`aggregate_encounters`].
//!
//! 3. [`ClosestApproach`] — per-body geometric close pass, for every body the
//!    trajectory does *not* enter the SOI of.  Lets the UI warn "20,000 km
//!    from Mars" even when no encounter occurs.  Built by
//!    [`scan_closest_approaches`].

use std::collections::HashSet;

use super::FlightPlan;
use super::Trajectory;
use super::numeric::NumericSegment;
use crate::body_state_provider::BodyStateProvider;
use crate::orbital_math::cartesian_to_elements;
use crate::types::{BodyDefinition, BodyId, StateVector, TrajectorySample};

/// Opaque, unique identifier for an event or encounter within a flight plan.
pub type EncounterId = u64;

// ---------------------------------------------------------------------------
// Low-level trajectory events
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajectoryEventKind {
    SoiEntry,
    SoiExit,
    Periapsis,
    Apoapsis,
    SurfaceImpact,
}

/// A discrete event detected while walking a propagated segment.
#[derive(Clone, Copy, Debug)]
pub struct TrajectoryEvent {
    pub id: EncounterId,
    pub body: BodyId,
    pub epoch: f64,
    pub kind: TrajectoryEventKind,
    pub craft_state: StateVector,
    pub body_state: StateVector,
    /// Which leg of the flight plan this event was detected on.
    pub leg_index: usize,
}

// Back-compat: historically a lot of event-consuming code used `EncounterKind`.
// Keep the name as an alias so existing references compile unchanged.
pub type EncounterKind = TrajectoryEventKind;

// ---------------------------------------------------------------------------
// Aggregated encounter (rich SOI window)
// ---------------------------------------------------------------------------

/// Outcome classification of a SOI-window encounter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CaptureStatus {
    /// Hyperbolic pass — eccentricity ≥ 1 at closest approach.
    Flyby,
    /// Eccentricity < 1 at closest approach; the craft is on a bound orbit
    /// around the encounter body.
    Captured,
    /// Trajectory crossed the body's surface.
    Impact,
    /// Very low periapsis relative to body radius.  `altitude` is the
    /// periapsis altitude in metres (craft's closest distance minus body
    /// radius, clamped to 0).
    Graze { altitude: f64 },
}

/// Threshold under which an extremely low periapsis is flagged as `Graze`
/// rather than a plain `Flyby`/`Captured`.  Currently 10 % of body radius.
pub const GRAZE_ALTITUDE_FRACTION: f64 = 0.10;

/// An aggregated SOI-window encounter.  Produced by pairing an `SoiEntry`
/// event with the minimum-distance sample inside the window (and an optional
/// `SoiExit`).  Carries everything the UI needs to display the encounter
/// without re-querying the ephemeris.
#[derive(Clone, Copy, Debug)]
pub struct Encounter {
    pub id: EncounterId,
    pub body: BodyId,
    pub leg_index: usize,
    pub entry_epoch: f64,
    /// SOI exit time, if the ship leaves during the propagated horizon.
    pub exit_epoch: Option<f64>,
    /// Time of minimum craft-to-body distance within `[entry_epoch, exit_epoch]`.
    pub closest_epoch: f64,
    /// Distance from craft to body centre at `closest_epoch` (m).
    pub closest_distance: f64,
    /// Altitude of closest approach above the body's surface (m).  Clamped
    /// to zero if the trajectory intersected the surface.
    pub periapsis_altitude: f64,
    /// Relative speed at closest approach (m/s).
    pub relative_velocity: f64,
    /// Eccentricity of the osculating orbit at closest approach.
    pub eccentricity: f64,
    /// Inclination (radians) of the osculating orbit at closest approach,
    /// measured in the body-centred frame.
    pub inclination_rad: f64,
    pub capture: CaptureStatus,
    /// Craft state (absolute coords) at `closest_epoch`.
    pub craft_state: StateVector,
    /// Body state (absolute coords) at `closest_epoch`.
    pub body_state: StateVector,
}

// ---------------------------------------------------------------------------
// Closest approach (non-encounter bodies)
// ---------------------------------------------------------------------------

/// Geometric minimum-distance pass of a flight plan to a body that was never
/// entered (no SOI entry).  One entry per body not covered by an `Encounter`.
#[derive(Clone, Copy, Debug)]
pub struct ClosestApproach {
    pub body: BodyId,
    pub epoch: f64,
    pub distance: f64,
    pub craft_state: StateVector,
    pub body_state: StateVector,
}

// ---------------------------------------------------------------------------
// Event detection on a numeric segment
// ---------------------------------------------------------------------------

/// Walk a propagated segment and emit discrete trajectory events.
///
/// Detection sources:
/// - **SoiEntry / SoiExit**: consecutive samples disagree on `soi_body`.
/// - **SurfaceImpact**: the segment terminated in a collision
///   (`collision_body` set); the event carries the refined surface-crossing
///   epoch.
/// - **Periapsis / Apoapsis**: sign flip of `(r_ship − r_body) · (v_ship −
///   v_body)`. `− → +` is periapsis, `+ → −` is apoapsis.
///
/// Each event epoch is refined by bisection on the segment's Hermite
/// interpolation plus a fresh ephemeris query for the body state.
pub(super) fn detect_segment_events(
    segment: &NumericSegment,
    bodies: &[BodyDefinition],
    ephemeris: &dyn BodyStateProvider,
    starting_id: &mut EncounterId,
    leg_index: usize,
) -> Vec<TrajectoryEvent> {
    let mut events = Vec::new();
    if segment.samples.len() < 2 {
        return events;
    }

    for pair in segment.samples.windows(2) {
        let a = pair[0];
        let b = pair[1];

        // SOI transitions (uses soi_body, not anchor_body, because the
        // rendering anchor is stepped up to the parent planet for moons).
        if a.anchor_body != b.anchor_body {
            let kind = if is_child_of(bodies, b.anchor_body, a.anchor_body) {
                TrajectoryEventKind::SoiEntry
            } else if is_child_of(bodies, a.anchor_body, b.anchor_body) {
                TrajectoryEventKind::SoiExit
            } else {
                TrajectoryEventKind::SoiEntry
            };
            let epoch = 0.5 * (a.time + b.time);
            if let Some(craft) = segment.state_at(epoch) {
                events.push(TrajectoryEvent {
                    id: next_id(starting_id),
                    body: b.anchor_body,
                    epoch,
                    kind,
                    craft_state: craft,
                    body_state: body_state(ephemeris, b.anchor_body, epoch),
                    leg_index,
                });
            }
        }

        // Periapsis / apoapsis on the SOI body.
        if a.anchor_body == b.anchor_body {
            let body_id = a.anchor_body;
            let rv_a = radial_velocity(&a, ephemeris, body_id, a.time);
            let rv_b = radial_velocity(&b, ephemeris, body_id, b.time);
            if rv_a.signum() != rv_b.signum() && rv_a != 0.0 {
                let epoch = bisect_zero(a.time, b.time, |t| {
                    let Some(state) = segment.state_at(t) else {
                        return 0.0;
                    };
                    let bs = ephemeris.query_body(body_id, t);
                    let r = state.position - bs.position;
                    let v = state.velocity - bs.velocity;
                    r.dot(v)
                });
                let kind = if rv_a < 0.0 {
                    TrajectoryEventKind::Periapsis
                } else {
                    TrajectoryEventKind::Apoapsis
                };
                if let Some(craft) = segment.state_at(epoch) {
                    events.push(TrajectoryEvent {
                        id: next_id(starting_id),
                        body: body_id,
                        epoch,
                        kind,
                        craft_state: craft,
                        body_state: body_state(ephemeris, body_id, epoch),
                        leg_index,
                    });
                }
            }
        }
    }

    // Surface impact (segment terminated in collision).
    if let Some(hit_id) = segment.collision_body
        && let Some(body_def) = bodies.get(hit_id)
        && body_def.radius_m > 0.0
        && segment.samples.len() >= 2
    {
        let body_radius = body_def.radius_m;
        let n = segment.samples.len();
        let last = &segment.samples[n - 1];
        let prev = &segment.samples[n - 2];
        let epoch = bisect_zero(prev.time, last.time, |t| {
            let Some(state) = segment.state_at(t) else {
                return 0.0;
            };
            let bs = ephemeris.query_body(hit_id, t);
            (state.position - bs.position).length() - body_radius
        });
        if let Some(craft) = segment.state_at(epoch) {
            events.push(TrajectoryEvent {
                id: next_id(starting_id),
                body: hit_id,
                epoch,
                kind: TrajectoryEventKind::SurfaceImpact,
                craft_state: craft,
                body_state: body_state(ephemeris, hit_id, epoch),
                leg_index,
            });
        }
    }

    events
}

// ---------------------------------------------------------------------------
// Encounter aggregation
// ---------------------------------------------------------------------------

/// Aggregate low-level events into rich `Encounter` windows.
///
/// Each `SoiEntry` event opens a window; the matching `SoiExit` (or the
/// segment end) closes it.  Within the window we scan the underlying segment
/// samples for the minimum-distance point, compute osculating elements at
/// that point, and classify the outcome.
pub(super) fn aggregate_encounters(
    events: &[TrajectoryEvent],
    segments: &[NumericSegment],
    bodies: &[BodyDefinition],
    ephemeris: &dyn BodyStateProvider,
    starting_id: &mut EncounterId,
) -> Vec<Encounter> {
    let mut out = Vec::new();

    for (idx, entry) in events.iter().enumerate() {
        if entry.kind != TrajectoryEventKind::SoiEntry {
            continue;
        }

        // Look ahead for the exit that returns from this entry body to
        // its parent frame. `TrajectoryEvent::body` stores the new frame
        // after a transition, so a Mira exit is recorded as `SoiExit`
        // with body = Thalos.
        let exit_body = bodies.get(entry.body).and_then(|body| body.parent);
        let exit = events[idx + 1..]
            .iter()
            .find(|e| e.kind == TrajectoryEventKind::SoiExit && Some(e.body) == exit_body);

        let window_end = exit.map(|e| e.epoch).unwrap_or(f64::INFINITY);

        // Scan every segment for the minimum-distance sample inside the
        // window.  Most encounters live on one leg, but a long capture can
        // span multiple coast sub-segments.
        let mut best: Option<(f64, TrajectorySample, StateVector)> = None;
        for seg in segments {
            for sample in &seg.samples {
                if sample.time < entry.epoch || sample.time > window_end {
                    continue;
                }
                let bs = ephemeris.query_body(entry.body, sample.time);
                let d_sq = (sample.position - bs.position).length_squared();
                if best.map(|(d, _, _)| d_sq < d * d).unwrap_or(true) {
                    best = Some((
                        d_sq.sqrt(),
                        *sample,
                        StateVector {
                            position: bs.position,
                            velocity: bs.velocity,
                        },
                    ));
                }
            }
        }

        let Some((closest_distance, closest_sample, closest_body_state)) = best else {
            continue;
        };

        let body_def = bodies.get(entry.body);
        let body_radius = body_def.map(|b| b.radius_m).unwrap_or(0.0);
        let body_gm = body_def.map(|b| b.gm).unwrap_or(0.0);

        let craft_state = StateVector {
            position: closest_sample.position,
            velocity: closest_sample.velocity,
        };

        let rel_state = StateVector {
            position: craft_state.position - closest_body_state.position,
            velocity: craft_state.velocity - closest_body_state.velocity,
        };
        let relative_velocity = rel_state.velocity.length();

        let (eccentricity, inclination_rad) = match cartesian_to_elements(rel_state, body_gm) {
            Some(el) => (el.eccentricity, el.inclination_rad),
            None => (0.0, 0.0),
        };

        let periapsis_altitude = (closest_distance - body_radius).max(0.0);
        let impacted = segments.iter().any(|s| {
            s.collision_body == Some(entry.body)
                && s.samples
                    .last()
                    .map(|last| last.time >= entry.epoch && last.time <= window_end)
                    .unwrap_or(false)
        });

        let capture = if impacted {
            CaptureStatus::Impact
        } else if body_radius > 0.0 && periapsis_altitude < body_radius * GRAZE_ALTITUDE_FRACTION {
            CaptureStatus::Graze {
                altitude: periapsis_altitude,
            }
        } else if eccentricity < 1.0 {
            CaptureStatus::Captured
        } else {
            CaptureStatus::Flyby
        };

        out.push(Encounter {
            id: next_id(starting_id),
            body: entry.body,
            leg_index: entry.leg_index,
            entry_epoch: entry.epoch,
            exit_epoch: exit.map(|e| e.epoch),
            closest_epoch: closest_sample.time,
            closest_distance,
            periapsis_altitude,
            relative_velocity,
            eccentricity,
            inclination_rad,
            capture,
            craft_state,
            body_state: closest_body_state,
        });
    }

    out
}

// ---------------------------------------------------------------------------
// Closest-approach scan (every non-encounter body)
// ---------------------------------------------------------------------------

/// Scan every segment sample against every body and produce the minimum
/// distance for bodies that are *not* already covered by an encounter.
///
/// Stars are skipped — a "closest approach to the sun" is rarely useful
/// information in practice and would dominate the list.  Bodies in
/// `exclude` (typically the set of bodies that have an aggregated
/// `Encounter`) are skipped.
pub(super) fn scan_closest_approaches(
    segments: &[NumericSegment],
    bodies: &[BodyDefinition],
    ephemeris: &dyn BodyStateProvider,
    exclude: &HashSet<BodyId>,
) -> Vec<ClosestApproach> {
    use crate::types::BodyKind;

    let mut best: Vec<Option<(f64, TrajectorySample, StateVector)>> =
        vec![None; ephemeris.body_count()];
    let mut body_buf = Vec::with_capacity(ephemeris.body_count());

    for seg in segments {
        for sample in &seg.samples {
            ephemeris.query_into(sample.time, &mut body_buf);
            for body_def in bodies {
                if body_def.kind == BodyKind::Star {
                    continue;
                }
                if exclude.contains(&body_def.id) {
                    continue;
                }
                let Some(bs) = body_buf.get(body_def.id) else {
                    continue;
                };
                let d = (sample.position - bs.position).length();
                let slot = &mut best[body_def.id];
                if slot.map(|(bd, _, _)| d < bd).unwrap_or(true) {
                    *slot = Some((
                        d,
                        *sample,
                        StateVector {
                            position: bs.position,
                            velocity: bs.velocity,
                        },
                    ));
                }
            }
        }
    }

    best.into_iter()
        .enumerate()
        .filter_map(|(body_id, slot)| {
            slot.map(|(distance, sample, body_state)| ClosestApproach {
                body: body_id,
                epoch: sample.time,
                distance,
                craft_state: StateVector {
                    position: sample.position,
                    velocity: sample.velocity,
                },
                body_state,
            })
        })
        .collect()
}

fn next_id(counter: &mut EncounterId) -> EncounterId {
    let id = *counter;
    *counter = counter.wrapping_add(1);
    id
}

fn is_child_of(bodies: &[BodyDefinition], child: BodyId, parent: BodyId) -> bool {
    bodies
        .get(child)
        .and_then(|b| b.parent)
        .map(|p| p == parent)
        .unwrap_or(false)
}

/// Body state at `time`, lifted to a craft-shaped `StateVector`.
fn body_state(ephemeris: &dyn BodyStateProvider, body: BodyId, time: f64) -> StateVector {
    let bs = ephemeris.query_body(body, time);
    StateVector {
        position: bs.position,
        velocity: bs.velocity,
    }
}

fn radial_velocity(
    sample: &TrajectorySample,
    ephemeris: &dyn BodyStateProvider,
    body_id: BodyId,
    time: f64,
) -> f64 {
    let bs = ephemeris.query_body(body_id, time);
    let r = sample.position - bs.position;
    let v = sample.velocity - bs.velocity;
    r.dot(v)
}

/// Bisect to find a zero of `f` in `[lo, hi]`. The caller pre-checks that the
/// interval brackets a sign change. Stops when `|hi − lo| < 1 ms` (tighter
/// than the typical sample interval) or `|f(mid)| < 1` in the units of `f`
/// (well below the resolution of any rendered event).
fn bisect_zero(mut lo: f64, mut hi: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
    let mut f_lo = f(lo);
    for _ in 0..40 {
        if (hi - lo).abs() < 1e-3 {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let f_mid = f(mid);
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
// Target closest approach (on-demand ephemeris-aware search)
// ---------------------------------------------------------------------------

/// Find the minimum-distance encounter between a [`FlightPlan`] and a target
/// body's ephemeris, using a coarse sweep + ternary refinement.
///
/// This is separate from [`scan_closest_approaches`]: the scan walks stored
/// samples and is cheap (O(N_bodies × N_samples)); this function uses the
/// plan's interpolation to subsample between stored points, giving tighter
/// precision for a single on-demand target query.
pub fn closest_approach(
    plan: &FlightPlan,
    target: BodyId,
    ephemeris: &dyn BodyStateProvider,
) -> Option<ClosestApproach> {
    let (start, end) = plan.epoch_range();
    if end <= start {
        return None;
    }

    let distance_at = |t: f64| -> Option<f64> {
        let craft = plan.state_at(t)?;
        let bs = ephemeris.query_body(target, t);
        Some((craft.position - bs.position).length())
    };

    let samples = 256usize;
    let step = (end - start) / samples as f64;
    let mut best_t = start;
    let mut best_d = f64::MAX;
    for i in 0..=samples {
        let t = start + step * i as f64;
        if let Some(d) = distance_at(t)
            && d < best_d
        {
            best_d = d;
            best_t = t;
        }
    }
    if best_d == f64::MAX {
        return None;
    }

    let mut lo = (best_t - step).max(start);
    let mut hi = (best_t + step).min(end);
    for _ in 0..40 {
        if (hi - lo) < 0.5 {
            break;
        }
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        let d1 = distance_at(m1).unwrap_or(f64::MAX);
        let d2 = distance_at(m2).unwrap_or(f64::MAX);
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
    Some(ClosestApproach {
        body: target,
        epoch: best_t,
        distance: best_d,
        craft_state,
        body_state: body_state(ephemeris, target, best_t),
    })
}

#[cfg(test)]
mod tests {
    use glam::DVec3;

    use super::*;
    use crate::types::{BodyKind, BodyState, BodyStates};

    struct StaticProvider {
        states: BodyStates,
    }

    impl BodyStateProvider for StaticProvider {
        fn query_into(&self, _time: f64, out: &mut BodyStates) {
            out.clear();
            out.extend_from_slice(&self.states);
        }

        fn query_body(&self, body_id: BodyId, _time: f64) -> BodyState {
            self.states[body_id]
        }

        fn body_count(&self) -> usize {
            self.states.len()
        }

        fn time_span(&self) -> f64 {
            1_000.0
        }
    }

    fn body(id: BodyId, parent: Option<BodyId>, kind: BodyKind) -> BodyDefinition {
        BodyDefinition {
            id,
            name: format!("body-{id}"),
            kind,
            parent,
            mass_kg: 1.0,
            radius_m: 1.0,
            color: [1.0, 1.0, 1.0],
            rotation_period_s: 0.0,
            axial_tilt_rad: 0.0,
            gm: 1.0,
            soi_radius_m: if parent.is_some() {
                100.0
            } else {
                f64::INFINITY
            },
            orbital_elements: None,
            terrain: thalos_terrain_gen::TerrainConfig::None,
            atmosphere: None,
            terrestrial_atmosphere: None,
            rings: None,
        }
    }

    fn state(position: DVec3) -> StateVector {
        StateVector {
            position,
            velocity: DVec3::X,
        }
    }

    fn sample(time: f64, position: DVec3, anchor_body: BodyId) -> TrajectorySample {
        TrajectorySample {
            time,
            position,
            velocity: DVec3::X,
            anchor_body,
            ref_pos: DVec3::ZERO,
        }
    }

    #[test]
    fn nested_soi_entries_match_parent_frame_exits() {
        let bodies = vec![
            body(0, None, BodyKind::Star),
            body(1, Some(0), BodyKind::Planet),
            body(2, Some(1), BodyKind::Moon),
        ];
        let ephemeris = StaticProvider {
            states: vec![
                BodyState {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                    mass_kg: 1.0,
                },
                BodyState {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                    mass_kg: 1.0,
                },
                BodyState {
                    position: DVec3::ZERO,
                    velocity: DVec3::ZERO,
                    mass_kg: 1.0,
                },
            ],
        };
        let events = vec![
            TrajectoryEvent {
                id: 0,
                body: 1,
                epoch: 1.0,
                kind: TrajectoryEventKind::SoiEntry,
                craft_state: state(DVec3::new(10.0, 0.0, 0.0)),
                body_state: state(DVec3::ZERO),
                leg_index: 0,
            },
            TrajectoryEvent {
                id: 1,
                body: 2,
                epoch: 10.0,
                kind: TrajectoryEventKind::SoiEntry,
                craft_state: state(DVec3::new(3.0, 0.0, 0.0)),
                body_state: state(DVec3::ZERO),
                leg_index: 0,
            },
            TrajectoryEvent {
                id: 2,
                body: 1,
                epoch: 30.0,
                kind: TrajectoryEventKind::SoiExit,
                craft_state: state(DVec3::new(4.0, 0.0, 0.0)),
                body_state: state(DVec3::ZERO),
                leg_index: 0,
            },
            TrajectoryEvent {
                id: 3,
                body: 0,
                epoch: 60.0,
                kind: TrajectoryEventKind::SoiExit,
                craft_state: state(DVec3::new(20.0, 0.0, 0.0)),
                body_state: state(DVec3::ZERO),
                leg_index: 0,
            },
        ];
        let segments = vec![NumericSegment {
            samples: vec![
                sample(5.0, DVec3::new(10.0, 0.0, 0.0), 1),
                sample(20.0, DVec3::new(2.0, 0.0, 0.0), 2),
                sample(40.0, DVec3::new(12.0, 0.0, 0.0), 1),
                sample(55.0, DVec3::new(18.0, 0.0, 0.0), 1),
            ],
            is_stable_orbit: false,
            stable_orbit_start_index: None,
            collision_body: None,
        }];

        let mut next_id = 10;
        let encounters =
            aggregate_encounters(&events, &segments, &bodies, &ephemeris, &mut next_id);

        let planet = encounters
            .iter()
            .find(|encounter| encounter.body == 1)
            .expect("planet encounter should be aggregated");
        let moon = encounters
            .iter()
            .find(|encounter| encounter.body == 2)
            .expect("moon encounter should be aggregated");

        assert_eq!(planet.exit_epoch, Some(60.0));
        assert_eq!(moon.exit_epoch, Some(30.0));
    }
}
