//! The fly-by-wire command vocabulary.
//!
//! Every control source — pilot stick, SAS/stability hold, nav-mode
//! pointing, the maneuver autopilot — produces a [`ControlDemand`]. The
//! demand is *what* is requested, never *how* it is realized: the
//! attitude controller ([`crate::attitude`]) turns a resolved demand into
//! a normalized torque, and the allocator ([`crate::allocator`])
//! distributes that torque across whatever effectors the craft has
//! (reaction wheels, aero control surfaces). No source ever touches an
//! effector directly.

use glam::DVec3;

use crate::flight::PlaneHoldTarget;

/// What a source wants the craft's attitude to do this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttitudeDemand {
    /// No attitude authority requested — the craft tumbles/coasts freely.
    /// A source that has nothing to say emits this so the arbiter falls
    /// through to a lower-priority source.
    Free,
    /// Hold the craft's current attitude (SAS kill-drift; centered stick
    /// under SAS). The controller captures a target quaternion and drives
    /// a critically-damped PD back to it.
    Hold,
    /// Track a world-frame nose direction (prograde / retrograde / normal
    /// / target / maneuver-node / scheduled-burn pointing). `+Y` body is
    /// the nose.
    PointNose(DVec3),
    /// Fly an explicit pitch attitude and bank angle through the plane FBW law.
    ///
    /// Unlike [`Self::Hold`], this target is supplied by a guidance source
    /// rather than captured from the current aircraft attitude. It still uses
    /// the same coordinated-turn, auto-trim, and AoA-protection path.
    FlightPath(PlaneHoldTarget),
    /// Direct normalized per-axis command in `[-1, 1]` (body frame:
    /// `x` = pitch, `y` = roll, `z` = yaw). Deflected stick. Under SAS
    /// this also slews the hold target so releasing the stick holds the
    /// new attitude; with SAS off it is applied as raw torque.
    Rate(DVec3),
}

impl AttitudeDemand {
    /// `true` when this demand actually asks for attitude authority.
    /// `Free` is the only no-op; a centered-stick `Rate(0)` is treated as
    /// no deflection by the controller but still counts as the pilot
    /// holding the stick.
    pub fn is_active(self) -> bool {
        !matches!(self, AttitudeDemand::Free)
    }
}

/// A single source's request for this frame. `throttle` is `None` when the
/// source has no opinion on engine throttle (e.g. the SAS hold).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlDemand {
    pub attitude: AttitudeDemand,
    pub throttle: Option<f64>,
    /// Normalized ground steering, positive = nose right.
    pub ground_steer: Option<f64>,
    /// Normalized wheel braking in `[0, 1]`.
    pub wheel_brake: Option<f64>,
}

impl ControlDemand {
    /// A demand that requests nothing — used as the arbiter's identity.
    pub const NONE: Self = Self {
        attitude: AttitudeDemand::Free,
        throttle: None,
        ground_steer: None,
        wheel_brake: None,
    };

    pub const fn attitude(attitude: AttitudeDemand) -> Self {
        Self {
            attitude,
            throttle: None,
            ground_steer: None,
            wheel_brake: None,
        }
    }

    pub const fn throttle(value: f64) -> Self {
        Self {
            attitude: AttitudeDemand::Free,
            throttle: Some(value),
            ground_steer: None,
            wheel_brake: None,
        }
    }

    pub const fn autoflight(
        attitude: AttitudeDemand,
        throttle: Option<f64>,
        ground_steer: Option<f64>,
        wheel_brake: Option<f64>,
    ) -> Self {
        Self {
            attitude,
            throttle,
            ground_steer,
            wheel_brake,
        }
    }

    pub const fn ground(steer: f64, brake: f64) -> Self {
        Self {
            attitude: AttitudeDemand::Free,
            throttle: None,
            ground_steer: Some(steer),
            wheel_brake: Some(brake),
        }
    }
}

/// Who is asking. Ordered by authority: a higher-priority source's active
/// attitude demand outranks every lower one. The pilot always wins when
/// the stick is touched; otherwise the autopilot, then nav-mode holds,
/// then bare SAS.
///
/// `Ord` is derived so the arbiter can compare priorities directly; the
/// declaration order *is* the priority order (later = higher).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DemandSource {
    /// Bare SAS stability when nothing else is engaged.
    Sas,
    /// A directional navigation-mode hold (prograde, target, …).
    NavMode,
    /// The scheduled-burn / maneuver autopilot.
    Autopilot,
    /// Direct pilot input. Highest authority: touching the stick always
    /// overrides programmatic attitude.
    Pilot,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pilot_outranks_everything() {
        assert!(DemandSource::Pilot > DemandSource::Autopilot);
        assert!(DemandSource::Autopilot > DemandSource::NavMode);
        assert!(DemandSource::NavMode > DemandSource::Sas);
    }

    #[test]
    fn free_is_the_only_inactive_demand() {
        assert!(!AttitudeDemand::Free.is_active());
        assert!(AttitudeDemand::Hold.is_active());
        assert!(
            AttitudeDemand::FlightPath(PlaneHoldTarget {
                pitch_rad: 0.0,
                bank_rad: 0.0
            })
            .is_active()
        );
        assert!(AttitudeDemand::Rate(DVec3::ZERO).is_active());
    }
}
