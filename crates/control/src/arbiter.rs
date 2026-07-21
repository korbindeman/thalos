//! Priority arbitration across the frame's control demands.
//!
//! Each source pushes one `(DemandSource, ControlDemand)`. Attitude and
//! throttle are arbitrated independently: the highest-priority source with
//! an *active* attitude demand owns attitude; the highest-priority source
//! with a throttle opinion owns throttle. Returning the attitude owner lets
//! the game derive UI gating (which controls to grey out) from the same
//! decision, rather than from a parallel lock flag.

use crate::demand::{AttitudeDemand, ControlDemand, DemandSource};

/// The arbitrated outcome for one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arbitration {
    /// The winning attitude demand (`Free` if no source asked).
    pub attitude: AttitudeDemand,
    /// Which source owns attitude this frame, if any did.
    pub attitude_owner: Option<DemandSource>,
    /// The winning throttle command (`None` if no source asked — the
    /// caller should then leave throttle untouched).
    pub throttle: Option<f64>,
    /// Which source owns throttle this frame, if any did.
    pub throttle_owner: Option<DemandSource>,
}

impl Arbitration {
    pub const NONE: Self = Self {
        attitude: AttitudeDemand::Free,
        attitude_owner: None,
        throttle: None,
        throttle_owner: None,
    };
}

/// Resolve the frame's demands. Ties (same source pushed twice) keep the
/// last entry, but in practice each source pushes at most once.
pub fn arbitrate(demands: &[(DemandSource, ControlDemand)]) -> Arbitration {
    let mut result = Arbitration::NONE;

    for &(source, demand) in demands {
        if demand.attitude.is_active() && result.attitude_owner.is_none_or(|owner| source >= owner)
        {
            result.attitude = demand.attitude;
            result.attitude_owner = Some(source);
        }
        if let Some(throttle) = demand.throttle
            && result.throttle_owner.is_none_or(|owner| source >= owner)
        {
            result.throttle = Some(throttle);
            result.throttle_owner = Some(source);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn pilot_attitude_beats_autopilot() {
        let demands = [
            (
                DemandSource::Autopilot,
                ControlDemand::attitude(AttitudeDemand::PointNose(DVec3::Y)),
            ),
            (
                DemandSource::Pilot,
                ControlDemand::attitude(AttitudeDemand::Rate(DVec3::X)),
            ),
        ];
        let out = arbitrate(&demands);
        assert_eq!(out.attitude, AttitudeDemand::Rate(DVec3::X));
        assert_eq!(out.attitude_owner, Some(DemandSource::Pilot));
    }

    #[test]
    fn free_demand_yields_to_lower_priority() {
        // Pilot is highest priority but emits Free (stick released, SAS
        // off): a lower-priority NavMode hold should still win.
        let demands = [
            (
                DemandSource::Pilot,
                ControlDemand::attitude(AttitudeDemand::Free),
            ),
            (
                DemandSource::NavMode,
                ControlDemand::attitude(AttitudeDemand::PointNose(DVec3::Z)),
            ),
        ];
        let out = arbitrate(&demands);
        assert_eq!(out.attitude_owner, Some(DemandSource::NavMode));
    }

    #[test]
    fn throttle_arbitrated_independently_of_attitude() {
        // Autopilot owns throttle; pilot owns attitude. Independent.
        let demands = [
            (DemandSource::Autopilot, ControlDemand::throttle(1.0)),
            (
                DemandSource::Pilot,
                ControlDemand::attitude(AttitudeDemand::Rate(DVec3::X)),
            ),
        ];
        let out = arbitrate(&demands);
        assert_eq!(out.throttle, Some(1.0));
        assert_eq!(out.throttle_owner, Some(DemandSource::Autopilot));
        assert_eq!(out.attitude_owner, Some(DemandSource::Pilot));
    }

    #[test]
    fn empty_is_none() {
        assert_eq!(arbitrate(&[]), Arbitration::NONE);
    }
}
