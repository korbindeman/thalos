//! Commanded staging: predict, cut, confirm, separate, ignite.
//!
//! Launch vehicles do not stage by noticing the engine died. Guidance
//! predicts propellant depletion from the flow rate (or commands cutoff on
//! an energy criterion), *commands* cutoff, waits out thrust tail-off, and
//! fires separation only once a set of interlocks pass — thrust below
//! threshold, rates within limits, timer elapsed. Thrust decay is the
//! **confirmation**, never the trigger.
//!
//! The ascent program previously had those two inverted: it staged when
//! `ActivePropulsion::total_thrust_n <= 1.0`, i.e. after the engine had
//! already flamed out. That costs a thrust dropout on every staging event,
//! and because the reactive path also had to throttle to zero and point at
//! local up while it waited for the acknowledgement, it threw the vehicle
//! off its gravity turn every time. Worse, `activate_stage` is gated to 1×
//! warp and unpaused, so a request issued in a state where it could not be
//! served left the program parked in `Wait` at zero throttle indefinitely.
//!
//! Here guidance keeps steering through the whole sequence. Only throttle is
//! surrendered, and only for the few hundred milliseconds the sequence
//! actually needs.
//!
//! The reactive trigger survives as a **backup**: if thrust collapses
//! without the predictor having armed, we still stage, and we log the
//! divergence as `stage_unpredicted`. That event is the falsifiable number
//! that says the predictor is wrong — which is the whole reason to keep it
//! rather than delete it (`CLAUDE.md` · *Observability*).

use bevy::prelude::*;

use crate::staging::StageDemand;

/// How far ahead of predicted burnout the sequence arms and annunciates.
/// Long enough for a countdown to be readable and for a player to
/// understand what is about to happen; short enough that a mid-ascent
/// throttle change re-predicts well before it matters.
pub const STAGE_ARM_LEAD_S: f64 = 5.0;

/// Cutoff is commanded this far before predicted depletion, so the stage
/// shuts down cleanly instead of sputtering to a stop on dregs. Also
/// absorbs the prediction error from a mass-flow estimate that is exact
/// only at constant throttle.
pub const STAGE_CUTOFF_LEAD_S: f64 = 0.2;

/// Thrust tail-off allowance between commanded cutoff and separation.
/// Firing a decoupler into a still-thrusting stage drives the spent stage
/// back into the one above it.
pub const STAGE_TAILOFF_S: f64 = 0.35;

/// Settle time after separation before guidance resumes commanding
/// throttle, so the newly ignited stage's part set and inertia are visible
/// to the controller.
pub const STAGE_IGNITION_SETTLE_S: f64 = 0.25;

/// Angular-rate interlock for separation, rad/s. Two bodies separating
/// while the stack is rotating leave with different angular momentum and
/// can re-contact. ~8.6°/s is loose enough not to fight a normal gravity
/// turn's pitch rate, tight enough to catch a tumble.
pub const STAGE_SEPARATION_RATE_LIMIT_RAD_S: f64 = 0.15;

/// Thrust below this is "shut down" for the separation interlock, N.
pub const STAGE_THRUST_SETTLED_N: f64 = 1.0;

/// Where the commanded staging sequence is.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum StageSequence {
    /// No staging pending. Guidance owns throttle.
    #[default]
    Idle,
    /// Burnout predicted and annunciated; still burning normally.
    Armed { burnout_at_s: f64 },
    /// Cutoff commanded. Throttle closed, waiting out tail-off.
    Cutoff { until_s: f64 },
    /// Interlocks passed and separation requested; awaiting acknowledgement
    /// from the one canonical staging operation.
    Separating { request_id: u64 },
    /// Separated and the next stage enabled; brief settle before guidance
    /// resumes.
    Ignition { until_s: f64 },
    /// The plan is spent — no stage left to fire.
    Exhausted,
}

/// What the sequencer needs the guidance loop to do this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageCommand {
    /// Fly normally; the sequencer is not asserting anything.
    Free,
    /// Hold throttle closed for the sequence. **Attitude is untouched** —
    /// guidance keeps steering, which is the point of the rewrite.
    HoldThrottleClosed,
    /// No stage available. The caller should abort its program.
    Exhausted,
}

/// Live inputs for one sequencer update.
#[derive(Debug, Clone, Copy)]
pub struct StageSequencerInput {
    pub now_s: f64,
    /// Remaining propellant in the currently burning stage, kg.
    pub active_stage_fuel_kg: f64,
    /// Mass flow of the enabled engines **at full throttle**, kg/s.
    pub mass_flow_full_kg_per_s: f64,
    /// The throttle guidance is currently commanding, 0..1.
    pub commanded_throttle: f64,
    /// Total thrust the enabled engines are producing, N.
    pub total_thrust_n: f64,
    /// Vehicle angular rate magnitude, rad/s.
    pub angular_rate_rad_s: f64,
    /// Whether a further stage exists to fire.
    pub stage_available: bool,
}

/// Seconds until the active stage runs dry at the current commanded
/// throttle, or `None` when it is not burning fast enough for the question
/// to mean anything (coasting, or an idle stage).
///
/// Pure, and the one place the prediction is defined. Deliberately uses
/// *commanded* throttle rather than an integrated forecast: guidance
/// re-evaluates every frame, so a constant-throttle estimate refreshed at
/// 60 Hz tracks a changing throttle better than a one-shot integral does,
/// and it cannot silently diverge.
pub fn seconds_to_burnout(
    active_stage_fuel_kg: f64,
    mass_flow_full_kg_per_s: f64,
    commanded_throttle: f64,
) -> Option<f64> {
    let throttle = commanded_throttle.clamp(0.0, 1.0);
    let flow = mass_flow_full_kg_per_s * throttle;
    if flow <= 1.0e-6 || active_stage_fuel_kg < 0.0 {
        return None;
    }
    Some(active_stage_fuel_kg / flow)
}

/// The commanded staging sequencer.
///
/// Owned by whichever flight program is flying; the ascent program drives
/// it every guidance frame. Kept separate from the program so a future
/// descent or transfer program gets the same sequence for free rather than
/// re-deriving one.
#[derive(Resource, Debug, Default)]
pub struct StageSequencer {
    pub state: StageSequence,
    /// Times thrust collapsed without the predictor having armed. A
    /// non-zero count means the prediction is wrong somewhere; it is the
    /// tell, not a cosmetic counter.
    pub unpredicted_events: u32,
    /// Completed separations this program.
    pub completed_events: u32,
}

impl StageSequencer {
    pub fn reset(&mut self) {
        self.state = StageSequence::Idle;
        self.unpredicted_events = 0;
        self.completed_events = 0;
    }

    /// Cancel any in-flight request. Called when the owning program aborts,
    /// so a pending demand cannot be acknowledged into a dead program.
    pub fn cancel(&mut self, demand: &mut StageDemand) {
        if let StageSequence::Separating { request_id } = self.state {
            demand.cancel(request_id);
        }
        self.state = StageSequence::Idle;
    }

    /// Seconds until the armed separation, for the annunciator.
    pub fn armed_in_s(&self, now_s: f64) -> Option<f64> {
        match self.state {
            StageSequence::Armed { burnout_at_s } => Some((burnout_at_s - now_s).max(0.0)),
            StageSequence::Cutoff { .. }
            | StageSequence::Separating { .. }
            | StageSequence::Ignition { .. } => Some(0.0),
            StageSequence::Idle | StageSequence::Exhausted => None,
        }
    }

    /// Advance the sequence one frame and say what guidance should do.
    pub fn update(
        &mut self,
        input: StageSequencerInput,
        demand: &mut StageDemand,
    ) -> StageCommand {
        let now = input.now_s;
        match self.state {
            StageSequence::Exhausted => StageCommand::Exhausted,

            StageSequence::Idle => {
                let remaining = seconds_to_burnout(
                    input.active_stage_fuel_kg,
                    input.mass_flow_full_kg_per_s,
                    input.commanded_throttle,
                );
                // Backup trigger. Thrust is gone but we never armed — the
                // prediction missed (a stage with no usable propellant
                // reading, crossfeed we could not see, a resource the
                // summary does not model). Stage anyway and record it.
                if input.total_thrust_n <= STAGE_THRUST_SETTLED_N && input.commanded_throttle > 0.0
                {
                    self.unpredicted_events += 1;
                    warn!(
                        target: "thalos::diagnostic::staging",
                        event = "stage_unpredicted",
                        active_stage_fuel_kg = input.active_stage_fuel_kg,
                        mass_flow_full_kg_per_s = input.mass_flow_full_kg_per_s,
                        commanded_throttle = input.commanded_throttle,
                        predicted_burnout_s = remaining.unwrap_or(f64::NAN),
                        "thrust collapsed without an armed staging prediction"
                    );
                    self.state = StageSequence::Cutoff {
                        until_s: now + STAGE_TAILOFF_S,
                    };
                    return StageCommand::HoldThrottleClosed;
                }
                match remaining {
                    Some(seconds) if seconds <= STAGE_CUTOFF_LEAD_S => {
                        self.command_cutoff(now, seconds);
                        StageCommand::HoldThrottleClosed
                    }
                    Some(seconds) if seconds <= STAGE_ARM_LEAD_S => {
                        self.state = StageSequence::Armed {
                            burnout_at_s: now + seconds,
                        };
                        StageCommand::Free
                    }
                    _ => StageCommand::Free,
                }
            }

            StageSequence::Armed { .. } => {
                let remaining = seconds_to_burnout(
                    input.active_stage_fuel_kg,
                    input.mass_flow_full_kg_per_s,
                    input.commanded_throttle,
                );
                match remaining {
                    Some(seconds) if seconds <= STAGE_CUTOFF_LEAD_S => {
                        self.command_cutoff(now, seconds);
                        StageCommand::HoldThrottleClosed
                    }
                    // Re-predict every frame: a throttle reduction pushes
                    // burnout out and must un-arm, or the countdown lies.
                    Some(seconds) if seconds <= STAGE_ARM_LEAD_S => {
                        self.state = StageSequence::Armed {
                            burnout_at_s: now + seconds,
                        };
                        StageCommand::Free
                    }
                    _ => {
                        self.state = StageSequence::Idle;
                        StageCommand::Free
                    }
                }
            }

            StageSequence::Cutoff { until_s } => {
                // Interlocks: thrust actually decayed, tail-off elapsed, and
                // the stack is not rotating fast enough for the halves to
                // re-contact. Confirmation, not trigger.
                let thrust_settled = input.total_thrust_n <= STAGE_THRUST_SETTLED_N;
                let tailoff_done = now >= until_s;
                let rates_ok = input.angular_rate_rad_s <= STAGE_SEPARATION_RATE_LIMIT_RAD_S;
                if thrust_settled && tailoff_done && rates_ok {
                    if !input.stage_available {
                        info!(
                            target: "thalos::diagnostic::staging",
                            event = "stage_exhausted",
                            completed_events = self.completed_events,
                            "no stage left to fire"
                        );
                        self.state = StageSequence::Exhausted;
                        return StageCommand::Exhausted;
                    }
                    let request_id = demand.request();
                    info!(
                        target: "thalos::diagnostic::staging",
                        event = "stage_commanded",
                        request_id,
                        angular_rate_rad_s = input.angular_rate_rad_s,
                        "separation commanded"
                    );
                    self.state = StageSequence::Separating { request_id };
                }
                StageCommand::HoldThrottleClosed
            }

            StageSequence::Separating { request_id } => match demand.outcome(request_id) {
                Some(true) => {
                    self.completed_events += 1;
                    info!(
                        target: "thalos::diagnostic::staging",
                        event = "stage_separated",
                        request_id,
                        completed_events = self.completed_events,
                        "separation acknowledged"
                    );
                    self.state = StageSequence::Ignition {
                        until_s: now + STAGE_IGNITION_SETTLE_S,
                    };
                    StageCommand::HoldThrottleClosed
                }
                Some(false) => {
                    warn!(
                        target: "thalos::diagnostic::staging",
                        event = "stage_refused",
                        request_id,
                        "staging request refused"
                    );
                    self.state = StageSequence::Exhausted;
                    StageCommand::Exhausted
                }
                None => StageCommand::HoldThrottleClosed,
            },

            StageSequence::Ignition { until_s } => {
                if now >= until_s {
                    self.state = StageSequence::Idle;
                    return StageCommand::Free;
                }
                StageCommand::HoldThrottleClosed
            }
        }
    }

    fn command_cutoff(&mut self, now_s: f64, predicted_remaining_s: f64) {
        info!(
            target: "thalos::diagnostic::staging",
            event = "stage_cutoff",
            predicted_remaining_s,
            "cutoff commanded on predicted depletion"
        );
        self.state = StageSequence::Cutoff {
            until_s: now_s + STAGE_TAILOFF_S,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn burning(fuel_kg: f64) -> StageSequencerInput {
        StageSequencerInput {
            now_s: 100.0,
            active_stage_fuel_kg: fuel_kg,
            mass_flow_full_kg_per_s: 100.0,
            commanded_throttle: 1.0,
            total_thrust_n: 500_000.0,
            angular_rate_rad_s: 0.01,
            stage_available: true,
        }
    }

    #[test]
    fn burnout_prediction_scales_with_throttle() {
        assert_eq!(seconds_to_burnout(1000.0, 100.0, 1.0), Some(10.0));
        assert_eq!(seconds_to_burnout(1000.0, 100.0, 0.5), Some(20.0));
        // Coasting: the question is meaningless, not "infinite".
        assert_eq!(seconds_to_burnout(1000.0, 100.0, 0.0), None);
        assert_eq!(seconds_to_burnout(1000.0, 0.0, 1.0), None);
    }

    #[test]
    fn arms_then_cuts_as_the_tank_empties() {
        let mut seq = StageSequencer::default();
        let mut demand = StageDemand::default();

        // 10 s of propellant: nothing yet.
        assert_eq!(
            seq.update(burning(1000.0), &mut demand),
            StageCommand::Free
        );
        assert_eq!(seq.state, StageSequence::Idle);

        // 3 s: armed and annunciating, still burning.
        assert_eq!(seq.update(burning(300.0), &mut demand), StageCommand::Free);
        assert!(matches!(seq.state, StageSequence::Armed { .. }));
        assert_eq!(seq.armed_in_s(100.0), Some(3.0));

        // 0.1 s: cutoff commanded, throttle surrendered.
        assert_eq!(
            seq.update(burning(10.0), &mut demand),
            StageCommand::HoldThrottleClosed
        );
        assert!(matches!(seq.state, StageSequence::Cutoff { .. }));
    }

    #[test]
    fn throttling_back_disarms_because_the_countdown_would_lie() {
        let mut seq = StageSequencer::default();
        let mut demand = StageDemand::default();
        seq.update(burning(300.0), &mut demand);
        assert!(matches!(seq.state, StageSequence::Armed { .. }));

        // Same propellant, quarter throttle: burnout is now 12 s out.
        let mut throttled = burning(300.0);
        throttled.commanded_throttle = 0.25;
        assert_eq!(seq.update(throttled, &mut demand), StageCommand::Free);
        assert_eq!(seq.state, StageSequence::Idle);
    }

    #[test]
    fn separation_waits_for_thrust_decay_and_rate_interlocks() {
        let mut seq = StageSequencer::default();
        let mut demand = StageDemand::default();
        seq.state = StageSequence::Cutoff { until_s: 100.0 };

        // Tail-off elapsed but thrust has not decayed: hold.
        let mut still_thrusting = burning(0.0);
        still_thrusting.now_s = 101.0;
        seq.update(still_thrusting, &mut demand);
        assert!(matches!(seq.state, StageSequence::Cutoff { .. }));

        // Thrust gone but tumbling: still hold.
        let mut tumbling = burning(0.0);
        tumbling.now_s = 101.0;
        tumbling.total_thrust_n = 0.0;
        tumbling.angular_rate_rad_s = 1.0;
        seq.update(tumbling, &mut demand);
        assert!(matches!(seq.state, StageSequence::Cutoff { .. }));

        // All interlocks pass: separation commanded.
        let mut clean = burning(0.0);
        clean.now_s = 101.0;
        clean.total_thrust_n = 0.0;
        seq.update(clean, &mut demand);
        assert!(matches!(seq.state, StageSequence::Separating { .. }));
    }

    #[test]
    fn thrust_collapse_still_stages_and_is_recorded_as_a_prediction_miss() {
        let mut seq = StageSequencer::default();
        let mut demand = StageDemand::default();
        // The predictor sees plenty of fuel, but thrust is gone.
        let mut dead = burning(5_000.0);
        dead.total_thrust_n = 0.0;
        assert_eq!(
            seq.update(dead, &mut demand),
            StageCommand::HoldThrottleClosed
        );
        assert!(matches!(seq.state, StageSequence::Cutoff { .. }));
        assert_eq!(
            seq.unpredicted_events, 1,
            "a backup-trigger staging must be visible as a prediction miss"
        );
    }

    #[test]
    fn no_stage_available_reports_exhausted_rather_than_hanging() {
        let mut seq = StageSequencer::default();
        let mut demand = StageDemand::default();
        seq.state = StageSequence::Cutoff { until_s: 100.0 };
        let mut last = burning(0.0);
        last.now_s = 101.0;
        last.total_thrust_n = 0.0;
        last.stage_available = false;
        assert_eq!(seq.update(last, &mut demand), StageCommand::Exhausted);
        assert_eq!(seq.state, StageSequence::Exhausted);
    }

    #[test]
    fn full_sequence_returns_control_to_guidance() {
        let mut seq = StageSequencer::default();
        let mut demand = StageDemand::default();
        seq.state = StageSequence::Cutoff { until_s: 100.0 };

        let mut clean = burning(0.0);
        clean.now_s = 101.0;
        clean.total_thrust_n = 0.0;
        seq.update(clean, &mut demand);
        let StageSequence::Separating { request_id } = seq.state else {
            panic!("expected a separation request");
        };

        // Acknowledge, as `activate_stage` would.
        demand.test_complete(true);
        assert_eq!(demand.outcome(request_id), Some(true));

        clean.now_s = 101.1;
        seq.update(clean, &mut demand);
        assert!(matches!(seq.state, StageSequence::Ignition { .. }));

        clean.now_s = 101.1 + STAGE_IGNITION_SETTLE_S;
        assert_eq!(seq.update(clean, &mut demand), StageCommand::Free);
        assert_eq!(seq.state, StageSequence::Idle);
        assert_eq!(seq.completed_events, 1);
    }
}
