//! Autoflight: the strategic/tactical split.
//!
//! Automation here is two layers, and keeping them separate is the whole
//! point of this module.
//!
//! **Strategic — [`FlightProgram`].** "Get to a 200 km orbit." "Land on
//! runway 09." A program owns *targets, sequencing, and events*: it plans,
//! installs maneuver nodes, commands staging, and decides when the vehicle
//! moves from one phase to the next. It does not fly the ship.
//!
//! **Tactical — the channels.** Attitude and throttle are each owned by
//! exactly one source per frame, resolved by [`thalos_control::arbitrate`].
//! A program *feeds* those channels (a guidance vector, a throttle setting)
//! and *arms* the shared burn executor for its own nodes, but it competes
//! for them on the same terms as the pilot stick and SAS.
//!
//! The previous design fused the two into one `AutoflightMode` enum with
//! four mutually exclusive slots (`Off | Maneuver | Land | Orbit`). That
//! forced three separate places to encode "…but ORBIT is *also* sort of
//! Maneuver" — the executor's enable gate, the control-bus match arm, and
//! the lock table — and let the HUD put the ship in a state where the
//! ascent program held authority while the panel annunciated `MNVR`. Two
//! layers, one arbiter, and locks declared rather than derived make that
//! class of contradiction unrepresentable instead of merely detected.
//!
//! Modelled on transport-category autoflight, which solved this a long time
//! ago: the FMS computes the plan, the autopilot flies a mode, the FMA tells
//! you which mode owns each axis, and modes are *armed* before they are
//! *engaged* so a transition is never a surprise.

use bevy::prelude::*;

/// The one strategic program in command of the vehicle.
///
/// Mutually exclusive by construction: a vehicle is ascending, or landing,
/// or neither. Unlike the enum this replaces, selecting a program says
/// nothing about which tactical channels are engaged — that is resolved
/// per-frame and annunciated in [`AutoflightAnnunciation`].
///
/// Sole writer: `thalos_runtime::autoflight::update_flight_program`.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum FlightProgram {
    /// No program. The pilot flies; the burn executor may still be armed
    /// for hand-placed maneuver nodes (that is `BurnArm::Pilot`, not a
    /// program — executing a node you placed yourself is not a strategy).
    #[default]
    None,
    /// Launch-to-orbit. Owns the pitch program, staging sequence, MECO
    /// criterion, and the circularisation nodes it installs.
    Ascent,
    /// Approach and landing. Owns the route, the descent profile, and the
    /// gear/flap/brake sequence.
    Landing,
}

impl FlightProgram {
    /// Short annunciator label, sized for the FMA's program column.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "----",
            Self::Ascent => "ASCENT",
            Self::Landing => "LAND",
        }
    }

    pub fn engaged(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Who armed the shared scheduled-burn executor.
///
/// The executor itself is producer-agnostic and always available; this says
/// on whose behalf it is running. Splitting it out is what lets a program
/// and the pilot use the *same* executor without either one having to
/// pretend to be the other — the defect that made `AutoflightMode::Orbit`
/// have to alias `Maneuver` in three places.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum BurnArm {
    /// Disarmed. Scheduled nodes are displayed but never flown.
    Off,
    /// Armed by the pilot for hand-placed maneuver nodes — the `MNVR`
    /// button. The historical default, preserved so a fresh session still
    /// executes nodes without the player arming anything.
    #[default]
    Pilot,
    /// Armed by the engaged [`FlightProgram`] for nodes it installed
    /// itself. Disarms with the program.
    Program,
}

impl BurnArm {
    pub fn armed(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// Which source owns attitude this frame, at annunciation granularity.
///
/// Finer than `thalos_control::DemandSource`, which cannot distinguish a
/// program's guidance law from the burn executor's node pointing — both
/// arrive as `DemandSource::Autopilot`. That distinction is exactly what a
/// pilot needs to see, so it is tracked here rather than pushed into the
/// arbiter's priority enum, where it would imply a priority difference that
/// does not exist.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AttitudeChannel {
    /// Nobody asked; the craft coasts or tumbles.
    #[default]
    Free,
    /// Bare SAS attitude hold.
    Sas,
    /// A directional nav-mode hold (prograde, target, …).
    NavMode,
    /// A flight program's continuous guidance law — the ascent pitch
    /// program, the approach profile.
    Guidance,
    /// The burn executor pointing at a scheduled directive.
    NodeBurn,
    /// Direct pilot input. Always wins when the stick is touched.
    Pilot,
}

impl AttitudeChannel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Free => "----",
            Self::Sas => "SAS",
            Self::NavMode => "NAV",
            Self::Guidance => "GUID",
            Self::NodeBurn => "NODE",
            Self::Pilot => "MAN",
        }
    }
}

/// Which source owns throttle this frame.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ThrottleChannel {
    /// The pilot's setpoint reaches the engines.
    #[default]
    Pilot,
    /// A program's guidance law is setting thrust (ascent q/accel limiting).
    Guidance,
    /// The burn executor is holding the throttle open for a scheduled burn.
    Burn,
    /// Programmatically commanded to zero — coast, engaging, tail-off.
    /// Distinct from `Pilot` at 0 %: this one the player cannot move.
    Idle,
}

impl ThrottleChannel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pilot => "MAN",
            Self::Guidance => "GUID",
            Self::Burn => "BURN",
            Self::Idle => "IDLE",
        }
    }
}

/// The next *armed* sequencing event — what the automation is about to do.
///
/// This is the column that answers "why did my rocket just do that", before
/// it does it rather than after. Armed-but-not-yet-fired is the single most
/// load-bearing idea in transport autoflight annunciation, and a launch
/// vehicle needs it more than an airliner does, because its events
/// (separation, cutoff) are irreversible.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum SequenceEvent {
    #[default]
    None,
    /// Staging is armed; separation commanded at the predicted burnout.
    Staging { stage_index: usize, in_s: f64 },
    /// Main engine cutoff on the guidance criterion. `in_s` is `None` when
    /// the criterion is a state threshold (apoapsis) with no clean
    /// countdown — the honest answer is "when the condition is met", not a
    /// fabricated number.
    Cutoff { in_s: Option<f64> },
    /// A scheduled burn the executor has armed.
    Burn { in_s: f64 },
}

/// What one automation source requires locked out of pilot reach.
///
/// **Declared by the executor, never derived from a mode enum.** The old
/// lock table read `warp: maneuver || landing || orbiting`, where
/// `orbiting` was true for the *entire* ascent program including the
/// ballistic coast — so warp-to-node was dead for the several minutes when
/// it was most wanted, and the auto-warp system cancelled itself on sight
/// of the flag. An executor knows whether it is time-critical *right now*;
/// a mode enum cannot. Each source answers for itself and the union is the
/// policy.
///
/// This also matches the intent already documented on
/// `thalos_control::arbitrate`: UI gating should fall out of the same
/// decision that resolves control, not out of a parallel flag.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AutoflightLocks {
    pub throttle: bool,
    pub attitude: bool,
    /// Lock warp only while a *single sim-time advance would break the
    /// thing you are doing* — a burn in progress, powered ascent. Coasting
    /// to a node is not that, and locking it there is what broke the WARP
    /// button.
    pub warp: bool,
    pub navigation_mode: bool,
    pub ground_steer: bool,
    pub wheel_brake: bool,
}

impl AutoflightLocks {
    pub const NONE: Self = Self {
        throttle: false,
        attitude: false,
        warp: false,
        navigation_mode: false,
        ground_steer: false,
        wheel_brake: false,
    };

    /// Everything a source that fully owns the vehicle needs: it is flying
    /// and burning, and a warp advance would integrate straight through it.
    pub const FULL_AUTHORITY: Self = Self {
        throttle: true,
        attitude: true,
        warp: true,
        navigation_mode: true,
        ground_steer: false,
        wheel_brake: false,
    };

    /// Owns the ship's flight path but is not time-critical — a coast under
    /// program control. Warp stays with the player.
    pub const GUIDANCE_COAST: Self = Self {
        throttle: true,
        attitude: true,
        warp: false,
        navigation_mode: true,
        ground_steer: false,
        wheel_brake: false,
    };

    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            throttle: self.throttle || other.throttle,
            attitude: self.attitude || other.attitude,
            warp: self.warp || other.warp,
            navigation_mode: self.navigation_mode || other.navigation_mode,
            ground_steer: self.ground_steer || other.ground_steer,
            wheel_brake: self.wheel_brake || other.wheel_brake,
        }
    }
}

/// The flight mode annunciator: what is engaged, on what, and what's next.
///
/// Read-only for everything except the resolver. The HUD renders it; no
/// panel may infer engagement from a button's own state, which is how
/// `MNVR` came to be annunciated while the ascent program held authority.
///
/// Sole writer: `thalos_runtime::control_bus::realize_control`.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq)]
pub struct AutoflightAnnunciation {
    pub program: FlightProgram,
    pub attitude: AttitudeChannel,
    pub throttle: ThrottleChannel,
    /// The next armed sequencing event, if any.
    pub armed: SequenceEvent,
    /// Sim time of the most recent change to any of the above. The HUD
    /// highlights a changed field for a few seconds — the FMA's box-and-
    /// flash convention, which exists because an un-narrated mode change is
    /// the thing pilots miss.
    pub changed_at_s: f64,
}

impl AutoflightAnnunciation {
    /// Update in place, stamping `changed_at_s` only on a real change so
    /// the highlight tracks transitions rather than frames.
    pub fn set(
        &mut self,
        program: FlightProgram,
        attitude: AttitudeChannel,
        throttle: ThrottleChannel,
        armed: SequenceEvent,
        now_s: f64,
    ) {
        let next = Self {
            program,
            attitude,
            throttle,
            armed,
            changed_at_s: self.changed_at_s,
        };
        if next != *self {
            *self = Self {
                changed_at_s: now_s,
                ..next
            };
        }
    }
}

/// What happens when the player commands a tactical mode that would
/// contradict the engaged program.
///
/// Kept as a policy with one consulting call site
/// (`thalos_runtime::autoflight::request_override`) because it is a
/// gameplay-feel question, not an architectural one, and the answer should
/// be changeable by editing a default rather than by rewriting the seam.
///
/// Real autoflight is `Immediate` — the pilot can always take the aircraft,
/// and refusing an input is considered more dangerous than obeying a wrong
/// one. A game trades differently: `ConfirmDisconnect` is the default here
/// because an accidental click during ascent costs the whole launch, while
/// one extra click costs nothing, and unlike `Refuse` it never leaves a
/// player staring at a dead button with no explanation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ProgramOverridePolicy {
    /// The control is annunciated as subordinate and does nothing. Smallest
    /// surface area, worst discoverability.
    Refuse,
    /// First press asks; second press within the confirmation window
    /// disconnects the program and hands the ship back.
    #[default]
    ConfirmDisconnect,
    /// Press disconnects the program at once. Real-avionics behaviour.
    Immediate,
}

/// How long a pending `ConfirmDisconnect` stays live before it lapses.
/// Long enough to read the prompt, short enough that a forgotten first
/// press cannot disconnect a program minutes later.
pub const OVERRIDE_CONFIRM_WINDOW_S: f64 = 5.0;

/// Player-facing autoflight policy plus the pending-confirmation state.
///
/// Sole writer: `thalos_runtime::autoflight::request_override`.
#[derive(Resource, Debug, Default)]
pub struct AutoflightPolicy {
    pub override_policy: ProgramOverridePolicy,
    /// Set when a `ConfirmDisconnect` override is awaiting its second
    /// press; carries the sim time of the first.
    pub pending_disconnect_s: Option<f64>,
}

impl AutoflightPolicy {
    /// Resolve an override request against the policy. Returns whether the
    /// caller should disconnect the program now.
    ///
    /// Idempotent per press: the caller drives this from an edge, and a
    /// lapsed pending confirmation restarts rather than accumulating.
    pub fn request_override(&mut self, program_engaged: bool, now_s: f64) -> OverrideOutcome {
        if !program_engaged {
            return OverrideOutcome::Proceed;
        }
        match self.override_policy {
            ProgramOverridePolicy::Refuse => OverrideOutcome::Refused,
            ProgramOverridePolicy::Immediate => {
                self.pending_disconnect_s = None;
                OverrideOutcome::Disconnect
            }
            ProgramOverridePolicy::ConfirmDisconnect => match self.pending_disconnect_s {
                Some(started) if now_s - started <= OVERRIDE_CONFIRM_WINDOW_S => {
                    self.pending_disconnect_s = None;
                    OverrideOutcome::Disconnect
                }
                _ => {
                    self.pending_disconnect_s = Some(now_s);
                    OverrideOutcome::ConfirmationPending
                }
            },
        }
    }

    /// Drop a stale pending confirmation. Called each frame so the HUD
    /// prompt clears on its own.
    pub fn expire_pending(&mut self, now_s: f64) {
        if let Some(started) = self.pending_disconnect_s
            && now_s - started > OVERRIDE_CONFIRM_WINDOW_S
        {
            self.pending_disconnect_s = None;
        }
    }
}

/// A player request to change what the automation is doing.
///
/// Emitted by the HUD, consumed by exactly one runtime system, which is
/// where [`ProgramOverridePolicy`] is applied. The panels deliberately
/// cannot mutate [`Autopilot`] or the programs directly — that is how a
/// button press came to silently decapitate a running ascent program while
/// its own widget kept annunciating a live program.
///
/// [`Autopilot`]: crate::nav::Autopilot
#[derive(Debug, Clone, Copy, PartialEq, Eq, Message)]
pub enum AutoflightRequest {
    /// The `MNVR` button: arm or disarm the scheduled-burn executor for
    /// hand-placed maneuver nodes.
    ToggleBurnArm,
    /// The `LAND` button: engage or disengage the landing program.
    ToggleLanding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverrideOutcome {
    /// No program engaged — the request was never a conflict.
    Proceed,
    /// Policy forbids overriding an engaged program.
    Refused,
    /// Awaiting a second press.
    ConfirmationPending,
    /// Disconnect the program and honour the request.
    Disconnect,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_union_is_permissive_to_restrictive() {
        let coast = AutoflightLocks::GUIDANCE_COAST;
        assert!(!coast.warp, "a coast must leave warp with the player");
        let both = coast.union(AutoflightLocks::FULL_AUTHORITY);
        assert!(both.warp, "a concurrent burn must still lock warp");
    }

    #[test]
    fn no_program_means_no_override_conflict() {
        let mut policy = AutoflightPolicy::default();
        assert_eq!(
            policy.request_override(false, 0.0),
            OverrideOutcome::Proceed
        );
        assert!(policy.pending_disconnect_s.is_none());
    }

    #[test]
    fn confirm_disconnect_needs_two_presses_inside_the_window() {
        let mut policy = AutoflightPolicy::default();
        assert_eq!(
            policy.request_override(true, 100.0),
            OverrideOutcome::ConfirmationPending
        );
        assert_eq!(
            policy.request_override(true, 102.0),
            OverrideOutcome::Disconnect
        );
    }

    #[test]
    fn lapsed_confirmation_restarts_instead_of_disconnecting() {
        let mut policy = AutoflightPolicy::default();
        policy.request_override(true, 100.0);
        // Well past the window: this is a fresh first press, not a confirm.
        assert_eq!(
            policy.request_override(true, 100.0 + OVERRIDE_CONFIRM_WINDOW_S + 0.1),
            OverrideOutcome::ConfirmationPending
        );
    }

    #[test]
    fn refuse_never_disconnects_and_immediate_always_does() {
        let mut refuse = AutoflightPolicy {
            override_policy: ProgramOverridePolicy::Refuse,
            pending_disconnect_s: None,
        };
        assert_eq!(refuse.request_override(true, 0.0), OverrideOutcome::Refused);
        assert_eq!(refuse.request_override(true, 1.0), OverrideOutcome::Refused);

        let mut immediate = AutoflightPolicy {
            override_policy: ProgramOverridePolicy::Immediate,
            pending_disconnect_s: None,
        };
        assert_eq!(
            immediate.request_override(true, 0.0),
            OverrideOutcome::Disconnect
        );
    }

    #[test]
    fn annunciation_stamps_only_real_changes() {
        let mut fma = AutoflightAnnunciation::default();
        fma.set(
            FlightProgram::Ascent,
            AttitudeChannel::Guidance,
            ThrottleChannel::Guidance,
            SequenceEvent::None,
            10.0,
        );
        assert_eq!(fma.changed_at_s, 10.0);
        // Same state one frame later: the highlight must not re-trigger.
        fma.set(
            FlightProgram::Ascent,
            AttitudeChannel::Guidance,
            ThrottleChannel::Guidance,
            SequenceEvent::None,
            10.016,
        );
        assert_eq!(fma.changed_at_s, 10.0);
    }
}
