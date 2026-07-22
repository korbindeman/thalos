//! Per-craft regime classification — the pure core of the `CraftRegime`
//! resolver (`docs/simulation/regimes.md`, Phase A2).
//!
//! **Shadow mode.** The game-side resolver (`thalos_runtime::regime`) computes
//! this record alongside the legacy machinery (`AvianRole` /
//! `manage_authority` / the scattered warp gates) and a drift checker
//! compares the two; no consumer reads the record yet. Classification must
//! therefore mirror today's behaviour *exactly*, quirks included — comments
//! marked "faithful mirror" flag spots where legacy logic is reproduced
//! rather than redesigned (cleanups happen in Phase A3+, once consumers read
//! the record and a behaviour change is a deliberate diff, not drift).
//!
//! Pure Rust, no Bevy, unit-tested. The game-side shim gathers
//! [`RegimeInputs`] from the ECS; everything here is plain data in, plain
//! data out. Naming says **Backend**, not Avian — the local rigid-body
//! engine is a swappable executor detail (`docs/simulation/regimes.md` §4).

use crate::canonical::AuthorityMode;

/// Throttle threshold below which a craft counts as "engine off" for the
/// quiet-contact / settle predicates. Faithful mirror of the `1.0e-3`
/// literals in `stable_contact_reached` and the game's grounded checks.
pub const STABLE_THROTTLE_EPSILON: f64 = 1.0e-3;

/// Commanded-throttle threshold that releases a landed (`BodyFixed`) ship
/// back to live physics. Mirror of the game's `LANDED_THROTTLE_RELEASE`.
pub const LANDED_THROTTLE_RELEASE: f64 = 1.0e-3;

// ---------------------------------------------------------------------------
// Record types
// ---------------------------------------------------------------------------

/// The medium the craft currently moves through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Medium {
    Vacuum,
    /// Below the dominant body's Kármán line (aero forces act). Faithful
    /// mirror of `craft_in_atmosphere`: altitude above the **mean radius**
    /// compared against `karman_line_m`.
    Atmosphere,
    /// Reserved for watercraft — never produced yet.
    WaterSurface,
    /// Reserved for submersibles — never produced yet.
    Submerged,
}

/// Ground interaction state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroundState {
    Airborne,
    /// Touching the ground (weight-on-wheels or hull contact) but not yet
    /// settled.
    Contact,
    /// Quiet contact held for the settle dwell time (or already landed
    /// under `BodyFixed`, or standing at rest on foot).
    Settled,
}

/// Who integrates/owns the craft's translation this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslationOwner {
    /// Canonical simulation: rails coast or analytic `BodyFixed` pose.
    Canonical,
    /// The local rigid-body backend (Avian today).
    Backend,
    /// A kinematic locomotion controller (walking).
    Kinematic,
}

/// Who integrates/owns the craft's rotation this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationOwner {
    /// Canonical: frozen under warp, analytic under `BodyFixed`.
    Canonical,
    /// The backend, under fly-by-wire torque.
    Backend,
    Kinematic,
}

/// Why the warp cap is what it is — informational, for the HUD and
/// diagnostics. The `max_level` value is what gets enforced (and
/// drift-checked); the constraint label is best-effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpConstraint {
    Unconstrained,
    /// Capped by the per-level minimum-altitude ladder.
    AltitudeLadder,
    /// Clamped to 1× inside the atmosphere shell (aero forces only run
    /// live).
    InAtmosphere,
    /// Reserved: today a craft moving on the surface is clamped via the
    /// altitude ladder (altitude ≈ 0), not a dedicated rule.
    MovingOnSurface,
    /// On foot and not standing at rest — KSP rule: surface warp only once
    /// fully stopped.
    NotAtRestOnFoot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpPolicy {
    /// Highest permitted index into the warp ladder. `usize::MAX` means
    /// unconstrained (mirrors `WarpLimits::max_level` semantics).
    pub max_level: usize,
    pub constraint: WarpConstraint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HideReason {
    /// Landed (`BodyFixed`) — no ballistic trajectory exists.
    Landed,
    /// In ground contact under the backend — velocity carries contact
    /// reactions Kepler prediction can't follow.
    GroundContact,
    /// Walking on foot — analytically glued to the rotating surface.
    OnFoot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionDisplay {
    Show,
    Hide(HideReason),
}

/// The per-frame regime decision record (`docs/simulation/regimes.md` §3). Sole writer:
/// the game-side resolver. Downstream systems become executors of this in
/// Phase A3; in A2 it is shadow-only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CraftRegime {
    pub medium: Medium,
    pub ground: GroundState,
    pub translation_owner: TranslationOwner,
    pub rotation_owner: RotationOwner,
    /// Should the backend's integrator clock step this frame? Rotation-only
    /// integration is `rotation_owner == Backend` with
    /// `translation_owner == Canonical` (today's `AttitudeOnly`).
    pub backend_clock_runs: bool,
    pub warp: WarpPolicy,
    pub prediction: PredictionDisplay,
    /// Warp/capability gate for the ground-collider systems. The
    /// attach/detach systems keep their AGL geometry + hysteresis; this is
    /// only the "may colliders exist at all" predicate.
    pub terrain_collider_allowed: bool,
}

/// Cross-frame resolver state: the settle dwell timer (absorbing the legacy
/// `LocalBubble::stable_contact_s`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RegimeMemory {
    pub settle_timer_s: f64,
}

/// Discriminant of [`AuthorityMode`] — the resolver classifies on the kind,
/// never the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityKind {
    OnRails,
    LocalRigidBody,
    BodyFixed,
}

impl From<AuthorityMode> for AuthorityKind {
    fn from(mode: AuthorityMode) -> Self {
        match mode {
            AuthorityMode::OnRails { .. } => Self::OnRails,
            AuthorityMode::LocalRigidBody { .. } => Self::LocalRigidBody,
            AuthorityMode::BodyFixed { .. } => Self::BodyFixed,
        }
    }
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// One rung of the warp ladder: discrete multiplier + the minimum altitude
/// (in body radii) required to engage it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarpLevel {
    pub speed: f64,
    pub min_altitude_radii: f64,
}

/// Inputs present only while the walking locomotion controller owns the
/// craft (today: grounded EVA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkingInputs {
    /// Standing on the surface (vs ballistic within the walking mode —
    /// a jump or a fall off a ledge).
    pub grounded: bool,
    /// Stationary long enough to be warp-eligible (KSP "landed and
    /// stationary").
    pub at_rest: bool,
    /// Movement input held this frame — drops warp immediately without
    /// waiting on the rest debounce.
    pub wants_to_move: bool,
}

/// Everything the classification reads, as plain data. Snapshot semantics
/// (`docs/simulation/regimes.md` §3.2): physics-derived signals (contacts,
/// weight-on-wheels, collider presence, speeds) are **previous-frame**;
/// command inputs (warp, throttle) are **current-frame**.
#[derive(Debug, Clone, PartialEq)]
pub struct RegimeInputs<'a> {
    /// Simulation clock delta this frame (zero while sim-paused). Drives
    /// the settle dwell timer.
    pub sim_delta_s: f64,
    /// Effective warp multiplier (mid-lerp values included).
    pub warp_speed: f64,
    /// Discrete multiplier of the selected target warp level.
    pub warp_target_speed: f64,
    pub warp_ladder: &'a [WarpLevel],
    /// Post-fuel-gate throttle actually producing thrust.
    pub throttle_effective: f64,
    /// Player/autopilot throttle setpoint (drives the landed release).
    pub throttle_commanded: f64,
    /// Canonical authority at the time of resolution (frame start).
    pub authority: AuthorityKind,
    /// `Some` while the walking locomotion controller is active.
    pub walking: Option<WalkingInputs>,
    /// Whether this craft has a physical collider at all (the EVA capsule's
    /// is removed at spawn). Capability proxy until parts declare it.
    pub craft_has_collider: bool,
    pub body_radius_m: f64,
    /// Altitude above the dominant body's mean radius (`|r| − radius`).
    /// Used for the atmosphere medium test (mirror of
    /// `craft_in_atmosphere`).
    pub altitude_above_mean_m: f64,
    /// Altitude above the conservative `radius + max_terrain_elevation`
    /// buffer. Used for the warp ladder and the in-atmosphere warp clamp
    /// (mirror of `enforce_warp_altitude_limits`). The two altitude bases
    /// differ near the Kármán line by up to the tallest peak — a legacy
    /// inconsistency carried over deliberately; flagged for Phase A4.
    pub altitude_above_terrain_buffer_m: f64,
    /// Dominant body's Kármán line in metres; `<= 0` means airless.
    pub karman_line_m: f64,
    /// A terrain collider patch is currently attached.
    pub terrain_collider_attached: bool,
    /// The hull is in solver contact with the terrain patch
    /// (`craft_contacts_terrain`). Runway contact is *not* included —
    /// wheeled craft report through `weight_on_wheels` instead (legacy
    /// behaviour).
    pub hull_contacts_terrain_patch: bool,
    pub weight_on_wheels: bool,
    /// Backend-frame (SLF) speeds — surface-relative by construction.
    pub linear_speed_m_s: f64,
    pub angular_speed_rad_s: f64,
    pub max_stable_speed_m_s: f64,
    pub max_stable_angular_speed_rad_s: f64,
    /// Quiet-contact dwell required to settle (legacy
    /// `stable_contact_time_s`).
    pub settle_dwell_s: f64,
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Classify the craft's regime for this frame.
///
/// Pure: same inputs + memory ⇒ same record. The game-side resolver is the
/// sole caller and sole writer of the resulting component.
pub fn resolve(inputs: &RegimeInputs, memory: &RegimeMemory) -> (CraftRegime, RegimeMemory) {
    let medium = classify_medium(inputs);
    let (ground, next_memory) = classify_ground(inputs, memory);
    let (translation_owner, rotation_owner, backend_clock_runs) = classify_owners(inputs, medium);
    let warp = warp_policy(inputs);
    let prediction = prediction_display(inputs);
    let terrain_collider_allowed = terrain_collider_gate(inputs, &warp);
    (
        CraftRegime {
            medium,
            ground,
            translation_owner,
            rotation_owner,
            backend_clock_runs,
            warp,
            prediction,
            terrain_collider_allowed,
        },
        next_memory,
    )
}

fn classify_medium(inputs: &RegimeInputs) -> Medium {
    if inputs.karman_line_m > 0.0 && inputs.altitude_above_mean_m < inputs.karman_line_m {
        Medium::Atmosphere
    } else {
        Medium::Vacuum
    }
}

fn classify_ground(inputs: &RegimeInputs, memory: &RegimeMemory) -> (GroundState, RegimeMemory) {
    if let Some(walking) = inputs.walking {
        let ground = if !walking.grounded {
            GroundState::Airborne
        } else if walking.at_rest {
            GroundState::Settled
        } else {
            GroundState::Contact
        };
        return (ground, RegimeMemory::default());
    }
    if inputs.authority == AuthorityKind::BodyFixed {
        // Landed pose is analytic — settled by definition; the dwell timer
        // resets (mirror of `collapse_or_constrain_warp`'s BodyFixed arm).
        return (GroundState::Settled, RegimeMemory::default());
    }
    let contact = inputs.weight_on_wheels || inputs.hull_contacts_terrain_patch;
    // Faithful mirror of `stable_contact_reached`.
    let quiet = contact
        && inputs.linear_speed_m_s < inputs.max_stable_speed_m_s
        && inputs.angular_speed_rad_s < inputs.max_stable_angular_speed_rad_s
        && inputs.throttle_effective <= STABLE_THROTTLE_EPSILON;
    let settle_timer_s = if quiet {
        memory.settle_timer_s + inputs.sim_delta_s.max(0.0)
    } else {
        0.0
    };
    let ground = if quiet && settle_timer_s >= inputs.settle_dwell_s {
        GroundState::Settled
    } else if contact {
        GroundState::Contact
    } else {
        GroundState::Airborne
    };
    (ground, RegimeMemory { settle_timer_s })
}

/// Faithful mirror of `avian_role_from_inputs`, plus the walking mode:
/// `Paused` ⇔ clock off, `AttitudeOnly` ⇔ Canonical translation + Backend
/// rotation, `Full` ⇔ Backend both.
fn classify_owners(
    inputs: &RegimeInputs,
    medium: Medium,
) -> (TranslationOwner, RotationOwner, bool) {
    if inputs.walking.is_some() {
        // The walking controller owns the capsule outright. (Legacy keeps
        // the backend clock running here while the controller overwrites
        // `Position` each frame; the record states the *intent* — the drift
        // checker does not compare the clock for walking.)
        return (TranslationOwner::Kinematic, RotationOwner::Kinematic, false);
    }
    let near_one_x = (inputs.warp_speed - 1.0).abs() <= f64::EPSILON;
    if !near_one_x {
        return (TranslationOwner::Canonical, RotationOwner::Canonical, false);
    }
    if inputs.authority == AuthorityKind::BodyFixed {
        return (TranslationOwner::Canonical, RotationOwner::Canonical, false);
    }
    let thrust_active = inputs.throttle_effective > 0.0;
    if thrust_active || inputs.terrain_collider_attached || medium == Medium::Atmosphere {
        (TranslationOwner::Backend, RotationOwner::Backend, true)
    } else {
        (TranslationOwner::Canonical, RotationOwner::Backend, true)
    }
}

/// Faithful mirror of `enforce_warp_altitude_limits`.
fn warp_policy(inputs: &RegimeInputs) -> WarpPolicy {
    let ladder = inputs.warp_ladder;
    let top = ladder.len().saturating_sub(1);

    if let Some(walking) = inputs.walking {
        let can_warp = walking.at_rest && !walking.wants_to_move;
        if !can_warp {
            let max_level = ladder
                .iter()
                .rposition(|level| level.speed <= 1.0)
                .unwrap_or(0);
            return WarpPolicy {
                max_level,
                constraint: WarpConstraint::NotAtRestOnFoot,
            };
        }
        return WarpPolicy {
            max_level: top,
            constraint: WarpConstraint::Unconstrained,
        };
    }

    // Quiet hull-contact ship — exempt from the altitude floor like
    // `BodyFixed` (it cannot phase through terrain). Capability-based
    // rather than `VesselKind`: a collider-less craft never has a terrain
    // patch attached, so the legacy `vessel == Ship` term is implied.
    let ship_grounded_stationary = inputs.authority == AuthorityKind::LocalRigidBody
        && inputs.terrain_collider_attached
        && inputs.hull_contacts_terrain_patch
        && inputs.linear_speed_m_s < inputs.max_stable_speed_m_s
        && inputs.angular_speed_rad_s < inputs.max_stable_angular_speed_rad_s
        && inputs.throttle_effective <= STABLE_THROTTLE_EPSILON;
    if ship_grounded_stationary || inputs.authority == AuthorityKind::BodyFixed {
        return WarpPolicy {
            max_level: top,
            constraint: WarpConstraint::Unconstrained,
        };
    }

    if inputs.body_radius_m <= 0.0 {
        return WarpPolicy {
            max_level: usize::MAX,
            constraint: WarpConstraint::Unconstrained,
        };
    }

    let alt_radii = inputs.altitude_above_terrain_buffer_m.max(0.0) / inputs.body_radius_m;
    let mut max_level = 0usize;
    for (index, level) in ladder.iter().enumerate() {
        if level.min_altitude_radii <= alt_radii {
            max_level = index;
        }
    }
    let mut constraint = if max_level < top {
        WarpConstraint::AltitudeLadder
    } else {
        WarpConstraint::Unconstrained
    };

    if inputs.karman_line_m > 0.0 && inputs.altitude_above_terrain_buffer_m < inputs.karman_line_m {
        let one_x = ladder
            .iter()
            .position(|level| (level.speed - 1.0).abs() <= f64::EPSILON)
            .unwrap_or(0);
        if one_x < max_level {
            constraint = WarpConstraint::InAtmosphere;
        }
        max_level = max_level.min(one_x);
    }

    WarpPolicy {
        max_level,
        constraint,
    }
}

/// Faithful mirror of `bridge::ship_is_ballistic` + the grounded-EVA
/// prediction clear.
fn prediction_display(inputs: &RegimeInputs) -> PredictionDisplay {
    if inputs.walking.is_some() {
        return PredictionDisplay::Hide(HideReason::OnFoot);
    }
    match inputs.authority {
        AuthorityKind::BodyFixed => PredictionDisplay::Hide(HideReason::Landed),
        AuthorityKind::LocalRigidBody => {
            if inputs.terrain_collider_attached && inputs.hull_contacts_terrain_patch {
                PredictionDisplay::Hide(HideReason::GroundContact)
            } else {
                PredictionDisplay::Show
            }
        }
        AuthorityKind::OnRails => PredictionDisplay::Show,
    }
}

/// Faithful mirror of `terrain_colliders_allowed_by_warp` plus the
/// collider-capability gate (the legacy per-`VesselKind` EVA skip).
fn terrain_collider_gate(inputs: &RegimeInputs, warp: &WarpPolicy) -> bool {
    if !inputs.craft_has_collider {
        return false;
    }
    let Some(one_x_index) = inputs
        .warp_ladder
        .iter()
        .position(|level| (level.speed - 1.0).abs() <= f64::EPSILON)
    else {
        return false;
    };
    (inputs.warp_speed - 1.0).abs() <= f64::EPSILON && warp.max_level <= one_x_index
}

// ---------------------------------------------------------------------------
// Canonical-authority projection
// ---------------------------------------------------------------------------

/// Map the regime decision onto the expected canonical [`AuthorityKind`] at
/// end of frame. In A2 this feeds the drift checker; in A3 it becomes the
/// core of the single authority executor (successor of `manage_authority` +
/// `collapse_or_constrain_warp` + `release_landed_ship_on_throttle`).
pub fn expected_authority(inputs: &RegimeInputs, regime: &CraftRegime) -> AuthorityKind {
    // Walking pin: never Kepler-coast a surface-co-rotating walker.
    if inputs.walking.is_some() {
        return AuthorityKind::LocalRigidBody;
    }

    let mut authority = inputs.authority;
    let mut just_released = false;
    if authority == AuthorityKind::BodyFixed {
        if inputs.throttle_commanded > LANDED_THROTTLE_RELEASE {
            // Landed release on commanded throttle. The release routes through
            // `OnRails` for the handoff frame *on purpose*: `OnRails` Kepler-
            // propagates canonical so it keeps pace with the body's orbital
            // motion (~hundreds of m/frame) while the backend takes over. A
            // direct `BodyFixed → LocalRigidBody` instead holds canonical
            // static in inertial space for that frame (see `Simulation::step`),
            // so the body flies out from under the craft and the surface-local
            // snap teleports it ~one orbital frame away. The settle dwell
            // restarts from zero after a release (legacy resets the timer while
            // `BodyFixed`), so the Settled collapse below must not re-pin the
            // craft on the release frame — `regime.ground` was classified from
            // the pre-release `BodyFixed` authority.
            authority = AuthorityKind::OnRails;
            just_released = true;
        } else {
            return AuthorityKind::BodyFixed;
        }
    }
    // Landed warp-request collapse (mirror of `manage_authority`): quiet
    // hull contact + warp requested above 1× ⇒ pin to the rotating surface.
    if authority == AuthorityKind::LocalRigidBody
        && inputs.warp_target_speed > 1.0
        && inputs.terrain_collider_attached
        && inputs.hull_contacts_terrain_patch
        && inputs.linear_speed_m_s < inputs.max_stable_speed_m_s
        && inputs.angular_speed_rad_s < inputs.max_stable_angular_speed_rad_s
        && inputs.throttle_effective <= STABLE_THROTTLE_EPSILON
    {
        return AuthorityKind::BodyFixed;
    }

    // Timed settle collapse (mirror of `collapse_or_constrain_warp`).
    if !just_released && regime.ground == GroundState::Settled {
        return AuthorityKind::BodyFixed;
    }

    if regime.translation_owner == TranslationOwner::Backend {
        AuthorityKind::LocalRigidBody
    } else {
        AuthorityKind::OnRails
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const LADDER: [WarpLevel; 5] = [
        WarpLevel {
            speed: 0.0,
            min_altitude_radii: 0.0,
        },
        WarpLevel {
            speed: 1.0,
            min_altitude_radii: 0.0,
        },
        WarpLevel {
            speed: 10.0,
            min_altitude_radii: 0.001,
        },
        WarpLevel {
            speed: 100.0,
            min_altitude_radii: 0.001,
        },
        WarpLevel {
            speed: 1000.0,
            min_altitude_radii: 0.01,
        },
    ];

    /// Vacuum coast at 1× in low orbit — the AttitudeOnly baseline.
    fn coast_inputs() -> RegimeInputs<'static> {
        RegimeInputs {
            sim_delta_s: 1.0 / 64.0,
            warp_speed: 1.0,
            warp_target_speed: 1.0,
            warp_ladder: &LADDER,
            throttle_effective: 0.0,
            throttle_commanded: 0.0,
            authority: AuthorityKind::OnRails,
            walking: None,
            craft_has_collider: true,
            body_radius_m: 3.186e6,
            altitude_above_mean_m: 400_000.0,
            altitude_above_terrain_buffer_m: 390_000.0,
            karman_line_m: 0.0,
            terrain_collider_attached: false,
            hull_contacts_terrain_patch: false,
            weight_on_wheels: false,
            linear_speed_m_s: 2_000.0,
            angular_speed_rad_s: 0.0,
            max_stable_speed_m_s: 0.5,
            max_stable_angular_speed_rad_s: 0.05,
            settle_dwell_s: 2.0,
        }
    }

    #[test]
    fn vacuum_coast_is_canonical_translation_backend_rotation() {
        let (regime, _) = resolve(&coast_inputs(), &RegimeMemory::default());
        assert_eq!(regime.medium, Medium::Vacuum);
        assert_eq!(regime.translation_owner, TranslationOwner::Canonical);
        assert_eq!(regime.rotation_owner, RotationOwner::Backend);
        assert!(regime.backend_clock_runs);
        assert_eq!(regime.ground, GroundState::Airborne);
        assert_eq!(regime.prediction, PredictionDisplay::Show);
        assert_eq!(
            expected_authority(&coast_inputs(), &regime),
            AuthorityKind::OnRails
        );
    }

    #[test]
    fn warp_freezes_backend() {
        let mut inputs = coast_inputs();
        inputs.warp_speed = 100.0;
        inputs.warp_target_speed = 100.0;
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(regime.translation_owner, TranslationOwner::Canonical);
        assert_eq!(regime.rotation_owner, RotationOwner::Canonical);
        assert!(!regime.backend_clock_runs);
        assert!(!regime.terrain_collider_allowed);
    }

    #[test]
    fn thrust_or_terrain_or_atmosphere_hands_translation_to_backend() {
        for setup in [
            |i: &mut RegimeInputs| i.throttle_effective = 0.5,
            |i: &mut RegimeInputs| i.terrain_collider_attached = true,
            |i: &mut RegimeInputs| {
                i.karman_line_m = 100_000.0;
                i.altitude_above_mean_m = 50_000.0;
            },
        ] {
            let mut inputs = coast_inputs();
            setup(&mut inputs);
            let (regime, _) = resolve(&inputs, &RegimeMemory::default());
            assert_eq!(regime.translation_owner, TranslationOwner::Backend);
            assert_eq!(regime.rotation_owner, RotationOwner::Backend);
            assert_eq!(
                expected_authority(&inputs, &regime),
                AuthorityKind::LocalRigidBody
            );
        }
    }

    #[test]
    fn atmosphere_clamps_warp_to_one_x() {
        let mut inputs = coast_inputs();
        inputs.karman_line_m = 100_000.0;
        inputs.altitude_above_mean_m = 50_000.0;
        inputs.altitude_above_terrain_buffer_m = 45_000.0;
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(regime.medium, Medium::Atmosphere);
        assert_eq!(regime.warp.max_level, 1); // index of 1× in LADDER
        assert_eq!(regime.warp.constraint, WarpConstraint::InAtmosphere);
    }

    #[test]
    fn altitude_ladder_caps_levels() {
        let mut inputs = coast_inputs();
        // 0.005 radii: clears the 0.001 floors but not the 0.01 floor.
        inputs.altitude_above_terrain_buffer_m = 0.005 * inputs.body_radius_m;
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(regime.warp.max_level, 3);
        assert_eq!(regime.warp.constraint, WarpConstraint::AltitudeLadder);

        inputs.altitude_above_terrain_buffer_m = 0.05 * inputs.body_radius_m;
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(regime.warp.max_level, 4);
        assert_eq!(regime.warp.constraint, WarpConstraint::Unconstrained);
    }

    #[test]
    fn body_fixed_is_settled_canonical_and_warp_free() {
        let mut inputs = coast_inputs();
        inputs.authority = AuthorityKind::BodyFixed;
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(regime.ground, GroundState::Settled);
        assert_eq!(regime.translation_owner, TranslationOwner::Canonical);
        assert!(!regime.backend_clock_runs);
        assert_eq!(regime.warp.max_level, LADDER.len() - 1);
        assert_eq!(
            regime.prediction,
            PredictionDisplay::Hide(HideReason::Landed)
        );
        assert_eq!(
            expected_authority(&inputs, &regime),
            AuthorityKind::BodyFixed
        );
    }

    #[test]
    fn landed_release_on_commanded_throttle() {
        let mut inputs = coast_inputs();
        inputs.authority = AuthorityKind::BodyFixed;
        inputs.throttle_commanded = 0.5;
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        // Effective throttle is still 0 this frame, so the release lands on
        // OnRails; the backend takes over once thrust becomes effective. The
        // OnRails hop is deliberate — it keeps canonical co-moving with the
        // body's orbit during the handoff frame (a direct jump to the backend
        // would strand the craft ~one orbital frame away in the surface-local
        // frame).
        assert_eq!(expected_authority(&inputs, &regime), AuthorityKind::OnRails);
    }

    #[test]
    fn settle_timer_collapses_after_dwell() {
        let mut inputs = coast_inputs();
        inputs.authority = AuthorityKind::LocalRigidBody;
        inputs.terrain_collider_attached = true;
        inputs.hull_contacts_terrain_patch = true;
        inputs.linear_speed_m_s = 0.1;
        inputs.angular_speed_rad_s = 0.01;
        inputs.sim_delta_s = 1.0;

        let mut memory = RegimeMemory::default();
        let (regime, next) = resolve(&inputs, &memory);
        assert_eq!(regime.ground, GroundState::Contact);
        assert_eq!(
            expected_authority(&inputs, &regime),
            AuthorityKind::LocalRigidBody
        );
        memory = next;

        let (regime, next) = resolve(&inputs, &memory);
        assert_eq!(regime.ground, GroundState::Settled);
        assert_eq!(
            expected_authority(&inputs, &regime),
            AuthorityKind::BodyFixed
        );
        memory = next;

        // Movement resets the dwell.
        inputs.linear_speed_m_s = 5.0;
        let (regime, next) = resolve(&inputs, &memory);
        assert_eq!(regime.ground, GroundState::Contact);
        assert_eq!(next.settle_timer_s, 0.0);
        let _ = regime;
    }

    #[test]
    fn warp_request_collapses_quiet_grounded_ship() {
        let mut inputs = coast_inputs();
        inputs.authority = AuthorityKind::LocalRigidBody;
        inputs.terrain_collider_attached = true;
        inputs.hull_contacts_terrain_patch = true;
        inputs.linear_speed_m_s = 0.1;
        inputs.angular_speed_rad_s = 0.01;
        inputs.warp_target_speed = 10.0;
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(
            expected_authority(&inputs, &regime),
            AuthorityKind::BodyFixed
        );
        // And the quiet grounded ship is exempt from the altitude floor.
        assert_eq!(regime.warp.max_level, LADDER.len() - 1);
    }

    #[test]
    fn walking_owns_the_craft_kinematically() {
        let mut inputs = coast_inputs();
        inputs.craft_has_collider = false;
        inputs.walking = Some(WalkingInputs {
            grounded: true,
            at_rest: false,
            wants_to_move: true,
        });
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(regime.translation_owner, TranslationOwner::Kinematic);
        assert_eq!(regime.rotation_owner, RotationOwner::Kinematic);
        assert!(!regime.backend_clock_runs);
        assert_eq!(regime.ground, GroundState::Contact);
        assert_eq!(
            regime.prediction,
            PredictionDisplay::Hide(HideReason::OnFoot)
        );
        assert!(!regime.terrain_collider_allowed);
        assert_eq!(
            expected_authority(&inputs, &regime),
            AuthorityKind::LocalRigidBody
        );
        // Moving on foot clamps warp to 1×.
        assert_eq!(regime.warp.max_level, 1);
        assert_eq!(regime.warp.constraint, WarpConstraint::NotAtRestOnFoot);
    }

    #[test]
    fn walking_at_rest_unlocks_surface_warp() {
        let mut inputs = coast_inputs();
        inputs.craft_has_collider = false;
        inputs.walking = Some(WalkingInputs {
            grounded: true,
            at_rest: true,
            wants_to_move: false,
        });
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(regime.ground, GroundState::Settled);
        assert_eq!(regime.warp.max_level, LADDER.len() - 1);
        assert_eq!(regime.warp.constraint, WarpConstraint::Unconstrained);
    }

    #[test]
    fn terrain_collider_gate_requires_one_x_and_low_cap() {
        let mut inputs = coast_inputs();
        inputs.altitude_above_terrain_buffer_m = 1_000.0; // cap → 1×
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(regime.warp.max_level, 1);
        assert!(regime.terrain_collider_allowed);

        // High cap (high altitude) means no collider even at 1×.
        let inputs = coast_inputs();
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert!(regime.warp.max_level > 1);
        assert!(!regime.terrain_collider_allowed);
    }

    #[test]
    fn ground_contact_hides_prediction() {
        let mut inputs = coast_inputs();
        inputs.authority = AuthorityKind::LocalRigidBody;
        inputs.terrain_collider_attached = true;
        inputs.hull_contacts_terrain_patch = true;
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(
            regime.prediction,
            PredictionDisplay::Hide(HideReason::GroundContact)
        );
        // Contact without the patch (e.g. mid-detach) still shows — mirror
        // of `ship_is_ballistic`.
        inputs.terrain_collider_attached = false;
        inputs.hull_contacts_terrain_patch = false;
        let (regime, _) = resolve(&inputs, &RegimeMemory::default());
        assert_eq!(regime.prediction, PredictionDisplay::Show);
    }
}
