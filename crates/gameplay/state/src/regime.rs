//! Per-craft regime records and the Avian-role projection — the shared
//! vocabulary of "who owns this craft's motion this frame". The resolver and
//! the authority executor live with the runtime; the classification itself is
//! the unit-tested `thalos_physics_canonical::regime`.

use bevy::prelude::*;
use thalos_physics_canonical::regime::{AuthorityKind, CraftRegime, RegimeMemory};

/// Per-craft regime record + resolver memory. **Sole writer:** the runtime's
/// `resolve_regime`. This per-craft component is the N-craft template
/// (`docs/roadmap/architecture_cleanup.md` §2.2): new per-craft state follows
/// this shape, not a new global resource.
#[derive(Component, Debug, Clone)]
pub struct CraftRegimeState {
    pub regime: CraftRegime,
    pub memory: RegimeMemory,
    /// Canonical authority the record projects for end of frame
    /// (`regime::expected_authority`), captured at resolve time so the
    /// drift checker compares against exactly what the resolver decided.
    pub expected_authority: AuthorityKind,
}

/// What role does Avian play this frame?
///
/// Three roles, corresponding to three regimes of canonical/Avian
/// authority. The split exists because two distinct questions need
/// independent answers:
///
/// 1. *Should Avian's PhysicsSchedule step at all?* — needed for rotation
///    integration (player attitude commands, SAS damping) and for contact
///    detection. False under warp (numerical integration explodes at large
///    `dt`) and under `BodyFixed` (landed pose is analytic).
/// 2. *Should Avian's translation be authoritative?* — only when there is
///    a non-gravity force to integrate (thrust, contact). Otherwise
///    canonical Kepler owns translation, and AP/PE do not drift even when
///    Avian's clock keeps stepping for rotation.
///
/// Conflating the two — pausing Avian whenever it didn't own translation —
/// also paused rotation integration, which broke player rotation while
/// coasting. The split here keeps Avian's clock alive for rotation/contact
/// in coast mode while leaving translation to Kepler.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum AvianRole {
    /// Avian's clock is paused; canonical owns everything (translation,
    /// rotation, pose). Used at non-1× warp and under `BodyFixed`. The
    /// snap writes canonical state into Avian's components each frame so
    /// render and contact queries stay coherent without an integrator
    /// race.
    #[default]
    Paused,
    /// Avian's clock runs to integrate rotation under player/SAS torque
    /// and to keep the contact graph live, but Kepler owns translation.
    /// Used at 1× warp when the ship is coasting in vacuum (no thrust,
    /// no terrain collider attached). The snap writes canonical pos/vel
    /// into Avian each frame; rotation is left alone for Avian to
    /// integrate.
    AttitudeOnly,
    /// Avian owns both rotation and translation. Used at 1× warp when
    /// there is a non-gravity force to integrate (throttle active or
    /// terrain collider attached so contact resolution may need to fire).
    Full,
}

/// Per-frame Avian role + previous-frame role for edge detection.
///
/// Since the A3 port (`docs/simulation/regimes.md`) this is a **projection of
/// the per-craft [`CraftRegimeState`] record**: the runtime's
/// `compute_avian_authority` (its **sole writer**) derives the role from the
/// record's owner/clock fields, keeping this resource as the distribution
/// vehicle every backend-side system reads — including the `previous_role`
/// edge the handoff snap depends on.
#[derive(Resource, Default, Debug, Clone, Copy, Reflect)]
#[reflect(Resource)]
pub struct AvianAuthority {
    pub role: AvianRole,
    pub previous_role: AvianRole,
}

impl AvianAuthority {
    /// True when Avian's `PhysicsSchedule` should step this frame —
    /// either coasting (rotation only) or full ownership.
    pub fn integrator_active(self) -> bool {
        !matches!(self.role, AvianRole::Paused)
    }

    /// True when Avian's translation (`Position`, `LinearVelocity`) is
    /// the authoritative source for canonical translation.
    pub fn owns_translation(self) -> bool {
        matches!(self.role, AvianRole::Full)
    }

    /// True on the single frame Avian transitions from not owning
    /// translation to owning it (Paused/AttitudeOnly → Full). The snap
    /// uses this to do a one-shot full-state push at the handoff so
    /// readback's conversion cancels exactly.
    pub fn just_took_translation(self) -> bool {
        matches!(self.role, AvianRole::Full) && !matches!(self.previous_role, AvianRole::Full)
    }
}
