//! Active velocity reference frame (navball speed mode) for the player craft.
//!
//! KSP-style Orbit / Surface / Target selection. The frame auto-switches
//! from the craft's situation (Surface when landed or below the dominant
//! body's surface-frame ceiling, else Orbit) with a sticky manual override
//! set by clicking the speed readout — modeled directly on the altitude
//! panel's SEA/GND toggle (`hud::orbital_panel`).
//!
//! The pure frame math lives in [`thalos_physics_canonical::velocity_frame`];
//! this module owns only *which* frame is active. Consumers (navball
//! markers, SAS holds, the speed readout) read [`VelocityFrameState::active`]
//! and evaluate `nav_basis` themselves with their stage-correct body state —
//! the SAS control path reads the ephemeris in `Physics`, the navball/HUD
//! path reads the per-frame solar-system snapshot after `Sync`.

use bevy::prelude::*;
use thalos_physics_canonical::canonical::{AuthorityMode, Epoch};
use thalos_physics_canonical::velocity_frame::VelocityReferenceFrame;

use crate::SimStage;
use crate::rendering::SimulationState;
use crate::target::TargetBody;

/// Fraction of body radius used as the surface-frame ceiling for an airless
/// body that authors neither a `terrestrial_atmosphere` (Kármán line) nor an
/// explicit `surface_frame_ceiling_m`. ~0.5% of radius — a few km on a small
/// moon, ~16 km on Thalos. Tunable per body via `surface_frame_ceiling_m`.
const DEFAULT_CEILING_RADIUS_FRACTION: f64 = 0.005;

/// The player craft's active navball speed mode plus the sticky-override
/// bookkeeping that drives auto-switching.
#[derive(Resource, Default, Debug)]
pub struct VelocityFrameState {
    /// Frame active this frame. **Sole writer:** [`update_velocity_frame`].
    pub active: VelocityReferenceFrame,
    /// Sticky user override (set by the readout click); cleared when the
    /// auto suggestion changes (situation-boundary crossing) or a Target
    /// override loses its target.
    override_choice: Option<VelocityReferenceFrame>,
    /// Previous frame's auto suggestion, for transition detection.
    last_suggested: Option<VelocityReferenceFrame>,
}

impl VelocityFrameState {
    /// Pin a manual override (called by the speed-readout click handler).
    pub fn set_override(&mut self, frame: VelocityReferenceFrame) {
        self.override_choice = Some(frame);
    }
}

/// Next frame in the click cycle: Orbit → Surface → Target → Orbit,
/// skipping Target when no target is selected.
pub fn next_frame(current: VelocityReferenceFrame, target_available: bool) -> VelocityReferenceFrame {
    use VelocityReferenceFrame::*;
    match current {
        Orbit => Surface,
        Surface => {
            if target_available {
                Target
            } else {
                Orbit
            }
        }
        Target => Orbit,
    }
}

pub struct VelocityFramePlugin;

impl Plugin for VelocityFramePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VelocityFrameState>().add_systems(
            Update,
            update_velocity_frame
                .in_set(SimStage::Physics)
                .before(crate::bridge::handle_attitude_controls),
        );
    }
}

/// **Sole writer** of [`VelocityFrameState::active`].
///
/// Suggests Surface when the craft is landed (`BodyFixed`) or below the
/// dominant body's surface-frame ceiling, Orbit otherwise; Target is never
/// auto-selected. Clears a stale override on a boundary crossing or target
/// loss, then resolves `active = override.unwrap_or(suggested)`.
pub fn update_velocity_frame(
    sim: Res<SimulationState>,
    target: Res<TargetBody>,
    mut state: ResMut<VelocityFrameState>,
) {
    let simulation = &sim.simulation;
    let ship = simulation.ship_state();
    let body_id = simulation.dominant_body();
    let body = &simulation.bodies()[body_id];

    let body_pos = simulation
        .ephemeris()
        .state(body_id, Epoch(simulation.sim_time()))
        .position;
    let altitude = (ship.position - body_pos).length() - body.radius_m;

    // Ceiling: atmosphere top where present, else authored override, else a
    // radius-derived default.
    let ceiling = body
        .terrestrial_atmosphere
        .as_ref()
        .map(|a| a.karman_line_m as f64)
        .or(body.surface_frame_ceiling_m)
        .unwrap_or(DEFAULT_CEILING_RADIUS_FRACTION * body.radius_m);

    let landed = matches!(simulation.authority(), AuthorityMode::BodyFixed { .. });
    let suggested = if landed || altitude < ceiling {
        VelocityReferenceFrame::Surface
    } else {
        VelocityReferenceFrame::Orbit
    };

    // Clear a stale override on a situation-boundary crossing...
    if state.last_suggested.is_some_and(|prev| prev != suggested) {
        state.override_choice = None;
    }
    // ...or when a Target override has lost its target.
    if state.override_choice == Some(VelocityReferenceFrame::Target) && target.target.is_none() {
        state.override_choice = None;
    }
    state.last_suggested = Some(suggested);

    state.active = state.override_choice.unwrap_or(suggested);
}
