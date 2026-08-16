//! Explicit simulation clock boundary.
//!
//! Bevy's default `Time`/`Time<Virtual>` is intentionally left as an app
//! clock. Presentation systems that need animation while the simulation is
//! paused should use `Time<Real>` directly. Systems that advance canonical
//! state, local physics ownership, resource burn, or surface walking consume
//! this resource instead.
//!
//! **Sole writer:** [`sync_sim_clock`]. Every pause source that should halt
//! canonical/local simulation folds into that one predicate.
//!
//! # Wall vs. driven
//!
//! [`SimClockDrive`] selects how *the app's clock itself* advances:
//!
//! * [`SimClockDrive::Wall`] — interactive. One frame lasts however long it
//!   took to produce.
//! * [`SimClockDrive::Driven`] — offline render. One frame lasts exactly
//!   `dt_s` no matter how long it took, so frame *n* lands at *n · dt* and a
//!   300 ms frame can still be one sixtieth of a second of world time. This is
//!   the foundation the cinematics work stands on (`cine §3`,
//!   ADR-20260730T212556Z): a wall clock cannot render `n / fps`.
//!
//! The mechanism is Bevy's own [`TimeUpdateStrategy`], applied by
//! [`apply_clock_drive`]. Driving `Time<Real>` — rather than intercepting the
//! delta here — is deliberate: everything derived from it (`Time<Virtual>` →
//! Avian's `Time<Physics>` and `Time<Fixed>`, plus every presentation
//! animation that reads `Time<Real>` directly) becomes driven *together*. A
//! seam that only fixed this resource would leave local physics and camera
//! smoothing running on wall time, which is precisely the drift a deterministic
//! render exists to remove.
//!
//! [`sync_sim_clock`] still reads the step from [`SimClockDrive`] rather than
//! from the now-driven `Time<Real>`, so this resource is correct on the frame
//! the mode changes regardless of system ordering. Both derive from the one
//! `SimClockDrive` value; it remains the single source.
//!
//! **What must *not* move onto the driven clock:** anything measuring real
//! elapsed time — frame cost, streaming holds, timeouts. Those take `Instant`
//! deltas directly (see `thalos_diagnostics_ui::FrameSamples` and the capture driver's
//! streaming/brake holds). A wall-clock measurement hung off `Time<Real>` reads
//! as a lie the moment the clock is driven, which is why those two moved in the
//! same change that introduced this mode.

use std::time::Duration;

use bevy::prelude::*;
use bevy::time::{TimeSystems, TimeUpdateStrategy};

use crate::SimStage;
use crate::freecam::FreeCam;
use crate::loading::AppState;
use crate::pause_menu::GamePause;
use crate::rendering::SimulationState;
use crate::scenario_menu::ScenarioMenu;

// `SimClock` / `SimClockDrive` and their bounds moved to the game-state
// blackboard (Phase 5a); this module keeps the sole writer and the
// wall/driven `TimeUpdateStrategy` mechanics.
pub use thalos_game_state::clock::{SimClock, SimClockDrive};

pub struct SimClockPlugin;

impl Plugin for SimClockPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimClock>()
            .init_resource::<SimClockDrive>()
            // Must land before Bevy reads the strategy to advance `Time<Real>`.
            .add_systems(
                First,
                apply_clock_drive
                    .before(TimeSystems)
                    .run_if(resource_changed::<SimClockDrive>),
            )
            .add_systems(
                Update,
                sync_sim_clock
                    .after(crate::pause_menu::handle_escape_input)
                    .after(crate::scenario_menu::sync_menu_to_destruction)
                    .before(SimStage::Physics),
            );
    }
}

/// Map [`SimClockDrive`] onto Bevy's [`TimeUpdateStrategy`].
///
/// Runs only when the drive changes (including the frame it is added), so the
/// interactive path pays nothing.
///
/// Deliberately silent. The mode is *reported* where it has a reader: the
/// capture receipt's `clock` block, beside the terrain-residency verdict
/// (capture.md §6.2), plus one human-facing line from the headless plugin when
/// the driven clock is selected. A lane event here would have no check reading
/// it, which is cost with no signal.
pub(crate) fn apply_clock_drive(
    drive: Res<SimClockDrive>,
    mut strategy: ResMut<TimeUpdateStrategy>,
) {
    *strategy = match *drive {
        SimClockDrive::Wall => TimeUpdateStrategy::Automatic,
        SimClockDrive::Driven { dt_s } => {
            TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(dt_s))
        }
    };
}

/// Sole writer of [`SimClock`].
///
/// Pause sources:
/// - Escape menu (`GamePause`)
/// - destruction scenario picker (`ScenarioMenu::open`)
/// - start screen (`AppState::MainMenu`)
/// - freecam when the craft could **not** time-warp on enter
///   (`FreeCam::active && !FreeCam::allow_sim_time`). When the craft *was*
///   warp-eligible, freecam leaves sim time under normal warp control so
///   `.`/`,` still advance the world while framing.
/// - a modal in-game context — VAB / base editor / space-center hub — via the
///   `GameContext` sub-state (`game_context::context_freezes_sim`), replacing
///   the former `ShipyardEditor`/`BaseEditor`/`SpaceCenter` boolean reads
/// - warp pause (`warp.speed() == 0`)
///
/// Loading is deliberately **not** a pause source: the deferred surface
/// placements settle the craft and stream tiles behind the loading screen,
/// which needs sim time advancing (see `crate::surface_settle`). The sub-state
/// is absent there, so `context_freezes_sim` reads `false`.
///
/// Pausing is orthogonal to [`SimClockDrive`]: a driven clock still yields a
/// zero delta while a pause source is active, because "the world does not
/// advance" and "a frame is 1/60 s long" are independent facts. A sequence that
/// needs the world to move sets warp, exactly as an interactive session does.
pub(crate) fn sync_sim_clock(
    time: Res<Time<Real>>,
    drive: Res<SimClockDrive>,
    pause: Res<GamePause>,
    scenario: Res<ScenarioMenu>,
    app_state: Res<State<AppState>>,
    freecam: Option<Res<FreeCam>>,
    game_context: Option<Res<State<crate::game_context::GameContext>>>,
    sim: Res<SimulationState>,
    mut clock: ResMut<SimClock>,
) {
    let step_s = match *drive {
        SimClockDrive::Wall => time.delta_secs_f64(),
        SimClockDrive::Driven { dt_s } => dt_s,
    };
    let freecam_freezes = freecam
        .as_deref()
        .map(|f| f.active && !f.allow_sim_time)
        .unwrap_or(false);
    let context_freezes = crate::game_context::context_freezes_sim(game_context.as_deref());
    let paused = pause.active
        || scenario.open
        || *app_state.get() == AppState::MainMenu
        || freecam_freezes
        || context_freezes
        || sim.simulation.warp.speed() == 0.0;

    *clock = SimClock::from_writer(if paused { 0.0 } else { step_s }, paused);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the mode: a driven step must reach Bevy's own clock,
    /// because `Time<Virtual>` → `Time<Physics>`/`Time<Fixed>` and every
    /// presentation animation derive from it. Asserting only on `SimClock`
    /// would pass while local physics still ran on wall time.
    #[test]
    fn driven_mode_installs_a_manual_duration_strategy() {
        let mut app = App::new();
        app.init_resource::<TimeUpdateStrategy>()
            .init_resource::<SimClockDrive>()
            .add_systems(Update, apply_clock_drive);

        app.update();
        assert!(matches!(
            *app.world().resource::<TimeUpdateStrategy>(),
            TimeUpdateStrategy::Automatic
        ));

        app.insert_resource(SimClockDrive::driven_from_fps(50.0).unwrap());
        app.update();
        match *app.world().resource::<TimeUpdateStrategy>() {
            TimeUpdateStrategy::ManualDuration(step) => {
                assert!((step.as_secs_f64() - 0.02).abs() < 1.0e-9);
            }
            _ => panic!("driven mode did not install a manual-duration strategy"),
        }

        app.insert_resource(SimClockDrive::Wall);
        app.update();
        assert!(matches!(
            *app.world().resource::<TimeUpdateStrategy>(),
            TimeUpdateStrategy::Automatic
        ));
    }
}
