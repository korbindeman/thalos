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

use bevy::prelude::*;

use crate::SimStage;
use crate::freecam::FreeCam;
use crate::loading::AppState;
use crate::pause_menu::GamePause;
use crate::rendering::SimulationState;
use crate::scenario_menu::ScenarioMenu;

/// Frame-local simulation clock.
///
/// `delta_s` is wall-clock delta while simulation is running and zero while
/// any sim pause source is active. Consumers should use `delta_secs_f64()` for
/// physical integration rather than reaching for Bevy's global `Time` resource.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct SimClock {
    delta_s: f64,
    paused: bool,
}

impl SimClock {
    pub fn delta_secs_f64(self) -> f64 {
        self.delta_s
    }

    pub fn is_paused(self) -> bool {
        self.paused
    }
}

pub struct SimClockPlugin;

impl Plugin for SimClockPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimClock>().add_systems(
            Update,
            sync_sim_clock
                .after(crate::pause_menu::handle_escape_input)
                .after(crate::scenario_menu::sync_menu_to_destruction)
                .before(SimStage::Physics),
        );
    }
}

/// Sole writer of [`SimClock`].
///
/// Pause sources:
/// - Escape menu (`GamePause`)
/// - destruction scenario picker (`ScenarioMenu::open`)
/// - start screen (`AppState::MainMenu`)
/// - freecam (`FreeCam::active`)
/// - a modal in-game context — VAB / base editor / space-center hub — via the
///   `GameContext` sub-state (`game_context::context_freezes_sim`), replacing
///   the former `ShipyardEditor`/`BaseEditor`/`SpaceCenter` boolean reads
/// - warp pause (`warp.speed() == 0`)
///
/// Loading is deliberately **not** a pause source: the deferred surface
/// placements settle the craft and stream tiles behind the loading screen,
/// which needs sim time advancing (see `crate::surface_settle`). The sub-state
/// is absent there, so `context_freezes_sim` reads `false`.
pub(crate) fn sync_sim_clock(
    time: Res<Time<Real>>,
    pause: Res<GamePause>,
    scenario: Res<ScenarioMenu>,
    app_state: Res<State<AppState>>,
    freecam: Option<Res<FreeCam>>,
    game_context: Option<Res<State<crate::game_context::GameContext>>>,
    sim: Res<SimulationState>,
    mut clock: ResMut<SimClock>,
) {
    let wall_delta_s = time.delta_secs_f64();
    let freecam_active = freecam.as_deref().map(|f| f.active).unwrap_or(false);
    let context_freezes = crate::game_context::context_freezes_sim(game_context.as_deref());
    let paused = pause.active
        || scenario.open
        || *app_state.get() == AppState::MainMenu
        || freecam_active
        || context_freezes
        || sim.simulation.warp.speed() == 0.0;

    *clock = SimClock {
        delta_s: if paused { 0.0 } else { wall_delta_s },
        paused,
    };
}
