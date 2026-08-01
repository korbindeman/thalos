//! Top-level application and world state.

use bevy::prelude::*;

/// Top-level app state. Starts in [`Loading`](AppState::Loading) so the very
/// first frame is covered by the loading screen; finishes into
/// [`MainMenu`](AppState::MainMenu) (bare launch) or
/// [`Running`](AppState::Running) (`just game <scenario>`), per
/// [`LoadDestination`]. The start screen re-enters `Loading` for scenarios
/// that need a deferred placement pass (runway).
#[derive(States, Default, Clone, Copy, Eq, PartialEq, Debug, Hash)]
pub enum AppState {
    #[default]
    Loading,
    MainMenu,
    Running,
}

/// Whether the game world — celestial-body entities, the player ship
/// visuals, the procedural sky — has been spawned. A bare menu boot starts
/// [`Absent`](WorldState::Absent): the start screen is a lightweight UI over
/// an empty scene (nothing simulates or streams behind it), and the world is
/// built only when the player picks PLAY / a scenario — the menu flips this
/// to [`Live`](WorldState::Live), and the world-spawn systems (registered on
/// `OnEnter(WorldState::Live)` across the runtime's `rendering`, `ship_view`,
/// `sky_render`) run behind that action's loading pass. A `just game
/// <scenario>` boot inserts `Live` directly, so the same `OnEnter` fires
/// on the first frame and the boot is unchanged.
///
/// One-way: nothing ever sets it back to `Absent` (the world is never torn
/// down; returning to the menu from flight keeps the world live and the
/// menu routes through the existing live-world paths).
#[derive(States, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum WorldState {
    /// No world entities exist (bare menu boot, before the first start).
    #[default]
    Absent,
    /// The world has been (or is being) spawned.
    Live,
}

/// Where the current loading pass goes when it completes. Inserted by
/// the launcher (start screen for a bare launch, `Running` otherwise); the
/// start screen sets it to `Running` before re-entering `Loading`.
#[derive(Resource, Debug, Clone, Copy)]
pub struct LoadDestination(pub AppState);

impl Default for LoadDestination {
    fn default() -> Self {
        Self(AppState::Running)
    }
}
