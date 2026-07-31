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
//! deltas directly (see `perf::collect_frame` and the capture driver's
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

/// Fastest driven step accepted (10 000 fps). Below this the `Duration`
/// conversion starts losing resolution and no output format wants it.
pub const MIN_DRIVEN_DT_S: f64 = 1.0e-4;
/// Slowest driven step accepted (1 fps). A larger step would push local physics
/// past what its integrator resolves in one frame.
pub const MAX_DRIVEN_DT_S: f64 = 1.0;

/// How the app's clock advances.
///
/// Input to [`sync_sim_clock`] and to [`apply_clock_drive`] — **not** a second
/// clock. See the module docs for why the driven mode goes through
/// [`TimeUpdateStrategy`] rather than being applied here.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq)]
pub enum SimClockDrive {
    /// Interactive: a frame lasts as long as it took.
    #[default]
    Wall,
    /// Offline render: a frame lasts exactly `dt_s`.
    Driven { dt_s: f64 },
}

impl SimClockDrive {
    /// Driven mode at a frame rate, e.g. `60.0` → a 1/60 s step.
    pub fn driven_from_fps(fps: f64) -> Result<Self, String> {
        if !fps.is_finite() || fps <= 0.0 {
            return Err(format!(
                "driven clock fps {fps} must be finite and positive"
            ));
        }
        Self::driven_from_dt_s(1.0 / fps)
    }

    /// Driven mode at an explicit step.
    pub fn driven_from_dt_s(dt_s: f64) -> Result<Self, String> {
        if !dt_s.is_finite() || !(MIN_DRIVEN_DT_S..=MAX_DRIVEN_DT_S).contains(&dt_s) {
            return Err(format!(
                "driven clock step {dt_s} s is outside {MIN_DRIVEN_DT_S}..={MAX_DRIVEN_DT_S}"
            ));
        }
        Ok(Self::Driven { dt_s })
    }

    /// Parse the CLI/environment adapter form: `wall`, `driven` (60 fps),
    /// `driven:<fps>`, or a bare frame rate.
    pub fn parse(raw: &str) -> Result<Self, String> {
        const DEFAULT_DRIVEN_FPS: f64 = 60.0;
        let raw = raw.trim();
        match raw.to_ascii_lowercase().as_str() {
            "" => Err("clock mode cannot be empty".into()),
            "wall" | "real" => Ok(Self::Wall),
            "driven" | "fixed" => Self::driven_from_fps(DEFAULT_DRIVEN_FPS),
            lowered => {
                let fps = lowered
                    .strip_prefix("driven:")
                    .or_else(|| lowered.strip_prefix("fixed:"))
                    .unwrap_or(lowered);
                let fps = fps.parse::<f64>().map_err(|_| {
                    format!("clock mode {raw:?} must be wall, driven, driven:<fps>, or <fps>")
                })?;
                Self::driven_from_fps(fps)
            }
        }
    }

    /// The fixed step, or `None` on the wall clock.
    pub fn dt_s(self) -> Option<f64> {
        match self {
            Self::Wall => None,
            Self::Driven { dt_s } => Some(dt_s),
        }
    }
}

/// Frame-local simulation clock.
///
/// `delta_s` is the frame's step while simulation is running and zero while any
/// sim pause source is active. Consumers should use `delta_secs_f64()` for
/// physical integration rather than reaching for Bevy's global `Time` resource.
/// The mode itself is **not** mirrored here — [`SimClockDrive`] is the one
/// place it lives, and consumers that need to report it (the capture receipt)
/// read that resource. This carries the delta; that carries the mode.
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

    *clock = SimClock {
        delta_s: if paused { 0.0 } else { step_s },
        paused,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drive_parses_the_adapter_forms() {
        assert_eq!(SimClockDrive::parse("wall").unwrap(), SimClockDrive::Wall);
        assert_eq!(SimClockDrive::parse(" Real ").unwrap(), SimClockDrive::Wall);
        assert_eq!(
            SimClockDrive::parse("driven").unwrap(),
            SimClockDrive::Driven { dt_s: 1.0 / 60.0 }
        );
        assert_eq!(
            SimClockDrive::parse("driven:30").unwrap(),
            SimClockDrive::Driven { dt_s: 1.0 / 30.0 }
        );
        assert_eq!(
            SimClockDrive::parse("24").unwrap(),
            SimClockDrive::Driven { dt_s: 1.0 / 24.0 }
        );
    }

    #[test]
    fn drive_rejects_unusable_steps() {
        // Out of range in both directions, and non-numeric.
        assert!(SimClockDrive::driven_from_fps(0.0).is_err());
        assert!(SimClockDrive::driven_from_fps(-60.0).is_err());
        assert!(SimClockDrive::driven_from_fps(f64::NAN).is_err());
        assert!(SimClockDrive::driven_from_fps(0.5).is_err()); // 2 s step
        assert!(SimClockDrive::driven_from_fps(1.0e6).is_err()); // 1 µs step
        assert!(SimClockDrive::parse("").is_err());
        assert!(SimClockDrive::parse("sometimes").is_err());
        assert!(SimClockDrive::parse("driven:fast").is_err());
    }

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
