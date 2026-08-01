//! # `thalos_game_state` — the game-state blackboard
//!
//! The shared resource/component vocabulary of the running game: app/world
//! state, the in-`Running` mode context, the sim clock, the evaluated solar
//! system, snapshots and mirrors, the view anchor, and the small scene
//! vocabulary (`PlayerShip`, `CelestialBody`, …).
//!
//! **Types, accessors, and single-writer doc comments only.** The systems
//! that write these resources live in `thalos_runtime` (or the feature crate
//! that owns them); each writable resource names its sole writer in its doc
//! comment. Feature crates depend on this crate — never on each other — and
//! this crate depends only downward on bevy and the pure domain crates
//! (ADR-20260731T024003Z; layers in `docs/architecture.md`).
//!
//! Append-biased: adding a type is cheap, reshaping one rebuilds every
//! feature crate — batch reshapes.

pub mod app;
pub mod autoflight;
pub mod camera;
pub mod clock;
pub mod context;
pub mod coords;
pub mod craft;
pub mod debug;
pub mod flight;
pub mod maneuver_plan;
pub mod map;
pub mod nav;
pub mod regime;
pub mod relaunch;
pub mod scenario;
pub mod scene;
pub mod sched;
pub mod solar_system;
pub mod structures;
pub mod surface_frame;
pub mod ui;
pub mod units;
pub mod view_anchor;

pub use app::{AppState, WorldState};
pub use autoflight::{
    AttitudeChannel, AutoflightAnnunciation, AutoflightLocks, AutoflightPolicy, AutoflightRequest,
    BurnArm, FlightProgram, OverrideOutcome, ProgramOverridePolicy, SequenceEvent, ThrottleChannel,
};
pub use clock::{SimClock, SimClockDrive};
pub use context::{ContextHistory, GameContext, InitialContext};
pub use coords::{RenderFrame, RenderGhostFocus, RenderOrigin, WorldScale};
pub use craft::CraftStateMirror;
pub use map::{MapContext, MapProjection, MapSnapshot, ProjectedBodyState};
pub use regime::{AvianAuthority, AvianRole, CraftRegimeState};
pub use relaunch::{RelaunchRequest, RelaunchSpec, SpaceportLaunchRequest};
pub use scenario::SpawnSituation;
pub use scene::{ActiveCraft, CameraExposure, CelestialBody, PlayerShip, ShipMarker};
pub use sched::{RealizeControlSet, SimStage};
pub use solar_system::{BodyEnvironmentState, SimulationState, SolarSystemState};
pub use ui::{HideInPhotoMode, HudPanel, PhotoMode, UiKeyboardGate, UiPointerGate};
pub use units::{AviationUnits, UnitDomain, UnitSystem, UnitsSettings};
pub use view_anchor::{AnchorBody, ViewAnchor};
