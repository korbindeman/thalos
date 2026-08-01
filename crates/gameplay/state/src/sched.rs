//! Shared schedule vocabulary: the stage sets systems across crates order
//! against. The runtime owns the schedule itself; feature crates only name
//! these sets.

use bevy::prelude::*;

/// Execution stages within `Update`, ordered so that physics advances before
/// positions are written, and positions are written before the camera reads
/// them.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum SimStage {
    /// Bridge: advance sim_time and ship state.
    Physics,
    /// Rendering: update body/ship transforms from sim state.
    Sync,
    /// Camera: compute camera transform from body transforms.
    Camera,
}

/// The set containing the runtime's `realize_control` (the one control
/// pipeline). Feature systems that must land their inputs before control is
/// realized order `.before(RealizeControlSet)`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RealizeControlSet;

/// The set containing the runtime's `sync_solar_system_state` (the sole
/// writer of `SolarSystemState`). Order `.after` it to read this frame's
/// evaluated states.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SolarSystemSyncSet;

/// The set containing the runtime's `update_prediction` (trajectory-plan
/// rebuild) — order `.after` it to read this frame's prediction.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PredictionSet;

/// The set containing the runtime's `update_render_frame` (map focus/frame
/// resolution).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderFrameSet;
