//! Bevy-free render-facing inputs shared by application adapters.
//!
//! Applications resolve simulation, pause, warp, and wall-clock policy before
//! constructing these immutable records. Rendering mechanisms consume the
//! records without importing either application's runtime state.

use std::fmt;

mod optics;
mod plan;
mod viewpoint;

pub use optics::{
    CameraLens, CameraLensModel, CameraOptics, CameraSensor, CapturedCameraState,
    FULL_FRAME_GATE_WIDTH_MM, MAX_FOCAL_LENGTH_MM, MIN_FOCAL_LENGTH_MM, SensorCrop, reduced_aspect,
};

pub use plan::{
    AtmosphereAdapter, CloudAdapter, FarBodyAdapter, LightingAdapter, OceanAdapter,
    RenderCapabilities, RenderPlan, RenderPlanError, SpatialAdapter, TerrainAdapter,
    ValidatedRenderPlan,
};
pub use viewpoint::{
    CAPTURE_PRESETS, ScriptedViewpoint, VIEWPOINT_CATALOG_SCHEMA, Viewpoint, ViewpointCatalog,
    ViewpointFrame, ViewpointSpawn, valid_viewpoint_id, viewpoint_id_from_name,
};

/// Validated current and previous render epochs for one frame.
///
/// Epochs are application-defined seconds. They may be negative, but must be
/// finite and monotonic within the frame. A paused or newly initialized frame
/// uses equal epochs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderFrameTime {
    previous_epoch_s: f64,
    current_epoch_s: f64,
}

impl RenderFrameTime {
    /// Validate a chronological pair of render epochs.
    pub fn new(previous_epoch_s: f64, current_epoch_s: f64) -> Result<Self, FrameTimeError> {
        if !previous_epoch_s.is_finite() || !current_epoch_s.is_finite() {
            return Err(FrameTimeError::NonFinite);
        }
        if previous_epoch_s > current_epoch_s {
            return Err(FrameTimeError::OutOfOrder);
        }
        Ok(Self {
            previous_epoch_s,
            current_epoch_s,
        })
    }

    pub fn previous_epoch_s(self) -> f64 {
        self.previous_epoch_s
    }

    pub fn current_epoch_s(self) -> f64 {
        self.current_epoch_s
    }

    pub fn delta_s(self) -> f64 {
        self.current_epoch_s - self.previous_epoch_s
    }
}

/// Why a render frame-time record was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameTimeError {
    NonFinite,
    OutOfOrder,
}

impl fmt::Display for FrameTimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => formatter.write_str("render frame epochs must be finite"),
            Self::OutOfOrder => {
                formatter.write_str("previous render epoch must not follow the current epoch")
            }
        }
    }
}

impl std::error::Error for FrameTimeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_stationary_and_advancing_frames() {
        let stationary = RenderFrameTime::new(42.0, 42.0).unwrap();
        assert_eq!(stationary.delta_s(), 0.0);

        let advancing = RenderFrameTime::new(42.0, 42.25).unwrap();
        assert_eq!(advancing.previous_epoch_s(), 42.0);
        assert_eq!(advancing.current_epoch_s(), 42.25);
        assert_eq!(advancing.delta_s(), 0.25);
    }

    #[test]
    fn rejects_non_finite_epochs() {
        assert_eq!(
            RenderFrameTime::new(f64::NAN, 0.0),
            Err(FrameTimeError::NonFinite)
        );
        assert_eq!(
            RenderFrameTime::new(0.0, f64::INFINITY),
            Err(FrameTimeError::NonFinite)
        );
    }

    #[test]
    fn rejects_previous_epoch_after_current_epoch() {
        assert_eq!(
            RenderFrameTime::new(10.0, 9.0),
            Err(FrameTimeError::OutOfOrder)
        );
    }

    #[test]
    fn preserves_large_epochs_without_narrowing() {
        let previous = 1.0e12;
        let current = previous + 0.25;
        let frame = RenderFrameTime::new(previous, current).unwrap();

        assert_eq!(frame.previous_epoch_s(), previous);
        assert_eq!(frame.current_epoch_s(), current);
        assert_eq!(frame.delta_s(), 0.25);
    }
}
