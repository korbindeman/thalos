//! Shared semantic input layer for Thalos Bevy binaries.
//!
//! This crate owns the `bevy_enhanced_input` action/context definitions and
//! the editable RON binding file. Runtime systems in each binary should read
//! the intent resources exported here instead of querying raw keyboard or
//! mouse button state.

use bevy::input::mouse::MouseScrollUnit;
use bevy::math::Vec2;

pub mod body_editor;
mod frame_input;
pub mod game;
pub mod gating;
pub mod joystick;
pub mod settings;
pub mod shipyard;

pub use frame_input::FrameIndependentInputPlugin;

pub use bevy_enhanced_input::prelude as enhanced;

// Camera zoom is a gesture, not UI list scrolling. Trackpads emit pixel
// scroll deltas; mapping those through Bevy's conservative UI line factor
// makes zoom feel heavy, so camera consumers use a tighter gain.
const PIXELS_PER_CAMERA_SCROLL_LINE: f32 = 35.0;

pub(crate) fn camera_scroll_delta(delta: Vec2, unit: MouseScrollUnit) -> Vec2 {
    match unit {
        MouseScrollUnit::Line => delta,
        MouseScrollUnit::Pixel => delta / PIXELS_PER_CAMERA_SCROLL_LINE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_scroll_uses_trackpad_camera_gain() {
        assert_eq!(
            camera_scroll_delta(Vec2::new(35.0, -70.0), MouseScrollUnit::Pixel),
            Vec2::new(1.0, -2.0)
        );
    }

    #[test]
    fn line_scroll_stays_in_logical_lines() {
        let delta = Vec2::new(1.0, -2.0);
        assert_eq!(camera_scroll_delta(delta, MouseScrollUnit::Line), delta);
    }
}
