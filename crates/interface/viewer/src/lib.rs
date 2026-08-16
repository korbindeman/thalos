//! Lightweight viewer mechanism shared by planetary and planar applications.
//!
//! This crate owns physical camera optics, semantic freecam motion, and its UI.
//! Applications retain spatial truth: they project stable poses into render
//! coordinates and apply their own floor, bounds, pause, and reference-frame
//! policies around the common motion core.

mod motion;
mod optics;
mod panel;
mod viewpoint_ui;
mod viewpoints;

use bevy::prelude::*;

pub use motion::{
    LevelLock, VIEWER_MAX_SPEED_M_S, VIEWER_MIN_SPEED_M_S, ViewerIntent, ViewerPose, drive_motion,
    format_speed, level_lock_authority, settle_level_lock, speed_multiplier, speed_reference,
    update_spring_zoom,
};
pub use optics::{CameraOptics, sync_camera_optics_projection};
pub use thalos_render_model::{
    ScriptedViewpoint, Viewpoint, ViewpointCatalog, ViewpointFrame, ViewpointSpawn,
    viewpoint_id_from_name,
};
pub use viewpoints::{
    CurrentViewpoint, PendingViewpointApply, ViewpointApplyTarget, ViewpointFallbacks,
    ViewpointPlugin, ViewpointSet, ViewpointSnapshot, ViewpointStartupSet, ViewpointStore,
    ViewpointUiState, read_viewpoint_catalog, unique_id, write_viewpoint_catalog,
};

/// Marks the camera whose optics and freecam panel are controlled by this crate.
#[derive(Component, Debug, Clone, Copy)]
pub struct ViewerCamera;

/// Marks shared viewer UI roots so applications can hide them for clean shots.
#[derive(Component, Debug, Clone, Copy)]
pub struct ViewerUiRoot;

/// User-controlled settings shared by both applications.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct ViewerPreferences {
    pub base_speed_m_s: f64,
    pub level_to_up: bool,
    pub ground_collision: bool,
}

impl Default for ViewerPreferences {
    fn default() -> Self {
        Self {
            base_speed_m_s: 100.0,
            level_to_up: true,
            ground_collision: true,
        }
    }
}

/// Application projection consumed by the shared panel.
#[derive(Resource, Debug, Clone, PartialEq)]
pub struct ViewerStatus {
    pub active: bool,
    pub panel_visible: bool,
    pub interaction_blocked: bool,
    pub anchor_label: String,
    pub altitude_agl_m: Option<f64>,
}

impl Default for ViewerStatus {
    fn default() -> Self {
        Self {
            active: false,
            panel_visible: true,
            interaction_blocked: false,
            anchor_label: "—".into(),
            altitude_agl_m: None,
        }
    }
}

/// True while the pointer is over the shared panel.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct ViewerUiCapture {
    pub pointer_busy: bool,
}

pub struct ViewerPlugin {
    interactive: bool,
    shortcut_hint: &'static str,
}

impl ViewerPlugin {
    pub const fn new(interactive: bool, shortcut_hint: &'static str) -> Self {
        Self {
            interactive,
            shortcut_hint,
        }
    }
}

impl Plugin for ViewerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewerPreferences>()
            .init_resource::<ViewerStatus>()
            .init_resource::<ViewerUiCapture>()
            .add_systems(
                Update,
                sync_camera_optics_projection.after(ViewpointSet::Apply),
            );

        if self.interactive {
            if !app.is_plugin_added::<thalos_ui::ThalosUiPlugin>() {
                app.add_plugins(thalos_ui::ThalosUiPlugin);
            }
            app.add_plugins(panel::ViewerPanelPlugin::new(self.shortcut_hint));
        }
    }
}
