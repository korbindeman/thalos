//! 3D render scale that does not change UI density.
//!
//! The scene camera stays on the window so picking and `world_to_viewport`
//! keep window-logical coordinates. After extract has applied, PrepareViews
//! shrinks the 3D view's main target to `window_physical × scale` before Bevy
//! allocates color and depth; Bevy's existing upscale blit fills the
//! swapchain. UI moves to a dedicated full-resolution camera whenever
//! scale is below 1. That overlay camera must clear to transparent and
//! alpha-blend: `ClearColorConfig::None` leaves the 2D target uninitialized,
//! and on Metal that magenta (alpha 1) overwrites the 3D blit.

use bevy::camera::{Camera, CameraOutputMode, ClearColorConfig, RenderTarget};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_resource::BlendState;
use bevy::render::view::{ExtractedView, prepare_view_targets};
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::ui::IsDefaultUiCamera;
use bevy::window::PrimaryWindow;
use thalos_render_foundation::SceneViewportOverride;

use crate::graphics::{
    GraphicsPreferences, QualityOverrides, RENDER_SCALE_MIN, effective_graphics,
};

/// Runs after application camera ownership so a scaled session can steal
/// [`IsDefaultUiCamera`] without fighting the active-camera system.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderScaleSet;

/// Live render-scale state. Sole writer: [`apply_render_scale`].
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct RenderScaleState {
    pub scale: f32,
    /// True when a dedicated UI camera owns [`IsDefaultUiCamera`].
    pub presents_ui_separately: bool,
}

impl Default for RenderScaleState {
    fn default() -> Self {
        Self {
            scale: 1.0,
            presents_ui_separately: false,
        }
    }
}

impl RenderScaleState {
    /// Physical size scene passes should allocate for this camera.
    pub fn physical_viewport(&self, camera: &Camera) -> Option<UVec2> {
        let native = camera.physical_viewport_size()?;
        Some(scaled_physical_size(native, self.scale))
    }
}

/// Even physical size of the 3D main target. `scale >= 1` returns `native`.
pub fn scaled_physical_size(native: UVec2, scale: f32) -> UVec2 {
    if !scale.is_finite() || scale >= 1.0 - 1.0e-3 {
        return native.max(UVec2::ONE);
    }
    let scale = scale.clamp(RENDER_SCALE_MIN, 1.0);
    let width = ((native.x as f32 * scale).round() as u32).max(1);
    let height = ((native.y as f32 * scale).round() as u32).max(1);
    UVec2::new((width & !1).max(1), (height & !1).max(1))
}

/// Extracted onto scene cameras so the render world can shrink the main target.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub(crate) struct SceneRenderScale(pub f32);

#[derive(Component)]
struct RenderScaleUiCamera;

pub(crate) struct RenderScalePlugin;

impl Plugin for RenderScalePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<SceneRenderScale>::default())
            .add_systems(Update, apply_render_scale.in_set(RenderScaleSet));

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            // Extract writes `ExtractedCamera` through deferred commands, so a
            // same-schedule mutation after `extract_cameras` is overwritten when
            // those commands apply. Shrink the 3D target in PrepareViews, after
            // extract has landed and before Bevy allocates color/depth.
            render_app.add_systems(
                Render,
                apply_extracted_render_scale
                    .in_set(RenderSystems::PrepareViews)
                    .before(prepare_view_targets),
            );
        }
    }
}

fn apply_render_scale(
    settings: Res<GraphicsPreferences>,
    overrides: Res<QualityOverrides>,
    mut state: ResMut<RenderScaleState>,
    mut scene_override: Option<ResMut<SceneViewportOverride>>,
    mut commands: Commands,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    mut scene_cameras: Query<
        (
            Entity,
            Option<&RenderTarget>,
            Option<&SceneRenderScale>,
            &Camera,
        ),
        (
            With<crate::graphics::PreferencesCamera>,
            Without<RenderScaleUiCamera>,
        ),
    >,
    ui_cameras: Query<Entity, With<RenderScaleUiCamera>>,
    mut default_ui: Query<(Entity, Has<IsDefaultUiCamera>)>,
) {
    if std::env::var("THALOS_SCALE").is_ok() {
        return;
    }

    let scale = effective_graphics(&settings, &overrides).render_scale;
    let mut window = windows.single_mut().ok();
    let native = window.as_ref().map(|window| {
        UVec2::new(
            window.resolution.physical_width().max(1),
            window.resolution.physical_height().max(1),
        )
    });
    if let Some(window) = window.as_mut()
        && window.resolution.scale_factor_override().is_some()
    {
        window.resolution.set_scale_factor_override(None);
    }
    let camera_physical = scene_cameras.iter().find_map(|(_, target, _, camera)| {
        let targets_window = matches!(target, None | Some(RenderTarget::Window(_)));
        if !targets_window {
            return None;
        }
        camera
            .physical_target_size()
            .map(|size| scaled_physical_size(size, scale))
    });
    let scene_physical =
        camera_physical.or_else(|| native.map(|size| scaled_physical_size(size, scale)));
    let present_ui = scale < 1.0 - 1.0e-3 && native.is_some();

    if state.scale != scale || state.presents_ui_separately != present_ui {
        *state = RenderScaleState {
            scale,
            presents_ui_separately: present_ui,
        };
    }

    if let Some(scene) = scene_override.as_mut() {
        scene.physical = if present_ui { scene_physical } else { None };
    }

    for (entity, target, existing, _) in &mut scene_cameras {
        let targets_window = matches!(target, None | Some(RenderTarget::Window(_)));
        if !targets_window {
            if existing.is_some() {
                commands.entity(entity).remove::<SceneRenderScale>();
            }
            continue;
        }
        let wanted = SceneRenderScale(scale);
        if existing.is_none_or(|value| value.0 != wanted.0) {
            commands.entity(entity).insert(wanted);
        }
    }

    if present_ui {
        if ui_cameras.iter().next().is_none() {
            commands.spawn((
                Camera2d,
                overlay_ui_camera(),
                IsDefaultUiCamera,
                RenderScaleUiCamera,
                Name::new("RenderScaleUiCamera"),
            ));
        }
        for (entity, has_ui) in &mut default_ui {
            if has_ui && scene_cameras.get(entity).is_ok() {
                commands.entity(entity).remove::<IsDefaultUiCamera>();
            }
        }
    } else {
        for entity in &ui_cameras {
            commands.entity(entity).despawn();
        }
    }
}

/// Shrink the extracted 3D camera to the scaled physical size.
///
/// Must run in [`RenderSystems::PrepareViews`] *after* extract has applied and
/// *before* [`prepare_view_targets`]. `extract_cameras` queues a full-window
/// `ExtractedCamera` insert; mutating that component in `ExtractSchedule`
/// is overwritten when those commands apply, so the depth copy then skips
/// every frame (empty scene depth → opaque sky on every pixel above the
/// geometric horizon).
fn apply_extracted_render_scale(
    mut views: Query<(&SceneRenderScale, &mut ExtractedCamera, &mut ExtractedView)>,
) {
    for (scale, mut camera, mut view) in &mut views {
        if scale.0 >= 1.0 - 1.0e-3 {
            continue;
        }
        if !matches!(
            camera.target,
            Some(bevy::camera::NormalizedRenderTarget::Window(_))
        ) {
            continue;
        }
        let Some(native) = camera.physical_target_size else {
            continue;
        };
        let size = scaled_physical_size(native, scale.0);
        if size.x >= native.x && size.y >= native.y {
            continue;
        }
        camera.physical_target_size = Some(size);
        camera.physical_viewport_size = Some(size);
        // Leave `camera.viewport` unset so the upscale blit fills the window
        // instead of scissoring to a postage stamp.
        view.viewport = UVec4::new(0, 0, size.x, size.y);
    }
}

/// Full-resolution HUD camera that composites over the upscaled 3D blit.
///
/// The main target clears to transparent so undrawn pixels do not cover the
/// scene. The swapchain write loads (does not clear) and alpha-blends.
fn overlay_ui_camera() -> Camera {
    Camera {
        order: 1_000,
        clear_color: ClearColorConfig::Custom(Color::NONE),
        output_mode: CameraOutputMode::Write {
            blend_state: Some(BlendState::ALPHA_BLENDING),
            clear_color: ClearColorConfig::None,
        },
        ..default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_scale_keeps_native_size() {
        assert_eq!(
            scaled_physical_size(UVec2::new(1920, 1080), 1.0),
            UVec2::new(1920, 1080)
        );
    }

    #[test]
    fn half_scale_is_even_and_half() {
        assert_eq!(
            scaled_physical_size(UVec2::new(3024, 1898), 0.5),
            UVec2::new(1512, 948)
        );
    }

    #[test]
    fn quarter_scale_stays_at_least_one() {
        assert_eq!(
            scaled_physical_size(UVec2::new(3, 3), 0.25),
            UVec2::new(1, 1)
        );
    }

    #[test]
    fn overlay_ui_camera_clears_transparent_and_blends() {
        let camera = overlay_ui_camera();
        assert!(matches!(
            camera.clear_color,
            ClearColorConfig::Custom(color) if color == Color::NONE
        ));
        match camera.output_mode {
            CameraOutputMode::Write {
                blend_state,
                clear_color,
            } => {
                assert_eq!(blend_state, Some(BlendState::ALPHA_BLENDING));
                assert!(matches!(clear_color, ClearColorConfig::None));
            }
            other => panic!("expected Write output mode, got {other:?}"),
        }
    }
}
