//! Thalos game UI kit.
//!
//! The one home for the interface's look and building blocks: design tokens
//! ([`tokens`]), the frosted-glass panel surface ([`glass`]), and the widget
//! library ([`widgets`]). Screens in the game crate compose these — they do
//! not define their own colours, fonts, paddings, or interaction styling.
//!
//! Iterate on the look with the kitchen-sink testbed:
//! `just ui-preview` renders every component headlessly to a PNG an agent can
//! read; `just ui-preview-window` opens it interactively.
//!
//! ## Usage
//!
//! ```ignore
//! app.add_plugins(ThalosUiPlugin);
//! // mark the scene camera so panels can frost over it:
//! commands.entity(camera).insert(UiBackdropSource);
//! // spawn a panel:
//! commands.spawn((panel_node(), theme.glass(), ...));
//! ```

#![allow(clippy::type_complexity)]

pub mod glass;
pub mod hud_theme;
pub mod tokens;
pub mod widgets;

use bevy::prelude::*;

pub use glass::{GlassMaterial, UiBackdrop, UiBackdropSource};
pub use hud_theme::HudTheme;
pub use tokens::*;
pub use widgets::*;

/// Adds the glass material + backdrop pass, loads the theme, and runs the
/// shared widget systems. Consumer `Startup` systems that read [`UiTheme`]
/// must be ordered `.after(init_ui_theme)`.
pub struct ThalosUiPlugin;

impl Plugin for ThalosUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(glass::GlassPlugin)
            .init_resource::<TextFieldFocus>()
            .add_message::<TextFieldSubmit>()
            .add_systems(Startup, (init_ui_theme, widgets::toast::setup_toast_area))
            .add_systems(
                Update,
                (
                    style_buttons,
                    drive_sliders,
                    update_slider_visuals.after(drive_sliders),
                    drive_checkboxes,
                    update_checkbox_visuals.after(drive_checkboxes),
                    drive_cycles,
                    update_cycle_visuals.after(drive_cycles),
                    scroll_scrollables,
                    focus_text_fields,
                    apply_text_field_input.after(focus_text_fields),
                    update_text_field_visuals.after(apply_text_field_input),
                    update_toasts,
                ),
            );
    }
}

/// Load fonts and build the shared glass materials. **Sole writer** of
/// [`UiTheme`].
pub fn init_ui_theme(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut glass_materials: ResMut<Assets<GlassMaterial>>,
    backdrop: Option<Res<UiBackdrop>>,
) {
    let backdrop_handle = backdrop.map(|b| b.handle.clone());
    let glass_regular = glass_materials.add(GlassMaterial::new(
        tokens::GLASS_TINT,
        backdrop_handle.clone(),
    ));
    let glass_strong = glass_materials.add(GlassMaterial::new(
        tokens::GLASS_TINT_STRONG,
        backdrop_handle,
    ));
    commands.insert_resource(UiTheme {
        font_display: FontSource::Handle(asset_server.load("fonts/Inter-Light.ttf")),
        font_ui: FontSource::Handle(asset_server.load("fonts/Inter-Regular.ttf")),
        font_strong: FontSource::Handle(asset_server.load("fonts/Inter-SemiBold.ttf")),
        font_mono: FontSource::Handle(asset_server.load("fonts/FiraCode-Regular.ttf")),
        glass_regular,
        glass_strong,
    });
}
