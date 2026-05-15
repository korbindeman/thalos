//! Bevy-UI HUD.
//!
//! Replaces the legacy egui `hud.rs` top-bar. Layout:
//!
//! - **Top-left**: warp speed + mission time
//! - **Top-middle**: altitude + apoapsis / periapsis
//! - **Bottom-left (navball cluster)**: navball + throttle arc
//! - **Bottom-left (above navball)**: orbital velocity
//! - **Bottom-right**: Δv estimate + fuel
//! - **Bottom-left (beside navball)**: circular navigation panel
//!
//! Each panel is one source file under this folder. Sub-modules share
//! the [`theme::HudTheme`] resource for fonts and colours.

mod delta_v_panel;
mod flight_panel;
mod format;
mod fps_overlay;
pub mod input_gate;
mod nav_attitude;
mod nav_panel;
mod orbital_panel;
pub mod theme;
mod view_mode_panel;
mod warp_time_panel;

pub use warp_time_panel::TimeDisplayMode;

use bevy::prelude::*;

use crate::photo_mode::not_in_photo_mode;

pub use input_gate::UiPointerGate;

pub struct HudPlugin;

/// Shared flex container at the top-left. `warp_time_panel` and
/// `view_mode_panel` parent into this so they sit flush regardless of
/// each panel's content width.
#[derive(Resource, Clone, Copy)]
pub(super) struct TopLeftRowAnchor(pub Entity);

fn setup_top_left_row(mut commands: Commands) {
    let entity = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(20.0),
                top: Val::Px(20.0),
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(8.0),
                align_items: AlignItems::FlexStart,
                ..default()
            },
            Name::new("HudTopLeftRow"),
        ))
        .id();
    commands.insert_resource(TopLeftRowAnchor(entity));
}

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(UiMaterialPlugin::<flight_panel::ThrottleArcMaterial>::default())
            .init_resource::<UiPointerGate>()
            .init_resource::<TimeDisplayMode>()
            .init_resource::<nav_panel::ManeuverPanelState>()
            .add_systems(Startup, theme::init_theme)
            .add_systems(bevy_egui::EguiPrimaryContextPass, theme::apply_egui_theme)
            .add_systems(Startup, setup_top_left_row.after(theme::init_theme))
            .add_systems(
                Startup,
                (
                    nav_attitude::setup,
                    (warp_time_panel::setup, view_mode_panel::setup).chain(),
                    orbital_panel::setup,
                    delta_v_panel::setup,
                    flight_panel::setup.after(crate::navball::ui::setup_navball_ui),
                    fps_overlay::setup,
                )
                    .after(theme::init_theme)
                    .after(setup_top_left_row),
            )
            .add_systems(
                Startup,
                nav_panel::setup
                    .after(theme::init_theme)
                    .after(nav_attitude::setup),
            )
            .add_systems(
                Update,
                (
                    warp_time_panel::update,
                    warp_time_panel::update_pause_glyph,
                    warp_time_panel::handle_pause_click,
                    warp_time_panel::handle_time_mode_click,
                    warp_time_panel::handle_warp_level_click,
                    warp_time_panel::update_button_visuals,
                    view_mode_panel::handle_clicks,
                    view_mode_panel::update_button_visuals,
                    orbital_panel::update,
                    delta_v_panel::update,
                    flight_panel::update,
                    fps_overlay::update,
                    nav_panel::handle_clicks,
                    nav_panel::update_button_visuals,
                    nav_panel::update_autopilot_visuals,
                    nav_panel::update_maneuver_visuals,
                    nav_attitude::update_attitude,
                )
                    .run_if(not_in_photo_mode),
            )
            .add_systems(Update, input_gate::update_ui_pointer_gate)
            .add_systems(Update, hide_in_photo_mode);
    }
}

/// Marker for every HUD root container so `hide_in_photo_mode` can toggle them all.
#[derive(Component)]
pub struct HudPanel;

fn hide_in_photo_mode(
    photo: Res<crate::photo_mode::PhotoMode>,
    mut q: Query<&mut Visibility, With<HudPanel>>,
) {
    if !photo.is_changed() {
        return;
    }
    let target = if photo.active {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut vis in &mut q {
        *vis = target;
    }
}
