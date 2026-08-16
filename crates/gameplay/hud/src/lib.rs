// Bevy system signatures routinely exceed clippy's argument and type
// complexity budgets; same crate-level allowance as thalos_runtime.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

//! Bevy-UI HUD.
//!
//! Replaces the legacy egui `hud.rs` top-bar. Layout:
//!
//! - **Top-left**: warp speed + mission time
//! - **Top-middle**: altitude + apoapsis / periapsis
//! - **Top-middle (below the bar)**: on-foot (EVA) status pill
//! - **Bottom-left (navball cluster)**: navball + throttle arc
//! - **Bottom-left (above navball)**: orbital velocity
//! - **Bottom-left (instrument stack)**: autopilot and conditional maneuver card
//! - **Bottom-right**: per-stage staging stack (Δv + fuel per stage)
//! - **Bottom-left (beside navball)**: circular navigation panel
//! - **Screen centre (HUD mode)**: PFD overlay replacing the navball
//!   cluster — pitch ladder, speed/altitude tapes, direction markers
//!   ([`pfd_panel`], toggled by the top-left BALL/HUD selector)
//! - **Top-right (MFD slot)**: a contextual, customizable widget slot
//!   ([`mfd`]) that auto-selects the widget relevant to the current
//!   flight context (orbital trajectory plot, airliner navigation
//!   display, …), with a manual pin/hide override
//!
//! Each panel is one source file under this folder. Sub-modules share
//! the [`theme::HudTheme`] resource for fonts and colours.

mod atmo_panel;
mod eva_panel;
mod flight_config_panel;
mod flight_panel;
pub mod format;
mod geo;
pub mod input_gate;
pub mod mfd;
mod nav_attitude;
mod nav_panel;
pub mod navball;
mod orbital_panel;
mod pfd_panel;
mod staging_panel;
pub mod theme;
pub mod velocity_frame;
mod view_mode_panel;
mod warp_time_panel;

pub use orbital_panel::OrbitWidgetState;
pub use warp_time_panel::TimeDisplayMode;

use bevy::prelude::*;

use thalos_photo_mode::not_in_photo_mode;

pub use input_gate::{UiKeyboardGate, UiPointerGate};

/// Density (kg/m³) above which the craft counts as "in atmosphere".
///
/// One definition for the whole HUD: the atmospheric readout pill's visibility,
/// [`mfd::FlightContext::in_atmosphere`] (which drives widget auto-selection),
/// and the display-unit situation all have to agree on where the atmosphere
/// starts, or a panel appears in units its neighbours don't share.
pub(crate) const IN_ATMOSPHERE_DENSITY: f64 = 1.0e-6;

pub struct HudPlugin;

/// Shared flex container at the top-left. `warp_time_panel` and
/// `view_mode_panel` parent into this so they sit flush regardless of
/// each panel's content width.
#[derive(Resource, Clone, Copy)]
pub(crate) struct TopLeftRowAnchor(pub Entity);

/// Shared vertical lane above the navball. The velocity readout, autopilot,
/// and conditional maneuver card parent here so dynamic content stacks instead
/// of competing through independent absolute offsets.
#[derive(Resource, Clone, Copy)]
pub(crate) struct BottomLeftFlightStackAnchor(pub Entity);

fn setup_top_left_row(mut commands: Commands) {
    let entity = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(16.0),
                top: Val::Px(16.0),
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(6.0),
                align_items: AlignItems::FlexStart,
                ..default()
            },
            Name::new("HudTopLeftRow"),
        ))
        .id();
    commands.insert_resource(TopLeftRowAnchor(entity));
}

fn setup_bottom_left_flight_stack(mut commands: Commands) {
    use navball::ui::{NAVBALL_BOTTOM_PX, NAVBALL_LEFT_PX, NAVBALL_SIZE_PX};

    let entity = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(NAVBALL_LEFT_PX),
                bottom: Val::Px(NAVBALL_BOTTOM_PX + NAVBALL_SIZE_PX + 8.0),
                flex_direction: FlexDirection::ColumnReverse,
                row_gap: Val::Px(7.0),
                align_items: AlignItems::FlexStart,
                ..default()
            },
            Name::new("HudBottomLeftFlightStack"),
        ))
        .id();
    commands.insert_resource(BottomLeftFlightStackAnchor(entity));
}

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(UiMaterialPlugin::<flight_panel::ThrottleArcMaterial>::default())
            .add_plugins(mfd::MfdPlugin)
            .init_resource::<UiPointerGate>()
            .init_resource::<UiKeyboardGate>()
            .init_resource::<TimeDisplayMode>()
            .init_resource::<nav_panel::ManeuverPanelState>()
            .init_resource::<orbital_panel::AltitudeDisplay>()
            .init_resource::<orbital_panel::OrbitWidgetState>()
            .init_resource::<pfd_panel::NavDisplayMode>()
            .register_type::<pfd_panel::NavDisplayMode>()
            .add_systems(Startup, theme::init_theme.after(thalos_ui::init_ui_theme))
            .add_systems(
                Startup,
                (setup_top_left_row, setup_bottom_left_flight_stack).after(theme::init_theme),
            )
            .add_systems(
                Startup,
                (
                    nav_attitude::setup,
                    (
                        warp_time_panel::setup,
                        view_mode_panel::setup,
                        pfd_panel::setup_toggle,
                    )
                        .chain(),
                    orbital_panel::setup,
                    staging_panel::setup,
                    flight_panel::setup.after(crate::navball::ui::setup_navball_ui),
                    eva_panel::setup,
                    atmo_panel::setup,
                    flight_config_panel::setup,
                    pfd_panel::setup,
                )
                    .after(theme::init_theme)
                    .after(setup_top_left_row)
                    .after(setup_bottom_left_flight_stack),
            )
            .add_systems(
                Startup,
                nav_panel::setup
                    .after(theme::init_theme)
                    .after(nav_attitude::setup)
                    .after(flight_panel::setup),
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
                    orbital_panel::handle_click,
                    orbital_panel::update_orbit_widget,
                    orbital_panel::handle_orbit_widget,
                    staging_panel::update,
                    flight_panel::update,
                    flight_panel::handle_velocity_frame_click,
                    eva_panel::update,
                    (
                        atmo_panel::update,
                        flight_config_panel::handle_clicks,
                        flight_config_panel::update.after(flight_config_panel::handle_clicks),
                    ),
                    // Nested tuple: keeps the outer system-tuple within Bevy's
                    // 20-element `IntoScheduleConfigs` arity limit.
                    (
                        nav_panel::handle_clicks,
                        nav_panel::update_button_visuals,
                        nav_panel::update_autopilot_visuals,
                        nav_panel::update_maneuver_visuals,
                        nav_attitude::update_attitude,
                    ),
                    (
                        pfd_panel::handle_mode_clicks,
                        pfd_panel::update_mode_button_visuals,
                        pfd_panel::sync_visibility,
                        pfd_panel::update_attitude_display,
                        pfd_panel::update_tapes.after(orbital_panel::update),
                        pfd_panel::update_annunciators,
                        pfd_panel::update_approach_guidance,
                    )
                        .chain(),
                )
                    .after(thalos_game_state::sched::SimStage::Sync)
                    // The HUD is part of the flight scene, not the modal editors
                    // / hub; its updates stand down outside `GameContext::Flight`
                    // (the panels are hidden too), so a per-frame visibility setter
                    // like `pfd_panel::sync_visibility` can't re-show the flight HUD
                    // over the god-view. `flight_or_no_context` also keeps HUD
                    // updates running during Loading / MainMenu, as the old
                    // `*_closed` chain did.
                    .run_if(
                        not_in_photo_mode
                            .and_then(thalos_game_state::context::flight_or_no_context),
                    ),
            )
            .add_systems(Update, input_gate::update_ui_input_gates)
            .add_systems(Update, hide_in_photo_mode);
    }
}

// Moved to the blackboard (Phase 5b) so the editors can hide the same set.
pub use thalos_game_state::ui::HudPanel;

fn hide_in_photo_mode(
    photo: Res<thalos_photo_mode::PhotoMode>,
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
