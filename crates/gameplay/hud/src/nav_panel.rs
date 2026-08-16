//! Circular navigation panel: 6 mode buttons + 3 colour-coded axis
//! lines crossing at the centre, with a 3D attitude indicator
//! ([`nav_attitude`]) in the middle. Maneuver/target-only modes live
//! in the straight strip below the circle so the orbital-frame wheel
//! stays visually stable. Ship-assist toggles (RCS/SAS) sit in a
//! compact strip above the circle.
//!
//! Button layout (hexagonal, 60° intervals):
//!   12 o'clock  Normal
//!    2          Prograde
//!    4          Radial-Out
//!    6          Anti-Normal
//!    8          Retrograde
//!   10          Radial-In
//!
//! Axes (each tinted by the mode pair's marker colour):
//!   vertical    Normal ↔ Anti-Normal      (purple)
//!   "/"         Prograde ↔ Retrograde     (yellow)
//!   "\"         Radial-Out ↔ Radial-In    (cyan)
//!
//! The 3D attitude indicator uses an isometric projection so the three
//! orthogonal orbital axes (prograde, normal, radial-out) project to
//! these six hex positions exactly.

use bevy::prelude::*;
use bevy::ui::Val2;

use crate::nav_attitude::NavAttitudeRenderTarget;
use crate::navball::markers::{MarkerIconState, MarkerKind, marker_icon_image};
use crate::navball::ui::{NAVBALL_BOTTOM_PX, NAVBALL_LEFT_PX, NAVBALL_SIZE_PX};
use crate::theme::HudTheme;
use crate::{BottomLeftFlightStackAnchor, HudPanel};
use thalos_game_state::SimulationState;
use thalos_game_state::autoflight::{
    AttitudeChannel, AutoflightAnnunciation, AutoflightPolicy, AutoflightRequest, FlightProgram,
    ThrottleChannel,
};
use thalos_game_state::flight::ControlLocks;
use thalos_game_state::maneuver_plan::ManeuverPlan;
use thalos_game_state::nav::LandAutopilot;
use thalos_game_state::nav::RouteState;
use thalos_game_state::nav::TargetBody;
use thalos_game_state::nav::{Autopilot, AutopilotBurnSchedule, AutopilotState};
use thalos_game_state::nav::{NavigationMode, NavigationState};
use thalos_game_state::nav::{WarpToManeuver, find_next_maneuver};
use thalos_game_state::units::UnitDomain;

/// Diameter of the circular panel (px).
const PANEL_DIAMETER: f32 = 168.0;
/// Sits immediately right of the navball, sharing its bottom-left margin.
const PANEL_LEFT_PX: f32 = NAVBALL_LEFT_PX + NAVBALL_SIZE_PX + 12.0;
/// Vertically centred against the navball.
const PANEL_BOTTOM_PX: f32 = NAVBALL_BOTTOM_PX + (NAVBALL_SIZE_PX - PANEL_DIAMETER) * 0.5;

const BUTTON_SIZE: f32 = 32.0;
const BUTTON_RING_RADIUS: f32 = PANEL_DIAMETER * 0.5 - BUTTON_SIZE * 0.5 - 9.0;

const CENTER_SIZE: f32 = 56.0;

const AXIS_LENGTH: f32 = (BUTTON_RING_RADIUS - BUTTON_SIZE * 0.5) * 2.0;
const AXIS_THICKNESS: f32 = 1.5;

const ASSIST_PANEL_BOTTOM_PX: f32 = PANEL_BOTTOM_PX + PANEL_DIAMETER + 7.0;
const ASSIST_PANEL_HEIGHT: f32 = 33.0;
const ASSIST_BUTTON_WIDTH: f32 = 64.0;
const ASSIST_BUTTON_HEIGHT: f32 = 25.0;

/// Minimum height for the autopilot panel. The panel sizes to its content so
/// the LAND reason line can appear and wrap without clipping; this only keeps
/// it from shrinking below the two-chip layout when there is nothing to say.
const AUTOPILOT_PANEL_MIN_HEIGHT: f32 = 83.0;
const AUTOPILOT_BUTTON_WIDTH: f32 = 82.0;
const AUTOPILOT_BUTTON_HEIGHT: f32 = 27.0;

const MANEUVER_PANEL_HEIGHT: f32 = 92.0;
const MANEUVER_BAR_HEIGHT: f32 = 7.0;
const MANEUVER_WARP_BUTTON_WIDTH: f32 = 50.0;
const MANEUVER_WARP_BUTTON_HEIGHT: f32 = 22.0;

const UTILITY_PANEL_BOTTOM_PX: f32 = 16.0;
const UTILITY_PANEL_HEIGHT: f32 = 39.0;
const UTILITY_BUTTON_SIZE: f32 = 30.0;
const UTILITY_ICON_SIZE: u32 = 25;

/// (mode, clockwise-angle-from-12-o'clock). Hex layout at 60° intervals.
const MODE_LAYOUT: [(NavigationMode, f32); 6] = [
    (NavigationMode::Normal, 0.0),
    (NavigationMode::Prograde, 60.0),
    (NavigationMode::RadialOut, 120.0),
    (NavigationMode::AntiNormal, 180.0),
    (NavigationMode::Retrograde, 240.0),
    (NavigationMode::RadialIn, 300.0),
];

const UTILITY_LAYOUT: [NavigationMode; 3] = [
    NavigationMode::ManeuverNode,
    NavigationMode::Target,
    NavigationMode::AntiTarget,
];

/// Each axis: rotation from horizontal in clockwise degrees, and the
/// `MarkerKind` whose colour the axis is tinted with.
const AXES: [(f32, MarkerKind); 3] = [
    (90.0, MarkerKind::Normal),    // vertical (12 ↔ 6)
    (-30.0, MarkerKind::Prograde), // "/" — 2 ↔ 8
    (30.0, MarkerKind::RadialOut), // "\" — 4 ↔ 10
];

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct NavModeButton {
    pub mode: NavigationMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NavAssistKind {
    Rcs,
    Sas,
}

impl NavAssistKind {
    fn label(self) -> &'static str {
        match self {
            Self::Rcs => "RCS",
            Self::Sas => "SAS",
        }
    }

    fn available(self) -> bool {
        match self {
            Self::Rcs => false,
            Self::Sas => true,
        }
    }
}

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct NavAssistButton {
    kind: NavAssistKind,
}

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct AutopilotToggleButton;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct AutopilotToggleText;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct LandToggleButton;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct LandToggleText;

/// Wrapper for the LAND reason line, so the whole row can be hidden when there
/// is nothing to say rather than leaving an empty gap in the panel.
#[derive(Component, Debug, Clone, Copy)]
pub(super) struct LandNoticeRoot;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct LandNoticeText;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct ManeuverPanelRoot;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct ManeuverBurnText;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct ManeuverDeltaVText;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct ManeuverStartText;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct ManeuverProgressFill;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct ManeuverWarpButton;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct ManeuverWarpButtonText;

#[derive(Component, Debug, Clone, Copy)]
pub(super) struct ManeuverDismissButton;

#[derive(Resource, Debug, Default)]
pub(super) struct ManeuverPanelState {
    sticky: bool,
    dismissed: bool,
    saw_node: bool,
    executed: bool,
    last_node_count: usize,
}

#[derive(Component, Debug, Clone)]
pub(super) struct NavButtonIcon {
    mode: NavigationMode,
    enabled: Handle<Image>,
    disabled: Handle<Image>,
}

pub fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    theme: Res<HudTheme>,
    center_image: Res<NavAttitudeRenderTarget>,
    flight_stack: Res<BottomLeftFlightStackAnchor>,
) {
    let icons: Vec<(NavigationMode, f32, NavButtonIcon)> = MODE_LAYOUT
        .iter()
        .map(|&(mode, angle_deg)| {
            let kind = marker_kind_for(mode);
            let icon = nav_button_icon(mode, kind, BUTTON_SIZE as u32, &mut images);
            (mode, angle_deg, icon)
        })
        .collect();
    let utility_icons: Vec<(NavigationMode, NavButtonIcon)> = UTILITY_LAYOUT
        .iter()
        .map(|&mode| {
            let icon = nav_button_icon(mode, marker_kind_for(mode), UTILITY_ICON_SIZE, &mut images);
            (mode, icon)
        })
        .collect();

    let root = Node {
        position_type: PositionType::Absolute,
        left: Val::Px(PANEL_LEFT_PX),
        bottom: Val::Px(PANEL_BOTTOM_PX),
        width: Val::Px(PANEL_DIAMETER),
        height: Val::Px(PANEL_DIAMETER),
        border: UiRect::all(Val::Px(1.0)),
        border_radius: BorderRadius::all(Val::Percent(50.0)),
        ..default()
    };

    commands
        .spawn((
            root,
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            HudPanel,
            Name::new("HudNavPanel"),
        ))
        .with_children(|p| {
            // Axis lines first — they sit underneath buttons + centre.
            for (angle_deg, kind) in AXES {
                spawn_axis_line(p, angle_deg, marker_color(kind));
            }

            // 3D attitude indicator render target.
            p.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(PANEL_DIAMETER * 0.5 - CENTER_SIZE * 0.5),
                    top: Val::Px(PANEL_DIAMETER * 0.5 - CENTER_SIZE * 0.5),
                    width: Val::Px(CENTER_SIZE),
                    height: Val::Px(CENTER_SIZE),
                    ..default()
                },
                ImageNode::new(center_image.image.clone()),
                ZIndex(1),
                Name::new("NavAttitudeIndicator"),
            ));

            // Mode buttons.
            for (mode, angle_deg, icon) in icons {
                let theta = angle_deg.to_radians();
                let cx = PANEL_DIAMETER * 0.5 + BUTTON_RING_RADIUS * theta.sin();
                let cy = PANEL_DIAMETER * 0.5 - BUTTON_RING_RADIUS * theta.cos();
                p.spawn(mode_button_bundle(&theme, mode, icon, cx, cy));
            }
        });

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(PANEL_LEFT_PX),
                bottom: Val::Px(ASSIST_PANEL_BOTTOM_PX),
                width: Val::Px(PANEL_DIAMETER),
                height: Val::Px(ASSIST_PANEL_HEIGHT),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(3.0)),
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(8.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            HudPanel,
            Name::new("HudNavAssistPanel"),
        ))
        .with_children(|p| {
            spawn_assist_button(p, &theme, NavAssistKind::Rcs);
            spawn_assist_button(p, &theme, NavAssistKind::Sas);
        });

    commands.entity(flight_stack.0).with_children(|stack| {
        spawn_autopilot_panel(stack, &theme);
        spawn_maneuver_panel(stack, &theme);
    });

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(PANEL_LEFT_PX),
                bottom: Val::Px(UTILITY_PANEL_BOTTOM_PX),
                width: Val::Px(PANEL_DIAMETER),
                height: Val::Px(UTILITY_PANEL_HEIGHT),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                padding: UiRect::axes(Val::Px(9.0), Val::Px(4.0)),
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(10.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            HudPanel,
            Name::new("HudNavUtilityPanel"),
        ))
        .with_children(|p| {
            for (mode, icon) in utility_icons {
                p.spawn(utility_button_bundle(&theme, mode, icon));
            }
        });
}

fn spawn_axis_line(p: &mut ChildSpawnerCommands<'_>, angle_deg: f32, color: Color) {
    let left = (PANEL_DIAMETER - AXIS_LENGTH) * 0.5;
    let top = PANEL_DIAMETER * 0.5 - AXIS_THICKNESS * 0.5;
    p.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(left),
            top: Val::Px(top),
            width: Val::Px(AXIS_LENGTH),
            height: Val::Px(AXIS_THICKNESS),
            ..default()
        },
        BackgroundColor(color),
        UiTransform {
            translation: Val2::ZERO,
            scale: Vec2::ONE,
            rotation: Rot2::degrees(angle_deg),
        },
    ));
}

fn mode_button_bundle(
    theme: &HudTheme,
    mode: NavigationMode,
    icon: NavButtonIcon,
    cx: f32,
    cy: f32,
) -> impl Bundle {
    let image = icon.enabled.clone();
    (
        Button,
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(cx - BUTTON_SIZE * 0.5),
            top: Val::Px(cy - BUTTON_SIZE * 0.5),
            width: Val::Px(BUTTON_SIZE),
            height: Val::Px(BUTTON_SIZE),
            border: UiRect::all(Val::Px(1.5)),
            border_radius: BorderRadius::all(Val::Percent(50.0)),
            padding: UiRect::all(Val::Px(4.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(theme.panel_bg),
        BorderColor::all(theme.panel_border),
        NavModeButton { mode },
        Interaction::None,
        ZIndex(2),
        Name::new(format!("NavModeButton_{:?}", mode)),
        children![(
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
            ImageNode::new(image),
            icon,
        )],
    )
}

fn utility_button_bundle(
    theme: &HudTheme,
    mode: NavigationMode,
    icon: NavButtonIcon,
) -> impl Bundle {
    let image = icon.enabled.clone();
    (
        Button,
        Node {
            width: Val::Px(UTILITY_BUTTON_SIZE),
            height: Val::Px(UTILITY_BUTTON_SIZE),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(4.0)),
            padding: UiRect::all(Val::Px(3.0)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(theme.panel_bg_alt),
        BorderColor::all(theme.panel_border),
        NavModeButton { mode },
        Interaction::None,
        Name::new(format!("NavUtilityButton_{:?}", mode)),
        children![(
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
            ImageNode::new(image),
            icon,
        )],
    )
}

fn spawn_autopilot_panel(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    parent
        .spawn((
            Node {
                position_type: PositionType::Relative,
                width: Val::Px(PANEL_DIAMETER),
                min_height: Val::Px(AUTOPILOT_PANEL_MIN_HEIGHT),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(5.0),
                align_items: AlignItems::FlexStart,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            HudPanel,
            Name::new("HudAutopilotPanel"),
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("AUTOPILOT"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(11.0),
                    ..default()
                },
                TextColor(theme.text_subtitle),
            ));
            p.spawn((
                Button,
                Node {
                    width: Val::Px(AUTOPILOT_BUTTON_WIDTH),
                    height: Val::Px(AUTOPILOT_BUTTON_HEIGHT),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(theme.panel_bg_alt),
                BorderColor::all(theme.panel_border),
                Interaction::None,
                AutopilotToggleButton,
                Name::new("AutopilotToggleButton"),
            ))
            .with_children(|c| {
                c.spawn((
                    Text::new("MNVR"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(12.0),
                        ..default()
                    },
                    TextColor(theme.text_primary),
                    AutopilotToggleText,
                ));
            });
            p.spawn((
                Button,
                Node {
                    width: Val::Px(AUTOPILOT_BUTTON_WIDTH),
                    height: Val::Px(AUTOPILOT_BUTTON_HEIGHT),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(theme.panel_bg_alt),
                BorderColor::all(theme.panel_border),
                Interaction::None,
                LandToggleButton,
                Name::new("LandToggleButton"),
            ))
            .with_children(|c| {
                c.spawn((
                    Text::new("LAND"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(12.0),
                        ..default()
                    },
                    TextColor(theme.text_primary),
                    LandToggleText,
                ));
            });
            p.spawn((
                Node {
                    width: Val::Percent(100.0),
                    ..default()
                },
                Visibility::Hidden,
                LandNoticeRoot,
                Name::new("LandNoticeRow"),
            ))
            .with_children(|c| {
                c.spawn((
                    Text::new(""),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(10.0),
                        ..default()
                    },
                    TextColor(theme.text_dim),
                    LandNoticeText,
                ));
            });
        });
}

fn spawn_maneuver_panel(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    parent
        .spawn((
            Node {
                position_type: PositionType::Relative,
                display: Display::None,
                width: Val::Px(PANEL_DIAMETER),
                height: Val::Px(MANEUVER_PANEL_HEIGHT),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            HudPanel,
            ManeuverPanelRoot,
            Name::new("HudManeuverPanel"),
        ))
        .with_children(|p| {
            p.spawn(Node {
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                column_gap: Val::Px(6.0),
                ..default()
            })
            .with_children(|row| {
                row.spawn((
                    Text::new("MANEUVER"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(11.0),
                        ..default()
                    },
                    TextColor(theme.text_subtitle),
                ));
                row.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(4.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|buttons| {
                    buttons
                        .spawn((
                            Button,
                            Node {
                                width: Val::Px(MANEUVER_WARP_BUTTON_WIDTH),
                                height: Val::Px(MANEUVER_WARP_BUTTON_HEIGHT),
                                border: UiRect::all(Val::Px(1.0)),
                                border_radius: BorderRadius::all(Val::Px(3.0)),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                ..default()
                            },
                            BackgroundColor(theme.panel_bg_alt),
                            BorderColor::all(theme.panel_border),
                            Interaction::None,
                            ManeuverWarpButton,
                            Name::new("ManeuverWarpButton"),
                        ))
                        .with_children(|c| {
                            c.spawn((
                                Text::new("WARP"),
                                TextFont {
                                    font: theme.font.clone(),
                                    font_size: FontSize::Px(10.0),
                                    ..default()
                                },
                                TextColor(theme.text_primary),
                                ManeuverWarpButtonText,
                            ));
                        });

                    buttons
                        .spawn((
                            Button,
                            Node {
                                width: Val::Px(MANEUVER_WARP_BUTTON_HEIGHT),
                                height: Val::Px(MANEUVER_WARP_BUTTON_HEIGHT),
                                border: UiRect::all(Val::Px(1.0)),
                                border_radius: BorderRadius::all(Val::Px(3.0)),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                ..default()
                            },
                            BackgroundColor(theme.panel_bg_alt),
                            BorderColor::all(theme.panel_border),
                            Interaction::None,
                            ManeuverDismissButton,
                            Name::new("ManeuverDismissButton"),
                        ))
                        .with_children(|c| {
                            c.spawn((
                                Text::new("×"),
                                TextFont {
                                    font: theme.font.clone(),
                                    font_size: FontSize::Px(12.0),
                                    ..default()
                                },
                                TextColor(theme.text_dim),
                            ));
                        });
                });
            });

            spawn_maneuver_readout(p, theme, "Burn", ManeuverBurnText);
            spawn_maneuver_readout(p, theme, "Δv", ManeuverDeltaVText);
            spawn_maneuver_readout(p, theme, "Start", ManeuverStartText);

            p.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(MANEUVER_BAR_HEIGHT),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(2.0)),
                    ..default()
                },
                BackgroundColor(theme.panel_bg_alt),
                BorderColor::all(theme.panel_border),
            ))
            .with_children(|bar| {
                bar.spawn((
                    Node {
                        width: Val::Percent(0.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    BackgroundColor(theme.text_warn),
                    ManeuverProgressFill,
                ));
            });
        });
}

fn spawn_maneuver_readout<M: Component + Copy>(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    label: &'static str,
    marker: M,
) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Baseline,
            column_gap: Val::Px(8.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(label),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(10.0),
                    ..default()
                },
                TextColor(theme.text_dim),
            ));
            row.spawn((
                Text::new("—"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(10.0),
                    ..default()
                },
                TextColor(theme.text_primary),
                marker,
            ));
        });
}

fn spawn_assist_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    kind: NavAssistKind,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(ASSIST_BUTTON_WIDTH),
                height: Val::Px(ASSIST_BUTTON_HEIGHT),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            NavAssistButton { kind },
            Name::new(format!("NavAssistButton_{}", kind.label())),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(kind.label()),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(if kind.available() {
                    theme.text_dim
                } else {
                    disabled_text_color()
                }),
            ));
        });
}

pub fn handle_clicks(
    interactions: Query<(&Interaction, &NavModeButton), Changed<Interaction>>,
    assist_interactions: Query<(&Interaction, &NavAssistButton), Changed<Interaction>>,
    autopilot_interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<AutopilotToggleButton>),
    >,
    land_interactions: Query<&Interaction, (Changed<Interaction>, With<LandToggleButton>)>,
    warp_interactions: Query<&Interaction, (Changed<Interaction>, With<ManeuverWarpButton>)>,
    dismiss_interactions: Query<&Interaction, (Changed<Interaction>, With<ManeuverDismissButton>)>,
    locks: Res<ControlLocks>,
    target: Res<TargetBody>,
    plan: thalos_game_state::ActiveCraftRef<ManeuverPlan>,
    sim: Res<SimulationState>,
    route: Res<RouteState>,
    mut nav: ResMut<NavigationState>,
    mut sas: ResMut<thalos_game_state::flight::SasState>,
    mut autoflight_requests: MessageWriter<AutoflightRequest>,
    mut warp_to: ResMut<WarpToManeuver>,
    mut maneuver_panel: ResMut<ManeuverPanelState>,
) {
    let Some(plan) = plan.get() else {
        return;
    };
    for (interaction, button) in &interactions {
        if matches!(interaction, Interaction::Pressed)
            && !locks.navigation_mode
            && mode_available(button.mode, &target, plan)
        {
            if nav.mode == Some(button.mode) {
                nav.mode = None;
            } else {
                nav.mode = Some(button.mode);
            }
        }
    }

    // The panel emits intent; it never mutates the executor or a
    // program. `thalos_runtime::autoflight::handle_autoflight_requests`
    // is the sole consumer and the sole place the program-conflict
    // interlock is applied. A panel that mutated directly is how `MNVR`
    // came to silently decapitate a running ascent.
    for interaction in &autopilot_interactions {
        if matches!(interaction, Interaction::Pressed) {
            autoflight_requests.write(AutoflightRequest::ToggleBurnArm);
        }
    }

    let land_available = route.destination_guidance.is_some() || route.guidance.is_some();
    for interaction in &land_interactions {
        if matches!(interaction, Interaction::Pressed) && land_available {
            autoflight_requests.write(AutoflightRequest::ToggleLanding);
        }
    }

    for interaction in &warp_interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        if warp_to.active {
            warp_to.cancel();
        } else if find_next_maneuver(sim.simulation.sim_time(), &sim.simulation).is_some() {
            warp_to.active = true;
        }
    }

    for interaction in &dismiss_interactions {
        if matches!(interaction, Interaction::Pressed) {
            maneuver_panel.dismissed = true;
            maneuver_panel.sticky = false;
        }
    }

    for (interaction, button) in &assist_interactions {
        if !matches!(interaction, Interaction::Pressed) || locks.navigation_mode {
            continue;
        }

        match button.kind {
            NavAssistKind::Rcs => {
                // Placeholder only: RCS authority is not wired into ControlInput yet.
            }
            NavAssistKind::Sas => {
                // The button drives the real SAS switch (`SasState`, same as
                // the `T` key). The legacy Stability nav mode is folded in:
                // if it is somehow set, turning SAS off clears it too, so the
                // button can never read off while a stability hold still flies.
                if sas.enabled || nav.mode == Some(NavigationMode::Stability) {
                    sas.enabled = false;
                    if nav.mode == Some(NavigationMode::Stability) {
                        nav.mode = None;
                    }
                } else {
                    sas.enabled = true;
                }
            }
        }
    }
}

pub fn update_button_visuals(
    mut nav: ResMut<NavigationState>,
    theme: Res<HudTheme>,
    locks: Res<ControlLocks>,
    target: Res<TargetBody>,
    plan: thalos_game_state::ActiveCraftRef<ManeuverPlan>,
    realized: thalos_game_state::ActiveCraftRef<thalos_game_state::flight::RealizedControl>,
    sas: Res<thalos_game_state::flight::SasState>,
    mut buttons: ParamSet<(
        Query<(
            &NavModeButton,
            &Interaction,
            &mut BorderColor,
            &mut BackgroundColor,
        )>,
        Query<(
            &NavAssistButton,
            &Interaction,
            &mut BorderColor,
            &mut BackgroundColor,
            &Children,
        )>,
    )>,
    mut icons: Query<(&NavButtonIcon, &mut ImageNode)>,
    mut text_q: Query<(&mut Text, &mut TextColor)>,
) {
    let (Some(plan), Some(realized)) = (plan.get(), realized.get()) else {
        return;
    };
    if let Some(mode) = nav.mode
        && !mode_available(mode, &target, plan)
    {
        nav.mode = None;
    }

    for (button, interaction, mut border, mut bg) in &mut buttons.p0() {
        let available = mode_available(button.mode, &target, plan);
        let active = nav.mode == Some(button.mode);
        let (border_color, bg_color) = nav_button_colors(
            &theme,
            active,
            available,
            locks.navigation_mode && !active,
            interaction,
        );
        apply_button_colors(&mut border, &mut bg, border_color, bg_color);
    }

    for (button, interaction, mut border, mut bg, children) in &mut buttons.p1() {
        let available = button.kind.available();
        let active = matches!(button.kind, NavAssistKind::Sas)
            && (sas.enabled || nav.mode == Some(NavigationMode::Stability));
        let (border_color, bg_color) = nav_button_colors(
            &theme,
            active,
            available,
            locks.navigation_mode && !active,
            interaction,
        );
        apply_button_colors(&mut border, &mut bg, border_color, bg_color);

        // The SAS button doubles as the flight-assist annunciator: it reads
        // FBW while the plane fly-by-wire law is flying, and goes warn-tinted
        // while stall protection is actively clamping the pitch command.
        let (label, label_color) = if !available {
            (button.kind.label(), disabled_text_color())
        } else if matches!(button.kind, NavAssistKind::Sas) && realized.assist.protection_active {
            ("FBW", theme.text_warn)
        } else if matches!(button.kind, NavAssistKind::Sas) && realized.assist.fbw_active {
            (
                "FBW",
                if active {
                    theme.text_accent
                } else {
                    theme.text_primary
                },
            )
        } else if active {
            (button.kind.label(), theme.text_accent)
        } else if locks.navigation_mode {
            (button.kind.label(), theme.text_dim)
        } else {
            (button.kind.label(), theme.text_primary)
        };
        if let Some(&child) = children.first()
            && let Ok((mut text, mut tc)) = text_q.get_mut(child)
        {
            if text.0 != label {
                text.0 = label.to_string();
            }
            if tc.0 != label_color {
                tc.0 = label_color;
            }
        }
    }

    for (icon, mut image) in &mut icons {
        let target_handle = if mode_available(icon.mode, &target, plan) {
            &icon.enabled
        } else {
            &icon.disabled
        };
        if image.image != *target_handle {
            image.image = target_handle.clone();
        }
    }
}

pub fn update_autopilot_visuals(
    autopilot: Res<Autopilot>,
    annunciation: Res<AutoflightAnnunciation>,
    policy: Res<AutoflightPolicy>,
    land: Res<LandAutopilot>,
    route: Res<RouteState>,
    theme: Res<HudTheme>,
    mut buttons: Query<
        (&Interaction, &mut BorderColor, &mut BackgroundColor),
        With<AutopilotToggleButton>,
    >,
    mut toggle_text: Query<(&mut Text, &mut TextColor), With<AutopilotToggleText>>,
    mut land_buttons: Query<
        (&Interaction, &mut BorderColor, &mut BackgroundColor),
        (With<LandToggleButton>, Without<AutopilotToggleButton>),
    >,
    mut land_text: Query<
        (&mut Text, &mut TextColor),
        (With<LandToggleText>, Without<AutopilotToggleText>),
    >,
    mut land_notice_text: Query<
        (&mut Text, &mut TextColor),
        (
            With<LandNoticeText>,
            Without<LandToggleText>,
            Without<AutopilotToggleText>,
        ),
    >,
    mut land_notice_root: Query<&mut Visibility, With<LandNoticeRoot>>,
) {
    // Engagement comes from the annunciator — the arbitration outcome —
    // so this chip can no longer contradict what is actually flying the
    // ship. `MNVR` lit now means "the burn executor is armed and no
    // program has taken it", not "someone pressed this button".
    let maneuver_active = autopilot.arm().armed() && annunciation.program == FlightProgram::None;
    for (interaction, mut border, mut bg) in &mut buttons {
        let (border_color, bg_color) =
            nav_button_colors(&theme, maneuver_active, true, false, interaction);
        apply_button_colors(&mut border, &mut bg, border_color, bg_color);
    }

    // Under an engaged program the chip annunciates *that program's*
    // ownership of the channel rather than pretending to be selectable,
    // and a pending confirm-disconnect says so in the label.
    // Program + owning channel in one chip: "ASCENT GUID", "MNVR NODE",
    // "LAND BURN". The channel comes from the arbitration outcome, so
    // the chip states who is flying rather than who was last clicked.
    let channel_suffix = match (annunciation.attitude, annunciation.throttle) {
        (AttitudeChannel::Guidance, _) => " GUID",
        (AttitudeChannel::NodeBurn, ThrottleChannel::Burn) => " BURN",
        (AttitudeChannel::NodeBurn, _) => " NODE",
        (AttitudeChannel::Pilot, _) => " MAN",
        _ => "",
    };
    let toggle_label = if policy.pending_disconnect_s.is_some() {
        "CONFIRM".to_string()
    } else {
        match annunciation.program {
            FlightProgram::None => format!("MNVR{}", channel_suffix),
            // The channel suffix is dropped for a program whose own chip
            // already annunciates a phase: `AUTOLAND GUID` does not fit the
            // button, and wrapping it pushed the phase chip out of the panel.
            // The phase says more than the channel does anyway.
            FlightProgram::Landing => FlightProgram::Landing.label().to_string(),
            program => format!("{}{}", program.label(), channel_suffix),
        }
    };
    let toggle_color = if policy.pending_disconnect_s.is_some() {
        theme.text_warn
    } else if maneuver_active || annunciation.program != FlightProgram::None {
        theme.text_accent
    } else {
        theme.text_dim
    };
    for (mut text, mut color) in &mut toggle_text {
        if text.0 != toggle_label {
            text.0.clone_from(&toggle_label);
        }
        if color.0 != toggle_color {
            color.0 = toggle_color;
        }
    }

    let land_active = annunciation.program == FlightProgram::Landing;
    let land_available = route.destination_guidance.is_some() || route.guidance.is_some();
    for (interaction, mut border, mut bg) in &mut land_buttons {
        let (border_color, bg_color) =
            nav_button_colors(&theme, land_active, land_available, false, interaction);
        apply_button_colors(&mut border, &mut bg, border_color, bg_color);
    }
    let land_color = if land_active {
        theme.text_accent
    } else if land_available {
        theme.text_dim
    } else {
        disabled_text_color()
    };
    // Whole words, and only the phase. `LAND CAP` / `LAND FNL` / `LAND G/A`
    // required a decoder ring, and paired with the `LAND GUID` chip above it
    // there was no way to tell which half of the readout had changed.
    let land_label = land.phase().label();
    for (mut text, mut color) in &mut land_text {
        if text.0 != land_label {
            text.0 = land_label.to_string();
        }
        if color.0 != land_color {
            color.0 = land_color;
        }
    }

    // The reason line. An autopilot that changes its mind must say why in the
    // same glance, or the change reads as a malfunction — which is exactly how
    // a correct go-around was received.
    let notice = land.notice;
    for (mut text, mut color) in &mut land_notice_text {
        let line = match notice {
            Some(n) => format!("{}: {}", n.label(), n.detail()),
            None => land.phase().describe().to_string(),
        };
        let tint = match notice {
            Some(n) if n.is_failure() => theme.text_warn,
            Some(_) => theme.text_accent,
            None => theme.text_dim,
        };
        if text.0 != line {
            text.0 = line;
        }
        if color.0 != tint {
            color.0 = tint;
        }
    }
    for mut visibility in &mut land_notice_root {
        let target = if notice.is_some() || land.active() {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != target {
            *visibility = target;
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn update_maneuver_visuals(
    autopilot: Res<Autopilot>,
    schedule: Res<AutopilotBurnSchedule>,
    plan: thalos_game_state::ActiveCraftRef<ManeuverPlan>,
    sim: Res<SimulationState>,
    warp_to: Res<WarpToManeuver>,
    theme: Res<HudTheme>,
    units: Res<thalos_game_state::units::UnitsSettings>,
    mut panel_state: ResMut<ManeuverPanelState>,
    mut panel_roots: Query<&mut Node, (With<ManeuverPanelRoot>, Without<ManeuverProgressFill>)>,
    mut burn_text: Query<
        (&mut Text, &mut TextColor),
        (
            With<ManeuverBurnText>,
            Without<ManeuverDeltaVText>,
            Without<ManeuverStartText>,
            Without<ManeuverWarpButtonText>,
        ),
    >,
    mut dv_text: Query<
        (&mut Text, &mut TextColor),
        (
            With<ManeuverDeltaVText>,
            Without<ManeuverBurnText>,
            Without<ManeuverStartText>,
            Without<ManeuverWarpButtonText>,
        ),
    >,
    mut start_text: Query<
        (&mut Text, &mut TextColor),
        (
            With<ManeuverStartText>,
            Without<ManeuverBurnText>,
            Without<ManeuverDeltaVText>,
            Without<ManeuverWarpButtonText>,
        ),
    >,
    mut progress_fill: Query<&mut Node, (With<ManeuverProgressFill>, Without<ManeuverPanelRoot>)>,
    mut warp_buttons: Query<
        (&Interaction, &mut BorderColor, &mut BackgroundColor),
        With<ManeuverWarpButton>,
    >,
    mut warp_text: Query<
        (&mut Text, &mut TextColor),
        (
            With<ManeuverWarpButtonText>,
            Without<ManeuverBurnText>,
            Without<ManeuverDeltaVText>,
            Without<ManeuverStartText>,
        ),
    >,
) {
    let Some(plan) = plan.get() else {
        return;
    };
    let directive = schedule.next();
    // The burn HUD tracks the upcoming/active burn only. Spent (`Executed`)
    // nodes linger in the plan for review but must not keep this panel pinned
    // — counting just the directive-driving nodes makes a completed burn fall
    // through the existing "executed, now sticky-fade" path unchanged.
    let active_node_count = plan.nodes.iter().filter(|n| n.drives_directive()).count();
    let has_node = active_node_count > 0;
    let autopilot_executing = matches!(
        autopilot.state(),
        AutopilotState::Engaging { .. } | AutopilotState::Burn { .. }
    );

    let node_count = active_node_count;
    if has_node {
        if node_count != panel_state.last_node_count {
            panel_state.dismissed = false;
            panel_state.executed = false;
        }
        panel_state.sticky = true;
        panel_state.saw_node = true;
        panel_state.last_node_count = node_count;
    } else {
        panel_state.last_node_count = 0;
    }
    if autopilot_executing {
        panel_state.sticky = true;
        panel_state.executed = true;
    }
    if panel_state.saw_node && !has_node && directive.is_none() && !panel_state.executed {
        // The authored node disappeared before execution; treat that as an
        // explicit deletion rather than a completed maneuver to keep around.
        panel_state.sticky = false;
        panel_state.dismissed = true;
        panel_state.saw_node = false;
    }

    let panel_visible =
        (has_node || directive.is_some() || panel_state.sticky) && !panel_state.dismissed;
    for mut node in &mut panel_roots {
        node.display = if panel_visible {
            Display::Flex
        } else {
            Display::None
        };
    }

    let now = sim.simulation.sim_time();
    let mut progress = 0.0;

    let (burn, burn_color, dv, dv_color, start, start_color) = if let Some(directive) = directive {
        let burn_start = directive.center_time - directive.duration_s / 2.0;
        let mut delivered = 0.0;
        if let AutopilotState::Burn {
            directive_id,
            anchor_delivered_dv,
            ..
        } = autopilot.state()
            && directive_id == directive.id
        {
            delivered = (sim.simulation.delivered_dv() - anchor_delivered_dv).max(0.0);
        }
        progress = if directive.delta_v_magnitude > 0.0 {
            (delivered / directive.delta_v_magnitude).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let remaining_dv = (directive.delta_v_magnitude - delivered).max(0.0);
        let start_label = if matches!(autopilot.state(), AutopilotState::Burn { directive_id, .. } if directive_id == directive.id)
        {
            "BURNING".to_string()
        } else {
            format!("in {}", format_mission_time(burn_start - now))
        };
        (
            format_duration_compact(directive.duration_s),
            theme.text_primary,
            crate::format::delta_v(remaining_dv, units.system_for(UnitDomain::General)),
            theme.text_primary,
            start_label,
            if burn_start - now <= 0.0 {
                theme.text_warn
            } else {
                theme.text_primary
            },
        )
    } else {
        (
            "—".to_string(),
            theme.text_dim,
            "—".to_string(),
            theme.text_dim,
            "no node".to_string(),
            theme.text_dim,
        )
    };

    for (mut text, mut text_color) in &mut burn_text {
        set_text(&mut text, &mut text_color, &burn, burn_color);
    }
    for (mut text, mut text_color) in &mut dv_text {
        set_text(&mut text, &mut text_color, &dv, dv_color);
    }
    for (mut text, mut text_color) in &mut start_text {
        set_text(&mut text, &mut text_color, &start, start_color);
    }

    for mut node in &mut progress_fill {
        let target = Val::Percent((progress * 100.0) as f32);
        if node.width != target {
            node.width = target;
        }
    }

    let warp_available = find_next_maneuver(now, &sim.simulation).is_some();
    for (interaction, mut border, mut bg) in &mut warp_buttons {
        let (border_color, bg_color) = nav_button_colors(
            &theme,
            warp_to.active,
            warp_available || warp_to.active,
            false,
            interaction,
        );
        apply_button_colors(&mut border, &mut bg, border_color, bg_color);
    }

    let warp_label = if warp_to.active { "STOP" } else { "WARP" };
    let warp_color = if !warp_available && !warp_to.active {
        disabled_text_color()
    } else if warp_to.active {
        theme.text_accent
    } else {
        theme.text_primary
    };
    for (mut text, mut text_color) in &mut warp_text {
        set_text(&mut text, &mut text_color, warp_label, warp_color);
    }
}

fn set_text(text: &mut Text, text_color: &mut TextColor, value: &str, color: Color) {
    if text.0 != value {
        text.0 = value.to_string();
    }
    if text_color.0 != color {
        text_color.0 = color;
    }
}

fn format_mission_time(seconds_until_event: f64) -> String {
    let rounded = seconds_until_event.round();
    let marker = if rounded >= 0.0 { '-' } else { '+' };
    format!("T{}{:.0}s", marker, rounded.abs())
}

fn format_duration_compact(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else if seconds < 3600.0 {
        let minutes = (seconds / 60.0).floor();
        let secs = seconds - minutes * 60.0;
        format!("{:.0}m {:02.0}s", minutes, secs)
    } else {
        let hours = (seconds / 3600.0).floor();
        let minutes = ((seconds - hours * 3600.0) / 60.0).floor();
        format!("{:.0}h {:02.0}m", hours, minutes)
    }
}

/// Shared HUD toggle-button styling (border, background) for the
/// active/hover/pressed states — also used by the flight-config pills so
/// every clickable HUD toggle reads the same.
pub(super) fn nav_button_colors(
    theme: &HudTheme,
    active: bool,
    available: bool,
    locked: bool,
    interaction: &Interaction,
) -> (Color, Color) {
    if !available {
        (
            Color::srgba(0.15, 0.14, 0.12, 0.70),
            Color::srgba(0.045, 0.042, 0.036, 0.72),
        )
    } else if locked {
        (theme.text_dim, theme.panel_bg)
    } else {
        match (active, interaction) {
            (true, _) => (theme.text_accent, theme.panel_bg),
            (false, Interaction::Pressed) => (theme.text_primary, theme.panel_border),
            (false, Interaction::Hovered) => (theme.text_primary, theme.panel_bg),
            (false, Interaction::None) => (theme.panel_border, theme.panel_bg),
        }
    }
}

pub(super) fn apply_button_colors(
    border: &mut BorderColor,
    bg: &mut BackgroundColor,
    border_color: Color,
    bg_color: Color,
) {
    let new_border = BorderColor::all(border_color);
    if border.top != new_border.top {
        *border = new_border;
    }
    if bg.0 != bg_color {
        bg.0 = bg_color;
    }
}

fn disabled_text_color() -> Color {
    Color::srgba(0.39, 0.37, 0.32, 1.0)
}

fn marker_kind_for(mode: NavigationMode) -> MarkerKind {
    match mode {
        NavigationMode::Prograde => MarkerKind::Prograde,
        NavigationMode::Retrograde => MarkerKind::Retrograde,
        NavigationMode::Normal => MarkerKind::Normal,
        NavigationMode::AntiNormal => MarkerKind::AntiNormal,
        NavigationMode::RadialOut => MarkerKind::RadialOut,
        NavigationMode::RadialIn => MarkerKind::RadialIn,
        NavigationMode::Target => MarkerKind::Target,
        NavigationMode::AntiTarget => MarkerKind::AntiTarget,
        NavigationMode::ManeuverNode => MarkerKind::ManeuverNode,
        NavigationMode::Stability => MarkerKind::Prograde,
    }
}

fn marker_color(kind: MarkerKind) -> Color {
    let [r, g, b] = kind.color();
    Color::srgba(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}

fn nav_button_icon(
    mode: NavigationMode,
    kind: MarkerKind,
    size: u32,
    images: &mut Assets<Image>,
) -> NavButtonIcon {
    NavButtonIcon {
        mode,
        enabled: images.add(marker_icon_image(kind, size, MarkerIconState::Visible)),
        disabled: images.add(marker_icon_image(kind, size, MarkerIconState::Disabled)),
    }
}

fn mode_available(mode: NavigationMode, target: &TargetBody, plan: &ManeuverPlan) -> bool {
    match mode {
        NavigationMode::Target | NavigationMode::AntiTarget => target.target.is_some(),
        NavigationMode::ManeuverNode => plan.nodes.iter().any(|n| n.drives_directive()),
        _ => true,
    }
}
