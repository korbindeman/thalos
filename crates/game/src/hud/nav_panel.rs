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

use crate::controls::ControlLocks;
use crate::hud::HudPanel;
use crate::hud::nav_attitude::NavAttitudeRenderTarget;
use crate::hud::theme::HudTheme;
use crate::maneuver::ManeuverPlan;
use crate::navball::markers::{MarkerIconState, MarkerKind, marker_icon_image};
use crate::navigation::{NavigationMode, NavigationState};
use crate::target::TargetBody;

/// Diameter of the circular panel (px).
const PANEL_DIAMETER: f32 = 190.0;
const PANEL_LEFT_PX: f32 = 40.0 + 256.0 + 14.0;
/// Vertically centred against the 256-px navball: (256-190)/2 = 33 px lift.
const PANEL_BOTTOM_PX: f32 = 40.0 + 33.0;

const BUTTON_SIZE: f32 = 36.0;
const BUTTON_RING_RADIUS: f32 = PANEL_DIAMETER * 0.5 - BUTTON_SIZE * 0.5 - 10.0;

const CENTER_SIZE: f32 = 64.0;

const AXIS_LENGTH: f32 = (BUTTON_RING_RADIUS - BUTTON_SIZE * 0.5) * 2.0;
const AXIS_THICKNESS: f32 = 1.5;

const ASSIST_PANEL_BOTTOM_PX: f32 = PANEL_BOTTOM_PX + PANEL_DIAMETER + 8.0;
const ASSIST_PANEL_HEIGHT: f32 = 38.0;
const ASSIST_BUTTON_WIDTH: f32 = 72.0;
const ASSIST_BUTTON_HEIGHT: f32 = 28.0;

const UTILITY_PANEL_BOTTOM_PX: f32 = 20.0;
const UTILITY_PANEL_HEIGHT: f32 = 44.0;
const UTILITY_BUTTON_SIZE: f32 = 34.0;
const UTILITY_ICON_SIZE: u32 = 28;

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
                padding: UiRect::axes(Val::Px(14.0), Val::Px(4.0)),
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(10.0),
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
                padding: UiRect::axes(Val::Px(12.0), Val::Px(5.0)),
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(12.0),
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
        BackgroundColor(Color::srgba(0.03, 0.045, 0.075, 0.82)),
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
                    font_size: 13.0,
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
    locks: Res<ControlLocks>,
    target: Res<TargetBody>,
    plan: Res<ManeuverPlan>,
    mut nav: ResMut<NavigationState>,
) {
    for (interaction, button) in &interactions {
        if matches!(interaction, Interaction::Pressed)
            && !locks.navigation_mode
            && mode_available(button.mode, &target, &plan)
        {
            if nav.mode == Some(button.mode) {
                nav.mode = None;
            } else {
                nav.mode = Some(button.mode);
            }
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
                if nav.mode == Some(NavigationMode::Stability) {
                    nav.mode = None;
                } else {
                    nav.mode = Some(NavigationMode::Stability);
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
    plan: Res<ManeuverPlan>,
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
    mut text_q: Query<&mut TextColor>,
) {
    if let Some(mode) = nav.mode
        && !mode_available(mode, &target, &plan)
    {
        nav.mode = None;
    }

    for (button, interaction, mut border, mut bg) in &mut buttons.p0() {
        let available = mode_available(button.mode, &target, &plan);
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
            && nav.mode == Some(NavigationMode::Stability);
        let (border_color, bg_color) = nav_button_colors(
            &theme,
            active,
            available,
            locks.navigation_mode && !active,
            interaction,
        );
        apply_button_colors(&mut border, &mut bg, border_color, bg_color);

        let label_color = if !available {
            disabled_text_color()
        } else if active {
            theme.text_accent
        } else if locks.navigation_mode {
            theme.text_dim
        } else {
            theme.text_primary
        };
        if let Some(&child) = children.first()
            && let Ok(mut tc) = text_q.get_mut(child)
            && tc.0 != label_color
        {
            tc.0 = label_color;
        }
    }

    for (icon, mut image) in &mut icons {
        let target_handle = if mode_available(icon.mode, &target, &plan) {
            &icon.enabled
        } else {
            &icon.disabled
        };
        if image.image != *target_handle {
            image.image = target_handle.clone();
        }
    }
}

fn nav_button_colors(
    theme: &HudTheme,
    active: bool,
    available: bool,
    locked: bool,
    interaction: &Interaction,
) -> (Color, Color) {
    if !available {
        (
            Color::srgba(0.16, 0.18, 0.22, 0.70),
            Color::srgba(0.025, 0.032, 0.05, 0.72),
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

fn apply_button_colors(
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
    Color::srgba(0.34, 0.38, 0.45, 1.0)
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
        NavigationMode::ManeuverNode => !plan.nodes.is_empty(),
        _ => true,
    }
}
