//! Circular navigation panel: 6 mode buttons + 3 colour-coded axis
//! lines crossing at the centre, with a 3D attitude indicator
//! ([`nav_attitude`]) in the middle.
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

use crate::hud::HudPanel;
use crate::hud::nav_attitude::NavAttitudeRenderTarget;
use crate::hud::theme::HudTheme;
use crate::navball::markers::{MarkerKind, generate_marker_icon, image_from_rgba8};
use crate::navigation::{NavigationMode, NavigationState};

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

/// (mode, clockwise-angle-from-12-o'clock). Hex layout at 60° intervals.
const MODE_LAYOUT: [(NavigationMode, f32); 6] = [
    (NavigationMode::Normal, 0.0),
    (NavigationMode::Prograde, 60.0),
    (NavigationMode::RadialOut, 120.0),
    (NavigationMode::AntiNormal, 180.0),
    (NavigationMode::Retrograde, 240.0),
    (NavigationMode::RadialIn, 300.0),
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

pub fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    theme: Res<HudTheme>,
    center_image: Res<NavAttitudeRenderTarget>,
) {
    let icons: Vec<(NavigationMode, f32, Handle<Image>)> = MODE_LAYOUT
        .iter()
        .map(|&(mode, angle_deg)| {
            let kind = marker_kind_for(mode);
            let handle = images.add(image_from_rgba8(
                BUTTON_SIZE as u32,
                generate_marker_icon(kind, BUTTON_SIZE as u32, false),
            ));
            (mode, angle_deg, handle)
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
            for (mode, angle_deg, handle) in icons {
                let theta = angle_deg.to_radians();
                let cx = PANEL_DIAMETER * 0.5 + BUTTON_RING_RADIUS * theta.sin();
                let cy = PANEL_DIAMETER * 0.5 - BUTTON_RING_RADIUS * theta.cos();
                p.spawn(mode_button_bundle(&theme, mode, handle, cx, cy));
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
    icon: Handle<Image>,
    cx: f32,
    cy: f32,
) -> impl Bundle {
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
            ImageNode::new(icon),
        )],
    )
}

pub fn handle_clicks(
    interactions: Query<(&Interaction, &NavModeButton), Changed<Interaction>>,
    mut nav: ResMut<NavigationState>,
) {
    for (interaction, button) in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            if nav.mode == Some(button.mode) {
                nav.mode = None;
            } else {
                nav.mode = Some(button.mode);
            }
        }
    }
}

pub fn update_button_visuals(
    nav: Res<NavigationState>,
    theme: Res<HudTheme>,
    mut buttons: Query<(
        &NavModeButton,
        &Interaction,
        &mut BorderColor,
        &mut BackgroundColor,
    )>,
) {
    for (button, interaction, mut border, mut bg) in &mut buttons {
        let active = nav.mode == Some(button.mode);
        let (border_color, bg_color) = match (active, interaction) {
            (true, _) => (theme.text_accent, theme.panel_bg),
            (false, Interaction::Pressed) => (theme.text_primary, theme.panel_border),
            (false, Interaction::Hovered) => (theme.text_primary, theme.panel_bg),
            (false, Interaction::None) => (theme.panel_border, theme.panel_bg),
        };
        let new_border = BorderColor::all(border_color);
        if border.top != new_border.top {
            *border = new_border;
        }
        if bg.0 != bg_color {
            bg.0 = bg_color;
        }
    }
}

fn marker_kind_for(mode: NavigationMode) -> MarkerKind {
    match mode {
        NavigationMode::Prograde => MarkerKind::Prograde,
        NavigationMode::Retrograde => MarkerKind::Retrograde,
        NavigationMode::Normal => MarkerKind::Normal,
        NavigationMode::AntiNormal => MarkerKind::AntiNormal,
        NavigationMode::RadialOut => MarkerKind::RadialOut,
        NavigationMode::RadialIn => MarkerKind::RadialIn,
        NavigationMode::Stability
        | NavigationMode::Target
        | NavigationMode::AntiTarget
        | NavigationMode::ManeuverNode => MarkerKind::Prograde,
    }
}

fn marker_color(kind: MarkerKind) -> Color {
    let [r, g, b] = kind.color();
    Color::srgba(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}
