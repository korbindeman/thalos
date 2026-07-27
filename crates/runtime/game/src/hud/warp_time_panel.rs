//! Top-left HUD panel: pause/play, time readout, warp-level buttons.
//!
//! Layout (left to right inside the panel):
//!
//! - Square pause/play button — toggles `WarpController::toggle_pause`.
//! - Time column:
//!     - Mode pill (`UT` / `MET`) — toggles [`TimeDisplayMode`].
//!     - Time readout in `T+NNNy,NNNd,HH:MM:SS` form.
//!     - Row of warp-level buttons (one per non-pause entry in
//!       `WarpController::levels`) plus the active warp amount. Active
//!       level is highlighted.
//! - Subtitle line: `TIME`.
//!
//! Both `UT` and `MET` currently read the same `sim_time` — the
//! simulation has no separate mission-start epoch yet. The toggle is in
//! place so the UI is ready when mission elapsed time is decoupled from
//! the world clock; until then the values are intentionally identical.

use bevy::prelude::*;

use crate::hud::HudPanel;
use crate::hud::TopLeftRowAnchor;
use crate::hud::format;
use crate::hud::theme::{HudTheme, panel_frame, panel_node};
use crate::rendering::SimulationState;

#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeDisplayMode {
    #[default]
    Universal,
    Mission,
}

impl TimeDisplayMode {
    fn pill_label(self) -> &'static str {
        match self {
            Self::Universal => "UT",
            Self::Mission => "MET",
        }
    }

    fn toggled(self) -> Self {
        match self {
            Self::Universal => Self::Mission,
            Self::Mission => Self::Universal,
        }
    }
}

#[derive(Component)]
pub(super) struct TimeYearsSpan;

#[derive(Component)]
pub(super) struct TimeDaysSpan;

#[derive(Component)]
pub(super) struct TimeClockSpan;

#[derive(Component)]
pub(super) struct PauseButton;

/// Child of the pause button shown while the simulation is paused
/// (two vertical bars). Hidden while the sim is running.
#[derive(Component)]
pub(super) struct PauseGlyphPaused;

/// Child of the pause button shown while the simulation is running
/// (play triangle). Hidden while paused.
#[derive(Component)]
pub(super) struct PauseGlyphPlaying;

#[derive(Component)]
pub(super) struct TimeModeButton;

#[derive(Component)]
pub(super) struct TimeModeButtonLabel;

#[derive(Component, Clone, Copy)]
pub(super) struct WarpLevelButton {
    /// Index into `WarpController::levels`.
    index: usize,
}

#[derive(Component)]
pub(super) struct WarpAmountLabel;

const PAUSE_BUTTON_SIZE: f32 = 38.0;
const WARP_BUTTON_WIDTH: f32 = 23.0;
const WARP_BUTTON_HEIGHT: f32 = 18.0;
const WARP_BUTTON_GAP: f32 = 3.0;

pub fn setup(
    mut commands: Commands,
    theme: Res<HudTheme>,
    sim: Res<SimulationState>,
    anchor: Res<TopLeftRowAnchor>,
) {
    let mut root = panel_node();
    root.position_type = PositionType::Relative;
    root.padding = UiRect::axes(Val::Px(8.0), Val::Px(6.0));
    root.row_gap = Val::Px(4.0);

    let (bg, border) = panel_frame(&theme);

    let levels = sim.simulation.warp.levels().to_vec();

    commands.entity(anchor.0).with_children(|row_parent| {
        row_parent
            .spawn((root, bg, border, HudPanel, Name::new("HudWarpTime")))
            .with_children(|p| {
                p.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                })
                .with_children(|row| {
                    spawn_pause_button(row, &theme);
                    row.spawn(Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(4.0),
                        ..default()
                    })
                    .with_children(|col| {
                        col.spawn(Node {
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: Val::Px(6.0),
                            ..default()
                        })
                        .with_children(|time_row| {
                            spawn_time_mode_button(time_row, &theme);
                            spawn_time_readout(time_row, &theme);
                        });
                        col.spawn(Node {
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: Val::Px(WARP_BUTTON_GAP),
                            ..default()
                        })
                        .with_children(|btn_row| {
                            // Skip the pause level (index 0) — it's exposed via the
                            // dedicated pause button.
                            for (idx, _speed) in levels.iter().enumerate().skip(1) {
                                spawn_warp_button(btn_row, &theme, idx);
                            }
                            spawn_warp_amount_label(btn_row, &theme, &sim.simulation.warp.label());
                        });
                    });
                });
                p.spawn(subtitle(&theme, "TIME"));
            });
    });
}

fn spawn_pause_button(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(PAUSE_BUTTON_SIZE),
                height: Val::Px(PAUSE_BUTTON_SIZE),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            PauseButton,
            Name::new("PauseButton"),
        ))
        .with_children(|c| {
            // Pause icon: two vertical bars. Drawn from Nodes rather than a
            // glyph so it doesn't depend on the font having a clean ⏸/‖
            // character. Hidden while running.
            c.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(5.0),
                    align_items: AlignItems::Center,
                    ..default()
                },
                Visibility::Hidden,
                PauseGlyphPaused,
            ))
            .with_children(|bars| {
                for _ in 0..2 {
                    bars.spawn((
                        Node {
                            width: Val::Px(3.5),
                            height: Val::Px(16.0),
                            ..default()
                        },
                        BackgroundColor(theme.text_accent),
                    ));
                }
            });
            // Play icon: triangle. Visible while running.
            c.spawn((
                Text::new("▶"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(16.0),
                    ..default()
                },
                TextColor(theme.text_primary),
                PauseGlyphPlaying,
            ));
        });
}

/// Multi-span time readout: letters/separators in the subtitle colour,
/// numeric fragments in the primary colour. The three numeric spans
/// carry marker components so [`update`] can rewrite them in place.
fn spawn_time_readout(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    let letter = theme.text_subtitle;
    let number = theme.text_primary;
    parent
        .spawn((Text::new("T+"), text_font(theme, 16.0), TextColor(letter)))
        .with_children(|c| {
            c.spawn((
                TextSpan::new("0"),
                text_font(theme, 16.0),
                TextColor(number),
                TimeYearsSpan,
            ));
            c.spawn((
                TextSpan::new("y,"),
                text_font(theme, 16.0),
                TextColor(letter),
            ));
            c.spawn((
                TextSpan::new("0"),
                text_font(theme, 16.0),
                TextColor(number),
                TimeDaysSpan,
            ));
            c.spawn((
                TextSpan::new("d,"),
                text_font(theme, 16.0),
                TextColor(letter),
            ));
            c.spawn((
                TextSpan::new("00:00:00"),
                text_font(theme, 16.0),
                TextColor(number),
                TimeClockSpan,
            ));
        });
}

fn text_font(theme: &HudTheme, size: f32) -> TextFont {
    TextFont {
        font: theme.font.clone(),
        font_size: FontSize::Px(size),
        ..default()
    }
}

fn spawn_time_mode_button(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    parent
        .spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            TimeModeButton,
            Name::new("TimeModeButton"),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(TimeDisplayMode::default().pill_label()),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(theme.text_subtitle),
                TimeModeButtonLabel,
            ));
        });
}

fn spawn_warp_button(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, index: usize) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(WARP_BUTTON_WIDTH),
                height: Val::Px(WARP_BUTTON_HEIGHT),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(2.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            WarpLevelButton { index },
            Name::new(format!("WarpLevel{}", index)),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new("▶"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(10.0),
                    ..default()
                },
                TextColor(theme.text_dim),
            ));
        });
}

fn spawn_warp_amount_label(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, label: &str) {
    parent.spawn((
        Text::new(label.to_string()),
        TextFont {
            font: theme.font.clone(),
            font_size: FontSize::Px(12.0),
            ..default()
        },
        TextColor(theme.text_accent),
        WarpAmountLabel,
        Name::new("WarpAmountLabel"),
    ));
}

fn subtitle(theme: &HudTheme, content: impl Into<String>) -> impl Bundle {
    (
        Text::new(content),
        TextFont {
            font: theme.font.clone(),
            font_size: FontSize::Px(11.0),
            ..default()
        },
        TextColor(theme.text_subtitle),
    )
}

pub fn handle_pause_click(
    interactions: Query<&Interaction, (With<PauseButton>, Changed<Interaction>)>,
    mut sim: ResMut<SimulationState>,
) {
    for interaction in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            sim.simulation.warp.toggle_pause();
        }
    }
}

pub fn handle_time_mode_click(
    interactions: Query<&Interaction, (With<TimeModeButton>, Changed<Interaction>)>,
    mut mode: ResMut<TimeDisplayMode>,
) {
    for interaction in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            *mode = mode.toggled();
        }
    }
}

pub fn handle_warp_level_click(
    interactions: Query<(&Interaction, &WarpLevelButton), Changed<Interaction>>,
    limits: Res<crate::bridge::WarpLimits>,
    mut sim: ResMut<SimulationState>,
) {
    for (interaction, button) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        // Same gate as the keyboard handler: refuse clicks that would
        // escalate past the altitude cap. The enforcement system would
        // clamp it back next frame anyway, but refusing the input keeps
        // the button feedback honest.
        if button.index > limits.max_level {
            continue;
        }
        let Some(&target) = sim.simulation.warp.levels().get(button.index) else {
            continue;
        };
        sim.simulation.warp.set_speed(target);
    }
}

pub fn update(
    sim: Res<SimulationState>,
    mode: Res<TimeDisplayMode>,
    mut labels: ParamSet<(
        Query<&mut Text, With<TimeModeButtonLabel>>,
        Query<&mut Text, With<WarpAmountLabel>>,
    )>,
    mut spans: ParamSet<(
        Query<&mut TextSpan, With<TimeYearsSpan>>,
        Query<&mut TextSpan, With<TimeDaysSpan>>,
        Query<&mut TextSpan, With<TimeClockSpan>>,
    )>,
) {
    // Both UT and MET currently mirror sim_time; see module doc.
    let parts = format::warp_panel_time(sim.simulation.sim_time());
    if let Ok(mut s) = spans.p0().single_mut()
        && s.0 != parts.years
    {
        s.0 = parts.years;
    }
    if let Ok(mut s) = spans.p1().single_mut()
        && s.0 != parts.days
    {
        s.0 = parts.days;
    }
    if let Ok(mut s) = spans.p2().single_mut()
        && s.0 != parts.clock
    {
        s.0 = parts.clock;
    }

    let label = mode.pill_label();
    if let Ok(mut t) = labels.p0().single_mut()
        && t.0 != label
    {
        t.0 = label.to_string();
    }

    let label = sim.simulation.warp.label();
    if let Ok(mut t) = labels.p1().single_mut()
        && t.0 != label
    {
        t.0 = label;
    }
}

pub fn update_pause_glyph(
    sim: Res<SimulationState>,
    mut glyphs: ParamSet<(
        Query<&mut Visibility, With<PauseGlyphPaused>>,
        Query<&mut Visibility, With<PauseGlyphPlaying>>,
    )>,
) {
    let paused = sim.simulation.warp.speed() == 0.0;
    let (paused_target, playing_target) = if paused {
        (Visibility::Inherited, Visibility::Hidden)
    } else {
        (Visibility::Hidden, Visibility::Inherited)
    };
    if let Ok(mut v) = glyphs.p0().single_mut()
        && *v != paused_target
    {
        *v = paused_target;
    }
    if let Ok(mut v) = glyphs.p1().single_mut()
        && *v != playing_target
    {
        *v = playing_target;
    }
}

pub fn update_button_visuals(
    sim: Res<SimulationState>,
    theme: Res<HudTheme>,
    mut buttons: ParamSet<(
        Query<(&Interaction, &mut BorderColor, &mut BackgroundColor), With<PauseButton>>,
        Query<(&Interaction, &mut BorderColor, &mut BackgroundColor), With<TimeModeButton>>,
        Query<(
            &WarpLevelButton,
            &Interaction,
            &mut BorderColor,
            &mut BackgroundColor,
            &Children,
        )>,
    )>,
    mut text_q: Query<&mut TextColor>,
) {
    let paused = sim.simulation.warp.speed() == 0.0;
    let active_index = sim.simulation.warp.level_index();
    let latched_index = sim.simulation.warp.latched_level_index();

    for (interaction, mut border, mut bg) in &mut buttons.p0() {
        let (border_color, bg_color) = button_colors(&theme, paused, interaction);
        apply_button_colors(&mut border, &mut bg, border_color, bg_color);
    }

    // Mode pill never sticks to "active" — both UT and MET are valid states,
    // so neither is highlighted. Hover/press still react.
    for (interaction, mut border, mut bg) in &mut buttons.p1() {
        let (border_color, bg_color) = match interaction {
            Interaction::Pressed => (theme.text_primary, theme.panel_border),
            Interaction::Hovered => (theme.text_primary, theme.panel_bg),
            Interaction::None => (theme.panel_border, theme.panel_bg),
        };
        apply_button_colors(&mut border, &mut bg, border_color, bg_color);
    }

    for (button, interaction, mut border, mut bg, children) in &mut buttons.p2() {
        let active = button.index == active_index;
        let latched = latched_index == Some(button.index);
        let (border_color, bg_color) = warp_button_colors(&theme, active, latched, interaction);
        apply_button_colors(&mut border, &mut bg, border_color, bg_color);
        let glyph_color = if latched {
            theme.text_primary
        } else if active {
            theme.text_accent
        } else {
            theme.text_dim
        };
        if let Some(&child) = children.first()
            && let Ok(mut tc) = text_q.get_mut(child)
            && tc.0 != glyph_color
        {
            tc.0 = glyph_color;
        }
    }
}

fn warp_button_colors(
    theme: &HudTheme,
    active: bool,
    latched: bool,
    interaction: &Interaction,
) -> (Color, Color) {
    if latched {
        (theme.text_primary, theme.panel_bg)
    } else {
        button_colors(theme, active, interaction)
    }
}

fn button_colors(theme: &HudTheme, active: bool, interaction: &Interaction) -> (Color, Color) {
    match (active, interaction) {
        (true, _) => (theme.text_accent, theme.panel_bg),
        (false, Interaction::Pressed) => (theme.text_primary, theme.panel_border),
        (false, Interaction::Hovered) => (theme.text_primary, theme.panel_bg),
        (false, Interaction::None) => (theme.panel_border, theme.panel_bg),
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
