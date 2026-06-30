//! Top-right HUD overlay: smoothed frame-rate readout.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

use crate::hud::HudPanel;
use crate::hud::theme::{HudTheme, panel_frame, panel_node};

#[derive(Component)]
pub(super) struct FpsText;

pub(super) fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    let mut root = panel_node();
    root.right = Val::Px(20.0);
    root.top = Val::Px(20.0);
    root.padding = UiRect::axes(Val::Px(10.0), Val::Px(6.0));
    root.align_items = AlignItems::FlexEnd;

    let (bg, border) = panel_frame(&theme);

    commands
        .spawn((root, bg, border, HudPanel, Name::new("HudFpsOverlay")))
        .with_children(|p| {
            p.spawn((
                Text::new("FPS --"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(theme.text_dim),
                Node {
                    min_width: Val::Px(56.0),
                    ..default()
                },
                FpsText,
            ));
        });
}

pub(super) fn update(diagnostics: Res<DiagnosticsStore>, mut q: Query<&mut Text, With<FpsText>>) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.smoothed());

    let new_value = match fps {
        Some(fps) if fps.is_finite() => format!("FPS {:>3.0}", fps),
        _ => "FPS --".to_string(),
    };

    if let Ok(mut text) = q.single_mut()
        && text.0 != new_value
    {
        text.0 = new_value;
    }
}
