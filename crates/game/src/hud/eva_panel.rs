//! Centred HUD status pill for the on-foot (EVA) player controller.
//!
//! Sits just *below* the top-centre instrument bar (altitude / orbital
//! row) so the player-state banner reads as its own line and never
//! collides with the altitude readout.
//!
//! Surfaces the surface-movement state the warp gate keys off of
//! ([`crate::bridge::enforce_warp_altitude_limits`]): the player can only
//! engage time warp above 1× once standing still, KSP-style, so the HUD has
//! to make "you are stopped, warp is available" legible. Hidden whenever the
//! player controller is inactive (i.e. flying a ship).

use bevy::prelude::*;

use crate::hud::HudPanel;
use crate::hud::theme::{HudTheme, panel_frame, panel_node};
use crate::player_controller::{EvaMode, PlayerControllerState};

#[derive(Component)]
pub(super) struct EvaStatusRoot;

#[derive(Component)]
pub(super) struct EvaStatusText;

pub(super) fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    // Full-width centring row so the pill sits centred just below the
    // top-centre altitude / orbital bar (which ends ~103px down), out of
    // its way.
    let row = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(112.0),
                left: Val::Px(0.0),
                width: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                ..default()
            },
            Visibility::Hidden,
            EvaStatusRoot,
            HudPanel,
            Name::new("HudEvaStatus"),
        ))
        .id();

    let mut pill = panel_node();
    pill.padding = UiRect::axes(Val::Px(12.0), Val::Px(6.0));
    let (bg, border) = panel_frame(&theme);

    commands.entity(row).with_children(|p| {
        p.spawn((pill, bg, border)).with_children(|p| {
            p.spawn((
                Text::new("ON FOOT"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 14.0,
                    ..default()
                },
                TextColor(theme.text_dim),
                EvaStatusText,
            ));
        });
    });
}

pub(super) fn update(
    state: Res<PlayerControllerState>,
    eva_mode: Res<EvaMode>,
    theme: Res<HudTheme>,
    mut roots: Query<&mut Visibility, With<EvaStatusRoot>>,
    mut texts: Query<(&mut Text, &mut TextColor), With<EvaStatusText>>,
) {
    let Ok(mut visibility) = roots.single_mut() else {
        return;
    };

    if !state.is_active() {
        if *visibility != Visibility::Hidden {
            *visibility = Visibility::Hidden;
        }
        return;
    }
    if *visibility != Visibility::Inherited {
        *visibility = Visibility::Inherited;
    }

    let (label, color) = if !eva_mode.is_grounded() {
        ("ON FOOT · IN FLIGHT".to_string(), theme.text_accent)
    } else if !state.is_grounded() {
        ("ON FOOT · FALLING".to_string(), theme.text_warn)
    } else if state.is_at_rest() {
        (
            "ON FOOT · STANDING — warp ready".to_string(),
            Color::srgb(0.55, 0.9, 0.55),
        )
    } else {
        let speed = state.surface_speed_m_s();
        (
            format!("ON FOOT · MOVING {speed:.1} m/s"),
            theme.text_accent,
        )
    };

    if let Ok((mut text, mut text_color)) = texts.single_mut() {
        if text.0 != label {
            text.0 = label;
        }
        if text_color.0 != color {
            text_color.0 = color;
        }
    }
}
