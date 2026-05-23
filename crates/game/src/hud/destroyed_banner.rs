//! Centered overlay shown when the player craft has been destroyed by a
//! terrain impact. The explicit, hard-to-miss cue the landing spec calls
//! for (no explosion FX yet) — paired with the control lockout in
//! `apply_local_forces` / the bridge input gates. See `docs/landing.md`.

use bevy::prelude::*;

use crate::hud::HudPanel;
use crate::hud::theme::HudTheme;
use crate::rendering::SimulationState;

/// Root container; its `Visibility` is toggled by [`update`].
#[derive(Component)]
pub(super) struct DestroyedBannerRoot;

/// The sub-line text node carrying the impact speed + recovery hint.
#[derive(Component)]
pub(super) struct DestroyedBannerDetail;

pub(super) fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(90.0),
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                justify_content: JustifyContent::Center,
                ..default()
            },
            // Start hidden; `update` reveals it on destruction.
            Visibility::Hidden,
            DestroyedBannerRoot,
            HudPanel,
            Name::new("HudDestroyedBanner"),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    border: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    padding: UiRect::axes(Val::Px(22.0), Val::Px(12.0)),
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    row_gap: Val::Px(4.0),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.14, 0.02, 0.01, 0.92)),
                BorderColor::all(theme.text_warn),
            ))
            .with_children(|banner| {
                banner.spawn((
                    Text::new("VESSEL DESTROYED"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: 26.0,
                        ..default()
                    },
                    TextColor(theme.text_warn),
                ));
                banner.spawn((
                    Text::new(""),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: 14.0,
                        ..default()
                    },
                    TextColor(theme.text_dim),
                    DestroyedBannerDetail,
                ));
            });
        });
}

pub(super) fn update(
    sim: Res<SimulationState>,
    mut root_q: Query<&mut Visibility, With<DestroyedBannerRoot>>,
    mut detail_q: Query<&mut Text, With<DestroyedBannerDetail>>,
) {
    let destroyed = sim.simulation.is_destroyed();

    if let Ok(mut vis) = root_q.single_mut() {
        let target = if destroyed {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *vis != target {
            *vis = target;
        }
    }

    if destroyed && let Ok(mut text) = detail_q.single_mut() {
        let new_value = format!(
            "impact {:.0} m/s  —  teleport to a body to recover",
            sim.simulation.last_impact_speed_m_s()
        );
        if text.0 != new_value {
            text.0 = new_value;
        }
    }
}
