//! Minimal in-world HUD for the base editor: a status/help panel that shows the
//! current mode, the controls, and the pending building footprint. Styled from
//! [`HudTheme`]. (A full interactive parts palette + slider inspector — mirroring
//! the shipyard editor's `ui/` — is a natural follow-up; the foundation slice
//! drives footprint sizing from the keyboard, surfaced here.)

use bevy::prelude::*;

use crate::hud::theme::{HudTheme, panel_frame};

use super::place::PendingKind;
use super::{BaseBuildState, BaseEditor, BaseEditorMode};

#[derive(Component)]
struct BaseEditorOverlay;

#[derive(Component)]
struct BaseEditorText;

pub(super) struct BaseEditorUiPlugin;

impl Plugin for BaseEditorUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_overlay.after(crate::hud::theme::init_theme))
            .add_systems(Update, (toggle_overlay, update_overlay_text));
    }
}

fn setup_overlay(mut commands: Commands, theme: Res<HudTheme>) {
    let (bg, border) = panel_frame(&theme);
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(16.0),
                top: Val::Px(16.0),
                width: Val::Px(300.0),
                padding: UiRect::axes(Val::Px(14.0), Val::Px(12.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            bg,
            border,
            GlobalZIndex(60),
            Visibility::Hidden,
            BaseEditorOverlay,
            Name::new("BaseEditorOverlay"),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("SURFACE BASE EDITOR"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 13.0,
                    ..default()
                },
                TextColor(theme.text_accent),
            ));
            panel.spawn((
                Text::new(""),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.text_primary),
                Node {
                    margin: UiRect::top(Val::Px(8.0)),
                    ..default()
                },
                BaseEditorText,
            ));
        });
}

fn toggle_overlay(
    editor: Res<BaseEditor>,
    mut overlay: Query<&mut Visibility, With<BaseEditorOverlay>>,
) {
    if !editor.is_changed() {
        return;
    }
    let target = if editor.open {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut overlay {
        if *vis != target {
            *vis = target;
        }
    }
}

fn update_overlay_text(
    editor: Res<BaseEditor>,
    build: Res<BaseBuildState>,
    mut text: Query<&mut Text, With<BaseEditorText>>,
) {
    if !editor.open {
        return;
    }
    let body = match editor.mode {
        BaseEditorMode::PickSite => "Mode: PICK SITE\n\nLMB  confirm site (flattens the land)\nQ/E  rotate footprint\nEsc  exit".to_string(),
        BaseEditorMode::PlaceBuildings => {
            let (kind_label, next) = match build.pending_kind {
                PendingKind::Building => {
                    let d = build.pending;
                    (
                        "BUILDING",
                        format!(
                            "[ ] footprint   - = height\nNext: {:.0} x {:.0} m, {:.0} m tall",
                            d.half_x_m * 2.0,
                            d.half_z_m * 2.0,
                            d.height_m,
                        ),
                    )
                }
                PendingKind::Launchpad => (
                    "LAUNCHPAD",
                    format!(
                        "[ ] radius   L: launch ship onto selected pad\nNext: {:.0} m across",
                        build.pending_radius_m * 2.0,
                    ),
                ),
            };
            format!(
                "Mode: PLACE — {kind_label}\n\nLMB  place / select\nTab  building / launchpad\nX    delete selected\nQ/E  rotate\n{next}\nEsc  exit",
            )
        }
    };
    for mut t in &mut text {
        if t.0 != body {
            t.0 = body.clone();
        }
    }
}
