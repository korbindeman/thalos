//! Space-center hub UI: a side panel with the facility shortcut buttons
//! (EDIT BASE, VAB) and EXIT-to-flight.
//!
//! Selecting/entering buildings is the hover picker's job (see
//! [`select`](super::select)): hover a building for its name in a floating
//! callout, click the VAB to enter. The panel's buttons are just always-available
//! shortcuts for the same facilities.
//!
//! Native Bevy UI in the [`HudTheme`](crate::hud::theme::HudTheme) style, built
//! once at startup and shown only while the hub is open (root visibility toggled
//! on the [`SpaceCenter`] change edge, like the pause menu).

use bevy::picking::prelude::Pickable;
use bevy::prelude::*;

use crate::base_editor::BaseEditor;
use crate::hud::theme::{HudTheme, panel_frame};
use crate::loading::AppState;
use crate::shipyard_editor::ShipyardEditor;
use crate::spawn::Homeworld;
use crate::structures::{Facility, StructureRegistry};

use super::{
    ReturnToSpaceCenter, SpaceCenter, enter_base_editor, enter_facility, home_base_site,
    space_center_open,
};

#[derive(Component)]
struct SpaceCenterUiRoot;

/// What a hub button does.
#[derive(Component, Clone, Copy)]
enum HubButton {
    EditBase,
    Vab,
    /// Close the hub back to flight.
    Exit,
}

pub(super) struct SpaceCenterUiPlugin;

impl Plugin for SpaceCenterUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup.after(crate::hud::theme::init_theme))
            .add_systems(Update, update_visibility)
            .add_systems(
                Update,
                (handle_clicks, update_button_visuals).run_if(space_center_open),
            );
    }
}

const PANEL_WIDTH: f32 = 300.0;

fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    let (bg, border) = panel_frame(&theme);
    commands
        .spawn((
            SpaceCenterUiRoot,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(24.0),
                top: Val::Px(24.0),
                width: Val::Px(PANEL_WIDTH),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                padding: UiRect::axes(Val::Px(18.0), Val::Px(16.0)),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Stretch,
                row_gap: Val::Px(8.0),
                ..default()
            },
            bg,
            border,
            GlobalZIndex(90),
            Visibility::Hidden,
            // `Interaction` makes the whole panel a pointer sink, so
            // `UiPointerGate` reports the cursor as over-UI anywhere on it and
            // the hub's building-pick raycast is suppressed (not just over the
            // buttons).
            Interaction::None,
            Pickable {
                is_hoverable: true,
                should_block_lower: true,
            },
            Name::new("SpaceCenterUi"),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("SPACE CENTER"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(20.0),
                    ..default()
                },
                TextColor(theme.text_accent),
                Name::new("SpaceCenterTitle"),
            ));
            panel.spawn((
                Text::new("Right-drag to orbit · scroll to zoom · WASD to pan"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(10.0),
                    ..default()
                },
                TextColor(theme.text_dim),
            ));
            panel.spawn((
                Text::new("Hover a building for its name · click the VAB to enter"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(10.0),
                    ..default()
                },
                TextColor(theme.text_dim),
            ));

            divider(panel, &theme);
            heading(panel, &theme, "FACILITIES");
            spawn_button(panel, &theme, HubButton::EditBase, "EDIT BASE", "reshape the site");
            spawn_button(panel, &theme, HubButton::Vab, "VAB", "assemble a craft");

            divider(panel, &theme);
            spawn_button(panel, &theme, HubButton::Exit, "EXIT", "Esc");
        });
}

fn heading(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, label: &str) {
    parent.spawn((
        Text::new(label.to_string()),
        TextFont {
            font: theme.font.clone(),
            font_size: FontSize::Px(11.0),
            ..default()
        },
        TextColor(theme.text_dim),
    ));
}

fn divider(parent: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    parent.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Px(1.0),
            margin: UiRect::vertical(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(theme.panel_border),
    ));
}

/// Spawn a hub facility/exit button with a label and a dim description.
fn spawn_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    action: HubButton,
    label: &str,
    desc: &str,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                height: Val::Px(30.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                padding: UiRect::horizontal(Val::Px(12.0)),
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(theme.panel_bg),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            action,
            Name::new(format!("SpaceCenter{label}Button")),
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(label.to_string()),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(theme.text_primary),
            ));
            if !desc.is_empty() {
                c.spawn((
                    Text::new(desc.to_string()),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(10.0),
                        ..default()
                    },
                    TextColor(theme.text_dim),
                ));
            }
        });
}

/// Show/hide the whole panel on the hub open/close edge.
fn update_visibility(sc: Res<SpaceCenter>, mut roots: Query<&mut Visibility, With<SpaceCenterUiRoot>>) {
    if !sc.is_changed() {
        return;
    }
    let target = if sc.open {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut roots {
        if *vis != target {
            *vis = target;
        }
    }
}

fn handle_clicks(
    interactions: Query<(&Interaction, &HubButton), Changed<Interaction>>,
    mut sc: ResMut<SpaceCenter>,
    mut shipyard: ResMut<ShipyardEditor>,
    mut base: ResMut<BaseEditor>,
    mut return_flag: ResMut<ReturnToSpaceCenter>,
    mut next_state: ResMut<NextState<AppState>>,
    homeworld: Res<Homeworld>,
    registry: Res<StructureRegistry>,
) {
    for (interaction, action) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        match action {
            HubButton::EditBase => {
                let base_site = home_base_site(&registry, homeworld.0).map(|s| s.id);
                enter_base_editor(&mut sc, &mut base, &mut return_flag, base_site);
            }
            HubButton::Vab => {
                enter_facility(Facility::Vab, &mut sc, &mut shipyard, &mut return_flag);
            }
            HubButton::Exit => {
                // Back to the main menu when the hub is the session root (PLAY),
                // otherwise back to the flight it was opened over.
                if sc.return_to_menu {
                    next_state.set(AppState::MainMenu);
                }
                sc.open = false;
                sc.hovered = None;
            }
        }
    }
}

fn update_button_visuals(
    theme: Res<HudTheme>,
    mut buttons: Query<
        (
            &Interaction,
            &mut BorderColor,
            &mut BackgroundColor,
            &Children,
        ),
        With<HubButton>,
    >,
    mut text_q: Query<&mut TextColor>,
) {
    for (interaction, mut border, mut bg, children) in &mut buttons {
        let (border_color, bg_color, label_color) = match interaction {
            Interaction::Pressed => (theme.text_primary, theme.panel_border, theme.text_primary),
            Interaction::Hovered => (theme.text_accent, theme.panel_bg, theme.text_accent),
            Interaction::None => (theme.panel_border, theme.panel_bg, theme.text_primary),
        };
        let new_border = BorderColor::all(border_color);
        if border.top != new_border.top {
            *border = new_border;
        }
        if bg.0 != bg_color {
            bg.0 = bg_color;
        }
        if let Some(&child) = children.first()
            && let Ok(mut tc) = text_q.get_mut(child)
            && tc.0 != label_color
        {
            tc.0 = label_color;
        }
    }
}
