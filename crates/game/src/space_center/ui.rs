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
use thalos_ui::{
    self as ui, SPACE_XS, UiTheme, spawn_divider, spawn_heading, spawn_menu_row,
};

use crate::base_editor::BaseEditor;
use crate::game_context::{ContextHistory, GameContext, back_out};
use crate::loading::AppState;
use crate::spawn::Homeworld;
use crate::structures::{Facility, StructureRegistry};

use super::{SpaceCenter, enter_base_editor, enter_facility, home_base_site, space_center_open};

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
        app.add_systems(Startup, setup.after(thalos_ui::init_ui_theme))
            .add_systems(Update, update_visibility)
            .add_systems(Update, handle_clicks.run_if(space_center_open));
    }
}

const PANEL_WIDTH: f32 = 300.0;

fn setup(mut commands: Commands, theme: Res<UiTheme>) {
    commands
        .spawn((
            SpaceCenterUiRoot,
            Node {
                left: Val::Px(24.0),
                top: Val::Px(24.0),
                width: Val::Px(PANEL_WIDTH),
                align_items: AlignItems::Stretch,
                ..ui::floating_panel_node()
            },
            theme.glass(),
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
            panel.spawn((theme.title("SPACE CENTER"), Name::new("SpaceCenterTitle")));
            panel.spawn(theme.small("Right-drag to orbit · scroll to zoom · WASD to pan"));
            panel.spawn(theme.small("Hover a building for its name · click the VAB to enter"));

            spawn_divider(panel);
            spawn_heading(panel, &theme, "FACILITIES", false);
            panel
                .spawn(Node {
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    row_gap: Val::Px(SPACE_XS + 2.0),
                    ..default()
                })
                .with_children(|buttons| {
                    spawn_menu_row(buttons, &theme, HubButton::EditBase, "EDIT BASE", "reshape the site");
                    spawn_menu_row(buttons, &theme, HubButton::Vab, "VAB", "assemble a craft");
                    spawn_divider(buttons);
                    spawn_menu_row(buttons, &theme, HubButton::Exit, "EXIT", "Esc");
                });
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
    mut base: ResMut<BaseEditor>,
    mut next_ctx: Option<ResMut<NextState<GameContext>>>,
    mut history: ResMut<ContextHistory>,
    mut next_state: ResMut<NextState<AppState>>,
    homeworld: Res<Homeworld>,
    registry: Res<StructureRegistry>,
) {
    for (interaction, action) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let Some(next) = next_ctx.as_mut() else {
            continue;
        };
        match action {
            HubButton::EditBase => {
                let base_site = home_base_site(&registry, homeworld.0).map(|s| s.id);
                enter_base_editor(&mut base, next, &mut history, base_site);
            }
            HubButton::Vab => {
                enter_facility(Facility::Vab, next, &mut history);
            }
            HubButton::Exit => {
                // Back out toward the parent the hub was opened over (a flight),
                // or to the start screen when the hub is the session root (PLAY):
                // an empty return stack means root. Escape never leaves the game,
                // but the EXIT button is a deliberate exit.
                if back_out(next, &mut history).is_none() {
                    next_state.set(AppState::MainMenu);
                }
            }
        }
    }
}

