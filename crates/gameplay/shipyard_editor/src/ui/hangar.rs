//! The **HANGAR** overlay — proper craft load/save management.
//!
//! A centred modal over the editor listing every saved craft (`ships/*.ron`):
//! click a row to load it onto the canvas, `×` to delete the file. The current
//! build saves from the top bar's SAVE (under the name field's name); this
//! overlay is the browse/restore side. Replaces the old saved-ship list that
//! was buried at the bottom of the parts palette.

use bevy::picking::Pickable;
use bevy::prelude::*;

use thalos_ui::{
    self as ui, ButtonDesc, ButtonLabel, ButtonVariant, SPACE_SM, SPACE_XS, UiButton, UiTheme,
    spawn_button, spawn_divider, tokens,
};

use crate::core::{EditorState, SavedShip};

/// Whether the hangar overlay is up. **Sole writer:** the top bar's HANGAR
/// toggle + this module's close/load paths.
#[derive(Resource, Default)]
pub(super) struct HangarOpen(pub bool);

#[derive(Component)]
pub(super) struct HangarRoot;

/// Container whose children are the saved-craft rows; rebuilt when the
/// listing changes.
#[derive(Component)]
pub(super) struct HangarList;

#[derive(Component)]
pub(super) struct HangarCloseButton;

#[derive(Component)]
pub(super) struct LoadShipButton {
    slug: String,
}

#[derive(Component)]
pub(super) struct DeleteShipButton {
    slug: String,
}

pub(super) fn setup(mut commands: Commands, theme: Res<UiTheme>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.35)),
            GlobalZIndex(70),
            Pickable {
                is_hoverable: true,
                should_block_lower: true,
            },
            Interaction::None,
            Visibility::Hidden,
            HangarRoot,
            Name::new("ShipyardHangar"),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(420.0),
                    max_height: Val::Percent(70.0),
                    align_items: AlignItems::Stretch,
                    ..ui::panel_node()
                },
                theme.glass_heavy(),
                Name::new("ShipyardHangarPanel"),
            ))
            .with_children(|panel| {
                panel
                    .spawn(Node {
                        width: Val::Percent(100.0),
                        justify_content: JustifyContent::SpaceBetween,
                        align_items: AlignItems::Center,
                        ..default()
                    })
                    .with_children(|row| {
                        row.spawn(theme.title("HANGAR"));
                        spawn_button(
                            row,
                            &theme,
                            HangarCloseButton,
                            "×",
                            ButtonVariant::Bare,
                            24.0,
                        );
                    });
                panel.spawn(theme.small(
                    "Click a craft to load it onto the canvas. SAVE in the top bar stores the current build under its name.",
                ));
                spawn_divider(panel);
                panel.spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        overflow: Overflow::scroll_y(),
                        flex_grow: 1.0,
                        row_gap: Val::Px(SPACE_XS + 2.0),
                        ..default()
                    },
                    ScrollPosition::default(),
                    bevy::ui::RelativeCursorPosition::default(),
                    Interaction::None,
                    thalos_ui::ScrollableColumn,
                    HangarList,
                    Name::new("ShipyardHangarList"),
                ));
            });
        });
}

/// Show/hide on the open flag; also close when the editor itself closes.
pub(super) fn sync_visibility(
    editor: Res<super::super::ShipyardEditor>,
    mut hangar: ResMut<HangarOpen>,
    mut roots: Query<&mut Visibility, With<HangarRoot>>,
) {
    if !editor.open && hangar.0 {
        hangar.0 = false;
    }
    let target = if hangar.0 {
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

/// Rebuild the craft rows whenever the listing changes.
pub(super) fn rebuild_list(
    mut commands: Commands,
    state: Res<EditorState>,
    theme: Res<UiTheme>,
    list: Query<(Entity, Option<&Children>), With<HangarList>>,
    mut shown: Local<Option<Vec<SavedShip>>>,
) {
    if shown.as_ref() == Some(&state.ship_list) {
        return;
    }
    *shown = Some(state.ship_list.clone());

    let Ok((list_entity, children)) = list.single() else {
        return;
    };
    if let Some(children) = children {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }

    commands.entity(list_entity).with_children(|list| {
        if state.ship_list.is_empty() {
            list.spawn(theme.faint("(no saved craft yet — SAVE stores the current build)"));
            return;
        }
        for saved in &state.ship_list {
            list.spawn(Node {
                width: Val::Percent(100.0),
                column_gap: Val::Px(SPACE_SM),
                align_items: AlignItems::Center,
                ..default()
            })
            .with_children(|row| {
                // The row itself loads the craft.
                row.spawn((
                    Button,
                    Node {
                        flex_grow: 1.0,
                        height: Val::Px(ui::CTRL_H_LG - 4.0),
                        border: UiRect::all(Val::Px(1.0)),
                        border_radius: BorderRadius::all(Val::Px(ui::RADIUS_CTRL)),
                        padding: UiRect::horizontal(Val::Px(ui::SPACE_MD)),
                        justify_content: JustifyContent::SpaceBetween,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::NONE),
                    BorderColor::all(tokens::STROKE),
                    Interaction::None,
                    UiButton::new(ButtonVariant::Ghost),
                    LoadShipButton {
                        slug: saved.slug.clone(),
                    },
                ))
                .with_children(|button| {
                    button.spawn((theme.body_strong(saved.name.clone()), ButtonLabel));
                    button.spawn((theme.faint(format!("{}.ron", saved.slug)), ButtonDesc));
                });
                spawn_button(
                    row,
                    &theme,
                    DeleteShipButton {
                        slug: saved.slug.clone(),
                    },
                    "×",
                    ButtonVariant::Danger,
                    ui::CTRL_H,
                );
            });
        }
    });
}

pub(super) fn handle_clicks(
    loads: Query<(&Interaction, &LoadShipButton), Changed<Interaction>>,
    deletes: Query<(&Interaction, &DeleteShipButton), Changed<Interaction>>,
    closes: Query<&Interaction, (Changed<Interaction>, With<HangarCloseButton>)>,
    mut state: ResMut<EditorState>,
    mut hangar: ResMut<HangarOpen>,
) {
    for (interaction, button) in &loads {
        if matches!(interaction, Interaction::Pressed) {
            state.load_target = Some(button.slug.clone());
            hangar.0 = false;
        }
    }
    for (interaction, button) in &deletes {
        if matches!(interaction, Interaction::Pressed) {
            state.delete_file = Some(button.slug.clone());
        }
    }
    for interaction in &closes {
        if matches!(interaction, Interaction::Pressed) {
            hangar.0 = false;
        }
    }
}
