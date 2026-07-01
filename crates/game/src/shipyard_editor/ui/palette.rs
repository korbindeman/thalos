//! Left panel: the parts palette (grouped by category, same ordering as the
//! standalone editor) and the saved-ship list with load/delete.

use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use thalos_shipyard::blueprint::default_params_for;
use crate::shipyard_editor::core::{
    EditorState, PendingPart, SavedShip, kind_order, palette_category_label,
    palette_category_order, palette_part_summary,
};
use thalos_shipyard::{CatalogEntry, CatalogId, PartCatalog};

use crate::hud::theme::{HudTheme, panel_frame};

use super::widgets::{self, EditorUiButton, ScrollableColumn};

#[derive(Component)]
pub(super) struct PalettePartButton {
    catalog_id: CatalogId,
}

#[derive(Component)]
pub(super) struct LoadShipButton {
    slug: String,
}

#[derive(Component)]
pub(super) struct DeleteShipButton {
    slug: String,
}

/// Container whose children are the saved-ship rows; rebuilt when the list
/// changes.
#[derive(Component)]
pub(super) struct SavedShipsSection;

pub(super) fn spawn(root: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, catalog: &PartCatalog) {
    let (bg, border) = panel_frame(theme);
    root.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(12.0),
            top: Val::Px(60.0),
            bottom: Val::Px(12.0),
            width: Val::Px(248.0),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(4.0)),
            padding: UiRect::axes(Val::Px(10.0), Val::Px(8.0)),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(6.0),
            ..default()
        },
        bg,
        border,
        Interaction::None,
        Name::new("ShipyardPalette"),
    ))
    .with_children(|panel| {
        panel.spawn((
            Text::new("PARTS"),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(12.0),
                ..default()
            },
            TextColor(theme.text_subtitle),
        ));

        panel
            .spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    overflow: Overflow::scroll_y(),
                    flex_grow: 1.0,
                    ..default()
                },
                ScrollPosition::default(),
                RelativeCursorPosition::default(),
                Interaction::None,
                ScrollableColumn,
                Name::new("ShipyardPaletteScroll"),
            ))
            .with_children(|list| {
                // Stable category/kind/name ordering (HashMap iteration is not).
                let mut entries: Vec<(&CatalogId, &CatalogEntry)> = catalog.parts.iter().collect();
                entries.sort_by_key(|(_, e)| {
                    (
                        palette_category_order(e),
                        kind_order(e),
                        e.display_name().to_string(),
                    )
                });
                let mut current_category = None;
                for (id, entry) in entries {
                    let category = palette_category_label(entry);
                    if current_category != Some(category) {
                        list.spawn((
                            Text::new(category.to_ascii_uppercase()),
                            TextFont {
                                font: theme.font.clone(),
                                font_size: FontSize::Px(10.0),
                                ..default()
                            },
                            TextColor(theme.text_accent),
                            Node {
                                margin: UiRect::top(Val::Px(if current_category.is_some() {
                                    8.0
                                } else {
                                    0.0
                                })),
                                ..default()
                            },
                        ));
                        current_category = Some(category);
                    }
                    spawn_part_button(list, theme, id, entry);
                }

                // Saved ships live in the same scroll column, below the parts.
                list.spawn((
                    Text::new("SAVED SHIPS"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(10.0),
                        ..default()
                    },
                    TextColor(theme.text_accent),
                    Node {
                        margin: UiRect::top(Val::Px(10.0)),
                        ..default()
                    },
                ));
                list.spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(3.0),
                        ..default()
                    },
                    SavedShipsSection,
                ));
            });
    });
}

fn spawn_part_button(
    list: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    id: &CatalogId,
    entry: &CatalogEntry,
) {
    list.spawn((
        Button,
        Node {
            width: Val::Percent(100.0),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(3.0)),
            padding: UiRect::axes(Val::Px(8.0), Val::Px(5.0)),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(1.0),
            ..default()
        },
        BackgroundColor(theme.panel_bg),
        BorderColor::all(theme.panel_border),
        Interaction::None,
        EditorUiButton::default(),
        PalettePartButton {
            catalog_id: id.clone(),
        },
    ))
    .with_children(|button| {
        button.spawn((
            Text::new(entry.display_name()),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(11.0),
                ..default()
            },
            TextColor(theme.text_primary),
        ));
        button.spawn((
            Text::new(palette_part_summary(entry)),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(9.0),
                ..default()
            },
            TextColor(theme.text_dim),
        ));
    });
}

pub(super) fn handle_part_clicks(
    interactions: Query<(&Interaction, &PalettePartButton), Changed<Interaction>>,
    catalog: Res<PartCatalog>,
    mut state: ResMut<EditorState>,
) {
    for (interaction, button) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        let Ok(entry) = catalog.resolve(&button.catalog_id) else {
            continue;
        };
        state.pending = Some(PendingPart {
            catalog_id: button.catalog_id.clone(),
            params: default_params_for(entry),
        });
        state.status = format!("{} armed", entry.display_name());
    }
}

pub(super) fn handle_saved_ship_clicks(
    loads: Query<(&Interaction, &LoadShipButton), Changed<Interaction>>,
    deletes: Query<(&Interaction, &DeleteShipButton), Changed<Interaction>>,
    mut state: ResMut<EditorState>,
) {
    for (interaction, button) in &loads {
        if matches!(interaction, Interaction::Pressed) {
            state.load_target = Some(button.slug.clone());
        }
    }
    for (interaction, button) in &deletes {
        if matches!(interaction, Interaction::Pressed) {
            state.delete_file = Some(button.slug.clone());
        }
    }
}

/// Rebuild the saved-ship rows whenever the listing changes.
pub(super) fn rebuild_saved_ships(
    mut commands: Commands,
    state: Res<EditorState>,
    theme: Res<HudTheme>,
    section: Query<(Entity, Option<&Children>), With<SavedShipsSection>>,
    mut shown: Local<Option<Vec<SavedShip>>>,
) {
    if shown.as_ref() == Some(&state.ship_list) {
        return;
    }
    *shown = Some(state.ship_list.clone());

    let Ok((section_entity, children)) = section.single() else {
        return;
    };
    if let Some(children) = children {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }

    commands.entity(section_entity).with_children(|list| {
        if state.ship_list.is_empty() {
            list.spawn((
                Text::new("(none)"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(10.0),
                    ..default()
                },
                TextColor(theme.text_dim),
            ));
            return;
        }
        for saved in &state.ship_list {
            list.spawn(Node {
                width: Val::Percent(100.0),
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(4.0),
                ..default()
            })
            .with_children(|row| {
                widgets::spawn_button(
                    row,
                    &theme,
                    LoadShipButton {
                        slug: saved.slug.clone(),
                    },
                    "LOAD",
                    9.0,
                    20.0,
                );
                widgets::spawn_button(
                    row,
                    &theme,
                    DeleteShipButton {
                        slug: saved.slug.clone(),
                    },
                    "×",
                    10.0,
                    20.0,
                );
                row.spawn((
                    Text::new(saved.name.clone()),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(10.0),
                        ..default()
                    },
                    TextColor(theme.text_primary),
                ));
            });
        }
    });
}
