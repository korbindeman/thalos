//! Left panel: the parts palette, grouped by category. (Saved craft moved to
//! the [`hangar`](super::hangar) overlay.)

use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use crate::core::{
    EditorState, PendingPart, kind_order, palette_category_label, palette_category_order,
    palette_part_summary,
};
use thalos_game_state::units::{UnitDomain, UnitSystem, UnitsSettings};
use thalos_shipyard::blueprint::default_params_for;
use thalos_shipyard::{CatalogEntry, CatalogId, PartCatalog};

use thalos_ui::{
    self as ui, ButtonDesc, ButtonLabel, ButtonVariant, SPACE_XS, ScrollableColumn, UiButton,
    UiTheme, spawn_heading, tokens,
};

#[derive(Component)]
pub(super) struct PalettePartButton {
    catalog_id: CatalogId,
}

/// The summary line under a part's name. Carries its catalog id so
/// [`refresh_summaries`] can re-render it when the measurement preference
/// changes — the palette is built once at startup and never otherwise rebuilt,
/// so without this it would keep whatever units it was born with.
#[derive(Component)]
pub(super) struct PalettePartSummary {
    catalog_id: CatalogId,
}

pub(super) fn spawn(
    root: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    catalog: &PartCatalog,
    system: UnitSystem,
) {
    root.spawn((
        Node {
            left: Val::Px(12.0),
            top: Val::Px(64.0),
            bottom: Val::Px(12.0),
            width: Val::Px(248.0),
            ..ui::floating_panel_node()
        },
        theme.glass(),
        Interaction::None,
        Name::new("ShipyardPalette"),
    ))
    .with_children(|panel| {
        spawn_heading(panel, theme, "PARTS", false);

        panel
            .spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(SPACE_XS),
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
                        spawn_heading(
                            list,
                            theme,
                            &category.to_ascii_uppercase(),
                            current_category.is_some(),
                        );
                        current_category = Some(category);
                    }
                    spawn_part_button(list, theme, id, entry, system);
                }
            });
    });
}

/// A two-line part card: name + dim summary, taking the shared button styling.
fn spawn_part_button(
    list: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    id: &CatalogId,
    entry: &CatalogEntry,
    system: UnitSystem,
) {
    list.spawn((
        Button,
        Node {
            width: Val::Percent(100.0),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(ui::RADIUS_CTRL)),
            padding: UiRect::axes(Val::Px(ui::SPACE_SM), Val::Px(5.0)),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(1.0),
            ..default()
        },
        BackgroundColor(Color::NONE),
        BorderColor::all(tokens::STROKE),
        Interaction::None,
        UiButton::new(ButtonVariant::Ghost),
        PalettePartButton {
            catalog_id: id.clone(),
        },
    ))
    .with_children(|button| {
        button.spawn((theme.body_strong(entry.display_name()), ButtonLabel));
        let mut summary = theme.faint(palette_part_summary(entry, system));
        summary.1.font_size = FontSize::Px(10.0);
        button.spawn((
            summary,
            ButtonDesc,
            PalettePartSummary {
                catalog_id: id.clone(),
            },
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

/// Re-render the part summaries when the measurement preference changes.
///
/// The palette is spawned once at startup, so a units change made in the
/// settings menu would otherwise leave every part card stating the old unit
/// until the next launch.
pub(super) fn refresh_summaries(
    units: Res<UnitsSettings>,
    catalog: Res<PartCatalog>,
    mut summaries: Query<(&PalettePartSummary, &mut Text)>,
) {
    if !units.is_changed() {
        return;
    }
    let system = units.system_for(UnitDomain::General);
    for (summary, mut text) in &mut summaries {
        let Some(entry) = catalog.parts.get(&summary.catalog_id) else {
            continue;
        };
        let line = palette_part_summary(entry, system);
        if text.0 != line {
            text.0 = line;
        }
    }
}
