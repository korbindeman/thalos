//! Editor top bar: title, ship name field, live build stats, the
//! mirror / snap / layout toggles, and save / new / exit.

use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;

use thalos_shipyard::Ship;
use thalos_shipyard::editor::{
    BuildOrientation, EditorPart, EditorState, PlacementSnap, SymmetryMode, format_delta_v,
    format_mass_kg,
};

use crate::hud::theme::{HudTheme, panel_frame};
use crate::relaunch::{RelaunchRequest, RelaunchSpec};
use crate::spawn::SpawnSituation;

use super::widgets::{self, EditorTextFocus, EditorUiButton, ShipNameField};
use super::{EditorStatsCache, ShipyardEditor};

#[derive(Component, Clone, Copy, PartialEq)]
pub(super) enum TopBarAction {
    ToggleMirror,
    ToggleSnap,
    ToggleLayout,
    Launch,
    Save,
    New,
    Exit,
}

#[derive(Component)]
pub(super) struct StatsText;

#[derive(Component)]
pub(super) struct ShipNameText;

pub(super) fn spawn(root: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    let (bg, border) = panel_frame(theme);
    root.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(12.0),
            right: Val::Px(12.0),
            top: Val::Px(12.0),
            height: Val::Px(40.0),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(4.0)),
            padding: UiRect::axes(Val::Px(14.0), Val::Px(4.0)),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        },
        bg,
        border,
        Interaction::None,
        Name::new("ShipyardTopBar"),
    ))
    .with_children(|bar| {
        bar.spawn((
            Text::new("SHIPYARD"),
            TextFont {
                font: theme.font.clone(),
                font_size: 15.0,
                ..default()
            },
            TextColor(theme.text_accent),
        ));

        // Ship name field — minimal single-line text input.
        bar.spawn((
            Button,
            Node {
                width: Val::Px(220.0),
                height: Val::Px(26.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(3.0)),
                padding: UiRect::axes(Val::Px(8.0), Val::Px(2.0)),
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.02, 0.02, 0.9)),
            BorderColor::all(theme.panel_border),
            Interaction::None,
            ShipNameField,
            Name::new("ShipyardNameField"),
        ))
        .with_children(|field| {
            field.spawn((
                Text::new(""),
                TextFont {
                    font: theme.font.clone(),
                    font_size: 12.0,
                    ..default()
                },
                TextColor(theme.text_primary),
                ShipNameText,
            ));
        });

        // Live stats, centred in the leftover space.
        bar.spawn((
            Text::new(""),
            TextFont {
                font: theme.font.clone(),
                font_size: 11.0,
                ..default()
            },
            TextColor(theme.text_dim),
            Node {
                flex_grow: 1.0,
                justify_content: JustifyContent::Center,
                ..default()
            },
            StatsText,
        ));

        widgets::spawn_button(bar, theme, TopBarAction::ToggleMirror, "MIRROR 2×", 10.0, 24.0);
        widgets::spawn_button(bar, theme, TopBarAction::ToggleSnap, "SNAP 15°", 10.0, 24.0);
        widgets::spawn_button(bar, theme, TopBarAction::ToggleLayout, "HANGAR", 10.0, 24.0);
        // LAUNCH is the headline action — fly the current design. The play
        // glyph sets it apart from the SAVE/NEW/EXIT housekeeping buttons.
        widgets::spawn_button(bar, theme, TopBarAction::Launch, "▶ LAUNCH", 11.0, 24.0);
        widgets::spawn_button(bar, theme, TopBarAction::Save, "SAVE", 11.0, 24.0);
        widgets::spawn_button(bar, theme, TopBarAction::New, "NEW", 11.0, 24.0);
        widgets::spawn_button(bar, theme, TopBarAction::Exit, "EXIT", 11.0, 24.0);
    });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_actions(
    interactions: Query<(&Interaction, &TopBarAction), Changed<Interaction>>,
    mut state: ResMut<EditorState>,
    mut symmetry: ResMut<SymmetryMode>,
    mut snap: ResMut<PlacementSnap>,
    mut orientation: ResMut<BuildOrientation>,
    mut editor: ResMut<ShipyardEditor>,
    cache: Res<EditorStatsCache>,
    mut relaunch: ResMut<RelaunchRequest>,
) {
    for (interaction, action) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        match action {
            TopBarAction::ToggleMirror => symmetry.mirror = !symmetry.mirror,
            TopBarAction::ToggleSnap => snap.enabled = !snap.enabled,
            TopBarAction::ToggleLayout => {
                let flipped = !orientation.bypass_change_detection().horizontal;
                orientation.horizontal = flipped;
            }
            TopBarAction::Launch => {
                let Some(blueprint) = cache.blueprint.clone() else {
                    state.status = "Nothing to launch — build a ship first".into();
                    continue;
                };
                // Aircraft (any wing area) fly airborne over land; everything
                // else launches into the parking orbit. Persist the design too.
                let is_aircraft = matches!(
                    &cache.stats,
                    Some(Ok(stats)) if stats.wing_area_m2 > 0.0
                );
                let situation = if is_aircraft {
                    SpawnSituation::Cruise
                } else {
                    SpawnSituation::ShipOrbit
                };
                state.save_requested = true;
                relaunch.0 = Some(RelaunchSpec {
                    blueprint,
                    situation,
                });
                editor.open = false;
                state.status = "Launching…".into();
            }
            TopBarAction::Save => state.save_requested = true,
            TopBarAction::New => {
                // Clear the canvas: deleting the root despawns the whole
                // build (the core's delete path).
                if state.ship_root.is_some() {
                    state.selected = state.ship_root;
                    state.delete_selected = true;
                } else {
                    state.status = "Canvas already empty".into();
                }
            }
            TopBarAction::Exit => editor.open = false,
        }
    }
}

/// Render the latched state of the three toggles.
pub(super) fn update_toggle_latches(
    symmetry: Res<SymmetryMode>,
    snap: Res<PlacementSnap>,
    orientation: Res<BuildOrientation>,
    mut buttons: Query<(&TopBarAction, &mut EditorUiButton)>,
) {
    for (action, mut button) in &mut buttons {
        let latched = match action {
            TopBarAction::ToggleMirror => symmetry.mirror,
            TopBarAction::ToggleSnap => snap.enabled,
            TopBarAction::ToggleLayout => orientation.horizontal,
            _ => continue,
        };
        if button.latched != latched {
            button.latched = latched;
        }
    }
}

pub(super) fn update_stats_text(
    cache: Res<EditorStatsCache>,
    mut texts: Query<&mut Text, With<StatsText>>,
) {
    let Ok(mut text) = texts.single_mut() else {
        return;
    };
    let line = match (&cache.stats, &cache.staging) {
        (Some(Ok(stats)), staging) => {
            // `.max(0.0)` keeps an engine-less build from reading "Δv -0 m/s"
            // (tiny negative / negative-zero sums from the stage estimate).
            let total_dv: f64 = staging
                .as_ref()
                .and_then(|s| s.as_ref().ok())
                .map(|s| s.iter().map(|st| st.delta_v_m_s).sum::<f64>().max(0.0))
                .unwrap_or(0.0);
            let mut line = format!(
                "MASS {}  ·  Δv {}",
                format_mass_kg(stats.wet_mass_kg()),
                format_delta_v(total_dv),
            );
            if stats.wet_mass_kg() > 0.0 && stats.total_thrust_n > 0.0 {
                line.push_str(&format!(
                    "  ·  TWR {:.2}",
                    stats.current_acceleration() / thalos_shipyard::G0
                ));
            }
            if stats.wing_area_m2 > 0.0 {
                line.push_str(&format!("  ·  WING {:.1} m²", stats.wing_area_m2));
            }
            line
        }
        (Some(Err(e)), _) => format!("stats error: {e}"),
        (None, _) => "EMPTY CANVAS — pick a part from the palette".into(),
    };
    if **text != line {
        **text = line;
    }
}

/// Feed key events into the ship name while the field is focused. Writes to
/// the live `Ship` entity when one exists, falling back to the staged
/// `EditorState::ship_name` used for the next root placement.
pub(super) fn apply_name_input(
    mut focus: ResMut<EditorTextFocus>,
    mut key_events: MessageReader<KeyboardInput>,
    mut state: ResMut<EditorState>,
    mut ships: Query<&mut Ship, With<EditorPart>>,
) {
    if !focus.is_focused() {
        // Drain so stale events don't replay on the next focus.
        key_events.clear();
        return;
    }
    if let Some(ship_entity) = state.ship_entity
        && let Ok(mut ship) = ships.get_mut(ship_entity)
    {
        let mut name = ship.name.clone();
        widgets::collect_text_edits(&mut focus, &mut key_events, &mut name);
        if ship.name != name {
            ship.name = name.clone();
        }
        // Keep the staged copy aligned so clearing the canvas keeps the
        // typed name.
        if state.ship_name != name {
            state.ship_name = name;
        }
        return;
    }
    let mut name = state.ship_name.clone();
    widgets::collect_text_edits(&mut focus, &mut key_events, &mut name);
    if state.ship_name != name {
        state.ship_name = name;
    }
}

pub(super) fn update_name_display(
    state: Res<EditorState>,
    focus: Res<EditorTextFocus>,
    theme: Res<HudTheme>,
    ships: Query<&Ship, With<EditorPart>>,
    mut name_text: Query<&mut Text, With<ShipNameText>>,
    mut field: Query<&mut BorderColor, With<ShipNameField>>,
) {
    let name = state
        .ship_entity
        .and_then(|e| ships.get(e).ok())
        .map(|s| s.name.clone())
        .unwrap_or_else(|| state.ship_name.clone());
    let shown = if focus.is_focused() {
        format!("{name}_")
    } else if name.is_empty() {
        "(unnamed)".to_string()
    } else {
        name
    };
    if let Ok(mut text) = name_text.single_mut()
        && **text != shown
    {
        **text = shown;
    }
    if let Ok(mut border) = field.single_mut() {
        let color = if focus.is_focused() {
            theme.text_accent
        } else {
            theme.panel_border
        };
        let target = BorderColor::all(color);
        if border.top != target.top {
            *border = target;
        }
    }
}
