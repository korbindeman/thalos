//! Maneuver node editor (native Bevy UI).
//!
//! A bottom-centre panel shown while a node is selected: header readouts (node
//! number, burn status, time-to-burn, total Δv, estimated burn time), a
//! Delete/Dismiss button, and per-component (prograde / normal / radial) Δv
//! nudge rows. Coarse on-screen editing is done by dragging the 3-D arrow
//! handles; these nudges fine-tune. A spent (`Executed`) node hides the nudge
//! controls and offers "Dismiss".

use bevy::math::DVec3;
use bevy::prelude::*;

use super::state::{ManeuverEvent, ManeuverPlan, NodeBurnPhase, NodeDeltaV, SelectedNode};
use crate::hud::theme::{HudTheme, panel_frame};
use crate::pause_menu::GamePause;
use crate::photo_mode::PhotoMode;
use crate::rendering::SimulationState;
use crate::scenario_menu::ScenarioMenu;
use crate::ui_widgets::spawn_button;
use crate::view::ViewMode;

// ── Markers ─────────────────────────────────────────────────────────────────

#[derive(Component)]
pub(super) struct ManeuverEditorRoot;

#[derive(Component)]
pub(super) struct NudgeContainer;

#[derive(Component)]
pub(super) struct DeleteButton;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(super) enum DvAxis {
    Prograde,
    Normal,
    Radial,
}

#[derive(Component, Clone, Copy)]
pub(super) struct NudgeButton {
    axis: DvAxis,
    delta: f64,
}

/// A text readout the update system rewrites each frame.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(super) enum EditorField {
    NodeNum,
    Status,
    Time,
    DvTotal,
    Burn,
    DeleteLabel,
    AxisValue(DvAxis),
}

// ── Setup ─────────────────────────────────────────────────────────────────────

pub(super) fn setup(mut commands: Commands, theme: Res<HudTheme>) {
    let (bg, border) = panel_frame(&theme);
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                bottom: Val::Px(14.0),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::Center,
                ..default()
            },
            Visibility::Hidden,
            ManeuverEditorRoot,
            Name::new("ManeuverEditor"),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(6.0),
                    align_items: AlignItems::Center,
                    ..default()
                },
                bg,
                border,
                Name::new("ManeuverEditorPanel"),
            ))
            .with_children(|panel| {
                spawn_header(panel, &theme);
                spawn_axis_rows(panel, &theme);
            });
        });
}

fn spawn_header(panel: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    panel
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(12.0),
            ..default()
        })
        .with_children(|row| {
            text(row, theme, "MANEUVER NODE", 12.0, theme.text_accent, None);
            text(row, theme, "", 11.0, theme.text_dim, Some(EditorField::NodeNum));
            text(row, theme, "", 11.0, theme.text_accent, Some(EditorField::Status));
            text(row, theme, "", 11.0, theme.text_primary, Some(EditorField::Time));
            text(row, theme, "", 11.0, theme.text_primary, Some(EditorField::DvTotal));
            text(row, theme, "", 11.0, theme.text_dim, Some(EditorField::Burn));
            // Delete / Dismiss button.
            row.spawn((
                Button,
                Node {
                    height: Val::Px(22.0),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(3.0)),
                    padding: UiRect::axes(Val::Px(10.0), Val::Px(2.0)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(theme.panel_bg),
                BorderColor::all(theme.panel_border),
                Interaction::None,
                crate::ui_widgets::UiButton::default(),
                DeleteButton,
            ))
            .with_children(|c| {
                c.spawn((
                    Text::new("Delete"),
                    TextFont {
                        font: theme.font.clone(),
                        font_size: FontSize::Px(10.0),
                        ..default()
                    },
                    TextColor(theme.text_primary),
                    EditorField::DeleteLabel,
                ));
            });
        });
}

fn spawn_axis_rows(panel: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    panel
        .spawn((
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..default()
            },
            NudgeContainer,
        ))
        .with_children(|col| {
            spawn_axis_row(col, theme, DvAxis::Prograde, "P");
            spawn_axis_row(col, theme, DvAxis::Normal, "N");
            spawn_axis_row(col, theme, DvAxis::Radial, "R");
        });
}

fn spawn_axis_row(col: &mut ChildSpawnerCommands<'_>, theme: &HudTheme, axis: DvAxis, label: &str) {
    col.spawn(Node {
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        column_gap: Val::Px(6.0),
        ..default()
    })
    .with_children(|row| {
        spawn_button(row, theme, NudgeButton { axis, delta: -10.0 }, "−10", 9.0, 20.0);
        spawn_button(row, theme, NudgeButton { axis, delta: -1.0 }, "−1", 9.0, 20.0);
        // Label + value, centred fixed-width so the buttons don't jump.
        row.spawn((
            Text::new(label.to_string()),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(11.0),
                ..default()
            },
            TextColor(theme.text_dim),
            Node {
                width: Val::Px(14.0),
                ..default()
            },
        ));
        row.spawn((
            Text::new(""),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(11.0),
                ..default()
            },
            TextColor(theme.text_primary),
            Node {
                width: Val::Px(96.0),
                justify_content: JustifyContent::Center,
                ..default()
            },
            EditorField::AxisValue(axis),
        ));
        spawn_button(row, theme, NudgeButton { axis, delta: 1.0 }, "+1", 9.0, 20.0);
        spawn_button(row, theme, NudgeButton { axis, delta: 10.0 }, "+10", 9.0, 20.0);
    });
}

fn text(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &HudTheme,
    content: &str,
    size: f32,
    color: Color,
    field: Option<EditorField>,
) {
    let mut e = parent.spawn((
        Text::new(content.to_string()),
        TextFont {
            font: theme.font.clone(),
            font_size: FontSize::Px(size),
            ..default()
        },
        TextColor(color),
    ));
    if let Some(field) = field {
        e.insert(field);
    }
}

// ── Update ──────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(super) fn update_editor(
    mut selected: ResMut<SelectedNode>,
    plan: Res<ManeuverPlan>,
    sim: Option<Res<SimulationState>>,
    node_dv: Res<NodeDeltaV>,
    view: Res<ViewMode>,
    pause: Res<GamePause>,
    scenario: Res<ScenarioMenu>,
    photo: Res<PhotoMode>,
    theme: Res<HudTheme>,
    units: Res<crate::units_settings::UnitsSettings>,
    mut roots: Query<&mut Visibility, (With<ManeuverEditorRoot>, Without<NudgeContainer>)>,
    mut nudge: Query<&mut Visibility, (With<NudgeContainer>, Without<ManeuverEditorRoot>)>,
    mut texts: Query<(&EditorField, &mut Text)>,
    mut colors: Query<(&EditorField, &mut TextColor)>,
) {
    let map_ok =
        *view == ViewMode::Map && !pause.active && !scenario.open && !photo.active;

    // Resolve the selected node; drop a dangling selection.
    let node = selected
        .id
        .and_then(|id| plan.nodes.iter().find(|n| n.id == id));
    if selected.id.is_some() && node.is_none() {
        selected.id = None;
    }
    let visible = map_ok && node.is_some();

    set_visibility(&mut roots, visible);
    if !visible {
        return;
    }
    let node = node.unwrap();
    let sel_id = selected.id.unwrap();

    let total_dv = node.delta_v.length();
    let phase = node.phase;
    let executed = phase == NodeBurnPhase::Executed;
    let executing = phase == NodeBurnPhase::Executing;
    let sim_time = sim.as_ref().map(|s| s.simulation.sim_time()).unwrap_or(0.0);
    let time_until = node.time - sim_time;
    let burn_duration = sim
        .as_ref()
        .map(|s| s.simulation.estimated_burn_duration(total_dv))
        .unwrap_or(0.0);

    set_visibility(&mut nudge, !executed);

    let status = if executed {
        "EXECUTED"
    } else if executing {
        "BURNING"
    } else {
        ""
    };
    let status_color = if executing {
        Color::srgb(0.90, 0.78, 0.35)
    } else {
        Color::srgb(0.47, 0.71, 0.47)
    };
    let burn_text = if executed {
        String::new()
    } else {
        format!("Est. burn: {burn_duration:.1}s")
    };

    for (field, mut text) in &mut texts {
        let value = match field {
            EditorField::NodeNum => format!("Node #{}", sel_id.0),
            EditorField::Status => status.to_string(),
            EditorField::Time => format!("T{time_until:+.0}s"),
            EditorField::DvTotal => {
                format!("Δv {}", crate::hud::format::delta_v_fine(total_dv, units.system))
            }
            EditorField::Burn => burn_text.clone(),
            EditorField::DeleteLabel => {
                if executed { "Dismiss".into() } else { "Delete".into() }
            }
            EditorField::AxisValue(DvAxis::Prograde) => {
                crate::hud::format::delta_v_fine(node_dv.prograde, units.system)
            }
            EditorField::AxisValue(DvAxis::Normal) => {
                crate::hud::format::delta_v_fine(node_dv.normal, units.system)
            }
            EditorField::AxisValue(DvAxis::Radial) => {
                crate::hud::format::delta_v_fine(node_dv.radial, units.system)
            }
        };
        if **text != value {
            **text = value;
        }
    }

    for (field, mut color) in &mut colors {
        let want = match field {
            EditorField::Status => status_color,
            EditorField::Time if time_until <= 0.0 && !executed => theme.text_warn,
            EditorField::Time => theme.text_primary,
            _ => continue,
        };
        if color.0 != want {
            color.0 = want;
        }
    }
}

fn set_visibility<F: bevy::ecs::query::QueryFilter>(
    query: &mut Query<&mut Visibility, F>,
    visible: bool,
) {
    let target = if visible {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in query {
        if *vis != target {
            *vis = target;
        }
    }
}

// ── Interaction ───────────────────────────────────────────────────────────────

pub(super) fn handle_buttons(
    delete_q: Query<&Interaction, (Changed<Interaction>, With<DeleteButton>)>,
    nudge_q: Query<(&Interaction, &NudgeButton), Changed<Interaction>>,
    plan: Res<ManeuverPlan>,
    mut selected: ResMut<SelectedNode>,
    mut node_dv: ResMut<NodeDeltaV>,
    mut writer: MessageWriter<ManeuverEvent>,
) {
    let Some(sel_id) = selected.id else {
        return;
    };
    let Some(node) = plan.nodes.iter().find(|n| n.id == sel_id) else {
        return;
    };
    let executed = node.phase == NodeBurnPhase::Executed;

    for interaction in &delete_q {
        if matches!(interaction, Interaction::Pressed) {
            writer.write(ManeuverEvent::DeleteNode { id: sel_id });
            selected.id = None;
            return;
        }
    }

    if executed {
        return;
    }

    let mut changed = false;
    for (interaction, nudge) in &nudge_q {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        match nudge.axis {
            DvAxis::Prograde => node_dv.prograde += nudge.delta,
            DvAxis::Normal => node_dv.normal += nudge.delta,
            DvAxis::Radial => node_dv.radial += nudge.delta,
        }
        changed = true;
    }
    if changed {
        writer.write(ManeuverEvent::AdjustNode {
            id: sel_id,
            delta_v: DVec3::new(node_dv.prograde, node_dv.normal, node_dv.radial),
        });
    }
}
