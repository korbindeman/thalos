//! In-world HUD for the base editor: a building-picker **palette** plus a
//! status/help hint, styled from [`HudTheme`].
//!
//! The palette is the "nice picker": a **Select / Move** tool (the default —
//! click structures to move or delete, never place) and a row of building
//! presets + a launchpad that *arm* the place tool. Clicking a structure with
//! the select tool picks it (drag to move, X to delete); right-click cancels
//! placement back to select. Footprint sizing is still keyboard-driven (`[ ]`,
//! `- =`), surfaced in the hint.

use bevy::prelude::*;
use thalos_ui::{
    self as ui, ButtonVariant, CTRL_H, SPACE_SM, UiButton, UiTheme, spawn_button,
};

use super::place::{BuildingDims, PendingKind, Tool};
use super::{BaseBuildState, BaseEditor, BaseEditorMode};

/// A palette item.
struct Preset {
    label: &'static str,
    kind: PendingKind,
    dims: BuildingDims,
    radius_m: f32,
}

const PRESETS: &[Preset] = &[
    Preset {
        label: "Habitat",
        kind: PendingKind::Building,
        dims: BuildingDims { half_x_m: 6.0, half_z_m: 6.0, height_m: 8.0 },
        radius_m: 0.0,
    },
    Preset {
        label: "Depot",
        kind: PendingKind::Building,
        dims: BuildingDims { half_x_m: 9.0, half_z_m: 6.0, height_m: 5.0 },
        radius_m: 0.0,
    },
    Preset {
        label: "Tower",
        kind: PendingKind::Building,
        dims: BuildingDims { half_x_m: 4.0, half_z_m: 4.0, height_m: 20.0 },
        radius_m: 0.0,
    },
    Preset {
        label: "Hangar",
        kind: PendingKind::Building,
        dims: BuildingDims { half_x_m: 11.0, half_z_m: 8.0, height_m: 9.0 },
        radius_m: 0.0,
    },
    Preset {
        label: "Launchpad",
        kind: PendingKind::Launchpad,
        dims: BuildingDims { half_x_m: 0.0, half_z_m: 0.0, height_m: 0.0 },
        radius_m: 18.0,
    },
];

#[derive(Component)]
struct BaseEditorOverlay;

#[derive(Component)]
struct BaseEditorText;

/// The overlay's heading, retitled in launch-select mode.
#[derive(Component)]
struct BaseEditorTitle;

#[derive(Component, Clone, Copy)]
enum PaletteAction {
    Select,
    Place(usize),
}

pub(super) struct BaseEditorUiPlugin;

impl Plugin for BaseEditorUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_overlay.after(thalos_ui::init_ui_theme))
            .add_systems(
                Update,
                (
                    toggle_overlay,
                    sync_overlay_for_mode,
                    handle_palette_clicks,
                    latch_active_tool,
                    update_overlay_text,
                ),
            );
    }
}

fn setup_overlay(mut commands: Commands, theme: Res<UiTheme>) {
    commands
        .spawn((
            Node {
                left: Val::Px(16.0),
                top: Val::Px(16.0),
                width: Val::Px(230.0),
                row_gap: Val::Px(SPACE_SM - 3.0),
                ..ui::floating_panel_node()
            },
            theme.glass(),
            GlobalZIndex(60),
            Visibility::Hidden,
            BaseEditorOverlay,
            Name::new("BaseEditorOverlay"),
        ))
        .with_children(|panel| {
            panel.spawn((
                theme.title("SURFACE BASE"),
                Node {
                    margin: UiRect::bottom(Val::Px(4.0)),
                    ..default()
                },
                BaseEditorTitle,
            ));
            spawn_palette_button(panel, &theme, PaletteAction::Select, "Select / Move");
            for (i, preset) in PRESETS.iter().enumerate() {
                spawn_palette_button(panel, &theme, PaletteAction::Place(i), preset.label);
            }
            panel.spawn((
                theme.small(""),
                Node {
                    margin: UiRect::top(Val::Px(SPACE_SM)),
                    ..default()
                },
                BaseEditorText,
            ));
        });
}

fn spawn_palette_button(
    parent: &mut ChildSpawnerCommands<'_>,
    theme: &UiTheme,
    action: PaletteAction,
    label: &str,
) {
    let button = spawn_button(parent, theme, action, label, ButtonVariant::Ghost, CTRL_H);
    // Left-aligned full-width rows read better in a tool palette.
    parent
        .commands_mut()
        .entity(button)
        .entry::<Node>()
        .and_modify(|mut node| {
            node.width = Val::Percent(100.0);
            node.justify_content = JustifyContent::FlexStart;
        });
}

/// Apply palette clicks to the build state.
fn handle_palette_clicks(
    interactions: Query<(&Interaction, &PaletteAction), Changed<Interaction>>,
    mut build: ResMut<BaseBuildState>,
) {
    for (interaction, action) in &interactions {
        if !matches!(interaction, Interaction::Pressed) {
            continue;
        }
        match action {
            PaletteAction::Select => build.tool = Tool::Select,
            PaletteAction::Place(i) => {
                let preset = &PRESETS[*i];
                build.tool = Tool::Place;
                build.pending_kind = preset.kind;
                match preset.kind {
                    PendingKind::Building => build.pending = preset.dims,
                    PendingKind::Launchpad => build.pending_radius_m = preset.radius_m,
                }
            }
        }
    }
}

/// Latch the active tool onto the shared button styling ([`UiButton`]).
fn latch_active_tool(
    build: Res<BaseBuildState>,
    mut buttons: Query<(&PaletteAction, &mut UiButton)>,
) {
    for (action, mut button) in &mut buttons {
        let active = match action {
            PaletteAction::Select => build.tool == Tool::Select,
            PaletteAction::Place(i) => {
                let p = &PRESETS[*i];
                build.tool == Tool::Place && build.pending_kind == p.kind
            }
        };
        if button.latched != active {
            button.latched = active;
        }
    }
}

/// In launch-select mode the overlay is a launch picker, not a building editor:
/// retitle it and collapse the building palette (the presets + Select/Move
/// buttons) so only the launch hint remains.
fn sync_overlay_for_mode(
    editor: Res<BaseEditor>,
    mut titles: Query<&mut Text, With<BaseEditorTitle>>,
    mut buttons: Query<&mut Node, With<PaletteAction>>,
) {
    if !editor.is_changed() {
        return;
    }
    let select_launch = editor.mode == BaseEditorMode::SelectLaunch;
    let title = if select_launch {
        "SELECT LAUNCH POINT"
    } else {
        "SURFACE BASE EDITOR"
    };
    for mut t in &mut titles {
        if t.0 != title {
            t.0 = title.to_string();
        }
    }
    let display = if select_launch {
        Display::None
    } else {
        Display::Flex
    };
    for mut node in &mut buttons {
        if node.display != display {
            node.display = display;
        }
    }
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
        BaseEditorMode::SelectLaunch => {
            "CHOOSE A LAUNCH POINT\nLMB  launch from the highlighted runway or pad\nWASD pan · scroll zoom · Esc  cancel"
                .to_string()
        }
        BaseEditorMode::PickSite => {
            "PICK A SITE\nLMB  confirm (flattens the land)\nQ/E  rotate   WASD  pan   Esc  exit"
                .to_string()
        }
        BaseEditorMode::PlaceBuildings => match build.tool {
            Tool::Select => "SELECT / MOVE\nLMB  pick · drag to move\nX  delete   L  launch ship onto pad\nWASD pan · scroll zoom · Esc exit\n\nPick an item above to place."
                .to_string(),
            Tool::Place => {
                let size = match build.pending_kind {
                    PendingKind::Building => format!(
                        "{:.0} x {:.0} m, {:.0} tall   [ ] /-= size",
                        build.pending.half_x_m * 2.0,
                        build.pending.half_z_m * 2.0,
                        build.pending.height_m,
                    ),
                    PendingKind::Launchpad => {
                        format!("{:.0} m across   [ ] size", build.pending_radius_m * 2.0)
                    }
                };
                format!(
                    "PLACE\nLMB  place · RMB  cancel\nQ/E  rotate · Tab  kind\nWASD pan · scroll zoom\n{size}",
                )
            }
        },
    };
    for mut t in &mut text {
        if t.0 != body {
            t.0 = body.clone();
        }
    }
}
