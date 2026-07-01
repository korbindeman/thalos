//! Native Bevy-UI front-end for the in-game shipyard editor.
//!
//! KSP-style layout, all styled from [`HudTheme`]:
//!
//! - **Top bar** — ship name field, build stats readout, mirror/snap/layout
//!   toggles, save / new / exit.
//! - **Left** — scrollable parts palette (by category) + saved-ship list.
//! - **Right** — parametric inspector for the selection (sliders write the
//!   part components live; the core rebuilds meshes/recomputes mass).
//! - **Right-inner** — per-stage Δv/fuel staging readout.
//! - **Bottom** — status line + pending-part hint.
//!
//! Everything drives the editor core exactly like the standalone egui
//! binary does: by reading and writing `crate::shipyard_editor::core::EditorState`.

mod inspector;
mod palette;
mod staging_panel;
mod top_bar;
pub mod widgets;

use bevy::picking::Pickable;
use bevy::prelude::*;

use crate::shipyard_editor::core::{CollectQuery, EditorPart, EditorState, collect_blueprint};
use thalos_shipyard::{
    Attachment, PartCatalog, PartParams, Ship, ShipBlueprint, ShipStats, StageSummary,
    SurfaceMount, SymmetryGroup,
};

use crate::hud::theme::{HudTheme, panel_frame};

pub use widgets::EditorTextFocus;

use super::{ShipyardEditor, editor_open};

/// Root node of the whole editor UI; visibility tracks [`ShipyardEditor`].
#[derive(Component)]
struct ShipyardUiRoot;

#[derive(Component)]
struct StatusText;

#[derive(Component)]
struct PendingHintRow;

#[derive(Component)]
struct PendingHintText;

#[derive(Component)]
struct CancelPendingButton;

/// Frame-local projection of the build into aggregate stats + staging,
/// computed once per frame while the editor is open and consumed by the top
/// bar and the staging panel.
///
/// **Sole writer:** [`refresh_stats_cache`].
#[derive(Resource, Default)]
pub struct EditorStatsCache {
    pub stats: Option<Result<ShipStats, String>>,
    pub staging: Option<Result<Vec<StageSummary>, String>>,
    /// The current build collected as a blueprint, or `None` for an empty
    /// canvas. The Launch action flies a clone of this directly — no file
    /// round-trip, so no save-timing race.
    pub blueprint: Option<ShipBlueprint>,
}

pub struct EditorUiPlugin;

impl Plugin for EditorUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EditorTextFocus>()
            .init_resource::<EditorStatsCache>()
            .add_systems(
                Startup,
                setup_editor_ui.after(crate::hud::theme::init_theme),
            )
            .add_systems(Update, sync_root_visibility)
            .add_systems(
                Update,
                (
                    refresh_stats_cache,
                    widgets::drive_sliders,
                    widgets::update_slider_visuals.after(widgets::drive_sliders),
                    widgets::scroll_scrollables,
                    widgets::style_editor_buttons,
                    widgets::focus_text_field_on_click,
                    update_status_bar,
                    handle_cancel_pending,
                )
                    .run_if(editor_open),
            )
            .add_systems(
                Update,
                (
                    top_bar::handle_actions,
                    top_bar::update_toggle_latches,
                    top_bar::update_stats_text.after(refresh_stats_cache),
                    top_bar::apply_name_input,
                    top_bar::update_name_display,
                    palette::handle_part_clicks,
                    palette::handle_saved_ship_clicks,
                    palette::rebuild_saved_ships,
                )
                    .run_if(editor_open),
            )
            .add_systems(
                Update,
                (
                    inspector::rebuild_inspector,
                    inspector::apply_param_bindings
                        .after(widgets::drive_sliders)
                        .before(inspector::refresh_sliders_from_model),
                    inspector::refresh_sliders_from_model,
                    inspector::update_info_text,
                    inspector::handle_actions,
                    staging_panel::rebuild_staging.after(refresh_stats_cache),
                )
                    .run_if(editor_open),
            );
    }
}

fn setup_editor_ui(mut commands: Commands, theme: Res<HudTheme>, catalog: Res<PartCatalog>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                top: Val::Px(0.0),
                bottom: Val::Px(0.0),
                ..default()
            },
            GlobalZIndex(50),
            Pickable {
                is_hoverable: false,
                should_block_lower: false,
            },
            Visibility::Hidden,
            ShipyardUiRoot,
            Name::new("ShipyardEditorUi"),
        ))
        .with_children(|root| {
            top_bar::spawn(root, &theme);
            palette::spawn(root, &theme, &catalog);
            inspector::spawn(root, &theme);
            staging_panel::spawn(root, &theme);
            spawn_status_bar(root, &theme);
        });
}

fn spawn_status_bar(root: &mut ChildSpawnerCommands<'_>, theme: &HudTheme) {
    let (bg, border) = panel_frame(theme);
    root.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(272.0),
            right: Val::Px(500.0),
            bottom: Val::Px(12.0),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(4.0)),
            padding: UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(12.0),
            ..default()
        },
        bg,
        border,
        Interaction::None,
        Name::new("ShipyardStatusBar"),
    ))
    .with_children(|bar| {
        bar.spawn((
            Text::new(""),
            TextFont {
                font: theme.font.clone(),
                font_size: FontSize::Px(11.0),
                ..default()
            },
            TextColor(theme.text_dim),
            Node {
                flex_grow: 1.0,
                ..default()
            },
            StatusText,
        ));
        bar.spawn((
            Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(8.0),
                ..default()
            },
            Visibility::Hidden,
            PendingHintRow,
        ))
        .with_children(|row| {
            row.spawn((
                Text::new("PENDING — click a node or surface"),
                TextFont {
                    font: theme.font.clone(),
                    font_size: FontSize::Px(11.0),
                    ..default()
                },
                TextColor(theme.text_accent),
                PendingHintText,
            ));
            widgets::spawn_button(row, theme, CancelPendingButton, "CANCEL", 10.0, 22.0);
        });
    });
}

fn sync_root_visibility(
    editor: Res<ShipyardEditor>,
    mut roots: Query<&mut Visibility, With<ShipyardUiRoot>>,
) {
    if !editor.is_changed() {
        return;
    }
    let target = if editor.open {
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

/// Collect the build into a blueprint once per frame; stats and staging are
/// both projections of it (same shape as the standalone editor's prologue).
fn refresh_stats_cache(
    state: Res<EditorState>,
    parts: CollectQuery,
    attachments: Query<(Entity, &Attachment), With<EditorPart>>,
    surface_mounts: Query<(Entity, &SurfaceMount), With<EditorPart>>,
    groups: Query<(Entity, &SymmetryGroup), With<EditorPart>>,
    ships: Query<&Ship, With<EditorPart>>,
    catalog: Res<PartCatalog>,
    mut cache: ResMut<EditorStatsCache>,
) {
    // The real ship name so the cached blueprint (which Launch flies) carries
    // it; stats/staging ignore the name.
    let name = state
        .ship_entity
        .and_then(|e| ships.get(e).ok())
        .map(|s| s.name.clone())
        .unwrap_or_default();
    let blueprint = state.ship_root.and_then(|root| {
        let ship = Ship {
            name: name.clone(),
            root,
        };
        collect_blueprint(&ship, &parts, &attachments, &surface_mounts, &groups)
    });
    cache.stats = blueprint
        .as_ref()
        .map(|bp| bp.stats(&catalog).map_err(|e| e.to_string()));
    cache.staging = blueprint
        .as_ref()
        .map(|bp| bp.stage_summaries(&catalog).map_err(|e| e.to_string()));
    cache.blueprint = blueprint;
}

fn update_status_bar(
    state: Res<EditorState>,
    catalog: Res<PartCatalog>,
    mut status: Query<&mut Text, With<StatusText>>,
    mut pending_row: Query<&mut Visibility, With<PendingHintRow>>,
    mut hint_text: Query<&mut Text, (With<PendingHintText>, Without<StatusText>)>,
) {
    if let Ok(mut text) = status.single_mut()
        && **text != state.status
    {
        **text = state.status.clone();
    }

    let pending_visible = state.pending.is_some();
    for mut vis in &mut pending_row {
        let target = if pending_visible {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != target {
            *vis = target;
        }
    }
    // Refine the hint wording for surface-mount parts.
    if let Some(pending) = &state.pending {
        let surface_hint = matches!(
            pending.params,
            PartParams::Wing { .. } | PartParams::Gear { .. }
        ) || catalog.resolve(&pending.catalog_id).is_ok_and(|entry| {
            matches!(
                entry,
                thalos_shipyard::CatalogEntry::Engine(e)
                    if e.geometry == thalos_shipyard::EngineGeometry::JetNacelle
            )
        });
        let wanted = if surface_hint {
            "PENDING — click a compatible surface"
        } else {
            "PENDING — click a glowing attach node"
        };
        for mut text in &mut hint_text {
            if **text != wanted {
                **text = wanted.to_string();
            }
        }
    }
}

fn handle_cancel_pending(
    interactions: Query<&Interaction, (Changed<Interaction>, With<CancelPendingButton>)>,
    mut state: ResMut<EditorState>,
) {
    for interaction in &interactions {
        if matches!(interaction, Interaction::Pressed) {
            state.pending = None;
        }
    }
}
