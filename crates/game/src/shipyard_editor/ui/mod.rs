//! Native Bevy-UI front-end for the in-game shipyard editor, built entirely
//! from the shared UI kit (`thalos_ui`):
//!
//! - **Top bar** — ship name field, live build stats, mirror/snap/layout
//!   toggles, HANGAR / SAVE / NEW, ▶ LAUNCH, EXIT.
//! - **Left** — scrollable parts palette (by category).
//! - **Right** — parametric inspector for the selection (sliders write the
//!   part components live; the core rebuilds meshes/recomputes mass).
//! - **Right-inner** — per-stage Δv/fuel staging readout.
//! - **Hangar** — a modal overlay listing saved craft (load / delete); see
//!   [`hangar`].
//!
//! There is no persistent status bar: core status messages
//! (`EditorState::status`) surface as transient toasts, and the
//! pending-placement state shows as a floating hint pill under the top bar.
//!
//! Everything drives the editor core by reading and writing
//! `crate::shipyard_editor::core::EditorState`.

mod hangar;
mod inspector;
mod palette;
mod staging_panel;
mod top_bar;

use bevy::picking::Pickable;
use bevy::prelude::*;

use crate::shipyard_editor::core::{CollectQuery, EditorPart, EditorState, collect_blueprint};
use thalos_shipyard::{
    Attachment, PartCatalog, PartParams, Ship, ShipBlueprint, ShipStats, StageSummary,
    SurfaceMount, SymmetryGroup,
};

use thalos_ui::{
    self as ui, ButtonVariant, ToastArea, ToastKind, UiTheme, spawn_button, spawn_toast, tokens,
};

use super::{ShipyardEditor, editor_open};

/// Root node of the whole editor UI; visibility tracks [`ShipyardEditor`].
#[derive(Component)]
struct ShipyardUiRoot;

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
        app.init_resource::<EditorStatsCache>()
            .init_resource::<hangar::HangarOpen>()
            .add_systems(
                Startup,
                (setup_editor_ui, hangar::setup).after(thalos_ui::init_ui_theme),
            )
            .add_systems(Update, (sync_root_visibility, hangar::sync_visibility))
            .add_systems(
                Update,
                (
                    refresh_stats_cache,
                    update_pending_hint,
                    surface_status_toasts,
                    handle_cancel_pending,
                    hangar::rebuild_list,
                    hangar::handle_clicks,
                )
                    .run_if(editor_open),
            )
            .add_systems(
                Update,
                (
                    top_bar::handle_actions,
                    top_bar::update_toggle_latches,
                    top_bar::update_stats_text.after(refresh_stats_cache),
                    top_bar::sync_ship_name,
                    palette::handle_part_clicks,
                )
                    .run_if(editor_open),
            )
            .add_systems(
                Update,
                (
                    inspector::rebuild_inspector,
                    inspector::apply_param_bindings
                        .after(thalos_ui::drive_sliders)
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

fn setup_editor_ui(mut commands: Commands, theme: Res<UiTheme>, catalog: Res<PartCatalog>) {
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
            spawn_pending_pill(root, &theme);
        });
}

/// A floating hint pill under the top bar, shown while a palette part is
/// armed and waiting for a placement click.
fn spawn_pending_pill(root: &mut ChildSpawnerCommands<'_>, theme: &UiTheme) {
    root.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(64.0),
            left: Val::Px(0.0),
            right: Val::Px(0.0),
            justify_content: JustifyContent::Center,
            ..default()
        },
        Pickable {
            is_hoverable: false,
            should_block_lower: false,
        },
        Visibility::Hidden,
        PendingHintRow,
        Name::new("ShipyardPendingPill"),
    ))
    .with_children(|row| {
        row.spawn((
            Node {
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(999.0)),
                padding: UiRect::axes(Val::Px(ui::SPACE_LG), Val::Px(4.0)),
                align_items: AlignItems::Center,
                column_gap: Val::Px(ui::SPACE_MD),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.025, 0.032, 0.92)),
            BorderColor::all(tokens::ACCENT.with_alpha(0.6)),
        ))
        .with_children(|pill| {
            let mut text = theme.small("");
            text.2 = TextColor(tokens::ACCENT);
            pill.spawn((text, PendingHintText));
            spawn_button(
                pill,
                theme,
                CancelPendingButton,
                "CANCEL",
                ButtonVariant::Bare,
                20.0,
            );
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

/// Show the pending pill while a part is armed, with mount-aware wording.
fn update_pending_hint(
    state: Res<EditorState>,
    catalog: Res<PartCatalog>,
    mut pending_row: Query<&mut Visibility, With<PendingHintRow>>,
    mut hint_text: Query<&mut Text, With<PendingHintText>>,
) {
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

/// Surface `EditorState::status` changes as transient toasts. Status writes
/// are event-like (saved / loaded / placed / failed), so a change edge is a
/// meaningful notification; identical repeats stay silent. The editor keeps a
/// single toast slot — a new message replaces the previous pill instead of
/// stacking.
fn surface_status_toasts(
    mut commands: Commands,
    state: Res<EditorState>,
    theme: Res<UiTheme>,
    area: Query<(Entity, Option<&Children>), With<ToastArea>>,
    mut last: Local<String>,
) {
    if !state.is_changed() || state.status == *last {
        return;
    }
    *last = state.status.clone();
    if state.status.is_empty() {
        return;
    }
    let Ok((area_entity, children)) = area.single() else {
        return;
    };
    if let Some(children) = children {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }
    let lower = state.status.to_ascii_lowercase();
    let kind = if lower.contains("failed") || lower.contains("error") {
        ToastKind::Warn
    } else if lower.starts_with("saved") || lower.starts_with("loaded") {
        ToastKind::Success
    } else {
        ToastKind::Info
    };
    spawn_toast(&mut commands, area_entity, &theme, state.status.clone(), kind);
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
