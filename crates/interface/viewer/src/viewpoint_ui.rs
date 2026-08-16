//! Canonical F8 manager and F9 quick-save UI.

use bevy::{input::mouse::AccumulatedMouseScroll, picking::prelude::Pickable, prelude::*};
use thalos_ui::{
    ButtonVariant, TextFieldFocus, TextFieldSubmit, ToastArea, ToastKind, UiTextField, UiTheme,
    spawn_button, spawn_divider, spawn_key_hint, spawn_text_field, spawn_toast, tokens,
};

use crate::{
    CurrentViewpoint, PendingViewpointApply, ViewerStatus, ViewerUiRoot, ViewpointApplyTarget,
    ViewpointFallbacks, ViewpointSet, ViewpointSnapshot, ViewpointStartupSet, ViewpointStore,
    ViewpointUiState,
};

pub(super) struct ViewpointUiPlugin;

impl Plugin for ViewpointUiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<QuickSave>()
            .add_systems(
                Startup,
                spawn_manager
                    .in_set(ViewpointStartupSet::Ui)
                    .after(thalos_ui::init_ui_theme),
            )
            .add_systems(
                Update,
                (
                    handle_shortcuts,
                    handle_manager_buttons,
                    resolve_quick_save.after(thalos_ui::apply_text_field_input),
                )
                    .chain()
                    .in_set(ViewpointSet::Input),
            )
            .add_systems(
                Update,
                (
                    sync_manager_visibility,
                    rebuild_manager_list,
                    update_manager_notice,
                    scroll_manager_list,
                )
                    .chain()
                    .in_set(ViewpointSet::Ui),
            );
    }
}

#[derive(Component)]
struct ManagerRoot;
#[derive(Component)]
struct ManagerList;
#[derive(Component)]
struct ManagerNotice;
#[derive(Component)]
struct ManagerNameField;
#[derive(Component)]
struct ManagerDescriptionField;

#[derive(Component, Clone)]
enum ManagerAction {
    Select(String),
    Reload,
    SaveNew,
    Apply,
    Replace,
    SaveMetadata,
    Delete,
    ClearSelection,
    Close,
}

#[derive(Resource, Default)]
struct QuickSave {
    pending: Option<ViewpointSnapshot>,
    root: Option<Entity>,
    field: Option<Entity>,
}

#[derive(Component)]
struct QuickSaveField;

fn spawn_manager(mut commands: Commands, theme: Res<UiTheme>) {
    commands
        .spawn((
            Node {
                display: Display::None,
                position_type: PositionType::Absolute,
                top: Val::Px(18.0),
                right: Val::Px(18.0),
                width: Val::Px(560.0),
                max_height: Val::Percent(88.0),
                padding: UiRect::axes(Val::Px(tokens::SPACE_LG), Val::Px(tokens::SPACE_MD)),
                row_gap: Val::Px(tokens::SPACE_SM),
                flex_direction: FlexDirection::Column,
                border_radius: BorderRadius::all(Val::Px(tokens::RADIUS_PANEL)),
                ..default()
            },
            theme.glass_heavy(),
            Interaction::None,
            Visibility::Inherited,
            GlobalZIndex(180),
            ViewerUiRoot,
            ManagerRoot,
            Name::new("ViewpointManager"),
        ))
        .with_children(|panel| {
            panel
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|header| {
                    header.spawn(theme.heading("VIEWPOINTS"));
                    spawn_button(
                        header,
                        &theme,
                        ManagerAction::Close,
                        "F8 CLOSE",
                        ButtonVariant::Bare,
                        tokens::CTRL_H,
                    );
                });
            panel.spawn(theme.faint("One catalog · exact pose + physical lens · F9 quick-save"));
            spawn_divider(panel);

            panel.spawn((
                ManagerList,
                Node {
                    width: Val::Percent(100.0),
                    max_height: Val::Px(330.0),
                    row_gap: Val::Px(tokens::SPACE_XS),
                    flex_direction: FlexDirection::Column,
                    overflow: Overflow::scroll_y(),
                    ..default()
                },
                ScrollPosition::default(),
            ));

            spawn_divider(panel);
            panel.spawn(theme.small("NAME"));
            spawn_text_field(
                panel,
                &theme,
                UiTextField::new("New viewpoint", "viewpoint name"),
                Val::Percent(100.0),
                ManagerNameField,
            );
            panel.spawn(theme.small("NOTES"));
            let mut description = UiTextField::new("", "optional description");
            description.max_len = 160;
            spawn_text_field(
                panel,
                &theme,
                description,
                Val::Percent(100.0),
                ManagerDescriptionField,
            );

            panel
                .spawn(Node {
                    width: Val::Percent(100.0),
                    column_gap: Val::Px(tokens::SPACE_SM),
                    flex_wrap: FlexWrap::Wrap,
                    ..default()
                })
                .with_children(|actions| {
                    for (action, label, variant) in [
                        (ManagerAction::SaveNew, "SAVE NEW", ButtonVariant::Primary),
                        (ManagerAction::Apply, "VIEW", ButtonVariant::Ghost),
                        (ManagerAction::Replace, "REPLACE", ButtonVariant::Ghost),
                        (
                            ManagerAction::SaveMetadata,
                            "SAVE DETAILS",
                            ButtonVariant::Ghost,
                        ),
                        (ManagerAction::ClearSelection, "NEW", ButtonVariant::Bare),
                        (ManagerAction::Reload, "RELOAD", ButtonVariant::Bare),
                        (ManagerAction::Delete, "DELETE", ButtonVariant::Danger),
                    ] {
                        spawn_button(actions, &theme, action, label, variant, tokens::CTRL_H);
                    }
                });
            panel.spawn((
                theme.faint("Select a row, then view, replace, edit, or delete it."),
                ManagerNotice,
            ));
        });
}

#[allow(clippy::too_many_arguments)]
fn handle_shortcuts(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    viewer: Res<ViewerStatus>,
    theme: Res<UiTheme>,
    current: Res<CurrentViewpoint>,
    fallbacks: Res<ViewpointFallbacks>,
    mut store: ResMut<ViewpointStore>,
    mut state: ResMut<ViewpointUiState>,
    mut quick: ResMut<QuickSave>,
    mut focus: ResMut<TextFieldFocus>,
    toast_area: Query<Entity, With<ToastArea>>,
) {
    if viewer.interaction_blocked {
        state.manager_open = false;
        close_quick_save(&mut commands, &mut quick, &mut state, &mut focus);
        return;
    }
    if focus.is_focused() {
        return;
    }
    if keys.just_pressed(KeyCode::F8) && !state.quick_open {
        state.manager_open = !state.manager_open;
        if state.manager_open {
            state.report(
                store
                    .reload(&fallbacks.0)
                    .map(|_| "Reloaded viewpoints".into()),
            );
        }
    }
    if keys.just_pressed(KeyCode::Escape) && state.manager_open {
        state.manager_open = false;
    }
    if keys.just_pressed(KeyCode::F9) && !state.is_open() {
        let Some(snapshot) = current.0.clone() else {
            toast(
                &mut commands,
                &toast_area,
                &theme,
                "CAN'T SAVE VIEWPOINT · camera pose unavailable",
                ToastKind::Warn,
            );
            return;
        };
        let suggestion = available_name(store.catalog(), &snapshot.suggested_name);
        open_quick_save(
            &mut commands,
            &theme,
            &mut quick,
            &mut state,
            &mut focus,
            snapshot,
            suggestion,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_manager_buttons(
    interactions: Query<(&Interaction, &ManagerAction), Changed<Interaction>>,
    fallbacks: Res<ViewpointFallbacks>,
    current: Res<CurrentViewpoint>,
    mut store: ResMut<ViewpointStore>,
    mut state: ResMut<ViewpointUiState>,
    mut pending: ResMut<PendingViewpointApply>,
    mut name_field: Single<&mut UiTextField, With<ManagerNameField>>,
    mut description_field: Single<
        &mut UiTextField,
        (With<ManagerDescriptionField>, Without<ManagerNameField>),
    >,
) {
    let Some(action) = interactions
        .iter()
        .find(|(interaction, _)| **interaction == Interaction::Pressed)
        .map(|(_, action)| action.clone())
    else {
        return;
    };

    match action {
        ManagerAction::Select(id) => {
            state.selected = Some(id.clone());
            if let Some(viewpoint) = store.catalog().find(&id) {
                name_field.value = viewpoint.name.clone();
                description_field.value = viewpoint.description.clone();
            } else if let Some(viewpoint) = store.catalog().find_scripted(&id) {
                name_field.value = viewpoint.name.clone();
                description_field.value = viewpoint.description.clone();
            }
            state.status = None;
        }
        ManagerAction::Reload => {
            state.report(
                store
                    .reload(&fallbacks.0)
                    .map(|_| "Reloaded viewpoints".into()),
            );
            if state
                .selected
                .as_deref()
                .is_some_and(|id| !store.catalog().contains(id))
            {
                clear_selection(&mut state, &mut name_field, &mut description_field);
            }
        }
        ManagerAction::SaveNew => {
            let result = current
                .0
                .as_ref()
                .ok_or_else(|| "camera pose unavailable".to_owned())
                .and_then(|snapshot| {
                    store.append_snapshot(snapshot, &name_field.value, &description_field.value)
                });
            if let Ok(id) = &result {
                state.selected = Some(id.clone());
            }
            state.report(result.map(|id| format!("Saved {id}")));
        }
        ManagerAction::Apply => {
            let result = state
                .selected
                .as_deref()
                .ok_or_else(|| "select a viewpoint first".to_owned())
                .and_then(|id| {
                    if let Some(viewpoint) = store.catalog().find(id) {
                        pending.0 = Some(ViewpointApplyTarget::Saved(viewpoint.clone()));
                    } else if let Some(viewpoint) = store.catalog().find_scripted(id) {
                        pending.0 = Some(ViewpointApplyTarget::Scripted(viewpoint.clone()));
                    } else {
                        return Err(format!("viewpoint {id:?} no longer exists"));
                    }
                    Ok(format!("Applying {id}"))
                });
            state.report(result);
        }
        ManagerAction::Replace => {
            let result = state
                .selected
                .clone()
                .ok_or_else(|| "select a viewpoint first".to_owned())
                .and_then(|id| {
                    let snapshot = current
                        .0
                        .as_ref()
                        .ok_or_else(|| "camera pose unavailable".to_owned())?;
                    store.replace_from_snapshot(
                        &id,
                        snapshot,
                        &name_field.value,
                        &description_field.value,
                    )
                });
            state.report(result.map(|id| format!("Replaced {id}")));
        }
        ManagerAction::SaveMetadata => {
            let result = state
                .selected
                .clone()
                .ok_or_else(|| "select a viewpoint first".to_owned())
                .and_then(|id| {
                    store
                        .update_metadata(&id, &name_field.value, &description_field.value)
                        .map(|()| id)
                });
            state.report(result.map(|id| format!("Saved details for {id}")));
        }
        ManagerAction::Delete => {
            let result = state
                .selected
                .clone()
                .ok_or_else(|| "select a viewpoint first".to_owned())
                .and_then(|id| store.delete(&id).map(|()| id));
            if result.is_ok() {
                clear_selection(&mut state, &mut name_field, &mut description_field);
            }
            state.report(result.map(|id| format!("Deleted {id}")));
        }
        ManagerAction::ClearSelection => {
            clear_selection(&mut state, &mut name_field, &mut description_field);
        }
        ManagerAction::Close => state.manager_open = false,
    }
}

fn clear_selection(
    state: &mut ViewpointUiState,
    name: &mut UiTextField,
    description: &mut UiTextField,
) {
    state.selected = None;
    name.value = "New viewpoint".into();
    description.value.clear();
}

fn sync_manager_visibility(
    state: Res<ViewpointUiState>,
    mut root: Single<&mut Node, With<ManagerRoot>>,
) {
    let display = if state.manager_open {
        Display::Flex
    } else {
        Display::None
    };
    if root.display != display {
        root.display = display;
    }
}

fn rebuild_manager_list(
    mut commands: Commands,
    theme: Res<UiTheme>,
    store: Res<ViewpointStore>,
    state: Res<ViewpointUiState>,
    list: Single<Entity, With<ManagerList>>,
    mut rendered: Local<Option<(u64, Option<String>)>>,
) {
    let key = (store.revision(), state.selected.clone());
    if rendered.as_ref() == Some(&key) {
        return;
    }
    *rendered = Some(key);
    commands
        .entity(*list)
        .despawn_children()
        .with_children(|list| {
            if store.catalog().viewpoints.is_empty()
                && store.catalog().scripted_viewpoints.is_empty()
            {
                list.spawn(theme.faint("No viewpoints yet. Press F9 to save one."));
                return;
            }
            for (id, name, kind) in store
                .catalog()
                .viewpoints
                .iter()
                .map(|viewpoint| (&viewpoint.id, &viewpoint.name, viewpoint.frame.label()))
                .chain(
                    store
                        .catalog()
                        .scripted_viewpoints
                        .iter()
                        .map(|viewpoint| (&viewpoint.id, &viewpoint.name, "scripted")),
                )
            {
                let selected = state.selected.as_deref() == Some(id.as_str());
                let label = if selected {
                    format!("› {name}")
                } else {
                    name.clone()
                };
                spawn_button(
                    list,
                    &theme,
                    ManagerAction::Select(id.clone()),
                    &format!("{label}  ·  {kind}"),
                    ButtonVariant::Bare,
                    tokens::CTRL_H,
                );
            }
        });
}

fn update_manager_notice(
    state: Res<ViewpointUiState>,
    store: Res<ViewpointStore>,
    mut notice: Single<&mut Text, With<ManagerNotice>>,
) {
    if !state.is_changed() && !store.is_changed() {
        return;
    }
    let next = if let Some((_, message)) = &state.status {
        message.clone()
    } else if let Some(error) = store.load_error() {
        error.to_owned()
    } else {
        format!(
            "{} saved · {} scripted · {}",
            store.catalog().viewpoints.len(),
            store.catalog().scripted_viewpoints.len(),
            store.path().display()
        )
    };
    if notice.0 != next {
        notice.0 = next;
    }
}

fn scroll_manager_list(
    scroll: Res<AccumulatedMouseScroll>,
    state: Res<ViewpointUiState>,
    mut list: Single<(&mut ScrollPosition, &ComputedNode), With<ManagerList>>,
) {
    if !state.manager_open || scroll.delta.y == 0.0 {
        return;
    }
    let max_offset = (list.1.content_size().y - list.1.size().y) * list.1.inverse_scale_factor();
    list.0.y = (list.0.y - scroll.delta.y * 28.0).clamp(0.0, max_offset.max(0.0));
}

fn open_quick_save(
    commands: &mut Commands,
    theme: &UiTheme,
    quick: &mut QuickSave,
    state: &mut ViewpointUiState,
    focus: &mut TextFieldFocus,
    snapshot: ViewpointSnapshot,
    suggestion: String,
) {
    let mut field = Entity::PLACEHOLDER;
    let root = commands
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
            Pickable::IGNORE,
            Visibility::Inherited,
            GlobalZIndex(200),
            ViewerUiRoot,
            Name::new("QuickSaveViewpointPrompt"),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(380.0),
                    align_items: AlignItems::Stretch,
                    ..thalos_ui::panel_node()
                },
                theme.glass_heavy(),
            ))
            .with_children(|panel| {
                panel.spawn(theme.heading("SAVE VIEWPOINT"));
                panel.spawn(theme.faint(format!(
                    "{} · {:.0} mm · {}:{} sensor",
                    snapshot.frame.label(),
                    snapshot.optics.lens.focal_length_mm,
                    snapshot.optics.sensor.aspect[0],
                    snapshot.optics.sensor.aspect[1]
                )));
                field = spawn_text_field(
                    panel,
                    theme,
                    UiTextField::new(suggestion, "viewpoint name").selected(),
                    Val::Percent(100.0),
                    QuickSaveField,
                );
                panel
                    .spawn(Node {
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(tokens::SPACE_SM),
                        ..default()
                    })
                    .with_children(|hints| {
                        spawn_key_hint(hints, theme, "Enter");
                        hints.spawn(theme.faint("save"));
                        spawn_key_hint(hints, theme, "Esc");
                        hints.spawn(theme.faint("cancel"));
                    });
            });
        })
        .id();
    quick.pending = Some(snapshot);
    quick.root = Some(root);
    quick.field = Some(field);
    state.quick_open = true;
    focus.field = Some(field);
}

fn resolve_quick_save(
    mut commands: Commands,
    mut submits: MessageReader<TextFieldSubmit>,
    mut quick: ResMut<QuickSave>,
    mut state: ResMut<ViewpointUiState>,
    mut store: ResMut<ViewpointStore>,
    mut focus: ResMut<TextFieldFocus>,
    fields: Query<&UiTextField, With<QuickSaveField>>,
    theme: Res<UiTheme>,
    toast_area: Query<Entity, With<ToastArea>>,
) {
    let (Some(field), true) = (quick.field, state.quick_open) else {
        submits.clear();
        return;
    };
    let Some(accepted) = submits
        .read()
        .find(|submit| submit.field == field)
        .map(|submit| submit.accepted)
    else {
        if focus.field != Some(field) && fields.contains(field) {
            focus.field = Some(field);
        }
        return;
    };
    let typed = fields
        .get(field)
        .map(|field| field.value.clone())
        .unwrap_or_default();
    let snapshot = quick.pending.clone();
    close_quick_save(&mut commands, &mut quick, &mut state, &mut focus);
    if !accepted {
        return;
    }
    let result = snapshot
        .as_ref()
        .ok_or_else(|| "camera pose unavailable".to_owned())
        .and_then(|snapshot| store.append_snapshot(snapshot, &typed, ""));
    let (message, kind) = match result {
        Ok(id) => (format!("SAVED VIEWPOINT · {id}"), ToastKind::Success),
        Err(error) => (format!("VIEWPOINT NOT SAVED · {error}"), ToastKind::Warn),
    };
    toast(&mut commands, &toast_area, &theme, message, kind);
}

fn close_quick_save(
    commands: &mut Commands,
    quick: &mut QuickSave,
    state: &mut ViewpointUiState,
    focus: &mut TextFieldFocus,
) {
    if let Some(root) = quick.root.take() {
        commands.entity(root).despawn();
    }
    if quick.field.is_some_and(|field| focus.field == Some(field)) {
        focus.field = None;
    }
    quick.pending = None;
    quick.field = None;
    state.quick_open = false;
}

fn available_name(catalog: &crate::ViewpointCatalog, base: &str) -> String {
    let base = if base.trim().is_empty() {
        "Viewpoint"
    } else {
        base.trim()
    };
    if crate::unique_id(catalog, &crate::viewpoint_id_from_name(base))
        == crate::viewpoint_id_from_name(base)
    {
        return base.to_owned();
    }
    (2..1000)
        .map(|number| format!("{base} {number}"))
        .find(|candidate| {
            crate::unique_id(catalog, &crate::viewpoint_id_from_name(candidate))
                == crate::viewpoint_id_from_name(candidate)
        })
        .unwrap_or_else(|| base.to_owned())
}

fn toast(
    commands: &mut Commands,
    area: &Query<Entity, With<ToastArea>>,
    theme: &UiTheme,
    message: impl Into<String>,
    kind: ToastKind,
) {
    if let Ok(area) = area.single() {
        spawn_toast(commands, area, theme, message, kind);
    }
}
