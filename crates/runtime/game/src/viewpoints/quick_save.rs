//! **F9 — save the current view as a viewpoint, in one keypress.**
//!
//! The F8 manager is the full catalog surface (browse, re-view, replace,
//! delete). This is the capture-side shortcut for the loop that actually
//! happens during play: *see something worth showing an agent → save it →
//! keep flying.* Two keys is the whole interaction — **F9, Enter** takes a
//! name derived from the view, **F9, type, Enter** overrides it, because the
//! suggestion starts fully selected.
//!
//! The pose is frozen the instant F9 is pressed, not when Enter commits: the
//! world keeps moving (and under warp, moving fast) while a name is typed, and
//! the viewpoint the user meant is the one that was on screen.
//!
//! Both entry points share [`super::capture_current_viewpoint`] and
//! [`super::write_catalog`]; this path parameterizes that core rather than
//! forking it.

use bevy::picking::prelude::Pickable;
use bevy::prelude::*;
use big_space::prelude::CellCoord;
use thalos_capture_protocol::{
    CAPTURE_PRESETS, Viewpoint, ViewpointCatalog, viewpoint_id_from_name,
};
use thalos_input::game::GameInputIntent;
use thalos_ui::{
    self as ui, SPACE_SM, TextFieldFocus, TextFieldSubmit, ToastArea, ToastKind, UiTextField,
    UiTheme, spawn_key_hint, spawn_text_field, spawn_toast,
};

use crate::camera::{ActiveCamera, ShipCamera};
use crate::camera_optics::CameraOptics;
use crate::rendering::{SimulationState, SolarSystemState, view_anchor::ViewAnchor};
use crate::spawn::SpawnSituation;
use crate::viewpoints::{ViewpointManager, capture_current_viewpoint, load_catalog, write_catalog};

/// Longest id we build before appending a `-<n>` uniquifier, leaving room
/// inside the catalog's 64-byte id limit.
const ID_STEM_MAX: usize = 58;

pub struct QuickSaveViewpointPlugin;

impl Plugin for QuickSaveViewpointPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<QuickSaveViewpoint>().add_systems(
            Update,
            (
                open_quick_save_prompt.run_if(in_state(crate::loading::AppState::Running)),
                // After the shared field has consumed this frame's keys, so a
                // commit is handled the frame Enter lands.
                resolve_quick_save_prompt.after(thalos_ui::apply_text_field_input),
            )
                .chain(),
        );
    }
}

/// The open quick-save prompt, if any.
///
/// **Sole writer:** this module. `pending` doubles as the open flag — a prompt
/// without a frozen pose would have nothing to save.
#[derive(Resource, Default)]
pub struct QuickSaveViewpoint {
    /// The viewpoint captured when F9 was pressed; `name`/`id` are replaced
    /// with the typed name on commit.
    pending: Option<Viewpoint>,
    root: Option<Entity>,
    field: Option<Entity>,
}

impl QuickSaveViewpoint {
    pub fn open(&self) -> bool {
        self.pending.is_some()
    }
}

#[derive(Component)]
struct QuickSaveRoot;

#[derive(Component)]
struct QuickSaveField;

#[allow(clippy::too_many_arguments)]
fn open_quick_save_prompt(
    mut commands: Commands,
    intent: Res<GameInputIntent>,
    mut quick: ResMut<QuickSaveViewpoint>,
    manager: Option<Res<ViewpointManager>>,
    theme: Res<UiTheme>,
    view_anchor: Res<ViewAnchor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    situation: Res<SpawnSituation>,
    space_center: Option<Res<crate::space_center::SpaceCenter>>,
    cameras: Query<(&CellCoord, &Transform, &CameraOptics), (With<ShipCamera>, With<ActiveCamera>)>,
    toast_area: Query<Entity, With<ToastArea>>,
) {
    if !intent.quick_save_viewpoint || quick.open() {
        return;
    }
    // The F8 manager owns viewpoint editing while it is up; two prompts
    // writing the same catalog is a race with no upside.
    if manager.is_some_and(|manager| manager.open) {
        return;
    }

    let captured = capture_current_viewpoint(
        String::new(),
        String::new(),
        String::new(),
        &view_anchor,
        &sim,
        &solar,
        *situation,
        space_center.as_deref(),
        &cameras,
    );
    let mut viewpoint = match captured {
        Ok(viewpoint) => viewpoint,
        Err(error) => {
            toast(
                &mut commands,
                &toast_area,
                &theme,
                format!("CAN'T SAVE VIEWPOINT · {error}"),
                ToastKind::Warn,
            );
            return;
        }
    };

    let catalog = load_catalog().unwrap_or_default();
    let agl_m = view_anchor.resolved.map(|anchor| anchor.agl_m);
    viewpoint.name = suggested_name(&catalog, &viewpoint.body, agl_m);

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
            // A transient prompt, not a modal: it must not swallow clicks on
            // the world behind it.
            Pickable::IGNORE,
            GlobalZIndex(200),
            QuickSaveRoot,
            Name::new("QuickSaveViewpointPrompt"),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: Val::Px(360.0),
                    align_items: AlignItems::Stretch,
                    ..ui::panel_node()
                },
                theme.glass_heavy(),
                Name::new("QuickSaveViewpointPanel"),
            ))
            .with_children(|panel| {
                panel
                    .spawn(Node {
                        width: Val::Percent(100.0),
                        justify_content: JustifyContent::SpaceBetween,
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(SPACE_SM),
                        ..default()
                    })
                    .with_children(|header| {
                        header.spawn(theme.heading("SAVE VIEWPOINT"));
                        header.spawn(theme.faint(format!(
                            "{} · {:.0} mm · {}:{}",
                            viewpoint.body,
                            viewpoint.optics.lens.focal_length_mm,
                            viewpoint.optics.sensor.aspect[0],
                            viewpoint.optics.sensor.aspect[1],
                        )));
                    });
                field = spawn_text_field(
                    panel,
                    &theme,
                    UiTextField::new(viewpoint.name.clone(), "viewpoint name").selected(),
                    Val::Percent(100.0),
                    QuickSaveField,
                );
                panel
                    .spawn(Node {
                        width: Val::Percent(100.0),
                        align_items: AlignItems::Center,
                        column_gap: Val::Px(SPACE_SM),
                        ..default()
                    })
                    .with_children(|hints| {
                        spawn_key_hint(hints, &theme, "Enter");
                        hints.spawn(theme.faint("save"));
                        spawn_key_hint(hints, &theme, "Esc");
                        hints.spawn(theme.faint("cancel"));
                    });
            });
        })
        .id();

    quick.pending = Some(viewpoint);
    quick.root = Some(root);
    quick.field = Some(field);
}

fn resolve_quick_save_prompt(
    mut commands: Commands,
    mut quick: ResMut<QuickSaveViewpoint>,
    mut submits: MessageReader<TextFieldSubmit>,
    mut focus: ResMut<TextFieldFocus>,
    fields: Query<&UiTextField, With<QuickSaveField>>,
    theme: Res<UiTheme>,
    toast_area: Query<Entity, With<ToastArea>>,
) {
    let (Some(field), true) = (quick.field, quick.open()) else {
        submits.clear();
        return;
    };

    let outcome = submits
        .read()
        .find(|submit| submit.field == field)
        .map(|submit| submit.accepted);
    let Some(accepted) = outcome else {
        // Still open. Hold the keyboard on our field so the prompt can always
        // be finished with Enter/Escape, even after a stray click elsewhere —
        // and so the first frame (spawned after the shared input system ran)
        // picks up focus.
        if focus.field != Some(field) && fields.contains(field) {
            focus.field = Some(field);
        }
        return;
    };

    let typed = fields
        .get(field)
        .map(|field| field.value.clone())
        .unwrap_or_default();
    let pending = quick.pending.take();
    if let Some(root) = quick.root.take() {
        commands.entity(root).despawn();
    }
    quick.field = None;
    if focus.field == Some(field) {
        focus.field = None;
    }

    if !accepted {
        return;
    }
    let Some(pending) = pending else {
        return;
    };
    let (message, kind) = match commit(pending, &typed) {
        Ok(id) => (format!("SAVED VIEWPOINT · {id}"), ToastKind::Success),
        Err(error) => (format!("VIEWPOINT NOT SAVED · {error}"), ToastKind::Warn),
    };
    toast(&mut commands, &toast_area, &theme, message, kind);
}

/// Name, slug, and append the captured viewpoint. Returns the id it landed
/// under — which the toast reports, since that is what
/// `just screenshot <id>` needs.
fn commit(mut viewpoint: Viewpoint, typed_name: &str) -> Result<String, String> {
    let typed_name = typed_name.trim();
    if !typed_name.is_empty() {
        viewpoint.name = typed_name.to_owned();
    }
    if viewpoint.name.is_empty() {
        return Err("give the viewpoint a name".to_owned());
    }
    let mut catalog = match load_catalog() {
        Ok(catalog) => catalog,
        // The first save on a checkout without the file should just work. A
        // file that *does* exist but won't parse is a different story: report
        // it rather than overwriting whatever is in there.
        Err(_) if !super::catalog_path().exists() => ViewpointCatalog::default(),
        Err(error) => return Err(error),
    };
    viewpoint.id = unique_id(&catalog, &viewpoint_id_from_name(&viewpoint.name));
    viewpoint.validate()?;
    let id = viewpoint.id.clone();
    catalog.viewpoints.push(viewpoint);
    write_catalog(&catalog)?;
    Ok(id)
}

/// A name derived from what is actually on screen — the body and the camera's
/// height over it — rather than from the scenario the session booted in, which
/// stops describing the view the moment the player flies somewhere else.
fn suggested_name(catalog: &ViewpointCatalog, body: &str, agl_m: Option<f64>) -> String {
    let base = match agl_m {
        Some(agl) if agl.is_finite() => {
            let agl = agl.max(0.0);
            if agl < 1000.0 {
                format!("{body} {} m", (agl / 10.0).round() as i64 * 10)
            } else {
                format!("{body} {} km", (agl / 1000.0).round() as i64)
            }
        }
        _ => body.to_owned(),
    };
    if id_free(catalog, &viewpoint_id_from_name(&base)) {
        return base;
    }
    // Numbered like a file manager would: the second view from the same place
    // is "… 2", not a silently reused name.
    (2..1000)
        .map(|n| format!("{base} {n}"))
        .find(|candidate| id_free(catalog, &viewpoint_id_from_name(candidate)))
        .unwrap_or(base)
}

/// The first free `<stem>`, `<stem>-2`, `<stem>-3`… — a typed name never fails
/// to save just because its slug collides.
fn unique_id(catalog: &ViewpointCatalog, stem: &str) -> String {
    let mut stem = if stem.is_empty() { "viewpoint" } else { stem };
    if stem.len() > ID_STEM_MAX {
        stem = stem[..ID_STEM_MAX].trim_end_matches('-');
    }
    if id_free(catalog, stem) {
        return stem.to_owned();
    }
    (2..1000)
        .map(|n| format!("{stem}-{n}"))
        .find(|candidate| id_free(catalog, candidate))
        .unwrap_or_else(|| stem.to_owned())
}

/// Free means "not in the catalog *and* not a name the capture interface has
/// already spoken for" — [`ViewpointCatalog::validate`] rejects the latter.
fn id_free(catalog: &ViewpointCatalog, id: &str) -> bool {
    !id.is_empty()
        && !catalog.contains(id)
        && !CAPTURE_PRESETS.contains(&id)
        && !matches!(id, "latest" | "perspective" | "latest-perspective")
}

fn toast(
    commands: &mut Commands,
    toast_area: &Query<Entity, With<ToastArea>>,
    theme: &UiTheme,
    message: String,
    kind: ToastKind,
) {
    if let Ok(area) = toast_area.single() {
        spawn_toast(commands, area, theme, message, kind);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_capture_protocol::ViewpointSpawn;

    fn catalog_with(ids: &[&str]) -> ViewpointCatalog {
        let mut catalog = ViewpointCatalog::default();
        for id in ids {
            catalog.viewpoints.push(Viewpoint {
                id: (*id).to_owned(),
                name: (*id).to_owned(),
                description: String::new(),
                saved_unix_ms: 0,
                body: "Thalos".to_owned(),
                spawn: ViewpointSpawn::Orbit,
                boots_hub: false,
                sim_time_s: 0.0,
                camera_position_body_m: [0.0, 0.0, 1.0],
                camera_rotation_body_xyzw: [0.0, 0.0, 0.0, 1.0],
                optics: thalos_capture_protocol::CameraOptics::from_vertical_fov(1.0, [16, 9])
                    .unwrap(),
            });
        }
        catalog
    }

    #[test]
    fn suggested_name_reads_the_view_not_the_scenario() {
        let catalog = ViewpointCatalog::default();
        assert_eq!(
            suggested_name(&catalog, "Thalos", Some(342.0)),
            "Thalos 340 m"
        );
        assert_eq!(
            suggested_name(&catalog, "Mira", Some(412_000.0)),
            "Mira 412 km"
        );
        assert_eq!(suggested_name(&catalog, "Thalos", None), "Thalos");
    }

    #[test]
    fn suggested_name_steps_past_a_taken_slug() {
        let catalog = catalog_with(&["thalos-340-m"]);
        assert_eq!(
            suggested_name(&catalog, "Thalos", Some(340.0)),
            "Thalos 340 m 2"
        );
    }

    #[test]
    fn unique_id_suffixes_a_collision() {
        let catalog = catalog_with(&["ridge", "ridge-2"]);
        assert_eq!(unique_id(&catalog, "ridge"), "ridge-3");
        assert_eq!(unique_id(&catalog, "gully"), "gully");
    }

    #[test]
    fn unique_id_avoids_ids_the_capture_interface_reserves() {
        let catalog = ViewpointCatalog::default();
        assert_eq!(unique_id(&catalog, "latest"), "latest-2");
        assert_eq!(unique_id(&catalog, ""), "viewpoint");
    }

    #[test]
    fn committed_ids_stay_valid_when_the_name_is_overlong() {
        let catalog = ViewpointCatalog::default();
        let stem = viewpoint_id_from_name(&"ridge ".repeat(20));
        let id = unique_id(&catalog, &stem);
        assert!(thalos_capture_protocol::valid_viewpoint_id(&id), "{id:?}");
    }
}
