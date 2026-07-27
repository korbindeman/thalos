//! Shared authored viewpoints, the interactive F8 manager, and the F9
//! quick-save prompt.
//!
//! The JSON catalog is source data: agents may edit it directly, while the
//! in-game manager performs the same CRUD operations. Headless capture reads
//! these types through `thalos_capture_protocol` and reuses [`pose_viewpoint`].
//!
//! Two entry points, one core: F8 opens the full catalog manager, F9 saves the
//! current view in one keypress. Both frame the camera through
//! [`capture_current_viewpoint`] and commit through [`write_catalog`] — the
//! quick path parameterizes that core, it does not fork it.

pub mod quick_save;

use std::{env, fs, path::PathBuf};

use bevy::{
    math::{DQuat, DVec3},
    prelude::*,
    window::PrimaryWindow,
};
use bevy_egui::{
    EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext, egui,
};
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_capture_protocol::{
    Viewpoint, ViewpointCatalog, ViewpointSpawn, viewpoint_id_from_name,
};
use thalos_input::game::GameInputIntent;
use thalos_physics_local::HeightSourceRegistry;

use crate::{
    bridge::WarpLimits,
    camera::{ActiveCamera, ShipCamera},
    freecam::FreeCam,
    rendering::{SimulationState, SolarSystemState, view_anchor::ViewAnchor},
    spawn::SpawnSituation,
    structures::StructureRegistry,
    terrain_registry::BodySurfaceRegistry,
};

const VIEWPOINT_CATALOG_FILENAME: &str = "viewpoints.json";

pub fn catalog_path() -> PathBuf {
    env::var_os("THALOS_VIEWPOINTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../assets")
                .join(VIEWPOINT_CATALOG_FILENAME)
        })
}

pub fn load_catalog() -> Result<ViewpointCatalog, String> {
    let path = catalog_path();
    let bytes =
        fs::read(&path).map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let catalog: ViewpointCatalog = serde_json::from_slice(&bytes)
        .map_err(|error| format!("could not parse {}: {error}", path.display()))?;
    catalog.validate()?;
    Ok(catalog)
}

pub fn write_catalog(catalog: &ViewpointCatalog) -> Result<(), String> {
    catalog.validate()?;
    let path = catalog_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    let mut canonical = catalog.clone();
    canonical
        .viewpoints
        .sort_by(|left, right| left.id.cmp(&right.id));
    canonical
        .scripted_viewpoints
        .sort_by(|left, right| left.id.cmp(&right.id));
    let mut json = serde_json::to_vec_pretty(&canonical)
        .map_err(|error| format!("could not encode viewpoint catalog: {error}"))?;
    json.push(b'\n');
    fs::write(&path, json).map_err(|error| format!("could not write {}: {error}", path.display()))
}

pub fn resolve_viewpoint(raw: &str) -> Result<Viewpoint, String> {
    let catalog = load_catalog()?;
    let requested = raw.trim().strip_prefix("viewpoint:").unwrap_or(raw.trim());
    let viewpoint = if matches!(
        requested.to_ascii_lowercase().as_str(),
        "latest" | "perspective" | "latest-perspective" | "latest_perspective"
    ) {
        catalog.latest()
    } else {
        catalog.find(&requested.to_ascii_lowercase())
    };
    viewpoint.cloned().ok_or_else(|| {
        if catalog.viewpoints.is_empty() {
            format!(
                "the viewpoint catalog {} is empty; press F8 in game to create one or add one to the JSON",
                catalog_path().display()
            )
        } else {
            format!(
                "viewpoint {requested:?} was not found in {}",
                catalog_path().display()
            )
        }
    })
}

pub fn viewpoint_scene_name(viewpoint: &Viewpoint) -> String {
    format!("viewpoint:{}", viewpoint.id)
}

impl From<SpawnSituation> for ViewpointSpawn {
    fn from(value: SpawnSituation) -> Self {
        match value {
            SpawnSituation::ShipOrbit => Self::Orbit,
            SpawnSituation::PolarOrbit => Self::Polar,
            SpawnSituation::Eva => Self::Eva,
            SpawnSituation::Landing => Self::Landing,
            SpawnSituation::FinalApproach => Self::Final,
            SpawnSituation::Runway => Self::Runway,
            SpawnSituation::RunwayApproach => Self::RunwayApproach,
            SpawnSituation::Launch => Self::Launch,
            SpawnSituation::Cruise => Self::Cruise,
        }
    }
}

impl From<ViewpointSpawn> for SpawnSituation {
    fn from(value: ViewpointSpawn) -> Self {
        match value {
            ViewpointSpawn::Orbit => Self::ShipOrbit,
            ViewpointSpawn::Polar => Self::PolarOrbit,
            ViewpointSpawn::Eva => Self::Eva,
            ViewpointSpawn::Landing => Self::Landing,
            ViewpointSpawn::Final => Self::FinalApproach,
            ViewpointSpawn::Runway => Self::Runway,
            ViewpointSpawn::RunwayApproach => Self::RunwayApproach,
            ViewpointSpawn::Launch => Self::Launch,
            ViewpointSpawn::Cruise => Self::Cruise,
        }
    }
}

pub struct ViewpointManagerPlugin;

impl Plugin for ViewpointManagerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            // Thalos creates the map camera before the ship camera. Egui's
            // default "first camera wins" behavior therefore attaches its
            // primary context to an inactive camera during normal flight.
            .insert_resource(EguiGlobalSettings {
                auto_create_primary_context: false,
                ..default()
            })
            .init_resource::<ViewpointManager>()
            // CameraPlugin creates both world cameras during Startup. Attach
            // after those commands have applied so egui has one explicit
            // presentation owner and never relies on camera creation order.
            .add_systems(PostStartup, attach_viewpoint_manager_to_ship_camera)
            .add_systems(Update, toggle_viewpoint_manager)
            .add_systems(EguiPrimaryContextPass, draw_viewpoint_manager);
    }
}

fn attach_viewpoint_manager_to_ship_camera(
    mut commands: Commands,
    camera: Single<Entity, With<ShipCamera>>,
) {
    commands.entity(*camera).insert(PrimaryEguiContext);
}

#[derive(Resource)]
pub struct ViewpointManager {
    pub open: bool,
    catalog: ViewpointCatalog,
    selected: Option<String>,
    edit_id: String,
    edit_name: String,
    edit_description: String,
    status: Option<(bool, String)>,
}

impl Default for ViewpointManager {
    fn default() -> Self {
        Self {
            open: false,
            catalog: ViewpointCatalog::default(),
            selected: None,
            edit_id: String::new(),
            edit_name: "New viewpoint".to_owned(),
            edit_description: String::new(),
            status: None,
        }
    }
}

fn toggle_viewpoint_manager(input: Res<GameInputIntent>, mut manager: ResMut<ViewpointManager>) {
    if !input.save_perspective {
        return;
    }
    manager.open = !manager.open;
    if manager.open {
        reload_manager(&mut manager);
    }
}

#[derive(Clone, Copy)]
enum ManagerAction {
    Reload,
    Create,
    SaveMetadata,
    ReplaceCamera,
    Apply,
    Delete,
}

#[allow(clippy::too_many_arguments)]
fn draw_viewpoint_manager(
    mut contexts: EguiContexts,
    mut manager: ResMut<ViewpointManager>,
    view_anchor: Res<ViewAnchor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    height_sources: Res<HeightSourceRegistry>,
    structures: Res<StructureRegistry>,
    surfaces: Res<BodySurfaceRegistry>,
    warp_limits: Res<WarpLimits>,
    mut freecam: ResMut<FreeCam>,
    situation: Res<SpawnSituation>,
    space_center: Option<Res<crate::space_center::SpaceCenter>>,
    root: Single<&Grid, With<BigSpace>>,
    mut cameras: ParamSet<(
        Query<(&CellCoord, &Transform, &Projection), (With<ShipCamera>, With<ActiveCamera>)>,
        Query<
            (&mut CellCoord, &mut Transform, &mut Projection),
            (With<ShipCamera>, With<ActiveCamera>),
        >,
    )>,
    windows: Query<&Window, With<PrimaryWindow>>,
) -> Result {
    if !manager.open {
        return Ok(());
    }

    let mut open = manager.open;
    let mut action = None;
    let entries = manager
        .catalog
        .viewpoints
        .iter()
        .map(|viewpoint| {
            (
                viewpoint.id.clone(),
                viewpoint.name.clone(),
                "saved".to_owned(),
            )
        })
        .chain(manager.catalog.scripted_viewpoints.iter().map(|viewpoint| {
            (
                viewpoint.id.clone(),
                viewpoint.name.clone(),
                "agent".to_owned(),
            )
        }))
        .collect::<Vec<_>>();

    egui::Window::new("Viewpoint manager")
        .open(&mut open)
        .default_width(680.0)
        .default_height(500.0)
        .resizable(true)
        .show(contexts.ctx_mut()?, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("F8 viewpoints").strong());
                if ui.button("Reload file").clicked() {
                    action = Some(ManagerAction::Reload);
                }
                ui.label(catalog_path().display().to_string());
            });
            ui.separator();

            ui.columns(2, |columns| {
                columns[0].set_min_width(210.0);
                columns[0].heading("Catalog");
                egui::ScrollArea::vertical().id_salt("viewpoint-list").show(
                    &mut columns[0],
                    |ui| {
                        if entries.is_empty() {
                            ui.weak("No viewpoints yet.");
                        }
                        for (id, name, kind) in &entries {
                            let selected = manager.selected.as_deref() == Some(id.as_str());
                            if ui
                                .selectable_label(selected, format!("{name}  [{kind}]\n{id}"))
                                .clicked()
                            {
                                select_viewpoint(&mut manager, id);
                            }
                        }
                    },
                );

                columns[1].heading(if manager.selected.is_some() {
                    "Selected viewpoint"
                } else {
                    "New viewpoint"
                });
                columns[1].label("Name");
                columns[1].text_edit_singleline(&mut manager.edit_name);
                columns[1].label("Stable id");
                columns[1].text_edit_singleline(&mut manager.edit_id);
                columns[1].label("Description");
                columns[1].text_edit_multiline(&mut manager.edit_description);
                columns[1].add_space(8.0);

                if let Some(selected_id) = manager.selected.as_deref()
                    && let Some(viewpoint) = manager.catalog.find(selected_id)
                {
                    columns[1].label(format!(
                        "{} · {:?}{} · {}×{} · {:.1}° FOV",
                        viewpoint.body,
                        viewpoint.spawn,
                        if viewpoint.boots_hub { " · hub" } else { "" },
                        viewpoint.viewport[0],
                        viewpoint.viewport[1],
                        viewpoint.vertical_fov_rad.to_degrees(),
                    ));
                } else if let Some(selected_id) = manager.selected.as_deref()
                    && let Some(viewpoint) = manager.catalog.find_scripted(selected_id)
                {
                    columns[1].label(format!(
                        "Agent-authored scripted view · driver {}",
                        viewpoint.driver
                    ));
                }

                columns[1].horizontal_wrapped(|ui| {
                    if manager.selected.is_none() {
                        if ui.button("Save current as new").clicked() {
                            action = Some(ManagerAction::Create);
                        }
                    } else {
                        if ui.button("View").clicked() {
                            action = Some(ManagerAction::Apply);
                        }
                        if ui.button("Replace from current").clicked() {
                            action = Some(ManagerAction::ReplaceCamera);
                        }
                        if ui.button("Save name / id / notes").clicked() {
                            action = Some(ManagerAction::SaveMetadata);
                        }
                        if ui
                            .add(
                                egui::Button::new("Delete")
                                    .fill(egui::Color32::from_rgb(105, 32, 32)),
                            )
                            .clicked()
                        {
                            action = Some(ManagerAction::Delete);
                        }
                    }
                });
                if manager.selected.is_some() && columns[1].button("Create another").clicked() {
                    manager.selected = None;
                    manager.edit_id.clear();
                    manager.edit_name = "New viewpoint".to_owned();
                    manager.edit_description.clear();
                }
            });

            if let Some((ok, message)) = &manager.status {
                ui.separator();
                ui.colored_label(
                    if *ok {
                        egui::Color32::LIGHT_GREEN
                    } else {
                        egui::Color32::LIGHT_RED
                    },
                    message,
                );
            }
            ui.separator();
            ui.weak(
                "Saved views restore an exact body-fixed camera and lens. Agent views reuse \
                 the procedural focus/framing driver; headless capture also applies its \
                 diagnostic setup and canonical boot scene.",
            );
        });
    manager.open = open;

    let Some(action) = action else {
        return Ok(());
    };
    let result = match action {
        ManagerAction::Reload => {
            reload_manager(&mut manager);
            return Ok(());
        }
        ManagerAction::Create => {
            let readable_cameras = cameras.p0();
            create_from_current(
                &mut manager,
                &view_anchor,
                &sim,
                &solar,
                *situation,
                space_center.as_deref(),
                &readable_cameras,
                &windows,
            )
        }
        ManagerAction::SaveMetadata => save_metadata(&mut manager),
        ManagerAction::ReplaceCamera => {
            let readable_cameras = cameras.p0();
            replace_from_current(
                &mut manager,
                &view_anchor,
                &sim,
                &solar,
                *situation,
                space_center.as_deref(),
                &readable_cameras,
                &windows,
            )
        }
        ManagerAction::Apply => {
            let selected = manager
                .selected
                .clone()
                .ok_or_else(|| "select a viewpoint first".to_owned());
            selected.and_then(|selected_id| {
                let mut writable_cameras = cameras.p1();
                let (mut cell, mut transform, mut projection) = writable_cameras
                    .single_mut()
                    .map_err(|_| "the active 3-D camera is unavailable".to_owned())?;
                let resolved = if let Some(viewpoint) = manager.catalog.find(&selected_id) {
                    let body_id = sim
                        .system
                        .bodies
                        .iter()
                        .position(|body| body.name.eq_ignore_ascii_case(&viewpoint.body))
                        .ok_or_else(|| {
                            format!("viewpoint body {:?} is not authored", viewpoint.body)
                        })?;
                    let message = pose_viewpoint(
                        viewpoint,
                        &sim.system.bodies,
                        &solar,
                        &root,
                        &mut cell,
                        &mut transform,
                        &mut projection,
                    )?;
                    Ok((body_id, message))
                } else if let Some(viewpoint) = manager.catalog.find_scripted(&selected_id) {
                    crate::screenshot::pose_scripted_viewpoint(
                        &viewpoint.driver,
                        &sim,
                        &solar,
                        &height_sources,
                        &structures,
                        &surfaces,
                        &root,
                        &mut cell,
                        &mut transform,
                    )
                } else {
                    Err(format!(
                        "{selected_id:?} changed on disk; reload the catalog"
                    ))
                };
                let (body_id, message) = resolved?;
                let fov = match projection.as_ref() {
                    Projection::Perspective(perspective) => perspective.fov,
                    _ => {
                        return Err("the active 3-D camera is not perspective-projected".to_owned());
                    }
                };
                let states = solar
                    .states
                    .as_deref()
                    .ok_or_else(|| "the solar-system state is not ready yet".to_owned())?;
                let body_state = states
                    .get(body_id)
                    .ok_or_else(|| format!("body state {body_id} is unavailable"))?;
                let camera_world = root.grid_position_double(&cell, &transform);
                freecam.activate_at_world_pose(
                    body_id,
                    body_state,
                    camera_world,
                    transform.rotation.as_dquat(),
                    fov,
                    &sim,
                    &warp_limits,
                );
                Ok(format!("{message}; freecam active"))
            })
        }
        ManagerAction::Delete => delete_selected(&mut manager),
    };
    manager.status = Some(match result {
        Ok(message) => (true, message),
        Err(error) => (false, error),
    });
    Ok(())
}

fn reload_manager(manager: &mut ViewpointManager) {
    match load_catalog() {
        Ok(catalog) => {
            manager.catalog = catalog;
            if manager
                .selected
                .as_deref()
                .is_some_and(|id| !manager.catalog.contains(id))
            {
                manager.selected = None;
            }
            if let Some(id) = manager.selected.clone() {
                select_viewpoint(manager, &id);
            }
            manager.status = Some((true, "Reloaded viewpoints.json".to_owned()));
        }
        Err(error) => manager.status = Some((false, error)),
    }
}

fn select_viewpoint(manager: &mut ViewpointManager, id: &str) {
    if let Some(viewpoint) = manager.catalog.find(id) {
        manager.selected = Some(viewpoint.id.clone());
        manager.edit_id = viewpoint.id.clone();
        manager.edit_name = viewpoint.name.clone();
        manager.edit_description = viewpoint.description.clone();
    } else if let Some(viewpoint) = manager.catalog.find_scripted(id) {
        manager.selected = Some(viewpoint.id.clone());
        manager.edit_id = viewpoint.id.clone();
        manager.edit_name = viewpoint.name.clone();
        manager.edit_description = viewpoint.description.clone();
    }
}

fn create_from_current(
    manager: &mut ViewpointManager,
    view_anchor: &ViewAnchor,
    sim: &SimulationState,
    solar: &SolarSystemState,
    situation: SpawnSituation,
    space_center: Option<&crate::space_center::SpaceCenter>,
    cameras: &Query<(&CellCoord, &Transform, &Projection), (With<ShipCamera>, With<ActiveCamera>)>,
    windows: &Query<&Window, With<PrimaryWindow>>,
) -> Result<String, String> {
    let name = manager.edit_name.trim();
    if name.is_empty() {
        return Err("give the viewpoint a name first".to_owned());
    }
    let requested_id = if manager.edit_id.trim().is_empty() {
        viewpoint_id_from_name(name)
    } else {
        manager.edit_id.trim().to_ascii_lowercase()
    };
    let mut catalog = load_catalog()?;
    if catalog.contains(&requested_id) {
        return Err(format!("viewpoint id {requested_id:?} already exists"));
    }
    let viewpoint = capture_current_viewpoint(
        requested_id.clone(),
        name.to_owned(),
        manager.edit_description.trim().to_owned(),
        view_anchor,
        sim,
        solar,
        situation,
        space_center,
        cameras,
        windows,
    )?;
    viewpoint.validate()?;
    catalog.viewpoints.push(viewpoint);
    write_catalog(&catalog)?;
    manager.catalog = catalog;
    select_viewpoint(manager, &requested_id);
    Ok(format!(
        "Saved {requested_id}; agents can run `just screenshot {requested_id}`"
    ))
}

fn save_metadata(manager: &mut ViewpointManager) -> Result<String, String> {
    let old_id = manager
        .selected
        .clone()
        .ok_or_else(|| "select a viewpoint first".to_owned())?;
    let new_id = manager.edit_id.trim().to_ascii_lowercase();
    let new_name = manager.edit_name.trim().to_owned();
    let new_description = manager.edit_description.trim().to_owned();
    let mut catalog = load_catalog()?;
    if new_id != old_id && catalog.contains(&new_id) {
        return Err(format!("viewpoint id {new_id:?} already exists"));
    }
    if let Some(viewpoint) = catalog
        .viewpoints
        .iter_mut()
        .find(|viewpoint| viewpoint.id == old_id)
    {
        viewpoint.id = new_id.clone();
        viewpoint.name = new_name;
        viewpoint.description = new_description;
        viewpoint.validate()?;
    } else if let Some(viewpoint) = catalog
        .scripted_viewpoints
        .iter_mut()
        .find(|viewpoint| viewpoint.id == old_id)
    {
        viewpoint.id = new_id.clone();
        viewpoint.name = new_name;
        viewpoint.description = new_description;
        viewpoint.validate()?;
    } else {
        return Err(format!("{old_id:?} changed on disk; reload the catalog"));
    }
    write_catalog(&catalog)?;
    manager.catalog = catalog;
    select_viewpoint(manager, &new_id);
    Ok(format!("Saved metadata for {new_id}"))
}

fn replace_from_current(
    manager: &mut ViewpointManager,
    view_anchor: &ViewAnchor,
    sim: &SimulationState,
    solar: &SolarSystemState,
    situation: SpawnSituation,
    space_center: Option<&crate::space_center::SpaceCenter>,
    cameras: &Query<(&CellCoord, &Transform, &Projection), (With<ShipCamera>, With<ActiveCamera>)>,
    windows: &Query<&Window, With<PrimaryWindow>>,
) -> Result<String, String> {
    let old_id = manager
        .selected
        .clone()
        .ok_or_else(|| "select a viewpoint first".to_owned())?;
    let replacement = capture_current_viewpoint(
        manager.edit_id.trim().to_ascii_lowercase(),
        manager.edit_name.trim().to_owned(),
        manager.edit_description.trim().to_owned(),
        view_anchor,
        sim,
        solar,
        situation,
        space_center,
        cameras,
        windows,
    )?;
    replacement.validate()?;
    let mut catalog = load_catalog()?;
    if replacement.id != old_id && catalog.contains(&replacement.id) {
        return Err(format!("viewpoint id {:?} already exists", replacement.id));
    }
    let new_id = replacement.id.clone();
    if let Some(entry) = catalog
        .viewpoints
        .iter_mut()
        .find(|viewpoint| viewpoint.id == old_id)
    {
        *entry = replacement;
    } else {
        let before = catalog.scripted_viewpoints.len();
        catalog
            .scripted_viewpoints
            .retain(|viewpoint| viewpoint.id != old_id);
        if catalog.scripted_viewpoints.len() == before {
            return Err(format!("{old_id:?} changed on disk; reload the catalog"));
        }
        catalog.viewpoints.push(replacement);
    }
    write_catalog(&catalog)?;
    manager.catalog = catalog;
    select_viewpoint(manager, &new_id);
    Ok(format!("Replaced {new_id} from the current camera"))
}

fn delete_selected(manager: &mut ViewpointManager) -> Result<String, String> {
    let id = manager
        .selected
        .clone()
        .ok_or_else(|| "select a viewpoint first".to_owned())?;
    let mut catalog = load_catalog()?;
    let before = catalog.viewpoints.len() + catalog.scripted_viewpoints.len();
    catalog.viewpoints.retain(|viewpoint| viewpoint.id != id);
    catalog
        .scripted_viewpoints
        .retain(|viewpoint| viewpoint.id != id);
    if catalog.viewpoints.len() + catalog.scripted_viewpoints.len() == before {
        return Err(format!("{id:?} changed on disk; reload the catalog"));
    }
    write_catalog(&catalog)?;
    manager.catalog = catalog;
    manager.selected = None;
    manager.edit_id.clear();
    manager.edit_name = "New viewpoint".to_owned();
    manager.edit_description.clear();
    Ok(format!("Deleted {id}"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_current_viewpoint(
    id: String,
    name: String,
    description: String,
    view_anchor: &ViewAnchor,
    sim: &SimulationState,
    solar: &SolarSystemState,
    situation: SpawnSituation,
    space_center: Option<&crate::space_center::SpaceCenter>,
    cameras: &Query<(&CellCoord, &Transform, &Projection), (With<ShipCamera>, With<ActiveCamera>)>,
    windows: &Query<&Window, With<PrimaryWindow>>,
) -> Result<Viewpoint, String> {
    let anchor = view_anchor
        .resolved
        .ok_or_else(|| "the 3-D view is not anchored to a terrain body yet".to_owned())?;
    let states = solar
        .states
        .as_deref()
        .ok_or_else(|| "the solar-system state is not ready yet".to_owned())?;
    let body_state = states
        .get(anchor.body)
        .ok_or_else(|| "the view body's state is unavailable".to_owned())?;
    let body = sim
        .system
        .bodies
        .get(anchor.body)
        .ok_or_else(|| "the view body is unavailable".to_owned())?;
    let (cell, transform, projection) = cameras
        .single()
        .map_err(|_| "switch to the active 3-D camera before saving a viewpoint".to_owned())?;
    let Projection::Perspective(perspective) = projection else {
        return Err("the active 3-D camera is not perspective-projected".to_owned());
    };

    let camera_world = DVec3::new(cell.x as f64, cell.y as f64, cell.z as f64)
        * crate::rendering::real_space::REAL_SPACE_CELL_SIZE_M as f64
        + transform.translation.as_dvec3();
    let surface_q = crate::rendering::transforms::surface_orientation_authored(
        &sim.system.bodies,
        anchor.body,
        states,
    )
    .unwrap_or_else(|| body_state.orientation.normalize());
    let camera_body = surface_q.inverse() * (camera_world - body_state.position);
    let rotation_world = DQuat::from_xyzw(
        transform.rotation.x as f64,
        transform.rotation.y as f64,
        transform.rotation.z as f64,
        transform.rotation.w as f64,
    );
    let rotation_body = (surface_q.inverse() * rotation_world).normalize();
    let viewport = windows
        .single()
        .map(|window| {
            [
                window.resolution.physical_width().max(1),
                window.resolution.physical_height().max(1),
            ]
        })
        .unwrap_or([1920, 1080]);

    Ok(Viewpoint {
        id,
        name,
        description,
        saved_unix_ms: crate::screenshot::timestamp_millis(),
        body: body.name.clone(),
        spawn: situation.into(),
        boots_hub: space_center.is_some_and(|hub| hub.open),
        sim_time_s: sim.simulation.sim_time(),
        camera_position_body_m: camera_body.to_array(),
        camera_rotation_body_xyzw: [
            rotation_body.x,
            rotation_body.y,
            rotation_body.z,
            rotation_body.w,
        ],
        vertical_fov_rad: perspective.fov,
        viewport,
    })
}

/// Re-project an authored viewpoint through the current body's surface frame.
pub fn pose_viewpoint(
    viewpoint: &Viewpoint,
    bodies: &[thalos_world::BodyDefinition],
    solar: &SolarSystemState,
    root: &Grid,
    cell: &mut CellCoord,
    transform: &mut Transform,
    projection: &mut Projection,
) -> Result<String, String> {
    let body_id = bodies
        .iter()
        .position(|body| body.name.eq_ignore_ascii_case(&viewpoint.body))
        .ok_or_else(|| format!("viewpoint body {:?} is not authored", viewpoint.body))?;
    let states = solar
        .states
        .as_deref()
        .ok_or_else(|| "the solar-system state is not ready yet".to_owned())?;
    let body_state = states
        .get(body_id)
        .ok_or_else(|| format!("body state for {:?} is unavailable", viewpoint.body))?;
    let surface_q =
        crate::rendering::transforms::surface_orientation_authored(bodies, body_id, states)
            .unwrap_or_else(|| body_state.orientation.normalize());
    let camera_body = DVec3::from_array(viewpoint.camera_position_body_m);
    let rotation_body = DQuat::from_xyzw(
        viewpoint.camera_rotation_body_xyzw[0],
        viewpoint.camera_rotation_body_xyzw[1],
        viewpoint.camera_rotation_body_xyzw[2],
        viewpoint.camera_rotation_body_xyzw[3],
    )
    .normalize();
    let camera_world = body_state.position + surface_q * camera_body;
    let rotation_world = (surface_q * rotation_body).normalize();
    let (next_cell, local) = root.translation_to_grid(camera_world);
    *cell = next_cell;
    transform.translation = local;
    transform.rotation = Quat::from_xyzw(
        rotation_world.x as f32,
        rotation_world.y as f32,
        rotation_world.z as f32,
        rotation_world.w as f32,
    )
    .normalize();
    if let Projection::Perspective(perspective) = projection {
        perspective.fov = viewpoint.vertical_fov_rad;
    }
    Ok(format!("Viewing {}", viewpoint.id))
}
