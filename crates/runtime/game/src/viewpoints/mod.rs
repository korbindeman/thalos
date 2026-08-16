//! Shared saved-viewpoint integration for the planetary game.
//!
//! Catalog persistence, F8/F9 UI, validation, and CRUD live in
//! `thalos_viewer`. This module is only the authored-body-fixed adapter and
//! the scripted-driver bridge required by the full game.

use std::{env, path::PathBuf};

use bevy::{
    math::{DQuat, DVec3},
    prelude::*,
};
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_capture_protocol::{Viewpoint, ViewpointCatalog, ViewpointFrame, ViewpointSpawn};
use thalos_physics_local::HeightSourceRegistry;
use thalos_viewer::{
    CurrentViewpoint, PendingViewpointApply, ViewpointApplyTarget, ViewpointSet, ViewpointSnapshot,
    ViewpointUiState,
};

use crate::{
    bridge::WarpLimits,
    camera::{ActiveCamera, ShipCamera},
    camera_optics::CameraOptics,
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
            crate::content::ContentRoot::discover()
                .expect("Failed to locate Thalos runtime content")
                .assets()
                .join(VIEWPOINT_CATALOG_FILENAME)
        })
}

pub fn load_catalog() -> Result<ViewpointCatalog, String> {
    thalos_viewer::read_viewpoint_catalog(&catalog_path())
}

pub fn write_catalog(catalog: &ViewpointCatalog) -> Result<(), String> {
    thalos_viewer::write_viewpoint_catalog(&catalog_path(), catalog)
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

pub fn authored_context(
    viewpoint: &Viewpoint,
) -> Result<(&str, ViewpointSpawn, bool, f64), String> {
    viewpoint.authored_body().ok_or_else(|| {
        format!(
            "viewpoint {} uses projected-local space and cannot boot the planetary game",
            viewpoint.id
        )
    })
}

pub(crate) fn viewpoint_spawn_of(value: SpawnSituation) -> ViewpointSpawn {
    match value {
        SpawnSituation::ShipOrbit => ViewpointSpawn::Orbit,
        SpawnSituation::PolarOrbit => ViewpointSpawn::Polar,
        SpawnSituation::Eva => ViewpointSpawn::Eva,
        SpawnSituation::Landing => ViewpointSpawn::Landing,
        SpawnSituation::FinalApproach => ViewpointSpawn::Final,
        SpawnSituation::Runway => ViewpointSpawn::Runway,
        SpawnSituation::RunwayApproach => ViewpointSpawn::RunwayApproach,
        SpawnSituation::Launch => ViewpointSpawn::Launch,
        SpawnSituation::Cruise => ViewpointSpawn::Cruise,
    }
}

pub(crate) fn situation_of_viewpoint(value: ViewpointSpawn) -> SpawnSituation {
    match value {
        ViewpointSpawn::Orbit => SpawnSituation::ShipOrbit,
        ViewpointSpawn::Polar => SpawnSituation::PolarOrbit,
        ViewpointSpawn::Eva => SpawnSituation::Eva,
        ViewpointSpawn::Landing => SpawnSituation::Landing,
        ViewpointSpawn::Final => SpawnSituation::FinalApproach,
        ViewpointSpawn::Runway => SpawnSituation::Runway,
        ViewpointSpawn::RunwayApproach => SpawnSituation::RunwayApproach,
        ViewpointSpawn::Launch => SpawnSituation::Launch,
        ViewpointSpawn::Cruise => SpawnSituation::Cruise,
    }
}

pub struct ViewpointManagerPlugin;

impl Plugin for ViewpointManagerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(thalos_viewer::ViewpointPlugin::new(catalog_path(), true))
            .configure_sets(
                Update,
                ViewpointSet::Snapshot.after(crate::SimStage::Camera),
            )
            .add_systems(
                Update,
                project_current_viewpoint.in_set(ViewpointSet::Snapshot),
            )
            .add_systems(Update, apply_pending_viewpoint.in_set(ViewpointSet::Apply));
    }
}

#[allow(clippy::too_many_arguments)]
fn project_current_viewpoint(
    view_anchor: Res<ViewAnchor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    situation: Res<SpawnSituation>,
    space_center: Option<Res<crate::space_center::SpaceCenter>>,
    cameras: Query<(&CellCoord, &Transform, &CameraOptics), (With<ShipCamera>, With<ActiveCamera>)>,
    mut current: ResMut<CurrentViewpoint>,
) {
    let snapshot = capture_current_snapshot(
        &view_anchor,
        &sim,
        &solar,
        *situation,
        space_center.as_deref(),
        &cameras,
    )
    .ok();
    if current.0 != snapshot {
        current.0 = snapshot;
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_pending_viewpoint(
    mut pending: ResMut<PendingViewpointApply>,
    mut ui: ResMut<ViewpointUiState>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    height_sources: Res<HeightSourceRegistry>,
    structures: Res<StructureRegistry>,
    surfaces: Res<BodySurfaceRegistry>,
    warp_limits: Res<WarpLimits>,
    root: Single<&Grid, With<BigSpace>>,
    mut freecam: ResMut<FreeCam>,
    mut camera: Query<
        (
            &mut CellCoord,
            &mut Transform,
            &mut Projection,
            &mut CameraOptics,
        ),
        (With<ShipCamera>, With<ActiveCamera>),
    >,
) {
    let Some(target) = pending.take() else {
        return;
    };
    let result = (|| {
        let (mut cell, mut transform, mut projection, mut optics) = camera
            .single_mut()
            .map_err(|_| "the active 3-D camera is unavailable".to_owned())?;
        let return_optics = optics.spec();
        let (body_id, message) = match target {
            ViewpointApplyTarget::Saved(viewpoint) => {
                let (body, _, _, _) = authored_context(&viewpoint)?;
                let body_id = sim
                    .system
                    .bodies
                    .iter()
                    .position(|definition| definition.name.eq_ignore_ascii_case(body))
                    .ok_or_else(|| format!("viewpoint body {body:?} is not authored"))?;
                let message = pose_viewpoint(
                    &viewpoint,
                    &sim.system.bodies,
                    &solar,
                    &root,
                    &mut cell,
                    &mut transform,
                    &mut projection,
                    &mut optics,
                )?;
                (body_id, message)
            }
            ViewpointApplyTarget::Scripted(viewpoint) => {
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
                )?
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
            &sim,
            &warp_limits,
            return_optics,
        );
        Ok(format!("{message}; freecam active"))
    })();
    ui.report(result);
}

#[allow(clippy::too_many_arguments)]
fn capture_current_snapshot(
    view_anchor: &ViewAnchor,
    sim: &SimulationState,
    solar: &SolarSystemState,
    situation: SpawnSituation,
    space_center: Option<&crate::space_center::SpaceCenter>,
    cameras: &Query<
        (&CellCoord, &Transform, &CameraOptics),
        (With<ShipCamera>, With<ActiveCamera>),
    >,
) -> Result<ViewpointSnapshot, String> {
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
    let (cell, transform, optics) = cameras
        .single()
        .map_err(|_| "switch to the active 3-D camera before saving a viewpoint".to_owned())?;

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
    Ok(ViewpointSnapshot {
        frame: ViewpointFrame::AuthoredBodyFixed {
            body: body.name.clone(),
            spawn: viewpoint_spawn_of(situation),
            boots_hub: space_center.is_some_and(|hub| hub.open),
            sim_time_s: sim.simulation.sim_time(),
        },
        camera_position_m: camera_body.to_array(),
        camera_rotation_xyzw: rotation_body.to_array(),
        optics: optics.spec(),
        suggested_name: suggested_name(&body.name, Some(anchor.agl_m)),
    })
}

fn suggested_name(body: &str, agl_m: Option<f64>) -> String {
    match agl_m {
        Some(agl) if agl.is_finite() => {
            let agl = agl.max(0.0);
            if agl < 1_000.0 {
                format!("{body} {} m", (agl / 10.0).round() as i64 * 10)
            } else {
                format!("{body} {} km", (agl / 1_000.0).round() as i64)
            }
        }
        _ => body.to_owned(),
    }
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
    optics: &mut CameraOptics,
) -> Result<String, String> {
    let (body, _, _, _) = authored_context(viewpoint)?;
    let body_id = bodies
        .iter()
        .position(|definition| definition.name.eq_ignore_ascii_case(body))
        .ok_or_else(|| format!("viewpoint body {body:?} is not authored"))?;
    let states = solar
        .states
        .as_deref()
        .ok_or_else(|| "the solar-system state is not ready yet".to_owned())?;
    let body_state = states
        .get(body_id)
        .ok_or_else(|| format!("body state for {body:?} is unavailable"))?;
    let surface_q =
        crate::rendering::transforms::surface_orientation_authored(bodies, body_id, states)
            .unwrap_or_else(|| body_state.orientation.normalize());
    let camera_body = DVec3::from_array(viewpoint.camera_position_m);
    let rotation_body = DQuat::from_array(viewpoint.camera_rotation_xyzw).normalize();
    let camera_world = body_state.position + surface_q * camera_body;
    let rotation_world = (surface_q * rotation_body).normalize();
    let (next_cell, local) = root.translation_to_grid(camera_world);
    *cell = next_cell;
    transform.translation = local;
    transform.rotation = rotation_world.as_quat();
    optics.set_spec(viewpoint.optics)?;
    optics.apply_to_projection(projection);
    Ok(format!("Viewing {}", viewpoint.id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewpoint_spawn_conversion_round_trips_every_scene() {
        for situation in [
            SpawnSituation::ShipOrbit,
            SpawnSituation::PolarOrbit,
            SpawnSituation::Eva,
            SpawnSituation::Landing,
            SpawnSituation::FinalApproach,
            SpawnSituation::Runway,
            SpawnSituation::RunwayApproach,
            SpawnSituation::Launch,
            SpawnSituation::Cruise,
        ] {
            assert_eq!(
                situation_of_viewpoint(viewpoint_spawn_of(situation)),
                situation
            );
        }
    }

    #[test]
    fn planetary_adapter_rejects_a_projected_local_viewpoint() {
        let viewpoint = Viewpoint {
            id: "westpunt".into(),
            name: "Westpunt".into(),
            description: String::new(),
            saved_unix_ms: 1,
            frame: ViewpointFrame::ProjectedLocal {
                reference: "EPSG:32619".into(),
            },
            camera_position_m: [1.0, 2.0, 3.0],
            camera_rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            optics: thalos_capture_protocol::CameraOptics::default(),
        };

        assert!(authored_context(&viewpoint).is_err());
    }
}
