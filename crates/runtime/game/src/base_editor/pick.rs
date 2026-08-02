//! Site picking (the `PickSite` phase): aim at the surface, see a ghost
//! footprint that tracks the cursor, rotate it with Q/E, left-click to confirm.
//!
//! Confirming scans the natural terrain over the footprint, picks a level pad
//! elevation `E = max height + margin`, registers a `BaseSite` `FlattenTo`
//! structure, applies the flatten, and requests a terrain rebuild so the ground
//! levels out live (the editor pauses the sim but terrain streaming keeps
//! running — see [`crate::rendering::terrain_residency::TerrainRebuildRequest`]).
//!
//! The cursor→surface raycast goes through the shared
//! [`super::cursor_body_dir`], which intersects the body sphere in the
//! heliocentric f64 frame using the god-view camera's *current-frame* pose
//! (its fresh `CellCoord` + `Transform`) rather than its one-frame-stale
//! `GlobalTransform` — so the pick stays glued to the cursor even when the
//! camera pans fast. That requires this system to run
//! `.after(`[`crate::god_view::GodViewCameraSet`]`)`.

use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_body_render::HeightSource;
use thalos_physics_canonical::types::BodyState;
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::camera::{ActiveCamera, ShipCamera};
use crate::coords::SHIP_SCALE;
use crate::rendering::ground_terrain::TerrainFlattenRegistry;
use crate::rendering::terrain_residency::TerrainRebuildRequest;
use crate::rendering::{RealSpaceBody, SimulationState, SolarSystemState};
use crate::structures::{StructurePlacement, StructureRegistry, apply_structure_flatten};

use super::{BaseEditor, BaseEditorMode, base_editor_open, cursor_body_dir};

/// Half-extent of the (square) building pad, metres.
const SITE_HALF_M: f64 = 75.0;
/// Width of the smoothstep blend back to natural terrain beyond the pad, metres.
const SITE_RAMP_M: f64 = 60.0;
/// The pad sits this far above the highest natural ground under the footprint,
/// so no terrain pokes through the level surface.
const SITE_MARGIN_M: f64 = 0.5;
/// Grid resolution (per axis) of the max-height footprint scan.
const FOOTPRINT_SAMPLES: usize = 16;
/// `tile_lod_m` for the height queries — coarse is fine for siting.
const PICK_LOD_M: f32 = 2.0;
/// Footprint heading rotation rate (Q/E), rad/s.
const HEADING_RATE: f32 = 1.5;

/// Transient pick state: the footprint heading and the latest cursor hit.
#[derive(Resource, Default)]
pub(super) struct PickState {
    heading_yaw: f32,
    hit: Option<PickHit>,
}

#[derive(Clone, Copy)]
struct PickHit {
    /// Body-fixed unit direction to the aimed surface point.
    dir_body: DVec3,
    /// Terrain height there, metres above the reference radius.
    height_m: f64,
}

pub(super) struct BaseEditorPickPlugin;

impl Plugin for BaseEditorPickPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PickState>().add_systems(
            Update,
            (update_site_pick, draw_pick_ghost)
                .chain()
                // Pick after the god-view camera moves so the raycast reads this
                // frame's camera pose (see `cursor_body_dir`).
                .after(crate::god_view::GodViewCameraSet)
                .run_if(base_editor_open),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn update_site_pick(
    time: Res<Time<Real>>,
    mut editor: ResMut<BaseEditor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    height_sources: Res<HeightSourceRegistry>,
    mut registry: ResMut<StructureRegistry>,
    mut flatten: ResMut<TerrainFlattenRegistry>,
    mut rebuild: ResMut<TerrainRebuildRequest>,
    mut pick: ResMut<PickState>,
    ui_gate: Res<crate::hud::UiPointerGate>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &CellCoord, &Transform), (With<ShipCamera>, With<ActiveCamera>)>,
    root_grid: Query<&Grid, With<BigSpace>>,
) {
    if editor.mode != BaseEditorMode::PickSite {
        return;
    }
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let body_id = sim.simulation.dominant_body();
    let Some(body_state) = states.get(body_id) else {
        return;
    };
    let Some(body) = sim.system.bodies.get(body_id) else {
        return;
    };
    let radius_m = body.radius_m;

    let dt = time.delta_secs();
    if keys.pressed(KeyCode::KeyQ) {
        pick.heading_yaw -= HEADING_RATE * dt;
    }
    if keys.pressed(KeyCode::KeyE) {
        pick.heading_yaw += HEADING_RATE * dt;
    }

    pick.hit = compute_pick_hit(
        &windows,
        &cameras,
        &root_grid,
        &height_sources,
        body_id,
        body_state,
        radius_m,
    );

    if !ui_gate.hovered
        && mouse.just_pressed(MouseButton::Left)
        && let Some(hit) = pick.hit
    {
        let hs = height_sources.get(body_id);
        confirm_site(
            hit,
            pick.heading_yaw,
            body_id,
            radius_m,
            hs.as_deref(),
            &mut registry,
            &mut flatten,
            &mut rebuild,
            &mut editor,
        );
    }
}

fn compute_pick_hit(
    windows: &Query<&Window, With<PrimaryWindow>>,
    cameras: &Query<(&Camera, &CellCoord, &Transform), (With<ShipCamera>, With<ActiveCamera>)>,
    root_grid: &Query<&Grid, With<BigSpace>>,
    height_sources: &HeightSourceRegistry,
    body_id: BodyId,
    body_state: &BodyState,
    radius_m: f64,
) -> Option<PickHit> {
    let window = windows.single().ok()?;
    let cursor = window.cursor_position()?;
    let (camera, cam_cell, cam_transform) = cameras.single().ok()?;
    let root_grid = root_grid.single().ok()?;
    let dir_body = cursor_body_dir(
        camera,
        cam_cell,
        cam_transform,
        root_grid,
        cursor,
        body_state.position,
        body_state.orientation,
        radius_m,
    )?;
    let height_m = height_sources
        .get(body_id)
        .and_then(|hs| hs.sample_height_m(dir_body.as_vec3(), PICK_LOD_M))
        .unwrap_or(0.0) as f64;
    Some(PickHit { dir_body, height_m })
}

#[allow(clippy::too_many_arguments)]
fn confirm_site(
    hit: PickHit,
    heading_yaw: f32,
    body_id: BodyId,
    radius_m: f64,
    height_source: Option<&dyn HeightSource>,
    registry: &mut StructureRegistry,
    flatten: &mut TerrainFlattenRegistry,
    rebuild: &mut TerrainRebuildRequest,
    editor: &mut BaseEditor,
) {
    let center_dir = hit.dir_body;
    let (heading, across) = site_tangent_frame(center_dir, heading_yaw);

    let max_h = footprint_max_height(height_source, center_dir, heading, across, radius_m)
        .unwrap_or(hit.height_m);
    let elevation_m = max_h + SITE_MARGIN_M;

    let base = registry.create_base(
        body_id,
        center_dir,
        heading,
        StructurePlacement::FlattenTo {
            elevation_m,
            half_along_m: SITE_HALF_M,
            half_across_m: SITE_HALF_M,
            ramp_m: SITE_RAMP_M,
            // Editor sites level a rectangle centred on their anchor.
            rect_offset_along_m: 0.0,
            rect_offset_across_m: 0.0,
        },
    );
    let id = base.root_site;
    if let Some(site) = registry.get(id).copied() {
        apply_structure_flatten(&site, radius_m, flatten);
    }
    rebuild.request(body_id);
    editor.active_site = Some(id);
    editor.mode = BaseEditorMode::PlaceBuildings;
    info!(
        "base site {:?} flattened to {:.1} m above reference on body {}",
        id, elevation_m, body_id
    );
}

/// Body-fixed `(heading, across)` tangent unit vectors at `center_dir`, with the
/// heading rotated by `yaw` around the local vertical.
fn site_tangent_frame(center_dir: DVec3, yaw: f32) -> (DVec3, DVec3) {
    let seed = if center_dir.dot(DVec3::Y).abs() < 0.99 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let tx = seed.cross(center_dir).normalize();
    let tz = center_dir.cross(tx).normalize();
    let yaw = yaw as f64;
    let heading = (tx * yaw.cos() + tz * yaw.sin()).normalize();
    let across = center_dir.cross(heading).normalize();
    (heading, across)
}

/// Highest natural terrain over the square footprint, metres above reference.
fn footprint_max_height(
    hs: Option<&dyn HeightSource>,
    center_dir: DVec3,
    heading: DVec3,
    across: DVec3,
    radius_m: f64,
) -> Option<f64> {
    let hs = hs?;
    let center_point = center_dir * radius_m;
    let mut max_h = f64::MIN;
    for i in 0..=FOOTPRINT_SAMPLES {
        let a = -SITE_HALF_M + 2.0 * SITE_HALF_M * (i as f64 / FOOTPRINT_SAMPLES as f64);
        for j in 0..=FOOTPRINT_SAMPLES {
            let b = -SITE_HALF_M + 2.0 * SITE_HALF_M * (j as f64 / FOOTPRINT_SAMPLES as f64);
            let dir = (center_point + heading * a + across * b).normalize();
            if let Some(h) = hs.sample_height_m(dir.as_vec3(), PICK_LOD_M) {
                max_h = max_h.max(h as f64);
            }
        }
    }
    (max_h != f64::MIN).then_some(max_h)
}

/// Draw the ghost footprint rectangle at the current cursor hit (render space).
fn draw_pick_ghost(
    editor: Res<BaseEditor>,
    pick: Res<PickState>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    bodies: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut gizmos: Gizmos<crate::god_view::GodViewGizmos>,
) {
    if editor.mode != BaseEditorMode::PickSite {
        return;
    }
    let Some(hit) = pick.hit else {
        return;
    };
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let body_id = sim.simulation.dominant_body();
    let Some(body_state) = states.get(body_id) else {
        return;
    };
    let Some(body) = sim.system.bodies.get(body_id) else {
        return;
    };
    let Some((_, body_gt)) = bodies.iter().find(|(rsb, _)| rsb.body_id == body_id) else {
        return;
    };
    let center_render = body_gt.translation();
    let radius_m = body.radius_m;
    let surf_r = radius_m + hit.height_m;

    let (heading, across) = site_tangent_frame(hit.dir_body, pick.heading_yaw);
    let corner = |a: f64, b: f64| -> Vec3 {
        let dir_body = (hit.dir_body * surf_r + heading * a + across * b).normalize();
        let world_dir = (body_state.orientation * dir_body).as_vec3();
        center_render + world_dir * (surf_r * SHIP_SCALE) as f32
    };
    let c = [
        corner(-SITE_HALF_M, -SITE_HALF_M),
        corner(SITE_HALF_M, -SITE_HALF_M),
        corner(SITE_HALF_M, SITE_HALF_M),
        corner(-SITE_HALF_M, SITE_HALF_M),
    ];
    let color = Color::srgb(0.2, 0.9, 1.0);
    for k in 0..4 {
        gizmos.line(c[k], c[(k + 1) % 4], color);
    }
    // Heading marker: a tick on the +heading edge so the pad's orientation
    // (which orients building placement) is legible.
    let edge_mid = (c[1] + c[2]) * 0.5;
    let inward = (center_render - edge_mid)
        .try_normalize()
        .unwrap_or(Vec3::ZERO);
    gizmos.line(
        edge_mid,
        edge_mid + inward * (SITE_HALF_M as f32 * 0.3),
        Color::srgb(1.0, 0.85, 0.3),
    );
}
