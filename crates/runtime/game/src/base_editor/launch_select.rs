//! In-world launch-point selection — the default flight flow's "pick where to
//! launch" step.
//!
//! After designing / loading a craft in the shipyard and hitting LAUNCH, the
//! game drops into a god-view of the spaceport (a
//! [`BaseEditorMode::SelectLaunch`](super::BaseEditorMode::SelectLaunch) session
//! reusing the base editor's camera + `SimClock`-pause gating) and the player
//! clicks a **runway** (→ horizontal, on gear) or a **launchpad** (→ vertical)
//! to place the craft on and fly. The craft has already been rebuilt into an
//! orbit hold by the relaunch flow; here it is teleported onto the chosen point.
//!
//! Flow:
//! 1. Something sets [`SpaceportLaunchRequest::arm`] (the shipyard's LAUNCH).
//! 2. [`begin_launch_flow`] (once the relaunch is idle) either opens the
//!    god-view immediately if the spaceport is already built (the
//!    [`RunwaySite`] resource exists), or — first launch — runs a brief loading
//!    pass that builds the spaceport ([`build_spaceport`]) behind the loading
//!    screen, then opens the god-view on `Loading → Running`.
//! 3. [`update_launch_pick`] raycasts the cursor against the base and latches
//!    the clicked launch point; [`apply_launch_placement`] measures the craft's
//!    ground clearance (retrying until its meshes / gear are resident) and
//!    places it via the shared [`place_on_runway`] / [`place_on_launchpad`]
//!    cores, then closes the god-view and resumes flight.

use bevy::camera::visibility::RenderLayers;
use bevy::math::{DMat3, DQuat, DVec3};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_physics_local::{ActiveLocalBubble, HeightSourceRegistry};
use thalos_shipyard::{AttachNodes, EngineActivation};
use thalos_world::BodyId;

use crate::camera::{ActiveCamera, ShipCamera};
use crate::coords::{SHIP_LAYER, SHIP_SCALE};
use crate::game_context::{ContextHistory, GameContext};
use crate::god_view::GodViewGizmos;
use crate::loading::{AppState, LoadDestination, LoadingTracker, StepDesc, step};
use crate::rendering::ground_terrain::TerrainFlattenRegistry;
use crate::rendering::real_space::RealSpaceRoot;
use crate::rendering::{PlayerShip, RealSpaceBody, SimulationState, SolarSystemState};
use crate::runway::{
    RunwaySite, SpaceportBuild, build_spaceport, craft_extent_below, measure_runway_clearance,
    place_on_runway,
};
use crate::shipyard_editor::core::EditorPart;
use crate::staging::StagingPlan;
use crate::structures::{StructureId, StructureKind, StructurePlacement, StructureRegistry};

use super::place::place_on_launchpad;
use super::{BaseEditor, BaseEditorMode, cursor_body_dir};

/// External trigger for the launch flow: the shipyard's LAUNCH sets `arm`.
/// Consumed by [`begin_launch_flow`].
///
/// **Writer:** `shipyard_editor::ui::top_bar` (the LAUNCH button).
#[derive(Resource, Default)]
pub struct SpaceportLaunchRequest {
    pub arm: bool,
}

/// Armed by [`begin_launch_flow`] when the spaceport isn't built yet; consumed
/// by [`finish_launch_spaceport`] during the loading pass.
#[derive(Resource, Default)]
struct LaunchSpaceportBuild {
    pending: bool,
}

/// Armed by [`begin_launch_flow`] when a loading pass is needed; consumed on
/// `OnEnter(Running)` to open the god-view once the spaceport is built.
#[derive(Resource, Default)]
struct LaunchSelectPending {
    pending: bool,
}

/// The launch point the player clicked, retried by [`apply_launch_placement`]
/// until the craft's clearance can be measured.
#[derive(Resource, Default)]
struct PendingLaunch(Option<StructureId>);

/// The launch point currently under the cursor (drives the highlight tint).
#[derive(Resource, Default)]
struct LaunchHover(Option<StructureId>);

/// The hovered-launch-point fill decal — a translucent unlit overlay mesh laid
/// over the hovered runway strip / launchpad (the material "active state"), on
/// top of the gizmo outlines every launch point gets. One entity, spawned
/// lazily, re-meshed/re-posed per frame by [`update_hover_decal`] and hidden
/// whenever nothing is hovered (or the picker closes).
#[derive(Resource, Default)]
struct LaunchHoverDecal {
    entity: Option<Entity>,
    quad: Handle<Mesh>,
    disc: Handle<Mesh>,
}

/// Metres the hover decal floats above the pad plane — above the asphalt lift
/// and painted markings, below the edge posts.
const HOVER_DECAL_LIFT_M: f64 = 0.4;

/// Set when a craft is placed on a **runway** via launch-select, so its jet
/// engines get lit for throttle-only flight — the `is_runway()`-gated
/// `enable_runway_engines` does not fire under the menu's `ShipOrbit` default.
/// Launchpad (rocket) launches leave the engines off (KSP stage-key ignition).
#[derive(Resource, Default)]
struct LaunchRelightEngines {
    pending: bool,
}

/// Run condition: launch-select is the active in-world mode.
fn launch_select_active(editor: Res<BaseEditor>) -> bool {
    editor.open && editor.mode == BaseEditorMode::SelectLaunch
}

pub struct BaseEditorLaunchSelectPlugin;

impl Plugin for BaseEditorLaunchSelectPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SpaceportLaunchRequest>()
            .init_resource::<LaunchSpaceportBuild>()
            .init_resource::<LaunchSelectPending>()
            .init_resource::<PendingLaunch>()
            .init_resource::<LaunchHover>()
            .init_resource::<LaunchHoverDecal>()
            .init_resource::<LaunchRelightEngines>()
            .add_systems(
                Update,
                begin_launch_flow
                    .run_if(in_state(AppState::Running))
                    .run_if(crate::relaunch::relaunch_idle),
            )
            .add_systems(
                Update,
                finish_launch_spaceport.run_if(in_state(AppState::Loading)),
            )
            .add_systems(OnEnter(AppState::Running), open_launch_select_on_running)
            .add_systems(
                Update,
                (
                    update_launch_pick,
                    apply_launch_placement,
                    draw_launch_highlights,
                    update_hover_decal,
                )
                    .chain()
                    // Pick after the god-view camera moves so the raycast reads
                    // this frame's camera pose (see `cursor_body_dir`).
                    .after(crate::god_view::GodViewCameraSet)
                    .run_if(launch_select_active),
            )
            // Ungated: it must also *restore* the craft / hide the decal after
            // the picker closes (placement applied, or Escape-out).
            .add_systems(Update, sync_select_view_state)
            .add_systems(
                Update,
                relight_launch_engines.run_if(in_state(AppState::Running)),
            );
    }
}

/// Open the god-view launch picker over `basin_id`, clearing any stale latch.
/// The picker is a `GameContext::BaseEditor` session parented to **Flight** on
/// the return stack (the craft is already rebuilt into an orbit hold), so both
/// placing the craft and Escaping out back out to flight — "launched to fly",
/// never back into the VAB or hub it was launched from.
fn open_launch_select(
    editor: &mut BaseEditor,
    basin_id: StructureId,
    pending: &mut PendingLaunch,
    hover: &mut LaunchHover,
    next: &mut NextState<GameContext>,
    history: &mut ContextHistory,
) {
    editor.mode = BaseEditorMode::SelectLaunch;
    editor.active_site = Some(basin_id);
    pending.0 = None;
    hover.0 = None;
    history.0 = vec![GameContext::Flight];
    next.set(GameContext::BaseEditor);
}

/// Consume a [`SpaceportLaunchRequest`]: open the god-view now if the spaceport
/// exists, else run a brief loading pass to build it first. Gated on
/// `relaunch_idle` so it waits for the shipyard craft rebuild (orbit hold)
/// before acting.
#[allow(clippy::too_many_arguments)]
fn begin_launch_flow(
    mut request: ResMut<SpaceportLaunchRequest>,
    runway_site: Option<Res<RunwaySite>>,
    mut editor: ResMut<BaseEditor>,
    mut build: ResMut<LaunchSpaceportBuild>,
    mut select_pending: ResMut<LaunchSelectPending>,
    mut pending: ResMut<PendingLaunch>,
    mut hover: ResMut<LaunchHover>,
    mut tracker: ResMut<LoadingTracker>,
    mut dest: ResMut<LoadDestination>,
    mut next_state: ResMut<NextState<AppState>>,
    mut next_ctx: Option<ResMut<NextState<GameContext>>>,
    mut history: ResMut<ContextHistory>,
) {
    if !request.arm {
        return;
    }
    request.arm = false;
    if let Some(site) = runway_site.as_deref() {
        // Spaceport already built (this session) — open the picker immediately.
        if let Some(next) = next_ctx.as_mut() {
            open_launch_select(
                &mut editor,
                site.basin_id,
                &mut pending,
                &mut hover,
                next,
                &mut history,
            );
        }
        info!("launch-select: spaceport ready — choose a launch point");
    } else {
        // First launch: build the spaceport behind a brief loading pass, then
        // open the picker on the Loading → Running edge. PLACEMENT only, no
        // SETTLE — the craft is in orbit, so a settle gate keyed on its
        // body-fixed point would never resolve; the site's ground streams in
        // live under the god-view instead.
        build.pending = true;
        select_pending.pending = true;
        tracker.begin([StepDesc::new(step::PLACEMENT, "Building spaceport", 1.0)]);
        dest.0 = AppState::Running;
        next_state.set(AppState::Loading);
        info!("launch-select: building spaceport…");
    }
}

/// Build the spaceport lazily during the launch-flow loading pass, then complete
/// the PLACEMENT step so the loading screen reveals into the god-view. Retries
/// each frame until the terrain height source is resident (as `finish_runway_spawn`
/// does at boot). Idempotent-guarded by the `RunwaySite`-absent decision in
/// [`begin_launch_flow`] plus its own one-shot `build.pending`.
#[allow(clippy::too_many_arguments)]
fn finish_launch_spaceport(
    mut build: ResMut<LaunchSpaceportBuild>,
    mut sim: ResMut<SimulationState>,
    mut tracker: ResMut<LoadingTracker>,
    height_sources: Res<HeightSourceRegistry>,
    mut flatten_registry: ResMut<TerrainFlattenRegistry>,
    mut structure_registry: ResMut<StructureRegistry>,
    mut paved: ResMut<crate::base_editor::PavedFootprints>,
    mut rebuild: ResMut<crate::rendering::terrain_residency::TerrainRebuildRequest>,
    root: Res<RealSpaceRoot>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<thalos_body_render::ShadowedStandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    if !build.pending {
        return;
    }
    let body_id = sim.simulation.dominant_body();
    let Some(height_source) = height_sources.get(body_id) else {
        return; // terrain height source not resident yet — retry next frame
    };
    let hs = height_source.as_ref();
    let SpaceportBuild { site, .. } = build_spaceport(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut images,
        &mut structure_registry,
        &mut paved,
        &mut flatten_registry,
        &mut rebuild,
        &mut sim,
        hs,
        root.entity,
        body_id,
    );
    commands.insert_resource(site);
    build.pending = false;
    tracker.complete(step::PLACEMENT);
    info!("launch-select: spaceport built");
}

/// After the launch-flow loading pass reveals into `Running`, open the god-view
/// picker on the freshly-built spaceport. No-op for boot / other reveals (only
/// [`begin_launch_flow`] arms `select_pending`).
fn open_launch_select_on_running(
    mut select_pending: ResMut<LaunchSelectPending>,
    runway_site: Option<Res<RunwaySite>>,
    mut editor: ResMut<BaseEditor>,
    mut pending: ResMut<PendingLaunch>,
    mut hover: ResMut<LaunchHover>,
    mut next_ctx: Option<ResMut<NextState<GameContext>>>,
    mut history: ResMut<ContextHistory>,
) {
    if !select_pending.pending {
        return;
    }
    select_pending.pending = false;
    let Some(site) = runway_site.as_deref() else {
        warn!("launch-select: spaceport not built on Running entry — aborting");
        return;
    };
    let Some(next) = next_ctx.as_mut() else {
        return;
    };
    open_launch_select(
        &mut editor,
        site.basin_id,
        &mut pending,
        &mut hover,
        next,
        &mut history,
    );
    info!("launch-select: choose a launch point");
}

/// Raycast the cursor against the pad sphere, resolve which launch point it
/// falls on, and latch it on a left-click. Holds the latch while a placement is
/// being retried (so a mid-retry cursor move doesn't reselect).
#[allow(clippy::too_many_arguments)]
fn update_launch_pick(
    editor: Res<BaseEditor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    registry: Res<StructureRegistry>,
    mut hover: ResMut<LaunchHover>,
    mut pending: ResMut<PendingLaunch>,
    mouse: Res<ButtonInput<MouseButton>>,
    ui_gate: Res<crate::hud::UiPointerGate>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &CellCoord, &Transform), (With<ShipCamera>, With<ActiveCamera>)>,
    root_grid: Query<&Grid, With<BigSpace>>,
) {
    if pending.0.is_some() {
        return; // placement in flight — don't move the latch
    }
    let Some(basin_id) = editor.active_site else {
        return;
    };
    let Some((pad_r, body_id, dir_body)) = cursor_pad_dir(
        &sim, &solar, &registry, basin_id, &windows, &cameras, &root_grid,
    ) else {
        hover.0 = None;
        return;
    };
    let hit = launch_point_under(&registry, basin_id, body_id, dir_body, pad_r);
    hover.0 = hit;
    if let Some(id) = hit
        && !ui_gate.hovered
        && mouse.just_pressed(MouseButton::Left)
    {
        pending.0 = Some(id);
    }
}

/// Raycast the cursor against the base's pad sphere. Returns `(pad_r, body_id,
/// dir_body)` — the pad radius `radius + E`, the body, and the body-fixed hit
/// direction. Delegates the ray math to the shared, lag-free
/// [`cursor_body_dir`].
fn cursor_pad_dir(
    sim: &SimulationState,
    solar: &SolarSystemState,
    registry: &StructureRegistry,
    basin_id: StructureId,
    windows: &Query<&Window, With<PrimaryWindow>>,
    cameras: &Query<(&Camera, &CellCoord, &Transform), (With<ShipCamera>, With<ActiveCamera>)>,
    root_grid: &Query<&Grid, With<BigSpace>>,
) -> Option<(f64, BodyId, DVec3)> {
    let basin = registry.get(basin_id)?;
    let body_id = basin.body_id;
    let states = solar.states.as_deref()?;
    let body_state = states.get(body_id)?;
    let radius_m = sim.system.bodies.get(body_id)?.radius_m;
    let elevation_m = match basin.placement {
        StructurePlacement::FlattenTo { elevation_m, .. } => elevation_m,
        StructurePlacement::Drape => 0.0,
    };
    let pad_r = radius_m + elevation_m;

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
        pad_r,
    )?;
    Some((pad_r, body_id, dir_body))
}

/// The runway or launchpad on `basin_id` whose footprint the pad point `dir_body`
/// falls within, nearest (by geodesic distance to its anchor) first.
fn launch_point_under(
    registry: &StructureRegistry,
    basin_id: StructureId,
    body_id: BodyId,
    dir_body: DVec3,
    pad_r: f64,
) -> Option<StructureId> {
    let mut best: Option<(StructureId, f64)> = None;
    for site in registry.sites_on(body_id) {
        if site.parent_site != Some(basin_id) {
            continue;
        }
        let inside = match site.kind {
            StructureKind::Runway {
                half_length_m,
                half_width_m,
            } => {
                let across = site.anchor_dir.cross(site.heading_tangent).normalize();
                let offset = (dir_body - site.anchor_dir) * pad_r;
                offset.dot(site.heading_tangent).abs() <= half_length_m as f64
                    && offset.dot(across).abs() <= half_width_m as f64
            }
            StructureKind::Launchpad { radius_m } => {
                let ang = site.anchor_dir.dot(dir_body).clamp(-1.0, 1.0).acos();
                ang * pad_r <= radius_m as f64
            }
            _ => false,
        };
        if !inside {
            continue;
        }
        let dist = site.anchor_dir.dot(dir_body).clamp(-1.0, 1.0).acos() * pad_r;
        if best.is_none_or(|(_, d)| dist < d) {
            best = Some((site.id, dist));
        }
    }
    best.map(|(id, _)| id)
}

/// Place the (orbit-holding) craft onto the latched launch point, retrying until
/// its clearance can be measured, then resume 1× and close the god-view.
#[allow(clippy::too_many_arguments)]
fn apply_launch_placement(
    mut pending: ResMut<PendingLaunch>,
    mut sim: ResMut<SimulationState>,
    solar: Res<SolarSystemState>,
    registry: Res<StructureRegistry>,
    mut active_bubble: ResMut<ActiveLocalBubble>,
    mut relight: ResMut<LaunchRelightEngines>,
    mut next_ctx: Option<ResMut<NextState<GameContext>>>,
    mut history: ResMut<ContextHistory>,
    mut commands: Commands,
    ship_q: Query<(Entity, &GlobalTransform), With<PlayerShip>>,
    children_q: Query<&Children>,
    mesh_q: Query<(&GlobalTransform, &Mesh3d)>,
    meshes: Res<Assets<Mesh>>,
    // Gear geometry for the runway rest-clearance, bundled to stay under the
    // 16-param system limit (as `runway::finish_runway_spawn` does).
    gear_geometry: (
        crate::local_physics::PartColliderQuery,
        crate::local_physics::GearPartQuery,
        Query<&AttachNodes>,
        Res<crate::local_physics::GearTuning>,
    ),
) {
    let Some(target_id) = pending.0 else {
        return;
    };
    let Some(target) = registry.get(target_id).copied() else {
        pending.0 = None;
        return; // launch point vanished
    };
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let body_id = target.body_id;
    let Some(body_state) = states.get(body_id) else {
        return;
    };
    let Some(radius_m) = sim.system.bodies.get(body_id).map(|b| b.radius_m) else {
        return;
    };
    let Ok((ship_entity, ship_gt)) = ship_q.single() else {
        return; // craft not built yet — retry
    };
    let elevation_m = target
        .parent_site
        .and_then(|p| registry.get(p))
        .and_then(|s| match s.placement {
            StructurePlacement::FlattenTo { elevation_m, .. } => Some(elevation_m),
            StructurePlacement::Drape => None,
        })
        .unwrap_or(0.0);

    match target.kind {
        StructureKind::Runway { half_length_m, .. } => {
            let (parts, gear_q, host_nodes, gear_tuning) = &gear_geometry;
            let Some(clearance_m) = measure_runway_clearance(
                &sim,
                body_id,
                ship_entity,
                ship_gt,
                &children_q,
                &mesh_q,
                &meshes,
                parts,
                gear_q,
                host_nodes,
                gear_tuning,
            ) else {
                return; // gear / geometry not resident yet — retry (keep latch)
            };
            let site = RunwaySite {
                body_id,
                center_dir: target.anchor_dir,
                heading_tangent: target.heading_tangent,
                elevation_m,
                basin_id: target.parent_site.unwrap_or(target_id),
            };
            place_on_runway(
                &mut sim,
                body_state,
                &site,
                radius_m,
                half_length_m as f64,
                clearance_m,
                &mut commands,
            );
            // The placement just teleported the canonical craft. The live
            // bubble was seeded from the *pre-placement* orbit hold, and the
            // render craft would fight it (jitter / coast the stale orbit)
            // instead of sitting parked — tear it down like every other ship
            // teleport does (`finish_runway_spawn`, `place_on_launchpad`);
            // `spawn_player_avian_body` rebuilds it seeded from the placed
            // state.
            crate::scenario_menu::clear_bubble(&mut commands, &mut active_bubble);
            commands.insert_resource(crate::local_physics::ParkingBrake { engaged: true });
            commands.insert_resource(crate::local_physics::GearState { down: true });
            relight.pending = true;
        }
        StructureKind::Launchpad { .. } => {
            let Some(clearance_m) = craft_extent_below(
                ship_entity,
                ship_gt,
                &children_q,
                &mesh_q,
                &meshes,
                Vec3::NEG_Y,
            ) else {
                return; // meshes not resident yet — retry
            };
            place_on_launchpad(
                &mut sim,
                body_state,
                body_id,
                target.anchor_dir,
                target.heading_tangent,
                radius_m,
                elevation_m,
                clearance_m,
                &mut commands,
                &mut active_bubble,
            );
        }
        _ => {
            pending.0 = None;
            return; // not a launch point
        }
    }

    // Placed: resume to 1× (the placement cores are warp-neutral) and close the
    // god-view by backing out to Flight (the picker was parented to Flight). On
    // close, `apply_open_state` restores the flight view / HUD and the flight
    // camera re-acquires the now-parked craft.
    sim.simulation.warp.reset();
    pending.0 = None;
    history.0.clear();
    if let Some(next) = next_ctx.as_mut() {
        next.set(GameContext::Flight);
    }
    info!("launch-select: launched from {:?}", target.kind);
}

/// Outline every launch point on the active base, brightening the hovered one.
/// Draws through [`GodViewGizmos`] — the default gizmo group is map-only.
fn draw_launch_highlights(
    editor: Res<BaseEditor>,
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    registry: Res<StructureRegistry>,
    hover: Res<LaunchHover>,
    bodies: Query<(&RealSpaceBody, &GlobalTransform)>,
    mut gizmos: Gizmos<GodViewGizmos>,
) {
    let Some(basin_id) = editor.active_site else {
        return;
    };
    let Some(basin) = registry.get(basin_id) else {
        return;
    };
    let Some(states) = solar.states.as_deref() else {
        return;
    };
    let body_id = basin.body_id;
    let Some(body_state) = states.get(body_id) else {
        return;
    };
    let Some(radius_m) = sim.system.bodies.get(body_id).map(|b| b.radius_m) else {
        return;
    };
    let Some((_, body_gt)) = bodies.iter().find(|(rsb, _)| rsb.body_id == body_id) else {
        return;
    };
    let center_render = body_gt.translation();
    let orientation = body_state.orientation.normalize();
    let elevation_m = match basin.placement {
        StructurePlacement::FlattenTo { elevation_m, .. } => elevation_m,
        StructurePlacement::Drape => 0.0,
    };
    let pad_r = radius_m + elevation_m;

    // Body-fixed point → render space (relative to the floating origin), the same
    // large-minus-large f32 map `place::draw_placement_ghost` uses for previews.
    let to_render = |p_body: DVec3| -> Vec3 {
        center_render + (orientation * p_body).as_vec3() * SHIP_SCALE as f32
    };

    let base = Color::srgb(0.30, 0.85, 1.0);
    let hot = Color::srgb(1.0, 0.85, 0.30);

    for site in registry.sites_on(body_id) {
        if site.parent_site != Some(basin_id) {
            continue;
        }
        let color = if hover.0 == Some(site.id) { hot } else { base };
        match site.kind {
            StructureKind::Runway {
                half_length_m,
                half_width_m,
            } => {
                let across = site.anchor_dir.cross(site.heading_tangent).normalize();
                let hl = half_length_m as f64;
                let hw = half_width_m as f64;
                let corner = |a: f64, w: f64| -> Vec3 {
                    to_render(site.anchor_dir * pad_r + site.heading_tangent * a + across * w)
                };
                let c = [
                    corner(hl, hw),
                    corner(hl, -hw),
                    corner(-hl, -hw),
                    corner(-hl, hw),
                ];
                for i in 0..4 {
                    gizmos.line(c[i], c[(i + 1) % 4], color);
                }
            }
            StructureKind::Launchpad { radius_m } => {
                let center = site.anchor_dir * pad_r;
                let t1 = site.heading_tangent.normalize();
                let t2 = site.anchor_dir.cross(t1).normalize();
                const SEGS: usize = 32;
                let r = radius_m as f64;
                let mut prev = to_render(center + t1 * r);
                for i in 1..=SEGS {
                    let a = i as f64 / SEGS as f64 * std::f64::consts::TAU;
                    let p = to_render(center + (t1 * a.cos() + t2 * a.sin()) * r);
                    gizmos.line(prev, p, color);
                    prev = p;
                }
            }
            _ => {}
        }
    }
}

/// Marker for the hover fill-decal entity (see [`LaunchHoverDecal`]).
#[derive(Component)]
struct HoverDecal;

/// Show the hovered launch point's "active state": a translucent unlit fill
/// laid over the runway strip / launchpad footprint, on top of the gizmo
/// outlines. Spawns the decal entity lazily (a root-grid big_space child, like
/// the placed structures), then re-poses — and re-meshes when the hovered kind
/// changes — it each frame in the body-fixed frame.
#[allow(clippy::too_many_arguments)]
fn update_hover_decal(
    sim: Res<SimulationState>,
    solar: Res<SolarSystemState>,
    registry: Res<StructureRegistry>,
    hover: Res<LaunchHover>,
    root: Res<RealSpaceRoot>,
    mut decal: ResMut<LaunchHoverDecal>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    root_grid: Query<&Grid, With<BigSpace>>,
    mut decal_q: Query<
        (&mut Mesh3d, &mut Transform, &mut CellCoord, &mut Visibility),
        With<HoverDecal>,
    >,
) {
    let Some(entity) = decal.entity else {
        // Lazy one-time spawn; components land next frame.
        decal.quad = meshes.add(Rectangle::new(1.0, 1.0));
        decal.disc = meshes.add(Circle::new(1.0));
        let material = materials.add(StandardMaterial {
            base_color: Color::srgba(1.0, 0.85, 0.3, 0.28),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            ..default()
        });
        decal.entity = Some(
            commands
                .spawn((
                    HoverDecal,
                    Mesh3d(decal.quad.clone()),
                    MeshMaterial3d(material),
                    Transform::default(),
                    Visibility::Hidden,
                    CellCoord::ZERO,
                    ChildOf(root.entity),
                    RenderLayers::layer(SHIP_LAYER),
                    Name::new("Launch Hover Decal"),
                ))
                .id(),
        );
        return;
    };
    let Ok((mut mesh3d, mut transform, mut cell, mut vis)) = decal_q.get_mut(entity) else {
        return; // spawn commands not applied yet
    };
    let Some((mesh, scale, center_world, rot_world)) =
        hover_decal_pose(&sim, &solar, &registry, &hover, &decal)
    else {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
        return;
    };
    if mesh3d.0 != mesh {
        mesh3d.0 = mesh;
    }
    let Ok(grid) = root_grid.single() else {
        return;
    };
    let (next_cell, local) = grid.translation_to_grid(center_world);
    *cell = next_cell;
    transform.translation = local;
    transform.rotation = rot_world.as_quat();
    transform.scale = scale;
    if *vis != Visibility::Inherited {
        *vis = Visibility::Inherited;
    }
}

/// Resolve the hover decal's mesh, footprint scale, and world pose from the
/// hovered launch point, or `None` when it should hide.
fn hover_decal_pose(
    sim: &SimulationState,
    solar: &SolarSystemState,
    registry: &StructureRegistry,
    hover: &LaunchHover,
    decal: &LaunchHoverDecal,
) -> Option<(Handle<Mesh>, Vec3, DVec3, DQuat)> {
    let site = hover.0.and_then(|id| registry.get(id).copied())?;
    let states = solar.states.as_deref()?;
    let body_state = states.get(site.body_id)?;
    let radius_m = sim.system.bodies.get(site.body_id)?.radius_m;
    let elevation_m = site
        .parent_site
        .and_then(|p| registry.get(p))
        .and_then(|s| match s.placement {
            StructurePlacement::FlattenTo { elevation_m, .. } => Some(elevation_m),
            StructurePlacement::Drape => None,
        })
        .unwrap_or(0.0);
    let pad_r = radius_m + elevation_m;

    let (mesh, scale) = match site.kind {
        StructureKind::Runway {
            half_length_m,
            half_width_m,
        } => (
            decal.quad.clone(),
            Vec3::new(half_length_m * 2.0, half_width_m * 2.0, 1.0),
        ),
        StructureKind::Launchpad { radius_m } => {
            (decal.disc.clone(), Vec3::new(radius_m, radius_m, 1.0))
        }
        _ => return None,
    };

    // Rectangle / Circle meshes lie in the local XY plane (+Z normal): X runs
    // along the strip, Y across it, Z is the pad normal. `(heading, across,
    // up)` is right-handed (heading × (up × heading) = up).
    let up = site.anchor_dir;
    let heading = site.heading_tangent;
    let across = up.cross(heading).normalize();
    let basis_body = DQuat::from_mat3(&DMat3::from_cols(heading, across, up));
    let orientation = body_state.orientation.normalize();
    let center_world = body_state.position + orientation * (up * (pad_r + HOVER_DECAL_LIFT_M));
    Some((mesh, scale, center_world, orientation * basis_body))
}

/// Hide the (orbit-holding) player craft while the launch picker is open — the
/// god-view should show the base, not the craft awaiting placement — and
/// restore it (plus hide the hover decal) when the picker closes. Ungated, and
/// re-hiding every active frame rather than edge-only, because the craft can
/// finish its relaunch rebuild *after* the picker opened.
fn sync_select_view_state(
    editor: Res<BaseEditor>,
    decal: Res<LaunchHoverDecal>,
    mut was_active: Local<bool>,
    mut ships: Query<&mut Visibility, (With<PlayerShip>, Without<HoverDecal>)>,
    mut decal_vis: Query<&mut Visibility, (With<HoverDecal>, Without<PlayerShip>)>,
) {
    let active = editor.open && editor.mode == BaseEditorMode::SelectLaunch;
    if active {
        for mut vis in &mut ships {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
        }
    } else if *was_active {
        for mut vis in &mut ships {
            if *vis != Visibility::Inherited {
                *vis = Visibility::Inherited;
            }
        }
        if let Some(entity) = decal.entity
            && let Ok(mut vis) = decal_vis.get_mut(entity)
            && *vis != Visibility::Hidden
        {
            *vis = Visibility::Hidden;
        }
    }
    *was_active = active;
}

/// Light the jet engines after a launch-select **runway** placement (throttle-only
/// flight), keyed per ship root + gated on the staging plan — the same one-shot
/// shape as `runway::enable_runway_engines`, but driven by the launch-select
/// signal instead of `is_runway()` (false under the menu's `ShipOrbit` default).
fn relight_launch_engines(
    mut lit: Local<Option<Entity>>,
    mut relight: ResMut<LaunchRelightEngines>,
    ships: Query<Entity, (With<PlayerShip>, With<StagingPlan>)>,
    mut activations: Query<&mut EngineActivation, Without<EditorPart>>,
) {
    if !relight.pending {
        return;
    }
    let Ok(ship) = ships.single() else {
        return; // craft not built / staging not derived yet
    };
    if *lit == Some(ship) {
        relight.pending = false;
        return;
    }
    let mut count = 0;
    for mut activation in &mut activations {
        activation.enabled = true;
        count += 1;
    }
    if count > 0 {
        *lit = Some(ship);
        relight.pending = false;
        info!("launch-select: lit {count} engine(s) for throttle-only flight");
    }
}
