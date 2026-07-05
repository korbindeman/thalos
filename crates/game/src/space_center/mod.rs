//! Space-center hub — the KSP-style overview you land in when you press PLAY.
//!
//! Like the [base editor](crate::base_editor), the hub is an **in-world modal
//! mode** (a [`crate::sim_clock`] pause source), not an `AppState`: the real
//! planet stays visible, the sim freezes, and a shared god-view camera
//! ([`crate::god_view`]) looks down at the spaceport. It is the *hub* the
//! facility editors hang off — from it you can:
//!
//! - **EDIT BASE** → the base editor on the existing basin;
//! - **VAB** → the shipyard editor;
//! - **hover a building** to highlight it and read its name in a floating
//!   callout, then **click** an enterable facility (the VAB) to enter it — the
//!   hover picker generalises to future facilities (runway/pad launch, tracking
//!   station, admin) via the [`Facility`](crate::structures::Facility) tag.
//!
//! Entry points: the start screen's **PLAY** (which loads the spaceport world
//! and opens the hub on reveal via [`OpenSpaceCenterOnStart`]), and the in-flight
//! pause menu's **SPACE CENTER** button. Escape closes it back to flight (owned
//! by `pause_menu::handle_escape_input`).
//!
//! When a facility is entered *from* the hub, [`ReturnToSpaceCenter`] is armed so
//! closing that facility drops back into the hub rather than into flight
//! ([`restore_after_facility`]) — the KSP scene loop.

mod camera;
mod select;
mod ui;

use bevy::math::DVec3;
use bevy::prelude::*;
use thalos_physics_local::HeightSourceRegistry;
use thalos_world::BodyId;

use crate::base_editor::{BaseEditor, BaseEditorMode};
use crate::loading::{AppState, LoadingTracker, step};
use crate::rendering::ground_terrain::TerrainFlattenRegistry;
use crate::rendering::real_space::RealSpaceRoot;
use crate::rendering::{SimulationState, SolarSystemState};
use crate::runway::{RunwaySite, SpaceportBuild, build_spaceport};
use crate::shipyard_editor::ShipyardEditor;
use crate::spawn::Homeworld;
use crate::structures::{
    Facility, StructureId, StructureKind, StructurePlacement, StructureRegistry, StructureSite,
};
use crate::view::ViewMode;

/// `tile_lod_m` for the fallback surface-height focus query (no base yet).
const FOCUS_HEIGHT_LOD_M: f32 = 2.0;

/// The space-center hub. A sim-clock pause source.
///
/// **Sole writer of `open`:** the start screen's PLAY (via
/// [`OpenSpaceCenterOnStart`]), the pause menu's SPACE CENTER button + Escape,
/// the hub's own EXIT, and [`restore_after_facility`].
#[derive(Resource, Debug, Default, Clone)]
pub struct SpaceCenter {
    pub open: bool,
    /// The structure the cursor is hovering over this frame (the hover picker —
    /// [`select`](self::select), its **sole writer**). It is highlighted and
    /// labelled with a floating callout; a left-click on an enterable
    /// [`Facility`] building enters it. `None` when nothing is under the cursor.
    pub hovered: Option<StructureId>,
    /// Where **EXIT / Escape** goes: `true` when the hub is the session root
    /// (opened by PLAY, no craft flying yet) → back to the main menu; `false`
    /// when opened over a live flight (pause menu) → back to that flight.
    pub return_to_menu: bool,
}

/// Open the hub the instant PLAY's world load reveals. Mirrors
/// [`crate::shipyard_editor::OpenShipyardOnStart`]: the hub must **not** open
/// during `AppState::Loading` (it gates the `Camera` `SimStage` off, and the
/// terrain systems that complete the load want the flight camera streaming), so
/// the open is deferred to `OnEnter(Running)`.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct OpenSpaceCenterOnStart(pub bool);

/// Armed by the start screen's PLAY to build the spaceport behind the loading
/// pass (the base only — **no craft is placed**; the player launches one from
/// the VAB), then reveal into the hub. Consumed by [`finish_hub_spaceport`]
/// during `AppState::Loading`.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct HubSpaceportBuild {
    pub pending: bool,
}

/// Set when a facility (VAB / base editor) is entered *from* the hub, so closing
/// it returns to the hub rather than to flight. **Sole writer:** the hub's enter
/// helpers and [`restore_after_facility`].
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct ReturnToSpaceCenter(pub bool);

/// Run condition: the hub is open.
pub fn space_center_open(sc: Option<Res<SpaceCenter>>) -> bool {
    sc.map(|s| s.open).unwrap_or(false)
}

pub struct SpaceCenterPlugin;

impl Plugin for SpaceCenterPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SpaceCenter>()
            .init_resource::<OpenSpaceCenterOnStart>()
            .init_resource::<HubSpaceportBuild>()
            .init_resource::<ReturnToSpaceCenter>()
            .add_plugins(camera::SpaceCenterCameraPlugin)
            .add_plugins(select::SpaceCenterSelectPlugin)
            .add_plugins(ui::SpaceCenterUiPlugin)
            .add_systems(OnEnter(AppState::Running), open_on_start)
            // PLAY's spaceport build (base only) runs behind the loading screen.
            .add_systems(
                Update,
                finish_hub_spaceport.run_if(in_state(AppState::Loading)),
            )
            .add_systems(Update, (apply_open_state, restore_after_facility))
            // Enforce HUD-hidden every frame the hub is open, so a facility's
            // close-restore (which un-hides the HUD) can't leave it flashing over
            // the hub during a facility→hub handoff.
            .add_systems(Update, enforce_hud_hidden.run_if(space_center_open));
    }
}

/// Open the hub on entry to `Running` when PLAY armed it — after the world (and
/// its spaceport) has loaded, never during it. One-shot (re-armed by the start
/// screen's PLAY). PLAY is the session root, so Escape/EXIT returns to the main
/// menu (`return_to_menu`).
fn open_on_start(mut flag: ResMut<OpenSpaceCenterOnStart>, mut sc: ResMut<SpaceCenter>) {
    if flag.0 {
        sc.open = true;
        sc.hovered = None;
        sc.return_to_menu = true;
        flag.0 = false;
    }
}

/// Build the spaceport (base only, no craft) during PLAY's loading pass, then
/// complete the PLACEMENT step so the loading screen reveals into the hub.
/// Retries until the terrain height source is resident (as `finish_runway_spawn`
/// does), and no-ops if the spaceport already exists this session. Mirrors
/// `launch_select::finish_launch_spaceport` — both are thin loaders over the
/// shared `runway::build_spaceport`, differing only in what the reveal opens.
#[allow(clippy::too_many_arguments)]
fn finish_hub_spaceport(
    mut build: ResMut<HubSpaceportBuild>,
    runway_site: Option<Res<RunwaySite>>,
    homeworld: Res<Homeworld>,
    mut sim: ResMut<SimulationState>,
    mut tracker: ResMut<LoadingTracker>,
    height_sources: Res<HeightSourceRegistry>,
    mut flatten_registry: ResMut<TerrainFlattenRegistry>,
    mut structure_registry: ResMut<StructureRegistry>,
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
    // Already built this session (e.g. returned to the menu then PLAY again) —
    // just release the gate; `build_spaceport` is not idempotent.
    if runway_site.is_some() {
        build.pending = false;
        tracker.complete(step::PLACEMENT);
        return;
    }
    // The spaceport is on the homeworld (Thalos) — not `dominant_body()`, which
    // would build a nonsensical base on whatever SOI the placeholder craft is in.
    let body_id = homeworld.0;
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
    info!("space center: spaceport built (no craft placed)");
}

/// Reopen the hub when a facility entered *from* it closes (the KSP scene loop).
/// Edge-detected on the facility open→closed transition so a facility opened from
/// flight (return flag clear) drops back to flight as before. A VAB **Launch**
/// (which closes the editor *and* queues a relaunch to flight) is distinguished
/// from an Escape-out: it clears the flag and drops to flight instead of the hub.
fn restore_after_facility(
    mut return_flag: ResMut<ReturnToSpaceCenter>,
    shipyard: Res<ShipyardEditor>,
    base: Res<BaseEditor>,
    relaunch_request: Res<crate::relaunch::RelaunchRequest>,
    relaunch_in_flight: Res<crate::relaunch::RelaunchInFlight>,
    mut sc: ResMut<SpaceCenter>,
    mut facility_was_open: Local<bool>,
) {
    let facility_open = shipyard.open || base.open;
    let relaunching = relaunch_request.0.is_some() || relaunch_in_flight.active();
    // A VAB Launch means "go fly": disarm the return-to-hub flag the moment a
    // relaunch is queued, not only when its window happens to overlap the
    // facility-close edge below. The edge-only check raced the launch flow —
    // VAB close → relaunch frames → the SelectLaunch picker opens (also a
    // "facility") — and when the edge fell outside the relaunch window the
    // stale flag survived into the picker session, whose close then bounced
    // the freshly placed flight straight back into the hub.
    if relaunching && return_flag.0 {
        return_flag.0 = false;
    }
    if *facility_was_open && !facility_open && return_flag.0 && !sc.open {
        sc.open = true;
        sc.hovered = None;
        return_flag.0 = false;
    }
    *facility_was_open = facility_open;
}

/// React to the open/close *edge*: force ship view (the god-view is a 3D view, so
/// the hub must not run over the orbital map) and, on close, restore the previous
/// view + the flight HUD. Mirrors [`crate::base_editor`]'s transition — the planet
/// stays visible and the shared god-view camera (gated `space_center_open`)
/// repositions the ship camera in place; on close the flight camera systems
/// (`SimStage::Camera`) un-gate and take the camera back.
///
/// HUD *hiding* while open is owned by [`enforce_hud_hidden`] (per-frame), not
/// here; this only *restores* the HUD on close — and only when returning to
/// flight, not when handing off to a facility (which keeps it hidden).
fn apply_open_state(
    sc: Res<SpaceCenter>,
    base: Res<BaseEditor>,
    shipyard: Res<ShipyardEditor>,
    mut view: ResMut<ViewMode>,
    mut last_open: Local<bool>,
    mut prev_view: Local<Option<ViewMode>>,
    mut hud: ParamSet<(
        Query<&mut Visibility, With<crate::hud::HudPanel>>,
        Query<&mut Visibility, With<crate::photo_mode::HideInPhotoMode>>,
    )>,
) {
    if sc.open == *last_open {
        return;
    }
    *last_open = sc.open;

    if sc.open {
        *prev_view = Some(*view);
        if *view != ViewMode::Ship {
            *view = ViewMode::Ship;
        }
        return;
    }

    // Closing.
    if let Some(prev) = prev_view.take()
        && *view != prev
    {
        *view = prev;
    }
    // Restore the flight HUD only when returning to flight, not when handing off
    // to a facility (base editor / shipyard) that keeps it hidden.
    if !base.open && !shipyard.open {
        set_hud_visibility(&mut hud, Visibility::Inherited);
    }
}

/// Hide the flight HUD every frame the hub is open (see [`apply_open_state`]).
fn enforce_hud_hidden(
    mut hud: ParamSet<(
        Query<&mut Visibility, With<crate::hud::HudPanel>>,
        Query<&mut Visibility, With<crate::photo_mode::HideInPhotoMode>>,
    )>,
) {
    set_hud_visibility(&mut hud, Visibility::Hidden);
}

/// Set both flight-HUD visibility sets (panels + photo-mode-hidden overlays),
/// change-guarded.
fn set_hud_visibility(
    hud: &mut ParamSet<(
        Query<&mut Visibility, With<crate::hud::HudPanel>>,
        Query<&mut Visibility, With<crate::photo_mode::HideInPhotoMode>>,
    )>,
    target: Visibility,
) {
    for mut vis in hud.p0().iter_mut() {
        if *vis != target {
            *vis = target;
        }
    }
    for mut vis in hud.p1().iter_mut() {
        if *vis != target {
            *vis = target;
        }
    }
}

/// The world-space frame + pad geometry the hub is looking at. Shared by the
/// camera (focus) and the building-selection raycast.
pub(crate) struct HubContext {
    pub body_id: BodyId,
    /// Focus point in heliocentric metres (the pad centre or the ship-surface
    /// fallback).
    pub center_world: DVec3,
    /// Local vertical at the focus (world-space, unit).
    pub up_world: DVec3,
    /// Surface radius of the pad (`body radius + flatten elevation`), metres —
    /// the sphere the selection raycast hits.
    pub pad_r: f64,
}

/// Resolve the hub's focus: the spaceport basin's flattened centre if a base
/// exists on the **homeworld**, otherwise the surface point under the ship (so
/// SPACE CENTER opened from orbit still frames *something*). `None` before body
/// state is available.
///
/// Anchored to `body_id` = the homeworld (Thalos), **not** `dominant_body()`:
/// the Space Center is always on the homeworld, and the unused placeholder craft
/// can drift out of the homeworld's SOI (its dominant body then resolves to the
/// star) after `build_spaceport` jumps the sim clock to the morning epoch.
pub(crate) fn hub_context(
    sim: &SimulationState,
    solar: &SolarSystemState,
    height_sources: &HeightSourceRegistry,
    registry: &StructureRegistry,
    body_id: BodyId,
) -> Option<HubContext> {
    let states = solar.states.as_deref()?;
    let body_state = states.get(body_id)?;
    let radius_m = sim.system.bodies.get(body_id)?.radius_m;

    if let Some(site) = home_base_site(registry, body_id) {
        let up_world = (body_state.orientation * site.anchor_dir).normalize();
        let elevation_m = match site.placement {
            StructurePlacement::FlattenTo { elevation_m, .. } => elevation_m,
            StructurePlacement::Drape => 0.0,
        };
        let pad_r = radius_m + elevation_m;
        return Some(HubContext {
            body_id,
            center_world: body_state.position + up_world * pad_r,
            up_world,
            pad_r,
        });
    }

    // Fallback: the surface directly under the ship (no base yet).
    let ship_pos = sim.simulation.ship_state().position;
    let dir_world = (ship_pos - body_state.position).normalize_or_zero();
    if dir_world == DVec3::ZERO {
        return None;
    }
    let dir_body = (body_state.orientation.inverse() * dir_world).normalize();
    let height_m = height_sources
        .get(body_id)
        .and_then(|hs| hs.sample_height_m(dir_body.as_vec3(), FOCUS_HEIGHT_LOD_M))
        .unwrap_or(0.0) as f64;
    let pad_r = radius_m + height_m.max(0.0);
    Some(HubContext {
        body_id,
        center_world: body_state.position + dir_world * pad_r,
        up_world: dir_world,
        pad_r,
    })
}

/// The home base site on `body_id` (the spaceport basin), if any.
pub(crate) fn home_base_site(registry: &StructureRegistry, body_id: BodyId) -> Option<StructureSite> {
    registry
        .sites_on(body_id)
        .iter()
        .find(|s| matches!(s.kind, StructureKind::BaseSite))
        .copied()
}

/// Footprint bounding radius (m) for a hoverable structure, or `None` for kinds
/// that aren't pickable in the hub (the runway and the invisible base site).
pub(crate) fn selectable_bound(kind: &StructureKind) -> Option<f64> {
    match kind {
        StructureKind::Building {
            half_x_m, half_z_m, ..
        } => Some(half_x_m.hypot(*half_z_m) as f64),
        StructureKind::Launchpad { radius_m } => Some(*radius_m as f64),
        StructureKind::Tank { radius_m, .. } => Some(*radius_m as f64),
        StructureKind::Runway { .. } | StructureKind::BaseSite => None,
    }
}

/// Human-readable fallback name for a non-facility structure kind, shown in the
/// hover callout when the structure carries no [`Facility`] tag.
pub(crate) fn kind_name(kind: &StructureKind) -> &'static str {
    match kind {
        StructureKind::Building { .. } => "BUILDING",
        StructureKind::Launchpad { .. } => "LAUNCHPAD",
        StructureKind::Tank { .. } => "STORAGE TANK",
        StructureKind::Runway { .. } => "RUNWAY",
        StructureKind::BaseSite => "BASE SITE",
    }
}

/// Enter a facility from the hub: open it, arm the return-to-hub flag, and close
/// the hub. Only the VAB (→ shipyard) is wired today.
pub(crate) fn enter_facility(
    facility: Facility,
    sc: &mut SpaceCenter,
    shipyard: &mut ShipyardEditor,
    return_flag: &mut ReturnToSpaceCenter,
) {
    match facility {
        Facility::Vab => shipyard.open = true,
    }
    return_flag.0 = true;
    sc.open = false;
    sc.hovered = None;
    info!("space center: entering {facility:?}");
}

/// Enter the base editor from the hub, on the existing base site if there is one
/// (straight to placing buildings) or picking a new site otherwise.
pub(crate) fn enter_base_editor(
    sc: &mut SpaceCenter,
    base: &mut BaseEditor,
    return_flag: &mut ReturnToSpaceCenter,
    base_site: Option<StructureId>,
) {
    base.open = true;
    if let Some(id) = base_site {
        base.mode = BaseEditorMode::PlaceBuildings;
        base.active_site = Some(id);
    } else {
        base.mode = BaseEditorMode::PickSite;
        base.active_site = None;
    }
    return_flag.0 = true;
    sc.open = false;
    sc.hovered = None;
    info!("space center: entering base editor");
}
