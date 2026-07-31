//! Player ship rendering — loads the spawn situation's ship blueprint on
//! startup (`ships/apollo.ron` by default, `ships/meridian.ron` for the runway
//! scenarios; see [`crate::spawn::SpawnSituation::ship_blueprint_path`]),
//! spawns its parts as children of a [`PlayerShip`] root, and keeps the root's
//! world position in sync with the physics ship state each frame.
//!
//! Ship parts are authored in metres. The root's scale is kept at
//! [`Vec3::ONE`]; in ship view the global [`WorldScale`] is `1.0`, so a
//! part's metre-sized mesh vertices end up in real-metre render units.
//! In map view [`WorldScale`] flips to `1e-6` and the ship collapses to
//! sub-unit size, but ship entities carry [`HideInMapView`] so they are
//! hidden anyway. Part meshes and the [`ShipPartMaterial`] uniforms are
//! rebuilt whenever `AttachNodes` changes and kept in sync via ported
//! versions of the editor's systems.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;

use crate::shipyard_editor::core::EditorPart;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::math::DVec3;
use bevy::mesh::Mesh;
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_physics_canonical::canonical::CraftId;
use thalos_physics_canonical::types::ShipParameters;
use thalos_shipyard::{
    Adapter, AirIntake, AttachNodes, Attachment, CommandPod, ControlSurfaceRole, Decoupler, Engine,
    EngineGeometry, FuelTank, Fuselage, Gear, JetNacelleMount, Part, PartCatalog, PartMaterial,
    PodGeometry, Ship, ShipBlueprint, ShipyardPlugin, SurfaceMount, SurfaceMountKind, Wing,
    build_cockpit_mesh, build_control_surface_mesh, build_fuselage_mesh, build_gear_mesh,
    build_gear_struct_mesh, build_jet_nacelle_body_mesh, build_jet_nacelle_pylon_mesh,
    build_wing_fairing_mesh, build_wing_mesh, host_mount_geometry, jet_nacelle_length,
    pod_visual_profile, wants_wing_fairing,
};

use thalos_body_render::{
    CraftRenderPlugin, ShadowedStandardMaterial, ShipPartExtension, ShipPartMaterial,
    landing_gear_base, shadowed, ship_part_params, stainless_steel_base,
};

use crate::SimStage;
use crate::camera::{CameraFocus, CameraTargetOffset, find_reference_body};
use crate::game_context::GameContext;
use crate::rendering::{CelestialBody, PlayerShip, ShipMarker, SimulationState, SolarSystemState};
use crate::view::{HideInMapView, HideInShipView, ViewMode};

/// Radial segments for cylinder / frustum part meshes. Matches the ship
/// editor's value so the two look identical side-by-side.
const PART_RESOLUTION: u32 = 128;

/// Whole-craft crash tolerance (m/s of surface-relative **sink rate** — the
/// into-surface radial component, not total speed, so a fast wheels-first
/// runway touchdown is judged by its descent, never its ground speed).
/// A terrain contact above this destroys the vessel. Forgiving first-slice
/// constant — a future per-part model derives it from the contacting parts.
/// See `docs/simulation/surface.md`.
const SHIP_IMPACT_TOLERANCE_M_S: f64 = 12.0;
/// Aggregate bluff-body drag coefficient for the player ship. Blunt
/// capsule-topped stacks sit around 1.0; a future per-shape model can derive
/// this from the nose part. The frontal area is per-vehicle
/// (`ShipStats::frontal_area_m2`).
const SHIP_DRAG_COEFFICIENT: f64 = 1.0;

/// Initial orbital distance (metres) when switching into ship view. The
/// camera snaps to this distance — close enough that a ~10 m ship fills
/// a reasonable fraction of the screen.
const SHIP_VIEW_INITIAL_DISTANCE_M: f64 = 30.0;

pub struct ShipViewPlugin;

impl Plugin for ShipViewPlugin {
    fn build(&self, app: &mut App) {
        let catalog = match PartCatalog::load_from_path("assets/parts.ron") {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to load parts catalog from assets/parts.ron: {e}");
                // Continue with an empty catalog so the rest of the app
                // can still come up; spawn_player_ship will log and skip.
                PartCatalog {
                    parts: Default::default(),
                }
            }
        };
        if !app.is_plugin_added::<CraftRenderPlugin>() {
            app.add_plugins(CraftRenderPlugin);
        }
        app.insert_resource(catalog)
            .add_plugins(ShipyardPlugin)
            // World-keyed, not Startup: a bare menu boot defers the world
            // (`WorldState::Absent`), and the menu sets the chosen scenario's
            // `SpawnSituation` + vessel kind *before* flipping `Live`, so this
            // builds the right blueprint (or EVA-skips) when it fires.
            .add_systems(OnEnter(crate::loading::WorldState::Live), spawn_player_ship)
            // Re-acquire the craft when returning to flight from any in-world
            // modal (launch-select / hub / base editor / VAB). Without this the
            // ship camera keeps whatever focus the modal left — on a session's
            // first launch, the world-spawn `Body(homeworld)` focus — and flies
            // off the placed craft until a view toggle happens to fix it.
            .add_systems(OnEnter(GameContext::Flight), reacquire_flight_focus)
            .add_systems(
                Update,
                (
                    rebuild_ship_visuals,
                    rebuild_ship_wing_visuals,
                    animate_ship_control_surfaces.after(rebuild_ship_wing_visuals),
                    rebuild_ship_nacelle_visuals,
                    rebuild_ship_gear_visuals,
                    sync_gear_visibility.after(rebuild_ship_gear_visuals),
                    // After the physics set so it reads this frame's
                    // compression, not last frame's.
                    sync_gear_compression
                        .after(rebuild_ship_gear_visuals)
                        .after(SimStage::Physics),
                    update_ship_part_transforms.after(rebuild_ship_visuals),
                    update_ship_part_shader_params.after(rebuild_ship_visuals),
                    update_ship_camera_offset.after(update_ship_part_transforms),
                    sync_view_mode_changed
                        .run_if(resource_changed::<ViewMode>)
                        .before(crate::SimStage::Physics),
                    update_player_ship_world_position
                        .in_set(SimStage::Sync)
                        .after(crate::rendering::update_render_origin),
                ),
            );
    }
}

/// Marker on the child mesh entity rendered for each ship part. The
/// engine-tint system in `crate::engine` queries this to find the
/// material it should mutate.
#[derive(Component)]
pub(crate) struct PartVisual;

/// Root of one rendered canonical craft. [`PlayerShip`] marks the selected
/// root; every other root remains a real visible vessel in the same scene.
#[derive(Component)]
pub(crate) struct CraftRoot;

/// Stable link from a rendered craft root or part to canonical fleet state.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CraftIdentity(pub CraftId);

/// Ownership of a flight part. Aggregations that affect player controls and
/// canonical active-vessel parameters must filter to the active [`CraftId`].
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CraftPart(pub CraftId);

/// Marker on the gear gearbox's rendered mesh child, so
/// [`sync_gear_visibility`] can hide it when the gear is retracted
/// (`local_physics::GearState`). Binary show/hide — no retraction animation.
#[derive(Component)]
pub(crate) struct GearVisual;

/// A hinged control-surface sub-mesh of a wing. Carries everything
/// [`animate_ship_control_surfaces`] needs to deflect it from the
/// fly-by-wire command: which command axis drives it (`role`), the
/// per-side differential sign for ailerons, the host-local hinge frame, and
/// the deflection limit. Built by [`rebuild_ship_wing_visuals`].
#[derive(Component)]
struct ControlSurfaceVisual {
    role: ControlSurfaceRole,
    /// +1 / −1 by mount side, so a roll command deflects the left and right
    /// ailerons in opposite senses (differential). Symmetric surfaces use 1.
    side_sign: f32,
    /// Host-local hinge axis (consistently oriented so +θ = trailing edge
    /// down on both panels — see `control_surface_geometry`).
    hinge_axis: Vec3,
    /// Host-local hinge anchor; the surface entity's local translation.
    hinge_anchor: Vec3,
    /// Maximum deflection magnitude, radians.
    max_deflection: f32,
}

#[derive(Component, Clone)]
struct PartShaderHandle(Handle<ShipPartMaterial>);

pub(crate) fn spawn_player_ship(
    mut commands: Commands,
    view: Res<ViewMode>,
    situation: Res<crate::spawn::SpawnSituation>,
    mut sim: ResMut<SimulationState>,
    catalog: Res<PartCatalog>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
) {
    // KSP-style vessel split: `VesselKind::Eva` means the player is on
    // foot, controlling a single-part "EVA vessel" instead of a rocket.
    // The ship visual and ship-derived collider belong only to the
    // `Ship` path; the EVA path spawns its own capsule body in
    // `local_physics::spawn_player_avian_body`.
    if sim.simulation.vessel_kind() == thalos_physics_canonical::types::VesselKind::Eva {
        return;
    }
    let Some(blueprint) = load_blueprint_from_path(situation.ship_blueprint_path()) else {
        return;
    };

    build_player_ship(
        &mut commands,
        &view,
        &blueprint,
        &mut sim,
        &catalog,
        &mut meshes,
        &mut std_materials,
    );
}

/// Load a ship blueprint RON from a workspace-relative path (e.g.
/// `ships/meridian.ron`). Logs and returns `None` on read/parse failure.
/// Shared by the startup spawn and the start screen's scenario starter
/// ([`crate::main_menu`]), which swaps craft per chosen scenario.
pub(crate) fn load_blueprint_from_path(path: &str) -> Option<ShipBlueprint> {
    let ron_path = PathBuf::from(path);
    let text = match std::fs::read_to_string(&ron_path) {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to read {}: {}", ron_path.display(), e);
            return None;
        }
    };
    match ShipBlueprint::from_ron(&text) {
        Ok(bp) => Some(bp),
        Err(e) => {
            error!("Failed to parse {}: {}", ron_path.display(), e);
            None
        }
    }
}

/// Build the rendered + physics-registered player craft from a parsed
/// blueprint: push `ShipParameters` / aero into the simulation, spawn the
/// part tree under a fresh [`PlayerShip`], and the map-view billboard.
///
/// Shared by the startup [`spawn_player_ship`] and the editor's Launch
/// relaunch ([`crate::relaunch::finish_relaunch`]). The caller has already
/// set the canonical craft **state** (where it flies); this sets **what**
/// it is. The spawned parts carry no [`crate::shipyard_editor::core::EditorPart`]
/// marker, so — unlike the editor's build — they enter the flight
/// aggregations (fuel, staging, inertia, colliders).
pub(crate) fn build_player_ship(
    commands: &mut Commands,
    view: &ViewMode,
    blueprint: &ShipBlueprint,
    sim: &mut SimulationState,
    catalog: &PartCatalog,
    meshes: &mut Assets<Mesh>,
    std_materials: &mut Assets<StandardMaterial>,
) {
    // Push spawn-time MOI + reaction-wheel torque into the physics
    // simulation so attitude integration knows what we're flying. Active
    // thrust, mass flow, dry mass, and wet mass are refreshed each frame
    // by `fuel.rs` from enabled engines and live tank state.
    let stats = match blueprint.stats(catalog) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to compute ship stats: {e}");
            return;
        }
    };
    sim.simulation.set_ship_params(ShipParameters {
        moment_of_inertia: stats.moment_of_inertia_kg_m2,
        center_of_mass: stats.center_of_mass_m,
        max_torque: DVec3::splat(stats.max_reaction_torque_n_m),
        // Filled live each frame by `staging::recompute_ship_inertia` (same as
        // max_torque / MOI — it tracks the CoM as fuel burns); ZERO until then.
        gimbal_torque_full: DVec3::ZERO,
        thrust_n: 0.0,
        mass_flow_kg_per_s: 0.0,
        dry_mass_kg: stats.dry_mass_kg,
        // Whole-craft crash tolerance (forgiving first-slice constant; a
        // future per-part model derives this from the contacting parts).
        // See docs/simulation/surface.md.
        impact_tolerance_m_s: SHIP_IMPACT_TOLERANCE_M_S,
        // Per-vehicle aerodynamic drag: frontal area from the actual part
        // geometry, blunt-body Cd. See docs/simulation/aerodynamics.md.
        reference_area_m2: stats.frontal_area_m2,
        drag_coefficient: SHIP_DRAG_COEFFICIENT,
    });
    sim.simulation.set_ship_mass(stats.wet_mass_kg());
    info!(
        target: "thalos::diagnostic::craft",
        event = "parameters",
        inertia_x_kg_m2 = stats.moment_of_inertia_kg_m2.x,
        inertia_y_kg_m2 = stats.moment_of_inertia_kg_m2.y,
        inertia_z_kg_m2 = stats.moment_of_inertia_kg_m2.z,
        max_torque_n_m = stats.max_reaction_torque_n_m,
        dry_mass_kg = stats.dry_mass_kg,
        wet_mass_kg = stats.wet_mass_kg(),
        "craft parameters"
    );

    // Whole-body aero config from the blueprint's wing parts (lift + stability
    // + per-surface control authority, with moment arms about the real CoM),
    // or a bluff-body drag config for a wingless craft. Consumed by
    // `aero::attach_ship_aero` when the Avian body spawns.
    match blueprint.wing_aero_panels(catalog) {
        Ok(panels) => {
            let config = crate::aero::build_ship_aero_config(
                &panels,
                stats.frontal_area_m2,
                SHIP_DRAG_COEFFICIENT,
                stats.center_of_mass_m,
            );
            info!(
                target: "thalos::diagnostic::craft",
                event = "aero_configuration",
                wing_panels = panels.len(),
                reference_area_m2 = config.reference_area_m2,
                reference_chord_m = config.reference_chord_m,
                reference_span_m = config.reference_span_m,
                lift_slope = config.lift_slope,
                "craft aerodynamic configuration"
            );
            commands.insert_resource(crate::aero::ShipAeroLayout { config });
        }
        Err(err) => error!("Failed to compute wing aero panels: {err}"),
    }

    let ship_entity = match blueprint.spawn(commands, catalog) {
        Ok(e) => e,
        Err(err) => {
            error!("Failed to spawn ship blueprint: {err}");
            return;
        }
    };
    info!(
        "spawned ship blueprint '{}' with {} parts",
        blueprint.name,
        blueprint.parts.len(),
    );

    // Visibility is driven by `apply_view_mode_visibility` via the
    // [`HideInMapView`] tag — start at whatever the tag implies for the
    // current view. Transform is overwritten every frame by
    // `update_player_ship_world_position`, so the initial value is
    // arbitrary.
    let initial_visibility = match *view {
        ViewMode::Map => Visibility::Hidden,
        ViewMode::Ship => Visibility::Inherited,
    };

    // Default the instance name to the blueprint's authored name. Both
    // the ship-view root and the map-view billboard carry the same name
    // so UI surfaces (body tree, focus indicator, debug picker) display
    // it consistently regardless of which entity is the focus target.
    let ship_name = blueprint.name.clone();
    let craft_id = sim.simulation.active_craft_id();

    let player_ship = commands
        .spawn((
            PlayerShip,
            CraftRoot,
            CraftIdentity(craft_id),
            HideInMapView,
            Transform::IDENTITY,
            initial_visibility,
            // Pivot the camera around the ship's centre of mass, recomputed
            // every frame by `update_ship_camera_offset` so it tracks staging
            // and design changes.
            CameraTargetOffset::default(),
            Name::new(ship_name.clone()),
        ))
        .id();
    // Map-view billboard for this ship. Position and scale are overwritten
    // every frame by `update_ship_position` (in `rendering.rs`), so the
    // initial transform is a placeholder. Material is unique per ship so
    // future per-ship marker styling (colour-by-faction, IFF tags, etc.)
    // doesn't bleed across instances.
    let marker_icon = meshes.add(Circle::new(1.0));
    let marker_material = std_materials.add(StandardMaterial {
        base_color: Color::WHITE,
        emissive: LinearRgba::WHITE * 2.0,
        unlit: true,
        double_sided: true,
        alpha_mode: AlphaMode::Blend,
        // Match body icon billboards: a small tie-breaker for same-depth
        // transparent markers, without bypassing normal depth occlusion.
        depth_bias: 10.0,
        ..default()
    });
    commands.spawn((
        Mesh3d(marker_icon),
        MeshMaterial3d(marker_material),
        Transform::IDENTITY,
        // Updated every frame by `update_ship_position` based on view mode,
        // photo mode, and the ship's current local system.
        Visibility::Hidden,
        ShipMarker,
        HideInShipView,
        NotShadowCaster,
        NotShadowReceiver,
        Name::new(ship_name),
    ));

    // Reparent all parts owned by this ship into the PlayerShip hierarchy
    // so they inherit its scale + translation. Runs as a deferred command
    // so the `Ship` component is committed first.
    commands.queue(move |world: &mut World| {
        let root = world
            .get::<Ship>(ship_entity)
            .map(|s| s.root)
            .unwrap_or(ship_entity);
        let mut attachments: HashMap<Entity, Vec<Entity>> = HashMap::new();
        let mut att_query = world.query::<(Entity, &Attachment)>();
        for (e, att) in att_query.iter(world) {
            attachments.entry(att.parent).or_default().push(e);
        }
        // Surface-mounted parts (wings) connect via `SurfaceMount`, not
        // `Attachment` — include them so they reparent under the ship too.
        let mut sm_query = world.query::<(Entity, &SurfaceMount)>();
        for (e, sm) in sm_query.iter(world) {
            attachments.entry(sm.parent).or_default().push(e);
        }
        let mut queue: VecDeque<Entity> = VecDeque::from([root]);
        let mut to_reparent: Vec<Entity> = Vec::new();
        while let Some(e) = queue.pop_front() {
            to_reparent.push(e);
            if let Some(kids) = attachments.get(&e) {
                queue.extend(kids.iter().copied());
            }
        }
        // Seat the PlayerShip root into the BigSpace hierarchy *here*, in the
        // same exclusive-world step that attaches its parts — not lazily via
        // `attach_player_ship_to_big_space` (Update), which lands the root's
        // `CellCoord` a frame *after* this reparent. `Grid::tag_low_precision_roots`
        // only marks a low-precision child as a `LowPrecisionRoot` on the frame its
        // `ChildOf` changes, and only if the parent is already a valid high-precision
        // parent (has a `CellCoord`). If the root is still un-seated on that frame the
        // parts miss the tag and are never re-evaluated (the query re-fires only on
        // `Changed<ChildOf>`/`Added<Transform>`), so they render un-propagated and trip
        // the big_space hierarchy validator ("child of a Non-root high precision spatial
        // entity …"). Boot craft happened to win this race; the runtime relaunch /
        // VAB-launch path loses it. Seating the root before the children attach makes it
        // a valid parent immediately. Idempotent with the Update fallback (whose
        // `Without<CellCoord>` filter skips an already-seated root).
        if let Some(real_root) = world
            .get_resource::<crate::rendering::real_space::RealSpaceRoot>()
            .map(|r| r.entity)
        {
            world
                .entity_mut(player_ship)
                .insert((CellCoord::ZERO, ChildOf(real_root)));
        }
        for part in &to_reparent {
            world.entity_mut(*part).insert(CraftPart(craft_id));
            world.entity_mut(player_ship).add_child(*part);
        }
    });
}

/// Sync every rendered craft root to its canonical fleet state. The active
/// [`PlayerShip`] gets local-physics render extrapolation; detached OnRails
/// vessels use their directly propagated canonical pose.
fn update_player_ship_world_position(
    sim: Res<SimulationState>,
    authority: Res<crate::local_physics::AvianAuthority>,
    clock: Res<crate::sim_clock::SimClock>,
    fixed_time: Res<Time<Fixed>>,
    grid: Query<&Grid, With<BigSpace>>,
    mut query: Query<
        (
            &CraftIdentity,
            Has<PlayerShip>,
            &mut CellCoord,
            &mut Transform,
        ),
        With<CraftRoot>,
    >,
) {
    let Ok(root_grid) = grid.single() else {
        return;
    };
    let active_id = sim.simulation.active_craft_id();

    // Render extrapolation for the Avian-owned regime (powered descent /
    // landing). Avian integrates the craft at a fixed timestep, but this system
    // and the renderer run at the (variable, usually higher) frame rate, so the
    // canonical position read back from Avian each frame holds for several
    // frames then jumps a step. The camera rigidly follows the ship, so that
    // hold/jump shows up as the *terrain* stuttering at the viewer's feet
    // (close-range parallax) while the ship and the parallax-free sky look
    // steady — the "terrain jitter" bug. Advance the *rendered* position by the
    // body-relative velocity across the fixed-step overstep so the ship (and
    // hence the camera and terrain) move smoothly between physics steps.
    //
    // Only the **surface-relative** velocity is extrapolated: the canonical
    // velocity is heliocentric (~30 km/s orbital) *plus* the planet co-rotation
    // (`ω×r`, ~260 m/s at a low-latitude site), and BOTH of those components
    // already advance smoothly — the readback re-converts the surface-local
    // Avian state with `body_state` at the *current* sim_time every frame, so
    // orbital motion and co-rotation are analytic. Only the craft's motion
    // relative to the ground stutters at the fixed physics tick. Subtracting
    // just `body.velocity` here (the pre-SLF form) double-counted `ω×r` and
    // injected a ground-speed-independent ~4 m sawtooth into the rendered pose —
    // the camera follows the ship, so it showed up as the *terrain and runway*
    // shaking violently while rolling. Physics, canonical state, and the terrain
    // collider are untouched. Kepler-owned coast (`OnRails`/`AttitudeOnly`)
    // already advances once per render frame, so it is left alone.
    //
    // Suppressed while the sim is paused: `Time<Fixed>` keeps accumulating
    // overstep across the escape pause (only `Time<Physics>`/`SimClock` halt),
    // so extrapolating a frozen canonical position by an ever-cycling overstep
    // makes the parked craft visibly jitter in the pause menu. When paused, the
    // overstep is meaningless — render the canonical position directly.
    for (identity, is_player_ship, mut cell, mut transform) in query.iter_mut() {
        let Some(vessel) = sim.simulation.vessel(identity.0) else {
            continue;
        };
        let ship = vessel.state();
        let mut position = ship.translation.position;
        if identity.0 == active_id
            && is_player_ship
            && authority.owns_translation()
            && !clock.is_paused()
        {
            let body_id = sim.simulation.dominant_body();
            let body = sim.ephemeris.state(
                body_id,
                thalos_physics_canonical::canonical::Epoch(sim.simulation.sim_time()),
            );
            let surface_velocity =
                body.velocity + body.angular_velocity.cross(position - body.position);
            let rel_velocity = ship.translation.velocity - surface_velocity;
            position += rel_velocity * fixed_time.overstep().as_secs_f64();
        }

        let (next_cell, local) = root_grid.translation_to_grid(position);
        *cell = next_cell;
        transform.translation = local;
        transform.rotation = ship.attitude.orientation.as_quat();
    }
}

/// React to [`ViewMode`] changes: switch the camera focus and snap it to
/// the right anchor for each view. Projection no longer needs swapping —
/// each camera carries its own fixed projection (see `spawn_camera`).
fn sync_view_mode_changed(
    view: Res<ViewMode>,
    sim: Res<SimulationState>,
    body_states: Res<SolarSystemState>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    mut focus: ResMut<CameraFocus>,
    bodies: Query<&CelestialBody>,
) {
    establish_flight_focus(
        *view,
        &sim,
        &body_states,
        player.as_deref(),
        &mut focus,
        &bodies,
    );
}

/// Re-acquire the flight camera focus on entry to [`GameContext::Flight`] — i.e.
/// returning to flying from an in-world modal (launch-select, the space-center
/// hub, the base editor, or the VAB). Those modes leave [`CameraFocus::target`]
/// on whatever they last used: the launch flow, on a session's *first* launch,
/// inherits the world-spawn `focus_camera_on_homeworld` `Body(homeworld)` focus,
/// so the ship camera would keep orbiting the planet's map-billboard position
/// instead of the just-placed craft (it "flies away"). Unlike a map↔ship toggle,
/// no [`ViewMode`] change fires here, so [`sync_view_mode_changed`] never runs to
/// correct it — hence this explicit re-acquire on the context edge. It applies
/// the same establishing framing a view toggle would.
fn reacquire_flight_focus(
    view: Res<ViewMode>,
    sim: Res<SimulationState>,
    body_states: Res<SolarSystemState>,
    player: Option<Res<crate::player_controller::PlayerControllerState>>,
    mut focus: ResMut<CameraFocus>,
    bodies: Query<&CelestialBody>,
) {
    establish_flight_focus(
        *view,
        &sim,
        &body_states,
        player.as_deref(),
        &mut focus,
        &bodies,
    );
}

/// Establish the flight camera focus for `view`: chase the craft (or the on-foot
/// player) in ship view, frame the SOI body in map view. The single home for the
/// "what should the flight camera look at" decision — shared by
/// [`sync_view_mode_changed`] (a view toggle) and [`reacquire_flight_focus`] (a
/// return to flight).
fn establish_flight_focus(
    view: ViewMode,
    sim: &SimulationState,
    body_states: &SolarSystemState,
    player: Option<&crate::player_controller::PlayerControllerState>,
    focus: &mut CameraFocus,
    bodies: &Query<&CelestialBody>,
) {
    match view {
        ViewMode::Ship => {
            let player_active = player.map(|state| state.is_active()).unwrap_or(false);
            focus.target = if player_active {
                crate::camera::CameraFocusTarget::PlayerController
            } else {
                crate::camera::CameraFocusTarget::Ship
            };
            let target_distance = if player_active {
                6.0
            } else {
                SHIP_VIEW_INITIAL_DISTANCE_M
            };
            focus.target_distance = target_distance;
            focus.distance = focus.distance.min(target_distance * 10.0);
            // Default view: behind the ship, slight tilt above the horizon.
            // Azimuth = π puts the camera at -forward, where `forward` is the
            // gravity-frame's horizon-projected prograde — KSP's chase angle.
            focus.azimuth = std::f32::consts::PI;
            focus.elevation = 0.15;
        }
        ViewMode::Map => {
            // Focus the body whose SOI currently contains the ship — the
            // same anchor the propagator uses. Falls back silently if the
            // body-state cache hasn't populated yet (first frame).
            let Some(states) = body_states.states.as_deref() else {
                return;
            };
            let ship_pos = sim.simulation.ship_state().position;
            let soi_id = find_reference_body(ship_pos, sim.simulation.bodies(), states);
            if bodies.iter().any(|b| b.body_id == soi_id) {
                focus.target = crate::camera::CameraFocusTarget::Body(soi_id);
                focus.target_distance = 2.0e7;
                focus.distance = focus.distance.max(2.0e7);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Part mesh + material rebuild (shares the mesh builders with the shipyard editor)
// ---------------------------------------------------------------------------

struct VisualSpec {
    mesh: Mesh,
    height: f32,
}

fn visual_spec(
    nodes: &AttachNodes,
    pod: Option<&CommandPod>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    fuselage: Option<&Fuselage>,
    engine: Option<&Engine>,
    intake: Option<&AirIntake>,
) -> Option<VisualSpec> {
    if let Some(p) = pod {
        // An inline cockpit has no body of its own — the fuselage nose is the
        // visible nose; it only supplies command/crew (+ a windshield morph).
        if matches!(p.geometry, PodGeometry::Inline) {
            return None;
        }
        let (radius_top, radius_bottom, h) = pod_visual_profile(p.diameter, p.geometry);
        let mesh = match p.geometry {
            // Rounded ogive nose (airliner radome) vs the plain capsule cone.
            PodGeometry::AircraftCockpit => build_cockpit_mesh(p.diameter, h),
            PodGeometry::Inline => unreachable!("handled above"),
            PodGeometry::Capsule => ConicalFrustum {
                radius_top,
                radius_bottom,
                height: h,
            }
            .mesh()
            .resolution(PART_RESOLUTION)
            .into(),
        };
        Some(VisualSpec { mesh, height: h })
    } else if dec.is_some() {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let h = 0.2;
        Some(VisualSpec {
            mesh: Cylinder::new(d * 0.5, h)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: h,
        })
    } else if let Some(a) = adapter {
        let top_d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let bot_d = a.target_diameter;
        let h = ((top_d + bot_d) * 0.5).max(0.4);
        Some(VisualSpec {
            mesh: ConicalFrustum {
                radius_top: top_d * 0.5,
                radius_bottom: bot_d * 0.5,
                height: h,
            }
            .mesh()
            .resolution(PART_RESOLUTION)
            .into(),
            height: h,
        })
    } else if let Some(t) = tank {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let h = t.length;
        Some(VisualSpec {
            mesh: Cylinder::new(d * 0.5, h)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: h,
        })
    } else if let Some(f) = fuselage {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(f.max_width);
        Some(VisualSpec {
            mesh: build_fuselage_mesh(f, d),
            height: f.length,
        })
    } else if let Some(e) = engine {
        match e.geometry {
            EngineGeometry::RocketBell => {
                let d = e.diameter;
                let (r_top, r_bot, h) = (d * 0.35, d * 0.5, d * 0.9);
                Some(VisualSpec {
                    mesh: ConicalFrustum {
                        radius_top: r_top,
                        radius_bottom: r_bot,
                        height: h,
                    }
                    .mesh()
                    .resolution(PART_RESOLUTION)
                    .into(),
                    height: h,
                })
            }
            EngineGeometry::JetNacelle => Some(VisualSpec {
                mesh: build_jet_nacelle_body_mesh(e),
                height: jet_nacelle_length(e),
            }),
        }
    } else if let Some(i) = intake {
        Some(VisualSpec {
            mesh: Cylinder::new(i.diameter * 0.5, i.length)
                .mesh()
                .resolution(PART_RESOLUTION)
                .into(),
            height: i.length,
        })
    } else {
        None
    }
}

// `ship_part_params` (part dims → material uniform) is shared with the in-game
// editor; it lives in `thalos_shipyard::appearance` (re-exported at the crate
// root) so the flight build and the editor build can't drift.

/// `top` node diameter of a host part, or a sensible default — the basis for
/// surface-mount radius lookups (see [`host_mount_geometry`]).
fn host_top_diameter(nodes: &Query<&AttachNodes>, host: Entity) -> f32 {
    nodes
        .get(host)
        .ok()
        .and_then(|n| n.get("top").map(|nd| nd.diameter))
        .unwrap_or(2.0)
}

type VisualQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static AttachNodes,
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Fuselage>,
        Option<&'static Engine>,
        Option<&'static AirIntake>,
        Option<&'static SurfaceMount>,
        Option<&'static Children>,
        Option<&'static PartShaderHandle>,
        Has<PartMaterial>,
    ),
    // The in-game shipyard editor owns its own visuals for `EditorPart`
    // entities; the flight-ship rebuild must not double-spawn them.
    (Or<(Added<Part>, Changed<AttachNodes>)>, Without<EditorPart>),
>;

/// Spawn (or respawn) the body mesh child for each part whose attach
/// layout just changed. Parts with [`PartMaterial`] use [`ShipPartMaterial`]
/// for the procedural stainless finish; the remainder fall back to a
/// [`ShadowedStandardMaterial`] (stock PBR + the shared sun-shadow receive,
/// F6 — a bare `StandardMaterial` would opt out of the one shadow world).
fn rebuild_ship_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    mut std_materials: ResMut<Assets<ShadowedStandardMaterial>>,
    parts: VisualQuery,
    stale: Query<(), With<PartVisual>>,
) {
    for (
        e,
        nodes,
        pod,
        dec,
        adapter,
        tank,
        fuselage,
        engine,
        intake,
        surface,
        children,
        part_shader,
        has_part_mat,
    ) in parts.iter()
    {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }

        if engine.is_some_and(|e| e.geometry == EngineGeometry::JetNacelle)
            && surface.is_some_and(|m| m.kind == SurfaceMountKind::WingPylon)
        {
            continue;
        }

        let Some(spec) = visual_spec(nodes, pod, dec, adapter, tank, fuselage, engine, intake)
        else {
            continue;
        };
        let mesh = meshes.add(spec.mesh);

        let body_id = if has_part_mat {
            let params = ship_part_params(nodes, tank, fuselage, dec, adapter, e.index_u32());
            let handle = match part_shader {
                Some(h) => h.0.clone(),
                None => {
                    let h = ship_materials.add(ShipPartMaterial {
                        base: stainless_steel_base(),
                        extension: ShipPartExtension {
                            params,
                            ..Default::default()
                        },
                    });
                    commands.entity(e).insert(PartShaderHandle(h.clone()));
                    h
                }
            };
            commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(handle),
                    Transform::from_xyz(0.0, -spec.height * 0.5, 0.0),
                    Visibility::default(),
                    NoFrustumCulling,
                    PartVisual,
                ))
                .id()
        } else {
            // CommandPod and Engine have no `PartMaterial` yet: give them the
            // same stainless-steel base finish as the fuel tank so the whole
            // craft reads as one material (no procedural seam shader on these).
            let mat = std_materials.add(shadowed(stainless_steel_base()));
            commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(mat),
                    Transform::from_xyz(0.0, -spec.height * 0.5, 0.0),
                    Visibility::default(),
                    NoFrustumCulling,
                    PartVisual,
                ))
                .id()
        };
        commands.entity(e).add_child(body_id);
    }
}

/// Build the mesh child for each wing on spawn (and on the rare runtime
/// change). Mirrors the editor's `rebuild_wing_visuals`: the mesh is in the
/// host-local frame and the wing entity's transform places it. Wings render
/// with a plain double-sided metal material until they get a dedicated skin.
fn rebuild_ship_wing_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<ShadowedStandardMaterial>>,
    wings: Query<
        (Entity, &Wing, &SurfaceMount, Option<&Children>),
        (
            Or<(Added<Wing>, Changed<Wing>, Changed<SurfaceMount>)>,
            Without<EditorPart>,
        ),
    >,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    stale: Query<(), With<PartVisual>>,
) {
    for (e, wing, mount, children) in wings.iter() {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }
        let top_d = host_top_diameter(&host_nodes, mount.parent);
        let (parent_radius, _) = host_mount_geometry(
            hosts.get(mount.parent).ok(),
            top_d,
            mount.station,
            mount.angle,
        );
        let mesh = meshes.add(build_wing_mesh(wing, mount.angle, parent_radius));
        let mat = std_materials.add(shadowed(StandardMaterial {
            double_sided: true,
            cull_mode: None,
            ..stainless_steel_base()
        }));
        let body = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(mat),
                Transform::IDENTITY,
                Visibility::default(),
                NoFrustumCulling,
                PartVisual,
            ))
            .id();
        commands.entity(e).add_child(body);

        // Wing-body junction fairing: the belly blister that merges a main
        // wing pair into the fuselage (and encloses the gear structure).
        // Derived geometry — generated for the right-hand panel of a low/mid
        // pair on a loft host, in the same host-local frame as the wing mesh.
        if let Ok(fus) = hosts.get(mount.parent)
            && wants_wing_fairing(wing, mount.angle, fus)
        {
            let mat = std_materials.add(shadowed(stainless_steel_base()));
            let fairing = commands
                .spawn((
                    Mesh3d(meshes.add(build_wing_fairing_mesh(fus, top_d, wing, mount.station))),
                    MeshMaterial3d(mat),
                    Transform::IDENTITY,
                    Visibility::default(),
                    NoFrustumCulling,
                    PartVisual,
                ))
                .id();
            commands.entity(e).add_child(fairing);
        }

        // One hinged child per control surface. Right panels (mount sin > 0)
        // take side_sign +1, left panels −1, so a roll command splits them.
        let side_sign = if mount.angle.sin() >= 0.0 { 1.0 } else { -1.0 };
        for surface in &wing.control_surfaces {
            let built = build_control_surface_mesh(wing, surface, mount.angle, parent_radius);
            let mat = std_materials.add(shadowed(StandardMaterial {
                double_sided: true,
                cull_mode: None,
                ..stainless_steel_base()
            }));
            let cs = commands
                .spawn((
                    Mesh3d(meshes.add(built.mesh)),
                    MeshMaterial3d(mat),
                    Transform::from_translation(built.geometry.hinge_anchor),
                    Visibility::default(),
                    NoFrustumCulling,
                    PartVisual,
                    ControlSurfaceVisual {
                        role: surface.role,
                        side_sign,
                        hinge_axis: built.geometry.hinge_axis,
                        hinge_anchor: built.geometry.hinge_anchor,
                        max_deflection: surface.max_deflection,
                    },
                ))
                .id();
            commands.entity(e).add_child(cs);
        }
    }
}

/// Deflect each control surface from the realized fly-by-wire command.
/// `+θ` about the (consistently oriented) hinge axis drops the trailing
/// edge, so the role signs convert command intent into geometry: nose-up
/// pitch raises the elevator's trailing edge, roll-right raises the right
/// aileron / lowers the left, nose-right yaw swings the rudder.
fn animate_ship_control_surfaces(
    realized: Res<crate::control_bus::RealizedControl>,
    flight_config: Res<crate::flight_config::FlightConfig>,
    mut q: Query<(&ControlSurfaceVisual, &mut Transform)>,
) {
    // Attitude surfaces deflect to the commanded attitude effort (full-scale),
    // not the allocated aero fraction — see `RealizedControl::command`. Flaps
    // and spoilers deflect to the flight-config actuator positions, the same
    // smoothed values the aero force model consumes.
    let cmd_vec = realized.command;
    for (cs, mut transform) in q.iter_mut() {
        let cmd = match cs.role {
            // −θ raises the trailing edge; nose-up pitch wants elevator up.
            ControlSurfaceRole::Elevator => -(cmd_vec.x as f32),
            // Differential: per-side sign, then −θ so roll-right raises the
            // right aileron's trailing edge.
            ControlSurfaceRole::Aileron => -(cmd_vec.y as f32) * cs.side_sign,
            ControlSurfaceRole::Rudder => cmd_vec.z as f32,
            // +θ drops the trailing edge: flaps run down, spoilers up.
            ControlSurfaceRole::Flap => flight_config.flap_fraction as f32,
            ControlSurfaceRole::Spoiler => -(flight_config.spoiler_fraction as f32),
        };
        let angle = cmd.clamp(-1.0, 1.0) * cs.max_deflection;
        transform.translation = cs.hinge_anchor;
        transform.rotation = Quat::from_axis_angle(cs.hinge_axis, angle);
    }
}

fn rebuild_ship_nacelle_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<ShadowedStandardMaterial>>,
    engines: Query<
        (Entity, &Engine, &SurfaceMount, Option<&Children>),
        (
            Or<(Added<Engine>, Changed<SurfaceMount>)>,
            Without<EditorPart>,
        ),
    >,
    wings: Query<&Wing>,
    surface_mounts: Query<&SurfaceMount>,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    stale: Query<(), With<PartVisual>>,
) {
    for (e, engine, mount, children) in engines.iter() {
        if engine.geometry != EngineGeometry::JetNacelle
            || mount.kind != SurfaceMountKind::WingPylon
        {
            continue;
        }
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }
        let Ok(wing) = wings.get(mount.parent) else {
            continue;
        };
        let Ok(wing_mount) = surface_mounts.get(mount.parent) else {
            continue;
        };
        let top_d = host_top_diameter(&host_nodes, wing_mount.parent);
        let (parent_radius, _) = host_mount_geometry(
            hosts.get(wing_mount.parent).ok(),
            top_d,
            wing_mount.station,
            wing_mount.angle,
        );
        let mesh = meshes.add(build_jet_nacelle_pylon_mesh(
            engine,
            JetNacelleMount {
                wing,
                wing_mount_angle: wing_mount.angle,
                parent_radius,
                span_fraction: mount.station,
                chord_fraction: mount.angle,
            },
        ));
        let mat = std_materials.add(shadowed(stainless_steel_base()));
        let body = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(mat),
                Transform::IDENTITY,
                Visibility::default(),
                NoFrustumCulling,
                PartVisual,
            ))
            .id();
        commands.entity(e).add_child(body);
    }
}

/// Build the mesh child for each gearbox on spawn (and the rare runtime
/// change). Mirrors the editor's `rebuild_gear_visuals`: the mesh is in the
/// host-local frame and the gear entity's transform places it. A gearbox draws
/// all its legs in one mesh — no symmetry. Plain double-sided metal until gear
/// gets a dedicated material.
fn rebuild_ship_gear_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<ShadowedStandardMaterial>>,
    gears: Query<
        (Entity, &Gear, &SurfaceMount, Option<&Children>),
        (
            Or<(Added<Gear>, Changed<Gear>, Changed<SurfaceMount>)>,
            Without<EditorPart>,
        ),
    >,
    host_nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    stale: Query<(), With<PartVisual>>,
) {
    for (e, gear, mount, children) in gears.iter() {
        if let Some(ch) = children {
            for c in ch.into_iter() {
                if stale.get(*c).is_ok() {
                    commands.entity(*c).despawn();
                }
            }
        }
        let top_d = host_top_diameter(&host_nodes, mount.parent);
        let (parent_radius, _) = host_mount_geometry(
            hosts.get(mount.parent).ok(),
            top_d,
            mount.station,
            mount.angle,
        );
        let mesh = meshes.add(build_gear_mesh(gear, mount.angle, parent_radius));
        let mat = std_materials.add(shadowed(StandardMaterial {
            double_sided: true,
            cull_mode: None,
            ..landing_gear_base()
        }));
        let body = commands
            .spawn((
                Mesh3d(mesh),
                MeshMaterial3d(mat),
                Transform::IDENTITY,
                Visibility::default(),
                NoFrustumCulling,
                PartVisual,
                GearVisual,
            ))
            .id();
        commands.entity(e).add_child(body);
        // The carrying structure of a wide-track leg (gear beam + side-stay)
        // is airframe, not undercarriage: hull finish, so it reads as part of
        // the wing/fairing rather than black scaffolding under the belly. It
        // retracts with the gear (same `GearVisual` visibility latch).
        if let Some(struct_mesh) = build_gear_struct_mesh(gear, mount.angle, parent_radius) {
            let mat = std_materials.add(shadowed(stainless_steel_base()));
            let structure = commands
                .spawn((
                    Mesh3d(meshes.add(struct_mesh)),
                    MeshMaterial3d(mat),
                    Transform::IDENTITY,
                    Visibility::default(),
                    NoFrustumCulling,
                    PartVisual,
                    GearVisual,
                ))
                .id();
            commands.entity(e).add_child(structure);
        }
    }
}

/// Hide/show the gear meshes to match [`GearState`]. Binary — there is no
/// retraction animation (the gear is a simple procedural mesh, so a sweep would
/// read as a stiff geometric fold rather than a real undercarriage). Touches
/// each [`GearVisual`] only when its target visibility changes.
fn sync_gear_visibility(
    gear_state: Res<crate::local_physics::GearState>,
    mut visuals: Query<&mut Visibility, With<GearVisual>>,
) {
    let target = if gear_state.down {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut visuals {
        if *vis != target {
            *vis = target;
        }
    }
}

/// Smoothing rate (Hz) for the visual gear-compression offset — fast enough
/// to track a landing stroke, slow enough to hide the raycast's frame noise.
const GEAR_COMPRESSION_SMOOTH_HZ: f32 = 12.0;

/// Slide each gearbox mesh **up into the hull** by its wheels' suspension
/// compression (the strut-swallow cheat). The gear mesh is rigid, authored at
/// full extension, while the physics suspension compresses up to
/// `max_travel_fraction·strut` — without this offset every centimetre of real
/// compression rendered as the wheels sinking through the pavement. Reads the
/// per-gearbox `(susp_dir, compression)` that [`crate::local_physics`]'s gear
/// system publishes ([`GearVisualCompression`]'s sole writer); an absent entry
/// (airborne wheel, gear up, no bubble) relaxes back to full extension.
fn sync_gear_compression(
    time: Res<Time>,
    compression: Res<crate::local_physics::GearVisualCompression>,
    mut visuals: Query<(&ChildOf, &mut Transform), With<GearVisual>>,
) {
    let blend = 1.0 - (-time.delta_secs() * GEAR_COMPRESSION_SMOOTH_HZ).exp();
    for (parent, mut transform) in &mut visuals {
        let target = compression
            .0
            .get(&parent.0)
            .map(|(susp_dir, compression_m)| (-*susp_dir * *compression_m).as_vec3())
            .unwrap_or(Vec3::ZERO);
        transform.translation = transform.translation.lerp(target, blend);
    }
}

/// BFS from the ship root, positioning each part's local Transform based
/// on its [`Attachment`]. Copied from the ship editor. A trailing pass
/// places surface-mounted parts (wings) on the host axis at their station.
fn update_ship_part_transforms(
    ships: Query<&Ship, Without<EditorPart>>,
    attachments: Query<(Entity, &Attachment), Without<EditorPart>>,
    surface_mounts: Query<(Entity, &SurfaceMount), Without<EditorPart>>,
    nodes: Query<&AttachNodes>,
    hosts: Query<&Fuselage>,
    mut transforms: Query<&mut Transform, (With<Part>, Without<EditorPart>)>,
) {
    let mut children_map: HashMap<Entity, Vec<(Entity, Attachment)>> = HashMap::new();
    for (e, att) in attachments.iter() {
        children_map
            .entry(att.parent)
            .or_default()
            .push((e, att.clone()));
    }

    for ship in ships.iter() {
        if let Ok(mut t) = transforms.get_mut(ship.root) {
            t.translation = Vec3::ZERO;
            t.rotation = Quat::IDENTITY;
        }
        let mut queue: VecDeque<Entity> = VecDeque::from([ship.root]);
        while let Some(parent) = queue.pop_front() {
            let parent_pos = transforms
                .get(parent)
                .map(|t| t.translation)
                .unwrap_or(Vec3::ZERO);
            let Ok(parent_nodes) = nodes.get(parent) else {
                continue;
            };
            let parent_pos_and_nodes: Vec<(Entity, Vec3)> = children_map
                .get(&parent)
                .map(|kids| {
                    kids.iter()
                        .filter_map(|(c, att)| {
                            let pn = parent_nodes.get(&att.parent_node)?;
                            let child_offset = nodes
                                .get(*c)
                                .ok()
                                .and_then(|cn| cn.get(&att.my_node))
                                .map(|n| n.offset)
                                .unwrap_or(Vec3::ZERO);
                            Some((*c, parent_pos + pn.offset - child_offset))
                        })
                        .collect()
                })
                .unwrap_or_default();
            for (child, pos) in parent_pos_and_nodes {
                if let Ok(mut ct) = transforms.get_mut(child) {
                    ct.translation = pos;
                    ct.rotation = Quat::IDENTITY;
                }
                queue.push_back(child);
            }
        }
    }

    // Surface-mounted parts sit in their host-local frame. Body-skin mounts
    // (wings) move down the host body axis; wing-pylon mounts (nacelles)
    // inherit the wing origin because the pylon mesh carries the offsets.
    for (part, mount) in surface_mounts.iter() {
        let Ok(parent_t) = transforms.get(mount.parent).map(|t| t.translation) else {
            continue;
        };
        let local_offset = match mount.kind {
            SurfaceMountKind::BodySkin => {
                let host_height = nodes
                    .get(mount.parent)
                    .ok()
                    .and_then(|n| n.get("bottom").map(|nd| -nd.offset.y))
                    .unwrap_or(0.0);
                // Follow a loft host's centerline upsweep/droop along +Z;
                // zero for a plain cylinder host.
                let top_d = host_top_diameter(&nodes, mount.parent);
                let (_, v_offset) =
                    host_mount_geometry(hosts.get(mount.parent).ok(), top_d, mount.station, 0.0);
                Vec3::new(0.0, -mount.station * host_height, v_offset)
            }
            SurfaceMountKind::WingPylon => Vec3::ZERO,
        };
        if let Ok(mut pt) = transforms.get_mut(part) {
            pt.translation = parent_t + local_offset;
            pt.rotation = Quat::IDENTITY;
        }
    }
}

/// Visual-AABB centre of all parts in the [`PlayerShip`]'s local frame.
/// Recomputed every frame so the camera pivot tracks staging and design
/// changes.
///
/// KSP uses true mass-weighted Centre of Mass, but that depends on each
/// part carrying realistic mass — including fuel. We don't model wet mass
/// yet, so a mass-weighted CoM on the current Apollo blueprint sits inside
/// the command pod (84% of dry mass). The geometric AABB centre always
/// frames the visible stack regardless of mass distribution; switch to a
/// proper wet-mass CoM once fuel mass is on the parts.
fn update_ship_camera_offset(
    ships: Query<(Entity, &Children), With<PlayerShip>>,
    parts: Query<
        (
            &Transform,
            &AttachNodes,
            Option<&CommandPod>,
            Option<&Decoupler>,
            Option<&Adapter>,
            Option<&FuelTank>,
            Option<&Fuselage>,
            Option<&Engine>,
            Option<&AirIntake>,
            Option<&Wing>,
            Option<&Gear>,
        ),
        With<Part>,
    >,
    mut offsets: Query<&mut CameraTargetOffset>,
) {
    for (ship_entity, children) in &ships {
        // Per-axis AABB. Y bounds use each part's visual height (the mesh is
        // offset by `-h/2` from the part transform, so it spans `[y-h, y]`).
        // X/Z bounds approximate the silhouette via each part's outer radius.
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut hits = 0;
        for child in children.iter() {
            let Ok((t, nodes, pod, dec, adapter, tank, fuselage, engine, intake, wing, gear)) =
                parts.get(child)
            else {
                continue;
            };
            let Some((height, radius)) = visual_extent(
                nodes, pod, dec, adapter, tank, fuselage, engine, intake, wing, gear,
            ) else {
                continue;
            };
            let lo = Vec3::new(
                t.translation.x - radius,
                t.translation.y - height,
                t.translation.z - radius,
            );
            let hi = Vec3::new(
                t.translation.x + radius,
                t.translation.y,
                t.translation.z + radius,
            );
            min = min.min(lo);
            max = max.max(hi);
            hits += 1;
        }
        let centre = if hits > 0 {
            (min + max) * 0.5
        } else {
            Vec3::ZERO
        };
        if let Ok(mut offset) = offsets.get_mut(ship_entity) {
            offset.0 = centre;
        }
    }
}

/// Visual `(height, max_radius)` for a part — mirrors [`visual_spec`]'s mesh
/// dimensions. Returns `None` for parts with no body mesh.
fn visual_extent(
    nodes: &AttachNodes,
    pod: Option<&CommandPod>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    fuselage: Option<&Fuselage>,
    engine: Option<&Engine>,
    intake: Option<&AirIntake>,
    wing: Option<&Wing>,
    gear: Option<&Gear>,
) -> Option<(f32, f32)> {
    if let Some(p) = pod {
        // An inline cockpit has no body mesh; it doesn't frame the camera.
        if matches!(p.geometry, PodGeometry::Inline) {
            return None;
        }
        Some((p.diameter * p.geometry.length_factor(), p.diameter * 0.5))
    } else if dec.is_some() {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        Some((0.2, d * 0.5))
    } else if let Some(a) = adapter {
        let top_d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        let bot_d = a.target_diameter;
        let h = ((top_d + bot_d) * 0.5).max(0.4);
        Some((h, top_d.max(bot_d) * 0.5))
    } else if let Some(t) = tank {
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(1.0);
        Some((t.length, d * 0.5))
    } else if let Some(f) = fuselage {
        // Whole-body extent: length tall, widest cross-section radius.
        let d = nodes.get("top").map(|n| n.diameter).unwrap_or(f.max_width);
        Some((f.length, 0.5 * d.max(f.max_height)))
    } else if let Some(w) = wing {
        // Chord runs along the body axis; the span reaches outboard in X/Z.
        Some((w.root_chord, w.span + w.root_chord))
    } else if let Some(g) = gear {
        // Struts + wheels hang below the belly along the mount radial; the
        // crude AABB just needs a reach large enough to keep the camera framing
        // it. Height ≈ strut + wheel, radius ≈ the same reach outboard.
        let reach = g.strut_length + g.wheel_radius * 2.0;
        Some((reach, reach))
    } else if let Some(i) = intake {
        Some((i.length, i.diameter * 0.5))
    } else {
        engine.map(|e| {
            let height = match e.geometry {
                EngineGeometry::RocketBell => e.diameter * 0.9,
                EngineGeometry::JetNacelle => jet_nacelle_length(e),
            };
            (height, e.diameter * 0.5)
        })
    }
}

/// Keep [`ShipPartMaterial`] uniforms in sync with part dimensions.
fn update_ship_part_shader_params(
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    parts: Query<
        (
            &AttachNodes,
            &PartShaderHandle,
            Option<&FuelTank>,
            Option<&Fuselage>,
            Option<&Decoupler>,
            Option<&Adapter>,
        ),
        Or<(
            Changed<FuelTank>,
            Changed<Fuselage>,
            Changed<Decoupler>,
            Changed<Adapter>,
            Changed<AttachNodes>,
        )>,
    >,
) {
    for (nodes, handle, tank, fuselage, dec, adapter) in parts.iter() {
        let Some(mut mat) = ship_materials.get_mut(&handle.0) else {
            continue;
        };
        let params = ship_part_params(
            nodes,
            tank,
            fuselage,
            dec,
            adapter,
            mat.extension.params.seed,
        );
        mat.extension.params.length = params.length;
        mat.extension.params.radius_top = params.radius_top;
        mat.extension.params.radius_bottom = params.radius_bottom;
    }
}
