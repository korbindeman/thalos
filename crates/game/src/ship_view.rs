//! Player ship rendering — loads the spawn situation's ship blueprint on
//! startup (`ships/apollo.ron` by default, `ships/a220.ron` for the runway
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

use bevy::camera::visibility::NoFrustumCulling;
use bevy::light::{NotShadowCaster, NotShadowReceiver};
use bevy::math::DVec3;
use bevy::mesh::Mesh;
use bevy::prelude::*;
use big_space::prelude::{BigSpace, CellCoord, Grid};
use thalos_physics_canonical::types::ShipParameters;
use thalos_shipyard::{
    Adapter, AirIntake, AttachNodes, Attachment, CommandPod, Decoupler, Engine, EngineGeometry,
    FuelTank, Gear, JetNacelleMount, Part, PartCatalog, PartMaterial, Ship, ShipBlueprint,
    ShipPartExtension, ShipPartMaterial, ShipPartParams, ShipyardPlugin, SurfaceMount,
    PodGeometry, SurfaceMountKind, Wing, build_cockpit_mesh, build_gear_mesh,
    build_jet_nacelle_body_mesh, build_jet_nacelle_pylon_mesh, build_wing_mesh, jet_nacelle_length,
    landing_gear_base, pod_visual_profile, stainless_steel_base,
};

use crate::SimStage;
use crate::camera::{CameraFocus, CameraTargetOffset, find_reference_body};
use crate::rendering::{CelestialBody, PlayerShip, ShipMarker, SimulationState, SolarSystemState};
use crate::view::{HideInMapView, HideInShipView, ViewMode};

/// Radial segments for cylinder / frustum part meshes. Matches the ship
/// editor's value so the two look identical side-by-side.
const PART_RESOLUTION: u32 = 128;

/// Whole-craft crash tolerance (m/s of surface-relative approach speed).
/// A terrain contact above this destroys the vessel. Forgiving first-slice
/// constant — a future per-part model derives it from the contacting parts.
/// See `docs/surface.md`.
// TEMP (camera-judder repro): survive a hard touchdown so the craft settles in
// Full on the terrain instead of being destroyed. Restore to 12.0 after.
const SHIP_IMPACT_TOLERANCE_M_S: f64 = 1.0e9;

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
        app.insert_resource(catalog)
            .add_plugins(ShipyardPlugin)
            .add_systems(Startup, spawn_player_ship)
            .add_systems(
                Update,
                (
                    rebuild_ship_visuals,
                    rebuild_ship_wing_visuals,
                    rebuild_ship_nacelle_visuals,
                    rebuild_ship_gear_visuals,
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
    let ron_path = PathBuf::from(situation.ship_blueprint_path());
    let text = match std::fs::read_to_string(&ron_path) {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to read {}: {}", ron_path.display(), e);
            return;
        }
    };
    let blueprint = match ShipBlueprint::from_ron(&text) {
        Ok(bp) => bp,
        Err(e) => {
            error!("Failed to parse {}: {}", ron_path.display(), e);
            return;
        }
    };

    // Push spawn-time MOI + reaction-wheel torque into the physics
    // simulation so attitude integration knows what we're flying. Active
    // thrust, mass flow, dry mass, and wet mass are refreshed each frame
    // by `fuel.rs` from enabled engines and live tank state.
    let stats = match blueprint.stats(&catalog) {
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
        thrust_n: 0.0,
        mass_flow_kg_per_s: 0.0,
        dry_mass_kg: stats.dry_mass_kg,
        // Whole-craft crash tolerance (forgiving first-slice constant; a
        // future per-part model derives this from the contacting parts).
        // See docs/surface.md.
        impact_tolerance_m_s: SHIP_IMPACT_TOLERANCE_M_S,
    });
    sim.simulation.set_ship_mass(stats.wet_mass_kg());
    info!(
        "ship params: MOI = ({:.0}, {:.0}, {:.0}) kg·m², max torque = {:.0} N·m/axis, m_dry = {:.0} kg, m₀ = {:.0} kg",
        stats.moment_of_inertia_kg_m2.x,
        stats.moment_of_inertia_kg_m2.y,
        stats.moment_of_inertia_kg_m2.z,
        stats.max_reaction_torque_n_m,
        stats.dry_mass_kg,
        stats.wet_mass_kg(),
    );

    let ship_entity = match blueprint.spawn(&mut commands, &catalog) {
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

    let player_ship = commands
        .spawn((
            PlayerShip,
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
        for part in &to_reparent {
            world.entity_mut(player_ship).add_child(*part);
        }
    });
}

/// Sync the [`PlayerShip`] root's cell, local translation, and orientation
/// to the canonical physics craft state each frame. Attitude is integrated
/// by [`Simulation`] under player control; map view's billboard ship marker
/// stays camera-aligned and is unaffected.
fn update_player_ship_world_position(
    sim: Res<SimulationState>,
    authority: Res<crate::local_physics::AvianAuthority>,
    fixed_time: Res<Time<Fixed>>,
    grid: Query<&Grid, With<BigSpace>>,
    mut query: Query<(&mut CellCoord, &mut Transform), With<PlayerShip>>,
) {
    let Ok(root_grid) = grid.single() else {
        return;
    };
    let Ok((mut cell, mut transform)) = query.single_mut() else {
        return;
    };
    let ship = sim.simulation.ship_state();

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
    // Only the relative velocity is extrapolated: the canonical velocity is
    // heliocentric (~30 km/s orbital), but the orbital component already moves
    // smoothly via the Kepler-evaluated body position in the readback — only the
    // body-relative descent stutters. Physics, canonical state, and the terrain
    // collider are untouched. Kepler-owned coast (`OnRails`/`AttitudeOnly`)
    // already advances once per render frame, so it is left alone.
    let position = if authority.owns_translation() {
        let body_id = sim.simulation.dominant_body();
        let body = sim.ephemeris.state(
            body_id,
            thalos_physics_canonical::canonical::Epoch(sim.simulation.sim_time()),
        );
        let rel_velocity = ship.velocity - body.velocity;
        ship.position + rel_velocity * fixed_time.overstep().as_secs_f64()
    } else {
        ship.position
    };

    let (next_cell, local) = root_grid.translation_to_grid(position);
    *cell = next_cell;
    transform.translation = local;
    transform.rotation = sim.simulation.attitude().orientation.as_quat();
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
    match *view {
        ViewMode::Ship => {
            let player_active = player
                .as_deref()
                .map(|state| state.is_active())
                .unwrap_or(false);
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
// Part mesh + material rebuild (ported from ship_editor)
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
    engine: Option<&Engine>,
    intake: Option<&AirIntake>,
) -> Option<VisualSpec> {
    if let Some(p) = pod {
        let (radius_top, radius_bottom, h) = pod_visual_profile(p.diameter, p.geometry);
        let mesh = match p.geometry {
            // Rounded ogive nose (airliner radome) vs the plain capsule cone.
            PodGeometry::AircraftCockpit => build_cockpit_mesh(p.diameter, h),
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

fn ship_part_params(
    nodes: &AttachNodes,
    tank: Option<&FuelTank>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    seed: u32,
) -> ShipPartParams {
    let top_r = nodes.get("top").map(|n| n.diameter * 0.5).unwrap_or(0.5);
    let (radius_top, radius_bottom, length) = if let Some(t) = tank {
        (top_r, top_r, t.length)
    } else if dec.is_some() {
        (top_r, top_r, 0.2)
    } else if let Some(a) = adapter {
        let bot_r = a.target_diameter * 0.5;
        let h = (top_r + bot_r).max(0.4);
        let dr = top_r - bot_r;
        let slant = (h * h + dr * dr).sqrt();
        (top_r, bot_r, slant)
    } else {
        (top_r, top_r, 1.0)
    };
    ShipPartParams {
        length,
        radius_top,
        radius_bottom,
        seed,
        ..default()
    }
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
        Option<&'static Engine>,
        Option<&'static AirIntake>,
        Option<&'static SurfaceMount>,
        Option<&'static Children>,
        Option<&'static PartShaderHandle>,
        Has<PartMaterial>,
    ),
    Or<(Added<Part>, Changed<AttachNodes>)>,
>;

/// Spawn (or respawn) the body mesh child for each part whose attach
/// layout just changed. Parts with [`PartMaterial`] use [`ShipPartMaterial`]
/// for the procedural stainless finish; the remainder fall back to a plain
/// `StandardMaterial`.
fn rebuild_ship_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut ship_materials: ResMut<Assets<ShipPartMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
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

        let Some(spec) = visual_spec(nodes, pod, dec, adapter, tank, engine, intake) else {
            continue;
        };
        let mesh = meshes.add(spec.mesh);

        let body_id = if has_part_mat {
            let params = ship_part_params(nodes, tank, dec, adapter, e.index_u32());
            let handle = match part_shader {
                Some(h) => h.0.clone(),
                None => {
                    let h = ship_materials.add(ShipPartMaterial {
                        base: stainless_steel_base(),
                        extension: ShipPartExtension { params },
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
            let mat = std_materials.add(stainless_steel_base());
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
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    wings: Query<
        (Entity, &Wing, &SurfaceMount, Option<&Children>),
        Or<(Added<Wing>, Changed<Wing>, Changed<SurfaceMount>)>,
    >,
    host_nodes: Query<&AttachNodes>,
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
        let parent_radius = host_nodes
            .get(mount.parent)
            .ok()
            .and_then(|n| n.get("top").map(|nd| nd.diameter * 0.5))
            .unwrap_or(1.0);
        let mesh = meshes.add(build_wing_mesh(wing, mount.angle, parent_radius));
        let mat = std_materials.add(StandardMaterial {
            double_sided: true,
            cull_mode: None,
            ..stainless_steel_base()
        });
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

fn rebuild_ship_nacelle_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    engines: Query<
        (Entity, &Engine, &SurfaceMount, Option<&Children>),
        Or<(Added<Engine>, Changed<SurfaceMount>)>,
    >,
    wings: Query<&Wing>,
    surface_mounts: Query<&SurfaceMount>,
    host_nodes: Query<&AttachNodes>,
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
        let parent_radius = host_nodes
            .get(wing_mount.parent)
            .ok()
            .and_then(|n| n.get("top").map(|nd| nd.diameter * 0.5))
            .unwrap_or(1.0);
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
        let mat = std_materials.add(stainless_steel_base());
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
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    gears: Query<
        (Entity, &Gear, &SurfaceMount, Option<&Children>),
        Or<(Added<Gear>, Changed<Gear>, Changed<SurfaceMount>)>,
    >,
    host_nodes: Query<&AttachNodes>,
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
        let parent_radius = host_nodes
            .get(mount.parent)
            .ok()
            .and_then(|n| n.get("top").map(|nd| nd.diameter * 0.5))
            .unwrap_or(1.0);
        let mesh = meshes.add(build_gear_mesh(gear, mount.angle, parent_radius));
        let mat = std_materials.add(StandardMaterial {
            double_sided: true,
            cull_mode: None,
            ..landing_gear_base()
        });
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

/// BFS from the ship root, positioning each part's local Transform based
/// on its [`Attachment`]. Copied from the ship editor. A trailing pass
/// places surface-mounted parts (wings) on the host axis at their station.
fn update_ship_part_transforms(
    ships: Query<&Ship>,
    attachments: Query<(Entity, &Attachment)>,
    surface_mounts: Query<(Entity, &SurfaceMount)>,
    nodes: Query<&AttachNodes>,
    mut transforms: Query<&mut Transform, With<Part>>,
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
                Vec3::new(0.0, -mount.station * host_height, 0.0)
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
            let Ok((t, nodes, pod, dec, adapter, tank, engine, intake, wing, gear)) =
                parts.get(child)
            else {
                continue;
            };
            let Some((height, radius)) =
                visual_extent(nodes, pod, dec, adapter, tank, engine, intake, wing, gear)
            else {
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
    engine: Option<&Engine>,
    intake: Option<&AirIntake>,
    wing: Option<&Wing>,
    gear: Option<&Gear>,
) -> Option<(f32, f32)> {
    if let Some(p) = pod {
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
            Option<&Decoupler>,
            Option<&Adapter>,
        ),
        Or<(
            Changed<FuelTank>,
            Changed<Decoupler>,
            Changed<Adapter>,
            Changed<AttachNodes>,
        )>,
    >,
) {
    for (nodes, handle, tank, dec, adapter) in parts.iter() {
        let Some(mat) = ship_materials.get_mut(&handle.0) else {
            continue;
        };
        let params = ship_part_params(nodes, tank, dec, adapter, mat.extension.params.seed);
        mat.extension.params.length = params.length;
        mat.extension.params.radius_top = params.radius_top;
        mat.extension.params.radius_bottom = params.radius_bottom;
    }
}
