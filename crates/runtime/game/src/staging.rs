//! KSP-style staging — Bevy/ECS layer.
//!
//! A vessel's parts are grouped into an ordered sequence of stages, derived
//! once from the **decoupler topology** of the attach tree — there is no
//! authored stage list. Activating a stage ignites that stage's engines and
//! fires its decouplers; firing a decoupler separates everything in its
//! attach subtree from the controlled vessel.
//!
//! Separation is a real graph cut. Each jettisoned subtree becomes its own
//! canonical vessel and rendered craft root, inherits the parent's motion,
//! receives the decoupler impulse, and remains in the world under OnRails
//! propagation. A separated subtree containing a command pod is recorded as
//! controllable; pod-less hardware remains persistent debris.
//!
//! Engines start **cold**: every engine is disabled when the plan is built,
//! and each stage lights its own. So the first stage activation is the launch
//! ignition, exactly like KSP.
//!
//! The pure stage derivation ([`derive_stages`]) and per-stage Δv accounting
//! ([`compute_stage_summaries`]) live in [`thalos_shipyard::staging`] so the
//! shipyard editor's staging preview and this live HUD share one derivation.
//! This module is the ECS layer that feeds them from live part entities.

use bevy::ecs::resource::IsResource;
use bevy::ecs::world::EntityRef;
use bevy::math::{DMat3, DVec3};
use bevy::prelude::*;
use bevy::transform::TransformSystems;
use big_space::prelude::CellCoord;
use std::collections::{HashMap, HashSet};
use std::io::Write;

use crate::shipyard_editor::core::EditorPart;
use thalos_input::game::GameInputIntent;
use thalos_physics_canonical::canonical::{AuthorityMode, CraftId};
use thalos_physics_canonical::types::{ShipParameters, VesselKind};
use thalos_shipyard::{
    Attachment, CommandPod, Decoupler, Engine, EngineActivation, Part, PartResources, PartRole,
    Resource, ResourceTotals, SummaryEngine, SummaryPart, SummaryStageInput,
    SurfaceMount, compute_stage_summaries, derive_stages, live_part_centroid_offset,
    live_part_dry_mass_kg, live_part_self_inertia, live_part_total_mass_kg, parallel_axis_inertia,
};
use thalos_world::StateVector;

use crate::SimStage;
use crate::rendering::{PlayerShip, SimulationState};
use crate::ship_view::{CraftIdentity, CraftPart, CraftRoot, PartVisual};
use crate::shrouds::{Shroud, ShroudFired};
use crate::view::HideInMapView;

pub use thalos_game_state::flight::StagingSummaries;

pub struct StagingPlugin;

impl Plugin for StagingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StagingSummaries>()
            .init_resource::<StageDemand>()
            .init_resource::<SeparationVisualAudits>()
            .add_systems(
                Update,
                (
                    build_staging_plan,
                    // Staging only acts at 1× live time (the warp check lives in
                    // the system); the Esc pause menu is gated here.
                    activate_stage
                        .after(build_staging_plan)
                        // A LocalRigidBody craft does not advance canonical
                        // translation in `Simulation::step`; the Avian
                        // readback installs its current-epoch inertial pose
                        // later in this set. Separation must clone that fresh
                        // pose. Cloning between the clock advance and readback
                        // seats the detached OnRails vessel one render frame
                        // behind — hundreds of metres at heliocentric speed.
                        .after(crate::local_physics::readback_local_craft)
                        .run_if(crate::pause_menu::not_game_paused),
                    // Recompute the aggregate inertia tensor from the live parts
                    // every frame, mirroring how `fuel.rs` recomputes mass/thrust.
                    // This makes attitude authority correct after a stage drops
                    // (and tracks fuel burn) without any ordering dependency on
                    // the staging or fuel systems — `set_ship_params` preserves
                    // the fields each system doesn't own.
                    recompute_ship_inertia,
                    // Publish the per-stage Δv / fuel readout consumed by the
                    // bottom-right HUD. After `activate_stage` so it reflects the
                    // post-staging part set.
                    publish_staging_summaries.after(activate_stage),
                )
                    .in_set(SimStage::Physics),
            )
            .add_systems(
                PostUpdate,
                audit_separated_vessel_visuals.after(TransformSystems::Propagate),
            );
    }
}

// ---------------------------------------------------------------------------
// Plan component + build
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Stage {
    engines: Vec<Entity>,
    decouplers: Vec<Entity>,
}

/// Staging plan for a vessel, derived once from its decoupler topology.
/// `next` is the index of the stage the next activation will fire; the plan
/// is spent once `next` reaches the stage count.
#[derive(Component, Debug)]
pub struct StagingPlan {
    stages: Vec<Stage>,
    next: usize,
}

impl StagingPlan {
    /// True once every stage has been activated.
    fn is_spent(&self) -> bool {
        self.next >= self.stages.len()
    }
}

/// Typed request/acknowledgement seam for automatic staging.
///
/// Flight programs request a stage through this resource instead of mutating
/// engine or decoupler state. [`activate_stage`] remains the one canonical
/// staging operation for both the space-bar and automation.
#[derive(Resource, Debug, Default)]
pub struct StageDemand {
    next_id: u64,
    pending: Option<u64>,
    completed: Option<(u64, bool)>,
}

impl StageDemand {
    pub fn request(&mut self) -> u64 {
        if let Some(id) = self.pending {
            return id;
        }
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.pending = Some(self.next_id);
        self.next_id
    }

    pub fn outcome(&self, id: u64) -> Option<bool> {
        self.completed
            .filter(|(completed_id, _)| *completed_id == id)
            .map(|(_, ok)| ok)
    }

    pub fn cancel(&mut self, id: u64) {
        if self.pending == Some(id) {
            self.pending = None;
            self.completed = Some((id, false));
        }
    }

    /// Test-only acknowledgement, standing in for [`activate_stage`] so the
    /// staging sequencer's state machine can be driven without a World.
    #[cfg(test)]
    pub(crate) fn test_complete(&mut self, ok: bool) {
        self.complete(ok);
    }

    fn complete(&mut self, ok: bool) {
        if let Some(id) = self.pending.take() {
            self.completed = Some((id, ok));
        }
    }
}

/// Build the [`StagingPlan`] once the player ship's parts exist, and disable
/// every engine so staging owns ignition. Runs until a plan is inserted,
/// then the `Without<StagingPlan>` filter retires it.
fn build_staging_plan(
    mut commands: Commands,
    sim: Res<SimulationState>,
    ships: Query<Entity, (With<PlayerShip>, Without<StagingPlan>)>,
    parts: Query<
        (
            Entity,
            &CraftPart,
            Option<&Attachment>,
            Option<&SurfaceMount>,
            Option<&Decoupler>,
            Option<&Engine>,
        ),
        // The in-game shipyard editor's build world shares these components;
        // it must never enter the flight craft's staging topology.
        (With<Part>, Without<EditorPart>),
    >,
    mut engine_activations: Query<(&CraftPart, &mut EngineActivation), Without<EditorPart>>,
) {
    let Ok(ship) = ships.single() else {
        return;
    };
    if parts.is_empty() {
        return;
    }

    let active_id = sim.simulation.active_craft_id();
    let entities: Vec<Entity> = parts
        .iter()
        .filter(|(_, owner, ..)| owner.0 == active_id)
        .map(|(e, ..)| e)
        .collect();
    if entities.is_empty() {
        return;
    }
    let index: HashMap<Entity, usize> = entities.iter().enumerate().map(|(i, &e)| (e, i)).collect();

    let mut roles = vec![PartRole::Other; entities.len()];
    let mut parent = vec![None; entities.len()];
    for (e, owner, attachment, surface_mount, decoupler, engine) in parts.iter() {
        if owner.0 != active_id {
            continue;
        }
        let i = index[&e];
        roles[i] = if decoupler.is_some() {
            PartRole::Decoupler
        } else if engine.is_some() {
            PartRole::Engine
        } else {
            PartRole::Other
        };
        parent[i] = attachment
            .map(|a| a.parent)
            .or_else(|| surface_mount.map(|m| m.parent))
            .and_then(|p| index.get(&p).copied());
    }

    let stages: Vec<Stage> = derive_stages(&roles, &parent)
        .into_iter()
        .map(|s| Stage {
            engines: s.engines.into_iter().map(|i| entities[i]).collect(),
            decouplers: s.decouplers.into_iter().map(|i| entities[i]).collect(),
        })
        .collect();

    // Engines spawn enabled (for throttle-only flight); staging takes over.
    for (owner, mut activation) in engine_activations.iter_mut() {
        if owner.0 == active_id {
            activation.enabled = false;
        }
    }

    let stage_count = stages.len();
    commands
        .entity(ship)
        .insert(StagingPlan { stages, next: 0 });
    info!("derived staging plan: {stage_count} stage(s)");
}

// ---------------------------------------------------------------------------
// Stage activation
// ---------------------------------------------------------------------------

/// On the stage input edge, ignite the current stage's engines and fire its
/// decouplers (dropping each one's live attach subtree), then advance.
///
/// **In-bubble limitation (flagged, not yet handled):** the craft's *motion*
/// is correct in every regime after a drop — `apply_local_forces` integrates
/// gravity/thrust from the live `ship_mass_kg`/`thrust_n` and
/// `compute_angular_acceleration` from the inertia [`recompute_ship_inertia`]
/// pushes each frame. But the Avian rigid body's compound *collider* and its
/// intrinsic contact `Mass`/`AngularInertia` are built once per bubble
/// session in `local_physics::spawn_player_avian_body`, so after staging
/// *inside the local bubble* they still describe the pre-drop stack. That
/// only matters on terrain contact following an in-bubble stage (a phantom
/// dropped section in the contact shape); ascent and orbital staging are
/// unaffected. The clean fix is bubble teardown-and-respawn, but doing it
/// without a one-frame duplicate-craft window needs the launchpad ascent
/// scenario to validate against — left as a follow-up (see staging task).
pub(crate) fn activate_stage(
    mut commands: Commands,
    intent: Res<GameInputIntent>,
    mut automatic: ResMut<StageDemand>,
    sim: Res<SimulationState>,
    mut plans: Query<(Entity, &mut StagingPlan), With<PlayerShip>>,
    mut engine_activations: Query<(&CraftPart, &mut EngineActivation)>,
    attachments: Query<(Entity, &Attachment, &CraftPart), Without<EditorPart>>,
    surface_mounts: Query<(Entity, &SurfaceMount, &CraftPart), Without<EditorPart>>,
) {
    if !intent.stage && automatic.pending.is_none() {
        return;
    }
    // Staging only acts at 1× live time — the same rule throttle follows in
    // `fuel.rs`. Paused (0×) and time-warp states freeze the world, so firing
    // decouplers there would drop parts with no dynamics. (The Esc pause menu
    // is handled by the system's `not_game_paused` run condition.)
    if (sim.simulation.warp.speed() - 1.0).abs() > f64::EPSILON {
        return;
    }
    let active_id = sim.simulation.active_craft_id();
    let Ok((active_root, mut plan)) = plans.single_mut() else {
        return;
    };
    if plan.is_spent() {
        automatic.complete(false);
        return;
    }

    let next = plan.next;
    let (engines, decouplers) = {
        let stage = &plan.stages[next];
        (stage.engines.clone(), stage.decouplers.clone())
    };

    for engine in engines {
        if let Ok((owner, mut activation)) = engine_activations.get_mut(engine)
            && owner.0 == active_id
        {
            activation.enabled = true;
        }
    }

    if !decouplers.is_empty() {
        let children = child_map(active_id, &attachments, &surface_mounts);
        let mut claimed = HashSet::new();
        let mut separated = Vec::new();
        for decoupler in decouplers {
            let parts = attach_subtree(decoupler, &children);
            if parts.iter().any(|entity| claimed.contains(entity)) {
                continue;
            }
            claimed.extend(parts.iter().copied());
            separated.push(SeparatedAssembly { decoupler, parts });
        }
        if !separated.is_empty() {
            for stage in plan.stages.iter_mut().skip(next + 1) {
                stage.engines.retain(|entity| !claimed.contains(entity));
                stage.decouplers.retain(|entity| !claimed.contains(entity));
            }
            commands.queue(move |world: &mut World| {
                materialize_separated_vessels(world, active_root, active_id, separated);
            });
        }
    }

    plan.next += 1;
    automatic.complete(true);
    info!(
        "staged: activated stage {next} ({}/{} fired)",
        plan.next,
        plan.stages.len()
    );
}

/// Map each part to its attach children, from the live attachment graph.
fn child_map(
    active_id: CraftId,
    attachments: &Query<(Entity, &Attachment, &CraftPart), Without<EditorPart>>,
    surface_mounts: &Query<(Entity, &SurfaceMount, &CraftPart), Without<EditorPart>>,
) -> HashMap<Entity, Vec<Entity>> {
    let mut map: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (child, attachment, owner) in attachments.iter() {
        if owner.0 == active_id {
            map.entry(attachment.parent).or_default().push(child);
        }
    }
    for (child, mount, owner) in surface_mounts.iter() {
        if owner.0 == active_id {
            map.entry(mount.parent).or_default().push(child);
        }
    }
    map
}

/// Every entity in the attach subtree rooted at `root` (inclusive).
fn attach_subtree(root: Entity, children: &HashMap<Entity, Vec<Entity>>) -> Vec<Entity> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(entity) = stack.pop() {
        out.push(entity);
        if let Some(kids) = children.get(&entity) {
            stack.extend(kids.iter().copied());
        }
    }
    out
}

#[derive(Debug)]
struct SeparatedAssembly {
    decoupler: Entity,
    parts: Vec<Entity>,
}

#[derive(Resource, Default)]
struct SeparationVisualAudits(Vec<SeparationVisualAudit>);

struct SeparationVisualAudit {
    craft_id: CraftId,
    root: Entity,
    parts: Vec<Entity>,
    age_frames: u32,
}

#[derive(Debug, Clone, Copy)]
struct PartAggregate {
    wet_mass_kg: f64,
    dry_mass_kg: f64,
    center_of_mass: DVec3,
    moment_of_inertia: DVec3,
    max_torque: f64,
    controllable: bool,
}

fn aggregate_world_parts(world: &World, entities: &[Entity]) -> Option<PartAggregate> {
    let mut wet_mass_kg = 0.0;
    let mut dry_mass_kg = 0.0;
    let mut weighted_center = DVec3::ZERO;
    let mut bodies = Vec::new();
    let mut max_torque = 0.0;
    let mut controllable = false;

    for &entity in entities {
        let Ok(part) = world.get_entity(entity) else {
            continue;
        };
        // Mass acts at the part's centroid, not its transform origin (the top
        // mating node) — see `live_part_centroid_offset`.
        let position = part
            .get::<Transform>()
            .map(|transform| {
                transform.translation.as_dvec3()
                    + transform.rotation.as_dquat() * live_part_centroid_offset(part)
            })
            .unwrap_or(DVec3::ZERO);
        let wet_mass = live_part_total_mass_kg(part);
        let dry_mass = live_part_dry_mass_kg(part) as f64;
        if let Some(pod) = part.get::<CommandPod>() {
            controllable = true;
            max_torque += pod.reaction_wheel_torque as f64;
        }
        if wet_mass <= 0.0 {
            continue;
        }
        wet_mass_kg += wet_mass;
        dry_mass_kg += dry_mass;
        weighted_center += position * wet_mass;
        bodies.push((wet_mass, position, live_part_self_inertia(part, wet_mass)));
    }

    if wet_mass_kg <= 0.0 {
        return None;
    }
    let center_of_mass = weighted_center / wet_mass_kg;
    let moment_of_inertia = bodies
        .into_iter()
        .map(|(mass, position, self_inertia)| {
            self_inertia + parallel_axis_inertia(mass, position - center_of_mass)
        })
        .sum();

    Some(PartAggregate {
        wet_mass_kg,
        dry_mass_kg,
        center_of_mass,
        moment_of_inertia,
        max_torque,
        controllable,
    })
}

fn vessel_params(aggregate: PartAggregate, inherited: ShipParameters) -> ShipParameters {
    ShipParameters {
        moment_of_inertia: aggregate.moment_of_inertia,
        center_of_mass: aggregate.center_of_mass,
        max_torque: DVec3::splat(aggregate.max_torque),
        gimbal_torque_full: DVec3::ZERO,
        // Separated hardware is deliberately ballistic in this vertical
        // slice. Engine ownership remains on its parts for a future vessel
        // switch/control pass, but it cannot borrow the active craft's scalar
        // propulsion bridge.
        thrust_n: 0.0,
        mass_flow_kg_per_s: 0.0,
        dry_mass_kg: aggregate.dry_mass_kg,
        impact_tolerance_m_s: inherited.impact_tolerance_m_s,
        reference_area_m2: inherited.reference_area_m2,
        drag_coefficient: inherited.drag_coefficient,
    }
}

fn separation_delta_velocities(
    world_impulse: DVec3,
    remaining_mass_kg: f64,
    detached_mass_kg: f64,
) -> (DVec3, DVec3) {
    (
        -world_impulse / remaining_mass_kg,
        world_impulse / detached_mass_kg,
    )
}

/// Wall-clock target for a jettisoned stage to fully clear the geometry it was
/// nested inside — an interstage shroud, or the decoupler's own face when there
/// is none. Separation reads as a deliberate push at this rate; much faster
/// looks like an explosive charge, much slower like the two stages are welded.
const SEPARATION_CLEARANCE_TIME_S: f64 = 2.0;

/// Axial clearance assumed for a decoupler with no shroud: the two mating faces
/// still have to visibly part, and a bare ring separating at millimetres per
/// second reads as a failure even though nothing is intersecting.
const BARE_SEPARATION_CLEARANCE_M: f64 = 1.0;

/// Axial distance the jettisoned assembly must travel before the shrouded part
/// above is clear of the decoupler's interstage. The shroud is a child of the
/// decoupler and its height *is* the engine's visual length
/// ([`crate::shrouds`]), so this is exactly the overlap to escape.
fn separation_clearance_m(world: &World, decoupler: Entity) -> f64 {
    let shroud_heights: Vec<f64> = world
        .get::<Children>(decoupler)
        .into_iter()
        .flat_map(|children| children.iter())
        .filter_map(|child| world.get::<Shroud>(child))
        .map(|shroud| f64::from(shroud.height))
        .collect();
    shroud_heights
        .into_iter()
        .fold(BARE_SEPARATION_CLEARANCE_M, f64::max)
}

/// Separation impulse actually applied: the authored spring impulse, raised to
/// whatever it takes to clear `clearance_m` in [`SEPARATION_CLEARANCE_TIME_S`].
///
/// The authored `ejection_impulse_per_diameter` is a fixed impulse, so the
/// relative separation *speed* it buys falls off as 1/mass — the 4 m ring gives
/// a light probe metres per second and a fully fuelled launch vehicle
/// centimetres per second, which is how an engine ended up creeping out of a
/// 3.6 m interstage for tens of seconds. Relative speed is `impulse / reduced
/// mass` (each side takes `J/m` of it, oppositely), so the impulse that buys a
/// given clearance rate is the reduced mass times that rate. Deriving the floor
/// that way makes clearance mass-independent by construction instead of a
/// constant that only holds for one vehicle.
fn separation_impulse_n_s(
    authored_n_s: f64,
    clearance_m: f64,
    remaining_mass_kg: f64,
    detached_mass_kg: f64,
) -> f64 {
    if remaining_mass_kg <= 0.0 || detached_mass_kg <= 0.0 {
        return authored_n_s;
    }
    let reduced_mass_kg =
        remaining_mass_kg * detached_mass_kg / (remaining_mass_kg + detached_mass_kg);
    let required_n_s = reduced_mass_kg * clearance_m / SEPARATION_CLEARANCE_TIME_S;
    authored_n_s.max(required_n_s)
}

/// Turn graph-cut part sets into independently propagated canonical vessels
/// and visible BigSpace roots. Runs as one exclusive deferred command so ECS
/// ownership and canonical fleet creation become visible atomically.
fn materialize_separated_vessels(
    world: &mut World,
    active_root: Entity,
    active_id: CraftId,
    separated: Vec<SeparatedAssembly>,
) {
    let dropped: HashSet<Entity> = separated
        .iter()
        .flat_map(|assembly| assembly.parts.iter().copied())
        .collect();
    let remaining_parts: Vec<Entity> = {
        let mut query = world.query_filtered::<(Entity, &CraftPart), With<Part>>();
        query
            .iter(world)
            .filter(|(entity, owner)| owner.0 == active_id && !dropped.contains(entity))
            .map(|(entity, _)| entity)
            .collect()
    };
    let Some(remaining) = aggregate_world_parts(world, &remaining_parts) else {
        error!("stage separation rejected: no mass remains on active craft");
        return;
    };

    let assemblies: Vec<(SeparatedAssembly, PartAggregate, f64, f64)> = separated
        .into_iter()
        .filter_map(|assembly| {
            let aggregate = aggregate_world_parts(world, &assembly.parts)?;
            let authored_impulse = world
                .get::<Decoupler>(assembly.decoupler)
                .map(|decoupler| decoupler.ejection_impulse as f64)
                .unwrap_or(0.0);
            // Read before the graph cut, while the shroud is still a child of
            // the decoupler and its `Shroud` component is still reachable.
            let clearance_m = separation_clearance_m(world, assembly.decoupler);
            Some((assembly, aggregate, authored_impulse, clearance_m))
        })
        .collect();
    if assemblies.is_empty() {
        return;
    }

    let root_transform = world
        .get::<Transform>(active_root)
        .cloned()
        .unwrap_or_default();
    let root_cell = world
        .get::<CellCoord>(active_root)
        .cloned()
        .unwrap_or(CellCoord::ZERO);
    let root_visibility = world
        .get::<Visibility>(active_root)
        .cloned()
        .unwrap_or(Visibility::Inherited);
    let real_root = world
        .get_resource::<crate::rendering::real_space::RealSpaceRoot>()
        .map(|root| root.entity);

    let mut created = Vec::new();
    let active_delta_v;
    {
        let mut sim = world.resource_mut::<SimulationState>();
        let Some(active_vessel) = sim.simulation.vessel(active_id) else {
            error!("stage separation rejected: active canonical vessel {active_id} is missing");
            return;
        };
        let active_state = active_vessel.state().clone();
        let inherited_params = *active_vessel.parameters();
        let mut active_velocity = active_state.translation.velocity;

        for (assembly, aggregate, authored_impulse, clearance_m) in &assemblies {
            let body_direction = (aggregate.center_of_mass - remaining.center_of_mass)
                .try_normalize()
                .unwrap_or(-DVec3::Y);
            let impulse = separation_impulse_n_s(
                *authored_impulse,
                *clearance_m,
                remaining.wet_mass_kg,
                aggregate.wet_mass_kg,
            );
            let world_impulse = active_state.attitude.orientation * body_direction * impulse;
            let (parent_delta_v, detached_delta_v) = separation_delta_velocities(
                world_impulse,
                remaining.wet_mass_kg,
                aggregate.wet_mass_kg,
            );

            let mut detached_state = active_state.clone();
            detached_state.translation.velocity += detached_delta_v;
            detached_state.mass.wet_mass_kg = aggregate.wet_mass_kg;
            detached_state.mass.dry_mass_kg = aggregate.dry_mass_kg;
            detached_state.mass.inertia_body_kg_m2 =
                DMat3::from_diagonal(aggregate.moment_of_inertia);
            detached_state.mass.center_of_mass_body_m = aggregate.center_of_mass;
            detached_state.authority = AuthorityMode::OnRails { trajectory: 0 };

            let id = sim.simulation.create_vessel(
                detached_state,
                vessel_params(*aggregate, inherited_params),
                VesselKind::Ship,
                aggregate.controllable,
            );
            active_velocity += parent_delta_v;
            let relative_speed_m_s = (detached_delta_v - parent_delta_v).length();
            created.push((
                id,
                assembly.decoupler,
                assembly.parts.clone(),
                aggregate.wet_mass_kg,
                impulse,
                *clearance_m,
                relative_speed_m_s,
            ));
        }

        let mut active_params = vessel_params(remaining, inherited_params);
        // The surviving stage's active engines are refreshed by `fuel.rs` on
        // the next frame; zeroing here prevents one stale frame of dropped
        // thrust from entering canonical state.
        active_params.thrust_n = 0.0;
        active_params.mass_flow_kg_per_s = 0.0;
        sim.simulation.set_ship_params(active_params);
        sim.simulation.set_ship_mass(remaining.wet_mass_kg);
        sim.simulation.set_ship_state_for(
            active_id,
            StateVector {
                position: active_state.translation.position,
                velocity: active_velocity,
            },
        );
        active_delta_v = active_velocity - active_state.translation.velocity;
    }
    crate::local_physics::apply_inertial_delta_v(world, active_delta_v);

    for (id, decoupler, parts, mass_kg, impulse_n_s, clearance_m, relative_speed_m_s) in created {
        let part_count = parts.len();
        let mut root = world.spawn((
            CraftRoot,
            CraftIdentity(id),
            HideInMapView,
            root_transform.clone(),
            root_cell.clone(),
            root_visibility.clone(),
            Name::new(format!("Separated stage {id}")),
        ));
        if let Some(real_root) = real_root {
            root.insert(ChildOf(real_root));
        }
        let detached_root = root.id();

        for part in parts {
            let mut part_entity = world.entity_mut(part);
            part_entity.insert(CraftPart(id));
            if part == decoupler {
                part_entity.remove::<Attachment>();
                part_entity.remove::<SurfaceMount>();
            }
            if let Some(mut activation) = part_entity.get_mut::<EngineActivation>() {
                activation.enabled = false;
            }
            drop(part_entity);
            if part == decoupler {
                // The interstage rides down with the decoupler, KSP-style. Its
                // `Attachment` is gone now, so `shrouds::sync_shrouds` would
                // read the provider as no longer qualifying and despawn the
                // shroud on the next frame — commit it instead.
                let shrouds: Vec<Entity> = world
                    .get::<Children>(part)
                    .into_iter()
                    .flat_map(|children| children.iter())
                    .filter(|child| world.get::<Shroud>(*child).is_some())
                    .collect();
                for shroud in shrouds {
                    world.entity_mut(shroud).insert(ShroudFired);
                }
            }
            world.entity_mut(detached_root).add_child(part);
        }
        let audit_parts = world
            .get::<Children>(detached_root)
            .map(|children| children.iter().collect())
            .unwrap_or_default();
        world
            .resource_mut::<SeparationVisualAudits>()
            .0
            .push(SeparationVisualAudit {
                craft_id: id,
                root: detached_root,
                parts: audit_parts,
                age_frames: 0,
            });
        info!(
            "stage separation created persistent vessel {id}: \
             {part_count} parts, {mass_kg:.0} kg, {impulse_n_s:.0} N·s, \
             {relative_speed_m_s:.3} m/s relative separation, \
             {clearance_m:.1} m to clear"
        );
    }
}

/// Short-lived structural trace for a newly separated vessel. The defect this
/// diagnoses is temporal and cannot be inferred from a later screenshot: the
/// canonical vessel may exist while its mesh descendants lose their hierarchy,
/// render layer, or propagated transform on the very next frame.
///
/// Samples are appended to `artifacts/diagnostics/stage_separation.jsonl` after
/// transform propagation. The trace retires after five seconds at 60 fps.
fn audit_separated_vessel_visuals(world: &mut World) {
    const SAMPLE_FRAMES: &[u32] = &[1, 2, 10, 60, 300];

    let mut audits = world
        .remove_resource::<SeparationVisualAudits>()
        .unwrap_or_default();
    for audit in &mut audits.0 {
        audit.age_frames += 1;
        if !SAMPLE_FRAMES.contains(&audit.age_frames) {
            continue;
        }

        let (canonical_position, active_position) = {
            let sim = world.resource::<SimulationState>();
            let canonical_position = sim
                .simulation
                .vessel(audit.craft_id)
                .map(|vessel| vessel.state().translation.position);
            let active_position = sim.simulation.craft_state().translation.position;
            (canonical_position, active_position)
        };

        let root = world.get_entity(audit.root).ok();
        let root_children = root
            .as_ref()
            .and_then(|entity| entity.get::<Children>())
            .map(|children| children.len())
            .unwrap_or(0);
        let root_cell = root
            .as_ref()
            .and_then(|entity| entity.get::<CellCoord>())
            .map(|cell| format!("{cell:?}"));
        let root_transform = root
            .as_ref()
            .and_then(|entity| entity.get::<Transform>())
            .map(|transform| transform.translation.to_array());
        let root_global = root
            .as_ref()
            .and_then(|entity| entity.get::<GlobalTransform>())
            .map(|transform| transform.translation().to_array());
        let root_visibility = root
            .as_ref()
            .and_then(|entity| entity.get::<Visibility>())
            .map(|visibility| format!("{visibility:?}"));
        let root_inherited_visible = root
            .as_ref()
            .and_then(|entity| entity.get::<InheritedVisibility>())
            .map(|visibility| visibility.get());
        let root_layers = root
            .as_ref()
            .and_then(|entity| entity.get::<bevy::camera::visibility::RenderLayers>())
            .map(|layers| format!("{layers:?}"));

        let part_samples: Vec<_> = audit
            .parts
            .iter()
            .map(|&part| {
                let entity = world.get_entity(part).ok();
                let parent = entity
                    .as_ref()
                    .and_then(|entity| entity.get::<ChildOf>())
                    .map(|parent| parent.parent().to_bits());
                let part_global = entity
                    .as_ref()
                    .and_then(|entity| entity.get::<GlobalTransform>())
                    .map(|transform| transform.translation().to_array());
                let part_layers = entity
                    .as_ref()
                    .and_then(|entity| {
                        entity.get::<bevy::camera::visibility::RenderLayers>()
                    })
                    .map(|layers| format!("{layers:?}"));
                let low_precision_root = entity.as_ref().is_some_and(|entity| {
                    entity.contains::<big_space::grid::propagation::LowPrecisionRoot>()
                });
                let inherited_visible = entity
                    .as_ref()
                    .and_then(|entity| entity.get::<InheritedVisibility>())
                    .map(|visibility| visibility.get());
                let view_visible = entity
                    .as_ref()
                    .and_then(|entity| entity.get::<ViewVisibility>())
                    .map(|visibility| visibility.get());
                let visual_children: Vec<_> = entity
                    .as_ref()
                    .and_then(|entity| entity.get::<Children>())
                    .into_iter()
                    .flat_map(|children| children.iter())
                    .filter(|child| {
                        world
                            .get_entity(*child)
                            .ok()
                            .is_some_and(|entity| entity.contains::<PartVisual>())
                    })
                    .map(|child| {
                        let visual = world.get_entity(child).ok();
                        serde_json::json!({
                            "entity": child.to_bits(),
                            "inherited_visible": visual
                                .as_ref()
                                .and_then(|entity| entity.get::<InheritedVisibility>())
                                .map(|visibility| visibility.get()),
                            "view_visible": visual
                                .as_ref()
                                .and_then(|entity| entity.get::<ViewVisibility>())
                                .map(|visibility| visibility.get()),
                            "layers": visual
                                .as_ref()
                                .and_then(|entity| entity.get::<bevy::camera::visibility::RenderLayers>())
                                .map(|layers| format!("{layers:?}")),
                            "global_translation": visual
                                .as_ref()
                                .and_then(|entity| entity.get::<GlobalTransform>())
                                .map(|transform| transform.translation().to_array()),
                        })
                    })
                    .collect();

                serde_json::json!({
                    "entity": part.to_bits(),
                    "exists": entity.is_some(),
                    "parent": parent,
                    "expected_parent": audit.root.to_bits(),
                    "low_precision_root": low_precision_root,
                    "inherited_visible": inherited_visible,
                    "view_visible": view_visible,
                    "layers": part_layers,
                    "global_translation": part_global,
                    "visual_children": visual_children,
                })
            })
            .collect();

        let record = serde_json::json!({
            "event": "separated_vessel_visual_audit",
            "craft_id": audit.craft_id.to_string(),
            "age_frames": audit.age_frames,
            "root": {
                "entity": audit.root.to_bits(),
                "exists": root.is_some(),
                "children": root_children,
                "cell": root_cell,
                "translation": root_transform,
                "global_translation": root_global,
                "visibility": root_visibility,
                "inherited_visible": root_inherited_visible,
                "layers": root_layers,
            },
            "canonical_position": canonical_position.map(|position| position.to_array()),
            "distance_from_active_m": canonical_position
                .map(|position| position.distance(active_position)),
            "parts": part_samples,
        });
        let path = thalos_diagnostics::paths::default_jsonl_path("stage_separation.jsonl");
        match thalos_diagnostics::paths::open_jsonl_append(&path) {
            Ok(mut file) => {
                if let Err(error) = writeln!(file, "{record}") {
                    warn!("could not append {}: {error}", path.display());
                }
            }
            Err(error) => warn!("could not open {}: {error}", path.display()),
        }
    }
    audits.0.retain(|audit| audit.age_frames < 300);
    world.insert_resource(audits);
}

// ---------------------------------------------------------------------------
// Live inertia recompute
// ---------------------------------------------------------------------------

// `Without<IsResource>` (bevy 0.19): resources are components now, so a broad
// `EntityRef` query conservatively conflicts with this system's resource access
// (B0001). Excluding resource entities makes the access disjoint.
type PartQuery<'w, 's> =
    Query<'w, 's, EntityRef<'static>, (With<Part>, Without<EditorPart>, Without<IsResource>)>;

/// Recompute the ship's aggregate moment of inertia and reaction-wheel
/// torque from the live parts and push them into [`ShipParameters`]. Skips
/// when there are no parts (e.g. EVA), leaving those parameters untouched.
///
/// Mass + per-part inertia come from the single `thalos_shipyard` live-part
/// aggregation ([`live_part_total_mass_kg`] / [`live_part_self_inertia`]), so
/// every mass-bearing kind — wings and gear included — contributes here exactly
/// as it does to the flight mass in `fuel.rs`. The dominant inertia term is the
/// parallel-axis offset, computed about the live CoM.
fn recompute_ship_inertia(mut sim: ResMut<SimulationState>, parts: PartQuery) {
    if parts.is_empty() {
        return;
    }
    let active_id = sim.simulation.active_craft_id();

    let mut total_mass = 0.0_f64;
    let mut weighted_center = DVec3::ZERO;
    let mut max_torque = 0.0_f64;
    // (mass, body-frame position, self-inertia about the part's own CoM) per
    // part. Self-inertia is orientation/CoM-independent, so it's resolved here
    // before the ship CoM is known; the parallel-axis term is added below.
    let mut bodies: Vec<(f64, DVec3, DVec3)> = Vec::new();
    // (rated thrust N, gimbal range rad, body-frame Y) per gimballed engine.
    // The moment arm needs the ship CoM, so it's finished in a second pass.
    let mut gimbal_engines: Vec<(f64, f64, f64)> = Vec::new();

    for part in parts.iter() {
        if part
            .get::<CraftPart>()
            .is_none_or(|owner| owner.0 != active_id)
        {
            continue;
        }
        if let Some(pod) = part.get::<CommandPod>() {
            max_torque += pod.reaction_wheel_torque as f64;
        }
        let mass = live_part_total_mass_kg(part) * surface_multiplier(part.get::<SurfaceMount>());
        if mass <= 0.0 {
            continue;
        }
        // Mass acts at the part's centroid, not its transform origin (the top
        // mating node) — see `live_part_centroid_offset`.
        let position = part
            .get::<Transform>()
            .map(|t| {
                t.translation.as_dvec3() + t.rotation.as_dquat() * live_part_centroid_offset(part)
            })
            .unwrap_or(DVec3::ZERO);
        if let Some(engine) = part.get::<Engine>() {
            if engine.gimbal_range_deg > 0.0 {
                gimbal_engines.push((
                    engine.thrust as f64,
                    (engine.gimbal_range_deg as f64).to_radians(),
                    position.y,
                ));
            }
        }
        let self_inertia = live_part_self_inertia(part, mass);
        total_mass += mass;
        weighted_center += position * mass;
        bodies.push((mass, position, self_inertia));
    }

    if bodies.is_empty() || total_mass <= 0.0 {
        return;
    }
    let center_of_mass = weighted_center / total_mass;

    let mut moment_of_inertia = DVec3::ZERO;
    for (mass, position, self_inertia) in bodies {
        moment_of_inertia += self_inertia + parallel_axis_inertia(mass, position - center_of_mass);
    }

    // Thrust-vectoring authority at full thrust: each gimballed engine produces
    // `thrust · sin(range)` of side force at its bell, times its axial arm to
    // the CoM. Pitch and yaw share it (an axisymmetric bell gimbals both ways);
    // roll stays 0 — a centred engine can't roll the stack. Scaled by the live
    // throttle where it's consumed, so it vanishes at zero thrust and coast.
    let gimbal_pitch_yaw: f64 = gimbal_engines
        .iter()
        .map(|&(thrust, range_rad, y)| thrust * range_rad.sin() * (y - center_of_mass.y).abs())
        .sum();

    let mut params = *sim.simulation.ship_params();
    params.moment_of_inertia = moment_of_inertia;
    params.max_torque = DVec3::splat(max_torque);
    params.gimbal_torque_full = DVec3::new(gimbal_pitch_yaw, 0.0, gimbal_pitch_yaw);
    sim.simulation.set_ship_params(params);
}

// ---------------------------------------------------------------------------
// Per-stage Δv / fuel readout
// ---------------------------------------------------------------------------


/// Gather the live parts + plan into the pure summary inputs, compute, and
/// publish [`StagingSummaries`] for the HUD. Per-part dry mass comes from the
/// single `thalos_shipyard` live-part enumeration ([`live_part_dry_mass_kg`]),
/// so the HUD Δv readout counts every part kind the flight mass does.
fn publish_staging_summaries(
    plans: Query<&StagingPlan>,
    parts: PartQuery,
    sim: Res<SimulationState>,
    mut summaries: ResMut<StagingSummaries>,
) {
    let Ok(plan) = plans.single() else {
        if !summaries.0.is_empty() {
            summaries.0.clear();
        }
        return;
    };

    let mut index: HashMap<Entity, usize> = HashMap::new();
    let mut summary_parts: Vec<SummaryPart> = Vec::new();
    let mut parents: Vec<Option<Entity>> = Vec::new();
    let active_id = sim.simulation.active_craft_id();

    for part in parts.iter() {
        if part
            .get::<CraftPart>()
            .is_none_or(|owner| owner.0 != active_id)
        {
            continue;
        }
        let surface_mount = part.get::<SurfaceMount>();
        let multiplier = surface_multiplier(surface_mount);
        index.insert(part.id(), summary_parts.len());
        parents.push(
            part.get::<Attachment>()
                .map(|a| a.parent)
                .or_else(|| surface_mount.map(|m| m.parent)),
        );
        summary_parts.push(SummaryPart {
            parent: None,
            dry_mass_kg: live_part_dry_mass_kg(part) as f64 * multiplier,
            resources: part
                .get::<PartResources>()
                .map(part_resource_totals)
                .unwrap_or_default(),
            engine: part.get::<Engine>().map(|en| SummaryEngine {
                thrust_n: en.thrust as f64 * multiplier,
                isp_s: en.isp as f64,
                reactants: en
                    .reactants
                    .iter()
                    .map(|r| (r.resource, r.mass_fraction as f64))
                    .collect(),
            }),
        });
    }
    for (i, parent) in parents.iter().enumerate() {
        summary_parts[i].parent = parent.and_then(|p| index.get(&p).copied());
    }

    let stage_inputs: Vec<SummaryStageInput> = plan
        .stages
        .iter()
        .enumerate()
        .map(|(k, s)| SummaryStageInput {
            number: k + 1,
            engines: s
                .engines
                .iter()
                .filter_map(|e| index.get(e).copied())
                .collect(),
            decouplers: s
                .decouplers
                .iter()
                .filter_map(|e| index.get(e).copied())
                .collect(),
        })
        .collect();

    summaries.0 = compute_stage_summaries(&stage_inputs, &summary_parts, plan.next);
}

fn surface_multiplier(_surface_mount: Option<&SurfaceMount>) -> f64 {
    // KSP symmetry: each mirror counterpart is its own part, counted once —
    // no per-mount doubling. Kept as a hook in case future footprint kinds
    // re-introduce a multiplier.
    1.0
}

fn part_resource_totals(resources: &PartResources) -> HashMap<Resource, ResourceTotals> {
    resources
        .pools
        .iter()
        .map(|(&res, pool)| {
            (
                res,
                ResourceTotals {
                    amount: pool.amount as f64,
                    capacity: pool.capacity as f64,
                    mass_kg: pool.mass_kg(res),
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_shipyard::{CatalogEntry, PartCatalog};

    #[test]
    fn automatic_stage_demand_is_edge_triggered_and_acknowledged() {
        let mut demand = StageDemand::default();
        let first = demand.request();
        assert_eq!(demand.request(), first);
        assert_eq!(demand.outcome(first), None);
        demand.complete(true);
        assert_eq!(demand.outcome(first), Some(true));

        let second = demand.request();
        assert_ne!(second, first);
        demand.cancel(second);
        assert_eq!(demand.outcome(second), Some(false));
    }

    #[test]
    fn separation_impulse_conserves_linear_momentum() {
        let parent_mass = 3_000.0;
        let detached_mass = 1_200.0;
        let impulse = DVec3::new(40.0, -250.0, 15.0);
        let (parent_delta_v, detached_delta_v) =
            separation_delta_velocities(impulse, parent_mass, detached_mass);

        let net_delta_momentum = parent_delta_v * parent_mass + detached_delta_v * detached_mass;
        assert!(net_delta_momentum.length() < 1.0e-10);
    }

    #[test]
    fn tuned_saturn_decoupler_opens_a_visible_gap() {
        let catalog =
            PartCatalog::load_from_str(include_str!("../../../../assets/parts.ron")).unwrap();
        let CatalogEntry::Decoupler(spec) = catalog.resolve("decoupler_std").unwrap() else {
            panic!("decoupler_std must remain a decoupler");
        };

        // Use conservative wet Saturn masses: the catalog's 4 m ring must
        // still open a metre-scale gap within a few seconds instead of
        // leaving both meshes superimposed.
        let impulse = DVec3::Y * f64::from(spec.ejection_impulse_per_diameter * 4.0);
        let (upper_delta_v, booster_delta_v) =
            separation_delta_velocities(impulse, 42_000.0, 87_000.0);
        let relative_speed_m_s = (booster_delta_v - upper_delta_v).length();

        assert!(
            relative_speed_m_s >= 0.25,
            "relative separation speed was only {relative_speed_m_s:.3} m/s"
        );
    }

    /// The authored 4 m spring impulse alone leaves a fully fuelled launch
    /// vehicle creeping out of its 3.6 m interstage for tens of seconds — the
    /// defect the clearance floor exists to remove. Fails on the pre-floor
    /// behaviour (`authored` alone) and passes on the derived impulse.
    #[test]
    fn clearance_floor_clears_a_saturn_interstage_on_time() {
        let catalog =
            PartCatalog::load_from_str(include_str!("../../../../assets/parts.ron")).unwrap();
        let CatalogEntry::Decoupler(spec) = catalog.resolve("decoupler_std").unwrap() else {
            panic!("decoupler_std must remain a decoupler");
        };
        let authored = f64::from(spec.ejection_impulse_per_diameter * 4.0);
        // Boreas is a 4 m engine, so its shroud is 0.9 × 4 m of overlap.
        let clearance_m = 3.6;
        let (upper_kg, booster_kg) = (42_000.0, 87_000.0);

        let bare_speed = {
            let (a, b) = separation_delta_velocities(DVec3::Y * authored, upper_kg, booster_kg);
            (b - a).length()
        };
        assert!(
            clearance_m / bare_speed > 4.0 * SEPARATION_CLEARANCE_TIME_S,
            "authored impulse alone already clears the interstage in \
             {:.1} s — the floor under test is no longer load-bearing",
            clearance_m / bare_speed
        );

        let impulse = separation_impulse_n_s(authored, clearance_m, upper_kg, booster_kg);
        let (upper_delta_v, booster_delta_v) =
            separation_delta_velocities(DVec3::Y * impulse, upper_kg, booster_kg);
        let relative_speed_m_s = (booster_delta_v - upper_delta_v).length();
        let clear_time_s = clearance_m / relative_speed_m_s;
        assert!(
            (clear_time_s - SEPARATION_CLEARANCE_TIME_S).abs() < 0.05,
            "interstage clears in {clear_time_s:.2} s, wanted \
             {SEPARATION_CLEARANCE_TIME_S:.2} s"
        );
    }

    /// A light pair keeps the authored spring impulse: the floor raises weak
    /// separations, it never caps a snappy one.
    #[test]
    fn clearance_floor_never_weakens_a_light_separation() {
        let authored = 8_000.0;
        let impulse = separation_impulse_n_s(authored, 3.6, 900.0, 1_100.0);
        assert!(
            (impulse - authored).abs() < 1.0e-9,
            "floor overrode the authored impulse: {impulse:.1} N·s"
        );
    }
}
