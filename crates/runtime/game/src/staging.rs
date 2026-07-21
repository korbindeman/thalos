//! KSP-style staging — Bevy/ECS layer.
//!
//! A vessel's parts are grouped into an ordered sequence of stages, derived
//! once from the **decoupler topology** of the attach tree — there is no
//! authored stage list. Activating a stage ignites that stage's engines and
//! fires its decouplers; firing a decoupler separates everything in its
//! attach subtree from the controlled vessel.
//!
//! For now separation simply **drops the mass**: the jettisoned subtree is
//! despawned and the craft's aggregate properties heal from the surviving
//! live parts — mass/thrust via [`crate::fuel`], inertia via
//! [`recompute_ship_inertia`] here. The separation is modelled as a graph
//! cut (which parts came off), so the later move to real debris / multi-craft
//! control (a separated subtree becoming a controllable craft when it
//! carries a command pod, else uncontrollable debris) reuses this same
//! computation rather than replacing it.
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
use bevy::math::DVec3;
use bevy::prelude::*;
use std::collections::HashMap;

use crate::shipyard_editor::core::EditorPart;
use thalos_input::game::GameInputIntent;
use thalos_shipyard::{
    Attachment, CommandPod, Decoupler, Engine, EngineActivation, Part, PartResources, PartRole,
    Resource, ResourceTotals, StageSummary, SummaryEngine, SummaryPart, SummaryStageInput,
    SurfaceMount, compute_stage_summaries, derive_stages, live_part_dry_mass_kg,
    live_part_self_inertia, live_part_total_mass_kg, parallel_axis_inertia,
};

use crate::SimStage;
use crate::rendering::{PlayerShip, SimulationState};

pub struct StagingPlugin;

impl Plugin for StagingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StagingSummaries>().add_systems(
            Update,
            (
                build_staging_plan,
                // Staging only acts at 1× live time (the warp check lives in
                // the system); the Esc pause menu is gated here.
                activate_stage
                    .after(build_staging_plan)
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

/// Build the [`StagingPlan`] once the player ship's parts exist, and disable
/// every engine so staging owns ignition. Runs until a plan is inserted,
/// then the `Without<StagingPlan>` filter retires it.
fn build_staging_plan(
    mut commands: Commands,
    ships: Query<Entity, (With<PlayerShip>, Without<StagingPlan>)>,
    parts: Query<
        (
            Entity,
            Option<&Attachment>,
            Option<&SurfaceMount>,
            Option<&Decoupler>,
            Option<&Engine>,
        ),
        // The in-game shipyard editor's build world shares these components;
        // it must never enter the flight craft's staging topology.
        (With<Part>, Without<EditorPart>),
    >,
    mut engine_activations: Query<&mut EngineActivation, Without<EditorPart>>,
) {
    let Ok(ship) = ships.single() else {
        return;
    };
    if parts.is_empty() {
        return;
    }

    // Single ship today, so every `Part` belongs to it. Multi-ship will need
    // to scope this to the ship's own subtree.
    let entities: Vec<Entity> = parts.iter().map(|(e, ..)| e).collect();
    let index: HashMap<Entity, usize> = entities.iter().enumerate().map(|(i, &e)| (e, i)).collect();

    let mut roles = vec![PartRole::Other; entities.len()];
    let mut parent = vec![None; entities.len()];
    for (e, attachment, surface_mount, decoupler, engine) in parts.iter() {
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
    for mut activation in engine_activations.iter_mut() {
        activation.enabled = false;
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
fn activate_stage(
    mut commands: Commands,
    intent: Res<GameInputIntent>,
    sim: Res<SimulationState>,
    mut plans: Query<&mut StagingPlan>,
    mut engine_activations: Query<&mut EngineActivation>,
    attachments: Query<(Entity, &Attachment), Without<EditorPart>>,
    surface_mounts: Query<(Entity, &SurfaceMount), Without<EditorPart>>,
) {
    if !intent.stage {
        return;
    }
    // Staging only acts at 1× live time — the same rule throttle follows in
    // `fuel.rs`. Paused (0×) and time-warp states freeze the world, so firing
    // decouplers there would drop parts with no dynamics. (The Esc pause menu
    // is handled by the system's `not_game_paused` run condition.)
    if (sim.simulation.warp.speed() - 1.0).abs() > f64::EPSILON {
        return;
    }
    let Ok(mut plan) = plans.single_mut() else {
        return;
    };
    if plan.is_spent() {
        return;
    }

    let next = plan.next;
    let (engines, decouplers) = {
        let stage = &plan.stages[next];
        (stage.engines.clone(), stage.decouplers.clone())
    };

    for engine in engines {
        if let Ok(mut activation) = engine_activations.get_mut(engine) {
            activation.enabled = true;
        }
    }

    if !decouplers.is_empty() {
        let children = child_map(&attachments, &surface_mounts);
        for decoupler in decouplers {
            for dropped in attach_subtree(decoupler, &children) {
                // `try_despawn` is recursive (drops each part's visual mesh
                // children) and tolerant of an already-gone entity.
                commands.entity(dropped).try_despawn();
            }
        }
    }

    plan.next += 1;
    info!(
        "staged: activated stage {next} ({}/{} fired)",
        plan.next,
        plan.stages.len()
    );
}

/// Map each part to its attach children, from the live attachment graph.
fn child_map(
    attachments: &Query<(Entity, &Attachment), Without<EditorPart>>,
    surface_mounts: &Query<(Entity, &SurfaceMount), Without<EditorPart>>,
) -> HashMap<Entity, Vec<Entity>> {
    let mut map: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (child, attachment) in attachments.iter() {
        map.entry(attachment.parent).or_default().push(child);
    }
    for (child, mount) in surface_mounts.iter() {
        map.entry(mount.parent).or_default().push(child);
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
        if let Some(pod) = part.get::<CommandPod>() {
            max_torque += pod.reaction_wheel_torque as f64;
        }
        let mass = live_part_total_mass_kg(part) * surface_multiplier(part.get::<SurfaceMount>());
        if mass <= 0.0 {
            continue;
        }
        let position = part
            .get::<Transform>()
            .map(|t| t.translation.as_dvec3())
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

/// Per-stage readout, published each frame from the live parts and the
/// [`StagingPlan`] and consumed by the bottom-right HUD panel. Empty when
/// there is no staged vessel (e.g. EVA). Each [`StageSummary`] is the shared
/// type from [`thalos_shipyard::staging`], so the HUD and the shipyard
/// editor's preview render the same shape.
///
/// **Sole writer:** [`publish_staging_summaries`].
#[derive(Resource, Default)]
pub struct StagingSummaries(pub Vec<StageSummary>);

/// Gather the live parts + plan into the pure summary inputs, compute, and
/// publish [`StagingSummaries`] for the HUD. Per-part dry mass comes from the
/// single `thalos_shipyard` live-part enumeration ([`live_part_dry_mass_kg`]),
/// so the HUD Δv readout counts every part kind the flight mass does.
fn publish_staging_summaries(
    plans: Query<&StagingPlan>,
    parts: PartQuery,
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

    for part in parts.iter() {
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
