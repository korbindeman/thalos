//! Engine activation, throttle input, and per-frame fuel consumption.
//!
//! The simulation knows nothing about reactant pools — its
//! [`ControlInput::throttle`] is a 0..1 multiplier on `thrust_n / mass`,
//! and it tracks ship mass internally. This module is the bridge
//! between that abstract throttle and the ECS-side per-tank reactant
//! state in [`PartResources`]:
//!
//! 1. Sample keyboard (Shift/Ctrl/Z/X, KSP convention) into
//!    [`ThrottleState::commanded`]. The scheduled-burn autopilot
//!    (`crates/game/src/autopilot.rs`) writes the same field while
//!    flying a scheduled directive — programmatic and manual control
//!    share the throttle surface.
//! 2. Cap the throttle to 0 when any required reactant runs out and
//!    push the gated value into the simulation via
//!    [`Simulation::set_throttle`]. Engine fires only at 1× warp;
//!    every other warp level forces the gated throttle to 0.
//! 3. Reconcile tank pools to whatever mass the simulation actually
//!    burned last frame. The simulation is the **single source of
//!    truth** for ship mass — its propagator integrates burn duration,
//!    SOI clipping, and dry-mass cutoff exactly. We read back the mass
//!    delta and drain tank pools by that amount, split across the
//!    engine set that actually fired, so the per-tank UI matches the
//!    physics regardless of short burns inside the frame.
//!
//! Engine activation is not staging. It is the lower-level capability:
//! enabled engines contribute to the scalar propulsion summary consumed
//! by physics and prediction; disabled engines stay cold. A later
//! staging system can mutate [`EngineActivation`] without changing the
//! fuel or physics bridge.

use std::collections::{HashMap, VecDeque};

use bevy::prelude::*;
use thalos_shipyard::{
    Adapter, Attachment, CommandPod, Decoupler, Engine, EngineActivation, FuelCrossfeed, FuelTank,
    G0, Part, PartResources, Resource,
};

use crate::SimStage;
use crate::autopilot::Autopilot;
use crate::controls::ControlLocks;
use crate::rendering::SimulationState;

/// Per-engine flow recipe for an enabled engine. Stored separately from
/// [`Engine`] so the drain reconciliation can use the recipe that was in
/// force during the previous physics step, even if activation changes
/// before the next frame drains tanks.
#[derive(Debug, Clone)]
pub struct ActiveEngineFlow {
    pub entity: Entity,
    pub mass_flow_kg_per_s: f64,
    pub reactants: Vec<(Resource, f64)>,
}

/// Current scalar propulsion and mass state derived from the ECS ship.
///
/// This is rebuilt before each simulation step from the currently
/// enabled engines and live resource pools. Physics still consumes a
/// scalar aggregate for this pass; the ECS side owns which engines
/// produced that aggregate and which tanks they can reach.
#[derive(Resource, Debug, Clone, Default)]
pub struct ActivePropulsion {
    /// Sum of every current part's dry mass, kg.
    pub dry_mass_kg: f64,
    /// Dry mass plus all mass-bearing resources currently stored.
    pub wet_mass_kg: f64,
    /// Total thrust at full throttle across enabled engines, N.
    pub total_thrust_n: f64,
    /// Total mass flow at full throttle across enabled engines, kg/s.
    pub mass_flow_kg_per_s: f64,
    /// Summed electrical draw while all enabled engines fire, kW.
    pub power_draw_kw: f64,
    /// What fraction of the total mass flow each reactant accounts
    /// for across enabled engines. Sums to 1 when any enabled engine has
    /// positive flow; empty otherwise.
    pub reactant_fractions: HashMap<Resource, f64>,
    /// Enabled engines that currently contribute to propulsion.
    pub engines: Vec<ActiveEngineFlow>,
}

#[derive(Resource, Debug, Clone, Default)]
pub(crate) struct LastBurnRecipe {
    engines: Vec<ActiveEngineFlow>,
}

/// Player throttle state, persisted across frames so Shift/Ctrl can
/// ramp continuously.
///
/// `commanded` is what the player asked for; `effective` is what was
/// actually applied this frame after the fuel-availability cap (the
/// two diverge only when a tank is running dry). HUD reads both.
#[derive(Resource, Debug, Default)]
pub struct ThrottleState {
    pub commanded: f64,
    pub effective: f64,
}

pub struct FuelPlugin;

impl Plugin for FuelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActivePropulsion>()
            .init_resource::<LastBurnRecipe>()
            .init_resource::<ThrottleState>()
            .add_systems(
                Update,
                (
                    // Sample the keyboard early so the scheduled-burn
                    // autopilot in `crates/game/src/autopilot.rs` can
                    // run between this and `handle_attitude_controls`,
                    // overriding `ThrottleState::commanded` for the
                    // upcoming gate step. Detached from the
                    // reconcile/gate chain because it has no data
                    // dependency on attitude.
                    handle_throttle_input
                        .in_set(SimStage::Physics)
                        .before(crate::bridge::handle_attitude_controls),
                    // Reconcile first: at this point the simulation's
                    // `ship_mass_kg` reflects the previous frame's step,
                    // so any burn that fired during that step is now
                    // accounted for in the sim mass. Subtract the delta
                    // from the tanks before we gate the upcoming throttle
                    // so the gate sees the post-drain availability.
                    (
                        reconcile_tanks_from_sim_drain,
                        refresh_active_propulsion,
                        gate_throttle_on_fuel_availability,
                    )
                        .chain()
                        .in_set(SimStage::Physics)
                        .after(crate::bridge::handle_attitude_controls)
                        .before(crate::bridge::advance_simulation),
                ),
            );
    }
}

/// How fast the throttle ramps when Shift/Ctrl is held, in fraction
/// per second. KSP uses something close to this — full ramp takes
/// ~2 s, fast enough for combat-style adjustments and slow enough to
/// hit a target setting without overshoot.
const THROTTLE_RAMP_RATE: f64 = 0.5;

/// State shared across the throttle/drain systems for prediction
/// management. Tracks the previous frame's effective throttle so the
/// engine-cut edge can dirty the prediction one final time after the
/// burn ends. Held as a [`Local`] so it's per-system and survives
/// across frames.
#[derive(Default)]
pub struct PredictionRefresh {
    prev_effective: f64,
}

/// Decide whether the throttle edge should invalidate the cached
/// prediction. Manual thrust keeps the live trail responsive; scheduled
/// maneuver burns hold their already-planned trajectory steady until
/// the engine cuts.
fn should_mark_prediction_dirty(
    active: bool,
    cut_edge: bool,
    hold_during_scheduled_burn: bool,
) -> bool {
    cut_edge || (active && !hold_during_scheduled_burn)
}

/// Push the effective throttle into both the bridge state resource
/// and the simulation, and dirty the prediction for manual thrust
/// updates or the final engine-cut edge after a scheduled burn.
///
/// `apply_live_thrust` in the simulation deliberately doesn't dirty
/// at all; this fn owns the policy so the bridge can tune it
/// without touching physics. Per-frame dirtying for scheduled burns
/// would rebuild the node trajectory from the mid-burn live ship state,
/// making the target orbit visibly drift while the maneuver executes.
fn finish_with_throttle(
    effective: f64,
    throttle: &mut ThrottleState,
    sim: &mut SimulationState,
    refresh: &mut PredictionRefresh,
    hold_during_scheduled_burn: bool,
) {
    throttle.effective = effective;
    sim.simulation.set_throttle(effective);

    let active = effective > 0.0;
    let cut_edge = refresh.prev_effective > 0.0 && !active;
    if should_mark_prediction_dirty(active, cut_edge, hold_during_scheduled_burn) {
        sim.simulation.prediction_state.mark_dirty();
    }

    refresh.prev_effective = effective;
}

/// Sample keyboard → [`ThrottleState::commanded`].
///
/// - `Z` snaps to full
/// - `X` snaps to zero
/// - `Shift` (held) ramps up
/// - `Ctrl` (held) ramps down
///
/// Gated by [`ControlLocks::throttle`] — when set, player input is
/// dropped because some programmatic system (today the scheduled-burn
/// autopilot) owns the throttle. See [`crate::controls`] for who
/// publishes the locks.
pub fn handle_throttle_input(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    locks: Res<ControlLocks>,
    mut throttle: ResMut<ThrottleState>,
) {
    if locks.throttle {
        return;
    }

    if keys.just_pressed(KeyCode::KeyZ) {
        throttle.commanded = 1.0;
        return;
    }
    if keys.just_pressed(KeyCode::KeyX) {
        throttle.commanded = 0.0;
        return;
    }

    let dt = time.delta_secs_f64();
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    if shift {
        throttle.commanded = (throttle.commanded + THROTTLE_RAMP_RATE * dt).min(1.0);
    }
    if ctrl {
        throttle.commanded = (throttle.commanded - THROTTLE_RAMP_RATE * dt).max(0.0);
    }
}

type DryMassQuery<'w, 's> = Query<
    'w,
    's,
    (
        Option<&'static CommandPod>,
        Option<&'static Decoupler>,
        Option<&'static Adapter>,
        Option<&'static FuelTank>,
        Option<&'static Engine>,
    ),
    With<Part>,
>;

fn part_dry_mass_kg(
    pod: Option<&CommandPod>,
    dec: Option<&Decoupler>,
    adapter: Option<&Adapter>,
    tank: Option<&FuelTank>,
    engine: Option<&Engine>,
) -> f64 {
    if let Some(pod) = pod {
        pod.dry_mass as f64
    } else if let Some(dec) = dec {
        dec.dry_mass as f64
    } else if let Some(adapter) = adapter {
        adapter.dry_mass as f64
    } else if let Some(tank) = tank {
        tank.dry_mass as f64
    } else if let Some(engine) = engine {
        engine.dry_mass as f64
    } else {
        0.0
    }
}

fn propulsion_config_changed(prev: &ActivePropulsion, next: &ActivePropulsion) -> bool {
    const EPS: f64 = 1e-6;
    (prev.total_thrust_n - next.total_thrust_n).abs() > EPS
        || (prev.mass_flow_kg_per_s - next.mass_flow_kg_per_s).abs() > EPS
        || (prev.power_draw_kw - next.power_draw_kw).abs() > EPS
        || (prev.dry_mass_kg - next.dry_mass_kg).abs() > EPS
        || prev.engines.len() != next.engines.len()
        || prev
            .engines
            .iter()
            .zip(&next.engines)
            .any(|(a, b)| a.entity != b.entity)
}

/// Rebuild the active propulsion summary from enabled engines and live
/// resource pools, then push the scalar fields into physics. This keeps
/// the existing propagator contract intact while letting gameplay decide
/// which engines currently exist from the craft's perspective.
fn refresh_active_propulsion(
    mut sim: ResMut<SimulationState>,
    mut active: ResMut<ActivePropulsion>,
    engines: Query<(Entity, &Engine, Option<&EngineActivation>)>,
    parts: DryMassQuery,
    resources: Query<&PartResources>,
) {
    let dry_mass_kg: f64 = parts
        .iter()
        .map(|(pod, dec, adapter, tank, engine)| part_dry_mass_kg(pod, dec, adapter, tank, engine))
        .sum();

    let resource_mass_kg: f64 = resources
        .iter()
        .flat_map(|part| part.pools.iter())
        .map(|(&res, pool)| pool.mass_kg(res))
        .sum();

    let mut total_thrust_n = 0.0_f64;
    let mut mass_flow_kg_per_s = 0.0_f64;
    let mut power_draw_kw = 0.0_f64;
    let mut per_resource_mdot: HashMap<Resource, f64> = HashMap::new();
    let mut active_engines = Vec::new();

    for (entity, engine, activation) in engines.iter() {
        if !activation.map(|a| a.enabled).unwrap_or(true) {
            continue;
        }
        if engine.thrust <= 0.0 || engine.isp <= 0.0 {
            continue;
        }

        let thrust_n = engine.thrust as f64;
        let mdot = thrust_n / (engine.isp as f64 * G0);

        let reactants: Vec<(Resource, f64)> = engine
            .reactants
            .iter()
            .filter(|r| r.mass_fraction > 0.0)
            .map(|r| {
                let frac = r.mass_fraction as f64;
                *per_resource_mdot.entry(r.resource).or_insert(0.0) += mdot * frac;
                (r.resource, frac)
            })
            .collect();
        if reactants.is_empty() {
            continue;
        }

        total_thrust_n += thrust_n;
        mass_flow_kg_per_s += mdot;
        power_draw_kw += engine.power_draw_kw as f64;

        active_engines.push(ActiveEngineFlow {
            entity,
            mass_flow_kg_per_s: mdot,
            reactants,
        });
    }

    let reactant_fractions = if mass_flow_kg_per_s > 0.0 {
        per_resource_mdot
            .into_iter()
            .map(|(res, mdot)| (res, mdot / mass_flow_kg_per_s))
            .collect()
    } else {
        HashMap::new()
    };

    let next = ActivePropulsion {
        dry_mass_kg,
        wet_mass_kg: dry_mass_kg + resource_mass_kg,
        total_thrust_n,
        mass_flow_kg_per_s,
        power_draw_kw,
        reactant_fractions,
        engines: active_engines,
    };
    let changed = propulsion_config_changed(&active, &next);
    *active = next;

    let mut params = *sim.simulation.ship_params();
    params.thrust_n = active.total_thrust_n;
    params.mass_flow_kg_per_s = active.mass_flow_kg_per_s;
    params.dry_mass_kg = active.dry_mass_kg;
    sim.simulation.set_ship_params(params);
    sim.simulation.set_ship_mass(active.wet_mass_kg);

    if changed {
        sim.simulation.prediction_state.mark_dirty();
    }
}

fn crossfeed_enabled(entity: Entity, crossfeeds: &Query<&FuelCrossfeed>) -> bool {
    crossfeeds.get(entity).map(|c| c.enabled).unwrap_or(true)
}

fn crossfeed_components(
    attachments: &Query<(Entity, &Attachment)>,
    crossfeeds: &Query<&FuelCrossfeed>,
) -> HashMap<Entity, usize> {
    let mut adjacency: HashMap<Entity, Vec<Entity>> = HashMap::new();

    for (child, attachment) in attachments.iter() {
        let parent = attachment.parent;
        adjacency.entry(child).or_default();
        adjacency.entry(parent).or_default();
        if crossfeed_enabled(child, crossfeeds) && crossfeed_enabled(parent, crossfeeds) {
            adjacency.entry(child).or_default().push(parent);
            adjacency.entry(parent).or_default().push(child);
        }
    }

    let mut component_by_entity = HashMap::new();
    let mut next_component = 0usize;
    for &start in adjacency.keys() {
        if component_by_entity.contains_key(&start) {
            continue;
        }

        let component = next_component;
        next_component += 1;

        let mut queue = VecDeque::from([start]);
        component_by_entity.insert(start, component);
        while let Some(entity) = queue.pop_front() {
            let Some(neighbors) = adjacency.get(&entity) else {
                continue;
            };
            for &neighbor in neighbors {
                if component_by_entity.contains_key(&neighbor) {
                    continue;
                }
                component_by_entity.insert(neighbor, component);
                queue.push_back(neighbor);
            }
        }
    }

    component_by_entity
}

fn component_for(entity: Entity, components: &HashMap<Entity, usize>) -> usize {
    components
        .get(&entity)
        .copied()
        .unwrap_or_else(|| 1_000_000 + entity.index_u32() as usize)
}

fn resource_mass_by_component(
    tanks: &Query<(Entity, &PartResources)>,
    components: &HashMap<Entity, usize>,
) -> HashMap<(usize, Resource), f64> {
    let mut available = HashMap::new();
    for (entity, tank) in tanks.iter() {
        let component = component_for(entity, components);
        for (&res, pool) in &tank.pools {
            *available.entry((component, res)).or_insert(0.0) += pool.mass_kg(res);
        }
    }
    available
}

fn engine_resource_requests(
    engines: &[ActiveEngineFlow],
    components: &HashMap<Entity, usize>,
    mass_scale: f64,
) -> HashMap<(usize, Resource), f64> {
    let mut requests = HashMap::new();
    if mass_scale <= 0.0 {
        return requests;
    }
    for engine in engines {
        let component = component_for(engine.entity, components);
        for &(res, frac) in &engine.reactants {
            if frac <= 0.0 {
                continue;
            }
            let req = engine.mass_flow_kg_per_s * frac * mass_scale;
            *requests.entry((component, res)).or_insert(0.0) += req;
        }
    }
    requests
}

fn drain_resource_from_component(
    tanks: &mut Query<(Entity, &mut PartResources)>,
    components: &HashMap<Entity, usize>,
    target_component: usize,
    resource: Resource,
    drain_kg: f64,
) {
    if drain_kg <= 0.0 {
        return;
    }
    let density = resource.density_kg_per_unit();
    if density <= 0.0 {
        return;
    }

    let available_kg: f64 = tanks
        .iter_mut()
        .filter(|(entity, _)| component_for(*entity, components) == target_component)
        .filter_map(|(_, tank)| tank.pools.get(&resource).map(|p| p.mass_kg(resource)))
        .sum();
    if available_kg <= 0.0 {
        return;
    }

    let drain_kg = drain_kg.min(available_kg);
    for (entity, mut tank) in tanks.iter_mut() {
        if component_for(entity, components) != target_component {
            continue;
        }
        let Some(pool) = tank.pools.get_mut(&resource) else {
            continue;
        };
        let pool_mass = pool.mass_kg(resource);
        if pool_mass <= 0.0 {
            continue;
        }
        let take_kg = drain_kg * (pool_mass / available_kg);
        let take_units = take_kg / density;
        pool.amount = ((pool.amount as f64 - take_units).max(0.0)) as f32;
    }
}

/// Drain tank reactant pools by however much mass the simulation burned
/// since this system last ran. The simulation's `ship_mass_kg` is the
/// authoritative record of "current ship mass" — it integrates short
/// burns inside long warp ticks correctly, while a naive
/// `mass_flow · real_dt · warp` overdrains the tanks. Reading the sim's
/// actual delta avoids that whole class of bug.
///
/// Drain follows the same crossfeed graph as throttle gating. Each
/// engine's mass-flow share drains only tanks in the component reachable
/// from that engine through [`FuelCrossfeed`] parts.
///
/// First-call behaviour: records the spawn-time ship mass and skips
/// drain. Without this, the very first frame would treat the ship's
/// entire propellant load as a one-frame burn.
fn reconcile_tanks_from_sim_drain(
    sim: Res<SimulationState>,
    last_burn: Res<LastBurnRecipe>,
    mut tanks: Query<(Entity, &mut PartResources)>,
    attachments: Query<(Entity, &Attachment)>,
    crossfeeds: Query<&FuelCrossfeed>,
    mut prev_sim_mass: Local<Option<f64>>,
) {
    let current = sim.simulation.ship_mass_kg();
    let Some(prev) = *prev_sim_mass else {
        *prev_sim_mass = Some(current);
        return;
    };
    *prev_sim_mass = Some(current);

    let drained_total_kg = (prev - current).max(0.0);
    if drained_total_kg <= 0.0 || last_burn.engines.is_empty() {
        return;
    }

    let total_mdot: f64 = last_burn.engines.iter().map(|e| e.mass_flow_kg_per_s).sum();
    if total_mdot <= 0.0 {
        return;
    }

    let components = crossfeed_components(&attachments, &crossfeeds);
    let requests = engine_resource_requests(
        &last_burn.engines,
        &components,
        drained_total_kg / total_mdot,
    );
    for ((component, resource), drain_kg) in requests {
        drain_resource_from_component(&mut tanks, &components, component, resource, drain_kg);
    }
}

/// Cap throttle on fuel shortage and push the gated value into the
/// simulation. No drain happens here — the simulation drains its own
/// `ship_mass_kg` during its step, and [`reconcile_tanks_from_sim_drain`]
/// drains the tanks by that delta on the next frame.
///
/// Engine fires only at 1× warp. Any other warp level (including
/// pause) gates the throttle to 0; the scheduled-burn autopilot is
/// responsible for stepping warp down to 1× before opening the
/// throttle on a scheduled maneuver.
pub fn gate_throttle_on_fuel_availability(
    time: Res<Time>,
    active: Res<ActivePropulsion>,
    autopilot: Res<Autopilot>,
    mut sim: ResMut<SimulationState>,
    mut throttle: ResMut<ThrottleState>,
    tanks: Query<(Entity, &PartResources)>,
    attachments: Query<(Entity, &Attachment)>,
    crossfeeds: Query<&FuelCrossfeed>,
    mut last_burn: ResMut<LastBurnRecipe>,
    mut refresh: Local<PredictionRefresh>,
) {
    let warp = sim.simulation.warp.speed();
    let real_dt = time.delta_secs_f64();
    let hold_prediction = autopilot.is_burning();

    if (warp - 1.0).abs() > f64::EPSILON {
        last_burn.engines.clear();
        finish_with_throttle(0.0, &mut throttle, &mut sim, &mut refresh, hold_prediction);
        return;
    }

    let throttle_in_use = throttle.commanded;
    let drain_dt = real_dt;

    if throttle_in_use <= 0.0
        || drain_dt <= 0.0
        || active.mass_flow_kg_per_s <= 0.0
        || active.reactant_fractions.is_empty()
    {
        last_burn.engines.clear();
        finish_with_throttle(
            throttle_in_use,
            &mut throttle,
            &mut sim,
            &mut refresh,
            hold_prediction,
        );
        return;
    }

    let components = crossfeed_components(&attachments, &crossfeeds);
    let available = resource_mass_by_component(&tanks, &components);
    let requests =
        engine_resource_requests(&active.engines, &components, throttle_in_use * drain_dt);

    let mut feasibility = 1.0_f64;
    for (&key, &req) in &requests {
        if req <= 0.0 {
            continue;
        }
        let avail = available.get(&key).copied().unwrap_or(0.0);
        if req > avail {
            feasibility = feasibility.min(avail / req);
        }
    }

    let throttle_effective = (throttle_in_use * feasibility).max(0.0);
    if throttle_effective > 0.0 {
        last_burn.engines = active.engines.clone();
    } else {
        last_burn.engines.clear();
    }

    finish_with_throttle(
        throttle_effective,
        &mut throttle,
        &mut sim,
        &mut refresh,
        hold_prediction,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::World;

    #[test]
    fn scheduled_burn_holds_prediction_while_thrusting() {
        assert!(!should_mark_prediction_dirty(true, false, true));
    }

    #[test]
    fn scheduled_burn_refreshes_on_engine_cut() {
        assert!(should_mark_prediction_dirty(false, true, true));
    }

    #[test]
    fn manual_burn_refreshes_while_thrusting() {
        assert!(should_mark_prediction_dirty(true, false, false));
    }

    #[test]
    fn engine_resource_requests_split_by_reactant_fraction() {
        let mut world = World::new();
        let engine = world.spawn_empty().id();
        let components = HashMap::from([(engine, 7usize)]);
        let engines = vec![ActiveEngineFlow {
            entity: engine,
            mass_flow_kg_per_s: 10.0,
            reactants: vec![(Resource::Methane, 0.25), (Resource::Lox, 0.75)],
        }];

        let requests = engine_resource_requests(&engines, &components, 2.0);

        assert_eq!(requests.get(&(7, Resource::Methane)).copied(), Some(5.0));
        assert_eq!(requests.get(&(7, Resource::Lox)).copied(), Some(15.0));
    }
}
