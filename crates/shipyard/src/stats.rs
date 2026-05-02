//! Aggregate physical properties of a ship blueprint.
//!
//! [`ShipStats`] is the bridge between the parametric editor representation
//! ([`ShipBlueprint`]) and any consumer that needs scalar physical quantities
//! — primarily the physics simulation. All quantities are in SI units
//! (kg, N, s, m/s², kW, kWh) and stored as `f64` to match the physics crate.
//!
//! # Engine propellant model
//!
//! Each engine declares a list of [`ReactantRatio`]: mass fractions of the
//! reactants it expels, summing to 1. At full throttle an engine's mass
//! flow rate is `thrust / (isp · g₀)`; each reactant's consumption rate is
//! that mass flow times its mass fraction. Across a multi-engine ship,
//! per-resource consumption rates are summed.
//!
//! Δv capacity is Tsiolkovsky applied to the expellable mass, which is
//! limited by whichever reactant runs out first. Electricity is not a
//! reactant — it never enters the rocket equation; engines instead declare
//! a continuous `power_draw_kw` for the duration of the burn.

use crate::blueprint::{Connection, PartBlueprint, PartParams, ShipBlueprint, pools_for};
use crate::catalog::{
    CatalogEntry, CatalogError, PartCatalog, adapter_surface_area, tank_surface_area,
};
use crate::part::ReactantRatio;
use crate::resource::{PartResources, Resource};
use bevy::math::Vec3;
use glam::DVec3;
use std::collections::{HashMap, VecDeque};

/// Standard gravity, m/s². Used to convert between Isp (s) and exhaust
/// velocity (m/s).
pub const G0: f64 = 9.806_65;

/// Per-resource aggregate across every pool on the ship.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResourceTotals {
    /// Current amount in the resource's native unit (L or kWh).
    pub amount: f64,
    /// Storage capacity in the same native unit.
    pub capacity: f64,
    /// Current mass contribution — 0 for non-mass-bearing resources.
    pub mass_kg: f64,
}

/// Environment used for a delta-v estimate.
///
/// Only vacuum is implemented today because engine definitions currently
/// expose vacuum thrust/Isp. Atmospheric estimates can extend this enum
/// with pressure/altitude/body context without changing UI call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaVEnvironment {
    Vacuum,
}

/// Inputs consumed by the delta-v calculator.
#[derive(Debug, Clone, Copy)]
pub struct DeltaVInputs<'a> {
    pub dry_mass_kg: f64,
    pub wet_mass_kg: f64,
    pub total_thrust_n: f64,
    pub mass_flow_kg_per_s: f64,
    pub power_draw_kw: f64,
    pub reactant_fractions: &'a HashMap<Resource, f64>,
    pub resources: &'a HashMap<Resource, ResourceTotals>,
}

/// Result of a delta-v estimate in a specific environment.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeltaVEstimate {
    pub delta_v_m_per_s: f64,
    pub burn_time_s: Option<f64>,
    pub exhaust_velocity_m_per_s: f64,
    pub initial_mass_kg: f64,
    pub final_mass_kg: f64,
    pub expelled_mass_kg: f64,
}

/// Aggregate all runtime resource pools into the same totals shape used
/// by [`ShipStats`].
pub fn aggregate_resource_totals<'a>(
    resources: impl IntoIterator<Item = &'a PartResources>,
) -> HashMap<Resource, ResourceTotals> {
    let mut totals: HashMap<Resource, ResourceTotals> = HashMap::new();
    for part in resources {
        for (&resource, pool) in &part.pools {
            let entry = totals.entry(resource).or_default();
            entry.amount += pool.amount as f64;
            entry.capacity += pool.capacity as f64;
            entry.mass_kg += pool.mass_kg(resource);
        }
    }
    totals
}

/// Estimate available delta-v for the supplied ship state.
pub fn estimate_delta_v(
    environment: DeltaVEnvironment,
    inputs: DeltaVInputs<'_>,
) -> DeltaVEstimate {
    match environment {
        DeltaVEnvironment::Vacuum => estimate_vacuum_delta_v(inputs),
    }
}

fn estimate_vacuum_delta_v(inputs: DeltaVInputs<'_>) -> DeltaVEstimate {
    let wet = inputs.wet_mass_kg.max(0.0);
    let exhaust_velocity = if inputs.mass_flow_kg_per_s > 0.0 {
        inputs.total_thrust_n / inputs.mass_flow_kg_per_s
    } else {
        0.0
    };
    let mut estimate = DeltaVEstimate {
        exhaust_velocity_m_per_s: exhaust_velocity,
        initial_mass_kg: wet,
        final_mass_kg: wet,
        ..Default::default()
    };

    if exhaust_velocity <= 0.0
        || wet <= 0.0
        || inputs.dry_mass_kg <= 0.0
        || inputs.mass_flow_kg_per_s <= 0.0
    {
        return estimate;
    }

    let Some(burn_s) = burn_time_limit_s(inputs) else {
        return estimate;
    };
    estimate.burn_time_s = Some(burn_s);

    let raw_expelled = inputs.mass_flow_kg_per_s * burn_s;
    if raw_expelled <= 0.0 {
        return estimate;
    }

    let max_expelled = (wet - inputs.dry_mass_kg).max(0.0);
    let expelled = raw_expelled.min(max_expelled);
    let remaining = wet - expelled;
    estimate.expelled_mass_kg = expelled;
    estimate.final_mass_kg = remaining;

    if remaining > 0.0 && remaining < wet {
        estimate.delta_v_m_per_s = exhaust_velocity * (wet / remaining).ln();
    }

    estimate
}

fn burn_time_limit_s(inputs: DeltaVInputs<'_>) -> Option<f64> {
    if inputs.mass_flow_kg_per_s <= 0.0 {
        return None;
    }

    let mut limit = f64::INFINITY;

    for (res, frac) in inputs.reactant_fractions {
        if *frac <= 0.0 {
            continue;
        }
        let rate_kg_per_s = inputs.mass_flow_kg_per_s * frac;
        let available = inputs.resources.get(res).map(|r| r.mass_kg).unwrap_or(0.0);
        limit = limit.min(available / rate_kg_per_s);
    }

    if inputs.power_draw_kw > 0.0 {
        let stored_kwh = inputs
            .resources
            .get(&Resource::Electricity)
            .map(|r| r.amount)
            .unwrap_or(0.0);
        let power_limit_s = stored_kwh / inputs.power_draw_kw * 3600.0;
        limit = limit.min(power_limit_s);
    }

    if limit.is_finite() { Some(limit) } else { None }
}

/// Snapshot of a ship's mass, thrust, and propulsion characteristics.
///
/// Derived from a [`ShipBlueprint`] via [`ShipBlueprint::stats`].
#[derive(Debug, Clone, Default)]
pub struct ShipStats {
    pub dry_mass_kg: f64,
    /// Sum across all pools of (amount × resource density) — only
    /// mass-bearing resources contribute.
    pub propellant_mass_kg: f64,
    pub total_thrust_n: f64,
    /// Mass-flow weighted Isp (s) across all engines at full throttle.
    /// Zero when no engines are present.
    pub combined_isp_s: f64,
    /// Total mass flow at full throttle (kg/s).
    pub mass_flow_kg_per_s: f64,
    /// Summed electrical draw while all engines fire, kW.
    pub power_draw_kw: f64,
    /// Aggregate reactant mass fractions across all engines. Sums to 1
    /// when any engine is present; empty otherwise.
    pub reactant_fractions: HashMap<Resource, f64>,
    /// Snapshot of every resource pool on the ship, aggregated by kind.
    pub resources: HashMap<Resource, ResourceTotals>,
    /// Principal-axis moment of inertia about the ship CoM, kg·m². Each
    /// part is approximated as a uniform solid cylinder along the body
    /// Y axis with `r = effective_diameter/2` and `L` from its visual
    /// height; per-part inertia is then shifted to the ship CoM via
    /// the parallel-axis theorem. Off-diagonal terms are ignored —
    /// adequate for axially-symmetric stacks.
    pub moment_of_inertia_kg_m2: DVec3,
    /// Sum of every [`crate::ReactionWheel`]'s `max_torque`, in N·m
    /// per body axis. Symmetric — the per-axis cap is the same on all
    /// three. Per-axis-asymmetric torque is reserved for RCS arrangements.
    pub max_reaction_torque_n_m: f64,
}

impl ShipStats {
    pub fn wet_mass_kg(&self) -> f64 {
        self.dry_mass_kg + self.propellant_mass_kg
    }

    /// Acceleration at current wet mass and full throttle, m/s².
    pub fn current_acceleration(&self) -> f64 {
        let m = self.wet_mass_kg();
        if m > 0.0 {
            self.total_thrust_n / m
        } else {
            0.0
        }
    }

    /// Exhaust velocity = Isp · g₀ (m/s).
    pub fn exhaust_velocity(&self) -> f64 {
        self.combined_isp_s * G0
    }

    /// Vacuum delta-v estimate from the current resource state.
    pub fn vacuum_delta_v(&self) -> DeltaVEstimate {
        estimate_delta_v(
            DeltaVEnvironment::Vacuum,
            DeltaVInputs {
                dry_mass_kg: self.dry_mass_kg,
                wet_mass_kg: self.wet_mass_kg(),
                total_thrust_n: self.total_thrust_n,
                mass_flow_kg_per_s: self.mass_flow_kg_per_s,
                power_draw_kw: self.power_draw_kw,
                reactant_fractions: &self.reactant_fractions,
                resources: &self.resources,
            },
        )
    }

    /// Burn time at full throttle before the bottleneck reactant or
    /// electricity runs out. Returns `None` when there is no thrust or
    /// no burnable propellant.
    pub fn burn_time_at_full_throttle_s(&self) -> Option<f64> {
        self.vacuum_delta_v().burn_time_s
    }

    /// Tsiolkovsky Δv available from current state, limited by the
    /// bottleneck reactant (and/or stored electricity if power-dependent).
    /// Returns 0 when any critical input is missing.
    pub fn delta_v_capacity(&self) -> f64 {
        self.vacuum_delta_v().delta_v_m_per_s
    }
}

impl ShipBlueprint {
    /// Compute aggregate stats from this blueprint by resolving every
    /// part against `catalog`. Returns [`CatalogError`] on the first
    /// unknown ID or mismatched [`PartParams`] variant.
    pub fn stats(&self, catalog: &PartCatalog) -> Result<ShipStats, CatalogError> {
        let mut dry_mass_kg = 0.0_f64;
        let mut total_thrust_n = 0.0_f64;
        // Σ (thrust / isp) — denominator of mass-flow-weighted Isp.
        let mut thrust_over_isp = 0.0_f64;
        let mut power_draw_kw = 0.0_f64;
        let mut per_resource_mdot: HashMap<Resource, f64> = HashMap::new();
        let mut max_reaction_torque_n_m = 0.0_f64;

        // Pre-resolve every entry once; bail on the first error.
        let entries: Vec<&CatalogEntry> = self
            .parts
            .iter()
            .map(|pb| catalog.resolve(&pb.catalog_id))
            .collect::<Result<_, _>>()?;

        for (pb, entry) in self.parts.iter().zip(&entries) {
            dry_mass_kg += part_dry_mass(entry, &pb.params) as f64;

            if let CatalogEntry::Engine(e) = entry {
                let t = e.thrust as f64;
                total_thrust_n += t;
                power_draw_kw += e.power_draw_kw as f64;
                if e.isp > 0.0 {
                    let isp_f = e.isp as f64;
                    thrust_over_isp += t / isp_f;
                    let mdot = t / (isp_f * G0);
                    accumulate_engine_reactants(&e.reactants, mdot, &mut per_resource_mdot);
                }
            }

            if let CatalogEntry::Pod(p) = entry {
                max_reaction_torque_n_m += p.reaction_wheel_torque as f64;
            }
        }

        // Aggregate pools by resource. Capacities come from catalog ×
        // params; amounts come from blueprint overrides (or default full).
        let mut resources: HashMap<Resource, ResourceTotals> = HashMap::new();
        for (pb, entry) in self.parts.iter().zip(&entries) {
            let pools = pools_for(entry, &pb.params, &pb.resources);
            for (res, pool) in pools {
                let e = resources.entry(res).or_default();
                e.amount += pool.amount as f64;
                e.capacity += pool.capacity as f64;
                e.mass_kg += pool.mass_kg(res);
            }
        }

        let propellant_mass_kg: f64 = resources.values().map(|r| r.mass_kg).sum();

        let combined_isp_s = if thrust_over_isp > 0.0 {
            total_thrust_n / thrust_over_isp
        } else {
            0.0
        };
        let mass_flow_kg_per_s = if combined_isp_s > 0.0 {
            total_thrust_n / (combined_isp_s * G0)
        } else {
            0.0
        };

        let reactant_fractions: HashMap<Resource, f64> = if mass_flow_kg_per_s > 0.0 {
            per_resource_mdot
                .into_iter()
                .map(|(r, m)| (r, m / mass_flow_kg_per_s))
                .collect()
        } else {
            HashMap::new()
        };

        let geo = ship_geometry(self, &entries);

        let mut com_total_mass = 0.0_f64;
        let mut com_weighted = DVec3::ZERO;
        for (i, (pb, entry)) in self.parts.iter().zip(&entries).enumerate() {
            let m = part_total_mass(pb, entry);
            com_total_mass += m;
            com_weighted += geo[i].position * m;
        }
        let com = if com_total_mass > 0.0 {
            com_weighted / com_total_mass
        } else {
            DVec3::ZERO
        };

        let mut moment_of_inertia_kg_m2 = DVec3::ZERO;
        for (i, (pb, entry)) in self.parts.iter().zip(&entries).enumerate() {
            let m = part_total_mass(pb, entry);
            let (r, l) = part_cylinder_dims(entry, &pb.params, geo[i].diameter);
            // Solid cylinder, long axis = body Y:
            //   I_yy = m·r²/2
            //   I_xx = I_zz = m·(3r² + L²)/12
            let i_yy_self = m * r * r * 0.5;
            let i_xz_self = m * (3.0 * r * r + l * l) / 12.0;

            let d = geo[i].position - com;
            let par = DVec3::new(
                m * (d.y * d.y + d.z * d.z),
                m * (d.x * d.x + d.z * d.z),
                m * (d.x * d.x + d.y * d.y),
            );

            moment_of_inertia_kg_m2 +=
                DVec3::new(i_xz_self + par.x, i_yy_self + par.y, i_xz_self + par.z);
        }

        Ok(ShipStats {
            dry_mass_kg,
            propellant_mass_kg,
            total_thrust_n,
            combined_isp_s,
            mass_flow_kg_per_s,
            power_draw_kw,
            reactant_fractions,
            resources,
            moment_of_inertia_kg_m2,
            max_reaction_torque_n_m,
        })
    }
}

// ---------------------------------------------------------------------------
// Geometry — per-part CoM positions in ship body frame
//
// `ship_geometry` mirrors the runtime's BFS in `sizing::propagate_node_sizes`
// + `update_ship_part_transforms`, but operates on blueprint indices instead
// of ECS entities. This keeps the inertia model honest for blueprints that
// rely on parametric diameter inheritance.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct PartGeometry {
    /// Position in ship body frame, metres. The root sits at the origin.
    position: DVec3,
    /// Effective outer diameter after parametric inheritance from the
    /// parent's mating node, metres.
    diameter: f32,
}

fn ship_geometry(blueprint: &ShipBlueprint, entries: &[&CatalogEntry]) -> Vec<PartGeometry> {
    let mut geo: Vec<PartGeometry> = blueprint
        .parts
        .iter()
        .zip(entries.iter())
        .map(|(pb, entry)| PartGeometry {
            position: DVec3::ZERO,
            diameter: declared_diameter(entry, &pb.params),
        })
        .collect();

    let mut children_map: HashMap<usize, Vec<&Connection>> = HashMap::new();
    for c in &blueprint.connections {
        children_map.entry(c.parent).or_default().push(c);
    }

    let mut visited = vec![false; blueprint.parts.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    if blueprint.root < visited.len() {
        visited[blueprint.root] = true;
        queue.push_back(blueprint.root);
    }

    while let Some(parent_idx) = queue.pop_front() {
        let parent_pb = &blueprint.parts[parent_idx];
        let parent_entry = entries[parent_idx];
        let parent_d = geo[parent_idx].diameter;
        let parent_pos = geo[parent_idx].position;
        let Some(kids) = children_map.get(&parent_idx) else {
            continue;
        };
        for c in kids {
            if c.child >= visited.len() || visited[c.child] {
                continue;
            }
            visited[c.child] = true;

            let child_pb = &blueprint.parts[c.child];
            let child_entry = entries[c.child];

            if is_parametric(child_entry)
                && let Some(input_d) =
                    node_diameter(parent_entry, &parent_pb.params, parent_d, &c.parent_node)
            {
                geo[c.child].diameter =
                    effective_diameter_for(child_entry, &child_pb.params, input_d);
            }

            let parent_offset =
                node_offset(parent_entry, &parent_pb.params, parent_d, &c.parent_node)
                    .unwrap_or(Vec3::ZERO);
            let child_offset = node_offset(
                child_entry,
                &child_pb.params,
                geo[c.child].diameter,
                &c.child_node,
            )
            .unwrap_or(Vec3::ZERO);
            geo[c.child].position = parent_pos + (parent_offset - child_offset).as_dvec3();

            queue.push_back(c.child);
        }
    }

    geo
}

fn declared_diameter(entry: &CatalogEntry, params: &PartParams) -> f32 {
    match (entry, params) {
        (CatalogEntry::Pod(p), _) => p.diameter,
        (CatalogEntry::Engine(e), _) => e.diameter,
        (CatalogEntry::Decoupler(_), PartParams::Decoupler { diameter }) => *diameter,
        (CatalogEntry::Adapter(_), PartParams::Adapter { diameter, .. }) => *diameter,
        (CatalogEntry::Tank(_), PartParams::Tank { diameter, .. }) => *diameter,
        _ => 0.0,
    }
}

fn is_parametric(entry: &CatalogEntry) -> bool {
    matches!(
        entry,
        CatalogEntry::Tank(_) | CatalogEntry::Adapter(_) | CatalogEntry::Decoupler(_)
    )
}

/// Effective single-cylinder diameter after a parametric child inherits
/// its top diameter from the parent. Adapters are tapered, so we average
/// top (= parent's mating diameter) and bottom (= `target_diameter`) —
/// the cylinder model can't represent both anyway.
fn effective_diameter_for(entry: &CatalogEntry, params: &PartParams, parent_node_d: f32) -> f32 {
    match (entry, params) {
        (
            CatalogEntry::Adapter(_),
            PartParams::Adapter {
                target_diameter, ..
            },
        ) => (parent_node_d + *target_diameter) * 0.5,
        _ => parent_node_d,
    }
}

/// Diameter of the named attach node on this part, given its (possibly
/// propagated) effective body diameter. Mirrors
/// [`crate::blueprint::nodes_for`] but with the propagated diameter
/// substituted for the declared one.
fn node_diameter(
    entry: &CatalogEntry,
    params: &PartParams,
    effective_d: f32,
    node: &str,
) -> Option<f32> {
    match (entry, params) {
        (CatalogEntry::Pod(_), _) => (node == "bottom").then_some(effective_d),
        (CatalogEntry::Engine(_), _) => (node == "top" || node == "bottom").then_some(effective_d),
        (CatalogEntry::Decoupler(_), _) | (CatalogEntry::Tank(_), _) => {
            (node == "top" || node == "bottom").then_some(effective_d)
        }
        (
            CatalogEntry::Adapter(_),
            PartParams::Adapter {
                target_diameter, ..
            },
        ) => match node {
            "top" => Some(effective_d),
            "bottom" => Some(*target_diameter),
            _ => None,
        },
        _ => None,
    }
}

/// Offset of the named attach node, in the part's local frame (Y points
/// out the top). Mirrors [`crate::blueprint::nodes_for`].
fn node_offset(
    entry: &CatalogEntry,
    params: &PartParams,
    effective_d: f32,
    node: &str,
) -> Option<Vec3> {
    match (entry, params) {
        (CatalogEntry::Pod(_), _) => {
            (node == "bottom").then_some(Vec3::new(0.0, -effective_d * 0.9, 0.0))
        }
        (CatalogEntry::Engine(_), _) => match node {
            "top" => Some(Vec3::ZERO),
            "bottom" => Some(Vec3::new(0.0, -effective_d * 0.9, 0.0)),
            _ => None,
        },
        (CatalogEntry::Decoupler(_), _) => match node {
            "top" => Some(Vec3::ZERO),
            "bottom" => Some(Vec3::new(0.0, -0.2, 0.0)),
            _ => None,
        },
        (
            CatalogEntry::Adapter(_),
            PartParams::Adapter {
                target_diameter, ..
            },
        ) => match node {
            "top" => Some(Vec3::ZERO),
            "bottom" => {
                let h = ((effective_d + *target_diameter) * 0.5).max(0.4);
                Some(Vec3::new(0.0, -h, 0.0))
            }
            _ => None,
        },
        (CatalogEntry::Tank(_), PartParams::Tank { length, .. }) => match node {
            "top" => Some(Vec3::ZERO),
            "bottom" => Some(Vec3::new(0.0, -*length, 0.0)),
            _ => None,
        },
        _ => None,
    }
}

/// Cylinder approximation `(radius, length)` in metres for a part. Length
/// matches the visual mesh's body height in the editor's `visual_spec` —
/// keeping rendering and physics in sync.
fn part_cylinder_dims(entry: &CatalogEntry, params: &PartParams, effective_d: f32) -> (f64, f64) {
    let r = (effective_d * 0.5) as f64;
    let l = match (entry, params) {
        (CatalogEntry::Pod(_), _) => (effective_d * 0.9) as f64,
        (CatalogEntry::Engine(_), _) => (effective_d * 0.9) as f64,
        (CatalogEntry::Decoupler(_), _) => 0.2,
        (
            CatalogEntry::Adapter(_),
            PartParams::Adapter {
                target_diameter, ..
            },
        ) => ((effective_d + *target_diameter) * 0.5).max(0.4) as f64,
        (CatalogEntry::Tank(_), PartParams::Tank { length, .. }) => *length as f64,
        _ => 0.0,
    };
    (r, l)
}

fn part_total_mass(pb: &PartBlueprint, entry: &CatalogEntry) -> f64 {
    let dry = part_dry_mass(entry, &pb.params) as f64;
    // Compose pools to get capacities + amounts in step. Mass uses the
    // composed amounts (which default to full when blueprint omits).
    let pools = pools_for(entry, &pb.params, &pb.resources);
    let prop: f64 = pools.iter().map(|(res, pool)| pool.mass_kg(*res)).sum();
    dry + prop
}

fn part_dry_mass(entry: &CatalogEntry, params: &PartParams) -> f32 {
    match (entry, params) {
        (CatalogEntry::Pod(p), _) => p.dry_mass,
        (CatalogEntry::Engine(e), _) => e.dry_mass,
        (CatalogEntry::Decoupler(d), PartParams::Decoupler { diameter }) => {
            d.mass_per_diameter * *diameter
        }
        (
            CatalogEntry::Adapter(a),
            PartParams::Adapter {
                diameter,
                target_diameter,
            },
        ) => a.wall_mass_per_m2 * adapter_surface_area(*diameter, *target_diameter),
        (CatalogEntry::Tank(t), PartParams::Tank { diameter, length }) => {
            t.wall_mass_per_m2 * tank_surface_area(*diameter, *length)
        }
        _ => 0.0,
    }
}

fn accumulate_engine_reactants(
    reactants: &[ReactantRatio],
    engine_mdot: f64,
    per_resource_mdot: &mut HashMap<Resource, f64>,
) {
    for r in reactants {
        *per_resource_mdot.entry(r.resource).or_default() += engine_mdot * r.mass_fraction as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> PartCatalog {
        PartCatalog::load_from_str(include_str!("../../../assets/parts.ron"))
            .expect("parse parts.ron")
    }

    #[test]
    fn empty_blueprint_yields_zero_stats() {
        let cat = catalog();
        let s = ShipBlueprint {
            name: "T".into(),
            root: 0,
            parts: vec![],
            connections: vec![],
        }
        .stats(&cat)
        .expect("stats");
        assert_eq!(s.dry_mass_kg, 0.0);
        assert_eq!(s.propellant_mass_kg, 0.0);
        assert_eq!(s.total_thrust_n, 0.0);
        assert_eq!(s.combined_isp_s, 0.0);
        assert_eq!(s.mass_flow_kg_per_s, 0.0);
        assert_eq!(s.power_draw_kw, 0.0);
        assert!(s.reactant_fractions.is_empty());
        assert!(s.resources.is_empty());
        assert_eq!(s.delta_v_capacity(), 0.0);
        assert!(s.burn_time_at_full_throttle_s().is_none());
        assert_eq!(s.moment_of_inertia_kg_m2, DVec3::ZERO);
        assert_eq!(s.max_reaction_torque_n_m, 0.0);
    }

    #[test]
    fn lone_argos_inertia_matches_solid_cylinder() {
        let cat = catalog();
        // Single 2.5 m pod at the origin — its CoM coincides with the
        // ship CoM, so MOI equals the part's self-inertia (no
        // parallel-axis term).
        let bp = ShipBlueprint {
            name: "P".into(),
            root: 0,
            parts: vec![PartBlueprint {
                catalog_id: "argos".into(),
                params: PartParams::None,
                resources: HashMap::new(),
            }],
            connections: vec![],
        };
        let s = bp.stats(&cat).unwrap();
        let m = 2720.0_f64;
        let r = 1.25_f64;
        let l = 2.5_f64 * 0.9;
        let expected_yy = m * r * r * 0.5;
        let expected_xz = m * (3.0 * r * r + l * l) / 12.0;
        assert!((s.moment_of_inertia_kg_m2.y - expected_yy).abs() < 1e-6);
        assert!((s.moment_of_inertia_kg_m2.x - expected_xz).abs() < 1e-6);
        assert!((s.moment_of_inertia_kg_m2.z - expected_xz).abs() < 1e-6);
        assert_eq!(s.max_reaction_torque_n_m, 15_000.0);
    }

    #[test]
    fn argos_zephyr_stack_burns_balanced() {
        let cat = catalog();
        // 2.5 m crew stack: pod + tank + engine, all the same diameter.
        let bp = ShipBlueprint {
            name: "stack".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    catalog_id: "argos".into(),
                    params: PartParams::None,
                    resources: HashMap::new(),
                },
                PartBlueprint {
                    catalog_id: "tank_methalox".into(),
                    params: PartParams::Tank {
                        diameter: 2.5,
                        length: 4.0,
                    },
                    resources: HashMap::new(),
                },
                PartBlueprint {
                    catalog_id: "zephyr".into(),
                    params: PartParams::None,
                    resources: HashMap::new(),
                },
            ],
            connections: vec![
                Connection {
                    parent: 0,
                    parent_node: "bottom".into(),
                    child: 1,
                    child_node: "top".into(),
                },
                Connection {
                    parent: 1,
                    parent_node: "bottom".into(),
                    child: 2,
                    child_node: "top".into(),
                },
            ],
        };
        let s = bp.stats(&cat).unwrap();
        assert!((s.total_thrust_n - 500_000.0).abs() < 1e-6);
        assert!((s.combined_isp_s - 355.0).abs() < 1e-6);
        assert!(s.propellant_mass_kg > 0.0);
        // Stoichiometric methalox tank → propellant burn is
        // mass-balanced; both reactants run out within a small margin.
        let burn_s = s.burn_time_at_full_throttle_s().unwrap();
        let expelled = s.mass_flow_kg_per_s * burn_s;
        assert!(
            (expelled - s.propellant_mass_kg).abs() / s.propellant_mass_kg < 1e-3,
            "expelled {expelled} should match propellant {}",
            s.propellant_mass_kg
        );
    }

    #[test]
    fn balanced_stack_delta_v_matches_rocket_equation() {
        let cat = catalog();
        let bp = ShipBlueprint {
            name: "stack".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    catalog_id: "argos".into(),
                    params: PartParams::None,
                    resources: HashMap::new(),
                },
                PartBlueprint {
                    catalog_id: "tank_methalox".into(),
                    params: PartParams::Tank {
                        diameter: 2.5,
                        length: 4.0,
                    },
                    resources: HashMap::new(),
                },
                PartBlueprint {
                    catalog_id: "zephyr".into(),
                    params: PartParams::None,
                    resources: HashMap::new(),
                },
            ],
            connections: vec![
                Connection {
                    parent: 0,
                    parent_node: "bottom".into(),
                    child: 1,
                    child_node: "top".into(),
                },
                Connection {
                    parent: 1,
                    parent_node: "bottom".into(),
                    child: 2,
                    child_node: "top".into(),
                },
            ],
        };
        let s = bp.stats(&cat).unwrap();
        let expected = s.exhaust_velocity() * (s.wet_mass_kg() / s.dry_mass_kg).ln();
        let got = s.vacuum_delta_v().delta_v_m_per_s;

        assert!((got - expected).abs() / expected < 1e-3);
        assert!((s.delta_v_capacity() - got).abs() < 1e-10);
    }

    #[test]
    fn delta_v_is_limited_by_bottleneck_reactant() {
        let reactant_fractions = HashMap::from([(Resource::Methane, 0.25), (Resource::Lox, 0.75)]);
        let resources = HashMap::from([
            (
                Resource::Methane,
                ResourceTotals {
                    amount: 0.0,
                    capacity: 0.0,
                    mass_kg: 10.0,
                },
            ),
            (
                Resource::Lox,
                ResourceTotals {
                    amount: 0.0,
                    capacity: 0.0,
                    mass_kg: 300.0,
                },
            ),
        ]);
        let estimate = estimate_delta_v(
            DeltaVEnvironment::Vacuum,
            DeltaVInputs {
                dry_mass_kg: 1_000.0,
                wet_mass_kg: 1_310.0,
                total_thrust_n: 3_000.0,
                mass_flow_kg_per_s: 3.0,
                power_draw_kw: 0.0,
                reactant_fractions: &reactant_fractions,
                resources: &resources,
            },
        );
        let burn_s = 10.0 / (3.0 * 0.25);
        let expelled = 3.0 * burn_s;
        let expected = 1_000.0 * (1_310.0_f64 / (1_310.0 - expelled)).ln();

        assert_eq!(estimate.burn_time_s, Some(burn_s));
        assert!((estimate.expelled_mass_kg - expelled).abs() < 1e-10);
        assert!((estimate.delta_v_m_per_s - expected).abs() < 1e-10);
    }

    #[test]
    fn parametric_tank_inherits_parent_diameter_for_inertia() {
        let cat = catalog();
        // Tank declared at 2.5 m but attached under a 4 m pod — the
        // geometry walk must propagate the pod's bottom-node diameter
        // into the tank, otherwise the cylinder radius (and therefore
        // MOI_y) is computed against the wrong dimension.
        let bp_inherited = ShipBlueprint {
            name: "inherit".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    catalog_id: "hyperion".into(),
                    params: PartParams::None,
                    resources: HashMap::new(),
                },
                PartBlueprint {
                    catalog_id: "tank_methalox".into(),
                    params: PartParams::Tank {
                        diameter: 2.5,
                        length: 2.0,
                    },
                    resources: HashMap::new(),
                },
            ],
            connections: vec![Connection {
                parent: 0,
                parent_node: "bottom".into(),
                child: 1,
                child_node: "top".into(),
            }],
        };
        // Same blueprint with no connection — the tank stays at its
        // declared 2.5 m and the pod is a separate root. Roll inertia
        // is just the sum of self-inertia at the declared diameters.
        let bp_unattached = ShipBlueprint {
            connections: vec![],
            ..bp_inherited.clone()
        };
        let inherited = bp_inherited.stats(&cat).unwrap().moment_of_inertia_kg_m2.y;
        let unattached = bp_unattached.stats(&cat).unwrap().moment_of_inertia_kg_m2.y;
        // Inherited 4 m tank has 2.56× the cross-section of the 2.5 m
        // tank, so its roll inertia must be strictly larger.
        assert!(
            inherited > unattached,
            "inherited MOI_y {inherited} should exceed unattached {unattached}",
        );
    }

    #[test]
    fn unknown_catalog_id_errors() {
        let cat = catalog();
        let bp = ShipBlueprint {
            name: "x".into(),
            root: 0,
            parts: vec![PartBlueprint {
                catalog_id: "nope".into(),
                params: PartParams::None,
                resources: HashMap::new(),
            }],
            connections: vec![],
        };
        assert!(matches!(bp.stats(&cat), Err(CatalogError::UnknownId(_))));
    }
}
