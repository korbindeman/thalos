//! Aggregate physical properties of a ship blueprint.
//!
//! [`ShipStats`] is the bridge between the parametric editor representation
//! ([`ShipBlueprint`]) and any consumer that needs scalar physical quantities
//! — primarily the physics simulation. All quantities are in SI units
//! (kg, N, s, m/s², kW, kWh) and stored as `f64` to match the physics crate.
//!
//! # Engine propellant model
//!
//! Each [`crate::Engine`] declares a list of [`crate::ReactantRatio`]: mass
//! fractions of the reactants it expels, summing to 1. At full throttle an
//! engine's mass flow rate is `thrust / (isp · g₀)`; each reactant's
//! consumption rate is that mass flow times its mass fraction. Across a
//! multi-engine ship, per-resource consumption rates are summed.
//!
//! Δv capacity is Tsiolkovsky applied to the expellable mass, which is
//! limited by whichever reactant runs out first. Electricity is not a
//! reactant — it never enters the rocket equation; engines instead declare
//! a continuous `power_draw_kw` for the duration of the burn.

use crate::blueprint::{Connection, PartBlueprint, PartData, ShipBlueprint};
use crate::part::ReactantRatio;
use crate::resource::Resource;
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
        if m > 0.0 { self.total_thrust_n / m } else { 0.0 }
    }

    /// Exhaust velocity = Isp · g₀ (m/s).
    pub fn exhaust_velocity(&self) -> f64 {
        self.combined_isp_s * G0
    }

    /// Burn time at full throttle before the bottleneck reactant or
    /// electricity runs out. Returns `None` when there is no thrust or
    /// no burnable propellant.
    pub fn burn_time_at_full_throttle_s(&self) -> Option<f64> {
        if self.mass_flow_kg_per_s <= 0.0 {
            return None;
        }

        let mut limit = f64::INFINITY;

        for (res, frac) in &self.reactant_fractions {
            if *frac <= 0.0 {
                continue;
            }
            let rate_kg_per_s = self.mass_flow_kg_per_s * frac;
            let available = self.resources.get(res).map(|r| r.mass_kg).unwrap_or(0.0);
            limit = limit.min(available / rate_kg_per_s);
        }

        if self.power_draw_kw > 0.0 {
            let stored_kwh = self
                .resources
                .get(&Resource::Electricity)
                .map(|r| r.amount)
                .unwrap_or(0.0);
            // kWh / kW = hours → convert to seconds.
            let power_limit_s = stored_kwh / self.power_draw_kw * 3600.0;
            limit = limit.min(power_limit_s);
        }

        if limit.is_finite() { Some(limit) } else { None }
    }

    /// Tsiolkovsky Δv available from current state, limited by the
    /// bottleneck reactant (and/or stored electricity if power-dependent).
    /// Returns 0 when any critical input is missing.
    pub fn delta_v_capacity(&self) -> f64 {
        let wet = self.wet_mass_kg();
        let ve = self.exhaust_velocity();
        if ve <= 0.0 || wet <= 0.0 || self.dry_mass_kg <= 0.0 {
            return 0.0;
        }
        let Some(burn_s) = self.burn_time_at_full_throttle_s() else {
            return 0.0;
        };
        let raw_expelled = self.mass_flow_kg_per_s * burn_s;
        if raw_expelled <= 0.0 {
            return 0.0;
        }
        // Clamp against total propellant so float error on a perfectly
        // balanced burn can't dip below dry mass.
        let expelled = raw_expelled.min(self.propellant_mass_kg);
        let remaining = wet - expelled;
        if remaining <= 0.0 {
            return 0.0;
        }
        ve * (wet / remaining).ln()
    }
}

impl ShipBlueprint {
    pub fn stats(&self) -> ShipStats {
        let mut dry_mass_kg = 0.0_f64;
        let mut total_thrust_n = 0.0_f64;
        // Σ (thrust / isp) — denominator of mass-flow-weighted Isp.
        let mut thrust_over_isp = 0.0_f64;
        let mut power_draw_kw = 0.0_f64;
        let mut per_resource_mdot: HashMap<Resource, f64> = HashMap::new();

        for pb in &self.parts {
            dry_mass_kg += part_dry_mass(&pb.data) as f64;
            if let PartData::Engine {
                thrust,
                isp,
                reactants,
                power_draw_kw: engine_power,
                ..
            } = &pb.data
            {
                let t = *thrust as f64;
                total_thrust_n += t;
                power_draw_kw += *engine_power as f64;
                if *isp > 0.0 {
                    let isp_f = *isp as f64;
                    thrust_over_isp += t / isp_f;
                    let mdot = t / (isp_f * G0);
                    accumulate_engine_reactants(reactants, mdot, &mut per_resource_mdot);
                }
            }
        }

        // Aggregate pools by resource.
        let mut resources: HashMap<Resource, ResourceTotals> = HashMap::new();
        for pb in &self.parts {
            for (res, pool) in &pb.resources {
                let e = resources.entry(*res).or_default();
                e.amount += pool.amount as f64;
                e.capacity += pool.capacity as f64;
                e.mass_kg += pool.mass_kg(*res);
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

        let geo = ship_geometry(self);

        let mut com_total_mass = 0.0_f64;
        let mut com_weighted = DVec3::ZERO;
        for (i, pb) in self.parts.iter().enumerate() {
            let m = part_total_mass(pb);
            com_total_mass += m;
            com_weighted += geo[i].position * m;
        }
        let com = if com_total_mass > 0.0 {
            com_weighted / com_total_mass
        } else {
            DVec3::ZERO
        };

        let mut moment_of_inertia_kg_m2 = DVec3::ZERO;
        for (i, pb) in self.parts.iter().enumerate() {
            let m = part_total_mass(pb);
            let (r, l) = part_cylinder_dims(&pb.data, geo[i].diameter);
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

            moment_of_inertia_kg_m2 += DVec3::new(
                i_xz_self + par.x,
                i_yy_self + par.y,
                i_xz_self + par.z,
            );
        }

        let mut max_reaction_torque_n_m = 0.0_f64;
        for pb in &self.parts {
            if let PartData::CommandPod {
                reaction_wheel_torque,
                ..
            } = pb.data
            {
                max_reaction_torque_n_m += reaction_wheel_torque as f64;
            }
        }

        ShipStats {
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
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry — per-part CoM positions in ship body frame
//
// `ship_geometry` mirrors the runtime's BFS in `sizing::propagate_node_sizes`
// + `update_ship_part_transforms`, but operates on blueprint indices instead
// of ECS entities. This keeps the inertia model honest for blueprints that
// rely on parametric diameter inheritance (e.g. a tank declared with the
// default 1.25 m diameter but attached under a 2.5 m pod).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct PartGeometry {
    /// Position in ship body frame, metres. The root sits at the origin.
    position: DVec3,
    /// Effective outer diameter after parametric inheritance from the
    /// parent's mating node, metres.
    diameter: f32,
}

fn ship_geometry(blueprint: &ShipBlueprint) -> Vec<PartGeometry> {
    let mut geo: Vec<PartGeometry> = blueprint
        .parts
        .iter()
        .map(|pb| PartGeometry {
            position: DVec3::ZERO,
            diameter: declared_diameter(&pb.data),
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

            if is_parametric(&child_pb.data)
                && let Some(input_d) = node_diameter(&parent_pb.data, parent_d, &c.parent_node)
            {
                geo[c.child].diameter = effective_diameter_for(&child_pb.data, input_d);
            }

            let parent_offset = node_offset(&parent_pb.data, parent_d, &c.parent_node)
                .unwrap_or(Vec3::ZERO);
            let child_offset =
                node_offset(&child_pb.data, geo[c.child].diameter, &c.child_node)
                    .unwrap_or(Vec3::ZERO);
            geo[c.child].position =
                parent_pos + (parent_offset - child_offset).as_dvec3();

            queue.push_back(c.child);
        }
    }

    geo
}

fn declared_diameter(data: &PartData) -> f32 {
    match data {
        PartData::CommandPod { diameter, .. }
        | PartData::Engine { diameter, .. }
        | PartData::Adapter { diameter, .. }
        | PartData::Decoupler { diameter, .. }
        | PartData::FuelTank { diameter, .. } => *diameter,
    }
}

fn is_parametric(data: &PartData) -> bool {
    matches!(
        data,
        PartData::FuelTank { .. } | PartData::Decoupler { .. } | PartData::Adapter { .. }
    )
}

/// Effective single-cylinder diameter after a parametric child inherits
/// its top diameter from the parent. Adapters are tapered, so we average
/// top (= parent's mating diameter) and bottom (= `target_diameter`) —
/// the cylinder model can't represent both anyway.
fn effective_diameter_for(data: &PartData, parent_node_d: f32) -> f32 {
    match data {
        PartData::Adapter {
            target_diameter, ..
        } => (parent_node_d + target_diameter) * 0.5,
        _ => parent_node_d,
    }
}

/// Diameter of the named attach node on this part, given its (possibly
/// propagated) effective body diameter. Mirrors [`crate::blueprint::default_nodes_for`]
/// but with the propagated diameter substituted for the declared one.
fn node_diameter(data: &PartData, effective_d: f32, node: &str) -> Option<f32> {
    match data {
        PartData::CommandPod { .. } => (node == "bottom").then_some(effective_d),
        PartData::Engine { .. } | PartData::Decoupler { .. } | PartData::FuelTank { .. } => {
            (node == "top" || node == "bottom").then_some(effective_d)
        }
        PartData::Adapter {
            target_diameter, ..
        } => match node {
            "top" => Some(effective_d),
            "bottom" => Some(*target_diameter),
            _ => None,
        },
    }
}

/// Offset of the named attach node, in the part's local frame (Y points
/// out the top). Mirrors [`crate::blueprint::default_nodes_for`].
fn node_offset(data: &PartData, effective_d: f32, node: &str) -> Option<Vec3> {
    match data {
        PartData::CommandPod { .. } => {
            (node == "bottom").then_some(Vec3::new(0.0, -effective_d * 0.9, 0.0))
        }
        PartData::Engine { .. } => match node {
            "top" => Some(Vec3::ZERO),
            "bottom" => Some(Vec3::new(0.0, -effective_d * 0.9, 0.0)),
            _ => None,
        },
        PartData::Decoupler { .. } => match node {
            "top" => Some(Vec3::ZERO),
            "bottom" => Some(Vec3::new(0.0, -0.2, 0.0)),
            _ => None,
        },
        PartData::Adapter {
            target_diameter, ..
        } => match node {
            "top" => Some(Vec3::ZERO),
            "bottom" => {
                let h = ((effective_d + target_diameter) * 0.5).max(0.4);
                Some(Vec3::new(0.0, -h, 0.0))
            }
            _ => None,
        },
        PartData::FuelTank { length, .. } => match node {
            "top" => Some(Vec3::ZERO),
            "bottom" => Some(Vec3::new(0.0, -length, 0.0)),
            _ => None,
        },
    }
}

/// Cylinder approximation `(radius, length)` in metres for a part.
/// Length matches the visual mesh's body height in [`crate::part`] / the
/// editor's `visual_spec` — keeping rendering and physics in sync.
fn part_cylinder_dims(data: &PartData, effective_d: f32) -> (f64, f64) {
    let r = (effective_d * 0.5) as f64;
    let l = match data {
        PartData::CommandPod { .. } => (effective_d * 0.9) as f64,
        PartData::Engine { .. } => (effective_d * 0.9) as f64,
        PartData::Decoupler { .. } => 0.2,
        PartData::Adapter {
            target_diameter, ..
        } => ((effective_d + target_diameter) * 0.5).max(0.4) as f64,
        PartData::FuelTank { length, .. } => *length as f64,
    };
    (r, l)
}

fn part_total_mass(pb: &PartBlueprint) -> f64 {
    let dry = part_dry_mass(&pb.data) as f64;
    let prop: f64 = pb
        .resources
        .iter()
        .map(|(res, pool)| pool.mass_kg(*res))
        .sum();
    dry + prop
}

fn accumulate_engine_reactants(
    reactants: &[ReactantRatio],
    engine_mdot: f64,
    per_resource_mdot: &mut HashMap<Resource, f64>,
) {
    for r in reactants {
        *per_resource_mdot.entry(r.resource).or_default() +=
            engine_mdot * r.mass_fraction as f64;
    }
}

fn part_dry_mass(data: &PartData) -> f32 {
    match data {
        PartData::CommandPod { dry_mass, .. } => *dry_mass,
        PartData::Decoupler { dry_mass, .. } => *dry_mass,
        PartData::Adapter { dry_mass, .. } => *dry_mass,
        PartData::FuelTank { dry_mass, .. } => *dry_mass,
        PartData::Engine { dry_mass, .. } => *dry_mass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::PartBlueprint;
    use crate::resource::ResourcePool;

    fn methalox_engine(thrust: f32, isp: f32, dry: f32) -> PartData {
        PartData::Engine {
            model: "Raptor".into(),
            diameter: 1.3,
            thrust,
            isp,
            dry_mass: dry,
            // O/F ≈ 3.6 by mass → fuel fraction 1/4.6, ox fraction 3.6/4.6.
            reactants: vec![
                ReactantRatio {
                    resource: Resource::Methane,
                    mass_fraction: 1.0 / 4.6,
                },
                ReactantRatio {
                    resource: Resource::Lox,
                    mass_fraction: 3.6 / 4.6,
                },
            ],
            power_draw_kw: 0.0,
        }
    }

    fn tank_with(resource: Resource, amount: f32) -> PartBlueprint {
        let mut pools = HashMap::new();
        pools.insert(
            resource,
            ResourcePool {
                capacity: amount,
                amount,
            },
        );
        PartBlueprint {
            data: PartData::FuelTank {
                diameter: 2.0,
                length: 3.0,
                dry_mass: 500.0,
            },
            resources: pools,
        }
    }

    fn simple_methalox_ship() -> ShipBlueprint {
        ShipBlueprint {
            name: "T".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    data: PartData::CommandPod {
                        model: "P".into(),
                        diameter: 2.0,
                        dry_mass: 2000.0,
                        reaction_wheel_torque: 0.0,
                    },
                    resources: HashMap::new(),
                },
                // Balanced so neither runs out first: 4600 kg total at 3.6:1 ⇒
                // 1000 kg CH4 (≈2370 L), 3600 kg LOX (≈3155 L). Use larger
                // numbers for precision.
                tank_with(Resource::Methane, 23700.0),
                tank_with(Resource::Lox, 31550.0),
                PartBlueprint {
                    data: methalox_engine(200_000.0, 350.0, 500.0),
                    resources: HashMap::new(),
                },
            ],
            connections: vec![],
        }
    }

    #[test]
    fn methalox_stats_are_balanced() {
        let s = simple_methalox_ship().stats();
        let ch4_mass = 23700.0 * Resource::Methane.density_kg_per_unit();
        let lox_mass = 31550.0 * Resource::Lox.density_kg_per_unit();
        let total_prop = ch4_mass + lox_mass;

        assert!((s.propellant_mass_kg - total_prop).abs() < 1.0);
        assert!((s.total_thrust_n - 200_000.0).abs() < 1e-6);
        assert!((s.combined_isp_s - 350.0).abs() < 1e-6);

        // Reactant fractions match engine declaration. Fractions are
        // stored as f32 on ReactantRatio, so widen the tolerance.
        let ch4_frac = s.reactant_fractions[&Resource::Methane];
        let lox_frac = s.reactant_fractions[&Resource::Lox];
        assert!((ch4_frac - 1.0 / 4.6).abs() < 1e-6);
        assert!((lox_frac - 3.6 / 4.6).abs() < 1e-6);

        // Balanced ratios should allow full propellant burn.
        let burn_s = s.burn_time_at_full_throttle_s().unwrap();
        let expelled = s.mass_flow_kg_per_s * burn_s;
        assert!((expelled - total_prop).abs() / total_prop < 1e-3);
    }

    #[test]
    fn missing_oxidizer_zeros_delta_v() {
        // CH4 tank, no LOX — engine can't burn.
        let mut bp = simple_methalox_ship();
        // Drop the LOX tank (index 2).
        bp.parts.remove(2);
        let s = bp.stats();
        assert!(s.propellant_mass_kg > 0.0); // CH4 still has mass
        // But burn time is zero because LOX rate > 0 with 0 available.
        let burn = s.burn_time_at_full_throttle_s().unwrap();
        assert!(burn.abs() < 1e-9, "expected 0 burn time, got {burn}");
        assert_eq!(s.delta_v_capacity(), 0.0);
    }

    #[test]
    fn electricity_limits_ion_style_burn() {
        // Ion-ish: small thrust, high Isp, power-hungry, xenon-style
        // reactant modeled as methane for simplicity.
        let engine = PartData::Engine {
            model: "ion".into(),
            diameter: 0.5,
            thrust: 1.0,
            isp: 3000.0,
            dry_mass: 50.0,
            reactants: vec![ReactantRatio {
                resource: Resource::Methane,
                mass_fraction: 1.0,
            }],
            power_draw_kw: 2.0,
        };
        let mut battery_pools = HashMap::new();
        battery_pools.insert(
            Resource::Electricity,
            ResourcePool {
                capacity: 1.0, // 1 kWh
                amount: 1.0,
            },
        );
        let bp = ShipBlueprint {
            name: "ion".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    data: PartData::CommandPod {
                        model: "P".into(),
                        diameter: 1.0,
                        dry_mass: 100.0,
                        reaction_wheel_torque: 0.0,
                    },
                    resources: battery_pools,
                },
                tank_with(Resource::Methane, 1_000_000.0), // plenty of reactant
                PartBlueprint {
                    data: engine,
                    resources: HashMap::new(),
                },
            ],
            connections: vec![],
        };
        let s = bp.stats();
        // 1 kWh at 2 kW = 0.5 h = 1800 s, well below propellant-limited burn.
        let burn = s.burn_time_at_full_throttle_s().unwrap();
        assert!((burn - 1800.0).abs() < 1.0);
    }

    #[test]
    fn combined_isp_is_mass_flow_weighted() {
        let bp = ShipBlueprint {
            name: "T".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    data: methalox_engine(100_000.0, 200.0, 0.0),
                    resources: HashMap::new(),
                },
                PartBlueprint {
                    data: methalox_engine(100_000.0, 400.0, 0.0),
                    resources: HashMap::new(),
                },
            ],
            connections: vec![],
        };
        let s = bp.stats();
        let expected = 200_000.0 / (100_000.0 / 200.0 + 100_000.0 / 400.0);
        assert!((s.combined_isp_s - expected).abs() < 1e-6);
    }

    #[test]
    fn engine_validate_catches_bad_fractions() {
        use crate::part::{Engine, EngineValidationError};
        let bad_sum = Engine {
            model: "e".into(),
            diameter: 1.0,
            thrust: 1.0,
            isp: 1.0,
            dry_mass: 0.0,
            reactants: vec![
                ReactantRatio {
                    resource: Resource::Methane,
                    mass_fraction: 0.5,
                },
                ReactantRatio {
                    resource: Resource::Lox,
                    mass_fraction: 0.4,
                },
            ],
            power_draw_kw: 0.0,
        };
        assert_eq!(
            bad_sum.validate(),
            Err(EngineValidationError::ReactantFractionsNotNormalized)
        );

        let empty = Engine {
            model: "e".into(),
            diameter: 1.0,
            thrust: 1.0,
            isp: 1.0,
            dry_mass: 0.0,
            reactants: vec![],
            power_draw_kw: 0.0,
        };
        assert_eq!(empty.validate(), Err(EngineValidationError::NoReactants));

        let electric_reactant = Engine {
            model: "e".into(),
            diameter: 1.0,
            thrust: 1.0,
            isp: 1.0,
            dry_mass: 0.0,
            reactants: vec![ReactantRatio {
                resource: Resource::Electricity,
                mass_fraction: 1.0,
            }],
            power_draw_kw: 0.0,
        };
        assert_eq!(
            electric_reactant.validate(),
            Err(EngineValidationError::ReactantNotMassBearing)
        );

        // The methalox helper is known-good.
        let good = methalox_engine(1.0, 1.0, 0.0);
        let PartData::Engine {
            model,
            diameter,
            thrust,
            isp,
            dry_mass,
            reactants,
            power_draw_kw,
        } = good
        else {
            unreachable!();
        };
        let good_engine = Engine {
            model,
            diameter,
            thrust,
            isp,
            dry_mass,
            reactants,
            power_draw_kw,
        };
        assert!(good_engine.validate().is_ok());
    }

    #[test]
    fn empty_blueprint_yields_zero_stats() {
        let s = ShipBlueprint {
            name: "T".into(),
            root: 0,
            parts: vec![],
            connections: vec![],
        }
        .stats();
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
    fn lone_command_pod_inertia_matches_solid_cylinder() {
        // Single pod at the origin — its CoM coincides with the ship CoM,
        // so MOI equals the part's self-inertia (no parallel-axis term).
        let bp = ShipBlueprint {
            name: "P".into(),
            root: 0,
            parts: vec![PartBlueprint {
                data: PartData::CommandPod {
                    model: "Mk1-3".into(),
                    diameter: 2.5,
                    dry_mass: 2720.0,
                    reaction_wheel_torque: 15_000.0,
                },
                resources: HashMap::new(),
            }],
            connections: vec![],
        };
        let s = bp.stats();
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
    fn stack_inertia_has_smaller_roll_than_pitch() {
        // Apollo-shaped column along body Y. Parallel-axis terms only
        // contribute to the transverse axes (X, Z) — the roll axis (Y)
        // stays at the cylinder self-inertia and remains the smallest.
        let bp = ShipBlueprint {
            name: "stack".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    data: PartData::CommandPod {
                        model: "Mk1-3".into(),
                        diameter: 2.5,
                        dry_mass: 2720.0,
                        reaction_wheel_torque: 15_000.0,
                    },
                    resources: HashMap::new(),
                },
                PartBlueprint {
                    data: PartData::FuelTank {
                        diameter: 2.5,
                        length: 3.0,
                        dry_mass: 250.0,
                    },
                    resources: HashMap::new(),
                },
                PartBlueprint {
                    data: PartData::Engine {
                        model: "Poodle".into(),
                        diameter: 2.5,
                        thrust: 0.0,
                        isp: 1.0,
                        dry_mass: 250.0,
                        reactants: vec![ReactantRatio {
                            resource: Resource::Methane,
                            mass_fraction: 1.0,
                        }],
                        power_draw_kw: 0.0,
                    },
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
        let s = bp.stats();
        // Roll (Y) is the long-axis self-inertia only; pitch (X) and yaw
        // (Z) pick up the parallel-axis term and dwarf it for a stack.
        assert!(s.moment_of_inertia_kg_m2.y > 0.0);
        assert!(s.moment_of_inertia_kg_m2.x > s.moment_of_inertia_kg_m2.y);
        assert!(s.moment_of_inertia_kg_m2.z > s.moment_of_inertia_kg_m2.y);
        // Stack is rotationally symmetric in body X/Z so those are equal.
        assert!(
            (s.moment_of_inertia_kg_m2.x - s.moment_of_inertia_kg_m2.z).abs() < 1e-6
        );
    }

    #[test]
    fn parametric_tank_inherits_parent_diameter_for_inertia() {
        // Tank declared at the default 1.25 m but attached under a 2.5 m
        // pod — the geometry walk must propagate the pod's bottom-node
        // diameter into the tank, otherwise the cylinder radius (and
        // therefore MOI_y) is computed against the wrong dimension.
        let bp = ShipBlueprint {
            name: "inherit".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    data: PartData::CommandPod {
                        model: "Mk1-3".into(),
                        diameter: 2.5,
                        dry_mass: 1000.0,
                        reaction_wheel_torque: 0.0,
                    },
                    resources: HashMap::new(),
                },
                PartBlueprint {
                    // No diameter ⇒ default 1.25 m on the blueprint, but
                    // the pod's bottom node should propagate 2.5 m down.
                    data: PartData::FuelTank {
                        diameter: 1.25,
                        length: 2.0,
                        dry_mass: 500.0,
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
        let s = bp.stats();
        // If propagation didn't run, the tank's r=0.625 would dominate
        // and the roll-axis MOI would be ~4× smaller than this.
        let m_tank = 500.0;
        let r_tank_propagated = 1.25; // = 2.5 / 2
        let tank_yy_self = m_tank * r_tank_propagated * r_tank_propagated * 0.5;
        // Pod's roll inertia at r=1.25:
        let pod_yy_self = 1000.0 * 1.25 * 1.25 * 0.5;
        let expected_y = tank_yy_self + pod_yy_self;
        assert!((s.moment_of_inertia_kg_m2.y - expected_y).abs() < 1e-6);
    }
}
