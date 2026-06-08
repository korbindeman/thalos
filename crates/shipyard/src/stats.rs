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

use crate::blueprint::{
    Connection, PartBlueprint, PartParams, ShipBlueprint, check_params_match,
    check_resource_amounts_allowed, pools_for,
};
use crate::catalog::{
    CatalogEntry, CatalogError, PartCatalog, WingRole, adapter_surface_area, fuselage_surface_area,
    gear_dry_mass, tank_surface_area, wing_mean_aerodynamic_chord, wing_panel_area,
};
use crate::part::{ReactantRatio, Wing};
use crate::resource::{PartResources, Resource};
use crate::wing_mesh::wing_panel_frame;
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
    /// Center of mass in the ship body frame (`X=right, Y=nose, Z=dorsal`),
    /// metres from the root-part origin. The same point
    /// [`moment_of_inertia_kg_m2`] is taken about; ground physics rotates the
    /// craft about it so gear that straddle the CoM hold the aircraft level.
    pub center_of_mass_m: DVec3,
    /// Sum of every [`crate::ReactionWheel`]'s `max_torque`, in N·m
    /// per body axis. Symmetric — the per-axis cap is the same on all
    /// three. Per-axis-asymmetric torque is reserved for RCS arrangements.
    pub max_reaction_torque_n_m: f64,
    /// Nose-on frontal (reference) area, m², from the widest propagated part
    /// diameter: π·(d_max/2)². Used as the aerodynamic reference area for the
    /// aggregate bluff-body drag of rockets/capsules. A crude but per-vehicle
    /// estimate (a slim rocket and a blunt capsule now differ); a richer model
    /// can integrate the true cross-section later.
    pub frontal_area_m2: f64,
    /// Total planform (top-down) wing area across every lifting surface,
    /// m² — mirrored pairs counted on both sides. Geometry-derived editor
    /// feedback ("will this fly"); there is no flight model yet.
    pub wing_area_m2: f64,
    /// Area-weighted mean aerodynamic chord across all wings, m. Zero when
    /// the vessel has no wings.
    pub mean_aerodynamic_chord_m: f64,
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
            check_params_match(&pb.catalog_id, entry, &pb.params)?;
            check_resource_amounts_allowed(&pb.catalog_id, entry, &pb.resources)?;
        }

        // KSP symmetry: a mirrored pair is two real parts, each counted once
        // here — no per-part doubling. The mirror counterpart is a separate
        // entry in `self.parts`.
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

        // Geometry-only "will it fly" feedback: total wing area (both
        // sides of a pair) and the area-weighted mean aerodynamic chord.
        let mut wing_area_m2 = 0.0_f64;
        let mut mac_area_weighted = 0.0_f64;
        for (pb, entry) in self.parts.iter().zip(&entries) {
            if let (
                CatalogEntry::Wing(_),
                PartParams::Wing {
                    span,
                    root_chord,
                    tip_chord,
                    ..
                },
            ) = (entry, &pb.params)
            {
                let area = wing_panel_area(*span, *root_chord, *tip_chord) as f64;
                wing_area_m2 += area;
                mac_area_weighted +=
                    wing_mean_aerodynamic_chord(*root_chord, *tip_chord) as f64 * area;
            }
        }
        let mean_aerodynamic_chord_m = if wing_area_m2 > 0.0 {
            mac_area_weighted / wing_area_m2
        } else {
            0.0
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
            let self_inertia = if let (
                CatalogEntry::Wing(_),
                PartParams::Wing {
                    span, root_chord, ..
                },
            ) = (entry, &pb.params)
            {
                wing_self_inertia(m, *span as f64, *root_chord as f64)
            } else {
                let (r, l) = part_cylinder_dims(entry, &pb.params, geo[i].diameter);
                cylinder_principal_inertia(m, r, l)
            };
            moment_of_inertia_kg_m2 +=
                self_inertia + parallel_axis_inertia(m, geo[i].position - com);
        }

        // Nose-on frontal area from the widest propagated part diameter.
        let max_diameter_m = geo
            .iter()
            .map(|g| g.diameter as f64)
            .fold(0.0_f64, f64::max);
        let frontal_area_m2 = std::f64::consts::PI * (max_diameter_m * 0.5).powi(2);

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
            center_of_mass_m: com,
            max_reaction_torque_n_m,
            frontal_area_m2,
            wing_area_m2,
            mean_aerodynamic_chord_m,
        })
    }

    /// Per-panel aerodynamic geometry for every [`Wing`] part, in the **ship
    /// body frame** (`+Y` = nose, `+X` = right, `+Z` = dorsal/up). One entry
    /// per wing entity (a mirrored pair is two entries). This is the geometry a
    /// flight model turns into lift/drag/control zones — the game maps each
    /// panel to an `avian_fdm` `AeroZone` (airfoil coefficients, control role,
    /// and the body→SAE frame conversion live game-side).
    ///
    /// `center_body_m` is the panel's aerodynamic centre (≈ quarter-chord at
    /// ~40% span); `fore_dir`/`thick_dir`/`span_dir` are the airfoil basis
    /// (chord-forward / lift-normal / spanwise) in the body frame.
    pub fn wing_aero_panels(&self, catalog: &PartCatalog) -> Result<Vec<WingAeroPanel>, CatalogError> {
        let entries: Vec<&CatalogEntry> = self
            .parts
            .iter()
            .map(|pb| catalog.resolve(&pb.catalog_id))
            .collect::<Result<_, _>>()?;
        let geo = ship_geometry(self, &entries);

        let mut panels = Vec::new();
        for m in &self.surface_mounts {
            if m.child >= self.parts.len() || m.parent >= geo.len() {
                continue;
            }
            let pb = &self.parts[m.child];
            let (
                CatalogEntry::Wing(wing_spec),
                PartParams::Wing {
                    span,
                    root_chord,
                    tip_chord,
                    sweep,
                    dihedral,
                    thickness,
                    incidence,
                },
            ) = (entries[m.child], &pb.params)
            else {
                continue;
            };
            let wing = Wing {
                span: *span,
                root_chord: *root_chord,
                tip_chord: *tip_chord,
                sweep: *sweep,
                dihedral: *dihedral,
                thickness: *thickness,
                incidence: *incidence,
                dry_mass: 0.0,
            };
            let parent_radius = geo[m.parent].diameter * 0.5;
            let frame = wing_panel_frame(&wing, m.angle, parent_radius);
            // Aerodynamic centre: ~40% span, shifted forward from mid-chord to
            // roughly quarter-chord.
            let ac_span_frac = 0.4_f32;
            let chord_here = frame.chord_at(&wing, ac_span_frac);
            let ac_local = frame.center_at(ac_span_frac) + frame.fore_dir * (chord_here * 0.25);
            let center_body_m = geo[m.child].position + ac_local.as_dvec3();

            panels.push(WingAeroPanel {
                center_body_m,
                fore_dir: frame.fore_dir.as_dvec3(),
                thick_dir: frame.thick_dir.as_dvec3(),
                span_dir: frame.span_dir.as_dvec3(),
                area_m2: wing_panel_area(*span, *root_chord, *tip_chord) as f64,
                chord_m: wing_mean_aerodynamic_chord(*root_chord, *tip_chord) as f64,
                span_m: *span as f64,
                station: m.station as f64,
                angle: m.angle as f64,
                role: wing_spec.role,
            });
        }
        Ok(panels)
    }
}

/// One wing panel's aerodynamic geometry in the ship body frame. See
/// [`ShipBlueprint::wing_aero_panels`].
#[derive(Clone, Copy, Debug)]
pub struct WingAeroPanel {
    /// Aerodynamic-centre position in the ship body frame (m).
    pub center_body_m: DVec3,
    /// Chord-forward unit vector (toward the nose), body frame.
    pub fore_dir: DVec3,
    /// Airfoil lift-normal unit vector ("up"/dorsal side), body frame.
    pub thick_dir: DVec3,
    /// Spanwise (outboard) unit vector, body frame.
    pub span_dir: DVec3,
    /// Single-panel planform area (m²).
    pub area_m2: f64,
    /// Mean aerodynamic chord (m).
    pub chord_m: f64,
    /// Panel half-span (m).
    pub span_m: f64,
    /// Mount station along the host (0 = nose end, 1 = tail end).
    pub station: f64,
    /// Mount azimuth (rad): 0 = dorsal, π/2 = right, π = belly, −π/2 = left.
    pub angle: f64,
    /// Authored aerodynamic role from the catalog: `Lift` (main wing) or
    /// `Stabilizer` (tailplane / fin). The consumer pairs this with the mount
    /// azimuth (horizontal vs vertical) to assign control surfaces.
    pub role: WingRole,
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

    // Surface-mounted parts (wings) sit on the host body axis at their
    // station. Approximate their CoM there — the radial/spanwise offset of
    // a single panel is small next to the lever from the ship CoM, and a
    // mirrored pair's panels cancel in X. Hosts are node-stacked, so their
    // positions are already resolved above.
    for m in &blueprint.surface_mounts {
        if m.parent < geo.len() && m.child < geo.len() {
            let parent_entry = entries[m.parent];
            let parent_pb = &blueprint.parts[m.parent];
            let (_, parent_h) =
                part_cylinder_dims(parent_entry, &parent_pb.params, geo[m.parent].diameter);
            geo[m.child].position =
                geo[m.parent].position + DVec3::new(0.0, -(m.station as f64) * parent_h, 0.0);
        }
    }

    geo
}

/// Crude thin-wing self-inertia about its own CoM: model the panel as a flat
/// rod along body X (chord along Y, negligible thickness). One panel per
/// entity now (a mirrored pair is two entities, summed by the caller).
/// Placeholder in the same spirit as the per-part cylinder model until an
/// airfoil mass distribution exists; the parallel-axis lever dominates for an
/// off-centre wing regardless.
fn wing_self_inertia(mass_kg: f64, span_m: f64, chord_m: f64) -> DVec3 {
    let i_span = mass_kg * span_m * span_m / 12.0; // bending about Y and Z
    let i_chord = mass_kg * chord_m * chord_m / 12.0; // about the span axis X
    DVec3::new(i_chord, i_span, i_span)
}

fn declared_diameter(entry: &CatalogEntry, params: &PartParams) -> f32 {
    match (entry, params) {
        (CatalogEntry::Pod(p), _) => p.diameter,
        (CatalogEntry::Engine(e), _) => e.diameter,
        (CatalogEntry::Intake(i), _) => i.diameter,
        (CatalogEntry::Decoupler(_), PartParams::Decoupler { diameter }) => *diameter,
        (CatalogEntry::Adapter(_), PartParams::Adapter { diameter, .. }) => *diameter,
        (CatalogEntry::Tank(_), PartParams::Tank { diameter, .. }) => *diameter,
        (CatalogEntry::Fuselage(_), PartParams::Fuselage { max_width, .. }) => *max_width,
        _ => 0.0,
    }
}

fn is_parametric(entry: &CatalogEntry) -> bool {
    matches!(
        entry,
        CatalogEntry::Tank(_)
            | CatalogEntry::Adapter(_)
            | CatalogEntry::Decoupler(_)
            | CatalogEntry::Fuselage(_)
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
        (CatalogEntry::Engine(_), _) | (CatalogEntry::Intake(_), _) => {
            (node == "top" || node == "bottom").then_some(effective_d)
        }
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
        (
            CatalogEntry::Fuselage(_),
            PartParams::Fuselage {
                tail_tip_diameter, ..
            },
        ) => match node {
            "top" => Some(effective_d),
            "bottom" => Some(*tail_tip_diameter),
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
        (CatalogEntry::Pod(p), _) => {
            (node == "bottom").then_some(Vec3::new(0.0, -effective_d * p.geometry.length_factor(), 0.0))
        }
        (CatalogEntry::Engine(_), _) => match node {
            "top" => Some(Vec3::ZERO),
            "bottom" => Some(Vec3::new(0.0, -effective_d * 0.9, 0.0)),
            _ => None,
        },
        (CatalogEntry::Intake(i), _) => match node {
            "top" => Some(Vec3::ZERO),
            "bottom" => Some(Vec3::new(0.0, -i.length, 0.0)),
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
        (CatalogEntry::Fuselage(_), PartParams::Fuselage { length, .. }) => match node {
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
        (CatalogEntry::Pod(p), _) => (effective_d * p.geometry.length_factor()) as f64,
        (CatalogEntry::Engine(_), _) => (effective_d * 0.9) as f64,
        (CatalogEntry::Intake(i), _) => i.length as f64,
        (CatalogEntry::Decoupler(_), _) => 0.2,
        (
            CatalogEntry::Adapter(_),
            PartParams::Adapter {
                target_diameter, ..
            },
        ) => ((effective_d + *target_diameter) * 0.5).max(0.4) as f64,
        (CatalogEntry::Tank(_), PartParams::Tank { length, .. }) => *length as f64,
        (CatalogEntry::Fuselage(_), PartParams::Fuselage { length, .. }) => *length as f64,
        _ => 0.0,
    };
    (r, l)
}

/// Principal-axis moment of inertia of a solid cylinder about its own
/// centre of mass, with the long axis along body Y. Returned as the
/// diagonal `(I_xx, I_yy, I_zz)` in kg·m²:
///
///   I_yy = m·r²/2            (about the long axis)
///   I_xx = I_zz = m·(3r² + L²)/12
///
/// This is the single home for the per-part inertia model. Both the
/// blueprint aggregation in [`ShipBlueprint::stats`] and the game's
/// live, per-frame recompute after a stage drops feed it the same way,
/// so spawn-time and post-staging MOI never disagree on the model.
pub fn cylinder_principal_inertia(mass_kg: f64, radius_m: f64, length_m: f64) -> DVec3 {
    let i_long = mass_kg * radius_m * radius_m * 0.5;
    let i_trans = mass_kg * (3.0 * radius_m * radius_m + length_m * length_m) / 12.0;
    DVec3::new(i_trans, i_long, i_trans)
}

/// Parallel-axis term for a part of mass `mass_kg` whose centre sits at
/// body-frame `offset_m` from the ship CoM, in kg·m² on each principal
/// axis. Add to [`cylinder_principal_inertia`] to shift a part's self
/// inertia onto the ship axes.
pub fn parallel_axis_inertia(mass_kg: f64, offset_m: DVec3) -> DVec3 {
    DVec3::new(
        mass_kg * (offset_m.y * offset_m.y + offset_m.z * offset_m.z),
        mass_kg * (offset_m.x * offset_m.x + offset_m.z * offset_m.z),
        mass_kg * (offset_m.x * offset_m.x + offset_m.y * offset_m.y),
    )
}

fn part_total_mass(pb: &PartBlueprint, entry: &CatalogEntry) -> f64 {
    let dry = part_dry_mass(entry, &pb.params) as f64;
    // Compose pools to get capacities + amounts in step. Mass uses the
    // composed amounts (which default to full when blueprint omits).
    let pools = pools_for(entry, &pb.params, &pb.resources);
    let prop: f64 = pools.iter().map(|(res, pool)| pool.mass_kg(*res)).sum();
    dry + prop
}

pub(crate) fn part_dry_mass(entry: &CatalogEntry, params: &PartParams) -> f32 {
    match (entry, params) {
        (CatalogEntry::Pod(p), _) => p.dry_mass,
        (CatalogEntry::Engine(e), _) => e.dry_mass,
        (CatalogEntry::Intake(i), _) => i.dry_mass,
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
        (
            CatalogEntry::Fuselage(f),
            PartParams::Fuselage {
                length,
                max_width,
                max_height,
                ..
            },
        ) => f.wall_mass_per_m2 * fuselage_surface_area(*length, *max_width, *max_height),
        // Single-panel mass; a mirrored mount doubles it at the call site.
        (
            CatalogEntry::Wing(w),
            PartParams::Wing {
                span,
                root_chord,
                tip_chord,
                ..
            },
        ) => w.mass_per_m2 * wing_panel_area(*span, *root_chord, *tip_chord),
        // A self-contained gearbox: the formula counts every leg (both legs
        // for `gear_main`), so this is the whole assembly's mass.
        (CatalogEntry::Gear(g), PartParams::Gear { strut_length, .. }) => {
            gear_dry_mass(g, *strut_length)
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
            surface_mounts: vec![],
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
                resources: None,
            }],
            connections: vec![],
            surface_mounts: vec![],
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
                    resources: None,
                },
                PartBlueprint {
                    catalog_id: "tank_methalox".into(),
                    params: PartParams::Tank {
                        diameter: 2.5,
                        length: 4.0,
                    },
                    resources: None,
                },
                PartBlueprint {
                    catalog_id: "zephyr".into(),
                    params: PartParams::None,
                    resources: None,
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
            surface_mounts: vec![],
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
    fn explicit_empty_tank_resources_disable_catalog_defaults() {
        let cat = catalog();
        let bp = ShipBlueprint {
            name: "dry".into(),
            root: 0,
            parts: vec![PartBlueprint {
                catalog_id: "tank_methalox".into(),
                params: PartParams::Tank {
                    diameter: 2.5,
                    length: 2.0,
                },
                resources: Some(HashMap::new()),
            }],
            connections: vec![],
            surface_mounts: vec![],
        };

        let s = bp.stats(&cat).unwrap();
        assert_eq!(s.propellant_mass_kg, 0.0);
        assert!(s.resources.is_empty());
    }

    #[test]
    fn non_whitelisted_resource_is_rejected() {
        let cat = catalog();
        let bp = ShipBlueprint {
            name: "bad".into(),
            root: 0,
            parts: vec![PartBlueprint {
                catalog_id: "tank_methalox".into(),
                params: PartParams::Tank {
                    diameter: 2.5,
                    length: 2.0,
                },
                resources: Some(HashMap::from([(Resource::Kerosene, 100.0)])),
            }],
            connections: vec![],
            surface_mounts: vec![],
        };

        assert!(matches!(
            bp.stats(&cat),
            Err(CatalogError::ResourceNotAllowed {
                resource: Resource::Kerosene,
                ..
            })
        ));
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
                    resources: None,
                },
                PartBlueprint {
                    catalog_id: "tank_methalox".into(),
                    params: PartParams::Tank {
                        diameter: 2.5,
                        length: 4.0,
                    },
                    resources: None,
                },
                PartBlueprint {
                    catalog_id: "zephyr".into(),
                    params: PartParams::None,
                    resources: None,
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
            surface_mounts: vec![],
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
                    resources: None,
                },
                PartBlueprint {
                    catalog_id: "tank_methalox".into(),
                    params: PartParams::Tank {
                        diameter: 2.5,
                        length: 2.0,
                    },
                    resources: None,
                },
            ],
            connections: vec![Connection {
                parent: 0,
                parent_node: "bottom".into(),
                child: 1,
                child_node: "top".into(),
            }],
            surface_mounts: vec![],
        };
        // Same blueprint with no connection — the tank stays at its
        // declared 2.5 m and the pod is a separate root. Roll inertia
        // is just the sum of self-inertia at the declared diameters.
        let bp_unattached = ShipBlueprint {
            connections: vec![],
            surface_mounts: vec![],
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
    fn mirrored_wing_pair_doubles_area_and_mass() {
        use crate::attach::SurfaceMountKind;
        use crate::blueprint::SurfaceConnection;
        let cat = catalog();
        // KSP symmetry: the mirrored pair is two real wing parts sharing a
        // symmetry group, so the area/mass doubling comes from there being
        // two entities — not a per-part multiplier.
        let wing = || PartBlueprint {
            catalog_id: "wing_std".into(),
            params: PartParams::Wing {
                span: 5.0,
                root_chord: 2.0,
                tip_chord: 1.0,
                sweep: 0.2,
                dihedral: 0.05,
                thickness: 0.12,
                incidence: 0.0,
            },
            resources: None,
        };
        let bp = ShipBlueprint {
            name: "plane".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    catalog_id: "hyperion".into(),
                    params: PartParams::None,
                    resources: None,
                },
                wing(), // 1 primary
                wing(), // 2 mirror counterpart
            ],
            connections: vec![],
            surface_mounts: vec![
                SurfaceConnection {
                    parent: 0,
                    child: 1,
                    kind: SurfaceMountKind::BodySkin,
                    station: 0.5,
                    angle: std::f32::consts::FRAC_PI_2,
                    symmetry_group: Some(0),
                },
                SurfaceConnection {
                    parent: 0,
                    child: 2,
                    kind: SurfaceMountKind::BodySkin,
                    station: 0.5,
                    angle: -std::f32::consts::FRAC_PI_2,
                    symmetry_group: Some(0),
                },
            ],
        };
        let s = bp.stats(&cat).unwrap();
        // Each panel area = 5 · (2 + 1)/2 = 7.5 m²; two panels → 15 m².
        assert!((s.wing_area_m2 - 15.0).abs() < 1e-3);
        assert!(s.mean_aerodynamic_chord_m > 0.0);
        // Wing dry mass = 22 kg/m² · 7.5 · 2 panels = 330 kg, plus the
        // 5800 kg Hyperion pod.
        assert!((s.dry_mass_kg - (5800.0 + 330.0)).abs() < 1.0);
    }

    #[test]
    fn nacelle_pair_doubles_thrust_and_mass() {
        use crate::attach::SurfaceMountKind;
        use crate::blueprint::SurfaceConnection;

        let cat = catalog();
        let wing = || PartBlueprint {
            catalog_id: "wing_std".into(),
            params: PartParams::Wing {
                span: 5.0,
                root_chord: 2.0,
                tip_chord: 1.0,
                sweep: 0.2,
                dihedral: 0.05,
                thickness: 0.12,
                incidence: 0.0,
            },
            resources: None,
        };
        let nacelle = || PartBlueprint {
            catalog_id: "mistral_jet".into(),
            params: PartParams::None,
            resources: None,
        };
        let bp = ShipBlueprint {
            name: "jet".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    catalog_id: "hyperion".into(),
                    params: PartParams::None,
                    resources: None,
                },
                wing(),    // 1 primary wing
                wing(),    // 2 mirror wing
                nacelle(), // 3 nacelle on primary wing
                nacelle(), // 4 nacelle on mirror wing
            ],
            connections: vec![],
            surface_mounts: vec![
                SurfaceConnection {
                    parent: 0,
                    child: 1,
                    kind: SurfaceMountKind::BodySkin,
                    station: 0.5,
                    angle: std::f32::consts::FRAC_PI_2,
                    symmetry_group: Some(0),
                },
                SurfaceConnection {
                    parent: 0,
                    child: 2,
                    kind: SurfaceMountKind::BodySkin,
                    station: 0.5,
                    angle: -std::f32::consts::FRAC_PI_2,
                    symmetry_group: Some(0),
                },
                SurfaceConnection {
                    parent: 1,
                    child: 3,
                    kind: SurfaceMountKind::WingPylon,
                    station: 0.45,
                    angle: 0.0,
                    symmetry_group: Some(1),
                },
                SurfaceConnection {
                    parent: 2,
                    child: 4,
                    kind: SurfaceMountKind::WingPylon,
                    station: 0.45,
                    angle: 0.0,
                    symmetry_group: Some(1),
                },
            ],
        };

        let s = bp.stats(&cat).unwrap();
        assert!((s.total_thrust_n - 240_000.0).abs() < 1e-6);
        assert!((s.combined_isp_s - 2600.0).abs() < 1e-6);
        assert!(s.dry_mass_kg > 5800.0 + 2.0 * 680.0);
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
                resources: None,
            }],
            connections: vec![],
            surface_mounts: vec![],
        };
        assert!(matches!(bp.stats(&cat), Err(CatalogError::UnknownId(_))));
    }
}
