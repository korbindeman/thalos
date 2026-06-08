//! KSP-style staging derivation and per-stage Δv / fuel bookkeeping.
//!
//! A vessel's parts are grouped into an ordered sequence of stages, derived
//! from the **decoupler topology** of the attach tree — there is no authored
//! stage list. This module is the single home for that derivation and the
//! per-stage Δv accounting. Both consumers feed it the same way, so they can
//! never disagree on stage boundaries:
//!
//! - the game's live ECS staging systems ([`thalos_game`]) build the inputs
//!   from the running part entities (current fuel after burn), and
//! - the shipyard editor previews staging from a [`ShipBlueprint`] via
//!   [`ShipBlueprint::stage_summaries`] (full tanks, design-time).
//!
//! This mirrors the [`crate::stats::cylinder_principal_inertia`] "single
//! home" precedent: one model, two callers, no drift.

use std::collections::{HashMap, HashSet};

use crate::attach::MountSymmetry;
use crate::blueprint::{
    ShipBlueprint, check_params_match, check_resource_amounts_allowed, pools_for,
};
use crate::catalog::{CatalogEntry, CatalogError, PartCatalog};
use crate::resource::Resource;
use crate::stats::{
    DeltaVEnvironment, DeltaVInputs, G0, ResourceTotals, estimate_delta_v, part_dry_mass,
};

// ---------------------------------------------------------------------------
// Stage derivation (pure)
// ---------------------------------------------------------------------------

/// Role of a part for stage derivation. Only decouplers (stage boundaries)
/// and engines (ignition targets) matter; everything else is structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartRole {
    Decoupler,
    Engine,
    Other,
}

/// One stage expressed as part-index groups. `stages[0]` fires first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageIndices {
    pub engines: Vec<usize>,
    pub decouplers: Vec<usize>,
}

/// Derive the staging sequence from attach topology. `parent[i]` is the
/// index of part `i`'s attach parent, or `None` if it is a root.
///
/// The stack is walked bottom-up (deepest part first). Engines ignite into
/// the current stage; each decoupler opens the next stage — it fires one
/// stage after the engines below it have burned — and belongs to that new
/// stage. So the bottom engines light at launch (stage 0), and each later
/// stage drops the spent section below a decoupler while lighting the
/// engines newly exposed above it. A vessel with no decouplers is a single
/// stage that lights all its engines.
///
/// A leading stage with nothing to do (dead weight below a decoupler, with
/// no engine at the very bottom) is dropped so every activation has an
/// effect. Linear stacks only; radial/parallel decouplers are future work.
pub fn derive_stages(roles: &[PartRole], parent: &[Option<usize>]) -> Vec<StageIndices> {
    let n = roles.len();
    debug_assert_eq!(n, parent.len());

    // Depth from root, used only to walk the stack bottom-up.
    let mut depth = vec![0usize; n];
    for i in 0..n {
        let mut cursor = parent[i];
        let mut d = 0;
        // Hop budget guards against a malformed cyclic parent array.
        let mut budget = n;
        while let Some(p) = cursor {
            if budget == 0 {
                break;
            }
            budget -= 1;
            d += 1;
            cursor = parent[p];
        }
        depth[i] = d;
    }

    // Deepest first; ties broken by index for determinism.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| depth[b].cmp(&depth[a]).then(a.cmp(&b)));

    let mut stages = vec![StageIndices::default()];
    let mut current = 0usize;
    for i in order {
        match roles[i] {
            PartRole::Engine => stages[current].engines.push(i),
            PartRole::Decoupler => {
                current += 1;
                stages.push(StageIndices::default());
                stages[current].decouplers.push(i);
            }
            PartRole::Other => {}
        }
    }

    stages.retain(|s| !s.engines.is_empty() || !s.decouplers.is_empty());
    stages
}

// ---------------------------------------------------------------------------
// Per-stage Δv / fuel readout
// ---------------------------------------------------------------------------

/// One stage's vacuum Δv and propellant. The list is ordered by firing
/// order — the earliest-firing stage first.
#[derive(Clone, Debug)]
pub struct StageSummary {
    /// 1-based stage number for display.
    pub number: usize,
    /// Vacuum Δv this stage delivers, m/s.
    pub delta_v_m_s: f64,
    /// Total propellant this stage burns before it separates, kg.
    pub fuel_kg: f64,
    /// Per-resource totals for this stage's fuel section (mass-bearing only),
    /// for per-stage reactant bars.
    pub resources: HashMap<Resource, ResourceTotals>,
    /// False for a decoupler-only "drop" stage with no engine.
    pub has_engine: bool,
    /// The stage currently ignited / burning (the most recently activated
    /// one). Always false for a design-time preview (`next == 0`).
    pub active: bool,
}

/// Engine inputs to [`compute_stage_summaries`].
pub struct SummaryEngine {
    pub thrust_n: f64,
    pub isp_s: f64,
    pub reactants: Vec<(Resource, f64)>,
}

/// Per-part input to [`compute_stage_summaries`], indexed `0..n`. Decoupled
/// from any ECS or blueprint type so the Δv bookkeeping is testable in
/// isolation and shared verbatim by both callers.
pub struct SummaryPart {
    pub parent: Option<usize>,
    pub dry_mass_kg: f64,
    pub resources: HashMap<Resource, ResourceTotals>,
    pub engine: Option<SummaryEngine>,
}

/// A plan stage reduced to indices into the [`SummaryPart`] slice (only the
/// still-live engines / decouplers).
pub struct SummaryStageInput {
    pub number: usize,
    pub engines: Vec<usize>,
    pub decouplers: Vec<usize>,
}

fn part_total_mass_kg(p: &SummaryPart) -> f64 {
    p.dry_mass_kg + part_propellant_kg(p)
}

fn part_propellant_kg(p: &SummaryPart) -> f64 {
    p.resources.values().map(|t| t.mass_kg).sum()
}

fn collect_subtree(root: usize, children: &HashMap<usize, Vec<usize>>, out: &mut HashSet<usize>) {
    let mut stack = vec![root];
    while let Some(e) = stack.pop() {
        if !out.insert(e) {
            continue;
        }
        if let Some(kids) = children.get(&e) {
            stack.extend(kids.iter().copied());
        }
    }
}

/// Pure per-stage Δv / fuel computation. Stages are walked from the current
/// (first live) one outward: each stage's burn starts at the mass of all
/// parts still attached, and the section that drops when the *next* stage
/// fires is this stage's spent section (its propellant is this stage's fuel).
/// The final stage's section is everything remaining. Reuses the tested
/// [`estimate_delta_v`] with stage-scoped masses, engines, and resources.
///
/// `next` is the 0-based index of the next stage to fire (the live staging
/// plan's cursor); the stage whose 1-based `number` equals it is the one
/// currently burning, so nothing is marked active before the first
/// activation. A design-time preview passes `next == 0`.
pub fn compute_stage_summaries(
    stages: &[SummaryStageInput],
    parts: &[SummaryPart],
    next: usize,
) -> Vec<StageSummary> {
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, p) in parts.iter().enumerate() {
        if let Some(parent) = p.parent {
            children.entry(parent).or_default().push(i);
        }
    }

    // Only stages that still have live parts, in firing order.
    let live: Vec<&SummaryStageInput> = stages
        .iter()
        .filter(|s| !s.engines.is_empty() || !s.decouplers.is_empty())
        .collect();

    let mut present: HashSet<usize> = (0..parts.len()).collect();
    let mut out = Vec::with_capacity(live.len());

    for (i, stage) in live.iter().enumerate() {
        let m_start: f64 = present.iter().map(|&j| part_total_mass_kg(&parts[j])).sum();

        let dropped: HashSet<usize> = if i + 1 < live.len() {
            let mut d = HashSet::new();
            for &dec in &live[i + 1].decouplers {
                collect_subtree(dec, &children, &mut d);
            }
            d.retain(|x| present.contains(x));
            d
        } else {
            present.clone()
        };

        let fuel_kg: f64 = dropped.iter().map(|&j| part_propellant_kg(&parts[j])).sum();

        let mut resources: HashMap<Resource, ResourceTotals> = HashMap::new();
        for &j in &dropped {
            for (&res, totals) in &parts[j].resources {
                let entry = resources.entry(res).or_default();
                entry.amount += totals.amount;
                entry.capacity += totals.capacity;
                entry.mass_kg += totals.mass_kg;
            }
        }

        let (thrust_n, mass_flow, reactant_fractions) =
            aggregate_stage_engines(&stage.engines, parts);
        let has_engine = thrust_n > 0.0 && mass_flow > 0.0;

        let delta_v_m_s = if has_engine {
            estimate_delta_v(
                DeltaVEnvironment::Vacuum,
                DeltaVInputs {
                    dry_mass_kg: (m_start - fuel_kg).max(0.0),
                    wet_mass_kg: m_start,
                    total_thrust_n: thrust_n,
                    mass_flow_kg_per_s: mass_flow,
                    power_draw_kw: 0.0,
                    reactant_fractions: &reactant_fractions,
                    resources: &resources,
                },
            )
            .delta_v_m_per_s
        } else {
            0.0
        };

        out.push(StageSummary {
            number: stage.number,
            delta_v_m_s,
            fuel_kg,
            resources,
            has_engine,
            active: stage.number == next,
        });

        for x in &dropped {
            present.remove(x);
        }
    }

    out
}

/// Aggregate `(thrust_n, mass_flow_kg_per_s, reactant_fractions)` over a
/// stage's engines — the same per-resource mass-flow split the live
/// propulsion model uses.
fn aggregate_stage_engines(
    engines: &[usize],
    parts: &[SummaryPart],
) -> (f64, f64, HashMap<Resource, f64>) {
    let mut thrust_n = 0.0;
    let mut mass_flow = 0.0;
    let mut per_resource_mdot: HashMap<Resource, f64> = HashMap::new();

    for &e in engines {
        let Some(engine) = parts.get(e).and_then(|p| p.engine.as_ref()) else {
            continue;
        };
        if engine.thrust_n <= 0.0 || engine.isp_s <= 0.0 {
            continue;
        }
        let mdot = engine.thrust_n / (engine.isp_s * G0);
        let mut any = false;
        for &(res, frac) in &engine.reactants {
            if frac > 0.0 {
                *per_resource_mdot.entry(res).or_insert(0.0) += mdot * frac;
                any = true;
            }
        }
        if !any {
            continue;
        }
        thrust_n += engine.thrust_n;
        mass_flow += mdot;
    }

    let fractions = if mass_flow > 0.0 {
        per_resource_mdot
            .into_iter()
            .map(|(res, mdot)| (res, mdot / mass_flow))
            .collect()
    } else {
        HashMap::new()
    };
    (thrust_n, mass_flow, fractions)
}

// ---------------------------------------------------------------------------
// Blueprint preview
// ---------------------------------------------------------------------------

impl ShipBlueprint {
    /// Per-stage vacuum Δv / fuel breakdown for this blueprint, resolving
    /// every part against `catalog`. Returns [`CatalogError`] on the first
    /// unknown ID, mirroring [`ShipBlueprint::stats`].
    ///
    /// This is the design-time preview: tanks are full (the blueprint's
    /// stored amounts, defaulting full when omitted) and no stage is marked
    /// active. Stage boundaries come from [`derive_stages`] over the same
    /// decoupler topology the live staging plan uses, so the editor preview
    /// and the in-flight HUD never disagree.
    pub fn stage_summaries(
        &self,
        catalog: &PartCatalog,
    ) -> Result<Vec<StageSummary>, CatalogError> {
        let entries: Vec<&CatalogEntry> = self
            .parts
            .iter()
            .map(|pb| catalog.resolve(&pb.catalog_id))
            .collect::<Result<_, _>>()?;
        for (pb, entry) in self.parts.iter().zip(&entries) {
            check_params_match(&pb.catalog_id, entry, &pb.params)?;
            check_resource_amounts_allowed(&pb.catalog_id, entry, &pb.resources)?;
        }

        // Parent index per part, from the connection graph. Surface mounts
        // (wings) record a parent too so a wing drops with the host stage it
        // sits on, rather than looking like a separate root.
        let mut parent: Vec<Option<usize>> = vec![None; self.parts.len()];
        for c in &self.connections {
            if c.child < parent.len() {
                parent[c.child] = Some(c.parent);
            }
        }
        let mut panels = vec![1.0_f64; self.parts.len()];
        for m in &self.surface_mounts {
            if m.child < parent.len() {
                parent[m.child] = Some(m.parent);
            }
            if m.child < panels.len() && m.symmetry == MountSymmetry::Mirrored {
                panels[m.child] = 2.0;
            }
        }

        let roles: Vec<PartRole> = entries
            .iter()
            .map(|e| match e {
                CatalogEntry::Decoupler(_) => PartRole::Decoupler,
                CatalogEntry::Engine(_) => PartRole::Engine,
                _ => PartRole::Other,
            })
            .collect();

        let stages: Vec<SummaryStageInput> = derive_stages(&roles, &parent)
            .into_iter()
            .enumerate()
            .map(|(k, s)| SummaryStageInput {
                number: k + 1,
                engines: s.engines,
                decouplers: s.decouplers,
            })
            .collect();

        let parts: Vec<SummaryPart> = self
            .parts
            .iter()
            .zip(&entries)
            .enumerate()
            .map(|(i, (pb, entry))| {
                let resources = pools_for(entry, &pb.params, &pb.resources)
                    .into_iter()
                    .map(|(res, pool)| {
                        (
                            res,
                            ResourceTotals {
                                amount: pool.amount as f64,
                                capacity: pool.capacity as f64,
                                mass_kg: pool.mass_kg(res),
                            },
                        )
                    })
                    .collect();
                let engine = match entry {
                    CatalogEntry::Engine(e) => Some(SummaryEngine {
                        thrust_n: e.thrust as f64 * panels[i],
                        isp_s: e.isp as f64,
                        reactants: e
                            .reactants
                            .iter()
                            .map(|r| (r.resource, r.mass_fraction as f64))
                            .collect(),
                    }),
                    _ => None,
                };
                SummaryPart {
                    parent: parent[i],
                    dry_mass_kg: part_dry_mass(entry, &pb.params) as f64 * panels[i],
                    resources,
                    engine,
                }
            })
            .collect();

        Ok(compute_stage_summaries(&stages, &parts, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint::{Connection, PartBlueprint, PartParams};

    fn catalog() -> PartCatalog {
        PartCatalog::load_from_str(include_str!("../../../assets/parts.ron"))
            .expect("parse parts.ron")
    }

    fn engines(stage: &StageIndices) -> Vec<usize> {
        let mut e = stage.engines.clone();
        e.sort_unstable();
        e
    }

    #[test]
    fn single_stage_when_no_decouplers() {
        // pod(0) → tank(1) → engine(2)
        let roles = [PartRole::Other, PartRole::Other, PartRole::Engine];
        let parent = [None, Some(0), Some(1)];
        let stages = derive_stages(&roles, &parent);
        assert_eq!(stages.len(), 1);
        assert_eq!(engines(&stages[0]), vec![2]);
        assert!(stages[0].decouplers.is_empty());
    }

    #[test]
    fn two_stage_stack_fires_bottom_first() {
        // pod(0) → tank(1) → upper engine(2) → decoupler(3) → tank(4) → lower engine(5)
        let roles = [
            PartRole::Other,
            PartRole::Other,
            PartRole::Engine,
            PartRole::Decoupler,
            PartRole::Other,
            PartRole::Engine,
        ];
        let parent = [None, Some(0), Some(1), Some(2), Some(3), Some(4)];
        let stages = derive_stages(&roles, &parent);

        assert_eq!(stages.len(), 2);
        // Launch: lower engine ignites, nothing drops.
        assert_eq!(engines(&stages[0]), vec![5]);
        assert!(stages[0].decouplers.is_empty());
        // Stage 1: decoupler fires (drops lower section), upper engine ignites.
        assert_eq!(engines(&stages[1]), vec![2]);
        assert_eq!(stages[1].decouplers, vec![3]);
    }

    #[test]
    fn three_stage_stack_orders_decouplers_deepest_first() {
        // pod(0) → tank(1) → eng(2) → decB(3) → tank(4) → eng(5) → decA(6) → tank(7) → eng(8)
        let roles = [
            PartRole::Other,
            PartRole::Other,
            PartRole::Engine,
            PartRole::Decoupler,
            PartRole::Other,
            PartRole::Engine,
            PartRole::Decoupler,
            PartRole::Other,
            PartRole::Engine,
        ];
        let parent = [
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            Some(7),
        ];
        let stages = derive_stages(&roles, &parent);

        assert_eq!(stages.len(), 3);
        assert_eq!(engines(&stages[0]), vec![8]); // bottom engine first
        assert!(stages[0].decouplers.is_empty());
        assert_eq!(engines(&stages[1]), vec![5]);
        assert_eq!(stages[1].decouplers, vec![6]); // deepest decoupler (A)
        assert_eq!(engines(&stages[2]), vec![2]);
        assert_eq!(stages[2].decouplers, vec![3]); // shallower decoupler (B)
    }

    #[test]
    fn deadweight_below_engine_collapses_empty_launch_stage() {
        // pod(0) → tank(1) → engine(2) → decoupler(3) → spent tank(4), no lower
        // engine. Nothing burns at the very bottom, so the empty launch stage
        // collapses: a single stage both drops the dead tank and lights the
        // engine (rather than wasting a press doing nothing).
        let roles = [
            PartRole::Other,
            PartRole::Other,
            PartRole::Engine,
            PartRole::Decoupler,
            PartRole::Other,
        ];
        let parent = [None, Some(0), Some(1), Some(2), Some(3)];
        let stages = derive_stages(&roles, &parent);

        assert_eq!(stages.len(), 1);
        assert_eq!(engines(&stages[0]), vec![2]);
        assert_eq!(stages[0].decouplers, vec![3]);
    }

    #[test]
    fn two_stage_summary_matches_tsiolkovsky() {
        // pod 0, upper tank 1, upper engine 2, decoupler 3, lower tank 4,
        // lower engine 5. Single methane reactant keeps the arithmetic clean.
        let methane = Resource::Methane;
        let g0 = 9.806_65_f64;
        let prop = |kg: f64| {
            HashMap::from([(
                methane,
                ResourceTotals {
                    amount: kg,
                    capacity: kg,
                    mass_kg: kg,
                },
            )])
        };
        let boreas = || SummaryEngine {
            thrust_n: 1.8e6,
            isp_s: 380.0,
            reactants: vec![(methane, 1.0)],
        };
        let structure = |parent, kg| SummaryPart {
            parent,
            dry_mass_kg: kg,
            resources: HashMap::new(),
            engine: None,
        };

        let parts = vec![
            structure(None, 5800.0), // 0 pod
            SummaryPart {
                parent: Some(0),
                dry_mass_kg: 1000.0,
                resources: prop(20000.0),
                engine: None,
            }, // 1 upper tank
            SummaryPart {
                parent: Some(1),
                dry_mass_kg: 1800.0,
                resources: HashMap::new(),
                engine: Some(boreas()),
            }, // 2 upper engine
            structure(Some(2), 100.0), // 3 decoupler
            SummaryPart {
                parent: Some(3),
                dry_mass_kg: 2000.0,
                resources: prop(40000.0),
                engine: None,
            }, // 4 lower tank
            SummaryPart {
                parent: Some(4),
                dry_mass_kg: 1800.0,
                resources: HashMap::new(),
                engine: Some(boreas()),
            }, // 5 lower engine
        ];
        let stages = vec![
            SummaryStageInput {
                number: 1,
                engines: vec![5],
                decouplers: vec![],
            },
            SummaryStageInput {
                number: 2,
                engines: vec![2],
                decouplers: vec![3],
            },
        ];

        // Pre-launch (next = 0): nothing activated yet, so no stage is marked
        // active — the highlight only appears once the player stages.
        let prelaunch = compute_stage_summaries(&stages, &parts, 0);
        assert!(prelaunch.iter().all(|s| !s.active));

        // next = 1: stage 1 has been activated (launched) and is burning.
        let summaries = compute_stage_summaries(&stages, &parts, 1);
        assert_eq!(summaries.len(), 2);

        let ve = 380.0 * g0;

        // Stage 1 (lower) burns from the full 72.5 t stack, dropping 40 t.
        assert_eq!(summaries[0].number, 1);
        assert!(summaries[0].active);
        assert!(summaries[0].has_engine);
        assert!((summaries[0].fuel_kg - 40000.0).abs() < 1e-6);
        assert!((summaries[0].resources[&methane].mass_kg - 40000.0).abs() < 1e-6);
        let dv0 = ve * (72500.0_f64 / 32500.0).ln();
        assert!(
            (summaries[0].delta_v_m_s - dv0).abs() < 1.0,
            "stage 1 Δv {} vs expected {dv0}",
            summaries[0].delta_v_m_s
        );

        // Stage 2 (upper, final): 28.6 t section burning 20 t.
        assert_eq!(summaries[1].number, 2);
        assert!(!summaries[1].active);
        assert!((summaries[1].fuel_kg - 20000.0).abs() < 1e-6);
        assert!((summaries[1].resources[&methane].mass_kg - 20000.0).abs() < 1e-6);
        let dv1 = ve * (28600.0_f64 / 8600.0).ln();
        assert!(
            (summaries[1].delta_v_m_s - dv1).abs() < 1.0,
            "stage 2 Δv {} vs expected {dv1}",
            summaries[1].delta_v_m_s
        );
    }

    #[test]
    fn blueprint_two_stage_methalox_previews_both_stages() {
        // pod → upper tank → upper engine → decoupler → lower tank → lower
        // engine, all methalox. Exercises the blueprint → summary path
        // (role/parent extraction from catalog + connections, full tanks).
        let cat = catalog();
        let tank = || PartParams::Tank {
            diameter: 2.5,
            length: 4.0,
        };
        let bp = ShipBlueprint {
            name: "two-stage".into(),
            root: 0,
            parts: vec![
                PartBlueprint {
                    catalog_id: "argos".into(),
                    params: PartParams::None,
                    resources: None,
                },
                PartBlueprint {
                    catalog_id: "tank_methalox".into(),
                    params: tank(),
                    resources: None,
                },
                PartBlueprint {
                    catalog_id: "zephyr".into(),
                    params: PartParams::None,
                    resources: None,
                },
                PartBlueprint {
                    catalog_id: "decoupler_std".into(),
                    params: PartParams::Decoupler { diameter: 2.5 },
                    resources: None,
                },
                PartBlueprint {
                    catalog_id: "tank_methalox".into(),
                    params: tank(),
                    resources: None,
                },
                PartBlueprint {
                    catalog_id: "boreas".into(),
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
                Connection {
                    parent: 2,
                    parent_node: "bottom".into(),
                    child: 3,
                    child_node: "top".into(),
                },
                Connection {
                    parent: 3,
                    parent_node: "bottom".into(),
                    child: 4,
                    child_node: "top".into(),
                },
                Connection {
                    parent: 4,
                    parent_node: "bottom".into(),
                    child: 5,
                    child_node: "top".into(),
                },
            ],
            surface_mounts: vec![],
        };

        let summaries = bp.stage_summaries(&cat).expect("stage summaries");
        assert_eq!(summaries.len(), 2);
        // Firing order: lower (Boreas) stage first, then the upper (Zephyr).
        assert_eq!(summaries[0].number, 1);
        assert_eq!(summaries[1].number, 2);
        // Both stages burn methalox, so both deliver Δv and consume fuel.
        for s in &summaries {
            assert!(s.has_engine, "stage {} should have an engine", s.number);
            assert!(s.delta_v_m_s > 0.0, "stage {} Δv should be > 0", s.number);
            assert!(s.fuel_kg > 0.0, "stage {} fuel should be > 0", s.number);
        }
        // Design-time preview: nothing burning yet.
        assert!(summaries.iter().all(|s| !s.active));
    }
}
