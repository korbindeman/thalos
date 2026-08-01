//! Can this vehicle actually fly?
//!
//! The shipyard will happily assemble a stack that is guaranteed to fail —
//! an engine that cannot reach any matching propellant, a design with no
//! propulsion at all, a stage that ignites and immediately runs dry. Nothing
//! caught those at build time, so the player discovered them in flight, often
//! at 40 km with an ascent program that could only abort.
//!
//! This module is the one definition of "unflyable", pure and index-based in
//! the same style as [`crate::staging`], so the editor and any flight-side
//! gate agree by construction rather than by convention.
//!
//! **What this does not do.** It refuses configurations that *cannot work*,
//! never ones that are merely bad. Too little Δv for a particular orbit, a
//! marginal TWR, an inefficient staging split — those are the player's
//! problem and the ORBIT preflight's business, and blocking them here would
//! turn a construction toy into a spreadsheet. The bar is: *is there a flight
//! in which this part could do its job?* If no, block it. If yes, say nothing.
//!
//! The crossfeed rule mirrors the live propulsion model exactly (`fuel.rs`'s
//! `crossfeed_components`): propellant flows across an attach edge only when
//! **both** endpoints permit crossfeed, which makes each decoupler a wall.
//! That is the rule that makes "the tank is right there" insufficient — an
//! engine below a decoupler cannot drink from tanks above it.

use std::collections::{HashMap, VecDeque};

use crate::blueprint::{ShipBlueprint, pools_for};
use crate::catalog::{CatalogEntry, CatalogError, PartCatalog};
use crate::resource::Resource;
use crate::staging::StageSummary;

/// How badly wrong a finding is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FlyabilitySeverity {
    /// The vehicle may fly; something is likely not what the player meant.
    Warning,
    /// The vehicle cannot work. Launch is refused.
    Blocking,
}

/// A specific way a build is broken.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlyabilityFault {
    /// No stage has a working engine — nothing can ever produce thrust.
    NoPropulsion,
    /// An engine's crossfeed component holds no capacity for one or more of
    /// its reactants. It can never fire, in any stage, at any point of the
    /// flight — the propellant is not merely spent, it is unreachable.
    EngineStarved {
        part: usize,
        label: String,
        missing: Vec<Resource>,
    },
    /// A stage ignites but has no propellant to burn: it would light and
    /// immediately stage. A warning, not a block — a mid-build stack passes
    /// through this state constantly, and an ullage or sepratron stage is a
    /// legitimate design.
    StageWithoutDeltaV { number: usize },
}

/// One problem found in a build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlyabilityFinding {
    pub severity: FlyabilitySeverity,
    pub fault: FlyabilityFault,
}

impl FlyabilityFinding {
    /// One-line player-facing text. Says what is wrong *and* what to do —
    /// a build-time error the player cannot act on is just a locked door.
    pub fn message(&self) -> String {
        match &self.fault {
            FlyabilityFault::NoPropulsion => {
                "No engine on any stage — the craft cannot produce thrust".to_string()
            }
            FlyabilityFault::EngineStarved { label, missing, .. } => {
                let names: Vec<&str> = missing.iter().map(|r| r.display_name()).collect();
                format!(
                    "{label} cannot reach any {} — add a tank on its side of the decoupler",
                    names.join(" or "),
                )
            }
            FlyabilityFault::StageWithoutDeltaV { number } => {
                format!("Stage {number} ignites with no propellant to burn")
            }
        }
    }
}

/// Per-part input, indexed `0..n`. Deliberately not a blueprint or an ECS
/// type so the rule is testable in isolation and shared verbatim by every
/// caller — the same contract [`crate::staging::SummaryPart`] keeps.
#[derive(Debug, Clone)]
pub struct FlyabilityPart {
    /// Attach parent, or `None` for a root. Surface mounts record a parent
    /// too, so a wing-mounted tank feeds the stack it hangs on.
    pub parent: Option<usize>,
    /// Whether propellant may pass *through* this part. False on decouplers,
    /// which is what makes them fuel walls as well as stage boundaries.
    pub crossfeed: bool,
    /// Propellant *capacity* by resource, in native units. Capacity rather
    /// than current amount: this asks whether the plumbing can ever work, not
    /// whether the tanks happen to be full right now.
    pub capacity: HashMap<Resource, f64>,
    /// The resources this part consumes, if it is an engine.
    pub engine_reactants: Option<Vec<Resource>>,
    /// Display name, for the message.
    pub label: String,
}

/// Connected components of the crossfeed graph.
///
/// Mirrors `fuel.rs`'s `crossfeed_components` exactly: an edge exists only
/// when both endpoints permit crossfeed, and an isolated part is its own
/// component. Kept as a separate function so the correspondence can be read
/// — and tested — without a World.
fn crossfeed_components(parts: &[FlyabilityPart]) -> Vec<usize> {
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); parts.len()];
    for (child, part) in parts.iter().enumerate() {
        let Some(parent) = part.parent else {
            continue;
        };
        if parent >= parts.len() {
            continue;
        }
        if part.crossfeed && parts[parent].crossfeed {
            adjacency[child].push(parent);
            adjacency[parent].push(child);
        }
    }

    let mut component = vec![usize::MAX; parts.len()];
    let mut next = 0usize;
    for start in 0..parts.len() {
        if component[start] != usize::MAX {
            continue;
        }
        let id = next;
        next += 1;
        component[start] = id;
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency[node] {
                if component[neighbor] != usize::MAX {
                    continue;
                }
                component[neighbor] = id;
                queue.push_back(neighbor);
            }
        }
    }
    component
}

/// Every reason this build cannot fly, worst first.
///
/// `stages` is the same [`StageSummary`] list the editor already shows, so
/// the Δv and has-engine judgements here are literally the numbers on screen
/// rather than a second opinion that could disagree with them.
pub fn check_flyability(
    parts: &[FlyabilityPart],
    stages: &[StageSummary],
) -> Vec<FlyabilityFinding> {
    let mut findings = Vec::new();

    // An empty canvas is not a broken build — it is an unstarted one. The
    // caller's own "nothing to launch" path handles that, and reporting a
    // fault here would light the panel red before the first part is placed.
    if parts.is_empty() {
        return findings;
    }

    if !stages.is_empty() && !stages.iter().any(|stage| stage.has_engine) {
        findings.push(FlyabilityFinding {
            severity: FlyabilitySeverity::Blocking,
            fault: FlyabilityFault::NoPropulsion,
        });
    }

    let component = crossfeed_components(parts);
    let mut capacity_by_component: HashMap<(usize, Resource), f64> = HashMap::new();
    for (index, part) in parts.iter().enumerate() {
        for (&resource, &capacity) in &part.capacity {
            *capacity_by_component
                .entry((component[index], resource))
                .or_default() += capacity;
        }
    }

    for (index, part) in parts.iter().enumerate() {
        let Some(reactants) = part.engine_reactants.as_ref() else {
            continue;
        };
        let missing: Vec<Resource> = reactants
            .iter()
            .copied()
            // Electricity is not stored in tanks and is generated, not
            // plumbed; treating it as a propellant would flag every
            // electric engine on a perfectly good stack.
            .filter(|resource| resource.is_mass_bearing())
            .filter(|resource| {
                capacity_by_component
                    .get(&(component[index], *resource))
                    .copied()
                    .unwrap_or(0.0)
                    <= 0.0
            })
            .collect();
        if !missing.is_empty() {
            findings.push(FlyabilityFinding {
                severity: FlyabilitySeverity::Blocking,
                fault: FlyabilityFault::EngineStarved {
                    part: index,
                    label: part.label.clone(),
                    missing,
                },
            });
        }
    }

    for stage in stages {
        if stage.has_engine && stage.delta_v_m_s <= 0.0 {
            findings.push(FlyabilityFinding {
                severity: FlyabilitySeverity::Warning,
                fault: FlyabilityFault::StageWithoutDeltaV {
                    number: stage.number,
                },
            });
        }
    }

    // Stable, so within a severity the order stays as discovered: propulsion
    // first, then engines in part order, then stages in firing order.
    findings.sort_by_key(|finding| std::cmp::Reverse(finding.severity));
    findings
}

/// `true` when any finding refuses launch.
pub fn blocks_launch(findings: &[FlyabilityFinding]) -> bool {
    findings
        .iter()
        .any(|finding| finding.severity == FlyabilitySeverity::Blocking)
}

impl ShipBlueprint {
    /// Everything wrong with this build, worst first.
    ///
    /// Shares [`ShipBlueprint::resolve_build`] with
    /// [`ShipBlueprint::stage_summaries`], so the topology this judges is the
    /// topology the editor displays — a second `parent` derivation is how a
    /// preview comes to disagree with the gate that refuses launch.
    pub fn flyability(
        &self,
        catalog: &PartCatalog,
    ) -> Result<Vec<FlyabilityFinding>, CatalogError> {
        let resolved = self.resolve_build(catalog)?;
        let stages = self.stage_summaries(catalog)?;

        let parts: Vec<FlyabilityPart> = self
            .parts
            .iter()
            .zip(&resolved.entries)
            .enumerate()
            .map(|(index, (pb, entry))| {
                let capacity = pools_for(entry, &pb.params, &pb.resources)
                    .into_iter()
                    .map(|(resource, pool)| (resource, pool.capacity as f64))
                    .collect();
                let engine_reactants = match entry {
                    CatalogEntry::Engine(e) => {
                        Some(e.reactants.iter().map(|r| r.resource).collect())
                    }
                    _ => None,
                };
                FlyabilityPart {
                    parent: resolved.parent[index],
                    // Mirrors `blueprint::spawn_part`: only a decoupler is
                    // built with `FuelCrossfeed { enabled: false }`; every
                    // other part takes the permissive default, and a part
                    // with no component at all reads as permitting flow.
                    crossfeed: !matches!(entry, CatalogEntry::Decoupler(_)),
                    capacity,
                    engine_reactants,
                    label: entry.display_name().to_string(),
                }
            })
            .collect();

        Ok(check_flyability(&parts, &stages))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tank(parent: Option<usize>, resources: &[(Resource, f64)]) -> FlyabilityPart {
        FlyabilityPart {
            parent,
            crossfeed: true,
            capacity: resources.iter().copied().collect(),
            engine_reactants: None,
            label: "Tank".to_string(),
        }
    }

    fn engine(parent: Option<usize>, reactants: &[Resource]) -> FlyabilityPart {
        FlyabilityPart {
            parent,
            crossfeed: true,
            capacity: HashMap::new(),
            engine_reactants: Some(reactants.to_vec()),
            label: "Engine".to_string(),
        }
    }

    fn decoupler(parent: Option<usize>) -> FlyabilityPart {
        FlyabilityPart {
            parent,
            crossfeed: false,
            capacity: HashMap::new(),
            engine_reactants: None,
            label: "Decoupler".to_string(),
        }
    }

    fn stage(number: usize, has_engine: bool, delta_v_m_s: f64) -> StageSummary {
        StageSummary {
            number,
            delta_v_m_s,
            fuel_kg: 1_000.0,
            initial_mass_kg: 10_000.0,
            thrust_n: if has_engine { 500_000.0 } else { 0.0 },
            resources: HashMap::new(),
            has_engine,
            active: false,
        }
    }

    #[test]
    fn a_working_stack_reports_nothing() {
        // pod(0) → tank(1) → engine(2)
        let parts = vec![
            tank(None, &[]),
            tank(
                Some(0),
                &[(Resource::Methane, 900.0), (Resource::Lox, 2_200.0)],
            ),
            engine(Some(1), &[Resource::Methane, Resource::Lox]),
        ];
        let findings = check_flyability(&parts, &[stage(1, true, 3_200.0)]);
        assert!(findings.is_empty(), "unexpected: {findings:?}");
    }

    #[test]
    fn an_engine_cut_off_by_a_decoupler_is_blocked() {
        // The whole point: the tank is *right there* in the tree, but the
        // decoupler between them is a fuel wall, so the engine is dead.
        // pod(0) → tank(1) → decoupler(2) → engine(3)
        let parts = vec![
            tank(None, &[]),
            tank(
                Some(0),
                &[(Resource::Methane, 900.0), (Resource::Lox, 2_200.0)],
            ),
            decoupler(Some(1)),
            engine(Some(2), &[Resource::Methane, Resource::Lox]),
        ];
        let findings = check_flyability(&parts, &[stage(1, true, 0.0)]);
        assert!(blocks_launch(&findings));
        let starved = findings
            .iter()
            .find(|f| matches!(f.fault, FlyabilityFault::EngineStarved { .. }))
            .expect("the isolated engine must be reported");
        let FlyabilityFault::EngineStarved { part, missing, .. } = &starved.fault else {
            unreachable!()
        };
        assert_eq!(*part, 3);
        assert_eq!(missing.len(), 2, "both reactants are unreachable");
    }

    #[test]
    fn a_partially_starved_engine_names_only_the_missing_reactant() {
        // Methane is reachable, LOX is not — the message must not claim both.
        let parts = vec![
            tank(None, &[(Resource::Methane, 900.0)]),
            engine(Some(0), &[Resource::Methane, Resource::Lox]),
        ];
        let findings = check_flyability(&parts, &[stage(1, true, 100.0)]);
        let FlyabilityFault::EngineStarved { missing, .. } = &findings[0].fault else {
            panic!("expected a starved engine");
        };
        assert_eq!(missing, &vec![Resource::Lox]);
    }

    #[test]
    fn a_stack_with_no_engine_at_all_is_blocked() {
        let parts = vec![tank(None, &[]), tank(Some(0), &[(Resource::Lox, 500.0)])];
        let findings = check_flyability(&parts, &[stage(1, false, 0.0)]);
        assert!(blocks_launch(&findings));
        assert!(
            findings
                .iter()
                .any(|f| f.fault == FlyabilityFault::NoPropulsion)
        );
    }

    #[test]
    fn a_drop_stage_without_an_engine_is_fine() {
        // Stage 2 is a pure decoupler stage — a legitimate design, not a fault.
        let parts = vec![
            tank(None, &[]),
            tank(
                Some(0),
                &[(Resource::Methane, 900.0), (Resource::Lox, 2_200.0)],
            ),
            engine(Some(1), &[Resource::Methane, Resource::Lox]),
        ];
        let findings = check_flyability(&parts, &[stage(1, true, 3_000.0), stage(2, false, 0.0)]);
        assert!(findings.is_empty(), "unexpected: {findings:?}");
    }

    #[test]
    fn a_dry_powered_stage_warns_without_blocking() {
        let parts = vec![
            tank(
                None,
                &[(Resource::Methane, 900.0), (Resource::Lox, 2_200.0)],
            ),
            engine(Some(0), &[Resource::Methane, Resource::Lox]),
        ];
        let findings = check_flyability(&parts, &[stage(1, true, 0.0)]);
        assert!(!blocks_launch(&findings), "a dry stage must not block");
        assert_eq!(
            findings[0].fault,
            FlyabilityFault::StageWithoutDeltaV { number: 1 }
        );
    }

    #[test]
    fn an_empty_canvas_is_not_a_broken_build() {
        assert!(check_flyability(&[], &[]).is_empty());
    }

    #[test]
    fn electricity_is_not_treated_as_plumbed_propellant() {
        let parts = vec![
            tank(
                None,
                &[(Resource::Methane, 900.0), (Resource::Lox, 2_200.0)],
            ),
            engine(
                Some(0),
                &[Resource::Methane, Resource::Lox, Resource::Electricity],
            ),
        ];
        let findings = check_flyability(&parts, &[stage(1, true, 3_000.0)]);
        assert!(findings.is_empty(), "unexpected: {findings:?}");
    }

    #[test]
    fn crossfeed_reaches_across_a_shared_parent() {
        // engine(2) and tank(1) are siblings under pod(0): the path runs
        // up through the pod and back down, which crossfeed permits.
        let parts = vec![
            tank(None, &[]),
            tank(Some(0), &[(Resource::Kerosene, 1_500.0)]),
            engine(Some(0), &[Resource::Kerosene]),
        ];
        let findings = check_flyability(&parts, &[stage(1, true, 2_000.0)]);
        assert!(findings.is_empty(), "unexpected: {findings:?}");
    }

    /// The strongest guard available: the vehicles we actually ship must
    /// pass their own gate. A false positive here would lock the player out
    /// of launching a stock design, which is far worse than the defect this
    /// module exists to catch — so this test is the one that decides whether
    /// a future rule is too strict.
    #[test]
    fn the_shipped_vehicles_are_flyable() {
        use crate::catalog::PartCatalog;

        let catalog = PartCatalog::load_from_str(include_str!("../../../../assets/parts.ron"))
            .expect("parse parts.ron");

        for (name, ron) in [
            ("apollo", include_str!("../../../../ships/apollo.ron")),
            ("atlas", include_str!("../../../../ships/atlas.ron")),
            ("meridian", include_str!("../../../../ships/meridian.ron")),
            ("saturn", include_str!("../../../../ships/saturn.ron")),
        ] {
            let blueprint = ShipBlueprint::from_ron(ron).unwrap_or_else(|e| {
                panic!("parse {name}.ron: {e}");
            });
            let findings = blueprint
                .flyability(&catalog)
                .unwrap_or_else(|e| panic!("{name} flyability: {e}"));
            assert!(
                !blocks_launch(&findings),
                "shipped vehicle {name} must not be refused launch: {:?}",
                findings
                    .iter()
                    .map(FlyabilityFinding::message)
                    .collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn blocking_findings_sort_ahead_of_warnings() {
        let parts = vec![tank(None, &[]), engine(Some(0), &[Resource::Kerosene])];
        let findings = check_flyability(&parts, &[stage(1, true, 0.0)]);
        assert!(findings.len() >= 2);
        assert_eq!(findings[0].severity, FlyabilitySeverity::Blocking);
    }
}
