//! Parts catalog. Single source of truth for part stats.
//!
//! Map keys are stable [`CatalogId`]s assigned once and never changed.
//! `display_name` is the mutable, user-facing label. Saved blueprints
//! reference parts by ID, not by name — renaming "Boreas" to "Aestus" is
//! a one-line catalog edit and old saves keep loading.
//!
//! Catalog entries fall into two broad shapes:
//! - Fixed-geometry parts (Pod, Engine, Intake): full stats, no per-instance
//!   geometry parameters.
//! - Parametric parts (Decoupler, Adapter, Tank, Wing): the entry holds
//!   recipes (mass per area, storage per volume) and the blueprint provides
//!   variable geometry.

use crate::part::ReactantRatio;
use crate::resource::Resource;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub type CatalogId = String;

/// Marker component carrying the [`CatalogId`] that this part was spawned
/// from. Consumed at save time to round-trip blueprints back to their
/// catalog references.
#[derive(Component, Debug, Clone)]
pub struct CatalogRef {
    pub id: CatalogId,
}

#[derive(Resource, Debug, Clone, Serialize, Deserialize)]
pub struct PartCatalog {
    pub parts: HashMap<CatalogId, CatalogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CatalogEntry {
    Pod(PodSpec),
    Engine(EngineSpec),
    Intake(IntakeSpec),
    Decoupler(DecouplerSpec),
    Adapter(AdapterSpec),
    Tank(TankSpec),
    Wing(WingSpec),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodSpec {
    pub display_name: String,
    pub diameter: f32,
    pub dry_mass: f32,
    pub reaction_wheel_torque: f32,
    /// Whitelisted resource storage this part may carry.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSpec {
    pub display_name: String,
    #[serde(default)]
    pub optimized_for: EngineOptimization,
    #[serde(default)]
    pub geometry: EngineGeometry,
    /// Air-breathing engines need a body atmosphere. This is a runtime gate;
    /// design-time stats still report their nominal thrust until the aero
    /// model grows environment-aware estimates.
    #[serde(default)]
    pub requires_atmosphere: bool,
    /// Ambient flow the engine needs at full rated thrust.
    #[serde(default)]
    pub intake_requirement: Option<IntakeRequirement>,
    /// Intake capture built into this engine's own housing. Jet nacelles use
    /// this; later buried/standalone engine cores can omit it and depend on
    /// separate intake parts.
    #[serde(default)]
    pub builtin_intake: Option<IntakeCapture>,
    pub diameter: f32,
    pub thrust: f32,
    pub isp: f32,
    pub dry_mass: f32,
    pub reactants: Vec<ReactantRatio>,
    pub power_draw_kw: f32,
    /// Whitelisted resource storage this part may carry.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntakeSpec {
    pub display_name: String,
    pub diameter: f32,
    pub length: f32,
    pub dry_mass: f32,
    pub capture: IntakeCapture,
    /// Whitelisted resource storage this part may carry.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AmbientIntakeKind {
    /// Oxygen-bearing atmospheric air for gas-turbine style engines.
    #[default]
    OxygenatedAir,
}

impl AmbientIntakeKind {
    pub fn label(self) -> &'static str {
        match self {
            AmbientIntakeKind::OxygenatedAir => "Oxygenated air",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IntakeRequirement {
    #[serde(default)]
    pub kind: AmbientIntakeKind,
    /// Effective intake capture area required at full rated thrust, m².
    pub area_m2: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IntakeCapture {
    #[serde(default)]
    pub kind: AmbientIntakeKind,
    /// Geometric/capture area, m².
    pub area_m2: f32,
    /// Scalar efficiency applied to area until a real aero/ram model exists.
    #[serde(default = "default_intake_efficiency")]
    pub efficiency: f32,
}

fn default_intake_efficiency() -> f32 {
    1.0
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineOptimization {
    Atmosphere,
    Vacuum,
    #[default]
    Balanced,
}

impl EngineOptimization {
    pub fn label(self) -> &'static str {
        match self {
            EngineOptimization::Atmosphere => "Atmosphere",
            EngineOptimization::Vacuum => "Vacuum",
            EngineOptimization::Balanced => "Balanced",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineGeometry {
    /// Bell/nozzle engine for stack-mounted rockets.
    #[default]
    RocketBell,
    /// Cylindrical jet/turbofan nacelle, optionally carried under a wing by
    /// an auto-generated pylon.
    JetNacelle,
}

impl EngineGeometry {
    pub fn label(self) -> &'static str {
        match self {
            EngineGeometry::RocketBell => "Rocket bell",
            EngineGeometry::JetNacelle => "Jet nacelle",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecouplerSpec {
    pub display_name: String,
    /// Linear scaling: dry_mass = mass_per_diameter × diameter (kg/m).
    pub mass_per_diameter: f32,
    /// Linear scaling: ejection impulse = factor × diameter (N·s/m).
    pub ejection_impulse_per_diameter: f32,
    /// Whitelisted resource storage this part may carry.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterSpec {
    pub display_name: String,
    /// dry_mass = wall_mass_per_m2 × frustum lateral surface area.
    pub wall_mass_per_m2: f32,
    /// Whitelisted resource storage this part may carry.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TankSpec {
    pub display_name: String,
    /// dry_mass = wall_mass_per_m2 × cylinder surface area (sides + caps).
    pub wall_mass_per_m2: f32,
    /// Whitelisted resources this tank can carry. Capacity is normally
    /// volume-scaled (`units_per_m3`) for tanks, but fixed capacity is also
    /// supported so the same schema works for small batteries, service bays,
    /// and future internal compartments.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
    /// Mass-fraction reactant ratios for stats aggregation.
    pub reactants: Vec<ReactantRatio>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WingSpec {
    pub display_name: String,
    /// dry_mass = wall_mass_per_m2 × planform area (per panel; doubled for
    /// a mirrored pair). Lifting surfaces are lighter per area than
    /// pressure tanks — spar + skin, not a sealed vessel.
    pub mass_per_m2: f32,
    /// Whitelisted resource storage this part may carry. Empty today; future
    /// wet wings can opt into kerosene here without changing the schema.
    #[serde(default)]
    pub storage: Vec<ResourceStorageSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStorageSpec {
    pub resource: Resource,
    /// Fixed capacity in the resource's native unit.
    #[serde(default)]
    pub units: f32,
    /// Capacity density in native units per cubic metre of this part's
    /// geometric storage volume.
    #[serde(default)]
    pub units_per_m3: f32,
    /// Whether a new/auto-loaded part starts with this resource pool active.
    /// Inactive whitelisted resources can be added later by the editor.
    #[serde(default = "default_storage_enabled")]
    pub default_enabled: bool,
    /// Initial fill ratio for auto/default pools.
    #[serde(default = "default_fill_fraction")]
    pub default_fill_fraction: f32,
}

fn default_storage_enabled() -> bool {
    true
}

fn default_fill_fraction() -> f32 {
    1.0
}

impl CatalogEntry {
    pub fn display_name(&self) -> &str {
        match self {
            CatalogEntry::Pod(p) => &p.display_name,
            CatalogEntry::Engine(e) => &e.display_name,
            CatalogEntry::Intake(i) => &i.display_name,
            CatalogEntry::Decoupler(d) => &d.display_name,
            CatalogEntry::Adapter(a) => &a.display_name,
            CatalogEntry::Tank(t) => &t.display_name,
            CatalogEntry::Wing(w) => &w.display_name,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            CatalogEntry::Pod(_) => "Pod",
            CatalogEntry::Engine(_) => "Engine",
            CatalogEntry::Intake(_) => "Intake",
            CatalogEntry::Decoupler(_) => "Decoupler",
            CatalogEntry::Adapter(_) => "Adapter",
            CatalogEntry::Tank(_) => "Tank",
            CatalogEntry::Wing(_) => "Wing",
        }
    }

    pub fn storage_options(&self) -> &[ResourceStorageSpec] {
        match self {
            CatalogEntry::Pod(p) => &p.storage,
            CatalogEntry::Engine(e) => &e.storage,
            CatalogEntry::Intake(i) => &i.storage,
            CatalogEntry::Decoupler(d) => &d.storage,
            CatalogEntry::Adapter(a) => &a.storage,
            CatalogEntry::Tank(t) => &t.storage,
            CatalogEntry::Wing(w) => &w.storage,
        }
    }
}

impl PartCatalog {
    /// Look up a catalog entry by ID. Returns [`CatalogError::UnknownId`]
    /// when the ID isn't present — load-time blueprint resolution should
    /// fail fast on this.
    pub fn resolve(&self, id: &str) -> Result<&CatalogEntry, CatalogError> {
        self.parts
            .get(id)
            .ok_or_else(|| CatalogError::UnknownId(id.to_string()))
    }

    /// Parse a catalog from a RON string.
    pub fn load_from_str(s: &str) -> Result<Self, CatalogError> {
        ron::from_str(s).map_err(|e| CatalogError::Parse(e.to_string()))
    }

    /// Read and parse a catalog file.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, CatalogError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| CatalogError::Io {
            path: path.to_path_buf(),
            source: e.to_string(),
        })?;
        Self::load_from_str(&text)
    }
}

#[derive(Debug, Clone)]
pub enum CatalogError {
    UnknownId(String),
    KindMismatch {
        id: String,
        expected: &'static str,
        got: &'static str,
    },
    /// Parametric catalog kind (Tank/Adapter/Decoupler) referenced with
    /// `PartParams::None`, or pure kind (Pod/Engine) referenced with
    /// non-None params.
    ParamMismatch {
        id: String,
        kind: &'static str,
    },
    ResourceNotAllowed {
        id: String,
        resource: Resource,
    },
    Io {
        path: PathBuf,
        source: String,
    },
    Parse(String),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::UnknownId(id) => write!(f, "unknown catalog id: {id}"),
            CatalogError::KindMismatch { id, expected, got } => {
                write!(
                    f,
                    "catalog kind mismatch for {id}: expected {expected}, got {got}"
                )
            }
            CatalogError::ParamMismatch { id, kind } => {
                write!(
                    f,
                    "blueprint params do not match catalog kind {kind} for id {id}"
                )
            }
            CatalogError::ResourceNotAllowed { id, resource } => {
                write!(f, "part {id} cannot store {}", resource.display_name())
            }
            CatalogError::Io { path, source } => {
                write!(f, "failed to read catalog at {}: {source}", path.display())
            }
            CatalogError::Parse(msg) => write!(f, "failed to parse catalog: {msg}"),
        }
    }
}

impl std::error::Error for CatalogError {}

// ---- geometry helpers used by catalog→part composition -------------------

/// Cylindrical tank surface area: lateral wall (2πrL) plus two flat caps
/// (2 × πr²). Hemispherical caps would be a refinement; flat is a fine
/// proxy for mass scaling.
pub fn tank_surface_area(diameter: f32, length: f32) -> f32 {
    let r = diameter * 0.5;
    let lateral = std::f32::consts::PI * diameter * length;
    let caps = 2.0 * std::f32::consts::PI * r * r;
    lateral + caps
}

/// Tank cylindrical volume in m³.
pub fn tank_volume(diameter: f32, length: f32) -> f32 {
    let r = diameter * 0.5;
    std::f32::consts::PI * r * r * length
}

/// Planform (top-down projected) area of one trapezoidal wing panel, m².
/// A tapered panel is a trapezoid of height `span` and parallel sides
/// `root_chord`, `tip_chord`. Sweep shears the trapezoid, which leaves the
/// area unchanged, so it does not enter here.
pub fn wing_panel_area(span: f32, root_chord: f32, tip_chord: f32) -> f32 {
    span * (root_chord + tip_chord) * 0.5
}

/// Mean aerodynamic chord of one trapezoidal panel, m. For a linear taper
/// with taper ratio λ = tip/root: MAC = (2/3)·root·(1 + λ + λ²)/(1 + λ).
/// Degenerates to the chord for an untapered panel.
pub fn wing_mean_aerodynamic_chord(root_chord: f32, tip_chord: f32) -> f32 {
    if root_chord <= 0.0 {
        return 0.0;
    }
    let lambda = (tip_chord / root_chord).clamp(0.0, 1.0);
    (2.0 / 3.0) * root_chord * (1.0 + lambda + lambda * lambda) / (1.0 + lambda)
}

/// Lateral surface area of a frustum with the two given diameters and a
/// height inferred the same way the editor's adapter mesh does it
/// (`((d_top + d_bot) / 2).max(0.4)`). Keeps mass and visual geometry
/// consistent with each other.
pub fn adapter_surface_area(top_diameter: f32, bottom_diameter: f32) -> f32 {
    let h = ((top_diameter + bottom_diameter) * 0.5).max(0.4);
    let r1 = top_diameter * 0.5;
    let r2 = bottom_diameter * 0.5;
    let dr = r1 - r2;
    let slant = (h * h + dr * dr).sqrt();
    std::f32::consts::PI * (r1 + r2) * slant
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_catalog() {
        let text = include_str!("../../../assets/parts.ron");
        let cat = PartCatalog::load_from_str(text).expect("parse parts.ron");
        assert!(cat.resolve("argos").is_ok());
        assert!(cat.resolve("hyperion").is_ok());
        assert!(cat.resolve("zephyr").is_ok());
        assert!(cat.resolve("boreas").is_ok());
        assert!(cat.resolve("tank_methalox").is_ok());
        assert!(cat.resolve("adapter_std").is_ok());
        assert!(cat.resolve("decoupler_std").is_ok());
        assert!(cat.resolve("nope").is_err());

        let CatalogEntry::Engine(zephyr) = cat.resolve("zephyr").unwrap() else {
            panic!("zephyr should be an engine");
        };
        assert_eq!(zephyr.optimized_for, EngineOptimization::Atmosphere);
        assert_eq!(zephyr.geometry, EngineGeometry::RocketBell);

        let CatalogEntry::Engine(mistral) = cat.resolve("mistral_jet").unwrap() else {
            panic!("mistral_jet should be an engine");
        };
        assert_eq!(mistral.geometry, EngineGeometry::JetNacelle);
        assert!(mistral.requires_atmosphere);
        assert!(mistral.intake_requirement.is_some());
        assert!(mistral.builtin_intake.is_some());
        assert_eq!(mistral.reactants[0].resource, Resource::Kerosene);

        let CatalogEntry::Intake(intake) = cat.resolve("intake_cone").unwrap() else {
            panic!("intake_cone should be an intake");
        };
        assert_eq!(intake.capture.kind, AmbientIntakeKind::OxygenatedAir);

        let CatalogEntry::Tank(jet_tank) = cat.resolve("tank_kerosene").unwrap() else {
            panic!("tank_kerosene should be a tank");
        };
        assert!(
            jet_tank
                .storage
                .iter()
                .any(|s| s.resource == Resource::Kerosene)
        );
    }

    #[test]
    fn frustum_surface_area_is_finite_and_positive() {
        assert!(adapter_surface_area(2.5, 4.0) > 0.0);
        assert!(adapter_surface_area(2.5, 2.5) > 0.0); // degenerate cylinder still > 0
    }

    #[test]
    fn tank_volume_scales_with_length() {
        let small = tank_volume(2.5, 1.0);
        let large = tank_volume(2.5, 4.0);
        assert!((large - 4.0 * small).abs() < 1e-3);
    }
}
