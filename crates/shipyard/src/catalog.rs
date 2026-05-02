//! Parts catalog. Single source of truth for part stats.
//!
//! Map keys are stable [`CatalogId`]s assigned once and never changed.
//! `display_name` is the mutable, user-facing label. Saved blueprints
//! reference parts by ID, not by name — renaming "Boreas" to "Aestus" is
//! a one-line catalog edit and old saves keep loading.
//!
//! Catalog entries fall into two shapes:
//! - Pure (Pod, Engine): full stats, no per-instance parameters.
//! - Parametric (Decoupler, Adapter, Tank): the entry holds *recipes*
//!   (mass per area, capacity per volume) and the blueprint provides
//!   the variable bits (length, target diameter).

use crate::part::ReactantRatio;
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
    Decoupler(DecouplerSpec),
    Adapter(AdapterSpec),
    Tank(TankSpec),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodSpec {
    pub display_name: String,
    pub diameter: f32,
    pub dry_mass: f32,
    pub reaction_wheel_torque: f32,
    /// Initial onboard battery capacity, kWh.
    pub base_electricity_kwh: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSpec {
    pub display_name: String,
    #[serde(default)]
    pub optimized_for: EngineOptimization,
    pub diameter: f32,
    pub thrust: f32,
    pub isp: f32,
    pub dry_mass: f32,
    pub reactants: Vec<ReactantRatio>,
    pub power_draw_kw: f32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecouplerSpec {
    pub display_name: String,
    /// Linear scaling: dry_mass = mass_per_diameter × diameter (kg/m).
    pub mass_per_diameter: f32,
    /// Linear scaling: ejection impulse = factor × diameter (N·s/m).
    pub ejection_impulse_per_diameter: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterSpec {
    pub display_name: String,
    /// dry_mass = wall_mass_per_m2 × frustum lateral surface area.
    pub wall_mass_per_m2: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TankSpec {
    pub display_name: String,
    /// dry_mass = wall_mass_per_m2 × cylinder surface area (sides + caps).
    pub wall_mass_per_m2: f32,
    /// Methane capacity in litres per cubic metre of tank volume.
    pub methane_l_per_m3: f32,
    /// LOX capacity in litres per cubic metre of tank volume.
    pub lox_l_per_m3: f32,
    /// Mass-fraction reactant ratios for stats aggregation.
    pub reactants: Vec<ReactantRatio>,
}

impl CatalogEntry {
    pub fn display_name(&self) -> &str {
        match self {
            CatalogEntry::Pod(p) => &p.display_name,
            CatalogEntry::Engine(e) => &e.display_name,
            CatalogEntry::Decoupler(d) => &d.display_name,
            CatalogEntry::Adapter(a) => &a.display_name,
            CatalogEntry::Tank(t) => &t.display_name,
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            CatalogEntry::Pod(_) => "Pod",
            CatalogEntry::Engine(_) => "Engine",
            CatalogEntry::Decoupler(_) => "Decoupler",
            CatalogEntry::Adapter(_) => "Adapter",
            CatalogEntry::Tank(_) => "Tank",
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
