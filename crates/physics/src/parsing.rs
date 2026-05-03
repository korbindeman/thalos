//! RON loader for the solar system definition.
//!
//! The file format is `assets/solar_system.ron`. Every body specifies every
//! field — there are no defaults — so missing fields fail at parse time.
//!
//! Angles in the file are in degrees (human-readable); the loader converts
//! to radians at parse time. Distances are in meters.

use std::collections::HashMap;

use serde::Deserialize;
use thalos_atmosphere_gen::{AtmosphereParams, RingSystem, TerrestrialAtmosphere};
use thalos_terrain_gen::{GeneratorParams, TerrainConfig};

use crate::debug_orbits::debug_parking_orbit_relative_state;
use crate::types::{
    BodyDefinition, BodyId, BodyKind, G, OrbitalElements, ShipDefinition, SolarSystemDefinition,
};

// ---------------------------------------------------------------------------
// File schema
// ---------------------------------------------------------------------------

/// Top-level container deserialized from the RON file.
#[derive(Debug, Deserialize)]
pub struct SolarSystemFile {
    pub name: String,
    pub epoch: String,
    pub homeworld: String,
    pub bodies: Vec<BodyFile>,
}

/// One body in the file. `parent` references another body by name. Bodies
/// without orbital elements are root-frame (the star). Bodies may use either
/// the new feature terrain block or the legacy generator block.
#[derive(Debug, Deserialize)]
pub struct BodyFile {
    pub name: String,
    pub kind: BodyKind,
    pub parent: Option<String>,
    pub physical: PhysicalParams,
    pub orbit: Option<OrbitFile>,
    #[serde(default)]
    pub generator: Option<GeneratorParams>,
    #[serde(default)]
    pub terrain: Option<TerrainConfig>,
    #[serde(default)]
    pub atmosphere: Option<AtmosphereParams>,
    #[serde(default)]
    pub terrestrial_atmosphere: Option<TerrestrialAtmosphere>,
    #[serde(default)]
    pub rings: Option<RingSystem>,
}

#[derive(Debug, Deserialize)]
pub struct PhysicalParams {
    pub mass_kg: f64,
    pub radius_m: f64,
    pub color: String,
    pub albedo: f32,
    pub rotation_period_s: f64,
    pub axial_tilt_deg: f64,
}

/// Keplerian elements as authored in the file. Angles in degrees.
#[derive(Debug, Deserialize)]
pub struct OrbitFile {
    pub semi_major_axis_m: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub lon_ascending_node_deg: f64,
    pub arg_periapsis_deg: f64,
    pub true_anomaly_deg: f64,
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Parse the RON solar system definition file.
pub fn load_solar_system(source: &str) -> Result<SolarSystemDefinition, String> {
    let file: SolarSystemFile =
        ron::from_str(source).map_err(|e| format!("RON parse error: {e}"))?;

    // First pass: assign IDs.
    let mut name_to_id: HashMap<String, BodyId> = HashMap::with_capacity(file.bodies.len());
    for (i, b) in file.bodies.iter().enumerate() {
        if name_to_id.insert(b.name.clone(), i).is_some() {
            return Err(format!("duplicate body name '{}'", b.name));
        }
    }

    // Second pass: build BodyDefinitions, resolving parent names to IDs.
    let mut bodies = Vec::with_capacity(file.bodies.len());
    for (id, b) in file.bodies.into_iter().enumerate() {
        let parent = match &b.parent {
            Some(name) => Some(*name_to_id.get(name).ok_or_else(|| {
                format!("body '{}' references unknown parent '{}'", b.name, name)
            })?),
            None => None,
        };

        let orbital_elements = b.orbit.map(|o| OrbitalElements {
            semi_major_axis_m: o.semi_major_axis_m,
            eccentricity: o.eccentricity,
            inclination_rad: o.inclination_deg.to_radians(),
            lon_ascending_node_rad: o.lon_ascending_node_deg.to_radians(),
            arg_periapsis_rad: o.arg_periapsis_deg.to_radians(),
            true_anomaly_rad: o.true_anomaly_deg.to_radians(),
        });

        let terrain = b.terrain.unwrap_or_else(|| match &b.generator {
            Some(generator) => TerrainConfig::LegacyPipeline(generator.clone()),
            None => TerrainConfig::None,
        });

        bodies.push(BodyDefinition {
            id,
            name: b.name,
            kind: b.kind,
            parent,
            mass_kg: b.physical.mass_kg,
            radius_m: b.physical.radius_m,
            color: parse_hex_color(&b.physical.color),
            albedo: b.physical.albedo,
            rotation_period_s: b.physical.rotation_period_s,
            axial_tilt_rad: b.physical.axial_tilt_deg.to_radians(),
            gm: G * b.physical.mass_kg,
            soi_radius_m: 0.0, // filled below once all bodies exist
            orbital_elements,
            terrain,
            generator: b.generator,
            atmosphere: b.atmosphere,
            terrestrial_atmosphere: b.terrestrial_atmosphere,
            rings: b.rings,
        });
    }

    // SOI radius pass.  Needs parent mass, so do it after all bodies exist.
    // r_SOI = a * (m / M_parent)^(2/5).  Root bodies (no parent) get
    // f64::INFINITY so they always serve as the fallback anchor.
    for i in 0..bodies.len() {
        let soi = match bodies[i].parent {
            None => f64::INFINITY,
            Some(parent_id) => {
                let parent_mass = bodies[parent_id].mass_kg;
                let a = bodies[i]
                    .orbital_elements
                    .map(|o| o.semi_major_axis_m)
                    .unwrap_or(0.0);
                if a > 0.0 && parent_mass > 0.0 {
                    a * (bodies[i].mass_kg / parent_mass).powf(0.4)
                } else {
                    0.0
                }
            }
        };
        bodies[i].soi_radius_m = soi;
    }

    // Debug spawn: park the ship in the same low orbit used by debug teleports.
    let homeworld = name_to_id
        .get(&file.homeworld)
        .map(|&id| &bodies[id])
        .ok_or_else(|| format!("homeworld '{}' not found", file.homeworld))?;

    let ship = ShipDefinition {
        initial_state: debug_parking_orbit_relative_state(homeworld),
    };

    Ok(SolarSystemDefinition {
        name: file.name,
        bodies,
        ship,
        name_to_id,
    })
}

fn parse_hex_color(hex: &str) -> [f32; 3] {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(255) as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(255) as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(255) as f32 / 255.0;
    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::*;
    use thalos_terrain_gen::{BodyArchetype, TerrainConfig};

    #[test]
    fn mira_uses_feature_terrain_when_loaded_from_asset() {
        let source = include_str!("../../../assets/solar_system.ron");
        let system = load_solar_system(source).expect("parse solar_system.ron");
        let mira = system.body_by_name("Mira").expect("Mira exists");

        match &mira.terrain {
            TerrainConfig::Feature(config) => {
                assert_eq!(config.archetype, BodyArchetype::AirlessImpactMoon);
            }
            other => panic!("Mira should use feature terrain, got {other:?}"),
        }
    }

    #[test]
    fn vaelen_uses_feature_terrain_when_loaded_from_asset() {
        let source = include_str!("../../../assets/solar_system.ron");
        let system = load_solar_system(source).expect("parse solar_system.ron");
        let vaelen = system.body_by_name("Vaelen").expect("Vaelen exists");

        match &vaelen.terrain {
            TerrainConfig::Feature(config) => {
                assert_eq!(config.archetype, BodyArchetype::ColdDesertFormerlyWet);
            }
            other => panic!("Vaelen should use feature terrain, got {other:?}"),
        }
    }

    #[test]
    fn migrated_airless_bodies_use_feature_terrain() {
        let source = include_str!("../../../assets/solar_system.ron");
        let system = load_solar_system(source).expect("parse solar_system.ron");

        for name in ["Selva", "Carpo", "Theron", "Nyx"] {
            let body = system.body_by_name(name).expect("body exists");
            assert!(
                body.generator.is_none(),
                "{name} should not retain a legacy generator block"
            );
            match &body.terrain {
                TerrainConfig::Feature(config) => {
                    assert_eq!(config.archetype, BodyArchetype::AirlessImpactMoon);
                }
                other => {
                    panic!("{name} should use AirlessImpactMoon feature terrain, got {other:?}")
                }
            }
        }

        let legacy: Vec<_> = system
            .bodies
            .iter()
            .filter(|body| matches!(body.terrain, TerrainConfig::LegacyPipeline(_)))
            .map(|body| body.name.as_str())
            .collect();
        assert_eq!(legacy, vec!["Thalos", "Pelagos"]);
    }
}
