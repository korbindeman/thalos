//! RON loader for the solar system definition.
//!
//! The file format is `assets/solar_system.ron`. Every body specifies every
//! field — there are no defaults — so missing fields fail at parse time.
//!
//! Angles in the file are in degrees (human-readable); the loader converts
//! to radians at parse time. Distances are in meters.

use std::collections::HashMap;

use glam::DVec3;
use serde::Deserialize;
use thalos_terrain_gen::GeneratorParams;

use crate::types::{
    BodyDefinition, BodyId, BodyKind, G, OrbitalElements, ShipDefinition,
    SolarSystemDefinition, StateVector,
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
/// without orbital elements are root-frame (the star). Bodies without a
/// `generator` block don't run the surface generator yet.
#[derive(Debug, Deserialize)]
pub struct BodyFile {
    pub name: String,
    pub kind: BodyKind,
    pub parent: Option<String>,
    pub physical: PhysicalParams,
    pub orbit: Option<OrbitFile>,
    pub generator: Option<GeneratorParams>,
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
            orbital_elements,
            generator: b.generator,
        });
    }

    // Build the ship state: 200 km altitude circular orbit around the homeworld.
    let homeworld = name_to_id
        .get(&file.homeworld)
        .map(|&id| &bodies[id])
        .ok_or_else(|| format!("homeworld '{}' not found", file.homeworld))?;

    let orbit_radius = homeworld.radius_m + 200_000.0;
    let orbital_speed = (homeworld.gm / orbit_radius).sqrt();

    let ship = ShipDefinition {
        initial_state: StateVector {
            position: DVec3::new(orbit_radius, 0.0, 0.0),
            velocity: DVec3::new(0.0, 0.0, orbital_speed),
        },
        thrust_acceleration: 10.0,
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
