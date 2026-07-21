//! RON loader for the solar system definition.
//!
//! The file format is `assets/solar_system.ron` for the main system structure,
//! with per-body detail files at `assets/bodies/<lowercase_name>.ron` for
//! terrain, atmosphere, and rings definitions.
//!
//! Angles in the file are in degrees (human-readable); the loader converts
//! to radians at parse time. Distances are in meters.

use std::collections::HashMap;
use std::path::Path;

use crate::atmosphere::{AtmosphereParams, RingSystem, TerrestrialAtmosphere};
use crate::ocean::OceanState;
use serde::Deserialize;
use thalos_terrain::{TectonicConfig, TerrainConfig};

use crate::body::{BodyDefinition, BodyId, BodyKind, G, OrbitalElements, SolarSystemDefinition};

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

/// Per-body detail file containing terrain, atmosphere, rings, tectonics.
/// All fields are optional — bodies without these blocks will have None fields.
#[derive(Debug, Deserialize, Default)]
pub struct BodyDetailsFile {
    #[serde(default)]
    pub terrain: Option<TerrainConfig>,
    #[serde(default)]
    pub ocean: Option<OceanState>,
    #[serde(default)]
    pub tectonics: Option<TectonicConfig>,
    #[serde(default)]
    pub atmosphere: Option<AtmosphereParams>,
    #[serde(default)]
    pub terrestrial_atmosphere: Option<TerrestrialAtmosphere>,
    #[serde(default)]
    pub rings: Option<RingSystem>,
    #[serde(default)]
    pub surface_frame_ceiling_m: Option<f64>,
}

/// One body in the file. `parent` references another body by name. Bodies
/// without orbital elements are root-frame (the star).
#[derive(Debug, Deserialize)]
pub struct BodyFile {
    pub name: String,
    pub kind: BodyKind,
    pub parent: Option<String>,
    pub physical: PhysicalParams,
    pub orbit: Option<OrbitFile>,
    #[serde(default)]
    pub terrain: Option<TerrainConfig>,
    #[serde(default)]
    pub ocean: Option<OceanState>,
    #[serde(default)]
    pub tectonics: Option<TectonicConfig>,
    #[serde(default)]
    pub atmosphere: Option<AtmosphereParams>,
    #[serde(default)]
    pub terrestrial_atmosphere: Option<TerrestrialAtmosphere>,
    #[serde(default)]
    pub rings: Option<RingSystem>,
    #[serde(default)]
    pub surface_frame_ceiling_m: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct PhysicalParams {
    pub mass_kg: f64,
    pub radius_m: f64,
    pub color: String,
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

/// Parse the RON solar system definition file (with no per-body details).
/// For the standard runtime usage, prefer [`load_solar_system_from_dir`] which
/// loads per-body files. This function is useful for tests that embed data.
pub fn load_solar_system(source: &str) -> Result<SolarSystemDefinition, String> {
    let file: SolarSystemFile =
        ron::from_str(source).map_err(|e| format!("RON parse error: {e}"))?;
    load_solar_system_impl(&file)
}

/// Load the solar system from `root/solar_system.ron` and per-body files at
/// `root/bodies/<lowercase_name>.ron`. Calls the in-memory loader
/// internally after loading and collating the per-body files.
pub fn load_solar_system_from_dir(root: &Path) -> Result<SolarSystemDefinition, String> {
    let system_path = root.join("solar_system.ron");
    let system_source = std::fs::read_to_string(&system_path)
        .map_err(|e| format!("Could not read {}: {}", system_path.display(), e))?;

    // Parse the main file first to get body names
    let file: SolarSystemFile = ron::from_str(&system_source)
        .map_err(|e| format!("RON parse error in solar_system.ron: {e}"))?;

    let bodies_dir = root.join("bodies");
    let mut body_details: HashMap<String, String> = HashMap::new();

    // Load per-body files for all bodies
    for body in &file.bodies {
        let details_path = bodies_dir.join(format!("{}.ron", body.name.to_lowercase()));
        if details_path.exists() {
            let details_source = std::fs::read_to_string(&details_path)
                .map_err(|e| format!("Could not read {}: {}", details_path.display(), e))?;
            body_details.insert(body.name.clone(), details_source);
        }
    }

    // Convert owned Strings to borrowed &str for the loader
    let body_details_refs: HashMap<String, &str> = body_details
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str()))
        .collect();

    load_solar_system_with_bodies(&system_source, &body_details_refs)
}

/// Parse solar system from in-memory sources: the main system definition and
/// a map of body names to their detail file contents. Used for testing and
/// for the path-based loader.
pub fn load_solar_system_with_bodies(
    system_source: &str,
    body_details: &HashMap<String, &str>,
) -> Result<SolarSystemDefinition, String> {
    let mut file: SolarSystemFile =
        ron::from_str(system_source).map_err(|e| format!("RON parse error: {e}"))?;

    // Merge per-body details into the system file
    for body in &mut file.bodies {
        if let Some(details_source) = body_details.get(&body.name) {
            let details: BodyDetailsFile = ron::from_str(details_source)
                .map_err(|e| format!("RON parse error in {}.ron: {e}", body.name))?;
            // Only override if the detail file provides the field
            if details.terrain.is_some() {
                body.terrain = details.terrain;
            }
            if details.ocean.is_some() {
                body.ocean = details.ocean;
            }
            if details.tectonics.is_some() {
                body.tectonics = details.tectonics;
            }
            if details.atmosphere.is_some() {
                body.atmosphere = details.atmosphere;
            }
            if details.terrestrial_atmosphere.is_some() {
                body.terrestrial_atmosphere = details.terrestrial_atmosphere;
            }
            if details.rings.is_some() {
                body.rings = details.rings;
            }
            if details.surface_frame_ceiling_m.is_some() {
                body.surface_frame_ceiling_m = details.surface_frame_ceiling_m;
            }
        }
    }

    // Now run the core loading logic on the merged file
    load_solar_system_impl(&file)
}

/// Core implementation that takes a parsed file and produces the final definition.
fn load_solar_system_impl(file: &SolarSystemFile) -> Result<SolarSystemDefinition, String> {
    // First pass: assign IDs.
    let mut name_to_id: HashMap<String, BodyId> = HashMap::with_capacity(file.bodies.len());
    for (i, b) in file.bodies.iter().enumerate() {
        if name_to_id.insert(b.name.clone(), i).is_some() {
            return Err(format!("duplicate body name '{}'", b.name));
        }
    }

    // Second pass: build BodyDefinitions, resolving parent names to IDs.
    let mut bodies = Vec::with_capacity(file.bodies.len());
    for (id, b) in file.bodies.iter().enumerate() {
        let parent = match &b.parent {
            Some(name) => Some(*name_to_id.get(name).ok_or_else(|| {
                format!("body '{}' references unknown parent '{}'", b.name, name)
            })?),
            None => None,
        };

        let orbital_elements = b.orbit.as_ref().map(|o| OrbitalElements {
            semi_major_axis_m: o.semi_major_axis_m,
            eccentricity: o.eccentricity,
            inclination_rad: o.inclination_deg.to_radians(),
            lon_ascending_node_rad: o.lon_ascending_node_deg.to_radians(),
            arg_periapsis_rad: o.arg_periapsis_deg.to_radians(),
            true_anomaly_rad: o.true_anomaly_deg.to_radians(),
        });

        let terrain = b.terrain.clone().unwrap_or(TerrainConfig::None);
        let tectonics = b.tectonics.clone();
        match (terrain.ocean_sea_level_m(), b.ocean) {
            (Some(_), Some(ocean)) => ocean
                .validate()
                .map_err(|reason| format!("body '{}' has invalid ocean state: {reason}", b.name))?,
            (Some(_), None) => {
                return Err(format!(
                    "body '{}' has ocean terrain but no authored ocean state",
                    b.name
                ));
            }
            (None, Some(_)) => {
                return Err(format!(
                    "body '{}' authors ocean state but its terrain has no sea level",
                    b.name
                ));
            }
            (None, None) => {}
        }

        bodies.push(BodyDefinition {
            id,
            name: b.name.clone(),
            kind: b.kind,
            parent,
            mass_kg: b.physical.mass_kg,
            radius_m: b.physical.radius_m,
            color: parse_hex_color(&b.physical.color),
            rotation_period_s: b.physical.rotation_period_s,
            axial_tilt_rad: b.physical.axial_tilt_deg.to_radians(),
            gm: G * b.physical.mass_kg,
            soi_radius_m: 0.0, // filled below once all bodies exist
            orbital_elements,
            terrain,
            ocean: b.ocean,
            tectonics,
            atmosphere: b.atmosphere.clone(),
            terrestrial_atmosphere: b.terrestrial_atmosphere.clone(),
            rings: b.rings.clone(),
            surface_frame_ceiling_m: b.surface_frame_ceiling_m,
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

    let homeworld_id = *name_to_id
        .get(&file.homeworld)
        .ok_or_else(|| format!("homeworld '{}' not found", file.homeworld))?;

    Ok(SolarSystemDefinition {
        name: file.name.clone(),
        bodies,
        name_to_id,
        homeworld_id,
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
    use thalos_terrain::TerrainConfig;

    #[test]
    fn mira_renders_as_placeholder_without_terrain() {
        // Terrain generation is intentionally disabled for Mira (no per-body
        // generator yet), so it must parse to `TerrainConfig::None` — the
        // discriminator `rendering::spawn` reads to route Mira to the
        // solid-colour placeholder impostor instead of the Earth-like
        // `ProceduralSurface`.
        let system_source = include_str!("../../../assets/solar_system.ron");
        let mira_details = include_str!("../../../assets/bodies/mira.ron");

        let mut details = HashMap::new();
        details.insert("Mira".to_string(), mira_details);

        let system =
            load_solar_system_with_bodies(system_source, &details).expect("parse solar_system.ron");
        let mira = system.body_by_name("Mira").expect("Mira exists");

        assert!(
            matches!(mira.terrain, TerrainConfig::None),
            "Mira should have no procedural terrain (placeholder), got {:?}",
            mira.terrain
        );
    }

    #[test]
    fn vaelen_renders_as_placeholder_without_terrain() {
        // See `mira_renders_as_placeholder_without_terrain`. Vaelen is likewise a
        // placeholder until a real generator exists; its retained
        // `terrestrial_atmosphere` must not resurrect procedural terrain.
        let system_source = include_str!("../../../assets/solar_system.ron");
        let vaelen_details = include_str!("../../../assets/bodies/vaelen.ron");

        let mut details = HashMap::new();
        details.insert("Vaelen".to_string(), vaelen_details);

        let system =
            load_solar_system_with_bodies(system_source, &details).expect("parse solar_system.ron");
        let vaelen = system.body_by_name("Vaelen").expect("Vaelen exists");

        assert!(
            matches!(vaelen.terrain, TerrainConfig::None),
            "Vaelen should have no procedural terrain (placeholder), got {:?}",
            vaelen.terrain
        );
    }

    #[test]
    fn pelagos_uses_ocean_terrain() {
        // Pelagos is the only Ocean body in the current authored set —
        // Thalos was migrated from the Ocean placeholder to
        // `Feature(AgingOceanicHomeworld)`. If more Ocean bodies are
        // authored later, fold them into a loop here.
        let system_source = include_str!("../../../assets/solar_system.ron");
        let pelagos_details = include_str!("../../../assets/bodies/pelagos.ron");

        let mut details = HashMap::new();
        details.insert("Pelagos".to_string(), pelagos_details);

        let system =
            load_solar_system_with_bodies(system_source, &details).expect("parse solar_system.ron");

        let pelagos = system.body_by_name("Pelagos").expect("Pelagos exists");
        assert!(
            matches!(pelagos.terrain, TerrainConfig::Ocean(_)),
            "Pelagos should use Ocean terrain, got {:?}",
            pelagos.terrain
        );
    }
}
