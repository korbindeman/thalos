// NOTE: Add `pub mod parsing;` to lib.rs

use std::collections::HashMap;

use glam::DVec3;

use crate::types::{
    BodyComposition, BodyDefinition, BodyId, BodyKind, BodyProceduralOverrides,
    BodyProceduralProfile, G, OrbitalElements, ShipDefinition, SolarSystemDefinition,
    StateVector, parse_hex_color,
};

const AU_TO_METERS: f64 = 1.496e11;

/// Parse the KDL solar system definition file.
pub fn load_solar_system(source: &str) -> Result<SolarSystemDefinition, String> {
    let doc: kdl::KdlDocument = source
        .parse()
        .map_err(|e| format!("KDL parse error: {e}"))?;

    let system_node = doc.get("system").ok_or("Missing 'system' node")?;
    let system_name = system_node
        .get("name")
        .and_then(|v| v.as_string())
        .unwrap_or("Unknown")
        .to_string();
    let homeworld_name = system_node
        .get("homeworld")
        .and_then(|v| v.as_string())
        .unwrap_or("Thalos");

    // First pass: collect all body nodes and assign IDs.
    let mut bodies = Vec::new();
    let mut name_to_id: HashMap<String, BodyId> = HashMap::new();

    for node in doc.nodes() {
        if node.name().to_string().as_str() != "body" {
            continue;
        }

        let name = node
            .get("name")
            .and_then(|v| v.as_string())
            .ok_or("Body missing 'name'")?
            .to_string();

        let kind_str = node
            .get("kind")
            .and_then(|v| v.as_string())
            .unwrap_or("Planet");
        let kind = match kind_str {
            "Star" => BodyKind::Star,
            "Planet" => BodyKind::Planet,
            "Moon" => BodyKind::Moon,
            "DwarfPlanet" => BodyKind::DwarfPlanet,
            "Centaur" => BodyKind::Centaur,
            "Comet" => BodyKind::Comet,
            other => return Err(format!("Unknown body kind: {other}")),
        };

        let parent_name = node.get("parent").and_then(|v| v.as_string());

        let mass_kg = node
            .get("mass_kg")
            .and_then(kdl_to_f64)
            .ok_or(format!("Body '{name}' missing mass_kg"))?;

        let radius_km = node
            .get("radius_km")
            .and_then(kdl_to_f64)
            .ok_or(format!("Body '{name}' missing radius_km"))?;

        let color_hex = node
            .get("color")
            .and_then(|v| v.as_string())
            .unwrap_or("#FFFFFF");

        let albedo = node
            .get("albedo")
            .and_then(kdl_to_f64)
            .map(|v| v as f32)
            .unwrap_or(0.3);

        let rotation_period_s = node
            .get("rotation_period_hours")
            .and_then(kdl_to_f64)
            .unwrap_or(0.0)
            * 3600.0;

        let axial_tilt_rad = node
            .get("axial_tilt_deg")
            .and_then(kdl_to_f64)
            .unwrap_or(0.0)
            .to_radians();

        // Parse orbital elements from child 'orbit' node.
        let orbital_elements = node.children().and_then(|children| {
            children.get("orbit").map(|orbit_node| {
                let sma_au = orbit_node
                    .get("semi_major_axis_au")
                    .and_then(kdl_to_f64)
                    .unwrap_or(1.0);
                let ecc = orbit_node
                    .get("eccentricity")
                    .and_then(kdl_to_f64)
                    .unwrap_or(0.0);
                let inc_deg = orbit_node
                    .get("inclination_deg")
                    .and_then(kdl_to_f64)
                    .unwrap_or(0.0);
                let lan_deg = orbit_node
                    .get("lon_ascending_node_deg")
                    .and_then(kdl_to_f64)
                    .unwrap_or(0.0);
                let aop_deg = orbit_node
                    .get("arg_periapsis_deg")
                    .and_then(kdl_to_f64)
                    .unwrap_or(0.0);
                let ta_deg = orbit_node
                    .get("true_anomaly_deg")
                    .and_then(kdl_to_f64)
                    .unwrap_or(0.0);

                OrbitalElements {
                    semi_major_axis_m: sma_au * AU_TO_METERS,
                    eccentricity: ecc,
                    inclination_rad: inc_deg.to_radians(),
                    lon_ascending_node_rad: lan_deg.to_radians(),
                    arg_periapsis_rad: aop_deg.to_radians(),
                    true_anomaly_rad: ta_deg.to_radians(),
                }
            })
        });

        let procedural = parse_procedural_profile(&name, node)?;

        let id = bodies.len();
        name_to_id.insert(name.clone(), id);

        bodies.push(BodyDefinition {
            id,
            name,
            kind,
            parent: parent_name.map(|_| {
                usize::MAX // placeholder, resolved in second pass below
            }),
            mass_kg,
            radius_m: radius_km * 1000.0,
            color: parse_hex_color(color_hex),
            albedo,
            rotation_period_s,
            axial_tilt_rad,
            gm: G * mass_kg,
            orbital_elements,
            procedural,
        });
    }

    // Second pass: resolve parent references.
    let name_to_id_clone = name_to_id.clone();
    for body in &mut bodies {
        if body.parent == Some(usize::MAX) {
            // Find the parent name from the original KDL
            // We need to re-extract it. Let's use a different approach.
        }
    }

    // Actually, let's do parent resolution properly by re-scanning the KDL.
    let mut body_idx = 0;
    for node in doc.nodes() {
        if node.name().to_string().as_str() != "body" {
            continue;
        }
        let parent_name = node.get("parent").and_then(|v| v.as_string());
        bodies[body_idx].parent = parent_name.and_then(|pn| name_to_id_clone.get(pn).copied());
        body_idx += 1;
    }

    // Compute initial ship state: 200km altitude circular orbit around the homeworld.
    let homeworld = name_to_id
        .get(homeworld_name)
        .map(|&id| &bodies[id])
        .ok_or("Homeworld not found")?;

    let orbit_radius = homeworld.radius_m + 200_000.0; // 200 km altitude
    let orbital_speed = (homeworld.gm / orbit_radius).sqrt();

    // We need the homeworld's initial position to place the ship.
    // The ship's absolute position will be set after ephemeris initialization.
    // For now, store the relative orbit parameters.
    let ship = ShipDefinition {
        initial_state: StateVector {
            position: DVec3::new(orbit_radius, 0.0, 0.0), // relative to homeworld, resolved later
            velocity: DVec3::new(0.0, 0.0, orbital_speed),
        },
        thrust_acceleration: 10.0, // ~1g, strong vacuum engine
    };

    Ok(SolarSystemDefinition {
        name: system_name,
        bodies,
        ship,
        name_to_id,
    })
}

fn kdl_to_f64(value: &kdl::KdlValue) -> Option<f64> {
    match value {
        kdl::KdlValue::Float(f) => Some(*f),
        kdl::KdlValue::Integer(i) => Some(*i as f64),
        kdl::KdlValue::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn kdl_to_u64(value: &kdl::KdlValue) -> Option<u64> {
    match value {
        kdl::KdlValue::Integer(i) => (*i).try_into().ok(),
        kdl::KdlValue::String(s) => s.parse::<u64>().ok(),
        _ => None,
    }
}

fn parse_procedural_profile(
    body_name: &str,
    node: &kdl::KdlNode,
) -> Result<Option<BodyProceduralProfile>, String> {
    let Some(children) = node.children() else {
        return Ok(None);
    };
    let Some(procedural_node) = children.get("procedural") else {
        return Ok(None);
    };

    let archetype = procedural_node
        .get("archetype")
        .and_then(|v| v.as_string())
        .unwrap_or("unspecified")
        .to_string();
    let surface_state = procedural_node
        .get("surface_state")
        .and_then(|v| v.as_string())
        .map(ToOwned::to_owned);
    let age_gyr = procedural_node
        .get("age_gyr")
        .and_then(kdl_to_f64)
        .ok_or_else(|| format!("Body '{body_name}' procedural block missing age_gyr"))?;
    let impact_flux_multiplier = procedural_node
        .get("impact_flux_multiplier")
        .and_then(kdl_to_f64)
        .ok_or_else(|| {
            format!("Body '{body_name}' procedural block missing impact_flux_multiplier")
        })?;
    let thermal_history = procedural_node
        .get("thermal_history")
        .and_then(kdl_to_f64)
        .ok_or_else(|| format!("Body '{body_name}' procedural block missing thermal_history"))?;
    let seed = procedural_node
        .get("seed")
        .and_then(kdl_to_u64)
        .ok_or_else(|| format!("Body '{body_name}' procedural block missing seed"))?;

    let procedural_children = procedural_node
        .children()
        .ok_or_else(|| format!("Body '{body_name}' procedural block missing children"))?;
    let composition_node = procedural_children
        .get("composition")
        .ok_or_else(|| format!("Body '{body_name}' procedural block missing composition"))?;
    let composition = BodyComposition {
        silicate: composition_node
            .get("silicate")
            .and_then(kdl_to_f64)
            .ok_or_else(|| format!("Body '{body_name}' composition missing silicate"))?,
        iron: composition_node
            .get("iron")
            .and_then(kdl_to_f64)
            .ok_or_else(|| format!("Body '{body_name}' composition missing iron"))?,
        ice: composition_node
            .get("ice")
            .and_then(kdl_to_f64)
            .ok_or_else(|| format!("Body '{body_name}' composition missing ice"))?,
        volatiles: composition_node
            .get("volatiles")
            .and_then(kdl_to_f64)
            .ok_or_else(|| format!("Body '{body_name}' composition missing volatiles"))?,
        hydrogen_helium: composition_node
            .get("hydrogen_helium")
            .and_then(kdl_to_f64)
            .ok_or_else(|| format!("Body '{body_name}' composition missing hydrogen_helium"))?,
    };

    let overrides = procedural_children
        .get("overrides")
        .map(|overrides_node| BodyProceduralOverrides {
            surface_gravity_m_s2: overrides_node
                .get("surface_gravity_m_s2")
                .and_then(kdl_to_f64),
            equilibrium_temperature_k: overrides_node
                .get("equilibrium_temperature_k")
                .and_then(kdl_to_f64),
            total_heat_budget_w: overrides_node.get("total_heat_budget_w").and_then(kdl_to_f64),
            hydrosphere_fraction: overrides_node
                .get("hydrosphere_fraction")
                .and_then(kdl_to_f64),
            magnetic_field_strength_rel: overrides_node
                .get("magnetic_field_strength_rel")
                .and_then(kdl_to_f64),
            tectonic_style: overrides_node
                .get("tectonic_style")
                .and_then(|v| v.as_string())
                .map(ToOwned::to_owned),
        })
        .unwrap_or_default();

    Ok(Some(BodyProceduralProfile {
        archetype,
        surface_state,
        age_gyr,
        impact_flux_multiplier,
        thermal_history,
        seed,
        composition,
        overrides,
    }))
}

#[cfg(test)]
mod tests {
    use super::load_solar_system;

    #[test]
    fn parses_authored_procedural_profile_from_solar_system() {
        let kdl = std::fs::read_to_string("../../assets/solar_system.kdl")
            .expect("run from workspace root");
        let system = load_solar_system(&kdl).expect("parse solar_system.kdl");
        let thalos = system.body_by_name("Thalos").expect("Thalos body");
        let profile = thalos.procedural.as_ref().expect("Thalos procedural profile");

        assert_eq!(profile.archetype, "temperate_terrestrial");
        assert_eq!(profile.seed, 1003);
        assert!((profile.composition.iron - 0.70).abs() < 1e-9);
        assert_eq!(profile.overrides.tectonic_style.as_deref(), Some("declining_plate"));
        assert!((thalos.rotation_period_s - 21.3 * 3600.0).abs() < 1e-9);
    }
}
