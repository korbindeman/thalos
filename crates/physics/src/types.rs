use glam::DVec3;
use std::collections::HashMap;

/// Gravitational constant in m^3 kg^-1 s^-2.
pub const G: f64 = 6.674_30e-11;

/// Unique identifier for a celestial body.
pub type BodyId = usize;

/// Position + velocity in heliocentric inertial frame.
#[derive(Debug, Clone, Copy)]
pub struct StateVector {
    pub position: DVec3,
    pub velocity: DVec3,
}

/// Static properties of a celestial body (immutable after load).
#[derive(Debug, Clone)]
pub struct BodyDefinition {
    pub id: BodyId,
    pub name: String,
    pub kind: BodyKind,
    pub parent: Option<BodyId>,
    pub mass_kg: f64,
    pub radius_m: f64,
    pub color: [f32; 3],
    pub gm: f64, // G * mass, precomputed
    pub orbital_elements: Option<OrbitalElements>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyKind {
    Star,
    Planet,
    Moon,
    DwarfPlanet,
    Centaur,
    Comet,
}

/// Keplerian orbital elements at epoch.
#[derive(Debug, Clone, Copy)]
pub struct OrbitalElements {
    pub semi_major_axis_m: f64,
    pub eccentricity: f64,
    pub inclination_rad: f64,
    pub lon_ascending_node_rad: f64,
    pub arg_periapsis_rad: f64,
    pub true_anomaly_rad: f64,
}

/// A timestamped state for a body — used in ephemeris samples.
#[derive(Debug, Clone, Copy)]
pub struct BodyState {
    pub position: DVec3,
    pub velocity: DVec3,
    pub mass_kg: f64,
}

/// Snapshot of all body states at a given time.
pub type BodyStates = Vec<BodyState>;

/// A single sample of the ship's propagated trajectory.
#[derive(Debug, Clone, Copy)]
pub struct TrajectorySample {
    pub time: f64,
    pub position: DVec3,
    pub velocity: DVec3,
    pub dominant_body: BodyId,
    pub perturbation_ratio: f64,
    pub step_size: f64,
}

/// Ship definition — placeholder for MVP.
#[derive(Debug, Clone)]
pub struct ShipDefinition {
    pub initial_state: StateVector,
    /// Thrust acceleration magnitude in m/s².
    pub thrust_acceleration: f64,
}

/// Full solar system definition loaded from file.
#[derive(Debug, Clone)]
pub struct SolarSystemDefinition {
    pub name: String,
    pub bodies: Vec<BodyDefinition>,
    pub ship: ShipDefinition,
    /// Map from body name to BodyId for convenience.
    pub name_to_id: HashMap<String, BodyId>,
}

impl SolarSystemDefinition {
    pub fn body_by_name(&self, name: &str) -> Option<&BodyDefinition> {
        self.name_to_id.get(name).map(|&id| &self.bodies[id])
    }
}

/// Parse a hex color string like "#FFE54D" into [r, g, b] floats in 0..1.
pub fn parse_hex_color(hex: &str) -> [f32; 3] {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(255) as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(255) as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(255) as f32 / 255.0;
    [r, g, b]
}

const AU_TO_METERS: f64 = 1.496e11;

/// Parse the KDL solar system definition file.
pub fn load_solar_system(source: &str) -> Result<SolarSystemDefinition, String> {
    let doc: kdl::KdlDocument = source.parse().map_err(|e| format!("KDL parse error: {e}"))?;

    let system_node = doc
        .get("system")
        .ok_or("Missing 'system' node")?;
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
            .and_then(|v| kdl_to_f64(v))
            .ok_or(format!("Body '{name}' missing mass_kg"))?;

        let radius_km = node
            .get("radius_km")
            .and_then(|v| kdl_to_f64(v))
            .ok_or(format!("Body '{name}' missing radius_km"))?;

        let color_hex = node
            .get("color")
            .and_then(|v| v.as_string())
            .unwrap_or("#FFFFFF");

        // Parse orbital elements from child 'orbit' node.
        let orbital_elements = node.children().and_then(|children| {
            children.get("orbit").map(|orbit_node| {
                let sma_au = orbit_node
                    .get("semi_major_axis_au")
                    .and_then(|v| kdl_to_f64(v))
                    .unwrap_or(1.0);
                let ecc = orbit_node
                    .get("eccentricity")
                    .and_then(|v| kdl_to_f64(v))
                    .unwrap_or(0.0);
                let inc_deg = orbit_node
                    .get("inclination_deg")
                    .and_then(|v| kdl_to_f64(v))
                    .unwrap_or(0.0);
                let lan_deg = orbit_node
                    .get("lon_ascending_node_deg")
                    .and_then(|v| kdl_to_f64(v))
                    .unwrap_or(0.0);
                let aop_deg = orbit_node
                    .get("arg_periapsis_deg")
                    .and_then(|v| kdl_to_f64(v))
                    .unwrap_or(0.0);
                let ta_deg = orbit_node
                    .get("true_anomaly_deg")
                    .and_then(|v| kdl_to_f64(v))
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
            gm: G * mass_kg,
            orbital_elements,
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
        thrust_acceleration: 0.5, // placeholder: ~0.05g
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

/// Convert orbital elements to Cartesian state vector relative to parent.
pub fn orbital_elements_to_cartesian(elements: &OrbitalElements, parent_gm: f64) -> StateVector {
    let a = elements.semi_major_axis_m;
    let e = elements.eccentricity;
    let i = elements.inclination_rad;
    let omega = elements.lon_ascending_node_rad;
    let w = elements.arg_periapsis_rad;
    let nu = elements.true_anomaly_rad;

    // Semi-latus rectum.
    let p = a * (1.0 - e * e);

    // Distance from focus.
    let r = p / (1.0 + e * nu.cos());

    // Position in orbital plane.
    let x_orb = r * nu.cos();
    let y_orb = r * nu.sin();

    // Velocity in orbital plane.
    let mu_over_p = parent_gm / p;
    let vx_orb = -mu_over_p.sqrt() * nu.sin();
    let vy_orb = mu_over_p.sqrt() * (e + nu.cos());

    // Rotation matrix from orbital plane to 3D (ecliptic) frame.
    // Using the standard aerospace convention: XZ ecliptic, Y up.
    let cos_o = omega.cos();
    let sin_o = omega.sin();
    let cos_w = w.cos();
    let sin_w = w.sin();
    let cos_i = i.cos();
    let sin_i = i.sin();

    let px = cos_o * cos_w - sin_o * sin_w * cos_i;
    let py = sin_i * sin_w;
    let pz = sin_o * cos_w + cos_o * sin_w * cos_i;

    let qx = -cos_o * sin_w - sin_o * cos_w * cos_i;
    let qy = sin_i * cos_w;
    let qz = -sin_o * sin_w + cos_o * cos_w * cos_i;

    // Note: Y is up in our coordinate system (ecliptic is XZ plane).
    let position = DVec3::new(
        x_orb * px + y_orb * qx,
        x_orb * py + y_orb * qy,
        x_orb * pz + y_orb * qz,
    );

    let velocity = DVec3::new(
        vx_orb * px + vy_orb * qx,
        vx_orb * py + vy_orb * qy,
        vx_orb * pz + vy_orb * qz,
    );

    StateVector { position, velocity }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_orbit_speed() {
        let elements = OrbitalElements {
            semi_major_axis_m: AU_TO_METERS,
            eccentricity: 0.0,
            inclination_rad: 0.0,
            lon_ascending_node_rad: 0.0,
            arg_periapsis_rad: 0.0,
            true_anomaly_rad: 0.0,
        };
        let sun_gm = G * 1.989e30;
        let sv = orbital_elements_to_cartesian(&elements, sun_gm);

        // For a circular orbit, |v| should equal sqrt(GM/r).
        let expected_speed = (sun_gm / AU_TO_METERS).sqrt();
        let actual_speed = sv.velocity.length();
        let rel_error = (actual_speed - expected_speed).abs() / expected_speed;
        assert!(rel_error < 1e-10, "Speed error: {rel_error}");

        // Position should be at (r, 0, 0) for zero angles.
        let rel_pos_error = (sv.position.length() - AU_TO_METERS).abs() / AU_TO_METERS;
        assert!(rel_pos_error < 1e-10, "Position error: {rel_pos_error}");
    }

    #[test]
    fn test_hex_color() {
        let c = parse_hex_color("#FF8000");
        assert!((c[0] - 1.0).abs() < 0.01);
        assert!((c[1] - 0.502).abs() < 0.01);
        assert!((c[2] - 0.0).abs() < 0.01);
    }
}
