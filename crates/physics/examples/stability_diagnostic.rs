//! Stability diagnostic report for the solar system definition.
//!
//! Loads `assets/solar_system.ron` and prints Hill-sphere analyses, planet
//! spacing, moon overlap checks, and period-ratio near-resonances. This is a
//! design-time tool, not a test — it asserts nothing.
//!
//! Run from the workspace root:
//!     cargo run -p thalos_physics --example stability_diagnostic

use thalos_physics::parsing::load_solar_system;
use thalos_physics::types::{AU_TO_METERS, BodyDefinition, BodyKind};

fn main() {
    let ron_src =
        std::fs::read_to_string("../../assets/solar_system.ron").expect("run from workspace root");
    let system = load_solar_system(&ron_src).unwrap();

    let star = system
        .bodies
        .iter()
        .find(|b| b.kind == BodyKind::Star)
        .unwrap();

    println!("\n═══ HILL SPHERE ANALYSIS ═══\n");

    // For each body with a parent, compute Hill sphere and check moons.
    for body in &system.bodies {
        if body.parent.is_none() || body.kind == BodyKind::Star {
            continue;
        }
        let parent = &system.bodies[body.parent.unwrap()];
        let elems = body.orbital_elements.as_ref().unwrap();
        let a = elems.semi_major_axis_m;
        let e = elems.eccentricity;

        // Hill sphere: r_H = a * (m / (3 * M_parent))^(1/3)
        let _hill_radius = a * (body.mass_kg / (3.0 * parent.mass_kg)).powf(1.0 / 3.0);

        // For moons, check if they're within their parent's Hill sphere
        if parent.kind != BodyKind::Star {
            // This body is a moon — check against parent's Hill sphere
            let grandparent = &system.bodies[parent.parent.unwrap()];
            let parent_elems = parent.orbital_elements.as_ref().unwrap();
            let parent_a = parent_elems.semi_major_axis_m;
            let parent_hill =
                parent_a * (parent.mass_kg / (3.0 * grandparent.mass_kg)).powf(1.0 / 3.0);

            let _periapsis = a * (1.0 - e);
            let apoapsis = a * (1.0 + e);
            let ratio = apoapsis / parent_hill;
            let is_retrograde = elems.inclination_rad > std::f64::consts::FRAC_PI_2;
            let limit = if is_retrograde { 0.50 } else { 0.33 };

            let status = if ratio > limit {
                "UNSTABLE"
            } else if ratio > limit * 0.85 {
                "MARGINAL"
            } else {
                "OK"
            };

            println!(
                "  {:<12} orbits {:<12}  a={:.2e} m  e={:.3}  apo={:.2e} m  Hill={:.2e} m  apo/Hill={:.3}  limit={:.2}  {}",
                body.name, parent.name, a, e, apoapsis, parent_hill, ratio, limit, status
            );
        }
    }

    println!("\n═══ PLANET SPACING (mutual Hill radii) ═══\n");

    // Collect planets (direct children of star) sorted by semi-major axis.
    let mut planets: Vec<&BodyDefinition> = system
        .bodies
        .iter()
        .filter(|b| b.parent == Some(star.id) && b.orbital_elements.is_some())
        .collect();
    planets.sort_by(|a, b| {
        let a_sma = a.orbital_elements.as_ref().unwrap().semi_major_axis_m;
        let b_sma = b.orbital_elements.as_ref().unwrap().semi_major_axis_m;
        a_sma.partial_cmp(&b_sma).unwrap()
    });

    for w in planets.windows(2) {
        let inner = w[0];
        let outer = w[1];
        let a1 = inner.orbital_elements.as_ref().unwrap().semi_major_axis_m;
        let e1 = inner.orbital_elements.as_ref().unwrap().eccentricity;
        let a2 = outer.orbital_elements.as_ref().unwrap().semi_major_axis_m;
        let e2 = outer.orbital_elements.as_ref().unwrap().eccentricity;

        let mutual_hill = ((a1 + a2) / 2.0)
            * ((inner.mass_kg + outer.mass_kg) / (3.0 * star.mass_kg)).powf(1.0 / 3.0);
        let separation = a2 - a1;
        let n_hill = separation / mutual_hill;

        let apo1 = a1 * (1.0 + e1);
        let peri2 = a2 * (1.0 - e2);
        let clearance = peri2 - apo1;

        let status = if clearance < 0.0 {
            "ORBIT CROSSING"
        } else if n_hill < 8.0 {
            "UNSTABLE (<8)"
        } else if n_hill < 10.0 {
            "MARGINAL (<10)"
        } else {
            "OK"
        };

        println!(
            "  {:<12} — {:<12}  Δa={:.3} AU  {:.1} mutual Hill radii  clearance={:.4} AU  {}",
            inner.name,
            outer.name,
            separation / AU_TO_METERS,
            n_hill,
            clearance / AU_TO_METERS,
            status
        );
    }

    println!("\n═══ MOON ORBIT OVERLAP CHECK ═══\n");

    // For each planet, check if its moons' orbits overlap.
    for planet in &planets {
        let moons: Vec<&BodyDefinition> = system
            .bodies
            .iter()
            .filter(|b| b.parent == Some(planet.id) && b.orbital_elements.is_some())
            .collect();
        if moons.len() < 2 {
            continue;
        }
        let mut sorted_moons = moons.clone();
        sorted_moons.sort_by(|a, b| {
            let a_sma = a.orbital_elements.as_ref().unwrap().semi_major_axis_m;
            let b_sma = b.orbital_elements.as_ref().unwrap().semi_major_axis_m;
            a_sma.partial_cmp(&b_sma).unwrap()
        });

        for w in sorted_moons.windows(2) {
            let inner = w[0];
            let outer = w[1];
            let ie = inner.orbital_elements.as_ref().unwrap();
            let oe = outer.orbital_elements.as_ref().unwrap();
            let apo_inner = ie.semi_major_axis_m * (1.0 + ie.eccentricity);
            let peri_outer = oe.semi_major_axis_m * (1.0 - oe.eccentricity);

            let status = if peri_outer < apo_inner {
                "ORBITS OVERLAP"
            } else {
                "OK"
            };

            println!(
                "  {:<12}  apo={:.2e} m  |  {:<12}  peri={:.2e} m  gap={:.2e} m  {}",
                inner.name,
                apo_inner,
                outer.name,
                peri_outer,
                peri_outer - apo_inner,
                status
            );
        }
    }

    println!("\n═══ PERIOD RATIOS (near-resonances) ═══\n");

    for planet in &planets {
        let moons: Vec<&BodyDefinition> = system
            .bodies
            .iter()
            .filter(|b| b.parent == Some(planet.id) && b.orbital_elements.is_some())
            .collect();
        if moons.len() < 2 {
            continue;
        }
        let mut sorted_moons = moons.clone();
        sorted_moons.sort_by(|a, b| {
            let a_sma = a.orbital_elements.as_ref().unwrap().semi_major_axis_m;
            let b_sma = b.orbital_elements.as_ref().unwrap().semi_major_axis_m;
            a_sma.partial_cmp(&b_sma).unwrap()
        });

        for w in sorted_moons.windows(2) {
            let inner = w[0];
            let outer = w[1];
            let a1 = inner.orbital_elements.as_ref().unwrap().semi_major_axis_m;
            let a2 = outer.orbital_elements.as_ref().unwrap().semi_major_axis_m;
            let period_ratio = (a2 / a1).powf(1.5);

            // Check for near-integer ratios
            let nearest_resonance = (period_ratio * 2.0).round() / 2.0;
            let resonance_error = (period_ratio - nearest_resonance).abs();

            if resonance_error < 0.05 {
                println!(
                    "  {:<12} / {:<12}  P_ratio={:.3}  ≈ {:.1}:1  (Δ={:.4})",
                    outer.name, inner.name, period_ratio, nearest_resonance, resonance_error
                );
            }
        }
    }
}
