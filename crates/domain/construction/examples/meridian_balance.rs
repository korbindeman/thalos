//! Ground-stance readout for the shipped Meridian: CoM, gear geometry, static
//! wheel-load split, and the roll-over threshold (effective half-track at the
//! CoM over CoM height) — the number that says whether the craft skids before
//! it tips (healthy: threshold ≈ or above the tire μ of 0.8). Run from the
//! workspace root:
//! `cargo run -p thalos_shipyard --example meridian_balance`

use thalos_shipyard::{CatalogEntry, PartCatalog, PartParams, ShipBlueprint};

fn main() {
    let cat = PartCatalog::load_from_str(include_str!("../../../../assets/parts.ron"))
        .expect("parse parts.ron");
    let bp = ShipBlueprint::from_ron(include_str!("../../../../ships/meridian.ron"))
        .expect("parse meridian.ron");
    let s = bp.stats(&cat).expect("stats");

    println!("dry mass      {:>10.1} kg", s.dry_mass_kg);
    println!("propellant    {:>10.1} kg", s.propellant_mass_kg);
    println!(
        "total         {:>10.1} kg",
        s.dry_mass_kg + s.propellant_mass_kg
    );
    println!(
        "CoM (body: X=right Y=nose Z=dorsal)  ({:.3}, {:.3}, {:.3}) m",
        s.center_of_mass_m.x, s.center_of_mass_m.y, s.center_of_mass_m.z
    );
    println!(
        "MoI           ({:.0}, {:.0}, {:.0}) kg·m²",
        s.moment_of_inertia_kg_m2.x, s.moment_of_inertia_kg_m2.y, s.moment_of_inertia_kg_m2.z
    );

    // Gear geometry off the fuselage-mounted gearboxes (BodySkin: y = -station × length).
    let mut nose_y = None;
    let mut main_y = None;
    let mut main_drop = 0.0_f64;
    let mut track_half = 0.0_f64;
    let mut fuselage_half_height = 0.0_f64;
    for m in &bp.surface_mounts {
        let part = &bp.parts[m.child];
        let Ok(entry) = cat.resolve(&part.catalog_id) else {
            continue;
        };
        let CatalogEntry::Gear(g) = entry else {
            continue;
        };
        let PartParams::Fuselage {
            length,
            max_width,
            max_height,
            ..
        } = &bp.parts[m.parent].params
        else {
            continue;
        };
        fuselage_half_height = *max_height as f64 * 0.5;
        let y = -(m.station as f64) * (*length as f64);
        let PartParams::Gear {
            strut_length,
            wheel_radius,
        } = &part.params
        else {
            continue;
        };
        let drop = (*strut_length + *wheel_radius) as f64;
        if g.track_fraction > 0.0 {
            main_y = Some(y);
            main_drop = drop;
            track_half = (g.track_fraction * max_width * 0.5) as f64;
        } else {
            nose_y = Some(y);
        }
        println!(
            "{:<14} station={:.3} -> y={:+.2} m  drop={:.2} m  track=±{:.2} m",
            part.catalog_id,
            m.station,
            y,
            drop,
            (g.track_fraction * max_width * 0.5)
        );
    }

    let (Some(nose_y), Some(main_y)) = (nose_y, main_y) else {
        println!("(no tricycle gear pair found — stance readout skipped)");
        return;
    };
    let com = s.center_of_mass_m;
    let wheelbase = nose_y - main_y; // y decreases aft, so this is positive
    let nose_share = (com.y - main_y) / wheelbase;
    let com_frac = (nose_y - com.y) / wheelbase; // 0 at nose wheel, 1 at mains
    let eff_half_track = track_half * com_frac;
    let com_height = com.z + fuselage_half_height + main_drop;
    println!("wheelbase     {wheelbase:>10.2} m");
    println!(
        "load split    {:>9.1} % nose / {:.1} % mains",
        nose_share * 100.0,
        (1.0 - nose_share) * 100.0
    );
    println!(
        "CoM->mains margin {:>6.2} m (must stay > 0 or it sits on its tail)",
        com.y - main_y
    );
    println!("effective half-track at CoM {eff_half_track:>5.2} m, CoM height {com_height:.2} m");
    println!(
        "roll-over threshold {:>5.2} g lateral (tire grip mu = 0.8: {})",
        eff_half_track / com_height,
        if eff_half_track / com_height >= 0.75 {
            "skids before it tips"
        } else {
            "TIPS BEFORE IT SKIDS"
        }
    );
}
