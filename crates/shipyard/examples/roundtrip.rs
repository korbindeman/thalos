//! Builds a small Argos+Zephyr methalox ship from the parts catalog,
//! serializes to RON, deserializes, spawns into a headless Bevy world,
//! runs one update (propagating node sizes), and asserts the parametric
//! parts adopted the right diameters. Also prints aggregated `ShipStats`
//! so the numbers are easy to eyeball.

use bevy::prelude::*;
use std::collections::HashMap;
use thalos_shipyard::*;

fn main() {
    let catalog =
        PartCatalog::load_from_path("assets/parts.ron").expect("load assets/parts.ron");

    let blueprint = ShipBlueprint {
        name: "TestRocket".into(),
        root: 0,
        parts: vec![
            PartBlueprint {
                catalog_id: "argos".into(),
                params: PartParams::None,
                resources: HashMap::new(),
            },
            PartBlueprint {
                catalog_id: "decoupler_std".into(),
                params: PartParams::Decoupler { diameter: 2.5 },
                resources: HashMap::new(),
            },
            PartBlueprint {
                catalog_id: "adapter_std".into(),
                params: PartParams::Adapter {
                    diameter: 2.5,
                    target_diameter: 4.0,
                },
                resources: HashMap::new(),
            },
            PartBlueprint {
                catalog_id: "tank_methalox".into(),
                params: PartParams::Tank {
                    diameter: 4.0,
                    length: 4.0,
                },
                resources: HashMap::new(),
            },
            PartBlueprint {
                catalog_id: "boreas".into(),
                params: PartParams::None,
                resources: HashMap::new(),
            },
        ],
        connections: vec![
            Connection {
                parent: 0,
                parent_node: "bottom".into(),
                child: 1,
                child_node: "top".into(),
            },
            Connection {
                parent: 1,
                parent_node: "bottom".into(),
                child: 2,
                child_node: "top".into(),
            },
            Connection {
                parent: 2,
                parent_node: "bottom".into(),
                child: 3,
                child_node: "top".into(),
            },
            Connection {
                parent: 3,
                parent_node: "bottom".into(),
                child: 4,
                child_node: "top".into(),
            },
        ],
    };

    let ron = blueprint.to_ron().expect("serialize");
    println!("--- ship.ron ---\n{ron}\n");

    let reloaded = ShipBlueprint::from_ron(&ron).expect("deserialize");
    assert_eq!(reloaded.parts.len(), blueprint.parts.len());

    let stats = reloaded.stats(&catalog).expect("stats");
    println!("--- ship stats ---");
    println!("  dry mass:     {:.0} kg", stats.dry_mass_kg);
    println!("  propellant:   {:.0} kg", stats.propellant_mass_kg);
    println!("  wet mass:     {:.0} kg", stats.wet_mass_kg());
    println!("  thrust:       {:.1} kN", stats.total_thrust_n / 1000.0);
    println!("  isp:          {:.0} s", stats.combined_isp_s);
    println!("  mdot:         {:.1} kg/s", stats.mass_flow_kg_per_s);
    println!("  initial a:    {:.2} m/s²", stats.current_acceleration());
    if let Some(burn) = stats.burn_time_at_full_throttle_s() {
        println!("  burn time:    {:.1} s", burn);
    }
    println!("  Δv capacity:  {:.0} m/s", stats.delta_v_capacity());
    println!("  reactant mix:");
    for (res, frac) in &stats.reactant_fractions {
        println!("    {}: {:.1}%", res.display_name(), frac * 100.0);
    }
    println!("  resources on board:");
    let mut res_list: Vec<_> = stats.resources.iter().collect();
    res_list.sort_by_key(|(r, _)| r.display_name());
    for (res, totals) in res_list {
        println!(
            "    {}: {:.0}/{:.0} {} ({:.0} kg)",
            res.display_name(),
            totals.amount,
            totals.capacity,
            res.unit_label(),
            totals.mass_kg,
        );
    }
    println!();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(ShipyardPlugin);
    app.insert_resource(catalog.clone());

    app.world_mut().commands().queue(move |world: &mut World| {
        let cat = world.resource::<PartCatalog>().clone();
        let mut commands = world.commands();
        reloaded.spawn(&mut commands, &cat).expect("spawn");
    });

    app.update();

    let world = app.world_mut();
    let mut q = world.query::<(
        &AttachNodes,
        Option<&Decoupler>,
        Option<&FuelTank>,
        Option<&Adapter>,
    )>();
    for (nodes, dec, tank, adapter) in q.iter(world) {
        if dec.is_some() {
            assert_eq!(nodes.get("top").unwrap().diameter, 2.5);
            assert_eq!(nodes.get("bottom").unwrap().diameter, 2.5);
            println!("decoupler sized to 2.5 OK");
        }
        if adapter.is_some() {
            assert_eq!(nodes.get("top").unwrap().diameter, 2.5);
            assert_eq!(nodes.get("bottom").unwrap().diameter, 4.0);
            println!("adapter 2.5 -> 4.0 OK");
        }
        if tank.is_some() {
            assert_eq!(nodes.get("top").unwrap().diameter, 4.0);
            assert_eq!(nodes.get("bottom").unwrap().diameter, 4.0);
            println!("tank sized to 4.0 OK");
        }
    }

    println!("roundtrip OK");
}
