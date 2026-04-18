//! Builds a small methalox ship blueprint, serializes to RON, deserializes,
//! spawns into a headless Bevy world, runs one update (propagating node
//! sizes), and asserts the parametric parts adopted the right diameters.
//! Also prints aggregated `ShipStats` so the numbers are easy to eyeball.

use bevy::prelude::*;
use std::collections::HashMap;
use thalos_shipyard::Resource as ShipResource;
use thalos_shipyard::*;

fn methalox_reactants() -> Vec<ReactantRatio> {
    vec![
        ReactantRatio {
            resource: ShipResource::Methane,
            mass_fraction: 1.0 / 4.6,
        },
        ReactantRatio {
            resource: ShipResource::Lox,
            mass_fraction: 3.6 / 4.6,
        },
    ]
}

fn single_pool(resource: ShipResource, amount: f32) -> HashMap<ShipResource, ResourcePool> {
    let mut m = HashMap::new();
    m.insert(
        resource,
        ResourcePool {
            capacity: amount,
            amount,
        },
    );
    m
}

fn main() {
    let pod = PartData::CommandPod {
        model: "Mk1".into(),
        diameter: 1.25,
        dry_mass: 840.0,
    };
    let blueprint = ShipBlueprint {
        name: "TestRocket".into(),
        root: 0,
        parts: vec![
            PartBlueprint {
                resources: blueprint::default_resources_for(&pod),
                data: pod,
            },
            PartBlueprint {
                data: PartData::Decoupler {
                    ejection_impulse: 250.0,
                    dry_mass: 50.0,
                },
                resources: HashMap::new(),
            },
            PartBlueprint {
                data: PartData::Adapter {
                    target_diameter: 2.5,
                    dry_mass: 100.0,
                },
                resources: HashMap::new(),
            },
            PartBlueprint {
                data: PartData::FuelTank {
                    length: 4.0,
                    dry_mass: 1000.0,
                },
                resources: single_pool(ShipResource::Methane, 8_000.0),
            },
            PartBlueprint {
                data: PartData::FuelTank {
                    length: 4.0,
                    dry_mass: 1200.0,
                },
                resources: single_pool(ShipResource::Lox, 10_800.0),
            },
            PartBlueprint {
                data: PartData::Engine {
                    model: "Raptor-2".into(),
                    diameter: 2.5,
                    thrust: 2_300_000.0,
                    isp: 330.0,
                    dry_mass: 1500.0,
                    reactants: methalox_reactants(),
                    power_draw_kw: 0.0,
                },
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
            Connection {
                parent: 4,
                parent_node: "bottom".into(),
                child: 5,
                child_node: "top".into(),
            },
        ],
    };

    let ron = blueprint.to_ron().expect("serialize");
    println!("--- ship.ron ---\n{ron}\n");

    let reloaded = ShipBlueprint::from_ron(&ron).expect("deserialize");
    assert_eq!(reloaded.parts.len(), blueprint.parts.len());

    let stats = reloaded.stats();
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
        println!(
            "    {}: {:.1}%",
            res.display_name(),
            frac * 100.0,
        );
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

    app.world_mut()
        .commands()
        .queue(move |world: &mut World| {
            let mut commands = world.commands();
            reloaded.spawn(&mut commands);
        });

    app.update();

    let world = app.world_mut();
    let mut q = world.query::<(&AttachNodes, Option<&Decoupler>, Option<&FuelTank>, Option<&Adapter>)>();
    for (nodes, dec, tank, adapter) in q.iter(world) {
        if dec.is_some() {
            assert_eq!(nodes.get("top").unwrap().diameter, 1.25);
            assert_eq!(nodes.get("bottom").unwrap().diameter, 1.25);
            println!("decoupler sized to 1.25 OK");
        }
        if adapter.is_some() {
            assert_eq!(nodes.get("top").unwrap().diameter, 1.25);
            assert_eq!(nodes.get("bottom").unwrap().diameter, 2.5);
            println!("adapter 1.25 -> 2.5 OK");
        }
        if tank.is_some() {
            assert_eq!(nodes.get("top").unwrap().diameter, 2.5);
            assert_eq!(nodes.get("bottom").unwrap().diameter, 2.5);
            println!("tank sized to 2.5 OK");
        }
    }

    println!("roundtrip OK");
}
