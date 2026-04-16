//! Builds a small ship blueprint, serializes to RON, deserializes, spawns
//! into a headless Bevy world, runs one update (propagating node sizes),
//! and asserts the parametric parts adopted the right diameters.

use bevy::prelude::*;
use std::collections::HashMap;
use thalos_shipyard::*;

fn main() {
    let blueprint = ShipBlueprint {
        name: "TestRocket".into(),
        root: 0,
        parts: vec![
            PartBlueprint {
                data: PartData::CommandPod {
                    model: "Mk1".into(),
                    diameter: 1.25,
                    dry_mass: 840.0,
                },
                resources: {
                    let mut r = HashMap::new();
                    r.insert(
                        "monopropellant".into(),
                        ResourcePool {
                            capacity: 50.0,
                            amount: 50.0,
                            density: 4.0,
                        },
                    );
                    r
                },
            },
            PartBlueprint {
                data: PartData::Decoupler {
                    ejection_impulse: 250.0,
                },
                resources: HashMap::new(),
            },
            PartBlueprint {
                data: PartData::Adapter {
                    target_diameter: 2.5,
                },
                resources: HashMap::new(),
            },
            PartBlueprint {
                data: PartData::FuelTank {
                    length: 4.0,
                    fuel_density: 5.0,
                },
                resources: {
                    let mut r = HashMap::new();
                    r.insert(
                        "liquid_fuel".into(),
                        ResourcePool {
                            capacity: 1800.0,
                            amount: 1800.0,
                            density: 5.0,
                        },
                    );
                    r
                },
            },
            PartBlueprint {
                data: PartData::Engine {
                    model: "LV-T45".into(),
                    diameter: 2.5,
                    thrust: 215_000.0,
                    isp: 320.0,
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
        ],
    };

    let ron = blueprint.to_ron().expect("serialize");
    println!("--- ship.ron ---\n{ron}\n");

    let reloaded = ShipBlueprint::from_ron(&ron).expect("deserialize");
    assert_eq!(reloaded.parts.len(), blueprint.parts.len());

    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(ShipyardPlugin);

    app.world_mut()
        .commands()
        .queue(move |world: &mut World| {
            let mut commands = world.commands();
            reloaded.spawn(&mut commands);
        });

    app.update();

    // Verify: decoupler, adapter top, fuel tank both sides adopted parent diameters.
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
