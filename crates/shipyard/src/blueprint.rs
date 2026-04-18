use crate::attach::{AttachNode, AttachNodes, Attachment, NodeId, Ship};
use crate::part::{Adapter, CommandPod, Decoupler, Engine, FuelTank, Part, ReactantRatio};
use crate::resource::{PartResources, Resource, ResourcePool};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "kind")]
pub enum PartData {
    CommandPod {
        model: String,
        diameter: f32,
        dry_mass: f32,
    },
    Decoupler {
        ejection_impulse: f32,
        dry_mass: f32,
    },
    Adapter {
        target_diameter: f32,
        dry_mass: f32,
    },
    FuelTank {
        length: f32,
        dry_mass: f32,
    },
    Engine {
        model: String,
        diameter: f32,
        thrust: f32,
        isp: f32,
        dry_mass: f32,
        reactants: Vec<ReactantRatio>,
        power_draw_kw: f32,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PartBlueprint {
    pub data: PartData,
    #[serde(default)]
    pub resources: HashMap<Resource, ResourcePool>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Connection {
    pub parent: usize,
    pub parent_node: String,
    pub child: usize,
    pub child_node: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ShipBlueprint {
    pub name: String,
    pub root: usize,
    pub parts: Vec<PartBlueprint>,
    pub connections: Vec<Connection>,
}

impl ShipBlueprint {
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(Into::into)
    }

    pub fn from_ron(s: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(s)
    }

    /// Spawn a single part into the world without attaching it or creating
    /// a `Ship` entity. Used by the editor to add parts incrementally.
    pub fn spawn_part(
        commands: &mut Commands,
        data: &PartData,
        resources: HashMap<Resource, ResourcePool>,
    ) -> Entity {
        let nodes = default_nodes_for(data);
        let mut ec = commands.spawn((
            Part,
            AttachNodes { nodes },
            PartResources { pools: resources },
        ));
        match data.clone() {
            PartData::CommandPod {
                model,
                diameter,
                dry_mass,
            } => {
                ec.insert(CommandPod {
                    model,
                    diameter,
                    dry_mass,
                });
            }
            PartData::Decoupler {
                ejection_impulse,
                dry_mass,
            } => {
                ec.insert(Decoupler {
                    ejection_impulse,
                    dry_mass,
                });
            }
            PartData::Adapter {
                target_diameter,
                dry_mass,
            } => {
                ec.insert(Adapter {
                    target_diameter,
                    dry_mass,
                });
            }
            PartData::FuelTank { length, dry_mass } => {
                ec.insert(FuelTank { length, dry_mass });
            }
            PartData::Engine {
                model,
                diameter,
                thrust,
                isp,
                dry_mass,
                reactants,
                power_draw_kw,
            } => {
                ec.insert(Engine {
                    model,
                    diameter,
                    thrust,
                    isp,
                    dry_mass,
                    reactants,
                    power_draw_kw,
                });
            }
        }
        ec.id()
    }

    /// Spawn the blueprint into the world, returning the `Ship` entity.
    pub fn spawn(&self, commands: &mut Commands) -> Entity {
        let ids: Vec<Entity> = (0..self.parts.len())
            .map(|_| commands.spawn_empty().id())
            .collect();

        for (i, pb) in self.parts.iter().enumerate() {
            let e = ids[i];
            let nodes = default_nodes_for(&pb.data);
            let mut ec = commands.entity(e);
            ec.insert((
                Part,
                AttachNodes { nodes },
                PartResources {
                    pools: pb.resources.clone(),
                },
            ));
            match pb.data.clone() {
                PartData::CommandPod {
                    model,
                    diameter,
                    dry_mass,
                } => {
                    ec.insert(CommandPod {
                        model,
                        diameter,
                        dry_mass,
                    });
                }
                PartData::Decoupler {
                    ejection_impulse,
                    dry_mass,
                } => {
                    ec.insert(Decoupler {
                        ejection_impulse,
                        dry_mass,
                    });
                }
                PartData::Adapter {
                    target_diameter,
                    dry_mass,
                } => {
                    ec.insert(Adapter {
                        target_diameter,
                        dry_mass,
                    });
                }
                PartData::FuelTank { length, dry_mass } => {
                    ec.insert(FuelTank { length, dry_mass });
                }
                PartData::Engine {
                    model,
                    diameter,
                    thrust,
                    isp,
                    dry_mass,
                    reactants,
                    power_draw_kw,
                } => {
                    ec.insert(Engine {
                        model,
                        diameter,
                        thrust,
                        isp,
                        dry_mass,
                        reactants,
                        power_draw_kw,
                    });
                }
            }
        }

        for c in &self.connections {
            commands.entity(ids[c.child]).insert(Attachment {
                parent: ids[c.parent],
                parent_node: c.parent_node.clone(),
                my_node: c.child_node.clone(),
            });
        }

        let root = ids[self.root];
        commands
            .spawn(Ship {
                name: self.name.clone(),
                root,
            })
            .id()
    }
}

/// Default resource pools a freshly-spawned part should carry.
///
/// - [`PartData::CommandPod`] ships with a small electricity reserve scaled
///   to its diameter (onboard battery for avionics + life support).
/// - [`PartData::FuelTank`] spawns pre-filled with a methalox mix at the
///   Raptor-style O/F ≈ 3.6 mass ratio, sized from the tank's length.
///   Volumes scale with `length`: 1000 L of CH4 per metre + a matching
///   1331 L of LOX.
/// - Other parts spawn dry.
pub fn default_resources_for(data: &PartData) -> HashMap<Resource, ResourcePool> {
    let mut pools = HashMap::new();
    match data {
        PartData::CommandPod { diameter, .. } => {
            // 1 kWh per metre of diameter — small but visible.
            let capacity = diameter.max(0.5);
            pools.insert(
                Resource::Electricity,
                ResourcePool {
                    capacity,
                    amount: capacity,
                },
            );
        }
        PartData::FuelTank { length, .. } => {
            let ch4 = (*length).max(0.5) * 1000.0;
            // Volume of LOX that carries 3.6× the mass of `ch4` litres of
            // methane, given the two canonical densities.
            let lox = ch4
                * 3.6
                * Resource::Methane.density_kg_per_unit() as f32
                / Resource::Lox.density_kg_per_unit() as f32;
            pools.insert(
                Resource::Methane,
                ResourcePool {
                    capacity: ch4,
                    amount: ch4,
                },
            );
            pools.insert(
                Resource::Lox,
                ResourcePool {
                    capacity: lox,
                    amount: lox,
                },
            );
        }
        _ => {}
    }
    pools
}

/// Build the initial attach-node layout for a part. Parametric parts are
/// populated with a placeholder diameter here; `sizing::propagate_node_sizes`
/// overwrites them once the part is attached to a parent.
pub fn default_nodes_for(data: &PartData) -> HashMap<NodeId, AttachNode> {
    let mut nodes = HashMap::new();
    match data {
        PartData::CommandPod { diameter, .. } => {
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: *diameter,
                    offset: Vec3::new(0.0, -diameter * 0.9, 0.0),
                },
            );
        }
        PartData::Decoupler { .. } => {
            nodes.insert(
                "top".into(),
                AttachNode {
                    diameter: 1.0,
                    offset: Vec3::ZERO,
                },
            );
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: 1.0,
                    offset: Vec3::new(0.0, -0.2, 0.0),
                },
            );
        }
        PartData::Adapter { target_diameter, .. } => {
            nodes.insert(
                "top".into(),
                AttachNode {
                    diameter: 1.0,
                    offset: Vec3::ZERO,
                },
            );
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: *target_diameter,
                    offset: Vec3::new(0.0, -0.5, 0.0),
                },
            );
        }
        PartData::FuelTank { length, .. } => {
            nodes.insert(
                "top".into(),
                AttachNode {
                    diameter: 1.0,
                    offset: Vec3::ZERO,
                },
            );
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: 1.0,
                    offset: Vec3::new(0.0, -*length, 0.0),
                },
            );
        }
        PartData::Engine { diameter, .. } => {
            nodes.insert(
                "top".into(),
                AttachNode {
                    diameter: *diameter,
                    offset: Vec3::ZERO,
                },
            );
            nodes.insert(
                "bottom".into(),
                AttachNode {
                    diameter: *diameter,
                    offset: Vec3::new(0.0, -*diameter * 0.9, 0.0),
                },
            );
        }
    }
    nodes
}
