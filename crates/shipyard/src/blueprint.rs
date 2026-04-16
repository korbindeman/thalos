use crate::attach::{AttachNode, AttachNodes, Attachment, NodeId, Ship};
use crate::part::{Adapter, CommandPod, Decoupler, Engine, FuelTank, Part};
use crate::resource::{PartResources, ResourcePool};
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
    },
    Adapter {
        target_diameter: f32,
    },
    FuelTank {
        length: f32,
        fuel_density: f32,
    },
    Engine {
        model: String,
        diameter: f32,
        thrust: f32,
        isp: f32,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PartBlueprint {
    pub data: PartData,
    #[serde(default)]
    pub resources: HashMap<String, ResourcePool>,
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
        resources: HashMap<String, ResourcePool>,
    ) -> Entity {
        let nodes = default_nodes_for(data);
        let mut ec = commands.spawn((
            Part,
            AttachNodes { nodes },
            PartResources { pools: resources },
            Transform::default(),
            Visibility::default(),
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
            PartData::Decoupler { ejection_impulse } => {
                ec.insert(Decoupler { ejection_impulse });
            }
            PartData::Adapter { target_diameter } => {
                ec.insert(Adapter { target_diameter });
            }
            PartData::FuelTank {
                length,
                fuel_density,
            } => {
                ec.insert(FuelTank {
                    length,
                    fuel_density,
                });
            }
            PartData::Engine {
                model,
                diameter,
                thrust,
                isp,
            } => {
                ec.insert(Engine {
                    model,
                    diameter,
                    thrust,
                    isp,
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
                PartData::Decoupler { ejection_impulse } => {
                    ec.insert(Decoupler { ejection_impulse });
                }
                PartData::Adapter { target_diameter } => {
                    ec.insert(Adapter { target_diameter });
                }
                PartData::FuelTank {
                    length,
                    fuel_density,
                } => {
                    ec.insert(FuelTank {
                        length,
                        fuel_density,
                    });
                }
                PartData::Engine {
                    model,
                    diameter,
                    thrust,
                    isp,
                } => {
                    ec.insert(Engine {
                        model,
                        diameter,
                        thrust,
                        isp,
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
        PartData::Adapter { target_diameter } => {
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
