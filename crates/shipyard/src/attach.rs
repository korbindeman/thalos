use bevy::prelude::*;
use std::collections::HashMap;

pub type NodeId = String;

#[derive(Clone, Debug)]
pub struct AttachNode {
    pub diameter: f32,
    pub offset: Vec3,
}

#[derive(Component, Default, Debug, Clone)]
pub struct AttachNodes {
    pub nodes: HashMap<NodeId, AttachNode>,
}

impl AttachNodes {
    pub fn get(&self, id: &str) -> Option<&AttachNode> {
        self.nodes.get(id)
    }

    pub fn set(&mut self, id: impl Into<NodeId>, node: AttachNode) {
        self.nodes.insert(id.into(), node);
    }
}

/// This entity is attached to `parent` — `my_node` mates with `parent_node`.
#[derive(Component, Debug, Clone)]
pub struct Attachment {
    pub parent: Entity,
    pub parent_node: NodeId,
    pub my_node: NodeId,
}

/// Root of a ship assembly.
#[derive(Component, Debug, Clone)]
pub struct Ship {
    pub name: String,
    pub root: Entity,
}
