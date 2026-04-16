use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generic resource pool: monopropellant, liquid fuel, electric charge, etc.
/// `density` is kg per unit so total mass can be computed without hard-coding
/// resource kinds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourcePool {
    pub capacity: f32,
    pub amount: f32,
    pub density: f32,
}

#[derive(Component, Default, Clone, Debug)]
pub struct PartResources {
    pub pools: HashMap<String, ResourcePool>,
}

impl PartResources {
    pub fn insert(&mut self, name: impl Into<String>, pool: ResourcePool) {
        self.pools.insert(name.into(), pool);
    }

    pub fn get(&self, name: &str) -> Option<&ResourcePool> {
        self.pools.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut ResourcePool> {
        self.pools.get_mut(name)
    }
}
