use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Canonical resources the ship system understands.
///
/// Each variant carries its own physical properties (density, units) via
/// methods — a pool that claims to hold [`Resource::Lox`] always uses
/// LOX's density, period. This keeps blueprints honest; a fuel tank
/// cannot drift from the physics model by declaring its own density.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Resource {
    /// Liquid methane (CH4), fuel. Stored cryogenic.
    Methane,
    /// Liquid oxygen (O2), oxidizer. Stored cryogenic.
    Lox,
    /// Electric charge. Not mass-bearing (for ship dynamics).
    Electricity,
}

impl Resource {
    /// Density in kg per native unit. For fluids (`Methane`, `Lox`) the
    /// native unit is litre (kg/L ≡ g/cm³). [`Resource::Electricity`] is
    /// dimensionally power × time, not matter, so returns 0 — it never
    /// contributes to wet mass.
    pub fn density_kg_per_unit(self) -> f64 {
        match self {
            Resource::Methane => 0.422, // LCH4 at ≈112 K
            Resource::Lox => 1.141,     // LOX at ≈90 K
            Resource::Electricity => 0.0,
        }
    }

    /// True if amount × density is a meaningful mass contribution.
    pub fn is_mass_bearing(self) -> bool {
        self.density_kg_per_unit() > 0.0
    }

    /// Label for the native unit of `amount` / `capacity`.
    pub fn unit_label(self) -> &'static str {
        match self {
            Resource::Methane | Resource::Lox => "L",
            Resource::Electricity => "kWh",
        }
    }

    /// Human display name.
    pub fn display_name(self) -> &'static str {
        match self {
            Resource::Methane => "Methane",
            Resource::Lox => "LOX",
            Resource::Electricity => "Electricity",
        }
    }
}

/// A stored quantity of a single [`Resource`]. Density is not stored here —
/// it is always taken from the [`Resource`] itself.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ResourcePool {
    pub capacity: f32,
    pub amount: f32,
}

impl ResourcePool {
    pub fn mass_kg(&self, resource: Resource) -> f64 {
        self.amount as f64 * resource.density_kg_per_unit()
    }

    pub fn capacity_mass_kg(&self, resource: Resource) -> f64 {
        self.capacity as f64 * resource.density_kg_per_unit()
    }
}

#[derive(Component, Default, Clone, Debug)]
pub struct PartResources {
    pub pools: HashMap<Resource, ResourcePool>,
}

impl PartResources {
    pub fn insert(&mut self, resource: Resource, pool: ResourcePool) {
        self.pools.insert(resource, pool);
    }

    pub fn get(&self, resource: Resource) -> Option<&ResourcePool> {
        self.pools.get(&resource)
    }

    pub fn get_mut(&mut self, resource: Resource) -> Option<&mut ResourcePool> {
        self.pools.get_mut(&resource)
    }
}
