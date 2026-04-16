use bevy::prelude::*;

/// Marker — entity is a ship part.
#[derive(Component, Debug, Clone, Copy)]
pub struct Part;

#[derive(Component, Debug, Clone)]
pub struct CommandPod {
    pub model: String,
    pub diameter: f32,
    pub dry_mass: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Decoupler {
    pub ejection_impulse: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Adapter {
    pub target_diameter: f32,
}

#[derive(Component, Debug, Clone)]
pub struct FuelTank {
    pub length: f32,
    pub fuel_density: f32,
}

#[derive(Component, Debug, Clone)]
pub struct Engine {
    pub model: String,
    pub diameter: f32,
    pub thrust: f32,
    pub isp: f32,
}
