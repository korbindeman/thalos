use bevy::prelude::*;

/// GPU textures consumed by a single [`crate::PlanetMaterial`].
pub struct PlanetTextures {
    pub albedo: Handle<Image>,
    pub height: Handle<Image>,
}
