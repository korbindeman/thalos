//! Procedural ship-part rendering.
//!
//! A single [`ShipPartMaterial`] — an [`bevy::pbr::ExtendedMaterial`] of
//! [`bevy::pbr::StandardMaterial`] — draws stainless-steel tank-like parts
//! with procedural panel seams and rivets. Lighting, shadows, and tone
//! mapping all come from the base PBR pipeline; the extension only adds
//! per-fragment albedo / roughness modulation and a normal-map
//! perturbation computed from a procedural surface-relief height field.
//!
//! The crate is intentionally thin: one asset type, one shader. Callers
//! are responsible for adding the plugin, creating per-part material
//! instances, and driving `tint` / dimensions from their own systems.

mod material;

pub use material::{ShipPartExtension, ShipPartMaterial, ShipPartParams, stainless_steel_base};

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

pub struct ShipRenderingPlugin;

impl Plugin for ShipRenderingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<ShipPartMaterial>::default());
    }
}
