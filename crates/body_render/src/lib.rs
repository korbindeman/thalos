//! Unified celestial-body rendering: one appearance model, two backends.
//! `shading` = shared lighting/atmosphere/BRDF libraries + uniforms;
//! `impostor` = distant billboard materials; `ground` = udlod terrain LOD.
pub mod ground;
pub mod impostor;
pub mod shading;

pub use ground::*;
pub use impostor::*;
pub use shading::*;

/// The vendored UDLOD terrain renderer, re-exported so the rest of the
/// workspace depends on it *through* `body_render` — its single consumer —
/// rather than importing `thalos_udlod` directly. Reach tile/atlas/precision
/// handles via `thalos_body_render::udlod::{prelude, math, big_space}`.
pub use thalos_udlod as udlod;

use bevy::prelude::*;

/// Adds the full body-render stack: shared shader libraries, impostor
/// materials, and the udlod ground-terrain pipeline.
pub struct BodyRenderPlugin;
impl Plugin for BodyRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(shading::PlanetLightingPlugin);
        app.add_plugins(impostor::PlanetRenderingPlugin);
        app.add_plugins(ground::ThalosTerrainPlugin);
    }
}
