//! Unified celestial-body rendering: one appearance model, two backends.
//! `shading` = shared lighting/atmosphere/BRDF libraries + uniforms;
//! `impostor` = distant billboard materials; `ground` = udlod terrain LOD;
//! `clouds` = body-fixed volumetric and temporal cloud render mechanism.
pub mod clouds;
pub mod composite_order;
pub mod craft;
pub mod ground;
pub mod impostor;
pub mod tiles;

pub use clouds::*;
pub use craft::*;
pub use ground::*;
pub use impostor::*;
/// The shared body-surface shading types + WGSL libraries, extracted to the
/// `thalos_body_shading` leaf crate (ADR-20260724T022732Z) so shading edits
/// recompile a small crate. Re-exported as the `shading` module and flat at the
/// crate root so existing `thalos_body_render::{shading::*, SceneLighting, …}`
/// and internal `crate::shading::…` paths keep resolving unchanged.
pub use thalos_body_shading::{self as shading, *};

/// The vendored UDLOD terrain renderer, re-exported so the rest of the
/// workspace depends on it *through* `body_render` — its single consumer —
/// rather than importing `thalos_udlod` directly. Reach tile/atlas/precision
/// handles via `thalos_body_render::udlod::{prelude, math, big_space}`.
pub use thalos_udlod as udlod;

use bevy::prelude::*;

/// Adds the full body-render stack: shared shader libraries, impostor
/// materials, the standard-path tile ground renderer, and the legacy udlod
/// ground-terrain pipeline it is replacing.
pub struct BodyRenderPlugin;
impl Plugin for BodyRenderPlugin {
    fn build(&self, app: &mut App) {
        // Guarded so this composes with `craft::CraftRenderPlugin` (added
        // separately via `ShipyardPlugin`), which also adds `PlanetLightingPlugin`
        // defensively — whichever builds first adds it, the other skips. An
        // unconditional add here would double-add (panic) under either order.
        if !app.is_plugin_added::<shading::PlanetLightingPlugin>() {
            app.add_plugins(shading::PlanetLightingPlugin);
        }
        app.add_plugins(impostor::PlanetRenderingPlugin);
        app.add_plugins(clouds::CloudsPlugin);
        // Legacy udlod ground (still the owner of the analytic BodySky/ocean
        // projections, which are NOT legacy — see `ground`'s module docs).
        app.add_plugins(ground::ThalosTerrainPlugin);
        // NTR-X1: the standard-path tile renderer — the default ground.
        // Inert until the game inserts a `tiles::TileTerrainRoot`.
        app.add_plugins(tiles::TileTerrainPlugin);
    }
}
