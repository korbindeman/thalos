//! Compatibility shim re-exporting the parts of `big_space` we use.
//!
//! `big_space` 0.12 dropped the precision generic in favor of feature flags
//! (`i8`/`i16`/`i32`/`i64`/`i128`). The consumer selects which feature to
//! enable; with none enabled, `big_space` defaults to `i64`. The old
//! `ReferenceFrame` type was renamed to [`big_space::grid::Grid`],
//! `GridCell<P>` to [`big_space::grid::cell::CellCoord`], and the
//! `GridTransform*` query types to `CellTransform*`.
use bevy::math::DQuat;
use bevy::prelude::Component;

pub use big_space::commands::BigSpaceCommands;
pub use big_space::floating_origins::FloatingOrigin;
pub use big_space::grid::Grid as ReferenceFrame;
pub use big_space::grid::cell::CellCoord as GridCell;
pub use big_space::grid::local_origin::Grids as ReferenceFrames;
pub use big_space::plugin::BigSpaceDefaultPlugins as BigSpacePlugin;
pub use big_space::world_query::{
    CellTransform as GridTransform, CellTransformItem as GridTransformItem,
    CellTransformOwned as GridTransformOwned, CellTransformReadOnly as GridTransformReadOnly,
};

/// The integer precision used for grid cell coordinates. Selected by the
/// consumer via `big_space`'s `i8`/`i16`/`i32`/`i64`/`i128` feature flags;
/// defaults to `i64` if none is enabled.
pub type GridPrecision = big_space::prelude::GridPrecision;

/// Optional f64 override for a terrain parent grid's body-fixed → world
/// rotation.
///
/// `big_space` stores every grid `Transform.rotation` in f32. At planetary
/// scale (radii ~10⁶ m) the f32 quaternion ULP, applied to the camera→body
/// vector, is a decimetre of positional error — and it *flickers*
/// frame-to-frame as a spinning body's rotation requantizes. The
/// high-precision Taylor vertex path ([`TileTree::compute_requests`] →
/// [`TerrainModelApproximation::compute`]) would otherwise inherit that
/// flicker as visible near-field jitter at the viewer's feet.
///
/// A consumer that holds the grid's rotation in f64 can place this component
/// on the terrain's parent grid entity; `compute_requests` then uses it
/// instead of the f32 `Transform.rotation`. Keep it in sync with that
/// `Transform.rotation` — write both from the same f64 source in the same
/// system — so the high- and low-precision vertex paths don't slip at the
/// LOD swap. Absent → the f32 transform rotation is used.
///
/// [`TileTree::compute_requests`]: crate::terrain_data::tile_tree::TileTree::compute_requests
/// [`TerrainModelApproximation::compute`]: crate::math::TerrainModelApproximation::compute
#[derive(Component, Debug, Clone, Copy)]
pub struct PreciseRotation(pub DQuat);
