//! Compatibility shim re-exporting the parts of `big_space` we use.
//!
//! `big_space` 0.12 dropped the precision generic in favor of feature flags
//! (`i8`/`i16`/`i32`/`i64`/`i128`). The consumer selects which feature to
//! enable; with none enabled, `big_space` defaults to `i64`. The old
//! `ReferenceFrame` type was renamed to [`big_space::grid::Grid`],
//! `GridCell<P>` to [`big_space::grid::cell::CellCoord`], and the
//! `GridTransform*` query types to `CellTransform*`.
pub use big_space::commands::BigSpaceCommands;
pub use big_space::floating_origins::FloatingOrigin;
pub use big_space::grid::cell::CellCoord as GridCell;
pub use big_space::grid::local_origin::Grids as ReferenceFrames;
pub use big_space::grid::Grid as ReferenceFrame;
pub use big_space::plugin::BigSpaceDefaultPlugins as BigSpacePlugin;
pub use big_space::world_query::{
    CellTransform as GridTransform, CellTransformItem as GridTransformItem,
    CellTransformOwned as GridTransformOwned, CellTransformReadOnly as GridTransformReadOnly,
};

/// The integer precision used for grid cell coordinates. Selected by the
/// consumer via `big_space`'s `i8`/`i16`/`i32`/`i64`/`i128` feature flags;
/// defaults to `i64` if none is enabled.
pub type GridPrecision = big_space::prelude::GridPrecision;
